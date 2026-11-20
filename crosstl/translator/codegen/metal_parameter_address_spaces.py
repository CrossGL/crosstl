"""Parameter address spaces in the Metal generator: global parameter bindings, address spaces and memory qualifiers, readonly parameter, raw buffer and mesh payload diagnostics, pointer arithmetic and member lookups, assignment targets and mesh payload parameters."""

import re

from ..ast import (
    ArrayAccessNode,
    BinaryOpNode,
    IdentifierNode,
    MemberAccessNode,
    PointerAccessNode,
    PointerReinterpretNode,
    PointerType,
    ReferenceType,
    TernaryOpNode,
    UnaryOpNode,
)
from ..refusals import MetalRefusal
from ..vulkan_memory_scope import metal_coherence_scope
from .array_utils import format_c_style_array_declaration
from .metal_buffer_coherence import (
    coherent_array_parameter_declaration,
    merge_memory_contracts,
)
from .metal_restrict import metal_restrict_suffix
from .metal_texture_coherence import (
    is_storage_texture_type_name,
    with_device_texture_coherence,
)


class MetalParameterAddressSpaceMixin:
    """Parameter address space members of ``MetalCodeGen``."""

    def reserve_global_parameter_resource_bindings(self, used_bindings, func_name=None):
        dependencies = (
            self.function_global_resource_dependencies.get(func_name, set())
            if func_name
            else None
        )
        for cbuffer in self.cbuffer_variables:
            self.reserve_resource_binding_range(
                used_bindings,
                "Metal",
                "buffer",
                self.cbuffer_binding_indices.get(id(cbuffer), 0),
                1,
                getattr(cbuffer, "name", "<anonymous>"),
            )
        for texture_variable, binding, _, _ in self.texture_variables:
            self.reserve_resource_binding_range(
                used_bindings,
                "Metal",
                "texture",
                binding,
                self.global_resource_shape(texture_variable)[1],
                getattr(texture_variable, "name", "<anonymous>"),
            )
        for (
            acceleration_structure_variable,
            binding,
            _,
            _,
        ) in self.acceleration_structure_variables:
            self.reserve_resource_binding_range(
                used_bindings,
                "Metal",
                "buffer",
                binding,
                self.global_resource_shape(acceleration_structure_variable)[1],
                getattr(acceleration_structure_variable, "name", "<anonymous>"),
            )
        for (
            visible_function_table_variable,
            binding,
            _,
            _,
        ) in self.visible_function_table_variables:
            self.reserve_resource_binding_range(
                used_bindings,
                "Metal",
                "buffer",
                binding,
                self.global_resource_shape(visible_function_table_variable)[1],
                getattr(visible_function_table_variable, "name", "<anonymous>"),
            )
        for (
            intersection_function_table_variable,
            binding,
            _,
            _,
        ) in self.intersection_function_table_variables:
            self.reserve_resource_binding_range(
                used_bindings,
                "Metal",
                "buffer",
                binding,
                self.global_resource_shape(intersection_function_table_variable)[1],
                getattr(intersection_function_table_variable, "name", "<anonymous>"),
            )
        for buffer_variable, binding, _, _ in self.structured_buffer_variables:
            self.reserve_resource_binding_range(
                used_bindings,
                "Metal",
                "buffer",
                binding,
                self.global_resource_shape(buffer_variable)[1],
                getattr(buffer_variable, "name", "<anonymous>"),
            )
        for (
            buffer_variable,
            binding,
            _,
            array_size,
            _,
        ) in self.metal_buffer_resource_variables:
            self.reserve_resource_binding_range(
                used_bindings,
                "Metal",
                "buffer",
                binding,
                self.metal_buffer_resource_binding_count(buffer_variable, array_size),
                getattr(buffer_variable, "name", "<anonymous>"),
            )
        for buffer_variable, binding, _, _ in self.structured_buffer_length_variables:
            self.reserve_resource_binding_range(
                used_bindings,
                "Metal",
                "buffer",
                binding,
                self.global_resource_shape(buffer_variable)[1],
                self.structured_buffer_length_resource_name(
                    getattr(buffer_variable, "name", "<anonymous>")
                ),
            )
        for buffer_variable, binding, _, _ in self.structured_buffer_counter_variables:
            self.reserve_resource_binding_range(
                used_bindings,
                "Metal",
                "buffer",
                binding,
                self.global_resource_shape(buffer_variable)[1],
                self.structured_buffer_counter_resource_name(
                    getattr(buffer_variable, "name", "<anonymous>")
                ),
            )
        for buffer_variable, binding, _, _ in self.glsl_buffer_block_variables:
            self.reserve_resource_binding_range(
                used_bindings,
                "Metal",
                "buffer",
                binding,
                self.global_resource_shape(buffer_variable)[1],
                getattr(buffer_variable, "name", "<anonymous>"),
            )
        for sampler_variable, binding, array_size in self.sampler_variables:
            sampler_name = getattr(sampler_variable, "name", None)
            if dependencies is not None and sampler_name not in dependencies:
                continue
            self.reserve_resource_binding_range(
                used_bindings,
                "Metal",
                "sampler",
                binding,
                self.resource_array_count(array_size),
                sampler_name or "<anonymous>",
            )

    def resource_parameter_attribute(self, raw_param_type, node=None):
        if node is None:
            return ""
        if self.is_sampler_type(raw_param_type):
            binding = self.explicit_resource_binding_index(
                node, {"binding", "sampler"}, ("s",)
            )
            return f" [[sampler({binding})]]" if binding is not None else ""
        if self.is_visible_function_table_type(raw_param_type):
            binding = self.explicit_resource_binding_index(
                node, {"binding", "buffer"}, ("b", "u", "t")
            )
            return f" [[buffer({binding})]]" if binding is not None else ""
        if self.is_intersection_function_table_type(raw_param_type):
            binding = self.explicit_resource_binding_index(
                node, {"binding", "buffer"}, ("b", "u", "t")
            )
            return f" [[buffer({binding})]]" if binding is not None else ""
        if self.is_acceleration_structure_type(raw_param_type):
            binding = self.explicit_resource_binding_index(
                node, {"binding", "buffer"}, ("b", "u", "t")
            )
            return f" [[buffer({binding})]]" if binding is not None else ""
        if self.is_structured_buffer_type(raw_param_type):
            binding = self.explicit_resource_binding_index(
                node, {"binding", "buffer"}, ("b", "u", "t")
            )
            return f" [[buffer({binding})]]" if binding is not None else ""
        if self.is_resource_parameter_type(raw_param_type):
            binding = self.explicit_resource_binding_index(
                node, {"binding", "texture"}, ("t", "u")
            )
            return f" [[texture({binding})]]" if binding is not None else ""
        if self.is_bound_scalar_value_parameter(raw_param_type, node, "compute"):
            binding = self.explicit_resource_binding_index(
                node, {"binding", "buffer"}, ("b", "u", "t")
            )
            return f" [[buffer({binding})]]" if binding is not None else ""
        if self.is_raw_buffer_parameter_type(raw_param_type, node):
            binding = self.explicit_resource_binding_index(
                node, {"binding", "buffer"}, ("b", "u", "t")
            )
            return f" [[buffer({binding})]]" if binding is not None else ""
        return ""

    def is_metal_storage_texture_node(self, node):
        raw_type = getattr(node, "param_type", getattr(node, "var_type", None))
        return is_storage_texture_type_name(self.resource_base_type(raw_type))

    def metal_coherent_texture_type(self, texture_type, function_name, name):
        """Add MSL 3.2 device coherence to a texture type that needs it (see metal_texture_coherence)."""
        if (function_name, name) not in self.metal_device_coherent_textures:
            return texture_type
        return with_device_texture_coherence(str(texture_type))

    def validate_metal_ray_resource_parameter_value_type(self, raw_param_type, name):
        resource_kind = self.metal_ray_resource_pointer_or_reference_kind(
            raw_param_type
        )
        if resource_kind is None:
            return

        type_name = self.type_name_string(raw_param_type) or str(raw_param_type)
        raise MetalRefusal(
            f"Metal ray tracing resource parameter '{name}' has pointer or "
            f"reference type '{type_name}' for {resource_kind}; Metal ray "
            "tracing resources must be passed by value without address-space "
            "qualifiers"
        )

    def metal_ray_resource_pointer_or_reference_kind(self, raw_param_type):
        if isinstance(raw_param_type, PointerType):
            base_type = raw_param_type.pointee_type
        elif isinstance(raw_param_type, ReferenceType):
            base_type = raw_param_type.referenced_type
        else:
            type_name = self.type_name_string(raw_param_type)
            if not type_name:
                return None
            type_name = str(type_name).strip()
            if not (type_name.endswith("*") or type_name.endswith("&")):
                return None
            base_type = type_name[:-1].strip()

        if self.is_acceleration_structure_type(base_type):
            return "acceleration_structure"
        if self.is_visible_function_table_type(base_type):
            return "visible_function_table"
        if self.is_intersection_function_table_type(base_type):
            return "intersection_function_table"
        return None

    def format_address_space_parameter_declaration(
        self, raw_param_type, mapped_type, name, node=None, shader_type=None
    ):
        if isinstance(raw_param_type, PointerType):
            address_space = self.effective_parameter_address_space(
                raw_param_type, node, shader_type
            )
            address_space = self.readonly_qualified_address_space(address_space, node)
            pointee_type = self.metal_native_narrow_bitcast_storage_type(
                raw_param_type.pointee_type
            ) or self.map_resource_type_with_format(raw_param_type.pointee_type, node)
            layout_storage = getattr(self, "metal_hlsl_pointer_parameters", {}).get(
                id(node)
            )
            if layout_storage is not None:
                pointee_type = layout_storage[0]
            memory_qualifiers = self.metal_parameter_memory_qualifier_prefix(
                node, raw_param_type, address_space
            )
            reference = (
                " thread&"
                if shader_type is None
                and self.parameter_qualifier_names(node) & {"out", "inout"}
                else ""
            )
            restrict = "" if reference else metal_restrict_suffix(node, raw_param_type)
            return (
                f"{memory_qualifiers}{address_space} {pointee_type}*{restrict}"
                f"{reference} {name}"
            )

        if isinstance(raw_param_type, ReferenceType):
            address_space = self.effective_parameter_address_space(
                raw_param_type, node, shader_type
            )
            address_space = self.readonly_qualified_address_space(address_space, node)
            referenced_type = self.metal_native_narrow_bitcast_storage_type(
                raw_param_type.referenced_type
            ) or self.map_resource_type_with_format(
                raw_param_type.referenced_type, node
            )
            memory_qualifiers = self.metal_parameter_memory_qualifier_prefix(
                node, raw_param_type, address_space
            )
            return f"{memory_qualifiers}{address_space} {referenced_type}& {name}"

        if self.is_array_type_node(raw_param_type):
            if self.resource_array_parameter(raw_param_type, node) is not None:
                return None
            binding = self.explicit_buffer_binding_index(node)
            qualifiers = self.parameter_qualifier_names(node)
            address_space = self.effective_parameter_address_space(
                raw_param_type,
                node,
                shader_type,
                default_for_stage_binding=binding is not None
                or bool(qualifiers & {"const", "readonly"}),
            )
            if address_space is None:
                return None
            address_space = self.readonly_qualified_address_space(address_space, node)
            if binding is not None and shader_type in {
                "vertex",
                "fragment",
                "compute",
                "ray_generation",
            }:
                element_type = self.map_resource_type_with_format(
                    raw_param_type.element_type, node
                )
                layout_storage = getattr(self, "metal_hlsl_pointer_parameters", {}).get(
                    id(node)
                )
                if layout_storage is not None:
                    element_type = layout_storage[0]
                memory_qualifiers = self.metal_parameter_memory_qualifier_prefix(
                    node, raw_param_type, address_space
                )
                return f"{memory_qualifiers}{address_space} {element_type}* {name}"
            memory_qualifiers = self.metal_parameter_memory_qualifier_prefix(
                node, raw_param_type, address_space
            )
            if "coherent" in memory_qualifiers:
                declaration = coherent_array_parameter_declaration(mapped_type, name)
            else:
                declaration = format_c_style_array_declaration(mapped_type, name)
            return f"{memory_qualifiers}{address_space} {declaration}"

        qualifiers = self.parameter_qualifier_names(node)
        address_space = self.effective_parameter_address_space(
            raw_param_type,
            node,
            shader_type,
            default_for_stage_binding=bool(qualifiers & {"out", "inout"}),
        )
        if address_space is not None:
            address_space = self.readonly_qualified_address_space(
                address_space,
                node,
            )
            memory_qualifiers = self.metal_parameter_memory_qualifier_prefix(
                node, raw_param_type, address_space
            )
            return f"{memory_qualifiers}{address_space} {mapped_type}& {name}"

        return None

    def effective_parameter_address_space(
        self,
        raw_param_type,
        node=None,
        shader_type=None,
        default_for_stage_binding=True,
    ):
        default_space = None
        if default_for_stage_binding or isinstance(
            raw_param_type, (PointerType, ReferenceType)
        ):
            default_space = self.default_parameter_address_space(
                raw_param_type, node, shader_type
            )
        return self.parameter_address_space(node, default=default_space)

    def default_parameter_address_space(
        self, raw_param_type, node=None, shader_type=None
    ):
        if shader_type in {"vertex", "fragment", "compute", "ray_generation"}:
            binding = self.explicit_buffer_binding_index(node)
            if binding is not None:
                if isinstance(raw_param_type, ReferenceType):
                    return "constant"
                return "device"
        return "thread"

    def explicit_buffer_binding_index(self, node):
        return self.explicit_resource_binding_index(
            node, {"binding", "buffer"}, ("b", "u", "t")
        )

    def parameter_address_space(self, node=None, default=None):
        qualifiers = self.parameter_qualifier_names(node)
        address_spaces = []
        if "constant" in qualifiers:
            address_spaces.append("constant")
        if qualifiers & {"object_data", "objectdata"}:
            address_spaces.append("object_data")
        if qualifiers & {"ray_data", "raydata"}:
            address_spaces.append("ray_data")
        if qualifiers & {"device", "global", "storage"} or "buffer" in (
            getattr(node, "qualifiers", []) or []
        ):
            address_spaces.append("device")
        if "threadgroup_imageblock" in qualifiers:
            address_spaces.append("threadgroup_imageblock")
        if qualifiers & {"threadgroup", "workgroup", "shared", "groupshared"}:
            address_spaces.append("threadgroup")
        if qualifiers & {"thread", "function", "local", "private"}:
            address_spaces.append("thread")

        address_spaces = list(dict.fromkeys(address_spaces))
        if len(address_spaces) > 1:
            name = getattr(node, "name", "<anonymous>")
            raise MetalRefusal(
                f"Metal parameter '{name}' has conflicting address-space qualifiers: "
                f"{', '.join(sorted(qualifiers))}"
            )
        if address_spaces:
            return address_spaces[0]
        return default

    def parameter_qualifier_names(self, node=None):
        qualifiers = {
            str(qualifier).lower()
            for qualifier in getattr(node, "qualifiers", []) or []
        }
        qualifiers.update(
            str(qualifier).lower()
            for qualifier in getattr(node, "resource_qualifiers", []) or []
        )
        qualifiers.update(
            str(getattr(attribute, "name", "")).lower()
            for attribute in getattr(node, "attributes", []) or []
        )
        raw_type = getattr(node, "param_type", getattr(node, "var_type", None))
        contract_type = raw_type
        while self.is_array_type_node(contract_type):
            contract_type = contract_type.element_type
        if isinstance(contract_type, (PointerType, ReferenceType)):
            qualifiers.add(str(getattr(contract_type, "address_space", "")).lower())
            qualifiers.add(str(getattr(contract_type, "access_mode", "")).lower())
        qualifier_aliases = {
            "read": "readonly",
            "write": "writeonly",
            "read_write": "readwrite",
            "access::read": "readonly",
            "access::write": "writeonly",
            "access::read_write": "readwrite",
        }
        qualifiers.update(
            alias
            for qualifier, alias in qualifier_aliases.items()
            if qualifier in qualifiers
        )
        qualifiers.discard("")
        return qualifiers

    def resource_memory_qualifier_contracts(self, node=None, raw_type=None):
        """Return ordered ``(kind, scope)`` resource-memory contracts."""
        values = []
        contract_type = raw_type
        while self.is_array_type_node(contract_type):
            contract_type = contract_type.element_type
        if isinstance(contract_type, (PointerType, ReferenceType)):
            values.extend(getattr(contract_type, "resource_qualifiers", []) or [])
        values.extend(getattr(node, "resource_qualifiers", []) or [])
        values.extend(getattr(node, "qualifiers", []) or [])

        contracts = []
        for value in values:
            kind = getattr(value, "kind", None)
            scope = getattr(value, "scope", None)
            text = str(value).lower()
            if kind is None and text == "globallycoherent":
                # HLSL's device-wide UAV coherence (Slang and the HLSL importer keep the HLSL name).
                kind, scope = "coherent", "device"
            if kind is None:
                match = re.fullmatch(
                    r"(?P<kind>volatile|coherent)(?:\((?P<scope>[a-z_][a-z0-9_]*)\))?",
                    text,
                )
                if match is None:
                    continue
                kind = match.group("kind")
                scope = match.group("scope")
            else:
                kind = str(kind).lower()
                scope = str(scope).lower() if scope is not None else None
            if kind not in {"volatile", "coherent"}:
                continue
            contract = (kind, scope)
            if contract not in contracts:
                contracts.append(contract)

        for attribute in getattr(node, "attributes", []) or []:
            kind = str(getattr(attribute, "name", "")).lower()
            if kind == "globallycoherent":
                if ("coherent", "device") not in contracts:
                    contracts.append(("coherent", "device"))
                continue
            if kind not in {"volatile", "coherent"}:
                continue
            arguments = getattr(attribute, "arguments", []) or []
            scope = self.attribute_value_to_string(arguments[0]) if arguments else None
            contract = (kind, str(scope).lower() if scope is not None else None)
            if contract not in contracts:
                contracts.append(contract)
        scope = metal_coherence_scope(node)
        if scope is not None and not any(kind == "coherent" for kind, _ in contracts):
            contracts.append(("coherent", scope))
        return contracts

    def resource_memory_qualifier_prefix(self, node=None, raw_type=None):
        return self.render_resource_memory_contracts(
            self.resource_memory_qualifier_contracts(node, raw_type), node
        )

    def metal_parameter_memory_qualifier_prefix(self, node, raw_type, address_space):
        """Memory qualifiers of a helper parameter, widened by its callers' arguments.

        See metal_buffer_coherence: MSL calls cannot drop coherence or ``volatile`` from a pointer.
        """
        own = self.resource_memory_qualifier_contracts(node, raw_type)
        inherited = self.metal_inherited_buffer_contracts.get(
            (self.current_function_name, getattr(node, "name", None))
        )
        if inherited and "device" in str(address_space or "").split():
            own = merge_memory_contracts(own, inherited)
        return self.render_resource_memory_contracts(own, node)

    def render_resource_memory_contracts(self, contracts, node=None):
        rendered = []
        for kind, scope in contracts:
            if kind == "coherent" and scope is not None:
                if scope not in {"threadgroup", "device", "system"}:
                    name = getattr(node, "name", "<anonymous>")
                    raise MetalRefusal(
                        f"Metal resource '{name}' has unsupported coherence scope "
                        f"'{scope}'"
                    )
                rendered.append(f"coherent({scope})")
            else:
                rendered.append(kind)
        return f"{' '.join(rendered)} " if rendered else ""

    def normalized_address_space(self, address_space):
        if address_space is None:
            return None
        address_space = str(address_space).strip()
        if address_space.startswith("const "):
            address_space = address_space[len("const ") :].strip()
        return address_space or None

    def readonly_qualified_address_space(self, address_space, node=None):
        qualifiers = self.parameter_qualifier_names(node)
        if address_space == "device" and qualifiers & {"in", "readonly"}:
            return "const device"
        if qualifiers & {"const", "in"} and address_space not in {None, "constant"}:
            return f"const {address_space}"
        return address_space

    def parameter_variable_address_space(
        self, raw_param_type, node=None, shader_type=None
    ):
        if self.is_structured_buffer_type(raw_param_type):
            return self.normalized_address_space(
                self.structured_buffer_address_space(raw_param_type, node)
            )
        if (
            self.is_array_type_node(raw_param_type)
            and self.resource_array_parameter(raw_param_type, node) is not None
        ):
            return None
        if not (
            isinstance(raw_param_type, (PointerType, ReferenceType))
            or self.is_array_type_node(raw_param_type)
        ):
            return None
        return self.normalized_address_space(
            self.effective_parameter_address_space(
                raw_param_type,
                node,
                shader_type,
                default_for_stage_binding=self.explicit_buffer_binding_index(node)
                is not None,
            )
        )

    def readonly_metal_parameter_reason(
        self, raw_param_type, node=None, shader_type=None
    ):
        if self.is_metal_mesh_payload_parameter(shader_type, node):
            return None
        if not (
            isinstance(raw_param_type, (PointerType, ReferenceType))
            or self.is_array_type_node(raw_param_type)
        ):
            return None
        qualifiers = self.parameter_qualifier_names(node)
        if self.array_indirect_element_type_node(raw_param_type) is not None:
            qualifiers -= {"const", "constant"}
        if self.is_array_type_node(raw_param_type):
            resource_type = self.resource_base_type(raw_param_type)
            if self.structured_buffer_type_name(resource_type) == "StructuredBuffer":
                return "read-only resource"
        if "const" in qualifiers:
            return "const-qualified"
        if "in" in qualifiers:
            return "input-only"
        if "constant" in qualifiers:
            return "constant address space"
        if "readonly" in qualifiers and not self.is_raw_buffer_parameter_type(
            raw_param_type, node
        ):
            return "readonly"
        return None

    def is_mutable_metal_parameter(self, raw_param_type, node=None):
        if not (
            isinstance(raw_param_type, (PointerType, ReferenceType))
            or self.is_array_type_node(raw_param_type)
        ):
            return False
        qualifiers = self.parameter_qualifier_names(node)
        if self.array_indirect_element_type_node(raw_param_type) is not None:
            qualifiers -= {"const", "constant"}
        if qualifiers & {
            "const",
            "constant",
            "in",
            "readonly",
        }:
            return False
        return self.readonly_metal_parameter_reason(raw_param_type, node) is None

    def is_readonly_raw_buffer_parameter(
        self, raw_param_type, node=None, shader_type=None
    ):
        if not (
            isinstance(raw_param_type, (PointerType, ReferenceType))
            or self.is_array_type_node(raw_param_type)
        ):
            return False
        qualifiers = self.parameter_qualifier_names(node)
        if qualifiers & {"in", "readonly"}:
            return True
        address_space = self.effective_parameter_address_space(
            raw_param_type,
            node,
            shader_type,
            default_for_stage_binding=self.explicit_buffer_binding_index(node)
            is not None,
        )
        return address_space == "constant"

    def is_array_type_node(self, vtype):
        return (
            hasattr(vtype, "element_type") and str(type(vtype)).find("ArrayType") != -1
        )

    def is_raw_buffer_parameter_type(self, raw_param_type, node=None):
        if isinstance(
            raw_param_type, (PointerType, ReferenceType)
        ) or self.is_array_type_node(raw_param_type):
            address_space = self.parameter_address_space(node)
            return address_space in {"device", "constant"} or (
                address_space is None
                and self.explicit_buffer_binding_index(node) is not None
            )
        return False

    def is_mutable_raw_buffer_parameter(self, raw_param_type, node=None):
        if not self.is_raw_buffer_parameter_type(raw_param_type, node):
            return False
        return not self.is_readonly_raw_buffer_parameter(raw_param_type, node)

    def metal_user_function_parameter_nodes_for_call(self, func_name, call_args):
        overloads = self.function_overloads_by_name.get(func_name, [])
        if overloads:
            selected = self.metal_user_function_overload_for_call(func_name, call_args)
            if selected is None:
                return None
            return list(
                getattr(
                    selected,
                    "parameters",
                    getattr(selected, "params", []),
                )
                or []
            )
        return list(self.function_parameter_nodes.get(func_name, []) or [])

    def readonly_raw_buffer_call_diagnostic(self, func_name, call_args):
        if func_name not in self.user_function_names:
            return None
        parameter_nodes = self.metal_user_function_parameter_nodes_for_call(
            func_name,
            call_args,
        )
        if parameter_nodes is None:
            return None
        for index, arg in enumerate(call_args):
            if index >= len(parameter_nodes):
                continue
            arg_name = self.assignment_target_root_name(arg)
            if arg_name not in self.current_readonly_raw_buffer_parameters:
                continue
            parameter = parameter_nodes[index]
            raw_param_type = getattr(
                parameter, "param_type", getattr(parameter, "vtype", None)
            )
            if not self.is_mutable_raw_buffer_parameter(raw_param_type, parameter):
                continue
            parameter_name = getattr(parameter, "name", f"arg{index}")
            return self.refuse_lowering(
                f"raw buffer call: readonly buffer '{arg_name}' cannot be passed to "
                f"mutable parameter '{parameter_name}' of '{func_name}'",
                "(void)0",
            )
        return None

    def readonly_metal_parameter_call_diagnostic(self, func_name, call_args):
        if func_name not in self.user_function_names:
            return None
        parameter_nodes = self.metal_user_function_parameter_nodes_for_call(
            func_name,
            call_args,
        )
        if parameter_nodes is None:
            return None
        for index, arg in enumerate(call_args):
            if index >= len(parameter_nodes):
                continue
            arg_name = self.assignment_target_root_name(arg)
            if arg_name not in self.current_readonly_metal_parameters:
                continue
            if self.mutable_metal_pointer_member_argument(arg):
                continue
            if self.metal_array_arguments.copies(func_name, index, arg, call_args):
                # A by-value array parameter receives a thread copy of the read-only array.
                continue
            reason = self.current_readonly_metal_parameter_reasons.get(
                arg_name, "readonly"
            )
            if (
                isinstance(arg, UnaryOpNode)
                and getattr(arg, "operator", None) == "&"
                and self.mutable_const_pointee_pointer_assignment(
                    getattr(arg, "operand", None), arg_name, reason
                )
            ):
                # ``const device T* row`` keeps a mutable thread-local pointer
                # slot even though its pointee is const.  Passing ``&row`` to a
                # mutable array/pointer parameter may rebind that slot; it does
                # not grant write access through the device pointer.
                continue
            parameter = parameter_nodes[index]
            raw_param_type = getattr(
                parameter, "param_type", getattr(parameter, "vtype", None)
            )
            if not self.is_mutable_metal_parameter(raw_param_type, parameter):
                continue
            parameter_name = getattr(parameter, "name", f"arg{index}")
            return self.refuse_lowering(
                f"parameter call: readonly parameter '{arg_name}' cannot be passed to "
                f"mutable parameter '{parameter_name}' of '{func_name}'",
                "(void)0",
            )
        return None

    def mutable_metal_pointer_member_argument(self, expression):
        if isinstance(expression, UnaryOpNode) and expression.operator == "&":
            operand = expression.operand
            if not isinstance(operand, ArrayAccessNode) and not (
                isinstance(operand, UnaryOpNode) and operand.operator == "*"
            ):
                # Taking the pointer slot's address still observes owner constness.
                return False
        member = self.metal_storage_pointer_reinterpret_member_node(expression)
        if member is None:
            return False
        raw_type = getattr(member, "member_type", None)
        return isinstance(raw_type, PointerType) and self.is_mutable_metal_parameter(
            raw_type, member
        )

    def readonly_metal_mesh_payload_call_diagnostic(self, func_name, call_args):
        if func_name not in self.user_function_names:
            return None
        parameter_nodes = self.metal_user_function_parameter_nodes_for_call(
            func_name,
            call_args,
        )
        if parameter_nodes is None:
            return None
        for index, arg in enumerate(call_args):
            if index >= len(parameter_nodes):
                continue
            arg_name = self.readonly_metal_mesh_payload_key(arg)
            if arg_name is None:
                continue
            parameter = parameter_nodes[index]
            raw_param_type = getattr(
                parameter, "param_type", getattr(parameter, "vtype", None)
            )
            expected_address_space = self.parameter_variable_address_space(
                raw_param_type, parameter
            )
            if expected_address_space != "object_data":
                continue
            if not self.is_mutable_metal_parameter(raw_param_type, parameter):
                continue
            parameter_name = getattr(parameter, "name", f"arg{index}")
            reason = self.current_readonly_metal_mesh_payload_reasons.get(
                arg_name, "const object_data"
            )
            reason_text = self.readonly_metal_mesh_payload_reason_text(reason)
            self.refuse_lowering(
                f"mesh payload call: mesh payload '{arg_name}' is {reason_text} and "
                f"cannot be passed to mutable parameter '{parameter_name}' of "
                f"'{func_name}'"
            )
            selected_function = self.metal_user_function_overload_for_call(
                func_name,
                call_args,
            )
            return_type = (
                self.metal_effective_function_return_type(selected_function)
                if selected_function is not None
                else self.function_return_types.get(func_name)
            )
            if self.map_type(return_type) == "void":
                return "(void)0"
            return self.diagnostic_zero_value_for_type(return_type)
        return None

    def pointer_arithmetic_operand(self, expr, operand_types=None):
        if not isinstance(expr, BinaryOpNode) or expr.operator not in {"+", "-"}:
            return None
        if operand_types is None:
            operand_types = (
                self.expression_result_type(expr.left),
                self.expression_result_type(expr.right),
            )
        left_pointer, right_pointer = map(
            self.metal_type_is_pointer_like, operand_types
        )
        if left_pointer and not right_pointer:
            return expr.left
        if right_pointer and not left_pointer and expr.operator == "+":
            return expr.right
        return None

    def argument_address_space(self, arg):
        pointer_operand = self.pointer_arithmetic_operand(arg)
        if pointer_operand is not None:
            # Offset values do not contribute storage to the resulting pointer.
            return self.argument_address_space(pointer_operand)
        if isinstance(arg, UnaryOpNode) and arg.operator in {"&", "*"}:
            return self.argument_address_space(arg.operand)
        if isinstance(arg, ArrayAccessNode):
            return self.argument_address_space(arg.array)
        if isinstance(arg, PointerReinterpretNode):
            # A nested view names its address space; lowering that view proves
            # it against the view's own source before this result is used.
            return getattr(getattr(arg, "target_type", None), "address_space", None)
        if isinstance(arg, TernaryOpNode):
            if self.argument_address_space_conflict(arg) is not None:
                return None
            true_space = self.argument_address_space(getattr(arg, "true_expr", None))
            false_space = self.argument_address_space(getattr(arg, "false_expr", None))
            if (
                true_space is not None
                and false_space is not None
                and true_space == false_space
            ):
                return true_space
            return None
        member_address_space = self.address_space_qualified_member_address_space(arg)
        if member_address_space is not None:
            return member_address_space
        arg_name = self.assignment_target_root_name(arg)
        if arg_name in self.current_address_space_variables:
            return self.current_address_space_variables[arg_name]
        if arg_name in self.local_variable_types:
            return "thread"
        return None

    def address_space_qualified_member_address_space(self, expr):
        if isinstance(expr, UnaryOpNode) and getattr(expr, "operator", None) == "&":
            return self.address_space_qualified_member_address_space(
                getattr(expr, "operand", None)
            )
        if isinstance(expr, BinaryOpNode) and getattr(expr, "operator", None) in {
            "+",
            "-",
        }:
            pointer_operand = self.pointer_arithmetic_operand(expr)
            return self.address_space_qualified_member_address_space(pointer_operand)
        if isinstance(expr, ArrayAccessNode):
            return self.address_space_qualified_member_address_space(
                getattr(expr, "array", getattr(expr, "array_expr", None))
            )
        if isinstance(expr, MemberAccessNode):
            object_expr = getattr(expr, "object", getattr(expr, "object_expr", None))
            object_type = self.expression_result_type(object_expr)
            object_type = self.member_lookup_type_name(object_type)
            member_space = self.struct_member_address_spaces.get(
                self.type_name_string(object_type), {}
            ).get(str(getattr(expr, "member", "")))
            return member_space or self.argument_address_space(object_expr)
        if isinstance(expr, PointerAccessNode):
            object_expr = getattr(expr, "pointer_expr", None)
            object_type = self.expression_result_type(object_expr)
            object_type = self.member_lookup_type_name(object_type)
            return self.struct_member_address_spaces.get(
                self.type_name_string(object_type), {}
            ).get(str(getattr(expr, "member", "")))
        return None

    def argument_address_space_conflict(self, arg):
        if arg is None:
            return None
        if isinstance(arg, TernaryOpNode):
            true_expr = getattr(arg, "true_expr", None)
            false_expr = getattr(arg, "false_expr", None)
            true_conflict = self.argument_address_space_conflict(true_expr)
            if true_conflict is not None:
                return true_conflict
            false_conflict = self.argument_address_space_conflict(false_expr)
            if false_conflict is not None:
                return false_conflict
            true_space = self.argument_address_space(true_expr)
            false_space = self.argument_address_space(false_expr)
            if (
                true_space is not None
                and false_space is not None
                and true_space != false_space
            ):
                return (
                    true_space,
                    false_space,
                    self.assignment_target_display_name(true_expr) or "<expr>",
                    self.assignment_target_display_name(false_expr) or "<expr>",
                )
            return None
        if isinstance(arg, UnaryOpNode) and getattr(arg, "operator", None) in {
            "&",
            "*",
        }:
            return self.argument_address_space_conflict(getattr(arg, "operand", None))
        if isinstance(arg, BinaryOpNode) and getattr(arg, "operator", None) in {
            "+",
            "-",
        }:
            return self.argument_address_space_conflict(
                self.pointer_arithmetic_operand(arg)
            )
        if isinstance(arg, ArrayAccessNode):
            return self.argument_address_space_conflict(
                getattr(arg, "array", getattr(arg, "array_expr", None))
            )
        return None

    def address_space_conflict_description(self, conflict):
        true_space, false_space, true_name, false_name = conflict
        return (
            f"branches '{true_name}' ({true_space}) and "
            f"'{false_name}' ({false_space})"
        )

    @staticmethod
    def metal_user_function_accepts_argument_count(function, argument_count):
        parameters = list(
            getattr(
                function,
                "parameters",
                getattr(function, "params", []),
            )
            or []
        )
        required_count = sum(
            getattr(parameter, "default_value", None) is None
            for parameter in parameters
        )
        return required_count <= argument_count <= len(parameters)

    def metal_user_function_overload_for_call(self, function_name, arguments):
        candidates = list(self.function_overloads_by_name.get(function_name, []))
        if not candidates:
            candidate = self.functions_by_name.get(function_name)
            candidates = [candidate] if candidate is not None else []
        argument_count = len(arguments or [])
        arity_matches = [
            candidate
            for candidate in candidates
            if self.metal_user_function_accepts_argument_count(
                candidate,
                argument_count,
            )
        ]
        if len(arity_matches) == 1:
            return arity_matches[0]
        return None

    def pointer_pointee_type_name(self, vtype):
        if isinstance(vtype, PointerType):
            return self.type_name_string(vtype.pointee_type)
        type_name = self.type_name_string(vtype)
        if not type_name:
            return None
        type_name = str(type_name).strip()
        return type_name[:-1].strip() if type_name.endswith("*") else None

    def reference_referent_type_name(self, vtype):
        if isinstance(vtype, ReferenceType):
            return self.type_name_string(vtype.referenced_type)
        type_name = self.type_name_string(vtype)
        if not type_name:
            return None
        type_name = str(type_name).strip()
        return type_name[:-1].strip() if type_name.endswith("&") else None

    def member_lookup_type_name(self, vtype):
        return (
            self.pointer_pointee_type_name(vtype)
            or self.reference_referent_type_name(vtype)
            or vtype
        )

    def member_access_uses_pointer_operator(self, expr):
        if not isinstance(expr, MemberAccessNode):
            return False
        object_expr = getattr(expr, "object", getattr(expr, "object_expr", None))
        object_type = self.expression_result_type(object_expr)
        return self.pointer_pointee_type_name(object_type) is not None

    def collect_struct_member_address_spaces(self, structs):
        member_address_spaces = {}
        for struct in structs or []:
            struct_name = getattr(struct, "name", None)
            members = getattr(struct, "members", None)
            if not struct_name or members is None:
                continue
            for member in members:
                member_name = getattr(member, "name", None)
                raw_member_type = getattr(member, "member_type", None)
                if not member_name or raw_member_type is None:
                    continue
                if not (
                    isinstance(raw_member_type, (PointerType, ReferenceType))
                    or self.is_array_type_node(raw_member_type)
                ):
                    continue
                address_space = self.parameter_variable_address_space(
                    raw_member_type, member
                )
                if address_space is not None:
                    member_address_spaces.setdefault(struct_name, {})[
                        member_name
                    ] = address_space
        return member_address_spaces

    def assignment_target_root_name(self, target):
        buffer_source = self.structured_buffer_load_source(target)
        if buffer_source is not None:
            return self.assignment_target_root_name(buffer_source)
        if isinstance(target, UnaryOpNode) and getattr(target, "operator", None) in {
            "&",
            "*",
        }:
            return self.assignment_target_root_name(getattr(target, "operand", None))
        if isinstance(target, MemberAccessNode):
            return self.assignment_target_root_name(
                getattr(target, "object", getattr(target, "object_expr", None))
            )
        if isinstance(target, PointerAccessNode):
            return self.assignment_target_root_name(
                getattr(target, "pointer_expr", None)
            )
        if isinstance(target, ArrayAccessNode):
            return self.assignment_target_root_name(
                getattr(target, "array", getattr(target, "array_expr", None))
            )
        if isinstance(target, BinaryOpNode) and getattr(target, "operator", None) in {
            "+",
            "-",
        }:
            return self.assignment_target_root_name(
                getattr(target, "left", None)
            ) or self.assignment_target_root_name(getattr(target, "right", None))
        return self.expression_name(target)

    def assignment_target_display_name(self, target):
        if isinstance(target, UnaryOpNode):
            operand_name = self.assignment_target_display_name(
                getattr(target, "operand", None)
            )
            if getattr(target, "operator", None) == "&":
                return operand_name
            if getattr(target, "operator", None) == "*":
                return f"*{operand_name}" if operand_name else None
        if isinstance(target, BinaryOpNode) and getattr(target, "operator", None) in {
            "+",
            "-",
        }:
            return self.assignment_target_display_name(
                getattr(target, "left", None)
            ) or self.assignment_target_display_name(getattr(target, "right", None))
        if isinstance(target, ArrayAccessNode):
            return self.assignment_target_display_name(
                getattr(target, "array", getattr(target, "array_expr", None))
            )
        if isinstance(target, MemberAccessNode):
            object_name = self.assignment_target_display_name(
                getattr(target, "object", getattr(target, "object_expr", None))
            )
            member_name = str(getattr(target, "member", ""))
            return f"{object_name}.{member_name}" if object_name else member_name
        if isinstance(target, PointerAccessNode):
            object_name = self.assignment_target_display_name(
                getattr(target, "pointer_expr", None)
            )
            member_name = str(getattr(target, "member", ""))
            return f"{object_name}->{member_name}" if object_name else member_name
        return self.expression_name(target)

    def assignment_target_access_display_name(self, target):
        if isinstance(target, UnaryOpNode):
            operand_name = self.assignment_target_access_display_name(
                getattr(target, "operand", None)
            )
            if getattr(target, "operator", None) == "&":
                return operand_name
            if getattr(target, "operator", None) == "*":
                return f"*{operand_name}" if operand_name else None
        if isinstance(target, BinaryOpNode) and getattr(target, "operator", None) in {
            "+",
            "-",
        }:
            return self.assignment_target_access_display_name(
                getattr(target, "left", None)
            ) or self.assignment_target_access_display_name(
                getattr(target, "right", None)
            )
        if isinstance(target, ArrayAccessNode):
            object_name = self.assignment_target_access_display_name(
                getattr(target, "array", getattr(target, "array_expr", None))
            )
            index_expr = getattr(target, "index", getattr(target, "index_expr", None))
            index_name = (
                self.safe_expression_to_string(index_expr)
                if index_expr is not None
                else ""
            )
            return f"{object_name}[{index_name}]" if object_name else None
        if isinstance(target, MemberAccessNode):
            object_name = self.assignment_target_access_display_name(
                getattr(target, "object", getattr(target, "object_expr", None))
            )
            member_name = str(getattr(target, "member", ""))
            return f"{object_name}.{member_name}" if object_name else member_name
        if isinstance(target, PointerAccessNode):
            object_name = self.assignment_target_access_display_name(
                getattr(target, "pointer_expr", None)
            )
            member_name = str(getattr(target, "member", ""))
            return f"{object_name}->{member_name}" if object_name else member_name
        return self.expression_name(target)

    def readonly_metal_mesh_payload_key(self, target):
        display_name = self.assignment_target_display_name(target)
        if display_name:
            candidates = [display_name]
            for separator in (".", "->", "["):
                if separator not in display_name:
                    continue
                parts = display_name.split(separator)
                prefix = parts[0]
                for part in parts[1:-1]:
                    prefix = f"{prefix}{separator}{part}"
                    candidates.append(prefix)
            for candidate in sorted(candidates, key=len, reverse=True):
                if candidate in self.current_readonly_metal_mesh_payload_parameters:
                    return candidate
            for candidate in self.current_readonly_metal_mesh_payload_parameters:
                if display_name.startswith(
                    (f"{candidate}.", f"{candidate}->", f"{candidate}[")
                ):
                    return candidate
        root_name = self.assignment_target_root_name(target)
        if root_name in self.current_readonly_metal_mesh_payload_parameters:
            return root_name
        return None

    def record_readonly_metal_mesh_payload_assignment_alias(self, target, value):
        if self.argument_address_space(target) != "object_data":
            return None
        readonly_key = self.readonly_metal_mesh_payload_key(value)
        if readonly_key is None:
            return None
        target_name = self.assignment_target_display_name(target)
        if not target_name:
            return None
        reason = self.current_readonly_metal_mesh_payload_reasons.get(
            readonly_key, "const object_data"
        )
        self.current_readonly_metal_mesh_payload_parameters.add(target_name)
        self.current_readonly_metal_mesh_payload_reasons[target_name] = reason
        return target_name, reason

    def readonly_metal_mesh_payload_alias_assignment_diagnostic(self, target, value):
        alias = self.record_readonly_metal_mesh_payload_assignment_alias(target, value)
        if alias is None:
            return None
        target_name, reason = alias
        reason_text = self.readonly_metal_mesh_payload_reason_text(reason)
        return self.refuse_lowering(
            f"mesh payload alias: mesh payload '{target_name}' cannot store "
            f"{reason_text}",
            "(void)0",
        )

    def readonly_raw_buffer_assignment_diagnostic(self, target):
        root_name = self.assignment_target_root_name(target)
        if root_name not in self.current_readonly_raw_buffer_parameters:
            return None
        if self.current_address_space_variables.get(
            root_name
        ) == "constant" and self.mutable_const_pointee_pointer_assignment(
            target, root_name, "constant address space"
        ):
            return None
        return self.refuse_lowering(
            f"raw buffer store: readonly buffer '{root_name}' cannot be written",
            "(void)0",
        )

    def mutable_const_pointee_pointer_assignment(self, target, root_name, reason):
        if reason not in {
            "const-qualified",
            "const-qualified local alias",
            "constant address space",
            "constant address-space local alias",
        }:
            return False
        if not isinstance(target, IdentifierNode):
            return False
        target_name = self.assignment_target_display_name(target)
        if not root_name or target_name != root_name:
            return False
        local_type = self.type_name_string(self.local_variable_types.get(root_name))
        return isinstance(local_type, str) and (
            "*" in local_type or self.is_crossgl_pointer_type(local_type)
        )

    def readonly_metal_parameter_assignment_diagnostic(self, target):
        root_name = self.assignment_target_root_name(target)
        if root_name not in self.current_readonly_metal_parameters:
            return None
        reason = self.current_readonly_metal_parameter_reasons.get(
            root_name, "readonly"
        )
        if self.mutable_const_pointee_pointer_assignment(target, root_name, reason):
            return None
        return self.refuse_lowering(
            f"parameter store: parameter '{root_name}' is {reason}", "(void)0"
        )

    def readonly_metal_program_scope_global_assignment_diagnostic(self, target):
        root_name = self.assignment_target_root_name(target)
        if (
            root_name not in self.metal_program_scope_value_globals
            or root_name in self.local_variable_types
        ):
            return None
        target_name = self.assignment_target_display_name(target) or root_name
        if root_name in self.metal_program_scope_groupshared_globals:
            return self.refuse_lowering(
                f"program-scope groupshared store: global '{target_name}' cannot be "
                "written because Metal threadgroup storage must be declared inside a "
                "kernel",
                "(void)0",
            )
        return self.refuse_lowering(
            f"program-scope global store: global '{target_name}' is emitted in the "
            "constant address space",
            "(void)0",
        )

    def readonly_metal_mesh_payload_assignment_diagnostic(self, target):
        root_name = self.readonly_metal_mesh_payload_key(target)
        if root_name is None:
            return None
        reason = self.current_readonly_metal_mesh_payload_reasons.get(
            root_name, "const object_data"
        )
        reason_text = self.readonly_metal_mesh_payload_reason_text(reason)
        return self.refuse_lowering(
            f"mesh payload store: mesh payload '{root_name}' is {reason_text}",
            "(void)0",
        )

    def readonly_metal_mesh_payload_reason_text(self, reason):
        if reason == "mesh stage payload":
            return "const object_data in mesh stages"
        if reason == "const-qualified payload parameter":
            return "const-qualified object_data"
        if reason == "address-space mismatch alias":
            return "read-only object_data after address-space mismatch"
        return reason

    def metal_mesh_payload_parameter_declaration(
        self, mapped_type, name, node=None, shader_type=None
    ):
        if not self.is_metal_mesh_payload_parameter(shader_type, node):
            return None

        readonly_reason = self.readonly_metal_mesh_payload_parameter_reason(
            shader_type, node
        )
        address_space = (
            "const object_data" if readonly_reason is not None else "object_data"
        )
        payload_type = self.metal_mesh_payload_value_type(mapped_type) or mapped_type
        return f"{address_space} {payload_type}& {name}"

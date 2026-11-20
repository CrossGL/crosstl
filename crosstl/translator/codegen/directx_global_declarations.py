"""HLSL declarations of global variables for ``HLSLCodeGen.declare_hlsl_global``: type resolution, lowered buffer blocks, register assignment per resource class, declaration qualifiers and implicit texture samplers."""

from ..ast import (
    PointerType,
)
from ..resource_identity import (
    resource_identity_marker,
)
from ..semantics.constant_buffers import (
    constant_buffer_element,
)
from ..semantics.unsized_arrays import (
    UnsizedArrayUnsupportedError,
)
from .array_utils import (
    format_c_style_array_declaration,
    split_array_type_suffix,
)
from .image_access_contracts import (
    record_explicit_image_metadata,
)


class DirectXGlobalDeclarationMixin:
    """Global-variable declaration members of ``HLSLCodeGen``."""

    def resolve_hlsl_global_type(
        self,
        node,
    ):
        """Resolve a global's declared type, array suffix and resource count."""
        resource_count = 1
        if hasattr(node, "var_type"):
            if (
                hasattr(node.var_type, "name")
                or hasattr(node.var_type, "element_type")
                or isinstance(node.var_type, PointerType)
            ):
                if (
                    hasattr(node.var_type, "element_type")
                    and str(type(node.var_type)).find("ArrayType") != -1
                ):  # ArrayType
                    base_type = self.convert_type_node_to_string(
                        node.var_type.element_type
                    )
                    # An array of arrays: the outer dimension is the first suffix.
                    base_type, element_suffix = split_array_type_suffix(base_type)
                    array_size = (
                        self.generate_expression(node.var_type.size)
                        if node.var_type.size
                        else self.resource_array_size_hints.get(node.name, "")
                    )
                    vtype = base_type
                    if array_size:
                        array_suffix = f"[{array_size}]"
                    elif (
                        self.is_resource_array_hint_type(base_type)
                        or constant_buffer_element(base_type) is not None
                    ):
                        # An unbounded resource array, ConstantBuffer<T> name[] included (SM 5.1).
                        array_suffix = "[]"
                    else:
                        raise UnsizedArrayUnsupportedError(
                            "directx", node.name, "is declared at module scope"
                        )
                    array_suffix += element_suffix
                    resource_count = self.resource_array_count(
                        node.var_type.size if node.var_type.size else array_size
                    ) * self.inner_resource_array_count(node.var_type.element_type)
                else:
                    vtype = self.convert_type_node_to_string(node.var_type)
                    array_suffix = ""
            else:
                # A promoted stage-entry resource array carries its HLSL type as text (``SamplerState[2]``).
                vtype, array_suffix = split_array_type_suffix(str(node.var_type))
                if array_suffix:
                    resource_count = self.resource_array_count(
                        array_suffix[1 : array_suffix.index("]")]
                    )
        elif hasattr(node, "vtype"):
            vtype = node.vtype
            array_suffix = ""
        else:
            vtype = "float"
            array_suffix = ""
        return resource_count, vtype, array_suffix

    def declare_hlsl_lowered_buffer_block(
        self,
        emission,
        node,
        array_suffix,
        lowered_block,
        resource_count,
        var_name,
    ):
        """Declare a GLSL buffer block that was lowered to an HLSL buffer resource."""
        mapped_type = (
            "ByteAddressBuffer" if lowered_block["readonly"] else "RWByteAddressBuffer"
        )
        if lowered_block["readonly"]:
            space = self.explicit_resource_register_space(node)
            binding = self.explicit_resource_binding_index(
                node, {"binding", "texture"}, ("t",)
            )
            if binding is None:
                binding = self.next_available_resource_register(
                    emission.used_resource_registers,
                    "t",
                    emission.texture_registers,
                    space,
                    resource_count,
                )
            self.reserve_resource_register_range(
                emission.used_resource_registers,
                "t",
                binding,
                resource_count,
                var_name,
                space,
            )
            register = self.resource_register_suffix("t", binding, node)
            self.advance_resource_register(
                emission.texture_registers, space, binding, resource_count
            )
        else:
            space = self.explicit_resource_register_space(node)
            binding = self.explicit_resource_binding_index(
                node, {"binding", "texture", "uav"}, ("u",)
            )
            if binding is None:
                binding = self.next_available_resource_register(
                    emission.used_resource_registers,
                    "u",
                    emission.uav_registers,
                    space,
                    resource_count,
                )
            self.reserve_resource_register_range(
                emission.used_resource_registers,
                "u",
                binding,
                resource_count,
                var_name,
                space,
            )
            register = self.resource_register_suffix("u", binding, node)
            self.advance_resource_register(
                emission.uav_registers, space, binding, resource_count
            )
        declaration = format_c_style_array_declaration(
            f"{mapped_type}{array_suffix}", var_name
        )
        qualifier = self.resource_memory_qualifier(mapped_type, node)
        qualifier = self.storage_image_format_prefix(mapped_type, node) + qualifier
        emission.code += f"{qualifier}{declaration}{register};\n"
        return

    def map_hlsl_global_resource_type(
        self,
        emission,
        node,
        array_suffix,
        var_name,
        vtype,
    ):
        """Map a global's type to its HLSL resource type and build the declarator."""
        mapped_type = self.hlsl_struct_buffer_resource_type(
            node, vtype
        ) or self.map_resource_type_with_format(vtype, node)
        mapped_type = self.hlsl_texel_buffers.declaration_type(
            var_name, vtype, node, mapped_type
        )
        self.hlsl_half_storage.register_resource(
            var_name,
            self.global_variable_types.get(var_name, vtype),
            array=bool(array_suffix),
        )
        if (
            var_name in emission.comparison_sampler_names
            and mapped_type == "SamplerState"
            and var_name not in self.global_mixed_sampler_names
        ):
            mapped_type = "SamplerComparisonState"
        is_hlsl_resource_global = (
            mapped_type.startswith("Texture")
            or self.is_hlsl_feedback_texture_type(mapped_type)
            or self.is_hlsl_rw_texture_type(mapped_type)
            or self.is_multisample_storage_image_resource_type(mapped_type)
            or self.is_hlsl_acceleration_structure_type(mapped_type)
            or self.is_hlsl_uav_buffer_type(mapped_type)
            or self.is_hlsl_constant_buffer_type(mapped_type)
            or self.is_hlsl_texture_buffer_type(mapped_type)
            or self.is_hlsl_readonly_buffer_type(mapped_type)
            or mapped_type in ["SamplerState", "SamplerComparisonState"]
        )
        declaration_type = self.directx_resource_declaration_type(
            f"{mapped_type}{array_suffix}"
        )
        return mapped_type, is_hlsl_resource_global, declaration_type

    def assign_hlsl_texture_register(
        self,
        emission,
        node,
        mapped_type,
        resource_count,
        var_name,
        vtype,
    ):
        """Assign the ``t`` register of a texture, multisample storage image or acceleration structure."""
        if self.is_hlsl_acceleration_structure_type(mapped_type):
            self.validate_hlsl_acceleration_structure_register(node)
        self.texture_variables.add(var_name)
        self.texture_variable_types[var_name] = mapped_type
        if self.is_image_type(vtype):
            record_explicit_image_metadata(
                var_name,
                node,
                self.attribute_value_to_string,
                image_formats=self.image_variable_formats,
                image_accesses=self.image_variable_accesses,
            )
        space = self.explicit_resource_register_space(node)
        binding = self.explicit_resource_binding_index(
            node, {"binding", "texture"}, ("t",)
        )
        if binding is None:
            binding = self.next_available_resource_register(
                emission.used_resource_registers,
                "t",
                emission.texture_registers,
                space,
                resource_count,
            )
        self.reserve_resource_register_range(
            emission.used_resource_registers,
            "t",
            binding,
            resource_count,
            var_name,
            space,
        )
        register = self.resource_register_suffix("t", binding, node)
        self.advance_resource_register(
            emission.texture_registers, space, binding, resource_count
        )
        return space, register

    def assign_hlsl_feedback_texture_register(
        self,
        emission,
        node,
        mapped_type,
        resource_count,
        var_name,
    ):
        """Assign the ``u`` register of a sampler feedback texture."""
        self.texture_variable_types[var_name] = mapped_type
        space = self.explicit_resource_register_space(node)
        binding = self.explicit_resource_binding_index(
            node, {"binding", "texture", "uav"}, ("u",)
        )
        if binding is None:
            binding = self.next_available_resource_register(
                emission.used_resource_registers,
                "u",
                emission.uav_registers,
                space,
                resource_count,
            )
        self.reserve_resource_register_range(
            emission.used_resource_registers,
            "u",
            binding,
            resource_count,
            var_name,
            space,
        )
        register = self.resource_register_suffix("u", binding, node)
        self.advance_resource_register(
            emission.uav_registers, space, binding, resource_count
        )
        return space, register

    def assign_hlsl_rw_texture_register(
        self,
        emission,
        node,
        mapped_type,
        resource_count,
        var_name,
    ):
        """Assign the ``u`` register of a read-write texture."""
        self.texture_variable_types[var_name] = mapped_type
        record_explicit_image_metadata(
            var_name,
            node,
            self.attribute_value_to_string,
            image_formats=self.image_variable_formats,
            image_accesses=self.image_variable_accesses,
        )
        space = self.explicit_resource_register_space(node)
        binding = self.explicit_resource_binding_index(
            node, {"binding", "texture", "uav"}, ("u",)
        )
        if binding is None:
            binding = self.next_available_resource_register(
                emission.used_resource_registers,
                "u",
                emission.uav_registers,
                space,
                resource_count,
            )
        self.reserve_resource_register_range(
            emission.used_resource_registers,
            "u",
            binding,
            resource_count,
            var_name,
            space,
        )
        register = self.resource_register_suffix("u", binding, node)
        self.advance_resource_register(
            emission.uav_registers, space, binding, resource_count
        )
        return space, register

    def assign_hlsl_uav_buffer_register(
        self,
        emission,
        node,
        resource_count,
        var_name,
    ):
        """Assign the ``u`` register of a read-write buffer."""
        space = self.explicit_resource_register_space(node)
        binding = self.explicit_resource_binding_index(
            node, {"binding", "buffer", "texture", "uav"}, ("u",)
        )
        if binding is None:
            binding = self.next_available_resource_register(
                emission.used_resource_registers,
                "u",
                emission.uav_registers,
                space,
                resource_count,
            )
        self.reserve_resource_register_range(
            emission.used_resource_registers,
            "u",
            binding,
            resource_count,
            var_name,
            space,
        )
        register = self.resource_register_suffix("u", binding, node)
        self.advance_resource_register(
            emission.uav_registers, space, binding, resource_count
        )
        return space, register

    def assign_hlsl_constant_buffer_register(
        self,
        emission,
        node,
        resource_count,
        var_name,
    ):
        """Assign the ``b`` register of a constant buffer."""
        space = self.explicit_resource_register_space(node)
        binding = self.explicit_resource_binding_index(
            node, {"binding", "buffer"}, ("b",)
        )
        if binding is None:
            binding = self.next_available_resource_register(
                emission.used_resource_registers,
                "b",
                emission.cbuffer_registers,
                space,
                resource_count,
            )
        self.reserve_resource_register_range(
            emission.used_resource_registers,
            "b",
            binding,
            resource_count,
            var_name,
            space,
        )
        register = self.resource_register_suffix("b", binding, node)
        self.advance_resource_register(
            emission.cbuffer_registers, space, binding, resource_count
        )
        return space, register

    def assign_hlsl_texture_buffer_register(
        self,
        emission,
        node,
        resource_count,
        var_name,
    ):
        """Assign the ``t`` register of a texture buffer."""
        space = self.explicit_resource_register_space(node)
        binding = self.explicit_resource_binding_index(
            node, {"binding", "buffer", "texture"}, ("t",)
        )
        if binding is None:
            binding = self.next_available_resource_register(
                emission.used_resource_registers,
                "t",
                emission.texture_registers,
                space,
                resource_count,
            )
        self.reserve_resource_register_range(
            emission.used_resource_registers,
            "t",
            binding,
            resource_count,
            var_name,
            space,
        )
        register = self.resource_register_suffix("t", binding, node)
        self.advance_resource_register(
            emission.texture_registers, space, binding, resource_count
        )
        return space, register

    def assign_hlsl_readonly_buffer_register(
        self,
        emission,
        node,
        resource_count,
        var_name,
    ):
        """Assign the ``t`` register of a read-only buffer."""
        space = self.explicit_resource_register_space(node)
        binding = self.explicit_resource_binding_index(
            node, {"binding", "buffer", "texture"}, ("t",)
        )
        if binding is None:
            binding = self.next_available_resource_register(
                emission.used_resource_registers,
                "t",
                emission.texture_registers,
                space,
                resource_count,
            )
        self.reserve_resource_register_range(
            emission.used_resource_registers,
            "t",
            binding,
            resource_count,
            var_name,
            space,
        )
        register = self.resource_register_suffix("t", binding, node)
        self.advance_resource_register(
            emission.texture_registers, space, binding, resource_count
        )
        return space, register

    def assign_hlsl_sampler_register(
        self,
        emission,
        node,
        resource_count,
        var_name,
    ):
        """Assign the ``s`` register of a sampler."""
        self.sampler_variables.add(var_name)
        space = self.explicit_resource_register_space(node)
        binding = self.explicit_resource_binding_index(
            node, {"binding", "sampler"}, ("s",)
        )
        if binding is None:
            binding = self.next_available_resource_register(
                emission.used_resource_registers,
                "s",
                emission.sampler_registers,
                space,
                resource_count,
            )
        self.reserve_resource_register_range(
            emission.used_resource_registers,
            "s",
            binding,
            resource_count,
            var_name,
            space,
        )
        register = self.resource_register_suffix("s", binding, node)
        self.advance_resource_register(
            emission.sampler_registers, space, binding, resource_count
        )
        return space, register

    def emit_hlsl_global_declaration(
        self,
        emission,
        node,
        array_suffix,
        declaration,
        is_hlsl_resource_global,
        mapped_type,
        register,
        var_name,
    ):
        """Write the qualified declaration, its initializer and any comparison-sampler alias."""

        qualifier = self.resource_memory_qualifier(mapped_type, node)
        if not qualifier and not is_hlsl_resource_global:
            qualifier = self.local_variable_qualifier(node)
            # A module variable without ``static`` is an implicitly constant member of the $Globals
            # constant buffer in HLSL; private per-invocation state needs ``static``.
            if {"thread", "static"} & {
                str(value) for value in getattr(node, "qualifiers", []) or []
            } and not qualifier.startswith("static"):
                qualifier = "static " + qualifier
        emission.code += resource_identity_marker(node, var_name)
        if id(node) in self.hlsl_inferred_global_node_ids:
            if not qualifier.startswith("static"):
                qualifier = "static " + qualifier
            initializer = self.generate_expression_with_expected(
                node.initial_value, node.var_type
            )
            declaration += f" = {initializer}"
        elif (
            qualifier.startswith("static")
            and getattr(node, "initial_value", None) is not None
        ):
            # A static global starts from its initializer, not from zero.
            initializer = self.generate_expression_with_expected(
                node.initial_value, node.var_type
            )
            declaration += f" = {initializer}"
        image_format_prefix = self.storage_image_format_prefix(mapped_type, node)
        emission.code += f"{image_format_prefix}{qualifier}{declaration}{register};\n"
        if (
            mapped_type == "SamplerState"
            and var_name in self.global_mixed_sampler_aliases
        ):
            alias_name = self.global_mixed_sampler_aliases[var_name]
            alias_type = f"SamplerComparisonState{array_suffix}"
            alias_declaration = format_c_style_array_declaration(alias_type, alias_name)
            self.sampler_variables.add(alias_name)
            emission.code += f"{alias_declaration}{register};\n"

    def declare_hlsl_implicit_texture_samplers(
        self,
        emission,
        space,
        var_name,
        vtype,
    ):
        """Declare the samplers that a texture's implicit sampling, comparison or level-of-detail queries need."""
        sampler_name = f"{var_name}Sampler"
        needs_implicit_sampler = (
            var_name in emission.global_implicit_sampler_texture_names
        )
        needs_query_lod_sampler = (
            var_name in emission.global_implicit_query_lod_texture_names
        )
        needs_regular_sampler = (
            var_name in emission.global_implicit_regular_sampler_texture_names
        )
        needs_comparison_sampler = var_name in emission.comparison_texture_names
        if (
            needs_implicit_sampler
            and sampler_name not in emission.explicit_sampler_names
            and not self.is_multisample_sampler_type(vtype)
        ):
            sampler_type = (
                "SamplerComparisonState" if needs_comparison_sampler else "SamplerState"
            )
            self.sampler_variables.add(sampler_name)
            sampler_binding = self.next_available_resource_register(
                emission.used_resource_registers,
                "s",
                emission.sampler_registers,
                space,
                1,
            )
            self.reserve_resource_register_range(
                emission.used_resource_registers,
                "s",
                sampler_binding,
                1,
                sampler_name,
                space,
            )
            sampler_register = self.resource_register_suffix_for_space(
                "s", sampler_binding, space
            )
            emission.code += f"{sampler_type} {sampler_name}{sampler_register};\n"
            self.advance_resource_register(
                emission.sampler_registers, space, sampler_binding, 1
            )
        if needs_regular_sampler and needs_comparison_sampler:
            regular_sampler_name = self.implicit_regular_sampler_name(
                var_name,
                {
                    "comparison": needs_comparison_sampler,
                    "regular": needs_regular_sampler,
                    "sampler_name": sampler_name,
                },
            )
            self.global_implicit_texture_regular_samplers[var_name] = (
                regular_sampler_name
            )
            if (
                regular_sampler_name not in emission.explicit_sampler_names
                and not self.is_multisample_sampler_type(vtype)
            ):
                self.sampler_variables.add(regular_sampler_name)
                sampler_binding = self.next_available_resource_register(
                    emission.used_resource_registers,
                    "s",
                    emission.sampler_registers,
                    space,
                    1,
                )
                self.reserve_resource_register_range(
                    emission.used_resource_registers,
                    "s",
                    sampler_binding,
                    1,
                    regular_sampler_name,
                    space,
                )
                sampler_register = self.resource_register_suffix_for_space(
                    "s", sampler_binding, space
                )
                emission.code += (
                    f"SamplerState {regular_sampler_name}{sampler_register};\n"
                )
                self.advance_resource_register(
                    emission.sampler_registers, space, sampler_binding, 1
                )
        if needs_query_lod_sampler:
            query_sampler_name = (
                f"{var_name}QuerySampler" if needs_comparison_sampler else sampler_name
            )
            self.global_implicit_texture_query_lod_samplers[var_name] = (
                query_sampler_name
            )
            if (
                needs_comparison_sampler
                and query_sampler_name not in emission.explicit_sampler_names
                and not self.is_multisample_sampler_type(vtype)
            ):
                self.sampler_variables.add(query_sampler_name)
                sampler_binding = self.next_available_resource_register(
                    emission.used_resource_registers,
                    "s",
                    emission.sampler_registers,
                    space,
                    1,
                )
                self.reserve_resource_register_range(
                    emission.used_resource_registers,
                    "s",
                    sampler_binding,
                    1,
                    query_sampler_name,
                    space,
                )
                sampler_register = self.resource_register_suffix_for_space(
                    "s", sampler_binding, space
                )
                emission.code += (
                    f"SamplerState {query_sampler_name}{sampler_register};\n"
                )
                self.advance_resource_register(
                    emission.sampler_registers, space, sampler_binding, 1
                )

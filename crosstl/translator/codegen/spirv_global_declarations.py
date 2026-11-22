"""Global declarations and interfaces of the SPIR-V generator: distance and interface built-ins, global variables, uniform array blocks, descriptor slots, sets and binding reservation, global resource nodes, buffer block type names, global interface locations and spans, interface integer attributes and decorations, declaration metadata names, mesh payload, mesh output and graphics output parameters, patch parameters, and tessellation patch constant built-ins."""

from typing import (
    Optional,
    Set,
    Tuple,
)

from ..ast import (
    ShaderNode,
    UnaryOpNode,
    VariableNode,
)
from ..refusals import VulkanRefusal
from .device_globals import device_global_refusal_reason, is_mutable_device_global
from .enum_utils import generic_type_parts
from .precise_dataflow import declaration_is_precise
from .spirv_conservative_depth import record_depth_layout
from .spirv_hlsl_registers import (
    reserve_exact_binding,
    reserve_hlsl_descriptor_slots,
)
from .spirv_invariant_outputs import decorate_invariant_builtin_output
from .spirv_location_conflicts import InterfaceLocationConflictError
from .spirv_module_constants import module_constant
from .spirv_per_vertex_builtins import sized_per_vertex_interface_type
from .spirv_placed_blocks import declare_placed_struct_uniform
from .spirv_push_constants import (
    declare_push_constant_variable,
    is_push_constant_block,
)
from .spirv_uniform_block_arrays import (
    declare_uniform_block_array,
    is_uniform_block_array_node,
    lowered_constant_buffer_nodes,
)
from .spirv_values import SpirvId


class SpirvGlobalDeclarationMixin:
    """Global declaration members of ``VulkanSPIRVCodeGen``."""

    # Clip and cull distances are float arrays whose size the shader declares (GLSL 4.60 section 7.1).
    DISTANCE_BUILTINS = {
        "gl_ClipDistance": ("ClipDistance", "ClipDistance"),
        "gl_CullDistance": ("CullDistance", "CullDistance"),
    }
    DISTANCE_BUILTIN_STAGES = {
        "Output": {"Vertex", "TessellationEvaluation", "Geometry"},
        "Input": {"Fragment"},
    }

    def global_interface_distance_builtin(
        self, node: VariableNode, storage_class: str, type_id: SpirvId
    ) -> Optional[SpirvId]:
        """The ``ClipDistance`` or ``CullDistance`` built-in a stage-level variable declares, or ``None``."""
        semantic = self.semantic_from_node(node)
        name = str(semantic) if semantic is not None else getattr(node, "name", "")
        if name not in self.DISTANCE_BUILTINS:
            return None
        builtin_name, capability = self.DISTANCE_BUILTINS[name]
        execution_model = self.current_execution_model
        if execution_model not in self.DISTANCE_BUILTIN_STAGES.get(storage_class, ()):
            raise VulkanRefusal(
                f"SPIR-V {name} is not a {storage_class.lower()} of the "
                f"{execution_model or 'current'} stage"
            )
        array_info = self.array_type_info_from_type(type_id)
        element_type = array_info[0] if array_info is not None else None
        if (
            element_type is None
            or self.normalize_primitive_name(element_type.type.base_type) != "float"
        ):
            raise VulkanRefusal(
                f"SPIR-V {name} must be declared as a sized float array"
            )
        cache_key = f"__distance_builtin::{name}::{storage_class}"
        if cache_key in self.global_variables:
            builtin = self.global_variables[cache_key]
            self.mark_builtin_interface_variable(builtin)
            return builtin
        self.require_capability(capability)
        builtin = self.register_builtin_variable(
            name, type_id, builtin_name, storage_class
        )
        self.global_variables[cache_key] = builtin
        self.mark_builtin_interface_variable(builtin)
        return builtin

    def global_interface_builtin_variable(
        self, node: VariableNode, storage_class: str, type_id: SpirvId
    ) -> Optional[SpirvId]:
        distance = self.global_interface_distance_builtin(node, storage_class, type_id)
        if distance is not None:
            return distance
        semantic = self.semantic_from_node(node)
        if semantic is None and self.stage_builtin_info(getattr(node, "name", "")):
            semantic = node.name
        if semantic is None:
            return None

        semantic = self.spirv_interface_semantic_alias(
            str(semantic), self.current_execution_model, storage_class
        )
        builtin = self.entry_point_builtin_variable(semantic, storage_class)
        if builtin is None:
            return None

        builtin_type = self.variable_value_types.get(builtin.id)
        if builtin_type is not None:
            self.validate_spirv_builtin_semantic_type(
                semantic,
                self.crossgl_spelling_of_type(type_id),
                "interface declaration",
            )
        return builtin

    def crossgl_spelling_of_type(self, type_id: SpirvId) -> str:
        """``type_id``'s CrossGL spelling, ``int[1]`` for a sized array, as the semantic type checks read it."""
        array_info = self.array_type_info_from_type(type_id)
        if array_info is None or array_info[0] is None:
            return type_id.type.base_type
        element_type, size = array_info
        if isinstance(size, int):
            return f"{self.crossgl_spelling_of_type(element_type)}[{size}]"
        return type_id.type.base_type

    def process_global_variable_declaration(
        self, node: VariableNode, default_storage_class: str = "Private"
    ) -> SpirvId:
        """Process a module-scope CrossGL variable declaration."""
        if self.is_default_uniform_node(node):
            return self.default_uniform_block
        var_type_source = getattr(node, "var_type", getattr(node, "vtype", "float"))
        var_type_name = self.type_name_from_value(var_type_source)
        if self.is_glsl_buffer_block_node(node):
            return self.process_glsl_buffer_block_declaration(node, var_type_name)

        if self.is_structured_buffer_declared_type_name(var_type_name):
            return self.process_structured_buffer_declaration(node, var_type_name)
        if is_mutable_device_global(
            node
        ) and not self.is_uniform_constant_resource_node(node):
            raise VulkanRefusal(device_global_refusal_reason(node, "SPIR-V"))

        var_type = self.map_resource_type_with_format(var_type_source, node)
        storage_class = self.infer_global_storage_class(
            node, default_storage_class, var_type_name
        )
        if is_uniform_block_array_node(node):
            return declare_uniform_block_array(self, node, var_type)
        if storage_class == "Uniform" and self.uniform_array_requires_block(var_type):
            return self.process_uniform_array_block_declaration(node, var_type)
        if storage_class in {"Uniform", "Private"} and is_push_constant_block(
            self, node
        ):
            # 'uniform T name' and HLSL's plain '[[vk::push_constant]] T name' global both name the push constants.
            push_constant = declare_push_constant_variable(self, node, var_type)
            if push_constant is not None:
                return push_constant
        if storage_class == "Uniform":
            # A ConstantBuffer<T> whose T places members with @offset or @align keeps those bytes.
            placed_variable = declare_placed_struct_uniform(self, node, var_type)
            if placed_variable is not None:
                return placed_variable
        if (
            storage_class == "Uniform"
            and self.current_struct_members.get(var_type.type.base_type) is not None
        ):
            var_type = self.storage_layout_type(var_type, "std140")

        initializer = None
        initial_value = getattr(node, "initial_value", None)
        if storage_class == "Private" and initial_value is not None:
            initializer = self.process_constant_expression(initial_value, var_type)
            if initializer is not None:
                initializer = self.coerce_scalar_constant_to_type(initializer, var_type)

        if (
            storage_class == "Private"
            and initializer is not None
            and self.is_specialization_constant_value(initializer)
        ):
            # A module-scope constant whose initializer folds to a
            # specialization-constant operation (e.g. `span_h = tgp_h * str_h + ...`)
            # is registered as a named constant rather than a Private variable, so
            # that references resolve to the OpSpecConstantOp result directly instead
            # of emitting an OpLoad of a runtime variable at constant scope.
            self.emit_named_constant_debug_name(node.name, initializer)
            self.named_constants[node.name] = initializer
            if declaration_is_precise(node):
                self.precise_global_variables.add(node.name)
            return initializer
        if storage_class == "Private" and not declaration_is_precise(node):
            constant = module_constant(self, node, var_type, initializer)
            if constant is not None:
                return constant

        if storage_class in {"Input", "Output"}:
            var_type = sized_per_vertex_interface_type(self, var_type, storage_class)
        if storage_class == "Input":
            var_id = self.global_interface_builtin_variable(node, "Input", var_type)
            record_depth_layout(self, var_id, node, storage_class)
            if var_id is None:
                location = self.global_interface_location(node, "Input")
                var_id = self.register_input(
                    node.name, var_type, location, 0, source_node=node
                )
                self.decorate_global_interface_variable(node, var_id)
        elif storage_class == "Output":
            var_id = self.global_interface_builtin_variable(node, "Output", var_type)
            record_depth_layout(self, var_id, node, storage_class)
            decorate_invariant_builtin_output(self, var_id, node, storage_class)
            if var_id is None:
                location = self.global_interface_location(node, "Output")
                var_id = self.register_output(
                    node.name, var_type, location, 0, source_node=node
                )
                self.decorate_global_interface_variable(node, var_id)
        else:
            var_id = self.create_variable(
                var_type, storage_class, node.name, initializer
            )
            if storage_class == "TaskPayloadWorkgroupEXT":
                self.require_capability("MeshShadingEXT")
                self.require_extension("SPV_EXT_mesh_shader")
                self.task_payload_shared_variables[node.name] = var_id
            if storage_class == "UniformConstant":
                descriptor_set, binding = self.resource_descriptor_slot(node)
                self.decorations.append(
                    f"OpDecorate %{var_id.id} DescriptorSet {descriptor_set}"
                )
                self.decorations.append(f"OpDecorate %{var_id.id} Binding {binding}")
                self.register_declared_resource_metadata(node, var_id, var_type)
                metadata = self.resource_metadata_for_value(var_id)
                if metadata and metadata.get("kind") == "input_attachment":
                    input_attachment_index = self.explicit_interface_integer_attribute(
                        node, "input_attachment_index"
                    )
                    if input_attachment_index is None:
                        input_attachment_index = binding
                    self.decorations.append(
                        f"OpDecorate %{var_id.id} "
                        f"InputAttachmentIndex {input_attachment_index}"
                    )
            elif storage_class == "Uniform":
                members = self.current_struct_members.get(var_type.type.base_type)
                if members is not None:
                    self.decorate_cbuffer_type(var_type, members)
                descriptor_set, binding = self.resource_descriptor_slot(node)
                self.decorations.append(
                    f"OpDecorate %{var_id.id} DescriptorSet {descriptor_set}"
                )
                self.decorations.append(f"OpDecorate %{var_id.id} Binding {binding}")
                self.uniform_buffers.append(var_id)

        self.register_global_variable_name(node.name, var_id, storage_class)
        if declaration_is_precise(node):
            self.precise_global_variables.add(node.name)
        return var_id

    def uniform_array_requires_block(self, type_id: SpirvId) -> bool:
        array_info = self.array_type_info_from_type(type_id)
        if array_info is None:
            return False

        element_type, _ = array_info
        while True:
            nested_array = self.array_type_info_from_type(element_type)
            if nested_array is None:
                break
            element_type, _ = nested_array

        return element_type.type.base_type in self.current_struct_members

    def process_uniform_array_block_declaration(
        self, node: VariableNode, value_type: SpirvId
    ) -> SpirvId:
        """Wrap a standalone uniform struct array in a Vulkan Block type."""
        layout_value_type = self.storage_layout_type(value_type, "std140")
        block_name = f"{node.name}Block"
        block_type = self.register_struct_type(
            block_name, [(layout_value_type, node.name)]
        )
        self.decorate_cbuffer_type(block_type, [(layout_value_type, node.name)])

        var_id = self.create_variable(block_type, "Uniform", node.name)
        descriptor_set, binding = self.resource_descriptor_slot(node)
        self.decorations.append(
            f"OpDecorate %{var_id.id} DescriptorSet {descriptor_set}"
        )
        self.decorations.append(f"OpDecorate %{var_id.id} Binding {binding}")

        self.global_variables[node.name] = var_id
        self.uniform_buffers.append(var_id)
        self.uniform_block_wrapped_variables[var_id.id] = {
            "member_index": 0,
            "member_type": layout_value_type,
        }
        return var_id

    def resource_descriptor_slot(
        self, node: VariableNode, *, allow_binding_reassignment: bool = False
    ) -> Tuple[int, int]:
        hlsl_slot = self.hlsl_descriptor_slots.get(id(node))
        if hlsl_slot is None:
            hlsl_slot = self.entry_descriptor_slots.get(getattr(node, "name", None))
        if hlsl_slot is not None:
            self.used_resource_bindings.add(hlsl_slot)
            return hlsl_slot
        descriptor_set = self.resource_descriptor_set(node)
        explicit_binding = self.explicit_descriptor_binding(node)

        if explicit_binding is not None:
            key = (descriptor_set, explicit_binding)
            if id(node) in self.shared_binding_nodes:
                # One half of a combined image sampler whose other half holds the same slot.
                self.used_resource_bindings.add(key)
                return key
            if key in self.used_resource_bindings or (
                allow_binding_reassignment and key in self.reserved_resource_bindings
            ):
                if allow_binding_reassignment:
                    binding = self.next_available_resource_binding(descriptor_set)
                    self.used_resource_bindings.add((descriptor_set, binding))
                    return descriptor_set, binding
                raise VulkanRefusal(
                    f"Duplicate SPIR-V resource binding set {descriptor_set} "
                    f"binding {explicit_binding}"
                )
            self.used_resource_bindings.add(key)
            return descriptor_set, explicit_binding

        binding = self.next_available_resource_binding(descriptor_set)
        self.used_resource_bindings.add((descriptor_set, binding))
        return descriptor_set, binding

    def resource_descriptor_set(self, node: VariableNode) -> int:
        descriptor_set = self.explicit_interface_integer_attribute(node, "set")
        if descriptor_set is None:
            descriptor_set = self.explicit_interface_integer_attribute(node, "group")
        if descriptor_set is None:
            descriptor_set = self.explicit_interface_integer_attribute(node, "space")
        return 0 if descriptor_set is None else descriptor_set

    def next_available_resource_binding(self, descriptor_set: int) -> int:
        binding = self.next_resource_bindings.get(descriptor_set, 0)
        while (descriptor_set, binding) in self.reserved_resource_bindings or (
            descriptor_set,
            binding,
        ) in self.used_resource_bindings:
            binding += 1

        self.next_resource_bindings[descriptor_set] = binding + 1
        if descriptor_set == 0:
            self.next_resource_binding = binding + 1
        return binding

    def resource_binding_state(self):
        return (
            self.next_resource_binding,
            dict(self.next_resource_bindings),
            set(self.reserved_resource_bindings),
            set(self.used_resource_bindings),
        )

    def restore_resource_binding_state(self, state):
        (
            self.next_resource_binding,
            next_resource_bindings,
            reserved_resource_bindings,
            used_resource_bindings,
        ) = state
        self.next_resource_bindings = dict(next_resource_bindings)
        self.reserved_resource_bindings = set(reserved_resource_bindings)
        self.used_resource_bindings = set(used_resource_bindings)

    def reserve_explicit_resource_bindings(self, ast: ShaderNode):
        for node in self.global_descriptor_binding_nodes(ast):
            explicit_binding = self.explicit_descriptor_binding(node)
            if explicit_binding is None:
                continue

            descriptor_set = self.resource_descriptor_set(node)
            reserve_exact_binding(
                self,
                (descriptor_set, explicit_binding),
                node,
                getattr(node, "name", "resource"),
            )
        stage_of = {
            id(node): str(stage_name)
            for stage_name, stage in (getattr(ast, "stages", None) or {}).items()
            for node in getattr(stage, "local_variables", []) or []
        }
        self.hlsl_descriptor_slots = reserve_hlsl_descriptor_slots(
            self,
            self.global_descriptor_binding_nodes(ast),
            skip=lambda node: is_push_constant_block(self, node),
            scope=lambda node: stage_of.get(id(node)),
        )

    def explicit_descriptor_binding(self, node: VariableNode) -> Optional[int]:
        binding = self.explicit_interface_integer_attribute(node, "binding")
        if binding is not None:
            return binding
        return self.explicit_interface_integer_attribute(node, "buffer")

    def global_descriptor_binding_nodes(self, ast: ShaderNode):
        yield from self.global_resource_nodes(ast)
        yield from self.global_structured_buffer_nodes(ast)
        yield from getattr(ast, "cbuffers", []) or []
        yield from lowered_constant_buffer_nodes(ast)

    def global_structured_buffer_nodes(self, ast: ShaderNode):
        nodes = list(getattr(ast, "global_variables", []) or [])
        for stage in (getattr(ast, "stages", None) or {}).values():
            nodes.extend(getattr(stage, "local_variables", []) or [])

        seen = set()
        for node in nodes:
            node_id = id(node)
            if node_id in seen:
                continue
            seen.add(node_id)
            type_source = getattr(node, "var_type", getattr(node, "vtype", "float"))
            type_name = self.type_name_from_value(type_source)
            if self.is_structured_buffer_declared_type_name(
                type_name
            ) or self.is_glsl_buffer_block_node(node):
                yield node

    def global_resource_nodes(self, ast: ShaderNode):
        nodes = list(getattr(ast, "global_variables", []) or [])
        for stage in (getattr(ast, "stages", None) or {}).values():
            nodes.extend(getattr(stage, "local_variables", []) or [])

        seen = set()
        for node in nodes:
            node_id = id(node)
            if node_id in seen:
                continue
            seen.add(node_id)
            if self.is_uniform_constant_resource_node(node):
                yield node

    def collect_source_variable_type_names(self, ast: ShaderNode) -> dict:
        type_names = {}

        def record_variable(node):
            name = getattr(node, "name", None)
            if not name:
                return
            type_source = getattr(
                node,
                "var_type",
                getattr(
                    node,
                    "member_type",
                    getattr(node, "param_type", getattr(node, "vtype", None)),
                ),
            )
            type_name = self.type_name_from_value(type_source)
            if type_name is not None:
                type_names[name] = type_name

        def record_cbuffer(cbuffer):
            record_variable(cbuffer)
            for member in getattr(cbuffer, "members", []) or []:
                record_variable(member)

        for node in getattr(ast, "constants", []) or []:
            # Program constants, such as ``const float LUT[3]``, are typed by ``const_type``.
            type_name = self.type_name_from_value(getattr(node, "const_type", None))
            if getattr(node, "name", None) and type_name is not None:
                type_names[node.name] = type_name
        for node in getattr(ast, "global_variables", []) or []:
            record_variable(node)
        for cbuffer in getattr(ast, "cbuffers", []) or []:
            record_cbuffer(cbuffer)
        for stage in (getattr(ast, "stages", None) or {}).values():
            for node in getattr(stage, "local_variables", []) or []:
                record_variable(node)
            for cbuffer in getattr(stage, "local_cbuffers", []) or []:
                record_cbuffer(cbuffer)

        return type_names

    def collect_glsl_buffer_block_type_names(self, ast: ShaderNode) -> set:
        nodes = list(getattr(ast, "global_variables", []) or [])
        for stage in (getattr(ast, "stages", None) or {}).values():
            nodes.extend(getattr(stage, "local_variables", []) or [])

        type_names = set()
        for node in nodes:
            if not self.is_glsl_buffer_block_node(node):
                continue
            type_source = getattr(node, "var_type", getattr(node, "vtype", None))
            type_name = self.type_name_from_value(type_source)
            base_type_name = self.array_base_type_name(type_name)
            if base_type_name in self.struct_declarations:
                type_names.add(base_type_name)
        return type_names

    def is_uniform_constant_resource_node(self, node: VariableNode) -> bool:
        type_source = getattr(node, "var_type", getattr(node, "vtype", "float"))
        type_name = self.type_name_from_value(type_source)
        base_type_name = self.array_base_type_name(type_name)
        return self.is_resource_type_name(base_type_name)

    def global_interface_location(
        self,
        node: VariableNode,
        storage_class: str,
        preferred_location: Optional[int] = None,
        patch: bool = False,
    ) -> int:
        if storage_class == "Input":
            counter_name = "next_input_location"
            used_slots = self.used_input_locations
        else:
            counter_name = "next_output_location"
            used_slots = self.used_output_locations

        scope = self.interface_location_scope()
        counter_key = (storage_class, scope)
        span = self.interface_location_span(node)

        explicit_location = self.explicit_location_attribute(node)
        if explicit_location is not None:
            slot_keys = self.interface_slot_keys(
                node, storage_class, explicit_location, patch=patch
            )
            if used_slots & slot_keys:
                raise InterfaceLocationConflictError(
                    storage_class, explicit_location, getattr(node, "name", None)
                )
            used_slots.update(slot_keys)
            return explicit_location

        if preferred_location is not None:
            slot_keys = self.interface_slot_keys(
                node, storage_class, preferred_location, patch=patch
            )
            if used_slots & slot_keys:
                raise InterfaceLocationConflictError(
                    storage_class, preferred_location, getattr(node, "name", None)
                )
            used_slots.update(slot_keys)
            self.record_global_interface_location(
                counter_name, counter_key, preferred_location, span
            )
            return preferred_location

        location = self.interface_location_counters.get(counter_key, 0)
        slot_keys = self.interface_slot_keys(node, storage_class, location, patch=patch)
        while used_slots & slot_keys:
            location += 1
            slot_keys = self.interface_slot_keys(
                node, storage_class, location, patch=patch
            )
        used_slots.update(slot_keys)
        self.record_global_interface_location(counter_name, counter_key, location, span)
        return location

    def record_global_interface_location(
        self, counter_name: str, counter_key: tuple, location: int, span: int
    ):
        next_location = location + max(span, 1)
        self.interface_location_counters[counter_key] = max(
            self.interface_location_counters.get(counter_key, 0), next_location
        )
        setattr(self, counter_name, max(getattr(self, counter_name), next_location))

    def interface_location_scope(self):
        return self.current_execution_model or self.current_function_name or "module"

    def explicit_location_attribute(self, node: VariableNode) -> Optional[int]:
        return self.explicit_interface_integer_attribute(node, "location")

    def explicit_component_attribute(self, node: VariableNode) -> Optional[int]:
        component = self.explicit_interface_integer_attribute(node, "component")
        if component is not None and component > 3:
            raise VulkanRefusal(f"SPIR-V component must be in 0..3: {component}")
        return component

    def interface_location_span(self, node: VariableNode) -> int:
        return max(len(self.interface_location_component_widths(node)), 1)

    def struct_member_interface_location_span(self, member_node) -> int:
        """Locations a stage-interface structure member occupies; built-ins and unknown members count one."""
        if member_node is None:
            return 1
        return self.interface_location_span(member_node)

    def explicit_interface_integer_attribute(
        self, node: VariableNode, attribute_name: str
    ) -> Optional[int]:
        for attr in getattr(node, "attributes", []) or []:
            if str(getattr(attr, "name", "")).lower() != attribute_name:
                continue

            arguments = getattr(attr, "arguments", None)
            if arguments is None:
                arguments = getattr(attr, "args", [])
            if not arguments:
                continue

            attribute_value = self.interface_integer_attribute_value(
                arguments[0], attribute_name
            )
            if attribute_value is None:
                continue

            if attribute_value < 0:
                raise VulkanRefusal(
                    f"SPIR-V {attribute_name} must be non-negative: "
                    f"{attribute_value}"
                )
            return attribute_value

        return None

    def interface_integer_attribute_value(self, value, attribute_name: str):
        if isinstance(value, UnaryOpNode) and value.operator == "-":
            operand_value = self.interface_integer_attribute_value(
                value.operand, attribute_name
            )
            return -operand_value if operand_value is not None else None

        value_text = self.attribute_value_to_string(value)
        if value_text is None:
            return None

        try:
            return int(str(value_text), 0)
        except ValueError as exc:
            raise VulkanRefusal(
                f"SPIR-V {attribute_name} must be an integer: {value_text}"
            ) from exc

    def decorate_global_interface_variable(self, node: VariableNode, var_id: SpirvId):
        self.validate_interface_interpolation_attributes(node)

        for attribute_name, decoration_name in {
            "component": "Component",
            "index": "Index",
        }.items():
            attribute_value = self.explicit_interface_integer_attribute(
                node, attribute_name
            )
            if attribute_value is not None:
                self.decorations.append(
                    f"OpDecorate %{var_id.id} {decoration_name} {attribute_value}"
                )

        for attribute_name, decoration_name in {
            "flat": "Flat",
            "noperspective": "NoPerspective",
            "centroid": "Centroid",
            "sample": "Sample",
            "invariant": "Invariant",
        }.items():
            if self.has_attribute(node, attribute_name):
                if attribute_name == "sample":
                    self.require_capability("SampleRateShading")
                self.decorations.append(f"OpDecorate %{var_id.id} {decoration_name}")
        if self.has_attribute(node, "patch") or "patch" in {
            str(qualifier).lower() for qualifier in getattr(node, "qualifiers", ())
        }:
            # GLSL `patch in`/`patch out` and CrossGL `@patch`: one value per patch, not per control point.
            self.decorations.append(f"OpDecorate %{var_id.id} Patch")
        if self.declaration_metadata_names(node) & {
            "pervertex",
            "pervertexext",
            "pervertexnv",
        }:
            # SPV_KHR_fragment_shader_barycentric: each element is one vertex's value, not an interpolant.
            self.require_capability("FragmentBarycentricKHR")
            self.require_extension("SPV_KHR_fragment_shader_barycentric")
            self.decorations.append(f"OpDecorate %{var_id.id} PerVertexKHR")
        if self.declaration_metadata_names(node) & {
            "perprimitive",
            "perprimitiveext",
            "per_primitive",
        }:
            # GL_EXT_mesh_shader `perprimitiveEXT`: one value per mesh primitive, written by the mesh stage and
            # read flat by the fragment stage (SPV_EXT_mesh_shader, PerPrimitiveEXT decoration).
            self.require_capability("MeshShadingEXT")
            self.require_extension("SPV_EXT_mesh_shader")
            self.decorations.append(f"OpDecorate %{var_id.id} PerPrimitiveEXT")

    def has_attribute(self, node, attribute_name: str) -> bool:
        return any(
            str(getattr(attr, "name", "")).lower() == attribute_name
            for attr in getattr(node, "attributes", []) or []
        )

    def normalized_metadata_name(self, name) -> str:
        normalized = str(name).lower().replace("-", "_")
        if normalized.startswith("glsl_"):
            normalized = normalized[len("glsl_") :]
        return normalized

    def declaration_metadata_names(self, node) -> set:
        names = {
            self.normalized_metadata_name(getattr(attribute, "name", ""))
            for attribute in getattr(node, "attributes", [])
        }
        names.update(
            self.normalized_metadata_name(qualifier)
            for qualifier in getattr(node, "qualifiers", [])
        )
        return names

    def is_mesh_payload_entry_parameter(
        self, param, execution_model: Optional[str]
    ) -> bool:
        """Whether a mesh entry parameter is the task payload (HLSL ``in payload``, CrossGL ``@mesh_payload``)."""
        if execution_model != "MeshEXT":
            return False
        return bool(
            self.declaration_metadata_names(param)
            & {"mesh_payload", "meshpayload", "payload"}
        )

    def process_mesh_payload_entry_parameter(self, param, value_type) -> SpirvId:
        """Declare a mesh entry's payload parameter as the TaskPayloadWorkgroupEXT variable it reads.

        The task stage's ``EmitMeshTasksEXT`` payload reaches each mesh workgroup through TaskPayloadWorkgroupEXT
        storage (SPV_EXT_mesh_shader); a stage ``Input`` holds per-vertex data that no mesh entry receives.
        """
        variable = self.create_variable(
            value_type, "TaskPayloadWorkgroupEXT", getattr(param, "name", "payload")
        )
        self.require_capability("MeshShadingEXT")
        self.require_extension("SPV_EXT_mesh_shader")
        self.mark_function_interface_variable(variable)
        return variable

    def is_task_payload_shared_node(self, node) -> bool:
        return bool(
            self.declaration_metadata_names(node)
            & {"task_payload_shared", "taskpayloadshared", "taskpayloadsharedext"}
        )

    def mesh_output_parameter_role(self, node) -> Optional[str]:
        metadata = self.declaration_metadata_names(node)
        if metadata & {"vertices", "vertex", "mesh_vertices"}:
            return "vertices"
        if metadata & {"indices", "index", "primitive_indices"}:
            return "indices"
        if metadata & {"primitives", "primitive", "mesh_primitives"}:
            return "primitives"
        return None

    def is_mesh_output_parameter(self, node) -> bool:
        if self.mesh_output_parameter_role(node) is None:
            return False
        qualifiers = {str(q).lower() for q in getattr(node, "qualifiers", []) or []}
        return bool(qualifiers & {"out", "inout"})

    def parameter_qualifier_names(self, node) -> Set[str]:
        qualifiers = {
            str(qualifier).lower()
            for qualifier in getattr(node, "qualifiers", []) or []
        }
        qualifiers.update(
            str(getattr(attribute, "name", "")).lower()
            for attribute in getattr(node, "attributes", []) or []
        )
        qualifiers.discard("")
        return qualifiers

    def is_graphics_output_parameter(
        self, node, execution_model: Optional[str], is_entry_point: bool
    ) -> bool:
        if not is_entry_point:
            return False
        if execution_model not in {"Vertex", "Fragment"}:
            return False
        if self.is_mesh_output_parameter(node):
            return False
        return bool(self.parameter_qualifier_names(node) & {"out", "inout"})

    def patch_type_info_from_name(self, type_name: str) -> Optional[dict]:
        base_name, generic_args = generic_type_parts(type_name)
        if base_name not in {"InputPatch", "OutputPatch"}:
            return None

        element_type_name = generic_args[0] if generic_args else "float"
        count_arg = generic_args[1] if len(generic_args) > 1 else None
        control_points = self.patch_control_point_count(count_arg, f"{base_name}<T, N>")
        return {
            "patch_type": base_name,
            "element_type_name": element_type_name,
            "control_points": control_points,
        }

    def patch_control_point_count(self, value, context: str) -> int:
        control_points = self.literal_int_argument(value)
        if control_points is None or control_points <= 0:
            self.refuse_lowering(
                f"SPIR-V tessellation patch type {context} requires a "
                "positive integer control-point count"
            )
            return 1
        return control_points

    def patch_parameter_storage_class(
        self, patch_type: str, execution_model: Optional[str]
    ) -> Optional[str]:
        if execution_model == "TessellationControl":
            return "Input" if patch_type == "InputPatch" else "Output"
        if execution_model == "TessellationEvaluation":
            return "Input"
        return None

    def function_is_entry_point(self, function_node, stage=None) -> bool:
        if stage is not None:
            return getattr(stage, "entry_point", None) is function_node
        qualifier = self.get_function_qualifier(function_node)
        return function_node.name == "main" or qualifier in {
            "vertex",
            "fragment",
            "compute",
            "geometry",
            "tessellation_control",
            "tessellation_evaluation",
            "mesh",
            "task",
            "object",
            "amplification",
            "ray_generation",
            "ray_intersection",
            "ray_closest_hit",
            "ray_miss",
            "ray_any_hit",
            "ray_callable",
        }

    def patch_parameter_info(
        self,
        param,
        execution_model: Optional[str],
        entry_point: bool,
    ) -> Optional[dict]:
        if not entry_point:
            return None

        param_type = getattr(param, "param_type", getattr(param, "vtype", None))
        patch_info = self.patch_type_info_from_name(
            self.type_name_from_value(param_type)
        )
        if patch_info is None:
            return None

        storage_class = self.patch_parameter_storage_class(
            patch_info["patch_type"], execution_model
        )
        param_name = getattr(param, "name", "patch")
        if storage_class is None:
            self.refuse_lowering(
                f"SPIR-V {patch_info['patch_type']} parameter "
                f"{param_name} is only valid in tessellation entry points"
            )
            return None

        return {
            **patch_info,
            "name": param_name,
            "storage_class": storage_class,
        }

    def function_attribute_arguments(self, function_node, attribute_name: str):
        for attr in getattr(function_node, "attributes", []) or []:
            if str(getattr(attr, "name", "")).lower() != attribute_name:
                continue
            arguments = getattr(attr, "arguments", None)
            if arguments is None:
                arguments = getattr(attr, "args", [])
            return list(arguments or [])
        return []

    def tessellation_patch_constant_function_name(self, function_node) -> Optional[str]:
        arguments = self.function_attribute_arguments(
            function_node, "patchconstantfunc"
        )
        if not arguments:
            return None
        if len(arguments) != 1:
            self.refuse_lowering(
                "SPIR-V tessellation_control patchconstantfunc "
                "requires exactly one function name"
            )
            return None

        function_name = self.attribute_value_to_string(arguments[0])
        if not function_name:
            self.refuse_lowering(
                "SPIR-V tessellation_control patchconstantfunc "
                "requires a function name"
            )
            return None
        return function_name

    def matching_patch_interface_variable(self, patch_info: dict) -> Optional[SpirvId]:
        for variable in self.local_variables.values():
            metadata = self.patch_parameter_metadata.get(variable.id)
            if metadata is None:
                continue
            if metadata.get("patch_type") != patch_info.get("patch_type"):
                continue
            if metadata.get("control_points") != patch_info.get("control_points"):
                continue
            if metadata.get("element_type_name") != patch_info.get("element_type_name"):
                continue
            return variable
        return None

    def tessellation_patch_constant_builtin_argument(
        self, param, param_type: SpirvId
    ) -> Optional[SpirvId]:
        semantic = self.semantic_from_node(param)
        normalized = self.normalized_metadata_name(semantic or "")
        builtin_names = {
            "gl_primitiveid": "gl_PrimitiveID",
            "primitiveid": "gl_PrimitiveID",
            "primitive_id": "gl_PrimitiveID",
            "sv_primitiveid": "gl_PrimitiveID",
        }
        builtin_name = builtin_names.get(normalized)
        if builtin_name is None:
            return None

        builtin = self.ensure_stage_builtin(builtin_name)
        if builtin is None:
            return None
        value_type = self.variable_value_types.get(builtin.id) or param_type
        return self.load_from_variable(builtin, value_type)

    def tessellation_patch_constant_call_arguments(self, function_node):
        arguments = []
        for param in getattr(
            function_node, "parameters", getattr(function_node, "params", [])
        ):
            param_type_source = getattr(
                param, "param_type", getattr(param, "vtype", None)
            )
            param_type_name = self.type_name_from_value(param_type_source)
            param_type = self.map_crossgl_type(param_type_source)
            patch_info = self.patch_type_info_from_name(param_type_name)
            param_name = getattr(param, "name", "<anonymous>")

            if patch_info is not None:
                if patch_info["patch_type"] != "InputPatch":
                    self.refuse_lowering(
                        "SPIR-V tessellation_control patchconstantfunc "
                        f"parameter {param_name} must use InputPatch<T, N>"
                    )
                    arguments.append(self.default_value_for_type(param_type))
                    continue

                source_variable = self.matching_patch_interface_variable(patch_info)
                if source_variable is None:
                    self.refuse_lowering(
                        "SPIR-V tessellation_control patchconstantfunc "
                        f"parameter {param_name} has no matching InputPatch "
                        "entry-point parameter"
                    )
                    arguments.append(self.default_value_for_type(param_type))
                    continue

                if self.variable_value_types.get(source_variable.id) is None:
                    arguments.append(self.default_value_for_type(param_type))
                else:
                    arguments.append(source_variable)
                continue

            builtin_argument = self.tessellation_patch_constant_builtin_argument(
                param, param_type
            )
            if builtin_argument is not None:
                arguments.append(builtin_argument)
                continue

            self.refuse_lowering(
                "SPIR-V tessellation_control patchconstantfunc "
                f"parameter {param_name} has no supported SPIR-V source"
            )
            arguments.append(self.default_value_for_type(param_type))
        return arguments

    def tessellation_patch_constant_builtin_name(
        self, semantic: Optional[str]
    ) -> Optional[str]:
        normalized = self.normalized_metadata_name(semantic or "")
        return {
            "gl_tesslevelouter": "gl_TessLevelOuter",
            "tesslevelouter": "gl_TessLevelOuter",
            "sv_tessfactor": "gl_TessLevelOuter",
            "gl_tesslevelinner": "gl_TessLevelInner",
            "tesslevelinner": "gl_TessLevelInner",
            "sv_insidetessfactor": "gl_TessLevelInner",
        }.get(normalized)

    def tessellation_patch_constant_member_components(
        self, value: SpirvId, value_type: SpirvId
    ):
        array_info = self.array_type_info_from_type(value_type)
        if array_info is not None:
            element_type, size = array_info
            for index in range(self.array_length_default_value(size, fallback=0)):
                yield index, self.composite_extract(value, element_type, index)
            return

        vector_info = self.vector_type_info_from_type(value_type)
        if vector_info is not None:
            component_type, count = vector_info
            for index in range(count):
                yield index, self.composite_extract(value, component_type, index)
            return

        yield 0, value

    def store_tessellation_patch_constant_builtin_component(
        self, builtin_name: str, component_index: int, value: SpirvId
    ):
        builtin_size = self.tessellation_patch_builtin_size(builtin_name)
        if builtin_size is None or component_index >= builtin_size:
            self.refuse_lowering(
                "SPIR-V tessellation_control patchconstantfunc "
                f"{builtin_name} component {component_index} is out of range"
            )
            return

        builtin = self.ensure_stage_builtin(builtin_name)
        if builtin is None:
            return

        int_type = self.primitive_types["int"]
        float_type = self.primitive_types["float"]
        index_id = self.register_constant(component_index, int_type)
        ptr_type = self.register_pointer_type(float_type, "Output")
        access = self.access_chain(builtin, [index_id], ptr_type)
        self.variable_value_types[access.id] = float_type
        self.store_to_variable(access, self.convert_value_to_type(value, float_type))

    def tessellation_patch_constant_user_member_key(
        self, member_name: str, metadata: dict
    ) -> Optional[str]:
        semantic = metadata.get("semantic")
        if self.tessellation_patch_constant_builtin_name(semantic) is not None:
            return None

        node = metadata.get("node")
        normalized = self.normalized_metadata_name(semantic or "")
        patch_markers = {
            "patch",
            "perpatch",
            "per_patch",
            "patchconstant",
            "patch_constant",
        }
        if normalized in patch_markers:
            return member_name
        if semantic is not None:
            return member_name
        if node is not None and self.explicit_location_attribute(node) is not None:
            return member_name
        return None

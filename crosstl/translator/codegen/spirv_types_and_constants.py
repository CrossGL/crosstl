"""Types, constants and variables of the SPIR-V generator: target profiles, id allocation and emission, capability and extension requirements, non-uniform values, registration of scalar, vector, matrix, pointer, image, resource, struct, function and ray query types, constants and specialization constants, image offset operands, and variable creation, loads and stores."""

import re
from typing import (
    List,
    Optional,
    Tuple,
    Union,
)

from ..ast import (
    AssignmentNode,
    BinaryOpNode,
    LiteralNode,
    UnaryOpNode,
    VariableNode,
)
from ..refusals import VulkanRefusal
from .array_utils import evaluate_literal_int_expression
from .spirv_constant_folding import spirv_evaluated_scalar_initializer
from .spirv_image_texels import SHADER_IMAGE_FORMATS
from .spirv_placed_blocks import matrix_from_column_array
from .spirv_refusals import SpirvRefusals
from .spirv_specialization_conversions import specialization_conversion_expression
from .spirv_standard_constants import spirv_non_finite_float_literal
from .spirv_values import (
    SpirvId,
    SpirvType,
    UnsupportedSPIRVFeatureError,
)


class SpirvTypeConstantMixin:
    """Type constant members of ``VulkanSPIRVCodeGen``."""

    def normalize_target_profile(self, target_profile: Optional[str]) -> Optional[str]:
        if target_profile is None:
            return None
        normalized = str(target_profile).strip().lower().replace("_", "-")
        if normalized in {"", "vulkan", "spirv", "spv"}:
            return None
        if normalized == self.COOPERATIVE_MATRIX_KHR_PROFILE:
            return normalized
        raise VulkanRefusal(
            "Vulkan target profile must be " f"'{self.COOPERATIVE_MATRIX_KHR_PROFILE}'"
        )

    def reset_generation_state(self):
        """Reset per-module SPIR-V ids, declarations, and symbol caches."""
        self.next_id = 1
        self.code_lines = []
        self.refusals = SpirvRefusals()
        self.decorations = []
        self.required_extensions = set()
        self.source_extensions = set()
        self.placed_uniform_decorations = {}
        self.placed_uniform_structs = set()

        self.primitive_types = {}
        self.vector_types = {}
        self.matrix_types = {}
        self.cooperative_matrix_types = {}
        self.cooperative_matrix_type_metadata = {}
        self.cooperative_matrix_containing_type_ids = set()
        self.struct_types = {}
        self.pointer_types = {}
        self.function_types = {}
        self.array_types = {}
        self.layout_array_types = {}
        self.layout_struct_types = {}
        self.layout_struct_source_types = {}
        self.std430_runtime_array_struct_types = set()
        self.resource_types = {}
        self.resource_image_types = {}
        # OpTypeSampledImage id per image type id: combined samplers and texture+sampler pairs share one type.
        self.sampled_image_type_ids = {}
        self.ray_query_types = {}
        self.ray_tracing_storage_variables = {}
        self.enum_type_names = set()
        self.enum_variant_values = {}
        self.enum_struct_type_names = set()
        self.enum_struct_fields = {}
        self.enum_struct_variant_fields = {}
        self.generic_enum_struct_definitions = {}
        self.generic_enum_specializations = {}
        self.struct_declarations = {}
        self.enum_declarations = {}
        self.struct_registration_stack = set()
        self.enum_struct_registration_stack = set()
        self.glsl_buffer_block_type_names = set()

        self.required_capabilities = set()
        self.geometry_streams_used = False
        self.geometry_output_streams = {}
        self.global_variables = {}
        # (ray tracing storage class, location) -> variable names, for payloads named by location.
        self.ray_storage_locations = {}
        self.stage_global_variables = {}
        self.local_variables = {}
        self.named_constants = {}
        self.named_constant_debug_ids = set()
        self.resource_alias_variables = set()
        self.variable_value_types = {}
        # Results of OpAccessChain: pointers that are not memory object declarations.
        self.access_chain_ids = set()
        self.value_types = {}
        self.constants = {}
        self.constant_values_by_id = {}
        self.literal_int_constants = {}
        self.current_function_literal_int_constants = {}
        self.current_function_literal_int_constant_shadows = set()
        self.literal_scalar_constants = {}
        self.vector_constants = {}
        self.composite_constants = {}
        self.specialization_constants = {}
        self.specialization_constant_result_ids = set()
        self.specialization_constant_default_values = {}
        self.undefined_values = {}
        self.resource_type_metadata = {}
        self.structured_buffer_metadata = {}
        self.storage_buffer_access_metadata = {}
        self.storage_buffer_pointer_offset_aliases = {}
        self.storage_pointer_reinterpretations = {}
        self.pointer_offset_reset_marker_count = 0
        self.source_variable_type_names = {}
        self.uniform_block_wrapped_variables = {}
        self.precise_global_variables = set()
        self.precise_local_variables = set()
        # Variables of the current function whose values flow into a precise variable (precise_dataflow.py).
        self.precise_flow_names = frozenset()
        self.no_contraction_ids = set()
        self.precise_expression_depth = 0
        self.non_uniform_ids = set()

        self.functions = {}
        self.function_nodes = {}
        self.function_nodes_by_name = {}
        self.function_signatures = {}
        self.function_overloads = {}
        self.stage_local_functions = {}
        self.stage_local_function_signatures = {}
        self.stage_local_function_overloads = {}
        self.stage_local_function_parameter_names = {}
        self.stage_local_function_image_access_requirements = {}
        self.stage_local_function_storage_buffer_access_requirements = {}
        self.function_parameter_names = {}
        self.function_parameter_names_by_id = {}
        self.function_image_access_requirements = {}
        self.function_storage_buffer_access_requirements = {}
        self.function_storage_buffer_access_requirements_by_id = {}
        self.inline_storage_buffer_functions = {}
        self.stage_local_inline_storage_buffer_functions = {}
        self.inline_storage_buffer_call_stack = []
        self.inline_return_context_stack = []
        self.inline_return_presence_by_node_id = {}
        self.generic_function_definitions = {}
        self.generic_function_definition_ordinals = {}
        self.generic_function_specializations = {}
        self.generic_function_specialized_names = {}
        self.current_generic_function_substitutions = {}
        self.local_variable_types = {}
        self.spirv_skipped_function_parameter_indices = {}
        self.spirv_skipped_function_parameter_indices_by_id = {}
        self.function_stage_input_dependencies = {}
        self.function_stage_output_dependencies = {}
        self.function_resource_array_params = {}
        self.stage_local_function_resource_array_params = {}
        self.function_value_array_parameter_indices_by_id = {}
        self.function_resource_array_type_hints = {}
        self.stage_local_function_resource_array_type_hints = {}
        self.function_storage_image_pointer_params = {}
        self.stage_local_function_storage_image_pointer_params = {}
        self.function_execution_models = {}
        self.current_execution_model = None
        self.current_function_node = None
        self.current_function_name = None
        self.current_function_id = None
        self.current_stage = None
        self.module_compute_entry_config = None
        # Per-entry workgroup sizes (spirv_entry_workgroup_sizes); planned before any function is generated.
        self.module_has_one_workgroup_size = True
        self.entry_workgroup_size_metadata = {}
        self.entry_workgroup_size_values = {}
        self.host_workgroup_size_spec_ids = None
        self.used_workgroup_specialization_ids = set()
        self.uses_local_size_id = False
        self.current_return_type = None
        self.current_return_type_source = None
        self.current_return_semantic_output = None
        self.current_entry_point_return_outputs = None
        self.current_expression_expected_type = None
        self.current_generic_type_substitutions = {}
        self.current_cooperative_matrix_role_overrides = {}
        self.cooperative_matrix_selection_depth = 0
        self.cooperative_matrix_loop_control_depth = 0
        self.cooperative_matrix_control_flow_diverged = False
        self.uses_device_scope_memory_operation = False
        self.mesh_output_counts_by_function = {}
        self.mesh_vertex_output_variable = None
        self.mesh_vertex_output_limit = None
        self.mesh_primitive_index_outputs = {}
        self.current_mesh_output_parameters = {}
        self.function_mesh_output_parameter_indices = {}
        self.stage_local_function_mesh_output_parameter_indices = {}
        self.function_interface_variables = {}
        self.function_interface_variables_by_name = {}
        self.interface_original_value_types = {}
        self.builtin_interface_variable_ids = set()
        self.builtin_names_by_variable_id = {}
        self.readonly_builtin_pointer_names = {}
        self.readonly_pointer_names = {}
        self.patch_parameter_metadata = {}
        self.tessellation_output_patch_locations = {}
        self.tessellation_patch_constant_interfaces = {}
        self.tessellation_patch_constant_output_variables = {}
        self.tessellation_patch_constant_input_variables = {}
        self.fragment_depth_replacing_function_ids = set()
        self.fragment_stencil_ref_replacing_function_ids = set()
        self.fragment_depth_layouts = {}
        self.mesh_output_member_variables = {}
        self.mesh_output_member_shadow_variables = {}
        self.mesh_output_member_locations = {}
        self.struct_member_metadata = {}
        self.task_payload_shared_variables = {}
        self.task_payload_interface_by_function = {}
        self.workgroup_variables_by_declaration = {}
        self.entry_point_private_variables = []
        self.local_size_warning_keys = set()
        self.tessellation_control_stage = None

        self.glsl_std450_id = None
        self.main_fn_id = None
        self.requires_compute_derivatives = False

        self.current_label = None
        self.loop_merge_labels = []
        self.loop_continue_labels = []
        self.defined_functions = set()
        self.current_struct_members = {}
        self.cbuffer_variables = {}
        self.cbuffer_members = {}
        self.entry_cbuffer_members = {}
        self.push_constant_block_name = None

        self.inputs = []
        self.outputs = []
        self.uniform_buffers = []
        self.next_input_location = 0
        self.next_output_location = 0
        self.interface_location_counters = {}
        self.entry_input_locations = {}
        self.used_input_locations = set()
        self.used_output_locations = set()
        self.next_resource_binding = 0
        self.next_resource_bindings = {}
        self.reserved_resource_bindings = set()
        self.used_resource_bindings = set()
        # The (combined-sampler half, node id) claims of each exactly bound slot, and the ids of the nodes that
        # share a slot as the image and sampler halves of one combined image sampler.
        self.binding_occupants = {}
        self.shared_binding_nodes = set()
        # Descriptor slots of resources bound by HLSL register or vk::binding, by node id.
        self.hlsl_descriptor_slots = {}
        # Descriptor slots of the current entry point's buffer/texture/sampler parameters, by parameter name.
        self.entry_descriptor_slots = {}
        # Uniform-block arrays of matrix columns, by array type id, with the matrix type a load rebuilds.
        self.matrix_column_arrays = {}
        self.entry_point_resource_binding_baseline = None

        self.is_vertex_shader = False
        self.bound_id = 0

    def get_id(self) -> int:
        """Get the next available SPIR-V ID."""
        id_value = self.next_id
        self.next_id += 1
        if id_value > self.bound_id:
            self.bound_id = id_value
        return id_value

    def emit(self, instruction: str):
        """Add a SPIR-V instruction to the code."""
        self.code_lines.append(instruction)

    def require_capability(self, capability: str):
        """Request a SPIR-V capability for instructions emitted later."""
        if capability != "Shader":
            self.required_capabilities.add(capability)

    def require_extension(self, extension: str):
        """Request a SPIR-V extension for instructions emitted later."""
        self.required_extensions.add(extension)

    def require_source_extension(self, extension: str):
        """Record a source-language extension the module's layout relies on (``OpSourceExtension``)."""
        self.source_extensions.add(extension)

    def require_non_uniform_descriptor_indexing(self):
        self.require_capability("ShaderNonUniform")
        self.require_extension("SPV_EXT_descriptor_indexing")

    def is_non_uniform_value(self, value_id) -> bool:
        id_number = value_id.id if isinstance(value_id, SpirvId) else value_id
        return id_number in self.non_uniform_ids

    def mark_non_uniform_result(self, value_id):
        id_number = value_id.id if isinstance(value_id, SpirvId) else value_id
        if id_number in self.non_uniform_ids:
            return
        self.require_non_uniform_descriptor_indexing()
        self.non_uniform_ids.add(id_number)
        self.decorations.append(f"OpDecorate %{id_number} NonUniform")

    def require_compute_derivatives(self):
        """Enable compute shader derivatives for derivative-dependent image ops."""
        self.requires_compute_derivatives = True
        # The entry's workgroup size picks the derivative group and its capability (emit_entry_point).
        self.require_extension("SPV_KHR_compute_shader_derivatives")

    def register_primitive_type(self, name: str) -> SpirvId:
        """Create and register a primitive type."""
        name = self.normalize_primitive_name(name)
        if name in self.primitive_types:
            return self.primitive_types[name]

        id_value = self.get_id()
        if name == "void":
            self.emit(f"%{id_value} = OpTypeVoid")
        elif name == "bool":
            self.emit(f"%{id_value} = OpTypeBool")
        elif name == "float":
            self.emit(f"%{id_value} = OpTypeFloat 32")
        elif name == "double":
            self.require_capability("Float64")
            self.emit(f"%{id_value} = OpTypeFloat 64")
        elif name == "int":
            self.emit(f"%{id_value} = OpTypeInt 32 1")
        elif name == "uint":
            self.emit(f"%{id_value} = OpTypeInt 32 0")
        elif name == "i64":
            self.require_capability("Int64")
            self.emit(f"%{id_value} = OpTypeInt 64 1")
        elif name == "u64":
            self.require_capability("Int64")
            self.emit(f"%{id_value} = OpTypeInt 64 0")
        elif name == self.BFLOAT16_TYPE_NAME:
            self.require_capability("Int16")
            self.emit(f"%{id_value} = OpTypeInt 16 0")

        spirv_type = SpirvType(name)
        spirv_id = SpirvId(id_value, spirv_type, name)
        self.primitive_types[name] = spirv_id
        return spirv_id

    def register_vector_type(self, component_type: SpirvId, count: int) -> SpirvId:
        """Create and register a vector type."""
        key = (component_type.id, count)
        if key in self.vector_types:
            return self.vector_types[key]

        id_value = self.get_id()
        self.emit(f"%{id_value} = OpTypeVector %{component_type.id} {count}")

        type_name = f"v{count}{component_type.type.base_type}"
        spirv_type = SpirvType(type_name)
        spirv_id = SpirvId(id_value, spirv_type, type_name)
        self.vector_types[key] = spirv_id
        return spirv_id

    def register_matrix_type(self, column_type: SpirvId, count: int) -> SpirvId:
        """Create and register a matrix type."""
        key = (column_type.id, count)
        if key in self.matrix_types:
            return self.matrix_types[key]

        id_value = self.get_id()
        self.emit(f"%{id_value} = OpTypeMatrix %{column_type.id} {count}")

        column_info = self.vector_component_type_and_count(column_type.type.base_type)
        if column_info is not None:
            component_type, row_count = column_info
            prefix = "dmat" if component_type == "double" else "mat"
            type_name = f"{prefix}{count}x{row_count}"
        else:
            type_name = f"mat{count}x{column_type.type.base_type[1]}"
        spirv_type = SpirvType(type_name)
        spirv_id = SpirvId(id_value, spirv_type, type_name)
        self.matrix_types[key] = spirv_id
        return spirv_id

    def register_pointer_type(
        self, pointed_type: SpirvId, storage_class: str
    ) -> SpirvId:
        """Create and register a pointer type."""
        if (
            pointed_type.id in self.cooperative_matrix_containing_type_ids
            and storage_class not in {"Function", "Private"}
        ):
            raise UnsupportedSPIRVFeatureError(
                "cooperative-matrix-storage-class",
                "SPIR-V KHR cooperative-matrix values and containing types can "
                "only be allocated in Function or Private storage; got "
                f"{storage_class}",
                missing_capabilities=("spirv.cooperative_matrix.storage-class",),
            )
        key = (pointed_type.id, storage_class)
        if key in self.pointer_types:
            return self.pointer_types[key]

        id_value = self.get_id()
        self.emit(f"%{id_value} = OpTypePointer {storage_class} %{pointed_type.id}")

        spirv_type = SpirvType(f"ptr_{pointed_type.type.base_type}", storage_class)
        spirv_id = SpirvId(id_value, spirv_type)
        self.pointer_types[key] = spirv_id
        return spirv_id

    def register_image_type(
        self,
        type_name: str,
        component_type: SpirvId,
        dim: str,
        depth: int,
        arrayed: int,
        multisampled: int,
        sampled: int,
        image_format: str = "Unknown",
    ) -> SpirvId:
        key = (
            component_type.id,
            dim,
            depth,
            arrayed,
            multisampled,
            sampled,
            image_format,
        )
        if key in self.resource_image_types:
            return self.resource_image_types[key]

        if sampled == 2 and multisampled:
            self.require_capability("StorageImageMultisample")
            if arrayed:
                self.require_capability("ImageMSArray")
        if dim == "Cube" and arrayed:
            if sampled == 2:
                self.require_capability("ImageCubeArray")
            else:
                self.require_capability("SampledCubeArray")
        if dim == "1D":
            if sampled == 2:
                self.require_capability("Image1D")
            else:
                self.require_capability("Sampled1D")
        if dim == "SubpassData":
            self.require_capability("InputAttachment")
        if dim == "Buffer":
            self.require_capability("ImageBuffer" if sampled == 2 else "SampledBuffer")
        if image_format not in SHADER_IMAGE_FORMATS:
            # SPIR-V 3.11: every other Image Format operand needs this capability.
            self.require_capability("StorageImageExtendedFormats")

        id_value = self.get_id()
        self.emit(
            f"%{id_value} = OpTypeImage %{component_type.id} {dim} "
            f"{depth} {arrayed} {multisampled} {sampled} {image_format}"
        )

        spirv_type = SpirvType(type_name)
        spirv_id = SpirvId(id_value, spirv_type, type_name)
        self.resource_image_types[key] = spirv_id
        return spirv_id

    def register_resource_type(
        self, type_name: str, image_format: Optional[str] = None
    ) -> SpirvId:
        if image_format is None and type_name in self.resource_types:
            return self.resource_types[type_name]

        info = self.resource_type_info(type_name)
        if info is None:
            raise VulkanRefusal(f"Unknown SPIR-V resource type {type_name}")

        info = dict(info)
        source_format = None
        if info["kind"] == "storage_image" and image_format:
            spirv_format = self.spirv_image_format_name(image_format)
            if spirv_format:
                source_format = str(image_format).lower()
                info["format"] = spirv_format
                info["component_type"] = self.image_format_component_type(image_format)

        cache_key = (
            type_name,
            info.get("kind"),
            info.get("component_type"),
            info.get("format"),
        )
        if info["kind"] == "sampler":
            # SPIR-V has one sampler type: a comparison_sampler differs only in the
            # descriptor's compare state, and OpTypeSampler must not be declared twice.
            cache_key = ("sampler", "sampler", None, None)
        if cache_key in self.resource_types:
            spirv_id = self.resource_types[cache_key]
            if info["kind"] == "sampler":
                self.resource_types.setdefault(type_name, spirv_id)
            return spirv_id

        if info["kind"] == "acceleration_structure":
            return self.register_acceleration_structure_type(type_name)

        if info["kind"] == "sampler":
            id_value = self.get_id()
            self.emit(f"%{id_value} = OpTypeSampler")
            spirv_id = SpirvId(id_value, SpirvType(type_name), type_name)
        else:
            component_type = self.register_primitive_type(info["component_type"])
            image_type = self.register_image_type(
                f"{type_name}_image",
                component_type,
                info["dim"],
                info["depth"],
                info["arrayed"],
                info["multisampled"],
                info["sampled"],
                info["format"],
            )

            if info["kind"] == "sampled_image":
                id_value = self.sampled_image_type_id(image_type)
                spirv_id = SpirvId(id_value, SpirvType(type_name), type_name)
            else:
                spirv_id = image_type
                spirv_id.type = SpirvType(type_name)
                spirv_id.name = type_name

            metadata = dict(info)
            metadata["type_name"] = type_name
            metadata["source_format"] = source_format
            metadata["image_type_id"] = image_type.id
            metadata["component_count"] = self.image_format_component_count(
                source_format
            )
            self.resource_type_metadata[image_type.id] = metadata

        if info["kind"] == "sampler":
            self.resource_type_metadata[spirv_id.id] = {
                "kind": "sampler",
                "type_name": type_name,
                "component_type": "float",
                "component_count": 0,
            }
        else:
            self.resource_type_metadata[spirv_id.id] = metadata

        self.resource_types[cache_key] = spirv_id
        if image_format is None:
            self.resource_types[type_name] = spirv_id
        return spirv_id

    def register_struct_type(
        self, name: str, members: List[Tuple[SpirvId, str]]
    ) -> SpirvId:
        """Create and register a struct type."""
        if name in self.struct_types:
            return self.struct_types[name]

        id_value = self.get_id()

        member_types = " ".join([f"%{member[0].id}" for member in members])
        self.emit(f"%{id_value} = OpTypeStruct {member_types}")

        self.emit(f'OpName %{id_value} "{name}"')

        for i, (_, member_name) in enumerate(members):
            self.emit(f'OpMemberName %{id_value} {i} "{member_name}"')

        spirv_type = SpirvType(name)
        spirv_id = SpirvId(id_value, spirv_type, name)
        self.struct_types[name] = spirv_id

        self.current_struct_members[name] = members
        if any(
            member_type.id in self.cooperative_matrix_containing_type_ids
            for member_type, _member_name in members
        ):
            self.cooperative_matrix_containing_type_ids.add(spirv_id.id)

        return spirv_id

    def struct_members_match(
        self, name: str, members: List[Tuple[SpirvId, str]]
    ) -> bool:
        existing_members = self.current_struct_members.get(name)
        if existing_members is None or len(existing_members) != len(members):
            return False
        return all(
            existing_type.id == member_type.id and existing_name == member_name
            for (existing_type, existing_name), (member_type, member_name) in zip(
                existing_members, members
            )
        )

    def unique_struct_type_name(
        self, preferred_name: str, members: List[Tuple[SpirvId, str]]
    ) -> str:
        if preferred_name not in self.struct_types or self.struct_members_match(
            preferred_name, members
        ):
            return preferred_name

        suffix = 1
        while True:
            candidate = f"{preferred_name}_{suffix}"
            if candidate not in self.struct_types or self.struct_members_match(
                candidate, members
            ):
                return candidate
            suffix += 1

    def register_layout_struct_type(
        self,
        source_type: SpirvId,
        layout: str,
        members: List[Tuple[SpirvId, str]],
    ) -> SpirvId:
        """Create a layout-specific struct clone for block-layout decorations."""
        key = (source_type.id, layout)
        if key in self.layout_struct_types:
            return self.layout_struct_types[key]

        id_value = self.get_id()
        member_types = " ".join([f"%{member[0].id}" for member in members])
        self.emit(f"%{id_value} = OpTypeStruct {member_types}")

        type_name = f"{source_type.type.base_type}_{layout}_{source_type.id}"
        self.emit(f'OpName %{id_value} "{type_name}"')
        for i, (_, member_name) in enumerate(members):
            self.emit(f'OpMemberName %{id_value} {i} "{member_name}"')

        spirv_id = SpirvId(id_value, SpirvType(type_name), type_name)
        self.layout_struct_types[key] = spirv_id
        self.layout_struct_source_types[id_value] = source_type
        self.current_struct_members[type_name] = members
        if any(
            member_type.id in self.cooperative_matrix_containing_type_ids
            for member_type, _member_name in members
        ):
            self.cooperative_matrix_containing_type_ids.add(spirv_id.id)
        return spirv_id

    def register_function_type(
        self, return_type: SpirvId, param_types: List[SpirvId]
    ) -> SpirvId:
        """Create and register a function type."""
        key = (return_type.id, tuple(p.id for p in param_types))
        if key in self.function_types:
            return self.function_types[key]

        id_value = self.get_id()

        params = " ".join([f"%{param.id}" for param in param_types])
        if params:
            self.emit(f"%{id_value} = OpTypeFunction %{return_type.id} {params}")
        else:
            self.emit(f"%{id_value} = OpTypeFunction %{return_type.id}")

        spirv_type = SpirvType(f"fn_{return_type.type.base_type}")
        spirv_id = SpirvId(id_value, spirv_type)
        self.function_types[key] = spirv_id
        return spirv_id

    def register_ray_query_type(self, type_name: str = "RayQuery") -> SpirvId:
        """Create and register the opaque SPIR-V ray-query type."""
        cache_key = re.sub(r"\s+", "", str(type_name))
        if cache_key in self.ray_query_types:
            return self.ray_query_types[cache_key]

        self.require_capability("RayQueryKHR")
        self.require_extension("SPV_KHR_ray_query")
        # One opaque type serves every RayQuery<FLAGS>: SPIR-V allows a single
        # OpTypeRayQueryKHR, and the flags are applied at initialization.
        opaque = next(iter(self.ray_query_types.values()), None)
        if opaque is None:
            id_value = self.get_id()
            self.emit(f"%{id_value} = OpTypeRayQueryKHR")
        else:
            id_value = opaque.id

        spirv_id = SpirvId(id_value, SpirvType(cache_key), cache_key)
        self.ray_query_types[cache_key] = spirv_id
        return spirv_id

    def register_acceleration_structure_type(self, type_name: str) -> SpirvId:
        """Create and register the opaque SPIR-V acceleration-structure type."""
        cache_key = (str(type_name), "acceleration_structure")
        if cache_key in self.resource_types:
            return self.resource_types[cache_key]

        self.require_capability("RayQueryKHR")
        self.require_extension("SPV_KHR_ray_query")
        id_value = self.get_id()
        self.emit(f"%{id_value} = OpTypeAccelerationStructureKHR")

        spirv_id = SpirvId(id_value, SpirvType(str(type_name)), str(type_name))
        metadata = {
            "kind": "acceleration_structure",
            "type_name": str(type_name),
        }
        self.resource_type_metadata[spirv_id.id] = metadata
        self.resource_types[cache_key] = spirv_id
        self.resource_types[str(type_name)] = spirv_id
        return spirv_id

    def register_constant(
        self, value: Union[bool, int, float], type_id: SpirvId
    ) -> SpirvId:
        """Create and register a constant value."""
        type_name = self.normalize_primitive_name(type_id.type.base_type)
        if type_name == self.BFLOAT16_TYPE_NAME and isinstance(value, float):
            value = self.bfloat16_bits_for_float(value)
        key = (value, type_id.id)
        if isinstance(value, float) and value == 0.0 and str(value).startswith("-"):
            # -0.0 == 0.0 in Python, but they are different constants.
            key = ("-0.0", type_id.id)
        if key in self.constants:
            return self.constants[key]

        id_value = self.get_id()

        type_name = type_id.type.base_type
        if type_name == "bool":
            opcode = "OpConstantTrue" if value else "OpConstantFalse"
            self.emit(f"%{id_value} = {opcode} %{type_id.id}")
        else:
            constant_value = self.spirv_constant_literal_text(value, type_name)
            self.emit(f"%{id_value} = OpConstant %{type_id.id} {constant_value}")

        spirv_id = SpirvId(id_value, type_id.type, f"{type_name}_{value}")
        self.value_types[id_value] = type_id
        self.constants[key] = spirv_id
        self.constant_values_by_id[id_value] = value
        return spirv_id

    def spirv_constant_literal_text(
        self, value: Union[bool, int, float], type_name: str
    ) -> str:
        """Return a SPIR-V assembly literal valid for the result type."""
        type_name = self.normalize_primitive_name(type_name)
        non_finite = spirv_non_finite_float_literal(value, type_name)
        if non_finite is not None:
            return non_finite
        if not isinstance(value, int) or isinstance(value, bool):
            return str(value)

        if (
            type_name == "int"
            and self.SIGNED_INT32_MAX < value <= self.UNSIGNED_INT32_MAX
        ):
            return str(value - (1 << 32))
        if (
            type_name == "i64"
            and self.SIGNED_INT64_MAX < value <= self.UNSIGNED_INT64_MAX
        ):
            return str(value - (1 << 64))
        return str(value)

    def integer_literal_type_for_value(
        self, primitive_type_name: str, value: int
    ) -> str:
        """Return a SPIR-V scalar integer type that can encode a literal value."""
        primitive_type_name = self.normalize_primitive_name(primitive_type_name)
        if primitive_type_name == "int":
            if self.SIGNED_INT32_MIN <= value <= self.SIGNED_INT32_MAX:
                return "int"
            if 0 <= value <= self.UNSIGNED_INT32_MAX:
                return "uint"
            if 0 <= value <= self.UNSIGNED_INT64_MAX:
                return "u64"
            if self.SIGNED_INT64_MIN <= value <= self.SIGNED_INT64_MAX:
                return "i64"
        elif primitive_type_name == "uint":
            if 0 <= value <= self.UNSIGNED_INT32_MAX:
                return "uint"
            if 0 <= value <= self.UNSIGNED_INT64_MAX:
                return "u64"
        elif primitive_type_name == "i64":
            if self.SIGNED_INT64_MIN <= value <= self.SIGNED_INT64_MAX:
                return "i64"
            if 0 <= value <= self.UNSIGNED_INT64_MAX:
                return "u64"
        return primitive_type_name

    def register_specialization_constant(
        self, value: Union[bool, int, float], type_id: SpirvId, spec_id: int
    ) -> SpirvId:
        """Create and register a scalar specialization constant."""
        key = (value, type_id.id, spec_id)
        if key in self.specialization_constants:
            return self.specialization_constants[key]

        id_value = self.get_id()
        type_name = type_id.type.base_type
        if type_name == "bool":
            opcode = "OpSpecConstantTrue" if value else "OpSpecConstantFalse"
            self.emit(f"%{id_value} = {opcode} %{type_id.id}")
        else:
            self.emit(f"%{id_value} = OpSpecConstant %{type_id.id} {value}")

        self.decorations.append(f"OpDecorate %{id_value} SpecId {spec_id}")
        spirv_id = SpirvId(id_value, type_id.type, f"{type_name}_{value}")
        self.value_types[id_value] = type_id
        self.specialization_constants[key] = spirv_id
        self.specialization_constant_result_ids.add(id_value)
        self.specialization_constant_default_values[id_value] = value
        return spirv_id

    def is_specialization_constant_value(self, value_id) -> bool:
        """Return whether an id is a specialization constant or spec-op result."""
        id_number = value_id.id if isinstance(value_id, SpirvId) else value_id
        return id_number in self.specialization_constant_result_ids

    def emit_spec_constant_op(
        self, opcode: str, result_type: SpirvId, operands: List[SpirvId]
    ) -> SpirvId:
        """Emit an OpSpecConstantOp specialization-constant operation.

        Module/constant-scope arithmetic whose operands are specialization
        constants must be encoded as an OpSpecConstantOp (a constant instruction)
        rather than an ordinary arithmetic instruction, which SPIR-V only permits
        inside a function block. The result is tracked as a specialization-constant
        result so nested expressions chain into further OpSpecConstantOp encodings.
        """
        id_value = self.get_id()
        operand_list = " ".join(f"%{operand.id}" for operand in operands)
        self.emit(
            f"%{id_value} = OpSpecConstantOp %{result_type.id} {opcode} {operand_list}"
        )
        spirv_id = SpirvId(id_value, result_type.type)
        self.value_types[id_value] = result_type
        self.specialization_constant_result_ids.add(id_value)
        return spirv_id

    def specialization_constant_binary_opcode(
        self, operator: str, component_type: str
    ) -> Optional[str]:
        """Return the OpSpecConstantOp opcode for an integer/bool binary operator.

        Only integer and boolean operations are valid specialization-constant
        operations under the Shader capability; floating-point spec-constant ops
        require the Kernel capability and are intentionally left unhandled here.
        """
        is_integer = component_type in self.INTEGER_TYPE_NAMES
        is_unsigned = component_type in self.UNSIGNED_INTEGER_TYPES
        is_signed = component_type in self.SIGNED_INTEGER_TYPES

        if is_integer:
            arithmetic = {"+": "IAdd", "-": "ISub", "*": "IMul", "MULTIPLY": "IMul"}
            if operator in arithmetic:
                return arithmetic[operator]
            if operator == "/":
                return "UDiv" if is_unsigned else "SDiv"
            if operator == "%":
                return "UMod" if is_unsigned else "SRem"
            if operator in {"&", "|", "^"}:
                return {
                    "&": "BitwiseAnd",
                    "|": "BitwiseOr",
                    "^": "BitwiseXor",
                }[operator]
            if operator == "<<":
                return "ShiftLeftLogical"
            if operator == ">>":
                return "ShiftRightArithmetic" if is_signed else "ShiftRightLogical"
            return {
                "==": "IEqual",
                "!=": "INotEqual",
                "<": "ULessThan" if is_unsigned else "SLessThan",
                ">": "UGreaterThan" if is_unsigned else "SGreaterThan",
                "<=": "ULessThanEqual" if is_unsigned else "SLessThanEqual",
                ">=": "UGreaterThanEqual" if is_unsigned else "SGreaterThanEqual",
            }.get(operator)

        if component_type == "bool":
            return {
                "&&": "LogicalAnd",
                "&": "LogicalAnd",
                "||": "LogicalOr",
                "|": "LogicalOr",
                "==": "LogicalEqual",
                "!=": "LogicalNotEqual",
                "^": "LogicalNotEqual",
            }.get(operator)

        return None

    def emit_specialization_constant_binary_operation(
        self, operator: str, left: SpirvId, right: SpirvId
    ) -> Optional[SpirvId]:
        """Lower a constant binary op with spec-constant operands to OpSpecConstantOp."""
        left_type = self.registered_value_type(left) or self.ensure_registered_type(
            left.type
        )
        right_type = self.registered_value_type(right) or self.ensure_registered_type(
            right.type
        )
        if (
            self.vector_component_type_and_count(left_type.type.base_type) is not None
            or self.vector_component_type_and_count(right_type.type.base_type)
            is not None
        ):
            return None

        if operator in {"==", "!=", "<", ">", "<=", ">="}:
            operand_component = self.promoted_numeric_type_name(
                [left_type.type.base_type, right_type.type.base_type]
            )
            if operand_component is None:
                return None
            opcode = self.specialization_constant_binary_opcode(
                operator, operand_component
            )
            if opcode is None:
                return None
            operand_type = self.register_primitive_type(operand_component)
            left = self.coerce_scalar_constant_to_type(left, operand_type)
            right = self.coerce_scalar_constant_to_type(right, operand_type)
            result_type = self.register_primitive_type("bool")
            return self.emit_spec_constant_op(opcode, result_type, [left, right])

        result_type = self.binary_expression_result_type(
            operator, left_type, right_type
        )
        if result_type is None:
            return None
        component_type = self.normalize_primitive_name(result_type.type.base_type)
        opcode = self.specialization_constant_binary_opcode(operator, component_type)
        if opcode is None:
            return None

        left = self.coerce_scalar_constant_to_type(left, result_type)
        if operator not in {"<<", ">>"}:
            right = self.coerce_scalar_constant_to_type(right, result_type)
        return self.emit_spec_constant_op(opcode, result_type, [left, right])

    def emit_specialization_constant_unary_operation(
        self, operator: str, operand: SpirvId
    ) -> Optional[SpirvId]:
        """Lower a constant unary op with a spec-constant operand to OpSpecConstantOp."""
        operand_type = self.registered_value_type(
            operand
        ) or self.ensure_registered_type(operand.type)
        if (
            self.vector_component_type_and_count(operand_type.type.base_type)
            is not None
        ):
            return None
        component_type = self.normalize_primitive_name(operand_type.type.base_type)

        if operator == "+":
            return operand
        if operator == "-":
            if component_type in self.SIGNED_INTEGER_TYPES:
                return self.emit_spec_constant_op("SNegate", operand_type, [operand])
            if component_type in self.UNSIGNED_INTEGER_TYPES:
                zero = self.register_constant(0, operand_type)
                return self.emit_spec_constant_op("ISub", operand_type, [zero, operand])
            return None
        if operator == "~" and component_type in self.INTEGER_TYPE_NAMES:
            return self.emit_spec_constant_op("Not", operand_type, [operand])
        if operator == "!" and component_type == "bool":
            return self.emit_spec_constant_op("LogicalNot", operand_type, [operand])
        return None

    def process_specialization_constant_operation(
        self, expr, target_type: Optional[SpirvId] = None
    ) -> Optional[SpirvId]:
        """Emit OpSpecConstantOp for module-scope ops over specialization constants.

        Returns None when no operand is a specialization constant so that plain
        constant expressions continue to use the existing folding/evaluation path.
        """
        if isinstance(expr, BinaryOpNode):
            left = self.process_constant_expression(getattr(expr, "left", None))
            right = self.process_constant_expression(getattr(expr, "right", None))
            if left is None or right is None:
                return None
            if not (
                self.is_specialization_constant_value(left)
                or self.is_specialization_constant_value(right)
            ):
                return None
            operator = getattr(expr, "operator", getattr(expr, "op", None))
            return self.emit_specialization_constant_binary_operation(
                operator, left, right
            )
        if isinstance(expr, UnaryOpNode):
            operand = self.process_constant_expression(getattr(expr, "operand", None))
            if operand is None or not self.is_specialization_constant_value(operand):
                return None
            operator = getattr(expr, "operator", getattr(expr, "op", None))
            return self.emit_specialization_constant_unary_operation(operator, operand)
        return specialization_conversion_expression(self, expr)

    def register_vector_constant(
        self, vector_type: SpirvId, components: List[SpirvId]
    ) -> SpirvId:
        """Create and register a composite vector constant."""
        key = (vector_type.id, tuple(c.id for c in components))
        if key in self.vector_constants:
            return self.vector_constants[key]

        id_value = self.get_id()
        is_specialized = any(
            self.is_specialization_constant_value(component) for component in components
        )
        opcode = "OpSpecConstantComposite" if is_specialized else "OpConstantComposite"

        component_list = " ".join([f"%{component.id}" for component in components])
        self.emit(f"%{id_value} = {opcode} %{vector_type.id} {component_list}")

        spirv_id = SpirvId(id_value, vector_type.type)
        self.value_types[id_value] = vector_type
        if is_specialized:
            self.specialization_constant_result_ids.add(id_value)
        self.vector_constants[key] = spirv_id
        return spirv_id

    def register_composite_constant(
        self, composite_type: SpirvId, components: List[SpirvId]
    ) -> SpirvId:
        """Create and register a composite constant."""
        composite_type = self.ensure_registered_type(composite_type)
        components = self.coerce_composite_components(
            composite_type, components, constant=True
        )
        key = (composite_type.id, tuple(component.id for component in components))
        if key in self.composite_constants:
            return self.composite_constants[key]

        id_value = self.get_id()
        is_specialized = any(
            self.is_specialization_constant_value(component) for component in components
        )
        opcode = "OpSpecConstantComposite" if is_specialized else "OpConstantComposite"
        component_list = " ".join(f"%{component.id}" for component in components)
        self.emit(f"%{id_value} = {opcode} %{composite_type.id} {component_list}")

        spirv_id = SpirvId(id_value, composite_type.type)
        self.value_types[id_value] = composite_type
        if is_specialized:
            self.specialization_constant_result_ids.add(id_value)
        self.composite_constants[key] = spirv_id
        return spirv_id

    def is_constant_instruction(self, value_id: SpirvId) -> bool:
        return (
            any(constant.id == value_id.id for constant in self.constants.values())
            or any(
                constant.id == value_id.id
                for constant in self.vector_constants.values()
            )
            or any(
                constant.id == value_id.id
                for constant in self.composite_constants.values()
            )
            or any(
                constant.id == value_id.id
                for constant in self.specialization_constants.values()
            )
            or value_id.id in self.specialization_constant_result_ids
        )

    def emit_named_constant_debug_name(self, name: str, value_id: SpirvId):
        """Attach a debug name to a named constant result id once."""
        if value_id.id in self.named_constant_debug_ids:
            return
        self.emit(f'OpName %{value_id.id} "{name}"')
        self.named_constant_debug_ids.add(value_id.id)

    def spirv_specialization_constant_attributes(self, node):
        """Return SPIR-V specialization-constant attributes on a declaration."""
        attributes = []
        for attr in getattr(node, "attributes", []) or []:
            attr_name = getattr(attr, "name", None)
            if not attr_name:
                continue
            normalized = str(attr_name).lower().replace("-", "_")
            for prefix in ("spirv_", "vulkan_", "vk_"):
                if normalized.startswith(prefix):
                    normalized = normalized[len(prefix) :]
                    break
            if normalized in {
                "constant_id",
                "spec_id",
                "specialization_constant",
                "function_constant",
            }:
                attributes.append(attr)
        return attributes

    def spirv_specialization_constant_id(self, node) -> Optional[int]:
        """Return the SPIR-V SpecId for a specialization-constant declaration."""
        attributes = self.spirv_specialization_constant_attributes(node)
        if not attributes:
            return None

        name = getattr(node, "name", getattr(node, "variable_name", "<unnamed>"))
        if len(attributes) > 1:
            raise VulkanRefusal(
                f"SPIR-V specialization constant '{name}' has multiple SpecId "
                "attributes"
            )

        arguments = getattr(attributes[0], "arguments", []) or []
        spec_id = (
            self.literal_int_argument(arguments[0]) if len(arguments) == 1 else None
        )
        if spec_id is None:
            raise VulkanRefusal(
                f"SPIR-V specialization constant '{name}' requires an integer id"
            )
        if spec_id < 0:
            raise VulkanRefusal(
                f"SPIR-V specialization constant '{name}' requires a non-negative id"
            )
        return spec_id

    def coerce_scalar_constant_value(
        self, expr, target_type: SpirvId
    ) -> Optional[Union[bool, int, float]]:
        """Return a Python scalar value for a literal constant expression."""
        type_name = self.normalize_primitive_name(target_type.type.base_type)
        if type_name not in {"bool", "float", "double", *self.INTEGER_TYPE_NAMES}:
            return None

        if isinstance(expr, LiteralNode):
            value = expr.value
        elif isinstance(expr, (bool, int, float)):
            value = expr
        elif expr is None:
            if type_name == "bool":
                return False
            if type_name in {"float", "double"}:
                return 0.0
            return 0
        else:
            return None

        if type_name == "bool":
            if isinstance(value, str):
                return value.lower() == "true"
            return bool(value)
        if type_name in {"float", "double"}:
            return float(value)
        return int(value)

    def process_named_constant_declaration(self, node) -> Optional[SpirvId]:
        """Register a CrossGL named constant for expression lookup."""
        name = getattr(node, "name", getattr(node, "variable_name", None))
        if not name:
            return None

        type_source = getattr(
            node, "const_type", getattr(node, "var_type", getattr(node, "vtype", "int"))
        )
        type_id = self.map_crossgl_type(type_source)
        value = getattr(node, "value", getattr(node, "initial_value", None))
        spec_id = self.spirv_specialization_constant_id(node)

        if spec_id is not None:
            scalar_value = self.coerce_scalar_constant_value(value, type_id)
            if scalar_value is None:
                self.refuse_lowering(
                    f"SPIR-V specialization constant {name} requires "
                    "a scalar literal default"
                )
            else:
                constant_id = self.register_specialization_constant(
                    scalar_value, type_id, spec_id
                )
                self.emit_named_constant_debug_name(name, constant_id)
                self.named_constants[name] = constant_id
                return constant_id

        scalar_value = self.evaluate_literal_scalar_constant(value, type_id)
        if scalar_value is not None:
            constant_id = self.register_constant(scalar_value, type_id)
            self.emit_named_constant_debug_name(name, constant_id)
            self.named_constants[name] = constant_id
            self.literal_scalar_constants[name] = scalar_value
            return constant_id

        constant_id = self.process_constant_expression(value, type_id)
        if constant_id is None:
            if value is not None:
                self.refuse_lowering(
                    f"SPIR-V module-scope constant {name} has an initializer "
                    "that is not a constant expression"
                )
            constant_id = self.default_value_for_type(type_id)
        self.emit_named_constant_debug_name(name, constant_id)
        self.named_constants[name] = constant_id
        return constant_id

    def spirv_specialization_constant_declaration_node(self, node):
        if isinstance(node, VariableNode):
            if self.spirv_specialization_constant_attributes(node):
                return node
            return None
        if isinstance(node, AssignmentNode) and isinstance(
            getattr(node, "left", None), VariableNode
        ):
            declaration = node.left
            if not self.spirv_specialization_constant_attributes(declaration):
                return None
            if getattr(declaration, "initial_value", None) is None:
                declaration.initial_value = getattr(node, "right", None)
            return declaration
        return None

    def evaluate_literal_scalar_constant(
        self, expr, type_id: SpirvId
    ) -> Optional[Union[bool, int, float]]:
        """Evaluate a narrow scalar constant expression for module-scope constants."""
        evaluated = spirv_evaluated_scalar_initializer(self, expr, type_id)
        if evaluated is not None:
            return evaluated
        type_name = self.normalize_primitive_name(type_id.type.base_type)
        if type_name == "bool":
            return self.coerce_scalar_constant_value(expr, type_id)
        if type_name in self.INTEGER_TYPE_NAMES:
            value = evaluate_literal_int_expression(
                expr, self.active_literal_int_constants()
            )
            if value is None:
                return None
            if type_name in self.UNSIGNED_INTEGER_TYPES and value < 0:
                return None
            return int(value)
        if type_name not in {"float", "double"}:
            return None

        value = self.evaluate_literal_float_expression(expr)
        return None if value is None else float(value)

    def evaluate_literal_float_expression(self, expr) -> Optional[float]:
        """Evaluate literal float arithmetic without emitting SPIR-V instructions."""
        if expr is None or isinstance(expr, bool):
            return None
        if isinstance(expr, (int, float)):
            return float(expr)
        if isinstance(expr, str):
            try:
                return float(expr)
            except ValueError:
                value = self.literal_scalar_constants.get(expr)
                if isinstance(value, (int, float)):
                    return float(value)
                int_value = self.active_literal_int_constants().get(expr)
                return None if int_value is None else float(int_value)
        if hasattr(expr, "value"):
            value = self.evaluate_literal_float_expression(getattr(expr, "value"))
            if value is not None:
                return value
        name = getattr(expr, "name", None)
        if isinstance(name, str):
            value = self.literal_scalar_constants.get(name)
            if isinstance(value, (int, float)):
                return float(value)
            int_value = self.active_literal_int_constants().get(name)
            return None if int_value is None else float(int_value)

        class_name = expr.__class__.__name__
        if "UnaryOp" in class_name:
            operand = self.evaluate_literal_float_expression(
                getattr(expr, "operand", None)
            )
            if operand is None:
                return None
            operator = getattr(expr, "operator", getattr(expr, "op", None))
            if operator == "+":
                return operand
            if operator == "-":
                return -operand
            return None
        if "BinaryOp" not in class_name:
            return None

        left = self.evaluate_literal_float_expression(getattr(expr, "left", None))
        right = self.evaluate_literal_float_expression(getattr(expr, "right", None))
        if left is None or right is None:
            return None
        operator = getattr(expr, "operator", getattr(expr, "op", None))
        if operator == "+":
            return left + right
        if operator == "-":
            return left - right
        if operator == "*":
            return left * right
        if operator == "/" and right != 0.0:
            return left / right
        return None

    def image_offset_operand(self, offset_id: SpirvId, *, gather: bool = False) -> str:
        """``ConstOffset`` for a constant offset; otherwise ``Offset``, which Vulkan allows on gathers only.

        Vulkan accepts a non-constant ``Offset`` image operand only on ``OpImage*Gather`` unless the
        ``maintenance8`` feature is enabled (VUID-RuntimeSpirv-Offset-10213), so it is refused elsewhere, as GLSL
        and HLSL refuse non-constant offsets of their sampling functions.
        """
        if self.is_constant_instruction(offset_id):
            return f"ConstOffset %{offset_id.id}"

        if not gather:
            self.refuse_lowering(
                "texture offset: a non-constant texel offset is valid only on gather operations "
                "(Vulkan requires maintenance8 for others, VUID-RuntimeSpirv-Offset-10213)"
            )
        self.require_capability("ImageGatherExtended")
        return f"Offset %{offset_id.id}"

    def image_const_offsets_operand(self, offset_ids: List[SpirvId]) -> Optional[str]:
        if len(offset_ids) != 4 or not all(
            self.is_constant_instruction(offset_id) for offset_id in offset_ids
        ):
            return None

        offset_type = self.value_types.get(offset_ids[0].id)
        if offset_type is None or any(
            getattr(self.value_types.get(offset_id.id), "id", None) != offset_type.id
            for offset_id in offset_ids
        ):
            return None

        offsets_type = self.register_array_type(offset_type, 4)
        offsets_id = self.register_composite_constant(offsets_type, offset_ids)
        self.require_capability("ImageGatherExtended")
        return f"ConstOffsets %{offsets_id.id}"

    def image_operands(self, *operands: str) -> str:
        operand_order = {
            "Bias": 0,
            "Lod": 1,
            "Grad": 2,
            "ConstOffset": 3,
            "Offset": 4,
            "ConstOffsets": 5,
            "Sample": 6,
            "MinLod": 7,
        }
        entries = []
        for operand in operands:
            if not operand:
                continue
            parts = operand.split()
            if not parts:
                continue
            entries.append((parts[0], parts[1:]))
        if not entries:
            return ""

        entries.sort(key=lambda entry: operand_order.get(entry[0], len(operand_order)))
        masks = [mask for mask, _ in entries]
        values = [value for _, entry_values in entries for value in entry_values]
        return " ".join(["|".join(masks)] + values)

    def create_variable(
        self,
        type_id: SpirvId,
        storage_class: str,
        name: Optional[str] = None,
        initializer: Optional[SpirvId] = None,
    ) -> SpirvId:
        """Create a new variable."""
        pointer_type = self.register_pointer_type(type_id, storage_class)

        id_value = self.get_id()
        initializer_operand = f" %{initializer.id}" if initializer is not None else ""
        self.emit(
            f"%{id_value} = OpVariable %{pointer_type.id} "
            f"{storage_class}{initializer_operand}"
        )

        if name:
            self.emit(f'OpName %{id_value} "{name}"')

        spirv_id = SpirvId(id_value, pointer_type.type, name)
        self.variable_value_types[id_value] = type_id
        return spirv_id

    def store_to_variable(self, variable_id: SpirvId, value_id: SpirvId):
        """Store a value to a variable."""
        readonly_builtin_name = self.readonly_builtin_pointer_names.get(variable_id.id)
        if readonly_builtin_name is not None:
            self.refuse_lowering(
                f"cannot assign to read-only SPIR-V builtin " f"{readonly_builtin_name}"
            )
            return
        readonly_pointer_name = self.readonly_pointer_names.get(variable_id.id)
        if readonly_pointer_name is not None:
            self.refuse_lowering(
                f"cannot assign to read-only SPIR-V pointer " f"{readonly_pointer_name}"
            )
            return

        metadata = self.storage_buffer_access_metadata_for_pointer(variable_id)
        if metadata is not None and metadata.get("readonly"):
            self.refuse_lowering("storage buffer store requires a writable buffer")
            return

        target_type = self.pointer_pointee_type(variable_id)
        if self.type_contains_runtime_array(target_type):
            self.refuse_lowering(
                "runtime-array aggregate values cannot be stored " "as SPIR-V values"
            )
            return

        value_id = self.convert_value_for_store(variable_id, value_id)
        self.emit(f"OpStore %{variable_id.id} %{value_id.id}")

    def load_from_variable(self, variable_id: SpirvId, result_type: SpirvId) -> SpirvId:
        """Load a value from a variable."""
        storage_metadata = self.storage_buffer_access_metadata_for_pointer(variable_id)
        if storage_metadata is not None and storage_metadata.get("writeonly"):
            self.refuse_lowering("storage buffer load requires a readable buffer")
            return self.default_value_for_type(result_type)

        if self.type_contains_runtime_array(result_type):
            return self.runtime_array_aggregate_fallback(
                "runtime-array aggregate values cannot be loaded as SPIR-V values"
            )

        id_value = self.get_id()
        self.emit(f"%{id_value} = OpLoad %{result_type.id} %{variable_id.id}")

        spirv_id = SpirvId(id_value, result_type.type)
        self.value_types[id_value] = result_type
        resource_metadata = self.resource_metadata_for_pointer(variable_id)
        if resource_metadata is not None:
            self.resource_type_metadata[id_value] = resource_metadata
        if self.is_non_uniform_value(variable_id):
            self.mark_non_uniform_result(spirv_id)
        matrix_type = self.matrix_column_arrays.get(result_type.id)
        if matrix_type is not None:
            return matrix_from_column_array(self, spirv_id, result_type, matrix_type)
        return spirv_id

    def convert_value_for_store(
        self, variable_id: SpirvId, value_id: SpirvId
    ) -> SpirvId:
        """Convert a stored value to the pointer's known pointee type when possible."""
        sample_mask_value = self.convert_sample_mask_store_value(variable_id, value_id)
        if sample_mask_value is not None:
            return sample_mask_value

        target_type = self.pointer_pointee_type(variable_id)
        if target_type is None:
            return value_id

        return self.convert_value_to_type(
            value_id, target_type, allow_vector_to_scalar=True
        )

    def pointer_pointee_type(self, variable_id: SpirvId) -> Optional[SpirvId]:
        target_type = self.variable_value_types.get(variable_id.id)
        if target_type is None and variable_id.type.storage_class:
            target_type = self.find_registered_type_by_base(
                variable_id.type.base_type.replace("ptr_", "", 1)
            )
        return target_type

    def pointer_type_pointee_type(
        self, pointer_type: Optional[SpirvId]
    ) -> Optional[SpirvId]:
        if pointer_type is None or pointer_type.type.storage_class is None:
            return None
        return self.find_registered_type_by_base(
            pointer_type.type.base_type.replace("ptr_", "", 1)
        )

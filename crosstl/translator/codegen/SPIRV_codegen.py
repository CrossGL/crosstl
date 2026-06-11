"""CrossGL-to-Vulkan SPIR-V code generator."""

import re
from typing import List, Optional, Set, Tuple, Union

from ..ast import (
    ArrayAccessNode,
    ArrayLiteralNode,
    ArrayNode,
    AssignmentNode,
    BinaryOpNode,
    BreakNode,
    ConstantNode,
    ConstructorNode,
    ConstructorPatternNode,
    ContinueNode,
    DoWhileNode,
    EnumNode,
    ForInNode,
    ForNode,
    FunctionCallNode,
    IdentifierNode,
    IdentifierPatternNode,
    IfNode,
    LiteralNode,
    LiteralPatternNode,
    LoopNode,
    MatchNode,
    MemberAccessNode,
    MeshOpNode,
    RangeNode,
    RayQueryOpNode,
    RayTracingOpNode,
    ReturnNode,
    ShaderNode,
    StructNode,
    StructPatternNode,
    SwitchNode,
    TernaryOpNode,
    UnaryOpNode,
    VariableNode,
    WaveOpNode,
    WhileNode,
    WildcardPatternNode,
)
from ..stage_utils import STAGE_QUALIFIER_NAMES, normalize_stage_name
from .array_utils import (
    collect_literal_int_constants,
    evaluate_literal_int_expression,
    parse_array_type,
)
from .enum_utils import (
    collect_generic_enum_specializations,
    collect_generic_enum_struct_definitions,
    enum_struct_fields,
    enum_variant_payload_fields,
    generic_enum_specialized_fields,
    generic_enum_specialized_variant_fields,
    generic_type_parts,
    resolve_generic_enum_specialization,
    substitute_generic_type_name,
)
from .generic_function_utils import (
    generic_function_call_name,
    generic_function_emission_list,
    generic_function_parameters,
    generic_function_value_arguments,
    iter_function_nodes,
    prepare_generic_function_specializations,
)
from .image_access_contracts import (
    collect_function_image_access_requirements,
    collect_function_parameter_names,
    image_access_diagnostic_name,
    image_access_requirement_label,
    image_access_satisfies_requirement,
)


class SpirvType:
    """Represents a SPIR-V type with storage class information."""

    def __init__(self, base_type: str, storage_class: Optional[str] = None):
        """Store the base type name and optional storage class."""
        self.base_type = base_type
        self.storage_class = storage_class

    def __str__(self) -> str:
        """Return a readable type label for debug output."""
        if self.storage_class:
            return f"{self.base_type} ({self.storage_class})"
        return self.base_type


class SpirvId:
    """Represents a SPIR-V ID with its associated type."""

    def __init__(
        self, id_value: int, spirv_type: SpirvType, name: Optional[str] = None
    ):
        """Store the numeric result id, type metadata, and optional name."""
        self.id = id_value
        self.type = spirv_type
        self.name = name

    def __str__(self) -> str:
        """Return a readable SPIR-V id label for debug output."""
        if self.name:
            return f"%{self.id} ({self.name}: {self.type})"
        return f"%{self.id} ({self.type})"


class UnsupportedSPIRVFeatureError(ValueError):
    """Raised when SPIR-V codegen sees a feature it cannot safely lower."""

    project_diagnostic_code = "project.translate.unsupported-feature"

    def __init__(
        self,
        feature: str,
        message: str,
        *,
        missing_capabilities=(),
        source_location=None,
    ):
        super().__init__(message)
        self.feature = feature
        self.missing_capabilities = tuple(missing_capabilities)
        self.source_location = source_location


class VulkanSPIRVCodeGen:
    """Generates SPIR-V code from a CrossGL shader AST."""

    SIGNED_INT32_MIN = -(1 << 31)
    SIGNED_INT32_MAX = (1 << 31) - 1
    UNSIGNED_INT32_MAX = (1 << 32) - 1
    SIGNED_INT64_MIN = -(1 << 63)
    SIGNED_INT64_MAX = (1 << 63) - 1
    UNSIGNED_INT64_MAX = (1 << 64) - 1
    INTEGER_TYPE_WIDTHS = {
        "int": 32,
        "uint": 32,
        "i64": 64,
        "u64": 64,
    }
    SIGNED_INTEGER_TYPES = {"int", "i64"}
    UNSIGNED_INTEGER_TYPES = {"uint", "u64"}
    INTEGER_TYPE_NAMES = SIGNED_INTEGER_TYPES | UNSIGNED_INTEGER_TYPES

    def __init__(self, *, include_resource_interface_variables: bool = False):
        """Initialize an empty SPIR-V module-generation state."""
        self.include_resource_interface_variables = include_resource_interface_variables
        self.reset_generation_state()

    def reset_generation_state(self):
        """Reset per-module SPIR-V ids, declarations, and symbol caches."""
        self.next_id = 1
        self.code_lines = []
        self.decorations = []
        self.required_extensions = set()

        self.primitive_types = {}
        self.vector_types = {}
        self.matrix_types = {}
        self.struct_types = {}
        self.pointer_types = {}
        self.function_types = {}
        self.array_types = {}
        self.layout_array_types = {}
        self.layout_struct_types = {}
        self.layout_struct_source_types = {}
        self.resource_types = {}
        self.resource_image_types = {}
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
        self.global_variables = {}
        self.stage_global_variables = {}
        self.local_variables = {}
        self.named_constants = {}
        self.named_constant_debug_ids = set()
        self.resource_alias_variables = set()
        self.variable_value_types = {}
        self.value_types = {}
        self.constants = {}
        self.literal_int_constants = {}
        self.literal_scalar_constants = {}
        self.vector_constants = {}
        self.composite_constants = {}
        self.specialization_constants = {}
        self.resource_type_metadata = {}
        self.structured_buffer_metadata = {}
        self.storage_buffer_access_metadata = {}
        self.uniform_block_wrapped_variables = {}
        self.precise_global_variables = set()
        self.precise_local_variables = set()
        self.no_contraction_ids = set()
        self.precise_expression_depth = 0
        self.non_uniform_ids = set()

        self.functions = {}
        self.function_nodes = {}
        self.function_nodes_by_name = {}
        self.function_signatures = {}
        self.stage_local_functions = {}
        self.stage_local_function_signatures = {}
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
        self.generic_function_definitions = {}
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
        self.function_resource_array_type_hints = {}
        self.stage_local_function_resource_array_type_hints = {}
        self.function_storage_image_pointer_params = {}
        self.stage_local_function_storage_image_pointer_params = {}
        self.function_execution_models = {}
        self.current_execution_model = None
        self.current_function_name = None
        self.current_function_id = None
        self.current_stage = None
        self.current_return_type = None
        self.current_return_type_source = None
        self.current_return_semantic_output = None
        self.current_entry_point_return_outputs = None
        self.current_expression_expected_type = None
        self.current_generic_type_substitutions = {}
        self.mesh_output_counts_by_function = {}
        self.mesh_vertex_output_variable = None
        self.mesh_vertex_output_limit = None
        self.mesh_primitive_index_outputs = {}
        self.current_mesh_output_parameters = {}
        self.function_mesh_output_parameter_indices = {}
        self.stage_local_function_mesh_output_parameter_indices = {}
        self.function_interface_variables = {}
        self.function_interface_variables_by_name = {}
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
        self.mesh_output_member_variables = {}
        self.mesh_output_member_shadow_variables = {}
        self.mesh_output_member_locations = {}
        self.struct_member_metadata = {}
        self.task_payload_shared_variables = {}
        self.task_payload_interface_by_function = {}
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

        self.inputs = []
        self.outputs = []
        self.uniform_buffers = []
        self.next_input_location = 0
        self.next_output_location = 0
        self.interface_location_counters = {}
        self.used_input_locations = set()
        self.used_output_locations = set()
        self.next_resource_binding = 0
        self.next_resource_bindings = {}
        self.reserved_resource_bindings = set()
        self.used_resource_bindings = set()

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
        self.require_capability("ComputeDerivativeGroupQuadsKHR")
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
            raise ValueError(f"Unknown SPIR-V resource type {type_name}")

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
        if cache_key in self.resource_types:
            return self.resource_types[cache_key]

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
                id_value = self.get_id()
                self.emit(f"%{id_value} = OpTypeSampledImage %{image_type.id}")
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

        return spirv_id

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
        id_value = self.get_id()
        self.emit(f"%{id_value} = OpTypeRayQueryKHR")

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
        key = (value, type_id.id)
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
        return spirv_id

    def spirv_constant_literal_text(
        self, value: Union[bool, int, float], type_name: str
    ) -> str:
        """Return a SPIR-V assembly literal valid for the result type."""
        type_name = self.normalize_primitive_name(type_name)
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
        return spirv_id

    def register_vector_constant(
        self, vector_type: SpirvId, components: List[SpirvId]
    ) -> SpirvId:
        """Create and register a composite vector constant."""
        key = (vector_type.id, tuple(c.id for c in components))
        if key in self.vector_constants:
            return self.vector_constants[key]

        id_value = self.get_id()

        component_list = " ".join([f"%{component.id}" for component in components])
        self.emit(
            f"%{id_value} = OpConstantComposite %{vector_type.id} {component_list}"
        )

        spirv_id = SpirvId(id_value, vector_type.type)
        self.value_types[id_value] = vector_type
        self.vector_constants[key] = spirv_id
        return spirv_id

    def register_composite_constant(
        self, composite_type: SpirvId, components: List[SpirvId]
    ) -> SpirvId:
        """Create and register a composite constant."""
        key = (composite_type.id, tuple(component.id for component in components))
        if key in self.composite_constants:
            return self.composite_constants[key]

        id_value = self.get_id()
        component_list = " ".join(f"%{component.id}" for component in components)
        self.emit(
            f"%{id_value} = OpConstantComposite %{composite_type.id} {component_list}"
        )

        spirv_id = SpirvId(id_value, composite_type.type)
        self.value_types[id_value] = composite_type
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
            raise ValueError(
                f"SPIR-V specialization constant '{name}' has multiple SpecId "
                "attributes"
            )

        arguments = getattr(attributes[0], "arguments", []) or []
        spec_id = (
            self.literal_int_argument(arguments[0]) if len(arguments) == 1 else None
        )
        if spec_id is None:
            raise ValueError(
                f"SPIR-V specialization constant '{name}' requires an integer id"
            )
        if spec_id < 0:
            raise ValueError(
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
                self.emit(
                    f"; WARNING: SPIR-V specialization constant {name} requires "
                    "a scalar literal default; emitting a plain constant fallback"
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
        type_name = self.normalize_primitive_name(type_id.type.base_type)
        if type_name == "bool":
            return self.coerce_scalar_constant_value(expr, type_id)
        if type_name in self.INTEGER_TYPE_NAMES:
            value = evaluate_literal_int_expression(expr, self.literal_int_constants)
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
                int_value = self.literal_int_constants.get(expr)
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
            int_value = self.literal_int_constants.get(name)
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

    def image_offset_operand(self, offset_id: SpirvId) -> str:
        if self.is_constant_instruction(offset_id):
            return f"ConstOffset %{offset_id.id}"

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
            self.emit(
                f"; WARNING: cannot assign to read-only SPIR-V builtin "
                f"{readonly_builtin_name}"
            )
            return
        readonly_pointer_name = self.readonly_pointer_names.get(variable_id.id)
        if readonly_pointer_name is not None:
            self.emit(
                f"; WARNING: cannot assign to read-only SPIR-V pointer "
                f"{readonly_pointer_name}"
            )
            return

        metadata = self.storage_buffer_access_metadata_for_pointer(variable_id)
        if metadata is not None and metadata.get("readonly"):
            self.emit("; WARNING: storage buffer store requires a writable buffer")
            return

        target_type = self.pointer_pointee_type(variable_id)
        if self.type_contains_runtime_array(target_type):
            self.emit(
                "; WARNING: runtime-array aggregate values cannot be stored "
                "as SPIR-V values"
            )
            return

        value_id = self.convert_value_for_store(variable_id, value_id)
        self.emit(f"OpStore %{variable_id.id} %{value_id.id}")

    def load_from_variable(self, variable_id: SpirvId, result_type: SpirvId) -> SpirvId:
        """Load a value from a variable."""
        storage_metadata = self.storage_buffer_access_metadata_for_pointer(variable_id)
        if storage_metadata is not None and storage_metadata.get("writeonly"):
            self.emit("; WARNING: storage buffer load requires a readable buffer")
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

        return self.convert_value_to_type(value_id, target_type)

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

    def copy_array_pointer_to_function_storage(
        self,
        source_pointer: SpirvId,
        target_array_type: SpirvId,
        name: Optional[str] = None,
    ) -> Optional[SpirvId]:
        target_array_info = self.array_type_info_from_type(target_array_type)
        if target_array_info is None:
            return None

        target_element_type, target_size = target_array_info
        if target_size is None:
            return None

        source_array_type = self.pointer_pointee_type(source_pointer)
        source_array_info = self.array_type_info_from_type(source_array_type)
        if source_array_info is None:
            return None

        source_element_type, source_size = source_array_info
        if source_size is not None and int(source_size) != int(target_size):
            return None

        scratch_array = self.create_variable(target_array_type, "Function", name)
        index_type = self.register_primitive_type("int")
        source_storage_class = source_pointer.type.storage_class or "Function"
        source_element_ptr_type = self.register_pointer_type(
            source_element_type, source_storage_class
        )
        target_element_ptr_type = self.register_pointer_type(
            target_element_type, "Function"
        )

        for index_value in range(int(target_size)):
            index = self.register_constant(index_value, index_type)
            source_element_pointer = self.access_chain(
                source_pointer, [index], source_element_ptr_type
            )
            self.variable_value_types[source_element_pointer.id] = source_element_type
            self.propagate_storage_buffer_access_metadata(
                source_pointer, source_element_pointer
            )
            self.propagate_structured_buffer_descriptor_access_metadata(
                source_pointer, source_element_pointer, index
            )
            self.propagate_resource_access_metadata(
                source_pointer, source_element_pointer, source_element_type
            )
            self.propagate_readonly_builtin_pointer_name(
                source_pointer, source_element_pointer
            )
            self.propagate_readonly_pointer_name(source_pointer, source_element_pointer)

            target_element_pointer = self.access_chain(
                scratch_array, [index], target_element_ptr_type
            )
            self.variable_value_types[target_element_pointer.id] = target_element_type

            value = self.load_from_variable(source_element_pointer, source_element_type)
            value = self.convert_value_to_type(value, target_element_type)
            self.store_to_variable(target_element_pointer, value)

        return scratch_array

    def prepare_function_call_argument(
        self, arg: SpirvId, expected_type: SpirvId
    ) -> SpirvId:
        if expected_type.type.storage_class is None:
            return self.convert_value_to_type(arg, expected_type)

        expected_pointee_type = self.pointer_type_pointee_type(expected_type)
        if expected_pointee_type is None:
            return arg

        if arg.type.storage_class == expected_type.type.storage_class:
            return arg

        if expected_type.type.storage_class == "Function":
            expected_array_info = self.array_type_info_from_type(expected_pointee_type)
            if expected_array_info is not None:
                copied_array = self.copy_array_pointer_to_function_storage(
                    arg, expected_pointee_type
                )
                if copied_array is not None:
                    return copied_array

        return arg

    def is_sample_mask_builtin_variable(self, variable_id: SpirvId) -> bool:
        return self.builtin_names_by_variable_id.get(variable_id.id) == "SampleMask"

    def convert_sample_mask_store_value(
        self, variable_id: SpirvId, value_id: SpirvId
    ) -> Optional[SpirvId]:
        if not self.is_sample_mask_builtin_variable(variable_id):
            return None

        target_type = self.pointer_pointee_type(variable_id)
        array_info = self.array_type_info_from_type(target_type)
        if array_info is None:
            return None

        element_type, size = array_info
        if size is None:
            return None

        source_type = self.value_types.get(
            value_id.id
        ) or self.find_registered_type_by_base(value_id.type.base_type)
        if source_type is not None:
            source_array_info = self.array_type_info_from_type(source_type)
            if source_array_info is not None:
                return self.convert_value_to_type(value_id, target_type)

        converted = self.convert_value_to_type(value_id, element_type)
        if not self.value_has_type(converted, element_type):
            return value_id

        components = [converted]
        for _ in range(1, int(size)):
            components.append(self.default_value_for_type(element_type))
        return self.composite_construct(target_type, components)

    def load_entry_point_interface_value(
        self, variable_id: SpirvId, result_type: SpirvId
    ) -> SpirvId:
        if not self.is_sample_mask_builtin_variable(variable_id):
            variable_type = self.pointer_pointee_type(variable_id)
            if (
                variable_type is not None
                and variable_type.type.base_type != result_type.type.base_type
            ):
                loaded = self.load_from_variable(variable_id, variable_type)
                return self.convert_value_to_type(loaded, result_type)
            return self.load_from_variable(variable_id, result_type)

        variable_type = self.pointer_pointee_type(variable_id)
        array_info = self.array_type_info_from_type(variable_type)
        result_array = self.array_type_info_from_type(result_type)
        if array_info is None or result_array is not None:
            return self.load_from_variable(variable_id, result_type)

        element_type, _ = array_info
        loaded = self.load_from_variable(variable_id, variable_type)
        scalar = self.composite_extract(loaded, element_type, 0)
        return self.convert_value_to_type(scalar, result_type)

    def convert_value_to_type(self, value_id: SpirvId, target_type: SpirvId) -> SpirvId:
        """Convert scalar values to a compatible scalar or vector target type."""
        target_type = self.ensure_registered_type(target_type)
        source_type = self.value_types.get(
            value_id.id
        ) or self.find_registered_type_by_base(value_id.type.base_type)
        if (
            source_type is not None
            and source_type.type.base_type == target_type.type.base_type
        ):
            return value_id

        if source_type is not None:
            aggregate_value = self.convert_aggregate_value_to_type(
                value_id, source_type, target_type
            )
            if aggregate_value is not None:
                return aggregate_value

        target_vector = self.vector_component_type_and_count(target_type.type.base_type)
        source_vector = self.vector_component_type_and_count(value_id.type.base_type)
        if target_vector is None:
            if source_vector is not None:
                return value_id
            converted = self.convert_scalar_to_type(value_id, target_type)
            if self.normalize_primitive_name(
                converted.type.base_type
            ) == self.normalize_primitive_name(target_type.type.base_type):
                return converted
            return value_id

        if source_vector is not None:
            if source_vector[1] != target_vector[1]:
                return value_id
            target_component_type = self.register_primitive_type(target_vector[0])
            source_component_type = self.register_primitive_type(source_vector[0])
            components = []
            for index in range(source_vector[1]):
                component = self.composite_extract(
                    value_id, source_component_type, index
                )
                converted = self.convert_scalar_to_type(
                    component, target_component_type
                )
                if (
                    self.normalize_primitive_name(converted.type.base_type)
                    != target_vector[0]
                ):
                    return value_id
                components.append(converted)
            return self.composite_construct(target_type, components)

        component_type = self.register_primitive_type(target_vector[0])
        converted = self.convert_scalar_to_type(value_id, component_type)
        if self.normalize_primitive_name(converted.type.base_type) != target_vector[0]:
            return value_id
        return self.splat_scalar_to_vector(converted, target_type)

    def value_has_type(self, value_id: SpirvId, target_type: SpirvId) -> bool:
        value_type = self.value_types.get(
            value_id.id
        ) or self.find_registered_type_by_base(value_id.type.base_type)
        if value_type is None:
            return value_id.type.base_type == target_type.type.base_type
        return (
            value_type.id == target_type.id
            or value_type.type.base_type == target_type.type.base_type
        )

    def aggregate_canonical_key(self, type_id: SpirvId):
        source_type = self.layout_struct_source_types.get(type_id.id)
        if source_type is not None:
            return ("struct", source_type.id)

        array_info = self.array_type_info_from_type(type_id)
        if array_info is not None:
            element_type, size = array_info
            return ("array", self.aggregate_canonical_key(element_type), size)

        if type_id.type.base_type in self.current_struct_members:
            return ("struct", type_id.id)

        return ("type", type_id.id)

    def aggregate_types_are_layout_compatible(
        self, source_type: SpirvId, target_type: SpirvId
    ) -> bool:
        source_is_struct = source_type.type.base_type in self.current_struct_members
        target_is_struct = target_type.type.base_type in self.current_struct_members
        source_is_array = self.array_type_info_from_type(source_type) is not None
        target_is_array = self.array_type_info_from_type(target_type) is not None
        if not (
            (source_is_struct and target_is_struct)
            or (source_is_array and target_is_array)
        ):
            return False
        return self.aggregate_canonical_key(
            source_type
        ) == self.aggregate_canonical_key(target_type)

    def convert_aggregate_value_to_type(
        self, value_id: SpirvId, source_type: SpirvId, target_type: SpirvId
    ) -> Optional[SpirvId]:
        if self.type_contains_runtime_array(
            source_type
        ) or self.type_contains_runtime_array(target_type):
            return None

        if not self.aggregate_types_are_layout_compatible(source_type, target_type):
            return None

        source_array = self.array_type_info_from_type(source_type)
        target_array = self.array_type_info_from_type(target_type)
        if source_array is not None or target_array is not None:
            if source_array is None or target_array is None:
                return None
            source_element_type, source_size = source_array
            target_element_type, target_size = target_array
            if source_size is None or target_size is None or source_size != target_size:
                return None

            elements = []
            for index in range(int(target_size)):
                source_element = self.composite_extract(
                    value_id, source_element_type, index
                )
                target_element = self.convert_value_to_type(
                    source_element, target_element_type
                )
                if not self.value_has_type(target_element, target_element_type):
                    return None
                elements.append(target_element)
            return self.composite_construct(target_type, elements)

        source_members = self.current_struct_members.get(source_type.type.base_type)
        target_members = self.current_struct_members.get(target_type.type.base_type)
        if source_members is None or target_members is None:
            return None
        if len(source_members) != len(target_members):
            return None

        values = []
        for index, (
            (source_member_type, source_name),
            (target_member_type, target_name),
        ) in enumerate(zip(source_members, target_members)):
            if source_name != target_name:
                return None
            source_member = self.composite_extract(value_id, source_member_type, index)
            target_member = self.convert_value_to_type(
                source_member, target_member_type
            )
            if not self.value_has_type(target_member, target_member_type):
                return None
            values.append(target_member)

        return self.composite_construct(target_type, values)

    def promoted_numeric_type_name(self, type_names: List[str]) -> Optional[str]:
        numeric_types = {"float", "double"} | self.INTEGER_TYPE_NAMES
        normalized = [self.normalize_primitive_name(name) for name in type_names]
        if any(name not in numeric_types for name in normalized):
            return None
        if "double" in normalized:
            return "double"
        if "float" in normalized:
            return "float"
        if any(self.integer_type_width(name) == 64 for name in normalized):
            if any(name in self.UNSIGNED_INTEGER_TYPES for name in normalized):
                return "u64"
            return "i64"
        if "int" in normalized:
            return "int"
        return "uint"

    def promoted_bitwise_integer_type_name(
        self, type_names: List[str]
    ) -> Optional[str]:
        normalized = [self.normalize_primitive_name(name) for name in type_names]
        if any(name not in self.INTEGER_TYPE_NAMES for name in normalized):
            return None
        if any(self.integer_type_width(name) == 64 for name in normalized):
            if any(name in self.UNSIGNED_INTEGER_TYPES for name in normalized):
                return "u64"
            return "i64"
        if any(name in self.UNSIGNED_INTEGER_TYPES for name in normalized):
            return "uint"
        return "int"

    def ternary_result_type(self, true_value: SpirvId, false_value: SpirvId) -> SpirvId:
        true_type = self.value_types.get(true_value.id) or self.ensure_registered_type(
            true_value.type
        )
        false_type = self.value_types.get(
            false_value.id
        ) or self.ensure_registered_type(false_value.type)
        if true_type.type.base_type == false_type.type.base_type:
            return true_type

        true_vector = self.vector_component_type_and_count(true_type.type.base_type)
        false_vector = self.vector_component_type_and_count(false_type.type.base_type)
        if true_vector is not None and false_vector is not None:
            if true_vector[1] != false_vector[1]:
                return true_type
            component_type_name = self.promoted_numeric_type_name(
                [true_vector[0], false_vector[0]]
            )
            if component_type_name is None:
                return true_type
            component_type = self.register_primitive_type(component_type_name)
            return self.register_vector_type(component_type, true_vector[1])

        if true_vector is not None:
            component_type_name = self.promoted_numeric_type_name(
                [true_vector[0], false_type.type.base_type]
            )
            if component_type_name is None:
                return true_type
            component_type = self.register_primitive_type(component_type_name)
            return self.register_vector_type(component_type, true_vector[1])

        if false_vector is not None:
            component_type_name = self.promoted_numeric_type_name(
                [true_type.type.base_type, false_vector[0]]
            )
            if component_type_name is None:
                return false_type
            component_type = self.register_primitive_type(component_type_name)
            return self.register_vector_type(component_type, false_vector[1])

        scalar_type_name = self.promoted_numeric_type_name(
            [true_type.type.base_type, false_type.type.base_type]
        )
        if scalar_type_name is None:
            return true_type
        return self.register_primitive_type(scalar_type_name)

    def normalize_ternary_values(
        self, true_value: SpirvId, false_value: SpirvId
    ) -> Tuple[SpirvId, SpirvId, SpirvId]:
        result_type = self.ternary_result_type(true_value, false_value)
        true_value = self.convert_value_to_type(true_value, result_type)
        false_value = self.convert_value_to_type(false_value, result_type)
        if (
            self.value_types.get(true_value.id, true_value).type.base_type
            != result_type.type.base_type
            or self.value_types.get(false_value.id, false_value).type.base_type
            != result_type.type.base_type
        ):
            return self.ensure_registered_type(true_value.type), true_value, false_value
        return result_type, true_value, false_value

    def access_chain(
        self, base_id: SpirvId, indices: List[SpirvId], result_type: SpirvId
    ) -> SpirvId:
        """Create an access chain to a struct or array member."""
        id_value = self.get_id()

        index_list = " ".join([f"%{index.id}" for index in indices])
        self.emit(
            f"%{id_value} = OpAccessChain %{result_type.id} %{base_id.id} {index_list}"
        )

        spirv_id = SpirvId(id_value, result_type.type)
        if self.is_non_uniform_value(base_id) or any(
            self.is_non_uniform_value(index) for index in indices
        ):
            self.mark_non_uniform_result(spirv_id)
        return spirv_id

    def composite_extract(
        self, composite: SpirvId, member_type: SpirvId, member_index: int
    ) -> SpirvId:
        """Extract a member from a composite value."""
        id_value = self.get_id()
        self.emit(
            f"%{id_value} = OpCompositeExtract %{member_type.id} "
            f"%{composite.id} {member_index}"
        )

        spirv_id = SpirvId(id_value, member_type.type)
        self.value_types[id_value] = member_type
        return spirv_id

    def composite_insert(
        self,
        composite: SpirvId,
        value: SpirvId,
        composite_type: SpirvId,
        member_index: int,
    ) -> SpirvId:
        """Insert a member value into a composite value."""
        id_value = self.get_id()
        self.emit(
            f"%{id_value} = OpCompositeInsert %{composite_type.id} "
            f"%{value.id} %{composite.id} {member_index}"
        )

        spirv_id = SpirvId(id_value, composite_type.type)
        self.value_types[id_value] = composite_type
        return spirv_id

    def vector_shuffle(
        self, vector: SpirvId, result_type: SpirvId, component_indices: List[int]
    ) -> SpirvId:
        """Create a vector by selecting components from an existing vector."""
        id_value = self.get_id()
        components = " ".join(str(index) for index in component_indices)
        self.emit(
            f"%{id_value} = OpVectorShuffle %{result_type.id} "
            f"%{vector.id} %{vector.id} {components}"
        )

        spirv_id = SpirvId(id_value, result_type.type)
        self.value_types[id_value] = result_type
        return spirv_id

    def composite_construct(
        self, result_type: SpirvId, components: List[SpirvId]
    ) -> SpirvId:
        """Construct a composite value from component values."""
        id_value = self.get_id()
        component_list = " ".join(f"%{component.id}" for component in components)
        self.emit(
            f"%{id_value} = OpCompositeConstruct %{result_type.id} {component_list}"
        )

        spirv_id = SpirvId(id_value, result_type.type)
        self.value_types[id_value] = result_type
        return spirv_id

    def vector_member_info(self, vector_type: str, member_name: str):
        vector_info = self.vector_component_type_and_count(vector_type)
        if vector_info is None:
            return None

        component_type, component_count = vector_info
        component_indices = {
            "x": 0,
            "r": 0,
            "s": 0,
            "y": 1,
            "g": 1,
            "t": 1,
            "z": 2,
            "b": 2,
            "p": 2,
            "w": 3,
            "a": 3,
            "q": 3,
        }
        member_index = component_indices.get(member_name)
        if member_index is None or member_index >= component_count:
            return None

        return member_index, self.register_primitive_type(component_type)

    def vector_swizzle_indices(self, vector_type: str, member_name: str):
        vector_info = self.vector_component_type_and_count(vector_type)
        if vector_info is None or not 2 <= len(member_name) <= 4:
            return None

        component_indices = (
            {"x": 0, "y": 1, "z": 2, "w": 3},
            {"r": 0, "g": 1, "b": 2, "a": 3},
            {"s": 0, "t": 1, "p": 2, "q": 3},
        )
        family = next(
            (
                indices
                for indices in component_indices
                if all(component in indices for component in member_name)
            ),
            None,
        )
        if family is None:
            return None

        _, component_count = vector_info
        indices = [family[component] for component in member_name]
        if any(index >= component_count for index in indices):
            return None

        return indices

    def vector_swizzle_info(self, vector_type: str, member_name: str):
        vector_info = self.vector_component_type_and_count(vector_type)
        if vector_info is None:
            return None

        indices = self.vector_swizzle_indices(vector_type, member_name)
        if indices is None:
            return None

        component_type_name, _ = vector_info
        component_type = self.register_primitive_type(component_type_name)
        result_type = self.register_vector_type(component_type, len(indices))
        return indices, component_type, result_type

    def struct_member_info(self, struct_type: str, member_name: str):
        members = self.current_struct_members.get(struct_type)
        if not members:
            return None
        for index, (member_type, name) in enumerate(members):
            if name == member_name:
                return index, member_type
        return None

    def struct_type_name_from_pointer(self, pointer: SpirvId):
        struct_type_id = self.variable_value_types.get(pointer.id)
        return struct_type_id.type.base_type if struct_type_id else None

    def create_member_access_pointer(
        self, base_pointer: SpirvId, member_name: str
    ) -> Optional[SpirvId]:
        struct_type = self.struct_type_name_from_pointer(base_pointer)
        member_info = self.struct_member_info(struct_type, member_name)
        if member_info is None:
            return None

        member_index, member_type = member_info
        int_type = self.primitive_types["int"]
        index = self.register_constant(member_index, int_type)
        storage_class = base_pointer.type.storage_class or "Function"
        ptr_type = self.register_pointer_type(member_type, storage_class)
        access = self.access_chain(base_pointer, [index], ptr_type)
        self.variable_value_types[access.id] = member_type
        metadata = self.structured_buffer_metadata_for_pointer(base_pointer)
        if metadata is not None and metadata.get("_access_path") in {
            "element",
            "member",
        }:
            self.structured_buffer_metadata[access.id] = {
                **metadata,
                "_access_path": "member",
            }
        self.propagate_storage_buffer_access_metadata(base_pointer, access)
        self.propagate_readonly_builtin_pointer_name(base_pointer, access)
        self.propagate_readonly_pointer_name(base_pointer, access)
        return access

    def create_function(
        self, name: str, return_type: SpirvId, param_types: List[SpirvId]
    ) -> SpirvId:
        """Create a function declaration."""
        function_type = self.register_function_type(return_type, param_types)

        id_value = self.get_id()
        self.emit(
            f"%{id_value} = OpFunction %{return_type.id} None %{function_type.id}"
        )

        spirv_id = SpirvId(id_value, return_type.type, name)
        self.functions[name] = spirv_id
        self.function_signatures[name] = (return_type, param_types)

        return spirv_id

    def stage_local_function_key(self, stage, name: str):
        if stage is None:
            return None
        return (id(stage), name)

    def stage_global_variable_key(self, stage, name: str):
        if stage is None:
            return None
        return (id(stage), name)

    def current_stage_global_variable(self, name: str) -> Optional[SpirvId]:
        key = self.stage_global_variable_key(self.current_stage, name)
        if key is None:
            return None
        return self.stage_global_variables.get(key)

    def resolve_global_variable(self, name: str) -> Optional[SpirvId]:
        return self.current_stage_global_variable(name) or self.global_variables.get(
            name
        )

    def global_variable_exists_for_current_stage(self, name: str) -> bool:
        if self.current_stage_global_variable(name) is not None:
            return True
        return name in self.global_variables

    def register_global_variable_name(
        self, name: str, variable: SpirvId, storage_class: str
    ):
        if self.current_stage is not None and storage_class in {"Input", "Output"}:
            key = self.stage_global_variable_key(self.current_stage, name)
            self.stage_global_variables[key] = variable
            return
        self.global_variables[name] = variable

    def register_stage_local_function(
        self,
        stage,
        name: str,
        function_id: SpirvId,
        return_type: SpirvId,
        param_types: List[SpirvId],
    ):
        key = self.stage_local_function_key(stage, name)
        if key is None:
            return
        self.stage_local_functions[key] = function_id
        self.stage_local_function_signatures[key] = (return_type, param_types)

    def resolve_function_reference(self, function_name: str):
        if self.current_stage is not None:
            key = self.stage_local_function_key(self.current_stage, function_name)
            if key in self.stage_local_functions:
                return (
                    self.stage_local_functions[key],
                    self.stage_local_function_signatures[key],
                )
        if function_name in self.functions:
            return (
                self.functions[function_name],
                self.function_signatures[function_name],
            )
        return None

    def resolve_function_signature(self, function_name: str):
        function_reference = self.resolve_function_reference(function_name)
        if function_reference is None:
            return None
        return function_reference[1]

    def has_function_reference(self, function_name: str) -> bool:
        return self.resolve_function_reference(function_name) is not None

    def stage_local_metadata(self, mapping, function_name: str):
        if self.current_stage is None:
            return None
        key = self.stage_local_function_key(self.current_stage, function_name)
        return mapping.get(key)

    def resolve_function_resource_array_params(self, function_name: str):
        params = self.stage_local_metadata(
            self.stage_local_function_resource_array_params, function_name
        )
        if params is not None:
            return params
        return self.function_resource_array_params.get(function_name, set())

    def resolve_function_resource_array_type_hints(self, function_name: str):
        hints = self.stage_local_metadata(
            self.stage_local_function_resource_array_type_hints, function_name
        )
        if hints is not None:
            return hints
        return self.function_resource_array_type_hints.get(function_name, {})

    def resolve_function_mesh_output_parameter_indices(self, function_name: str):
        indices = self.stage_local_metadata(
            self.stage_local_function_mesh_output_parameter_indices, function_name
        )
        if indices is not None:
            return indices
        return self.function_mesh_output_parameter_indices.get(function_name, set())

    def resolve_function_parameter_names(self, function_name: str):
        names = self.stage_local_metadata(
            self.stage_local_function_parameter_names, function_name
        )
        if names is not None:
            return names
        return self.function_parameter_names.get(function_name, [])

    def resolve_function_image_access_requirements(self, function_name: str):
        requirements = self.stage_local_metadata(
            self.stage_local_function_image_access_requirements, function_name
        )
        if requirements is not None:
            return requirements
        return self.function_image_access_requirements.get(function_name)

    def resolve_function_storage_buffer_access_requirements(self, function_name: str):
        requirements = self.stage_local_metadata(
            self.stage_local_function_storage_buffer_access_requirements,
            function_name,
        )
        if requirements is not None:
            return requirements
        return self.function_storage_buffer_access_requirements.get(function_name)

    def resolve_function_storage_image_pointer_params(self, function_name: str):
        params = self.stage_local_metadata(
            self.stage_local_function_storage_image_pointer_params,
            function_name,
        )
        if params is not None:
            return params
        return self.function_storage_image_pointer_params.get(function_name, set())

    def function_parameter_requires_storage_image_pointer(
        self, function_name: str, arg_index: int
    ) -> bool:
        pointer_param_names = self.resolve_function_storage_image_pointer_params(
            function_name
        )
        if not pointer_param_names:
            return False

        parameter_names = self.resolve_function_parameter_names(function_name)
        return (
            arg_index < len(parameter_names)
            and parameter_names[arg_index] in pointer_param_names
        )

    def resolve_inline_storage_buffer_function(
        self, function_name: str, call_args, call_node=None
    ):
        candidates = []
        seen_candidates = set()
        stage_candidates = self.stage_local_metadata(
            self.stage_local_inline_storage_buffer_functions, function_name
        )
        if stage_candidates is not None:
            for candidate in stage_candidates:
                candidates.append(candidate)
                seen_candidates.add(id(candidate))

        for candidate in self.inline_storage_buffer_functions.get(function_name, []):
            candidate_key = id(candidate)
            if candidate_key in seen_candidates:
                continue
            candidates.append(candidate)
            seen_candidates.add(candidate_key)

        if not candidates:
            return None

        scored_matches = []
        rejection_reasons = {}
        for candidate in candidates:
            match, rejection_reason = self.storage_buffer_candidate_call_match(
                candidate, call_args
            )
            if match is None:
                rejection_reasons[id(candidate)] = rejection_reason
                continue
            scored_matches.append((match["rank_score"], match["score"], candidate))

        if not scored_matches:
            if len(candidates) == 1:
                return candidates[0]
            if self.has_matching_non_storage_function_candidate(
                function_name, call_args
            ):
                return None
            raise self.storage_buffer_overload_diagnostic(
                function_name,
                call_args,
                candidates,
                "no candidate has compatible argument alignment",
                call_node=call_node,
                rejection_reasons=rejection_reasons,
            )

        scored_matches.sort(key=lambda item: (item[0], item[1]), reverse=True)
        best_rank_score, best_score, best_candidate = scored_matches[0]
        tied_candidates = [
            candidate
            for rank_score, score, candidate in scored_matches
            if rank_score == best_rank_score and score == best_score
        ]
        if len(tied_candidates) > 1:
            raise self.storage_buffer_overload_diagnostic(
                function_name,
                call_args,
                tied_candidates,
                "multiple candidates have compatible argument types",
                call_node=call_node,
                rejection_reasons=rejection_reasons,
            )
        return best_candidate

    def create_function_parameter(
        self, param_type: SpirvId, name: Optional[str] = None
    ) -> SpirvId:
        """Create a function parameter."""
        id_value = self.get_id()
        self.emit(f"%{id_value} = OpFunctionParameter %{param_type.id}")

        if name:
            self.emit(f'OpName %{id_value} "{name}"')

        spirv_id = SpirvId(id_value, param_type.type, name)
        self.value_types[id_value] = param_type
        return spirv_id

    def begin_block(self) -> SpirvId:
        """Begin a new basic block."""
        id_value = self.get_id()
        self.emit(f"%{id_value} = OpLabel")

        self.current_label = id_value
        return SpirvId(id_value, SpirvType("label"))

    def end_function(self):
        """End the current function."""
        self.emit("OpFunctionEnd")
        self.current_label = None

    def binary_operation(
        self, op: str, result_type: SpirvId, left: SpirvId, right: SpirvId
    ) -> SpirvId:
        """Create a binary operation."""
        matrix_result = self.matrix_binary_operation(op, result_type, left, right)
        if matrix_result is not None:
            return matrix_result

        arithmetic_ops = {
            "+": ("OpFAdd", "OpIAdd", "OpIAdd"),
            "-": ("OpFSub", "OpISub", "OpISub"),
            "*": ("OpFMul", "OpIMul", "OpIMul"),
            "MULTIPLY": ("OpFMul", "OpIMul", "OpIMul"),
            "/": ("OpFDiv", "OpSDiv", "OpUDiv"),
            "%": ("OpFMod", "OpSMod", "OpUMod"),
        }
        comparison_ops = {
            "==": ("OpFOrdEqual", "OpIEqual", "OpIEqual"),
            "!=": ("OpFOrdNotEqual", "OpINotEqual", "OpINotEqual"),
            "<": ("OpFOrdLessThan", "OpSLessThan", "OpULessThan"),
            ">": ("OpFOrdGreaterThan", "OpSGreaterThan", "OpUGreaterThan"),
            "<=": ("OpFOrdLessThanEqual", "OpSLessThanEqual", "OpULessThanEqual"),
            ">=": (
                "OpFOrdGreaterThanEqual",
                "OpSGreaterThanEqual",
                "OpUGreaterThanEqual",
            ),
        }

        if op in {"&&", "||"}:
            result_type, left, right = self.logical_result_type_and_operands(
                left, right
            )
            spv_op = "OpLogicalAnd" if op == "&&" else "OpLogicalOr"
        elif op in comparison_ops:
            result_type, left, right = self.comparison_result_type_and_operands(
                left, right
            )
            float_op, signed_op, unsigned_op = comparison_ops[op]
            component_type = self.scalar_or_vector_component_type(left.type)
            if component_type == "bool" and op in {"==", "!="}:
                spv_op = "OpLogicalEqual" if op == "==" else "OpLogicalNotEqual"
            else:
                spv_op = (
                    unsigned_op
                    if component_type in self.UNSIGNED_INTEGER_TYPES
                    else (
                        signed_op
                        if component_type in self.SIGNED_INTEGER_TYPES
                        else float_op
                    )
                )
        elif op in arithmetic_ops:
            result_type, left, right = self.align_binary_arithmetic_operands(
                result_type, left, right
            )
            float_op, signed_op, unsigned_op = arithmetic_ops[op]
            component_type = self.scalar_or_vector_component_type(left.type)
            spv_op = (
                unsigned_op
                if component_type in self.UNSIGNED_INTEGER_TYPES
                else (
                    signed_op
                    if component_type in self.SIGNED_INTEGER_TYPES
                    else float_op
                )
            )
        elif op in {"&", "|", "^"}:
            result_type, left, right = self.align_binary_bitwise_operands(
                result_type, left, right
            )
            spv_op = {
                "&": "OpBitwiseAnd",
                "|": "OpBitwiseOr",
                "^": "OpBitwiseXor",
            }[op]
        elif op in {"<<", ">>"}:
            result_type, left, right = self.align_binary_shift_operands(
                result_type, left, right
            )
            spv_op = {
                "<<": "OpShiftLeftLogical",
                ">>": "OpShiftRightLogical",
            }[op]
        else:
            spv_op = f"Op{op}"

        id_value = self.get_id()
        self.emit(f"%{id_value} = {spv_op} %{result_type.id} %{left.id} %{right.id}")
        self.decorate_no_contraction_result(id_value, spv_op, result_type)

        spirv_id = SpirvId(id_value, result_type.type)
        self.value_types[id_value] = result_type
        return spirv_id

    def align_binary_bitwise_operands(
        self, result_type: SpirvId, left: SpirvId, right: SpirvId
    ) -> Tuple[SpirvId, SpirvId, SpirvId]:
        """Convert bitwise operands to the result integer type before emission."""
        result_type = self.ensure_registered_type(result_type)
        left = self.convert_value_to_type(left, result_type)
        right = self.convert_value_to_type(right, result_type)
        return result_type, left, right

    def align_binary_shift_operands(
        self, result_type: SpirvId, left: SpirvId, right: SpirvId
    ) -> Tuple[SpirvId, SpirvId, SpirvId]:
        """Match SPIR-V shift count shape to the left/result operand shape."""
        result_type = self.ensure_registered_type(result_type)
        result_vector = self.vector_component_type_and_count(result_type.type.base_type)

        left = self.convert_value_to_type(left, result_type)
        if result_vector is None:
            right = self.convert_value_to_type(right, result_type)
            return result_type, left, right

        component_type_name, component_count = result_vector
        component_type = self.register_primitive_type(component_type_name)
        right_vector = self.vector_component_type_and_count(right.type.base_type)

        if right_vector is None:
            right = self.convert_scalar_to_type(right, component_type)
            if self.scalar_or_vector_component_type(right.type) == component_type_name:
                right = self.splat_scalar_to_vector(right, result_type)
        elif right_vector[1] == component_count:
            right = self.convert_value_to_type(right, result_type)

        return result_type, left, right

    def matrix_binary_operation(
        self, op: str, result_type: SpirvId, left: SpirvId, right: SpirvId
    ) -> Optional[SpirvId]:
        """Lower matrix arithmetic to SPIR-V matrix or column-vector operations."""
        left_type = self.registered_value_type(left)
        right_type = self.registered_value_type(right)
        left_matrix = self.matrix_operand_info(left_type)
        right_matrix = self.matrix_operand_info(right_type)
        if left_matrix is None and right_matrix is None:
            return None

        if op not in {"+", "-", "*", "/"}:
            return self.unsupported_matrix_operation(op, result_type)

        if left_matrix is not None and right_matrix is not None:
            if op in {"+", "-"}:
                return self.componentwise_matrix_operation(
                    op, left, left_type, left_matrix, right, right_type, right_matrix
                )
            if op == "*":
                return self.matrix_times_matrix(
                    left, left_matrix, right, right_matrix, result_type
                )
            return self.unsupported_matrix_operation(op, result_type)

        if left_matrix is not None:
            right_vector = self.vector_type_info_from_type(right_type)
            if op == "*" and right_vector is not None:
                return self.matrix_times_vector(left, left_matrix, right, right_vector)
            if right_vector is None:
                if op == "*":
                    return self.matrix_times_scalar(left, left_type, left_matrix, right)
                if op == "/":
                    return self.matrix_divided_by_scalar(
                        left, left_type, left_matrix, right
                    )
            return self.unsupported_matrix_operation(op, result_type)

        left_vector = self.vector_type_info_from_type(left_type)
        if op == "*" and left_vector is not None and right_matrix is not None:
            return self.vector_times_matrix(left, left_vector, right, right_matrix)
        if op == "*" and left_vector is None and right_matrix is not None:
            return self.matrix_times_scalar(right, right_type, right_matrix, left)

        return self.unsupported_matrix_operation(op, result_type)

    def matrix_operand_info(self, type_id: SpirvId):
        matrix_info = self.matrix_type_info_from_type(type_id)
        if matrix_info is None:
            return None

        column_type, column_count = matrix_info
        column_info = self.vector_type_info_from_type(column_type)
        if column_info is None:
            return None

        component_type, row_count = column_info
        return {
            "type": type_id,
            "column_type": column_type,
            "column_count": column_count,
            "component_type": component_type,
            "row_count": row_count,
        }

    def matrix_shapes_match(self, left_info, right_info) -> bool:
        return (
            left_info["column_count"] == right_info["column_count"]
            and left_info["row_count"] == right_info["row_count"]
            and left_info["component_type"].id == right_info["component_type"].id
        )

    def matrix_scalar_operand(self, value: SpirvId, component_type: SpirvId):
        value_type = self.registered_value_type(value)
        if value_type is not None:
            if self.vector_type_info_from_type(value_type) is not None:
                return None
            if self.matrix_type_info_from_type(value_type) is not None:
                return None

        converted = self.convert_scalar_to_type(value, component_type)
        if self.normalize_primitive_name(
            converted.type.base_type
        ) != self.normalize_primitive_name(component_type.type.base_type):
            return None
        return converted

    def unsupported_matrix_operation(self, op: str, fallback_type: SpirvId) -> SpirvId:
        self.emit(f"; WARNING: matrix operation '{op}' is not supported for operands")
        return self.default_value_for_type(self.ensure_registered_type(fallback_type))

    def componentwise_matrix_operation(
        self,
        op: str,
        left: SpirvId,
        left_type: SpirvId,
        left_info,
        right: SpirvId,
        right_type: SpirvId,
        right_info,
    ) -> SpirvId:
        if not self.matrix_shapes_match(left_info, right_info):
            return self.unsupported_matrix_operation(op, left_type)

        columns = []
        for index in range(left_info["column_count"]):
            left_column = self.composite_extract(left, left_info["column_type"], index)
            right_column = self.composite_extract(
                right, right_info["column_type"], index
            )
            columns.append(
                self.binary_operation(
                    op, left_info["column_type"], left_column, right_column
                )
            )
        return self.composite_construct(left_type, columns)

    def matrix_times_scalar(
        self, matrix: SpirvId, matrix_type: SpirvId, matrix_info, scalar: SpirvId
    ) -> SpirvId:
        scalar = self.matrix_scalar_operand(scalar, matrix_info["component_type"])
        if scalar is None:
            return self.unsupported_matrix_operation("*", matrix_type)

        id_value = self.get_id()
        self.emit(
            f"%{id_value} = OpMatrixTimesScalar %{matrix_type.id} "
            f"%{matrix.id} %{scalar.id}"
        )

        spirv_id = SpirvId(id_value, matrix_type.type)
        self.value_types[id_value] = matrix_type
        self.decorate_no_contraction_result(
            id_value, "OpMatrixTimesScalar", matrix_type
        )
        return spirv_id

    def matrix_divided_by_scalar(
        self, matrix: SpirvId, matrix_type: SpirvId, matrix_info, scalar: SpirvId
    ) -> SpirvId:
        scalar = self.matrix_scalar_operand(scalar, matrix_info["component_type"])
        if scalar is None:
            return self.unsupported_matrix_operation("/", matrix_type)

        columns = []
        for index in range(matrix_info["column_count"]):
            column = self.composite_extract(matrix, matrix_info["column_type"], index)
            columns.append(
                self.binary_operation("/", matrix_info["column_type"], column, scalar)
            )
        return self.composite_construct(matrix_type, columns)

    def matrix_times_vector(
        self, matrix: SpirvId, matrix_info, vector: SpirvId, vector_info
    ) -> SpirvId:
        if (
            vector_info[0].id != matrix_info["component_type"].id
            or vector_info[1] != matrix_info["column_count"]
        ):
            return self.unsupported_matrix_operation("*", matrix_info["column_type"])

        id_value = self.get_id()
        self.emit(
            f"%{id_value} = OpMatrixTimesVector %{matrix_info['column_type'].id} "
            f"%{matrix.id} %{vector.id}"
        )

        spirv_id = SpirvId(id_value, matrix_info["column_type"].type)
        self.value_types[id_value] = matrix_info["column_type"]
        self.decorate_no_contraction_result(
            id_value, "OpMatrixTimesVector", matrix_info["column_type"]
        )
        return spirv_id

    def vector_times_matrix(
        self, vector: SpirvId, vector_info, matrix: SpirvId, matrix_info
    ) -> SpirvId:
        if (
            vector_info[0].id != matrix_info["component_type"].id
            or vector_info[1] != matrix_info["row_count"]
        ):
            return self.unsupported_matrix_operation("*", matrix_info["column_type"])

        result_type = self.register_vector_type(
            matrix_info["component_type"], matrix_info["column_count"]
        )
        id_value = self.get_id()
        self.emit(
            f"%{id_value} = OpVectorTimesMatrix %{result_type.id} "
            f"%{vector.id} %{matrix.id}"
        )

        spirv_id = SpirvId(id_value, result_type.type)
        self.value_types[id_value] = result_type
        self.decorate_no_contraction_result(
            id_value, "OpVectorTimesMatrix", result_type
        )
        return spirv_id

    def matrix_times_matrix(
        self,
        left: SpirvId,
        left_info,
        right: SpirvId,
        right_info,
        fallback_type: SpirvId,
    ) -> SpirvId:
        if (
            left_info["component_type"].id != right_info["component_type"].id
            or left_info["column_count"] != right_info["row_count"]
        ):
            return self.unsupported_matrix_operation("*", fallback_type)

        result_column_type = self.register_vector_type(
            left_info["component_type"], left_info["row_count"]
        )
        result_type = self.register_matrix_type(
            result_column_type, right_info["column_count"]
        )
        id_value = self.get_id()
        self.emit(
            f"%{id_value} = OpMatrixTimesMatrix %{result_type.id} "
            f"%{left.id} %{right.id}"
        )

        spirv_id = SpirvId(id_value, result_type.type)
        self.value_types[id_value] = result_type
        self.decorate_no_contraction_result(
            id_value, "OpMatrixTimesMatrix", result_type
        )
        return spirv_id

    def decorate_no_contraction_result(
        self, id_value: int, opcode: str, result_type: SpirvId
    ):
        if self.precise_expression_depth <= 0:
            return
        if opcode not in {
            "OpFAdd",
            "OpFSub",
            "OpFMul",
            "OpFDiv",
            "OpFMod",
            "OpFNegate",
            "OpDot",
            "OpMatrixTimesScalar",
            "OpMatrixTimesVector",
            "OpVectorTimesMatrix",
            "OpMatrixTimesMatrix",
        }:
            return
        if not self.is_float_like_type_id(result_type):
            return
        self.decorate_no_contraction(id_value)

    def decorate_no_contraction(self, id_value):
        id_number = id_value.id if isinstance(id_value, SpirvId) else id_value
        if id_number in self.no_contraction_ids:
            return
        self.no_contraction_ids.add(id_number)
        self.decorations.append(f"OpDecorate %{id_number} NoContraction")

    def is_float_like_type_id(self, type_id: SpirvId) -> bool:
        component_type = self.scalar_or_vector_component_type(type_id.type)
        if component_type in {"float", "double"}:
            return True

        matrix_info = self.matrix_type_info_from_type(type_id)
        if matrix_info is None:
            return False

        column_type, _ = matrix_info
        return self.is_float_like_type_id(column_type)

    def process_expression_with_precision(
        self, expr, precise: bool
    ) -> Optional[SpirvId]:
        if not precise:
            return self.process_expression(expr)

        self.precise_expression_depth += 1
        try:
            return self.process_expression(expr)
        finally:
            self.precise_expression_depth -= 1

    def logical_result_type_and_operands(
        self, left: SpirvId, right: SpirvId
    ) -> Tuple[SpirvId, SpirvId, SpirvId]:
        bool_type = self.register_primitive_type("bool")
        left_vector = self.vector_component_type_and_count(left.type.base_type)
        right_vector = self.vector_component_type_and_count(right.type.base_type)
        if left_vector is None and right_vector is None:
            return (
                bool_type,
                self.ensure_bool_value(left),
                self.ensure_bool_value(right),
            )

        component_type, component_count = left_vector or right_vector
        if component_type != "bool":
            return bool_type, left, right

        vector_operand_type = self.ensure_registered_type(
            left.type if left_vector is not None else right.type
        )

        if left_vector is not None and right_vector is None:
            if self.scalar_or_vector_component_type(right.type) == "bool":
                right = self.splat_scalar_to_vector(right, vector_operand_type)
        elif right_vector is not None and left_vector is None:
            if self.scalar_or_vector_component_type(left.type) == "bool":
                left = self.splat_scalar_to_vector(left, vector_operand_type)

        return self.register_vector_type(bool_type, component_count), left, right

    def ensure_bool_value(self, value: SpirvId) -> SpirvId:
        bool_type = self.register_primitive_type("bool")
        if value.type.base_type == "bool":
            return value
        return self.convert_value_to_type(value, bool_type)

    def comparison_result_type_and_operands(
        self, left: SpirvId, right: SpirvId
    ) -> Tuple[SpirvId, SpirvId, SpirvId]:
        bool_type = self.register_primitive_type("bool")
        left_vector = self.vector_component_type_and_count(left.type.base_type)
        right_vector = self.vector_component_type_and_count(right.type.base_type)
        if left_vector is None and right_vector is None:
            component_type_name = self.promoted_numeric_type_name(
                [left.type.base_type, right.type.base_type]
            )
            if component_type_name is not None:
                operand_type = self.register_primitive_type(component_type_name)
                left = self.convert_value_to_type(left, operand_type)
                right = self.convert_value_to_type(right, operand_type)
            return bool_type, left, right

        component_type, component_count = left_vector or right_vector
        promoted_component_type = component_type
        if left_vector is not None and right_vector is not None:
            if left_vector[1] != right_vector[1]:
                return (
                    self.register_vector_type(bool_type, component_count),
                    left,
                    right,
                )
            promoted_component_type = self.promoted_numeric_type_name(
                [left_vector[0], right_vector[0]]
            )
            if promoted_component_type is None:
                promoted_component_type = (
                    left_vector[0]
                    if left_vector[0] == right_vector[0]
                    else component_type
                )
        elif left_vector is not None:
            right_component = self.scalar_or_vector_component_type(right.type)
            promoted_component_type = self.promoted_numeric_type_name(
                [left_vector[0], right_component]
            )
            if promoted_component_type is None:
                promoted_component_type = (
                    left_vector[0]
                    if right_component == left_vector[0]
                    else component_type
                )
        elif right_vector is not None:
            left_component = self.scalar_or_vector_component_type(left.type)
            promoted_component_type = self.promoted_numeric_type_name(
                [left_component, right_vector[0]]
            )
            if promoted_component_type is None:
                promoted_component_type = (
                    right_vector[0]
                    if left_component == right_vector[0]
                    else component_type
                )

        vector_operand_type = self.register_vector_type(
            self.register_primitive_type(promoted_component_type),
            component_count,
        )

        if left_vector is not None and right_vector is None:
            left = self.convert_value_to_type(left, vector_operand_type)
            component = self.register_primitive_type(promoted_component_type)
            right = self.convert_scalar_to_type(right, component)
            if (
                self.scalar_or_vector_component_type(right.type)
                == promoted_component_type
            ):
                right = self.splat_scalar_to_vector(right, vector_operand_type)
        elif right_vector is not None and left_vector is None:
            right = self.convert_value_to_type(right, vector_operand_type)
            component = self.register_primitive_type(promoted_component_type)
            left = self.convert_scalar_to_type(left, component)
            if (
                self.scalar_or_vector_component_type(left.type)
                == promoted_component_type
            ):
                left = self.splat_scalar_to_vector(left, vector_operand_type)
        elif left_vector is not None and right_vector is not None:
            left = self.convert_value_to_type(left, vector_operand_type)
            right = self.convert_value_to_type(right, vector_operand_type)

        return self.register_vector_type(bool_type, component_count), left, right

    def align_binary_arithmetic_operands(
        self, result_type: SpirvId, left: SpirvId, right: SpirvId
    ) -> Tuple[SpirvId, SpirvId, SpirvId]:
        result_type = self.ensure_registered_type(result_type)
        left_vector = self.vector_component_type_and_count(left.type.base_type)
        right_vector = self.vector_component_type_and_count(right.type.base_type)

        if left_vector is not None and right_vector is None:
            result_vector = self.vector_component_type_and_count(
                result_type.type.base_type
            )
            vector_type = (
                result_type
                if result_vector is not None and result_vector[1] == left_vector[1]
                else self.ensure_registered_type(left.type)
            )
            component_type_name = self.vector_component_type_and_count(
                vector_type.type.base_type
            )[0]
            component_type = self.register_primitive_type(component_type_name)
            left = self.convert_value_to_type(left, vector_type)
            right = self.convert_scalar_to_type(right, component_type)
            if self.scalar_or_vector_component_type(right.type) == component_type_name:
                return (
                    vector_type,
                    left,
                    self.splat_scalar_to_vector(right, vector_type),
                )
        if right_vector is not None and left_vector is None:
            result_vector = self.vector_component_type_and_count(
                result_type.type.base_type
            )
            vector_type = (
                result_type
                if result_vector is not None and result_vector[1] == right_vector[1]
                else self.ensure_registered_type(right.type)
            )
            component_type_name = self.vector_component_type_and_count(
                vector_type.type.base_type
            )[0]
            component_type = self.register_primitive_type(component_type_name)
            right = self.convert_value_to_type(right, vector_type)
            left = self.convert_scalar_to_type(left, component_type)
            if self.scalar_or_vector_component_type(left.type) == component_type_name:
                return (
                    vector_type,
                    self.splat_scalar_to_vector(left, vector_type),
                    right,
                )
        if left_vector is not None and right_vector is not None:
            result_vector = self.vector_component_type_and_count(
                result_type.type.base_type
            )
            if (
                result_vector is not None
                and left_vector[1] == result_vector[1]
                and right_vector[1] == result_vector[1]
            ):
                left = self.convert_value_to_type(left, result_type)
                right = self.convert_value_to_type(right, result_type)
                if self.value_has_type(left, result_type) and self.value_has_type(
                    right, result_type
                ):
                    return result_type, left, right

        if left_vector is None and right_vector is None:
            left = self.convert_value_to_type(left, result_type)
            right = self.convert_value_to_type(right, result_type)

        return result_type, left, right

    def binary_expression_result_type(
        self, op: str, left_type: Optional[SpirvId], right_type: Optional[SpirvId]
    ) -> Optional[SpirvId]:
        """Infer the SPIR-V result type for a binary expression."""
        if left_type is None or right_type is None:
            return left_type or right_type

        left_type = self.ensure_registered_type(left_type)
        right_type = self.ensure_registered_type(right_type)
        left_vector = self.vector_component_type_and_count(left_type.type.base_type)
        right_vector = self.vector_component_type_and_count(right_type.type.base_type)

        if op in {"&&", "||"}:
            if left_vector is not None or right_vector is not None:
                component_count = (left_vector or right_vector)[1]
                return self.register_vector_type(
                    self.register_primitive_type("bool"), component_count
                )
            return self.register_primitive_type("bool")

        if op in {"==", "!=", "<", ">", "<=", ">="}:
            if left_vector is not None or right_vector is not None:
                component_count = (left_vector or right_vector)[1]
                return self.register_vector_type(
                    self.register_primitive_type("bool"), component_count
                )
            return self.register_primitive_type("bool")

        if op in {"&", "|", "^"}:
            if left_vector is not None or right_vector is not None:
                component_count = (left_vector or right_vector)[1]
                component_type = self.promoted_bitwise_integer_type_name(
                    [
                        (
                            left_vector[0]
                            if left_vector is not None
                            else left_type.type.base_type
                        ),
                        (
                            right_vector[0]
                            if right_vector is not None
                            else right_type.type.base_type
                        ),
                    ]
                )
                if component_type is None:
                    return left_type
                return self.register_vector_type(
                    self.register_primitive_type(component_type), component_count
                )

            scalar_type = self.promoted_bitwise_integer_type_name(
                [left_type.type.base_type, right_type.type.base_type]
            )
            if scalar_type is None:
                return left_type
            return self.register_primitive_type(scalar_type)

        if op not in {"+", "-", "*", "MULTIPLY", "/", "%"}:
            return left_type

        if left_vector is not None and right_vector is not None:
            if left_vector[1] != right_vector[1]:
                return left_type
            component_type = self.promoted_numeric_type_name(
                [left_vector[0], right_vector[0]]
            )
            if component_type is None:
                return left_type
            return self.register_vector_type(
                self.register_primitive_type(component_type), left_vector[1]
            )

        if left_vector is not None:
            component_type = self.promoted_numeric_type_name(
                [left_vector[0], right_type.type.base_type]
            )
            if component_type is None:
                return left_type
            return self.register_vector_type(
                self.register_primitive_type(component_type), left_vector[1]
            )

        if right_vector is not None:
            component_type = self.promoted_numeric_type_name(
                [left_type.type.base_type, right_vector[0]]
            )
            if component_type is None:
                return right_type
            return self.register_vector_type(
                self.register_primitive_type(component_type), right_vector[1]
            )

        scalar_type = self.promoted_numeric_type_name(
            [left_type.type.base_type, right_type.type.base_type]
        )
        if scalar_type is None:
            return left_type
        return self.register_primitive_type(scalar_type)

    def bitwise_expression_operand_type(
        self, left_type: Optional[SpirvId], right_type: Optional[SpirvId]
    ) -> Optional[SpirvId]:
        """Return a shared integer type for bitwise operands when it is knowable."""
        if left_type is None and right_type is None:
            return None

        left_type = self.ensure_registered_type(left_type) if left_type else None
        right_type = self.ensure_registered_type(right_type) if right_type else None
        left_vector = (
            self.vector_component_type_and_count(left_type.type.base_type)
            if left_type is not None
            else None
        )
        right_vector = (
            self.vector_component_type_and_count(right_type.type.base_type)
            if right_type is not None
            else None
        )

        if left_vector is not None or right_vector is not None:
            if (
                left_vector is not None
                and right_vector is not None
                and left_vector[1] != right_vector[1]
            ):
                return None
            component_count = (left_vector or right_vector)[1]
            component_names = [
                vector[0]
                for vector in (left_vector, right_vector)
                if vector is not None
            ]
            for scalar_type in (left_type, right_type):
                if scalar_type is not None:
                    scalar_component = self.normalize_primitive_name(
                        scalar_type.type.base_type
                    )
                    if scalar_component in {"int", "uint"}:
                        component_names.append(scalar_component)
            if not component_names or any(
                component not in {"int", "uint"} for component in component_names
            ):
                return None
            component_type = "uint" if "uint" in component_names else "int"
            return self.register_vector_type(
                self.register_primitive_type(component_type), component_count
            )

        type_names = [
            self.normalize_primitive_name(value.type.base_type)
            for value in (left_type, right_type)
            if value is not None
        ]
        if not type_names or any(
            type_name not in {"int", "uint"} for type_name in type_names
        ):
            return None
        return self.register_primitive_type("uint" if "uint" in type_names else "int")

    def splat_scalar_to_vector(
        self, scalar_id: SpirvId, vector_type: SpirvId
    ) -> SpirvId:
        vector_info = self.vector_component_type_and_count(vector_type.type.base_type)
        if vector_info is None:
            return scalar_id

        _, component_count = vector_info
        id_value = self.get_id()
        component_list = " ".join(f"%{scalar_id.id}" for _ in range(component_count))
        self.emit(
            f"%{id_value} = OpCompositeConstruct %{vector_type.id} {component_list}"
        )
        self.value_types[id_value] = vector_type
        return SpirvId(id_value, vector_type.type)

    def convert_scalar_to_type(self, value: SpirvId, target_type: SpirvId) -> SpirvId:
        """Convert a scalar value to a compatible scalar target type."""
        source_type_name = self.normalize_primitive_name(value.type.base_type)
        target_type_name = self.normalize_primitive_name(target_type.type.base_type)
        if source_type_name == target_type_name:
            return value

        if source_type_name == "bool" and target_type_name in self.INTEGER_TYPE_NAMES:
            true_value = self.register_constant(1, target_type)
            false_value = self.register_constant(0, target_type)
            return self.select_operation(target_type, value, true_value, false_value)

        if source_type_name in self.INTEGER_TYPE_NAMES and target_type_name == "bool":
            source_type = self.ensure_registered_type(value.type)
            zero_value = self.register_constant(0, source_type)
            bool_type = self.register_primitive_type("bool")
            return self.binary_operation("!=", bool_type, value, zero_value)

        float_types = {"float", "double"}
        integer_types = self.INTEGER_TYPE_NAMES
        scalar_types = float_types | integer_types
        if source_type_name not in scalar_types or target_type_name not in scalar_types:
            return value

        if source_type_name in float_types and target_type_name in float_types:
            opcode = "OpFConvert"
        elif source_type_name in integer_types and target_type_name in float_types:
            opcode = (
                "OpConvertUToF"
                if source_type_name in self.UNSIGNED_INTEGER_TYPES
                else "OpConvertSToF"
            )
        elif source_type_name in float_types and target_type_name in integer_types:
            opcode = (
                "OpConvertFToU"
                if target_type_name in self.UNSIGNED_INTEGER_TYPES
                else "OpConvertFToS"
            )
        elif source_type_name in integer_types and target_type_name in integer_types:
            source_width = self.integer_type_width(source_type_name)
            target_width = self.integer_type_width(target_type_name)
            if source_width == target_width:
                opcode = "OpBitcast"
            elif (
                source_type_name in self.UNSIGNED_INTEGER_TYPES
                and target_type_name in self.UNSIGNED_INTEGER_TYPES
            ):
                opcode = "OpUConvert"
            else:
                opcode = "OpSConvert"
        else:
            return value

        id_value = self.get_id()
        self.emit(f"%{id_value} = {opcode} %{target_type.id} %{value.id}")
        self.value_types[id_value] = target_type
        return SpirvId(id_value, target_type.type)

    def is_integer_type(self, spirv_type: SpirvType) -> bool:
        return (
            self.normalize_primitive_name(spirv_type.base_type)
            in self.INTEGER_TYPE_NAMES
        )

    def is_unsigned_type(self, spirv_type: SpirvType) -> bool:
        return (
            self.normalize_primitive_name(spirv_type.base_type)
            in self.UNSIGNED_INTEGER_TYPES
        )

    def integer_type_width(self, type_name: str) -> Optional[int]:
        return self.INTEGER_TYPE_WIDTHS.get(self.normalize_primitive_name(type_name))

    def unary_operation(
        self, op: str, result_type: Union[SpirvId, SpirvType], operand: SpirvId
    ) -> SpirvId:
        """Create a unary operation."""
        result_type = self.ensure_registered_type(result_type)

        if op == "+":
            spv_op = None
        elif op == "-":
            component_type = self.scalar_or_vector_component_type(result_type.type)
            if component_type == "uint":
                zero = self.default_value_for_type(result_type)
                return self.binary_operation("-", result_type, zero, operand)
            spv_op = "OpSNegate" if component_type == "int" else "OpFNegate"
        elif op == "!":
            bool_type = self.register_primitive_type("bool")
            operand_vector = self.vector_component_type_and_count(
                operand.type.base_type
            )
            if operand_vector is not None:
                _, component_count = operand_vector
                result_type = self.register_vector_type(bool_type, component_count)
                operand = self.convert_value_to_type(operand, result_type)
                if not self.value_has_type(operand, result_type):
                    self.emit(
                        "; WARNING: logical not requires a bool scalar or vector "
                        "operand; using default value"
                    )
                    return self.default_value_for_type(result_type)
            else:
                operand = self.ensure_bool_value(operand)
                result_type = bool_type
            spv_op = "OpLogicalNot"
        else:
            spv_op = {
                "~": "OpNot",
            }.get(op)

        if spv_op is None:
            return operand

        id_value = self.get_id()
        self.emit(f"%{id_value} = {spv_op} %{result_type.id} %{operand.id}")
        self.decorate_no_contraction_result(id_value, spv_op, result_type)

        spirv_id = SpirvId(id_value, result_type.type)
        self.value_types[id_value] = result_type
        return spirv_id

    def select_operation(
        self,
        result_type: SpirvId,
        condition: SpirvId,
        true_value: SpirvId,
        false_value: SpirvId,
    ) -> SpirvId:
        """Create a SPIR-V select operation for ternary expressions."""
        true_value = self.convert_value_to_type(true_value, result_type)
        false_value = self.convert_value_to_type(false_value, result_type)
        if self.vector_component_type_and_count(condition.type.base_type) is None:
            condition = self.ensure_bool_value(condition)
        if not self.can_use_select_operation(result_type, condition):
            return self.select_composite_operation(
                result_type, condition, true_value, false_value
            )

        id_value = self.get_id()
        self.emit(
            f"%{id_value} = OpSelect %{result_type.id} %{condition.id} "
            f"%{true_value.id} %{false_value.id}"
        )

        spirv_id = SpirvId(id_value, result_type.type)
        self.value_types[id_value] = result_type
        return spirv_id

    def any_bool_vector_operation(self, vector: SpirvId) -> SpirvId:
        """Reduce a SPIR-V bool vector to a scalar bool using OpAny."""
        bool_type = self.register_primitive_type("bool")
        id_value = self.get_id()
        self.emit(f"%{id_value} = OpAny %{bool_type.id} %{vector.id}")
        spirv_id = SpirvId(id_value, bool_type.type)
        self.value_types[id_value] = bool_type
        return spirv_id

    def is_select_result_type(self, result_type: SpirvId) -> bool:
        """Return whether OpSelect can directly produce this result type."""
        base_type = self.normalize_primitive_name(result_type.type.base_type)
        if base_type in {"bool", "float", "double"} | self.INTEGER_TYPE_NAMES:
            return True
        return any(
            vector_type.type.base_type == base_type
            for vector_type in self.vector_types.values()
        )

    def can_use_select_operation(
        self, result_type: SpirvId, condition: SpirvId
    ) -> bool:
        if not self.is_select_result_type(result_type):
            return False

        result_vector = self.vector_component_type_and_count(result_type.type.base_type)
        condition_vector = self.vector_component_type_and_count(
            condition.type.base_type
        )
        if result_vector is not None:
            return (
                condition_vector is not None
                and condition_vector[0] == "bool"
                and condition_vector[1] == result_vector[1]
            )

        return condition.type.base_type == "bool"

    def select_composite_operation(
        self,
        result_type: SpirvId,
        condition: SpirvId,
        true_value: SpirvId,
        false_value: SpirvId,
    ) -> SpirvId:
        """Select composite values through control flow instead of OpSelect."""
        result_variable = self.create_variable(result_type, "Function")
        merge_label = SpirvId(self.get_id(), SpirvType("label"))
        then_label = SpirvId(self.get_id(), SpirvType("label"))
        else_label = SpirvId(self.get_id(), SpirvType("label"))
        condition = self.ensure_bool_value(condition)

        self.create_selection_merge(merge_label)
        self.create_conditional_branch(condition, then_label, else_label)

        self.emit(f"%{then_label.id} = OpLabel")
        self.current_label = then_label.id
        self.store_to_variable(result_variable, true_value)
        if not self.current_block_has_terminator():
            self.create_branch(merge_label)

        self.emit(f"%{else_label.id} = OpLabel")
        self.current_label = else_label.id
        self.store_to_variable(result_variable, false_value)
        if not self.current_block_has_terminator():
            self.create_branch(merge_label)

        self.emit(f"%{merge_label.id} = OpLabel")
        self.current_label = merge_label.id
        return self.load_from_variable(result_variable, result_type)

    def call_function(
        self, function_name: str, args: List[SpirvId]
    ) -> Optional[SpirvId]:
        """Call a function with arguments."""
        function_reference = self.resolve_function_reference(function_name)
        if function_reference is None:
            return self.call_builtin_function(function_name, args)

        function_id, (return_type, param_types) = function_reference
        self.merge_function_interface_variables_from_callee(
            function_name, function_id.id
        )
        prepared_args = []
        for index, arg in enumerate(args):
            if index < len(param_types):
                arg = self.prepare_function_call_argument(arg, param_types[index])
            prepared_args.append(arg)
        args = prepared_args

        id_value = self.get_id()

        arg_list = " ".join([f"%{arg.id}" for arg in args])
        self.emit(
            f"%{id_value} = OpFunctionCall %{return_type.id} %{function_id.id} {arg_list}"
        )

        spirv_id = SpirvId(id_value, return_type.type)
        self.value_types[id_value] = return_type
        return spirv_id

    def resource_function_names(self):
        return {
            "imageLoad",
            "imageStore",
            "imageAtomicAdd",
            "imageAtomicMin",
            "imageAtomicMax",
            "imageAtomicAnd",
            "imageAtomicOr",
            "imageAtomicXor",
            "imageAtomicExchange",
            "imageAtomicCompSwap",
            "atomicAdd",
            "atomicMin",
            "atomicMax",
            "atomicAnd",
            "atomicOr",
            "atomicXor",
            "atomicExchange",
            "atomicCompSwap",
            "buffer_load",
            "buffer_load2",
            "buffer_load3",
            "buffer_load4",
            "buffer_store",
            "buffer_store2",
            "buffer_store3",
            "buffer_store4",
            "buffer_dimensions",
            "buffer_append",
            "buffer_consume",
            "buffer_increment_counter",
            "buffer_decrement_counter",
            "texture",
            "texture2D",
            "textureCube",
            "textureProj",
            "textureProjOffset",
            "textureProjLod",
            "textureProjLodOffset",
            "textureProjGrad",
            "textureProjGradOffset",
            "textureCompare",
            "textureCompareLod",
            "textureCompareLodOffset",
            "textureCompareGrad",
            "textureCompareGradOffset",
            "textureCompareOffset",
            "textureCompareProj",
            "textureCompareProjOffset",
            "textureCompareProjLod",
            "textureCompareProjLodOffset",
            "textureCompareProjGrad",
            "textureCompareProjGradOffset",
            "textureGatherCompare",
            "textureGatherCompareOffset",
            "textureLod",
            "textureLodOffset",
            "textureGrad",
            "textureGradOffset",
            "textureOffset",
            "textureGather",
            "textureGatherOffset",
            "textureGatherOffsets",
            "texelFetch",
            "texelFetchOffset",
            "textureSize",
            "imageSize",
            "textureSamples",
            "imageSamples",
            "textureQueryLevels",
            "textureQueryLod",
        }

    def image_atomic_function_names(self):
        return {
            "imageAtomicAdd",
            "imageAtomicMin",
            "imageAtomicMax",
            "imageAtomicAnd",
            "imageAtomicOr",
            "imageAtomicXor",
            "imageAtomicExchange",
            "imageAtomicCompSwap",
        }

    def buffer_atomic_function_names(self):
        return {
            "atomicAdd",
            "atomicMin",
            "atomicMax",
            "atomicAnd",
            "atomicOr",
            "atomicXor",
            "atomicExchange",
            "atomicCompSwap",
        }

    def buffer_function_names(self):
        return {
            "buffer_load",
            "buffer_load2",
            "buffer_load3",
            "buffer_load4",
            "buffer_store",
            "buffer_store2",
            "buffer_store3",
            "buffer_store4",
            "buffer_dimensions",
            "buffer_append",
            "buffer_consume",
            "buffer_increment_counter",
            "buffer_decrement_counter",
        }

    def resource_query_size_result_type(self, metadata) -> SpirvId:
        dim = metadata.get("dim", "2D") if metadata else "2D"
        component_count = {
            "1D": 1,
            "Buffer": 1,
            "2D": 2,
            "Rect": 2,
            "Cube": 2,
            "3D": 3,
        }.get(dim, 2)

        if metadata and metadata.get("arrayed"):
            component_count += 1

        component_count = min(max(component_count, 1), 4)
        int_type = self.register_primitive_type("int")
        if component_count <= 1:
            return int_type
        return self.register_vector_type(int_type, component_count)

    def resource_query_lod_coordinate_components(self, metadata) -> int:
        dim = metadata.get("dim", "2D") if metadata else "2D"
        return {
            "1D": 1,
            "Buffer": 1,
            "2D": 2,
            "Rect": 2,
            "3D": 3,
            "Cube": 3,
        }.get(dim, 2)

    def resource_query_size_default_value(self, metadata) -> SpirvId:
        if metadata and metadata.get("kind") in {"sampled_image", "storage_image"}:
            return self.default_value_for_type(
                self.resource_query_size_result_type(metadata)
            )
        return self.register_constant(0, self.register_primitive_type("int"))

    def validate_resource_query_lod_operand(
        self, function_name: str, lod_id: SpirvId
    ) -> bool:
        if self.integer_value_component_count(lod_id) == 1:
            return True

        self.emit(f"; WARNING: {function_name} requires a scalar integer LOD operand")
        return False

    def resource_query_lod_default_value(self) -> SpirvId:
        float_type = self.register_primitive_type("float")
        result_type = self.register_vector_type(float_type, 2)
        return self.default_value_for_type(result_type)

    def floating_value_component_count(self, value_id: SpirvId) -> Optional[int]:
        value_type = self.value_types.get(
            value_id.id
        ) or self.find_registered_type_by_base(value_id.type.base_type)
        type_name = (
            value_type.type.base_type
            if value_type is not None
            else value_id.type.base_type
        )
        vector_info = self.vector_component_type_and_count(type_name)
        if vector_info is not None:
            component_type_name, component_count = vector_info
            if component_type_name in {"float", "double"}:
                return component_count
            return None

        if self.normalize_primitive_name(type_name) in {"float", "double"}:
            return 1
        return None

    def validate_resource_query_lod_coordinate(
        self, function_name: str, metadata, coord_id: SpirvId
    ) -> bool:
        required_count = self.resource_query_lod_coordinate_components(metadata)
        actual_count = self.floating_value_component_count(coord_id)
        if actual_count is not None and actual_count >= required_count:
            return True

        self.emit(
            f"; WARNING: {function_name} requires a {required_count}-component "
            "floating-point coordinate operand"
        )
        return False

    def trim_image_query_lod_coordinate(self, coord_id: SpirvId, metadata) -> SpirvId:
        required_count = self.resource_query_lod_coordinate_components(metadata)
        vector_info = self.vector_component_type_and_count(coord_id.type.base_type)
        if vector_info is None:
            return coord_id

        component_type_name, source_count = vector_info
        if source_count <= required_count:
            return coord_id

        component_type = self.register_primitive_type(component_type_name)
        components = [
            self.composite_extract(coord_id, component_type, index)
            for index in range(required_count)
        ]
        if required_count <= 1:
            return components[0]

        result_type = self.register_vector_type(component_type, required_count)
        id_value = self.get_id()
        component_list = " ".join(f"%{component.id}" for component in components)
        self.emit(
            f"%{id_value} = OpCompositeConstruct %{result_type.id} {component_list}"
        )
        self.value_types[id_value] = result_type
        return SpirvId(id_value, result_type.type)

    def texture_gather_offsets_arguments(
        self, extra_args: List[SpirvId]
    ) -> Tuple[List[SpirvId], Optional[SpirvId]]:
        if len(extra_args) >= 4:
            component_id = extra_args[4] if len(extra_args) >= 5 else None
            return extra_args[:4], component_id

        component_id = extra_args[1] if len(extra_args) >= 2 else None
        if not extra_args:
            return [], component_id

        offsets_value = extra_args[0]
        offsets_type = self.value_types.get(offsets_value.id)
        element_type = self.array_element_type_from_type(offsets_type)
        if element_type is not None:
            return [
                self.composite_extract(offsets_value, element_type, index)
                for index in range(4)
            ], component_id

        return [offsets_value] * 4, component_id

    def emit_image_gather(
        self,
        sampled_image_id: SpirvId,
        coord_id: SpirvId,
        component_id: SpirvId,
        result_type: SpirvId,
        image_operand: Optional[Union[SpirvId, str]] = None,
    ) -> SpirvId:
        image_operands = ""
        if isinstance(image_operand, str):
            image_operands = f" {image_operand}"
        elif image_operand is not None:
            image_operands = f" {self.image_offset_operand(image_operand)}"

        id_value = self.get_id()
        self.emit(
            f"%{id_value} = OpImageGather %{result_type.id} "
            f"%{sampled_image_id.id} %{coord_id.id} %{component_id.id}"
            f"{image_operands}"
        )
        self.value_types[id_value] = result_type
        return SpirvId(id_value, result_type.type)

    def emit_texture_gather_offsets(
        self,
        sampled_image_id: SpirvId,
        coord_id: SpirvId,
        extra_args: List[SpirvId],
        metadata,
        int_type: SpirvId,
    ) -> SpirvId:
        result_type = self.resource_access_result_type(metadata)
        offsets, component_id = self.texture_gather_offsets_arguments(extra_args)
        if len(offsets) != 4:
            self.emit("; WARNING: textureGatherOffsets requires four offset operands")
            return self.default_value_for_type(result_type)

        if component_id is None:
            component_id = self.register_constant(0, int_type)

        for offset_id in offsets:
            if not self.validate_sampled_texture_offset_operand(
                "textureGatherOffsets", metadata, offset_id
            ):
                return self.default_value_for_type(result_type)
        if not self.validate_texture_gather_component_operand(
            "textureGatherOffsets", component_id
        ):
            return self.default_value_for_type(result_type)

        const_offsets_operand = self.image_const_offsets_operand(offsets)
        if const_offsets_operand is not None:
            return self.emit_image_gather(
                sampled_image_id,
                coord_id,
                component_id,
                result_type,
                const_offsets_operand,
            )

        component_type = self.register_primitive_type(
            metadata.get("component_type", "float")
        )
        gathered_components = []
        for index, offset_id in enumerate(offsets):
            gathered = self.emit_image_gather(
                sampled_image_id,
                coord_id,
                component_id,
                result_type,
                offset_id,
            )
            gathered_components.append(
                self.composite_extract(gathered, component_type, index)
            )

        id_value = self.get_id()
        component_list = " ".join(
            f"%{component.id}" for component in gathered_components
        )
        self.emit(
            f"%{id_value} = OpCompositeConstruct %{result_type.id} " f"{component_list}"
        )
        self.value_types[id_value] = result_type
        return SpirvId(id_value, result_type.type)

    def extract_image_from_sampled_image(
        self, sampled_image_id: SpirvId, metadata
    ) -> Optional[SpirvId]:
        image_type_id = metadata.get("image_type_id") if metadata else None
        image_type = (
            self.find_registered_type_by_id(image_type_id)
            if image_type_id is not None
            else None
        )
        if image_type is None:
            self.emit("; WARNING: Could not determine image type for sampled image")
            return None

        id_value = self.get_id()
        self.emit(f"%{id_value} = OpImage %{image_type.id} %{sampled_image_id.id}")
        self.value_types[id_value] = image_type
        return SpirvId(id_value, image_type.type)

    def image_operand_for_query(
        self, resource_id: SpirvId, metadata
    ) -> Optional[SpirvId]:
        if not metadata:
            return None
        if metadata.get("kind") == "sampled_image":
            return self.extract_image_from_sampled_image(resource_id, metadata)
        if metadata.get("kind") == "storage_image":
            return resource_id
        return None

    def shadow_compare_operands(
        self, function_name: str, args: List[SpirvId], extra_arg_count: int
    ):
        coord_index = 1
        if len(args) > 1:
            sampler_metadata = self.resource_metadata_for_value(args[1])
            if sampler_metadata and sampler_metadata.get("kind") == "sampler":
                coord_index = 2

        required_arg_count = coord_index + 2 + extra_arg_count
        if len(args) < required_arg_count:
            self.emit(
                f"; WARNING: {function_name} requires a shadow texture, "
                "coordinate, depth, and operation operands"
            )
            return None

        sampled_image_id = args[0]
        coord_id = args[coord_index]
        depth_id = args[coord_index + 1]
        extra_args = args[coord_index + 2 : required_arg_count]
        metadata = self.resource_metadata_for_value(sampled_image_id)
        if not metadata or metadata.get("kind") != "sampled_image":
            self.emit(
                f"; WARNING: {function_name} requires a shadow sampled image operand"
            )
            return None
        if not self.validate_non_multisample_sampled_image(function_name, metadata):
            return None
        if int(metadata.get("depth", 0)) != 1:
            self.emit(
                f"; WARNING: {function_name} requires a shadow sampled image operand"
            )
            return None
        if len(args) > required_arg_count:
            self.emit(self.shadow_compare_excess_operand_warning(function_name))
            return None

        return sampled_image_id, coord_id, depth_id, extra_args

    def shadow_compare_excess_operand_warning(self, function_name: str) -> str:
        extra_operands = {
            "textureCompare": "",
            "textureCompareOffset": "offset",
            "textureCompareLod": "LOD",
            "textureCompareLodOffset": "LOD and offset",
            "textureCompareGrad": "dx and dy gradient",
            "textureCompareGradOffset": "dx and dy gradient and offset",
            "textureCompareProj": "",
            "textureCompareProjOffset": "offset",
            "textureCompareProjLod": "LOD",
            "textureCompareProjLodOffset": "LOD and offset",
            "textureCompareProjGrad": "dx and dy gradient",
            "textureCompareProjGradOffset": "dx and dy gradient and offset",
            "textureGatherCompare": "",
            "textureGatherCompareOffset": "offset",
        }[function_name]
        if extra_operands:
            return (
                f"; WARNING: {function_name} accepts only texture, optional "
                "sampler, coordinate, depth, and "
                f"{extra_operands} operands"
            )
        return (
            f"; WARNING: {function_name} accepts only texture, optional sampler, "
            "coordinate, and depth operands"
        )

    def sampled_texture_operands(
        self, function_name: str, args: List[SpirvId], extra_arg_count: int = 0
    ):
        coord_index = 1
        sampler_id = None
        if len(args) > 1:
            sampler_metadata = self.resource_metadata_for_value(args[1])
            if sampler_metadata and sampler_metadata.get("kind") == "sampler":
                coord_index = 2
                sampler_id = args[1]

        required_arg_count = coord_index + 1 + extra_arg_count
        if len(args) < required_arg_count:
            self.emit(
                f"; WARNING: {function_name} requires a texture, coordinate, "
                "and operation operands"
            )
            return None

        sampled_image_id = args[0]
        coord_id = args[coord_index]
        extra_args = args[coord_index + 1 :]
        metadata = self.resource_metadata_for_value(sampled_image_id)
        if metadata and metadata.get("kind") == "texture":
            if sampler_id is None:
                self.emit(
                    f"; WARNING: {function_name} requires a sampler for separate "
                    "texture operands"
                )
                return None
            sampled_image_id = self.combine_texture_and_sampler(
                function_name, sampled_image_id, sampler_id, metadata
            )
            if sampled_image_id is None:
                return None
            metadata = self.resource_metadata_for_value(sampled_image_id)

        if not metadata or metadata.get("kind") != "sampled_image":
            self.emit(f"; WARNING: {function_name} requires a sampled image operand")
            return None

        return sampled_image_id, coord_id, extra_args, metadata

    def call_resource_constructor(
        self, type_name: str, args: List[SpirvId]
    ) -> Optional[SpirvId]:
        metadata = self.resource_type_info(type_name)
        if metadata is None or not args:
            return None

        first_metadata = self.resource_metadata_for_value(args[0])
        if metadata.get("kind") == "sampled_image":
            if first_metadata and first_metadata.get("kind") == "sampled_image":
                return args[0]
            if (
                len(args) >= 2
                and first_metadata
                and first_metadata.get("kind") == "texture"
            ):
                return self.combine_texture_and_sampler(
                    type_name, args[0], args[1], first_metadata
                )

        if metadata.get("kind") == "texture":
            if first_metadata and first_metadata.get("kind") in {
                "texture",
                "sampled_image",
            }:
                return args[0]

        return None

    def combine_texture_and_sampler(
        self,
        function_name: str,
        texture_id: SpirvId,
        sampler_id: SpirvId,
        texture_metadata,
    ) -> Optional[SpirvId]:
        sampler_metadata = self.resource_metadata_for_value(sampler_id)
        if not sampler_metadata or sampler_metadata.get("kind") != "sampler":
            self.emit(f"; WARNING: {function_name} requires a sampler operand")
            return None

        image_type_id = texture_metadata.get("image_type_id")
        if image_type_id is None:
            image_type_id = self.value_types.get(texture_id.id, texture_id).id
        sampled_image_type = self.register_sampled_image_type_for_image(
            image_type_id, texture_metadata
        )
        if sampled_image_type is None:
            self.emit(
                f"; WARNING: {function_name} could not construct a sampled image "
                "from separate texture and sampler operands"
            )
            return None

        id_value = self.get_id()
        self.emit(
            f"%{id_value} = OpSampledImage %{sampled_image_type.id} "
            f"%{texture_id.id} %{sampler_id.id}"
        )
        self.value_types[id_value] = sampled_image_type
        self.resource_type_metadata[id_value] = (
            self.sampled_image_metadata_from_texture(
                sampled_image_type, texture_metadata
            )
        )
        sampled_image = SpirvId(id_value, sampled_image_type.type)
        if self.is_non_uniform_value(texture_id) or self.is_non_uniform_value(
            sampler_id
        ):
            self.mark_non_uniform_result(sampled_image)
        return sampled_image

    def register_sampled_image_type_for_image(
        self, image_type_id: int, texture_metadata
    ) -> Optional[SpirvId]:
        image_type = self.find_registered_type_by_id(image_type_id)
        if image_type is None:
            return None

        type_name = f"{texture_metadata.get('type_name', 'texture')}_sampled"
        cache_key = (
            type_name,
            "sampled_image",
            texture_metadata.get("component_type", "float"),
            texture_metadata.get("format", "Unknown"),
            image_type_id,
        )
        if cache_key in self.resource_types:
            return self.resource_types[cache_key]

        id_value = self.get_id()
        self.emit(f"%{id_value} = OpTypeSampledImage %{image_type.id}")
        sampled_image_type = SpirvId(id_value, SpirvType(type_name), type_name)
        self.resource_types[cache_key] = sampled_image_type
        self.resource_type_metadata[id_value] = (
            self.sampled_image_metadata_from_texture(
                sampled_image_type, texture_metadata
            )
        )
        return sampled_image_type

    def sampled_image_metadata_from_texture(self, sampled_image_type, texture_metadata):
        metadata = dict(texture_metadata)
        metadata["kind"] = "sampled_image"
        metadata["type_name"] = sampled_image_type.type.base_type
        return metadata

    def sampled_texture_excess_operand_warning(self, function_name: str) -> str:
        operation_operands = {
            "textureLod": "LOD",
            "textureLodOffset": "LOD and offset",
            "textureGrad": "dx and dy gradient",
            "textureGradOffset": "dx and dy gradient and offset",
            "textureGatherOffsets": (
                "one offsets value or four offsets and optional component"
            ),
            "texelFetch": "LOD/sample",
            "texelFetchOffset": "LOD/sample and offset",
        }[function_name]
        return (
            f"; WARNING: {function_name} accepts only texture, optional sampler, "
            "coordinate, and "
            f"{operation_operands} operands"
        )

    def is_scalar_numeric_value(self, value_id: SpirvId) -> bool:
        value_type = self.value_types.get(
            value_id.id
        ) or self.find_registered_type_by_base(value_id.type.base_type)
        if value_type is None:
            return False
        if self.vector_component_type_and_count(value_type.type.base_type) is not None:
            return False
        return self.normalize_primitive_name(value_type.type.base_type) in {
            "int",
            "uint",
            "float",
            "double",
        }

    def texture_bias_operand(
        self, function_name: str, bias_id: SpirvId
    ) -> Optional[str]:
        if not self.is_scalar_numeric_value(bias_id):
            self.emit(f"; WARNING: {function_name} requires a scalar numeric bias")
            return None
        if self.requires_explicit_lod_sampling():
            self.emit(
                f"; WARNING: {function_name} bias is not valid for explicit-lod "
                "SPIR-V sampling"
            )
            return None
        return f"Bias %{bias_id.id}"

    def sampled_texture_axis_component_count(self, metadata) -> int:
        dim = metadata.get("dim", "2D") if metadata else "2D"
        return {
            "1D": 1,
            "Buffer": 1,
            "2D": 2,
            "Rect": 2,
            "3D": 3,
            "Cube": 3,
        }.get(dim, 2)

    def sampled_texture_coordinate_component_count(self, metadata) -> int:
        component_count = self.sampled_texture_axis_component_count(metadata)
        if metadata and metadata.get("arrayed"):
            component_count += 1
        return component_count

    def validate_sampled_texture_coordinate(
        self, function_name: str, metadata, coord_id: SpirvId, *, integer: bool = False
    ) -> bool:
        expected_count = self.sampled_texture_coordinate_component_count(metadata)
        actual_count = (
            self.integer_value_component_count(coord_id)
            if integer
            else self.floating_value_component_count(coord_id)
        )
        if actual_count == expected_count:
            return True

        coordinate_kind = "integer" if integer else "floating-point"
        self.emit(
            f"; WARNING: {function_name} requires a {expected_count}-component "
            f"{coordinate_kind} coordinate operand"
        )
        return False

    def validate_non_multisample_sampled_image(
        self, function_name: str, metadata
    ) -> bool:
        if metadata and metadata.get("multisampled"):
            self.emit(f"; WARNING: {function_name} is not valid for multisample images")
            return False
        return True

    def validate_sampled_texture_lod_operand(
        self, function_name: str, lod_id: SpirvId
    ) -> bool:
        if self.floating_value_component_count(lod_id) == 1:
            return True

        self.emit(
            f"; WARNING: {function_name} requires a scalar floating-point LOD operand"
        )
        return False

    def validate_shadow_compare_depth_operand(
        self, function_name: str, depth_id: SpirvId
    ) -> bool:
        if self.floating_value_component_count(depth_id) == 1:
            return True

        self.emit(
            f"; WARNING: {function_name} requires a scalar floating-point depth operand"
        )
        return False

    def validate_sampled_texture_fetch_operand(
        self, function_name: str, operand_id: SpirvId, operand_name: str
    ) -> bool:
        if self.integer_value_component_count(operand_id) == 1:
            return True

        self.emit(
            f"; WARNING: {function_name} requires a scalar integer {operand_name} operand"
        )
        return False

    def validate_sampled_texture_gradient_operand(
        self, function_name: str, metadata, gradient_id: SpirvId, operand_name: str
    ) -> bool:
        expected_count = self.sampled_texture_axis_component_count(metadata)
        if self.floating_value_component_count(gradient_id) == expected_count:
            return True

        self.emit(
            f"; WARNING: {function_name} requires a {expected_count}-component "
            f"floating-point {operand_name} gradient operand"
        )
        return False

    def validate_sampled_texture_offset_operand(
        self, function_name: str, metadata, offset_id: SpirvId
    ) -> bool:
        if metadata and metadata.get("dim") == "Cube":
            self.emit(
                f"; WARNING: {function_name} offsets are not valid for cube images"
            )
            return False

        expected_count = self.sampled_texture_axis_component_count(metadata)
        if self.integer_value_component_count(offset_id) == expected_count:
            return True

        self.emit(
            f"; WARNING: {function_name} requires a {expected_count}-component "
            "integer offset operand"
        )
        return False

    def validate_texture_gather_component_operand(
        self, function_name: str, component_id: SpirvId
    ) -> bool:
        if self.integer_value_component_count(component_id) == 1:
            return True

        self.emit(
            f"; WARNING: {function_name} requires a scalar integer component operand"
        )
        return False

    def projected_coordinate_axes(self, metadata) -> int:
        dim = metadata.get("dim", "2D") if metadata else "2D"
        return {
            "1D": 1,
            "Buffer": 1,
            "2D": 2,
            "Rect": 2,
            "3D": 3,
            "Cube": 3,
        }.get(dim, 2)

    def project_texture_coordinate(
        self, function_name: str, coord_id: SpirvId, metadata
    ) -> Optional[SpirvId]:
        vector_info = self.vector_component_type_and_count(coord_id.type.base_type)
        if vector_info is None:
            self.emit(
                f"; WARNING: {function_name} requires a projected vector coordinate"
            )
            return None

        component_type_name, source_count = vector_info
        if component_type_name not in {"float", "double"}:
            self.emit(
                f"; WARNING: {function_name} requires a floating-point projected coordinate"
            )
            return None

        axis_count = self.projected_coordinate_axes(metadata)
        output_count = axis_count + (1 if metadata and metadata.get("arrayed") else 0)
        if source_count < output_count + 1:
            self.emit(
                f"; WARNING: {function_name} requires a projection component after "
                "the texture coordinate"
            )
            return None

        component_type = self.register_primitive_type(component_type_name)
        projection = self.composite_extract(coord_id, component_type, source_count - 1)
        components = []
        for index in range(axis_count):
            component = self.composite_extract(coord_id, component_type, index)
            components.append(
                self.binary_operation("/", component_type, component, projection)
            )

        if metadata and metadata.get("arrayed"):
            components.append(
                self.composite_extract(coord_id, component_type, axis_count)
            )

        if len(components) == 1:
            return components[0]

        result_type = self.register_vector_type(component_type, len(components))
        return self.composite_construct(result_type, components)

    def requires_explicit_lod_sampling(self) -> bool:
        return self.current_execution_model in {"GLCompute", "MeshEXT", "TaskEXT"}

    def default_lod_operand(self) -> str:
        lod_id = self.register_constant(0.0, self.register_primitive_type("float"))
        return f"Lod %{lod_id.id}"

    def image_atomic_operand_names(
        self, function_name: str, multisampled: bool
    ) -> List[str]:
        operands = ["image", "coordinate"]
        if multisampled:
            operands.append("sample")
        if function_name == "imageAtomicCompSwap":
            operands.append("compare")
        operands.append("value")
        return operands

    def image_atomic_operand_description(
        self, function_name: str, multisampled: bool
    ) -> str:
        operands = self.image_atomic_operand_names(function_name, multisampled)
        return ", ".join(operands[:-1]) + f", and {operands[-1]} operands"

    def call_image_atomic_function(
        self, function_name: str, args: List[SpirvId]
    ) -> Optional[SpirvId]:
        if len(args) < 3:
            self.emit(
                f"; WARNING: {function_name} requires image, coordinate, "
                "and value operands"
            )
            return self.register_constant(0, self.register_primitive_type("uint"))

        image_pointer = args[0]
        coord_id = args[1]
        metadata = self.resource_metadata_for_pointer(image_pointer)
        if not metadata or metadata.get("kind") != "storage_image":
            self.emit(
                f"; WARNING: {function_name} requires a storage image pointer operand"
            )
            return self.register_constant(0, self.register_primitive_type("uint"))

        component_type_name = metadata.get("component_type", "uint")
        result_type = self.register_primitive_type(component_type_name)
        is_multisampled = bool(metadata.get("multisampled"))
        expected_operands = self.image_atomic_operand_names(
            function_name, is_multisampled
        )
        if len(args) > len(expected_operands):
            self.emit(
                f"; WARNING: {function_name} accepts only "
                f"{self.image_atomic_operand_description(function_name, is_multisampled)}"
            )
            return self.default_value_for_type(result_type)

        if metadata.get("readonly") or metadata.get("writeonly"):
            self.emit(f"; WARNING: {function_name} requires a read-write storage image")
            return self.default_value_for_type(result_type)

        if not self.validate_storage_image_coordinate(
            function_name, metadata, coord_id
        ):
            return self.default_value_for_type(result_type)

        if component_type_name not in {"int", "uint"}:
            self.emit(f"; WARNING: {function_name} requires an integer storage image")
            return self.register_constant(0, self.register_primitive_type("uint"))

        if int(metadata.get("component_count", 1)) != 1:
            self.emit(
                f"; WARNING: {function_name} requires a scalar storage image format"
            )
            return self.register_constant(0, self.register_primitive_type("uint"))

        value_arg_index = 2
        if is_multisampled:
            if len(args) < 4:
                self.emit(f"; WARNING: {function_name} requires a sample operand")
                return self.register_constant(
                    0, self.register_primitive_type(component_type_name)
                )
            sample_id = args[2]
            if not self.validate_storage_image_sample(function_name, sample_id):
                return self.default_value_for_type(result_type)
            value_arg_index = 3
        else:
            sample_id = self.register_constant(0, self.register_primitive_type("uint"))

        if function_name == "imageAtomicCompSwap":
            if len(args) <= value_arg_index + 1:
                self.emit(
                    f"; WARNING: {function_name} requires compare and value operands"
                )
                return self.register_constant(
                    0, self.register_primitive_type(component_type_name)
                )
            comparator_id = args[value_arg_index]
            value_id = args[value_arg_index + 1]
        else:
            value_id = args[value_arg_index]
            comparator_id = None

        value_id = self.convert_storage_image_value_operand(
            function_name, "value", value_id, result_type
        )
        if value_id is None:
            return self.default_value_for_type(result_type)
        if comparator_id is not None:
            comparator_id = self.convert_storage_image_value_operand(
                function_name, "compare", comparator_id, result_type
            )
            if comparator_id is None:
                return self.default_value_for_type(result_type)

        pointer_type = self.register_pointer_type(result_type, "Image")
        texel_pointer_id = self.get_id()
        self.emit(
            f"%{texel_pointer_id} = OpImageTexelPointer %{pointer_type.id} "
            f"%{image_pointer.id} %{coord_id.id} %{sample_id.id}"
        )
        self.variable_value_types[texel_pointer_id] = result_type

        atomic_operation = {
            "imageAtomicAdd": "OpAtomicIAdd",
            "imageAtomicAnd": "OpAtomicAnd",
            "imageAtomicOr": "OpAtomicOr",
            "imageAtomicXor": "OpAtomicXor",
            "imageAtomicExchange": "OpAtomicExchange",
        }.get(function_name)
        if atomic_operation is None and function_name == "imageAtomicMin":
            atomic_operation = (
                "OpAtomicSMin" if component_type_name == "int" else "OpAtomicUMin"
            )
        if atomic_operation is None and function_name == "imageAtomicMax":
            atomic_operation = (
                "OpAtomicSMax" if component_type_name == "int" else "OpAtomicUMax"
            )

        scope = self.spirv_scope_constant("Device")
        semantics = self.spirv_memory_semantics_constant()
        id_value = self.get_id()
        if function_name == "imageAtomicCompSwap":
            self.emit(
                f"%{id_value} = OpAtomicCompareExchange %{result_type.id} "
                f"%{texel_pointer_id} %{scope.id} %{semantics.id} %{semantics.id} "
                f"%{value_id.id} %{comparator_id.id}"
            )
        else:
            self.emit(
                f"%{id_value} = {atomic_operation} %{result_type.id} "
                f"%{texel_pointer_id} %{scope.id} %{semantics.id} %{value_id.id}"
            )

        self.value_types[id_value] = result_type
        return SpirvId(id_value, result_type.type)

    def default_value_for_buffer_atomic_failure(
        self,
        function_name: str,
        args: List[SpirvId],
        target_type: Optional[SpirvId] = None,
    ) -> SpirvId:
        value_index = 2 if function_name == "atomicCompSwap" and len(args) > 2 else 1
        if len(args) > value_index:
            value_type = self.value_types.get(args[value_index].id)
            if value_type is None:
                value_type = self.find_registered_type_by_base(
                    args[value_index].type.base_type
                )
            if value_type is not None:
                return self.default_value_for_type(value_type)

        if target_type is not None:
            return self.default_value_for_type(target_type)

        return self.register_constant(0, self.register_primitive_type("uint"))

    def default_value_for_buffer_load_failure(self, args: List[SpirvId]) -> SpirvId:
        if args:
            metadata = self.structured_buffer_metadata_for_pointer(args[0])
            if metadata is not None:
                element_type = metadata.get("element_type")
                if element_type is not None:
                    return self.default_value_for_type(element_type)
        return self.register_constant(0.0, self.register_primitive_type("float"))

    def call_buffer_atomic_function(
        self, function_name: str, args: List[SpirvId]
    ) -> Optional[SpirvId]:
        if len(args) < 2:
            self.emit(f"; WARNING: {function_name} requires target and value operands")
            return self.register_constant(0, self.register_primitive_type("uint"))

        expected_arg_count = 3 if function_name == "atomicCompSwap" else 2
        if len(args) > expected_arg_count:
            if function_name == "atomicCompSwap":
                self.emit(
                    f"; WARNING: {function_name} accepts only target, compare, "
                    "and value operands"
                )
            else:
                self.emit(
                    f"; WARNING: {function_name} accepts only target and value operands"
                )
            return self.default_value_for_buffer_atomic_failure(function_name, args)

        target_pointer = args[0]
        target_type = self.variable_value_types.get(target_pointer.id)
        if target_type is None and target_pointer.type.storage_class:
            target_type = self.find_registered_type_by_base(
                target_pointer.type.base_type.replace("ptr_", "", 1)
            )
        if target_type is None:
            self.emit(f"; WARNING: {function_name} requires an addressable target")
            return self.default_value_for_buffer_atomic_failure(function_name, args)

        metadata = self.storage_buffer_access_metadata_for_pointer(target_pointer)
        if metadata is None:
            self.emit(f"; WARNING: {function_name} requires a storage buffer target")
            return self.default_value_for_buffer_atomic_failure(
                function_name, args, target_type
            )
        if metadata.get("readonly") or metadata.get("writeonly"):
            self.emit(
                f"; WARNING: {function_name} requires a read-write storage buffer"
            )
            return self.default_value_for_buffer_atomic_failure(
                function_name, args, target_type
            )

        type_name = self.normalize_primitive_name(target_type.type.base_type)
        if self.vector_type_info_from_type(target_type) is not None:
            self.emit(
                f"; WARNING: {function_name} requires a scalar int or uint buffer member"
            )
            return self.default_value_for_buffer_atomic_failure(
                function_name, args, target_type
            )
        if self.matrix_type_info_from_type(target_type) is not None:
            self.emit(
                f"; WARNING: {function_name} requires a scalar int or uint buffer member"
            )
            return self.default_value_for_buffer_atomic_failure(
                function_name, args, target_type
            )
        if type_name not in {"int", "uint"}:
            self.emit(
                f"; WARNING: {function_name} currently supports only int or uint "
                "buffer members"
            )
            return self.default_value_for_buffer_atomic_failure(
                function_name, args, target_type
            )

        if function_name == "atomicCompSwap":
            if len(args) < 3:
                self.emit(
                    f"; WARNING: {function_name} requires compare and value operands"
                )
                return self.default_value_for_buffer_atomic_failure(
                    function_name, args, target_type
                )
            comparator_id = self.convert_value_to_type(args[1], target_type)
            value_id = self.convert_value_to_type(args[2], target_type)
        else:
            comparator_id = None
            value_id = self.convert_value_to_type(args[1], target_type)

        atomic_operation = {
            "atomicAdd": "OpAtomicIAdd",
            "atomicAnd": "OpAtomicAnd",
            "atomicOr": "OpAtomicOr",
            "atomicXor": "OpAtomicXor",
            "atomicExchange": "OpAtomicExchange",
        }.get(function_name)
        if atomic_operation is None and function_name == "atomicMin":
            atomic_operation = "OpAtomicSMin" if type_name == "int" else "OpAtomicUMin"
        if atomic_operation is None and function_name == "atomicMax":
            atomic_operation = "OpAtomicSMax" if type_name == "int" else "OpAtomicUMax"

        scope = self.spirv_scope_constant("Device")
        semantics = self.spirv_memory_semantics_constant()
        id_value = self.get_id()
        if function_name == "atomicCompSwap":
            self.emit(
                f"%{id_value} = OpAtomicCompareExchange %{target_type.id} "
                f"%{target_pointer.id} %{scope.id} %{semantics.id} %{semantics.id} "
                f"%{value_id.id} %{comparator_id.id}"
            )
        else:
            self.emit(
                f"%{id_value} = {atomic_operation} %{target_type.id} "
                f"%{target_pointer.id} %{scope.id} %{semantics.id} %{value_id.id}"
            )

        self.value_types[id_value] = target_type
        return SpirvId(id_value, target_type.type)

    def projected_texture_function_names(self):
        return {
            "textureProj",
            "textureProjOffset",
            "textureProjLod",
            "textureProjLodOffset",
            "textureProjGrad",
            "textureProjGradOffset",
        }

    def projected_shadow_function_names(self):
        return {
            "textureCompareProj",
            "textureCompareProjOffset",
            "textureCompareProjLod",
            "textureCompareProjLodOffset",
            "textureCompareProjGrad",
            "textureCompareProjGradOffset",
        }

    def projected_texture_operand_counts(self, function_name: str) -> int:
        return {
            "textureProj": 0,
            "textureProjOffset": 1,
            "textureProjLod": 1,
            "textureProjLodOffset": 2,
            "textureProjGrad": 2,
            "textureProjGradOffset": 3,
        }[function_name]

    def projected_texture_max_operand_counts(self, function_name: str) -> int:
        return {
            "textureProj": 1,
            "textureProjOffset": 2,
            "textureProjLod": 1,
            "textureProjLodOffset": 2,
            "textureProjGrad": 2,
            "textureProjGradOffset": 3,
        }[function_name]

    def projected_texture_operand_description(self, function_name: str) -> str:
        return {
            "textureProj": "optional bias",
            "textureProjOffset": "offset and optional bias",
            "textureProjLod": "LOD",
            "textureProjLodOffset": "LOD and offset",
            "textureProjGrad": "dx and dy gradient",
            "textureProjGradOffset": "dx and dy gradient and offset",
        }[function_name]

    def projected_shadow_operand_counts(self, function_name: str) -> int:
        return {
            "textureCompareProj": 0,
            "textureCompareProjOffset": 1,
            "textureCompareProjLod": 1,
            "textureCompareProjLodOffset": 2,
            "textureCompareProjGrad": 2,
            "textureCompareProjGradOffset": 3,
        }[function_name]

    def projected_texture_image_operands(
        self, function_name: str, extra_args: List[SpirvId]
    ) -> Optional[str]:
        if function_name == "textureProj":
            if extra_args:
                return self.texture_bias_operand(function_name, extra_args[0])
            if self.requires_explicit_lod_sampling():
                return self.default_lod_operand()
            return ""

        if function_name == "textureProjOffset":
            offset_operand = self.image_offset_operand(extra_args[0])
            if len(extra_args) >= 2:
                bias_operand = self.texture_bias_operand(function_name, extra_args[1])
                if bias_operand is None:
                    return None
                return self.image_operands(offset_operand, bias_operand)
            if self.requires_explicit_lod_sampling():
                return self.image_operands(self.default_lod_operand(), offset_operand)
            return offset_operand

        if function_name == "textureProjLod":
            return f"Lod %{extra_args[0].id}"
        if function_name == "textureProjLodOffset":
            return self.image_operands(
                f"Lod %{extra_args[0].id}", self.image_offset_operand(extra_args[1])
            )
        if function_name == "textureProjGrad":
            return f"Grad %{extra_args[0].id} %{extra_args[1].id}"
        return self.image_operands(
            f"Grad %{extra_args[0].id} %{extra_args[1].id}",
            self.image_offset_operand(extra_args[2]),
        )

    def call_projected_texture_function(
        self, function_name: str, args: List[SpirvId]
    ) -> Optional[SpirvId]:
        sample_args = self.sampled_texture_operands(
            function_name,
            args,
            self.projected_texture_operand_counts(function_name),
        )
        if sample_args is None:
            return self.register_constant(0.0, self.register_primitive_type("float"))

        sampled_image_id, coord_id, extra_args, metadata = sample_args
        result_type = self.resource_access_result_type(metadata)
        if len(extra_args) > self.projected_texture_max_operand_counts(function_name):
            self.emit(
                f"; WARNING: {function_name} accepts only texture, optional "
                "sampler, coordinate, and "
                f"{self.projected_texture_operand_description(function_name)} operands"
            )
            return self.default_value_for_type(result_type)
        if not self.validate_non_multisample_sampled_image(function_name, metadata):
            return self.default_value_for_type(result_type)
        if function_name in {"textureProjLod", "textureProjLodOffset"}:
            if not self.validate_sampled_texture_lod_operand(
                function_name, extra_args[0]
            ):
                return self.default_value_for_type(result_type)
        if function_name in {"textureProjGrad", "textureProjGradOffset"}:
            if not self.validate_sampled_texture_gradient_operand(
                function_name, metadata, extra_args[0], "dx"
            ):
                return self.default_value_for_type(result_type)
            if not self.validate_sampled_texture_gradient_operand(
                function_name, metadata, extra_args[1], "dy"
            ):
                return self.default_value_for_type(result_type)
        offset_indices = {
            "textureProjOffset": 0,
            "textureProjLodOffset": 1,
            "textureProjGradOffset": 2,
        }
        offset_index = offset_indices.get(function_name)
        if (
            offset_index is not None
            and not self.validate_sampled_texture_offset_operand(
                function_name, metadata, extra_args[offset_index]
            )
        ):
            return self.default_value_for_type(result_type)

        projected_coord = self.project_texture_coordinate(
            function_name, coord_id, metadata
        )
        if projected_coord is None:
            return self.default_value_for_type(result_type)

        image_operands = self.projected_texture_image_operands(
            function_name, extra_args
        )
        if image_operands is None:
            return self.default_value_for_type(result_type)

        id_value = self.get_id()
        opcode = (
            "OpImageSampleExplicitLod"
            if image_operands and ("Lod" in image_operands or "Grad" in image_operands)
            else "OpImageSampleImplicitLod"
        )
        self.emit(
            f"%{id_value} = {opcode} %{result_type.id} "
            f"%{sampled_image_id.id} %{projected_coord.id}"
            f"{(' ' + image_operands) if image_operands else ''}"
        )
        self.value_types[id_value] = result_type
        return SpirvId(id_value, result_type.type)

    def projected_shadow_image_operands(
        self, function_name: str, extra_args: List[SpirvId]
    ) -> str:
        if function_name == "textureCompareProj":
            return (
                self.default_lod_operand()
                if self.requires_explicit_lod_sampling()
                else ""
            )
        if function_name == "textureCompareProjOffset":
            offset_operand = self.image_offset_operand(extra_args[0])
            if self.requires_explicit_lod_sampling():
                return self.image_operands(self.default_lod_operand(), offset_operand)
            return offset_operand
        if function_name == "textureCompareProjLod":
            return f"Lod %{extra_args[0].id}"
        if function_name == "textureCompareProjLodOffset":
            return self.image_operands(
                f"Lod %{extra_args[0].id}", self.image_offset_operand(extra_args[1])
            )
        if function_name == "textureCompareProjGrad":
            return f"Grad %{extra_args[0].id} %{extra_args[1].id}"
        return self.image_operands(
            f"Grad %{extra_args[0].id} %{extra_args[1].id}",
            self.image_offset_operand(extra_args[2]),
        )

    def call_projected_shadow_function(
        self, function_name: str, args: List[SpirvId]
    ) -> Optional[SpirvId]:
        compare_args = self.shadow_compare_operands(
            function_name,
            args,
            self.projected_shadow_operand_counts(function_name),
        )
        if compare_args is None:
            return self.register_constant(0.0, self.register_primitive_type("float"))

        sampled_image_id, coord_id, depth_id, extra_args = compare_args
        metadata = self.resource_metadata_for_value(sampled_image_id)
        result_type = self.register_primitive_type("float")
        if not self.validate_shadow_compare_depth_operand(function_name, depth_id):
            return self.default_value_for_type(result_type)

        if function_name in {
            "textureCompareProjLod",
            "textureCompareProjLodOffset",
        }:
            if not self.validate_sampled_texture_lod_operand(
                function_name, extra_args[0]
            ):
                return self.default_value_for_type(result_type)
        if function_name in {
            "textureCompareProjGrad",
            "textureCompareProjGradOffset",
        }:
            if not self.validate_sampled_texture_gradient_operand(
                function_name, metadata, extra_args[0], "dx"
            ):
                return self.default_value_for_type(result_type)
            if not self.validate_sampled_texture_gradient_operand(
                function_name, metadata, extra_args[1], "dy"
            ):
                return self.default_value_for_type(result_type)
        if "Offset" in function_name:
            if not self.validate_sampled_texture_offset_operand(
                function_name, metadata, extra_args[-1]
            ):
                return self.default_value_for_type(result_type)

        projected_coord = self.project_texture_coordinate(
            function_name, coord_id, metadata
        )
        if projected_coord is None:
            return self.default_value_for_type(result_type)

        image_operands = self.projected_shadow_image_operands(function_name, extra_args)
        id_value = self.get_id()
        opcode = (
            "OpImageSampleDrefExplicitLod"
            if image_operands and ("Lod" in image_operands or "Grad" in image_operands)
            else "OpImageSampleDrefImplicitLod"
        )
        self.emit(
            f"%{id_value} = {opcode} %{result_type.id} "
            f"%{sampled_image_id.id} %{projected_coord.id} %{depth_id.id}"
            f"{(' ' + image_operands) if image_operands else ''}"
        )
        self.value_types[id_value] = result_type
        return SpirvId(id_value, result_type.type)

    def call_resource_function(
        self, function_name: str, args: List[SpirvId]
    ) -> Optional[SpirvId]:
        if function_name in self.image_atomic_function_names():
            return self.call_image_atomic_function(function_name, args)
        if function_name in self.buffer_atomic_function_names():
            return self.call_buffer_atomic_function(function_name, args)
        if function_name in self.projected_texture_function_names():
            return self.call_projected_texture_function(function_name, args)
        if function_name in self.projected_shadow_function_names():
            return self.call_projected_shadow_function(function_name, args)

        byte_address_load_width = self.byte_address_helper_load_width(function_name)
        if byte_address_load_width is not None:
            return self.call_byte_address_buffer_load_helper(
                function_name, args, byte_address_load_width
            )

        byte_address_store_width = self.byte_address_helper_store_width(function_name)
        if byte_address_store_width is not None:
            return self.call_byte_address_buffer_store_helper(
                function_name, args, byte_address_store_width
            )

        if function_name == "buffer_append":
            if len(args) < 2:
                self.emit("; WARNING: buffer_append requires buffer and value operands")
                return None
            if len(args) > 2:
                self.emit(
                    "; WARNING: buffer_append accepts only buffer and value operands"
                )
                return None

            metadata = self.structured_buffer_metadata_for_pointer(args[0])
            if metadata is None:
                self.emit(
                    "; WARNING: buffer_append requires an AppendStructuredBuffer "
                    "operand"
                )
                return None
            return self.process_structured_buffer_append_call(
                args[0], metadata, [args[1]], "buffer_append"
            )[1]

        if function_name == "buffer_consume":
            if not args:
                self.emit("; WARNING: buffer_consume requires a buffer operand")
                return self.register_constant(
                    0.0, self.register_primitive_type("float")
                )
            if len(args) > 1:
                self.emit("; WARNING: buffer_consume accepts only a buffer operand")
                return self.default_value_for_buffer_load_failure(args)

            metadata = self.structured_buffer_metadata_for_pointer(args[0])
            if metadata is None:
                self.emit(
                    "; WARNING: buffer_consume requires a ConsumeStructuredBuffer "
                    "operand"
                )
                return self.default_value_for_buffer_load_failure(args)
            return self.process_structured_buffer_consume_call(
                args[0], metadata, [], "buffer_consume"
            )[1]

        if function_name in {"buffer_increment_counter", "buffer_decrement_counter"}:
            if not args:
                self.emit(f"; WARNING: {function_name} requires a buffer operand")
                return self.structured_buffer_counter_default_value()
            if len(args) > 1:
                self.emit(f"; WARNING: {function_name} accepts only a buffer operand")
                return self.structured_buffer_counter_default_value()

            metadata = self.structured_buffer_metadata_for_pointer(args[0])
            if metadata is None:
                self.emit(
                    f"; WARNING: {function_name} requires an RWStructuredBuffer "
                    "operand"
                )
                return self.structured_buffer_counter_default_value()
            return self.process_structured_buffer_counter_method_call(
                args[0],
                metadata,
                [],
                function_name,
                function_name == "buffer_increment_counter",
            )[1]

        if function_name == "buffer_dimensions":
            if not args:
                self.emit("; WARNING: buffer_dimensions requires a buffer operand")
                return self.structured_buffer_dimensions_default_value()
            if len(args) > 1:
                self.emit(
                    "; WARNING: buffer_dimensions expression form accepts only a "
                    "buffer operand"
                )
                return self.structured_buffer_dimensions_default_value()

            metadata = self.structured_buffer_metadata_for_pointer(args[0])
            if metadata is None:
                self.emit(
                    "; WARNING: buffer_dimensions requires a structured or "
                    "byte-address buffer operand"
                )
                return self.structured_buffer_dimensions_default_value()
            return self.emit_structured_buffer_dimensions(
                args[0], metadata, "buffer_dimensions"
            )

        if function_name == "buffer_load":
            if len(args) < 2:
                self.emit("; WARNING: buffer_load requires buffer and index operands")
                return self.default_value_for_buffer_load_failure(args)

            if len(args) > 2:
                self.emit(
                    "; WARNING: buffer_load accepts only buffer and index operands"
                )
                return self.default_value_for_buffer_load_failure(args)

            metadata = self.structured_buffer_metadata_for_pointer(args[0])
            if metadata is not None and metadata.get("writeonly"):
                self.emit("; WARNING: buffer_load requires a readable buffer")
                return self.default_value_for_buffer_load_failure(args)
            if (
                metadata is not None
                and not metadata.get("byte_address")
                and metadata.get("buffer_kind")
                not in {
                    "StructuredBuffer",
                    "RWStructuredBuffer",
                }
            ):
                self.emit("; WARNING: buffer_load requires a StructuredBuffer operand")
                return self.default_value_for_buffer_load_failure(args)

            element_pointer = self.structured_buffer_element_pointer(args[0], args[1])
            if element_pointer is None:
                self.emit("; WARNING: buffer_load requires a StructuredBuffer operand")
                return self.default_value_for_buffer_load_failure(args)

            element_type = self.variable_value_types[element_pointer.id]
            return self.load_from_variable(element_pointer, element_type)

        if function_name == "buffer_store":
            if len(args) < 3:
                self.emit(
                    "; WARNING: buffer_store requires buffer, index, and value operands"
                )
                return None

            if len(args) > 3:
                self.emit(
                    "; WARNING: buffer_store accepts only buffer, index, and "
                    "value operands"
                )
                return None

            metadata = self.structured_buffer_metadata_for_pointer(args[0])
            if metadata is None:
                self.emit(
                    "; WARNING: buffer_store requires an RWStructuredBuffer operand"
                )
                return None
            if (
                not metadata.get("byte_address")
                and metadata.get("buffer_kind") != "RWStructuredBuffer"
            ):
                self.emit(
                    "; WARNING: buffer_store requires an RWStructuredBuffer operand"
                )
                return None
            if metadata.get("readonly"):
                self.emit("; WARNING: buffer_store requires an RWStructuredBuffer")
                return None

            element_pointer = self.structured_buffer_element_pointer(args[0], args[1])
            if element_pointer is None:
                self.emit(
                    "; WARNING: buffer_store requires an RWStructuredBuffer operand"
                )
                return None

            self.store_to_variable(element_pointer, args[2])
            return None

        if function_name == "imageLoad":
            if len(args) < 2:
                self.emit("; WARNING: imageLoad requires image and coordinate operands")
                return self.register_constant(
                    0.0, self.register_primitive_type("float")
                )

            image_id, coord_id = args[0], args[1]
            metadata = self.resource_metadata_for_value(image_id)
            if not metadata or metadata.get("kind") != "storage_image":
                self.emit("; WARNING: imageLoad requires a storage image operand")
                return self.register_constant(
                    0.0, self.register_primitive_type("float")
                )

            result_type = self.resource_access_result_type(metadata)
            if metadata.get("writeonly"):
                self.emit("; WARNING: imageLoad requires a readable storage image")
                return self.default_value_for_type(result_type)

            if not self.validate_storage_image_coordinate(
                function_name, metadata, coord_id
            ):
                return self.default_value_for_type(result_type)

            image_operands = ""
            if metadata.get("multisampled"):
                if len(args) < 3:
                    self.emit("; WARNING: imageLoad requires a sample operand")
                    return self.default_value_for_type(result_type)
                if not self.validate_storage_image_sample(function_name, args[2]):
                    return self.default_value_for_type(result_type)
                image_operands = f" Sample %{args[2].id}"

            id_value = self.get_id()
            self.require_storage_image_without_format_capability(
                metadata, "StorageImageReadWithoutFormat"
            )
            self.emit(
                f"%{id_value} = OpImageRead %{result_type.id} "
                f"%{image_id.id} %{coord_id.id}{image_operands}"
            )
            self.value_types[id_value] = result_type
            return SpirvId(id_value, result_type.type)

        if function_name == "imageStore":
            if len(args) < 3:
                self.emit(
                    "; WARNING: imageStore requires image, coordinate, and value operands"
                )
                return None

            image_id, coord_id, texel_id = args[0], args[1], args[2]
            metadata = self.resource_metadata_for_value(image_id)
            if not metadata or metadata.get("kind") != "storage_image":
                self.emit("; WARNING: imageStore requires a storage image operand")
                return None
            if metadata.get("readonly"):
                self.emit("; WARNING: imageStore requires a writable storage image")
                return None

            if not self.validate_storage_image_coordinate(
                function_name, metadata, coord_id
            ):
                return None

            texel_type = self.resource_access_result_type(metadata)
            image_operands = ""
            if metadata.get("multisampled"):
                if len(args) < 4:
                    self.emit("; WARNING: imageStore requires a sample operand")
                    return None
                sample_id, texel_id = args[2], args[3]
                if not self.validate_storage_image_sample(function_name, sample_id):
                    return None
                image_operands = f" Sample %{sample_id.id}"

            texel_id = self.convert_storage_image_value_operand(
                function_name, "value", texel_id, texel_type
            )
            if texel_id is None:
                return None

            self.require_storage_image_without_format_capability(
                metadata, "StorageImageWriteWithoutFormat"
            )
            self.emit(
                f"OpImageWrite %{image_id.id} %{coord_id.id} %{texel_id.id}"
                f"{image_operands}"
            )
            return None

        if function_name in {"texture", "texture2D", "textureCube"}:
            sample_args = self.sampled_texture_operands(function_name, args)
            if sample_args is None:
                return self.register_constant(
                    0.0, self.register_primitive_type("float")
                )

            sampled_image_id, coord_id, extra_args, metadata = sample_args

            result_type = self.resource_access_result_type(metadata)
            if len(extra_args) > 1:
                self.emit(
                    f"; WARNING: {function_name} accepts only texture, optional "
                    "sampler, coordinate, and optional bias operands"
                )
                return self.default_value_for_type(result_type)
            if not self.validate_non_multisample_sampled_image(function_name, metadata):
                return self.default_value_for_type(result_type)
            if not self.validate_sampled_texture_coordinate(
                function_name, metadata, coord_id
            ):
                return self.default_value_for_type(result_type)

            bias_operand = None
            if extra_args:
                bias_operand = self.texture_bias_operand(function_name, extra_args[0])
                if bias_operand is None:
                    return self.default_value_for_type(result_type)

            id_value = self.get_id()
            if self.requires_explicit_lod_sampling():
                self.emit(
                    f"%{id_value} = OpImageSampleExplicitLod %{result_type.id} "
                    f"%{sampled_image_id.id} %{coord_id.id} "
                    f"{self.default_lod_operand()}"
                )
            else:
                self.emit(
                    f"%{id_value} = OpImageSampleImplicitLod %{result_type.id} "
                    f"%{sampled_image_id.id} %{coord_id.id}"
                    f"{(' ' + bias_operand) if bias_operand else ''}"
                )
            self.value_types[id_value] = result_type
            return SpirvId(id_value, result_type.type)

        if function_name in {
            "textureCompare",
            "textureCompareLod",
            "textureCompareLodOffset",
            "textureCompareGrad",
            "textureCompareGradOffset",
            "textureCompareOffset",
        }:
            extra_arg_count = {
                "textureCompare": 0,
                "textureCompareLod": 1,
                "textureCompareLodOffset": 2,
                "textureCompareGrad": 2,
                "textureCompareGradOffset": 3,
                "textureCompareOffset": 1,
            }[function_name]
            compare_args = self.shadow_compare_operands(
                function_name, args, extra_arg_count
            )
            if compare_args is None:
                return self.register_constant(
                    0.0, self.register_primitive_type("float")
                )

            sampled_image_id, coord_id, depth_id, extra_args = compare_args
            result_type = self.register_primitive_type("float")
            metadata = self.resource_metadata_for_value(sampled_image_id)
            if not self.validate_sampled_texture_coordinate(
                function_name, metadata, coord_id
            ):
                return self.default_value_for_type(result_type)
            if not self.validate_shadow_compare_depth_operand(function_name, depth_id):
                return self.default_value_for_type(result_type)

            if function_name in {"textureCompareLod", "textureCompareLodOffset"}:
                if not self.validate_sampled_texture_lod_operand(
                    function_name, extra_args[0]
                ):
                    return self.default_value_for_type(result_type)
            if function_name in {"textureCompareGrad", "textureCompareGradOffset"}:
                if not self.validate_sampled_texture_gradient_operand(
                    function_name, metadata, extra_args[0], "dx"
                ):
                    return self.default_value_for_type(result_type)
                if not self.validate_sampled_texture_gradient_operand(
                    function_name, metadata, extra_args[1], "dy"
                ):
                    return self.default_value_for_type(result_type)
            if function_name in {
                "textureCompareOffset",
                "textureCompareLodOffset",
                "textureCompareGradOffset",
            }:
                offset_index = -1
                if not self.validate_sampled_texture_offset_operand(
                    function_name, metadata, extra_args[offset_index]
                ):
                    return self.default_value_for_type(result_type)

            id_value = self.get_id()
            if function_name == "textureCompare":
                if self.requires_explicit_lod_sampling():
                    self.emit(
                        f"%{id_value} = OpImageSampleDrefExplicitLod "
                        f"%{result_type.id} %{sampled_image_id.id} "
                        f"%{coord_id.id} %{depth_id.id} {self.default_lod_operand()}"
                    )
                else:
                    self.emit(
                        f"%{id_value} = OpImageSampleDrefImplicitLod "
                        f"%{result_type.id} %{sampled_image_id.id} "
                        f"%{coord_id.id} %{depth_id.id}"
                    )
            elif function_name == "textureCompareOffset":
                offset_operand = self.image_offset_operand(extra_args[0])
                if self.requires_explicit_lod_sampling():
                    self.emit(
                        f"%{id_value} = OpImageSampleDrefExplicitLod "
                        f"%{result_type.id} %{sampled_image_id.id} "
                        f"%{coord_id.id} %{depth_id.id} "
                        f"{self.image_operands(self.default_lod_operand(), offset_operand)}"
                    )
                else:
                    self.emit(
                        f"%{id_value} = OpImageSampleDrefImplicitLod "
                        f"%{result_type.id} %{sampled_image_id.id} "
                        f"%{coord_id.id} %{depth_id.id} {offset_operand}"
                    )
            elif function_name == "textureCompareLod":
                self.emit(
                    f"%{id_value} = OpImageSampleDrefExplicitLod %{result_type.id} "
                    f"%{sampled_image_id.id} %{coord_id.id} %{depth_id.id} "
                    f"Lod %{extra_args[0].id}"
                )
            elif function_name == "textureCompareLodOffset":
                offset_operand = self.image_offset_operand(extra_args[1])
                self.emit(
                    f"%{id_value} = OpImageSampleDrefExplicitLod %{result_type.id} "
                    f"%{sampled_image_id.id} %{coord_id.id} %{depth_id.id} "
                    f"{self.image_operands(f'Lod %{extra_args[0].id}', offset_operand)}"
                )
            elif function_name == "textureCompareGrad":
                self.emit(
                    f"%{id_value} = OpImageSampleDrefExplicitLod %{result_type.id} "
                    f"%{sampled_image_id.id} %{coord_id.id} %{depth_id.id} "
                    f"Grad %{extra_args[0].id} %{extra_args[1].id}"
                )
            else:
                offset_operand = self.image_offset_operand(extra_args[2])
                self.emit(
                    f"%{id_value} = OpImageSampleDrefExplicitLod %{result_type.id} "
                    f"%{sampled_image_id.id} %{coord_id.id} %{depth_id.id} "
                    f"{self.image_operands(f'Grad %{extra_args[0].id} %{extra_args[1].id}', offset_operand)}"
                )

            self.value_types[id_value] = result_type
            return SpirvId(id_value, result_type.type)

        if function_name in {"textureGatherCompare", "textureGatherCompareOffset"}:
            extra_arg_count = 1 if function_name == "textureGatherCompareOffset" else 0
            compare_args = self.shadow_compare_operands(
                function_name, args, extra_arg_count
            )
            float_type = self.register_primitive_type("float")
            result_type = self.register_vector_type(float_type, 4)
            if compare_args is None:
                return self.default_value_for_type(result_type)

            sampled_image_id, coord_id, depth_id, extra_args = compare_args
            metadata = self.resource_metadata_for_value(sampled_image_id)
            if not self.validate_sampled_texture_coordinate(
                function_name, metadata, coord_id
            ):
                return self.default_value_for_type(result_type)
            if not self.validate_shadow_compare_depth_operand(function_name, depth_id):
                return self.default_value_for_type(result_type)
            if function_name == "textureGatherCompareOffset":
                if not self.validate_sampled_texture_offset_operand(
                    function_name, metadata, extra_args[0]
                ):
                    return self.default_value_for_type(result_type)

            id_value = self.get_id()
            image_operands = (
                f" {self.image_offset_operand(extra_args[0])}"
                if function_name == "textureGatherCompareOffset"
                else ""
            )
            self.emit(
                f"%{id_value} = OpImageDrefGather %{result_type.id} "
                f"%{sampled_image_id.id} %{coord_id.id} %{depth_id.id}"
                f"{image_operands}"
            )
            self.value_types[id_value] = result_type
            return SpirvId(id_value, result_type.type)

        if function_name == "textureOffset":
            sample_args = self.sampled_texture_operands(function_name, args, 1)
            if sample_args is None:
                return self.register_constant(
                    0.0, self.register_primitive_type("float")
                )

            sampled_image_id, coord_id, extra_args, metadata = sample_args
            offset_id = extra_args[0]

            result_type = self.resource_access_result_type(metadata)
            if len(extra_args) > 2:
                self.emit(
                    "; WARNING: textureOffset accepts only texture, optional "
                    "sampler, coordinate, offset, and optional bias operands"
                )
                return self.default_value_for_type(result_type)
            if not self.validate_non_multisample_sampled_image(function_name, metadata):
                return self.default_value_for_type(result_type)
            if not self.validate_sampled_texture_coordinate(
                function_name, metadata, coord_id
            ):
                return self.default_value_for_type(result_type)
            if not self.validate_sampled_texture_offset_operand(
                function_name, metadata, offset_id
            ):
                return self.default_value_for_type(result_type)

            bias_operand = None
            if len(extra_args) >= 2:
                bias_operand = self.texture_bias_operand(function_name, extra_args[1])
                if bias_operand is None:
                    return self.default_value_for_type(result_type)

            id_value = self.get_id()
            offset_operand = self.image_offset_operand(offset_id)
            if self.requires_explicit_lod_sampling():
                self.emit(
                    f"%{id_value} = OpImageSampleExplicitLod %{result_type.id} "
                    f"%{sampled_image_id.id} %{coord_id.id} "
                    f"{self.image_operands(self.default_lod_operand(), offset_operand)}"
                )
            else:
                self.emit(
                    f"%{id_value} = OpImageSampleImplicitLod %{result_type.id} "
                    f"%{sampled_image_id.id} %{coord_id.id} "
                    f"{self.image_operands(offset_operand, bias_operand)}"
                )
            self.value_types[id_value] = result_type
            return SpirvId(id_value, result_type.type)

        if function_name in {
            "textureGather",
            "textureGatherOffset",
            "textureGatherOffsets",
        }:
            required_extra_count = 0 if function_name == "textureGather" else 1
            sample_args = self.sampled_texture_operands(
                function_name, args, required_extra_count
            )
            if sample_args is None:
                return self.register_constant(
                    0.0, self.register_primitive_type("float")
                )

            sampled_image_id, coord_id, extra_args, metadata = sample_args
            int_type = self.register_primitive_type("int")
            result_type = self.resource_access_result_type(metadata)
            if not self.validate_non_multisample_sampled_image(function_name, metadata):
                return self.default_value_for_type(result_type)
            if not self.validate_sampled_texture_coordinate(
                function_name, metadata, coord_id
            ):
                return self.default_value_for_type(result_type)

            if function_name == "textureGather":
                if len(extra_args) > 1:
                    self.emit(
                        "; WARNING: textureGather accepts only texture, optional "
                        "sampler, coordinate, and optional component operands"
                    )
                    return self.default_value_for_type(result_type)
                component_id = (
                    extra_args[0] if extra_args else self.register_constant(0, int_type)
                )
                if not self.validate_texture_gather_component_operand(
                    function_name, component_id
                ):
                    return self.default_value_for_type(result_type)
                offset_id = None
            elif function_name == "textureGatherOffsets":
                if len(extra_args) not in {1, 2, 4, 5}:
                    self.emit(
                        self.sampled_texture_excess_operand_warning(function_name)
                    )
                    return self.default_value_for_type(result_type)
                return self.emit_texture_gather_offsets(
                    sampled_image_id,
                    coord_id,
                    extra_args,
                    metadata,
                    int_type,
                )
            else:
                if len(extra_args) > 2:
                    self.emit(
                        "; WARNING: textureGatherOffset accepts only texture, "
                        "optional sampler, coordinate, offset, and optional "
                        "component operands"
                    )
                    return self.default_value_for_type(result_type)
                offset_id = extra_args[0]
                if not self.validate_sampled_texture_offset_operand(
                    function_name, metadata, offset_id
                ):
                    return self.default_value_for_type(result_type)
                component_id = (
                    extra_args[1]
                    if len(extra_args) >= 2
                    else self.register_constant(0, int_type)
                )
                if not self.validate_texture_gather_component_operand(
                    function_name, component_id
                ):
                    return self.default_value_for_type(result_type)

            return self.emit_image_gather(
                sampled_image_id,
                coord_id,
                component_id,
                result_type,
                offset_id,
            )

        if function_name in {"texelFetch", "texelFetchOffset"}:
            required_extra_count = 1 if function_name == "texelFetch" else 2
            sample_args = self.sampled_texture_operands(
                function_name, args, required_extra_count
            )
            if sample_args is None:
                return self.register_constant(
                    0.0, self.register_primitive_type("float")
                )

            sampled_image_id, coord_id, extra_args, metadata = sample_args
            operand_id = extra_args[0]
            offset_id = extra_args[1] if function_name == "texelFetchOffset" else None
            result_type = self.resource_access_result_type(metadata)
            if len(extra_args) > required_extra_count:
                self.emit(self.sampled_texture_excess_operand_warning(function_name))
                return self.default_value_for_type(result_type)

            if metadata.get("dim") == "Cube":
                self.emit(f"; WARNING: {function_name} is not valid for cube images")
                return self.default_value_for_type(result_type)

            if not self.validate_sampled_texture_coordinate(
                function_name, metadata, coord_id, integer=True
            ):
                return self.default_value_for_type(result_type)
            if metadata.get("multisampled"):
                if not self.validate_sampled_texture_fetch_operand(
                    function_name, operand_id, "sample"
                ):
                    return self.default_value_for_type(result_type)
            elif not self.validate_sampled_texture_fetch_operand(
                function_name, operand_id, "LOD"
            ):
                return self.default_value_for_type(result_type)

            if metadata.get("multisampled") and offset_id is not None:
                self.emit(
                    "; WARNING: texelFetchOffset is not valid for multisample images"
                )
                return self.default_value_for_type(result_type)
            if (
                offset_id is not None
                and not self.validate_sampled_texture_offset_operand(
                    function_name, metadata, offset_id
                )
            ):
                return self.default_value_for_type(result_type)

            image_id = self.extract_image_from_sampled_image(sampled_image_id, metadata)
            if image_id is None:
                return self.default_value_for_type(result_type)

            id_value = self.get_id()
            image_operand = (
                f"Sample %{operand_id.id}"
                if metadata.get("multisampled")
                else f"Lod %{operand_id.id}"
            )
            if offset_id is not None:
                image_operand = self.image_operands(
                    image_operand,
                    self.image_offset_operand(offset_id),
                )
            self.emit(
                f"%{id_value} = OpImageFetch %{result_type.id} "
                f"%{image_id.id} %{coord_id.id} {image_operand}"
            )
            self.value_types[id_value] = result_type
            return SpirvId(id_value, result_type.type)

        if function_name in {"textureSize", "imageSize"}:
            if not args:
                self.emit(f"; WARNING: {function_name} requires an image operand")
                return self.register_constant(0, self.register_primitive_type("int"))

            resource_id = args[0]
            metadata = self.resource_metadata_for_value(resource_id)
            expected_kind = (
                "sampled_image" if function_name == "textureSize" else "storage_image"
            )
            if not metadata or metadata.get("kind") != expected_kind:
                self.emit(
                    f"; WARNING: {function_name} requires a {expected_kind} operand"
                )
                return self.resource_query_size_default_value(metadata)

            image_id = self.image_operand_for_query(resource_id, metadata)
            if image_id is None:
                return self.resource_query_size_default_value(metadata)

            result_type = self.resource_query_size_result_type(metadata)
            if function_name == "imageSize" and len(args) > 1:
                self.emit("; WARNING: imageSize accepts only an image operand")
                return self.default_value_for_type(result_type)

            if function_name == "textureSize" and len(args) > 2:
                self.emit(
                    "; WARNING: textureSize accepts only texture and optional "
                    "LOD operands"
                )
                return self.default_value_for_type(result_type)

            if (
                function_name == "textureSize"
                and len(args) >= 2
                and not self.validate_resource_query_lod_operand(function_name, args[1])
            ):
                return self.default_value_for_type(result_type)

            id_value = self.get_id()
            self.require_capability("ImageQuery")
            if (
                function_name == "textureSize"
                and len(args) >= 2
                and not metadata.get("multisampled")
            ):
                self.emit(
                    f"%{id_value} = OpImageQuerySizeLod %{result_type.id} "
                    f"%{image_id.id} %{args[1].id}"
                )
            else:
                self.emit(
                    f"%{id_value} = OpImageQuerySize %{result_type.id} "
                    f"%{image_id.id}"
                )
            self.value_types[id_value] = result_type
            return SpirvId(id_value, result_type.type)

        if function_name in {"textureSamples", "imageSamples"}:
            if not args:
                self.emit(f"; WARNING: {function_name} requires an image operand")
                return self.register_constant(0, self.register_primitive_type("int"))

            if len(args) > 1:
                self.emit(f"; WARNING: {function_name} accepts only an image operand")
                return self.register_constant(0, self.register_primitive_type("int"))

            resource_id = args[0]
            metadata = self.resource_metadata_for_value(resource_id)
            expected_kind = (
                "sampled_image"
                if function_name == "textureSamples"
                else "storage_image"
            )
            if not metadata or metadata.get("kind") != expected_kind:
                self.emit(
                    f"; WARNING: {function_name} requires a {expected_kind} operand"
                )
                return self.register_constant(0, self.register_primitive_type("int"))

            if metadata.get("dim") != "2D" or not metadata.get("multisampled"):
                self.emit(f"; WARNING: {function_name} requires a multisample 2D image")
                return self.register_constant(0, self.register_primitive_type("int"))

            image_id = self.image_operand_for_query(resource_id, metadata)
            if image_id is None:
                return self.register_constant(0, self.register_primitive_type("int"))

            result_type = self.register_primitive_type("int")
            id_value = self.get_id()
            self.require_capability("ImageQuery")
            self.emit(
                f"%{id_value} = OpImageQuerySamples %{result_type.id} "
                f"%{image_id.id}"
            )
            self.value_types[id_value] = result_type
            return SpirvId(id_value, result_type.type)

        if function_name == "textureQueryLevels":
            if not args:
                self.emit("; WARNING: textureQueryLevels requires a texture operand")
                return self.register_constant(0, self.register_primitive_type("int"))

            if len(args) > 1:
                self.emit(
                    "; WARNING: textureQueryLevels accepts only a texture operand"
                )
                return self.register_constant(0, self.register_primitive_type("int"))

            sampled_image_id = args[0]
            metadata = self.resource_metadata_for_value(sampled_image_id)
            if not metadata or metadata.get("kind") != "sampled_image":
                self.emit(
                    "; WARNING: textureQueryLevels requires a sampled image operand"
                )
                return self.register_constant(0, self.register_primitive_type("int"))

            if metadata.get("multisampled"):
                self.emit(
                    "; WARNING: textureQueryLevels requires a non-multisample "
                    "sampled image operand"
                )
                return self.register_constant(0, self.register_primitive_type("int"))

            image_id = self.extract_image_from_sampled_image(sampled_image_id, metadata)
            if image_id is None:
                return self.register_constant(0, self.register_primitive_type("int"))

            result_type = self.register_primitive_type("int")
            id_value = self.get_id()
            self.require_capability("ImageQuery")
            self.emit(
                f"%{id_value} = OpImageQueryLevels %{result_type.id} " f"%{image_id.id}"
            )
            self.value_types[id_value] = result_type
            return SpirvId(id_value, result_type.type)

        if function_name == "textureQueryLod":
            if len(args) < 2:
                self.emit(
                    "; WARNING: textureQueryLod requires texture and coordinate operands"
                )
                return self.resource_query_lod_default_value()

            sampled_image_id = args[0]
            coord_id = args[1]
            sampler_id = None
            sampler_metadata = self.resource_metadata_for_value(args[1])
            if sampler_metadata and sampler_metadata.get("kind") == "sampler":
                sampler_id = args[1]
                if len(args) < 3:
                    self.emit(
                        "; WARNING: textureQueryLod requires a coordinate operand "
                        "after the sampler"
                    )
                    return self.resource_query_lod_default_value()
                if len(args) > 3:
                    self.emit(
                        "; WARNING: textureQueryLod accepts only texture, sampler, "
                        "and coordinate operands"
                    )
                    return self.resource_query_lod_default_value()
                coord_id = args[2]
            elif len(args) > 2:
                self.emit(
                    "; WARNING: textureQueryLod accepts only texture and coordinate "
                    "operands unless the second operand is a sampler"
                )
                return self.resource_query_lod_default_value()

            metadata = self.resource_metadata_for_value(sampled_image_id)
            if metadata and metadata.get("kind") == "texture":
                if sampler_id is None:
                    self.emit(
                        "; WARNING: textureQueryLod requires a sampler for separate "
                        "texture operands"
                    )
                    return self.resource_query_lod_default_value()
                sampled_image_id = self.combine_texture_and_sampler(
                    function_name, sampled_image_id, sampler_id, metadata
                )
                if sampled_image_id is None:
                    return self.resource_query_lod_default_value()
                metadata = self.resource_metadata_for_value(sampled_image_id)

            if not metadata or metadata.get("kind") != "sampled_image":
                self.emit("; WARNING: textureQueryLod requires a sampled image operand")
                return self.resource_query_lod_default_value()

            if not self.validate_non_multisample_sampled_image(function_name, metadata):
                return self.resource_query_lod_default_value()

            if not self.validate_resource_query_lod_coordinate(
                function_name, metadata, coord_id
            ):
                return self.resource_query_lod_default_value()

            coord_id = self.trim_image_query_lod_coordinate(coord_id, metadata)
            float_type = self.register_primitive_type("float")
            result_type = self.register_vector_type(float_type, 2)
            id_value = self.get_id()
            self.require_capability("ImageQuery")
            if self.requires_explicit_lod_sampling():
                self.require_compute_derivatives()
            self.emit(
                f"%{id_value} = OpImageQueryLod %{result_type.id} "
                f"%{sampled_image_id.id} %{coord_id.id}"
            )
            self.value_types[id_value] = result_type
            return SpirvId(id_value, result_type.type)

        if function_name in {
            "textureLod",
            "textureLodOffset",
            "textureGrad",
            "textureGradOffset",
        }:
            required_arg_count = {
                "textureLod": 3,
                "textureLodOffset": 4,
                "textureGrad": 4,
                "textureGradOffset": 5,
            }[function_name]
            extra_arg_count = required_arg_count - 2
            sample_args = self.sampled_texture_operands(
                function_name, args, extra_arg_count
            )
            if sample_args is None:
                return self.register_constant(
                    0.0, self.register_primitive_type("float")
                )

            sampled_image_id, coord_id, extra_args, metadata = sample_args
            result_type = self.resource_access_result_type(metadata)
            if len(extra_args) > extra_arg_count:
                self.emit(self.sampled_texture_excess_operand_warning(function_name))
                return self.default_value_for_type(result_type)
            if not self.validate_non_multisample_sampled_image(function_name, metadata):
                return self.default_value_for_type(result_type)
            if not self.validate_sampled_texture_coordinate(
                function_name, metadata, coord_id
            ):
                return self.default_value_for_type(result_type)

            if function_name == "textureLod":
                if not self.validate_sampled_texture_lod_operand(
                    function_name, extra_args[0]
                ):
                    return self.default_value_for_type(result_type)
                image_operands = f"Lod %{extra_args[0].id}"
            elif function_name == "textureLodOffset":
                if not self.validate_sampled_texture_lod_operand(
                    function_name, extra_args[0]
                ):
                    return self.default_value_for_type(result_type)
                if not self.validate_sampled_texture_offset_operand(
                    function_name, metadata, extra_args[1]
                ):
                    return self.default_value_for_type(result_type)
                image_operands = self.image_operands(
                    f"Lod %{extra_args[0].id}",
                    self.image_offset_operand(extra_args[1]),
                )
            elif function_name == "textureGrad":
                if not self.validate_sampled_texture_gradient_operand(
                    function_name, metadata, extra_args[0], "dx"
                ):
                    return self.default_value_for_type(result_type)
                if not self.validate_sampled_texture_gradient_operand(
                    function_name, metadata, extra_args[1], "dy"
                ):
                    return self.default_value_for_type(result_type)
                image_operands = f"Grad %{extra_args[0].id} %{extra_args[1].id}"
            else:
                if not self.validate_sampled_texture_gradient_operand(
                    function_name, metadata, extra_args[0], "dx"
                ):
                    return self.default_value_for_type(result_type)
                if not self.validate_sampled_texture_gradient_operand(
                    function_name, metadata, extra_args[1], "dy"
                ):
                    return self.default_value_for_type(result_type)
                if not self.validate_sampled_texture_offset_operand(
                    function_name, metadata, extra_args[2]
                ):
                    return self.default_value_for_type(result_type)
                image_operands = self.image_operands(
                    f"Grad %{extra_args[0].id} %{extra_args[1].id}",
                    self.image_offset_operand(extra_args[2]),
                )

            id_value = self.get_id()
            self.emit(
                f"%{id_value} = OpImageSampleExplicitLod %{result_type.id} "
                f"%{sampled_image_id.id} %{coord_id.id} {image_operands}"
            )
            self.value_types[id_value] = result_type
            return SpirvId(id_value, result_type.type)

        return None

    def resource_offset_argument_indices(self, function_name: str):
        return {
            "textureOffset": {2, 3},
            "textureProjOffset": {2, 3},
            "textureProjLodOffset": {3, 4},
            "textureProjGradOffset": {4, 5},
            "textureLodOffset": {3, 4},
            "textureGradOffset": {4, 5},
            "textureGatherOffset": {2, 3},
            "textureGatherOffsets": {2, 3, 4, 5, 6},
            "texelFetchOffset": {3, 4},
            "textureCompareOffset": {3, 4},
            "textureCompareProjOffset": {3, 4},
            "textureCompareProjLodOffset": {4, 5},
            "textureCompareProjGradOffset": {5, 6},
            "textureCompareLodOffset": {4, 5},
            "textureCompareGradOffset": {5, 6},
            "textureGatherCompareOffset": {3, 4},
        }.get(function_name, set())

    def literal_integer_vector_constant(self, expr) -> Optional[SpirvId]:
        if not isinstance(expr, FunctionCallNode):
            return None

        callee_expr = getattr(expr, "function", getattr(expr, "name", None))
        if hasattr(callee_expr, "name"):
            function_name = callee_expr.name
        elif isinstance(callee_expr, str):
            function_name = callee_expr
        else:
            return None

        vector_info = self.vector_component_type_and_count(function_name)
        if vector_info is None:
            return None

        component_type_name, component_count = vector_info
        if component_type_name not in {"int", "uint"}:
            return None
        if len(expr.args) != component_count:
            return None

        component_values = [
            self.literal_integer_value(arg, component_type_name) for arg in expr.args
        ]
        if any(value is None for value in component_values):
            return None

        component_type = self.register_primitive_type(component_type_name)
        vector_type = self.register_vector_type(component_type, component_count)
        components = [
            self.register_constant(value, component_type) for value in component_values
        ]
        return self.register_vector_constant(vector_type, components)

    def literal_integer_value(self, expr, component_type_name: str) -> Optional[int]:
        if isinstance(expr, UnaryOpNode):
            value = self.literal_integer_value(expr.operand, component_type_name)
            if value is None:
                return None
            if expr.op == "-":
                value = -value
            elif expr.op != "+":
                return None
            if component_type_name == "uint" and value < 0:
                return None
            return value

        if not isinstance(expr, LiteralNode):
            return None

        literal_type = self.normalize_primitive_name(
            self.convert_type_node_to_string(expr.literal_type)
        )
        if literal_type not in {"int", "uint"}:
            return None

        value = int(expr.value)
        if component_type_name == "uint" and value < 0:
            return None
        return value

    def process_call_argument(self, function_name, arg, arg_index):
        if arg_index in self.resource_offset_argument_indices(function_name):
            offset_constant = self.literal_integer_vector_constant(arg)
            if offset_constant is not None:
                return offset_constant

        if function_name in self.image_atomic_function_names() and arg_index == 0:
            pointer_arg = self.variable_pointer_from_expression(arg)
            if pointer_arg is not None:
                return pointer_arg

        if function_name in self.buffer_atomic_function_names() and arg_index == 0:
            pointer_arg = self.variable_pointer_from_expression(arg)
            if pointer_arg is not None:
                return pointer_arg

        if function_name in self.buffer_function_names() and arg_index == 0:
            pointer_arg = self.variable_pointer_from_expression(arg)
            if pointer_arg is not None:
                return pointer_arg

        if self.function_parameter_requires_storage_image_pointer(
            function_name, arg_index
        ):
            pointer_arg = self.variable_pointer_from_expression(arg)
            if pointer_arg is not None:
                return pointer_arg

        resource_array_params = self.resolve_function_resource_array_params(
            function_name
        )
        if arg_index in resource_array_params:
            pointer_arg = self.variable_pointer_from_expression(arg)
            if pointer_arg is not None:
                return pointer_arg

        return self.process_expression(arg)

    def contains_lambda_expression(self, expr) -> bool:
        if isinstance(expr, FunctionCallNode):
            callee_expr = getattr(expr, "function", getattr(expr, "name", None))
            if hasattr(callee_expr, "name"):
                callee_name = callee_expr.name
            elif isinstance(callee_expr, str):
                callee_name = callee_expr
            else:
                callee_name = None
            if callee_name == "lambda":
                return True
            return any(self.contains_lambda_expression(arg) for arg in expr.args)

        for attr in (
            "left",
            "right",
            "condition",
            "true_expr",
            "false_expr",
            "operand",
            "object",
            "array",
            "index",
        ):
            child = getattr(expr, attr, None)
            if child is not None and self.contains_lambda_expression(child):
                return True

        elements = getattr(expr, "elements", None)
        if elements is not None:
            return any(self.contains_lambda_expression(element) for element in elements)

        return False

    def unsupported_lambda_default_value(
        self, context: str, result_type=None
    ) -> SpirvId:
        self.emit(
            "; WARNING: SPIR-V backend does not support CrossGL "
            f"lambda expressions in {context}; using a default value"
        )
        if result_type is None:
            result_type = self.register_primitive_type("float")
        return self.default_value_for_type(result_type)

    def represented_ir_diagnostic_default_value(
        self, category: str, operation: str
    ) -> SpirvId:
        result_type = self.represented_ir_diagnostic_result_type(category, operation)
        result_type_label = self.diagnostic_type_label(result_type)
        self.emit(
            f"; WARNING: SPIR-V backend does not lower {category} operation "
            f"{operation} yet; using a default {result_type_label} value"
        )
        return self.default_value_for_type(result_type)

    def diagnostic_type_label(self, type_id: SpirvId) -> str:
        vector_info = self.vector_component_type_and_count(type_id.type.base_type)
        if vector_info is not None:
            component_type, count = vector_info
            vector_prefixes = {
                "float": "vec",
                "double": "dvec",
                "int": "ivec",
                "uint": "uvec",
                "bool": "bvec",
            }
            return f"{vector_prefixes.get(component_type, 'vec')}{count}"
        return type_id.type.base_type

    def represented_ir_diagnostic_result_type(
        self, category: str, operation: str
    ) -> SpirvId:
        if category == "ray tracing" and operation == "ReportHit":
            return self.register_primitive_type("bool")

        if category == "ray query":
            if operation == "Proceed":
                return self.register_primitive_type("bool")
            getter_info = self.ray_query_state_getter_info(operation)
            if getter_info is not None:
                _, result_kind = getter_info
                return self.ray_query_result_type_for_kind(result_kind)
            getter_info = self.ray_query_candidate_getter_info(operation)
            if getter_info is not None:
                _, result_kind = getter_info
                return self.ray_query_result_type_for_kind(result_kind)
            getter_info = self.ray_query_intersection_getter_info(operation)
            if getter_info is not None:
                _, result_kind, _ = getter_info
                return self.ray_query_result_type_for_kind(result_kind)
            return self.register_primitive_type("uint")

        return self.register_primitive_type("uint")

    def ray_tracing_operation_result_type(self, operation: str) -> SpirvId:
        if operation == "ReportHit":
            return self.register_primitive_type("bool")
        return self.register_primitive_type("uint")

    def ray_tracing_default_value(self, operation: str) -> SpirvId:
        return self.default_value_for_type(
            self.ray_tracing_operation_result_type(operation)
        )

    def ray_tracing_allowed_execution_models(self, operation: str):
        return {
            "TraceRay": {"RayGenerationKHR", "ClosestHitKHR", "MissKHR"},
            "CallShader": {
                "RayGenerationKHR",
                "ClosestHitKHR",
                "MissKHR",
                "CallableKHR",
            },
            "ReportHit": {"IntersectionKHR"},
            "IgnoreHit": {"AnyHitKHR"},
            "AcceptHitAndEndSearch": {"AnyHitKHR"},
        }.get(operation)

    def validate_ray_tracing_operation_context(self, operation: str) -> bool:
        allowed_models = self.ray_tracing_allowed_execution_models(operation)
        if allowed_models is None:
            return False
        if self.current_execution_model is None:
            self.emit(
                f"; WARNING: SPIR-V ray tracing operation {operation} requires "
                "a ray tracing stage context"
            )
            return False
        if self.current_execution_model in allowed_models:
            return True

        allowed = ", ".join(sorted(allowed_models))
        self.emit(
            f"; WARNING: SPIR-V ray tracing operation {operation} is only valid "
            f"in {allowed} stages"
        )
        return False

    def ray_tracing_acceleration_structure_value_from_expression(
        self, expr, operation: str
    ) -> Optional[SpirvId]:
        pointer = self.variable_pointer_from_expression(expr)
        if pointer is not None:
            if pointer.type.storage_class != "Function":
                self.mark_function_interface_variable(pointer)
            value = self.get_variable_value(pointer)
        else:
            value = self.process_expression(expr)
        if value is None:
            self.emit(
                f"; WARNING: SPIR-V {operation} acceleration structure argument "
                "could not be evaluated"
            )
            return None

        value_type = self.registered_value_type(value)
        type_name = (
            value_type.type.base_type
            if value_type is not None
            else value.type.base_type
        )
        if not self.is_acceleration_structure_type_name(type_name):
            self.emit(
                f"; WARNING: SPIR-V {operation} acceleration structure argument "
                f"must be accelerationStructureEXT, got {type_name}"
            )
            return None

        return value

    def ray_tracing_uint_operand(
        self, expr, operation: str, role: str
    ) -> Optional[SpirvId]:
        value = self.process_expression(expr)
        if value is None:
            self.emit(
                f"; WARNING: SPIR-V {operation} {role} argument could not be "
                "evaluated"
            )
            return None

        value_type = self.registered_value_type(value) or self.ensure_registered_type(
            value.type
        )
        if self.vector_component_type_and_count(value_type.type.base_type) is not None:
            self.emit(
                f"; WARNING: SPIR-V {operation} {role} argument must be a "
                "32-bit integer scalar"
            )
            return None

        component_type = self.normalize_primitive_name(value_type.type.base_type)
        if component_type not in {"int", "uint"}:
            self.emit(
                f"; WARNING: SPIR-V {operation} {role} argument must be a "
                f"32-bit integer scalar, got {component_type}"
            )
            return None

        return self.convert_value_to_type(value, self.register_primitive_type("uint"))

    def ray_tracing_float_operand(
        self, expr, operation: str, role: str
    ) -> Optional[SpirvId]:
        value = self.process_expression(expr)
        if value is None:
            self.emit(
                f"; WARNING: SPIR-V {operation} {role} argument could not be "
                "evaluated"
            )
            return None

        value_type = self.registered_value_type(value) or self.ensure_registered_type(
            value.type
        )
        if self.vector_component_type_and_count(value_type.type.base_type) is not None:
            self.emit(
                f"; WARNING: SPIR-V {operation} {role} argument must be a "
                "32-bit floating-point scalar"
            )
            return None

        component_type = self.normalize_primitive_name(value_type.type.base_type)
        if component_type not in {"float", "double", "int", "uint"}:
            self.emit(
                f"; WARNING: SPIR-V {operation} {role} argument must be a "
                f"32-bit floating-point scalar, got {component_type}"
            )
            return None

        return self.convert_value_to_type(value, self.register_primitive_type("float"))

    def ray_tracing_vec3_operand(
        self, expr, operation: str, role: str
    ) -> Optional[SpirvId]:
        value = self.process_expression(expr)
        if value is None:
            self.emit(
                f"; WARNING: SPIR-V {operation} {role} argument could not be "
                "evaluated"
            )
            return None

        float_type = self.register_primitive_type("float")
        vec3_type = self.register_vector_type(float_type, 3)
        value = self.convert_value_to_type(value, vec3_type)
        vector_info = self.vector_component_type_and_count(value.type.base_type)
        if vector_info != ("float", 3):
            self.emit(
                f"; WARNING: SPIR-V {operation} {role} argument must be a "
                "32-bit floating-point vec3"
            )
            return None

        return value

    def ray_tracing_ray_desc_member_expression(
        self, ray_desc_expr, field_names, operation: str
    ) -> Optional[MemberAccessNode]:
        ray_desc_pointer = self.variable_pointer_from_expression(ray_desc_expr)
        if ray_desc_pointer is None:
            self.emit(
                f"; WARNING: SPIR-V {operation} RayDesc argument must be an "
                "addressable value"
            )
            return None

        ray_desc_type = self.pointer_pointee_type(ray_desc_pointer)
        members = self.current_struct_members.get(
            ray_desc_type.type.base_type if ray_desc_type is not None else None, []
        )
        available_names = {member_name for _, member_name in members}
        for field_name in field_names:
            if field_name in available_names:
                return MemberAccessNode(ray_desc_expr, field_name)

        expected = "/".join(field_names)
        self.emit(
            f"; WARNING: SPIR-V {operation} RayDesc argument does not provide "
            f"{expected}"
        )
        return None

    def trace_ray_argument_expressions(self, arguments):
        if len(arguments) == 11:
            return tuple(arguments)

        acceleration, ray_flags, cull_mask, sbt_offset, sbt_stride, miss_index = (
            arguments[:6]
        )
        ray_desc = arguments[6]
        payload = arguments[7]
        origin = self.ray_tracing_ray_desc_member_expression(
            ray_desc, ("Origin", "origin", "rayOrigin", "RayOrigin"), "TraceRay"
        )
        tmin = self.ray_tracing_ray_desc_member_expression(
            ray_desc, ("TMin", "tMin", "Tmin", "tmin"), "TraceRay"
        )
        direction = self.ray_tracing_ray_desc_member_expression(
            ray_desc,
            ("Direction", "direction", "rayDirection", "RayDirection"),
            "TraceRay",
        )
        tmax = self.ray_tracing_ray_desc_member_expression(
            ray_desc, ("TMax", "tMax", "Tmax", "tmax"), "TraceRay"
        )
        if None in {origin, tmin, direction, tmax}:
            return None

        return (
            acceleration,
            ray_flags,
            cull_mask,
            sbt_offset,
            sbt_stride,
            miss_index,
            origin,
            tmin,
            direction,
            tmax,
            payload,
        )

    def ray_tracing_storage_variable_name(self, pointer: SpirvId, storage_class: str):
        base_name = pointer.name or f"value_{pointer.id}"
        suffix = re.sub(r"[^A-Za-z0-9_]", "_", storage_class).lower()
        return f"{base_name}_{suffix}"

    def ray_tracing_storage_pointer(
        self, expr, storage_class: str, operation: str, role: str
    ) -> Tuple[Optional[SpirvId], Optional[SpirvId]]:
        pointer = self.variable_pointer_from_expression(expr)
        if pointer is None:
            self.emit(
                f"; WARNING: SPIR-V {operation} {role} argument must be an "
                "addressable value"
            )
            return None, None

        value_type = self.pointer_pointee_type(pointer)
        if value_type is None:
            self.emit(
                f"; WARNING: SPIR-V {operation} {role} argument has no value type"
            )
            return None, None

        if pointer.type.storage_class == storage_class:
            self.mark_function_interface_variable(pointer)
            return pointer, None

        key = (storage_class, pointer.id)
        storage_pointer = self.ray_tracing_storage_variables.get(key)
        if storage_pointer is None:
            storage_pointer = self.create_variable(
                value_type,
                storage_class,
                self.ray_tracing_storage_variable_name(pointer, storage_class),
            )
            self.ray_tracing_storage_variables[key] = storage_pointer
        self.mark_function_interface_variable(storage_pointer)

        value = self.get_variable_value(pointer)
        if value is not None:
            self.store_to_variable(
                storage_pointer, self.convert_value_to_type(value, value_type)
            )

        return storage_pointer, pointer

    def copy_ray_tracing_storage_back(
        self, storage_pointer: Optional[SpirvId], destination: Optional[SpirvId]
    ) -> None:
        if storage_pointer is None or destination is None:
            return

        destination_type = self.pointer_pointee_type(destination)
        value = self.get_variable_value(storage_pointer)
        if value is None:
            return
        if destination_type is not None:
            value = self.convert_value_to_type(value, destination_type)
        self.store_to_variable(destination, value)

    def process_trace_ray_operation(self, arguments) -> Optional[SpirvId]:
        expressions = self.trace_ray_argument_expressions(arguments)
        if expressions is None:
            return self.ray_tracing_default_value("TraceRay")

        (
            acceleration_expr,
            ray_flags_expr,
            cull_mask_expr,
            sbt_offset_expr,
            sbt_stride_expr,
            miss_index_expr,
            origin_expr,
            tmin_expr,
            direction_expr,
            tmax_expr,
            payload_expr,
        ) = expressions

        acceleration = self.ray_tracing_acceleration_structure_value_from_expression(
            acceleration_expr, "TraceRay"
        )
        ray_flags = self.ray_tracing_uint_operand(
            ray_flags_expr, "TraceRay", "ray flags"
        )
        cull_mask = self.ray_tracing_uint_operand(
            cull_mask_expr, "TraceRay", "cull mask"
        )
        sbt_offset = self.ray_tracing_uint_operand(
            sbt_offset_expr, "TraceRay", "shader binding table offset"
        )
        sbt_stride = self.ray_tracing_uint_operand(
            sbt_stride_expr, "TraceRay", "shader binding table stride"
        )
        miss_index = self.ray_tracing_uint_operand(
            miss_index_expr, "TraceRay", "miss shader index"
        )
        origin = self.ray_tracing_vec3_operand(origin_expr, "TraceRay", "origin")
        tmin = self.ray_tracing_float_operand(tmin_expr, "TraceRay", "Tmin")
        direction = self.ray_tracing_vec3_operand(
            direction_expr, "TraceRay", "direction"
        )
        tmax = self.ray_tracing_float_operand(tmax_expr, "TraceRay", "Tmax")
        payload, payload_destination = self.ray_tracing_storage_pointer(
            payload_expr, "RayPayloadKHR", "TraceRay", "payload"
        )

        operands = {
            acceleration,
            ray_flags,
            cull_mask,
            sbt_offset,
            sbt_stride,
            miss_index,
            origin,
            tmin,
            direction,
            tmax,
            payload,
        }
        if None in operands:
            return self.ray_tracing_default_value("TraceRay")

        self.emit(
            f"OpTraceRayKHR %{acceleration.id} %{ray_flags.id} %{cull_mask.id} "
            f"%{sbt_offset.id} %{sbt_stride.id} %{miss_index.id} %{origin.id} "
            f"%{tmin.id} %{direction.id} %{tmax.id} %{payload.id}"
        )
        self.copy_ray_tracing_storage_back(payload, payload_destination)
        return None

    def process_call_shader_operation(self, arguments) -> Optional[SpirvId]:
        shader_index = self.ray_tracing_uint_operand(
            arguments[0], "CallShader", "shader index"
        )
        callable_data, callable_destination = self.ray_tracing_storage_pointer(
            arguments[1], "CallableDataKHR", "CallShader", "callable data"
        )
        if shader_index is None or callable_data is None:
            return self.ray_tracing_default_value("CallShader")

        self.emit(f"OpExecuteCallableKHR %{shader_index.id} %{callable_data.id}")
        self.copy_ray_tracing_storage_back(callable_data, callable_destination)
        return None

    def process_report_hit_operation(self, arguments) -> SpirvId:
        hit_t = self.ray_tracing_float_operand(arguments[0], "ReportHit", "hit T")
        hit_kind = self.ray_tracing_uint_operand(arguments[1], "ReportHit", "hit kind")
        if len(arguments) == 3:
            self.ray_tracing_storage_pointer(
                arguments[2], "HitAttributeKHR", "ReportHit", "hit attribute"
            )
        if hit_t is None or hit_kind is None:
            return self.ray_tracing_default_value("ReportHit")

        result_type = self.register_primitive_type("bool")
        id_value = self.get_id()
        self.emit(
            f"%{id_value} = OpReportIntersectionKHR %{result_type.id} "
            f"%{hit_t.id} %{hit_kind.id}"
        )
        self.value_types[id_value] = result_type
        return SpirvId(id_value, result_type.type)

    def process_ray_tracing_operation(
        self, expr: RayTracingOpNode
    ) -> Optional[SpirvId]:
        operation = expr.operation
        arguments = getattr(expr, "arguments", getattr(expr, "args", [])) or []
        supported_argument_counts = {
            "TraceRay": {8, 11},
            "CallShader": {2},
            "ReportHit": {2, 3},
            "IgnoreHit": {0},
            "AcceptHitAndEndSearch": {0},
        }

        expected_counts = supported_argument_counts.get(operation)
        if expected_counts is None:
            return self.represented_ir_diagnostic_default_value(
                "ray tracing", operation
            )
        if len(arguments) not in expected_counts:
            expected = self.format_expected_argument_counts(expected_counts)
            self.emit(
                f"; WARNING: SPIR-V ray tracing operation {operation} requires "
                f"{expected} arguments"
            )
            return self.ray_tracing_default_value(operation)
        if not self.validate_ray_tracing_operation_context(operation):
            return self.ray_tracing_default_value(operation)

        self.require_capability("RayTracingKHR")
        self.require_extension("SPV_KHR_ray_tracing")

        if operation == "TraceRay":
            return self.process_trace_ray_operation(arguments)
        if operation == "CallShader":
            return self.process_call_shader_operation(arguments)
        if operation == "ReportHit":
            return self.process_report_hit_operation(arguments)
        if operation == "IgnoreHit":
            self.emit("OpIgnoreIntersectionKHR")
            return None
        if operation == "AcceptHitAndEndSearch":
            self.emit("OpTerminateRayKHR")
            return None

        return self.represented_ir_diagnostic_default_value("ray tracing", operation)

    def ray_query_pointer_from_expression(self, expr) -> Optional[SpirvId]:
        query_pointer = self.variable_pointer_from_expression(expr)
        if query_pointer is None:
            return None

        query_type = self.pointer_pointee_type(query_pointer)
        if query_type is None:
            return None

        if not self.is_ray_query_type_name(query_type.type.base_type):
            return None

        return query_pointer

    def ray_query_intersection_selector(self, operation: str) -> Optional[int]:
        if operation.startswith("Candidate"):
            return 0
        if operation.startswith("Committed"):
            return 1
        return None

    def ray_query_intersection_constant(self, operation: str) -> Optional[SpirvId]:
        selector = self.ray_query_intersection_selector(operation)
        if selector is None:
            return None

        return self.register_constant(selector, self.register_primitive_type("uint"))

    def ray_query_result_type_for_kind(self, result_kind: str) -> SpirvId:
        if result_kind == "bool":
            return self.register_primitive_type("bool")
        if result_kind == "float":
            return self.register_primitive_type("float")
        if result_kind == "vec2":
            return self.register_vector_type(self.register_primitive_type("float"), 2)
        if result_kind == "vec3":
            return self.register_vector_type(self.register_primitive_type("float"), 3)
        if result_kind == "vec3_array3":
            vec3_type = self.register_vector_type(
                self.register_primitive_type("float"), 3
            )
            return self.register_array_type(vec3_type, 3)
        if result_kind == "mat4x3":
            column_type = self.register_vector_type(
                self.register_primitive_type("float"), 3
            )
            return self.register_matrix_type(column_type, 4)
        return self.register_primitive_type("uint")

    def ray_query_state_getter_info(self, operation: str):
        getters = {
            "RayTMin": ("OpRayQueryGetRayTMinKHR", "float"),
            "RayFlags": ("OpRayQueryGetRayFlagsKHR", "uint"),
            "WorldRayOrigin": ("OpRayQueryGetWorldRayOriginKHR", "vec3"),
            "WorldRayDirection": ("OpRayQueryGetWorldRayDirectionKHR", "vec3"),
        }
        return getters.get(operation)

    def ray_query_state_getter_operations(self):
        return {
            "RayTMin",
            "RayFlags",
            "WorldRayOrigin",
            "WorldRayDirection",
        }

    def ray_query_candidate_getter_info(self, operation: str):
        getters = {
            "CandidateAABBOpaque": (
                "OpRayQueryGetIntersectionCandidateAABBOpaqueKHR",
                "bool",
            ),
        }
        return getters.get(operation)

    def ray_query_candidate_getter_operations(self):
        return {
            "CandidateAABBOpaque",
        }

    def ray_query_intersection_getter_info(self, operation: str):
        getters = {
            "CandidateType": ("OpRayQueryGetIntersectionTypeKHR", "uint"),
            "CommittedType": ("OpRayQueryGetIntersectionTypeKHR", "uint"),
            "CandidateRayT": ("OpRayQueryGetIntersectionTKHR", "float"),
            "CommittedRayT": ("OpRayQueryGetIntersectionTKHR", "float"),
            "CandidatePrimitiveIndex": (
                "OpRayQueryGetIntersectionPrimitiveIndexKHR",
                "uint",
            ),
            "CommittedPrimitiveIndex": (
                "OpRayQueryGetIntersectionPrimitiveIndexKHR",
                "uint",
            ),
            "CandidateInstanceID": (
                "OpRayQueryGetIntersectionInstanceIdKHR",
                "uint",
            ),
            "CommittedInstanceID": (
                "OpRayQueryGetIntersectionInstanceIdKHR",
                "uint",
            ),
            "CandidateInstanceCustomIndex": (
                "OpRayQueryGetIntersectionInstanceCustomIndexKHR",
                "uint",
            ),
            "CommittedInstanceCustomIndex": (
                "OpRayQueryGetIntersectionInstanceCustomIndexKHR",
                "uint",
            ),
            "CandidateInstanceShaderBindingTableRecordOffset": (
                "OpRayQueryGetIntersectionInstanceShaderBindingTableRecordOffsetKHR",
                "uint",
            ),
            "CommittedInstanceShaderBindingTableRecordOffset": (
                "OpRayQueryGetIntersectionInstanceShaderBindingTableRecordOffsetKHR",
                "uint",
            ),
            "CandidateObjectRayOrigin": (
                "OpRayQueryGetIntersectionObjectRayOriginKHR",
                "vec3",
            ),
            "CommittedObjectRayOrigin": (
                "OpRayQueryGetIntersectionObjectRayOriginKHR",
                "vec3",
            ),
            "CandidateObjectRayDirection": (
                "OpRayQueryGetIntersectionObjectRayDirectionKHR",
                "vec3",
            ),
            "CommittedObjectRayDirection": (
                "OpRayQueryGetIntersectionObjectRayDirectionKHR",
                "vec3",
            ),
            "CandidateGeometryIndex": (
                "OpRayQueryGetIntersectionGeometryIndexKHR",
                "uint",
            ),
            "CommittedGeometryIndex": (
                "OpRayQueryGetIntersectionGeometryIndexKHR",
                "uint",
            ),
            "CandidateTriangleBarycentrics": (
                "OpRayQueryGetIntersectionBarycentricsKHR",
                "vec2",
            ),
            "CommittedTriangleBarycentrics": (
                "OpRayQueryGetIntersectionBarycentricsKHR",
                "vec2",
            ),
            "CandidateTriangleFrontFace": (
                "OpRayQueryGetIntersectionFrontFaceKHR",
                "bool",
            ),
            "CommittedTriangleFrontFace": (
                "OpRayQueryGetIntersectionFrontFaceKHR",
                "bool",
            ),
            "CandidateTriangleNormal": ("", "vec3"),
            "CommittedTriangleNormal": ("", "vec3"),
            "CandidateTriangleArea": ("", "float"),
            "CommittedTriangleArea": ("", "float"),
            "CandidateTriangleCentroid": ("", "vec3"),
            "CommittedTriangleCentroid": ("", "vec3"),
            "CandidateTriangleVertexPositions": (
                "OpRayQueryGetIntersectionTriangleVertexPositionsKHR",
                "vec3_array3",
            ),
            "CommittedTriangleVertexPositions": (
                "OpRayQueryGetIntersectionTriangleVertexPositionsKHR",
                "vec3_array3",
            ),
            "CandidateObjectToWorld": (
                "OpRayQueryGetIntersectionObjectToWorldKHR",
                "mat4x3",
            ),
            "CommittedObjectToWorld": (
                "OpRayQueryGetIntersectionObjectToWorldKHR",
                "mat4x3",
            ),
            "CandidateObjectToWorld3x4": (
                "OpRayQueryGetIntersectionObjectToWorldKHR",
                "mat4x3",
            ),
            "CommittedObjectToWorld3x4": (
                "OpRayQueryGetIntersectionObjectToWorldKHR",
                "mat4x3",
            ),
            "CandidateWorldToObject": (
                "OpRayQueryGetIntersectionWorldToObjectKHR",
                "mat4x3",
            ),
            "CommittedWorldToObject": (
                "OpRayQueryGetIntersectionWorldToObjectKHR",
                "mat4x3",
            ),
            "CandidateWorldToObject3x4": (
                "OpRayQueryGetIntersectionWorldToObjectKHR",
                "mat4x3",
            ),
            "CommittedWorldToObject3x4": (
                "OpRayQueryGetIntersectionWorldToObjectKHR",
                "mat4x3",
            ),
        }
        getter = getters.get(operation)
        if getter is None:
            return None

        selector = self.ray_query_intersection_selector(operation)
        if selector is None:
            return None

        opcode, result_kind = getter
        return opcode, result_kind, selector

    def ray_query_intersection_getter_operations(self):
        return {
            "CandidateType",
            "CommittedType",
            "CandidateRayT",
            "CommittedRayT",
            "CandidatePrimitiveIndex",
            "CommittedPrimitiveIndex",
            "CandidateInstanceID",
            "CommittedInstanceID",
            "CandidateInstanceCustomIndex",
            "CommittedInstanceCustomIndex",
            "CandidateInstanceShaderBindingTableRecordOffset",
            "CommittedInstanceShaderBindingTableRecordOffset",
            "CandidateObjectRayOrigin",
            "CommittedObjectRayOrigin",
            "CandidateObjectRayDirection",
            "CommittedObjectRayDirection",
            "CandidateGeometryIndex",
            "CommittedGeometryIndex",
            "CandidateTriangleBarycentrics",
            "CommittedTriangleBarycentrics",
            "CandidateTriangleFrontFace",
            "CommittedTriangleFrontFace",
            "CandidateTriangleNormal",
            "CommittedTriangleNormal",
            "CandidateTriangleArea",
            "CommittedTriangleArea",
            "CandidateTriangleCentroid",
            "CommittedTriangleCentroid",
            "CandidateTriangleVertexPositions",
            "CommittedTriangleVertexPositions",
            "CandidateObjectToWorld",
            "CommittedObjectToWorld",
            "CandidateObjectToWorld3x4",
            "CommittedObjectToWorld3x4",
            "CandidateWorldToObject",
            "CommittedWorldToObject",
            "CandidateWorldToObject3x4",
            "CommittedWorldToObject3x4",
        }

    def process_ray_query_state_getter(
        self, query_pointer: SpirvId, operation: str
    ) -> SpirvId:
        getter_info = self.ray_query_state_getter_info(operation)
        if getter_info is None:
            return self.represented_ir_diagnostic_default_value("ray query", operation)

        opcode, result_kind = getter_info
        result_type = self.ray_query_result_type_for_kind(result_kind)
        id_value = self.get_id()
        self.emit(f"%{id_value} = {opcode} %{result_type.id} %{query_pointer.id}")
        self.value_types[id_value] = result_type
        return SpirvId(id_value, result_type.type)

    def process_ray_query_candidate_getter(
        self, query_pointer: SpirvId, operation: str
    ) -> SpirvId:
        getter_info = self.ray_query_candidate_getter_info(operation)
        if getter_info is None:
            return self.represented_ir_diagnostic_default_value("ray query", operation)

        opcode, result_kind = getter_info
        result_type = self.ray_query_result_type_for_kind(result_kind)
        id_value = self.get_id()
        self.emit(f"%{id_value} = {opcode} %{result_type.id} %{query_pointer.id}")
        self.value_types[id_value] = result_type
        return SpirvId(id_value, result_type.type)

    def process_ray_query_intersection_getter(
        self, query_pointer: SpirvId, operation: str
    ) -> SpirvId:
        getter_info = self.ray_query_intersection_getter_info(operation)
        if getter_info is None:
            return self.represented_ir_diagnostic_default_value("ray query", operation)

        if operation in {"CandidateTriangleNormal", "CommittedTriangleNormal"}:
            return self.process_ray_query_triangle_normal_getter(
                query_pointer, operation
            )
        if operation in {"CandidateTriangleArea", "CommittedTriangleArea"}:
            return self.process_ray_query_triangle_area_getter(query_pointer, operation)
        if operation in {"CandidateTriangleCentroid", "CommittedTriangleCentroid"}:
            return self.process_ray_query_triangle_centroid_getter(
                query_pointer, operation
            )

        opcode, result_kind, _ = getter_info
        if operation in {
            "CandidateTriangleVertexPositions",
            "CommittedTriangleVertexPositions",
        }:
            self.require_capability("RayQueryPositionFetchKHR")
            self.require_extension("SPV_KHR_ray_tracing_position_fetch")

        intersection = self.ray_query_intersection_constant(operation)
        if intersection is None:
            return self.represented_ir_diagnostic_default_value("ray query", operation)

        result_type = self.ray_query_result_type_for_kind(result_kind)
        id_value = self.get_id()
        self.emit(
            f"%{id_value} = {opcode} %{result_type.id} "
            f"%{query_pointer.id} %{intersection.id}"
        )
        self.value_types[id_value] = result_type
        return SpirvId(id_value, result_type.type)

    def emit_glsl_std450_instruction(
        self, instruction: str, result_type: SpirvId, args: List[SpirvId]
    ) -> SpirvId:
        if self.glsl_std450_id is None:
            self.glsl_std450_id = self.get_id()
            self.emit(f'%{self.glsl_std450_id} = OpExtInstImport "GLSL.std.450"')

        id_value = self.get_id()
        arg_list = " ".join(f"%{arg.id}" for arg in args)
        self.emit(
            f"%{id_value} = OpExtInst %{result_type.id} "
            f"%{self.glsl_std450_id} {instruction} {arg_list}"
        )
        self.value_types[id_value] = result_type
        return SpirvId(id_value, result_type.type)

    def ray_query_triangle_vertices(
        self, query_pointer: SpirvId, operation: str
    ) -> Optional[Tuple[SpirvId, SpirvId, SpirvId, SpirvId]]:
        intersection = self.ray_query_intersection_constant(operation)
        if intersection is None:
            return None

        self.require_capability("RayQueryPositionFetchKHR")
        self.require_extension("SPV_KHR_ray_tracing_position_fetch")

        vec3_type = self.ray_query_result_type_for_kind("vec3")
        positions_type = self.ray_query_result_type_for_kind("vec3_array3")
        positions_id = self.get_id()
        self.emit(
            f"%{positions_id} = "
            f"OpRayQueryGetIntersectionTriangleVertexPositionsKHR "
            f"%{positions_type.id} %{query_pointer.id} %{intersection.id}"
        )
        self.value_types[positions_id] = positions_type
        positions = SpirvId(positions_id, positions_type.type)

        p0 = self.composite_extract(positions, vec3_type, 0)
        p1 = self.composite_extract(positions, vec3_type, 1)
        p2 = self.composite_extract(positions, vec3_type, 2)
        return vec3_type, p0, p1, p2

    def ray_query_triangle_edges(
        self, query_pointer: SpirvId, operation: str
    ) -> Optional[Tuple[SpirvId, SpirvId, SpirvId]]:
        vertices = self.ray_query_triangle_vertices(query_pointer, operation)
        if vertices is None:
            return None

        vec3_type, p0, p1, p2 = vertices
        edge0 = self.binary_operation("-", vec3_type, p1, p0)
        edge1 = self.binary_operation("-", vec3_type, p2, p0)
        return vec3_type, edge0, edge1

    def process_ray_query_triangle_normal_getter(
        self, query_pointer: SpirvId, operation: str
    ) -> SpirvId:
        edges = self.ray_query_triangle_edges(query_pointer, operation)
        if edges is None:
            return self.represented_ir_diagnostic_default_value("ray query", operation)

        vec3_type, edge0, edge1 = edges
        cross = self.emit_glsl_std450_instruction("Cross", vec3_type, [edge0, edge1])
        return self.emit_glsl_std450_instruction("Normalize", vec3_type, [cross])

    def process_ray_query_triangle_area_getter(
        self, query_pointer: SpirvId, operation: str
    ) -> SpirvId:
        edges = self.ray_query_triangle_edges(query_pointer, operation)
        if edges is None:
            return self.represented_ir_diagnostic_default_value("ray query", operation)

        vec3_type, edge0, edge1 = edges
        float_type = self.ray_query_result_type_for_kind("float")
        cross = self.emit_glsl_std450_instruction("Cross", vec3_type, [edge0, edge1])
        double_area = self.emit_glsl_std450_instruction("Length", float_type, [cross])
        half = self.register_constant(0.5, float_type)
        return self.binary_operation("*", float_type, double_area, half)

    def process_ray_query_triangle_centroid_getter(
        self, query_pointer: SpirvId, operation: str
    ) -> SpirvId:
        vertices = self.ray_query_triangle_vertices(query_pointer, operation)
        if vertices is None:
            return self.represented_ir_diagnostic_default_value("ray query", operation)

        vec3_type, p0, p1, p2 = vertices
        float_type = self.ray_query_result_type_for_kind("float")
        p0_plus_p1 = self.binary_operation("+", vec3_type, p0, p1)
        total = self.binary_operation("+", vec3_type, p0_plus_p1, p2)
        three = self.register_constant(3.0, float_type)
        return self.binary_operation("/", vec3_type, total, three)

    def format_expected_argument_counts(self, expected_counts) -> str:
        return " or ".join(str(count) for count in sorted(expected_counts))

    def registered_value_type(self, value_id: SpirvId) -> Optional[SpirvId]:
        return self.value_types.get(value_id.id) or self.find_registered_type_by_base(
            value_id.type.base_type
        )

    def acceleration_structure_value_from_expression(self, expr) -> Optional[SpirvId]:
        value = self.process_expression(expr)
        if value is None:
            self.emit(
                "; WARNING: SPIR-V RayQuery.TraceRayInline acceleration structure "
                "argument could not be evaluated"
            )
            return None

        value_type = self.registered_value_type(value)
        type_name = (
            value_type.type.base_type
            if value_type is not None
            else value.type.base_type
        )
        if not self.is_acceleration_structure_type_name(type_name):
            self.emit(
                "; WARNING: SPIR-V RayQuery.TraceRayInline acceleration structure "
                f"argument must be accelerationStructureEXT, got {type_name}"
            )
            return None

        return value

    def ray_query_uint_operand(self, expr, role: str) -> Optional[SpirvId]:
        value = self.process_expression(expr)
        if value is None:
            self.emit(
                f"; WARNING: SPIR-V RayQuery.TraceRayInline {role} argument "
                "could not be evaluated"
            )
            return None

        value_type = self.registered_value_type(value) or self.ensure_registered_type(
            value.type
        )
        if self.vector_component_type_and_count(value_type.type.base_type) is not None:
            self.emit(
                f"; WARNING: SPIR-V RayQuery.TraceRayInline {role} argument "
                "must be a 32-bit integer scalar"
            )
            return None

        component_type = self.normalize_primitive_name(value_type.type.base_type)
        if component_type not in {"int", "uint"}:
            self.emit(
                f"; WARNING: SPIR-V RayQuery.TraceRayInline {role} argument "
                f"must be a 32-bit integer scalar, got {component_type}"
            )
            return None

        return self.convert_value_to_type(value, self.register_primitive_type("uint"))

    def ray_query_float_operand(
        self, expr, role: str, operation: str = "TraceRayInline"
    ) -> Optional[SpirvId]:
        value = self.process_expression(expr)
        if value is None:
            self.emit(
                f"; WARNING: SPIR-V RayQuery.{operation} {role} argument "
                "could not be evaluated"
            )
            return None

        value_type = self.registered_value_type(value) or self.ensure_registered_type(
            value.type
        )
        if self.vector_component_type_and_count(value_type.type.base_type) is not None:
            self.emit(
                f"; WARNING: SPIR-V RayQuery.{operation} {role} argument "
                "must be a 32-bit floating-point scalar"
            )
            return None

        component_type = self.normalize_primitive_name(value_type.type.base_type)
        if component_type not in {"float", "double", "int", "uint"}:
            self.emit(
                f"; WARNING: SPIR-V RayQuery.{operation} {role} argument "
                f"must be a 32-bit floating-point scalar, got {component_type}"
            )
            return None

        return self.convert_value_to_type(value, self.register_primitive_type("float"))

    def ray_query_vec3_operand(self, expr, role: str) -> Optional[SpirvId]:
        value = self.process_expression(expr)
        if value is None:
            self.emit(
                f"; WARNING: SPIR-V RayQuery.TraceRayInline {role} argument "
                "could not be evaluated"
            )
            return None

        float_type = self.register_primitive_type("float")
        vec3_type = self.register_vector_type(float_type, 3)
        value = self.convert_value_to_type(value, vec3_type)
        vector_info = self.vector_component_type_and_count(value.type.base_type)
        if vector_info != ("float", 3):
            self.emit(
                f"; WARNING: SPIR-V RayQuery.TraceRayInline {role} argument "
                "must be a 32-bit floating-point vec3"
            )
            return None

        return value

    def ray_desc_member_expression(
        self, ray_desc_expr, field_names
    ) -> Optional[MemberAccessNode]:
        ray_desc_pointer = self.variable_pointer_from_expression(ray_desc_expr)
        if ray_desc_pointer is None:
            self.emit(
                "; WARNING: SPIR-V RayQuery.TraceRayInline RayDesc argument "
                "must be an addressable value"
            )
            return None

        ray_desc_type = self.pointer_pointee_type(ray_desc_pointer)
        members = self.current_struct_members.get(
            ray_desc_type.type.base_type if ray_desc_type is not None else None, []
        )
        available_names = {member_name for _, member_name in members}
        for field_name in field_names:
            if field_name in available_names:
                return MemberAccessNode(ray_desc_expr, field_name)

        expected = "/".join(field_names)
        self.emit(
            "; WARNING: SPIR-V RayQuery.TraceRayInline RayDesc argument "
            f"does not provide {expected}"
        )
        return None

    def ray_query_initialize_argument_expressions(self, arguments):
        if len(arguments) == 7:
            return tuple(arguments)

        acceleration, ray_flags, cull_mask, ray_desc = arguments
        origin = self.ray_desc_member_expression(
            ray_desc, ("Origin", "origin", "rayOrigin", "RayOrigin")
        )
        tmin = self.ray_desc_member_expression(
            ray_desc, ("TMin", "tMin", "Tmin", "tmin")
        )
        direction = self.ray_desc_member_expression(
            ray_desc, ("Direction", "direction", "rayDirection", "RayDirection")
        )
        tmax = self.ray_desc_member_expression(
            ray_desc, ("TMax", "tMax", "Tmax", "tmax")
        )
        if None in {origin, tmin, direction, tmax}:
            return None
        return acceleration, ray_flags, cull_mask, origin, tmin, direction, tmax

    def process_ray_query_initialize(self, query_pointer: SpirvId, arguments) -> None:
        expressions = self.ray_query_initialize_argument_expressions(arguments)
        if expressions is None:
            return None

        (
            acceleration_expr,
            ray_flags_expr,
            cull_mask_expr,
            origin_expr,
            tmin_expr,
            direction_expr,
            tmax_expr,
        ) = expressions

        acceleration = self.acceleration_structure_value_from_expression(
            acceleration_expr
        )
        ray_flags = self.ray_query_uint_operand(ray_flags_expr, "ray flags")
        cull_mask = self.ray_query_uint_operand(cull_mask_expr, "cull mask")
        origin = self.ray_query_vec3_operand(origin_expr, "origin")
        tmin = self.ray_query_float_operand(tmin_expr, "Tmin")
        direction = self.ray_query_vec3_operand(direction_expr, "direction")
        tmax = self.ray_query_float_operand(tmax_expr, "Tmax")

        if None in {acceleration, ray_flags, cull_mask, origin, tmin, direction, tmax}:
            return None

        self.emit(
            f"OpRayQueryInitializeKHR %{query_pointer.id} %{acceleration.id} "
            f"%{ray_flags.id} %{cull_mask.id} %{origin.id} %{tmin.id} "
            f"%{direction.id} %{tmax.id}"
        )
        return None

    def ray_query_method_names(self):
        return (
            {
                "Abort",
                "CommitNonOpaqueTriangleHit",
                "CommitProceduralPrimitiveHit",
                "ConfirmIntersection",
                "Proceed",
                "GenerateIntersection",
                "Terminate",
                "TraceRayInline",
            }
            | self.ray_query_state_getter_operations()
            | self.ray_query_candidate_getter_operations()
            | (self.ray_query_intersection_getter_operations())
        )

    def ray_query_call_from_function_call(self, expr) -> Optional[RayQueryOpNode]:
        if not isinstance(expr, FunctionCallNode):
            return None

        func_expr = getattr(expr, "function", getattr(expr, "name", None))
        if not isinstance(func_expr, MemberAccessNode):
            return None

        operation = str(getattr(func_expr, "member", ""))
        if operation not in self.ray_query_method_names():
            return None

        return RayQueryOpNode(
            operation,
            getattr(func_expr, "object", getattr(func_expr, "object_expr", None)),
            getattr(expr, "arguments", getattr(expr, "args", [])),
        )

    def process_ray_query_operation(self, expr: RayQueryOpNode) -> SpirvId:
        operation = expr.operation
        arguments = getattr(expr, "args", getattr(expr, "arguments", [])) or []

        supported_argument_counts = {
            "Proceed": 0,
            "Abort": 0,
            "Terminate": 0,
            "ConfirmIntersection": 0,
            "CommitNonOpaqueTriangleHit": 0,
            "GenerateIntersection": 1,
            "CommitProceduralPrimitiveHit": 1,
            "TraceRayInline": {4, 7},
        }
        for getter_operation in self.ray_query_state_getter_operations():
            supported_argument_counts[getter_operation] = 0
        for getter_operation in self.ray_query_candidate_getter_operations():
            supported_argument_counts[getter_operation] = 0
        for getter_operation in self.ray_query_intersection_getter_operations():
            supported_argument_counts[getter_operation] = 0
        if operation not in supported_argument_counts:
            return self.represented_ir_diagnostic_default_value("ray query", operation)

        expected_counts = supported_argument_counts[operation]
        if isinstance(expected_counts, int):
            expected_counts = {expected_counts}
        if len(arguments) not in expected_counts:
            self.emit(
                f"; WARNING: SPIR-V RayQuery.{operation} requires "
                f"{self.format_expected_argument_counts(expected_counts)} arguments"
            )
            if operation == "TraceRayInline":
                return None
            return self.default_value_for_type(
                self.represented_ir_diagnostic_result_type("ray query", operation)
            )

        query_pointer = self.ray_query_pointer_from_expression(expr.query_expr)
        if query_pointer is None:
            return self.represented_ir_diagnostic_default_value("ray query", operation)

        self.require_capability("RayQueryKHR")
        self.require_extension("SPV_KHR_ray_query")

        if operation == "TraceRayInline":
            return self.process_ray_query_initialize(query_pointer, arguments)

        if operation in {"Abort", "Terminate"}:
            self.emit(f"OpRayQueryTerminateKHR %{query_pointer.id}")
            return None

        if operation in {"ConfirmIntersection", "CommitNonOpaqueTriangleHit"}:
            self.emit(f"OpRayQueryConfirmIntersectionKHR %{query_pointer.id}")
            return None

        if operation in {"GenerateIntersection", "CommitProceduralPrimitiveHit"}:
            hit_t = self.ray_query_float_operand(
                arguments[0], "hit distance", operation=operation
            )
            if hit_t is None:
                return None
            self.emit(
                f"OpRayQueryGenerateIntersectionKHR %{query_pointer.id} %{hit_t.id}"
            )
            return None

        if operation == "Proceed":
            result_type = self.register_primitive_type("bool")
            id_value = self.get_id()
            self.emit(
                f"%{id_value} = OpRayQueryProceedKHR %{result_type.id} "
                f"%{query_pointer.id}"
            )
            self.value_types[id_value] = result_type
            return SpirvId(id_value, result_type.type)

        if operation in self.ray_query_state_getter_operations():
            return self.process_ray_query_state_getter(query_pointer, operation)

        if operation in self.ray_query_candidate_getter_operations():
            return self.process_ray_query_candidate_getter(query_pointer, operation)

        if operation in self.ray_query_intersection_getter_operations():
            return self.process_ray_query_intersection_getter(query_pointer, operation)

        return self.represented_ir_diagnostic_default_value("ray query", operation)

    def process_mesh_operation(self, expr: MeshOpNode) -> Optional[SpirvId]:
        """Process represented mesh/task shader intrinsics."""
        operation = expr.operation
        if operation == "DispatchMesh":
            return self.process_dispatch_mesh_operation(expr)
        if operation != "SetMeshOutputCounts":
            return self.represented_ir_diagnostic_default_value(
                "mesh shader", operation
            )

        if self.current_execution_model != "MeshEXT":
            return self.represented_ir_diagnostic_default_value(
                "mesh shader", operation
            )

        arguments = getattr(expr, "arguments", []) or []
        if len(arguments) != 2:
            self.emit(
                "; WARNING: SPIR-V mesh SetMeshOutputCounts requires exactly "
                "2 arguments"
            )
            return self.register_constant(0, self.register_primitive_type("uint"))

        uint_type = self.register_primitive_type("uint")
        vertex_count = self.process_expression(arguments[0])
        primitive_count = self.process_expression(arguments[1])
        if vertex_count is None or primitive_count is None:
            self.emit(
                "; WARNING: SPIR-V mesh SetMeshOutputCounts requires count operands"
            )
            return self.register_constant(0, uint_type)
        if not self.validate_mesh_count_operands(
            "SetMeshOutputCounts",
            [vertex_count, primitive_count],
            argument_exprs=arguments,
        ):
            return self.register_constant(0, uint_type)
        if not self.validate_mesh_set_output_count_limits(arguments):
            return self.register_constant(0, uint_type)

        vertex_count = self.convert_value_to_type(vertex_count, uint_type)
        primitive_count = self.convert_value_to_type(primitive_count, uint_type)

        self.require_capability("MeshShadingEXT")
        self.require_extension("SPV_EXT_mesh_shader")
        self.emit(f"OpSetMeshOutputsEXT %{vertex_count.id} %{primitive_count.id}")

        if self.current_function_id is not None:
            previous_vertices, previous_primitives = (
                self.mesh_output_counts_by_function.get(
                    self.current_function_id, (None, None)
                )
            )
            observed_vertices = self.literal_int_argument(arguments[0])
            observed_primitives = self.literal_int_argument(arguments[1])
            self.mesh_output_counts_by_function[self.current_function_id] = (
                self.max_optional_int(previous_vertices, observed_vertices),
                self.max_optional_int(previous_primitives, observed_primitives),
            )

        return self.register_constant(0, uint_type)

    def validate_mesh_set_output_count_limits(self, arguments: List) -> bool:
        """Reject literal mesh output counts above declared stage limits."""
        vertex_limit = self.mesh_stage_limit(self.current_stage, "max_vertices")
        primitive_limit = self.mesh_stage_limit(self.current_stage, "max_primitives")
        requested_vertices = self.literal_int_argument(arguments[0])
        requested_primitives = self.literal_int_argument(arguments[1])
        valid = True

        if (
            vertex_limit is not None
            and requested_vertices is not None
            and requested_vertices > vertex_limit
        ):
            self.emit(
                "; WARNING: SPIR-V mesh SetMeshOutputCounts vertex count "
                "exceeds declared max_vertices"
            )
            valid = False

        if (
            primitive_limit is not None
            and requested_primitives is not None
            and requested_primitives > primitive_limit
        ):
            self.emit(
                "; WARNING: SPIR-V mesh SetMeshOutputCounts primitive count "
                "exceeds declared max_primitives"
            )
            valid = False

        return valid

    def validate_mesh_count_operands(
        self,
        operation: str,
        operands: List[SpirvId],
        count_label: str = "count",
        argument_exprs: Optional[List] = None,
    ) -> bool:
        """Require SPIR-V mesh/task count operands to be scalar integer values."""
        if all(
            self.integer_value_component_count(operand) == 1 for operand in operands
        ):
            if argument_exprs is not None and any(
                (literal := self.literal_int_argument(argument)) is not None
                and literal < 0
                for argument in argument_exprs
            ):
                self.emit(
                    f"; WARNING: SPIR-V mesh {operation} {count_label} operands "
                    "must be non-negative integer values"
                )
                return False
            return True

        self.emit(
            f"; WARNING: SPIR-V mesh {operation} {count_label} operands must be "
            "scalar integer values"
        )
        return False

    def process_mesh_output_function_call(
        self, function_name: str, args: List
    ) -> Optional[SpirvId]:
        """Lower represented mesh output helper calls to SPIR-V Output stores."""
        uint_type = self.register_primitive_type("uint")
        if self.current_execution_model != "MeshEXT":
            self.emit(
                f"; WARNING: SPIR-V mesh {function_name} is only valid in mesh stages"
            )
            return self.register_constant(0, uint_type)

        if len(args) != 2:
            self.emit(
                f"; WARNING: SPIR-V mesh {function_name} requires exactly 2 arguments"
            )
            return self.register_constant(0, uint_type)

        if function_name == "SetVertex":
            self.process_mesh_set_vertex(args)
        elif function_name == "SetPrimitive":
            self.process_mesh_set_primitive(args)

        return self.register_constant(0, uint_type)

    def process_geometry_stream_function_call(
        self, function_name: str, args: List
    ) -> Optional[SpirvId]:
        """Lower geometry stream-control calls to SPIR-V instructions."""
        if self.current_execution_model != "Geometry":
            self.emit(
                f"; WARNING: SPIR-V geometry {function_name} is only valid in "
                "geometry stages"
            )
            return None

        if args:
            self.emit(
                f"; WARNING: SPIR-V geometry {function_name} requires no arguments"
            )
            return None

        opcode = {
            "EmitVertex": "OpEmitVertex",
            "EndPrimitive": "OpEndPrimitive",
        }[function_name]
        self.emit(opcode)
        return None

    def process_mesh_set_vertex(self, args: List):
        """Lower SetVertex(index, value) to a Position builtin output store."""
        index = self.process_expression(args[0])
        if index is None:
            self.emit("; WARNING: SPIR-V mesh SetVertex requires an index operand")
            return

        position_value = self.mesh_vertex_position_value(args[1])
        if position_value is None:
            return

        vertex_output = self.ensure_mesh_vertex_position_output(
            self.literal_int_argument(args[0])
        )
        if vertex_output is None:
            return

        int_type = self.register_primitive_type("int")
        member_index = self.register_constant(0, int_type)
        vec4_type = self.register_vector_type(self.register_primitive_type("float"), 4)
        ptr_type = self.register_pointer_type(vec4_type, "Output")
        access = self.access_chain(vertex_output, [index, member_index], ptr_type)
        self.variable_value_types[access.id] = vec4_type
        self.store_to_variable(access, position_value)

    def process_mesh_set_primitive(self, args: List):
        """Lower SetPrimitive(index, value) to primitive-index builtin output."""
        index = self.process_expression(args[0])
        if index is None:
            self.emit("; WARNING: SPIR-V mesh SetPrimitive requires an index operand")
            return

        primitive_output, element_type = self.ensure_mesh_primitive_index_output(
            self.literal_int_argument(args[0])
        )
        if primitive_output is None or element_type is None:
            return

        value = self.process_expression(args[1])
        if value is None:
            self.emit("; WARNING: SPIR-V mesh SetPrimitive requires a value operand")
            return

        value = self.convert_value_to_type(value, element_type)
        if not self.value_has_type(value, element_type):
            self.emit(
                "; WARNING: SPIR-V mesh SetPrimitive value does not match "
                f"{element_type.type.base_type}"
            )
            return

        ptr_type = self.register_pointer_type(element_type, "Output")
        access = self.access_chain(primitive_output, [index], ptr_type)
        self.variable_value_types[access.id] = element_type
        self.store_to_variable(access, value)

    def mesh_stage_current_output_limits(self) -> Tuple[int, int]:
        """Return current mesh output limits from observed counts and metadata."""
        observed_vertices, observed_primitives = (None, None)
        if self.current_function_id is not None:
            observed_vertices, observed_primitives = (
                self.mesh_output_counts_by_function.get(
                    self.current_function_id, (None, None)
                )
            )

        max_vertices = self.mesh_stage_limit(self.current_stage, "max_vertices")
        max_primitives = self.mesh_stage_limit(self.current_stage, "max_primitives")
        if observed_vertices is not None:
            max_vertices = max(observed_vertices, max_vertices or 0)
        if observed_primitives is not None:
            max_primitives = max(observed_primitives, max_primitives or 0)
        return max(1, max_vertices or 1), max(1, max_primitives or 1)

    def mesh_output_literal_index_limit(
        self, role: str, info: Optional[dict] = None
    ) -> Optional[int]:
        if info is not None and info.get("count") is not None:
            return int(info["count"])

        max_vertices, max_primitives = self.mesh_stage_current_output_limits()
        if role == "vertices":
            return max_vertices
        if role in {"indices", "primitives"}:
            return max_primitives
        return None

    def validate_mesh_output_literal_index(
        self,
        role: str,
        literal_index: Optional[int],
        diagnostic_name: str,
        info: Optional[dict] = None,
    ) -> bool:
        if literal_index is None:
            return True
        if literal_index < 0:
            self.emit(
                f"; WARNING: SPIR-V mesh {diagnostic_name} literal index "
                "must be non-negative"
            )
            return False

        limit = self.mesh_output_literal_index_limit(role, info)
        if limit is None or literal_index < limit:
            return True

        limit_label = "vertex" if role == "vertices" else "primitive"
        self.emit(
            f"; WARNING: SPIR-V mesh {diagnostic_name} literal index exceeds "
            f"the declared mesh {limit_label} output limit"
        )
        return False

    def ensure_mesh_vertex_position_output(
        self, literal_index: Optional[int] = None
    ) -> Optional[SpirvId]:
        """Create the mesh Position output array used by SetVertex."""
        if not self.validate_mesh_output_literal_index(
            "vertices", literal_index, "SetVertex"
        ):
            return None

        minimum_size = (literal_index + 1) if literal_index is not None else 1
        max_vertices, _ = self.mesh_stage_current_output_limits()
        max_vertices = max(max_vertices, minimum_size)

        if self.mesh_vertex_output_variable is not None:
            if (
                self.mesh_vertex_output_limit is not None
                and minimum_size > self.mesh_vertex_output_limit
            ):
                self.emit(
                    "; WARNING: SPIR-V mesh SetVertex literal index exceeds the "
                    "declared mesh vertex output limit"
                )
                return None
            self.mark_function_interface_variable(self.mesh_vertex_output_variable)
            return self.mesh_vertex_output_variable

        float_type = self.register_primitive_type("float")
        vec4_type = self.register_vector_type(float_type, 4)
        vertex_type = self.register_struct_type(
            "_CrossGLMeshVertexOutputEXT", [(vec4_type, "position")]
        )
        self.decorations.append(
            f"OpMemberDecorate %{vertex_type.id} 0 BuiltIn Position"
        )
        array_type = self.register_array_type(vertex_type, max_vertices)
        variable = self.create_variable(array_type, "Output", "_CrossGLMeshVerticesEXT")
        self.outputs.append(variable)
        self.mesh_vertex_output_variable = variable
        self.mesh_vertex_output_limit = max_vertices
        self.mark_function_interface_variable(variable)
        return variable

    def ensure_mesh_member_output_variable(
        self,
        info: dict,
        member_name: str,
        member_type: SpirvId,
        semantic: Optional[str],
        literal_index: Optional[int] = None,
    ) -> Optional[SpirvId]:
        """Create a mesh output array for a signature member assignment."""
        role = info["role"]
        if not self.validate_mesh_output_literal_index(
            role, literal_index, f"{info['name']} output", info
        ):
            return None

        if role == "vertices":
            max_vertices, _ = self.mesh_stage_current_output_limits()
            element_count = max(info.get("count") or max_vertices, max_vertices)
        else:
            _, max_primitives = self.mesh_stage_current_output_limits()
            element_count = max(info.get("count") or max_primitives, max_primitives)

        if literal_index is not None:
            element_count = max(element_count, literal_index + 1)

        builtin = self.mesh_output_member_builtin(role, semantic)
        location = (
            None
            if builtin
            else self.mesh_output_member_location(semantic, role, member_name)
        )
        key = (
            role,
            member_name,
            member_type.id,
            builtin,
            location,
            element_count,
        )
        if key in self.mesh_output_member_variables:
            self.mark_function_interface_variable(
                self.mesh_output_member_variables[key]
            )
            return self.mesh_output_member_variables[key]

        array_type = self.register_array_type(member_type, element_count)
        variable_name = f"_CrossGLMesh_{role}_{member_name}"
        if not builtin:
            self.validate_user_defined_interface_type(
                array_type, "Output", variable_name
            )
        variable = self.create_variable(array_type, "Output", variable_name)
        if builtin:
            self.decorations.append(f"OpDecorate %{variable.id} BuiltIn {builtin}")
        else:
            self.decorations.append(f"OpDecorate %{variable.id} Location {location}")
        if role == "primitives":
            self.decorations.append(f"OpDecorate %{variable.id} PerPrimitiveEXT")
        self.outputs.append(variable)
        self.mesh_output_member_variables[key] = variable
        self.mark_function_interface_variable(variable)
        return variable

    def mesh_output_member_builtin(
        self, role: str, semantic: Optional[str]
    ) -> Optional[str]:
        """Return a SPIR-V BuiltIn decoration for mesh output member semantics."""
        if semantic is None:
            return None

        semantic_map = {
            "SV_Position": "gl_Position",
            "POSITION": "POSITION",
            "gl_Position": "gl_Position",
            "gl_PrimitiveID": "gl_PrimitiveID",
            "SV_PrimitiveID": "gl_PrimitiveID",
            "gl_Layer": "gl_Layer",
            "gl_ViewportIndex": "gl_ViewportIndex",
            "gl_CullPrimitiveEXT": "gl_CullPrimitiveEXT",
        }
        normalized = semantic_map.get(str(semantic), str(semantic))
        if role == "vertices" and normalized == "gl_Position":
            return "Position"
        if role == "primitives":
            return {
                "gl_PrimitiveID": "PrimitiveId",
                "gl_Layer": "Layer",
                "gl_ViewportIndex": "ViewportIndex",
                "gl_CullPrimitiveEXT": "CullPrimitiveEXT",
            }.get(normalized)
        return None

    def mesh_output_member_location(
        self, semantic: Optional[str], role: str, member_name: str
    ) -> int:
        """Return or allocate a Location decoration for user mesh outputs."""
        key = (role, member_name, str(semantic or ""))
        if key in self.mesh_output_member_locations:
            return self.mesh_output_member_locations[key]

        location = self.mesh_output_semantic_location(semantic)
        if location is None:
            location = self.next_output_location
            self.next_output_location += 1

        self.mesh_output_member_locations[key] = location
        return location

    def mesh_output_semantic_location(self, semantic: Optional[str]) -> Optional[int]:
        if semantic is None:
            return None

        semantic_text = str(semantic)
        fixed_locations = {
            "POSITION": 0,
            "NORMAL": 1,
            "TANGENT": 2,
            "BINORMAL": 3,
            "TEXCOORD": 4,
            "COLOR": 13,
            "COLOR0": 13,
        }
        if semantic_text in fixed_locations:
            return fixed_locations[semantic_text]

        match = re.fullmatch(r"TEXCOORD(\d+)", semantic_text)
        if match:
            return 5 + int(match.group(1))

        match = re.fullmatch(r"COLOR(\d+)", semantic_text)
        if match:
            return 13 + int(match.group(1))

        return None

    def ensure_mesh_primitive_index_output(
        self, literal_index: Optional[int] = None
    ) -> Tuple[Optional[SpirvId], Optional[SpirvId]]:
        """Create the topology-specific primitive-index output array."""
        if not self.validate_mesh_output_literal_index(
            "indices", literal_index, "SetPrimitive"
        ):
            return None, None

        info = self.mesh_primitive_index_builtin_info()
        if info is None:
            self.emit(
                "; WARNING: SPIR-V mesh SetPrimitive requires point, line, or "
                "triangle output topology"
            )
            return None, None

        builtin_name, element_type = info
        minimum_size = (literal_index + 1) if literal_index is not None else 1
        _, max_primitives = self.mesh_stage_current_output_limits()
        max_primitives = max(max_primitives, minimum_size)

        cached = self.mesh_primitive_index_outputs.get(builtin_name)
        if cached is not None:
            variable, cached_type, cached_limit = cached
            if minimum_size > cached_limit:
                self.emit(
                    "; WARNING: SPIR-V mesh SetPrimitive literal index exceeds the "
                    "declared mesh primitive output limit"
                )
                return None, None
            self.mark_function_interface_variable(variable)
            return variable, cached_type

        array_type = self.register_array_type(element_type, max_primitives)
        variable = self.create_variable(
            array_type, "Output", f"_CrossGLMesh{builtin_name}"
        )
        self.decorations.append(f"OpDecorate %{variable.id} BuiltIn {builtin_name}")
        self.outputs.append(variable)
        self.mesh_primitive_index_outputs[builtin_name] = (
            variable,
            element_type,
            max_primitives,
        )
        self.mark_function_interface_variable(variable)
        return variable, element_type

    def mesh_primitive_index_builtin_info(
        self,
    ) -> Optional[Tuple[str, SpirvId]]:
        """Return primitive-index builtin metadata for the current topology."""
        mode = self.mesh_stage_topology_mode(self.current_stage)
        uint_type = self.register_primitive_type("uint")
        if mode == "OutputPoints":
            return "PrimitivePointIndicesEXT", uint_type
        if mode == "OutputLinesEXT":
            return (
                "PrimitiveLineIndicesEXT",
                self.register_vector_type(uint_type, 2),
            )
        if mode == "OutputTrianglesEXT":
            return (
                "PrimitiveTriangleIndicesEXT",
                self.register_vector_type(uint_type, 3),
            )
        return None

    def mesh_vertex_position_value(self, expr) -> Optional[SpirvId]:
        """Return a vec4 position value for SetVertex."""
        value = self.process_expression(expr)
        if value is None:
            self.emit("; WARNING: SPIR-V mesh SetVertex requires a value operand")
            return None

        position = self.mesh_vertex_position_from_value(value)
        if position is None:
            self.emit(
                "; WARNING: SPIR-V mesh SetVertex value must be a vec3, vec4, "
                "or struct with a position member"
            )
        return position

    def mesh_vertex_position_from_value(self, value: SpirvId) -> Optional[SpirvId]:
        """Convert a SetVertex value to the Position builtin's vec4 type."""
        float_type = self.register_primitive_type("float")
        vec4_type = self.register_vector_type(float_type, 4)
        vector_info = self.vector_component_type_and_count(value.type.base_type)

        if vector_info is not None:
            component_type_name, component_count = vector_info
            if component_count == 4:
                converted = self.convert_value_to_type(value, vec4_type)
                return converted if self.value_has_type(converted, vec4_type) else None
            if component_count == 3:
                component_type = self.register_primitive_type(component_type_name)
                components = [
                    self.convert_scalar_to_type(
                        self.composite_extract(value, component_type, index),
                        float_type,
                    )
                    for index in range(3)
                ]
                components.append(self.register_constant(1.0, float_type))
                return self.composite_construct(vec4_type, components)
            return None

        member_info = self.struct_member_info(value.type.base_type, "position")
        if member_info is None:
            member_info = self.struct_member_info(value.type.base_type, "gl_Position")
        if member_info is None:
            return None

        member_index, member_type = member_info
        member_value = self.composite_extract(value, member_type, member_index)
        return self.mesh_vertex_position_from_value(member_value)

    def process_mesh_output_assignment(self, target, value_expr) -> bool:
        """Lower assignments to mesh output signature parameters."""
        access = self.mesh_output_assignment_access(target)
        if access is None:
            return False

        info, index_expr, member_name, member_component = access
        index = self.process_expression(index_expr)
        if index is None:
            self.emit("; WARNING: SPIR-V mesh output assignment requires an index")
            return True

        literal_index = self.literal_int_argument(index_expr)
        if not self.validate_mesh_output_literal_index(
            info["role"], literal_index, f"{info['name']} output", info
        ):
            return True

        role = info["role"]
        if role == "indices":
            self.store_mesh_index_output(info, index, literal_index, value_expr)
            return True

        if member_component is not None:
            member_info = info["members"].get(member_name)
            if member_info is None:
                self.emit(
                    f"; WARNING: SPIR-V mesh output {info['name']} has no "
                    f"member {member_name}"
                )
                return True
            if not self.validate_mesh_output_member_component_target(
                info, member_name, member_component
            ):
                return True
            member_type = member_info["type"]
            value = self.process_expression_with_expected_type(
                value_expr, member_type.type.base_type
            )
            if value is not None:
                self.store_mesh_output_member_component(
                    info,
                    member_name,
                    member_component,
                    index,
                    literal_index,
                    value,
                )
            return True

        if member_name is not None:
            member_info = info["members"].get(member_name)
            if member_info is None:
                self.emit(
                    f"; WARNING: SPIR-V mesh output {info['name']} has no "
                    f"member {member_name}"
                )
                return True
            member_type = member_info["type"]
            value = self.process_expression_with_expected_type(
                value_expr, member_type.type.base_type
            )
            if value is not None:
                self.store_mesh_output_member(
                    info, member_name, index, literal_index, value
                )
            return True

        value = self.process_expression(value_expr)
        if value is None:
            return True

        element_type = info["element_type"]
        value = self.convert_value_to_type(value, element_type)
        if not self.value_has_type(value, element_type):
            self.emit(
                f"; WARNING: SPIR-V mesh {info['name']} output assignment value "
                f"does not match {element_type.type.base_type}"
            )
            return True

        for member_name, member_info in info["members"].items():
            member_value = self.composite_extract(
                value, member_info["type"], member_info["index"]
            )
            self.store_mesh_output_member(
                info, member_name, index, literal_index, member_value
            )
        return True

    def validate_mesh_output_member_component_target(
        self, info: dict, member_name: str, component_name: str
    ) -> bool:
        member_info = info["members"][member_name]
        member_type = member_info["type"]

        swizzle_info = self.vector_swizzle_info(
            member_type.type.base_type, component_name
        )
        if swizzle_info is not None:
            component_indices, _, _ = swizzle_info
            if len(set(component_indices)) != len(component_indices):
                self.emit(
                    f"; WARNING: Cannot assign to vector swizzle {component_name} "
                    "with duplicate components"
                )
                return False
            return True

        if self.vector_member_info(member_type.type.base_type, component_name):
            return True

        if (
            self.vector_component_type_and_count(member_type.type.base_type) is not None
            and len(component_name) > 1
        ):
            self.emit(
                f"; WARNING: Invalid vector swizzle {component_name} "
                f"for {member_type.type.base_type}"
            )
        else:
            self.emit(
                f"; WARNING: SPIR-V mesh output {member_name} has no "
                f"vector component {component_name}"
            )
        return False

    def process_mesh_output_compound_assignment(
        self,
        target,
        value_expr,
        spv_operator: str,
        target_is_precise: bool,
    ) -> bool:
        """Lower compound assignments to mesh output signature parameters."""
        access = self.mesh_output_assignment_access(target)
        if access is None:
            return False

        info, index_expr, member_name, member_component = access
        index = self.process_expression(index_expr)
        if index is None:
            self.emit("; WARNING: SPIR-V mesh output assignment requires an index")
            return True

        literal_index = self.literal_int_argument(index_expr)
        if not self.validate_mesh_output_literal_index(
            info["role"], literal_index, f"{info['name']} output", info
        ):
            return True

        if info["role"] == "indices":
            self.emit(
                "; WARNING: Compound assignment to SPIR-V mesh indices output "
                "cannot be lowered"
            )
            return True
        if member_name is None:
            self.emit(
                "; WARNING: Compound assignment to whole SPIR-V mesh output "
                "elements cannot be lowered"
            )
            return True

        member_info = info["members"].get(member_name)
        if member_info is None:
            self.emit(
                f"; WARNING: SPIR-V mesh output {info['name']} has no "
                f"member {member_name}"
            )
            return True
        if member_component is not None:
            return self.process_mesh_output_member_component_compound_assignment(
                info,
                member_name,
                member_component,
                index,
                literal_index,
                value_expr,
                spv_operator,
                target_is_precise,
            )

        return self.process_mesh_output_member_compound_assignment(
            info,
            member_name,
            index,
            literal_index,
            value_expr,
            spv_operator,
            target_is_precise,
        )

    def process_mesh_output_member_compound_assignment(
        self,
        info: dict,
        member_name: str,
        index: SpirvId,
        literal_index: Optional[int],
        value_expr,
        spv_operator: str,
        target_is_precise: bool,
    ) -> bool:
        member_pointer, member_type = self.mesh_output_member_shadow_pointer(
            info, member_name, index, literal_index
        )
        if member_pointer is None or member_type is None:
            return True

        if target_is_precise:
            self.precise_expression_depth += 1

        try:
            current_value = self.load_from_variable(member_pointer, member_type)
            rhs_value = self.process_expression(value_expr)
            if rhs_value is None:
                return True
            result = self.binary_operation(
                spv_operator, member_type, current_value, rhs_value
            )
        finally:
            if target_is_precise:
                self.precise_expression_depth -= 1

        self.store_mesh_output_member(info, member_name, index, literal_index, result)
        return True

    def process_mesh_output_member_component_compound_assignment(
        self,
        info: dict,
        member_name: str,
        component_name: str,
        index: SpirvId,
        literal_index: Optional[int],
        value_expr,
        spv_operator: str,
        target_is_precise: bool,
    ) -> bool:
        member_info = info["members"][member_name]
        member_type = member_info["type"]
        swizzle_info = self.vector_swizzle_info(
            member_type.type.base_type, component_name
        )
        if swizzle_info is not None:
            component_indices, component_type, swizzle_type = swizzle_info
            if len(set(component_indices)) != len(component_indices):
                self.emit(
                    f"; WARNING: Cannot assign to vector swizzle {component_name} "
                    "with duplicate components"
                )
                return True

            member_pointer, storage_member_type = (
                self.mesh_output_member_shadow_pointer(
                    info, member_name, index, literal_index
                )
            )
            if member_pointer is None or storage_member_type is None:
                return True

            if target_is_precise:
                self.precise_expression_depth += 1

            try:
                rhs_value = self.process_expression(value_expr)
                if rhs_value is None:
                    return True
                rhs_components = self.convert_vector_swizzle_assignment_components(
                    rhs_value, component_type, len(component_indices), component_name
                )
                if rhs_components is None:
                    return True

                rhs_vector = self.composite_construct(swizzle_type, rhs_components)
                member_value = self.load_from_variable(
                    member_pointer, storage_member_type
                )
                current_value = self.vector_shuffle(
                    member_value, swizzle_type, component_indices
                )
                result = self.binary_operation(
                    spv_operator, swizzle_type, current_value, rhs_vector
                )
            finally:
                if target_is_precise:
                    self.precise_expression_depth -= 1

            for result_index, component_index in enumerate(component_indices):
                component_value = self.composite_extract(
                    result, component_type, result_index
                )
                self.store_mesh_output_member_component_value(
                    info,
                    member_name,
                    index,
                    literal_index,
                    component_index,
                    component_type,
                    component_value,
                )
            return True

        component_info = self.vector_member_info(
            member_type.type.base_type, component_name
        )
        if component_info is None:
            if (
                self.vector_component_type_and_count(member_type.type.base_type)
                is not None
                and len(component_name) > 1
            ):
                self.emit(
                    f"; WARNING: Invalid vector swizzle {component_name} "
                    f"for {member_type.type.base_type}"
                )
            else:
                self.emit(
                    f"; WARNING: SPIR-V mesh output {member_name} has no "
                    f"vector component {component_name}"
                )
            return True

        component_index, component_type = component_info
        component_pointer = self.mesh_output_member_shadow_component_pointer(
            info,
            member_name,
            index,
            literal_index,
            component_index,
            component_type,
        )
        if component_pointer is None:
            return True

        if target_is_precise:
            self.precise_expression_depth += 1

        try:
            current_value = self.load_from_variable(component_pointer, component_type)
            rhs_value = self.process_expression(value_expr)
            if rhs_value is None:
                return True
            rhs_value = self.convert_vector_component_value(
                rhs_value, component_type, component_name
            )
            if rhs_value is None:
                return True

            result = self.binary_operation(
                spv_operator, component_type, current_value, rhs_value
            )
        finally:
            if target_is_precise:
                self.precise_expression_depth -= 1

        self.store_mesh_output_member_component_value(
            info,
            member_name,
            index,
            literal_index,
            component_index,
            component_type,
            result,
        )
        return True

    def mesh_output_assignment_access(self, target):
        """Return mesh output parameter assignment metadata for a target."""
        if isinstance(target, MemberAccessNode):
            member_access = self.mesh_output_member_access_info(target)
            if member_access is not None:
                info, index_expr, member_name = member_access
                return info, index_expr, member_name, None

            parent_access = self.mesh_output_member_access_info(
                getattr(target, "object", None)
            )
            if parent_access is not None:
                info, index_expr, member_name = parent_access
                return info, index_expr, member_name, str(target.member)

        if isinstance(target, ArrayAccessNode):
            access = self.mesh_output_array_access_info(target)
            if access is None:
                return None
            info, index_expr = access
            if info["role"] not in {"vertices", "indices", "primitives"}:
                return None
            return info, index_expr, None, None

        return None

    def mesh_output_member_shadow_variable(
        self,
        info: dict,
        member_name: str,
        literal_index: Optional[int],
    ) -> Tuple[Optional[SpirvId], Optional[SpirvId]]:
        member_info = info["members"][member_name]
        semantic = member_info.get("semantic")
        member_type = member_info["type"]

        if not self.validate_mesh_output_literal_index(
            info["role"], literal_index, f"{info['name']} output", info
        ):
            return None, None

        if info["role"] == "vertices":
            max_vertices, _ = self.mesh_stage_current_output_limits()
            element_count = max(info.get("count") or max_vertices, max_vertices)
        else:
            _, max_primitives = self.mesh_stage_current_output_limits()
            element_count = max(info.get("count") or max_primitives, max_primitives)

        if literal_index is not None:
            element_count = max(element_count, literal_index + 1)

        if (
            info["role"] == "vertices"
            and self.mesh_output_member_builtin("vertices", semantic) == "Position"
        ):
            storage_member_type = self.register_vector_type(
                self.register_primitive_type("float"), 4
            )
        else:
            storage_member_type = member_type

        key = (
            info["role"],
            member_name,
            storage_member_type.id,
            element_count,
        )
        if key in self.mesh_output_member_shadow_variables:
            return self.mesh_output_member_shadow_variables[key], storage_member_type

        array_type = self.register_array_type(storage_member_type, element_count)
        variable = self.create_variable(
            array_type, "Private", f"_CrossGLMeshShadow_{info['role']}_{member_name}"
        )
        self.entry_point_private_variables.append(variable)
        self.mesh_output_member_shadow_variables[key] = variable
        return variable, storage_member_type

    def mesh_output_member_shadow_pointer(
        self,
        info: dict,
        member_name: str,
        index: SpirvId,
        literal_index: Optional[int],
    ) -> Tuple[Optional[SpirvId], Optional[SpirvId]]:
        variable, member_type = self.mesh_output_member_shadow_variable(
            info, member_name, literal_index
        )
        if variable is None or member_type is None:
            return None, None

        ptr_type = self.register_pointer_type(member_type, "Private")
        access = self.access_chain(variable, [index], ptr_type)
        self.variable_value_types[access.id] = member_type
        return access, member_type

    def mesh_output_member_shadow_component_pointer(
        self,
        info: dict,
        member_name: str,
        index: SpirvId,
        literal_index: Optional[int],
        component_index: int,
        component_type: SpirvId,
    ) -> Optional[SpirvId]:
        variable, member_type = self.mesh_output_member_shadow_variable(
            info, member_name, literal_index
        )
        if variable is None or member_type is None:
            return None

        int_type = self.register_primitive_type("int")
        component_index_id = self.register_constant(component_index, int_type)
        ptr_type = self.register_pointer_type(component_type, "Private")
        access = self.access_chain(variable, [index, component_index_id], ptr_type)

        self.variable_value_types[access.id] = component_type
        return access

    def mesh_output_member_access_info(self, target):
        if not isinstance(target, MemberAccessNode):
            return None
        array_access = getattr(target, "object", None)
        if not isinstance(array_access, ArrayAccessNode):
            return None
        access = self.mesh_output_array_access_info(array_access)
        if access is None:
            return None
        info, index_expr = access
        if info["role"] not in {"vertices", "primitives"}:
            return None
        return info, index_expr, str(target.member)

    def mesh_output_array_access_info(self, node):
        array_expr = getattr(node, "array", getattr(node, "array_expr", None))
        name = self.expression_name(array_expr)
        info = self.current_mesh_output_parameters.get(name)
        if info is None:
            return None
        index_expr = getattr(node, "index", getattr(node, "index_expr", None))
        return info, index_expr

    def store_mesh_index_output(
        self, info: dict, index: SpirvId, literal_index: Optional[int], value_expr
    ):
        builtin_info = self.mesh_primitive_index_builtin_info()
        if builtin_info is None:
            self.emit(
                "; WARNING: SPIR-V mesh SetPrimitive requires point, line, or "
                "triangle output topology"
            )
            return
        _, element_type = builtin_info

        value = self.process_expression(value_expr)
        if value is None:
            return
        value = self.convert_value_to_type(value, element_type)
        if not self.value_has_type(value, element_type):
            self.emit(
                "; WARNING: SPIR-V mesh indices output assignment value does not "
                f"match {element_type.type.base_type}"
            )
            return

        primitive_output, _ = self.ensure_mesh_primitive_index_output(literal_index)
        if primitive_output is None:
            return

        ptr_type = self.register_pointer_type(element_type, "Output")
        access = self.access_chain(primitive_output, [index], ptr_type)
        self.variable_value_types[access.id] = element_type
        self.store_to_variable(access, value)

    def store_mesh_output_member(
        self,
        info: dict,
        member_name: str,
        index: SpirvId,
        literal_index: Optional[int],
        value: SpirvId,
    ):
        member_info = info["members"][member_name]
        semantic = member_info.get("semantic")
        member_type = member_info["type"]

        if (
            info["role"] == "vertices"
            and self.mesh_output_member_builtin("vertices", semantic) == "Position"
        ):
            value = self.mesh_vertex_position_from_value(value)
            if value is None:
                self.emit(
                    "; WARNING: SPIR-V mesh vertex output position assignment "
                    "requires a vec3, vec4, or struct with a position member"
                )
                return
            vertex_output = self.ensure_mesh_vertex_position_output(literal_index)
            if vertex_output is None:
                return
            int_type = self.register_primitive_type("int")
            member_index = self.register_constant(0, int_type)
            vec4_type = self.register_vector_type(
                self.register_primitive_type("float"), 4
            )
            ptr_type = self.register_pointer_type(vec4_type, "Output")
            access = self.access_chain(vertex_output, [index, member_index], ptr_type)
            self.variable_value_types[access.id] = vec4_type
            self.store_to_variable(access, value)
            shadow_access, _ = self.mesh_output_member_shadow_pointer(
                info, member_name, index, literal_index
            )
            if shadow_access is not None:
                self.store_to_variable(shadow_access, value)
            return

        variable = self.ensure_mesh_member_output_variable(
            info, member_name, member_type, semantic, literal_index
        )
        if variable is None:
            return

        value = self.convert_value_to_type(value, member_type)
        if not self.value_has_type(value, member_type):
            self.emit(
                f"; WARNING: SPIR-V mesh output {member_name} assignment value "
                f"does not match {member_type.type.base_type}"
            )
            return

        ptr_type = self.register_pointer_type(member_type, "Output")
        access = self.access_chain(variable, [index], ptr_type)
        self.variable_value_types[access.id] = member_type
        self.store_to_variable(access, value)
        shadow_access, _ = self.mesh_output_member_shadow_pointer(
            info, member_name, index, literal_index
        )
        if shadow_access is not None:
            self.store_to_variable(shadow_access, value)

    def store_mesh_output_member_component(
        self,
        info: dict,
        member_name: str,
        component_name: str,
        index: SpirvId,
        literal_index: Optional[int],
        value: SpirvId,
    ):
        member_info = info["members"][member_name]
        member_type = member_info["type"]

        component_indices = None
        component_type = None
        component_values = None

        swizzle_info = self.vector_swizzle_info(
            member_type.type.base_type, component_name
        )
        if swizzle_info is not None:
            component_indices, component_type, _ = swizzle_info
            if len(set(component_indices)) != len(component_indices):
                self.emit(
                    f"; WARNING: Cannot assign to vector swizzle {component_name} "
                    "with duplicate components"
                )
                return
            component_values = self.convert_vector_swizzle_assignment_components(
                value, component_type, len(component_indices), component_name
            )
        else:
            member_component_info = self.vector_member_info(
                member_type.type.base_type, component_name
            )
            if member_component_info is not None:
                component_index, component_type = member_component_info
                component_indices = [component_index]
                component_value = self.convert_vector_component_value(
                    value, component_type, component_name
                )
                component_values = (
                    [component_value] if component_value is not None else None
                )

        if component_indices is None or component_type is None:
            if (
                self.vector_component_type_and_count(member_type.type.base_type)
                is not None
                and len(component_name) > 1
            ):
                self.emit(
                    f"; WARNING: Invalid vector swizzle {component_name} "
                    f"for {member_type.type.base_type}"
                )
            else:
                self.emit(
                    f"; WARNING: SPIR-V mesh output {member_name} has no "
                    f"vector component {component_name}"
                )
            return
        if component_values is None:
            return

        for component_index, component_value in zip(
            component_indices, component_values
        ):
            self.store_mesh_output_member_component_value(
                info,
                member_name,
                index,
                literal_index,
                component_index,
                component_type,
                component_value,
            )

    def store_mesh_output_member_component_value(
        self,
        info: dict,
        member_name: str,
        index: SpirvId,
        literal_index: Optional[int],
        component_index: int,
        component_type: SpirvId,
        component_value: SpirvId,
    ):
        member_info = info["members"][member_name]
        semantic = member_info.get("semantic")
        member_type = member_info["type"]
        int_type = self.register_primitive_type("int")
        component_index_id = self.register_constant(component_index, int_type)
        ptr_type = self.register_pointer_type(component_type, "Output")

        if (
            info["role"] == "vertices"
            and self.mesh_output_member_builtin("vertices", semantic) == "Position"
        ):
            vertex_output = self.ensure_mesh_vertex_position_output(literal_index)
            if vertex_output is None:
                return
            member_index = self.register_constant(0, int_type)
            access = self.access_chain(
                vertex_output, [index, member_index, component_index_id], ptr_type
            )
        else:
            variable = self.ensure_mesh_member_output_variable(
                info, member_name, member_type, semantic, literal_index
            )
            if variable is None:
                return
            access = self.access_chain(variable, [index, component_index_id], ptr_type)

        self.variable_value_types[access.id] = component_type
        self.store_to_variable(access, component_value)
        shadow_access = self.mesh_output_member_shadow_component_pointer(
            info,
            member_name,
            index,
            literal_index,
            component_index,
            component_type,
        )
        if shadow_access is not None:
            self.store_to_variable(shadow_access, component_value)

    def process_dispatch_mesh_operation(self, expr: MeshOpNode) -> Optional[SpirvId]:
        """Lower task-shader DispatchMesh to the SPIR-V mesh-task terminator."""
        operation = expr.operation
        if self.current_execution_model != "TaskEXT":
            return self.represented_ir_diagnostic_default_value(
                "mesh shader", operation
            )

        arguments = getattr(expr, "arguments", []) or []
        if len(arguments) not in {3, 4}:
            self.emit(
                "; WARNING: SPIR-V mesh DispatchMesh requires exactly 3 arguments, "
                "or 4 with a task payload"
            )
            return self.register_constant(0, self.register_primitive_type("uint"))

        uint_type = self.register_primitive_type("uint")
        group_counts = []
        for argument in arguments[:3]:
            group_count = self.process_expression(argument)
            if group_count is None:
                self.emit(
                    "; WARNING: SPIR-V mesh DispatchMesh requires group-count operands"
                )
                return self.register_constant(0, uint_type)
            if not self.validate_mesh_count_operands(
                "DispatchMesh",
                [group_count],
                "group-count",
                argument_exprs=[argument],
            ):
                return self.register_constant(0, uint_type)
            group_counts.append(self.convert_value_to_type(group_count, uint_type))

        payload_pointer = None
        if len(arguments) == 4:
            payload_pointer = self.dispatch_mesh_payload_pointer(arguments[3])
            if payload_pointer is None:
                self.emit(
                    "; WARNING: SPIR-V mesh DispatchMesh payload argument requires "
                    "taskPayloadSharedEXT storage"
                )
                return self.register_constant(0, uint_type)

        self.require_capability("MeshShadingEXT")
        self.require_extension("SPV_EXT_mesh_shader")
        operands = [f"%{group_count.id}" for group_count in group_counts]
        if payload_pointer is not None:
            operands.append(f"%{payload_pointer.id}")
        self.emit("OpEmitMeshTasksEXT " + " ".join(operands))
        return None

    def dispatch_mesh_payload_pointer(self, payload_expr) -> Optional[SpirvId]:
        """Return a TaskPayloadWorkgroupEXT pointer for DispatchMesh payloads."""
        target = self.dispatch_mesh_payload_target(payload_expr)
        if target is None:
            return None

        if self.direct_expression_name(payload_expr) == target.name:
            self.record_task_payload_interface(target)
            return target

        payload_value = self.process_expression(payload_expr)
        if payload_value is None:
            return None

        target_type = self.pointer_pointee_type(target)
        if target_type is not None:
            payload_value = self.convert_value_to_type(payload_value, target_type)
            if not self.value_has_type(payload_value, target_type):
                self.emit(
                    "; WARNING: SPIR-V mesh DispatchMesh payload argument type "
                    f"{payload_value.type.base_type} does not match "
                    f"{target_type.type.base_type}"
                )
                return None

        self.store_to_variable(target, payload_value)
        self.record_task_payload_interface(target)
        return target

    def record_task_payload_interface(self, payload_pointer: SpirvId):
        """Associate a task payload variable with the current task entry point."""
        if self.current_function_id is None:
            return
        self.task_payload_interface_by_function[self.current_function_id] = (
            payload_pointer
        )

    def dispatch_mesh_payload_target(self, payload_expr) -> Optional[SpirvId]:
        """Find the shared task payload variable used by DispatchMesh."""
        if not self.task_payload_shared_variables:
            return None

        direct_name = self.direct_expression_name(payload_expr)
        if direct_name in self.task_payload_shared_variables:
            return self.task_payload_shared_variables[direct_name]

        payload_type = self.dispatch_mesh_payload_expression_type(payload_expr)
        if payload_type is not None:
            matches = [
                variable
                for variable in self.task_payload_shared_variables.values()
                if (
                    self.pointer_pointee_type(variable) is not None
                    and self.pointer_pointee_type(variable).type.base_type
                    == payload_type.type.base_type
                )
            ]
            if len(matches) == 1:
                return matches[0]
            if len(matches) > 1:
                names = ", ".join(sorted(variable.name or "" for variable in matches))
                self.emit(
                    "; WARNING: SPIR-V mesh DispatchMesh payload target is "
                    f"ambiguous for {payload_type.type.base_type}: {names}"
                )
                return None

        if len(self.task_payload_shared_variables) == 1:
            return next(iter(self.task_payload_shared_variables.values()))

        names = ", ".join(sorted(self.task_payload_shared_variables))
        self.emit(
            "; WARNING: SPIR-V mesh DispatchMesh payload target is ambiguous: "
            f"{names}"
        )
        return None

    def dispatch_mesh_payload_expression_type(self, expr) -> Optional[SpirvId]:
        """Return a statically known payload value type without emitting code."""
        direct_name = self.direct_expression_name(expr)
        if direct_name:
            variable = self.local_variables.get(
                direct_name
            ) or self.resolve_global_variable(direct_name)
            if variable is not None:
                value_type = self.variable_value_types.get(variable.id)
                if value_type is not None:
                    return value_type
                registered_type = self.value_types.get(variable.id)
                if registered_type is not None:
                    return registered_type

        if isinstance(expr, FunctionCallNode):
            callee_name = self.function_call_name(expr)
            function_signature = self.resolve_function_signature(callee_name)
            if function_signature is not None:
                return_type = function_signature[0]
                if return_type.type.base_type != "void":
                    return return_type
            if callee_name in self.struct_types:
                return self.struct_types[callee_name]

        return None

    def flatten_vector_constructor_args(
        self,
        function_name: str,
        args: List[SpirvId],
        component_type: SpirvId,
        component_count: int,
    ) -> List[SpirvId]:
        if (
            len(args) == 1
            and self.vector_component_type_and_count(args[0].type.base_type) is None
        ):
            component_arg = self.convert_vector_constructor_scalar(
                args[0], component_type, function_name
            )
            return [component_arg] * component_count

        flattened_args = []
        for arg in args:
            vector_info = self.vector_component_type_and_count(arg.type.base_type)
            if vector_info is None:
                flattened_args.append(
                    self.convert_vector_constructor_scalar(
                        arg, component_type, function_name
                    )
                )
                continue

            source_component_type_name, source_component_count = vector_info
            source_component_type = self.register_primitive_type(
                source_component_type_name
            )
            for index in range(source_component_count):
                source_value = self.composite_extract(arg, source_component_type, index)
                flattened_args.append(
                    self.convert_vector_constructor_scalar(
                        source_value, component_type, function_name
                    )
                )

        if len(flattened_args) < component_count:
            self.emit(
                f"; WARNING: Constructor {function_name} expected {component_count} "
                f"components but got {len(flattened_args)}; padding with defaults"
            )
            default_value = self.default_value_for_type(component_type)
            flattened_args.extend(
                [default_value] * (component_count - len(flattened_args))
            )
        elif len(flattened_args) > component_count:
            self.emit(
                f"; WARNING: Constructor {function_name} expected {component_count} "
                f"components but got {len(flattened_args)}; truncating extra components"
            )
            flattened_args = flattened_args[:component_count]

        return flattened_args

    def convert_vector_constructor_scalar(
        self, value: SpirvId, component_type: SpirvId, function_name: str
    ) -> SpirvId:
        target_type = self.normalize_primitive_name(component_type.type.base_type)
        source_type = self.normalize_primitive_name(value.type.base_type)
        if target_type == "bool" and source_type != "bool":
            self.emit(
                f"; WARNING: Constructor {function_name} cannot convert "
                f"{source_type} component to {target_type}; using default value"
            )
            return self.default_value_for_type(component_type)

        converted_value = self.convert_scalar_to_type(value, component_type)
        source_type = self.normalize_primitive_name(converted_value.type.base_type)
        if source_type != target_type:
            self.emit(
                f"; WARNING: Constructor {function_name} cannot convert "
                f"{source_type} component to {target_type}; using default value"
            )
            return self.default_value_for_type(component_type)

        return converted_value

    def flatten_matrix_constructor_components(
        self,
        function_name: str,
        args: List[SpirvId],
        component_type: SpirvId,
        component_count: int,
    ) -> List[SpirvId]:
        flattened_args = []
        for arg in args:
            flattened_args.extend(
                self.flatten_matrix_constructor_arg(function_name, arg, component_type)
            )

        if len(flattened_args) < component_count:
            self.emit(
                f"; WARNING: Constructor {function_name} expected {component_count} "
                f"components but got {len(flattened_args)}; padding with defaults"
            )
            default_value = self.default_value_for_type(component_type)
            flattened_args.extend(
                [default_value] * (component_count - len(flattened_args))
            )
        elif len(flattened_args) > component_count:
            self.emit(
                f"; WARNING: Constructor {function_name} expected {component_count} "
                f"components but got {len(flattened_args)}; truncating extra components"
            )
            flattened_args = flattened_args[:component_count]

        return flattened_args

    def flatten_matrix_constructor_arg(
        self, function_name: str, arg: SpirvId, component_type: SpirvId
    ) -> List[SpirvId]:
        vector_info = self.vector_component_type_and_count(arg.type.base_type)
        if vector_info is not None:
            source_component_type_name, source_component_count = vector_info
            source_component_type = self.register_primitive_type(
                source_component_type_name
            )
            return [
                self.convert_vector_constructor_scalar(
                    self.composite_extract(arg, source_component_type, index),
                    component_type,
                    function_name,
                )
                for index in range(source_component_count)
            ]

        registered_type = self.find_registered_type_by_base(arg.type.base_type)
        matrix_info = self.matrix_type_info_from_type(registered_type)
        if matrix_info is not None:
            column_type, column_count = matrix_info
            column_info = self.vector_type_info_from_type(column_type)
            if column_info is None:
                return []

            source_component_type, row_count = column_info
            flattened_components = []
            for column_index in range(column_count):
                column_value = self.composite_extract(arg, column_type, column_index)
                for row_index in range(row_count):
                    flattened_components.append(
                        self.convert_vector_constructor_scalar(
                            self.composite_extract(
                                column_value, source_component_type, row_index
                            ),
                            component_type,
                            function_name,
                        )
                    )
            return flattened_components

        return [
            self.convert_vector_constructor_scalar(arg, component_type, function_name)
        ]

    def spirv_scope_constant(self, scope: str) -> SpirvId:
        scope_values = {
            "CrossDevice": 0,
            "Device": 1,
            "Workgroup": 2,
            "Subgroup": 3,
            "Invocation": 4,
        }
        return self.register_constant(
            scope_values[scope], self.register_primitive_type("uint")
        )

    def spirv_memory_semantics_constant(self, *semantics: str) -> SpirvId:
        semantic_values = {
            "AcquireRelease": 0x8,
            "UniformMemory": 0x40,
            "WorkgroupMemory": 0x100,
            "ImageMemory": 0x800,
        }
        value = 0
        for semantic in semantics:
            value |= semantic_values[semantic]
        return self.register_constant(value, self.register_primitive_type("uint"))

    def current_synchronization_execution_models(self) -> set:
        if self.current_stage is None and self.current_function_name is not None:
            execution_models = self.function_execution_models.get(
                self.current_function_name, ()
            )
            if execution_models:
                return set(execution_models)
        if self.current_execution_model is not None:
            return {self.current_execution_model}
        if self.current_function_name is None:
            return set()
        return set(self.function_execution_models.get(self.current_function_name, ()))

    def current_execution_model_label(self) -> str:
        execution_models = self.current_synchronization_execution_models()
        if execution_models:
            return ", ".join(sorted(execution_models))
        return "unknown"

    def can_emit_workgroup_synchronization(self) -> bool:
        execution_models = self.current_synchronization_execution_models()
        workgroup_execution_models = {
            "GLCompute",
            "MeshEXT",
            "TaskEXT",
        }
        return bool(execution_models) and execution_models.issubset(
            workgroup_execution_models
        )

    def call_synchronization_function(
        self, function_name: str, args: List[SpirvId]
    ) -> Optional[SpirvId]:
        synchronization_functions = {
            "allMemoryBarrier",
            "barrier",
            "groupMemoryBarrier",
            "workgroupBarrier",
            "memoryBarrier",
            "memoryBarrierBuffer",
            "memoryBarrierImage",
            "memoryBarrierShared",
        }
        if function_name not in synchronization_functions:
            return None

        if args:
            self.emit(
                f"; WARNING: synchronization builtin '{function_name}' "
                "requires 0 operands"
            )
            return self.register_constant(0, self.register_primitive_type("int"))

        workgroup_synchronization_functions = {
            "barrier",
            "groupMemoryBarrier",
            "memoryBarrierShared",
            "workgroupBarrier",
        }
        if (
            function_name in workgroup_synchronization_functions
            and not self.can_emit_workgroup_synchronization()
        ):
            self.emit(
                f"; WARNING: synchronization builtin '{function_name}' requires "
                "a workgroup-capable execution model; current execution model: "
                f"{self.current_execution_model_label()}"
            )
            return self.register_constant(0, self.register_primitive_type("int"))

        if function_name in {"barrier", "workgroupBarrier"}:
            workgroup_scope = self.spirv_scope_constant("Workgroup")
            workgroup_semantics = self.spirv_memory_semantics_constant(
                "AcquireRelease", "WorkgroupMemory"
            )
            self.emit(
                f"OpControlBarrier %{workgroup_scope.id} %{workgroup_scope.id} "
                f"%{workgroup_semantics.id}"
            )
            return self.register_constant(0, self.register_primitive_type("int"))

        if function_name in {"groupMemoryBarrier", "memoryBarrierShared"}:
            scope = self.spirv_scope_constant("Workgroup")
            semantics = self.spirv_memory_semantics_constant(
                "AcquireRelease", "WorkgroupMemory"
            )
        elif function_name == "memoryBarrierBuffer":
            scope = self.spirv_scope_constant("Device")
            semantics = self.spirv_memory_semantics_constant(
                "AcquireRelease", "UniformMemory"
            )
        elif function_name == "memoryBarrierImage":
            scope = self.spirv_scope_constant("Device")
            semantics = self.spirv_memory_semantics_constant(
                "AcquireRelease", "ImageMemory"
            )
        else:
            scope = self.spirv_scope_constant("Device")
            semantics = self.spirv_memory_semantics_constant(
                "AcquireRelease", "UniformMemory", "WorkgroupMemory", "ImageMemory"
            )

        self.emit(f"OpMemoryBarrier %{scope.id} %{semantics.id}")
        return self.register_constant(0, self.register_primitive_type("int"))

    def require_group_non_uniform(self, capability: Optional[str] = None):
        self.require_capability("GroupNonUniform")
        if capability is not None:
            self.require_capability(capability)

    def require_group_non_uniform_partitioned_nv(self):
        self.require_capability("GroupNonUniformPartitionedNV")
        self.require_extension("SPV_NV_shader_subgroup_partitioned")

    def subgroup_scope_id(self) -> SpirvId:
        self.require_group_non_uniform()
        return self.spirv_scope_constant("Subgroup")

    def wave_result_default(self, operation: str, args: List[SpirvId]) -> SpirvId:
        if operation in {
            "WaveActiveAllTrue",
            "WaveActiveAnyTrue",
            "WaveIsFirstLane",
        }:
            return self.default_value_for_type(self.register_primitive_type("bool"))
        if operation in {"WaveActiveBallot", "WaveMatch"}:
            uint_type = self.register_primitive_type("uint")
            return self.default_value_for_type(self.register_vector_type(uint_type, 4))
        if operation in {"WaveGetLaneCount", "WaveGetLaneIndex"}:
            return self.default_value_for_type(self.register_primitive_type("uint"))
        if args:
            return self.default_value_for_type(
                self.ensure_registered_type(args[0].type)
            )
        return self.default_value_for_type(self.register_primitive_type("uint"))

    def call_wave_operation(
        self, operation: str, argument_exprs: List
    ) -> Optional[SpirvId]:
        args = []
        for argument in argument_exprs:
            value = self.process_expression(argument)
            if value is None:
                self.emit(f"; WARNING: Failed to evaluate argument for {operation}")
                value = self.default_value_for_type(
                    self.register_primitive_type("uint")
                )
            args.append(value)

        if operation == "WaveGetLaneCount":
            if args:
                self.emit("; WARNING: WaveGetLaneCount takes no arguments")
                return self.wave_result_default(operation, args)
            builtin = self.ensure_compute_builtin("gl_SubgroupSize")
            return self.get_variable_value(builtin) if builtin is not None else None

        if operation == "WaveGetLaneIndex":
            if args:
                self.emit("; WARNING: WaveGetLaneIndex takes no arguments")
                return self.wave_result_default(operation, args)
            builtin = self.ensure_compute_builtin("gl_SubgroupInvocationID")
            return self.get_variable_value(builtin) if builtin is not None else None

        if operation == "WaveIsFirstLane":
            if args:
                self.emit("; WARNING: WaveIsFirstLane takes no arguments")
                return self.wave_result_default(operation, args)
            self.require_group_non_uniform("GroupNonUniformVote")
            bool_type = self.register_primitive_type("bool")
            scope = self.subgroup_scope_id()
            id_value = self.get_id()
            self.emit(
                f"%{id_value} = OpGroupNonUniformElect %{bool_type.id} %{scope.id}"
            )
            spirv_id = SpirvId(id_value, bool_type.type)
            self.value_types[id_value] = bool_type
            return spirv_id

        if operation in {
            "WaveActiveSum",
            "WaveActiveProduct",
            "WaveActiveMin",
            "WaveActiveMax",
            "WaveActiveBitAnd",
            "WaveActiveBitOr",
            "WaveActiveBitXor",
            "WavePrefixSum",
            "WavePrefixProduct",
        }:
            return self.call_group_non_uniform_arithmetic(operation, args)

        if operation in {"WaveActiveAllTrue", "WaveActiveAnyTrue"}:
            return self.call_group_non_uniform_vote(operation, args)

        if operation == "WaveActiveBallot":
            return self.call_group_non_uniform_ballot(operation, args)

        if operation == "WaveReadLaneAt":
            return self.call_group_non_uniform_broadcast(operation, args)

        if operation == "WaveReadLaneFirst":
            return self.call_group_non_uniform_broadcast_first(operation, args)

        if operation == "WaveMatch":
            return self.call_group_non_uniform_partition(args)

        if operation in {
            "WaveMultiPrefixSum",
            "WaveMultiPrefixProduct",
            "WaveMultiPrefixBitAnd",
            "WaveMultiPrefixBitOr",
            "WaveMultiPrefixBitXor",
        }:
            return self.call_group_non_uniform_partitioned_arithmetic(operation, args)

        if operation in {
            "QuadReadAcrossX",
            "QuadReadAcrossY",
            "QuadReadAcrossDiagonal",
            "QuadReadLaneAt",
        }:
            return self.call_group_non_uniform_quad(operation, argument_exprs, args)

        self.emit(f"; WARNING: {operation} is not supported by the SPIR-V backend yet")
        return self.wave_result_default(operation, args)

    def call_group_non_uniform_arithmetic(
        self, operation: str, args: List[SpirvId]
    ) -> SpirvId:
        if len(args) != 1:
            self.emit(f"; WARNING: {operation} requires exactly one argument")
            return self.wave_result_default(operation, args)

        result_type = self.ensure_registered_type(args[0].type)
        component_type = self.scalar_or_vector_component_type(result_type.type)
        group_operation = (
            "ExclusiveScan"
            if operation in {"WavePrefixSum", "WavePrefixProduct"}
            else "Reduce"
        )

        arithmetic_ops = {
            "WaveActiveSum": {
                "float": "OpGroupNonUniformFAdd",
                "double": "OpGroupNonUniformFAdd",
                "int": "OpGroupNonUniformIAdd",
                "uint": "OpGroupNonUniformIAdd",
            },
            "WavePrefixSum": {
                "float": "OpGroupNonUniformFAdd",
                "double": "OpGroupNonUniformFAdd",
                "int": "OpGroupNonUniformIAdd",
                "uint": "OpGroupNonUniformIAdd",
            },
            "WaveActiveProduct": {
                "float": "OpGroupNonUniformFMul",
                "double": "OpGroupNonUniformFMul",
                "int": "OpGroupNonUniformIMul",
                "uint": "OpGroupNonUniformIMul",
            },
            "WavePrefixProduct": {
                "float": "OpGroupNonUniformFMul",
                "double": "OpGroupNonUniformFMul",
                "int": "OpGroupNonUniformIMul",
                "uint": "OpGroupNonUniformIMul",
            },
            "WaveActiveMin": {
                "float": "OpGroupNonUniformFMin",
                "double": "OpGroupNonUniformFMin",
                "int": "OpGroupNonUniformSMin",
                "uint": "OpGroupNonUniformUMin",
            },
            "WaveActiveMax": {
                "float": "OpGroupNonUniformFMax",
                "double": "OpGroupNonUniformFMax",
                "int": "OpGroupNonUniformSMax",
                "uint": "OpGroupNonUniformUMax",
            },
            "WaveActiveBitAnd": {
                "int": "OpGroupNonUniformBitwiseAnd",
                "uint": "OpGroupNonUniformBitwiseAnd",
            },
            "WaveActiveBitOr": {
                "int": "OpGroupNonUniformBitwiseOr",
                "uint": "OpGroupNonUniformBitwiseOr",
            },
            "WaveActiveBitXor": {
                "int": "OpGroupNonUniformBitwiseXor",
                "uint": "OpGroupNonUniformBitwiseXor",
            },
        }
        opcode = arithmetic_ops[operation].get(component_type)
        if opcode is None:
            self.emit(
                f"; WARNING: {operation} requires a compatible arithmetic or "
                f"bitwise operand; got {result_type.type.base_type}"
            )
            return self.wave_result_default(operation, args)

        self.require_group_non_uniform("GroupNonUniformArithmetic")
        scope = self.subgroup_scope_id()
        id_value = self.get_id()
        self.emit(
            f"%{id_value} = {opcode} %{result_type.id} %{scope.id} "
            f"{group_operation} %{args[0].id}"
        )

        spirv_id = SpirvId(id_value, result_type.type)
        self.value_types[id_value] = result_type
        return spirv_id

    def call_group_non_uniform_vote(
        self, operation: str, args: List[SpirvId]
    ) -> SpirvId:
        if len(args) != 1:
            self.emit(f"; WARNING: {operation} requires exactly one bool argument")
            return self.wave_result_default(operation, args)

        bool_type = self.register_primitive_type("bool")
        if args[0].type.base_type != "bool":
            self.emit(f"; WARNING: {operation} requires a scalar bool argument")
            return self.wave_result_default(operation, args)

        self.require_group_non_uniform("GroupNonUniformVote")
        scope = self.subgroup_scope_id()
        opcode = (
            "OpGroupNonUniformAll"
            if operation == "WaveActiveAllTrue"
            else "OpGroupNonUniformAny"
        )
        id_value = self.get_id()
        self.emit(f"%{id_value} = {opcode} %{bool_type.id} %{scope.id} %{args[0].id}")

        spirv_id = SpirvId(id_value, bool_type.type)
        self.value_types[id_value] = bool_type
        return spirv_id

    def call_group_non_uniform_ballot(
        self, operation: str, args: List[SpirvId]
    ) -> SpirvId:
        uint_type = self.register_primitive_type("uint")
        ballot_type = self.register_vector_type(uint_type, 4)
        if len(args) != 1:
            self.emit(f"; WARNING: {operation} requires exactly one bool argument")
            return self.wave_result_default(operation, args)

        if args[0].type.base_type != "bool":
            self.emit(f"; WARNING: {operation} requires a scalar bool argument")
            return self.wave_result_default(operation, args)

        self.require_group_non_uniform("GroupNonUniformBallot")
        scope = self.subgroup_scope_id()
        id_value = self.get_id()
        self.emit(
            f"%{id_value} = OpGroupNonUniformBallot %{ballot_type.id} "
            f"%{scope.id} %{args[0].id}"
        )

        spirv_id = SpirvId(id_value, ballot_type.type)
        self.value_types[id_value] = ballot_type
        return spirv_id

    def call_group_non_uniform_broadcast(
        self, operation: str, args: List[SpirvId]
    ) -> SpirvId:
        if len(args) != 2:
            self.emit(f"; WARNING: {operation} requires value and lane arguments")
            return self.wave_result_default(operation, args)

        result_type = self.ensure_registered_type(args[0].type)
        lane = self.convert_wave_lane_operand(operation, args[1])
        if lane is None:
            return self.wave_result_default(operation, args)

        self.require_group_non_uniform("GroupNonUniformShuffle")
        scope = self.subgroup_scope_id()
        id_value = self.get_id()
        self.emit(
            f"%{id_value} = OpGroupNonUniformBroadcast %{result_type.id} "
            f"%{scope.id} %{args[0].id} %{lane.id}"
        )

        spirv_id = SpirvId(id_value, result_type.type)
        self.value_types[id_value] = result_type
        return spirv_id

    def call_group_non_uniform_broadcast_first(
        self, operation: str, args: List[SpirvId]
    ) -> SpirvId:
        if len(args) != 1:
            self.emit(f"; WARNING: {operation} requires exactly one argument")
            return self.wave_result_default(operation, args)

        result_type = self.ensure_registered_type(args[0].type)
        self.require_group_non_uniform("GroupNonUniformBallot")
        scope = self.subgroup_scope_id()
        id_value = self.get_id()
        self.emit(
            f"%{id_value} = OpGroupNonUniformBroadcastFirst %{result_type.id} "
            f"%{scope.id} %{args[0].id}"
        )

        spirv_id = SpirvId(id_value, result_type.type)
        self.value_types[id_value] = result_type
        return spirv_id

    def call_group_non_uniform_partition(self, args: List[SpirvId]) -> SpirvId:
        if len(args) != 1:
            self.emit("; WARNING: WaveMatch requires exactly one argument")
            return self.wave_result_default("WaveMatch", args)

        uint_type = self.register_primitive_type("uint")
        result_type = self.register_vector_type(uint_type, 4)
        self.require_group_non_uniform_partitioned_nv()
        id_value = self.get_id()
        self.emit(
            f"%{id_value} = OpGroupNonUniformPartitionNV %{result_type.id} "
            f"%{args[0].id}"
        )

        spirv_id = SpirvId(id_value, result_type.type)
        self.value_types[id_value] = result_type
        return spirv_id

    def call_group_non_uniform_partitioned_arithmetic(
        self, operation: str, args: List[SpirvId]
    ) -> SpirvId:
        if len(args) != 2:
            self.emit(f"; WARNING: {operation} requires value and ballot arguments")
            return self.wave_result_default(operation, args)

        result_type = self.ensure_registered_type(args[0].type)
        component_type = self.scalar_or_vector_component_type(result_type.type)
        arithmetic_ops = {
            "WaveMultiPrefixSum": {
                "float": "OpGroupNonUniformFAdd",
                "double": "OpGroupNonUniformFAdd",
                "int": "OpGroupNonUniformIAdd",
                "uint": "OpGroupNonUniformIAdd",
            },
            "WaveMultiPrefixProduct": {
                "float": "OpGroupNonUniformFMul",
                "double": "OpGroupNonUniformFMul",
                "int": "OpGroupNonUniformIMul",
                "uint": "OpGroupNonUniformIMul",
            },
            "WaveMultiPrefixBitAnd": {
                "int": "OpGroupNonUniformBitwiseAnd",
                "uint": "OpGroupNonUniformBitwiseAnd",
            },
            "WaveMultiPrefixBitOr": {
                "int": "OpGroupNonUniformBitwiseOr",
                "uint": "OpGroupNonUniformBitwiseOr",
            },
            "WaveMultiPrefixBitXor": {
                "int": "OpGroupNonUniformBitwiseXor",
                "uint": "OpGroupNonUniformBitwiseXor",
            },
        }
        opcode = arithmetic_ops[operation].get(component_type)
        if opcode is None:
            self.emit(
                f"; WARNING: {operation} requires a compatible arithmetic or "
                f"bitwise operand; got {result_type.type.base_type}"
            )
            return self.wave_result_default(operation, args)

        if self.vector_component_type_and_count(args[1].type.base_type) != ("uint", 4):
            self.emit(f"; WARNING: {operation} requires a uvec4 ballot argument")
            return self.wave_result_default(operation, args)

        self.require_group_non_uniform("GroupNonUniformArithmetic")
        self.require_group_non_uniform_partitioned_nv()
        scope = self.subgroup_scope_id()
        id_value = self.get_id()
        self.emit(
            f"%{id_value} = {opcode} %{result_type.id} %{scope.id} "
            f"PartitionedExclusiveScanNV %{args[0].id} %{args[1].id}"
        )

        spirv_id = SpirvId(id_value, result_type.type)
        self.value_types[id_value] = result_type
        return spirv_id

    def convert_wave_lane_operand(
        self, operation: str, lane: SpirvId
    ) -> Optional[SpirvId]:
        if self.vector_component_type_and_count(lane.type.base_type) is not None:
            self.emit(f"; WARNING: {operation} requires a scalar lane index")
            return None

        uint_type = self.register_primitive_type("uint")
        converted = self.convert_scalar_to_type(lane, uint_type)
        if converted.type.base_type != "uint":
            self.emit(f"; WARNING: {operation} requires an integer lane index")
            return None
        return converted

    def call_group_non_uniform_quad(
        self, operation: str, argument_exprs: List, args: List[SpirvId]
    ) -> SpirvId:
        if operation in {
            "QuadReadAcrossX",
            "QuadReadAcrossY",
            "QuadReadAcrossDiagonal",
        }:
            if len(args) != 1:
                self.emit(f"; WARNING: {operation} requires exactly one argument")
                return self.wave_result_default(operation, args)

            directions = {
                "QuadReadAcrossX": 0,
                "QuadReadAcrossY": 1,
                "QuadReadAcrossDiagonal": 2,
            }
            result_type = self.ensure_registered_type(args[0].type)
            direction = self.register_constant(
                directions[operation], self.register_primitive_type("uint")
            )
            self.require_group_non_uniform("GroupNonUniformQuad")
            scope = self.subgroup_scope_id()
            id_value = self.get_id()
            self.emit(
                f"%{id_value} = OpGroupNonUniformQuadSwap %{result_type.id} "
                f"%{scope.id} %{args[0].id} %{direction.id}"
            )

            spirv_id = SpirvId(id_value, result_type.type)
            self.value_types[id_value] = result_type
            return spirv_id

        if len(args) != 2:
            self.emit("; WARNING: QuadReadLaneAt requires value and lane arguments")
            return self.wave_result_default(operation, args)

        result_type = self.ensure_registered_type(args[0].type)
        lane_value = self.literal_integer_value(argument_exprs[1], "uint")
        if lane_value is None or not 0 <= lane_value <= 3:
            self.emit(
                "; WARNING: QuadReadLaneAt requires a literal lane index "
                "between 0 and 3"
            )
            return self.wave_result_default(operation, args)

        lane = self.register_constant(lane_value, self.register_primitive_type("uint"))
        self.require_group_non_uniform("GroupNonUniformQuad")
        scope = self.subgroup_scope_id()
        id_value = self.get_id()
        self.emit(
            f"%{id_value} = OpGroupNonUniformQuadBroadcast %{result_type.id} "
            f"%{scope.id} %{args[0].id} %{lane.id}"
        )

        spirv_id = SpirvId(id_value, result_type.type)
        self.value_types[id_value] = result_type
        return spirv_id

    def bitcast_builtin_target_component_type(
        self, function_name: str
    ) -> Optional[str]:
        return {
            "asfloat": "float",
            "asint": "int",
            "asuint": "uint",
            "floatBitsToInt": "int",
            "floatBitsToUint": "uint",
            "intBitsToFloat": "float",
            "uintBitsToFloat": "float",
        }.get(function_name)

    def bitcast_builtin_result_type(
        self, function_name: str, source_type: Optional[SpirvId]
    ) -> Optional[SpirvId]:
        target_component_name = self.bitcast_builtin_target_component_type(
            function_name
        )
        if target_component_name is None:
            return None

        target_component_type = self.register_primitive_type(target_component_name)
        if source_type is None:
            return target_component_type

        source_vector = self.vector_component_type_and_count(source_type.type.base_type)
        if source_vector is None:
            return target_component_type

        return self.register_vector_type(target_component_type, source_vector[1])

    def bitcast_builtin_source_is_valid(self, source_type: SpirvId) -> bool:
        source_vector = self.vector_component_type_and_count(source_type.type.base_type)
        if source_vector is not None:
            source_component_name = source_vector[0]
        else:
            source_component_name = self.normalize_primitive_name(
                source_type.type.base_type
            )
        return source_component_name in {"float", "int", "uint"}

    def call_bitcast_builtin_function(
        self, function_name: str, args: List[SpirvId]
    ) -> Optional[SpirvId]:
        if function_name in self.struct_types:
            return None

        target_component_name = self.bitcast_builtin_target_component_type(
            function_name
        )
        if target_component_name is None:
            return None

        if len(args) != 1:
            self.emit(f"; WARNING: {function_name} requires exactly one operand")
            result_type = self.register_primitive_type(target_component_name)
            return self.default_value_for_type(result_type)

        source_type = self.registered_value_type(
            args[0]
        ) or self.ensure_registered_type(args[0].type)
        result_type = self.bitcast_builtin_result_type(function_name, source_type)
        if result_type is None:
            return None

        if not self.bitcast_builtin_source_is_valid(source_type):
            self.emit(
                f"; WARNING: {function_name} requires a 32-bit numeric scalar "
                "or vector operand"
            )
            return self.default_value_for_type(result_type)

        if source_type.type.base_type == result_type.type.base_type:
            return args[0]

        id_value = self.get_id()
        self.emit(f"%{id_value} = OpBitcast %{result_type.id} %{args[0].id}")
        spirv_id = SpirvId(id_value, result_type.type)
        self.value_types[id_value] = result_type
        if self.is_non_uniform_value(args[0]):
            self.mark_non_uniform_result(spirv_id)
        return spirv_id

    def infer_bitcast_builtin_result_type(
        self, function_name: str, args: List
    ) -> Optional[SpirvId]:
        target_component_name = self.bitcast_builtin_target_component_type(
            function_name
        )
        if target_component_name is None:
            return None
        if len(args) != 1:
            return self.register_primitive_type(target_component_name)

        source_type = self.infer_expression_result_type(args[0])
        return self.bitcast_builtin_result_type(function_name, source_type)

    def normalize_integer_bit_builtin_name(self, function_name: str) -> str:
        return {
            "countbits": "bitCount",
            "reversebits": "bitfieldReverse",
            "firstbitlow": "findLSB",
            "firstbithigh": "findMSB",
        }.get(function_name, function_name)

    def integer_bit_builtin_result_type(
        self, function_name: str, source_type: Optional[SpirvId]
    ) -> Optional[SpirvId]:
        function_name = self.normalize_integer_bit_builtin_name(function_name)
        if function_name not in {"bitCount", "bitfieldReverse", "findLSB", "findMSB"}:
            return None

        if source_type is None:
            return self.register_primitive_type("int")

        source_type = self.ensure_registered_type(source_type)
        vector_info = self.vector_component_type_and_count(source_type.type.base_type)
        if vector_info is not None:
            component_type, component_count = vector_info
            if component_type in {"int", "uint"}:
                return source_type
            return self.register_vector_type(
                self.register_primitive_type("int"), component_count
            )

        component_type = self.normalize_primitive_name(source_type.type.base_type)
        if component_type in {"int", "uint"}:
            return source_type
        return self.register_primitive_type("int")

    def integer_bit_builtin_operand_is_valid(self, operand: SpirvId) -> bool:
        operand_type = self.registered_value_type(
            operand
        ) or self.ensure_registered_type(operand.type)
        component_type = self.scalar_or_vector_component_type(operand_type.type)
        return self.normalize_primitive_name(component_type) in {"int", "uint"}

    def call_integer_bit_builtin_function(
        self, function_name: str, args: List[SpirvId]
    ) -> Optional[SpirvId]:
        if function_name in self.struct_types:
            return None

        normalized_name = self.normalize_integer_bit_builtin_name(function_name)
        if normalized_name not in {"bitCount", "bitfieldReverse", "findLSB", "findMSB"}:
            return None

        if len(args) != 1:
            self.emit(f"; WARNING: {function_name} requires exactly one operand")
            result_type = self.integer_bit_builtin_result_type(normalized_name, None)
            return self.default_value_for_type(result_type)

        source_type = self.registered_value_type(
            args[0]
        ) or self.ensure_registered_type(args[0].type)
        result_type = self.integer_bit_builtin_result_type(normalized_name, source_type)
        if result_type is None:
            return None

        if not self.integer_bit_builtin_operand_is_valid(args[0]):
            self.emit(
                f"; WARNING: {function_name} requires an integer scalar or vector "
                "operand"
            )
            return self.default_value_for_type(result_type)

        if normalized_name == "bitCount":
            opcode = "OpBitCount"
        elif normalized_name == "bitfieldReverse":
            opcode = "OpBitReverse"
        else:
            component_type = self.scalar_or_vector_component_type(source_type.type)
            if normalized_name == "findLSB":
                instruction = "FindILsb"
            elif self.normalize_primitive_name(component_type) == "uint":
                instruction = "FindUMsb"
            else:
                instruction = "FindSMsb"
            return self.emit_glsl_std450_instruction(instruction, result_type, args)

        id_value = self.get_id()
        self.emit(f"%{id_value} = {opcode} %{result_type.id} %{args[0].id}")
        spirv_id = SpirvId(id_value, result_type.type)
        self.value_types[id_value] = result_type
        if self.is_non_uniform_value(args[0]):
            self.mark_non_uniform_result(spirv_id)
        return spirv_id

    def infer_integer_bit_builtin_result_type(
        self, function_name: str, args: List
    ) -> Optional[SpirvId]:
        normalized_name = self.normalize_integer_bit_builtin_name(function_name)
        if normalized_name not in {"bitCount", "bitfieldReverse", "findLSB", "findMSB"}:
            return None
        if len(args) != 1:
            return self.integer_bit_builtin_result_type(normalized_name, None)
        return self.integer_bit_builtin_result_type(
            normalized_name, self.infer_expression_result_type(args[0])
        )

    def call_builtin_function(
        self, function_name: str, args: List[SpirvId]
    ) -> Optional[SpirvId]:
        """Call a built-in function."""
        function_name = {
            "frac": "fract",
            "inverseSqrt": "inversesqrt",
            "lerp": "mix",
            "rsqrt": "inversesqrt",
        }.get(function_name, function_name)
        synchronization_call = self.call_synchronization_function(function_name, args)
        if synchronization_call is not None:
            return synchronization_call

        scalar_constructor = self.call_scalar_constructor(function_name, args)
        if scalar_constructor is not None:
            return scalar_constructor

        derivative_call = self.call_derivative_function(function_name, args)
        if derivative_call is not None:
            return derivative_call

        bitcast_call = self.call_bitcast_builtin_function(function_name, args)
        if bitcast_call is not None:
            return bitcast_call

        integer_bit_call = self.call_integer_bit_builtin_function(function_name, args)
        if integer_bit_call is not None:
            return integer_bit_call

        exact_operand_counts = {
            "min": 2,
            "max": 2,
            "clamp": 3,
            "mix": 3,
            "pow": 2,
            "fma": 3,
            "faceforward": 3,
            "refract": 3,
            "length": 1,
            "distance": 2,
            "normalize": 1,
            "reflect": 2,
            "step": 2,
            "smoothstep": 3,
        }
        expected_operand_count = exact_operand_counts.get(function_name)
        if expected_operand_count is not None and len(args) != expected_operand_count:
            operand_label = "operand" if expected_operand_count == 1 else "operands"
            self.emit(
                f"; WARNING: {function_name} requires "
                f"{expected_operand_count} {operand_label}"
            )
            fallback_type = self.register_primitive_type("float")
            if function_name in {"length", "distance"}:
                fallback_type = self.metric_result_type(args)
            elif args:
                fallback_type = self.ensure_registered_type(args[0].type)
            return self.default_value_for_type(fallback_type)

        if function_name == "mix" and len(args) == 3:
            bool_mix = self.call_bool_mix_function(args)
            if bool_mix is not None:
                return bool_mix

        if self.glsl_std450_id is None:
            self.glsl_std450_id = self.get_id()
            self.emit(f'%{self.glsl_std450_id} = OpExtInstImport "GLSL.std.450"')

        if function_name == "saturate" and len(args) == 1:
            saturated = self.call_saturate_function(args[0])
            if saturated is not None:
                return saturated

        if function_name == "sign" and len(args) == 1:
            signed = self.call_sign_function(args[0])
            if signed is not None:
                return signed

        if function_name == "abs" and len(args) == 1:
            absolute = self.call_abs_function(args[0])
            if absolute is not None:
                return absolute
            self.emit(
                "; WARNING: abs requires floating-point or signed integer "
                "scalar or vector operand"
            )
            return self.default_value_for_type(
                self.ensure_registered_type(args[0].type)
            )

        if function_name == "pow" and len(args) == 2:
            power = self.call_pow_function(args)
            if power is not None:
                return power
            self.emit(
                "; WARNING: pow requires compatible 32-bit floating-point "
                "scalar or vector operands"
            )
            return self.default_value_for_type(
                self.ensure_registered_type(args[0].type)
            )

        if function_name == "fma" and len(args) == 3:
            fused = self.call_fma_function(args)
            if fused is not None:
                return fused
            self.emit(
                "; WARNING: fma requires compatible floating-point scalar "
                "or vector operands"
            )
            return self.default_value_for_type(
                self.ensure_registered_type(args[0].type)
            )

        if function_name == "faceforward" and len(args) == 3:
            faced = self.call_faceforward_function(args)
            if faced is not None:
                return faced
            self.emit(
                "; WARNING: faceforward requires compatible floating-point "
                "scalar or vector operands"
            )
            return self.default_value_for_type(
                self.ensure_registered_type(args[0].type)
            )

        if function_name == "refract" and len(args) == 3:
            refracted = self.call_refract_function(args)
            if refracted is not None:
                return refracted
            self.emit(
                "; WARNING: refract requires matching floating-point incident "
                "and normal operands plus scalar eta"
            )
            return self.default_value_for_type(
                self.ensure_registered_type(args[0].type)
            )

        if function_name == "length" and len(args) == 1:
            measured = self.call_length_function(args[0])
            if measured is not None:
                return measured
            self.emit(
                "; WARNING: length requires floating-point scalar or vector operand"
            )
            return self.default_value_for_type(self.metric_result_type(args))

        if function_name == "distance" and len(args) == 2:
            measured = self.call_distance_function(args)
            if measured is not None:
                return measured
            self.emit(
                "; WARNING: distance requires matching floating-point scalar "
                "or vector operands"
            )
            return self.default_value_for_type(self.metric_result_type(args))

        if function_name == "normalize" and len(args) == 1:
            normalized = self.call_normalize_function(args[0])
            if normalized is not None:
                return normalized
            self.emit(
                "; WARNING: normalize requires floating-point scalar or vector operand"
            )
            return self.default_value_for_type(
                self.ensure_registered_type(args[0].type)
            )

        if function_name == "reflect" and len(args) == 2:
            reflected = self.call_reflect_function(args)
            if reflected is not None:
                return reflected
            self.emit(
                "; WARNING: reflect requires matching floating-point scalar "
                "or vector operands"
            )
            return self.default_value_for_type(
                self.ensure_registered_type(args[0].type)
            )

        if function_name == "step" and len(args) == 2:
            stepped = self.call_step_smoothstep_function("Step", args, 1)
            if stepped is not None:
                return stepped
            self.emit(
                "; WARNING: step requires compatible floating-point scalar "
                "or vector operands"
            )
            return self.default_value_for_type(
                self.ensure_registered_type(args[1].type)
            )

        if function_name == "smoothstep" and len(args) == 3:
            smoothed = self.call_step_smoothstep_function("SmoothStep", args, 2)
            if smoothed is not None:
                return smoothed
            self.emit(
                "; WARNING: smoothstep requires compatible floating-point "
                "scalar or vector operands"
            )
            return self.default_value_for_type(
                self.ensure_registered_type(args[2].type)
            )

        if (
            function_name in {"min", "max"}
            and len(args) == 2
            or function_name == "clamp"
            and len(args) == 3
        ):
            min_max_clamp = self.call_min_max_clamp_function(function_name, args)
            if min_max_clamp is not None:
                return min_max_clamp
            self.emit(
                f"; WARNING: {function_name} requires compatible scalar or "
                "vector numeric operands"
            )
            return self.default_value_for_type(
                self.ensure_registered_type(args[0].type)
            )

        if function_name == "mix" and len(args) == 3:
            numeric_mix = self.call_numeric_mix_function(args)
            if numeric_mix is not None:
                return numeric_mix
            self.emit("; WARNING: mix requires compatible value operands and selector")
            return self.default_value_for_type(
                self.ensure_registered_type(args[0].type)
            )

        vector_info = self.vector_component_type_and_count(function_name)
        if vector_info:
            component_type_name, component_count = vector_info
            component_type = self.register_primitive_type(component_type_name)
            vector_type = self.register_vector_type(component_type, component_count)
            if not args:
                if component_type_name == "bool":
                    zero_value = False
                    one_value = True
                elif component_type_name in {"int", "uint"}:
                    zero_value = 0
                    one_value = 1
                else:
                    zero_value = 0.0
                    one_value = 1.0

                # Preserve old defaults: first component one, rest zero.
                component_zero = self.register_constant(zero_value, component_type)
                component_one = self.register_constant(one_value, component_type)

                default_args = [component_zero] * component_count
                if component_count > 0:
                    default_args[0] = component_one

                constructor_args = default_args
            else:
                constructor_args = self.flatten_vector_constructor_args(
                    function_name, args, component_type, component_count
                )

            return self.composite_construct(vector_type, constructor_args)

        if function_name in self.struct_types:
            struct_type = self.struct_types[function_name]
            constructor_args = self.struct_constructor_args(function_name, args)
            return self.composite_construct(struct_type, constructor_args)

        matrix_function_name = self.normalize_hlsl_matrix_type(function_name)

        # Matrix constructors
        if re.fullmatch(r"(d)?mat([234])(?:x([234]))?", matrix_function_name):
            match = re.fullmatch(r"(d)?mat([234])(?:x([234]))?", matrix_function_name)
            is_double, cols, rows = match.groups()
            cols = int(cols)
            rows = int(rows or cols)

            component_type = self.register_primitive_type(
                "double" if is_double else "float"
            )
            vector_type = self.register_vector_type(component_type, rows)
            matrix_type = self.register_matrix_type(vector_type, cols)

            if not args:
                zero_value = self.register_constant(0.0, component_type)
                one_value = self.register_constant(1.0, component_type)

                col_vectors = []
                for col in range(cols):
                    col_components = []
                    for row in range(rows):
                        if col == row:
                            col_components.append(one_value)
                        else:
                            col_components.append(zero_value)

                    col_vectors.append(
                        self.composite_construct(vector_type, col_components)
                    )
            else:
                components = self.flatten_matrix_constructor_components(
                    function_name, args, component_type, cols * rows
                )
                col_vectors = []
                for col in range(cols):
                    start = col * rows
                    col_vectors.append(
                        self.composite_construct(
                            vector_type, components[start : start + rows]
                        )
                    )

            return self.composite_construct(matrix_type, col_vectors)

        # Special case for dot product - use OpDot instead of OpExtInst
        elif function_name == "dot" and len(args) == 2:
            if not self.dot_operands_are_valid(args):
                self.emit(
                    "; WARNING: dot requires matching floating-point vector operands"
                )
                return self.default_value_for_type(self.dot_result_type(args))

            component_type = self.scalar_or_vector_component_type(args[0].type)
            result_type = self.register_primitive_type(component_type)

            id_value = self.get_id()
            self.emit(
                f"%{id_value} = OpDot %{result_type.id} %{args[0].id} %{args[1].id}"
            )
            self.decorate_no_contraction_result(id_value, "OpDot", result_type)

            spirv_id = SpirvId(id_value, result_type.type)
            self.value_types[id_value] = result_type
            return spirv_id

        elif function_name in {"fmod", "mod"} and len(args) == 2:
            result_type = self.ensure_registered_type(args[0].type)
            return self.binary_operation("%", result_type, args[0], args[1])

        # GLSL standard library functions
        else:
            float_type = self.primitive_types["float"]
            glsl_std450_map = {
                "sin": "Sin",
                "cos": "Cos",
                "tan": "Tan",
                "atan2": "Atan2",
                "asin": "Asin",
                "acos": "Acos",
                "atan": "Atan",
                "sinh": "Sinh",
                "cosh": "Cosh",
                "tanh": "Tanh",
                "exp": "Exp",
                "log": "Log",
                "exp2": "Exp2",
                "log2": "Log2",
                "sqrt": "Sqrt",
                "inversesqrt": "InverseSqrt",
                "abs": "FAbs",
                "floor": "Floor",
                "ceil": "Ceil",
                "fract": "Fract",
                "trunc": "Trunc",
                "round": "Round",
                "roundEven": "RoundEven",
                "degrees": "Degrees",
                "radians": "Radians",
                "length": "Length",
                "distance": "Distance",
                "cross": "Cross",
                "normalize": "Normalize",
                "reflect": "Reflect",
                "refract": "Refract",
            }
            std450_function_name = (
                "atan2" if function_name == "atan" and len(args) == 2 else function_name
            )
            if std450_function_name not in glsl_std450_map:
                self.emit(
                    f"; WARNING: SPIR-V backend cannot lower unknown function "
                    f"'{function_name}'; using default value"
                )
                return self.default_value_for_type(
                    self.unknown_function_fallback_type(args)
                )

            unary_std450_functions = {
                "sin",
                "cos",
                "tan",
                "asin",
                "acos",
                "atan",
                "sinh",
                "cosh",
                "tanh",
                "exp",
                "log",
                "exp2",
                "log2",
                "sqrt",
                "inversesqrt",
                "abs",
                "floor",
                "ceil",
                "fract",
                "trunc",
                "round",
                "roundEven",
                "degrees",
                "radians",
                "length",
                "normalize",
            }
            std450_operand_counts = {name: 1 for name in unary_std450_functions}
            std450_operand_counts.update(
                {
                    "atan2": 2,
                    "distance": 2,
                    "cross": 2,
                    "reflect": 2,
                    "refract": 3,
                }
            )
            expected_operand_count = std450_operand_counts.get(std450_function_name)
            if (
                expected_operand_count is not None
                and len(args) != expected_operand_count
            ):
                operand_label = "operand" if expected_operand_count == 1 else "operands"
                self.emit(
                    f"; WARNING: {function_name} requires "
                    f"{expected_operand_count} {operand_label}"
                )
                fallback_type = float_type
                if args and std450_function_name not in {"length", "distance"}:
                    fallback_type = self.ensure_registered_type(args[0].type)
                return self.default_value_for_type(fallback_type)

            vector_math_diagnostic = self.vector_math_operand_diagnostic(
                std450_function_name, args
            )
            if vector_math_diagnostic is not None:
                warning, fallback_type = vector_math_diagnostic
                self.emit(warning)
                return self.default_value_for_type(fallback_type)

            result_type = float_type.type

            if args:
                if std450_function_name in [
                    "sin",
                    "cos",
                    "tan",
                    "atan2",
                    "asin",
                    "acos",
                    "atan",
                    "sinh",
                    "cosh",
                    "tanh",
                    "exp",
                    "log",
                    "exp2",
                    "log2",
                    "sqrt",
                    "inversesqrt",
                    "abs",
                    "floor",
                    "ceil",
                    "fract",
                    "trunc",
                    "round",
                    "roundEven",
                    "degrees",
                    "radians",
                ]:
                    # These functions return the same type as their first argument
                    result_type = args[0].type
                elif std450_function_name in ["length", "distance"]:
                    # These functions return the scalar component type.
                    result_type = self.metric_result_type(args).type
                elif std450_function_name in ["normalize", "reflect", "refract"]:
                    # These functions return the same vector type as their first argument
                    result_type = args[0].type
                elif std450_function_name in ["cross"]:
                    # cross product returns a vec3
                    component_type_name = "float"
                    vector_info = self.vector_component_type_and_count(
                        args[0].type.base_type
                    )
                    if vector_info is not None:
                        component_type_name = vector_info[0]
                    component_type = self.register_primitive_type(component_type_name)
                    vector_type = self.register_vector_type(component_type, 3)
                    result_type = vector_type.type

            result_type_id = self.ensure_registered_type(result_type)
            return self.emit_glsl_std450_instruction(
                glsl_std450_map[std450_function_name], result_type_id, args
            )

    def derivative_function_opcode(self, function_name: str) -> Optional[str]:
        aliases = {
            "ddx": "dFdx",
            "ddx_fine": "dFdxFine",
            "ddx_coarse": "dFdxCoarse",
            "ddy": "dFdy",
            "ddy_fine": "dFdyFine",
            "ddy_coarse": "dFdyCoarse",
            "fwidth_fine": "fwidthFine",
            "fwidth_coarse": "fwidthCoarse",
        }
        function_name = aliases.get(function_name, function_name)
        opcodes = {
            "dFdx": "OpDPdx",
            "dFdxFine": "OpDPdxFine",
            "dFdxCoarse": "OpDPdxCoarse",
            "dFdy": "OpDPdy",
            "dFdyFine": "OpDPdyFine",
            "dFdyCoarse": "OpDPdyCoarse",
            "fwidth": "OpFwidth",
            "fwidthFine": "OpFwidthFine",
            "fwidthCoarse": "OpFwidthCoarse",
        }
        return opcodes.get(function_name)

    def derivative_operand_is_valid(self, value: SpirvId) -> bool:
        result_type = self.ensure_registered_type(value.type)
        component_type = self.scalar_or_vector_component_type(result_type.type)
        if component_type not in {"float", "double"}:
            return False

        type_name = self.normalize_primitive_name(result_type.type.base_type)
        return (
            type_name in {"float", "double"}
            or self.vector_component_type_and_count(result_type.type.base_type)
            is not None
        )

    def current_derivative_execution_models(self) -> set:
        return self.current_synchronization_execution_models()

    def can_emit_derivative_function(self) -> bool:
        execution_models = self.current_derivative_execution_models()
        derivative_execution_models = {"Fragment", "GLCompute", "MeshEXT", "TaskEXT"}
        return bool(execution_models) and execution_models.issubset(
            derivative_execution_models
        )

    def call_derivative_function(
        self, function_name: str, args: List[SpirvId]
    ) -> Optional[SpirvId]:
        opcode = self.derivative_function_opcode(function_name)
        if opcode is None:
            return None

        if len(args) != 1:
            self.emit(f"; WARNING: {function_name} requires 1 operand")
            return self.register_constant(0.0, self.register_primitive_type("float"))

        result_type = self.ensure_registered_type(args[0].type)
        if not self.derivative_operand_is_valid(args[0]):
            self.emit(
                f"; WARNING: {function_name} requires floating-point scalar "
                "or vector operand"
            )
            return self.default_value_for_type(result_type)

        if not self.can_emit_derivative_function():
            self.emit(
                f"; WARNING: derivative builtin '{function_name}' requires Fragment "
                "or compute-derivative-capable execution model; current execution "
                f"model: {self.current_execution_model_label()}"
            )
            return self.default_value_for_type(result_type)

        compute_derivative_models = {"GLCompute", "MeshEXT", "TaskEXT"}
        if self.current_derivative_execution_models() & compute_derivative_models:
            self.require_compute_derivatives()

        if opcode.endswith("Fine") or opcode.endswith("Coarse"):
            self.require_capability("DerivativeControl")

        id_value = self.get_id()
        self.emit(f"%{id_value} = {opcode} %{result_type.id} %{args[0].id}")

        spirv_id = SpirvId(id_value, result_type.type)
        self.value_types[id_value] = result_type
        return spirv_id

    def call_scalar_constructor(
        self, function_name: str, args: List[SpirvId]
    ) -> Optional[SpirvId]:
        primitive_name = self.normalize_primitive_name(function_name)
        if primitive_name not in {"bool", "float", "double"} | self.INTEGER_TYPE_NAMES:
            return None

        target_type = self.register_primitive_type(primitive_name)
        if not args:
            return self.default_value_for_type(target_type)
        if len(args) != 1:
            self.emit(
                f"; WARNING: Constructor {function_name} expected 1 component "
                f"but got {len(args)}; using first component"
            )

        value = args[0]
        if self.vector_component_type_and_count(value.type.base_type) is not None:
            self.emit(
                f"; WARNING: Constructor {function_name} requires a scalar operand; "
                "using default value"
            )
            return self.default_value_for_type(target_type)

        converted = self.convert_scalar_to_type(value, target_type)
        if self.normalize_primitive_name(
            converted.type.base_type
        ) == self.normalize_primitive_name(target_type.type.base_type):
            return converted

        self.emit(
            f"; WARNING: Constructor {function_name} cannot convert "
            f"{value.type.base_type}; using default value"
        )
        return self.default_value_for_type(target_type)

    def dot_result_type(self, args: List[SpirvId]) -> SpirvId:
        for arg in args:
            vector_info = self.vector_component_type_and_count(arg.type.base_type)
            if vector_info is not None and vector_info[0] in {"float", "double"}:
                return self.register_primitive_type(vector_info[0])
        return self.register_primitive_type("float")

    def dot_operands_are_valid(self, args: List[SpirvId]) -> bool:
        left_vector = self.vector_component_type_and_count(args[0].type.base_type)
        right_vector = self.vector_component_type_and_count(args[1].type.base_type)
        if left_vector is None or right_vector is None:
            return False
        if left_vector != right_vector:
            return False
        return left_vector[0] in {"float", "double"}

    def vector_math_operand_diagnostic(
        self, function_name: str, args: List[SpirvId]
    ) -> Optional[Tuple[str, SpirvId]]:
        if self.is_float32_unary_std450_function(
            function_name
        ) and not self.float32_scalar_or_vector_operand_is_valid(args[0]):
            return (
                f"; WARNING: {function_name} requires 32-bit floating-point "
                "scalar or vector operand",
                self.ensure_registered_type(args[0].type),
            )

        if function_name == "length" and not self.metric_operand_is_valid(args[0]):
            return (
                "; WARNING: length requires floating-point scalar or vector operand",
                self.metric_result_type(args),
            )

        if function_name == "distance" and not self.distance_operands_are_valid(args):
            return (
                "; WARNING: distance requires matching floating-point scalar "
                "or vector operands",
                self.metric_result_type(args),
            )

        if function_name == "normalize" and not self.metric_operand_is_valid(args[0]):
            return (
                "; WARNING: normalize requires floating-point scalar or vector operand",
                self.ensure_registered_type(args[0].type),
            )

        if function_name == "atan2" and not self.atan2_operands_are_valid(args):
            return (
                "; WARNING: atan2 requires matching 32-bit floating-point "
                "scalar or vector operands",
                self.ensure_registered_type(args[0].type),
            )

        if function_name == "cross" and not self.cross_operands_are_valid(args):
            return (
                "; WARNING: cross requires matching 3-component "
                "floating-point vector operands",
                self.cross_result_type(args),
            )

        if function_name == "reflect" and not self.reflect_operands_are_valid(args):
            return (
                "; WARNING: reflect requires matching floating-point scalar "
                "or vector operands",
                self.ensure_registered_type(args[0].type),
            )

        if function_name == "refract" and not self.refract_operands_are_valid(args):
            return (
                "; WARNING: refract requires matching floating-point incident "
                "and normal operands plus scalar eta",
                self.ensure_registered_type(args[0].type),
            )

        return None

    def unknown_function_fallback_type(self, args: List[SpirvId]) -> SpirvId:
        if not args:
            return self.register_primitive_type("float")

        component_type = self.scalar_or_vector_component_type(args[0].type)
        primitive_type = self.normalize_primitive_name(component_type)
        if primitive_type in {"float", "double", "bool"} | self.INTEGER_TYPE_NAMES:
            return self.register_primitive_type(primitive_type)

        return self.register_primitive_type("float")

    def is_float32_unary_std450_function(self, function_name: str) -> bool:
        return function_name in {
            "sin",
            "cos",
            "tan",
            "asin",
            "acos",
            "atan",
            "sinh",
            "cosh",
            "tanh",
            "exp",
            "log",
            "exp2",
            "log2",
            "sqrt",
            "inversesqrt",
            "floor",
            "ceil",
            "fract",
            "trunc",
            "round",
            "roundEven",
            "degrees",
            "radians",
        }

    def float32_scalar_or_vector_operand_is_valid(self, arg: SpirvId) -> bool:
        component_type = self.scalar_or_vector_component_type(arg.type)
        if component_type != "float":
            return False
        scalar_type = self.normalize_primitive_name(arg.type.base_type)
        return (
            scalar_type == "float"
            or self.vector_component_type_and_count(arg.type.base_type) is not None
        )

    def metric_result_type(self, args: List[SpirvId]) -> SpirvId:
        for arg in args:
            component_type = self.scalar_or_vector_component_type(arg.type)
            if component_type in {"float", "double"}:
                return self.register_primitive_type(component_type)
        return self.register_primitive_type("float")

    def metric_operand_is_valid(self, arg: SpirvId) -> bool:
        component_type = self.scalar_or_vector_component_type(arg.type)
        if component_type not in {"float", "double"}:
            return False
        scalar_type = self.normalize_primitive_name(arg.type.base_type)
        return (
            scalar_type in {"float", "double"}
            or self.vector_component_type_and_count(arg.type.base_type) is not None
        )

    def distance_operands_are_valid(self, args: List[SpirvId]) -> bool:
        return (
            self.matching_floating_scalar_or_vector_operands(args[0], args[1])
            is not None
        )

    def atan2_operands_are_valid(self, args: List[SpirvId]) -> bool:
        operand_shape = self.matching_floating_scalar_or_vector_operands(
            args[0], args[1]
        )
        if operand_shape is None:
            return False
        component_type, _ = operand_shape
        return component_type == "float"

    def cross_result_type(self, args: List[SpirvId]) -> SpirvId:
        for arg in args:
            vector_info = self.vector_component_type_and_count(arg.type.base_type)
            if vector_info is not None and vector_info[0] in {"float", "double"}:
                component_type = self.register_primitive_type(vector_info[0])
                return self.register_vector_type(component_type, 3)
        return self.register_vector_type(self.register_primitive_type("float"), 3)

    def cross_operands_are_valid(self, args: List[SpirvId]) -> bool:
        left_vector = self.vector_component_type_and_count(args[0].type.base_type)
        right_vector = self.vector_component_type_and_count(args[1].type.base_type)
        return (
            left_vector is not None
            and right_vector is not None
            and left_vector == right_vector
            and left_vector[0] in {"float", "double"}
            and left_vector[1] == 3
        )

    def matching_floating_scalar_or_vector_operands(
        self, left: SpirvId, right: SpirvId
    ) -> Optional[Tuple[str, Optional[int]]]:
        left_vector = self.vector_component_type_and_count(left.type.base_type)
        right_vector = self.vector_component_type_and_count(right.type.base_type)
        left_component = self.scalar_or_vector_component_type(left.type)
        right_component = self.scalar_or_vector_component_type(right.type)
        if left_component not in {"float", "double"}:
            return None
        if left_component != right_component:
            return None
        if left_vector is None and right_vector is None:
            return left_component, None
        if left_vector is None or right_vector is None:
            return None
        if left_vector != right_vector:
            return None
        return left_component, left_vector[1]

    def reflect_operands_are_valid(self, args: List[SpirvId]) -> bool:
        return (
            self.matching_floating_scalar_or_vector_operands(args[0], args[1])
            is not None
        )

    def refract_operands_are_valid(self, args: List[SpirvId]) -> bool:
        operand_shape = self.matching_floating_scalar_or_vector_operands(
            args[0], args[1]
        )
        if operand_shape is None:
            return False
        eta_vector = self.vector_component_type_and_count(args[2].type.base_type)
        eta_component = self.scalar_or_vector_component_type(args[2].type)
        return eta_vector is None and eta_component in {"float", "double"}

    def struct_constructor_args(
        self, struct_name: str, args: List[SpirvId]
    ) -> List[SpirvId]:
        members = self.current_struct_members.get(struct_name, [])
        constructor_args = []
        for index, (member_type, _) in enumerate(members):
            if index < len(args):
                constructor_args.append(
                    self.convert_value_to_type(args[index], member_type)
                )
            else:
                constructor_args.append(self.default_value_for_type(member_type))

        if len(args) < len(members):
            self.emit(
                f"; WARNING: Constructor {struct_name} expected {len(members)} "
                f"members but got {len(args)}; padding with defaults"
            )
        elif len(args) > len(members):
            self.emit(
                f"; WARNING: Constructor {struct_name} expected {len(members)} "
                f"members but got {len(args)}; truncating extra arguments"
            )

        return constructor_args

    def call_numeric_mix_function(self, args: List[SpirvId]) -> Optional[SpirvId]:
        """Lower floating mix with result-typed operands."""
        result_type = self.ensure_registered_type(args[0].type)
        component_type = self.scalar_or_vector_component_type(result_type.type)
        if component_type not in {"float", "double"}:
            return None

        operands = self.match_extinst_operands_to_result_type(result_type, args)
        if operands is None:
            return None

        id_value = self.get_id()
        arg_list = " ".join(f"%{arg.id}" for arg in operands)
        self.emit(
            f"%{id_value} = OpExtInst %{result_type.id} %{self.glsl_std450_id} "
            f"FMix {arg_list}"
        )

        spirv_id = SpirvId(id_value, result_type.type)
        self.value_types[id_value] = result_type
        return spirv_id

    def call_bool_mix_function(self, args: List[SpirvId]) -> Optional[SpirvId]:
        """Lower mix(x, y, bool/bvec) to selector semantics."""
        false_value, true_value, condition = args
        result_type = self.ensure_registered_type(false_value.type)
        true_type = self.ensure_registered_type(true_value.type)
        if result_type.type.base_type != true_type.type.base_type:
            return None

        condition_vector = self.vector_component_type_and_count(
            condition.type.base_type
        )
        result_vector = self.vector_component_type_and_count(result_type.type.base_type)
        if condition_vector is not None:
            if condition_vector[0] != "bool":
                return None
            if result_vector is None or result_vector[1] != condition_vector[1]:
                return None
        elif condition.type.base_type != "bool":
            return None

        return self.select_operation(result_type, condition, true_value, false_value)

    def call_abs_function(self, value: SpirvId) -> Optional[SpirvId]:
        """Lower abs to typed GLSL.std.450 floating or signed-integer ops."""
        result_type = self.ensure_registered_type(value.type)
        component_type = self.scalar_or_vector_component_type(result_type.type)
        glsl_function = {
            "float": "FAbs",
            "double": "FAbs",
            "int": "SAbs",
        }.get(component_type)
        if glsl_function is None:
            return None

        id_value = self.get_id()
        self.emit(
            f"%{id_value} = OpExtInst %{result_type.id} %{self.glsl_std450_id} "
            f"{glsl_function} %{value.id}"
        )

        spirv_id = SpirvId(id_value, result_type.type)
        self.value_types[id_value] = result_type
        return spirv_id

    def call_pow_function(self, args: List[SpirvId]) -> Optional[SpirvId]:
        """Lower pow to GLSL.std.450 with validator-compatible operand types."""
        result_type = self.ensure_registered_type(args[0].type)
        component_type = self.scalar_or_vector_component_type(result_type.type)
        if component_type != "float":
            return None

        operands = self.match_extinst_operands_to_result_type(result_type, args)
        if operands is None:
            return None

        id_value = self.get_id()
        arg_list = " ".join(f"%{arg.id}" for arg in operands)
        self.emit(
            f"%{id_value} = OpExtInst %{result_type.id} %{self.glsl_std450_id} "
            f"Pow {arg_list}"
        )

        spirv_id = SpirvId(id_value, result_type.type)
        self.value_types[id_value] = result_type
        return spirv_id

    def call_fma_function(self, args: List[SpirvId]) -> Optional[SpirvId]:
        """Lower fma to GLSL.std.450 with validator-compatible operand types."""
        result_type = self.ensure_registered_type(args[0].type)
        component_type = self.scalar_or_vector_component_type(result_type.type)
        if component_type not in {"float", "double"}:
            return None

        operands = self.match_extinst_operands_to_result_type(result_type, args)
        if operands is None:
            return None

        id_value = self.get_id()
        arg_list = " ".join(f"%{arg.id}" for arg in operands)
        self.emit(
            f"%{id_value} = OpExtInst %{result_type.id} %{self.glsl_std450_id} "
            f"Fma {arg_list}"
        )

        spirv_id = SpirvId(id_value, result_type.type)
        self.value_types[id_value] = result_type
        return spirv_id

    def call_faceforward_function(self, args: List[SpirvId]) -> Optional[SpirvId]:
        """Lower faceforward to GLSL.std.450 with validator-compatible types."""
        result_type = self.ensure_registered_type(args[0].type)
        component_type = self.scalar_or_vector_component_type(result_type.type)
        if component_type not in {"float", "double"}:
            return None

        operands = self.match_extinst_operands_to_result_type(result_type, args)
        if operands is None:
            return None

        id_value = self.get_id()
        arg_list = " ".join(f"%{arg.id}" for arg in operands)
        self.emit(
            f"%{id_value} = OpExtInst %{result_type.id} %{self.glsl_std450_id} "
            f"FaceForward {arg_list}"
        )

        spirv_id = SpirvId(id_value, result_type.type)
        self.value_types[id_value] = result_type
        return spirv_id

    def call_refract_function(self, args: List[SpirvId]) -> Optional[SpirvId]:
        """Lower refract to GLSL.std.450 with result-typed scalar eta."""
        result_type = self.ensure_registered_type(args[0].type)
        component_type = self.scalar_or_vector_component_type(result_type.type)
        if component_type not in {"float", "double"}:
            return None

        operand_shape = self.matching_floating_scalar_or_vector_operands(
            args[0], args[1]
        )
        if operand_shape is None:
            return None
        operand_component, _ = operand_shape
        if operand_component != component_type:
            return None

        eta_vector = self.vector_component_type_and_count(args[2].type.base_type)
        eta_component = self.scalar_or_vector_component_type(args[2].type)
        if eta_vector is not None or eta_component not in {"float", "double"}:
            return None

        eta_type = self.register_primitive_type(component_type)
        eta = self.convert_scalar_to_type(args[2], eta_type)
        if self.normalize_primitive_name(eta.type.base_type) != component_type:
            return None

        id_value = self.get_id()
        self.emit(
            f"%{id_value} = OpExtInst %{result_type.id} %{self.glsl_std450_id} "
            f"Refract %{args[0].id} %{args[1].id} %{eta.id}"
        )

        spirv_id = SpirvId(id_value, result_type.type)
        self.value_types[id_value] = result_type
        return spirv_id

    def call_length_function(self, value: SpirvId) -> Optional[SpirvId]:
        """Lower length to GLSL.std.450 with scalar component result type."""
        if not self.metric_operand_is_valid(value):
            return None

        result_type = self.metric_result_type([value])
        id_value = self.get_id()
        self.emit(
            f"%{id_value} = OpExtInst %{result_type.id} %{self.glsl_std450_id} "
            f"Length %{value.id}"
        )

        spirv_id = SpirvId(id_value, result_type.type)
        self.value_types[id_value] = result_type
        return spirv_id

    def call_distance_function(self, args: List[SpirvId]) -> Optional[SpirvId]:
        """Lower distance to GLSL.std.450 with matching scalar/vector operands."""
        operand_shape = self.matching_floating_scalar_or_vector_operands(
            args[0], args[1]
        )
        if operand_shape is None:
            return None

        component_type, _ = operand_shape
        result_type = self.register_primitive_type(component_type)
        id_value = self.get_id()
        self.emit(
            f"%{id_value} = OpExtInst %{result_type.id} %{self.glsl_std450_id} "
            f"Distance %{args[0].id} %{args[1].id}"
        )

        spirv_id = SpirvId(id_value, result_type.type)
        self.value_types[id_value] = result_type
        return spirv_id

    def call_normalize_function(self, value: SpirvId) -> Optional[SpirvId]:
        """Lower normalize to GLSL.std.450 with the operand result type."""
        if not self.metric_operand_is_valid(value):
            return None

        result_type = self.ensure_registered_type(value.type)
        id_value = self.get_id()
        self.emit(
            f"%{id_value} = OpExtInst %{result_type.id} %{self.glsl_std450_id} "
            f"Normalize %{value.id}"
        )

        spirv_id = SpirvId(id_value, result_type.type)
        self.value_types[id_value] = result_type
        return spirv_id

    def call_reflect_function(self, args: List[SpirvId]) -> Optional[SpirvId]:
        """Lower reflect to GLSL.std.450 with matching scalar/vector operands."""
        operand_shape = self.matching_floating_scalar_or_vector_operands(
            args[0], args[1]
        )
        if operand_shape is None:
            return None

        result_type = self.ensure_registered_type(args[0].type)
        component_type = self.scalar_or_vector_component_type(result_type.type)
        if component_type not in {"float", "double"}:
            return None

        id_value = self.get_id()
        self.emit(
            f"%{id_value} = OpExtInst %{result_type.id} %{self.glsl_std450_id} "
            f"Reflect %{args[0].id} %{args[1].id}"
        )

        spirv_id = SpirvId(id_value, result_type.type)
        self.value_types[id_value] = result_type
        return spirv_id

    def call_step_smoothstep_function(
        self, glsl_function: str, args: List[SpirvId], value_index: int
    ) -> Optional[SpirvId]:
        """Lower step/smoothstep using the value operand's scalar/vector type."""
        result_type = self.ensure_registered_type(args[value_index].type)
        component_type = self.scalar_or_vector_component_type(result_type.type)
        if component_type not in {"float", "double"}:
            return None

        operands = self.match_extinst_operands_to_result_type(result_type, args)
        if operands is None:
            return None

        id_value = self.get_id()
        arg_list = " ".join(f"%{arg.id}" for arg in operands)
        self.emit(
            f"%{id_value} = OpExtInst %{result_type.id} %{self.glsl_std450_id} "
            f"{glsl_function} {arg_list}"
        )

        spirv_id = SpirvId(id_value, result_type.type)
        self.value_types[id_value] = result_type
        return spirv_id

    def call_sign_function(self, value: SpirvId) -> Optional[SpirvId]:
        """Lower sign to typed GLSL.std.450 or unsigned select operations."""
        result_type = self.ensure_registered_type(value.type)
        component_type = self.scalar_or_vector_component_type(result_type.type)
        if component_type in {"float", "double", "int"}:
            glsl_function = "SSign" if component_type == "int" else "FSign"
            id_value = self.get_id()
            self.emit(
                f"%{id_value} = OpExtInst %{result_type.id} %{self.glsl_std450_id} "
                f"{glsl_function} %{value.id}"
            )

            spirv_id = SpirvId(id_value, result_type.type)
            self.value_types[id_value] = result_type
            return spirv_id

        if component_type == "uint":
            return self.call_unsigned_sign_function(value, result_type)

        return value if component_type == "bool" else None

    def call_unsigned_sign_function(
        self, value: SpirvId, result_type: SpirvId
    ) -> SpirvId:
        """Lower unsigned sign as value > 0 ? 1 : 0."""
        vector_info = self.vector_component_type_and_count(result_type.type.base_type)
        uint_type = self.register_primitive_type("uint")
        bool_type = self.register_primitive_type("bool")
        zero = self.register_constant(0, uint_type)
        one = self.register_constant(1, uint_type)

        if vector_info is not None:
            _, component_count = vector_info
            zero_value = self.register_vector_constant(
                result_type, [zero] * component_count
            )
            one_value = self.register_vector_constant(
                result_type, [one] * component_count
            )
        else:
            zero_value = zero
            one_value = one

        condition = self.binary_operation(">", bool_type, value, zero_value)
        return self.select_operation(result_type, condition, one_value, zero_value)

    def call_min_max_clamp_function(
        self, function_name: str, args: List[SpirvId]
    ) -> Optional[SpirvId]:
        """Lower min/max/clamp to typed GLSL.std.450 extinsts."""
        result_type = self.ensure_registered_type(args[0].type)
        component_type = self.scalar_or_vector_component_type(result_type.type)
        prefix = {
            "float": "F",
            "double": "F",
            "int": "S",
            "uint": "U",
        }.get(component_type)
        if prefix is None:
            return None

        operands = self.match_extinst_operands_to_result_type(result_type, args)
        if operands is None:
            return None

        op_suffix = {
            "min": "Min",
            "max": "Max",
            "clamp": "Clamp",
        }[function_name]
        id_value = self.get_id()
        arg_list = " ".join(f"%{arg.id}" for arg in operands)
        self.emit(
            f"%{id_value} = OpExtInst %{result_type.id} %{self.glsl_std450_id} "
            f"{prefix}{op_suffix} {arg_list}"
        )

        spirv_id = SpirvId(id_value, result_type.type)
        self.value_types[id_value] = result_type
        return spirv_id

    def match_extinst_operands_to_result_type(
        self, result_type: SpirvId, args: List[SpirvId]
    ) -> Optional[List[SpirvId]]:
        """Return operands shaped to the given GLSL.std.450 result type."""
        result_vector = self.vector_component_type_and_count(result_type.type.base_type)
        result_component_type = self.scalar_or_vector_component_type(result_type.type)

        operands = []
        for arg in args:
            arg_vector = self.vector_component_type_and_count(arg.type.base_type)
            if result_vector is None:
                if arg_vector is not None:
                    return None
                scalar_arg = self.convert_scalar_to_type(arg, result_type)
                if (
                    self.normalize_primitive_name(scalar_arg.type.base_type)
                    != result_component_type
                ):
                    return None
                operands.append(scalar_arg)
                continue

            if arg_vector is not None:
                if arg_vector != result_vector:
                    return None
                operands.append(arg)
                continue

            component_type = self.register_primitive_type(result_component_type)
            scalar_arg = self.convert_scalar_to_type(arg, component_type)
            if (
                self.normalize_primitive_name(scalar_arg.type.base_type)
                != result_component_type
            ):
                return None
            operands.append(self.splat_scalar_to_vector(scalar_arg, result_type))

        return operands

    def call_saturate_function(self, value: SpirvId) -> Optional[SpirvId]:
        """Lower floating saturate(x) to GLSL.std.450 FClamp."""
        result_type = self.ensure_registered_type(value.type)
        vector_info = self.vector_component_type_and_count(result_type.type.base_type)
        if vector_info is not None:
            component_type_name, component_count = vector_info
            if component_type_name not in {"float", "double"}:
                return None

            component_type = self.register_primitive_type(component_type_name)
            zero = self.register_constant(0.0, component_type)
            one = self.register_constant(1.0, component_type)
            zero_vector = self.register_vector_constant(
                result_type, [zero] * component_count
            )
            one_vector = self.register_vector_constant(
                result_type, [one] * component_count
            )

            id_value = self.get_id()
            self.emit(
                f"%{id_value} = OpExtInst %{result_type.id} %{self.glsl_std450_id} "
                f"FClamp %{value.id} %{zero_vector.id} %{one_vector.id}"
            )

            spirv_id = SpirvId(id_value, result_type.type)
            self.value_types[id_value] = result_type
            return spirv_id

        if result_type.type.base_type not in {"float", "double"}:
            return None

        zero = self.register_constant(0.0, result_type)
        one = self.register_constant(1.0, result_type)
        id_value = self.get_id()
        self.emit(
            f"%{id_value} = OpExtInst %{result_type.id} %{self.glsl_std450_id} "
            f"FClamp %{value.id} %{zero.id} %{one.id}"
        )

        spirv_id = SpirvId(id_value, result_type.type)
        self.value_types[id_value] = result_type
        return spirv_id

    def create_branch(self, target_label: SpirvId):
        """Create an unconditional branch."""
        self.emit(f"OpBranch %{target_label.id}")

    def create_conditional_branch(
        self, condition: SpirvId, true_label: SpirvId, false_label: SpirvId
    ):
        """Create a conditional branch."""
        self.emit(
            f"OpBranchConditional %{condition.id} %{true_label.id} %{false_label.id}"
        )

    def create_selection_merge(
        self, merge_label: SpirvId, selection_control: str = "None"
    ):
        """Create a selection merge instruction for if/switch statements."""
        self.emit(f"OpSelectionMerge %{merge_label.id} {selection_control}")

    def create_loop_merge(
        self, merge_label: SpirvId, continue_label: SpirvId, loop_control: str = "None"
    ):
        """Create a loop merge instruction for loops."""
        self.emit(f"OpLoopMerge %{merge_label.id} %{continue_label.id} {loop_control}")

    def current_spirv_return_type(self) -> Optional[SpirvId]:
        """Return the actual SPIR-V return type for the current function."""
        if self.current_function_name is not None:
            signature = self.resolve_function_signature(self.current_function_name)
            if signature is not None:
                return signature[0]
        return self.current_return_type

    def create_return(self):
        """Create a return instruction."""
        return_type = self.current_spirv_return_type()
        if return_type is not None and return_type.type.base_type != "void":
            function_name = self.current_function_name or "<unknown>"
            self.emit(
                f"; WARNING: Bare return in non-void function {function_name}; "
                "using default return value"
            )
            default_value = self.default_value_for_type(return_type)
            self.create_return_value(default_value)
            return

        self.emit("OpReturn")

    def create_return_value(self, value: SpirvId):
        """Create a return value instruction."""
        if self.store_entry_point_return_value(value):
            return

        if self.current_return_semantic_output is not None:
            target_type = self.pointer_pointee_type(self.current_return_semantic_output)
            if target_type is not None:
                value = self.convert_value_to_type(value, target_type)
            self.store_to_variable(self.current_return_semantic_output, value)
            self.create_return()
            return

        if self.current_return_type is not None:
            value = self.convert_value_to_type(value, self.current_return_type)
        self.emit(f"OpReturnValue %{value.id}")

    def create_unreachable(self):
        """Create an unreachable terminator for impossible fallthrough paths."""
        self.emit("OpUnreachable")

    def current_block_has_terminator(self) -> bool:
        """Return whether the current block already ends in a terminator."""
        for line in reversed(self.code_lines):
            stripped = line.strip()
            if not stripped or stripped.startswith(";"):
                continue
            if re.match(r"%\d+ = OpLabel$", stripped):
                return False
            return stripped.startswith(
                (
                    "OpBranch",
                    "OpReturn",
                    "OpKill",
                    "OpUnreachable",
                    "OpEmitMeshTasksEXT",
                    "OpIgnoreIntersectionKHR",
                    "OpTerminateRayKHR",
                )
            )
        return False

    def normalize_primitive_name(self, type_name: str) -> str:
        aliases = {
            "f32": "float",
            "f64": "double",
            "f16": "float",
            "float16": "float",
            "float16_t": "float",
            "half": "float",
            "min16float": "float",
            "i32": "int",
            "u32": "uint",
            "int64": "i64",
            "int64_t": "i64",
            "long": "i64",
            "uint64": "u64",
            "uint64_t": "u64",
            "ulong": "u64",
        }
        return aliases.get(str(type_name), str(type_name))

    def normalize_generic_vector_type(self, type_str: str) -> str:
        compact = re.sub(r"\s+", "", str(type_str))
        half_vector_match = re.fullmatch(
            r"(?:f16vec|half|min16float|float16_t)([234])", compact
        )
        if half_vector_match:
            return f"vec{half_vector_match.group(1)}"
        match = re.fullmatch(r"vec([234])<([^>]+)>", compact)
        if match:
            size, element_type = match.groups()
            element_type = self.normalize_primitive_name(element_type)
        else:
            match = re.fullmatch(r"(float|double|int|uint|bool)([234])", compact)
            if not match:
                return compact
            element_type, size = match.groups()

        prefixes = {
            "float": "vec",
            "double": "dvec",
            "int": "ivec",
            "uint": "uvec",
            "bool": "bvec",
        }
        return f"{prefixes.get(element_type, 'vec')}{size}"

    def normalize_hlsl_matrix_type(self, type_str: str) -> str:
        compact = re.sub(r"\s+", "", str(type_str))
        match = re.fullmatch(
            r"(float|double|half|min16float|float16_t)([234])x([234])",
            compact,
        )
        if not match:
            return str(type_str)

        component_type, rows, cols = match.groups()
        prefix = "dmat" if component_type == "double" else "mat"
        rows = int(rows)
        cols = int(cols)
        if rows == cols:
            return f"{prefix}{cols}"
        return f"{prefix}{cols}x{rows}"

    def vector_component_type_and_count(
        self, type_str: str
    ) -> Optional[Tuple[str, int]]:
        type_str = self.normalize_generic_vector_type(type_str)
        internal_match = re.fullmatch(r"v([234])(float|double|int|uint|bool)", type_str)
        if internal_match:
            size, component_type = internal_match.groups()
            return component_type, int(size)

        vector_prefixes = (
            ("dvec", "double"),
            ("ivec", "int"),
            ("uvec", "uint"),
            ("bvec", "bool"),
            ("vec", "float"),
        )
        for prefix, component_type in vector_prefixes:
            if type_str.startswith(prefix) and type_str[len(prefix) :].isdigit():
                return component_type, int(type_str[len(prefix) :])
        return None

    def normalize_resource_type_name(self, type_str: str) -> str:
        compact = re.sub(r"\s+", "", str(type_str))
        direct_aliases = {
            "Texture1D": "sampler1D",
            "Texture1DArray": "sampler1DArray",
            "Texture2D": "sampler2D",
            "Texture2DArray": "sampler2DArray",
            "Texture2DMS": "sampler2DMS",
            "Texture2DMSArray": "sampler2DMSArray",
            "Texture3D": "sampler3D",
            "TextureCube": "samplerCube",
            "TextureCubeArray": "samplerCubeArray",
        }
        if compact in direct_aliases:
            return direct_aliases[compact]

        resource_match = re.fullmatch(
            r"([iu]?(?:sampler|image)|texture)"
            r"(1d|2d|3d|cube|buffer)"
            r"(ms)?(array)?(shadow)?",
            compact,
            re.IGNORECASE,
        )
        if resource_match:
            prefix, dim, ms_suffix, array_suffix, shadow_suffix = (
                resource_match.groups()
            )
            dim = {
                "1d": "1D",
                "2d": "2D",
                "3d": "3D",
                "cube": "Cube",
                "buffer": "Buffer",
            }[dim.lower()]
            return (
                f"{prefix.lower()}{dim}"
                f"{'MS' if ms_suffix else ''}"
                f"{'Array' if array_suffix else ''}"
                f"{'Shadow' if shadow_suffix else ''}"
            )

        texture_match = re.fullmatch(
            r"Texture(1D|2D|3D|Cube)(MS)?(Array)?<([^>]+)>", compact
        )
        if texture_match:
            dim, ms_suffix, array_suffix, component_type = texture_match.groups()
            prefix = {
                "int": "i",
                "uint": "u",
            }.get(self.normalize_primitive_name(component_type), "")
            return f"{prefix}sampler{dim}" f"{ms_suffix or ''}{array_suffix or ''}"

        storage_match = re.fullmatch(
            r"RWTexture(1D|2D|3D|Cube)(MS)?(Array)?(?:<([^>]+)>)?", compact
        )
        if storage_match:
            dim, ms_suffix, array_suffix, component_type = storage_match.groups()
            prefix = {
                "int": "i",
                "uint": "u",
            }.get(self.normalize_primitive_name(component_type or "float"), "")
            return f"{prefix}image{dim}{ms_suffix or ''}{array_suffix or ''}"

        return compact

    def resource_type_info(self, type_str: str):
        type_str = self.normalize_resource_type_name(type_str)
        sampler_info = {
            "sampler": {"kind": "sampler"},
            "texture1D": {
                "kind": "texture",
                "component_type": "float",
                "dim": "1D",
                "depth": 0,
                "arrayed": 0,
                "multisampled": 0,
                "sampled": 1,
                "format": "Unknown",
            },
            "texture2D": {
                "kind": "texture",
                "component_type": "float",
                "dim": "2D",
                "depth": 0,
                "arrayed": 0,
                "multisampled": 0,
                "sampled": 1,
                "format": "Unknown",
            },
            "texture3D": {
                "kind": "texture",
                "component_type": "float",
                "dim": "3D",
                "depth": 0,
                "arrayed": 0,
                "multisampled": 0,
                "sampled": 1,
                "format": "Unknown",
            },
            "textureCube": {
                "kind": "texture",
                "component_type": "float",
                "dim": "Cube",
                "depth": 0,
                "arrayed": 0,
                "multisampled": 0,
                "sampled": 1,
                "format": "Unknown",
            },
            "texture2DArray": {
                "kind": "texture",
                "component_type": "float",
                "dim": "2D",
                "depth": 0,
                "arrayed": 1,
                "multisampled": 0,
                "sampled": 1,
                "format": "Unknown",
            },
            "textureCubeArray": {
                "kind": "texture",
                "component_type": "float",
                "dim": "Cube",
                "depth": 0,
                "arrayed": 1,
                "multisampled": 0,
                "sampled": 1,
                "format": "Unknown",
            },
            "sampler1D": {
                "kind": "sampled_image",
                "component_type": "float",
                "dim": "1D",
                "depth": 0,
                "arrayed": 0,
                "multisampled": 0,
                "sampled": 1,
                "format": "Unknown",
            },
            "sampler1DArray": {
                "kind": "sampled_image",
                "component_type": "float",
                "dim": "1D",
                "depth": 0,
                "arrayed": 1,
                "multisampled": 0,
                "sampled": 1,
                "format": "Unknown",
            },
            "sampler2D": {
                "kind": "sampled_image",
                "component_type": "float",
                "dim": "2D",
                "depth": 0,
                "arrayed": 0,
                "multisampled": 0,
                "sampled": 1,
                "format": "Unknown",
            },
            "sampler3D": {
                "kind": "sampled_image",
                "component_type": "float",
                "dim": "3D",
                "depth": 0,
                "arrayed": 0,
                "multisampled": 0,
                "sampled": 1,
                "format": "Unknown",
            },
            "samplerCube": {
                "kind": "sampled_image",
                "component_type": "float",
                "dim": "Cube",
                "depth": 0,
                "arrayed": 0,
                "multisampled": 0,
                "sampled": 1,
                "format": "Unknown",
            },
            "sampler2DArray": {
                "kind": "sampled_image",
                "component_type": "float",
                "dim": "2D",
                "depth": 0,
                "arrayed": 1,
                "multisampled": 0,
                "sampled": 1,
                "format": "Unknown",
            },
            "sampler2DShadow": {
                "kind": "sampled_image",
                "component_type": "float",
                "dim": "2D",
                "depth": 1,
                "arrayed": 0,
                "multisampled": 0,
                "sampled": 1,
                "format": "Unknown",
            },
            "sampler2DArrayShadow": {
                "kind": "sampled_image",
                "component_type": "float",
                "dim": "2D",
                "depth": 1,
                "arrayed": 1,
                "multisampled": 0,
                "sampled": 1,
                "format": "Unknown",
            },
            "samplerCubeShadow": {
                "kind": "sampled_image",
                "component_type": "float",
                "dim": "Cube",
                "depth": 1,
                "arrayed": 0,
                "multisampled": 0,
                "sampled": 1,
                "format": "Unknown",
            },
            "samplerCubeArray": {
                "kind": "sampled_image",
                "component_type": "float",
                "dim": "Cube",
                "depth": 0,
                "arrayed": 1,
                "multisampled": 0,
                "sampled": 1,
                "format": "Unknown",
            },
            "samplerCubeArrayShadow": {
                "kind": "sampled_image",
                "component_type": "float",
                "dim": "Cube",
                "depth": 1,
                "arrayed": 1,
                "multisampled": 0,
                "sampled": 1,
                "format": "Unknown",
            },
            "comparison_sampler": {"kind": "sampler"},
            "sampler2DMS": {
                "kind": "sampled_image",
                "component_type": "float",
                "dim": "2D",
                "depth": 0,
                "arrayed": 0,
                "multisampled": 1,
                "sampled": 1,
                "format": "Unknown",
            },
            "sampler2DMSArray": {
                "kind": "sampled_image",
                "component_type": "float",
                "dim": "2D",
                "depth": 0,
                "arrayed": 1,
                "multisampled": 1,
                "sampled": 1,
                "format": "Unknown",
            },
        }
        if type_str in sampler_info:
            return sampler_info[type_str]

        if self.is_acceleration_structure_type_name(type_str):
            return {"kind": "acceleration_structure"}

        sampled_image_match = re.fullmatch(
            r"([iu]?sampler)(1D|2D|3D|Cube|Buffer)(MS)?(Array)?(?:Shadow)?",
            type_str,
        )
        if sampled_image_match:
            prefix, dim, ms_suffix, array_suffix = sampled_image_match.groups()
            if ms_suffix and dim != "2D":
                return None
            if dim == "Buffer" and (ms_suffix or array_suffix):
                return None
            component_type = {
                "sampler": "float",
                "isampler": "int",
                "usampler": "uint",
            }[prefix]
            return {
                "kind": "sampled_image",
                "component_type": component_type,
                "dim": dim,
                "depth": 1 if type_str.endswith("Shadow") else 0,
                "arrayed": 1 if array_suffix else 0,
                "multisampled": 1 if ms_suffix else 0,
                "sampled": 1,
                "format": "Unknown",
            }

        image_match = re.fullmatch(
            r"([iu]?image)(1D|2D|3D|Cube)(MS)?(Array)?", type_str
        )
        if image_match:
            prefix, dim, ms_suffix, array_suffix = image_match.groups()
            if ms_suffix and dim != "2D":
                return None
            component_type = {
                "image": "float",
                "iimage": "int",
                "uimage": "uint",
            }[prefix]
            return {
                "kind": "storage_image",
                "component_type": component_type,
                "dim": "Cube" if dim == "Cube" else dim,
                "depth": 0,
                "arrayed": 1 if array_suffix else 0,
                "multisampled": 1 if ms_suffix else 0,
                "sampled": 2,
                "format": "Unknown",
            }

        return None

    def is_resource_type_name(self, type_str: str) -> bool:
        return self.resource_type_info(type_str) is not None

    def is_acceleration_structure_type_name(self, type_str: str) -> bool:
        compact = re.sub(r"\s+", "", str(type_str))
        return compact in {
            "accelerationStructureEXT",
            "AccelerationStructure",
            "RaytracingAccelerationStructure",
            "acceleration_structure",
        }

    def is_ray_query_type_name(self, type_str: str) -> bool:
        compact = re.sub(r"\s+", "", str(type_str))
        return compact == "RayQuery" or compact.startswith("RayQuery<")

    def structured_buffer_type_info(self, type_str: str):
        type_str = re.sub(r"\s+", "", str(type_str))
        if type_str in {"ByteAddressBuffer", "RWByteAddressBuffer"}:
            return {
                "kind": "structured_buffer",
                "buffer_kind": type_str,
                "element_type_name": "uint",
                "readonly": type_str == "ByteAddressBuffer",
                "byte_address": True,
            }

        match = re.fullmatch(
            r"(StructuredBuffer|RWStructuredBuffer|AppendStructuredBuffer|"
            r"ConsumeStructuredBuffer)<(.+)>",
            type_str,
        )
        if not match:
            return None

        buffer_kind, element_type_name = match.groups()
        return {
            "kind": "structured_buffer",
            "buffer_kind": buffer_kind,
            "element_type_name": element_type_name,
            "readonly": buffer_kind in {"StructuredBuffer", "ConsumeStructuredBuffer"},
            "default_writeonly": buffer_kind == "AppendStructuredBuffer",
            "append_consume": (
                buffer_kind in {"AppendStructuredBuffer", "ConsumeStructuredBuffer"}
            ),
        }

    def is_structured_buffer_type_name(self, type_str: str) -> bool:
        return self.structured_buffer_type_info(type_str) is not None

    def structured_buffer_declared_type_info(self, type_str: str):
        type_str = re.sub(r"\s+", "", str(type_str))
        metadata = self.structured_buffer_type_info(type_str)
        if metadata is not None:
            return metadata

        base_type = self.array_base_type_name(type_str)
        if base_type == type_str:
            return None

        return self.structured_buffer_type_info(base_type)

    def is_structured_buffer_declared_type_name(self, type_str: str) -> bool:
        return self.structured_buffer_declared_type_info(type_str) is not None

    def spirv_image_format_map(self):
        return {
            "r8": "R8",
            "r8_snorm": "R8Snorm",
            "r8i": "R8i",
            "r8ui": "R8ui",
            "r16": "R16",
            "r16_snorm": "R16Snorm",
            "r16f": "R16f",
            "r16i": "R16i",
            "r16ui": "R16ui",
            "r32f": "R32f",
            "r32i": "R32i",
            "r32ui": "R32ui",
            "rg8": "Rg8",
            "rg8_snorm": "Rg8Snorm",
            "rg8i": "Rg8i",
            "rg8ui": "Rg8ui",
            "rg16": "Rg16",
            "rg16_snorm": "Rg16Snorm",
            "rg16f": "Rg16f",
            "rg16i": "Rg16i",
            "rg16ui": "Rg16ui",
            "rg32f": "Rg32f",
            "rg32i": "Rg32i",
            "rg32ui": "Rg32ui",
            "rgba8": "Rgba8",
            "rgba8_snorm": "Rgba8Snorm",
            "rgba8i": "Rgba8i",
            "rgba8ui": "Rgba8ui",
            "rgba16": "Rgba16",
            "rgba16_snorm": "Rgba16Snorm",
            "rgba16f": "Rgba16f",
            "rgba16i": "Rgba16i",
            "rgba16ui": "Rgba16ui",
            "rgba32f": "Rgba32f",
            "rgba32i": "Rgba32i",
            "rgba32ui": "Rgba32ui",
        }

    def spirv_image_format_name(self, image_format: Optional[str]) -> Optional[str]:
        if image_format is None:
            return None
        return self.spirv_image_format_map().get(str(image_format).lower())

    def image_format_component_type(self, image_format: str) -> str:
        image_format = str(image_format).lower()
        if image_format.endswith("ui"):
            return "uint"
        if image_format.endswith("i") and not image_format.endswith("_snorm"):
            return "int"
        return "float"

    def image_format_component_count(self, image_format: Optional[str]) -> int:
        if image_format is None:
            return 4

        image_format = str(image_format).lower()
        if image_format.startswith("rgba"):
            return 4
        if image_format.startswith("rg"):
            return 2
        if image_format.startswith("r"):
            return 1
        return 4

    def resource_access_result_type(self, metadata) -> SpirvId:
        component_type = self.register_primitive_type(
            metadata.get("component_type", "float")
        )
        component_count = int(metadata.get("component_count", 4))
        if component_count <= 1:
            return component_type
        return self.register_vector_type(component_type, component_count)

    def image_coordinate_component_count(self, metadata) -> int:
        component_count = {
            "1D": 1,
            "Buffer": 1,
            "2D": 2,
            "Rect": 2,
            "3D": 3,
            "Cube": 3,
        }.get(metadata.get("dim", "2D"), 2)
        if metadata.get("arrayed"):
            component_count += 1
        return component_count

    def integer_value_component_count(self, value_id: SpirvId) -> Optional[int]:
        value_type = self.value_types.get(
            value_id.id
        ) or self.find_registered_type_by_base(value_id.type.base_type)
        type_name = (
            value_type.type.base_type
            if value_type is not None
            else value_id.type.base_type
        )
        vector_info = self.vector_component_type_and_count(type_name)
        if vector_info is not None:
            component_type_name, component_count = vector_info
            if component_type_name in {"int", "uint"}:
                return component_count
            return None

        if self.normalize_primitive_name(type_name) in {"int", "uint"}:
            return 1
        return None

    def validate_storage_image_coordinate(
        self, function_name: str, metadata, coord_id: SpirvId
    ) -> bool:
        expected_components = self.image_coordinate_component_count(metadata)
        actual_components = self.integer_value_component_count(coord_id)
        if actual_components == expected_components:
            return True

        self.emit(
            f"; WARNING: {function_name} requires a {expected_components}-component "
            f"integer coordinate for {metadata.get('type_name', 'storage image')}"
        )
        return False

    def validate_storage_image_sample(
        self, function_name: str, sample_id: SpirvId
    ) -> bool:
        if self.integer_value_component_count(sample_id) == 1:
            return True

        self.emit(
            f"; WARNING: {function_name} requires a scalar integer sample operand"
        )
        return False

    def require_storage_image_without_format_capability(
        self, metadata, capability: str
    ):
        if metadata.get("format", "Unknown") == "Unknown":
            self.require_capability(capability)

    def convert_storage_image_value_operand(
        self,
        function_name: str,
        operand_name: str,
        value_id: SpirvId,
        target_type: SpirvId,
    ) -> Optional[SpirvId]:
        converted = self.convert_value_to_type(value_id, target_type)
        if self.value_has_type(converted, target_type):
            return converted

        self.emit(
            f"; WARNING: {function_name} {operand_name} operand does not match "
            f"{target_type.type.base_type} storage image texel type"
        )
        return None

    def resource_metadata_for_value(self, value_id: SpirvId):
        metadata = self.resource_type_metadata.get(value_id.id)
        if metadata is not None:
            return metadata

        result_type = self.value_types.get(value_id.id)
        if result_type is not None:
            metadata = self.resource_type_metadata.get(result_type.id)
            if metadata is not None:
                return metadata

        return None

    def resource_metadata_for_pointer(self, pointer_id: SpirvId):
        metadata = self.resource_type_metadata.get(pointer_id.id)
        if metadata is not None:
            return metadata

        pointee_type = self.variable_value_types.get(pointer_id.id)
        if pointee_type is not None:
            metadata = self.resource_type_metadata.get(pointee_type.id)
            if metadata is not None:
                return metadata

        return self.resource_metadata_for_value(pointer_id)

    def propagate_resource_access_metadata(
        self, source_pointer: SpirvId, access_pointer: SpirvId, access_type: SpirvId
    ):
        target_metadata = self.resource_type_metadata.get(access_type.id)
        if target_metadata is None:
            return

        source_metadata = self.resource_metadata_for_pointer(source_pointer)
        if source_metadata is None or source_metadata.get(
            "kind"
        ) != target_metadata.get("kind"):
            source_metadata = target_metadata

        self.resource_type_metadata[access_pointer.id] = source_metadata

    def attribute_value_to_string(self, value):
        if value is None:
            return None
        if isinstance(value, str):
            return value
        if hasattr(value, "value"):
            return str(value.value).strip('"')
        if hasattr(value, "name") and value.name is not None:
            return str(value.name)
        return str(value)

    def literal_int_argument(self, value) -> Optional[int]:
        """Return a literal integer value when an AST argument is constant."""
        if isinstance(value, bool):
            return int(value)
        if isinstance(value, int):
            return value
        if isinstance(value, UnaryOpNode) and getattr(value, "op", None) == "-":
            operand = self.literal_int_argument(value.operand)
            return -operand if operand is not None else None
        value_text = self.attribute_value_to_string(value)
        if value_text is None:
            return None
        try:
            return int(str(value_text), 0)
        except ValueError:
            return None

    def semantic_from_node(self, node) -> Optional[str]:
        semantic = getattr(node, "semantic", None)
        if semantic is not None:
            return semantic

        ignored = {
            "binding",
            "component",
            "domain",
            "format",
            "group",
            "index",
            "input",
            "input_primitive",
            "inputprimitive",
            "invocations",
            "location",
            "local_size",
            "local_size_x",
            "local_size_y",
            "local_size_z",
            "localsizex",
            "localsizey",
            "localsizez",
            "max_primitives",
            "max_total_threads_per_threadgroup",
            "max_vertices",
            "maxprimitives",
            "maxvertices",
            "numthreads",
            "output",
            "output_control_points",
            "output_topology",
            "outputcontrolpoints",
            "outputtopology",
            "partitioning",
            "patch_constant_func",
            "patchconstantfunc",
            "readonly",
            "set",
            "space",
            "stage_entry",
            "compute",
            "fragment",
            "geometry",
            "mesh",
            "object",
            "primitive",
            "vertices",
            "workgroup_size",
            "indices",
            "primitives",
            "task",
            "tessellation_control",
            "tessellation_evaluation",
            "vertex",
            "writeonly",
        }
        for attr in getattr(node, "attributes", []) or []:
            attr_name = getattr(attr, "name", None)
            if not attr_name:
                continue
            normalized = self.normalized_metadata_name(attr_name)
            if normalized in ignored:
                continue
            return str(attr_name)
        return None

    def spirv_semantic_output_kind(self, semantic) -> Optional[str]:
        if semantic is None:
            return None

        semantic_name = self.spirv_output_semantic_alias(str(semantic))
        lower_name = semantic_name.lower()
        upper_name = semantic_name.upper()
        input_only_sources = {
            "gl_baseinstance",
            "gl_basevertex",
            "gl_barycoordext",
            "gl_barycoordnoperspext",
            "gl_drawid",
            "gl_fragcoord",
            "gl_frontfacing",
            "gl_globalinvocationid",
            "gl_instanceid",
            "gl_invocationid",
            "gl_localinvocationid",
            "gl_localinvocationindex",
            "gl_pointcoord",
            "gl_sampleid",
            "gl_samplemaskin",
            "gl_tesscoord",
            "gl_vertexid",
            "gl_workgroupid",
        }
        if lower_name in input_only_sources or upper_name in {
            "SV_INSTANCEID",
            "SV_ISFRONTFACE",
            "SV_VERTEXID",
        }:
            return "input_only"

        if lower_name == "gl_position" or upper_name == "SV_POSITION":
            return "position"
        if lower_name == "gl_fragdepth" or upper_name == "SV_DEPTH":
            return "depth"
        if lower_name == "gl_samplemask" or upper_name == "SV_COVERAGE":
            return "sample_mask"
        if lower_name == "gl_fragstencilrefext" or upper_name == "SV_STENCILREF":
            return "stencil_ref"
        if lower_name.startswith("gl_fragcolor"):
            suffix = lower_name[len("gl_fragcolor") :]
            if suffix == "" or suffix.isdigit():
                return "color"
        if lower_name == "gl_fragdata":
            return "color"
        if upper_name.startswith("SV_TARGET"):
            suffix = upper_name[len("SV_TARGET") :]
            if suffix == "" or suffix.isdigit():
                return "color"
        if upper_name.startswith("COLOR"):
            suffix = upper_name[len("COLOR") :]
            if suffix == "" or suffix.isdigit():
                return "color"
        return None

    def spirv_return_semantic_builtin_info(
        self, semantic, return_type: SpirvId
    ) -> Optional[Tuple[str, str, SpirvId]]:
        kind = self.spirv_semantic_output_kind(semantic)
        if kind == "position":
            return ("gl_Position", "Position", return_type)
        if kind == "depth":
            return ("gl_FragDepth", "FragDepth", return_type)
        if kind == "sample_mask":
            return ("gl_SampleMask", "SampleMask", self.map_crossgl_type("int[1]"))
        if kind == "stencil_ref":
            return ("gl_FragStencilRefEXT", "FragStencilRefEXT", return_type)
        return None

    def spirv_color_semantic_location(self, semantic, node=None) -> Optional[int]:
        if semantic is None:
            return None

        semantic_name = str(semantic)
        lower_name = semantic_name.lower()
        upper_name = semantic_name.upper()

        if lower_name == "gl_fragdata" and node is not None:
            for attr in getattr(node, "attributes", []) or []:
                if str(getattr(attr, "name", "")).lower() != "gl_fragdata":
                    continue
                arguments = getattr(attr, "arguments", None) or getattr(
                    attr, "args", []
                )
                if arguments:
                    return self.literal_int_argument(arguments[0])
            return 0

        for prefix in ("gl_fragcolor",):
            if lower_name.startswith(prefix):
                suffix = lower_name[len(prefix) :]
                if suffix == "":
                    return 0
                if suffix.isdigit():
                    return int(suffix)

        for prefix in ("SV_TARGET", "COLOR"):
            if upper_name.startswith(prefix):
                suffix = upper_name[len(prefix) :]
                if suffix == "":
                    return 0
                if suffix.isdigit():
                    return int(suffix)

        return None

    def is_spirv_float_vector_width(self, type_name, width: int) -> bool:
        vector_info = self.vector_component_type_and_count(
            self.type_name_string(type_name)
        )
        return vector_info == ("float", width)

    def is_spirv_float_scalar_type(self, type_name) -> bool:
        return (
            self.normalize_primitive_name(self.type_name_string(type_name)) == "float"
        )

    def is_spirv_integer_scalar_type(self, type_name) -> bool:
        return self.normalize_primitive_name(self.type_name_string(type_name)) in {
            "int",
            "uint",
        }

    def is_spirv_sample_mask_type(self, type_name) -> bool:
        type_text = self.type_name_string(type_name)
        if self.normalize_primitive_name(type_text) in {"int", "uint"}:
            return True

        array_info = self.split_outer_array_type(type_text)
        if array_info is None:
            return False
        element_type, _ = array_info
        return self.normalize_primitive_name(element_type) in {"int", "uint"}

    def validate_spirv_builtin_semantic_type(self, semantic, type_name, context):
        kind = self.spirv_semantic_output_kind(semantic)
        if kind is None or kind == "input_only":
            return

        if kind in {"position", "color"}:
            if self.is_spirv_float_vector_width(type_name, 4):
                return
            raise ValueError(
                f"Unsupported {semantic} {context} for SPIR-V codegen; "
                "expected vec4-compatible type"
            )

        if kind == "depth" and not self.is_spirv_float_scalar_type(type_name):
            raise ValueError(
                f"Unsupported {semantic} {context} for SPIR-V codegen; "
                "expected float type"
            )

        if kind == "sample_mask" and not self.is_spirv_sample_mask_type(type_name):
            raise ValueError(
                f"Unsupported {semantic} {context} for SPIR-V codegen; "
                "expected int-compatible sample-mask type"
            )

        if kind == "stencil_ref" and not self.is_spirv_integer_scalar_type(type_name):
            raise ValueError(
                f"Unsupported {semantic} {context} for SPIR-V codegen; "
                "expected integer scalar type"
            )

    def validate_spirv_output_semantic_stage(
        self, execution_model: Optional[str], semantic, context
    ):
        kind = self.spirv_semantic_output_kind(semantic)
        if kind is None:
            return
        if kind == "input_only":
            raise ValueError(
                f"Unsupported {semantic} {context} for SPIR-V codegen; "
                "input-only builtin semantics cannot be used as outputs"
            )
        if execution_model is None:
            return

        allowed_models = {
            "position": {"TessellationEvaluation", "Vertex"},
            "color": {"Fragment"},
            "depth": {"Fragment"},
            "sample_mask": {"Fragment"},
            "stencil_ref": {"Fragment"},
        }[kind]
        if execution_model in allowed_models:
            return

        stage_name = {
            "GLCompute": "compute",
            "MeshEXT": "mesh",
            "TaskEXT": "task",
        }.get(execution_model, execution_model.lower())
        allowed = ", ".join(
            {
                "GLCompute": "compute",
                "MeshEXT": "mesh",
                "TaskEXT": "task",
            }.get(model, model.lower())
            for model in sorted(allowed_models)
        )
        raise ValueError(
            f"Unsupported {semantic} {context} for SPIR-V {stage_name} stage; "
            f"valid stage is {allowed}"
        )

    def validate_spirv_return_semantic(
        self, execution_model: Optional[str], return_type, semantic
    ):
        if semantic is None:
            return
        if self.type_name_string(return_type) == "void":
            raise ValueError(
                f"Unsupported {semantic} return semantic for SPIR-V codegen; "
                "void return type"
            )
        self.validate_spirv_output_semantic_stage(
            execution_model, semantic, "return semantic"
        )
        self.validate_spirv_builtin_semantic_type(
            semantic, return_type, "return semantic"
        )

    def direct_return_output_name(self, function_name: str, semantic) -> str:
        normalized = re.sub(r"[^0-9A-Za-z_]+", "_", str(semantic)).strip("_")
        return f"{function_name}_return_{normalized or 'semantic'}"

    def register_direct_return_output(
        self, function_node, semantic, return_type: SpirvId
    ) -> Optional[SpirvId]:
        builtin_info = self.spirv_return_semantic_builtin_info(semantic, return_type)
        if builtin_info is not None:
            name, builtin_name, builtin_type = builtin_info
            cache_key = f"__return_semantic_builtin::{name}"
            if cache_key in self.global_variables:
                variable = self.global_variables[cache_key]
                self.mark_fragment_depth_replacing_if_needed(builtin_name, "Output")
                self.mark_function_interface_variable(variable)
                return variable

            variable = self.register_builtin_variable(
                name, builtin_type, builtin_name, "Output"
            )
            self.global_variables[cache_key] = variable
            self.mark_function_interface_variable(variable)
            return variable

        output_node = VariableNode(
            self.direct_return_output_name(function_node.name, semantic),
            function_node.return_type,
            attributes=list(getattr(function_node, "attributes", []) or []),
        )
        location = self.global_interface_location(
            output_node,
            "Output",
            preferred_location=self.spirv_color_semantic_location(
                semantic, function_node
            ),
        )
        variable = self.create_variable(return_type, "Output", output_node.name)
        self.decorations.append(f"OpDecorate %{variable.id} Location {location}")
        self.decorate_global_interface_variable(output_node, variable)
        self.mark_function_interface_variable(variable)
        return variable

    def initialize_entry_point_parameters(
        self,
        runtime_parameters: List[Tuple[VariableNode, SpirvId, SpirvId]],
        execution_model: Optional[str],
    ):
        for param, _param_type, param_value_type in runtime_parameters:
            param_name = getattr(param, "name", None) or "param"
            if self.is_storage_resource_parameter(param):
                variable = self.process_entry_point_storage_buffer_parameter(param)
                self.local_variables[param_name] = variable
                continue

            if self.is_bound_uniform_value_parameter(param, execution_model):
                initial_value = self.entry_point_uniform_parameter_value(
                    param, param_value_type
                )
            else:
                initial_value = self.entry_point_parameter_value(
                    param, param_value_type, execution_model
                )
            local_variable = self.create_variable(
                param_value_type, "Function", param_name
            )
            self.local_variables[param_name] = local_variable
            if initial_value is not None:
                self.store_to_variable(local_variable, initial_value)
            self.register_declared_resource_metadata(
                param, local_variable, param_value_type
            )

    def resource_parameter_qualifier_names(self, param) -> set:
        return {
            str(qualifier).lower()
            for qualifier in getattr(param, "resource_qualifiers", []) or []
        }

    def is_bound_uniform_value_parameter(
        self, param, execution_model: Optional[str]
    ) -> bool:
        if execution_model != "GLCompute":
            return False
        if self.explicit_descriptor_binding(param) is None:
            return False
        qualifiers = self.parameter_qualifier_names(param)
        qualifiers.update(self.resource_parameter_qualifier_names(param))
        if not qualifiers & {"uniform", "constant"}:
            return False

        param_type = getattr(param, "param_type", getattr(param, "vtype", None))
        type_name = self.type_name_from_value(param_type)
        type_name = self.normalize_reference_type_name(type_name)
        if type_name is None:
            return False
        return not (
            self.is_resource_type_name(type_name)
            or self.is_structured_buffer_declared_type_name(type_name)
            or self.pointer_pointee_type_name_from_string(type_name) is not None
        )

    def is_storage_resource_parameter(self, param) -> bool:
        param_type = getattr(param, "param_type", getattr(param, "vtype", None))
        if param_type is None:
            return False
        type_name = self.type_name_from_value(param_type)

        qualifiers = self.resource_parameter_qualifier_names(param)
        parameter_qualifiers = self.parameter_qualifier_names(param)
        if self.pointer_pointee_type_name_from_string(type_name) is not None:
            return bool(
                qualifiers & {"storage", "buffer"}
                or parameter_qualifiers & {"device", "constant", "storage", "buffer"}
            )

        if "storage" not in qualifiers:
            return False
        param_type_id = self.map_crossgl_type(param_type)
        return self.array_type_info_from_type(param_type_id) is not None

    def process_entry_point_storage_buffer_parameter(self, param) -> SpirvId:
        variable = VariableNode(
            getattr(param, "name", "param"),
            getattr(param, "param_type", getattr(param, "vtype", None)),
            attributes=list(getattr(param, "attributes", []) or []),
            qualifiers=list(getattr(param, "qualifiers", []) or []),
        )
        variable.resource_qualifiers = list(
            getattr(param, "resource_qualifiers", []) or []
        )
        type_name = self.type_name_from_value(variable.var_type)
        return self.process_glsl_buffer_block_declaration(variable, type_name)

    def entry_point_uniform_parameter_value(
        self, param, param_value_type: SpirvId
    ) -> SpirvId:
        param_name = getattr(param, "name", None) or "param"
        member_type = self.storage_layout_type(param_value_type, "std140")
        block_name = f"{self.current_function_name}_{param_name}UniformBlock"
        block_type = self.register_struct_type(block_name, [(member_type, param_name)])
        self.decorate_cbuffer_type(block_type, [(member_type, param_name)])

        var_id = self.create_variable(block_type, "Uniform", f"{param_name}Uniform")
        descriptor_set, binding = self.resource_descriptor_slot(param)
        self.decorations.append(
            f"OpDecorate %{var_id.id} DescriptorSet {descriptor_set}"
        )
        self.decorations.append(f"OpDecorate %{var_id.id} Binding {binding}")
        self.uniform_buffers.append(var_id)

        int_type = self.register_primitive_type("int")
        index = self.register_constant(0, int_type)
        ptr_type = self.register_pointer_type(member_type, "Uniform")
        member_pointer = self.access_chain(var_id, [index], ptr_type)
        self.variable_value_types[member_pointer.id] = member_type
        loaded = self.get_variable_value(member_pointer)
        return self.convert_value_to_type(loaded, param_value_type)

    def entry_point_parameter_value(
        self, param, param_value_type: SpirvId, execution_model: Optional[str]
    ) -> Optional[SpirvId]:
        members = self.current_struct_members.get(param_value_type.type.base_type)
        if members:
            components = []
            metadata = self.struct_member_metadata.get(
                param_value_type.type.base_type, {}
            )
            for member_index, (member_type, member_name) in enumerate(members):
                member_info = metadata.get(member_name, {})
                member_node = member_info.get("node")
                variable = self.register_entry_point_interface_variable(
                    execution_model,
                    "Input",
                    self.entry_point_interface_name(
                        execution_model,
                        "input",
                        getattr(param, "name", "param"),
                        member_name,
                    ),
                    member_type,
                    member_node,
                    member_name=member_name,
                    member_index=member_index,
                )
                components.append(
                    self.load_entry_point_interface_value(variable, member_type)
                )
            return self.composite_construct(param_value_type, components)

        variable = self.register_entry_point_interface_variable(
            execution_model,
            "Input",
            self.entry_point_interface_name(
                execution_model, "input", getattr(param, "name", "param")
            ),
            param_value_type,
            param,
        )
        return self.load_entry_point_interface_value(variable, param_value_type)

    def register_entry_point_return_outputs(
        self,
        function_node,
        return_type: SpirvId,
        execution_model: Optional[str],
    ):
        members = self.current_struct_members.get(return_type.type.base_type)
        if members:
            outputs = []
            metadata = self.struct_member_metadata.get(return_type.type.base_type, {})
            for member_index, (member_type, member_name) in enumerate(members):
                member_info = metadata.get(member_name, {})
                member_node = member_info.get("node")
                variable = self.register_entry_point_interface_variable(
                    execution_model,
                    "Output",
                    self.entry_point_interface_name(
                        execution_model,
                        "output",
                        getattr(function_node, "name", "main"),
                        member_name,
                    ),
                    member_type,
                    member_node,
                    member_name=member_name,
                    member_index=member_index,
                )
                outputs.append((member_index, variable, member_type))
            return {"kind": "struct", "type": return_type, "outputs": outputs}

        variable = self.register_entry_point_interface_variable(
            execution_model,
            "Output",
            self.entry_point_interface_name(
                execution_model, "output", getattr(function_node, "name", "main")
            ),
            return_type,
            function_node,
        )
        return {"kind": "value", "type": return_type, "variable": variable}

    def entry_point_interface_name(
        self,
        execution_model: Optional[str],
        direction: str,
        base_name: str,
        member_name: Optional[str] = None,
    ) -> str:
        stage = (execution_model or "stage").lower()
        parts = ["_CrossGL", stage, direction, str(base_name)]
        if member_name is not None:
            parts.append(str(member_name))
        return "_".join(part.strip("_") for part in parts if part)

    def register_entry_point_interface_variable(
        self,
        execution_model: Optional[str],
        storage_class: str,
        name: str,
        type_id: SpirvId,
        node,
        member_name: Optional[str] = None,
        member_index: Optional[int] = None,
    ) -> SpirvId:
        semantic = self.entry_point_interface_semantic(
            execution_model, storage_class, node, type_id, member_name
        )
        builtin = self.entry_point_builtin_variable(semantic, storage_class)
        if builtin is not None:
            return builtin

        interface_node = node
        if interface_node is None:
            interface_node = VariableNode(name, type_id.type.base_type)
        self.validate_user_defined_interface_type(
            type_id, storage_class, name, interface_node
        )
        preferred_location = self.entry_point_interface_location(
            semantic, storage_class, node, member_index
        )
        location = self.global_interface_location(
            interface_node, storage_class, preferred_location=preferred_location
        )
        variable = self.create_variable(type_id, storage_class, name)
        self.decorations.append(f"OpDecorate %{variable.id} Location {location}")
        if (
            storage_class == "Input"
            and execution_model == "Fragment"
            and self.interface_type_requires_flat(type_id)
        ):
            self.decorations.append(f"OpDecorate %{variable.id} Flat")
        self.decorate_global_interface_variable(interface_node, variable)
        if storage_class == "Input":
            self.inputs.append(variable)
        else:
            self.outputs.append(variable)
        self.mark_function_interface_variable(variable)
        return variable

    def entry_point_builtin_variable(
        self, semantic: Optional[str], storage_class: str
    ) -> Optional[SpirvId]:
        if semantic is None:
            return None
        if storage_class == "Input":
            return self.ensure_builtin_variable(semantic)
        if storage_class == "Output":
            return self.ensure_stage_builtin(semantic)
        return None

    def entry_point_interface_semantic(
        self,
        execution_model: Optional[str],
        storage_class: str,
        node,
        type_id: SpirvId,
        member_name: Optional[str],
    ) -> Optional[str]:
        semantic = self.semantic_from_node(node)
        if semantic is not None:
            return self.spirv_interface_semantic_alias(
                str(semantic), execution_model, storage_class
            )

        normalized_member = str(member_name or getattr(node, "name", "")).lower()
        if (
            storage_class == "Output"
            and execution_model in {"Vertex", "TessellationEvaluation"}
            and normalized_member in {"position", "pos"}
            and self.vector_component_type_and_count(type_id.type.base_type)
            == ("float", 4)
        ):
            return "gl_Position"
        if (
            storage_class == "Output"
            and execution_model == "Fragment"
            and normalized_member in {"color", "colour", "fragcolor"}
            and self.vector_component_type_and_count(type_id.type.base_type)
            == ("float", 4)
        ):
            return "gl_FragColor"
        if (
            storage_class == "Input"
            and execution_model == "Fragment"
            and normalized_member in {"fragcoord", "frag_coord"}
            and self.vector_component_type_and_count(type_id.type.base_type)
            == ("float", 4)
        ):
            return "gl_FragCoord"
        return None

    def entry_point_interface_location(
        self, semantic, storage_class: str, node, member_index: Optional[int]
    ) -> Optional[int]:
        if semantic is None:
            return None
        if storage_class == "Output":
            color_location = self.spirv_color_semantic_location(semantic, node)
            if color_location is not None:
                return color_location
        semantic_location = self.mesh_output_semantic_location(semantic)
        if semantic_location is not None:
            return semantic_location
        return member_index

    def interface_type_requires_flat(self, type_id: SpirvId) -> bool:
        component_type = self.scalar_or_vector_component_type(type_id.type)
        return self.normalize_primitive_name(component_type) in {"int", "uint", "bool"}

    def interface_type_contains_bool(
        self, type_id: Optional[SpirvId], seen=None
    ) -> bool:
        if type_id is None:
            return False
        if seen is None:
            seen = set()
        if type_id.id in seen:
            return False
        seen.add(type_id.id)

        type_name = type_id.type.base_type
        if self.normalize_primitive_name(type_name) == "bool":
            return True

        vector_info = self.vector_component_type_and_count(type_name)
        if vector_info is not None:
            component_type, _ = vector_info
            return self.normalize_primitive_name(component_type) == "bool"

        array_info = self.array_type_info_from_type(type_id)
        if array_info is not None:
            element_type, _ = array_info
            return self.interface_type_contains_bool(element_type, seen)

        members = self.current_struct_members.get(type_name)
        if members is not None:
            return any(
                self.interface_type_contains_bool(member_type, seen)
                for member_type, _ in members
            )

        return False

    def interface_source_declaration_text(
        self,
        source_node,
        fallback_name: Optional[str],
        fallback_type: Optional[SpirvId],
    ) -> Optional[str]:
        if isinstance(source_node, AssignmentNode):
            source_node = getattr(source_node, "left", None)

        declaration_name = getattr(source_node, "name", None) or fallback_name
        type_value = (
            getattr(source_node, "var_type", None)
            or getattr(source_node, "param_type", None)
            or getattr(source_node, "const_type", None)
        )
        type_name = self.type_name_from_value(type_value)
        if type_name is None and fallback_type is not None:
            type_name = fallback_type.type.base_type

        qualifiers = [
            str(qualifier)
            for qualifier in getattr(source_node, "qualifiers", None) or []
            if qualifier
        ]
        parts = [*qualifiers]
        if type_name:
            parts.append(str(type_name))
        if declaration_name:
            parts.append(str(declaration_name))
        declaration = " ".join(parts).strip()
        return declaration or None

    def validate_user_defined_interface_type(
        self,
        type_id: SpirvId,
        storage_class: str,
        name: Optional[str],
        source_node=None,
    ):
        if storage_class not in {"Input", "Output"}:
            return
        if not self.interface_type_contains_bool(type_id):
            return

        direction = storage_class.lower()
        variable_name = f" '{name}'" if name else ""
        declaration = self.interface_source_declaration_text(source_node, name, type_id)
        declaration_suffix = (
            f" from source declaration '{declaration}'" if declaration else ""
        )
        source_location = getattr(source_node, "source_location", None)
        location_label = self.format_source_location(source_location)
        location_suffix = f" at {location_label}" if location_label else ""
        raise UnsupportedSPIRVFeatureError(
            "spirv.bool_interface",
            f"Unsupported user-defined SPIR-V {direction} variable{variable_name}"
            f"{declaration_suffix}{location_suffix}; Vulkan requires "
            "Input/Output OpTypeBool interfaces to use BuiltIn. Lower boolean "
            "input-like values as int or uint uniforms or specialization "
            "constants before SPIR-V generation.",
            missing_capabilities=("spirv.bool_interface_lowering",),
            source_location=source_location,
        )

    def store_entry_point_return_value(self, value: SpirvId) -> bool:
        outputs = self.current_entry_point_return_outputs
        if outputs is None:
            return False
        if outputs["kind"] == "value":
            target = outputs["variable"]
            target_type = self.pointer_pointee_type(target) or outputs["type"]
            self.store_to_variable(
                target, self.convert_value_to_type(value, target_type)
            )
            self.create_return()
            return True

        for member_index, target, member_type in outputs["outputs"]:
            member_value = self.composite_extract(value, member_type, member_index)
            self.store_to_variable(target, member_value)
        self.create_return()
        return True

    def max_optional_int(self, first: Optional[int], second: Optional[int]):
        if first is None:
            return second
        if second is None:
            return first
        return max(first, second)

    def explicit_image_format(self, node) -> Optional[str]:
        if not hasattr(node, "attributes"):
            return None

        supported_formats = self.spirv_image_format_map()
        for attr in node.attributes:
            attr_name = getattr(attr, "name", None)
            if not attr_name:
                continue

            attr_name = str(attr_name).lower()
            if attr_name in supported_formats:
                return attr_name

            if attr_name != "format":
                continue

            arguments = getattr(attr, "arguments", []) or []
            if not arguments:
                continue

            format_name = self.attribute_value_to_string(arguments[0])
            if format_name is None:
                continue

            format_name = str(format_name).lower()
            if format_name in supported_formats:
                return format_name

        return None

    def map_resource_type_with_format(self, type_name, node=None) -> SpirvId:
        type_str = self.type_name_from_value(type_name)
        if type_str is None:
            type_str = "None"

        type_str = self.normalize_generic_vector_type(type_str)
        type_str = self.normalize_hlsl_matrix_type(type_str)
        explicit_format = self.explicit_image_format(node) if node is not None else None

        array_type = self.split_outer_array_type(type_str)
        if array_type is not None:
            base_type = self.array_base_type_name(type_str)
            element_type_name, size = array_type
            if self.is_resource_type_name(base_type):
                element_type = self.map_resource_type_with_format(
                    element_type_name, node
                )
                return self.register_array_type(element_type, size)

        if self.is_resource_type_name(type_str):
            return self.register_resource_type(type_str, explicit_format)

        return self.map_crossgl_type(type_str)

    def spirv_struct_member_storage_type_name(
        self, type_name, allow_runtime_array: bool = False
    ) -> str:
        type_str = self.type_name_from_value(type_name)
        if type_str is None:
            return type_str

        type_str = self.normalize_generic_vector_type(type_str)
        base_type_name = self.array_base_type_name(type_str)
        if (
            self.is_resource_type_name(base_type_name)
            or self.is_acceleration_structure_type_name(base_type_name)
            or self.is_ray_query_type_name(base_type_name)
        ):
            return f"uint{type_str[len(base_type_name):]}"
        if "[]" in type_str and not allow_runtime_array:
            return type_str.replace("[]", "[1]")

        return type_str

    def storage_buffer_parameter_type_name(self, param) -> Optional[str]:
        param_type = getattr(param, "param_type", getattr(param, "vtype", None))
        if param_type is None:
            return None

        type_name = self.type_name_from_value(param_type)
        if self.pointer_pointee_type_name_from_string(type_name) is not None:
            return type_name
        if self.is_structured_buffer_declared_type_name(type_name):
            return type_name
        if self.has_attribute(param, "glsl_buffer_block"):
            return type_name
        return None

    def storage_buffer_expression_type_name(self, expr) -> Optional[str]:
        if isinstance(expr, ArrayAccessNode):
            base_type = self.storage_buffer_expression_type_name(
                getattr(expr, "array", getattr(expr, "array_expr", None))
            )
            if base_type is not None:
                array_type = self.split_outer_array_type(base_type)
                if array_type is not None:
                    return array_type[0]

        pointer = self.variable_pointer_from_expression(expr)
        if pointer is None:
            return None
        metadata = self.structured_buffer_metadata_for_pointer(pointer)
        if metadata is None:
            return None
        return metadata.get("declared_type_name")

    def storage_buffer_parameter_type_is_compatible(
        self, declared_type: str, actual_type: str
    ) -> bool:
        declared_type = re.sub(r"\s+", "", str(declared_type))
        actual_type = re.sub(r"\s+", "", str(actual_type))
        declared_info = self.structured_buffer_type_info(
            self.array_base_type_name(declared_type)
        )
        actual_info = self.structured_buffer_type_info(
            self.array_base_type_name(actual_type)
        )
        if declared_info is None or actual_info is None:
            return True

        declared_dimensions = self.array_dimensions(declared_type) or []
        actual_dimensions = self.array_dimensions(actual_type) or []
        if len(declared_dimensions) != len(actual_dimensions):
            return False
        for declared_dimension, actual_dimension in zip(
            declared_dimensions, actual_dimensions
        ):
            if declared_dimension and declared_dimension != actual_dimension:
                return False

        if bool(declared_info.get("byte_address")) != bool(
            actual_info.get("byte_address")
        ):
            return False
        if declared_info.get("byte_address"):
            return True

        if declared_info.get("element_type_name") != actual_info.get(
            "element_type_name"
        ):
            return False

        declared_kind = declared_info.get("buffer_kind")
        actual_kind = actual_info.get("buffer_kind")
        if declared_kind == "StructuredBuffer":
            return actual_kind in {"StructuredBuffer", "RWStructuredBuffer"}
        if declared_kind == "RWStructuredBuffer":
            return actual_kind in {"StructuredBuffer", "RWStructuredBuffer"}
        return declared_kind == actual_kind

    def function_storage_buffer_parameters(self, function_node) -> set:
        return {
            getattr(param, "name", None)
            for param in getattr(
                function_node, "parameters", getattr(function_node, "params", [])
            )
            if self.storage_buffer_parameter_type_name(param) is not None
        } - {None}

    def function_has_storage_buffer_parameters(self, function_node) -> bool:
        return bool(self.function_storage_buffer_parameters(function_node))

    def format_array_size(self, size):
        if size is None:
            return None
        literal_size = evaluate_literal_int_expression(size, self.literal_int_constants)
        if literal_size is not None:
            return literal_size
        if hasattr(size, "value"):
            return size.value
        return size

    def find_registered_type_by_base(self, base_type: str) -> Optional[SpirvId]:
        for type_dict in [
            self.primitive_types,
            self.vector_types,
            self.matrix_types,
            self.struct_types,
            self.array_types,
            self.layout_struct_types,
            self.layout_array_types,
            self.resource_types,
            self.resource_image_types,
            self.ray_query_types,
        ]:
            for type_id in type_dict.values():
                if type_id.type.base_type == base_type:
                    return type_id
        return None

    def find_registered_type_by_id(self, id_value: int) -> Optional[SpirvId]:
        for type_dict in [
            self.primitive_types,
            self.vector_types,
            self.matrix_types,
            self.struct_types,
            self.array_types,
            self.layout_struct_types,
            self.layout_array_types,
            self.resource_types,
            self.resource_image_types,
            self.ray_query_types,
        ]:
            for type_id in type_dict.values():
                if type_id.id == id_value:
                    return type_id
        return None

    def ensure_registered_type(self, type_ref: Union[SpirvId, SpirvType]) -> SpirvId:
        if isinstance(type_ref, SpirvId):
            return type_ref

        registered_type = self.find_registered_type_by_base(type_ref.base_type)
        if registered_type is not None:
            return registered_type

        return self.map_crossgl_type(type_ref.base_type)

    def scalar_or_vector_component_type(self, spirv_type: SpirvType) -> str:
        vector_info = self.vector_component_type_and_count(spirv_type.base_type)
        if vector_info is not None:
            return vector_info[0]
        return spirv_type.base_type

    def map_crossgl_type(self, type_name) -> SpirvId:
        """Map a CrossGL type name to a SPIR-V type ID."""
        type_str = self.type_name_from_value(type_name)
        if type_str is None:
            type_str = "None"

        type_str = self.normalize_reference_type_name(type_str)
        type_str = self.normalize_generic_vector_type(type_str)
        type_str = self.normalize_hlsl_matrix_type(type_str)
        if type_str.startswith("&"):
            return self.map_crossgl_type(type_str[1:].strip())

        array_type = self.split_outer_array_type(type_str)
        if array_type is not None:
            element_type_name, size = array_type
            element_type = self.map_crossgl_type(element_type_name)
            return self.register_array_type(element_type, size)

        primitive_type = self.normalize_primitive_name(type_str)
        if primitive_type in {
            "float",
            "double",
            "int",
            "uint",
            "i64",
            "u64",
            "bool",
            "void",
        }:
            return self.register_primitive_type(primitive_type)

        vector_info = self.vector_component_type_and_count(type_str)
        if vector_info:
            component_type, size = vector_info
            component_type_id = self.register_primitive_type(component_type)
            return self.register_vector_type(component_type_id, size)

        matrix_match = re.fullmatch(r"(d)?mat([234])(?:x([234]))?", type_str)
        if matrix_match:
            is_double, cols, rows = matrix_match.groups()
            component_type = self.register_primitive_type(
                "double" if is_double else "float"
            )
            row_count = int(rows or cols)
            col_count = int(cols)
            col_type = self.register_vector_type(component_type, row_count)
            return self.register_matrix_type(col_type, col_count)

        patch_info = self.patch_type_info_from_name(type_str)
        if patch_info is not None:
            element_type = self.map_crossgl_type(patch_info["element_type_name"])
            return self.register_array_type(element_type, patch_info["control_points"])

        if type_str == "str":
            return self.register_primitive_type("int")

        atomic_element_type_name = self.atomic_element_type_name(type_str)
        if atomic_element_type_name is not None:
            return self.map_crossgl_type(atomic_element_type_name)

        generic_enum_type = self.generic_enum_specialization_for_type(type_str)
        if generic_enum_type is not None:
            return generic_enum_type

        generic_base_name, generic_args = generic_type_parts(type_str)
        if generic_args:
            declared_generic_struct_type = self.ensure_declared_struct_type(
                generic_base_name
            )
            if declared_generic_struct_type is not None:
                return declared_generic_struct_type
        if generic_args and generic_base_name in self.struct_types:
            return self.struct_types[generic_base_name]

        declared_struct_type = self.ensure_declared_struct_type(type_str)
        if declared_struct_type is not None:
            return declared_struct_type

        if type_str in self.enum_struct_type_names:
            enum_struct_type = self.ensure_enum_struct_type_registered(type_str)
            if enum_struct_type is not None:
                return enum_struct_type

        if type_str in self.enum_type_names:
            return self.register_primitive_type("int")

        option_payload = self.lowerable_option_payload_type_name(type_str)
        if option_payload is not None:
            return self.map_crossgl_type(option_payload)

        registered_type = self.find_registered_type_by_base(type_str)
        if registered_type:
            return registered_type

        if self.is_ray_query_type_name(type_str):
            return self.register_ray_query_type(type_str)

        if self.is_resource_type_name(type_str):
            return self.register_resource_type(type_str)

        if type_str in self.struct_types:
            return self.struct_types[type_str]
        else:
            self.emit(f"; WARNING: Unknown type {type_str}, using float as default")
            return self.register_primitive_type("float")

    def ensure_declared_struct_type(self, type_name: str) -> Optional[SpirvId]:
        if type_name in self.struct_types:
            return self.struct_types[type_name]

        struct_node = self.struct_declarations.get(type_name)
        if struct_node is None:
            return None
        if type_name in self.struct_registration_stack:
            return None

        self.struct_registration_stack.add(type_name)
        try:
            return self.process_crossgl_struct(struct_node)
        finally:
            self.struct_registration_stack.remove(type_name)

    def ensure_enum_struct_type_registered(self, enum_name: str) -> Optional[SpirvId]:
        if enum_name in self.struct_types:
            return self.struct_types[enum_name]
        if enum_name not in self.enum_struct_type_names:
            return None

        if (
            enum_name not in self.enum_declarations
            or enum_name in self.enum_struct_registration_stack
        ):
            return None

        self.enum_struct_registration_stack.add(enum_name)
        try:
            int_type = self.register_primitive_type("int")
            members = [(int_type, "variant")]
            for field_name, field_type in self.enum_struct_fields.get(enum_name, []):
                members.append((self.map_crossgl_type(field_type), field_name))
            return self.register_struct_type(enum_name, members)
        finally:
            self.enum_struct_registration_stack.remove(enum_name)

    def convert_type_node_to_string(self, type_node) -> str:
        """Convert new AST TypeNode to string representation."""
        if isinstance(type_node, FunctionCallNode):
            callee = self.convert_type_node_to_string(type_node.function)
            args = ", ".join(
                self.convert_type_node_to_string(arg)
                for arg in getattr(type_node, "arguments", []) or []
            )
            return f"{callee}({args})"
        if isinstance(type_node, IdentifierNode):
            return type_node.name
        if isinstance(type_node, LiteralNode):
            return str(type_node.value)
        if type_node.__class__.__name__ == "ArrayType":
            element_type = self.convert_type_node_to_string(type_node.element_type)
            size = self.format_array_size(type_node.size)
            return (
                f"{element_type}[{size}]" if size is not None else f"{element_type}[]"
            )
        if hasattr(type_node, "pointee_type"):
            pointee_type = self.convert_type_node_to_string(type_node.pointee_type)
            return f"{pointee_type}*"
        if hasattr(type_node, "referenced_type"):
            referenced_type = self.convert_type_node_to_string(
                type_node.referenced_type
            )
            return f"&{referenced_type}"
        if hasattr(type_node, "return_type") and hasattr(type_node, "param_types"):
            return_type = self.convert_type_node_to_string(type_node.return_type)
            param_types = ", ".join(
                self.convert_type_node_to_string(param_type)
                for param_type in getattr(type_node, "param_types", []) or []
            )
            return f"fn({param_types}) -> {return_type}"
        if hasattr(type_node, "name") and type_node.name is not None:
            generic_args = getattr(type_node, "generic_args", [])
            if generic_args:
                args = ", ".join(
                    self.convert_type_node_to_string(arg) for arg in generic_args
                )
                return f"{type_node.name}<{args}>"
            return type_node.name
        elif hasattr(type_node, "element_type") and hasattr(type_node, "rows"):
            element_type = self.convert_type_node_to_string(type_node.element_type)
            prefix = "dmat" if element_type in {"double", "f64"} else "mat"
            if type_node.rows == type_node.cols:
                return f"{prefix}{type_node.rows}"
            return f"{prefix}{type_node.rows}x{type_node.cols}"
        elif hasattr(type_node, "element_type") and hasattr(type_node, "size"):
            element_type = self.convert_type_node_to_string(type_node.element_type)
            size = type_node.size
            if element_type in {"float", "f32"}:
                return f"vec{size}"
            elif element_type in {"int", "i32"}:
                return f"ivec{size}"
            elif element_type in {"uint", "u32"}:
                return f"uvec{size}"
            elif element_type in {"double", "f64"}:
                return f"dvec{size}"
            elif element_type == "bool":
                return f"bvec{size}"
            else:
                return f"{element_type}{size}"
        else:
            return str(type_node)

    def pointer_pointee_type_name_from_string(self, type_name) -> Optional[str]:
        """Return the pointee type for CrossGL pointer spelling such as T*."""
        type_str = self.type_name_from_value(type_name)
        if type_str is None:
            return None

        type_str = type_str.strip()
        if not type_str.endswith("*"):
            return None

        pointee = type_str[:-1].strip()
        return pointee or None

    def normalize_reference_type_name(self, type_name) -> Optional[str]:
        """Return the value type for CrossGL reference spelling such as &T or T&."""
        if type_name is None:
            return None

        type_str = str(type_name).strip()
        while type_str.startswith("&"):
            type_str = type_str[1:].strip()
        while type_str.endswith("&"):
            type_str = type_str[:-1].strip()
        return type_str or None

    def atomic_element_type_name(self, type_name) -> Optional[str]:
        """Return the scalar payload type for atomic<T> aliases."""
        type_str = self.type_name_from_value(type_name)
        if type_str is None:
            return None

        compact = re.sub(r"\s+", "", str(type_str))
        atomic_match = re.fullmatch(r"atomic<(.+)>", compact)
        if atomic_match:
            return atomic_match.group(1)

        return {
            "atomic_int": "int",
            "atomic_uint": "uint",
        }.get(compact)

    def storage_buffer_element_type_name(self, type_name) -> str:
        atomic_type_name = self.atomic_element_type_name(type_name)
        return atomic_type_name or self.type_name_from_value(type_name)

    def collect_enum_metadata(self, nodes):
        """Collect SPIR-V-lowerable enum types and variant discriminants."""
        for node in nodes or []:
            if isinstance(node, EnumNode):
                self.register_enum_metadata(node)
            elif isinstance(node, StructNode):
                if node.name in self.generic_enum_struct_definitions:
                    continue
                self.collect_enum_metadata(getattr(node, "members", []) or [])

    def register_enum_metadata(self, enum_node: EnumNode):
        variants = getattr(enum_node, "variants", []) or []
        has_payload = any(
            self.enum_variant_has_payload(variant) for variant in variants
        )
        if has_payload:
            fields = enum_struct_fields(enum_node)
            if fields is not None:
                self.enum_struct_type_names.add(enum_node.name)
                self.enum_struct_fields[enum_node.name] = fields
                self.enum_struct_variant_fields[enum_node.name] = {
                    variant.name: enum_variant_payload_fields(variant) or []
                    for variant in variants
                }
        else:
            self.enum_type_names.add(enum_node.name)

        next_value = 0
        for variant in variants:
            explicit_value = self.literal_int_argument(getattr(variant, "value", None))
            if explicit_value is not None:
                next_value = explicit_value
            self.enum_variant_values[f"{enum_node.name}::{variant.name}"] = next_value
            next_value += 1

    def register_generic_enum_metadata(self):
        """Collect generic enum wrapper variant tags shared by all specializations."""
        for name, definition in self.generic_enum_struct_definitions.items():
            self.enum_struct_type_names.add(name)
            next_value = 0
            for variant in definition["enum"].variants or []:
                explicit_value = self.literal_int_argument(
                    getattr(variant, "value", None)
                )
                if explicit_value is not None:
                    next_value = explicit_value
                self.enum_variant_values[f"{name}::{variant.name}"] = next_value
                next_value += 1

    def generic_enum_specialization_for_type(self, type_value) -> Optional[SpirvId]:
        specialization = resolve_generic_enum_specialization(self, type_value)
        if specialization is None:
            return None
        return self.ensure_generic_enum_specialization_registered(specialization)

    def generic_enum_specialization_for_struct_name(self, struct_name: str):
        for specialization in self.generic_enum_specializations.values():
            if specialization["struct_name"] == struct_name:
                return specialization
        return None

    def generic_enum_constructor_specialization(self, enum_name: str):
        expected_type = self.current_expression_expected_type
        specialization = resolve_generic_enum_specialization(
            self, expected_type, expected_base=enum_name
        )
        if specialization is not None:
            self.ensure_generic_enum_specialization_registered(specialization)
            return specialization

        specialization = self.generic_enum_specialization_for_struct_name(
            str(expected_type)
        )
        if specialization is not None and specialization["base_name"] == enum_name:
            return specialization

        candidates = [
            specialization
            for specialization in self.generic_enum_specializations.values()
            if specialization["base_name"] == enum_name
        ]
        if len(candidates) == 1:
            return candidates[0]
        return None

    def ensure_generic_enum_specialization_registered(self, specialization) -> SpirvId:
        type_name = specialization["type_name"]
        struct_name = specialization["struct_name"]
        self.generic_enum_specializations.setdefault(type_name, specialization)
        self.enum_struct_type_names.add(struct_name)

        if struct_name not in self.enum_struct_fields:
            self.enum_struct_fields[struct_name] = generic_enum_specialized_fields(
                self, specialization
            )
            self.enum_struct_variant_fields[struct_name] = {
                variant.name: (
                    generic_enum_specialized_variant_fields(
                        self, specialization, variant.name
                    )
                    or []
                )
                for variant in specialization["definition"]["enum"].variants or []
            }

        if struct_name not in self.struct_types:
            int_type = self.register_primitive_type("int")
            members = [(int_type, "variant")]
            for field_name, field_type in self.enum_struct_fields.get(struct_name, []):
                members.append((self.map_crossgl_type(field_type), field_name))
            self.register_struct_type(struct_name, members)

        return self.struct_types[struct_name]

    def enum_variant_has_payload(self, variant) -> bool:
        return bool(getattr(variant, "data", None) or getattr(variant, "fields", None))

    def enum_variant_constant(self, name: str) -> Optional[SpirvId]:
        value = self.enum_variant_values.get(name)
        if value is None:
            return None
        return self.register_constant(value, self.register_primitive_type("int"))

    def process_enum_structs(self, nodes):
        """Register tagged-struct representations for payload enums."""
        for node in nodes or []:
            if isinstance(node, EnumNode) and node.name in self.enum_struct_type_names:
                self.ensure_enum_struct_type_registered(node.name)
            elif isinstance(node, StructNode):
                self.process_enum_structs(getattr(node, "members", []) or [])

    def enum_path_parts(self, path: str):
        if "::" not in str(path):
            return None
        enum_name, variant_name = str(path).split("::", 1)
        return enum_name, variant_name

    def enum_variant_is_payload_path(self, path: str) -> bool:
        parts = self.enum_path_parts(path)
        if parts is None:
            return False
        enum_name, _variant_name = parts
        return enum_name in self.enum_struct_type_names

    def enum_variant_fields_for_path(
        self, path: str, expression: Optional[SpirvId] = None
    ):
        parts = self.enum_path_parts(path)
        if parts is None:
            return None
        enum_name, variant_name = parts
        if enum_name in self.generic_enum_struct_definitions:
            specialization = None
            if expression is not None:
                specialization = self.generic_enum_specialization_for_struct_name(
                    expression.type.base_type
                )
            if specialization is None:
                specialization = self.generic_enum_constructor_specialization(enum_name)
            if specialization is None:
                return None
            return generic_enum_specialized_variant_fields(
                self, specialization, variant_name
            )
        return self.enum_struct_variant_fields.get(enum_name, {}).get(variant_name)

    def process_enum_variant_constructor(
        self,
        path: str,
        positional_args,
        named_args=None,
    ) -> Optional[SpirvId]:
        parts = self.enum_path_parts(path)
        if parts is None:
            return None
        enum_name, _variant_name = parts
        if enum_name in self.generic_enum_struct_definitions:
            return self.process_generic_enum_variant_constructor(
                path, positional_args, named_args
            )

        if enum_name not in self.enum_struct_type_names:
            if positional_args or named_args:
                return None
            return self.enum_variant_constant(path)

        enum_type = self.struct_types.get(enum_name)
        if enum_type is None:
            return None

        variant_fields = self.enum_variant_fields_for_path(path)
        if variant_fields is None:
            return None

        named_args = dict(named_args or {})
        if len(positional_args) > len(variant_fields):
            raise ValueError(
                f"Enum constructor {path} expects at most {len(variant_fields)} "
                f"arguments, got {len(positional_args)}"
            )

        unknown_names = sorted(set(named_args) - {name for name, _ in variant_fields})
        if unknown_names:
            raise ValueError(
                f"Enum constructor {path} has no field {', '.join(unknown_names)}"
            )

        active_values = {}
        missing_names = []
        for index, (field_name, field_type) in enumerate(variant_fields):
            if index < len(positional_args):
                value_expr = positional_args[index]
            elif field_name in named_args:
                value_expr = named_args[field_name]
            else:
                missing_names.append(field_name)
                continue

            value = self.process_expression(value_expr)
            if value is None:
                return None
            active_values[field_name] = self.convert_value_to_type(
                value, self.map_crossgl_type(field_type)
            )

        if missing_names:
            raise ValueError(
                f"Enum constructor {path} is missing field {', '.join(missing_names)}"
            )

        components = [self.enum_variant_constant(path)]
        for field_name, field_type in self.enum_struct_fields.get(enum_name, []):
            field_type_id = self.map_crossgl_type(field_type)
            components.append(
                active_values.get(field_name)
                or self.default_value_for_type(field_type_id)
            )

        return self.composite_construct(enum_type, components)

    def process_generic_enum_variant_constructor(
        self,
        path: str,
        positional_args,
        named_args=None,
    ) -> Optional[SpirvId]:
        enum_name, variant_name = self.enum_path_parts(path)
        specialization = self.generic_enum_constructor_specialization(enum_name)
        if specialization is None:
            return None

        enum_type = self.ensure_generic_enum_specialization_registered(specialization)
        variant_fields = generic_enum_specialized_variant_fields(
            self, specialization, variant_name
        )
        if variant_fields is None:
            return None

        named_args = dict(named_args or {})
        if len(positional_args) > len(variant_fields):
            raise ValueError(
                f"Enum constructor {path} expects at most {len(variant_fields)} "
                f"arguments, got {len(positional_args)}"
            )

        unknown_names = sorted(set(named_args) - {name for name, _ in variant_fields})
        if unknown_names:
            raise ValueError(
                f"Enum constructor {path} has no field {', '.join(unknown_names)}"
            )

        active_values = {}
        missing_names = []
        for index, (field_name, field_type) in enumerate(variant_fields):
            if index < len(positional_args):
                value_expr = positional_args[index]
            elif field_name in named_args:
                value_expr = named_args[field_name]
            else:
                missing_names.append(field_name)
                continue

            value = self.process_expression_with_expected_type(value_expr, field_type)
            if value is None:
                return None
            active_values[field_name] = self.convert_value_to_type(
                value, self.map_crossgl_type(field_type)
            )

        if missing_names:
            raise ValueError(
                f"Enum constructor {path} is missing field {', '.join(missing_names)}"
            )

        components = [self.enum_variant_constant(path)]
        for field_name, field_type in self.enum_struct_fields.get(
            specialization["struct_name"], []
        ):
            field_type_id = self.map_crossgl_type(field_type)
            components.append(
                active_values.get(field_name)
                or self.default_value_for_type(field_type_id)
            )

        return self.composite_construct(enum_type, components)

    def process_struct_constructor_node(
        self, expr: ConstructorNode
    ) -> Optional[SpirvId]:
        type_name = self.convert_type_node_to_string(expr.constructor_type)
        if "::" in type_name:
            return self.process_enum_variant_constructor(
                type_name,
                list(getattr(expr, "arguments", []) or []),
                getattr(expr, "named_arguments", {}) or {},
            )

        struct_type = self.struct_types.get(type_name)
        if struct_type is None:
            return None

        members = self.current_struct_members.get(type_name, [])
        positional_args = list(getattr(expr, "arguments", []) or [])
        named_args = dict(getattr(expr, "named_arguments", {}) or {})
        member_names = [member_name for _member_type, member_name in members]
        unknown_names = sorted(set(named_args) - set(member_names))
        if unknown_names:
            raise ValueError(
                f"Struct constructor {type_name} has no field "
                f"{', '.join(unknown_names)}"
            )

        components = []
        for index, (member_type, member_name) in enumerate(members):
            if index < len(positional_args):
                value_expr = positional_args[index]
            elif member_name in named_args:
                value_expr = named_args[member_name]
            else:
                components.append(self.default_value_for_type(member_type))
                continue

            value = self.process_expression_with_expected_type(
                value_expr, member_type.type.base_type
            )
            if value is None:
                return None
            components.append(self.convert_value_to_type(value, member_type))

        if len(positional_args) > len(members):
            self.emit(
                f"; WARNING: Constructor {type_name} expected {len(members)} "
                f"members but got {len(positional_args)}; truncating extra arguments"
            )

        return self.composite_construct(struct_type, components)

    def process_crossgl_struct(self, struct_node: StructNode) -> SpirvId:
        """Process a CrossGL struct definition."""
        if struct_node.name in self.generic_enum_struct_definitions:
            return self.struct_types.get(struct_node.name)

        members = []
        member_metadata = {}
        allow_runtime_array = struct_node.name in self.glsl_buffer_block_type_names

        for member in struct_node.members:
            member_type = None
            member_name = member.name

            if isinstance(member, ArrayNode):
                element_type = member.element_type
                if hasattr(element_type, "name") or hasattr(
                    element_type, "element_type"
                ):
                    element_type = self.convert_type_node_to_string(element_type)
                size = self.format_array_size(member.size)
                member_type_name = (
                    f"{element_type}[{size}]"
                    if size is not None
                    else f"{element_type}[]"
                )
                member_type = self.map_crossgl_type(
                    self.spirv_struct_member_storage_type_name(
                        member_type_name, allow_runtime_array=allow_runtime_array
                    )
                )
            else:
                member_type_source = getattr(
                    member,
                    "member_type",
                    getattr(member, "var_type", getattr(member, "vtype", None)),
                )
                if member_type_source is not None:
                    member_type = self.map_crossgl_type(
                        self.spirv_struct_member_storage_type_name(
                            member_type_source,
                            allow_runtime_array=allow_runtime_array,
                        )
                    )

            if member_type:
                members.append((member_type, member_name))
                member_metadata[member_name] = {
                    "node": member,
                    "semantic": self.semantic_from_node(member),
                    "type": member_type,
                }

        self.struct_member_metadata[struct_node.name] = member_metadata
        return self.register_struct_type(struct_node.name, members)

    def process_cbuffer_declaration(self, cbuffer_node: StructNode) -> SpirvId:
        """Emit a CrossGL cbuffer as a SPIR-V Uniform block."""
        members = []
        for member in getattr(cbuffer_node, "members", []) or []:
            member_type_source = getattr(
                member,
                "member_type",
                getattr(member, "var_type", getattr(member, "vtype", None)),
            )
            if member_type_source is None:
                continue
            member_type = self.map_crossgl_type(member_type_source)
            members.append((member_type, member.name))

        cbuffer_type = self.register_struct_type(cbuffer_node.name, members)
        self.decorate_cbuffer_type(cbuffer_type, members)

        var_id = self.create_variable(cbuffer_type, "Uniform", cbuffer_node.name)
        descriptor_set, binding = self.resource_descriptor_slot(cbuffer_node)
        self.decorations.append(
            f"OpDecorate %{var_id.id} DescriptorSet {descriptor_set}"
        )
        self.decorations.append(f"OpDecorate %{var_id.id} Binding {binding}")

        self.global_variables[cbuffer_node.name] = var_id
        self.cbuffer_variables[cbuffer_node.name] = var_id
        self.uniform_buffers.append(var_id)
        for member_index, (member_type, member_name) in enumerate(members):
            if member_name in self.cbuffer_members:
                raise ValueError(f"Ambiguous SPIR-V cbuffer member {member_name}")
            self.cbuffer_members[member_name] = (var_id, member_type, member_index)

        return var_id

    def process_structured_buffer_declaration(
        self, node: VariableNode, type_name: str
    ) -> SpirvId:
        """Emit a StructuredBuffer/RWStructuredBuffer as a Vulkan BufferBlock."""
        metadata = self.structured_buffer_declared_type_info(type_name)
        if metadata is None:
            raise ValueError(f"Invalid SPIR-V structured buffer type {type_name}")
        memory_flags = self.storage_buffer_memory_flags(
            node, default_readonly=metadata.get("readonly", False)
        )
        if (
            metadata.get("default_writeonly")
            and not memory_flags.get("readonly")
            and not memory_flags.get("readwrite")
        ):
            memory_flags["writeonly"] = True

        element_type = self.storage_layout_type(
            self.map_crossgl_type(metadata["element_type_name"]), "std430"
        )
        self.decorate_storage_buffer_nested_type(element_type)
        runtime_array_type = self.register_array_type(element_type, None)
        self.decorations.append(
            f"OpDecorate %{runtime_array_type.id} "
            f"ArrayStride {self.storage_array_stride(element_type)}"
        )

        block_name = f"{node.name}Buffer"
        block_type = self.register_struct_type(
            block_name,
            [(runtime_array_type, node.name)],
        )
        self.decorations.append(f"OpDecorate %{block_type.id} BufferBlock")
        self.decorations.append(f"OpMemberDecorate %{block_type.id} 0 Offset 0")
        self.decorate_storage_buffer_member_memory_qualifiers(block_type, memory_flags)

        variable_type, is_descriptor_array = (
            self.structured_buffer_descriptor_variable_type(type_name, block_type)
        )
        var_id = self.create_variable(variable_type, "Uniform", node.name)
        descriptor_set, binding = self.resource_descriptor_slot(node)
        self.decorations.append(
            f"OpDecorate %{var_id.id} DescriptorSet {descriptor_set}"
        )
        self.decorations.append(f"OpDecorate %{var_id.id} Binding {binding}")

        buffer_metadata = {
            **metadata,
            **memory_flags,
            "buffer_variable": var_id,
            "declaration_node": node,
            "declared_type_name": type_name,
            "element_type": element_type,
            "runtime_array_type": runtime_array_type,
            "block_type": block_type,
            "member_index": 0,
        }
        if is_descriptor_array:
            buffer_metadata["descriptor_array"] = True
            buffer_metadata["descriptor_array_type"] = variable_type
        if metadata.get("append_consume"):
            counter_metadata = self.process_structured_buffer_counter_declaration(
                node, type_name
            )
            buffer_metadata.update(counter_metadata)
        self.global_variables[node.name] = var_id
        self.structured_buffer_metadata[var_id.id] = buffer_metadata
        self.structured_buffer_metadata[block_type.id] = buffer_metadata
        return var_id

    def process_structured_buffer_counter_declaration(
        self, node: VariableNode, type_name: str
    ) -> dict:
        """Emit the sidecar counter SSBO used for structured-buffer counters."""
        uint_type = self.register_primitive_type("uint")
        block_name = f"{node.name}CounterBuffer"
        block_type = self.register_struct_type(block_name, [(uint_type, "counter")])
        self.decorations.append(f"OpDecorate %{block_type.id} BufferBlock")
        self.decorations.append(f"OpMemberDecorate %{block_type.id} 0 Offset 0")

        variable_type, is_descriptor_array = (
            self.structured_buffer_descriptor_variable_type(type_name, block_type)
        )
        counter_name = f"{node.name}Counter"
        var_id = self.create_variable(variable_type, "Uniform", counter_name)
        descriptor_set = self.resource_descriptor_set(node)
        binding = self.next_available_resource_binding(descriptor_set)
        self.used_resource_bindings.add((descriptor_set, binding))
        self.decorations.append(
            f"OpDecorate %{var_id.id} DescriptorSet {descriptor_set}"
        )
        self.decorations.append(f"OpDecorate %{var_id.id} Binding {binding}")

        counter_access_metadata = {
            "kind": "structured_buffer_counter",
            "block_type": block_type,
            "member_index": 0,
            "element_type": uint_type,
            "readonly": False,
            "writeonly": False,
        }
        self.storage_buffer_access_metadata[var_id.id] = counter_access_metadata
        self.storage_buffer_access_metadata[block_type.id] = counter_access_metadata
        if is_descriptor_array:
            self.storage_buffer_access_metadata[variable_type.id] = (
                counter_access_metadata
            )

        return {
            "counter_variable": var_id,
            "counter_block_type": block_type,
            "counter_variable_type": variable_type,
            "counter_descriptor_array": is_descriptor_array,
            "counter_member_index": 0,
        }

    def structured_buffer_descriptor_variable_type(
        self, type_name: str, block_type: SpirvId
    ) -> Tuple[SpirvId, bool]:
        array_info = self.split_outer_array_type(type_name)
        if array_info is None:
            return block_type, False

        element_type_name, size = array_info
        element_type, _ = self.structured_buffer_descriptor_variable_type(
            element_type_name, block_type
        )
        if size is None:
            self.require_capability("RuntimeDescriptorArray")
            self.require_extension("SPV_EXT_descriptor_indexing")
        return self.register_array_type(element_type, size), True

    def is_glsl_buffer_block_node(self, node: VariableNode) -> bool:
        qualifiers = {
            str(qualifier).lower() for qualifier in getattr(node, "qualifiers", [])
        }
        return "buffer" in qualifiers or self.has_attribute(node, "glsl_buffer_block")

    def glsl_buffer_block_layout(self, node: VariableNode) -> str:
        for attr in getattr(node, "attributes", []) or []:
            if str(getattr(attr, "name", "")).lower() != "glsl_buffer_block":
                continue
            arguments = getattr(attr, "arguments", None)
            if arguments is None:
                arguments = getattr(attr, "args", [])
            if not arguments:
                continue
            layout = self.attribute_value_to_string(arguments[0])
            if layout is not None:
                return str(layout).lower()
        return "std430"

    def process_glsl_buffer_block_declaration(
        self, node: VariableNode, type_name: str
    ) -> SpirvId:
        """Emit a GLSL-style buffer block or buffer-qualified array variable."""
        layout = self.glsl_buffer_block_layout(node)
        base_type_name = self.array_base_type_name(type_name)
        outer_array = self.split_outer_array_type(type_name)
        pointer_element_type_name = self.pointer_pointee_type_name_from_string(
            type_name
        )
        is_pointer_descriptor_array = False
        pointer_descriptor_array_size = None
        if pointer_element_type_name is None and outer_array is not None:
            pointer_element_type_name = self.pointer_pointee_type_name_from_string(
                outer_array[0]
            )
            if pointer_element_type_name is not None:
                is_pointer_descriptor_array = True
                pointer_descriptor_array_size = outer_array[1]
        is_named_block = base_type_name in self.struct_types
        if is_named_block and outer_array is not None and outer_array[1] is None:
            self.require_capability("RuntimeDescriptorArray")
            self.require_extension("SPV_EXT_descriptor_indexing")
        if is_pointer_descriptor_array and pointer_descriptor_array_size is None:
            self.require_capability("RuntimeDescriptorArray")
            self.require_extension("SPV_EXT_descriptor_indexing")

        if is_named_block:
            value_type = self.storage_layout_type(
                self.map_crossgl_type(type_name), layout
            )
            block_type = self.storage_layout_type(
                self.struct_types[base_type_name], layout
            )
            block_members = self.current_struct_members.get(base_type_name, [])
            block_members = self.current_struct_members.get(
                block_type.type.base_type, block_members
            )
            variable_member_name = None
            variable_member_type = None
        else:
            variable_member_name = node.name
            if pointer_element_type_name is not None:
                element_type_name = self.storage_buffer_element_type_name(
                    pointer_element_type_name
                )
                element_type = self.storage_layout_type(
                    self.map_crossgl_type(element_type_name),
                    layout,
                )
                variable_member_type = self.register_array_type(element_type, None)
            else:
                variable_member_type = self.storage_layout_type(
                    self.map_crossgl_type(
                        getattr(node, "var_type", getattr(node, "vtype", "float"))
                    ),
                    layout,
                )
            block_type = self.register_struct_type(
                (
                    f"{node.name}Buffer"
                    if layout == "std430"
                    else f"{node.name}Buffer_{layout}"
                ),
                [(variable_member_type, variable_member_name)],
            )
            value_type = block_type
            block_members = [(variable_member_type, variable_member_name)]

        if is_pointer_descriptor_array:
            value_type = self.register_array_type(
                block_type, pointer_descriptor_array_size
            )

        self.decorate_storage_buffer_block_type(block_type, layout)
        memory_flags = self.storage_buffer_memory_flags(node)
        self.decorate_storage_buffer_member_memory_qualifiers(block_type, memory_flags)

        var_id = self.create_variable(value_type, "Uniform", node.name)
        descriptor_set, binding = self.resource_descriptor_slot(node)
        self.decorations.append(
            f"OpDecorate %{var_id.id} DescriptorSet {descriptor_set}"
        )
        self.decorations.append(f"OpDecorate %{var_id.id} Binding {binding}")

        self.global_variables[node.name] = var_id
        self.register_glsl_buffer_access_metadata(node, var_id, value_type, block_type)
        self.register_single_array_storage_buffer_metadata(
            node,
            var_id,
            block_type,
            block_members,
            variable_member_name,
            variable_member_type,
            descriptor_array_type=(value_type if is_pointer_descriptor_array else None),
        )
        return var_id

    def register_single_array_storage_buffer_metadata(
        self,
        node: VariableNode,
        var_id: SpirvId,
        block_type: SpirvId,
        block_members: List[Tuple[SpirvId, str]],
        variable_member_name: Optional[str],
        variable_member_type: Optional[SpirvId],
        descriptor_array_type: Optional[SpirvId] = None,
    ):
        if len(block_members) != 1:
            return

        member_type, member_name = block_members[0]
        array_info = self.array_type_info_from_type(member_type)
        if array_info is None:
            return

        element_type, _ = array_info
        memory_flags = self.storage_buffer_memory_flags(node)
        metadata = {
            "kind": "structured_buffer",
            "buffer_kind": (
                "StructuredBuffer"
                if memory_flags.get("readonly")
                else "RWStructuredBuffer"
            ),
            "element_type_name": element_type.type.base_type,
            **memory_flags,
            "element_type": element_type,
            "runtime_array_type": member_type,
            "block_type": block_type,
            "member_index": 0,
            "member_name": variable_member_name or member_name,
            "declared_type_name": self.type_name_from_value(
                getattr(node, "var_type", getattr(node, "vtype", None))
            ),
        }
        if descriptor_array_type is not None:
            metadata["descriptor_array"] = True
            metadata["descriptor_array_type"] = descriptor_array_type
        self.structured_buffer_metadata[var_id.id] = metadata
        self.structured_buffer_metadata[block_type.id] = metadata
        if variable_member_type is not None:
            self.structured_buffer_metadata[variable_member_type.id] = metadata
        if descriptor_array_type is not None:
            self.structured_buffer_metadata[descriptor_array_type.id] = metadata

    def register_glsl_buffer_access_metadata(
        self,
        node: VariableNode,
        var_id: SpirvId,
        value_type: SpirvId,
        block_type: SpirvId,
    ):
        memory_flags = self.storage_buffer_memory_flags(node)
        metadata = {
            "kind": "glsl_buffer_block",
            **memory_flags,
            "block_type": block_type,
        }
        self.storage_buffer_access_metadata[var_id.id] = metadata
        self.storage_buffer_access_metadata[block_type.id] = metadata
        if value_type.id != block_type.id:
            self.storage_buffer_access_metadata[value_type.id] = metadata

    def storage_buffer_is_readonly(self, node: VariableNode) -> bool:
        return self.storage_buffer_memory_flags(node).get("readonly", False)

    def storage_buffer_is_writeonly(self, node: VariableNode) -> bool:
        return self.storage_buffer_memory_flags(node).get("writeonly", False)

    def resource_memory_qualifier_names(self, node) -> set:
        supported = {
            "coherent",
            "globallycoherent",
            "volatile",
            "restrict",
            "readonly",
            "read",
            "writeonly",
            "write",
            "readwrite",
            "read_write",
            "access::read",
            "access::write",
            "access::read_write",
        }
        aliases = {
            "read": "readonly",
            "write": "writeonly",
            "read_write": "readwrite",
            "access::read": "readonly",
            "access::write": "writeonly",
            "access::read_write": "readwrite",
        }
        qualifiers = set()

        for qualifier in getattr(node, "qualifiers", []) or []:
            qualifier = str(qualifier).lower()
            if qualifier in supported:
                qualifiers.add(aliases.get(qualifier, qualifier))

        for qualifier in getattr(node, "resource_qualifiers", []) or []:
            qualifier = str(qualifier).lower()
            if qualifier in supported:
                qualifiers.add(aliases.get(qualifier, qualifier))

        for attr in getattr(node, "attributes", []) or []:
            attr_name = getattr(attr, "name", None)
            if not attr_name:
                continue

            attr_name = str(attr_name).lower()
            if attr_name in supported:
                qualifiers.add(aliases.get(attr_name, attr_name))

            if attr_name != "access":
                continue

            arguments = getattr(attr, "arguments", None)
            if arguments is None:
                arguments = getattr(attr, "args", [])
            if not arguments:
                continue

            access_name = self.attribute_value_to_string(arguments[0])
            if access_name is None:
                continue
            access_name = str(access_name).lower()
            if access_name in supported:
                qualifiers.add(aliases.get(access_name, access_name))

        if "globallycoherent" in qualifiers:
            qualifiers.add("coherent")

        return qualifiers

    def resource_memory_qualifier_flags(self, node) -> dict:
        qualifiers = self.resource_memory_qualifier_names(node)
        readwrite = "readwrite" in qualifiers
        return {
            "readonly": "readonly" in qualifiers and not readwrite,
            "writeonly": "writeonly" in qualifiers and not readwrite,
            "coherent": "coherent" in qualifiers,
            "volatile": "volatile" in qualifiers,
            "restrict": "restrict" in qualifiers,
            "readwrite": readwrite,
        }

    def storage_buffer_memory_flags(
        self, node: VariableNode, default_readonly: bool = False
    ) -> dict:
        flags = self.resource_memory_qualifier_flags(node)
        if (
            default_readonly
            and not flags.get("writeonly")
            and not flags.get("readwrite")
        ):
            flags["readonly"] = True
        return flags

    def resource_memory_decoration_names(self, flags: dict) -> List[str]:
        decorations = []
        if flags.get("readonly"):
            decorations.append("NonWritable")
        if flags.get("writeonly"):
            decorations.append("NonReadable")
        if flags.get("coherent"):
            decorations.append("Coherent")
        if flags.get("volatile"):
            decorations.append("Volatile")
        if flags.get("restrict"):
            decorations.append("Restrict")
        return decorations

    def decorate_storage_buffer_member_memory_qualifiers(
        self, block_type: SpirvId, flags: dict
    ):
        decoration_names = self.resource_memory_decoration_names(flags)
        if not decoration_names:
            return

        members = self.current_struct_members.get(block_type.type.base_type, [])
        for member_index in range(len(members)):
            for decoration_name in decoration_names:
                self.decorations.append(
                    f"OpMemberDecorate %{block_type.id} "
                    f"{member_index} {decoration_name}"
                )

    def decorate_resource_variable_memory_qualifiers(
        self, var_id: SpirvId, metadata: dict
    ):
        for decoration_name in self.resource_memory_decoration_names(metadata):
            self.decorations.append(f"OpDecorate %{var_id.id} {decoration_name}")

    def metadata_with_resource_memory_qualifiers(self, metadata: dict, node) -> dict:
        return {
            **metadata,
            **self.resource_memory_qualifier_flags(node),
        }

    def resource_metadata_for_declared_type(self, type_id: SpirvId):
        metadata = self.resource_type_metadata.get(type_id.id)
        if metadata is not None:
            return metadata

        element_type = self.array_element_type_from_type(type_id)
        while element_type is not None:
            metadata = self.resource_type_metadata.get(element_type.id)
            if metadata is not None:
                return metadata
            element_type = self.array_element_type_from_type(element_type)

        return None

    def register_declared_resource_metadata(
        self, node: VariableNode, var_id: SpirvId, type_id: SpirvId
    ):
        metadata = self.resource_metadata_for_declared_type(type_id)
        if metadata is None or metadata.get("kind") not in {
            "sampled_image",
            "storage_image",
            "texture",
            "sampler",
        }:
            return

        metadata = self.metadata_with_resource_memory_qualifiers(metadata, node)
        self.resource_type_metadata[var_id.id] = metadata
        if metadata.get("kind") == "storage_image":
            self.decorate_resource_variable_memory_qualifiers(var_id, metadata)

    def decorate_cbuffer_type(
        self, cbuffer_type: SpirvId, members: List[Tuple[SpirvId, str]]
    ):
        self.decorations.append(f"OpDecorate %{cbuffer_type.id} Block")
        self.decorate_uniform_nested_type(cbuffer_type)

        offset = 0
        for member_index, (member_type, _) in enumerate(members):
            offset = self.align_to(offset, self.uniform_layout_alignment(member_type))
            self.decorations.append(
                f"OpMemberDecorate %{cbuffer_type.id} {member_index} Offset {offset}"
            )
            if self.uniform_member_needs_matrix_layout(member_type):
                self.decorations.append(
                    f"OpMemberDecorate %{cbuffer_type.id} {member_index} ColMajor"
                )
                self.decorations.append(
                    f"OpMemberDecorate %{cbuffer_type.id} {member_index} MatrixStride 16"
                )
            self.decorate_uniform_array_strides(member_type)
            offset += self.uniform_layout_size(member_type)

    def decorate_uniform_nested_type(
        self, type_id: SpirvId, decorated_structs: Optional[set] = None
    ):
        if decorated_structs is None:
            decorated_structs = set()

        array_info = self.array_type_info_from_type(type_id)
        if array_info is not None:
            element_type, _ = array_info
            stride = self.uniform_array_stride(element_type)
            self.decorations.append(f"OpDecorate %{type_id.id} ArrayStride {stride}")
            self.decorate_uniform_nested_type(element_type, decorated_structs)
            return

        members = self.current_struct_members.get(type_id.type.base_type)
        if members is None or type_id.id in decorated_structs:
            return

        decorated_structs.add(type_id.id)
        offset = 0
        for member_index, (member_type, _) in enumerate(members):
            offset = self.align_to(offset, self.uniform_layout_alignment(member_type))
            self.decorations.append(
                f"OpMemberDecorate %{type_id.id} {member_index} Offset {offset}"
            )
            if self.uniform_member_needs_matrix_layout(member_type):
                self.decorations.append(
                    f"OpMemberDecorate %{type_id.id} {member_index} ColMajor"
                )
                self.decorations.append(
                    f"OpMemberDecorate %{type_id.id} {member_index} MatrixStride 16"
                )
            self.decorate_uniform_nested_type(member_type, decorated_structs)
            offset += self.uniform_layout_size(member_type)

    def align_to(self, value: int, alignment: int) -> int:
        if alignment <= 1:
            return value
        return ((value + alignment - 1) // alignment) * alignment

    def uniform_layout_alignment(self, type_id: SpirvId) -> int:
        if self.matrix_type_info_from_type(type_id) is not None:
            return 16

        array_info = self.array_type_info_from_type(type_id)
        if array_info is not None:
            element_type, _ = array_info
            return max(16, self.uniform_layout_alignment(element_type))

        struct_members = self.current_struct_members.get(type_id.type.base_type)
        if struct_members is not None:
            return max(
                16,
                max(
                    (
                        self.uniform_layout_alignment(member_type)
                        for member_type, _ in struct_members
                    ),
                    default=1,
                ),
            )

        vector_info = self.vector_type_info_from_type(type_id)
        if vector_info is not None:
            _, component_count = vector_info
            return 8 if component_count == 2 else 16

        return 4

    def uniform_layout_size(self, type_id: SpirvId) -> int:
        matrix_info = self.matrix_type_info_from_type(type_id)
        if matrix_info is not None:
            _, column_count = matrix_info
            return 16 * column_count

        array_info = self.array_type_info_from_type(type_id)
        if array_info is not None:
            element_type, size = array_info
            stride = self.uniform_array_stride(element_type)
            return stride * int(size or 1)

        struct_members = self.current_struct_members.get(type_id.type.base_type)
        if struct_members is not None:
            offset = 0
            for member_type, _ in struct_members:
                alignment = self.uniform_layout_alignment(member_type)
                offset = self.align_to(offset, alignment)
                offset += self.uniform_layout_size(member_type)
            return self.align_to(offset, self.uniform_layout_alignment(type_id))

        vector_info = self.vector_type_info_from_type(type_id)
        if vector_info is not None:
            component_type, component_count = vector_info
            component_size = self.uniform_scalar_size(component_type)
            return self.align_to(component_size * component_count, 16)

        return self.uniform_scalar_size(type_id)

    def uniform_scalar_size(self, type_id: SpirvId) -> int:
        return (
            8
            if self.normalize_primitive_name(type_id.type.base_type) == "double"
            else 4
        )

    def uniform_array_stride(self, element_type: SpirvId) -> int:
        return self.align_to(self.uniform_layout_size(element_type), 16)

    def uniform_layout_contains_matrix(self, type_id: SpirvId) -> bool:
        if self.matrix_type_info_from_type(type_id) is not None:
            return True

        array_info = self.array_type_info_from_type(type_id)
        if array_info is not None:
            element_type, _ = array_info
            return self.uniform_layout_contains_matrix(element_type)

        struct_members = self.current_struct_members.get(type_id.type.base_type)
        if struct_members is not None:
            return any(
                self.uniform_layout_contains_matrix(member_type)
                for member_type, _ in struct_members
            )

        return False

    def uniform_member_needs_matrix_layout(self, type_id: SpirvId) -> bool:
        if self.matrix_type_info_from_type(type_id) is not None:
            return True

        array_info = self.array_type_info_from_type(type_id)
        if array_info is not None:
            element_type, _ = array_info
            return self.uniform_member_needs_matrix_layout(element_type)

        return False

    def decorate_uniform_array_strides(self, type_id: SpirvId):
        array_info = self.array_type_info_from_type(type_id)
        if array_info is None:
            return

        element_type, _ = array_info
        stride = self.uniform_array_stride(element_type)
        self.decorations.append(f"OpDecorate %{type_id.id} ArrayStride {stride}")
        self.decorate_uniform_array_strides(element_type)

    def storage_layout_type(self, type_id: SpirvId, layout: str) -> SpirvId:
        if type_id.type.base_type == "bool":
            return self.register_primitive_type("uint")

        array_info = self.array_type_info_from_type(type_id)
        if array_info is not None:
            element_type, size = array_info
            layout_element_type = self.storage_layout_type(element_type, layout)
            if layout == "std430" and layout_element_type.id == element_type.id:
                return type_id
            return self.register_layout_array_type(layout_element_type, size, layout)

        struct_members = self.current_struct_members.get(type_id.type.base_type)
        if struct_members is not None:
            cloned_members = [
                (self.storage_layout_type(member_type, layout), member_name)
                for member_type, member_name in struct_members
            ]
            if layout == "std430" and all(
                cloned_type.id == member_type.id
                for (cloned_type, _), (member_type, _member_name) in zip(
                    cloned_members, struct_members
                )
            ):
                return type_id
            return self.register_layout_struct_type(type_id, layout, cloned_members)

        return type_id

    def decorate_storage_buffer_block_type(
        self, block_type: SpirvId, layout: str = "std430"
    ):
        self.decorations.append(f"OpDecorate %{block_type.id} BufferBlock")
        for member_index, (member_type, _) in enumerate(
            self.current_struct_members.get(block_type.type.base_type, [])
        ):
            offset = self.storage_struct_member_offset(block_type, member_index, layout)
            self.decorations.append(
                f"OpMemberDecorate %{block_type.id} {member_index} Offset {offset}"
            )
            matrix_stride = self.storage_matrix_stride_for_member(member_type, layout)
            if matrix_stride is not None:
                self.decorations.append(
                    f"OpMemberDecorate %{block_type.id} {member_index} ColMajor"
                )
                self.decorations.append(
                    f"OpMemberDecorate %{block_type.id} {member_index} "
                    f"MatrixStride {matrix_stride}"
                )
            self.decorate_storage_buffer_nested_type(member_type, layout)

    def decorate_storage_buffer_nested_type(
        self, type_id: SpirvId, layout: str = "std430"
    ):
        struct_members = self.current_struct_members.get(type_id.type.base_type)
        if struct_members is not None:
            for member_index, (member_type, _) in enumerate(struct_members):
                offset = self.storage_struct_member_offset(
                    type_id, member_index, layout
                )
                self.decorations.append(
                    f"OpMemberDecorate %{type_id.id} {member_index} Offset {offset}"
                )
                matrix_stride = self.storage_matrix_stride_for_member(
                    member_type, layout
                )
                if matrix_stride is not None:
                    self.decorations.append(
                        f"OpMemberDecorate %{type_id.id} {member_index} ColMajor"
                    )
                    self.decorations.append(
                        f"OpMemberDecorate %{type_id.id} {member_index} "
                        f"MatrixStride {matrix_stride}"
                    )
                self.decorate_storage_buffer_nested_type(member_type, layout)
            return

        array_info = self.array_type_info_from_type(type_id)
        if array_info is not None:
            element_type, _ = array_info
            self.decorations.append(
                f"OpDecorate %{type_id.id} "
                f"ArrayStride {self.storage_array_stride(element_type, layout)}"
            )
            self.decorate_storage_buffer_nested_type(element_type, layout)

    def storage_struct_member_offset(
        self,
        struct_type: SpirvId,
        target_member_index: int,
        layout: str = "std430",
    ) -> int:
        offset = 0
        for member_index, (member_type, _) in enumerate(
            self.current_struct_members.get(struct_type.type.base_type, [])
        ):
            offset = self.align_to(
                offset, self.storage_layout_alignment(member_type, layout)
            )
            if member_index == target_member_index:
                return offset
            offset += self.storage_layout_size(member_type, layout)
        return offset

    def storage_layout_alignment(self, type_id: SpirvId, layout: str = "std430") -> int:
        if layout == "std140":
            return self.uniform_layout_alignment(type_id)
        if layout == "scalar":
            return self.scalar_layout_alignment(type_id)

        matrix_info = self.matrix_type_info_from_type(type_id)
        if matrix_info is not None:
            column_type, _ = matrix_info
            return self.storage_layout_alignment(column_type, layout)

        array_info = self.array_type_info_from_type(type_id)
        if array_info is not None:
            element_type, _ = array_info
            return self.storage_layout_alignment(element_type, layout)

        struct_members = self.current_struct_members.get(type_id.type.base_type)
        if struct_members is not None:
            return max(
                (
                    self.storage_layout_alignment(member_type, layout)
                    for member_type, _ in struct_members
                ),
                default=1,
            )

        vector_info = self.vector_type_info_from_type(type_id)
        if vector_info is not None:
            _, component_count = vector_info
            if component_count == 2:
                return 8
            if component_count in {3, 4}:
                return 16

        return self.uniform_scalar_size(type_id)

    def storage_layout_size(self, type_id: SpirvId, layout: str = "std430") -> int:
        if layout == "std140":
            return self.uniform_layout_size(type_id)
        if layout == "scalar":
            return self.scalar_layout_size(type_id)

        matrix_info = self.matrix_type_info_from_type(type_id)
        if matrix_info is not None:
            column_type, column_count = matrix_info
            return self.storage_array_stride(column_type, layout) * column_count

        array_info = self.array_type_info_from_type(type_id)
        if array_info is not None:
            element_type, size = array_info
            return self.storage_array_stride(element_type, layout) * int(size or 1)

        struct_members = self.current_struct_members.get(type_id.type.base_type)
        if struct_members is not None:
            offset = 0
            max_alignment = 1
            for member_type, _ in struct_members:
                alignment = self.storage_layout_alignment(member_type, layout)
                max_alignment = max(max_alignment, alignment)
                offset = self.align_to(offset, alignment)
                offset += self.storage_layout_size(member_type, layout)
            return self.align_to(offset, max_alignment)

        vector_info = self.vector_type_info_from_type(type_id)
        if vector_info is not None:
            component_type, component_count = vector_info
            return self.uniform_scalar_size(component_type) * component_count

        return self.uniform_scalar_size(type_id)

    def storage_array_stride(
        self, element_type: SpirvId, layout: str = "std430"
    ) -> int:
        if layout == "std140":
            return self.uniform_array_stride(element_type)
        if layout == "scalar":
            return self.scalar_array_stride(element_type)

        return self.align_to(
            self.storage_layout_size(element_type, layout),
            self.storage_layout_alignment(element_type, layout),
        )

    def scalar_layout_alignment(self, type_id: SpirvId) -> int:
        matrix_info = self.matrix_type_info_from_type(type_id)
        if matrix_info is not None:
            column_type, _ = matrix_info
            return self.scalar_layout_alignment(column_type)

        array_info = self.array_type_info_from_type(type_id)
        if array_info is not None:
            element_type, _ = array_info
            return self.scalar_layout_alignment(element_type)

        struct_members = self.current_struct_members.get(type_id.type.base_type)
        if struct_members is not None:
            return max(
                (
                    self.scalar_layout_alignment(member_type)
                    for member_type, _ in struct_members
                ),
                default=1,
            )

        vector_info = self.vector_type_info_from_type(type_id)
        if vector_info is not None:
            component_type, _ = vector_info
            return self.uniform_scalar_size(component_type)

        return self.uniform_scalar_size(type_id)

    def scalar_layout_size(self, type_id: SpirvId) -> int:
        matrix_info = self.matrix_type_info_from_type(type_id)
        if matrix_info is not None:
            column_type, column_count = matrix_info
            return self.scalar_array_stride(column_type) * column_count

        array_info = self.array_type_info_from_type(type_id)
        if array_info is not None:
            element_type, size = array_info
            return self.scalar_array_stride(element_type) * int(size or 1)

        struct_members = self.current_struct_members.get(type_id.type.base_type)
        if struct_members is not None:
            offset = 0
            max_alignment = 1
            for member_type, _ in struct_members:
                alignment = self.scalar_layout_alignment(member_type)
                max_alignment = max(max_alignment, alignment)
                offset = self.align_to(offset, alignment)
                offset += self.scalar_layout_size(member_type)
            return self.align_to(offset, max_alignment)

        vector_info = self.vector_type_info_from_type(type_id)
        if vector_info is not None:
            component_type, component_count = vector_info
            return self.uniform_scalar_size(component_type) * component_count

        return self.uniform_scalar_size(type_id)

    def scalar_array_stride(self, element_type: SpirvId) -> int:
        return self.align_to(
            self.scalar_layout_size(element_type),
            self.scalar_layout_alignment(element_type),
        )

    def storage_matrix_stride_for_member(
        self, type_id: SpirvId, layout: str = "std430"
    ) -> Optional[int]:
        matrix_info = self.matrix_type_info_from_type(type_id)
        if matrix_info is not None:
            column_type, _ = matrix_info
            return self.storage_array_stride(column_type, layout)

        array_info = self.array_type_info_from_type(type_id)
        if array_info is not None:
            element_type, _ = array_info
            return self.storage_matrix_stride_for_member(element_type, layout)

        return None

    def execution_model_hint_for_function(self, function_node, stage=None):
        if stage is not None:
            return self.spirv_execution_model(self.stage_key(stage.stage))

        qualifier = self.get_function_qualifier(function_node)
        if qualifier is not None:
            return self.spirv_execution_model(qualifier)

        execution_models = self.function_execution_models.get(function_node.name, set())
        if "GLCompute" in execution_models:
            return "GLCompute"
        if len(execution_models) == 1:
            return next(iter(execution_models))
        return None

    def process_function_node(self, function_node, stage=None):
        """Process a CrossGL function definition."""
        return_type = self.map_crossgl_type(function_node.return_type)
        self.function_nodes[function_node.name] = function_node
        previous_return_type = self.current_return_type
        previous_return_type_source = self.current_return_type_source
        previous_return_semantic_output = self.current_return_semantic_output
        previous_entry_point_return_outputs = self.current_entry_point_return_outputs
        previous_stage = self.current_stage
        previous_function_name = self.current_function_name
        previous_generic_function_substitutions = (
            self.current_generic_function_substitutions
        )
        self.current_return_type = return_type
        self.current_return_type_source = self.type_name_from_value(
            function_node.return_type
        )
        self.current_return_semantic_output = None
        self.current_entry_point_return_outputs = None
        self.current_function_name = function_node.name
        self.current_generic_function_substitutions = (
            getattr(function_node, "_generic_substitutions", {}) or {}
        )
        if stage is not None:
            self.current_stage = stage

        execution_model_hint = self.execution_model_hint_for_function(
            function_node, stage
        )
        is_entry_point = self.function_is_entry_point(function_node, stage)
        if execution_model_hint is None and is_entry_point and stage is None:
            execution_model_hint = self.spirv_execution_model(
                self.get_function_qualifier(function_node)
            )
        return_semantic = self.semantic_from_node(function_node)
        has_direct_return_semantic = is_entry_point and return_semantic is not None
        if return_semantic is not None:
            self.validate_spirv_return_semantic(
                execution_model_hint, function_node.return_type, return_semantic
            )
        entry_point_uses_void_signature = is_entry_point
        function_return_type = (
            self.register_primitive_type("void")
            if entry_point_uses_void_signature
            else return_type
        )
        mesh_output_parameters = {}
        output_parameters = []
        runtime_parameters = []
        param_types = []
        param_value_types = []
        resource_array_param_indices = set()
        value_array_param_indices = set()
        patch_interface_parameters = []
        param_type_hints = self.resolve_function_resource_array_type_hints(
            function_node.name
        )
        storage_image_pointer_params = (
            self.resolve_function_storage_image_pointer_params(function_node.name)
        )
        parameters = getattr(
            function_node, "parameters", getattr(function_node, "params", [])
        )
        parameters = list(parameters or [])
        skipped_parameter_indices = (
            set()
            if is_entry_point
            else self.skipped_function_parameter_indices_for_node(function_node)
        )
        if skipped_parameter_indices:
            parameters = [
                param
                for index, param in enumerate(parameters)
                if index not in skipped_parameter_indices
            ]
        if not is_entry_point:
            required_stage_input = self.required_function_stage_input_type(
                function_node.name
            )
            required_stage_output = self.required_function_stage_output_type(
                function_node.name
            )
            if required_stage_input is not None:
                parameters.append(
                    self.stage_object_pointer_parameter("input", required_stage_input)
                )
            if required_stage_output is not None:
                parameters.append(
                    self.stage_object_pointer_parameter("output", required_stage_output)
                )
        has_mesh_output_parameters = any(
            self.is_mesh_output_parameter(param) for param in parameters
        )
        mesh_parameter_execution_model = execution_model_hint
        if mesh_parameter_execution_model is None and has_mesh_output_parameters:
            mesh_parameter_execution_model = "MeshEXT"
        mesh_output_parameter_indices = set()

        for source_param_index, param in enumerate(parameters):
            param_type_source = getattr(
                param, "param_type", getattr(param, "vtype", None)
            )
            param_name = getattr(param, "name", None)
            mesh_output_info = self.mesh_output_parameter_info(
                param, mesh_parameter_execution_model
            )
            if mesh_output_info is not None:
                mesh_output_parameters[param_name] = mesh_output_info
                mesh_output_parameter_indices.add(source_param_index)
                continue

            patch_info = self.patch_parameter_info(
                param, execution_model_hint, is_entry_point
            )
            if patch_info is not None:
                patch_interface_parameters.append((param, patch_info))
                continue

            if param_name in param_type_hints:
                param_type_source = param_type_hints[param_name]
            if param_type_source is not None:
                param_type = self.map_resource_type_with_format(
                    param_type_source, param
                )
            else:
                param_type = self.map_crossgl_type("float")

            if self.is_graphics_output_parameter(
                param, execution_model_hint, is_entry_point
            ):
                output_parameters.append((param, param_type))
                continue

            param_value_types.append(param_type)
            param_resource_metadata = self.resource_type_metadata.get(param_type.id)
            is_stage_object_pointer_param = bool(
                getattr(param, "_spirv_stage_object_pointer", False)
            )
            is_storage_image_param = (
                param_resource_metadata is not None
                and param_resource_metadata.get("kind") == "storage_image"
                and param_name in storage_image_pointer_params
            )
            is_value_array_param = (
                self.array_type_info_from_type(param_type) is not None
            )
            if (
                self.is_resource_array_type(param_type)
                or is_storage_image_param
                or is_value_array_param
                or is_stage_object_pointer_param
            ):
                resource_array_param_indices.add(len(param_types))
                if is_value_array_param and not (
                    self.is_resource_array_type(param_type)
                    or is_storage_image_param
                    or is_stage_object_pointer_param
                ):
                    value_array_param_indices.add(len(param_types))
                storage_class = (
                    "UniformConstant"
                    if self.is_resource_array_type(param_type) or is_storage_image_param
                    else "Function"
                )
                param_type = self.register_pointer_type(param_type, storage_class)

            if not entry_point_uses_void_signature:
                param_types.append(param_type)
            runtime_parameters.append((param, param_type, param_value_types[-1]))

        had_global_function = function_node.name in self.functions
        previous_global_function = self.functions.get(function_node.name)
        previous_global_signature = self.function_signatures.get(function_node.name)
        function_id = self.create_function(
            function_node.name, function_return_type, param_types
        )
        if stage is not None:
            self.register_stage_local_function(
                stage,
                function_node.name,
                function_id,
                function_return_type,
                param_types,
            )
            if had_global_function:
                self.functions[function_node.name] = previous_global_function
                self.function_signatures[function_node.name] = previous_global_signature
            else:
                self.functions.pop(function_node.name, None)
                self.function_signatures.pop(function_node.name, None)
        self.function_resource_array_params[function_node.name] = (
            resource_array_param_indices
        )
        self.function_mesh_output_parameter_indices[function_node.name] = (
            mesh_output_parameter_indices
        )
        if stage is not None:
            key = self.stage_local_function_key(stage, function_node.name)
            self.stage_local_function_resource_array_params[key] = (
                resource_array_param_indices
            )
            self.stage_local_function_mesh_output_parameter_indices[key] = (
                mesh_output_parameter_indices
            )

        for i, (param, param_type, param_value_type) in enumerate(runtime_parameters):
            if entry_point_uses_void_signature:
                continue
            if hasattr(param, "name"):
                param_name = param.name
            else:
                param_name = f"param{i}"

            param_id = self.create_function_parameter(param_type, param_name)
            self.local_variables[param_name] = param_id
            if i in resource_array_param_indices:
                self.variable_value_types[param_id.id] = param_value_type
            self.register_declared_resource_metadata(param, param_id, param_value_type)

        self.begin_block()

        if not entry_point_uses_void_signature:
            for i, (param, _param_type, param_value_type) in enumerate(
                runtime_parameters
            ):
                if i not in value_array_param_indices:
                    continue
                param_name = getattr(param, "name", f"param{i}")
                param_id = self.local_variables.get(param_name)
                if param_id is None:
                    continue
                local_copy = self.copy_array_pointer_to_function_storage(
                    param_id, param_value_type, name=param_name
                )
                if local_copy is not None:
                    self.local_variables[param_name] = local_copy

        previous_execution_model = self.current_execution_model
        previous_function_id = self.current_function_id
        previous_mesh_output_parameters = self.current_mesh_output_parameters
        self.current_function_id = function_id.id
        if execution_model_hint is not None:
            self.current_execution_model = execution_model_hint
        elif has_mesh_output_parameters:
            self.current_execution_model = "MeshEXT"
        elif self.current_execution_model is None:
            self.current_execution_model = self.execution_model_hint_for_function(
                function_node
            )
        self.current_mesh_output_parameters = mesh_output_parameters
        if has_direct_return_semantic:
            self.current_return_semantic_output = self.register_direct_return_output(
                function_node, return_semantic, return_type
            )
        elif entry_point_uses_void_signature and return_type.type.base_type != "void":
            self.current_entry_point_return_outputs = (
                self.register_entry_point_return_outputs(
                    function_node, return_type, execution_model_hint
                )
            )
        if entry_point_uses_void_signature:
            for param, param_value_type in output_parameters:
                param_name = getattr(param, "name", None) or "param"
                variable = self.register_entry_point_interface_variable(
                    execution_model_hint,
                    "Output",
                    self.entry_point_interface_name(
                        execution_model_hint, "output", param_name
                    ),
                    param_value_type,
                    param,
                )
                self.local_variables[param_name] = variable
                self.register_declared_resource_metadata(
                    param, variable, param_value_type
                )
        for param, patch_info in patch_interface_parameters:
            patch_variable = self.register_patch_parameter_interface_variable(
                param, patch_info
            )
            self.local_variables[patch_info["name"]] = patch_variable
            self.mark_function_interface_variable(patch_variable)

        if entry_point_uses_void_signature:
            self.initialize_entry_point_parameters(
                runtime_parameters, execution_model_hint
            )

        self.process_statements(function_node.body)
        self.process_tessellation_patch_constant_function(function_node)

        if not self.current_block_has_terminator():
            if (
                entry_point_uses_void_signature
                or self.convert_type_node_to_string(function_node.return_type) == "void"
            ):
                self.create_return()
            else:
                self.create_unreachable()

        self.end_function()

        self.current_execution_model = previous_execution_model
        self.current_function_id = previous_function_id
        self.current_mesh_output_parameters = previous_mesh_output_parameters
        self.current_function_name = previous_function_name
        self.current_stage = previous_stage
        self.current_return_type = previous_return_type
        self.current_return_type_source = previous_return_type_source
        self.current_return_semantic_output = previous_return_semantic_output
        self.current_entry_point_return_outputs = previous_entry_point_return_outputs
        self.current_generic_function_substitutions = (
            previous_generic_function_substitutions
        )
        self.local_variables.clear()
        self.precise_local_variables.clear()
        self.resource_alias_variables.clear()
        return function_id

    def process_statements(self, statements):
        """Process a list of CrossGL statements."""
        if hasattr(statements, "statements"):
            stmt_list = statements.statements
        elif isinstance(statements, list):
            stmt_list = statements
        else:
            stmt_list = [statements]

        for stmt in stmt_list:
            if self.current_block_has_terminator():
                break
            self.process_statement(stmt)

    def process_statement(self, stmt):
        """Process a single CrossGL statement."""
        if isinstance(stmt, AssignmentNode):
            self.process_assignment(stmt)
        elif isinstance(stmt, VariableNode):
            self.process_variable_declaration(stmt)
        elif isinstance(stmt, ReturnNode):
            self.process_return(stmt)
        elif isinstance(stmt, IfNode):
            self.process_if(stmt)
        elif isinstance(stmt, ForNode):
            self.process_for(stmt)
        elif isinstance(stmt, ForInNode):
            self.process_for_in(stmt)
        elif isinstance(stmt, WhileNode):
            self.process_while(stmt)
        elif isinstance(stmt, DoWhileNode):
            self.process_do_while(stmt)
        elif isinstance(stmt, LoopNode):
            self.process_loop(stmt)
        elif isinstance(stmt, SwitchNode):
            self.process_switch(stmt)
        elif isinstance(stmt, MatchNode):
            self.process_match(stmt)
        elif isinstance(stmt, BreakNode):
            self.process_break(stmt)
        elif isinstance(stmt, ContinueNode):
            self.process_continue(stmt)
        elif isinstance(stmt, FunctionCallNode):
            if self.process_fragment_discard_statement(stmt):
                return
            self.process_expression(stmt)  # Just evaluate and discard result
        elif isinstance(stmt, (UnaryOpNode, BinaryOpNode)):
            self.process_expression(stmt)
        elif hasattr(stmt, "expression"):
            expression = stmt.expression
            if self.process_fragment_discard_statement(expression):
                return
            if isinstance(expression, AssignmentNode):
                self.process_assignment(expression)
            elif (
                getattr(stmt, "is_tail_expression", False)
                and self.current_return_type is not None
                and self.current_return_type.type.base_type != "void"
            ):
                return_value = self.process_expression_with_expected_type(
                    expression, self.current_return_type_source
                )
                if return_value is not None:
                    self.create_return_value(return_value)
                else:
                    self.create_return()
            else:
                self.process_expression(expression)

    def process_fragment_discard_statement(self, expr) -> bool:
        """Lower CrossGL/GLSL/HLSL fragment discard statements to SPIR-V."""
        if isinstance(expr, IdentifierNode) and expr.name == "discard":
            return self.emit_fragment_discard("discard")
        if isinstance(expr, str) and expr == "discard":
            return self.emit_fragment_discard("discard")
        if not isinstance(expr, FunctionCallNode):
            return False

        callee_name = self.function_call_name(expr)
        if not isinstance(callee_name, str):
            return False

        args = list(getattr(expr, "arguments", getattr(expr, "args", [])) or [])
        if (
            callee_name == "discard"
            and not args
            and not self.has_function_reference(callee_name)
        ):
            return self.emit_fragment_discard("discard")

        if callee_name != "clip" or self.has_function_reference(callee_name):
            return False
        if len(args) != 1:
            self.emit("; WARNING: SPIR-V clip discard requires exactly one operand")
            return True

        return self.process_clip_discard_statement(args[0])

    def emit_fragment_discard(self, source_name: str) -> bool:
        """Emit OpKill for fragment discard, preserving SPIR-V stage validity."""
        if self.current_execution_model != "Fragment":
            self.emit(
                f"; WARNING: SPIR-V {source_name} lowers to OpKill, which is "
                "only valid in the Fragment execution model"
            )
            return True

        self.emit("OpKill")
        return True

    def process_clip_discard_statement(self, value_expr) -> bool:
        """Lower HLSL-style clip(x) to conditional fragment discard."""
        if self.current_execution_model != "Fragment":
            self.emit(
                "; WARNING: SPIR-V clip lowers to OpKill, which is only valid "
                "in the Fragment execution model"
            )
            return True

        value = self.process_expression(value_expr)
        if value is None:
            self.emit("; WARNING: SPIR-V clip operand could not be evaluated")
            return True

        condition = self.clip_discard_condition(value)
        if condition is None:
            self.emit(
                "; WARNING: SPIR-V clip requires a numeric scalar or vector operand"
            )
            return True

        merge_label = SpirvId(self.get_id(), SpirvType("label"))
        kill_label = SpirvId(self.get_id(), SpirvType("label"))
        self.create_selection_merge(merge_label)
        self.create_conditional_branch(condition, kill_label, merge_label)

        self.emit(f"%{kill_label.id} = OpLabel")
        self.current_label = kill_label.id
        self.emit("OpKill")

        self.emit(f"%{merge_label.id} = OpLabel")
        self.current_label = merge_label.id
        return True

    def clip_discard_condition(self, value: SpirvId) -> Optional[SpirvId]:
        """Return the scalar bool condition for HLSL clip discard semantics."""
        value_type = self.registered_value_type(value) or self.ensure_registered_type(
            value.type
        )
        type_name = value_type.type.base_type
        vector_info = self.vector_component_type_and_count(type_name)
        if vector_info is not None:
            component_type_name, component_count = vector_info
            if component_type_name == "bool":
                return None
            component_type = self.register_primitive_type(component_type_name)
            zero = self.clip_zero_value(component_type_name, component_type)
            zero_vector = self.splat_scalar_to_vector(zero, value_type)
            bool_type = self.register_primitive_type("bool")
            bool_vector_type = self.register_vector_type(bool_type, component_count)
            condition_vector = self.binary_operation(
                "<", bool_vector_type, value, zero_vector
            )
            return self.any_bool_vector_operation(condition_vector)

        scalar_type_name = self.normalize_primitive_name(type_name)
        if scalar_type_name not in {"float", "double"} | self.INTEGER_TYPE_NAMES:
            return None

        scalar_type = self.register_primitive_type(scalar_type_name)
        value = self.convert_value_to_type(value, scalar_type)
        zero = self.clip_zero_value(scalar_type_name, scalar_type)
        bool_type = self.register_primitive_type("bool")
        return self.binary_operation("<", bool_type, value, zero)

    def clip_zero_value(self, type_name: str, type_id: SpirvId) -> SpirvId:
        """Return a zero constant compatible with a clip operand component."""
        if type_name in {"float", "double"}:
            return self.register_constant(0.0, type_id)
        return self.register_constant(0, type_id)

    def infer_expression_result_type(self, expr) -> Optional[SpirvId]:
        if expr is None:
            return None
        if isinstance(expr, bool):
            return self.register_primitive_type("bool")
        if isinstance(expr, int):
            return self.register_primitive_type("int")
        if isinstance(expr, float):
            return self.register_primitive_type("float")
        if isinstance(expr, LiteralNode):
            literal_type = self.convert_type_node_to_string(expr.literal_type)
            primitive_type = self.normalize_primitive_name(literal_type)
            if primitive_type in self.INTEGER_TYPE_NAMES:
                primitive_type = self.integer_literal_type_for_value(
                    primitive_type, int(expr.value)
                )
                return self.register_primitive_type(primitive_type)
            return self.map_crossgl_type(expr.literal_type)
        if isinstance(expr, (IdentifierNode, VariableNode, str)):
            name = expr if isinstance(expr, str) else expr.name
            local_type_name = self.local_variable_types.get(name)
            if local_type_name is not None:
                return self.map_resource_type_with_format(local_type_name)
            variable = self.local_variables.get(name) or self.resolve_global_variable(
                name
            )
            if variable is not None:
                wrapped_type = self.uniform_block_wrapped_member_type(variable)
                if wrapped_type is not None:
                    return wrapped_type
                return self.variable_value_types.get(
                    variable.id
                ) or self.value_types.get(variable.id)
            builtin_type = self.infer_builtin_expression_result_type(name)
            if builtin_type is not None:
                return builtin_type
            return None
        if isinstance(expr, FunctionCallNode):
            callee_name = self.function_call_name(expr)
            if callee_name is None:
                return None
            vector_info = self.vector_component_type_and_count(callee_name)
            if vector_info is not None:
                component_type, component_count = vector_info
                return self.register_vector_type(
                    self.register_primitive_type(component_type), component_count
                )
            matrix_type_name = self.normalize_hlsl_matrix_type(callee_name)
            if re.fullmatch(r"(d)?mat([234])(?:x([234]))?", matrix_type_name):
                return self.map_crossgl_type(matrix_type_name)
            primitive_name = self.normalize_primitive_name(callee_name)
            if primitive_name in {
                "bool",
                "int",
                "uint",
                "i64",
                "u64",
                "float",
                "double",
            }:
                return self.register_primitive_type(primitive_name)
            specialized_callee_name = generic_function_call_name(
                self, callee_name, expr.args
            )
            if specialized_callee_name is not None:
                signature = self.resolve_function_signature(specialized_callee_name)
                if signature is not None:
                    return signature[0]
                for specialized_func in (
                    self.generic_function_specializations or {}
                ).values():
                    if (
                        getattr(specialized_func, "name", None)
                        == specialized_callee_name
                    ):
                        return self.map_crossgl_type(
                            getattr(specialized_func, "return_type", "void")
                        )
            signature = self.resolve_function_signature(callee_name)
            if signature is not None:
                return signature[0]
            if callee_name in self.struct_types:
                return self.struct_types[callee_name]
            bitcast_type = self.infer_bitcast_builtin_result_type(
                callee_name, expr.args
            )
            if bitcast_type is not None:
                return bitcast_type
            integer_bit_type = self.infer_integer_bit_builtin_result_type(
                callee_name, expr.args
            )
            if integer_bit_type is not None:
                return integer_bit_type
            return None
        if isinstance(expr, ArrayAccessNode):
            array_type = self.infer_expression_result_type(expr.array)
            if array_type is None:
                return None
            element_type = self.array_element_type_from_type(array_type)
            if element_type is not None:
                return element_type
            vector_info = self.vector_type_info_from_type(array_type)
            if vector_info is not None:
                return vector_info[0]
            matrix_info = self.matrix_type_info_from_type(array_type)
            if matrix_info is not None:
                return matrix_info[0]
            return None
        if isinstance(expr, MemberAccessNode):
            base_type = self.infer_expression_result_type(expr.object)
            if base_type is None:
                return None
            member_info = self.struct_member_info(base_type.type.base_type, expr.member)
            if member_info is not None:
                return member_info[1]
            vector_member = self.vector_member_info(
                base_type.type.base_type, expr.member
            )
            if vector_member is not None:
                return vector_member[1]
            swizzle_info = self.vector_swizzle_info(
                base_type.type.base_type, expr.member
            )
            if swizzle_info is not None:
                return swizzle_info[2]
            return None
        if isinstance(expr, BinaryOpNode):
            left_type = self.infer_expression_result_type(expr.left)
            right_type = self.infer_expression_result_type(expr.right)
            return self.binary_expression_result_type(expr.op, left_type, right_type)
        if isinstance(expr, TernaryOpNode):
            true_type = self.infer_expression_result_type(expr.true_expr)
            false_type = self.infer_expression_result_type(expr.false_expr)
            if true_type is not None and false_type is not None:
                return self.ternary_result_type(
                    SpirvId(0, true_type.type), SpirvId(0, false_type.type)
                )
            return true_type or false_type
        if isinstance(expr, MatchNode):
            return self.infer_match_expression_result_type(expr)
        return None

    def infer_builtin_expression_result_type(self, name: str) -> Optional[SpirvId]:
        builtin_type = self.infer_builtin_variable_result_type(name)
        if builtin_type is not None:
            return builtin_type

        if not isinstance(name, str) or "." not in name:
            return None

        base_name, member_name = name.rsplit(".", 1)
        base_type = self.infer_builtin_variable_result_type(base_name)
        if base_type is None:
            return None

        member_info = self.vector_member_info(base_type.type.base_type, member_name)
        if member_info is not None:
            return member_info[1]

        swizzle_info = self.vector_swizzle_info(base_type.type.base_type, member_name)
        if swizzle_info is not None:
            return swizzle_info[2]

        return None

    def infer_builtin_variable_result_type(self, name: str) -> Optional[SpirvId]:
        compute_info = self.compute_builtin_info(name)
        if compute_info is not None:
            return self.map_crossgl_type(compute_info[0])

        stage_info = self.stage_builtin_info(name)
        if stage_info is not None:
            return self.map_crossgl_type(stage_info[0])

        return None

    def glsl_builtin_limit_constant(self, name: str) -> Optional[SpirvId]:
        limits = {
            "gl_MaxImageUnits": 8,
        }
        value = limits.get(name)
        if value is None:
            return None
        return self.register_constant(value, self.register_primitive_type("int"))

    def infer_match_expression_result_type(self, node: MatchNode) -> Optional[SpirvId]:
        for arm in getattr(node, "arms", []) or []:
            expression = self.match_arm_tail_expression(getattr(arm, "body", None))
            result_type = self.infer_expression_result_type(expression)
            if result_type is not None:
                return result_type
        return None

    def match_arm_tail_expression(self, body):
        statements = getattr(body, "statements", None)
        if statements is not None:
            if statements and getattr(statements[-1], "is_tail_expression", False):
                return getattr(statements[-1], "expression", None)
            return None
        if hasattr(body, "expression"):
            return getattr(body, "expression", None)
        return body

    def process_variable_declaration(self, node: VariableNode):
        """Process a local CrossGL variable declaration."""
        var_type_source = getattr(node, "var_type", getattr(node, "vtype", "float"))
        if self.type_name_from_value(var_type_source) == "auto":
            var_type_source = None
        var_type_name = self.type_name_from_value(var_type_source)
        if self.process_local_resource_alias_declaration(
            node, var_type_source, var_type_name
        ):
            return

        if self.local_variable_requires_descriptor_storage(node, var_type_name):
            raise ValueError(
                f"SPIR-V descriptor resource '{node.name}' cannot be declared "
                "inside a function; declare it at shader or stage scope"
            )

        initial_value = getattr(node, "initial_value", None)
        is_precise = self.has_attribute(node, "precise")
        if (
            var_type_source is None
            and initial_value is not None
            and not isinstance(initial_value, (ArrayLiteralNode, MatchNode))
        ):
            rhs_value = self.process_expression_with_precision(
                initial_value, is_precise
            )
            if rhs_value is not None:
                var_type = self.ensure_registered_type(rhs_value.type)
                var_id = self.create_variable(var_type, "Function", node.name)
                self.local_variables[node.name] = var_id
                if is_precise:
                    self.precise_local_variables.add(node.name)
                self.store_to_variable(var_id, rhs_value)
                return

        if var_type_source is None and isinstance(initial_value, MatchNode):
            var_type = self.infer_match_expression_result_type(initial_value)
            if var_type is None:
                var_type = self.map_resource_type_with_format(var_type_source, node)
        else:
            var_type = self.map_resource_type_with_format(var_type_source, node)
        if self.type_contains_runtime_array(var_type):
            self.emit(
                f"; WARNING: local variable {node.name} has a runtime-array "
                "aggregate type that cannot be materialized in SPIR-V"
            )
            return

        var_id = self.create_variable(var_type, "Function", node.name)
        self.local_variables[node.name] = var_id
        if is_precise:
            self.precise_local_variables.add(node.name)

        if initial_value is not None:
            if isinstance(initial_value, MatchNode):
                self.process_match_expression_assignment(
                    initial_value, var_id, var_type
                )
            elif isinstance(initial_value, ArrayLiteralNode):
                rhs_value = self.process_array_literal(initial_value, var_type)
                if rhs_value is not None:
                    self.store_to_variable(var_id, rhs_value)
            else:
                if is_precise:
                    rhs_value = self.process_expression_with_precision(
                        initial_value, is_precise
                    )
                else:
                    rhs_value = self.process_expression_with_expected_type(
                        initial_value, var_type_name
                    )
                if rhs_value is not None:
                    self.store_to_variable(var_id, rhs_value)

    def process_local_resource_alias_declaration(
        self, node: VariableNode, var_type_source, var_type_name: str
    ) -> bool:
        initial_value = getattr(node, "initial_value", None)
        if initial_value is None:
            return False

        source_pointer = self.variable_pointer_from_expression(initial_value)
        if source_pointer is None:
            return False

        source_metadata = self.resource_metadata_for_pointer(source_pointer)
        if source_metadata is None:
            return False

        if var_type_source is not None:
            base_type_name = self.array_base_type_name(var_type_name)
            if not (
                self.is_resource_type_name(base_type_name)
                or self.is_acceleration_structure_type_name(base_type_name)
            ):
                return False

        self.local_variables[node.name] = source_pointer
        self.resource_alias_variables.add(node.name)
        return True

    def local_variable_requires_descriptor_storage(
        self, node: VariableNode, type_name: str
    ) -> bool:
        if not type_name:
            return False
        base_type = self.array_base_type_name(type_name)
        return (
            self.is_resource_type_name(base_type)
            or self.is_acceleration_structure_type_name(base_type)
            or self.is_structured_buffer_type_name(base_type)
            or self.is_glsl_buffer_block_node(node)
        )

    def global_interface_builtin_variable(
        self, node: VariableNode, storage_class: str, type_id: SpirvId
    ) -> Optional[SpirvId]:
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
                semantic, type_id.type.base_type, "interface declaration"
            )
        return builtin

    def process_global_variable_declaration(
        self, node: VariableNode, default_storage_class: str = "Private"
    ) -> SpirvId:
        """Process a module-scope CrossGL variable declaration."""
        var_type_source = getattr(node, "var_type", getattr(node, "vtype", "float"))
        var_type_name = self.type_name_from_value(var_type_source)
        if self.is_glsl_buffer_block_node(node):
            return self.process_glsl_buffer_block_declaration(node, var_type_name)

        if self.is_structured_buffer_declared_type_name(var_type_name):
            return self.process_structured_buffer_declaration(node, var_type_name)

        var_type = self.map_resource_type_with_format(var_type_source, node)
        storage_class = self.infer_global_storage_class(
            node, default_storage_class, var_type_name
        )
        if storage_class == "Uniform" and self.uniform_array_requires_block(var_type):
            return self.process_uniform_array_block_declaration(node, var_type)
        if (
            storage_class == "Uniform"
            and self.current_struct_members.get(var_type.type.base_type) is not None
        ):
            var_type = self.storage_layout_type(var_type, "std140")

        initializer = None
        initial_value = getattr(node, "initial_value", None)
        if storage_class == "Private" and initial_value is not None:
            initializer = self.process_constant_expression(initial_value, var_type)

        if storage_class == "Input":
            var_id = self.global_interface_builtin_variable(node, "Input", var_type)
            if var_id is None:
                location = self.global_interface_location(node, "Input")
                var_id = self.register_input(
                    node.name, var_type, location, 0, source_node=node
                )
                self.decorate_global_interface_variable(node, var_id)
        elif storage_class == "Output":
            var_id = self.global_interface_builtin_variable(node, "Output", var_type)
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
        if self.has_attribute(node, "precise"):
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

    def resource_descriptor_slot(self, node: VariableNode) -> Tuple[int, int]:
        descriptor_set = self.resource_descriptor_set(node)
        explicit_binding = self.explicit_descriptor_binding(node)

        if explicit_binding is not None:
            key = (descriptor_set, explicit_binding)
            if key in self.used_resource_bindings:
                raise ValueError(
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

    def reserve_explicit_resource_bindings(self, ast: ShaderNode):
        for node in self.global_descriptor_binding_nodes(ast):
            explicit_binding = self.explicit_descriptor_binding(node)
            if explicit_binding is None:
                continue

            descriptor_set = self.resource_descriptor_set(node)
            key = (descriptor_set, explicit_binding)
            if key in self.reserved_resource_bindings:
                raise ValueError(
                    f"Duplicate SPIR-V resource binding set {descriptor_set} "
                    f"binding {explicit_binding}"
                )
            self.reserved_resource_bindings.add(key)

    def explicit_descriptor_binding(self, node: VariableNode) -> Optional[int]:
        binding = self.explicit_interface_integer_attribute(node, "binding")
        if binding is not None:
            return binding
        return self.explicit_interface_integer_attribute(node, "buffer")

    def global_descriptor_binding_nodes(self, ast: ShaderNode):
        yield from self.global_resource_nodes(ast)
        yield from self.global_structured_buffer_nodes(ast)
        yield from getattr(ast, "cbuffers", []) or []

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
                raise ValueError(
                    f"Duplicate SPIR-V {storage_class.lower()} location "
                    f"{explicit_location}"
                )
            used_slots.update(slot_keys)
            return explicit_location

        if preferred_location is not None:
            slot_keys = self.interface_slot_keys(
                node, storage_class, preferred_location, patch=patch
            )
            if used_slots & slot_keys:
                raise ValueError(
                    f"Duplicate SPIR-V {storage_class.lower()} location "
                    f"{preferred_location}"
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
            raise ValueError(f"SPIR-V component must be in 0..3: {component}")
        return component

    def interface_location_span(self, node: VariableNode) -> int:
        return max(len(self.interface_location_component_widths(node)), 1)

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
                raise ValueError(
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
            raise ValueError(
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
            self.emit(
                f"; WARNING: SPIR-V tessellation patch type {context} requires a "
                "positive integer control-point count; using 1"
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
            self.emit(
                f"; WARNING: SPIR-V {patch_info['patch_type']} parameter "
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
            self.emit(
                "; WARNING: SPIR-V tessellation_control patchconstantfunc "
                "requires exactly one function name"
            )
            return None

        function_name = self.attribute_value_to_string(arguments[0])
        if not function_name:
            self.emit(
                "; WARNING: SPIR-V tessellation_control patchconstantfunc "
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
                    self.emit(
                        "; WARNING: SPIR-V tessellation_control patchconstantfunc "
                        f"parameter {param_name} must use InputPatch<T, N>"
                    )
                    arguments.append(self.default_value_for_type(param_type))
                    continue

                source_variable = self.matching_patch_interface_variable(patch_info)
                if source_variable is None:
                    self.emit(
                        "; WARNING: SPIR-V tessellation_control patchconstantfunc "
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

            self.emit(
                "; WARNING: SPIR-V tessellation_control patchconstantfunc "
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
            for index in range(size or 0):
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
            self.emit(
                "; WARNING: SPIR-V tessellation_control patchconstantfunc "
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

    def register_tessellation_patch_constant_interface_variable(
        self,
        name: str,
        type_id: SpirvId,
        storage_class: str,
        source_node,
        preferred_location: Optional[int] = None,
    ):
        self.require_capability("Tessellation")
        self.validate_user_defined_interface_type(
            type_id, storage_class, name, source_node
        )
        variable = self.create_variable(type_id, storage_class, name)
        location = self.global_interface_location(
            source_node, storage_class, preferred_location, patch=True
        )
        self.decorations.append(f"OpDecorate %{variable.id} Location {location}")
        self.decorations.append(f"OpDecorate %{variable.id} Patch")
        self.decorate_global_interface_variable(source_node, variable)
        if storage_class == "Input":
            self.readonly_pointer_names[variable.id] = name
        self.mark_function_interface_variable(variable)
        return variable, location

    def store_tessellation_patch_constant_user_member(
        self,
        member_name: str,
        member_type: SpirvId,
        member_value: SpirvId,
        metadata: dict,
    ) -> bool:
        interface_key = self.tessellation_patch_constant_user_member_key(
            member_name, metadata
        )
        if interface_key is None:
            return False

        source_node = metadata.get("node")
        if source_node is None:
            return False

        variable = self.tessellation_patch_constant_output_variables.get(interface_key)
        if variable is None:
            variable, location = (
                self.register_tessellation_patch_constant_interface_variable(
                    member_name, member_type, "Output", source_node
                )
            )
            self.tessellation_patch_constant_output_variables[interface_key] = variable
            self.tessellation_patch_constant_interfaces[interface_key] = {
                "location": location,
                "name": member_name,
                "node": source_node,
                "type": member_type,
            }

        self.store_to_variable(
            variable, self.convert_value_to_type(member_value, member_type)
        )
        return True

    def ensure_tessellation_patch_constant_input(self, name: str) -> Optional[SpirvId]:
        if self.current_execution_model != "TessellationEvaluation":
            return None

        metadata = self.tessellation_patch_constant_interfaces.get(name)
        if metadata is None:
            return None

        variable = self.tessellation_patch_constant_input_variables.get(name)
        if variable is not None:
            self.mark_function_interface_variable(variable)
            return variable

        variable, _location = (
            self.register_tessellation_patch_constant_interface_variable(
                metadata["name"],
                metadata["type"],
                "Input",
                metadata["node"],
                metadata["location"],
            )
        )
        self.tessellation_patch_constant_input_variables[name] = variable
        return variable

    def store_tessellation_patch_constant_semantics(
        self, result: SpirvId, result_type: SpirvId, function_name: str
    ):
        struct_name = result_type.type.base_type
        members = self.current_struct_members.get(struct_name, [])
        metadata_by_member = self.struct_member_metadata.get(struct_name, {})
        stored_any = False

        for member_index, (member_type, member_name) in enumerate(members):
            metadata = metadata_by_member.get(member_name, {})
            builtin_name = self.tessellation_patch_constant_builtin_name(
                metadata.get("semantic")
            )
            member_value = self.composite_extract(result, member_type, member_index)

            if builtin_name is not None:
                for (
                    component_index,
                    component,
                ) in self.tessellation_patch_constant_member_components(
                    member_value, member_type
                ):
                    self.store_tessellation_patch_constant_builtin_component(
                        builtin_name, component_index, component
                    )
                    stored_any = True
                continue

            self.store_tessellation_patch_constant_user_member(
                member_name, member_type, member_value, metadata
            )

        if not stored_any:
            self.emit(
                "; WARNING: SPIR-V tessellation_control patchconstantfunc "
                f"'{function_name}' returned no tessellation factor semantics"
            )

    def process_tessellation_patch_constant_function(self, function_node):
        if self.current_execution_model != "TessellationControl":
            return
        if not self.function_is_entry_point(function_node, self.current_stage):
            return
        if self.current_block_has_terminator():
            return

        function_name = self.tessellation_patch_constant_function_name(function_node)
        if function_name is None:
            return
        function_signature = self.resolve_function_signature(function_name)
        if function_signature is None:
            self.emit(
                "; WARNING: SPIR-V tessellation_control patchconstantfunc "
                f"'{function_name}' does not reference a generated function"
            )
            return

        return_type, _param_types = function_signature
        if return_type.type.base_type == "void":
            self.emit(
                "; WARNING: SPIR-V tessellation_control patchconstantfunc "
                f"'{function_name}' requires a non-void return type"
            )
            return

        patch_function = self.function_nodes.get(function_name)
        if patch_function is None:
            self.emit(
                "; WARNING: SPIR-V tessellation_control patchconstantfunc "
                f"'{function_name}' has no available function definition"
            )
            return

        arguments = self.tessellation_patch_constant_call_arguments(patch_function)
        result = self.call_function(function_name, arguments)
        if result is None:
            return
        self.store_tessellation_patch_constant_semantics(
            result, return_type, function_name
        )

    def register_patch_parameter_interface_variable(self, param, patch_info: dict):
        self.require_capability("Tessellation")
        element_type = self.map_crossgl_type(patch_info["element_type_name"])
        array_type = self.register_array_type(
            element_type, patch_info["control_points"]
        )
        storage_class = patch_info["storage_class"]
        self.validate_user_defined_interface_type(
            array_type, storage_class, patch_info["name"], param
        )
        variable = self.create_variable(array_type, storage_class, patch_info["name"])
        location = self.global_interface_location(
            param,
            storage_class,
            self.matching_tessellation_output_patch_location(patch_info),
        )
        self.decorations.append(f"OpDecorate %{variable.id} Location {location}")
        self.decorate_global_interface_variable(param, variable)
        self.patch_parameter_metadata[variable.id] = patch_info
        self.record_tessellation_output_patch_location(patch_info, location)
        if storage_class == "Input":
            self.readonly_pointer_names[variable.id] = patch_info["name"]
        return variable

    def tessellation_patch_location_key(self, patch_info: dict) -> tuple:
        return (patch_info.get("element_type_name"), patch_info.get("control_points"))

    def record_tessellation_output_patch_location(
        self, patch_info: dict, location: int
    ):
        if (
            self.current_execution_model != "TessellationControl"
            or patch_info.get("patch_type") != "OutputPatch"
            or patch_info.get("storage_class") != "Output"
        ):
            return
        self.tessellation_output_patch_locations[
            self.tessellation_patch_location_key(patch_info)
        ] = location

    def matching_tessellation_output_patch_location(
        self, patch_info: dict
    ) -> Optional[int]:
        if (
            self.current_execution_model != "TessellationEvaluation"
            or patch_info.get("patch_type") != "OutputPatch"
            or patch_info.get("storage_class") != "Input"
        ):
            return None
        return self.tessellation_output_patch_locations.get(
            self.tessellation_patch_location_key(patch_info)
        )

    def mesh_output_parameter_info(self, param, execution_model: Optional[str]):
        if execution_model != "MeshEXT" or not self.is_mesh_output_parameter(param):
            return None

        param_name = getattr(param, "name", None)
        if not param_name:
            return None

        type_name = self.type_name_from_value(
            getattr(param, "param_type", getattr(param, "vtype", None))
        )
        array_info = self.split_outer_array_type(type_name)
        if array_info is None:
            self.emit(
                f"; WARNING: SPIR-V mesh output parameter {param_name} must be "
                "an array type"
            )
            element_type_name = type_name
            element_count = None
        else:
            element_type_name, element_count = array_info

        element_type = self.map_crossgl_type(element_type_name)
        members = {}
        for index, (member_type, member_name) in enumerate(
            self.current_struct_members.get(element_type_name, [])
        ):
            metadata = self.struct_member_metadata.get(element_type_name, {}).get(
                member_name, {}
            )
            members[member_name] = {
                "index": index,
                "type": member_type,
                "semantic": metadata.get("semantic"),
            }

        return {
            "name": param_name,
            "role": self.mesh_output_parameter_role(param),
            "element_type_name": element_type_name,
            "element_type": element_type,
            "count": element_count,
            "members": members,
        }

    def validate_interface_interpolation_attributes(self, node: VariableNode):
        for first, second in (("flat", "noperspective"), ("centroid", "sample")):
            if self.has_attribute(node, first) and self.has_attribute(node, second):
                raise ValueError(
                    "SPIR-V interpolation attributes "
                    f"@{first} and @{second} cannot be combined"
                )

    def interface_slot_keys(
        self,
        node: VariableNode,
        storage_class: str,
        location: int,
        patch: bool = False,
    ) -> set:
        component = self.explicit_component_attribute(node)
        component_start = component if component is not None else 0
        index = self.explicit_interface_integer_attribute(node, "index") or 0
        scope = self.interface_location_scope()
        patch_key = "patch" if patch else "per_vertex"
        slots = set()

        for offset, component_width in enumerate(
            self.interface_location_component_widths(node)
        ):
            if component_start + component_width > 4:
                raise ValueError(
                    f"SPIR-V component range overflows location {location + offset}: "
                    f"{component_start}..{component_start + component_width - 1}"
                )

            slots.update(
                (scope, patch_key, location + offset, index, component)
                for component in range(
                    component_start, component_start + component_width
                )
            )

        return slots

    def interface_component_width(self, node: VariableNode) -> int:
        widths = self.interface_location_component_widths(node)
        if widths:
            return widths[0]
        return 4

    def interface_location_component_widths(self, node: VariableNode) -> List[int]:
        type_source = getattr(node, "var_type", getattr(node, "vtype", "float"))
        type_source = getattr(node, "member_type", type_source)
        type_name = self.type_name_from_value(type_source)
        return self.interface_type_location_component_widths(type_name)

    def interface_type_location_component_widths(self, type_name: str) -> List[int]:
        if type_name is None:
            return [4]

        type_name = self.normalize_generic_vector_type(type_name)
        type_name = self.normalize_hlsl_matrix_type(type_name)
        array_type = self.split_outer_array_type(type_name)
        if array_type is not None:
            element_type_name, size = array_type
            element_widths = self.interface_type_location_component_widths(
                element_type_name
            )
            element_count = max(size or 1, 1)
            return element_widths * element_count

        matrix_match = re.fullmatch(r"(d)?mat([234])(?:x([234]))?", type_name)
        if matrix_match:
            _is_double, cols, rows = matrix_match.groups()
            return [int(rows or cols)] * int(cols)

        vector_info = self.vector_component_type_and_count(type_name)
        if vector_info is not None:
            _, component_count = vector_info
            return [component_count]

        if self.normalize_primitive_name(type_name) in {
            "float",
            "double",
            "int",
            "uint",
            "bool",
        }:
            return [1]

        return [4]

    def infer_global_storage_class(
        self, node: VariableNode, default_storage_class: str, type_name: str = None
    ) -> str:
        attribute_names = {
            getattr(attribute, "name", "").lower()
            for attribute in getattr(node, "attributes", [])
        }
        qualifiers = {
            str(qualifier).lower() for qualifier in getattr(node, "qualifiers", [])
        }

        if self.is_task_payload_shared_node(node):
            return "TaskPayloadWorkgroupEXT"
        if attribute_names & {"input", "in"} or qualifiers & {"input", "in"}:
            return "Input"
        if attribute_names & {"output", "out"} or qualifiers & {"output", "out"}:
            return "Output"
        if type_name:
            base_type_name, _ = parse_array_type(type_name)
            if self.is_resource_type_name(base_type_name):
                return "UniformConstant"
        if attribute_names & {"uniform"} or qualifiers & {"uniform"}:
            base_type_name, _ = parse_array_type(type_name)
            if base_type_name in self.struct_types:
                return "Uniform"
        return default_storage_class

    def type_name_from_value(self, type_value) -> str:
        if type_value is None:
            return None
        if self.is_type_node_like(type_value):
            type_name = self.convert_type_node_to_string(type_value)
        else:
            type_name = str(type_value)

        substitutions = {}
        substitutions.update(
            getattr(self, "current_generic_type_substitutions", {}) or {}
        )
        substitutions.update(
            getattr(self, "current_generic_function_substitutions", {}) or {}
        )
        if substitutions:
            if type_name.startswith("&"):
                referenced = substitute_generic_type_name(
                    type_name[1:].strip(), substitutions
                )
                return f"&{referenced}"
            if type_name.endswith("*"):
                pointee = substitute_generic_type_name(
                    type_name[:-1].strip(), substitutions
                )
                return f"{pointee}*"
            return substitute_generic_type_name(type_name, substitutions)
        return type_name

    def type_name_string(self, type_value) -> str:
        return self.type_name_from_value(type_value)

    def expression_result_type(self, expr):
        inferred_type = self.infer_expression_result_type(expr)
        if inferred_type is None:
            return None
        return inferred_type.type.base_type

    def format_source_location(self, source_location) -> Optional[str]:
        if source_location is None:
            return None
        if isinstance(source_location, dict):
            line = source_location.get("line") or source_location.get("lineno")
            column = source_location.get("column") or source_location.get("col")
            if line is not None and column is not None:
                return f"line {line}, column {column}"
            if line is not None:
                return f"line {line}"
        if isinstance(source_location, (list, tuple)):
            if len(source_location) >= 2:
                return f"line {source_location[0]}, column {source_location[1]}"
            if len(source_location) == 1:
                return f"line {source_location[0]}"
        line = getattr(source_location, "line", None) or getattr(
            source_location, "lineno", None
        )
        column = getattr(source_location, "column", None) or getattr(
            source_location, "col", None
        )
        if line is not None and column is not None:
            return f"line {line}, column {column}"
        if line is not None:
            return f"line {line}"
        return str(source_location)

    def unsupported_generic_function_error(self, function_node, call_node=None):
        helper_name = getattr(function_node, "name", "<unknown>")
        generic_params = generic_function_parameters(function_node)
        params_label = ", ".join(generic_params) if generic_params else "<none>"
        source_location = getattr(call_node, "source_location", None) or getattr(
            function_node, "source_location", None
        )
        location_label = self.format_source_location(source_location)
        location_suffix = f" at {location_label}" if location_label else ""
        return UnsupportedSPIRVFeatureError(
            "generic-helper-specialization",
            "SPIR-V codegen does not support generic functions: "
            "unspecialized generic helper "
            f"'{helper_name}' with generic parameters ({params_label})"
            f"{location_suffix}; specialize the function before SPIR-V generation",
            missing_capabilities=("spirv.generic_function_specialization",),
            source_location=source_location,
        )

    def option_payload_type_name(self, type_value) -> Optional[str]:
        type_name = self.type_name_from_value(type_value)
        if not isinstance(type_name, str):
            return None

        base_name, generic_args = generic_type_parts(type_name.strip())
        if base_name.rsplit("::", 1)[-1] != "Option" or len(generic_args) != 1:
            return None
        return generic_args[0]

    def lowerable_option_payload_type_name(self, type_value) -> Optional[str]:
        type_name = self.type_name_from_value(type_value)
        payload_type = self.option_payload_type_name(type_name)
        if payload_type is None:
            return None
        if resolve_generic_enum_specialization(self, type_name) is not None:
            return None
        return payload_type

    def option_or_expected_payload_type_name(self) -> Optional[str]:
        expected_type = self.current_expression_expected_type
        payload_type = self.lowerable_option_payload_type_name(expected_type)
        if payload_type is not None:
            return payload_type
        if self.option_payload_type_name(expected_type) is not None:
            return None
        return self.type_name_from_value(expected_type)

    def option_none_default_value(self) -> Optional[SpirvId]:
        payload_type = self.lowerable_option_payload_type_name(
            self.current_expression_expected_type
        )
        if payload_type is None:
            return None
        return self.default_value_for_type(self.map_crossgl_type(payload_type))

    def expected_primitive_type_name(self) -> Optional[str]:
        expected_type = self.option_or_expected_payload_type_name()
        if expected_type is None:
            return None
        return self.normalize_primitive_name(expected_type)

    def is_type_node_like(self, value) -> bool:
        return any(
            hasattr(value, attribute)
            for attribute in (
                "name",
                "element_type",
                "pointee_type",
                "referenced_type",
                "return_type",
            )
        )

    def process_expression_with_expected_type(self, expr, expected_type):
        previous_expected_type = self.current_expression_expected_type
        self.current_expression_expected_type = self.type_name_string(expected_type)
        try:
            return self.process_expression(expr)
        finally:
            self.current_expression_expected_type = previous_expected_type

    def collect_ast_functions(self, root):
        functions = []
        visited = set()

        def walk(value):
            if value is None or isinstance(value, (str, int, float, bool)):
                return
            if isinstance(value, dict):
                for item in value.values():
                    walk(item)
                return
            if isinstance(value, (list, tuple, set)):
                for item in value:
                    walk(item)
                return

            value_id = id(value)
            if value_id in visited:
                return
            visited.add(value_id)

            if hasattr(value, "body") and hasattr(value, "parameters"):
                functions.append(value)

            if hasattr(value, "__dict__"):
                for child in vars(value).values():
                    walk(child)

        walk(root)
        return functions

    def collect_functions_by_name(self, functions):
        functions_by_name = {}
        for function_node in functions:
            function_name = getattr(function_node, "name", None)
            if not function_name:
                continue
            functions_by_name.setdefault(function_name, []).append(function_node)
        return functions_by_name

    def function_parameters(self, function_node):
        return list(
            getattr(function_node, "parameters", getattr(function_node, "params", []))
            or []
        )

    def function_parameter_type_name(self, param) -> Optional[str]:
        param_type = getattr(param, "param_type", getattr(param, "vtype", None))
        return self.type_name_from_value(param_type)

    def function_parameter_type_names(self, function_node):
        return [
            self.function_parameter_type_name(param) or "unknown"
            for param in self.function_parameters(function_node)
        ]

    def storage_buffer_effective_call_signature(self, function_node, call_args):
        parameters = self.function_parameters(function_node)
        args = list(call_args or [])
        skipped_indices = self.skipped_function_parameter_indices_for_node(
            function_node
        )
        if not skipped_indices:
            return parameters, args
        return (
            [
                param
                for index, param in enumerate(parameters)
                if index not in skipped_indices
            ],
            [arg for index, arg in enumerate(args) if index not in skipped_indices],
        )

    def storage_buffer_effective_arity_matches(self, function_node, call_args):
        parameters, args = self.storage_buffer_effective_call_signature(
            function_node, call_args
        )
        return len(parameters) == len(args)

    def storage_buffer_argument_match_score(
        self, param, arg, substitutions=None
    ) -> Optional[int]:
        storage_buffer_type_name = self.storage_buffer_parameter_type_name(param)
        if storage_buffer_type_name is not None:
            storage_buffer_type_name = self.substitute_generic_signature_type(
                storage_buffer_type_name, substitutions
            )
            actual_type_name = self.storage_buffer_expression_type_name(arg)
            if actual_type_name is None:
                return None
            if not self.storage_buffer_parameter_type_is_compatible(
                storage_buffer_type_name, actual_type_name
            ):
                return None
            score = 4
            if self.normalize_signature_type_name(
                storage_buffer_type_name
            ) == self.normalize_signature_type_name(actual_type_name):
                score += 1
            return score

        declared_type_name = self.function_parameter_type_name(param)
        declared_type_name = self.substitute_generic_signature_type(
            declared_type_name, substitutions
        )
        actual_type_name = self.call_argument_type_name(arg)
        if not self.scalar_or_vector_type_compatible(
            declared_type_name, actual_type_name
        ):
            return None
        score = 0
        if actual_type_name is not None:
            score += 1
            if self.normalize_signature_type_name(
                declared_type_name
            ) == self.normalize_signature_type_name(actual_type_name):
                score += 1
        return score

    def storage_buffer_argument_rejection_reason(
        self, param, arg, arg_index, substitutions=None
    ) -> str:
        param_name = getattr(param, "name", f"param{arg_index}")
        storage_buffer_type_name = self.storage_buffer_parameter_type_name(param)
        if storage_buffer_type_name is not None:
            storage_buffer_type_name = self.substitute_generic_signature_type(
                storage_buffer_type_name, substitutions
            )
            actual_type_name = self.storage_buffer_expression_type_name(arg)
            if actual_type_name is None:
                actual_type_name = self.call_argument_type_name(arg) or "unknown"
                return (
                    f"argument {arg_index + 1} type {actual_type_name} is not a "
                    f"storage buffer for parameter {param_name} "
                    f"({storage_buffer_type_name})"
                )
            return (
                f"argument {arg_index + 1} type {actual_type_name} is incompatible "
                f"with parameter {param_name} ({storage_buffer_type_name})"
            )

        declared_type_name = self.substitute_generic_signature_type(
            self.function_parameter_type_name(param), substitutions
        )
        declared_type_name = declared_type_name or "unknown"
        actual_type_name = self.call_argument_type_name(arg) or "unknown"
        return (
            f"argument {arg_index + 1} type {actual_type_name} is incompatible "
            f"with parameter {param_name} ({declared_type_name})"
        )

    def storage_buffer_candidate_call_match(self, function_node, call_args):
        parameters = self.function_parameters(function_node)
        args = list(call_args or [])
        skipped_param_indices = self.skipped_function_parameter_indices_for_node(
            function_node
        )
        effective_parameters = [
            (index, param)
            for index, param in enumerate(parameters)
            if index not in skipped_param_indices
        ]
        substitutions = self.generic_storage_function_type_bindings(function_node, args)
        if substitutions is None:
            return None, "generic type arguments could not be resolved"

        reason = None
        incompatibility_reasons = []
        best_match = None

        def search(param_index, arg_index, score, bindings, skipped_arg_indices):
            nonlocal best_match, reason
            remaining_params = len(effective_parameters) - param_index
            remaining_args = len(args) - arg_index
            if remaining_args < remaining_params:
                reason = (
                    f"effective arity {len(effective_parameters)} exceeds remaining "
                    f"actual arguments {remaining_args}"
                )
                return
            if param_index == len(effective_parameters):
                skipped = skipped_arg_indices + list(range(arg_index, len(args)))
                candidate = (
                    score - len(skipped),
                    score,
                    -len(skipped),
                    bindings,
                    skipped,
                )
                if best_match is None or candidate[:3] > best_match[:3]:
                    best_match = candidate
                return
            if arg_index >= len(args):
                reason = (
                    f"effective arity {len(effective_parameters)} exceeds actual "
                    f"arity {len(args)}"
                )
                return

            original_param_index, param = effective_parameters[param_index]
            arg = args[arg_index]
            arg_score = self.storage_buffer_argument_match_score(
                param, arg, substitutions
            )
            if arg_score is not None:
                search(
                    param_index + 1,
                    arg_index + 1,
                    score + arg_score,
                    bindings + [(original_param_index, param, arg)],
                    skipped_arg_indices,
                )
            else:
                incompatibility_reasons.append(
                    self.storage_buffer_argument_rejection_reason(
                        param, arg, arg_index, substitutions
                    )
                )
            search(
                param_index,
                arg_index + 1,
                score,
                bindings,
                skipped_arg_indices + [arg_index],
            )

        search(0, 0, 0, [], [])
        if best_match is None:
            if incompatibility_reasons:
                reason = incompatibility_reasons[0]
            elif reason is None:
                reason = "no compatible ordered argument alignment"
            return None, reason

        rank, score, _skip_rank, bindings, skipped_arg_indices = best_match
        return {
            "rank_score": rank,
            "score": score,
            "bindings": bindings,
            "skipped_arg_indices": skipped_arg_indices,
            "effective_parameter_count": len(effective_parameters),
        }, None

    def storage_buffer_effective_call_bindings(self, function_node, call_args):
        match, _reason = self.storage_buffer_candidate_call_match(
            function_node, call_args
        )
        if match is not None:
            return match["bindings"]

        parameters = self.function_parameters(function_node)
        args = list(call_args or [])
        skipped_indices = self.skipped_function_parameter_indices_for_node(
            function_node
        )
        return [
            (index, param, args[index])
            for index, param in enumerate(parameters)
            if index not in skipped_indices and index < len(args)
        ]

    def format_function_candidate_signature(self, function_node) -> str:
        function_name = getattr(function_node, "name", "unknown")
        params = ", ".join(self.function_parameter_type_names(function_node))
        return f"{function_name}({params})"

    def normalize_signature_type_name(self, type_name) -> Optional[str]:
        if type_name is None:
            return None
        type_name = self.type_name_from_value(type_name)
        if type_name is None:
            return None
        type_name = self.normalize_reference_type_name(type_name)
        type_name = self.normalize_generic_vector_type(type_name)
        type_name = self.normalize_hlsl_matrix_type(type_name)
        return re.sub(r"\s+", "", str(type_name))

    def call_argument_type_name(self, arg) -> Optional[str]:
        storage_buffer_type = self.storage_buffer_expression_type_name(arg)
        if storage_buffer_type is not None:
            return storage_buffer_type
        inferred_type = self.infer_expression_result_type(arg)
        if inferred_type is None:
            return None
        return inferred_type.type.base_type

    def call_argument_type_names(self, args):
        return [self.call_argument_type_name(arg) or "unknown" for arg in args]

    def scalar_or_vector_type_compatible(
        self, declared_type: Optional[str], actual_type: Optional[str]
    ) -> bool:
        declared_type = self.normalize_signature_type_name(declared_type)
        actual_type = self.normalize_signature_type_name(actual_type)
        if declared_type is None or actual_type is None:
            return True
        if declared_type == actual_type:
            return True

        numeric_types = {"float", "double", "int", "uint", "i64", "u64", "bool"}
        if declared_type in numeric_types and actual_type in numeric_types:
            return True

        declared_vector = self.vector_component_type_and_count(declared_type)
        actual_vector = self.vector_component_type_and_count(actual_type)
        if declared_vector is not None or actual_vector is not None:
            return declared_vector == actual_vector

        return False

    def collect_generic_type_bindings(
        self, expected_type, actual_type, generic_params, substitutions
    ) -> None:
        if expected_type is None or actual_type is None:
            return

        expected_type = self.normalize_signature_type_name(expected_type)
        actual_type = self.normalize_signature_type_name(actual_type)
        if expected_type is None or actual_type is None:
            return

        if expected_type in generic_params:
            substitutions.setdefault(expected_type, actual_type)
            return

        expected_base, expected_args = generic_type_parts(expected_type)
        actual_base, actual_args = generic_type_parts(actual_type)
        if expected_base != actual_base or len(expected_args) != len(actual_args):
            return

        for expected_arg, actual_arg in zip(expected_args, actual_args):
            self.collect_generic_type_bindings(
                expected_arg, actual_arg, generic_params, substitutions
            )

    def generic_storage_function_type_bindings(self, function_node, call_args):
        generic_params = set(generic_function_parameters(function_node))
        if not generic_params:
            return {}

        substitutions = {}
        for param, arg in zip(self.function_parameters(function_node), call_args):
            declared_type_name = self.function_parameter_type_name(param)
            actual_type_name = self.call_argument_type_name(arg)
            self.collect_generic_type_bindings(
                declared_type_name, actual_type_name, generic_params, substitutions
            )

        if any(param not in substitutions for param in generic_params):
            return None
        return substitutions

    def substitute_generic_signature_type(self, type_name, substitutions):
        if not substitutions:
            return type_name
        type_name = self.type_name_from_value(type_name)
        if type_name is None:
            return None
        if type_name.startswith("&"):
            referenced = substitute_generic_type_name(
                type_name[1:].strip(), substitutions
            )
            return f"&{referenced}"
        if type_name.endswith("*"):
            pointee = substitute_generic_type_name(
                type_name[:-1].strip(), substitutions
            )
            return f"{pointee}*"
        return substitute_generic_type_name(type_name, substitutions)

    def storage_buffer_candidate_match_score(self, function_node, call_args):
        match, _reason = self.storage_buffer_candidate_call_match(
            function_node, call_args
        )
        if match is None:
            return None
        return match["score"]

    def has_matching_non_storage_function_candidate(self, function_name, call_args):
        for function_node in self.function_nodes_by_name.get(function_name, []):
            if self.function_has_storage_buffer_parameters(function_node):
                continue
            parameters, args = self.storage_buffer_effective_call_signature(
                function_node, call_args
            )
            if len(parameters) != len(args):
                continue
            return True
        return False

    def has_matching_concrete_function_candidate(self, function_name, call_args):
        for function_node in self.function_nodes_by_name.get(function_name, []):
            if generic_function_parameters(function_node):
                continue
            parameters, args = self.storage_buffer_effective_call_signature(
                function_node, call_args
            )
            if len(parameters) != len(args):
                continue
            return True
        return False

    def storage_buffer_overload_diagnostic(
        self,
        function_name,
        call_args,
        candidates,
        reason,
        call_node=None,
        rejection_reasons=None,
    ):
        actual_types = ", ".join(self.call_argument_type_names(call_args))
        actual_signature = f"{function_name}({actual_types})"
        candidate_signatures = ", ".join(
            self.format_function_candidate_signature(candidate)
            for candidate in candidates
        )
        if not candidate_signatures:
            candidate_signatures = "<none>"
        rejection_detail = ""
        if rejection_reasons:
            details = []
            for candidate in candidates:
                signature = self.format_function_candidate_signature(candidate)
                candidate_reason = rejection_reasons.get(id(candidate))
                if candidate_reason is None:
                    candidate_reason = "not rejected"
                details.append(f"{signature}: {candidate_reason}")
            rejection_detail = "; candidate rejection reasons: " + "; ".join(details)
        return UnsupportedSPIRVFeatureError(
            "storage-buffer-function-overload",
            "SPIR-V storage-buffer function inlining could not resolve "
            f"overloaded call '{function_name}': {reason}; actual call "
            f"arity/types: {len(call_args)} ({actual_signature}); candidate "
            f"signatures: {candidate_signatures}{rejection_detail}",
            missing_capabilities=("spirv.storage_buffer_function_overload",),
            source_location=(
                getattr(call_node, "source_location", None)
                if call_node is not None
                else None
            ),
        )

    def order_functions_by_dependencies(self, functions):
        ordered = []
        visiting = set()
        visited = set()
        function_list = list(functions)
        function_names = [getattr(func, "name", None) for func in function_list]
        unique_names = {
            name for name in function_names if name and function_names.count(name) == 1
        }
        functions_by_name = {
            func.name: func
            for func in function_list
            if getattr(func, "name", None) in unique_names
        }

        def visit(func):
            func_id = id(func)
            if func_id in visited:
                return
            if func_id in visiting:
                return

            visiting.add(func_id)
            for node in self.walk_ast_nodes(getattr(func, "body", [])):
                if not isinstance(node, FunctionCallNode):
                    continue

                dependency_name = self.function_call_name(node)
                dependency = functions_by_name.get(dependency_name)
                if dependency is not None and dependency is not func:
                    visit(dependency)

            visiting.remove(func_id)
            visited.add(func_id)
            ordered.append(func)

        for func in function_list:
            visit(func)

        return ordered

    def walk_ast_nodes(self, root):
        visited = set()

        def walk(value):
            if value is None or isinstance(value, (str, int, float, bool)):
                return
            if isinstance(value, dict):
                for item in value.values():
                    yield from walk(item)
                return
            if isinstance(value, (list, tuple, set)):
                for item in value:
                    yield from walk(item)
                return

            value_id = id(value)
            if value_id in visited:
                return
            visited.add(value_id)
            yield value

            if hasattr(value, "__dict__"):
                for child in vars(value).values():
                    yield from walk(child)

        yield from walk(root)

    def array_dimensions(self, type_name: str):
        if not type_name or "[" not in type_name:
            return None

        suffix = type_name[type_name.find("[") :]
        dimensions = []
        offset = 0
        while offset < len(suffix):
            if suffix[offset] != "[":
                return None
            end = suffix.find("]", offset + 1)
            if end == -1:
                return None
            dimensions.append(suffix[offset + 1 : end])
            offset = end + 1
        return dimensions

    def is_unsized_resource_array_type_name(self, type_name: str) -> bool:
        type_name = self.normalize_generic_vector_type(str(type_name))
        array_type = self.split_outer_array_type(type_name)
        return (
            array_type is not None
            and array_type[1] is None
            and self.is_resource_type_name(self.array_base_type_name(type_name))
        )

    def is_fixed_resource_array_type_name(self, type_name: str) -> bool:
        type_name = self.normalize_generic_vector_type(str(type_name))
        array_type = self.split_outer_array_type(type_name)
        return (
            array_type is not None
            and array_type[1] is not None
            and self.is_resource_type_name(self.array_base_type_name(type_name))
        )

    def fixed_type_for_unsized_resource_param(self, declared_type: str, arg_type: str):
        declared_type = self.normalize_generic_vector_type(str(declared_type))
        arg_type = self.normalize_generic_vector_type(str(arg_type))

        if not self.is_unsized_resource_array_type_name(declared_type):
            return None
        if not self.is_fixed_resource_array_type_name(arg_type):
            return None
        if self.array_base_type_name(declared_type) != self.array_base_type_name(
            arg_type
        ):
            return None

        declared_dimensions = self.array_dimensions(declared_type)
        arg_dimensions = self.array_dimensions(arg_type)
        if not declared_dimensions or not arg_dimensions:
            return None
        if len(declared_dimensions) != len(arg_dimensions):
            return None
        if declared_dimensions[0] != "":
            return None
        if declared_dimensions[1:] != arg_dimensions[1:]:
            return None

        return arg_type

    def resource_array_param_type_is_compatible(
        self, declared_type: str, arg_type: str
    ) -> bool:
        declared_type = self.normalize_generic_vector_type(str(declared_type))
        arg_type = self.normalize_generic_vector_type(str(arg_type))

        if not self.is_unsized_resource_array_type_name(declared_type):
            return True

        if self.fixed_type_for_unsized_resource_param(declared_type, arg_type):
            return True

        if not self.is_unsized_resource_array_type_name(arg_type):
            return False

        return self.array_base_type_name(declared_type) == self.array_base_type_name(
            arg_type
        ) and self.array_dimensions(declared_type) == self.array_dimensions(arg_type)

    def expression_name(self, expr):
        if isinstance(expr, str):
            return expr
        if hasattr(expr, "name") and isinstance(expr.name, str):
            return expr.name
        if isinstance(expr, ArrayAccessNode):
            array_expr = getattr(expr, "array", getattr(expr, "array_expr", None))
            return self.expression_name(array_expr)
        return None

    def diagnostic_expression(self, expr) -> str:
        """Render expression nodes for SPIR-V warnings without leaking reprs."""
        if expr is None:
            return "<unknown>"
        if isinstance(expr, str):
            return expr
        if isinstance(expr, IdentifierNode):
            return expr.name
        if isinstance(expr, VariableNode):
            return expr.name
        if isinstance(expr, LiteralNode):
            return str(expr.value)
        if isinstance(expr, ArrayAccessNode):
            array_expr = getattr(expr, "array", getattr(expr, "array_expr", None))
            index_expr = getattr(expr, "index", getattr(expr, "index_expr", None))
            return (
                f"{self.diagnostic_expression(array_expr)}"
                f"[{self.diagnostic_expression(index_expr)}]"
            )
        if isinstance(expr, MemberAccessNode):
            object_expr = getattr(expr, "object", getattr(expr, "object_expr", None))
            return f"{self.diagnostic_expression(object_expr)}.{expr.member}"
        if isinstance(expr, FunctionCallNode):
            callee = getattr(expr, "function", getattr(expr, "name", None))
            args = ", ".join(
                self.diagnostic_expression(arg)
                for arg in getattr(expr, "args", getattr(expr, "arguments", []))
            )
            return f"{self.diagnostic_expression(callee)}({args})"
        if isinstance(expr, UnaryOpNode):
            operand = self.diagnostic_expression(expr.operand)
            return f"{operand}{expr.op}" if expr.is_postfix else f"{expr.op}{operand}"
        if isinstance(expr, BinaryOpNode):
            return (
                f"{self.diagnostic_expression(expr.left)} {expr.op} "
                f"{self.diagnostic_expression(expr.right)}"
            )
        name = self.expression_name(expr)
        if name is not None:
            return name
        return type(expr).__name__

    def direct_expression_name(self, expr):
        if isinstance(expr, str):
            return expr
        if isinstance(expr, (IdentifierNode, VariableNode)):
            return getattr(expr, "name", None)
        return None

    def function_call_name(self, call):
        callee = getattr(call, "function", getattr(call, "name", None))
        if hasattr(callee, "name"):
            return callee.name
        if isinstance(callee, str):
            return callee
        return None

    def collect_function_image_access_requirements_for_ast(self, ast):
        functions = self.collect_ast_functions(ast)
        self.function_parameter_names = collect_function_parameter_names(functions)
        return collect_function_image_access_requirements(
            functions,
            self.function_parameter_names,
            self.walk_ast_nodes,
            self.function_call_name,
            self.expression_name,
        )

    def collect_stage_local_function_image_access_metadata(self, ast):
        self.stage_local_function_parameter_names = {}
        self.stage_local_function_image_access_requirements = {}
        global_functions = list(getattr(ast, "functions", []) or [])

        for stage in (getattr(ast, "stages", None) or {}).values():
            local_functions = list(getattr(stage, "local_functions", []) or [])
            if not local_functions:
                continue

            local_names = {
                getattr(func, "name", None)
                for func in local_functions
                if getattr(func, "name", None)
            }
            visible_functions = [
                func
                for func in global_functions
                if getattr(func, "name", None) not in local_names
            ] + local_functions
            parameter_names = collect_function_parameter_names(visible_functions)
            requirements = collect_function_image_access_requirements(
                visible_functions,
                parameter_names,
                self.walk_ast_nodes,
                self.function_call_name,
                self.expression_name,
            )

            for function_node in local_functions:
                function_name = getattr(function_node, "name", None)
                if not function_name:
                    continue
                key = self.stage_local_function_key(stage, function_name)
                if key is None:
                    continue
                self.stage_local_function_parameter_names[key] = parameter_names.get(
                    function_name, []
                )
                self.stage_local_function_image_access_requirements[key] = (
                    requirements.get(function_name, {})
                )

    def buffer_operation_access_requirement(self, func_name):
        if (
            func_name == "buffer_load"
            or self.byte_address_helper_load_width(func_name) is not None
        ):
            return "read"
        if (
            func_name == "buffer_store"
            or self.byte_address_helper_store_width(func_name) is not None
        ):
            return "write"
        if func_name == "buffer_append":
            return "write"
        if func_name == "buffer_consume":
            return "read_write"
        if func_name in {"buffer_increment_counter", "buffer_decrement_counter"}:
            return "read_write"
        if func_name in self.buffer_atomic_function_names():
            return "read_write"
        return None

    def storage_buffer_member_method_access_requirement(self, method_name):
        if method_name in {"Load"} or self.byte_address_method_load_width(method_name):
            return "read"
        if method_name in {"Store"} or self.byte_address_method_store_width(
            method_name
        ):
            return "write"
        if method_name == "Append":
            return "write"
        if method_name == "Consume":
            return "read_write"
        if method_name in {"IncrementCounter", "DecrementCounter"}:
            return "read_write"
        if self.byte_address_method_interlocked_info(method_name) is not None:
            return "read_write"
        return None

    def storage_buffer_parameter_root_name(self, expr, storage_buffer_parameters):
        if isinstance(expr, str):
            return expr if expr in storage_buffer_parameters else None
        if isinstance(expr, (IdentifierNode, VariableNode)):
            name = getattr(expr, "name", None)
            return name if name in storage_buffer_parameters else None
        if isinstance(expr, ArrayAccessNode):
            return self.storage_buffer_parameter_root_name(
                getattr(expr, "array", getattr(expr, "array_expr", None)),
                storage_buffer_parameters,
            )
        if isinstance(expr, MemberAccessNode):
            return self.storage_buffer_parameter_root_name(
                getattr(expr, "object", getattr(expr, "object_expr", None)),
                storage_buffer_parameters,
            )
        return None

    def function_storage_buffer_parameter_indices(self, function_node) -> set:
        return {
            index
            for index, param in enumerate(
                getattr(
                    function_node, "parameters", getattr(function_node, "params", [])
                )
            )
            if self.storage_buffer_parameter_type_name(param) is not None
        }

    def scan_storage_buffer_access_path_indices(
        self,
        expr,
        func_name,
        storage_buffer_parameters,
        callee_storage_buffer_parameter_indices,
        requirements,
        visited,
    ):
        if isinstance(expr, ArrayAccessNode):
            self.scan_storage_buffer_requirement_node(
                getattr(expr, "index", getattr(expr, "index_expr", None)),
                func_name,
                storage_buffer_parameters,
                callee_storage_buffer_parameter_indices,
                requirements,
                visited,
            )
            self.scan_storage_buffer_access_path_indices(
                getattr(expr, "array", getattr(expr, "array_expr", None)),
                func_name,
                storage_buffer_parameters,
                callee_storage_buffer_parameter_indices,
                requirements,
                visited,
            )
        elif isinstance(expr, MemberAccessNode):
            self.scan_storage_buffer_access_path_indices(
                getattr(expr, "object", getattr(expr, "object_expr", None)),
                func_name,
                storage_buffer_parameters,
                callee_storage_buffer_parameter_indices,
                requirements,
                visited,
            )

    def merge_storage_buffer_access_requirement_for_parameter(
        self, requirements, func_name, parameter_name, required_access
    ):
        if parameter_name is None:
            return
        current = requirements[func_name].get(parameter_name)
        requirements[func_name][parameter_name] = (
            self.merge_resource_access_requirement(current, required_access)
        )

    def scan_storage_buffer_requirement_expression(
        self,
        expr,
        func_name,
        storage_buffer_parameters,
        callee_storage_buffer_parameter_indices,
        requirements,
        visited,
    ):
        if expr is None or isinstance(expr, (str, int, float, bool)):
            return

        if isinstance(expr, FunctionCallNode):
            callee_expr = getattr(expr, "function", getattr(expr, "name", None))
            if isinstance(callee_expr, MemberAccessNode):
                args = getattr(expr, "arguments", getattr(expr, "args", []))
                required_access = self.storage_buffer_member_method_access_requirement(
                    getattr(callee_expr, "member", None)
                )
                target_name = self.storage_buffer_parameter_root_name(
                    getattr(callee_expr, "object", None),
                    storage_buffer_parameters,
                )
                if required_access is not None and target_name is not None:
                    self.merge_storage_buffer_access_requirement_for_parameter(
                        requirements, func_name, target_name, required_access
                    )
                    self.scan_storage_buffer_access_path_indices(
                        getattr(callee_expr, "object", None),
                        func_name,
                        storage_buffer_parameters,
                        callee_storage_buffer_parameter_indices,
                        requirements,
                        visited,
                    )
                    for arg in args:
                        self.scan_storage_buffer_requirement_node(
                            arg,
                            func_name,
                            storage_buffer_parameters,
                            callee_storage_buffer_parameter_indices,
                            requirements,
                            visited,
                        )
                    return

            callee_name = self.function_call_name(expr)
            args = getattr(expr, "arguments", getattr(expr, "args", []))
            required_access = self.buffer_operation_access_requirement(callee_name)
            if required_access is not None and args:
                target_name = self.storage_buffer_parameter_root_name(
                    args[0], storage_buffer_parameters
                )
                self.merge_storage_buffer_access_requirement_for_parameter(
                    requirements, func_name, target_name, required_access
                )
                self.scan_storage_buffer_access_path_indices(
                    args[0],
                    func_name,
                    storage_buffer_parameters,
                    callee_storage_buffer_parameter_indices,
                    requirements,
                    visited,
                )
                for arg in args[1:]:
                    self.scan_storage_buffer_requirement_node(
                        arg,
                        func_name,
                        storage_buffer_parameters,
                        callee_storage_buffer_parameter_indices,
                        requirements,
                        visited,
                    )
                return

            storage_buffer_indices = callee_storage_buffer_parameter_indices.get(
                callee_name, set()
            )
            for arg_index, arg in enumerate(args):
                if (
                    arg_index in storage_buffer_indices
                    and self.storage_buffer_parameter_root_name(
                        arg, storage_buffer_parameters
                    )
                    is not None
                ):
                    self.scan_storage_buffer_access_path_indices(
                        arg,
                        func_name,
                        storage_buffer_parameters,
                        callee_storage_buffer_parameter_indices,
                        requirements,
                        visited,
                    )
                    continue
                self.scan_storage_buffer_requirement_node(
                    arg,
                    func_name,
                    storage_buffer_parameters,
                    callee_storage_buffer_parameter_indices,
                    requirements,
                    visited,
                )
            return

        target_name = self.storage_buffer_parameter_root_name(
            expr, storage_buffer_parameters
        )
        if target_name is not None:
            self.merge_storage_buffer_access_requirement_for_parameter(
                requirements, func_name, target_name, "read"
            )
            self.scan_storage_buffer_access_path_indices(
                expr,
                func_name,
                storage_buffer_parameters,
                callee_storage_buffer_parameter_indices,
                requirements,
                visited,
            )
            return

        if isinstance(expr, ArrayAccessNode):
            self.scan_storage_buffer_requirement_node(
                getattr(expr, "array", getattr(expr, "array_expr", None)),
                func_name,
                storage_buffer_parameters,
                callee_storage_buffer_parameter_indices,
                requirements,
                visited,
            )
            self.scan_storage_buffer_requirement_node(
                getattr(expr, "index", getattr(expr, "index_expr", None)),
                func_name,
                storage_buffer_parameters,
                callee_storage_buffer_parameter_indices,
                requirements,
                visited,
            )
            return
        if isinstance(expr, MemberAccessNode):
            self.scan_storage_buffer_requirement_node(
                getattr(expr, "object", getattr(expr, "object_expr", None)),
                func_name,
                storage_buffer_parameters,
                callee_storage_buffer_parameter_indices,
                requirements,
                visited,
            )
            return

        if isinstance(expr, BinaryOpNode):
            self.scan_storage_buffer_requirement_node(
                expr.left,
                func_name,
                storage_buffer_parameters,
                callee_storage_buffer_parameter_indices,
                requirements,
                visited,
            )
            self.scan_storage_buffer_requirement_node(
                expr.right,
                func_name,
                storage_buffer_parameters,
                callee_storage_buffer_parameter_indices,
                requirements,
                visited,
            )
            return
        if isinstance(expr, UnaryOpNode):
            self.scan_storage_buffer_requirement_node(
                expr.operand,
                func_name,
                storage_buffer_parameters,
                callee_storage_buffer_parameter_indices,
                requirements,
                visited,
            )
            return
        if isinstance(expr, TernaryOpNode):
            for child in (expr.condition, expr.true_expr, expr.false_expr):
                self.scan_storage_buffer_requirement_node(
                    child,
                    func_name,
                    storage_buffer_parameters,
                    callee_storage_buffer_parameter_indices,
                    requirements,
                    visited,
                )
            return
        if isinstance(expr, ArrayLiteralNode):
            for element in getattr(expr, "elements", []):
                self.scan_storage_buffer_requirement_node(
                    element,
                    func_name,
                    storage_buffer_parameters,
                    callee_storage_buffer_parameter_indices,
                    requirements,
                    visited,
                )

    def scan_storage_buffer_assignment_target_requirement(
        self,
        target,
        operator,
        func_name,
        storage_buffer_parameters,
        callee_storage_buffer_parameter_indices,
        requirements,
        visited,
    ):
        target_name = self.storage_buffer_parameter_root_name(
            target, storage_buffer_parameters
        )
        if target_name is not None:
            required_access = "write" if operator == "=" else "read_write"
            self.merge_storage_buffer_access_requirement_for_parameter(
                requirements, func_name, target_name, required_access
            )
            self.scan_storage_buffer_access_path_indices(
                target,
                func_name,
                storage_buffer_parameters,
                callee_storage_buffer_parameter_indices,
                requirements,
                visited,
            )
            return

        self.scan_storage_buffer_requirement_node(
            target,
            func_name,
            storage_buffer_parameters,
            callee_storage_buffer_parameter_indices,
            requirements,
            visited,
        )

    def scan_storage_buffer_requirement_node(
        self,
        node,
        func_name,
        storage_buffer_parameters,
        callee_storage_buffer_parameter_indices,
        requirements,
        visited,
    ):
        if node is None or isinstance(node, (str, int, float, bool)):
            return
        if isinstance(node, dict):
            for value in node.values():
                self.scan_storage_buffer_requirement_node(
                    value,
                    func_name,
                    storage_buffer_parameters,
                    callee_storage_buffer_parameter_indices,
                    requirements,
                    visited,
                )
            return
        if isinstance(node, (list, tuple, set)):
            for value in node:
                self.scan_storage_buffer_requirement_node(
                    value,
                    func_name,
                    storage_buffer_parameters,
                    callee_storage_buffer_parameter_indices,
                    requirements,
                    visited,
                )
            return

        node_id = id(node)
        if node_id in visited:
            return
        visited.add(node_id)

        if isinstance(node, AssignmentNode):
            target = getattr(
                node, "target", getattr(node, "left", getattr(node, "name", None))
            )
            operator = getattr(node, "operator", "=")
            self.scan_storage_buffer_assignment_target_requirement(
                target,
                operator,
                func_name,
                storage_buffer_parameters,
                callee_storage_buffer_parameter_indices,
                requirements,
                visited,
            )
            self.scan_storage_buffer_requirement_node(
                getattr(node, "value", getattr(node, "right", None)),
                func_name,
                storage_buffer_parameters,
                callee_storage_buffer_parameter_indices,
                requirements,
                visited,
            )
            return

        if isinstance(node, ReturnNode):
            self.scan_storage_buffer_requirement_node(
                getattr(node, "value", None),
                func_name,
                storage_buffer_parameters,
                callee_storage_buffer_parameter_indices,
                requirements,
                visited,
            )
            return

        if isinstance(node, VariableNode):
            self.scan_storage_buffer_requirement_node(
                getattr(node, "initial_value", None),
                func_name,
                storage_buffer_parameters,
                callee_storage_buffer_parameter_indices,
                requirements,
                visited,
            )
            return

        if isinstance(
            node,
            (
                ArrayAccessNode,
                ArrayLiteralNode,
                BinaryOpNode,
                FunctionCallNode,
                MemberAccessNode,
                TernaryOpNode,
                UnaryOpNode,
            ),
        ):
            self.scan_storage_buffer_requirement_expression(
                node,
                func_name,
                storage_buffer_parameters,
                callee_storage_buffer_parameter_indices,
                requirements,
                visited,
            )
            return

        if hasattr(node, "expression"):
            self.scan_storage_buffer_requirement_node(
                node.expression,
                func_name,
                storage_buffer_parameters,
                callee_storage_buffer_parameter_indices,
                requirements,
                visited,
            )
            return

        if hasattr(node, "__dict__"):
            for field_name, child in vars(node).items():
                if field_name in {"annotations", "parent"}:
                    continue
                self.scan_storage_buffer_requirement_node(
                    child,
                    func_name,
                    storage_buffer_parameters,
                    callee_storage_buffer_parameter_indices,
                    requirements,
                    visited,
                )

    def collect_function_storage_buffer_access_requirements_for_functions(
        self, functions
    ):
        functions = list(functions)
        functions_by_name = self.collect_functions_by_name(functions)
        function_parameter_names_by_id = {
            id(func): [
                getattr(param, "name", None)
                for param in self.function_parameters(func)
                if getattr(param, "name", None)
            ]
            for func in functions
            if getattr(func, "name", None)
        }
        parameter_sets = {
            func_key: set(param_names)
            for func_key, param_names in function_parameter_names_by_id.items()
        }
        requirements = {
            id(func): {} for func in functions if getattr(func, "name", None)
        }
        storage_buffer_parameter_sets = {
            id(func): self.function_storage_buffer_parameters(func)
            for func in functions
            if getattr(func, "name", None)
        }
        storage_buffer_parameter_indices = {}
        for func in functions:
            func_name = getattr(func, "name", None)
            if not func_name:
                continue
            storage_buffer_parameter_indices.setdefault(func_name, set()).update(
                self.function_storage_buffer_parameter_indices(func)
            )

        for func in functions:
            func_name = getattr(func, "name", None)
            if not func_name:
                continue
            func_key = id(func)

            self.scan_storage_buffer_requirement_node(
                getattr(func, "body", []),
                func_key,
                storage_buffer_parameter_sets.get(func_key, set()),
                storage_buffer_parameter_indices,
                requirements,
                set(),
            )

        changed = True
        while changed:
            changed = False
            for func in functions:
                func_name = getattr(func, "name", None)
                if not func_name:
                    continue
                func_key = id(func)

                parameter_set = parameter_sets.get(func_key, set())
                if not parameter_set:
                    continue

                for node in self.walk_ast_nodes(getattr(func, "body", [])):
                    if not isinstance(node, FunctionCallNode):
                        continue

                    callee_name = self.function_call_name(node)
                    args = getattr(node, "arguments", getattr(node, "args", []))
                    callee_candidates = [
                        candidate
                        for candidate in functions_by_name.get(callee_name, [])
                        if len(self.function_parameters(candidate)) == len(args)
                    ]
                    if not callee_candidates:
                        continue

                    for callee_candidate in callee_candidates:
                        callee_key = id(callee_candidate)
                        callee_requirements = requirements.get(callee_key)
                        if not callee_requirements:
                            continue
                        callee_parameters = function_parameter_names_by_id.get(
                            callee_key, []
                        )
                        for (
                            callee_param,
                            required_access,
                        ) in callee_requirements.items():
                            try:
                                index = callee_parameters.index(callee_param)
                            except ValueError:
                                continue
                            if index >= len(args):
                                continue

                            target_name = self.expression_name(args[index])
                            if target_name not in parameter_set:
                                continue

                            current = requirements[func_key].get(target_name)
                            merged = self.merge_resource_access_requirement(
                                current, required_access
                            )
                            if merged != current:
                                requirements[func_key][target_name] = merged
                                changed = True

        self.function_parameter_names_by_id.update(function_parameter_names_by_id)
        filtered_requirements = {
            func_key: reqs for func_key, reqs in requirements.items() if reqs
        }
        self.function_storage_buffer_access_requirements_by_id.update(
            filtered_requirements
        )
        legacy_requirements = {}
        for func in functions:
            func_name = getattr(func, "name", None)
            if not func_name:
                continue
            for param_name, required_access in filtered_requirements.get(
                id(func), {}
            ).items():
                legacy_requirements.setdefault(func_name, {})
                current = legacy_requirements[func_name].get(param_name)
                legacy_requirements[func_name][param_name] = (
                    self.merge_resource_access_requirement(current, required_access)
                )
        return legacy_requirements

    def collect_function_storage_buffer_access_requirements_for_ast(self, ast):
        return self.collect_function_storage_buffer_access_requirements_for_functions(
            self.collect_ast_functions(ast)
        )

    def collect_stage_local_function_storage_buffer_metadata(self, ast):
        self.stage_local_function_storage_buffer_access_requirements = {}
        self.stage_local_inline_storage_buffer_functions = {}
        global_functions = list(getattr(ast, "functions", []) or [])

        for stage in (getattr(ast, "stages", None) or {}).values():
            local_functions = list(getattr(stage, "local_functions", []) or [])
            if not local_functions:
                continue

            local_names = {
                getattr(func, "name", None)
                for func in local_functions
                if getattr(func, "name", None)
            }
            visible_functions = [
                func
                for func in global_functions
                if getattr(func, "name", None) not in local_names
            ] + local_functions
            requirements = (
                self.collect_function_storage_buffer_access_requirements_for_functions(
                    visible_functions
                )
            )

            for function_node in local_functions:
                function_name = getattr(function_node, "name", None)
                if not function_name:
                    continue
                key = self.stage_local_function_key(stage, function_name)
                if key is None:
                    continue
                self.stage_local_function_storage_buffer_access_requirements[key] = (
                    requirements.get(function_name, {})
                )
                self.stage_local_inline_storage_buffer_functions.setdefault(key, [])
                if self.function_has_storage_buffer_parameters(function_node):
                    self.stage_local_inline_storage_buffer_functions[key].append(
                        function_node
                    )

    def merge_resource_access_requirement(self, current, incoming):
        if incoming is None:
            return current
        if current is None or current == incoming:
            return incoming
        return "read_write"

    def storage_image_access_for_expression(self, expr):
        name = self.expression_name(expr)
        if name is None:
            return None

        pointer = self.local_variables.get(name) or self.resolve_global_variable(name)
        metadata = (
            self.resource_metadata_for_pointer(pointer) if pointer is not None else None
        )
        if metadata is None:
            return None
        if metadata.get("kind") != "storage_image":
            return None
        if metadata.get("readonly"):
            return "read"
        if metadata.get("writeonly"):
            return "write"
        if metadata.get("readwrite"):
            return "read_write"
        return None

    def storage_buffer_access_for_expression(self, expr):
        name = self.expression_name(expr)
        if name is None:
            return None

        pointer = self.local_variables.get(name) or self.resolve_global_variable(name)
        metadata = (
            self.storage_buffer_access_metadata_for_pointer(pointer)
            if pointer is not None
            else None
        )
        if metadata is None:
            return None
        if metadata.get("append_consume"):
            return "read_write"
        if metadata.get("readonly"):
            return "read"
        if metadata.get("writeonly"):
            return "write"
        if metadata.get("readwrite"):
            return "read_write"
        return None

    def expression_debug_name(self, expr) -> str:
        name = self.expression_name(expr)
        return name if name is not None else str(expr)

    def validate_function_image_access_arguments(self, func_name, args) -> bool:
        callee_requirements = self.resolve_function_image_access_requirements(func_name)
        if not callee_requirements:
            return True

        param_names = self.resolve_function_parameter_names(func_name)
        for index, param_name in enumerate(param_names):
            required_access = callee_requirements.get(param_name)
            if required_access is None or index >= len(args):
                continue

            actual_access = self.storage_image_access_for_expression(args[index])
            if image_access_satisfies_requirement(required_access, actual_access):
                continue

            required_label = image_access_requirement_label(required_access)
            actual_label = image_access_diagnostic_name(actual_access)
            self.emit(
                f"; WARNING: function call '{func_name}' requires {required_label} "
                "storage image access for argument "
                f"{self.expression_debug_name(args[index])} passed to parameter "
                f"{param_name}: got {actual_label}"
            )
            return False

        return True

    def validate_function_storage_buffer_access_arguments(
        self, function_node, args
    ) -> bool:
        func_name = getattr(function_node, "name", "unknown")
        callee_requirements = (
            self.function_storage_buffer_access_requirements_by_id.get(
                id(function_node), {}
            )
        )
        if not callee_requirements:
            return True

        param_names = self.function_parameter_names_by_id.get(
            id(function_node),
            [
                getattr(param, "name", None)
                for param in self.function_parameters(function_node)
            ],
        )
        for index, param, arg in self.storage_buffer_effective_call_bindings(
            function_node, args
        ):
            param_name = (
                param_names[index]
                if index < len(param_names)
                else getattr(param, "name", None)
            )
            required_access = callee_requirements.get(param_name)
            if required_access is None:
                continue

            actual_access = self.storage_buffer_access_for_expression(arg)
            if image_access_satisfies_requirement(required_access, actual_access):
                continue

            required_label = image_access_requirement_label(required_access)
            actual_label = image_access_diagnostic_name(actual_access)
            self.emit(
                f"; WARNING: function call '{func_name}' requires {required_label} "
                "storage buffer access for argument "
                f"{self.expression_debug_name(arg)} passed to parameter "
                f"{param_name}: got {actual_label}"
            )
            return False

        return True

    def collect_inline_storage_buffer_functions(self, ast):
        functions = {}
        for func in self.collect_ast_functions(ast):
            if not getattr(func, "name", None):
                continue
            if not self.function_has_storage_buffer_parameters(func):
                continue
            functions.setdefault(func.name, []).append(func)
        return functions

    def default_value_for_function(self, function_node) -> Optional[SpirvId]:
        return_type = self.map_crossgl_type(function_node.return_type)
        if return_type.type.base_type == "void":
            return None
        return self.default_value_for_type(return_type)

    def inline_storage_buffer_function_call(
        self, function_node, call_args, call_node=None
    ):
        func_name = getattr(function_node, "name", "unknown")
        if not self.validate_function_storage_buffer_access_arguments(
            function_node, call_args
        ):
            return self.default_value_for_function(function_node)

        match, rejection_reason = self.storage_buffer_candidate_call_match(
            function_node, call_args
        )
        if match is None:
            parameters, effective_call_args = (
                self.storage_buffer_effective_call_signature(function_node, call_args)
            )
            if len(effective_call_args) > len(parameters):
                raise self.storage_buffer_overload_diagnostic(
                    func_name,
                    call_args,
                    [function_node],
                    "selected candidate has too few parameters",
                    call_node=call_node,
                    rejection_reasons={id(function_node): rejection_reason},
                )
            if len(effective_call_args) < len(parameters):
                self.emit(
                    f"; WARNING: function call '{func_name}' requires "
                    f"{len(parameters)} arguments"
                )
                return self.default_value_for_function(function_node)

        function_key = id(function_node)
        if any(
            key == function_key for key, _name in self.inline_storage_buffer_call_stack
        ):
            cycle = [
                name
                for key, name in self.inline_storage_buffer_call_stack
                if key == function_key
            ]
            cycle.append(func_name)
            raise UnsupportedSPIRVFeatureError(
                "recursive-storage-buffer-function-inline",
                "SPIR-V storage-buffer function inlining does not support "
                f"recursive helper calls ({' -> '.join(cycle)})",
                missing_capabilities=("spirv.recursive_storage_buffer_function",),
            )

        previous_locals = self.local_variables.copy()
        previous_precise_locals = set(self.precise_local_variables)
        previous_return_type = self.current_return_type
        previous_generic_type_substitutions = self.current_generic_type_substitutions
        generic_type_substitutions = self.generic_storage_function_type_bindings(
            function_node, call_args
        )
        if generic_type_substitutions is None:
            generic_type_substitutions = {}
        self.current_generic_type_substitutions = generic_type_substitutions
        self.current_return_type = self.map_crossgl_type(function_node.return_type)
        self.inline_storage_buffer_call_stack.append((function_key, func_name))

        try:
            for index, param, arg in self.storage_buffer_effective_call_bindings(
                function_node, call_args
            ):
                param_name = getattr(param, "name", f"param{index}")
                storage_buffer_type_name = self.storage_buffer_parameter_type_name(
                    param
                )
                if storage_buffer_type_name is not None:
                    pointer_arg = self.variable_pointer_from_expression(arg)
                    if pointer_arg is None:
                        self.emit(
                            f"; WARNING: function call '{func_name}' requires a "
                            f"storage buffer argument for parameter {param_name}"
                        )
                        return self.default_value_for_function(function_node)
                    actual_type_name = self.storage_buffer_expression_type_name(arg)
                    if (
                        actual_type_name is not None
                        and not self.storage_buffer_parameter_type_is_compatible(
                            storage_buffer_type_name, actual_type_name
                        )
                    ):
                        self.emit(
                            f"; WARNING: function call '{func_name}' requires "
                            f"{storage_buffer_type_name} storage buffer type for "
                            "argument "
                            f"{self.expression_debug_name(arg)} "
                            f"passed to parameter {param_name}: got {actual_type_name}"
                        )
                        return self.default_value_for_function(function_node)
                    self.local_variables[param_name] = pointer_arg
                    continue

                arg_value = self.process_call_argument(func_name, arg, index)
                if arg_value is None:
                    self.emit(f"; WARNING: Failed to evaluate argument for {func_name}")
                    return self.default_value_for_function(function_node)
                self.local_variables[param_name] = arg_value

            result = self.inline_function_body(function_node)
            if result is not None:
                return result
            return self.default_value_for_function(function_node)
        finally:
            self.inline_storage_buffer_call_stack.pop()
            self.local_variables = previous_locals
            self.precise_local_variables = previous_precise_locals
            self.current_return_type = previous_return_type
            self.current_generic_type_substitutions = (
                previous_generic_type_substitutions
            )

    def inline_function_body(self, function_node) -> Optional[SpirvId]:
        body = getattr(function_node, "body", [])
        statements = (
            body.statements
            if hasattr(body, "statements")
            else body if isinstance(body, list) else [body]
        )

        for stmt in statements:
            if isinstance(stmt, ReturnNode):
                if getattr(stmt, "value", None) is None:
                    return None
                if isinstance(stmt.value, ArrayLiteralNode):
                    return self.process_array_literal(
                        stmt.value, self.current_return_type
                    )
                return self.process_expression(stmt.value)

            self.process_statement(stmt)

        return None

    def collect_function_execution_models(self, ast):
        functions = {
            getattr(func, "name", None): func
            for func in self.collect_ast_functions(ast)
        }
        functions = {name: func for name, func in functions.items() if name}

        calls_by_function = {}
        for function_name, func in functions.items():
            calls_by_function[function_name] = {
                call_name
                for call_name in (
                    self.function_call_name(call)
                    for call in self.walk_ast_nodes(getattr(func, "body", []))
                    if isinstance(call, FunctionCallNode)
                )
                if call_name in functions
            }

        execution_models = {function_name: set() for function_name in functions}

        def mark_callgraph(entry_name: str, execution_model: str):
            pending = [entry_name]
            visited = set()
            while pending:
                function_name = pending.pop()
                if function_name in visited or function_name not in functions:
                    continue
                visited.add(function_name)
                execution_models[function_name].add(execution_model)
                pending.extend(calls_by_function.get(function_name, ()))

        stage_qualifiers = {
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
        for func in getattr(ast, "functions", []):
            qualifier = self.get_function_qualifier(func)
            if func.name == "main" or qualifier in stage_qualifiers:
                mark_callgraph(func.name, self.spirv_execution_model(qualifier))

        for stage_type, stage in (getattr(ast, "stages", None) or {}).items():
            stage_name = self.stage_key(stage_type)
            execution_model = self.spirv_execution_model(stage_name)
            entry_function = getattr(stage, "entry_point", None)
            if entry_function is not None:
                mark_callgraph(entry_function.name, execution_model)

        return {
            function_name: models
            for function_name, models in execution_models.items()
            if models
        }

    def unused_array_parameter_indices(self, func):
        indices = set()
        body = getattr(func, "body", [])
        for index, param in enumerate(
            getattr(func, "parameters", getattr(func, "params", [])) or []
        ):
            param_name = getattr(param, "name", None)
            if not param_name:
                continue
            param_type = getattr(param, "param_type", getattr(param, "vtype", None))
            type_name = str(self.type_name_from_value(param_type) or "")
            if "[" not in type_name:
                continue
            if self.is_resource_type_name(self.array_base_type_name(type_name)):
                continue
            if not self.function_body_uses_identifier(body, param_name):
                indices.add(index)
        return indices

    def collect_unused_array_parameter_index_maps(self, functions):
        by_name_values = {}
        by_id = {}
        for func in functions or []:
            func_name = getattr(func, "name", None)
            if not func_name:
                continue
            indices = self.unused_array_parameter_indices(func)
            by_id[id(func)] = indices
            by_name_values.setdefault(func_name, []).append(indices)

        by_name = {}
        for func_name, index_sets in by_name_values.items():
            if not index_sets:
                continue
            first_indices = index_sets[0]
            if not first_indices:
                continue
            if all(indices == first_indices for indices in index_sets[1:]):
                by_name[func_name] = set(first_indices)
        return by_name, by_id

    def function_body_uses_identifier(self, body, name):
        for node in self.walk_ast_nodes(body):
            if getattr(node, "name", None) == name:
                return True
        return False

    def skipped_function_parameter_indices(self, func_name):
        return self.spirv_skipped_function_parameter_indices.get(func_name, set())

    def skipped_function_parameter_indices_for_node(self, function_node):
        function_key = id(function_node)
        if function_key in self.spirv_skipped_function_parameter_indices_by_id:
            return self.spirv_skipped_function_parameter_indices_by_id[function_key]
        return self.skipped_function_parameter_indices(
            getattr(function_node, "name", None)
        )

    def collect_function_stage_object_dependencies(
        self, ast, target_stage, object_name
    ):
        direct_dependencies = {}
        function_calls = {}
        object_types = {}

        for func in self.collect_ast_functions(ast):
            func_name = getattr(func, "name", None)
            if func_name:
                direct_dependencies.setdefault(func_name, None)
                function_calls.setdefault(
                    func_name, self.called_user_function_names(func)
                )

        for stage_type, stage in getattr(ast, "stages", {}).items():
            stage_name = self.stage_key(stage_type)
            if target_stage is not None and stage_name != self.stage_key(target_stage):
                continue
            entry_point = getattr(stage, "entry_point", None)
            if object_name == "output":
                object_type = getattr(entry_point, "return_type", None)
                if (
                    object_type is None
                    or self.type_name_from_value(object_type) == "void"
                ):
                    continue
            else:
                object_type = None
                for param in (
                    getattr(
                        entry_point, "parameters", getattr(entry_point, "params", [])
                    )
                    or []
                ):
                    if getattr(param, "name", None) == object_name:
                        object_type = getattr(
                            param, "param_type", getattr(param, "vtype", None)
                        )
                        break
                if object_type is None:
                    continue

            stage_functions = list(getattr(stage, "local_functions", []) or [])
            if entry_point is not None:
                stage_functions.append(entry_point)
            for func in stage_functions:
                func_name = getattr(func, "name", None)
                if not func_name:
                    continue
                object_types[func_name] = object_type
                if self.direct_stage_object_dependency(func, object_name):
                    direct_dependencies[func_name] = object_type
                function_calls[func_name] = self.called_user_function_names(func)

        dependencies = dict(direct_dependencies)
        changed = True
        while changed:
            changed = False
            for func_name, calls in function_calls.items():
                if dependencies.get(func_name) is not None:
                    continue
                for called_name in calls:
                    if dependencies.get(called_name) is not None:
                        dependencies[func_name] = object_types.get(
                            func_name
                        ) or dependencies.get(called_name)
                        changed = True
                        break

        return {
            func_name: object_type
            for func_name, object_type in dependencies.items()
            if object_type is not None
        }

    def direct_stage_object_dependency(self, func, object_name):
        parameter_names = {
            getattr(param, "name", None)
            for param in getattr(func, "parameters", getattr(func, "params", [])) or []
        }
        if object_name in parameter_names:
            return False
        local_names = set(parameter_names)
        for node in self.walk_ast_nodes(getattr(func, "body", [])):
            if isinstance(node, VariableNode) and getattr(node, "name", None):
                local_names.add(node.name)
        if object_name in local_names:
            return False
        for node in self.walk_ast_nodes(getattr(func, "body", [])):
            if isinstance(node, MemberAccessNode):
                if self.expression_name(getattr(node, "object", None)) == object_name:
                    return True
            if getattr(node, "name", None) == object_name:
                return True
        return False

    def called_user_function_names(self, func):
        names = set()
        for node in self.walk_ast_nodes(getattr(func, "body", [])):
            if not isinstance(node, FunctionCallNode):
                continue
            name = self.function_call_name(node)
            if name:
                names.add(name)
        return names

    def required_function_stage_input_type(self, func_name):
        return self.function_stage_input_dependencies.get(func_name)

    def required_function_stage_output_type(self, func_name):
        return self.function_stage_output_dependencies.get(func_name)

    def required_function_stage_object_argument_names(self, func_name):
        names = []
        if self.required_function_stage_input_type(func_name) is not None:
            names.append("input")
        if self.required_function_stage_output_type(func_name) is not None:
            names.append("output")
        return names

    def stage_object_pointer_parameter(self, name, type_source):
        parameter = VariableNode(name, type_source)
        parameter._spirv_stage_object_pointer = True
        return parameter

    def collect_storage_image_pointer_parameters_for_functions(self, function_nodes):
        functions = {getattr(func, "name", None): func for func in function_nodes}
        functions = {name: func for name, func in functions.items() if name}
        parameter_names = {
            func_name: [
                getattr(param, "name", None)
                for param in getattr(func, "parameters", getattr(func, "params", []))
            ]
            for func_name, func in functions.items()
        }
        parameter_name_sets = {
            func_name: {name for name in names if name}
            for func_name, names in parameter_names.items()
        }

        pointer_params = {func_name: set() for func_name in functions}
        for func_name, func in functions.items():
            for call in self.walk_ast_nodes(getattr(func, "body", [])):
                if not isinstance(call, FunctionCallNode):
                    continue
                if (
                    self.function_call_name(call)
                    not in self.image_atomic_function_names()
                ):
                    continue
                args = getattr(call, "arguments", getattr(call, "args", []))
                if not args:
                    continue
                arg_name = self.expression_name(args[0])
                if arg_name in parameter_name_sets[func_name]:
                    pointer_params[func_name].add(arg_name)

        changed = True
        while changed:
            changed = False
            for caller_name, func in functions.items():
                caller_params = parameter_name_sets[caller_name]
                for call in self.walk_ast_nodes(getattr(func, "body", [])):
                    if not isinstance(call, FunctionCallNode):
                        continue
                    callee_name = self.function_call_name(call)
                    if callee_name not in functions:
                        continue
                    callee_pointer_params = pointer_params.get(callee_name, set())
                    if not callee_pointer_params:
                        continue

                    args = getattr(call, "arguments", getattr(call, "args", []))
                    callee_params = parameter_names.get(callee_name, [])
                    for index, arg in enumerate(args):
                        if index >= len(callee_params):
                            continue
                        if callee_params[index] not in callee_pointer_params:
                            continue
                        arg_name = self.expression_name(arg)
                        if arg_name not in caller_params:
                            continue
                        if arg_name not in pointer_params[caller_name]:
                            pointer_params[caller_name].add(arg_name)
                            changed = True

        return {
            func_name: params for func_name, params in pointer_params.items() if params
        }

    def collect_storage_image_pointer_parameters(self, ast):
        return self.collect_storage_image_pointer_parameters_for_functions(
            self.collect_ast_functions(ast)
        )

    def collect_stage_local_storage_image_pointer_metadata(self, ast):
        self.stage_local_function_storage_image_pointer_params = {}
        global_functions = list(getattr(ast, "functions", []) or [])

        for stage in (getattr(ast, "stages", None) or {}).values():
            local_functions = list(getattr(stage, "local_functions", []) or [])
            if not local_functions:
                continue

            local_names = {
                getattr(func, "name", None)
                for func in local_functions
                if getattr(func, "name", None)
            }
            visible_functions = [
                func
                for func in global_functions
                if getattr(func, "name", None) not in local_names
            ] + local_functions
            pointer_params = (
                self.collect_storage_image_pointer_parameters_for_functions(
                    visible_functions
                )
            )

            for function_node in local_functions:
                function_name = getattr(function_node, "name", None)
                if not function_name:
                    continue
                key = self.stage_local_function_key(stage, function_name)
                if key is None:
                    continue
                self.stage_local_function_storage_image_pointer_params[key] = (
                    pointer_params.get(function_name, set())
                )

    def collect_resource_array_parameter_type_hints_for_functions(
        self, function_nodes, global_nodes
    ):
        functions = {getattr(func, "name", None): func for func in function_nodes}
        functions = {name: func for name, func in functions.items() if name}

        global_types = {}
        for node in self.walk_ast_nodes(global_nodes):
            if isinstance(node, VariableNode):
                global_types[node.name] = self.type_name_from_value(
                    getattr(node, "var_type", getattr(node, "vtype", "float"))
                )

        declared_param_types = {}
        for func_name, func in functions.items():
            declared_param_types[func_name] = {}
            for param in getattr(func, "parameters", getattr(func, "params", [])):
                param_name = getattr(param, "name", None)
                param_type = getattr(param, "param_type", getattr(param, "vtype", None))
                if param_name and param_type is not None:
                    declared_param_types[func_name][param_name] = (
                        self.type_name_from_value(param_type)
                    )

        hints = {func_name: {} for func_name in functions}

        def visible_types(func_name):
            visible = dict(global_types)
            for param_name, param_type in declared_param_types.get(
                func_name, {}
            ).items():
                visible[param_name] = hints.get(func_name, {}).get(
                    param_name, param_type
                )
            return visible

        changed = True
        while changed:
            changed = False
            for caller_name, func in functions.items():
                caller_visible_types = visible_types(caller_name)
                for call in self.walk_ast_nodes(getattr(func, "body", [])):
                    if not isinstance(call, FunctionCallNode):
                        continue
                    callee_name = self.function_call_name(call)
                    callee = functions.get(callee_name)
                    if callee is None:
                        continue

                    callee_params = getattr(
                        callee, "parameters", getattr(callee, "params", [])
                    )
                    args = getattr(call, "arguments", getattr(call, "args", []))
                    for index, arg in enumerate(args):
                        if index >= len(callee_params):
                            continue

                        param = callee_params[index]
                        param_name = getattr(param, "name", None)
                        declared_type = declared_param_types.get(callee_name, {}).get(
                            param_name
                        )
                        if not param_name or declared_type is None:
                            continue

                        arg_name = self.expression_name(arg)
                        arg_type = caller_visible_types.get(arg_name)
                        if arg_type is None:
                            continue

                        if not self.resource_array_param_type_is_compatible(
                            declared_type, arg_type
                        ):
                            raise ValueError(
                                "Incompatible SPIR-V resource array parameter shape "
                                f"for '{param_name}': expected {declared_type}, "
                                f"got {arg_type}"
                            )

                        fixed_type = self.fixed_type_for_unsized_resource_param(
                            declared_type, arg_type
                        )
                        if fixed_type is None:
                            continue

                        existing = hints.setdefault(callee_name, {}).get(param_name)
                        if existing is not None and existing != fixed_type:
                            raise ValueError(
                                "Conflicting SPIR-V resource array parameter sizes for "
                                f"'{param_name}': {existing} and {fixed_type}"
                            )
                        if existing != fixed_type:
                            hints[callee_name][param_name] = fixed_type
                            changed = True

        return {
            func_name: param_hints
            for func_name, param_hints in hints.items()
            if param_hints
        }

    def collect_resource_array_parameter_type_hints(self, ast):
        global_nodes = list(getattr(ast, "global_variables", []) or [])
        for stage in (getattr(ast, "stages", None) or {}).values():
            global_nodes.extend(getattr(stage, "local_variables", []) or [])

        return self.collect_resource_array_parameter_type_hints_for_functions(
            self.collect_ast_functions(ast),
            global_nodes,
        )

    def collect_stage_local_resource_array_parameter_type_hints(self, ast):
        self.stage_local_function_resource_array_type_hints = {}
        global_functions = list(getattr(ast, "functions", []) or [])

        for stage in (getattr(ast, "stages", None) or {}).values():
            local_functions = list(getattr(stage, "local_functions", []) or [])
            if not local_functions:
                continue

            local_names = {
                getattr(func, "name", None)
                for func in local_functions
                if getattr(func, "name", None)
            }
            visible_functions = [
                func
                for func in global_functions
                if getattr(func, "name", None) not in local_names
            ] + local_functions
            entry_function = getattr(stage, "entry_point", None)
            if entry_function is not None:
                visible_functions.append(entry_function)

            global_nodes = list(getattr(ast, "global_variables", []) or [])
            global_nodes.extend(getattr(stage, "local_variables", []) or [])
            hints = self.collect_resource_array_parameter_type_hints_for_functions(
                visible_functions,
                global_nodes,
            )

            for function_node in local_functions:
                function_name = getattr(function_node, "name", None)
                if not function_name:
                    continue
                key = self.stage_local_function_key(stage, function_name)
                if key is None:
                    continue
                self.stage_local_function_resource_array_type_hints[key] = hints.get(
                    function_name, {}
                )

    def split_outer_array_type(self, type_name: str):
        if not type_name or "[" not in type_name or not type_name.endswith("]"):
            return None

        open_bracket = type_name.find("[")
        close_bracket = type_name.find("]", open_bracket)
        if close_bracket == -1:
            return None

        base_type = type_name[:open_bracket]
        remaining_suffix = type_name[close_bracket + 1 :]
        element_type = (
            f"{base_type}{remaining_suffix}" if remaining_suffix else base_type
        )
        size_text = type_name[open_bracket + 1 : close_bracket].strip()
        if not size_text:
            return element_type, None

        literal_size = evaluate_literal_int_expression(
            size_text, self.literal_int_constants
        )
        return element_type, literal_size

    def array_base_type_name(self, type_name: str):
        if not type_name or "[" not in type_name:
            return type_name
        return type_name[: type_name.find("[")]

    def get_variable_value(self, variable_id: SpirvId) -> SpirvId:
        self.mark_interface_variable_if_needed(variable_id)
        wrapped_pointer = self.uniform_block_wrapped_member_pointer(variable_id)
        if wrapped_pointer is not None:
            return self.get_variable_value(wrapped_pointer)

        value_type = self.variable_value_types.get(variable_id.id)
        if value_type:
            return self.load_from_variable(variable_id, value_type)

        if variable_id.type.storage_class:
            base_type = variable_id.type.base_type.replace("ptr_", "", 1)
            var_type = self.find_registered_type_by_base(base_type)
            if var_type:
                return self.load_from_variable(variable_id, var_type)
        return variable_id

    def cbuffer_member_pointer(self, name: str) -> Optional[SpirvId]:
        member_info = self.cbuffer_members.get(name)
        if member_info is None:
            return None

        cbuffer_var, member_type, member_index = member_info
        int_type = self.primitive_types["int"]
        index = self.register_constant(member_index, int_type)
        ptr_type = self.register_pointer_type(member_type, "Uniform")
        access = self.access_chain(cbuffer_var, [index], ptr_type)
        self.variable_value_types[access.id] = member_type
        return access

    def uniform_block_wrapped_member_type(
        self, variable_id: Optional[SpirvId]
    ) -> Optional[SpirvId]:
        if variable_id is None:
            return None
        metadata = self.uniform_block_wrapped_variables.get(variable_id.id)
        if metadata is None:
            return None
        return metadata["member_type"]

    def uniform_block_wrapped_member_pointer(
        self, variable_id: Optional[SpirvId]
    ) -> Optional[SpirvId]:
        metadata = (
            self.uniform_block_wrapped_variables.get(variable_id.id)
            if variable_id is not None
            else None
        )
        if metadata is None:
            return None

        int_type = self.primitive_types["int"]
        index = self.register_constant(metadata["member_index"], int_type)
        member_type = metadata["member_type"]
        ptr_type = self.register_pointer_type(member_type, "Uniform")
        access = self.access_chain(variable_id, [index], ptr_type)
        self.variable_value_types[access.id] = member_type
        return access

    def structured_buffer_metadata_for_pointer(self, pointer_id: SpirvId):
        metadata = self.structured_buffer_metadata.get(pointer_id.id)
        if metadata is not None:
            return metadata

        pointee_type = self.variable_value_types.get(pointer_id.id)
        if pointee_type is not None:
            return self.structured_buffer_metadata.get(pointee_type.id)

        return None

    def storage_buffer_access_metadata_for_pointer(self, pointer_id: SpirvId):
        if pointer_id is None:
            return None

        metadata = self.storage_buffer_access_metadata.get(pointer_id.id)
        if metadata is not None:
            return metadata

        metadata = self.structured_buffer_metadata_for_pointer(pointer_id)
        if metadata is not None:
            return metadata

        pointee_type = self.variable_value_types.get(pointer_id.id)
        if pointee_type is not None:
            return self.storage_buffer_access_metadata.get(pointee_type.id)

        return None

    def propagate_storage_buffer_access_metadata(
        self, source_pointer: SpirvId, target_pointer: SpirvId
    ):
        metadata = self.storage_buffer_access_metadata_for_pointer(source_pointer)
        if metadata is not None:
            self.storage_buffer_access_metadata[target_pointer.id] = metadata

    def propagate_structured_buffer_descriptor_access_metadata(
        self, source_pointer: SpirvId, target_pointer: SpirvId, index: SpirvId
    ):
        metadata = self.structured_buffer_metadata_for_pointer(source_pointer)
        if metadata is None or not metadata.get("descriptor_array"):
            return

        source_type = self.variable_value_types.get(source_pointer.id)
        block_type = metadata.get("block_type")
        if not self.array_type_contains_element_type(source_type, block_type):
            return

        descriptor_indices = list(metadata.get("_descriptor_indices", []))
        descriptor_indices.append(index)
        access_metadata = {**metadata, "_descriptor_indices": descriptor_indices}
        self.structured_buffer_metadata[target_pointer.id] = access_metadata
        self.storage_buffer_access_metadata[target_pointer.id] = access_metadata

    def structured_buffer_element_pointer(
        self, buffer_pointer: SpirvId, index_id: SpirvId
    ) -> Optional[SpirvId]:
        metadata = self.structured_buffer_metadata_for_pointer(buffer_pointer)
        if metadata is None:
            return None
        if metadata.get("_access_path") == "member":
            return None

        pointee_type = self.variable_value_types.get(buffer_pointer.id)
        descriptor_array = self.array_type_info_from_type(pointee_type)
        block_type = metadata.get("block_type")
        if (
            descriptor_array is not None
            and block_type is not None
            and self.array_type_contains_element_type(pointee_type, block_type)
        ):
            return None

        element_type = metadata["element_type"]
        int_type = self.primitive_types["int"]
        member_index = self.register_constant(metadata.get("member_index", 0), int_type)
        ptr_type = self.register_pointer_type(
            element_type, buffer_pointer.type.storage_class or "Uniform"
        )
        access = self.access_chain(buffer_pointer, [member_index, index_id], ptr_type)
        self.variable_value_types[access.id] = element_type
        access_metadata = {**metadata, "_access_path": "element"}
        self.structured_buffer_metadata[access.id] = access_metadata
        return access

    def single_struct_buffer_zero_index_alias(
        self, buffer_pointer: SpirvId, index_expr
    ) -> Optional[SpirvId]:
        """Treat a single buffer block indexed as [0] as an alias for the block."""
        if self.literal_int_argument(index_expr) != 0:
            return None

        if self.structured_buffer_metadata_for_pointer(buffer_pointer) is not None:
            return None

        metadata = self.storage_buffer_access_metadata_for_pointer(buffer_pointer)
        if metadata is None or metadata.get("kind") != "glsl_buffer_block":
            return None

        value_type = self.variable_value_types.get(buffer_pointer.id)
        if value_type is None:
            return None
        if self.array_type_info_from_type(value_type) is not None:
            return None
        if value_type.type.base_type not in self.current_struct_members:
            return None

        return buffer_pointer

    def structured_buffer_default_value(self, metadata) -> SpirvId:
        element_type = metadata.get("element_type") if metadata is not None else None
        if element_type is not None:
            return self.default_value_for_type(element_type)
        return self.register_constant(0.0, self.register_primitive_type("float"))

    def structured_buffer_counter_default_value(self) -> SpirvId:
        return self.register_constant(0, self.register_primitive_type("uint"))

    def structured_buffer_dimensions_default_value(self) -> SpirvId:
        return self.register_constant(0, self.register_primitive_type("uint"))

    def emit_structured_buffer_dimensions(
        self, buffer_pointer: SpirvId, metadata, diagnostic_name: str
    ) -> SpirvId:
        block_type = metadata.get("block_type") if metadata is not None else None
        if block_type is None:
            self.emit(f"; WARNING: {diagnostic_name} requires a buffer block operand")
            return self.structured_buffer_dimensions_default_value()

        uint_type = self.register_primitive_type("uint")
        id_value = self.get_id()
        self.emit(
            f"%{id_value} = OpArrayLength %{uint_type.id} %{buffer_pointer.id} "
            f"{metadata.get('member_index', 0)}"
        )
        self.value_types[id_value] = uint_type
        length = SpirvId(id_value, uint_type.type)
        if not metadata.get("byte_address"):
            return length

        byte_stride = self.register_constant(4, uint_type)
        return self.binary_operation("*", uint_type, length, byte_stride)

    def store_structured_buffer_dimensions_result(
        self,
        target_expr,
        value: SpirvId,
        diagnostic_name: str,
    ) -> bool:
        target_pointer = self.assignable_pointer_from_expression(target_expr)
        if target_pointer is None:
            self.emit(
                f"; WARNING: {diagnostic_name} output operand must be an assignable "
                "integer target"
            )
            return False

        target_type = self.pointer_pointee_type(target_pointer)
        if target_type is None:
            self.emit(
                f"; WARNING: {diagnostic_name} output operand type could not be "
                "determined"
            )
            return False

        target_type_name = self.normalize_primitive_name(target_type.type.base_type)
        if target_type_name not in {"int", "uint"}:
            self.emit(
                f"; WARNING: {diagnostic_name} output operand must be an integer "
                "target"
            )
            return False

        self.store_to_variable(
            target_pointer, self.convert_value_to_type(value, target_type)
        )
        return True

    def process_buffer_dimensions_function_call(
        self, expr: FunctionCallNode
    ) -> SpirvId:
        diagnostic_name = "buffer_dimensions"
        args = list(getattr(expr, "args", []) or [])
        if not args:
            self.emit("; WARNING: buffer_dimensions requires a buffer operand")
            return self.structured_buffer_dimensions_default_value()
        if len(args) > 2:
            self.emit(
                "; WARNING: buffer_dimensions accepts only buffer and optional "
                "output operands"
            )
            return self.structured_buffer_dimensions_default_value()

        buffer_pointer = self.variable_pointer_from_expression(args[0])
        if buffer_pointer is None:
            self.emit(
                "; WARNING: buffer_dimensions requires a structured or byte-address "
                "buffer operand"
            )
            return self.structured_buffer_dimensions_default_value()

        metadata = self.structured_buffer_metadata_for_pointer(buffer_pointer)
        if metadata is None:
            self.emit(
                "; WARNING: buffer_dimensions requires a structured or byte-address "
                "buffer operand"
            )
            return self.structured_buffer_dimensions_default_value()

        length = self.emit_structured_buffer_dimensions(
            buffer_pointer, metadata, diagnostic_name
        )
        if len(args) == 2:
            self.store_structured_buffer_dimensions_result(
                args[1], length, diagnostic_name
            )
        return length

    def process_structured_buffer_dimensions_method_call(
        self,
        buffer_pointer: SpirvId,
        metadata,
        args,
        diagnostic_name: str,
    ) -> SpirvId:
        if len(args) > 1:
            self.emit(
                f"; WARNING: {diagnostic_name} accepts only an optional output "
                "operand"
            )
            return self.structured_buffer_dimensions_default_value()

        length = self.emit_structured_buffer_dimensions(
            buffer_pointer, metadata, diagnostic_name
        )
        if args:
            self.store_structured_buffer_dimensions_result(
                args[0], length, diagnostic_name
            )
        return length

    def ensure_structured_buffer_counter_metadata(self, metadata) -> bool:
        if metadata.get("counter_variable") is not None:
            return True
        if metadata.get("buffer_kind") != "RWStructuredBuffer":
            return False

        root_pointer = metadata.get("buffer_variable")
        root_metadata = (
            self.structured_buffer_metadata.get(root_pointer.id)
            if root_pointer is not None
            else None
        )
        if root_metadata is not None and root_metadata.get("counter_variable"):
            counter_metadata = {
                key: value
                for key, value in root_metadata.items()
                if key.startswith("counter_")
            }
            counter_metadata["counter_variable"] = root_metadata["counter_variable"]
            metadata.update(counter_metadata)
            return True

        declaration_node = metadata.get("declaration_node")
        declared_type_name = metadata.get("declared_type_name")
        if declaration_node is None or declared_type_name is None:
            return False

        counter_metadata = self.process_structured_buffer_counter_declaration(
            declaration_node, declared_type_name
        )
        metadata.update(counter_metadata)
        if root_metadata is not None:
            root_metadata.update(counter_metadata)
            self.structured_buffer_metadata[root_pointer.id] = root_metadata
            block_type = root_metadata.get("block_type")
            if block_type is not None:
                self.structured_buffer_metadata[block_type.id] = root_metadata
        return True

    def structured_buffer_counter_pointer(
        self, buffer_pointer: SpirvId
    ) -> Optional[SpirvId]:
        metadata = self.structured_buffer_metadata_for_pointer(buffer_pointer)
        if metadata is None:
            return None

        counter_pointer = metadata.get("counter_variable")
        counter_block_type = metadata.get("counter_block_type")
        if counter_pointer is None or counter_block_type is None:
            return None

        counter_value_type = self.variable_value_types.get(counter_pointer.id)
        for index in metadata.get("_descriptor_indices", []):
            counter_element_type = self.array_element_type_from_type(counter_value_type)
            if counter_element_type is None:
                return None

            ptr_type = self.register_pointer_type(counter_element_type, "Uniform")
            counter_pointer = self.access_chain(counter_pointer, [index], ptr_type)
            self.variable_value_types[counter_pointer.id] = counter_element_type
            counter_value_type = counter_element_type

        if counter_value_type is None or counter_value_type.id != counter_block_type.id:
            return None

        uint_type = self.register_primitive_type("uint")
        member_index = self.register_constant(
            metadata.get("counter_member_index", 0), self.primitive_types["int"]
        )
        ptr_type = self.register_pointer_type(uint_type, "Uniform")
        access = self.access_chain(counter_pointer, [member_index], ptr_type)
        self.variable_value_types[access.id] = uint_type
        self.storage_buffer_access_metadata[access.id] = {
            "kind": "structured_buffer_counter",
            "block_type": counter_block_type,
            "readonly": False,
            "writeonly": False,
        }
        return access

    def emit_structured_buffer_counter_atomic(
        self, opcode: str, counter_pointer: SpirvId, value: SpirvId
    ) -> SpirvId:
        uint_type = self.register_primitive_type("uint")
        value = self.convert_value_to_type(value, uint_type)
        scope = self.spirv_scope_constant("Device")
        semantics = self.spirv_memory_semantics_constant()
        id_value = self.get_id()
        self.emit(
            f"%{id_value} = {opcode} %{uint_type.id} %{counter_pointer.id} "
            f"%{scope.id} %{semantics.id} %{value.id}"
        )
        self.value_types[id_value] = uint_type
        return SpirvId(id_value, uint_type.type)

    def process_structured_buffer_append_call(
        self,
        buffer_pointer: SpirvId,
        metadata,
        args,
        diagnostic_name: str,
    ) -> Tuple[bool, Optional[SpirvId]]:
        if len(args) < 1:
            self.emit(f"; WARNING: {diagnostic_name} requires a value operand")
            return True, None
        if len(args) > 1:
            self.emit(f"; WARNING: {diagnostic_name} accepts only a value operand")
            return True, None
        if metadata.get("buffer_kind") != "AppendStructuredBuffer":
            self.emit(
                f"; WARNING: {diagnostic_name} requires an AppendStructuredBuffer"
            )
            return True, None
        if metadata.get("readonly"):
            self.emit(f"; WARNING: {diagnostic_name} requires a writable buffer")
            return True, None

        value = (
            args[0]
            if isinstance(args[0], SpirvId)
            else self.process_expression(args[0])
        )
        if value is None:
            self.emit(f"; WARNING: {diagnostic_name} value could not be evaluated")
            return True, None
        value = self.convert_value_to_type(value, metadata["element_type"])

        counter_pointer = self.structured_buffer_counter_pointer(buffer_pointer)
        if counter_pointer is None:
            self.emit(f"; WARNING: {diagnostic_name} requires a counter buffer")
            return True, None

        one = self.register_constant(1, self.register_primitive_type("uint"))
        index = self.emit_structured_buffer_counter_atomic(
            "OpAtomicIAdd", counter_pointer, one
        )
        element_pointer = self.structured_buffer_element_pointer(buffer_pointer, index)
        if element_pointer is None:
            self.emit(
                f"; WARNING: {diagnostic_name} requires an AppendStructuredBuffer "
                "element"
            )
            return True, None

        self.store_to_variable(element_pointer, value)
        return True, None

    def process_structured_buffer_consume_call(
        self,
        buffer_pointer: SpirvId,
        metadata,
        args,
        diagnostic_name: str,
    ) -> Tuple[bool, Optional[SpirvId]]:
        if args:
            self.emit(f"; WARNING: {diagnostic_name} accepts no operands")
            return True, self.structured_buffer_default_value(metadata)
        if metadata.get("buffer_kind") != "ConsumeStructuredBuffer":
            self.emit(
                f"; WARNING: {diagnostic_name} requires a ConsumeStructuredBuffer"
            )
            return True, self.structured_buffer_default_value(metadata)
        if metadata.get("writeonly"):
            self.emit(f"; WARNING: {diagnostic_name} requires a readable buffer")
            return True, self.structured_buffer_default_value(metadata)

        counter_pointer = self.structured_buffer_counter_pointer(buffer_pointer)
        if counter_pointer is None:
            self.emit(f"; WARNING: {diagnostic_name} requires a counter buffer")
            return True, self.structured_buffer_default_value(metadata)

        uint_type = self.register_primitive_type("uint")
        one = self.register_constant(1, uint_type)
        old_count = self.emit_structured_buffer_counter_atomic(
            "OpAtomicISub", counter_pointer, one
        )
        index = self.binary_operation("-", uint_type, old_count, one)
        element_pointer = self.structured_buffer_element_pointer(buffer_pointer, index)
        if element_pointer is None:
            self.emit(
                f"; WARNING: {diagnostic_name} requires a ConsumeStructuredBuffer "
                "element"
            )
            return True, self.structured_buffer_default_value(metadata)

        element_type = self.variable_value_types[element_pointer.id]
        return True, self.load_from_variable(element_pointer, element_type)

    def process_structured_buffer_counter_method_call(
        self,
        buffer_pointer: SpirvId,
        metadata,
        args,
        diagnostic_name: str,
        increment: bool,
    ) -> Tuple[bool, SpirvId]:
        if args:
            self.emit(f"; WARNING: {diagnostic_name} accepts no operands")
            return True, self.structured_buffer_counter_default_value()
        if metadata.get("buffer_kind") != "RWStructuredBuffer":
            self.emit(f"; WARNING: {diagnostic_name} requires an RWStructuredBuffer")
            return True, self.structured_buffer_counter_default_value()
        if metadata.get("readonly") or metadata.get("writeonly"):
            self.emit(f"; WARNING: {diagnostic_name} requires a read-write buffer")
            return True, self.structured_buffer_counter_default_value()
        if not self.ensure_structured_buffer_counter_metadata(metadata):
            self.emit(f"; WARNING: {diagnostic_name} requires a counter buffer")
            return True, self.structured_buffer_counter_default_value()

        counter_pointer = self.structured_buffer_counter_pointer(buffer_pointer)
        if counter_pointer is None:
            self.emit(f"; WARNING: {diagnostic_name} requires a counter buffer")
            return True, self.structured_buffer_counter_default_value()

        uint_type = self.register_primitive_type("uint")
        one = self.register_constant(1, uint_type)
        if increment:
            counter_value = self.emit_structured_buffer_counter_atomic(
                "OpAtomicIAdd", counter_pointer, one
            )
            return True, counter_value

        old_count = self.emit_structured_buffer_counter_atomic(
            "OpAtomicISub", counter_pointer, one
        )
        return True, self.binary_operation("-", uint_type, old_count, one)

    def process_structured_buffer_method_call(
        self, expr: FunctionCallNode
    ) -> Tuple[bool, Optional[SpirvId]]:
        callee_expr = getattr(expr, "function", getattr(expr, "name", None))
        if not isinstance(callee_expr, MemberAccessNode):
            return False, None

        method_name = getattr(callee_expr, "member", None)
        if method_name not in {
            "Load",
            "Store",
            "GetDimensions",
            "Append",
            "Consume",
            "IncrementCounter",
            "DecrementCounter",
        }:
            return False, None

        buffer_pointer = self.variable_pointer_from_expression(callee_expr.object)
        if buffer_pointer is None:
            return False, None

        metadata = self.structured_buffer_metadata_for_pointer(buffer_pointer)
        if metadata is None or metadata.get("byte_address"):
            return False, None

        args = list(getattr(expr, "args", []) or [])
        if method_name == "GetDimensions":
            return True, self.process_structured_buffer_dimensions_method_call(
                buffer_pointer,
                metadata,
                args,
                "StructuredBuffer.GetDimensions",
            )
        if method_name == "Append":
            return self.process_structured_buffer_append_call(
                buffer_pointer,
                metadata,
                args,
                "AppendStructuredBuffer.Append",
            )
        if method_name == "Consume":
            return self.process_structured_buffer_consume_call(
                buffer_pointer,
                metadata,
                args,
                "ConsumeStructuredBuffer.Consume",
            )
        if method_name in {"IncrementCounter", "DecrementCounter"}:
            return self.process_structured_buffer_counter_method_call(
                buffer_pointer,
                metadata,
                args,
                f"RWStructuredBuffer.{method_name}",
                method_name == "IncrementCounter",
            )

        if method_name == "Load":
            diagnostic_name = "StructuredBuffer.Load"
            if len(args) < 1:
                self.emit(f"; WARNING: {diagnostic_name} requires an index operand")
                return True, self.structured_buffer_default_value(metadata)
            if len(args) > 1:
                self.emit(f"; WARNING: {diagnostic_name} accepts only an index operand")
                return True, self.structured_buffer_default_value(metadata)
            if metadata.get("writeonly"):
                self.emit(f"; WARNING: {diagnostic_name} requires a readable buffer")
                return True, self.structured_buffer_default_value(metadata)
            if metadata.get("buffer_kind") not in {
                "StructuredBuffer",
                "RWStructuredBuffer",
            }:
                self.emit(
                    f"; WARNING: {diagnostic_name} requires a StructuredBuffer "
                    "element"
                )
                return True, self.structured_buffer_default_value(metadata)

            index = self.process_expression(args[0])
            if index is None:
                self.emit(f"; WARNING: {diagnostic_name} index could not be evaluated")
                return True, self.structured_buffer_default_value(metadata)
            element_pointer = self.structured_buffer_element_pointer(
                buffer_pointer, index
            )
            if element_pointer is None:
                self.emit(
                    f"; WARNING: {diagnostic_name} requires a StructuredBuffer "
                    "element"
                )
                return True, self.structured_buffer_default_value(metadata)

            element_type = self.variable_value_types[element_pointer.id]
            return True, self.load_from_variable(element_pointer, element_type)

        diagnostic_name = "RWStructuredBuffer.Store"
        if len(args) < 2:
            self.emit(f"; WARNING: {diagnostic_name} requires index and value operands")
            return True, None
        if len(args) > 2:
            self.emit(
                f"; WARNING: {diagnostic_name} accepts only index and value operands"
            )
            return True, None
        if metadata.get("readonly"):
            self.emit(f"; WARNING: {diagnostic_name} requires a writable buffer")
            return True, None
        if metadata.get("buffer_kind") != "RWStructuredBuffer":
            self.emit(
                f"; WARNING: {diagnostic_name} requires an RWStructuredBuffer element"
            )
            return True, None

        index = self.process_expression(args[0])
        if index is None:
            self.emit(f"; WARNING: {diagnostic_name} index could not be evaluated")
            return True, None
        element_pointer = self.structured_buffer_element_pointer(buffer_pointer, index)
        if element_pointer is None:
            self.emit(
                f"; WARNING: {diagnostic_name} requires an RWStructuredBuffer element"
            )
            return True, None

        value = self.process_expression(args[1])
        if value is None:
            self.emit(f"; WARNING: {diagnostic_name} value could not be evaluated")
            return True, None

        self.store_to_variable(element_pointer, value)
        return True, None

    def array_type_contains_element_type(
        self, array_type: Optional[SpirvId], target_type: Optional[SpirvId]
    ) -> bool:
        if array_type is None or target_type is None:
            return False

        element_type = self.array_element_type_from_type(array_type)
        while element_type is not None:
            if element_type.id == target_type.id:
                return True
            element_type = self.array_element_type_from_type(element_type)
        return False

    def byte_address_method_load_width(self, method_name: str) -> Optional[int]:
        return {"Load": 1, "Load2": 2, "Load3": 3, "Load4": 4}.get(method_name)

    def byte_address_method_store_width(self, method_name: str) -> Optional[int]:
        return {"Store": 1, "Store2": 2, "Store3": 3, "Store4": 4}.get(method_name)

    def byte_address_method_interlocked_info(self, method_name: str):
        return {
            "InterlockedAdd": {
                "opcode": "OpAtomicIAdd",
                "value_roles": ("value",),
                "min_args": 2,
                "max_args": 3,
                "required": "byte offset and value operands",
                "accepted": "byte offset, value, and optional original operands",
                "original_index": 2,
            },
            "InterlockedAnd": {
                "opcode": "OpAtomicAnd",
                "value_roles": ("value",),
                "min_args": 2,
                "max_args": 3,
                "required": "byte offset and value operands",
                "accepted": "byte offset, value, and optional original operands",
                "original_index": 2,
            },
            "InterlockedOr": {
                "opcode": "OpAtomicOr",
                "value_roles": ("value",),
                "min_args": 2,
                "max_args": 3,
                "required": "byte offset and value operands",
                "accepted": "byte offset, value, and optional original operands",
                "original_index": 2,
            },
            "InterlockedXor": {
                "opcode": "OpAtomicXor",
                "value_roles": ("value",),
                "min_args": 2,
                "max_args": 3,
                "required": "byte offset and value operands",
                "accepted": "byte offset, value, and optional original operands",
                "original_index": 2,
            },
            "InterlockedMin": {
                "opcode": "OpAtomicUMin",
                "value_roles": ("value",),
                "min_args": 2,
                "max_args": 3,
                "required": "byte offset and value operands",
                "accepted": "byte offset, value, and optional original operands",
                "original_index": 2,
            },
            "InterlockedMax": {
                "opcode": "OpAtomicUMax",
                "value_roles": ("value",),
                "min_args": 2,
                "max_args": 3,
                "required": "byte offset and value operands",
                "accepted": "byte offset, value, and optional original operands",
                "original_index": 2,
            },
            "InterlockedExchange": {
                "opcode": "OpAtomicExchange",
                "value_roles": ("value",),
                "min_args": 2,
                "max_args": 3,
                "required": "byte offset and value operands",
                "accepted": "byte offset, value, and optional original operands",
                "original_index": 2,
            },
            "InterlockedCompareExchange": {
                "opcode": "OpAtomicCompareExchange",
                "value_roles": ("compare", "value"),
                "min_args": 4,
                "max_args": 4,
                "required": "byte offset, compare, value, and original operands",
                "accepted": "byte offset, compare, value, and original operands",
                "original_index": 3,
            },
            "InterlockedCompareStore": {
                "opcode": "OpAtomicCompareExchange",
                "value_roles": ("compare", "value"),
                "min_args": 3,
                "max_args": 3,
                "required": "byte offset, compare, and value operands",
                "accepted": "byte offset, compare, and value operands",
                "original_index": None,
            },
        }.get(method_name)

    def byte_address_helper_load_width(self, function_name: str) -> Optional[int]:
        return {"buffer_load2": 2, "buffer_load3": 3, "buffer_load4": 4}.get(
            function_name
        )

    def byte_address_helper_store_width(self, function_name: str) -> Optional[int]:
        return {"buffer_store2": 2, "buffer_store3": 3, "buffer_store4": 4}.get(
            function_name
        )

    def byte_address_value_type(self, component_count: int) -> SpirvId:
        uint_type = self.register_primitive_type("uint")
        if component_count == 1:
            return uint_type
        return self.register_vector_type(uint_type, component_count)

    def byte_address_default_value(self, component_count: int) -> SpirvId:
        return self.default_value_for_type(
            self.byte_address_value_type(component_count)
        )

    def byte_address_interlocked_default_value(self) -> SpirvId:
        return self.default_value_for_type(self.register_primitive_type("uint"))

    def byte_address_interlocked_uint_operand(
        self, expr, diagnostic_name: str, role: str
    ) -> Optional[SpirvId]:
        value = self.process_expression(expr)
        if value is None:
            self.emit(
                f"; WARNING: {diagnostic_name} {role} operand could not be evaluated"
            )
            return None

        value_type = self.value_types.get(
            value.id
        ) or self.find_registered_type_by_base(value.type.base_type)
        if value_type is None:
            self.emit(
                f"; WARNING: {diagnostic_name} {role} operand type could not be "
                "determined"
            )
            return None

        if self.vector_type_info_from_type(value_type) is not None:
            self.emit(
                f"; WARNING: {diagnostic_name} {role} operand must be a scalar integer"
            )
            return None
        if self.matrix_type_info_from_type(value_type) is not None:
            self.emit(
                f"; WARNING: {diagnostic_name} {role} operand must be a scalar integer"
            )
            return None

        type_name = self.normalize_primitive_name(value_type.type.base_type)
        if type_name not in {"int", "uint"}:
            self.emit(f"; WARNING: {diagnostic_name} {role} operand must be an integer")
            return None

        return self.convert_value_to_type(value, self.register_primitive_type("uint"))

    def byte_address_interlocked_original_pointer(
        self, expr, diagnostic_name: str
    ) -> Optional[SpirvId]:
        original_pointer = self.assignable_pointer_from_expression(expr)
        if original_pointer is None:
            self.emit(
                f"; WARNING: {diagnostic_name} original operand must be an "
                "assignable scalar uint target"
            )
            return None

        original_type = self.pointer_pointee_type(original_pointer)
        if original_type is None:
            self.emit(
                f"; WARNING: {diagnostic_name} original operand type could not be "
                "determined"
            )
            return None

        if self.vector_type_info_from_type(original_type) is not None:
            self.emit(
                f"; WARNING: {diagnostic_name} original operand must be scalar uint"
            )
            return None
        if self.matrix_type_info_from_type(original_type) is not None:
            self.emit(
                f"; WARNING: {diagnostic_name} original operand must be scalar uint"
            )
            return None

        type_name = self.normalize_primitive_name(original_type.type.base_type)
        if type_name != "uint":
            self.emit(
                f"; WARNING: {diagnostic_name} original operand must be scalar uint"
            )
            return None

        return original_pointer

    def emit_byte_address_interlocked_atomic(
        self,
        opcode: str,
        target_pointer: SpirvId,
        value_operands: List[SpirvId],
    ) -> SpirvId:
        uint_type = self.register_primitive_type("uint")
        scope = self.spirv_scope_constant("Device")
        semantics = self.spirv_memory_semantics_constant()
        id_value = self.get_id()

        if opcode == "OpAtomicCompareExchange":
            compare_id, value_id = value_operands
            self.emit(
                f"%{id_value} = OpAtomicCompareExchange %{uint_type.id} "
                f"%{target_pointer.id} %{scope.id} %{semantics.id} %{semantics.id} "
                f"%{value_id.id} %{compare_id.id}"
            )
        else:
            value_id = value_operands[0]
            self.emit(
                f"%{id_value} = {opcode} %{uint_type.id} %{target_pointer.id} "
                f"%{scope.id} %{semantics.id} %{value_id.id}"
            )

        self.value_types[id_value] = uint_type
        return SpirvId(id_value, uint_type.type)

    def byte_address_buffer_element_index_from_value(
        self, byte_offset: SpirvId
    ) -> Optional[SpirvId]:
        offset_type_name = self.normalize_primitive_name(byte_offset.type.base_type)
        if offset_type_name not in {"int", "uint"}:
            self.emit("; WARNING: ByteAddressBuffer byte offset must be an integer")
            return None

        uint_type = self.register_primitive_type("uint")
        byte_offset = self.convert_value_to_type(byte_offset, uint_type)
        element_size = self.register_constant(4, uint_type)
        return self.binary_operation("/", uint_type, byte_offset, element_size)

    def byte_address_buffer_element_index(self, offset_expr) -> Optional[SpirvId]:
        byte_offset = self.process_expression(offset_expr)
        if byte_offset is None:
            self.emit("; WARNING: ByteAddressBuffer byte offset could not be evaluated")
            return None

        return self.byte_address_buffer_element_index_from_value(byte_offset)

    def byte_address_component_index(
        self, element_index: SpirvId, component_index: int
    ) -> SpirvId:
        if component_index == 0:
            return element_index

        uint_type = self.register_primitive_type("uint")
        component_offset = self.register_constant(component_index, uint_type)
        return self.binary_operation("+", uint_type, element_index, component_offset)

    def load_byte_address_buffer_value(
        self,
        buffer_pointer: SpirvId,
        element_index: SpirvId,
        component_count: int,
        diagnostic_name: str,
    ) -> SpirvId:
        metadata = self.structured_buffer_metadata_for_pointer(buffer_pointer)
        if metadata is not None and metadata.get("writeonly"):
            self.emit(f"; WARNING: {diagnostic_name} requires a readable buffer")
            return self.byte_address_default_value(component_count)

        uint_type = self.register_primitive_type("uint")
        components = []
        for component_index in range(component_count):
            index = self.byte_address_component_index(element_index, component_index)
            component = self.call_resource_function(
                "buffer_load", [buffer_pointer, index]
            )
            if component is None:
                component = self.default_value_for_type(uint_type)
            components.append(self.convert_value_to_type(component, uint_type))

        if component_count == 1:
            return components[0]

        return self.composite_construct(
            self.byte_address_value_type(component_count), components
        )

    def store_byte_address_buffer_value(
        self,
        buffer_pointer: SpirvId,
        element_index: SpirvId,
        value: SpirvId,
        component_count: int,
        diagnostic_name: str,
    ) -> None:
        uint_type = self.register_primitive_type("uint")
        if component_count == 1:
            value = self.convert_value_to_type(value, uint_type)
            self.call_resource_function(
                "buffer_store", [buffer_pointer, element_index, value]
            )
            return

        value_type = self.value_types.get(
            value.id
        ) or self.find_registered_type_by_base(value.type.base_type)
        value_type_name = (
            value_type.type.base_type
            if value_type is not None
            else value.type.base_type
        )
        vector_info = self.vector_component_type_and_count(value_type_name)
        if vector_info is None or vector_info[1] != component_count:
            self.emit(
                f"; WARNING: {diagnostic_name} requires a uvec{component_count} value"
            )
            return
        if vector_info[0] not in {"int", "uint"}:
            self.emit(f"; WARNING: {diagnostic_name} requires an integer vector value")
            return

        vector_type = self.byte_address_value_type(component_count)
        value = self.convert_value_to_type(value, vector_type)
        if not self.value_has_type(value, vector_type):
            self.emit(
                f"; WARNING: {diagnostic_name} value could not be converted to "
                f"uvec{component_count}"
            )
            return

        for component_index in range(component_count):
            component = self.composite_extract(value, uint_type, component_index)
            index = self.byte_address_component_index(element_index, component_index)
            self.call_resource_function(
                "buffer_store", [buffer_pointer, index, component]
            )

    def call_byte_address_buffer_load_helper(
        self, function_name: str, args: List[SpirvId], component_count: int
    ) -> SpirvId:
        if len(args) < 2:
            self.emit(
                f"; WARNING: {function_name} requires buffer and byte offset operands"
            )
            return self.byte_address_default_value(component_count)
        if len(args) > 2:
            self.emit(
                f"; WARNING: {function_name} accepts only buffer and byte offset "
                "operands"
            )
            return self.byte_address_default_value(component_count)

        metadata = self.structured_buffer_metadata_for_pointer(args[0])
        if metadata is None or not metadata.get("byte_address"):
            self.emit(
                f"; WARNING: {function_name} requires a ByteAddressBuffer operand"
            )
            return self.byte_address_default_value(component_count)

        element_index = self.byte_address_buffer_element_index_from_value(args[1])
        if element_index is None:
            return self.byte_address_default_value(component_count)

        return self.load_byte_address_buffer_value(
            args[0], element_index, component_count, function_name
        )

    def call_byte_address_buffer_store_helper(
        self, function_name: str, args: List[SpirvId], component_count: int
    ) -> None:
        if len(args) < 3:
            self.emit(
                f"; WARNING: {function_name} requires buffer, byte offset, and "
                "value operands"
            )
            return None
        if len(args) > 3:
            self.emit(
                f"; WARNING: {function_name} accepts only buffer, byte offset, and "
                "value operands"
            )
            return None

        metadata = self.structured_buffer_metadata_for_pointer(args[0])
        if metadata is None or not metadata.get("byte_address"):
            self.emit(
                f"; WARNING: {function_name} requires an RWByteAddressBuffer operand"
            )
            return None
        if metadata.get("readonly"):
            self.emit(f"; WARNING: {function_name} requires a writable buffer")
            return None

        element_index = self.byte_address_buffer_element_index_from_value(args[1])
        if element_index is None:
            return None

        self.store_byte_address_buffer_value(
            args[0], element_index, args[2], component_count, function_name
        )
        return None

    def process_byte_address_buffer_interlocked_call(
        self,
        buffer_pointer: SpirvId,
        metadata,
        method_name: str,
        args,
    ) -> SpirvId:
        diagnostic_name = f"RWByteAddressBuffer.{method_name}"
        info = self.byte_address_method_interlocked_info(method_name)
        if info is None:
            return self.byte_address_interlocked_default_value()

        if len(args) < info["min_args"]:
            self.emit(f"; WARNING: {diagnostic_name} requires {info['required']}")
            return self.byte_address_interlocked_default_value()
        if len(args) > info["max_args"]:
            self.emit(f"; WARNING: {diagnostic_name} accepts only {info['accepted']}")
            return self.byte_address_interlocked_default_value()
        if metadata.get("readonly") or metadata.get("writeonly"):
            self.emit(f"; WARNING: {diagnostic_name} requires a read-write buffer")
            return self.byte_address_interlocked_default_value()

        element_index = self.byte_address_buffer_element_index(args[0])
        if element_index is None:
            return self.byte_address_interlocked_default_value()

        value_operands = []
        for role_index, role in enumerate(info["value_roles"], start=1):
            value_operand = self.byte_address_interlocked_uint_operand(
                args[role_index], diagnostic_name, role
            )
            if value_operand is None:
                return self.byte_address_interlocked_default_value()
            value_operands.append(value_operand)

        original_pointer = None
        original_index = info["original_index"]
        if original_index is not None and len(args) > original_index:
            original_pointer = self.byte_address_interlocked_original_pointer(
                args[original_index], diagnostic_name
            )
            if original_pointer is None:
                return self.byte_address_interlocked_default_value()

        target_pointer = self.structured_buffer_element_pointer(
            buffer_pointer, element_index
        )
        if target_pointer is None:
            self.emit(
                f"; WARNING: {diagnostic_name} requires a byte-address buffer element"
            )
            return self.byte_address_interlocked_default_value()

        atomic_result = self.emit_byte_address_interlocked_atomic(
            info["opcode"], target_pointer, value_operands
        )

        if original_pointer is not None:
            self.store_to_variable(original_pointer, atomic_result)

        return atomic_result

    def process_byte_address_buffer_method_call(
        self, expr: FunctionCallNode
    ) -> Tuple[bool, Optional[SpirvId]]:
        callee_expr = getattr(expr, "function", getattr(expr, "name", None))
        if not isinstance(callee_expr, MemberAccessNode):
            return False, None

        method_name = getattr(callee_expr, "member", None)
        load_width = self.byte_address_method_load_width(method_name)
        store_width = self.byte_address_method_store_width(method_name)
        interlocked_info = self.byte_address_method_interlocked_info(method_name)
        if (
            load_width is None
            and store_width is None
            and interlocked_info is None
            and method_name != "GetDimensions"
        ):
            return False, None

        buffer_pointer = self.variable_pointer_from_expression(callee_expr.object)
        if buffer_pointer is None:
            return False, None

        metadata = self.structured_buffer_metadata_for_pointer(buffer_pointer)
        if metadata is None or not metadata.get("byte_address"):
            return False, None

        args = list(getattr(expr, "args", []) or [])
        if method_name == "GetDimensions":
            return True, self.process_structured_buffer_dimensions_method_call(
                buffer_pointer,
                metadata,
                args,
                "ByteAddressBuffer.GetDimensions",
            )

        if interlocked_info is not None:
            return True, self.process_byte_address_buffer_interlocked_call(
                buffer_pointer, metadata, method_name, args
            )

        if load_width is not None:
            diagnostic_name = f"ByteAddressBuffer.{method_name}"
            if len(args) < 1:
                self.emit(f"; WARNING: {diagnostic_name} requires a byte offset")
                return True, self.byte_address_default_value(load_width)
            if len(args) > 1:
                self.emit(f"; WARNING: {diagnostic_name} accepts only a byte offset")
                return True, self.byte_address_default_value(load_width)

            element_index = self.byte_address_buffer_element_index(args[0])
            if element_index is None:
                return True, self.byte_address_default_value(load_width)
            return True, self.load_byte_address_buffer_value(
                buffer_pointer, element_index, load_width, diagnostic_name
            )

        diagnostic_name = f"RWByteAddressBuffer.{method_name}"
        if len(args) < 2:
            self.emit(
                f"; WARNING: {diagnostic_name} requires byte offset and value operands"
            )
            return True, None
        if len(args) > 2:
            self.emit(
                f"; WARNING: {diagnostic_name} accepts only byte offset and value "
                "operands"
            )
            return True, None
        if metadata.get("readonly"):
            self.emit(f"; WARNING: {diagnostic_name} requires a writable buffer")
            return True, None

        element_index = self.byte_address_buffer_element_index(args[0])
        if element_index is None:
            return True, None

        value = self.process_expression(args[1])
        if value is None:
            self.emit(f"; WARNING: {diagnostic_name} value could not be evaluated")
            return True, None

        self.store_byte_address_buffer_value(
            buffer_pointer, element_index, value, store_width, diagnostic_name
        )
        return True, None

    def variable_pointer_from_expression(self, expr) -> Optional[SpirvId]:
        if isinstance(expr, IdentifierNode):
            name = expr.name
        elif isinstance(expr, VariableNode):
            name = expr.name
        elif isinstance(expr, str):
            name = expr
        elif isinstance(expr, ArrayAccessNode):
            index = self.process_expression(expr.index)
            if index is None:
                return None
            access, _ = self.create_array_element_access(expr.array, index, expr.index)
            return access
        elif isinstance(expr, MemberAccessNode):
            base_pointer = self.variable_pointer_from_expression(expr.object)
            if base_pointer is None:
                return None

            return self.create_member_access_pointer(base_pointer, expr.member)
        else:
            return None

        pointer = (
            self.local_variables.get(name)
            or self.resolve_global_variable(name)
            or self.cbuffer_member_pointer(name)
            or self.ensure_tessellation_patch_constant_input(name)
            or self.ensure_compute_builtin(name)
            or self.ensure_stage_builtin(name)
        )
        self.mark_interface_variable_if_needed(pointer)
        wrapped_pointer = self.uniform_block_wrapped_member_pointer(pointer)
        if wrapped_pointer is not None:
            return wrapped_pointer
        return pointer

    def array_element_type_from_type(self, array_type: Optional[SpirvId]):
        if array_type is None:
            return None

        array_info = self.array_type_info_from_type(array_type)
        if array_info is not None:
            return array_info[0]

        vector_info = self.vector_type_info_from_type(array_type)
        if vector_info is not None:
            return vector_info[0]

        matrix_info = self.matrix_type_info_from_type(array_type)
        if matrix_info is not None:
            return matrix_info[0]

        return None

    def array_type_info_from_type(self, array_type: Optional[SpirvId]):
        if array_type is None:
            return None

        for (element_type_id, size), arr_type_id in self.array_types.items():
            if arr_type_id.id == array_type.id:
                return self.find_registered_type_by_id(element_type_id), size
        for (element_type_id, size, _), arr_type_id in self.layout_array_types.items():
            if arr_type_id.id == array_type.id:
                return self.find_registered_type_by_id(element_type_id), size

        return None

    def type_contains_runtime_array(
        self, type_id: Optional[SpirvId], seen: Optional[set] = None
    ) -> bool:
        if type_id is None:
            return False
        if seen is None:
            seen = set()
        if type_id.id in seen:
            return False
        seen.add(type_id.id)

        array_info = self.array_type_info_from_type(type_id)
        if array_info is not None:
            element_type, size = array_info
            return size is None or self.type_contains_runtime_array(element_type, seen)

        struct_members = self.current_struct_members.get(type_id.type.base_type)
        if struct_members is not None:
            return any(
                self.type_contains_runtime_array(member_type, seen)
                for member_type, _ in struct_members
            )

        return False

    def runtime_array_aggregate_fallback(self, reason: str) -> SpirvId:
        self.emit(f"; WARNING: {reason}")
        return self.register_constant(0.0, self.register_primitive_type("float"))

    def vector_type_info_from_type(self, vector_type: Optional[SpirvId]):
        if vector_type is None:
            return None

        for (component_type_id, count), vec_type_id in self.vector_types.items():
            if vec_type_id.id == vector_type.id:
                return self.find_registered_type_by_id(component_type_id), count

        return None

    def matrix_type_info_from_type(self, matrix_type: Optional[SpirvId]):
        if matrix_type is None:
            return None

        for (column_type_id, count), mat_type_id in self.matrix_types.items():
            if mat_type_id.id == matrix_type.id:
                return self.find_registered_type_by_id(column_type_id), count

        return None

    def default_value_for_type(self, type_id: SpirvId) -> SpirvId:
        primitive_name = self.normalize_primitive_name(type_id.type.base_type)
        if primitive_name in {"float", "double"}:
            return self.register_constant(0.0, type_id)
        if primitive_name in self.INTEGER_TYPE_NAMES:
            return self.register_constant(0, type_id)
        if primitive_name == "bool":
            return self.register_constant(False, type_id)
        if self.type_contains_runtime_array(type_id):
            return self.runtime_array_aggregate_fallback(
                "runtime-array aggregate default values cannot be materialized "
                "in SPIR-V"
            )

        vector_info = self.vector_type_info_from_type(type_id)
        if vector_info is not None:
            component_type, count = vector_info
            components = [
                self.default_value_for_type(component_type) for _ in range(count)
            ]
            return self.register_vector_constant(type_id, components)

        matrix_info = self.matrix_type_info_from_type(type_id)
        if matrix_info is not None:
            column_type, count = matrix_info
            columns = [self.default_value_for_type(column_type) for _ in range(count)]
            return self.register_composite_constant(type_id, columns)

        array_info = self.array_type_info_from_type(type_id)
        if array_info is not None:
            element_type, size = array_info
            elements = [
                self.default_value_for_type(element_type) for _ in range(size or 0)
            ]
            return self.register_composite_constant(type_id, elements)

        members = self.current_struct_members.get(type_id.type.base_type)
        if members is not None:
            values = [
                self.default_value_for_type(member_type) for member_type, _ in members
            ]
            return self.register_composite_constant(type_id, values)

        return self.register_constant(0.0, self.register_primitive_type("float"))

    def process_array_literal(
        self,
        expr: ArrayLiteralNode,
        target_type: Optional[SpirvId] = None,
        constant: bool = False,
    ) -> Optional[SpirvId]:
        array_type = target_type
        element_type = None
        target_size = None

        if array_type is not None:
            array_type = self.ensure_registered_type(array_type)
            array_info = self.array_type_info_from_type(array_type)
            if array_info is not None:
                element_type, target_size = array_info

        values = []
        for element in expr.elements:
            value = self.process_array_literal_element(element, element_type, constant)
            if value is None:
                return None
            values.append(value)

        if array_type is None:
            if values:
                element_type = self.value_types.get(
                    values[0].id
                ) or self.find_registered_type_by_base(values[0].type.base_type)
            if element_type is None:
                element_type = self.register_primitive_type("float")
            target_size = len(values)
            array_type = self.register_array_type(element_type, target_size)

        if target_size is not None:
            values = values[:target_size]
            if element_type is not None:
                while len(values) < target_size:
                    values.append(self.default_value_for_type(element_type))

        if constant:
            if not all(self.is_constant_instruction(value) for value in values):
                return None
            return self.register_composite_constant(array_type, values)

        id_value = self.get_id()
        component_list = " ".join(f"%{value.id}" for value in values)
        self.emit(
            f"%{id_value} = OpCompositeConstruct %{array_type.id} {component_list}"
        )
        spirv_id = SpirvId(id_value, array_type.type)
        self.value_types[id_value] = array_type
        return spirv_id

    def process_array_literal_element(
        self,
        element,
        target_type: Optional[SpirvId],
        constant: bool,
    ) -> Optional[SpirvId]:
        if isinstance(element, ArrayLiteralNode):
            return self.process_array_literal(element, target_type, constant)

        if constant:
            return self.process_constant_expression(element, target_type)

        value = self.process_expression(element)
        if value is not None and target_type is not None:
            value = self.convert_value_to_type(value, target_type)
        return value

    def constant_literal_for_type(
        self, expr, target_type: Optional[SpirvId]
    ) -> Optional[SpirvId]:
        if target_type is None:
            return None

        target_type = self.ensure_registered_type(target_type)
        if self.vector_component_type_and_count(target_type.type.base_type) is not None:
            return None

        target_type_name = self.normalize_primitive_name(target_type.type.base_type)
        value = self.constant_scalar_literal_value(expr)
        if value is None:
            return None

        if target_type_name == "bool":
            if isinstance(value, bool):
                return self.register_constant(value, target_type)
            return None

        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None

        if target_type_name in {"float", "double"}:
            return self.register_constant(float(value), target_type)
        if target_type_name in self.INTEGER_TYPE_NAMES:
            if target_type_name in self.UNSIGNED_INTEGER_TYPES and value < 0:
                return None
            return self.register_constant(int(value), target_type)
        return None

    def constant_scalar_literal_value(self, expr):
        if isinstance(expr, UnaryOpNode):
            value = self.constant_scalar_literal_value(expr.operand)
            if value is None or isinstance(value, bool):
                return None
            operator = getattr(expr, "op", getattr(expr, "operator", None))
            if operator == "-":
                return -value
            if operator == "+":
                return value
            return None

        if isinstance(expr, LiteralNode):
            return expr.value

        if isinstance(expr, (bool, int, float)):
            return expr

        return None

    def process_constant_expression(
        self,
        expr,
        target_type: Optional[SpirvId] = None,
    ) -> Optional[SpirvId]:
        if isinstance(expr, ArrayLiteralNode):
            return self.process_array_literal(expr, target_type, constant=True)

        converted_literal = self.constant_literal_for_type(expr, target_type)
        if converted_literal is not None:
            return converted_literal

        if isinstance(expr, LiteralNode):
            return self.process_expression(expr)

        if isinstance(expr, FunctionCallNode):
            callee_expr = getattr(expr, "function", getattr(expr, "name", None))
            callee_name = getattr(callee_expr, "name", callee_expr)
            if target_type is not None:
                struct_constant = self.process_constant_struct_constructor(
                    callee_name, expr, target_type
                )
                if struct_constant is not None:
                    return struct_constant

            vector_type = target_type
            if vector_type is None and isinstance(callee_name, str):
                if self.vector_component_type_and_count(callee_name) is not None:
                    vector_type = self.map_crossgl_type(callee_name)

            if vector_type is not None:
                vector_info = self.vector_type_info_from_type(vector_type)
                if vector_info is not None:
                    component_type, component_count = vector_info
                    components = []
                    for arg in getattr(expr, "args", []):
                        value = self.process_constant_expression(arg, component_type)
                        if value is None:
                            return None
                        components.append(value)

                    if len(components) == 1 and component_count > 1:
                        components *= component_count

                    components = components[:component_count]
                    while len(components) < component_count:
                        components.append(self.default_value_for_type(component_type))

                    return self.register_vector_constant(vector_type, components)

        value = self.process_expression(expr)
        if value is not None and self.is_constant_instruction(value):
            return value
        return None

    def process_constant_struct_constructor(
        self, callee_name, expr, target_type: SpirvId
    ) -> Optional[SpirvId]:
        struct_name = target_type.type.base_type
        if callee_name != struct_name:
            return None

        members = self.current_struct_members.get(struct_name)
        if members is None:
            return None

        args = getattr(expr, "args", [])
        values = []
        for index, (member_type, _) in enumerate(members):
            if index < len(args):
                value = self.process_constant_expression(args[index], member_type)
                if value is None:
                    return None
                values.append(value)
            else:
                values.append(self.default_value_for_type(member_type))

        if len(args) < len(members):
            self.emit(
                f"; WARNING: Constructor {struct_name} expected {len(members)} "
                f"members but got {len(args)}; padding with defaults"
            )
        elif len(args) > len(members):
            self.emit(
                f"; WARNING: Constructor {struct_name} expected {len(members)} "
                f"members but got {len(args)}; truncating extra arguments"
            )

        return self.register_composite_constant(target_type, values)

    def ensure_assignable_pointer_for_name(self, name: str) -> Optional[SpirvId]:
        var_id = self.local_variables.get(name)
        if var_id is not None:
            if var_id.type.storage_class:
                return var_id

            value_type = self.value_types.get(var_id.id)
            if value_type is None:
                return None

            mutable_id = self.create_variable(value_type, "Function", name)
            self.store_to_variable(mutable_id, var_id)
            self.local_variables[name] = mutable_id
            return mutable_id

        var_id = (
            self.resolve_global_variable(name)
            or self.ensure_tessellation_patch_constant_input(name)
            or self.ensure_stage_builtin(name)
        )
        self.mark_interface_variable_if_needed(var_id)
        wrapped_pointer = self.uniform_block_wrapped_member_pointer(var_id)
        if wrapped_pointer is not None:
            return wrapped_pointer
        return var_id

    def assignable_pointer_from_expression(self, expr) -> Optional[SpirvId]:
        if isinstance(expr, IdentifierNode):
            return self.ensure_assignable_pointer_for_name(expr.name)
        if isinstance(expr, VariableNode):
            return self.ensure_assignable_pointer_for_name(expr.name)
        if isinstance(expr, str):
            return self.ensure_assignable_pointer_for_name(expr)
        if isinstance(expr, MemberAccessNode):
            base_pointer = self.assignable_pointer_from_expression(expr.object)
            if base_pointer is None:
                return None
            return self.create_member_access_pointer(base_pointer, expr.member)
        if isinstance(expr, ArrayAccessNode):
            index = self.process_expression(expr.index)
            if index is None:
                return None
            if not self.validate_tessellation_patch_builtin_index(
                expr.array, index, expr.index
            ):
                return None

            array_variable = self.assignable_pointer_from_expression(expr.array)
            if array_variable is None:
                return None
            if not self.validate_tessellation_patch_parameter_index(
                array_variable, index, expr.index
            ):
                return None

            structured_access = self.structured_buffer_element_pointer(
                array_variable, index
            )
            if structured_access is not None:
                return structured_access

            array_type = self.variable_value_types.get(array_variable.id)
            element_type = self.array_element_type_from_type(array_type)
            if element_type is None:
                element_type = self.determine_array_element_type(array_variable)
            if element_type is None:
                return None

            storage_class = array_variable.type.storage_class or "Function"
            ptr_type = self.register_pointer_type(element_type, storage_class)
            access = self.access_chain(array_variable, [index], ptr_type)
            self.variable_value_types[access.id] = element_type
            self.propagate_storage_buffer_access_metadata(array_variable, access)
            self.propagate_structured_buffer_descriptor_access_metadata(
                array_variable, access, index
            )
            self.propagate_resource_access_metadata(
                array_variable, access, element_type
            )
            self.propagate_readonly_builtin_pointer_name(array_variable, access)
            self.propagate_readonly_pointer_name(array_variable, access)
            return access
        return None

    def is_resource_array_type(self, array_type: Optional[SpirvId]) -> bool:
        element_type = self.array_element_type_from_type(array_type)
        while element_type is not None:
            if element_type.id in self.resource_type_metadata:
                return True
            element_type = self.array_element_type_from_type(element_type)
        return False

    def tessellation_patch_builtin_size(self, name: Optional[str]) -> Optional[int]:
        return {"gl_TessLevelOuter": 4, "gl_TessLevelInner": 2}.get(name or "")

    def validate_tessellation_integer_index(
        self,
        diagnostic_subject: str,
        size: int,
        index: SpirvId,
        index_expr=None,
    ) -> bool:
        index_type = self.value_types.get(
            index.id
        ) or self.find_registered_type_by_base(index.type.base_type)
        index_type_name = (
            self.normalize_primitive_name(index_type.type.base_type)
            if index_type is not None
            else index.type.base_type
        )
        if index_type_name not in {"int", "uint"}:
            self.emit(
                f"; WARNING: {diagnostic_subject} "
                f"index requires a scalar integer value, got {index.type.base_type}"
            )
            return False

        literal_index = self.literal_int_argument(index_expr)
        if literal_index is not None and not 0 <= literal_index < size:
            self.emit(
                f"; WARNING: {diagnostic_subject} "
                f"index {literal_index} out of range; valid range is 0..{size - 1}"
            )
            return False

        return True

    def validate_tessellation_patch_builtin_index(
        self, array_expr, index: SpirvId, index_expr=None
    ) -> bool:
        name = self.direct_expression_name(array_expr)
        size = self.tessellation_patch_builtin_size(name)
        if size is None:
            return True

        return self.validate_tessellation_integer_index(
            f"SPIR-V tessellation patch builtin {name} component",
            size,
            index,
            index_expr,
        )

    def validate_tessellation_patch_parameter_index(
        self, array_variable: SpirvId, index: SpirvId, index_expr=None
    ) -> bool:
        metadata = self.patch_parameter_metadata.get(array_variable.id)
        if metadata is None:
            return True

        return self.validate_tessellation_integer_index(
            f"SPIR-V tessellation patch parameter {metadata['name']} " "control-point",
            metadata["control_points"],
            index,
            index_expr,
        )

    def propagate_readonly_builtin_pointer_name(
        self, source_pointer: SpirvId, target_pointer: SpirvId
    ):
        builtin_name = self.readonly_builtin_pointer_names.get(source_pointer.id)
        if builtin_name is not None:
            self.readonly_builtin_pointer_names[target_pointer.id] = builtin_name

    def propagate_readonly_pointer_name(
        self, source_pointer: SpirvId, target_pointer: SpirvId
    ):
        pointer_name = self.readonly_pointer_names.get(source_pointer.id)
        if pointer_name is not None:
            self.readonly_pointer_names[target_pointer.id] = pointer_name

    def create_array_element_access(self, array_expr, index: SpirvId, index_expr=None):
        if not self.validate_tessellation_patch_builtin_index(
            array_expr, index, index_expr
        ):
            return None, None

        array_variable = self.variable_pointer_from_expression(array_expr)
        if array_variable is None or not array_variable.type.storage_class:
            addressable_array = self.assignable_pointer_from_expression(array_expr)
            if addressable_array is not None:
                array_variable = addressable_array

        if (
            array_variable is None
            and self.tessellation_patch_builtin_size(
                self.direct_expression_name(array_expr)
            )
            is not None
        ):
            return None, None

        if array_variable is not None:
            if not self.validate_tessellation_patch_parameter_index(
                array_variable, index, index_expr
            ):
                return None, None

            block_alias = self.single_struct_buffer_zero_index_alias(
                array_variable, index_expr
            )
            if block_alias is not None:
                return block_alias, self.variable_value_types.get(block_alias.id)

            structured_access = self.structured_buffer_element_pointer(
                array_variable, index
            )
            if structured_access is not None:
                return structured_access, self.variable_value_types.get(
                    structured_access.id
                )

            array_type = self.variable_value_types.get(array_variable.id)
            element_type = self.array_element_type_from_type(array_type)
            if element_type is None:
                element_type = self.determine_array_element_type(array_variable)
            if element_type is None:
                return None, None

            storage_class = array_variable.type.storage_class or "Function"
            ptr_type = self.register_pointer_type(element_type, storage_class)
            access = self.access_chain(array_variable, [index], ptr_type)
            self.variable_value_types[access.id] = element_type
            self.propagate_storage_buffer_access_metadata(array_variable, access)
            self.propagate_structured_buffer_descriptor_access_metadata(
                array_variable, access, index
            )
            self.propagate_resource_access_metadata(
                array_variable, access, element_type
            )
            self.propagate_readonly_builtin_pointer_name(array_variable, access)
            self.propagate_readonly_pointer_name(array_variable, access)
            return access, element_type

        array = self.process_expression(array_expr)
        if array is None:
            return None, None

        element_type = self.determine_array_element_type(array)
        if element_type is None:
            return None, None

        array_type = self.value_types.get(
            array.id
        ) or self.find_registered_type_by_base(array.type.base_type)
        if array_type is not None and not array.type.storage_class:
            array_variable = self.create_variable(array_type, "Function")
            self.store_to_variable(array_variable, array)
            storage_class = array_variable.type.storage_class or "Function"
            ptr_type = self.register_pointer_type(element_type, storage_class)
            access = self.access_chain(array_variable, [index], ptr_type)
            self.variable_value_types[access.id] = element_type
            self.propagate_storage_buffer_access_metadata(array_variable, access)
            self.propagate_structured_buffer_descriptor_access_metadata(
                array_variable, access, index
            )
            self.propagate_resource_access_metadata(
                array_variable, access, element_type
            )
            self.propagate_readonly_builtin_pointer_name(array_variable, access)
            self.propagate_readonly_pointer_name(array_variable, access)
            return access, element_type

        storage_class = array.type.storage_class or "Function"
        ptr_type = self.register_pointer_type(element_type, storage_class)
        access = self.access_chain(array, [index], ptr_type)
        self.variable_value_types[access.id] = element_type
        self.propagate_storage_buffer_access_metadata(array, access)
        self.propagate_structured_buffer_descriptor_access_metadata(
            array, access, index
        )
        self.propagate_resource_access_metadata(array, access, element_type)
        self.propagate_readonly_builtin_pointer_name(array, access)
        self.propagate_readonly_pointer_name(array, access)
        return access, element_type

    def process_assignment(self, node: AssignmentNode):
        """Process a CrossGL assignment statement."""
        target = getattr(
            node, "name", getattr(node, "target", getattr(node, "left", None))
        )
        target_is_precise = self.assignment_target_is_precise(target)
        operator = getattr(node, "operator", "=")

        if operator != "=":
            self.process_compound_assignment(node, target, operator, target_is_precise)
            return

        if self.process_mesh_output_assignment(target, node.value):
            return

        if isinstance(node.value, MatchNode):
            target_pointer = self.assignable_pointer_from_expression(target)
            target_type = (
                self.variable_value_types.get(target_pointer.id)
                if target_pointer is not None
                else None
            )
            if target_pointer is None or target_type is None:
                self.emit("; WARNING: Could not determine match assignment target type")
                return
            self.process_match_expression_assignment(
                node.value, target_pointer, target_type
            )
            return

        if isinstance(node.value, ArrayLiteralNode):
            target_pointer = self.assignable_pointer_from_expression(target)
            target_type = (
                self.variable_value_types.get(target_pointer.id)
                if target_pointer is not None
                else None
            )
            rhs_value = self.process_array_literal(node.value, target_type)
        else:
            rhs_value = self.process_expression_with_precision(
                node.value, target_is_precise
            )

        if rhs_value is None:
            return

        if isinstance(target, IdentifierNode):
            target = target.name

        if isinstance(target, str):
            if target in self.resource_alias_variables:
                self.emit(
                    f"; WARNING: cannot assign to SPIR-V descriptor resource alias "
                    f"{target}"
                )
                return

            var_id = self.ensure_assignable_pointer_for_name(target)
            if var_id is None:
                var_type = self.primitive_types["float"]
                if hasattr(rhs_value, "type"):
                    var_type = (
                        self.find_registered_type_by_base(rhs_value.type.base_type)
                        or var_type
                    )

                var_id = self.create_variable(var_type, "Function", target)
                self.local_variables[target] = var_id

            self.store_to_variable(var_id, rhs_value)

        elif isinstance(target, MemberAccessNode):
            base_pointer = self.assignable_pointer_from_expression(target.object)
            if base_pointer is None:
                return

            member_name = target.member
            access = self.create_member_access_pointer(base_pointer, member_name)
            if access is not None:
                self.store_to_variable(access, rhs_value)
                return
            if self.store_to_vector_swizzle(base_pointer, member_name, rhs_value):
                return
            if self.store_to_vector_component(base_pointer, member_name, rhs_value):
                return
            if self.emit_invalid_vector_swizzle_warning(base_pointer, member_name):
                return

            struct_type = self.struct_type_name_from_pointer(base_pointer)
            self.emit(
                f"; WARNING: Could not find member {member_name} in {struct_type}"
            )

        elif isinstance(target, ArrayAccessNode):
            access = self.assignable_pointer_from_expression(target)
            element_type = (
                self.variable_value_types.get(access.id) if access is not None else None
            )

            if access is None or element_type is None:
                self.emit(
                    "; WARNING: Could not determine array element type for "
                    f"{self.diagnostic_expression(target.array)}"
                )
                return

            self.store_to_variable(access, rhs_value)
        else:
            self.emit(
                f"; WARNING: Unsupported LHS type in assignment: {type(target).__name__}"
            )

    def process_compound_assignment(
        self, node: AssignmentNode, target, operator: str, target_is_precise: bool
    ):
        spv_operator = self.compound_assignment_operator(operator)
        if spv_operator is None:
            self.emit(f"; WARNING: Unsupported compound assignment operator {operator}")
            return
        if isinstance(node.value, ArrayLiteralNode):
            self.emit("; WARNING: Compound array literal assignment is unsupported")
            return

        if self.process_mesh_output_compound_assignment(
            target, node.value, spv_operator, target_is_precise
        ):
            return

        if isinstance(target, MemberAccessNode):
            base_pointer = self.assignable_pointer_from_expression(target.object)
            if (
                base_pointer is not None
                and self.process_vector_swizzle_compound_assignment(
                    base_pointer,
                    target.member,
                    node.value,
                    spv_operator,
                    target_is_precise,
                )
            ):
                return
            if (
                base_pointer is not None
                and self.process_vector_component_compound_assignment(
                    base_pointer,
                    target.member,
                    node.value,
                    spv_operator,
                    target_is_precise,
                )
            ):
                return
            if base_pointer is not None and self.emit_invalid_vector_swizzle_warning(
                base_pointer, target.member
            ):
                return

        target_pointer = self.assignable_pointer_from_expression(target)
        if target_pointer is None:
            self.emit(
                f"; WARNING: Unsupported LHS type in assignment: {type(target).__name__}"
            )
            return

        target_type = self.variable_value_types.get(target_pointer.id)
        if target_type is None:
            target_type = self.find_registered_type_by_base(
                target_pointer.type.base_type.replace("ptr_", "", 1)
            )
        if target_type is None:
            self.emit("; WARNING: Could not determine compound assignment type")
            return

        if target_is_precise:
            self.precise_expression_depth += 1

        try:
            current_value = self.load_from_variable(target_pointer, target_type)
            rhs_value = self.process_expression(node.value)
            if rhs_value is None:
                return
            result = self.binary_operation(
                spv_operator, target_type, current_value, rhs_value
            )
        finally:
            if target_is_precise:
                self.precise_expression_depth -= 1

        self.store_to_variable(target_pointer, result)

    def vector_component_update_info(self, base_pointer: SpirvId, member_name: str):
        vector_type = self.variable_value_types.get(base_pointer.id)
        if vector_type is None:
            vector_type = self.find_registered_type_by_base(
                base_pointer.type.base_type.replace("ptr_", "", 1)
            )
        if vector_type is None:
            return None

        member_info = self.vector_member_info(vector_type.type.base_type, member_name)
        if member_info is None:
            return None

        member_index, member_type = member_info
        return vector_type, member_index, member_type

    def store_to_vector_component(
        self, base_pointer: SpirvId, member_name: str, value: SpirvId
    ) -> bool:
        update_info = self.vector_component_update_info(base_pointer, member_name)
        if update_info is None:
            return False

        vector_type, member_index, member_type = update_info
        component_value = self.convert_vector_component_value(
            value, member_type, member_name
        )
        if component_value is None:
            return True

        vector_value = self.load_from_variable(base_pointer, vector_type)
        updated_value = self.composite_insert(
            vector_value, component_value, vector_type, member_index
        )
        self.store_to_variable(base_pointer, updated_value)
        return True

    def vector_swizzle_update_info(self, base_pointer: SpirvId, member_name: str):
        vector_type = self.variable_value_types.get(base_pointer.id)
        if vector_type is None:
            vector_type = self.find_registered_type_by_base(
                base_pointer.type.base_type.replace("ptr_", "", 1)
            )
        if vector_type is None:
            return None

        swizzle_info = self.vector_swizzle_info(vector_type.type.base_type, member_name)
        if swizzle_info is None:
            return None

        indices, member_type, swizzle_type = swizzle_info
        return vector_type, indices, member_type, swizzle_type

    def emit_invalid_vector_swizzle_warning(
        self, base_pointer: SpirvId, member_name: str
    ) -> bool:
        if len(member_name) <= 1:
            return False

        vector_type = self.variable_value_types.get(base_pointer.id)
        if vector_type is None:
            vector_type = self.find_registered_type_by_base(
                base_pointer.type.base_type.replace("ptr_", "", 1)
            )
        if vector_type is None:
            return False

        vector_type_name = vector_type.type.base_type
        if self.vector_component_type_and_count(vector_type_name) is None:
            return False
        if self.vector_swizzle_info(vector_type_name, member_name) is not None:
            return False

        self.emit(
            f"; WARNING: Invalid vector swizzle {member_name} for {vector_type_name}"
        )
        return True

    def store_to_vector_swizzle(
        self, base_pointer: SpirvId, member_name: str, value: SpirvId
    ) -> bool:
        update_info = self.vector_swizzle_update_info(base_pointer, member_name)
        if update_info is None:
            return False

        vector_type, member_indices, member_type, _ = update_info
        if len(set(member_indices)) != len(member_indices):
            self.emit(
                f"; WARNING: Cannot assign to vector swizzle {member_name} "
                "with duplicate components"
            )
            return True

        component_values = self.convert_vector_swizzle_assignment_components(
            value, member_type, len(member_indices), member_name
        )
        if component_values is None:
            return True

        updated_value = self.load_from_variable(base_pointer, vector_type)
        for member_index, component_value in zip(member_indices, component_values):
            updated_value = self.composite_insert(
                updated_value, component_value, vector_type, member_index
            )
        self.store_to_variable(base_pointer, updated_value)
        return True

    def convert_vector_swizzle_assignment_components(
        self,
        value: SpirvId,
        member_type: SpirvId,
        component_count: int,
        member_name: str,
    ) -> Optional[List[SpirvId]]:
        source_vector_info = self.vector_component_type_and_count(value.type.base_type)
        if source_vector_info is None:
            self.emit(
                f"; WARNING: Cannot assign scalar value to vector swizzle {member_name}"
            )
            return None

        source_component_type_name, source_component_count = source_vector_info
        if source_component_count != component_count:
            self.emit(
                f"; WARNING: Cannot assign {source_component_count}-component vector "
                f"to {component_count}-component swizzle {member_name}"
            )
            return None

        source_component_type = self.register_primitive_type(source_component_type_name)
        target_type_name = self.normalize_primitive_name(member_type.type.base_type)
        component_values = []
        for source_index in range(component_count):
            source_value = self.composite_extract(
                value, source_component_type, source_index
            )
            component_value = self.convert_scalar_to_type(source_value, member_type)
            converted_type_name = self.normalize_primitive_name(
                component_value.type.base_type
            )
            if converted_type_name != target_type_name:
                self.emit(
                    f"; WARNING: Cannot assign {converted_type_name} value to "
                    f"vector swizzle {member_name} of type {target_type_name}"
                )
                return None
            component_values.append(component_value)

        return component_values

    def process_vector_swizzle_compound_assignment(
        self,
        base_pointer: SpirvId,
        member_name: str,
        value_expr,
        spv_operator: str,
        target_is_precise: bool,
    ) -> bool:
        update_info = self.vector_swizzle_update_info(base_pointer, member_name)
        if update_info is None:
            return False

        vector_type, member_indices, member_type, swizzle_type = update_info
        if len(set(member_indices)) != len(member_indices):
            self.emit(
                f"; WARNING: Cannot assign to vector swizzle {member_name} "
                "with duplicate components"
            )
            return True

        if target_is_precise:
            self.precise_expression_depth += 1

        try:
            rhs_value = self.process_expression(value_expr)
            if rhs_value is None:
                return True
            rhs_components = self.convert_vector_swizzle_assignment_components(
                rhs_value, member_type, len(member_indices), member_name
            )
            if rhs_components is None:
                return True

            rhs_vector = self.composite_construct(swizzle_type, rhs_components)
            vector_value = self.load_from_variable(base_pointer, vector_type)
            current_value = self.vector_shuffle(
                vector_value, swizzle_type, member_indices
            )
            result = self.binary_operation(
                spv_operator, swizzle_type, current_value, rhs_vector
            )
        finally:
            if target_is_precise:
                self.precise_expression_depth -= 1

        updated_value = vector_value
        for result_index, member_index in enumerate(member_indices):
            component_value = self.composite_extract(result, member_type, result_index)
            updated_value = self.composite_insert(
                updated_value, component_value, vector_type, member_index
            )
        self.store_to_variable(base_pointer, updated_value)
        return True

    def convert_vector_component_value(
        self, value: SpirvId, member_type: SpirvId, member_name: str
    ) -> Optional[SpirvId]:
        if self.vector_component_type_and_count(value.type.base_type) is not None:
            self.emit(
                f"; WARNING: Cannot assign composite value to vector component {member_name}"
            )
            return None

        component_value = self.convert_scalar_to_type(value, member_type)
        source_type = self.normalize_primitive_name(component_value.type.base_type)
        target_type = self.normalize_primitive_name(member_type.type.base_type)
        if source_type != target_type:
            self.emit(
                f"; WARNING: Cannot assign {source_type} value to vector component "
                f"{member_name} of type {target_type}"
            )
            return None

        return component_value

    def process_vector_component_compound_assignment(
        self,
        base_pointer: SpirvId,
        member_name: str,
        value_expr,
        spv_operator: str,
        target_is_precise: bool,
    ) -> bool:
        update_info = self.vector_component_update_info(base_pointer, member_name)
        if update_info is None:
            return False

        vector_type, member_index, member_type = update_info
        if target_is_precise:
            self.precise_expression_depth += 1

        try:
            vector_value = self.load_from_variable(base_pointer, vector_type)
            current_value = self.composite_extract(
                vector_value, member_type, member_index
            )
            rhs_value = self.process_expression(value_expr)
            if rhs_value is None:
                return True
            rhs_value = self.convert_vector_component_value(
                rhs_value, member_type, member_name
            )
            if rhs_value is None:
                return True

            result = self.binary_operation(
                spv_operator, member_type, current_value, rhs_value
            )
        finally:
            if target_is_precise:
                self.precise_expression_depth -= 1

        updated_value = self.composite_insert(
            vector_value, result, vector_type, member_index
        )
        self.store_to_variable(base_pointer, updated_value)
        return True

    def compound_assignment_operator(self, operator: str) -> Optional[str]:
        return {
            "+=": "+",
            "-=": "-",
            "*=": "*",
            "/=": "/",
            "%=": "%",
            "&=": "&",
            "|=": "|",
            "^=": "^",
            "<<=": "<<",
            ">>=": ">>",
        }.get(operator)

    def assignment_target_is_precise(self, target) -> bool:
        if isinstance(target, IdentifierNode):
            return self.variable_name_is_precise(target.name)
        if isinstance(target, str):
            return self.variable_name_is_precise(target)
        if isinstance(target, MemberAccessNode):
            return self.assignment_target_is_precise(target.object)
        if isinstance(target, ArrayAccessNode):
            return self.assignment_target_is_precise(target.array)
        return False

    def variable_name_is_precise(self, name: str) -> bool:
        if name in self.local_variables:
            return name in self.precise_local_variables
        return name in self.precise_global_variables

    def process_return(self, node: ReturnNode):
        """Process a CrossGL return statement."""
        if hasattr(node, "value") and node.value:
            if isinstance(node.value, list) and node.value:
                return_value = self.process_expression_with_expected_type(
                    node.value[0], self.current_return_type_source
                )
                if return_value:
                    self.create_return_value(return_value)
                else:
                    self.create_return()
            else:
                if isinstance(node.value, ArrayLiteralNode):
                    return_value = self.process_array_literal(
                        node.value, self.current_return_type
                    )
                elif isinstance(node.value, MatchNode):
                    return_value = self.process_match_expression_return(node.value)
                else:
                    return_value = self.process_expression_with_expected_type(
                        node.value, self.current_return_type_source
                    )
                if return_value:
                    self.create_return_value(return_value)
                else:
                    self.create_return()
        else:
            self.create_return()

    def process_match_expression_return(self, node: MatchNode) -> Optional[SpirvId]:
        """Lower return-position matches through a temporary selected value."""
        return_type = self.current_return_type
        if return_type is None or return_type.type.base_type == "void":
            self.process_match(node)
            return None

        result_pointer = self.create_variable(return_type, "Function", "__match_return")
        default_value = self.default_value_for_type(return_type)
        if default_value is not None:
            self.store_to_variable(result_pointer, default_value)
        self.process_match_expression_assignment(node, result_pointer, return_type)
        return self.load_from_variable(result_pointer, return_type)

    def process_if(self, node: IfNode):
        """Process a CrossGL if statement."""
        condition = self.process_expression(node.if_condition)
        if condition is None:
            condition = self.register_constant(True, self.primitive_types["bool"])
        condition = self.ensure_bool_value(condition)

        merge_label = SpirvId(self.get_id(), SpirvType("label"))
        then_label = SpirvId(self.get_id(), SpirvType("label"))
        else_label = SpirvId(self.get_id(), SpirvType("label"))

        self.create_selection_merge(merge_label)
        self.create_conditional_branch(condition, then_label, else_label)

        self.emit(f"%{then_label.id} = OpLabel")
        self.current_label = then_label.id
        self.process_statements(node.if_body)
        if not self.current_block_has_terminator():
            self.create_branch(merge_label)

        self.emit(f"%{else_label.id} = OpLabel")
        self.current_label = else_label.id
        if node.else_body:
            self.process_statements(node.else_body)
        if not self.current_block_has_terminator():
            self.create_branch(merge_label)

        self.emit(f"%{merge_label.id} = OpLabel")
        self.current_label = merge_label.id

    def process_for(self, node: ForNode):
        """Process a CrossGL for loop."""
        if node.init:
            self.process_statement(node.init)

        header_label = SpirvId(self.get_id(), SpirvType("label"))
        body_label = SpirvId(self.get_id(), SpirvType("label"))
        continue_label = SpirvId(self.get_id(), SpirvType("label"))
        merge_label = SpirvId(self.get_id(), SpirvType("label"))

        self.create_branch(header_label)

        self.emit(f"%{header_label.id} = OpLabel")
        self.current_label = header_label.id

        condition = self.process_expression(node.condition)
        if condition is None:
            condition = self.register_constant(True, self.primitive_types["bool"])
        condition = self.ensure_bool_value(condition)

        self.create_loop_merge(merge_label, continue_label)
        self.create_conditional_branch(condition, body_label, merge_label)

        self.emit(f"%{body_label.id} = OpLabel")
        self.current_label = body_label.id
        self.loop_merge_labels.append(merge_label)
        self.loop_continue_labels.append(continue_label)
        try:
            if node.body:
                self.process_statements(node.body)
            if not self.current_block_has_terminator():
                self.create_branch(continue_label)
        finally:
            self.loop_continue_labels.pop()
            self.loop_merge_labels.pop()

        self.emit(f"%{continue_label.id} = OpLabel")
        self.current_label = continue_label.id
        if node.update:
            self.process_statement(node.update)
        if not self.current_block_has_terminator():
            self.create_branch(header_label)

        self.emit(f"%{merge_label.id} = OpLabel")
        self.current_label = merge_label.id

    def process_for_in(self, node: ForInNode):
        """Process numeric CrossGL for-in loops as structured counted loops."""
        pattern = getattr(node, "pattern", "item")
        iterable = getattr(node, "iterable", None)
        int_type = self.register_primitive_type("int")

        if isinstance(iterable, RangeNode):
            start = self.process_expression(iterable.start)
            end_expr = iterable.end
            comparator = "<=" if iterable.inclusive else "<"
        else:
            start = self.register_constant(0, int_type)
            end_expr = iterable
            comparator = "<"

        if start is None:
            start = self.register_constant(0, int_type)

        previous_variable = self.local_variables.get(pattern)
        loop_variable = self.create_variable(int_type, "Function", pattern)
        self.local_variables[pattern] = loop_variable
        self.store_to_variable(loop_variable, start)

        header_label = SpirvId(self.get_id(), SpirvType("label"))
        body_label = SpirvId(self.get_id(), SpirvType("label"))
        continue_label = SpirvId(self.get_id(), SpirvType("label"))
        merge_label = SpirvId(self.get_id(), SpirvType("label"))

        self.create_branch(header_label)

        self.emit(f"%{header_label.id} = OpLabel")
        self.current_label = header_label.id

        loop_value = self.get_variable_value(loop_variable)
        end_value = self.process_expression(end_expr)
        if end_value is None:
            end_value = self.register_constant(0, int_type)
        condition = self.binary_operation(comparator, int_type, loop_value, end_value)

        self.create_loop_merge(merge_label, continue_label)
        self.create_conditional_branch(condition, body_label, merge_label)

        self.emit(f"%{body_label.id} = OpLabel")
        self.current_label = body_label.id
        self.loop_merge_labels.append(merge_label)
        self.loop_continue_labels.append(continue_label)
        try:
            if node.body:
                self.process_statements(node.body)
            if not self.current_block_has_terminator():
                self.create_branch(continue_label)
        finally:
            self.loop_continue_labels.pop()
            self.loop_merge_labels.pop()

        self.emit(f"%{continue_label.id} = OpLabel")
        self.current_label = continue_label.id
        current_value = self.get_variable_value(loop_variable)
        one = self.register_constant(1, int_type)
        next_value = self.binary_operation("+", int_type, current_value, one)
        self.store_to_variable(loop_variable, next_value)
        if not self.current_block_has_terminator():
            self.create_branch(header_label)

        self.emit(f"%{merge_label.id} = OpLabel")
        self.current_label = merge_label.id

        if previous_variable is None:
            self.local_variables.pop(pattern, None)
        else:
            self.local_variables[pattern] = previous_variable

    def process_match(self, node: MatchNode):
        """Process CrossGL match statements as an ordered SPIR-V selection chain."""
        self.process_match_selection(
            node,
            lambda arm: self.process_match_statement_body(getattr(arm, "body", [])),
        )

    def process_match_expression_assignment(
        self, node: MatchNode, target_pointer: SpirvId, target_type: SpirvId
    ):
        """Lower a value-position match into stores to an existing local."""
        self.process_match_selection(
            node,
            lambda arm: self.process_match_expression_arm_body(
                getattr(arm, "body", []), target_pointer, target_type
            ),
        )

    def process_match_selection(self, node: MatchNode, process_arm_body):
        arms = getattr(node, "arms", []) or []
        expression = self.process_expression(getattr(node, "expression", None))
        if expression is None:
            expression = self.register_constant(0, self.register_primitive_type("int"))

        if not arms:
            return

        bool_type = self.register_primitive_type("bool")
        matched_variable = self.create_variable(
            bool_type, "Function", "__match_matched"
        )
        self.store_to_variable(
            matched_variable, self.register_constant(False, bool_type)
        )
        matched_true = self.register_constant(True, bool_type)

        for arm in arms:
            condition, bindings = self.lower_match_pattern_condition(
                getattr(arm, "pattern", None), expression
            )
            restore_bindings = self.apply_match_bindings(bindings)
            try:
                guard = getattr(arm, "guard", None)
                if guard is not None:
                    guard_condition = self.process_expression(guard)
                    condition = self.combine_match_conditions(
                        condition, guard_condition
                    )

                already_matched = self.load_from_variable(matched_variable, bool_type)
                not_matched = self.unary_operation("!", bool_type, already_matched)
                condition = self.combine_match_conditions(not_matched, condition)

                body_label = SpirvId(self.get_id(), SpirvType("label"))
                next_label = SpirvId(self.get_id(), SpirvType("label"))

                self.create_selection_merge(next_label)
                self.create_conditional_branch(condition, body_label, next_label)

                self.emit(f"%{body_label.id} = OpLabel")
                self.current_label = body_label.id
                process_arm_body(arm)
                if not self.current_block_has_terminator():
                    self.store_to_variable(matched_variable, matched_true)
                    self.create_branch(next_label)

                self.emit(f"%{next_label.id} = OpLabel")
                self.current_label = next_label.id
            finally:
                restore_bindings()

    def process_match_statement_body(self, body):
        self.process_statements(body)

    def process_match_expression_arm_body(
        self, body, target_pointer: SpirvId, target_type: SpirvId
    ):
        statements = getattr(body, "statements", None)
        if statements is not None:
            statements = list(statements)
            if statements and getattr(statements[-1], "is_tail_expression", False):
                self.process_statements(statements[:-1])
                self.store_match_expression_result(
                    getattr(statements[-1], "expression", None),
                    target_pointer,
                    target_type,
                )
                return
            self.process_statements(statements)
            return

        if hasattr(body, "expression"):
            self.store_match_expression_result(
                getattr(body, "expression", None), target_pointer, target_type
            )
            return

        self.process_statement(body)

    def store_match_expression_result(
        self, expression, target_pointer: SpirvId, target_type: SpirvId
    ):
        value = self.process_expression_with_expected_type(
            expression, target_type.type.base_type
        )
        if value is None:
            return
        self.store_to_variable(
            target_pointer, self.convert_value_to_type(value, target_type)
        )

    def lower_match_pattern_condition(self, pattern, expression: SpirvId):
        if isinstance(pattern, LiteralPatternNode):
            pattern_value = self.process_expression(pattern.literal)
            if pattern_value is None:
                pattern_value = self.register_constant(
                    0, self.register_primitive_type("int")
                )
            return self.compare_match_values(expression, pattern_value), []

        if isinstance(pattern, WildcardPatternNode):
            return None, []

        if isinstance(pattern, IdentifierPatternNode):
            return self.lower_identifier_match_pattern(pattern, expression)

        if isinstance(pattern, StructPatternNode):
            return self.lower_struct_match_pattern(pattern, expression)

        if isinstance(pattern, ConstructorPatternNode):
            return self.lower_constructor_match_pattern(pattern, expression)

        self.raise_match_pattern_gap(
            f"{type(pattern).__name__} patterns are not lowerable"
        )

    def lower_identifier_match_pattern(
        self, pattern: IdentifierPatternNode, expression: SpirvId
    ):
        name = pattern.name
        if name in {"_", ".."}:
            return None, []

        enum_constant = self.enum_variant_constant(name)
        if enum_constant is not None:
            return self.enum_variant_match_condition(expression, name), []

        if "::" in name:
            self.raise_match_pattern_gap(
                "enum path patterns without a registered plain enum "
                f"discriminant are not lowerable: {name}"
            )

        return None, [(name, expression)]

    def lower_constructor_match_pattern(
        self, pattern: ConstructorPatternNode, expression: SpirvId
    ):
        if pattern.type_name in self.enum_variant_values:
            condition = self.enum_variant_match_condition(expression, pattern.type_name)
            field_condition, bindings = self.lower_enum_payload_pattern_bindings(
                expression,
                pattern.type_name,
                list(getattr(pattern, "arguments", []) or []),
            )
            return self.combine_match_conditions(condition, field_condition), bindings

        self.raise_match_pattern_gap(
            "payload enum constructor patterns require tagged enum struct lowering"
        )

    def lower_struct_match_pattern(
        self, pattern: StructPatternNode, expression: SpirvId
    ):
        if "::" in pattern.type_name:
            condition = self.enum_variant_match_condition(expression, pattern.type_name)
            field_condition, bindings = self.lower_enum_struct_pattern_bindings(
                expression,
                pattern.type_name,
                getattr(pattern, "field_patterns", {}) or {},
            )
            return self.combine_match_conditions(condition, field_condition), bindings

        struct_type_name = expression.type.base_type
        if pattern.type_name != struct_type_name:
            self.raise_match_pattern_gap(
                "struct pattern " f"{pattern.type_name} cannot match {struct_type_name}"
            )

        condition = None
        bindings = []
        for field_name, field_pattern in (pattern.field_patterns or {}).items():
            member_info = self.struct_member_info(struct_type_name, field_name)
            if member_info is None:
                self.raise_match_pattern_gap(
                    "struct pattern "
                    f"field {field_name} does not exist on {struct_type_name}"
                )
            member_index, member_type = member_info
            field_value = self.composite_extract(expression, member_type, member_index)
            field_condition, field_bindings = self.lower_match_pattern_condition(
                field_pattern, field_value
            )
            condition = self.combine_match_conditions(condition, field_condition)
            bindings.extend(field_bindings)

        return condition, bindings

    def enum_variant_match_condition(self, expression: SpirvId, path: str) -> SpirvId:
        enum_constant = self.enum_variant_constant(path)
        if enum_constant is None:
            self.raise_match_pattern_gap(f"unknown enum variant {path}")

        parts = self.enum_path_parts(path)
        enum_name = parts[0] if parts else None
        if enum_name in self.enum_struct_type_names:
            variant_value = self.enum_struct_variant_value(expression, enum_name)
            return self.compare_match_values(variant_value, enum_constant)

        return self.compare_match_values(expression, enum_constant)

    def enum_struct_variant_value(self, expression: SpirvId, enum_name: str) -> SpirvId:
        struct_name = expression.type.base_type
        if struct_name != enum_name:
            specialization = self.generic_enum_specialization_for_struct_name(
                struct_name
            )
            if specialization is None or specialization["base_name"] != enum_name:
                self.raise_match_pattern_gap(
                    f"enum pattern {enum_name} cannot match {struct_name}"
                )
        member_info = self.struct_member_info(struct_name, "variant")
        if member_info is None:
            self.raise_match_pattern_gap(f"enum {struct_name} has no variant tag")
        member_index, member_type = member_info
        return self.composite_extract(expression, member_type, member_index)

    def lower_enum_payload_pattern_bindings(
        self, expression: SpirvId, path: str, argument_patterns
    ):
        variant_fields = self.enum_variant_fields_for_path(path, expression)
        if variant_fields is None:
            return []
        if len(argument_patterns) != len(variant_fields):
            raise ValueError(
                f"Enum pattern {path} expects {len(variant_fields)} arguments, "
                f"got {len(argument_patterns)}"
            )

        condition = None
        bindings = []
        for pattern, (field_name, _field_type) in zip(
            argument_patterns, variant_fields
        ):
            field_value = self.enum_struct_field_value(expression, field_name)
            field_condition, field_bindings = self.lower_match_pattern_condition(
                pattern, field_value
            )
            condition = self.combine_match_conditions(condition, field_condition)
            bindings.extend(field_bindings)
        return condition, bindings

    def lower_enum_struct_pattern_bindings(
        self, expression: SpirvId, path: str, field_patterns
    ):
        variant_fields = dict(self.enum_variant_fields_for_path(path, expression) or [])
        condition = None
        bindings = []
        for field_name, field_pattern in field_patterns.items():
            if field_name not in variant_fields:
                self.raise_match_pattern_gap(
                    f"enum pattern {path} has no payload field {field_name}"
                )
            field_value = self.enum_struct_field_value(expression, field_name)
            field_condition, field_bindings = self.lower_match_pattern_condition(
                field_pattern, field_value
            )
            condition = self.combine_match_conditions(condition, field_condition)
            bindings.extend(field_bindings)
        return condition, bindings

    def enum_struct_field_value(self, expression: SpirvId, field_name: str) -> SpirvId:
        member_info = self.struct_member_info(expression.type.base_type, field_name)
        if member_info is None:
            self.raise_match_pattern_gap(
                f"enum payload field {field_name} does not exist on "
                f"{expression.type.base_type}"
            )
        member_index, member_type = member_info
        return self.composite_extract(expression, member_type, member_index)

    def raise_match_pattern_gap(self, detail: str):
        raise ValueError(f"SPIR-V match pattern unsupported: {detail}")

    def compare_match_values(self, expression: SpirvId, pattern_value: SpirvId):
        return self.binary_operation(
            "==",
            self.ensure_registered_type(expression.type),
            expression,
            pattern_value,
        )

    def combine_match_conditions(
        self, left: Optional[SpirvId], right: Optional[SpirvId]
    ) -> Optional[SpirvId]:
        if left is None:
            return right
        if right is None:
            return left
        return self.binary_operation(
            "&&", self.register_primitive_type("bool"), left, right
        )

    def apply_match_bindings(self, bindings):
        missing = object()
        previous_values = []
        for name, value in bindings:
            if name in {"_", ".."}:
                continue
            previous_values.append((name, self.local_variables.get(name, missing)))
            value_type = self.ensure_registered_type(value.type)
            variable = self.create_variable(value_type, "Function", name)
            self.local_variables[name] = variable
            self.store_to_variable(variable, value)

        def restore():
            for name, previous in reversed(previous_values):
                if previous is missing:
                    self.local_variables.pop(name, None)
                else:
                    self.local_variables[name] = previous

        return restore

    def is_supported_match_arm(self, arm):
        if getattr(arm, "guard", None) is not None:
            return False
        pattern = getattr(arm, "pattern", None)
        return isinstance(pattern, (LiteralPatternNode, WildcardPatternNode))

    def validate_match_arms(self, arms):
        wildcard_index = None
        for index, arm in enumerate(arms):
            if not self.is_supported_match_arm(arm):
                return False
            if isinstance(getattr(arm, "pattern", None), WildcardPatternNode):
                if wildcard_index is not None:
                    return False
                wildcard_index = index
        return wildcard_index is None or wildcard_index == len(arms) - 1

    def switch_case_statements(self, case):
        """Return the body statements for a switch case/default node."""
        if case is None:
            return []
        if hasattr(case, "statements"):
            return getattr(case, "statements") or []
        if hasattr(case, "body"):
            return getattr(case, "body") or []
        if isinstance(case, list):
            return case
        return [case]

    def process_switch(self, node: SwitchNode):
        """Process CrossGL switch statements as a structured OpSwitch."""
        cases = getattr(node, "cases", []) or []
        expression = self.process_expression(getattr(node, "expression", None))
        if expression is None:
            expression = self.register_constant(0, self.register_primitive_type("int"))

        explicit_cases = [
            case
            for case in cases
            if hasattr(case, "value") and getattr(case, "value", None) is not None
        ]
        default_case = next(
            (
                case
                for case in cases
                if hasattr(case, "value") and getattr(case, "value", None) is None
            ),
            None,
        )
        if default_case is None:
            default_case = getattr(node, "default_case", None)

        if not explicit_cases and default_case is None:
            return

        merge_label = SpirvId(self.get_id(), SpirvType("label"))
        case_labels = []
        for case in explicit_cases:
            literal_value = self.switch_case_literal_value(getattr(case, "value", None))
            if literal_value is None:
                raise ValueError(
                    "SPIR-V switch case values must be scalar integer literals"
                )
            case_labels.append(
                (literal_value, SpirvId(self.get_id(), SpirvType("label")), case)
            )

        default_label = (
            SpirvId(self.get_id(), SpirvType("label"))
            if default_case is not None
            else merge_label
        )

        switch_operands = " ".join(
            f"{literal_value} %{label.id}"
            for literal_value, label, _case in case_labels
        )
        self.create_selection_merge(merge_label)
        if switch_operands:
            self.emit(
                f"OpSwitch %{expression.id} %{default_label.id} {switch_operands}"
            )
        else:
            self.emit(f"OpSwitch %{expression.id} %{default_label.id}")

        if not case_labels:
            self.emit(f"%{default_label.id} = OpLabel")
            self.current_label = default_label.id
            self.loop_merge_labels.append(merge_label)
            try:
                self.process_statements(self.switch_case_statements(default_case))
                if not self.current_block_has_terminator():
                    self.create_branch(merge_label)
            finally:
                self.loop_merge_labels.pop()

            self.emit(f"%{merge_label.id} = OpLabel")
            self.current_label = merge_label.id
            return

        for _literal_value, label, case in case_labels:
            self.emit(f"%{label.id} = OpLabel")
            self.current_label = label.id
            self.loop_merge_labels.append(merge_label)
            try:
                self.process_statements(self.switch_case_statements(case))
                if not self.current_block_has_terminator():
                    self.create_branch(merge_label)
            finally:
                self.loop_merge_labels.pop()

        if default_case is not None:
            self.emit(f"%{default_label.id} = OpLabel")
            self.current_label = default_label.id
            self.loop_merge_labels.append(merge_label)
            try:
                self.process_statements(self.switch_case_statements(default_case))
            finally:
                self.loop_merge_labels.pop()

        if not self.current_block_has_terminator():
            self.create_branch(merge_label)

        self.emit(f"%{merge_label.id} = OpLabel")
        self.current_label = merge_label.id

    def switch_case_literal_value(self, value):
        if value is None:
            return None
        if isinstance(value, bool):
            return int(value)
        if isinstance(value, int):
            return value
        literal_value = getattr(value, "value", None)
        if isinstance(literal_value, bool):
            return int(literal_value)
        if isinstance(literal_value, int):
            return literal_value
        name = getattr(value, "name", None)
        if name in self.enum_variant_values:
            return self.enum_variant_values[name]
        return None

    def process_while(self, node: WhileNode):
        """Process a CrossGL while loop."""
        header_label = SpirvId(self.get_id(), SpirvType("label"))
        body_label = SpirvId(self.get_id(), SpirvType("label"))
        continue_label = SpirvId(self.get_id(), SpirvType("label"))
        merge_label = SpirvId(self.get_id(), SpirvType("label"))

        self.create_branch(header_label)

        self.emit(f"%{header_label.id} = OpLabel")
        self.current_label = header_label.id

        condition = self.process_expression(node.condition)
        if condition is None:
            condition = self.register_constant(True, self.primitive_types["bool"])
        condition = self.ensure_bool_value(condition)

        self.create_loop_merge(merge_label, continue_label)
        self.create_conditional_branch(condition, body_label, merge_label)

        self.emit(f"%{body_label.id} = OpLabel")
        self.current_label = body_label.id
        self.loop_merge_labels.append(merge_label)
        self.loop_continue_labels.append(continue_label)
        try:
            if node.body:
                self.process_statements(node.body)
            if not self.current_block_has_terminator():
                self.create_branch(continue_label)
        finally:
            self.loop_continue_labels.pop()
            self.loop_merge_labels.pop()

        self.emit(f"%{continue_label.id} = OpLabel")
        self.current_label = continue_label.id
        self.create_branch(header_label)

        self.emit(f"%{merge_label.id} = OpLabel")
        self.current_label = merge_label.id

    def process_do_while(self, node: DoWhileNode):
        """Process a CrossGL do-while loop."""
        header_label = SpirvId(self.get_id(), SpirvType("label"))
        body_label = SpirvId(self.get_id(), SpirvType("label"))
        continue_label = SpirvId(self.get_id(), SpirvType("label"))
        merge_label = SpirvId(self.get_id(), SpirvType("label"))

        self.create_branch(header_label)

        self.emit(f"%{header_label.id} = OpLabel")
        self.current_label = header_label.id
        self.create_loop_merge(merge_label, continue_label)
        self.create_branch(body_label)

        self.emit(f"%{body_label.id} = OpLabel")
        self.current_label = body_label.id
        self.loop_merge_labels.append(merge_label)
        self.loop_continue_labels.append(continue_label)
        try:
            if node.body:
                self.process_statements(node.body)
            if not self.current_block_has_terminator():
                self.create_branch(continue_label)
        finally:
            self.loop_continue_labels.pop()
            self.loop_merge_labels.pop()

        self.emit(f"%{continue_label.id} = OpLabel")
        self.current_label = continue_label.id

        condition = self.process_expression(node.condition)
        if condition is None:
            condition = self.register_constant(True, self.primitive_types["bool"])
        condition = self.ensure_bool_value(condition)

        self.create_conditional_branch(condition, header_label, merge_label)

        self.emit(f"%{merge_label.id} = OpLabel")
        self.current_label = merge_label.id

    def process_loop(self, node: LoopNode):
        """Process a CrossGL unconditional loop."""
        header_label = SpirvId(self.get_id(), SpirvType("label"))
        body_label = SpirvId(self.get_id(), SpirvType("label"))
        continue_label = SpirvId(self.get_id(), SpirvType("label"))
        merge_label = SpirvId(self.get_id(), SpirvType("label"))

        self.create_branch(header_label)

        self.emit(f"%{header_label.id} = OpLabel")
        self.current_label = header_label.id
        self.create_loop_merge(merge_label, continue_label)
        self.create_branch(body_label)

        self.emit(f"%{body_label.id} = OpLabel")
        self.current_label = body_label.id
        self.loop_merge_labels.append(merge_label)
        self.loop_continue_labels.append(continue_label)
        try:
            if node.body:
                self.process_statements(node.body)
            if not self.current_block_has_terminator():
                self.create_branch(continue_label)
        finally:
            self.loop_continue_labels.pop()
            self.loop_merge_labels.pop()

        self.emit(f"%{continue_label.id} = OpLabel")
        self.current_label = continue_label.id
        self.create_branch(header_label)

        self.emit(f"%{merge_label.id} = OpLabel")
        self.current_label = merge_label.id

    def process_break(self, node: BreakNode):
        """Process a CrossGL break statement."""
        if self.loop_merge_labels:
            self.create_branch(self.loop_merge_labels[-1])
            return
        self.emit("; WARNING: break used outside a loop or switch")

    def process_continue(self, node: ContinueNode):
        """Process a CrossGL continue statement."""
        if not self.loop_continue_labels:
            self.emit("; WARNING: continue used outside a loop")
            return
        self.create_branch(self.loop_continue_labels[-1])

    def process_increment_expression(self, node: UnaryOpNode) -> SpirvId:
        """Process prefix/postfix ++ and -- as load/update/store operations."""
        variable_id = self.variable_pointer_from_expression(node.operand)
        if variable_id is None:
            self.emit("; WARNING: increment target is not assignable")
            int_type = self.register_primitive_type("int")
            return self.register_constant(0, int_type)

        value_type = self.variable_value_types.get(variable_id.id)
        if value_type is None:
            value_type = self.find_registered_type_by_base(
                variable_id.type.base_type.replace("ptr_", "", 1)
            )
        if value_type is None:
            value_type = self.register_primitive_type("int")

        old_value = self.load_from_variable(variable_id, value_type)
        step_value = (
            self.register_constant(1.0, value_type)
            if value_type.type.base_type == "float"
            else self.register_constant(1, value_type)
        )
        operator = "+" if node.op == "++" else "-"
        new_value = self.binary_operation(operator, value_type, old_value, step_value)
        self.store_to_variable(variable_id, new_value)

        if getattr(node, "is_postfix", getattr(node, "postfix", False)):
            return old_value
        return new_value

    def process_non_uniform_function_call(self, function_name, args) -> SpirvId:
        if len(args) != 1:
            self.emit(f"; WARNING: {function_name} requires exactly one operand")
            return self.register_constant(0, self.register_primitive_type("int"))

        value = self.process_expression(args[0])
        if value is None:
            self.emit(f"; WARNING: {function_name} operand could not be evaluated")
            return self.register_constant(0, self.register_primitive_type("int"))

        result_type = self.value_types.get(value.id) or self.ensure_registered_type(
            value.type
        )
        id_value = self.get_id()
        self.emit(f"%{id_value} = OpCopyObject %{result_type.id} %{value.id}")
        copied = SpirvId(id_value, result_type.type)
        self.value_types[id_value] = result_type
        self.mark_non_uniform_result(copied)
        return copied

    def process_expression(self, expr) -> Optional[SpirvId]:
        """Process a CrossGL expression."""
        if expr is None:
            return None

        if isinstance(expr, bool):
            bool_type = self.register_primitive_type("bool")
            return self.register_constant(expr, bool_type)
        elif isinstance(expr, int):
            expected_type = self.expected_primitive_type_name()
            primitive_type = (
                expected_type
                if expected_type in self.UNSIGNED_INTEGER_TYPES and expr >= 0
                else "int"
            )
            primitive_type = self.integer_literal_type_for_value(primitive_type, expr)
            int_type = self.register_primitive_type(primitive_type)
            return self.register_constant(expr, int_type)
        elif isinstance(expr, float):
            float_type = self.register_primitive_type("float")
            return self.register_constant(expr, float_type)

        elif isinstance(expr, str):
            if expr.rsplit("::", 1)[-1] == "None":
                none_default = self.option_none_default_value()
                if none_default is not None:
                    return none_default

            if self.enum_variant_is_payload_path(expr):
                enum_value = self.process_enum_variant_constructor(expr, [])
                if enum_value is not None:
                    return enum_value
            else:
                enum_constant = self.enum_variant_constant(expr)
                if enum_constant is not None:
                    return enum_constant

            if expr in self.local_variables:
                var_id = self.local_variables[expr]
                return self.get_variable_value(var_id)
            elif expr in self.named_constants:
                return self.named_constants[expr]
            else:
                var_id = self.resolve_global_variable(expr)
                if var_id is not None:
                    self.mark_interface_variable_if_needed(var_id)
                    self.mark_builtin_interface_variable(var_id)
                    return self.get_variable_value(var_id)

                if expr in self.cbuffer_members:
                    member_pointer = self.cbuffer_member_pointer(expr)
                    if member_pointer is not None:
                        return self.get_variable_value(member_pointer)

                patch_input = self.ensure_tessellation_patch_constant_input(expr)
                if patch_input is not None:
                    return self.get_variable_value(patch_input)

                builtin_component = self.process_dotted_builtin(expr)
                if builtin_component is not None:
                    return builtin_component

                builtin = self.ensure_builtin_variable(expr)
                if builtin is not None:
                    return self.get_variable_value(builtin)

                builtin_limit = self.glsl_builtin_limit_constant(expr)
                if builtin_limit is not None:
                    return builtin_limit

                # Create a default float constant for missing variables in examples
                # This is to make the SPIR-V code valid even if we can't find the variable
                if expr.replace(".", "", 1).isdigit():  # Check if it's a numeric string
                    float_type = self.register_primitive_type("float")
                    try:
                        value = float(expr)
                        return self.register_constant(value, float_type)
                    except ValueError:
                        pass

                self.emit(f"; WARNING: Unknown variable {expr}")
                float_type = self.register_primitive_type("float")
                return self.register_constant(0.0, float_type)

        elif isinstance(expr, LiteralNode):
            literal_type = self.convert_type_node_to_string(expr.literal_type)
            primitive_type_name = self.normalize_primitive_name(literal_type)
            if primitive_type_name in {"float", "double"}:
                literal_type_id = self.register_primitive_type(primitive_type_name)
                return self.register_constant(float(expr.value), literal_type_id)
            if primitive_type_name in self.INTEGER_TYPE_NAMES:
                literal_value = int(expr.value)
                if (
                    primitive_type_name in self.SIGNED_INTEGER_TYPES
                    and self.expected_primitive_type_name()
                    in self.UNSIGNED_INTEGER_TYPES
                    and literal_value >= 0
                ):
                    primitive_type_name = self.expected_primitive_type_name()
                primitive_type_name = self.integer_literal_type_for_value(
                    primitive_type_name, literal_value
                )
                literal_type_id = self.register_primitive_type(primitive_type_name)
                return self.register_constant(literal_value, literal_type_id)
            if primitive_type_name == "bool":
                literal_type_id = self.register_primitive_type("bool")
                if isinstance(expr.value, str):
                    value = expr.value.lower() == "true"
                else:
                    value = bool(expr.value)
                return self.register_constant(value, literal_type_id)
            return self.process_expression(expr.value)

        elif isinstance(expr, IdentifierNode):
            return self.process_expression(expr.name)

        elif isinstance(expr, VariableNode):
            if expr.name in self.local_variables:
                var_id = self.local_variables[expr.name]
                return self.get_variable_value(var_id)
            elif expr.name in self.named_constants:
                return self.named_constants[expr.name]
            else:
                var_id = self.resolve_global_variable(expr.name)
                if var_id is not None:
                    self.mark_interface_variable_if_needed(var_id)
                    self.mark_builtin_interface_variable(var_id)
                    return self.get_variable_value(var_id)

                if expr.name in self.cbuffer_members:
                    member_pointer = self.cbuffer_member_pointer(expr.name)
                    if member_pointer is not None:
                        return self.get_variable_value(member_pointer)

                patch_input = self.ensure_tessellation_patch_constant_input(expr.name)
                if patch_input is not None:
                    return self.get_variable_value(patch_input)

                builtin = self.ensure_builtin_variable(expr.name)
                if builtin is not None:
                    return self.get_variable_value(builtin)

                builtin_limit = self.glsl_builtin_limit_constant(expr.name)
                if builtin_limit is not None:
                    return builtin_limit

                self.emit(f"; WARNING: Unknown variable {expr.name}")
                float_type = self.register_primitive_type("float")
                return self.register_constant(0.0, float_type)

        elif isinstance(expr, ArrayLiteralNode):
            return self.process_array_literal(expr)

        elif isinstance(expr, ArrayAccessNode):
            index = self.process_expression(expr.index)

            if index is None:
                self.emit(f"; WARNING: Failed to evaluate array access")
                float_type = self.register_primitive_type("float")
                return self.register_constant(0.0, float_type)

            access, element_type = self.create_array_element_access(
                expr.array, index, expr.index
            )

            if access is None or element_type is None:
                self.emit(
                    "; WARNING: Could not determine array element type for "
                    f"{self.diagnostic_expression(expr.array)}"
                )
                element_type = self.primitive_types["float"]
                return self.register_constant(0.0, element_type)

            return self.load_from_variable(access, element_type)

        elif isinstance(expr, BinaryOpNode):
            if expr.op in {"&", "|", "^"}:
                expected_type = self.bitwise_expression_operand_type(
                    self.infer_expression_result_type(expr.left),
                    self.infer_expression_result_type(expr.right),
                )
                if expected_type is not None:
                    left = self.process_expression_with_expected_type(
                        expr.left, expected_type
                    )
                    right = self.process_expression_with_expected_type(
                        expr.right, expected_type
                    )
                else:
                    left = self.process_expression(expr.left)
                    right = self.process_expression(expr.right)
            else:
                left = self.process_expression(expr.left)
                right = self.process_expression(expr.right)

            if left is None or right is None:
                float_type = self.register_primitive_type("float")
                return self.register_constant(0.0, float_type)

            left_type = self.registered_value_type(left) or self.ensure_registered_type(
                left.type
            )
            right_type = self.registered_value_type(
                right
            ) or self.ensure_registered_type(right.type)
            result_type = self.binary_expression_result_type(
                expr.op, left_type, right_type
            )
            if result_type is None:
                result_type = left_type

            return self.binary_operation(expr.op, result_type, left, right)

        elif isinstance(expr, UnaryOpNode):
            if expr.op in {"++", "--"}:
                return self.process_increment_expression(expr)

            operand = self.process_expression(expr.operand)
            if operand is None:
                float_type = self.register_primitive_type("float")
                return self.register_constant(0.0, float_type)

            return self.unary_operation(expr.op, operand.type, operand)

        elif isinstance(expr, TernaryOpNode):
            condition = self.process_expression(expr.condition)
            true_value = self.process_expression(expr.true_expr)
            false_value = self.process_expression(expr.false_expr)

            if condition is None:
                condition = self.register_constant(
                    False, self.register_primitive_type("bool")
                )
            if true_value is None or false_value is None:
                float_type = self.register_primitive_type("float")
                fallback = self.register_constant(0.0, float_type)
                true_value = true_value or fallback
                false_value = false_value or fallback

            result_type, true_value, false_value = self.normalize_ternary_values(
                true_value, false_value
            )
            return self.select_operation(
                result_type, condition, true_value, false_value
            )

        elif isinstance(expr, WaveOpNode):
            return self.call_wave_operation(expr.operation, expr.arguments)

        elif isinstance(expr, RayTracingOpNode):
            return self.process_ray_tracing_operation(expr)

        elif isinstance(expr, RayQueryOpNode):
            return self.process_ray_query_operation(expr)

        elif isinstance(expr, MeshOpNode):
            return self.process_mesh_operation(expr)

        elif isinstance(expr, ConstructorNode):
            constructed = self.process_struct_constructor_node(expr)
            if constructed is not None:
                return constructed
            self.emit(
                f"; WARNING: Unsupported constructor {self.convert_type_node_to_string(expr.constructor_type)}"
            )
            return None

        elif isinstance(expr, FunctionCallNode):
            ray_query_call = self.ray_query_call_from_function_call(expr)
            if ray_query_call is not None:
                return self.process_ray_query_operation(ray_query_call)

            handled_byte_address_call, byte_address_result = (
                self.process_byte_address_buffer_method_call(expr)
            )
            if handled_byte_address_call:
                return byte_address_result

            handled_structured_buffer_call, structured_buffer_result = (
                self.process_structured_buffer_method_call(expr)
            )
            if handled_structured_buffer_call:
                return structured_buffer_result

            callee_expr = getattr(expr, "function", getattr(expr, "name", None))
            callee_name = None
            if hasattr(callee_expr, "name"):
                callee_name = callee_expr.name
            elif isinstance(callee_expr, str):
                callee_name = callee_expr

            if (
                isinstance(callee_name, str)
                and callee_name.rsplit("::", 1)[-1] == "Some"
            ):
                payload_type = self.option_or_expected_payload_type_name()
                if payload_type is not None and not expr.args:
                    return self.default_value_for_type(
                        self.map_crossgl_type(payload_type)
                    )
                if payload_type is not None:
                    payload_value = self.process_expression_with_expected_type(
                        expr.args[0],
                        payload_type,
                    )
                    if payload_value is None:
                        return None
                    return self.convert_value_to_type(
                        payload_value,
                        self.map_crossgl_type(payload_type),
                    )

            if isinstance(callee_name, str) and callee_name.lower() == "nonuniformext":
                return self.process_non_uniform_function_call(callee_name, expr.args)

            if callee_name in {"SetVertex", "SetPrimitive"}:
                return self.process_mesh_output_function_call(callee_name, expr.args)

            if callee_name in {"EmitVertex", "EndPrimitive"}:
                return self.process_geometry_stream_function_call(
                    callee_name, expr.args
                )

            if (
                isinstance(callee_name, str)
                and self.enum_path_parts(callee_name) is not None
            ):
                enum_value = self.process_enum_variant_constructor(
                    callee_name,
                    list(getattr(expr, "args", []) or []),
                )
                if enum_value is not None:
                    return enum_value

            if callee_name == "lambda":
                return self.unsupported_lambda_default_value("lambda expression")

            if callee_name == "buffer_dimensions":
                return self.process_buffer_dimensions_function_call(expr)

            if any(self.contains_lambda_expression(arg) for arg in expr.args):
                result_type = None
                function_signature = self.resolve_function_signature(callee_name)
                if function_signature is not None:
                    result_type = function_signature[0]
                return self.unsupported_lambda_default_value(
                    f"call to {callee_name or 'unknown callee'}",
                    result_type,
                )

            inline_storage_buffer_function = (
                self.resolve_inline_storage_buffer_function(
                    callee_name, expr.args, call_node=expr
                )
            )
            if inline_storage_buffer_function is not None:
                return self.inline_storage_buffer_function_call(
                    inline_storage_buffer_function, expr.args, call_node=expr
                )

            generic_function_definition = self.generic_function_definitions.get(
                callee_name
            )
            if (
                generic_function_definition is not None
                and not self.has_matching_concrete_function_candidate(
                    callee_name, expr.args
                )
            ):
                specialized_callee_name = generic_function_call_name(
                    self, callee_name, expr.args
                )
                if specialized_callee_name is None:
                    raise self.unsupported_generic_function_error(
                        generic_function_definition, call_node=expr
                    )
                call_args = generic_function_value_arguments(
                    self,
                    callee_name,
                    expr.args,
                )
                callee_name = specialized_callee_name
            else:
                call_args = list(expr.args)

            args = []
            has_errors = False
            skipped_arg_indices = self.skipped_function_parameter_indices(callee_name)
            mesh_output_arg_indices = (
                self.resolve_function_mesh_output_parameter_indices(callee_name)
            )
            for arg_index, arg in enumerate(call_args):
                if arg_index in skipped_arg_indices:
                    continue
                if arg_index in mesh_output_arg_indices:
                    continue
                arg_value = self.process_call_argument(callee_name, arg, arg_index)
                if arg_value is None:
                    self.emit(
                        f"; WARNING: Failed to evaluate argument for {callee_name or callee_expr}"
                    )
                    has_errors = True
                    float_type = self.register_primitive_type("float")
                    arg_value = self.register_constant(0.0, float_type)
                args.append(arg_value)

            for stage_object_name in self.required_function_stage_object_argument_names(
                callee_name
            ):
                stage_object = self.local_variables.get(stage_object_name)
                if stage_object is None:
                    self.emit(
                        "; WARNING: Failed to evaluate stage object argument "
                        f"{stage_object_name} for {callee_name or callee_expr}"
                    )
                    has_errors = True
                    continue
                args.append(stage_object)

            if has_errors and callee_name == "vec2":
                float_type = self.register_primitive_type("float")
                vector_type = self.register_vector_type(float_type, 2)
                id_value = self.get_id()

                while len(args) < 2:
                    args.append(self.register_constant(0.0, float_type))

                arg_list = " ".join([f"%{arg.id}" for arg in args[:2]])
                self.emit(
                    f"%{id_value} = OpCompositeConstruct %{vector_type.id} {arg_list}"
                )
                return SpirvId(id_value, vector_type.type)

            if callee_name is None:
                # Non-identifier callee (e.g., function table call) not supported in SPIR-V path
                self.emit("; WARNING: Unsupported callee expression in SPIR-V backend")
                float_type = self.register_primitive_type("float")
                return self.register_constant(0.0, float_type)

            if not self.validate_function_image_access_arguments(
                callee_name, expr.args
            ):
                result_type = None
                function_signature = self.resolve_function_signature(callee_name)
                if function_signature is not None:
                    result_type = function_signature[0]
                if result_type is None or result_type.type.base_type == "void":
                    return None
                return self.default_value_for_type(result_type)

            if self.is_resource_type_name(callee_name):
                constructed_resource = self.call_resource_constructor(callee_name, args)
                if constructed_resource is not None:
                    return constructed_resource

            if (
                callee_name in self.resource_function_names()
                and not self.has_function_reference(callee_name)
            ):
                return self.call_resource_function(callee_name, args)

            return self.call_function(callee_name, args)

        elif isinstance(expr, MemberAccessNode):
            member_name = expr.member
            base_pointer = self.variable_pointer_from_expression(expr.object)
            if base_pointer is not None:
                access = self.create_member_access_pointer(base_pointer, member_name)
                if access is not None:
                    member_type = self.variable_value_types.get(access.id)
                    return self.load_from_variable(access, member_type)

            base = self.process_expression(expr.object)
            if base is None:
                return None

            struct_type = base.type.base_type
            member_info = self.struct_member_info(struct_type, member_name)
            if member_info is not None:
                member_index, member_type = member_info
                return self.composite_extract(base, member_type, member_index)

            member_info = self.vector_member_info(struct_type, member_name)
            if member_info is not None:
                member_index, member_type = member_info
                return self.composite_extract(base, member_type, member_index)

            swizzle_info = self.vector_swizzle_info(struct_type, member_name)
            if swizzle_info is not None:
                indices, _, result_type = swizzle_info
                return self.vector_shuffle(base, result_type, indices)

            if (
                self.vector_component_type_and_count(struct_type) is not None
                and len(member_name) > 1
            ):
                self.emit(
                    f"; WARNING: Invalid vector swizzle {member_name} for {struct_type}"
                )
                return None

            self.emit(
                f"; WARNING: Could not find member {member_name} in {struct_type}"
            )
            return None

        else:
            self.emit(f"; WARNING: Unknown expression type {type(expr).__name__}")
            return None

    def register_input(
        self,
        name: str,
        type_id: SpirvId,
        location: int,
        binding: int,
        source_node=None,
    ) -> SpirvId:
        """Register an input variable with location decoration."""
        self.validate_user_defined_interface_type(type_id, "Input", name, source_node)
        ptr_type = self.register_pointer_type(type_id, "Input")

        id_value = self.get_id()
        self.emit(f"%{id_value} = OpVariable %{ptr_type.id} Input")

        self.decorations.append(f"OpDecorate %{id_value} Location {location}")

        if name:
            self.emit(f'OpName %{id_value} "{name}"')

        spirv_id = SpirvId(id_value, ptr_type.type, name)
        self.variable_value_types[id_value] = type_id
        self.inputs.append(spirv_id)
        return spirv_id

    def register_output(
        self,
        name: str,
        type_id: SpirvId,
        location: int,
        binding: int,
        source_node=None,
    ) -> SpirvId:
        """Register an output variable with location decoration."""
        self.validate_user_defined_interface_type(type_id, "Output", name, source_node)
        ptr_type = self.register_pointer_type(type_id, "Output")

        id_value = self.get_id()
        self.emit(f"%{id_value} = OpVariable %{ptr_type.id} Output")

        self.decorations.append(f"OpDecorate %{id_value} Location {location}")

        if name:
            self.emit(f'OpName %{id_value} "{name}"')

        spirv_id = SpirvId(id_value, ptr_type.type, name)
        self.variable_value_types[id_value] = type_id
        self.outputs.append(spirv_id)
        return spirv_id

    def compute_builtin_info(self, name: str):
        name = self.spirv_builtin_alias(name)
        builtins = {
            "gl_GlobalInvocationID": ("uvec3", "GlobalInvocationId", "Input"),
            "gl_LocalInvocationID": ("uvec3", "LocalInvocationId", "Input"),
            "gl_WorkGroupID": ("uvec3", "WorkgroupId", "Input"),
            "gl_NumWorkGroups": ("uvec3", "NumWorkgroups", "Input"),
            "gl_LocalInvocationIndex": ("uint", "LocalInvocationIndex", "Input"),
            "gl_WorkGroupSize": ("uvec3", "WorkgroupSize", "Constant"),
            "gl_SubgroupSize": ("uint", "SubgroupSize", "Input"),
            "gl_SubgroupInvocationID": (
                "uint",
                "SubgroupLocalInvocationId",
                "Input",
            ),
        }
        return builtins.get(name)

    def spirv_builtin_alias(self, name: str) -> str:
        aliases = {
            "SV_DISPATCHTHREADID": "gl_GlobalInvocationID",
            "SV_GROUPTHREADID": "gl_LocalInvocationID",
            "SV_GROUPID": "gl_WorkGroupID",
            "SV_GROUPINDEX": "gl_LocalInvocationIndex",
            "SV_VERTEXID": "gl_VertexID",
            "SV_INSTANCEID": "gl_InstanceID",
            "SV_PRIMITIVEID": "gl_PrimitiveID",
            "SV_ISFRONTFACE": "gl_FrontFacing",
            "SV_BARYCENTRICS": "gl_BaryCoordEXT",
            "SV_STENCILREF": "gl_FragStencilRefEXT",
            "SV_POSITION": (
                "gl_FragCoord"
                if self.current_execution_model == "Fragment"
                else "gl_Position"
            ),
            "SV_DEPTH": "gl_FragDepth",
            "SV_COVERAGE": "gl_SampleMask",
        }
        return aliases.get(str(name).upper(), name)

    def spirv_output_semantic_alias(self, name: str) -> str:
        aliases = {
            "SV_POSITION": "gl_Position",
            "SV_DEPTH": "gl_FragDepth",
            "SV_COVERAGE": "gl_SampleMask",
            "SV_STENCILREF": "gl_FragStencilRefEXT",
        }
        return aliases.get(str(name).upper(), name)

    def spirv_interface_semantic_alias(
        self,
        semantic: str,
        execution_model: Optional[str],
        storage_class: str,
    ) -> str:
        upper_name = str(semantic).upper()
        if semantic == "gl_Position":
            return (
                "gl_FragCoord"
                if execution_model == "Fragment" and storage_class == "Input"
                else semantic
            )
        if upper_name == "SV_POSITION":
            return (
                "gl_FragCoord"
                if execution_model == "Fragment" and storage_class == "Input"
                else "gl_Position"
            )
        if upper_name == "SV_COVERAGE":
            return (
                "gl_SampleMaskIn"
                if execution_model == "Fragment" and storage_class == "Input"
                else "gl_SampleMask"
            )
        if upper_name == "SV_BARYCENTRICS":
            return "gl_BaryCoordEXT"
        if upper_name == "SV_STENCILREF":
            return "gl_FragStencilRefEXT"
        if upper_name == "SV_DEPTH":
            return "gl_FragDepth"
        return semantic

    def spirv_builtin_cache_name(self, name: str) -> str:
        return self.spirv_builtin_alias(name)

    def stage_builtin_info(self, name: str):
        name = self.spirv_builtin_alias(name)
        builtins = {
            "gl_VertexID": (
                "uint",
                "VertexIndex",
                "Input",
                {"Vertex"},
            ),
            "gl_InstanceID": (
                "uint",
                "InstanceIndex",
                "Input",
                {"Vertex"},
            ),
            "gl_FragCoord": (
                "vec4",
                "FragCoord",
                "Input",
                {"Fragment"},
            ),
            "gl_FrontFacing": (
                "bool",
                "FrontFacing",
                "Input",
                {"Fragment"},
            ),
            "gl_PointCoord": (
                "vec2",
                "PointCoord",
                "Input",
                {"Fragment"},
            ),
            "gl_SampleMaskIn": (
                "int[1]",
                "SampleMask",
                "Input",
                {"Fragment"},
            ),
            "gl_SampleMask": (
                "int[1]",
                "SampleMask",
                "Output",
                {"Fragment"},
            ),
            "gl_BaryCoordEXT": (
                "vec3",
                "BaryCoordKHR",
                "Input",
                {"Fragment"},
            ),
            "gl_BaryCoordNoPerspEXT": (
                "vec3",
                "BaryCoordNoPerspKHR",
                "Input",
                {"Fragment"},
            ),
            "gl_Position": (
                "vec4",
                "Position",
                "Output",
                {"Vertex", "TessellationEvaluation"},
            ),
            "gl_FragDepth": (
                "float",
                "FragDepth",
                "Output",
                {"Fragment"},
            ),
            "gl_FragStencilRefEXT": (
                "int",
                "FragStencilRefEXT",
                "Output",
                {"Fragment"},
            ),
            "gl_InvocationID": (
                "int",
                "InvocationId",
                "Input",
                {"Geometry", "TessellationControl"},
            ),
            "gl_PrimitiveID": (
                "int",
                "PrimitiveId",
                "Input",
                {
                    "Fragment",
                    "Geometry",
                    "TessellationControl",
                    "TessellationEvaluation",
                },
            ),
            "gl_TessCoord": (
                "vec3",
                "TessCoord",
                "Input",
                {"TessellationEvaluation"},
            ),
            "gl_TessLevelOuter": (
                "float[4]",
                "TessLevelOuter",
                {
                    "TessellationControl": "Output",
                    "TessellationEvaluation": "Input",
                },
                {"TessellationControl", "TessellationEvaluation"},
            ),
            "gl_TessLevelInner": (
                "float[2]",
                "TessLevelInner",
                {
                    "TessellationControl": "Output",
                    "TessellationEvaluation": "Input",
                },
                {"TessellationControl", "TessellationEvaluation"},
            ),
            "gl_LaunchIDEXT": (
                "uvec3",
                "LaunchIdKHR",
                "Input",
                {
                    "RayGenerationKHR",
                    "IntersectionKHR",
                    "AnyHitKHR",
                    "ClosestHitKHR",
                    "MissKHR",
                    "CallableKHR",
                },
            ),
            "gl_LaunchSizeEXT": (
                "uvec3",
                "LaunchSizeKHR",
                "Input",
                {
                    "RayGenerationKHR",
                    "IntersectionKHR",
                    "AnyHitKHR",
                    "ClosestHitKHR",
                    "MissKHR",
                    "CallableKHR",
                },
            ),
            "gl_HitTEXT": (
                "float",
                "RayTmaxKHR",
                "Input",
                {"IntersectionKHR", "AnyHitKHR", "ClosestHitKHR"},
            ),
            "gl_HitKindEXT": (
                "uint",
                "HitKindKHR",
                "Input",
                {"AnyHitKHR", "ClosestHitKHR"},
            ),
            "gl_WorldRayOriginEXT": (
                "vec3",
                "WorldRayOriginKHR",
                "Input",
                {"IntersectionKHR", "AnyHitKHR", "ClosestHitKHR", "MissKHR"},
            ),
            "gl_WorldRayDirectionEXT": (
                "vec3",
                "WorldRayDirectionKHR",
                "Input",
                {"IntersectionKHR", "AnyHitKHR", "ClosestHitKHR", "MissKHR"},
            ),
            "gl_ObjectRayOriginEXT": (
                "vec3",
                "ObjectRayOriginKHR",
                "Input",
                {"IntersectionKHR", "AnyHitKHR", "ClosestHitKHR"},
            ),
            "gl_ObjectRayDirectionEXT": (
                "vec3",
                "ObjectRayDirectionKHR",
                "Input",
                {"IntersectionKHR", "AnyHitKHR", "ClosestHitKHR"},
            ),
            "gl_RayTminEXT": (
                "float",
                "RayTminKHR",
                "Input",
                {"IntersectionKHR", "AnyHitKHR", "ClosestHitKHR", "MissKHR"},
            ),
            "gl_InstanceCustomIndexEXT": (
                "int",
                "InstanceCustomIndexKHR",
                "Input",
                {"IntersectionKHR", "AnyHitKHR", "ClosestHitKHR"},
            ),
            "gl_GeometryIndexEXT": (
                "int",
                "GeometryIndexKHR",
                "Input",
                {"IntersectionKHR", "AnyHitKHR", "ClosestHitKHR"},
            ),
        }
        return builtins.get(name)

    def stage_builtin_execution_label(self, execution_models: set) -> str:
        order = [
            "Vertex",
            "Fragment",
            "GLCompute",
            "Geometry",
            "TessellationControl",
            "TessellationEvaluation",
            "MeshEXT",
            "TaskEXT",
            "RayGenerationKHR",
            "IntersectionKHR",
            "AnyHitKHR",
            "ClosestHitKHR",
            "MissKHR",
            "CallableKHR",
        ]
        models = [model for model in order if model in execution_models]
        models.extend(sorted(execution_models - set(models)))
        return ", ".join(models)

    def resolve_stage_builtin_storage_class(
        self, name: str, storage_spec, execution_models: set
    ) -> Optional[str]:
        execution_model = self.current_stage_builtin_execution_model()

        if execution_model is not None and execution_model not in execution_models:
            self.emit(
                f"; WARNING: SPIR-V builtin {name} is only valid in "
                f"{self.stage_builtin_execution_label(execution_models)} stages"
            )
            return None

        if not isinstance(storage_spec, dict):
            return storage_spec

        if execution_model is None:
            self.emit(
                f"; WARNING: SPIR-V builtin {name} requires a single "
                "tessellation execution model to choose Input or Output storage"
            )
            return None

        return storage_spec.get(execution_model)

    def current_stage_builtin_execution_model(self) -> Optional[str]:
        execution_model = self.current_execution_model
        if execution_model is not None:
            return execution_model
        if self.current_function_name is None:
            return None
        function_models = self.function_execution_models.get(
            self.current_function_name, set()
        )
        if len(function_models) == 1:
            return next(iter(function_models))
        return None

    def stage_builtin_cache_key(
        self, name: str, storage_spec, storage_class: str
    ) -> str:
        if isinstance(storage_spec, dict):
            return f"{name}::{storage_class}"
        return name

    def mark_function_interface_variable(self, variable: SpirvId):
        if self.current_function_id is None:
            return

        variables = self.function_interface_variables.setdefault(
            self.current_function_id, []
        )
        if all(existing.id != variable.id for existing in variables):
            variables.append(variable)

        if self.current_function_name is not None:
            named_variables = self.function_interface_variables_by_name.setdefault(
                self.current_function_name, []
            )
            if all(existing.id != variable.id for existing in named_variables):
                named_variables.append(variable)

    def mark_interface_variable_if_needed(self, variable: Optional[SpirvId]):
        if variable is None:
            return

        for interface_variable in self.inputs + self.outputs:
            if interface_variable.id == variable.id:
                self.mark_function_interface_variable(interface_variable)
                return
        if (
            self.include_resource_interface_variables
            and variable.type.storage_class in {"Uniform", "UniformConstant"}
            and any(
                global_variable.id == variable.id
                for global_variable in self.global_variables.values()
            )
        ):
            self.mark_function_interface_variable(variable)

    def merge_function_interface_variables_from_callee(
        self, function_name: str, function_id: Optional[int] = None
    ):
        if function_id is not None:
            for variable in self.function_interface_variables.get(function_id, []):
                self.mark_function_interface_variable(variable)
            if function_id in self.function_interface_variables:
                return

        for variable in self.function_interface_variables_by_name.get(
            function_name, []
        ):
            self.mark_function_interface_variable(variable)

    def mark_builtin_interface_variable(self, variable: SpirvId):
        if variable.id in self.builtin_interface_variable_ids:
            self.mark_function_interface_variable(variable)

    def mark_fragment_depth_replacing_if_needed(
        self, builtin_name: str, storage_class: str
    ):
        if (
            builtin_name != "FragDepth"
            or storage_class != "Output"
            or self.current_execution_model != "Fragment"
            or self.current_function_id is None
        ):
            return
        self.fragment_depth_replacing_function_ids.add(self.current_function_id)

    def mark_fragment_stencil_ref_replacing_if_needed(
        self, builtin_name: str, storage_class: str
    ):
        if (
            builtin_name != "FragStencilRefEXT"
            or storage_class != "Output"
            or self.current_execution_model != "Fragment"
            or self.current_function_id is None
        ):
            return
        self.fragment_stencil_ref_replacing_function_ids.add(self.current_function_id)

    def mark_fragment_builtin_execution_mode_if_needed(
        self, builtin_name: str, storage_class: str
    ):
        self.mark_fragment_depth_replacing_if_needed(builtin_name, storage_class)
        self.mark_fragment_stencil_ref_replacing_if_needed(builtin_name, storage_class)

    def require_stage_builtin_capabilities(self, builtin_name: str):
        if builtin_name in {"BaryCoordKHR", "BaryCoordNoPerspKHR"}:
            self.require_capability("FragmentBarycentricKHR")
            self.require_extension("SPV_KHR_fragment_shader_barycentric")
        elif builtin_name == "FragStencilRefEXT":
            self.require_capability("StencilExportEXT")
            self.require_extension("SPV_EXT_shader_stencil_export")

    def ensure_builtin_variable(self, name: str) -> Optional[SpirvId]:
        builtin = self.ensure_compute_builtin(name)
        if builtin is not None:
            return builtin
        return self.ensure_stage_builtin(name)

    def ensure_compute_builtin(self, name: str) -> Optional[SpirvId]:
        info = self.compute_builtin_info(name)
        if info is None:
            return None

        cache_name = self.spirv_builtin_cache_name(name)
        if cache_name in self.global_variables:
            builtin_id = self.global_variables[cache_name]
            self.mark_builtin_interface_variable(builtin_id)
            return builtin_id

        type_name, builtin_name, storage_class = info
        if builtin_name.startswith("Subgroup"):
            self.require_group_non_uniform()
        type_id = self.map_crossgl_type(type_name)
        if storage_class == "Constant":
            builtin_id = self.register_workgroup_size_builtin(
                name, type_id, builtin_name
            )
        else:
            builtin_id = self.register_builtin_input(name, type_id, builtin_name)

        self.global_variables[cache_name] = builtin_id
        self.mark_builtin_interface_variable(builtin_id)
        return builtin_id

    def ensure_stage_builtin(self, name: str) -> Optional[SpirvId]:
        info = self.stage_builtin_info(name)
        if info is None:
            return None

        type_name, builtin_name, storage_spec, execution_models = info
        storage_class = self.resolve_stage_builtin_storage_class(
            name, storage_spec, execution_models
        )
        if storage_class is None:
            return None

        cache_key = self.stage_builtin_cache_key(
            self.spirv_builtin_cache_name(name), storage_spec, storage_class
        )
        if cache_key in self.global_variables:
            builtin_id = self.global_variables[cache_key]
            self.mark_fragment_builtin_execution_mode_if_needed(
                builtin_name, storage_class
            )
            self.mark_builtin_interface_variable(builtin_id)
            return builtin_id

        type_id = self.map_crossgl_type(type_name)
        if storage_class not in {"Input", "Output"}:
            return None

        if self.current_stage_builtin_execution_model() in {
            "TessellationControl",
            "TessellationEvaluation",
        }:
            self.require_capability("Tessellation")

        builtin_id = self.register_builtin_variable(
            name, type_id, builtin_name, storage_class
        )
        self.global_variables[cache_key] = builtin_id
        self.mark_builtin_interface_variable(builtin_id)
        return builtin_id

    def register_builtin_input(
        self, name: str, type_id: SpirvId, builtin_name: str
    ) -> SpirvId:
        return self.register_builtin_variable(name, type_id, builtin_name, "Input")

    def register_builtin_variable(
        self, name: str, type_id: SpirvId, builtin_name: str, storage_class: str
    ) -> SpirvId:
        self.require_stage_builtin_capabilities(builtin_name)
        ptr_type = self.register_pointer_type(type_id, storage_class)
        id_value = self.get_id()
        self.emit(f"%{id_value} = OpVariable %{ptr_type.id} {storage_class}")
        self.emit(f'OpName %{id_value} "{name}"')
        self.decorations.append(f"OpDecorate %{id_value} BuiltIn {builtin_name}")
        self.mark_fragment_builtin_execution_mode_if_needed(builtin_name, storage_class)

        spirv_id = SpirvId(id_value, ptr_type.type, name)
        self.variable_value_types[id_value] = type_id
        self.builtin_names_by_variable_id[id_value] = builtin_name
        if storage_class == "Input":
            self.inputs.append(spirv_id)
            self.readonly_builtin_pointer_names[id_value] = name
        elif storage_class == "Output":
            self.outputs.append(spirv_id)
        self.builtin_interface_variable_ids.add(id_value)
        return spirv_id

    def register_workgroup_size_builtin(
        self, name: str, type_id: SpirvId, builtin_name: str
    ) -> SpirvId:
        uint_type = self.register_primitive_type("uint")
        x, y, z = self.compute_local_size(self.current_stage)
        components = [self.register_constant(value, uint_type) for value in (x, y, z)]

        id_value = self.get_id()
        component_list = " ".join(f"%{component.id}" for component in components)
        self.emit(f"%{id_value} = OpConstantComposite %{type_id.id} {component_list}")
        self.emit(f'OpName %{id_value} "{name}"')
        self.decorations.append(f"OpDecorate %{id_value} BuiltIn {builtin_name}")

        spirv_id = SpirvId(id_value, type_id.type, name)
        self.value_types[id_value] = type_id
        return spirv_id

    def process_dotted_builtin(self, name: str) -> Optional[SpirvId]:
        if "." not in name:
            return None

        base_name, member_name = name.rsplit(".", 1)
        builtin = self.ensure_builtin_variable(base_name)
        if builtin is None:
            return None

        base_value = self.get_variable_value(builtin)
        member_info = self.vector_member_info(base_value.type.base_type, member_name)
        if member_info is None:
            swizzle_info = self.vector_swizzle_info(
                base_value.type.base_type, member_name
            )
            if swizzle_info is None:
                return None

            indices, _, result_type = swizzle_info
            return self.vector_shuffle(base_value, result_type, indices)

        member_index, member_type = member_info
        return self.composite_extract(base_value, member_type, member_index)

    def register_array_type(
        self, element_type: SpirvId, size: Optional[int] = None
    ) -> SpirvId:
        """Create and register an array type."""
        key = (element_type.id, size)
        if key in self.array_types:
            return self.array_types[key]

        id_value = self.get_id()

        if size is not None:
            size_const = self.register_constant(
                size, self.register_primitive_type("int")
            )
            self.emit(f"%{id_value} = OpTypeArray %{element_type.id} %{size_const.id}")
        else:
            self.emit(f"%{id_value} = OpTypeRuntimeArray %{element_type.id}")

        type_name = f"array_{element_type.type.base_type}_{size if size else 'rt'}"
        spirv_type = SpirvType(type_name)
        spirv_id = SpirvId(id_value, spirv_type, type_name)
        self.array_types[key] = spirv_id
        return spirv_id

    def register_layout_array_type(
        self, element_type: SpirvId, size: Optional[int], layout: str
    ) -> SpirvId:
        """Create a layout-specific array clone for distinct ArrayStride values."""
        key = (element_type.id, size, layout)
        if key in self.layout_array_types:
            return self.layout_array_types[key]

        id_value = self.get_id()
        if size is not None:
            size_const = self.register_constant(
                size, self.register_primitive_type("int")
            )
            self.emit(f"%{id_value} = OpTypeArray %{element_type.id} %{size_const.id}")
        else:
            self.emit(f"%{id_value} = OpTypeRuntimeArray %{element_type.id}")

        type_name = (
            f"array_{element_type.type.base_type}_{size if size else 'rt'}_{layout}"
        )
        spirv_id = SpirvId(id_value, SpirvType(type_name), type_name)
        self.layout_array_types[key] = spirv_id
        return spirv_id

    def determine_array_element_type(self, array_id: "SpirvId") -> Optional["SpirvId"]:
        """Determine the element type of an array based on its SpirvId.

        Args:
            array_id: The SpirvId of the array

        Returns:
            SpirvId of the element type, or None if it cannot be determined
        """
        if (
            not array_id
            or not hasattr(array_id, "type")
            or not hasattr(array_id.type, "base_type")
        ):
            return None

        array_type = array_id.type.base_type

        for (element_type_id, _), arr_type_id in self.array_types.items():
            if arr_type_id.type.base_type == array_type:
                return self.find_registered_type_by_id(element_type_id)

        if array_type.startswith("ptr_"):
            base_type = array_type.replace("ptr_", "", 1)
            for (element_type_id, _), arr_type_id in self.array_types.items():
                if arr_type_id.type.base_type == base_type:
                    return self.find_registered_type_by_id(element_type_id)

            match = re.search(r"array_([^_]+)_", base_type)
            if match:
                element_type_name = match.group(1)

                for type_dict in [
                    self.primitive_types,
                    self.vector_types,
                    self.matrix_types,
                ]:
                    for type_id in type_dict.values():
                        if type_id.type.base_type == element_type_name:
                            return type_id

        # Last resort: Try to parse from type name
        for type_dict in [
            self.primitive_types,
            self.vector_types,
            self.matrix_types,
        ]:
            for type_id in type_dict.values():
                if type_id.type.base_type in array_type:
                    return type_id

        return self.primitive_types["float"]

    def get_function_qualifier(self, func) -> Optional[str]:
        """Return the shader-stage qualifier from old or new function AST shapes."""
        if hasattr(func, "qualifiers") and func.qualifiers:
            return normalize_stage_name(func.qualifiers[0]) if func.qualifiers else None
        if hasattr(func, "qualifier"):
            qualifier = normalize_stage_name(func.qualifier)
            if qualifier in STAGE_QUALIFIER_NAMES:
                return qualifier

        for attr in getattr(func, "attributes", []) or []:
            attr_name = normalize_stage_name(getattr(attr, "name", attr))
            if attr_name in STAGE_QUALIFIER_NAMES:
                return attr_name
        return None

    def stage_key(self, stage_type) -> str:
        """Normalize a stage enum or string to a registry key."""
        if hasattr(stage_type, "value"):
            return stage_type.value
        return str(stage_type).split(".")[-1].lower()

    def spirv_execution_model(self, stage_name: Optional[str]) -> str:
        """Map a CrossGL stage name to a SPIR-V execution model."""
        stage_map = {
            "vertex": "Vertex",
            "fragment": "Fragment",
            "compute": "GLCompute",
            "geometry": "Geometry",
            "tessellation_control": "TessellationControl",
            "tessellation_evaluation": "TessellationEvaluation",
            "mesh": "MeshEXT",
            "task": "TaskEXT",
            "object": "TaskEXT",
            "amplification": "TaskEXT",
            "ray_generation": "RayGenerationKHR",
            "ray_intersection": "IntersectionKHR",
            "ray_closest_hit": "ClosestHitKHR",
            "ray_miss": "MissKHR",
            "ray_any_hit": "AnyHitKHR",
            "ray_callable": "CallableKHR",
        }
        return stage_map.get(stage_name or "fragment", "Fragment")

    def compute_local_size(self, stage) -> Tuple[int, int, int]:
        """Return compute workgroup dimensions from a stage execution config."""
        config = getattr(stage, "execution_config", {}) or {}
        for key in ("local_size", "workgroup_size", "numthreads"):
            value = config.get(key)
            if isinstance(value, (list, tuple)) and len(value) >= 3:
                return tuple(
                    self.positive_local_size_dimension(
                        value[index], key, "xyz"[index], stage
                    )
                    for index in range(3)
                )

        return (
            self.positive_local_size_dimension(
                config.get("local_size_x", 1), "local_size_x", "x", stage
            ),
            self.positive_local_size_dimension(
                config.get("local_size_y", 1), "local_size_y", "y", stage
            ),
            self.positive_local_size_dimension(
                config.get("local_size_z", 1), "local_size_z", "z", stage
            ),
        )

    def positive_local_size_dimension(
        self, value, source: str, axis: str, stage
    ) -> int:
        """Coerce a LocalSize dimension to a valid positive SPIR-V literal."""
        invalid = isinstance(value, bool)
        if invalid:
            dimension = None
        else:
            try:
                dimension = int(value)
            except (TypeError, ValueError):
                dimension = None

        if dimension is not None and dimension > 0:
            return dimension

        warning_key = (id(stage), source, axis)
        if warning_key not in self.local_size_warning_keys:
            self.local_size_warning_keys.add(warning_key)
            self.emit(
                "; WARNING: SPIR-V LocalSize "
                f"{axis} dimension from {source} must be a positive integer "
                "literal; using 1"
            )
        return 1

    def stage_attribute_value(self, stage, attribute_name: str):
        """Return the first argument for a stage entry-point attribute."""
        entry_point = getattr(stage, "entry_point", stage)
        for attr in getattr(entry_point, "attributes", []) or []:
            if str(getattr(attr, "name", "")).lower() != attribute_name:
                continue
            arguments = getattr(attr, "arguments", None)
            if arguments is None:
                arguments = getattr(attr, "args", [])
            if arguments:
                return arguments[0]
        return None

    def stage_has_attribute(self, stage, attribute_name: str) -> bool:
        """Return whether a stage entry point has a flag-style attribute."""
        entry_point = getattr(stage, "entry_point", stage)
        for attr in getattr(entry_point, "attributes", []) or []:
            if str(getattr(attr, "name", "")).lower() == attribute_name:
                return True
        return False

    def stage_attribute_flag_names(self, stage) -> List[str]:
        """Return normalized no-argument stage entry-point attribute names."""
        entry_point = getattr(stage, "entry_point", stage)
        flags = []
        for attr in getattr(entry_point, "attributes", []) or []:
            arguments = getattr(attr, "arguments", None)
            if arguments is None:
                arguments = getattr(attr, "args", [])
            if not arguments:
                flags.append(str(getattr(attr, "name", "")).lower())
        return flags

    def stage_layout_value(self, stage, attribute_name: str):
        """Return the first argument for a stage layout qualifier."""
        for layout in getattr(stage, "layout_qualifiers", []) or []:
            for entry in getattr(layout, "entries", []) or []:
                if str(getattr(entry, "name", "")).lower() != attribute_name:
                    continue
                arguments = getattr(entry, "arguments", None)
                if arguments is None:
                    arguments = getattr(entry, "args", [])
                if arguments:
                    return arguments[0]
        return None

    def stage_layout_direction_value(self, stage, direction: str, attribute_names):
        """Return a stage layout value matching a direction and name set."""
        names = {str(name).lower() for name in attribute_names}
        for layout in getattr(stage, "layout_qualifiers", []) or []:
            if getattr(layout, "direction", None) != direction:
                continue
            for entry in getattr(layout, "entries", []) or []:
                if str(getattr(entry, "name", "")).lower() not in names:
                    continue
                arguments = getattr(entry, "arguments", None)
                if arguments is None:
                    arguments = getattr(entry, "args", [])
                if arguments:
                    return arguments[0]
        return None

    def stage_layout_flag_names(self, stage, direction: Optional[str] = None):
        """Return normalized no-argument stage layout entries."""
        flags = []
        for layout in getattr(stage, "layout_qualifiers", []) or []:
            if (
                direction is not None
                and getattr(layout, "direction", None) != direction
            ):
                continue
            for entry in getattr(layout, "entries", []) or []:
                arguments = getattr(entry, "arguments", None)
                if arguments is None:
                    arguments = getattr(entry, "args", [])
                if not arguments:
                    flags.append(str(getattr(entry, "name", "")).lower())
        return flags

    def normalized_stage_value(self, value) -> Optional[str]:
        """Return a normalized string for stage metadata values."""
        text = self.attribute_value_to_string(value)
        if text is None:
            return None
        return str(text).strip().strip('"').lower()

    def positive_stage_int_value(self, value, context: str, default: int = 1) -> int:
        """Return a positive integer stage metadata value or a safe default."""
        literal = self.literal_int_argument(value)
        if literal is not None and literal > 0:
            return literal
        self.emit(
            f"; WARNING: SPIR-V {context} must be a positive integer literal; "
            f"using {default}"
        )
        return default

    def stage_positive_int_metadata(
        self,
        stage,
        context: str,
        attribute_names=(),
        layout_direction: Optional[str] = None,
        layout_names=(),
        default: int = 1,
    ) -> int:
        """Return positive integer metadata from attributes or layouts."""
        for attribute_name in attribute_names:
            value = self.stage_attribute_value(stage, attribute_name)
            if value is not None:
                return self.positive_stage_int_value(value, context, default)

        if layout_direction is not None and layout_names:
            value = self.stage_layout_direction_value(
                stage, layout_direction, layout_names
            )
            if value is not None:
                return self.positive_stage_int_value(value, context, default)

        return default

    def mesh_stage_limit(self, stage, attribute_name: str) -> Optional[int]:
        """Return a literal mesh output limit from function or layout metadata."""
        for value in (
            self.stage_attribute_value(stage, attribute_name),
            self.stage_layout_value(stage, attribute_name),
        ):
            limit = self.literal_int_argument(value)
            if limit is not None:
                return max(0, limit)
        return None

    def mesh_stage_topology_mode(self, stage) -> str:
        """Return the SPIR-V mesh output-topology execution mode."""
        topology = self.stage_attribute_value(stage, "outputtopology")
        topology_name = self.attribute_value_to_string(topology)
        if topology_name is None:
            for layout in getattr(stage, "layout_qualifiers", []) or []:
                if getattr(layout, "direction", None) != "out":
                    continue
                for entry in getattr(layout, "entries", []) or []:
                    entry_name = str(getattr(entry, "name", "")).lower()
                    if entry_name in {"point", "points", "line", "lines"}:
                        topology_name = entry_name
                        break
                    if entry_name in {"triangle", "triangles"}:
                        topology_name = entry_name
                        break
                if topology_name is not None:
                    break

        topology_modes = {
            "point": "OutputPoints",
            "points": "OutputPoints",
            "line": "OutputLinesEXT",
            "lines": "OutputLinesEXT",
            "triangle": "OutputTrianglesEXT",
            "triangles": "OutputTrianglesEXT",
        }
        normalized = str(topology_name or "triangle").lower()
        if normalized not in topology_modes:
            raise ValueError(
                "SPIR-V mesh stage outputtopology must be point, line, or "
                f"triangle: {topology_name}"
            )
        return topology_modes[normalized]

    def geometry_input_mode(self, stage) -> str:
        """Return the SPIR-V geometry input primitive execution mode."""
        mode_map = {
            "point": "InputPoints",
            "points": "InputPoints",
            "line": "InputLines",
            "lines": "InputLines",
            "line_adjacency": "InputLinesAdjacency",
            "lines_adjacency": "InputLinesAdjacency",
            "lineadj": "InputLinesAdjacency",
            "triangle": "Triangles",
            "triangles": "Triangles",
            "tri": "Triangles",
            "triangle_adjacency": "InputTrianglesAdjacency",
            "triangles_adjacency": "InputTrianglesAdjacency",
            "triangleadj": "InputTrianglesAdjacency",
        }
        for flag in self.stage_attribute_flag_names(stage):
            if flag in mode_map:
                return mode_map[flag]
        for flag in self.stage_layout_flag_names(stage, "in"):
            if flag in mode_map:
                return mode_map[flag]
        return "InputPoints"

    def geometry_output_mode(self, stage) -> str:
        """Return the SPIR-V geometry output primitive execution mode."""
        mode_map = {
            "point": "OutputPoints",
            "points": "OutputPoints",
            "line": "OutputLineStrip",
            "lines": "OutputLineStrip",
            "line_strip": "OutputLineStrip",
            "triangle": "OutputTriangleStrip",
            "triangles": "OutputTriangleStrip",
            "tri": "OutputTriangleStrip",
            "triangle_strip": "OutputTriangleStrip",
        }
        value = self.stage_attribute_value(stage, "outputtopology")
        normalized = self.normalized_stage_value(value)
        if normalized in mode_map:
            return mode_map[normalized]
        if normalized is not None:
            self.emit(
                "; WARNING: SPIR-V geometry outputtopology must be point, "
                f"line_strip, or triangle_strip; using points: {normalized}"
            )
        for flag in self.stage_layout_flag_names(stage, "out"):
            if flag in mode_map:
                return mode_map[flag]
        return "OutputPoints"

    def emit_geometry_execution_modes(self, function_id: SpirvId, stage):
        """Emit geometry capability and execution modes."""
        self.require_capability("Geometry")
        max_vertices = self.stage_positive_int_metadata(
            stage,
            "geometry OutputVertices",
            attribute_names=("max_vertices", "maxvertexcount"),
            layout_direction="out",
            layout_names=("max_vertices", "maxvertexcount"),
        )
        invocations = self.stage_positive_int_metadata(
            stage,
            "geometry Invocations",
            attribute_names=("invocations",),
            layout_direction="in",
            layout_names=("invocations",),
        )
        self.emit(
            f"OpExecutionMode %{function_id.id} {self.geometry_input_mode(stage)}"
        )
        self.emit(
            f"OpExecutionMode %{function_id.id} {self.geometry_output_mode(stage)}"
        )
        self.emit(f"OpExecutionMode %{function_id.id} OutputVertices {max_vertices}")
        if invocations != 1:
            self.emit(f"OpExecutionMode %{function_id.id} Invocations {invocations}")

    def tessellation_control_output_vertices(self, stage) -> int:
        """Return tessellation-control output control-point count."""
        return self.stage_positive_int_metadata(
            stage,
            "tessellation_control OutputVertices",
            attribute_names=("outputcontrolpoints", "vertices"),
            layout_direction="out",
            layout_names=("vertices", "outputcontrolpoints"),
        )

    def tessellation_metadata_sources(self, stage):
        """Return evaluation metadata sources, preferring evaluation metadata."""
        sources = [stage]
        if (
            self.tessellation_control_stage is not None
            and self.tessellation_control_stage is not stage
        ):
            sources.append(self.tessellation_control_stage)
        return sources

    def tessellation_domain_mode(self, stage) -> str:
        """Return the tessellation-evaluation domain execution mode."""
        mode_map = {
            "tri": "Triangles",
            "triangle": "Triangles",
            "triangles": "Triangles",
            "triangle_cw": "Triangles",
            "triangle_ccw": "Triangles",
            "quad": "Quads",
            "quads": "Quads",
            "isoline": "Isolines",
            "isolines": "Isolines",
        }
        for source in self.tessellation_metadata_sources(stage):
            for attribute_name in ("domain", "outputtopology"):
                value = self.stage_attribute_value(source, attribute_name)
                normalized = self.normalized_stage_value(value)
                if normalized in mode_map:
                    return mode_map[normalized]
                if attribute_name == "domain" and normalized is not None:
                    self.emit(
                        "; WARNING: SPIR-V tessellation domain must be "
                        f"triangle, quads, or isolines; using triangles: {normalized}"
                    )
            for flag in self.stage_layout_flag_names(source, "in"):
                if flag in mode_map:
                    return mode_map[flag]
        return "Triangles"

    def tessellation_spacing_mode(self, stage) -> str:
        """Return the tessellation-evaluation spacing execution mode."""
        mode_map = {
            "integer": "SpacingEqual",
            "equal": "SpacingEqual",
            "equal_spacing": "SpacingEqual",
            "fractional_even": "SpacingFractionalEven",
            "fractional_even_spacing": "SpacingFractionalEven",
            "fractional_odd": "SpacingFractionalOdd",
            "fractional_odd_spacing": "SpacingFractionalOdd",
        }
        for source in self.tessellation_metadata_sources(stage):
            value = self.stage_attribute_value(source, "partitioning")
            normalized = self.normalized_stage_value(value)
            if normalized in mode_map:
                return mode_map[normalized]
            if normalized is not None:
                self.emit(
                    "; WARNING: SPIR-V tessellation partitioning must be "
                    "integer, fractional_even, or fractional_odd; using equal "
                    f"spacing: {normalized}"
                )
            for flag in self.stage_layout_flag_names(source, "in"):
                if flag in mode_map:
                    return mode_map[flag]
        return "SpacingEqual"

    def tessellation_vertex_order_mode(self, stage) -> str:
        """Return the tessellation-evaluation vertex-order execution mode."""
        mode_map = {
            "cw": "VertexOrderCw",
            "triangle_cw": "VertexOrderCw",
            "ccw": "VertexOrderCcw",
            "triangle_ccw": "VertexOrderCcw",
        }
        for source in self.tessellation_metadata_sources(stage):
            value = self.stage_attribute_value(source, "outputtopology")
            normalized = self.normalized_stage_value(value)
            if normalized in mode_map:
                return mode_map[normalized]
            for flag in self.stage_attribute_flag_names(source):
                if flag in mode_map:
                    return mode_map[flag]
            for flag in self.stage_layout_flag_names(source, "in"):
                if flag in mode_map:
                    return mode_map[flag]
        return "VertexOrderCcw"

    def tessellation_point_mode(self, stage) -> bool:
        """Return whether tessellation evaluation should emit PointMode."""
        for source in self.tessellation_metadata_sources(stage):
            topology = self.normalized_stage_value(
                self.stage_attribute_value(source, "outputtopology")
            )
            if topology in {"point", "points"}:
                return True
            if self.stage_has_attribute(source, "point_mode"):
                return True
            if "point_mode" in self.stage_layout_flag_names(source, "in"):
                return True
        return False

    def emit_tessellation_execution_modes(
        self, execution_model: str, function_id: SpirvId, stage
    ):
        """Emit tessellation capability and execution modes."""
        self.require_capability("Tessellation")
        if execution_model == "TessellationControl":
            self.emit(
                f"OpExecutionMode %{function_id.id} OutputVertices "
                f"{self.tessellation_control_output_vertices(stage)}"
            )
            return

        self.emit(
            f"OpExecutionMode %{function_id.id} {self.tessellation_domain_mode(stage)}"
        )
        self.emit(
            f"OpExecutionMode %{function_id.id} {self.tessellation_spacing_mode(stage)}"
        )
        self.emit(
            f"OpExecutionMode %{function_id.id} "
            f"{self.tessellation_vertex_order_mode(stage)}"
        )
        if self.tessellation_point_mode(stage):
            self.emit(f"OpExecutionMode %{function_id.id} PointMode")

    def mesh_stage_output_limits(self, function_id: SpirvId, stage) -> Tuple[int, int]:
        """Return OutputVertices and OutputPrimitivesEXT execution-mode limits."""
        observed_vertices, observed_primitives = (
            self.mesh_output_counts_by_function.get(function_id.id, (None, None))
        )
        max_vertices = self.mesh_stage_limit(stage, "max_vertices")
        max_primitives = self.mesh_stage_limit(stage, "max_primitives")
        if observed_vertices is not None:
            max_vertices = max(observed_vertices, max_vertices or 0)
        if observed_primitives is not None:
            max_primitives = max(observed_primitives, max_primitives or 0)
        return max(1, max_vertices or 1), max(1, max_primitives or 1)

    def emit_entry_point(
        self, execution_model: str, function_id: SpirvId, name: str, stage=None
    ):
        """Emit SPIR-V entry-point and execution-mode declarations."""
        interface_variables = list(
            self.function_interface_variables.get(function_id.id, [])
        )
        interface_variables.extend(self.entry_point_private_variables)
        if execution_model == "TaskEXT":
            task_payload = self.task_payload_interface_by_function.get(function_id.id)
            if task_payload is not None:
                interface_variables.append(task_payload)

        seen_interface_ids = set()
        interface_ids = []
        for variable in interface_variables:
            if variable.id in seen_interface_ids:
                continue
            seen_interface_ids.add(variable.id)
            interface_ids.append(f"%{variable.id}")

        interface_ids = " ".join(interface_ids)
        interface_suffix = f" {interface_ids}" if interface_ids else ""
        self.emit(
            f'OpEntryPoint {execution_model} %{function_id.id} "{name}"'
            f"{interface_suffix}"
        )
        if execution_model == "Fragment":
            self.emit(f"OpExecutionMode %{function_id.id} OriginUpperLeft")
            if function_id.id in self.fragment_depth_replacing_function_ids:
                self.emit(f"OpExecutionMode %{function_id.id} DepthReplacing")
            if function_id.id in self.fragment_stencil_ref_replacing_function_ids:
                self.emit(f"OpExecutionMode %{function_id.id} StencilRefReplacingEXT")
        elif execution_model == "Geometry":
            self.emit_geometry_execution_modes(function_id, stage)
        elif execution_model in {"TessellationControl", "TessellationEvaluation"}:
            self.emit_tessellation_execution_modes(execution_model, function_id, stage)
        elif execution_model in {"GLCompute", "MeshEXT", "TaskEXT"}:
            x, y, z = self.compute_local_size(stage)
            if self.requires_compute_derivatives:
                x = max(2, x + (x % 2))
                y = max(2, y + (y % 2))
            self.emit(f"OpExecutionMode %{function_id.id} LocalSize {x} {y} {z}")
            if self.requires_compute_derivatives:
                self.emit(f"OpExecutionMode %{function_id.id} DerivativeGroupQuadsKHR")
            if execution_model in {"MeshEXT", "TaskEXT"}:
                self.require_capability("MeshShadingEXT")
                self.require_extension("SPV_EXT_mesh_shader")
            if execution_model == "MeshEXT":
                max_vertices, max_primitives = self.mesh_stage_output_limits(
                    function_id, stage
                )
                self.emit(
                    f"OpExecutionMode %{function_id.id} OutputVertices {max_vertices}"
                )
                self.emit(
                    f"OpExecutionMode %{function_id.id} "
                    f"OutputPrimitivesEXT {max_primitives}"
                )
                self.emit(
                    f"OpExecutionMode %{function_id.id} "
                    f"{self.mesh_stage_topology_mode(stage)}"
                )
        elif execution_model in {
            "RayGenerationKHR",
            "IntersectionKHR",
            "AnyHitKHR",
            "ClosestHitKHR",
            "MissKHR",
            "CallableKHR",
        }:
            self.require_capability("RayTracingKHR")
            self.require_extension("SPV_KHR_ray_tracing")

    def spirv_module_version(self) -> str:
        if self.include_resource_interface_variables:
            return "1.4"
        if "MeshShadingEXT" in self.required_capabilities:
            return "1.4"
        if "RayTracingKHR" in self.required_capabilities:
            return "1.4"
        if any(
            capability.startswith("GroupNonUniform")
            for capability in self.required_capabilities
        ):
            return "1.3"
        return "1.0"

    def ordered_module_lines(self) -> List[str]:
        """Return SPIR-V assembly lines in logical module-layout order."""
        header_lines = self.code_lines[:3]
        if len(header_lines) > 1:
            header_lines[1] = f"; Version: {self.spirv_module_version()}"
        bound_line = f"; Bound: {self.next_id}"
        raw_lines = self.code_lines[4:]

        capabilities = ["OpCapability Shader"] + [
            f"OpCapability {capability}"
            for capability in sorted(self.required_capabilities)
        ]
        extensions = [
            f'OpExtension "{extension}"'
            for extension in sorted(self.required_extensions)
        ]
        imports = []
        memory_model = []
        entry_points = []
        execution_modes = []
        debug_names = []
        annotations = []
        declarations = []
        global_variables = []
        body = []

        for line in raw_lines:
            if line.startswith("OpCapability "):
                if line not in capabilities:
                    capabilities.append(line)
            elif line.startswith("OpExtension "):
                if line not in extensions:
                    extensions.append(line)
            elif " = OpExtInstImport " in line:
                imports.append(line)
            elif line.startswith("OpMemoryModel "):
                memory_model.append(line)
            elif line.startswith("OpEntryPoint "):
                entry_points.append(line)
            elif line.startswith("OpExecutionMode"):
                execution_modes.append(line)
            elif line.startswith(("OpName ", "OpMemberName ", "OpString ", "OpLine ")):
                debug_names.append(line)
            elif line.startswith(("OpDecorate ", "OpMemberDecorate ")):
                annotations.append(line)
            elif re.match(r"%\d+ = (OpType|OpConstant|OpSpecConstant|OpUndef)", line):
                declarations.append(line)
            elif re.match(r"%\d+ = OpVariable %\d+ (?!Function\b)", line):
                global_variables.append(line)
            else:
                body.append(line)

        annotations.extend(self.decorations)

        def unique(lines: List[str]) -> List[str]:
            seen = set()
            result = []
            for line in lines:
                if line in seen:
                    continue
                seen.add(line)
                result.append(line)
            return result

        body = self.ordered_function_body_lines(body)

        return (
            header_lines
            + [bound_line]
            + unique(capabilities)
            + unique(extensions)
            + imports
            + memory_model
            + entry_points
            + execution_modes
            + unique(debug_names)
            + unique(annotations)
            + declarations
            + global_variables
            + body
        )

    def ordered_function_body_lines(self, lines: List[str]) -> List[str]:
        """Move function-scope variables into each function's first block."""
        ordered = []
        in_function = False
        first_block_insert_at = None
        pending_variables = []

        def is_function_variable(line: str) -> bool:
            return re.match(r"%\d+ = OpVariable %\d+ Function\b", line) is not None

        def is_function_start(line: str) -> bool:
            return re.match(r"%\d+ = OpFunction\b", line) is not None

        def is_label(line: str) -> bool:
            return re.match(r"%\d+ = OpLabel\b", line) is not None

        for line in lines:
            if is_function_start(line):
                in_function = True
                first_block_insert_at = None
                pending_variables = []
                ordered.append(line)
                continue

            if in_function and line == "OpFunctionEnd":
                if pending_variables:
                    if first_block_insert_at is None:
                        ordered.extend(pending_variables)
                    else:
                        ordered[first_block_insert_at:first_block_insert_at] = (
                            pending_variables
                        )
                ordered.append(line)
                in_function = False
                first_block_insert_at = None
                pending_variables = []
                continue

            if in_function and is_function_variable(line):
                if first_block_insert_at is None:
                    pending_variables.append(line)
                else:
                    ordered.insert(first_block_insert_at, line)
                    first_block_insert_at += 1
                continue

            ordered.append(line)

            if in_function and first_block_insert_at is None and is_label(line):
                first_block_insert_at = len(ordered)
                if pending_variables:
                    ordered[first_block_insert_at:first_block_insert_at] = (
                        pending_variables
                    )
                    first_block_insert_at += len(pending_variables)
                    pending_variables = []

        return ordered

    def reject_unsupported_generic_functions(self, ast_node):
        """Reject generic functions that have no concrete SPIR-V specialization."""
        specialized_source_names = {
            key[0]
            for key in self.generic_function_specializations or {}
            if isinstance(key, tuple) and key
        }
        functions = list(iter_function_nodes(ast_node))
        call_scan_roots = [
            func for func in functions if not generic_function_parameters(func)
        ] + list((self.generic_function_specializations or {}).values())
        called_function_names = set()
        for function_node in call_scan_roots:
            called_function_names.update(
                self.function_call_name(node)
                for node in self.walk_ast_nodes(getattr(function_node, "body", None))
                if isinstance(node, FunctionCallNode) and self.function_call_name(node)
            )
        top_level_functions = {
            id(func) for func in getattr(ast_node, "functions", []) or []
        }
        for func in functions:
            generic_params = generic_function_parameters(func)
            if not generic_params:
                continue
            if getattr(func, "name", None) in specialized_source_names:
                continue
            if id(func) in top_level_functions and (
                (
                    func.name not in called_function_names
                    and getattr(func, "source_location", None) is None
                )
                or self.has_concrete_function_overload(func, functions)
                or self.function_has_storage_buffer_parameters(func)
            ):
                continue
            raise self.unsupported_generic_function_error(func)

    def has_concrete_function_overload(self, function_node, functions):
        function_name = getattr(function_node, "name", None)
        if not function_name:
            return False
        return any(
            other is not function_node
            and getattr(other, "name", None) == function_name
            and not generic_function_parameters(other)
            for other in functions
        )

    def generate(self, ast):
        """Generate SPIR-V code from a CrossGL AST."""
        if not isinstance(ast, ShaderNode):
            return "; Error: Not a shader node"

        self.reset_generation_state()

        self.emit("; SPIR-V")
        self.emit("; Version: 1.0")
        self.emit("; Generator: CrossGL Vulkan SPIR-V Generator")
        self.emit("; Schema: 0")

        self.emit("OpCapability Shader")

        self.glsl_std450_id = self.get_id()
        self.emit(f'%{self.glsl_std450_id} = OpExtInstImport "GLSL.std.450"')

        self.emit("OpMemoryModel Logical GLSL450")

        self.register_primitive_type("void")
        self.register_primitive_type("bool")
        self.register_primitive_type("int")
        self.register_primitive_type("float")

        self.literal_int_constants = collect_literal_int_constants(
            getattr(ast, "constants", [])
        )

        float_type = self.primitive_types["float"]
        for i in range(2, 5):
            self.register_vector_type(float_type, i)

        struct_declarations = list(getattr(ast, "structs", []) or [])
        for stage in (getattr(ast, "stages", None) or {}).values():
            struct_declarations.extend(getattr(stage, "local_structs", []) or [])

        self.struct_declarations = {}
        self.enum_declarations = {}

        def collect_type_declarations(nodes):
            for node in nodes or []:
                if isinstance(node, StructNode):
                    self.struct_declarations.setdefault(node.name, node)
                    collect_type_declarations(getattr(node, "members", []) or [])
                elif isinstance(node, EnumNode):
                    self.enum_declarations.setdefault(node.name, node)

        collect_type_declarations(struct_declarations)
        self.glsl_buffer_block_type_names = self.collect_glsl_buffer_block_type_names(
            ast
        )

        self.generic_enum_struct_definitions = collect_generic_enum_struct_definitions(
            struct_declarations
        )
        self.generic_enum_specializations = collect_generic_enum_specializations(
            ast,
            self.generic_enum_struct_definitions,
            self.type_name_string,
        )

        self.collect_enum_metadata(ast.structs)
        for stage in (getattr(ast, "stages", None) or {}).values():
            self.collect_enum_metadata(getattr(stage, "local_structs", []) or [])
        self.register_generic_enum_metadata()

        self.process_enum_structs(ast.structs)
        for stage in (getattr(ast, "stages", None) or {}).values():
            self.process_enum_structs(getattr(stage, "local_structs", []) or [])

        for struct in ast.structs:
            if not isinstance(struct, StructNode):
                continue
            self.process_crossgl_struct(struct)
        for stage in (getattr(ast, "stages", None) or {}).values():
            for struct in getattr(stage, "local_structs", []) or []:
                if not isinstance(struct, StructNode):
                    continue
                self.process_crossgl_struct(struct)

        for constant in getattr(ast, "constants", []) or []:
            if isinstance(constant, ConstantNode):
                self.process_named_constant_declaration(constant)

        global_specialization_constant_declarations = set()
        for var in getattr(ast, "global_variables", []) or []:
            declaration = self.spirv_specialization_constant_declaration_node(var)
            if declaration is None:
                continue
            self.process_named_constant_declaration(declaration)
            global_specialization_constant_declarations.add(id(declaration))

        all_functions = self.collect_ast_functions(ast)
        generic_function_specializations = prepare_generic_function_specializations(
            self,
            all_functions,
        )
        if generic_function_specializations:
            all_functions = all_functions + list(
                generic_function_specializations.values()
            )
        self.reject_unsupported_generic_functions(ast)
        self.function_nodes_by_name = self.collect_functions_by_name(all_functions)

        self.function_resource_array_type_hints = (
            self.collect_resource_array_parameter_type_hints(ast)
        )
        self.collect_stage_local_resource_array_parameter_type_hints(ast)
        self.function_image_access_requirements = (
            self.collect_function_image_access_requirements_for_ast(ast)
        )
        self.collect_stage_local_function_image_access_metadata(ast)
        self.function_storage_buffer_access_requirements = (
            self.collect_function_storage_buffer_access_requirements_for_ast(ast)
        )
        self.collect_stage_local_function_storage_buffer_metadata(ast)
        self.inline_storage_buffer_functions = (
            self.collect_inline_storage_buffer_functions(ast)
        )
        self.function_execution_models = self.collect_function_execution_models(ast)
        (
            self.spirv_skipped_function_parameter_indices,
            self.spirv_skipped_function_parameter_indices_by_id,
        ) = self.collect_unused_array_parameter_index_maps(all_functions)
        self.function_stage_input_dependencies = (
            self.collect_function_stage_object_dependencies(ast, None, "input")
        )
        self.function_stage_output_dependencies = (
            self.collect_function_stage_object_dependencies(ast, None, "output")
        )
        self.function_storage_image_pointer_params = (
            self.collect_storage_image_pointer_parameters(ast)
        )
        self.collect_stage_local_storage_image_pointer_metadata(ast)
        self.reserve_explicit_resource_bindings(ast)
        for stage_type, stage in (getattr(ast, "stages", None) or {}).items():
            if self.stage_key(stage_type) == "tessellation_control":
                self.tessellation_control_stage = stage
                break

        for cbuffer in getattr(ast, "cbuffers", []) or []:
            self.process_cbuffer_declaration(cbuffer)

        for var in getattr(ast, "global_variables", []):
            declaration = self.spirv_specialization_constant_declaration_node(var)
            if (
                declaration is not None
                and id(declaration) in global_specialization_constant_declarations
            ):
                continue
            self.process_global_variable_declaration(var)

        top_level_entries = []
        helper_functions = []
        for func in ast.functions:
            if getattr(func, "body", None) is None:
                continue
            if generic_function_parameters(func):
                helper_functions.extend(generic_function_emission_list(self, func))
                continue
            qualifier = self.get_function_qualifier(func)

            if func.name == "main" or qualifier in [
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
            ]:
                top_level_entries.append((func, qualifier))
            else:
                helper_functions.append(func)

        if self.tessellation_control_stage is None:
            for func, qualifier in top_level_entries:
                if qualifier == "tessellation_control":
                    self.tessellation_control_stage = func
                    break

        for func in self.order_functions_by_dependencies(helper_functions):
            if self.function_has_storage_buffer_parameters(func):
                continue
            self.process_function_node(func)

        entry_points = []

        if getattr(ast, "stages", None):
            for stage_type, stage in ast.stages.items():
                previous_execution_model = self.current_execution_model
                previous_stage = self.current_stage
                self.current_stage = stage
                self.current_execution_model = self.spirv_execution_model(
                    self.stage_key(stage_type)
                )
                try:
                    for var in getattr(stage, "local_variables", []):
                        if not self.global_variable_exists_for_current_stage(var.name):
                            self.process_global_variable_declaration(var)
                finally:
                    self.current_execution_model = previous_execution_model
                    self.current_stage = previous_stage

            processed_local_functions = set()
            for stage in ast.stages.values():
                local_functions = [
                    func
                    for func in getattr(stage, "local_functions", [])
                    if id(func) not in processed_local_functions
                    and getattr(func, "body", None) is not None
                ]
                for func in self.order_functions_by_dependencies(local_functions):
                    if id(func) not in processed_local_functions:
                        if generic_function_parameters(func):
                            for specialized_func in generic_function_emission_list(
                                self, func
                            ):
                                self.process_function_node(
                                    specialized_func, stage=stage
                                )
                            processed_local_functions.add(id(func))
                            continue
                        if self.function_has_storage_buffer_parameters(func):
                            processed_local_functions.add(id(func))
                            continue
                        self.process_function_node(func, stage=stage)
                        processed_local_functions.add(id(func))

            for stage_type, stage in ast.stages.items():
                entry_function = stage.entry_point
                function_id = self.process_function_node(entry_function, stage=stage)
                stage_name = self.stage_key(stage_type)
                execution_model = self.spirv_execution_model(stage_name)
                entry_points.append(
                    (execution_model, function_id, entry_function.name, stage)
                )
        else:
            for func, qualifier in top_level_entries:
                function_id = self.process_function_node(func)
                execution_model = self.spirv_execution_model(qualifier)
                is_tessellation_entry = execution_model in {
                    "TessellationControl",
                    "TessellationEvaluation",
                }
                stage_metadata = func if is_tessellation_entry else None
                entry_points.append(
                    (execution_model, function_id, func.name, stage_metadata)
                )

        if entry_points:
            self.main_fn_id = entry_points[0][1].id
            for execution_model, function_id, entry_name, stage in entry_points:
                self.emit_entry_point(execution_model, function_id, entry_name, stage)

        return "\n".join(self.ordered_module_lines())

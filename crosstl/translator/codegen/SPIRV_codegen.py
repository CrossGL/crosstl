"""CrossGL-to-Vulkan SPIR-V code generator."""

import re
from typing import List, Optional, Tuple, Union

from .array_utils import parse_array_type, detect_array_element_type
from .image_access_contracts import (
    collect_function_image_access_requirements,
    collect_function_parameter_names,
    image_access_diagnostic_name,
    image_access_requirement_label,
    image_access_satisfies_requirement,
)
from ..ast import (
    AssignmentNode,
    ArrayAccessNode,
    ArrayLiteralNode,
    ArrayNode,
    BinaryOpNode,
    BreakNode,
    ContinueNode,
    DoWhileNode,
    ForInNode,
    ForNode,
    FunctionCallNode,
    IdentifierNode,
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
    SwitchNode,
    TernaryOpNode,
    UnaryOpNode,
    VariableNode,
    WaveOpNode,
    WhileNode,
    WildcardPatternNode,
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


class VulkanSPIRVCodeGen:
    """Generates SPIR-V code from a CrossGL shader AST."""

    def __init__(self):
        """Initialize an empty SPIR-V module-generation state."""
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

        self.required_capabilities = set()
        self.global_variables = {}
        self.local_variables = {}
        self.variable_value_types = {}
        self.value_types = {}
        self.constants = {}
        self.vector_constants = {}
        self.composite_constants = {}
        self.resource_type_metadata = {}
        self.structured_buffer_metadata = {}
        self.storage_buffer_access_metadata = {}
        self.precise_global_variables = set()
        self.precise_local_variables = set()
        self.no_contraction_ids = set()
        self.precise_expression_depth = 0

        self.functions = {}
        self.function_signatures = {}
        self.function_parameter_names = {}
        self.function_image_access_requirements = {}
        self.function_storage_buffer_access_requirements = {}
        self.inline_storage_buffer_functions = {}
        self.function_resource_array_params = {}
        self.function_resource_array_type_hints = {}
        self.function_storage_image_pointer_params = {}
        self.function_execution_models = {}
        self.current_execution_model = None
        self.current_function_id = None
        self.current_stage = None
        self.current_return_type = None
        self.mesh_output_counts_by_function = {}

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
        if sampled == 1 and dim == "1D":
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
            constant_value = str(value)
            self.emit(f"%{id_value} = OpConstant %{type_id.id} {constant_value}")

        spirv_id = SpirvId(id_value, type_id.type, f"{type_name}_{value}")
        self.value_types[id_value] = type_id
        self.constants[key] = spirv_id
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
        )

    def image_offset_operand(self, offset_id: SpirvId) -> str:
        if self.is_constant_instruction(offset_id):
            return f"ConstOffset %{offset_id.id}"

        self.require_capability("ImageGatherExtended")
        return f"Offset %{offset_id.id}"

    def image_operands(self, *operands: str) -> str:
        masks = []
        values = []
        for operand in operands:
            if not operand:
                continue
            parts = operand.split()
            masks.append(parts[0])
            values.extend(parts[1:])
        if not masks:
            return ""
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
        return spirv_id

    def convert_value_for_store(
        self, variable_id: SpirvId, value_id: SpirvId
    ) -> SpirvId:
        """Convert a stored value to the pointer's known pointee type when possible."""
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
        numeric_types = {"int", "uint", "float", "double"}
        normalized = [self.normalize_primitive_name(name) for name in type_names]
        if any(name not in numeric_types for name in normalized):
            return None
        if "double" in normalized:
            return "double"
        if "float" in normalized:
            return "float"
        if "int" in normalized:
            return "int"
        return "uint"

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
            spv_op = (
                unsigned_op
                if component_type == "uint"
                else signed_op if component_type == "int" else float_op
            )
        elif op in arithmetic_ops:
            result_type, left, right = self.align_binary_arithmetic_operands(
                result_type, left, right
            )
            float_op, signed_op, unsigned_op = arithmetic_ops[op]
            component_type = self.scalar_or_vector_component_type(left.type)
            spv_op = (
                unsigned_op
                if component_type == "uint"
                else signed_op if component_type == "int" else float_op
            )
        else:
            spv_op = {
                "&": "OpBitwiseAnd",
                "|": "OpBitwiseOr",
                "^": "OpBitwiseXor",
                "<<": "OpShiftLeftLogical",
                ">>": "OpShiftRightLogical",
            }.get(op, f"Op{op}")

        id_value = self.get_id()
        self.emit(f"%{id_value} = {spv_op} %{result_type.id} %{left.id} %{right.id}")
        self.decorate_no_contraction_result(id_value, spv_op, result_type)

        spirv_id = SpirvId(id_value, result_type.type)
        self.value_types[id_value] = result_type
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
            return bool_type, left, right

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

    def comparison_result_type_and_operands(
        self, left: SpirvId, right: SpirvId
    ) -> Tuple[SpirvId, SpirvId, SpirvId]:
        bool_type = self.register_primitive_type("bool")
        left_vector = self.vector_component_type_and_count(left.type.base_type)
        right_vector = self.vector_component_type_and_count(right.type.base_type)
        if left_vector is None and right_vector is None:
            return bool_type, left, right

        component_type, component_count = left_vector or right_vector
        vector_operand_type = self.ensure_registered_type(
            left.type if left_vector is not None else right.type
        )

        if left_vector is not None and right_vector is None:
            if self.scalar_or_vector_component_type(right.type) == component_type:
                right = self.splat_scalar_to_vector(right, vector_operand_type)
        elif right_vector is not None and left_vector is None:
            if self.scalar_or_vector_component_type(left.type) == component_type:
                left = self.splat_scalar_to_vector(left, vector_operand_type)

        return self.register_vector_type(bool_type, component_count), left, right

    def align_binary_arithmetic_operands(
        self, result_type: SpirvId, left: SpirvId, right: SpirvId
    ) -> Tuple[SpirvId, SpirvId, SpirvId]:
        left_vector = self.vector_component_type_and_count(left.type.base_type)
        right_vector = self.vector_component_type_and_count(right.type.base_type)

        if left_vector is not None and right_vector is None:
            vector_type = self.ensure_registered_type(left.type)
            if self.scalar_or_vector_component_type(right.type) == left_vector[0]:
                return (
                    vector_type,
                    left,
                    self.splat_scalar_to_vector(right, vector_type),
                )
        if right_vector is not None and left_vector is None:
            vector_type = self.ensure_registered_type(right.type)
            if self.scalar_or_vector_component_type(left.type) == right_vector[0]:
                return (
                    vector_type,
                    self.splat_scalar_to_vector(left, vector_type),
                    right,
                )

        return result_type, left, right

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

        float_types = {"float", "double"}
        integer_types = {"int", "uint"}
        scalar_types = float_types | integer_types
        if source_type_name not in scalar_types or target_type_name not in scalar_types:
            return value

        if source_type_name in float_types and target_type_name in float_types:
            opcode = "OpFConvert"
        elif source_type_name in integer_types and target_type_name in float_types:
            opcode = "OpConvertUToF" if source_type_name == "uint" else "OpConvertSToF"
        elif source_type_name in float_types and target_type_name in integer_types:
            opcode = "OpConvertFToU" if target_type_name == "uint" else "OpConvertFToS"
        elif source_type_name in integer_types and target_type_name in integer_types:
            opcode = "OpBitcast"
        else:
            return value

        id_value = self.get_id()
        self.emit(f"%{id_value} = {opcode} %{target_type.id} %{value.id}")
        self.value_types[id_value] = target_type
        return SpirvId(id_value, target_type.type)

    def is_integer_type(self, spirv_type: SpirvType) -> bool:
        return spirv_type.base_type in {"int", "uint"}

    def is_unsigned_type(self, spirv_type: SpirvType) -> bool:
        return spirv_type.base_type == "uint"

    def unary_operation(
        self, op: str, result_type: Union[SpirvId, SpirvType], operand: SpirvId
    ) -> SpirvId:
        """Create a unary operation."""
        result_type = self.ensure_registered_type(result_type)
        id_value = self.get_id()

        if op == "+":
            spv_op = None
        elif op == "-":
            component_type = self.scalar_or_vector_component_type(result_type.type)
            spv_op = "OpSNegate" if component_type in {"int", "uint"} else "OpFNegate"
        else:
            spv_op = {
                "!": "OpLogicalNot",
                "~": "OpNot",
            }.get(op)

        if spv_op is None:
            return operand

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

    def is_select_result_type(self, result_type: SpirvId) -> bool:
        """Return whether OpSelect can directly produce this result type."""
        base_type = result_type.type.base_type
        if base_type in {"bool", "int", "uint", "float", "double"}:
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
        if function_name not in self.functions:
            # Handle built-in function
            return self.call_builtin_function(function_name, args)

        function_id = self.functions[function_name]
        return_type, param_types = self.function_signatures[function_name]
        args = [
            (
                self.convert_value_to_type(arg, param_types[index])
                if index < len(param_types)
                else arg
            )
            for index, arg in enumerate(args)
        ]

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
            "buffer_store",
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
        return {"buffer_load", "buffer_store"}

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
        offset_id: Optional[SpirvId] = None,
    ) -> SpirvId:
        image_operands = ""
        if offset_id is not None:
            image_operands = f" {self.image_offset_operand(offset_id)}"

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
        offsets, component_id = self.texture_gather_offsets_arguments(extra_args)
        if len(offsets) != 4:
            self.emit("; WARNING: textureGatherOffsets requires four offset operands")
            return self.register_constant(0.0, self.register_primitive_type("float"))

        if component_id is None:
            component_id = self.register_constant(0, int_type)

        result_type = self.resource_access_result_type(metadata)
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
        if (
            not metadata
            or metadata.get("kind") != "sampled_image"
            or int(metadata.get("depth", 0)) != 1
        ):
            self.emit(
                f"; WARNING: {function_name} requires a shadow sampled image operand"
            )
            return None

        return sampled_image_id, coord_id, depth_id, extra_args

    def sampled_texture_operands(
        self, function_name: str, args: List[SpirvId], extra_arg_count: int = 0
    ):
        coord_index = 1
        if len(args) > 1:
            sampler_metadata = self.resource_metadata_for_value(args[1])
            if sampler_metadata and sampler_metadata.get("kind") == "sampler":
                coord_index = 2

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
        if not metadata or metadata.get("kind") != "sampled_image":
            self.emit(f"; WARNING: {function_name} requires a sampled image operand")
            return None

        return sampled_image_id, coord_id, extra_args, metadata

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
        if metadata.get("readonly") or metadata.get("writeonly"):
            self.emit(f"; WARNING: {function_name} requires a read-write storage image")
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
        if metadata.get("multisampled"):
            if len(args) < 4:
                self.emit(f"; WARNING: {function_name} requires a sample operand")
                return self.register_constant(
                    0, self.register_primitive_type(component_type_name)
                )
            sample_id = args[2]
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

        value_id = self.convert_value_to_type(value_id, result_type)
        if comparator_id is not None:
            comparator_id = self.convert_value_to_type(comparator_id, result_type)

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

    def call_buffer_atomic_function(
        self, function_name: str, args: List[SpirvId]
    ) -> Optional[SpirvId]:
        if len(args) < 2:
            self.emit(f"; WARNING: {function_name} requires target and value operands")
            return self.register_constant(0, self.register_primitive_type("uint"))

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
        projected_coord = self.project_texture_coordinate(
            function_name, coord_id, metadata
        )
        result_type = self.resource_access_result_type(metadata)
        if "Offset" in function_name and metadata.get("dim") == "Cube":
            self.emit(
                f"; WARNING: {function_name} offsets are not valid for cube images"
            )
            return self.default_value_for_type(result_type)
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
        projected_coord = self.project_texture_coordinate(
            function_name, coord_id, metadata
        )
        result_type = self.register_primitive_type("float")
        if "Offset" in function_name and metadata.get("dim") == "Cube":
            self.emit(
                f"; WARNING: {function_name} offsets are not valid for cube images"
            )
            return self.default_value_for_type(result_type)
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

        if function_name == "buffer_load":
            if len(args) < 2:
                self.emit("; WARNING: buffer_load requires buffer and index operands")
                return self.register_constant(
                    0.0, self.register_primitive_type("float")
                )

            metadata = self.structured_buffer_metadata_for_pointer(args[0])
            if metadata is not None and metadata.get("writeonly"):
                self.emit("; WARNING: buffer_load requires a readable buffer")
                return self.register_constant(
                    0.0, self.register_primitive_type("float")
                )

            element_pointer = self.structured_buffer_element_pointer(args[0], args[1])
            if element_pointer is None:
                self.emit("; WARNING: buffer_load requires a StructuredBuffer operand")
                return self.register_constant(
                    0.0, self.register_primitive_type("float")
                )

            element_type = self.variable_value_types[element_pointer.id]
            return self.load_from_variable(element_pointer, element_type)

        if function_name == "buffer_store":
            if len(args) < 3:
                self.emit(
                    "; WARNING: buffer_store requires buffer, index, and value operands"
                )
                return None

            metadata = self.structured_buffer_metadata_for_pointer(args[0])
            if metadata is None:
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

            image_operands = ""
            if metadata.get("multisampled"):
                if len(args) < 3:
                    self.emit("; WARNING: imageLoad requires a sample operand")
                    return self.register_constant(
                        0.0, self.register_primitive_type("float")
                    )
                image_operands = f" Sample %{args[2].id}"

            id_value = self.get_id()
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

            image_operands = ""
            if metadata.get("multisampled"):
                if len(args) < 4:
                    self.emit("; WARNING: imageStore requires a sample operand")
                    return None
                sample_id, texel_id = args[2], args[3]
                image_operands = f" Sample %{sample_id.id}"

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
            if compare_args is None:
                return self.register_constant(
                    0.0, self.register_primitive_type("float")
                )

            sampled_image_id, coord_id, depth_id, extra_args = compare_args
            float_type = self.register_primitive_type("float")
            result_type = self.register_vector_type(float_type, 4)
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
            if function_name == "textureGather":
                component_id = (
                    extra_args[0] if extra_args else self.register_constant(0, int_type)
                )
                offset_id = None
            elif function_name == "textureGatherOffsets":
                return self.emit_texture_gather_offsets(
                    sampled_image_id,
                    coord_id,
                    extra_args,
                    metadata,
                    int_type,
                )
            else:
                offset_id = extra_args[0]
                component_id = (
                    extra_args[1]
                    if len(extra_args) >= 2
                    else self.register_constant(0, int_type)
                )

            result_type = self.resource_access_result_type(metadata)
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

            if metadata.get("dim") == "Cube":
                self.emit(f"; WARNING: {function_name} is not valid for cube images")
                result_type = self.resource_access_result_type(metadata)
                return self.default_value_for_type(result_type)

            image_id = self.extract_image_from_sampled_image(sampled_image_id, metadata)
            if image_id is None:
                return self.register_constant(
                    0.0, self.register_primitive_type("float")
                )

            if metadata.get("multisampled") and offset_id is not None:
                self.emit(
                    "; WARNING: texelFetchOffset is not valid for multisample images"
                )
                result_type = self.resource_access_result_type(metadata)
                return self.default_value_for_type(result_type)

            result_type = self.resource_access_result_type(metadata)
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
                return self.register_constant(0, self.register_primitive_type("int"))

            image_id = self.image_operand_for_query(resource_id, metadata)
            if image_id is None:
                return self.register_constant(0, self.register_primitive_type("int"))

            result_type = self.resource_query_size_result_type(metadata)
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

            sampled_image_id = args[0]
            metadata = self.resource_metadata_for_value(sampled_image_id)
            if not metadata or metadata.get("kind") != "sampled_image":
                self.emit(
                    "; WARNING: textureQueryLevels requires a sampled image operand"
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
                return self.register_constant(
                    0.0, self.register_primitive_type("float")
                )

            sampled_image_id = args[0]
            coord_id = args[1]
            if len(args) >= 3:
                sampler_metadata = self.resource_metadata_for_value(args[1])
                if sampler_metadata and sampler_metadata.get("kind") == "sampler":
                    coord_id = args[2]

            metadata = self.resource_metadata_for_value(sampled_image_id)
            if not metadata or metadata.get("kind") != "sampled_image":
                self.emit("; WARNING: textureQueryLod requires a sampled image operand")
                return self.register_constant(
                    0.0, self.register_primitive_type("float")
                )

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
            if function_name == "textureLod":
                image_operands = f"Lod %{extra_args[0].id}"
            elif function_name == "textureLodOffset":
                image_operands = self.image_operands(
                    f"Lod %{extra_args[0].id}",
                    self.image_offset_operand(extra_args[1]),
                )
            elif function_name == "textureGrad":
                image_operands = f"Grad %{extra_args[0].id} %{extra_args[1].id}"
            else:
                image_operands = self.image_operands(
                    f"Grad %{extra_args[0].id} %{extra_args[1].id}",
                    self.image_offset_operand(extra_args[2]),
                )

            result_type = self.resource_access_result_type(metadata)
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

        resource_array_params = self.function_resource_array_params.get(
            function_name, set()
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

    def process_dispatch_mesh_operation(self, expr: MeshOpNode) -> Optional[SpirvId]:
        """Lower task-shader DispatchMesh to the SPIR-V mesh-task terminator."""
        operation = expr.operation
        if self.current_execution_model != "TaskEXT":
            return self.represented_ir_diagnostic_default_value(
                "mesh shader", operation
            )

        arguments = getattr(expr, "arguments", []) or []
        if len(arguments) != 3:
            self.emit(
                "; WARNING: SPIR-V mesh DispatchMesh requires exactly 3 arguments"
            )
            return self.register_constant(0, self.register_primitive_type("uint"))

        uint_type = self.register_primitive_type("uint")
        group_counts = []
        for argument in arguments:
            group_count = self.process_expression(argument)
            if group_count is None:
                self.emit(
                    "; WARNING: SPIR-V mesh DispatchMesh requires group-count operands"
                )
                return self.register_constant(0, uint_type)
            group_counts.append(self.convert_value_to_type(group_count, uint_type))

        self.require_capability("MeshShadingEXT")
        self.require_extension("SPV_EXT_mesh_shader")
        self.emit(
            "OpEmitMeshTasksEXT "
            + " ".join(f"%{group_count.id}" for group_count in group_counts)
        )
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
        converted_value = self.convert_scalar_to_type(value, component_type)
        source_type = self.normalize_primitive_name(converted_value.type.base_type)
        target_type = self.normalize_primitive_name(component_type.type.base_type)
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

    def call_synchronization_function(
        self, function_name: str, args: List[SpirvId]
    ) -> Optional[SpirvId]:
        if args or function_name not in {
            "allMemoryBarrier",
            "barrier",
            "groupMemoryBarrier",
            "workgroupBarrier",
            "memoryBarrier",
            "memoryBarrierBuffer",
            "memoryBarrierImage",
            "memoryBarrierShared",
        }:
            return None

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

    def call_builtin_function(
        self, function_name: str, args: List[SpirvId]
    ) -> Optional[SpirvId]:
        """Call a built-in function."""
        function_name = {"frac": "fract"}.get(function_name, function_name)
        synchronization_call = self.call_synchronization_function(function_name, args)
        if synchronization_call is not None:
            return synchronization_call

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

        if (
            function_name in {"min", "max"}
            and len(args) == 2
            or function_name == "clamp"
            and len(args) == 3
        ):
            min_max_clamp = self.call_min_max_clamp_function(function_name, args)
            if min_max_clamp is not None:
                return min_max_clamp

        if function_name == "mix" and len(args) == 3:
            numeric_mix = self.call_numeric_mix_function(args)
            if numeric_mix is not None:
                return numeric_mix

        vector_info = self.vector_component_type_and_count(function_name)
        if vector_info:
            component_type_name, component_count = vector_info
            component_type = self.register_primitive_type(component_type_name)
            vector_type = self.register_vector_type(component_type, component_count)

            # If no arguments are provided, construct a default vector
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

                # Create default vector components
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

        # Matrix constructors
        elif re.fullmatch(r"(d)?mat([234])(?:x([234]))?", function_name):
            match = re.fullmatch(r"(d)?mat([234])(?:x([234]))?", function_name)
            is_double, cols, rows = match.groups()
            cols = int(cols)
            rows = int(rows or cols)

            component_type = self.register_primitive_type(
                "double" if is_double else "float"
            )
            vector_type = self.register_vector_type(component_type, rows)
            matrix_type = self.register_matrix_type(vector_type, cols)

            # If no arguments provided, create identity matrix
            if not args:
                # Create identity matrix: 1's on diagonal, 0's elsewhere
                zero_value = self.register_constant(0.0, component_type)
                one_value = self.register_constant(1.0, component_type)

                # Create column vectors
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
            component_type = self.scalar_or_vector_component_type(args[0].type)
            if component_type not in {"float", "double"}:
                component_type = "float"
            result_type = self.register_primitive_type(component_type)

            # Generate a direct OpDot instruction
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
            # Determine result type based on the function name
            float_type = self.primitive_types["float"]

            # Default to float if we can't determine or no args provided
            result_type = float_type.type

            # Try to infer result type from arguments if available
            if args:
                if function_name in [
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
                ]:
                    # These functions return the same type as their first argument
                    result_type = args[0].type
                elif function_name in ["length", "distance"]:
                    # These functions return a float regardless of input
                    result_type = float_type.type
                elif function_name in ["normalize", "reflect", "refract"]:
                    # These functions return the same vector type as their first argument
                    result_type = args[0].type
                elif function_name in ["cross"]:
                    # cross product returns a vec3
                    vector_type = self.register_vector_type(float_type, 3)
                    result_type = vector_type.type

            id_value = self.get_id()
            arg_list = " ".join([f"%{arg.id}" for arg in args])

            # Use a proper mapping for GLSL.std.450 extended instructions
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
                "length": "Length",
                "distance": "Distance",
                "cross": "Cross",
                "normalize": "Normalize",
                "reflect": "Reflect",
                "refract": "Refract",
            }

            glsl_function = glsl_std450_map.get(
                function_name, function_name[0].upper() + function_name[1:]
            )

            # Find the result type ID
            result_type_id = None
            for id_obj in (
                [self.primitive_types.get(result_type.base_type)]
                + list(self.vector_types.values())
                + list(self.matrix_types.values())
            ):
                if id_obj and id_obj.type.base_type == result_type.base_type:
                    result_type_id = id_obj.id
                    break

            if result_type_id is None:
                result_type_id = float_type.id

            self.emit(
                f"%{id_value} = OpExtInst %{result_type_id} %{self.glsl_std450_id} {glsl_function} {arg_list}"
            )

            return SpirvId(id_value, result_type)

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

    def create_return(self):
        """Create a return instruction."""
        self.emit("OpReturn")

    def create_return_value(self, value: SpirvId):
        """Create a return value instruction."""
        if self.current_return_type is not None:
            value = self.convert_value_to_type(value, self.current_return_type)
        self.emit(f"OpReturnValue %{value.id}")

    def current_block_has_terminator(self) -> bool:
        """Return whether the current block already ends in a terminator."""
        for line in reversed(self.code_lines):
            stripped = line.strip()
            if not stripped or stripped.startswith(";"):
                continue
            if re.match(r"%\d+ = OpLabel$", stripped):
                return False
            return stripped.startswith(
                ("OpBranch", "OpReturn", "OpKill", "OpEmitMeshTasksEXT")
            )
        return False

    def normalize_primitive_name(self, type_name: str) -> str:
        aliases = {
            "f32": "float",
            "f64": "double",
            "i32": "int",
            "u32": "uint",
        }
        return aliases.get(str(type_name), str(type_name))

    def normalize_generic_vector_type(self, type_str: str) -> str:
        compact = re.sub(r"\s+", "", str(type_str))
        match = re.fullmatch(r"vec([234])<([^>]+)>", compact)
        if not match:
            return compact

        size, element_type = match.groups()
        element_type = self.normalize_primitive_name(element_type)
        prefixes = {
            "float": "vec",
            "double": "dvec",
            "int": "ivec",
            "uint": "uvec",
            "bool": "bvec",
        }
        return f"{prefixes.get(element_type, 'vec')}{size}"

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

    def resource_type_info(self, type_str: str):
        sampler_info = {
            "sampler": {"kind": "sampler"},
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

        image_match = re.fullmatch(r"([iu]?image)(2D|3D|Cube)(MS)?(Array)?", type_str)
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
        match = re.fullmatch(r"(StructuredBuffer|RWStructuredBuffer)<(.+)>", type_str)
        if not match:
            return None

        buffer_kind, element_type_name = match.groups()
        return {
            "kind": "structured_buffer",
            "buffer_kind": buffer_kind,
            "element_type_name": element_type_name,
            "readonly": buffer_kind == "StructuredBuffer",
        }

    def is_structured_buffer_type_name(self, type_str: str) -> bool:
        return self.structured_buffer_type_info(type_str) is not None

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
        if hasattr(type_name, "name") or hasattr(type_name, "element_type"):
            type_str = self.convert_type_node_to_string(type_name)
        else:
            type_str = str(type_name)

        type_str = self.normalize_generic_vector_type(type_str)
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

        return self.map_crossgl_type(type_name)

    def storage_buffer_parameter_type_name(self, param) -> Optional[str]:
        param_type = getattr(param, "param_type", getattr(param, "vtype", None))
        if param_type is None:
            return None

        type_name = self.type_name_from_value(param_type)
        if self.is_structured_buffer_type_name(type_name):
            return type_name
        if self.has_attribute(param, "glsl_buffer_block"):
            return type_name
        return None

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
        if hasattr(type_name, "name") or hasattr(type_name, "element_type"):
            type_str = self.convert_type_node_to_string(type_name)
        else:
            type_str = str(type_name)

        type_str = self.normalize_generic_vector_type(type_str)

        array_type = self.split_outer_array_type(type_str)
        if array_type is not None:
            element_type_name, size = array_type
            element_type = self.map_crossgl_type(element_type_name)
            return self.register_array_type(element_type, size)

        primitive_type = self.normalize_primitive_name(type_str)
        if primitive_type in {"float", "double", "int", "uint", "bool", "void"}:
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

        registered_type = self.find_registered_type_by_base(type_str)
        if registered_type:
            return registered_type

        if self.is_ray_query_type_name(type_str):
            return self.register_ray_query_type(type_str)

        if self.is_resource_type_name(type_str):
            return self.register_resource_type(type_str)

        if type_str in self.struct_types:
            # Struct type (reference to existing struct)
            return self.struct_types[type_str]
        else:
            # If type is unknown, return a default float type
            self.emit(f"; WARNING: Unknown type {type_str}, using float as default")
            return self.register_primitive_type("float")

    def convert_type_node_to_string(self, type_node) -> str:
        """Convert new AST TypeNode to string representation."""
        if type_node.__class__.__name__ == "ArrayType":
            element_type = self.convert_type_node_to_string(type_node.element_type)
            size = self.format_array_size(type_node.size)
            return (
                f"{element_type}[{size}]" if size is not None else f"{element_type}[]"
            )
        if hasattr(type_node, "name"):
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

    def process_crossgl_struct(self, struct_node: StructNode) -> SpirvId:
        """Process a CrossGL struct definition."""
        members = []

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
                member_type = self.map_crossgl_type(
                    f"{element_type}[{size}]"
                    if size is not None
                    else f"{element_type}[]"
                )
            else:
                member_type_source = getattr(
                    member,
                    "member_type",
                    getattr(member, "var_type", getattr(member, "vtype", None)),
                )
                if member_type_source is not None:
                    member_type = self.map_crossgl_type(member_type_source)

            if member_type:
                members.append((member_type, member_name))

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
        metadata = self.structured_buffer_type_info(type_name)
        if metadata is None:
            raise ValueError(f"Invalid SPIR-V structured buffer type {type_name}")
        memory_flags = self.storage_buffer_memory_flags(
            node, default_readonly=metadata.get("readonly", False)
        )

        element_type = self.map_crossgl_type(metadata["element_type_name"])
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

        var_id = self.create_variable(block_type, "Uniform", node.name)
        descriptor_set, binding = self.resource_descriptor_slot(node)
        self.decorations.append(
            f"OpDecorate %{var_id.id} DescriptorSet {descriptor_set}"
        )
        self.decorations.append(f"OpDecorate %{var_id.id} Binding {binding}")

        buffer_metadata = {
            **metadata,
            **memory_flags,
            "element_type": element_type,
            "runtime_array_type": runtime_array_type,
            "block_type": block_type,
            "member_index": 0,
        }
        self.global_variables[node.name] = var_id
        self.structured_buffer_metadata[var_id.id] = buffer_metadata
        self.structured_buffer_metadata[block_type.id] = buffer_metadata
        return var_id

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
        is_named_block = base_type_name in self.struct_types
        outer_array = self.split_outer_array_type(type_name)
        if is_named_block and outer_array is not None and outer_array[1] is None:
            self.require_capability("RuntimeDescriptorArray")
            self.require_extension("SPV_EXT_descriptor_indexing")

        if is_named_block:
            value_type = self.map_crossgl_type(type_name)
            block_type = self.struct_types[base_type_name]
            if layout != "std430":
                value_type = self.storage_layout_type(value_type, layout)
                block_type = self.storage_layout_type(block_type, layout)
            block_members = self.current_struct_members.get(base_type_name, [])
            block_members = self.current_struct_members.get(
                block_type.type.base_type, block_members
            )
            variable_member_name = None
            variable_member_type = None
        else:
            variable_member_name = node.name
            variable_member_type = self.map_crossgl_type(
                getattr(node, "var_type", getattr(node, "vtype", "float"))
            )
            if layout != "std430":
                variable_member_type = self.storage_layout_type(
                    variable_member_type, layout
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
        }
        self.structured_buffer_metadata[var_id.id] = metadata
        self.structured_buffer_metadata[block_type.id] = metadata
        if variable_member_type is not None:
            self.structured_buffer_metadata[variable_member_type.id] = metadata

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
        if metadata is None or metadata.get("kind") != "storage_image":
            return

        metadata = self.metadata_with_resource_memory_qualifiers(metadata, node)
        self.resource_type_metadata[var_id.id] = metadata
        self.decorate_resource_variable_memory_qualifiers(var_id, metadata)

    def decorate_cbuffer_type(
        self, cbuffer_type: SpirvId, members: List[Tuple[SpirvId, str]]
    ):
        self.decorations.append(f"OpDecorate %{cbuffer_type.id} Block")

        offset = 0
        for member_index, (member_type, _) in enumerate(members):
            offset = self.align_to(offset, self.uniform_layout_alignment(member_type))
            self.decorations.append(
                f"OpMemberDecorate %{cbuffer_type.id} {member_index} Offset {offset}"
            )
            if self.uniform_layout_contains_matrix(member_type):
                self.decorations.append(
                    f"OpMemberDecorate %{cbuffer_type.id} {member_index} ColMajor"
                )
                self.decorations.append(
                    f"OpMemberDecorate %{cbuffer_type.id} {member_index} MatrixStride 16"
                )
            self.decorate_uniform_array_strides(member_type)
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

    def decorate_uniform_array_strides(self, type_id: SpirvId):
        array_info = self.array_type_info_from_type(type_id)
        if array_info is None:
            return

        element_type, _ = array_info
        stride = self.uniform_array_stride(element_type)
        self.decorations.append(f"OpDecorate %{type_id.id} ArrayStride {stride}")
        self.decorate_uniform_array_strides(element_type)

    def storage_layout_type(self, type_id: SpirvId, layout: str) -> SpirvId:
        if layout == "std430":
            return type_id

        array_info = self.array_type_info_from_type(type_id)
        if array_info is not None:
            element_type, size = array_info
            return self.register_layout_array_type(
                self.storage_layout_type(element_type, layout), size, layout
            )

        struct_members = self.current_struct_members.get(type_id.type.base_type)
        if struct_members is not None:
            cloned_members = [
                (self.storage_layout_type(member_type, layout), member_name)
                for member_type, member_name in struct_members
            ]
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

    def process_function_node(self, function_node, stage=None):
        """Process a CrossGL function definition."""
        return_type = self.map_crossgl_type(function_node.return_type)
        previous_return_type = self.current_return_type
        previous_stage = self.current_stage
        self.current_return_type = return_type
        if stage is not None:
            self.current_stage = stage

        param_types = []
        param_value_types = []
        resource_array_param_indices = set()
        param_type_hints = self.function_resource_array_type_hints.get(
            function_node.name, {}
        )
        storage_image_pointer_params = self.function_storage_image_pointer_params.get(
            function_node.name, set()
        )
        for param in getattr(
            function_node, "parameters", getattr(function_node, "params", [])
        ):
            param_type_source = getattr(
                param, "param_type", getattr(param, "vtype", None)
            )
            param_name = getattr(param, "name", None)
            if param_name in param_type_hints:
                param_type_source = param_type_hints[param_name]
            if param_type_source is not None:
                param_type = self.map_resource_type_with_format(
                    param_type_source, param
                )
            else:
                param_type = self.map_crossgl_type("float")

            param_value_types.append(param_type)
            param_resource_metadata = self.resource_type_metadata.get(param_type.id)
            is_storage_image_param = (
                param_resource_metadata is not None
                and param_resource_metadata.get("kind") == "storage_image"
                and param_name in storage_image_pointer_params
            )
            if self.is_resource_array_type(param_type) or is_storage_image_param:
                resource_array_param_indices.add(len(param_types))
                param_type = self.register_pointer_type(param_type, "UniformConstant")

            param_types.append(param_type)

        function_id = self.create_function(function_node.name, return_type, param_types)
        self.function_resource_array_params[function_node.name] = (
            resource_array_param_indices
        )

        for i, param in enumerate(
            getattr(function_node, "parameters", getattr(function_node, "params", []))
        ):
            if hasattr(param, "name"):
                param_name = param.name
            else:
                param_name = f"param{i}"

            param_id = self.create_function_parameter(param_types[i], param_name)
            self.local_variables[param_name] = param_id
            if i in resource_array_param_indices:
                self.variable_value_types[param_id.id] = param_value_types[i]
            self.register_declared_resource_metadata(
                param, param_id, param_value_types[i]
            )

        self.begin_block()

        previous_execution_model = self.current_execution_model
        previous_function_id = self.current_function_id
        self.current_function_id = function_id.id
        if self.current_execution_model is None:
            execution_models = self.function_execution_models.get(
                function_node.name, set()
            )
            if "GLCompute" in execution_models:
                self.current_execution_model = "GLCompute"
            elif len(execution_models) == 1:
                self.current_execution_model = next(iter(execution_models))

        self.process_statements(function_node.body)

        if (
            self.convert_type_node_to_string(function_node.return_type) == "void"
            and not self.current_block_has_terminator()
        ):
            self.create_return()

        self.end_function()

        self.current_execution_model = previous_execution_model
        self.current_function_id = previous_function_id
        self.current_stage = previous_stage
        self.current_return_type = previous_return_type
        self.local_variables.clear()
        self.precise_local_variables.clear()
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
            self.process_expression(stmt)  # Just evaluate and discard result
        elif isinstance(stmt, (UnaryOpNode, BinaryOpNode)):
            self.process_expression(stmt)
        elif hasattr(stmt, "expression"):
            expression = stmt.expression
            if isinstance(expression, AssignmentNode):
                self.process_assignment(expression)
            else:
                self.process_expression(expression)

    def process_variable_declaration(self, node: VariableNode):
        """Process a local CrossGL variable declaration."""
        var_type_source = getattr(node, "var_type", getattr(node, "vtype", "float"))
        var_type = self.map_resource_type_with_format(var_type_source, node)
        if self.type_contains_runtime_array(var_type):
            self.emit(
                f"; WARNING: local variable {node.name} has a runtime-array "
                "aggregate type that cannot be materialized in SPIR-V"
            )
            return

        var_id = self.create_variable(var_type, "Function", node.name)
        self.local_variables[node.name] = var_id
        is_precise = self.has_attribute(node, "precise")
        if is_precise:
            self.precise_local_variables.add(node.name)

        initial_value = getattr(node, "initial_value", None)
        if initial_value is not None:
            if isinstance(initial_value, ArrayLiteralNode):
                rhs_value = self.process_array_literal(initial_value, var_type)
            else:
                rhs_value = self.process_expression_with_precision(
                    initial_value, is_precise
                )
            if rhs_value is not None:
                self.store_to_variable(var_id, rhs_value)

    def process_global_variable_declaration(
        self, node: VariableNode, default_storage_class: str = "Private"
    ) -> SpirvId:
        """Process a module-scope CrossGL variable declaration."""
        var_type_source = getattr(node, "var_type", getattr(node, "vtype", "float"))
        var_type_name = self.type_name_from_value(var_type_source)
        if self.is_glsl_buffer_block_node(node):
            return self.process_glsl_buffer_block_declaration(node, var_type_name)

        if self.is_structured_buffer_type_name(var_type_name):
            return self.process_structured_buffer_declaration(node, var_type_name)

        var_type = self.map_resource_type_with_format(var_type_source, node)
        storage_class = self.infer_global_storage_class(
            node, default_storage_class, var_type_name
        )

        initializer = None
        initial_value = getattr(node, "initial_value", None)
        if storage_class == "Private" and initial_value is not None:
            initializer = self.process_constant_expression(initial_value, var_type)

        if storage_class == "Input":
            location = self.global_interface_location(node, "Input")
            var_id = self.register_input(node.name, var_type, location, 0)
            self.decorate_global_interface_variable(node, var_id)
        elif storage_class == "Output":
            location = self.global_interface_location(node, "Output")
            var_id = self.register_output(node.name, var_type, location, 0)
            self.decorate_global_interface_variable(node, var_id)
        else:
            var_id = self.create_variable(
                var_type, storage_class, node.name, initializer
            )
            if storage_class == "UniformConstant":
                descriptor_set, binding = self.resource_descriptor_slot(node)
                self.decorations.append(
                    f"OpDecorate %{var_id.id} DescriptorSet {descriptor_set}"
                )
                self.decorations.append(f"OpDecorate %{var_id.id} Binding {binding}")
                self.register_declared_resource_metadata(node, var_id, var_type)

        self.global_variables[node.name] = var_id
        if self.has_attribute(node, "precise"):
            self.precise_global_variables.add(node.name)
        return var_id

    def resource_descriptor_slot(self, node: VariableNode) -> Tuple[int, int]:
        descriptor_set = self.resource_descriptor_set(node)
        explicit_binding = self.explicit_interface_integer_attribute(node, "binding")

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
            explicit_binding = self.explicit_interface_integer_attribute(
                node, "binding"
            )
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
            if self.is_structured_buffer_type_name(
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

    def is_uniform_constant_resource_node(self, node: VariableNode) -> bool:
        type_source = getattr(node, "var_type", getattr(node, "vtype", "float"))
        type_name = self.type_name_from_value(type_source)
        base_type_name = self.array_base_type_name(type_name)
        return self.is_resource_type_name(base_type_name)

    def global_interface_location(self, node: VariableNode, storage_class: str) -> int:
        if storage_class == "Input":
            counter_name = "next_input_location"
            used_slots = self.used_input_locations
        else:
            counter_name = "next_output_location"
            used_slots = self.used_output_locations

        explicit_location = self.explicit_location_attribute(node)
        if explicit_location is not None:
            slot_keys = self.interface_slot_keys(node, storage_class, explicit_location)
            if used_slots & slot_keys:
                raise ValueError(
                    f"Duplicate SPIR-V {storage_class.lower()} location "
                    f"{explicit_location}"
                )
            used_slots.update(slot_keys)
            return explicit_location

        location = getattr(self, counter_name)
        slot_keys = self.interface_slot_keys(node, storage_class, location)
        while used_slots & slot_keys:
            location += 1
            slot_keys = self.interface_slot_keys(node, storage_class, location)
        used_slots.update(slot_keys)
        setattr(self, counter_name, location + 1)
        return location

    def explicit_location_attribute(self, node: VariableNode) -> Optional[int]:
        return self.explicit_interface_integer_attribute(node, "location")

    def explicit_component_attribute(self, node: VariableNode) -> Optional[int]:
        component = self.explicit_interface_integer_attribute(node, "component")
        if component is not None and component > 3:
            raise ValueError(f"SPIR-V component must be in 0..3: {component}")
        return component

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

    def validate_interface_interpolation_attributes(self, node: VariableNode):
        for first, second in (("flat", "noperspective"), ("centroid", "sample")):
            if self.has_attribute(node, first) and self.has_attribute(node, second):
                raise ValueError(
                    "SPIR-V interpolation attributes "
                    f"@{first} and @{second} cannot be combined"
                )

    def interface_slot_keys(
        self, node: VariableNode, storage_class: str, location: int
    ) -> set:
        component = self.explicit_component_attribute(node)
        component_start = component if component is not None else 0
        component_width = self.interface_component_width(node)
        if component_start + component_width > 4:
            raise ValueError(
                f"SPIR-V component range overflows location {location}: "
                f"{component_start}..{component_start + component_width - 1}"
            )

        index = self.explicit_interface_integer_attribute(node, "index") or 0
        return {
            (location, index, component)
            for component in range(component_start, component_start + component_width)
        }

    def interface_component_width(self, node: VariableNode) -> int:
        type_source = getattr(node, "var_type", getattr(node, "vtype", "float"))
        type_name = self.type_name_from_value(type_source)
        vector_info = self.vector_component_type_and_count(type_name)
        if vector_info is not None:
            _, component_count = vector_info
            return component_count

        if self.normalize_primitive_name(type_name) in {
            "float",
            "double",
            "int",
            "uint",
            "bool",
        }:
            return 1

        return 4

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

        if attribute_names & {"input", "in"} or qualifiers & {"input", "in"}:
            return "Input"
        if attribute_names & {"output", "out"} or qualifiers & {"output", "out"}:
            return "Output"
        if type_name:
            base_type_name, _ = parse_array_type(type_name)
            if self.is_resource_type_name(base_type_name):
                return "UniformConstant"
        return default_storage_class

    def type_name_from_value(self, type_value) -> str:
        if hasattr(type_value, "name") or hasattr(type_value, "element_type"):
            return self.convert_type_node_to_string(type_value)
        return str(type_value)

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

    def expression_name(self, expr):
        if isinstance(expr, str):
            return expr
        if hasattr(expr, "name") and isinstance(expr.name, str):
            return expr.name
        if isinstance(expr, ArrayAccessNode):
            array_expr = getattr(expr, "array", getattr(expr, "array_expr", None))
            return self.expression_name(array_expr)
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

    def buffer_operation_access_requirement(self, func_name):
        if func_name == "buffer_load":
            return "read"
        if func_name == "buffer_store":
            return "write"
        if func_name in self.buffer_atomic_function_names():
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

    def collect_function_storage_buffer_access_requirements_for_ast(self, ast):
        functions = self.collect_ast_functions(ast)
        function_parameter_names = collect_function_parameter_names(functions)
        parameter_sets = {
            func_name: set(param_names)
            for func_name, param_names in function_parameter_names.items()
        }
        requirements = {
            getattr(func, "name", None): {}
            for func in functions
            if getattr(func, "name", None)
        }
        storage_buffer_parameter_sets = {
            getattr(func, "name", None): self.function_storage_buffer_parameters(func)
            for func in functions
            if getattr(func, "name", None)
        }
        storage_buffer_parameter_indices = {
            getattr(func, "name", None): self.function_storage_buffer_parameter_indices(
                func
            )
            for func in functions
            if getattr(func, "name", None)
        }

        for func in functions:
            func_name = getattr(func, "name", None)
            if not func_name:
                continue

            self.scan_storage_buffer_requirement_node(
                getattr(func, "body", []),
                func_name,
                storage_buffer_parameter_sets.get(func_name, set()),
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

                parameter_set = parameter_sets.get(func_name, set())
                if not parameter_set:
                    continue

                for node in self.walk_ast_nodes(getattr(func, "body", [])):
                    if not isinstance(node, FunctionCallNode):
                        continue

                    callee_name = self.function_call_name(node)
                    callee_requirements = requirements.get(callee_name)
                    if not callee_requirements:
                        continue

                    callee_parameters = function_parameter_names.get(callee_name, [])
                    args = getattr(node, "arguments", getattr(node, "args", []))
                    for callee_param, required_access in callee_requirements.items():
                        try:
                            index = callee_parameters.index(callee_param)
                        except ValueError:
                            continue
                        if index >= len(args):
                            continue

                        target_name = self.expression_name(args[index])
                        if target_name not in parameter_set:
                            continue

                        current = requirements[func_name].get(target_name)
                        merged = self.merge_resource_access_requirement(
                            current, required_access
                        )
                        if merged != current:
                            requirements[func_name][target_name] = merged
                            changed = True

        return {name: reqs for name, reqs in requirements.items() if reqs}

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

        pointer = self.local_variables.get(name) or self.global_variables.get(name)
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

        pointer = self.local_variables.get(name) or self.global_variables.get(name)
        metadata = (
            self.storage_buffer_access_metadata_for_pointer(pointer)
            if pointer is not None
            else None
        )
        if metadata is None:
            return None
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
        callee_requirements = self.function_image_access_requirements.get(func_name)
        if not callee_requirements:
            return True

        param_names = self.function_parameter_names.get(func_name, [])
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
        self, func_name, args
    ) -> bool:
        callee_requirements = self.function_storage_buffer_access_requirements.get(
            func_name
        )
        if not callee_requirements:
            return True

        param_names = self.function_parameter_names.get(func_name, [])
        for index, param_name in enumerate(param_names):
            required_access = callee_requirements.get(param_name)
            if required_access is None or index >= len(args):
                continue

            actual_access = self.storage_buffer_access_for_expression(args[index])
            if image_access_satisfies_requirement(required_access, actual_access):
                continue

            required_label = image_access_requirement_label(required_access)
            actual_label = image_access_diagnostic_name(actual_access)
            self.emit(
                f"; WARNING: function call '{func_name}' requires {required_label} "
                "storage buffer access for argument "
                f"{self.expression_debug_name(args[index])} passed to parameter "
                f"{param_name}: got {actual_label}"
            )
            return False

        return True

    def collect_inline_storage_buffer_functions(self, ast):
        return {
            func.name: func
            for func in self.collect_ast_functions(ast)
            if getattr(func, "name", None)
            and self.function_has_storage_buffer_parameters(func)
        }

    def default_value_for_function(self, function_node) -> Optional[SpirvId]:
        return_type = self.map_crossgl_type(function_node.return_type)
        if return_type.type.base_type == "void":
            return None
        return self.default_value_for_type(return_type)

    def inline_storage_buffer_function_call(self, function_node, call_args):
        func_name = getattr(function_node, "name", "unknown")
        if not self.validate_function_storage_buffer_access_arguments(
            func_name, call_args
        ):
            return self.default_value_for_function(function_node)

        parameters = getattr(
            function_node, "parameters", getattr(function_node, "params", [])
        )
        if len(call_args) < len(parameters):
            self.emit(
                f"; WARNING: function call '{func_name}' requires "
                f"{len(parameters)} arguments"
            )
            return self.default_value_for_function(function_node)

        previous_locals = self.local_variables.copy()
        previous_precise_locals = set(self.precise_local_variables)
        previous_return_type = self.current_return_type
        self.current_return_type = self.map_crossgl_type(function_node.return_type)

        try:
            for index, param in enumerate(parameters):
                param_name = getattr(param, "name", f"param{index}")
                if self.storage_buffer_parameter_type_name(param) is not None:
                    pointer_arg = self.variable_pointer_from_expression(
                        call_args[index]
                    )
                    if pointer_arg is None:
                        self.emit(
                            f"; WARNING: function call '{func_name}' requires a "
                            f"storage buffer argument for parameter {param_name}"
                        )
                        return self.default_value_for_function(function_node)
                    self.local_variables[param_name] = pointer_arg
                    continue

                arg_value = self.process_call_argument(
                    func_name, call_args[index], index
                )
                if arg_value is None:
                    self.emit(f"; WARNING: Failed to evaluate argument for {func_name}")
                    return self.default_value_for_function(function_node)
                self.local_variables[param_name] = arg_value

            result = self.inline_function_body(function_node)
            if result is not None:
                return result
            return self.default_value_for_function(function_node)
        finally:
            self.local_variables = previous_locals
            self.precise_local_variables = previous_precise_locals
            self.current_return_type = previous_return_type

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

    def collect_storage_image_pointer_parameters(self, ast):
        functions = {
            getattr(func, "name", None): func
            for func in self.collect_ast_functions(ast)
        }
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

    def collect_resource_array_parameter_type_hints(self, ast):
        functions = {
            getattr(func, "name", None): func
            for func in self.collect_ast_functions(ast)
        }
        functions = {name: func for name, func in functions.items() if name}

        global_nodes = list(getattr(ast, "global_variables", []) or [])
        for stage in (getattr(ast, "stages", None) or {}).values():
            global_nodes.extend(getattr(stage, "local_variables", []) or [])

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

        try:
            return element_type, int(size_text)
        except ValueError:
            return element_type, None

    def array_base_type_name(self, type_name: str):
        if not type_name or "[" not in type_name:
            return type_name
        return type_name[: type_name.find("[")]

    def get_variable_value(self, variable_id: SpirvId) -> SpirvId:
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

    def structured_buffer_element_pointer(
        self, buffer_pointer: SpirvId, index_id: SpirvId
    ) -> Optional[SpirvId]:
        metadata = self.structured_buffer_metadata_for_pointer(buffer_pointer)
        if metadata is None:
            return None

        pointee_type = self.variable_value_types.get(buffer_pointer.id)
        descriptor_array = self.array_type_info_from_type(pointee_type)
        block_type = metadata.get("block_type")
        if (
            descriptor_array is not None
            and block_type is not None
            and descriptor_array[0] is not None
            and descriptor_array[0].id == block_type.id
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
            access, _ = self.create_array_element_access(expr.array, index)
            return access
        elif isinstance(expr, MemberAccessNode):
            base_pointer = self.variable_pointer_from_expression(expr.object)
            if base_pointer is None:
                return None

            return self.create_member_access_pointer(base_pointer, expr.member)
        else:
            return None

        return (
            self.local_variables.get(name)
            or self.global_variables.get(name)
            or self.cbuffer_member_pointer(name)
            or self.ensure_compute_builtin(name)
        )

    def array_element_type_from_type(self, array_type: Optional[SpirvId]):
        if array_type is None:
            return None

        array_info = self.array_type_info_from_type(array_type)
        if array_info is not None:
            return array_info[0]

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
        if primitive_name in {"int", "uint"}:
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
        if isinstance(expr, LiteralNode):
            value = expr.value
        elif isinstance(expr, (bool, int, float)):
            value = expr
        else:
            return None

        if target_type_name == "bool":
            if isinstance(value, bool):
                return self.register_constant(value, target_type)
            return None

        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None

        if target_type_name in {"float", "double"}:
            return self.register_constant(float(value), target_type)
        if target_type_name in {"int", "uint"}:
            return self.register_constant(int(value), target_type)
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

        return self.global_variables.get(name)

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

            array_variable = self.assignable_pointer_from_expression(expr.array)
            if array_variable is None:
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
            return access
        return None

    def is_resource_array_type(self, array_type: Optional[SpirvId]) -> bool:
        element_type = self.array_element_type_from_type(array_type)
        while element_type is not None:
            if element_type.id in self.resource_type_metadata:
                return True
            element_type = self.array_element_type_from_type(element_type)
        return False

    def create_array_element_access(self, array_expr, index: SpirvId):
        array_variable = self.variable_pointer_from_expression(array_expr)
        if array_variable is None or not array_variable.type.storage_class:
            addressable_array = self.assignable_pointer_from_expression(array_expr)
            if addressable_array is not None:
                array_variable = addressable_array

        if array_variable is not None:
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
            return access, element_type

        storage_class = array.type.storage_class or "Function"
        ptr_type = self.register_pointer_type(element_type, storage_class)
        access = self.access_chain(array, [index], ptr_type)
        self.variable_value_types[access.id] = element_type
        self.propagate_storage_buffer_access_metadata(array, access)
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

            # Default handling if member not found
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
                    f"; WARNING: Could not determine array element type for {target.array}"
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
                return_value = self.process_expression(node.value[0])
                if return_value:
                    self.create_return_value(return_value)
                else:
                    self.create_return()
            else:
                if isinstance(node.value, ArrayLiteralNode):
                    return_value = self.process_array_literal(
                        node.value, self.current_return_type
                    )
                else:
                    return_value = self.process_expression(node.value)
                if return_value:
                    self.create_return_value(return_value)
                else:
                    self.create_return()
        else:
            self.create_return()

    def process_if(self, node: IfNode):
        """Process a CrossGL if statement."""
        condition = self.process_expression(node.if_condition)
        if condition is None:
            condition = self.register_constant(True, self.primitive_types["bool"])

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
        """Process simple literal/wildcard matches as a structured if-chain."""
        arms = getattr(node, "arms", []) or []
        if not self.validate_match_arms(arms):
            raise ValueError(
                "Unsupported match arm for SPIR-V codegen; only unguarded "
                "literal patterns and a final wildcard are supported"
            )

        expression = self.process_expression(getattr(node, "expression", None))
        if expression is None:
            expression = self.register_constant(0, self.register_primitive_type("int"))

        literal_arms = [
            arm
            for arm in arms
            if isinstance(getattr(arm, "pattern", None), LiteralPatternNode)
        ]
        wildcard_body = None
        for arm in arms:
            if isinstance(getattr(arm, "pattern", None), WildcardPatternNode):
                wildcard_body = getattr(arm, "body", [])
                break

        if not literal_arms:
            if wildcard_body is not None:
                self.process_statements(wildcard_body)
            return

        merge_label = SpirvId(self.get_id(), SpirvType("label"))

        for arm in literal_arms:
            body_label = SpirvId(self.get_id(), SpirvType("label"))
            next_label = SpirvId(self.get_id(), SpirvType("label"))
            pattern_value = self.process_expression(arm.pattern.literal)
            if pattern_value is None:
                pattern_value = self.register_constant(
                    0, self.register_primitive_type("int")
                )
            condition = self.binary_operation(
                "==",
                self.ensure_registered_type(expression.type),
                expression,
                pattern_value,
            )

            self.create_selection_merge(merge_label)
            self.create_conditional_branch(condition, body_label, next_label)

            self.emit(f"%{body_label.id} = OpLabel")
            self.current_label = body_label.id
            self.process_statements(getattr(arm, "body", []))
            if not self.current_block_has_terminator():
                self.create_branch(merge_label)

            self.emit(f"%{next_label.id} = OpLabel")
            self.current_label = next_label.id

        if wildcard_body is not None:
            self.process_statements(wildcard_body)

        if not self.current_block_has_terminator():
            self.create_branch(merge_label)

        self.emit(f"%{merge_label.id} = OpLabel")
        self.current_label = merge_label.id

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
        """Process CrossGL switch statements as a structured selection chain."""
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

        if not explicit_cases:
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

        for case in explicit_cases:
            body_label = SpirvId(self.get_id(), SpirvType("label"))
            next_label = SpirvId(self.get_id(), SpirvType("label"))
            case_value = self.process_expression(getattr(case, "value", None))
            if case_value is None:
                case_value = self.register_constant(
                    0, self.register_primitive_type("int")
                )
            condition = self.binary_operation(
                "==",
                self.ensure_registered_type(expression.type),
                expression,
                case_value,
            )

            self.create_selection_merge(merge_label)
            self.create_conditional_branch(condition, body_label, next_label)

            self.emit(f"%{body_label.id} = OpLabel")
            self.current_label = body_label.id
            self.loop_merge_labels.append(merge_label)
            try:
                self.process_statements(self.switch_case_statements(case))
                if not self.current_block_has_terminator():
                    self.create_branch(merge_label)
            finally:
                self.loop_merge_labels.pop()

            self.emit(f"%{next_label.id} = OpLabel")
            self.current_label = next_label.id

        if default_case is not None:
            self.loop_merge_labels.append(merge_label)
            try:
                self.process_statements(self.switch_case_statements(default_case))
            finally:
                self.loop_merge_labels.pop()

        if not self.current_block_has_terminator():
            self.create_branch(merge_label)

        self.emit(f"%{merge_label.id} = OpLabel")
        self.current_label = merge_label.id

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

    def process_expression(self, expr) -> Optional[SpirvId]:
        """Process a CrossGL expression."""
        if expr is None:
            return None

        if isinstance(expr, bool):
            bool_type = self.register_primitive_type("bool")
            return self.register_constant(expr, bool_type)
        elif isinstance(expr, int):
            int_type = self.register_primitive_type("int")
            return self.register_constant(expr, int_type)
        elif isinstance(expr, float):
            float_type = self.register_primitive_type("float")
            return self.register_constant(expr, float_type)

        elif isinstance(expr, str):
            if expr in self.local_variables:
                var_id = self.local_variables[expr]
                return self.get_variable_value(var_id)
            elif expr in self.global_variables:
                return self.get_variable_value(self.global_variables[expr])
            elif expr in self.cbuffer_members:
                member_pointer = self.cbuffer_member_pointer(expr)
                if member_pointer is not None:
                    return self.get_variable_value(member_pointer)
            else:
                builtin_component = self.process_dotted_compute_builtin(expr)
                if builtin_component is not None:
                    return builtin_component

                builtin = self.ensure_compute_builtin(expr)
                if builtin is not None:
                    return self.get_variable_value(builtin)

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
                # Return a default value instead of None
                float_type = self.register_primitive_type("float")
                return self.register_constant(0.0, float_type)

        elif isinstance(expr, LiteralNode):
            literal_type = self.convert_type_node_to_string(expr.literal_type)
            primitive_type_name = self.normalize_primitive_name(literal_type)
            if primitive_type_name in {"float", "double"}:
                literal_type_id = self.register_primitive_type(primitive_type_name)
                return self.register_constant(float(expr.value), literal_type_id)
            if primitive_type_name in {"int", "uint"}:
                literal_type_id = self.register_primitive_type(primitive_type_name)
                return self.register_constant(int(expr.value), literal_type_id)
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
            elif expr.name in self.global_variables:
                return self.get_variable_value(self.global_variables[expr.name])
            elif expr.name in self.cbuffer_members:
                member_pointer = self.cbuffer_member_pointer(expr.name)
                if member_pointer is not None:
                    return self.get_variable_value(member_pointer)
            else:
                builtin = self.ensure_compute_builtin(expr.name)
                if builtin is not None:
                    return self.get_variable_value(builtin)

                self.emit(f"; WARNING: Unknown variable {expr.name}")
                # Return a default value instead of None
                float_type = self.register_primitive_type("float")
                return self.register_constant(0.0, float_type)

        elif isinstance(expr, ArrayLiteralNode):
            return self.process_array_literal(expr)

        # Array access
        elif isinstance(expr, ArrayAccessNode):
            index = self.process_expression(expr.index)

            if index is None:
                self.emit(f"; WARNING: Failed to evaluate array access")
                float_type = self.register_primitive_type("float")
                return self.register_constant(0.0, float_type)

            access, element_type = self.create_array_element_access(expr.array, index)

            if access is None or element_type is None:
                self.emit(
                    f"; WARNING: Could not determine array element type for {expr.array}"
                )
                element_type = self.primitive_types["float"]
                return self.register_constant(0.0, element_type)

            return self.load_from_variable(access, element_type)

        elif isinstance(expr, BinaryOpNode):
            left = self.process_expression(expr.left)
            right = self.process_expression(expr.right)

            if left is None or right is None:
                # Return a default value instead of None
                float_type = self.register_primitive_type("float")
                return self.register_constant(0.0, float_type)

            # Determine result type
            result_type = left.type  # Default to left operand's type

            return self.binary_operation(
                expr.op, self.map_crossgl_type(result_type.base_type), left, right
            )

        elif isinstance(expr, UnaryOpNode):
            if expr.op in {"++", "--"}:
                return self.process_increment_expression(expr)

            operand = self.process_expression(expr.operand)
            if operand is None:
                # Return a default value instead of None
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
            return self.represented_ir_diagnostic_default_value(
                "ray tracing", expr.operation
            )

        elif isinstance(expr, RayQueryOpNode):
            return self.process_ray_query_operation(expr)

        elif isinstance(expr, MeshOpNode):
            return self.process_mesh_operation(expr)

        elif isinstance(expr, FunctionCallNode):
            ray_query_call = self.ray_query_call_from_function_call(expr)
            if ray_query_call is not None:
                return self.process_ray_query_operation(ray_query_call)

            callee_expr = getattr(expr, "function", getattr(expr, "name", None))
            callee_name = None
            if hasattr(callee_expr, "name"):
                callee_name = callee_expr.name
            elif isinstance(callee_expr, str):
                callee_name = callee_expr

            if callee_name == "lambda":
                return self.unsupported_lambda_default_value("lambda expression")

            if any(self.contains_lambda_expression(arg) for arg in expr.args):
                result_type = None
                if callee_name in self.function_signatures:
                    result_type = self.function_signatures[callee_name][0]
                return self.unsupported_lambda_default_value(
                    f"call to {callee_name or 'unknown callee'}",
                    result_type,
                )

            inline_storage_buffer_function = self.inline_storage_buffer_functions.get(
                callee_name
            )
            if inline_storage_buffer_function is not None:
                return self.inline_storage_buffer_function_call(
                    inline_storage_buffer_function, expr.args
                )

            # Evaluate arguments
            args = []
            has_errors = False
            for arg_index, arg in enumerate(expr.args):
                arg_value = self.process_call_argument(callee_name, arg, arg_index)
                if arg_value is None:
                    self.emit(
                        f"; WARNING: Failed to evaluate argument for {callee_name or callee_expr}"
                    )
                    has_errors = True
                    # Create a default argument
                    float_type = self.register_primitive_type("float")
                    arg_value = self.register_constant(0.0, float_type)
                args.append(arg_value)

            if has_errors and callee_name == "vec2":
                # Special handling for vec2 constructor with errors
                float_type = self.register_primitive_type("float")
                vector_type = self.register_vector_type(float_type, 2)
                id_value = self.get_id()

                # Create default values if needed
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
                if callee_name in self.function_signatures:
                    result_type = self.function_signatures[callee_name][0]
                if result_type is None or result_type.type.base_type == "void":
                    return None
                return self.default_value_for_type(result_type)

            if (
                callee_name in self.resource_function_names()
                and callee_name not in self.functions
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

            # Default handling if member not found
            self.emit(
                f"; WARNING: Could not find member {member_name} in {struct_type}"
            )
            return None

        else:
            self.emit(f"; WARNING: Unknown expression type {type(expr).__name__}")
            return None

    def register_input(
        self, name: str, type_id: SpirvId, location: int, binding: int
    ) -> SpirvId:
        """Register an input variable with location decoration."""
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
        self, name: str, type_id: SpirvId, location: int, binding: int
    ) -> SpirvId:
        """Register an output variable with location decoration."""
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

    def ensure_compute_builtin(self, name: str) -> Optional[SpirvId]:
        if name in self.global_variables:
            return self.global_variables[name]

        info = self.compute_builtin_info(name)
        if info is None:
            return None

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

        self.global_variables[name] = builtin_id
        return builtin_id

    def register_builtin_input(
        self, name: str, type_id: SpirvId, builtin_name: str
    ) -> SpirvId:
        ptr_type = self.register_pointer_type(type_id, "Input")
        id_value = self.get_id()
        self.emit(f"%{id_value} = OpVariable %{ptr_type.id} Input")
        self.emit(f'OpName %{id_value} "{name}"')
        self.decorations.append(f"OpDecorate %{id_value} BuiltIn {builtin_name}")

        spirv_id = SpirvId(id_value, ptr_type.type, name)
        self.variable_value_types[id_value] = type_id
        self.inputs.append(spirv_id)
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

    def process_dotted_compute_builtin(self, name: str) -> Optional[SpirvId]:
        if "." not in name:
            return None

        base_name, member_name = name.rsplit(".", 1)
        builtin = self.ensure_compute_builtin(base_name)
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

        # Check if it's a known array type in our registry
        for (element_type_id, _), arr_type_id in self.array_types.items():
            if arr_type_id.type.base_type == array_type:
                return self.find_registered_type_by_id(element_type_id)

        # If it's a pointer type, extract the base type
        if array_type.startswith("ptr_"):
            base_type = array_type.replace("ptr_", "", 1)
            for (element_type_id, _), arr_type_id in self.array_types.items():
                if arr_type_id.type.base_type == base_type:
                    return self.find_registered_type_by_id(element_type_id)

            # Look for array type pattern in the base type
            match = re.search(r"array_([^_]+)_", base_type)
            if match:
                element_type_name = match.group(1)

                # Look up the element type ID
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
                # Check if type name is a substring of the array type
                if type_id.type.base_type in array_type:
                    return type_id

        # Default to float if we can't determine the element type
        return self.primitive_types["float"]

    def get_function_qualifier(self, func) -> Optional[str]:
        """Return the shader-stage qualifier from old or new function AST shapes."""
        if hasattr(func, "qualifiers") and func.qualifiers:
            return func.qualifiers[0] if func.qualifiers else None
        if hasattr(func, "qualifier"):
            return func.qualifier
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
        }
        return stage_map.get(stage_name or "fragment", "Fragment")

    def compute_local_size(self, stage) -> Tuple[int, int, int]:
        """Return compute workgroup dimensions from a stage execution config."""
        config = getattr(stage, "execution_config", {}) or {}
        for key in ("local_size", "workgroup_size", "numthreads"):
            value = config.get(key)
            if isinstance(value, (list, tuple)) and len(value) >= 3:
                return int(value[0]), int(value[1]), int(value[2])

        return (
            int(config.get("local_size_x", 1)),
            int(config.get("local_size_y", 1)),
            int(config.get("local_size_z", 1)),
        )

    def stage_attribute_value(self, stage, attribute_name: str):
        """Return the first argument for a stage entry-point attribute."""
        entry_point = getattr(stage, "entry_point", None)
        for attr in getattr(entry_point, "attributes", []) or []:
            if str(getattr(attr, "name", "")).lower() != attribute_name:
                continue
            arguments = getattr(attr, "arguments", None)
            if arguments is None:
                arguments = getattr(attr, "args", [])
            if arguments:
                return arguments[0]
        return None

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
        interface_ids = " ".join(
            f"%{variable.id}" for variable in self.inputs + self.outputs
        )
        interface_suffix = f" {interface_ids}" if interface_ids else ""
        self.emit(
            f'OpEntryPoint {execution_model} %{function_id.id} "{name}"'
            f"{interface_suffix}"
        )
        if execution_model == "Fragment":
            self.emit(f"OpExecutionMode %{function_id.id} OriginUpperLeft")
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

    def spirv_module_version(self) -> str:
        if "MeshShadingEXT" in self.required_capabilities:
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

        float_type = self.primitive_types["float"]
        for i in range(2, 5):
            self.register_vector_type(float_type, i)

        for struct in ast.structs:
            self.process_crossgl_struct(struct)
        for stage in (getattr(ast, "stages", None) or {}).values():
            for struct in getattr(stage, "local_structs", []) or []:
                self.process_crossgl_struct(struct)

        self.function_resource_array_type_hints = (
            self.collect_resource_array_parameter_type_hints(ast)
        )
        self.function_image_access_requirements = (
            self.collect_function_image_access_requirements_for_ast(ast)
        )
        self.function_storage_buffer_access_requirements = (
            self.collect_function_storage_buffer_access_requirements_for_ast(ast)
        )
        self.inline_storage_buffer_functions = (
            self.collect_inline_storage_buffer_functions(ast)
        )
        self.function_execution_models = self.collect_function_execution_models(ast)
        self.function_storage_image_pointer_params = (
            self.collect_storage_image_pointer_parameters(ast)
        )
        self.reserve_explicit_resource_bindings(ast)

        for cbuffer in getattr(ast, "cbuffers", []) or []:
            self.process_cbuffer_declaration(cbuffer)

        for var in getattr(ast, "global_variables", []):
            self.process_global_variable_declaration(var)

        top_level_entries = []
        helper_functions = []
        for func in ast.functions:
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
            ]:
                top_level_entries.append((func, qualifier))
            else:
                helper_functions.append(func)

        for func in self.order_functions_by_dependencies(helper_functions):
            if func.name in self.inline_storage_buffer_functions:
                continue
            self.process_function_node(func)

        entry_points = []

        if getattr(ast, "stages", None):
            for stage in ast.stages.values():
                for var in getattr(stage, "local_variables", []):
                    if var.name not in self.global_variables:
                        self.process_global_variable_declaration(var)

            processed_local_functions = set()
            for stage in ast.stages.values():
                local_functions = [
                    func
                    for func in getattr(stage, "local_functions", [])
                    if id(func) not in processed_local_functions
                ]
                for func in self.order_functions_by_dependencies(local_functions):
                    if id(func) not in processed_local_functions:
                        if func.name in self.inline_storage_buffer_functions:
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
                entry_points.append((execution_model, function_id, func.name, None))

        if entry_points:
            self.main_fn_id = entry_points[0][1].id
            for execution_model, function_id, entry_name, stage in entry_points:
                self.emit_entry_point(execution_model, function_id, entry_name, stage)

        return "\n".join(self.ordered_module_lines())

"""CrossGL-to-Vulkan SPIR-V code generator."""

import re
from typing import List, Optional, Tuple, Union

from .array_utils import parse_array_type, detect_array_element_type
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
    RangeNode,
    ReturnNode,
    ShaderNode,
    StructNode,
    SwitchNode,
    TernaryOpNode,
    UnaryOpNode,
    VariableNode,
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
        self.resource_types = {}
        self.resource_image_types = {}

        self.required_capabilities = set()
        self.global_variables = {}
        self.local_variables = {}
        self.variable_value_types = {}
        self.value_types = {}
        self.constants = {}
        self.vector_constants = {}
        self.composite_constants = {}
        self.resource_type_metadata = {}
        self.precise_global_variables = set()
        self.precise_local_variables = set()
        self.no_contraction_ids = set()
        self.precise_expression_depth = 0

        self.functions = {}
        self.function_signatures = {}
        self.function_resource_array_params = {}
        self.function_resource_array_type_hints = {}
        self.function_execution_models = {}
        self.current_execution_model = None
        self.current_stage = None
        self.current_return_type = None

        self.glsl_std450_id = None
        self.main_fn_id = None
        self.requires_compute_derivatives = False

        self.current_label = None
        self.loop_merge_labels = []
        self.loop_continue_labels = []
        self.defined_functions = set()
        self.current_struct_members = {}

        self.inputs = []
        self.outputs = []
        self.uniform_buffers = []
        self.next_input_location = 0
        self.next_output_location = 0
        self.used_input_locations = set()
        self.used_output_locations = set()
        self.next_resource_binding = 0

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
        self.emit(f"OpStore %{variable_id.id} %{value_id.id}")

    def load_from_variable(self, variable_id: SpirvId, result_type: SpirvId) -> SpirvId:
        """Load a value from a variable."""
        id_value = self.get_id()
        self.emit(f"%{id_value} = OpLoad %{result_type.id} %{variable_id.id}")

        spirv_id = SpirvId(id_value, result_type.type)
        self.value_types[id_value] = result_type
        return spirv_id

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
        return_type, _ = self.function_signatures[function_name]

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
            "texture",
            "texture2D",
            "textureCube",
            "textureCompare",
            "textureCompareLod",
            "textureCompareGrad",
            "textureCompareOffset",
            "textureGatherCompare",
            "textureGatherCompareOffset",
            "textureLod",
            "textureGrad",
            "textureOffset",
            "textureGather",
            "textureGatherOffset",
            "textureGatherOffsets",
            "texelFetch",
            "textureSize",
            "imageSize",
            "textureSamples",
            "imageSamples",
            "textureQueryLevels",
            "textureQueryLod",
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

    def requires_explicit_lod_sampling(self) -> bool:
        return self.current_execution_model in {"GLCompute", "MeshEXT", "TaskEXT"}

    def default_lod_operand(self) -> str:
        lod_id = self.register_constant(0.0, self.register_primitive_type("float"))
        return f"Lod %{lod_id.id}"

    def call_resource_function(
        self, function_name: str, args: List[SpirvId]
    ) -> Optional[SpirvId]:
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

            image_operands = ""
            if metadata.get("multisampled"):
                if len(args) < 3:
                    self.emit("; WARNING: imageLoad requires a sample operand")
                    return self.register_constant(
                        0.0, self.register_primitive_type("float")
                    )
                image_operands = f" Sample %{args[2].id}"

            result_type = self.resource_access_result_type(metadata)
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

            sampled_image_id, coord_id, _, metadata = sample_args

            result_type = self.resource_access_result_type(metadata)
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
                )
            self.value_types[id_value] = result_type
            return SpirvId(id_value, result_type.type)

        if function_name in {
            "textureCompare",
            "textureCompareLod",
            "textureCompareGrad",
            "textureCompareOffset",
        }:
            extra_arg_count = {
                "textureCompare": 0,
                "textureCompareLod": 1,
                "textureCompareGrad": 2,
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
            else:
                self.emit(
                    f"%{id_value} = OpImageSampleDrefExplicitLod %{result_type.id} "
                    f"%{sampled_image_id.id} %{coord_id.id} %{depth_id.id} "
                    f"Grad %{extra_args[0].id} %{extra_args[1].id}"
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
                    f"%{sampled_image_id.id} %{coord_id.id} {offset_operand}"
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

        if function_name == "texelFetch":
            sample_args = self.sampled_texture_operands(function_name, args, 1)
            if sample_args is None:
                return self.register_constant(
                    0.0, self.register_primitive_type("float")
                )

            sampled_image_id, coord_id, extra_args, metadata = sample_args
            operand_id = extra_args[0]

            image_id = self.extract_image_from_sampled_image(sampled_image_id, metadata)
            if image_id is None:
                return self.register_constant(
                    0.0, self.register_primitive_type("float")
                )

            result_type = self.resource_access_result_type(metadata)
            id_value = self.get_id()
            image_operand = "Sample" if metadata.get("multisampled") else "Lod"
            self.emit(
                f"%{id_value} = OpImageFetch %{result_type.id} "
                f"%{image_id.id} %{coord_id.id} {image_operand} %{operand_id.id}"
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

        if function_name in {"textureLod", "textureGrad"}:
            required_arg_count = 3 if function_name == "textureLod" else 4
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
            else:
                image_operands = f"Grad %{extra_args[0].id} %{extra_args[1].id}"

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
            "textureGatherOffset": {2, 3},
            "textureGatherOffsets": {2, 3, 4, 5, 6},
            "textureCompareOffset": {3, 4},
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

        resource_array_params = self.function_resource_array_params.get(
            function_name, set()
        )
        if arg_index in resource_array_params:
            pointer_arg = self.variable_pointer_from_expression(arg)
            if pointer_arg is not None:
                return pointer_arg

        return self.process_expression(arg)

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

    def call_builtin_function(
        self, function_name: str, args: List[SpirvId]
    ) -> Optional[SpirvId]:
        """Call a built-in function."""
        if self.glsl_std450_id is None:
            self.glsl_std450_id = self.get_id()
            self.emit(f'%{self.glsl_std450_id} = OpExtInstImport "GLSL.std.450"')

        function_name = {"frac": "fract"}.get(function_name, function_name)
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
            id_value = self.get_id()
            arg_list = " ".join([f"%{arg.id}" for arg in args])
            self.emit(
                f"%{id_value} = OpCompositeConstruct %{struct_type.id} {arg_list}"
            )
            spirv_id = SpirvId(id_value, struct_type.type)
            self.value_types[id_value] = struct_type
            return spirv_id

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
        self.emit(f"OpReturnValue %{value.id}")

    def current_block_has_terminator(self) -> bool:
        """Return whether the current block already ends in a terminator."""
        for line in reversed(self.code_lines):
            stripped = line.strip()
            if not stripped or stripped.startswith(";"):
                continue
            if re.match(r"%\d+ = OpLabel$", stripped):
                return False
            return stripped.startswith(("OpBranch", "OpReturn", "OpKill"))
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
        result_type = self.value_types.get(value_id.id)
        if result_type is not None:
            metadata = self.resource_type_metadata.get(result_type.id)
            if metadata is not None:
                return metadata

        return self.resource_type_metadata.get(value_id.id)

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
            self.resource_types,
            self.resource_image_types,
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
            self.resource_types,
            self.resource_image_types,
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
            if self.is_resource_array_type(param_type):
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

        self.begin_block()

        previous_execution_model = self.current_execution_model
        if self.current_execution_model is None:
            execution_models = self.function_execution_models.get(
                function_node.name, set()
            )
            if "GLCompute" in execution_models:
                self.current_execution_model = "GLCompute"
            elif len(execution_models) == 1:
                self.current_execution_model = next(iter(execution_models))

        self.process_statements(function_node.body)

        if self.convert_type_node_to_string(function_node.return_type) == "void":
            self.create_return()

        self.end_function()

        self.current_execution_model = previous_execution_model
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
        var_type = self.map_resource_type_with_format(var_type_source, node)
        storage_class = self.infer_global_storage_class(
            node, default_storage_class, var_type_name
        )

        initializer = None
        initial_value = getattr(node, "initial_value", None)
        if storage_class == "Private" and isinstance(initial_value, ArrayLiteralNode):
            initializer = self.process_array_literal(
                initial_value, var_type, constant=True
            )

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
                binding = self.next_resource_binding
                self.next_resource_binding += 1
                self.decorations.append(f"OpDecorate %{var_id.id} DescriptorSet 0")
                self.decorations.append(f"OpDecorate %{var_id.id} Binding {binding}")

        self.global_variables[node.name] = var_id
        if self.has_attribute(node, "precise"):
            self.precise_global_variables.add(node.name)
        return var_id

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
            or self.ensure_compute_builtin(name)
        )

    def array_element_type_from_type(self, array_type: Optional[SpirvId]):
        if array_type is None:
            return None

        for (element_type_id, _), arr_type_id in self.array_types.items():
            if arr_type_id.id == array_type.id:
                return self.find_registered_type_by_id(element_type_id)

        return None

    def array_type_info_from_type(self, array_type: Optional[SpirvId]):
        if array_type is None:
            return None

        for (element_type_id, size), arr_type_id in self.array_types.items():
            if arr_type_id.id == array_type.id:
                return self.find_registered_type_by_id(element_type_id), size

        return None

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

        return self.process_expression(element)

    def process_constant_expression(
        self,
        expr,
        target_type: Optional[SpirvId] = None,
    ) -> Optional[SpirvId]:
        if isinstance(expr, ArrayLiteralNode):
            return self.process_array_literal(expr, target_type, constant=True)

        if isinstance(expr, LiteralNode):
            return self.process_expression(expr)

        if isinstance(expr, FunctionCallNode):
            callee_expr = getattr(expr, "function", getattr(expr, "name", None))
            callee_name = getattr(callee_expr, "name", callee_expr)
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
            return access, element_type

        storage_class = array.type.storage_class or "Function"
        ptr_type = self.register_pointer_type(element_type, storage_class)
        access = self.access_chain(array, [index], ptr_type)
        self.variable_value_types[access.id] = element_type
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

            result_type = self.map_crossgl_type(true_value.type.base_type)
            return self.select_operation(
                result_type, condition, true_value, false_value
            )

        elif isinstance(expr, FunctionCallNode):
            callee_expr = getattr(expr, "function", getattr(expr, "name", None))
            callee_name = None
            if hasattr(callee_expr, "name"):
                callee_name = callee_expr.name
            elif isinstance(callee_expr, str):
                callee_name = callee_expr

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
        }
        return builtins.get(name)

    def ensure_compute_builtin(self, name: str) -> Optional[SpirvId]:
        if name in self.global_variables:
            return self.global_variables[name]

        info = self.compute_builtin_info(name)
        if info is None:
            return None

        type_name, builtin_name, storage_class = info
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
        elif execution_model == "GLCompute":
            x, y, z = self.compute_local_size(stage)
            if self.requires_compute_derivatives:
                x = max(2, x + (x % 2))
                y = max(2, y + (y % 2))
            self.emit(f"OpExecutionMode %{function_id.id} LocalSize {x} {y} {z}")
            if self.requires_compute_derivatives:
                self.emit(f"OpExecutionMode %{function_id.id} DerivativeGroupQuadsKHR")

    def ordered_module_lines(self) -> List[str]:
        """Return SPIR-V assembly lines in logical module-layout order."""
        header_lines = self.code_lines[:3]
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

        self.function_resource_array_type_hints = (
            self.collect_resource_array_parameter_type_hints(ast)
        )
        self.function_execution_models = self.collect_function_execution_models(ast)

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
            ]:
                top_level_entries.append((func, qualifier))
            else:
                helper_functions.append(func)

        for func in self.order_functions_by_dependencies(helper_functions):
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

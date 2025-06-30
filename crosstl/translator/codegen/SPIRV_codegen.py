import re
from typing import List, Optional, Tuple, Union

from .array_utils import parse_array_type, detect_array_element_type
from ..ast import (
    AssignmentNode,
    ArrayAccessNode,
    ArrayNode,
    BinaryOpNode,
    ForNode,
    FunctionCallNode,
    IfNode,
    MemberAccessNode,
    ReturnNode,
    ShaderNode,
    StructNode,
    UnaryOpNode,
    VariableNode,
)


class SpirvType:
    """Represents a SPIR-V type with storage class information."""

    def __init__(self, base_type: str, storage_class: Optional[str] = None):
        self.base_type = base_type
        self.storage_class = storage_class

    def __str__(self) -> str:
        if self.storage_class:
            return f"{self.base_type} ({self.storage_class})"
        return self.base_type


class SpirvId:
    """Represents a SPIR-V ID with its associated type."""

    def __init__(
        self, id_value: int, spirv_type: SpirvType, name: Optional[str] = None
    ):
        self.id = id_value
        self.type = spirv_type
        self.name = name

    def __str__(self) -> str:
        if self.name:
            return f"%{self.id} ({self.name}: {self.type})"
        return f"%{self.id} ({self.type})"


class VulkanSPIRVCodeGen:
    """Generates SPIR-V code from a CrossGL shader AST."""

    def __init__(self):
        # Core tracking
        self.next_id = 1
        self.code_lines = []
        self.decorations = []  # List for OpDecorate instructions

        # Type system
        self.primitive_types = {}  # name -> SpirvId
        self.vector_types = {}  # (component_type, count) -> SpirvId
        self.matrix_types = {}  # (column_type, count) -> SpirvId
        self.struct_types = {}  # name -> SpirvId
        self.pointer_types = {}  # (pointed_type, storage_class) -> SpirvId
        self.function_types = {}  # (return_type, [param_types]) -> SpirvId
        self.array_types = {}  # (element_type, size) -> SpirvId

        # Variables and constants
        self.global_variables = {}  # name -> SpirvId
        self.local_variables = {}  # name -> SpirvId  (cleared per function)
        self.constants = {}  # value -> SpirvId
        self.vector_constants = {}  # (type, components) -> SpirvId

        # Functions
        self.functions = {}  # name -> SpirvId
        self.function_signatures = {}  # name -> (return_type, [param_types])

        # Special IDs
        self.glsl_std450_id = None
        self.main_fn_id = None

        # Tracking
        self.current_label = None
        self.defined_functions = set()
        self.current_struct_members = {}  # struct_name -> [(type, name)]

        # Shader interface
        self.inputs = []  # (name, type, location)
        self.outputs = []  # (name, type, location)
        self.uniform_buffers = []  # (name, bindings)

        # State
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

    def register_primitive_type(self, name: str) -> SpirvId:
        """Create and register a primitive type."""
        if name in self.primitive_types:
            return self.primitive_types[name]

        id_value = self.get_id()
        if name == "void":
            self.emit(f"%{id_value} = OpTypeVoid")
        elif name == "bool":
            self.emit(f"%{id_value} = OpTypeBool")
        elif name == "float":
            self.emit(f"%{id_value} = OpTypeFloat 32")
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

    def register_struct_type(
        self, name: str, members: List[Tuple[SpirvId, str]]
    ) -> SpirvId:
        """Create and register a struct type."""
        if name in self.struct_types:
            return self.struct_types[name]

        id_value = self.get_id()

        # Define the struct type
        member_types = " ".join([f"%{member[0].id}" for member in members])
        self.emit(f"%{id_value} = OpTypeStruct {member_types}")

        # Add name decoration
        self.emit(f'OpName %{id_value} "{name}"')

        # Add member names
        for i, (_, member_name) in enumerate(members):
            self.emit(f'OpMemberName %{id_value} {i} "{member_name}"')

        spirv_type = SpirvType(name)
        spirv_id = SpirvId(id_value, spirv_type, name)
        self.struct_types[name] = spirv_id

        # Store member info for later access
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
            constant_value = "true" if value else "false"
        else:
            constant_value = str(value)

        self.emit(f"%{id_value} = OpConstant %{type_id.id} {constant_value}")

        spirv_id = SpirvId(id_value, type_id.type, f"{type_name}_{value}")
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
        self.vector_constants[key] = spirv_id
        return spirv_id

    def create_variable(
        self, type_id: SpirvId, storage_class: str, name: Optional[str] = None
    ) -> SpirvId:
        """Create a new variable."""
        pointer_type = self.register_pointer_type(type_id, storage_class)

        id_value = self.get_id()
        self.emit(f"%{id_value} = OpVariable %{pointer_type.id} {storage_class}")

        if name:
            self.emit(f'OpName %{id_value} "{name}"')

        spirv_id = SpirvId(id_value, pointer_type.type, name)
        return spirv_id

    def store_to_variable(self, variable_id: SpirvId, value_id: SpirvId):
        """Store a value to a variable."""
        self.emit(f"OpStore %{variable_id.id} %{value_id.id}")

    def load_from_variable(self, variable_id: SpirvId, result_type: SpirvId) -> SpirvId:
        """Load a value from a variable."""
        id_value = self.get_id()
        self.emit(f"%{id_value} = OpLoad %{result_type.id} %{variable_id.id}")

        spirv_id = SpirvId(id_value, result_type.type)
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
        id_value = self.get_id()

        # Map CrossGL operators to SPIR-V instructions
        spv_op = {
            "+": "OpFAdd",
            "-": "OpFSub",
            "*": "OpFMul",
            "/": "OpFDiv",
            "%": "OpFMod",
            "==": "OpFOrdEqual",
            "!=": "OpFOrdNotEqual",
            "<": "OpFOrdLessThan",
            ">": "OpFOrdGreaterThan",
            "<=": "OpFOrdLessThanEqual",
            ">=": "OpFOrdGreaterThanEqual",
            "&&": "OpLogicalAnd",
            "||": "OpLogicalOr",
            "&": "OpBitwiseAnd",
            "|": "OpBitwiseOr",
            "^": "OpBitwiseXor",
            "<<": "OpShiftLeftLogical",
            ">>": "OpShiftRightLogical",
            # Add more mappings as needed
            "MULTIPLY": "OpFMul",  # Fix MULTIPLY to be OpFMul
        }.get(op, f"Op{op}")

        self.emit(f"%{id_value} = {spv_op} %{result_type.id} %{left.id} %{right.id}")

        spirv_id = SpirvId(id_value, result_type.type)
        return spirv_id

    def unary_operation(
        self, op: str, result_type: SpirvId, operand: SpirvId
    ) -> SpirvId:
        """Create a unary operation."""
        id_value = self.get_id()

        # Map CrossGL unary operators to SPIR-V instructions
        spv_op = {
            "+": None,  # No operation needed
            "-": "OpFNegate",
            "!": "OpLogicalNot",
            "~": "OpNot",
        }.get(op)

        if spv_op is None:
            return operand

        self.emit(f"%{id_value} = {spv_op} %{result_type.id} %{operand.id}")

        spirv_id = SpirvId(id_value, result_type.type)
        return spirv_id

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
        return spirv_id

    def call_builtin_function(
        self, function_name: str, args: List[SpirvId]
    ) -> Optional[SpirvId]:
        """Call a built-in function."""
        # Ensure we have the GLSL.std450 extended instruction set imported
        if self.glsl_std450_id is None:
            self.glsl_std450_id = self.get_id()
            self.emit(f'%{self.glsl_std450_id} = OpExtInstImport "GLSL.std.450"')

        # Vector constructors
        if function_name in ["vec2", "vec3", "vec4"]:
            component_count = int(function_name[3:])
            float_type = self.primitive_types["float"]
            vector_type = self.register_vector_type(float_type, component_count)

            id_value = self.get_id()

            # If no arguments are provided, construct a default vector
            if not args:
                # Create a vector with first component 1.0 and others 0.0
                float_zero = self.register_constant(0.0, float_type)
                float_one = self.register_constant(1.0, float_type)

                # Create default vector components
                default_args = [float_zero] * component_count
                if component_count > 0:
                    default_args[0] = float_one

                arg_list = " ".join([f"%{arg.id}" for arg in default_args])
            else:
                arg_list = " ".join([f"%{arg.id}" for arg in args])

            self.emit(
                f"%{id_value} = OpCompositeConstruct %{vector_type.id} {arg_list}"
            )

            return SpirvId(id_value, vector_type.type)

        # Matrix constructors
        elif re.match(r"mat(\d)x\d", function_name) or re.match(
            r"mat\d", function_name
        ):
            if "x" in function_name:
                match = re.match(r"mat(\d)x(\d)", function_name)
                cols, rows = int(match.group(1)), int(match.group(2))
            else:
                match = re.match(r"mat(\d)", function_name)
                cols = rows = int(match.group(1))

            float_type = self.primitive_types["float"]
            vector_type = self.register_vector_type(float_type, rows)
            matrix_type = self.register_matrix_type(vector_type, cols)

            id_value = self.get_id()

            # If no arguments provided, create identity matrix
            if not args:
                # Create identity matrix: 1's on diagonal, 0's elsewhere
                float_zero = self.register_constant(0.0, float_type)
                float_one = self.register_constant(1.0, float_type)

                # Create column vectors
                col_vectors = []
                for col in range(cols):
                    col_components = []
                    for row in range(rows):
                        if col == row:
                            col_components.append(float_one)
                        else:
                            col_components.append(float_zero)

                    col_id = self.get_id()
                    col_args = " ".join([f"%{comp.id}" for comp in col_components])
                    self.emit(
                        f"%{col_id} = OpCompositeConstruct %{vector_type.id} {col_args}"
                    )
                    col_vectors.append(SpirvId(col_id, vector_type.type))

                arg_list = " ".join([f"%{vec.id}" for vec in col_vectors])
            else:
                arg_list = " ".join([f"%{arg.id}" for arg in args])

            self.emit(
                f"%{id_value} = OpCompositeConstruct %{matrix_type.id} {arg_list}"
            )

            return SpirvId(id_value, matrix_type.type)

        # Special case for dot product - use OpDot instead of OpExtInst
        elif function_name == "dot" and len(args) == 2:
            # Get the result type (always float)
            float_type = self.primitive_types["float"]
            result_type_id = float_type.id

            # Generate a direct OpDot instruction
            id_value = self.get_id()
            self.emit(
                f"%{id_value} = OpDot %{result_type_id} %{args[0].id} %{args[1].id}"
            )

            return SpirvId(id_value, float_type.type)

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
                    "sign",
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
                "sign": "FSign",
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

    def map_crossgl_type(self, type_name) -> SpirvId:
        """Map a CrossGL type name to a SPIR-V type ID."""
        # Handle TypeNode objects by converting to string first
        if hasattr(type_name, "name") or hasattr(type_name, "element_type"):
            type_str = self.convert_type_node_to_string(type_name)
        else:
            type_str = str(type_name)

        if type_str == "float":
            return self.register_primitive_type("float")
        elif type_str == "int":
            return self.register_primitive_type("int")
        elif type_name == "bool":
            return self.register_primitive_type("bool")
        elif type_str == "void":
            return self.register_primitive_type("void")
        elif type_str.startswith("vec"):
            # Vector types (vec2, vec3, vec4)
            size = int(type_str[3:])
            float_type = self.register_primitive_type("float")
            return self.register_vector_type(float_type, size)
        elif type_str.startswith("ivec"):
            # Integer vector types (ivec2, ivec3, ivec4)
            size = int(type_str[4:])
            int_type = self.register_primitive_type("int")
            return self.register_vector_type(int_type, size)
        elif type_str.startswith("bvec"):
            # Boolean vector types (bvec2, bvec3, bvec4)
            size = int(type_str[4:])
            bool_type = self.register_primitive_type("bool")
            return self.register_vector_type(bool_type, size)
        elif type_str.startswith("mat"):
            # Matrix types (mat2, mat3, mat4)
            dim = int(type_str[3:])
            float_type = self.register_primitive_type("float")
            col_type = self.register_vector_type(float_type, dim)
            return self.register_matrix_type(col_type, dim)
        elif type_str in self.struct_types:
            # Struct type (reference to existing struct)
            return self.struct_types[type_str]
        else:
            # If type is unknown, return a default float type
            self.emit(f"; WARNING: Unknown type {type_str}, using float as default")
            return self.register_primitive_type("float")

    def convert_type_node_to_string(self, type_node) -> str:
        """Convert new AST TypeNode to string representation."""
        if hasattr(type_node, "name"):
            # PrimitiveType
            return type_node.name
        elif hasattr(type_node, "element_type") and hasattr(type_node, "size"):
            # VectorType or ArrayType
            if hasattr(type_node, "rows"):
                # MatrixType
                element_type = self.convert_type_node_to_string(type_node.element_type)
                return f"mat{type_node.rows}x{type_node.cols}"
            elif str(type(type_node)).find("ArrayType") != -1:
                # ArrayType
                element_type = self.convert_type_node_to_string(type_node.element_type)
                if type_node.size is not None:
                    return f"{element_type}[{type_node.size}]"
                else:
                    return f"{element_type}[]"
            else:
                # VectorType
                element_type = self.convert_type_node_to_string(type_node.element_type)
                size = type_node.size
                if element_type == "float":
                    return f"vec{size}"
                elif element_type == "int":
                    return f"ivec{size}"
                elif element_type == "uint":
                    return f"uvec{size}"
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

            if isinstance(member, VariableNode):
                # Regular struct member - handle both old and new AST
                if hasattr(member, "var_type"):
                    # New AST structure
                    vtype_str = self.convert_type_node_to_string(member.var_type)
                elif hasattr(member, "vtype"):
                    # Old AST structure
                    if hasattr(member.vtype, "name") or hasattr(
                        member.vtype, "element_type"
                    ):
                        # TypeNode object
                        vtype_str = self.convert_type_node_to_string(member.vtype)
                    else:
                        # String
                        vtype_str = str(member.vtype)
                else:
                    vtype_str = "float"

                if vtype_str == "float":
                    member_type = self.register_primitive_type("float")
                elif vtype_str == "int":
                    member_type = self.register_primitive_type("int")
                elif vtype_str == "bool":
                    member_type = self.register_primitive_type("bool")
                elif vtype_str.startswith("vec"):
                    size = int(vtype_str[3:])
                    float_type = self.register_primitive_type("float")
                    member_type = self.register_vector_type(float_type, size)
                elif vtype_str.startswith("mat"):
                    dim = int(vtype_str[3:])
                    float_type = self.register_primitive_type("float")
                    col_type = self.register_vector_type(float_type, dim)
                    member_type = self.register_matrix_type(col_type, dim)
                else:
                    # Struct type (reference to another struct)
                    if vtype_str in self.struct_types:
                        member_type = self.struct_types[vtype_str]
                    else:
                        # Use a default type if unknown
                        self.emit(
                            f"; WARNING: Unknown type {vtype_str} in struct {struct_node.name}"
                        )
                        member_type = self.register_primitive_type("float")

            elif isinstance(member, ArrayNode):
                # Array member
                # Determine base type of the array
                if member.element_type == "float":
                    base_type = self.register_primitive_type("float")
                elif member.element_type == "int":
                    base_type = self.register_primitive_type("int")
                elif member.element_type == "bool":
                    base_type = self.register_primitive_type("bool")
                elif member.element_type.startswith("vec"):
                    size = int(member.element_type[3:])
                    float_type = self.register_primitive_type("float")
                    base_type = self.register_vector_type(float_type, size)
                elif member.element_type.startswith("mat"):
                    dim = int(member.element_type[3:])
                    float_type = self.register_primitive_type("float")
                    col_type = self.register_vector_type(float_type, dim)
                    base_type = self.register_matrix_type(col_type, dim)
                else:
                    # Use a default type if unknown
                    self.emit(
                        f"; WARNING: Unknown type {member.element_type} in array member of struct {struct_node.name}"
                    )
                    base_type = self.register_primitive_type("float")

                # Create array type with size (if provided)
                if member.size:
                    # Try to parse the size as an integer
                    try:
                        array_size = int(member.size)
                        member_type = self.register_array_type(base_type, array_size)
                    except ValueError:
                        # Dynamic size or non-integer size
                        self.emit(
                            f"; WARNING: Non-integer array size {member.size} in struct {struct_node.name}"
                        )
                        member_type = self.register_array_type(base_type)
                else:
                    # Runtime array
                    member_type = self.register_array_type(base_type)

            if member_type:
                members.append((member_type, member_name))

        # Create struct type
        return self.register_struct_type(struct_node.name, members)

    def process_function_node(self, function_node: "FunctionNode"):
        """Process a CrossGL function definition."""
        # Map return type
        return_type = self.map_crossgl_type(function_node.return_type)

        # Map parameter types
        param_types = []
        for param in getattr(
            function_node, "parameters", getattr(function_node, "params", [])
        ):
            if hasattr(param, "vtype"):
                param_type = self.map_crossgl_type(param.vtype)
            else:
                # Default to float if type info missing
                param_type = self.map_crossgl_type("float")
            param_types.append(param_type)

        # Create function
        self.create_function(function_node.name, return_type, param_types)

        # Add parameters
        parameters = []
        for i, param in enumerate(
            getattr(function_node, "parameters", getattr(function_node, "params", []))
        ):
            if hasattr(param, "name"):
                param_name = param.name
            else:
                param_name = f"param{i}"

            param_id = self.create_function_parameter(param_types[i], param_name)
            parameters.append(param_id)
            self.local_variables[param_name] = param_id

        # Create entry block
        self.begin_block()

        # Process function body
        self.process_statements(function_node.body)

        # Add implicit return if needed
        if function_node.return_type == "void":
            self.create_return()

        # End function
        self.end_function()

        # Clear local variables
        self.local_variables.clear()

    def process_statements(self, statements):
        """Process a list of CrossGL statements."""
        # Handle both List and BlockNode structures
        if hasattr(statements, "statements"):
            # BlockNode structure (new AST)
            stmt_list = statements.statements
        elif isinstance(statements, list):
            # List of statements (old AST)
            stmt_list = statements
        else:
            # Single statement - wrap in list
            stmt_list = [statements]

        for stmt in stmt_list:
            self.process_statement(stmt)

    def process_statement(self, stmt):
        """Process a single CrossGL statement."""
        if isinstance(stmt, AssignmentNode):
            self.process_assignment(stmt)
        elif isinstance(stmt, ReturnNode):
            self.process_return(stmt)
        elif isinstance(stmt, IfNode):
            self.process_if(stmt)
        elif isinstance(stmt, ForNode):
            self.process_for(stmt)
        elif isinstance(stmt, FunctionCallNode):
            self.process_expression(stmt)  # Just evaluate and discard result

    def process_assignment(self, node: AssignmentNode):
        """Process a CrossGL assignment statement."""
        # Evaluate right-hand side
        rhs_value = self.process_expression(node.value)
        if rhs_value is None:
            return

        # Handle different types of left-hand side
        if isinstance(node.name, str):
            # Simple variable assignment
            if node.name in self.local_variables:
                # Existing local variable
                var_id = self.local_variables[node.name]
            elif node.name in self.global_variables:
                # Existing global variable
                var_id = self.global_variables[node.name]
            else:
                # Create new local variable
                var_type = self.primitive_types["float"]  # Default if unknown
                if hasattr(rhs_value, "type"):
                    for type_dict in [
                        self.primitive_types,
                        self.vector_types,
                        self.matrix_types,
                        self.struct_types,
                    ]:
                        for type_name, type_id in type_dict.items():
                            if type_id.type.base_type == rhs_value.type.base_type:
                                var_type = type_id
                                break

                var_id = self.create_variable(var_type, "Function", node.name)
                self.local_variables[node.name] = var_id

            # Store value to variable
            self.store_to_variable(var_id, rhs_value)

        elif isinstance(node.name, MemberAccessNode):
            # Member access (e.g., struct.field)
            base = self.process_expression(node.name.object)
            if base is None:
                return

            # Find member index
            member_name = node.name.member
            struct_type = base.type.base_type

            if struct_type in self.current_struct_members:
                for i, (_, name) in enumerate(self.current_struct_members[struct_type]):
                    if name == member_name:
                        # Create access chain
                        int_type = self.primitive_types["int"]
                        index = self.register_constant(i, int_type)

                        # Determine member type
                        member_type = self.current_struct_members[struct_type][i][0]

                        # Create pointer to member
                        ptr_type = self.register_pointer_type(member_type, "Function")

                        # Create access chain
                        access = self.access_chain(base, [index], ptr_type)

                        # Store value to member
                        self.store_to_variable(access, rhs_value)
                        return

            # Default handling if member not found
            self.emit(
                f"; WARNING: Could not find member {member_name} in {struct_type}"
            )

        elif isinstance(node.name, ArrayAccessNode):
            # Array access (e.g., array[index])
            array = self.process_expression(node.name.array)
            index = self.process_expression(node.name.index)

            if array is None or index is None:
                self.emit(f"; WARNING: Failed to evaluate array in assignment")
                return

            # Use the improved element type detection
            element_type = self.determine_array_element_type(array)

            if element_type is None:
                self.emit(
                    f"; WARNING: Could not determine array element type for {node.name.array}"
                )
                # Default to float if we can't determine the element type
                element_type = self.primitive_types["float"]

            # Create pointer to element
            ptr_type = self.register_pointer_type(element_type, "Function")

            # Create access chain
            access = self.access_chain(array, [index], ptr_type)

            # Store value to array element
            self.store_to_variable(access, rhs_value)
        else:
            self.emit(
                f"; WARNING: Unsupported LHS type in assignment: {type(node.name).__name__}"
            )

    def process_return(self, node: ReturnNode):
        """Process a CrossGL return statement."""
        if hasattr(node, "value") and node.value:
            # Handle return values
            if isinstance(node.value, list) and node.value:
                # Return first value from list (most common case)
                return_value = self.process_expression(node.value[0])
                if return_value:
                    self.create_return_value(return_value)
                else:
                    self.create_return()
            else:
                # Single return value
                return_value = self.process_expression(node.value)
                if return_value:
                    self.create_return_value(return_value)
                else:
                    self.create_return()
        else:
            # Void return
            self.create_return()

    def process_if(self, node: IfNode):
        """Process a CrossGL if statement."""
        # Evaluate condition
        condition = self.process_expression(node.if_condition)
        if condition is None:
            # Use default true condition if expression failed
            condition = self.register_constant(True, self.primitive_types["bool"])

        # Create merge block for end of if statement
        merge_label = SpirvId(self.get_id(), SpirvType("label"))

        # Create label for then block
        then_label = SpirvId(self.get_id(), SpirvType("label"))

        # Create label for else block
        else_label = SpirvId(self.get_id(), SpirvType("label"))

        # Create the selection structure
        self.create_selection_merge(merge_label)
        self.create_conditional_branch(condition, then_label, else_label)

        # Then block
        self.emit(f"%{then_label.id} = OpLabel")
        self.current_label = then_label.id
        self.process_statements(node.if_body)
        self.create_branch(merge_label)

        # Else block
        self.emit(f"%{else_label.id} = OpLabel")
        self.current_label = else_label.id
        if node.else_body:
            self.process_statements(node.else_body)
        self.create_branch(merge_label)

        # Merge block
        self.emit(f"%{merge_label.id} = OpLabel")
        self.current_label = merge_label.id

    def process_for(self, node: ForNode):
        """Process a CrossGL for loop."""
        # Process loop initialization
        if node.init:
            self.process_statement(node.init)

        # Create loop header block
        header_label = SpirvId(self.get_id(), SpirvType("label"))

        # Create loop body block
        body_label = SpirvId(self.get_id(), SpirvType("label"))

        # Create continue block
        continue_label = SpirvId(self.get_id(), SpirvType("label"))

        # Create merge block
        merge_label = SpirvId(self.get_id(), SpirvType("label"))

        # Branch to header
        self.create_branch(header_label)

        # Header block
        self.emit(f"%{header_label.id} = OpLabel")
        self.current_label = header_label.id

        # Evaluate condition
        condition = self.process_expression(node.condition)
        if condition is None:
            # Use default true condition if expression failed
            condition = self.register_constant(True, self.primitive_types["bool"])

        # Create loop structure
        self.create_loop_merge(merge_label, continue_label)
        self.create_conditional_branch(condition, body_label, merge_label)

        # Body block
        self.emit(f"%{body_label.id} = OpLabel")
        self.current_label = body_label.id
        if node.body:
            self.process_statements(node.body)
        self.create_branch(continue_label)

        # Continue block
        self.emit(f"%{continue_label.id} = OpLabel")
        self.current_label = continue_label.id
        if node.update:
            self.process_statement(node.update)
        self.create_branch(header_label)

        # Merge block
        self.emit(f"%{merge_label.id} = OpLabel")
        self.current_label = merge_label.id

    def process_expression(self, expr) -> Optional[SpirvId]:
        """Process a CrossGL expression."""
        if expr is None:
            return None

        # Literal values
        if isinstance(expr, bool):
            bool_type = self.register_primitive_type("bool")
            return self.register_constant(expr, bool_type)
        elif isinstance(expr, int):
            int_type = self.register_primitive_type("int")
            return self.register_constant(expr, int_type)
        elif isinstance(expr, float):
            float_type = self.register_primitive_type("float")
            return self.register_constant(expr, float_type)

        # Variable references
        elif isinstance(expr, str):
            if expr in self.local_variables:
                var_id = self.local_variables[expr]
                if var_id.type.storage_class:
                    # Load from variable
                    var_type = None
                    for type_dict in [
                        self.primitive_types,
                        self.vector_types,
                        self.matrix_types,
                        self.struct_types,
                    ]:
                        for type_name, type_id in type_dict.items():
                            if type_id.type.base_type == var_id.type.base_type.replace(
                                "ptr_", ""
                            ):
                                var_type = type_id
                                break
                    if var_type:
                        return self.load_from_variable(var_id, var_type)
                return var_id
            elif expr in self.global_variables:
                return self.global_variables[expr]
            else:
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

        # Variable node
        elif isinstance(expr, VariableNode):
            if expr.name in self.local_variables:
                var_id = self.local_variables[expr.name]
                if var_id.type.storage_class:
                    # Load from variable
                    var_type = self.map_crossgl_type(expr.vtype)
                    return self.load_from_variable(var_id, var_type)
                return var_id
            elif expr.name in self.global_variables:
                return self.global_variables[expr.name]
            else:
                self.emit(f"; WARNING: Unknown variable {expr.name}")
                # Return a default value instead of None
                float_type = self.register_primitive_type("float")
                return self.register_constant(0.0, float_type)

        # Array access
        elif isinstance(expr, ArrayAccessNode):
            # Process the array and index expressions
            array = self.process_expression(expr.array)
            index = self.process_expression(expr.index)

            if array is None or index is None:
                self.emit(f"; WARNING: Failed to evaluate array access")
                # Return a default value instead of None
                float_type = self.register_primitive_type("float")
                return self.register_constant(0.0, float_type)

            # Use the improved element type detection
            element_type = self.determine_array_element_type(array)

            if element_type is None:
                self.emit(
                    f"; WARNING: Could not determine array element type for {expr.array}"
                )
                # Default to float if we can't determine the element type
                element_type = self.primitive_types["float"]

            # Create pointer to element
            ptr_type = self.register_pointer_type(element_type, "Function")

            # Create access chain
            access = self.access_chain(array, [index], ptr_type)

            # Load value from array element
            return self.load_from_variable(access, element_type)

        # Binary operations
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

        # Unary operations
        elif isinstance(expr, UnaryOpNode):
            operand = self.process_expression(expr.operand)
            if operand is None:
                # Return a default value instead of None
                float_type = self.register_primitive_type("float")
                return self.register_constant(0.0, float_type)

            return self.unary_operation(expr.op, operand.type, operand)

        # Function calls
        elif isinstance(expr, FunctionCallNode):
            # Evaluate arguments
            args = []
            has_errors = False
            for arg in expr.args:
                arg_value = self.process_expression(arg)
                if arg_value is None:
                    self.emit(f"; WARNING: Failed to evaluate argument for {expr.name}")
                    has_errors = True
                    # Create a default argument
                    float_type = self.register_primitive_type("float")
                    arg_value = self.register_constant(0.0, float_type)
                args.append(arg_value)

            if has_errors and expr.name == "vec2":
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

            return self.call_function(expr.name, args)

        # Member access
        elif isinstance(expr, MemberAccessNode):
            base = self.process_expression(expr.object)
            if base is None:
                return None

            # Find member index
            member_name = expr.member
            struct_type = base.type.base_type

            if struct_type in self.current_struct_members:
                for i, (member_type, name) in enumerate(
                    self.current_struct_members[struct_type]
                ):
                    if name == member_name:
                        # Create access chain
                        int_type = self.primitive_types["int"]
                        index = self.register_constant(i, int_type)

                        # Create pointer to member
                        ptr_type = self.register_pointer_type(member_type, "Function")

                        # Create access chain
                        access = self.access_chain(base, [index], ptr_type)

                        # Load value from member
                        return self.load_from_variable(access, member_type)

            # Default handling if member not found
            self.emit(
                f"; WARNING: Could not find member {member_name} in {struct_type}"
            )
            return None

        # Unknown expression type
        else:
            self.emit(f"; WARNING: Unknown expression type {type(expr).__name__}")
            return None

    def register_input(
        self, name: str, type_id: SpirvId, location: int, binding: int
    ) -> SpirvId:
        """Register an input variable with location decoration."""
        # Create an Input pointer type
        ptr_type = self.register_pointer_type(type_id, "Input")

        # Create the variable
        id_value = self.get_id()
        self.emit(f"%{id_value} = OpVariable %{ptr_type.id} Input")

        # Add the location decoration
        self.decorations.append(f"OpDecorate %{id_value} Location {location}")

        # Add the name
        if name:
            self.emit(f'OpName %{id_value} "{name}"')

        # Create and return the ID
        spirv_id = SpirvId(id_value, ptr_type.type, name)
        return spirv_id

    def register_output(
        self, name: str, type_id: SpirvId, location: int, binding: int
    ) -> SpirvId:
        """Register an output variable with location decoration."""
        # Create an Output pointer type
        ptr_type = self.register_pointer_type(type_id, "Output")

        # Create the variable
        id_value = self.get_id()
        self.emit(f"%{id_value} = OpVariable %{ptr_type.id} Output")

        # Add the location decoration
        self.decorations.append(f"OpDecorate %{id_value} Location {location}")

        # Add the name
        if name:
            self.emit(f'OpName %{id_value} "{name}"')

        # Create and return the ID
        spirv_id = SpirvId(id_value, ptr_type.type, name)
        return spirv_id

    def register_array_type(
        self, element_type: SpirvId, size: Optional[int] = None
    ) -> SpirvId:
        """Create and register an array type."""
        key = (element_type.id, size)
        if key in self.array_types:
            return self.array_types[key]

        id_value = self.get_id()

        if size is not None:
            # Fixed-size array
            size_const = self.register_constant(
                size, self.register_primitive_type("int")
            )
            self.emit(f"%{id_value} = OpTypeArray %{element_type.id} %{size_const.id}")
        else:
            # Runtime array
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
        for arr_type, arr_type_id in self.array_types.items():
            if arr_type_id.type.base_type == array_type:
                # Extract the element type from the array type info
                element_type_name = arr_type.split("_")[
                    1
                ]  # Format: "array_float_4" -> "float"

                # Look up the element type ID
                for type_dict in [
                    self.primitive_types,
                    self.vector_types,
                    self.matrix_types,
                ]:
                    for type_name, type_id in type_dict.items():
                        if type_name == element_type_name:
                            return type_id

        # If it's a pointer type, extract the base type
        if array_type.startswith("ptr_"):
            base_type = array_type.replace("ptr_", "", 1)

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
                    for type_name, type_id in type_dict.items():
                        if type_name == element_type_name:
                            return type_id

        # Last resort: Try to parse from type name
        for type_dict in [
            self.primitive_types,
            self.vector_types,
            self.matrix_types,
        ]:
            for type_name, type_id in type_dict.items():
                # Check if type name is a substring of the array type
                if type_name in array_type:
                    return type_id

        # Default to float if we can't determine the element type
        return self.primitive_types["float"]

    def generate(self, ast):
        """Generate SPIR-V code from a CrossGL AST."""
        if not isinstance(ast, ShaderNode):
            return "; Error: Not a shader node"

        # Initialize code
        self.code_lines = []

        # Shader preprocess - identify vertex/fragment shaders
        for func in ast.functions:
            # Handle both old and new AST structures
            if hasattr(func, "qualifiers") and func.qualifiers:
                qualifier = func.qualifiers[0] if func.qualifiers else None
            elif hasattr(func, "qualifier"):
                qualifier = func.qualifier
            else:
                qualifier = None

            if qualifier == "vertex":
                self.is_vertex_shader = True
                break

        # === Header section ===
        self.emit("; SPIR-V")
        self.emit("; Version: 1.0")
        self.emit("; Generator: CrossGL Vulkan SPIR-V Generator")
        self.emit("; Schema: 0")

        # Capabilities and extensions
        self.emit("OpCapability Shader")

        # Import GLSL.std450
        self.glsl_std450_id = self.get_id()
        self.emit(f'%{self.glsl_std450_id} = OpExtInstImport "GLSL.std.450"')

        # Memory model
        self.emit("OpMemoryModel Logical GLSL450")

        # Register primitive types
        self.register_primitive_type("void")
        self.register_primitive_type("bool")
        self.register_primitive_type("int")
        self.register_primitive_type("float")

        # Register common vector types
        float_type = self.primitive_types["float"]
        for i in range(2, 5):
            self.register_vector_type(float_type, i)

        # Process structs first
        for struct in ast.structs:
            self.process_crossgl_struct(struct)

        # Process functions
        for func in ast.functions:
            # Handle both old and new AST structures for function qualifiers
            if hasattr(func, "qualifiers") and func.qualifiers:
                qualifier = func.qualifiers[0] if func.qualifiers else None
            elif hasattr(func, "qualifier"):
                qualifier = func.qualifier
            else:
                qualifier = None

            if func.name == "main" or qualifier in ["vertex", "fragment"]:
                # Main shader function - save for later
                if self.main_fn_id is None:
                    self.main_fn_id = self.get_id()
            else:
                # Helper function
                self.process_function_node(func)

        # Process main shader function last
        for func in ast.functions:
            # Handle both old and new AST structures for function qualifiers
            if hasattr(func, "qualifiers") and func.qualifiers:
                qualifier = func.qualifiers[0] if func.qualifiers else None
            elif hasattr(func, "qualifier"):
                qualifier = func.qualifier
            else:
                qualifier = None

            if qualifier in ["vertex", "fragment"]:
                self.process_function_node(func)

        # === Entry point ===
        shader_stage = "Vertex" if self.is_vertex_shader else "Fragment"
        if self.main_fn_id:
            self.emit(f'OpEntryPoint {shader_stage} %{self.main_fn_id} "main"')

        # Execution mode
        if not self.is_vertex_shader:
            self.emit("OpExecutionMode %main OriginUpperLeft")

        # === Finalize ===
        # Update bound with final ID count
        header_lines = self.code_lines[:3]
        bound_line = f"; Bound: {self.next_id}"
        rest_lines = self.code_lines[4:]

        final_code = "\n".join(header_lines + [bound_line] + rest_lines)
        return final_code

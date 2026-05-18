"""CrossGL-to-Mojo code generator."""

from ..ast import (
    ArrayNode,
    ArrayAccessNode,
    AssignmentNode,
    BinaryOpNode,
    CbufferNode,
    ForNode,
    FunctionCallNode,
    FunctionNode,
    IfNode,
    MemberAccessNode,
    ReturnNode,
    ShaderNode,
    StructNode,
    TernaryOpNode,
    UnaryOpNode,
    VariableNode,
)
from .array_utils import parse_array_type, format_array_type, get_array_size_from_node


MOJO_VECTOR_TYPES = {
    "vec2": ("DType.float32", 2, 2, None),
    "vec3": ("DType.float32", 3, 4, "0.0"),
    "vec4": ("DType.float32", 4, 4, None),
    "vec2<f32>": ("DType.float32", 2, 2, None),
    "vec3<f32>": ("DType.float32", 3, 4, "0.0"),
    "vec4<f32>": ("DType.float32", 4, 4, None),
    "vec2<f64>": ("DType.float64", 2, 2, None),
    "vec3<f64>": ("DType.float64", 3, 4, "0.0"),
    "vec4<f64>": ("DType.float64", 4, 4, None),
    "vec2<i32>": ("DType.int32", 2, 2, None),
    "vec3<i32>": ("DType.int32", 3, 4, "0"),
    "vec4<i32>": ("DType.int32", 4, 4, None),
    "vec2<u32>": ("DType.uint32", 2, 2, None),
    "vec3<u32>": ("DType.uint32", 3, 4, "0"),
    "vec4<u32>": ("DType.uint32", 4, 4, None),
    "vec2<bool>": ("DType.bool", 2, 2, None),
    "vec3<bool>": ("DType.bool", 3, 4, "False"),
    "vec4<bool>": ("DType.bool", 4, 4, None),
    "ivec2": ("DType.int32", 2, 2, None),
    "ivec3": ("DType.int32", 3, 4, "0"),
    "ivec4": ("DType.int32", 4, 4, None),
    "uvec2": ("DType.uint32", 2, 2, None),
    "uvec3": ("DType.uint32", 3, 4, "0"),
    "uvec4": ("DType.uint32", 4, 4, None),
    "dvec2": ("DType.float64", 2, 2, None),
    "dvec3": ("DType.float64", 3, 4, "0.0"),
    "dvec4": ("DType.float64", 4, 4, None),
    "bvec2": ("DType.bool", 2, 2, None),
    "bvec3": ("DType.bool", 3, 4, "False"),
    "bvec4": ("DType.bool", 4, 4, None),
    "bool2": ("DType.bool", 2, 2, None),
    "bool3": ("DType.bool", 3, 4, "False"),
    "bool4": ("DType.bool", 4, 4, None),
}

SWIZZLE_SETS = {
    "xyzw": {"x": 0, "y": 1, "z": 2, "w": 3},
    "rgba": {"r": 0, "g": 1, "b": 2, "a": 3},
}

MOJO_DTYPE_INFO = {
    "DType.float32": ("float", "vec", "0.0"),
    "DType.float64": ("double", "dvec", "0.0"),
    "DType.int32": ("int", "ivec", "0"),
    "DType.uint32": ("uint", "uvec", "0"),
    "DType.bool": ("bool", "bvec", "False"),
}

MOJO_DTYPE_SUFFIX = {
    "DType.float32": "f32",
    "DType.float64": "f64",
    "DType.int32": "i32",
    "DType.uint32": "u32",
    "DType.bool": "bool",
}

MOJO_VECTOR_ARITHMETIC_OPS = {
    "+": "add",
    "-": "sub",
    "*": "mul",
    "/": "div",
}


class MojoCodeGen:
    def __init__(self):
        self.vector_constructor_info = MOJO_VECTOR_TYPES
        self.struct_types = {}
        self.function_return_types = {}
        self.variable_types = {}
        self.required_helpers = set()
        self.required_splat_helpers = set()
        self.required_swizzle_helpers = set()
        self.required_constructor_helpers = {}
        self.current_shader = None
        self.type_mapping = {
            # Scalar Types
            "void": "None",
            "int": "Int32",
            "short": "Int16",
            "long": "Int64",
            "uint": "UInt32",
            "ushort": "UInt16",
            "ulong": "UInt64",
            "float": "Float32",
            "double": "Float64",
            "half": "Float16",
            "bool": "Bool",
            "string": "String",
            "char": "String",
            **{
                name: f"SIMD[{dtype}, {storage_width}]"
                for name, (dtype, _, storage_width, _) in MOJO_VECTOR_TYPES.items()
            },
            # Matrix Types
            "mat2": "Matrix[DType.float32, 2, 2]",
            "mat3": "Matrix[DType.float32, 3, 3]",
            "mat4": "Matrix[DType.float32, 4, 4]",
            "mat2x2": "Matrix[DType.float32, 2, 2]",
            "mat2x3": "Matrix[DType.float32, 2, 3]",
            "mat2x4": "Matrix[DType.float32, 2, 4]",
            "mat3x2": "Matrix[DType.float32, 3, 2]",
            "mat3x3": "Matrix[DType.float32, 3, 3]",
            "mat3x4": "Matrix[DType.float32, 3, 4]",
            "mat4x2": "Matrix[DType.float32, 4, 2]",
            "mat4x3": "Matrix[DType.float32, 4, 3]",
            "mat4x4": "Matrix[DType.float32, 4, 4]",
            "dmat2": "Matrix[DType.float64, 2, 2]",
            "dmat3": "Matrix[DType.float64, 3, 3]",
            "dmat4": "Matrix[DType.float64, 4, 4]",
            "dmat2x2": "Matrix[DType.float64, 2, 2]",
            "dmat2x3": "Matrix[DType.float64, 2, 3]",
            "dmat2x4": "Matrix[DType.float64, 2, 4]",
            "dmat3x2": "Matrix[DType.float64, 3, 2]",
            "dmat3x3": "Matrix[DType.float64, 3, 3]",
            "dmat3x4": "Matrix[DType.float64, 3, 4]",
            "dmat4x2": "Matrix[DType.float64, 4, 2]",
            "dmat4x3": "Matrix[DType.float64, 4, 3]",
            "dmat4x4": "Matrix[DType.float64, 4, 4]",
            # Texture Types (Mojo equivalents)
            "sampler2D": "Texture2D",
            "samplerCube": "TextureCube",
            "sampler": "Sampler",
        }

        self.semantic_map = {
            # Vertex attributes
            "gl_VertexID": "vertex_id",
            "gl_InstanceID": "instance_id",
            "gl_Position": "position",
            "gl_PointSize": "point_size",
            "gl_ClipDistance": "clip_distance",
            # Fragment attributes
            "gl_FragColor": "color(0)",
            "gl_FragColor0": "color(0)",
            "gl_FragColor1": "color(1)",
            "gl_FragColor2": "color(2)",
            "gl_FragColor3": "color(3)",
            "gl_FragDepth": "depth(any)",
            "gl_FragCoord": "position",
            "gl_FrontFacing": "front_facing",
            "gl_PointCoord": "point_coord",
            # Standard vertex semantics
            "POSITION": "position",
            "NORMAL": "normal",
            "TANGENT": "tangent",
            "BINORMAL": "binormal",
            "TEXCOORD": "texcoord",
            "TEXCOORD0": "texcoord0",
            "TEXCOORD1": "texcoord1",
            "TEXCOORD2": "texcoord2",
            "TEXCOORD3": "texcoord3",
            "COLOR": "color",
            "COLOR0": "color0",
            "COLOR1": "color1",
        }

        # Function mapping for common shader functions
        self.function_map = {
            "texture": "sample",
            "normalize": "normalize",
            "dot": "dot_product",
            "cross": "cross_product",
            "length": "magnitude",
            "reflect": "reflect",
            "refract": "refract",
            "sin": "sin",
            "cos": "cos",
            "tan": "tan",
            "sqrt": "sqrt",
            "pow": "power",
            "abs": "abs",
            "min": "min",
            "max": "max",
            "clamp": "clamp",
            "mix": "lerp",
            "smoothstep": "smoothstep",
            "step": "step",
        }

    def generate(self, ast):
        self.struct_types = {}
        self.function_return_types = {}
        self.variable_types = {}
        self.required_helpers = set()
        self.required_splat_helpers = set()
        self.required_swizzle_helpers = set()
        self.required_constructor_helpers = {}
        self.collect_function_return_types(ast)

        header = "# Generated Mojo Shader Code\n"
        header += "from math import *\n"
        header += "from simd import *\n"
        header += "from gpu import *\n\n"
        code = ""

        structs = getattr(ast, "structs", [])
        for node in structs:
            if isinstance(node, StructNode):
                code += self.generate_struct(node)

        global_vars = getattr(ast, "global_variables", [])
        for node in global_vars:
            if isinstance(node, ArrayNode):
                code += self.generate_array_declaration(node)
            else:
                # Handle both old and new AST variable structures
                if hasattr(node, "var_type"):
                    vtype = self.convert_type_node_to_string(node.var_type)
                elif hasattr(node, "vtype"):
                    vtype = node.vtype
                else:
                    vtype = "float"
                code += f"var {node.name}: {self.map_type(vtype)}\n"

        cbuffers = getattr(ast, "cbuffers", [])
        if cbuffers:
            code += "# Constant Buffers\n"
            code += self.generate_cbuffers(ast)

        functions = getattr(ast, "functions", [])
        for func in functions:
            # Handle both old and new AST function structures
            if hasattr(func, "qualifiers") and func.qualifiers:
                qualifier = func.qualifiers[0] if func.qualifiers else None
            else:
                qualifier = getattr(func, "qualifier", None)

            if qualifier == "vertex":
                code += "# Vertex Shader\n"
                code += self.generate_function(func, shader_type="vertex")
            elif qualifier == "fragment":
                code += "# Fragment Shader\n"
                code += self.generate_function(func, shader_type="fragment")
            elif qualifier == "compute":
                code += "# Compute Shader\n"
                code += self.generate_function(func, shader_type="compute")
            else:
                code += self.generate_function(func)

        # Handle shader stages (new AST structure)
        if hasattr(ast, "stages") and ast.stages:
            for stage_type, stage in ast.stages.items():
                if hasattr(stage, "entry_point"):
                    stage_name = (
                        str(stage_type).split(".")[-1].lower()
                    )  # Extract stage name from enum
                    code += f"# {stage_name.title()} Shader\n"
                    code += self.generate_function(
                        stage.entry_point, shader_type=stage_name
                    )
                if hasattr(stage, "local_functions"):
                    for func in stage.local_functions:
                        code += self.generate_function(func)

        return header + self.generate_required_helpers() + code

    def collect_function_return_types(self, ast):
        functions = list(getattr(ast, "functions", []))
        stages = getattr(ast, "stages", {})
        if stages:
            for stage in stages.values():
                entry_point = getattr(stage, "entry_point", None)
                if entry_point is not None:
                    functions.append(entry_point)
                functions.extend(getattr(stage, "local_functions", []))

        for func in functions:
            self.register_function_return_type(func)

    def register_function_return_type(self, func):
        if not hasattr(func, "name"):
            return

        if hasattr(func, "return_type"):
            return_type = self.convert_type_node_to_string(func.return_type)
        else:
            return_type = "void"
        self.function_return_types[func.name] = return_type

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
        elif hasattr(type_node, "element_type") and hasattr(type_node, "size"):
            element_type = self.convert_type_node_to_string(type_node.element_type)
            size = type_node.size
            if element_type == "float":
                return f"vec{size}"
            elif element_type == "int":
                return f"ivec{size}"
            elif element_type == "uint":
                return f"uvec{size}"
            elif element_type == "double":
                return f"dvec{size}"
            elif element_type == "bool":
                return f"bvec{size}"
            else:
                return f"{element_type}{size}"
        elif hasattr(type_node, "element_type") and hasattr(type_node, "rows"):
            element_type = self.convert_type_node_to_string(type_node.element_type)
            prefix = "dmat" if element_type == "double" else "mat"
            if type_node.rows == type_node.cols:
                return f"{prefix}{type_node.rows}"
            return f"{prefix}{type_node.rows}x{type_node.cols}"
        else:
            return str(type_node)

    def format_array_size(self, size):
        if size is None:
            return None
        if hasattr(size, "value"):
            return size.value
        return size

    def extract_semantic_from_attributes(self, attributes):
        """Extract semantic information from new AST attributes."""
        semantic_attrs = [
            "position",
            "color",
            "texcoord",
            "normal",
            "tangent",
            "binormal",
            "POSITION",
            "COLOR",
            "TEXCOORD",
            "NORMAL",
            "TANGENT",
            "BINORMAL",
            "TEXCOORD0",
            "TEXCOORD1",
            "TEXCOORD2",
            "TEXCOORD3",
        ]

        for attr in attributes:
            if hasattr(attr, "name") and attr.name in semantic_attrs:
                return attr.name
        return None

    def generate_struct(self, node):
        code = f"@value\nstruct {node.name}:\n"
        self.struct_types[node.name] = {}

        members = getattr(node, "members", [])
        for member in members:
            if isinstance(member, ArrayNode):
                element_type = getattr(
                    member, "element_type", getattr(member, "vtype", "float")
                )
                if member.size:
                    code += f"    var {member.name}: StaticTuple[{self.map_type(element_type)}, {member.size}]\n"
                else:
                    code += f"    var {member.name}: DynamicVector[{self.map_type(element_type)}]\n"
            else:
                if hasattr(member, "member_type"):
                    member_type = self.convert_type_node_to_string(member.member_type)
                elif hasattr(member, "vtype"):
                    member_type = member.vtype
                else:
                    member_type = "float"

                self.struct_types[node.name][member.name] = member_type

                semantic = None
                if hasattr(member, "semantic"):
                    semantic = member.semantic
                elif hasattr(member, "attributes"):
                    semantic = self.extract_semantic_from_attributes(member.attributes)

                semantic_comment = (
                    f"  # {self.map_semantic(semantic)}" if semantic else ""
                )
                code += f"    var {member.name}: {self.map_type(member_type)}{semantic_comment}\n"

        code += "\n"
        return code

    def generate_cbuffers(self, ast):
        code = ""
        cbuffers = getattr(ast, "cbuffers", [])
        for node in cbuffers:
            if isinstance(node, StructNode):
                code += f"@value\nstruct {node.name}:\n"
                members = getattr(node, "members", [])
                for member in members:
                    if isinstance(member, ArrayNode):
                        element_type = getattr(
                            member, "element_type", getattr(member, "vtype", "float")
                        )
                        if member.size:
                            code += f"    var {member.name}: StaticTuple[{self.map_type(element_type)}, {member.size}]\n"
                        else:
                            code += f"    var {member.name}: DynamicVector[{self.map_type(element_type)}]\n"
                    else:
                        # Handle both old and new AST member structures
                        if hasattr(member, "member_type"):
                            member_type = self.map_type(str(member.member_type))
                        else:
                            member_type = self.map_type(
                                getattr(member, "vtype", "float")
                            )
                        code += f"    var {member.name}: {member_type}\n"
                code += "\n"
            elif hasattr(node, "name") and hasattr(node, "members"):  # CbufferNode
                code += f"@value\nstruct {node.name}:\n"
                for member in node.members:
                    if isinstance(member, ArrayNode):
                        element_type = getattr(
                            member, "element_type", getattr(member, "vtype", "float")
                        )
                        if member.size:
                            code += f"    var {member.name}: StaticTuple[{self.map_type(element_type)}, {member.size}]\n"
                        else:
                            code += f"    var {member.name}: DynamicVector[{self.map_type(element_type)}]\n"
                    else:
                        # Handle both old and new AST member structures
                        if hasattr(member, "member_type"):
                            member_type = self.map_type(str(member.member_type))
                        else:
                            member_type = self.map_type(
                                getattr(member, "vtype", "float")
                            )
                        code += f"    var {member.name}: {member_type}\n"
                code += "\n"
        return code

    def generate_function(self, func, indent=0, shader_type=None):
        code = ""
        "    " * indent
        previous_variable_types = self.variable_types.copy()

        param_list = getattr(func, "parameters", getattr(func, "params", []))
        params = []
        for p in param_list:
            if hasattr(p, "param_type"):
                param_type = self.convert_type_node_to_string(p.param_type)
            elif hasattr(p, "vtype"):
                param_type = p.vtype
            else:
                param_type = "float"

            semantic = None
            if hasattr(p, "semantic"):
                semantic = p.semantic
            elif hasattr(p, "attributes"):
                semantic = self.extract_semantic_from_attributes(p.attributes)

            self.register_variable_type(p.name, param_type)
            param_semantic = f"  # {self.map_semantic(semantic)}" if semantic else ""
            params.append(f"{p.name}: {self.map_type(param_type)}{param_semantic}")

        params_str = ", ".join(params) if params else ""

        if hasattr(func, "return_type"):
            return_type = self.convert_type_node_to_string(func.return_type)
        else:
            return_type = "void"
        self.function_return_types[func.name] = return_type

        if shader_type == "vertex":
            code += f"@vertex_shader\n"
        elif shader_type == "fragment":
            code += f"@fragment_shader\n"
        elif shader_type == "compute":
            code += f"@compute_shader\n"

        code += f"fn {func.name}({params_str}) -> {self.map_type(return_type)}:\n"

        body = getattr(func, "body", [])
        if hasattr(body, "statements"):
            for stmt in body.statements:
                code += self.generate_statement(stmt, indent + 1)
        elif isinstance(body, list):
            for stmt in body:
                code += self.generate_statement(stmt, indent + 1)
        else:
            code += "    pass\n"

        code += "\n"
        self.variable_types = previous_variable_types
        return code

    def generate_statement(self, stmt, indent=0):
        indent_str = "    " * indent

        if isinstance(stmt, VariableNode):
            if hasattr(stmt, "var_type"):
                var_type = self.convert_type_node_to_string(stmt.var_type)
            elif hasattr(stmt, "vtype") and stmt.vtype:
                # Old AST structure - check if this is actually an array declaration disguised as a variable
                vtype_str = str(stmt.vtype)
                if (
                    "ArrayAccessNode" in vtype_str
                    and "array=" in vtype_str
                    and "index=" in vtype_str
                ):
                    # This is likely an array declaration
                    import re

                    array_match = re.search(r"array=(\w+).*?index=(\w+)", vtype_str)
                    if array_match:
                        array_match.group(1)
                        size = array_match.group(2)
                        base_type = "Float32"  # Default, could be improved
                        return f"{indent_str}var {stmt.name}: StaticTuple[{base_type}, {size}]\n"
                var_type = stmt.vtype
            else:
                var_type = "float"

            self.register_variable_type(stmt.name, var_type)
            if hasattr(stmt, "initial_value") and stmt.initial_value is not None:
                init_expr = self.generate_expression(stmt.initial_value)
                return f"{indent_str}var {stmt.name}: {self.map_type(var_type)} = {init_expr}\n"
            else:
                return f"{indent_str}var {stmt.name}: {self.map_type(var_type)}\n"
        elif isinstance(stmt, ArrayNode):
            return self.generate_array_declaration(stmt, indent)
        elif isinstance(stmt, AssignmentNode):
            return f"{indent_str}{self.generate_assignment(stmt)}\n"
        elif isinstance(stmt, IfNode):
            return self.generate_if(stmt, indent)
        elif isinstance(stmt, ForNode):
            return self.generate_for(stmt, indent)
        elif isinstance(stmt, ReturnNode):
            if isinstance(stmt.value, list):
                # Multiple return values
                values = ", ".join(self.generate_expression(val) for val in stmt.value)
                return f"{indent_str}return {values}\n"
            else:
                return f"{indent_str}return {self.generate_expression(stmt.value)}\n"
        elif isinstance(stmt, ArrayAccessNode):
            # ArrayAccessNode should not appear as a statement by itself - it's likely a misclassified array declaration
            # Try to handle it gracefully
            return f"{indent_str}# Unhandled ArrayAccessNode: {stmt}\n"
        else:
            # Handle expressions that may be used as statements
            expr_result = self.generate_expression(stmt)
            if expr_result.strip():
                return f"{indent_str}{expr_result}\n"
            else:
                return f"{indent_str}# Unhandled statement: {type(stmt).__name__}\n"

    def generate_array_declaration(self, node, indent=0):
        indent_str = "    " * indent
        element_type = self.map_type(node.element_type)
        size = get_array_size_from_node(node)

        if size is None:
            return f"{indent_str}var {node.name}: DynamicVector[{element_type}]\n"
        else:
            return f"{indent_str}var {node.name}: StaticTuple[{element_type}, {size}]\n"

    def generate_assignment(self, node):
        left = self.generate_expression(node.left)
        right = self.generate_expression(node.right)
        op = self.map_operator(node.operator)
        return f"{left} {op} {right}"

    def generate_if(self, node, indent):
        indent_str = "    " * indent
        condition = self.generate_expression(
            node.condition if hasattr(node, "condition") else node.if_condition
        )
        code = f"{indent_str}if {condition}:\n"

        if_body = getattr(node, "then_branch", getattr(node, "if_body", None))
        if hasattr(if_body, "statements"):
            for stmt in if_body.statements:
                code += self.generate_statement(stmt, indent + 1)
        elif isinstance(if_body, list):
            for stmt in if_body:
                code += self.generate_statement(stmt, indent + 1)

        else_branch = getattr(node, "else_branch", None)
        if else_branch:
            if hasattr(else_branch, "__class__") and "If" in str(else_branch.__class__):
                # Generate elif by recursively generating the nested if with elif prefix
                elif_condition = self.generate_expression(
                    else_branch.condition
                    if hasattr(else_branch, "condition")
                    else else_branch.if_condition
                )
                code += f"{indent_str}elif {elif_condition}:\n"

                # Generate elif body
                elif_body = getattr(
                    else_branch, "then_branch", getattr(else_branch, "if_body", None)
                )
                if hasattr(elif_body, "statements"):
                    for stmt in elif_body.statements:
                        code += self.generate_statement(stmt, indent + 1)
                elif isinstance(elif_body, list):
                    for stmt in elif_body:
                        code += self.generate_statement(stmt, indent + 1)

                nested_else = getattr(else_branch, "else_branch", None)
                if nested_else:
                    if hasattr(nested_else, "__class__") and "If" in str(
                        nested_else.__class__
                    ):
                        # Another elif
                        remaining_code = self.generate_if(nested_else, indent)
                        # Remove the "if" prefix and replace with "elif"
                        remaining_lines = remaining_code.split("\n")
                        if remaining_lines[0].strip().startswith("if "):
                            remaining_lines[0] = remaining_lines[0].replace(
                                "if ", "elif ", 1
                            )
                        code += "\n".join(remaining_lines)
                    else:
                        # Final else clause
                        code += f"{indent_str}else:\n"
                        if hasattr(nested_else, "statements"):
                            for stmt in nested_else.statements:
                                code += self.generate_statement(stmt, indent + 1)
                        elif isinstance(nested_else, list):
                            for stmt in nested_else:
                                code += self.generate_statement(stmt, indent + 1)
                        else:
                            code += self.generate_statement(nested_else, indent + 1)
            else:
                code += f"{indent_str}else:\n"
                if hasattr(else_branch, "statements"):
                    for stmt in else_branch.statements:
                        code += self.generate_statement(stmt, indent + 1)
                elif isinstance(else_branch, list):
                    for stmt in else_branch:
                        code += self.generate_statement(stmt, indent + 1)
                else:
                    code += self.generate_statement(else_branch, indent + 1)

        return code

    def generate_for(self, node, indent):
        indent_str = "    " * indent

        init = self.generate_statement(node.init, 0).strip()
        condition = self.generate_expression(node.condition)
        update = self.generate_expression(node.update)

        code = f"{indent_str}{init}\n"
        code += f"{indent_str}while {condition}:\n"

        if hasattr(node.body, "statements"):
            for stmt in node.body.statements:
                code += self.generate_statement(stmt, indent + 1)
        elif isinstance(node.body, list):
            for stmt in node.body:
                code += self.generate_statement(stmt, indent + 1)
        else:
            code += self.generate_statement(node.body, indent + 1)

        # Add update at the end of the loop
        code += f"{indent_str}    {update}\n"

        return code

    def generate_expression(self, expr):
        if isinstance(expr, str):
            return expr
        elif isinstance(expr, (int, float, bool)):
            return self.format_literal(expr)
        elif isinstance(expr, VariableNode):
            if hasattr(expr, "vtype") and expr.vtype and expr.name:
                return f"{expr.name}"
            elif hasattr(expr, "name"):
                return expr.name
            else:
                return str(expr)
        elif isinstance(expr, BinaryOpNode):
            vector_binary = self.generate_vector_binary_op(expr)
            if vector_binary is not None:
                return vector_binary
            left = self.generate_expression(expr.left)
            right = self.generate_expression(expr.right)
            op = self.map_operator(expr.op)
            return f"({left} {op} {right})"
        elif isinstance(expr, AssignmentNode):
            return self.generate_assignment(expr)
        elif isinstance(expr, UnaryOpNode):
            operand = self.generate_expression(expr.operand)
            op = self.map_operator(expr.op)
            if op in ["++", "--"]:
                assignment_op = "+=" if op == "++" else "-="
                return f"{operand} {assignment_op} 1"
            return f"({op}{operand})"
        elif isinstance(expr, ArrayAccessNode):
            # Handle array access properly
            if hasattr(expr, "array") and hasattr(expr, "index"):
                array = self.generate_expression(expr.array)
                index = self.generate_expression(expr.index)
                return f"{array}[{index}]"
            else:
                # Fallback for malformed ArrayAccessNode
                return str(expr)
        elif isinstance(expr, FunctionCallNode):
            # Extract function name properly (might be IdentifierNode)
            func_expr = getattr(expr, "function", None)
            if func_expr is None:
                func_expr = expr.name
            func_name = None
            if hasattr(func_expr, "name"):
                # It's an IdentifierNode, extract the name
                func_name = func_expr.name
                callee = func_name
            elif isinstance(func_expr, str):
                func_name = func_expr
                callee = func_expr
            else:
                callee = self.generate_expression(func_expr)

            # Map function names to Mojo equivalents
            func_name = self.function_map.get(func_name, func_name)

            # Handle vector constructors
            if func_name in self.vector_constructor_info:
                return self.generate_vector_constructor(func_name, expr.args)

            if func_name in [
                "mat2",
                "mat3",
                "mat4",
                "mat2x2",
                "mat2x3",
                "mat2x4",
                "mat3x2",
                "mat3x3",
                "mat3x4",
                "mat4x2",
                "mat4x3",
                "mat4x4",
                "dmat2",
                "dmat3",
                "dmat4",
                "dmat2x2",
                "dmat2x3",
                "dmat2x4",
                "dmat3x2",
                "dmat3x3",
                "dmat3x4",
                "dmat4x2",
                "dmat4x3",
                "dmat4x4",
            ]:
                mojo_type = self.map_type(func_name)
                args = ", ".join(self.generate_expression(arg) for arg in expr.args)
                return f"{mojo_type}({args})"

            # Handle standard function calls
            args = ", ".join(self.generate_expression(arg) for arg in expr.args)
            return f"{callee}({args})"
        elif isinstance(expr, MemberAccessNode):
            obj = self.generate_expression(expr.object)
            swizzle_indices = self.get_swizzle_indices(expr.member)
            if swizzle_indices is not None:
                obj_type = self.expression_result_type(expr.object)
                return self.generate_swizzle(
                    expr.object, obj, obj_type, expr.member, swizzle_indices
                )
            return f"{obj}.{expr.member}"
        elif isinstance(expr, TernaryOpNode):
            condition = self.generate_expression(expr.condition)
            true_expr = self.generate_expression(expr.true_expr)
            false_expr = self.generate_expression(expr.false_expr)
            return f"({true_expr} if {condition} else {false_expr})"
        elif hasattr(expr, "__class__") and "Literal" in str(expr.__class__):
            # Handle LiteralNode
            if hasattr(expr, "value"):
                literal_type = getattr(
                    getattr(expr, "literal_type", None), "name", None
                )
                return self.format_literal(expr.value, literal_type)
            return str(expr)
        elif hasattr(expr, "__class__") and "Identifier" in str(expr.__class__):
            # Handle IdentifierNode
            return getattr(expr, "name", str(expr))
        elif hasattr(expr, "__class__") and "ExpressionStatement" in str(
            expr.__class__
        ):
            # Handle ExpressionStatementNode
            if hasattr(expr, "expression"):
                return self.generate_expression(expr.expression)
            else:
                return self.generate_expression(expr)
        else:
            # For unknown expression types, handle special cases
            expr_str = str(expr)
            # Check if this looks like an array declaration being misinterpreted
            if (
                "ArrayAccessNode" in expr_str
                and "array=" in expr_str
                and "index=" in expr_str
            ):
                # Try to extract array name and size for array declarations
                import re

                array_match = re.search(r"array=(\w+).*?index=(\w+)", expr_str)
                if array_match:
                    array_name = array_match.group(1)
                    array_match.group(2)
                    return f"{array_name}"  # Just return the array name for now
            return expr_str

    def generate_vector_constructor(self, func_name, args):
        helper_call = self.generate_constructor_helper_call(func_name, args)
        if helper_call is not None:
            return helper_call

        dtype, source_width, storage_width, pad_literal = self.vector_constructor_info[
            func_name
        ]
        mojo_type = f"SIMD[{dtype}, {storage_width}]"
        emitted_args = []

        if len(args) == 1:
            arg = args[0]
            arg_components = self.vector_components_for_expression(arg)
            if arg_components is not None:
                emitted_args.extend(arg_components[:source_width])
            elif source_width == 3:
                arg_expr = self.generate_expression(arg)
                if self.is_duplicate_sensitive_expression(arg):
                    helper_name = self.vec3_splat_helper_name(dtype)
                    self.required_splat_helpers.add(dtype)
                    return f"{helper_name}({arg_expr})"
                emitted_args.extend([arg_expr] * source_width)
            else:
                emitted_args.append(self.generate_expression(arg))
        else:
            for arg in args:
                arg_components = self.vector_components_for_expression(arg)
                if arg_components is not None:
                    emitted_args.extend(arg_components)
                else:
                    emitted_args.append(self.generate_expression(arg))

        if source_width == 3 and len(emitted_args) == 3:
            emitted_args.append(pad_literal)

        return f"{mojo_type}({', '.join(emitted_args)})"

    def generate_vector_binary_op(self, expr):
        op = self.map_operator(expr.op)
        if op not in MOJO_VECTOR_ARITHMETIC_OPS:
            return None

        left_type = self.expression_result_type(expr.left)
        right_type = self.expression_result_type(expr.right)
        left_info = self.vector_type_info(left_type)
        right_info = self.vector_type_info(right_type)
        left_is_vec3 = left_info is not None and left_info[1] == 3
        right_is_vec3 = right_info is not None and right_info[1] == 3

        if not left_is_vec3 and not right_is_vec3:
            return None

        if left_is_vec3 and right_is_vec3:
            if left_info[0] != right_info[0]:
                return None
            dtype = left_info[0]
            helper_kind = "vv"
        elif left_is_vec3:
            if right_info is not None:
                return None
            dtype = left_info[0]
            helper_kind = "vs"
        else:
            if left_info is not None:
                return None
            dtype = right_info[0]
            helper_kind = "sv"

        if dtype == "DType.bool" or dtype not in MOJO_DTYPE_SUFFIX:
            return None

        left = self.generate_expression(expr.left)
        right = self.generate_expression(expr.right)
        helper_name = self.vector_binary_helper_name(dtype, op, helper_kind)
        self.required_helpers.add((dtype, op, helper_kind))
        return f"{helper_name}({left}, {right})"

    def generate_required_helpers(self):
        if (
            not self.required_helpers
            and not self.required_splat_helpers
            and not self.required_swizzle_helpers
            and not self.required_constructor_helpers
        ):
            return ""

        code = "# CrossGL vector helpers\n"
        for dtype, op, helper_kind in sorted(self.required_helpers):
            code += self.generate_vector_binary_helper(dtype, op, helper_kind)
        for dtype in sorted(self.required_splat_helpers):
            code += self.generate_vec3_splat_helper(dtype)
        for dtype, source_width, member in sorted(self.required_swizzle_helpers):
            code += self.generate_swizzle_helper(dtype, source_width, member)
        for key in sorted(self.required_constructor_helpers):
            code += self.generate_constructor_helper(
                self.required_constructor_helpers[key]
            )
        return code + "\n"

    def generate_vector_binary_helper(self, dtype, op, helper_kind):
        scalar_type, _, pad_literal = MOJO_DTYPE_INFO[dtype]
        mojo_scalar_type = self.map_type(scalar_type)
        vector_type = f"SIMD[{dtype}, 4]"
        helper_name = self.vector_binary_helper_name(dtype, op, helper_kind)

        if helper_kind == "vv":
            params = f"a: {vector_type}, b: {vector_type}"
            components = [f"a[{index}] {op} b[{index}]" for index in range(3)]
        elif helper_kind == "vs":
            params = f"v: {vector_type}, s: {mojo_scalar_type}"
            components = [f"v[{index}] {op} s" for index in range(3)]
        else:
            params = f"s: {mojo_scalar_type}, v: {vector_type}"
            components = [f"s {op} v[{index}]" for index in range(3)]

        components.append(pad_literal)
        args = ", ".join(components)
        code = f"fn {helper_name}({params}) -> {vector_type}:\n"
        code += f"    return {vector_type}({args})\n\n"
        return code

    def vector_binary_helper_name(self, dtype, op, helper_kind):
        op_name = MOJO_VECTOR_ARITHMETIC_OPS[op]
        dtype_suffix = MOJO_DTYPE_SUFFIX[dtype]
        return f"_crossgl_vec3_{op_name}_{dtype_suffix}_{helper_kind}"

    def generate_vec3_splat_helper(self, dtype):
        scalar_type, _, pad_literal = MOJO_DTYPE_INFO[dtype]
        mojo_scalar_type = self.map_type(scalar_type)
        vector_type = f"SIMD[{dtype}, 4]"
        helper_name = self.vec3_splat_helper_name(dtype)
        code = f"fn {helper_name}(s: {mojo_scalar_type}) -> {vector_type}:\n"
        code += f"    return {vector_type}(s, s, s, {pad_literal})\n\n"
        return code

    def vec3_splat_helper_name(self, dtype):
        return f"_crossgl_vec3_splat_{MOJO_DTYPE_SUFFIX[dtype]}"

    def generate_swizzle_helper(self, dtype, source_width, member):
        _, _, pad_literal = MOJO_DTYPE_INFO[dtype]
        swizzle_indices = self.get_swizzle_indices(member)
        result_width = 2 if len(swizzle_indices) == 2 else 4
        source_type = f"SIMD[{dtype}, {source_width}]"
        result_type = f"SIMD[{dtype}, {result_width}]"
        helper_name = self.swizzle_helper_name(dtype, source_width, member)
        components = [f"v[{index}]" for index in swizzle_indices]
        if len(swizzle_indices) == 3:
            components.append(pad_literal)

        code = f"fn {helper_name}(v: {source_type}) -> {result_type}:\n"
        code += f"    return {result_type}({', '.join(components)})\n\n"
        return code

    def swizzle_helper_name(self, dtype, source_width, member):
        dtype_suffix = MOJO_DTYPE_SUFFIX[dtype]
        return f"_crossgl_swizzle_{dtype_suffix}_{source_width}_{member}"

    def generate_constructor_helper_call(self, func_name, args):
        dtype, source_width, storage_width, pad_literal = self.vector_constructor_info[
            func_name
        ]
        pieces = []
        has_duplicate_sensitive_vector = False
        component_count = 0

        for arg in args:
            piece = self.constructor_piece_for_expression(arg, dtype)
            if piece is None:
                return None
            pieces.append(piece)
            if piece["kind"] == "vector":
                component_count += len(piece["indices"])
                has_duplicate_sensitive_vector = (
                    has_duplicate_sensitive_vector or piece["duplicate_sensitive"]
                )
            else:
                component_count += 1

        if not has_duplicate_sensitive_vector or component_count != source_width:
            return None

        key = self.constructor_helper_key(dtype, source_width, storage_width, pieces)
        helper_name = self.constructor_helper_name(key)
        self.required_constructor_helpers[key] = {
            "key": key,
            "dtype": dtype,
            "storage_width": storage_width,
            "pad_literal": pad_literal,
            "pieces": pieces,
        }

        call_args = [self.generate_expression(piece["expr"]) for piece in pieces]
        return f"{helper_name}({', '.join(call_args)})"

    def constructor_piece_for_expression(self, expr, target_dtype):
        if isinstance(expr, MemberAccessNode):
            swizzle_indices = self.get_swizzle_indices(expr.member)
            if swizzle_indices is not None:
                source_type = self.expression_result_type(expr.object)
                source_info = self.vector_type_info(source_type)
                if source_info is None or source_info[0] != target_dtype:
                    return None
                return {
                    "kind": "vector",
                    "dtype": source_info[0],
                    "storage_width": source_info[2],
                    "indices": tuple(swizzle_indices),
                    "expr": expr.object,
                    "duplicate_sensitive": self.is_duplicate_sensitive_expression(
                        expr.object
                    ),
                }

        expr_type = self.expression_result_type(expr)
        info = self.vector_type_info(expr_type)
        if info is not None:
            if info[0] != target_dtype:
                return None
            _, source_width, storage_width, _ = info
            return {
                "kind": "vector",
                "dtype": info[0],
                "storage_width": storage_width,
                "indices": tuple(range(source_width)),
                "expr": expr,
                "duplicate_sensitive": self.is_duplicate_sensitive_expression(expr),
            }

        return {"kind": "scalar", "expr": expr}

    def constructor_helper_key(self, dtype, source_width, storage_width, pieces):
        signature = []
        for piece in pieces:
            if piece["kind"] == "vector":
                signature.append(
                    (
                        "v",
                        piece["dtype"],
                        piece["storage_width"],
                        piece["indices"],
                    )
                )
            else:
                signature.append(("s",))
        return (dtype, source_width, storage_width, tuple(signature))

    def constructor_helper_name(self, key):
        dtype, _, storage_width, signature = key
        parts = []
        for piece in signature:
            if piece[0] == "v":
                _, piece_dtype, piece_storage_width, indices = piece
                index_text = "".join(str(index) for index in indices)
                parts.append(
                    f"v{MOJO_DTYPE_SUFFIX[piece_dtype]}{piece_storage_width}_{index_text}"
                )
            else:
                parts.append("s")
        suffix = "_".join(parts)
        return f"_crossgl_construct_{MOJO_DTYPE_SUFFIX[dtype]}_{storage_width}_{suffix}"

    def generate_constructor_helper(self, helper):
        dtype = helper["dtype"]
        scalar_type, _, _ = MOJO_DTYPE_INFO[dtype]
        mojo_scalar_type = self.map_type(scalar_type)
        result_type = f"SIMD[{dtype}, {helper['storage_width']}]"
        params = []
        components = []

        for index, piece in enumerate(helper["pieces"]):
            if piece["kind"] == "vector":
                param_name = f"v{index}"
                vector_type = f"SIMD[{piece['dtype']}, {piece['storage_width']}]"
                params.append(f"{param_name}: {vector_type}")
                components.extend(
                    f"{param_name}[{component_index}]"
                    for component_index in piece["indices"]
                )
            else:
                param_name = f"s{index}"
                params.append(f"{param_name}: {mojo_scalar_type}")
                components.append(param_name)

        if helper["pad_literal"] is not None and len(components) == 3:
            components.append(helper["pad_literal"])

        helper_name = self.constructor_helper_name(helper["key"])
        code = f"fn {helper_name}({', '.join(params)}) -> {result_type}:\n"
        code += f"    return {result_type}({', '.join(components)})\n\n"
        return code

    def register_variable_type(self, name, var_type):
        if name and var_type:
            self.variable_types[name] = self.type_name(var_type)

    def type_name(self, type_value):
        if hasattr(type_value, "name") or hasattr(type_value, "element_type"):
            return self.convert_type_node_to_string(type_value)
        return str(type_value)

    def expression_result_type(self, expr):
        if isinstance(expr, str):
            return self.variable_types.get(expr)
        if isinstance(expr, VariableNode) and hasattr(expr, "name"):
            return self.variable_types.get(expr.name)
        if isinstance(expr, ArrayAccessNode):
            return None
        if isinstance(expr, BinaryOpNode):
            left_type = self.expression_result_type(expr.left)
            right_type = self.expression_result_type(expr.right)
            left_info = self.vector_type_info(left_type)
            right_info = self.vector_type_info(right_type)
            if left_info is not None and right_info is not None:
                return left_type if left_info == right_info else left_type
            if left_info is not None:
                return left_type
            if right_info is not None:
                return right_type
            return left_type if left_type == right_type else left_type or right_type
        if isinstance(expr, FunctionCallNode):
            func_name = self.function_call_name(expr)
            if func_name in self.vector_constructor_info:
                return func_name
            return self.function_return_types.get(func_name)
        if isinstance(expr, MemberAccessNode):
            swizzle_indices = self.get_swizzle_indices(expr.member)
            if swizzle_indices is not None:
                obj_type = self.expression_result_type(expr.object)
                return self.swizzle_result_type(obj_type, len(swizzle_indices))

            obj_type = self.expression_result_type(expr.object)
            if obj_type in self.struct_types:
                return self.struct_types[obj_type].get(expr.member)
        if hasattr(expr, "__class__") and "Identifier" in str(expr.__class__):
            return self.variable_types.get(getattr(expr, "name", ""))
        if hasattr(expr, "__class__") and "Literal" in str(expr.__class__):
            literal_type = getattr(getattr(expr, "literal_type", None), "name", None)
            if literal_type:
                return literal_type
        return None

    def function_call_name(self, expr):
        func_expr = getattr(expr, "function", None)
        if func_expr is None:
            func_expr = expr.name
        if hasattr(func_expr, "name"):
            return func_expr.name
        if isinstance(func_expr, str):
            return func_expr
        return None

    def vector_type_info(self, type_name):
        if type_name in self.vector_constructor_info:
            return self.vector_constructor_info[type_name]
        return None

    def swizzle_result_type(self, obj_type, component_count):
        info = self.vector_type_info(obj_type)
        dtype = info[0] if info else "DType.float32"
        scalar_type, prefix, _ = MOJO_DTYPE_INFO.get(
            dtype, MOJO_DTYPE_INFO["DType.float32"]
        )
        if component_count == 1:
            return scalar_type
        return f"{prefix}{component_count}"

    def get_swizzle_indices(self, member):
        if not member:
            return None
        for components in SWIZZLE_SETS.values():
            if all(component in components for component in member):
                return [components[component] for component in member]
        return None

    def vector_components_for_expression(self, expr):
        if isinstance(expr, MemberAccessNode):
            obj = self.generate_expression(expr.object)
            swizzle_indices = self.get_swizzle_indices(expr.member)
            if swizzle_indices is not None:
                return [f"{obj}[{index}]" for index in swizzle_indices]

        expr_type = self.expression_result_type(expr)
        info = self.vector_type_info(expr_type)
        if info is None:
            return None

        _, source_width, _, _ = info
        if source_width <= 1:
            return None

        expr_text = self.generate_expression(expr)
        return [f"{expr_text}[{index}]" for index in range(source_width)]

    def generate_swizzle(self, source_expr, obj, obj_type, member, swizzle_indices):
        if len(swizzle_indices) == 1:
            return f"{obj}[{swizzle_indices[0]}]"

        info = self.vector_type_info(obj_type)
        dtype = info[0] if info else "DType.float32"
        source_width = info[2] if info else 4
        if info is not None and self.is_duplicate_sensitive_expression(source_expr):
            helper_name = self.swizzle_helper_name(dtype, source_width, member)
            self.required_swizzle_helpers.add((dtype, source_width, member))
            return f"{helper_name}({obj})"

        _, _, pad_literal = MOJO_DTYPE_INFO.get(dtype, MOJO_DTYPE_INFO["DType.float32"])
        storage_width = 2 if len(swizzle_indices) == 2 else 4
        components = [f"{obj}[{index}]" for index in swizzle_indices]
        if len(swizzle_indices) == 3:
            components.append(pad_literal)

        return f"SIMD[{dtype}, {storage_width}]({', '.join(components)})"

    def is_duplicate_sensitive_expression(self, expr):
        if isinstance(expr, (FunctionCallNode, BinaryOpNode, TernaryOpNode)):
            return True
        if isinstance(expr, UnaryOpNode):
            return self.is_duplicate_sensitive_expression(expr.operand)
        if isinstance(expr, MemberAccessNode):
            return self.is_duplicate_sensitive_expression(expr.object)
        if isinstance(expr, ArrayAccessNode):
            return self.is_duplicate_sensitive_expression(
                expr.array
            ) or self.is_duplicate_sensitive_expression(expr.index)
        return False

    def format_literal(self, value, literal_type=None):
        if isinstance(value, bool):
            return "True" if value else "False"
        if literal_type == "bool" and isinstance(value, str):
            lower_value = value.lower()
            if lower_value == "true":
                return "True"
            if lower_value == "false":
                return "False"
        if isinstance(value, str):
            escaped = self.escape_literal(value)
            return f'"{escaped}"'
        return str(value)

    def escape_literal(self, value):
        text = str(value)
        escaped = []
        for index, char in enumerate(text):
            if char == "\n":
                escaped.append("\\n")
            elif char == "\r":
                escaped.append("\\r")
            elif char == "\t":
                escaped.append("\\t")
            elif char == '"' and (index == 0 or text[index - 1] != "\\"):
                escaped.append('\\"')
            else:
                escaped.append(char)
        return "".join(escaped)

    def map_type(self, vtype):
        if vtype is None:
            return "Float32"

        if hasattr(vtype, "name") or hasattr(vtype, "element_type"):
            vtype_str = self.convert_type_node_to_string(vtype)
        else:
            vtype_str = str(vtype)

        if "[" in vtype_str and "]" in vtype_str:
            base_type, size = parse_array_type(vtype_str)
            base_mapped = self.type_mapping.get(base_type, base_type)
            if size:
                return f"StaticTuple[{base_mapped}, {size}]"
            else:
                return f"DynamicVector[{base_mapped}]"

        return self.type_mapping.get(vtype_str, vtype_str)

    def map_operator(self, op):
        op_map = {
            "PLUS": "+",
            "MINUS": "-",
            "MULTIPLY": "*",
            "DIVIDE": "/",
            "BITWISE_XOR": "^",
            "BITWISE_OR": "|",
            "BITWISE_AND": "&",
            "LESS_THAN": "<",
            "GREATER_THAN": ">",
            "ASSIGN_ADD": "+=",
            "ASSIGN_SUB": "-=",
            "ASSIGN_MUL": "*=",
            "ASSIGN_DIV": "/=",
            "ASSIGN_MOD": "%=",
            "ASSIGN_XOR": "^=",
            "ASSIGN_OR": "|=",
            "ASSIGN_AND": "&=",
            "LESS_EQUAL": "<=",
            "GREATER_EQUAL": ">=",
            "EQUAL": "==",
            "NOT_EQUAL": "!=",
            "AND": "and",
            "OR": "or",
            "EQUALS": "=",
            "ASSIGN_SHIFT_LEFT": "<<=",
            "ASSIGN_SHIFT_RIGHT": ">>=",
            "LOGICAL_AND": "and",
            "LOGICAL_OR": "or",
            "BITWISE_SHIFT_RIGHT": ">>",
            "BITWISE_SHIFT_LEFT": "<<",
            "MOD": "%",
            "NOT": "not",
        }
        return op_map.get(op, op)

    def map_semantic(self, semantic):
        if semantic:
            return self.semantic_map.get(semantic, semantic)
        return ""

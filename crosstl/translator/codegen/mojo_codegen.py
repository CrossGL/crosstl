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


class MojoCodeGen:
    def __init__(self):
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
            # Vector Types
            "vec2": "SIMD[DType.float32, 2]",
            "vec3": "SIMD[DType.float32, 3]",
            "vec4": "SIMD[DType.float32, 4]",
            "ivec2": "SIMD[DType.int32, 2]",
            "ivec3": "SIMD[DType.int32, 3]",
            "ivec4": "SIMD[DType.int32, 4]",
            "uvec2": "SIMD[DType.uint32, 2]",
            "uvec3": "SIMD[DType.uint32, 3]",
            "uvec4": "SIMD[DType.uint32, 4]",
            "bvec2": "SIMD[DType.bool, 2]",
            "bvec3": "SIMD[DType.bool, 3]",
            "bvec4": "SIMD[DType.bool, 4]",
            # Matrix Types
            "mat2": "Matrix[DType.float32, 2, 2]",
            "mat3": "Matrix[DType.float32, 3, 3]",
            "mat4": "Matrix[DType.float32, 4, 4]",
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
        code = "# Generated Mojo Shader Code\n"
        code += "from math import *\n"
        code += "from simd import *\n"
        code += "from gpu import *\n\n"

        # Generate structs
        for node in ast.structs:
            if isinstance(node, StructNode):
                code += self.generate_struct(node)

        # Generate global variables
        for node in ast.global_variables:
            if isinstance(node, ArrayNode):
                code += self.generate_array_declaration(node)
            else:
                code += f"var {node.name}: {self.map_type(node.vtype)}\n"

        # Generate cbuffers as structs
        if ast.cbuffers:
            code += "# Constant Buffers\n"
            code += self.generate_cbuffers(ast)

        # Generate functions
        for func in ast.functions:
            if func.qualifier == "vertex":
                code += "# Vertex Shader\n"
                code += self.generate_function(func, shader_type="vertex")
            elif func.qualifier == "fragment":
                code += "# Fragment Shader\n"
                code += self.generate_function(func, shader_type="fragment")
            elif func.qualifier == "compute":
                code += "# Compute Shader\n"
                code += self.generate_function(func, shader_type="compute")
            else:
                code += self.generate_function(func)

        return code

    def generate_struct(self, node):
        code = f"@value\nstruct {node.name}:\n"

        # Generate struct members
        for member in node.members:
            if isinstance(member, ArrayNode):
                if member.size:
                    code += f"    var {member.name}: StaticTuple[{self.map_type(member.element_type)}, {member.size}]\n"
                else:
                    code += f"    var {member.name}: DynamicVector[{self.map_type(member.element_type)}]\n"
            else:
                semantic = (
                    f"  # {self.map_semantic(member.semantic)}"
                    if member.semantic
                    else ""
                )
                code += (
                    f"    var {member.name}: {self.map_type(member.vtype)}{semantic}\n"
                )

        code += "\n"
        return code

    def generate_cbuffers(self, ast):
        code = ""
        for node in ast.cbuffers:
            if isinstance(node, StructNode):
                code += f"@value\nstruct {node.name}:\n"
                for member in node.members:
                    if isinstance(member, ArrayNode):
                        if member.size:
                            code += f"    var {member.name}: StaticTuple[{self.map_type(member.element_type)}, {member.size}]\n"
                        else:
                            code += f"    var {member.name}: DynamicVector[{self.map_type(member.element_type)}]\n"
                    else:
                        code += (
                            f"    var {member.name}: {self.map_type(member.vtype)}\n"
                        )
                code += "\n"
            elif hasattr(node, "name") and hasattr(node, "members"):  # CbufferNode
                code += f"@value\nstruct {node.name}:\n"
                for member in node.members:
                    if isinstance(member, ArrayNode):
                        if member.size:
                            code += f"    var {member.name}: StaticTuple[{self.map_type(member.element_type)}, {member.size}]\n"
                        else:
                            code += f"    var {member.name}: DynamicVector[{self.map_type(member.element_type)}]\n"
                    else:
                        code += (
                            f"    var {member.name}: {self.map_type(member.vtype)}\n"
                        )
                code += "\n"
        return code

    def generate_function(self, func, indent=0, shader_type=None):
        code = ""
        "    " * indent

        # Generate function parameters
        params = []
        for p in func.params:
            param_semantic = (
                f"  # {self.map_semantic(p.semantic)}" if p.semantic else ""
            )
            params.append(f"{p.name}: {self.map_type(p.vtype)}{param_semantic}")

        params_str = ", ".join(params) if params else ""
        return_type = self.map_type(func.return_type) if func.return_type else "None"

        # Add shader type decorators for Mojo GPU programming
        if shader_type == "vertex":
            code += f"@vertex_shader\n"
        elif shader_type == "fragment":
            code += f"@fragment_shader\n"
        elif shader_type == "compute":
            code += f"@compute_shader\n"

        code += f"fn {func.name}({params_str}) -> {return_type}:\n"

        # Generate function body
        if func.body:
            for stmt in func.body:
                code += self.generate_statement(stmt, indent + 1)
        else:
            code += "    pass\n"

        code += "\n"
        return code

    def generate_statement(self, stmt, indent=0):
        indent_str = "    " * indent

        if isinstance(stmt, VariableNode):
            return f"{indent_str}var {stmt.name}: {self.map_type(stmt.vtype)}\n"
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
        else:
            return f"{indent_str}{self.generate_expression(stmt)}\n"

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
        condition = self.generate_expression(node.if_condition)
        code = f"{indent_str}if {condition}:\n"

        # Generate if body
        for stmt in node.if_body:
            code += self.generate_statement(stmt, indent + 1)

        # Generate else if conditions
        if hasattr(node, "else_if_conditions") and node.else_if_conditions:
            for else_if_condition, else_if_body in zip(
                node.else_if_conditions, node.else_if_bodies
            ):
                condition = self.generate_expression(else_if_condition)
                code += f"{indent_str}elif {condition}:\n"
                for stmt in else_if_body:
                    code += self.generate_statement(stmt, indent + 1)

        # Generate else body
        if node.else_body:
            code += f"{indent_str}else:\n"
            for stmt in node.else_body:
                code += self.generate_statement(stmt, indent + 1)

        return code

    def generate_for(self, node, indent):
        indent_str = "    " * indent

        # Extract init, condition, and update
        init = self.generate_statement(node.init, 0).strip()
        condition = self.generate_expression(node.condition)
        update = self.generate_expression(node.update)

        # In Mojo, we'll use a while loop for C-style for loops
        code = f"{indent_str}{init}\n"
        code += f"{indent_str}while {condition}:\n"

        # Generate loop body
        for stmt in node.body:
            code += self.generate_statement(stmt, indent + 1)

        # Add update at the end of the loop
        code += f"{indent_str}    {update}\n"

        return code

    def generate_expression(self, expr):
        if isinstance(expr, str):
            return expr
        elif isinstance(expr, (int, float, bool)):
            return str(expr)
        elif isinstance(expr, VariableNode):
            if hasattr(expr, "vtype") and expr.vtype and expr.name:
                return f"{expr.name}"
            elif hasattr(expr, "name"):
                return expr.name
            else:
                return str(expr)
        elif isinstance(expr, BinaryOpNode):
            left = self.generate_expression(expr.left)
            right = self.generate_expression(expr.right)
            op = self.map_operator(expr.op)
            return f"({left} {op} {right})"
        elif isinstance(expr, AssignmentNode):
            return self.generate_assignment(expr)
        elif isinstance(expr, UnaryOpNode):
            operand = self.generate_expression(expr.operand)
            op = self.map_operator(expr.op)
            return f"({op}{operand})"
        elif isinstance(expr, ArrayAccessNode):
            array = self.generate_expression(expr.array)
            index = self.generate_expression(expr.index)
            return f"{array}[{index}]"
        elif isinstance(expr, FunctionCallNode):
            # Map function names to Mojo equivalents
            func_name = self.function_map.get(expr.name, expr.name)

            # Handle vector constructors
            if expr.name in [
                "vec2",
                "vec3",
                "vec4",
                "ivec2",
                "ivec3",
                "ivec4",
                "uvec2",
                "uvec3",
                "uvec4",
            ]:
                mojo_type = self.map_type(expr.name)
                args = ", ".join(self.generate_expression(arg) for arg in expr.args)
                return f"{mojo_type}({args})"

            # Handle standard function calls
            args = ", ".join(self.generate_expression(arg) for arg in expr.args)
            return f"{func_name}({args})"
        elif isinstance(expr, MemberAccessNode):
            obj = self.generate_expression(expr.object)
            return f"{obj}.{expr.member}"
        elif isinstance(expr, TernaryOpNode):
            condition = self.generate_expression(expr.condition)
            true_expr = self.generate_expression(expr.true_expr)
            false_expr = self.generate_expression(expr.false_expr)
            return f"({true_expr} if {condition} else {false_expr})"
        else:
            return str(expr)

    def map_type(self, vtype):
        if vtype:
            # Handle array types
            if "[" in vtype and "]" in vtype:
                base_type, size = parse_array_type(vtype)
                base_mapped = self.type_mapping.get(base_type, base_type)
                if size:
                    return f"StaticTuple[{base_mapped}, {size}]"
                else:
                    return f"DynamicVector[{base_mapped}]"

            # Use regular type mapping
            return self.type_mapping.get(vtype, vtype)
        return vtype

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

"""Reverse code generator that emits CrossGL from Slang AST nodes."""

from .SlangAst import *
from .SlangParser import *
from .SlangLexer import *


class SlangToCrossGLConverter:
    """Serialize Slang backend AST nodes back into CrossGL source."""

    BINARY_PRECEDENCE = {
        "||": 1,
        "&&": 2,
        "|": 3,
        "^": 4,
        "&": 5,
        "==": 6,
        "!=": 6,
        "<": 7,
        ">": 7,
        "<=": 7,
        ">=": 7,
        "<<": 8,
        ">>": 8,
        "+": 9,
        "-": 9,
        "*": 10,
        "/": 10,
        "%": 10,
    }
    ASSOCIATIVE_BINARY_OPS = {"+", "*", "&&", "||", "&", "|", "^"}

    def __init__(self):
        """Initialize Slang-to-CrossGL type, semantic, and resource mappings."""
        self.vertex_inputs = []
        self.vertex_outputs = []
        self.fragment_inputs = []
        self.fragment_outputs = []
        self.cbuffers = []
        self.type_map = {
            "void": "void",
            "float2": "vec2",
            "float3": "vec3",
            "float4": "vec4",
            "float2x2": "mat2",
            "float3x3": "mat3",
            "float4x4": "mat4",
            "int": "int",
            "int2": "ivec2",
            "int3": "ivec3",
            "int4": "ivec4",
            "uint": "uint",
            "uint2": "uvec2",
            "uint3": "uvec3",
            "uint4": "uvec4",
            "bool": "bool",
            "bool2": "bvec2",
            "bool3": "bvec3",
            "bool4": "bvec4",
            "float": "float",
            "double": "double",
            "Texture1D": "sampler1D",
            "Texture2D": "sampler2D",
            "Texture2DArray": "sampler2DArray",
            "Texture2DMS": "sampler2DMS",
            "Texture3D": "sampler3D",
            "TextureCube": "samplerCube",
            "TextureCubeArray": "samplerCubeArray",
        }
        self.function_map = {
            "frac": "fract",
            "fmod": "mod",
            "lerp": "mix",
            "rsqrt": "inversesqrt",
        }
        self.user_function_names = set()

        self.semantic_map = {
            # Vertex inputs position
            "POSITION": "in_Position",
            "POSITION0": "in_Position0",
            "POSITION1": "in_Position1",
            "POSITION2": "in_Position2",
            "POSITION3": "in_Position3",
            "POSITION4": "in_Position4",
            "POSITION5": "in_Position5",
            "POSITION6": "in_Position6",
            "POSITION7": "in_Position7",
            # Vertex inputs normal
            "NORMAL": "in_Normal",
            "NORMAL0": "in_Normal0",
            "NORMAL1": "in_Normal1",
            "NORMAL2": "in_Normal2",
            "NORMAL3": "in_Normal3",
            "NORMAL4": "in_Normal4",
            "NORMAL5": "in_Normal5",
            "NORMAL6": "in_Normal6",
            "NORMAL7": "in_Normal7",
            # Vertex inputs tangent
            "TANGENT": "in_Tangent",
            "TANGENT0": "in_Tangent0",
            "TANGENT1": "in_Tangent1",
            "TANGENT2": "in_Tangent2",
            "TANGENT3": "in_Tangent3",
            "TANGENT4": "in_Tangent4",
            "TANGENT5": "in_Tangent5",
            "TANGENT6": "in_Tangent6",
            "TANGENT7": "in_Tangent7",
            # Vertex inputs binormal
            "BINORMAL": "in_Binormal",
            "BINORMAL0": "in_Binormal0",
            "BINORMAL1": "in_Binormal1",
            "BINORMAL2": "in_Binormal2",
            "BINORMAL3": "in_Binormal3",
            "BINORMAL4": "in_Binormal4",
            "BINORMAL5": "in_Binormal5",
            "BINORMAL6": "in_Binormal6",
            "BINORMAL7": "in_Binormal7",
            # Vertex inputs color
            "COLOR": "Color",
            "COLOR0": "Color0",
            "COLOR1": "Color1",
            "COLOR2": "Color2",
            "COLOR3": "Color3",
            "COLOR4": "Color4",
            "COLOR5": "Color5",
            "COLOR6": "Color6",
            "COLOR7": "Color7",
            # Vertex inputs texcoord
            "TEXCOORD": "TexCoord",
            "TEXCOORD0": "TexCoord0",
            "TEXCOORD1": "TexCoord1",
            "TEXCOORD2": "TexCoord2",
            "TEXCOORD3": "TexCoord3",
            "TEXCOORD4": "TexCoord4",
            "TEXCOORD5": "TexCoord5",
            "TEXCOORD6": "TexCoord6",
            # Vertex inputs instance
            "FRONT_FACE": "gl_IsFrontFace",
            "PRIMITIVE_ID": "gl_PrimitiveID",
            "INSTANCE_ID": "gl_InstanceID",
            "VERTEX_ID": "gl_VertexID",
            # Vertex outputs
            "SV_Position": "Out_Position",
            "SV_Position0": "Out_Position0",
            "SV_Position1": "Out_Position1",
            "SV_Position2": "Out_Position2",
            "SV_Position3": "Out_Position3",
            "SV_Position4": "Out_Position4",
            "SV_Position5": "Out_Position5",
            "SV_Position6": "Out_Position6",
            "SV_Position7": "Out_Position7",
            # Fragment inputs
            "SV_Target": "Out_Color",
            "SV_Target0": "Out_Color0",
            "SV_Target1": "Out_Color1",
            "SV_Target2": "Out_Color2",
            "SV_Target3": "Out_Color3",
            "SV_Target4": "Out_Color4",
            "SV_Target5": "Out_Color5",
            "SV_Target6": "Out_Color6",
            "SV_Target7": "Out_Color7",
            "SV_Depth": "Out_Depth",
            "SV_Depth0": "Out_Depth0",
            "SV_Depth1": "Out_Depth1",
            "SV_Depth2": "Out_Depth2",
            "SV_Depth3": "Out_Depth3",
            "SV_Depth4": "Out_Depth4",
            "SV_Depth5": "Out_Depth5",
            "SV_Depth6": "Out_Depth6",
            "SV_Depth7": "Out_Depth7",
        }

    def generate(self, ast):
        """Generate complete CrossGL source from a parsed Slang AST."""
        self.user_function_names = {
            getattr(func, "name", None) for func in getattr(ast, "functions", [])
        }
        self.user_function_names.discard(None)
        code = "shader main {\n"
        if ast.imports:
            for imp in ast.imports:
                code += f"    import {imp.module_name};\n"
            code += "\n"
        if ast.exports:
            for exp in ast.exports:
                code += f"    export {exp.item};\n"
            code += "\n"
        for node in ast.typedefs:
            code += (
                f"    typedef {self.map_type(node.original_type)} {node.new_type};\n"
            )
        for node in ast.structs:
            if isinstance(node, StructNode):
                code += f"    struct {node.name} {{\n"
                for member in node.members:
                    code += f"        {self.map_type(member.vtype)} {member.name} {self.map_semantic(member.semantic)};\n"
                code += "    }\n"
        for node in ast.global_vars:
            code += (
                f"    {self.map_type(node.vtype)} "
                f"{node.name}{self.format_array_suffixes(node)};\n"
            )
        if ast.cbuffers:
            code += "    // Constant Buffers\n"
            code += self.generate_cbuffers(ast)

        for func in ast.functions:
            if func.qualifier == "vertex":
                code += "    vertex {\n"
                code += self.generate_function(func)
                code += "    }\n\n"
            elif func.qualifier == "fragment":
                code += "    fragment {\n"
                code += self.generate_function(func)
                code += "    }\n\n"

            elif func.qualifier == "compute":
                code += "    compute {\n"
                code += self.generate_numthreads_layout(func)
                code += self.generate_function(func)
                code += "    }\n\n"
            else:
                code += self.generate_function(func)

        code += "}\n"
        return code

    def generate_numthreads_layout(self, func):
        numthreads = getattr(func, "numthreads", None)
        if not numthreads:
            return ""

        x, y, z = numthreads
        return (
            "        "
            f"layout(local_size_x = {x}, local_size_y = {y}, local_size_z = {z}) in;\n"
        )

    def generate_cbuffers(self, ast):
        code = ""
        for node in ast.cbuffers:
            if isinstance(node, StructNode):
                code += f"    cbuffer {node.name} {{\n"
                for member in node.members:
                    code += f"        {self.map_type(member.vtype)} {member.name};\n"
                code += "    }\n"
        return code

    def generate_function(self, func, indent=1):
        """Render one Slang function node as a CrossGL function."""
        code = " "
        code += "  " * indent
        params = ", ".join(self.generate_parameter(p) for p in func.params)
        semantic = self.map_semantic(func.semantic)
        semantic_suffix = f" {semantic}" if semantic else ""
        code += (
            f"    {self.map_type(func.return_type)} "
            f"{func.name}({params}){semantic_suffix} {{\n"
        )
        code += self.generate_function_body(func.body, indent=indent + 1)
        code += "    }\n\n"
        return code

    def generate_parameter(self, param):
        parameter = f"{self.map_type(param.vtype)} {param.name}"
        semantic = self.map_semantic(param.semantic)
        if semantic:
            parameter += f" {semantic}"
        return parameter

    def format_array_suffixes(self, node, is_main=False):
        sizes = getattr(node, "array_sizes", None)
        if not sizes:
            return ""
        parts = []
        for size in sizes:
            if size is None:
                parts.append("[]")
            else:
                parts.append(f"[{self.generate_expression(size, is_main)}]")
        return "".join(parts)

    def generate_function_body(self, body, indent=0, is_main=False):
        code = ""
        for stmt in body:
            code += "    " * indent
            if isinstance(stmt, VariableNode):
                code += f"{self.map_type(stmt.vtype)} {stmt.name};\n"
            elif isinstance(stmt, AssignmentNode):
                code += self.generate_assignment(stmt, is_main) + ";\n"
            elif isinstance(stmt, (FunctionCallNode, MethodCallNode, CallNode)):
                code += f"{self.generate_expression(stmt, is_main)};\n"
            elif isinstance(stmt, BinaryOpNode):
                code += f"{self.generate_expression(stmt, is_main)};\n"
            elif isinstance(stmt, ReturnNode):
                if not is_main:
                    if stmt.value is None:
                        code += "return;\n"
                    else:
                        code += (
                            f"return {self.generate_expression(stmt.value, is_main)};\n"
                        )
            elif isinstance(stmt, ForNode):
                code += self.generate_for_loop(stmt, indent, is_main)
            elif isinstance(stmt, WhileNode):
                code += self.generate_while_loop(stmt, indent, is_main)
            elif isinstance(stmt, DoWhileNode):
                code += self.generate_do_while_loop(stmt, indent, is_main)
            elif isinstance(stmt, SwitchNode):
                code += self.generate_switch_statement(stmt, indent, is_main)
            elif isinstance(stmt, IfNode):
                code += self.generate_if_statement(stmt, indent, is_main)
            elif isinstance(stmt, BreakNode):
                code += "break;\n"
            elif isinstance(stmt, ContinueNode):
                code += "continue;\n"
        return code

    def generate_for_loop(self, node, indent, is_main):
        init = self.generate_expression(node.init, is_main)
        condition = self.generate_expression(node.condition, is_main)
        update = self.generate_expression(node.update, is_main)

        code = f"for ({init}; {condition}; {update}) {{\n"
        code += self.generate_function_body(node.body, indent + 1, is_main)
        code += "    " * indent + "}\n"
        return code

    def generate_while_loop(self, node, indent, is_main):
        condition = self.generate_expression(node.condition, is_main)

        code = f"while ({condition}) {{\n"
        code += self.generate_function_body(node.body, indent + 1, is_main)
        code += "    " * indent + "}\n"
        return code

    def generate_do_while_loop(self, node, indent, is_main):
        condition = self.generate_expression(node.condition, is_main)

        code = "do {\n"
        code += self.generate_function_body(node.body, indent + 1, is_main)
        code += "    " * indent + f"}} while ({condition});\n"
        return code

    def generate_switch_statement(self, node, indent, is_main):
        expression = self.generate_expression(node.expression, is_main)

        code = f"switch ({expression}) {{\n"
        for case in node.cases:
            value = self.generate_expression(case.value, is_main)
            code += "    " * (indent + 1) + f"case {value}:\n"
            code += self.generate_function_body(case.body, indent + 2, is_main)

        if node.default_case is not None:
            code += "    " * (indent + 1) + "default:\n"
            code += self.generate_function_body(node.default_case, indent + 2, is_main)

        code += "    " * indent + "}\n"
        return code

    def generate_if_statement(self, node, indent, is_main):
        condition = self.generate_expression(node.condition, is_main)

        code = f"if ({condition}) {{\n"
        code += self.generate_function_body(node.if_body, indent + 1, is_main)
        code += "    " * indent + "}"

        if node.else_body:
            if isinstance(node.else_body, IfNode):
                code += " else "
                code += self.generate_if_statement(node.else_body, indent, is_main)
            else:
                code += " else {\n"
                code += self.generate_function_body(node.else_body, indent + 1, is_main)
                code += "    " * indent + "}"

        code += "\n"
        return code

    def generate_assignment(self, node, is_main):
        lhs = self.generate_expression(node.left, is_main)
        rhs = self.generate_expression(node.right, is_main)
        op = node.operator
        return f"{lhs} {op} {rhs}"

    def binary_precedence(self, op):
        return self.BINARY_PRECEDENCE.get(op, 0)

    def binary_child_needs_parentheses(self, parent_op, child, is_right_child=False):
        if not isinstance(child, BinaryOpNode):
            return False

        parent_precedence = self.binary_precedence(parent_op)
        child_precedence = self.binary_precedence(child.op)
        if child_precedence < parent_precedence:
            return True
        if child_precedence > parent_precedence:
            return False
        return is_right_child and (
            parent_op not in self.ASSOCIATIVE_BINARY_OPS or child.op != parent_op
        )

    def generate_binary_expression(self, expr, is_main):
        left = self.generate_expression(expr.left, is_main)
        right = self.generate_expression(expr.right, is_main)
        if self.binary_child_needs_parentheses(expr.op, expr.left):
            left = f"({left})"
        if self.binary_child_needs_parentheses(expr.op, expr.right, True):
            right = f"({right})"
        return f"{left} {expr.op} {right}"

    def generate_expression(self, expr, is_main=False):
        """Render a Slang backend expression node as CrossGL syntax."""
        if isinstance(expr, str):
            return expr
        elif isinstance(expr, VariableNode):
            if expr.vtype:
                return f"{self.map_type(expr.vtype)} {expr.name}"
            return expr.name
        elif isinstance(expr, BinaryOpNode):
            return self.generate_binary_expression(expr, is_main)
        elif isinstance(expr, AssignmentNode):
            left = self.generate_expression(expr.left, is_main)
            right = self.generate_expression(expr.right, is_main)
            return f"{left} {expr.operator} {right}"
        elif isinstance(expr, UnaryOpNode):
            operand = self.generate_expression(expr.operand, is_main)
            if isinstance(expr.operand, BinaryOpNode):
                operand = f"({operand})"
            return f"{expr.op}{operand}"
        elif isinstance(expr, FunctionCallNode):
            args = ", ".join(
                self.generate_expression(arg, is_main) for arg in expr.args
            )
            if (
                expr.name == "saturate"
                and len(expr.args) == 1
                and expr.name not in self.user_function_names
            ):
                return f"clamp({args}, 0.0, 1.0)"
            if expr.name in self.user_function_names:
                name = expr.name
            else:
                name = self.function_map.get(expr.name, expr.name)
            return f"{name}({args})"
        elif isinstance(expr, MethodCallNode):
            obj = self.generate_expression(expr.object, is_main)
            args = ", ".join(
                self.generate_expression(arg, is_main) for arg in expr.args
            )
            return f"{obj}.{expr.method}({args})"
        elif isinstance(expr, CallNode):
            callee = self.generate_expression(expr.callee, is_main)
            args = ", ".join(
                self.generate_expression(arg, is_main) for arg in expr.args
            )
            return f"{callee}({args})"
        elif isinstance(expr, MemberAccessNode):
            obj = self.generate_expression(expr.object, is_main)
            return f"{obj}.{expr.member}"
        elif isinstance(expr, ArrayAccessNode):
            array = self.generate_expression(expr.array, is_main)
            index = self.generate_expression(expr.index, is_main)
            return f"{array}[{index}]"
        elif isinstance(expr, TernaryOpNode):
            return f"{self.generate_expression(expr.condition, is_main)} ? {self.generate_expression(expr.true_expr, is_main)} : {self.generate_expression(expr.false_expr, is_main)}"
        elif isinstance(expr, VectorConstructorNode):
            args = ", ".join(
                self.generate_expression(arg, is_main) for arg in expr.args
            )
            return f"{self.map_type(expr.type_name)}({args})"
        else:
            return str(expr)

    def map_type(self, slang_type):
        """Map a Slang type name to the closest CrossGL type name."""
        if slang_type:
            slang_type = slang_type.strip()
            base_type = slang_type.split("<", 1)[0].strip()
            return self.type_map.get(
                slang_type, self.type_map.get(base_type, slang_type)
            )
        return slang_type

    def map_semantic(self, semantic):
        """Map a Slang semantic to CrossGL semantic annotation syntax."""
        if semantic is not None:
            return f"@ {self.semantic_map.get(semantic, semantic)}"
        else:
            return ""

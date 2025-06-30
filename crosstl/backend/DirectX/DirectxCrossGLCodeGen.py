from .DirectxAst import *
from .DirectxParser import *
from .DirectxLexer import *


class HLSLToCrossGLConverter:
    def __init__(self):
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
            "half": "float16_t",
            "half2": "f16vec2",
            "half3": "f16vec3",
            "half4": "f16vec4",
            "double2": "dvec2",
            "double3": "dvec3",
            "double4": "dvec4",
            "Texture2D": "sampler2D",
            "TextureCube": "samplerCube",
            "int64_t": "int64_t",
            "uint64_t": "uint64_t",
        }
        self.semantic_map = {
            # Vertex inputs instance
            "FRONT_FACE": "gl_IsFrontFace",
            "PRIMITIVE_ID": "gl_PrimitiveID",
            "INSTANCE_ID": "InstanceID",
            "VERTEX_ID": "VertexID",
            "SV_InstanceID": "gl_InstanceID",
            "SV_VertexID": "gl_VertexID",
            # Vertex outputs
            "SV_POSITION": "gl_Position",
            # Fragment inputs
            "SV_TARGET": "gl_FragColor",
            "SV_TARGET0": "gl_FragColor0",
            "SV_TARGET1": "gl_FragColor1",
            "SV_TARGET2": "gl_FragColor2",
            "SV_TARGET3": "gl_FragColor3",
            "SV_TARGET4": "gl_FragColor4",
            "SV_TARGET5": "gl_FragColor5",
            "SV_TARGET6": "gl_FragColor6",
            "SV_TARGET7": "gl_FragColor7",
            "SV_DEPTH": "gl_FragDepth",
            "SV_DEPTH0": "gl_FragDepth0",
            "SV_DEPTH1": "gl_FragDepth1",
            "SV_DEPTH2": "gl_FragDepth2",
            "SV_DEPTH3": "gl_FragDepth3",
            "SV_DEPTH4": "gl_FragDepth4",
            "SV_DEPTH5": "gl_FragDepth5",
            "SV_DEPTH6": "gl_FragDepth6",
            "SV_DEPTH7": "gl_FragDepth7",
            # Additional mappings
            "POSITION": "Position",
            "NORMAL": "Normal",
            "TANGENT": "Tangent",
            "BINORMAL": "Binormal",
            "TEXCOORD": "TexCoord",
            "TEXCOORD0": "TexCoord0",
            "TEXCOORD1": "TexCoord1",
            "TEXCOORD2": "TexCoord2",
            "TEXCOORD3": "TexCoord3",
            "SV_IsFrontFace": "gl_IsFrontFace",
            "SV_PrimitiveID": "gl_PrimitiveID",
            "SV_PointCoord": "gl_PointCoord",
        }
        self.bitwise_op_map = {
            "&": "bitAnd",
            "|": "bitOr",
            "^": "bitXor",
            "~": "bitNot",
            "<<": "bitShiftLeft",
            ">>": "bitShiftRight",
        }
        self.indentation = 0
        self.code = []

    def get_indent(self):
        return "    " * self.indentation

    def visit(self, node):
        # Special case for SwitchStatementNode and SwitchCaseNode
        if isinstance(node, SwitchStatementNode):
            return self.visit_SwitchStatementNode(node)
        elif isinstance(node, SwitchCaseNode):
            return self.visit_SwitchCaseNode(node)
        elif isinstance(node, StructNode):
            return self.visit_StructNode(node)
        elif isinstance(node, BinaryOpNode):
            return self.visit_BinaryOpNode(node)
        elif isinstance(node, UnaryOpNode):
            return self.visit_UnaryOpNode(node)

        # For other node types, use existing methods
        if hasattr(self, f"generate_{type(node).__name__}"):
            method = getattr(self, f"generate_{type(node).__name__}")
            return method(node)
        return self.generate_expression(node)

    def generate(self, ast):
        code = "shader main {\n"
        # Generate structs
        for node in ast.structs:
            if isinstance(node, StructNode):
                code += f"    struct {node.name} {{\n"
                for member in node.members:
                    code += f"        {self.map_type(member.vtype)} {member.name} {self.map_semantic(member.semantic)};\n"
                code += "    }\n"
            elif isinstance(node, PragmaNode):
                code += f"    #pragma {node.directive} {node.value};\n"
            elif isinstance(node, IncludeNode):
                code += f"    #include {node.path}\n"
        # Generate global variables
        for node in ast.global_variables:
            code += f"    {self.map_type(node.vtype)} {node.name};\n"
        # Generate cbuffers
        if ast.cbuffers:
            code += "    // Constant Buffers\n"
            code += self.generate_cbuffers(ast)

        # Generate custom functions
        for func in ast.functions:
            if func.qualifier == "vertex":
                code += "    // Vertex Shader\n"
                code += "    vertex {\n"
                code += self.generate_function(func)
                code += "    }\n\n"
            elif func.qualifier == "fragment":
                code += "    // Fragment Shader\n"
                code += "    fragment {\n"
                code += self.generate_function(func)
                code += "    }\n\n"

            elif func.qualifier == "compute":
                code += "    // Compute Shader\n"
                code += "    compute {\n"
                code += self.generate_function(func)
                code += "    }\n\n"
            else:
                code += self.generate_function(func)

        code += "}\n"
        return code

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
        code = " "
        code += "  " * indent
        params = ", ".join(
            f"{self.map_type(p.vtype)} {p.name} {self.map_semantic(p.semantic)}"
            for p in func.params
        )
        code += f"    {self.map_type(func.return_type)} {func.name}({params}) {self.map_semantic(func.semantic)} {{\n"
        code += self.generate_function_body(func.body, indent=indent + 1)
        code += "    }\n\n"
        return code

    def generate_function_body(self, body, indent=0, is_main=False):
        code = ""
        for stmt in body:
            code += "    " * indent
            if isinstance(stmt, VariableNode):
                code += f"{self.map_type(stmt.vtype)} {stmt.name};\n"
            elif isinstance(stmt, AssignmentNode):
                code += self.generate_assignment(stmt, is_main) + ";\n"

            elif isinstance(stmt, BinaryOpNode):
                code += f"{self.generate_expression(stmt.left, is_main)} {stmt.op} {self.generate_expression(stmt.right, is_main)};\n"
            elif isinstance(stmt, ReturnNode):
                if not is_main:
                    code += f"return {self.generate_expression(stmt.value, is_main)};\n"
            elif isinstance(stmt, ForNode):
                code += self.generate_for_loop(stmt, indent, is_main)
            elif isinstance(stmt, WhileNode):
                code += self.generate_while_loop(stmt, indent, is_main)
            elif isinstance(stmt, DoWhileNode):
                code += self.generate_do_while_loop(stmt, indent, is_main)
            elif isinstance(stmt, IfNode):
                code += self.generate_if_statement(stmt, indent, is_main)
            elif isinstance(stmt, SwitchNode):
                code += self.generate_switch_statement(stmt, indent, is_main)
            elif isinstance(stmt, FunctionCallNode):
                code += f"{self.generate_expression(stmt, is_main)};\n"
            elif isinstance(stmt, str):
                code += f"{stmt};\n"
            else:
                # For any unhandled statement type
                code += f"// Unhandled statement type: {type(stmt).__name__}\n"
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
        code += "    " * indent + "} "
        code += f"while ({condition});\n"
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

    def generate_expression(self, expr, is_main=False):
        if isinstance(expr, str):
            return expr
        elif isinstance(expr, VariableNode):
            if expr.vtype:
                return f"{self.map_type(expr.vtype)} {expr.name}"
            else:
                return expr.name
        elif isinstance(expr, BinaryOpNode):
            left = self.generate_expression(expr.left, is_main)
            right = self.generate_expression(expr.right, is_main)
            if expr.op in self.bitwise_op_map:
                # Use the appropriate function for bitwise operations
                func = self.bitwise_op_map.get(expr.op)
                return f"{func}({left}, {right})"
            return f"{left} {expr.op} {right}"

        elif isinstance(expr, AssignmentNode):
            left = self.generate_expression(expr.left, is_main)
            right = self.generate_expression(expr.right, is_main)
            op = expr.operator
            # Handle special assignment operators that might involve bitwise operations
            if op in ["&=", "|=", "^=", "<<=", ">>="]:
                base_op = op[0:-1]  # Remove the '=' character
                if base_op in self.bitwise_op_map:
                    func = self.bitwise_op_map.get(base_op)
                    return f"{left} = {func}({left}, {right})"
            return f"{left} {op} {right}"

        elif isinstance(expr, UnaryOpNode):
            operand = self.generate_expression(expr.operand, is_main)
            if expr.op in self.bitwise_op_map:
                func = self.bitwise_op_map.get(expr.op)
                return f"{func}({operand})"
            return f"{expr.op}{operand}"
        elif isinstance(expr, FunctionCallNode):
            args = ", ".join(
                self.generate_expression(arg, is_main) for arg in expr.args
            )
            return f"{expr.name}({args})"
        elif isinstance(expr, MemberAccessNode):
            obj = self.generate_expression(expr.object, is_main)
            return f"{obj}.{expr.member}"

        elif isinstance(expr, TernaryOpNode):
            return f"{self.generate_expression(expr.condition, is_main)} ? {self.generate_expression(expr.true_expr, is_main)} : {self.generate_expression(expr.false_expr, is_main)}"

        elif isinstance(expr, VectorConstructorNode):
            args = ", ".join(
                self.generate_expression(arg, is_main) for arg in expr.args
            )
            return f"{self.map_type(expr.type_name)}({args})"
        elif isinstance(expr, float) or isinstance(expr, int) or isinstance(expr, bool):
            return str(expr)
        else:
            return str(expr)

    def map_type(self, hlsl_type):
        if hlsl_type:
            return self.type_map.get(hlsl_type, hlsl_type)
        return hlsl_type

    def map_semantic(self, semantic):
        if semantic is not None:
            mapped_semantic = self.semantic_map.get(semantic, semantic)
            return f"@ {mapped_semantic}"
        else:
            return ""

    def generate_switch_statement(self, node, indent=1, is_main=False):
        code = (
            "    " * indent
            + f"switch ({self.generate_expression(node.condition, is_main)}) {{\n"
        )

        for case in node.cases:
            code += (
                "    " * (indent + 1)
                + f"case {self.generate_expression(case.value, is_main)}:\n"
            )
            code += self.generate_function_body(case.body, indent + 2, is_main)
            code += "    " * (indent + 2) + "break;\n"

        if node.default_body:
            code += "    " * (indent + 1) + "default:\n"
            code += self.generate_function_body(node.default_body, indent + 2, is_main)
            code += "    " * (indent + 2) + "break;\n"

        code += "    " * indent + "}\n"
        return code

    def visit_BinaryOpNode(self, node):
        if hasattr(node.left, "visit"):
            left = node.visit_child(self, node.left)
        else:
            left = self.generate_expression(node.left)

        if hasattr(node.right, "visit"):
            right = node.visit_child(self, node.right)
        else:
            right = self.generate_expression(node.right)

        # Handle bitwise operations based on token value
        if hasattr(node.op, "token_type"):
            if node.op.token_type in ("BITWISE_AND", "AMPERSAND", "&"):
                return f"({left} & {right})"
            elif node.op.token_type in ("BITWISE_OR", "PIPE", "|"):
                return f"({left} | {right})"
            elif node.op.token_type in ("BITWISE_XOR", "CARET", "^"):
                return f"({left} ^ {right})"
        elif hasattr(node.op, "value"):
            # Handle string values
            if node.op.value in ("&", "BITWISE_AND", "AMPERSAND"):
                return f"({left} & {right})"
            elif node.op.value in ("|", "BITWISE_OR", "PIPE"):
                return f"({left} | {right})"
            elif node.op.value in ("^", "BITWISE_XOR", "CARET"):
                return f"({left} ^ {right})"
        elif isinstance(node.op, str):
            # Direct string comparison
            if node.op in ("&", "BITWISE_AND", "AMPERSAND"):
                return f"({left} & {right})"
            elif node.op in ("|", "BITWISE_OR", "PIPE"):
                return f"({left} | {right})"
            elif node.op in ("^", "BITWISE_XOR", "CARET"):
                return f"({left} ^ {right})"

        # Falls back to string representation of the operator
        op_str = node.op.value if hasattr(node.op, "value") else str(node.op)
        return f"{left} {op_str} {right}"

    def visit_UnaryOpNode(self, node):
        if hasattr(node, "expr"):
            expr_node = node.expr
        else:
            expr_node = node.operand

        if hasattr(expr_node, "visit"):
            expr = node.visit_child(self, expr_node)
        else:
            expr = self.generate_expression(expr_node)

        # Handle bitwise NOT based on token type or value
        if hasattr(node.op, "token_type") and node.op.token_type in (
            "BITWISE_NOT",
            "TILDE",
            "~",
        ):
            return f"(~{expr})"
        elif hasattr(node.op, "value") and node.op.value in (
            "~",
            "BITWISE_NOT",
            "TILDE",
        ):
            return f"(~{expr})"
        elif isinstance(node.op, str) and node.op in ("~", "BITWISE_NOT", "TILDE"):
            return f"(~{expr})"

        # Falls back to string representation of the operator
        op_str = node.op.value if hasattr(node.op, "value") else str(node.op)
        return f"{op_str}{expr}"

    def visit_SwitchStatementNode(self, node):
        # Handle the alternative SwitchStatementNode type if needed
        return self.visit_SwitchNode(node)

    def visit_SwitchCaseNode(self, node):
        # Handle the alternative SwitchCaseNode type if needed
        return self.visit_CaseNode(node)

    def visit_StructNode(self, node):
        # Generate code for a struct definition
        code = f"struct {node.name} {{\n"
        self.indentation += 1

        for member in node.members:
            semantic = ""
            if member.semantic:
                semantic = f" @ {self.map_semantic(member.semantic)}"

            code += (
                self.get_indent()
                + f"{self.map_type(member.vtype)} {member.name}{semantic};\n"
            )

        self.indentation -= 1
        code += self.get_indent() + "}\n"
        return code

    def visit_SwitchNode(self, node):
        # Generate the switch statement code
        condition = self.generate_expression(node.condition)
        code = f"switch ({condition}) {{\n"

        # Generate case statements
        for case in node.cases:
            code += self.visit_CaseNode(case)

        # Generate default case if exists
        if node.default_body:
            code += self.get_indent() + "default:\n"
            self.indentation += 1
            for stmt in node.default_body:
                code += self.get_indent() + self.generate_statement(stmt) + "\n"
            code += self.get_indent() + "break;\n"
            self.indentation -= 1

        code += self.get_indent() + "}\n"
        return code

    def visit_CaseNode(self, node):
        # Generate a case statement
        value = self.generate_expression(node.value)
        code = self.get_indent() + f"case {value}:\n"

        # Generate the case body
        self.indentation += 1
        for stmt in node.body:
            code += self.get_indent() + self.generate_statement(stmt) + "\n"
        code += self.get_indent() + "break;\n"
        self.indentation -= 1

        return code

    def generate_statement(self, node):
        """Generate a statement in CrossGL syntax"""
        if isinstance(node, str):
            return node
        elif hasattr(self, f"visit_{type(node).__name__}"):
            method = getattr(self, f"visit_{type(node).__name__}")
            return method(node)
        else:
            return self.generate_expression(node)

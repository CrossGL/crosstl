from ..ast import (
    AssignmentNode,
    BinaryOpNode,
    ForNode,
    FunctionCallNode,
    IfNode,
    MemberAccessNode,
    ReturnNode,
    StructNode,
    TernaryOpNode,
    UnaryOpNode,
    VariableNode,
)


class GLSLCodeGen:
    def __init__(self):
        self.semantic_map = {
            "gl_VertexID": "gl_VertexID",
            "gl_InstanceID": "gl_InstanceID",
            "gl_IsFrontFace": "gl_FrontFacing",
            "gl_PrimitiveID": "gl_PrimitiveID",
            "POSITION": "layout(location = 0)",
            "NORMAL": "layout(location = 1)",
            "TANGENT": "layout(location = 2)",
            "BINORMAL": "layout(location = 3)",
            "TEXCOORD": "layout(location = 4)",
            "TEXCOORD0": "layout(location = 5)",
            "TEXCOORD1": "layout(location = 6)",
            "TEXCOORD2": "layout(location = 7)",
            "TEXCOORD3": "layout(location = 8)",
            "TEXCOORD4": "layout(location = 9)",
            "TEXCOORD5": "layout(location = 10)",
            "TEXCOORD6": "layout(location = 11)",
            "TEXCOORD7": "layout(location = 12)",
            # Vertex outputs
            "gl_Position": "gl_Position",
            "gl_PointSize": "gl_PointSize",
            "gl_ClipDistance": "gl_ClipDistance",
            # Fragment outputs
            "gl_FragColor": "layout(location = 0)",
            "gl_FragColor1": "layout(location = 1)",
            "gl_FragColor2": "layout(location = 2)",
            "gl_FragColor3": "layout(location = 3)",
            "gl_FragColor4": "layout(location = 4)",
            "gl_FragColor5": "layout(location = 5)",
            "gl_FragColor6": "layout(location = 6)",
            "gl_FragColor7": "layout(location = 7)",
            "gl_FragDepth": "gl_FragDepth",
            # Additional fragment inputs
            "gl_FragCoord": "gl_FragCoord",
            "gl_FrontFacing": "gl_FrontFacing",
            "gl_PointCoord": "gl_PointCoord",
            # Compute shader specific
            "gl_GlobalInvocationID": "gl_GlobalInvocationID",
            "gl_LocalInvocationID": "gl_LocalInvocationID",
            "gl_WorkGroupID": "gl_WorkGroupID",
            "gl_LocalInvocationIndex": "gl_LocalInvocationIndex",
            "gl_WorkGroupSize": "gl_WorkGroupSize",
            "gl_NumWorkGroups": "gl_NumWorkGroups",
        }

    def generate(self, ast):
        code = "\n"
        code += "#version 450 core\n"
        # Generate structs
        for node in ast.structs:
            if isinstance(node, StructNode):
                if node.name == "VSInput":
                    for member in node.members:
                        code += f"{self.map_semantic(member.semantic)} in {member.vtype} {member.name};\n"
                elif node.name == "VSOutput":
                    for member in node.members:
                        code += f"out {member.vtype} {member.name};\n"
                elif node.name == "PSInput":
                    for member in node.members:
                        code += f"in {member.vtype} {member.name};\n"
                elif node.name == "PSOutput":
                    for member in node.members:
                        code += f"{self.map_semantic(member.semantic)} out {member.vtype} {member.name};\n"
                else:
                    code = ""
                    code += f"struct {node.name} {{\n"
                    for member in node.members:
                        code += f"    {self.map_type(member.vtype)} {member.name};\n"
                    code += "}\n"

        # Generate global variables
        for i, node in enumerate(ast.global_variables):
            code += f"layout(std140, binding = {i}) {self.map_type(node.vtype)} {node.name};\n"
        # Generate cbuffers
        if ast.cbuffers:
            code += "// Constant Buffers\n"
            code += self.generate_cbuffers(ast)

        # Generate custom functions
        for func in ast.functions:
            if func.qualifier == "vertex":
                code += "// Vertex Shader\n"
                code += self.generate_function(func, shader_type="vertex")
            elif func.qualifier == "fragment":
                code += "// Fragment Shader\n"
                code += self.generate_function(func, shader_type="fragment")

            elif func.qualifier == "compute":
                code += "// Compute Shader\n"
                code += self.generate_function(func, shader_type="compute")
            else:
                code += self.generate_function(func)

        return code

    def generate_cbuffers(self, ast):
        code = ""
        for i, node in enumerate(ast.cbuffers):
            if isinstance(node, StructNode):
                code += f"layout(std140, binding = {i}) uniform {node.name} {{\n"
                for member in node.members:
                    code += f"    {self.map_type(member.vtype)} {member.name};\n"
                code += "}\n"
        return code

    def generate_function(self, func, indent=0, shader_type=None):
        code = ""
        code += "  " * indent
        params = ", ".join(
            f"{self.map_type(p.vtype)} {p.name} {self.map_semantic(p.semantic)}"
            for p in func.params
        )
        if shader_type == "vertex":
            code += f"void main(){{\n"
        elif shader_type == "fragment":
            code += f"void main() {{\n"
        elif shader_type == "compute":
            code += f"void main() {{\n"
        else:
            code += f"{self.map_type(func.return_type)} {func.name}({params}) {{\n"

        for stmt in func.body:
            code += self.generate_statement(stmt, 1)
        code += "}\n\n"

        return code

    def generate_statement(self, stmt, indent=0, is_main=False):
        indent_str = "    " * indent
        if isinstance(stmt, VariableNode):
            return f"{indent_str}{self.map_type(stmt.vtype)} {stmt.name};\n"
        elif isinstance(stmt, AssignmentNode):
            return f"{indent_str}{self.generate_assignment(stmt, is_main)};\n"
        elif isinstance(stmt, IfNode):
            return self.generate_if(stmt, indent, is_main)
        elif isinstance(stmt, ForNode):
            return self.generate_for(stmt, indent, is_main)
        elif isinstance(stmt, ReturnNode):
            code = ""
            for i, return_stmt in enumerate(stmt.value):
                code += f"{self.generate_expression(return_stmt, is_main)}"
                if i < len(stmt.value) - 1:
                    code += ", "
            return f"{indent_str}return {code};\n"
        else:
            return f"{indent_str}{self.generate_expression(stmt, is_main)};\n"

    def generate_assignment(self, node, is_main):
        lhs = self.generate_expression(node.left, is_main)
        rhs = self.generate_expression(node.right, is_main)
        op = self.map_operator(node.operator)
        if is_main and isinstance(node.left, MemberAccessNode):
            if node.left.object == "output" and node.left.member == "position":
                return f"gl_Position = {rhs}"
            elif node.left.object in ["input", "output"]:
                return f"{node.left.member} = {rhs}"
        return f"{lhs} {op} {rhs}"

    def generate_if(self, node, indent, is_main):
        indent_str = "    " * indent
        code = f"{indent_str}if ({self.generate_expression(node.if_condition)}) {{\n"
        for stmt in node.if_body:
            code += self.generate_statement(stmt, indent + 1)
        code += f"{indent_str}}}"

        for else_if_condition, else_if_body in zip(
            node.else_if_conditions, node.else_if_bodies
        ):
            code += f" else if ({self.generate_expression(else_if_condition)}) {{\n"
            for stmt in else_if_body:
                code += self.generate_statement(stmt, indent + 1)
            code += f"{indent_str}}}"

        if node.else_body:
            code += " else {\n"
            for stmt in node.else_body:
                code += self.generate_statement(stmt, indent + 1)
            code += f"{indent_str}}}"
        code += "\n"
        return code

    def generate_for(self, node, indent, is_main):
        indent_str = "    " * indent

        init = self.generate_statement(node.init, 0).strip()[
            :-1
        ]  # Remove trailing semicolon

        condition = self.generate_statement(node.condition, 0).strip()[
            :-1
        ]  # Remove trailing semicolon

        update = self.generate_statement(node.update, 0).strip()[:-1]

        code = f"{indent_str}for ({init}; {condition}; {update}) {{\n"
        for stmt in node.body:
            code += self.generate_statement(stmt, indent + 1)
        code += f"{indent_str}}}\n"
        return code

    def generate_expression(self, expr, is_main=False):
        if isinstance(expr, str):
            return expr
        elif isinstance(expr, VariableNode):
            name = self.generate_expression(expr.name, is_main)
            return f"{self.map_type(expr.vtype)} {name}"
        elif isinstance(expr, BinaryOpNode):
            left = self.generate_expression(expr.left, is_main)
            right = self.generate_expression(expr.right, is_main)
            return f"{left} {self.map_operator(expr.op)} {right}"
        elif isinstance(expr, AssignmentNode):
            return self.generate_assignment(expr, is_main)
        elif isinstance(expr, UnaryOpNode):
            operand = self.generate_expression(expr.operand, is_main)
            return f"{self.map_operator(expr.op)}{operand}"
        elif isinstance(expr, FunctionCallNode):
            args = ", ".join(
                self.generate_expression(arg, is_main) for arg in expr.args
            )
            return f"{expr.name}({args})"
        elif isinstance(expr, MemberAccessNode):
            if is_main and expr.object in ["input", "output"]:
                return expr.member
            obj = self.generate_expression(expr.object, is_main)
            return f"{obj}.{expr.member}"
        elif isinstance(expr, TernaryOpNode):
            condition = self.generate_expression(expr.condition, is_main)
            true_expr = self.generate_expression(expr.true_expr, is_main)
            false_expr = self.generate_expression(expr.false_expr, is_main)
            return f"{condition} ? {true_expr} : {false_expr}"
        else:
            return str(expr)

    def map_type(self, vtype):
        return vtype

    def map_operator(self, op):
        op_map = {
            "PLUS": "+",
            "MINUS": "-",
            "MULTIPLY": "*",
            "DIVIDE": "/",
            "BITWISE_XOR": "^",
            "LESS_THAN": "<",
            "GREATER_THAN": ">",
            "ASSIGN_ADD": "+=",
            "ASSIGN_SUB": "-=",
            "ASSIGN_OR": "|=",
            "ASSIGN_MUL": "*=",
            "ASSIGN_DIV": "/=",
            "ASSIGN_MOD": "%=",
            "ASSIGN_AND": "&=",
            "LOGICAL_AND": "&&",
            "GREATER_THAN": ">",
            "ASSIGN_XOR": "^=",
            "LESS_EQUAL": "<=",
            "GREATER_EQUAL": ">=",
            "EQUAL": "==",
            "NOT_EQUAL": "!=",
            "AND": "&&",
            "OR": "||",
            "EQUALS": "=",
            "MOD": "%",
            "BITWISE_OR": "|",
            "BITWISE_AND": "&",
            "BITWISE_XOR": "^",
            "ASSIGN_SHIFT_LEFT": "<<=",
            "ASSIGN_SHIFT_RIGHT": ">>=",
            "BITWISE_SHIFT_RIGHT": ">>",
            "BITWISE_SHIFT_LEFT": "<<",
        }
        return op_map.get(op, op)

    def map_semantic(self, semantic):
        if semantic is not None:
            return f"{self.semantic_map.get(semantic, semantic)}"
        else:
            return ""

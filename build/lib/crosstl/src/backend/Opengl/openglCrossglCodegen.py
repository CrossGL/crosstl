from .OpenglAst import *
from .OpenglParser import *
from .OpenglLexer import *


class GLSLToCrossGLConverter:
    def __init__(self):
        self.current_shader = None
        self.vertex_item = None
        self.fragment_item = None

    def generate(self, ast):
        if isinstance(ast, ShaderNode):
            self.current_shader = ast
            return self.generate_shader(ast)
        return ""

    def generate_shader(self, node):
        # Set up the shader
        self.shader_inputs = node.global_inputs
        self.shader_outputs = node.global_outputs
        self.uniforms = node.uniforms

        code = "shader main {\n"

        # Generate vertex shader section
        self.vertex_item = node.vertex_section
        if self.vertex_item:
            code += "    vertex {\n"
            code += self.generate_layouts(self.vertex_item.layout_qualifiers)
            for vtype, name in self.shader_inputs:
                code += f"        input {self.map_type(vtype)} {name};\n"
            for vtype, name in self.shader_outputs:
                code += f"        output {self.map_type(vtype)} {name};\n"
            for vtype, name in self.vertex_item.inputs:
                code += f"        input {self.map_type(vtype)} {name};\n"
            for vtype, name in self.vertex_item.outputs:
                code += f"        output {self.map_type(vtype)} {name};\n"
                code += "\n"
            code += self.generate_uniforms() + "\n"
            code += "\n"

            # Generate functions
            code += self.generate_functions(self.vertex_item.functions, "vertex")
            code += "    }\n"
        else:
            raise ValueError("No vertex shader section to generate.")

        # Generate fragment shader section if present
        self.fragment_item = node.fragment_section
        if self.fragment_item and (
            self.fragment_item.layout_qualifiers or self.fragment_item.functions
        ):
            code += "    fragment {\n"
            code += self.generate_layouts(self.fragment_item.layout_qualifiers)
            # Process inputs and outputs
            for vtype, name in self.fragment_item.inputs:
                code += f"        input {self.map_type(vtype)} {name};\n"
            for vtype, name in self.fragment_item.outputs:
                code += f"        output {self.map_type(vtype)} {name};\n"
                code += "\n"
            code += self.generate_uniforms() + "\n"
            code += "\n"

            # Generate functions
            code += self.generate_functions(self.fragment_item.functions, "fragment")
            code += "    }\n"
        else:
            raise ValueError("No fragment shader section to generate.")

        code += "}\n"

        return code

    def generate_uniforms(self):
        uniform_lines = []
        for uniform in self.uniforms:
            uniform_lines.append(f"        {uniform}")
        return "\n".join(uniform_lines)

    def generate_layouts(self, layouts):
        code = ""
        for layout in layouts:
            if layout.io_type == "input":
                code += f"        input {layout.dtype} {layout.name};\n"
            elif layout.io_type == "output":
                code += f"        output {layout.dtype} {layout.name};\n"
        return code

    def generate_functions(self, functions, shader_type):
        code = ""

        if shader_type in ["vertex", "fragment"]:
            for function_node in functions:
                # Generate parameter list
                params = ", ".join(
                    f"{self.map_type(param[0])} {param[1]}"
                    for param in function_node.params
                )
                # Generate function header
                code += f"        {self.map_type(function_node.return_type)} {function_node.name}({params}) {{\n"
                # Generate function body
                for stmt in function_node.body:
                    code += self.generate_statement(stmt, shader_type, 2)
                # Close function definition
                code += "        }\n"

        return code

    def generate_statement(
        self,
        stmt,
        shader_type,
        indent=0,
    ):
        indent_str = "    " * indent
        if isinstance(stmt, VariableNode):
            return f"{indent_str}{self.map_type(stmt.vtype)} {stmt.name};\n"
        elif isinstance(stmt, AssignmentNode):
            return f"{indent_str}{self.generate_assignment(stmt,shader_type)};\n"
        elif isinstance(stmt, IfNode):
            return self.generate_if(stmt, shader_type, indent)
        elif isinstance(stmt, ForNode):
            return self.generate_for(stmt, shader_type, indent)
        elif isinstance(stmt, ReturnNode):
            return f"{indent_str}return {self.generate_expression(stmt.value,shader_type)};\n"
        else:
            return f"{indent_str}{self.generate_expression(stmt,shader_type)};\n"

    def generate_assignment(self, node, shader_type):
        lhs = self.generate_expression(node.name, shader_type)
        rhs = self.generate_expression(node.value, shader_type)
        return f"{lhs} = {rhs}"

    def generate_if(self, node: IfNode, shader_type, indent=0):
        indent_str = "    " * indent
        code = f"{indent_str}if {self.generate_expression(node.condition, shader_type)} {{\n"
        for stmt in node.if_body:
            code += self.generate_statement(stmt, shader_type, indent + 1)
        code += f"{indent_str}}}"

        # Handle else_if_chain
        for elif_condition, elif_body in node.else_if_chain:
            code += f" else if ({self.generate_expression(elif_condition, shader_type)}) {{\n"
            for stmt in elif_body:
                code += self.generate_statement(stmt, shader_type, indent + 1)
            code += f"{indent_str}}}"

        # Handle 'else' block if present
        if node.else_body:
            code += " else {\n"
            for stmt in node.else_body:
                code += self.generate_statement(stmt, shader_type, indent + 1)
            code += f"{indent_str}}}"

        code += "\n"
        return code

    def generate_else_if(self, node: IfNode, shader_type, indent):
        code = f" else if {self.generate_expression(node.condition, shader_type)} {{\n"
        for stmt in node.if_body:
            code += self.generate_statement(stmt, shader_type, indent + 1)
        code += f"{'    ' * indent}}}"
        return code

    def generate_for(self, node: ForNode, shader_type, indent):
        indent_str = "    " * indent
        init = self.generate_statement(node.init, shader_type).strip().rstrip(";")
        condition = self.generate_expression(node.condition, shader_type).strip()
        update = self.generate_update(node.update, shader_type).strip().rstrip(";")
        code = f"{indent_str}for ({init}; {condition}; {update}) {{\n"
        for stmt in node.body:
            code += self.generate_statement(stmt, shader_type, indent + 1)
        code += f"{indent_str}}}\n"

        return code

    def generate_update(self, node, shader_type):
        if isinstance(node, AssignmentNode):
            if isinstance(node.value, UnaryOpNode):
                operand = self.generate_expression(
                    node.value.operand, shader_type
                ).strip()
                if node.value.op == "++":
                    return f"++{operand}"
                elif node.value.op == "POST_INCREMENT":
                    return f"{operand}++"
                elif node.value.op == "--":
                    return f"--{operand}"
                elif node.value.op == "POST_DECREMENT":
                    return f"{operand}--"
            else:
                lhs = self.generate_expression(node.name, shader_type).strip()
                rhs = self.generate_expression(node.value, shader_type).strip()
                return f"{lhs} = {rhs}"

        elif isinstance(node, UnaryOpNode):
            operand = self.generate_expression(node.operand, shader_type).strip()
            if node.op == "++":
                return f"++{operand}"
            elif node.op == "POST_INCREMENT":
                return f"{operand}++"
            elif node.op == "--":
                return f"--{operand}"
            elif node.op == "POST_DECREMENT":
                return f"{operand}--"
            else:
                return f"{node.op}{operand}"

        elif isinstance(node, BinaryOpNode):
            left = self.generate_expression(node.left, shader_type).strip()
            right = self.generate_expression(node.right, shader_type).strip()
            op = self.map_operator(node.op)
            return f"{left} {op} {right}"

        else:
            raise ValueError(f"Unsupported update node type: {type(node)}")

    def generate_expression(self, expr, shader_type):
        if isinstance(expr, str):
            return self.translate_expression(expr, shader_type)
        elif isinstance(expr, VariableNode):
            return f"{expr.vtype} {self.translate_expression(expr.name, shader_type)}"
        elif isinstance(expr, BinaryOpNode):
            left = self.generate_expression(expr.left, shader_type)
            right = self.generate_expression(expr.right, shader_type)
            op = self.map_operator(expr.op)
            return f"{left} {op} {right}"
        elif isinstance(expr, FunctionCallNode):
            args = ", ".join(
                self.generate_expression(arg, shader_type) for arg in expr.args
            )
            func_name = self.translate_expression(expr.name, shader_type)
            return f"{func_name}({args})"
        elif isinstance(expr, UnaryOpNode):
            operand = self.generate_expression(expr.operand, shader_type)
            if expr.op in ["++", "--"]:
                if expr.op == "++":
                    return f"++{operand}"
                elif expr.op == "--":
                    return f"--{operand}"
            else:
                return f"{expr.op}{operand}"
        elif isinstance(expr, TernaryOpNode):
            condition = self.generate_expression(expr.condition, shader_type)
            true_expr = self.generate_expression(expr.true_expr, shader_type)
            false_expr = self.generate_expression(expr.false_expr, shader_type)
            return f"{condition} ? {true_expr} : {false_expr}"
        elif isinstance(expr, MemberAccessNode):
            obj = self.generate_expression(expr.object, shader_type)
            member = expr.member
            return f"{obj}.{member}"
        else:
            return str(expr)

    def translate_expression(self, expr, shader_type):
        if shader_type == "vertex":
            if self.vertex_item:
                if self.vertex_item.inputs:
                    for _, input_name in self.vertex_item.inputs:
                        if expr == input_name:
                            return input_name
                if self.vertex_item.outputs:
                    for _, output_name in self.vertex_item.outputs:
                        if expr == output_name:
                            return output_name

        elif shader_type == "fragment":
            if self.fragment_item:
                if self.fragment_item.inputs:
                    for _, input_name in self.fragment_item.inputs:
                        if expr == input_name:
                            return input_name
                if self.fragment_item.outputs:
                    for _, output_name in self.fragment_item.outputs:
                        if expr == output_name:
                            return output_name
        return expr

    def map_type(self, vtype):
        type_map = {
            "vec3": "vec3",
            "vec4": "vec4",
            "float": "float",
            "int": "int",
            "bool": "bool",
        }
        return type_map.get(vtype, vtype)

    def map_operator(self, op):
        op_map = {
            "PLUS": "+",
            "MINUS": "-",
            "MULTIPLY": "*",
            "DIVIDE": "/",
            "LESS_THAN": "<",
            "GREATER_THAN": ">",
            "LESS_EQUAL": "<=",
            "GREATER_EQUAL": ">=",
            "EQUAL": "==",
            "NOT_EQUAL": "!=",
            "AND": "&&",
            "OR": "||",
        }
        return op_map.get(op, op)

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
    ShaderNode,
)


class MojoCodeGen:
    def __init__(self):
        self.indent_level = 0
        self.indent_str = "    "

    def indent(self):
        return self.indent_str * self.indent_level

    def generate(self, ast):
        if isinstance(ast, list):
            result = ""
            for node in ast:
                result += self.generate(node) + "\n"
            return result
        elif isinstance(ast, ShaderNode):
            return self.generate_shader(ast)
        elif isinstance(ast, StructNode):
            return self.generate_struct(ast)
        else:
            # Return empty string for unhandled node types
            return ""

    def generate_shader(self, node):
        result = ""

        # Generate struct definitions
        for struct in node.structs:
            result += self.generate_struct(struct) + "\n\n"

        # Generate vertex and fragment shaders
        for function in node.functions:
            if function.qualifier == "vertex":
                result += "# Vertex Shader\n"
                result += self.generate_function(function) + "\n\n"
            elif function.qualifier == "fragment":
                result += "# Fragment Shader\n"
                result += self.generate_function(function) + "\n\n"
            else:
                result += self.generate_function(function) + "\n\n"

        return result

    def generate_struct(self, node):
        result = f"struct {node.name}:\n"
        self.indent_level += 1

        # Generate struct members
        for member in node.members:
            semantic = f" # {member.semantic}" if member.semantic else ""
            result += f"{self.indent()}{member.name}: {self.convert_type(member.vtype)}{semantic}\n"

        self.indent_level -= 1
        return result

    def generate_function(self, node):
        ret_type = self.convert_type(node.return_type)

        # Handle parameters which are VariableNode objects
        params_str = ""
        if node.params:
            if isinstance(node.params[0], VariableNode):
                # Handle list of VariableNode objects
                params_str = ", ".join(
                    [
                        f"{param.name}: {self.convert_type(param.vtype)}"
                        for param in node.params
                    ]
                )
            else:
                # Handle tuples of (type, name)
                params_str = ", ".join(
                    [
                        f"{param_name}: {self.convert_type(param_type)}"
                        for param_type, param_name in node.params
                    ]
                )

        result = f"fn {node.name}({params_str}) -> {ret_type}:\n"
        self.indent_level += 1

        # Generate function body
        for stmt in node.body:
            result += self.indent() + self.generate_statement(stmt) + "\n"

        self.indent_level -= 1
        return result

    def generate_statement(self, node):
        if isinstance(node, ReturnNode):
            return f"return {self.generate_expression(node.value)}"
        elif isinstance(node, AssignmentNode):
            return self.generate_assignment(node)
        elif isinstance(node, VariableNode):
            return f"var {node.name}: {self.convert_type(node.vtype)}"
        elif isinstance(node, IfNode):
            return self.generate_if(node)
        elif isinstance(node, ForNode):
            return self.generate_for(node)
        else:
            return self.generate_expression(node)

    def generate_assignment(self, node):
        left = self.generate_expression(node.left)
        right = self.generate_expression(node.right)
        return f"{left} {node.operator} {right}"

    def generate_expression(self, node):
        if isinstance(node, VariableNode):
            return node.name
        elif isinstance(node, MemberAccessNode):
            obj = self.generate_expression(node.object)
            return f"{obj}.{node.member}"
        elif isinstance(node, BinaryOpNode):
            left = self.generate_expression(node.left)
            right = self.generate_expression(node.right)
            return f"{left} {node.op} {right}"
        elif isinstance(node, FunctionCallNode):
            args = ", ".join([self.generate_expression(arg) for arg in node.args])
            return f"{node.name}({args})"
        elif isinstance(node, UnaryOpNode):
            operand = self.generate_expression(node.operand)
            return f"{node.op}{operand}"
        elif isinstance(node, str):
            return node
        else:
            return str(node)

    def generate_if(self, node):
        condition = self.generate_expression(node.if_condition)
        result = f"if {condition}:\n"

        self.indent_level += 1
        for stmt in node.if_body:
            result += self.indent() + self.generate_statement(stmt) + "\n"
        self.indent_level -= 1

        if node.else_body:
            result += self.indent() + "else:\n"
            self.indent_level += 1
            for stmt in node.else_body:
                result += self.indent() + self.generate_statement(stmt) + "\n"
            self.indent_level -= 1

        return result.rstrip()

    def generate_for(self, node):
        init = self.generate_statement(node.init)
        condition = self.generate_expression(node.condition)
        update = self.generate_statement(node.update)

        result = f"for {init}; {condition}; {update}:\n"

        self.indent_level += 1
        for stmt in node.body:
            result += self.indent() + self.generate_statement(stmt) + "\n"
        self.indent_level -= 1

        return result.rstrip()

    def convert_type(self, type_name):
        # Map CrossGL types to Mojo types
        type_map = {
            "vec2": "Vec2",
            "vec3": "Vec3",
            "vec4": "Vec4",
            "float": "Float32",
            "int": "Int32",
            "bool": "Bool",
            "void": "None",
        }

        return type_map.get(type_name, type_name)

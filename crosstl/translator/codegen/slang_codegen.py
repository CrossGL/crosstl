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


class SlangCodeGen:
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
            # Handle new AST structure
            result = ""

            # Generate structs - handle both old and new AST
            structs = getattr(ast, "structs", [])
            for struct in structs:
                result += self.generate_struct(struct) + "\n\n"

            # Generate global variables - handle both old and new AST
            global_vars = getattr(ast, "global_variables", [])
            for node in global_vars:
                # Handle both old and new AST variable structures
                if hasattr(node, "var_type"):
                    if hasattr(node.var_type, "name"):
                        vtype = node.var_type.name
                    else:
                        vtype = str(node.var_type)
                elif hasattr(node, "vtype"):
                    vtype = node.vtype
                else:
                    vtype = "float"
                result += f"{self.convert_type(vtype)} {node.name};\n"

            # Generate cbuffers - handle both old and new AST
            cbuffers = getattr(ast, "cbuffers", [])
            for node in cbuffers:
                if isinstance(node, StructNode):
                    result += (
                        "cbuffer " + self.generate_struct_definition(node) + "\n\n"
                    )
                elif hasattr(node, "name") and hasattr(node, "members"):
                    result += f"cbuffer {node.name} {{\n"
                    for member in node.members:
                        # Handle both old and new AST member structures
                        if hasattr(member, "member_type"):
                            member_type = str(member.member_type)
                        else:
                            member_type = getattr(member, "vtype", "float")
                        result += (
                            f"    {self.convert_type(member_type)} {member.name};\n"
                        )
                    result += "};\n\n"

            # Generate functions - handle both old and new AST
            functions = getattr(ast, "functions", [])
            for function in functions:
                # Handle both old and new AST function structures
                if hasattr(function, "qualifiers") and function.qualifiers:
                    qualifier = function.qualifiers[0] if function.qualifiers else None
                else:
                    qualifier = getattr(function, "qualifier", None)

                if qualifier == "vertex":
                    result += "// Vertex Shader\n"
                    result += self.generate_function(function) + "\n\n"
                elif qualifier == "fragment":
                    result += "// Fragment Shader\n"
                    result += self.generate_function(function) + "\n\n"
                else:
                    result += self.generate_function(function) + "\n\n"

            # Handle shader stages (new AST structure)
            if hasattr(ast, "stages") and ast.stages:
                for stage_type, stage in ast.stages.items():
                    if hasattr(stage, "entry_point"):
                        stage_name = (
                            str(stage_type).split(".")[-1].lower()
                        )  # Extract stage name from enum
                        result += f"// {stage_name.title()} Shader\n"
                        result += self.generate_function(stage.entry_point) + "\n\n"
                    if hasattr(stage, "local_functions"):
                        for func in stage.local_functions:
                            result += self.generate_function(func) + "\n\n"

            return result

    def generate_shader(self, node):
        result = ""

        # Generate struct definitions - handle both old and new AST
        structs = getattr(node, "structs", [])
        for struct in structs:
            result += self.generate_struct(struct) + "\n\n"

        # Generate vertex and fragment shaders - handle both old and new AST
        functions = getattr(node, "functions", [])
        for function in functions:
            # Handle both old and new AST function structures
            if hasattr(function, "qualifiers") and function.qualifiers:
                qualifier = function.qualifiers[0] if function.qualifiers else None
            else:
                qualifier = getattr(function, "qualifier", None)

            if qualifier == "vertex":
                result += "// Vertex Shader\n"
                result += self.generate_function(function) + "\n\n"
            elif qualifier == "fragment":
                result += "// Fragment Shader\n"
                result += self.generate_function(function) + "\n\n"
            else:
                result += self.generate_function(function) + "\n\n"

        return result

    def generate_struct(self, node):
        result = f"struct {node.name}\n{{\n"
        self.indent_level += 1

        # Generate struct members - handle both old and new AST
        members = getattr(node, "members", [])
        for member in members:
            # Handle both old and new AST member structures
            if hasattr(member, "member_type"):
                # New AST structure
                if hasattr(member.member_type, "name"):
                    member_type = self.convert_type(member.member_type.name)
                else:
                    member_type = self.convert_type(str(member.member_type))
            elif hasattr(member, "vtype"):
                # Old AST structure
                member_type = self.convert_type(member.vtype)
            else:
                member_type = "float"

            # Handle semantic - get from attributes in new AST
            semantic = None
            if hasattr(member, "semantic"):
                semantic = member.semantic
            elif hasattr(member, "attributes"):
                for attr in member.attributes:
                    if hasattr(attr, "name") and attr.name in [
                        "position",
                        "color",
                        "texcoord",
                        "normal",
                    ]:
                        semantic = attr.name
                        break

            semantic_str = f" : {semantic}" if semantic else ""
            result += f"{self.indent()}{member_type} {member.name}{semantic_str};\n"

        self.indent_level -= 1
        result += "};"
        return result

    def generate_struct_definition(self, node):
        result = f"{node.name}\n{{\n"

        # Generate struct members - handle both old and new AST
        members = getattr(node, "members", [])
        for member in members:
            # Handle both old and new AST member structures
            if hasattr(member, "member_type"):
                member_type = self.convert_type(str(member.member_type))
            else:
                member_type = self.convert_type(getattr(member, "vtype", "float"))
            result += f"    {member_type} {member.name};\n"

        result += "};"
        return result

    def generate_function(self, node):
        # Handle return type - support both old and new AST
        if hasattr(node, "return_type"):
            if hasattr(node.return_type, "name"):
                ret_type = self.convert_type(node.return_type.name)
            else:
                ret_type = self.convert_type(str(node.return_type))
        else:
            ret_type = "void"

        # Handle semantic
        semantic = None
        if hasattr(node, "semantic"):
            semantic = node.semantic
        elif hasattr(node, "attributes"):
            for attr in node.attributes:
                if hasattr(attr, "name"):
                    semantic = attr.name
                    break

        semantic_str = f" : {semantic}" if semantic else ""

        # Handle parameters - support both old and new AST
        param_list = getattr(node, "parameters", getattr(node, "params", []))
        params_str = ""
        if param_list:
            if param_list and hasattr(param_list[0], "name"):
                # Handle list of parameter objects
                params = []
                for param in param_list:
                    if hasattr(param, "param_type"):
                        # New AST structure
                        if hasattr(param.param_type, "name"):
                            param_type = self.convert_type(param.param_type.name)
                        else:
                            param_type = self.convert_type(str(param.param_type))
                    elif hasattr(param, "vtype"):
                        # Old AST structure
                        param_type = self.convert_type(param.vtype)
                    else:
                        param_type = "float"
                    params.append(f"{param_type} {param.name}")
                params_str = ", ".join(params)
            else:
                # Handle tuples of (type, name)
                params_str = ", ".join(
                    [
                        f"{self.convert_type(param_type)} {param_name}"
                        for param_type, param_name in param_list
                    ]
                )

        result = f"{ret_type} {node.name}({params_str}){semantic_str}\n{{\n"
        self.indent_level += 1

        # Generate function body - handle both old and new AST
        body = getattr(node, "body", [])
        if hasattr(body, "statements"):
            # New AST BlockNode structure
            for stmt in body.statements:
                result += self.indent() + self.generate_statement(stmt) + "\n"
        elif isinstance(body, list):
            # Old AST structure
            for stmt in body:
                result += self.indent() + self.generate_statement(stmt) + "\n"

        self.indent_level -= 1
        result += "}"
        return result

    def generate_statement(self, node):
        if isinstance(node, ReturnNode):
            return f"return {self.generate_expression(node.value)};"
        elif isinstance(node, AssignmentNode):
            return self.generate_assignment(node) + ";"
        elif isinstance(node, VariableNode):
            return f"{self.convert_type(node.vtype)} {node.name};"
        elif isinstance(node, IfNode):
            return self.generate_if(node)
        elif isinstance(node, ForNode):
            return self.generate_for(node)
        else:
            return self.generate_expression(node) + ";"

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
        result = f"if ({condition})\n{{\n"

        self.indent_level += 1
        for stmt in node.if_body:
            result += self.indent() + self.generate_statement(stmt) + "\n"
        self.indent_level -= 1

        result += self.indent() + "}"

        if node.else_body:
            result += "\nelse\n{\n"
            self.indent_level += 1
            for stmt in node.else_body:
                result += self.indent() + self.generate_statement(stmt) + "\n"
            self.indent_level -= 1
            result += self.indent() + "}"

        return result

    def generate_for(self, node):
        init = self.generate_statement(node.init).rstrip(";")
        condition = self.generate_expression(node.condition)
        update = self.generate_statement(node.update).rstrip(";")

        result = f"for ({init}; {condition}; {update})\n{{\n"

        self.indent_level += 1
        for stmt in node.body:
            result += self.indent() + self.generate_statement(stmt) + "\n"
        self.indent_level -= 1

        result += self.indent() + "}"
        return result

    def convert_type(self, type_name):
        # Map CrossGL types to Slang types
        type_map = {
            "vec2": "float2",
            "vec3": "float3",
            "vec4": "float4",
            "float": "float",
            "int": "int",
            "bool": "bool",
            "void": "void",
        }

        return type_map.get(type_name, type_name)

from ..ast import (
    ArrayNode,
    ArrayAccessNode,
    AssignmentNode,
    BinaryOpNode,
    BreakNode,
    CaseNode,
    CbufferNode,
    ContinueNode,
    ExpressionStatementNode,
    ForNode,
    FunctionCallNode,
    IdentifierNode,
    FunctionNode,
    IfNode,
    LiteralNode,
    MemberAccessNode,
    ReturnNode,
    ShaderNode,
    StructNode,
    SwitchNode,
    TernaryOpNode,
    UnaryOpNode,
    VariableNode,
    WhileNode,
)
from .array_utils import (
    format_c_style_array_declaration,
    get_array_size_from_node,
)


class SlangCodeGen:
    def __init__(self):
        self.indent_level = 0
        self.indent_str = "    "
        self.variable_types = {}
        self.helper_functions = {}
        self._generating = False

    def indent(self):
        return self.indent_str * self.indent_level

    def generate(self, ast):
        outermost = not self._generating
        if outermost:
            self._generating = True
            self.variable_types = {}
            self.helper_functions = {}

        if isinstance(ast, list):
            result = ""
            for node in ast:
                result += self.generate(node) + "\n"
            return self.finish_generation(result, outermost)
        elif isinstance(ast, ShaderNode):
            return self.finish_generation(self.generate_shader(ast), outermost)
        elif isinstance(ast, StructNode):
            return self.finish_generation(self.generate_struct(ast), outermost)
        else:
            # Handle new AST structure
            result = ""

            structs = getattr(ast, "structs", [])
            for struct in structs:
                result += self.generate_struct(struct) + "\n\n"

            global_vars = getattr(ast, "global_variables", [])
            for node in global_vars:
                result += self.generate_global_variable(node)

            cbuffers = getattr(ast, "cbuffers", [])
            for node in cbuffers:
                if isinstance(node, StructNode):
                    result += (
                        "cbuffer " + self.generate_struct_definition(node) + "\n\n"
                    )
                elif hasattr(node, "name") and hasattr(node, "members"):
                    result += f"cbuffer {node.name} {{\n"
                    for member in node.members:
                        if hasattr(member, "member_type"):
                            member_type = str(member.member_type)
                        else:
                            member_type = getattr(member, "vtype", "float")
                        result += (
                            f"    {self.convert_type(member_type)} {member.name};\n"
                        )
                    result += "};\n\n"

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
                    result += self.generate_stage(stage_type, stage)

            return self.finish_generation(result, outermost)

    def finish_generation(self, result, outermost):
        if not outermost:
            return result

        helpers = self.emit_helper_functions()
        self._generating = False
        if helpers:
            return helpers + result
        return result

    def emit_helper_functions(self):
        if not self.helper_functions:
            return ""
        return "\n\n".join(self.helper_functions.values()) + "\n\n"

    def generate_shader(self, node):
        result = ""

        structs = getattr(node, "structs", [])
        for struct in structs:
            result += self.generate_struct(struct) + "\n\n"

        global_vars = getattr(node, "global_variables", [])
        for global_var in global_vars:
            result += self.generate_global_variable(global_var)

        functions = getattr(node, "functions", [])
        for function in functions:
            stage_name = self.get_function_stage(function)
            if stage_name:
                result += f"// {stage_name.title()} Shader\n"
                result += self.generate_function(function, shader_type=stage_name)
                result += "\n\n"
            else:
                result += self.generate_function(function) + "\n\n"

        stages = getattr(node, "stages", {})
        for stage_type, stage in stages.items():
            result += self.generate_stage(stage_type, stage)

        return result

    def get_stage_name(self, stage_type):
        if hasattr(stage_type, "value"):
            return stage_type.value
        return str(stage_type).split(".")[-1].lower()

    def get_function_stage(self, function):
        if hasattr(function, "qualifiers") and function.qualifiers:
            qualifier = function.qualifiers[0]
        else:
            qualifier = getattr(function, "qualifier", None)

        if qualifier in {"vertex", "fragment", "compute"}:
            return qualifier
        return None

    def generate_stage(self, stage_type, stage):
        stage_name = self.get_stage_name(stage_type)
        result = f"// {stage_name.title()} Shader\n"

        local_variables = getattr(stage, "local_variables", [])
        for local_var in local_variables:
            result += self.generate_global_variable(local_var)

        for func in getattr(stage, "local_functions", []):
            result += self.generate_function(func) + "\n\n"

        entry_point = getattr(stage, "entry_point", None)
        if entry_point is not None:
            result += self.generate_function(entry_point, shader_type=stage_name)
            result += "\n\n"

        return result

    def convert_type_node_to_string(self, type_node) -> str:
        if hasattr(type_node, "name"):
            generic_args = getattr(type_node, "generic_args", [])
            if generic_args:
                args = ", ".join(
                    self.convert_type_node_to_string(arg) for arg in generic_args
                )
                return f"{type_node.name}<{args}>"
            return type_node.name
        if hasattr(type_node, "rows") and hasattr(type_node, "cols"):
            element_type = self.convert_type_node_to_string(type_node.element_type)
            if element_type == "float":
                if type_node.rows == type_node.cols:
                    return f"mat{type_node.rows}"
                return f"mat{type_node.rows}x{type_node.cols}"
            return f"{element_type}{type_node.rows}x{type_node.cols}"
        if hasattr(type_node, "element_type") and hasattr(type_node, "size"):
            element_type = self.convert_type_node_to_string(type_node.element_type)
            if type_node.__class__.__name__ == "ArrayType":
                if type_node.size is None:
                    return f"{element_type}[]"
                size = (
                    str(type_node.size)
                    if isinstance(type_node.size, int)
                    else self.generate_expression(type_node.size)
                )
                return f"{element_type}[{size}]"
            if element_type == "float":
                return f"vec{type_node.size}"
            if element_type == "int":
                return f"ivec{type_node.size}"
            if element_type == "uint":
                return f"uvec{type_node.size}"
            if element_type == "bool":
                return f"bvec{type_node.size}"
            return f"{element_type}{type_node.size}"
        return str(type_node)

    def format_declaration(self, type_name, name):
        if not isinstance(type_name, str):
            type_name = self.convert_type_node_to_string(type_name)
        if "[" in type_name and "]" in type_name:
            open_bracket = type_name.find("[")
            base_type = type_name[:open_bracket]
            suffix = type_name[open_bracket:]
            type_name = f"{self.convert_type(base_type)}{suffix}"
        else:
            type_name = self.convert_type(type_name)
        return format_c_style_array_declaration(type_name, name)

    def get_variable_type(self, node):
        if hasattr(node, "var_type"):
            return self.convert_type_node_to_string(node.var_type)
        if hasattr(node, "vtype"):
            return node.vtype
        return "float"

    def register_variable_type(self, name, type_name):
        if not name or type_name is None:
            return
        if not isinstance(type_name, str):
            type_name = self.convert_type_node_to_string(type_name)
        self.variable_types[name] = type_name

    def generate_global_variable(self, node):
        if isinstance(node, ArrayNode):
            self.register_variable_type(node.name, node.element_type)
            element_type = self.convert_type(node.element_type)
            size = get_array_size_from_node(node)
            if size is None:
                return f"{element_type} {node.name}[];\n"
            return f"{element_type} {node.name}[{size}];\n"

        vtype = self.get_variable_type(node)
        self.register_variable_type(node.name, vtype)
        return f"{self.format_declaration(vtype, node.name)};\n"

    def generate_struct(self, node):
        result = f"struct {node.name}\n{{\n"
        self.indent_level += 1

        members = getattr(node, "members", [])
        for member in members:
            if hasattr(member, "member_type"):
                member_type = self.convert_type(
                    self.convert_type_node_to_string(member.member_type)
                )
            elif hasattr(member, "vtype"):
                member_type = self.convert_type(member.vtype)
            else:
                member_type = "float"

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
            declaration = self.format_declaration(member_type, member.name)
            result += f"{self.indent()}{declaration}{semantic_str};\n"

        self.indent_level -= 1
        result += "};"
        return result

    def generate_struct_definition(self, node):
        result = f"{node.name}\n{{\n"

        members = getattr(node, "members", [])
        for member in members:
            if hasattr(member, "member_type"):
                member_type = self.convert_type_node_to_string(member.member_type)
            else:
                member_type = getattr(member, "vtype", "float")
            result += f"    {self.format_declaration(member_type, member.name)};\n"

        result += "};"
        return result

    def generate_function(self, node, shader_type=None):
        saved_variable_types = self.variable_types.copy()
        if hasattr(node, "return_type"):
            ret_type = self.convert_type(
                self.convert_type_node_to_string(node.return_type)
            )
        else:
            ret_type = "void"

        semantic = None
        if hasattr(node, "semantic"):
            semantic = node.semantic
        elif hasattr(node, "attributes"):
            for attr in node.attributes:
                if hasattr(attr, "name"):
                    semantic = attr.name
                    break

        semantic_str = f" : {semantic}" if semantic else ""

        param_list = getattr(node, "parameters", getattr(node, "params", []))
        params_str = ""
        if param_list:
            if param_list and hasattr(param_list[0], "name"):
                params = []
                for param in param_list:
                    if hasattr(param, "param_type"):
                        param_type_name = self.convert_type_node_to_string(
                            param.param_type
                        )
                        self.register_variable_type(param.name, param_type_name)
                        param_type = self.convert_type(param_type_name)
                    elif hasattr(param, "vtype"):
                        self.register_variable_type(param.name, param.vtype)
                        param_type = self.convert_type(param.vtype)
                    else:
                        param_type = "float"
                    params.append(self.format_declaration(param_type, param.name))
                params_str = ", ".join(params)
            else:
                for param_type, param_name in param_list:
                    self.register_variable_type(param_name, param_type)
                params_str = ", ".join(
                    [
                        f"{self.convert_type(param_type)} {param_name}"
                        for param_type, param_name in param_list
                    ]
                )

        result = ""
        if shader_type:
            result += f'[shader("{shader_type}")]\n'
        result += f"{ret_type} {node.name}({params_str}){semantic_str}\n{{\n"
        self.indent_level += 1

        body = getattr(node, "body", [])
        if hasattr(body, "statements"):
            for stmt in body.statements:
                result += self.emit_statement(stmt) + "\n"
        elif isinstance(body, list):
            for stmt in body:
                result += self.emit_statement(stmt) + "\n"

        self.indent_level -= 1
        result += "}"
        self.variable_types = saved_variable_types
        return result

    def emit_statement(self, node):
        statement = self.generate_statement(node)
        lines = statement.splitlines()
        return "\n".join(
            self.indent() + line if line and not line[0].isspace() else line
            for line in lines
        )

    def generate_statement(self, node):
        if isinstance(node, ReturnNode):
            if node.value is None:
                return "return;"
            return f"return {self.generate_expression(node.value)};"
        elif isinstance(node, AssignmentNode):
            return self.generate_assignment(node) + ";"
        elif isinstance(node, ExpressionStatementNode):
            return self.generate_expression(node.expression) + ";"
        elif isinstance(node, VariableNode):
            var_type = self.get_variable_type(node)
            self.register_variable_type(node.name, var_type)
            declaration = self.format_declaration(var_type, node.name)
            initial_value = getattr(node, "initial_value", getattr(node, "value", None))
            if initial_value is not None:
                return f"{declaration} = {self.generate_expression(initial_value)};"
            return f"{declaration};"
        elif isinstance(node, IfNode):
            return self.generate_if(node)
        elif isinstance(node, ForNode):
            return self.generate_for(node)
        elif isinstance(node, WhileNode):
            return self.generate_while(node)
        elif isinstance(node, SwitchNode):
            return self.generate_switch(node)
        elif isinstance(node, BreakNode):
            return "break;"
        elif isinstance(node, ContinueNode):
            return "continue;"
        else:
            return self.generate_expression(node) + ";"

    def generate_assignment(self, node):
        left = self.generate_expression(node.left)
        right = self.generate_expression(node.right)
        return f"{left} {node.operator} {right}"

    def generate_literal(self, node):
        value = node.value
        literal_type = getattr(getattr(node, "literal_type", None), "name", None)

        if isinstance(value, bool):
            return "true" if value else "false"
        if literal_type == "bool" and isinstance(value, str):
            lower_value = value.lower()
            if lower_value in {"true", "false"}:
                return lower_value
        if literal_type == "char":
            escaped = self.escape_literal(value, quote="'")
            return f"'{escaped}'"
        if isinstance(value, str):
            escaped = self.escape_literal(value, quote='"')
            return f'"{escaped}"'
        return str(value)

    def escape_literal(self, value, quote):
        text = str(value)
        escaped = []
        for index, char in enumerate(text):
            if char == "\n":
                escaped.append("\\n")
            elif char == "\r":
                escaped.append("\\r")
            elif char == "\t":
                escaped.append("\\t")
            elif char == quote and (index == 0 or text[index - 1] != "\\"):
                escaped.append("\\" + char)
            else:
                escaped.append(char)
        return "".join(escaped)

    def generate_expression(self, node):
        if isinstance(node, VariableNode):
            return node.name
        elif isinstance(node, IdentifierNode):
            return node.name
        elif isinstance(node, LiteralNode):
            return self.generate_literal(node)
        elif isinstance(node, ExpressionStatementNode):
            return self.generate_expression(node.expression)
        elif isinstance(node, AssignmentNode):
            return self.generate_assignment(node)
        elif isinstance(node, ArrayAccessNode):
            array = self.generate_expression(
                getattr(node, "array", getattr(node, "array_expr", None))
            )
            index = self.generate_expression(
                getattr(node, "index", getattr(node, "index_expr", None))
            )
            return f"{array}[{index}]"
        elif isinstance(node, MemberAccessNode):
            obj = self.generate_expression(node.object)
            return f"{obj}.{node.member}"
        elif isinstance(node, BinaryOpNode):
            left = self.generate_expression(node.left)
            right = self.generate_expression(node.right)
            return f"{left} {node.op} {right}"
        elif isinstance(node, FunctionCallNode):
            func_expr = getattr(node, "function", None)
            if func_expr is None:
                func_expr = node.name
            if hasattr(func_expr, "name"):
                callee = func_expr.name
            elif isinstance(func_expr, str):
                callee = func_expr
            else:
                callee = self.generate_expression(func_expr)
            resource_call = self.generate_resource_call(callee, node.args)
            if resource_call is not None:
                return resource_call
            args = ", ".join([self.generate_expression(arg) for arg in node.args])
            callee = self.convert_type(callee)
            return f"{callee}({args})"
        elif isinstance(node, UnaryOpNode):
            operand = self.generate_expression(node.operand)
            if getattr(node, "is_postfix", False):
                return f"{operand}{node.op}"
            return f"{node.op}{operand}"
        elif isinstance(node, TernaryOpNode):
            condition = self.generate_expression(node.condition)
            true_expr = self.generate_expression(node.true_expr)
            false_expr = self.generate_expression(node.false_expr)
            return f"({condition} ? {true_expr} : {false_expr})"
        elif isinstance(node, str):
            return node
        else:
            return str(node)

    def generate_if(self, node):
        condition = self.generate_expression(
            getattr(node, "condition", getattr(node, "if_condition", None))
        )
        result = f"if ({condition})\n{{\n"

        self.indent_level += 1
        for stmt in self.get_statements(getattr(node, "if_body", [])):
            result += self.emit_statement(stmt) + "\n"
        self.indent_level -= 1

        result += self.indent() + "}"

        else_body = getattr(node, "else_body", None)
        if else_body:
            result += "\nelse\n{\n"
            self.indent_level += 1
            for stmt in self.get_statements(else_body):
                result += self.emit_statement(stmt) + "\n"
            self.indent_level -= 1
            result += self.indent() + "}"

        return result

    def generate_for(self, node):
        init = self.generate_statement(node.init).rstrip(";")
        condition = self.generate_expression(node.condition)
        update = self.generate_statement(node.update).rstrip(";")

        result = f"for ({init}; {condition}; {update})\n{{\n"

        self.indent_level += 1
        for stmt in self.get_statements(node.body):
            result += self.emit_statement(stmt) + "\n"
        self.indent_level -= 1

        result += self.indent() + "}"
        return result

    def generate_while(self, node):
        condition = self.generate_expression(node.condition)
        result = f"while ({condition})\n{{\n"

        self.indent_level += 1
        for stmt in self.get_statements(node.body):
            result += self.emit_statement(stmt) + "\n"
        self.indent_level -= 1

        result += self.indent() + "}"
        return result

    def generate_switch(self, node):
        expression = self.generate_expression(node.expression)
        result = f"switch ({expression})\n{{\n"

        self.indent_level += 1
        for case in getattr(node, "cases", []):
            if not isinstance(case, CaseNode):
                continue

            if case.value is None:
                result += self.indent() + "default:\n"
            else:
                case_value = self.generate_expression(case.value)
                result += self.indent() + f"case {case_value}:\n"

            self.indent_level += 1
            for stmt in self.get_statements(case.statements):
                result += self.emit_statement(stmt) + "\n"
            self.indent_level -= 1
        self.indent_level -= 1

        result += self.indent() + "}"
        return result

    def get_statements(self, body):
        if body is None:
            return []
        if hasattr(body, "statements"):
            return body.statements
        if isinstance(body, list):
            return body
        return [body]

    def convert_type(self, type_name):
        # Map CrossGL types to Slang types
        type_map = {
            "vec2<f32>": "float2",
            "vec3<f32>": "float3",
            "vec4<f32>": "float4",
            "vec2<f64>": "double2",
            "vec3<f64>": "double3",
            "vec4<f64>": "double4",
            "vec2<i32>": "int2",
            "vec3<i32>": "int3",
            "vec4<i32>": "int4",
            "vec2<u32>": "uint2",
            "vec3<u32>": "uint3",
            "vec4<u32>": "uint4",
            "vec2<bool>": "bool2",
            "vec3<bool>": "bool3",
            "vec4<bool>": "bool4",
            "vec2": "float2",
            "vec3": "float3",
            "vec4": "float4",
            "ivec2": "int2",
            "ivec3": "int3",
            "ivec4": "int4",
            "uvec2": "uint2",
            "uvec3": "uint3",
            "uvec4": "uint4",
            "dvec2": "double2",
            "dvec3": "double3",
            "dvec4": "double4",
            "bvec2": "bool2",
            "bvec3": "bool3",
            "bvec4": "bool4",
            "mat2": "float2x2",
            "mat3": "float3x3",
            "mat4": "float4x4",
            "mat2x2": "float2x2",
            "mat2x3": "float2x3",
            "mat2x4": "float2x4",
            "mat3x2": "float3x2",
            "mat3x3": "float3x3",
            "mat3x4": "float3x4",
            "mat4x2": "float4x2",
            "mat4x3": "float4x3",
            "mat4x4": "float4x4",
            "dmat2": "double2x2",
            "dmat3": "double3x3",
            "dmat4": "double4x4",
            "dmat2x2": "double2x2",
            "dmat2x3": "double2x3",
            "dmat2x4": "double2x4",
            "dmat3x2": "double3x2",
            "dmat3x3": "double3x3",
            "dmat3x4": "double3x4",
            "dmat4x2": "double4x2",
            "dmat4x3": "double4x3",
            "dmat4x4": "double4x4",
            "float": "float",
            "int": "int",
            "uint": "uint",
            "bool": "bool",
            "void": "void",
            "sampler": "SamplerState",
            "sampler1D": "Sampler1D<float4>",
            "sampler2D": "Sampler2D<float4>",
            "sampler3D": "Sampler3D<float4>",
            "samplerCube": "SamplerCube<float4>",
            "sampler2DArray": "Sampler2DArray<float4>",
            "samplerCubeArray": "SamplerCubeArray<float4>",
            "sampler2DMS": "Sampler2DMS<float4>",
            "sampler2DMSArray": "Sampler2DMSArray<float4>",
            "sampler2DShadow": "Sampler2DShadow",
            "sampler2DArrayShadow": "Sampler2DArrayShadow",
            "samplerCubeShadow": "SamplerCubeShadow",
            "samplerCubeArrayShadow": "SamplerCubeArrayShadow",
            "iimage2D": "RWTexture2D<int>",
            "iimage3D": "RWTexture3D<int>",
            "iimage2DArray": "RWTexture2DArray<int>",
            "iimage2DMS": "RWTexture2DMS<int>",
            "iimage2DMSArray": "RWTexture2DMSArray<int>",
            "uimage2D": "RWTexture2D<uint>",
            "uimage3D": "RWTexture3D<uint>",
            "uimage2DArray": "RWTexture2DArray<uint>",
            "uimage2DMS": "RWTexture2DMS<uint>",
            "uimage2DMSArray": "RWTexture2DMSArray<uint>",
            "image2D": "RWTexture2D<float4>",
            "image3D": "RWTexture3D<float4>",
            "image2DArray": "RWTexture2DArray<float4>",
            "image2DMS": "RWTexture2DMS<float4>",
            "image2DMSArray": "RWTexture2DMSArray<float4>",
        }

        return type_map.get(type_name, type_name)

    def generate_resource_call(self, func_name, args):
        if func_name == "imageLoad" and len(args) >= 2:
            image_name = self.generate_expression(args[0])
            coord = self.generate_expression(args[1])
            if len(args) >= 3:
                sample = self.generate_expression(args[2])
                return f"{image_name}[{coord}, {sample}]"
            return f"{image_name}[{coord}]"

        if func_name == "imageStore" and len(args) >= 3:
            image_name = self.generate_expression(args[0])
            coord = self.generate_expression(args[1])
            if len(args) >= 4:
                sample = self.generate_expression(args[2])
                value = self.generate_expression(args[3])
                return f"{image_name}[{coord}, {sample}] = {value}"
            value = self.generate_expression(args[2])
            return f"{image_name}[{coord}] = {value}"

        if func_name == "texture" and len(args) >= 2:
            texture_name = self.generate_expression(args[0])
            coord = self.generate_expression(args[1])
            if len(args) >= 3:
                bias = self.generate_expression(args[2])
                return f"{texture_name}.SampleBias({coord}, {bias})"
            return f"{texture_name}.Sample({coord})"

        if func_name == "textureLod" and len(args) >= 3:
            texture_name = self.generate_expression(args[0])
            coord = self.generate_expression(args[1])
            lod = self.generate_expression(args[2])
            return f"{texture_name}.SampleLevel({coord}, {lod})"

        if func_name == "textureGrad" and len(args) >= 4:
            texture_name = self.generate_expression(args[0])
            coord = self.generate_expression(args[1])
            ddx = self.generate_expression(args[2])
            ddy = self.generate_expression(args[3])
            return f"{texture_name}.SampleGrad({coord}, {ddx}, {ddy})"

        if func_name == "texelFetch" and len(args) >= 3:
            texture_name = self.generate_expression(args[0])
            coord = self.generate_expression(args[1])
            lod_or_sample = self.generate_expression(args[2])
            texture_type = self.get_expression_type(args[0])
            if self.is_multisample_sampler_type(texture_type):
                return f"{texture_name}[{coord}, {lod_or_sample}]"
            coord_constructor = self.texel_fetch_coord_constructor(texture_type)
            return f"{texture_name}.Load({coord_constructor}({coord}, {lod_or_sample}))"

        if func_name in {"textureSize", "imageSize"}:
            return self.generate_dimension_query(func_name, args)

        if func_name in {"textureSamples", "imageSamples"}:
            return self.generate_sample_count_query(func_name, args)

        return None

    def generate_dimension_query(self, func_name, args):
        if not args:
            return None

        resource_name = self.generate_expression(args[0])
        resource_type = self.resource_base_type(self.get_expression_type(args[0]))
        spec = self.dimension_query_spec(resource_type)
        if spec is None:
            return None

        helper_name = f"cgl_{func_name}_{resource_type}"
        self.register_helper_function(
            helper_name,
            self.build_dimension_query_helper(helper_name, resource_type, spec),
        )

        if spec["mip"]:
            lod = self.generate_expression(args[1]) if len(args) > 1 else "0"
            return f"{helper_name}({resource_name}, {lod})"
        return f"{helper_name}({resource_name})"

    def generate_sample_count_query(self, func_name, args):
        if not args:
            return None

        resource_name = self.generate_expression(args[0])
        resource_type = self.resource_base_type(self.get_expression_type(args[0]))
        spec = self.dimension_query_spec(resource_type)
        if spec is None or not spec["samples"]:
            return None

        helper_name = f"cgl_{func_name}_{resource_type}"
        self.register_helper_function(
            helper_name,
            self.build_sample_count_query_helper(helper_name, resource_type, spec),
        )
        return f"{helper_name}({resource_name})"

    def register_helper_function(self, name, source):
        if name not in self.helper_functions:
            self.helper_functions[name] = source

    def build_dimension_query_helper(self, helper_name, resource_type, spec):
        resource_slang_type = self.convert_type(resource_type)
        return_type = self.query_return_type(spec["dimensions"])
        params = f"{resource_slang_type} tex"
        if spec["mip"]:
            params += ", uint mipLevel"

        declarations = self.query_local_declarations(spec)
        get_dimensions_args = self.get_dimensions_args(spec)
        dimensions = ", ".join(spec["dimensions"])
        if len(spec["dimensions"]) == 1:
            return_value = spec["dimensions"][0]
        else:
            return_value = f"{return_type}({dimensions})"

        return (
            f"{return_type} {helper_name}({params})\n"
            "{\n"
            f"{declarations}"
            f"    tex.GetDimensions({get_dimensions_args});\n"
            f"    return {return_value};\n"
            "}"
        )

    def build_sample_count_query_helper(self, helper_name, resource_type, spec):
        resource_slang_type = self.convert_type(resource_type)
        declarations = self.query_local_declarations(spec)
        get_dimensions_args = self.get_dimensions_args(spec)
        return (
            f"int {helper_name}({resource_slang_type} tex)\n"
            "{\n"
            f"{declarations}"
            f"    tex.GetDimensions({get_dimensions_args});\n"
            "    return samples;\n"
            "}"
        )

    def query_return_type(self, dimensions):
        if len(dimensions) == 1:
            return "int"
        return f"int{len(dimensions)}"

    def query_local_declarations(self, spec):
        names = list(spec["dimensions"])
        if spec["samples"]:
            names.append("samples")
        if spec["mip"]:
            names.append("levels")
        return "".join(f"    int {name};\n" for name in names)

    def get_dimensions_args(self, spec):
        args = []
        if spec["mip"]:
            args.append("mipLevel")
        args.extend(spec["dimensions"])
        if spec["samples"]:
            args.append("samples")
        if spec["mip"]:
            args.append("levels")
        return ", ".join(args)

    def dimension_query_spec(self, type_name):
        specs = {
            "sampler1D": (("width",), True, False),
            "sampler1DArray": (("width", "elements"), True, False),
            "sampler2D": (("width", "height"), True, False),
            "sampler2DArray": (("width", "height", "elements"), True, False),
            "sampler3D": (("width", "height", "depth"), True, False),
            "samplerCube": (("width", "height"), True, False),
            "samplerCubeArray": (("width", "height", "elements"), True, False),
            "sampler2DMS": (("width", "height"), False, True),
            "sampler2DMSArray": (("width", "height", "elements"), False, True),
            "image2D": (("width", "height"), False, False),
            "iimage2D": (("width", "height"), False, False),
            "uimage2D": (("width", "height"), False, False),
            "image2DArray": (("width", "height", "elements"), False, False),
            "iimage2DArray": (("width", "height", "elements"), False, False),
            "uimage2DArray": (("width", "height", "elements"), False, False),
            "image3D": (("width", "height", "depth"), False, False),
            "iimage3D": (("width", "height", "depth"), False, False),
            "uimage3D": (("width", "height", "depth"), False, False),
            "image2DMS": (("width", "height"), False, True),
            "iimage2DMS": (("width", "height"), False, True),
            "uimage2DMS": (("width", "height"), False, True),
            "image2DMSArray": (("width", "height", "elements"), False, True),
            "iimage2DMSArray": (("width", "height", "elements"), False, True),
            "uimage2DMSArray": (("width", "height", "elements"), False, True),
        }
        spec = specs.get(type_name)
        if spec is None:
            return None
        dimensions, mip, samples = spec
        return {
            "dimensions": dimensions,
            "mip": mip,
            "samples": samples,
        }

    def get_expression_type(self, node):
        name = self.get_expression_name(node)
        if name is None:
            return None
        return self.variable_types.get(name)

    def get_expression_name(self, node):
        if isinstance(node, IdentifierNode):
            return node.name
        if isinstance(node, VariableNode):
            return node.name
        if isinstance(node, str):
            return node
        if isinstance(node, ArrayAccessNode):
            return self.get_expression_name(
                getattr(node, "array", getattr(node, "array_expr", None))
            )
        return None

    def resource_base_type(self, type_name):
        if not isinstance(type_name, str):
            return None
        return type_name.split("[", 1)[0]

    def is_multisample_sampler_type(self, type_name):
        return self.resource_base_type(type_name) in {
            "sampler2DMS",
            "sampler2DMSArray",
        }

    def texel_fetch_coord_constructor(self, type_name):
        base_type = self.resource_base_type(type_name)
        if base_type in {"sampler1D", "sampler1DArray"}:
            return "int2" if base_type == "sampler1D" else "int3"
        if base_type in {"sampler3D", "sampler2DArray"}:
            return "int4"
        return "int3"

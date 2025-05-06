from .VulkanAst import (
    AssignmentNode,
    BinaryOpNode,
    BreakNode,
    CaseNode,
    DefaultNode,
    DoWhileNode,
    ForNode,
    FunctionCallNode,
    FunctionNode,
    IfNode,
    LayoutNode,
    MemberAccessNode,
    ReturnNode,
    StructNode,
    SwitchNode,
    TernaryOpNode,
    UnaryOpNode,
    VariableNode,
    WhileNode,
)


class VulkanToCrossGLConverter:
    def __init__(self):
        self.type_map = {
            "void": "void",
            "vec2": "float2",
            "vec3": "float3",
            "vec4": "float4",
            "mat2": "float2x2",
            "mat3": "float3x3",
            "mat4": "float4x4",
            "int": "int",
            "ivec2": "int2",
            "ivec3": "int3",
            "ivec4": "int4",
            "uint": "uint",
            "uvec2": "uint2",
            "uvec3": "uint3",
            "uvec4": "uint4",
            "bool": "bool",
            "bvec2": "bool2",
            "bvec3": "bool3",
            "bvec4": "bool4",
            "float": "float",
            "double": "double",
            "sampler2D": "Texture2D",
            "samplerCube": "TextureCube",
            "sampler3D": "Texture3D",
            "sampler2DArray": "Texture2DArray",
            "samplerCubeArray": "TextureCubeArray",
            "sampler1D": "Texture1D",
            "sampler1DArray": "Texture1DArray",
            "image2D": "RWTexture2D",
            "image3D": "RWTexture3D",
            "imageCube": "RWTextureCube",
            "atomic_uint": "RWStructuredBuffer<uint>",
        }
        self.semantic_map = {
            # Vertex inputs
            "gl_VertexID": "SV_VertexID",
            "gl_InstanceID": "SV_InstanceID",
            # Vertex outputs
            "gl_Position": "SV_Position",
            # Fragment inputs
            "gl_FragCoord": "SV_Position",
            "gl_FrontFacing": "SV_IsFrontFace",
            "gl_PointCoord": "SV_PointCoord",
            "gl_PrimitiveID": "SV_PrimitiveID",
            # Fragment outputs
            "gl_FragColor": "SV_Target",
            "gl_FragData[0]": "SV_Target0",
            "gl_FragData[1]": "SV_Target1",
            "gl_FragData[2]": "SV_Target2",
            "gl_FragData[3]": "SV_Target3",
            "gl_FragDepth": "SV_Depth",
        }
        self.bitwise_op_map = {
            "&": "&",
            "|": "|",
            "^": "^",
            "~": "~",
            "<<": "<<",
            ">>": ">>",
        }
        self.indentation = 0
        self.code = []

    def get_indent(self):
        return "    " * self.indentation

    def generate(self, ast):
        code = "shader main {\n"
        # Generate layout definitions for uniforms, buffers, etc.
        for node in ast.functions:
            if isinstance(node, LayoutNode):
                code += self.generate_layout(node)
            elif isinstance(node, StructNode):
                code += self.generate_struct(node)
            elif isinstance(node, FunctionNode):
                # Determine if this is a vertex or fragment shader based on the function name
                if node.name == "main":
                    pass
                    # Check for gl_Position writes to identify a vertex shader
                    is_vertex_shader = False
                    for stmt in node.body:
                        if self.is_position_assignment(stmt):
                            is_vertex_shader = True
                            break

                    if is_vertex_shader:
                        code += "    // Vertex Shader\n"
                        code += "    vertex {\n"
                        code += self.generate_function(node)
                        code += "    }\n\n"
                    else:
                        code += "    // Fragment Shader\n"
                        code += "    fragment {\n"
                        code += self.generate_function(node)
                        code += "    }\n\n"
                else:
                    # Regular function
                    code += self.generate_function(node)

        code += "}\n"
        return code

    def is_position_assignment(self, stmt):
        """Check if a statement is assigning to gl_Position"""
        if isinstance(stmt, AssignmentNode):
            if isinstance(stmt.name, str) and "gl_Position" in stmt.name:
                return True
            elif hasattr(stmt.name, "name") and "gl_Position" in stmt.name.name:
                return True
        return False

    def generate_layout(self, node):
        code = ""
        layout_type = node.layout_type.lower() if node.layout_type else ""

        if layout_type == "uniform":
            # Handle uniform buffers
            if node.struct_fields:
                code += f"    cbuffer {node.variable_name or 'UniformBuffer'} {{\n"
                for field_type, field_name in node.struct_fields:
                    code += f"        {self.map_type(field_type)} {field_name};\n"
                code += "    }\n\n"
            else:
                # Single uniform
                code += f"    {self.map_type(node.data_type)} {node.variable_name};\n"
        elif layout_type == "buffer":
            # Handle storage buffers
            if node.struct_fields:
                code += f"    struct {node.variable_name or 'StorageBuffer'}_t {{\n"
                for field_type, field_name in node.struct_fields:
                    code += f"        {self.map_type(field_type)} {field_name};\n"
                code += "    };\n\n"
                code += f"    RWStructuredBuffer<{node.variable_name or 'StorageBuffer'}_t> {node.variable_name};\n\n"
        elif layout_type == "in" or layout_type == "out":
            # Input/output variables are handled differently in HLSL - typically as function parameters or return values
            pass

        return code

    def generate_struct(self, node):
        code = f"    struct {node.name} {{\n"
        for member in node.members:
            if isinstance(member, VariableNode):
                code += f"        {self.map_type(member.var_type)} {member.name};\n"
            elif isinstance(member, AssignmentNode):
                code += f"        {self.map_type(member.name.var_type)} {member.name.name};\n"
        code += "    }\n\n"
        return code

    def generate_function(self, node, indent=1):
        code = "  " * indent
        return_type = self.map_type(node.return_type)
        params = ", ".join(
            f"{self.map_type(p.var_type)} {p.name}" for p in node.parameters
        )
        code += f"    {return_type} {node.name}({params}) {{\n"
        code += self.generate_function_body(node.body, indent=indent + 1)
        code += "    }\n\n"
        return code

    def generate_function_body(self, body, indent=1):
        code = ""
        # Check if body is a list or a single node
        if not isinstance(body, list):
            body = [body]

        for stmt in body:
            code += "    " * indent
            if isinstance(stmt, VariableNode):
                code += f"{self.map_type(stmt.var_type)} {stmt.name};\n"
            elif isinstance(stmt, AssignmentNode):
                code += self.generate_assignment(stmt) + ";\n"
            elif isinstance(stmt, BinaryOpNode):
                code += f"{self.generate_expression(stmt.left)} {stmt.op} {self.generate_expression(stmt.right)};\n"
            elif isinstance(stmt, ReturnNode):
                if stmt.value:
                    code += f"return {self.generate_expression(stmt.value)};\n"
                else:
                    code += "return;\n"
            elif isinstance(stmt, ForNode):
                code += self.generate_for_loop(stmt, indent)
            elif isinstance(stmt, WhileNode):
                code += self.generate_while_loop(stmt, indent)
            elif isinstance(stmt, DoWhileNode):
                code += self.generate_do_while_loop(stmt, indent)
            elif isinstance(stmt, IfNode):
                code += self.generate_if_statement(stmt, indent)
            elif isinstance(stmt, SwitchNode):
                code += self.generate_switch_statement(stmt, indent)
            elif isinstance(stmt, FunctionCallNode):
                code += f"{self.generate_function_call(stmt)};\n"
            elif isinstance(stmt, BreakNode):
                code += "break;\n"
            elif isinstance(stmt, str):
                code += f"{stmt};\n"
            else:
                # For any unhandled statement type
                code += f"// Unhandled statement type: {type(stmt).__name__}\n"
        return code

    def generate_assignment(self, node):
        lhs = self.generate_expression(node.name)
        rhs = self.generate_expression(node.value)
        return f"{lhs} = {rhs}"

    def generate_expression(self, expr):
        if isinstance(expr, str):
            return expr
        elif isinstance(expr, int) or isinstance(expr, float):
            return str(expr)
        elif isinstance(expr, VariableNode):
            return f"{expr.name}"
        elif isinstance(expr, BinaryOpNode):
            left = self.generate_expression(expr.left)
            right = self.generate_expression(expr.right)

            # Handle bitwise operations
            if expr.op in self.bitwise_op_map:
                op = self.bitwise_op_map[expr.op]
                return f"({left} {op} {right})"

            return f"({left} {expr.op} {right})"
        elif isinstance(expr, UnaryOpNode):
            operand = self.generate_expression(expr.operand)
            if expr.op == "~":
                return f"(~{operand})"
            return f"{expr.op}{operand}"
        elif isinstance(expr, TernaryOpNode):
            condition = self.generate_expression(expr.condition)
            true_expr = self.generate_expression(expr.true_expr)
            false_expr = self.generate_expression(expr.false_expr)
            return f"({condition} ? {true_expr} : {false_expr})"
        elif isinstance(expr, FunctionCallNode):
            return self.generate_function_call(expr)
        elif isinstance(expr, MemberAccessNode):
            obj = self.generate_expression(expr.object)
            return f"{obj}.{expr.member}"
        else:
            # Handle any other expression type
            return str(expr)

    def generate_function_call(self, node):
        args = ", ".join(self.generate_expression(arg) for arg in node.args)
        return f"{node.name}({args})"

    def generate_for_loop(self, node, indent):
        init = (
            self.generate_expression(node.init)
            if isinstance(node.init, (BinaryOpNode, AssignmentNode))
            else node.init
        )
        condition = self.generate_expression(node.condition)
        update = self.generate_expression(node.update)

        code = f"for ({init}; {condition}; {update}) {{\n"
        code += self.generate_function_body(node.body, indent=indent + 1)
        code += "    " * indent + "}\n"
        return code

    def generate_while_loop(self, node, indent):
        condition = self.generate_expression(node.condition)

        code = f"while ({condition}) {{\n"
        code += self.generate_function_body(node.body, indent=indent + 1)
        code += "    " * indent + "}\n"
        return code

    def generate_do_while_loop(self, node, indent):
        condition = self.generate_expression(node.condition)

        code = "do {\n"
        code += self.generate_function_body(node.body, indent=indent + 1)
        code += "    " * indent + "} "
        code += f"while ({condition});\n"
        return code

    def generate_if_statement(self, node, indent):
        condition = self.generate_expression(node.if_condition)

        code = f"if ({condition}) {{\n"
        code += self.generate_function_body(node.if_body, indent=indent + 1)
        code += "    " * indent + "}"

        # Handle else-if chains
        if hasattr(node, "else_if_conditions") and node.else_if_conditions:
            for i in range(len(node.else_if_conditions)):
                else_if_condition = self.generate_expression(node.else_if_conditions[i])
                code += f" else if ({else_if_condition}) {{\n"
                code += self.generate_function_body(
                    node.else_if_bodies[i], indent=indent + 1
                )
                code += "    " * indent + "}"

        # Handle else block
        if node.else_body:
            code += " else {\n"
            code += self.generate_function_body(node.else_body, indent=indent + 1)
            code += "    " * indent + "}"

        code += "\n"
        return code

    def generate_switch_statement(self, node, indent):
        expression = self.generate_expression(node.expression)

        code = f"switch ({expression}) {{\n"

        for case in node.cases:
            if isinstance(case, CaseNode):
                value = self.generate_expression(case.value)
                code += "    " * (indent + 1) + f"case {value}:\n"
                code += self.generate_function_body(case.body, indent=indent + 2)
                code += "    " * (indent + 2) + "break;\n"
            elif isinstance(case, DefaultNode):
                code += "    " * (indent + 1) + "default:\n"
                code += self.generate_function_body(case.statements, indent=indent + 2)
                code += "    " * (indent + 2) + "break;\n"

        code += "    " * indent + "}\n"
        return code

    def map_type(self, vulkan_type):
        """Map Vulkan/GLSL types to HLSL/DirectX types"""
        if vulkan_type in self.type_map:
            return self.type_map[vulkan_type]
        return vulkan_type

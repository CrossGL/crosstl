from .MojoAst import *
from .MojoParser import *
from .MojoLexer import *


class MojoToCrossGLConverter:
    def __init__(self):
        self.type_map = {
            # Scalar Types
            "void": "void",
            "Int": "int",
            "Int8": "int8_t",
            "Int16": "int16_t",
            "Int32": "int",
            "Int64": "int64_t",
            "UInt": "uint",
            "UInt8": "uint8_t",
            "UInt16": "uint16_t",
            "UInt32": "uint",
            "UInt64": "uint64_t",
            "Float": "float",
            "Float16": "half",
            "Float32": "float",
            "Float64": "double",
            "Bool": "bool",
            # Vector Types (Mojo SIMD types)
            "SIMD[DType.float32, 2]": "vec2",
            "SIMD[DType.float32, 3]": "vec3",
            "SIMD[DType.float32, 4]": "vec4",
            "SIMD[DType.int32, 2]": "ivec2",
            "SIMD[DType.int32, 3]": "ivec3",
            "SIMD[DType.int32, 4]": "ivec4",
            "SIMD[DType.uint32, 2]": "uvec2",
            "SIMD[DType.uint32, 3]": "uvec3",
            "SIMD[DType.uint32, 4]": "uvec4",
            "SIMD[DType.bool, 2]": "bvec2",
            "SIMD[DType.bool, 3]": "bvec3",
            "SIMD[DType.bool, 4]": "bvec4",
            # Matrix Types
            "Matrix[DType.float32, 2, 2]": "mat2",
            "Matrix[DType.float32, 3, 3]": "mat3",
            "Matrix[DType.float32, 4, 4]": "mat4",
            "Matrix[DType.float16, 2, 2]": "half2x2",
            "Matrix[DType.float16, 3, 3]": "half3x3",
            "Matrix[DType.float16, 4, 4]": "half4x4",
            # Texture Types (hypothetical Mojo texture types)
            "Texture2D": "sampler2D",
            "TextureCube": "samplerCube",
            "Texture3D": "sampler3D",
            # Common simplified aliases
            "float2": "vec2",
            "float3": "vec3",
            "float4": "vec4",
            "int2": "ivec2",
            "int3": "ivec3",
            "int4": "ivec4",
            "uint2": "uvec2",
            "uint3": "uvec3",
            "uint4": "uvec4",
            "bool2": "bvec2",
            "bool3": "bvec3",
            "bool4": "bvec4",
            "mat2": "mat2",
            "mat3": "mat3",
            "mat4": "mat4",
        }

        self.semantic_map = {
            # Vertex attributes
            "position": "Position",
            "normal": "Normal",
            "tangent": "Tangent",
            "binormal": "Binormal",
            "texcoord": "TexCoord",
            "texcoord0": "TexCoord0",
            "texcoord1": "TexCoord1",
            "texcoord2": "TexCoord2",
            "texcoord3": "TexCoord3",
            "color": "Color",
            "color0": "Color0",
            "color1": "Color1",
            "color2": "Color2",
            "color3": "Color3",
            # Built-in variables
            "vertex_id": "gl_VertexID",
            "instance_id": "gl_InstanceID",
            "gl_position": "gl_Position",
            "gl_fragcolor": "gl_FragColor",
            "gl_fragdepth": "gl_FragDepth",
            "gl_frontfacing": "gl_IsFrontFace",
            "gl_pointcoord": "gl_PointCoord",
            "gl_primitiveId": "gl_PrimitiveID",
        }

        self.function_map = {
            # Math functions
            "math.sqrt": "sqrt",
            "math.pow": "pow",
            "math.sin": "sin",
            "math.cos": "cos",
            "math.tan": "tan",
            "math.abs": "abs",
            "math.floor": "floor",
            "math.ceil": "ceil",
            "math.min": "min",
            "math.max": "max",
            "math.clamp": "clamp",
            # SIMD functions
            "simd.dot": "dot",
            "simd.cross": "cross",
            "simd.normalize": "normalize",
            "simd.length": "length",
            "simd.reflect": "reflect",
            "simd.refract": "refract",
        }

    def generate(self, ast):
        code = "shader main {\n"

        # Generate imports as comments
        if hasattr(ast, "functions") and ast.functions:
            imports = [f for f in ast.functions if isinstance(f, ImportNode)]
            if imports:
                code += "    // Imports\n"
                for imp in imports:
                    code += f"    // import {imp.module_name}"
                    if imp.alias:
                        code += f" as {imp.alias}"
                    code += "\n"
                code += "\n"

        # Generate struct definitions
        if hasattr(ast, "functions"):
            structs = [f for f in ast.functions if isinstance(f, StructNode)]
            for struct_node in structs:
                code += f"    // Structs\n"
                code += f"    struct {struct_node.name} {{\n"
                for member in struct_node.members:
                    semantic = (
                        self.map_semantic(member.attributes)
                        if hasattr(member, "attributes")
                        else ""
                    )
                    code += f"        {self.map_type(member.vtype)} {member.name}{semantic};\n"
                code += "    }\n\n"

        # Generate constant buffers
        if hasattr(ast, "functions"):
            cbuffers = [f for f in ast.functions if isinstance(f, ConstantBufferNode)]
            if cbuffers:
                code += "    // Constant Buffers\n"
                for cbuffer in cbuffers:
                    code += f"    cbuffer {cbuffer.name} {{\n"
                    for member in cbuffer.members:
                        code += (
                            f"        {self.map_type(member.vtype)} {member.name};\n"
                        )
                    code += "    }\n\n"

        # Generate functions
        if hasattr(ast, "functions"):
            functions = [f for f in ast.functions if isinstance(f, FunctionNode)]
            for func in functions:
                if (
                    func.qualifier == "vertex"
                    or self.has_vertex_attribute(func)
                    or "vertex" in func.name
                ):
                    code += "    // Vertex Shader\n"
                    code += "    vertex {\n"
                    code += self.generate_function(func, indent=2)
                    code += "    }\n\n"
                elif (
                    func.qualifier == "fragment"
                    or self.has_fragment_attribute(func)
                    or "fragment" in func.name
                ):
                    code += "    // Fragment Shader\n"
                    code += "    fragment {\n"
                    code += self.generate_function(func, indent=2)
                    code += "    }\n\n"
                elif (
                    func.qualifier == "compute"
                    or self.has_compute_attribute(func)
                    or "compute" in func.name
                ):
                    code += "    // Compute Shader\n"
                    code += "    compute {\n"
                    code += self.generate_function(func, indent=2)
                    code += "    }\n\n"
                else:
                    code += self.generate_function(func, indent=1)

        code += "}\n"
        return code

    def has_vertex_attribute(self, func):
        """Check if function has vertex-related attributes"""
        if not hasattr(func, "attributes") or not func.attributes:
            return False
        for attr in func.attributes:
            if hasattr(attr, "name") and attr.name in ["vertex", "vertex_main"]:
                return True
        return False

    def has_fragment_attribute(self, func):
        """Check if function has fragment-related attributes"""
        if not hasattr(func, "attributes") or not func.attributes:
            return False
        for attr in func.attributes:
            if hasattr(attr, "name") and attr.name in ["fragment", "fragment_main"]:
                return True
        return False

    def has_compute_attribute(self, func):
        """Check if function has compute-related attributes"""
        if not hasattr(func, "attributes") or not func.attributes:
            return False
        for attr in func.attributes:
            if hasattr(attr, "name") and attr.name in ["compute", "compute_main"]:
                return True
        return False

    def generate_function(self, func, indent=1):
        code = ""
        indent_str = "    " * indent

        # Generate function parameters
        params = []
        if hasattr(func, "params") and func.params:
            for p in func.params:
                param_str = f"{self.map_type(p.vtype)} {p.name}"
                if hasattr(p, "attributes") and p.attributes:
                    semantic = self.map_semantic(p.attributes)
                    if semantic:
                        param_str += f" {semantic}"
                params.append(param_str)

        params_str = ", ".join(params) if params else ""
        return_type = self.map_type(func.return_type) if func.return_type else "void"

        # Generate function attributes
        func_attributes = ""
        if hasattr(func, "attributes") and func.attributes:
            func_attributes = self.map_semantic(func.attributes)

        code += (
            f"{indent_str}{return_type} {func.name}({params_str}){func_attributes} {{\n"
        )

        # Generate function body
        if hasattr(func, "body") and func.body:
            code += self.generate_function_body(func.body, indent + 1)

        code += f"{indent_str}}}\n\n"
        return code

    def generate_function_body(self, body, indent=0):
        code = ""
        indent_str = "    " * indent

        for stmt in body:
            code += indent_str
            if isinstance(stmt, VariableDeclarationNode):
                code += self.generate_variable_declaration(stmt) + ";\n"
            elif isinstance(stmt, VariableNode):
                if hasattr(stmt, "vtype") and stmt.vtype:
                    code += f"{self.map_type(stmt.vtype)} {stmt.name};\n"
                else:
                    code += f"{stmt.name};\n"
            elif isinstance(stmt, AssignmentNode):
                code += self.generate_assignment(stmt) + ";\n"
            elif isinstance(stmt, ReturnNode):
                code += f"return {self.generate_expression(stmt.value)};\n"
            elif isinstance(stmt, BinaryOpNode):
                code += f"{self.generate_expression(stmt)};\n"
            elif isinstance(stmt, ForNode):
                code += self.generate_for_loop(stmt, indent)
            elif isinstance(stmt, WhileNode):
                code += self.generate_while_loop(stmt, indent)
            elif isinstance(stmt, IfNode):
                code += self.generate_if_statement(stmt, indent)
            elif isinstance(stmt, SwitchNode):
                code += self.generate_switch_statement(stmt, indent)
            elif isinstance(stmt, FunctionCallNode):
                code += f"{self.generate_expression(stmt)};\n"
            elif isinstance(stmt, str):
                code += f"{stmt};\n"
            else:
                # For any unhandled statement type
                code += f"// Unhandled statement type: {type(stmt).__name__}\n"

        return code

    def generate_variable_declaration(self, node):
        var_type = "var" if node.var_type == "var" else "let"
        if hasattr(node, "initial_value") and node.initial_value:
            return f"{var_type} {node.name} = {self.generate_expression(node.initial_value)}"
        else:
            return f"{var_type} {node.name}"

    def generate_assignment(self, node):
        left = self.generate_expression(node.left)
        right = self.generate_expression(node.right)
        op = node.operator if hasattr(node, "operator") else "="
        return f"{left} {op} {right}"

    def generate_for_loop(self, node, indent):
        indent_str = "    " * indent
        init = self.generate_expression(node.init) if node.init else ""
        condition = (
            self.generate_expression(node.condition) if node.condition else "true"
        )
        update = self.generate_expression(node.update) if node.update else ""

        code = f"for ({init}; {condition}; {update}) {{\n"
        if hasattr(node, "body") and node.body:
            code += self.generate_function_body(node.body, indent + 1)
        code += indent_str + "}\n"
        return code

    def generate_while_loop(self, node, indent):
        indent_str = "    " * indent
        condition = (
            self.generate_expression(node.condition) if node.condition else "true"
        )

        code = f"while ({condition}) {{\n"
        if hasattr(node, "body") and node.body:
            code += self.generate_function_body(node.body, indent + 1)
        code += indent_str + "}\n"
        return code

    def generate_if_statement(self, node, indent):
        indent_str = "    " * indent
        condition = self.generate_expression(node.condition)

        code = f"if ({condition}) {{\n"
        if hasattr(node, "if_body") and node.if_body:
            code += self.generate_function_body(node.if_body, indent + 1)
        code += indent_str + "}"

        if hasattr(node, "else_body") and node.else_body:
            # Check if else_body is a list
            if isinstance(node.else_body, list):
                if len(node.else_body) == 1 and isinstance(node.else_body[0], IfNode):
                    # This is an elif statement
                    code += " else "
                    code += self.generate_if_statement(node.else_body[0], indent)
                else:
                    # Regular else with multiple statements
                    code += " else {\n"
                    code += self.generate_function_body(node.else_body, indent + 1)
                    code += indent_str + "}"
            elif isinstance(node.else_body, IfNode):
                # Direct IfNode for elif
                code += " else "
                code += self.generate_if_statement(node.else_body, indent)
            else:
                # Regular else body
                code += " else {\n"
                code += self.generate_function_body([node.else_body], indent + 1)
                code += indent_str + "}"

        code += "\n"
        return code

    def generate_switch_statement(self, node, indent):
        indent_str = "    " * indent
        expression = self.generate_expression(node.expression)

        code = f"switch ({expression}) {{\n"

        # Generate case statements
        if hasattr(node, "cases") and node.cases:
            for case in node.cases:
                if hasattr(case, "condition") and case.condition is not None:
                    case_value = self.generate_expression(case.condition)
                    code += indent_str + f"    case {case_value}:\n"
                else:
                    code += indent_str + "    default:\n"

                if hasattr(case, "body") and case.body:
                    code += self.generate_function_body(case.body, indent + 2)
                code += indent_str + "        break;\n"

        code += indent_str + "}\n"
        return code

    def generate_expression(self, expr):
        if expr is None:
            return ""
        elif isinstance(expr, str):
            return expr
        elif isinstance(expr, (int, float, bool)):
            return str(expr)
        elif isinstance(expr, VariableNode):
            if hasattr(expr, "vtype") and expr.vtype:
                # If this is a type declaration, format it properly
                return f"{self.map_type(expr.vtype)} {expr.name}"
            else:
                # Just a variable reference
                return expr.name
        elif isinstance(expr, VariableDeclarationNode):
            return self.generate_variable_declaration(expr)
        elif isinstance(expr, AssignmentNode):
            return self.generate_assignment(expr)
        elif isinstance(expr, BinaryOpNode):
            left = self.generate_expression(expr.left)
            right = self.generate_expression(expr.right)
            op = expr.op if hasattr(expr, "op") else "+"
            return f"({left} {op} {right})"
        elif isinstance(expr, UnaryOpNode):
            operand = self.generate_expression(expr.operand)
            op = expr.op if hasattr(expr, "op") else "+"
            return f"({op}{operand})"
        elif isinstance(expr, FunctionCallNode):
            args = []
            if hasattr(expr, "args") and expr.args:
                args = [self.generate_expression(arg) for arg in expr.args]
            args_str = ", ".join(args)
            func_name = self.map_function(expr.name)
            return f"{func_name}({args_str})"
        elif isinstance(expr, MemberAccessNode):
            obj = self.generate_expression(expr.object)
            return f"{obj}.{expr.member}"
        elif isinstance(expr, TernaryOpNode):
            condition = self.generate_expression(expr.condition)
            true_expr = self.generate_expression(expr.true_expr)
            false_expr = self.generate_expression(expr.false_expr)
            return f"({condition} ? {true_expr} : {false_expr})"
        elif isinstance(expr, VectorConstructorNode):
            args = []
            if hasattr(expr, "args") and expr.args:
                args = [self.generate_expression(arg) for arg in expr.args]
            args_str = ", ".join(args)
            type_name = self.map_type(expr.type_name)
            return f"{type_name}({args_str})"
        elif isinstance(expr, ArrayAccessNode):
            array = self.generate_expression(expr.array)
            index = self.generate_expression(expr.index)
            return f"{array}[{index}]"
        else:
            # For any unhandled expression type
            return f"/* Unhandled expression: {type(expr).__name__} */"

    def map_type(self, mojo_type):
        if mojo_type is None:
            return "void"
        return self.type_map.get(mojo_type, mojo_type)

    def map_semantic(self, attributes):
        if not attributes:
            return ""

        for attr in attributes:
            if hasattr(attr, "name"):
                semantic = self.semantic_map.get(attr.name, attr.name)
                return f"@ {semantic}"
        return ""

    def map_function(self, func_name):
        return self.function_map.get(func_name, func_name)

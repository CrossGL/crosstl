from .RustAst import *
from .RustParser import *
from .RustLexer import *


class RustToCrossGLConverter:
    def __init__(self):
        self.type_map = {
            # Rust primitive types to CrossGL
            "()": "void",
            "f32": "float",
            "f64": "double",
            "i32": "int",
            "i64": "int64_t",
            "u32": "uint",
            "u64": "uint64_t",
            "i8": "int8_t",
            "u8": "uint8_t",
            "i16": "int16_t",
            "u16": "uint16_t",
            "bool": "bool",
            "usize": "uint",
            "isize": "int",
            # Rust vector types to CrossGL
            "Vec2<f32>": "vec2",
            "Vec3<f32>": "vec3",
            "Vec4<f32>": "vec4",
            "Vec2<i32>": "ivec2",
            "Vec3<i32>": "ivec3",
            "Vec4<i32>": "ivec4",
            "Vec2<u32>": "uvec2",
            "Vec3<u32>": "uvec3",
            "Vec4<u32>": "uvec4",
            "Vec2<bool>": "bvec2",
            "Vec3<bool>": "bvec3",
            "Vec4<bool>": "bvec4",
            # Simplified vector names
            "Vec2": "vec2",
            "Vec3": "vec3",
            "Vec4": "vec4",
            # Rust matrix types to CrossGL
            "Mat2<f32>": "mat2",
            "Mat3<f32>": "mat3",
            "Mat4<f32>": "mat4",
            "Mat2": "mat2",
            "Mat3": "mat3",
            "Mat4": "mat4",
            # GPU-specific types
            "Texture2D": "sampler2D",
            "TextureCube": "samplerCube",
            "Sampler": "sampler",
            # Reference types (stripped in CrossGL)
            "&f32": "float",
            "&i32": "int",
            "&u32": "uint",
            "&bool": "bool",
            "&mut f32": "float",
            "&mut i32": "int",
            "&mut u32": "uint",
            "&mut bool": "bool",
        }

        self.semantic_map = {
            # Rust shader attributes to CrossGL semantics
            "vertex_position": "Position",
            "vertex_normal": "Normal",
            "vertex_tangent": "Tangent",
            "vertex_binormal": "Binormal",
            "vertex_texcoord": "TexCoord",
            "vertex_texcoord0": "TexCoord0",
            "vertex_texcoord1": "TexCoord1",
            "vertex_texcoord2": "TexCoord2",
            "vertex_texcoord3": "TexCoord3",
            "vertex_color": "Color",
            "instance_id": "InstanceID",
            "vertex_id": "VertexID",
            # Fragment shader semantics
            "fragment_position": "gl_Position",
            "fragment_color": "gl_FragColor",
            "fragment_depth": "gl_FragDepth",
            "front_face": "gl_IsFrontFace",
            "primitive_id": "gl_PrimitiveID",
            "point_coord": "gl_PointCoord",
            # Compute shader semantics
            "local_invocation_id": "gl_LocalInvocationID",
            "global_invocation_id": "gl_GlobalInvocationID",
            "workgroup_id": "gl_WorkGroupID",
            "num_workgroups": "gl_NumWorkGroups",
        }

        self.function_map = {
            # Rust math functions to CrossGL
            "abs": "abs",
            "min": "min",
            "max": "max",
            "clamp": "clamp",
            "floor": "floor",
            "ceil": "ceil",
            "round": "round",
            "sqrt": "sqrt",
            "pow": "pow",
            "exp": "exp",
            "exp2": "exp2",
            "log": "log",
            "log2": "log2",
            "sin": "sin",
            "cos": "cos",
            "tan": "tan",
            "asin": "asin",
            "acos": "acos",
            "atan": "atan",
            "atan2": "atan2",
            "sinh": "sinh",
            "cosh": "cosh",
            "tanh": "tanh",
            "degrees": "degrees",
            "radians": "radians",
            # Vector operations
            "dot": "dot",
            "cross": "cross",
            "length": "length",
            "normalize": "normalize",
            "distance": "distance",
            "reflect": "reflect",
            "refract": "refract",
            "faceforward": "faceforward",
            # Matrix operations
            "transpose": "transpose",
            "determinant": "determinant",
            "inverse": "inverse",
            # Texture sampling
            "sample": "texture",
            "sample_level": "textureLod",
            "sample_grad": "textureGrad",
            "sample_offset": "textureOffset",
        }

        self.attribute_map = {
            # Rust shader attributes to CrossGL qualifiers
            "vertex_shader": "vertex",
            "fragment_shader": "fragment",
            "compute_shader": "compute",
            "binding": "binding",
            "location": "location",
            "group": "group",
            "builtin": "builtin",
            "stage": "stage",
            "workgroup_size": "workgroup_size",
        }

        self.indentation = 0
        self.code = []

    def get_indent(self):
        return "    " * self.indentation

    def visit(self, node):
        if isinstance(node, StructNode):
            return self.visit_StructNode(node)
        elif isinstance(node, FunctionNode):
            return self.visit_FunctionNode(node)
        elif isinstance(node, BinaryOpNode):
            return self.visit_BinaryOpNode(node)
        elif isinstance(node, UnaryOpNode):
            return self.visit_UnaryOpNode(node)
        elif isinstance(node, ImplNode):
            return self.visit_ImplNode(node)
        elif isinstance(node, UseNode):
            return self.visit_UseNode(node)
        elif isinstance(node, ConstNode):
            return self.visit_ConstNode(node)
        elif isinstance(node, StaticNode):
            return self.visit_StaticNode(node)

        # For other node types, use existing methods
        if hasattr(self, f"generate_{type(node).__name__}"):
            method = getattr(self, f"generate_{type(node).__name__}")
            return method(node)
        return self.generate_expression(node)

    def generate(self, ast):
        code = "shader main {\n"

        # Generate use statements as comments
        for use_stmt in ast.use_statements:
            code += f"    // use {use_stmt.path}\n"

        # Generate constants and statics
        for global_var in ast.global_variables:
            if isinstance(global_var, ConstNode):
                code += f"    const {self.map_type(global_var.vtype)} {global_var.name} = {self.generate_expression(global_var.value)};\n"
            elif isinstance(global_var, StaticNode):
                mutability = "mut " if global_var.is_mutable else ""
                code += f"    static {mutability}{self.map_type(global_var.vtype)} {global_var.name} = {self.generate_expression(global_var.value)};\n"

        # Generate structs
        for struct in ast.structs:
            if isinstance(struct, StructNode):
                code += f"    struct {struct.name} {{\n"
                for member in struct.members:
                    semantic = self.get_semantic_from_attributes(member.attributes)
                    type_str = self.map_type(member.vtype)
                    code += f"        {type_str} {member.name}{semantic};\n"
                code += "    }\n\n"

        # Generate functions (including shader entry points)
        for func in ast.functions:
            shader_type = self.get_shader_type_from_attributes(func.attributes)
            if shader_type:
                code += f"    // {shader_type.title()} Shader\n"
                code += f"    {shader_type} {{\n"
                code += self.generate_function_body(func.body, indent=2)
                code += "    }\n\n"
            else:
                # Regular function
                code += self.generate_function(func, indent=1)

        # Generate impl blocks as helper functions
        for impl_block in ast.impl_blocks:
            code += f"    // Implementation for {impl_block.struct_name}\n"
            for func in impl_block.functions:
                code += self.generate_function(
                    func, indent=1, struct_name=impl_block.struct_name
                )

        code += "}\n"
        return code

    def generate_function(self, func, indent=1, struct_name=None):
        code = ""
        indent_str = "    " * indent

        # Generate function signature
        params = []
        for param in func.params:
            param_type = self.map_type(param.vtype)
            params.append(f"{param_type} {param.name}")

        params_str = ", ".join(params)
        return_type = self.map_type(func.return_type)

        if struct_name:
            func_name = f"{struct_name}_{func.name}"
        else:
            func_name = func.name

        code += f"{indent_str}{return_type} {func_name}({params_str}) {{\n"
        code += self.generate_function_body(func.body, indent=indent + 1)
        code += f"{indent_str}}}\n\n"

        return code

    def generate_function_body(self, body, indent=1):
        code = ""
        indent_str = "    " * indent

        for stmt in body:
            if isinstance(stmt, LetNode):
                code += self.generate_let_statement(stmt, indent)
            elif isinstance(stmt, AssignmentNode):
                code += f"{indent_str}{self.generate_assignment(stmt)};\n"
            elif isinstance(stmt, ReturnNode):
                if stmt.value:
                    code += (
                        f"{indent_str}return {self.generate_expression(stmt.value)};\n"
                    )
                else:
                    code += f"{indent_str}return;\n"
            elif isinstance(stmt, IfNode):
                code += self.generate_if_statement(stmt, indent)
            elif isinstance(stmt, ForNode):
                code += self.generate_for_loop(stmt, indent)
            elif isinstance(stmt, WhileNode):
                code += self.generate_while_loop(stmt, indent)
            elif isinstance(stmt, LoopNode):
                code += self.generate_loop(stmt, indent)
            elif isinstance(stmt, MatchNode):
                code += self.generate_match_statement(stmt, indent)
            elif isinstance(stmt, BreakNode):
                code += f"{indent_str}break;\n"
            elif isinstance(stmt, ContinueNode):
                code += f"{indent_str}continue;\n"
            elif isinstance(stmt, FunctionCallNode):
                code += f"{indent_str}{self.generate_expression(stmt)};\n"
            elif isinstance(stmt, BinaryOpNode):
                code += f"{indent_str}{self.generate_expression(stmt)};\n"
            elif isinstance(stmt, str):
                code += f"{indent_str}{stmt};\n"
            else:
                # Handle other statement types
                expr = self.generate_expression(stmt)
                if expr:
                    code += f"{indent_str}{expr};\n"

        return code

    def generate_let_statement(self, stmt, indent):
        indent_str = "    " * indent
        type_str = ""

        if stmt.vtype:
            type_str = f"{self.map_type(stmt.vtype)} "
        elif stmt.value:
            # Try to infer type from value
            type_str = ""  # Let CrossGL infer the type

        if stmt.value:
            value_str = self.generate_expression(stmt.value)
            return f"{indent_str}{type_str}{stmt.name} = {value_str};\n"
        else:
            return f"{indent_str}{type_str}{stmt.name};\n"

    def generate_assignment(self, node):
        left = self.generate_expression(node.left)
        right = self.generate_expression(node.right)
        return f"{left} {node.operator} {right}"

    def generate_if_statement(self, node, indent):
        indent_str = "    " * indent
        condition = self.generate_expression(node.condition)

        code = f"{indent_str}if ({condition}) {{\n"
        code += self.generate_function_body(node.if_body, indent + 1)
        code += f"{indent_str}}}"

        if node.else_body:
            if (
                isinstance(node.else_body, list)
                and len(node.else_body) == 1
                and isinstance(node.else_body[0], IfNode)
            ):
                # else if
                code += " else "
                code += self.generate_if_statement(node.else_body[0], 0).lstrip()
            else:
                # else block
                code += " else {\n"
                if isinstance(node.else_body, list):
                    code += self.generate_function_body(node.else_body, indent + 1)
                else:
                    code += self.generate_function_body([node.else_body], indent + 1)
                code += f"{indent_str}}}"

        code += "\n"
        return code

    def generate_for_loop(self, node, indent):
        indent_str = "    " * indent
        pattern = node.pattern
        iterable = self.generate_expression(node.iterable)

        # Convert Rust for-in loop to C-style for loop
        code = f"{indent_str}for (int {pattern} = 0; {pattern} < {iterable}; {pattern}++) {{\n"
        code += self.generate_function_body(node.body, indent + 1)
        code += f"{indent_str}}}\n"

        return code

    def generate_while_loop(self, node, indent):
        indent_str = "    " * indent
        condition = self.generate_expression(node.condition)

        code = f"{indent_str}while ({condition}) {{\n"
        code += self.generate_function_body(node.body, indent + 1)
        code += f"{indent_str}}}\n"

        return code

    def generate_loop(self, node, indent):
        indent_str = "    " * indent

        # Convert Rust infinite loop to while(true)
        code = f"{indent_str}while (true) {{\n"
        code += self.generate_function_body(node.body, indent + 1)
        code += f"{indent_str}}}\n"

        return code

    def generate_match_statement(self, node, indent):
        indent_str = "    " * indent
        expression = self.generate_expression(node.expression)

        # Convert Rust match to switch statement
        code = f"{indent_str}switch ({expression}) {{\n"

        for arm in node.arms:
            pattern = self.generate_expression(arm.pattern)
            code += f"{indent_str}    case {pattern}:\n"
            code += self.generate_function_body(arm.body, indent + 2)
            code += f"{indent_str}        break;\n"

        code += f"{indent_str}}}\n"
        return code

    def generate_expression(self, expr):
        if isinstance(expr, str):
            return expr
        elif isinstance(expr, VariableNode):
            return expr.name
        elif isinstance(expr, BinaryOpNode):
            left = self.generate_expression(expr.left)
            right = self.generate_expression(expr.right)
            return f"({left} {expr.op} {right})"
        elif isinstance(expr, UnaryOpNode):
            operand = self.generate_expression(expr.operand)
            return f"({expr.op}{operand})"
        elif isinstance(expr, FunctionCallNode):
            if isinstance(expr.name, str):
                func_name = self.map_function(expr.name)
            else:
                func_name = self.generate_expression(expr.name)

            args = ", ".join(self.generate_expression(arg) for arg in expr.args)
            return f"{func_name}({args})"
        elif isinstance(expr, MemberAccessNode):
            obj = self.generate_expression(expr.object)
            return f"{obj}.{expr.member}"
        elif isinstance(expr, ArrayAccessNode):
            array = self.generate_expression(expr.array)
            index = self.generate_expression(expr.index)
            return f"{array}[{index}]"
        elif isinstance(expr, VectorConstructorNode):
            args = ", ".join(self.generate_expression(arg) for arg in expr.args)
            type_name = self.map_type(expr.type_name)
            return f"{type_name}({args})"
        elif isinstance(expr, TernaryOpNode):
            condition = self.generate_expression(expr.condition)
            true_expr = self.generate_expression(expr.true_expr)
            false_expr = self.generate_expression(expr.false_expr)
            return f"({condition} ? {true_expr} : {false_expr})"
        elif isinstance(expr, CastNode):
            expression = self.generate_expression(expr.expression)
            target_type = self.map_type(expr.target_type)
            return f"({target_type}){expression}"
        elif isinstance(expr, ReferenceNode):
            # References are handled differently in CrossGL
            return self.generate_expression(expr.expression)
        elif isinstance(expr, DereferenceNode):
            # Dereferences are typically not needed in CrossGL
            return self.generate_expression(expr.expression)
        elif isinstance(expr, TupleNode):
            # Tuples might not be directly supported, convert to struct or multiple variables
            elements = ", ".join(
                self.generate_expression(elem) for elem in expr.elements
            )
            return f"({elements})"
        elif isinstance(expr, ArrayNode):
            elements = ", ".join(
                self.generate_expression(elem) for elem in expr.elements
            )
            return f"{{{elements}}}"
        elif isinstance(expr, BlockNode):
            # Block expressions might need special handling
            if expr.expression:
                return self.generate_expression(expr.expression)
            else:
                return ""
        elif isinstance(expr, (int, float, bool)):
            return str(expr).lower() if isinstance(expr, bool) else str(expr)
        else:
            return str(expr)

    def map_type(self, rust_type):
        if not rust_type:
            return "void"

        # Handle generic types
        if "<" in rust_type and ">" in rust_type:
            base_type = rust_type.split("<")[0]
            if base_type in ["Vec2", "Vec3", "Vec4"]:
                return self.type_map.get(
                    rust_type, self.type_map.get(base_type, rust_type)
                )

        return self.type_map.get(rust_type, rust_type)

    def map_function(self, rust_func):
        return self.function_map.get(rust_func, rust_func)

    def get_shader_type_from_attributes(self, attributes):
        if not attributes:
            return None

        for attr in attributes:
            if isinstance(attr, AttributeNode):
                mapped = self.attribute_map.get(attr.name)
                if mapped in ["vertex", "fragment", "compute"]:
                    return mapped
        return None

    def get_semantic_from_attributes(self, attributes):
        if not attributes:
            return ""

        for attr in attributes:
            if isinstance(attr, AttributeNode):
                if attr.name in self.semantic_map:
                    return f" @ {self.semantic_map[attr.name]}"
                elif attr.name == "location" and attr.args:
                    return f" @ location({attr.args[0]})"
                elif attr.name == "binding" and attr.args:
                    return f" @ binding({attr.args[0]})"
        return ""

    def visit_StructNode(self, node):
        code = f"struct {node.name} {{\n"
        self.indentation += 1

        for member in node.members:
            semantic = self.get_semantic_from_attributes(member.attributes)
            type_str = self.map_type(member.vtype)
            code += f"{self.get_indent()}{type_str} {member.name}{semantic};\n"

        self.indentation -= 1
        code += f"{self.get_indent()}}}\n"
        return code

    def visit_FunctionNode(self, node):
        shader_type = self.get_shader_type_from_attributes(node.attributes)
        if shader_type:
            return f"{shader_type} {{\n{self.generate_function_body(node.body, 1)}}}\n"
        else:
            return self.generate_function(node, 0)

    def visit_BinaryOpNode(self, node):
        left = self.generate_expression(node.left)
        right = self.generate_expression(node.right)
        return f"({left} {node.op} {right})"

    def visit_UnaryOpNode(self, node):
        operand = self.generate_expression(node.operand)
        return f"({node.op}{operand})"

    def visit_ImplNode(self, node):
        code = f"// Implementation for {node.struct_name}\n"
        for func in node.functions:
            code += self.generate_function(func, 0, node.struct_name)
        return code

    def visit_UseNode(self, node):
        return f"// use {node.path}\n"

    def visit_ConstNode(self, node):
        type_str = self.map_type(node.vtype)
        value = self.generate_expression(node.value)
        return f"const {type_str} {node.name} = {value};\n"

    def visit_StaticNode(self, node):
        mutability = "mut " if node.is_mutable else ""
        type_str = self.map_type(node.vtype)
        value = self.generate_expression(node.value)
        return f"static {mutability}{type_str} {node.name} = {value};\n"

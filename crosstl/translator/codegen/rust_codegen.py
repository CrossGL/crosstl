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


class RustCodeGen:
    def __init__(self):
        self.current_shader = None
        self.type_mapping = {
            # Scalar Types
            "void": "()",
            "int": "i32",
            "short": "i16",
            "long": "i64",
            "uint": "u32",
            "ushort": "u16",
            "ulong": "u64",
            "float": "f32",
            "double": "f64",
            "half": "f16",
            "bool": "bool",
            # Vector Types (using GPU-style vector types)
            "vec2": "Vec2<f32>",
            "vec3": "Vec3<f32>",
            "vec4": "Vec4<f32>",
            "ivec2": "Vec2<i32>",
            "ivec3": "Vec3<i32>",
            "ivec4": "Vec4<i32>",
            "uvec2": "Vec2<u32>",
            "uvec3": "Vec3<u32>",
            "uvec4": "Vec4<u32>",
            "bvec2": "Vec2<bool>",
            "bvec3": "Vec3<bool>",
            "bvec4": "Vec4<bool>",
            # Matrix Types
            "mat2": "Mat2<f32>",
            "mat3": "Mat3<f32>",
            "mat4": "Mat4<f32>",
            # Texture Types
            "sampler2D": "Texture2D<f32>",
            "samplerCube": "TextureCube<f32>",
            "sampler": "Sampler",
        }

        self.semantic_map = {
            # Vertex attributes
            "gl_VertexID": "vertex_id",
            "gl_InstanceID": "instance_id",
            "gl_Position": "position",
            "gl_PointSize": "point_size",
            "gl_ClipDistance": "clip_distance",
            # Fragment attributes
            "gl_FragColor": "target(0)",
            "gl_FragColor0": "target(0)",
            "gl_FragColor1": "target(1)",
            "gl_FragColor2": "target(2)",
            "gl_FragColor3": "target(3)",
            "gl_FragDepth": "depth(any)",
            "gl_FragCoord": "position",
            "gl_FrontFacing": "front_facing",
            "gl_PointCoord": "point_coord",
            # Standard vertex semantics
            "POSITION": "position",
            "NORMAL": "normal",
            "TANGENT": "tangent",
            "BINORMAL": "binormal",
            "TEXCOORD": "texcoord",
            "TEXCOORD0": "texcoord(0)",
            "TEXCOORD1": "texcoord(1)",
            "TEXCOORD2": "texcoord(2)",
            "TEXCOORD3": "texcoord(3)",
            "COLOR": "color",
            "COLOR0": "color(0)",
            "COLOR1": "color(1)",
        }

        # Function mapping for common shader functions
        self.function_map = {
            "texture": "sample",
            "normalize": "normalize",
            "dot": "dot",
            "cross": "cross",
            "length": "length",
            "reflect": "reflect",
            "refract": "refract",
            "sin": "sin",
            "cos": "cos",
            "tan": "tan",
            "sqrt": "sqrt",
            "pow": "pow",
            "abs": "abs",
            "min": "min",
            "max": "max",
            "clamp": "clamp",
            "mix": "lerp",
            "smoothstep": "smoothstep",
            "step": "step",
            "floor": "floor",
            "ceil": "ceil",
            "fract": "fract",
            "mod": "modulo",
        }

    def generate(self, ast):
        code = "// Generated Rust GPU Shader Code\n"
        code += "use gpu::*;\n"
        code += "use math::*;\n\n"

        # Generate structs - handle both old and new AST
        structs = getattr(ast, "structs", [])
        for node in structs:
            if isinstance(node, StructNode):
                code += self.generate_struct(node)

        # Generate global variables - handle both old and new AST
        global_vars = getattr(ast, "global_variables", [])
        for node in global_vars:
            if isinstance(node, ArrayNode):
                code += self.generate_array_declaration(node)
            else:
                # Handle both old and new AST variable structures
                if hasattr(node, "var_type"):
                    var_type = self.convert_type_node_to_string(node.var_type)
                elif hasattr(node, "vtype"):
                    var_type = node.vtype
                else:
                    var_type = "float"
                code += f"static {node.name}: {self.map_type(var_type)} = Default::default();\n"

        # Generate cbuffers/constants as structs
        cbuffers = getattr(ast, "cbuffers", None) or getattr(ast, "constants", [])
        if cbuffers:
            code += "// Constant Buffers\n"
            code += self.generate_cbuffers(ast)

        # Generate functions - handle both old and new AST
        functions = getattr(ast, "functions", [])
        for func in functions:
            # Handle both old and new AST function structures
            if hasattr(func, "qualifiers") and func.qualifiers:
                qualifier = func.qualifiers[0] if func.qualifiers else None
            else:
                qualifier = getattr(func, "qualifier", None)

            if qualifier == "vertex":
                code += "// Vertex Shader\n"
                code += self.generate_function(func, shader_type="vertex")
            elif qualifier == "fragment":
                code += "// Fragment Shader\n"
                code += self.generate_function(func, shader_type="fragment")
            elif qualifier == "compute":
                code += "// Compute Shader\n"
                code += self.generate_function(func, shader_type="compute")
            else:
                code += self.generate_function(func)

        # Handle shader stages (new AST structure)
        if hasattr(ast, "stages") and ast.stages:
            for stage_type, stage in ast.stages.items():
                if hasattr(stage, "entry_point"):
                    stage_name = str(stage_type).split(".")[-1].lower()
                    code += f"// {stage_name.title()} Shader\n"
                    code += self.generate_function(
                        stage.entry_point, shader_type=stage_name
                    )
                if hasattr(stage, "local_functions"):
                    for func in stage.local_functions:
                        code += self.generate_function(func)

        return code

    def generate_struct(self, node):
        code = f"#[repr(C)]\n#[derive(Debug, Clone, Copy)]\n"
        code += f"pub struct {node.name} {{\n"

        # Generate struct members - handle both old and new AST
        members = getattr(node, "members", [])
        for member in members:
            if isinstance(member, ArrayNode):
                element_type = getattr(
                    member, "element_type", getattr(member, "vtype", "float")
                )
                if member.size:
                    code += f"    pub {member.name}: [{self.map_type_to_rust(element_type)}; {member.size}],\n"
                else:
                    code += f"    pub {member.name}: Vec<{self.map_type_to_rust(element_type)}>,\n"
            else:
                # Handle both old and new AST member structures
                if hasattr(member, "member_type"):
                    # New AST structure
                    member_type = self.convert_type_node_to_string(member.member_type)
                elif hasattr(member, "vtype"):
                    # Old AST structure
                    member_type = member.vtype
                else:
                    member_type = "float"

                # Handle semantic - get from attributes in new AST
                semantic = None
                if hasattr(member, "semantic"):
                    semantic = member.semantic
                elif hasattr(member, "attributes"):
                    semantic = self.extract_semantic_from_attributes(member.attributes)

                semantic_comment = (
                    f"  // {self.map_semantic(semantic)}" if semantic else ""
                )
                code += f"    pub {member.name}: {self.map_type(member_type)},{semantic_comment}\n"

        code += "}\n\n"
        return code

    def convert_type_node_to_string(self, type_node) -> str:
        """Convert new AST TypeNode to string representation."""
        # Handle different TypeNode types
        if hasattr(type_node, "name"):
            # PrimitiveType
            return type_node.name
        elif hasattr(type_node, "element_type") and hasattr(type_node, "size"):
            # VectorType - map to proper Rust vector types
            element_type = self.convert_type_node_to_string(type_node.element_type)
            size = type_node.size

            # Map to Rust vector types
            if element_type == "float":
                return f"vec{size}"  # This will be mapped to Vec{size}<f32> later
            elif element_type == "int":
                return f"ivec{size}"  # This will be mapped to Vec{size}<i32> later
            elif element_type == "uint":
                return f"uvec{size}"  # This will be mapped to Vec{size}<u32> later
            else:
                return f"{element_type}{size}"
        elif hasattr(type_node, "element_type") and hasattr(type_node, "rows"):
            # MatrixType
            element_type = self.convert_type_node_to_string(type_node.element_type)
            return f"mat{type_node.rows}x{type_node.cols}"  # Will be mapped later
        else:
            # Fallback
            return str(type_node)

    def extract_semantic_from_attributes(self, attributes):
        """Extract semantic information from new AST attributes."""
        semantic_attrs = [
            "position",
            "color",
            "texcoord",
            "normal",
            "tangent",
            "binormal",
            "POSITION",
            "COLOR",
            "TEXCOORD",
            "NORMAL",
            "TANGENT",
            "BINORMAL",
            "TEXCOORD0",
            "TEXCOORD1",
            "TEXCOORD2",
            "TEXCOORD3",
        ]

        for attr in attributes:
            if hasattr(attr, "name") and attr.name in semantic_attrs:
                return attr.name
        return None

    def map_type_to_rust(self, type_str):
        """Enhanced type mapping for Rust."""
        # Handle vector types first
        if type_str.startswith("float") and len(type_str) > 5:
            size = type_str[5:]
            if size.isdigit():
                return f"Vec{size}<f32>"
        elif type_str.startswith("int") and len(type_str) > 3:
            size = type_str[3:]
            if size.isdigit():
                return f"Vec{size}<i32>"

        # Standard type mapping
        type_map = {
            "void": "()",
            "bool": "bool",
            "int": "i32",
            "uint": "u32",
            "float": "f32",
            "double": "f64",
            "vec2": "Vec2<f32>",
            "vec3": "Vec3<f32>",
            "vec4": "Vec4<f32>",
            "ivec2": "Vec2<i32>",
            "ivec3": "Vec3<i32>",
            "ivec4": "Vec4<i32>",
            "uvec2": "Vec2<u32>",
            "uvec3": "Vec3<u32>",
            "uvec4": "Vec4<u32>",
            "mat2": "Mat2<f32>",
            "mat3": "Mat3<f32>",
            "mat4": "Mat4<f32>",
            "float2": "Vec2<f32>",
            "float3": "Vec3<f32>",
            "float4": "Vec4<f32>",
        }
        return type_map.get(type_str, type_str)

    def generate_cbuffers(self, ast):
        code = ""
        cbuffers = getattr(ast, "cbuffers", None) or getattr(ast, "constants", [])
        for node in cbuffers:
            if isinstance(node, StructNode):
                code += f"#[repr(C)]\n#[derive(Debug, Clone, Copy)]\n"
                code += f"pub struct {node.name} {{\n"
                for member in node.members:
                    if isinstance(member, ArrayNode):
                        if member.size:
                            code += f"    pub {member.name}: [{self.map_type(member.element_type)}; {member.size}],\n"
                        else:
                            code += f"    pub {member.name}: Vec<{self.map_type(member.element_type)}>,\n"
                    else:
                        code += (
                            f"    pub {member.name}: {self.map_type(member.vtype)},\n"
                        )
                code += "}\n\n"
            elif hasattr(node, "name") and hasattr(node, "members"):  # CbufferNode
                code += f"#[repr(C)]\n#[derive(Debug, Clone, Copy)]\n"
                code += f"pub struct {node.name} {{\n"
                for member in node.members:
                    if isinstance(member, ArrayNode):
                        if member.size:
                            code += f"    pub {member.name}: [{self.map_type(member.element_type)}; {member.size}],\n"
                        else:
                            code += f"    pub {member.name}: Vec<{self.map_type(member.element_type)}>,\n"
                    else:
                        code += (
                            f"    pub {member.name}: {self.map_type(member.vtype)},\n"
                        )
                code += "}\n\n"
        return code

    def generate_function(self, func, indent=0, shader_type=None):
        code = ""
        code += "  " * indent

        # Handle parameters - support both old and new AST
        param_list = getattr(func, "parameters", getattr(func, "params", []))
        params = []
        for p in param_list:
            if hasattr(p, "param_type"):
                # New AST structure
                param_type = self.convert_type_node_to_string(p.param_type)
            elif hasattr(p, "vtype"):
                # Old AST structure
                param_type = p.vtype
            else:
                param_type = "float"

            params.append(f"{p.name}: {self.map_type(param_type)}")

        params_str = ", ".join(params) if params else ""

        # Handle return type - support both old and new AST
        if hasattr(func, "return_type"):
            return_type = self.convert_type_node_to_string(func.return_type)
        else:
            return_type = "void"

        # Add shader type decorators
        if shader_type == "vertex":
            code += f"#[vertex_shader]\n"
        elif shader_type == "fragment":
            code += f"#[fragment_shader]\n"
        elif shader_type == "compute":
            code += f"#[compute_shader]\n"

        code += f"pub fn {func.name}({params_str}) -> {self.map_type(return_type)} {{\n"

        # Handle function body - support both old and new AST
        body = getattr(func, "body", [])
        if hasattr(body, "statements"):
            # New AST BlockNode structure
            for stmt in body.statements:
                code += self.generate_statement(stmt, indent + 1)
        elif isinstance(body, list):
            # Old AST structure
            for stmt in body:
                code += self.generate_statement(stmt, indent + 1)

        code += "  " * indent + "}\n\n"
        return code

    def generate_param_attributes(self, param):
        """Generate Rust GPU parameter attributes based on semantic"""
        if not param.semantic:
            return ""

        semantic = param.semantic.lower()
        if "position" in semantic:
            return "#[location(0)] "
        elif "normal" in semantic:
            return "#[location(1)] "
        elif "texcoord" in semantic:
            if "texcoord0" in semantic:
                return "#[location(2)] "
            elif "texcoord1" in semantic:
                return "#[location(3)] "
            else:
                return "#[location(2)] "
        elif "color" in semantic:
            return "#[location(4)] "
        elif "gl_position" in semantic:
            return "#[builtin(position)] "
        elif "gl_fragcolor" in semantic:
            return "#[location(0)] "
        return ""

    def generate_statement(self, stmt, indent=0):
        indent_str = "    " * indent

        if isinstance(stmt, VariableNode):
            # Handle both old and new AST variable structures
            if hasattr(stmt, "var_type"):
                vtype = stmt.var_type
            elif hasattr(stmt, "vtype"):
                vtype = stmt.vtype
            else:
                vtype = "f32"

            # Handle initialization
            if hasattr(stmt, "initial_value") and stmt.initial_value is not None:
                init_expr = self.generate_expression(stmt.initial_value)
                return f"{indent_str}let mut {stmt.name}: {self.map_type(vtype)} = {init_expr};\n"
            else:
                return f"{indent_str}let mut {stmt.name}: {self.map_type(vtype)};\n"

        elif isinstance(stmt, ArrayNode):
            return self.generate_array_declaration(stmt, indent)

        elif isinstance(stmt, AssignmentNode):
            return f"{indent_str}{self.generate_assignment(stmt)};\n"

        elif isinstance(stmt, IfNode):
            return self.generate_if(stmt, indent)

        elif isinstance(stmt, ForNode):
            return self.generate_for(stmt, indent)

        elif isinstance(stmt, ReturnNode):
            if hasattr(stmt, "value") and stmt.value is not None:
                # Handle both single values and lists
                if isinstance(stmt.value, list):
                    # Multiple return values (tuple)
                    values = ", ".join(
                        self.generate_expression(val) for val in stmt.value
                    )
                    return f"{indent_str}return ({values});\n"
                else:
                    # Single return value
                    return (
                        f"{indent_str}return {self.generate_expression(stmt.value)};\n"
                    )
            else:
                # Void return
                return f"{indent_str}return;\n"

        elif hasattr(stmt, "__class__") and "ExpressionStatement" in str(
            stmt.__class__
        ):
            # Handle ExpressionStatementNode
            if hasattr(stmt, "expression"):
                return f"{indent_str}{self.generate_expression(stmt.expression)};\n"
            else:
                return f"{indent_str}{self.generate_expression(stmt)};\n"

        elif isinstance(stmt, ArrayAccessNode):
            # ArrayAccessNode as statement - likely misclassified
            return f"{indent_str}// Unhandled ArrayAccessNode: {stmt}\n"

        else:
            # Try to generate as expression
            expr_result = self.generate_expression(stmt)
            if expr_result and expr_result.strip():
                return f"{indent_str}{expr_result};\n"
            else:
                return f"{indent_str}// Unhandled statement: {type(stmt).__name__}\n"

    def generate_array_declaration(self, node, indent=0):
        indent_str = "    " * indent
        element_type = self.map_type(node.element_type)
        size = get_array_size_from_node(node)

        if size is None:
            return f"{indent_str}let {node.name}: Vec<{element_type}> = Vec::new();\n"
        else:
            return f"{indent_str}let {node.name}: [{element_type}; {size}] = [Default::default(); {size}];\n"

    def generate_assignment(self, node):
        # Handle both old and new AST assignment structures
        if hasattr(node, "target") and hasattr(node, "value"):
            # New AST structure
            lhs = self.generate_expression(node.target)
            rhs = self.generate_expression(node.value)
            op = getattr(node, "operator", "=")
        else:
            # Old AST structure
            lhs = self.generate_expression(node.left)
            rhs = self.generate_expression(node.right)
            op = getattr(node, "operator", "=")
        return f"{lhs} {op} {rhs}"

    def generate_if(self, node, indent):
        indent_str = "    " * indent
        condition = self.generate_expression(
            node.condition if hasattr(node, "condition") else node.if_condition
        )
        code = f"{indent_str}if {condition} {{\n"

        # Generate if body - handle BlockNode structure
        if_body = getattr(node, "then_branch", getattr(node, "if_body", None))
        if hasattr(if_body, "statements"):
            # New AST BlockNode structure
            for stmt in if_body.statements:
                code += self.generate_statement(stmt, indent + 1)
        elif isinstance(if_body, list):
            # Old AST structure - list of statements
            for stmt in if_body:
                code += self.generate_statement(stmt, indent + 1)

        code += f"{indent_str}}}"

        # Handle else branch - check if it's another if statement (else-if chain)
        else_branch = getattr(node, "else_branch", None)
        if else_branch:
            # Check if else branch is another IfNode (else-if chain)
            if hasattr(else_branch, "__class__") and "If" in str(else_branch.__class__):
                # Generate else if by recursively generating the nested if with else if prefix
                elif_condition = self.generate_expression(
                    else_branch.condition
                    if hasattr(else_branch, "condition")
                    else else_branch.if_condition
                )
                code += f" else if {elif_condition} {{\n"

                # Generate elif body
                elif_body = getattr(
                    else_branch, "then_branch", getattr(else_branch, "if_body", None)
                )
                if hasattr(elif_body, "statements"):
                    for stmt in elif_body.statements:
                        code += self.generate_statement(stmt, indent + 1)
                elif isinstance(elif_body, list):
                    for stmt in elif_body:
                        code += self.generate_statement(stmt, indent + 1)

                code += f"{indent_str}}}"

                # Recursively handle any remaining else-if chain
                nested_else = getattr(else_branch, "else_branch", None)
                if nested_else:
                    if hasattr(nested_else, "__class__") and "If" in str(
                        nested_else.__class__
                    ):
                        # Another else if - recursively handle
                        remaining_code = self.generate_if(nested_else, indent)
                        # Remove the "if" prefix and replace with "else if"
                        remaining_lines = remaining_code.split("\n")
                        if remaining_lines[0].strip().startswith("if "):
                            remaining_lines[0] = remaining_lines[0].replace(
                                "if ", " else if ", 1
                            )
                        code += "\n".join(
                            remaining_lines[1:]
                        )  # Skip first line as we already handled it
                    else:
                        # Final else clause
                        code += f" else {{\n"
                        if hasattr(nested_else, "statements"):
                            for stmt in nested_else.statements:
                                code += self.generate_statement(stmt, indent + 1)
                        elif isinstance(nested_else, list):
                            for stmt in nested_else:
                                code += self.generate_statement(stmt, indent + 1)
                        else:
                            code += self.generate_statement(nested_else, indent + 1)
                        code += f"{indent_str}}}"
            else:
                # Regular else clause
                code += f" else {{\n"
                if hasattr(else_branch, "statements"):
                    # New AST BlockNode structure
                    for stmt in else_branch.statements:
                        code += self.generate_statement(stmt, indent + 1)
                elif isinstance(else_branch, list):
                    # Old AST structure
                    for stmt in else_branch:
                        code += self.generate_statement(stmt, indent + 1)
                else:
                    # Single statement
                    code += self.generate_statement(else_branch, indent + 1)
                code += f"{indent_str}}}"

        code += "\n"
        return code

    def generate_for(self, node, indent):
        indent_str = "    " * indent

        # Extract init, condition, and update
        init = self.generate_statement(node.init, 0).strip()
        condition = self.generate_expression(node.condition)
        update = self.generate_expression(node.update)

        # In Rust, we'll convert C-style for loops to while loops
        code = f"{indent_str}{init};\n"
        code += f"{indent_str}while {condition} {{\n"

        # Generate loop body
        # Handle BlockNode structure
        if hasattr(node.body, "statements"):
            # New AST BlockNode structure
            for stmt in node.body.statements:
                code += self.generate_statement(stmt, indent + 1)
        elif isinstance(node.body, list):
            # Old AST list structure
            for stmt in node.body:
                code += self.generate_statement(stmt, indent + 1)
        else:
            # Single statement
            code += self.generate_statement(node.body, indent + 1)

        # Add update at the end of the loop
        code += f"{indent_str}    {update};\n"
        code += f"{indent_str}}}\n"

        return code

    def generate_expression(self, expr):
        if expr is None:
            return ""
        elif isinstance(expr, str):
            return expr
        elif isinstance(expr, (int, float, bool)):
            if isinstance(expr, bool):
                return "true" if expr else "false"
            return str(expr)
        elif hasattr(expr, "__class__") and "Literal" in str(expr.__class__):
            # Handle LiteralNode
            if hasattr(expr, "value"):
                value = expr.value
                if isinstance(value, str) and not (
                    value.startswith('"') and value.endswith('"')
                ):
                    return f'"{value}"'  # Add quotes for string literals
                return str(value)
            return str(expr)
        elif hasattr(expr, "__class__") and "Identifier" in str(expr.__class__):
            # Handle IdentifierNode
            return getattr(expr, "name", str(expr))
        elif isinstance(expr, VariableNode):
            if hasattr(expr, "name"):
                return expr.name
            else:
                return str(expr)
        elif hasattr(expr, "__class__") and "BinaryOp" in str(expr.__class__):
            # Handle BinaryOpNode
            left = self.generate_expression(getattr(expr, "left", ""))
            right = self.generate_expression(getattr(expr, "right", ""))
            op = getattr(expr, "operator", getattr(expr, "op", "+"))
            return f"({left} {self.map_operator(op)} {right})"
        elif isinstance(expr, AssignmentNode):
            return self.generate_assignment(expr)
        elif hasattr(expr, "__class__") and "UnaryOp" in str(expr.__class__):
            # Handle UnaryOpNode
            operand = self.generate_expression(getattr(expr, "operand", ""))
            op = getattr(expr, "operator", getattr(expr, "op", "+"))
            return f"({self.map_operator(op)}{operand})"
        elif hasattr(expr, "__class__") and "ArrayAccess" in str(expr.__class__):
            # Handle ArrayAccessNode
            array_expr = getattr(expr, "array_expr", getattr(expr, "array", ""))
            index_expr = getattr(expr, "index_expr", getattr(expr, "index", ""))
            array = self.generate_expression(array_expr)
            index = self.generate_expression(index_expr)
            return f"{array}[{index}]"
        elif hasattr(expr, "__class__") and "FunctionCall" in str(expr.__class__):
            # Handle FunctionCallNode
            func_name = getattr(expr, "function", getattr(expr, "name", "unknown"))
            if hasattr(func_name, "name"):
                func_name = func_name.name
            args = getattr(expr, "arguments", getattr(expr, "args", []))

            # Map function names to Rust equivalents
            func_name = self.function_map.get(func_name, func_name)

            # Handle vector constructors
            if func_name in [
                "vec2",
                "vec3",
                "vec4",
                "ivec2",
                "ivec3",
                "ivec4",
                "uvec2",
                "uvec3",
                "uvec4",
                "bvec2",
                "bvec3",
                "bvec4",
            ]:
                rust_type = self.map_type(func_name)
                args_str = ", ".join(self.generate_expression(arg) for arg in args)
                return f"{rust_type}::new({args_str})"

            # Handle matrix constructors
            if func_name in ["mat2", "mat3", "mat4"]:
                rust_type = self.map_type(func_name)
                args_str = ", ".join(self.generate_expression(arg) for arg in args)
                return f"{rust_type}::new({args_str})"

            # Handle standard function calls
            args_str = ", ".join(self.generate_expression(arg) for arg in args)
            return f"{func_name}({args_str})"
        elif hasattr(expr, "__class__") and "MemberAccess" in str(expr.__class__):
            # Handle MemberAccessNode
            obj_expr = getattr(expr, "object_expr", getattr(expr, "object", ""))
            member = getattr(expr, "member", "")
            obj = self.generate_expression(obj_expr)
            return f"{obj}.{member}"
        elif hasattr(expr, "__class__") and "TernaryOp" in str(expr.__class__):
            # Handle TernaryOpNode
            condition = self.generate_expression(getattr(expr, "condition", ""))
            true_expr = self.generate_expression(getattr(expr, "true_expr", ""))
            false_expr = self.generate_expression(getattr(expr, "false_expr", ""))
            return f"(if {condition} {{ {true_expr} }} else {{ {false_expr} }})"
        else:
            # Fallback - return string representation
            return str(expr)

    def map_type(self, vtype):
        if vtype is None:
            return "f32"

        # Handle TypeNode objects by converting to string first
        if hasattr(vtype, "name") or hasattr(vtype, "element_type"):
            vtype_str = self.convert_type_node_to_string(vtype)
        else:
            vtype_str = str(vtype)

        # Handle array types
        if "[" in vtype_str and "]" in vtype_str:
            base_type, size = parse_array_type(vtype_str)
            base_mapped = self.type_mapping.get(base_type, base_type)
            if size:
                return f"[{base_mapped}; {size}]"
            else:
                return f"Vec<{base_mapped}>"

        # Use regular type mapping
        return self.type_mapping.get(vtype_str, vtype_str)

    def map_operator(self, op):
        op_map = {
            "PLUS": "+",
            "MINUS": "-",
            "MULTIPLY": "*",
            "DIVIDE": "/",
            "BITWISE_XOR": "^",
            "BITWISE_OR": "|",
            "BITWISE_AND": "&",
            "LESS_THAN": "<",
            "GREATER_THAN": ">",
            "ASSIGN_ADD": "+=",
            "ASSIGN_SUB": "-=",
            "ASSIGN_MUL": "*=",
            "ASSIGN_DIV": "/=",
            "ASSIGN_MOD": "%=",
            "ASSIGN_XOR": "^=",
            "ASSIGN_OR": "|=",
            "ASSIGN_AND": "&=",
            "LESS_EQUAL": "<=",
            "GREATER_EQUAL": ">=",
            "EQUAL": "==",
            "NOT_EQUAL": "!=",
            "AND": "&&",
            "OR": "||",
            "EQUALS": "=",
            "ASSIGN_SHIFT_LEFT": "<<=",
            "ASSIGN_SHIFT_RIGHT": ">>=",
            "LOGICAL_AND": "&&",
            "LOGICAL_OR": "||",
            "BITWISE_SHIFT_RIGHT": ">>",
            "BITWISE_SHIFT_LEFT": "<<",
            "MOD": "%",
            "NOT": "!",
        }
        return op_map.get(op, op)

    def map_semantic(self, semantic):
        if semantic:
            return self.semantic_map.get(semantic, semantic)
        return ""

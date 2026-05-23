"""CrossGL-to-Rust code generator."""

from ..ast import (
    ArrayNode,
    ArrayAccessNode,
    ArrayLiteralNode,
    AssignmentNode,
    BinaryOpNode,
    BreakNode,
    CaseNode,
    CbufferNode,
    ContinueNode,
    DoWhileNode,
    ForInNode,
    ForNode,
    FunctionCallNode,
    FunctionNode,
    IdentifierNode,
    IfNode,
    LiteralNode,
    LiteralPatternNode,
    LoopNode,
    MatchNode,
    MemberAccessNode,
    ReturnNode,
    RangeNode,
    ShaderNode,
    StructNode,
    SwitchNode,
    TernaryOpNode,
    UnaryOpNode,
    VariableNode,
    WhileNode,
    WildcardPatternNode,
)
from .array_utils import parse_array_type, format_array_type, get_array_size_from_node


class RustCodeGen:
    """Emit Rust-like GPU shader source from the shared CrossGL AST."""

    def __init__(self):
        """Initialize Rust type maps and expression-generation state."""
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
            "string": "&'static str",
            "char": "char",
            # Vector Types (using GPU-style vector types)
            "vec2<f32>": "Vec2<f32>",
            "vec3<f32>": "Vec3<f32>",
            "vec4<f32>": "Vec4<f32>",
            "vec2<f64>": "Vec2<f64>",
            "vec3<f64>": "Vec3<f64>",
            "vec4<f64>": "Vec4<f64>",
            "vec2<i32>": "Vec2<i32>",
            "vec3<i32>": "Vec3<i32>",
            "vec4<i32>": "Vec4<i32>",
            "vec2<u32>": "Vec2<u32>",
            "vec3<u32>": "Vec3<u32>",
            "vec4<u32>": "Vec4<u32>",
            "vec2<bool>": "Vec2<bool>",
            "vec3<bool>": "Vec3<bool>",
            "vec4<bool>": "Vec4<bool>",
            "vec2": "Vec2<f32>",
            "vec3": "Vec3<f32>",
            "vec4": "Vec4<f32>",
            "ivec2": "Vec2<i32>",
            "ivec3": "Vec3<i32>",
            "ivec4": "Vec4<i32>",
            "uvec2": "Vec2<u32>",
            "uvec3": "Vec3<u32>",
            "uvec4": "Vec4<u32>",
            "dvec2": "Vec2<f64>",
            "dvec3": "Vec3<f64>",
            "dvec4": "Vec4<f64>",
            "bvec2": "Vec2<bool>",
            "bvec3": "Vec3<bool>",
            "bvec4": "Vec4<bool>",
            "bool2": "Vec2<bool>",
            "bool3": "Vec3<bool>",
            "bool4": "Vec4<bool>",
            # Matrix Types
            "mat2": "Mat2<f32>",
            "mat3": "Mat3<f32>",
            "mat4": "Mat4<f32>",
            "mat2x2": "Mat2<f32>",
            "mat2x3": "Mat2x3<f32>",
            "mat2x4": "Mat2x4<f32>",
            "mat3x2": "Mat3x2<f32>",
            "mat3x3": "Mat3<f32>",
            "mat3x4": "Mat3x4<f32>",
            "mat4x2": "Mat4x2<f32>",
            "mat4x3": "Mat4x3<f32>",
            "mat4x4": "Mat4<f32>",
            "dmat2": "Mat2<f64>",
            "dmat3": "Mat3<f64>",
            "dmat4": "Mat4<f64>",
            "dmat2x2": "Mat2<f64>",
            "dmat2x3": "Mat2x3<f64>",
            "dmat2x4": "Mat2x4<f64>",
            "dmat3x2": "Mat3x2<f64>",
            "dmat3x3": "Mat3<f64>",
            "dmat3x4": "Mat3x4<f64>",
            "dmat4x2": "Mat4x2<f64>",
            "dmat4x3": "Mat4x3<f64>",
            "dmat4x4": "Mat4<f64>",
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
            "inversesqrt": "rsqrt",
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
            "frac": "fract",
            "fract": "fract",
            "mod": "modulo",
        }
        self.variable_types = {}
        self.struct_member_types = {}
        self.current_return_type = None
        self.do_while_contexts = []
        self.for_contexts = []
        self.loop_depth = 0
        self.do_while_counter = 0

    def generate(self, ast):
        """Generate complete Rust-like shader source for a CrossGL AST."""
        self.variable_types = {}
        self.struct_member_types = {}
        self.current_return_type = None
        self.do_while_contexts = []
        self.for_contexts = []
        self.loop_depth = 0
        self.do_while_counter = 0
        code = "// Generated Rust GPU Shader Code\n"
        code += "use gpu::*;\n"
        code += "use math::*;\n\n"

        structs = getattr(ast, "structs", [])
        for node in structs:
            if isinstance(node, StructNode):
                code += self.generate_struct(node)

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
                self.register_variable_type(node.name, var_type)
                initial_value = getattr(
                    node, "initial_value", getattr(node, "value", None)
                )
                if initial_value is not None:
                    init_expr = self.generate_expression_with_type(
                        initial_value, var_type
                    )
                else:
                    init_expr = "Default::default()"
                code += (
                    f"static {node.name}: {self.map_type(var_type)} = "
                    f"{init_expr};\n"
                )

        cbuffers = self.get_cbuffer_nodes(ast)
        if cbuffers:
            code += "// Constant Buffers\n"
            code += self.generate_cbuffers(ast)

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
        code = f"#[repr(C)]\n#[derive(Debug, Clone, Copy, Default)]\n"
        code += f"pub struct {node.name} {{\n"
        member_types = {}

        members = getattr(node, "members", [])
        for member in members:
            if isinstance(member, ArrayNode):
                element_type = getattr(
                    member, "element_type", getattr(member, "vtype", "float")
                )
                member_types[member.name] = (
                    f"{self.convert_type_node_to_string(element_type)}[{member.size}]"
                    if member.size
                    else f"{self.convert_type_node_to_string(element_type)}[]"
                )
                if member.size:
                    code += f"    pub {member.name}: [{self.map_type_to_rust(element_type)}; {member.size}],\n"
                else:
                    code += f"    pub {member.name}: Vec<{self.map_type_to_rust(element_type)}>,\n"
            else:
                if hasattr(member, "member_type"):
                    member_type = self.convert_type_node_to_string(member.member_type)
                elif hasattr(member, "vtype"):
                    member_type = member.vtype
                else:
                    member_type = "float"
                member_types[member.name] = member_type

                semantic = None
                if hasattr(member, "semantic"):
                    semantic = member.semantic
                elif hasattr(member, "attributes"):
                    semantic = self.extract_semantic_from_attributes(member.attributes)

                semantic_comment = (
                    f"  // {self.map_semantic(semantic)}" if semantic else ""
                )
                code += f"    pub {member.name}: {self.map_type(member_type)},{semantic_comment}\n"

        self.struct_member_types[node.name] = member_types
        code += "}\n\n"
        return code

    def convert_type_node_to_string(self, type_node) -> str:
        """Convert new AST TypeNode to string representation."""
        if type_node.__class__.__name__ == "ArrayType":
            element_type = self.convert_type_node_to_string(type_node.element_type)
            size = self.format_array_size(type_node.size)
            return (
                f"{element_type}[{size}]" if size is not None else f"{element_type}[]"
            )
        if hasattr(type_node, "name"):
            generic_args = getattr(type_node, "generic_args", [])
            if generic_args:
                args = ", ".join(
                    self.convert_type_node_to_string(arg) for arg in generic_args
                )
                return f"{type_node.name}<{args}>"
            return type_node.name
        elif hasattr(type_node, "element_type") and hasattr(type_node, "size"):
            element_type = self.convert_type_node_to_string(type_node.element_type)
            size = type_node.size
            if element_type == "float":
                return f"vec{size}"
            elif element_type == "int":
                return f"ivec{size}"
            elif element_type == "uint":
                return f"uvec{size}"
            elif element_type == "double":
                return f"dvec{size}"
            elif element_type == "bool":
                return f"bvec{size}"
            else:
                return f"{element_type}{size}"
        elif hasattr(type_node, "element_type") and hasattr(type_node, "rows"):
            element_type = self.convert_type_node_to_string(type_node.element_type)
            prefix = "dmat" if element_type == "double" else "mat"
            if type_node.rows == type_node.cols:
                return f"{prefix}{type_node.rows}"
            return f"{prefix}{type_node.rows}x{type_node.cols}"
        else:
            return str(type_node)

    def format_array_size(self, size):
        if size is None:
            return None
        if hasattr(size, "value"):
            return size.value
        return size

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

    def get_member_type(self, member):
        if hasattr(member, "member_type"):
            return self.convert_type_node_to_string(member.member_type)
        if hasattr(member, "vtype"):
            return member.vtype
        return "float"

    def get_cbuffer_nodes(self, ast):
        nodes = []
        seen = set()
        for attr in ("cbuffers", "constants"):
            for node in getattr(ast, attr, None) or []:
                node_id = id(node)
                if node_id not in seen:
                    nodes.append(node)
                    seen.add(node_id)
        return nodes

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
        cbuffers = self.get_cbuffer_nodes(ast)
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
                        code += f"    pub {member.name}: {self.map_type(self.get_member_type(member))},\n"
                code += "}\n\n"
                code += self.generate_cbuffer_member_statics(node.members)
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
                        code += f"    pub {member.name}: {self.map_type(self.get_member_type(member))},\n"
                code += "}\n\n"
                code += self.generate_cbuffer_member_statics(node.members)
        return code

    def generate_cbuffer_member_statics(self, members):
        code = ""
        for member in members:
            if isinstance(member, ArrayNode):
                if member.size:
                    member_type = (
                        f"[{self.map_type(member.element_type)}; {member.size}]"
                    )
                else:
                    member_type = f"Vec<{self.map_type(member.element_type)}>"
            else:
                member_type = self.map_type(self.get_member_type(member))
            code += f"static {member.name}: {member_type} = Default::default();\n"
        return code

    def generate_function(self, func, indent=0, shader_type=None):
        """Render one CrossGL function or shader entry point as Rust code."""
        code = ""
        code += "  " * indent
        saved_variable_types = self.variable_types.copy()
        saved_return_type = self.current_return_type

        param_list = getattr(func, "parameters", getattr(func, "params", []))
        params = []
        for p in param_list:
            if hasattr(p, "param_type"):
                param_type = self.convert_type_node_to_string(p.param_type)
            elif hasattr(p, "vtype"):
                param_type = p.vtype
            else:
                param_type = "float"

            self.register_variable_type(p.name, param_type)
            params.append(f"{p.name}: {self.map_type(param_type)}")

        params_str = ", ".join(params) if params else ""

        if hasattr(func, "return_type"):
            return_type = self.convert_type_node_to_string(func.return_type)
        else:
            return_type = "void"
        self.current_return_type = return_type

        if shader_type == "vertex":
            code += f"#[vertex_shader]\n"
        elif shader_type == "fragment":
            code += f"#[fragment_shader]\n"
        elif shader_type == "compute":
            code += f"#[compute_shader]\n"

        code += f"pub fn {func.name}({params_str}) -> {self.map_type(return_type)} {{\n"

        body = getattr(func, "body", [])
        if hasattr(body, "statements"):
            for stmt in body.statements:
                code += self.generate_statement(stmt, indent + 1)
        elif isinstance(body, list):
            for stmt in body:
                code += self.generate_statement(stmt, indent + 1)

        code += "  " * indent + "}\n\n"
        self.variable_types = saved_variable_types
        self.current_return_type = saved_return_type
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
        """Render a single CrossGL statement as Rust code."""
        indent_str = "    " * indent

        if isinstance(stmt, VariableNode):
            initial_value = getattr(stmt, "initial_value", None)
            vtype = self.variable_declaration_type(stmt, initial_value)
            self.register_variable_type(stmt.name, vtype)
            if initial_value is not None:
                increment_init = self.generate_increment_initializer_declaration(
                    stmt,
                    initial_value,
                    vtype,
                    indent,
                )
                if increment_init is not None:
                    return increment_init
                init_expr = self.generate_expression_with_type(initial_value, vtype)
                return f"{indent_str}let mut {stmt.name}: {self.map_type(vtype)} = {init_expr};\n"
            elif self.is_generated_struct_type(vtype):
                return f"{indent_str}let mut {stmt.name}: {self.map_type(vtype)} = Default::default();\n"
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

        elif isinstance(stmt, ForInNode):
            return self.generate_for_in(stmt, indent)

        elif isinstance(stmt, WhileNode):
            return self.generate_while(stmt, indent)

        elif isinstance(stmt, LoopNode):
            return self.generate_loop(stmt, indent)

        elif isinstance(stmt, DoWhileNode):
            return self.generate_do_while(stmt, indent)

        elif isinstance(stmt, MatchNode):
            return self.generate_match(stmt, indent)

        elif isinstance(stmt, SwitchNode):
            return self.generate_switch(stmt, indent)

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
                    if isinstance(stmt.value, ArrayLiteralNode):
                        return_expr = self.generate_expression_with_type(
                            stmt.value, self.current_return_type
                        )
                        return f"{indent_str}return {return_expr};\n"
                    return (
                        f"{indent_str}return {self.generate_expression(stmt.value)};\n"
                    )
            else:
                # Void return
                return f"{indent_str}return;\n"

        elif isinstance(stmt, BreakNode):
            context = self.active_do_while_context()
            if context:
                break_flag = context["break_flag"]
                return f"{indent_str}{break_flag} = true;\n{indent_str}break;\n"
            return f"{indent_str}break;\n"

        elif isinstance(stmt, ContinueNode):
            if self.active_do_while_context():
                return f"{indent_str}break;\n"
            context = self.active_for_context()
            if context:
                update = context["update"]
                return f"{indent_str}{update};\n{indent_str}continue;\n"
            return f"{indent_str}continue;\n"

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

    def generate_increment_initializer_declaration(
        self,
        stmt,
        initial_value,
        vtype,
        indent,
    ):
        if not isinstance(initial_value, UnaryOpNode):
            return None

        op = getattr(initial_value, "operator", getattr(initial_value, "op", ""))
        op = self.map_operator(op)
        if op not in {"++", "--"}:
            return None

        operand = self.generate_expression(getattr(initial_value, "operand", ""))
        assignment_op = "+=" if op == "++" else "-="
        update = f"{'    ' * indent}{operand} {assignment_op} 1;\n"
        declaration = (
            f"{'    ' * indent}let mut {stmt.name}: "
            f"{self.map_type(vtype)} = {operand};\n"
        )
        is_postfix = getattr(
            initial_value,
            "is_postfix",
            getattr(initial_value, "postfix", False),
        )
        if is_postfix:
            return declaration + update
        return update + declaration

    def generate_switch(self, node, indent):
        indent_str = "    " * indent
        arm_indent = "    " * (indent + 1)
        expression = self.generate_expression(getattr(node, "expression", ""))

        code = f"{indent_str}match {expression} {{\n"
        has_default = False
        for case in getattr(node, "cases", []) or []:
            if not isinstance(case, CaseNode):
                continue
            value = getattr(case, "value", None)
            pattern = "_" if value is None else self.generate_expression(value)
            has_default = has_default or value is None
            code += f"{arm_indent}{pattern} => {{\n"
            code += self.generate_switch_case_body(
                getattr(case, "statements", []), indent + 2
            )
            code += f"{arm_indent}}},\n"

        default_case = getattr(node, "default_case", None)
        if default_case is not None:
            has_default = True
            code += f"{arm_indent}_ => {{\n"
            code += self.generate_switch_case_body(default_case, indent + 2)
            code += f"{arm_indent}}},\n"
        elif not has_default:
            code += f"{arm_indent}_ => {{}},\n"

        code += f"{indent_str}}}\n"
        return code

    def generate_match(self, node, indent):
        indent_str = "    " * indent
        arm_indent = "    " * (indent + 1)
        expression = self.generate_expression(getattr(node, "expression", ""))

        code = f"{indent_str}match {expression} {{\n"
        has_wildcard = False
        for arm in getattr(node, "arms", []) or []:
            if not self.is_supported_match_arm(arm):
                raise ValueError(
                    "Unsupported match arm for Rust codegen; only unguarded "
                    "literal and wildcard patterns are supported"
                )

            pattern = getattr(arm, "pattern", None)
            if isinstance(pattern, WildcardPatternNode):
                arm_pattern = "_"
                has_wildcard = True
            else:
                arm_pattern = self.generate_expression(pattern.literal)

            code += f"{arm_indent}{arm_pattern} => {{\n"
            code += self.generate_switch_case_body(getattr(arm, "body", []), indent + 2)
            code += f"{arm_indent}}},\n"

        if not has_wildcard:
            code += f"{arm_indent}_ => {{}},\n"

        code += f"{indent_str}}}\n"
        return code

    def is_supported_match_arm(self, arm):
        if getattr(arm, "guard", None) is not None:
            return False
        pattern = getattr(arm, "pattern", None)
        return isinstance(pattern, (LiteralPatternNode, WildcardPatternNode))

    def generate_switch_case_body(self, body, indent):
        statements = self.statement_list(body)
        code = ""
        for stmt in statements:
            if isinstance(stmt, BreakNode):
                continue
            code += self.generate_statement(stmt, indent)
        return code

    def statement_list(self, body):
        if hasattr(body, "statements"):
            return body.statements
        if isinstance(body, list):
            return body
        if body is None:
            return []
        return [body]

    def active_do_while_context(self):
        if not self.do_while_contexts:
            return None
        context = self.do_while_contexts[-1]
        if context["loop_depth"] == self.loop_depth:
            return context
        return None

    def active_for_context(self):
        if not self.for_contexts:
            return None
        context = self.for_contexts[-1]
        if context["loop_depth"] == self.loop_depth:
            return context
        return None

    def get_variable_type(self, node):
        if hasattr(node, "var_type"):
            vtype = node.var_type
        elif hasattr(node, "vtype"):
            vtype = node.vtype
        else:
            return None

        if vtype is None:
            return None
        if isinstance(vtype, str) and vtype.strip() in {"", "None"}:
            return None
        return vtype

    def variable_declaration_type(self, node, initial_value=None):
        declared_type = self.get_variable_type(node)
        if declared_type is not None:
            return declared_type

        inferred_type = self.expression_result_type(initial_value)
        if inferred_type is not None:
            return inferred_type
        return "float"

    def statement_body_terminates_inner_loop(self, body):
        statements = self.statement_list(body)
        if not statements:
            return False
        return isinstance(statements[-1], (BreakNode, ContinueNode, ReturnNode))

    def generate_array_declaration(self, node, indent=0):
        indent_str = "    " * indent
        element_type = self.map_type(node.element_type)
        size = get_array_size_from_node(node)

        if size is None:
            return f"{indent_str}let {node.name}: Vec<{element_type}> = Vec::new();\n"
        else:
            return f"{indent_str}let {node.name}: [{element_type}; {size}] = [Default::default(); {size}];\n"

    def generate_expression_with_type(self, expr, target_type):
        if isinstance(expr, ArrayLiteralNode):
            return self.generate_array_literal_expression(expr, target_type)
        return self.generate_expression(expr)

    def is_array_type_name(self, type_name):
        return type_name is not None and "[" in str(type_name) and "]" in str(type_name)

    def generate_array_literal_expression(self, expr, target_type=None):
        elements = [self.generate_expression(element) for element in expr.elements]

        if self.is_array_type_name(target_type):
            _, size = parse_array_type(str(target_type))
            if size is None:
                return f"vec![{', '.join(elements)}]"

            elements = elements[:size]
            while len(elements) < size:
                elements.append("Default::default()")

        return f"[{', '.join(elements)}]"

    def generate_assignment(self, node):
        # Handle both old and new AST assignment structures
        if hasattr(node, "target") and hasattr(node, "value"):
            # New AST structure
            lhs = self.generate_expression(node.target)
            lhs_type = self.expression_result_type(node.target)
            rhs = self.generate_expression_with_type(node.value, lhs_type)
            op = getattr(node, "operator", "=")
        else:
            # Old AST structure
            lhs = self.generate_expression(node.left)
            lhs_type = self.expression_result_type(node.left)
            rhs = self.generate_expression_with_type(node.right, lhs_type)
            op = getattr(node, "operator", "=")
        return f"{lhs} {op} {rhs}"

    def generate_if(self, node, indent):
        indent_str = "    " * indent
        condition = self.generate_expression(
            node.condition if hasattr(node, "condition") else node.if_condition
        )
        code = f"{indent_str}if {condition} {{\n"

        if_body = getattr(node, "then_branch", getattr(node, "if_body", None))
        if hasattr(if_body, "statements"):
            for stmt in if_body.statements:
                code += self.generate_statement(stmt, indent + 1)
        elif isinstance(if_body, list):
            for stmt in if_body:
                code += self.generate_statement(stmt, indent + 1)

        code += f"{indent_str}}}"

        else_branch = getattr(node, "else_branch", None)
        if else_branch:
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
                code += f" else {{\n"
                if hasattr(else_branch, "statements"):
                    for stmt in else_branch.statements:
                        code += self.generate_statement(stmt, indent + 1)
                elif isinstance(else_branch, list):
                    for stmt in else_branch:
                        code += self.generate_statement(stmt, indent + 1)
                else:
                    code += self.generate_statement(else_branch, indent + 1)
                code += f"{indent_str}}}"

        code += "\n"
        return code

    def generate_for(self, node, indent):
        indent_str = "    " * indent

        init = self.generate_statement(node.init, 0).strip()
        if init.endswith(";"):
            init = init[:-1]
        condition = self.generate_expression(node.condition)
        update = self.generate_expression(node.update)

        code = f"{indent_str}{init};\n"
        code += f"{indent_str}while {condition} {{\n"

        self.loop_depth += 1
        self.for_contexts.append({"loop_depth": self.loop_depth, "update": update})
        try:
            for stmt in self.statement_list(node.body):
                code += self.generate_statement(stmt, indent + 1)
        finally:
            self.for_contexts.pop()
            self.loop_depth -= 1

        # Add update at the end of the loop
        code += f"{indent_str}    {update};\n"
        code += f"{indent_str}}}\n"

        return code

    def generate_for_in(self, node, indent):
        indent_str = "    " * indent
        pattern = getattr(node, "pattern", "item")
        iterable = self.generate_for_in_iterable(getattr(node, "iterable", None))

        code = f"{indent_str}for {pattern} in {iterable} {{\n"

        self.loop_depth += 1
        try:
            for stmt in self.statement_list(getattr(node, "body", [])):
                code += self.generate_statement(stmt, indent + 1)
        finally:
            self.loop_depth -= 1

        code += f"{indent_str}}}\n"
        return code

    def generate_for_in_iterable(self, iterable_node):
        if isinstance(iterable_node, RangeNode):
            start = self.generate_expression(iterable_node.start)
            end = self.generate_expression(iterable_node.end)
            operator = "..=" if iterable_node.inclusive else ".."
            return f"{start}{operator}{end}"

        iterable = self.generate_expression(iterable_node)
        return f"0..{iterable}"

    def generate_while(self, node, indent):
        indent_str = "    " * indent
        condition = self.generate_expression(node.condition)

        code = f"{indent_str}while {condition} {{\n"

        self.loop_depth += 1
        try:
            for stmt in self.statement_list(node.body):
                code += self.generate_statement(stmt, indent + 1)
        finally:
            self.loop_depth -= 1

        code += f"{indent_str}}}\n"
        return code

    def generate_loop(self, node, indent):
        indent_str = "    " * indent
        code = f"{indent_str}loop {{\n"

        self.loop_depth += 1
        try:
            for stmt in self.statement_list(node.body):
                code += self.generate_statement(stmt, indent + 1)
        finally:
            self.loop_depth -= 1

        code += f"{indent_str}}}\n"
        return code

    def generate_do_while(self, node, indent):
        indent_str = "    " * indent
        break_flag = f"__cgl_do_break_{self.do_while_counter}"
        self.do_while_counter += 1
        condition = self.generate_expression(node.condition)

        code = f"{indent_str}let mut {break_flag}: bool = false;\n"
        code += f"{indent_str}loop {{\n"
        code += f"{indent_str}    loop {{\n"

        self.loop_depth += 1
        self.do_while_contexts.append(
            {"loop_depth": self.loop_depth, "break_flag": break_flag}
        )
        try:
            for stmt in self.statement_list(node.body):
                code += self.generate_statement(stmt, indent + 2)
        finally:
            self.do_while_contexts.pop()
            self.loop_depth -= 1

        if not self.statement_body_terminates_inner_loop(node.body):
            code += f"{indent_str}        break;\n"
        code += f"{indent_str}    }}\n"
        code += f"{indent_str}    if {break_flag} {{\n"
        code += f"{indent_str}        break;\n"
        code += f"{indent_str}    }}\n"
        code += f"{indent_str}    if !({condition}) {{\n"
        code += f"{indent_str}        break;\n"
        code += f"{indent_str}    }}\n"
        code += f"{indent_str}}}\n"

        return code

    def generate_expression(self, expr):
        """Render a CrossGL expression as Rust expression syntax."""
        if expr is None:
            return ""
        elif isinstance(expr, str):
            return expr
        elif isinstance(expr, (int, float, bool)):
            if isinstance(expr, bool):
                return "true" if expr else "false"
            return str(expr)
        elif hasattr(expr, "__class__") and "Literal" in str(expr.__class__):
            if hasattr(expr, "value"):
                literal_type = getattr(
                    getattr(expr, "literal_type", None), "name", None
                )
                return self.format_literal(expr.value, literal_type)
            return str(expr)
        elif hasattr(expr, "__class__") and "Identifier" in str(expr.__class__):
            return getattr(expr, "name", str(expr))
        elif isinstance(expr, VariableNode):
            if hasattr(expr, "name"):
                return expr.name
            else:
                return str(expr)
        elif hasattr(expr, "__class__") and "BinaryOp" in str(expr.__class__):
            left = self.generate_expression(getattr(expr, "left", ""))
            right = self.generate_expression(getattr(expr, "right", ""))
            op = getattr(expr, "operator", getattr(expr, "op", "+"))
            return f"({left} {self.map_operator(op)} {right})"
        elif isinstance(expr, AssignmentNode):
            return self.generate_assignment(expr)
        elif isinstance(expr, ArrayLiteralNode):
            return self.generate_array_literal_expression(expr)
        elif hasattr(expr, "__class__") and "UnaryOp" in str(expr.__class__):
            operand = self.generate_expression(getattr(expr, "operand", ""))
            op = getattr(expr, "operator", getattr(expr, "op", "+"))
            op = self.map_operator(op)
            if op in ["++", "--"]:
                assignment_op = "+=" if op == "++" else "-="
                return f"{operand} {assignment_op} 1"
            return f"({op}{operand})"
        elif hasattr(expr, "__class__") and "ArrayAccess" in str(expr.__class__):
            array_expr = getattr(expr, "array_expr", getattr(expr, "array", ""))
            index_expr = getattr(expr, "index_expr", getattr(expr, "index", ""))
            array = self.generate_expression(array_expr)
            index = self.generate_expression(index_expr)
            return f"{array}[{index}]"
        elif hasattr(expr, "__class__") and "FunctionCall" in str(expr.__class__):
            func_expr = getattr(expr, "function", getattr(expr, "name", "unknown"))
            func_name = None
            if hasattr(func_expr, "name"):
                func_name = func_expr.name
                callee = func_name
            elif isinstance(func_expr, str):
                func_name = func_expr
                callee = func_expr
            else:
                callee = self.generate_expression(func_expr)
            args = getattr(expr, "arguments", getattr(expr, "args", []))

            func_name = self.function_map.get(func_name, func_name)
            if func_name == "saturate" and len(args) == 1:
                arg = self.generate_expression(args[0])
                return f"clamp({arg}, 0.0, 1.0)"

            scalar_cast = self.generate_scalar_constructor_call(func_name, args)
            if scalar_cast is not None:
                return scalar_cast

            vector_info = self.vector_type_info(func_name)
            if vector_info:
                rust_type = self.map_type(func_name)
                generated_args = self.generate_vector_constructor_args(
                    vector_info, args
                )
                args_str = ", ".join(generated_args)
                return f"{self.rust_constructor_path(rust_type)}::new({args_str})"

            if func_name in [
                "mat2",
                "mat3",
                "mat4",
                "mat2x2",
                "mat2x3",
                "mat2x4",
                "mat3x2",
                "mat3x3",
                "mat3x4",
                "mat4x2",
                "mat4x3",
                "mat4x4",
                "dmat2",
                "dmat3",
                "dmat4",
                "dmat2x2",
                "dmat2x3",
                "dmat2x4",
                "dmat3x2",
                "dmat3x3",
                "dmat3x4",
                "dmat4x2",
                "dmat4x3",
                "dmat4x4",
            ]:
                rust_type = self.map_type(func_name)
                args_str = ", ".join(self.generate_expression(arg) for arg in args)
                return f"{self.rust_constructor_path(rust_type)}::new({args_str})"

            args_str = ", ".join(self.generate_expression(arg) for arg in args)
            return f"{func_name or callee}({args_str})"
        elif hasattr(expr, "__class__") and "MemberAccess" in str(expr.__class__):
            obj_expr = getattr(expr, "object_expr", getattr(expr, "object", ""))
            member = getattr(expr, "member", "")
            obj = self.generate_expression(obj_expr)
            return f"{obj}.{member}"
        elif hasattr(expr, "__class__") and "TernaryOp" in str(expr.__class__):
            condition = self.generate_expression(getattr(expr, "condition", ""))
            true_expr = self.generate_expression(getattr(expr, "true_expr", ""))
            false_expr = self.generate_expression(getattr(expr, "false_expr", ""))
            return f"(if {condition} {{ {true_expr} }} else {{ {false_expr} }})"
        else:
            return str(expr)

    def generate_scalar_constructor_call(self, func_name, args):
        rust_type = self.scalar_constructor_type(func_name)
        if rust_type is None or len(args) != 1:
            return None

        arg = args[0]
        arg_expr = self.generate_expression(arg)

        if rust_type == "bool":
            arg_type = self.expression_result_type(arg)
            if arg_type == "bool":
                return arg_expr
            zero_literal = "0.0" if arg_type in {"float", "double", "half"} else "0"
            return f"({arg_expr} != {zero_literal})"

        return f"({arg_expr} as {rust_type})"

    def generate_vector_constructor_args(self, vector_info, args):
        if len(args) == 1:
            arg_type = self.expression_result_type(args[0])
            if arg_type is not None and not self.vector_type_info(arg_type):
                arg_expr = self.generate_expression(args[0])
                return [arg_expr] * vector_info["size"]

        generated_args = []
        for arg in args:
            arg_expr = self.generate_expression(arg)
            arg_info = self.vector_type_info(self.expression_result_type(arg))
            if arg_info is None:
                generated_args.append(arg_expr)
            else:
                generated_args.extend(
                    self.vector_argument_lane_expressions(arg, arg_expr, arg_info)
                )

            if len(generated_args) >= vector_info["size"]:
                return generated_args[: vector_info["size"]]

        return generated_args

    def vector_argument_lane_expressions(self, arg, arg_expr, arg_info):
        swizzle_components = self.member_swizzle_components(arg)
        if swizzle_components is not None:
            object_expr = getattr(arg, "object_expr", getattr(arg, "object", None))
            object_value = self.generate_expression(object_expr)
            return [f"{object_value}.{component}" for component in swizzle_components]

        components = ("x", "y", "z", "w")[: arg_info["size"]]
        return [f"{arg_expr}.{component}" for component in components]

    def member_swizzle_components(self, expr):
        if not isinstance(expr, MemberAccessNode):
            return None

        object_expr = getattr(expr, "object_expr", getattr(expr, "object", None))
        if not self.vector_type_info(self.expression_result_type(object_expr)):
            return None

        component_aliases = {
            "x": "x",
            "y": "y",
            "z": "z",
            "w": "w",
            "r": "x",
            "g": "y",
            "b": "z",
            "a": "w",
        }
        member = getattr(expr, "member", "")
        components = [component_aliases.get(component) for component in member]
        if not components or any(component is None for component in components):
            return None
        return components

    def scalar_constructor_type(self, func_name):
        scalar_types = {
            "bool": "bool",
            "char": "char",
            "short": "i16",
            "ushort": "u16",
            "int": "i32",
            "uint": "u32",
            "long": "i64",
            "ulong": "u64",
            "float": "f32",
            "double": "f64",
            "half": "f16",
            "i16": "i16",
            "u16": "u16",
            "i32": "i32",
            "u32": "u32",
            "i64": "i64",
            "u64": "u64",
            "f16": "f16",
            "f32": "f32",
            "f64": "f64",
        }
        return scalar_types.get(func_name)

    def format_literal(self, value, literal_type=None):
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

    def register_variable_type(self, name, type_name):
        if not name or type_name is None:
            return
        if hasattr(type_name, "name") or hasattr(type_name, "element_type"):
            type_name = self.convert_type_node_to_string(type_name)
        else:
            type_name = str(type_name)
        self.variable_types[name] = type_name

    def is_generated_struct_type(self, type_name):
        if type_name is None:
            return False
        if hasattr(type_name, "name") or hasattr(type_name, "element_type"):
            type_name = self.convert_type_node_to_string(type_name)
        else:
            type_name = str(type_name)
        return type_name in self.struct_member_types

    def get_expression_name(self, expr):
        if isinstance(expr, IdentifierNode):
            return expr.name
        if isinstance(expr, VariableNode):
            return expr.name
        if isinstance(expr, str):
            return expr
        if isinstance(expr, ArrayAccessNode):
            array_expr = getattr(expr, "array_expr", getattr(expr, "array", None))
            return self.get_expression_name(array_expr)
        return None

    def expression_result_type(self, expr):
        if expr is None:
            return None
        if isinstance(expr, (IdentifierNode, VariableNode, ArrayAccessNode)):
            return self.variable_types.get(self.get_expression_name(expr))
        if isinstance(expr, LiteralNode):
            literal_type = getattr(getattr(expr, "literal_type", None), "name", None)
            if literal_type:
                return literal_type
            if isinstance(expr.value, bool):
                return "bool"
            if isinstance(expr.value, float):
                return "float"
            if isinstance(expr.value, int):
                return "int"
            return None
        if isinstance(expr, FunctionCallNode):
            func_expr = getattr(expr, "function", getattr(expr, "name", None))
            func_name = getattr(func_expr, "name", func_expr)
            if isinstance(func_name, str) and self.vector_type_info(func_name):
                return func_name
            return None
        if isinstance(expr, BinaryOpNode):
            left_type = self.expression_result_type(expr.left)
            right_type = self.expression_result_type(expr.right)
            if self.vector_type_info(left_type):
                return left_type
            if self.vector_type_info(right_type):
                return right_type
            return left_type or right_type
        if isinstance(expr, UnaryOpNode):
            return self.expression_result_type(expr.operand)
        if isinstance(expr, TernaryOpNode):
            return self.expression_result_type(
                expr.true_expr
            ) or self.expression_result_type(expr.false_expr)
        if isinstance(expr, MemberAccessNode):
            object_expr = getattr(expr, "object_expr", getattr(expr, "object", None))
            object_type = self.expression_result_type(object_expr)
            object_type_name = (
                self.convert_type_node_to_string(object_type)
                if object_type is not None
                and (
                    hasattr(object_type, "name") or hasattr(object_type, "element_type")
                )
                else object_type
            )
            member = getattr(expr, "member", "")
            struct_members = self.struct_member_types.get(object_type_name, {})
            if member in struct_members:
                return struct_members[member]

            vector_info = self.vector_type_info(object_type)
            if not vector_info:
                return None
            if len(member) == 1:
                return vector_info["component_type"]
            if all(component in "xyzwrgba" for component in member):
                return self.vector_type_for_components(
                    vector_info["component_type"], len(member)
                )
        return None

    def vector_type_info(self, type_name):
        if type_name is None:
            return None
        if hasattr(type_name, "name") or hasattr(type_name, "element_type"):
            type_name = self.convert_type_node_to_string(type_name)
        else:
            type_name = str(type_name)

        mapped_type = self.map_type(type_name)
        vector_details = {
            "Vec2<f32>": ("float", 2),
            "Vec3<f32>": ("float", 3),
            "Vec4<f32>": ("float", 4),
            "Vec2<f64>": ("double", 2),
            "Vec3<f64>": ("double", 3),
            "Vec4<f64>": ("double", 4),
            "Vec2<i32>": ("int", 2),
            "Vec3<i32>": ("int", 3),
            "Vec4<i32>": ("int", 4),
            "Vec2<u32>": ("uint", 2),
            "Vec3<u32>": ("uint", 3),
            "Vec4<u32>": ("uint", 4),
            "Vec2<bool>": ("bool", 2),
            "Vec3<bool>": ("bool", 3),
            "Vec4<bool>": ("bool", 4),
        }
        details = vector_details.get(mapped_type)
        if details is None:
            return None
        component_type, size = details
        return {"component_type": component_type, "size": size}

    def vector_type_for_components(self, component_type, component_count):
        if component_count < 2 or component_count > 4:
            return component_type
        prefixes = {
            "float": "vec",
            "double": "dvec",
            "int": "ivec",
            "uint": "uvec",
            "bool": "bvec",
        }
        prefix = prefixes.get(component_type)
        if prefix is None:
            return None
        return f"{prefix}{component_count}"

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

    def map_type(self, vtype):
        """Map a CrossGL type name or type node to a Rust type string."""
        if vtype is None:
            return "f32"

        if hasattr(vtype, "name") or hasattr(vtype, "element_type"):
            vtype_str = self.convert_type_node_to_string(vtype)
        else:
            vtype_str = str(vtype)

        if "[" in vtype_str and "]" in vtype_str:
            base_type, size = parse_array_type(vtype_str)
            base_mapped = self.type_mapping.get(base_type, base_type)
            if size:
                return f"[{base_mapped}; {size}]"
            else:
                return f"Vec<{base_mapped}>"

        return self.type_mapping.get(vtype_str, vtype_str)

    def rust_constructor_path(self, rust_type):
        """Return a Rust path suitable for associated constructor calls."""
        rust_type = str(rust_type)
        if "<" not in rust_type:
            return rust_type
        return rust_type.replace("<", "::<", 1)

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
        """Map a CrossGL semantic to the Rust backend attribute name."""
        if semantic:
            return self.semantic_map.get(semantic, semantic)
        return ""

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
            "sampler3D": "Texture3D<f32>",
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
        self.user_function_names = set()
        self.user_function_return_types = {}
        self.user_function_param_types = {}
        self.swizzle_temp_counter = 0
        self.vector_arg_temp_counter = 0
        self.matrix_arg_temp_counter = 0

    def generate(self, ast):
        """Generate complete Rust-like shader source for a CrossGL AST."""
        self.variable_types = {}
        self.struct_member_types = {}
        self.current_return_type = None
        self.do_while_contexts = []
        self.for_contexts = []
        self.loop_depth = 0
        self.do_while_counter = 0
        self.user_function_names = self.collect_user_function_names(ast)
        self.user_function_return_types = self.collect_user_function_return_types(ast)
        self.user_function_param_types = self.collect_user_function_param_types(ast)
        self.swizzle_temp_counter = 0
        self.vector_arg_temp_counter = 0
        self.matrix_arg_temp_counter = 0
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
                code += self.generate_global_array_declaration(node)
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
                rust_type = self.map_type(var_type)
                if initial_value is not None:
                    init_expr = self.generate_expression_with_type(
                        initial_value, var_type, static_context=True
                    )
                    lazy_lock = self.static_array_literal_requires_lazy_lock(
                        var_type, initial_value
                    )
                else:
                    init_expr = self.rust_static_default_initializer(rust_type)
                    lazy_lock = False
                code += self.generate_static_declaration(
                    node.name, rust_type, init_expr, lazy_lock=lazy_lock
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

    def collect_user_function_names(self, node):
        names = set()

        def collect(current):
            if current is None:
                return
            if isinstance(current, list):
                for item in current:
                    collect(item)
                return
            if isinstance(current, FunctionNode):
                names.add(current.name)
            for function in getattr(current, "functions", []):
                collect(function)
            for function in getattr(current, "local_functions", []):
                collect(function)
            collect(getattr(current, "entry_point", None))
            stages = getattr(current, "stages", {})
            if isinstance(stages, dict):
                for stage in stages.values():
                    collect(stage)

        collect(node)
        names.discard(None)
        return names

    def is_user_defined_function(self, func_name):
        return isinstance(func_name, str) and func_name in self.user_function_names

    def collect_user_function_return_types(self, node):
        return_types = {}

        def collect(current):
            if current is None:
                return
            if isinstance(current, list):
                for item in current:
                    collect(item)
                return
            if isinstance(current, FunctionNode):
                return_type = getattr(current, "return_type", None)
                if current.name and return_type is not None:
                    return_types[current.name] = self.convert_type_node_to_string(
                        return_type
                    )
            for function in getattr(current, "functions", []):
                collect(function)
            for function in getattr(current, "local_functions", []):
                collect(function)
            collect(getattr(current, "entry_point", None))
            stages = getattr(current, "stages", {})
            if isinstance(stages, dict):
                for stage in stages.values():
                    collect(stage)

        collect(node)
        return return_types

    def collect_user_function_param_types(self, node):
        param_types = {}

        def collect(current):
            if current is None:
                return
            if isinstance(current, list):
                for item in current:
                    collect(item)
                return
            if isinstance(current, FunctionNode) and current.name:
                params = getattr(current, "parameters", getattr(current, "params", []))
                param_types[current.name] = [
                    self.function_parameter_type(param) for param in params
                ]
            for function in getattr(current, "functions", []):
                collect(function)
            for function in getattr(current, "local_functions", []):
                collect(function)
            collect(getattr(current, "entry_point", None))
            stages = getattr(current, "stages", {})
            if isinstance(stages, dict):
                for stage in stages.values():
                    collect(stage)

        collect(node)
        return param_types

    def derive_traits_for_members(self, members, include_default=False):
        traits = ["Debug", "Clone"]
        if not self.members_have_unsized_arrays(members):
            traits.append("Copy")
        if include_default:
            traits.append("Default")
        return ", ".join(traits)

    def members_have_unsized_arrays(self, members):
        return any(self.member_has_unsized_array(member) for member in members)

    def member_has_unsized_array(self, member):
        if isinstance(member, ArrayNode):
            return getattr(member, "size", None) is None

        member_type = getattr(member, "member_type", None)
        if member_type is not None and member_type.__class__.__name__ == "ArrayType":
            return getattr(member_type, "size", None) is None

        vtype = getattr(member, "vtype", None)
        return isinstance(vtype, str) and vtype.endswith("[]")

    def generate_struct(self, node):
        members = getattr(node, "members", [])
        derive_traits = self.derive_traits_for_members(members, include_default=True)
        code = f"#[repr(C)]\n#[derive({derive_traits})]\n"
        code += f"pub struct {node.name} {{\n"
        member_types = {}

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
                members = getattr(node, "members", [])
                derive_traits = self.derive_traits_for_members(members)
                code += f"#[repr(C)]\n#[derive({derive_traits})]\n"
                code += f"pub struct {node.name} {{\n"
                for member in members:
                    if isinstance(member, ArrayNode):
                        if member.size:
                            code += f"    pub {member.name}: [{self.map_type(member.element_type)}; {member.size}],\n"
                        else:
                            code += f"    pub {member.name}: Vec<{self.map_type(member.element_type)}>,\n"
                    else:
                        code += f"    pub {member.name}: {self.map_type(self.get_member_type(member))},\n"
                code += "}\n\n"
                code += self.generate_cbuffer_member_statics(members)
            elif hasattr(node, "name") and hasattr(node, "members"):  # CbufferNode
                members = getattr(node, "members", [])
                derive_traits = self.derive_traits_for_members(members)
                code += f"#[repr(C)]\n#[derive({derive_traits})]\n"
                code += f"pub struct {node.name} {{\n"
                for member in members:
                    if isinstance(member, ArrayNode):
                        if member.size:
                            code += f"    pub {member.name}: [{self.map_type(member.element_type)}; {member.size}],\n"
                        else:
                            code += f"    pub {member.name}: Vec<{self.map_type(member.element_type)}>,\n"
                    else:
                        code += f"    pub {member.name}: {self.map_type(self.get_member_type(member))},\n"
                code += "}\n\n"
                code += self.generate_cbuffer_member_statics(members)
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
            initializer = self.rust_static_default_initializer(member_type)
            code += f"static {member.name}: {member_type} = {initializer};\n"
        return code

    def generate_global_array_declaration(self, node):
        element_type_name = self.convert_type_node_to_string(
            getattr(node, "element_type", "float")
        )
        element_type = self.map_type(element_type_name)
        size = self.format_array_size(getattr(node, "size", None))
        initial_value = getattr(node, "initial_value", getattr(node, "value", None))

        if size is None:
            rust_type = f"Vec<{element_type}>"
            if isinstance(initial_value, ArrayLiteralNode):
                target_type = f"{element_type_name}[]"
                initializer = self.generate_expression_with_type(
                    initial_value, target_type, static_context=True
                )
                lazy_lock = self.static_array_literal_requires_lazy_lock(
                    target_type, initial_value
                )
            else:
                initializer = self.rust_static_default_initializer(rust_type)
                lazy_lock = False
        else:
            rust_type = f"[{element_type}; {size}]"
            if isinstance(initial_value, ArrayLiteralNode):
                target_type = f"{element_type_name}[{size}]"
                initializer = self.generate_expression_with_type(
                    initial_value, target_type, static_context=True
                )
                lazy_lock = self.static_array_literal_requires_lazy_lock(
                    target_type, initial_value
                )
            else:
                initializer = self.rust_static_default_initializer(rust_type)
                lazy_lock = False

        return self.generate_static_declaration(
            node.name, rust_type, initializer, lazy_lock=lazy_lock
        )

    def generate_static_declaration(
        self, name, rust_type, initializer, lazy_lock=False
    ):
        if lazy_lock:
            return (
                f"static {name}: std::sync::LazyLock<{rust_type}> = "
                f"std::sync::LazyLock::new(|| {initializer});\n"
            )
        return f"static {name}: {rust_type} = {initializer};\n"

    def static_array_literal_requires_lazy_lock(self, target_type, initial_value):
        if not isinstance(initial_value, ArrayLiteralNode):
            return False
        if not self.is_array_type_name(target_type):
            return False

        base_type, size = parse_array_type(str(target_type))
        if size is None:
            return True

        return self.is_rust_shader_pod_value_type(self.map_type(base_type))

    def rust_static_default_initializer(self, rust_type):
        rust_type = str(rust_type)

        if rust_type.startswith("Vec<"):
            return "Vec::new()"

        array_type = self.parse_rust_array_type(rust_type)
        if array_type is not None:
            element_type, size = array_type
            element_initializer = self.rust_scalar_default_literal(element_type)
            if element_initializer is not None:
                return f"[{element_initializer}; {size}]"
            if self.is_rust_shader_pod_value_type(element_type):
                return "unsafe { std::mem::zeroed() }"
            return "Default::default()"

        scalar_initializer = self.rust_scalar_default_literal(rust_type)
        if scalar_initializer is not None:
            return scalar_initializer

        return "Default::default()"

    def parse_rust_array_type(self, rust_type):
        if not (rust_type.startswith("[") and rust_type.endswith("]")):
            return None
        body = rust_type[1:-1]
        if ";" not in body:
            return None
        element_type, size = body.rsplit(";", 1)
        return element_type.strip(), size.strip()

    def rust_scalar_default_literal(self, rust_type):
        if rust_type in {"f16", "f32", "f64"}:
            return "0.0"
        if rust_type in {
            "i8",
            "i16",
            "i32",
            "i64",
            "i128",
            "isize",
            "u8",
            "u16",
            "u32",
            "u64",
            "u128",
            "usize",
        }:
            return "0"
        if rust_type == "bool":
            return "false"
        if rust_type == "char":
            return r"'\0'"
        if rust_type == "&'static str":
            return '""'
        return None

    def is_rust_shader_pod_value_type(self, rust_type):
        return str(rust_type).startswith(
            (
                "Vec2<",
                "Vec3<",
                "Vec4<",
                "Mat2<",
                "Mat3<",
                "Mat4<",
                "Mat2x",
                "Mat3x",
                "Mat4x",
            )
        ) and str(rust_type).endswith(">")

    def generate_function(self, func, indent=0, shader_type=None):
        """Render one CrossGL function or shader entry point as Rust code."""
        code = ""
        code += "  " * indent
        saved_variable_types = self.variable_types.copy()
        saved_return_type = self.current_return_type

        param_list = getattr(func, "parameters", getattr(func, "params", []))
        params = []
        for p in param_list:
            param_type = self.function_parameter_type(p)
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

    def function_parameter_type(self, param):
        if hasattr(param, "param_type"):
            return self.convert_type_node_to_string(param.param_type)
        if hasattr(param, "vtype"):
            vtype = param.vtype
            if hasattr(vtype, "name") or hasattr(vtype, "element_type"):
                return self.convert_type_node_to_string(vtype)
            return vtype
        return "float"

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
                init_expr = self.normalize_assignment_rhs(
                    vtype, initial_value, init_expr, "="
                )
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
                    return_expr = self.generate_expression_with_type(
                        stmt.value, self.current_return_type
                    )
                    return_expr = self.normalize_assignment_rhs(
                        self.current_return_type, stmt.value, return_expr, "="
                    )
                    return f"{indent_str}return {return_expr};\n"
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

        if self.is_inferred_declaration_type(vtype):
            return None
        return vtype

    def is_inferred_declaration_type(self, type_name):
        if type_name is None:
            return True
        if hasattr(type_name, "name") or hasattr(type_name, "element_type"):
            type_name = self.convert_type_node_to_string(type_name)
        else:
            type_name = str(type_name)
        return type_name.strip() in {"", "None", "auto"}

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

    def generate_expression_with_type(self, expr, target_type, static_context=False):
        if isinstance(expr, ArrayLiteralNode):
            return self.generate_array_literal_expression(
                expr, target_type, static_context=static_context
            )
        if isinstance(expr, BinaryOpNode):
            return self.generate_binary_expression(expr, target_type)
        if isinstance(expr, TernaryOpNode):
            return self.generate_ternary_expression(expr, target_type)
        return self.generate_expression(expr)

    def generate_binary_expression(self, expr, target_type=None):
        left_expr = getattr(expr, "left", "")
        right_expr = getattr(expr, "right", "")
        op = getattr(expr, "operator", getattr(expr, "op", "+"))
        mapped_op = self.map_operator(op)

        left_type = self.expression_result_type(left_expr)
        right_type = self.expression_result_type(right_expr)
        bool_vector_logical = self.generate_bool_vector_logical_expression(
            left_expr, right_expr, left_type, right_type, mapped_op
        )
        if bool_vector_logical is not None:
            return bool_vector_logical

        if mapped_op in {"&&", "||"}:
            left = self.generate_condition_expression(left_expr)
            right = self.generate_condition_expression(right_expr)
            return f"({left} {mapped_op} {right})"

        vector_comparison = self.generate_vector_comparison_expression(
            left_expr, right_expr, left_type, right_type, mapped_op
        )
        if vector_comparison is not None:
            return vector_comparison

        matrix_vector_plan = self.binary_matrix_vector_plan(
            left_type, right_type, mapped_op, target_type
        )
        if matrix_vector_plan is not None:
            left = self.generate_binary_composite_operand(
                left_expr, left_type, matrix_vector_plan["left_target_type"]
            )
            right = self.generate_binary_composite_operand(
                right_expr, right_type, matrix_vector_plan["right_target_type"]
            )
            return f"({left} {mapped_op} {right})"

        composite_operand_type = self.binary_composite_operand_type(
            left_type, right_type, mapped_op
        )
        if composite_operand_type is not None:
            left = self.generate_binary_composite_operand(
                left_expr, left_type, composite_operand_type
            )
            right = self.generate_binary_composite_operand(
                right_expr, right_type, composite_operand_type
            )
            return f"({left} {mapped_op} {right})"

        operand_type = self.binary_scalar_operand_type(
            left_type,
            right_type,
            mapped_op,
        )
        left = self.generate_expression(left_expr)
        right = self.generate_expression(right_expr)
        if operand_type is not None:
            left = self.normalize_binary_scalar_operand(
                left_expr,
                left,
                left_type,
                operand_type,
            )
            right = self.normalize_binary_scalar_operand(
                right_expr,
                right,
                right_type,
                operand_type,
            )

        return f"({left} {mapped_op} {right})"

    def generate_binary_composite_operand(self, expr, source_type, composite_type):
        if self.vector_type_info(source_type) or self.matrix_type_info(source_type):
            operand = self.generate_expression_with_type(expr, composite_type)
            return self.normalize_typed_expression_value(expr, operand, composite_type)

        component_type = self.composite_component_type(composite_type)
        if component_type is not None and self.normalize_scalar_type(source_type):
            operand = self.generate_expression_with_type(expr, component_type)
            return self.normalize_scalar_assignment_value(
                expr, operand, source_type, component_type
            )

        return self.generate_expression(expr)

    def composite_component_type(self, composite_type):
        vector_info = self.vector_type_info(composite_type)
        if vector_info is not None:
            return vector_info["component_type"]

        matrix_info = self.matrix_type_info(composite_type)
        if matrix_info is not None:
            return matrix_info["component_type"]

        return None

    def generate_bool_vector_logical_expression(
        self, left_expr, right_expr, left_type, right_type, operator
    ):
        plan = self.bool_vector_logical_plan(left_type, right_type, operator)
        if plan is None:
            return None

        temp_bindings = []
        left_lanes = self.vector_comparison_operand_lanes(
            left_expr, left_type, "bool", plan["size"], temp_bindings
        )
        right_lanes = self.vector_comparison_operand_lanes(
            right_expr, right_type, "bool", plan["size"], temp_bindings
        )
        lanes = [
            f"({left_lane} {operator} {right_lane})"
            for left_lane, right_lane in zip(left_lanes, right_lanes)
        ]
        return self.generate_constructor_call(
            self.map_type(plan["result_type"]), lanes, temp_bindings
        )

    def generate_bool_vector_not_expression(self, expr, operator):
        if operator != "!":
            return None

        vector_info = self.vector_type_info(self.expression_result_type(expr))
        if vector_info is None or vector_info["component_type"] != "bool":
            return None

        temp_bindings = []
        lanes = self.vector_argument_lane_expressions(
            expr, vector_info, temp_bindings, "bool"
        )
        lanes = [f"(!{lane})" for lane in lanes[: vector_info["size"]]]
        result_type = self.vector_type_for_components("bool", vector_info["size"])
        return self.generate_constructor_call(
            self.map_type(result_type), lanes, temp_bindings
        )

    def generate_vector_comparison_expression(
        self, left_expr, right_expr, left_type, right_type, operator
    ):
        plan = self.vector_comparison_plan(left_type, right_type, operator)
        if plan is None:
            return None

        temp_bindings = []
        left_lanes = self.vector_comparison_operand_lanes(
            left_expr,
            left_type,
            plan["component_type"],
            plan["size"],
            temp_bindings,
        )
        right_lanes = self.vector_comparison_operand_lanes(
            right_expr,
            right_type,
            plan["component_type"],
            plan["size"],
            temp_bindings,
        )
        lanes = [
            f"({left_lane} {operator} {right_lane})"
            for left_lane, right_lane in zip(left_lanes, right_lanes)
        ]
        return self.generate_constructor_call(
            self.map_type(plan["result_type"]), lanes, temp_bindings
        )

    def vector_comparison_operand_lanes(
        self, expr, source_type, target_component_type, size, temp_bindings
    ):
        source_info = self.vector_type_info(source_type)
        if source_info is not None:
            return self.vector_argument_lane_expressions(
                expr, source_info, temp_bindings, target_component_type
            )

        scalar_expr = self.generate_expression_with_type(expr, target_component_type)
        scalar_expr = self.normalize_scalar_assignment_value(
            expr, scalar_expr, source_type, target_component_type
        )
        if not self.is_repeat_safe_expression(expr):
            temp_name = self.next_vector_arg_temp_name()
            temp_bindings.append((temp_name, scalar_expr))
            scalar_expr = temp_name
        return [scalar_expr] * size

    def generate_condition_expression(self, expr):
        if isinstance(expr, UnaryOpNode):
            op = self.map_operator(getattr(expr, "operator", getattr(expr, "op", "")))
            if op == "!":
                operand = getattr(expr, "operand", "")
                return f"!({self.generate_condition_expression(operand)})"

        condition = self.generate_expression(expr)
        return self.normalize_condition_expression(expr, condition)

    def normalize_condition_expression(self, expr, generated_expr):
        condition_type = self.normalize_scalar_type(self.expression_result_type(expr))
        if condition_type is None or condition_type == "bool":
            return generated_expr

        zero_literal = "0.0" if condition_type in {"f16", "f32", "f64"} else "0"
        return f"({generated_expr} != {zero_literal})"

    def generate_ternary_expression(self, expr, target_type=None):
        condition_expr = getattr(expr, "condition", "")
        true_expr = getattr(expr, "true_expr", "")
        false_expr = getattr(expr, "false_expr", "")

        bool_vector_ternary = self.generate_bool_vector_ternary_expression(
            condition_expr, true_expr, false_expr, target_type
        )
        if bool_vector_ternary is not None:
            return bool_vector_ternary

        condition = self.generate_condition_expression(condition_expr)
        branch_type = target_type or self.expression_result_type(expr)

        true_value = self.generate_expression_with_type(true_expr, branch_type)
        false_value = self.generate_expression_with_type(false_expr, branch_type)
        true_value = self.normalize_typed_expression_value(
            true_expr, true_value, branch_type
        )
        false_value = self.normalize_typed_expression_value(
            false_expr, false_value, branch_type
        )
        return f"(if {condition} {{ {true_value} }} else {{ {false_value} }})"

    def generate_bool_vector_ternary_expression(
        self, condition_expr, true_expr, false_expr, target_type=None
    ):
        plan = self.bool_vector_ternary_plan(
            condition_expr, true_expr, false_expr, target_type
        )
        if plan is None:
            return None

        temp_bindings = []
        condition_info = self.vector_type_info(
            self.expression_result_type(condition_expr)
        )
        condition_lanes = self.vector_argument_lane_expressions(
            condition_expr, condition_info, temp_bindings, "bool"
        )
        true_lanes = self.vector_comparison_operand_lanes(
            true_expr,
            self.expression_result_type(true_expr),
            plan["component_type"],
            plan["size"],
            temp_bindings,
        )
        false_lanes = self.vector_comparison_operand_lanes(
            false_expr,
            self.expression_result_type(false_expr),
            plan["component_type"],
            plan["size"],
            temp_bindings,
        )
        lanes = [
            f"(if {condition_lane} {{ {true_lane} }} else {{ {false_lane} }})"
            for condition_lane, true_lane, false_lane in zip(
                condition_lanes, true_lanes, false_lanes
            )
        ]
        return self.generate_constructor_call(
            self.map_type(plan["result_type"]), lanes, temp_bindings
        )

    def is_array_type_name(self, type_name):
        return type_name is not None and "[" in str(type_name) and "]" in str(type_name)

    def generate_array_literal_expression(
        self, expr, target_type=None, static_context=False
    ):
        elements = [self.generate_expression(element) for element in expr.elements]

        if self.is_array_type_name(target_type):
            base_type, size = parse_array_type(str(target_type))
            if size is None:
                return f"vec![{', '.join(elements)}]"

            elements = elements[:size]
            padding = self.rust_array_padding_expression(
                base_type, static_context=static_context
            )
            while len(elements) < size:
                elements.append(padding)

        return f"[{', '.join(elements)}]"

    def rust_array_padding_expression(self, base_type, static_context=False):
        if static_context:
            rust_type = self.map_type(base_type)
            scalar_initializer = self.rust_scalar_default_literal(rust_type)
            if scalar_initializer is not None:
                return scalar_initializer
            if self.is_rust_shader_pod_value_type(rust_type):
                return "unsafe { std::mem::zeroed() }"
        return "Default::default()"

    def generate_assignment(self, node):
        # Handle both old and new AST assignment structures
        if hasattr(node, "target") and hasattr(node, "value"):
            # New AST structure
            lhs = self.generate_expression(node.target)
            lhs_type = self.expression_result_type(node.target)
            rhs = self.generate_expression_with_type(node.value, lhs_type)
            rhs = self.normalize_assignment_rhs(
                lhs_type, node.value, rhs, getattr(node, "operator", "=")
            )
            op = getattr(node, "operator", "=")
        else:
            # Old AST structure
            lhs = self.generate_expression(node.left)
            lhs_type = self.expression_result_type(node.left)
            rhs = self.generate_expression_with_type(node.right, lhs_type)
            rhs = self.normalize_assignment_rhs(
                lhs_type, node.right, rhs, getattr(node, "operator", "=")
            )
            op = getattr(node, "operator", "=")
        return f"{lhs} {op} {rhs}"

    def normalize_assignment_rhs(self, lhs_type, rhs_expr, generated_rhs, operator):
        if operator == "=":
            return self.normalize_typed_expression_value(
                rhs_expr, generated_rhs, lhs_type
            )

        compound_ops = {
            "+=",
            "-=",
            "*=",
            "/=",
            "%=",
            "^=",
            "|=",
            "&=",
            "<<=",
            ">>=",
        }
        if operator not in compound_ops:
            return generated_rhs

        target_type = self.normalize_scalar_type(lhs_type)
        source_type = self.expression_result_type(rhs_expr)
        if target_type is None or self.normalize_scalar_type(source_type) is None:
            return generated_rhs
        return self.normalize_binary_scalar_operand(
            rhs_expr, generated_rhs, source_type, target_type
        )

    def normalize_typed_expression_value(self, expr, generated_expr, target_type):
        if isinstance(expr, TernaryOpNode):
            return generated_expr
        if self.generated_binary_expression_matches_target(expr, target_type):
            return generated_expr

        vector_expr = self.normalize_vector_typed_expression(expr, target_type)
        if vector_expr is not None:
            return vector_expr
        matrix_expr = self.normalize_matrix_typed_expression(expr, target_type)
        if matrix_expr is not None:
            return matrix_expr
        return self.normalize_scalar_assignment_value(
            expr, generated_expr, self.expression_result_type(expr), target_type
        )

    def generated_binary_expression_matches_target(self, expr, target_type):
        if not isinstance(expr, BinaryOpNode) or target_type is None:
            return False

        left_type = self.expression_result_type(expr.left)
        right_type = self.expression_result_type(expr.right)
        operator = self.map_operator(
            getattr(expr, "operator", getattr(expr, "op", None))
        )
        matrix_vector_plan = self.binary_matrix_vector_plan(
            left_type, right_type, operator, target_type
        )
        if matrix_vector_plan is None:
            return False
        return self.type_names_match(matrix_vector_plan["result_type"], target_type)

    def type_names_match(self, left_type, right_type):
        left_scalar = self.normalize_scalar_type(left_type)
        right_scalar = self.normalize_scalar_type(right_type)
        if left_scalar is not None or right_scalar is not None:
            return left_scalar == right_scalar
        return self.map_type(left_type) == self.map_type(right_type)

    def normalize_scalar_assignment_value(
        self, expr, generated_expr, source_type, target_type
    ):
        source_type = self.normalize_scalar_type(source_type)
        target_type = self.normalize_scalar_type(target_type)
        if source_type is None or target_type is None or source_type == target_type:
            return generated_expr

        if target_type == "bool":
            zero_literal = "0.0" if source_type in {"f16", "f32", "f64"} else "0"
            return f"({generated_expr} != {zero_literal})"

        if source_type == "bool" and target_type in {"f16", "f32", "f64"}:
            return f"(if {generated_expr} {{ 1.0 }} else {{ 0.0 }})"

        if self.is_integer_literal_expression(expr):
            if target_type in {"f32", "f64"}:
                return f"{expr.value}.0"
            if target_type == "f16":
                return f"({generated_expr} as f16)"
            return generated_expr

        return f"({generated_expr} as {target_type})"

    def generate_if(self, node, indent):
        indent_str = "    " * indent
        condition = self.generate_condition_expression(
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
                elif_condition = self.generate_condition_expression(
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
        condition = self.generate_condition_expression(node.condition)
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
        condition = self.generate_condition_expression(node.condition)

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
        condition = self.generate_condition_expression(node.condition)

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
            return self.generate_binary_expression(expr)
        elif isinstance(expr, AssignmentNode):
            return self.generate_assignment(expr)
        elif isinstance(expr, ArrayLiteralNode):
            return self.generate_array_literal_expression(expr)
        elif hasattr(expr, "__class__") and "UnaryOp" in str(expr.__class__):
            operand_expr = getattr(expr, "operand", "")
            op = getattr(expr, "operator", getattr(expr, "op", "+"))
            op = self.map_operator(op)
            if op in ["++", "--"]:
                operand = self.generate_expression(operand_expr)
                assignment_op = "+=" if op == "++" else "-="
                return f"{operand} {assignment_op} 1"
            bool_vector_not = self.generate_bool_vector_not_expression(operand_expr, op)
            if bool_vector_not is not None:
                return bool_vector_not
            operand = self.generate_expression(operand_expr)
            return f"({op}{operand})"
        elif hasattr(expr, "__class__") and "ArrayAccess" in str(expr.__class__):
            array_expr = getattr(expr, "array_expr", getattr(expr, "array", ""))
            index_expr = getattr(expr, "index_expr", getattr(expr, "index", ""))
            array = self.generate_expression(array_expr)
            index = self.generate_array_index_expression(index_expr)
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

            if self.is_user_defined_function(func_name):
                args_str = ", ".join(
                    self.generate_user_function_call_args(func_name, args)
                )
                return f"{callee}({args_str})"

            func_name = self.function_map.get(func_name, func_name)
            if func_name == "saturate" and len(args) == 1:
                arg = self.generate_expression(args[0])
                return f"clamp({arg}, 0.0, 1.0)"

            scalar_cast = self.generate_scalar_constructor_call(func_name, args)
            if scalar_cast is not None:
                return scalar_cast

            vector_info = self.vector_type_info(func_name)
            if vector_info:
                return self.generate_vector_constructor_call(
                    func_name, vector_info, args
                )

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
                return self.generate_matrix_constructor_call(func_name, args)

            args_str = ", ".join(self.generate_expression(arg) for arg in args)
            return f"{func_name or callee}({args_str})"
        elif hasattr(expr, "__class__") and "MemberAccess" in str(expr.__class__):
            return self.generate_member_access_expression(expr)
        elif hasattr(expr, "__class__") and "TernaryOp" in str(expr.__class__):
            return self.generate_ternary_expression(expr)
        else:
            return str(expr)

    def generate_user_function_call_args(self, func_name, args):
        param_types = self.user_function_param_types.get(func_name, [])
        generated_args = []
        for index, arg in enumerate(args):
            param_type = param_types[index] if index < len(param_types) else None
            if param_type is None:
                generated_args.append(self.generate_expression(arg))
                continue

            arg_expr = self.generate_expression_with_type(arg, param_type)
            arg_expr = self.normalize_user_function_call_arg(arg, arg_expr, param_type)
            generated_args.append(arg_expr)
        return generated_args

    def normalize_user_function_call_arg(self, arg, generated_arg, param_type):
        return self.normalize_typed_expression_value(arg, generated_arg, param_type)

    def normalize_vector_typed_expression(self, expr, target_type):
        target_info = self.vector_type_info(target_type)
        source_info = self.vector_type_info(self.expression_result_type(expr))
        if target_info is None or source_info is None:
            return None
        if target_info["size"] != source_info["size"]:
            return None
        if target_info["component_type"] == source_info["component_type"]:
            return None

        temp_bindings = []
        lanes = self.vector_argument_lane_expressions(
            expr, source_info, temp_bindings, target_info["component_type"]
        )
        lanes = lanes[: target_info["size"]]
        return self.generate_constructor_call(
            self.map_type(target_type), lanes, temp_bindings
        )

    def normalize_matrix_typed_expression(self, expr, target_type):
        target_info = self.matrix_type_info(target_type)
        source_info = self.matrix_type_info(self.expression_result_type(expr))
        if target_info is None or source_info is None:
            return None
        if (
            target_info["columns"] != source_info["columns"]
            or target_info["rows"] != source_info["rows"]
        ):
            return None
        if target_info["component_type"] == source_info["component_type"]:
            return None

        temp_bindings = []
        lanes = self.matrix_argument_lane_expressions(
            expr, source_info, temp_bindings, target_info["component_type"]
        )
        return self.generate_constructor_call(
            self.map_type(target_type), lanes, temp_bindings
        )

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

    def generate_vector_constructor_call(self, func_name, vector_info, args):
        rust_type = self.map_type(func_name)
        generated_args, temp_bindings = self.generate_vector_constructor_args(
            vector_info, args
        )
        return self.generate_constructor_call(rust_type, generated_args, temp_bindings)

    def generate_matrix_constructor_call(self, func_name, args):
        rust_type = self.map_type(func_name)
        matrix_info = self.matrix_type_info(func_name)
        generated_args, temp_bindings = self.generate_matrix_constructor_args(
            matrix_info, args
        )
        return self.generate_constructor_call(rust_type, generated_args, temp_bindings)

    def generate_constructor_call(self, rust_type, generated_args, temp_bindings):
        args_str = ", ".join(generated_args)
        constructor = f"{self.rust_constructor_path(rust_type)}::new({args_str})"
        if not temp_bindings:
            return constructor

        bindings = " ".join(f"let {name} = {expr};" for name, expr in temp_bindings)
        return f"{{ {bindings} {constructor} }}"

    def generate_matrix_constructor_args(self, matrix_info, args):
        generated_args = []
        temp_bindings = []
        expected_count = matrix_info["columns"] * matrix_info["rows"]
        component_type = matrix_info["component_type"]

        for arg in args:
            arg_info = self.vector_type_info(self.expression_result_type(arg))
            if arg_info is None:
                arg_expr = self.generate_expression(arg)
                arg_expr = self.normalize_constructor_scalar_lane(
                    arg,
                    arg_expr,
                    self.expression_result_type(arg),
                    component_type,
                )
                generated_args.append(arg_expr)
            else:
                generated_args.extend(
                    self.vector_argument_lane_expressions(
                        arg, arg_info, temp_bindings, component_type
                    )
                )

            if len(generated_args) >= expected_count:
                return generated_args[:expected_count], temp_bindings

        return generated_args, temp_bindings

    def generate_vector_constructor_args(self, vector_info, args):
        component_type = vector_info["component_type"]
        if len(args) == 1:
            arg_type = self.expression_result_type(args[0])
            if arg_type is not None and not self.vector_type_info(arg_type):
                temp_bindings = []
                arg_expr = self.generate_expression(args[0])
                arg_expr = self.normalize_constructor_scalar_lane(
                    args[0], arg_expr, arg_type, component_type
                )
                if not self.is_repeat_safe_expression(args[0]):
                    temp_name = self.next_vector_arg_temp_name()
                    temp_bindings.append((temp_name, arg_expr))
                    arg_expr = temp_name
                return [arg_expr] * vector_info["size"], temp_bindings

        generated_args = []
        temp_bindings = []
        for arg in args:
            arg_info = self.vector_type_info(self.expression_result_type(arg))
            if arg_info is None:
                arg_expr = self.generate_expression(arg)
                arg_expr = self.normalize_constructor_scalar_lane(
                    arg,
                    arg_expr,
                    self.expression_result_type(arg),
                    component_type,
                )
                generated_args.append(arg_expr)
            else:
                generated_args.extend(
                    self.vector_argument_lane_expressions(
                        arg, arg_info, temp_bindings, component_type
                    )
                )

            if len(generated_args) >= vector_info["size"]:
                return generated_args[: vector_info["size"]], temp_bindings

        return generated_args, temp_bindings

    def vector_argument_lane_expressions(
        self, arg, arg_info, temp_bindings, target_component_type=None
    ):
        swizzle_components = self.member_swizzle_components(arg)
        if swizzle_components is not None:
            object_expr = getattr(arg, "object_expr", getattr(arg, "object", None))
            object_value = self.generate_vector_lane_source(
                object_expr, self.generate_expression(object_expr), temp_bindings
            )
            return [
                self.normalize_constructor_scalar_lane(
                    None,
                    f"{object_value}.{component}",
                    arg_info["component_type"],
                    target_component_type or arg_info["component_type"],
                )
                for component in swizzle_components
            ]

        arg_expr = self.generate_expression(arg)
        arg_expr = self.generate_vector_lane_source(arg, arg_expr, temp_bindings)
        components = ("x", "y", "z", "w")[: arg_info["size"]]
        return [
            self.normalize_constructor_scalar_lane(
                None,
                f"{arg_expr}.{component}",
                arg_info["component_type"],
                target_component_type or arg_info["component_type"],
            )
            for component in components
        ]

    def normalize_constructor_scalar_lane(
        self, expr, generated_expr, source_type, target_type
    ):
        source_type = self.normalize_scalar_type(source_type)
        target_type = self.normalize_scalar_type(target_type)
        if (
            isinstance(expr, LiteralNode)
            and isinstance(expr.value, float)
            and source_type in {"f32", "f64"}
            and target_type in {"f32", "f64"}
        ):
            return generated_expr
        return self.normalize_scalar_assignment_value(
            expr, generated_expr, source_type, target_type
        )

    def matrix_argument_lane_expressions(
        self, arg, matrix_info, temp_bindings, target_component_type=None
    ):
        arg_expr = self.generate_expression(arg)
        arg_expr = self.generate_matrix_lane_source(arg, arg_expr, temp_bindings)
        target_component_type = target_component_type or matrix_info["component_type"]
        components = ("x", "y", "z", "w")[: matrix_info["rows"]]
        lanes = []
        for column in range(matrix_info["columns"]):
            for component in components:
                lanes.append(
                    self.normalize_constructor_scalar_lane(
                        None,
                        f"{arg_expr}.c{column}.{component}",
                        matrix_info["component_type"],
                        target_component_type,
                    )
                )
        return lanes

    def generate_matrix_lane_source(self, expr, generated_expr, temp_bindings):
        if self.is_repeat_safe_expression(expr):
            return generated_expr
        temp_name = self.next_matrix_arg_temp_name()
        temp_bindings.append((temp_name, generated_expr))
        return temp_name

    def generate_vector_lane_source(self, expr, generated_expr, temp_bindings):
        if self.is_repeat_safe_expression(expr):
            return generated_expr
        temp_name = self.next_vector_arg_temp_name()
        temp_bindings.append((temp_name, generated_expr))
        return temp_name

    def generate_member_access_expression(self, expr):
        obj_expr = getattr(expr, "object_expr", getattr(expr, "object", ""))
        member = getattr(expr, "member", "")
        obj = self.generate_expression(obj_expr)

        swizzle_components = self.member_swizzle_components(expr)
        if swizzle_components is None:
            return f"{obj}.{member}"

        if len(swizzle_components) == 1:
            return f"{obj}.{swizzle_components[0]}"

        rust_type = self.swizzle_constructor_type(obj_expr, len(swizzle_components))
        if not self.is_repeat_safe_expression(obj_expr):
            temp_name = self.next_swizzle_temp_name()
            args = ", ".join(
                f"{temp_name}.{component}" for component in swizzle_components
            )
            return (
                f"{{ let {temp_name} = {obj}; "
                f"{self.rust_constructor_path(rust_type)}::new({args}) }}"
            )

        args = ", ".join(f"{obj}.{component}" for component in swizzle_components)
        return f"{self.rust_constructor_path(rust_type)}::new({args})"

    def swizzle_constructor_type(self, obj_expr, component_count):
        object_type = self.expression_result_type(obj_expr)
        vector_info = self.vector_type_info(object_type)
        component_type = vector_info["component_type"]
        result_type = self.vector_type_for_components(component_type, component_count)
        return self.map_type(result_type)

    def next_swizzle_temp_name(self):
        name = f"__cgl_swizzle_{self.swizzle_temp_counter}"
        self.swizzle_temp_counter += 1
        return name

    def next_vector_arg_temp_name(self):
        name = f"__cgl_vec_arg_{self.vector_arg_temp_counter}"
        self.vector_arg_temp_counter += 1
        return name

    def next_matrix_arg_temp_name(self):
        name = f"__cgl_mat_arg_{self.matrix_arg_temp_counter}"
        self.matrix_arg_temp_counter += 1
        return name

    def is_repeat_safe_expression(self, expr):
        if isinstance(expr, (IdentifierNode, VariableNode, LiteralNode)):
            return True
        if isinstance(expr, str):
            return True
        if isinstance(expr, MemberAccessNode):
            object_expr = getattr(expr, "object_expr", getattr(expr, "object", None))
            return self.is_repeat_safe_expression(object_expr)
        if isinstance(expr, ArrayAccessNode):
            array_expr = getattr(expr, "array_expr", getattr(expr, "array", None))
            index_expr = getattr(expr, "index_expr", getattr(expr, "index", None))
            return self.is_repeat_safe_expression(
                array_expr
            ) and self.is_repeat_safe_expression(index_expr)
        return False

    def generate_array_index_expression(self, index_expr):
        index = self.generate_expression(index_expr)
        if self.is_usize_compatible_index(index_expr, index):
            return index
        return f"{index} as usize"

    def is_usize_compatible_index(self, index_expr, generated_index):
        if isinstance(index_expr, LiteralNode) and isinstance(index_expr.value, int):
            return index_expr.value >= 0
        if isinstance(index_expr, int):
            return index_expr >= 0
        if isinstance(index_expr, str) and index_expr.isdigit():
            return True
        return generated_index.endswith(" as usize")

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

    def binary_scalar_result_type(self, left_type, right_type, operator=None):
        comparison_ops = {"<", ">", "<=", ">=", "==", "!="}
        logical_ops = {"&&", "||"}
        if operator in comparison_ops or operator in logical_ops:
            return "bool"

        return self.promoted_scalar_type(left_type, right_type)

    def binary_scalar_operand_type(self, left_type, right_type, operator=None):
        if operator in {"&&", "||"}:
            return None
        return self.promoted_scalar_type(left_type, right_type)

    def binary_composite_operand_type(self, left_type, right_type, operator=None):
        arithmetic_ops = {"+", "-", "*", "/", "%"}
        bitwise_ops = {"&", "|", "^", "<<", ">>"}

        vector_type = self.promoted_vector_type(left_type, right_type)
        if vector_type is not None:
            vector_info = self.vector_type_info(vector_type)
            component_type = self.normalize_scalar_type(vector_info["component_type"])
            if operator in arithmetic_ops:
                return vector_type
            if operator in bitwise_ops and component_type in {
                "i16",
                "u16",
                "i32",
                "u32",
                "i64",
                "u64",
            }:
                return vector_type
            return None

        matrix_type = self.promoted_matrix_type(left_type, right_type)
        if matrix_type is not None and operator in {"+", "-", "*", "/"}:
            return matrix_type

        vector_scalar_type = self.promoted_vector_scalar_type(left_type, right_type)
        if vector_scalar_type is not None:
            vector_info = self.vector_type_info(vector_scalar_type)
            component_type = self.normalize_scalar_type(vector_info["component_type"])
            if operator in arithmetic_ops:
                return vector_scalar_type
            if operator in bitwise_ops and component_type in {
                "i16",
                "u16",
                "i32",
                "u32",
                "i64",
                "u64",
            }:
                return vector_scalar_type
            return None

        matrix_scalar_type = self.promoted_matrix_scalar_type(left_type, right_type)
        if matrix_scalar_type is not None and operator in {"+", "-", "*", "/"}:
            return matrix_scalar_type
        return None

    def bool_vector_logical_plan(self, left_type, right_type, operator=None):
        if operator not in {"&&", "||"}:
            return None

        left_vector = self.vector_type_info(left_type)
        right_vector = self.vector_type_info(right_type)
        left_scalar = self.normalize_scalar_type(left_type)
        right_scalar = self.normalize_scalar_type(right_type)

        if left_vector is not None and left_vector["component_type"] != "bool":
            return None
        if right_vector is not None and right_vector["component_type"] != "bool":
            return None

        if left_vector is not None and right_vector is not None:
            if left_vector["size"] != right_vector["size"]:
                return None
            size = left_vector["size"]
        elif left_vector is not None and right_scalar == "bool":
            size = left_vector["size"]
        elif right_vector is not None and left_scalar == "bool":
            size = right_vector["size"]
        else:
            return None

        result_type = self.vector_type_for_components("bool", size)
        if result_type is None:
            return None
        return {"size": size, "result_type": result_type}

    def bool_vector_ternary_plan(
        self, condition_expr, true_expr, false_expr, target_type=None
    ):
        condition_info = self.vector_type_info(
            self.expression_result_type(condition_expr)
        )
        if condition_info is None or condition_info["component_type"] != "bool":
            return None

        size = condition_info["size"]
        true_type = self.expression_result_type(true_expr)
        false_type = self.expression_result_type(false_expr)
        result_type = self.bool_vector_ternary_result_type(
            true_type, false_type, size, target_type
        )
        result_info = self.vector_type_info(result_type)
        if result_info is None or result_info["size"] != size:
            return None
        return {
            "size": size,
            "result_type": result_type,
            "component_type": result_info["component_type"],
        }

    def bool_vector_ternary_result_type(
        self, true_type, false_type, condition_size, target_type=None
    ):
        target_info = self.vector_type_info(target_type)
        if target_info is not None and target_info["size"] == condition_size:
            return target_type

        true_info = self.vector_type_info(true_type)
        false_info = self.vector_type_info(false_type)
        true_scalar = self.normalize_scalar_type(true_type)
        false_scalar = self.normalize_scalar_type(false_type)

        if true_info is not None and true_info["size"] == condition_size:
            if false_info is not None and false_info["size"] == condition_size:
                return self.promoted_vector_type(true_type, false_type)
            if false_scalar is not None:
                return self.vector_type_for_promoted_scalar(true_info, false_scalar)
            return true_type

        if false_info is not None and false_info["size"] == condition_size:
            if true_scalar is not None:
                return self.vector_type_for_promoted_scalar(false_info, true_scalar)
            return false_type

        if true_scalar is None or false_scalar is None:
            return None
        component_type = self.promoted_scalar_type(true_scalar, false_scalar)
        if component_type is None:
            return None
        return self.vector_type_for_components(component_type, condition_size)

    def vector_comparison_plan(self, left_type, right_type, operator=None):
        if operator not in {"<", ">", "<=", ">=", "==", "!="}:
            return None

        left_vector = self.vector_type_info(left_type)
        right_vector = self.vector_type_info(right_type)
        left_scalar = self.normalize_scalar_type(left_type)
        right_scalar = self.normalize_scalar_type(right_type)

        if left_vector is not None and right_vector is not None:
            if left_vector["size"] != right_vector["size"]:
                return None
            component_type = self.promoted_scalar_type(
                left_vector["component_type"], right_vector["component_type"]
            )
            size = left_vector["size"]
        elif left_vector is not None and right_scalar is not None:
            component_type = self.promoted_scalar_type(
                left_vector["component_type"], right_scalar
            )
            size = left_vector["size"]
        elif right_vector is not None and left_scalar is not None:
            component_type = self.promoted_scalar_type(
                left_scalar, right_vector["component_type"]
            )
            size = right_vector["size"]
        else:
            return None

        if component_type is None:
            return None
        component_type = self.normalize_scalar_type(component_type)
        if component_type == "bool" and operator not in {"==", "!="}:
            return None

        result_type = self.vector_type_for_components("bool", size)
        if result_type is None:
            return None
        return {
            "component_type": component_type,
            "size": size,
            "result_type": result_type,
        }

    def binary_matrix_vector_plan(
        self, left_type, right_type, operator=None, target_type=None
    ):
        if operator != "*":
            return None

        left_matrix = self.matrix_type_info(left_type)
        right_matrix = self.matrix_type_info(right_type)
        left_vector = self.vector_type_info(left_type)
        right_vector = self.vector_type_info(right_type)

        if left_matrix is not None and right_vector is not None:
            if right_vector["size"] != left_matrix["columns"]:
                return None
            return self.build_matrix_vector_plan(
                left_matrix,
                right_vector,
                result_size=left_matrix["rows"],
                target_type=target_type,
                matrix_on_left=True,
            )

        if left_vector is not None and right_matrix is not None:
            if left_vector["size"] != right_matrix["rows"]:
                return None
            return self.build_matrix_vector_plan(
                right_matrix,
                left_vector,
                result_size=right_matrix["columns"],
                target_type=target_type,
                matrix_on_left=False,
            )

        return None

    def build_matrix_vector_plan(
        self, matrix_info, vector_info, result_size, target_type, matrix_on_left
    ):
        component_type = self.promoted_scalar_type(
            matrix_info["component_type"], vector_info["component_type"]
        )
        if component_type is None:
            return None

        operation_component_type = self.promoted_component_with_target(
            component_type, target_type, result_size
        )
        result_type = self.vector_type_for_components(
            operation_component_type, result_size
        )
        matrix_type = self.matrix_type_for_dimensions(
            operation_component_type, matrix_info["columns"], matrix_info["rows"]
        )
        vector_type = self.vector_type_for_components(
            operation_component_type, vector_info["size"]
        )
        if result_type is None or matrix_type is None or vector_type is None:
            return None

        if matrix_on_left:
            left_target_type = matrix_type
            right_target_type = vector_type
        else:
            left_target_type = vector_type
            right_target_type = matrix_type

        return {
            "result_type": result_type,
            "left_target_type": left_target_type,
            "right_target_type": right_target_type,
        }

    def promoted_component_with_target(self, component_type, target_type, result_size):
        target_info = self.vector_type_info(target_type)
        if target_info is None or target_info["size"] != result_size:
            return component_type

        target_component_type = self.normalize_scalar_type(
            target_info["component_type"]
        )
        promoted_type = self.promoted_scalar_type(component_type, target_component_type)
        if promoted_type == target_component_type:
            return promoted_type
        return component_type

    def promoted_vector_scalar_type(self, left_type, right_type):
        left_vector = self.vector_type_info(left_type)
        right_vector = self.vector_type_info(right_type)
        left_scalar = self.normalize_scalar_type(left_type)
        right_scalar = self.normalize_scalar_type(right_type)

        if left_vector is not None and right_scalar is not None:
            return self.vector_type_for_promoted_scalar(left_vector, right_scalar)
        if right_vector is not None and left_scalar is not None:
            return self.vector_type_for_promoted_scalar(right_vector, left_scalar)
        return None

    def vector_type_for_promoted_scalar(self, vector_info, scalar_type):
        component_type = self.promoted_scalar_type(
            vector_info["component_type"], scalar_type
        )
        if component_type is None:
            return None
        return self.vector_type_for_components(component_type, vector_info["size"])

    def promoted_matrix_scalar_type(self, left_type, right_type):
        left_matrix = self.matrix_type_info(left_type)
        right_matrix = self.matrix_type_info(right_type)
        left_scalar = self.normalize_scalar_type(left_type)
        right_scalar = self.normalize_scalar_type(right_type)

        if left_matrix is not None and right_scalar is not None:
            return self.matrix_type_for_promoted_scalar(left_matrix, right_scalar)
        if right_matrix is not None and left_scalar is not None:
            return self.matrix_type_for_promoted_scalar(right_matrix, left_scalar)
        return None

    def matrix_type_for_promoted_scalar(self, matrix_info, scalar_type):
        component_type = self.promoted_scalar_type(
            matrix_info["component_type"], scalar_type
        )
        if component_type is None:
            return None
        return self.matrix_type_for_dimensions(
            component_type, matrix_info["columns"], matrix_info["rows"]
        )

    def promoted_scalar_type(self, left_type, right_type):
        left = self.normalize_scalar_type(left_type)
        right = self.normalize_scalar_type(right_type)
        if left is None or right is None:
            return None

        ranks = {
            "bool": 0,
            "i16": 1,
            "u16": 2,
            "i32": 3,
            "u32": 4,
            "i64": 5,
            "u64": 6,
            "f16": 7,
            "f32": 8,
            "f64": 9,
        }
        return left if ranks[left] >= ranks[right] else right

    def promoted_vector_type(self, left_type, right_type):
        left_info = self.vector_type_info(left_type)
        right_info = self.vector_type_info(right_type)
        if left_info is None or right_info is None:
            return None
        if left_info["size"] != right_info["size"]:
            return None

        component_type = self.promoted_scalar_type(
            left_info["component_type"], right_info["component_type"]
        )
        if component_type is None:
            return None
        return self.vector_type_for_components(component_type, left_info["size"])

    def promoted_matrix_type(self, left_type, right_type):
        left_info = self.matrix_type_info(left_type)
        right_info = self.matrix_type_info(right_type)
        if left_info is None or right_info is None:
            return None
        if (
            left_info["columns"] != right_info["columns"]
            or left_info["rows"] != right_info["rows"]
        ):
            return None

        component_type = self.promoted_scalar_type(
            left_info["component_type"], right_info["component_type"]
        )
        if component_type is None:
            return None
        return self.matrix_type_for_dimensions(
            component_type, left_info["columns"], left_info["rows"]
        )

    def normalize_binary_scalar_operand(
        self, expr, generated_expr, source_type, target_type
    ):
        source_type = self.normalize_scalar_type(source_type)
        if source_type is None or source_type == target_type:
            return generated_expr

        if self.is_integer_literal_expression(expr):
            if target_type in {"f32", "f64"}:
                return f"{expr.value}.0"
            if target_type == "f16":
                return f"({generated_expr} as f16)"
            return generated_expr

        return f"({generated_expr} as {target_type})"

    def is_integer_literal_expression(self, expr):
        return (
            isinstance(expr, LiteralNode)
            and isinstance(expr.value, int)
            and not isinstance(expr.value, bool)
        )

    def normalize_scalar_type(self, type_name):
        if type_name is None:
            return None
        if hasattr(type_name, "name") or hasattr(type_name, "element_type"):
            type_name = self.convert_type_node_to_string(type_name)
        else:
            type_name = str(type_name)

        aliases = {
            "bool": "bool",
            "char": "i32",
            "short": "i16",
            "ushort": "u16",
            "int": "i32",
            "uint": "u32",
            "long": "i64",
            "ulong": "u64",
            "half": "f16",
            "float": "f32",
            "double": "f64",
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
        return aliases.get(type_name)

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
        if isinstance(expr, ArrayAccessNode):
            return self.array_access_element_type(expr)
        if isinstance(expr, (IdentifierNode, VariableNode)):
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
            if isinstance(func_name, str) and self.matrix_type_info(func_name):
                return func_name
            scalar_type = self.scalar_constructor_type(func_name)
            if scalar_type is not None:
                return scalar_type
            return_type = self.user_function_return_types.get(func_name)
            if return_type and return_type != "void":
                return return_type
            return None
        if isinstance(expr, BinaryOpNode):
            left_type = self.expression_result_type(expr.left)
            right_type = self.expression_result_type(expr.right)
            operator = self.map_operator(
                getattr(expr, "operator", getattr(expr, "op", None))
            )
            bool_vector_logical_plan = self.bool_vector_logical_plan(
                left_type, right_type, operator
            )
            if bool_vector_logical_plan is not None:
                return bool_vector_logical_plan["result_type"]
            vector_comparison_plan = self.vector_comparison_plan(
                left_type, right_type, operator
            )
            if vector_comparison_plan is not None:
                return vector_comparison_plan["result_type"]
            matrix_vector_plan = self.binary_matrix_vector_plan(
                left_type, right_type, operator
            )
            if matrix_vector_plan is not None:
                return matrix_vector_plan["result_type"]
            composite_type = self.binary_composite_operand_type(
                left_type, right_type, operator
            )
            if composite_type is not None:
                return composite_type
            if self.vector_type_info(left_type):
                return left_type
            if self.vector_type_info(right_type):
                return right_type
            if self.matrix_type_info(left_type):
                return left_type
            if self.matrix_type_info(right_type):
                return right_type
            scalar_type = self.binary_scalar_result_type(
                left_type,
                right_type,
                operator,
            )
            if scalar_type is not None:
                return scalar_type
            return left_type or right_type
        if isinstance(expr, UnaryOpNode):
            return self.expression_result_type(expr.operand)
        if isinstance(expr, TernaryOpNode):
            vector_ternary_plan = self.bool_vector_ternary_plan(
                expr.condition, expr.true_expr, expr.false_expr
            )
            if vector_ternary_plan is not None:
                return vector_ternary_plan["result_type"]
            true_type = self.expression_result_type(expr.true_expr)
            false_type = self.expression_result_type(expr.false_expr)
            vector_type = self.promoted_vector_type(true_type, false_type)
            if vector_type is not None:
                return vector_type
            if self.vector_type_info(true_type):
                return true_type
            if self.vector_type_info(false_type):
                return false_type
            matrix_type = self.promoted_matrix_type(true_type, false_type)
            if matrix_type is not None:
                return matrix_type
            if self.matrix_type_info(true_type):
                return true_type
            if self.matrix_type_info(false_type):
                return false_type
            scalar_type = self.promoted_scalar_type(true_type, false_type)
            if scalar_type is not None:
                return scalar_type
            return true_type or false_type
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

    def array_access_element_type(self, expr):
        array_name = self.get_expression_name(expr)
        array_type = self.variable_types.get(array_name)
        if array_type is None:
            return None
        if hasattr(array_type, "name") or hasattr(array_type, "element_type"):
            array_type = self.convert_type_node_to_string(array_type)
        else:
            array_type = str(array_type)
        if "[" not in array_type or "]" not in array_type:
            return None
        base_type, _ = parse_array_type(array_type)
        return base_type or None

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

    def matrix_type_info(self, type_name):
        if type_name is None:
            return None
        if hasattr(type_name, "name") or hasattr(type_name, "element_type"):
            type_name = self.convert_type_node_to_string(type_name)
        else:
            type_name = str(type_name)

        matrix_details = {
            "mat2": (2, 2),
            "mat3": (3, 3),
            "mat4": (4, 4),
            "mat2x2": (2, 2),
            "mat2x3": (2, 3),
            "mat2x4": (2, 4),
            "mat3x2": (3, 2),
            "mat3x3": (3, 3),
            "mat3x4": (3, 4),
            "mat4x2": (4, 2),
            "mat4x3": (4, 3),
            "mat4x4": (4, 4),
            "dmat2": (2, 2),
            "dmat3": (3, 3),
            "dmat4": (4, 4),
            "dmat2x2": (2, 2),
            "dmat2x3": (2, 3),
            "dmat2x4": (2, 4),
            "dmat3x2": (3, 2),
            "dmat3x3": (3, 3),
            "dmat3x4": (3, 4),
            "dmat4x2": (4, 2),
            "dmat4x3": (4, 3),
            "dmat4x4": (4, 4),
        }
        details = matrix_details.get(type_name)
        if details is None:
            return None
        columns, rows = details
        component_type = "double" if type_name.startswith("dmat") else "float"
        return {"columns": columns, "rows": rows, "component_type": component_type}

    def scalar_type_for_type_constructor(self, scalar_type):
        scalar_type = self.normalize_scalar_type(scalar_type)
        aliases = {
            "bool": "bool",
            "i16": "short",
            "u16": "ushort",
            "i32": "int",
            "u32": "uint",
            "i64": "long",
            "u64": "ulong",
            "f16": "half",
            "f32": "float",
            "f64": "double",
        }
        return aliases.get(scalar_type)

    def matrix_type_for_dimensions(self, component_type, columns, rows):
        component_type = self.scalar_type_for_type_constructor(component_type)
        if component_type == "double":
            prefix = "dmat"
        elif component_type == "float":
            prefix = "mat"
        else:
            return None

        if columns == rows:
            return f"{prefix}{columns}"
        return f"{prefix}{columns}x{rows}"

    def vector_type_for_components(self, component_type, component_count):
        component_type = self.scalar_type_for_type_constructor(component_type)
        if component_type is None:
            return None
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

"""Reverse code generator that emits CrossGL from Mojo AST nodes."""

import re

from .MojoAst import *
from .MojoLexer import *
from .MojoParser import *


class MojoToCrossGLConverter:
    """Serialize Mojo backend AST nodes back into CrossGL source."""

    VERTEX_ATTRIBUTES = {"vertex", "vertex_main", "vertex_shader"}
    FRAGMENT_ATTRIBUTES = {"fragment", "fragment_main", "fragment_shader"}
    COMPUTE_ATTRIBUTES = {"compute", "compute_main", "compute_shader"}
    SHADER_STAGE_ATTRIBUTES = (
        VERTEX_ATTRIBUTES | FRAGMENT_ATTRIBUTES | COMPUTE_ATTRIBUTES
    )
    IMPORT_MODULE_MAP = {
        "math": "math",
        "std.math": "math",
        "std.math.math": "math",
        "simd": "simd",
    }
    MATRIX_TYPE_PATTERN = re.compile(r"^Matrix\[(DType\.\w+),\s*(\d+),\s*(\d+)\]$")
    REFERENCE_TYPE_PATTERN = re.compile(r"^ref\[[^\]]*\]\s+(.+)$")
    MLIR_BACKTICK_TYPE_PATTERN = re.compile(r"^__mlir_type\.`([^`]+)`$")
    BACKTICK_IDENTIFIER_PATTERN = re.compile(r"^`([^`]+)`$")
    STRING_LITERAL_PREFIXES = {
        "r",
        "R",
        "t",
        "T",
        "rt",
        "rT",
        "Rt",
        "RT",
        "tr",
        "tR",
        "Tr",
        "TR",
    }
    MATRIX_DTYPE_PREFIXES = {
        "DType.float16": "half",
        "DType.float32": "mat",
        "DType.float64": "dmat",
    }
    MLIR_TYPE_MAP = {
        "!kgen.none": "void",
        "!kgen.string": "String",
        "bf16": "half",
        "f16": "half",
        "f32": "float",
        "f64": "double",
        "i1": "bool",
        "i8": "int8_t",
        "i16": "int16_t",
        "i32": "int",
        "i64": "int64_t",
        "index": "int",
    }
    CROSSGL_RESERVED_IDENTIFIERS = {
        "as",
        "async",
        "await",
        "bool",
        "box",
        "break",
        "buffer",
        "case",
        "cbuffer",
        "char",
        "class",
        "compute",
        "const",
        "continue",
        "default",
        "do",
        "double",
        "elif",
        "else",
        "enum",
        "extern",
        "float",
        "for",
        "fragment",
        "from",
        "fn",
        "geometry",
        "global",
        "half",
        "if",
        "impl",
        "import",
        "in",
        "interface",
        "internal",
        "kernel",
        "layout",
        "let",
        "local",
        "loop",
        "match",
        "module",
        "move",
        "mut",
        "namespace",
        "priv",
        "protected",
        "pub",
        "ref",
        "return",
        "safe",
        "shader",
        "shared",
        "static",
        "string",
        "struct",
        "switch",
        "tessellation",
        "threadgroup",
        "trait",
        "uniform",
        "unsafe",
        "use",
        "var",
        "vertex",
        "void",
        "while",
        "workgroup",
        "yield",
    }

    def __init__(self):
        self.user_function_names = set()
        self.user_function_arities = {}
        self.imported_module_aliases = {}
        self.imported_function_aliases = {}
        self.scoped_value_names = []
        self.function_body_depth = 0
        self.named_result_stack = []
        self.loop_else_guard_counter = 0
        self.loop_else_guard_stack = []
        self.type_map = {
            # Scalar Types
            "void": "void",
            "None": "void",
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
            "SIMD[DType.float64, 2]": "dvec2",
            "SIMD[DType.float64, 3]": "dvec3",
            "SIMD[DType.float64, 4]": "dvec4",
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
            "math.exp": "exp",
            "math.fma": "fma",
            "math.fract": "fract",
            "math.log": "log",
            "math.fmod": "mod",
            "fmod": "mod",
            "power": "pow",
            "rsqrt": "inversesqrt",
            "math.rsqrt": "inversesqrt",
            "lerp": "mix",
            "math.lerp": "mix",
            "math.min": "min",
            "math.max": "max",
            "math.clamp": "clamp",
            # SIMD functions
            "dot_product": "dot",
            "cross_product": "cross",
            "magnitude": "length",
            "simd.dot": "dot",
            "simd.cross": "cross",
            "simd.normalize": "normalize",
            "simd.length": "length",
            "simd.reflect": "reflect",
            "simd.refract": "refract",
        }

    def generate(self, ast):
        self.user_function_arities = self.collect_user_function_arities(ast)
        self.user_function_names = set(self.user_function_arities)
        self.imported_module_aliases = self.collect_imported_module_aliases(ast)
        self.imported_function_aliases = self.collect_imported_function_aliases(
            getattr(ast, "includes", []) or []
        )
        self.scoped_value_names = []
        global_value_names = [
            getattr(variable, "name", None)
            for variable in getattr(ast, "global_variables", []) or []
            if isinstance(variable, (VariableDeclarationNode, VariableNode))
        ]
        self.push_value_scope(global_value_names)
        try:
            return self.generate_shader(ast)
        finally:
            self.pop_value_scope()

    def generate_shader(self, ast):
        code = "shader main {\n"

        if hasattr(ast, "functions") and ast.functions:
            imports = [f for f in ast.functions if isinstance(f, ImportNode)]
            if imports:
                code += "    // Imports\n"
                for imp in imports:
                    code += f"    {self.generate_import_comment(imp)}\n"
                code += "\n"

        if hasattr(ast, "functions"):
            structs = [f for f in ast.functions if isinstance(f, StructNode)]
            for struct_node in structs:
                code += self.generate_struct_like_declaration(
                    struct_node.name,
                    struct_node.members,
                )

            classes = [f for f in ast.functions if isinstance(f, ClassNode)]
            for class_node in classes:
                code += self.generate_struct_like_declaration(
                    class_node.name,
                    class_node.members,
                )

        if hasattr(ast, "functions"):
            cbuffers = [f for f in ast.functions if isinstance(f, ConstantBufferNode)]
            if cbuffers:
                code += "    // Constant Buffers\n"
                for cbuffer in cbuffers:
                    code += self.generate_constant_buffer(cbuffer)

        globals_code = self.generate_global_variables(ast)
        if globals_code:
            code += globals_code

        if hasattr(ast, "functions"):
            functions = [f for f in ast.functions if isinstance(f, FunctionNode)]
            for struct_node in [f for f in ast.functions if isinstance(f, StructNode)]:
                functions.extend(getattr(struct_node, "methods", []))
            for class_node in [f for f in ast.functions if isinstance(f, ClassNode)]:
                functions.extend(getattr(class_node, "methods", []))
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

    def collect_user_function_arities(self, ast):
        arities = {}

        def record(function):
            if not isinstance(function, FunctionNode) or function.name is None:
                return
            arities.setdefault(function.name, set()).add(
                len(getattr(function, "params", []) or [])
            )

        for node in getattr(ast, "functions", []):
            if isinstance(node, FunctionNode):
                record(node)
            elif isinstance(node, StructNode):
                for method in getattr(node, "methods", []):
                    record(method)
            elif isinstance(node, ClassNode):
                for method in getattr(node, "methods", []):
                    record(method)
        return arities

    def is_user_defined_function(self, func_name, arg_count=None):
        if not isinstance(func_name, str):
            return False
        if arg_count is None:
            return func_name in self.user_function_arities
        return arg_count in self.user_function_arities.get(func_name, set())

    def collect_imported_module_aliases(self, ast):
        aliases = {}
        for import_node in getattr(ast, "includes", []) or []:
            if getattr(import_node, "items", None):
                continue

            module_name = getattr(
                import_node, "module_name", getattr(import_node, "module", None)
            )
            canonical_module = self.normalize_import_module_name(module_name)
            if canonical_module is None:
                continue

            alias = getattr(import_node, "alias", None)
            if alias:
                aliases[self.map_identifier_name(alias)] = canonical_module
            else:
                aliases[self.map_identifier_name(module_name)] = canonical_module
        return aliases

    def collect_imported_function_aliases(self, imports):
        aliases = {}
        for import_node in imports:
            aliases.update(self.imported_function_aliases_from_node(import_node))
        return aliases

    def imported_function_aliases_from_node(self, import_node):
        aliases = {}
        if not getattr(import_node, "items", None):
            return aliases

        module_name = getattr(
            import_node, "module_name", getattr(import_node, "module", None)
        )
        canonical_module = self.normalize_import_module_name(module_name)
        if canonical_module is None:
            return aliases

        for item in import_node.items:
            import_name, alias = self.split_import_item_alias(item)
            if not import_name or import_name == "*":
                continue

            local_name = alias or import_name
            resolved_name = f"{canonical_module}.{import_name}"
            if resolved_name in self.function_map:
                aliases[self.map_identifier_name(local_name)] = resolved_name
        return aliases

    def normalize_import_module_name(self, module_name):
        if not isinstance(module_name, str):
            return None
        return self.IMPORT_MODULE_MAP.get(module_name)

    def split_import_item_alias(self, item):
        if not isinstance(item, str):
            return item, None

        parts = item.split(" as ", 1)
        if len(parts) == 2:
            return parts[0].strip(), parts[1].strip()
        return item.strip(), None

    def resolve_imported_function_name(self, func_name):
        if isinstance(func_name, str):
            alias = self.imported_function_aliases.get(
                self.map_identifier_name(func_name)
            )
            if alias:
                return alias

        if not isinstance(func_name, str) or "." not in func_name:
            return func_name

        for qualifier, module_name in sorted(
            self.imported_module_aliases.items(),
            key=lambda item: len(item[0]),
            reverse=True,
        ):
            prefix = f"{qualifier}."
            if func_name.startswith(prefix):
                return f"{module_name}.{func_name[len(prefix):]}"

        qualifier, member = func_name.split(".", 1)
        module_name = self.imported_module_aliases.get(qualifier)
        if not module_name:
            return func_name
        return f"{module_name}.{member}"

    def push_value_scope(self, names=None):
        scope = set()
        if names:
            scope.update(self.map_identifier_name(name) for name in names if name)
        self.scoped_value_names.append(scope)

    def pop_value_scope(self):
        if self.scoped_value_names:
            self.scoped_value_names.pop()

    def add_scoped_value_name(self, name):
        if name and self.scoped_value_names:
            self.scoped_value_names[-1].add(self.map_identifier_name(name))

    def is_scoped_value_name(self, name):
        name = self.map_identifier_name(name)
        return any(name in scope for scope in reversed(self.scoped_value_names))

    def generate_global_variables(self, ast):
        global_variables = getattr(ast, "global_variables", []) or []
        if not global_variables:
            return ""

        code = "    // Global Variables\n"
        for variable in global_variables:
            if not isinstance(variable, VariableDeclarationNode):
                continue
            code += f"    {self.generate_variable_declaration(variable)};\n"
        return code + "\n"

    def generate_struct_like_declaration(self, name, members):
        code = f"    struct {name} {{\n"
        for member in members:
            if not isinstance(member, (VariableDeclarationNode, VariableNode)):
                continue
            if getattr(member, "is_comptime", False) or getattr(
                member, "is_alias", False
            ):
                continue

            semantic = (
                self.map_attributes(member.attributes)
                if hasattr(member, "attributes")
                else ""
            )
            semantic_suffix = f" {semantic}" if semantic else ""
            code += (
                f"        {self.map_type(member.vtype)} "
                f"{self.map_identifier_name(member.name)}{semantic_suffix};\n"
            )
        code += "    }\n\n"
        return code

    def generate_constant_buffer(self, cbuffer):
        attributes = self.map_attributes(getattr(cbuffer, "attributes", []))
        attributes_suffix = f" {attributes}" if attributes else ""
        code = f"    cbuffer {cbuffer.name}{attributes_suffix} {{\n"
        for member in cbuffer.members:
            member_attributes = self.map_attributes(getattr(member, "attributes", []))
            member_suffix = f" {member_attributes}" if member_attributes else ""
            code += (
                f"        {self.map_type(member.vtype)} "
                f"{self.map_identifier_name(member.name)}{member_suffix};\n"
            )
        code += "    }\n\n"
        return code

    def has_vertex_attribute(self, func):
        if not hasattr(func, "attributes") or not func.attributes:
            return False
        for attr in func.attributes:
            if hasattr(attr, "name") and attr.name in self.VERTEX_ATTRIBUTES:
                return True
        return False

    def has_fragment_attribute(self, func):
        if not hasattr(func, "attributes") or not func.attributes:
            return False
        for attr in func.attributes:
            if hasattr(attr, "name") and attr.name in self.FRAGMENT_ATTRIBUTES:
                return True
        return False

    def has_compute_attribute(self, func):
        if not hasattr(func, "attributes") or not func.attributes:
            return False
        for attr in func.attributes:
            if hasattr(attr, "name") and attr.name in self.COMPUTE_ATTRIBUTES:
                return True
        return False

    def generate_function(self, func, indent=1):
        """Render one Mojo function node as a CrossGL function."""
        code = ""
        indent_str = "    " * indent
        named_result = self.get_named_result_parameter(func)
        named_result_name = (
            self.map_identifier_name(named_result.name) if named_result else None
        )

        params = []
        param_names = []
        if hasattr(func, "params") and func.params:
            for p in func.params:
                if p is named_result:
                    continue
                if not p.vtype:
                    continue
                param_name = self.map_identifier_name(p.name)
                param_str = f"{self.map_type(p.vtype)} {param_name}"
                if getattr(p, "default_value", None) is not None:
                    default_value = self.generate_expression(p.default_value)
                    param_str += f" = {default_value}"
                if hasattr(p, "attributes") and p.attributes:
                    semantic = self.map_semantic(p.attributes)
                    if semantic:
                        param_str += f" {semantic}"
                params.append(param_str)
                param_names.append(param_name)

        params_str = ", ".join(params) if params else ""
        if func.return_type:
            return_type = self.map_type(func.return_type)
        elif named_result:
            return_type = self.map_type(named_result.vtype)
        else:
            return_type = "void"

        func_attributes = self.map_function_attributes(func)
        func_name = self.map_identifier_name(func.name)

        code += (
            f"{indent_str}{return_type} {func_name}({params_str}){func_attributes} {{\n"
        )
        if named_result:
            result_type = self.map_type(named_result.vtype)
            code += f"{indent_str}    {result_type} {named_result_name};\n"

        saved_body_depth = self.function_body_depth
        self.function_body_depth = 0
        saved_loop_else_guard_stack = self.loop_else_guard_stack
        self.loop_else_guard_stack = []
        scope_names = list(param_names)
        if named_result:
            scope_names.append(named_result.name)
        self.push_value_scope(scope_names)
        self.named_result_stack.append(named_result_name)
        try:
            if hasattr(func, "body") and func.body:
                code += self.generate_function_body(func.body, indent + 1)
            if named_result and self.needs_named_result_fallthrough_return(func.body):
                code += f"{indent_str}    return {named_result_name};\n"
        finally:
            self.named_result_stack.pop()
            self.pop_value_scope()
            self.loop_else_guard_stack = saved_loop_else_guard_stack
            self.function_body_depth = saved_body_depth

        code += f"{indent_str}}}\n\n"
        return code

    def get_named_result_parameter(self, func):
        if getattr(func, "return_type", None):
            return None

        candidates = [
            param
            for param in getattr(func, "params", []) or []
            if getattr(param, "parameter_convention", None) == "out"
            and getattr(param, "name", None) != "self"
            and getattr(param, "vtype", None)
        ]
        if len(candidates) != 1:
            return None
        return candidates[0]

    def current_named_result(self):
        if not self.named_result_stack:
            return None
        return self.named_result_stack[-1]

    def needs_named_result_fallthrough_return(self, body):
        if not body:
            return True
        return not isinstance(body[-1], ReturnNode)

    def generate_function_body(self, body, indent=0):
        code = ""
        indent_str = "    " * indent

        self.function_body_depth += 1
        self.push_value_scope()
        try:
            for stmt in body:
                if isinstance(stmt, PassNode):
                    continue

                if isinstance(stmt, TraitNode):
                    continue

                if isinstance(stmt, FunctionNode):
                    code += self.generate_function(stmt, indent)
                    self.add_scoped_value_name(stmt.name)
                    continue

                code += indent_str
                if isinstance(stmt, VariableDeclarationNode):
                    code += self.generate_variable_declaration(stmt) + ";\n"
                    self.add_scoped_declaration_names(stmt.name)
                elif isinstance(stmt, VariableNode):
                    name = self.map_identifier_name(stmt.name)
                    if hasattr(stmt, "vtype") and stmt.vtype:
                        code += f"{self.map_type(stmt.vtype)} {name};\n"
                    else:
                        code += f"{name};\n"
                    self.add_scoped_value_name(stmt.name)
                elif isinstance(stmt, AssignmentNode):
                    if self.is_implicit_assignment_declaration(stmt):
                        code += (
                            self.generate_implicit_assignment_declaration(stmt) + ";\n"
                        )
                        self.add_scoped_value_name(stmt.left.name)
                    else:
                        code += self.generate_assignment(stmt) + ";\n"
                elif isinstance(stmt, ReturnNode):
                    if stmt.value is None or self.is_none_literal(stmt.value):
                        named_result = self.current_named_result()
                        if named_result:
                            code += f"return {named_result};\n"
                        else:
                            code += "return;\n"
                    else:
                        code += f"return {self.generate_expression(stmt.value)};\n"
                elif isinstance(stmt, BreakNode):
                    loop_else_guard = self.current_loop_else_guard()
                    if loop_else_guard:
                        code += f"{loop_else_guard} = false;\n"
                        code += f"{indent_str}break;\n"
                    else:
                        code += "break;\n"
                elif isinstance(stmt, ContinueNode):
                    code += "continue;\n"
                elif isinstance(stmt, BinaryOpNode):
                    code += f"{self.generate_expression(stmt)};\n"
                elif isinstance(stmt, ForNode):
                    code += self.generate_for_loop(stmt, indent)
                elif isinstance(stmt, RangeForNode):
                    code += self.generate_range_for_loop(stmt, indent)
                elif isinstance(stmt, WhileNode):
                    code += self.generate_while_loop(stmt, indent)
                elif isinstance(stmt, WithNode):
                    code += self.generate_with_block(stmt, indent)
                elif isinstance(stmt, TryExceptNode):
                    code += self.generate_try_except_block(stmt, indent)
                elif isinstance(stmt, ImportNode):
                    code += f"{self.generate_import_comment(stmt)}\n"
                elif isinstance(stmt, IfNode):
                    code += self.generate_if_statement(stmt, indent)
                elif isinstance(stmt, SwitchNode):
                    code += self.generate_switch_statement(stmt, indent)
                elif isinstance(
                    stmt,
                    (FunctionCallNode, MethodCallNode, CallNode, VectorConstructorNode),
                ):
                    code += f"{self.generate_expression(stmt)};\n"
                elif isinstance(stmt, str):
                    code += f"{stmt};\n"
                else:
                    # For any unhandled statement type
                    code += f"// Unhandled statement type: {type(stmt).__name__}\n"
        finally:
            self.pop_value_scope()
            self.function_body_depth -= 1

        return code

    def is_implicit_assignment_declaration(self, node):
        if self.function_body_depth != 1:
            return False
        if getattr(node, "operator", "=") != "=":
            return False
        if not isinstance(node.left, VariableNode):
            return False
        if getattr(node.left, "vtype", ""):
            return False
        return node.left.name != "_" and not self.is_scoped_value_name(node.left.name)

    def is_none_literal(self, node):
        return (
            isinstance(node, VariableNode)
            and node.name == "None"
            and not getattr(node, "vtype", "")
        )

    def generate_implicit_assignment_declaration(self, node):
        value = self.generate_expression(node.right)
        return f"var {self.map_identifier_name(node.left.name)} = {value}"

    def generate_import_comment(self, node):
        if getattr(node, "items", None):
            items = ", ".join(node.items)
            return f"// from {node.module_name} import {items}"
        comment = f"// import {node.module_name}"
        if node.alias:
            comment += f" as {node.alias}"
        return comment

    def generate_variable_declaration(self, node):
        name = self.generate_declaration_name(node.name)
        if node.vtype:
            declaration = f"{self.map_type(node.vtype)} {name}"
        else:
            var_type = "var" if node.var_type == "var" else "let"
            declaration = f"{var_type} {name}"

        attributes = self.map_attributes(getattr(node, "attributes", []))
        if attributes:
            declaration += f" {attributes}"

        if hasattr(node, "initial_value") and node.initial_value is not None:
            value = self.generate_expression(node.initial_value)
            declaration += f" = {value}"
        return declaration

    def generate_declaration_name(self, name):
        if isinstance(name, TupleNode):
            return self.generate_expression(name)
        return self.normalize_type_expression_operators(self.map_identifier_name(name))

    def add_scoped_declaration_names(self, name):
        if isinstance(name, TupleNode):
            for element in name.elements:
                if isinstance(element, VariableNode):
                    self.add_scoped_value_name(element.name)
            return
        self.add_scoped_value_name(name)

    def generate_assignment(self, node):
        left = self.generate_expression(node.left)
        right = self.generate_expression(node.right)
        op = self.map_operator(node.operator if hasattr(node, "operator") else "=")
        return f"{left} {op} {right}"

    def generate_nested_expression(self, expr):
        rendered = self.generate_expression(expr)
        if isinstance(expr, AssignmentNode):
            return f"({rendered})"
        return rendered

    def generate_for_loop(self, node, indent):
        indent_str = "    " * indent
        init = self.generate_for_initializer(node.init) if node.init else ""
        condition = (
            self.generate_expression(node.condition) if node.condition else "true"
        )
        update = self.generate_expression(node.update) if node.update else ""
        else_body = getattr(node, "else_body", []) or []
        loop_else_guard = self.begin_loop_else_guard(else_body)

        code = self.generate_loop_else_guard_declaration(loop_else_guard, indent)
        code += f"for ({init}; {condition}; {update}) {{\n"
        try:
            if hasattr(node, "body") and node.body:
                code += self.generate_function_body(node.body, indent + 1)
        finally:
            self.end_loop_else_guard(loop_else_guard)
        code += indent_str + "}\n"
        code += self.generate_loop_else_block(loop_else_guard, else_body, indent)
        return code

    def generate_with_block(self, node, indent):
        indent_str = "    " * indent
        contexts = getattr(node, "contexts", None) or [(node.context_expr, node.alias)]
        context_items = []
        for context_expr, alias in contexts:
            context = self.generate_expression(context_expr)
            if alias:
                context += f" as {alias}"
            context_items.append(context)
        code = f"{{ // with {', '.join(context_items)}\n"
        if getattr(node, "body", None):
            code += self.generate_function_body(node.body, indent + 1)
        code += indent_str + "}\n"
        return code

    def generate_try_except_block(self, node, indent):
        indent_str = "    " * indent
        code = "try {\n"
        if getattr(node, "try_body", None):
            code += self.generate_function_body(node.try_body, indent + 1)
        code += indent_str + "}"
        if getattr(node, "except_body", None):
            exception_name = f" ({node.exception_name})" if node.exception_name else ""
            code += f" catch{exception_name} {{\n"
            code += self.generate_function_body(node.except_body, indent + 1)
            code += indent_str + "}"
        if getattr(node, "else_body", None):
            code += " else {\n"
            code += self.generate_function_body(node.else_body, indent + 1)
            code += indent_str + "}"
        if getattr(node, "finally_body", None):
            code += " finally {\n"
            code += self.generate_function_body(node.finally_body, indent + 1)
            code += indent_str + "}"
        code += "\n"
        return code

    def generate_for_initializer(self, node):
        if isinstance(node, VariableDeclarationNode):
            name = self.generate_declaration_name(node.name)
            if node.vtype:
                declaration = f"{self.map_type(node.vtype)} {name}"
            else:
                declaration = f"{node.var_type} {name}"
            if getattr(node, "initial_value", None) is not None:
                value = self.generate_expression(node.initial_value)
                declaration += f" = {value}"
            return declaration
        return self.generate_expression(node)

    def generate_range_for_loop(self, node, indent):
        indent_str = "    " * indent
        target = self.generate_declaration_name(node.name)
        else_body = getattr(node, "else_body", []) or []
        loop_else_guard = self.begin_loop_else_guard(else_body)

        if isinstance(node.name, str) and self.is_range_call(node.iterable):
            code = self.generate_loop_else_guard_declaration(loop_else_guard, indent)
            code += self.generate_range_call_for_loop(node)
        else:
            iterable = self.generate_expression(node.iterable)
            code = self.generate_loop_else_guard_declaration(loop_else_guard, indent)
            code += f"for {target} in {iterable} {{\n"

        try:
            if hasattr(node, "body") and node.body:
                code += self.generate_function_body(node.body, indent + 1)
        finally:
            self.end_loop_else_guard(loop_else_guard)
        code += indent_str + "}\n"
        code += self.generate_loop_else_block(loop_else_guard, else_body, indent)
        return code

    def is_range_call(self, node):
        return isinstance(node, FunctionCallNode) and node.name == "range"

    def generate_range_call_for_loop(self, node):
        args = getattr(node.iterable, "args", [])
        if len(args) == 1:
            start = "0"
            stop = self.generate_expression(args[0])
            step = "1"
        elif len(args) == 2:
            start = self.generate_expression(args[0])
            stop = self.generate_expression(args[1])
            step = "1"
        elif len(args) == 3:
            start = self.generate_expression(args[0])
            stop = self.generate_expression(args[1])
            step = self.generate_expression(args[2])
        else:
            iterable = self.generate_expression(node.iterable)
            target = self.generate_declaration_name(node.name)
            return f"for {target} in {iterable} {{\n"

        condition = self.generate_range_condition(node.name, stop, args, step)
        target = self.map_identifier_name(node.name)
        update = f"{target}++" if step == "1" else f"{target} += {step}"
        return f"for (int {target} = {start}; {condition}; {update}) {{\n"

    def generate_range_condition(self, name, stop, args, generated_step):
        name = self.map_identifier_name(name)
        if self.is_negative_range_step(args, generated_step):
            return f"{name} > {stop}"
        if self.has_dynamic_range_step(args, generated_step):
            return (
                f"(({generated_step} > 0) ? " f"({name} < {stop}) : ({name} > {stop}))"
            )
        return f"{name} < {stop}"

    def has_dynamic_range_step(self, args, generated_step):
        return len(args) == 3 and not self.is_numeric_range_step(args, generated_step)

    def is_negative_range_step(self, args, generated_step):
        if len(args) != 3:
            return False

        step = args[2]
        if isinstance(step, UnaryOpNode) and step.op == "-":
            return self.is_numeric_literal(step.operand)
        return self.is_negative_numeric_literal(generated_step)

    def is_numeric_range_step(self, args, generated_step):
        if len(args) != 3:
            return True

        step = args[2]
        if isinstance(step, UnaryOpNode) and step.op == "-":
            return self.is_numeric_literal(step.operand)
        return self.is_numeric_literal(
            generated_step
        ) or self.is_negative_numeric_literal(generated_step)

    def is_numeric_literal(self, value):
        if isinstance(value, (int, float)):
            return True
        if not isinstance(value, str):
            return False
        try:
            float(value.rstrip("fF"))
        except ValueError:
            return False
        return True

    def is_negative_numeric_literal(self, value):
        if isinstance(value, (int, float)):
            return value < 0
        if not isinstance(value, str):
            return False
        normalized = value.strip()
        if normalized.startswith("(") and normalized.endswith(")"):
            normalized = normalized[1:-1].strip()
        if not normalized.startswith("-"):
            return False
        return self.is_numeric_literal(normalized[1:])

    def generate_while_loop(self, node, indent):
        indent_str = "    " * indent
        condition = (
            self.generate_expression(node.condition) if node.condition else "true"
        )
        else_body = getattr(node, "else_body", []) or []
        loop_else_guard = self.begin_loop_else_guard(else_body)

        code = self.generate_loop_else_guard_declaration(loop_else_guard, indent)
        code += f"while ({condition}) {{\n"
        try:
            if hasattr(node, "body") and node.body:
                code += self.generate_function_body(node.body, indent + 1)
        finally:
            self.end_loop_else_guard(loop_else_guard)
        code += indent_str + "}\n"
        code += self.generate_loop_else_block(loop_else_guard, else_body, indent)
        return code

    def begin_loop_else_guard(self, else_body):
        guard = None
        if else_body:
            guard = f"__mojo_loop_completed_{self.loop_else_guard_counter}"
            self.loop_else_guard_counter += 1
        self.loop_else_guard_stack.append(guard)
        return guard

    def end_loop_else_guard(self, guard):
        self.loop_else_guard_stack.pop()

    def current_loop_else_guard(self):
        if not self.loop_else_guard_stack:
            return None
        return self.loop_else_guard_stack[-1]

    def generate_loop_else_guard_declaration(self, guard, indent):
        if guard is None:
            return ""
        return f"bool {guard} = true;\n{'    ' * indent}"

    def generate_loop_else_block(self, guard, else_body, indent):
        if guard is None:
            return ""

        indent_str = "    " * indent
        code = f"{indent_str}if ({guard}) {{\n"
        code += self.generate_function_body(else_body, indent + 1)
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
            if isinstance(node.else_body, list):
                if len(node.else_body) == 1 and isinstance(node.else_body[0], IfNode):
                    code += " else "
                    code += self.generate_if_statement(node.else_body[0], indent)
                else:
                    code += " else {\n"
                    code += self.generate_function_body(node.else_body, indent + 1)
                    code += indent_str + "}"
            elif isinstance(node.else_body, IfNode):
                code += " else "
                code += self.generate_if_statement(node.else_body, indent)
            else:
                code += " else {\n"
                code += self.generate_function_body([node.else_body], indent + 1)
                code += indent_str + "}"

        code += "\n"
        return code

    def generate_switch_statement(self, node, indent):
        indent_str = "    " * indent
        expression = self.generate_expression(node.expression)

        code = f"switch ({expression}) {{\n"

        self.loop_else_guard_stack.append(None)
        try:
            if hasattr(node, "cases") and node.cases:
                for case in node.cases:
                    if hasattr(case, "condition") and case.condition is not None:
                        case_value = self.generate_expression(case.condition)
                        code += indent_str + f"    case {case_value}:\n"
                    else:
                        code += indent_str + "    default:\n"

                    if hasattr(case, "body") and case.body:
                        code += self.generate_function_body(case.body, indent + 2)
        finally:
            self.loop_else_guard_stack.pop()

        code += indent_str + "}\n"
        return code

    def generate_expression(self, expr):
        """Render a Mojo backend expression node as CrossGL syntax."""
        if expr is None:
            return ""
        elif isinstance(expr, str):
            return self.normalize_string_literal(expr)
        elif isinstance(expr, (int, float, bool)):
            return str(expr)
        elif isinstance(expr, VariableNode):
            name = self.map_identifier_name(expr.name)
            if hasattr(expr, "vtype") and expr.vtype:
                return f"{self.map_type(expr.vtype)} {name}"
            else:
                return name
        elif isinstance(expr, VariableDeclarationNode):
            return self.generate_variable_declaration(expr)
        elif isinstance(expr, TupleNode):
            elements = ", ".join(self.generate_expression(e) for e in expr.elements)
            return f"({elements})"
        elif isinstance(expr, ListLiteralNode):
            elements = ", ".join(self.generate_expression(e) for e in expr.elements)
            return f"[{elements}]"
        elif isinstance(expr, ListComprehensionNode):
            return self.generate_list_comprehension(expr)
        elif isinstance(expr, DictLiteralNode):
            return self.generate_dict_literal(expr)
        elif isinstance(expr, BracedLiteralNode):
            elements = ", ".join(
                self.generate_expression(element) for element in expr.elements
            )
            return f"{{{elements}}}"
        elif isinstance(expr, SetComprehensionNode):
            return self.generate_set_comprehension(expr)
        elif isinstance(expr, DictComprehensionNode):
            return self.generate_dict_comprehension(expr)
        elif isinstance(expr, SliceNode):
            return self.generate_slice_index(expr)
        elif isinstance(expr, SpreadExpressionNode):
            return self.generate_expression(expr.expression)
        elif isinstance(expr, AssignmentNode):
            return self.generate_assignment(expr)
        elif isinstance(expr, BinaryOpNode):
            left = self.generate_nested_expression(expr.left)
            right = self.generate_nested_expression(expr.right)
            op = expr.op if hasattr(expr, "op") else "+"
            if op == "not in":
                return f"(!({left} in {right}))"
            op = self.map_operator(op)
            return f"({left} {op} {right})"
        elif isinstance(expr, UnaryOpNode):
            operand = self.generate_nested_expression(expr.operand)
            op = expr.op if hasattr(expr, "op") else "+"
            if op == "await":
                return operand
            return f"({op}{operand})"
        elif isinstance(expr, CastNode):
            target_type = self.map_type(expr.target_type)
            value = self.generate_expression(expr.expression)
            return f"{target_type}({value})"
        elif isinstance(expr, FunctionCallNode):
            args = []
            if hasattr(expr, "args") and expr.args:
                args = [self.generate_expression(arg) for arg in expr.args]
            args_str = ", ".join(args)
            func_name = self.map_function(expr.name, len(args))
            return f"{func_name}({args_str})"
        elif isinstance(expr, MethodCallNode):
            args = []
            if hasattr(expr, "args") and expr.args:
                args = [self.generate_expression(arg) for arg in expr.args]
            args_str = ", ".join(args)
            if expr.method == "splat":
                simd_type = self.map_simd_type_expression(expr.object)
                if simd_type and len(args) == 1:
                    return f"{simd_type}({args_str})"

            obj = self.generate_expression(expr.object)
            method_name = self.map_member_name(expr.method)
            full_name = f"{obj}.{method_name}"
            if self.is_scoped_value_name(obj):
                return f"{obj}.{method_name}({args_str})"
            func_name = self.map_function(full_name, len(args))
            if func_name != full_name:
                return f"{func_name}({args_str})"
            return f"{obj}.{method_name}({args_str})"
        elif isinstance(expr, CallNode):
            callee = self.generate_expression(expr.callee)
            args = []
            if hasattr(expr, "args") and expr.args:
                args = [self.generate_expression(arg) for arg in expr.args]
            args_str = ", ".join(args)
            return f"{callee}({args_str})"
        elif isinstance(expr, MemberAccessNode):
            obj = self.generate_expression(expr.object)
            return f"{obj}.{self.map_member_name(expr.member)}"
        elif isinstance(expr, TernaryOpNode):
            condition = self.generate_nested_expression(expr.condition)
            true_expr = self.generate_nested_expression(expr.true_expr)
            false_expr = self.generate_nested_expression(expr.false_expr)
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
            index = self.generate_array_index(expr.index)
            return f"{array}[{index}]"
        else:
            # For any unhandled expression type
            return f"/* Unhandled expression: {type(expr).__name__} */"

    def generate_array_index(self, index):
        if isinstance(index, SliceNode):
            return self.generate_slice_index(index)
        if isinstance(index, TupleNode):
            return ", ".join(
                self.generate_expression(element) for element in index.elements
            )
        return self.generate_expression(index)

    def map_simd_type_expression(self, node):
        if not isinstance(node, ArrayAccessNode):
            return None
        if not isinstance(node.array, VariableNode) or node.array.name != "SIMD":
            return None

        if isinstance(node.index, TupleNode):
            elements = node.index.elements
        else:
            elements = [node.index]
        if len(elements) != 2:
            return None

        dtype = self.generate_expression(elements[0])
        width = self.generate_expression(elements[1])
        simd_type = f"SIMD[{dtype}, {width}]"
        mapped_type = self.map_type(simd_type)
        if mapped_type == simd_type:
            return None
        return mapped_type

    def generate_list_comprehension(self, expr):
        pieces = [self.generate_expression(expr.expression)]
        pieces.extend(self.generate_comprehension_clauses(expr.clauses))
        return f"[{' '.join(pieces)}]"

    def generate_set_comprehension(self, expr):
        pieces = [self.generate_expression(expr.expression)]
        pieces.extend(self.generate_comprehension_clauses(expr.clauses))
        return f"{{{' '.join(pieces)}}}"

    def generate_dict_literal(self, expr):
        entries = [
            f"{self.generate_expression(key)}: {self.generate_expression(value)}"
            for key, value in expr.entries
        ]
        return f"{{{', '.join(entries)}}}"

    def generate_dict_comprehension(self, expr):
        key = self.generate_expression(expr.key)
        value = self.generate_expression(expr.value)
        pieces = [f"{key}: {value}"]
        pieces.extend(self.generate_comprehension_clauses(expr.clauses))
        return f"{{{' '.join(pieces)}}}"

    def generate_comprehension_clauses(self, clauses):
        pieces = []
        for clause in clauses:
            if clause["kind"] == "for":
                pattern = self.generate_comprehension_pattern(clause["pattern"])
                iterable = self.generate_expression(clause["iterable"])
                pieces.append(f"for {pattern} in {iterable}")
            elif clause["kind"] == "if":
                condition = self.generate_expression(clause["condition"])
                pieces.append(f"if {condition}")
        return pieces

    def generate_comprehension_pattern(self, pattern):
        if isinstance(pattern, TupleNode):
            return ", ".join(
                self.generate_expression(element) for element in pattern.elements
            )
        return self.generate_expression(pattern)

    def generate_slice_index(self, index):
        start = self.generate_expression(index.start) if index.start is not None else ""
        stop = self.generate_expression(index.stop) if index.stop is not None else ""
        if index.has_step:
            step = (
                self.generate_expression(index.step) if index.step is not None else ""
            )
            return f"{start}:{stop}:{step}"
        return f"{start}:{stop}"

    def map_type(self, mojo_type):
        """Map a Mojo type name to the closest CrossGL type name."""
        if mojo_type is None:
            return "void"
        if isinstance(mojo_type, str) and mojo_type.startswith("*"):
            mojo_type = mojo_type[1:].lstrip()
        mojo_type = self.strip_reference_type(mojo_type)
        mapped_type = self.type_map.get(mojo_type)
        if mapped_type:
            return mapped_type
        mlir_type = self.map_mlir_backtick_type(mojo_type)
        if mlir_type:
            return mlir_type
        matrix_type = self.map_matrix_type(mojo_type)
        if matrix_type:
            return matrix_type
        return self.normalize_type_expression_operators(mojo_type)

    def normalize_type_expression_operators(self, mojo_type):
        """Normalize Mojo-only operators embedded in source-like type strings."""
        if not isinstance(mojo_type, str):
            return mojo_type
        mojo_type = re.sub(r"(?<=[\[,])\s*\*(?=[A-Za-z_`])", "", mojo_type)
        mojo_type = re.sub(r"(?<=[\[,])\s*//\s*(?=,|\])", " /", mojo_type)
        return re.sub(r"\s*//\s*", " / ", mojo_type)

    def strip_reference_type(self, mojo_type):
        if not isinstance(mojo_type, str):
            return mojo_type
        match = self.REFERENCE_TYPE_PATTERN.match(mojo_type)
        if match:
            return match.group(1)
        return mojo_type

    def map_matrix_type(self, mojo_type):
        if not isinstance(mojo_type, str):
            return None

        match = self.MATRIX_TYPE_PATTERN.match(mojo_type)
        if not match:
            return None

        dtype, columns, rows = match.groups()
        prefix = self.MATRIX_DTYPE_PREFIXES.get(dtype)
        if prefix is None:
            return None

        if columns == rows and prefix in {"mat", "dmat"}:
            return f"{prefix}{columns}"
        return f"{prefix}{columns}x{rows}"

    def map_mlir_backtick_type(self, mojo_type):
        if not isinstance(mojo_type, str):
            return None

        match = self.MLIR_BACKTICK_TYPE_PATTERN.match(mojo_type)
        if not match:
            return None

        payload = match.group(1)
        return self.MLIR_TYPE_MAP.get(
            payload, f"MLIR_{self.sanitize_identifier(payload)}"
        )

    def map_member_name(self, member):
        return self.map_identifier_name(member)

    def map_identifier_name(self, member):
        if not isinstance(member, str):
            return member

        match = self.BACKTICK_IDENTIFIER_PATTERN.match(member)
        if not match:
            if member in self.CROSSGL_RESERVED_IDENTIFIERS:
                return f"{member}_"
            return member

        return self.sanitize_identifier(match.group(1))

    def sanitize_identifier(self, name):
        sanitized = re.sub(r"[^A-Za-z0-9_]+", "_", name).strip("_")
        if not sanitized:
            sanitized = "_".join(
                f"u{ord(char):x}" for char in name if not char.isspace()
            )
            if not sanitized:
                return "metadata"
        if sanitized[0].isdigit():
            sanitized = f"_{sanitized}"
        if sanitized.lower() in self.CROSSGL_RESERVED_IDENTIFIERS:
            return f"{sanitized}_"
        return sanitized

    def map_semantic(self, attributes):
        """Map Mojo decorators or attributes to CrossGL semantic annotations."""
        return self.map_attributes(attributes)

    def map_attributes(self, attributes):
        """Map Mojo decorators or attributes to CrossGL annotation syntax."""
        if not attributes:
            return ""

        mapped_attributes = []
        for attr in attributes:
            if hasattr(attr, "name"):
                mapped_attributes.append(self.map_attribute(attr))
        return " ".join(mapped_attributes)

    def map_attribute(self, attr):
        name = self.semantic_map.get(attr.name, attr.name)
        args = getattr(attr, "args", getattr(attr, "arguments", [])) or []
        if not args:
            return f"@ {name}"

        args_str = ", ".join(self.generate_attribute_argument(arg) for arg in args)
        return f"@ {name}({args_str})"

    def generate_attribute_argument(self, arg):
        if isinstance(arg, str):
            return self.normalize_string_literal(arg).strip()
        return self.generate_expression(arg)

    def normalize_string_literal(self, value):
        for quote in ("'", '"'):
            quote_index = value.find(quote)
            if quote_index <= 0:
                continue
            prefix = value[:quote_index]
            if prefix in self.STRING_LITERAL_PREFIXES and value.endswith(quote):
                return value[quote_index:]
        return value

    def map_function_attributes(self, func):
        if not hasattr(func, "attributes") or not func.attributes:
            return ""

        non_stage_attributes = [
            attr
            for attr in func.attributes
            if not (hasattr(attr, "name") and attr.name in self.SHADER_STAGE_ATTRIBUTES)
        ]
        semantic = self.map_semantic(non_stage_attributes)
        return f" {semantic}" if semantic else ""

    def map_function(self, func_name, arg_count=None):
        mapped_name = self.map_identifier_name(func_name)
        if self.is_user_defined_function(func_name, arg_count):
            return mapped_name
        resolved_name = self.resolve_imported_function_name(func_name)
        mapped_function = self.function_map.get(resolved_name)
        if mapped_function:
            return mapped_function

        constructor_type = self.map_constructor_function(func_name)
        if constructor_type:
            return constructor_type

        return mapped_name

    def map_constructor_function(self, func_name):
        if not isinstance(func_name, str):
            return None
        if "." in func_name:
            return None

        mapped_type = self.map_type(func_name)
        if mapped_type == func_name:
            return None
        return mapped_type

    def map_operator(self, op):
        if op == "is":
            return "=="
        if op == "is not":
            return "!="
        if op == "@":
            return "*"
        if op == "@=":
            return "*="
        if op == ":=":
            return "="
        if op == "//":
            return "/"
        if op == "//=":
            return "/="
        return op

"""Reverse code generator that emits CrossGL from Slang AST nodes."""

import re
from decimal import Decimal, InvalidOperation

from crosstl.translator.stage_utils import normalize_stage_name

from .SlangAst import *
from .SlangLexer import *
from .SlangParser import *


class SlangToCrossGLConverter:
    """Serialize Slang backend AST nodes back into CrossGL source."""

    BINARY_PRECEDENCE = {
        "||": 1,
        "&&": 2,
        "|": 3,
        "^": 4,
        "&": 5,
        "==": 6,
        "!=": 6,
        "<": 7,
        ">": 7,
        "<=": 7,
        ">=": 7,
        "<<": 8,
        ">>": 8,
        "+": 9,
        "-": 9,
        "*": 10,
        "/": 10,
        "%": 10,
    }
    ASSOCIATIVE_BINARY_OPS = {"+", "*", "&&", "||", "&", "|", "^"}
    DECIMAL_NUMERIC_LITERAL = re.compile(
        r"^(?P<body>(?:\d+\.\d*|\.\d+|\d+)"
        r"(?:[eE][+-]?\d+)?)(?P<suffix>[fFhHuUlL]*)$"
    )
    HEX_NUMERIC_LITERAL = re.compile(r"^0[xX][0-9a-fA-F]+[uUlL]*$")
    RAY_PAYLOAD_ACCESS_SEMANTIC = re.compile(r"^(read|write)\((.*)\)$")
    SAMPLE_METHOD_MAP = {
        "Sample": "texture",
        "SampleBias": "texture",
        "SampleCmp": "textureCompare",
        "SampleCmpLevel": "textureCompareLod",
        "SampleCmpLevelZero": "textureCompareLod",
        "SampleCmpGrad": "textureCompareGrad",
        "SampleLevel": "textureLod",
        "SampleLOD": "textureLod",
        "SampleGrad": "textureGrad",
        "Load": "texelFetch",
    }
    SAMPLEABLE_RESOURCE_TYPES = {
        "Texture1D",
        "Texture1DArray",
        "Texture2D",
        "Texture2DArray",
        "Texture2DMS",
        "Texture2DMSArray",
        "Texture3D",
        "TextureCube",
        "TextureCubeArray",
        "Sampler1D",
        "Sampler1DArray",
        "Sampler2D",
        "Sampler2DArray",
        "Sampler2DMS",
        "Sampler2DMSArray",
        "Sampler3D",
        "SamplerCube",
        "SamplerCubeArray",
        "Sampler2DShadow",
        "Sampler2DArrayShadow",
        "SamplerCubeShadow",
        "SamplerCubeArrayShadow",
    }
    STORAGE_IMAGE_RESOURCE_TYPES = {
        "RWTexture1D",
        "RWTexture1DArray",
        "RWTexture2D",
        "RWTexture2DArray",
        "RWTexture2DMS",
        "RWTexture2DMSArray",
        "RWTexture3D",
        "RWTextureCube",
        "RWTextureCubeArray",
    }
    CROSSGL_RESERVED_IDENTIFIERS = {
        "as",
        "async",
        "await",
        "bool",
        "box",
        "break",
        "buffer",
        "cbuffer",
        "char",
        "class",
        "compute",
        "const",
        "continue",
        "default",
        "do",
        "double",
        "else",
        "elif",
        "enum",
        "extern",
        "false",
        "float",
        "for",
        "fragment",
        "from",
        "fn",
        "geometry",
        "global",
        "half",
        "if",
        "image1D",
        "image1DArray",
        "image2D",
        "image2DArray",
        "image2DMS",
        "image2DMSArray",
        "image3D",
        "imageCube",
        "imageCubeArray",
        "impl",
        "import",
        "in",
        "int",
        "interface",
        "kernel",
        "layout",
        "let",
        "local",
        "loop",
        "match",
        "mesh",
        "module",
        "move",
        "mut",
        "namespace",
        "private",
        "protected",
        "pub",
        "ref",
        "return",
        "safe",
        "sampler",
        "sampler1D",
        "sampler1DArray",
        "sampler2DArray",
        "sampler2DMS",
        "sampler2DMSArray",
        "sampler2DShadow",
        "sampler2DArrayShadow",
        "sampler3D",
        "samplerCube",
        "samplerCubeArray",
        "samplerCubeShadow",
        "samplerCubeArrayShadow",
        "shader",
        "shared",
        "static",
        "string",
        "struct",
        "switch",
        "task",
        "tessellation",
        "trait",
        "true",
        "uint",
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
        self.vertex_inputs = []
        self.vertex_outputs = []
        self.fragment_inputs = []
        self.fragment_outputs = []
        self.cbuffers = []
        self.type_map = {
            "void": "void",
            "float2": "vec2",
            "float3": "vec3",
            "float4": "vec4",
            "float2x2": "mat2",
            "float3x3": "mat3",
            "float4x4": "mat4",
            "int": "int",
            "int2": "ivec2",
            "int3": "ivec3",
            "int4": "ivec4",
            "uint": "uint",
            "uint2": "uvec2",
            "uint3": "uvec3",
            "uint4": "uvec4",
            "bool": "bool",
            "bool2": "bvec2",
            "bool3": "bvec3",
            "bool4": "bvec4",
            "float": "float",
            "double": "double",
            "Texture1D": "sampler1D",
            "Texture1DArray": "sampler1DArray",
            "Texture2D": "sampler2D",
            "Texture2DArray": "sampler2DArray",
            "Texture2DMS": "sampler2DMS",
            "Texture2DMSArray": "sampler2DMSArray",
            "Texture3D": "sampler3D",
            "TextureCube": "samplerCube",
            "TextureCubeArray": "samplerCubeArray",
            "SamplerState": "sampler",
            "SamplerComparisonState": "sampler",
            "Sampler1D": "sampler1D",
            "Sampler1DArray": "sampler1DArray",
            "Sampler2D": "sampler2D",
            "Sampler2DArray": "sampler2DArray",
            "Sampler2DMS": "sampler2DMS",
            "Sampler2DMSArray": "sampler2DMSArray",
            "Sampler3D": "sampler3D",
            "SamplerCube": "samplerCube",
            "SamplerCubeArray": "samplerCubeArray",
            "Sampler2DShadow": "sampler2DShadow",
            "Sampler2DArrayShadow": "sampler2DArrayShadow",
            "SamplerCubeShadow": "samplerCubeShadow",
            "SamplerCubeArrayShadow": "samplerCubeArrayShadow",
            "RWTexture1D": "image1D",
            "RWTexture1DArray": "image1DArray",
            "RWTexture2D": "image2D",
            "RWTexture2DArray": "image2DArray",
            "RWTexture2DMS": "image2DMS",
            "RWTexture2DMSArray": "image2DMSArray",
            "RWTexture3D": "image3D",
            "RWTextureCube": "imageCube",
            "RWTextureCubeArray": "imageCubeArray",
        }
        self.function_map = {
            "frac": "fract",
            "fmod": "mod",
            "lerp": "mix",
            "rsqrt": "inversesqrt",
        }
        self.user_function_names = set()
        self.function_name_map = {}
        self.sampleable_resource_scopes = [set()]
        self.sampleable_resource_type_scopes = [{}]
        self.storage_image_resource_scopes = [set()]
        self.storage_image_resource_type_scopes = [{}]
        self.variable_type_scopes = [{}]
        self.struct_member_types = {}
        self.identifier_rename_scopes = [{}]
        self.identifier_used_name_scopes = [set()]
        self.struct_member_name_maps = {}

        self.semantic_map = {
            # Vertex inputs position
            "POSITION": "in_Position",
            "POSITION0": "in_Position0",
            "POSITION1": "in_Position1",
            "POSITION2": "in_Position2",
            "POSITION3": "in_Position3",
            "POSITION4": "in_Position4",
            "POSITION5": "in_Position5",
            "POSITION6": "in_Position6",
            "POSITION7": "in_Position7",
            # Vertex inputs normal
            "NORMAL": "in_Normal",
            "NORMAL0": "in_Normal0",
            "NORMAL1": "in_Normal1",
            "NORMAL2": "in_Normal2",
            "NORMAL3": "in_Normal3",
            "NORMAL4": "in_Normal4",
            "NORMAL5": "in_Normal5",
            "NORMAL6": "in_Normal6",
            "NORMAL7": "in_Normal7",
            # Vertex inputs tangent
            "TANGENT": "in_Tangent",
            "TANGENT0": "in_Tangent0",
            "TANGENT1": "in_Tangent1",
            "TANGENT2": "in_Tangent2",
            "TANGENT3": "in_Tangent3",
            "TANGENT4": "in_Tangent4",
            "TANGENT5": "in_Tangent5",
            "TANGENT6": "in_Tangent6",
            "TANGENT7": "in_Tangent7",
            # Vertex inputs binormal
            "BINORMAL": "in_Binormal",
            "BINORMAL0": "in_Binormal0",
            "BINORMAL1": "in_Binormal1",
            "BINORMAL2": "in_Binormal2",
            "BINORMAL3": "in_Binormal3",
            "BINORMAL4": "in_Binormal4",
            "BINORMAL5": "in_Binormal5",
            "BINORMAL6": "in_Binormal6",
            "BINORMAL7": "in_Binormal7",
            # Vertex inputs color
            "COLOR": "Color",
            "COLOR0": "Color0",
            "COLOR1": "Color1",
            "COLOR2": "Color2",
            "COLOR3": "Color3",
            "COLOR4": "Color4",
            "COLOR5": "Color5",
            "COLOR6": "Color6",
            "COLOR7": "Color7",
            # Vertex inputs texcoord
            "TEXCOORD": "TexCoord",
            "TEXCOORD0": "TexCoord0",
            "TEXCOORD1": "TexCoord1",
            "TEXCOORD2": "TexCoord2",
            "TEXCOORD3": "TexCoord3",
            "TEXCOORD4": "TexCoord4",
            "TEXCOORD5": "TexCoord5",
            "TEXCOORD6": "TexCoord6",
            # Vertex inputs instance
            "FRONT_FACE": "gl_IsFrontFace",
            "PRIMITIVE_ID": "gl_PrimitiveID",
            "INSTANCE_ID": "gl_InstanceID",
            "VERTEX_ID": "gl_VertexID",
            # Vertex outputs
            "SV_Position": "Out_Position",
            "SV_Position0": "Out_Position0",
            "SV_Position1": "Out_Position1",
            "SV_Position2": "Out_Position2",
            "SV_Position3": "Out_Position3",
            "SV_Position4": "Out_Position4",
            "SV_Position5": "Out_Position5",
            "SV_Position6": "Out_Position6",
            "SV_Position7": "Out_Position7",
            # Fragment inputs
            "SV_Target": "Out_Color",
            "SV_Target0": "Out_Color0",
            "SV_Target1": "Out_Color1",
            "SV_Target2": "Out_Color2",
            "SV_Target3": "Out_Color3",
            "SV_Target4": "Out_Color4",
            "SV_Target5": "Out_Color5",
            "SV_Target6": "Out_Color6",
            "SV_Target7": "Out_Color7",
            "SV_Depth": "Out_Depth",
            "SV_Depth0": "Out_Depth0",
            "SV_Depth1": "Out_Depth1",
            "SV_Depth2": "Out_Depth2",
            "SV_Depth3": "Out_Depth3",
            "SV_Depth4": "Out_Depth4",
            "SV_Depth5": "Out_Depth5",
            "SV_Depth6": "Out_Depth6",
            "SV_Depth7": "Out_Depth7",
        }

    def generate(self, ast):
        self.raise_for_unsupported_conformance_constructs(ast)
        exported_functions = [
            exp.item
            for exp in getattr(ast, "exports", [])
            if isinstance(getattr(exp, "item", None), FunctionNode)
        ]
        functions = [*getattr(ast, "functions", []), *exported_functions]
        self.user_function_names = {getattr(func, "name", None) for func in functions}
        self.user_function_names.discard(None)
        self.function_name_map = self.collect_function_name_map(functions)
        resource_types = self.collect_sampleable_resource_types(ast)
        storage_image_types = self.collect_storage_image_resource_types(ast)
        self.struct_member_types = self.collect_struct_member_types(ast)
        self.struct_member_name_maps = self.collect_struct_member_name_maps(ast)
        self.identifier_rename_scopes = [{}]
        self.identifier_used_name_scopes = [set()]
        self.register_global_identifier_renames(ast)
        self.variable_type_scopes = [self.collect_global_variable_types(ast)]
        self.sampleable_resource_scopes = [set(resource_types)]
        self.sampleable_resource_type_scopes = [resource_types]
        self.storage_image_resource_scopes = [set(storage_image_types)]
        self.storage_image_resource_type_scopes = [storage_image_types]
        code = ""
        if ast.imports:
            for imp in ast.imports:
                code += f"import {self.format_import_path(imp.module_name)};\n"
            code += "\n"
        if getattr(ast, "includes", None):
            for include in ast.includes:
                code += f"import {self.format_import_path(include)};\n"
            code += "\n"
        code += "shader main {\n"
        if ast.exports:
            for exp in ast.exports:
                code += self.generate_export(exp)
            code += "\n"
        for node in ast.typedefs:
            code += (
                f"    typedef {self.map_type(node.original_type)} {node.new_type};\n"
            )
        for enum in getattr(ast, "enums", []) or []:
            if isinstance(enum, EnumNode):
                code += f"    enum {enum.name} {{\n"
                for member_name, member_value in enum.members:
                    if member_value is None:
                        code += f"        {member_name},\n"
                    else:
                        value = self.generate_expression(member_value)
                        code += f"        {member_name} = {value},\n"
                code += "    }\n"
        for node in ast.structs:
            if isinstance(node, StructNode):
                code += f"    struct {node.name} {{\n"
                for member in node.members:
                    semantic = self.map_semantic(member.semantic)
                    semantic_suffix = f" {semantic}" if semantic else ""
                    member_name = self.format_struct_member_name(node, member.name)
                    code += (
                        f"        {self.map_type(member.vtype)} {member_name}"
                        f"{self.format_array_suffixes(member)}{semantic_suffix};\n"
                    )
                code += "    }\n"
        for node in ast.global_vars:
            code += self.generate_global_variable(node)
        if ast.cbuffers:
            code += "    // Constant Buffers\n"
            code += self.generate_cbuffers(ast)

        for func in ast.functions:
            qualifier = normalize_stage_name(func.qualifier)
            if qualifier == "vertex":
                code += "    vertex {\n"
                code += self.generate_function(func)
                code += "    }\n\n"
            elif qualifier == "fragment":
                code += "    fragment {\n"
                code += self.generate_function(func)
                code += "    }\n\n"

            elif qualifier == "compute":
                code += "    compute {\n"
                code += self.generate_numthreads_layout(func)
                code += self.generate_function(func)
                code += "    }\n\n"
            elif qualifier:
                code += f"    {qualifier} {{\n"
                code += self.generate_function(func)
                code += "    }\n\n"
            else:
                code += self.generate_function(func)

        code += "}\n"
        return code

    def raise_for_unsupported_conformance_constructs(self, ast):
        constructs = []

        for interface in getattr(ast, "interfaces", []) or []:
            constructs.append(f"interface {interface.name}")

        for struct in getattr(ast, "structs", []) or []:
            conformances = getattr(struct, "conformances", []) or []
            if conformances:
                constructs.append(f"struct {struct.name} : {', '.join(conformances)}")

        for extension in getattr(ast, "extensions", []) or []:
            conformances = getattr(extension, "conformances", []) or []
            suffix = f" : {', '.join(conformances)}" if conformances else ""
            constructs.append(f"extension {extension.extended_type}{suffix}")

        for function in getattr(ast, "functions", []) or []:
            constructs.extend(self.format_function_generic_constraints(function))

        for export in getattr(ast, "exports", []) or []:
            item = getattr(export, "item", None)
            if isinstance(item, InterfaceNode):
                constructs.append(f"interface {item.name}")
            elif isinstance(item, ExtensionNode):
                conformances = getattr(item, "conformances", []) or []
                suffix = f" : {', '.join(conformances)}" if conformances else ""
                constructs.append(f"extension {item.extended_type}{suffix}")
            elif isinstance(item, StructNode):
                conformances = getattr(item, "conformances", []) or []
                if conformances:
                    constructs.append(f"struct {item.name} : {', '.join(conformances)}")
            elif isinstance(item, FunctionNode):
                constructs.extend(self.format_function_generic_constraints(item))

        if constructs:
            details = ", ".join(constructs)
            raise NotImplementedError(
                "Reverse Slang to CrossGL does not support "
                f"interface/conformance constructs: {details}"
            )

    def format_function_generic_constraints(self, function):
        constraints = []
        for constraint in getattr(function, "generic_constraints", []) or []:
            relation = getattr(constraint, "relation", ":")
            constraints.append(
                f"function {function.name} where "
                f"{constraint.parameter} {relation} {constraint.constraint_type}"
            )
        return constraints

    def format_import_path(self, path):
        path = str(path)
        if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*", path):
            return path
        escaped = path.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'

    def collect_function_name_map(self, functions):
        name_map = {}
        used_names = set()
        for function in functions:
            name = getattr(function, "name", None)
            if not name or name in name_map:
                continue
            safe_name = self.sanitize_crossgl_identifier(name, used_names)
            name_map[name] = safe_name
            used_names.add(safe_name)
        return name_map

    def format_function_name(self, name):
        return self.function_name_map.get(name, name)

    def register_global_identifier_renames(self, ast):
        for node in getattr(ast, "global_vars", []) or []:
            declaration = node.left if isinstance(node, AssignmentNode) else node
            if isinstance(declaration, VariableNode):
                self.register_identifier_declaration(declaration)

    def push_identifier_scope(self):
        self.identifier_rename_scopes.append({})
        self.identifier_used_name_scopes.append(set())

    def pop_identifier_scope(self):
        if len(self.identifier_rename_scopes) > 1:
            self.identifier_rename_scopes.pop()
        if len(self.identifier_used_name_scopes) > 1:
            self.identifier_used_name_scopes.pop()

    def register_identifier_declaration(self, node):
        name = getattr(node, "name", None)
        if not name:
            return name
        current_scope = self.identifier_rename_scopes[-1]
        if name in current_scope:
            return current_scope[name]
        safe_name = self.sanitize_crossgl_identifier(
            name, self.identifier_used_name_scopes[-1]
        )
        current_scope[name] = safe_name
        self.identifier_used_name_scopes[-1].add(safe_name)
        return safe_name

    def format_identifier(self, name):
        if not name:
            return name
        for scope in reversed(self.identifier_rename_scopes):
            if name in scope:
                return scope[name]
        return name

    def collect_struct_member_name_maps(self, ast):
        maps = {}
        for struct in getattr(ast, "structs", []) or []:
            self.collect_struct_member_name_map_from_node(struct, maps)
        for cbuffer in getattr(ast, "cbuffers", []) or []:
            self.collect_struct_member_name_map_from_node(cbuffer, maps)
        for export in getattr(ast, "exports", []) or []:
            item = getattr(export, "item", None)
            if isinstance(item, StructNode):
                self.collect_struct_member_name_map_from_node(item, maps)
        return maps

    def collect_struct_member_name_map_from_node(self, struct, maps):
        used_names = set()
        member_names = {}
        for member in getattr(struct, "members", []) or []:
            name = getattr(member, "name", None)
            if not name:
                continue
            safe_name = self.sanitize_crossgl_identifier(name, used_names)
            member_names[name] = safe_name
            used_names.add(safe_name)
        if member_names:
            maps[struct.name] = member_names

        for nested in getattr(struct, "structs", []) or []:
            self.collect_struct_member_name_map_from_node(nested, maps)

    def format_struct_member_name(self, struct, member_name):
        struct_name = getattr(struct, "name", struct)
        return self.struct_member_name_maps.get(struct_name, {}).get(
            member_name, member_name
        )

    def format_member_access_name(self, expr):
        object_type = self.unwrap_resource_container_type(
            self.expression_type(expr.object)
        )
        if object_type:
            member_map = self.struct_member_name_maps.get(object_type, {})
            if expr.member in member_map:
                return member_map[expr.member]
        if self.crossgl_identifier_needs_sanitizing(expr.member):
            return self.sanitize_crossgl_identifier(expr.member)
        return expr.member

    def sanitize_crossgl_identifier(self, name, used_names=None):
        used_names = used_names or set()
        safe_name = str(name)
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", safe_name):
            safe_name = re.sub(r"\W+", "_", safe_name)
            if not safe_name or safe_name[0].isdigit():
                safe_name = f"_{safe_name}"

        if self.crossgl_identifier_needs_sanitizing(safe_name):
            safe_name = f"{safe_name}_"

        while (
            self.crossgl_identifier_needs_sanitizing(safe_name)
            or safe_name in used_names
        ):
            safe_name = f"{safe_name}_"
        return safe_name

    def crossgl_identifier_needs_sanitizing(self, name):
        return str(name) in self.CROSSGL_RESERVED_IDENTIFIERS

    def generate_export(self, exp):
        item = exp.item
        if isinstance(item, FunctionNode):
            return self.generate_function(item)
        if isinstance(item, StructNode):
            code = f"    struct {item.name} {{\n"
            for member in item.members:
                semantic = self.map_semantic(member.semantic)
                semantic_suffix = f" {semantic}" if semantic else ""
                member_name = self.format_struct_member_name(item, member.name)
                code += (
                    f"        {self.map_type(member.vtype)} {member_name}"
                    f"{self.format_array_suffixes(member)}{semantic_suffix};\n"
                )
            code += "    }\n"
            return code
        if isinstance(item, (VariableNode, AssignmentNode)):
            return self.generate_global_variable(item)
        return ""

    def generate_numthreads_layout(self, func):
        numthreads = getattr(func, "numthreads", None)
        if not numthreads:
            return ""

        x, y, z = numthreads
        return (
            "        "
            f"layout(local_size_x = {x}, local_size_y = {y}, local_size_z = {z}) in;\n"
        )

    def generate_cbuffers(self, ast):
        code = ""
        for node in ast.cbuffers:
            if isinstance(node, StructNode):
                metadata = self.format_variable_metadata(node)
                code += f"    cbuffer {node.name}{metadata} {{\n"
                for member in node.members:
                    member_name = self.format_struct_member_name(node, member.name)
                    code += (
                        f"        {self.map_type(member.vtype)} {member_name}"
                        f"{self.format_array_suffixes(member)};\n"
                    )
                code += "    }\n"
        return code

    def generate_function(self, func, indent=1):
        """Render one Slang function node as a CrossGL function."""
        code = " "
        code += "  " * indent
        self.push_identifier_scope()
        self.push_variable_type_scope(func.params)
        self.push_sampleable_resource_scope(func.params)
        self.push_storage_image_resource_scope(func.params)
        for param in func.params:
            self.register_identifier_declaration(param)
        params = ", ".join(self.generate_parameter(p) for p in func.params)
        semantic = self.map_semantic(func.semantic)
        semantic_suffix = f" {semantic}" if semantic else ""
        code += (
            f"    {self.map_type(func.return_type)} "
            f"{self.format_function_name(func.name)}({params}){semantic_suffix} {{\n"
        )
        code += self.generate_function_body(func.body, indent=indent + 1)
        code += "    }\n\n"
        self.pop_storage_image_resource_scope()
        self.pop_sampleable_resource_scope()
        self.pop_variable_type_scope()
        self.pop_identifier_scope()
        return code

    def generate_parameter(self, param):
        parameter = (
            f"{self.map_type(param.vtype)} {self.format_identifier(param.name)}"
            f"{self.format_array_suffixes(param)}"
        )
        semantic = self.map_semantic(param.semantic)
        if semantic:
            parameter += f" {semantic}"
        return parameter

    def format_array_suffixes(self, node, is_main=False):
        sizes = getattr(node, "array_sizes", None)
        if not sizes:
            return ""
        parts = []
        for size in sizes:
            if size is None:
                parts.append("[]")
            else:
                parts.append(f"[{self.generate_expression(size, is_main)}]")
        return "".join(parts)

    def generate_function_body(self, body, indent=0, is_main=False):
        code = ""
        for stmt in body:
            code += "    " * indent
            if isinstance(stmt, VariableNode):
                self.register_variable_type(stmt)
                self.register_sampleable_resource(stmt)
                self.register_storage_image_resource(stmt)
                self.register_identifier_declaration(stmt)
                code += (
                    f"{self.map_type(stmt.vtype)} {self.format_identifier(stmt.name)}"
                    f"{self.format_array_suffixes(stmt, is_main)};\n"
                )
            elif isinstance(stmt, AssignmentNode):
                if isinstance(stmt.left, VariableNode) and stmt.left.vtype:
                    self.register_variable_type(stmt.left)
                    self.register_sampleable_resource(stmt.left)
                    self.register_storage_image_resource(stmt.left)
                    self.register_identifier_declaration(stmt.left)
                code += self.generate_assignment(stmt, is_main) + ";\n"
            elif isinstance(stmt, (FunctionCallNode, MethodCallNode, CallNode)):
                code += f"{self.generate_expression(stmt, is_main)};\n"
            elif isinstance(stmt, BinaryOpNode):
                code += f"{self.generate_expression(stmt, is_main)};\n"
            elif isinstance(stmt, UnaryOpNode):
                code += f"{self.generate_expression(stmt, is_main)};\n"
            elif isinstance(stmt, ReturnNode):
                if not is_main:
                    if stmt.value is None:
                        code += "return;\n"
                    else:
                        code += (
                            f"return {self.generate_expression(stmt.value, is_main)};\n"
                        )
            elif isinstance(stmt, ForNode):
                code += self.generate_for_loop(stmt, indent, is_main)
            elif isinstance(stmt, WhileNode):
                code += self.generate_while_loop(stmt, indent, is_main)
            elif isinstance(stmt, DoWhileNode):
                code += self.generate_do_while_loop(stmt, indent, is_main)
            elif isinstance(stmt, SwitchNode):
                code += self.generate_switch_statement(stmt, indent, is_main)
            elif isinstance(stmt, IfNode):
                code += self.generate_if_statement(stmt, indent, is_main)
            elif isinstance(stmt, BreakNode):
                code += "break;\n"
            elif isinstance(stmt, ContinueNode):
                code += "continue;\n"
            elif isinstance(stmt, DiscardNode):
                code += "discard;\n"
        return code

    def generate_for_loop(self, node, indent, is_main):
        init = self.generate_for_clause(node.init, is_main)
        condition = self.generate_for_clause(node.condition, is_main)
        update = self.generate_for_clause(node.update, is_main)

        code = f"for ({init}; {condition}; {update}) {{\n"
        code += self.generate_function_body(node.body, indent + 1, is_main)
        code += "    " * indent + "}\n"
        return code

    def generate_for_clause(self, clause, is_main):
        if clause is None:
            return ""
        if isinstance(clause, list):
            return ", ".join(self.generate_expression(item, is_main) for item in clause)
        return self.generate_expression(clause, is_main)

    def generate_while_loop(self, node, indent, is_main):
        condition = self.generate_expression(node.condition, is_main)

        code = f"while ({condition}) {{\n"
        code += self.generate_function_body(node.body, indent + 1, is_main)
        code += "    " * indent + "}\n"
        return code

    def generate_do_while_loop(self, node, indent, is_main):
        condition = self.generate_expression(node.condition, is_main)

        code = "do {\n"
        code += self.generate_function_body(node.body, indent + 1, is_main)
        code += "    " * indent + f"}} while ({condition});\n"
        return code

    def generate_switch_statement(self, node, indent, is_main):
        expression = self.generate_expression(node.expression, is_main)

        code = f"switch ({expression}) {{\n"
        ordered_cases = getattr(node, "ordered_cases", None)
        if ordered_cases is not None:
            for case in ordered_cases:
                if case.value is None:
                    code += "    " * (indent + 1) + "default:\n"
                else:
                    value = self.generate_expression(case.value, is_main)
                    code += "    " * (indent + 1) + f"case {value}:\n"
                code += self.generate_function_body(case.body, indent + 2, is_main)
            code += "    " * indent + "}\n"
            return code

        for case in node.cases:
            value = self.generate_expression(case.value, is_main)
            code += "    " * (indent + 1) + f"case {value}:\n"
            code += self.generate_function_body(case.body, indent + 2, is_main)

        if node.default_case is not None:
            code += "    " * (indent + 1) + "default:\n"
            code += self.generate_function_body(node.default_case, indent + 2, is_main)

        code += "    " * indent + "}\n"
        return code

    def generate_if_statement(self, node, indent, is_main):
        condition = self.generate_expression(node.condition, is_main)

        code = f"if ({condition}) {{\n"
        code += self.generate_function_body(node.if_body, indent + 1, is_main)
        code += "    " * indent + "}"

        if node.else_body:
            if isinstance(node.else_body, IfNode):
                code += " else "
                code += self.generate_if_statement(node.else_body, indent, is_main)
            else:
                code += " else {\n"
                code += self.generate_function_body(node.else_body, indent + 1, is_main)
                code += "    " * indent + "}"

        code += "\n"
        return code

    def generate_assignment(self, node, is_main):
        if isinstance(node.left, VariableNode) and node.left.vtype:
            self.register_identifier_declaration(node.left)
        lhs = self.generate_expression(node.left, is_main)
        rhs = self.generate_expression(node.right, is_main)
        op = node.operator
        return f"{lhs} {op} {rhs}"

    def generate_global_variable(self, node):
        if isinstance(node, AssignmentNode):
            left = self.generate_variable_declaration(node.left)
            right = self.generate_expression(node.right, False)
            return f"    {left} {node.operator} {right};\n"
        return f"    {self.generate_variable_declaration(node)};\n"

    def generate_variable_declaration(self, node):
        return (
            f"{self.map_type(node.vtype)} "
            f"{self.format_identifier(node.name)}{self.format_array_suffixes(node)}"
            f"{self.format_variable_metadata(node)}"
        )

    def format_variable_metadata(self, node):
        metadata = []
        seen = set()

        def append_metadata(key, text):
            if key in seen:
                return
            seen.add(key)
            metadata.append(text)

        for qualifier in getattr(node, "qualifiers", []) or []:
            for name, value in self.layout_metadata_entries(qualifier):
                if name in {"set", "group"} and value:
                    append_metadata("set", f"@set({value})")
                elif name == "binding" and value:
                    append_metadata("binding", f"@binding({value})")
                elif name == "push_constant":
                    append_metadata("push_constant", "@push_constant")

        for attribute in getattr(node, "attributes", []) or []:
            name = str(attribute.get("name", "")).lower()
            arguments = attribute.get("arguments", [])
            if name == "vk::binding" and arguments:
                if len(arguments) > 1:
                    append_metadata("set", f"@set({arguments[1]})")
                append_metadata("binding", f"@binding({arguments[0]})")
            elif name == "vk::push_constant":
                append_metadata("push_constant", "@push_constant")

        register_name = getattr(node, "register", None)
        if self.should_emit_register_metadata(register_name, node):
            register_arguments = self.format_register_metadata_arguments(register_name)
            append_metadata("register", f"@register({register_arguments})")

        if not metadata:
            return ""
        return " " + " ".join(metadata)

    def layout_metadata_entries(self, qualifier):
        text = str(qualifier).strip()
        if not text.lower().startswith("layout(") or not text.endswith(")"):
            return []

        inner = text[text.find("(") + 1 : -1]
        entries = []
        for entry in self.split_top_level_commas(inner):
            if not entry:
                continue
            if "=" in entry:
                name, value = entry.split("=", 1)
                entries.append((name.strip().lower(), value.strip()))
            else:
                entries.append((entry.strip().lower(), None))
        return entries

    def split_top_level_commas(self, text):
        parts = []
        current = []
        depth = 0
        for char in str(text):
            if char in "([{":
                depth += 1
            elif char in ")]}" and depth > 0:
                depth -= 1

            if char == "," and depth == 0:
                parts.append("".join(current).strip())
                current = []
                continue
            current.append(char)

        if current or not parts:
            parts.append("".join(current).strip())
        return parts

    def should_emit_register_metadata(self, register_name, node=None):
        if not register_name:
            return False
        register_name = str(register_name).strip().lower()
        if re.match(r"^[tsu]\d", register_name):
            return True
        if not re.match(r"^b\d", register_name):
            return False
        node_type = getattr(node, "vtype", None)
        if not node_type:
            return False
        base_type = str(node_type).strip().split("<", 1)[0].strip()
        return base_type in {"ConstantBuffer", "ParameterBlock"}

    def format_register_metadata_arguments(self, register_name):
        return ", ".join(self.split_top_level_commas(str(register_name)))

    def binary_precedence(self, op):
        return self.BINARY_PRECEDENCE.get(op, 0)

    def binary_child_needs_parentheses(self, parent_op, child, is_right_child=False):
        if isinstance(child, AssignmentNode):
            return True
        if not isinstance(child, BinaryOpNode):
            return False

        parent_precedence = self.binary_precedence(parent_op)
        child_precedence = self.binary_precedence(child.op)
        if child_precedence < parent_precedence:
            return True
        if child_precedence > parent_precedence:
            return False
        return is_right_child and (
            parent_op not in self.ASSOCIATIVE_BINARY_OPS or child.op != parent_op
        )

    def generate_binary_expression(self, expr, is_main):
        left = self.generate_expression(expr.left, is_main)
        right = self.generate_expression(expr.right, is_main)
        if self.binary_child_needs_parentheses(expr.op, expr.left):
            left = f"({left})"
        if self.binary_child_needs_parentheses(expr.op, expr.right, True):
            right = f"({right})"
        return f"{left} {expr.op} {right}"

    def maybe_parenthesize_expression(self, expr, rendered):
        if isinstance(expr, (AssignmentNode, BinaryOpNode, TernaryOpNode)):
            return f"({rendered})"
        return rendered

    def generate_hlsl_mul_call(self, expr, is_main):
        if (
            expr.name != "mul"
            or len(expr.args) != 2
            or expr.name in self.user_function_names
        ):
            return None

        left = self.generate_expression(expr.args[0], is_main)
        right = self.generate_expression(expr.args[1], is_main)
        left = self.maybe_parenthesize_expression(expr.args[0], left)
        right = self.maybe_parenthesize_expression(expr.args[1], right)
        return f"({left} * {right})"

    def generate_expression(self, expr, is_main=False):
        """Render a Slang backend expression node as CrossGL syntax."""
        if isinstance(expr, str):
            return self.normalize_numeric_literal(expr)
        elif isinstance(expr, VariableNode):
            if expr.vtype:
                self.register_identifier_declaration(expr)
                return (
                    f"{self.map_type(expr.vtype)} {self.format_identifier(expr.name)}"
                    f"{self.format_array_suffixes(expr, is_main)}"
                )
            return self.format_identifier(expr.name)
        elif isinstance(expr, BinaryOpNode):
            return self.generate_binary_expression(expr, is_main)
        elif isinstance(expr, AssignmentNode):
            left = self.generate_expression(expr.left, is_main)
            right = self.generate_expression(expr.right, is_main)
            return f"{left} {expr.operator} {right}"
        elif isinstance(expr, CastNode):
            value = self.generate_expression(expr.expression, is_main)
            return f"{self.map_type(expr.target_type)}({value})"
        elif isinstance(expr, UnaryOpNode):
            operand = self.generate_expression(expr.operand, is_main)
            if isinstance(expr.operand, (AssignmentNode, BinaryOpNode)):
                operand = f"({operand})"
            if expr.op == "POST_INCREMENT":
                return f"{operand}++"
            if expr.op == "POST_DECREMENT":
                return f"{operand}--"
            if expr.op == "PRE_INCREMENT":
                return f"++{operand}"
            if expr.op == "PRE_DECREMENT":
                return f"--{operand}"
            return f"{expr.op}{operand}"
        elif isinstance(expr, FunctionCallNode):
            mul_call = self.generate_hlsl_mul_call(expr, is_main)
            if mul_call is not None:
                return mul_call
            args = ", ".join(
                self.generate_expression(arg, is_main) for arg in expr.args
            )
            if (
                expr.name == "saturate"
                and len(expr.args) == 1
                and expr.name not in self.user_function_names
            ):
                return f"clamp({args}, 0.0, 1.0)"
            if expr.name in self.user_function_names:
                name = self.format_function_name(expr.name)
            elif expr.name in self.type_map and expr.name[:1].islower():
                name = self.map_type(expr.name)
            else:
                name = self.function_map.get(expr.name, expr.name)
            return f"{name}({args})"
        elif isinstance(expr, MethodCallNode):
            obj = self.generate_expression(expr.object, is_main)
            image_call = self.generate_storage_image_method_call(expr, obj, is_main)
            if image_call is not None:
                return image_call

            texture_call = self.generate_texture_method_call(expr, obj, is_main)
            if texture_call is not None:
                return texture_call

            args = [self.generate_expression(arg, is_main) for arg in expr.args]
            args = ", ".join(args)
            return f"{obj}.{expr.method}({args})"
        elif isinstance(expr, CallNode):
            callee = self.generate_expression(expr.callee, is_main)
            args = ", ".join(
                self.generate_expression(arg, is_main) for arg in expr.args
            )
            return f"{callee}({args})"
        elif isinstance(expr, MemberAccessNode):
            obj = self.generate_expression(expr.object, is_main)
            if isinstance(
                expr.object, (AssignmentNode, BinaryOpNode, TernaryOpNode, UnaryOpNode)
            ):
                obj = f"({obj})"
            return f"{obj}.{self.format_member_access_name(expr)}"
        elif isinstance(expr, ArrayAccessNode):
            array = self.generate_expression(expr.array, is_main)
            index = self.generate_expression(expr.index, is_main)
            return f"{array}[{index}]"
        elif isinstance(expr, TernaryOpNode):
            condition = self.generate_expression(expr.condition, is_main)
            true_expr = self.generate_expression(expr.true_expr, is_main)
            false_expr = self.generate_expression(expr.false_expr, is_main)
            if isinstance(expr.condition, AssignmentNode):
                condition = f"({condition})"
            if isinstance(expr.true_expr, AssignmentNode):
                true_expr = f"({true_expr})"
            if isinstance(expr.false_expr, AssignmentNode):
                false_expr = f"({false_expr})"
            return f"({condition} ? {true_expr} : {false_expr})"
        elif isinstance(expr, InitializerListNode):
            elements = ", ".join(
                self.generate_expression(element, is_main) for element in expr.elements
            )
            return f"{{{elements}}}"
        elif isinstance(expr, VectorConstructorNode):
            args = ", ".join(
                self.generate_expression(arg, is_main) for arg in expr.args
            )
            return f"{self.map_type(expr.type_name)}({args})"
        elif isinstance(expr, ParenthesizedCommaNode):
            expressions = ", ".join(
                self.generate_expression(item, is_main) for item in expr.expressions
            )
            return f"({expressions})"
        else:
            return str(expr)

    def normalize_numeric_literal(self, value):
        if not isinstance(value, str):
            return value
        if self.HEX_NUMERIC_LITERAL.match(value):
            return value

        match = self.DECIMAL_NUMERIC_LITERAL.match(value)
        if not match:
            return value

        body = match.group("body")
        suffix = match.group("suffix")
        float_suffix = any(char in "fFhH" for char in suffix)
        integer_suffix = "".join(char for char in suffix if char not in "fFhH")
        float_like = "." in body or "e" in body.lower() or float_suffix
        if not float_like or integer_suffix:
            return value

        try:
            normalized = format(Decimal(body), "f")
        except InvalidOperation:
            return value

        if "." not in normalized:
            normalized = f"{normalized}.0"
        return normalized

    def map_type(self, slang_type):
        """Map a Slang type name to the closest CrossGL type name."""
        if slang_type:
            slang_type = slang_type.strip()
            pointer_suffix = ""
            while slang_type.endswith("*"):
                pointer_suffix += "*"
                slang_type = slang_type[:-1].strip()
            base_type = slang_type.split("<", 1)[0].strip()
            mapped_type = self.type_map.get(
                slang_type, self.type_map.get(base_type, slang_type)
            )
            return f"{mapped_type}{pointer_suffix}"
        return slang_type

    def collect_sampleable_resource_types(self, ast):
        resources = {}
        for node in getattr(ast, "global_vars", []) or []:
            declaration = node.left if isinstance(node, AssignmentNode) else node
            if self.is_sampleable_resource_type(getattr(declaration, "vtype", None)):
                name = getattr(declaration, "name", None)
                if name is not None:
                    resources[name] = getattr(declaration, "vtype", None)
        return resources

    def collect_storage_image_resource_types(self, ast):
        resources = {}
        for node in getattr(ast, "global_vars", []) or []:
            declaration = node.left if isinstance(node, AssignmentNode) else node
            if self.is_storage_image_resource_type(getattr(declaration, "vtype", None)):
                name = getattr(declaration, "name", None)
                if name is not None:
                    resources[name] = getattr(declaration, "vtype", None)
        return resources

    def collect_global_variable_types(self, ast):
        variables = {}
        for node in getattr(ast, "global_vars", []) or []:
            declaration = node.left if isinstance(node, AssignmentNode) else node
            name = getattr(declaration, "name", None)
            vtype = getattr(declaration, "vtype", None)
            if name is not None and vtype:
                variables[name] = vtype
        return variables

    def collect_struct_member_types(self, ast):
        structs = {}
        for struct in getattr(ast, "structs", []) or []:
            self.collect_struct_member_types_from_node(struct, structs)
        for export in getattr(ast, "exports", []) or []:
            item = getattr(export, "item", None)
            if isinstance(item, StructNode):
                self.collect_struct_member_types_from_node(item, structs)
        return structs

    def collect_struct_member_types_from_node(self, struct, structs):
        members = {}
        for member in getattr(struct, "members", []) or []:
            name = getattr(member, "name", None)
            vtype = getattr(member, "vtype", None)
            if name is not None and vtype:
                members[name] = vtype
        if members:
            structs[struct.name] = members

        for nested in getattr(struct, "structs", []) or []:
            self.collect_struct_member_types_from_node(nested, structs)

    def push_variable_type_scope(self, params):
        scope = {}
        for param in params or []:
            name = getattr(param, "name", None)
            vtype = getattr(param, "vtype", None)
            if name is not None and vtype:
                scope[name] = vtype
        self.variable_type_scopes.append(scope)

    def pop_variable_type_scope(self):
        if len(self.variable_type_scopes) > 1:
            self.variable_type_scopes.pop()

    def register_variable_type(self, node):
        name = getattr(node, "name", None)
        vtype = getattr(node, "vtype", None)
        if name is not None and vtype:
            self.variable_type_scopes[-1][name] = vtype

    def push_sampleable_resource_scope(self, params):
        scope = set()
        type_scope = {}
        for param in params or []:
            if self.is_sampleable_resource_type(getattr(param, "vtype", None)):
                name = getattr(param, "name", None)
                if name is not None:
                    scope.add(name)
                    type_scope[name] = getattr(param, "vtype", None)
        self.sampleable_resource_scopes.append(scope)
        self.sampleable_resource_type_scopes.append(type_scope)

    def push_storage_image_resource_scope(self, params):
        scope = set()
        type_scope = {}
        for param in params or []:
            if self.is_storage_image_resource_type(getattr(param, "vtype", None)):
                name = getattr(param, "name", None)
                if name is not None:
                    scope.add(name)
                    type_scope[name] = getattr(param, "vtype", None)
        self.storage_image_resource_scopes.append(scope)
        self.storage_image_resource_type_scopes.append(type_scope)

    def pop_sampleable_resource_scope(self):
        if len(self.sampleable_resource_scopes) > 1:
            self.sampleable_resource_scopes.pop()
        if len(self.sampleable_resource_type_scopes) > 1:
            self.sampleable_resource_type_scopes.pop()

    def pop_storage_image_resource_scope(self):
        if len(self.storage_image_resource_scopes) > 1:
            self.storage_image_resource_scopes.pop()
        if len(self.storage_image_resource_type_scopes) > 1:
            self.storage_image_resource_type_scopes.pop()

    def register_sampleable_resource(self, node):
        if self.is_sampleable_resource_type(getattr(node, "vtype", None)):
            name = getattr(node, "name", None)
            if name is not None:
                self.sampleable_resource_scopes[-1].add(name)
                self.sampleable_resource_type_scopes[-1][name] = getattr(
                    node, "vtype", None
                )

    def register_storage_image_resource(self, node):
        if self.is_storage_image_resource_type(getattr(node, "vtype", None)):
            name = getattr(node, "name", None)
            if name is not None:
                self.storage_image_resource_scopes[-1].add(name)
                self.storage_image_resource_type_scopes[-1][name] = getattr(
                    node, "vtype", None
                )

    def is_sampleable_resource_type(self, type_name):
        if not type_name:
            return False
        base_type = str(type_name).strip().split("<", 1)[0].strip()
        return base_type in self.SAMPLEABLE_RESOURCE_TYPES

    def is_storage_image_resource_type(self, type_name):
        if not type_name:
            return False
        base_type = str(type_name).strip().split("<", 1)[0].strip()
        return base_type in self.STORAGE_IMAGE_RESOURCE_TYPES

    def expression_type(self, expr):
        if isinstance(expr, VariableNode):
            if expr.vtype:
                return expr.vtype
            return self.lookup_variable_type(expr.name)
        if isinstance(expr, ArrayAccessNode):
            return self.expression_type(expr.array)
        if isinstance(expr, MemberAccessNode):
            object_type = self.expression_type(expr.object)
            struct_type = self.unwrap_resource_container_type(object_type)
            if not struct_type:
                return None
            members = self.struct_member_types.get(struct_type)
            if members is None:
                return None
            return members.get(expr.member)
        return None

    def lookup_variable_type(self, name):
        if not name:
            return None
        base_name = str(name).split("[", 1)[0]
        for scope in reversed(self.variable_type_scopes):
            if base_name in scope:
                return scope[base_name]
        return None

    def unwrap_resource_container_type(self, type_name):
        if not type_name:
            return None
        type_name = str(type_name).strip()
        while True:
            base_type = type_name.split("<", 1)[0].strip()
            if base_type not in {"ParameterBlock", "ConstantBuffer"}:
                return type_name
            inner_types = self.generic_type_arguments(type_name)
            if not inner_types:
                return type_name
            type_name = inner_types[0]

    def generic_type_arguments(self, type_name):
        text = str(type_name).strip()
        start = text.find("<")
        if start < 0 or not text.endswith(">"):
            return []
        return self.split_top_level_commas(text[start + 1 : -1])

    def generate_storage_image_method_call(self, expr, obj, is_main=False):
        if expr.method != "Load":
            return None
        if not self.is_storage_image_resource_expression(expr.object):
            return None
        args = [self.generate_expression(arg, is_main) for arg in expr.args or []]
        return f"imageLoad({', '.join([obj] + args)})"

    def generate_texture_method_call(self, expr, obj, is_main=False):
        if expr.method not in self.SAMPLE_METHOD_MAP:
            return None
        if not self.is_sampleable_resource_expression(expr.object):
            return None
        texture_func, args = self.crossgl_texture_call_parts(expr, is_main)
        return f"{texture_func}({', '.join([obj] + args)})"

    def crossgl_texture_call_parts(self, expr, is_main=False):
        if expr.method == "Load":
            args = self.format_texture_load_args(
                expr.args,
                is_main,
                self.sampleable_resource_expression_type(expr.object),
            )
            texture_func = (
                "texelFetchOffset" if len(expr.args or []) > 1 else "texelFetch"
            )
            return texture_func, args

        prefix, coord, extra_args = self.split_texture_sample_args(expr, is_main)
        if coord is None:
            return self.SAMPLE_METHOD_MAP[expr.method], prefix + extra_args

        if expr.method == "Sample":
            if extra_args:
                return "textureOffset", prefix + [coord, *extra_args]
            return "texture", prefix + [coord]

        if expr.method == "SampleBias":
            if len(extra_args) > 1:
                bias, offset, *rest = extra_args
                return "textureOffset", prefix + [coord, offset, bias, *rest]
            return "texture", prefix + [coord, *extra_args]

        if expr.method in {"SampleLevel", "SampleLOD"}:
            if len(extra_args) > 1:
                lod, offset, *rest = extra_args
                return "textureLodOffset", prefix + [coord, lod, offset, *rest]
            return "textureLod", prefix + [coord, *extra_args]

        if expr.method == "SampleGrad":
            if len(extra_args) > 2:
                ddx, ddy, offset, *rest = extra_args
                return "textureGradOffset", prefix + [coord, ddx, ddy, offset, *rest]
            return "textureGrad", prefix + [coord, *extra_args]

        if expr.method == "SampleCmp":
            if len(extra_args) > 1:
                compare, offset, *rest = extra_args
                return "textureCompareOffset", prefix + [coord, compare, offset, *rest]
            return "textureCompare", prefix + [coord, *extra_args]

        if expr.method == "SampleCmpLevel":
            if len(extra_args) > 2:
                compare, lod, offset, *rest = extra_args
                return (
                    "textureCompareLodOffset",
                    prefix + [coord, compare, lod, offset, *rest],
                )
            return "textureCompareLod", prefix + [coord, *extra_args]

        if expr.method == "SampleCmpLevelZero":
            if len(extra_args) > 1:
                compare, offset, *rest = extra_args
                return (
                    "textureCompareLodOffset",
                    prefix + [coord, compare, "0.0", offset, *rest],
                )
            return "textureCompareLod", prefix + [coord, *extra_args, "0.0"]

        if expr.method == "SampleCmpGrad":
            if len(extra_args) > 3:
                compare, ddx, ddy, offset, *rest = extra_args
                return (
                    "textureCompareGradOffset",
                    prefix + [coord, compare, ddx, ddy, offset, *rest],
                )
            return "textureCompareGrad", prefix + [coord, *extra_args]

        return self.SAMPLE_METHOD_MAP[expr.method], prefix + [coord, *extra_args]

    def format_texture_method_args(self, expr, is_main=False):
        if expr.method == "Load":
            return self.format_texture_load_args(
                expr.args,
                is_main,
                self.sampleable_resource_expression_type(expr.object),
            )
        return [self.generate_expression(arg, is_main) for arg in expr.args]

    def split_texture_sample_args(self, expr, is_main=False):
        generated_args = [
            self.generate_expression(arg, is_main) for arg in expr.args or []
        ]
        coord_index = 1 if self.texture_method_uses_explicit_sampler(expr) else 0
        if len(generated_args) <= coord_index:
            return generated_args, None, []
        prefix = generated_args[:coord_index]
        coord = generated_args[coord_index]
        extra_args = generated_args[coord_index + 1 :]
        return prefix, coord, extra_args

    def texture_method_uses_explicit_sampler(self, expr):
        type_name = self.sampleable_resource_expression_type(expr.object)
        if not type_name:
            return False
        base_type = str(type_name).strip().split("<", 1)[0].strip()
        return base_type.startswith("Texture")

    def format_texture_load_args(self, args, is_main=False, resource_type=None):
        if args:
            load_args = self.split_texture_load_vector_argument(
                args[0], is_main, resource_type
            )
            if load_args is not None:
                trailing_args = [
                    self.generate_expression(arg, is_main) for arg in args[1:]
                ]
                return load_args + trailing_args
        return [self.generate_expression(arg, is_main) for arg in args]

    def split_texture_load_vector_argument(
        self, arg, is_main=False, resource_type=None
    ):
        vector_type = None
        vector_args = None

        if isinstance(arg, VectorConstructorNode):
            vector_type = arg.type_name
            vector_args = arg.args
        elif isinstance(arg, FunctionCallNode):
            vector_type = arg.name
            vector_args = arg.args

        if vector_type in {"int4", "uint4"}:
            base_type = str(resource_type or "").strip().split("<", 1)[0].strip()
            if base_type != "Texture3D" or len(vector_args) != 4:
                return None
            coord_type = "ivec3" if vector_type == "int4" else "uvec3"
            x = self.generate_expression(vector_args[0], is_main)
            y = self.generate_expression(vector_args[1], is_main)
            z = self.generate_expression(vector_args[2], is_main)
            mip = self.generate_expression(vector_args[3], is_main)
            return [f"{coord_type}({x}, {y}, {z})", mip]

        if vector_type not in {"int3", "uint3"}:
            return None

        if len(vector_args) == 2:
            return [self.generate_expression(item, is_main) for item in vector_args]

        if len(vector_args) == 3:
            coord_type = "ivec2" if vector_type == "int3" else "uvec2"
            x = self.generate_expression(vector_args[0], is_main)
            y = self.generate_expression(vector_args[1], is_main)
            mip = self.generate_expression(vector_args[2], is_main)
            return [f"{coord_type}({x}, {y})", mip]

        return None

    def is_sampleable_resource_expression(self, expr):
        return self.sampleable_resource_expression_type(expr) is not None

    def is_storage_image_resource_expression(self, expr):
        name = self.expression_base_name(expr)
        if name is not None and any(
            name in scope for scope in reversed(self.storage_image_resource_scopes)
        ):
            return True
        return self.is_storage_image_resource_type(self.expression_type(expr))

    def sampleable_resource_expression_type(self, expr):
        name = self.expression_base_name(expr)
        if name is not None:
            for scope in reversed(self.sampleable_resource_type_scopes):
                if name in scope:
                    return scope[name]
        type_name = self.expression_type(expr)
        if self.is_sampleable_resource_type(type_name):
            return type_name
        return None

    def expression_base_name(self, expr):
        if isinstance(expr, str):
            return expr.split("[", 1)[0]
        if isinstance(expr, VariableNode):
            return expr.name.split("[", 1)[0]
        if isinstance(expr, ArrayAccessNode):
            return self.expression_base_name(expr.array)
        return None

    def map_semantic(self, semantic):
        """Map a Slang semantic to CrossGL semantic annotation syntax."""
        if semantic is None:
            return ""
        ray_payload_access = self.map_ray_payload_access_semantic(semantic)
        if ray_payload_access:
            return ray_payload_access
        return f"@ {self.semantic_map.get(semantic, semantic)}"

    def map_ray_payload_access_semantic(self, semantic):
        access_parts = self.split_semantic_chain(semantic)
        mapped = []
        for part in access_parts:
            match = self.RAY_PAYLOAD_ACCESS_SEMANTIC.match(part)
            if not match:
                return ""
            access_kind, access_args = match.groups()
            mapped.append(f"@ ray_payload_{access_kind}({access_args})")
        return " ".join(mapped)

    def split_semantic_chain(self, semantic):
        parts = []
        start = 0
        depth = 0
        for index, char in enumerate(semantic):
            if char == "(":
                depth += 1
            elif char == ")":
                depth = max(0, depth - 1)
            elif char == ":" and depth == 0:
                parts.append(semantic[start:index])
                start = index + 1
        parts.append(semantic[start:])
        return parts

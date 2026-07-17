"""Reverse code generator that emits CrossGL from Slang AST nodes."""

import re
from decimal import Decimal, InvalidOperation

from crosstl.translator.stage_utils import normalize_stage_name

from .SlangAst import *
from .SlangLexer import *
from .SlangParser import *


class UnsupportedSlangConformanceError(NotImplementedError):
    """Raised when Slang conformance semantics cannot be represented."""

    project_diagnostic_code = "project.translate.unsupported-feature"
    missing_capabilities = ("slang.interface-conformance-lowering",)

    def __init__(self, constructs):
        self.constructs = tuple(constructs)
        self.feature = "slang.interface-conformance"
        self.suggested_action = (
            "Remove the conformance dependency or add an explicit CrossGL "
            "interface/conformance lowering."
        )
        details = ", ".join(self.constructs)
        super().__init__(
            "Reverse Slang to CrossGL does not support interface/conformance "
            f"constructs: {details}. Suggested action: {self.suggested_action}"
        )


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
    HEX_NUMERIC_LITERAL = re.compile(
        r"^(?P<body>0[xX][0-9a-fA-F]+)(?P<suffix>[uUlL]*)$"
    )
    HEX_FLOAT_NUMERIC_LITERAL = re.compile(
        r"^(?P<body>0[xX](?:(?:[0-9a-fA-F]+(?:\.[0-9a-fA-F]*)?)|"
        r"(?:\.[0-9a-fA-F]+))[pP][+-]?\d+)(?P<suffix>[fFhH]?)$"
    )
    RAY_PAYLOAD_ACCESS_SEMANTIC = re.compile(r"^(read|write)\((.*)\)$")
    ASSOCIATED_DIFFERENTIAL_TYPE = re.compile(
        r"^(?P<base>[A-Za-z_][A-Za-z0-9_]*(?:[234](?:x[234])?)?)\.Differential$"
    )
    HLSL_NAMESPACE_PREFIX = "hlsl::"
    HLSL_NAMESPACE_SPECIAL_BUILTINS = {"mad", "mul", "rcp", "saturate", "sincos"}
    HLSL_NAMESPACE_BITCAST_BUILTINS = {"asfloat", "asint", "asuint"}
    HLSL_NAMESPACE_PASSTHROUGH_BUILTINS = {
        "abs",
        "acos",
        "acosh",
        "all",
        "any",
        "asin",
        "asinh",
        "atan",
        "atanh",
        "ceil",
        "clamp",
        "cos",
        "cosh",
        "cross",
        "degrees",
        "determinant",
        "distance",
        "dot",
        "exp",
        "exp2",
        "floor",
        "fwidth",
        "isfinite",
        "isinf",
        "isnan",
        "length",
        "log",
        "log2",
        "max",
        "min",
        "normalize",
        "pow",
        "radians",
        "reflect",
        "refract",
        "round",
        "sign",
        "sin",
        "sinh",
        "smoothstep",
        "sqrt",
        "step",
        "tan",
        "tanh",
        "transpose",
        "trunc",
    }
    SAMPLE_METHOD_MAP = {
        "Sample": "texture",
        "SampleBias": "texture",
        "SampleCmp": "textureCompare",
        "SampleCmpBias": "textureCompare",
        "SampleCmpLevel": "textureCompareLod",
        "SampleCmpLevelZero": "textureCompareLod",
        "SampleCmpGrad": "textureCompareGrad",
        "SampleLevel": "textureLod",
        "SampleLOD": "textureLod",
        "SampleGrad": "textureGrad",
        "Load": "texelFetch",
    }
    GATHER_METHOD_COMPONENTS = {
        "Gather": None,
        "GatherRed": "0",
        "GatherGreen": "1",
        "GatherBlue": "2",
        "GatherAlpha": "3",
    }
    GATHER_COMPARE_METHODS = {"GatherCmp", "GatherCmpRed"}
    LOD_QUERY_METHOD_COMPONENTS = {
        "CalculateLevelOfDetail": "x",
        "CalculateLevelOfDetailUnclamped": "y",
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
    BUFFER_RESOURCE_TYPES = {
        "StructuredBuffer",
        "RWStructuredBuffer",
        "ByteAddressBuffer",
        "RWByteAddressBuffer",
    }
    BYTE_ADDRESS_BUFFER_RESOURCE_TYPES = {
        "ByteAddressBuffer",
        "RWByteAddressBuffer",
    }
    CROSSGL_BUILTIN_TYPE_NAMES = {
        "void",
        "bool",
        "i8",
        "i16",
        "i32",
        "i64",
        "u8",
        "u16",
        "u32",
        "u64",
        "f16",
        "f32",
        "f64",
        "int",
        "uint",
        "float",
        "double",
        "half",
        "char",
        "string",
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
        "dvec2",
        "dvec3",
        "dvec4",
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
        "sampler",
        "image1D",
        "image1DArray",
        "image2D",
        "image2DArray",
        "image2DMS",
        "image2DMSArray",
        "image3D",
        "imageCube",
        "imageCubeArray",
        "iimage1D",
        "iimage1DArray",
        "iimage2D",
        "iimage2DArray",
        "iimage2DMS",
        "iimage2DMSArray",
        "iimage3D",
        "iimageCube",
        "iimageCubeArray",
        "uimage1D",
        "uimage1DArray",
        "uimage2D",
        "uimage2DArray",
        "uimage2DMS",
        "uimage2DMSArray",
        "uimage3D",
        "uimageCube",
        "uimageCubeArray",
    }
    BYTE_ADDRESS_BUFFER_METHOD_MAP = {
        "Load": "buffer_load",
        "Load2": "buffer_load2",
        "Load3": "buffer_load3",
        "Load4": "buffer_load4",
        "Store": "buffer_store",
        "Store2": "buffer_store2",
        "Store3": "buffer_store3",
        "Store4": "buffer_store4",
    }
    GET_DIMENSIONS_TEXTURE_DIMENSIONS = {
        "Texture1D": 1,
        "Texture1DArray": 2,
        "Texture2D": 2,
        "Texture2DArray": 3,
        "Texture3D": 3,
        "TextureCube": 2,
        "TextureCubeArray": 3,
        "Sampler1D": 1,
        "Sampler1DArray": 2,
        "Sampler2D": 2,
        "Sampler2DArray": 3,
        "Sampler3D": 3,
        "SamplerCube": 2,
        "SamplerCubeArray": 3,
        "Sampler2DShadow": 2,
        "Sampler2DArrayShadow": 3,
        "SamplerCubeShadow": 2,
        "SamplerCubeArrayShadow": 3,
    }
    GET_DIMENSIONS_IMAGE_DIMENSIONS = {
        "RWTexture1D": 1,
        "RWTexture1DArray": 2,
        "RWTexture2D": 2,
        "RWTexture2DArray": 3,
        "RWTexture3D": 3,
        "RWTextureCube": 2,
        "RWTextureCubeArray": 3,
    }
    GET_DIMENSIONS_MULTISAMPLE_DIMENSIONS = {
        "Texture2DMS": 2,
        "Texture2DMSArray": 3,
        "Sampler2DMS": 2,
        "Sampler2DMSArray": 3,
        "RWTexture2DMS": 2,
        "RWTexture2DMSArray": 3,
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
            "ddx": "dFdx",
            "ddx_coarse": "dFdxCoarse",
            "ddx_fine": "dFdxFine",
            "ddy": "dFdy",
            "ddy_coarse": "dFdyCoarse",
            "ddy_fine": "dFdyFine",
            "fwidth_coarse": "fwidthCoarse",
            "fwidth_fine": "fwidthFine",
            "frac": "fract",
            "fmod": "mod",
            "lerp": "mix",
            "rsqrt": "inversesqrt",
            "atan2": "atan",
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
        self.generated_temp_index = 0

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
            "TEXCOORD7": "TexCoord7",
            # Vertex inputs instance
            "FRONT_FACE": "gl_IsFrontFace",
            "PRIMITIVE_ID": "gl_PrimitiveID",
            "INSTANCE_ID": "gl_InstanceID",
            "VERTEX_ID": "gl_VertexID",
            "SV_VertexID": "gl_VertexID",
            "SV_InstanceID": "gl_InstanceID",
            "SV_PrimitiveID": "gl_PrimitiveID",
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
            "SV_IsFrontFace": "gl_FrontFacing",
            "SV_SampleIndex": "gl_SampleID",
            "SV_Barycentrics": "gl_BaryCoordEXT",
            # Fragment outputs
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
            # SV_Coverage is also a fragment input, handled contextually
            # for parameters in map_semantic.
            "SV_Coverage": "gl_SampleMask",
            "SV_ViewID": "gl_ViewID",
            "SV_RenderTargetArrayIndex": "gl_Layer",
            "SV_ViewportArrayIndex": "gl_ViewportIndex",
            "SV_ClipDistance": "gl_ClipDistance",
            "SV_CullDistance": "gl_CullDistance",
            # Compute shader
            "SV_GroupID": "gl_WorkGroupID",
            "SV_GroupThreadID": "gl_LocalInvocationID",
            "SV_DispatchThreadID": "gl_GlobalInvocationID",
            "SV_GroupIndex": "gl_LocalInvocationIndex",
        }
        self.hlsl_system_semantic_map = {
            semantic.lower(): mapped
            for semantic, mapped in self.semantic_map.items()
            if semantic.lower().startswith("sv_")
        }
        self.interpolation_qualifiers = {
            "centroid",
            "flat",
            "linear",
            "linear_centroid",
            "linear_noperspective",
            "linear_noperspective_centroid",
            "linear_sample",
            "nointerpolation",
            "noperspective",
            "pervertex",
            "sample",
            "smooth",
        }

    def generate(self, ast):
        self.raise_for_unsupported_conformance_constructs(ast)
        exported_functions = [
            exp.item
            for exp in getattr(ast, "exports", [])
            if isinstance(getattr(exp, "item", None), FunctionNode)
        ]
        extension_methods = self.collect_lowerable_extension_methods(ast)
        functions = [
            *getattr(ast, "functions", []),
            *extension_methods,
            *exported_functions,
        ]
        local_structs = self.collect_function_local_structs(functions)
        self.user_function_names = {getattr(func, "name", None) for func in functions}
        self.user_function_names.discard(None)
        self.function_name_map = self.collect_function_name_map(functions)
        resource_types = self.collect_sampleable_resource_types(ast)
        storage_image_types = self.collect_storage_image_resource_types(ast)
        self.struct_member_types = self.collect_struct_member_types(ast, local_structs)
        self.struct_member_name_maps = self.collect_struct_member_name_maps(
            ast, local_structs
        )
        self.identifier_rename_scopes = [{}]
        self.identifier_used_name_scopes = [set()]
        self.generated_temp_index = 0
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
            code += f"    {self.generate_typedef(node)}\n"
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
        for node in [*ast.structs, *local_structs]:
            if self.is_forward_struct_declaration(node):
                continue
            if isinstance(node, StructNode):
                code += f"    struct {self.format_struct_declaration_name(node)} {{\n"
                for member in node.members:
                    code += f"        {self.generate_struct_member(node, member)};\n"
                code += "    }\n"
        for node in ast.global_vars:
            code += self.generate_global_variable(node)
        if ast.cbuffers:
            code += "    // Constant Buffers\n"
            code += self.generate_cbuffers(ast)

        for func in [*ast.functions, *extension_methods]:
            qualifier = self.effective_function_qualifier(func)
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

    def generate_typedef(self, node):
        original_type = self.map_type(node.original_type)
        new_type = node.new_type
        if self.requires_typealias_spelling(original_type):
            return f"typealias {new_type} = {original_type};"
        return f"typedef {original_type} {new_type};"

    def requires_typealias_spelling(self, type_name):
        type_name = str(type_name)
        return (
            type_name not in self.CROSSGL_BUILTIN_TYPE_NAMES
            and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", type_name) is not None
        )

    def raise_for_unsupported_conformance_constructs(self, ast):
        constructs = []

        for interface in getattr(ast, "interfaces", []) or []:
            constructs.extend(self.format_interface_conformance_constructs(interface))

        for struct in getattr(ast, "structs", []) or []:
            constructs.extend(self.format_struct_conformance_constructs(struct))
            constructs.extend(
                self.format_typedef_generic_constraints(getattr(struct, "typedefs", []))
            )

        for extension in getattr(ast, "extensions", []) or []:
            conformances = getattr(extension, "conformances", []) or []
            if not self.is_lowerable_extension(extension):
                suffix = f" : {', '.join(conformances)}" if conformances else ""
                constructs.append(f"extension {extension.extended_type}{suffix}")
            constructs.extend(
                self.format_typedef_generic_constraints(
                    getattr(extension, "typedefs", [])
                )
            )
            for method in getattr(extension, "methods", []) or []:
                constructs.extend(self.format_function_generic_constraints(method))

        for function in getattr(ast, "functions", []) or []:
            constructs.extend(self.format_function_generic_constraints(function))

        constructs.extend(
            self.format_typedef_generic_constraints(getattr(ast, "typedefs", []))
        )

        for export in getattr(ast, "exports", []) or []:
            item = getattr(export, "item", None)
            if isinstance(item, InterfaceNode):
                constructs.extend(self.format_interface_conformance_constructs(item))
            elif isinstance(item, StructNode):
                constructs.extend(self.format_struct_conformance_constructs(item))
                constructs.extend(
                    self.format_typedef_generic_constraints(
                        getattr(item, "typedefs", [])
                    )
                )
            elif isinstance(item, ExtensionNode):
                conformances = getattr(item, "conformances", []) or []
                suffix = f" : {', '.join(conformances)}" if conformances else ""
                constructs.append(f"extension {item.extended_type}{suffix}")
            elif isinstance(item, FunctionNode):
                constructs.extend(self.format_function_generic_constraints(item))

        if constructs:
            raise UnsupportedSlangConformanceError(constructs)

    def format_interface_conformance_constructs(self, interface):
        constructs = []
        name = getattr(interface, "name", "anonymous")
        conformances = getattr(interface, "conformances", []) or []
        generic_parameters = getattr(interface, "generic_parameters", None)
        generic_constraints = getattr(interface, "generic_constraints", []) or []
        associated_types = getattr(interface, "associated_types", []) or []
        properties = getattr(interface, "properties", []) or []
        value_requirements = getattr(interface, "value_requirements", []) or []
        if (
            conformances
            or generic_parameters
            or generic_constraints
            or associated_types
            or properties
            or value_requirements
            or getattr(interface, "methods", None)
        ):
            suffix = f" : {', '.join(conformances)}" if conformances else ""
            constructs.append(f"interface {name}{suffix}")
        for method in getattr(interface, "methods", []) or []:
            constructs.extend(self.format_function_generic_constraints(method))
        for constraint in generic_constraints:
            if self.is_erased_generic_constraint(constraint):
                continue
            relation = getattr(constraint, "relation", ":")
            constructs.append(
                f"interface {name} where "
                f"{constraint.parameter} {relation} {constraint.constraint_type}"
            )
        return constructs

    def format_struct_conformance_constructs(self, struct):
        conformances = getattr(struct, "conformances", []) or []
        if not conformances:
            return []
        return [f"struct {struct.name} : {', '.join(conformances)}"]

    def format_typedef_generic_constraints(self, typedefs):
        constraints = []
        for typedef in typedefs or []:
            for constraint in getattr(typedef, "generic_constraints", []) or []:
                if self.is_erased_generic_constraint(constraint):
                    continue
                relation = getattr(constraint, "relation", ":")
                constraints.append(
                    f"typealias {typedef.new_type} where "
                    f"{constraint.parameter} {relation} {constraint.constraint_type}"
                )
        return constraints

    def format_function_generic_constraints(self, function):
        constraints = []
        for constraint in getattr(function, "generic_constraints", []) or []:
            if self.is_erased_generic_constraint(constraint):
                continue
            relation = getattr(constraint, "relation", ":")
            constraints.append(
                f"function {function.name} where "
                f"{constraint.parameter} {relation} {constraint.constraint_type}"
            )
        return constraints

    def collect_lowerable_extension_methods(self, ast):
        methods = []
        for extension in getattr(ast, "extensions", []) or []:
            if not self.is_lowerable_extension(extension):
                continue
            for method in getattr(extension, "methods", []) or []:
                if getattr(method, "is_declaration", False):
                    continue
                if getattr(method, "generic_constraints", None):
                    continue
                methods.append(method)
        return methods

    def is_lowerable_extension(self, extension):
        return not (
            getattr(extension, "conformances", None)
            or getattr(extension, "typedefs", None)
            or getattr(extension, "generic_parameters", None)
            or getattr(extension, "generic_constraints", None)
        )

    def is_erased_generic_constraint(self, constraint):
        relation = getattr(constraint, "relation", ":")
        parameter = getattr(constraint, "parameter", "")
        constraint_type = getattr(constraint, "constraint_type", "")
        if relation == ":" and constraint_type.startswith("__Builtin"):
            return True
        if relation == "==" and parameter == f"{constraint_type}.Differential":
            return True
        return relation == "==" and self.is_plain_generic_type_parameter(parameter)

    def is_plain_generic_type_parameter(self, parameter):
        return bool(re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", str(parameter)))

    def is_forward_struct_declaration(self, node):
        return isinstance(node, StructNode) and getattr(
            node, "is_forward_declaration", False
        )

    def effective_function_qualifier(self, func):
        qualifier = normalize_stage_name(getattr(func, "qualifier", None))
        if qualifier:
            return qualifier
        if getattr(func, "numthreads", None):
            return "compute"
        return qualifier

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
        for instance in self.glsl_block_instances(ast):
            self.register_identifier_declaration(instance)

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

    def format_struct_declaration_name(self, node):
        generic_parameters = ""
        if getattr(node, "generic_parameters_from_prefix", False):
            generic_parameters = getattr(node, "generic_parameters", None) or ""
        return f"{self.format_identifier(node.name)}{generic_parameters}"

    def collect_struct_member_name_maps(self, ast, local_structs=None):
        maps = {}
        for struct in getattr(ast, "structs", []) or []:
            self.collect_struct_member_name_map_from_node(struct, maps)
        for struct in local_structs or []:
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
            if self.is_forward_struct_declaration(item):
                return ""
            code = f"    struct {self.format_struct_declaration_name(item)} {{\n"
            for member in item.members:
                code += f"        {self.generate_struct_member(item, member)};\n"
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
                if self.is_glsl_storage_block(node):
                    code += self.generate_glsl_storage_block(node)
                    continue
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

    def is_glsl_storage_block(self, node):
        return getattr(node, "glsl_block_kind", None) == "buffer"

    def generate_glsl_storage_block(self, node):
        code = f"    struct {self.format_struct_declaration_name(node)} {{\n"
        for member in node.members:
            code += f"        {self.generate_struct_member(node, member)};\n"
        code += "    }\n"
        for instance in getattr(node, "instances", []) or []:
            code += f"    {self.generate_variable_declaration(instance)};\n"
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
        function_qualifier = self.effective_function_qualifier(func)
        preserve_parameter_qualifiers = self.is_entry_like_function(func)
        unnamed_parameter_names = self.collect_unnamed_parameter_names(func.params)
        params = ", ".join(
            self.generate_parameter(
                p,
                preserve_parameter_qualifiers,
                function_qualifier,
                unnamed_parameter_names.get(id(p)),
            )
            for p in func.params
        )
        semantic = self.map_semantic(getattr(func, "semantic", None))
        semantic_suffix = f" {semantic}" if semantic else ""
        if self.should_preserve_slang_stage_entry_name(func, function_qualifier):
            semantic_suffix += " @ stage_entry"
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

    def collect_unnamed_parameter_names(self, params):
        names = {}
        used_names = self.identifier_used_name_scopes[-1]
        for index, param in enumerate(params or []):
            if getattr(param, "name", None):
                continue
            safe_name = self.sanitize_crossgl_identifier(f"_param{index}", used_names)
            used_names.add(safe_name)
            names[id(param)] = safe_name
        return names

    def generate_parameter(
        self,
        param,
        preserve_qualifiers=False,
        function_qualifier=None,
        fallback_name=None,
    ):
        qualifier_prefix = self.format_parameter_qualifier_prefix(
            param, preserve_qualifiers
        )
        parameter_name = param.name or fallback_name or "_param"
        parameter_type = self.map_parameter_type(param.vtype, function_qualifier)
        array_suffix = self.format_array_suffixes(param)
        array_suffix += self.non_tessellation_patch_parameter_array_suffix(
            param.vtype, function_qualifier
        )
        parameter = (
            f"{qualifier_prefix}{parameter_type} "
            f"{self.format_identifier(parameter_name)}"
            f"{array_suffix}"
        )
        metadata = self.format_parameter_metadata(param)
        if metadata:
            parameter += metadata
        semantic = self.map_semantic(
            param.semantic,
            function_qualifier=function_qualifier,
            parameter=param,
        )
        if semantic:
            parameter += f" {semantic}"
        return parameter

    def format_parameter_metadata(self, param):
        register_name = getattr(param, "register", None)
        if not self.should_emit_register_metadata(register_name, param):
            return ""
        register_arguments = self.format_register_metadata_arguments(register_name)
        return f" @register({register_arguments})"

    def is_entry_like_function(self, func):
        return bool(
            self.effective_function_qualifier(func)
            or getattr(func, "semantic", None)
            or getattr(func, "name", None) == "main"
        )

    def should_preserve_slang_stage_entry_name(self, func, function_qualifier):
        return bool(
            function_qualifier == "compute" and getattr(func, "name", None) != "main"
        )

    def format_parameter_qualifier_prefix(self, param, preserve_qualifiers=False):
        if not preserve_qualifiers:
            return ""
        allowed_qualifiers = {"in", "out", "inout"} | self.interpolation_qualifiers
        qualifiers = [
            str(qualifier)
            for qualifier in getattr(param, "qualifiers", []) or []
            if str(qualifier).lower() in allowed_qualifiers
        ]
        return f"{' '.join(qualifiers)} " if qualifiers else ""

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

    def generate_function_body(
        self, body, indent=0, is_main=False, deferred_scopes=None
    ):
        code = ""
        deferred_scopes = deferred_scopes or []
        deferred_statements = []
        for stmt in body:
            if isinstance(stmt, DeferNode):
                deferred_statements.append(stmt.body)
                continue
            if isinstance(stmt, StructNode) and getattr(
                stmt, "is_local_declaration", False
            ):
                continue
            expanded_statement = self.generate_resource_get_dimensions_statement(
                stmt, indent, is_main
            )
            if expanded_statement is not None:
                code += expanded_statement
                continue
            sincos_statement = self.generate_sincos_statement(stmt, indent, is_main)
            if sincos_statement is not None:
                code += sincos_statement
                continue
            if isinstance(stmt, ReturnNode):
                code += self.generate_statement_label(stmt, indent)
                code += self.generate_deferred_scope_exits(
                    [*deferred_scopes, deferred_statements], indent, is_main
                )
                code += "    " * indent
                if not is_main:
                    if stmt.value is None:
                        code += "return;\n"
                    else:
                        code += (
                            f"return {self.generate_expression(stmt.value, is_main)};\n"
                        )
                continue
            code += self.generate_statement_label(stmt, indent)
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
            elif isinstance(stmt, ForNode):
                code += self.generate_for_loop(
                    stmt, indent, is_main, [*deferred_scopes, deferred_statements]
                )
            elif isinstance(stmt, WhileNode):
                code += self.generate_while_loop(
                    stmt, indent, is_main, [*deferred_scopes, deferred_statements]
                )
            elif isinstance(stmt, DoWhileNode):
                code += self.generate_do_while_loop(
                    stmt, indent, is_main, [*deferred_scopes, deferred_statements]
                )
            elif isinstance(stmt, SwitchNode):
                code += self.generate_switch_statement(
                    stmt, indent, is_main, [*deferred_scopes, deferred_statements]
                )
            elif isinstance(stmt, IfNode):
                code += self.generate_if_statement(
                    stmt, indent, is_main, [*deferred_scopes, deferred_statements]
                )
            elif isinstance(stmt, BreakNode):
                code += self.generate_jump_statement("break", stmt) + "\n"
            elif isinstance(stmt, ContinueNode):
                code += self.generate_jump_statement("continue", stmt) + "\n"
            elif isinstance(stmt, DiscardNode):
                code += "discard;\n"
        code += self.generate_deferred_bodies(deferred_statements, indent, is_main)
        return code

    def generate_statement_label(self, stmt, indent=0):
        label = getattr(stmt, "label", None)
        if not label:
            return ""
        return "    " * indent + f"{self.format_label_identifier(label)}:\n"

    def generate_jump_statement(self, keyword, node):
        target_label = getattr(node, "target_label", None)
        if not target_label:
            return f"{keyword};"
        return f"{keyword} {self.format_label_identifier(target_label)};"

    def format_label_identifier(self, label):
        return self.sanitize_crossgl_identifier(str(label), set())

    def generate_deferred_scope_exits(self, deferred_scopes, indent=0, is_main=False):
        code = ""
        for deferred_statements in reversed(deferred_scopes or []):
            code += self.generate_deferred_bodies(deferred_statements, indent, is_main)
        return code

    def generate_deferred_bodies(self, deferred_statements, indent=0, is_main=False):
        code = ""
        for deferred_body in reversed(deferred_statements or []):
            code += self.generate_function_body(deferred_body, indent, is_main)
        return code

    def generate_sincos_statement(self, stmt, indent=0, is_main=False):
        if not isinstance(stmt, FunctionCallNode):
            return None
        function_name, _ = self.canonical_function_call_name(stmt.name)
        if function_name != "sincos":
            return None
        if len(stmt.args) != 3:
            return None
        value = self.generate_expression(stmt.args[0], is_main)
        sine_target = self.generate_expression(stmt.args[1], is_main)
        cosine_target = self.generate_expression(stmt.args[2], is_main)
        indent_text = "    " * indent
        return (
            f"{indent_text}{sine_target} = sin({value});\n"
            f"{indent_text}{cosine_target} = cos({value});\n"
        )

    def generate_for_loop(self, node, indent, is_main, deferred_scopes=None):
        init = self.generate_for_clause(node.init, is_main)
        condition = self.generate_for_clause(node.condition, is_main)
        update = self.generate_for_clause(node.update, is_main)

        code = f"for ({init}; {condition}; {update}) {{\n"
        code += self.generate_function_body(
            node.body, indent + 1, is_main, deferred_scopes
        )
        code += "    " * indent + "}\n"
        return code

    def generate_for_clause(self, clause, is_main):
        if clause is None:
            return ""
        if isinstance(clause, list):
            return ", ".join(self.generate_expression(item, is_main) for item in clause)
        return self.generate_expression(clause, is_main)

    def generate_while_loop(self, node, indent, is_main, deferred_scopes=None):
        condition = self.generate_expression(node.condition, is_main)

        code = f"while ({condition}) {{\n"
        code += self.generate_function_body(
            node.body, indent + 1, is_main, deferred_scopes
        )
        code += "    " * indent + "}\n"
        return code

    def generate_do_while_loop(self, node, indent, is_main, deferred_scopes=None):
        condition = self.generate_expression(node.condition, is_main)

        code = "do {\n"
        code += self.generate_function_body(
            node.body, indent + 1, is_main, deferred_scopes
        )
        code += "    " * indent + f"}} while ({condition});\n"
        return code

    def generate_switch_statement(self, node, indent, is_main, deferred_scopes=None):
        expression = self.generate_expression(node.expression, is_main)

        code = f"switch ({expression}) {{\n"
        unlabeled_statements = getattr(node, "unlabeled_statements", []) or []
        if unlabeled_statements:
            code += self.generate_function_body(
                unlabeled_statements, indent + 1, is_main, deferred_scopes
            )
        ordered_cases = getattr(node, "ordered_cases", None)
        if ordered_cases is not None:
            for case in ordered_cases:
                if case.value is None:
                    code += "    " * (indent + 1) + "default:\n"
                else:
                    value = self.generate_expression(case.value, is_main)
                    code += "    " * (indent + 1) + f"case {value}:\n"
                code += self.generate_function_body(
                    case.body, indent + 2, is_main, deferred_scopes
                )
            code += "    " * indent + "}\n"
            return code

        for case in node.cases:
            value = self.generate_expression(case.value, is_main)
            code += "    " * (indent + 1) + f"case {value}:\n"
            code += self.generate_function_body(
                case.body, indent + 2, is_main, deferred_scopes
            )

        if node.default_case is not None:
            code += "    " * (indent + 1) + "default:\n"
            code += self.generate_function_body(
                node.default_case, indent + 2, is_main, deferred_scopes
            )

        code += "    " * indent + "}\n"
        return code

    def generate_if_statement(self, node, indent, is_main, deferred_scopes=None):
        binding = getattr(node, "if_binding", None)
        code = ""
        if binding is not None:
            code += self.generate_assignment(binding, is_main) + ";\n"
            code += "    " * indent

        condition = self.generate_expression(node.condition, is_main)

        code += f"if ({condition}) {{\n"
        code += self.generate_function_body(
            node.if_body, indent + 1, is_main, deferred_scopes
        )
        code += "    " * indent + "}"

        if node.else_body:
            if isinstance(node.else_body, IfNode):
                code += " else "
                code += self.generate_if_statement(
                    node.else_body, indent, is_main, deferred_scopes
                )
            else:
                code += " else {\n"
                code += self.generate_function_body(
                    node.else_body, indent + 1, is_main, deferred_scopes
                )
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
            left = self.generate_variable_declaration(
                node.left, has_initializer=True, initializer=node.right
            )
            right = self.generate_expression(node.right, False)
            return f"    {left} {node.operator} {right};\n"
        return f"    {self.generate_variable_declaration(node)};\n"

    def generate_variable_declaration(
        self, node, has_initializer=False, initializer=None
    ):
        qualifiers = self.format_global_variable_qualifiers(
            node, has_initializer=has_initializer
        )
        if qualifiers:
            qualifiers += " "
        variable_type = self.resolve_variable_declaration_type(node, initializer)
        return (
            f"{qualifiers}{self.map_type(variable_type)} "
            f"{self.format_identifier(node.name)}{self.format_array_suffixes(node)}"
            f"{self.format_variable_metadata(node)}"
        )

    def resolve_variable_declaration_type(self, node, initializer=None):
        vtype = getattr(node, "vtype", None)
        storage_modifier = getattr(node, "storage_modifier", None)
        if vtype in {"let", "var"} and storage_modifier in {"let", "var"}:
            inferred_type = self.infer_type_from_initializer(initializer)
            if inferred_type is not None:
                return inferred_type
        return vtype

    def infer_type_from_initializer(self, initializer):
        if isinstance(initializer, VectorConstructorNode):
            return initializer.type_name
        return None

    def format_global_variable_qualifiers(self, node, has_initializer=False):
        allowed_qualifiers = {"extern", "static", "const", "constexpr"}
        excluded_qualifiers = set()
        if not has_initializer and self.is_plain_uninitialized_const_global(node):
            # CrossGL requires plain top-level const declarations to carry an
            # initializer; static const declarations are accepted and can keep
            # their original storage semantics.
            excluded_qualifiers = {"const"}
        return self.format_ordered_storage_qualifiers(
            node, allowed_qualifiers, excluded_qualifiers=excluded_qualifiers
        )

    def is_plain_uninitialized_const_global(self, node):
        qualifiers = {
            str(qualifier).lower()
            for qualifier in getattr(node, "qualifiers", []) or []
        }
        return "const" in qualifiers and "static" not in qualifiers

    def generate_struct_member(self, struct_node, member):
        qualifiers = self.format_struct_member_qualifiers(member)
        if qualifiers:
            qualifiers += " "
        member_name = self.format_struct_member_name(struct_node, member.name)
        declaration = (
            f"{qualifiers}{self.map_type(member.vtype)} {member_name}"
            f"{self.format_array_suffixes(member)}"
        )
        if getattr(member, "value", None) is not None:
            declaration += f" = {self.generate_expression(member.value)}"
        semantic = self.map_semantic(member.semantic)
        if semantic:
            declaration += f" {semantic}"
        return declaration

    def format_struct_member_qualifiers(self, member):
        allowed_qualifiers = {"static", "const"} | self.interpolation_qualifiers
        return self.format_ordered_storage_qualifiers(member, allowed_qualifiers)

    def format_ordered_storage_qualifiers(
        self, node, allowed_qualifiers, excluded_qualifiers=None
    ):
        allowed = {str(qualifier).lower() for qualifier in allowed_qualifiers}
        excluded = {
            str(qualifier).lower() for qualifier in excluded_qualifiers or set()
        }
        qualifiers = [
            (index, str(qualifier))
            for index, qualifier in enumerate(getattr(node, "qualifiers", []) or [])
            if str(qualifier).lower() in allowed
            and str(qualifier).lower() not in excluded
        ]
        storage_order = {"extern": 0, "static": 1, "const": 2, "constexpr": 3}
        return " ".join(
            qualifier
            for _index, qualifier in sorted(
                qualifiers,
                key=lambda item: (
                    storage_order.get(item[1].lower(), len(storage_order)),
                    item[0],
                ),
            )
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
            qualifier_name = str(qualifier).strip().lower()
            if qualifier_name == "in":
                append_metadata("stage_io", "@input")
            elif qualifier_name == "out":
                append_metadata("stage_io", "@output")

            for name, value in self.layout_metadata_entries(qualifier):
                if name in {"set", "group"} and value:
                    append_metadata("set", f"@set({value})")
                elif name == "binding" and value:
                    append_metadata("binding", f"@binding({value})")
                elif name == "location" and value:
                    append_metadata("location", f"@location({value})")
                elif name == "index" and value:
                    append_metadata("index", f"@index({value})")
                elif name == "push_constant":
                    append_metadata("push_constant", "@push_constant")

        if getattr(node, "glsl_block_kind", None) == "buffer":
            layout = self.glsl_buffer_block_layout(node)
            append_metadata("glsl_buffer_block", f"@glsl_buffer_block({layout})")

        for attribute in getattr(node, "attributes", []) or []:
            name = str(attribute.get("name", "")).lower()
            arguments = attribute.get("arguments", [])
            if name == "vk::binding" and arguments:
                if len(arguments) > 1:
                    append_metadata("set", f"@set({arguments[1]})")
                append_metadata("binding", f"@binding({arguments[0]})")
            elif name == "vk::location" and arguments:
                append_metadata("location", f"@location({arguments[0]})")
            elif name == "vk::index" and arguments:
                append_metadata("index", f"@index({arguments[0]})")
            elif name == "vk::push_constant":
                append_metadata("push_constant", "@push_constant")

        register_name = getattr(node, "register", None)
        if self.should_emit_register_metadata(register_name, node):
            register_arguments = self.format_register_metadata_arguments(register_name)
            append_metadata("register", f"@register({register_arguments})")

        if not metadata:
            return ""
        return " " + " ".join(metadata)

    def glsl_buffer_block_layout(self, node):
        for qualifier in getattr(node, "qualifiers", []) or []:
            for name, _value in self.layout_metadata_entries(qualifier):
                if name in {"std140", "std430", "scalar"}:
                    return name
        return "std430"

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
            if char in "([{<":
                depth += 1
            elif char in ")]}>" and depth > 0:
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
        if getattr(node, "buffer_kind", None) in {"cbuffer", "tbuffer"}:
            return True
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

    def generate_hlsl_mul_call(self, expr, is_main, function_name=None):
        function_name = function_name or expr.name
        if (
            function_name != "mul"
            or len(expr.args) != 2
            or expr.name in self.user_function_names
        ):
            return None

        left = self.generate_expression(expr.args[0], is_main)
        right = self.generate_expression(expr.args[1], is_main)
        left = self.maybe_parenthesize_expression(expr.args[0], left)
        right = self.maybe_parenthesize_expression(expr.args[1], right)
        return f"({left} * {right})"

    def generate_mad_call(self, expr, is_main, function_name=None):
        function_name = function_name or expr.name
        if (
            function_name != "mad"
            or len(expr.args) != 3
            or expr.name in self.user_function_names
        ):
            return None

        multiplier = self.generate_expression(expr.args[0], is_main)
        multiplicand = self.generate_expression(expr.args[1], is_main)
        addend = self.generate_expression(expr.args[2], is_main)
        multiplier = self.maybe_parenthesize_expression(expr.args[0], multiplier)
        multiplicand = self.maybe_parenthesize_expression(expr.args[1], multiplicand)
        addend = self.maybe_parenthesize_expression(expr.args[2], addend)
        return f"(({multiplier} * {multiplicand}) + {addend})"

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
            return self.format_expression_identifier(expr.name)
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
            function_name, is_hlsl_namespace_builtin = (
                self.canonical_function_call_name(expr.name)
            )
            mul_call = self.generate_hlsl_mul_call(expr, is_main, function_name)
            if mul_call is not None:
                return mul_call
            mad_call = self.generate_mad_call(expr, is_main, function_name)
            if mad_call is not None:
                return mad_call
            rendered_args = [
                self.generate_expression(arg, is_main) for arg in expr.args
            ]
            bitcast_call = self.generate_bitcast_intrinsic_call(
                function_name,
                expr.args,
                rendered_args,
                is_hlsl_namespace_builtin or expr.name not in self.user_function_names,
            )
            if bitcast_call is not None:
                return bitcast_call
            args = ", ".join(rendered_args)
            if (
                function_name == "saturate"
                and len(expr.args) == 1
                and (
                    is_hlsl_namespace_builtin
                    or expr.name not in self.user_function_names
                )
            ):
                return f"clamp({args}, 0.0, 1.0)"
            if (
                function_name == "rcp"
                and len(expr.args) == 1
                and (
                    is_hlsl_namespace_builtin
                    or expr.name not in self.user_function_names
                )
            ):
                value = self.generate_expression(expr.args[0], is_main)
                value = self.maybe_parenthesize_expression(expr.args[0], value)
                return f"(1.0 / {value})"
            generic_type_name = self.map_generic_vector_or_matrix_type(function_name)
            if not is_hlsl_namespace_builtin and expr.name in self.user_function_names:
                name = self.format_function_name(expr.name)
            elif generic_type_name:
                name = generic_type_name
            elif function_name in self.type_map and function_name[:1].islower():
                name = self.map_type(function_name)
            else:
                name = self.function_map.get(function_name, function_name)
            return f"{name}({args})"
        elif isinstance(expr, MethodCallNode):
            obj = self.generate_expression(expr.object, is_main)
            image_call = self.generate_storage_image_method_call(expr, obj, is_main)
            if image_call is not None:
                return image_call

            byte_address_buffer_call = self.generate_byte_address_buffer_method_call(
                expr, obj, is_main
            )
            if byte_address_buffer_call is not None:
                return byte_address_buffer_call

            texture_call = self.generate_texture_method_call(expr, obj, is_main)
            if texture_call is not None:
                return texture_call

            buffer_call = self.generate_structured_buffer_method_call(
                expr, obj, is_main
            )
            if buffer_call is not None:
                return buffer_call

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
            qualified_name = self.member_access_qualified_name(expr)
            if qualified_name and self.lookup_variable_type(qualified_name):
                return self.format_identifier(qualified_name)
            obj = self.generate_expression(expr.object, is_main)
            if isinstance(
                expr.object, (AssignmentNode, BinaryOpNode, TernaryOpNode, UnaryOpNode)
            ):
                obj = f"({obj})"
            return f"{obj}.{self.format_member_access_name(expr)}"
        elif isinstance(expr, ArrayAccessNode):
            array = self.generate_expression(expr.array, is_main)
            index = self.generate_subscript_index(expr.index, is_main)
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

    def generate_subscript_index(self, index, is_main=False):
        if index is None:
            return ""
        if isinstance(index, ParenthesizedCommaNode):
            return ", ".join(
                self.generate_expression(item, is_main) for item in index.expressions
            )
        return self.generate_expression(index, is_main)

    def format_expression_identifier(self, name):
        name = self.format_identifier(name)
        return self.normalize_scoped_generic_expression_identifier(name)

    def normalize_scoped_generic_expression_identifier(self, name):
        return re.sub(r"(?<=>)::(?=[A-Za-z_])", ".", str(name))

    def normalize_numeric_literal(self, value):
        if not isinstance(value, str):
            return value
        hex_float_match = self.HEX_FLOAT_NUMERIC_LITERAL.match(value)
        if hex_float_match:
            return self.normalize_hex_float_literal(hex_float_match.group("body"))

        hex_match = self.HEX_NUMERIC_LITERAL.match(value)
        if hex_match:
            return self.normalize_integer_suffix(
                hex_match.group("body"), hex_match.group("suffix")
            )

        match = self.DECIMAL_NUMERIC_LITERAL.match(value)
        if not match:
            return value

        body = match.group("body")
        suffix = match.group("suffix")
        float_suffix = any(char in "fFhH" for char in suffix)
        integer_suffix = "".join(char for char in suffix if char not in "fFhH")
        float_like = "." in body or "e" in body.lower() or float_suffix
        if not float_like and integer_suffix:
            return self.normalize_integer_suffix(body, integer_suffix)
        if not float_like or integer_suffix:
            return value

        try:
            normalized = format(Decimal(body), "f")
        except InvalidOperation:
            return value

        if "." not in normalized:
            normalized = f"{normalized}.0"
        return normalized

    def normalize_hex_float_literal(self, body):
        try:
            normalized = format(Decimal.from_float(float.fromhex(body)), "f")
        except (InvalidOperation, OverflowError, ValueError):
            return body
        if "." not in normalized:
            normalized = f"{normalized}.0"
        return normalized

    def normalize_integer_suffix(self, body, suffix):
        if not suffix:
            return body
        return f"{body}u" if "u" in suffix.lower() else body

    def map_type(self, slang_type):
        """Map a Slang type name to the closest CrossGL type name."""
        if slang_type:
            slang_type = slang_type.strip()
            pointer_suffix = ""
            while slang_type.endswith("*"):
                pointer_suffix += "*"
                slang_type = slang_type[:-1].strip()
            differential_type = self.map_associated_differential_type(slang_type)
            if differential_type:
                return f"{differential_type}{pointer_suffix}"
            generic_vector_or_matrix = self.map_generic_vector_or_matrix_type(
                slang_type
            )
            if generic_vector_or_matrix:
                return f"{generic_vector_or_matrix}{pointer_suffix}"
            base_type = slang_type.split("<", 1)[0].strip()
            pointer_buffer_type = self.map_pointer_element_buffer_type(
                slang_type, base_type
            )
            if pointer_buffer_type:
                return f"{pointer_buffer_type}{pointer_suffix}"
            mapped_type = self.type_map.get(
                slang_type, self.type_map.get(base_type, slang_type)
            )
            mapped_type = self.sanitize_generic_type_expression_arguments(mapped_type)
            return f"{mapped_type}{pointer_suffix}"
        return slang_type

    def sanitize_generic_type_expression_arguments(self, type_name):
        text = str(type_name)
        if "<" not in text:
            return type_name

        start = text.find("<")
        if start < 0 or not text.endswith(">"):
            return re.sub(r"!(?=[A-Za-z_])", "not_", text)

        arguments = self.generic_type_arguments(text)
        if not arguments:
            return re.sub(r"!(?=[A-Za-z_])", "not_", text)

        base_type = text[:start]
        sanitized_arguments = [
            self.sanitize_generic_type_argument(argument) for argument in arguments
        ]
        return f"{base_type}<{', '.join(sanitized_arguments)}>"

    def sanitize_generic_type_argument(self, argument):
        argument = re.sub(r"!(?=[A-Za-z_])", "not_", str(argument).strip())
        if "<" in argument:
            argument = self.sanitize_generic_type_expression_arguments(argument)
        if not self.requires_generic_value_argument_encoding(argument):
            return argument
        return self.encode_generic_value_argument(argument)

    def requires_generic_value_argument_encoding(self, argument):
        return re.search(r"[()+\-*/%]", str(argument)) is not None

    def encode_generic_value_argument(self, argument):
        text = str(argument).strip()
        replacements = [
            ("&&", "_and_"),
            ("||", "_or_"),
            ("!=", "_ne_"),
            ("==", "_eq_"),
            ("<=", "_le_"),
            (">=", "_ge_"),
            ("::", "_"),
            ("+", "_plus_"),
            ("-", "_minus_"),
            ("*", "_mul_"),
            ("/", "_div_"),
            ("%", "_mod_"),
            ("<", "_lt_"),
            (">", "_gt_"),
            ("!", "not_"),
        ]
        for old, new in replacements:
            text = text.replace(old, new)
        text = re.sub(r"[^A-Za-z0-9_]+", "_", text)
        text = re.sub(r"_+", "_", text).strip("_")
        if not text or not re.match(r"[A-Za-z_]", text):
            text = f"value_{text}" if text else "value"
        elif not text.startswith("value_"):
            text = f"value_{text}"
        return text

    def map_pointer_element_buffer_type(self, slang_type, base_type):
        if base_type not in {"StructuredBuffer", "RWStructuredBuffer"}:
            return None
        args = self.generic_type_arguments(slang_type)
        if len(args) != 1 or "*" not in args[0]:
            return None
        return f"buffer<{self.map_type(args[0])}>"

    def map_associated_differential_type(self, slang_type):
        match = self.ASSOCIATED_DIFFERENTIAL_TYPE.fullmatch(str(slang_type))
        if not match:
            return None
        base_type = match.group("base")
        if base_type not in self.type_map and base_type != "half":
            return None
        return self.map_type(base_type)

    def map_generic_vector_or_matrix_type(self, slang_type):
        base_type = str(slang_type).split("<", 1)[0].strip()
        if base_type == "vector":
            return self.map_generic_vector_type(slang_type)
        if base_type == "matrix":
            return self.map_generic_matrix_type(slang_type)
        return None

    def map_generic_vector_type(self, slang_type):
        args = self.generic_type_arguments(slang_type)
        if len(args) != 2:
            return None

        scalar_type = args[0].strip()
        vector_size = args[1].strip()
        if vector_size not in {"1", "2", "3", "4"}:
            return None

        if vector_size == "1":
            return self.type_map.get(scalar_type, scalar_type)

        vector_prefixes = {
            "float": "vec",
            "half": "half",
            "int": "ivec",
            "uint": "uvec",
            "bool": "bvec",
            "double": "dvec",
        }
        prefix = vector_prefixes.get(scalar_type)
        if prefix is None:
            return None
        return f"{prefix}{vector_size}"

    def map_generic_matrix_type(self, slang_type):
        args = self.generic_type_arguments(slang_type)
        if len(args) != 3:
            return None

        scalar_type = args[0].strip()
        rows = args[1].strip()
        columns = args[2].strip()
        if rows not in {"1", "2", "3", "4"} or columns not in {"1", "2", "3", "4"}:
            return None
        if rows != columns:
            return None

        matrix_prefixes = {
            "float": "mat",
            "double": "dmat",
        }
        prefix = matrix_prefixes.get(scalar_type)
        if prefix is None:
            return None
        return f"{prefix}{rows}"

    def map_parameter_type(self, slang_type, function_qualifier=None):
        """Map Slang parameter wrapper types to CrossGL value/resource types."""
        patch_element_type = self.non_tessellation_patch_parameter_element_type(
            slang_type, function_qualifier
        )
        if patch_element_type:
            return self.map_type(patch_element_type)
        return self.map_type(self.unwrap_resource_container_type(slang_type))

    def non_tessellation_patch_parameter_array_suffix(
        self, slang_type, function_qualifier
    ):
        patch_type = self.non_tessellation_patch_parameter_type_info(
            slang_type, function_qualifier
        )
        if patch_type is None:
            return ""
        _element_type, control_points = patch_type
        return f"[{control_points}]" if control_points else ""

    def non_tessellation_patch_parameter_element_type(
        self, slang_type, function_qualifier
    ):
        patch_type = self.non_tessellation_patch_parameter_type_info(
            slang_type, function_qualifier
        )
        if patch_type is None:
            return None
        element_type, _control_points = patch_type
        return element_type

    def non_tessellation_patch_parameter_type_info(
        self, slang_type, function_qualifier
    ):
        if function_qualifier in {"tessellation_control", "tessellation_evaluation"}:
            return None
        base_type = str(slang_type or "").split("<", 1)[0].strip()
        if base_type not in {"InputPatch", "OutputPatch"}:
            return None
        arguments = self.generic_type_arguments(slang_type)
        if not arguments:
            return None
        element_type = arguments[0]
        control_points = arguments[1] if len(arguments) > 1 else None
        return element_type, control_points

    def canonical_function_call_name(self, name):
        """Return the CrossGL-facing name for known namespace-qualified builtins."""
        raw_name = str(name)
        base_name = self.method_base_name(raw_name)
        if base_name.startswith(self.HLSL_NAMESPACE_PREFIX):
            unqualified_name = base_name[len(self.HLSL_NAMESPACE_PREFIX) :]
            if self.is_known_hlsl_namespace_builtin(unqualified_name):
                return unqualified_name, True
        return raw_name, False

    def is_known_hlsl_namespace_builtin(self, name):
        return (
            name in self.HLSL_NAMESPACE_SPECIAL_BUILTINS
            or name in self.HLSL_NAMESPACE_BITCAST_BUILTINS
            or name in self.HLSL_NAMESPACE_PASSTHROUGH_BUILTINS
            or name in self.function_map
        )

    def generate_bitcast_intrinsic_call(
        self, function_name, original_args, rendered_args, allow_builtin_lowering
    ):
        if (
            not allow_builtin_lowering
            or function_name not in self.HLSL_NAMESPACE_BITCAST_BUILTINS
            or len(original_args) != 1
        ):
            return None

        source_family = self.numeric_type_family(self.expression_type(original_args[0]))
        if function_name == "asfloat":
            if source_family == "uint":
                return f"uintBitsToFloat({rendered_args[0]})"
            if source_family == "int":
                return f"intBitsToFloat({rendered_args[0]})"
            if source_family == "float":
                return rendered_args[0]
        elif function_name == "asint" and source_family == "float":
            return f"floatBitsToInt({rendered_args[0]})"
        elif function_name == "asuint" and source_family == "float":
            return f"floatBitsToUint({rendered_args[0]})"
        return None

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
        for instance in self.glsl_block_instances(ast):
            name = getattr(instance, "name", None)
            vtype = getattr(instance, "vtype", None)
            if name is not None and vtype:
                variables[name] = vtype
        return variables

    def glsl_block_instances(self, ast):
        for block in getattr(ast, "cbuffers", []) or []:
            yield from getattr(block, "instances", []) or []

    def collect_struct_member_types(self, ast, local_structs=None):
        structs = {}
        for struct in getattr(ast, "structs", []) or []:
            self.collect_struct_member_types_from_node(struct, structs)
        for struct in local_structs or []:
            self.collect_struct_member_types_from_node(struct, structs)
        for cbuffer in getattr(ast, "cbuffers", []) or []:
            if isinstance(cbuffer, StructNode):
                self.collect_struct_member_types_from_node(cbuffer, structs)
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

    def collect_function_local_structs(self, functions):
        structs = []
        seen = set()

        def visit_statement(statement):
            if isinstance(statement, StructNode):
                if getattr(statement, "is_local_declaration", False):
                    marker = id(statement)
                    if marker not in seen:
                        seen.add(marker)
                        structs.append(statement)
                return
            if isinstance(statement, IfNode):
                visit_statements(statement.if_body)
                visit_statements(statement.else_body or [])
            elif isinstance(statement, ForNode):
                visit_statements(statement.body)
            elif isinstance(statement, WhileNode):
                visit_statements(statement.body)
            elif isinstance(statement, DoWhileNode):
                visit_statements(statement.body)
            elif isinstance(statement, SwitchNode):
                for case in getattr(statement, "ordered_cases", []) or []:
                    visit_statements(case.body)
                visit_statements(getattr(statement, "unlabeled_statements", []) or [])
            elif isinstance(statement, DeferNode):
                visit_statements(statement.body)

        def visit_statements(statements):
            if not statements:
                return
            if not isinstance(statements, list):
                statements = [statements]
            for statement in statements:
                visit_statement(statement)

        for function in functions or []:
            visit_statements(getattr(function, "body", []) or [])
        return structs

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

    def is_buffer_resource_type(self, type_name):
        if not type_name:
            return False
        base_type = str(type_name).strip().split("<", 1)[0].strip()
        return base_type in self.BUFFER_RESOURCE_TYPES

    def expression_type(self, expr):
        if isinstance(expr, VariableNode):
            if expr.vtype:
                return expr.vtype
            return self.lookup_variable_type(expr.name)
        if isinstance(expr, ArrayAccessNode):
            return self.expression_type(expr.array)
        if isinstance(expr, MemberAccessNode):
            qualified_name = self.member_access_qualified_name(expr)
            if qualified_name:
                qualified_type = self.lookup_variable_type(qualified_name)
                if qualified_type:
                    return qualified_type
            object_type = self.expression_type(expr.object)
            struct_type = self.unwrap_resource_container_type(object_type)
            if not struct_type:
                return None
            members = self.struct_member_types.get(struct_type)
            if members is not None:
                member_type = members.get(expr.member)
                if member_type:
                    return member_type
            return self.vector_swizzle_type(object_type, expr.member)
        return None

    def vector_swizzle_type(self, type_name, swizzle):
        scalar_type = self.vector_scalar_type(type_name)
        if scalar_type is None:
            return None
        if not re.fullmatch(r"[xyzwrgba]{1,4}", str(swizzle)):
            return None
        if len(swizzle) == 1:
            return scalar_type
        return f"{scalar_type}{len(swizzle)}"

    def vector_scalar_type(self, type_name):
        if not type_name:
            return None

        generic_type = self.map_generic_vector_type(type_name)
        if generic_type:
            return self.vector_scalar_type(generic_type)

        base_type = self.resource_type_base(type_name)
        match = re.fullmatch(r"(float|half|double|int|uint|bool)([1-4])", base_type)
        if match:
            return match.group(1)

        crossgl_vectors = {
            "vec": "float",
            "dvec": "double",
            "ivec": "int",
            "uvec": "uint",
            "bvec": "bool",
        }
        for prefix, scalar_type in crossgl_vectors.items():
            if re.fullmatch(rf"{prefix}[2-4]", base_type):
                return scalar_type
        return None

    def numeric_type_family(self, type_name):
        if not type_name:
            return None
        scalar_type = self.vector_scalar_type(type_name)
        base_type = scalar_type or self.resource_type_base(type_name)
        if base_type.startswith(("uint", "uvec")):
            return "uint"
        if base_type.startswith(("int", "ivec")):
            return "int"
        if base_type.startswith(("float", "vec")):
            return "float"
        return None

    def lookup_variable_type(self, name):
        if not name:
            return None
        base_name = str(name).split("[", 1)[0]
        for scope in reversed(self.variable_type_scopes):
            if base_name in scope:
                return scope[base_name]
        return None

    def member_access_qualified_name(self, expr):
        if isinstance(expr, VariableNode) and not expr.vtype:
            return expr.name
        if isinstance(expr, MemberAccessNode):
            prefix = self.member_access_qualified_name(expr.object)
            if prefix:
                return f"{prefix}.{expr.member}"
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
        if not self.is_storage_image_resource_expression(expr.object):
            return None

        resource_type = self.storage_image_resource_expression_type(expr.object)
        resource_base = self.resource_type_base(resource_type)
        args = [self.generate_expression(arg, is_main) for arg in expr.args or []]
        if expr.method == "Load":
            return f"imageLoad({', '.join([obj] + args)})"
        if (
            expr.method == "Store"
            and len(args) == 2
            and not self.is_multisample_resource_base(resource_base)
        ):
            return f"imageStore({', '.join([obj] + args)})"
        return None

    def generate_byte_address_buffer_method_call(self, expr, obj, is_main=False):
        method = self.method_base_name(expr.method)
        helper = self.BYTE_ADDRESS_BUFFER_METHOD_MAP.get(method)
        if helper is None:
            return None

        resource_type = self.buffer_resource_expression_type(expr.object)
        if (
            self.resource_type_base(resource_type)
            not in self.BYTE_ADDRESS_BUFFER_RESOURCE_TYPES
        ):
            return None

        args = [self.generate_expression(arg, is_main) for arg in expr.args or []]
        if method.startswith("Load") and len(args) != 1:
            return None
        if method.startswith("Store") and len(args) != 2:
            return None
        return f"{helper}({', '.join([obj] + args)})"

    def generate_structured_buffer_method_call(self, expr, obj, is_main=False):
        if self.method_base_name(expr.method) != "Load":
            return None
        resource_type = self.buffer_resource_expression_type(expr.object)
        if self.resource_type_base(resource_type) not in {
            "StructuredBuffer",
            "RWStructuredBuffer",
        }:
            return None
        if len(expr.args or []) != 1:
            return None
        index = self.generate_expression(expr.args[0], is_main)
        return f"{obj}[{index}]"

    def generate_resource_get_dimensions_statement(self, stmt, indent=0, is_main=False):
        if not isinstance(stmt, MethodCallNode):
            return None
        if self.method_base_name(stmt.method) != "GetDimensions":
            return None

        resource_type = self.sampleable_resource_expression_type(stmt.object)
        storage_image = False
        if resource_type is None:
            resource_type = self.storage_image_resource_expression_type(stmt.object)
            storage_image = resource_type is not None
        if resource_type is None:
            if self.is_buffer_resource_expression(stmt.object):
                return self.generate_buffer_get_dimensions_statement(
                    stmt, indent, is_main
                )
            return None

        resource_base = self.resource_type_base(resource_type)
        if not resource_base:
            return None

        obj = self.generate_expression(stmt.object, is_main)
        rendered_args = [
            self.generate_expression(arg, is_main) for arg in stmt.args or []
        ]
        layout = self.get_dimensions_layout(
            resource_base, obj, rendered_args, storage_image
        )
        if layout is None:
            return None

        indent_text = "    " * indent
        lines = []
        temp_type = self.get_dimensions_size_temp_type(layout["dimensions"])
        temp_name = self.next_generated_temp_name("cgl_getDimensionsSize", temp_type)
        lines.append(
            f"{indent_text}{self.map_type(temp_type)} {temp_name} = "
            f"{layout['size_expr']};\n"
        )

        for target, suffix, arg_index in zip(
            layout["dimension_targets"],
            self.dimension_component_suffixes(layout["dimensions"]),
            layout["dimension_indices"],
        ):
            value = self.cast_get_dimensions_value(
                f"{temp_name}{suffix}", stmt.args[arg_index]
            )
            lines.append(f"{indent_text}{target} = {value};\n")

        if layout.get("levels_target") is not None:
            value = self.cast_get_dimensions_value(
                f"textureQueryLevels({obj})", stmt.args[layout["levels_index"]]
            )
            lines.append(f"{indent_text}{layout['levels_target']} = {value};\n")

        if layout.get("samples_target") is not None:
            sample_query = "imageSamples" if storage_image else "textureSamples"
            value = self.cast_get_dimensions_value(
                f"{sample_query}({obj})", stmt.args[layout["samples_index"]]
            )
            lines.append(f"{indent_text}{layout['samples_target']} = {value};\n")

        return "".join(lines)

    def generate_buffer_get_dimensions_statement(self, stmt, indent=0, is_main=False):
        rendered_args = [
            self.generate_expression(arg, is_main) for arg in stmt.args or []
        ]
        if not rendered_args:
            return None

        obj = self.generate_expression(stmt.object, is_main)
        indent_text = "    " * indent
        args = ", ".join([obj, *rendered_args])
        return f"{indent_text}buffer_dimensions({args});\n"

    def get_dimensions_layout(self, resource_base, obj, rendered_args, storage_image):
        args_count = len(rendered_args)
        if args_count == 0:
            return None

        dimensions = self.GET_DIMENSIONS_MULTISAMPLE_DIMENSIONS.get(resource_base)
        if dimensions is not None:
            if args_count not in {dimensions, dimensions + 1}:
                return None
            size_function = "imageSize" if storage_image else "textureSize"
            return {
                "dimensions": dimensions,
                "size_expr": f"{size_function}({obj})",
                "dimension_targets": rendered_args[:dimensions],
                "dimension_indices": list(range(dimensions)),
                "samples_target": (
                    rendered_args[dimensions] if args_count == dimensions + 1 else None
                ),
                "samples_index": dimensions if args_count == dimensions + 1 else None,
            }

        if storage_image:
            dimensions = self.GET_DIMENSIONS_IMAGE_DIMENSIONS.get(resource_base)
            if dimensions is None or args_count != dimensions:
                return None
            return {
                "dimensions": dimensions,
                "size_expr": f"imageSize({obj})",
                "dimension_targets": rendered_args,
                "dimension_indices": list(range(dimensions)),
            }

        dimensions = self.GET_DIMENSIONS_TEXTURE_DIMENSIONS.get(resource_base)
        if dimensions is None:
            return None

        lod = "0"
        out_start = 0
        levels_index = None
        if args_count == dimensions:
            pass
        elif args_count == dimensions + 1:
            levels_index = dimensions
        elif args_count == dimensions + 2:
            lod = rendered_args[0]
            out_start = 1
            levels_index = dimensions + 1
        else:
            return None

        dimension_targets = rendered_args[out_start : out_start + dimensions]
        if len(dimension_targets) != dimensions:
            return None
        return {
            "dimensions": dimensions,
            "size_expr": f"textureSize({obj}, {lod})",
            "dimension_targets": dimension_targets,
            "dimension_indices": list(range(out_start, out_start + dimensions)),
            "levels_target": (
                rendered_args[levels_index] if levels_index is not None else None
            ),
            "levels_index": levels_index,
        }

    def get_dimensions_size_temp_type(self, dimensions):
        if dimensions == 1:
            return "int"
        return f"int{dimensions}"

    def dimension_component_suffixes(self, dimensions):
        if dimensions == 1:
            return [""]
        return [f".{component}" for component in ("x", "y", "z")[:dimensions]]

    def cast_get_dimensions_value(self, value, target_arg):
        if self.get_dimensions_target_scalar_type(target_arg) == "uint":
            return f"uint({value})"
        return value

    def get_dimensions_target_scalar_type(self, target_arg):
        target_type = self.expression_type(target_arg)
        if not target_type:
            return None
        target_base = self.resource_type_base(target_type)
        if target_base in {"int", "uint"}:
            return target_base
        return None

    def next_generated_temp_name(self, base_name, vtype):
        while True:
            raw_name = f"_{base_name}{self.generated_temp_index}"
            self.generated_temp_index += 1
            safe_name = self.sanitize_crossgl_identifier(
                raw_name, self.identifier_used_name_scopes[-1]
            )
            if safe_name not in self.identifier_used_name_scopes[-1]:
                self.identifier_rename_scopes[-1][raw_name] = safe_name
                self.identifier_used_name_scopes[-1].add(safe_name)
                self.variable_type_scopes[-1][raw_name] = vtype
                return safe_name

    def generate_texture_method_call(self, expr, obj, is_main=False):
        if (
            expr.method not in self.SAMPLE_METHOD_MAP
            and expr.method not in self.GATHER_METHOD_COMPONENTS
            and expr.method not in self.GATHER_COMPARE_METHODS
            and expr.method not in self.LOD_QUERY_METHOD_COMPONENTS
        ):
            return None
        if not self.is_sampleable_resource_expression(expr.object):
            return None

        if expr.method in self.LOD_QUERY_METHOD_COMPONENTS:
            return self.generate_texture_lod_query_method_call(expr, obj, is_main)

        call_parts = self.crossgl_texture_call_parts(expr, is_main)
        if call_parts is None:
            return None
        texture_func, args = call_parts
        return f"{texture_func}({', '.join([obj] + args)})"

    def generate_texture_lod_query_method_call(self, expr, obj, is_main=False):
        component = self.LOD_QUERY_METHOD_COMPONENTS.get(expr.method)
        if component is None:
            return None

        prefix, coord, extra_args = self.split_texture_sample_args(expr, is_main)
        if coord is None or extra_args:
            return None

        args = [obj, *prefix, coord]
        return f"textureQueryLod({', '.join(args)}).{component}"

    def crossgl_texture_call_parts(self, expr, is_main=False):
        if expr.method == "Load":
            resource_type = self.sampleable_resource_expression_type(expr.object)
            args = self.format_texture_load_args(
                expr.args,
                is_main,
                resource_type,
            )
            resource_base = self.resource_type_base(resource_type)
            if self.is_multisample_resource_base(resource_base):
                texture_func = "texelFetch"
            else:
                texture_func = (
                    "texelFetchOffset" if len(expr.args or []) > 1 else "texelFetch"
                )
            return texture_func, args

        if expr.method in self.GATHER_METHOD_COMPONENTS:
            return self.crossgl_texture_gather_call_parts(expr, is_main)

        if expr.method in self.GATHER_COMPARE_METHODS:
            return self.crossgl_texture_gather_compare_call_parts(expr, is_main)

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
                texture_func = "textureGradOffset"
                if rest:
                    diagnostic = self.texture_overload_extras_diagnostic(
                        "SampleGrad",
                        self.texture_gradient_extra_parameters(rest),
                    )
                    texture_func = f"{diagnostic} {texture_func}"
                return texture_func, prefix + [coord, ddx, ddy, offset]
            return "textureGrad", prefix + [coord, *extra_args]

        if expr.method == "SampleCmp":
            if len(extra_args) > 1:
                compare, offset, *rest = extra_args
                return "textureCompareOffset", prefix + [coord, compare, offset, *rest]
            return "textureCompare", prefix + [coord, *extra_args]

        if expr.method == "SampleCmpBias":
            return self.crossgl_texture_compare_bias_call_parts(
                expr, prefix, coord, extra_args
            )

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
                texture_func = "textureCompareGradOffset"
                if rest:
                    diagnostic = self.texture_overload_extras_diagnostic(
                        "SampleCmpGrad",
                        self.texture_gradient_extra_parameters(
                            rest, includes_lod_clamp=False
                        ),
                    )
                    texture_func = f"{diagnostic} {texture_func}"
                return (
                    texture_func,
                    prefix + [coord, compare, ddx, ddy, offset],
                )
            return "textureCompareGrad", prefix + [coord, *extra_args]

        return self.SAMPLE_METHOD_MAP[expr.method], prefix + [coord, *extra_args]

    def crossgl_texture_compare_bias_call_parts(self, expr, prefix, coord, extra_args):
        if len(extra_args) < 2:
            return "textureCompare", prefix + [coord, *extra_args]

        compare = extra_args[0]
        resource_base = self.resource_type_base(
            self.sampleable_resource_expression_type(expr.object)
        )
        offset_index = (
            2
            if len(extra_args) > 2
            and resource_base
            not in {
                "TextureCube",
                "TextureCubeArray",
                "SamplerCube",
                "SamplerCubeArray",
                "SamplerCubeShadow",
                "SamplerCubeArrayShadow",
            }
            else None
        )
        dropped_parameters = ["LOD bias"]

        if offset_index is None:
            texture_func = "textureCompare"
            args = prefix + [coord, compare]
            trailing_args = extra_args[2:]
        else:
            texture_func = "textureCompareOffset"
            args = prefix + [coord, compare, extra_args[offset_index]]
            trailing_args = extra_args[offset_index + 1 :]

        if trailing_args:
            dropped_parameters.append("LOD clamp")
        if len(trailing_args) > 1:
            dropped_parameters.append("status output")
        if len(trailing_args) > 2:
            dropped_parameters.append("extra arguments")

        diagnostic = self.texture_overload_extras_diagnostic(
            "SampleCmpBias", dropped_parameters
        )
        return f"{diagnostic} {texture_func}", args

    def texture_overload_extras_diagnostic(self, member, dropped_parameters):
        parameters = ", ".join(dropped_parameters)
        return (
            f"/* unsupported Slang texture overload extras for {member}: "
            f"dropped {parameters} */"
        )

    def texture_gradient_extra_parameters(self, trailing_args, includes_lod_clamp=True):
        dropped_parameters = []
        extra_start_index = 2 if includes_lod_clamp else 1
        if includes_lod_clamp and trailing_args:
            dropped_parameters.append("LOD clamp")
        if len(trailing_args) > int(includes_lod_clamp):
            dropped_parameters.append("status output")
        if len(trailing_args) > extra_start_index:
            dropped_parameters.append("extra arguments")
        return dropped_parameters

    def crossgl_texture_gather_call_parts(self, expr, is_main=False):
        prefix, coord, extra_args = self.split_texture_sample_args(expr, is_main)
        if coord is None:
            return "textureGather", prefix + extra_args

        component = self.GATHER_METHOD_COMPONENTS[expr.method]
        args = prefix + [coord]

        if extra_args:
            offset, *rest = extra_args
            args.append(offset)
            if component is not None:
                args.append(component)
            args.extend(rest)
            return "textureGatherOffset", args

        if component is not None:
            args.append(component)
        return "textureGather", args

    def crossgl_texture_gather_compare_call_parts(self, expr, is_main=False):
        prefix, coord, extra_args = self.split_texture_sample_args(expr, is_main)
        if coord is None or len(extra_args) not in {1, 2, 5}:
            return None

        compare = extra_args[0]
        if len(extra_args) == 2:
            offset = extra_args[1]
            return "textureGatherCompareOffset", prefix + [coord, compare, offset]

        if len(extra_args) == 5:
            return "textureGatherCompareOffsets", prefix + [
                coord,
                compare,
                *extra_args[1:],
            ]

        return "textureGatherCompare", prefix + [coord, compare]

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

        base_type = str(resource_type or "").strip().split("<", 1)[0].strip()
        if self.is_multisample_resource_base(base_type):
            return None

        layout = self.texture_load_vector_layout(base_type)
        if layout is None:
            return None

        expected_vector_size, coord_rank = layout
        if vector_type is None:
            vector_type = self.expression_type(arg)

        vector_kind = self.texture_load_vector_kind(vector_type, expected_vector_size)
        if vector_kind is None:
            return None

        if vector_args is None:
            rendered = self.generate_expression(arg, is_main)
            return [
                self.texture_load_swizzle(rendered, coord_rank),
                f"{rendered}.{self.texture_load_mip_component(coord_rank)}",
            ]

        return self.split_texture_load_constructor_args(
            vector_kind, vector_args, coord_rank, is_main
        )

    def texture_load_vector_layout(self, resource_base):
        return {
            "Texture1D": (2, 1),
            "Sampler1D": (2, 1),
            "Texture1DArray": (3, 2),
            "Sampler1DArray": (3, 2),
            "Texture2D": (3, 2),
            "Sampler2D": (3, 2),
            "Texture2DArray": (4, 3),
            "Sampler2DArray": (4, 3),
            "Texture3D": (4, 3),
            "Sampler3D": (4, 3),
        }.get(resource_base)

    def texture_load_vector_kind(self, vector_type, expected_size):
        if not vector_type:
            return None
        vector_type = str(vector_type).strip()
        if vector_type == f"int{expected_size}":
            return "int"
        if vector_type == f"uint{expected_size}":
            return "uint"
        return None

    def split_texture_load_constructor_args(
        self, vector_kind, vector_args, coord_rank, is_main=False
    ):
        if len(vector_args) == coord_rank + 1:
            components = [
                self.generate_expression(component, is_main)
                for component in vector_args[:coord_rank]
            ]
            mip = self.generate_expression(vector_args[-1], is_main)
            return [
                self.texture_load_coord_from_components(
                    vector_kind, coord_rank, components
                ),
                mip,
            ]

        if len(vector_args) == 2:
            coord_arg = vector_args[0]
            if self.expression_vector_rank(coord_arg) == coord_rank:
                coord = self.generate_expression(coord_arg, is_main)
                mip = self.generate_expression(vector_args[1], is_main)
                return [coord, mip]

        if len(vector_args) == 3 and coord_rank == 3:
            prefix_arg = vector_args[0]
            if self.expression_vector_rank(prefix_arg) == 2:
                prefix = self.generate_expression(prefix_arg, is_main)
                layer_or_depth = self.generate_expression(vector_args[1], is_main)
                mip = self.generate_expression(vector_args[2], is_main)
                coord = self.texture_load_coord_from_components(
                    vector_kind, coord_rank, [prefix, layer_or_depth]
                )
                return [coord, mip]

        return None

    def texture_load_coord_from_components(self, vector_kind, coord_rank, components):
        if coord_rank == 1:
            return components[0]
        coord_type = f"{'i' if vector_kind == 'int' else 'u'}vec{coord_rank}"
        return f"{coord_type}({', '.join(components)})"

    def expression_vector_rank(self, expr):
        type_name = None
        if isinstance(expr, VectorConstructorNode):
            type_name = expr.type_name
        elif isinstance(expr, FunctionCallNode):
            type_name = expr.name
        else:
            type_name = self.expression_type(expr)
        if not type_name:
            return None
        match = re.fullmatch(r"(?:u?int|float|double|bool)([2-4])", str(type_name))
        if match:
            return int(match.group(1))
        return None

    def texture_load_swizzle(self, value, coord_rank):
        if coord_rank == 1:
            return f"{value}.x"
        return f"{value}.{self.texture_load_components(coord_rank)}"

    def texture_load_mip_component(self, coord_rank):
        return self.texture_load_components(coord_rank + 1)[-1]

    def texture_load_components(self, count):
        return "xyzw"[:count]

    def is_multisample_resource_base(self, resource_base):
        return resource_base in {
            "Texture2DMS",
            "Texture2DMSArray",
            "Sampler2DMS",
            "Sampler2DMSArray",
            "RWTexture2DMS",
            "RWTexture2DMSArray",
        }

    def is_sampleable_resource_expression(self, expr):
        return self.sampleable_resource_expression_type(expr) is not None

    def is_storage_image_resource_expression(self, expr):
        return self.storage_image_resource_expression_type(expr) is not None

    def is_buffer_resource_expression(self, expr):
        return self.buffer_resource_expression_type(expr) is not None

    def storage_image_resource_expression_type(self, expr):
        name = self.expression_base_name(expr)
        if name is not None:
            for scope in reversed(self.storage_image_resource_type_scopes):
                if name in scope:
                    return scope[name]
        type_name = self.expression_type(expr)
        if self.is_storage_image_resource_type(type_name):
            return type_name
        return None

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

    def buffer_resource_expression_type(self, expr):
        type_name = self.expression_type(expr)
        if self.is_buffer_resource_type(type_name):
            return type_name
        return None

    def resource_type_base(self, type_name):
        if not type_name:
            return None
        return str(type_name).strip().split("<", 1)[0].strip()

    def method_base_name(self, method_name):
        return str(method_name).split("<", 1)[0]

    def expression_base_name(self, expr):
        if isinstance(expr, str):
            return expr.split("[", 1)[0]
        if isinstance(expr, VariableNode):
            return expr.name.split("[", 1)[0]
        if isinstance(expr, ArrayAccessNode):
            return self.expression_base_name(expr.array)
        return None

    def is_fragment_position_input_parameter(
        self, semantic, function_qualifier=None, parameter=None
    ):
        if semantic is None or str(semantic).upper() != "SV_POSITION":
            return False
        if str(function_qualifier or "").lower() not in {"fragment", "pixel"}:
            return False
        qualifiers = {
            str(qualifier).lower()
            for qualifier in getattr(parameter, "qualifiers", []) or []
        }
        return not qualifiers.intersection({"out", "inout"})

    def is_fragment_coverage_input_parameter(
        self, semantic, function_qualifier=None, parameter=None
    ):
        if semantic is None or str(semantic).lower() != "sv_coverage":
            return False
        if str(function_qualifier or "").lower() not in {"fragment", "pixel"}:
            return False
        qualifiers = {
            str(qualifier).lower()
            for qualifier in getattr(parameter, "qualifiers", []) or []
        }
        return not qualifiers.intersection({"out", "inout"})

    def is_no_perspective_barycentric_parameter(self, parameter):
        qualifiers = {
            str(qualifier).strip().lower()
            for qualifier in getattr(parameter, "qualifiers", []) or []
        }
        return any("noperspective" in qualifier for qualifier in qualifiers)

    def map_semantic(self, semantic, function_qualifier=None, parameter=None):
        """Map a Slang semantic to CrossGL semantic annotation syntax."""
        if semantic is None:
            return ""
        ray_payload_access = self.map_ray_payload_access_semantic(semantic)
        if ray_payload_access:
            return ray_payload_access
        if str(semantic).lower() == "sv_barycentrics":
            if self.is_no_perspective_barycentric_parameter(parameter):
                return "@ gl_BaryCoordNoPerspEXT"
            return "@ gl_BaryCoordEXT"
        if self.is_fragment_position_input_parameter(
            semantic, function_qualifier, parameter
        ):
            return "@ gl_FragCoord"
        if self.is_fragment_coverage_input_parameter(
            semantic, function_qualifier, parameter
        ):
            return "@ gl_SampleMaskIn"
        mapped_semantic = self.semantic_map.get(semantic)
        if mapped_semantic is None:
            mapped_semantic = self.hlsl_system_semantic_map.get(str(semantic).lower())
        if mapped_semantic is None:
            texcoord_match = re.fullmatch(r"TEXCOORD(\d+)", str(semantic).upper())
            if texcoord_match:
                mapped_semantic = f"TexCoord{texcoord_match.group(1)}"
        if mapped_semantic is None:
            color_match = re.fullmatch(r"COLOR(\d+)", str(semantic).upper())
            if color_match:
                mapped_semantic = f"Color{color_match.group(1)}"
        if mapped_semantic is None:
            target_match = re.fullmatch(r"SV_TARGET(\d*)", str(semantic).upper())
            if target_match:
                target_index = target_match.group(1)
                mapped_semantic = f"Out_Color{target_index}"
        if mapped_semantic is None:
            depth_match = re.fullmatch(
                r"SV_DEPTH(?:GREATEREQUAL|LESSEQUAL)?", str(semantic).upper()
            )
            if depth_match:
                mapped_semantic = "Out_Depth"
        if mapped_semantic is None:
            for hlsl_prefix, crossgl_semantic in (
                ("SV_CLIPDISTANCE", "gl_ClipDistance"),
                ("SV_CULLDISTANCE", "gl_CullDistance"),
            ):
                semantic_upper = str(semantic).upper()
                if not semantic_upper.startswith(hlsl_prefix):
                    continue
                suffix = semantic_upper[len(hlsl_prefix) :]
                if not suffix or suffix.isdigit():
                    mapped_semantic = crossgl_semantic
                    break
        if mapped_semantic is None:
            for hlsl_prefix, crossgl_prefix in (
                ("NORMAL", "in_Normal"),
                ("TANGENT", "in_Tangent"),
                ("BINORMAL", "in_Binormal"),
                ("BLENDINDICES", "in_BlendIndices"),
                ("BLENDWEIGHT", "in_BlendWeight"),
            ):
                indexed_match = re.fullmatch(
                    rf"{hlsl_prefix}(\d+)", str(semantic).upper()
                )
                if indexed_match:
                    mapped_semantic = f"{crossgl_prefix}{indexed_match.group(1)}"
                    break
        return f"@ {mapped_semantic or semantic}"

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

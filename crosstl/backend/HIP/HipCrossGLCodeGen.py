"""HIP to CrossGL Code Generator"""

import re

from ...translator.diagnostics import CODE_PREFIX, TranslationDiagnostic
from ...translator.semantics.warp import ORDERED_WORKGROUP_BARRIER
from ..CUDA.buffer_layout_import import EmittedBufferLayouts
from ..CUDA.keyword_renames import source_identifier_names
from ..CUDA.reference_parameters import ReferenceParameters
from ..import_refusals import HipImportRefused, ImportRefusalMixin
from ..no_contraction import (
    PRECISE_FUNCTION_ATTRIBUTE,
    function_requires_no_contraction,
)
from ..thread_hierarchy import (
    kernel_workgroup_size_attribute,
    normalize_threadgroup_size_option,
)
from .crossgl_codegen.declarations import HipDeclarationMixin
from .crossgl_codegen.expressions import HipExpressionMixin
from .crossgl_codegen.intrinsics import HipIntrinsicMixin
from .crossgl_codegen.literals import HipLiteralMixin
from .crossgl_codegen.math_intrinsics import HipMathIntrinsicMixin
from .crossgl_codegen.name_scopes import HipNameScopeMixin
from .crossgl_codegen.record_calls import HipRecordCallMixin
from .crossgl_codegen.record_materialization import HipRecordMaterializationMixin
from .crossgl_codegen.record_semantics import HipRecordSemanticMixin
from .crossgl_codegen.records import HipRecordMixin
from .crossgl_codegen.resource_hints import HipResourceHintMixin
from .crossgl_codegen.runtime_calls import HipRuntimeCallMixin
from .crossgl_codegen.statements import HipStatementMixin
from .crossgl_codegen.surfaces import HipSurfaceMixin
from .crossgl_codegen.textures import HipTextureMixin
from .crossgl_codegen.type_conversion import HipTypeConversionMixin
from .HipAst import (
    ArrayAccessNode,
    EnumNode,
    FunctionCallNode,
    FunctionNode,
    MemberAccessNode,
    StructNode,
    TypeAliasNode,
    UnaryOpNode,
    VariableNode,
)


class HipRecordSemanticError(ValueError):
    """Raised when used C++ record semantics cannot survive CrossGL lowering."""

    project_diagnostic_code = "project.translate.hip-record-semantics-unsupported"
    missing_capabilities = ("hip.record-lifecycle-layout-lowering",)

    def __init__(self, feature, record_name, context):
        self.feature = feature
        self.record_name = record_name
        self.context = context
        super().__init__(
            f"HIP {feature} is unsupported for record '{record_name}' in {context}"
        )


class HipSurfaceByteOffsetError(TranslationDiagnostic):
    """A HIP surface call whose x byte offset cannot be shown to be a multiple of the element size.

    HIP surface functions take x as a byte offset while CrossGL ``imageLoad``/``imageStore``
    take a texel coordinate, so the frontend must divide the offset by the element size. When
    the offset cannot be proven to be a multiple of that size, passing it through unchanged
    would silently address the wrong texel, so the call is refused instead.
    """

    code = CODE_PREFIX + "hip-surface-byte-offset-unproven"


class HipToCrossGLConverter(
    HipRecordMaterializationMixin,
    HipLiteralMixin,
    HipResourceHintMixin,
    HipNameScopeMixin,
    HipRuntimeCallMixin,
    HipRecordMixin,
    HipDeclarationMixin,
    HipRecordSemanticMixin,
    HipRecordCallMixin,
    HipIntrinsicMixin,
    HipMathIntrinsicMixin,
    HipTextureMixin,
    HipSurfaceMixin,
    HipExpressionMixin,
    HipStatementMixin,
    HipTypeConversionMixin,
    ImportRefusalMixin,
):
    """Serialize HIP backend AST nodes back into CrossGL source.

    A HIP construct with no faithful CrossGL form is refused with :class:`HipImportRefused`; no CrossGL program is
    returned while a refusal is pending.
    """

    IMPORT_REFUSAL_TYPE = HipImportRefused

    SOURCE_LANGUAGE_NAME = "HIP"
    # Whether a file-scope FP_CONTRACT OFF pragma is in effect (crosstl/backend/no_contraction.py).
    fp_contract_file_scope_off = False
    ERASE_CONCRETE_TYPE_ALIASES = True
    RECORD_SEMANTIC_ERROR = HipRecordSemanticError

    CPP_NUMERIC_LITERAL_WITH_SEPARATOR = re.compile(
        r"^(?=.*')"
        r"(?:"
        r"0[xX](?:(?:[0-9a-fA-F](?:'?[0-9a-fA-F])*)?\.(?:[0-9a-fA-F](?:'?[0-9a-fA-F])*)|(?:[0-9a-fA-F](?:'?[0-9a-fA-F])*)\.?)[pP][+-]?\d(?:'?\d)*"
        r"|"
        r"0[xX][0-9a-fA-F](?:'?[0-9a-fA-F])*"
        r"|0[bB][01](?:'?[01])*"
        r"|(?:\d(?:'?\d)*)(?:\.(?:\d(?:'?\d)*)?)?"
        r"(?:[eE][+-]?\d(?:'?\d)*)?"
        r"|(?:\d(?:'?\d)*)?\.(?:\d(?:'?\d)*)(?:[eE][+-]?\d(?:'?\d)*)?"
        r")"
        r"[fFdDlLuU]*$"
    )
    CPP_INTEGER_LITERAL = re.compile(
        r"^(?P<body>"
        r"0[xX][0-9a-fA-F](?:'?[0-9a-fA-F])*"
        r"|0[bB][01](?:'?[01])*"
        r"|\d(?:'?\d)*"
        r")(?P<suffix>[uUlL]*)$"
    )

    HIP_RUNTIME_ERROR_WRAPPER_NAMES = {
        "CHECK_HIP",
        "HIP_CHECK",
        "HIPCHECK",
        "checkHip",
        "checkHipErrors",
        "hipCheck",
    }

    VECTOR_TYPE_MAPPING = {
        "half2": "vec2<f16>",
        "float2": "vec2<f32>",
        "float3": "vec3<f32>",
        "float4": "vec4<f32>",
        "double2": "vec2<f64>",
        "double3": "vec3<f64>",
        "double4": "vec4<f64>",
        "int2": "vec2<i32>",
        "int3": "vec3<i32>",
        "int4": "vec4<i32>",
        "uint2": "vec2<u32>",
        "uint3": "vec3<u32>",
        "uint4": "vec4<u32>",
        "char2": "vec2<i8>",
        "char3": "vec3<i8>",
        "char4": "vec4<i8>",
        "uchar2": "vec2<u8>",
        "uchar3": "vec3<u8>",
        "uchar4": "vec4<u8>",
        "short2": "vec2<i16>",
        "short3": "vec3<i16>",
        "short4": "vec4<i16>",
        "ushort2": "vec2<u16>",
        "ushort3": "vec3<u16>",
        "ushort4": "vec4<u16>",
        "long2": "vec2<i64>",
        "long3": "vec3<i64>",
        "long4": "vec4<i64>",
        "ulong2": "vec2<u64>",
        "ulong3": "vec3<u64>",
        "ulong4": "vec4<u64>",
        "longlong2": "vec2<i64>",
        "longlong3": "vec3<i64>",
        "longlong4": "vec4<i64>",
        "ulonglong2": "vec2<u64>",
        "ulonglong3": "vec3<u64>",
        "ulonglong4": "vec4<u64>",
    }
    VECTOR1_TYPE_MAPPING = {
        "char1": "i8",
        "uchar1": "u8",
        "short1": "i16",
        "ushort1": "u16",
        "int1": "i32",
        "uint1": "u32",
        "long1": "i64",
        "ulong1": "u64",
        "longlong1": "i64",
        "ulonglong1": "u64",
        "float1": "f32",
        "double1": "f64",
    }
    HIP_SURFACE_VECTOR_COMPONENTS = ("x", "y", "z", "w")
    HIP_SURFACE_SCALAR_TYPES = frozenset(
        {
            "char",
            "uchar",
            "short",
            "ushort",
            "int",
            "uint",
            "long",
            "ulong",
            "longlong",
            "ulonglong",
            "half",
            "__half",
            "float",
            "double",
            "bool",
        }
    )
    CROSSL_SCALAR_TYPES = {
        "bool",
        "f16",
        "f32",
        "f64",
        "i8",
        "i16",
        "i32",
        "i64",
        "u8",
        "u16",
        "u32",
        "u64",
    }
    VECTOR_CONSTRUCTOR_MAPPING = {
        **VECTOR1_TYPE_MAPPING,
        **VECTOR_TYPE_MAPPING,
        **{
            f"make_{name}": mapped
            for name, mapped in {
                **VECTOR1_TYPE_MAPPING,
                **VECTOR_TYPE_MAPPING,
            }.items()
        },
    }
    HIP_TEXTURE_TYPE_MAPPING = {
        "1": "sampler1D",
        "2": "sampler2D",
        "3": "sampler3D",
        "hipTextureType1D": "sampler1D",
        "hipTextureType1DLayered": "sampler1DArray",
        "hipTextureType2D": "sampler2D",
        "hipTextureType2DLayered": "sampler2DArray",
        "hipTextureType3D": "sampler3D",
        "hipTextureTypeCubemap": "samplerCube",
        "hipTextureTypeCubemapLayered": "samplerCubeArray",
        "cudaTextureType1D": "sampler1D",
        "cudaTextureType1DLayered": "sampler1DArray",
        "cudaTextureType2D": "sampler2D",
        "cudaTextureType2DLayered": "sampler2DArray",
        "cudaTextureType3D": "sampler3D",
        "cudaTextureTypeCubemap": "samplerCube",
        "cudaTextureTypeCubemapLayered": "samplerCubeArray",
    }
    HIP_SURFACE_TYPE_MAPPING = {
        "1": "image1D",
        "2": "image2D",
        "3": "image3D",
        "hipSurfaceType1D": "image1D",
        "hipSurfaceType1DLayered": "image1DArray",
        "hipSurfaceType2D": "image2D",
        "hipSurfaceType2DLayered": "image2DArray",
        "hipSurfaceType3D": "image3D",
        "hipSurfaceTypeCubemap": "imageCube",
        "hipSurfaceTypeCubemapLayered": "imageCubeArray",
        "cudaSurfaceType1D": "image1D",
        "cudaSurfaceType1DLayered": "image1DArray",
        "cudaSurfaceType2D": "image2D",
        "cudaSurfaceType2DLayered": "image2DArray",
        "cudaSurfaceType3D": "image3D",
        "cudaSurfaceTypeCubemap": "imageCube",
        "cudaSurfaceTypeCubemapLayered": "imageCubeArray",
    }
    HIP_TEXTURE_CALL_TYPE_HINTS = {
        "tex1D": "sampler1D",
        "tex1Dfetch": "sampler1D",
        "tex1DLod": "sampler1D",
        "tex1DGrad": "sampler1D",
        "tex2D": "sampler2D",
        "tex2DLod": "sampler2D",
        "tex2DGrad": "sampler2D",
        "tex2Dgather": "sampler2D",
        "tex3D": "sampler3D",
        "tex3DLod": "sampler3D",
        "tex3DGrad": "sampler3D",
        "texCubemap": "samplerCube",
        "texCubemapLod": "samplerCube",
        "texCubemapGrad": "samplerCube",
        "tex1DLayered": "sampler1DArray",
        "tex1DLayeredLod": "sampler1DArray",
        "tex1DLayeredGrad": "sampler1DArray",
        "tex2DLayered": "sampler2DArray",
        "tex2DLayeredLod": "sampler2DArray",
        "tex2DLayeredGrad": "sampler2DArray",
        "texCubemapLayered": "samplerCubeArray",
        "texCubemapLayeredLod": "samplerCubeArray",
        "texCubemapLayeredGrad": "samplerCubeArray",
    }
    HIP_SURFACE_CALL_TYPE_HINTS = {
        "surf1Dread": "image1D",
        "surf1Dwrite": "image1D",
        "surf1DLayeredread": "image1DArray",
        "surf1DLayeredwrite": "image1DArray",
        "surf2Dread": "image2D",
        "surf2Dwrite": "image2D",
        "surf3Dread": "image3D",
        "surf3Dwrite": "image3D",
        "surf2DLayeredread": "image2DArray",
        "surf2DLayeredwrite": "image2DArray",
        "surfCubemapread": "imageCube",
        "surfCubemapwrite": "imageCube",
        "surfCubemapLayeredread": "imageCubeArray",
        "surfCubemapLayeredwrite": "imageCubeArray",
    }
    HIP_FUNCTION_ATTRIBUTE_MEMBERS = {
        "binaryVersion",
        "cacheModeCA",
        "constSizeBytes",
        "localSizeBytes",
        "maxDynamicSharedSizeBytes",
        "maxThreadsPerBlock",
        "numRegs",
        "preferredShmemCarveout",
        "ptxVersion",
        "sharedSizeBytes",
    }
    HIP_CHANNEL_DESCRIPTOR_MEMBERS = {"f", "w", "x", "y", "z"}
    HIP_ARRAY_DESCRIPTOR_MEMBERS = {"Format", "Height", "NumChannels", "Width"}
    HIP_ARRAY3D_DESCRIPTOR_MEMBERS = {
        "Depth",
        "Flags",
        "Format",
        "Height",
        "NumChannels",
        "Width",
    }
    HIP_EXTENT_MEMBERS = {"depth", "height", "width"}
    HIP_RESOURCE_DESCRIPTOR_MEMBERS = {"resType"}
    HIP_RESOURCE_DESCRIPTOR_NESTED_MEMBERS = {
        "res.linear.desc.f",
        "res.linear.desc.w",
        "res.linear.desc.x",
        "res.linear.desc.y",
        "res.linear.desc.z",
        "res.linear.sizeInBytes",
        "res.pitch2D.desc.f",
        "res.pitch2D.desc.w",
        "res.pitch2D.desc.x",
        "res.pitch2D.desc.y",
        "res.pitch2D.desc.z",
        "res.pitch2D.height",
        "res.pitch2D.pitchInBytes",
        "res.pitch2D.width",
    }
    HIP_TEXTURE_DESCRIPTOR_MEMBERS = {
        "disableTrilinearOptimization",
        "filterMode",
        "flags",
        "maxAnisotropy",
        "maxMipmapLevelClamp",
        "minMipmapLevelClamp",
        "mipmapFilterMode",
        "mipmapLevelBias",
        "normalizedCoords",
        "readMode",
        "sRGB",
    }
    HIP_TEXTURE_DESCRIPTOR_INDEXED_MEMBERS = {
        "addressMode[0]",
        "addressMode[1]",
        "addressMode[2]",
        "borderColor[0]",
        "borderColor[1]",
        "borderColor[2]",
        "borderColor[3]",
    }
    HIP_RESOURCE_VIEW_DESCRIPTOR_MEMBERS = {
        "depth",
        "firstLayer",
        "firstMipmapLevel",
        "format",
        "height",
        "lastLayer",
        "lastMipmapLevel",
        "width",
    }
    # ``let``, ``match`` and ``var`` are CrossGL lexer keywords.
    CROSSGL_RESERVED_IDENTIFIERS = {"buffer", "in", "let", "match", "precision", "var"}
    CPP_OPERATOR_FUNCTION_NAME_PARTS = {
        "+": "plus",
        "-": "minus",
        "*": "mul",
        "/": "div",
        "%": "mod",
        "&": "bit_and",
        "|": "bit_or",
        "^": "bit_xor",
        "~": "bit_not",
        "!": "not",
        "=": "assign",
        "<": "lt",
        ">": "gt",
        "+=": "plus_assign",
        "-=": "minus_assign",
        "*=": "mul_assign",
        "/=": "div_assign",
        "%=": "mod_assign",
        "&=": "bit_and_assign",
        "|=": "bit_or_assign",
        "^=": "bit_xor_assign",
        "==": "eq",
        "!=": "ne",
        "<=>": "three_way",
        "<=": "le",
        ">=": "ge",
        "&&": "logical_and",
        "||": "logical_or",
        "<<": "shift_left",
        ">>": "shift_right",
        "<<=": "shift_left_assign",
        ">>=": "shift_right_assign",
        "++": "increment",
        "--": "decrement",
        "[]": "index",
        "()": "call",
    }
    CPP_SCALAR_TYPE_ALIASES = {
        **{
            f"std::{name}": name
            for name in (
                "int8_t",
                "uint8_t",
                "int16_t",
                "uint16_t",
                "int32_t",
                "uint32_t",
                "int64_t",
                "uint64_t",
                "size_t",
            )
        },
        **{
            f"cuda::std::{name}": name
            for name in (
                "int8_t",
                "uint8_t",
                "int16_t",
                "uint16_t",
                "int32_t",
                "uint32_t",
                "int64_t",
                "uint64_t",
                "size_t",
            )
        },
        **{
            f"hip::std::{name}": name
            for name in (
                "int8_t",
                "uint8_t",
                "int16_t",
                "uint16_t",
                "int32_t",
                "uint32_t",
                "int64_t",
                "uint64_t",
                "size_t",
            )
        },
    }
    HIP_FIXED_SCALAR_REPRESENTATION_BYTES = {
        "bool": 1,
        "char": 1,
        "signed char": 1,
        "unsigned char": 1,
        "__int8": 1,
        "signed __int8": 1,
        "unsigned __int8": 1,
        "int8": 1,
        "uint8": 1,
        "int8_t": 1,
        "uint8_t": 1,
        "i8": 1,
        "u8": 1,
        "short": 2,
        "short int": 2,
        "signed short": 2,
        "signed short int": 2,
        "unsigned short": 2,
        "unsigned short int": 2,
        "__int16": 2,
        "signed __int16": 2,
        "unsigned __int16": 2,
        "int16": 2,
        "uint16": 2,
        "int16_t": 2,
        "uint16_t": 2,
        "i16": 2,
        "u16": 2,
        "half": 2,
        "__half": 2,
        "float16_t": 2,
        "f16": 2,
        "int": 4,
        "signed": 4,
        "signed int": 4,
        "unsigned": 4,
        "unsigned int": 4,
        "int32": 4,
        "uint32": 4,
        "int32_t": 4,
        "uint32_t": 4,
        "i32": 4,
        "u32": 4,
        "float": 4,
        "float32_t": 4,
        "f32": 4,
        "long long": 8,
        "long long int": 8,
        "signed long long": 8,
        "signed long long int": 8,
        "unsigned long long": 8,
        "unsigned long long int": 8,
        "__int64": 8,
        "signed __int64": 8,
        "unsigned __int64": 8,
        "int64": 8,
        "uint64": 8,
        "int64_t": 8,
        "uint64_t": 8,
        # HIP runs only on 64-bit hosts (LP64 and LLP64 alike), where both are 64-bit.
        "size_t": 8,
        "ptrdiff_t": 8,
        "double": 8,
        "float64_t": 8,
        "f64": 8,
    }
    HIP_FIXED_SCALAR_EXPRESSION_TYPES = frozenset(
        {
            "bool",
            "i8",
            "u8",
            "i16",
            "u16",
            "i32",
            "f16",
            "f32",
            "f64",
        }
    )
    HIP_EXTERNAL_RECORD_NARROW_INTEGER_TYPES = frozenset(
        {
            "char",
            "char8_t",
            "char16_t",
            "signed char",
            "unsigned char",
            "short",
            "short int",
            "signed short",
            "signed short int",
            "unsigned short",
            "unsigned short int",
            "__int8",
            "unsigned __int8",
            "__int16",
            "unsigned __int16",
            "i8",
            "u8",
            "i16",
            "u16",
            "int8",
            "uint8",
            "int16",
            "uint16",
            "int8_t",
            "uint8_t",
            "int16_t",
            "uint16_t",
        }
    )

    def __init__(self, threadgroup_size=None):
        # The host picks a HIP kernel's block size at launch time; this option names it.
        self.threadgroup_size = normalize_threadgroup_size_option(threadgroup_size)
        self.indent_level = 0
        self.output = []
        # Inside a __global__ kernel or a __device__ function, where host runtime calls are refused.
        self.in_device_code = False
        self.identifier_name_scopes = [{}]
        self.packed_argument_scopes = []
        self.unique_ptr_scopes = [set()]
        self.type_alias_scopes = [{}]
        self.concrete_type_alias_scopes = [{}]
        self.vector1_name_scopes = [{}]
        self.user_function_names = set()
        self.global_resource_object_type_hints = {}
        self.resource_object_hint_scopes = []
        self.cooperative_group_scopes = [{}]
        self.variable_type_scopes = [{}]
        self.device_property_source_scopes = [{}]
        self.device_attribute_source_scopes = [{}]
        self.device_query_source_scopes = [{}]
        self.member_query_source_scopes = [{}]
        self.suppress_device_property_member_access = 0
        self.suppress_device_attribute_value_access = 0
        self.suppress_device_query_value_access = 0
        self.suppress_identifier_name_rewrite = 0
        self.generated_matrix_helper_types = set()
        self.namespace_aliases = {}
        self.source_identifier_names = frozenset()
        self.anonymous_enum_count = 0
        self.global_resource_bindings = {}
        self.global_resource_binding_count = 0
        self.function_local_record_definitions = {}
        self.function_local_record_node_ids = set()
        self.function_local_record_declaration_types = {}
        self.record_name_stack = []
        self.record_definitions = {}
        self.record_qualified_names = {}
        self.enum_value_aliases = {}
        self.enum_underlying_types = {}
        self.unsupported_record_static_constants = set()
        self.function_definitions = {}
        self.return_type_by_node_id = {}
        self.union_active_member_scopes = [{}]
        self.union_object_identity_scopes = [{}]
        self.union_indirect_object_scopes = [set()]
        self.namespace_scope_variable = False
        self.namespace_constant_names = set()
        self.function_body_depth = 0

    def generate(self, ast_node):
        """Render ``ast_node`` as CrossGL, or raise :class:`HipImportRefused` listing every construct it cannot carry."""
        return self.refusals.run(self.render_program, ast_node)

    def render_program(self, ast_node):
        self.output = []
        self.indent_level = 0
        self.source_identifier_names = source_identifier_names(ast_node)
        self.identifier_name_scopes = [{}]
        self.packed_argument_scopes = []
        self.unique_ptr_scopes = [set()]
        self.type_alias_scopes = [{}]
        self.concrete_type_alias_scopes = [{}]
        self.vector1_name_scopes = [{}]
        self.generated_matrix_helper_types = self.collect_generated_matrix_helper_types(
            ast_node
        )
        self.user_function_names = self.collect_user_function_names(ast_node)
        self.global_resource_object_type_hints = (
            self.collect_global_resource_object_type_hints(ast_node)
        )
        self.resource_object_hint_scopes = []
        self.cooperative_group_scopes = [{}]
        self.variable_type_scopes = [{}]
        self.device_property_source_scopes = [{}]
        self.device_attribute_source_scopes = [{}]
        self.device_query_source_scopes = [{}]
        self.member_query_source_scopes = [{}]
        self.union_active_member_scopes = [{}]
        self.union_object_identity_scopes = [{}]
        self.union_indirect_object_scopes = [set()]
        self.suppress_device_property_member_access = 0
        self.suppress_device_attribute_value_access = 0
        self.suppress_device_query_value_access = 0
        self.suppress_identifier_name_rewrite = 0
        self.namespace_aliases = getattr(ast_node, "namespace_aliases", {}) or {}
        self.anonymous_enum_count = 0
        self.global_resource_binding_count = 0
        self.namespace_constant_names = set()
        self.function_body_depth = 0
        self.reset_function_local_record_materializations()
        self.prepare_function_local_record_materializations(ast_node)
        self.prepare_record_definitions(ast_node)
        self.refuse_incomplete_record_values(ast_node)
        self.prepare_global_resource_bindings(ast_node)
        self.reference_parameters = ReferenceParameters(ast_node)
        self.emitted_buffer_layouts = EmittedBufferLayouts("HIP")
        self.visit(ast_node)
        self.emitted_buffer_layouts.apply(self.output)
        return "\n".join(self.output)

    def refuse_incomplete_record_values(self, ast_node):
        """Refuse a by-value use of a record that is declared without a body but not yet completed.

        A record is incomplete from ``struct Never;`` until its closing brace (C11 6.7.2.3), so a member,
        local, global or function parameter of that type that comes before the body, or in a program that
        never gives one, has no size and the C family compilers reject it. A pointer or reference to it, and
        the parameter of a prototype, are fine.
        """
        forward_declared = set()
        completed = set()
        # Records whose body is being walked: a record is incomplete until its own closing brace too.
        open_bodies = set()
        words_to_skip = {
            "const",
            "volatile",
            "struct",
            "union",
            "class",
            "private",
            "__private",
        }

        def incomplete_record(vtype):
            if not isinstance(vtype, str):
                return None
            compact = vtype.replace(" ", "")
            if any(mark in compact for mark in ("*", "&", "ptr<")):
                return None
            words = [
                word
                for word in re.split(r"[\s\[]", vtype.split("[", 1)[0])
                if word and word not in words_to_skip
            ]
            if len(words) != 1:
                return None
            if words[0] in open_bodies or words[0] in forward_declared - completed:
                return words[0]
            return None

        def refuse(record, name, owner):
            context = f"the declaration of '{name}'"
            if owner:
                context += f" in '{owner}'"
            raise self.RECORD_SEMANTIC_ERROR("incomplete record type", record, context)

        def visit(current, owner):
            if isinstance(current, (list, tuple)):
                for item in current:
                    visit(item, owner)
                return
            if not hasattr(current, "__dict__") or isinstance(
                current, (str, int, float, bool)
            ):
                return
            if isinstance(current, StructNode):
                name = current.name
                if not getattr(current, "has_body", True):
                    if name and name not in completed:
                        forward_declared.add(name)
                    return
                opened = bool(name) and name not in open_bodies
                if opened:
                    open_bodies.add(name)
                for member in current.members or []:
                    visit(member, name or owner)
                if opened:
                    open_bodies.discard(name)
                if name:
                    completed.add(name)
                return
            if isinstance(current, FunctionNode):
                owner = current.name or owner
                # A method body sees its record complete.
                enclosing = set(open_bodies)
                open_bodies.clear()
                if current.body is not None:
                    for param in current.params or []:
                        if isinstance(param, dict):
                            record = incomplete_record(param.get("type"))
                            if record is not None:
                                refuse(record, param.get("name"), owner)
                visit(current.body, owner)
                open_bodies.update(enclosing)
                return
            if isinstance(current, VariableNode):
                record = incomplete_record(current.vtype)
                if record is not None:
                    refuse(record, current.name, owner)
            for value in vars(current).values():
                if isinstance(value, (list, tuple)) or hasattr(value, "__dict__"):
                    visit(value, owner)

        visit(getattr(ast_node, "statements", ast_node), None)

    def collect_resource_object_type_hints(self, node, declared_names=None):
        hints = {}

        def collect(current):
            if current is None:
                return
            if isinstance(current, (list, tuple)):
                for item in current:
                    collect(item)
                return
            if isinstance(current, dict):
                for item in current.values():
                    collect(item)
                return
            if isinstance(current, (str, int, float, bool)):
                return
            if not hasattr(current, "__dict__"):
                return

            if isinstance(current, FunctionCallNode):
                hint = self.get_resource_object_call_hint(current.name, current.args)
                if hint is not None:
                    arg_index, resource_type = hint
                    if len(current.args) > arg_index:
                        self.add_resource_object_type_hint(
                            hints,
                            self.get_resource_object_expression_name(
                                current.args[arg_index]
                            ),
                            resource_type,
                        )

            for value in vars(current).values():
                collect(value)

        collect(node)
        for name in declared_names or []:
            hints.setdefault(name, None)
        return hints

    def collect_global_resource_object_type_hints_from_function(
        self, node, global_names, hints
    ):
        local_names = self.collect_declared_variable_names(node)

        def collect(current):
            if current is None:
                return
            if isinstance(current, (list, tuple)):
                for item in current:
                    collect(item)
                return
            if isinstance(current, dict):
                for item in current.values():
                    collect(item)
                return
            if isinstance(current, (str, int, float, bool)):
                return
            if not hasattr(current, "__dict__"):
                return

            if isinstance(current, FunctionCallNode):
                hint = self.get_resource_object_call_hint(current.name, current.args)
                if hint is not None:
                    arg_index, resource_type = hint
                    if len(current.args) > arg_index:
                        name = self.get_resource_object_expression_name(
                            current.args[arg_index]
                        )
                        if name in global_names and name not in local_names:
                            self.add_resource_object_type_hint(
                                hints, name, resource_type
                            )

            for value in vars(current).values():
                collect(value)

        collect(getattr(node, "body", []))

    def collect_declared_variable_names(self, node):
        names = set()
        for param in getattr(node, "params", []) or []:
            if isinstance(param, dict):
                name = param.get("name")
            else:
                name = getattr(param, "name", None)
            if name:
                names.add(name)

        def collect(current):
            if current is None:
                return
            if isinstance(current, (list, tuple)):
                for item in current:
                    collect(item)
                return
            if isinstance(current, dict):
                for item in current.values():
                    collect(item)
                return
            if isinstance(current, (str, int, float, bool)):
                return
            if not hasattr(current, "__dict__"):
                return

            if isinstance(current, VariableNode):
                names.add(current.name)

            for value in vars(current).values():
                collect(value)

        collect(getattr(node, "body", []))
        return names

    def emit_statement(self, stmt):
        """Render and append one converted statement."""
        if isinstance(stmt, list):
            for item in stmt:
                self.emit_statement(item)
            return
        if (
            isinstance(stmt, StructNode)
            and id(stmt) in self.function_local_record_node_ids
        ):
            return

        if self.emit_hip_runtime_call_statement(stmt):
            return
        if self.emit_thread_block_sync_statement(stmt):
            return

        result = self.visit(stmt)
        if isinstance(result, str) and result.strip():
            self.emit(f"{result};")

    def emit_thread_block_sync_statement(self, stmt):
        """Emit ``g.sync()`` or ``cg::sync(g)`` of a thread block as ``__syncthreads()``; as a value it is refused."""
        if not isinstance(stmt, FunctionCallNode):
            return False
        if isinstance(stmt.name, MemberAccessNode):
            member_base_name, _ = self.parse_cpp_template(stmt.name.member)
            member_name = (
                self.cooperative_group_base_name(member_base_name) or stmt.name.member
            )
            if member_name != "sync" or stmt.args:
                return False
            group = stmt.name.object
        elif isinstance(stmt.name, str) and len(stmt.args) == 1:
            base_call_name, _ = self.parse_cpp_template(stmt.name)
            if self.cooperative_group_base_name(base_call_name) != "sync":
                return False
            group = stmt.args[0]
        else:
            return False
        group_metadata = self.resolve_cooperative_group_metadata(group)
        if group_metadata is None or group_metadata["kind"] != "thread_block":
            return False
        self.emit_block_barrier()
        return True

    def emit_hip_runtime_call_statement(self, stmt):
        if not isinstance(stmt, FunctionCallNode):
            return False
        if self.is_user_defined_function(stmt.name):
            return False

        runtime_wrapper = self.format_hip_runtime_wrapper_expression(stmt)
        if runtime_wrapper is not None:
            comments, _ = runtime_wrapper
            for comment in comments:
                self.emit(comment)
            return True

        comments = self.format_hip_runtime_call(stmt)
        if comments is None:
            return False

        for comment in comments:
            self.emit(comment)
        return True

    def format_hip_runtime_call(self, node):
        """Describe a host runtime call as comments; in device code the call is refused and described by none."""
        comments = self.describe_hip_runtime_call(node)
        if comments is not None and self.in_device_code:
            self.refuse_device_runtime_call(f"runtime call {node.name}()")
            return []
        return comments

    def refuse_device_runtime_call(self, what):
        """Refuse a HIP runtime call or kernel launch made from device code."""
        return self.refuse_import(
            f"HIP {what} in device code: CrossGL kernels have no device-side runtime (allocation, memory "
            "operations, streams or launches); only the host can perform it",
            "0",
        )

    def describe_hip_runtime_call(self, node):
        args = [self.visit_runtime_argument_expression(arg) for arg in node.args]
        name = node.name

        if name in {"hipMalloc", "hipMallocManaged"}:
            if len(node.args) >= 2:
                target = self.format_runtime_raw_output_target(node.args[0])
                size = self.visit(node.args[1])
                return [f"// HIP memory allocate: {target}, bytes: {size}"]
        elif name == "hipExtMallocWithFlags":
            if len(args) >= 3:
                target = self.format_runtime_raw_output_target(node.args[0])
                return [
                    f"// HIP extended memory allocate: {target}, "
                    f"bytes: {args[1]}, flags: {args[2]}"
                ]
        elif name == "hipMallocAsync":
            if len(args) >= 3:
                target = self.format_runtime_raw_output_target(node.args[0])
                return [
                    f"// HIP async memory allocate: {target}, bytes: {args[1]}, "
                    f"stream: {args[2]}"
                ]
        elif name == "hipMallocFromPoolAsync":
            if len(args) >= 4:
                target = self.format_runtime_raw_output_target(node.args[0])
                return [
                    f"// HIP async memory allocate from pool: {target}, "
                    f"bytes: {args[1]}, pool: {args[2]}, stream: {args[3]}"
                ]
        elif name in {"hipHostMalloc", "hipHostAlloc"}:
            if len(args) >= 2:
                target = self.format_runtime_raw_output_target(node.args[0])
                comment = f"// HIP host memory allocate: {target}, bytes: {args[1]}"
                if len(args) >= 3:
                    comment += f", flags: {args[2]}"
                return [comment]
        elif name == "hipMallocPitch":
            if len(args) >= 4:
                target = self.format_runtime_raw_output_target(node.args[0])
                pitch = self.format_runtime_raw_output_target(node.args[1])
                return [
                    f"// HIP pitched memory allocate: {target}, pitch: {pitch}, "
                    f"width: {args[2]}, height: {args[3]}"
                ]
        elif name == "hipMalloc3D":
            if len(args) >= 2:
                target = self.format_runtime_raw_output_target(node.args[0])
                return [f"// HIP 3D memory allocate: {target}, extent: {args[1]}"]
        elif name in {"hipMallocArray", "hipMalloc3DArray"}:
            if len(args) >= 4:
                target = self.format_runtime_raw_output_target(node.args[0])
                descriptor = self.format_runtime_pointer_target(node.args[1])
                if name == "hipMallocArray":
                    comment = (
                        f"// HIP array allocate: {target}, desc: {descriptor}, "
                        f"width: {args[2]}, height: {args[3]}"
                    )
                    if len(args) >= 5:
                        comment += f", flags: {args[4]}"
                else:
                    comment = (
                        f"// HIP 3D array allocate: {target}, desc: {descriptor}, "
                        f"extent: {args[2]}, flags: {args[3]}"
                    )
                return [comment]
        elif name == "hipMallocMipmappedArray":
            if len(args) >= 5:
                target = self.format_runtime_raw_output_target(node.args[0])
                descriptor = self.format_runtime_pointer_target(node.args[1])
                return [
                    f"// HIP mipmapped array allocate: output: {target}, "
                    f"desc: {descriptor}, extent: {args[2]}, levels: {args[3]}, "
                    f"flags: {args[4]}"
                ]
        elif name == "hipMipmappedArrayCreate":
            if len(args) >= 3:
                target = self.format_runtime_raw_output_target(node.args[0])
                descriptor = self.format_runtime_pointer_target(node.args[1])
                return [
                    f"// HIP mipmapped array create: output: {target}, "
                    f"descriptor: {descriptor}, levels: {args[2]}"
                ]
        elif name in {"hipGetMipmappedArrayLevel", "hipMipmappedArrayGetLevel"}:
            if len(args) >= 3:
                output = self.format_runtime_raw_output_target(node.args[0])
                return [
                    f"// HIP mipmapped array get level: output: {output}, "
                    f"mipmapped array: {args[1]}, level: {args[2]}"
                ]
        elif name == "hipMipmappedArrayDestroy":
            if args:
                return [f"// HIP free mipmapped array: {args[0]}"]
        elif name in {"hipFree", "hipHostFree", "hipFreeHost"}:
            if args:
                return [f"// HIP memory free: {args[0]}"]
        elif name == "hipFreeAsync":
            if len(args) >= 2:
                return [f"// HIP async memory free: {args[0]}, stream: {args[1]}"]
        elif name in {"hipFreeArray", "hipArrayDestroy"}:
            if args:
                return [f"// HIP array free: {args[0]}"]
        elif name in {"hipArrayCreate", "hipArray3DCreate"}:
            if len(args) >= 2:
                output = self.format_runtime_raw_output_target(node.args[0])
                descriptor = self.format_runtime_pointer_target(node.args[1])
                dimension = "3D " if name == "hipArray3DCreate" else ""
                return [
                    f"// HIP {dimension}array create: output: {output}, "
                    f"descriptor: {descriptor}"
                ]
        elif name in {"hipArrayGetDescriptor", "hipArray3DGetDescriptor"}:
            if len(args) >= 2:
                output = self.format_runtime_raw_output_target(node.args[0])
                output_name = self.get_runtime_pointer_target_name(node.args[0])
                dimension = "3D " if name == "hipArray3DGetDescriptor" else ""
                if output_name is not None:
                    if name == "hipArray3DGetDescriptor":
                        members = self.HIP_ARRAY3D_DESCRIPTOR_MEMBERS
                        query_prefix = "array.descriptor3D"
                    else:
                        members = self.HIP_ARRAY_DESCRIPTOR_MEMBERS
                        query_prefix = "array.descriptor"
                    self.register_member_query_sources(
                        output_name,
                        {
                            member: f"{query_prefix}.{member}({args[1]})"
                            for member in members
                        },
                    )
                return [
                    f"// HIP array get {dimension}descriptor: output: {output}, "
                    f"array: {args[1]}"
                ]
        elif name == "hipArrayGetInfo":
            if len(args) >= 4:
                descriptor_output = self.format_runtime_raw_output_target(node.args[0])
                extent_output = self.format_runtime_raw_output_target(node.args[1])
                flags_output = self.format_runtime_raw_output_target(node.args[2])
                descriptor_output_name = self.get_runtime_pointer_target_name(
                    node.args[0]
                )
                extent_output_name = self.get_runtime_pointer_target_name(node.args[1])
                flags_output_name = self.get_runtime_pointer_target_name(node.args[2])
                if descriptor_output_name is not None:
                    self.register_member_query_sources(
                        descriptor_output_name,
                        {
                            member: f"array.info.channelDesc.{member}({args[3]})"
                            for member in self.HIP_CHANNEL_DESCRIPTOR_MEMBERS
                        },
                    )
                if extent_output_name is not None:
                    self.register_member_query_sources(
                        extent_output_name,
                        {
                            member: f"array.info.extent.{member}({args[3]})"
                            for member in self.HIP_EXTENT_MEMBERS
                        },
                    )
                if flags_output_name is not None:
                    self.register_device_query_source(
                        flags_output_name,
                        f"array.info.flags({args[3]})",
                    )
                return [
                    f"// HIP array get info: desc output: {descriptor_output}, "
                    f"extent output: {extent_output}, flags output: {flags_output}, "
                    f"array: {args[3]}"
                ]
        elif name == "hipMemPoolCreate":
            if len(args) >= 2:
                output = self.format_runtime_raw_output_target(node.args[0])
                return [
                    f"// HIP memory pool create: output: {output}, properties: {args[1]}"
                ]
        elif name == "hipMemPoolDestroy":
            if args:
                return [f"// HIP memory pool destroy: {args[0]}"]
        elif name == "hipMemPoolTrimTo":
            if len(args) >= 2:
                return [
                    f"// HIP memory pool trim: pool: {args[0]}, "
                    f"minimum bytes: {args[1]}"
                ]
        elif name == "hipMemPoolSetAttribute":
            if len(args) >= 3:
                value = self.format_runtime_pointer_target(node.args[2])
                return [
                    f"// HIP memory pool set attribute: pool: {args[0]}, "
                    f"attribute: {args[1]}, value: {value}"
                ]
        elif name == "hipMemPoolGetAttribute":
            if len(args) >= 3:
                output = self.format_runtime_raw_output_target(node.args[2])
                output_name = self.get_runtime_pointer_target_name(node.args[2])
                if output_name is not None:
                    self.register_device_query_source(
                        output_name,
                        f"memoryPool.attribute({args[0]}, {args[1]})",
                    )
                return [
                    f"// HIP memory pool get attribute: pool: {args[0]}, "
                    f"attribute: {args[1]}, output: {output}"
                ]
        elif name == "hipMemPoolSetAccess":
            if len(args) >= 3:
                return [
                    f"// HIP memory pool set access: pool: {args[0]}, "
                    f"descriptors: {args[1]}, count: {args[2]}"
                ]
        elif name == "hipMemPoolGetAccess":
            if len(args) >= 3:
                output = self.format_runtime_raw_output_target(node.args[0])
                output_name = self.get_runtime_pointer_target_name(node.args[0])
                location = self.format_runtime_pointer_target(node.args[2])
                if output_name is not None:
                    self.register_device_query_source(
                        output_name,
                        f"memoryPool.accessFlags({args[1]}, {location})",
                    )
                return [
                    f"// HIP memory pool get access: output: {output}, "
                    f"pool: {args[1]}, location: {args[2]}"
                ]
        elif name == "hipMemPoolExportToShareableHandle":
            if len(args) >= 4:
                output = self.format_runtime_raw_output_target(node.args[0])
                return [
                    f"// HIP memory pool export to shareable handle: output: {output}, "
                    f"pool: {args[1]}, handle type: {args[2]}, flags: {args[3]}"
                ]
        elif name == "hipMemPoolImportFromShareableHandle":
            if len(args) >= 4:
                output = self.format_runtime_raw_output_target(node.args[0])
                return [
                    f"// HIP memory pool import from shareable handle: output: {output}, "
                    f"handle: {args[1]}, handle type: {args[2]}, flags: {args[3]}"
                ]
        elif name == "hipMemPoolExportPointer":
            if len(args) >= 2:
                output = self.format_runtime_raw_output_target(node.args[0])
                return [
                    f"// HIP memory pool export pointer: output: {output}, "
                    f"pointer: {args[1]}"
                ]
        elif name == "hipMemPoolImportPointer":
            if len(args) >= 3:
                output = self.format_runtime_raw_output_target(node.args[0])
                return [
                    f"// HIP memory pool import pointer: output: {output}, "
                    f"pool: {args[1]}, export data: {args[2]}"
                ]
        elif name == "hipDeviceGetDefaultMemPool":
            if len(args) >= 2:
                output = self.format_runtime_raw_output_target(node.args[0])
                return [
                    f"// HIP get default memory pool: output: {output}, "
                    f"device: {args[1]}"
                ]
        elif name == "hipDeviceSetMemPool":
            if len(args) >= 2:
                return [
                    f"// HIP set device memory pool: device: {args[0]}, pool: {args[1]}"
                ]
        elif name == "hipDeviceGetMemPool":
            if len(args) >= 2:
                output = self.format_runtime_raw_output_target(node.args[0])
                return [
                    f"// HIP get device memory pool: output: {output}, "
                    f"device: {args[1]}"
                ]
        elif name == "hipMemPrefetchAsync":
            if len(args) >= 4:
                return [
                    f"// HIP memory prefetch: pointer: {args[0]}, bytes: {args[1]}, "
                    f"device: {args[2]}, stream: {args[3]}"
                ]
        elif name == "hipMemPrefetchAsync_v2":
            if len(args) >= 5:
                return [
                    f"// HIP memory prefetch v2: pointer: {args[0]}, "
                    f"bytes: {args[1]}, location: {args[2]}, flags: {args[3]}, "
                    f"stream: {args[4]}"
                ]
        elif name == "hipMemAdvise":
            if len(args) >= 4:
                return [
                    f"// HIP memory advise: pointer: {args[0]}, bytes: {args[1]}, "
                    f"advice: {args[2]}, device: {args[3]}"
                ]
        elif name == "hipMemAdvise_v2":
            if len(args) >= 4:
                return [
                    f"// HIP memory advise v2: pointer: {args[0]}, "
                    f"bytes: {args[1]}, advice: {args[2]}, location: {args[3]}"
                ]
        elif name == "hipMemRangeGetAttribute":
            if len(args) >= 5:
                output = self.format_runtime_raw_output_target(node.args[0])
                output_name = self.get_runtime_pointer_target_name(node.args[0])
                if output_name is not None:
                    self.register_device_query_source(
                        output_name,
                        f"memory.rangeAttribute({args[2]}, {args[3]}, {args[4]})",
                    )
                return [
                    f"// HIP memory range get attribute: output: {output}, "
                    f"output bytes: {args[1]}, attribute: {args[2]}, "
                    f"pointer: {args[3]}, range bytes: {args[4]}"
                ]
        elif name == "hipMemRangeGetAttributes":
            if len(args) >= 6:
                return [
                    f"// HIP memory range get attributes: outputs: {args[0]}, "
                    f"output sizes: {args[1]}, attributes: {args[2]}, "
                    f"attribute count: {args[3]}, pointer: {args[4]}, "
                    f"range bytes: {args[5]}"
                ]
        elif name == "hipStreamAttachMemAsync":
            if len(args) >= 4:
                return [
                    f"// HIP stream attach memory: stream: {args[0]}, "
                    f"pointer: {args[1]}, bytes: {args[2]}, flags: {args[3]}"
                ]
        elif name == "hipMemGetAllocationGranularity":
            if len(args) >= 3:
                output = self.format_runtime_raw_output_target(node.args[0])
                output_name = self.get_runtime_pointer_target_name(node.args[0])
                properties = self.format_runtime_pointer_target(node.args[1])
                if output_name is not None:
                    self.register_device_query_source(
                        output_name,
                        "virtualMemory.allocationGranularity("
                        f"{properties}, {args[2]})",
                    )
                return [
                    f"// HIP virtual memory allocation granularity: output: {output}, "
                    f"properties: {args[1]}, option: {args[2]}"
                ]
        elif name == "hipMemCreate":
            if len(args) >= 4:
                output = self.format_runtime_raw_output_target(node.args[0])
                return [
                    f"// HIP virtual memory create allocation: output: {output}, "
                    f"bytes: {args[1]}, properties: {args[2]}, flags: {args[3]}"
                ]
        elif name == "hipMemRelease":
            if args:
                return [f"// HIP virtual memory release allocation: {args[0]}"]
        elif name == "hipMemAddressReserve":
            if len(args) >= 5:
                output = self.format_runtime_raw_output_target(node.args[0])
                return [
                    f"// HIP virtual memory reserve address: output: {output}, "
                    f"bytes: {args[1]}, alignment: {args[2]}, address: {args[3]}, "
                    f"flags: {args[4]}"
                ]
        elif name == "hipMemAddressFree":
            if len(args) >= 2:
                return [
                    f"// HIP virtual memory free address: pointer: {args[0]}, "
                    f"bytes: {args[1]}"
                ]
        elif name == "hipMemMap":
            if len(args) >= 5:
                return [
                    f"// HIP virtual memory map: pointer: {args[0]}, "
                    f"bytes: {args[1]}, offset: {args[2]}, handle: {args[3]}, "
                    f"flags: {args[4]}"
                ]
        elif name == "hipMemUnmap":
            if len(args) >= 2:
                return [
                    f"// HIP virtual memory unmap: pointer: {args[0]}, "
                    f"bytes: {args[1]}"
                ]
        elif name == "hipMemSetAccess":
            if len(args) >= 4:
                return [
                    f"// HIP virtual memory set access: pointer: {args[0]}, "
                    f"bytes: {args[1]}, descriptors: {args[2]}, count: {args[3]}"
                ]
        elif name == "hipMemGetAccess":
            if len(args) >= 3:
                output = self.format_runtime_raw_output_target(node.args[0])
                output_name = self.get_runtime_pointer_target_name(node.args[0])
                location = self.format_runtime_pointer_target(node.args[1])
                if output_name is not None:
                    self.register_device_query_source(
                        output_name,
                        f"virtualMemory.accessFlags({location}, {args[2]})",
                    )
                return [
                    f"// HIP virtual memory get access: output: {output}, "
                    f"location: {args[1]}, pointer: {args[2]}"
                ]
        elif name == "hipMemGetAllocationPropertiesFromHandle":
            if len(args) >= 2:
                output = self.format_runtime_raw_output_target(node.args[0])
                return [
                    f"// HIP virtual memory allocation properties: output: {output}, "
                    f"handle: {args[1]}"
                ]
        elif name == "hipMemRetainAllocationHandle":
            if len(args) >= 2:
                output = self.format_runtime_raw_output_target(node.args[0])
                return [
                    f"// HIP virtual memory retain allocation handle: output: {output}, "
                    f"address: {args[1]}"
                ]
        elif name == "hipMemExportToShareableHandle":
            if len(args) >= 4:
                output = self.format_runtime_raw_output_target(node.args[0])
                return [
                    f"// HIP virtual memory export shareable handle: output: {output}, "
                    f"handle: {args[1]}, handle type: {args[2]}, flags: {args[3]}"
                ]
        elif name == "hipMemImportFromShareableHandle":
            if len(args) >= 3:
                output = self.format_runtime_raw_output_target(node.args[0])
                return [
                    f"// HIP virtual memory import shareable handle: output: {output}, "
                    f"shareable handle: {args[1]}, handle type: {args[2]}"
                ]
        elif name == "hipHostRegister":
            if len(args) >= 3:
                return [
                    f"// HIP host memory register: {args[0]}, bytes: {args[1]}, "
                    f"flags: {args[2]}"
                ]
        elif name == "hipHostUnregister":
            if args:
                return [f"// HIP host memory unregister: {args[0]}"]
        elif name == "hipHostGetDevicePointer":
            if len(args) >= 3:
                output = self.format_runtime_raw_output_target(node.args[0])
                return [
                    f"// HIP host device pointer: output: {output}, host: {args[1]}, "
                    f"flags: {args[2]}"
                ]
        elif name == "hipHostGetFlags":
            if len(args) >= 2:
                output = self.format_runtime_raw_output_target(node.args[0])
                output_name = self.get_runtime_pointer_target_name(node.args[0])
                if output_name is not None:
                    self.register_device_query_source(
                        output_name,
                        f"hostMemory.flags({args[1]})",
                    )
                return [f"// HIP host memory flags: output: {output}, host: {args[1]}"]
        elif name in {"hipMemcpy", "hipMemcpyAsync", "hipMemcpyWithStream"}:
            if len(args) >= 4:
                comment = (
                    f"// HIP memory copy: {args[1]} -> {args[0]}, "
                    f"bytes: {args[2]}, kind: {args[3]}"
                )
                if len(args) >= 5:
                    comment += f", stream: {args[4]}"
                return [comment]
        elif name in {"hipMemcpyPeer", "hipMemcpyPeerAsync"}:
            if len(args) >= 5:
                comment = (
                    f"// HIP peer memory copy: source: {args[2]}, "
                    f"source device: {args[3]}, destination: {args[0]}, "
                    f"destination device: {args[1]}, bytes: {args[4]}"
                )
                if len(args) >= 6:
                    comment += f", stream: {args[5]}"
                return [comment]
        elif name in {"hipMemcpy2D", "hipMemcpy2DAsync"}:
            if len(args) >= 7:
                comment = (
                    f"// HIP 2D memory copy: {args[2]} -> {args[0]}, "
                    f"dst pitch: {args[1]}, src pitch: {args[3]}, "
                    f"width: {args[4]}, height: {args[5]}, kind: {args[6]}"
                )
                if len(args) >= 8:
                    comment += f", stream: {args[7]}"
                return [comment]
        elif name in {"hipMemcpyToArray", "hipMemcpyToArrayAsync"}:
            if len(args) >= 6:
                comment = (
                    f"// HIP memory copy to array: source: {args[3]}, "
                    f"destination array: {args[0]}, w offset: {args[1]}, "
                    f"h offset: {args[2]}, bytes: {args[4]}, kind: {args[5]}"
                )
                if len(args) >= 7:
                    comment += f", stream: {args[6]}"
                return [comment]
        elif name in {"hipMemcpyFromArray", "hipMemcpyFromArrayAsync"}:
            if len(args) >= 6:
                comment = (
                    f"// HIP memory copy from array: source array: {args[1]}, "
                    f"w offset: {args[2]}, h offset: {args[3]}, "
                    f"destination: {args[0]}, bytes: {args[4]}, kind: {args[5]}"
                )
                if len(args) >= 7:
                    comment += f", stream: {args[6]}"
                return [comment]
        elif name in {"hipMemcpy2DToArray", "hipMemcpy2DToArrayAsync"}:
            if len(args) >= 8:
                comment = (
                    f"// HIP 2D memory copy to array: source: {args[3]}, "
                    f"source pitch: {args[4]}, destination array: {args[0]}, "
                    f"w offset: {args[1]}, h offset: {args[2]}, width: {args[5]}, "
                    f"height: {args[6]}, kind: {args[7]}"
                )
                if len(args) >= 9:
                    comment += f", stream: {args[8]}"
                return [comment]
        elif name in {"hipMemcpy2DFromArray", "hipMemcpy2DFromArrayAsync"}:
            if len(args) >= 8:
                comment = (
                    f"// HIP 2D memory copy from array: source array: {args[2]}, "
                    f"w offset: {args[3]}, h offset: {args[4]}, "
                    f"destination: {args[0]}, destination pitch: {args[1]}, "
                    f"width: {args[5]}, height: {args[6]}, kind: {args[7]}"
                )
                if len(args) >= 9:
                    comment += f", stream: {args[8]}"
                return [comment]
        elif name == "hipMemcpyArrayToArray":
            if len(args) >= 8:
                return [
                    f"// HIP memory copy array to array: source array: {args[3]}, "
                    f"source w offset: {args[4]}, source h offset: {args[5]}, "
                    f"destination array: {args[0]}, destination w offset: {args[1]}, "
                    f"destination h offset: {args[2]}, bytes: {args[6]}, "
                    f"kind: {args[7]}"
                ]
        elif name == "hipMemcpy2DArrayToArray":
            if len(args) >= 9:
                return [
                    f"// HIP 2D memory copy array to array: source array: {args[3]}, "
                    f"source w offset: {args[4]}, source h offset: {args[5]}, "
                    f"destination array: {args[0]}, destination w offset: {args[1]}, "
                    f"destination h offset: {args[2]}, width: {args[6]}, "
                    f"height: {args[7]}, kind: {args[8]}"
                ]
        elif name in {"hipMemcpy3D", "hipMemcpy3DAsync"}:
            if args:
                params = self.format_runtime_pointer_target(node.args[0])
                comment = f"// HIP 3D memory copy: params: {params}"
                if len(args) >= 2:
                    comment += f", stream: {args[1]}"
                return [comment]
        elif name in {"hipMemcpy3DPeer", "hipMemcpy3DPeerAsync"}:
            if args:
                params = self.format_runtime_pointer_target(node.args[0])
                comment = f"// HIP 3D peer memory copy: params: {params}"
                if len(args) >= 2:
                    comment += f", stream: {args[1]}"
                return [comment]
        elif name in {"hipMemcpyParam2D", "hipMemcpyParam2DAsync"}:
            if args:
                params = self.format_runtime_pointer_target(node.args[0])
                comment = f"// HIP 2D parameterized memory copy: params: {params}"
                if len(args) >= 2:
                    comment += f", stream: {args[1]}"
                return [comment]
        elif name in {"hipDrvMemcpy3D", "hipDrvMemcpy3DAsync"}:
            if args:
                params = self.format_runtime_pointer_target(node.args[0])
                comment = f"// HIP driver 3D memory copy: params: {params}"
                if len(args) >= 2:
                    comment += f", stream: {args[1]}"
                return [comment]
        elif name == "hipMemcpyBatchAsync":
            if len(args) >= 9:
                fail_output = self.format_runtime_raw_output_target(node.args[7])
                return [
                    f"// HIP batched memory copy: destinations: {args[0]}, "
                    f"sources: {args[1]}, sizes: {args[2]}, count: {args[3]}, "
                    f"attributes: {args[4]}, attribute indices: {args[5]}, "
                    f"attribute count: {args[6]}, fail index output: {fail_output}, "
                    f"stream: {args[8]}"
                ]
        elif name == "hipMemcpy3DBatchAsync":
            if len(args) >= 5:
                fail_output = self.format_runtime_raw_output_target(node.args[2])
                return [
                    f"// HIP batched 3D memory copy: count: {args[0]}, "
                    f"operations: {args[1]}, fail index output: {fail_output}, "
                    f"flags: {args[3]}, stream: {args[4]}"
                ]
        elif name == "hipGetSymbolAddress":
            if len(args) >= 2:
                output = self.format_runtime_raw_output_target(node.args[0])
                return [
                    f"// HIP get symbol address: output: {output}, "
                    f"symbol: {args[1]}"
                ]
        elif name == "hipGetSymbolSize":
            if len(args) >= 2:
                output = self.format_runtime_raw_output_target(node.args[0])
                output_name = self.get_runtime_pointer_target_name(node.args[0])
                if output_name is not None:
                    self.register_device_query_source(
                        output_name,
                        f"symbol.size({args[1]})",
                    )
                return [f"// HIP get symbol size: output: {output}, symbol: {args[1]}"]
        elif name in {"hipMemcpyToSymbol", "hipMemcpyToSymbolAsync"}:
            if len(args) >= 3:
                comment = (
                    f"// HIP symbol copy to: {args[0]}, source: {args[1]}, "
                    f"bytes: {args[2]}"
                )
                if len(args) >= 4:
                    comment += f", offset: {args[3]}"
                if len(args) >= 5:
                    comment += f", kind: {args[4]}"
                if len(args) >= 6:
                    comment += f", stream: {args[5]}"
                return [comment]
        elif name in {"hipMemcpyFromSymbol", "hipMemcpyFromSymbolAsync"}:
            if len(args) >= 3:
                comment = (
                    f"// HIP symbol copy from: {args[1]}, destination: {args[0]}, "
                    f"bytes: {args[2]}"
                )
                if len(args) >= 4:
                    comment += f", offset: {args[3]}"
                if len(args) >= 5:
                    comment += f", kind: {args[4]}"
                if len(args) >= 6:
                    comment += f", stream: {args[5]}"
                return [comment]
        elif name in {"hipMemset", "hipMemsetAsync"}:
            if len(args) >= 3:
                comment = (
                    f"// HIP memory set: {args[0]}, value: {args[1]}, "
                    f"bytes: {args[2]}"
                )
                if len(args) >= 4:
                    comment += f", stream: {args[3]}"
                return [comment]
        elif name in {"hipMemset2D", "hipMemset2DAsync"}:
            if len(args) >= 5:
                comment = (
                    f"// HIP 2D memory set: {args[0]}, pitch: {args[1]}, "
                    f"value: {args[2]}, width: {args[3]}, height: {args[4]}"
                )
                if len(args) >= 6:
                    comment += f", stream: {args[5]}"
                return [comment]
        elif name in {"hipMemset3D", "hipMemset3DAsync"}:
            if len(args) >= 3:
                comment = (
                    f"// HIP 3D memory set: {args[0]}, value: {args[1]}, "
                    f"extent: {args[2]}"
                )
                if len(args) >= 4:
                    comment += f", stream: {args[3]}"
                return [comment]
        elif name == "hipMemAlloc":
            if len(args) >= 2:
                output = self.format_runtime_raw_output_target(node.args[0])
                return [
                    f"// HIP driver memory allocate: output: {output}, "
                    f"bytes: {args[1]}"
                ]
        elif name == "hipMemAllocPitch":
            if len(args) >= 5:
                output = self.format_runtime_raw_output_target(node.args[0])
                pitch = self.format_runtime_raw_output_target(node.args[1])
                return [
                    f"// HIP driver pitched memory allocate: output: {output}, "
                    f"pitch output: {pitch}, width: {args[2]}, height: {args[3]}, "
                    f"element bytes: {args[4]}"
                ]
        elif name == "hipMemFree":
            if args:
                return [f"// HIP driver memory free: {args[0]}"]
        elif name in {"hipMemAllocHost", "hipMemHostAlloc"}:
            if len(args) >= 2:
                output = self.format_runtime_raw_output_target(node.args[0])
                comment = (
                    f"// HIP driver host memory allocate: output: {output}, "
                    f"bytes: {args[1]}"
                )
                if len(args) >= 3:
                    comment += f", flags: {args[2]}"
                return [comment]
        elif name == "hipMemFreeHost":
            if args:
                return [f"// HIP driver host memory free: {args[0]}"]
        elif name == "hipMemHostGetDevicePointer":
            if len(args) >= 3:
                output = self.format_runtime_raw_output_target(node.args[0])
                return [
                    f"// HIP driver host device pointer: output: {output}, "
                    f"host: {args[1]}, flags: {args[2]}"
                ]
        elif name == "hipMemGetAddressRange":
            if len(args) >= 3:
                base_output = self.format_runtime_pointer_target(node.args[0])
                size_output = self.format_runtime_pointer_target(node.args[1])
                base_output_name = self.get_runtime_pointer_target_name(node.args[0])
                size_output_name = self.get_runtime_pointer_target_name(node.args[1])
                if base_output_name is not None:
                    self.register_device_query_source(
                        base_output_name,
                        f"memory.addressRange.base({args[2]})",
                    )
                if size_output_name is not None:
                    self.register_device_query_source(
                        size_output_name,
                        f"memory.addressRange.size({args[2]})",
                    )
                return [
                    f"// HIP driver memory address range: base output: {base_output}, "
                    f"size output: {size_output}, pointer: {args[2]}"
                ]
        elif name in {
            "hipMemcpyHtoD",
            "hipMemcpyHtoDAsync",
            "hipMemcpyDtoH",
            "hipMemcpyDtoHAsync",
            "hipMemcpyDtoD",
            "hipMemcpyDtoDAsync",
        }:
            if len(args) >= 3:
                copy_kind = {
                    "hipMemcpyHtoD": "host to device",
                    "hipMemcpyHtoDAsync": "host to device",
                    "hipMemcpyDtoH": "device to host",
                    "hipMemcpyDtoHAsync": "device to host",
                    "hipMemcpyDtoD": "device to device",
                    "hipMemcpyDtoDAsync": "device to device",
                }[name]
                comment = (
                    f"// HIP driver memory copy {copy_kind}: "
                    f"source: {args[1]}, destination: {args[0]}, bytes: {args[2]}"
                )
                if len(args) >= 4:
                    comment += f", stream: {args[3]}"
                return [comment]
        elif name in {"hipMemcpyAtoH", "hipMemcpyAtoHAsync"}:
            if len(args) >= 4:
                comment = (
                    "// HIP driver memory copy array to host: "
                    f"source array: {args[1]}, source offset: {args[2]}, "
                    f"destination host: {args[0]}, bytes: {args[3]}"
                )
                if len(args) >= 5:
                    comment += f", stream: {args[4]}"
                return [comment]
        elif name in {"hipMemcpyHtoA", "hipMemcpyHtoAAsync"}:
            if len(args) >= 4:
                comment = (
                    "// HIP driver memory copy host to array: "
                    f"source host: {args[2]}, destination array: {args[0]}, "
                    f"destination offset: {args[1]}, bytes: {args[3]}"
                )
                if len(args) >= 5:
                    comment += f", stream: {args[4]}"
                return [comment]
        elif name == "hipMemcpyAtoD":
            if len(args) >= 4:
                return [
                    "// HIP driver memory copy array to device: "
                    f"source array: {args[1]}, source offset: {args[2]}, "
                    f"destination device: {args[0]}, bytes: {args[3]}"
                ]
        elif name == "hipMemcpyDtoA":
            if len(args) >= 4:
                return [
                    "// HIP driver memory copy device to array: "
                    f"source device: {args[2]}, destination array: {args[0]}, "
                    f"destination offset: {args[1]}, bytes: {args[3]}"
                ]
        elif name == "hipMemcpyAtoA":
            if len(args) >= 5:
                return [
                    "// HIP driver memory copy array to array: "
                    f"source array: {args[2]}, source offset: {args[3]}, "
                    f"destination array: {args[0]}, destination offset: {args[1]}, "
                    f"bytes: {args[4]}"
                ]
        elif name in {
            "hipMemsetD2D8",
            "hipMemsetD2D8Async",
            "hipMemsetD2D16",
            "hipMemsetD2D16Async",
            "hipMemsetD2D32",
            "hipMemsetD2D32Async",
        }:
            if len(args) >= 5:
                width = {
                    "hipMemsetD2D8": "8-bit",
                    "hipMemsetD2D8Async": "8-bit",
                    "hipMemsetD2D16": "16-bit",
                    "hipMemsetD2D16Async": "16-bit",
                    "hipMemsetD2D32": "32-bit",
                    "hipMemsetD2D32Async": "32-bit",
                }[name]
                comment = (
                    f"// HIP driver 2D memory set {width}: pointer: {args[0]}, "
                    f"pitch: {args[1]}, value: {args[2]}, width: {args[3]}, "
                    f"height: {args[4]}"
                )
                if len(args) >= 6:
                    comment += f", stream: {args[5]}"
                return [comment]
        elif name in {
            "hipMemsetD8",
            "hipMemsetD8Async",
            "hipMemsetD16",
            "hipMemsetD16Async",
            "hipMemsetD32",
            "hipMemsetD32Async",
        }:
            if len(args) >= 3:
                width = {
                    "hipMemsetD8": "8-bit",
                    "hipMemsetD8Async": "8-bit",
                    "hipMemsetD16": "16-bit",
                    "hipMemsetD16Async": "16-bit",
                    "hipMemsetD32": "32-bit",
                    "hipMemsetD32Async": "32-bit",
                }[name]
                comment = (
                    f"// HIP driver memory set {width}: pointer: {args[0]}, "
                    f"value: {args[1]}, count: {args[2]}"
                )
                if len(args) >= 4:
                    comment += f", stream: {args[3]}"
                return [comment]
        elif name == "hipIpcGetMemHandle":
            if len(args) >= 2:
                output = self.format_runtime_raw_output_target(node.args[0])
                return [
                    f"// HIP IPC get memory handle: output: {output}, "
                    f"pointer: {args[1]}"
                ]
        elif name == "hipIpcOpenMemHandle":
            if len(args) >= 3:
                output = self.format_runtime_raw_output_target(node.args[0])
                return [
                    f"// HIP IPC open memory handle: output: {output}, "
                    f"handle: {args[1]}, flags: {args[2]}"
                ]
        elif name == "hipIpcCloseMemHandle":
            if args:
                return [f"// HIP IPC close memory handle: pointer: {args[0]}"]
        elif name == "hipIpcGetEventHandle":
            if len(args) >= 2:
                output = self.format_runtime_raw_output_target(node.args[0])
                return [
                    f"// HIP IPC get event handle: output: {output}, event: {args[1]}"
                ]
        elif name == "hipIpcOpenEventHandle":
            if len(args) >= 2:
                output = self.format_runtime_raw_output_target(node.args[0])
                return [
                    f"// HIP IPC open event handle: output: {output}, "
                    f"handle: {args[1]}"
                ]
        elif name in {
            "hipStreamWaitValue32",
            "hipStreamWaitValue64",
            "hipStreamWriteValue32",
            "hipStreamWriteValue64",
        }:
            if len(args) >= 4:
                operation = "wait" if "Wait" in name else "write"
                width = "32" if name.endswith("32") else "64"
                return [
                    f"// HIP stream {operation} value{width}: stream: {args[0]}, "
                    f"address: {args[1]}, value: {args[2]}, flags: {args[3]}"
                ]
        elif name == "hipStreamBatchMemOp":
            if len(args) >= 4:
                return [
                    f"// HIP stream batch memory op: stream: {args[0]}, "
                    f"count: {args[1]}, params: {args[2]}, flags: {args[3]}"
                ]
        elif name in {"hipStreamSynchronize"}:
            if args:
                return [f"// HIP synchronize: {args[0]}"]
        elif name == "hipStreamAddCallback":
            if len(args) >= 4:
                return [
                    f"// HIP stream add callback: stream: {args[0]}, "
                    f"callback: {args[1]}, user data: {args[2]}, flags: {args[3]}"
                ]
        elif name == "hipLaunchHostFunc":
            if len(args) >= 3:
                return [
                    f"// HIP launch host function: stream: {args[0]}, "
                    f"function: {args[1]}, user data: {args[2]}"
                ]
        elif name in {"hipConfigureCall", "__hipPushCallConfiguration"}:
            if len(args) >= 4:
                action = (
                    "push call configuration"
                    if name.startswith("__")
                    else "configure call"
                )
                return [
                    f"// HIP {action}: grid: {args[0]}, block: {args[1]}, "
                    f"shared memory: {args[2]}, stream: {args[3]}"
                ]
        elif name == "__hipPopCallConfiguration":
            if len(args) >= 4:
                grid_output = self.format_runtime_raw_output_target(node.args[0])
                block_output = self.format_runtime_raw_output_target(node.args[1])
                shared_output = self.format_runtime_raw_output_target(node.args[2])
                stream_output = self.format_runtime_raw_output_target(node.args[3])
                return [
                    f"// HIP pop call configuration: grid output: {grid_output}, "
                    f"block output: {block_output}, shared memory output: "
                    f"{shared_output}, stream output: {stream_output}"
                ]
        elif name == "hipSetupArgument":
            if len(args) >= 3:
                return [
                    f"// HIP setup kernel argument: value: {args[0]}, "
                    f"bytes: {args[1]}, offset: {args[2]}"
                ]
        elif name == "hipLaunchByPtr":
            if args:
                return [f"// HIP launch by pointer: function: {args[0]}"]
        elif name == "hipLaunchKernelExC":
            if len(args) >= 3:
                return [
                    f"// HIP launch kernel ex: config: {args[0]}, "
                    f"function: {args[1]}, args: {args[2]}"
                ]
        elif name == "hipDrvLaunchKernelEx":
            if len(args) >= 4:
                return [
                    f"// HIP driver launch kernel ex: config: {args[0]}, "
                    f"function: {args[1]}, params: {args[2]}, extra: {args[3]}"
                ]
        elif name == "hipExtLaunchKernel":
            if len(args) >= 9:
                function = self.format_runtime_pointer_target(node.args[0])
                return [
                    f"// HIP extended kernel launch: function: {function}, "
                    f"grid: {args[1]}, block: {args[2]}, args: {args[3]}, "
                    f"shared memory: {args[4]}, stream: {args[5]}, "
                    f"start event: {args[6]}, stop event: {args[7]}, "
                    f"flags: {args[8]}"
                ]
        elif name == "hipExtLaunchKernelGGL":
            if len(args) >= 8:
                call_args = ", ".join(args[8:]) if len(args) > 8 else "<none>"
                return [
                    f"// HIP extended kernel launch GGL: function: {args[0]}, "
                    f"grid: {args[1]}, block: {args[2]}, "
                    f"shared memory: {args[3]}, stream: {args[4]}, "
                    f"start event: {args[5]}, stop event: {args[6]}, "
                    f"flags: {args[7]}, args: {call_args}"
                ]
        elif name == "hipExtLaunchMultiKernelMultiDevice":
            if len(args) >= 3:
                return [
                    f"// HIP extended multi-kernel multi-device launch: "
                    f"params: {args[0]}, devices: {args[1]}, flags: {args[2]}"
                ]
        elif name == "hipDeviceGetStreamPriorityRange":
            if len(args) >= 2:
                least_output = self.format_runtime_raw_output_target(node.args[0])
                greatest_output = self.format_runtime_raw_output_target(node.args[1])
                least_output_name = self.get_runtime_pointer_target_name(node.args[0])
                greatest_output_name = self.get_runtime_pointer_target_name(
                    node.args[1]
                )
                if least_output_name is not None:
                    self.register_device_query_source(
                        least_output_name, "streamPriorityRange.least"
                    )
                if greatest_output_name is not None:
                    self.register_device_query_source(
                        greatest_output_name, "streamPriorityRange.greatest"
                    )
                return [
                    f"// HIP get stream priority range: "
                    f"least output: {least_output}, greatest output: {greatest_output}"
                ]
        elif name == "hipStreamGetFlags":
            if len(args) >= 2:
                output = self.format_runtime_raw_output_target(node.args[1])
                output_name = self.get_runtime_pointer_target_name(node.args[1])
                if output_name is not None:
                    self.register_device_query_source(
                        output_name, f"stream.flags({args[0]})"
                    )
                return [f"// HIP get stream flags: stream: {args[0]}, output: {output}"]
        elif name == "hipStreamGetPriority":
            if len(args) >= 2:
                output = self.format_runtime_raw_output_target(node.args[1])
                output_name = self.get_runtime_pointer_target_name(node.args[1])
                if output_name is not None:
                    self.register_device_query_source(
                        output_name, f"stream.priority({args[0]})"
                    )
                return [
                    f"// HIP get stream priority: stream: {args[0]}, output: {output}"
                ]
        elif name == "hipStreamBeginCapture":
            if len(args) >= 2:
                return [
                    f"// HIP stream begin capture: stream: {args[0]}, mode: {args[1]}"
                ]
        elif name == "hipStreamEndCapture":
            if len(args) >= 2:
                output = self.format_runtime_raw_output_target(node.args[1])
                return [
                    f"// HIP stream end capture: stream: {args[0]}, graph output: {output}"
                ]
        elif name == "hipStreamIsCapturing":
            if len(args) >= 2:
                output = self.format_runtime_raw_output_target(node.args[1])
                output_name = self.get_runtime_pointer_target_name(node.args[1])
                if output_name is not None:
                    self.register_device_query_source(
                        output_name, f"stream.captureStatus({args[0]})"
                    )
                return [
                    f"// HIP stream is capturing: stream: {args[0]}, output: {output}"
                ]
        elif name in {"hipStreamGetCaptureInfo", "hipStreamGetCaptureInfo_v2"}:
            if len(args) >= 3:
                status_output = self.format_runtime_raw_output_target(node.args[1])
                id_output = self.format_runtime_raw_output_target(node.args[2])
                status_output_name = self.get_runtime_pointer_target_name(node.args[1])
                id_output_name = self.get_runtime_pointer_target_name(node.args[2])
                if status_output_name is not None:
                    self.register_device_query_source(
                        status_output_name, f"stream.captureStatus({args[0]})"
                    )
                if id_output_name is not None:
                    self.register_device_query_source(
                        id_output_name, f"stream.captureId({args[0]})"
                    )
                comment = (
                    f"// HIP stream capture info: stream: {args[0]}, "
                    f"status output: {status_output}, id output: {id_output}"
                )
                if len(args) >= 6:
                    graph_output = self.format_runtime_raw_output_target(node.args[3])
                    dependencies_output = self.format_runtime_raw_output_target(
                        node.args[4]
                    )
                    count_output = self.format_runtime_raw_output_target(node.args[5])
                    count_output_name = self.get_runtime_pointer_target_name(
                        node.args[5]
                    )
                    if count_output_name is not None:
                        self.register_device_query_source(
                            count_output_name,
                            f"stream.captureDependencyCount({args[0]})",
                        )
                    comment += (
                        f", graph output: {graph_output}, "
                        f"dependencies output: {dependencies_output}, "
                        f"dependency count output: {count_output}"
                    )
                return [comment]
        elif name == "hipStreamUpdateCaptureDependencies":
            if len(args) >= 4:
                return [
                    f"// HIP stream update capture dependencies: stream: {args[0]}, "
                    f"dependencies: {args[1]}, count: {args[2]}, flags: {args[3]}"
                ]
        elif name == "hipDeviceSynchronize":
            return ["// HIP device synchronize"]
        elif name in {
            "hipStreamCreate",
            "hipStreamCreateWithFlags",
            "hipStreamCreateWithPriority",
            "hipStreamDestroy",
        }:
            if args:
                action = "destroy" if name == "hipStreamDestroy" else "create"
                stream = (
                    self.format_runtime_pointer_target(node.args[0])
                    if action == "create"
                    else args[0]
                )
                comment = f"// HIP stream {action}: {stream}"
                if (
                    name in {
                        "hipStreamCreateWithFlags",
                        "hipStreamCreateWithPriority",
                    }
                    and len(args) >= 2
                ):
                    comment += f", flags: {args[1]}"
                if name == "hipStreamCreateWithPriority" and len(args) >= 3:
                    comment += f", priority: {args[2]}"
                return [comment]
        elif name in {"hipEventCreate", "hipEventCreateWithFlags"}:
            if args:
                event = self.format_runtime_pointer_target(node.args[0])
                comment = f"// HIP event create: {event}"
                if len(args) >= 2:
                    comment += f", flags: {args[1]}"
                return [comment]
        elif name == "hipEventRecord":
            if args:
                comment = f"// HIP event record: {args[0]}"
                if len(args) >= 2:
                    comment += f", stream: {args[1]}"
                return [comment]
        elif name == "hipEventSynchronize":
            if args:
                return [f"// HIP event synchronize: {args[0]}"]
        elif name == "hipEventElapsedTime":
            if len(node.args) >= 3:
                output = self.format_runtime_raw_output_target(node.args[0])
                output_name = self.get_runtime_pointer_target_name(node.args[0])
                if output_name is not None:
                    self.register_device_query_source(
                        output_name,
                        f"event.elapsedTime({args[1]}, {args[2]})",
                    )
                return [
                    f"// HIP event elapsed time: {args[1]} -> {args[2]}, "
                    f"output: {output}"
                ]
        elif name == "hipEventDestroy":
            if args:
                return [f"// HIP event destroy: {args[0]}"]
        elif name == "hipEventQuery":
            if args:
                return [f"// HIP event query: {args[0]}"]
        elif name == "hipStreamWaitEvent":
            if len(args) >= 2:
                comment = f"// HIP stream wait event: {args[0]} waits for {args[1]}"
                if len(args) >= 3:
                    comment += f", flags: {args[2]}"
                return [comment]
        elif name == "hipGetLastError":
            return ["// HIP get last error"]
        elif name == "hipPeekAtLastError":
            return ["// HIP peek at last error"]
        elif name == "hipGetDevice":
            if args:
                output = self.format_runtime_raw_output_target(node.args[0])
                output_name = self.get_runtime_pointer_target_name(node.args[0])
                if output_name is not None:
                    self.register_device_query_source(output_name, "currentDevice")
                return [f"// HIP get current device: output: {output}"]
        elif name == "hipGetDeviceCount":
            if args:
                output = self.format_runtime_raw_output_target(node.args[0])
                output_name = self.get_runtime_pointer_target_name(node.args[0])
                if output_name is not None:
                    self.register_device_query_source(output_name, "deviceCount")
                return [f"// HIP get device count: output: {output}"]
        elif name == "hipSetDevice":
            if args:
                return [f"// HIP set device: {args[0]}"]
        elif name == "hipSetValidDevices":
            if len(args) >= 2:
                return [
                    f"// HIP set valid devices: devices: {args[0]}, count: {args[1]}"
                ]
        elif name == "hipGetDeviceProperties":
            if len(args) >= 2:
                output = self.format_runtime_raw_output_target(node.args[0])
                output_name = self.get_runtime_pointer_target_name(node.args[0])
                if output_name is not None:
                    self.register_device_property_source(output_name, args[1])
                return [f"// HIP get device properties: {output}, device: {args[1]}"]
        elif name == "hipDeviceGetAttribute":
            if len(args) >= 3:
                output = self.format_runtime_raw_output_target(node.args[0])
                output_name = self.get_runtime_pointer_target_name(node.args[0])
                if output_name is not None:
                    self.register_device_attribute_source(output_name, args[1], args[2])
                return [
                    f"// HIP get device attribute: output: {output}, "
                    f"attribute: {args[1]}, device: {args[2]}"
                ]
        elif name == "hipDeviceGetName":
            if len(args) >= 3:
                output = self.format_runtime_raw_output_target(node.args[0])
                return [
                    f"// HIP get device name: output: {output}, "
                    f"length: {args[1]}, device: {args[2]}"
                ]
        elif name == "hipDeviceGetUuid":
            if len(args) >= 2:
                output = self.format_runtime_raw_output_target(node.args[0])
                return [f"// HIP get device UUID: output: {output}, device: {args[1]}"]
        elif name == "hipDeviceTotalMem":
            if len(args) >= 2:
                output = self.format_runtime_raw_output_target(node.args[0])
                output_name = self.get_runtime_pointer_target_name(node.args[0])
                if output_name is not None:
                    self.register_device_query_source(output_name, "totalMem", args[1])
                return [
                    f"// HIP get device total memory: output: {output}, "
                    f"device: {args[1]}"
                ]
        elif name == "hipDeviceComputeCapability":
            if len(args) >= 3:
                major_output = self.format_runtime_raw_output_target(node.args[0])
                minor_output = self.format_runtime_raw_output_target(node.args[1])
                major_output_name = self.get_runtime_pointer_target_name(node.args[0])
                minor_output_name = self.get_runtime_pointer_target_name(node.args[1])
                if major_output_name is not None:
                    self.register_device_query_source(
                        major_output_name, "computeCapability.major", args[2]
                    )
                if minor_output_name is not None:
                    self.register_device_query_source(
                        minor_output_name, "computeCapability.minor", args[2]
                    )
                return [
                    f"// HIP get device compute capability: "
                    f"major output: {major_output}, minor output: {minor_output}, "
                    f"device: {args[2]}"
                ]
        elif name == "hipChooseDevice":
            if len(args) >= 2:
                output = self.format_runtime_raw_output_target(node.args[0])
                output_name = self.get_runtime_pointer_target_name(node.args[0])
                if output_name is not None:
                    self.register_device_query_source(output_name, "selectedDevice")
                return [
                    f"// HIP choose device: output: {output}, properties: {args[1]}"
                ]
        elif name == "hipExtGetLinkTypeAndHopCount":
            if len(args) >= 4:
                link_type_output = self.format_runtime_raw_output_target(node.args[2])
                hop_count_output = self.format_runtime_raw_output_target(node.args[3])
                link_type_output_name = self.get_runtime_pointer_target_name(
                    node.args[2]
                )
                hop_count_output_name = self.get_runtime_pointer_target_name(
                    node.args[3]
                )
                if link_type_output_name is not None:
                    self.register_device_query_source(
                        link_type_output_name, f"linkType({args[0]}, {args[1]})"
                    )
                if hop_count_output_name is not None:
                    self.register_device_query_source(
                        hop_count_output_name, f"hopCount({args[0]}, {args[1]})"
                    )
                return [
                    f"// HIP get link type and hop count: device 1: {args[0]}, "
                    f"device 2: {args[1]}, link type output: {link_type_output}, "
                    f"hop count output: {hop_count_output}"
                ]
        elif name == "hipDeviceGetPCIBusId":
            if len(args) >= 3:
                output = self.format_runtime_raw_output_target(node.args[0])
                return [
                    f"// HIP get device PCI bus id: output: {output}, "
                    f"length: {args[1]}, device: {args[2]}"
                ]
        elif name == "hipDeviceGetByPCIBusId":
            if len(args) >= 2:
                output = self.format_runtime_raw_output_target(node.args[0])
                output_name = self.get_runtime_pointer_target_name(node.args[0])
                if output_name is not None:
                    self.register_device_query_source(
                        output_name, f"deviceByPCIBusId({args[1]})"
                    )
                return [
                    f"// HIP get device by PCI bus id: output: {output}, "
                    f"bus id: {args[1]}"
                ]
        elif name == "hipDeviceGetCacheConfig":
            if args:
                output = self.format_runtime_raw_output_target(node.args[0])
                output_name = self.get_runtime_pointer_target_name(node.args[0])
                if output_name is not None:
                    self.register_device_query_source(output_name, "cacheConfig")
                return [f"// HIP get device cache config: output: {output}"]
        elif name == "hipDeviceSetCacheConfig":
            if args:
                return [f"// HIP set device cache config: {args[0]}"]
        elif name == "hipDeviceGetSharedMemConfig":
            if args:
                output = self.format_runtime_raw_output_target(node.args[0])
                output_name = self.get_runtime_pointer_target_name(node.args[0])
                if output_name is not None:
                    self.register_device_query_source(output_name, "sharedMemConfig")
                return [f"// HIP get device shared memory config: output: {output}"]
        elif name == "hipDeviceSetSharedMemConfig":
            if args:
                return [f"// HIP set device shared memory config: {args[0]}"]
        elif name == "hipDeviceGetLimit":
            if len(args) >= 2:
                output = self.format_runtime_raw_output_target(node.args[0])
                output_name = self.get_runtime_pointer_target_name(node.args[0])
                if output_name is not None:
                    self.register_device_query_source(output_name, f"limit.{args[1]}")
                return [f"// HIP get device limit: output: {output}, limit: {args[1]}"]
        elif name == "hipDeviceSetLimit":
            if len(args) >= 2:
                return [f"// HIP set device limit: limit: {args[0]}, value: {args[1]}"]
        elif name == "hipDeviceReset":
            return ["// HIP device reset"]
        elif name == "hipGetDeviceFlags":
            if args:
                output = self.format_runtime_raw_output_target(node.args[0])
                output_name = self.get_runtime_pointer_target_name(node.args[0])
                if output_name is not None:
                    self.register_device_query_source(output_name, "deviceFlags")
                return [f"// HIP get device flags: output: {output}"]
        elif name == "hipSetDeviceFlags":
            if args:
                return [f"// HIP set device flags: {args[0]}"]
        elif name == "hipGetProcAddress":
            if len(args) >= 5:
                output = self.format_runtime_raw_output_target(node.args[1])
                status_output = self.format_runtime_raw_output_target(node.args[4])
                return [
                    f"// HIP get proc address: symbol: {args[0]}, output: {output}, "
                    f"version: {args[2]}, flags: {args[3]}, "
                    f"status output: {status_output}"
                ]
        elif name == "hipMemGetInfo":
            if len(args) >= 2:
                free_output = self.format_runtime_raw_output_target(node.args[0])
                total_output = self.format_runtime_raw_output_target(node.args[1])
                free_output_name = self.get_runtime_pointer_target_name(node.args[0])
                total_output_name = self.get_runtime_pointer_target_name(node.args[1])
                if free_output_name is not None:
                    self.register_device_query_source(
                        free_output_name,
                        "memory.info.free",
                    )
                if total_output_name is not None:
                    self.register_device_query_source(
                        total_output_name,
                        "memory.info.total",
                    )
                return [
                    f"// HIP memory info: free output: {free_output}, "
                    f"total output: {total_output}"
                ]
        elif name == "hipPointerGetAttributes":
            if len(args) >= 2:
                output = self.format_runtime_raw_output_target(node.args[0])
                return [
                    f"// HIP pointer attributes: output: {output}, pointer: {args[1]}"
                ]
        elif name == "hipDrvPointerGetAttributes":
            if len(args) >= 4:
                return [
                    f"// HIP driver pointer attributes: count: {args[0]}, "
                    f"attributes: {args[1]}, data: {args[2]}, pointer: {args[3]}"
                ]
        elif name == "hipPointerGetAttribute":
            if len(args) >= 3:
                output = self.format_runtime_raw_output_target(node.args[0])
                output_name = self.get_runtime_pointer_target_name(node.args[0])
                if output_name is not None:
                    self.register_device_query_source(
                        output_name,
                        f"pointer.attribute({args[1]}, {args[2]})",
                    )
                return [
                    f"// HIP pointer attribute: output: {output}, "
                    f"attribute: {args[1]}, pointer: {args[2]}"
                ]
        elif name == "hipPointerSetAttribute":
            if len(args) >= 3:
                value = self.format_runtime_pointer_target(node.args[0])
                return [
                    f"// HIP pointer set attribute: value: {value}, "
                    f"attribute: {args[1]}, pointer: {args[2]}"
                ]
        elif name == "hipMemPtrGetInfo":
            if len(args) >= 2:
                size_output = self.format_runtime_raw_output_target(node.args[1])
                size_output_name = self.get_runtime_pointer_target_name(node.args[1])
                if size_output_name is not None:
                    self.register_device_query_source(
                        size_output_name,
                        f"memoryPointer.size({args[0]})",
                    )
                return [
                    f"// HIP memory pointer info: pointer: {args[0]}, "
                    f"size output: {size_output}"
                ]
        elif name in {
            "hipOccupancyMaxPotentialBlockSize",
            "hipOccupancyMaxPotentialBlockSizeVariableSMem",
            "hipOccupancyMaxPotentialBlockSizeVariableSMemWithFlags",
        }:
            if len(args) >= 5:
                grid_output = self.format_runtime_raw_output_target(node.args[0])
                block_output = self.format_runtime_raw_output_target(node.args[1])
                grid_output_name = self.get_runtime_pointer_target_name(node.args[0])
                block_output_name = self.get_runtime_pointer_target_name(node.args[1])
                query_kind = {
                    "hipOccupancyMaxPotentialBlockSize": "maxPotentialBlockSize",
                    "hipOccupancyMaxPotentialBlockSizeVariableSMem": (
                        "maxPotentialBlockSizeVariableSMem"
                    ),
                    "hipOccupancyMaxPotentialBlockSizeVariableSMemWithFlags": (
                        "maxPotentialBlockSizeVariableSMemWithFlags"
                    ),
                }[name]
                query_operands = [args[2], args[3], args[4]]
                if len(args) >= 6:
                    query_operands.append(args[5])
                query_operands_text = ", ".join(query_operands)
                if grid_output_name is not None:
                    self.register_device_query_source(
                        grid_output_name,
                        f"occupancy.{query_kind}.grid({query_operands_text})",
                    )
                if block_output_name is not None:
                    self.register_device_query_source(
                        block_output_name,
                        f"occupancy.{query_kind}.block({query_operands_text})",
                    )
                comment = (
                    f"// HIP occupancy max potential block size: "
                    f"grid output: {grid_output}, block output: {block_output}, "
                    f"kernel: {args[2]}, dynamic shared memory: {args[3]}, "
                    f"block size limit: {args[4]}"
                )
                if len(args) >= 6:
                    comment += f", flags: {args[5]}"
                return [comment]
        elif name in {
            "hipOccupancyMaxActiveBlocksPerMultiprocessor",
            "hipOccupancyMaxActiveBlocksPerMultiprocessorWithFlags",
        }:
            if len(args) >= 4:
                output = self.format_runtime_raw_output_target(node.args[0])
                output_name = self.get_runtime_pointer_target_name(node.args[0])
                query_kind = (
                    "maxActiveBlocksPerMultiprocessorWithFlags"
                    if name.endswith("WithFlags")
                    else "maxActiveBlocksPerMultiprocessor"
                )
                query_operands = [args[1], args[2], args[3]]
                if len(args) >= 5:
                    query_operands.append(args[4])
                if output_name is not None:
                    self.register_device_query_source(
                        output_name,
                        f"occupancy.{query_kind}({', '.join(query_operands)})",
                    )
                comment = (
                    f"// HIP occupancy active blocks per multiprocessor: "
                    f"output: {output}, kernel: {args[1]}, "
                    f"block size: {args[2]}, dynamic shared memory: {args[3]}"
                )
                if len(args) >= 5:
                    comment += f", flags: {args[4]}"
                return [comment]
        elif name == "hipGetFuncBySymbol":
            if len(args) >= 2:
                output = self.format_runtime_pointer_target(node.args[0])
                self.clear_lvalue_metadata_source(node.args[0])
                return [
                    f"// HIP get function by symbol: output: {output}, symbol: {args[1]}"
                ]
        elif name == "hipFuncGetAttribute":
            if len(args) >= 3:
                output = self.format_runtime_raw_output_target(node.args[0])
                output_name = self.get_runtime_pointer_target_name(node.args[0])
                if output_name is not None:
                    self.register_device_query_source(
                        output_name,
                        f"function.attribute({args[1]}, {args[2]})",
                    )
                return [
                    f"// HIP function get attribute: output: {output}, "
                    f"attribute: {args[1]}, function: {args[2]}"
                ]
        elif name == "hipFuncGetAttributes":
            if len(args) >= 2:
                output = self.format_runtime_raw_output_target(node.args[0])
                output_name = self.get_runtime_pointer_target_name(node.args[0])
                if output_name is not None:
                    self.register_member_query_sources(
                        output_name,
                        {
                            member: f"function.attributes.{member}({args[1]})"
                            for member in self.HIP_FUNCTION_ATTRIBUTE_MEMBERS
                        },
                    )
                return [
                    f"// HIP function get attributes: output: {output}, "
                    f"function: {args[1]}"
                ]
        elif name == "hipFuncSetAttribute":
            if len(args) >= 3:
                return [
                    f"// HIP function set attribute: function: {args[0]}, "
                    f"attribute: {args[1]}, value: {args[2]}"
                ]
        elif name == "hipFuncSetCacheConfig":
            if len(args) >= 2:
                return [
                    f"// HIP function set cache config: function: {args[0]}, "
                    f"config: {args[1]}"
                ]
        elif name == "hipFuncSetSharedMemConfig":
            if len(args) >= 2:
                return [
                    f"// HIP function set shared memory config: function: {args[0]}, "
                    f"config: {args[1]}"
                ]
        elif name in {"hipModuleLoad", "hipModuleLoadData"}:
            if len(args) >= 2:
                output = self.format_runtime_raw_output_target(node.args[0])
                source_label = "file" if name == "hipModuleLoad" else "image"
                return [
                    f"// HIP module load: output: {output}, "
                    f"{source_label}: {args[1]}"
                ]
        elif name == "hipModuleLoadDataEx":
            if len(args) >= 5:
                output = self.format_runtime_raw_output_target(node.args[0])
                return [
                    f"// HIP module load data ex: output: {output}, image: {args[1]}, "
                    f"options: {args[2]}, option keys: {args[3]}, "
                    f"option values: {args[4]}"
                ]
        elif name == "hipModuleLoadFatBinary":
            if len(args) >= 2:
                output = self.format_runtime_raw_output_target(node.args[0])
                return [
                    f"// HIP module load fat binary: output: {output}, "
                    f"fat binary: {args[1]}"
                ]
        elif name == "hipModuleUnload":
            if args:
                return [f"// HIP module unload: {args[0]}"]
        elif name == "hipModuleGetFunction":
            if len(args) >= 3:
                output = self.format_runtime_pointer_target(node.args[0])
                self.clear_lvalue_metadata_source(node.args[0])
                return [
                    f"// HIP module get function: output: {output}, "
                    f"module: {args[1]}, name: {args[2]}"
                ]
        elif name == "hipModuleGetFunctionCount":
            if len(args) >= 2:
                output = self.format_runtime_pointer_target(node.args[0])
                output_name = self.get_runtime_pointer_target_name(node.args[0])
                if output_name is not None:
                    self.register_device_query_source(
                        output_name,
                        f"module.functionCount({args[1]})",
                    )
                else:
                    self.clear_lvalue_metadata_source(node.args[0])
                return [
                    f"// HIP module get function count: output: {output}, "
                    f"module: {args[1]}"
                ]
        elif name == "hipModuleGetGlobal":
            if len(args) >= 4:
                pointer_output = self.format_runtime_pointer_target(node.args[0])
                size_output = self.format_runtime_pointer_target(node.args[1])
                size_output_name = self.get_runtime_pointer_target_name(node.args[1])
                self.clear_lvalue_metadata_source(node.args[0])
                if size_output_name is not None:
                    self.register_device_query_source(
                        size_output_name,
                        f"module.global.size({args[2]}, {args[3]})",
                    )
                else:
                    self.clear_lvalue_metadata_source(node.args[1])
                return [
                    f"// HIP module get global: pointer output: {pointer_output}, "
                    f"size output: {size_output}, module: {args[2]}, name: {args[3]}"
                ]
        elif name == "hipModuleGetTexRef":
            if len(args) >= 3:
                output = self.format_runtime_pointer_target(node.args[0])
                self.clear_lvalue_metadata_source(node.args[0])
                return [
                    f"// HIP module get texture reference: output: {output}, "
                    f"module: {args[1]}, name: {args[2]}"
                ]
        elif name == "hipBindTexture":
            if len(args) >= 5:
                offset = self.format_runtime_raw_output_target(node.args[0])
                return [
                    "// HIP texture reference bind: "
                    f"offset output: {offset}, texture: {args[1]}, "
                    f"pointer: {args[2]}, desc: {args[3]}, bytes: {args[4]}"
                ]
        elif name == "hipBindTexture2D":
            if len(args) >= 7:
                offset = self.format_runtime_raw_output_target(node.args[0])
                return [
                    "// HIP texture reference bind 2D: "
                    f"offset output: {offset}, texture: {args[1]}, "
                    f"pointer: {args[2]}, desc: {args[3]}, width: {args[4]}, "
                    f"height: {args[5]}, pitch: {args[6]}"
                ]
        elif name == "hipBindTextureToArray":
            if len(args) >= 3:
                return [
                    "// HIP texture reference bind array: "
                    f"texture: {args[0]}, array: {args[1]}, desc: {args[2]}"
                ]
        elif name == "hipBindTextureToMipmappedArray":
            if len(args) >= 3:
                return [
                    "// HIP texture reference bind mipmapped array: "
                    f"texture: {args[0]}, mipmapped array: {args[1]}, "
                    f"desc: {args[2]}"
                ]
        elif name == "hipGetTextureReference":
            if len(args) >= 2:
                output = self.format_runtime_pointer_target(node.args[0])
                self.clear_lvalue_metadata_source(node.args[0])
                return [
                    f"// HIP get texture reference: output: {output}, "
                    f"symbol: {args[1]}"
                ]
        elif name == "hipGetTextureAlignmentOffset":
            if len(args) >= 2:
                output = self.format_runtime_raw_output_target(node.args[0])
                output_name = self.get_runtime_pointer_target_name(node.args[0])
                if output_name is not None:
                    self.register_device_query_source(
                        output_name,
                        f"textureReference.alignmentOffset({args[1]})",
                    )
                return [
                    "// HIP texture alignment offset query: "
                    f"output: {output}, texture: {args[1]}"
                ]
        elif name == "hipUnbindTexture":
            if args:
                return [f"// HIP texture reference unbind: {args[0]}"]
        elif name == "hipTexRefGetAddress":
            if len(args) >= 2:
                output = self.format_runtime_pointer_target(node.args[0])
                self.clear_lvalue_metadata_source(node.args[0])
                return [
                    "// HIP texture reference get address: "
                    f"output: {output}, texture: {args[1]}"
                ]
        elif name == "hipTexRefGetAddressMode":
            if len(args) >= 3:
                output = self.format_runtime_raw_output_target(node.args[0])
                output_name = self.get_runtime_pointer_target_name(node.args[0])
                if output_name is not None:
                    self.register_device_query_source(
                        output_name,
                        f"textureReference.addressMode({args[1]}, {args[2]})",
                    )
                return [
                    "// HIP texture reference get address mode: "
                    f"output: {output}, texture: {args[1]}, dim: {args[2]}"
                ]
        elif name in {
            "hipTexRefGetArray",
            "hipTexRefGetBorderColor",
            "hipTexRefGetFilterMode",
            "hipTexRefGetFlags",
            "hipTexRefGetMaxAnisotropy",
            "hipTexRefGetMipmapFilterMode",
            "hipTexRefGetMipmapLevelBias",
            "hipTexRefGetMipMappedArray",
            "hipTexRefGetMipmappedArray",
        }:
            if len(args) >= 2:
                output = self.format_runtime_raw_output_target(node.args[0])
                output_name = self.get_runtime_pointer_target_name(node.args[0])
                labels = {
                    "hipTexRefGetArray": "array",
                    "hipTexRefGetBorderColor": "border color",
                    "hipTexRefGetFilterMode": "filter mode",
                    "hipTexRefGetFlags": "flags",
                    "hipTexRefGetMaxAnisotropy": "max anisotropy",
                    "hipTexRefGetMipmapFilterMode": "mipmap filter mode",
                    "hipTexRefGetMipmapLevelBias": "mipmap level bias",
                    "hipTexRefGetMipMappedArray": "mipmapped array",
                    "hipTexRefGetMipmappedArray": "mipmapped array",
                }
                query_names = {
                    "hipTexRefGetFilterMode": "filterMode",
                    "hipTexRefGetFlags": "flags",
                    "hipTexRefGetMaxAnisotropy": "maxAnisotropy",
                    "hipTexRefGetMipmapFilterMode": "mipmapFilterMode",
                    "hipTexRefGetMipmapLevelBias": "mipmapLevelBias",
                }
                query_name = query_names.get(name)
                if output_name is not None and query_name is not None:
                    self.register_device_query_source(
                        output_name,
                        f"textureReference.{query_name}({args[1]})",
                    )
                elif output_name is not None and name == "hipTexRefGetBorderColor":
                    self.register_member_query_sources(
                        output_name,
                        {
                            f"[{index}]": (
                                f"textureReference.borderColor[{index}]({args[1]})"
                            )
                            for index in range(4)
                        },
                    )
                else:
                    self.clear_lvalue_metadata_source(node.args[0])
                return [
                    f"// HIP texture reference get {labels[name]}: "
                    f"output: {output}, texture: {args[1]}"
                ]
        elif name == "hipTexRefGetFormat":
            if len(args) >= 3:
                format_output = self.format_runtime_raw_output_target(node.args[0])
                channels_output = self.format_runtime_raw_output_target(node.args[1])
                format_output_name = self.get_runtime_pointer_target_name(node.args[0])
                channels_output_name = self.get_runtime_pointer_target_name(
                    node.args[1]
                )
                if format_output_name is not None:
                    self.register_device_query_source(
                        format_output_name,
                        f"textureReference.format({args[2]})",
                    )
                if channels_output_name is not None:
                    self.register_device_query_source(
                        channels_output_name,
                        f"textureReference.channelCount({args[2]})",
                    )
                return [
                    "// HIP texture reference get format: "
                    f"format output: {format_output}, "
                    f"channels output: {channels_output}, texture: {args[2]}"
                ]
        elif name == "hipTexRefGetMipmapLevelClamp":
            if len(args) >= 3:
                min_output = self.format_runtime_raw_output_target(node.args[0])
                max_output = self.format_runtime_raw_output_target(node.args[1])
                min_output_name = self.get_runtime_pointer_target_name(node.args[0])
                max_output_name = self.get_runtime_pointer_target_name(node.args[1])
                if min_output_name is not None:
                    self.register_device_query_source(
                        min_output_name,
                        f"textureReference.mipmapLevelClamp.min({args[2]})",
                    )
                if max_output_name is not None:
                    self.register_device_query_source(
                        max_output_name,
                        f"textureReference.mipmapLevelClamp.max({args[2]})",
                    )
                return [
                    "// HIP texture reference get mipmap level clamp: "
                    f"min output: {min_output}, max output: {max_output}, "
                    f"texture: {args[2]}"
                ]
        elif name == "hipTexRefSetAddress":
            if len(args) >= 4:
                offset = self.format_runtime_raw_output_target(node.args[0])
                return [
                    "// HIP texture reference set address: "
                    f"offset output: {offset}, texture: {args[1]}, "
                    f"pointer: {args[2]}, bytes: {args[3]}"
                ]
        elif name == "hipTexRefSetAddress2D":
            if len(args) >= 4:
                return [
                    "// HIP texture reference set address 2D: "
                    f"texture: {args[0]}, desc: {args[1]}, pointer: {args[2]}, "
                    f"pitch: {args[3]}"
                ]
        elif name in {
            "hipTexRefSetAddressMode",
            "hipTexRefSetArray",
            "hipTexRefSetBorderColor",
            "hipTexRefSetFilterMode",
            "hipTexRefSetFlags",
            "hipTexRefSetFormat",
            "hipTexRefSetMaxAnisotropy",
            "hipTexRefSetMipmapFilterMode",
            "hipTexRefSetMipmapLevelBias",
            "hipTexRefSetMipmapLevelClamp",
            "hipTexRefSetMipmappedArray",
        }:
            texture = args[0] if args else "<missing>"
            if name == "hipTexRefSetAddressMode" and len(args) >= 3:
                return [
                    "// HIP texture reference set address mode: "
                    f"texture: {texture}, dim: {args[1]}, mode: {args[2]}"
                ]
            if name == "hipTexRefSetArray" and len(args) >= 3:
                return [
                    "// HIP texture reference set array: "
                    f"texture: {texture}, array: {args[1]}, flags: {args[2]}"
                ]
            if name == "hipTexRefSetBorderColor" and len(args) >= 2:
                return [
                    "// HIP texture reference set border color: "
                    f"texture: {texture}, color: {args[1]}"
                ]
            if name == "hipTexRefSetFilterMode" and len(args) >= 2:
                return [
                    "// HIP texture reference set filter mode: "
                    f"texture: {texture}, mode: {args[1]}"
                ]
            if name == "hipTexRefSetFlags" and len(args) >= 2:
                return [
                    "// HIP texture reference set flags: "
                    f"texture: {texture}, flags: {args[1]}"
                ]
            if name == "hipTexRefSetFormat" and len(args) >= 3:
                return [
                    "// HIP texture reference set format: "
                    f"texture: {texture}, format: {args[1]}, "
                    f"components: {args[2]}"
                ]
            if name == "hipTexRefSetMaxAnisotropy" and len(args) >= 2:
                return [
                    "// HIP texture reference set max anisotropy: "
                    f"texture: {texture}, value: {args[1]}"
                ]
            if name == "hipTexRefSetMipmapFilterMode" and len(args) >= 2:
                return [
                    "// HIP texture reference set mipmap filter mode: "
                    f"texture: {texture}, mode: {args[1]}"
                ]
            if name == "hipTexRefSetMipmapLevelBias" and len(args) >= 2:
                return [
                    "// HIP texture reference set mipmap level bias: "
                    f"texture: {texture}, bias: {args[1]}"
                ]
            if name == "hipTexRefSetMipmapLevelClamp" and len(args) >= 3:
                return [
                    "// HIP texture reference set mipmap level clamp: "
                    f"texture: {texture}, min: {args[1]}, max: {args[2]}"
                ]
            if name == "hipTexRefSetMipmappedArray" and len(args) >= 3:
                return [
                    "// HIP texture reference set mipmapped array: "
                    f"texture: {texture}, mipmapped array: {args[1]}, "
                    f"flags: {args[2]}"
                ]
        elif name == "hipGetDriverEntryPoint":
            if len(args) >= 4:
                output = self.format_runtime_raw_output_target(node.args[1])
                status_output = self.format_runtime_raw_output_target(node.args[3])
                return [
                    f"// HIP get driver entry point: symbol: {args[0]}, "
                    f"output: {output}, flags: {args[2]}, "
                    f"status output: {status_output}"
                ]
        elif name in {"hipLibraryLoadData", "hipLibraryLoadFromFile"}:
            if len(args) >= 8:
                output = self.format_runtime_raw_output_target(node.args[0])
                source_label = "file" if name == "hipLibraryLoadFromFile" else "code"
                return [
                    f"// HIP library load: output: {output}, "
                    f"{source_label}: {args[1]}, jit options: {args[2]}, "
                    f"jit option values: {args[3]}, jit option count: {args[4]}, "
                    f"library options: {args[5]}, "
                    f"library option values: {args[6]}, "
                    f"library option count: {args[7]}"
                ]
        elif name == "hipLibraryUnload":
            if args:
                return [f"// HIP library unload: {args[0]}"]
        elif name == "hipLibraryGetKernel":
            if len(args) >= 3:
                output = self.format_runtime_pointer_target(node.args[0])
                self.clear_lvalue_metadata_source(node.args[0])
                return [
                    f"// HIP library get kernel: output: {output}, "
                    f"library: {args[1]}, name: {args[2]}"
                ]
        elif name == "hipLibraryGetKernelCount":
            if len(args) >= 2:
                output = self.format_runtime_pointer_target(node.args[0])
                output_name = self.get_runtime_pointer_target_name(node.args[0])
                if output_name is not None:
                    self.register_device_query_source(
                        output_name,
                        f"library.kernelCount({args[1]})",
                    )
                else:
                    self.clear_lvalue_metadata_source(node.args[0])
                return [
                    f"// HIP library get kernel count: output: {output}, "
                    f"library: {args[1]}"
                ]
        elif name == "hipLibraryEnumerateKernels":
            if len(args) >= 3:
                self.clear_lvalue_metadata_source(node.args[0])
                output = self.format_runtime_pointer_target(node.args[0])
                return [
                    f"// HIP library enumerate kernels: output: {output}, "
                    f"max kernels: {args[1]}, library: {args[2]}"
                ]
        elif name == "hipKernelGetLibrary":
            if len(args) >= 2:
                output = self.format_runtime_pointer_target(node.args[0])
                self.clear_lvalue_metadata_source(node.args[0])
                return [
                    f"// HIP kernel get library: output: {output}, "
                    f"kernel: {args[1]}"
                ]
        elif name == "hipKernelGetName":
            if len(args) >= 2:
                output = self.format_runtime_pointer_target(node.args[0])
                self.clear_lvalue_metadata_source(node.args[0])
                return [f"// HIP kernel get name: output: {output}, kernel: {args[1]}"]
        elif name == "hipKernelGetParamInfo":
            if len(args) >= 4:
                offset_output = self.format_runtime_pointer_target(node.args[2])
                size_output = self.format_runtime_pointer_target(node.args[3])
                offset_output_name = self.get_runtime_pointer_target_name(node.args[2])
                size_output_name = self.get_runtime_pointer_target_name(node.args[3])
                if offset_output_name is not None:
                    self.register_device_query_source(
                        offset_output_name,
                        f"kernel.param.offset({args[0]}, {args[1]})",
                    )
                else:
                    self.clear_lvalue_metadata_source(node.args[2])
                if size_output_name is not None:
                    self.register_device_query_source(
                        size_output_name,
                        f"kernel.param.size({args[0]}, {args[1]})",
                    )
                else:
                    self.clear_lvalue_metadata_source(node.args[3])
                return [
                    f"// HIP kernel get parameter info: kernel: {args[0]}, "
                    f"param index: {args[1]}, offset output: {offset_output}, "
                    f"size output: {size_output}"
                ]
        elif name == "hipKernelGetFunction":
            if len(args) >= 2:
                output = self.format_runtime_pointer_target(node.args[0])
                self.clear_lvalue_metadata_source(node.args[0])
                return [
                    f"// HIP kernel get function: output: {output}, "
                    f"kernel: {args[1]}"
                ]
        elif name == "hipKernelGetAttribute":
            if len(args) >= 4:
                output = self.format_runtime_pointer_target(node.args[0])
                output_name = self.get_runtime_pointer_target_name(node.args[0])
                if output_name is not None:
                    self.register_device_query_source(
                        output_name,
                        f"kernel.attribute({args[1]}, {args[2]}, {args[3]})",
                    )
                else:
                    self.clear_lvalue_metadata_source(node.args[0])
                return [
                    f"// HIP kernel get attribute: output: {output}, "
                    f"attribute: {args[1]}, kernel: {args[2]}, device: {args[3]}"
                ]
        elif name == "hipKernelSetAttribute":
            if len(args) >= 4:
                return [
                    f"// HIP kernel set attribute: attribute: {args[0]}, "
                    f"value: {args[1]}, kernel: {args[2]}, device: {args[3]}"
                ]
        elif name == "hipLinkCreate":
            if len(args) >= 4:
                output = self.format_runtime_raw_output_target(node.args[3])
                return [
                    f"// HIP link create: options: {args[0]}, "
                    f"option keys: {args[1]}, option values: {args[2]}, "
                    f"state output: {output}"
                ]
        elif name == "hipLinkAddFile":
            if len(args) >= 6:
                return [
                    f"// HIP link add file: state: {args[0]}, input type: {args[1]}, "
                    f"path: {args[2]}, options: {args[3]}, option keys: {args[4]}, "
                    f"option values: {args[5]}"
                ]
        elif name == "hipLinkAddData":
            if len(args) >= 8:
                return [
                    f"// HIP link add data: state: {args[0]}, input type: {args[1]}, "
                    f"image: {args[2]}, bytes: {args[3]}, name: {args[4]}, "
                    f"options: {args[5]}, option keys: {args[6]}, "
                    f"option values: {args[7]}"
                ]
        elif name == "hipLinkComplete":
            if len(args) >= 3:
                self.clear_lvalue_metadata_source(node.args[1])
                binary_output = self.format_runtime_pointer_target(node.args[1])
                self.clear_lvalue_metadata_source(node.args[2])
                size_output = self.format_runtime_pointer_target(node.args[2])
                size_output_name = self.get_runtime_pointer_target_name(node.args[2])
                if size_output_name is not None:
                    self.register_device_query_source(
                        size_output_name,
                        f"link.complete.size({args[0]})",
                    )
                return [
                    f"// HIP link complete: state: {args[0]}, "
                    f"binary output: {binary_output}, size output: {size_output}"
                ]
        elif name == "hipLinkDestroy":
            if args:
                return [f"// HIP link destroy: state: {args[0]}"]
        elif name == "hipMemGetHandleForAddressRange":
            if len(args) >= 5:
                output = self.format_runtime_raw_output_target(node.args[0])
                return [
                    f"// HIP memory get handle for address range: output: {output}, "
                    f"device pointer: {args[1]}, bytes: {args[2]}, "
                    f"handle type: {args[3]}, flags: {args[4]}"
                ]
        elif name == "hipModuleLaunchKernel":
            if len(args) >= 11:
                return [
                    f"// HIP module launch kernel: function: {args[0]}, "
                    f"grid: ({args[1]}, {args[2]}, {args[3]}), "
                    f"block: ({args[4]}, {args[5]}, {args[6]}), "
                    f"shared memory: {args[7]}, stream: {args[8]}, "
                    f"params: {args[9]}, extra: {args[10]}"
                ]
        elif name in {"hipExtModuleLaunchKernel", "hipHccModuleLaunchKernel"}:
            if len(args) >= 13:
                label = "extended" if name == "hipExtModuleLaunchKernel" else "HCC"
                comment = (
                    f"// HIP {label} module launch kernel: function: {args[0]}, "
                    f"global work size: ({args[1]}, {args[2]}, {args[3]}), "
                    f"local work size: ({args[4]}, {args[5]}, {args[6]}), "
                    f"shared memory: {args[7]}, stream: {args[8]}, "
                    f"params: {args[9]}, extra: {args[10]}, "
                    f"start event: {args[11]}, stop event: {args[12]}"
                )
                if len(args) >= 14:
                    comment += f", flags: {args[13]}"
                return [comment]
        elif name == "hipLaunchCooperativeKernel":
            if len(args) >= 6:
                return [
                    f"// HIP cooperative kernel launch: function: {args[0]}, "
                    f"grid: {args[1]}, block: {args[2]}, params: {args[3]}, "
                    f"shared memory: {args[4]}, stream: {args[5]}"
                ]
        elif name == "hipLaunchCooperativeKernelMultiDevice":
            if len(args) >= 3:
                return [
                    f"// HIP cooperative multi-device launch: params: {args[0]}, "
                    f"devices: {args[1]}, flags: {args[2]}"
                ]
        elif name == "hipModuleLaunchCooperativeKernel":
            if len(args) >= 10:
                return [
                    f"// HIP module cooperative kernel launch: function: {args[0]}, "
                    f"grid: ({args[1]}, {args[2]}, {args[3]}), "
                    f"block: ({args[4]}, {args[5]}, {args[6]}), "
                    f"shared memory: {args[7]}, stream: {args[8]}, "
                    f"params: {args[9]}"
                ]
        elif name == "hipModuleLaunchCooperativeKernelMultiDevice":
            if len(args) >= 3:
                return [
                    f"// HIP module cooperative multi-device launch: "
                    f"params: {args[0]}, devices: {args[1]}, flags: {args[2]}"
                ]

        rtc_comments = self.format_hip_rtc_runtime_call(name, node.args, args)
        if rtc_comments is not None:
            return rtc_comments

        interop_comments = self.format_hip_interop_runtime_call(name, node.args, args)
        if interop_comments is not None:
            return interop_comments

        driver_comments = self.format_hip_driver_runtime_call(name, node.args, args)
        if driver_comments is not None:
            return driver_comments

        graph_comments = self.format_hip_graph_runtime_call(name, node.args, args)
        if graph_comments is not None:
            return graph_comments

        if name in {"hipCreateTextureObject", "hipTexObjectCreate"}:
            if len(args) >= 4:
                output = self.format_runtime_pointer_target(node.args[0])
                return [
                    f"// HIP texture object create: {output}, resource: {args[1]}, "
                    f"texture desc: {args[2]}, resource view: {args[3]}"
                ]
        elif name in {"hipDestroyTextureObject", "hipTexObjectDestroy"}:
            if args:
                return [f"// HIP texture object destroy: {args[0]}"]
        elif name in {
            "hipGetTextureObjectResourceDesc",
            "hipTexObjectGetResourceDesc",
        }:
            if len(args) >= 2:
                output = self.format_runtime_raw_output_target(node.args[0])
                output_name = self.get_runtime_pointer_target_name(node.args[0])
                if output_name is not None:
                    self.register_member_query_sources(
                        output_name,
                        {
                            member: f"textureObject.resourceDesc.{member}({args[1]})"
                            for member in (
                                self.HIP_RESOURCE_DESCRIPTOR_MEMBERS
                                | self.HIP_RESOURCE_DESCRIPTOR_NESTED_MEMBERS
                            )
                        },
                    )
                return [
                    f"// HIP texture object get resource desc: output: {output}, "
                    f"texture: {args[1]}"
                ]
        elif name in {"hipGetTextureObjectTextureDesc", "hipTexObjectGetTextureDesc"}:
            if len(args) >= 2:
                output = self.format_runtime_raw_output_target(node.args[0])
                output_name = self.get_runtime_pointer_target_name(node.args[0])
                if output_name is not None:
                    self.register_member_query_sources(
                        output_name,
                        {
                            member: f"textureObject.textureDesc.{member}({args[1]})"
                            for member in (
                                self.HIP_TEXTURE_DESCRIPTOR_MEMBERS
                                | self.HIP_TEXTURE_DESCRIPTOR_INDEXED_MEMBERS
                            )
                        },
                    )
                return [
                    f"// HIP texture object get texture desc: output: {output}, "
                    f"texture: {args[1]}"
                ]
        elif name in {
            "hipGetTextureObjectResourceViewDesc",
            "hipTexObjectGetResourceViewDesc",
        }:
            if len(args) >= 2:
                output = self.format_runtime_raw_output_target(node.args[0])
                output_name = self.get_runtime_pointer_target_name(node.args[0])
                if output_name is not None:
                    self.register_member_query_sources(
                        output_name,
                        {
                            member: (
                                f"textureObject.resourceViewDesc.{member}({args[1]})"
                            )
                            for member in self.HIP_RESOURCE_VIEW_DESCRIPTOR_MEMBERS
                        },
                    )
                return [
                    f"// HIP texture object get resource view desc: output: {output}, "
                    f"texture: {args[1]}"
                ]
        elif name == "hipGetChannelDesc":
            if len(args) >= 2:
                output = self.format_runtime_raw_output_target(node.args[0])
                output_name = self.get_runtime_pointer_target_name(node.args[0])
                if output_name is not None:
                    self.register_member_query_sources(
                        output_name,
                        {
                            member: f"array.channelDesc.{member}({args[1]})"
                            for member in self.HIP_CHANNEL_DESCRIPTOR_MEMBERS
                        },
                    )
                return [f"// HIP get channel desc: output: {output}, array: {args[1]}"]
        elif name == "hipCreateSurfaceObject":
            if len(args) >= 2:
                output = self.format_runtime_pointer_target(node.args[0])
                return [f"// HIP surface object create: {output}, resource: {args[1]}"]
        elif name == "hipGetSurfaceObjectResourceDesc":
            if len(args) >= 2:
                output = self.format_runtime_pointer_target(node.args[0])
                return [
                    "// HIP surface object resource descriptor query not supported "
                    f"by HIP runtime: surface: {args[1]}, output: {output}"
                ]
        elif name == "hipDestroySurfaceObject":
            if args:
                return [f"// HIP surface object destroy: {args[0]}"]

        return None

    def format_hip_graph_runtime_call(self, name, raw_args, args):
        if name == "hipStreamBeginCaptureToGraph":
            if len(args) >= 6:
                return [
                    f"// HIP stream begin capture to graph: stream: {args[0]}, "
                    f"graph: {args[1]}, dependencies: {args[2]}, "
                    f"dependency data: {args[3]}, count: {args[4]}, mode: {args[5]}"
                ]
        if name == "hipThreadExchangeStreamCaptureMode":
            if args:
                output = self.format_runtime_raw_output_target(raw_args[0])
                return [f"// HIP exchange stream capture mode: output: {output}"]
        if name == "hipGraphCreate":
            if len(args) >= 2:
                output = self.format_runtime_raw_output_target(raw_args[0])
                return [f"// HIP graph create: output: {output}, flags: {args[1]}"]
        if name == "hipGraphDestroy":
            if args:
                return [f"// HIP graph destroy: {args[0]}"]
        if name == "hipGraphClone":
            if len(args) >= 2:
                output = self.format_runtime_raw_output_target(raw_args[0])
                return [f"// HIP graph clone: output: {output}, source: {args[1]}"]
        if name == "hipGraphAddNode":
            if len(args) >= 5:
                output = self.format_runtime_raw_output_target(raw_args[0])
                return [
                    f"// HIP graph add generic node: output: {output}, "
                    f"graph: {args[1]}, dependencies: {args[2]}, "
                    f"count: {args[3]}, params: {args[4]}"
                ]
        if name == "hipDrvGraphAddMemcpyNode":
            if len(args) >= 6:
                output = self.format_runtime_raw_output_target(raw_args[0])
                return [
                    f"// HIP driver graph add memcpy node: output: {output}, "
                    f"graph: {args[1]}, dependencies: {args[2]}, "
                    f"count: {args[3]}, params: {args[4]}, context: {args[5]}"
                ]
        if name == "hipDrvGraphAddMemsetNode":
            if len(args) >= 6:
                output = self.format_runtime_raw_output_target(raw_args[0])
                return [
                    f"// HIP driver graph add memset node: output: {output}, "
                    f"graph: {args[1]}, dependencies: {args[2]}, "
                    f"count: {args[3]}, params: {args[4]}, context: {args[5]}"
                ]
        if name == "hipDrvGraphAddMemFreeNode":
            if len(args) >= 5:
                output = self.format_runtime_raw_output_target(raw_args[0])
                return [
                    f"// HIP driver graph add memory free node: output: {output}, "
                    f"graph: {args[1]}, dependencies: {args[2]}, "
                    f"count: {args[3]}, pointer: {args[4]}"
                ]
        if name in {"hipDrvGraphMemcpyNodeGetParams", "hipDrvGraphMemcpyNodeSetParams"}:
            if len(args) >= 2:
                action = "get" if "GetParams" in name else "set"
                params = (
                    self.format_runtime_raw_output_target(raw_args[1])
                    if action == "get"
                    else args[1]
                )
                label = "params output" if action == "get" else "params"
                return [
                    f"// HIP driver graph memcpy node {action} params: "
                    f"node: {args[0]}, {label}: {params}"
                ]
        if name in {
            "hipDrvGraphExecMemcpyNodeSetParams",
            "hipDrvGraphExecMemsetNodeSetParams",
        }:
            if len(args) >= 4:
                node_kind = "memcpy" if "Memcpy" in name else "memset"
                return [
                    f"// HIP driver graph exec {node_kind} node set params: "
                    f"exec: {args[0]}, node: {args[1]}, params: {args[2]}, "
                    f"context: {args[3]}"
                ]
        if name in {
            "hipGraphAddEmptyNode",
            "hipGraphAddHostNode",
            "hipGraphAddKernelNode",
            "hipGraphAddMemcpyNode",
            "hipGraphAddMemsetNode",
        }:
            if len(args) >= 4:
                output = self.format_runtime_raw_output_target(raw_args[0])
                node_kind = {
                    "hipGraphAddEmptyNode": "empty",
                    "hipGraphAddHostNode": "host",
                    "hipGraphAddKernelNode": "kernel",
                    "hipGraphAddMemcpyNode": "memcpy",
                    "hipGraphAddMemsetNode": "memset",
                }[name]
                comment = (
                    f"// HIP graph add {node_kind} node: output: {output}, "
                    f"graph: {args[1]}, dependencies: {args[2]}, count: {args[3]}"
                )
                if len(args) >= 5:
                    comment += f", params: {args[4]}"
                return [comment]
        if name == "hipGraphAddChildGraphNode":
            if len(args) >= 5:
                output = self.format_runtime_raw_output_target(raw_args[0])
                return [
                    f"// HIP graph add child graph node: output: {output}, "
                    f"graph: {args[1]}, dependencies: {args[2]}, "
                    f"count: {args[3]}, child graph: {args[4]}"
                ]
        if name in {"hipGraphAddEventRecordNode", "hipGraphAddEventWaitNode"}:
            if len(args) >= 5:
                output = self.format_runtime_raw_output_target(raw_args[0])
                action = "record" if name == "hipGraphAddEventRecordNode" else "wait"
                return [
                    f"// HIP graph add event {action} node: output: {output}, "
                    f"graph: {args[1]}, dependencies: {args[2]}, "
                    f"count: {args[3]}, event: {args[4]}"
                ]
        if name in {
            "hipGraphAddMemAllocNode",
            "hipGraphAddMemFreeNode",
        }:
            if len(args) >= 5:
                output = self.format_runtime_raw_output_target(raw_args[0])
                node_kind = "alloc" if name == "hipGraphAddMemAllocNode" else "free"
                detail_label = "params" if node_kind == "alloc" else "pointer"
                return [
                    f"// HIP graph add memory {node_kind} node: output: {output}, "
                    f"graph: {args[1]}, dependencies: {args[2]}, "
                    f"count: {args[3]}, {detail_label}: {args[4]}"
                ]
        if name in {"hipGraphMemAllocNodeGetParams", "hipGraphMemFreeNodeGetParams"}:
            if len(args) >= 2:
                output = self.format_runtime_raw_output_target(raw_args[1])
                node_kind = "alloc" if "Alloc" in name else "free"
                detail_label = (
                    "params output" if node_kind == "alloc" else "pointer output"
                )
                return [
                    f"// HIP graph memory {node_kind} node get params: "
                    f"node: {args[0]}, {detail_label}: {output}"
                ]
        if name in {
            "hipGraphAddMemcpyNode1D",
            "hipGraphAddMemcpyNodeFromSymbol",
            "hipGraphAddMemcpyNodeToSymbol",
        }:
            if name == "hipGraphAddMemcpyNode1D" and len(args) >= 8:
                output = self.format_runtime_raw_output_target(raw_args[0])
                return [
                    f"// HIP graph add memcpy 1D node: output: {output}, "
                    f"graph: {args[1]}, dependencies: {args[2]}, count: {args[3]}, "
                    f"destination: {args[4]}, source: {args[5]}, bytes: {args[6]}, "
                    f"kind: {args[7]}"
                ]
            if len(args) >= 9:
                output = self.format_runtime_raw_output_target(raw_args[0])
                copy_kind = {
                    "hipGraphAddMemcpyNodeFromSymbol": "from symbol",
                    "hipGraphAddMemcpyNodeToSymbol": "to symbol",
                }[name]
                return [
                    f"// HIP graph add memcpy {copy_kind} node: output: {output}, "
                    f"graph: {args[1]}, dependencies: {args[2]}, count: {args[3]}, "
                    f"destination: {args[4]}, source: {args[5]}, bytes: {args[6]}, "
                    f"offset: {args[7]}, kind: {args[8]}"
                ]
        if name in {
            "hipGraphMemcpyNodeSetParams1D",
            "hipGraphMemcpyNodeSetParamsFromSymbol",
            "hipGraphMemcpyNodeSetParamsToSymbol",
        }:
            if name == "hipGraphMemcpyNodeSetParams1D" and len(args) >= 5:
                return [
                    f"// HIP graph memcpy 1D node set params: "
                    f"node: {args[0]}, destination: {args[1]}, source: {args[2]}, "
                    f"bytes: {args[3]}, kind: {args[4]}"
                ]
            if len(args) >= 6:
                copy_kind = {
                    "hipGraphMemcpyNodeSetParamsFromSymbol": "from symbol",
                    "hipGraphMemcpyNodeSetParamsToSymbol": "to symbol",
                }[name]
                return [
                    f"// HIP graph memcpy {copy_kind} node set params: "
                    f"node: {args[0]}, destination: {args[1]}, source: {args[2]}, "
                    f"bytes: {args[3]}, offset: {args[4]}, kind: {args[5]}"
                ]
        if name in {
            "hipGraphExecMemcpyNodeSetParams1D",
            "hipGraphExecMemcpyNodeSetParamsFromSymbol",
            "hipGraphExecMemcpyNodeSetParamsToSymbol",
        }:
            if name == "hipGraphExecMemcpyNodeSetParams1D" and len(args) >= 6:
                return [
                    f"// HIP graph exec memcpy 1D node set params: "
                    f"exec: {args[0]}, node: {args[1]}, destination: {args[2]}, "
                    f"source: {args[3]}, bytes: {args[4]}, kind: {args[5]}"
                ]
            if len(args) >= 7:
                copy_kind = {
                    "hipGraphExecMemcpyNodeSetParamsFromSymbol": "from symbol",
                    "hipGraphExecMemcpyNodeSetParamsToSymbol": "to symbol",
                }[name]
                return [
                    f"// HIP graph exec memcpy {copy_kind} node set params: "
                    f"exec: {args[0]}, node: {args[1]}, destination: {args[2]}, "
                    f"source: {args[3]}, bytes: {args[4]}, offset: {args[5]}, "
                    f"kind: {args[6]}"
                ]
        if name in {"hipGraphAddDependencies", "hipGraphRemoveDependencies"}:
            if len(args) >= 4:
                action = "add" if name == "hipGraphAddDependencies" else "remove"
                return [
                    f"// HIP graph {action} dependencies: graph: {args[0]}, "
                    f"from: {args[1]}, to: {args[2]}, count: {args[3]}"
                ]
        if name in {"hipGraphGetNodes", "hipGraphGetRootNodes"}:
            if len(args) >= 3:
                output = self.format_runtime_raw_output_target(raw_args[2])
                node_set = "nodes" if name == "hipGraphGetNodes" else "root nodes"
                return [
                    f"// HIP graph get {node_set}: graph: {args[0]}, "
                    f"nodes output: {args[1]}, count output: {output}"
                ]
        if name == "hipGraphGetEdges":
            if len(args) >= 4:
                output = self.format_runtime_raw_output_target(raw_args[3])
                return [
                    f"// HIP graph get edges: graph: {args[0]}, "
                    f"from output: {args[1]}, to output: {args[2]}, "
                    f"count output: {output}"
                ]
        if name in {"hipGraphNodeGetDependencies", "hipGraphNodeGetDependentNodes"}:
            if len(args) >= 3:
                output = self.format_runtime_raw_output_target(raw_args[2])
                node_set = (
                    "dependencies"
                    if name == "hipGraphNodeGetDependencies"
                    else "dependent nodes"
                )
                return [
                    f"// HIP graph node get {node_set}: node: {args[0]}, "
                    f"nodes output: {args[1]}, count output: {output}"
                ]
        if name == "hipGraphNodeFindInClone":
            if len(args) >= 3:
                output = self.format_runtime_raw_output_target(raw_args[0])
                return [
                    f"// HIP graph node find in clone: output: {output}, "
                    f"original: {args[1]}, clone graph: {args[2]}"
                ]
        if name == "hipGraphNodeGetType":
            if len(args) >= 2:
                output = self.format_runtime_raw_output_target(raw_args[1])
                return [
                    f"// HIP graph node get type: node: {args[0]}, output: {output}"
                ]
        if name == "hipGraphChildGraphNodeGetGraph":
            if len(args) >= 2:
                output = self.format_runtime_raw_output_target(raw_args[1])
                return [
                    f"// HIP graph child node get graph: "
                    f"node: {args[0]}, output: {output}"
                ]
        if name == "hipGraphKernelNodeCopyAttributes":
            if len(args) >= 2:
                return [
                    f"// HIP graph kernel node copy attributes: "
                    f"source: {args[0]}, destination: {args[1]}"
                ]
        if name in {"hipGraphKernelNodeGetAttribute", "hipGraphKernelNodeSetAttribute"}:
            if len(args) >= 3:
                action = "get" if name == "hipGraphKernelNodeGetAttribute" else "set"
                value = (
                    self.format_runtime_raw_output_target(raw_args[2])
                    if action == "get"
                    else self.format_runtime_pointer_target(raw_args[2])
                )
                label = "output" if action == "get" else "value"
                return [
                    f"// HIP graph kernel node {action} attribute: "
                    f"node: {args[0]}, attribute: {args[1]}, {label}: {value}"
                ]
        if name in {"hipGraphNodeSetEnabled", "hipGraphNodeGetEnabled"}:
            if len(args) >= 3:
                action = "get" if name == "hipGraphNodeGetEnabled" else "set"
                value = (
                    self.format_runtime_raw_output_target(raw_args[2])
                    if action == "get"
                    else args[2]
                )
                label = "output" if action == "get" else "value"
                return [
                    f"// HIP graph node {action} enabled: "
                    f"exec: {args[0]}, node: {args[1]}, {label}: {value}"
                ]
        if name == "hipGraphDestroyNode":
            if args:
                return [f"// HIP graph destroy node: {args[0]}"]
        if name == "hipGraphInstantiate":
            if len(args) >= 5:
                output = self.format_runtime_raw_output_target(raw_args[0])
                error_output = self.format_runtime_raw_output_target(raw_args[2])
                return [
                    f"// HIP graph instantiate: output: {output}, graph: {args[1]}, "
                    f"error node output: {error_output}, log buffer: {args[3]}, "
                    f"log bytes: {args[4]}"
                ]
        if name == "hipGraphInstantiateWithFlags":
            if len(args) >= 3:
                output = self.format_runtime_raw_output_target(raw_args[0])
                return [
                    f"// HIP graph instantiate with flags: output: {output}, "
                    f"graph: {args[1]}, flags: {args[2]}"
                ]
        if name == "hipGraphInstantiateWithParams":
            if len(args) >= 3:
                output = self.format_runtime_raw_output_target(raw_args[0])
                return [
                    f"// HIP graph instantiate with params: output: {output}, "
                    f"graph: {args[1]}, params: {args[2]}"
                ]
        if name == "hipGraphUpload":
            if len(args) >= 2:
                return [f"// HIP graph upload: exec: {args[0]}, stream: {args[1]}"]
        if name == "hipGraphLaunch":
            if len(args) >= 2:
                return [f"// HIP graph launch: exec: {args[0]}, stream: {args[1]}"]
        if name == "hipGraphExecDestroy":
            if args:
                return [f"// HIP graph exec destroy: {args[0]}"]
        if name == "hipGraphExecUpdate":
            if len(args) >= 4:
                error_output = self.format_runtime_raw_output_target(raw_args[2])
                result_output = self.format_runtime_raw_output_target(raw_args[3])
                return [
                    f"// HIP graph exec update: exec: {args[0]}, graph: {args[1]}, "
                    f"error node output: {error_output}, result output: {result_output}"
                ]
        if name == "hipGraphExecGetFlags":
            if len(args) >= 2:
                output = self.format_runtime_raw_output_target(raw_args[1])
                return [
                    f"// HIP graph exec get flags: exec: {args[0]}, output: {output}"
                ]
        if name in {"hipGraphNodeSetParams", "hipGraphExecNodeSetParams"}:
            if len(args) >= 2:
                if name == "hipGraphNodeSetParams":
                    return [
                        f"// HIP graph generic node set params: "
                        f"node: {args[0]}, params: {args[1]}"
                    ]
                if len(args) >= 3:
                    return [
                        f"// HIP graph exec generic node set params: "
                        f"exec: {args[0]}, node: {args[1]}, params: {args[2]}"
                    ]
        if name in {
            "hipGraphKernelNodeGetParams",
            "hipGraphKernelNodeSetParams",
            "hipGraphMemcpyNodeGetParams",
            "hipGraphMemcpyNodeSetParams",
            "hipGraphMemsetNodeGetParams",
            "hipGraphMemsetNodeSetParams",
            "hipGraphHostNodeGetParams",
            "hipGraphHostNodeSetParams",
        }:
            if len(args) >= 2:
                if "GetParams" in name:
                    self.clear_lvalue_metadata_source(raw_args[1])
                return [self.format_hip_graph_node_params_comment(name, args)]
        if name in {
            "hipGraphExecKernelNodeSetParams",
            "hipGraphExecMemcpyNodeSetParams",
            "hipGraphExecMemsetNodeSetParams",
            "hipGraphExecHostNodeSetParams",
            "hipGraphExecChildGraphNodeSetParams",
        }:
            if len(args) >= 3:
                node_kind = self.get_hip_graph_node_kind(name)
                return [
                    f"// HIP graph exec set {node_kind} node params: exec: {args[0]}, "
                    f"node: {args[1]}, params: {args[2]}"
                ]
        if name in {
            "hipGraphEventRecordNodeGetEvent",
            "hipGraphEventWaitNodeGetEvent",
        }:
            if len(args) >= 2:
                output = self.format_runtime_raw_output_target(raw_args[1])
                action = (
                    "record" if name == "hipGraphEventRecordNodeGetEvent" else "wait"
                )
                return [
                    f"// HIP graph event {action} node get event: "
                    f"node: {args[0]}, output: {output}"
                ]
        if name in {
            "hipGraphEventRecordNodeSetEvent",
            "hipGraphEventWaitNodeSetEvent",
        }:
            if len(args) >= 2:
                action = (
                    "record" if name == "hipGraphEventRecordNodeSetEvent" else "wait"
                )
                return [
                    f"// HIP graph event {action} node set event: "
                    f"node: {args[0]}, event: {args[1]}"
                ]
        if name in {
            "hipGraphExecEventRecordNodeSetEvent",
            "hipGraphExecEventWaitNodeSetEvent",
        }:
            if len(args) >= 3:
                action = (
                    "record"
                    if name == "hipGraphExecEventRecordNodeSetEvent"
                    else "wait"
                )
                return [
                    f"// HIP graph exec event {action} node set event: "
                    f"exec: {args[0]}, node: {args[1]}, event: {args[2]}"
                ]
        if name in {
            "hipGraphAddExternalSemaphoresSignalNode",
            "hipGraphAddExternalSemaphoresWaitNode",
        }:
            if len(args) >= 5:
                output = self.format_runtime_raw_output_target(raw_args[0])
                action = (
                    "signal"
                    if name == "hipGraphAddExternalSemaphoresSignalNode"
                    else "wait"
                )
                return [
                    f"// HIP graph add external semaphore {action} node: "
                    f"output: {output}, graph: {args[1]}, dependencies: {args[2]}, "
                    f"count: {args[3]}, params: {args[4]}"
                ]
        if name in {
            "hipGraphExternalSemaphoresSignalNodeGetParams",
            "hipGraphExternalSemaphoresSignalNodeSetParams",
            "hipGraphExternalSemaphoresWaitNodeGetParams",
            "hipGraphExternalSemaphoresWaitNodeSetParams",
        }:
            if len(args) >= 2:
                action = "signal" if "Signal" in name else "wait"
                direction = "get" if "GetParams" in name else "set"
                if direction == "get":
                    self.clear_lvalue_metadata_source(raw_args[1])
                return [
                    f"// HIP graph external semaphore {action} node {direction} params: "
                    f"node: {args[0]}, params: {args[1]}"
                ]
        if name in {
            "hipGraphExecExternalSemaphoresSignalNodeSetParams",
            "hipGraphExecExternalSemaphoresWaitNodeSetParams",
        }:
            if len(args) >= 3:
                action = "signal" if "Signal" in name else "wait"
                return [
                    f"// HIP graph exec external semaphore {action} node set params: "
                    f"exec: {args[0]}, node: {args[1]}, params: {args[2]}"
                ]
        if name in {"hipDeviceGetGraphMemAttribute", "hipDeviceSetGraphMemAttribute"}:
            if len(args) >= 3:
                action = "get" if name == "hipDeviceGetGraphMemAttribute" else "set"
                value = (
                    self.format_runtime_raw_output_target(raw_args[2])
                    if action == "get"
                    else self.format_runtime_pointer_target(raw_args[2])
                )
                label = "output" if action == "get" else "value"
                return [
                    f"// HIP device graph memory {action} attribute: "
                    f"device: {args[0]}, attribute: {args[1]}, {label}: {value}"
                ]
        if name == "hipDeviceGraphMemTrim":
            if args:
                return [f"// HIP device graph memory trim: device: {args[0]}"]
        if name == "hipGraphDebugDotPrint":
            if len(args) >= 3:
                return [
                    f"// HIP graph debug dot print: graph: {args[0]}, "
                    f"path: {args[1]}, flags: {args[2]}"
                ]
        if name == "hipUserObjectCreate":
            if len(args) >= 5:
                output = self.format_runtime_raw_output_target(raw_args[0])
                return [
                    f"// HIP user object create: output: {output}, "
                    f"resource: {args[1]}, destructor: {args[2]}, "
                    f"initial refcount: {args[3]}, flags: {args[4]}"
                ]
        if name in {"hipUserObjectRetain", "hipUserObjectRelease"}:
            if len(args) >= 2:
                action = "retain" if name == "hipUserObjectRetain" else "release"
                return [
                    f"// HIP user object {action}: object: {args[0]}, count: {args[1]}"
                ]
        if name in {"hipGraphRetainUserObject", "hipGraphReleaseUserObject"}:
            if len(args) >= 3:
                action = "retain" if name == "hipGraphRetainUserObject" else "release"
                comment = (
                    f"// HIP graph {action} user object: graph: {args[0]}, "
                    f"object: {args[1]}, count: {args[2]}"
                )
                if len(args) >= 4:
                    comment += f", flags: {args[3]}"
                return [comment]
        return None

    def format_hip_runtime_expression_call(self, node, args):
        """A host runtime query's value, with a comment naming the query; in device code it is refused."""
        value = self.describe_hip_runtime_expression_call(node, args)
        if value is not None and self.in_device_code:
            return self.refuse_device_runtime_call(f"runtime call {node.name}()")
        return value

    def describe_hip_runtime_expression_call(self, node, args):
        name = node.name
        if name == "hipGetErrorString" and args:
            return f'/* HIP error string: {args[0]} */ ""'
        if name == "hipGetErrorName" and args:
            return f'/* HIP error name: {args[0]} */ ""'
        if name == "hiprtcGetErrorString" and args:
            return f'/* HIPRTC error string: {args[0]} */ ""'
        if name == "hipApiName" and args:
            return f'/* HIP API name: {args[0]} */ ""'
        if name == "hipKernelNameRef" and args:
            return f'/* HIP kernel name for function: {args[0]} */ ""'
        if name == "hipKernelNameRefByPtr" and len(args) >= 2:
            return (
                f"/* HIP kernel name for host function: {args[0]}, "
                f'stream: {args[1]} */ ""'
            )
        if name == "hipGetStreamDeviceId" and args:
            return f"/* HIP stream device id: {args[0]} */ 0"

        return self.format_hip_runtime_status_inline_expression(node)

    def visit_HipProgramNode(self, node):
        """Render a HIP program AST as a CrossGL shader block."""
        self.emit("// HIP to CrossGL conversion")
        has_kernel = any(
            isinstance(stmt, FunctionNode)
            and "__global__" in getattr(stmt, "qualifiers", [])
            for stmt in node.statements
        )

        for stmt in node.statements:
            if isinstance(stmt, FunctionNode):
                if self.is_generated_matrix_helper_function(stmt):
                    continue
                if hasattr(stmt, "qualifiers") and "__global__" in getattr(
                    stmt, "qualifiers", []
                ):
                    self.emit_function_local_record_definitions(stmt)
                    self.emit(f"// Kernel: {stmt.name}")
                    self.visit_kernel_as_compute_shader(stmt)
                else:
                    if has_kernel and stmt.name == "main":
                        continue
                    self.emit_function_local_record_definitions(stmt)
                    self.emit(f"// Function: {stmt.name}")
                    self.visit(stmt)
                self.emit("")
            elif isinstance(stmt, StructNode):
                if self.is_generated_matrix_helper_struct(stmt):
                    continue
                self.visit(stmt)
                self.emit("")
            elif isinstance(stmt, EnumNode):
                self.visit(stmt)
                self.emit("")
            elif isinstance(stmt, VariableNode):
                self.namespace_scope_variable = True
                try:
                    self.visit(stmt)
                finally:
                    self.namespace_scope_variable = False
                self.emit("")
            elif isinstance(stmt, TypeAliasNode):
                self.visit(stmt)
                self.emit("")
            else:
                self.visit(stmt)

    def format_preprocessor_content(self, content):
        text = str(content).strip()
        compact = text.replace(" ", "")
        if compact.startswith("<") and compact.endswith(">"):
            return compact
        return text

    def visit_FunctionNode(self, node):
        """Render a HIP function node as a CrossGL function, tracking whether its body is device code."""
        previous = self.in_device_code
        self.in_device_code = previous or "__device__" in (
            getattr(node, "qualifiers", None) or []
        )
        try:
            return self.render_function_node(node)
        finally:
            self.in_device_code = previous

    def render_function_node(self, node):
        """Render a HIP function node as a CrossGL function."""
        raw_return_type = node.return_type if hasattr(node, "return_type") else "void"
        self.validate_record_layout_use(
            self.record_definition_from_type(raw_return_type),
            f"return type of function '{getattr(node, 'name', '')}'",
        )
        return_type = self.convert_hip_type_to_crossgl(raw_return_type)

        self.push_resource_object_hint_scope(
            self.collect_resource_object_type_hints(
                node, self.collect_declared_variable_names(node)
            )
        )
        self.push_variable_type_scope()
        self.push_identifier_name_scope()
        try:
            params = []

            if hasattr(node, "params") and node.params:
                for index, param in enumerate(node.params):
                    param_name = (
                        param.get("name", "param")
                        if isinstance(param, dict)
                        else getattr(param, "name", "param")
                    ) or f"_param{index}"
                    raw_type = (
                        param.get("type", "int")
                        if isinstance(param, dict)
                        else getattr(param, "vtype", "int")
                    )
                    param_type = self.convert_hip_variable_type_to_crossgl(
                        raw_type, param_name
                    )
                    parameter_record = self.record_definition_from_type(raw_type)
                    self.validate_record_layout_use(
                        parameter_record, f"parameter '{param_name}'"
                    )
                    self.register_union_object_declaration(
                        param_name, parameter_record, param
                    )
                    self.register_union_indirect_object(
                        param_name, raw_type, parameter_record
                    )
                    output_name = self.register_identifier_name(param_name)
                    self.register_vector1_name(param_name, raw_type)
                    self.register_variable_type(param_name, param_type)
                    if getattr(self, "reference_parameters", None) is None:
                        self.reference_parameters = ReferenceParameters(None)
                    mode = self.reference_parameters.mode(node, index)
                    params.append(
                        f"{mode} {param_type} {output_name}"
                        if mode
                        else f"{param_type} {output_name}"
                    )

            param_str = ", ".join(params)
            function_name = self.format_function_declaration_name(node.name)
            if function_requires_no_contraction(
                node, file_scope_off=self.fp_contract_file_scope_off
            ):
                self.emit(PRECISE_FUNCTION_ATTRIBUTE)
            self.emit(f"{return_type} {function_name}({param_str}) {{")

            self.indent_level += 1
            self.push_packed_argument_scope()
            self.push_type_alias_scope()
            self.push_unique_ptr_scope()
            self.push_cooperative_group_scope()
            if hasattr(node, "params") and node.params:
                for param in node.params:
                    self.register_unique_ptr_parameter(param)
                    self.register_cooperative_group_parameter(param)
            if hasattr(node, "body") and node.body:
                try:
                    self.function_body_depth += 1
                    if isinstance(node.body, list):
                        for stmt in node.body:
                            self.emit_statement(stmt)
                    else:
                        self.emit_statement(node.body)
                finally:
                    self.function_body_depth -= 1
                    self.pop_cooperative_group_scope()
                    self.pop_unique_ptr_scope()
                    self.pop_type_alias_scope()
                    self.pop_packed_argument_scope()
                    self.indent_level -= 1
            else:
                self.pop_cooperative_group_scope()
                self.pop_unique_ptr_scope()
                self.pop_type_alias_scope()
                self.pop_packed_argument_scope()
                self.indent_level -= 1
        finally:
            self.pop_identifier_name_scope()
            self.pop_variable_type_scope()
            self.pop_resource_object_hint_scope()

        self.emit("}")

    def visit_kernel_as_compute_shader(self, kernel):
        """Render a HIP ``__global__`` kernel as a CrossGL compute entry point; its body is device code."""
        previous = self.in_device_code
        self.in_device_code = True
        try:
            return self.render_kernel_as_compute_shader(kernel)
        finally:
            self.in_device_code = previous

    def render_kernel_as_compute_shader(self, kernel):
        """Render a HIP kernel as a CrossGL compute shader block."""
        for attribute in getattr(kernel, "attributes", []) or []:
            attribute_text = str(attribute)
            if attribute_text.startswith("__launch_bounds__"):
                bounds = attribute_text[len("__launch_bounds__") :]
                self.emit(f"// HIP launch bounds: {bounds}")
                continue

            flat_work_group_size = re.search(
                r"amdgpu_flat_work_group_size\((.*?)\)", attribute_text
            )
            if flat_work_group_size:
                bounds = re.sub(r"\s*,\s*", ", ", flat_work_group_size.group(1))
                self.emit(f"// HIP AMDGPU flat work group size: ({bounds})")

        self.emit("@compute")
        if function_requires_no_contraction(
            kernel, file_scope_off=self.fp_contract_file_scope_off
        ):
            self.emit(PRECISE_FUNCTION_ATTRIBUTE)
        self.emit("@stage_entry")
        self.emit(
            kernel_workgroup_size_attribute(self.threadgroup_size, whole_groups=True)
        )

        params = []
        buffer_params = (
            {}
        )  # index in params -> (context, element type) of a buffer parameter
        self.push_resource_object_hint_scope(
            self.collect_resource_object_type_hints(
                kernel, self.collect_declared_variable_names(kernel)
            )
        )
        self.push_variable_type_scope()
        self.push_identifier_name_scope()
        try:
            if hasattr(kernel, "params") and kernel.params:
                for index, param in enumerate(kernel.params):
                    if isinstance(param, dict):
                        raw_type = param.get("type", "int")
                        param_name = param.get("name", "param")
                    else:
                        raw_type = getattr(param, "vtype", "int")
                        param_name = getattr(param, "name", "param")
                    param_name = param_name or f"_param{index}"
                    parameter_record = self.record_definition_from_type(raw_type)
                    self.validate_record_layout_use(
                        parameter_record, f"parameter '{param_name}'"
                    )
                    self.validate_record_kernel_parameter_semantics(
                        raw_type, parameter_record, f"parameter '{param_name}'"
                    )
                    self.register_union_object_declaration(
                        param_name, parameter_record, param
                    )
                    self.register_union_indirect_object(
                        param_name, raw_type, parameter_record
                    )

                    if "*" in raw_type:
                        element_type = self.convert_hip_pointer_element_type(raw_type)
                        output_name = self.register_identifier_name(param_name)
                        self.register_variable_type(
                            param_name, f"array<{element_type}>"
                        )
                        access = (
                            "read"
                            if self.is_readonly_pointer_type(raw_type)
                            else "read_write"
                        )
                        buffer_params[len(params)] = (
                            f"HIP kernel {kernel.name} parameter {param_name}",
                            element_type,
                        )
                        params.append(
                            f"@group(0) @binding({self.global_resource_binding_count + len(params)}) "
                            f"var<storage, {access}> "
                            f"{output_name}: array<{element_type}>"
                        )
                    else:
                        param_type = self.convert_hip_variable_type_to_crossgl(
                            raw_type, param_name
                        )
                        output_name = self.register_identifier_name(param_name)
                        self.register_vector1_name(param_name, raw_type)
                        self.register_variable_type(param_name, param_type)
                        if self.is_crossgl_scalar_type(param_type):
                            params.append(
                                f"@group(0) @binding({self.global_resource_binding_count + len(params)}) "
                                f"var<uniform> {output_name}: {param_type}"
                            )
                        else:
                            params.append(f"{param_type} {output_name}")

            kernel_name = self.format_function_declaration_name(kernel.name)
            self.emit(f"fn {kernel_name}(")
            self.indent_level += 1
            layouts = getattr(self, "emitted_buffer_layouts", None)
            for i, param in enumerate(params):
                if layouts is not None and i in buffer_params:
                    layouts.buffer(*buffer_params[i], len(self.output))
                if i == len(params) - 1:
                    self.emit(f"{param}")
                else:
                    self.emit(f"{param},")
            self.indent_level -= 1
            self.emit(") {")

            self.indent_level += 1
            if hasattr(kernel, "body") and kernel.body:
                self.push_packed_argument_scope()
                self.push_type_alias_scope()
                self.push_unique_ptr_scope()
                self.push_cooperative_group_scope()
                if hasattr(kernel, "params") and kernel.params:
                    for param in kernel.params:
                        self.register_unique_ptr_parameter(param)
                        self.register_cooperative_group_parameter(param)
                try:
                    self.function_body_depth += 1
                    if isinstance(kernel.body, list):
                        for stmt in kernel.body:
                            self.emit_statement(stmt)
                    else:
                        self.emit_statement(kernel.body)
                finally:
                    self.function_body_depth -= 1
                    self.pop_cooperative_group_scope()
                    self.pop_unique_ptr_scope()
                    self.pop_type_alias_scope()
                    self.pop_packed_argument_scope()

            self.indent_level -= 1
            self.emit("}")
        finally:
            self.pop_identifier_name_scope()
            self.pop_variable_type_scope()
            self.pop_resource_object_hint_scope()

    def visit_KernelLaunchNode(self, node):
        kernel_name = (
            node.kernel_name
            if isinstance(node.kernel_name, str)
            else self.visit(node.kernel_name)
        )
        if self.in_device_code:
            self.refuse_device_runtime_call(f"kernel launch {kernel_name}<<<...>>>")
            return
        config = [self.visit(node.blocks), self.visit(node.threads)]
        if node.shared_mem is not None:
            config.append(self.visit(node.shared_mem))
        if node.stream is not None:
            config.append(self.visit(node.stream))

        self.emit(f"// Kernel launch: {kernel_name}<<<{', '.join(config)}>>>()")
        if node.args:
            args = self.resolve_packed_launch_args(node.args)
            args_str = ", ".join([self.format_kernel_launch_arg(arg) for arg in args])
            self.emit(f"// Arguments: {args_str}")

    def format_hip_timer_intrinsic_call(self, function_name, args):
        if isinstance(function_name, str) and function_name.startswith("::"):
            function_name = function_name[2:]

        if function_name not in {"clock", "clock64", "wall_clock64"}:
            return None

        return self.refuse_import(
            f"HIP device timer {function_name}({', '.join(args)}): CrossGL has no clock counter",
            "0",
        )

    def format_hip_load_cache_intrinsic_call(
        self, function_name, raw_args, formatted_args
    ):
        if isinstance(function_name, str) and function_name.startswith("::"):
            function_name = function_name[2:]

        if function_name not in {
            "__ldca",
            "__ldcg",
            "__ldcs",
            "__ldcv",
            "__ldg",
            "__ldlu",
        }:
            return None

        if (
            len(raw_args) == 1
            and isinstance(raw_args[0], UnaryOpNode)
            and raw_args[0].op == "&"
        ):
            return self.visit(raw_args[0].operand)

        return self.refuse_import(
            f"HIP load cache intrinsic {function_name}({', '.join(formatted_args)}): CrossGL loads take one "
            "address",
            "0",
        )

    def format_hip_warp_intrinsic_call(self, function_name, args):
        if isinstance(function_name, str) and function_name.startswith("::"):
            function_name = function_name[2:]

        if function_name in {"__activemask", "__ballot_sync"}:
            return self.format_unsupported_hip_warp_intrinsic(function_name, args)

        if function_name in {"__any", "__all"}:
            if len(args) != 1:
                return self.format_unsupported_hip_warp_intrinsic(function_name, args)
            predicate = self.format_wave_predicate(args[0])
            if function_name == "__any":
                return f"(WaveActiveAnyTrue({predicate}) ? 1 : 0)"
            return f"(WaveActiveAllTrue({predicate}) ? 1 : 0)"

        if function_name in {"__any_sync", "__all_sync"}:
            if len(args) != 2 or not self.is_full_or_active_warp_mask(args[0]):
                return self.format_unsupported_hip_warp_intrinsic(function_name, args)
            predicate = self.format_wave_predicate(args[1])
            if function_name == "__any_sync":
                return f"(WaveActiveAnyTrue({predicate}) ? 1 : 0)"
            return f"(WaveActiveAllTrue({predicate}) ? 1 : 0)"

        if function_name == "__shfl_sync":
            width_limit = self.hip_warp_shuffle_width_limit(args[3:] or None)
            if (
                len(args) not in {3, 4}
                or not self.is_full_or_active_warp_mask(args[0])
                or width_limit is None
            ):
                return self.format_unsupported_hip_warp_intrinsic(function_name, args)
            if width_limit == "WaveGetLaneCount()":
                return f"WaveReadLaneAt({args[1]}, {args[2]})"
            # HIP reads lane srcLane & (width - 1) of the caller's width-lane segment (HIP __shfl).
            return (
                f"WaveReadLaneAt({args[1]}, ((WaveGetLaneIndex() - "
                f"(WaveGetLaneIndex() % {width_limit})) + (({args[2]}) & ({width_limit} - 1))))"
            )

        if function_name in {"__shfl_up_sync", "__shfl_down_sync"}:
            width_limit = self.hip_warp_shuffle_width_limit(args[3:] or None)
            if (
                len(args) not in {3, 4}
                or not self.is_full_or_active_warp_mask(args[0])
                or width_limit is None
            ):
                return self.format_unsupported_hip_warp_intrinsic(function_name, args)
            value, delta = args[1], args[2]
            # Lanes shift within their width-lane segment; a source lane outside it keeps the caller's value.
            segment_lane = (
                "WaveGetLaneIndex()"
                if width_limit == "WaveGetLaneCount()"
                else f"(WaveGetLaneIndex() % {width_limit})"
            )
            if function_name == "__shfl_up_sync":
                return (
                    f"(({segment_lane} >= ({delta})) ? "
                    f"WaveReadLaneAt({value}, (WaveGetLaneIndex() - ({delta}))) "
                    f": {value})"
                )
            return (
                f"((({segment_lane} + ({delta})) < {width_limit}) ? "
                f"WaveReadLaneAt({value}, (WaveGetLaneIndex() + ({delta}))) "
                f": {value})"
            )

        if function_name == "__shfl_xor_sync":
            width_limit = self.hip_warp_shuffle_width_limit(args[3:] or None)
            if (
                len(args) not in {3, 4}
                or not self.is_full_or_active_warp_mask(args[0])
                or width_limit is None
            ):
                return self.format_unsupported_hip_warp_intrinsic(function_name, args)
            if width_limit == "WaveGetLaneCount()":
                return (
                    f"WaveReadLaneAt({args[1]}, " f"(WaveGetLaneIndex() ^ {args[2]}))"
                )
            # HIP __shfl_xor: a partner beyond the caller's width-lane segment reads the caller's own lane.
            partner = f"(WaveGetLaneIndex() ^ {args[2]})"
            segment_end = (
                f"((WaveGetLaneIndex() - (WaveGetLaneIndex() % {width_limit})) "
                f"+ {width_limit})"
            )
            return (
                f"WaveReadLaneAt({args[1]}, (({partner} < {segment_end}) ? "
                f"{partner} : WaveGetLaneIndex()))"
            )

        if function_name in {"__shfl", "__shfl_up", "__shfl_down", "__shfl_xor"}:
            # HIP's own (non-``_sync``) shuffle family takes the same
            # (value, lane/delta[, width]) arguments minus the mask. hipcc
            # 5.7.1 does not declare the ``_sync`` forms at all, so these are
            # the ones real kernels compile against.
            width_limit = self.hip_warp_shuffle_segment_width(args[2:] or None)
            if len(args) not in {2, 3} or width_limit is None:
                return self.format_unsupported_hip_warp_intrinsic(function_name, args)
            value, lane = args[0], args[1]
            if width_limit == "WaveGetLaneCount()":
                if function_name == "__shfl":
                    # HIP reads lane src & (width - 1), so a negative or
                    # out-of-range source lane still reads inside the wave
                    # (HIP __shfl). Handing the raw index to WaveReadLaneAt is
                    # undefined outside the subgroup.
                    return (
                        f"WaveReadLaneAt({value}, "
                        f"(({lane}) & (WaveGetLaneCount() - 1)))"
                    )
                if function_name == "__shfl_xor":
                    # The partner lane may leave the wave (HIP __shfl_xor
                    # returns the caller's own value there), so the index is
                    # selected before the read; ROCm returns 0 for a disabled
                    # source lane and the partner is inactive when it leaves.
                    partner = f"(WaveGetLaneIndex() ^ {lane})"
                    return (
                        f"WaveReadLaneAt({value}, "
                        f"(({partner} < WaveGetLaneCount()) ? {partner} "
                        f": WaveGetLaneIndex()))"
                    )
                # Select the source index first and read in every lane: the
                # shuffle must not sit in a branch, because the source lane is
                # inactive in the lanes that take the other arm.
                if function_name == "__shfl_up":
                    return (
                        f"WaveReadLaneAt({value}, "
                        f"((WaveGetLaneIndex() >= ({lane})) ? "
                        f"(WaveGetLaneIndex() - ({lane})) : WaveGetLaneIndex()))"
                    )
                return (
                    f"WaveReadLaneAt({value}, "
                    f"(((WaveGetLaneIndex() + ({lane})) < WaveGetLaneCount()) ? "
                    f"(WaveGetLaneIndex() + ({lane})) : WaveGetLaneIndex()))"
                )
            # A width splits the wave into width-lane segments: __shfl reads
            # the same offset inside the caller's segment, and the up/down/xor
            # forms return the caller's own value when the source lane leaves
            # it (HIP __shfl family). These are the same expressions the _sync
            # family uses.
            segment_lane = f"(WaveGetLaneIndex() % {width_limit})"
            segment_base = (
                f"(WaveGetLaneIndex() - (WaveGetLaneIndex() % {width_limit}))"
            )
            if function_name == "__shfl":
                return (
                    f"WaveReadLaneAt({value}, ({segment_base} + "
                    f"(({lane}) & ({width_limit} - 1))))"
                )
            if function_name == "__shfl_up":
                return (
                    f"WaveReadLaneAt({value}, "
                    f"(({segment_lane} >= ({lane})) ? "
                    f"(WaveGetLaneIndex() - ({lane})) : WaveGetLaneIndex()))"
                )
            if function_name == "__shfl_down":
                return (
                    f"WaveReadLaneAt({value}, "
                    f"((({segment_lane} + ({lane})) < {width_limit}) ? "
                    f"(WaveGetLaneIndex() + ({lane})) : WaveGetLaneIndex()))"
                )
            partner = f"(WaveGetLaneIndex() ^ {lane})"
            segment_end = f"({segment_base} + {width_limit})"
            return (
                f"WaveReadLaneAt({value}, (({partner} < {segment_end}) ? "
                f"{partner} : WaveGetLaneIndex()))"
            )

        if function_name == "__ballot":
            return self.format_unsupported_hip_warp_intrinsic(function_name, args)

        return None

    def format_unsupported_hip_warp_intrinsic(self, function_name, args):
        return self.refuse_import(
            f"HIP warp intrinsic {function_name}({', '.join(args)}) has no CrossGL form",
            "0",
        )

    def format_cooperative_group_member_call(self, group_metadata, member, args):
        group_kind = group_metadata["kind"]
        member_base_name, _ = self.parse_cpp_template(member)
        member_name = self.cooperative_group_base_name(member_base_name) or member
        # A thread block's sync() is a statement (emit_thread_block_sync_statement); as a value it is refused below.
        if group_kind == "thread_block" and not args:
            if member_name == "thread_rank":
                return "gl_LocalInvocationIndex"
            if member_name in {"size", "num_threads"}:
                return self.format_thread_block_size_expression()
            if member_name == "thread_index":
                return "gl_LocalInvocationID"
            if member_name == "dim_threads":
                return "gl_WorkGroupSize"

        if group_kind == "thread_block_tile" and not args:
            tile_size = group_metadata.get("tile_size")
            if member_name in {"size", "num_threads"} and tile_size:
                return tile_size
            if (
                member_name == "thread_rank"
                and tile_size
                and group_metadata.get("parent_kind") == "thread_block"
            ):
                return f"(gl_LocalInvocationIndex % {tile_size})"

        if group_kind == "thread_block_tile" and member_name in {
            "shfl",
            "shfl_down",
            "shfl_up",
            "shfl_xor",
        }:
            return self.format_cooperative_group_tile_shuffle(
                group_metadata, member_name, args
            )

        if member_name in {"thread_rank", "size", "num_threads"}:
            return self.format_unsupported_cooperative_group_expression(
                group_kind, member_name
            )
        if member_name in {"thread_index", "dim_threads"}:
            return self.format_unsupported_cooperative_group_expression(
                group_kind, member_name, "vec3<u32>(0, 0, 0)"
            )
        return self.refuse_import(
            f"HIP cooperative_groups {group_kind}.{member_name} has no CrossGL form"
        )

    def format_cooperative_group_tile_shuffle(
        self, group_metadata, member_name, raw_args
    ):
        args = [self.visit(arg) for arg in raw_args]
        if len(args) != 2:
            return self.format_unsupported_cooperative_group_expression(
                group_metadata["kind"], member_name
            )

        value, lane_arg = args
        tile_size = group_metadata.get("tile_size")
        if not tile_size:
            return self.format_unsupported_cooperative_group_expression(
                group_metadata["kind"], member_name
            )

        lane = "WaveGetLaneIndex()"
        local_lane = f"({lane} % {tile_size})"
        if member_name == "shfl":
            source_lane = f"(({lane} - {local_lane}) + {lane_arg})"
            return f"WaveReadLaneAt({value}, {source_lane})"
        if member_name == "shfl_down":
            return (
                f"((({local_lane} + ({lane_arg})) < {tile_size}) ? "
                f"WaveReadLaneAt({value}, ({lane} + ({lane_arg}))) : {value})"
            )
        if member_name == "shfl_up":
            return (
                f"(({local_lane} >= ({lane_arg})) ? "
                f"WaveReadLaneAt({value}, ({lane} - ({lane_arg}))) : {value})"
            )

        source_local_lane = f"({local_lane} ^ ({lane_arg}))"
        source_lane = f"(({lane} - {local_lane}) + {source_local_lane})"
        return (
            f"(({source_local_lane} < {tile_size}) ? "
            f"WaveReadLaneAt({value}, {source_lane}) : {value})"
        )

    def format_unsupported_cooperative_group_expression(
        self, group_kind, member, fallback="0"
    ):
        return self.refuse_import(
            f"HIP cooperative_groups {group_kind}.{member} has no CrossGL form",
            fallback,
        )

    def format_hip_sync_vote_intrinsic_call(self, function_name, args):
        if isinstance(function_name, str) and function_name.startswith("::"):
            function_name = function_name[2:]

        if function_name not in {
            "__syncthreads_count",
            "__syncthreads_and",
            "__syncthreads_or",
        }:
            return None

        return self.refuse_import(
            f"HIP block-wide sync vote {function_name}({', '.join(args)}): CrossGL has no barrier that "
            "reduces a predicate",
            "0",
        )

    def format_unsupported_hip_resource_expression(self, kind, member, fallback):
        return self.refuse_import(f"HIP {kind}.{member} has no CrossGL form", fallback)

    def scale_hip_surface_byte_offset(self, raw_offset, value_type, function_name):
        """Scale a HIP surface x byte offset down to a texel coordinate.

        HIP surface functions take x as a byte offset while CrossGL
        ``imageLoad``/``imageStore`` take a texel coordinate, so the offset is divided by
        the element size. The quotient is folded when the offset is provably a multiple of
        the element size (``x * 4 + 4`` with a 4-byte element becomes ``x + 1``). Anything
        else is refused: passing an unscaled byte offset through would silently address
        the wrong texel.
        """
        offset_text = self.visit(raw_offset) if raw_offset is not None else "?"
        if value_type is None:
            raise HipSurfaceByteOffsetError(
                f"{function_name} cannot scale its x byte offset {offset_text}: "
                "the element type is unknown, so the offset cannot be shown to be "
                "a multiple of the element size"
            )
        byte_size = self.hip_surface_value_type_size(value_type)
        divided = self.divide_hip_surface_byte_offset(raw_offset, byte_size, value_type)
        if divided is None:
            if byte_size is None:
                raise HipSurfaceByteOffsetError(
                    f"{function_name} cannot scale its x byte offset {offset_text}: "
                    f"element type {value_type!r} has no known size, so the offset "
                    "cannot be shown to be a multiple of the element size; write the "
                    f"offset with sizeof({value_type}), for example "
                    f"x * sizeof({value_type})"
                )
            raise HipSurfaceByteOffsetError(
                f"{function_name} x byte offset {offset_text} is not provably a multiple "
                f"of sizeof({value_type}) ({byte_size} bytes); write the offset as a "
                f"multiple of the element size, for example x * sizeof({value_type})"
            )
        return divided[0]

    def visit_NewNode(self, node):
        placement_args = getattr(node, "placement_args", None)
        if placement_args is not None:
            target_type = self.convert_hip_type_to_crossgl(node.target_type)
            args = ", ".join(self.visit(arg) for arg in node.args)
            placement = ", ".join(self.visit(arg) for arg in placement_args)
            fallback = self.visit(placement_args[0]) if placement_args else "nullptr"
            return self.refuse_import(
                f"HIP placement new({placement}) {target_type}({args}): CrossGL has no object construction "
                "in place",
                fallback,
            )

        target_type = self.convert_hip_type_to_crossgl(node.target_type)
        if node.is_array:
            size = self.visit(node.size) if node.size is not None else ""
            return f"new_array<{target_type}>({size})"

        args = ", ".join(self.visit(arg) for arg in node.args)
        return f"new<{target_type}>({args})"

    def get_member_query_path(self, node):
        path = []

        def collect(current):
            if isinstance(current, str):
                return current
            if isinstance(current, MemberAccessNode):
                if getattr(current, "is_pointer", False):
                    return None
                root = collect(current.object)
                if root is None:
                    return None
                path.append(current.member)
                return root
            if isinstance(current, ArrayAccessNode):
                root = collect(current.array)
                if root is None:
                    return None
                index = self.format_member_query_index(current.index)
                if index is None:
                    return None
                if path:
                    path[-1] = f"{path[-1]}[{index}]"
                else:
                    path.append(f"[{index}]")
                return root
            return None

        root_name = collect(node)
        if root_name is None or not path:
            return None
        return root_name, ".".join(path)

    def format_hip_device_attribute_read(self, name):
        if self.suppress_device_attribute_value_access != 0:
            return None

        source = self.lookup_device_attribute_source(name)
        if source is None:
            return None

        attribute_name, device_id = source
        return (
            f"(/* HIP device attribute: {attribute_name}, " f"device: {device_id} */ 0)"
        )

    def format_hip_device_query_read(self, name):
        if self.suppress_device_query_value_access != 0:
            return None

        source = self.lookup_device_query_source(name)
        if source is None:
            return None

        query_name, device_id = source
        if device_id is None:
            return f"(/* HIP device query: {query_name} */ 0)"
        return f"(/* HIP device query: {query_name}, device: {device_id} */ 0)"

    def visit_SyncNode(self, node):
        if node.sync_type == "__syncthreads":
            # HIP's __syncthreads orders global and shared memory for the block, as CUDA's does.
            self.emit_block_barrier()
        elif node.sync_type == "hipDeviceSynchronize" and self.in_device_code:
            self.refuse_device_runtime_call("runtime call hipDeviceSynchronize()")
        elif node.sync_type == "hipDeviceSynchronize":
            self.emit("// HIP device synchronize")
        elif node.sync_type == "__syncwarp" and self.is_full_warp_mask(node.args):
            # __syncwarp over the whole warp waits for every lane and orders their memory accesses, as
            # subgroupBarrier does for the subgroup.
            self.emit("subgroupBarrier();")
        elif node.sync_type == "__syncwarp":
            args = ", ".join(self.visit(arg) for arg in node.args)
            self.refuse_import(
                f"HIP __syncwarp({args}): CrossGL has no barrier over a subset of the subgroup"
            )
        else:
            self.refuse_import(f"HIP {node.sync_type}() has no CrossGL form")

    def emit_block_barrier(self):
        """Emit ``__syncthreads()``, which orders global and shared memory for the block as CUDA's does."""
        for function_name in ORDERED_WORKGROUP_BARRIER:
            self.emit(f"{function_name}();")

    def is_full_warp_mask(self, args):
        """Whether ``__syncwarp`` arguments name every lane: none, or an all-ones 32- or 64-bit literal mask."""
        if not args:
            return True
        if len(args) != 1:
            return False
        value = self.visit(args[0]).strip().rstrip("uUlL")
        try:
            return int(value, 0) in (0xFFFFFFFF, 0xFFFFFFFFFFFFFFFF)
        except ValueError:
            return False

    def visit_CastNode(self, node):
        self.validate_record_conversion(
            node.expression, node.target_type, "explicit conversion"
        )
        target_type = self.convert_hip_type_to_crossgl(node.target_type)
        expression = self.visit(node.expression)
        return f"{target_type}({expression})"

    def convert_hip_type_to_crossgl(self, hip_type):
        """Map a HIP type name to the closest CrossGL type name."""
        if hip_type is None:
            return "void"

        if not isinstance(hip_type, str):
            hip_type = str(hip_type)

        hip_type = self.strip_type_qualifiers(hip_type)
        hip_type = self.strip_variadic_type_marker(hip_type)
        concrete_alias = self.resolve_concrete_type_alias(hip_type)
        if concrete_alias != hip_type:
            hip_type = concrete_alias
        elif "::" in hip_type:
            hip_type = self.resolve_type_alias(hip_type)
        hip_type = self.strip_elaborated_record_type_keyword(hip_type)
        hip_type = self.resolve_namespace_alias_name(hip_type)
        hip_type = self.canonical_cpp_scalar_type(hip_type)
        matrix_type = self.convert_native_matrix_helper_name_to_crossgl(hip_type)
        if matrix_type is not None:
            return matrix_type
        cooperative_group_type = self.convert_cooperative_group_type(hip_type)
        if cooperative_group_type is not None:
            return cooperative_group_type

        type_mapping = {
            # Basic types
            "void": "void",
            "bool": "bool",
            "char": "i8",
            "signed char": "i8",
            "unsigned char": "u8",
            "short": "i16",
            "signed short": "i16",
            "short int": "i16",
            "signed short int": "i16",
            "unsigned short": "u16",
            "unsigned short int": "u16",
            "int": "i32",
            "signed": "i32",
            "unsigned": "u32",
            "signed int": "i32",
            "uint": "u32",
            "unsigned int": "u32",
            "long": "i64",
            "signed long": "i64",
            "long int": "i64",
            "signed long int": "i64",
            "unsigned long": "u64",
            "unsigned long int": "u64",
            "long unsigned int": "u64",
            "long long": "i64",
            "signed long long": "i64",
            "long long int": "i64",
            "signed long long int": "i64",
            "unsigned long long": "u64",
            "unsigned long long int": "u64",
            "long long unsigned int": "u64",
            "__int64": "i64",
            "signed __int64": "i64",
            "unsigned __int64": "u64",
            "int8_t": "i8",
            "uint8_t": "u8",
            "int16_t": "i16",
            "uint16_t": "u16",
            "int32_t": "i32",
            "uint32_t": "u32",
            "int64_t": "i64",
            "uint64_t": "u64",
            "half": "f16",
            "__half": "f16",
            "__half2": "vec2<f16>",
            "float16_t": "f16",
            "float32_t": "f32",
            "float64_t": "f64",
            "rocwmma::float16_t": "f16",
            "rocwmma::float32_t": "f32",
            "rocwmma::float64_t": "f64",
            "hipComplex": "vec2<f32>",
            "hipFloatComplex": "vec2<f32>",
            "hipDoubleComplex": "vec2<f64>",
            "float": "f32",
            "double": "f64",
            "size_t": "u64",
            "ptrdiff_t": "i64",
            "hipArray": "ptr<void>",
            "hipArray_t": "ptr<void>",
            "hipTextureObject_t": "sampler",
            "hipSurfaceObject_t": "image2D",
            # HIP vector types
            **self.VECTOR1_TYPE_MAPPING,
            **self.VECTOR_TYPE_MAPPING,
            "dim3": "vec3<u32>",
        }

        unique_ptr_type = self.convert_unique_ptr_type(hip_type)
        if unique_ptr_type is not None:
            return unique_ptr_type

        std_container_type = self.convert_std_container_type(hip_type)
        if std_container_type is not None:
            return std_container_type

        enable_if_type = self.convert_enable_if_type(hip_type)
        if enable_if_type is not None:
            return enable_if_type

        resource_type = self.convert_hip_resource_type(hip_type)
        if resource_type is not None:
            return resource_type

        if self.has_array_suffix(hip_type):
            return self.convert_hip_array_type(hip_type, type_mapping)

        _, pointer_depth = self.split_pointer_declarators(hip_type)
        if pointer_depth:
            return self.convert_hip_pointer_type(hip_type)

        if self.is_decltype_type_name(hip_type):
            return "auto"

        if hip_type in self.CROSSGL_RESERVED_IDENTIFIERS:
            # A struct named after a CrossGL keyword is declared under its renamed name.
            return self.sanitize_identifier_name(hip_type)
        if hip_type in type_mapping:
            return type_mapping[hip_type]
        # A type the program declares, or a HIP built-in type carried into the IR as a plain name; each generator
        # maps the built-ins it knows and refuses the rest (written_type_names).
        return hip_type

    # Legacy method for backwards compatibility
    def convert(self, node):
        """Legacy convert method for compatibility"""
        return self.generate(node)


def hip_to_crossgl(hip_ast) -> str:
    """Convert HIP AST to CrossGL code string"""
    converter = HipToCrossGLConverter()
    return converter.generate(hip_ast)

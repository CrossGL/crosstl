"""Class-level tables of the GLSL generator: software subgroup and wave operations, built-in functions, boolean and arithmetic operators, ray query names, extension and version requirements, and qualifier prefixes."""

from ..refusals import (
    OpenGLRefusal,
    TranslatorArgumentError,
)
from .glsl_keywords import GLSL_TYPE_KEYWORDS
from .glsl_refusals import OpenGLLoweringRefused
from .glsl_wave_extended import (
    GLSL_CLUSTERED_BITWISE_OPERATIONS,
    GLSL_CLUSTERED_NUMERIC_OPERATIONS,
    GLSL_EXTENDED_WAVE_ARITIES,
    GLSL_EXTENDED_WAVE_DIRECT_MAPPINGS,
    GLSL_EXTENDED_WAVE_EXTENSIONS,
    GLSL_QUAD_VOTE_FUNCTIONS,
)
from .index_range_contracts import OPENGL_INDEX_PROFILE
from .opengl_values import (
    OpenGLArithmeticConversionError,
    OpenGLBooleanCompoundAssignmentError,
    OpenGLBooleanOrderedIntrinsicError,
    OpenGLIndexTypeError,
    OpenGLPreciseQualifierError,
    OpenGLStructConstructionError,
)
from .software_quad_reductions import SOFTWARE_QUAD_REDUCTIONS


class OpenGLTableMixin:
    """Table members of ``GLSLCodeGen``."""

    GLSL_TARGET_DISPLAY_NAME = "OpenGL"
    # Separate samplers and runtime texture arrays keep their own declarations under ``#ifdef VULKAN``
    # (glsl_vulkan_resources).
    GLSL_VULKAN_RESOURCE_FORMS = True
    # One compilation unit holds one entry per stage, main (glsl_stage_entries.refuse_repeated_stage_entries).
    one_entry_per_stage = True

    GLSL_UNAVAILABLE_FUNCTION_PATTERNS = ()
    LOWERING_REFUSAL_TYPE = OpenGLLoweringRefused
    # The type names the program being rendered declares, and the foreign built-in type names its rendering has
    # written so far (``None`` outside a rendering); see :func:`written_type_name`.
    program_type_names = frozenset()
    written_foreign_type_names = None
    GLSL_ARITHMETIC_CONVERSION_ERROR = OpenGLArithmeticConversionError
    GLSL_BOOLEAN_COMPOUND_ASSIGNMENT_ERROR = OpenGLBooleanCompoundAssignmentError
    GLSL_BOOLEAN_ORDERED_INTRINSIC_ERROR = OpenGLBooleanOrderedIntrinsicError
    GLSL_STRUCT_CONSTRUCTION_ERROR = OpenGLStructConstructionError
    GLSL_INDEX_RANGE_ERROR = OpenGLIndexTypeError
    GLSL_PRECISE_QUALIFIER_ERROR = OpenGLPreciseQualifierError
    GLSL_PRECISE_QUALIFIER_REMEDY = "select GLSL ES 3.20 or desktop GLSL 4.00"
    GLSL_INDEX_TARGET_PROFILE = OPENGL_INDEX_PROFILE
    GLSL_SUPPORTED_ARITHMETIC_INTEGER_WIDTHS = frozenset({32, 64})
    GLSL_SUPPORTED_ARITHMETIC_FLOATING_WIDTHS = frozenset({32, 64})
    GLSL_PRIVATE_POINTER_MAX_REBASE_ITERATIONS = 4096
    GLSL_PRIVATE_POINTER_BASE_MAX = (1 << 31) - 1
    METAL_ONLY_SYSTEM_INCLUDES = frozenset(
        {
            "metal_math",
            "metal_math.h",
            "metal_stdlib",
            "metal_stdlib.h",
            # Apple's platform-conditional macros, which shared Metal and host headers include.
            "targetconditionals.h",
        }
    )
    MESH_STAGE_NAMES = {"mesh", "task", "amplification", "object"}
    UNSIGNED_VECTOR_COMPUTE_BUILTINS = {
        "gl_GlobalInvocationID",
        "gl_LocalInvocationID",
        "gl_WorkGroupID",
        "gl_NumWorkGroups",
        "gl_WorkGroupSize",
    }
    GLSL_STAGE_GUARD_MACROS = {
        "vertex": "GL_VERTEX_SHADER",
        "fragment": "GL_FRAGMENT_SHADER",
        "compute": "GL_COMPUTE_SHADER",
        "geometry": "GL_GEOMETRY_SHADER",
        "tessellation_control": "GL_TESS_CONTROL_SHADER",
        "tessellation_evaluation": "GL_TESS_EVALUATION_SHADER",
        "mesh": "GL_MESH_SHADER_EXT",
        "task": "GL_TASK_SHADER_EXT",
        "amplification": "GL_TASK_SHADER_EXT",
        "object": "GL_TASK_SHADER_EXT",
        "ray_generation": "GL_RAYGEN_SHADER_EXT",
        "ray_intersection": "GL_INTERSECTION_SHADER_EXT",
        "ray_any_hit": "GL_ANY_HIT_SHADER_EXT",
        "ray_closest_hit": "GL_CLOSEST_HIT_SHADER_EXT",
        "ray_miss": "GL_MISS_SHADER_EXT",
        "ray_callable": "GL_CALLABLE_SHADER_EXT",
    }
    TEXTURE_SIZE_NO_LOD_SUFFIXES = ("2DRect", "Buffer")
    sampled_resource_type_aliases = {
        "Texture1D": "sampler1D",
        "Texture1DArray": "sampler1DArray",
        "Texture2D": "sampler2D",
        "Texture2DArray": "sampler2DArray",
        "Texture2DMS": "sampler2DMS",
        "Texture2DMSArray": "sampler2DMSArray",
        "Texture3D": "sampler3D",
        "TextureCube": "samplerCube",
        "TextureCubeArray": "samplerCubeArray",
    }
    storage_resource_type_aliases = {
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
    sampler_state_type_aliases = {
        "SamplerState": "sampler",
        "SamplerComparisonState": "comparison_sampler",
    }
    RAY_STAGE_NAMES = {
        "ray_generation",
        "ray_intersection",
        "ray_any_hit",
        "ray_closest_hit",
        "ray_miss",
        "ray_callable",
        "intersection",
        "anyhit",
        "closesthit",
        "miss",
        "callable",
    }
    RAY_QUERY_METHOD_MAP = {
        "TraceRayInline": ("rayQueryInitializeEXT", None),
        "Proceed": ("rayQueryProceedEXT", None),
        "Abort": ("rayQueryTerminateEXT", None),
        "CommitProceduralPrimitiveHit": ("rayQueryGenerateIntersectionEXT", None),
        "CommitNonOpaqueTriangleHit": ("rayQueryConfirmIntersectionEXT", None),
        "RayTMin": ("rayQueryGetRayTMinEXT", None),
        "RayFlags": ("rayQueryGetRayFlagsEXT", None),
        "WorldRayOrigin": ("rayQueryGetWorldRayOriginEXT", None),
        "WorldRayDirection": ("rayQueryGetWorldRayDirectionEXT", None),
        "CandidateType": ("rayQueryGetIntersectionTypeEXT", "false"),
        "CommittedStatus": ("rayQueryGetIntersectionTypeEXT", "true"),
        "CandidatePrimitiveIndex": (
            "rayQueryGetIntersectionPrimitiveIndexEXT",
            "false",
        ),
        "CommittedPrimitiveIndex": (
            "rayQueryGetIntersectionPrimitiveIndexEXT",
            "true",
        ),
        # HLSL semantics: InstanceID is the application-defined custom index,
        # InstanceIndex the instance's position in the top-level structure.
        "CandidateInstanceID": (
            "rayQueryGetIntersectionInstanceCustomIndexEXT",
            "false",
        ),
        "CommittedInstanceID": (
            "rayQueryGetIntersectionInstanceCustomIndexEXT",
            "true",
        ),
        "CandidateInstanceIndex": ("rayQueryGetIntersectionInstanceIdEXT", "false"),
        "CommittedInstanceIndex": ("rayQueryGetIntersectionInstanceIdEXT", "true"),
        "CandidateGeometryIndex": (
            "rayQueryGetIntersectionGeometryIndexEXT",
            "false",
        ),
        "CommittedGeometryIndex": (
            "rayQueryGetIntersectionGeometryIndexEXT",
            "true",
        ),
        "CandidateInstanceContributionToHitGroupIndex": (
            "rayQueryGetIntersectionInstanceShaderBindingTableRecordOffsetEXT",
            "false",
        ),
        "CommittedInstanceContributionToHitGroupIndex": (
            "rayQueryGetIntersectionInstanceShaderBindingTableRecordOffsetEXT",
            "true",
        ),
        "CandidateObjectRayOrigin": (
            "rayQueryGetIntersectionObjectRayOriginEXT",
            "false",
        ),
        "CandidateObjectRayDirection": (
            "rayQueryGetIntersectionObjectRayDirectionEXT",
            "false",
        ),
        "CommittedObjectRayOrigin": (
            "rayQueryGetIntersectionObjectRayOriginEXT",
            "true",
        ),
        "CommittedObjectRayDirection": (
            "rayQueryGetIntersectionObjectRayDirectionEXT",
            "true",
        ),
        "CandidateTriangleRayT": ("rayQueryGetIntersectionTEXT", "false"),
        "CommittedRayT": ("rayQueryGetIntersectionTEXT", "true"),
        "CandidateTriangleBarycentrics": (
            "rayQueryGetIntersectionBarycentricsEXT",
            "false",
        ),
        "CommittedTriangleBarycentrics": (
            "rayQueryGetIntersectionBarycentricsEXT",
            "true",
        ),
        "CandidateTriangleFrontFace": (
            "rayQueryGetIntersectionFrontFaceEXT",
            "false",
        ),
        "CommittedTriangleFrontFace": (
            "rayQueryGetIntersectionFrontFaceEXT",
            "true",
        ),
        "CandidateTriangleVertexPositions": (
            "rayQueryGetIntersectionTriangleVertexPositionsEXT",
            "false",
        ),
        "CommittedTriangleVertexPositions": (
            "rayQueryGetIntersectionTriangleVertexPositionsEXT",
            "true",
        ),
        # DXR asks whether the candidate is non-opaque; GLSL whether it is opaque.
        "CandidateProceduralPrimitiveNonOpaque": (
            "rayQueryGetIntersectionCandidateAABBOpaqueEXT",
            None,
        ),
        "CandidateObjectToWorld": (
            "rayQueryGetIntersectionObjectToWorldEXT",
            "false",
        ),
        "CommittedObjectToWorld": (
            "rayQueryGetIntersectionObjectToWorldEXT",
            "true",
        ),
        "CandidateObjectToWorld3x4": (
            "rayQueryGetIntersectionObjectToWorldEXT",
            "false",
        ),
        "CommittedObjectToWorld3x4": (
            "rayQueryGetIntersectionObjectToWorldEXT",
            "true",
        ),
        "CandidateWorldToObject": (
            "rayQueryGetIntersectionWorldToObjectEXT",
            "false",
        ),
        "CommittedWorldToObject": (
            "rayQueryGetIntersectionWorldToObjectEXT",
            "true",
        ),
        "CandidateWorldToObject3x4": (
            "rayQueryGetIntersectionWorldToObjectEXT",
            "false",
        ),
        "CommittedWorldToObject3x4": (
            "rayQueryGetIntersectionWorldToObjectEXT",
            "true",
        ),
    }
    # Ray query methods with a bool result, so `!rq.Proceed()` has a typed operand.
    RAY_QUERY_BOOL_METHODS = frozenset(
        {
            "Proceed",
            "CandidateTriangleFrontFace",
            "CommittedTriangleFrontFace",
            "CandidateProceduralPrimitiveNonOpaque",
        }
    )
    RAY_QUERY_POSITION_FETCH_METHODS = {
        "CandidateTriangleVertexPositions",
        "CommittedTriangleVertexPositions",
    }
    RAY_QUERY_METHOD_ARITIES = {
        "TraceRayInline": (4, 7),
        "Proceed": (0,),
        "Abort": (0,),
        "CommitProceduralPrimitiveHit": (1,),
        "CommitNonOpaqueTriangleHit": (0,),
        "RayTMin": (0,),
        "RayFlags": (0,),
        "WorldRayOrigin": (0,),
        "WorldRayDirection": (0,),
        "CandidateType": (0,),
        "CommittedStatus": (0,),
        "CandidatePrimitiveIndex": (0,),
        "CommittedPrimitiveIndex": (0,),
        "CandidateInstanceID": (0,),
        "CommittedInstanceID": (0,),
        "CandidateGeometryIndex": (0,),
        "CommittedGeometryIndex": (0,),
        "CandidateInstanceIndex": (0,),
        "CommittedInstanceIndex": (0,),
        "CandidateInstanceContributionToHitGroupIndex": (0,),
        "CommittedInstanceContributionToHitGroupIndex": (0,),
        "CandidateObjectRayOrigin": (0,),
        "CandidateObjectRayDirection": (0,),
        "CommittedObjectRayOrigin": (0,),
        "CommittedObjectRayDirection": (0,),
        "CandidateTriangleRayT": (0,),
        "CommittedRayT": (0,),
        "CandidateTriangleBarycentrics": (0,),
        "CommittedTriangleBarycentrics": (0,),
        "CandidateTriangleFrontFace": (0,),
        "CommittedTriangleFrontFace": (0,),
        "CandidateTriangleVertexPositions": (1,),
        "CommittedTriangleVertexPositions": (1,),
        "CandidateProceduralPrimitiveNonOpaque": (0,),
        "CandidateObjectToWorld": (0,),
        "CommittedObjectToWorld": (0,),
        "CandidateObjectToWorld3x4": (0,),
        "CommittedObjectToWorld3x4": (0,),
        "CandidateWorldToObject": (0,),
        "CommittedWorldToObject": (0,),
        "CandidateWorldToObject3x4": (0,),
        "CommittedWorldToObject3x4": (0,),
    }
    RAY_QUERY_RAY_DESC_FIELDS = (
        ("Origin", "origin", "rayOrigin", "RayOrigin"),
        ("TMin", "tMin", "Tmin", "tmin"),
        ("Direction", "direction", "rayDirection", "RayDirection"),
        ("TMax", "tMax", "Tmax", "tmax"),
    )
    GLSL_LAYOUT_ATTRIBUTE_NAMES = (
        "location",
        "component",
        "index",
        "stream",
        "xfb_buffer",
        "xfb_offset",
        "xfb_stride",
    )
    GLSL_BUFFER_BLOCK_MEMORY_LAYOUT_NAMES = {
        "packed",
        "scalar",
        "shared",
        "std140",
        "std430",
    }
    GLSL_BLEND_SUPPORT_LAYOUT_NAMES = {
        "blend_support_multiply",
        "blend_support_screen",
        "blend_support_overlay",
        "blend_support_darken",
        "blend_support_lighten",
        "blend_support_colordodge",
        "blend_support_colorburn",
        "blend_support_hardlight",
        "blend_support_softlight",
        "blend_support_difference",
        "blend_support_exclusion",
        "blend_support_hsl_hue",
        "blend_support_hsl_saturation",
        "blend_support_hsl_color",
        "blend_support_hsl_luminosity",
        "blend_support_all_equations",
    }
    GLSL_BARE_LAYOUT_ATTRIBUTE_NAMES = {
        "depth_any",
        "depth_greater",
        "depth_less",
        "depth_unchanged",
        *GLSL_BLEND_SUPPORT_LAYOUT_NAMES,
    }
    GLSL_VARIABLE_QUALIFIER_ATTRIBUTE_NAMES = {
        "invariant",
        "precise",
        "patch",
        "flat",
        "smooth",
        "noperspective",
        "centroid",
        "sample",
        "in",
        "out",
        "inout",
        "lowp",
        "mediump",
        "highp",
    }
    GLSL_STORAGE_QUALIFIER_ATTRIBUTE_NAMES = {
        "shared",
        "groupshared",
        "workgroup",
        "taskpayloadshared",
        "taskpayloadsharedext",
        "task_payload_shared",
        "raypayload",
        "raypayloadext",
        "raypayloadin",
        "raypayloadinext",
        "hitattribute",
        "hitattributeext",
        "callabledata",
        "callabledataext",
        "callabledatain",
        "callabledatainext",
        "tileimage",
        "tileimageext",
    }
    GLSL_PRECISION_QUALIFIERS = {"lowp", "mediump", "highp"}
    GLSL_PARAMETER_QUALIFIERS = {"out", "inout"}
    # Words that GLSL 4.60 reserves for future use (section 3.6), qualifiers and
    # non-type keywords. Source languages such as HLSL accept them as ordinary
    # identifiers (``this``, ``resource``, ``filter``), so they are escaped.
    # Mesa also lexes ``packed``, reserved up to GLSL 1.30, as a keyword.
    GLSL_RESERVED_IDENTIFIERS = (
        GLSL_VARIABLE_QUALIFIER_ATTRIBUTE_NAMES
        | {
            "active",
            "asm",
            "attribute",
            "barrier",
            "buffer",
            "cast",
            "centroid",
            "class",
            "coherent",
            "common",
            "enum",
            "extern",
            "external",
            "filter",
            "fixed",
            "flat",
            "fvec2",
            "fvec3",
            "fvec4",
            "goto",
            "half",
            "highp",
            "hvec2",
            "hvec3",
            "hvec4",
            "inline",
            "input",
            "interface",
            "invariant",
            "layout",
            "long",
            "lowp",
            "mediump",
            "namespace",
            "noinline",
            "noperspective",
            "output",
            "packed",
            "partition",
            "patch",
            "precise",
            "precision",
            "public",
            "readonly",
            "resource",
            "restrict",
            "sample",
            "sampler3DRect",
            "shared",
            "short",
            "sizeof",
            "smooth",
            "static",
            "subroutine",
            "superp",
            "template",
            "this",
            "typedef",
            "uniform",
            "union",
            "unsigned",
            "using",
            "varying",
            "volatile",
            "writeonly",
        }
        | GLSL_TYPE_KEYWORDS
    )
    GLSL_RESERVED_IDENTIFIER_PREFIXES = ("gl_",)
    # GLSL ES rejects a global or a function that spells a built-in function; desktop GLSL only a call it hides.
    GLSL_RESERVES_BUILTIN_FUNCTION_NAMES = False
    GLSL_BUILTIN_INTERFACE_BLOCK_NAMES = {"gl_PerVertex"}
    COMPOUND_ASSIGNMENT_BINARY_OPERATORS = {
        "+=": "+",
        "-=": "-",
        "*=": "*",
        "/=": "/",
        "%=": "%",
        "<<=": "<<",
        ">>=": ">>",
        "&=": "&",
        "^=": "^",
        "|=": "|",
    }
    BOOLEAN_ARITHMETIC_COMPOUND_ASSIGNMENT_BINARY_OPERATORS = {
        "+=": "+",
        "-=": "-",
        "*=": "*",
        "/=": "/",
        "%=": "%",
        "<<=": "<<",
        ">>=": ">>",
    }
    GLSL_BFLOAT16_ALIASES = {"bfloat", "bfloat16", "bfloat16_t"}
    GLSL_COPYSIGN_NAMES = {
        "copysign",
        "metal::copysign",
        "metal_u3a_u3acopysign",
    }
    GLSL_ISFINITE_NAMES = {
        "isfinite",
        "metal::isfinite",
        "metal_u3a_u3aisfinite",
    }
    GLSL_SIGNBIT_NAMES = {
        "signbit",
        "metal::signbit",
        "metal_u3a_u3asignbit",
    }
    # Built-in functions the generator calls in place of source operations (HLSL methods and intrinsics). A local
    # of the same name would hide the function in its scope ("can't use function syntax on variable").
    GLSL_ALIAS_TARGET_LOCAL_IDENTIFIERS = {
        "clamp",
        "dFdx",
        "dFdy",
        "floatBitsToInt",
        "floatBitsToUint",
        "fract",
        "imageLoad",
        "imageSamples",
        "imageSize",
        "imageStore",
        "intBitsToFloat",
        "inversesqrt",
        "mix",
        "mod",
        "texelFetch",
        "texelFetchOffset",
        "texture",
        "textureGather",
        "textureGatherOffset",
        "textureGrad",
        "textureLod",
        "textureLodOffset",
        "textureOffset",
        "textureProj",
        "textureQueryLevels",
        "textureQueryLod",
        "textureSamples",
        "textureSize",
        "uintBitsToFloat",
    }
    GLSL_BITCAST_FUNCTIONS = {"asfloat", "asint", "asuint"}
    GLSL_COMPONENTWISE_UNARY_FUNCTIONS = {
        "abs",
        "acos",
        "acosh",
        "asin",
        "asinh",
        "atan",
        "atanh",
        "ceil",
        "cos",
        "cosh",
        "degrees",
        "exp",
        "exp2",
        "floor",
        "frac",
        "fract",
        "inversesqrt",
        "log",
        "log2",
        "radians",
        "round",
        "roundEven",
        "sign",
        "sin",
        "sinh",
        "sqrt",
        "tan",
        "tanh",
        "trunc",
    }
    GLSL_COMPONENTWISE_BOOLEAN_FUNCTIONS = {"isfinite", "isinf", "isnan"}
    GLSL_BOOLEAN_REDUCTION_FUNCTIONS = {"all", "any"}
    GLSL_INT64_EXTENSION = "GL_ARB_gpu_shader_int64"
    GLSL_INT64_EXTENSION_LINE = "#extension GL_ARB_gpu_shader_int64 : require"
    GLSL_INT64_SCALAR_TYPES = {"int64_t", "uint64_t"}
    GLSL_SUBGROUP_BASIC_EXTENSION_LINE = (
        "#extension GL_KHR_shader_subgroup_basic : require"
    )
    GLSL_EXACT_SUBGROUP_WIDTHS = frozenset({1, 2, 4, 8, 16, 32, 64, 128})
    GLSL_REQUIRED_SUBGROUP_WIDTH_MACRO = "CROSSTL_REQUIRED_SUBGROUP_WIDTH"
    GLSL_SOFTWARE_SUBGROUP_WIDTH_MACRO = "CROSSTL_SOFTWARE_SUBGROUP_WIDTH"
    GLSL_SOFTWARE_SUBGROUP_SUPPORTED_WIDTH = 32
    GLSL_SOFTWARE_SUBGROUP_VOTES = frozenset({"WaveActiveAllTrue", "WaveActiveAnyTrue"})
    GLSL_SOFTWARE_SUBGROUP_SCANS = {
        "WavePrefixSum": ("+", False),
        "WavePrefixProduct": ("*", False),
        "WavePrefixInclusiveSum": ("+", True),
        "WavePrefixInclusiveProduct": ("*", True),
    }
    # GL_KHR_shader_subgroup_basic barriers, plus CrossGL's execution-only subgroup barrier.
    GLSL_SUBGROUP_BARRIER_FUNCTIONS = frozenset(
        {
            "subgroupBarrier",
            "subgroupExecutionBarrier",
            "subgroupMemoryBarrier",
            "subgroupMemoryBarrierBuffer",
            "subgroupMemoryBarrierShared",
            "subgroupMemoryBarrierImage",
        }
    )
    # Subgroup barriers that software subgroups lower to the workgroup barrier.
    GLSL_SOFTWARE_SUBGROUP_BARRIERS = frozenset(
        {"subgroupExecutionBarrier", "subgroupBarrier"}
    )
    # Workgroup-scope fences that order the memory of each subgroup fence for the subgroup as well.
    GLSL_SUBGROUP_FENCE_WIDENINGS = {
        "subgroupMemoryBarrier": "groupMemoryBarrier",
        "subgroupMemoryBarrierBuffer": "memoryBarrierBuffer",
        "subgroupMemoryBarrierShared": "memoryBarrierShared",
        "subgroupMemoryBarrierImage": "memoryBarrierImage",
    }
    GLSL_SOFTWARE_SUBGROUP_OPERATIONS = frozenset(
        {
            "WaveActiveSum",
            "WaveActiveProduct",
            "WaveActiveMin",
            "WaveActiveMax",
            "WaveShuffleDown",
            "WaveShuffleUp",
            "WaveShuffleXor",
            "WaveReadLaneAt",
            "WaveShuffleAndFillUp",
            "subgroupExecutionBarrier",
            "subgroupBarrier",
            "CooperativeMatrixMultiplyAccumulate",
            *GLSL_SOFTWARE_SUBGROUP_VOTES,
            *GLSL_SOFTWARE_SUBGROUP_SCANS,
            *SOFTWARE_QUAD_REDUCTIONS,
        }
    )
    # Relative shuffles: source-lane validity and source lane. A source lane
    # outside the active subgroup yields the calling lane's value.
    GLSL_SOFTWARE_SUBGROUP_RELATIVE_SHUFFLES = {
        "WaveShuffleDown": ("delta < ({active} - lane)", "lane + delta"),
        "WaveShuffleUp": ("delta <= lane", "lane - delta"),
        "WaveShuffleXor": ("(lane ^ delta) < {active}", "lane ^ delta"),
    }
    GLSL_SOFTWARE_SUBGROUP_LANE_SHUFFLES = frozenset(
        {"WaveReadLaneAt", *GLSL_SOFTWARE_SUBGROUP_RELATIVE_SHUFFLES}
    )
    GLSL_SOFTWARE_SUBGROUP_VALUE_TYPES = frozenset({"float", "int", "uint"})
    GLSL_SOFTWARE_SUBGROUP_BUILTIN_ALIASES = frozenset(
        {
            "gl_NumSubgroups",
            "gl_SubgroupID",
            "gl_SubgroupSize",
            "gl_SubgroupInvocationID",
        }
    )
    GLSL_SOFTWARE_SUBGROUP_WORKGROUP_UNIFORM_BUILTINS = frozenset(
        {
            "gl_WorkGroupID",
            "gl_NumWorkGroups",
            "gl_WorkGroupSize",
            "gl_NumSubgroups",
            "gl_SubgroupSize",
        }
    )
    GLSL_SOFTWARE_SUBGROUP_TYPE_SUFFIXES = {
        "bool": "Bool",
        "float": "Float",
        "half": "Half",
        "int": "Int",
        "uint": "Uint",
        "uvec2": "Uvec2",
    }
    GLSL_CLANG_TRAILING_ZERO_BUILTINS = frozenset(
        {"__builtin_ctz", "__builtin_ctzl", "__builtin_ctzll"}
    )
    GLSL_CLANG_TRAILING_ZERO_WIDTHS = {
        "__builtin_ctz": 32,
        "__builtin_ctzl": 64,
        "__builtin_ctzll": 64,
    }
    GLSL_NARROW_INTEGER_TYPE_NAMES = frozenset(
        {
            "char",
            "signed char",
            "unsigned char",
            "uchar",
            "i8",
            "u8",
            "int8",
            "uint8",
            "int8_t",
            "uint8_t",
            "short",
            "signed short",
            "unsigned short",
            "ushort",
            "i16",
            "u16",
            "int16",
            "uint16",
            "int16_t",
            "uint16_t",
        }
    )
    GLSL_INTERPOLATION_FUNCTIONS = {
        "interpolateAtCentroid": 1,
        "interpolateAtSample": 2,
        "interpolateAtOffset": 2,
    }
    GLSL_DERIVATIVE_FUNCTIONS = {
        "dFdx": 1,
        "dFdy": 1,
        "fwidth": 1,
        "dFdxFine": 1,
        "dFdxCoarse": 1,
        "dFdyFine": 1,
        "dFdyCoarse": 1,
        "fwidthFine": 1,
        "fwidthCoarse": 1,
    }
    GLSL_DERIVATIVE_FUNCTION_ALIASES = {
        "ddx": "dFdx",
        "ddy": "dFdy",
        "fwidth": "fwidth",
        "ddx_fine": "dFdxFine",
        "ddx_coarse": "dFdxCoarse",
        "ddy_fine": "dFdyFine",
        "ddy_coarse": "dFdyCoarse",
        "fwidth_fine": "fwidthFine",
        "fwidth_coarse": "fwidthCoarse",
    }
    GLSL_WAVE_INTRINSIC_ARITIES = {
        "WaveGetLaneCount": 0,
        "WaveGetLaneIndex": 0,
        "WaveIsFirstLane": 0,
        "WaveActiveSum": 1,
        "WaveActiveProduct": 1,
        "WaveActiveBitAnd": 1,
        "WaveActiveBitOr": 1,
        "WaveActiveBitXor": 1,
        "WaveActiveMin": 1,
        "WaveActiveMax": 1,
        "WaveActiveAllTrue": 1,
        "WaveActiveAnyTrue": 1,
        "WaveActiveAllEqual": 1,
        "WaveActiveCountBits": 1,
        "WaveActiveBallot": 1,
        "WaveReadLaneAt": 2,
        "WaveReadLaneFirst": 1,
        "WaveShuffleDown": 2,
        "WaveShuffleUp": 2,
        "WaveShuffleAndFillUp": 3,
        "WaveShuffleXor": 2,
        "WavePrefixSum": 1,
        "WavePrefixProduct": 1,
        "WavePrefixInclusiveSum": 1,
        "WavePrefixInclusiveProduct": 1,
        "WavePrefixExclusiveMin": 1,
        "WavePrefixInclusiveMin": 1,
        "WavePrefixExclusiveMax": 1,
        "WavePrefixInclusiveMax": 1,
        "WavePrefixExclusiveBitAnd": 1,
        "WavePrefixInclusiveBitAnd": 1,
        "WavePrefixExclusiveBitOr": 1,
        "WavePrefixInclusiveBitOr": 1,
        "WavePrefixExclusiveBitXor": 1,
        "WavePrefixInclusiveBitXor": 1,
        "WavePrefixCountBits": 1,
        "QuadActiveSum": 1,
        "QuadReadAcrossX": 1,
        "QuadReadAcrossY": 1,
        "QuadReadAcrossDiagonal": 1,
        "QuadReadLaneAt": 2,
        "WaveMatch": 1,
        "WaveMultiPrefixSum": 2,
        "WaveMultiPrefixCountBits": 2,
        "WaveMultiPrefixProduct": 2,
        "WaveMultiPrefixBitAnd": 2,
        "WaveMultiPrefixBitOr": 2,
        "WaveMultiPrefixBitXor": 2,
        **GLSL_EXTENDED_WAVE_ARITIES,
    }
    GLSL_WAVE_INTRINSIC_ALTERNATE_ARITIES = {
        "WaveShuffleAndFillUp": (3, 4),
    }
    GLSL_WAVE_DIRECT_MAPPINGS = {
        "WaveActiveSum": "subgroupAdd",
        "WaveActiveProduct": "subgroupMul",
        "WaveActiveBitAnd": "subgroupAnd",
        "WaveActiveBitOr": "subgroupOr",
        "WaveActiveBitXor": "subgroupXor",
        "WaveActiveMin": "subgroupMin",
        "WaveActiveMax": "subgroupMax",
        "WaveActiveAllTrue": "subgroupAll",
        "WaveActiveAnyTrue": "subgroupAny",
        "WaveActiveAllEqual": "subgroupAllEqual",
        "WaveActiveBallot": "subgroupBallot",
        "WaveReadLaneAt": "subgroupShuffle",
        "WaveReadLaneFirst": "subgroupBroadcastFirst",
        "WaveShuffleDown": "subgroupShuffleDown",
        "WaveShuffleUp": "subgroupShuffleUp",
        "WaveShuffleXor": "subgroupShuffleXor",
        "WavePrefixSum": "subgroupExclusiveAdd",
        "WavePrefixProduct": "subgroupExclusiveMul",
        "WavePrefixInclusiveSum": "subgroupInclusiveAdd",
        "WavePrefixInclusiveProduct": "subgroupInclusiveMul",
        "WavePrefixExclusiveMin": "subgroupExclusiveMin",
        "WavePrefixInclusiveMin": "subgroupInclusiveMin",
        "WavePrefixExclusiveMax": "subgroupExclusiveMax",
        "WavePrefixInclusiveMax": "subgroupInclusiveMax",
        "WavePrefixExclusiveBitAnd": "subgroupExclusiveAnd",
        "WavePrefixInclusiveBitAnd": "subgroupInclusiveAnd",
        "WavePrefixExclusiveBitOr": "subgroupExclusiveOr",
        "WavePrefixInclusiveBitOr": "subgroupInclusiveOr",
        "WavePrefixExclusiveBitXor": "subgroupExclusiveXor",
        "WavePrefixInclusiveBitXor": "subgroupInclusiveXor",
        "QuadReadAcrossX": "subgroupQuadSwapHorizontal",
        "QuadReadAcrossY": "subgroupQuadSwapVertical",
        "QuadReadAcrossDiagonal": "subgroupQuadSwapDiagonal",
        "QuadReadLaneAt": "subgroupQuadBroadcast",
        **GLSL_EXTENDED_WAVE_DIRECT_MAPPINGS,
    }
    GLSL_WAVE_BITWISE_VALUE_OPERATIONS = {
        "WaveActiveBitAnd",
        "WaveActiveBitOr",
        "WaveActiveBitXor",
    }
    GLSL_METAL_SIMD_GROUP_HELPER_PREFIX = "__metal_simd"
    GLSL_METAL_SIMD_GROUP_PLAIN_HELPER_OPERATIONS = {
        "simd_sum": "WaveActiveSum",
        "simd_product": "WaveActiveProduct",
        "simd_min": "WaveActiveMin",
        "simd_max": "WaveActiveMax",
        "simd_and": "WaveActiveBitAnd",
        "simd_or": "WaveActiveBitOr",
        "simd_xor": "WaveActiveBitXor",
        "simd_any": "WaveActiveAnyTrue",
        "simd_all": "WaveActiveAllTrue",
        "simd_broadcast": "WaveReadLaneAt",
        "simd_broadcast_first": "WaveReadLaneFirst",
        "simd_prefix_exclusive_sum": "WavePrefixSum",
        "simd_prefix_inclusive_sum": "WavePrefixInclusiveSum",
        "simd_prefix_exclusive_product": "WavePrefixProduct",
        "simd_prefix_inclusive_product": "WavePrefixInclusiveProduct",
        "simd_prefix_exclusive_min": "WavePrefixExclusiveMin",
        "simd_prefix_inclusive_min": "WavePrefixInclusiveMin",
        "simd_prefix_exclusive_max": "WavePrefixExclusiveMax",
        "simd_prefix_inclusive_max": "WavePrefixInclusiveMax",
        "simd_prefix_exclusive_and": "WavePrefixExclusiveBitAnd",
        "simd_prefix_inclusive_and": "WavePrefixInclusiveBitAnd",
        "simd_prefix_exclusive_or": "WavePrefixExclusiveBitOr",
        "simd_prefix_inclusive_or": "WavePrefixInclusiveBitOr",
        "simd_prefix_exclusive_xor": "WavePrefixExclusiveBitXor",
        "simd_prefix_inclusive_xor": "WavePrefixInclusiveBitXor",
    }
    GLSL_METAL_SIMD_GROUP_HELPER_OPERATIONS = {
        "__metal_simd_sum": "WaveActiveSum",
        "__metal_simd_product": "WaveActiveProduct",
        "__metal_simd_min": "WaveActiveMin",
        "__metal_simd_max": "WaveActiveMax",
        "__metal_simd_and": "WaveActiveBitAnd",
        "__metal_simd_or": "WaveActiveBitOr",
        "__metal_simd_xor": "WaveActiveBitXor",
        "__metal_simd_any": "WaveActiveAnyTrue",
        "__metal_simd_all": "WaveActiveAllTrue",
        "__metal_simd_broadcast": "WaveReadLaneAt",
        "__metal_simd_broadcast_first": "WaveReadLaneFirst",
        "__metal_simd_prefix_exclusive_sum": "WavePrefixSum",
        "__metal_simd_prefix_inclusive_sum": "WavePrefixInclusiveSum",
        "__metal_simd_prefix_exclusive_product": "WavePrefixProduct",
        "__metal_simd_prefix_inclusive_product": "WavePrefixInclusiveProduct",
        "__metal_simd_prefix_exclusive_min": "WavePrefixExclusiveMin",
        "__metal_simd_prefix_inclusive_min": "WavePrefixInclusiveMin",
        "__metal_simd_prefix_exclusive_max": "WavePrefixExclusiveMax",
        "__metal_simd_prefix_inclusive_max": "WavePrefixInclusiveMax",
        "__metal_simd_prefix_exclusive_and": "WavePrefixExclusiveBitAnd",
        "__metal_simd_prefix_inclusive_and": "WavePrefixInclusiveBitAnd",
        "__metal_simd_prefix_exclusive_or": "WavePrefixExclusiveBitOr",
        "__metal_simd_prefix_inclusive_or": "WavePrefixInclusiveBitOr",
        "__metal_simd_prefix_exclusive_xor": "WavePrefixExclusiveBitXor",
        "__metal_simd_prefix_inclusive_xor": "WavePrefixInclusiveBitXor",
        **GLSL_METAL_SIMD_GROUP_PLAIN_HELPER_OPERATIONS,
    }
    GLSL_VECTOR_RELATIONAL_FUNCTIONS = {
        "==": "equal",
        "!=": "notEqual",
        "<": "lessThan",
        "<=": "lessThanEqual",
        ">": "greaterThan",
        ">=": "greaterThanEqual",
    }
    GLSL_SUBGROUP_BASIC_BUILTINS = frozenset(
        {
            "gl_NumSubgroups",
            "gl_SubgroupID",
            "gl_SubgroupSize",
            "gl_SubgroupInvocationID",
        }
    )
    GLSL_WAVE_EXTENSION_REQUIREMENTS = {
        "WaveGetLaneCount": "#extension GL_KHR_shader_subgroup_basic : require",
        "WaveGetLaneIndex": "#extension GL_KHR_shader_subgroup_basic : require",
        "WaveIsFirstLane": "#extension GL_KHR_shader_subgroup_basic : require",
        "WaveActiveAllTrue": "#extension GL_KHR_shader_subgroup_vote : require",
        "WaveActiveAnyTrue": "#extension GL_KHR_shader_subgroup_vote : require",
        "WaveActiveAllEqual": "#extension GL_KHR_shader_subgroup_vote : require",
        "WaveActiveSum": "#extension GL_KHR_shader_subgroup_arithmetic : require",
        "WaveActiveProduct": "#extension GL_KHR_shader_subgroup_arithmetic : require",
        "WaveActiveBitAnd": "#extension GL_KHR_shader_subgroup_arithmetic : require",
        "WaveActiveBitOr": "#extension GL_KHR_shader_subgroup_arithmetic : require",
        "WaveActiveBitXor": "#extension GL_KHR_shader_subgroup_arithmetic : require",
        "WaveActiveMin": "#extension GL_KHR_shader_subgroup_arithmetic : require",
        "WaveActiveMax": "#extension GL_KHR_shader_subgroup_arithmetic : require",
        "WavePrefixSum": "#extension GL_KHR_shader_subgroup_arithmetic : require",
        "WavePrefixProduct": "#extension GL_KHR_shader_subgroup_arithmetic : require",
        "WavePrefixInclusiveSum": (
            "#extension GL_KHR_shader_subgroup_arithmetic : require"
        ),
        "WavePrefixInclusiveProduct": (
            "#extension GL_KHR_shader_subgroup_arithmetic : require"
        ),
        "WavePrefixExclusiveMin": (
            "#extension GL_KHR_shader_subgroup_arithmetic : require"
        ),
        "WavePrefixInclusiveMin": (
            "#extension GL_KHR_shader_subgroup_arithmetic : require"
        ),
        "WavePrefixExclusiveMax": (
            "#extension GL_KHR_shader_subgroup_arithmetic : require"
        ),
        "WavePrefixInclusiveMax": (
            "#extension GL_KHR_shader_subgroup_arithmetic : require"
        ),
        "WavePrefixExclusiveBitAnd": (
            "#extension GL_KHR_shader_subgroup_arithmetic : require"
        ),
        "WavePrefixInclusiveBitAnd": (
            "#extension GL_KHR_shader_subgroup_arithmetic : require"
        ),
        "WavePrefixExclusiveBitOr": (
            "#extension GL_KHR_shader_subgroup_arithmetic : require"
        ),
        "WavePrefixInclusiveBitOr": (
            "#extension GL_KHR_shader_subgroup_arithmetic : require"
        ),
        "WavePrefixExclusiveBitXor": (
            "#extension GL_KHR_shader_subgroup_arithmetic : require"
        ),
        "WavePrefixInclusiveBitXor": (
            "#extension GL_KHR_shader_subgroup_arithmetic : require"
        ),
        "WaveActiveBallot": "#extension GL_KHR_shader_subgroup_ballot : require",
        "WaveActiveCountBits": "#extension GL_KHR_shader_subgroup_ballot : require",
        "WavePrefixCountBits": "#extension GL_KHR_shader_subgroup_ballot : require",
        "WaveReadLaneAt": "#extension GL_KHR_shader_subgroup_shuffle : require",
        "WaveShuffleDown": (
            "#extension GL_KHR_shader_subgroup_shuffle_relative : require"
        ),
        "WaveShuffleUp": "#extension GL_KHR_shader_subgroup_shuffle_relative : require",
        "WaveShuffleAndFillUp": "#extension GL_KHR_shader_subgroup_shuffle : require",
        "WaveShuffleXor": "#extension GL_KHR_shader_subgroup_shuffle : require",
        "WaveReadLaneFirst": "#extension GL_KHR_shader_subgroup_ballot : require",
        "QuadActiveSum": "#extension GL_KHR_shader_subgroup_clustered : require",
        "QuadReadAcrossX": "#extension GL_KHR_shader_subgroup_quad : require",
        "QuadReadAcrossY": "#extension GL_KHR_shader_subgroup_quad : require",
        "QuadReadAcrossDiagonal": "#extension GL_KHR_shader_subgroup_quad : require",
        "QuadReadLaneAt": "#extension GL_KHR_shader_subgroup_quad : require",
        "WaveMatch": "#extension GL_KHR_shader_subgroup_ballot : require",
        "WaveMultiPrefixSum": "#extension GL_KHR_shader_subgroup_ballot : require",
        "WaveMultiPrefixCountBits": (
            "#extension GL_KHR_shader_subgroup_ballot : require"
        ),
        "WaveMultiPrefixProduct": "#extension GL_KHR_shader_subgroup_ballot : require",
        "WaveMultiPrefixBitAnd": "#extension GL_KHR_shader_subgroup_ballot : require",
        "WaveMultiPrefixBitOr": "#extension GL_KHR_shader_subgroup_ballot : require",
        "WaveMultiPrefixBitXor": "#extension GL_KHR_shader_subgroup_ballot : require",
        **GLSL_EXTENDED_WAVE_EXTENSIONS,
    }

    GLSL_WAVE_MULTI_PREFIX_NUMERIC_OPERATIONS = {
        "WaveMultiPrefixSum",
        "WaveMultiPrefixProduct",
    }
    GLSL_WAVE_MULTI_PREFIX_INTEGER_OPERATIONS = {
        "WaveMultiPrefixBitAnd",
        "WaveMultiPrefixBitOr",
        "WaveMultiPrefixBitXor",
    }
    GLSL_WAVE_MULTI_PREFIX_OPERATIONS = {
        *GLSL_WAVE_MULTI_PREFIX_NUMERIC_OPERATIONS,
        "WaveMultiPrefixCountBits",
        *GLSL_WAVE_MULTI_PREFIX_INTEGER_OPERATIONS,
    }
    GLSL_WAVE_MULTI_PREFIX_HELPERS = {
        "WaveMultiPrefixSum": "crossglWaveMultiPrefixSum",
        "WaveMultiPrefixCountBits": "crossglWaveMultiPrefixCountBits",
        "WaveMultiPrefixProduct": "crossglWaveMultiPrefixProduct",
        "WaveMultiPrefixBitAnd": "crossglWaveMultiPrefixBitAnd",
        "WaveMultiPrefixBitOr": "crossglWaveMultiPrefixBitOr",
        "WaveMultiPrefixBitXor": "crossglWaveMultiPrefixBitXor",
    }
    GLSL_WAVE_NUMERIC_OPERATIONS = {
        "QuadActiveSum",
        "WaveActiveSum",
        "WaveActiveProduct",
        "WaveActiveMin",
        "WaveActiveMax",
        "WavePrefixSum",
        "WavePrefixProduct",
        "WavePrefixInclusiveSum",
        "WavePrefixInclusiveProduct",
        "WavePrefixExclusiveMin",
        "WavePrefixInclusiveMin",
        "WavePrefixExclusiveMax",
        "WavePrefixInclusiveMax",
        *GLSL_CLUSTERED_NUMERIC_OPERATIONS,
    }
    GLSL_WAVE_INTEGER_OR_BOOLEAN_OPERATIONS = {
        "WaveActiveBitAnd",
        "WaveActiveBitOr",
        "WaveActiveBitXor",
        "WavePrefixExclusiveBitAnd",
        "WavePrefixInclusiveBitAnd",
        "WavePrefixExclusiveBitOr",
        "WavePrefixInclusiveBitOr",
        "WavePrefixExclusiveBitXor",
        "WavePrefixInclusiveBitXor",
        *GLSL_CLUSTERED_BITWISE_OPERATIONS,
    }
    GLSL_WAVE_BOOLEAN_OPERATIONS = {
        "WaveActiveAllTrue",
        "WaveActiveAnyTrue",
        "WaveActiveBallot",
        "WaveActiveCountBits",
        "WavePrefixCountBits",
        *GLSL_QUAD_VOTE_FUNCTIONS,
    }
    GLSL_WAVE_SCALAR_OR_VECTOR_OPERATIONS = {
        "WaveActiveAllEqual",
    }
    GLSL_WAVE_LANE_INDEX_ARGUMENTS = {
        "WaveReadLaneAt": 1,
        "QuadReadLaneAt": 1,
    }
    GLSL_TESSELLATION_FACTOR_BUILTINS = {
        "gl_TessLevelOuter",
        "gl_TessLevelInner",
    }
    GLSL_TESSELLATION_FACTOR_COMPONENT_COUNTS = {
        "gl_TessLevelOuter": 4,
        "gl_TessLevelInner": 2,
    }
    GLSL_TESSELLATION_CONTROL_BARRIER_BUILTINS = {
        "barrier",
        "workgroupBarrier",
        "workgroupExecutionBarrier",
    }
    GLSL_MESH_PRIMITIVE_INDEX_TYPES = {
        "points": "uint",
        "lines": "uvec2",
        "triangles": "uvec3",
    }
    GLSL_MEMORY_ATOMIC_FUNCTIONS = {
        "atomicCompareExchangeWeak",
        "atomicLoad",
        "atomicStore",
        "atomicAdd",
        "atomicMin",
        "atomicMax",
        "atomicAnd",
        "atomicOr",
        "atomicXor",
        "atomicExchange",
        "atomicCompSwap",
    }
    GLSL_ATOMIC_FENCE_MEMORY_FLAGS = frozenset(
        {
            "mem_none",
            "mem_device",
            "mem_threadgroup",
            "mem_texture",
            "mem_threadgroup_imageblock",
            "mem_object_data",
        }
    )
    GLSL_ATOMIC_FENCE_MEMORY_ORDERS = frozenset(
        {
            "memory_order_relaxed",
            "memory_order_acquire",
            "memory_order_release",
            "memory_order_acq_rel",
            "memory_order_seq_cst",
        }
    )
    GLSL_ATOMIC_FENCE_THREAD_SCOPES = frozenset(
        {
            "thread_scope_thread",
            "thread_scope_simdgroup",
            "thread_scope_threadgroup",
            "thread_scope_device",
            "thread_scope_system",
        }
    )
    GLSL_THREADS_PER_GRID_EXPRESSION = "(gl_NumWorkGroups * gl_WorkGroupSize)"
    GLSL_COOPERATIVE_MATRIX_COMPONENT_TYPES = {
        "float": "float",
        "f32": "float",
        "double": "double",
        "f64": "double",
        "int": "int",
        "i32": "int",
        "int32": "int",
        "int32_t": "int",
        "uint": "uint",
        "u32": "uint",
        "uint32": "uint",
        "uint32_t": "uint",
        "i64": "int64_t",
        "int64": "int64_t",
        "int64_t": "int64_t",
        "u64": "uint64_t",
        "uint64": "uint64_t",
        "uint64_t": "uint64_t",
    }

    def set_cooperative_matrix_software_lowering(self, enabled):
        """Enable exact lane-local cooperative-matrix fragment lowering."""
        if not isinstance(enabled, bool):
            raise TranslatorArgumentError(
                "cooperative_matrix_software_lowering must be a boolean"
            )
        self.cooperative_matrix_software_lowering = enabled

    def set_private_pointer_out_of_bounds_read(self, policy):
        """Configure explicit recovery for statically bounded private reads."""
        if not isinstance(policy, str) or policy not in {"error", "zero"}:
            raise OpenGLRefusal(
                "private_pointer_out_of_bounds_read must be 'error' or 'zero'"
            )
        self.private_pointer_out_of_bounds_read = policy

    def set_cooperative_matrix_mma_profile(self, profile):
        """Select explicit binary32 rounding for cooperative matrix products."""
        if profile is not None and (
            not isinstance(profile, str)
            or profile not in {
                "rne-gradual",
                "rne-flush",
                "fma-rne-gradual",
                "fma-rne-flush",
            }
        ):
            raise OpenGLRefusal(
                "cooperative_matrix_mma_profile must be 'rne-gradual', 'rne-flush', "
                "'fma-rne-gradual', 'fma-rne-flush', or None"
            )
        self.cooperative_matrix_mma_profile = profile

    GLSL_SHADER_CLOCK_EXTENSIONS = {
        "clockARB": "GL_ARB_shader_clock",
        "clock2x32ARB": "GL_ARB_shader_clock",
        "clockRealtimeEXT": "GL_EXT_shader_realtime_clock",
        "clockRealtime2x32EXT": "GL_EXT_shader_realtime_clock",
    }

    SOFTWARE_SUBGROUP_SYSTEM_INPUTS = frozenset(
        {
            "gl_SubgroupInvocationID",
            "gl_SubgroupSize",
            "gl_SubgroupID",
            "gl_NumSubgroups",
        }
    )

    GLSL_SOFTWARE_SUBGROUP_UNIFORM_GUARD_OPERATORS = {
        "==",
        "!=",
        "<",
        "<=",
        ">",
        ">=",
        "&&",
        "||",
        "!",
        "+",
        "-",
        "*",
        "/",
        "%",
        "&",
        "|",
        "^",
        "<<",
        ">>",
    }

    GLSL_VERSION_RESOLUTION_PASSES = 3

    GLSL_EARLY_EXIT_RETRY_REASONS = frozenset(
        {"early-return-unproven", "potentially-divergent-control-flow"}
    )

    # Arrayed interfaces whose outer dimension indexes vertices, not locations (GLSL 4.60 section 4.4.1):
    # geometry inputs, tessellation inputs and non-patch control outputs, mesh outputs, pervertex inputs.
    PER_VERTEX_STAGE_IO = {
        ("geometry", "input"),
        ("tessellation_control", "input"),
        ("tessellation_control", "output"),
        ("tessellation_evaluation", "input"),
        ("mesh", "output"),
    }

    # Lane-local transfers address each element through the registered
    # fragment mapping; this closed form must reproduce it exactly.
    GLSL_COOPERATIVE_MATRIX_TRANSFER_MAPPINGS = {
        "tile_4x4_row_pair": (
            "((lane >> 2u) & 4u) + ((lane >> 1u) & 3u)",
            "((lane >> 2u) & 2u) * 2u + (lane & 1u) * 2u + element",
        )
    }

    # GLSL's sinh, tanh, asinh and atanh are defined through exp and log, and
    # Mesa evaluates them that way: the sign of zero is lost, tanh clamps a
    # NaN operand to -1 and the formulas cancel near zero.  Metal's precise::
    # implementations keep all three; glsl_precise_odd describes the helpers.
    GLSL_PRECISE_ODD_FUNCTIONS = frozenset({"asinh", "atanh", "sinh", "tanh"})

    # Mesa folds a product with a constant zero factor to +0, 0.0 / x to +0,
    # x + 0.0 to x and 0.0 - x to -x, also when the operation is precise.
    # IEEE 754 keeps signed zeros and gives NaN for 0 * inf and 0 / 0, and
    # Metal's -fno-fast-math build computes MLX's complex operators that way,
    # also for ArcTan's i * x and (1 / 2i) * log.  Integer tests of the
    # operand bits survive constant folding, so these helpers settle zero
    # operands themselves and leave every other operand to the operation.
    GLSL_IEEE_OPERATOR_HELPERS = {
        "+": "ieee-sum",
        "-": "ieee-difference",
        "*": "ieee-product",
        "/": "ieee-quotient",
    }

    # HLSL pack_u8/pack_s8 keep the low byte of each lane, pack_clamp_u8 and
    # pack_clamp_s8 saturate each lane to [0, 255] or [-128, 127], and all four
    # put lane x in the least significant byte. packUnorm4x8 rounds each
    # clamped lane of v * 255.0 into the same byte order, and k / 255.0 * 255.0
    # rounds back to k for every byte k, so it packs the bytes exactly while
    # evaluating the argument once. The signed clamp packs a + 128 and flips
    # the top bit of every byte back to two's complement.
    # unpack_u8u32 and unpack_s8s32 split the bytes into a four-lane vector,
    # sign-extending a signed lane by shifting it into the top byte and
    # arithmetically shifting it back down.
    PACKED_BYTE_INTRINSICS = {
        "pack_u8": "packUnorm4x8(vec4(uvec4({value}) & uvec4(255u)) / 255.0)",
        "pack_s8": "packUnorm4x8(vec4(uvec4({value}) & uvec4(255u)) / 255.0)",
        "pack_clamp_u8": "packUnorm4x8(vec4({value}) / 255.0)",
        "pack_clamp_s8": (
            "(packUnorm4x8((vec4({value}) + 128.0) / 255.0) ^ 0x80808080u)"
        ),
        "unpack_u8u32": (
            "((uvec4(uint({value})) >> uvec4(0u, 8u, 16u, 24u)) & uvec4(255u))"
        ),
        "unpack_s8s32": "((ivec4(int({value})) << ivec4(24, 16, 8, 0)) >> ivec4(24))",
    }

    # HLSL dot4add_i8packed(a, b, acc) and dot4add_u8packed(a, b, acc) add the
    # dot product of the four 8-bit lanes packed in a and b to acc, with byte 0
    # the least significant lane. GLSL has no packed integer dot product, so
    # each operand is split into a four-lane vector once and the lanes go
    # through dot(). Every lane product and their sum stay below 2^24 in
    # magnitude, so the float dot product is exact. A signed lane is
    # sign-extended by shifting it into the top byte and arithmetically
    # shifting it back down.
    PACKED_DOT4ADD_LANES = {
        "dot4add_i8packed": (
            "int",
            "((ivec4(int({value})) << ivec4(24, 16, 8, 0)) >> ivec4(24))",
        ),
        "dot4add_u8packed": (
            "uint",
            "((uvec4(uint({value})) >> uvec4(0u, 8u, 16u, 24u)) & uvec4(255u))",
        ),
    }

    # GL_EXT_texture_shadow_lod (glsl_shadow_lod); WebGL 2 has no such extension.
    GLSL_SHADOW_LOD_EXTENSION_SUPPORTED = True

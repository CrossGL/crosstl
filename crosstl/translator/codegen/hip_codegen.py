"""
CrossGL to HIP Code Generator

This module provides code generation functionality to convert CrossGL AST to HIP source code.
HIP (Heterogeneous-Compute Interface for Portability) is AMD's CUDA-compatible runtime API
for GPU programming.
"""

from ..ast import (
    ASTNode,
    ConstructorNode,
    FunctionCallNode,
    FunctionNode,
    IdentifierNode,
    MatchNode,
    MemberAccessNode,
    PointerAccessNode,
    RayQueryOpNode,
    RayTracingOpNode,
    ShaderNode,
    VariableNode,
    WaveOpNode,
)
from ..refusals import HipRefusal
from ..semantics.declared_types import declared_type_names
from ..semantics.structured_buffer_counters import (
    materialize_structured_buffer_counters,
)
from .barrier_peephole import merge_adjacent_cuda_barriers
from .c_reference_parameters import is_output_parameter, reference_declarator
from .comma_expressions import (
    CommaExpressionVisitorMixin,
    comma_result_operand,
    is_comma_expression,
)
from .cooperative_matrix_refusals import refuse_cooperative_matrices
from .cuda_library_functions import rename_cuda_library_functions
from .enum_utils import infer_enum_constructor_type
from .float_remainder_assignment import float_remainder_assignment
from .generic_function_utils import (
    generic_function_call_name,
    iter_function_nodes,
    numeric_trait_method_result_type,
    prepare_generic_function_specializations,
)
from .generic_global_types import collect_constant_and_cbuffer_member_types
from .generic_struct_utils import infer_struct_constructor_type
from .hip_bindings import HipBindingMixin
from .hip_buffer_calls import HipBufferCallMixin
from .hip_buffer_metadata import HipBufferMetadataMixin
from .hip_declarations import HipDeclarationMixin
from .hip_expression_types import HipExpressionTypeMixin
from .hip_expressions import HipExpressionMixin
from .hip_function_calls import HipFunctionCallMixin
from .hip_math_calls import HipMathCallMixin
from .hip_refusals import HipLoweringRefused
from .hip_semantics import HipSemanticMixin
from .hip_stage_attributes import HipStageAttributeMixin
from .hip_stage_checks import HipStageCheckMixin
from .hip_statements import HipStatementMixin
from .hip_tables import (
    HIP_64BIT_BITCAST_FUNCTIONS,
    HIP_BITCAST_FUNCTION_TARGETS,
    HIP_FP16_VECTOR_TYPES,
    HIP_INTEGER_BIT_FUNCTION_ALIASES,
    HIP_SCALAR_CONSTRUCTOR_TYPE_ALIASES,
    HIP_WARP_SYNC_BUILTIN_ARITIES,
    HIP_WAVE_BOOL_RESULT_OPS,
    HIP_WAVE_OP_ARITIES,
    HIP_WAVE_PREDICATE_ARGUMENT_OPS,
    HIP_WAVE_UINT_RESULT_OPS,
    HIP_WAVE_UVEC4_RESULT_OPS,
)
from .hip_textures import HipTextureMixin
from .hip_types import HipTypeMixin
from .hip_vector_types import HipVectorTypeMixin
from .hip_visitor import HipVisitorMixin
from .hip_wave_ops import HipWaveOpMixin
from .lowering_refusals import LoweringRefusalMixin
from .match_utils import infer_match_expression_result_type
from .precise_evaluation import PreciseEvaluationMixin
from .relational_builtins import lower_relational_builtins
from .resource_diagnostics import ResourceDiagnosticMixin
from .resource_query import ResourceQueryMixin
from .sampler_pairs import canonicalize_separate_textures
from .stage_utils import is_fragment_output_parameter, normalize_stage_name
from .value_array_parameters import (
    collect_function_parameters_by_name,
    collect_struct_names,
    fixed_array_dimensions,
    plan_value_array_copies,
    value_array_copy_lines,
)
from .vector_arithmetic import VectorArithmeticMixin
from .written_function_names import (
    CUDA_CALLABLE_FUNCTION_NAMES,
    refuse_written_foreign_calls,
)
from .written_type_names import refuse_written_foreign_types

# Most generator members live in the hip_*.py mixin modules next to this one. These names stay importable from here.
__all__ = [
    "HIP_64BIT_BITCAST_FUNCTIONS",
    "HIP_BITCAST_FUNCTION_TARGETS",
    "HIP_FP16_VECTOR_TYPES",
    "HIP_INTEGER_BIT_FUNCTION_ALIASES",
    "HIP_SCALAR_CONSTRUCTOR_TYPE_ALIASES",
    "HIP_WARP_SYNC_BUILTIN_ARITIES",
    "HIP_WAVE_BOOL_RESULT_OPS",
    "HIP_WAVE_OP_ARITIES",
    "HIP_WAVE_PREDICATE_ARGUMENT_OPS",
    "HIP_WAVE_UINT_RESULT_OPS",
    "HIP_WAVE_UVEC4_RESULT_OPS",
    "HipCodeGen",
    "UnsupportedHIPAstNodeError",
    "generate_hip_code",
]


def _refuse_buffer_counter(reason):
    raise HipRefusal(f"HIP buffer counter: {reason}")


class UnsupportedHIPAstNodeError(ValueError):
    """Raised when HIP codegen cannot safely lower an AST node."""

    project_diagnostic_code = "project.translate.unsupported-feature"
    missing_capabilities = ("hip.ast-node-lowering",)

    def __init__(self, node):
        self.node_type = type(node).__name__
        self.feature = f"hip.ast-node.{self.node_type}"
        self.source_location = getattr(node, "source_location", None)
        self.suggested_action = (
            f"Implement visit_{self.node_type} with semantics-preserving HIP lowering."
        )
        super().__init__(
            f"HIP code generation does not support AST node {self.node_type}. "
            f"Suggested action: {self.suggested_action}"
        )


class HipCodeGen(
    HipStageCheckMixin,
    HipVisitorMixin,
    HipSemanticMixin,
    HipStageAttributeMixin,
    HipDeclarationMixin,
    HipStatementMixin,
    HipExpressionMixin,
    HipFunctionCallMixin,
    HipWaveOpMixin,
    HipMathCallMixin,
    HipVectorTypeMixin,
    HipTextureMixin,
    HipBufferCallMixin,
    HipBufferMetadataMixin,
    HipTypeMixin,
    HipBindingMixin,
    HipExpressionTypeMixin,
    LoweringRefusalMixin,
    PreciseEvaluationMixin,
    CommaExpressionVisitorMixin,
    VectorArithmeticMixin,
    ResourceQueryMixin,
    ResourceDiagnosticMixin,
):
    """Emit HIP source from the shared CrossGL translator AST."""

    LOWERING_REFUSAL_TYPE = HipLoweringRefused
    # Clang disables contraction for the compound statement the pragma starts; see precise_evaluation.
    PRECISE_BODY_PRAGMA = "#pragma clang fp contract(off)"
    PRECISE_TARGET_LABEL = "HIP"

    resource_diagnostic_backend = "HIP"
    struct_constructor_uses_braces = True
    synchronization_builtins = {
        "barrier",
        "groupMemoryBarrier",
        "memoryBarrier",
        "memoryBarrierShared",
        "memoryBarrierBuffer",
        "memoryBarrierImage",
        "allMemoryBarrier",
        "deviceMemoryBarrier",
        "workgroupBarrier",
        "workgroupExecutionBarrier",
    }
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
        "SamplerComparisonState": "sampler",
    }
    # The type names the program declares, and the foreign built-in type names written so far (``None`` outside
    # a rendering); see :func:`written_type_name`.
    program_type_names = frozenset()
    written_foreign_type_names = None

    def __init__(self):
        """Initialize HIP type maps and per-generation visitor state."""
        self.indent_level = 0
        self.code_lines = []
        self.current_function = None
        self.current_function_return_type = None
        self.variable_counter = 0
        self.match_temp_variable_index = 0
        self.variable_types = {}
        self.image_resource_accesses = {}
        self.buffer_resource_accesses = {}
        self.glsl_buffer_block_accesses = {}
        self.glsl_buffer_block_layouts = {}
        self.hip_resource_binding_cursors = {}
        self.hip_used_resource_bindings = {}
        self.struct_member_types = {}
        self.struct_member_semantics = {}
        self.function_return_types = {}
        self.function_parameters_by_name = {}
        self.value_array_struct_names = frozenset()
        self.helper_functions = {}
        self.query_resource_names = set()
        self.query_metadata_function_params = {}
        self.query_functions_by_name = {}
        self.structured_buffer_length_names = set()
        self.structured_buffer_length_function_params = {}
        self.current_structured_buffer_length_parameters = {}
        self.current_function_name = None
        self.current_stage_name = None
        self.hip_function_capture_params = {}
        self.resource_query_info_required = False
        self.assignment_lhs_depth = 0
        # Subscripts visited as the index of another subscript: reads, even on an assignment's left side.
        self.subscript_index_depth = 0
        self.stage_builtin_aliases = {}
        self.stage_builtin_alias_types = {}
        self.current_function_is_kernel_entry = False
        self.current_expression_expected_type = None
        self.structs_by_name = {}
        self.generic_enum_struct_definitions = {}
        self.generic_enum_specializations = {}
        self.generic_struct_definitions = {}
        self.generic_struct_specializations = {}
        self.plain_enums = []
        self.struct_payload_enums = []
        self.enum_type_names = set()
        self.enum_struct_type_names = set()
        self.enum_struct_variant_fields = {}
        self.enum_variant_constructors = {}
        self.enum_variant_constructor_fields = {}
        self.enum_variant_constants = {}

        # CrossGL to HIP type mapping
        self.type_map = {
            # Basic types
            "int": "int",
            "float": "float",
            "double": "double",
            "bool": "bool",
            "void": "void",
            "i8": "char",
            "u8": "unsigned char",
            "i16": "short",
            "u16": "unsigned short",
            "i32": "int",
            "u32": "unsigned int",
            "i64": "long long",
            "u64": "unsigned long long",
            "int64": "long long",
            "uint64": "unsigned long long",
            # The other sized and minimum-precision scalar spellings of the IR (hub_types): exact widths, except
            # that a minimum-precision type may be computed at full precision (HLSL "Minimum precision").
            "int8": "char",
            "uint8": "unsigned char",
            "uchar": "unsigned char",
            "int16": "short",
            "uint16": "unsigned short",
            "int32": "int",
            "uint32": "unsigned int",
            "ulong": "unsigned long long",
            "float16_t": "half",
            "min16float": "float",
            "min16int": "int",
            "min16uint": "unsigned int",
            "f16": "half",
            "f32": "float",
            "f64": "double",
            "uint": "unsigned int",
            "float16": "half",
            "half": "half",
            "str": "int",
            # Vector types
            "vec2": "float2",
            "vec3": "float3",
            "vec4": "float4",
            "vec2<f16>": "half2",
            "vec2<i8>": "char2",
            "vec3<i8>": "char3",
            "vec4<i8>": "char4",
            "vec2<u8>": "uchar2",
            "vec3<u8>": "uchar3",
            "vec4<u8>": "uchar4",
            "vec2<i16>": "short2",
            "vec3<i16>": "short3",
            "vec4<i16>": "short4",
            "vec2<u16>": "ushort2",
            "vec3<u16>": "ushort3",
            "vec4<u16>": "ushort4",
            "vec2<f32>": "float2",
            "vec3<f32>": "float3",
            "vec4<f32>": "float4",
            "ivec2": "int2",
            "ivec3": "int3",
            "ivec4": "int4",
            "vec2<i32>": "int2",
            "vec3<i32>": "int3",
            "vec4<i32>": "int4",
            "uvec2": "uint2",
            "uvec3": "uint3",
            "uvec4": "uint4",
            "vec2<u32>": "uint2",
            "vec3<u32>": "uint3",
            "vec4<u32>": "uint4",
            "dvec2": "double2",
            "dvec3": "double3",
            "dvec4": "double4",
            "vec2<f64>": "double2",
            "vec3<f64>": "double3",
            "vec4<f64>": "double4",
            "f16vec2": "half2",
            "f16vec3": "cgl_half3",
            "f16vec4": "cgl_half4",
            "float16vec3": "cgl_half3",
            "float16vec4": "cgl_half4",
            "half3": "cgl_half3",
            "half4": "cgl_half4",
            "half2": "half2",
            "vec3<f16>": "cgl_half3",
            "vec3<float16>": "cgl_half3",
            "vec3<half>": "cgl_half3",
            "vec4<f16>": "cgl_half4",
            "vec4<float16>": "cgl_half4",
            "vec4<half>": "cgl_half4",
            "bvec2": "uchar2",
            "bvec3": "uchar3",
            "bvec4": "uchar4",
            "vec2<bool>": "uchar2",
            "vec3<bool>": "uchar3",
            "vec4<bool>": "uchar4",
            "bool2": "uchar2",
            "bool3": "uchar3",
            "bool4": "uchar4",
            # Matrix types
            "mat2": "float2x2",
            "mat3": "float3x3",
            "mat4": "float4x4",
            "mat2x2": "float2x2",
            "mat2x3": "float2x3",
            "mat2x4": "float2x4",
            "mat3x2": "float3x2",
            "mat3x3": "float3x3",
            "mat3x4": "float3x4",
            "mat4x2": "float4x2",
            "mat4x3": "float4x3",
            "mat4x4": "float4x4",
            "dmat2": "double2x2",
            "dmat3": "double3x3",
            "dmat4": "double4x4",
            "dmat2x2": "double2x2",
            "dmat2x3": "double2x3",
            "dmat2x4": "double2x4",
            "dmat3x2": "double3x2",
            "dmat3x3": "double3x3",
            "dmat3x4": "double3x4",
            "dmat4x2": "double4x2",
            "dmat4x3": "double4x3",
            "dmat4x4": "double4x4",
            # Texture/resource types
            "sampler": "hipTextureObject_t",
            "sampler1D": "hipTextureObject_t",
            "sampler1DArray": "hipTextureObject_t",
            "sampler2D": "hipTextureObject_t",
            "sampler3D": "hipTextureObject_t",
            "samplerCube": "hipTextureObject_t",
            "sampler2DArray": "hipTextureObject_t",
            "sampler2DShadow": "hipTextureObject_t",
            "sampler2DArrayShadow": "hipTextureObject_t",
            "samplerCubeShadow": "hipTextureObject_t",
            "samplerCubeArray": "hipTextureObject_t",
            "samplerCubeArrayShadow": "hipTextureObject_t",
            "sampler2DMS": "hipTextureObject_t",
            "sampler2DMSArray": "hipTextureObject_t",
            "isampler1D": "hipTextureObject_t",
            "isampler1DArray": "hipTextureObject_t",
            "isampler2D": "hipTextureObject_t",
            "isampler3D": "hipTextureObject_t",
            "isamplerCube": "hipTextureObject_t",
            "isampler2DArray": "hipTextureObject_t",
            "isamplerCubeArray": "hipTextureObject_t",
            "isampler2DMS": "hipTextureObject_t",
            "isampler2DMSArray": "hipTextureObject_t",
            "usampler1D": "hipTextureObject_t",
            "usampler1DArray": "hipTextureObject_t",
            "usampler2D": "hipTextureObject_t",
            "usampler3D": "hipTextureObject_t",
            "usamplerCube": "hipTextureObject_t",
            "usampler2DArray": "hipTextureObject_t",
            "usamplerCubeArray": "hipTextureObject_t",
            "usampler2DMS": "hipTextureObject_t",
            "usampler2DMSArray": "hipTextureObject_t",
            "image1D": "hipSurfaceObject_t",
            "image1DArray": "hipSurfaceObject_t",
            "image2D": "hipSurfaceObject_t",
            "image3D": "hipSurfaceObject_t",
            "imageCube": "hipSurfaceObject_t",
            "imageCubeArray": "hipSurfaceObject_t",
            "image2DArray": "hipSurfaceObject_t",
            "image2DMS": "hipSurfaceObject_t",
            "image2DMSArray": "hipSurfaceObject_t",
            "iimage1D": "hipSurfaceObject_t",
            "iimage1DArray": "hipSurfaceObject_t",
            "iimage2D": "hipSurfaceObject_t",
            "iimage3D": "hipSurfaceObject_t",
            "iimageCube": "hipSurfaceObject_t",
            "iimageCubeArray": "hipSurfaceObject_t",
            "iimage2DArray": "hipSurfaceObject_t",
            "iimage2DMS": "hipSurfaceObject_t",
            "iimage2DMSArray": "hipSurfaceObject_t",
            "uimage1D": "hipSurfaceObject_t",
            "uimage1DArray": "hipSurfaceObject_t",
            "uimage2D": "hipSurfaceObject_t",
            "uimage3D": "hipSurfaceObject_t",
            "uimageCube": "hipSurfaceObject_t",
            "uimageCubeArray": "hipSurfaceObject_t",
            "uimage2DArray": "hipSurfaceObject_t",
            "uimage2DMS": "hipSurfaceObject_t",
            "uimage2DMSArray": "hipSurfaceObject_t",
            "buffer": "hipDeviceptr_t",
            "accelerationStructureEXT": "CglRayTracingAccelerationStructure",
            "AccelerationStructure": "CglRayTracingAccelerationStructure",
            "acceleration_structure": "CglRayTracingAccelerationStructure",
            "RaytracingAccelerationStructure": "CglRayTracingAccelerationStructure",
            "RayTracingAccelerationStructure": "CglRayTracingAccelerationStructure",
            "RayDesc": "CglRayDesc",
            "RayQuery": "CglRayQuery",
            "BuiltInTriangleIntersectionAttributes": (
                "CglBuiltInTriangleIntersectionAttributes"
            ),
        }

        # CrossGL to HIP function mapping
        self.function_map = {
            # Math functions
            "sin": "sinf",
            "cos": "cosf",
            "tan": "tanf",
            "asin": "asinf",
            "acos": "acosf",
            "atan": "atanf",
            "atan2": "atan2f",
            "sinh": "sinhf",
            "cosh": "coshf",
            "tanh": "tanhf",
            "exp": "expf",
            "exp2": "exp2f",
            "log": "logf",
            "log2": "log2f",
            "sqrt": "sqrtf",
            "inversesqrt": "rsqrtf",
            "inverseSqrt": "rsqrtf",
            "rsqrt": "rsqrtf",
            "pow": "powf",
            "fma": "fmaf",
            "mad": "fmaf",
            "abs": "fabsf",
            "floor": "floorf",
            "ceil": "ceilf",
            "round": "roundf",
            "roundEven": "rintf",
            "trunc": "truncf",
            "mod": "fmodf",
            "min": "fminf",
            "max": "fmaxf",
            "clamp": "fmaxf(fminf",  # Special handling needed
            "step": "step",
            "smoothstep": "smoothstep",
            # Vector functions
            "length": "length",
            "distance": "distance",
            "dot": "dot",
            "cross": "cross",
            "normalize": "normalize",
            "reflect": "reflect",
            "refract": "refract",
            # Geometric functions
            "faceforward": "faceforward",
            # Vector constructors
            "vec2": "make_float2",
            "vec3": "make_float3",
            "vec4": "make_float4",
            "float2": "make_float2",
            "float3": "make_float3",
            "float4": "make_float4",
            "f16vec3": "cgl_make_half3",
            "float16vec3": "cgl_make_half3",
            "half3": "cgl_make_half3",
            "vec3<f16>": "cgl_make_half3",
            "vec3<float16>": "cgl_make_half3",
            "vec3<half>": "cgl_make_half3",
            "f16vec4": "cgl_make_half4",
            "float16vec4": "cgl_make_half4",
            "half4": "cgl_make_half4",
            "vec4<f16>": "cgl_make_half4",
            "vec4<float16>": "cgl_make_half4",
            "vec4<half>": "cgl_make_half4",
            "vec2<f32>": "make_float2",
            "vec3<f32>": "make_float3",
            "vec4<f32>": "make_float4",
            "vec2<i8>": "make_char2",
            "vec3<i8>": "make_char3",
            "vec4<i8>": "make_char4",
            "vec2<u8>": "make_uchar2",
            "vec3<u8>": "make_uchar3",
            "vec4<u8>": "make_uchar4",
            "vec2<i16>": "make_short2",
            "vec3<i16>": "make_short3",
            "vec4<i16>": "make_short4",
            "vec2<u16>": "make_ushort2",
            "vec3<u16>": "make_ushort3",
            "vec4<u16>": "make_ushort4",
            "ivec2": "make_int2",
            "ivec3": "make_int3",
            "ivec4": "make_int4",
            "int2": "make_int2",
            "int3": "make_int3",
            "int4": "make_int4",
            "vec2<i32>": "make_int2",
            "vec3<i32>": "make_int3",
            "vec4<i32>": "make_int4",
            "uvec2": "make_uint2",
            "uvec3": "make_uint3",
            "uvec4": "make_uint4",
            "uint2": "make_uint2",
            "uint3": "make_uint3",
            "uint4": "make_uint4",
            "vec2<u32>": "make_uint2",
            "vec3<u32>": "make_uint3",
            "vec4<u32>": "make_uint4",
            "dvec2": "make_double2",
            "dvec3": "make_double3",
            "dvec4": "make_double4",
            "double2": "make_double2",
            "double3": "make_double3",
            "double4": "make_double4",
            "vec2<f64>": "make_double2",
            "vec3<f64>": "make_double3",
            "vec4<f64>": "make_double4",
            "bvec2": "make_uchar2",
            "bvec3": "make_uchar3",
            "bvec4": "make_uchar4",
            "uchar2": "make_uchar2",
            "uchar3": "make_uchar3",
            "uchar4": "make_uchar4",
            "vec2<bool>": "make_uchar2",
            "vec3<bool>": "make_uchar3",
            "vec4<bool>": "make_uchar4",
            "bool2": "make_uchar2",
            "bool3": "make_uchar3",
            "bool4": "make_uchar4",
            # Matrix constructors
            "mat2": "float2x2",
            "mat3": "float3x3",
            "mat4": "float4x4",
            "mat2x2": "float2x2",
            "mat2x3": "float2x3",
            "mat2x4": "float2x4",
            "mat3x2": "float3x2",
            "mat3x3": "float3x3",
            "mat3x4": "float3x4",
            "mat4x2": "float4x2",
            "mat4x3": "float4x3",
            "mat4x4": "float4x4",
            "dmat2": "double2x2",
            "dmat3": "double3x3",
            "dmat4": "double4x4",
            "dmat2x2": "double2x2",
            "dmat2x3": "double2x3",
            "dmat2x4": "double2x4",
            "dmat3x2": "double3x2",
            "dmat3x3": "double3x3",
            "dmat3x4": "double3x4",
            "dmat4x2": "double4x2",
            "dmat4x3": "double4x3",
            "dmat4x4": "double4x4",
            # Atomic operations
            "atomicAdd": "atomicAdd",
            "atomicSub": "atomicSub",
            "atomicMin": "atomicMin",
            "atomicMax": "atomicMax",
            "atomicAnd": "atomicAnd",
            "atomicOr": "atomicOr",
            "atomicXor": "atomicXor",
            "atomicExchange": "atomicExch",
            "atomicCompareExchange": "atomicCAS",
            "atomicCompSwap": "atomicCAS",
            # Synchronization
            "barrier": "__syncthreads",
            # GL_EXT_shader_realtime_clock's device-scope counter is HIP's constant-rate wall clock.
            "clockRealtimeEXT": "wall_clock64",
            "groupMemoryBarrier": "__threadfence_block",
            "memoryBarrier": "__threadfence",
            "memoryBarrierShared": "__threadfence_block",
            "memoryBarrierBuffer": "__threadfence",
            "memoryBarrierImage": "__threadfence",
            "allMemoryBarrier": "__threadfence",
            "deviceMemoryBarrier": "__threadfence",
            "workgroupBarrier": "__syncthreads",
            "workgroupExecutionBarrier": "__syncthreads",
            # Ray tracing placeholders
            "RayDesc": "CglRayDesc",
            "RayQuery": "CglRayQuery",
            "BuiltInTriangleIntersectionAttributes": (
                "CglBuiltInTriangleIntersectionAttributes"
            ),
            "TraceRay": "cgl_trace_ray",
            "CallShader": "cgl_call_shader",
            "ReportHit": "cgl_report_hit",
            "IgnoreHit": "cgl_ignore_hit",
            "AcceptHitAndEndSearch": "cgl_accept_hit_and_end_search",
            # Texture functions
            "texture": "tex2D",
            "textureLod": "tex2DLod",
            "textureGrad": "tex2DGrad",
        }

        # Built-in variable mappings
        self.builtin_map = {
            "gl_LocalInvocationID": "make_uint3(threadIdx.x, threadIdx.y, threadIdx.z)",
            "gl_LocalInvocationID.x": "threadIdx.x",
            "gl_LocalInvocationID.y": "threadIdx.y",
            "gl_LocalInvocationID.z": "threadIdx.z",
            "gl_WorkGroupID": "make_uint3(blockIdx.x, blockIdx.y, blockIdx.z)",
            "gl_WorkGroupID.x": "blockIdx.x",
            "gl_WorkGroupID.y": "blockIdx.y",
            "gl_WorkGroupID.z": "blockIdx.z",
            "gl_WorkGroupSize": "make_uint3(blockDim.x, blockDim.y, blockDim.z)",
            "gl_WorkGroupSize.x": "blockDim.x",
            "gl_WorkGroupSize.y": "blockDim.y",
            "gl_WorkGroupSize.z": "blockDim.z",
            "gl_NumWorkGroups": "make_uint3(gridDim.x, gridDim.y, gridDim.z)",
            "gl_NumWorkGroups.x": "gridDim.x",
            "gl_NumWorkGroups.y": "gridDim.y",
            "gl_NumWorkGroups.z": "gridDim.z",
            "gl_GlobalInvocationID": (
                "make_uint3((blockIdx.x * blockDim.x + threadIdx.x), "
                "(blockIdx.y * blockDim.y + threadIdx.y), "
                "(blockIdx.z * blockDim.z + threadIdx.z))"
            ),
            "gl_GlobalInvocationID.x": "(blockIdx.x * blockDim.x + threadIdx.x)",
            "gl_GlobalInvocationID.y": "(blockIdx.y * blockDim.y + threadIdx.y)",
            "gl_GlobalInvocationID.z": "(blockIdx.z * blockDim.z + threadIdx.z)",
            "gl_LocalInvocationIndex": (
                "(threadIdx.z * blockDim.y * blockDim.x + "
                "threadIdx.y * blockDim.x + threadIdx.x)"
            ),
            "SV_GroupThreadID": "make_uint3(threadIdx.x, threadIdx.y, threadIdx.z)",
            "SV_GroupThreadID.x": "threadIdx.x",
            "SV_GroupThreadID.y": "threadIdx.y",
            "SV_GroupThreadID.z": "threadIdx.z",
            "SV_GROUPTHREADID": "make_uint3(threadIdx.x, threadIdx.y, threadIdx.z)",
            "SV_GROUPTHREADID.x": "threadIdx.x",
            "SV_GROUPTHREADID.y": "threadIdx.y",
            "SV_GROUPTHREADID.z": "threadIdx.z",
            "SV_GroupID": "make_uint3(blockIdx.x, blockIdx.y, blockIdx.z)",
            "SV_GroupID.x": "blockIdx.x",
            "SV_GroupID.y": "blockIdx.y",
            "SV_GroupID.z": "blockIdx.z",
            "SV_GROUPID": "make_uint3(blockIdx.x, blockIdx.y, blockIdx.z)",
            "SV_GROUPID.x": "blockIdx.x",
            "SV_GROUPID.y": "blockIdx.y",
            "SV_GROUPID.z": "blockIdx.z",
            "SV_DispatchThreadID": (
                "make_uint3((blockIdx.x * blockDim.x + threadIdx.x), "
                "(blockIdx.y * blockDim.y + threadIdx.y), "
                "(blockIdx.z * blockDim.z + threadIdx.z))"
            ),
            "SV_DispatchThreadID.x": "(blockIdx.x * blockDim.x + threadIdx.x)",
            "SV_DispatchThreadID.y": "(blockIdx.y * blockDim.y + threadIdx.y)",
            "SV_DispatchThreadID.z": "(blockIdx.z * blockDim.z + threadIdx.z)",
            "SV_DISPATCHTHREADID": (
                "make_uint3((blockIdx.x * blockDim.x + threadIdx.x), "
                "(blockIdx.y * blockDim.y + threadIdx.y), "
                "(blockIdx.z * blockDim.z + threadIdx.z))"
            ),
            "SV_DISPATCHTHREADID.x": "(blockIdx.x * blockDim.x + threadIdx.x)",
            "SV_DISPATCHTHREADID.y": "(blockIdx.y * blockDim.y + threadIdx.y)",
            "SV_DISPATCHTHREADID.z": "(blockIdx.z * blockDim.z + threadIdx.z)",
            "SV_GroupIndex": (
                "(threadIdx.z * blockDim.y * blockDim.x + "
                "threadIdx.y * blockDim.x + threadIdx.x)"
            ),
            "SV_GROUPINDEX": (
                "(threadIdx.z * blockDim.y * blockDim.x + "
                "threadIdx.y * blockDim.x + threadIdx.x)"
            ),
        }

    def generate(self, node):
        """Generate complete HIP source for a CrossGL AST, or raise :class:`HipLoweringRefused`.

        A construct with no faithful HIP lowering is refused; no program is returned while a refusal is
        pending, and a rendered program that still carries a ``/* unsupported ... */`` loss marker is refused.
        """
        refuse_cooperative_matrices(node, label="HIP", target="hip")
        return merge_adjacent_cuda_barriers(
            self.refusals.run(self.render_checked_program, node)
        )

    def render_checked_program(self, node):
        """Render the program and refuse each built-in type and function of another language it still names."""
        self.program_type_names = declared_type_names(node)
        self.written_foreign_type_names = set()
        try:
            output = self.render_program(node)
            refuse_written_foreign_types(output, self.written_foreign_type_names, "HIP")
            refuse_written_foreign_calls(output, CUDA_CALLABLE_FUNCTION_NAMES, "HIP")
        finally:
            self.written_foreign_type_names = None
        return output

    def render_program(self, node: ASTNode) -> str:
        """Render complete HIP source for a CrossGL AST."""
        # A HIP texture object is the image whatever sampler state a separate CrossGL texture is paired with.
        node = canonicalize_separate_textures(node)
        # A user function named like a device library function would redefine it or make its calls ambiguous.
        node = rename_cuda_library_functions(node)
        # GLSL's lessThan, equal, not and the other relational built-ins are the componentwise operators.
        lower_relational_builtins(node)
        # HIP has no hidden buffer counter, so a counted RWStructuredBuffer gets an explicit one.
        node = materialize_structured_buffer_counters(node, _refuse_buffer_counter)
        self.code_lines = []
        self.indent_level = 0
        self.validate_supported_stage_types(node)
        self.variable_types = {}
        self.image_resource_accesses = {}
        self.buffer_resource_accesses = {}
        self.glsl_buffer_block_accesses = {}
        self.glsl_buffer_block_layouts = {}
        self.hip_resource_binding_cursors = {}
        self.hip_used_resource_bindings = {}
        self.current_function_return_type = None
        self.match_temp_variable_index = 0
        self.struct_member_types = {}
        self.struct_member_semantics = {}
        self.function_return_types = self.collect_function_return_types(node)
        self.function_parameters_by_name = collect_function_parameters_by_name(
            node, self.type_name_string
        )
        self.value_array_struct_names = collect_struct_names(node)
        self.helper_functions = {}
        self.resource_query_info_required = False
        self.assignment_lhs_depth = 0
        # Subscripts visited as the index of another subscript: reads, even on an assignment's left side.
        self.subscript_index_depth = 0
        self.stage_builtin_aliases = {}
        self.stage_builtin_alias_types = {}
        self.current_function_is_kernel_entry = False
        self.current_stage_name = None
        self.hip_function_capture_params = {}
        self.current_expression_expected_type = None
        self.setup_enum_and_generic_metadata(node)
        self.generic_function_definitions = {}
        self.generic_function_definition_ordinals = {}
        self.generic_function_specializations = {}
        self.generic_function_specialized_names = {}
        self.current_generic_function_substitutions = {}
        generic_function_specializations = prepare_generic_function_specializations(
            self,
            list(iter_function_nodes(node)),
            collect_constant_and_cbuffer_member_types(self, node),
            ast=node,
        )
        self.function_return_types.update(
            {
                func.name: self.type_name_string(getattr(func, "return_type", "void"))
                for func in generic_function_specializations.values()
            }
        )
        self.reject_unsupported_generic_functions(node)
        (
            self.query_resource_names,
            self.query_metadata_function_params,
        ) = self.collect_resource_query_requirements(node)
        (
            self.structured_buffer_length_names,
            self.structured_buffer_length_function_params,
        ) = self.collect_structured_buffer_length_requirements(node)
        self.query_functions_by_name = {
            getattr(func, "name", None): func
            for func in self.query_collect_functions(node)
        }
        self.query_functions_by_name = {
            name: func for name, func in self.query_functions_by_name.items() if name
        }
        self.reserve_explicit_hip_resource_bindings(node)

        self.add_includes()
        self.add_generated_code(self.generate_hip_matrix_type_helpers())
        self.add_line()
        if self.has_geometry_stage(node):
            self.add_generated_code(self.generate_hip_geometry_stream_helpers())
            self.add_line()
        if self.has_tessellation_stage(node):
            self.add_generated_code(self.generate_hip_tessellation_patch_helpers())
            self.add_line()
        if self.has_mesh_task_stage(node):
            self.add_generated_code(self.generate_hip_mesh_task_helpers())
            self.add_line()
        self.visit(node)
        self.insert_helper_functions()

        return "\n".join(self.code_lines)

    def visit(self, node: ASTNode) -> str:
        """Dispatch an AST node to its HIP visitor method."""
        method_name = f"visit_{type(node).__name__}"
        visitor = getattr(self, method_name, self.generic_visit)
        return visitor(node)

    def generic_visit(self, node: ASTNode) -> str:
        """Fail closed when no semantics-preserving HIP visitor exists."""
        raise UnsupportedHIPAstNodeError(node)

    def visit_FunctionNode(self, node: FunctionNode) -> str:
        """Render a CrossGL function or compute entry point as HIP code."""
        saved_variable_types = self.variable_types.copy()
        saved_image_resource_accesses = self.image_resource_accesses.copy()
        saved_buffer_resource_accesses = self.buffer_resource_accesses.copy()
        saved_glsl_buffer_block_accesses = self.glsl_buffer_block_accesses.copy()
        saved_glsl_buffer_block_layouts = self.glsl_buffer_block_layouts.copy()
        self.current_function = node.name
        saved_current_function_return_type = self.current_function_return_type
        saved_current_function_name = self.current_function_name
        saved_stage_builtin_aliases = self.stage_builtin_aliases
        saved_stage_builtin_alias_types = self.stage_builtin_alias_types
        saved_current_function_is_kernel_entry = self.current_function_is_kernel_entry
        saved_generic_function_substitutions = (
            self.current_generic_function_substitutions
        )
        saved_structured_buffer_length_parameters = (
            self.current_structured_buffer_length_parameters
        )
        self.current_function_name = node.name
        self.current_generic_function_substitutions = (
            getattr(node, "_generic_substitutions", {}) or {}
        )
        self.current_structured_buffer_length_parameters = {}
        self.stage_builtin_aliases = {}
        self.stage_builtin_alias_types = {}
        self.current_function_is_kernel_entry = False

        qualifiers = []
        source_qualifiers = list(getattr(node, "qualifiers", []) or [])
        stage_attribute_names = {
            "compute",
            "fragment",
            "geometry",
            "kernel",
            "task",
            "mesh",
            "vertex",
        } | self.hip_ray_stage_names()
        source_qualifiers.extend(
            attribute_name
            for attribute_name in (
                getattr(attribute, "name", None)
                for attribute in getattr(node, "attributes", []) or []
            )
            if normalize_stage_name(attribute_name) in stage_attribute_names
        )
        source_qualifiers = [
            qualifier for qualifier in source_qualifiers if qualifier is not None
        ]
        if source_qualifiers:
            for qualifier in source_qualifiers:
                qualifier_name = normalize_stage_name(qualifier)
                if qualifier_name in {"kernel", "compute"}:
                    qualifiers.append("__global__")
                elif qualifier_name == "device":
                    qualifiers.append("__device__")
                else:
                    qualifiers.append("__device__")
        elif hasattr(node, "qualifier") and node.qualifier:
            qualifier_name = normalize_stage_name(node.qualifier)
            if qualifier_name in {"kernel", "compute"}:
                qualifiers.append("__global__")
            elif qualifier_name == "device":
                qualifiers.append("__device__")
            else:
                qualifiers.append("__device__")
        else:
            qualifiers.append("__device__")

        stage_name = self.function_stage_name(node)
        is_kernel_entry = "__global__" in qualifiers
        self.current_function_is_kernel_entry = is_kernel_entry

        if hasattr(node, "return_type"):
            self.current_function_return_type = node.return_type
            return_type = self.map_type(node.return_type)
        else:
            self.current_function_return_type = "void"
            return_type = "void"

        param_list = list(getattr(node, "parameters", getattr(node, "params", [])))
        param_list.extend(self.hip_function_capture_params.get(node.name, []))
        return_semantic = self.semantic_from_node(node)
        fragment_output_param = None
        fragment_output_params = [
            param
            for param in param_list
            if is_fragment_output_parameter(
                stage_name,
                param,
                self.semantic_from_node(param),
            )
        ]
        if (
            fragment_output_params
            and return_semantic is None
            and self.map_type(self.current_function_return_type) == "void"
        ):
            fragment_output_param = fragment_output_params[0]
            self.current_function_return_type = self.get_parameter_type(
                fragment_output_param
            )
            return_type = self.map_type(self.current_function_return_type)
            return_semantic = self.semantic_from_node(fragment_output_param)
            param_list = [
                param for param in param_list if param is not fragment_output_param
            ]

        self.validate_hip_return_semantic(
            stage_name, self.current_function_return_type, return_semantic
        )
        self.validate_hip_struct_return_semantics(
            stage_name, self.current_function_return_type
        )
        if is_kernel_entry:
            self.current_function_return_type = "void"
            return_type = "void"

        self.validate_hip_stage_parameter_semantics(stage_name, param_list)
        if stage_name == "geometry":
            self.validate_hip_geometry_stage(node, param_list)
        if stage_name in {"tessellation_control", "tessellation_evaluation"}:
            self.validate_hip_tessellation_stage(node, stage_name)
        if stage_name in self.hip_mesh_task_stage_names():
            self.validate_hip_mesh_task_stage(node, stage_name)
        value_array_copies = (
            {}
            if stage_name is not None
            else plan_value_array_copies(
                node,
                param_list,
                self.type_name_string,
                self.function_parameters_by_name,
                self.value_array_struct_names,
            )
        )
        value_array_copy_statements = []
        param_declarations = []
        for param in param_list:
            param_type = self.get_parameter_type(param)
            param_name = getattr(param, "name", getattr(param, "param_name", None))
            builtin_role = self.hip_compute_builtin_parameter_role(stage_name, param)
            if param_name and builtin_role is not None:
                self.register_variable_type(param_name, param_type, param)
                self.stage_builtin_aliases[param_name] = builtin_role
                self.stage_builtin_alias_types[param_name] = param_type
                continue

            if param_name in value_array_copies:
                incoming = value_array_copies[param_name]
                declaration_type = self.resource_type_with_access(param_type, param)
                self.register_variable_type(param_name, declaration_type, param)
                param_declarations.append(
                    self.apply_resource_memory_qualifiers_to_declaration(
                        self.format_typed_declarator(declaration_type, incoming),
                        incoming,
                        param,
                    )
                )
                value_array_copy_statements.extend(
                    value_array_copy_lines(
                        self.apply_resource_memory_qualifiers_to_declaration(
                            self.format_typed_declarator(declaration_type, param_name),
                            param_name,
                            param,
                        ),
                        param_name,
                        incoming,
                        fixed_array_dimensions(self.type_name_string(declaration_type)),
                    )
                )
            else:
                declaration = self.visit_parameter(param)
                if stage_name is None and is_output_parameter(param):
                    # out and inout copy back to the caller; a C++ reference binds it.
                    declaration = reference_declarator(declaration, param_name)
                param_declarations.append(declaration)
            metadata_param = self.query_metadata_parameter(param_name, param_type)
            if metadata_param:
                param_declarations.append(metadata_param)
            length_param = self.structured_buffer_length_parameter(
                node.name, param_name, param_type
            )
            if length_param:
                self.current_structured_buffer_length_parameters[param_name] = (
                    self.structured_buffer_length_name(param_name)
                )
                param_declarations.append(length_param)
            counter_param = self.structured_buffer_counter_parameter(
                param_name, param_type
            )
            if counter_param:
                param_declarations.append(counter_param)
        params = ", ".join(param_declarations)

        qualifier_str = " ".join(qualifiers)
        function_name = getattr(self, "current_stage_entry_function_name", None)
        if not function_name:
            function_name = node.name
        if is_kernel_entry and function_name == "main":
            function_name = "compute_main"
        launch_bounds = self.hip_compute_launch_bounds_attribute(
            node, stage_name, is_kernel_entry
        )
        if launch_bounds is not None:
            signature = (
                f"{qualifier_str} {return_type} {launch_bounds} "
                f"{function_name}({params})"
            )
        else:
            signature = f"{qualifier_str} {return_type} {function_name}({params})"
        body = getattr(node, "body", None)

        if stage_name in self.hip_ray_stage_names():
            self.add_line(
                "// CrossGL ray stage: "
                f"{self.hip_ray_stage_metadata_name(stage_name)}"
            )
        if stage_name == "geometry":
            self.add_generated_code(
                self.generate_hip_geometry_stage_comments(node, param_list)
            )
        if stage_name in {"tessellation_control", "tessellation_evaluation"}:
            self.add_generated_code(
                self.generate_hip_tessellation_stage_comments(node, stage_name)
            )
        if stage_name in self.hip_mesh_task_stage_names():
            self.add_generated_code(
                self.generate_hip_mesh_task_stage_comments(node, stage_name)
            )
        if return_semantic:
            self.add_line(
                f"// CrossGL return semantic: {self.map_semantic(return_semantic)}"
            )
        if stage_name:
            for param in param_list:
                param_semantic = self.semantic_from_node(param)
                if param_semantic:
                    param_name = getattr(
                        param, "name", getattr(param, "param_name", "param")
                    )
                    self.add_line(
                        f"// CrossGL parameter semantic: {param_name}: "
                        f"{self.map_semantic(param_semantic)}"
                    )
        if body is None:
            self.add_line(f"{signature};")
        else:
            self.add_line(signature)
            self.add_line("{")
            self.indent_level += 1
            precise_state = self.enter_precise_evaluation(node)
            for line in self.precise_body_prologue():
                self.add_line(line)
            for statement in value_array_copy_statements:
                self.add_line(statement)
            fragment_output_name = None
            if fragment_output_param is not None:
                fragment_output_name = getattr(
                    fragment_output_param,
                    "name",
                    getattr(fragment_output_param, "param_name", None),
                )
                fragment_output_type = self.resource_type_with_access(
                    self.get_parameter_type(fragment_output_param),
                    fragment_output_param,
                )
                self.register_variable_type(
                    fragment_output_name,
                    fragment_output_type,
                    fragment_output_param,
                )
                self.add_line(
                    f"{self.format_typed_declarator(fragment_output_type, fragment_output_name)};"
                )
            for local_var in getattr(self, "current_stage_local_variables", []) or []:
                if self.is_hip_shared_variable(local_var):
                    self.visit(local_var)
            try:
                self.emit_body(body)
            finally:
                self.leave_precise_evaluation(precise_state)
            if fragment_output_name is not None and not self.statement_body_terminates(
                body
            ):
                self.add_line(f"return {fragment_output_name};")

            self.indent_level -= 1
            self.add_line("}")

        self.add_line()
        self.current_function = None
        self.current_function_return_type = saved_current_function_return_type
        self.variable_types = saved_variable_types
        self.image_resource_accesses = saved_image_resource_accesses
        self.buffer_resource_accesses = saved_buffer_resource_accesses
        self.glsl_buffer_block_accesses = saved_glsl_buffer_block_accesses
        self.glsl_buffer_block_layouts = saved_glsl_buffer_block_layouts
        self.current_function_name = saved_current_function_name
        self.stage_builtin_aliases = saved_stage_builtin_aliases
        self.stage_builtin_alias_types = saved_stage_builtin_alias_types
        self.current_function_is_kernel_entry = saved_current_function_is_kernel_entry
        self.current_generic_function_substitutions = (
            saved_generic_function_substitutions
        )
        self.current_structured_buffer_length_parameters = (
            saved_structured_buffer_length_parameters
        )
        return ""

    def format_variable_declaration(self, node: VariableNode) -> str:
        initial_value = getattr(node, "initial_value", getattr(node, "value", None))
        var_type = self.get_variable_node_type(node)

        if var_type is None and initial_value is not None:
            inferred_type = self.expression_result_type(initial_value)
            self.register_variable_type(node.name, inferred_type, node, initial_value)
            declaration = self.format_typed_declarator(
                inferred_type or "auto", node.name
            )
        else:
            var_type = var_type or "int"
            self.register_variable_type(node.name, var_type, node, initial_value)
            var_type = self.glsl_buffer_block_declaration_type(var_type, node)
            declaration = self.format_typed_declarator(var_type, node.name)

        declaration = self.apply_resource_memory_qualifiers_to_declaration(
            declaration, node.name, node
        )
        if (
            initial_value is not None
            and self.declaration_qualifier_names(node) & {"const", "constant"}
            and "*" not in declaration
            and not declaration.startswith("const ")
        ):
            # A CrossGL constant, or an MSL program-scope ``constant`` value, stays one: file-scope device code
            # can read a const scalar with a constant initializer, but not a mutable host variable.
            declaration = f"const {declaration}"

        qualifiers = self.variable_memory_qualifiers(node, declaration)
        if qualifiers:
            declaration = f"{' '.join(qualifiers)} {declaration}"

        if initial_value is not None:
            declaration += f" = {self.visit(initial_value)}"

        return declaration

    def format_assignment_expression(self, node) -> str:
        operator = getattr(node, "operator", getattr(node, "op", "="))
        diagnostic = self.glsl_buffer_block_write_diagnostic(node.left, "assignment")
        if diagnostic is not None:
            return diagnostic
        if operator == "=":
            diagnostic = self.structured_buffer_element_write_diagnostic(
                node.left, "assignment"
            )
        else:
            diagnostic = self.structured_buffer_element_read_write_diagnostic(
                node.left, "assignment"
            )
        if diagnostic is not None:
            return diagnostic
        # C++ has no floating %=; x = x % y lowers the remainder through fmodf.
        remainder = float_remainder_assignment(
            node.left,
            node.right,
            operator,
            type_of=self.expression_result_type,
            is_floating=self.is_floating_value_type,
            backend="hip",
        )
        if remainder is not None:
            return self.format_assignment_expression(remainder)
        swizzle_assignment = self.format_swizzle_assignment_expression(
            node.left,
            node.right,
            operator,
        )
        if swizzle_assignment is not None:
            return swizzle_assignment
        self.assignment_lhs_depth += 1
        try:
            left = self.visit(node.left)
        finally:
            self.assignment_lhs_depth -= 1
        right = self.visit(node.right)
        compound_binary_ops = {
            "+=": "+",
            "-=": "-",
            "*=": "*",
            "/=": "/",
            "%=": "%",
        }
        if operator in compound_binary_ops:
            lowered_right = self.lower_vector_binary_operation(
                node.left,
                left,
                node.right,
                right,
                compound_binary_ops[operator],
            )
            if lowered_right is not None:
                return f"{left} = {lowered_right}"
            if operator == "%=":
                modulo = self.lower_scalar_modulo_operation(
                    node.left,
                    left,
                    node.right,
                    right,
                )
                if modulo is not None:
                    return f"{left} = {modulo}"
        compound_bitwise_ops = {
            "&=": "&",
            "|=": "|",
            "^=": "^",
            "<<=": "<<",
            ">>=": ">>",
        }
        if operator in compound_bitwise_ops:
            lowered_right = self.lower_vector_bitwise_operation(
                node.left,
                left,
                node.right,
                right,
                compound_bitwise_ops[operator],
            )
            if lowered_right is not None:
                return f"{left} = {lowered_right}"
        return f"{left} {operator} {right}"

    def hip_half_constructor_expression(self, constructor_type, raw_args, args):
        """Lower CrossGL FP16 constructors to HIP's documented half intrinsics."""
        if constructor_type in {"f16", "half", "float16"}:
            if not args:
                return "half{}"
            return f"__float2half({args[0]})"

        if constructor_type not in {"vec2<f16>", "half2", "f16vec2"}:
            return None

        if not args:
            return "__float2half2_rn(0.0f)"

        if len(args) == 1:
            arg_type = self.type_name_string(self.expression_result_type(raw_args[0]))
            if arg_type in {"vec2<f16>", "half2", "f16vec2"}:
                return args[0]
            return f"__float2half2_rn({args[0]})"

        return f"__floats2half2_rn({args[0]}, {args[1]})"

    @property
    def local_variable_types(self):
        return self.variable_types

    @local_variable_types.setter
    def local_variable_types(self, value):
        self.variable_types = value

    def is_buffer_resource_type(self, type_name):
        """Return whether a type is a structured or byte-address buffer resource."""
        type_name = self.type_name_string(type_name)
        if not type_name:
            return False
        return (
            self.structured_buffer_type_parts(type_name) is not None
            or self.byte_address_buffer_base_type(type_name) is not None
        )

    def vector_type_info(self, type_name):
        half_vector_info = self.hip_half_vector_type_info(type_name)
        if half_vector_info is not None:
            return half_vector_info
        narrow_info = self.hip_narrow_integer_vector_type_info(type_name)
        if narrow_info is not None:
            return narrow_info
        return super().vector_type_info(type_name)

    def vector_type_for_components(self, component_type, component_count):
        # half has no vecN spelling: f16vec3 and f16vec4 lower to the
        # cgl_half helpers and f16vec2 to the native half2
        if component_type == "half" and 2 <= component_count <= 4:
            return f"f16vec{component_count}"
        return super().vector_type_for_components(component_type, component_count)

    def generate_resource_call(self, func_name, raw_args, args):
        if func_name in {"textureSize", "imageSize"}:
            return self.generate_dimension_query(func_name, raw_args, args)

        if func_name in {"textureSamples", "imageSamples"}:
            return self.generate_sample_count_query(func_name, raw_args, args)

        if func_name == "textureQueryLevels":
            return self.generate_texture_query_levels(raw_args)

        raw_args, args = self.normalize_split_sampler_texture_args(
            func_name, raw_args, args
        )

        if func_name == "textureQueryLod" and len(args) >= 2:
            texture_type = self.resource_base_type(
                self.get_expression_type(raw_args[0])
            )
            if texture_type is not None:
                return self.unsupported_resource_query_call(
                    func_name, texture_type, args
                )

        if (
            func_name in {
                "texture",
                "textureLod",
                "textureGrad",
                "textureGather",
                "textureCompare",
                "textureCompareLod",
                "textureCompareGrad",
                "textureCompareOffset",
                "textureCompareLodOffset",
                "textureCompareGradOffset",
                "textureCompareProj",
                "textureCompareProjOffset",
                "textureCompareProjLod",
                "textureCompareProjLodOffset",
                "textureCompareProjGrad",
                "textureCompareProjGradOffset",
                "textureGatherCompare",
                "textureGatherCompareOffset",
            }
            and len(args) >= 2
        ):
            texture_type = self.resource_base_type(
                self.get_expression_type(raw_args[0])
            )
            if self.is_shadow_resource_type(texture_type):
                return self.unsupported_shadow_resource_call(
                    func_name, texture_type, args
                )

        if func_name == "textureGather" and len(args) >= 2:
            texture_gather = self.generate_texture_gather_call(raw_args, args)
            if texture_gather is not None:
                return texture_gather

        if (
            func_name in {
                "textureGather",
                "textureGatherOffset",
                "textureGatherOffsets",
            }
            and len(args) >= 2
        ):
            texture_type = self.resource_base_type(
                self.get_expression_type(raw_args[0])
            )
            if texture_type is not None:
                return self.unsupported_sampled_resource_call(
                    func_name, texture_type, args
                )

        if (
            func_name in {
                "textureProj",
                "textureProjLod",
                "textureProjGrad",
            }
            and len(args) >= 2
        ):
            projected_texture = self.generate_projected_texture_call(
                func_name, raw_args, args
            )
            if projected_texture is not None:
                return projected_texture

        if func_name == "texelFetchOffset" and len(args) >= 4:
            texel_fetch_offset = self.generate_texel_fetch_offset_call(raw_args, args)
            if texel_fetch_offset is not None:
                return texel_fetch_offset

        if (
            func_name in {
                "textureOffset",
                "textureLodOffset",
                "textureGradOffset",
                "textureProj",
                "textureProjOffset",
                "textureProjLod",
                "textureProjLodOffset",
                "textureProjGrad",
                "textureProjGradOffset",
                "texelFetchOffset",
            }
            and raw_args
        ):
            texture_type = self.resource_base_type(
                self.get_expression_type(raw_args[0])
            )
            if texture_type is not None:
                if self.is_shadow_resource_type(texture_type):
                    return self.unsupported_shadow_resource_call(
                        func_name, texture_type, args
                    )
                if self.is_multisample_resource_type(texture_type):
                    return self.unsupported_multisample_resource_call(
                        func_name, texture_type, args
                    )
                return self.unsupported_sampled_resource_call(
                    func_name, texture_type, args
                )

        if func_name in {
            "imageAtomicAdd",
            "imageAtomicMin",
            "imageAtomicMax",
            "imageAtomicAnd",
            "imageAtomicOr",
            "imageAtomicXor",
            "imageAtomicExchange",
            "imageAtomicCompSwap",
        }:
            image_type = None
            if raw_args:
                image_type = self.resource_base_type(
                    self.get_expression_type(raw_args[0])
                )
                access_diagnostic = self.image_atomic_access_diagnostic(
                    func_name, image_type, raw_args[0]
                )
                if access_diagnostic is not None:
                    return access_diagnostic
            if len(raw_args) >= 2:
                coordinate_diagnostic = self.image_atomic_coordinate_rank_diagnostic(
                    func_name, image_type, raw_args[1]
                )
                if coordinate_diagnostic is not None:
                    return coordinate_diagnostic
            return self.unsupported_image_atomic_resource_call(
                func_name, image_type, args
            )

        if func_name in {"texture", "textureLod", "textureGrad"} and len(args) >= 2:
            texture_type = self.resource_base_type(
                self.get_expression_type(raw_args[0])
            )
            texture_shape = self.sampled_texture_shape_type(texture_type)
            value_type = self.sampled_texture_value_type(texture_type)
            if self.is_multisample_resource_type(texture_type):
                return self.unsupported_multisample_resource_call(
                    func_name, texture_type, args
                )
            coordinate_diagnostic = self.texture_coordinate_rank_diagnostic(
                func_name, texture_type, raw_args[1]
            )
            if coordinate_diagnostic is not None:
                return coordinate_diagnostic
            grad_x = None
            grad_y = None
            if func_name == "textureGrad" and len(args) >= 4:
                gradient_diagnostic = self.texture_gradient_rank_diagnostic(
                    texture_type,
                    raw_args[2],
                    raw_args[3],
                )
                if gradient_diagnostic is not None:
                    return gradient_diagnostic
                grad_x = self.hip_texture_gradient_argument(
                    texture_type,
                    raw_args[2],
                    args[2],
                )
                grad_y = self.hip_texture_gradient_argument(
                    texture_type,
                    raw_args[3],
                    args[3],
                )

            texture_name = args[0]
            coord = args[1]
            if texture_shape == "sampler1D":
                if func_name == "texture":
                    return f"tex1D<{value_type}>({texture_name}, {coord})"
                if func_name == "textureLod" and len(args) >= 3:
                    return f"tex1DLod<{value_type}>({texture_name}, {coord}, {args[2]})"
                if func_name == "textureGrad" and len(args) >= 4:
                    return (
                        f"tex1DGrad<{value_type}>"
                        f"({texture_name}, {coord}, {grad_x}, {grad_y})"
                    )

            if texture_shape == "sampler2D":
                coord_args = (
                    f"{texture_name}, "
                    f"{self.coord_component(coord, 'x')}, "
                    f"{self.coord_component(coord, 'y')}"
                )
                if func_name == "texture":
                    return f"tex2D<{value_type}>({coord_args})"
                if func_name == "textureLod" and len(args) >= 3:
                    return f"tex2DLod<{value_type}>({coord_args}, {args[2]})"
                if func_name == "textureGrad" and len(args) >= 4:
                    return f"tex2DGrad<{value_type}>({coord_args}, {grad_x}, {grad_y})"

            if texture_shape == "sampler1DArray":
                coord_args = (
                    f"{texture_name}, "
                    f"{self.coord_component(coord, 'x')}, "
                    f"{self.coord_component(coord, 'y')}"
                )
                if func_name == "texture":
                    return f"tex1DLayered<{value_type}>({coord_args})"
                if func_name == "textureLod" and len(args) >= 3:
                    return f"tex1DLayeredLod<{value_type}>({coord_args}, {args[2]})"
                if func_name == "textureGrad" and len(args) >= 4:
                    return (
                        f"tex1DLayeredGrad<{value_type}>"
                        f"({coord_args}, {grad_x}, {grad_y})"
                    )

            if texture_shape == "sampler2DArray":
                coord_args = (
                    f"{texture_name}, "
                    f"{self.coord_component(coord, 'x')}, "
                    f"{self.coord_component(coord, 'y')}, "
                    f"{self.coord_component(coord, 'z')}"
                )
                if func_name == "texture":
                    return f"tex2DLayered<{value_type}>({coord_args})"
                if func_name == "textureLod" and len(args) >= 3:
                    return f"tex2DLayeredLod<{value_type}>({coord_args}, {args[2]})"
                if func_name == "textureGrad" and len(args) >= 4:
                    return (
                        f"tex2DLayeredGrad<{value_type}>"
                        f"({coord_args}, {grad_x}, {grad_y})"
                    )

            if texture_shape == "sampler3D":
                coord_args = (
                    f"{texture_name}, "
                    f"{self.coord_component(coord, 'x')}, "
                    f"{self.coord_component(coord, 'y')}, "
                    f"{self.coord_component(coord, 'z')}"
                )
                if func_name == "texture":
                    return f"tex3D<{value_type}>({coord_args})"
                if func_name == "textureLod" and len(args) >= 3:
                    return f"tex3DLod<{value_type}>({coord_args}, {args[2]})"
                if func_name == "textureGrad" and len(args) >= 4:
                    return f"tex3DGrad<{value_type}>({coord_args}, {grad_x}, {grad_y})"

            if texture_shape == "samplerCube":
                coord_args = (
                    f"{texture_name}, "
                    f"{self.coord_component(coord, 'x')}, "
                    f"{self.coord_component(coord, 'y')}, "
                    f"{self.coord_component(coord, 'z')}"
                )
                if func_name == "texture":
                    return f"texCubemap<{value_type}>({coord_args})"
                if func_name == "textureLod" and len(args) >= 3:
                    return f"texCubemapLod<{value_type}>({coord_args}, {args[2]})"
                if func_name == "textureGrad" and len(args) >= 4:
                    return (
                        f"texCubemapGrad<{value_type}>"
                        f"({coord_args}, {grad_x}, {grad_y})"
                    )

            if texture_shape == "samplerCubeArray":
                coord_args = (
                    f"{texture_name}, "
                    f"{self.coord_component(coord, 'x')}, "
                    f"{self.coord_component(coord, 'y')}, "
                    f"{self.coord_component(coord, 'z')}, "
                    f"{self.coord_component(coord, 'w')}"
                )
                if func_name == "texture":
                    return f"texCubemapLayered<{value_type}>({coord_args})"
                if func_name == "textureLod" and len(args) >= 3:
                    return (
                        f"texCubemapLayeredLod<{value_type}>({coord_args}, {args[2]})"
                    )
                if func_name == "textureGrad" and len(args) >= 4:
                    return (
                        f"texCubemapLayeredGrad<{value_type}>"
                        f"({coord_args}, {grad_x}, {grad_y})"
                    )

        if func_name == "texelFetch" and len(args) >= 3:
            texture_type = self.resource_base_type(
                self.get_expression_type(raw_args[0])
            )
            if texture_type is not None and not self.is_sampled_texture_family_type(
                texture_type
            ):
                return self.unsupported_sampled_resource_call(
                    func_name, texture_type, args
                )
            if self.is_shadow_resource_type(texture_type):
                return self.unsupported_shadow_resource_call(
                    func_name, texture_type, args
                )
            if self.is_multisample_resource_type(texture_type):
                return self.unsupported_multisample_resource_call(
                    func_name, texture_type, args
                )
            texture_shape = self.sampled_texture_shape_type(texture_type)
            if texture_shape in {"samplerCube", "samplerCubeArray"}:
                return self.unsupported_sampled_resource_call(
                    func_name, texture_type, args
                )

            texture_name = args[0]
            coord = args[1]
            value_type = self.sampled_texture_value_type(texture_type)
            expected_coord_count = self.expected_texel_fetch_coordinate_count(
                texture_shape
            )
            actual_coord_count = self.texel_fetch_coordinate_count(raw_args[1])
            if (
                expected_coord_count is not None
                and actual_coord_count is not None
                and actual_coord_count != expected_coord_count
            ):
                return self.unsupported_sampled_resource_call(
                    "texelFetch coordinate rank",
                    texture_type,
                    args,
                )
            if texture_shape == "sampler1D":
                return f"tex1Dfetch<{value_type}>({texture_name}, {coord})"
            if texture_shape == "sampler1DArray":
                return (
                    f"tex1DLayered<{value_type}>({texture_name}, "
                    f"{self.coord_component(coord, 'x')}, "
                    f"{self.coord_component(coord, 'y')})"
                )
            if texture_shape == "sampler2D":
                return (
                    f"tex2D<{value_type}>({texture_name}, "
                    f"{self.coord_component(coord, 'x')}, "
                    f"{self.coord_component(coord, 'y')})"
                )
            if texture_shape == "sampler2DArray":
                return (
                    f"tex2DLayered<{value_type}>({texture_name}, "
                    f"{self.coord_component(coord, 'x')}, "
                    f"{self.coord_component(coord, 'y')}, "
                    f"{self.coord_component(coord, 'z')})"
                )
            if texture_shape == "sampler3D":
                return (
                    f"tex3D<{value_type}>({texture_name}, "
                    f"{self.coord_component(coord, 'x')}, "
                    f"{self.coord_component(coord, 'y')}, "
                    f"{self.coord_component(coord, 'z')})"
                )

        if func_name == "imageLoad" and len(args) >= 2:
            image_type = self.resource_base_type(self.get_expression_type(raw_args[0]))
            if image_type is None:
                return None
            if self.is_multisample_resource_type(image_type):
                return self.unsupported_multisample_resource_call(
                    func_name, image_type, args
                )
            access_diagnostic = self.image_access_diagnostic(
                func_name, image_type, raw_args[0]
            )
            if access_diagnostic is not None:
                return access_diagnostic

            image_name = args[0]
            coord = args[1]
            coordinate_diagnostic = self.image_coordinate_rank_diagnostic(
                func_name, image_type, raw_args[1]
            )
            if coordinate_diagnostic is not None:
                return coordinate_diagnostic
            value_type = self.image_value_type(image_type)
            x = self.image_surface_x_offset(coord, value_type, image_type)
            y = self.coord_component(coord, "y")

            if "1DArray" in image_type:
                self.require_surface_read_helper("cgl_surf1DLayeredread")
                layer = self.coord_component(coord, "y")
                return (
                    f"cgl_surf1DLayeredread<{value_type}>"
                    f"({image_name}, {x}, {layer})"
                )
            if "1D" in image_type:
                self.require_surface_read_helper("cgl_surf1Dread")
                return f"cgl_surf1Dread<{value_type}>({image_name}, {x})"
            if "CubeArray" in image_type:
                self.require_surface_read_helper("cgl_surfCubemapLayeredread")
                face = self.coord_component(coord, "z")
                layer = self.coord_component(coord, "w")
                return (
                    f"cgl_surfCubemapLayeredread<{value_type}>"
                    f"({image_name}, {x}, {y}, {face}, {layer})"
                )
            if "Cube" in image_type:
                self.require_surface_read_helper("cgl_surfCubemapread")
                face = self.coord_component(coord, "z")
                return (
                    f"cgl_surfCubemapread<{value_type}>"
                    f"({image_name}, {x}, {y}, {face})"
                )
            if "3D" in image_type:
                self.require_surface_read_helper("cgl_surf3Dread")
                z = self.coord_component(coord, "z")
                return f"cgl_surf3Dread<{value_type}>({image_name}, {x}, {y}, {z})"
            if "Array" in image_type:
                self.require_surface_read_helper("cgl_surf2DLayeredread")
                layer = self.coord_component(coord, "z")
                return (
                    f"cgl_surf2DLayeredread<{value_type}>"
                    f"({image_name}, {x}, {y}, {layer})"
                )
            if "2D" in image_type:
                self.require_surface_read_helper("cgl_surf2Dread")
                return f"cgl_surf2Dread<{value_type}>({image_name}, {x}, {y})"

        if func_name == "imageStore" and len(args) >= 3:
            image_type = self.resource_base_type(self.get_expression_type(raw_args[0]))
            if image_type is None:
                return None
            if self.is_multisample_resource_type(image_type):
                return self.unsupported_multisample_resource_call(
                    func_name, image_type, args
                )
            access_diagnostic = self.image_access_diagnostic(
                func_name, image_type, raw_args[0]
            )
            if access_diagnostic is not None:
                return access_diagnostic

            image_name = args[0]
            coord = args[1]
            value = args[2]
            coordinate_diagnostic = self.image_coordinate_rank_diagnostic(
                func_name, image_type, raw_args[1]
            )
            if coordinate_diagnostic is not None:
                return coordinate_diagnostic
            value_type = self.image_value_type(image_type)
            x = self.image_surface_x_offset(coord, value_type, image_type)
            y = self.coord_component(coord, "y")

            if "1DArray" in image_type:
                layer = self.coord_component(coord, "y")
                return f"surf1DLayeredwrite({value}, {image_name}, {x}, {layer})"
            if "1D" in image_type:
                return f"surf1Dwrite({value}, {image_name}, {x})"
            if "CubeArray" in image_type:
                face = self.coord_component(coord, "z")
                layer = self.coord_component(coord, "w")
                return (
                    f"surfCubemapLayeredwrite"
                    f"({value}, {image_name}, {x}, {y}, {face}, {layer})"
                )
            if "Cube" in image_type:
                face = self.coord_component(coord, "z")
                return f"surfCubemapwrite({value}, {image_name}, {x}, {y}, {face})"
            if "3D" in image_type:
                z = self.coord_component(coord, "z")
                return f"surf3Dwrite({value}, {image_name}, {x}, {y}, {z})"
            if "Array" in image_type:
                layer = self.coord_component(coord, "z")
                return f"surf2DLayeredwrite({value}, {image_name}, {x}, {y}, {layer})"
            if "2D" in image_type:
                return f"surf2Dwrite({value}, {image_name}, {x}, {y})"

        return None

    def array_access_element_type(self, type_name):
        """Return the element type for HIP arrays and structured buffers."""
        array_element_type = super().array_access_element_type(type_name)
        if array_element_type is not None:
            return array_element_type

        indirect_info = self.hip_indirect_type_info(type_name)
        if indirect_info is not None and indirect_info["kind"] == "pointer":
            return indirect_info["pointee_type"]

        parts = self.structured_buffer_type_parts(type_name)
        if parts is not None:
            return parts[1]
        return None

    def resource_call_result_type(self, func_name, raw_args):
        if func_name in {"texelFetch", "texelFetchOffset"} and raw_args:
            resource_type = self.resource_base_type(
                self.resource_expression_type(raw_args[0])
            )
            if self.is_shadow_resource_type(resource_type):
                return "float"
            if self.is_sampled_texture_family_type(resource_type):
                return self.sampled_texture_value_type(resource_type)

        if func_name in {
            "texture",
            "textureLod",
            "textureGrad",
            "textureOffset",
            "textureLodOffset",
            "textureGradOffset",
            "textureProj",
            "textureProjOffset",
            "textureProjLod",
            "textureProjLodOffset",
            "textureProjGrad",
            "textureProjGradOffset",
            "texelFetch",
            "texelFetchOffset",
        }:
            resource_type = None
            if raw_args:
                resource_type = self.resource_base_type(
                    self.get_expression_type(raw_args[0])
                )
            if self.is_shadow_resource_type(resource_type):
                return "float"
            if self.is_sampled_texture_family_type(resource_type):
                return self.sampled_texture_value_type(resource_type)

        if func_name in {
            "textureProj",
            "textureProjLod",
            "textureProjGrad",
            "texelFetchOffset",
        }:
            resource_type = None
            if raw_args:
                resource_type = self.resource_base_type(
                    self.get_expression_type(raw_args[0])
                )
            if self.is_shadow_resource_type(resource_type):
                return "float"
            return self.sampled_texture_value_type(resource_type)
        return super().resource_call_result_type(func_name, raw_args)

    def expression_result_type(self, node):
        """Infer expression result types with HIP buffer operations."""
        if is_comma_expression(node):
            return self.expression_result_type(comma_result_operand(node))
        if isinstance(node, (IdentifierNode, VariableNode)):
            name = getattr(node, "name", None)
            if not self.source_identifier_shadows_builtin(name):
                builtin_type = self.hip_compute_builtin_type(name)
                if builtin_type is not None:
                    return builtin_type
        if isinstance(node, WaveOpNode):
            return self.wave_result_type(
                getattr(node, "operation", ""), getattr(node, "arguments", []) or []
            )
        if isinstance(node, RayTracingOpNode):
            if getattr(node, "operation", "") == "ReportHit":
                return "bool"
            return None
        if isinstance(node, RayQueryOpNode):
            operation = getattr(node, "operation", "")
            if operation == "Proceed":
                return "bool"
            if operation in {"CandidateTriangleRayT", "CommittedRayT"}:
                return "float"
            return "uint"
        if isinstance(node, FunctionCallNode):
            function_expr = getattr(node, "function", getattr(node, "name", None))
            func_name = getattr(function_expr, "name", function_expr)
            numeric_result_type = numeric_trait_method_result_type(self, node)
            if numeric_result_type is not None:
                return numeric_result_type
            raw_args = getattr(node, "arguments", getattr(node, "args", [])) or []
            specialized_func_name = generic_function_call_name(
                self,
                func_name,
                raw_args,
            )
            if specialized_func_name in self.function_return_types:
                return self.function_return_types[specialized_func_name]
            if self.is_user_defined_function(func_name):
                return self.function_return_types.get(func_name)
            if func_name in HIP_SCALAR_CONSTRUCTOR_TYPE_ALIASES and len(raw_args) <= 1:
                return func_name
            if func_name in HIP_WAVE_OP_ARITIES:
                return self.wave_result_type(
                    func_name,
                    raw_args,
                )
            if func_name == "ReportHit":
                return "bool"
            scalar_boolean_type = self.scalar_boolean_builtin_result_type(
                func_name, raw_args
            )
            if scalar_boolean_type is not None:
                return scalar_boolean_type
            if func_name in self.HIP_SHADER_CLOCK_RESULT_TYPES:
                return self.HIP_SHADER_CLOCK_RESULT_TYPES[func_name]
            if (
                func_name in {"inverseSqrt", "rsqrt"}
                and raw_args
                and not self.is_user_defined_function(func_name)
            ):
                return self.expression_result_type(raw_args[0])
            reflection_result_type = self.reflection_result_type(func_name, raw_args)
            if reflection_result_type is not None:
                return reflection_result_type
            bitcast_result_type = self.hip_bitcast_result_type(func_name, raw_args)
            if bitcast_result_type is not None:
                return bitcast_result_type
            integer_bit_result_type = self.hip_integer_bit_result_type(
                func_name,
                raw_args,
            )
            if integer_bit_result_type is not None:
                return integer_bit_result_type
            if func_name in {"fma", "mad"}:
                cached_type = getattr(node, "expression_type", None) or getattr(
                    node, "vtype", None
                )
                if cached_type is not None:
                    return cached_type
                return self.fused_multiply_add_result_type(raw_args)
            if (
                func_name == "lerp"
                and len(raw_args) == 3
                and not self.is_user_defined_function(func_name)
            ):
                return self.expression_result_type(raw_args[0]) or (
                    self.expression_result_type(raw_args[1])
                    if len(raw_args) > 1
                    else None
                )
            if func_name == "step" and len(raw_args) == 2:
                return self.step_result_type(raw_args)
            buffer_result_type = self.buffer_call_result_type(node)
            if buffer_result_type is not None:
                return buffer_result_type
        if isinstance(node, MemberAccessNode):
            object_node = getattr(
                node,
                "object_expr",
                getattr(node, "object", None),
            )
            object_name = getattr(object_node, "name", None)
            if not self.source_identifier_shadows_builtin(object_name):
                builtin_member_type = self.hip_compute_builtin_member_type(
                    object_name,
                    getattr(node, "member", ""),
                )
                if builtin_member_type is not None:
                    return builtin_member_type
            member_type = self.member_access_member_type(node)
            if member_type is not None:
                return member_type
        if isinstance(node, PointerAccessNode):
            member_type = self.pointer_access_member_type(node)
            if member_type is not None:
                return member_type
        if isinstance(node, ConstructorNode):
            constructor_type = self.type_name_string(
                getattr(node, "constructor_type", getattr(node, "vtype", None))
            )
            return (
                infer_enum_constructor_type(self, node)
                or infer_struct_constructor_type(self, node)
                or (
                    constructor_type
                    if constructor_type in HIP_SCALAR_CONSTRUCTOR_TYPE_ALIASES
                    else None
                )
            )
        if isinstance(node, MatchNode):
            return infer_match_expression_result_type(self, node)
        return super().expression_result_type(node)

    def dimension_query_spec(self, type_name):
        """Return HIP-specific resource metadata dimensions before shared specs."""
        base_type = self.resource_base_type(type_name)
        hip_specs = {
            "image1D": (("width",), False, False),
            "iimage1D": (("width",), False, False),
            "uimage1D": (("width",), False, False),
            "image1DArray": (("width", "elements"), False, False),
            "iimage1DArray": (("width", "elements"), False, False),
            "uimage1DArray": (("width", "elements"), False, False),
            "iimageCube": (("width", "height"), False, False),
            "uimageCube": (("width", "height"), False, False),
            "imageCubeArray": (("width", "height", "elements"), False, False),
            "iimageCubeArray": (("width", "height", "elements"), False, False),
            "uimageCubeArray": (("width", "height", "elements"), False, False),
        }
        spec = hip_specs.get(base_type)
        if spec is None:
            return super().dimension_query_spec(base_type)
        dimensions, mip, samples = spec
        return {"dimensions": dimensions, "mip": mip, "samples": samples}


def generate_hip_code(ast: ShaderNode) -> str:
    """Generate HIP source from a CrossGL shader AST."""
    generator = HipCodeGen()
    return generator.generate(ast)

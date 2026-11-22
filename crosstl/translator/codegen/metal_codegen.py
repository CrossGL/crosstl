"""CrossGL-to-Metal code generator."""

import re
from copy import deepcopy
from functools import cached_property

from ...backend.common_ast import AssignmentNode as BackendAssignmentNode
from ...backend.common_ast import ReturnNode as BackendReturnNode
from ...backend.common_ast import VariableNode as BackendVariableNode
from ..ast import (
    ArrayAccessNode,
    ArrayLiteralNode,
    ArrayNode,
    AssignmentNode,
    BinaryOpNode,
    BlockNode,
    BreakNode,
    ConstructorNode,
    ContinueNode,
    CooperativeMatrixOpNode,
    DoWhileNode,
    ForInNode,
    ForNode,
    FunctionCallNode,
    IdentifierNode,
    IfNode,
    LoopNode,
    MatchNode,
    MemberAccessNode,
    MeshOpNode,
    PointerAccessNode,
    PointerReinterpretNode,
    PointerType,
    RangeNode,
    RayQueryOpNode,
    RayTracingOpNode,
    ReferenceType,
    ReturnNode,
    StageMap,
    SwitchNode,
    TernaryOpNode,
    UnaryOpNode,
    VariableNode,
    WaveOpNode,
    WhileNode,
)
from ..refusals import MetalRefusal
from ..resource_identity import resource_identity_marker
from ..semantics.unsized_arrays import UnsizedArrayUnsupportedError
from ..structure_conversions import (
    StructureConversionKind,
    StructureFieldValue,
    registered_structure_conversion_for_identity,
)
from ..unhandled import UNHANDLED
from .array_utils import (
    format_c_style_array_declaration,
    get_array_size_from_node,
)
from .assignment_values import assignment_value_expression
from .comma_expressions import (
    c_family_comma_expression,
    comma_result_operand,
    is_comma_expression,
)
from .enum_utils import (
    infer_enum_constructor_type,
)
from .generic_struct_utils import (
    infer_struct_constructor_type,
)
from .match_utils import (
    generate_match_expression_assignment,
    infer_match_expression_result_type,
)
from .metal_argument_tables import metal_entry_argument_table
from .metal_array_arguments import MetalArrayArguments
from .metal_array_copies import generate_metal_array_initializer
from .metal_atomic_addresses import metal_vector_component_address
from .metal_buffer_scalars import metal_buffer_scalar_type
from .metal_call_expressions import MetalCallExpressionMixin
from .metal_call_lowering import MetalCallLoweringMixin
from .metal_constant_initializers import (
    metal_constant_initializer,
    restore_static_initializer_helpers,
)
from .metal_entry_return import (
    metal_entry_return_semantic_key,
)
from .metal_expression_forms import MetalExpressionFormMixin
from .metal_feature_errors import UnsupportedMetalFeatureError
from .metal_function_dependencies import MetalFunctionDependencyMixin
from .metal_function_parts import MetalFunctionPartsMixin
from .metal_function_state import (
    METAL_FUNCTION_BODY_STATE_ATTRIBUTES,
    METAL_FUNCTION_RESOURCE_PARAMETER_ATTRIBUTES,
    METAL_FUNCTION_STATE_ATTRIBUTES,
    restore_metal_function_state,
    save_metal_function_state,
)
from .metal_glsl_buffer_blocks import MetalGLSLBufferBlockMixin
from .metal_hlsl_functions import MetalHlslFunctionMixin
from .metal_hlsl_resources import MetalHlslResourceMixin
from .metal_hlsl_values import MetalHlslValueMixin
from .metal_keyword_identifiers import METAL_KEYWORD_IDENTIFIERS
from .metal_local_declarations import MetalLocalDeclarationMixin
from .metal_matrix_builtins import MetalMatrixBuiltins
from .metal_mesh_outputs import MetalMeshOutputMixin
from .metal_narrow_vectors import MetalNarrowVectorMixin
from .metal_parameter_address_spaces import MetalParameterAddressSpaceMixin
from .metal_post_tessellation import MetalPostTessellationMixin
from .metal_program_entries import MetalProgramEntryMixin
from .metal_program_scope_variables import MetalProgramScopeVariableMixin
from .metal_program_sections import MetalProgramSectionMixin
from .metal_program_state import MetalProgramStateMixin
from .metal_ray_tracing import MetalRayTracingMixin
from .metal_reference_argument_conditions import generate_header_writeback_statement
from .metal_reference_argument_order import generate_ordered_writeback_statement
from .metal_reference_argument_returns import generate_return_writeback_statement
from .metal_reference_call_results import generate_call_result_statement
from .metal_reference_sequenced_calls import generate_conditional_call_statement
from .metal_refusals import metal_refusals
from .metal_render_walks import MetalRenderWalks
from .metal_resource_arguments import MetalResourceArgumentMixin
from .metal_result_types import MetalResultTypeMixin
from .metal_signed_builtins import (
    metal_signed_member_assignment,
)
from .metal_stage_entries import MetalStageEntryMixin
from .metal_stage_interfaces import MetalStageInterfaceMixin
from .metal_stage_io_structs import MetalStageIOStructMixin
from .metal_statements import MetalStatementMixin
from .metal_texel_buffers import (
    METAL_TEXEL_BUFFER_TYPES,
    TEXEL_BUFFER_TYPES,
    MetalTexelBuffers,
)
from .metal_texture_operations import MetalTextureOperationMixin
from .metal_texture_validation import MetalTextureValidationMixin
from .metal_threadgroup_forwarding import (
    METAL_GROUPSHARED_ENTRY_STAGES,
)
from .metal_type_mapping import MetalTypeMappingMixin
from .metal_wave_operations import MetalWaveOperationMixin
from .metal_wide_atomics import (
    discarded_wide_atomic_call,
    is_metal_wide_atomic,
    metal_wide_atomic_call,
)
from .stage_io_semantic_locations import SEMANTIC_LOCATIONS
from .stage_utils import (
    collect_stage_entry_records,
    function_stage_name,
    normalize_stage_name,
)

FRAGMENT_INVOCATION_DENSITY_CAPABILITIES = (
    "glsl.extension.GL_EXT_fragment_invocation_density",
    "glsl.builtin.gl_FragSizeEXT",
)

# The value types of an MSL colour attachment output: a scalar or a vector of up to four components.
METAL_COLOR_ATTACHMENT_TYPE = re.compile(
    r"(?:float|half|int|uint|short|ushort|char|uchar)[234]?"
)


class MetalStructConversionError(ValueError):
    """Raised when a registered source structure conversion is not faithful."""

    project_diagnostic_code = "project.translate.metal-struct-construction-unsupported"
    missing_capabilities = ("metal.struct-conversion-construction",)

    def __init__(
        self,
        message,
        *,
        destination_type=None,
        source_type=None,
        conversion_kind=None,
        reason=None,
        source_location=None,
    ):
        super().__init__(message)
        self.destination_type = destination_type
        self.source_type = source_type
        self.conversion_kind = conversion_kind
        self.reason = reason
        self.source_location = source_location


class MetalEntryPointSelectionError(ValueError):
    """Raised when a requested standalone Metal entry cannot be selected."""

    project_diagnostic_code = "project.translate.metal-entry-point-unavailable"
    missing_capabilities = ("artifact.entry-point-selection",)

    def __init__(
        self,
        message,
        *,
        entry_point=None,
        available_entry_points=None,
        reason=None,
        stage=None,
        source_location=None,
    ):
        super().__init__(message)
        self.entry_point = entry_point
        self.available_entry_points = tuple(available_entry_points or ())
        self.reason = reason
        self.stage = stage
        self.source_location = source_location


class MetalCodeGen(
    MetalProgramEntryMixin,
    MetalStageIOStructMixin,
    MetalStageInterfaceMixin,
    MetalLocalDeclarationMixin,
    MetalStatementMixin,
    MetalWaveOperationMixin,
    MetalRayTracingMixin,
    MetalCallLoweringMixin,
    MetalParameterAddressSpaceMixin,
    MetalStageEntryMixin,
    MetalMeshOutputMixin,
    MetalFunctionDependencyMixin,
    MetalGLSLBufferBlockMixin,
    MetalResourceArgumentMixin,
    MetalTextureValidationMixin,
    MetalTextureOperationMixin,
    MetalTypeMappingMixin,
    MetalHlslResourceMixin,
    MetalHlslValueMixin,
    MetalHlslFunctionMixin,
    MetalNarrowVectorMixin,
    MetalProgramScopeVariableMixin,
    MetalProgramStateMixin,
    MetalProgramSectionMixin,
    MetalFunctionPartsMixin,
    MetalPostTessellationMixin,
    MetalExpressionFormMixin,
    MetalCallExpressionMixin,
    MetalResultTypeMixin,
):
    """Emit Metal Shading Language from the shared CrossGL translator AST."""

    # The type names the program being rendered declares, and the foreign built-in type names its rendering has
    # written so far (``None`` outside a rendering); see :func:`written_type_name`.
    program_type_names = frozenset()
    written_foreign_type_names = None
    # The body walks a render records once and replays; see :mod:`.metal_render_walks`.
    metal_render_walks = cached_property(lambda self: MetalRenderWalks())

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
    # Numeric grouping determines floating rounding and integer promotion.
    ASSOCIATIVE_BINARY_OPS = {"&&", "||"}
    # Higher-precedence children that clang's -Wlogical-op-parentheses,
    # -Wbitwise-op-parentheses, -Wparentheses (bitwise over comparison) and
    # -Wshift-op-parentheses reject under -Werror when left unparenthesized.
    _COMPARISON_OPS = frozenset({"==", "!=", "<", ">", "<=", ">="})
    CLANG_PARENTHESIZED_BINARY_CHILDREN = {
        "||": frozenset({"&&"}),
        "|": frozenset({"&", "^"}) | _COMPARISON_OPS,
        "^": frozenset({"&"}) | _COMPARISON_OPS,
        "&": _COMPARISON_OPS,
        "<<": frozenset({"+", "-"}),
        ">>": frozenset({"+", "-"}),
    }
    METAL_ATOMIC_FENCE_MEMORY_FLAGS = frozenset(
        {
            "mem_none",
            "mem_device",
            "mem_threadgroup",
            "mem_texture",
            "mem_threadgroup_imageblock",
            "mem_object_data",
        }
    )
    METAL_ATOMIC_FENCE_MEMORY_ORDERS = frozenset(
        {
            "memory_order_relaxed",
            "memory_order_acquire",
            "memory_order_release",
            "memory_order_acq_rel",
            "memory_order_seq_cst",
        }
    )
    METAL_ATOMIC_FENCE_THREAD_SCOPES = frozenset(
        {
            "thread_scope_thread",
            "thread_scope_simdgroup",
            "thread_scope_threadgroup",
            "thread_scope_device",
            "thread_scope_system",
        }
    )
    METAL_COOPERATIVE_MATRIX_FUNCTIONS = {
        "load": "simdgroup_load",
        "store": "simdgroup_store",
        "multiply_accumulate": "simdgroup_multiply_accumulate",
    }
    METAL_RESERVED_LOCAL_IDENTIFIERS = {
        "kernel",
        "vertex",
        "fragment",
        "compute",
        "intersection",
        "anyhit",
        "closesthit",
        "miss",
        "callable",
        "mesh",
        "object",
        "amplification",
    } | METAL_KEYWORD_IDENTIFIERS
    METAL_INTERPOLATION_ATTRIBUTES = {
        "center_no_perspective": "center_no_perspective",
        "center_perspective": "center_perspective",
        "centroid": "centroid_perspective",
        "centroid_no_perspective": "centroid_no_perspective",
        "centroid_perspective": "centroid_perspective",
        "flat": "flat",
        "linear_centroid": "centroid_perspective",
        "linear_noperspective": "center_no_perspective",
        "linear_noperspective_centroid": "centroid_no_perspective",
        "linear_sample": "sample_perspective",
        "nointerpolation": "flat",
        "noperspective": "center_no_perspective",
        "sample": "sample_perspective",
        "sample_no_perspective": "sample_no_perspective",
        "sample_perspective": "sample_perspective",
    }
    # MSL has only dfdx, dfdy and fwidth and does not let a shader pick fine or
    # coarse derivatives, so the GLSL fine and coarse forms lower to the plain
    # functions, whose granularity GLSL also leaves to the implementation. The
    # ddx_fine and fwidth_fine spellings are the ones the GLSL importer writes.
    METAL_DERIVATIVE_FUNCTION_ALIASES = {
        "ddx": "dfdx",
        "dFdx": "dfdx",
        "dFdxCoarse": "dfdx",
        "dFdxFine": "dfdx",
        "ddx_coarse": "dfdx",
        "ddx_fine": "dfdx",
        "ddy": "dfdy",
        "dFdy": "dfdy",
        "dFdyCoarse": "dfdy",
        "dFdyFine": "dfdy",
        "ddy_coarse": "dfdy",
        "ddy_fine": "dfdy",
        "fwidthCoarse": "fwidth",
        "fwidthFine": "fwidth",
        "fwidth_coarse": "fwidth",
        "fwidth_fine": "fwidth",
    }
    # Metal's native `abs` overload set does not accept bfloat directly: the
    # simultaneous half/float/integer conversions are ambiguous.  Promote only
    # the source bfloat arguments for operations with this proven native gap;
    # do not make half/float -> bfloat a general implicit conversion.
    METAL_BFLOAT_ARGUMENT_PROMOTED_FUNCTIONS = {"abs", "max", "min"}
    METAL_STDLIB_BUILTIN_FUNCTIONS = {
        "abs",
        "acos",
        "asin",
        "atan",
        "atan2",
        "ceil",
        "clamp",
        "cos",
        "cross",
        "distance",
        "dot",
        "exp",
        "exp2",
        "faceforward",
        "floor",
        "fract",
        "length",
        "log",
        "log2",
        "max",
        "min",
        "mix",
        "normalize",
        "pow",
        "reflect",
        "refract",
        "round",
        "rsqrt",
        "saturate",
        "sign",
        "signbit",
        "sin",
        "smoothstep",
        "sqrt",
        "step",
        "tan",
        "trunc",
    }
    METAL_BITCAST_FUNCTION_TARGETS = {
        "asfloat": "float",
        "asint": "int",
        "asuint": "uint",
        "floatBitsToInt": "int",
        "floatBitsToUint": "uint",
        "intBitsToFloat": "float",
        "uintBitsToFloat": "float",
    }
    METAL_INTEGER_BIT_FUNCTION_TARGETS = {
        "bitCount": "popcount",
        "bitfieldReverse": "reverse_bits",
        "countbits": "popcount",
        "reversebits": "reverse_bits",
    }
    METAL_WAVE_INTRINSIC_ARITIES = {
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
        "WavePrefixCountBits": 1,
        "WaveMatch": 1,
        "WaveMultiPrefixSum": 2,
        "WaveMultiPrefixCountBits": 2,
        "WaveMultiPrefixProduct": 2,
        "WaveMultiPrefixBitAnd": 2,
        "WaveMultiPrefixBitOr": 2,
        "WaveMultiPrefixBitXor": 2,
        "QuadActiveSum": 1,
        "QuadReadAcrossX": 1,
        "QuadReadAcrossY": 1,
        "QuadReadAcrossDiagonal": 1,
        "QuadReadLaneAt": 2,
        "QuadAny": 1,
        "QuadAll": 1,
    }
    METAL_WAVE_DIRECT_MAPPINGS = {
        "WaveActiveSum": "simd_sum",
        "WaveActiveProduct": "simd_product",
        "WaveActiveBitAnd": "simd_and",
        "WaveActiveBitOr": "simd_or",
        "WaveActiveBitXor": "simd_xor",
        "WaveActiveMin": "simd_min",
        "WaveActiveMax": "simd_max",
        "WaveActiveAllTrue": "simd_all",
        "WaveActiveAnyTrue": "simd_any",
        "WaveReadLaneFirst": "simd_broadcast_first",
        "WaveShuffleDown": "simd_shuffle_down",
        "WaveShuffleUp": "simd_shuffle_up",
        "WaveShuffleAndFillUp": "simd_shuffle_and_fill_up",
        "WaveShuffleXor": "simd_shuffle_xor",
        "WavePrefixSum": "simd_prefix_exclusive_sum",
        "WavePrefixProduct": "simd_prefix_exclusive_product",
        "WavePrefixInclusiveSum": "simd_prefix_inclusive_sum",
        "WavePrefixInclusiveProduct": "simd_prefix_inclusive_product",
        "QuadActiveSum": "quad_sum",
        "QuadAny": "quad_any",
        "QuadAll": "quad_all",
    }
    METAL_WAVE_UNSUPPORTED_OPERATIONS = {}
    METAL_WAVE_MULTI_PREFIX_INTRINSICS = {
        "WaveMultiPrefixSum",
        "WaveMultiPrefixCountBits",
        "WaveMultiPrefixProduct",
        "WaveMultiPrefixBitAnd",
        "WaveMultiPrefixBitOr",
        "WaveMultiPrefixBitXor",
    }
    METAL_WAVE_MULTI_PREFIX_NUMERIC_INTRINSICS = {
        "WaveMultiPrefixSum",
        "WaveMultiPrefixProduct",
    }
    METAL_WAVE_MULTI_PREFIX_INTEGER_INTRINSICS = {
        "WaveMultiPrefixBitAnd",
        "WaveMultiPrefixBitOr",
        "WaveMultiPrefixBitXor",
    }
    METAL_WAVE_MULTI_PREFIX_HELPERS = {
        "WaveMultiPrefixSum": "__crossgl_metal_wave_multi_prefix_sum",
        "WaveMultiPrefixCountBits": "__crossgl_metal_wave_multi_prefix_count_bits",
        "WaveMultiPrefixProduct": "__crossgl_metal_wave_multi_prefix_product",
        "WaveMultiPrefixBitAnd": "__crossgl_metal_wave_multi_prefix_bit_and",
        "WaveMultiPrefixBitOr": "__crossgl_metal_wave_multi_prefix_bit_or",
        "WaveMultiPrefixBitXor": "__crossgl_metal_wave_multi_prefix_bit_xor",
    }
    METAL_WAVE_BOOL_ARGUMENT_INTRINSICS = {
        "WaveActiveAllTrue",
        "WaveActiveAnyTrue",
        "WaveActiveBallot",
        "WaveActiveCountBits",
        "WavePrefixCountBits",
        "QuadAny",
        "QuadAll",
    }
    METAL_WAVE_NUMERIC_VALUE_INTRINSICS = {
        "QuadActiveSum",
        "WaveActiveSum",
        "WaveActiveProduct",
        "WaveActiveMin",
        "WaveActiveMax",
        "WavePrefixSum",
        "WavePrefixProduct",
        "WavePrefixInclusiveSum",
        "WavePrefixInclusiveProduct",
    }
    METAL_WAVE_INTEGER_VALUE_INTRINSICS = {
        "WaveActiveBitAnd",
        "WaveActiveBitOr",
        "WaveActiveBitXor",
    }
    METAL_WAVE_SIMDGROUP_VALUE_INTRINSICS = {
        "WaveReadLaneAt",
        "WaveReadLaneFirst",
        "WaveActiveAllEqual",
        "WaveShuffleDown",
        "WaveShuffleUp",
        "WaveShuffleXor",
        "QuadReadAcrossX",
        "QuadReadAcrossY",
        "QuadReadAcrossDiagonal",
        "QuadReadLaneAt",
    }
    METAL_WAVE_SHUFFLE_AND_FILL_INTRINSICS = {"WaveShuffleAndFillUp"}
    METAL_WAVE_UINT_RESULT_INTRINSICS = {
        "WaveGetLaneCount",
        "WaveGetLaneIndex",
        "WaveActiveCountBits",
        "WavePrefixCountBits",
        "WaveMultiPrefixCountBits",
    }
    METAL_WAVE_BOOL_RESULT_INTRINSICS = {
        "WaveIsFirstLane",
        "WaveActiveAllTrue",
        "WaveActiveAnyTrue",
        "WaveActiveAllEqual",
        "QuadAny",
        "QuadAll",
    }
    METAL_WAVE_UINT4_RESULT_INTRINSICS = {
        "WaveActiveBallot",
        "WaveMatch",
    }
    METAL_WAVE_VALUE_RESULT_INTRINSICS = (
        METAL_WAVE_NUMERIC_VALUE_INTRINSICS
        | METAL_WAVE_INTEGER_VALUE_INTRINSICS
        | METAL_WAVE_SIMDGROUP_VALUE_INTRINSICS
        | METAL_WAVE_SHUFFLE_AND_FILL_INTRINSICS
        | {
            "WaveMultiPrefixSum",
            "WaveMultiPrefixProduct",
            "WaveMultiPrefixBitAnd",
            "WaveMultiPrefixBitOr",
            "WaveMultiPrefixBitXor",
        }
    )
    METAL_WAVE_NUMERIC_COMPONENT_TYPES = {"float", "half", "int", "uint"}
    METAL_WAVE_INTEGER_COMPONENT_TYPES = {"int", "uint"}
    METAL_GLSL_SUBGROUP_LANE_BUILTINS = {
        "gl_SubgroupInvocationID": (
            "thread_index_in_simdgroup",
            "uint",
        ),
        "gl_SubgroupSize": (
            "threads_per_simdgroup",
            "uint",
        ),
        "gl_SubgroupID": (
            "simdgroup_index_in_threadgroup",
            "uint",
        ),
        "gl_NumSubgroups": (
            "simdgroups_per_threadgroup",
            "uint",
        ),
    }
    METAL_RAY_FLAG_VALUES = {
        "RAY_FLAG_NONE": 0x00,
        "RAY_FLAG_FORCE_OPAQUE": 0x01,
        "RAY_FLAG_FORCE_NON_OPAQUE": 0x02,
        "RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH": 0x04,
        "RAY_FLAG_SKIP_CLOSEST_HIT_SHADER": 0x08,
        "RAY_FLAG_CULL_BACK_FACING_TRIANGLES": 0x10,
        "RAY_FLAG_CULL_FRONT_FACING_TRIANGLES": 0x20,
        "RAY_FLAG_CULL_OPAQUE": 0x40,
        "RAY_FLAG_CULL_NON_OPAQUE": 0x80,
        "RAY_FLAG_SKIP_TRIANGLES": 0x100,
        "RAY_FLAG_SKIP_PROCEDURAL_PRIMITIVES": 0x200,
    }
    METAL_RAY_QUERY_RETURN_TYPES = {
        "Proceed": "bool",
        "Abort": "void",
        "CommitNonOpaqueTriangleHit": "void",
        "CommitProceduralPrimitiveHit": "void",
        "TraceRayInline": "void",
        "CandidateType": "uint",
        "CommittedStatus": "uint",
        "CandidatePrimitiveIndex": "uint",
        "CommittedPrimitiveIndex": "uint",
        "CandidateInstanceID": "uint",
        "CommittedInstanceID": "uint",
        "CandidateInstanceIndex": "uint",
        "CommittedInstanceIndex": "uint",
        "CandidateGeometryIndex": "uint",
        "CommittedGeometryIndex": "uint",
        "CandidateInstanceContributionToHitGroupIndex": "uint",
        "CommittedInstanceContributionToHitGroupIndex": "uint",
        "RayFlags": "uint",
        "CandidateObjectRayOrigin": "float3",
        "CandidateObjectRayDirection": "float3",
        "CommittedObjectRayOrigin": "float3",
        "CommittedObjectRayDirection": "float3",
        "WorldRayOrigin": "float3",
        "WorldRayDirection": "float3",
        "CandidateTriangleRayT": "float",
        "CommittedRayT": "float",
        "CandidateObjectRayTMin": "float",
        "RayTMin": "float",
        "CandidateTriangleBarycentrics": "float2",
        "CommittedTriangleBarycentrics": "float2",
        "CandidateTriangleFrontFace": "bool",
        "CommittedTriangleFrontFace": "bool",
        "CandidateProceduralPrimitiveNonOpaque": "bool",
        "CandidateObjectToWorld3x4": "float3x4",
        "CandidateWorldToObject3x4": "float3x4",
        "CommittedObjectToWorld3x4": "float3x4",
        "CommittedWorldToObject3x4": "float3x4",
        "CandidateTriangleVertexPositions": "void",
        "CommittedTriangleVertexPositions": "void",
    }

    def __init__(self):
        """Initialize Metal type maps and per-generation resource state."""
        self.refusals = metal_refusals()
        self.current_shader = None
        self.vertex_item = None
        self.fragment_item = None
        self.gl_position = False
        self.texture_variables = []
        self.metal_device_coherent_textures = frozenset()
        self.metal_inherited_buffer_contracts = {}
        self.acceleration_structure_variables = []
        self.visible_function_table_variables = []
        self.intersection_function_table_variables = []
        self.unsupported_metal_acceleration_structure_array_variables = {}
        self.unsupported_metal_ray_function_table_array_variables = {}
        self.sampler_variables = []
        self.structured_buffer_variables = []
        self.structured_buffer_length_variables = []
        self.structured_buffer_counter_variables = []
        self.metal_buffer_resource_variables = []
        self.metal_resource_binding_indices_by_id = {}
        self.metal_deferred_register_bindings = []
        self.metal_hlsl_register_bindings_by_id = {}
        self.metal_source_binding_stage_by_id = {}
        # The argument tables of the declaration whose binding is being reserved (metal_argument_tables.py).
        self.metal_binding_tables = None
        self.current_metal_argument_table = None
        self.metal_program_scope_value_globals = set()
        self.metal_program_scope_groupshared_globals = set()
        self.metal_lowered_program_scope_groupshared_globals_by_function = {}
        self.metal_lowered_program_scope_groupshared_global_ids = set()
        self.metal_forwarded_thread_private_globals = {}
        self.metal_program_scope_value_global_types = {}
        self.program_scope_sampler_constants = set()
        self.metal_specialization_macro_constants = set()
        self.metal_type_aliases = {}
        self.cbuffer_variables = []
        self.cbuffer_binding_indices = {}
        self.cbuffer_parameter_names = {}
        self.cbuffer_member_references = {}
        self.ambiguous_cbuffer_members = set()
        self.metal_hlsl_cbuffer_plans = {}
        self.metal_hlsl_struct_plans = {}
        self.metal_hlsl_structured_buffers = {}
        self.metal_hlsl_cbuffer_array_members = {}
        self.metal_runtime_initialized_constants = set()
        self.metal_emulated_matrix_shapes = set()
        self.metal_prelude_insertion_index = 0
        self.hlsl_program_constant_global_ids = set()
        self.cbuffers_by_name = {}
        self.user_function_names = set()
        self.function_parameter_names = {}
        self.function_parameter_infos = {}
        self.function_parameter_nodes = {}
        self.functions_by_name = {}
        self.function_overloads_by_name = {}
        self.metal_skipped_function_parameter_indices = {}
        self.function_return_types = {}
        # Locals of the current function that must be ``array<T, N>`` values
        # because the function returns them as its array result.
        self.current_metal_value_array_locals = frozenset()
        self.function_image_access_requirements = {}
        self.function_cbuffer_dependencies = {}
        self.function_global_resource_dependencies = {}
        self.function_stage_parameter_dependencies = {}
        self.function_stage_output_dependencies = {}
        self.function_metal_wave_lane_dependencies = {}
        self.function_metal_wave_lane_parameter_names = {}
        self.unsupported_glsl_buffer_block_functions = {}
        self.unsupported_glsl_buffer_block_struct_names = set()
        self.current_sampler_parameters = set()
        self.current_sampler_parameter_array_sizes = {}
        self.texture_variable_types = {}
        self.metal_texel_buffers = MetalTexelBuffers(self)
        self.texture_variable_raw_types = {}
        self.current_texture_parameters = {}
        self.current_texture_parameter_raw_types = {}
        self.current_texture_parameter_array_sizes = {}
        self.current_texture_alias_sources = {}
        self.image_variable_formats = {}
        self.current_image_format_parameters = {}
        self.current_structured_buffer_length_parameters = {}
        self.current_structured_buffer_counter_parameters = {}
        self.function_structured_buffer_length_dependencies = {}
        self.global_structured_buffer_length_dependencies = set()
        self.resource_array_size_hints = {}
        self.function_resource_array_size_hints = {}
        self.literal_int_constants = {}
        self.current_function_name = None
        self.current_function_return_type = None
        self.current_function_return_wrapper = None
        self.current_expression_expected_type = None
        self.suppress_image_load_component_suffix = False
        self.current_generic_function_substitutions = {}
        self.local_variable_types = {}
        self.current_address_space_variables = {}
        self.current_metal_native_width_local_array_declaration_ids = set()
        self.current_local_identifier_remaps = {}
        self.struct_member_types = {}
        self.struct_member_address_spaces = {}
        self.metal_static_struct_members = {}
        self.structs_by_name = {}
        self.metal_generated_struct_names = set()
        self.metal_union_layouts = {}
        self.generic_struct_definitions = {}
        self.generic_struct_specializations = {}
        self.generic_function_definitions = {}
        self.generic_function_specializations = {}
        self.generic_function_specialized_names = {}
        self.current_generic_function_substitutions = {}
        self.struct_constructor_uses_braces = True
        self.glsl_buffer_block_struct_names = set()
        self.lowered_glsl_buffer_blocks = {}
        self.required_buffer_atomic_compare_helpers = set()
        self.required_metal_wave_ballot_helper = False
        self.metal_wave_extension_helpers = {}
        self.metal_exponent_helpers = {}
        self.metal_texture_residency_helpers = {}
        self.required_metal_wave_match_helper = False
        self.required_metal_wave_mask_contains_helper = False
        self.required_metal_wave_multi_prefix_helpers = set()
        self.required_metal_inverse_helpers = set()
        self.metal_matrix_builtins = MetalMatrixBuiltins(self)
        self.metal_array_arguments = MetalArrayArguments(self)
        self.required_metal_integer_dot_helper = False
        self.metal_argument_substitutions = {}
        self.metal_writeback_statement_ids = set()
        self.metal_sequenced_call_ids = set()
        self.metal_writeback_temporary_count = 0
        self.required_metal_ray_query_runtime = False
        self.required_metal_ray_desc_runtime = False
        self.required_metal_ray_query_helpers = set()
        self.current_unused_local_declaration_names = set()
        self.current_metal_wave_lane_index_parameter = None
        self.current_metal_wave_lane_count_parameter = None
        self.current_metal_graphics_builtin_parameter_names = {}
        self.current_metal_compute_builtin_parameter_names = {}
        self.required_glsl_buffer_aggregate_load_helpers = {}
        self.unsupported_glsl_buffer_block_variables = set()
        self.unsupported_glsl_buffer_block_variable_types = {}
        self.current_glsl_buffer_block_parameters = {}
        self.current_structured_buffer_length_parameters = {}
        self.current_structured_buffer_counter_parameters = {}
        self.current_unsupported_glsl_buffer_block_parameters = set()
        self.current_unsupported_glsl_buffer_block_local_variables = set()
        self.current_glsl_buffer_block_parameter_failures = {}
        self.current_glsl_buffer_block_parameter_struct_failures = {}
        self.current_metal_mesh_output_config = None
        self.current_metal_mesh_output_parameter = None
        self.current_metal_mesh_grid_properties_parameter = None
        self.current_metal_mesh_payload_parameter = None
        self.current_metal_mesh_payload_type = None
        self.current_metal_mesh_output_accumulators = {}
        self.current_metal_non_thread_payload_parameters = set()
        self.current_readonly_metal_mesh_payload_parameters = set()
        self.current_readonly_metal_mesh_payload_reasons = {}
        self.current_readonly_raw_buffer_parameters = set()
        self.current_readonly_metal_parameters = set()
        self.current_readonly_metal_parameter_reasons = {}
        self.metal_ray_payload_parameter_types = set()
        self.metal_program_mesh_payload_type = None
        self.function_metal_mesh_dispatch_contexts = {}
        self.function_metal_mesh_dispatch_contexts_by_id = {}
        self.lowered_glsl_buffer_block_struct_names = set()
        self.glsl_buffer_block_lowering_failures = {}
        self.glsl_buffer_block_struct_lowering_failures = {}
        self.glsl_buffer_block_variables = []
        self.metal_temp_variable_index = 0
        self.generated_return_wrapper_struct_names = set()
        self.metal_vertex_entry_input_struct_names = set()
        self.metal_vertex_entry_output_struct_names = set()
        self.metal_mesh_primitive_struct_names = set()
        self.metal_fragment_entry_input_struct_names = set()
        self.metal_fragment_entry_output_struct_names = set()
        self.metal_stage_io_struct_names = set()
        self.metal_stage_io_member_lowerings = {}
        self.metal_signed_builtin_members = {}
        self.metal_signed_member_raw_reads = set()
        self.type_mapping = {
            # Scalar Types
            "void": "void",
            "short": "int",
            "signed short": "int",
            "unsigned short": "uint",
            "int": "int",
            "signed int": "int",
            "unsigned int": "uint",
            "long": "int64_t",
            "signed long": "int64_t",
            "unsigned long": "uint64_t",
            "ulong": "uint64_t",
            "f16": "half",
            "f32": "float",
            "f64": "double",
            "float": "float",
            "half": "half",
            "float16": "half",
            "float16_t": "half",
            "min16float": "half",
            "min10float": "half",
            "bfloat": "bfloat",
            "bfloat16": "bfloat",
            "bfloat16_t": "bfloat",
            "bfloat16vec2": "bfloat2",
            "bfloat16vec3": "bfloat3",
            "bfloat16vec4": "bfloat4",
            "i8": "int",
            "u8": "uint",
            "i16": "int",
            "u16": "uint",
            "i32": "int",
            "u32": "uint",
            "i64": "int64_t",
            "u64": "uint64_t",
            "str": "int",
            "char": "int",
            "signed char": "int",
            "int8": "int",
            "int8_t": "int",
            "uchar": "uint",
            "unsigned char": "uint",
            "uint8": "uint",
            "uint8_t": "uint",
            "int16": "int",
            "int16_t": "int",
            "int32_t": "int",
            "int64": "int64_t",
            "int64_t": "int64_t",
            "ptrdiff_t": "int64_t",
            "uint16": "uint",
            "uint16_t": "uint",
            "uint32_t": "uint",
            "uint64": "uint64_t",
            "uint64_t": "uint64_t",
            "size_t": "uint64_t",
            "min16int": "int",
            "min12int": "int",
            "min16uint": "uint",
            "bool": "bool",
            # Vector Types
            "vec2": "float2",
            "vec3": "float3",
            "vec4": "float4",
            "vec2<f16>": "half2",
            "vec3<f16>": "half3",
            "vec4<f16>": "half4",
            "vec2<f32>": "float2",
            "vec3<f32>": "float3",
            "vec4<f32>": "float4",
            "vec2<f64>": "double2",
            "vec3<f64>": "double3",
            "vec4<f64>": "double4",
            "vec2<i8>": "int2",
            "vec3<i8>": "int3",
            "vec4<i8>": "int4",
            "vec2<u8>": "uint2",
            "vec3<u8>": "uint3",
            "vec4<u8>": "uint4",
            "vec2<i16>": "int2",
            "vec3<i16>": "int3",
            "vec4<i16>": "int4",
            "vec2<u16>": "uint2",
            "vec3<u16>": "uint3",
            "vec4<u16>": "uint4",
            "vec2<i32>": "int2",
            "vec3<i32>": "int3",
            "vec4<i32>": "int4",
            "vec2<u32>": "uint2",
            "vec3<u32>": "uint3",
            "vec4<u32>": "uint4",
            "vec2<i64>": "long2",
            "vec3<i64>": "long3",
            "vec4<i64>": "long4",
            "vec2<u64>": "ulong2",
            "vec3<u64>": "ulong3",
            "vec4<u64>": "ulong4",
            "vec2<bool>": "bool2",
            "vec3<bool>": "bool3",
            "vec4<bool>": "bool4",
            "ivec2": "int2",
            "ivec3": "int3",
            "ivec4": "int4",
            "i64vec2": "long2",
            "i64vec3": "long3",
            "i64vec4": "long4",
            "short2": "int2",
            "short3": "int3",
            "short4": "int4",
            "ushort2": "uint2",
            "ushort3": "uint3",
            "ushort4": "uint4",
            "char2": "int2",
            "char3": "int3",
            "char4": "int4",
            "uchar2": "uint2",
            "uchar3": "uint3",
            "uchar4": "uint4",
            "packed_int2": "int2",
            "packed_int3": "int3",
            "packed_int4": "int4",
            "simd_int2": "int2",
            "simd_int3": "int3",
            "simd_int4": "int4",
            "packed_uint2": "uint2",
            "packed_uint3": "uint3",
            "packed_uint4": "uint4",
            "simd_uint2": "uint2",
            "simd_uint3": "uint3",
            "simd_uint4": "uint4",
            "int2": "int2",
            "int3": "int3",
            "int4": "int4",
            "uint2": "uint2",
            "uint3": "uint3",
            "uint4": "uint4",
            "uvec2": "uint2",
            "uvec3": "uint3",
            "uvec4": "uint4",
            "u64vec2": "ulong2",
            "u64vec3": "ulong3",
            "u64vec4": "ulong4",
            "float2": "float2",
            "float3": "float3",
            "float4": "float4",
            "packed_float2": "float2",
            "packed_float3": "float3",
            "packed_float4": "float4",
            "simd_float2": "float2",
            "simd_float3": "float3",
            "simd_float4": "float4",
            "half2": "half2",
            "half3": "half3",
            "half4": "half4",
            "packed_half2": "half2",
            "packed_half3": "half3",
            "packed_half4": "half4",
            "simd_half2": "half2",
            "simd_half3": "half3",
            "simd_half4": "half4",
            "f16vec2": "half2",
            "f16vec3": "half3",
            "f16vec4": "half4",
            "i8vec2": "int2",
            "i8vec3": "int3",
            "i8vec4": "int4",
            "i16vec2": "int2",
            "i16vec3": "int3",
            "i16vec4": "int4",
            "u8vec2": "uint2",
            "u8vec3": "uint3",
            "u8vec4": "uint4",
            "u16vec2": "uint2",
            "u16vec3": "uint3",
            "u16vec4": "uint4",
            "min16float2": "half2",
            "min16float3": "half3",
            "min16float4": "half4",
            "min10float2": "half2",
            "min10float3": "half3",
            "min10float4": "half4",
            "min16int2": "int2",
            "min16int3": "int3",
            "min16int4": "int4",
            "min12int2": "int2",
            "min12int3": "int3",
            "min12int4": "int4",
            "min16uint2": "uint2",
            "min16uint3": "uint3",
            "min16uint4": "uint4",
            "bvec2": "bool2",
            "bvec3": "bool3",
            "bvec4": "bool4",
            "bool2": "bool2",
            "bool3": "bool3",
            "bool4": "bool4",
            "sampler": "sampler",
            "comparison_sampler": "sampler",
            "sampler1D": "texture1d<float>",
            "sampler1DArray": "texture1d_array<float>",
            "sampler2D": "texture2d<float>",
            "sampler3D": "texture3d<float>",
            "samplerCube": "texturecube<float>",
            "sampler2DArray": "texture2d_array<float>",
            "samplerCubeArray": "texturecube_array<float>",
            "sampler2DMS": "texture2d_ms<float>",
            "sampler2DMSArray": "texture2d_ms_array<float>",
            "sampler2DShadow": "depth2d<float>",
            "sampler2DArrayShadow": "depth2d_array<float>",
            "samplerCubeShadow": "depthcube<float>",
            "samplerCubeArrayShadow": "depthcube_array<float>",
            "isampler1D": "texture1d<int>",
            "isampler1DArray": "texture1d_array<int>",
            "isampler2D": "texture2d<int>",
            "isampler3D": "texture3d<int>",
            "isamplerCube": "texturecube<int>",
            "isampler2DArray": "texture2d_array<int>",
            "isamplerCubeArray": "texturecube_array<int>",
            "isampler2DMS": "texture2d_ms<int>",
            "isampler2DMSArray": "texture2d_ms_array<int>",
            "usampler1D": "texture1d<uint>",
            "usampler1DArray": "texture1d_array<uint>",
            "usampler2D": "texture2d<uint>",
            "usampler3D": "texture3d<uint>",
            "usamplerCube": "texturecube<uint>",
            "usampler2DArray": "texture2d_array<uint>",
            "usamplerCubeArray": "texturecube_array<uint>",
            "usampler2DMS": "texture2d_ms<uint>",
            "usampler2DMSArray": "texture2d_ms_array<uint>",
            "accelerationStructureEXT": "instance_acceleration_structure",
            "RaytracingAccelerationStructure": "instance_acceleration_structure",
            "AccelerationStructure": "instance_acceleration_structure",
            "acceleration_structure": "instance_acceleration_structure",
            "instance_acceleration_structure": "instance_acceleration_structure",
            "primitive_acceleration_structure": "primitive_acceleration_structure",
            "visible_function_table": "visible_function_table",
            "visibleFunctionTable": "visible_function_table",
            "intersection_function_table": "intersection_function_table",
            "intersectionFunctionTable": "intersection_function_table",
            "iimage1D": "texture1d<int, access::read_write>",
            "iimage1DArray": "texture1d_array<int, access::read_write>",
            "iimage2D": "texture2d<int, access::read_write>",
            "iimage3D": "texture3d<int, access::read_write>",
            "iimage2DArray": "texture2d_array<int, access::read_write>",
            "iimage2DMS": "texture2d_ms<int, access::read>",
            "iimage2DMSArray": "texture2d_ms_array<int, access::read>",
            "uimage1D": "texture1d<uint, access::read_write>",
            "uimage1DArray": "texture1d_array<uint, access::read_write>",
            "uimage2D": "texture2d<uint, access::read_write>",
            "uimage3D": "texture3d<uint, access::read_write>",
            "uimage2DArray": "texture2d_array<uint, access::read_write>",
            "uimage2DMS": "texture2d_ms<uint, access::read>",
            "uimage2DMSArray": "texture2d_ms_array<uint, access::read>",
            "image1D": "texture1d<float, access::read_write>",
            "image1DArray": "texture1d_array<float, access::read_write>",
            "image2D": "texture2d<float, access::read_write>",
            "image3D": "texture3d<float, access::read_write>",
            "imageCube": "texture2d_array<float, access::read_write>",
            "image2DArray": "texture2d_array<float, access::read_write>",
            "image2DMS": "texture2d_ms<float, access::read>",
            "image2DMSArray": "texture2d_ms_array<float, access::read>",
            **METAL_TEXEL_BUFFER_TYPES,
            # Matrix Types
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
            "half2x2": "half2x2",
            "half2x3": "half2x3",
            "half2x4": "half2x4",
            "half3x2": "half3x2",
            "half3x3": "half3x3",
            "half3x4": "half3x4",
            "half4x2": "half4x2",
            "half4x3": "half4x3",
            "half4x4": "half4x4",
            "f16mat2": "half2x2",
            "f16mat3": "half3x3",
            "f16mat4": "half4x4",
            "f16mat2x2": "half2x2",
            "f16mat2x3": "half2x3",
            "f16mat2x4": "half2x4",
            "f16mat3x2": "half3x2",
            "f16mat3x3": "half3x3",
            "f16mat3x4": "half3x4",
            "f16mat4x2": "half4x2",
            "f16mat4x3": "half4x3",
            "f16mat4x4": "half4x4",
            "simd_float2x2": "float2x2",
            "simd_float2x3": "float2x3",
            "simd_float2x4": "float2x4",
            "simd_float3x2": "float3x2",
            "simd_float3x3": "float3x3",
            "simd_float3x4": "float3x4",
            "simd_float4x2": "float4x2",
            "simd_float4x3": "float4x3",
            "simd_float4x4": "float4x4",
            "simd_half2x2": "half2x2",
            "simd_half2x3": "half2x3",
            "simd_half2x4": "half2x4",
            "simd_half3x2": "half3x2",
            "simd_half3x3": "half3x3",
            "simd_half3x4": "half3x4",
            "simd_half4x2": "half4x2",
            "simd_half4x3": "half4x3",
            "simd_half4x4": "half4x4",
            "min16float2x2": "half2x2",
            "min16float2x3": "half2x3",
            "min16float2x4": "half2x4",
            "min16float3x2": "half3x2",
            "min16float3x3": "half3x3",
            "min16float3x4": "half3x4",
            "min16float4x2": "half4x2",
            "min16float4x3": "half4x3",
            "min16float4x4": "half4x4",
            "min10float2x2": "half2x2",
            "min10float2x3": "half2x3",
            "min10float2x4": "half2x4",
            "min10float3x2": "half3x2",
            "min10float3x3": "half3x3",
            "min10float3x4": "half3x4",
            "min10float4x2": "half4x2",
            "min10float4x3": "half4x3",
            "min10float4x4": "half4x4",
        }

        self.semantic_map = {
            # Vertex inputs
            "gl_VertexIndex": "vertex_id",
            "gl_VertexID": "vertex_id",
            "SV_VertexID": "vertex_id",
            "SV_VertexId": "vertex_id",
            "sv_vertex_id": "vertex_id",
            "sv_vertexid": "vertex_id",
            "gl_InstanceID": "instance_id",
            "gl_InstanceIndex": "instance_id",
            "SV_InstanceID": "instance_id",
            "SV_InstanceId": "instance_id",
            "sv_instance_id": "instance_id",
            "sv_instanceid": "instance_id",
            # GLSL 4.60 / GL_ARB_shader_draw_parameters; SV_Start*Location is HLSL SM 6.8.
            "gl_BaseVertex": "base_vertex",
            "gl_BaseVertexARB": "base_vertex",
            "SV_StartVertexLocation": "base_vertex",
            "gl_BaseInstance": "base_instance",
            "gl_BaseInstanceARB": "base_instance",
            "SV_StartInstanceLocation": "base_instance",
            "gl_IsFrontFace": "front_facing",
            "SV_IsFrontFace": "front_facing",
            "SV_IsFrontFacing": "front_facing",
            "sv_is_front_face": "front_facing",
            "sv_isfrontface": "front_facing",
            "gl_PrimitiveID": "primitive_id",
            "SV_PrimitiveID": "primitive_id",
            "SV_PrimitiveId": "primitive_id",
            "sv_primitive_id": "primitive_id",
            "sv_primitiveid": "primitive_id",
            # A mesh primitive's cull flag: a member of the mesh<> primitive structure.
            "gl_CullPrimitiveEXT": "primitive_culled",
            "SV_CullPrimitive": "primitive_culled",
            # The classic vertex semantics keep the location every target gives them.
            **{
                semantic: f"attribute({location})"
                for semantic, location in SEMANTIC_LOCATIONS.items()
            },
            # Vertex outputs
            "gl_Position": "position",
            "SV_POSITION": "position",
            "SV_Position": "position",
            "gl_PointSize": "point_size",
            "gl_ClipDistance": "clip_distance",
            "gl_Layer": "render_target_array_index",
            "gl_ViewportIndex": "viewport_array_index",
            # Vertex amplification is Metal's multiview (MSL 2.2 section 5.2.3.1).
            "gl_ViewIndex": "amplification_id",
            # Fragment inputs
            "gl_FragColor": "[[color(0)]]",
            "gl_FragColor0": "[[color(0)]]",
            "gl_FragColor1": "[[color(1)]]",
            "gl_FragColor2": "[[color(2)]]",
            "gl_FragColor3": "[[color(3)]]",
            "gl_FragColor4": "[[color(4)]]",
            "gl_FragColor5": "[[color(5)]]",
            "gl_FragColor6": "[[color(6)]]",
            "gl_FragColor7": "[[color(7)]]",
            "gl_FragDepth": "depth(any)",
            "gl_FragStencilRefEXT": "stencil",
            "SV_TARGET": "color(0)",
            "SV_TARGET0": "color(0)",
            "SV_TARGET1": "color(1)",
            "SV_TARGET2": "color(2)",
            "SV_TARGET3": "color(3)",
            "SV_TARGET4": "color(4)",
            "SV_TARGET5": "color(5)",
            "SV_TARGET6": "color(6)",
            "SV_TARGET7": "color(7)",
            "SV_Target": "color(0)",
            "SV_Target0": "color(0)",
            "SV_Target1": "color(1)",
            "SV_Target2": "color(2)",
            "SV_Target3": "color(3)",
            "SV_Target4": "color(4)",
            "SV_Target5": "color(5)",
            "SV_Target6": "color(6)",
            "SV_Target7": "color(7)",
            "SV_DEPTH": "depth(any)",
            "SV_Depth": "depth(any)",
            # Additional Metal-specific attributes
            "gl_FragCoord": "position",
            "gl_BaryCoordEXT": "barycentric_coord",
            "gl_BaryCoordNoPerspEXT": "barycentric_coord",
            "gl_BaryCoordNV": "barycentric_coord",
            "gl_BaryCoordNoPerspNV": "barycentric_coord",
            "gl_FrontFacing": "front_facing",
            "gl_PointCoord": "point_coord",
            "gl_SampleID": "sample_id",
            "SV_SampleIndex": "sample_id",
            "gl_SampleMask": "sample_mask",
            "gl_SampleMaskIn": "sample_mask",
            "SV_Coverage": "sample_mask",
            # Compute shader specific
            "gl_GlobalInvocationID": "thread_position_in_grid",
            "SV_DispatchThreadID": "thread_position_in_grid",
            "SV_DispatchThreadId": "thread_position_in_grid",
            "sv_dispatch_thread_id": "thread_position_in_grid",
            "sv_dispatchthreadid": "thread_position_in_grid",
            "gl_LocalInvocationID": "thread_position_in_threadgroup",
            "SV_GroupThreadID": "thread_position_in_threadgroup",
            "SV_GroupThreadId": "thread_position_in_threadgroup",
            "sv_group_thread_id": "thread_position_in_threadgroup",
            "sv_groupthreadid": "thread_position_in_threadgroup",
            "gl_WorkGroupID": "threadgroup_position_in_grid",
            "SV_GroupID": "threadgroup_position_in_grid",
            "SV_GroupId": "threadgroup_position_in_grid",
            "sv_group_id": "threadgroup_position_in_grid",
            "sv_groupid": "threadgroup_position_in_grid",
            "gl_LocalInvocationIndex": "thread_index_in_threadgroup",
            "SV_GroupIndex": "thread_index_in_threadgroup",
            "sv_group_index": "thread_index_in_threadgroup",
            "sv_groupindex": "thread_index_in_threadgroup",
            "gl_QuadGroupID": "quadgroup_index_in_threadgroup",
            "gl_QuadGroupInvocationID": "thread_index_in_quadgroup",
            "gl_WorkGroupSize": "threads_per_threadgroup",
            "gl_NumWorkGroups": "threadgroups_per_grid",
            "gl_SubgroupInvocationID": "thread_index_in_simdgroup",
            "gl_SubgroupSize": "threads_per_simdgroup",
            "gl_SubgroupID": "simdgroup_index_in_threadgroup",
            "gl_NumSubgroups": "simdgroups_per_threadgroup",
            # Ray tracing / payload semantics
            "payload": "payload",
            "mesh_payload": "payload",
            "hlsl_mesh_payload": "payload",
            "task_payload": "payload",
            "taskPayloadSharedEXT": "payload",
            "hit_attribute": "hit_attribute",
            "callable_data": "callable_data",
            "shader_record": "shader_record",
        }

    def entry_scoped_ast(self, ast, entry_point):
        if not isinstance(entry_point, str) or not entry_point.strip():
            raise MetalEntryPointSelectionError(
                "Metal entry-point selection requires a non-empty source entry name",
                entry_point=entry_point,
                reason="invalid-name",
            )
        entry_point = entry_point.strip()
        stage_entry_types = self.stage_entry_types()
        entries = collect_stage_entry_records(ast, None, stage_entry_types)
        available = sorted(
            {
                str(getattr(function, "name", ""))
                for _entry_id, _stage_name, function in entries
                if getattr(function, "name", None)
            }
        )
        matches = [
            (entry_id, stage_name, function)
            for entry_id, stage_name, function in entries
            if getattr(function, "name", None) == entry_point
        ]
        if not matches:
            raise MetalEntryPointSelectionError(
                f"Metal source entry point '{entry_point}' was not found",
                entry_point=entry_point,
                available_entry_points=available,
                reason="not-found",
            )
        if len(matches) != 1:
            raise MetalEntryPointSelectionError(
                f"Metal source entry point '{entry_point}' is ambiguous",
                entry_point=entry_point,
                available_entry_points=available,
                reason="ambiguous",
            )

        _selected_id, selected_stage_name, selected_function = matches[0]
        selected_stage_name = normalize_stage_name(selected_stage_name)
        if selected_stage_name != "compute":
            raise MetalEntryPointSelectionError(
                f"Metal standalone entry point '{entry_point}' must be a compute "
                f"entry; found stage '{selected_stage_name}'",
                entry_point=entry_point,
                available_entry_points=available,
                reason="unsupported-stage",
                stage=selected_stage_name,
                source_location=getattr(selected_function, "source_location", None),
            )
        selected_stage = next(
            (
                (stage_type, stage)
                for stage_type, stage in getattr(ast, "stages", {}).items()
                if getattr(stage, "entry_point", None) is selected_function
            ),
            None,
        )
        scoped_ast = deepcopy(ast)
        selected_copy_function = None
        scoped_ast.functions = [
            function
            for function in getattr(scoped_ast, "functions", []) or []
            if function_stage_name(function) not in stage_entry_types
            or (
                selected_stage is None
                and function_stage_name(function) == selected_stage_name
                and getattr(function, "name", None) == entry_point
            )
        ]
        selected_stages = StageMap()
        if selected_stage is not None:
            stage_type, stage = selected_stage
            selected_stage_copy = deepcopy(stage)
            selected_stages.append(stage_type, selected_stage_copy)
            selected_copy_function = selected_stage_copy.entry_point
        else:
            selected_copy_function = next(
                (
                    function
                    for function in scoped_ast.functions
                    if function_stage_name(function) == selected_stage_name
                    and getattr(function, "name", None) == entry_point
                ),
                None,
            )
        scoped_ast.stages = selected_stages
        if selected_copy_function is None:
            raise MetalEntryPointSelectionError(
                f"Metal source entry point '{entry_point}' could not be copied",
                entry_point=entry_point,
                available_entry_points=available,
                reason="copy-failed",
            )

        reachable_names = self.entry_reachable_function_names(
            scoped_ast,
            selected_copy_function,
        )
        scoped_ast.functions = [
            function
            for function in getattr(scoped_ast, "functions", []) or []
            if function is selected_copy_function
            or getattr(function, "name", None) in reachable_names
        ]
        for _stage_type, stage in scoped_ast.stages.items():
            stage.local_functions = [
                function
                for function in getattr(stage, "local_functions", []) or []
                if getattr(function, "name", None) in reachable_names
            ]
        self.prune_entry_scoped_declarations(scoped_ast, selected_copy_function)
        self.prune_entry_scoped_structs(scoped_ast, selected_copy_function)
        # Scoping drops conversion helpers no call reaches; the struct members
        # they came from are still needed (see
        # ``collect_metal_constant_conversion_operators``).
        scoped_ast.metal_constant_conversion_operators = (
            self.collect_metal_constant_conversion_operators(self.all_functions(ast))
        )
        restore_static_initializer_helpers(self, ast, scoped_ast)
        return scoped_ast

    def render_program(self, ast, target_stage=None):
        with self.metal_render_walks.suspended():
            ast, target_stage = self.rewrite_rendered_program(ast, target_stage)
        with self.metal_render_walks.replaying():
            return self.render_rewritten_program(ast, target_stage)

    def render_rewritten_program(self, ast, target_stage):
        """Render ``ast`` once its rewrites are done; generation only reads the AST from here on."""
        self.reset_metal_program_resource_state(ast)
        all_functions = self.collect_metal_program_function_tables(ast)
        self.reset_metal_program_parameter_state(ast)
        structs = self.collect_metal_program_type_tables(ast, target_stage)
        global_vars = self.collect_metal_program_global_tables(
            ast, target_stage, all_functions
        )
        self.reset_metal_program_generation_state()
        self.collect_metal_buffer_block_layouts(
            ast, all_functions, global_vars, structs
        )
        generic_function_specializations = (
            self.prepare_metal_generic_function_specializations(
                ast, all_functions, global_vars, structs
            )
        )
        code = self.render_metal_program_prelude(
            ast, all_functions, global_vars, structs
        )

        code = self.render_metal_program_scope_variables(code, global_vars)

        code, stage_entry_names = self.collect_metal_program_function_dependencies(
            ast,
            target_stage,
            all_functions,
            code,
            generic_function_specializations,
            global_vars,
        )

        functions_code = self.render_metal_program_functions(
            ast, target_stage, stage_entry_names
        )

        return self.append_metal_program_helpers(code, functions_code)

    def declare_metal_program_scope_variable(self, emission, node):
        """Declare one program-scope variable: a resource gets its binding, and a value gets its address space and initializer."""
        if self.is_spirv_stage_interface_layout(node):
            return

        resource_count, array_size, vtype, array_suffix = (
            self.metal_program_scope_variable_shape(node)
        )

        var_name = getattr(node, "name", getattr(node, "variable_name", None))
        if id(node) in self.hlsl_program_constant_global_ids:
            return
        if id(node) in self.metal_lowered_program_scope_groupshared_global_ids:
            return
        if self.is_metal_function_constant_variable(node):
            return

        handled = self.declare_metal_buffer_variable(
            emission, node, array_size, resource_count, var_name, vtype
        )
        if handled is not UNHANDLED:
            return handled

        handled = self.declare_metal_ray_tracing_variable(
            emission, node, array_size, resource_count, var_name, vtype
        )
        if handled is not UNHANDLED:
            return handled

        if self.is_structured_buffer_type(vtype):
            self.declare_metal_structured_buffer_variable(
                emission, node, array_size, resource_count, var_name, vtype
            )
        elif vtype in [
            "sampler1D",
            "sampler1DArray",
            "sampler2D",
            "sampler3D",
            "samplerCube",
            "sampler2DArray",
            "samplerCubeArray",
            "sampler2DMS",
            "sampler2DMSArray",
            "sampler2DShadow",
            "sampler2DArrayShadow",
            "samplerCubeShadow",
            "samplerCubeArrayShadow",
            "isampler1D",
            "isampler1DArray",
            "isampler2D",
            "isampler3D",
            "isamplerCube",
            "isampler2DArray",
            "isamplerCubeArray",
            "isampler2DMS",
            "isampler2DMSArray",
            "usampler1D",
            "usampler1DArray",
            "usampler2D",
            "usampler3D",
            "usamplerCube",
            "usampler2DArray",
            "usamplerCubeArray",
            "usampler2DMS",
            "usampler2DMSArray",
            "iimage1D",
            "iimage1DArray",
            "iimage2D",
            "iimage3D",
            "iimage2DArray",
            "iimage2DMS",
            "iimage2DMSArray",
            "uimage1D",
            "uimage1DArray",
            "uimage2D",
            "uimage3D",
            "uimage2DArray",
            "uimage2DMS",
            "uimage2DMSArray",
            "image1D",
            "image1DArray",
            "image2D",
            "image3D",
            "imageCube",
            "image2DArray",
            "image2DMS",
            "image2DMSArray",
            *TEXEL_BUFFER_TYPES,
        ]:
            self.declare_metal_texture_variable(
                emission, node, array_size, resource_count, var_name, vtype
            )
        elif self.resource_base_type(vtype) in {"sampler", "comparison_sampler"}:
            self.declare_metal_sampler_variable(
                emission, node, array_size, resource_count, var_name
            )
        elif self.is_input_attachment_type_name(vtype):
            self.refuse_lowering(
                f"input attachment declaration: '{var_name}' uses "
                f"{self.type_name_string(vtype)}; subpassInput resources require "
                "Vulkan subpass lowering"
            )
        else:
            return self.declare_metal_value_variable(
                emission, node, array_suffix, var_name, vtype
            )

    def metal_struct_alignment_error(self, node, reason, detail):
        struct_name = getattr(node, "name", "<anonymous>")
        raise UnsupportedMetalFeatureError(
            "struct-alignment-contract",
            f"Metal codegen cannot reconstruct alignment for struct "
            f"'{struct_name}': {detail}",
            missing_capabilities=("metal.struct-alignment-contract",),
            operation=struct_name,
            reason=reason,
            source_location=getattr(node, "source_location", None),
        )

    def metal_union_layout_error(self, node, reason, detail, member=None):
        union_name = getattr(node, "name", "<anonymous>")
        member_name = getattr(member, "name", None)
        subject = (
            f"member '{member_name}' of Metal union '{union_name}'"
            if member_name
            else f"Metal union '{union_name}'"
        )
        return UnsupportedMetalFeatureError(
            "union-layout-contract",
            f"Metal codegen cannot reconstruct {subject}: {detail}",
            missing_capabilities=("metal.union-layout-contract",),
            operation=union_name,
            reason=reason,
            source_location=getattr(member or node, "source_location", None),
        )

    def metal_static_struct_member_declaration(self, member):
        attributes = {
            self.normalized_metal_abi_attribute_name(attribute)
            for attribute in getattr(member, "attributes", []) or []
        }
        if "static" not in attributes:
            return None

        member_name = getattr(member, "name", None)
        if "constant" not in attributes:
            raise UnsupportedMetalFeatureError(
                "static-struct-member",
                "Metal codegen requires static struct members to reside in "
                "the constant address space",
                missing_capabilities=("metal.static-struct-constant-storage",),
                operation=member_name or "<unnamed>",
                reason="constant-address-space-required",
                source_location=getattr(member, "source_location", None),
            )

        default_value = getattr(member, "default_value", None)
        if not member_name or default_value is None:
            raise UnsupportedMetalFeatureError(
                "static-struct-member",
                "Metal codegen requires a static struct member to have a "
                "compile-time initializer",
                missing_capabilities=("metal.static-struct-member-initializer",),
                operation=member_name or "<unnamed>",
                reason="initializer-missing",
                source_location=getattr(member, "source_location", None),
            )

        raw_type = getattr(member, "member_type", getattr(member, "vtype", None))
        if self.is_array_type_node(raw_type):
            raise UnsupportedMetalFeatureError(
                "static-struct-member",
                f"Metal codegen cannot emit static array member '{member_name}'",
                missing_capabilities=("metal.static-struct-array-member",),
                operation=member_name,
                reason="array-static-member-unsupported",
                source_location=getattr(member, "source_location", None),
            )
        mapped_type = self.map_type(self.convert_type_node_to_string(raw_type))
        if "*" in mapped_type or "&" in mapped_type:
            raise UnsupportedMetalFeatureError(
                "static-struct-member",
                f"Metal codegen cannot emit static pointer member '{member_name}'",
                missing_capabilities=("metal.static-struct-pointer-member",),
                operation=member_name,
                reason="pointer-static-member-unsupported",
                source_location=getattr(member, "source_location", None),
            )

        # A member-wise constructor helper is not a constant expression.
        initializer = metal_constant_initializer(
            self, default_value, raw_type
        ) or self.generate_expression_with_expected(default_value, raw_type)
        dependencies = self.metal_struct_type_dependencies(mapped_type)
        return (
            f"    static constant constexpr {mapped_type} {member_name} = "
            f"{initializer};\n",
            dependencies,
        )

    def generate_function(
        self,
        func,
        indent=0,
        shader_type=None,
        execution_config=None,
        entry_name=None,
        stage_local_variables=None,
        stage_node=None,
    ):
        """Render a function or stage entry point with Metal attributes."""
        code = ""
        code += "  " * indent

        param_list = getattr(func, "parameters", getattr(func, "params", []))
        params = []
        reserved_parameter_names = {
            getattr(parameter, "name", None)
            for parameter in param_list
            if getattr(parameter, "name", None)
        }
        sampler_parameters = set()
        sampler_parameter_array_sizes = {}
        texture_parameters = {}
        texture_parameter_raw_types = {}
        texture_parameter_array_sizes = {}
        image_format_parameters = {}
        vertex_stage_input_parameters = []
        fragment_stage_input_parameters = []
        stage_builtin_parameter_alias_declarations = []
        value_array_parameter_copy_statements = []
        vertex_stage_input_alias_declarations = []
        fragment_stage_input_alias_declarations = []
        self.metal_entry_buffer_array_declarations = []
        self.metal_entry_buffer_array_names = set()
        stage_output_parameters = []
        stage_output_struct_name = None
        previous_function_state = save_metal_function_state(
            self, METAL_FUNCTION_STATE_ATTRIBUTES
        )
        previous_function_body_state = save_metal_function_state(
            self, METAL_FUNCTION_BODY_STATE_ATTRIBUTES
        )
        self.current_function_name = getattr(func, "name", None)
        self.current_metal_argument_table = metal_entry_argument_table(func)
        self.current_function_return_wrapper = None
        self.current_local_identifier_remaps = {}
        self.current_metal_stage_output_return_active = False
        self.current_generic_function_substitutions = (
            getattr(func, "_generic_substitutions", {}) or {}
        )
        unsupported_metal_ray_function_table_parameter_diagnostics = (
            self.prepare_metal_function_parameter_scope(
                func,
                shader_type,
                param_list,
                stage_output_parameters,
                vertex_stage_input_parameters,
            )
        )
        self.metal_function_parameter_declarations(
            func,
            shader_type,
            fragment_stage_input_parameters,
            image_format_parameters,
            param_list,
            params,
            reserved_parameter_names,
            sampler_parameter_array_sizes,
            sampler_parameters,
            stage_builtin_parameter_alias_declarations,
            stage_output_parameters,
            texture_parameter_array_sizes,
            texture_parameter_raw_types,
            texture_parameters,
            unsupported_metal_ray_function_table_parameter_diagnostics,
            value_array_parameter_copy_statements,
            vertex_stage_input_parameters,
        )

        self.metal_stage_entry_parameters(
            func, shader_type, param_list, params, reserved_parameter_names
        )

        reserved_parameter_names.update(self.global_resource_parameter_names())
        # An entry reads constant-block members through the blocks it takes.
        function_cbuffers = [
            cbuffer
            for cbuffer in self.cbuffer_variables
            if self.metal_current_entry_takes(cbuffer)
        ]
        self.cbuffer_parameter_names = self.collect_cbuffer_parameter_names(
            function_cbuffers, reserved_names=reserved_parameter_names
        )
        self.cbuffer_member_references = self.collect_cbuffer_member_references(
            function_cbuffers
        )
        self.current_local_identifier_remaps = (
            self.collect_metal_local_identifier_remaps(
                func,
                param_list,
                vertex_stage_input_parameters
                + fragment_stage_input_parameters
                + stage_output_parameters,
                stage_local_variables,
            )
        )

        params_str, effective_return_type, raw_return_type, return_type = (
            self.metal_function_return_plan(
                func,
                shader_type,
                params,
                reserved_parameter_names,
                stage_output_parameters,
            )
        )
        self.current_function_return_type = effective_return_type
        self.validate_metal_ray_stage_signature(func, shader_type, raw_return_type)
        parameter_diagnostics = self.glsl_buffer_block_parameter_diagnostics(
            "Metal", param_list, indent
        )
        if parameter_diagnostics:
            code += parameter_diagnostics
            code += "  " * indent
        unsupported_function_reason = self.unsupported_glsl_buffer_block_functions.get(
            getattr(func, "name", None)
        )
        if unsupported_function_reason is not None:
            code += self.unsupported_glsl_buffer_block_function_placeholder(
                "Metal", getattr(func, "name", None), unsupported_function_reason
            )
            restore_metal_function_state(self, previous_function_state)
            return code

        body = getattr(func, "body", None)
        precise_arithmetic = (
            body is not None and self.metal_function_requires_no_contraction(func)
        )
        if shader_type is None and body is None:
            semantic = self.semantic_from_node(func)
            function_name = entry_name or func.name
            semantic_attr = self.map_non_stage_function_semantic(semantic)
            code += (
                f"{self.metal_function_linkage_prefix(func)}"
                f"{return_type} {function_name}({params_str}){semantic_attr};\n\n"
            )
            restore_metal_function_state(self, previous_function_state)
            return code

        (
            function_name,
            code,
            vertex_stage_input_alias_declarations,
            stage_output_struct_name,
            return_type,
            fragment_stage_input_alias_declarations,
        ) = self.metal_function_signature(
            func,
            shader_type,
            execution_config,
            entry_name,
            stage_node,
            code,
            fragment_stage_input_parameters,
            params_str,
            raw_return_type,
            reserved_parameter_names,
            return_type,
            stage_output_parameters,
            vertex_stage_input_parameters,
            fragment_stage_input_alias_declarations,
            stage_output_struct_name,
            vertex_stage_input_alias_declarations,
        )

        if shader_type is not None:
            code = (
                "".join(
                    resource_identity_marker(p, p.name, entry_point=function_name)
                    for p in param_list
                )
                + code
            )

        if precise_arithmetic:
            code += "    #pragma clang fp contract(off)\n"
            code += "    #pragma clang fp reassociate(off)\n"

        previous_resource_parameter_state = save_metal_function_state(
            self, METAL_FUNCTION_RESOURCE_PARAMETER_ATTRIBUTES
        )
        self.current_sampler_parameters = sampler_parameters
        self.current_sampler_parameter_array_sizes = sampler_parameter_array_sizes
        self.current_texture_parameters = texture_parameters
        self.current_texture_parameter_raw_types = texture_parameter_raw_types
        self.current_texture_parameter_array_sizes = texture_parameter_array_sizes
        self.current_texture_alias_sources = {}
        self.current_image_format_parameters = image_format_parameters
        self.register_metal_buffer_resource_parameter_scope(
            self.current_function_name,
            include_all=shader_type in {
                "vertex",
                "fragment",
                "geometry",
                "tessellation_control",
                "tessellation_evaluation",
                "compute",
                "ray_generation",
            },
        )
        if shader_type == "mesh" and self.current_metal_mesh_output_config is not None:
            self.current_metal_mesh_output_accumulators = (
                self.collect_metal_mesh_output_accumulators(
                    func, reserved_parameter_names
                )
            )
            code += self.generate_metal_mesh_output_accumulator_declarations()
        for diagnostic in unsupported_metal_ray_function_table_parameter_diagnostics:
            code += f"    {diagnostic}\n"
        for declaration in self.metal_entry_buffer_array_declarations:
            code += f"    {declaration}\n"
        for declaration in stage_builtin_parameter_alias_declarations:
            code += f"    {declaration}\n"
        for statement in value_array_parameter_copy_statements:
            code += f"    {statement}\n"
        for declaration in vertex_stage_input_alias_declarations:
            code += f"    {declaration}\n"
        for declaration in fragment_stage_input_alias_declarations:
            code += f"    {declaration}\n"
        code += self.generate_metal_stage_output_parameter_locals(
            stage_output_parameters
        )
        self.current_metal_stage_output_return_active = (
            self.metal_stage_output_return_state(
                stage_output_struct_name,
                stage_output_parameters,
                reserved_parameter_names
                | self.metal_function_local_variable_names(func),
            )
            if stage_output_parameters
            else False
        )
        stage_value_variables = [
            local_var
            for local_var in stage_local_variables or []
            if not self.is_stage_local_resource_variable(local_var)
        ]
        code += self.generate_statement_body(stage_value_variables, 1)
        code += self.generate_statement_body(body, 1)
        if stage_output_parameters:
            code += self.generate_metal_stage_output_parameter_return(
                stage_output_struct_name,
                stage_output_parameters,
                reserved_parameter_names
                | self.metal_function_local_variable_names(func),
            )
        elif self.metal_needs_fallback_return(body, return_type):
            fallback = self.metal_default_value_expression(raw_return_type)
            code += (
                f"    return {fallback} "
                "/* fallback for unmatched generated control flow */;\n"
            )
        restore_metal_function_state(self, previous_resource_parameter_state)
        restore_metal_function_state(self, previous_function_body_state)
        restore_metal_function_state(self, previous_function_state)

        code += "}\n"
        code += "\n"
        return code

    def validate_metal_fragment_invocation_density(self, func, shader_type):
        if shader_type != "fragment":
            return
        for reachable_func in self.metal_reachable_functions(func):
            if self.function_uses_metal_fragment_invocation_density(reachable_func):
                raise UnsupportedMetalFeatureError(
                    "gl_FragSizeEXT",
                    (
                        "Metal target does not expose a fragment invocation density "
                        "or fragment-size input equivalent for "
                        "GL_EXT_fragment_invocation_density / gl_FragSizeEXT; keep "
                        "this shader on a target with fragment density builtins or "
                        "specialize the density path before Metal generation."
                    ),
                    missing_capabilities=FRAGMENT_INVOCATION_DENSITY_CAPABILITIES,
                )

    def validate_metal_kernel_derivatives(self, func, shader_type):
        """Refuse derivatives reached from a Metal ``kernel``, ``mesh`` or ``object`` function.

        MSL has ``dfdx``, ``dfdy`` and ``fwidth`` only in fragment functions, so a kernel, mesh or object function
        that calls them, directly or through a helper, does not compile. SPIR-V derivative groups and Shader Model
        6.6 compute derivatives have no Metal equivalent.
        """
        function_kind = self.METAL_DERIVATIVE_FREE_FUNCTION_KINDS[shader_type]
        article = "an" if function_kind == "object" else "a"
        for reachable_func in self.metal_reachable_functions(func):
            derivative_name = self.metal_kernel_derivative_call_name(reachable_func)
            if derivative_name is None:
                continue
            entry_name = getattr(func, "name", None) or "<entry>"
            caller_name = getattr(reachable_func, "name", None) or entry_name
            via = "" if reachable_func is func else f" through '{caller_name}'"
            raise UnsupportedMetalFeatureError(
                derivative_name,
                (
                    f"Metal {shader_type} entry point '{entry_name}' calls "
                    f"{derivative_name}{via}; Metal has derivatives only in fragment "
                    f"functions, so {article} {function_kind} function cannot call dfdx, dfdy "
                    "or fwidth."
                ),
                missing_capabilities=(f"metal.{function_kind}-derivatives",),
                operation=derivative_name,
                reason="derivatives are fragment-only in Metal",
            )

    def validate_struct_member_semantic_types(self, struct_node):
        signed_members = self.metal_signed_builtin_members.get(
            getattr(struct_node, "name", None), ()
        )
        for member in getattr(struct_node, "members", []) or []:
            if getattr(member, "name", None) in signed_members:
                continue
            semantic = self.semantic_from_node(member)
            metal_semantic = self.canonical_metal_semantic(semantic)
            expected_type = self.struct_member_builtin_semantic_type(metal_semantic)
            if expected_type is None:
                continue
            if expected_type == "invalid":
                struct_name = getattr(struct_node, "name", "<anonymous>")
                member_name = getattr(member, "name", "<anonymous>")
                raise MetalRefusal(
                    f"Metal struct '{struct_name}' semantic '{semantic}' maps to "
                    f"'{metal_semantic}' and is not supported"
                )
            actual_type = self.map_type(self.struct_member_raw_type(member))
            # [[color(n)]] takes any scalar or vector of up to four float, half or integer components,
            # such as float depth [[color(3)]] (MSL 3.2 section 5.2.3.4).
            color_value = metal_semantic.startswith("color(") and (
                METAL_COLOR_ATTACHMENT_TYPE.fullmatch(str(actual_type)) is not None
            )
            if actual_type != expected_type and not color_value:
                struct_name = getattr(struct_node, "name", "<anonymous>")
                member_name = getattr(member, "name", "<anonymous>")
                raise MetalRefusal(
                    f"Metal struct '{struct_name}' semantic '{semantic}' maps to "
                    f"'{metal_semantic}' and requires member '{member_name}' to "
                    f"have type {expected_type}, got {actual_type}"
                )

    def validate_function_return_semantic_type(self, func, raw_return_type, stage_name):
        semantic = self.function_return_semantic(func)
        if semantic is not None and self.map_type(raw_return_type) == "void":
            function_name = getattr(func, "name", "<anonymous>")
            raise MetalRefusal(
                f"Metal {stage_name} function '{function_name}' cannot use "
                f"return semantic '{semantic}' with void return type"
            )

        metal_semantic = metal_entry_return_semantic_key(
            self, func, semantic, stage_name
        )
        self.validate_function_return_semantic_stage(
            stage_name, semantic, metal_semantic
        )
        expected_type = self.function_return_builtin_semantic_type(
            semantic, metal_semantic
        )
        if expected_type is None:
            return
        if expected_type == "invalid":
            raise MetalRefusal(
                f"Metal {stage_name} semantic '{semantic}' cannot be used as a "
                "function return semantic"
            )
        actual_type = self.map_type(raw_return_type)
        # A fragment function may return any scalar or vector colour value, as HLSL's float main() : SV_Target
        # does (MSL 3.2 section 5.2.3.4).
        color_value = bool(metal_semantic and metal_semantic.startswith("color(")) and (
            METAL_COLOR_ATTACHMENT_TYPE.fullmatch(str(actual_type)) is not None
        )
        if actual_type != expected_type and not color_value:
            function_name = getattr(func, "name", "<anonymous>")
            raise MetalRefusal(
                f"Metal {stage_name} return semantic '{semantic}' maps to "
                f"'{metal_semantic}' and requires function '{function_name}' "
                f"to return {expected_type}, got {actual_type}"
            )

    def generate_statement(self, stmt, indent=0):
        """Render a single CrossGL AST statement as Metal source."""
        if id(stmt) not in self.metal_writeback_statement_ids:
            return_code = generate_return_writeback_statement(self, stmt, indent)
            if return_code is not None:
                return return_code
            conditional_calls = []
            writebacks = self.metal_reference_argument_writebacks(
                stmt, conditional_calls
            )
            if conditional_calls:
                return generate_conditional_call_statement(
                    self, stmt, indent, conditional_calls, writebacks
                )
            if writebacks:
                call_code = generate_call_result_statement(
                    self, stmt, indent, writebacks
                )
                if call_code is not None:
                    return call_code
                return generate_ordered_writeback_statement(
                    self, stmt, indent, writebacks
                )
            header_code = generate_header_writeback_statement(self, stmt, indent)
            if header_code is not None:
                return header_code
        indent_str = "    " * indent
        if isinstance(stmt, (VariableNode, BackendVariableNode)):
            var_type = self.local_variable_declared_type(stmt)
            texture_alias_type = self.local_variable_texture_alias_resource_type(
                stmt, var_type
            )
            if texture_alias_type is not None:
                initial_raw_type = self.texture_argument_raw_resource_type(
                    getattr(stmt, "initial_value", None)
                )
                var_type = self.local_variable_texture_alias_declaration_type(
                    stmt, texture_alias_type
                )
                self.current_texture_parameters[stmt.name] = texture_alias_type
                if initial_raw_type is not None:
                    self.current_texture_parameter_raw_types[stmt.name] = (
                        self.resource_base_type(initial_raw_type)
                    )
                self.record_local_texture_alias_sampler_source(stmt)
                self.record_local_image_alias_metadata(stmt)
            sampler_alias_type = self.local_variable_sampler_alias_resource_type(
                stmt, var_type
            )
            if sampler_alias_type is not None:
                var_type = sampler_alias_type
                self.current_sampler_parameters.add(stmt.name)
                _sampler_type, sampler_array_size = self.metal_array_type_parts(
                    sampler_alias_type
                )
                if sampler_array_size is not None:
                    self.current_sampler_parameter_array_sizes[stmt.name] = (
                        sampler_array_size
                    )
            acceleration_structure_alias_type = (
                self.local_variable_acceleration_structure_alias_resource_type(
                    stmt, var_type
                )
            )
            if acceleration_structure_alias_type is not None:
                var_type = acceleration_structure_alias_type
            if var_type is None:
                # Resource aliases above are typed from their source; anything still untyped is refused.
                var_type = self.local_variable_declared_type(stmt, refuse_untyped=True)
            initial_value = getattr(stmt, "initial_value", None)
            acceleration_structure_array_alias = (
                self.unsupported_metal_acceleration_structure_array_alias_source(
                    initial_value
                )
            )
            if (
                acceleration_structure_array_alias is not None
                and self.is_acceleration_structure_type(var_type)
            ):
                self.local_variable_types[stmt.name] = var_type
                self.unsupported_metal_acceleration_structure_array_variables[
                    stmt.name
                ] = self.unsupported_metal_acceleration_structure_array_variables[
                    acceleration_structure_array_alias
                ]
                diagnostic = self.unsupported_metal_acceleration_structure_array_alias_diagnostic(
                    stmt.name, acceleration_structure_array_alias
                )
                return f"{indent_str}{diagnostic}"
            ray_function_table_array_alias = (
                self.unsupported_metal_ray_function_table_array_alias_source(
                    initial_value
                )
            )
            if ray_function_table_array_alias is not None:
                self.local_variable_types[stmt.name] = var_type
                self.unsupported_metal_ray_function_table_array_variables[stmt.name] = (
                    self.unsupported_metal_ray_function_table_array_variables[
                        ray_function_table_array_alias
                    ]
                )
                diagnostic = (
                    self.unsupported_metal_ray_function_table_array_alias_diagnostic(
                        stmt.name, ray_function_table_array_alias
                    )
                )
                return f"{indent_str}{diagnostic}"
            self.local_variable_types[stmt.name] = var_type
            self.current_address_space_variables[stmt.name] = (
                self.local_variable_address_space(stmt)
            )
            self.record_readonly_metal_mesh_payload_local_alias(stmt)
            self.record_readonly_metal_local_alias(stmt)
            is_atomic_local = self.is_metal_atomic_value_type(var_type)
            if self.is_unsupported_glsl_buffer_block_struct_type(var_type):
                self.current_unsupported_glsl_buffer_block_local_variables.add(
                    stmt.name
                )
                return (
                    f"{indent_str}"
                    f"{self.unsupported_glsl_buffer_block_local_variable_placeholder('Metal', var_type, stmt.name)};\n"
                )

            local_name = self.metal_local_identifier_name(stmt.name)
            value_array_type = self.metal_local_value_array_type(stmt, var_type)
            if value_array_type is not None:
                declaration = f"{value_array_type} {local_name}"
            else:
                declaration = format_c_style_array_declaration(
                    self.metal_local_storage_declaration_type(stmt, var_type),
                    local_name,
                )
            declaration = f"{self.local_variable_qualifier(stmt)}{declaration}"
            declaration = self.maybe_format_unused_local_declaration(
                declaration, stmt.name
            )
            if isinstance(initial_value, MatchNode):
                code = f"{indent_str}{declaration};\n"
                code += generate_match_expression_assignment(
                    self,
                    initial_value,
                    local_name,
                    var_type,
                    indent,
                    "Metal",
                )
                return code
            if initial_value is not None:
                address_space_conflict = (
                    self.local_variable_address_space_conflict_diagnostic(stmt)
                )
                if address_space_conflict is not None:
                    conflict_declaration = (
                        self.local_variable_address_space_conflict_declaration(
                            stmt, declaration
                        )
                    )
                    return (
                        f"{indent_str}{address_space_conflict}\n"
                        f"{indent_str}{conflict_declaration};\n"
                    )
                # An array initializer renders the value's elements, so the value itself is rendered only when
                # it is used: a whole buffer block array has no Metal expression.
                array_initializer = None
                if value_array_type is None and not is_atomic_local:
                    array_initializer = generate_metal_array_initializer(
                        self, initial_value, var_type
                    )
                if array_initializer is not None:
                    init_expr = array_initializer
                else:
                    init_expr = self.generate_expression_with_expected(
                        initial_value, var_type
                    )
                if self.local_pointer_initializer_needs_address(stmt, initial_value):
                    init_expr = f"&{init_expr}"
                address_space_mismatch = (
                    self.local_variable_address_space_mismatch_diagnostic(stmt)
                )
                if is_atomic_local:
                    array_initializer = (
                        self.generate_metal_atomic_array_initializer_stores(
                            local_name, var_type, initial_value, indent_str
                        )
                    )
                    if array_initializer is not None:
                        return f"{indent_str}{declaration};\n{array_initializer}"
                    return (
                        f"{indent_str}{declaration};\n"
                        f"{indent_str}atomic_store_explicit(&{local_name}, {init_expr}, "
                        "memory_order_relaxed);\n"
                    )
                diagnostic_prefix = (
                    f"{indent_str}{address_space_mismatch}\n"
                    if address_space_mismatch is not None
                    else ""
                )
                return f"{diagnostic_prefix}{indent_str}{declaration} = {init_expr};\n"
            else:
                return f"{indent_str}{declaration};\n"
        elif isinstance(stmt, ArrayNode):
            element_type = self.map_type(stmt.element_type)
            size = get_array_size_from_node(stmt)
            local_name = self.metal_local_identifier_name(stmt.name)

            if size is None:
                raise UnsizedArrayUnsupportedError(
                    "metal", stmt.name, "is a local variable"
                )
            else:
                declaration = self.maybe_format_unused_local_declaration(
                    f"array<{element_type}, {size}> {local_name}",
                    stmt.name,
                )
                return f"{indent_str}{declaration};\n"
        elif isinstance(stmt, (AssignmentNode, BackendAssignmentNode)):
            return self.generate_statement_code(self.generate_assignment(stmt), indent)
        elif isinstance(stmt, BlockNode):
            return self.generate_block(stmt, indent)
        elif isinstance(stmt, BreakNode):
            return f"{indent_str}break;\n"
        elif isinstance(stmt, ContinueNode):
            return f"{indent_str}continue;\n"
        elif isinstance(stmt, IfNode):
            return self.generate_if(stmt, indent)
        elif isinstance(stmt, ForNode):
            return self.generate_for(stmt, indent)
        elif isinstance(stmt, ForInNode):
            return self.generate_for_in(stmt, indent)
        elif isinstance(stmt, WhileNode):
            return self.generate_while(stmt, indent)
        elif isinstance(stmt, DoWhileNode):
            return self.generate_do_while(stmt, indent)
        elif isinstance(stmt, LoopNode):
            return self.generate_loop(stmt, indent)
        elif isinstance(stmt, SwitchNode):
            return self.generate_switch(stmt, indent)
        elif isinstance(stmt, MatchNode):
            return self.generate_match(stmt, indent)
        elif isinstance(stmt, (ReturnNode, BackendReturnNode)):
            stage_output_return = getattr(
                self, "current_metal_stage_output_return_active", False
            )
            if getattr(stmt, "value", None) is None:
                if stage_output_return:
                    return self.generate_metal_stage_output_early_return(
                        stage_output_return, indent_str
                    )
                return f"{indent_str}return;\n"
            if stage_output_return and not isinstance(stmt.value, list):
                value = self.generate_expression_with_expected(
                    stmt.value, self.current_function_return_type
                )
                return self.generate_metal_stage_output_early_return(
                    stage_output_return, indent_str, value
                )
            if isinstance(stmt.value, list):
                code = ""
                for i, return_stmt in enumerate(stmt.value):
                    code += f"{self.generate_expression(return_stmt)}"
                    if i < len(stmt.value) - 1:
                        code += ", "
                return f"{indent_str}return {code};\n"
            else:
                return_wrapper = self.current_function_return_wrapper
                if return_wrapper is not None:
                    value = self.generate_expression_with_expected(
                        stmt.value, return_wrapper["source_type"]
                    )
                    return (
                        f"{indent_str}return {return_wrapper['struct_name']}"
                        f"{{{value}}};\n"
                    )
                array_return = self.metal_glsl_buffer_array_return(stmt.value)
                if array_return is not None:
                    return f"{indent_str}return {array_return};\n"
                value = self.generate_expression_with_expected(
                    stmt.value, self.current_function_return_type
                )
                fallback_return = (
                    self.generate_metal_vertex_output_position_fallback_return(
                        value, indent_str
                    )
                )
                if fallback_return is not None:
                    return fallback_return
                return f"{indent_str}return " f"{value};\n"
        elif hasattr(stmt, "__class__") and "ExpressionStatementNode" in str(
            type(stmt)
        ):
            tail_return = self.generate_tail_expression_statement(stmt, indent)
            if tail_return is not None:
                return tail_return
            expr_code = self.generate_expression_statement(stmt)
            return self.generate_statement_code(
                expr_code, indent
            ) + self.metal_image_store_fence_statement(
                getattr(stmt, "expression", None), indent_str
            )
        else:
            return (
                f"{indent_str}{self.generate_discarded_expression(stmt)};\n"
                f"{self.metal_image_store_fence_statement(stmt, indent_str)}"
            )

    def local_variable_is_address_space_alias(self, node):
        raw_type = self.local_variable_type_node(node)
        return (
            isinstance(raw_type, (PointerType, ReferenceType))
            or self.local_variable_crossgl_pointer_pointee(node) is not None
            or (self.array_indirect_element_type_node(raw_type) is not None)
        )

    def points_into_readonly_metal_storage(self, expr):
        """Whether a pointer value may point into read-only storage.

        Either branch of ``c ? in_ : out`` counts, as the C conditional
        operator gives the result the union of both pointees' qualifiers. A
        CrossGL pointer local that aliases read-only storage is recorded as
        read-only itself (record_readonly_metal_local_alias), so a pointer
        initialized from it, such as ``b = a + 1``, keeps ``const`` too.
        """
        if expr is None:
            return False
        if isinstance(expr, TernaryOpNode):
            return self.points_into_readonly_metal_storage(
                getattr(expr, "true_expr", None)
            ) or self.points_into_readonly_metal_storage(
                getattr(expr, "false_expr", None)
            )
        root = self.assignment_target_root_name(expr)
        return root in getattr(self, "current_readonly_metal_parameters", set())

    def collect_metal_lowered_program_scope_groupshared_globals(
        self, ast, target_stage, global_vars, all_functions
    ):
        groupshared_globals = [
            node
            for node in global_vars or []
            if self.is_program_scope_groupshared_global(node)
        ]
        if not groupshared_globals:
            return {}

        entries = self.metal_program_scope_groupshared_stage_entries(ast, target_stage)
        entry_ids = {id(func) for _stage_name, func, _stage_locals in entries}
        lowered_by_function = {}

        for node in groupshared_globals:
            name = self.metal_variable_name(node)
            if not name or self.program_scope_groupshared_initializer(node) is not None:
                continue

            referenced_by_helper = any(
                id(func) not in entry_ids
                and self.node_uses_identifier(getattr(func, "body", []), name)
                for func in all_functions or []
            )
            if referenced_by_helper:
                continue

            used_entries = []
            blocked_entry_reference = False
            for stage_name, func, stage_locals in entries:
                if not self.node_uses_identifier(getattr(func, "body", []), name):
                    continue
                if stage_name not in METAL_GROUPSHARED_ENTRY_STAGES or (
                    name in self.metal_entry_local_names(func, stage_locals)
                ):
                    blocked_entry_reference = True
                    break
                used_entries.append(func)

            if blocked_entry_reference:
                continue
            for func in used_entries:
                lowered_by_function.setdefault(id(func), []).append(node)

        return lowered_by_function

    def metal_narrow_integer_update_expression(
        self, node, target, operator, value=None
    ):
        value_type = self.expression_result_type(target)
        if self.metal_narrow_integer_conversion_expression("value", value_type) is None:
            return None
        if not self.metal_narrow_update_repeatable_lvalue(target):
            raise UnsupportedMetalFeatureError(
                "byte-update",
                "Narrow integer updates require a stable variable or field",
                reason="byte-update-lvalue-unsupported",
                source_location=getattr(node, "source_location", None),
            )
        lhs = self.generate_expression(target)
        mapped = self.map_type(value_type)
        promoted = mapped.replace("uint", "int")
        rhs = (
            self.generate_expression_with_expected(value, None)
            if value is not None
            else "1"
        )
        native = self.metal_native_narrow_bitcast_storage_type(value_type)
        if native[-1:] in {"2", "3", "4"} and (
            self.metal_native_narrow_bitcast_storage_type(
                self.expression_result_type(value)
            )
            == native
        ):
            rhs = f"{promoted}({rhs})"
        binary = operator[0] if operator in {"++", "--"} else operator[:-1]
        converted = self.metal_narrow_integer_conversion_expression(
            f"({promoted}({lhs}) {binary} ({rhs}))", value_type
        )
        updated = f"({lhs} = {converted})"
        if value is None and getattr(node, "is_postfix", False):
            inverse = "-" if operator == "++" else "+"
            # Fixed-width increment is bijective; invert the stored result.
            previous = self.metal_narrow_integer_conversion_expression(
                f"({promoted}({lhs}) {inverse} 1)", value_type
            )
            return f"({updated}, {previous})"
        return updated

    def metal_registered_structure_assignment(self, target, value):
        """Lower an implicit registered-structure conversion in ``target = value``.

        MLX's ``complex_t`` converts implicitly from a scalar (``real`` takes the
        value, ``imag`` is zero) and to a scalar (its ``real`` part). CrossGL
        carries the plain fields, so Metal accepts neither assignment between
        the structure and a scalar without an explicit conversion. Returns the
        converted right-hand side, or ``None`` when no such conversion applies.
        """
        target_type = self.type_name_string(self.expression_result_type(target))
        value_type = self.type_name_string(self.expression_result_type(value))
        if not target_type or not value_type:
            return None
        mapped_target = self.map_type(target_type)
        mapped_value = self.map_type(value_type)
        target_kind = self.METAL_STRUCTURE_PROMOTION_SCALAR_KINDS.get(mapped_target)
        value_kind = self.METAL_STRUCTURE_PROMOTION_SCALAR_KINDS.get(mapped_value)
        if target_kind is not None and value_kind is None:
            return self.metal_registered_structure_scalar_conversion(
                value, target_type, [value]
            )
        if value_kind is None or target_kind is not None:
            return None
        struct_name = self.resolve_metal_type_alias(target_type)
        contracts = {
            contract
            for identity in {target_type, struct_name, mapped_target}
            if (contract := registered_structure_conversion_for_identity(identity))
            is not None
        }
        if len(contracts) != 1:
            return None
        (contract,) = contracts
        member_types = self.struct_member_types.get(
            struct_name
        ) or self.struct_member_types.get(mapped_target)
        actual_shape = (
            None
            if member_types is None
            else tuple(
                (name, self.map_type(member_type))
                for name, member_type in member_types.items()
            )
        )
        expected_shape = tuple(
            (field.name, self.map_type(field.type_name)) for field in contract.fields
        )
        reason = None
        if actual_shape != expected_shape:
            reason = (
                "destination-shape-unknown"
                if actual_shape is None
                else "destination-shape-mismatch"
            )
        elif value_kind not in contract.source_kinds:
            reason = "source-kind-unsupported"
        elif contract.scalar_source_use_count != 1:
            reason = "source-single-evaluation-unsupported"
        if reason is not None:
            raise MetalStructConversionError(
                "Metal cannot preserve registered structure scalar promotion "
                f"from '{value_type}' to '{target_type}': "
                f"{reason.replace('-', ' ')}",
                destination_type=target_type,
                source_type=value_type,
                conversion_kind=StructureConversionKind.SCALAR_PROMOTION.value,
                reason=reason,
                source_location=getattr(value, "source_location", None),
            )
        rendered = self.generate_expression_with_expected(value, None)
        fields = []
        for field in contract.fields:
            field_type = self.map_type(field.type_name)
            if field.scalar_value is StructureFieldValue.ZERO:
                fields.append(f"{field_type}(0)")
            else:
                fields.append(f"{field_type}({rendered})")
        return f"{mapped_target}{{{', '.join(fields)}}}"

    def metal_registered_structure_scalar_conversion(
        self,
        expr,
        destination_type,
        arguments,
    ):
        """Lower a registered structure-to-scalar value conversion exactly.

        CrossGL intentionally carries structure fields rather than source C++
        conversion operators.  A round trip may therefore reconstruct a scalar
        constructor call whose source operand is a registered structure-backed
        value.  Admit only the exact registered destination identity and field
        shape, and evaluate the source expression once through its designated
        converted field.
        """
        arguments = list(arguments or [])
        if (
            not isinstance(destination_type, str)
            or destination_type in getattr(self, "user_function_names", set())
            or len(arguments) != 1
        ):
            return None

        mapped_destination = self.map_type(destination_type)
        if mapped_destination not in {
            "bool",
            "char",
            "uchar",
            "short",
            "ushort",
            "int",
            "uint",
            "long",
            "ulong",
            "int64_t",
            "uint64_t",
            "half",
            "float",
            "double",
            "bfloat",
        }:
            return None

        argument = arguments[0]
        source_type = self.expression_result_type(argument)
        source_type_name = self.type_name_string(source_type)
        if not source_type_name:
            return None

        mapped_source = self.map_type(source_type_name)
        source_struct_name = self.resolve_metal_type_alias(source_type_name)
        source_identities = {
            source_type_name,
            source_struct_name,
            mapped_source,
        }
        matching_contracts = {
            contract
            for identity in source_identities
            if (contract := registered_structure_conversion_for_identity(identity))
            is not None
        }
        if not matching_contracts:
            return None
        if len(matching_contracts) != 1:
            raise MetalStructConversionError(
                "Metal structure value conversion source identity is ambiguous: "
                f"'{source_type_name}'",
                destination_type=destination_type,
                source_type=source_type_name,
                conversion_kind=StructureConversionKind.VALUE_CONVERSION.value,
                reason="source-identity-ambiguous",
                source_location=getattr(expr, "source_location", None),
            )
        contract = next(iter(matching_contracts))

        member_types = self.struct_member_types.get(source_struct_name)
        if member_types is None:
            member_types = self.struct_member_types.get(mapped_source)
        actual_shape = (
            None
            if member_types is None
            else tuple(
                (name, self.map_type(member_type))
                for name, member_type in member_types.items()
            )
        )
        expected_shape = tuple(
            (field.name, self.map_type(field.type_name)) for field in contract.fields
        )
        if actual_shape != expected_shape:
            reason = (
                "source-shape-unknown"
                if actual_shape is None
                else "source-shape-mismatch"
            )
            raise MetalStructConversionError(
                "Metal cannot preserve registered structure value conversion "
                f"from '{source_type_name}' to '{destination_type}': "
                f"{reason.replace('-', ' ')}",
                destination_type=destination_type,
                source_type=source_type_name,
                conversion_kind=StructureConversionKind.VALUE_CONVERSION.value,
                reason=reason,
                source_location=getattr(expr, "source_location", None),
            )

        converted_fields = [
            field
            for field in contract.fields
            if field.scalar_value is StructureFieldValue.CONVERTED_SOURCE
        ]
        if len(converted_fields) != 1:
            raise MetalStructConversionError(
                "Metal registered structure value conversion requires exactly "
                "one converted source field",
                destination_type=destination_type,
                source_type=source_type_name,
                conversion_kind=StructureConversionKind.VALUE_CONVERSION.value,
                reason="source-single-evaluation-unsupported",
                source_location=getattr(expr, "source_location", None),
            )

        rendered = self.generate_expression_with_expected(argument, None)
        source_field = converted_fields[0].name
        return f"{mapped_destination}(({rendered}).{source_field})"

    def expression_result_type(self, expr):
        if expr is None:
            return None
        if is_comma_expression(expr):
            return self.expression_result_type(comma_result_operand(expr))
        if isinstance(expr, IdentifierNode):
            builtin_type = self.metal_graphics_builtin_result_type(expr.name)
            if builtin_type is not None:
                return builtin_type
            builtin_type = self.metal_compute_builtin_result_type(expr.name)
            if builtin_type is not None:
                return builtin_type
            return (
                self.local_variable_types.get(expr.name)
                or self.metal_program_scope_value_global_types.get(expr.name)
                or self.metal_global_resource_value_type(expr.name)
                or getattr(self, "metal_cbuffer_member_types", {}).get(expr.name)
                or getattr(self, "metal_program_scope_constant_types", {}).get(
                    expr.name
                )
            )
        if isinstance(expr, (VariableNode, BackendVariableNode)):
            name = getattr(expr, "name", None)
            builtin_type = self.metal_graphics_builtin_result_type(name)
            if builtin_type is not None:
                return builtin_type
            builtin_type = self.metal_compute_builtin_result_type(name)
            if builtin_type is not None:
                return builtin_type
            return (
                self.local_variable_types.get(name)
                or self.metal_program_scope_value_global_types.get(name)
                or self.metal_global_resource_value_type(name)
                or getattr(self, "metal_cbuffer_member_types", {}).get(name)
                or getattr(self, "metal_program_scope_constant_types", {}).get(name)
            )
        if isinstance(expr, (int, float)):
            return "float" if isinstance(expr, float) else "int"
        handled = self.metal_binary_expression_result_type(expr)
        if handled is not UNHANDLED:
            return handled
        if isinstance(expr, PointerReinterpretNode):
            return getattr(expr, "target_type", None)
        if isinstance(expr, UnaryOpNode):
            operand_type = self.expression_result_type(expr.operand)
            if getattr(expr, "operator", None) == "*":
                return self.pointer_pointee_type_name(operand_type) or operand_type
            if getattr(expr, "operator", None) == "&" and operand_type is not None:
                return f"{self.type_name_string(operand_type)}*"
            if getattr(expr, "operator", None) == "!" and operand_type is not None:
                # Logical negation applies componentwise and yields bool
                # with the operand's shape.
                mapped = self.map_type(operand_type)
                if isinstance(mapped, str) and self.is_vector_value_type(mapped):
                    return f"bool{self.vector_value_width(mapped)}"
                if isinstance(mapped, str) and self.is_scalar_value_type(mapped):
                    return "bool"
            return operand_type
        if isinstance(expr, TernaryOpNode):
            true_type = self.expression_result_type(getattr(expr, "true_expr", None))
            false_type = self.expression_result_type(getattr(expr, "false_expr", None))
            common_type = self.metal_conditional_result_type(true_type, false_type)
            if common_type is not None:
                return common_type
            if self.is_vector_value_type(true_type):
                return true_type
            if self.is_vector_value_type(false_type):
                return false_type
            if true_type == false_type:
                return true_type
            if true_type == "float" or false_type == "float":
                return "float"
            return true_type or false_type
        if isinstance(expr, (AssignmentNode, BackendAssignmentNode)):
            return self.expression_result_type(
                getattr(expr, "target", getattr(expr, "left", None))
            )
        handled = self.metal_access_expression_result_type(expr)
        if handled is not UNHANDLED:
            return handled
        if isinstance(expr, ConstructorNode):
            aggregate_type = infer_enum_constructor_type(
                self, expr
            ) or infer_struct_constructor_type(self, expr)
            if aggregate_type is not None:
                return aggregate_type
            constructor_type = getattr(expr, "constructor_type", None)
            if self.is_builtin_value_constructor_type(constructor_type):
                return self.type_name_string(constructor_type)
            return None
        if isinstance(expr, MatchNode):
            return infer_match_expression_result_type(self, expr)
        if isinstance(expr, WaveOpNode):
            return self.metal_wave_result_type(expr.operation, expr.arguments)
        if isinstance(expr, RayQueryOpNode):
            return self.metal_ray_query_method_return_type(expr.operation)
        handled = self.metal_call_expression_result_type(expr)
        if handled is not UNHANDLED:
            return handled
        if hasattr(expr, "__class__") and "Literal" in str(expr.__class__):
            literal_type_name = self.type_name_string(
                getattr(expr, "literal_type", None)
            )
            if literal_type_name in {"bool", "int", "uint", "float", "half"}:
                return literal_type_name
            value = getattr(expr, "value", None)
            if isinstance(value, float):
                return "float"
            if isinstance(value, int):
                return "int"
            if isinstance(value, str):
                return "float" if "." in value else "int"
        if hasattr(expr, "__class__") and "Identifier" in str(expr.__class__):
            return self.local_variable_types.get(getattr(expr, "name", None))
        return None

    def generate_discarded_expression(self, expr):
        """Render an unused result without changing nested value expressions."""
        wide_atomic = discarded_wide_atomic_call(self, expr)
        if wide_atomic is not None:
            return wide_atomic
        if isinstance(expr, (AssignmentNode, BackendAssignmentNode)):
            return self.generate_assignment(expr)
        if is_comma_expression(expr):
            return c_family_comma_expression(
                expr,
                self.generate_expression,
                self.generate_discarded_expression,
                discarded=True,
            )
        rendered = self.generate_expression(expr)
        narrow_update = (
            isinstance(expr, UnaryOpNode)
            and self.map_operator(expr.op) in {"++", "--"}
            and (
                self.metal_narrow_integer_conversion_expression(
                    "value", self.expression_result_type(expr.operand)
                )
                is not None
                or self.is_metal_signed_builtin_member(expr.operand)
            )
        )
        if (
            narrow_update
            or self.is_discarded_metal_type_constructor(expr)
            or self.is_discarded_metal_texel_atomic(expr, rendered)
        ):
            # Postfix lowering reconstructs the old value after the stored update.
            # A discarded result must not trigger Metal's unused-value diagnostic.
            return f"(void)({rendered})"
        return rendered

    def metal_matrix_compound_assignment(self, target, value, op, lhs, rhs):
        """``m op= v`` for a componentwise matrix update MSL has no operator for, or ``None``.

        The update becomes ``m = helper(m, v)``, which evaluates ``m`` twice, so a target with side effects is
        refused.
        """
        if op not in {"+=", "-=", "/="}:
            return None
        update = self.metal_matrix_builtins.componentwise(
            op[:-1], target, value, lhs, rhs
        )
        if update is None:
            return None
        if not self.metal_expression_is_side_effect_free(target):
            raise UnsupportedMetalFeatureError(
                "matrix compound assignment",
                f"Metal has no '{op}' for this matrix operand, and the target "
                "of the rewritten update has side effects",
                operation=op,
                reason="target-has-side-effects",
            )
        return f"{lhs} = {update}"

    def generate_metal_signed_builtin_member_assignment(self, target, rhs, op):
        lhs = self.generate_metal_signed_builtin_member_storage(target)
        assignment = metal_signed_member_assignment(
            lhs, op, rhs, self.metal_expression_is_side_effect_free(target)
        )
        if assignment is None:
            raise UnsupportedMetalFeatureError(
                "signed built-in member update",
                f"Metal declares '{lhs}' uint; the signed '{op}' update of it "
                "reads the target twice, and the target has side effects",
                operation=op,
                reason="target-has-side-effects",
            )
        return assignment

    def generate_for_in(self, node, indent):
        indent_str = "    " * indent
        pattern = getattr(node, "pattern", "item")
        iterable_node = getattr(node, "iterable", "")
        previous_local_variable_types = dict(self.local_variable_types)
        previous_unsupported_locals = set(
            self.current_unsupported_glsl_buffer_block_local_variables
        )

        try:
            if getattr(node, "binding_type", None) is not None and (
                isinstance(iterable_node, RangeNode)
                or self.metal_for_in_array_element_type(
                    self.expression_result_type(iterable_node)
                )
                is None
            ):
                raise UnsupportedMetalFeatureError(
                    "typed for-in iterable",
                    f"Metal typed for-in binding '{pattern}' requires a fixed array",
                    missing_capabilities=("metal.fixed-array-for-in-lowering",),
                    operation="for-in",
                    reason="unsupported-iterable-type",
                    source_location=getattr(node, "source_location", None),
                )
            if isinstance(iterable_node, RangeNode):
                start = self.generate_expression(iterable_node.start)
                end = self.generate_expression(iterable_node.end)
                pattern_type = "int"
                comparator = "<=" if iterable_node.inclusive else "<"
                code = (
                    f"{indent_str}for (int {pattern} = {start}; "
                    f"{pattern} {comparator} {end}; ++{pattern}) {{\n"
                )
            else:
                iterable_type = self.expression_result_type(iterable_node)
                element_type = self.metal_for_in_array_element_type(iterable_type)
                iterable = self.generate_expression(iterable_node)
                if element_type is not None:
                    binding_type = getattr(node, "binding_type", None)
                    reference = isinstance(binding_type, ReferenceType)
                    value_type = (
                        binding_type.referenced_type if reference else binding_type
                    )
                    pattern_type = self.type_name_string(value_type)
                    if pattern_type in {None, "auto"}:
                        pattern_type = element_type
                    qualifiers = set(getattr(node, "binding_qualifiers", []) or [])
                    readonly = bool(qualifiers & {"const", "constant", "readonly"})
                    readonly |= reference and not binding_type.is_mutable
                    declaration = self.map_type(pattern_type)
                    if reference:
                        if (
                            self.metal_for_in_array_element_type(pattern_type)
                            is not None
                        ):
                            declaration = "auto"
                        address_space = self.argument_address_space(iterable_node)
                        address_space = address_space or "thread"
                        declaration = f"{address_space} {declaration}&"
                    if readonly:
                        declaration = f"const {declaration}"
                    code = (
                        f"{indent_str}for ({declaration} {pattern} : {iterable}) {{\n"
                    )
                elif self.is_scalar_integer_type(iterable_type):
                    pattern_type = "int"
                    code = (
                        f"{indent_str}for (int {pattern} = 0; {pattern} < {iterable}; "
                        f"++{pattern}) {{\n"
                    )
                else:
                    raise UnsupportedMetalFeatureError(
                        "for-in iterable",
                        f"Metal for-in binding '{pattern}' requires a fixed array or "
                        f"integer bound, got '{self.type_name_string(iterable_type)}'",
                        missing_capabilities=("metal.fixed-array-for-in-lowering",),
                        operation="for-in",
                        reason="unsupported-iterable-type",
                        source_location=getattr(node, "source_location", None),
                    )

            self.local_variable_types[pattern] = pattern_type
            self.current_unsupported_glsl_buffer_block_local_variables.discard(pattern)

            code += self.generate_scoped_statement_body(
                getattr(node, "body", []), indent + 1
            )
            code += f"{indent_str}}}\n"
            return code
        finally:
            self.local_variable_types = previous_local_variable_types
            self.current_unsupported_glsl_buffer_block_local_variables = (
                previous_unsupported_locals
            )

    def node_uses_identifier(self, node, name):
        if node is None:
            return False
        if isinstance(node, str):
            return re.search(rf"\b{re.escape(name)}\b", node) is not None
        if isinstance(node, IdentifierNode):
            return getattr(node, "name", None) == name
        if isinstance(node, VariableNode):
            if (
                getattr(node, "initial_value", None) is not None
                or getattr(node, "var_type", None) is not None
                or getattr(node, "vtype", None) is not None
            ):
                return self.node_uses_identifier(
                    getattr(node, "initial_value", None),
                    name,
                )
            return getattr(node, "name", None) == name
        if isinstance(node, ArrayNode):
            return False
        if isinstance(node, dict):
            return any(
                self.node_uses_identifier(value, name) for value in node.values()
            )
        if isinstance(node, (list, tuple, set)):
            return any(self.node_uses_identifier(value, name) for value in node)
        if hasattr(node, "__dict__"):
            for key, value in vars(node).items():
                if key in {"name", "type_name", "var_type", "vtype", "parent"}:
                    continue
                if key in {"body", "if_body", "else_body", "statements"}:
                    if self.statement_body_uses_identifier(value, name):
                        return True
                    continue
                if self.node_uses_identifier(value, name):
                    return True
            return False
        return False

    def generate_expression(self, expr):
        """Render a CrossGL AST expression into Metal expression syntax."""
        if expr is None:
            return ""
        substitutions = self.metal_argument_substitutions
        if substitutions and id(expr) in substitutions:
            return substitutions[id(expr)]
        elif isinstance(expr, str):
            return self.generate_metal_raw_string_expression(expr)
        elif isinstance(expr, IdentifierNode):
            return self.generate_metal_identifier_node_expression(expr)
        elif isinstance(expr, bool):
            return "true" if expr else "false"
        elif isinstance(expr, int) or isinstance(expr, float):
            return str(expr)
        elif isinstance(expr, (VariableNode, BackendVariableNode)):
            # Fix infinite recursion - directly return the name
            return self.generate_metal_variable_expression(expr)
        elif isinstance(expr, BinaryOpNode):
            return self.generate_metal_binary_expression(expr)
        elif isinstance(expr, (AssignmentNode, BackendAssignmentNode)):
            return assignment_value_expression(self.generate_assignment(expr))
        elif is_comma_expression(expr):
            return c_family_comma_expression(
                expr, self.generate_expression, self.generate_discarded_expression
            )
        elif isinstance(expr, ArrayLiteralNode):
            return self.generate_metal_array_literal_expression(expr)
        elif isinstance(expr, PointerReinterpretNode):
            return self.generate_metal_storage_pointer_reinterpret(expr)
        elif isinstance(expr, UnaryOpNode):
            # In C++/Metal, ``9223372036854775808`` cannot be represented by a
            # signed literal type, so spelling INT64_MIN as unary minus applied
            # to that token first creates an unsigned value and triggers a
            # warning-fatal conversion back to ``long``.  Keep the exact value
            # in signed arithmetic using two representable long literals.
            return self.generate_metal_unary_expression(expr)
        elif isinstance(expr, CooperativeMatrixOpNode):
            return self.generate_cooperative_matrix_operation(expr)
        elif isinstance(expr, WaveOpNode):
            return self.generate_metal_wave_op_expression(expr)
        elif isinstance(expr, RayTracingOpNode):
            return self.generate_ray_tracing_op_expression(expr)
        elif isinstance(expr, MeshOpNode):
            mesh_call = self.generate_mesh_op_expression(expr)
            if mesh_call is not None:
                return mesh_call
            args = ", ".join(self.generate_expression(arg) for arg in expr.arguments)
            return f"{expr.operation}({args})"
        elif isinstance(expr, RayQueryOpNode):
            return self.generate_metal_ray_query_call(
                expr.operation,
                expr.query_expr,
                expr.arguments,
            )
        elif isinstance(expr, ArrayAccessNode):
            return self.generate_metal_array_access_expression(expr)
        elif isinstance(expr, ConstructorNode):
            return self.generate_metal_constructor_expression(expr)
        elif isinstance(expr, FunctionCallNode):
            return self.generate_metal_function_call_expression(expr)
        elif isinstance(expr, MemberAccessNode):
            return self.generate_metal_member_access_expression(expr)
        elif isinstance(expr, PointerAccessNode):
            pointer = self.generate_expression_with_expected(expr.pointer_expr, None)
            if isinstance(
                expr.pointer_expr, (BinaryOpNode, TernaryOpNode, UnaryOpNode)
            ) and not self.metal_expression_is_parenthesized(pointer):
                # `->` binds tighter than every operator, so `(s + 1)->a` must
                # not become `s + 1->a`.
                pointer = f"({pointer})"
            return f"{pointer}->{expr.member}"
        elif isinstance(expr, TernaryOpNode):
            return self.generate_metal_ternary_expression(expr)
        elif hasattr(expr, "__class__") and "Literal" in str(expr.__class__):
            return self.generate_metal_literal_class_expression(expr)
        elif hasattr(expr, "__class__") and "Identifier" in str(expr.__class__):
            return self.generate_metal_identifier_class_expression(expr)
        else:
            return str(expr)

    def metal_atomic_fence_contract_error(
        self,
        reason,
        args,
        *,
        source_location=None,
    ):
        rendered = [self.atomic_fence_operand_text(arg) for arg in args]
        requested_contract = ", ".join(rendered) or "<no operands>"
        capability = {
            "invalid-argument-count": "metal.atomic-thread-fence.contract",
            "unsupported-memory-flags": "metal.atomic-thread-fence.memory-flags",
            "unsupported-memory-order": "metal.atomic-thread-fence.memory-order",
            "unsupported-thread-scope": "metal.atomic-thread-fence.thread-scope",
        }[reason]
        return UnsupportedMetalFeatureError(
            "atomic-thread-fence-contract",
            "Metal codegen cannot represent atomicThreadFence("
            f"{requested_contract}) exactly: {reason.replace('-', ' ')}",
            missing_capabilities=(capability,),
            operation="atomicThreadFence",
            reason=reason,
            source_location=source_location,
        )

    def generate_cooperative_matrix_operation(self, node):
        """Render a canonical cooperative-matrix operation with Metal semantics."""
        operation = node.operation
        arguments = list(node.arguments)
        exact_arities = {
            "element": 2,
            "multiply": 2,
            "multiply_accumulate": 4,
            "elementwise_add": 2,
            "elementwise_subtract": 2,
            "negate": 1,
        }
        expected_arity = exact_arities.get(operation)
        if expected_arity is not None and len(arguments) != expected_arity:
            raise UnsupportedMetalFeatureError(
                "cooperative-matrix-operation-arity",
                f"Metal cooperative-matrix operation '{operation}' expects "
                f"{expected_arity} arguments, got {len(arguments)}",
                operation=operation,
                reason="invalid-argument-count",
                source_location=getattr(node, "source_location", None),
            )
        if operation in {"load", "store"} and len(arguments) < 2:
            raise UnsupportedMetalFeatureError(
                "cooperative-matrix-operation-arity",
                f"Metal cooperative-matrix operation '{operation}' expects at "
                f"least 2 arguments, got {len(arguments)}",
                operation=operation,
                reason="invalid-argument-count",
                source_location=getattr(node, "source_location", None),
            )

        rendered = [self.generate_expression(argument) for argument in arguments]
        if operation in {"fill", "diagonal"}:
            result_type = getattr(node, "result_type", None)
            contract = (
                self.cooperative_matrix_contract(result_type)
                if result_type is not None
                else None
            )
            if contract is None or len(arguments) != 1:
                raise UnsupportedMetalFeatureError(
                    "cooperative-matrix-operation-lowering",
                    f"Metal cooperative-matrix {operation} requires one value and a "
                    "declared matrix result",
                    missing_capabilities=(
                        "metal.cooperative-matrix-operation-lowering",
                    ),
                    operation=operation,
                    reason="missing-result-contract",
                    source_location=getattr(node, "source_location", None),
                )
            matrix_type = self.map_cooperative_matrix_type(result_type, contract)
            if operation == "diagonal":
                # The one-value simdgroup_matrix constructor is the diagonal.
                return f"{matrix_type}({rendered[0]})"
            fill_arguments = matrix_type[len("simdgroup_matrix") :]
            return f"make_filled_simdgroup_matrix{fill_arguments}({rendered[0]})"
        if operation == "element":
            return f"{rendered[0]}.thread_elements()[{rendered[1]}]"
        if operation == "multiply":
            return f"({rendered[0]} * {rendered[1]})"
        if operation == "elementwise_add":
            return f"({rendered[0]} + {rendered[1]})"
        if operation == "elementwise_subtract":
            return f"({rendered[0]} - {rendered[1]})"
        if operation == "negate":
            return f"(-{rendered[0]})"
        if operation in {"load", "store"}:
            # The pointer operand names the memory it addresses. An element
            # such as ``input[64]`` is that element's address in the other
            # generators, but ``simdgroup_load`` and ``simdgroup_store`` take
            # a ``device``/``threadgroup`` pointer, so take the address here.
            # An element that is itself an array, such as ``tile[1]`` of a
            # ``float tile[4][64]``, already decays to the pointer.
            pointer_index = 1 if operation == "load" else 0
            pointer = arguments[pointer_index]
            pointee = str(self.expression_result_type(pointer) or "")
            if (
                isinstance(pointer, ArrayAccessNode)
                and "[" not in pointee
                and "*" not in pointee
                and not pointee.startswith("ptr<")
            ):
                rendered[pointer_index] = f"&{rendered[pointer_index]}"
        if operation == "store":
            rendered = [rendered[1], rendered[0], *rendered[2:]]
        intrinsic = self.METAL_COOPERATIVE_MATRIX_FUNCTIONS.get(operation)
        if intrinsic is not None:
            return f"{intrinsic}({', '.join(rendered)})"

        raise UnsupportedMetalFeatureError(
            "cooperative-matrix-operation-lowering",
            f"Metal codegen cannot lower cooperative-matrix operation "
            f"'{operation}' without changing its semantics",
            missing_capabilities=("metal.cooperative-matrix-operation-lowering",),
            operation=operation,
            reason="operation-not-representable",
            source_location=getattr(node, "source_location", None),
        )

    def generate_metal_buffer_resource_atomic_call(
        self, func_name, args, *, wide_result_discarded=False
    ):
        if func_name in self.user_function_names:
            return None
        operation_info = self.buffer_atomic_operations().get(func_name)
        if operation_info is None:
            return None

        operation, expected_arity = operation_info
        if len(args) != expected_arity:
            return self.unsupported_metal_buffer_resource_atomic_call(
                func_name,
                args[0] if args else None,
                f"requires {expected_arity} argument(s), got {len(args)}",
            )

        target = args[0]
        target_type = self.expression_result_type(target)
        mapped_target_type = self.map_type(target_type)
        wide = is_metal_wide_atomic(operation, mapped_target_type)
        if not wide and not self.buffer_atomic_supports_scalar_type(
            operation, mapped_target_type
        ):
            return self.unsupported_metal_buffer_resource_atomic_call(
                func_name,
                target,
                (
                    "requires a supported scalar device/threadgroup target, "
                    f"got {mapped_target_type or 'unknown'}"
                ),
            )

        address_space = self.argument_address_space(target)
        if address_space not in {"device", "threadgroup"}:
            return self.unsupported_metal_buffer_resource_atomic_call(
                func_name,
                target,
                "requires a device or threadgroup target",
            )

        root_name = self.assignment_target_root_name(target)
        readonly = (
            root_name in self.current_readonly_metal_parameters
            or root_name in self.current_readonly_raw_buffer_parameters
            or self.structured_buffer_type_name(
                self.local_variable_types.get(root_name)
            )
            == "StructuredBuffer"
        )
        if readonly and operation != "load":
            return self.unsupported_metal_buffer_resource_atomic_call(
                func_name, target, "requires writable storage"
            )

        target_expr = metal_vector_component_address(
            self, target, address_space, mapped_target_type
        )
        if target_expr is None:
            target_expr = self.generate_expression(target)
            if not self.is_metal_address_expression(target, target_expr):
                target_expr = f"&{target_expr}"
        if wide:
            return metal_wide_atomic_call(
                self,
                func_name,
                operation,
                target,
                target_expr,
                address_space,
                args[1],
                discarded=wide_result_discarded,
            )
        if operation == "compare_exchange":
            compare = self.generate_expression_with_expected(
                args[1], mapped_target_type
            )
            value = self.generate_expression_with_expected(args[2], mapped_target_type)
            self.required_pointer_atomic_compare_helpers.add(
                (address_space, mapped_target_type)
            )
            helper_name = self.pointer_atomic_compare_helper_name(
                address_space, mapped_target_type
            )
            return f"{helper_name}({target_expr}, {compare}, {value})"
        bitwise_float = (
            mapped_target_type == "float"
            and address_space == "threadgroup"
            and operation in {"load", "store", "compare_exchange_expected"}
        )
        atomic_type = f"atomic_{'uint' if bitwise_float else mapped_target_type}"
        qualifier = "const " if readonly else ""
        atomic_target = f"reinterpret_cast<{qualifier}{address_space} {atomic_type}*>({target_expr})"
        if operation == "load":
            load = f"atomic_load_explicit({atomic_target}, memory_order_relaxed)"
            return f"as_type<float>({load})" if bitwise_float else load
        if operation == "compare_exchange_expected":
            return self.generate_metal_expected_compare_exchange(
                atomic_target, args, mapped_target_type, bitwise_float=bitwise_float
            )
        value = self.generate_expression_with_expected(args[1], mapped_target_type)
        if bitwise_float:
            value = f"as_type<uint>({value})"
        return (
            f"atomic_{operation}_explicit("
            f"{atomic_target}, {value}, memory_order_relaxed)"
        )

    def format_parameter_declaration(
        self, raw_param_type, mapped_type, name, node=None, shader_type=None
    ):
        self.validate_metal_ray_resource_parameter_value_type(raw_param_type, name)

        geometry_stream_declaration = self.format_metal_geometry_stream_parameter(
            raw_param_type,
            name,
        )
        if geometry_stream_declaration is not None:
            return geometry_stream_declaration

        tessellation_patch_declaration = self.format_metal_tessellation_patch_parameter(
            raw_param_type, name
        )
        if tessellation_patch_declaration is not None:
            return tessellation_patch_declaration

        ray_payload_declaration = self.metal_ray_payload_parameter_declaration(
            mapped_type, name, node, shader_type
        )
        if ray_payload_declaration is not None:
            return ray_payload_declaration

        mesh_payload_declaration = self.metal_mesh_payload_parameter_declaration(
            mapped_type, name, node, shader_type
        )
        if mesh_payload_declaration is not None:
            return mesh_payload_declaration

        lowered_block = self.current_glsl_buffer_block_parameters.get(name)
        if lowered_block is not None:
            array_size = self.glsl_buffer_block_parameter_array_size(
                raw_param_type, node
            )
            return self.format_glsl_buffer_block_parameter(
                lowered_block, name, array_size
            )
        if self.is_structured_buffer_type(raw_param_type):
            array_size = self.structured_buffer_parameter_array_size(
                raw_param_type, node
            )
            return self.format_structured_buffer_parameter(
                raw_param_type, name, array_size, node
            )
        if self.is_visible_function_table_type(raw_param_type):
            return self.format_visible_function_table_parameter(raw_param_type, name)
        if self.is_intersection_function_table_type(raw_param_type):
            return self.format_intersection_function_table_parameter(
                raw_param_type, name
            )
        if self.is_bound_scalar_value_parameter(raw_param_type, node, shader_type):
            scalar_type = metal_buffer_scalar_type(
                self.type_name_string(raw_param_type), mapped_type
            )
            return f"constant {scalar_type}& {name}"
        address_space_declaration = self.format_address_space_parameter_declaration(
            raw_param_type, mapped_type, name, node, shader_type
        )
        if address_space_declaration is not None:
            return address_space_declaration
        array_type = self.resource_array_parameter(raw_param_type, node)
        function_name = self.current_function_name
        if array_type is not None:
            resource_type, array_size = array_type
            return self.format_resource_parameter(
                self.metal_coherent_texture_type(resource_type, function_name, name),
                name,
                array_size,
            )
        return format_c_style_array_declaration(
            self.metal_coherent_texture_type(mapped_type, function_name, name), name
        )

    def validate_address_space_call(self, func_name, call_args):
        if func_name not in self.user_function_names:
            return None
        overloads = self.function_overloads_by_name.get(func_name, [])
        selected_function = self.metal_user_function_overload_for_call(
            func_name, call_args
        )
        if overloads and selected_function is None:
            # Same-name, same-arity overloads require type-aware source overload
            # resolution. Leave those calls intact for native Metal resolution
            # rather than validating against arbitrary overwritten metadata.
            return None
        if selected_function is not None:
            parameter_nodes = list(
                getattr(
                    selected_function,
                    "parameters",
                    getattr(selected_function, "params", []),
                )
                or []
            )
        else:
            parameter_nodes = self.function_parameter_nodes.get(func_name, [])
        for index, arg in enumerate(call_args):
            if index >= len(parameter_nodes):
                continue
            parameter = parameter_nodes[index]
            raw_param_type = getattr(
                parameter, "param_type", getattr(parameter, "vtype", None)
            )
            expected_address_space = self.parameter_variable_address_space(
                raw_param_type, parameter
            )
            if expected_address_space is None:
                continue
            address_space_conflict = self.argument_address_space_conflict(arg)
            if address_space_conflict is not None:
                arg_name = self.assignment_target_display_name(arg) or "<expr>"
                parameter_name = getattr(parameter, "name", f"arg{index}")
                message = (
                    "Unsupported Metal address-space call: argument "
                    f"'{arg_name}' mixes "
                    f"{self.address_space_conflict_description(address_space_conflict)} "
                    f"but parameter '{parameter_name}' of '{func_name}' requires "
                    f"{expected_address_space}"
                )
                raise UnsupportedMetalFeatureError(
                    "address-space call",
                    message,
                    operation=func_name,
                    reason="mixed-pointer-address-spaces",
                    source_location=getattr(arg, "source_location", None),
                )
            actual_address_space = self.argument_address_space(arg)
            if (
                actual_address_space is None
                or actual_address_space == expected_address_space
            ):
                continue
            arg_name = self.assignment_target_display_name(arg)
            parameter_name = getattr(parameter, "name", f"arg{index}")
            message = (
                "Unsupported Metal address-space call: argument "
                f"'{arg_name}' uses {actual_address_space} address space but "
                f"parameter '{parameter_name}' of '{func_name}' requires "
                f"{expected_address_space}"
            )
            raise UnsupportedMetalFeatureError(
                "address-space call",
                message,
                operation=func_name,
                reason="incompatible-pointer-address-space",
                source_location=getattr(arg, "source_location", None),
            )
        return None

    def stage_entry_base_name(self, stage_name, func):
        func_name = getattr(func, "name", None) or "main"
        # Metal rejects an entry function called `main`, so it is renamed like
        # an entry without the attribute.
        if func_name != "main" and self.function_has_explicit_stage_entry_attribute(
            func
        ):
            return func_name
        if stage_name == "vertex":
            return f"vertex_{func_name}"
        if stage_name == "fragment":
            return f"fragment_{func_name}"
        if stage_name in {"compute", "ray_generation"}:
            return f"kernel_{func_name}"
        if stage_name == "geometry":
            return f"geometry_{func_name}"
        if stage_name == "tessellation_control":
            return f"tessellation_control_{func_name}"
        if stage_name == "tessellation_evaluation":
            return f"tessellation_evaluation_{func_name}"
        if stage_name in {"mesh", "object", "task", "amplification"}:
            stage_keyword = "mesh" if stage_name == "mesh" else "object"
            return f"{stage_keyword}_{func_name}"

        rt_stage_map = {
            "ray_intersection": "intersection",
            "ray_any_hit": "anyhit",
            "ray_closest_hit": "closesthit",
            "ray_miss": "miss",
            "ray_callable": "callable",
            "intersection": "intersection",
            "anyhit": "anyhit",
            "closesthit": "closesthit",
            "miss": "miss",
            "callable": "callable",
        }
        stage_keyword = rt_stage_map.get(stage_name)
        if stage_keyword:
            return f"{stage_keyword}_{func_name}"
        return func_name

    def buffer_atomic_operations(self):
        return {
            "atomicLoad": ("load", 1),
            "atomicCompareExchangeWeak": ("compare_exchange_expected", 3),
            "atomicStore": ("store", 2),
            "atomicAdd": ("fetch_add", 2),
            "atomicMin": ("fetch_min", 2),
            "atomicMax": ("fetch_max", 2),
            "atomicAnd": ("fetch_and", 2),
            "atomicOr": ("fetch_or", 2),
            "atomicXor": ("fetch_xor", 2),
            "atomicExchange": ("exchange", 2),
            "atomicCompSwap": ("compare_exchange", 3),
            "atomicCompareExchange": ("compare_exchange", 3),
        }

    @staticmethod
    def buffer_atomic_supports_scalar_type(operation, component_type):
        return component_type in {"int", "uint"} or (
            component_type == "float"
            and operation in {
                "load",
                "store",
                "fetch_add",
                "exchange",
                "compare_exchange_expected",
            }
        )

    def generate_glsl_buffer_block_atomic_call(self, func_name, args):
        operations = self.buffer_atomic_operations()
        operation_info = operations.get(func_name)
        if operation_info is None or not args:
            return None

        operation, expected_args = operation_info
        if len(args) < expected_args:
            return None

        target = args[0]
        access, offset = self.glsl_buffer_block_atomic_access(target)
        if access is None:
            return None
        if len(args) != expected_args:
            # Extra operands used to be dropped unread.
            return self.unsupported_glsl_buffer_block_atomic_call(
                target,
                func_name,
                f"requires {expected_args} argument(s), got {len(args)}",
                access,
            )
        if access.get("readonly"):
            return self.unsupported_glsl_buffer_block_atomic_call(
                target,
                func_name,
                "cannot write readonly device buffer",
                access,
            )
        if access.get("components") != 1 or access.get("matrix_columns"):
            return self.unsupported_glsl_buffer_block_atomic_call(
                target,
                func_name,
                "requires a scalar int or uint buffer member",
                access,
            )
        if not self.buffer_atomic_supports_scalar_type(
            operation, access.get("component_type")
        ):
            return self.unsupported_glsl_buffer_block_atomic_call(
                target,
                func_name,
                f"does not support {access.get('component_type')} buffer members",
                access,
            )

        if operation == "compare_exchange":
            memory = access.get("memory", "")
            self.required_buffer_atomic_compare_helpers.add(
                (access["component_type"], memory)
            )
            helper_name = self.buffer_atomic_compare_helper_name(
                access["component_type"], memory
            )
            compare_value = self.generate_expression_with_expected(
                args[1], access["type"]
            )
            replacement = self.generate_expression_with_expected(
                args[2], access["type"]
            )
            return (
                f"{helper_name}({access['buffer']}, {offset}, "
                f"{compare_value}, {replacement})"
            )

        atomic_type = f"atomic_{access['component_type']}"
        atomic_target = (
            f"reinterpret_cast<{access.get('memory', '')}device {atomic_type}*>"
            f"({access['buffer']} + {offset})"
        )
        if operation == "load":
            return f"atomic_load_explicit({atomic_target}, memory_order_relaxed)"
        if operation == "compare_exchange_expected":
            return self.generate_metal_expected_compare_exchange(
                atomic_target, args, access["component_type"]
            )
        value = self.generate_expression_with_expected(args[1], access["type"])
        return f"atomic_{operation}_explicit({atomic_target}, {value}, memory_order_relaxed)"

    def validate_texture_resource_argument(self, func_name, args):
        if not args or func_name not in self.texture_resource_operation_names():
            return
        if self.texture_resource_type(args[0]) is not None:
            return
        arg_type = self.expression_result_type(args[0])
        if arg_type is not None and self.is_texture_or_image_resource_type(arg_type):
            return

        texture_name = self.expression_name(args[0]) or str(args[0])
        shadow_type = self.resource_base_type(arg_type)
        if shadow_type in {
            "sampler1DShadow",
            "sampler1DArrayShadow",
            "texture1DShadow",
            "texture1DArrayShadow",
        }:
            # Metal has no one-dimensional depth texture: refuse with a
            # typed diagnostic naming the type instead of the untyped
            # "requires a declared texture" ValueError below.
            raise UnsupportedMetalFeatureError(
                "1d-depth-texture",
                "Metal has no one-dimensional depth texture for "
                f"'{texture_name}' of type '{shadow_type}'",
                missing_capabilities=("metal.1d-depth-texture",),
                operation=func_name,
                reason="unsupported-type",
                source_location=getattr(args[0], "source_location", None),
            )
        raise MetalRefusal(
            f"Metal texture operation '{func_name}' requires a declared "
            f"texture or image resource argument: {texture_name}"
        )

    def map_cooperative_matrix_type(self, matrix_type, contract):
        """Map the native subset of the canonical cooperative-matrix contract."""
        (
            element_type,
            rows,
            cols,
            scope,
            use,
            layout,
            fragment_layout,
            subgroup_size,
            elements_per_lane,
            fragment_provenance,
            _fragment_mapping,
            _fragment_mapping_provenance,
        ) = contract
        metadata = (scope, use, layout)
        if metadata != ("subgroup", "unspecified", "unspecified"):
            raise UnsupportedMetalFeatureError(
                "cooperative-matrix-type-contract",
                "Metal simdgroup_matrix cannot preserve cooperative-matrix "
                "scope/use/layout contract "
                f"{metadata!r}",
                missing_capabilities=("metal.cooperative-matrix-contract-mapping",),
                reason="unsupported-type-contract",
                source_location=getattr(matrix_type, "source_location", None),
            )

        def fragment_label(value):
            if value is None:
                return "unspecified"
            name = getattr(value, "name", None)
            return str(name if name is not None else value)

        fragment_metadata = (
            fragment_label(fragment_layout),
            fragment_label(subgroup_size),
            fragment_label(elements_per_lane),
            fragment_label(fragment_provenance),
        )
        if (
            fragment_metadata != (
                "unspecified",
                "unspecified",
                "unspecified",
                "unspecified",
            )
            and fragment_metadata[0] != "metal_thread_elements"
        ):
            raise UnsupportedMetalFeatureError(
                "cooperative-matrix-fragment-contract",
                "Metal simdgroup_matrix cannot preserve cooperative-matrix "
                f"fragment contract {fragment_metadata!r}",
                missing_capabilities=(
                    "metal.cooperative-matrix-fragment-contract-mapping",
                ),
                reason="unsupported-fragment-contract",
                source_location=getattr(matrix_type, "source_location", None),
            )

        mapped_element_type = self.map_type(element_type)
        rows_text = self.convert_type_node_to_string(rows)
        cols_text = self.convert_type_node_to_string(cols)
        return f"simdgroup_matrix<{mapped_element_type}, {rows_text}, {cols_text}>"

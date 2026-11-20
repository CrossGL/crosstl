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
    ArrayType,
    AssignmentNode,
    BinaryOpNode,
    BlockNode,
    BreakNode,
    CastNode,
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
    StructNode,
    SwitchNode,
    TernaryOpNode,
    UnaryOpNode,
    VariableNode,
    WaveOpNode,
    WhileNode,
)
from ..matrix_shape import length_method_object
from ..refusals import MetalRefusal
from ..resource_identity import resource_identity_marker
from ..semantics.unsized_arrays import UnsizedArrayUnsupportedError
from ..source_licenses import source_license_comments
from ..structure_conversions import (
    StructureConversionKind,
    StructureFieldValue,
    registered_structure_conversion_for_identity,
)
from .array_utils import (
    collect_literal_int_constants,
    collect_struct_member_types,
    format_c_style_array_declaration,
    get_array_size_from_node,
    split_array_type_suffix,
)
from .assignment_values import assignment_value_expression
from .bitfield_builtins import BITFIELD_ARITIES, metal_bitfield_call
from .boolean_mix import boolean_mix_selector, metal_boolean_mix
from .builtin_result_types import (
    C_FAMILY,
    boolean_builtin_result_type,
    component_access_type,
)
from .c_family_operands import postfix_operand
from .comma_expressions import (
    c_family_comma_expression,
    comma_result_operand,
    is_comma_expression,
)
from .constant_ordering import partition_constants_by_struct_dependency
from .enum_utils import (
    collect_enum_struct_variant_fields,
    collect_enum_type_names,
    collect_enum_variant_constants,
    collect_enum_variant_constructor_fields,
    collect_enum_variant_constructors,
    collect_generic_enum_specialization_member_types,
    collect_generic_enum_specializations,
    collect_generic_enum_struct_definitions,
    collect_generic_enum_variant_constants,
    collect_plain_enums,
    collect_struct_payload_enums,
    enum_value_expression,
    generate_enum_constants,
    generate_enum_constructor_call,
    generate_enum_constructor_expression,
    generate_generic_enum_constants,
    infer_enum_constructor_type,
)
from .exponent_builtins import emit_metal_exponent_helpers, metal_exponent_call
from .extended_arithmetic_builtins import (
    emit_metal_extended_arithmetic_helpers,
    metal_extended_arithmetic_call,
)
from .generic_emission_order import mark_generic_function_emitted
from .generic_function_utils import (
    generate_numeric_trait_method_call,
    generate_static_generic_numeric_call,
    generic_function_call_name,
    generic_function_emission_list,
    generic_function_parameters,
    numeric_trait_method_result_type,
    prepare_generic_function_specializations,
    reject_unresolved_generic_function_call,
)
from .generic_global_types import metal_generic_global_types
from .generic_struct_utils import (
    collect_generic_struct_definitions,
    collect_generic_struct_specialization_member_types,
    collect_generic_struct_specializations,
    generate_struct_constructor_expression,
    infer_struct_constructor_type,
)
from .glsl_buffer_array_values import glsl_buffer_array_value_type
from .glsl_buffer_block_arrays import (
    glsl_buffer_indexed_value_type,
    glsl_buffer_whole_array_reason,
)
from .glsl_buffer_layout import collect_lowered_glsl_buffer_blocks
from .image_access_contracts import (
    collect_function_image_access_requirements,
    collect_function_parameter_names,
    is_resource_size_query_operation,
    is_texel_fetch_operation,
    is_texture_sampling_operation,
    record_explicit_image_metadata,
)
from .match_utils import (
    generate_match_expression_assignment,
    infer_match_expression_result_type,
)
from .metal_array_arguments import MetalArrayArguments
from .metal_array_copies import generate_metal_array_initializer
from .metal_atomic_addresses import metal_vector_component_address
from .metal_bit_scans import emit_metal_bit_scan_helpers, metal_bit_scan_call
from .metal_buffer_coherence import collect_inherited_buffer_contracts
from .metal_buffer_component_stores import metal_glsl_buffer_component_refusal
from .metal_buffer_scalars import metal_buffer_scalar_type
from .metal_call_lowering import MetalCallLoweringMixin
from .metal_comparison_signs import metal_sign_matched_comparison_operands
from .metal_constant_initializers import (
    folded_helper_is_unused,
    metal_constant_initializer,
    restore_static_initializer_helpers,
)
from .metal_debug_printf import uses_debug_printf
from .metal_entry_return import (
    metal_entry_return_semantic_key,
    refuse_unreturnable_metal_entry_semantic,
)
from .metal_function_dependencies import MetalFunctionDependencyMixin
from .metal_glsl_buffer_blocks import MetalGLSLBufferBlockMixin
from .metal_hlsl_functions import MetalHlslFunctionMixin
from .metal_hlsl_matrix import hlsl_emulated_matrix_shape, metal_hlsl_matrix_prelude
from .metal_hlsl_resources import MetalHlslResourceMixin
from .metal_hlsl_values import MetalHlslValueMixin
from .metal_keyword_identifiers import METAL_KEYWORD_IDENTIFIERS
from .metal_local_declarations import MetalLocalDeclarationMixin
from .metal_matrix_builtins import MetalMatrixBuiltins
from .metal_mesh_outputs import MetalMeshOutputMixin
from .metal_narrow_vectors import MetalNarrowVectorMixin
from .metal_parameter_address_spaces import MetalParameterAddressSpaceMixin
from .metal_program_entries import MetalProgramEntryMixin
from .metal_ray_tracing import MetalRayTracingMixin
from .metal_reference_argument_conditions import generate_header_writeback_statement
from .metal_reference_argument_order import generate_ordered_writeback_statement
from .metal_reference_argument_returns import generate_return_writeback_statement
from .metal_reference_call_results import generate_call_result_statement
from .metal_reference_sequenced_calls import generate_conditional_call_statement
from .metal_refusals import metal_refusals
from .metal_render_walks import MetalRenderWalks
from .metal_resource_arguments import MetalResourceArgumentMixin
from .metal_safe_math import METAL_SAFE_MATH_PRAGMA
from .metal_signed_builtins import (
    collect_metal_signed_builtin_members,
    metal_signed_member_assignment,
)
from .metal_specialization_constants import (
    block_member_array_types,
    flattened_stage_io_array_types,
    specialization_constants_sizing_arrays,
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
from .metal_texture_coherence import collect_device_coherent_textures
from .metal_texture_operations import MetalTextureOperationMixin
from .metal_texture_validation import MetalTextureValidationMixin
from .metal_threadgroup_forwarding import collect_metal_forwarded_threadgroup_storage
from .metal_type_mapping import MetalTypeMappingMixin
from .metal_varyings import metal_varying_attribute
from .metal_wave_extended import emit_metal_wave_extension_helpers
from .metal_wave_operations import MetalWaveOperationMixin
from .pack_builtins import PACK_FUNCTIONS, metal_pack_call
from .relational_builtins import metal_logical_not_operand
from .sizeof_calls import sizeof_call_bytes
from .stage_io_semantic_locations import SEMANTIC_LOCATIONS
from .stage_utils import (
    collect_stage_entry_records,
    collect_stage_local_structs,
    collect_stage_local_variables,
    deduplicate_named_declarations,
    function_stage_name,
    normalize_stage_name,
    order_functions_by_dependencies,
    requests_early_fragment_tests,
    should_emit_qualified_function,
    stage_matches,
)
from .storage_image_texels import is_whole_texel_load
from .texture_lod_clamp import (
    is_texture_lod_clamp_operation,
    metal_texture_lod_clamp_call,
)
from .texture_residency import (
    TEXTURE_RESIDENCY_OPERATION,
    TEXTURE_RESIDENT_OPERATION,
    emit_metal_texture_residency_helpers,
    metal_texture_residency_call,
    metal_texture_resident_call,
)

FRAGMENT_INVOCATION_DENSITY_CAPABILITIES = (
    "glsl.extension.GL_EXT_fragment_invocation_density",
    "glsl.builtin.gl_FragSizeEXT",
)

# The value types of an MSL colour attachment output: a scalar or a vector of up to four components.
METAL_COLOR_ATTACHMENT_TYPE = re.compile(
    r"(?:float|half|int|uint|short|ushort|char|uchar)[234]?"
)


class UnsupportedMetalFeatureError(ValueError):
    """Raised when Metal has no equivalent target representation."""

    project_diagnostic_code = "project.translate.unsupported-feature"

    def __init__(
        self,
        feature,
        message,
        *,
        missing_capabilities=(),
        operation=None,
        reason=None,
        source_location=None,
    ):
        super().__init__(message)
        self.feature = feature
        self.missing_capabilities = tuple(missing_capabilities)
        self.operation = operation
        self.reason = reason
        self.source_location = source_location


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


# Entry stages whose threadgroup memory Metal declares inside the entry function, so a
# program-scope groupshared value used only by such entries becomes an entry local.
METAL_GROUPSHARED_ENTRY_STAGES = frozenset(
    {"compute", "task", "amplification", "object", "mesh"}
)


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
        self.metal_type_aliases = self.collect_metal_type_aliases(ast)
        self.metal_fenced_image_names = self.collect_metal_read_after_write_images(ast)
        self.metal_program_identifier_names = (
            self.collect_metal_program_identifier_names(ast)
        )

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
        self.metal_program_scope_value_globals = set()
        self.metal_program_scope_groupshared_globals = set()
        self.metal_lowered_program_scope_groupshared_globals_by_function = {}
        self.metal_lowered_program_scope_groupshared_global_ids = set()
        self.metal_forwarded_thread_private_globals = {}
        self.metal_program_scope_value_global_types = {}
        self.program_scope_sampler_constants = set()
        self.metal_specialization_macro_constants = set()
        self.glsl_buffer_block_variables = []
        self.lowered_glsl_buffer_blocks = {}
        self.lowered_glsl_buffer_block_struct_names = set()
        self.glsl_buffer_block_lowering_failures = {}
        self.glsl_buffer_block_struct_lowering_failures = {}
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
        self.hlsl_program_constant_global_ids = set()
        self.cbuffer_variables = list(getattr(ast, "cbuffers", []) or [])
        hlsl_program_constants = self.synthetic_hlsl_program_constants_cbuffer(ast)
        if hlsl_program_constants is not None:
            self.cbuffer_variables.append(hlsl_program_constants)
        self.cbuffer_binding_indices = {}
        self.cbuffers_by_name = {
            cbuffer.name: cbuffer
            for cbuffer in self.cbuffer_variables
            if getattr(cbuffer, "name", None)
        }
        all_functions = self.all_functions(ast)
        self.metal_constant_conversion_operators = (
            self.collect_metal_constant_conversion_operators(all_functions)
        )
        for struct_name, receivers in (
            getattr(ast, "metal_constant_conversion_operators", None) or {}
        ).items():
            known = self.metal_constant_conversion_operators.setdefault(struct_name, [])
            known += [item for item in receivers if item not in known]
        self.validate_metal_mesh_payload_parameter_placement(ast, all_functions)
        self.user_function_names = {
            func.name for func in all_functions if getattr(func, "name", None)
        }
        self.metal_atomic_fence_calls = self.collect_metal_atomic_fence_calls(ast)
        self.requires_metal_atomic_fence = bool(self.metal_atomic_fence_calls)
        self.requires_metal_system_thread_scope = any(
            self.atomic_fence_operand_identifier(call.args[2]) == "thread_scope_system"
            for call in self.metal_atomic_fence_calls
            if len(call.args) == 3
        )
        self.metal_resource_memory_contracts = (
            self.collect_metal_resource_memory_contracts(ast)
        )
        self.requires_metal_resource_coherence = any(
            kind == "coherent" for kind, _scope in self.metal_resource_memory_contracts
        )
        self.functions_by_name = {
            func.name: func for func in all_functions if getattr(func, "name", None)
        }
        self.function_overloads_by_name = {}
        for func in all_functions:
            func_name = getattr(func, "name", None)
            if func_name:
                self.function_overloads_by_name.setdefault(func_name, []).append(func)
        self.function_parameter_infos = self.collect_function_parameter_infos(
            all_functions
        )
        self.function_parameter_nodes = self.collect_function_parameter_nodes(
            all_functions
        )
        self.metal_skipped_function_parameter_indices = (
            self.collect_unused_array_parameter_indices(all_functions)
        )
        self.function_return_types = {
            func.name: self.metal_effective_function_return_type(func)
            for func in all_functions
            if getattr(func, "name", None)
        }
        self.function_parameter_names = collect_function_parameter_names(all_functions)
        self.function_image_access_requirements = (
            collect_function_image_access_requirements(
                all_functions,
                self.function_parameter_names,
                self.iter_ast_nodes,
                self.function_call_name,
                self.expression_name,
            )
        )
        self.cbuffer_parameter_names = self.collect_cbuffer_parameter_names(
            self.cbuffer_variables
        )
        self.cbuffer_member_references = self.collect_cbuffer_member_references(
            self.cbuffer_variables
        )
        if not self.cbuffer_variables:
            self.ambiguous_cbuffer_members = set()
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
        self.function_global_resource_dependencies = {}
        self.function_stage_parameter_dependencies = {}
        self.function_metal_wave_lane_dependencies = {}
        self.function_metal_wave_lane_parameter_names = {}
        self.unsupported_glsl_buffer_block_functions = {}
        self.unsupported_glsl_buffer_block_struct_names = set()
        self.required_image_atomic_compare_helpers = set()
        self.required_pointer_atomic_compare_helpers = set()
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
        self.required_glsl_buffer_aggregate_load_helpers = {}
        self.current_glsl_buffer_block_parameters = {}
        self.unsupported_glsl_buffer_block_variables = set()
        self.unsupported_glsl_buffer_block_variable_types = {}
        self.current_unsupported_glsl_buffer_block_parameters = set()
        self.current_unsupported_glsl_buffer_block_local_variables = set()
        self.metal_emulated_matrix_shapes = set()
        self.current_glsl_buffer_block_parameter_failures = {}
        self.current_glsl_buffer_block_parameter_struct_failures = {}
        self.current_metal_mesh_output_config = None
        self.current_metal_mesh_output_parameter = None
        self.current_metal_mesh_grid_properties_parameter = None
        self.current_metal_mesh_payload_parameter = None
        self.current_metal_mesh_payload_type = None
        self.current_metal_mesh_output_accumulators = {}
        self.current_metal_non_thread_payload_parameters = set()
        self.metal_ray_payload_parameter_types = set()
        self.function_metal_mesh_dispatch_contexts = {}
        self.function_metal_mesh_dispatch_contexts_by_id = {}
        self.metal_program_mesh_payload_type = self.metal_mesh_payload_type_for_program(
            ast
        )
        self.literal_int_constants = collect_literal_int_constants(
            getattr(ast, "constants", [])
        )
        # Declared types of program-scope constants, so expressions that read
        # them (such as an array constant copied into an array member) are typed.
        self.metal_program_scope_constant_types = {
            node.name: self.type_name_string(
                getattr(node, "const_type", getattr(node, "vtype", None))
            )
            for node in getattr(ast, "constants", []) or []
            if getattr(node, "name", None)
        }
        structs = deduplicate_named_declarations(
            list(getattr(ast, "structs", []) or [])
            + collect_stage_local_structs(ast, target_stage),
            "struct",
        )
        self.structs_by_name = {
            node.name: node for node in structs if isinstance(node, StructNode)
        }
        self.metal_static_struct_members = {
            node.name: {
                getattr(member, "name", None)
                for member in getattr(node, "members", []) or []
                if any(
                    self.normalized_metal_abi_attribute_name(attribute) == "static"
                    for attribute in getattr(member, "attributes", []) or []
                )
            }
            for node in structs
            if isinstance(node, StructNode)
        }
        self.metal_union_layouts = {}
        for node in structs:
            layout = self.metal_union_layout_contract(node)
            if layout is not None:
                self.metal_union_layouts[node.name] = layout
        self.collect_metal_hlsl_buffer_layouts(ast)
        if self.metal_hlsl_cbuffer_plans:
            self.cbuffer_member_references = self.collect_cbuffer_member_references(
                self.cbuffer_variables
            )
        self.metal_generated_struct_names = set()
        self.metal_vertex_entry_input_struct_names = (
            self.collect_metal_stage_entry_parameter_struct_names(
                ast, target_stage, "vertex"
            )
        )
        # A mesh function's vertex and primitive structures feed the fragment ``[[stage_in]]`` as a vertex
        # function's output does, and link by the same ``[[user(...)]]`` keys (:mod:`.metal_varyings`).
        self.metal_vertex_entry_output_struct_names = (
            self.collect_metal_stage_entry_return_struct_names(
                ast, target_stage, "vertex"
            )
            | self.collect_metal_mesh_output_struct_names(ast, target_stage, "vertices")
        )
        self.metal_mesh_primitive_struct_names = (
            self.collect_metal_mesh_output_struct_names(ast, target_stage, "primitives")
        )
        self.metal_fragment_entry_input_struct_names = (
            self.collect_metal_stage_entry_parameter_struct_names(
                ast, target_stage, "fragment"
            )
        )
        self.metal_fragment_entry_output_struct_names = (
            self.collect_metal_stage_entry_return_struct_names(
                ast, target_stage, "fragment"
            )
        )
        self.metal_stage_io_struct_names = (
            set(self.metal_vertex_entry_input_struct_names)
            | set(self.metal_vertex_entry_output_struct_names)
            | set(self.metal_fragment_entry_input_struct_names)
            | set(self.metal_fragment_entry_output_struct_names)
        )
        self.metal_ray_payload_parameter_types = (
            self.collect_metal_ray_payload_parameter_types(getattr(ast, "stages", {}))
        )
        self.generic_enum_struct_definitions = collect_generic_enum_struct_definitions(
            structs
        )
        self.generic_struct_definitions = collect_generic_struct_definitions(
            structs,
            excluded_names=set(self.generic_enum_struct_definitions),
        )
        self.generic_enum_specializations = collect_generic_enum_specializations(
            ast,
            self.generic_enum_struct_definitions,
            self.type_name_string,
        )
        self.generic_struct_specializations = collect_generic_struct_specializations(
            ast,
            self.generic_struct_definitions,
            self.type_name_string,
        )
        self.plain_enums = collect_plain_enums(structs)
        self.struct_payload_enums = collect_struct_payload_enums(structs)
        self.enum_type_names = collect_enum_type_names(self.plain_enums)
        self.enum_struct_type_names = (
            collect_enum_type_names(self.struct_payload_enums)
            | set(self.generic_enum_struct_definitions)
            | {
                specialization["struct_name"]
                for specialization in self.generic_enum_specializations.values()
            }
        )
        self.enum_struct_variant_fields = collect_enum_struct_variant_fields(
            self.struct_payload_enums
        )
        self.enum_variant_constructors = collect_enum_variant_constructors(
            self.struct_payload_enums
        )
        self.enum_variant_constructor_fields = collect_enum_variant_constructor_fields(
            self.struct_payload_enums
        )
        self.enum_variant_constants = {
            **collect_enum_variant_constants(
                self.plain_enums + self.struct_payload_enums
            ),
            **collect_generic_enum_variant_constants(
                self.generic_enum_struct_definitions
            ),
        }
        stage_local_resource_variables = collect_stage_local_variables(
            ast, target_stage, self.is_stage_local_resource_variable
        )
        self.metal_source_binding_stage_by_id = (
            self.collect_stage_local_resource_stage_scopes(
                ast, target_stage, stage_local_resource_variables
            )
        )
        global_vars = deduplicate_named_declarations(
            [
                node
                for node in list(getattr(ast, "global_variables", []) or [])
                if not self.is_metal_type_alias_declaration(node)
            ]
            + stage_local_resource_variables,
            "Metal resource",
        )
        self.metal_lowered_program_scope_groupshared_globals_by_function = (
            self.collect_metal_lowered_program_scope_groupshared_globals(
                ast, target_stage, global_vars, all_functions
            )
        )
        self.metal_lowered_program_scope_groupshared_global_ids = {
            id(node)
            for nodes in (
                self.metal_lowered_program_scope_groupshared_globals_by_function.values()
            )
            for node in nodes
        }
        thread_private_globals, thread_private_ids = (
            self.collect_metal_lowered_thread_private_globals(
                ast, target_stage, global_vars, all_functions
            )
        )
        for function_id, nodes in thread_private_globals.items():
            self.metal_lowered_program_scope_groupshared_globals_by_function.setdefault(
                function_id, []
            ).extend(nodes)
        self.metal_lowered_program_scope_groupshared_global_ids.update(
            thread_private_ids
        )
        threadgroup_storage, threadgroup_ids = (
            collect_metal_forwarded_threadgroup_storage(
                self,
                ast,
                target_stage,
                global_vars,
                all_functions,
                METAL_GROUPSHARED_ENTRY_STAGES,
            )
        )
        for function_id, nodes in threadgroup_storage.items():
            self.metal_lowered_program_scope_groupshared_globals_by_function.setdefault(
                function_id, []
            ).extend(nodes)
        self.metal_lowered_program_scope_groupshared_global_ids.update(threadgroup_ids)
        self.metal_runtime_initialized_constants = (
            self.collect_metal_runtime_initialized_constants(global_vars)
        )
        self.metal_spirv_interface_variables = [
            node for node in global_vars if self.is_spirv_stage_interface_layout(node)
        ]
        self.glsl_buffer_block_struct_names = (
            self.collect_glsl_buffer_block_struct_names(
                list(global_vars) + self.collect_function_parameters(all_functions)
            )
        )
        (
            self.resource_array_size_hints,
            self.function_resource_array_size_hints,
        ) = self.collect_resource_array_size_hints(ast)
        self.function_structured_buffer_length_dependencies = (
            self.collect_function_structured_buffer_length_dependencies(all_functions)
        )
        self.global_structured_buffer_length_dependencies = (
            self.collect_global_structured_buffer_length_dependencies(
                all_functions, global_vars
            )
        )
        self.validate_global_resource_shadows(ast)
        self.current_function_name = None
        self.current_function_return_type = None
        self.current_function_return_wrapper = None
        self.current_expression_expected_type = None
        self.suppress_image_load_component_suffix = False
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
        self.local_variable_types = {}
        self.current_metal_native_width_local_array_declaration_ids = set()
        self.current_metal_graphics_builtin_parameter_names = {}
        (
            self.lowered_glsl_buffer_blocks,
            self.glsl_buffer_block_lowering_failures,
            self.glsl_buffer_block_struct_lowering_failures,
        ) = collect_lowered_glsl_buffer_blocks(
            global_vars,
            structs_by_name=self.structs_by_name,
            is_glsl_buffer_block_variable=self.is_glsl_buffer_block_variable,
            resource_base_type=self.resource_base_type,
            glsl_buffer_block_layout=self.glsl_buffer_block_layout,
            convert_type_node_to_string=self.convert_type_node_to_string,
            literal_int_value=lambda expr: self.literal_int_value(
                expr, self.literal_int_constants
            ),
            map_type=self.map_type,
            target_type_key="metal_type",
            unsupported_type_message=(
                "type is not supported by Metal pointer/offset lowering"
            ),
            component_refusal=metal_glsl_buffer_component_refusal,
        )
        self.record_metal_glsl_buffer_block_layout_failures(
            global_vars,
            self.lowered_glsl_buffer_blocks,
            self.glsl_buffer_block_lowering_failures,
            self.glsl_buffer_block_struct_lowering_failures,
        )
        self.lowered_glsl_buffer_block_struct_names = {
            block["type_name"] for block in self.lowered_glsl_buffer_blocks.values()
        }
        for node in global_vars:
            block = self.lowered_glsl_buffer_blocks.get(getattr(node, "name", None))
            if block is not None:
                block["memory"] = self.resource_memory_qualifier_prefix(
                    node, getattr(node, "var_type", None)
                )
        self.metal_specialization_macro_constants = (
            specialization_constants_sizing_arrays(
                ast,
                lambda node: self.metal_specialization_constant_id(node) is not None,
                flattened_stage_io_array_types(
                    structs,
                    self.metal_stage_io_struct_names,
                    self.metal_stage_io_member_array_info,
                )
                + block_member_array_types(
                    self.cbuffer_variables,
                    structs,
                    self.lowered_glsl_buffer_block_struct_names,
                ),
            )
        )
        self.unsupported_glsl_buffer_block_struct_names = set(
            self.glsl_buffer_block_struct_lowering_failures
        )
        _, _, parameter_struct_failures = (
            self.collect_lowered_glsl_buffer_block_parameters(
                self.collect_function_parameters(all_functions)
            )
        )
        self.unsupported_glsl_buffer_block_struct_names.update(
            parameter_struct_failures
        )
        self.unsupported_glsl_buffer_block_functions = (
            self.collect_unsupported_glsl_buffer_block_functions(all_functions)
        )
        self.struct_member_types = collect_struct_member_types(
            structs, self.type_name_string
        )
        for owner, names in self.metal_static_struct_members.items():
            for name in names:
                self.struct_member_types.get(owner, {}).pop(name, None)
        self.struct_member_address_spaces = self.collect_struct_member_address_spaces(
            structs
        )
        self.struct_member_types.update(
            collect_generic_enum_specialization_member_types(
                self,
                self.generic_enum_specializations,
            )
        )
        self.struct_member_types.update(
            collect_generic_struct_specialization_member_types(
                self,
                self.generic_struct_specializations,
            )
        )
        generic_function_specializations = prepare_generic_function_specializations(
            self,
            all_functions,
            {
                **getattr(self, "metal_program_scope_constant_types", {}),
                **getattr(self, "metal_cbuffer_member_types", {}),
                **metal_generic_global_types(self, global_vars),
            },
            ast=ast,
        )
        self.function_return_types.update(
            {
                func.name: self.metal_effective_function_return_type(func)
                for func in generic_function_specializations.values()
            }
        )
        self.user_function_names.update(
            func.name for func in generic_function_specializations.values()
        )
        if generic_function_specializations:
            self.function_parameter_names.update(
                collect_function_parameter_names(
                    generic_function_specializations.values()
                )
            )
            self.function_parameter_nodes.update(
                self.collect_function_parameter_nodes(
                    generic_function_specializations.values()
                )
            )
            generic_specialization_roots = [
                ast,
                *generic_function_specializations.values(),
            ]
            self.generic_enum_specializations = collect_generic_enum_specializations(
                generic_specialization_roots,
                self.generic_enum_struct_definitions,
                self.type_name_string,
            )
            self.generic_struct_specializations = (
                collect_generic_struct_specializations(
                    generic_specialization_roots,
                    self.generic_struct_definitions,
                    self.type_name_string,
                )
            )
            for func in generic_function_specializations.values():
                self.add_metal_generic_specializations_from_type_text(
                    self.type_name_string(getattr(func, "return_type", None))
                )
                for parameter in getattr(
                    func, "parameters", getattr(func, "params", [])
                ):
                    self.add_metal_generic_specializations_from_type_text(
                        self.type_name_string(getattr(parameter, "param_type", None))
                    )
            self.enum_struct_type_names = (
                collect_enum_type_names(self.struct_payload_enums)
                | set(self.generic_enum_struct_definitions)
                | {
                    specialization["struct_name"]
                    for specialization in self.generic_enum_specializations.values()
                }
            )
            self.enum_variant_constants = {
                **collect_enum_variant_constants(
                    self.plain_enums + self.struct_payload_enums
                ),
                **collect_generic_enum_variant_constants(
                    self.generic_enum_struct_definitions
                ),
            }
            self.struct_member_types.update(
                collect_generic_enum_specialization_member_types(
                    self,
                    self.generic_enum_specializations,
                )
            )
            self.struct_member_types.update(
                collect_generic_struct_specialization_member_types(
                    self,
                    self.generic_struct_specializations,
                )
            )
        # Collect cbuffer dependencies after generic specializations exist, so a
        # specialization that reads a cbuffer member (e.g. `scaled_float` reading
        # `gain`) gets the cbuffer threaded as a parameter like any other helper.
        # The unspecialized generic templates are excluded, mirroring the
        # global-resource collection below: emission and call sites query the
        # specialization name, not the template name.
        self.function_cbuffer_dependencies = self.collect_function_cbuffer_dependencies(
            [f for f in all_functions if not generic_function_parameters(f)]
            + list(generic_function_specializations.values())
        )
        wave_dependency_functions = list(all_functions) + list(
            generic_function_specializations.values()
        )
        self.function_metal_wave_lane_dependencies = (
            self.collect_function_metal_wave_lane_dependencies(
                wave_dependency_functions
            )
        )
        self.function_metal_wave_lane_parameter_names = (
            self.collect_function_metal_wave_lane_parameter_names(
                wave_dependency_functions
            )
        )
        self.function_metal_mesh_dispatch_contexts = (
            self.collect_function_metal_mesh_dispatch_contexts(
                list(all_functions) + list(generic_function_specializations.values())
            )
        )
        self.metal_stage_io_member_lowerings = (
            self.collect_metal_stage_io_member_lowerings(structs)
        )
        self.metal_signed_builtin_members = collect_metal_signed_builtin_members(
            self, structs
        )
        code = "\n"
        code += source_license_comments(ast, "metal")
        if getattr(self, "metal_safe_math", False):
            code += METAL_SAFE_MATH_PRAGMA + "\n"
        preprocessors = getattr(ast, "preprocessors", []) or []
        pre_lines = []
        for directive in preprocessors:
            line = self.generate_preprocessor_directive(directive)
            if line:
                pre_lines.append(line)
        if (
            self.requires_metal_system_thread_scope
            or self.requires_metal_resource_coherence
        ) and not any("#pragma metal internals" in line.lower() for line in pre_lines):
            code += "#pragma METAL internals : enable\n"
        if pre_lines:
            code += "\n".join(pre_lines) + "\n"
        if not any("metal_stdlib" in line for line in pre_lines):
            code += "#include <metal_stdlib>\n"
        if self.requires_metal_atomic_fence and not any(
            "metal_atomic" in line for line in pre_lines
        ):
            code += "#include <metal_atomic>\n"
        if uses_debug_printf(self, ast) and not any(
            "metal_logging" in line for line in pre_lines
        ):
            code += "#include <metal_logging>\n"
        if self.uses_cooperative_matrix(ast) and not any(
            "metal_simdgroup_matrix" in line for line in pre_lines
        ):
            code += "#include <metal_simdgroup_matrix>\n"
        code += "using namespace metal;\n"
        if self.uses_metal_raytracing_namespace(ast, global_vars, all_functions):
            code += "using namespace metal::raytracing;\n"
        code += "\n"
        self.metal_prelude_insertion_index = len(code)
        if self.requires_metal_system_thread_scope:
            code += self.generate_metal_system_thread_scope_support()
        code += generate_enum_constants(
            self,
            self.plain_enums + self.struct_payload_enums,
            qualifier=self.metal_unused_declaration_qualifier("constant"),
        )
        code += generate_generic_enum_constants(
            self,
            self.generic_enum_struct_definitions,
            qualifier=self.metal_unused_declaration_qualifier("constant"),
        )
        leading_constants, struct_dependent_constants = (
            partition_constants_by_struct_dependency(
                getattr(ast, "constants", []) or [], structs
            )
        )
        code += self.generate_constants(ast, leading_constants)
        code += self.generate_metal_builtin_limit_fallbacks(ast, all_functions)
        code += self.generate_metal_struct_declarations(structs)
        code += self.generate_constants(
            ast, struct_dependent_constants, include_function_constants=False
        )
        code += self.generate_metal_stage_io_array_helpers()
        code += self.generate_metal_enum_constructor_functions(
            self.struct_payload_enums
        )
        code += self.generate_metal_generic_enum_constructor_functions(
            self.generic_enum_specializations
        )

        texture_register = 0
        sampler_register = 0
        buffer_register = 0
        used_resource_bindings = {}
        source_resource_bindings = {}
        self.pre_reserve_cbuffer_bindings(
            used_resource_bindings, source_resource_bindings
        )
        self.reserve_explicit_global_resource_bindings(
            global_vars, used_resource_bindings, source_resource_bindings
        )
        buffer_register = self.reserve_cbuffer_bindings(
            used_resource_bindings,
            source_resource_bindings,
        )
        for i, node in enumerate(global_vars):
            if self.is_spirv_stage_interface_layout(node):
                continue

            resource_count = 1
            array_size = None
            if getattr(node, "var_type", None) is not None:
                if (
                    hasattr(node.var_type, "name")
                    or hasattr(node.var_type, "element_type")
                    or isinstance(node.var_type, (PointerType, ReferenceType))
                ):
                    if (
                        hasattr(node.var_type, "element_type")
                        and str(type(node.var_type)).find("ArrayType") != -1
                    ):  # ArrayType
                        base_type = self.convert_type_node_to_string(
                            node.var_type.element_type
                        )
                        # An array of arrays: the outer dimension is the first suffix.
                        base_type, element_suffix = split_array_type_suffix(base_type)
                        array_size = (
                            self.expression_to_string(node.var_type.size)
                            if node.var_type.size
                            else self.resource_array_size_hints.get(node.name, "")
                        )
                        vtype = base_type
                        array_suffix = (
                            f"[{array_size}]" if array_size else "[]"
                        ) + element_suffix
                        resource_count = self.resource_array_count(
                            node.var_type.size if node.var_type.size else array_size
                        )
                        element_type = node.var_type.element_type
                        while isinstance(element_type, ArrayType):
                            resource_count *= self.resource_array_count(
                                element_type.size
                            )
                            element_type = element_type.element_type
                    else:
                        vtype = self.convert_type_node_to_string(node.var_type)
                        array_suffix = ""
                else:
                    vtype = str(node.var_type)
                    array_suffix = ""
            elif getattr(node, "vtype", None) is not None:
                vtype = node.vtype
                array_suffix = ""
            else:
                # No written type: take it from the initializer so a shader-scope
                # `let scale = 2.0` is `constant float scale = 2.0`, not
                # `constant None scale` (#2444).
                vtype = self.infer_metal_global_value_type(node)
                array_suffix = ""

            var_name = getattr(node, "name", getattr(node, "variable_name", None))
            if id(node) in self.hlsl_program_constant_global_ids:
                continue
            if id(node) in self.metal_lowered_program_scope_groupshared_global_ids:
                continue
            if self.is_metal_function_constant_variable(node):
                continue

            if self.is_metal_argument_buffer_global(node):
                binding = self.metal_argument_buffer_binding(node)
                declaration = self.format_metal_argument_buffer_global(
                    node, vtype, var_name
                )
                binding = self.reserve_or_remap_resource_binding(
                    used_resource_bindings,
                    source_resource_bindings,
                    "buffer",
                    binding,
                    resource_count,
                    var_name,
                    node,
                    buffer_register,
                )
                code += declaration
                buffer_register = max(buffer_register, binding + resource_count)
                continue

            lowered_block = self.lowered_glsl_buffer_blocks.get(var_name)
            if lowered_block is not None:
                binding = self.explicit_resource_binding_index(
                    node, {"binding", "buffer"}, ("b", "u", "t")
                )
                if binding is None:
                    binding = self.next_available_resource_binding(
                        used_resource_bindings,
                        "buffer",
                        buffer_register,
                        resource_count,
                    )
                binding = self.reserve_or_remap_resource_binding(
                    used_resource_bindings,
                    source_resource_bindings,
                    "buffer",
                    binding,
                    resource_count,
                    var_name,
                    node,
                    buffer_register,
                )
                self.glsl_buffer_block_variables.append(
                    (node, binding, lowered_block, array_size)
                )
                buffer_register = max(buffer_register, binding + resource_count)
                continue

            if self.is_glsl_buffer_block_variable(node, vtype):
                code += self.glsl_buffer_block_diagnostic(
                    "Metal", vtype, var_name, node
                )
                self.unsupported_glsl_buffer_block_variables.add(var_name)
                self.unsupported_glsl_buffer_block_variable_types[var_name] = (
                    self.type_name_string(vtype)
                )
                code += self.unsupported_glsl_buffer_block_variable_placeholder(
                    "Metal", vtype, var_name
                )
                continue

            metal_buffer_address_space = self.metal_buffer_resource_address_space(
                node, vtype
            )
            if metal_buffer_address_space is not None:
                binding_count = self.metal_buffer_resource_binding_count(
                    node, array_size
                )
                binding = self.explicit_resource_binding_index(
                    node, {"binding", "buffer"}, ("b", "u", "t")
                )
                if binding is None:
                    binding = self.next_available_resource_binding(
                        used_resource_bindings,
                        "buffer",
                        buffer_register,
                        binding_count,
                    )
                binding = self.reserve_or_remap_resource_binding(
                    used_resource_bindings,
                    source_resource_bindings,
                    "buffer",
                    binding,
                    binding_count,
                    var_name,
                    node,
                    buffer_register,
                )
                self.metal_buffer_resource_variables.append(
                    (node, binding, vtype, array_size, metal_buffer_address_space)
                )
                buffer_register = max(buffer_register, binding + binding_count)
                continue

            if self.is_acceleration_structure_type(vtype):
                binding = self.explicit_resource_binding_index(
                    node, {"binding", "buffer"}, ("b", "u", "t")
                )
                if binding is None:
                    binding = self.next_available_resource_binding(
                        used_resource_bindings,
                        "buffer",
                        buffer_register,
                        resource_count,
                    )
                binding = self.reserve_or_remap_resource_binding(
                    used_resource_bindings,
                    source_resource_bindings,
                    "buffer",
                    binding,
                    resource_count,
                    var_name,
                    node,
                    buffer_register,
                )
                mapped_type = self.map_resource_type_with_format(vtype, node)
                if array_size is not None:
                    self.unsupported_metal_acceleration_structure_array_variables[
                        var_name
                    ] = mapped_type
                    code += (
                        self.unsupported_metal_acceleration_structure_array_diagnostic(
                            var_name
                        )
                    )
                    buffer_register = max(buffer_register, binding + resource_count)
                    continue
                self.acceleration_structure_variables.append(
                    (node, binding, mapped_type, array_size)
                )
                buffer_register = max(buffer_register, binding + resource_count)
                continue

            if self.is_visible_function_table_type(vtype):
                binding = self.explicit_resource_binding_index(
                    node, {"binding", "buffer"}, ("b", "u", "t")
                )
                if binding is None:
                    binding = self.next_available_resource_binding(
                        used_resource_bindings,
                        "buffer",
                        buffer_register,
                        resource_count,
                    )
                binding = self.reserve_or_remap_resource_binding(
                    used_resource_bindings,
                    source_resource_bindings,
                    "buffer",
                    binding,
                    resource_count,
                    var_name,
                    node,
                    buffer_register,
                )
                if array_size is not None:
                    self.unsupported_metal_ray_function_table_array_variables[
                        var_name
                    ] = "visible_function_table"
                    code += self.unsupported_metal_ray_function_table_array_diagnostic(
                        "visible_function_table", var_name
                    )
                    buffer_register = max(buffer_register, binding + resource_count)
                    continue
                mapped_type = self.map_visible_function_table_type(vtype)
                self.visible_function_table_variables.append(
                    (node, binding, mapped_type, array_size)
                )
                buffer_register = max(buffer_register, binding + resource_count)
                continue

            if self.is_intersection_function_table_type(vtype):
                binding = self.explicit_resource_binding_index(
                    node, {"binding", "buffer"}, ("b", "u", "t")
                )
                if binding is None:
                    binding = self.next_available_resource_binding(
                        used_resource_bindings,
                        "buffer",
                        buffer_register,
                        resource_count,
                    )
                binding = self.reserve_or_remap_resource_binding(
                    used_resource_bindings,
                    source_resource_bindings,
                    "buffer",
                    binding,
                    resource_count,
                    var_name,
                    node,
                    buffer_register,
                )
                if array_size is not None:
                    self.unsupported_metal_ray_function_table_array_variables[
                        var_name
                    ] = "intersection_function_table"
                    code += self.unsupported_metal_ray_function_table_array_diagnostic(
                        "intersection_function_table", var_name
                    )
                    buffer_register = max(buffer_register, binding + resource_count)
                    continue
                mapped_type = self.map_intersection_function_table_type(vtype)
                self.intersection_function_table_variables.append(
                    (node, binding, mapped_type, array_size)
                )
                buffer_register = max(buffer_register, binding + resource_count)
                continue

            if self.is_structured_buffer_type(vtype):
                binding = self.explicit_resource_binding_index(
                    node, {"binding", "buffer"}, ("b", "u", "t")
                )
                if binding is None:
                    binding = self.next_available_resource_binding(
                        used_resource_bindings,
                        "buffer",
                        buffer_register,
                        resource_count,
                    )
                binding = self.reserve_or_remap_resource_binding(
                    used_resource_bindings,
                    source_resource_bindings,
                    "buffer",
                    binding,
                    resource_count,
                    var_name,
                    node,
                    buffer_register,
                )
                self.structured_buffer_variables.append(
                    (node, binding, vtype, array_size)
                )
                buffer_register = max(buffer_register, binding + resource_count)
                if self.structured_buffer_requires_length(var_name):
                    length_binding = self.next_available_resource_binding(
                        used_resource_bindings,
                        "buffer",
                        buffer_register,
                        resource_count,
                    )
                    length_name = self.structured_buffer_length_resource_name(var_name)
                    self.reserve_resource_binding_range(
                        used_resource_bindings,
                        "Metal",
                        "buffer",
                        length_binding,
                        resource_count,
                        length_name,
                    )
                    self.structured_buffer_length_variables.append(
                        (node, length_binding, vtype, array_size)
                    )
                    buffer_register = max(
                        buffer_register, length_binding + resource_count
                    )
                if self.structured_buffer_requires_counter(vtype):
                    counter_binding = self.next_available_resource_binding(
                        used_resource_bindings,
                        "buffer",
                        buffer_register,
                        resource_count,
                    )
                    counter_name = self.structured_buffer_counter_resource_name(
                        var_name
                    )
                    self.reserve_resource_binding_range(
                        used_resource_bindings,
                        "Metal",
                        "buffer",
                        counter_binding,
                        resource_count,
                        counter_name,
                    )
                    self.structured_buffer_counter_variables.append(
                        (node, counter_binding, vtype, array_size)
                    )
                    buffer_register = max(
                        buffer_register, counter_binding + resource_count
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
                mapped_type = self.map_resource_type_with_format(vtype, node)
                binding = self.explicit_resource_binding_index(
                    node, {"binding", "texture"}, ("t", "u")
                )
                if binding is None:
                    binding = self.next_available_resource_binding(
                        used_resource_bindings,
                        "texture",
                        texture_register,
                        resource_count,
                    )
                binding = self.reserve_or_remap_resource_binding(
                    used_resource_bindings,
                    source_resource_bindings,
                    "texture",
                    binding,
                    resource_count,
                    var_name,
                    node,
                    texture_register,
                )
                self.texture_variables.append((node, binding, mapped_type, array_size))
                self.texture_variable_types[node.name] = mapped_type
                self.texture_variable_raw_types[node.name] = self.resource_base_type(
                    vtype
                )
                record_explicit_image_metadata(
                    node.name,
                    node,
                    self.attribute_value_to_string,
                    image_formats=self.image_variable_formats,
                )
                texture_register = max(texture_register, binding + resource_count)
            elif self.resource_base_type(vtype) in {"sampler", "comparison_sampler"}:
                binding = self.explicit_resource_binding_index(
                    node, {"binding", "sampler"}, ("s",)
                )
                if binding is None:
                    binding = self.next_available_resource_binding(
                        used_resource_bindings,
                        "sampler",
                        sampler_register,
                        resource_count,
                    )
                binding = self.reserve_or_remap_resource_binding(
                    used_resource_bindings,
                    source_resource_bindings,
                    "sampler",
                    binding,
                    resource_count,
                    var_name,
                    node,
                    sampler_register,
                )
                self.sampler_variables.append((node, binding, array_size))
                sampler_register = max(sampler_register, binding + resource_count)
            elif self.is_input_attachment_type_name(vtype):
                self.refuse_lowering(
                    f"input attachment declaration: '{var_name}' uses "
                    f"{self.type_name_string(vtype)}; subpassInput resources require "
                    "Vulkan subpass lowering"
                )
            else:
                mapped_type = self.map_type(vtype)
                if (
                    var_name in self.metal_runtime_initialized_constants
                    and not array_suffix
                ):
                    qualifier = self.global_variable_qualifier(node)
                    self.record_metal_program_scope_value_global(
                        var_name, vtype, qualifier, node
                    )
                    init_expr = self.generate_expression_with_expected(
                        getattr(node, "initial_value", None),
                        getattr(node, "var_type", vtype),
                    )
                    # Metal program-scope constants need constant expressions;
                    # a source constant computed by a function call becomes an
                    # accessor that each use calls.
                    code += (
                        f"inline {mapped_type} {var_name}() {{\n"
                        f"    return {init_expr};\n"
                        "}\n"
                    )
                    continue
                declaration = format_c_style_array_declaration(mapped_type, var_name)
                if array_suffix:
                    declaration = f"{declaration}{array_suffix}"
                qualifier = self.global_variable_qualifier(node)
                declaration = f"{qualifier}{declaration}"
                resolved_var_type = getattr(node, "var_type", None) or vtype
                self.record_metal_program_scope_value_global(
                    var_name, resolved_var_type, qualifier, node
                )
                initial_value = getattr(node, "initial_value", None)
                if initial_value is not None:
                    expected_type = getattr(node, "var_type", None) or vtype
                    init_expr = self.generate_expression_with_expected(
                        initial_value, expected_type
                    )
                    code += f"{declaration} = {init_expr};\n"
                elif self.global_value_variable_requires_initializer(qualifier):
                    self.refuse_program_scope_groupshared_global(var_name, node)
                    init_expr = self.metal_program_scope_global_default_initializer(
                        mapped_type, array_suffix
                    )
                    code += f"{declaration} = {init_expr};\n"
                else:
                    code += f"{declaration};\n"

        self.function_global_resource_dependencies = (
            self.collect_function_global_resource_dependencies(
                [f for f in all_functions if not generic_function_parameters(f)]
                + list(generic_function_specializations.values())
            )
        )
        self.function_stage_parameter_dependencies = (
            self.collect_function_stage_parameter_dependencies(ast, target_stage)
        )
        self.function_stage_output_dependencies = (
            self.collect_function_stage_output_dependencies(ast, target_stage)
        )
        self.metal_device_coherent_textures = collect_device_coherent_textures(
            [entry[0] for entry in self.texture_variables],
            all_functions,
            is_storage_texture=self.is_metal_storage_texture_node,
            contracts_of=lambda node: self.resource_memory_qualifier_contracts(
                node, getattr(node, "param_type", getattr(node, "var_type", None))
            ),
            call_name=self.function_call_name,
            iter_nodes=self.iter_ast_nodes,
            refuse=self.refuse_lowering,
        )
        self.metal_inherited_buffer_contracts = collect_inherited_buffer_contracts(
            global_vars,
            all_functions,
            contracts_of=lambda node: self.resource_memory_qualifier_contracts(
                node, getattr(node, "param_type", getattr(node, "var_type", None))
            ),
            call_name=self.function_call_name,
            iter_nodes=self.iter_ast_nodes,
        )

        cbuffers = self.cbuffer_variables
        if cbuffers:
            code += "// Constant Buffers\n"
            code += self.generate_cbuffers(ast)
        code += self.metal_hlsl_structured_element_definitions()

        stage_entry_names = self.stage_entry_names(ast, target_stage)
        self.metal_compute_only_program = self.metal_program_is_compute_only(
            ast, target_stage
        )

        functions = order_functions_by_dependencies(
            getattr(ast, "functions", []),
            self.iter_ast_nodes,
            self.function_call_name,
            FunctionCallNode,
            include_overloads=True,
        )
        functions_code = ""
        for func in functions:
            qualifier_name = function_stage_name(func)

            if not should_emit_qualified_function(target_stage, qualifier_name):
                continue

            if generic_function_parameters(func):
                mark_generic_function_emitted(self, func)
                for specialized_func in generic_function_emission_list(self, func):
                    functions_code += self.generate_function(specialized_func)
                continue

            if qualifier_name == "vertex":
                functions_code += "// Vertex Shader\n"
                functions_code += self.generate_function(
                    func,
                    shader_type="vertex",
                    entry_name=stage_entry_names.get(id(func)),
                )
            elif qualifier_name == "fragment":
                functions_code += "// Fragment Shader\n"
                functions_code += self.generate_function(
                    func,
                    shader_type="fragment",
                    entry_name=stage_entry_names.get(id(func)),
                )
            elif qualifier_name == "compute":
                functions_code += "// Compute Shader\n"
                functions_code += self.generate_function(
                    func,
                    shader_type="compute",
                    entry_name=stage_entry_names.get(id(func)),
                    stage_local_variables=self.metal_stage_local_variables_with_lowered_program_groupshared(
                        func
                    ),
                )
            elif not folded_helper_is_unused(self, func):
                functions_code += self.generate_function(func)

        if hasattr(ast, "stages") and ast.stages:
            for stage_type, stage in ast.stages.items():
                stage_name = normalize_stage_name(stage_type)
                if not stage_matches(target_stage, stage_name):
                    continue

                for func in order_functions_by_dependencies(
                    getattr(stage, "local_functions", []) or [],
                    self.iter_ast_nodes,
                    self.function_call_name,
                    FunctionCallNode,
                ):
                    if generic_function_parameters(func):
                        mark_generic_function_emitted(self, func)
                        for specialized_func in generic_function_emission_list(
                            self,
                            func,
                        ):
                            functions_code += self.generate_function(specialized_func)
                    else:
                        functions_code += self.generate_function(func)

                if hasattr(stage, "entry_point"):
                    functions_code += f"// {stage_name.title()} Shader\n"
                    functions_code += self.generate_function(
                        stage.entry_point,
                        shader_type=stage_name,
                        execution_config=getattr(stage, "execution_config", None),
                        entry_name=stage_entry_names.get(id(stage.entry_point)),
                        stage_local_variables=self.metal_stage_local_variables_with_lowered_program_groupshared(
                            stage.entry_point,
                            getattr(stage, "local_variables", []),
                        ),
                        stage_node=stage,
                    )

        code += self.generate_image_atomic_compare_helpers()
        code += self.metal_texel_buffers.code()
        code += self.generate_buffer_atomic_compare_helpers()
        code += self.generate_pointer_atomic_compare_helpers()
        code += self.generate_metal_wave_helpers()
        code += emit_metal_wave_extension_helpers(self)
        code += emit_metal_exponent_helpers(self)
        code += emit_metal_bit_scan_helpers(self)
        code += emit_metal_texture_residency_helpers(self)
        code += emit_metal_extended_arithmetic_helpers(self)
        code += self.generate_metal_ray_query_helpers()
        code += self.generate_glsl_buffer_aggregate_load_helpers()
        code += self.generate_metal_inverse_helpers()
        code += self.metal_matrix_builtins.code()
        code += self.generate_metal_integer_dot_helper()
        code += self.metal_array_arguments.code()
        code += functions_code
        if self.metal_emulated_matrix_shapes:
            index = self.metal_prelude_insertion_index
            code = (
                code[:index]
                + metal_hlsl_matrix_prelude(self.metal_emulated_matrix_shapes)
                + code[index:]
            )
        return code

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
        previous_function_name = self.current_function_name
        previous_function_return_type = self.current_function_return_type
        previous_function_return_wrapper = self.current_function_return_wrapper
        previous_metal_value_array_locals = self.current_metal_value_array_locals
        previous_local_identifier_remaps = self.current_local_identifier_remaps
        previous_stage_output_return_active = getattr(
            self, "current_metal_stage_output_return_active", False
        )
        previous_local_variable_types = self.local_variable_types
        previous_address_space_variables = self.current_address_space_variables
        previous_native_width_local_array_declaration_ids = (
            self.current_metal_native_width_local_array_declaration_ids
        )
        previous_generic_function_substitutions = (
            self.current_generic_function_substitutions
        )
        previous_cbuffer_parameter_names = self.cbuffer_parameter_names
        previous_cbuffer_member_references = self.cbuffer_member_references
        previous_ambiguous_cbuffer_members = self.ambiguous_cbuffer_members
        previous_glsl_buffer_block_parameters = (
            self.current_glsl_buffer_block_parameters
        )
        previous_unsupported_glsl_buffer_block_parameters = (
            self.current_unsupported_glsl_buffer_block_parameters
        )
        previous_unsupported_glsl_buffer_block_local_variables = (
            self.current_unsupported_glsl_buffer_block_local_variables
        )
        previous_glsl_buffer_block_parameter_failures = (
            self.current_glsl_buffer_block_parameter_failures
        )
        previous_glsl_buffer_block_parameter_struct_failures = (
            self.current_glsl_buffer_block_parameter_struct_failures
        )
        previous_unsupported_metal_ray_function_table_array_variables = dict(
            self.unsupported_metal_ray_function_table_array_variables
        )
        previous_unsupported_metal_acceleration_structure_array_variables = dict(
            self.unsupported_metal_acceleration_structure_array_variables
        )
        previous_structured_buffer_length_parameters = (
            self.current_structured_buffer_length_parameters
        )
        previous_structured_buffer_counter_parameters = (
            self.current_structured_buffer_counter_parameters
        )
        previous_metal_mesh_output_config = self.current_metal_mesh_output_config
        previous_metal_mesh_output_parameter = self.current_metal_mesh_output_parameter
        previous_metal_mesh_grid_properties_parameter = (
            self.current_metal_mesh_grid_properties_parameter
        )
        previous_metal_mesh_payload_parameter = (
            self.current_metal_mesh_payload_parameter
        )
        previous_metal_mesh_payload_type = self.current_metal_mesh_payload_type
        previous_metal_mesh_output_accumulators = (
            self.current_metal_mesh_output_accumulators
        )
        previous_metal_non_thread_payload_parameters = (
            self.current_metal_non_thread_payload_parameters
        )
        previous_readonly_metal_mesh_payload_parameters = (
            self.current_readonly_metal_mesh_payload_parameters
        )
        previous_readonly_metal_mesh_payload_reasons = (
            self.current_readonly_metal_mesh_payload_reasons
        )
        previous_readonly_raw_buffer_parameters = (
            self.current_readonly_raw_buffer_parameters
        )
        previous_readonly_metal_parameters = self.current_readonly_metal_parameters
        previous_readonly_metal_parameter_reasons = (
            self.current_readonly_metal_parameter_reasons
        )
        previous_metal_wave_lane_index_parameter = (
            self.current_metal_wave_lane_index_parameter
        )
        previous_metal_wave_lane_count_parameter = (
            self.current_metal_wave_lane_count_parameter
        )
        previous_metal_graphics_builtin_parameter_names = (
            self.current_metal_graphics_builtin_parameter_names
        )
        previous_metal_compute_builtin_parameter_names = (
            self.current_metal_compute_builtin_parameter_names
        )
        self.current_function_name = getattr(func, "name", None)
        self.current_function_return_wrapper = None
        self.current_local_identifier_remaps = {}
        self.current_metal_stage_output_return_active = False
        self.current_generic_function_substitutions = (
            getattr(func, "_generic_substitutions", {}) or {}
        )
        (
            self.current_glsl_buffer_block_parameters,
            self.current_glsl_buffer_block_parameter_failures,
            self.current_glsl_buffer_block_parameter_struct_failures,
        ) = self.collect_lowered_glsl_buffer_block_parameters(param_list)
        for param in param_list or []:
            block = self.current_glsl_buffer_block_parameters.get(
                getattr(param, "name", None)
            )
            if block is not None:
                block["memory"] = self.metal_parameter_memory_qualifier_prefix(
                    param, getattr(param, "param_type", None), "device"
                )
        if shader_type in {"vertex", "fragment", "compute", "ray_generation"}:
            self.validate_stage_parameter_resource_bindings(
                param_list, self.current_function_name
            )
        if shader_type in {"vertex", "fragment"}:
            self.validate_graphics_builtin_parameter_types(param_list, shader_type)
            self.validate_metal_fragment_invocation_density(func, shader_type)
        if shader_type == "compute":
            self.validate_compute_builtin_parameter_types(param_list)
        if shader_type in self.METAL_DERIVATIVE_FREE_FUNCTION_KINDS:
            self.validate_metal_kernel_derivatives(func, shader_type)
        self.current_unsupported_glsl_buffer_block_parameters = (
            self.collect_unsupported_glsl_buffer_block_parameter_names(param_list)
        )
        self.current_unsupported_glsl_buffer_block_local_variables = set()
        unsupported_metal_ray_function_table_parameter_diagnostics = []
        self.current_structured_buffer_length_parameters = {}
        self.current_structured_buffer_counter_parameters = {}
        self.current_metal_mesh_payload_parameter = None
        self.current_metal_mesh_payload_type = None
        self.current_metal_mesh_output_accumulators = {}
        self.current_metal_non_thread_payload_parameters = set()
        self.current_readonly_metal_mesh_payload_parameters = set()
        self.current_readonly_metal_mesh_payload_reasons = {}
        self.current_readonly_raw_buffer_parameters = set()
        self.current_readonly_metal_parameters = set()
        self.current_readonly_metal_parameter_reasons = {}
        self.current_metal_wave_lane_index_parameter = None
        self.current_metal_wave_lane_count_parameter = None
        self.current_metal_graphics_builtin_parameter_names = {}
        self.current_metal_compute_builtin_parameter_names = {}
        self.local_variable_types = {}
        self.current_metal_value_array_locals = self.metal_value_array_returned_locals(
            func
        )
        self.current_address_space_variables = {}
        self.current_metal_native_width_local_array_declaration_ids = (
            self.metal_native_width_local_array_declaration_ids(func)
        )
        if shader_type in {"vertex", "fragment"}:
            for interface in self.spirv_stage_input_layouts(shader_type):
                name = getattr(interface, "variable_name", None)
                raw_type = getattr(interface, "data_type", None)
                if not name or not raw_type or "[" in str(name):
                    continue
                mapped_type = self.map_type(raw_type)
                if shader_type == "vertex":
                    vertex_stage_input_parameters.append(
                        {
                            "name": name,
                            "raw_type": raw_type,
                            "mapped_type": mapped_type,
                            "attribute": self.spirv_stage_input_attribute(interface),
                            "semantic": self.spirv_stage_input_semantic(interface),
                        }
                    )
                    self.local_variable_types[name] = self.type_name_string(raw_type)

            for interface in self.spirv_stage_output_layouts(shader_type):
                name = getattr(interface, "variable_name", None)
                raw_type = getattr(interface, "data_type", None)
                if not name or not raw_type or "[" in str(name):
                    continue
                stage_output_parameters.append(
                    {
                        "name": name,
                        "raw_type": raw_type,
                        "mapped_type": self.map_type(raw_type),
                        "semantic": self.spirv_stage_output_semantic(interface),
                        "attribute": self.spirv_stage_output_attribute(interface),
                    }
                )
                self.local_variable_types[name] = self.type_name_string(raw_type)
        skipped_parameter_indices = self.skipped_function_parameter_indices(
            self.current_function_name
        )
        for index, p in enumerate(param_list):
            if index in skipped_parameter_indices:
                continue
            if hasattr(p, "param_type"):
                raw_param_type = (
                    self.type_name_string(p.param_type)
                    if getattr(p.param_type, "generic_args", None)
                    else p.param_type
                )
            elif hasattr(p, "vtype"):
                raw_param_type = p.vtype
            else:
                raw_param_type = "float"
            self.local_variable_types[p.name] = self.type_name_string(raw_param_type)
            address_space = self.parameter_variable_address_space(
                raw_param_type, p, shader_type
            )
            if address_space is not None:
                self.current_address_space_variables[p.name] = address_space
            value_array_dimensions = None
            if shader_type is None:
                value_array_dimensions = self.metal_value_array_parameter_dimensions(
                    raw_param_type,
                    p,
                    self.map_resource_type_with_format(raw_param_type, p),
                )
                if (
                    value_array_dimensions is not None
                    and not self.metal_value_array_parameter_is_mutated(func, p.name)
                ):
                    value_array_dimensions = None
            if (
                value_array_dimensions is None
                and self.is_readonly_raw_buffer_parameter(
                    raw_param_type, p, shader_type
                )
            ):
                self.current_readonly_raw_buffer_parameters.add(p.name)
            readonly_parameter_reason = (
                None
                if value_array_dimensions is not None
                else self.readonly_metal_parameter_reason(
                    raw_param_type, p, shader_type
                )
            )
            if readonly_parameter_reason is not None:
                self.current_readonly_metal_parameters.add(p.name)
                self.current_readonly_metal_parameter_reasons[p.name] = (
                    readonly_parameter_reason
                )

            if self.is_metal_mesh_output_parameter(shader_type, p):
                continue

            table_array_kind = self.metal_ray_function_table_array_parameter_kind(
                raw_param_type, p
            )
            if table_array_kind is not None:
                self.unsupported_metal_ray_function_table_array_variables[p.name] = (
                    table_array_kind
                )
                unsupported_metal_ray_function_table_parameter_diagnostics.append(
                    self.unsupported_metal_ray_function_table_array_diagnostic(
                        table_array_kind, p.name
                    ).rstrip()
                )
                continue

            acceleration_structure_array = (
                self.metal_acceleration_structure_array_parameter_kind(
                    raw_param_type, p
                )
            )
            if acceleration_structure_array is not None:
                self.unsupported_metal_acceleration_structure_array_variables[
                    p.name
                ] = acceleration_structure_array
                unsupported_metal_ray_function_table_parameter_diagnostics.append(
                    self.unsupported_metal_acceleration_structure_array_diagnostic(
                        p.name
                    ).rstrip()
                )
                continue

            if self.is_sampler_type(raw_param_type):
                sampler_parameters.add(p.name)
                resource_array = self.resource_array_parameter(raw_param_type, p)
                if resource_array is not None:
                    _, array_size = resource_array
                    sampler_parameter_array_sizes[p.name] = array_size
            elif self.is_texture_or_image_resource_type(raw_param_type):
                texture_parameters[p.name] = self.map_resource_type_with_format(
                    self.resource_base_type(raw_param_type), p
                )
                texture_parameter_raw_types[p.name] = self.resource_base_type(
                    raw_param_type
                )
                resource_array = self.resource_array_parameter(raw_param_type, p)
                if resource_array is not None:
                    _, array_size = resource_array
                    texture_parameter_array_sizes[p.name] = array_size
                record_explicit_image_metadata(
                    p.name,
                    p,
                    self.attribute_value_to_string,
                    image_formats=image_format_parameters,
                )
            param_type = self.map_resource_type_with_format(raw_param_type, p)

            semantic = self.semantic_from_node(p)
            if self.is_graphics_stage_output_parameter(p, shader_type):
                stage_output_parameters.append(
                    {
                        "name": p.name,
                        "raw_type": raw_param_type,
                        "mapped_type": param_type,
                        "semantic": semantic,
                        "attribute": self.metal_stage_output_parameter_attribute(
                            p, shader_type
                        ),
                    }
                )
                continue
            if self.is_metal_mesh_payload_parameter(shader_type, p):
                self.current_metal_mesh_payload_parameter = p.name
                self.current_metal_mesh_payload_type = self.map_type(raw_param_type)
                self.current_address_space_variables[p.name] = "object_data"
                readonly_reason = self.readonly_metal_mesh_payload_parameter_reason(
                    shader_type, p
                )
                if readonly_reason is not None:
                    self.current_readonly_metal_mesh_payload_parameters.add(p.name)
                    self.current_readonly_metal_mesh_payload_reasons[p.name] = (
                        readonly_reason
                    )
            ray_payload_declaration = self.metal_ray_payload_parameter_declaration(
                param_type, p.name, p, shader_type
            )
            if ray_payload_declaration is not None:
                self.current_metal_non_thread_payload_parameters.add(p.name)
                ray_payload_address_space = self.metal_ray_payload_address_space(
                    shader_type
                )
                if ray_payload_address_space is not None:
                    self.current_address_space_variables[p.name] = (
                        ray_payload_address_space
                    )

            builtin_lowering = self.metal_graphics_builtin_parameter_lowering(
                raw_param_type,
                param_type,
                p,
                shader_type,
                reserved_parameter_names
                | set(params)
                | self.metal_function_local_variable_names(func),
            )
            if builtin_lowering is not None:
                params.append(
                    f"{builtin_lowering['type']} {builtin_lowering['name']} "
                    f"[[{builtin_lowering['attribute']}]]"
                )
                reserved_parameter_names.add(builtin_lowering["name"])
                stage_builtin_parameter_alias_declarations.append(
                    f"{param_type} {p.name} = {param_type}({builtin_lowering['name']});"
                )
                continue

            if self.is_plain_metal_vertex_stage_input_parameter(
                raw_param_type, param_type, p, shader_type
            ):
                vertex_stage_input_parameters.append(
                    {
                        "name": p.name,
                        "raw_type": raw_param_type,
                        "mapped_type": param_type,
                        "attribute": self.metal_vertex_input_location_attribute(p),
                        "semantic": semantic,
                    }
                )
                continue
            if self.is_plain_metal_fragment_stage_input_parameter(
                raw_param_type, param_type, p, shader_type
            ):
                fragment_stage_input_parameters.append(
                    {
                        "name": p.name,
                        "raw_type": raw_param_type,
                        "mapped_type": param_type,
                        "attribute": metal_varying_attribute(self, p, semantic),
                        "semantic": semantic,
                        "preserve_user_attribute": True,
                    }
                )
                continue

            param_attr = (
                self.metal_vertex_input_location_attribute(p)
                if shader_type == "vertex"
                else None
            )
            if param_attr is None and self.metal_vertex_input_takes_next_attribute(
                raw_param_type, shader_type, p, semantic
            ):
                # The [[stage_in]] structure gives a user-semantic input the next free [[attribute(n)]].
                param_attr = ""
            if param_attr is None:
                param_attr = self.parameter_attribute(
                    raw_param_type, semantic, shader_type, p
                )
            param_attr += self.metal_conditional_resource_attribute(p)
            if value_array_dimensions is not None:
                input_name = self.metal_value_array_input_name(
                    p.name,
                    reserved_parameter_names
                    | self.metal_function_local_variable_names(func),
                )
                reserved_parameter_names.add(input_name)
                declaration = format_c_style_array_declaration(param_type, input_name)
                value_array_parameter_copy_statements.extend(
                    self.metal_value_array_copy_statements(
                        param_type, p.name, input_name, value_array_dimensions
                    )
                )
            else:
                declaration = self.format_parameter_declaration(
                    raw_param_type, param_type, p.name, p, shader_type
                )
            if any(
                str(getattr(attribute, "name", "")).lower() == "maybe_unused"
                for attribute in getattr(p, "attributes", []) or []
            ):
                declaration = self.format_unused_metal_declaration(declaration)
            if self.should_wrap_metal_vertex_stage_input_parameter(
                raw_param_type, shader_type, p
            ):
                vertex_stage_input_parameters.append(
                    {
                        "name": p.name,
                        "declaration": declaration,
                        "attribute": param_attr,
                        "semantic": semantic,
                    }
                )
                continue
            params.append(f"{declaration}{param_attr}")
            if self.is_structured_buffer_type(
                raw_param_type
            ) and self.structured_buffer_parameter_requires_length(
                self.current_function_name, p.name
            ):
                length_name = self.structured_buffer_length_parameter_name(p.name)
                self.current_structured_buffer_length_parameters[p.name] = length_name
                array_size = self.structured_buffer_parameter_array_size(
                    raw_param_type, p
                )
                params.append(
                    self.format_structured_buffer_length_parameter(p.name, array_size)
                )
            if self.structured_buffer_requires_counter(raw_param_type):
                counter_name = self.structured_buffer_counter_parameter_name(p.name)
                self.current_structured_buffer_counter_parameters[p.name] = counter_name
                array_size = self.structured_buffer_parameter_array_size(
                    raw_param_type, p
                )
                params.append(
                    self.format_structured_buffer_counter_parameter(p.name, array_size)
                )

        if shader_type == "vertex":
            explicit_stage_builtins = self.explicit_graphics_stage_builtin_parameters(
                param_list, shader_type
            )
            if explicit_stage_builtins.get("vertex_id"):
                self.current_metal_graphics_builtin_parameter_names[
                    "gl_VertexIndex"
                ] = explicit_stage_builtins["vertex_id"]
            if explicit_stage_builtins.get("instance_id"):
                self.current_metal_graphics_builtin_parameter_names[
                    "gl_InstanceIndex"
                ] = explicit_stage_builtins["instance_id"]
            reserved_builtin_names = set(reserved_parameter_names)
            reserved_builtin_names.update(
                self.metal_function_local_variable_names(func)
            )
            for (
                builtin_name,
                name,
                param_type,
                attribute,
            ) in self.required_metal_vertex_builtin_parameters(
                func, reserved_builtin_names, explicit_stage_builtins
            ):
                params.append(f"{param_type} {name} [[{attribute}]]")
                reserved_parameter_names.add(name)
                self.current_metal_graphics_builtin_parameter_names[builtin_name] = name

        if shader_type == "fragment":
            explicit_stage_builtins = self.explicit_graphics_stage_builtin_parameters(
                param_list, shader_type
            )
            if explicit_stage_builtins.get("position"):
                self.current_metal_graphics_builtin_parameter_names["gl_FragCoord"] = (
                    explicit_stage_builtins["position"]
                )
            reserved_builtin_names = set(reserved_parameter_names)
            reserved_builtin_names.update(
                self.metal_function_local_variable_names(func)
            )
            for (
                builtin_name,
                name,
                param_type,
                attribute,
            ) in self.required_metal_fragment_builtin_parameters(
                func, reserved_builtin_names, explicit_stage_builtins
            ):
                params.append(f"{param_type} {name} [[{attribute}]]")
                reserved_parameter_names.add(name)
                self.current_metal_graphics_builtin_parameter_names[builtin_name] = name

        if shader_type in {"compute", "ray_generation"}:
            existing_param_names = {getattr(p, "name", None) for p in param_list}
            explicit_stage_builtins = self.explicit_compute_stage_builtin_parameters(
                param_list
            )
            self.current_metal_compute_builtin_parameter_names.update(
                {
                    builtin_name: explicit_stage_builtins[attribute]
                    for builtin_name, _param_type, attribute in (
                        self.compute_builtin_parameter_specs()
                    )
                    if attribute in explicit_stage_builtins
                    and explicit_stage_builtins[attribute]
                }
            )
            self.current_metal_wave_lane_index_parameter = explicit_stage_builtins.get(
                "thread_index_in_simdgroup"
            )
            self.current_metal_wave_lane_count_parameter = explicit_stage_builtins.get(
                "threads_per_simdgroup"
            )
            reserved_builtin_names = set(existing_param_names)
            reserved_builtin_names.update(
                self.metal_function_local_variable_names(func)
            )
            for name, param_type, attribute in self.required_compute_builtin_parameters(
                func, reserved_builtin_names, explicit_stage_builtins
            ):
                if name not in existing_param_names:
                    params.append(f"{param_type} {name} [[{attribute}]]")
                    reserved_parameter_names.add(name)
                builtin_name = self.compute_builtin_name_for_metal_attribute(attribute)
                if builtin_name is not None:
                    self.current_metal_compute_builtin_parameter_names[builtin_name] = (
                        name
                    )
                if attribute == "thread_index_in_simdgroup":
                    self.current_metal_wave_lane_index_parameter = name
                elif attribute == "threads_per_simdgroup":
                    self.current_metal_wave_lane_count_parameter = name

        reserved_parameter_names.update(self.global_resource_parameter_names())
        self.cbuffer_parameter_names = self.collect_cbuffer_parameter_names(
            self.cbuffer_variables, reserved_names=reserved_parameter_names
        )
        self.cbuffer_member_references = self.collect_cbuffer_member_references(
            self.cbuffer_variables
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

        params_str = ", ".join(params)
        if shader_type is None:
            params_str = self.append_required_stage_output_parameter(
                params_str, self.current_function_name
            )
            params_str = self.append_required_stage_parameter_parameters(
                params_str, self.current_function_name
            )
            params_str = self.append_required_cbuffer_parameters(
                params_str, self.current_function_name
            )
            params_str = self.append_required_thread_private_global_parameters(
                params_str, self.current_function_name
            )
            params_str = self.append_required_global_resource_parameters(
                params_str, self.current_function_name
            )
            params_str = self.append_metal_mesh_dispatch_context_parameters(
                params_str,
                func,
                reserved_parameter_names,
            )
            params_str = self.append_required_metal_wave_lane_parameters(
                params_str, self.current_function_name
            )
            for parameter in self.required_function_stage_parameters(
                self.current_function_name
            ):
                self.local_variable_types[parameter.name] = self.type_name_string(
                    self.parameter_raw_type(parameter)
                )
            stage_output_type = self.required_function_stage_output_type(
                self.current_function_name
            )
            if stage_output_type is not None:
                self.local_variable_types["output"] = self.type_name_string(
                    stage_output_type
                )

        if hasattr(func, "return_type"):
            effective_return_type = self.metal_effective_function_return_type(func)
            raw_return_type = self.type_name_string(effective_return_type)
            if "[" in raw_return_type:
                # Metal cannot return C arrays; ``array<T, N>`` keeps the
                # value semantics of a source ``metal::array`` result.
                return_type = self.metal_value_array_type(raw_return_type)
                if return_type is None:
                    raise MetalRefusal(
                        "Metal output does not support unsized C-style array "
                        "return types; wrap the array in a struct or use an "
                        "output parameter"
                    )
            elif isinstance(effective_return_type, PointerType):
                return_type = self.format_metal_pointer_return_type(
                    effective_return_type
                )
            else:
                return_type = self.map_type(raw_return_type)
        else:
            effective_return_type = "void"
            raw_return_type = "void"
            return_type = "void"
        if shader_type in {"vertex", "fragment"}:
            self.validate_function_return_semantic_type(
                func, raw_return_type, shader_type
            )
            if (
                return_type != "void"
                and str(raw_return_type) not in self.structs_by_name
            ):
                return_semantic = self.function_return_semantic(func)
                refuse_unreturnable_metal_entry_semantic(
                    self,
                    func,
                    shader_type,
                    return_semantic,
                    metal_entry_return_semantic_key(
                        self, func, return_semantic, shader_type
                    ),
                    return_type,
                    in_structure=bool(stage_output_parameters),
                )
            if stage_output_parameters and return_type != "void":
                stage_output_parameters.append(
                    self.metal_stage_return_value_output(
                        func,
                        shader_type,
                        raw_return_type,
                        return_type,
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
            self.current_function_name = previous_function_name
            self.current_function_return_type = previous_function_return_type
            self.current_function_return_wrapper = previous_function_return_wrapper
            self.current_metal_value_array_locals = previous_metal_value_array_locals
            self.current_local_identifier_remaps = previous_local_identifier_remaps
            self.local_variable_types = previous_local_variable_types
            self.current_address_space_variables = previous_address_space_variables
            self.current_generic_function_substitutions = (
                previous_generic_function_substitutions
            )
            self.cbuffer_parameter_names = previous_cbuffer_parameter_names
            self.cbuffer_member_references = previous_cbuffer_member_references
            self.ambiguous_cbuffer_members = previous_ambiguous_cbuffer_members
            self.current_glsl_buffer_block_parameters = (
                previous_glsl_buffer_block_parameters
            )
            self.current_unsupported_glsl_buffer_block_parameters = (
                previous_unsupported_glsl_buffer_block_parameters
            )
            self.current_unsupported_glsl_buffer_block_local_variables = (
                previous_unsupported_glsl_buffer_block_local_variables
            )
            self.unsupported_metal_ray_function_table_array_variables = (
                previous_unsupported_metal_ray_function_table_array_variables
            )
            self.unsupported_metal_acceleration_structure_array_variables = (
                previous_unsupported_metal_acceleration_structure_array_variables
            )
            self.current_glsl_buffer_block_parameter_failures = (
                previous_glsl_buffer_block_parameter_failures
            )
            self.current_glsl_buffer_block_parameter_struct_failures = (
                previous_glsl_buffer_block_parameter_struct_failures
            )
            self.current_structured_buffer_length_parameters = (
                previous_structured_buffer_length_parameters
            )
            self.current_structured_buffer_counter_parameters = (
                previous_structured_buffer_counter_parameters
            )
            self.current_metal_mesh_output_config = previous_metal_mesh_output_config
            self.current_metal_mesh_output_parameter = (
                previous_metal_mesh_output_parameter
            )
            self.current_metal_mesh_grid_properties_parameter = (
                previous_metal_mesh_grid_properties_parameter
            )
            self.current_metal_mesh_payload_parameter = (
                previous_metal_mesh_payload_parameter
            )
            self.current_metal_mesh_payload_type = previous_metal_mesh_payload_type
            self.current_metal_mesh_output_accumulators = (
                previous_metal_mesh_output_accumulators
            )
            self.current_metal_non_thread_payload_parameters = (
                previous_metal_non_thread_payload_parameters
            )
            self.current_readonly_metal_mesh_payload_parameters = (
                previous_readonly_metal_mesh_payload_parameters
            )
            self.current_readonly_metal_mesh_payload_reasons = (
                previous_readonly_metal_mesh_payload_reasons
            )
            self.current_readonly_raw_buffer_parameters = (
                previous_readonly_raw_buffer_parameters
            )
            self.current_readonly_metal_parameters = previous_readonly_metal_parameters
            self.current_readonly_metal_parameter_reasons = (
                previous_readonly_metal_parameter_reasons
            )
            self.current_metal_wave_lane_index_parameter = (
                previous_metal_wave_lane_index_parameter
            )
            self.current_metal_wave_lane_count_parameter = (
                previous_metal_wave_lane_count_parameter
            )
            self.current_metal_graphics_builtin_parameter_names = (
                previous_metal_graphics_builtin_parameter_names
            )
            self.current_metal_compute_builtin_parameter_names = (
                previous_metal_compute_builtin_parameter_names
            )
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
            self.current_function_name = previous_function_name
            self.current_function_return_type = previous_function_return_type
            self.current_function_return_wrapper = previous_function_return_wrapper
            self.current_metal_value_array_locals = previous_metal_value_array_locals
            self.current_local_identifier_remaps = previous_local_identifier_remaps
            self.local_variable_types = previous_local_variable_types
            self.current_address_space_variables = previous_address_space_variables
            self.current_generic_function_substitutions = (
                previous_generic_function_substitutions
            )
            self.cbuffer_parameter_names = previous_cbuffer_parameter_names
            self.cbuffer_member_references = previous_cbuffer_member_references
            self.ambiguous_cbuffer_members = previous_ambiguous_cbuffer_members
            self.current_glsl_buffer_block_parameters = (
                previous_glsl_buffer_block_parameters
            )
            self.current_unsupported_glsl_buffer_block_parameters = (
                previous_unsupported_glsl_buffer_block_parameters
            )
            self.current_unsupported_glsl_buffer_block_local_variables = (
                previous_unsupported_glsl_buffer_block_local_variables
            )
            self.unsupported_metal_ray_function_table_array_variables = (
                previous_unsupported_metal_ray_function_table_array_variables
            )
            self.unsupported_metal_acceleration_structure_array_variables = (
                previous_unsupported_metal_acceleration_structure_array_variables
            )
            self.current_glsl_buffer_block_parameter_failures = (
                previous_glsl_buffer_block_parameter_failures
            )
            self.current_glsl_buffer_block_parameter_struct_failures = (
                previous_glsl_buffer_block_parameter_struct_failures
            )
            self.current_structured_buffer_length_parameters = (
                previous_structured_buffer_length_parameters
            )
            self.current_structured_buffer_counter_parameters = (
                previous_structured_buffer_counter_parameters
            )
            self.current_metal_mesh_output_config = previous_metal_mesh_output_config
            self.current_metal_mesh_output_parameter = (
                previous_metal_mesh_output_parameter
            )
            self.current_metal_mesh_grid_properties_parameter = (
                previous_metal_mesh_grid_properties_parameter
            )
            self.current_metal_mesh_payload_parameter = (
                previous_metal_mesh_payload_parameter
            )
            self.current_metal_mesh_payload_type = previous_metal_mesh_payload_type
            self.current_metal_mesh_output_accumulators = (
                previous_metal_mesh_output_accumulators
            )
            self.current_metal_non_thread_payload_parameters = (
                previous_metal_non_thread_payload_parameters
            )
            self.current_readonly_metal_mesh_payload_parameters = (
                previous_readonly_metal_mesh_payload_parameters
            )
            self.current_readonly_metal_mesh_payload_reasons = (
                previous_readonly_metal_mesh_payload_reasons
            )
            self.current_readonly_raw_buffer_parameters = (
                previous_readonly_raw_buffer_parameters
            )
            self.current_readonly_metal_parameters = previous_readonly_metal_parameters
            self.current_readonly_metal_parameter_reasons = (
                previous_readonly_metal_parameter_reasons
            )
            self.current_metal_wave_lane_index_parameter = (
                previous_metal_wave_lane_index_parameter
            )
            self.current_metal_wave_lane_count_parameter = (
                previous_metal_wave_lane_count_parameter
            )
            self.current_metal_graphics_builtin_parameter_names = (
                previous_metal_graphics_builtin_parameter_names
            )
            self.current_metal_compute_builtin_parameter_names = (
                previous_metal_compute_builtin_parameter_names
            )
            return code

        if shader_type == "vertex":
            function_name = entry_name or f"vertex_{func.name}"
            if vertex_stage_input_parameters:
                stage_input_struct_name = self.unique_vertex_stage_input_struct_name(
                    function_name
                )
                stage_input_parameter_name = self.unique_metal_generated_name(
                    "_crossglInput",
                    reserved_parameter_names
                    | self.metal_function_local_variable_names(func),
                )
                self.local_variable_types[stage_input_parameter_name] = (
                    stage_input_struct_name
                )
                code += self.generate_metal_vertex_stage_input_parameter_struct(
                    stage_input_struct_name, vertex_stage_input_parameters
                )
                params_str = self.append_parameter_declaration(
                    params_str,
                    f"{stage_input_struct_name} {stage_input_parameter_name} "
                    "[[stage_in]]",
                )
                vertex_stage_input_alias_declarations = (
                    self.generate_metal_vertex_stage_input_alias_declarations(
                        stage_input_parameter_name, vertex_stage_input_parameters
                    )
                )
            params_str = self.append_global_resource_parameters(
                params_str,
                self.current_function_name,
                func,
                filter_writable_textures=True,
            )
            if stage_output_parameters:
                stage_output_struct_name = self.unique_return_wrapper_struct_name(
                    function_name
                )
                code += self.generate_metal_stage_output_parameter_struct(
                    stage_output_struct_name, stage_output_parameters
                )
                return_type = stage_output_struct_name
                self.current_function_return_wrapper = None
            else:
                return_wrapper = self.function_return_semantic_wrapper(
                    func, raw_return_type, return_type, shader_type, function_name
                )
                if return_wrapper is not None:
                    code += self.generate_return_wrapper_struct(return_wrapper)
                    return_type = return_wrapper["struct_name"]
                self.current_function_return_wrapper = return_wrapper
            code += f"vertex {return_type} {function_name}({params_str}) {{\n"
        elif shader_type == "fragment":
            function_name = entry_name or f"fragment_{func.name}"
            if fragment_stage_input_parameters:
                stage_input_struct_name = self.unique_vertex_stage_input_struct_name(
                    function_name
                )
                stage_input_parameter_name = self.unique_metal_generated_name(
                    "_crossglInput",
                    reserved_parameter_names
                    | self.metal_function_local_variable_names(func),
                )
                self.local_variable_types[stage_input_parameter_name] = (
                    stage_input_struct_name
                )
                code += self.generate_metal_vertex_stage_input_parameter_struct(
                    stage_input_struct_name, fragment_stage_input_parameters
                )
                params_str = self.append_parameter_declaration(
                    params_str,
                    f"{stage_input_struct_name} {stage_input_parameter_name} "
                    "[[stage_in]]",
                )
                fragment_stage_input_alias_declarations = (
                    self.generate_metal_vertex_stage_input_alias_declarations(
                        stage_input_parameter_name, fragment_stage_input_parameters
                    )
                )
            params_str = self.append_global_resource_parameters(
                params_str, self.current_function_name, func
            )
            if stage_output_parameters:
                stage_output_struct_name = self.unique_return_wrapper_struct_name(
                    function_name
                )
                code += self.generate_metal_stage_output_parameter_struct(
                    stage_output_struct_name, stage_output_parameters
                )
                return_type = stage_output_struct_name
                self.current_function_return_wrapper = None
            else:
                return_wrapper = self.function_return_semantic_wrapper(
                    func, raw_return_type, return_type, shader_type, function_name
                )
                if return_wrapper is not None:
                    code += self.generate_return_wrapper_struct(return_wrapper)
                    return_type = return_wrapper["struct_name"]
                self.current_function_return_wrapper = return_wrapper
            # MSL's [[early_fragment_tests]] precedes the fragment function (MSL 4.1 section 5.1.2).
            early_tests = (
                "[[early_fragment_tests]] "
                if requests_early_fragment_tests(func, stage_node)
                else ""
            )
            code += f"{early_tests}fragment {return_type} {function_name}({params_str}) {{\n"
        elif shader_type in ["compute", "ray_generation"]:
            params_str = self.append_global_resource_parameters(
                params_str, self.current_function_name, func
            )
            function_name = entry_name or self.stage_entry_base_name(shader_type, func)
            self.validate_metal_kernel_return_type(func, shader_type, raw_return_type)
            code += f"kernel {return_type} {function_name}({params_str}) {{\n"
        elif shader_type in ["mesh", "object", "task", "amplification"]:
            stage_keyword = "mesh" if shader_type == "mesh" else "object"
            function_name = entry_name or f"{stage_keyword}_{func.name}"
            params_str = self.append_global_resource_parameters(
                params_str, self.current_function_name, func
            )
            stage_attribute = self.metal_mesh_stage_attribute(
                shader_type, func, execution_config
            )
            generated_mesh_payload_parameter = (
                self.generated_metal_mesh_payload_parameter(
                    shader_type, func, reserved_parameter_names
                )
            )
            if generated_mesh_payload_parameter is not None:
                payload_type, payload_name = generated_mesh_payload_parameter
                params_str = self.append_parameter_declaration(
                    params_str,
                    f"object_data {payload_type}& {payload_name} [[payload]]",
                )
                self.current_metal_mesh_payload_parameter = payload_name
                self.current_metal_mesh_payload_type = payload_type
                self.current_address_space_variables[payload_name] = "object_data"
            mesh_grid_properties_parameter = self.metal_mesh_grid_properties_parameter(
                shader_type, func, reserved_parameter_names
            )
            if mesh_grid_properties_parameter is not None:
                params_str = self.append_parameter_declaration(
                    params_str,
                    f"mesh_grid_properties {mesh_grid_properties_parameter}",
                )
                self.current_metal_mesh_grid_properties_parameter = (
                    mesh_grid_properties_parameter
                )
            mesh_output = self.metal_mesh_stage_output_config(
                shader_type, func, function_name, reserved_parameter_names
            )
            if mesh_output is not None:
                if mesh_output.get("generated_vertex_struct", False):
                    code += self.generate_metal_mesh_vertex_output_struct(mesh_output)
                params_str = self.append_parameter_declaration(
                    params_str,
                    self.metal_mesh_stage_output_parameter_declaration(mesh_output),
                )
                self.current_metal_mesh_output_parameter = mesh_output["parameter_name"]
                self.current_metal_mesh_output_config = mesh_output
                self.validate_metal_mesh_output_usage(func)
            code += (
                f"{stage_attribute} {return_type} {function_name}({params_str}) {{\n"
            )
        elif shader_type in [
            "ray_any_hit",
            "ray_closest_hit",
            "ray_miss",
            "ray_callable",
            "anyhit",
            "closesthit",
            "miss",
            "callable",
        ]:
            rt_stage_map = {
                "ray_any_hit": "anyhit",
                "ray_closest_hit": "closesthit",
                "ray_miss": "miss",
                "ray_callable": "callable",
                "anyhit": "anyhit",
                "closesthit": "closesthit",
                "miss": "miss",
                "callable": "callable",
            }
            stage_keyword = rt_stage_map[shader_type]
            function_name = entry_name or f"{stage_keyword}_{func.name}"
            code += f"[[visible]] {return_type} {function_name}({params_str}) {{\n"
        elif shader_type in ["ray_intersection", "intersection"]:
            function_name = entry_name or f"intersection_{func.name}"
            stage_attribute = self.metal_ray_intersection_stage_attribute(
                func, raw_return_type
            )
            code += (
                f"{stage_attribute} {return_type} "
                f"{function_name}({params_str}) {{\n"
            )
        else:
            semantic = self.semantic_from_node(func)
            function_name = entry_name or func.name
            semantic_attr = self.map_non_stage_function_semantic(semantic)
            # Lowered helpers can become unused after native operation selection.
            if (
                getattr(func, "linkage", None) == "internal"
                and getattr(func, "is_inline", False)
            ) or re.fullmatch(
                r"CrossGLMetalVectorIndex_[iu](?:8|16)vec[234]_set", function_name
            ):
                code += "__attribute__((unused))\n"
            code += (
                f"{self.metal_function_linkage_prefix(func)}"
                f"{return_type} {function_name}({params_str}){semantic_attr} {{\n"
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

        previous_sampler_parameters = self.current_sampler_parameters
        previous_sampler_parameter_array_sizes = (
            self.current_sampler_parameter_array_sizes
        )
        previous_texture_parameters = self.current_texture_parameters
        previous_texture_parameter_raw_types = self.current_texture_parameter_raw_types
        previous_texture_parameter_array_sizes = (
            self.current_texture_parameter_array_sizes
        )
        previous_texture_alias_sources = self.current_texture_alias_sources
        previous_image_format_parameters = self.current_image_format_parameters
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
        self.current_sampler_parameters = previous_sampler_parameters
        self.current_sampler_parameter_array_sizes = (
            previous_sampler_parameter_array_sizes
        )
        self.current_texture_parameters = previous_texture_parameters
        self.current_texture_parameter_raw_types = previous_texture_parameter_raw_types
        self.current_texture_parameter_array_sizes = (
            previous_texture_parameter_array_sizes
        )
        self.current_texture_alias_sources = previous_texture_alias_sources
        self.current_image_format_parameters = previous_image_format_parameters
        self.current_structured_buffer_length_parameters = (
            previous_structured_buffer_length_parameters
        )
        self.current_structured_buffer_counter_parameters = (
            previous_structured_buffer_counter_parameters
        )
        self.current_metal_mesh_output_config = previous_metal_mesh_output_config
        self.current_metal_mesh_output_parameter = previous_metal_mesh_output_parameter
        self.current_metal_mesh_grid_properties_parameter = (
            previous_metal_mesh_grid_properties_parameter
        )
        self.current_metal_mesh_payload_parameter = (
            previous_metal_mesh_payload_parameter
        )
        self.current_metal_mesh_payload_type = previous_metal_mesh_payload_type
        self.current_metal_mesh_output_accumulators = (
            previous_metal_mesh_output_accumulators
        )
        self.current_metal_non_thread_payload_parameters = (
            previous_metal_non_thread_payload_parameters
        )
        self.current_readonly_metal_mesh_payload_parameters = (
            previous_readonly_metal_mesh_payload_parameters
        )
        self.current_readonly_metal_mesh_payload_reasons = (
            previous_readonly_metal_mesh_payload_reasons
        )
        self.current_readonly_raw_buffer_parameters = (
            previous_readonly_raw_buffer_parameters
        )
        self.current_readonly_metal_parameters = previous_readonly_metal_parameters
        self.current_readonly_metal_parameter_reasons = (
            previous_readonly_metal_parameter_reasons
        )
        self.current_metal_wave_lane_index_parameter = (
            previous_metal_wave_lane_index_parameter
        )
        self.current_metal_wave_lane_count_parameter = (
            previous_metal_wave_lane_count_parameter
        )
        self.current_metal_graphics_builtin_parameter_names = (
            previous_metal_graphics_builtin_parameter_names
        )
        self.current_function_name = previous_function_name
        self.current_function_return_type = previous_function_return_type
        self.current_function_return_wrapper = previous_function_return_wrapper
        self.current_metal_value_array_locals = previous_metal_value_array_locals
        self.current_local_identifier_remaps = previous_local_identifier_remaps
        self.current_metal_stage_output_return_active = (
            previous_stage_output_return_active
        )
        self.local_variable_types = previous_local_variable_types
        self.current_address_space_variables = previous_address_space_variables
        self.current_metal_native_width_local_array_declaration_ids = (
            previous_native_width_local_array_declaration_ids
        )
        self.current_generic_function_substitutions = (
            previous_generic_function_substitutions
        )
        self.cbuffer_parameter_names = previous_cbuffer_parameter_names
        self.cbuffer_member_references = previous_cbuffer_member_references
        self.ambiguous_cbuffer_members = previous_ambiguous_cbuffer_members
        self.current_glsl_buffer_block_parameters = (
            previous_glsl_buffer_block_parameters
        )
        self.current_unsupported_glsl_buffer_block_parameters = (
            previous_unsupported_glsl_buffer_block_parameters
        )
        self.current_unsupported_glsl_buffer_block_local_variables = (
            previous_unsupported_glsl_buffer_block_local_variables
        )
        self.unsupported_metal_ray_function_table_array_variables = (
            previous_unsupported_metal_ray_function_table_array_variables
        )
        self.unsupported_metal_acceleration_structure_array_variables = (
            previous_unsupported_metal_acceleration_structure_array_variables
        )
        self.current_glsl_buffer_block_parameter_failures = (
            previous_glsl_buffer_block_parameter_failures
        )
        self.current_glsl_buffer_block_parameter_struct_failures = (
            previous_glsl_buffer_block_parameter_struct_failures
        )
        self.current_metal_compute_builtin_parameter_names = (
            previous_metal_compute_builtin_parameter_names
        )

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
        if isinstance(expr, BinaryOpNode):
            left_type = self.expression_result_type(expr.left)
            right_type = self.expression_result_type(expr.right)
            operator = self.map_operator(getattr(expr, "op", ""))
            emulated_result = self.metal_emulated_matrix_binary_result_type(
                operator, left_type, right_type
            )
            if emulated_result is not None:
                return emulated_result
            pointer_operand = self.pointer_arithmetic_operand(
                expr, (left_type, right_type)
            )
            if pointer_operand is not None:
                return left_type if pointer_operand is expr.left else right_type
            if operator in {"<", ">", "<=", ">=", "==", "!=", "&&", "||"}:
                for candidate_type in (left_type, right_type):
                    mapped_type = self.map_type(candidate_type)
                    component_type = self.vector_component_type(mapped_type)
                    if component_type is None:
                        continue
                    width = mapped_type[-1] if mapped_type[-1:].isdigit() else ""
                    if width:
                        return f"bool{width}"
                return "bool"
            matrix_result = self.metal_matrix_binary_result_type(
                operator, left_type, right_type
            )
            if matrix_result is not None:
                return matrix_result
            if (
                self.is_vector_value_type(left_type)
                and self.is_vector_value_type(right_type)
                and operator not in {"<<", ">>"}
            ):
                left_mapped = self.map_type(left_type)
                right_mapped = self.map_type(right_type)
                left_component = self.vector_component_type(left_mapped)
                right_component = self.vector_component_type(right_mapped)
                if (
                    left_component != right_component
                    and self.vector_value_width(left_mapped)
                    == self.vector_value_width(right_mapped)
                    and left_component in self.METAL_ARITHMETIC_COMPONENT_RANKS
                    and right_component in self.METAL_ARITHMETIC_COMPONENT_RANKS
                ):
                    common = self.metal_binary_common_component(
                        operator, left_component, right_component
                    )
                    if common is not None:
                        return f"{common}{self.vector_value_width(left_mapped)}"
            if self.is_vector_value_type(left_type) != self.is_vector_value_type(
                right_type
            ):
                # metal_hlsl_values.py: vector op scalar has the common component type.
                vector_type, scalar = (
                    (left_type, expr.right)
                    if self.is_vector_value_type(left_type)
                    else (right_type, expr.left)
                )
                vector_mapped = self.map_type(vector_type)
                component = self.vector_component_type(vector_mapped)
                common = (
                    self.metal_vector_scalar_common_component(
                        operator, component, scalar
                    )
                    if component in self.METAL_ARITHMETIC_COMPONENT_RANKS
                    else None
                )
                if common is not None and common != component:
                    return f"{common}{self.vector_value_width(vector_mapped)}"
            if self.is_vector_value_type(left_type):
                return left_type
            if self.is_vector_value_type(right_type):
                return right_type
            if left_type == "float" or right_type == "float":
                return "float"
            return left_type or right_type
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
        if isinstance(expr, ArrayAccessNode):
            block_access = self.glsl_buffer_block_array_access(expr)
            if block_access is not None:
                return glsl_buffer_indexed_value_type(block_access)
            array_type = self.expression_result_type(expr.array)
            if not array_type:
                array_type = self.unsupported_glsl_buffer_block_expression_type(
                    expr.array
                )
            array_type_name = self.type_name_string(array_type)
            if array_type_name and "[" in array_type_name and "]" in array_type_name:
                return self.metal_for_in_array_element_type(
                    array_type_name, allow_unsized=True
                )
            metal_array_element_type = self.metal_array_element_type(array_type)
            if metal_array_element_type is not None:
                return metal_array_element_type
            if self.is_structured_buffer_type(array_type):
                return self.structured_buffer_element_type(array_type)
            pointee_type = self.pointer_pointee_type_name(array_type)
            if pointee_type is not None:
                return pointee_type
            emulated_matrix = (
                hlsl_emulated_matrix_shape(array_type_name) if array_type_name else None
            )
            if emulated_matrix is not None:
                # Like Metal matrices, indexing selects a column.
                scalar, rows, _columns = emulated_matrix
                return f"{scalar}{rows}"
            if self.is_vector_value_type(array_type):
                native = self.metal_native_narrow_bitcast_storage_type(array_type)
                if native is not None and native[-1:] in {"2", "3", "4"}:
                    return native[:-1]
                return self.vector_component_type(array_type)
            matrix_dimensions = self.metal_matrix_dimensions(
                self.type_name_string(array_type) or ""
            )
            if matrix_dimensions is not None:
                # An MSL floatCxR subscript is a column of R components, as in CrossGL.
                return f"{matrix_dimensions[0]}{matrix_dimensions[2]}"
            component_type = component_access_type(
                self.type_name_string(array_type), C_FAMILY
            )
            if component_type is not None:
                return component_type
            mapped_array_type = (
                self.map_type(array_type) if array_type is not None else None
            )
            if isinstance(mapped_array_type, str):
                if self.is_vector_value_type(mapped_array_type):
                    return self.vector_component_type(mapped_array_type)
                if self.is_matrix_value_type(mapped_array_type):
                    scalar, dimensions = re.fullmatch(
                        r"(float|half|double)(\d)x(\d)", mapped_array_type
                    ).group(1, 3)
                    return f"{scalar}{dimensions}"
            return array_type
        if isinstance(expr, MemberAccessNode):
            block_access = self.glsl_buffer_block_member_access(expr)
            if block_access is not None:
                if glsl_buffer_whole_array_reason(block_access, indexed=False):
                    return glsl_buffer_array_value_type(block_access)
                return block_access["type"]
            object_type = self.expression_result_type(
                expr.object
            ) or self.unsupported_glsl_buffer_block_expression_type(expr.object)
            object_type = self.member_lookup_type_name(object_type)
            member = str(expr.member)
            if object_type and all(ch in "xyzwrgba" for ch in member):
                native = self.metal_native_narrow_bitcast_storage_type(object_type)
                component_type = (
                    native[:-1]
                    if native is not None and native[-1:] in {"2", "3", "4"}
                    else self.vector_component_type(object_type)
                )
                if component_type and len(member) == 1:
                    return component_type
                if component_type:
                    return f"{component_type}{len(member)}"
            if object_type:
                member_type = self.struct_member_types.get(
                    self.type_name_string(object_type), {}
                ).get(member)
                if member_type:
                    return member_type
            member_types = {
                self.type_name_string(members[member])
                for members in self.struct_member_types.values()
                if member in members
            }
            if len(member_types) == 1:
                return next(iter(member_types))
            return None
        if isinstance(expr, PointerAccessNode):
            object_type = self.expression_result_type(
                expr.pointer_expr
            ) or self.unsupported_glsl_buffer_block_expression_type(expr.pointer_expr)
            object_type = self.member_lookup_type_name(object_type)
            member = str(expr.member)
            if object_type:
                member_type = self.struct_member_types.get(
                    self.type_name_string(object_type), {}
                ).get(member)
                if member_type:
                    return member_type
            return None
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
        if isinstance(expr, FunctionCallNode):
            buffer_source = self.structured_buffer_load_source(expr)
            if buffer_source is not None:
                return self.structured_buffer_element_type(
                    self.expression_result_type(buffer_source)
                )
            ray_query_operation = self.ray_query_operation_from_function_call(expr)
            if ray_query_operation is not None:
                return self.metal_ray_query_method_return_type(ray_query_operation)
            func_expr = getattr(expr, "function", None) or getattr(expr, "name", None)
            func_name = getattr(func_expr, "name", func_expr)
            args = getattr(expr, "arguments", getattr(expr, "args", []))
            if isinstance(func_name, str) and func_name.rsplit("::", 1)[-1] == "Some":
                payload_type = self.expected_option_payload_type_name()
                if payload_type is not None:
                    return payload_type
                if self.option_payload_type_name(self.current_expression_expected_type):
                    return None
                return self.expression_result_type(args[0]) if args else None
            if func_name == "RayDesc" and self.requires_metal_builtin_ray_desc(
                func_name
            ):
                return "RayDesc"
            if isinstance(func_name, str) and hlsl_emulated_matrix_shape(func_name):
                return func_name
            if func_name in {"and", "or"} and len(args) == 2:
                for arg in args:
                    shape = self.metal_expression_emulated_matrix_shape(arg)
                    if shape is not None:
                        return f"bool{shape[1]}x{shape[2]}"
                logical_type = self.metal_hlsl_logical_result_type(func_name, args)
                if logical_type is not None:
                    return logical_type
            if func_name == "select" and len(args) == 3:
                for arg in args[:2]:
                    if self.metal_expression_emulated_matrix_shape(arg) is not None:
                        return self.expression_result_type(arg)
            if func_name in self.METAL_WAVE_INTRINSIC_ARITIES:
                return self.metal_wave_result_type(func_name, args)
            numeric_result_type = numeric_trait_method_result_type(self, expr)
            if numeric_result_type:
                return numeric_result_type
            if func_name in {"normalize", "reflect"} and args:
                return self.expression_result_type(args[0])
            if func_name == "inverse" and args:
                return self.expression_result_type(args[0])
            if func_name == "transpose" and args:
                arg_type = self.expression_result_type(args[0])
                dimensions = self.metal_matrix_dimensions(arg_type)
                if dimensions is not None:
                    component_prefix, columns, rows = dimensions
                    return f"{component_prefix}{rows}x{columns}"
                return arg_type
            if func_name == "dot" and args:
                unified_type = self.metal_unified_builtin_result_type(func_name, args)
                if unified_type is not None:
                    return unified_type
                return (
                    self.vector_component_type(self.expression_result_type(args[0]))
                    or "float"
                )
            if func_name == "cross" and args:
                return self.expression_result_type(args[0])
            if (
                func_name == "mod"
                and args
                and func_name not in self.user_function_names
            ):
                return self.expression_result_type(args[0])
            if (
                func_name == "frac"
                and args
                and func_name not in self.user_function_names
            ):
                return self.expression_result_type(args[0])
            if (
                func_name == "lerp"
                and args
                and func_name not in self.user_function_names
            ):
                return self.expression_result_type(args[0])
            componentwise_type = self.metal_componentwise_builtin_result_type(
                func_name, args
            )
            if componentwise_type is not None:
                return componentwise_type
            if (
                func_name in {"inverseSqrt", "inversesqrt", "roundEven"}
                and args
                and func_name not in self.user_function_names
            ):
                unified_type = self.metal_unified_builtin_result_type(func_name, args)
                if unified_type is not None:
                    return unified_type
                return self.expression_result_type(args[0])
            if (
                func_name in self.METAL_DERIVATIVE_FUNCTION_ALIASES
                and args
                and func_name not in self.user_function_names
            ):
                return self.expression_result_type(args[0])
            explicit_as_type = self.metal_explicit_as_type_target(func_name, args)
            if explicit_as_type is not None:
                return explicit_as_type
            bitcast_result_type = self.metal_bitcast_result_type(func_name, args)
            if bitcast_result_type is not None:
                return bitcast_result_type
            integer_bit_result_type = self.metal_integer_bit_result_type(
                func_name, args
            )
            if integer_bit_result_type is not None:
                return integer_bit_result_type
            if (
                func_name in self.PACKED_BYTE_INTRINSICS
                and func_name not in self.user_function_names
            ):
                return self.PACKED_BYTE_INTRINSICS[func_name][0]
            if (
                func_name in self.METAL_ANGLE_CONVERSION_FACTORS
                and len(args or []) == 1
                and func_name not in self.user_function_names
            ):
                argument_type = self.map_type(self.expression_result_type(args[0]))
                if self.vector_component_type(argument_type) in {"int", "uint"}:
                    width = self.vector_value_width(argument_type)
                    return f"float{width or ''}"
                return self.expression_result_type(args[0])
            if func_name in {"mix", "clamp", "min", "max", "saturate"} and args:
                unified_type = self.metal_unified_builtin_result_type(func_name, args)
                if unified_type is not None:
                    return unified_type
                return self.expression_result_type(args[0])
            if is_resource_size_query_operation(func_name) and args:
                texture_type = self.texture_argument_resource_type(args[0])
                raw_type = self.expression_result_type(args[0])
                descriptor = self.texture_query_size_descriptor_for_argument(
                    texture_type, raw_type
                )
                if descriptor is not None:
                    return descriptor["return_type"]
            if (
                is_texture_sampling_operation(func_name)
                or is_texel_fetch_operation(func_name)
            ) and args:
                if self.is_plain_depth_read(func_name, args):
                    # generate_texture_call widens the scalar MSL depth read to (d, 0, 0, 1).
                    return "float4"
                return self.texture_sample_result_type(args[0])
            specialized_func_name = generic_function_call_name(self, func_name, args)
            if specialized_func_name in getattr(self, "function_return_types", {}):
                return self.function_return_types[specialized_func_name]
            overload = self.metal_user_function_overload_for_call(func_name, args)
            if overload is not None:
                return self.metal_effective_function_return_type(overload)
            overloads = self.function_overloads_by_name.get(func_name, [])
            if overloads:
                argument_count = len(args or [])
                viable_return_types = {
                    self.metal_effective_function_return_type(candidate)
                    for candidate in overloads
                    if self.metal_user_function_accepts_argument_count(
                        candidate,
                        argument_count,
                    )
                }
                if len(viable_return_types) == 1:
                    return next(iter(viable_return_types))
                # Type-aware source overload resolution is unavailable here.
                # Differing same-arity results must remain unknown rather than
                # consuming the last declaration in function_return_types.
                return None
            if func_name in getattr(self, "function_return_types", {}):
                return self.function_return_types[func_name]
            boolean_type = boolean_builtin_result_type(
                func_name,
                [self.expression_result_type(arg) for arg in args or ()],
                C_FAMILY,
            )
            if boolean_type is not None:
                return boolean_type
            aggregate_constructor = self.metal_aggregate_constructor_name(func_name)
            if aggregate_constructor is not None:
                return aggregate_constructor
            unsupported_functions = getattr(
                self, "unsupported_glsl_buffer_block_functions", {}
            )
            if func_name in unsupported_functions:
                return unsupported_functions[func_name].get("return_type")
            if func_name == "imageLoad" and args:
                return self.image_load_result_type(args[0])
            if func_name == "subpassLoad":
                return "vec4"
            if self.is_builtin_value_constructor_type(func_name):
                return str(func_name)
            if func_name in {
                "float",
                "half",
                "float16",
                "min16float",
                "min10float",
                "double",
                "f16",
                "f32",
                "f64",
                "int",
                "i8",
                "i16",
                "i32",
                "char",
                "signed char",
                "int8",
                "int16",
                "int8_t",
                "int16_t",
                "int32_t",
                "int64",
                "int64_t",
                "long",
                "signed long",
                "ptrdiff_t",
                "min16int",
                "min12int",
                "uint",
                "u8",
                "u16",
                "u32",
                "i64",
                "u64",
                "uchar",
                "unsigned char",
                "uint8",
                "uint16",
                "uint8_t",
                "uint16_t",
                "uint32_t",
                "uint64",
                "uint64_t",
                "ulong",
                "unsigned long",
                "size_t",
                "min16uint",
                "short",
                "signed short",
                "ushort",
                "unsigned short",
                "bool",
                "vec2",
                "vec3",
                "vec4",
                "vec2<f16>",
                "vec3<f16>",
                "vec4<f16>",
                "vec2<f32>",
                "vec3<f32>",
                "vec4<f32>",
                "vec2<f64>",
                "vec3<f64>",
                "vec4<f64>",
                "vec2<i8>",
                "vec3<i8>",
                "vec4<i8>",
                "vec2<u8>",
                "vec3<u8>",
                "vec4<u8>",
                "vec2<i16>",
                "vec3<i16>",
                "vec4<i16>",
                "vec2<u16>",
                "vec3<u16>",
                "vec4<u16>",
                "vec2<i32>",
                "vec3<i32>",
                "vec4<i32>",
                "vec2<u32>",
                "vec3<u32>",
                "vec4<u32>",
                "vec2<i64>",
                "vec3<i64>",
                "vec4<i64>",
                "vec2<u64>",
                "vec3<u64>",
                "vec4<u64>",
                "vec2<bool>",
                "vec3<bool>",
                "vec4<bool>",
                "ivec2",
                "ivec3",
                "ivec4",
                "i64vec2",
                "i64vec3",
                "i64vec4",
                "uvec2",
                "uvec3",
                "uvec4",
                "u64vec2",
                "u64vec3",
                "u64vec4",
                "bvec2",
                "bvec3",
                "bvec4",
                "float2",
                "float3",
                "float4",
                "packed_float2",
                "packed_float3",
                "packed_float4",
                "simd_float2",
                "simd_float3",
                "simd_float4",
                "int2",
                "int3",
                "int4",
                "packed_int2",
                "packed_int3",
                "packed_int4",
                "simd_int2",
                "simd_int3",
                "simd_int4",
                "uint2",
                "uint3",
                "uint4",
                "packed_uint2",
                "packed_uint3",
                "packed_uint4",
                "simd_uint2",
                "simd_uint3",
                "simd_uint4",
                "bool2",
                "bool3",
                "bool4",
                "half2",
                "half3",
                "half4",
                "packed_half2",
                "packed_half3",
                "packed_half4",
                "simd_half2",
                "simd_half3",
                "simd_half4",
                "f16vec2",
                "f16vec3",
                "f16vec4",
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
                "f16mat2",
                "f16mat3",
                "f16mat4",
                "f16mat2x2",
                "f16mat2x3",
                "f16mat2x4",
                "f16mat3x2",
                "f16mat3x3",
                "f16mat3x4",
                "f16mat4x2",
                "f16mat4x3",
                "f16mat4x4",
                "simd_float2x2",
                "simd_float2x3",
                "simd_float2x4",
                "simd_float3x2",
                "simd_float3x3",
                "simd_float3x4",
                "simd_float4x2",
                "simd_float4x3",
                "simd_float4x4",
                "simd_half2x2",
                "simd_half2x3",
                "simd_half2x4",
                "simd_half3x2",
                "simd_half3x3",
                "simd_half3x4",
                "simd_half4x2",
                "simd_half4x3",
                "simd_half4x4",
                "char2",
                "char3",
                "char4",
                "uchar2",
                "uchar3",
                "uchar4",
                "i8vec2",
                "i8vec3",
                "i8vec4",
                "u8vec2",
                "u8vec3",
                "u8vec4",
                "short2",
                "short3",
                "short4",
                "ushort2",
                "ushort3",
                "ushort4",
                "i16vec2",
                "i16vec3",
                "i16vec4",
                "u16vec2",
                "u16vec3",
                "u16vec4",
                "min16float2",
                "min16float3",
                "min16float4",
                "min10float2",
                "min10float3",
                "min10float4",
                "min16int2",
                "min16int3",
                "min16int4",
                "min12int2",
                "min12int3",
                "min12int4",
                "min16uint2",
                "min16uint3",
                "min16uint4",
                "min16float2x2",
                "min16float2x3",
                "min16float2x4",
                "min16float3x2",
                "min16float3x3",
                "min16float3x4",
                "min16float4x2",
                "min16float4x3",
                "min16float4x4",
                "min10float2x2",
                "min10float2x3",
                "min10float2x4",
                "min10float3x2",
                "min10float3x3",
                "min10float3x4",
                "min10float4x2",
                "min10float4x3",
                "min10float4x4",
            }:
                return str(func_name)
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
            builtin_name = self.metal_builtin_expression_name(expr)
            if builtin_name != expr:
                return builtin_name
            if expr.rsplit("::", 1)[-1] == "None":
                option_default = self.option_none_default_expression()
                if option_default is not None:
                    return option_default
            if expr in self.local_variable_types:
                return self.metal_local_identifier_name(expr)
            return expr
        elif isinstance(expr, IdentifierNode):
            name = expr.name
            builtin_name = self.metal_builtin_expression_name(name)
            if builtin_name != name:
                return builtin_name
            if isinstance(name, str) and name.rsplit("::", 1)[-1] == "None":
                option_default = self.option_none_default_expression()
                if option_default is not None:
                    return option_default
            buffer_block_value = self.unsupported_glsl_buffer_block_access_value(expr)
            if buffer_block_value is not None:
                return buffer_block_value
            if name in self.METAL_RAY_FLAG_VALUES:
                return f"{self.METAL_RAY_FLAG_VALUES[name]}u"
            if name in getattr(self, "enum_variant_constants", {}):
                return enum_value_expression(self, name)
            if (
                name not in self.local_variable_types
                and name in self.ambiguous_cbuffer_members
            ):
                raise MetalRefusal(
                    f"Ambiguous cbuffer member reference '{name}' appears in multiple cbuffers"
                )
            if (
                name not in self.local_variable_types
                and name in self.cbuffer_member_references
            ):
                return self.cbuffer_member_references[name]
            if (
                name not in self.local_variable_types
                and name in self.metal_runtime_initialized_constants
            ):
                return f"{name}()"
            return self.metal_local_identifier_name(name)
        elif isinstance(expr, bool):
            return "true" if expr else "false"
        elif isinstance(expr, int) or isinstance(expr, float):
            return str(expr)
        elif isinstance(expr, (VariableNode, BackendVariableNode)):
            # Fix infinite recursion - directly return the name
            unsupported_value = self.unsupported_glsl_buffer_block_access_value(expr)
            if unsupported_value is not None:
                return unsupported_value
            unsupported_value = (
                self.unsupported_metal_acceleration_structure_array_expression(expr)
            )
            if unsupported_value is not None:
                return unsupported_value
            unsupported_value = (
                self.unsupported_metal_ray_function_table_array_expression(expr)
            )
            if unsupported_value is not None:
                return unsupported_value
            if hasattr(expr, "name"):
                builtin_name = self.metal_builtin_expression_name(expr.name)
                if builtin_name != expr.name:
                    return builtin_name
                if (
                    isinstance(expr.name, str)
                    and expr.name.rsplit("::", 1)[-1] == "None"
                ):
                    option_default = self.option_none_default_expression()
                    if option_default is not None:
                        return option_default
                if expr.name in self.METAL_RAY_FLAG_VALUES:
                    return f"{self.METAL_RAY_FLAG_VALUES[expr.name]}u"
                if expr.name in self.local_variable_types:
                    return self.metal_local_identifier_name(expr.name)
                enum_value = enum_value_expression(self, expr.name)
                if enum_value is not None:
                    return enum_value
                # Not an enum value: a variable this program declares keeps its declared spelling.
                return expr.name
            else:
                return str(expr)
        elif isinstance(expr, BinaryOpNode):
            operator = self.map_operator(expr.op)
            left = self.metal_narrow_vector_operand(
                expr.left, self.generate_binary_operand(expr.left, operator)
            )
            right = self.metal_narrow_vector_operand(
                expr.right, self.generate_binary_operand(expr.right, operator, True)
            )
            if operator in {"<", ">", "<=", ">="}:
                if self.map_type(self.expression_result_type(expr.left)) == "bool":
                    left = f"int({left})"
                if self.map_type(self.expression_result_type(expr.right)) == "bool":
                    right = f"int({right})"
            if operator in {"<", ">", "<=", ">=", "==", "!="}:
                left, right = metal_sign_matched_comparison_operands(
                    self.map_type(self.expression_result_type(expr.left)),
                    self.map_type(self.expression_result_type(expr.right)),
                    left,
                    right,
                )
            left, right, common = self.metal_mixed_vector_binary_operands(
                expr, operator, left, right
            )
            if operator == "%" and common in {"float", "half"}:
                return f"fmod({left}, {right})"
            matrix_arithmetic = self.metal_matrix_builtins.componentwise(
                operator, expr.left, expr.right, left, right
            )
            if matrix_arithmetic is not None:
                return matrix_arithmetic
            # The operands keep their own (common) type and the declaration
            # converts the result: converting each operand to the declared
            # type would turn an integer division into a floating one, or a
            # signed one into an unsigned one (#3166).
            return self.metal_narrow_vector_wrap(
                expr, operator, f"{left} {operator} {right}"
            )
        elif isinstance(expr, (AssignmentNode, BackendAssignmentNode)):
            return assignment_value_expression(self.generate_assignment(expr))
        elif is_comma_expression(expr):
            return c_family_comma_expression(
                expr, self.generate_expression, self.generate_discarded_expression
            )
        elif isinstance(expr, ArrayLiteralNode):
            element_type, array_suffix = split_array_type_suffix(
                self.current_expression_expected_type or ""
            )
            matrix_element_type = (
                element_type
                if array_suffix.count("[") == 1
                and self.is_matrix_value_type(element_type)
                else None
            )
            # A structure's brace list gives each nested list its member's type,
            # and an array of structures' list gives each its element type, so a
            # matrix array member is written as the array above is.
            nested_types = []
            if element_type in self.struct_member_types and array_suffix:
                nested_types = [
                    self.metal_for_in_array_element_type(
                        self.current_expression_expected_type
                    )
                ] * len(expr.elements)
            elif element_type in self.struct_member_types:
                nested_types = list(self.struct_member_types[element_type].values())
            rendered_elements = []
            for index, element in enumerate(expr.elements):
                nested_type = (
                    nested_types[index]
                    if isinstance(element, ArrayLiteralNode)
                    and index < len(nested_types)
                    else None
                )
                rendered = self.generate_expression_with_expected(element, nested_type)
                # A scalar cannot initialize a matrix element of a brace list, so
                # write the matrix constructor, as GLSL output's mat3(0.0) does.
                if (
                    matrix_element_type is not None
                    and self.expression_component_count(element) == 1
                ):
                    rendered = f"{self.map_type(matrix_element_type)}({rendered})"
                rendered_elements.append(rendered)
            elements = ", ".join(rendered_elements)
            return f"{{{elements}}}"
        elif isinstance(expr, PointerReinterpretNode):
            return self.generate_metal_storage_pointer_reinterpret(expr)
        elif isinstance(expr, UnaryOpNode):
            # In C++/Metal, ``9223372036854775808`` cannot be represented by a
            # signed literal type, so spelling INT64_MIN as unary minus applied
            # to that token first creates an unsigned value and triggers a
            # warning-fatal conversion back to ``long``.  Keep the exact value
            # in signed arithmetic using two representable long literals.
            operand_value = getattr(getattr(expr, "operand", None), "value", None)
            if (
                getattr(expr, "op", getattr(expr, "operator", None)) == "-"
                and not getattr(expr, "is_postfix", False)
                and isinstance(operand_value, int)
                and not isinstance(operand_value, bool)
                and operand_value == 1 << 63
                and self.map_type(
                    getattr(getattr(expr.operand, "literal_type", None), "name", None)
                )
                not in {"ulong", "uint64_t"}
            ):
                return "(-9223372036854775807L - 1L)"
            negated_literal = self.metal_negated_wide_integer_literal(expr)
            if negated_literal is not None:
                return negated_literal
            local_reinterpret = self.generate_metal_local_reinterpret_read(expr)
            if local_reinterpret is not None:
                return local_reinterpret
            operator = self.map_operator(expr.op)
            if operator in {"++", "--"} and self.is_metal_signed_builtin_member(
                expr.operand
            ):
                # The uint storage steps with the int's bits; the value read is converted back.
                storage = self.generate_metal_signed_builtin_member_storage(
                    expr.operand
                )
                if getattr(expr, "is_postfix", False):
                    return f"int({storage}{operator})"
                return f"int({operator}{storage})"
            operand = self.generate_unary_operand(expr.operand)
            if operator in {"++", "--"}:
                byte_update = self.metal_narrow_integer_update_expression(
                    expr, expr.operand, operator
                )
                if byte_update is not None:
                    return byte_update
                # Buffer storage whose reads convert (a packed vector's
                # components) is updated in place through the storage itself.
                layout_store = self.metal_hlsl_layout_store_target(expr.operand)
                if layout_store is not None and not layout_store[1]:
                    operand = layout_store[0]
            if getattr(expr, "is_postfix", False):
                return f"{postfix_operand(operand)}{operator}"
            if operator == "!":
                operand = metal_logical_not_operand(
                    operand, self.expression_result_type(expr.operand)
                )
            operand = self.metal_narrow_vector_operand(expr.operand, operand)
            if operator == "-":
                matrix_negation = self.metal_matrix_builtins.negation(
                    expr.operand, operand
                )
                if matrix_negation is not None:
                    return matrix_negation
            if operator in {"+", "-"} and operand.startswith(("+", "-")):
                # Keep adjacent signs from becoming an increment or decrement.
                operand = f"({operand})"
            return self.metal_narrow_vector_wrap(expr, operator, f"{operator}{operand}")
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
            unsupported_value = self.unsupported_glsl_buffer_block_access_value(expr)
            if unsupported_value is not None:
                return unsupported_value
            stage_io_access = self.generate_metal_stage_io_lowered_array_access(expr)
            if stage_io_access is not None:
                return stage_io_access
            unsupported_value = (
                self.unsupported_metal_acceleration_structure_array_expression(expr)
            )
            if unsupported_value is not None:
                return unsupported_value
            unsupported_value = (
                self.unsupported_metal_ray_function_table_array_expression(expr)
            )
            if unsupported_value is not None:
                return unsupported_value
            block_load = self.generate_glsl_buffer_block_array_load(expr)
            if block_load is not None:
                return block_load
            layout_element = self.metal_hlsl_layout_element_access(expr)
            if layout_element is not None:
                access, conversion = layout_element
                return f"{conversion}({access})" if conversion else access
            if is_whole_texel_load(expr.array, self.function_call_name):
                # The subscript selects a component of the whole texel.
                array = self.generate_expression_with_expected(expr.array, None)
            else:
                array = self.generate_expression(expr.array)
            if isinstance(
                expr.array,
                (
                    AssignmentNode,
                    BinaryOpNode,
                    TernaryOpNode,
                    CastNode,
                    PointerReinterpretNode,
                    UnaryOpNode,
                ),
            ):
                array = f"({array})"
            index = self.generate_expression(expr.index)
            return f"{array}[{index}]"
        elif isinstance(expr, ConstructorNode):
            registered_conversion = self.metal_registered_structure_scalar_conversion(
                expr,
                getattr(expr, "constructor_type", None),
                getattr(expr, "arguments", []),
            )
            if registered_conversion is not None:
                return registered_conversion
            enum_constructor = generate_enum_constructor_expression(self, expr)
            if enum_constructor is not None:
                return enum_constructor
            constructor = generate_struct_constructor_expression(self, expr)
            if constructor is not None:
                return constructor
            constructor_type = getattr(expr, "constructor_type", None)
            matrix_constructor = self.generate_metal_matrix_constructor_override(
                constructor_type, getattr(expr, "arguments", [])
            )
            if matrix_constructor is not None:
                return matrix_constructor
            if self.is_builtin_value_constructor_type(constructor_type):
                metal_type = self.map_type(constructor_type)
                matrix_resize = self.generate_metal_matrix_resize_constructor(
                    metal_type, getattr(expr, "arguments", [])
                ) or self.metal_matrix_builtins.component_constructor(
                    metal_type, getattr(expr, "arguments", [])
                )
                if matrix_resize is not None:
                    return matrix_resize
                args = ", ".join(
                    self.generate_metal_value_constructor_arguments(
                        metal_type, getattr(expr, "arguments", [])
                    )
                )
                if getattr(
                    expr, "is_braced_constructor", False
                ) and self.is_vector_value_type(constructor_type):
                    width = self.value_component_count(constructor_type)
                    counts = [
                        self.expression_component_count(arg) for arg in expr.arguments
                    ]
                    if (
                        width
                        and all(count is not None for count in counts)
                        and sum(counts) > width
                    ):
                        raise UnsupportedMetalFeatureError(
                            "vector-list-initialization",
                            f"Vector initializer for {metal_type} exceeds its {width} components",
                            reason="element-count-mismatch",
                            source_location=getattr(expr, "source_location", None),
                        )
                    rendered = f"{metal_type}{{{args}}}"
                else:
                    rendered = f"{metal_type}({args})"
                return (
                    self.metal_narrow_integer_conversion_expression(
                        rendered, constructor_type
                    )
                    or rendered
                )
            return str(expr)
        elif isinstance(expr, FunctionCallNode):
            option_payload = self.option_constructor_payload_expression(expr)
            if option_payload is not None:
                return option_payload

            func_expr = getattr(expr, "function", None)
            if func_expr is None:
                func_expr = expr.name
            numeric_trait_call = generate_numeric_trait_method_call(
                self,
                func_expr,
                expr.args,
            )
            if numeric_trait_call is not None:
                return numeric_trait_call

            func_name = None
            if hasattr(func_expr, "name") and isinstance(func_expr.name, str):
                func_name = func_expr.name
                callee = func_name
            elif isinstance(func_expr, str):
                func_name = func_expr
                callee = func_expr
            else:
                callee = self.generate_expression(func_expr)

            if (
                func_name == "select"
                and func_name not in self.user_function_names
                and len(expr.args) == 3
                and all(
                    self.is_matrix_value_type(self.expression_result_type(argument))
                    for argument in expr.args[:2]
                )
            ):
                false_value, true_value, condition = expr.args
                condition_type = self.map_type(self.expression_result_type(condition))
                if condition_type != "bool":
                    return self.refuse_lowering(
                        "matrix select condition must be a scalar bool",
                        self.generate_expression(false_value),
                    )
                if not all(
                    self.metal_expression_is_side_effect_free(value)
                    for value in (false_value, true_value)
                ):
                    return self.refuse_lowering(
                        "matrix select evaluates both value arguments, but a "
                        "conditional expression would skip one side-effecting argument",
                        self.generate_expression(false_value),
                    )
                return (
                    f"({self.generate_expression(condition)} ? "
                    f"{self.generate_expression(true_value)} : "
                    f"{self.generate_expression(false_value)})"
                )

            if (
                func_name in PACK_FUNCTIONS
                and func_name not in self.user_function_names
            ):
                return metal_pack_call(
                    self, func_name, [self.generate_expression(a) for a in expr.args]
                )
            packed_dot = self.generate_packed_dot4add_call(func_name, expr.args)
            if packed_dot is not None:
                return packed_dot
            if (
                func_name in BITFIELD_ARITIES
                and func_name not in self.user_function_names
            ):
                return metal_bitfield_call(
                    self, func_name, [self.generate_expression(a) for a in expr.args]
                )
            boolean_mix = boolean_mix_selector(
                self, func_name, expr.args, self.user_function_names
            )
            if boolean_mix is not None:
                return metal_boolean_mix(self, expr.args, boolean_mix)
            exponent_call = metal_exponent_call(self, func_name, expr.args)
            if exponent_call is not None:
                return exponent_call
            bit_scan_call = metal_bit_scan_call(self, func_name, expr.args)
            if bit_scan_call is not None:
                return bit_scan_call
            extended_arithmetic = metal_extended_arithmetic_call(
                self, func_name, expr.args
            )
            if extended_arithmetic is not None:
                return extended_arithmetic
            packed_bytes = self.generate_packed_byte_call(func_name, expr.args)
            if packed_bytes is not None:
                return packed_bytes

            vector_set = re.fullmatch(
                r"CrossGLMetalVectorIndex_([iu](?:8|16)vec[234])_set",
                str(func_name),
            )
            if vector_set and len(expr.args) == 3:
                member = expr.args[0]
                while isinstance(member, ArrayAccessNode):
                    member = member.array
                if (
                    isinstance(member, MemberAccessNode)
                    and self.member_lookup_type_name(
                        self.expression_result_type(member.object)
                    )
                    in self.structs_by_name
                ):
                    native = self.metal_native_narrow_bitcast_storage_type(
                        vector_set[1]
                    )
                    target = self.generate_expression(expr.args[0])
                    lane = self.generate_expression_with_expected(expr.args[1], "uint")
                    selected = self.generate_expression(expr.args[2])
                    return f"({target}[{lane}] = {native[:-1]}({selected}))"

            block_length = self.glsl_buffer_block_length_call(func_expr, expr.args)
            if block_length is not None:
                return block_length
            length_object = length_method_object(func_expr, expr.args)
            matrix_dimensions = (
                self.metal_matrix_dimensions(
                    self.expression_result_type(length_object) or ""
                )
                if length_object is not None
                else None
            )
            if matrix_dimensions is not None:
                # A matrix's length() is its column count (GLSL 4.60 section 5.10).
                return f"int({matrix_dimensions[1]})"

            unsupported_table_call = (
                self.unsupported_metal_ray_function_table_array_member_call(func_expr)
            )
            if unsupported_table_call is not None:
                return unsupported_table_call

            ray_query_operation = self.ray_query_operation_from_function_call(expr)
            if ray_query_operation is not None and isinstance(
                func_expr, MemberAccessNode
            ):
                return self.generate_metal_ray_query_call(
                    ray_query_operation,
                    getattr(
                        func_expr,
                        "object",
                        getattr(func_expr, "object_expr", None),
                    ),
                    expr.args,
                )

            sizeof_bytes = sizeof_call_bytes(self, func_name, expr.args, "metal")
            if sizeof_bytes is not None:
                return str(sizeof_bytes)

            static_generic_call = generate_static_generic_numeric_call(self, func_name)
            if static_generic_call is not None:
                return static_generic_call

            if isinstance(func_name, str) and not self.metal_function_name_is_shadowed(
                func_name
            ):
                matrix_constructor = self.generate_metal_matrix_constructor_override(
                    func_name, expr.args
                )
                if matrix_constructor is not None:
                    return matrix_constructor
                logical_intrinsic = self.generate_metal_hlsl_logical_intrinsic(
                    func_name, expr.args
                )
                if logical_intrinsic is not None:
                    return logical_intrinsic

            if func_name == "RayDesc" and self.requires_metal_builtin_ray_desc(
                func_name
            ):
                self.required_metal_ray_desc_runtime = True
                args = ", ".join(self.generate_expression(arg) for arg in expr.args)
                return f"CglRayDesc({args})"

            unsupported_call = self.unsupported_glsl_buffer_block_function_call(
                func_name
            )
            if unsupported_call is not None:
                return unsupported_call

            input_attachment_call = self.unsupported_input_attachment_call(func_name)
            if input_attachment_call is not None:
                return input_attachment_call

            enum_constructor = generate_enum_constructor_call(
                self, func_name, expr.args
            )
            if enum_constructor is not None:
                return enum_constructor

            glsl_block_atomic_call = self.generate_glsl_buffer_block_atomic_call(
                func_name, expr.args
            )
            if glsl_block_atomic_call is not None:
                return glsl_block_atomic_call

            buffer_atomic_call = self.generate_metal_buffer_resource_atomic_call(
                func_name, expr.args
            )
            if buffer_atomic_call is not None:
                return buffer_atomic_call

            synchronization_call = self.synchronization_function_call(
                func_name,
                expr.args,
                source_location=getattr(expr, "source_location", None),
            )
            if synchronization_call is not None:
                return synchronization_call

            wave_call = self.generate_metal_wave_operation(func_name, expr.args)
            if wave_call is not None:
                return wave_call

            atomic_call = self.generate_atomic_function_call(func_name, expr.args)
            if atomic_call is not None:
                return atomic_call

            mesh_output_call = self.generate_metal_mesh_output_call(
                func_name, expr.args
            )
            if mesh_output_call is not None:
                return mesh_output_call

            if func_name == TEXTURE_RESIDENCY_OPERATION:
                return metal_texture_residency_call(self, expr.args, expr)
            if func_name == TEXTURE_RESIDENT_OPERATION:
                return metal_texture_resident_call(self, expr.args, expr)

            if is_texture_lod_clamp_operation(func_name):
                return metal_texture_lod_clamp_call(self, func_name, expr.args, expr)

            texture_call = self.generate_texture_call(func_name, expr.args)
            if texture_call is not None:
                return texture_call

            buffer_call = self.generate_buffer_call(func_name, expr.args)
            if buffer_call is not None:
                return buffer_call
            specialized_func_name = generic_function_call_name(
                self,
                func_name,
                expr.args,
            )
            argument_func_name = func_name
            if specialized_func_name is not None:
                callee = specialized_func_name
                func_name = specialized_func_name
            else:
                reject_unresolved_generic_function_call(self, func_name, "Metal")
            scalar_geometric_call = self.metal_scalar_geometric_call(
                func_name, expr.args
            )
            if scalar_geometric_call is not None:
                return scalar_geometric_call
            if func_name == "normalize" and func_name not in self.user_function_names:
                args = ", ".join(self.generate_expression(arg) for arg in expr.args)
                call_name = (
                    "metal::normalize"
                    if self.metal_function_name_is_shadowed("normalize")
                    else "normalize"
                )
                return f"{call_name}({args})"
            matrix_builtin = self.metal_matrix_builtins.call(func_name, expr.args)
            if matrix_builtin is not None:
                return matrix_builtin
            if (
                func_name == "inverse"
                and len(expr.args) == 1
                and func_name not in self.user_function_names
            ):
                arg_type = self.map_type(self.expression_result_type(expr.args[0]))
                if arg_type in {"float2x2", "float3x3", "float4x4"}:
                    self.required_metal_inverse_helpers.add(arg_type)
                    arg = self.generate_expression(expr.args[0])
                    return f"__crossgl_inverse_{arg_type}({arg})"
            if (
                func_name == "mod"
                and len(expr.args) == 2
                and func_name not in self.user_function_names
            ):
                left = self.generate_expression(expr.args[0])
                right = self.generate_expression(expr.args[1])
                return f"(({left}) - (({right}) * floor(({left}) / ({right}))))"
            if (
                func_name == "frac"
                and len(expr.args) == 1
                and func_name not in self.user_function_names
            ):
                arg = self.generate_expression(expr.args[0])
                fract_name = (
                    "metal::fract"
                    if self.metal_function_name_is_shadowed("fract")
                    else "fract"
                )
                return f"{fract_name}({arg})"
            if (
                func_name == "lerp"
                and len(expr.args) == 3
                and func_name not in self.user_function_names
            ):
                args = ", ".join(self.generate_expression(arg) for arg in expr.args)
                mix_name = (
                    "metal::mix"
                    if self.metal_function_name_is_shadowed("mix")
                    else "mix"
                )
                return f"{mix_name}({args})"
            if (
                func_name == "atan"
                and len(expr.args) == 2
                and func_name not in self.user_function_names
            ):
                args = ", ".join(self.generate_expression(arg) for arg in expr.args)
                atan2_name = (
                    "metal::atan2"
                    if self.metal_function_name_is_shadowed("atan2")
                    else "atan2"
                )
                return f"{atan2_name}({args})"
            if (
                func_name in {"inverseSqrt", "inversesqrt"}
                and len(expr.args) == 1
                and func_name not in self.user_function_names
            ):
                arg = self.generate_expression(expr.args[0])
                rsqrt_name = (
                    "metal::rsqrt"
                    if self.metal_function_name_is_shadowed("rsqrt")
                    else "rsqrt"
                )
                return f"{rsqrt_name}({arg})"
            if (
                func_name == "roundEven"
                and len(expr.args) == 1
                and func_name not in self.user_function_names
            ):
                # MSL rint() rounds halfway cases to the nearest even integer (MSL 4.1 section 6.5).
                # It has only float and half overloads, so integer operands are converted to float.
                unified_args = self.generate_metal_unified_builtin_arguments(
                    func_name, expr.args
                )
                arg = (
                    unified_args[0]
                    if unified_args is not None
                    else self.generate_expression(expr.args[0])
                )
                rint_name = (
                    "metal::rint"
                    if self.metal_function_name_is_shadowed("rint")
                    else "rint"
                )
                return f"{rint_name}({arg})"
            derivative_name = self.METAL_DERIVATIVE_FUNCTION_ALIASES.get(func_name)
            if (
                derivative_name is not None
                and len(expr.args) == 1
                and func_name not in self.user_function_names
            ):
                arg = self.generate_expression(expr.args[0])
                derivative_call_name = (
                    f"metal::{derivative_name}"
                    if self.metal_function_name_is_shadowed(derivative_name)
                    else derivative_name
                )
                return f"{derivative_call_name}({arg})"
            explicit_as_type = self.generate_metal_explicit_as_type_call(
                func_name, expr.args
            )
            if explicit_as_type is not None:
                return explicit_as_type
            bitcast_call = self.generate_metal_bitcast_call(func_name, expr.args)
            if bitcast_call is not None:
                return bitcast_call
            integer_bit_call = self.generate_metal_integer_bit_call(
                func_name, expr.args
            )
            if integer_bit_call is not None:
                return integer_bit_call
            angle_conversion = self.generate_metal_angle_conversion_call(
                func_name, expr.args
            )
            if angle_conversion is not None:
                return angle_conversion
            if (
                func_name in self.METAL_STDLIB_BUILTIN_FUNCTIONS
                and func_name not in self.user_function_names
            ):
                integer_dot = self.generate_metal_integer_dot_call(func_name, expr.args)
                if integer_dot is not None:
                    return integer_dot
                unified_args = self.generate_metal_unified_builtin_arguments(
                    func_name, expr.args
                )
                if unified_args is not None:
                    args = ", ".join(unified_args)
                elif func_name in self.METAL_BFLOAT_ARGUMENT_PROMOTED_FUNCTIONS:
                    args = ", ".join(
                        self.generate_metal_bfloat_promoted_argument(arg)
                        for arg in expr.args
                    )
                else:
                    args = ", ".join(self.generate_expression(arg) for arg in expr.args)
                if self.metal_fast_math_call(expr, func_name):
                    return f"metal::fast::{func_name}({args})"
                if self.metal_precise_math_call(
                    expr, func_name
                ) or self.metal_safe_math_call(expr, func_name):
                    return f"metal::precise::{func_name}({args})"
                call_name = (
                    f"metal::{func_name}"
                    if self.metal_function_name_is_shadowed(func_name)
                    else func_name
                )
                return f"{call_name}({args})"
            registered_conversion = self.metal_registered_structure_scalar_conversion(
                expr,
                func_name,
                expr.args,
            )
            if registered_conversion is not None:
                return registered_conversion
            aggregate_constructor = self.generate_metal_aggregate_constructor_call(
                expr,
                func_name,
            )
            if aggregate_constructor is not None:
                return aggregate_constructor
            if (
                func_name in self.metal_type_aliases
                or (
                    func_name not in self.user_function_names
                    and self.is_vector_value_type(func_name)
                )
                or re.fullmatch(
                    r"(?:u?int(?:8|16|32|64)(?:_t)?|"
                    r"(?:float(?:16|32|64)|bfloat16)(?:_t)?)[234]",
                    str(func_name),
                )
                is not None
                or func_name in [
                    "float",
                    "half",
                    "float16",
                    "float16_t",
                    "min16float",
                    "min10float",
                    "double",
                    "f16",
                    "f32",
                    "f64",
                    "bfloat",
                    "bfloat16",
                    "bfloat16_t",
                    "int",
                    "i8",
                    "i16",
                    "i32",
                    "char",
                    "signed char",
                    "int8",
                    "int16",
                    "int8_t",
                    "int16_t",
                    "int32_t",
                    "int64",
                    "int64_t",
                    "long",
                    "signed long",
                    "ptrdiff_t",
                    "min16int",
                    "min12int",
                    "uint",
                    "u8",
                    "u16",
                    "u32",
                    "i64",
                    "u64",
                    "uchar",
                    "unsigned char",
                    "uint8",
                    "uint16",
                    "uint8_t",
                    "uint16_t",
                    "uint32_t",
                    "uint64",
                    "uint64_t",
                    "ulong",
                    "unsigned long",
                    "size_t",
                    "min16uint",
                    "short",
                    "signed short",
                    "ushort",
                    "unsigned short",
                    "bool",
                    "vec2",
                    "vec3",
                    "vec4",
                    "vec2<f16>",
                    "vec3<f16>",
                    "vec4<f16>",
                    "vec2<f32>",
                    "vec3<f32>",
                    "vec4<f32>",
                    "vec2<f64>",
                    "vec3<f64>",
                    "vec4<f64>",
                    "vec2<i8>",
                    "vec3<i8>",
                    "vec4<i8>",
                    "vec2<u8>",
                    "vec3<u8>",
                    "vec4<u8>",
                    "vec2<i16>",
                    "vec3<i16>",
                    "vec4<i16>",
                    "vec2<u16>",
                    "vec3<u16>",
                    "vec4<u16>",
                    "vec2<i32>",
                    "vec3<i32>",
                    "vec4<i32>",
                    "vec2<u32>",
                    "vec3<u32>",
                    "vec4<u32>",
                    "vec2<i64>",
                    "vec3<i64>",
                    "vec4<i64>",
                    "vec2<u64>",
                    "vec3<u64>",
                    "vec4<u64>",
                    "vec2<bool>",
                    "vec3<bool>",
                    "vec4<bool>",
                    "ivec2",
                    "ivec3",
                    "ivec4",
                    "i64vec2",
                    "i64vec3",
                    "i64vec4",
                    "uvec2",
                    "uvec3",
                    "uvec4",
                    "u64vec2",
                    "u64vec3",
                    "u64vec4",
                    "bvec2",
                    "bvec3",
                    "bvec4",
                    "packed_float2",
                    "packed_float3",
                    "packed_float4",
                    "simd_float2",
                    "simd_float3",
                    "simd_float4",
                    "half2",
                    "half3",
                    "half4",
                    "packed_half2",
                    "packed_half3",
                    "packed_half4",
                    "simd_half2",
                    "simd_half3",
                    "simd_half4",
                    "f16vec2",
                    "f16vec3",
                    "f16vec4",
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
                    "f16mat2",
                    "f16mat3",
                    "f16mat4",
                    "f16mat2x2",
                    "f16mat2x3",
                    "f16mat2x4",
                    "f16mat3x2",
                    "f16mat3x3",
                    "f16mat3x4",
                    "f16mat4x2",
                    "f16mat4x3",
                    "f16mat4x4",
                    "char2",
                    "char3",
                    "char4",
                    "uchar2",
                    "uchar3",
                    "uchar4",
                    "packed_int2",
                    "packed_int3",
                    "packed_int4",
                    "packed_uint2",
                    "packed_uint3",
                    "packed_uint4",
                    "simd_int2",
                    "simd_int3",
                    "simd_int4",
                    "simd_uint2",
                    "simd_uint3",
                    "simd_uint4",
                    "simd_float2x2",
                    "simd_float2x3",
                    "simd_float2x4",
                    "simd_float3x2",
                    "simd_float3x3",
                    "simd_float3x4",
                    "simd_float4x2",
                    "simd_float4x3",
                    "simd_float4x4",
                    "simd_half2x2",
                    "simd_half2x3",
                    "simd_half2x4",
                    "simd_half3x2",
                    "simd_half3x3",
                    "simd_half3x4",
                    "simd_half4x2",
                    "simd_half4x3",
                    "simd_half4x4",
                    "i8vec2",
                    "i8vec3",
                    "i8vec4",
                    "u8vec2",
                    "u8vec3",
                    "u8vec4",
                    "short2",
                    "short3",
                    "short4",
                    "ushort2",
                    "ushort3",
                    "ushort4",
                    "i16vec2",
                    "i16vec3",
                    "i16vec4",
                    "u16vec2",
                    "u16vec3",
                    "u16vec4",
                    "min16float2",
                    "min16float3",
                    "min16float4",
                    "min10float2",
                    "min10float3",
                    "min10float4",
                    "min16int2",
                    "min16int3",
                    "min16int4",
                    "min12int2",
                    "min12int3",
                    "min12int4",
                    "min16uint2",
                    "min16uint3",
                    "min16uint4",
                    "min16float2x2",
                    "min16float2x3",
                    "min16float2x4",
                    "min16float3x2",
                    "min16float3x3",
                    "min16float3x4",
                    "min16float4x2",
                    "min16float4x3",
                    "min16float4x4",
                    "min10float2x2",
                    "min10float2x3",
                    "min10float2x4",
                    "min10float3x2",
                    "min10float3x3",
                    "min10float3x4",
                    "min10float4x2",
                    "min10float4x3",
                    "min10float4x4",
                ]
            ):
                metal_type = self.map_type(func_name)
                matrix_resize = self.generate_metal_matrix_resize_constructor(
                    metal_type, expr.args
                ) or self.metal_matrix_builtins.component_constructor(
                    metal_type, expr.args
                )
                if matrix_resize is not None:
                    return matrix_resize
                args = ", ".join(
                    self.generate_metal_value_constructor_arguments(
                        metal_type, expr.args
                    )
                )
                direct = self.metal_direct_narrow_float_conversion(
                    func_name, expr.args, args
                )
                if direct is not None:
                    return direct
                rendered = f"{metal_type}({args})"
                return (
                    self.metal_narrow_integer_conversion_expression(rendered, func_name)
                    or rendered
                )
            readonly_raw_buffer_call = self.readonly_raw_buffer_call_diagnostic(
                argument_func_name, expr.args
            )
            if readonly_raw_buffer_call is not None:
                return readonly_raw_buffer_call
            readonly_parameter_call = self.readonly_metal_parameter_call_diagnostic(
                argument_func_name, expr.args
            )
            if readonly_parameter_call is not None:
                return readonly_parameter_call
            mesh_context_call = self.metal_mesh_dispatch_context_call_diagnostic(
                func_name
            )
            if mesh_context_call is not None:
                return mesh_context_call
            self.validate_address_space_call(argument_func_name, expr.args)
            wave_lane_call = self.metal_wave_lane_helper_call_diagnostic(func_name)
            if wave_lane_call is not None:
                return wave_lane_call
            readonly_mesh_payload_call = (
                self.readonly_metal_mesh_payload_call_diagnostic(
                    argument_func_name, expr.args
                )
            )
            if readonly_mesh_payload_call is not None:
                return readonly_mesh_payload_call
            self.validate_function_resource_argument_types(func_name, expr.args)
            self.validate_function_image_access_arguments(func_name, expr.args)
            args = self.generate_function_call_arguments(argument_func_name, expr.args)
            if self.function_call_matches_known_signature(func_name, expr.args):
                args.extend(
                    self.required_function_stage_output_argument_names(func_name)
                )
                args.extend(
                    self.required_function_stage_parameter_argument_names(func_name)
                )
                args.extend(
                    self.cbuffer_parameter_name(cbuffer)
                    for cbuffer in self.required_function_cbuffers(func_name)
                )
                args.extend(
                    self.required_function_thread_private_global_argument_names(
                        func_name
                    )
                )
                args.extend(self.required_function_resource_argument_names(func_name))
                args.extend(
                    self.required_metal_mesh_dispatch_context_arguments(func_name)
                )
                wave_lane_args = self.required_metal_wave_lane_context_arguments(
                    func_name
                )
                if wave_lane_args is not None:
                    args.extend(wave_lane_args)
            args = ", ".join(args)
            if callee == func_name and (
                self.metal_precise_math_call(expr, func_name)
                or self.metal_safe_math_call(expr, func_name)
            ):
                return f"metal::precise::{func_name}({args})"
            return f"{callee}({args})"
        elif isinstance(expr, MemberAccessNode):
            unsupported_value = self.unsupported_glsl_buffer_block_access_value(expr)
            if unsupported_value is not None:
                return unsupported_value
            stage_io_access = self.generate_metal_stage_io_lowered_member_access(expr)
            if stage_io_access is not None:
                return stage_io_access
            if id(
                expr
            ) not in self.metal_signed_member_raw_reads and self.is_metal_signed_builtin_member(
                expr
            ):
                return f"int({self.generate_metal_signed_builtin_member_storage(expr)})"
            unsupported_value = (
                self.unsupported_metal_ray_function_table_array_expression(expr)
            )
            if unsupported_value is not None:
                return unsupported_value
            block_load = self.generate_glsl_buffer_block_member_load(expr)
            if block_load is not None:
                return block_load
            suppress_suffix = self.member_access_is_image_load_component(expr)
            object_name = self.expression_name(expr.object)
            if (
                object_name in self.metal_static_struct_members
                and str(expr.member)
                in self.metal_static_struct_members.get(object_name, set())
                and object_name not in self.local_variable_types
                and object_name not in self.metal_program_scope_value_global_types
            ):
                return f"{object_name}::{expr.member}"
            previous_suppression = self.suppress_image_load_component_suffix
            self.suppress_image_load_component_suffix = (
                previous_suppression or suppress_suffix
            )
            try:
                obj = self.generate_expression_with_expected(expr.object, None)
            finally:
                self.suppress_image_load_component_suffix = previous_suppression
            if isinstance(
                expr.object, (BinaryOpNode, TernaryOpNode, UnaryOpNode)
            ) and not self.metal_expression_is_parenthesized(obj):
                # Member access binds tighter than every operator; keep the
                # operand grouping that CrossGL expresses through the tree.
                obj = f"({obj})"
            if self.member_access_uses_pointer_operator(expr):
                return f"{obj}->{expr.member}"
            layout_member = self.metal_hlsl_layout_member(expr)
            if (
                layout_member is not None
                and layout_member.read_conversion
                and layout_member.array_count is None
            ):
                return f"{layout_member.read_conversion}({obj}.{expr.member})"
            return f"{obj}.{expr.member}"
        elif isinstance(expr, PointerAccessNode):
            pointer = self.generate_expression_with_expected(expr.pointer_expr, None)
            return f"{pointer}->{expr.member}"
        elif isinstance(expr, TernaryOpNode):
            true_code, false_code = self.metal_conditional_arms(
                expr,
                self.generate_expression(expr.true_expr),
                self.generate_expression(expr.false_expr),
            )
            condition = self.generate_expression(expr.condition)
            # `?:` is right-associative: an unparenthesized conditional
            # condition would regroup as part of the false arm.
            if isinstance(expr.condition, TernaryOpNode):
                condition = f"({condition})"
            return f"{condition} ? {true_code} : {false_code}"
        elif hasattr(expr, "__class__") and "Literal" in str(expr.__class__):
            if hasattr(expr, "value"):
                value = expr.value
                literal_type = getattr(
                    getattr(expr, "literal_type", None), "name", None
                )
                if isinstance(value, int) and not isinstance(value, bool):
                    mapped_type = self.map_type(literal_type) if literal_type else None
                    if mapped_type in {"long", "ulong", "int64_t", "uint64_t"}:
                        suffix = "ul" if mapped_type in {"ulong", "uint64_t"} else "l"
                        return f"{value}{suffix}"
                if (
                    literal_type == "uint"
                    and isinstance(value, int)
                    and not isinstance(value, bool)
                ):
                    return f"{value}u"
                if isinstance(value, bool):
                    return "true" if value else "false"
                if isinstance(value, int):
                    # Unsuffixed decimal literals beyond int are long in
                    # Metal; the source type of such a literal is uint up to
                    # 2^32 - 1 (as in GLSL and HLSL) and 64-bit above it.
                    if value > 0x7FFFFFFFFFFFFFFF:
                        return f"{value}ul"
                    if 0x7FFFFFFF < value <= 0xFFFFFFFF:
                        return f"{value}u"
                    if value < -0x80000000:
                        return f"{value}l"
                if isinstance(value, str) and not (
                    value.startswith('"') and value.endswith('"')
                ):
                    return f'"{value}"'  # Add quotes for string literals
                return str(value)
            return str(expr)
        elif hasattr(expr, "__class__") and "Identifier" in str(expr.__class__):
            name = getattr(expr, "name", str(expr))
            if isinstance(name, str) and name.rsplit("::", 1)[-1] == "None":
                option_default = self.option_none_default_expression()
                if option_default is not None:
                    return option_default
            unsupported_value = self.unsupported_glsl_buffer_block_access_value(expr)
            if unsupported_value is not None:
                return unsupported_value
            if name in self.METAL_RAY_FLAG_VALUES:
                return f"{self.METAL_RAY_FLAG_VALUES[name]}u"
            if name in getattr(self, "enum_variant_constants", {}):
                return enum_value_expression(self, name)
            if (
                name not in self.local_variable_types
                and name in self.ambiguous_cbuffer_members
            ):
                raise MetalRefusal(
                    f"Ambiguous cbuffer member reference '{name}' appears in multiple cbuffers"
                )
            if (
                name not in self.local_variable_types
                and name in self.cbuffer_member_references
            ):
                return self.cbuffer_member_references[name]
            if (
                name not in self.local_variable_types
                and name in self.metal_runtime_initialized_constants
            ):
                return f"{name}()"
            return name
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

    def generate_metal_buffer_resource_atomic_call(self, func_name, args):
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
        if not self.buffer_atomic_supports_scalar_type(operation, mapped_target_type):
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
        if self.function_has_explicit_stage_entry_attribute(func):
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

"""CrossGL-to-HLSL code generator."""

import re

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
    ExpressionStatementNode,
    ForInNode,
    ForNode,
    FunctionCallNode,
    IdentifierNode,
    IfNode,
    LoopNode,
    MatchNode,
    MemberAccessNode,
    MeshOpNode,
    PointerReinterpretNode,
    PointerType,
    RayQueryOpNode,
    RayTracingOpNode,
    ReferenceType,
    ReturnNode,
    SwitchNode,
    TernaryOpNode,
    UnaryOpNode,
    VariableNode,
    WaveOpNode,
    WhileNode,
)
from ..cooperative_matrix import get_cooperative_matrix_fragment_mapping
from ..refusals import DirectXRefusal
from ..semantics.hlsl_ray_payloads import PAYLOAD_ACCESS_ATTRIBUTES
from ..semantics.unsized_arrays import UnsizedArrayUnsupportedError
from ..unhandled import UNHANDLED
from . import hlsl_private_pointer_call_intervals
from .array_utils import (
    _UnsignedLiteralInt,
    format_c_style_array_declaration,
    get_array_size_from_node,
)
from .assignment_values import assignment_value_expression
from .buffer_length import HlslBufferLength
from .buffer_member_layout import MEMBER_LAYOUT_QUALIFIERS
from .call_effects import builtin_argument_write_indices
from .comma_expressions import (
    c_family_comma_expression,
    comma_result_operand,
    is_comma_expression,
    split_comma_initializer,
)
from .constant_guards import (
    constant_bool_locals,
    guard_bool_value,
)
from .counter_loops import any_start_counter_loops
from .directx_aggregate_initializers import DirectXAggregateInitializerMixin
from .directx_atomic_resources import DirectXAtomicResourceMixin
from .directx_call_expressions import DirectXCallExpressionMixin
from .directx_compile_time_globals import DirectXCompileTimeGlobalMixin
from .directx_compute_builtins import DirectXComputeBuiltinMixin
from .directx_cooperative_matrix import DirectXCooperativeMatrixMixin
from .directx_entry_functions import DirectXEntryFunctionMixin
from .directx_expression_forms import DirectXExpressionFormMixin
from .directx_function_emission import HLSLFunctionEmissionMixin
from .directx_function_parameters import HLSLFunctionParameterMixin
from .directx_function_state import (
    HLSL_FUNCTION_BODY_STATE_ATTRIBUTES,
    HLSL_FUNCTION_STATE_ATTRIBUTES,
    HLSLFunctionParameterContext,
    restore_hlsl_function_state,
    save_hlsl_function_state,
)
from .directx_global_declarations import DirectXGlobalDeclarationMixin
from .directx_glsl_blocks import DirectXGlslBlockMixin
from .directx_image_arguments import DirectXImageArgumentMixin
from .directx_integer_arithmetic import DirectXIntegerArithmeticMixin
from .directx_mesh_dispatch import DirectXMeshDispatchMixin
from .directx_mesh_outputs import DirectXMeshOutputMixin
from .directx_numeric_conversions import DirectXNumericConversionMixin
from .directx_pointer_arguments import DirectXPointerArgumentMixin
from .directx_pointer_assignments import DirectXPointerAssignmentMixin
from .directx_pointer_parameters import DirectXPointerParameterMixin
from .directx_pointer_storage_views import DirectXPointerStorageViewMixin
from .directx_pointer_variants import DirectXPointerVariantMixin
from .directx_precise_helpers import DirectXPreciseHelperMixin
from .directx_private_word_views import DirectXPrivateWordViewMixin
from .directx_program_emission import DirectXProgramEmissionMixin
from .directx_program_emission_state import DirectXGlobalEmission
from .directx_program_options import DirectXProgramOptionMixin
from .directx_program_preparation import DirectXProgramPreparationMixin
from .directx_ray_tracing import DirectXRayTracingMixin
from .directx_register_bindings import DirectXRegisterBindingMixin
from .directx_resource_arrays import DirectXResourceArrayMixin
from .directx_resource_globals import DirectXResourceGlobalMixin
from .directx_result_types import HLSLResultTypeMixin
from .directx_stage_features import DirectXStageFeatureMixin
from .directx_stage_parameters import DirectXStageParameterMixin
from .directx_stage_semantics import DirectXStageSemanticMixin
from .directx_statements import DirectXStatementMixin
from .directx_struct_members import DirectXStructMemberMixin
from .directx_subgroup_control_flow import DirectXSubgroupControlFlowMixin
from .directx_subgroup_helpers import DirectXSubgroupHelperMixin
from .directx_tables import DirectXTableMixin
from .directx_texture_operations import DirectXTextureOperationMixin
from .directx_typed_buffer_atomics import DirectXTypedBufferAtomicMixin
from .directx_values import (  # noqa: F401  (also imported from this module)
    HLSL_NULL_WORKGROUP_POINTER_CONTRACT,
    HLSL_PRIVATE_POINTER_BASE_MAX,
    HLSL_PRIVATE_POINTER_MAX_REBASE_ITERATIONS,
    HLSL_PRIVATE_POINTER_MAX_REBASE_SIMULATION_STEPS,
    HLSL_PRIVATE_POINTER_OFFSET_KEY,
    HLSL_RAY_SYSTEM_VALUE_TYPES,
    DirectXAggregateConditionalError,
    DirectXAggregateInitializerError,
    DirectXAtomicFenceLoweringError,
    DirectXBFloat16UnsupportedError,
    DirectXBooleanCompoundAssignmentError,
    DirectXBooleanOrderedIntrinsicError,
    DirectXCompileTimeGlobalError,
    DirectXComplexCompoundAssignmentError,
    DirectXContextualConversionError,
    DirectXCooperativeMatrixUnsupportedError,
    DirectXCopySignError,
    DirectXEntryPointSelectionError,
    DirectXForInIterableError,
    DirectXInverseHyperbolicError,
    DirectXMappedOverloadError,
    DirectXMeshOutputCountsError,
    DirectXMeshStageProfileError,
    DirectXMinimumPrecisionIntegerArithmeticError,
    DirectXNative16BitArithmeticError,
    DirectXNative16BitUnsupportedError,
    DirectXPrivatePointerParameterError,
    DirectXRayTracingStageProfileError,
    DirectXRecursiveFunctionError,
    DirectXResourcePointerArrayError,
    DirectXResourcePointerParameterError,
    DirectXSemanticArraySizeError,
    DirectXSignBitError,
    DirectXSoftwareSubgroupError,
    DirectXSpecializationConstantError,
    DirectXStructConversionError,
    DirectXTextureOffsetError,
    DirectXTrailingZeroBuiltinError,
    DirectXUnionLayoutError,
    DirectXUnresolvedSourceTypeError,
    DirectXWaveSizeProfileError,
    DirectXWorkgroupPointerError,
    DirectXWorkgroupSizeError,
    _directx_cooperative_matrix_contract_type,
    _directx_cooperative_matrix_detail_value,
    _directx_cooperative_matrix_fragment_contract,
    _directx_cooperative_matrix_fragment_contract_message,
    _DirectXResourceRegisterAllocator,
)
from .directx_wave_intrinsics import DirectXWaveIntrinsicMixin
from .enum_utils import (
    infer_enum_constructor_type,
)
from .generic_function_utils import (
    generic_function_parameters,
)
from .generic_struct_utils import (
    infer_struct_constructor_type,
)
from .hlsl_array_value_arguments import HlslArrayValueArguments
from .hlsl_atomic_resources import HLSLAtomicResources
from .hlsl_bfloat16_vectors import HLSLBfloat16Vectors
from .hlsl_buffer_block_atomic_values import (
    generate_hlsl_buffer_block_atomic_value_assignment,
    generate_hlsl_buffer_block_interlocked_statement,
)
from .hlsl_byte_address_atomics import (
    hlsl_byte_address_atomic_statement,
)
from .hlsl_call_argument_order import hlsl_ordered_call_expression
from .hlsl_exact_float_builtins import (
    hlsl_exact_float_expression,
)
from .hlsl_extension_builtins import (
    HLSL_GLSL_SYSTEM_VALUES,
    hlsl_stage_builtin_read,
)
from .hlsl_function_body_scans import HlslFunctionBodyScans
from .hlsl_glsl_block_registers import HlslGlslBlockRegisters
from .hlsl_half_storage import HLSLHalfStorage
from .hlsl_host_workgroup_size import hlsl_entry_local_size
from .hlsl_matrix_orientation import HLSLMatrixOrientation
from .hlsl_narrow_integers import HLSLNarrowIntegerMixin
from .hlsl_precise_functions import HLSLPreciseFunctionMixin
from .hlsl_ray_queries import hlsl_ray_query_expression
from .hlsl_resource_access_variants import HLSLResourceAccessVariantsMixin
from .hlsl_tall_matrix_rows import HlslTallMatrixRows
from .hlsl_texel_buffers import HLSL_TEXEL_BUFFER_TYPES, HLSLTexelBuffers
from .hlsl_texture_residency import (
    hlsl_texture_residency_call,
    hlsl_texture_resident_call,
)
from .hlsl_unavailable_builtins import hlsl_unavailable_builtin
from .hlsl_wave_extended import (
    hlsl_wave_extension_call,
)
from .image_access_contracts import (
    is_image_atomic_operation,
    is_projected_texture_operation,
    is_resource_samples_query_operation,
    is_resource_size_query_operation,
    is_texel_fetch_basic_operation,
    is_texel_fetch_offset_operation,
    is_texture_compare_operation,
    is_texture_gather_compare_operation,
    is_texture_gather_operation,
    is_texture_query_levels_operation,
    is_texture_query_lod_operation,
    is_texture_sample_basic_operation,
    is_texture_sample_grad_operation,
    is_texture_sample_lod_operation,
    is_texture_sample_offset_operation,
    is_texture_sample_operation,
    record_explicit_image_metadata,
    unsupported_multisample_texel_fetch_offset_expression,
)
from .lane_symmetry import SYMMETRIC, UNIFORM
from .loop_induction import (
    init_less_counter_loops,
    loop_constant_bindings,
    metadata_loop_bindings,
    modular_stride_loops,
)
from .lowering_refusals import LoweringRefusalMixin
from .match_utils import (
    generate_match_expression_assignment,
    infer_match_expression_result_type,
)
from .metadata_call_bounds import (
    call_parameter_bounds,
    inherited_metadata_bindings,
    join_parameter_bounds,
)
from .pointer_reinterpret import PointerReinterpretationError, scalar_storage_layout
from .private_array_rows import aliased_row_view
from .private_member_views import hlsl_aliased_scalar_view
from .private_pointer_alias_views import (
    hlsl_aliased_writable_view,
    hlsl_writable_pointer_parameters,
)
from .software_quad_reductions import SOFTWARE_QUAD_REDUCTIONS
from .standard_constant_types import standard_constant_type
from .storage_image_texels import (
    TEXEL_COMPONENTS,
    reshaped_texel_expression,
)
from .storage_parameter_texels import StorageParameterTexels
from .storage_word_views import reinterpret_source_layout
from .subgroup_loop_carried import (
    carried_fixpoint,
    follow_carried_write,
    leaves_trip_early,
    loop_carried_candidates,
    loop_carried_names,
)
from .subgroup_reassignments import follow_member_write, follow_offset, followed_offsets
from .texture_lod_clamp import (
    hlsl_texture_lod_clamp_call,
    is_texture_lod_clamp_operation,
)
from .texture_residency import (
    TEXTURE_RESIDENCY_OPERATION,
    TEXTURE_RESIDENT_OPERATION,
)
from .uniform_returns import UniformReturnAnalysis, storage_load_call


class HLSLCodeGen(
    DirectXTableMixin,
    DirectXProgramOptionMixin,
    DirectXSubgroupControlFlowMixin,
    DirectXSubgroupHelperMixin,
    DirectXPreciseHelperMixin,
    DirectXStructMemberMixin,
    DirectXCompileTimeGlobalMixin,
    DirectXNumericConversionMixin,
    DirectXIntegerArithmeticMixin,
    DirectXPrivateWordViewMixin,
    DirectXPointerParameterMixin,
    DirectXPointerArgumentMixin,
    DirectXPointerStorageViewMixin,
    DirectXPointerAssignmentMixin,
    DirectXStatementMixin,
    DirectXCooperativeMatrixMixin,
    DirectXComputeBuiltinMixin,
    DirectXWaveIntrinsicMixin,
    DirectXResourceGlobalMixin,
    DirectXStageParameterMixin,
    DirectXRayTracingMixin,
    DirectXMeshOutputMixin,
    DirectXMeshDispatchMixin,
    DirectXStageSemanticMixin,
    DirectXEntryFunctionMixin,
    DirectXPointerVariantMixin,
    DirectXResourceArrayMixin,
    DirectXRegisterBindingMixin,
    DirectXImageArgumentMixin,
    DirectXTextureOperationMixin,
    DirectXTypedBufferAtomicMixin,
    DirectXAggregateInitializerMixin,
    DirectXAtomicResourceMixin,
    DirectXGlslBlockMixin,
    DirectXStageFeatureMixin,
    HlslGlslBlockRegisters,
    HLSLNarrowIntegerMixin,
    HLSLPreciseFunctionMixin,
    HLSLResourceAccessVariantsMixin,
    LoweringRefusalMixin,
    HLSLBfloat16Vectors,
    DirectXExpressionFormMixin,
    DirectXCallExpressionMixin,
    DirectXProgramPreparationMixin,
    DirectXProgramEmissionMixin,
    DirectXGlobalDeclarationMixin,
    HLSLFunctionParameterMixin,
    HLSLFunctionEmissionMixin,
    HLSLResultTypeMixin,
):
    """Emit HLSL source from the shared CrossGL translator AST."""

    def __init__(
        self,
        target_profile=None,
        *,
        cooperative_matrix_software_lowering=False,
        cooperative_matrix_mma_profile=None,
        relative_wave_shuffle_out_of_range="undefined",
        software_subgroup_width=None,
        widen_native_float16=False,
        groupshared_bool_storage="bool",
    ):
        """Initialize DirectX type maps and per-generation resource state."""
        self.target_profile = self.normalize_directx_target_profile(target_profile)
        self.metadata_preconditions = []
        self.immutable_storage = []
        self.hlsl_immutable_storage_names = frozenset()
        self.set_cooperative_matrix_software_lowering(
            cooperative_matrix_software_lowering
        )
        self.set_cooperative_matrix_mma_profile(cooperative_matrix_mma_profile)
        self.set_relative_wave_shuffle_out_of_range(relative_wave_shuffle_out_of_range)
        self.set_software_subgroup_width(software_subgroup_width)
        self.set_widen_native_float16(widen_native_float16)
        self.set_groupshared_bool_storage(groupshared_bool_storage)
        self.directx_cooperative_matrix_lowerings = {}
        self.required_hlsl_cooperative_matrix_mma = False
        self.required_hlsl_cooperative_matrix_fragment_offset = False
        self.required_hlsl_cooperative_matrix_diagonals = {}
        self.texture_variables = set()
        self.sampler_variables = set()
        self.global_mixed_sampler_names = set()
        self.global_mixed_sampler_aliases = {}
        self.current_sampler_parameters = set()
        self.texture_variable_types = {}
        self.current_texture_parameters = {}
        self.image_variable_accesses = {}
        self.current_image_access_parameters = {}
        self.image_variable_formats = {}
        self.current_image_format_parameters = {}
        self.current_implicit_texture_samplers = {}
        self.current_implicit_texture_regular_samplers = {}
        self.current_implicit_texture_query_lod_samplers = {}
        self.global_implicit_texture_regular_samplers = {}
        self.global_implicit_texture_query_lod_samplers = {}
        self.implicit_member_texture_samplers = {}
        self.required_texture_query_helpers = set()
        self.hlsl_buffer_length = HlslBufferLength()
        self.hlsl_array_value_arguments = HlslArrayValueArguments(self)
        self.hlsl_texel_buffers = HLSLTexelBuffers(self)
        self.hlsl_tall_matrix_rows = HlslTallMatrixRows(self)
        self.hlsl_atomic_resources = HLSLAtomicResources(self)
        self.storage_parameter_texels = StorageParameterTexels(self.resource_base_type)
        self.hlsl_image_load_swizzle_extent = None
        self.hlsl_matrix_orientation = HLSLMatrixOrientation(self)
        self.required_image_atomic_helpers = set()
        self.required_byteaddress_atomic_helpers = set()
        self.hlsl_float_atomic_helpers = {}
        self.hlsl_expected_compare_helpers = {}
        self.hlsl_float_atomic_reserved_names = set()
        self.required_glsl_buffer_aggregate_load_helpers = {}
        self.comparison_sampler_parameters = {}
        self.regular_sampler_parameters = {}
        self.implicit_texture_sampler_parameters = {}
        self.function_parameter_names = {}
        self.function_parameter_types = {}
        self.hlsl_function_name_aliases = {}
        self.hlsl_function_overloads_by_name = {}
        self.hlsl_function_target_names = {}
        self.hlsl_function_target_names_by_signature = {}
        self.hlsl_mapped_overload_names = set()
        self.function_stage_parameter_dependencies = {}
        self.function_hlsl_fragment_builtin_dependencies = {}
        self.function_hlsl_fragment_builtin_parameter_names = {}
        self.function_return_types = {}
        self.function_image_access_requirements = {}
        self.hlsl_pixel_only_feedback_function_names = {}
        self.hlsl_interpolation_function_names = {}
        self.hlsl_synchronization_function_names = {}
        self.hlsl_derivative_function_names = {}
        self.unsupported_glsl_buffer_block_functions = {}
        self.unsupported_glsl_buffer_block_struct_names = set()
        self.required_hlsl_fragment_shading_rate_helper = False
        self.resource_array_size_hints = {}
        self.function_resource_array_size_hints = {}
        self.function_private_pointer_array_size_hints = {}
        self.function_private_pointer_required_spans = {}
        self.function_private_pointer_parameters = {}
        self.function_private_pointer_parameter_indices = {}
        self.function_private_pointer_backing_variants = {}
        self.function_private_pointer_local_array_extents = {}
        self.function_private_pointer_local_backing_types = {}
        self.current_hlsl_private_pointer_condition_interval_names = set()
        self.function_private_pointer_view_calls = {}
        self.function_private_pointer_base_names = {}
        self.function_private_pointer_full_span_parameters = {}
        self.function_private_pointer_scalar_parameters = {}
        self.hlsl_private_pointer_fixed_array_parameters = set()
        self.hlsl_private_pointer_word_views = {}
        self.hlsl_resource_pointer_postincrement_helpers = {}
        self.hlsl_private_pointer_reinterpret_word_views = {}
        self.hlsl_private_pointer_zero_index_access_ids = set()
        self.hlsl_private_pointer_rebase_ids = set()
        self.hlsl_private_pointer_nonzero_rebase_ids = set()
        self.hlsl_private_pointer_rejected_rebases = {}
        self.hlsl_private_pointer_parameter_seeds = {}
        self.current_hlsl_private_pointer_variant = {}
        self.current_hlsl_private_pointer_base_names = {}
        self.function_hlsl_workgroup_pointer_parameters = {}
        self.function_hlsl_workgroup_pointer_parameter_indices = {}
        self.function_hlsl_workgroup_pointer_backing_variants = {}
        self.function_hlsl_workgroup_pointer_base_names = {}
        self.function_hlsl_resource_pointer_parameters = {}
        self.function_hlsl_resource_pointer_parameter_indices = {}
        self.function_hlsl_resource_pointer_array_parameters = {}
        self.function_hlsl_resource_pointer_array_parameter_indices = {}
        self.function_hlsl_resource_pointer_array_mutated_parameters = {}
        self.hlsl_resource_pointer_analysis_call_source_types = {}
        self.hlsl_view_storage_array_parameters = {}
        self.hlsl_omitted_resource_pointer_parameter_keys = set()
        self.hlsl_omitted_function_names = set()
        self.current_hlsl_workgroup_pointer_variant = {}
        self.current_hlsl_workgroup_pointer_base_names = {}
        self.hlsl_global_groupshared_pointer_aliases = {}
        self.hlsl_groupshared_pointer_bindings_by_declaration = {}
        self.directx_resource_register_overrides = {}
        self.directx_resource_register_allocator = None
        self.literal_int_constants = {}
        self.literal_bool_constants = {}
        self.hlsl_texture_offset_global_constants = {}
        self.hlsl_texture_offset_parameter_constants = {}
        self.current_hlsl_texture_offset_constants = None
        self.directx_specialization_constant_records = []
        self.directx_specialization_constant_global_ids = set()
        self.directx_specialization_constant_types = {}
        self.directx_compile_time_global_node_ids = set()
        self.directx_omitted_compile_time_global_node_ids = set()
        self.standard_math_constant_shadow_names = set()
        self.current_identifier_aliases = {}
        self.current_identifier_reserved_names = set()
        self.current_function_name = None
        self.current_function_return_type = None
        self.current_hlsl_precise_return_type = None
        self.current_hlsl_parameter_epilogue = ()
        self.current_hlsl_tail_recursive_call_ids = set()
        self.current_hlsl_tail_recursive_function = None
        self.current_expression_expected_type = None
        self.current_hlsl_visible_int_constants = None
        self.allow_hlsl_byteaddress_interlocked_member_expression = False
        self.current_generic_function_substitutions = {}
        self.local_variable_types = {}
        self.local_variable_source_types = {}
        self.global_variable_types = {}
        self.hlsl_struct_buffer_resource_types = {}
        self.hlsl_promoted_entry_resource_parameter_names = {}
        self.hlsl_promoted_entry_constant_members = {}
        self.current_hlsl_resource_pointer_offsets = {}
        self.current_hlsl_resource_pointer_aliases = {}
        self.current_hlsl_bounded_loop_indices = {}
        self.current_hlsl_parameter_resource_types = {}
        self.struct_member_types = {}
        self.structs_by_name = {}
        self.hlsl_union_layouts = {}
        self.hlsl_fixed_array_return_structs = {}
        self.hlsl_fixed_array_return_struct_nodes = {}
        self.hlsl_fixed_array_return_struct_metadata = {}
        self.required_hlsl_fixed_array_read_helpers = set()
        self.hlsl_struct_constructor_names = {}
        self.hlsl_struct_constructor_reserved_names = set()
        self.required_hlsl_complex64_helpers = set()
        self.required_hlsl_bfloat16_helpers = set()
        self.hlsl_bfloat16_storage_resource_source_types = {}
        self.hlsl_half_storage = HLSLHalfStorage(self)
        self.requires_hlsl_bfloat16_storage = False
        self.requires_hlsl_native_16_bit_types = False
        self.required_hlsl_explicit_bitcast_helpers = set()
        self.required_hlsl_trailing_zero_helpers = set()
        self.hlsl_trailing_zero_helper_names = {}
        self.hlsl_wide_integer_compound_helpers = {}
        self.hlsl_wide_integer_compound_indices = None
        self.hlsl_wide_integer_reserved_names = set()
        self.required_hlsl_inverse_hyperbolic_helpers = set()
        self.hlsl_inverse_hyperbolic_helper_names = {}
        self.required_hlsl_precise_hyperbolic_helpers = set()
        self.hlsl_precise_hyperbolic_helper_names = {}
        self.required_hlsl_precise_difference_helpers = set()
        self.hlsl_precise_difference_helper_names = {}
        self.required_hlsl_atan2_helpers = set()
        self.hlsl_atan2_helper_names = {}
        self.required_hlsl_half_helpers = set()
        self.hlsl_half_helper_names = {}
        self.required_hlsl_unsigned_float_helpers = set()
        self.hlsl_unsigned_float_helper_names = {}
        self.required_hlsl_wide_integer_float_helpers = set()
        self.hlsl_wide_integer_float_helper_names = {}
        self.required_hlsl_half_selection_helpers = set()
        self.hlsl_half_selection_helper_names = {}
        self.required_hlsl_wave_shuffle_and_fill_up_types = set()
        self.hlsl_wave_extension_helpers = {}
        self.hlsl_pack_helpers = {}
        self.hlsl_exact_float_helpers = {}
        self.hlsl_byte_address_atomic_helpers = {}
        self.hlsl_bitfield_helpers = {}
        self.required_hlsl_defined_relative_wave_shuffle_helpers = set()
        self.hlsl_defined_relative_wave_shuffle_helper_names = {}
        self.required_hlsl_software_subgroup_helpers = set()
        self.hlsl_software_subgroup_helper_names = {}
        self.hlsl_software_subgroup_workgroup_size = None
        self.hlsl_software_subgroup_invocation_count = None
        self.hlsl_software_subgroup_function_names = set()
        self.hlsl_software_subgroup_invocation_expressions = {}
        self.hlsl_software_subgroup_invocation_variable = None
        self.required_hlsl_physical_subgroup_id_helper = False
        self.hlsl_physical_subgroup_id_helper_names = {}
        self.hlsl_lowered_struct_resource_members = {}
        self.generic_struct_definitions = {}
        self.generic_struct_specializations = {}
        self.generic_function_definitions = {}
        self.generic_function_specializations = {}
        self.generic_function_specialized_names = {}
        self.current_generic_function_substitutions = {}
        self.current_hlsl_available_functions = {}
        self.hlsl_function_body_scans = HlslFunctionBodyScans(self)
        self.hlsl_program_functions = None
        self.current_hlsl_hull_output_control_points = None
        self.current_hlsl_hull_output_element_type = None
        self.current_hlsl_hull_domain = None
        self.current_hlsl_mesh_payload_types = set()
        self.current_hlsl_dispatch_mesh_payload_types = set()
        self.current_hlsl_has_amplification_stage = False
        self.hlsl_hoisted_groupshared_declarations = []
        self.hlsl_hoisted_groupshared_aliases_by_declaration = {}
        self.hlsl_hoisted_groupshared_constants_by_declaration = {}
        self.hlsl_hoisted_groupshared_scope_names_by_declaration = {}
        self.hlsl_hoisted_groupshared_function_names_by_declaration = {}
        self.hlsl_hoisted_groupshared_declaration_ids = set()
        self.hlsl_compute_num_workgroups_uniform = None
        self.vertex_entry_input_struct_names = set()
        self.vertex_entry_output_struct_names = set()
        self.fragment_entry_input_struct_names = set()
        self.fragment_entry_output_struct_names = set()
        self.geometry_entry_vertex_struct_names = set()
        self.hlsl_mesh_output_struct_offsets = {}
        self.hlsl_temp_variable_index = 0
        self.glsl_buffer_block_struct_names = set()
        self.lowered_glsl_buffer_blocks = {}
        self.unsupported_glsl_buffer_block_variables = set()
        self.unsupported_glsl_buffer_block_variable_types = {}
        self.current_glsl_buffer_block_parameters = {}
        self.current_unsupported_glsl_buffer_block_parameters = set()
        self.current_unsupported_glsl_buffer_block_local_variables = set()
        self.current_glsl_buffer_block_parameter_failures = {}
        self.current_glsl_buffer_block_parameter_struct_failures = {}
        self.lowered_glsl_buffer_block_struct_names = set()
        self.glsl_buffer_block_lowering_failures = {}
        self.glsl_buffer_block_struct_lowering_failures = {}
        self.comparison_sampler_struct_members = set()
        self.regular_sampler_struct_members = set()
        self.type_mapping = {
            "void": "void",
            "vec2": "float2",
            "vec3": "float3",
            "vec4": "float4",
            "mat2": "float2x2",
            "mat3": "float3x3",
            "mat4": "float4x4",
            "mat2x2": "float2x2",
            "mat2x3": "float3x2",
            "mat2x4": "float4x2",
            "mat3x2": "float2x3",
            "mat3x3": "float3x3",
            "mat3x4": "float4x3",
            "mat4x2": "float2x4",
            "mat4x3": "float3x4",
            "mat4x4": "float4x4",
            "int": "int",
            "ivec2": "int2",
            "ivec3": "int3",
            "ivec4": "int4",
            "uint": "uint",
            "uvec2": "uint2",
            "uvec3": "uint3",
            "uvec4": "uint4",
            "bool": "bool",
            "bvec2": "bool2",
            "bvec3": "bool3",
            "bvec4": "bool4",
            "f16": "half",
            "f32": "float",
            "f64": "double",
            "float": "float",
            "packed_float2": "float2",
            "packed_float3": "float3",
            "packed_float4": "float4",
            "simd_float2": "float2",
            "simd_float3": "float3",
            "simd_float4": "float4",
            "half": "half",
            "half2": "half2",
            "half3": "half3",
            "half4": "half4",
            "packed_half2": "half2",
            "packed_half3": "half3",
            "packed_half4": "half4",
            # Source bfloat values are carried in the low 16 bits of a uint.
            # Numeric conversion and storage use dedicated lowering paths below.
            "bfloat": "uint",
            "bfloat16": "uint",
            "bfloat16_t": "uint",
            "float16": "half",
            "float16_t": "half",
            "f16vec2": "half2",
            "f16vec3": "half3",
            "f16vec4": "half4",
            "vec2<f16>": "half2",
            "vec3<f16>": "half3",
            "vec4<f16>": "half4",
            "vec2<f32>": "float2",
            "vec3<f32>": "float3",
            "vec4<f32>": "float4",
            "vec2<f64>": "double2",
            "vec3<f64>": "double3",
            "vec4<f64>": "double4",
            "double": "double",
            "i8": "int",
            "u8": "uint",
            "i16": "min16int",
            "u16": "min16uint",
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
            "short": "min16int",
            "signed short": "min16int",
            "ushort": "min16uint",
            "unsigned short": "min16uint",
            "int16": "min16int",
            "int16_t": "min16int",
            "int32_t": "int",
            "int64": "int64_t",
            "int64_t": "int64_t",
            "long": "int64_t",
            "signed long": "int64_t",
            "ptrdiff_t": "int64_t",
            "uint16": "min16uint",
            "uint16_t": "min16uint",
            "uint32_t": "uint",
            "uint64": "uint64_t",
            "uint64_t": "uint64_t",
            "ulong": "uint64_t",
            "unsigned long": "uint64_t",
            "size_t": "uint64_t",
            "thread_scope": "uint",
            "metal::thread_scope": "uint",
            "char2": "int2",
            "char3": "int3",
            "char4": "int4",
            "packed_int2": "int2",
            "packed_int3": "int3",
            "packed_int4": "int4",
            "simd_int2": "int2",
            "simd_int3": "int3",
            "simd_int4": "int4",
            "uchar2": "uint2",
            "uchar3": "uint3",
            "uchar4": "uint4",
            "packed_uint2": "uint2",
            "packed_uint3": "uint3",
            "packed_uint4": "uint4",
            "simd_uint2": "uint2",
            "simd_uint3": "uint3",
            "simd_uint4": "uint4",
            "i8vec2": "int2",
            "i8vec3": "int3",
            "i8vec4": "int4",
            "vec2<i8>": "int2",
            "vec3<i8>": "int3",
            "vec4<i8>": "int4",
            "u8vec2": "uint2",
            "u8vec3": "uint3",
            "u8vec4": "uint4",
            "vec2<u8>": "uint2",
            "vec3<u8>": "uint3",
            "vec4<u8>": "uint4",
            "short2": "min16int2",
            "short3": "min16int3",
            "short4": "min16int4",
            "ushort2": "min16uint2",
            "ushort3": "min16uint3",
            "ushort4": "min16uint4",
            "i16vec2": "min16int2",
            "i16vec3": "min16int3",
            "i16vec4": "min16int4",
            "vec2<i16>": "min16int2",
            "vec3<i16>": "min16int3",
            "vec4<i16>": "min16int4",
            "u16vec2": "min16uint2",
            "u16vec3": "min16uint3",
            "u16vec4": "min16uint4",
            "vec2<u16>": "min16uint2",
            "vec3<u16>": "min16uint3",
            "vec4<u16>": "min16uint4",
            "vec2<i32>": "int2",
            "vec3<i32>": "int3",
            "vec4<i32>": "int4",
            "vec2<u32>": "uint2",
            "vec3<u32>": "uint3",
            "vec4<u32>": "uint4",
            "i64vec2": "int64_t2",
            "i64vec3": "int64_t3",
            "i64vec4": "int64_t4",
            "u64vec2": "uint64_t2",
            "u64vec3": "uint64_t3",
            "u64vec4": "uint64_t4",
            "dvec2": "double2",
            "dvec3": "double3",
            "dvec4": "double4",
            "dmat2": "double2x2",
            "dmat3": "double3x3",
            "dmat4": "double4x4",
            "dmat2x2": "double2x2",
            "dmat2x3": "double3x2",
            "dmat2x4": "double4x2",
            "dmat3x2": "double2x3",
            "dmat3x3": "double3x3",
            "dmat3x4": "double4x3",
            "dmat4x2": "double2x4",
            "dmat4x3": "double3x4",
            "dmat4x4": "double4x4",
            "vec2<bool>": "bool2",
            "vec3<bool>": "bool3",
            "vec4<bool>": "bool4",
            "f16mat2": "half2x2",
            "f16mat3": "half3x3",
            "f16mat4": "half4x4",
            "f16mat2x2": "half2x2",
            "f16mat2x3": "half3x2",
            "f16mat2x4": "half4x2",
            "f16mat3x2": "half2x3",
            "f16mat3x3": "half3x3",
            "f16mat3x4": "half4x3",
            "f16mat4x2": "half2x4",
            "f16mat4x3": "half3x4",
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
            "sampler1D": "Texture1D",
            "sampler1DArray": "Texture1DArray",
            "sampler2D": "Texture2D",
            "sampler3D": "Texture3D",
            "samplerCube": "TextureCube",
            "sampler2DArray": "Texture2DArray",
            "samplerCubeArray": "TextureCubeArray",
            "isampler1D": "Texture1D<int4>",
            "isampler1DArray": "Texture1DArray<int4>",
            "isampler2D": "Texture2D<int4>",
            "isampler3D": "Texture3D<int4>",
            "isamplerCube": "TextureCube<int4>",
            "isampler2DArray": "Texture2DArray<int4>",
            "isamplerCubeArray": "TextureCubeArray<int4>",
            "usampler1D": "Texture1D<uint4>",
            "usampler1DArray": "Texture1DArray<uint4>",
            "usampler2D": "Texture2D<uint4>",
            "usampler3D": "Texture3D<uint4>",
            "usamplerCube": "TextureCube<uint4>",
            "usampler2DArray": "Texture2DArray<uint4>",
            "usamplerCubeArray": "TextureCubeArray<uint4>",
            "sampler2DMS": "Texture2DMS<float4>",
            "sampler2DMSArray": "Texture2DMSArray<float4>",
            "isampler2DMS": "Texture2DMS<int4>",
            "isampler2DMSArray": "Texture2DMSArray<int4>",
            "usampler2DMS": "Texture2DMS<uint4>",
            "usampler2DMSArray": "Texture2DMSArray<uint4>",
            "sampler1DShadow": "Texture1D",
            "sampler1DArrayShadow": "Texture1DArray",
            "sampler2DShadow": "Texture2D",
            "sampler2DArrayShadow": "Texture2DArray",
            "samplerCubeShadow": "TextureCube",
            "samplerCubeArrayShadow": "TextureCubeArray",
            "feedbackTexture2D": "FeedbackTexture2D<SAMPLER_FEEDBACK_MIN_MIP>",
            "feedbackTexture2DArray": (
                "FeedbackTexture2DArray<SAMPLER_FEEDBACK_MIN_MIP>"
            ),
            "accelerationStructure": "RaytracingAccelerationStructure",
            "accelerationStructureEXT": "RaytracingAccelerationStructure",
            "iimage1D": "RWTexture1D<int>",
            "iimage1DArray": "RWTexture1DArray<int>",
            "iimage2D": "RWTexture2D<int>",
            "iimage3D": "RWTexture3D<int>",
            "iimage2DArray": "RWTexture2DArray<int>",
            "iimage2DMS": "RWTexture2DMS<int>",
            "iimage2DMSArray": "RWTexture2DMSArray<int>",
            "uimage1D": "RWTexture1D<uint>",
            "uimage1DArray": "RWTexture1DArray<uint>",
            "uimage2D": "RWTexture2D<uint>",
            "uimage3D": "RWTexture3D<uint>",
            "uimage2DArray": "RWTexture2DArray<uint>",
            "uimage2DMS": "RWTexture2DMS<uint>",
            "uimage2DMSArray": "RWTexture2DMSArray<uint>",
            "image1D": "RWTexture1D<float4>",
            "image1DArray": "RWTexture1DArray<float4>",
            "image2D": "RWTexture2D<float4>",
            "image3D": "RWTexture3D<float4>",
            "imageCube": "RWTextureCube<float4>",
            "image2DArray": "RWTexture2DArray<float4>",
            "image2DMS": "RWTexture2DMS<float4>",
            "image2DMSArray": "RWTexture2DMSArray<float4>",
            # CrossGL's spelling of a float storage image without a format (the SPIR-V importer's); HLSL's
            # default element type for these templates is float4 as well.
            "RWTexture1D": "RWTexture1D<float4>",
            "RWTexture1DArray": "RWTexture1DArray<float4>",
            "RWTexture2D": "RWTexture2D<float4>",
            "RWTexture2DArray": "RWTexture2DArray<float4>",
            "RWTexture3D": "RWTexture3D<float4>",
            "sampler": "SamplerState",
            "comparison_sampler": "SamplerComparisonState",
            **HLSL_TEXEL_BUFFER_TYPES,
        }
        self.function_map = {
            "atomicAdd": "InterlockedAdd",
            "bitCount": "countbits",
            "bitfieldReverse": "reversebits",
            "dFdx": "ddx",
            "dFdxCoarse": "ddx_coarse",
            "dFdxFine": "ddx_fine",
            "dFdy": "ddy",
            "dFdyCoarse": "ddy_coarse",
            "dFdyFine": "ddy_fine",
            "fract": "frac",
            "floatBitsToInt": "asint",
            "floatBitsToUint": "asuint",
            "findLSB": "firstbitlow",
            "findMSB": "firstbithigh",
            "fma": "mad",
            "inverseSqrt": "rsqrt",
            "inversesqrt": "rsqrt",
            "intBitsToFloat": "asfloat",
            "mix": "lerp",
            "metal::rint": "round",
            "metal_u3a_u3arint": "round",
            "rint": "round",
            # HLSL round() rounds halfway cases to the nearest even integer.
            "roundEven": "round",
            "uintBitsToFloat": "asfloat",
        }
        # Call names that are component-wise float unary functions in HLSL: the HLSL names, which CrossGL spells
        # the same way unless function_map renames them, and the CrossGL names function_map renames to one.
        self.hlsl_componentwise_float_unary_calls = frozenset(
            {
                name
                for name in self.HLSL_COMPONENTWISE_FLOAT_UNARY_FUNCTIONS
                if name not in self.function_map
            }
            | {
                name
                for name, intrinsic in self.function_map.items()
                if intrinsic in self.HLSL_COMPONENTWISE_FLOAT_UNARY_FUNCTIONS
            }
        )

        self.semantic_map = {
            "gl_VertexID": "SV_VertexID",
            "gl_VertexIndex": "SV_VertexID",
            "gl_InstanceID": "SV_InstanceID",
            "gl_InstanceIndex": "SV_InstanceID",
            "gl_IsFrontFace": "SV_IsFrontFace",
            "gl_FrontFacing": "SV_IsFrontFace",
            "gl_PrimitiveID": "SV_PrimitiveID",
            "gl_CullPrimitiveEXT": "SV_CullPrimitive",
            "gl_ViewIndex": "SV_ViewID",
            "gl_Layer": "SV_RenderTargetArrayIndex",
            "gl_ViewportIndex": "SV_ViewportArrayIndex",
            "InstanceID": "INSTANCE_ID",
            "VertexID": "VERTEX_ID",
            "gl_Position": "SV_POSITION",
            "gl_FragCoord": "SV_Position",
            "gl_BaryCoordEXT": "SV_Barycentrics",
            # Semantic index 1, so that it does not overlap a perspective-correct SV_Barycentrics input (DXC
            # rejects two SV_Barycentrics parameters with the same index).
            "gl_BaryCoordNoPerspEXT": "SV_Barycentrics1",
            "gl_PointSize": "PSIZE",
            "gl_ClipDistance": "SV_ClipDistance",
            "gl_CullDistance": "SV_CullDistance",
            "gl_TessLevelOuter": "SV_TessFactor",
            "gl_TessLevelInner": "SV_InsideTessFactor",
            "gl_TessCoord": "SV_DomainLocation",
            "gl_FragColor": "SV_TARGET",
            "gl_FragColor0": "SV_TARGET0",
            "gl_FragColor1": "SV_TARGET1",
            "gl_FragColor2": "SV_TARGET2",
            "gl_FragColor3": "SV_TARGET3",
            "gl_FragColor4": "SV_TARGET4",
            "gl_FragColor5": "SV_TARGET5",
            "gl_FragColor6": "SV_TARGET6",
            "gl_FragColor7": "SV_TARGET7",
            "gl_FragDepth": "SV_DEPTH",
            "gl_FragStencilRefEXT": "SV_StencilRef",
            "gl_SampleMask": "SV_Coverage",
            "gl_SampleMaskIn": "SV_Coverage",
            "gl_GlobalInvocationID": "SV_DispatchThreadID",
            "gl_LocalInvocationID": "SV_GroupThreadID",
            "gl_WorkGroupID": "SV_GroupID",
            "gl_LocalInvocationIndex": "SV_GroupIndex",
            "thread_position_in_grid": "SV_DispatchThreadID",
            "thread_position_in_threadgroup": "SV_GroupThreadID",
            "threadgroup_position_in_grid": "SV_GroupID",
            "thread_index_in_threadgroup": "SV_GroupIndex",
            "mesh_DispatchMeshID": "SV_DispatchMeshID",
            "payload": "payload",
            "hit_attribute": "hit_attribute",
            "callable_data": "callable_data",
            "shader_record": "shader_record",
            **HLSL_GLSL_SYSTEM_VALUES,
        }

    def _generate_program_attempt(self, ast, target_stage, early_exits, owned=False):
        ast, target_stage = self.lower_hlsl_program_ast(
            ast, target_stage, early_exits, owned
        )
        self.reset_hlsl_program_state()
        self.reset_hlsl_stage_state(ast)
        self.collect_hlsl_specialization_constant_facts(ast)
        structs, cbuffers, global_vars, functions = (
            self.collect_hlsl_struct_and_enum_facts(ast, target_stage)
        )
        self.collect_hlsl_stage_struct_facts(
            ast, target_stage, cbuffers, functions, global_vars, structs
        )
        generic_function_specializations = self.prepare_hlsl_generic_functions(
            ast, functions
        )
        self.prepare_hlsl_function_facts(
            ast, target_stage, functions, generic_function_specializations
        )
        code, half_helper_offset, storage_helper_offset = (
            self.generate_hlsl_program_header(ast, global_vars, structs)
        )
        (
            comparison_sampler_names,
            comparison_texture_names,
            explicit_sampler_names,
            global_implicit_sampler_texture_names,
            global_implicit_regular_sampler_texture_names,
            global_implicit_query_lod_texture_names,
            implicit_member_samplers,
        ) = self.collect_hlsl_global_sampler_facts(ast, global_vars)
        self.implicit_member_texture_samplers = implicit_member_samplers

        texture_registers = {}
        sampler_registers = {}
        uav_registers = {}
        cbuffer_registers = {}
        used_resource_registers = self.new_directx_resource_register_allocator()
        self.reserve_explicit_global_resource_registers(
            global_vars, used_resource_registers, cbuffers
        )
        emission = DirectXGlobalEmission(
            code,
            comparison_sampler_names,
            comparison_texture_names,
            explicit_sampler_names,
            global_implicit_sampler_texture_names,
            global_implicit_regular_sampler_texture_names,
            global_implicit_query_lod_texture_names,
            texture_registers,
            sampler_registers,
            uav_registers,
            cbuffer_registers,
            used_resource_registers,
        )
        for i, node in enumerate(global_vars):
            self.declare_hlsl_global(emission, i, node)
        code = emission.code

        (
            code,
            stage_entry_names,
            global_functions_by_name,
            functions_code,
            function_declaration_sources,
            ordered_functions,
        ) = self.generate_hlsl_global_storage(
            ast,
            target_stage,
            cbuffer_registers,
            cbuffers,
            code,
            implicit_member_samplers,
            sampler_registers,
            texture_registers,
            uav_registers,
            used_resource_registers,
        )
        functions_code = self.generate_hlsl_program_functions(
            target_stage,
            function_declaration_sources,
            functions_code,
            ordered_functions,
            stage_entry_names,
        )

        functions_code = self.generate_hlsl_program_stages(
            ast,
            target_stage,
            function_declaration_sources,
            functions_code,
            global_functions_by_name,
            stage_entry_names,
        )
        return self.finish_hlsl_program(
            code,
            function_declaration_sources,
            functions_code,
            half_helper_offset,
            storage_helper_offset,
        )

    def declare_hlsl_global(self, emission, i, node):
        """Declare one global variable: a resource with its register and samplers, a groupshared or static variable, or a constant."""
        resource_count, vtype, array_suffix = self.resolve_hlsl_global_type(node)

        if hasattr(node, "name"):
            var_name = node.name
        elif hasattr(node, "variable_name"):
            var_name = node.variable_name
        else:
            var_name = f"var{i}"

        if id(node) in self.directx_omitted_compile_time_global_node_ids:
            return

        if self.hlsl_should_omit_metal_global_declaration(var_name, vtype):
            return

        if id(node) in self.directx_compile_time_global_node_ids:
            emission.code += self.generate_directx_compile_time_global_declaration(node)
            return

        attribute_array_size = self.hlsl_resource_array_size_expression(node, vtype)
        if not array_suffix and attribute_array_size is not None:
            array_size = self.expression_to_string(attribute_array_size)
            array_suffix = f"[{array_size}]"
            resource_count = self.resource_array_count(attribute_array_size)

        lowered_block = self.lowered_glsl_buffer_blocks.get(var_name)
        if lowered_block is not None:
            return self.declare_hlsl_lowered_buffer_block(
                emission, node, array_suffix, lowered_block, resource_count, var_name
            )

        if self.is_glsl_buffer_block_variable(node, vtype):
            emission.code += self.glsl_buffer_block_diagnostic(
                "HLSL", vtype, var_name, node
            )
            self.unsupported_glsl_buffer_block_variables.add(var_name)
            self.unsupported_glsl_buffer_block_variable_types[var_name] = (
                self.type_name_string(vtype)
            )
            emission.code += self.unsupported_glsl_buffer_block_variable_placeholder(
                "HLSL", vtype, var_name
            )
            return

        if self.is_input_attachment_type_name(vtype):
            self.refuse_lowering(
                f"input attachment declaration: '{var_name}' uses "
                f"{self.type_name_string(vtype)}; subpassInput resources "
                "require Vulkan subpass lowering"
            )
            return

        mapped_type, is_hlsl_resource_global, declaration_type = (
            self.map_hlsl_global_resource_type(
                emission, node, array_suffix, var_name, vtype
            )
        )
        declaration = format_c_style_array_declaration(declaration_type, var_name)
        register = ""
        if (
            mapped_type.startswith("Texture")
            or self.is_multisample_storage_image_resource_type(mapped_type)
            or self.is_hlsl_acceleration_structure_type(mapped_type)
        ):
            space, register = self.assign_hlsl_texture_register(
                emission, node, mapped_type, resource_count, var_name, vtype
            )
        elif self.is_hlsl_feedback_texture_type(mapped_type):
            space, register = self.assign_hlsl_feedback_texture_register(
                emission, node, mapped_type, resource_count, var_name
            )
        elif self.is_hlsl_rw_texture_type(mapped_type):
            space, register = self.assign_hlsl_rw_texture_register(
                emission, node, mapped_type, resource_count, var_name
            )
        elif self.is_hlsl_uav_buffer_type(mapped_type):
            space, register = self.assign_hlsl_uav_buffer_register(
                emission, node, resource_count, var_name
            )
        elif self.is_hlsl_constant_buffer_type(mapped_type):
            space, register = self.assign_hlsl_constant_buffer_register(
                emission, node, resource_count, var_name
            )
        elif self.is_hlsl_texture_buffer_type(mapped_type):
            space, register = self.assign_hlsl_texture_buffer_register(
                emission, node, resource_count, var_name
            )
        elif self.is_hlsl_readonly_buffer_type(mapped_type):
            space, register = self.assign_hlsl_readonly_buffer_register(
                emission, node, resource_count, var_name
            )
        elif mapped_type in ["SamplerState", "SamplerComparisonState"]:
            space, register = self.assign_hlsl_sampler_register(
                emission, node, resource_count, var_name
            )
        self.emit_hlsl_global_declaration(
            emission,
            node,
            array_suffix,
            declaration,
            is_hlsl_resource_global,
            mapped_type,
            register,
            var_name,
        )

        if mapped_type.startswith("Texture"):
            self.declare_hlsl_implicit_texture_samplers(
                emission, space, var_name, vtype
            )

    def hlsl_software_subgroup_operation_records(self, root):
        records = []
        for node in self.walk_ast(root):
            operation = None
            if (
                self.cooperative_matrix_software_lowering
                and isinstance(node, CooperativeMatrixOpNode)
                and node.operation in {"multiply_accumulate", "multiply"}
            ):
                records.append(("CooperativeMatrixMultiplyAccumulate", node))
                continue
            if isinstance(node, WaveOpNode):
                operation = node.operation
            elif isinstance(node, FunctionCallNode):
                function_name = self.function_call_name(node)
                if (
                    function_name in self.HLSL_SOFTWARE_SUBGROUP_BARRIERS
                    and function_name not in self.function_return_types
                ):
                    records.append((function_name, node))
                    continue
                metal_name = self.hlsl_metal_simd_shuffle_name(function_name)
                if metal_name is not None:
                    operation = self.HLSL_METAL_SIMD_SHUFFLE_OPERATIONS[metal_name]
                elif function_name in self.HLSL_WAVE_INTRINSIC_ARITIES:
                    operation = function_name
            if operation in self.HLSL_WAVE_INTRINSIC_ARITIES:
                records.append((operation, node))
        return records

    def hlsl_software_subgroup_uses_system_inputs(self, root):
        """Whether ``root`` reads a subgroup lane, ID, size or count input."""
        inputs = self.SOFTWARE_SUBGROUP_SYSTEM_INPUTS
        return any(
            self.semantic_from_node(node) in inputs
            or (isinstance(node, IdentifierNode) and node.name in inputs)
            for node in self.walk_ast(root)
        )

    def hlsl_software_subgroup_lane_local_matrix_nodes(self, root):
        """Matrix fills and transfers, which software lowering keeps per lane."""
        if not self.cooperative_matrix_software_lowering:
            return []
        return [
            node
            for node in self.walk_ast(root)
            if isinstance(node, CooperativeMatrixOpNode)
            and node.operation in {"fill", "diagonal", "load", "store"}
        ]

    def validate_hlsl_software_subgroup_control_flow(
        self,
        function,
        dependent_calls,
        uniform_names,
        call_mutations,
        mutable_names,
        uniform_components,
        uniform_call_arguments=None,
        *,
        entry=False,
        offset_names=frozenset(),
        metadata_bounds=None,
        unreachable_calls=None,
    ):
        body = getattr(function, "body", None)
        operation_records = self.hlsl_software_subgroup_operation_records(body)
        operation = operation_records[0][0] if operation_records else "WaveShuffleDown"
        source_location = (
            getattr(operation_records[0][1], "source_location", None)
            if operation_records
            else getattr(function, "source_location", None)
        )
        loop_constants = loop_constant_bindings(
            function,
            self.initial_literal_int_constants(function),
            mutable_names,
            self.type_name_string,
        )
        if metadata_bounds is None:
            metadata_bounds = metadata_loop_bindings(
                function,
                self.metadata_preconditions if entry else (),
                mutable_names,
                self.walk_ast,
            )
        self.hlsl_software_subgroup_modular_stride_loops.update(
            modular_stride_loops(function, self.type_name_string, self.walk_ast)
        )
        reassigned_names = self.hlsl_software_subgroup_reassigned_candidates(
            function, mutable_names, call_mutations
        )
        fixed_bools = constant_bool_locals(
            self, function, self.hlsl_initial_bool_constants(function)
        )
        # A chunk counter written only by its own initializer-free loop's
        # update follows that loop: it stays uniform when the loop is.

        def counter_writes(node):
            if id(node) in call_mutations:
                return call_mutations[id(node)]
            if isinstance(node, VariableNode):
                return self.hlsl_software_subgroup_expression_identifier_names(
                    node.initial_value
                )
            return self.hlsl_software_subgroup_expression_identifier_names(
                list(node.arguments or [])
            )

        fresh_counter_loops = (
            init_less_counter_loops(function, self.walk_ast, counter_writes)
            if entry
            else {}
        )
        # Loops entered with an unknown counter need the flow analysis to
        # prove the counter uniform where each loop starts.
        self.hlsl_software_subgroup_counter_loops = (
            any_start_counter_loops(function, self.walk_ast, counter_writes)
            if entry
            else {}
        )
        self.hlsl_software_subgroup_counter_loops.update(fresh_counter_loops)
        self.hlsl_software_subgroup_fresh_counter_loops = set(fresh_counter_loops)
        counter_names = {
            record[0] for record in self.hlsl_software_subgroup_counter_loops.values()
        }
        # Only followed locals may be records written member by member.
        record_names = set(reassigned_names)
        reassigned_names = set(reassigned_names) | counter_names | set(offset_names)
        uniform_returns = self.hlsl_software_subgroup_uniform_returns
        # Locals a uniform loop advances on every trip, as a block-mask offset.
        carried_candidates = loop_carried_candidates(
            function,
            mutable_names,
            call_mutations,
            walk=self.walk_ast,
            map_operator=self.map_operator,
            identifier_names=self.hlsl_software_subgroup_expression_identifier_names,
            value_type=uniform_returns.value_type,
            volatile=lambda node: uniform_returns._qualified(node, {"volatile"}),
            excluded=reassigned_names,
        )
        reassigned_names |= carried_candidates
        masked_if_plans = getattr(self, "hlsl_software_subgroup_masked_if_plans", {})

        def carried_trip_names(loop, names, components):
            """The facts at each trip's start of a loop that carries locals."""
            carried = loop_carried_names(
                loop, carried_candidates, self.walk_ast, self.map_operator
            )
            if not carried:
                return set(names), frozenset()
            if (
                not isinstance(loop, ForNode)
                or leaves_trip_early(loop, self.walk_ast)
                or not self.hlsl_software_subgroup_uniform_for(
                    loop,
                    names,
                    call_mutations,
                    components,
                    loop_constants.get(id(loop), {}),
                    metadata_bounds,
                )
            ):
                return set(names) - carried, frozenset()

            def follow_trip(trial):
                follow_branch(getattr(loop, "body", None), trial, components)
                update = getattr(loop, "update", None)
                if update is not None:
                    apply_reassignments(update, trial, components)

            kept = carried_fixpoint(carried, names, follow_trip)
            return (set(names) - carried) | kept, kept

        def declares_uniform(statement, names, components):
            # Facts are lexical and apply only to locals never written or
            # passed to a possibly mutating call in this function, or written
            # only where their writes can be followed.
            return (
                (
                    statement.name not in mutable_names
                    or statement.name in reassigned_names
                )
                and (
                    self.hlsl_software_subgroup_uniform_returns.value_type(
                        statement.var_type
                    )
                    or statement.name in self.hlsl_software_subgroup_resource_names
                )
                and self.hlsl_software_subgroup_uniform_expression(
                    statement.initial_value, names, components
                )
            )

        def follow_branch(statements, names, components):
            """Follow a branch every invocation takes, scoping its declarations.

            A local the branch declares from uniform values, as MLX's causal
            attention bound ``int q_max = (tid.x + 1) * BQ + params->qL_off;``,
            makes the writes computed from it uniform. Its fact ends with the
            branch, so an outer name it shadows keeps its own fact.
            """
            components = dict(components)
            outer = {}
            for child in self.hlsl_software_subgroup_statement_list(statements):
                if isinstance(child, VariableNode) and getattr(child, "name", None):
                    outer.setdefault(child.name, child.name in names)
                    names.discard(child.name)
                    components.pop(child.name, None)
                    if declares_uniform(child, names, components):
                        names.add(child.name)
                    continue
                apply_reassignments(child, names, components)
            for name, was_uniform in outer.items():
                if was_uniform:
                    names.add(name)
                else:
                    names.discard(name)

        def apply_reassignments(statement, names, components):
            """Follow the writes of ``reassigned_names`` in ``statement``."""
            written = reassigned_names & self.hlsl_software_subgroup_written_names(
                statement
            )
            if not written:
                return
            counter = self.hlsl_software_subgroup_counter_loops.get(id(statement))
            if (
                counter is not None
                and id(statement) in self.hlsl_software_subgroup_proven_counter_loops
                and written == {counter[0]}
            ):
                # Every invocation ran the uniform loop's trips.
                return
            node = statement
            if isinstance(node, ExpressionStatementNode):
                node = getattr(node, "expression", None)
            kept = follow_offset(
                node,
                offset_names,
                names,
                lambda value: self.hlsl_software_subgroup_uniform_expression(
                    value, names, components
                ),
                self.map_operator,
            )
            if kept is None:
                kept = follow_member_write(
                    node,
                    record_names,
                    names,
                    lambda value: self.hlsl_software_subgroup_uniform_expression(
                        value, names, components
                    ),
                    self.map_operator,
                )
            if kept is not None:
                if not kept:
                    names.difference_update(written)
                return
            if (
                follow_carried_write(
                    node,
                    carried_candidates,
                    names,
                    lambda value: self.hlsl_software_subgroup_uniform_expression(
                        value, names, components
                    ),
                    self.map_operator,
                )
                is not None
            ):
                return
            if isinstance(node, ForNode) and written & carried_candidates:
                # Every invocation ran the same trips, each keeping ``kept``.
                _trip, kept = carried_trip_names(node, names, components)
                dropped = written - kept
                if counter is not None and self.hlsl_software_subgroup_uniform_for(
                    node,
                    names,
                    call_mutations,
                    components,
                    loop_constants.get(id(node), {}),
                    metadata_bounds,
                ):
                    dropped.discard(counter[0])
                names.difference_update(dropped)
                return
            if isinstance(node, AssignmentNode):
                name = self.hlsl_software_subgroup_assignment_target_name(node)
                if self.map_operator(
                    getattr(node, "operator", getattr(node, "op", "="))
                ) == "=" and self.hlsl_software_subgroup_uniform_expression(
                    getattr(node, "value", getattr(node, "right", None)),
                    names,
                    components,
                ):
                    names.add(name)
                else:
                    names.discard(name)
                return
            if isinstance(node, (BlockNode, list)):
                follow_branch(node, names, components)
                return
            if isinstance(node, IfNode) and not getattr(
                node, "else_if_conditions", None
            ):
                fixed = guard_bool_value(self, node.condition, fixed_bools)
                if fixed is not None:
                    # A constant guard selects one branch for every invocation.
                    taken = node.then_branch if fixed else node.else_branch
                    follow_branch(taken, names, components)
                    return
            if (
                isinstance(node, IfNode)
                and not getattr(node, "else_if_conditions", None)
                and self.hlsl_software_subgroup_uniform_expression(
                    node.condition, names, components
                )
            ):
                # Every invocation takes the same branch, so a fact holds
                # afterwards when it holds at the end of both branches.
                results = []
                for branch in (node.then_branch, node.else_branch):
                    branch_names = set(names)
                    follow_branch(branch, branch_names, components)
                    results.append(branch_names)
                merged = set.intersection(*results)
                names.clear()
                names.update(merged)
                return
            names.difference_update(written)

        def validate_statements(statements, names, components, work_follows=True):
            # ``work_follows``: a barrier or collective may run after this
            # list completes, in program order or in a later loop iteration.
            previous = None
            sibling_statements = self.hlsl_software_subgroup_statement_list(statements)
            later_work = [False] * (len(sibling_statements) + 1)
            later_work[-1] = work_follows
            for index in range(len(sibling_statements) - 1, -1, -1):
                later_work[index] = later_work[index + 1] or (
                    self.hlsl_software_subgroup_contains_work(
                        sibling_statements[index], dependent_calls
                    )
                )
            for index, statement in enumerate(sibling_statements):
                work_after = later_work[index + 1]
                if previous is not None:
                    apply_reassignments(previous, names, components)
                previous = statement
                if isinstance(statement, VariableNode):
                    names.discard(statement.name)
                    components.pop(statement.name, None)
                    if declares_uniform(statement, names, components):
                        names.add(statement.name)
                has_return = any(
                    isinstance(node, ReturnNode) for node in self.walk_ast(statement)
                )
                has_work = self.hlsl_software_subgroup_contains_work(
                    statement, dependent_calls
                )
                if not has_return and not has_work:
                    continue
                if not has_work and not work_after:
                    # No barrier or collective of this function runs after
                    # this return, so an invocation that leaves early misses
                    # none of them. A helper's return resumes its caller.
                    continue
                if isinstance(statement, (BlockNode,)) or hasattr(
                    statement, "statements"
                ):
                    validate_statements(
                        statement, set(names), dict(components), work_after
                    )
                    continue
                if isinstance(statement, IfNode):
                    fixed = (
                        None
                        if getattr(statement, "else_if_conditions", None)
                        else guard_bool_value(self, statement.condition, fixed_bools)
                    )
                    if fixed is not None:
                        # Every invocation takes the branch the constant guard
                        # selects; the other branch never runs.
                        taken = (
                            statement.then_branch if fixed else statement.else_branch
                        )
                        skipped = (
                            statement.else_branch if fixed else statement.then_branch
                        )
                        if unreachable_calls is not None and skipped is not None:
                            # Helper calls there never run, so they contribute
                            # no argument facts to their callees.
                            unreachable_calls.update(
                                id(node)
                                for node in self.walk_ast(skipped)
                                if isinstance(node, FunctionCallNode)
                                and id(node) in dependent_calls
                            )
                        if taken is not None:
                            validate_statements(
                                taken, set(names), dict(components), work_after
                            )
                        continue
                    conditions = [statement.condition] + list(
                        getattr(statement, "else_if_conditions", []) or []
                    )
                    calls_helpers = any(
                        isinstance(node, FunctionCallNode)
                        and id(node) in dependent_calls
                        for node in self.walk_ast(statement)
                    )
                    if (
                        entry
                        and id(statement) not in masked_if_plans
                        and not calls_helpers
                        and not all(
                            self.hlsl_software_subgroup_uniform_expression(
                                condition, names, components
                            )
                            for condition in conditions
                        )
                    ):
                        # A guard that may diverge can still predicate its
                        # reductions, even inside a uniform loop. A single
                        # reduction may declare its target in the guard, as
                        # MLX's two-pass SDPA ``U score = 0; ...; score =
                        # simd_sum(score);``, when the declaration can move
                        # before the guard without capturing another name.
                        plan = self.hlsl_software_subgroup_masked_if_plan(
                            statement,
                            getattr(
                                self, "hlsl_software_subgroup_guard_names", frozenset()
                            ),
                            getattr(
                                self,
                                "hlsl_software_subgroup_guard_constant_names",
                                frozenset(),
                            ),
                            allow_multiple=True,
                            allow_hoisted_target=True,
                        )
                        if (
                            plan is not None
                            and plan.get("hoistedTargetDeclarationIndex") is not None
                            and not self.hlsl_software_subgroup_hoisted_target_is_private(
                                sibling_statements, statement, plan
                            )
                        ):
                            plan = None
                        if plan is not None:
                            masked_if_plans[id(statement)] = plan
                    if entry and id(statement) in masked_if_plans and not calls_helpers:
                        # Every invocation reaches the predicated collective;
                        # only the planned prefix and suffix stay conditional.
                        validate_statements(
                            statement.then_branch,
                            set(names),
                            dict(components),
                            work_after,
                        )
                        continue
                    if not all(
                        self.hlsl_software_subgroup_uniform_expression(
                            condition, names, components
                        )
                        for condition in conditions
                    ):
                        raise self.hlsl_software_subgroup_error(
                            "DirectX software subgroup barriers and early returns "
                            "require statically uniform control flow",
                            workgroup_size=self.hlsl_software_subgroup_workgroup_size,
                            operation=operation,
                            reason=(
                                "early-return-unproven"
                                if has_return
                                else "potentially-divergent-control-flow"
                            ),
                            source_location=getattr(
                                statement, "source_location", source_location
                            ),
                        )
                    for branch in [
                        statement.then_branch,
                        *(getattr(statement, "else_if_bodies", []) or []),
                        statement.else_branch,
                    ]:
                        validate_statements(
                            branch, set(names), dict(components), work_after
                        )
                    continue
                if isinstance(statement, ForNode):
                    strided_plan = None
                    trip_uniform = False
                    if not self.hlsl_software_subgroup_uniform_for(
                        statement,
                        names,
                        call_mutations,
                        components,
                        loop_constants.get(id(statement), {}),
                        metadata_bounds,
                    ):
                        strided_plan = (
                            self.hlsl_software_subgroup_strided_for_plan(
                                statement,
                                names,
                                components,
                                loop_constants.get(id(statement), {}),
                            )
                            if entry
                            else None
                        )
                        trip_uniform = (
                            strided_plan is None
                            and entry
                            and self.hlsl_software_subgroup_trip_uniform_for(
                                statement,
                                names,
                                call_mutations,
                                loop_constants.get(id(statement), {}),
                            )
                        )
                        if strided_plan is None and not trip_uniform:
                            raise self.hlsl_software_subgroup_error(
                                "DirectX software subgroup barriers require a "
                                "statically uniform for loop",
                                workgroup_size=self.hlsl_software_subgroup_workgroup_size,
                                operation=operation,
                                reason="potentially-divergent-control-flow",
                                source_location=getattr(
                                    statement, "source_location", source_location
                                ),
                            )
                    if strided_plan is not None:
                        # Every invocation runs the workgroup's rounds; the
                        # reduction's guard is predicated with the counter test.
                        self.hlsl_software_subgroup_strided_for_plans[id(statement)] = (
                            strided_plan
                        )
                        operation_if = self.hlsl_software_subgroup_statement_list(
                            statement.body
                        )[strided_plan["operationIfIndex"]]
                        masked_if_plans[id(operation_if)] = strided_plan["maskedIfPlan"]
                    initializer = getattr(statement, "init", None)
                    nested_names, _kept = carried_trip_names(
                        statement, names, components
                    )
                    nested_components = dict(components)
                    if isinstance(initializer, VariableNode) and initializer.name:
                        if strided_plan is None and not trip_uniform:
                            nested_names.add(initializer.name)
                        else:
                            # The counter differs between subgroups.
                            nested_names.discard(initializer.name)
                        nested_components.pop(initializer.name, None)
                    # A later iteration runs the loop's own work again.
                    validate_statements(
                        getattr(statement, "body", None),
                        nested_names,
                        nested_components,
                        work_after or has_work,
                    )
                    continue
                if isinstance(
                    statement,
                    (
                        WhileNode,
                        DoWhileNode,
                        LoopNode,
                        SwitchNode,
                        MatchNode,
                        ForInNode,
                    ),
                ):
                    raise self.hlsl_software_subgroup_error(
                        "DirectX software subgroup barriers require statically "
                        "uniform control flow",
                        workgroup_size=self.hlsl_software_subgroup_workgroup_size,
                        operation=operation,
                        reason="potentially-divergent-control-flow",
                        source_location=getattr(
                            statement, "source_location", source_location
                        ),
                    )
                if any(
                    isinstance(child, TernaryOpNode)
                    and self.hlsl_software_subgroup_contains_work(
                        child, dependent_calls
                    )
                    for child in self.walk_ast(statement)
                ):
                    raise self.hlsl_software_subgroup_error(
                        "DirectX software subgroup barriers cannot be selected by "
                        "a conditional expression",
                        workgroup_size=self.hlsl_software_subgroup_workgroup_size,
                        operation=operation,
                        reason="potentially-divergent-control-flow",
                        source_location=source_location,
                    )
                if any(
                    isinstance(child, BinaryOpNode)
                    and self.map_operator(
                        getattr(child, "op", getattr(child, "operator", None))
                    )
                    in {"&&", "||"}
                    and self.hlsl_software_subgroup_contains_work(
                        child, dependent_calls
                    )
                    for child in self.walk_ast(statement)
                ):
                    raise self.hlsl_software_subgroup_error(
                        "DirectX software subgroup barriers cannot be selected by "
                        "a short-circuit expression",
                        workgroup_size=self.hlsl_software_subgroup_workgroup_size,
                        operation=operation,
                        reason="potentially-divergent-control-flow",
                        source_location=source_location,
                    )

                if uniform_call_arguments is not None:
                    for call in self.walk_ast(statement):
                        if (
                            isinstance(call, FunctionCallNode)
                            and id(call) in dependent_calls
                        ):
                            uniform_call_arguments[id(call)] = {
                                index
                                for index, argument in enumerate(call.arguments)
                                if self.hlsl_software_subgroup_uniform_expression(
                                    argument, names, components
                                )
                            }

        validate_statements(
            body,
            set(uniform_names),
            dict(uniform_components),
            work_follows=False,
        )

    def prepare_hlsl_software_subgroup_contract(self, ast, target_stage=None):
        self.required_hlsl_software_subgroup_helpers = set()
        self.hlsl_software_subgroup_helper_names = {}
        self.hlsl_software_subgroup_reserved_names = set()
        self.hlsl_software_subgroup_workgroup_size = None
        self.hlsl_software_subgroup_invocation_count = None
        self.hlsl_software_subgroup_function_names = set()
        self.hlsl_software_subgroup_invocation_expressions = {}
        self.hlsl_software_subgroup_invocation_variable = None
        self.hlsl_software_subgroup_uniform_calls = set()
        self.hlsl_software_subgroup_resource_names = set()
        self.hlsl_software_subgroup_uniform_returns = UniformReturnAnalysis({}, set())
        self.hlsl_software_subgroup_masked_if_plans = {}
        self.hlsl_software_subgroup_strided_for_plans = {}
        self.hlsl_software_subgroup_result_overrides = {}
        self.hlsl_software_subgroup_guard_names = set()
        self.hlsl_software_subgroup_guard_constant_names = set()
        self.hlsl_software_subgroup_counter_loops = {}
        self.hlsl_software_subgroup_fresh_counter_loops = set()
        self.hlsl_software_subgroup_proven_counter_loops = set()
        self.hlsl_software_subgroup_modular_stride_loops = set()
        self.hlsl_software_subgroup_symmetry = None
        if self.software_subgroup_width is None:
            return

        entries = self.hlsl_software_subgroup_stage_entries(ast, target_stage)
        if len(entries) != 1 or entries[0][0] != "compute":
            raise self.hlsl_software_subgroup_error(
                "DirectX software subgroup lowering requires exactly one compute "
                "entry point and no other emitted stages",
                reason="entry-point-contract-invalid",
            )
        _stage_name, entry_function, execution_config = entries[0]
        raw_workgroup_size = tuple(hlsl_entry_local_size(execution_config))
        concrete_workgroup_size = []
        for value in raw_workgroup_size:
            text = str(value).strip()
            if not re.fullmatch(r"[0-9]+", text):
                concrete_workgroup_size = []
                break
            concrete_workgroup_size.append(int(text))
        concrete_workgroup_size = tuple(concrete_workgroup_size)
        invocation_count = (
            concrete_workgroup_size[0]
            * concrete_workgroup_size[1]
            * concrete_workgroup_size[2]
            if len(concrete_workgroup_size) == 3
            else 0
        )
        if not (
            len(concrete_workgroup_size) == 3
            and all(value > 0 for value in concrete_workgroup_size)
            and invocation_count <= 1024
        ):
            raise self.hlsl_software_subgroup_error(
                "DirectX software subgroup lowering requires concrete positive "
                "local dimensions and at most 1024 invocations",
                workgroup_size=raw_workgroup_size,
                reason="workgroup-size-mismatch",
                source_location=getattr(entry_function, "source_location", None),
            )
        self.hlsl_software_subgroup_workgroup_size = concrete_workgroup_size
        self.hlsl_software_subgroup_invocation_count = invocation_count

        all_records = self.hlsl_software_subgroup_operation_records(ast)
        lane_local_matrix_nodes = self.hlsl_software_subgroup_lane_local_matrix_nodes(
            ast
        )
        if lane_local_matrix_nodes and invocation_count % self.software_subgroup_width:
            raise self.hlsl_software_subgroup_error(
                "DirectX matrix fills and transfers require complete logical subgroups",
                workgroup_size=concrete_workgroup_size,
                operation="CooperativeMatrixTransfer",
                reason="incomplete-matrix-subgroup",
                source_location=getattr(
                    lane_local_matrix_nodes[0], "source_location", None
                ),
            )
        # Subgroup system inputs, matrix fills and matrix transfers without
        # collective operations need only the logical split of SV_GroupIndex
        # into subgroup and lane IDs. That split is exact under any control
        # flow, so no shuffle policy applies.
        system_inputs_only = not all_records and (
            bool(lane_local_matrix_nodes)
            or self.hlsl_software_subgroup_uses_system_inputs(ast)
        )
        if self.relative_wave_shuffle_out_of_range != "self" and not system_inputs_only:
            raise self.hlsl_software_subgroup_error(
                "DirectX software relative shuffles require the explicit "
                "calling-lane out-of-range policy",
                workgroup_size=concrete_workgroup_size,
                reason="out-of-range-policy-unproven",
                source_location=getattr(entry_function, "source_location", None),
            )
        if not system_inputs_only:
            self.prepare_hlsl_software_subgroup_masked_if_plans(entry_function)

        wave_attributes = [
            attribute
            for attribute in getattr(entry_function, "attributes", []) or []
            if self.hlsl_wave_size_attribute(attribute)
        ]
        for attribute in wave_attributes:
            arguments = list(getattr(attribute, "arguments", []) or [])
            width = (
                self.hlsl_int_literal_value(arguments[0])
                if len(arguments) == 1
                else None
            )
            if width != self.software_subgroup_width:
                raise self.hlsl_software_subgroup_error(
                    "DirectX software subgroup width conflicts with the source "
                    "WaveSize contract",
                    workgroup_size=concrete_workgroup_size,
                    reason="hardware-contract-conflict",
                    source_location=getattr(attribute, "source_location", None),
                )

        if system_inputs_only:
            self.hlsl_software_subgroup_bind_lane_local_matrix_functions(
                ast, lane_local_matrix_nodes
            )
            return
        if not all_records:
            raise self.hlsl_software_subgroup_error(
                "DirectX software subgroup lowering was requested but no subgroup "
                "operation is present",
                workgroup_size=concrete_workgroup_size,
                reason="operation-set-empty",
                source_location=getattr(entry_function, "source_location", None),
            )
        for operation, node in all_records:
            if operation == "CooperativeMatrixMultiplyAccumulate":
                if invocation_count % self.software_subgroup_width:
                    raise self.hlsl_software_subgroup_error(
                        "DirectX cooperative matrices require complete logical subgroups",
                        workgroup_size=concrete_workgroup_size,
                        operation=operation,
                        reason="incomplete-matrix-subgroup",
                        source_location=getattr(node, "source_location", None),
                    )
                continue
            if operation not in self.HLSL_SOFTWARE_SUBGROUP_OPERATIONS:
                raise self.hlsl_software_subgroup_error(
                    f"DirectX software subgroup lowering does not support "
                    f"'{operation}'",
                    workgroup_size=concrete_workgroup_size,
                    operation=operation,
                    reason="operation-unsupported",
                    source_location=getattr(node, "source_location", None),
                )

        functions = self.collect_functions(ast)
        definitions = {}
        for function in functions:
            name = getattr(function, "name", None)
            if not name:
                continue
            key = (name, self.hlsl_source_function_signature(function))
            if key not in definitions or getattr(function, "body", None) is not None:
                definitions[key] = function
        functions = list(definitions.values())
        functions_by_id = {id(function): function for function in functions}
        functions_by_name = {}
        for function in functions:
            functions_by_name.setdefault(function.name, []).append(function)
        direct_ids = {
            id(function)
            for function in functions
            if self.hlsl_software_subgroup_operation_records(
                getattr(function, "body", None)
            )
        }
        resolved_calls = {}
        unresolved_calls = []
        call_source_types = {}
        declarations = {}
        constant_nodes = list(ast.constants) + [
            node
            for node in ast.global_variables
            if isinstance(node, VariableNode)
            and set(node.qualifiers) & {"const", "constant"}
            and "volatile" not in node.qualifiers
        ]
        edges = {id(function): [] for function in functions}
        for function in functions:
            lexical = self.hlsl_function_lexical_type_bindings(function)
            call_source_types.update(lexical["call_source_types"])
            declared_names = {
                binding["name"] for binding in lexical["declarations"].values()
            }
            constant_types = {
                node.name: self.directx_specialization_constant_type(node)
                for node in constant_nodes
                if node.name not in declared_names
            }
            for binding in lexical["declarations"].values():
                binding["initializer_source_types"] = {
                    **constant_types,
                    **binding.get("initializer_source_types", {}),
                }
            declarations.update(lexical["declarations"])
            for call in self.walk_ast(getattr(function, "body", None)):
                if not isinstance(call, FunctionCallNode):
                    continue
                name = self.function_call_name(call)
                if name not in functions_by_name:
                    continue
                previous_source_types = self.local_variable_source_types
                previous_types = self.local_variable_types
                source_types = lexical["call_source_types"].get(id(call), {})
                self.local_variable_source_types = source_types
                self.local_variable_types = source_types
                try:
                    callee = self.resolve_hlsl_function_overload(
                        name, call.arguments, call_node=call
                    )
                except DirectXMappedOverloadError:
                    callee = None
                finally:
                    self.local_variable_source_types = previous_source_types
                    self.local_variable_types = previous_types
                if callee is None:
                    unresolved_calls.append((function, call))
                    continue
                key = (callee.name, self.hlsl_source_function_signature(callee))
                callee = definitions[key]
                resolved_calls[id(call)] = callee
                edges[id(function)].append((call, id(callee)))

        dependent_ids = set(direct_ids)
        changed = True
        while changed:
            changed = False
            for function_id, calls in edges.items():
                if function_id in dependent_ids:
                    continue
                if any(target in dependent_ids for _call, target in calls):
                    dependent_ids.add(function_id)
                    changed = True
        for _function, call in unresolved_calls:
            name = self.function_call_name(call)
            if any(
                id(candidate) in dependent_ids for candidate in functions_by_name[name]
            ):
                raise self.hlsl_software_subgroup_error(
                    "DirectX software subgroup lowering cannot resolve one source "
                    f"overload for helper '{name}'",
                    workgroup_size=concrete_workgroup_size,
                    reason="helper-identity-ambiguous",
                    source_location=getattr(call, "source_location", None),
                )
        if id(entry_function) not in dependent_ids:
            raise self.hlsl_software_subgroup_error(
                "DirectX software subgroup operations are not reached from the "
                "compute entry point",
                workgroup_size=concrete_workgroup_size,
                reason="helper-call-unproven",
                source_location=getattr(entry_function, "source_location", None),
            )

        # Process callers before callees, intersecting facts from every call.
        # Recursive collective graphs cannot establish a finite barrier schedule.
        call_edges = {}
        incoming = {function_id: 0 for function_id in dependent_ids}
        for function_id in functions_by_id:
            if function_id not in dependent_ids:
                continue
            call_edges[function_id] = [
                (call, target)
                for call, target in edges[function_id]
                if target in dependent_ids
            ]
            for _call, target in call_edges[function_id]:
                incoming[target] += 1
        pending = [key for key in functions_by_id if incoming.get(key) == 0]
        ordered_ids = []
        while pending:
            function_id = pending.pop(0)
            ordered_ids.append(function_id)
            for _call, target in call_edges[function_id]:
                incoming[target] -= 1
                if incoming[target] == 0:
                    pending.append(target)
        if len(ordered_ids) != len(dependent_ids):
            raise self.hlsl_software_subgroup_error(
                "DirectX software subgroup helpers cannot use recursive calls",
                workgroup_size=concrete_workgroup_size,
                reason="helper-call-recursive",
                source_location=getattr(entry_function, "source_location", None),
            )

        dependent_calls = {
            id(call) for calls in call_edges.values() for call, _target in calls
        }
        self.hlsl_software_subgroup_function_names = {
            functions_by_id[key].name for key in dependent_ids
        }
        reserved_names = self.hlsl_helper_reserved_names(functions)
        self.hlsl_software_subgroup_reserved_names = reserved_names
        variable = "__crossgl_software_subgroup_invocation"
        while variable in reserved_names:
            variable += "_"
        self.hlsl_software_subgroup_invocation_variable = variable
        for function_id in direct_ids:
            self.hlsl_software_subgroup_invocation_expressions[
                functions_by_id[function_id].name
            ] = variable
        self.hlsl_software_subgroup_bind_lane_local_matrix_functions(
            ast, lane_local_matrix_nodes
        )

        calls = [
            node for node in self.walk_ast(ast) if isinstance(node, FunctionCallNode)
        ]
        storage_loads = set()
        for node in calls:
            name = self.function_call_name(node)
            if name in functions_by_name or not isinstance(
                node.function, (str, IdentifierNode)
            ):
                continue
            if (
                (name in {"min", "max"} and len(node.arguments) == 2)
                or (name == "clamp" and len(node.arguments) == 3)
                or self.directx_compile_time_constructor_type(name) is not None
                # A bit reinterpretation, such as the binary16 storage
                # lowering's, writes nothing and keeps its operand's value.
                or (
                    len(node.arguments) == 1
                    and self.hlsl_metal_as_type_target_type(name) is not None
                )
            ):
                self.hlsl_software_subgroup_uniform_calls.add(id(node))
            elif storage_load_call(name, node.arguments):
                storage_loads.add(id(node))
        self.hlsl_software_subgroup_uniform_returns = UniformReturnAnalysis(
            resolved_calls,
            self.hlsl_software_subgroup_uniform_calls,
            self.structs_by_name,
            storage_loads,
        )
        call_mutations = {}
        for binding in declarations.values():
            declaration = binding["node"]
            if not isinstance(getattr(declaration, "var_type", None), PointerType):
                continue
            roots = self.hlsl_software_subgroup_uniform_returns.pointer_argument_write_roots(
                declaration.initial_value,
                binding.get("initializer_source_types", {}),
            )
            if roots is not None:
                call_mutations[id(declaration)] = roots
        for node in calls:
            call_name = self.function_call_name(node)
            arguments = list(getattr(node, "arguments", []) or [])
            callee = resolved_calls.get(id(node))
            builtin_writes = (
                builtin_argument_write_indices(call_name, len(arguments))
                if call_name not in functions_by_name
                and isinstance(node.function, (str, IdentifierNode))
                else None
            )
            pure_call = id(node) in self.hlsl_software_subgroup_uniform_calls
            written_roots = set()
            if (
                self.hlsl_software_subgroup_uniform_returns.dependencies(node)
                is not None
            ):
                mutations = []
            elif callee is not None:
                parameters = getattr(callee, "parameters", []) or []
                mutations = []
                for index, argument in enumerate(arguments):
                    if self.hlsl_software_subgroup_uniform_returns.read_only_resource_parameter(
                        callee, index
                    ):
                        continue
                    if index >= len(parameters):
                        mutations.append(argument)
                    elif isinstance(parameters[index].param_type, PointerType):
                        roots = self.hlsl_software_subgroup_uniform_returns.pointer_argument_write_roots(
                            argument, call_source_types.get(id(node), {})
                        )
                        if roots is None:
                            mutations.append(argument)
                        else:
                            written_roots.update(roots)
                    elif isinstance(parameters[index].param_type, ReferenceType) or set(
                        self.hlsl_parameter_qualifiers(parameters[index])
                    ) & {"out", "inout"}:
                        # A reference writes its selected lvalue, not its indices.
                        # Argument evaluation effects are visited independently.
                        root = self.hlsl_texture_offset_write_root_name(argument)
                        if root is None:
                            mutations.append(argument)
                        else:
                            written_roots.add(root)
            elif builtin_writes is not None:
                mutations = [arguments[index] for index in builtin_writes]
            elif call_name not in functions_by_name and (
                call_name in self.HLSL_WAVE_INTRINSIC_ARITIES
                or self.hlsl_metal_simd_shuffle_name(call_name) is not None
                or pure_call
            ):
                mutations = []
            else:
                mutations = arguments
            call_mutations[id(node)] = written_roots.union(
                *(
                    self.hlsl_software_subgroup_expression_identifier_names(argument)
                    for argument in mutations
                )
            )

        uniform_semantics = {
            "gl_WorkGroupSize",
            "gl_NumWorkGroups",
            "gl_WorkGroupID",
            "gl_SubgroupSize",
            "gl_NumSubgroups",
            "SV_GroupID",
        }
        if invocation_count <= self.software_subgroup_width:
            uniform_semantics.add("gl_SubgroupID")
        lane_symmetry = self.hlsl_software_subgroup_lane_symmetry_analysis(
            resolved_calls, call_mutations, invocation_count
        )
        # Lane-symmetric parameters, and whether every subgroup enters a
        # helper with the same lanes, from every reachable call.
        symmetric_parameters = {}
        symmetric_entries = {}
        uniform_parameters = {}
        # Checked metadata bounds every call passes to a helper parameter.
        parameter_bounds = {}
        # Calls in branches a constant guard never selects, and the helpers
        # only such calls reach: neither runs, so neither constrains a proof.
        unreachable_calls = set()
        called_ids = {target for edges in call_edges.values() for _c, target in edges}
        for function_id in ordered_ids:
            function = functions_by_id[function_id]
            if (
                function is not entry_function
                and function_id in called_ids
                and function_id not in uniform_parameters
            ):
                unreachable_calls.update(
                    id(call) for call, _target in call_edges[function_id]
                )
                continue
            parameters = getattr(function, "parameters", []) or []
            analysis = self.hlsl_software_subgroup_uniform_returns
            declarations = [
                *parameters,
                *(
                    node
                    for node in self.walk_ast(function.body)
                    if isinstance(node, VariableNode)
                ),
            ]
            counts = {}
            for declaration in declarations:
                counts[declaration.name] = counts.get(declaration.name, 0) + 1
            self.hlsl_software_subgroup_resource_names = {
                declaration.name
                for declaration in declarations
                if counts[declaration.name] == 1
                and analysis.resource_type(
                    getattr(
                        declaration,
                        "param_type",
                        getattr(declaration, "var_type", None),
                    ),
                    declaration,
                )
            }
            uniform_names = set(self.literal_int_constants) - {
                parameter.name for parameter in parameters
            }
            uniform_names.update(
                parameter.name
                for index, parameter in enumerate(parameters)
                if index in uniform_parameters.get(function_id, set())
                and not isinstance(parameter.param_type, ReferenceType)
                and not set(self.hlsl_parameter_qualifiers(parameter))
                & {"out", "inout"}
                and (
                    analysis.value_type(parameter.param_type)
                    or parameter.name in self.hlsl_software_subgroup_resource_names
                )
            )
            uniform_components = {}
            if function is entry_function:
                unit_dimensions = {
                    index
                    for index, size in enumerate(concrete_workgroup_size)
                    if size == 1
                }
                uniform_components = {
                    parameter.name: unit_dimensions
                    for parameter in parameters
                    if self.hlsl_canonical_semantic(self.semantic_from_node(parameter))
                    in {"SV_DispatchThreadID", "SV_GroupThreadID"}
                }
                uniform_names.update(
                    parameter.name
                    for parameter in parameters
                    if self.semantic_from_node(parameter) in uniform_semantics
                    or set(getattr(parameter, "qualifiers", []) or [])
                    & {"uniform", "constant"}
                    or (
                        parameter.name in self.hlsl_software_subgroup_resource_names
                        and (
                            analysis._qualified(parameter, {"constant"})
                            # A checked contract: no writable binding aliases it.
                            or parameter.name in self.hlsl_immutable_storage_names
                        )
                    )
                )
            # A name alone is not a uniformity proof after writes, shadowing,
            # or a call that could receive it through an output parameter.
            mutable_names = set()
            for node in self.walk_ast(getattr(function, "body", None)):
                if isinstance(node, VariableNode):
                    uniform_names.discard(node.name)
                    if isinstance(node.var_type, ReferenceType):
                        mutable_names.update(
                            self.hlsl_software_subgroup_expression_identifier_names(
                                node.initial_value
                            )
                        )
                elif isinstance(node, AssignmentNode):
                    mutable_names.add(
                        self.hlsl_software_subgroup_assignment_target_name(node)
                    )
                elif isinstance(node, UnaryOpNode) and self.map_operator(
                    getattr(node, "op", getattr(node, "operator", None))
                ) in {"++", "--", "&"}:
                    mutable_names.add(
                        self.hlsl_texture_offset_write_root_name(node.operand)
                    )
                elif isinstance(node, FunctionCallNode):
                    mutable_names.update(call_mutations[id(node)])
            # Immutable entry storage moved only by offsets outside loops keeps
            # its uniform view where the flow can follow those offsets.
            offset_names = (
                followed_offsets(
                    function,
                    {
                        parameter.name
                        for parameter in parameters
                        if parameter.name in uniform_names
                        and parameter.name in self.hlsl_software_subgroup_resource_names
                    },
                    call_mutations,
                    walk=self.walk_ast,
                    map_operator=self.map_operator,
                    identifier_names=(
                        self.hlsl_software_subgroup_expression_identifier_names
                    ),
                    write_roots=lambda target: {
                        self.hlsl_texture_offset_write_root_name(target)
                    }
                    - {None},
                )
                if function is entry_function
                else set()
            )
            uniform_names.difference_update(mutable_names - offset_names)
            uniform_components = {
                name: components
                for name, components in uniform_components.items()
                if name not in mutable_names
            }
            uniform_call_arguments = {}
            metadata_bounds = (
                metadata_loop_bindings(
                    function, self.metadata_preconditions, mutable_names, self.walk_ast
                )
                if function is entry_function
                else inherited_metadata_bindings(
                    function,
                    parameter_bounds.get(function_id),
                    mutable_names,
                    self.walk_ast,
                )
            )
            seeds = {name: UNIFORM for name in uniform_names}
            for index, parameter in enumerate(parameters):
                if parameter.name not in seeds and (
                    index in symmetric_parameters.get(function_id, set())
                    or (
                        function is entry_function
                        and self.semantic_from_node(parameter)
                        == "gl_SubgroupInvocationID"
                    )
                ):
                    seeds[parameter.name] = SYMMETRIC
            symmetry = lane_symmetry.analyze(
                function,
                seeds,
                function is entry_function or symmetric_entries.get(function_id, False),
            )
            self.hlsl_software_subgroup_symmetry = symmetry
            try:
                self.validate_hlsl_software_subgroup_control_flow(
                    function,
                    dependent_calls,
                    uniform_names,
                    call_mutations,
                    mutable_names,
                    uniform_components,
                    uniform_call_arguments,
                    entry=function is entry_function,
                    offset_names=offset_names,
                    metadata_bounds=metadata_bounds,
                    unreachable_calls=unreachable_calls,
                )
            finally:
                self.hlsl_software_subgroup_symmetry = None
            for call, target in call_edges[function_id]:
                if id(call) in unreachable_calls:
                    continue
                levels = [symmetry.level(argument) for argument in call.arguments]
                symmetric = {
                    index for index, level in enumerate(levels) if level >= SYMMETRIC
                }
                if target not in symmetric_parameters:
                    symmetric_parameters[target] = symmetric
                else:
                    symmetric_parameters[target].intersection_update(symmetric)
                symmetric_entries[target] = symmetric_entries.get(
                    target, True
                ) and symmetry.entered_symmetrically(call)
                # A uniformly reached call also passes the uniform values the
                # lane facts prove: collective results, and names the checks
                # below accept where the existing call-site proof stops, such
                # as the right operand of a uniform short-circuit condition.
                facts = uniform_call_arguments.get(id(call), set())
                if symmetry.reach(call) == UNIFORM:
                    facts = facts | {
                        index
                        for index, level in enumerate(levels)
                        if level == UNIFORM
                        and (
                            symmetry.collective_derived(call.arguments[index])
                            or self.hlsl_software_subgroup_uniform_expression(
                                call.arguments[index],
                                uniform_names,
                                uniform_components,
                            )
                        )
                    }
                if target not in uniform_parameters:
                    uniform_parameters[target] = set(facts)
                else:
                    uniform_parameters[target].intersection_update(facts)
                parameter_bounds[target] = join_parameter_bounds(
                    parameter_bounds.get(target),
                    call_parameter_bounds(
                        getattr(functions_by_id[target], "parameters", ()) or (),
                        getattr(call, "arguments", ()) or (),
                        metadata_bounds,
                        self.type_name_string,
                    ),
                )

    def generate_function(
        self,
        func,
        indent=0,
        shader_type=None,
        execution_config=None,
        entry_name=None,
        stage_output_lowering=None,
        stage_node=None,
        _private_pointer_variant=None,
        _workgroup_pointer_variant=None,
        _resource_access_variant=None,
    ):
        """Render a function or stage entry point with HLSL semantics."""
        function_name = getattr(func, "name", None)
        handled = self.generate_hlsl_function_variants(
            func,
            indent,
            shader_type,
            execution_config,
            entry_name,
            stage_output_lowering,
            stage_node,
            _private_pointer_variant,
            _workgroup_pointer_variant,
            _resource_access_variant,
            function_name,
        )
        if handled is not UNHANDLED:
            return handled

        code = ""
        code += "  " * indent

        param_list = getattr(func, "parameters", getattr(func, "params", []))
        params = []
        parameter_prologue_statements = []
        parameter_epilogue_statements = []
        sampler_parameters = set()
        texture_parameters = {}
        image_access_parameters = {}
        image_format_parameters = {}
        comparison_sampler_parameters = self.comparison_sampler_parameters.get(
            getattr(func, "name", None), set()
        )
        implicit_texture_samplers = self.implicit_texture_sampler_parameters.get(
            getattr(func, "name", None), {}
        )
        implicit_existing_comparison_samplers = {
            data["sampler_name"]
            for data in implicit_texture_samplers.values()
            if data["comparison"] and not data["synthetic"]
        }
        param_names = {getattr(param, "name", None) for param in param_list}
        emitted_param_names = set(param_names)
        saved_function_state = save_hlsl_function_state(
            self, HLSL_FUNCTION_STATE_ATTRIBUTES
        )
        saved_function_body_state = save_hlsl_function_state(
            self, HLSL_FUNCTION_BODY_STATE_ATTRIBUTES
        )
        self.reset_hlsl_function_emission_state(
            func, entry_name, stage_output_lowering, param_list
        )
        if hasattr(func, "qualifiers") and func.qualifiers:
            qualifier = func.qualifiers[0] if func.qualifiers else None
        else:
            qualifier = getattr(func, "qualifier", None)
        effective_shader_type = shader_type or qualifier
        self.current_hlsl_semantic_stage = effective_shader_type
        promoted_entry_resource_parameter_ids, default_vertex_parameter_semantics = (
            self.hlsl_entry_parameter_facts(
                func, effective_shader_type, param_list, parameter_prologue_statements
            )
        )
        signature = HLSLFunctionParameterContext(
            func,
            param_list,
            param_names,
            emitted_param_names,
            function_name,
            effective_shader_type,
            execution_config,
            promoted_entry_resource_parameter_ids,
            default_vertex_parameter_semantics,
            params,
            parameter_prologue_statements,
            parameter_epilogue_statements,
            sampler_parameters,
            texture_parameters,
            image_access_parameters,
            image_format_parameters,
            comparison_sampler_parameters,
            implicit_texture_samplers,
            implicit_existing_comparison_samplers,
        )
        for p in param_list:
            self.emit_hlsl_function_parameter(signature, p)

        params_str = ", ".join(signature.params)
        software_subgroup_entry_index, params_str = (
            self.append_hlsl_required_builtin_parameters(
                func,
                execution_config,
                effective_shader_type,
                param_list,
                params_str,
                signature,
            )
        )
        shader_map = {"vertex": "VSMain", "fragment": "PSMain", "compute": "CSMain"}
        shader_attr_map = {
            "geometry": "geometry",
            "tessellation_control": "hull",
            "tessellation_evaluation": "domain",
            "mesh": "mesh",
            "amplification": "amplification",
            "task": "amplification",
            "object": "amplification",
            "ray_generation": "raygeneration",
            "ray_intersection": "intersection",
            "ray_closest_hit": "closesthit",
            "ray_any_hit": "anyhit",
            "ray_miss": "miss",
            "ray_callable": "callable",
        }

        declaration_return_type, raw_return_type, return_type = (
            self.hlsl_function_return_types(func, stage_output_lowering)
        )
        self.validate_hlsl_local_groupshared_declarations(func)
        self.validate_hlsl_nonuniform_resource_index_calls(func)
        parameter_diagnostics = self.glsl_buffer_block_parameter_diagnostics(
            "HLSL", param_list, indent
        )
        if parameter_diagnostics:
            code += parameter_diagnostics
            code += "  " * indent
        unsupported_function_reason = self.unsupported_glsl_buffer_block_functions.get(
            getattr(func, "name", None)
        )
        if unsupported_function_reason is not None:
            code += self.unsupported_glsl_buffer_block_function_placeholder(
                "HLSL", getattr(func, "name", None), unsupported_function_reason
            )
            restore_hlsl_function_state(self, saved_function_state)
            return code

        return_semantic = self.hlsl_function_return_semantic(func)
        if effective_shader_type is not None and return_semantic is None:
            return_semantic = self.default_hlsl_stage_return_semantic(
                effective_shader_type,
                return_type,
            )
        if (
            stage_output_lowering is not None
            and stage_output_lowering.get("struct_name") is not None
        ):
            return_semantic = None
        body = getattr(func, "body", None)
        if body is not None:
            self.current_hlsl_tail_recursive_call_ids = (
                self.hlsl_tail_recursive_void_call_ids(func)
            )
            if self.current_hlsl_tail_recursive_call_ids:
                if signature.parameter_epilogue_statements:
                    target_name = self.hlsl_function_declaration_name(func)
                    raise DirectXRecursiveFunctionError(
                        "DirectX cannot lower tail recursion that requires "
                        "resource-parameter writeback after each call",
                        function_name=target_name,
                        cycle=(target_name, target_name),
                        reason="tail-resource-writeback-unsupported",
                        source_location=getattr(func, "source_location", None),
                    )
                self.current_hlsl_tail_recursive_function = func
        if effective_shader_type is None and body is None:
            function_name = self.hlsl_function_declaration_name(func, entry_name)
            if _workgroup_pointer_variant is not None:
                function_name = self.hlsl_workgroup_pointer_specialization_name(
                    getattr(func, "name", None),
                    _workgroup_pointer_variant,
                    function_name,
                )
            return_semantic_attr = self.map_semantic(return_semantic)
            code += (
                f"{declaration_return_type} {function_name}({params_str})"
                f"{return_semantic_attr};\n\n"
            )
            restore_hlsl_function_state(self, saved_function_state)
            return code

        code, function_name = self.generate_hlsl_function_header(
            func,
            execution_config,
            entry_name,
            stage_output_lowering,
            stage_node,
            _workgroup_pointer_variant,
            code,
            declaration_return_type,
            effective_shader_type,
            params_str,
            raw_return_type,
            return_semantic,
            shader_attr_map,
            shader_map,
        )

        return self.generate_hlsl_function_body(
            func,
            indent,
            shader_type,
            entry_name,
            stage_output_lowering,
            body,
            code,
            effective_shader_type,
            function_name,
            raw_return_type,
            return_type,
            saved_function_body_state,
            saved_function_state,
            signature,
            software_subgroup_entry_index,
        )

    def emit_hlsl_function_parameter(self, signature, p):
        """Render one parameter of a function: its declaration and semantic, and the aliases, prologue and epilogue statements and resource facts it adds."""
        if id(p) in signature.promoted_entry_resource_parameter_ids:
            return
        if self.hlsl_resource_pointer_parameter_is_omitted(signature.func, p):
            return
        self.current_hlsl_resource_pointer_aliases.pop(p.name, None)
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
        raw_param_type_name = self.type_name_string(raw_param_type)
        self.local_variable_types[p.name] = raw_param_type_name
        self.local_variable_source_types[p.name] = raw_param_type_name

        handled = self.emit_hlsl_resource_pointer_array_parameter(signature, p)
        if handled is not UNHANDLED:
            return handled

        handled = self.emit_hlsl_workgroup_pointer_parameter(signature, p)
        if handled is not UNHANDLED:
            return handled

        param_decl_type = self.type_name_string(raw_param_type)
        resource_pointer_contract = self.hlsl_resource_pointer_parameter_contract(
            p, signature.function_name
        )
        param_type = self.hlsl_workgroup_pointer_parameter_array_type(
            p, getattr(signature.func, "name", None)
        )
        if param_type is None:
            param_type = self.hlsl_private_pointer_parameter_array_type(
                p, getattr(signature.func, "name", None)
            )
        if param_type is None and resource_pointer_contract is not None:
            param_type = resource_pointer_contract["resource_type"]
        if param_type is None:
            param_type = self.map_resource_parameter_type_with_hint(
                param_decl_type, p, getattr(signature.func, "name", None)
            )
        param_type = self.directx_resource_declaration_type(param_type)
        self.current_hlsl_parameter_resource_types[p.name] = param_type
        if self.is_texture_type(raw_param_type) or self.is_image_type(raw_param_type):
            signature.texture_parameters[p.name] = self.map_resource_type_with_format(
                self.resource_base_type(raw_param_type), p
            )
            if self.is_image_type(raw_param_type):
                record_explicit_image_metadata(
                    p.name,
                    p,
                    self.attribute_value_to_string,
                    image_formats=signature.image_format_parameters,
                    image_accesses=signature.image_access_parameters,
                )
        if self.is_sampler_type(raw_param_type):
            signature.sampler_parameters.add(p.name)
            if (
                p.name in signature.comparison_sampler_parameters
                or p.name in signature.implicit_existing_comparison_samplers
            ):
                param_type = param_type.replace(
                    "SamplerState", "SamplerComparisonState", 1
                )

        semantic = self.semantic_from_node(p)
        if semantic is None:
            semantic = signature.default_vertex_parameter_semantics.get(id(p))
        handled = self.emit_hlsl_builtin_input_parameter(
            signature, p, param_type, semantic
        )
        if handled is not UNHANDLED:
            return handled

        self.emit_hlsl_parameter_declaration(
            signature, p, param_type, resource_pointer_contract, semantic
        )

    def generate_statement(self, stmt, indent=0):
        """Render a single CrossGL AST statement as HLSL source."""
        indent_str = "    " * indent
        # The expression whose value a statement discards (see hlsl_call_argument_order).
        self.hlsl_statement_expression = getattr(stmt, "expression", stmt)

        if isinstance(stmt, VariableNode):
            comma_initializer = split_comma_initializer(stmt)
            if comma_initializer is not None:
                effects, declaration = comma_initializer
                return "".join(
                    self.generate_statement(statement, indent)
                    for statement in (*effects, declaration)
                )
            source_vtype = self.hlsl_source_variable_declared_type(stmt)
            vtype = self.local_variable_declared_type(stmt)
            self.local_variable_types[stmt.name] = self.type_name_string(vtype)
            self.local_variable_source_types[stmt.name] = self.type_name_string(
                source_vtype
            )
            if id(stmt) in self.hlsl_hoisted_groupshared_declaration_ids:
                return ""
            resource_pointer_alias = (
                self.generate_hlsl_resource_pointer_alias_declaration(stmt, indent)
            )
            if resource_pointer_alias is not None:
                return self.hlsl_record_generated_statement_int_constants(
                    stmt, resource_pointer_alias
                )
            if isinstance(source_vtype, PointerType):
                address_space = str(source_vtype.address_space or "").strip().lower()
                if address_space in {"", "thread", "private", "function"}:
                    raise DirectXPrivatePointerParameterError(
                        "DirectX cannot emit an unlowered private pointer local "
                        f"'{self.current_function_name}.{stmt.name}'",
                        function_name=self.current_function_name,
                        parameter_name=stmt.name,
                        reason="local-pointer-declaration-unsupported",
                        source_location=getattr(stmt, "source_location", None),
                    )
            if isinstance(source_vtype, ReferenceType):
                raise DirectXPrivatePointerParameterError(
                    "DirectX cannot preserve aliasing for local reference "
                    f"'{self.current_function_name}.{stmt.name}'",
                    function_name=self.current_function_name,
                    parameter_name=stmt.name,
                    reason="local-reference-declaration-unsupported",
                    source_location=getattr(stmt, "source_location", None),
                )
            stmt_name = self.hlsl_declaration_identifier_name(stmt.name)
            if self.is_unsupported_glsl_buffer_block_struct_type(vtype):
                self.current_unsupported_glsl_buffer_block_local_variables.add(
                    stmt.name
                )
                return self.hlsl_record_generated_statement_int_constants(
                    stmt,
                    f"{indent_str}"
                    f"{self.unsupported_glsl_buffer_block_local_variable_placeholder('HLSL', vtype, stmt.name)};\n",
                )

            declaration_type = self.hlsl_crossgl_fixed_array_value_type(
                vtype
            ) or self.map_type(vtype)
            base_declaration = format_c_style_array_declaration(
                declaration_type, stmt_name
            )
            declaration = f"{self.local_variable_qualifier(stmt)}{base_declaration}"
            initial_value = getattr(stmt, "initial_value", None)
            if isinstance(initial_value, MatchNode):
                code = f"{indent_str}{declaration};\n"
                code += generate_match_expression_assignment(
                    self,
                    initial_value,
                    stmt_name,
                    vtype,
                    indent,
                    "HLSL",
                )
                return self.hlsl_record_generated_statement_int_constants(stmt, code)
            if initial_value is not None:
                storage_struct_init = self.generate_hlsl_storage_struct_view_copy(
                    stmt_name,
                    vtype,
                    initial_value,
                    indent,
                    declaration=declaration,
                )
                if storage_struct_init is not None:
                    return self.hlsl_record_generated_statement_int_constants(
                        stmt, storage_struct_init
                    )
                aggregate_init = (
                    self.generate_hlsl_aggregate_conditional_initialization(
                        initial_value,
                        f"{self.local_variable_qualifier(stmt, omit_const=True)}"
                        f"{base_declaration}",
                        stmt_name,
                        vtype,
                        indent,
                    )
                )
                if aggregate_init is not None:
                    return self.hlsl_record_generated_statement_int_constants(
                        stmt, aggregate_init
                    )
                aggregate_value = (
                    self.render_hlsl_aggregate_conditional_value_expression(
                        initial_value,
                        vtype,
                        indent,
                        context="initializer",
                    )
                )
                if aggregate_value is not None:
                    aggregate_code, init_expr = aggregate_value
                    aggregate_code += f"{indent_str}{declaration} = {init_expr};\n"
                    return self.hlsl_record_generated_statement_int_constants(
                        stmt, aggregate_code
                    )
                ternary_init = (
                    self.generate_hlsl_typed_buffer_atomic_ternary_initialization(
                        initial_value,
                        declaration,
                        stmt_name,
                        vtype,
                        indent,
                    )
                )
                if ternary_init is not None:
                    return self.hlsl_record_generated_statement_int_constants(
                        stmt, ternary_init
                    )
                atomic_init = self.generate_hlsl_typed_buffer_atomic_statement(
                    initial_value, stmt_name, vtype
                )
                if atomic_init is not None:
                    return self.hlsl_record_generated_statement_int_constants(
                        stmt,
                        f"{indent_str}{declaration};\n"
                        f"{self.generate_statement_code(atomic_init, indent)}",
                    )
                if self.hlsl_expression_contains_typed_buffer_atomic(initial_value):
                    code, init_expr = (
                        self.render_hlsl_typed_buffer_atomic_value_expression(
                            initial_value, vtype, indent
                        )
                    )
                    code += f"{indent_str}{declaration} = {init_expr};\n"
                    return self.hlsl_record_generated_statement_int_constants(
                        stmt, code
                    )
                lifted_init = self.hlsl_typed_buffer_atomic_lifted_expression(
                    initial_value, vtype
                )
                if lifted_init is not None:
                    lift_statements, init_expr = lifted_init
                    code = self.generate_statement_code(
                        "\n".join(lift_statements), indent
                    )
                    code += f"{indent_str}{declaration} = {init_expr};\n"
                    return self.hlsl_record_generated_statement_int_constants(
                        stmt, code
                    )
                struct_initializer = self.hlsl_struct_initializer_components(
                    vtype, initial_value
                )
                if struct_initializer is not None:
                    _type_name, fields, rendered_args, field_exprs = struct_initializer
                    is_const = "const" in self.local_variable_qualifier(stmt).split()
                    target = (
                        self.next_hlsl_temp_variable("struct_init")
                        if is_const
                        else stmt_name
                    )
                    struct_init = self.render_hlsl_struct_field_assignments(
                        target, fields, rendered_args, field_exprs, indent
                    )
                    if is_const:
                        self.local_variable_types[target] = self.type_name_string(vtype)
                        code = (
                            f"{indent_str}{declaration_type} {target};\n{struct_init}"
                        )
                        code += f"{indent_str}{declaration} = {target};\n"
                    else:
                        code = f"{indent_str}{declaration};\n{struct_init}"
                    return self.hlsl_record_generated_statement_int_constants(
                        stmt, code
                    )
                init_expr = self.generate_expression_with_expected(initial_value, vtype)
                return self.hlsl_record_generated_statement_int_constants(
                    stmt, f"{indent_str}{declaration} = {init_expr};\n"
                )
            else:
                return self.hlsl_record_generated_statement_int_constants(
                    stmt, f"{indent_str}{declaration};\n"
                )

        elif isinstance(stmt, ArrayNode):
            element_type = self.map_type(stmt.element_type)
            size = get_array_size_from_node(stmt)
            stmt_name = self.hlsl_declaration_identifier_name(stmt.name)

            if size is None:
                raise UnsizedArrayUnsupportedError(
                    "directx", stmt.name, "is a local variable"
                )
            else:
                return f"{indent_str}{element_type}[{size}] {stmt_name};\n"

        elif isinstance(stmt, AssignmentNode):
            self.hlsl_storage_struct_view_write_error(
                getattr(stmt, "target", getattr(stmt, "left", None))
            )
            aggregate_assignment = (
                self.generate_hlsl_aggregate_conditional_assignment_statement(
                    stmt, indent
                )
            )
            if aggregate_assignment is not None:
                return self.hlsl_record_generated_statement_int_constants(
                    stmt, aggregate_assignment
                )
            block_atomic_assignment = (
                generate_hlsl_buffer_block_atomic_value_assignment(self, stmt, indent)
            )
            if block_atomic_assignment is not None:
                return self.hlsl_record_generated_statement_int_constants(
                    stmt, block_atomic_assignment
                )
            ternary_assignment = (
                self.generate_hlsl_typed_buffer_atomic_ternary_assignment_statement(
                    stmt, indent
                )
            )
            if ternary_assignment is not None:
                return self.hlsl_record_generated_statement_int_constants(
                    stmt, ternary_assignment
                )
            atomic_assignment = (
                self.generate_hlsl_typed_buffer_atomic_value_assignment_statement(
                    stmt, indent
                )
            )
            if atomic_assignment is not None:
                return self.hlsl_record_generated_statement_int_constants(
                    stmt, atomic_assignment
                )
            return self.hlsl_record_generated_statement_int_constants(
                stmt,
                self.generate_statement_code(
                    self.generate_assignment(stmt, statement_context=True), indent
                ),
            )

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

        elif isinstance(stmt, ReturnNode):
            if hasattr(stmt, "value") and stmt.value is not None:
                if isinstance(stmt.value, list):
                    code = ""
                    for i, return_stmt in enumerate(stmt.value):
                        code += f"{self.generate_expression(return_stmt)}"
                        if i < len(stmt.value) - 1:
                            code += ", "
                    return self.generate_hlsl_return_value_statement(code, indent)
                else:
                    aggregate_return = self.generate_hlsl_aggregate_conditional_return(
                        stmt.value, indent
                    )
                    if aggregate_return is not None:
                        return aggregate_return
                    aggregate_value = (
                        self.render_hlsl_aggregate_conditional_value_expression(
                            stmt.value,
                            self.current_function_return_type,
                            indent,
                            context="return",
                        )
                    )
                    if aggregate_value is not None:
                        aggregate_code, return_expr = aggregate_value
                        aggregate_code += self.generate_hlsl_return_value_statement(
                            return_expr, indent
                        )
                        return aggregate_code
                    ternary_return = (
                        self.generate_hlsl_typed_buffer_atomic_ternary_return(
                            stmt.value, indent
                        )
                    )
                    if ternary_return is not None:
                        return ternary_return
                    atomic_return = self.generate_hlsl_typed_buffer_atomic_return(
                        stmt.value, indent
                    )
                    if atomic_return is not None:
                        return atomic_return
                    if self.hlsl_expression_contains_typed_buffer_atomic(stmt.value):
                        code, return_expr = (
                            self.render_hlsl_typed_buffer_atomic_value_expression(
                                stmt.value,
                                self.current_function_return_type,
                                indent,
                            )
                        )
                        code += self.generate_hlsl_return_value_statement(
                            return_expr, indent
                        )
                        return code
                    lifted_return = self.hlsl_typed_buffer_atomic_lifted_expression(
                        stmt.value, self.current_function_return_type
                    )
                    if lifted_return is not None:
                        lift_statements, return_expr = lifted_return
                        code = self.generate_statement_code(
                            "\n".join(lift_statements), indent
                        )
                        code += self.generate_hlsl_return_value_statement(
                            return_expr, indent
                        )
                        return code
                    struct_return = self.render_hlsl_struct_return(stmt.value, indent)
                    if struct_return is not None:
                        return struct_return
                    return self.generate_hlsl_return_value_statement(
                        self.generate_expression_with_expected(
                            stmt.value, self.current_function_return_type
                        ),
                        indent,
                    )
            else:
                stage_output_lowering = self.current_hlsl_stage_output_lowering
                if (
                    stage_output_lowering is not None
                    and stage_output_lowering.get("local_name") is not None
                ):
                    return self.generate_hlsl_return_value_statement(
                        stage_output_lowering["local_name"], indent
                    )
                return (
                    self.generate_hlsl_parameter_epilogue(indent)
                    + f"{indent_str}return;\n"
                )

        elif hasattr(stmt, "__class__") and "ExpressionStatement" in str(
            stmt.__class__
        ):
            if hasattr(stmt, "expression"):
                tail_return = self.generate_tail_expression_statement(stmt, indent)
                if tail_return is not None:
                    return tail_return
                tail_recursive_call = self.generate_hlsl_tail_recursive_call_statement(
                    stmt.expression, indent
                )
                if tail_recursive_call is not None:
                    return tail_recursive_call
                pointer_array_base_unary = (
                    self.generate_hlsl_resource_pointer_array_base_unary(
                        stmt.expression,
                        statement_context=True,
                    )
                )
                if pointer_array_base_unary is not None:
                    return f"{indent_str}{pointer_array_base_unary};\n"
                pointer_step = self.generate_hlsl_resource_pointer_step(
                    stmt.expression, statement_context=True
                )
                if pointer_step is not None:
                    return f"{indent_str}{pointer_step};\n"
                if isinstance(getattr(stmt, "expression", None), AssignmentNode):
                    self.hlsl_storage_struct_view_write_error(
                        getattr(
                            stmt.expression,
                            "target",
                            getattr(stmt.expression, "left", None),
                        )
                    )
                    aggregate_assignment = (
                        self.generate_hlsl_aggregate_conditional_assignment_statement(
                            stmt.expression, indent
                        )
                    )
                    if aggregate_assignment is not None:
                        return self.hlsl_record_generated_statement_int_constants(
                            stmt, aggregate_assignment
                        )
                    block_atomic_assignment = (
                        generate_hlsl_buffer_block_atomic_value_assignment(
                            self, stmt.expression, indent
                        )
                    )
                    if block_atomic_assignment is not None:
                        return self.hlsl_record_generated_statement_int_constants(
                            stmt, block_atomic_assignment
                        )
                    ternary_assignment = self.generate_hlsl_typed_buffer_atomic_ternary_assignment_statement(
                        stmt.expression, indent
                    )
                    if ternary_assignment is not None:
                        return self.hlsl_record_generated_statement_int_constants(
                            stmt, ternary_assignment
                        )
                    atomic_assignment = self.generate_hlsl_typed_buffer_atomic_value_assignment_statement(
                        stmt.expression, indent
                    )
                    if atomic_assignment is not None:
                        return self.hlsl_record_generated_statement_int_constants(
                            stmt, atomic_assignment
                        )
                    return self.hlsl_record_generated_statement_int_constants(
                        stmt,
                        self.generate_statement_code(
                            self.generate_assignment(
                                stmt.expression, statement_context=True
                            ),
                            indent,
                        ),
                    )
                aggregate_value = (
                    self.render_hlsl_aggregate_conditional_value_expression(
                        stmt.expression,
                        self.expression_result_type(stmt.expression),
                        indent,
                        context="expression-statement",
                    )
                )
                if aggregate_value is not None:
                    aggregate_code, expression = aggregate_value
                    return f"{aggregate_code}{indent_str}{expression};\n"
                interlocked_block = generate_hlsl_buffer_block_interlocked_statement(
                    self, stmt.expression, indent
                )
                if interlocked_block is not None:
                    return interlocked_block
                atomic_statement = self.generate_hlsl_typed_buffer_atomic_statement(
                    stmt.expression
                )
                if atomic_statement is not None:
                    return self.generate_statement_code(atomic_statement, indent)
                byte_address_atomic = hlsl_byte_address_atomic_statement(
                    self, stmt.expression
                )
                if byte_address_atomic is not None:
                    return f"{indent_str}{byte_address_atomic};\n"
                if (
                    self.hlsl_byteaddress_interlocked_member_call_parts(stmt.expression)
                    is not None
                ):
                    expression = (
                        self.generate_hlsl_byteaddress_interlocked_statement_expression(
                            stmt.expression
                        )
                    )
                    return f"{indent_str}{expression};\n"
                if self.hlsl_expression_contains_typed_buffer_atomic(stmt.expression):
                    atomic_code, expression = (
                        self.render_hlsl_typed_buffer_atomic_value_expression(
                            stmt.expression, None, indent
                        )
                    )
                    return f"{atomic_code}{indent_str}{expression};\n"
                array_call = self.render_hlsl_fixed_array_literal_call_statement(
                    stmt.expression, indent
                )
                if array_call is not None:
                    return array_call
                storage_statement = self.hlsl_half_storage.statement(stmt.expression)
                if storage_statement is not None:
                    return f"{indent_str}{storage_statement};\n"
                expression = self.generate_expression(stmt.expression)
                return f"{indent_str}{expression};\n"
            else:
                return f"{indent_str}{self.generate_expression(stmt)};\n"

        else:
            tail_recursive_call = self.generate_hlsl_tail_recursive_call_statement(
                stmt, indent
            )
            if tail_recursive_call is not None:
                return tail_recursive_call
            pointer_array_base_unary = (
                self.generate_hlsl_resource_pointer_array_base_unary(
                    stmt,
                    statement_context=True,
                )
            )
            if pointer_array_base_unary is not None:
                return f"{indent_str}{pointer_array_base_unary};\n"
            pointer_step = self.generate_hlsl_resource_pointer_step(
                stmt, statement_context=True
            )
            if pointer_step is not None:
                return f"{indent_str}{pointer_step};\n"
            aggregate_value = self.render_hlsl_aggregate_conditional_value_expression(
                stmt,
                self.expression_result_type(stmt),
                indent,
                context="expression-statement",
            )
            if aggregate_value is not None:
                aggregate_code, expression = aggregate_value
                return f"{aggregate_code}{indent_str}{expression};\n"
            atomic_statement = self.generate_hlsl_typed_buffer_atomic_statement(stmt)
            if atomic_statement is not None:
                return self.generate_statement_code(atomic_statement, indent)
            byte_address_atomic = hlsl_byte_address_atomic_statement(self, stmt)
            if byte_address_atomic is not None:
                return f"{indent_str}{byte_address_atomic};\n"
            if self.hlsl_byteaddress_interlocked_member_call_parts(stmt) is not None:
                expression = (
                    self.generate_hlsl_byteaddress_interlocked_statement_expression(
                        stmt
                    )
                )
                return f"{indent_str}{expression};\n"
            if self.hlsl_expression_contains_typed_buffer_atomic(stmt):
                atomic_code, expression = (
                    self.render_hlsl_typed_buffer_atomic_value_expression(
                        stmt, None, indent
                    )
                )
                return f"{atomic_code}{indent_str}{expression};\n"
            array_call = self.render_hlsl_fixed_array_literal_call_statement(
                stmt, indent
            )
            if array_call is not None:
                return array_call
            storage_statement = self.hlsl_half_storage.statement(stmt)
            if storage_statement is not None:
                return f"{indent_str}{storage_statement};\n"
            return f"{indent_str}{self.generate_expression(stmt)};\n"

    def is_hlsl_declaration_metadata_attribute(self, attr):
        metadata = {
            "hlsl_program_constant",
            "maybe_unused",
            "source_resource",
            "precise",
            # GLSL's invariance qualifier decorates an output (GLSL 4.60 section 4.8.1); it never names one.
            "invariant",
            # GLSL precision qualifiers set a minimum precision (GLSL ES 3.00 section 4.5.2); HLSL's 32-bit types
            # meet it, so they carry no semantic and must not become one.
            "highp",
            "mediump",
            "lowp",
            # A declared buffer layout is a layout contract, not a semantic.
            "buffer_layout",
            # Matrix orientation is a storage qualifier and is emitted as one.
            "row_major",
            "column_major",
            # MSL's packed vector storage is a buffer layout, which the layout planners read.
            "packed",
            # A member's own offset and alignment, which the layout planners read as well.
            *MEMBER_LAYOUT_QUALIFIERS,
            # Ray payload access qualifiers follow the semantic (hlsl_payload_access_qualifiers).
            *PAYLOAD_ACCESS_ATTRIBUTES,
            # MSL's half texture element type; the image format attribute gives HLSL its element type.
            "metal_texture_element_half",
        }
        metadata.update(self.METAL_ADDRESS_SPACE_METADATA_ATTRIBUTES)
        # A specialization constant ID (MSL function_constant, GLSL constant_id) is read by
        # directx_specialization_constant_attributes. On a resource parameter it makes the binding conditional
        # (MSL specification section 5.8); it never names a semantic.
        if self.directx_is_specialization_constant_attribute(attr):
            return True
        return str(getattr(attr, "name", "")).lower() in metadata

    def expression_result_type(self, expr):
        if expr is None:
            return None
        if is_comma_expression(expr):
            return self.expression_result_type(comma_result_operand(expr))
        if isinstance(expr, PointerReinterpretNode):
            target_type = getattr(expr.target_type, "pointee_type", None)
            return self.map_type(self.type_name_string(target_type))
        if isinstance(expr, VariableNode):
            return self.variable_type_by_name(getattr(expr, "name", None))
        if isinstance(expr, IdentifierNode) or (
            hasattr(expr, "__class__") and "Identifier" in str(expr.__class__)
        ):
            name = getattr(expr, "name", None)
            builtin_type = self.hlsl_graphics_builtin_result_type(name)
            if builtin_type is not None:
                return builtin_type
            variable_type = self.variable_type_by_name(name)
            if (
                variable_type is None
                and self.hlsl_standard_math_constant_expression(name) is not None
            ):
                return standard_constant_type(name)
            return variable_type
        if isinstance(expr, bool):
            return "bool"
        if isinstance(expr, (int, float)):
            return "float" if isinstance(expr, float) else "int"
        handled = self.hlsl_binary_expression_result_type(expr)
        if handled is not UNHANDLED:
            return handled
        if isinstance(expr, UnaryOpNode):
            if self.map_operator(expr.op) == "!":
                return self.hlsl_boolean_expression_result_type(
                    self.expression_result_type(expr.operand)
                )
            if expr.op == "*" and not getattr(expr, "is_postfix", False):
                binding = self.hlsl_resource_pointer_binding(expr.operand)
                if binding is not None and binding.get("element_type") is not None:
                    return binding["element_type"]
                pointee_type = self.hlsl_pointer_pointee_type_once(
                    self.expression_result_type(expr.operand)
                )
                if pointee_type is not None:
                    return pointee_type
            narrow_scalar_unary = self.hlsl_narrow_scalar_unary_result_type(expr)
            if narrow_scalar_unary is not None:
                return narrow_scalar_unary
            return self.expression_result_type(expr.operand)
        if isinstance(expr, TernaryOpNode) or (
            hasattr(expr, "__class__") and "TernaryOp" in str(expr.__class__)
        ):
            true_type = self.expression_result_type(getattr(expr, "true_expr", None))
            false_type = self.expression_result_type(getattr(expr, "false_expr", None))
            wide_integer_contract = self.hlsl_wide_integer_arithmetic_contract(
                expr, "?:", true_type, false_type
            )
            if wide_integer_contract is not None:
                return wide_integer_contract["operation_type"]
            if self.is_vector_value_type(true_type):
                return true_type
            if self.is_vector_value_type(false_type):
                return false_type
            if true_type == false_type:
                return true_type
            if true_type == "float" or false_type == "float":
                return "float"
            return true_type or false_type
        if isinstance(expr, AssignmentNode):
            target = getattr(expr, "target", getattr(expr, "left", None))
            return self.expression_result_type(target)
        handled = self.hlsl_access_expression_result_type(expr)
        if handled is not UNHANDLED:
            return handled
        if isinstance(expr, ConstructorNode):
            return infer_enum_constructor_type(
                self, expr
            ) or infer_struct_constructor_type(self, expr)
        if isinstance(expr, MatchNode):
            return infer_match_expression_result_type(self, expr)
        if isinstance(expr, RayQueryOpNode):
            return self.hlsl_ray_query_method_return_type(expr.operation)
        if isinstance(expr, WaveOpNode):
            return self.hlsl_wave_intrinsic_return_type(expr.operation, expr.arguments)
        handled = self.hlsl_call_expression_result_type(expr)
        if handled is not UNHANDLED:
            return handled
        if hasattr(expr, "__class__") and "Literal" in str(expr.__class__):
            return self.hlsl_literal_result_type(expr)
        if hasattr(expr, "__class__") and "Identifier" in str(expr.__class__):
            return self.variable_type_by_name(getattr(expr, "name", None))
        return None

    def hlsl_pointer_reinterpret_binding(self, expression, binding):
        target_type = getattr(expression.target_type, "pointee_type", None)
        target_type_name = self.type_name_string(target_type)
        mapped_target_type = self.map_type(target_type_name)
        view_type_name = (
            binding.get("view_element_type")
            or binding.get("source_element_type")
            or binding.get("element_type")
        )
        source_type_name = binding.get("source_element_type") or view_type_name
        # A one-scalar wrapper record backs the view with its member's words.
        view_layout, _ = reinterpret_source_layout(view_type_name, self.structs_by_name)
        source_layout, source_member = reinterpret_source_layout(
            source_type_name, self.structs_by_name
        )
        target_layout = scalar_storage_layout(target_type_name)
        if (
            view_layout is not None
            and target_layout == view_layout
            and scalar_storage_layout(view_type_name) == view_layout
        ):
            # A cast to the scalar type the pointer already views (such as one
            # that only drops ``const``) keeps the binding's view unchanged. A
            # one-scalar wrapper record's view still becomes its member's words.
            return binding
        lane_gather = self.hlsl_same_scalar_lane_gather_binding(
            binding,
            target_type,
            source_type_name=source_type_name,
            source_layout=source_layout,
            view_layout=view_layout,
        )
        if lane_gather is not None:
            return lane_gather
        target_width = 1
        if (
            target_layout is None
            and source_layout is not None
            and source_layout.bit_width == 32
            and self.is_vector_value_type(mapped_target_type)
        ):
            target_layout = scalar_storage_layout(
                self.vector_component_type(mapped_target_type)
            )
            target_width = int(mapped_target_type[-1])
        if (
            view_layout is None
            or source_layout is None
            or target_layout is None
            or source_layout.bit_width not in {8, 32}
            or target_layout.bit_width not in {8, 16, 32}
            or target_width > 1
            and target_layout.bit_width != 32
            or target_layout.kind == "floating"
            and target_layout.bit_width != 32
        ):
            raise PointerReinterpretationError(
                "DirectX storage pointer reinterpretation requires an 8- or 32-bit "
                "scalar backing element and either an 8-, 16-, or 32-bit scalar "
                "view or a 2- to 4-lane 32-bit vector view",
                source_type=source_type_name,
                target_type=target_type_name,
                address_space="storage",
                alignment=target_layout.byte_width if target_layout else None,
                access=binding.get("access"),
                reason="unsupported-scalar-layout",
                source_location=getattr(expression, "source_location", None),
            )

        rendered_offset = str(binding.get("offset", "0"))
        scaled_offset = rendered_offset
        if view_layout.byte_width != 1 and rendered_offset not in {"0", "0u"}:
            scaled_offset = f"({rendered_offset} * {view_layout.byte_width})"
        existing_byte_offset = str(binding.get("byte_offset", "0"))
        if existing_byte_offset in {"0", "0u"}:
            byte_offset = scaled_offset
        elif scaled_offset in {"0", "0u"}:
            byte_offset = existing_byte_offset
        else:
            byte_offset = f"({existing_byte_offset} + {scaled_offset})"

        return {
            **binding,
            "offset": "0",
            "byte_offset": byte_offset,
            "element_type": target_type_name,
            "view_element_type": target_type_name,
            "source_element_type": source_type_name,
            "pointer_reinterpretation": {
                "source_layout": source_layout,
                "target_layout": target_layout,
                "target_width": target_width,
                "source_member": source_member,
            },
        }

    def generate_hlsl_cooperative_matrix_mma_helper(self):
        mapping = get_cooperative_matrix_fragment_mapping(
            "tile_4x4_row_pair", 8, 8, 32, 2
        )
        helper = self.hlsl_software_subgroup_identifier(
            ("matrix", "mma"), "__crossgl_software_matrix_mma"
        )
        scratch = [
            self.hlsl_software_subgroup_identifier(
                ("matrix", operand), f"__crossgl_software_matrix_{operand}"
            )
            for operand in ("left", "right")
        ]
        count = self.hlsl_software_subgroup_invocation_count
        lines = [f"groupshared float2 {name}[{count}];" for name in scratch]
        rounded = self.cooperative_matrix_mma_profile is not None
        if rounded:
            from ..fused_math import FMA_HELPER_KEYS, binary32_fma_support
            from ..lexer import Lexer
            from ..parser import Parser

            names = {
                key: self.hlsl_software_subgroup_identifier(
                    ("matrix_round", key), f"__crossgl_software_matrix_round_{key}"
                )
                for key in FMA_HELPER_KEYS
            }
            source = "shader MatrixRounding {\n" + binary32_fma_support(names) + "\n}"
            support = HLSLCodeGen().generate(Parser(Lexer(source).tokens).parse())
            lines.insert(0, support)
            flush = (
                "true"
                if self.cooperative_matrix_mma_profile.endswith("-flush")
                else "false"
            )
        lines.extend(
            [
                f"float2 {helper}(float2 left, float2 right, float2 accumulator, uint invocation) {{",
                "    uint lane = invocation % 32u;",
                "    uint subgroupBase = invocation - lane;",
                f"    {scratch[0]}[invocation] = left;",
                f"    {scratch[1]}[invocation] = right;",
                "    GroupMemoryBarrierWithGroupSync();",
                (
                    "    uint2 result = asuint(accumulator);"
                    if rounded
                    else "    precise float2 result = accumulator;"
                ),
            ]
        )
        # Tables come from the registered source mapping, not the physical wave layout.
        owners = [mapping.owner(row, column) for row in range(8) for column in range(8)]
        tables = {
            "ownerLanes": [owner[0] for owner in owners],
            "ownerElements": [owner[1] for owner in owners],
        }
        for element in range(2):
            tables[f"coordinates{element}"] = [
                row * 8 + column
                for lane in range(32)
                for row, column in [mapping.coordinate(lane, element)]
            ]
        for name, values in tables.items():
            rendered = ", ".join(f"{value}u" for value in values)
            lines.append(f"    const uint {name}[{len(values)}] = {{{rendered}}};")
        for element in range(2):
            lines.extend(
                [
                    f"    uint row{element} = coordinates{element}[lane] / 8u;",
                    f"    uint column{element} = coordinates{element}[lane] % 8u;",
                ]
            )
            if rounded:
                # Keep integer rounding in an ordered loop instead of duplicating
                # its control flow eight times per fragment component.
                prefix = f"term{element}"
                lines.extend(
                    [
                        "    [loop]",
                        f"    for (uint {prefix}Index = 0u; {prefix}Index < 8u; ++{prefix}Index) {{",
                        f"        uint {prefix}A = row{element} * 8u + {prefix}Index;",
                        f"        uint {prefix}B = {prefix}Index * 8u + column{element};",
                        f"        uint {prefix}Left = asuint({scratch[0]}[subgroupBase + "
                        f"ownerLanes[{prefix}A]][ownerElements[{prefix}A]]);",
                        f"        uint {prefix}Right = asuint({scratch[1]}[subgroupBase + "
                        f"ownerLanes[{prefix}B]][ownerElements[{prefix}B]]);",
                    ]
                )
                if self.cooperative_matrix_mma_profile.startswith("fma-"):
                    lines.append(
                        f"        result[{element}] = {names['bits']}({prefix}Left, "
                        f"{prefix}Right, result[{element}], {flush});"
                    )
                else:
                    # Separate rounding boundaries retain the product's signed zero.
                    lines.extend(
                        [
                            f"        uint {prefix} = {names['bits']}({prefix}Left, {prefix}Right, "
                            f"({prefix}Left ^ {prefix}Right) & 0x80000000u, {flush});",
                            f"        result[{element}] = {names['bits']}(result[{element}], "
                            f"0x3f800000u, {prefix}, {flush});",
                        ]
                    )
                lines.append("    }")
                continue
            for inner in range(8):
                prefix = f"term{element}_{inner}"
                lines.extend(
                    [
                        f"    uint {prefix}A = row{element} * 8u + {inner}u;",
                        f"    uint {prefix}B = {inner}u * 8u + column{element};",
                    ]
                )
                left = f"{scratch[0]}[subgroupBase + ownerLanes[{prefix}A]][ownerElements[{prefix}A]]"
                right = f"{scratch[1]}[subgroupBase + ownerLanes[{prefix}B]][ownerElements[{prefix}B]]"
                lines.extend(
                    [
                        f"    precise float {prefix} = {left} * {right};",
                        f"    result[{element}] = result[{element}] + {prefix};",
                    ]
                )
        lines.extend(
            [
                "    GroupMemoryBarrierWithGroupSync();",
                "    return asfloat(result);" if rounded else "    return result;",
                "}",
                "",
            ]
        )
        return "\n".join(lines)

    def generate_expression(self, expr):
        """Render a CrossGL AST expression into HLSL expression syntax."""
        storage_read = self.hlsl_half_storage.read(expr)
        if storage_read is not None:
            return storage_read
        ray_query = hlsl_ray_query_expression(self, expr)
        if ray_query is not None:
            return ray_query
        unavailable = hlsl_unavailable_builtin(self, expr)
        if unavailable is not None:
            return unavailable
        helper_lane = self.hlsl_helper_invocation_expression(expr)
        if helper_lane is not None:
            return helper_lane
        builtin_read = hlsl_stage_builtin_read(self, expr)
        if builtin_read is not None:
            return builtin_read
        # Arguments first: the ordered call renders the call itself through this method again.
        ordered_call = hlsl_ordered_call_expression(self, expr)
        if ordered_call is not None:
            return ordered_call
        exact_float = hlsl_exact_float_expression(self, expr)
        if exact_float is not None:
            return exact_float
        if expr is None:
            return ""
        elif isinstance(expr, CooperativeMatrixOpNode):
            return self.generate_hlsl_cooperative_matrix_operation_expression(expr)
        elif isinstance(expr, str):
            if self.is_hlsl_builtin_option_none_expression(expr):
                none_value = self.generate_hlsl_builtin_option_none_value()
                if none_value is not None:
                    return none_value
            self.hlsl_resource_pointer_array_bare_expression_error(expr)
            self.hlsl_workgroup_pointer_bare_expression_error(expr)
            return expr
        elif isinstance(expr, bool):
            return "true" if expr else "false"
        elif isinstance(expr, (int, float)):
            return str(expr)
        elif isinstance(expr, ArrayLiteralNode):
            return self.hlsl_array_literal_expression(expr)
        elif is_comma_expression(expr):
            return c_family_comma_expression(
                expr, self.generate_expression, self.generate_discarded_expression
            )
        elif hasattr(expr, "__class__") and "Literal" in str(expr.__class__):
            return self.generate_hlsl_literal_expression(expr)
        elif hasattr(expr, "__class__") and "Identifier" in str(expr.__class__):
            return self.generate_hlsl_identifier_node_expression(expr)
        elif isinstance(expr, VariableNode):
            return self.generate_hlsl_variable_expression(expr)
        elif isinstance(expr, PointerReinterpretNode):
            target_type = getattr(expr.target_type, "pointee_type", None)
            target_layout = scalar_storage_layout(self.type_name_string(target_type))
            raise PointerReinterpretationError(
                "DirectX pointer reinterpretation reached an expression context "
                "without a resource or private backing contract",
                source_type=self.type_name_string(
                    self.expression_result_type(expr.expression)
                ),
                target_type=self.type_name_string(target_type),
                address_space=getattr(expr.target_type, "address_space", None),
                alignment=target_layout.byte_width if target_layout else None,
                target_backend="directx",
                reason="unconsumed-pointer-reinterpretation",
                source_location=getattr(expr, "source_location", None),
            )
        elif hasattr(expr, "__class__") and "BinaryOp" in str(expr.__class__):
            return self.generate_hlsl_binary_expression(expr)
        elif isinstance(expr, AssignmentNode):
            return assignment_value_expression(self.generate_assignment(expr))
        elif hasattr(expr, "__class__") and "UnaryOp" in str(expr.__class__):
            return self.generate_hlsl_unary_expression(expr)
        elif isinstance(expr, WaveOpNode):
            return self.generate_wave_op_expression(expr)
        elif isinstance(expr, RayTracingOpNode):
            args_str = ", ".join(
                self.generate_expression(arg) for arg in expr.arguments
            )
            return f"{expr.operation}({args_str})"
        elif isinstance(expr, MeshOpNode):
            args_str = ", ".join(
                self.generate_expression(arg) for arg in expr.arguments
            )
            return f"{expr.operation}({args_str})"
        elif isinstance(expr, RayQueryOpNode):
            query = self.generate_expression(expr.query_expr)
            args_str = ", ".join(
                self.generate_expression(arg) for arg in expr.arguments
            )
            return f"{query}.{expr.operation}({args_str})"
        elif hasattr(expr, "__class__") and "ArrayAccess" in str(expr.__class__):
            return self.generate_hlsl_array_access_expression(expr)
        elif isinstance(expr, ConstructorNode):
            return self.generate_hlsl_constructor_expression(expr)
        elif hasattr(expr, "__class__") and "FunctionCall" in str(expr.__class__):
            return self.generate_hlsl_function_call_expression(expr)
        elif hasattr(expr, "__class__") and "MemberAccess" in str(expr.__class__):
            return self.generate_hlsl_member_access_expression(expr)
        elif hasattr(expr, "__class__") and "TernaryOp" in str(expr.__class__):
            return self.generate_hlsl_ternary_expression(expr)
        else:
            return str(expr)

    def generate_hlsl_wave_intrinsic_call(self, operation, args):
        extension = hlsl_wave_extension_call(self, operation, args)
        if extension is not None:
            return extension
        expected_args = self.HLSL_WAVE_INTRINSIC_ARITIES.get(operation)
        if expected_args is None:
            raise DirectXRefusal(
                f"DirectX wave intrinsic '{operation}' is not recognized"
            )
        self.validate_directx_wave_intrinsic_profile(operation)

        actual_args = len(args)
        accepted_arities = self.HLSL_WAVE_INTRINSIC_ALTERNATE_ARITIES.get(
            operation, (expected_args,)
        )
        if actual_args not in accepted_arities:
            expected = " or ".join(str(arity) for arity in accepted_arities)
            raise DirectXRefusal(
                f"DirectX wave intrinsic '{operation}' requires "
                f"{expected} argument(s), got {actual_args}"
            )

        if operation in self.HLSL_WAVE_BITWISE_VALUE_INTRINSICS:
            component, _ = self.hlsl_wave_value_component_type(args[0])
            if component in self.HLSL_WAVE_FLOAT_BITCAST_COMPONENT_TYPES:
                value = self.generate_expression(args[0])
                return f"asfloat({operation}(asuint({value})))"

        self.validate_hlsl_wave_intrinsic_arguments(operation, args)
        self.validate_hlsl_wave_intrinsic_result_context(operation, args)
        if self.software_subgroup_width is not None and operation == "WaveReadLaneAt":
            value_type = self.expression_result_type(args[0])
            value = self.generate_expression_with_expected(args[0], value_type)
            lane = self.generate_expression_with_expected(args[1], "uint")
            result = self.hlsl_software_subgroup_call(
                operation, value_type, value, lane
            )
            return self.hlsl_wave_shuffle_result_conversion(operation, args, result)
        if (
            operation in SOFTWARE_QUAD_REDUCTIONS
            and self.software_subgroup_width is None
        ):
            # Native HLSL quads follow lane or thread-ID layouts that differ
            # from four consecutive linear invocations.
            raise DirectXRefusal(
                f"DirectX wave intrinsic '{operation}' requires software subgroup "
                "lowering, which defines quads as four consecutive invocations"
            )
        if self.software_subgroup_width is not None and operation in (
            self.HLSL_SOFTWARE_SUBGROUP_REDUCTIONS.keys()
            | self.HLSL_SOFTWARE_SUBGROUP_SCANS.keys()
            | SOFTWARE_QUAD_REDUCTIONS
        ):
            value_type = (
                self.hlsl_source_expression_type(args[0])
                if operation == "WaveActiveProduct"
                or operation in self.HLSL_SOFTWARE_SUBGROUP_SCANS
                or operation in SOFTWARE_QUAD_REDUCTIONS
                else self.expression_result_type(args[0])
            )
            value = self.generate_expression_with_expected(args[0], value_type)
            return self.hlsl_software_subgroup_call(operation, value_type, value)
        if operation in self.HLSL_WAVE_SHUFFLE_AND_FILL_INTRINSICS:
            return self.generate_hlsl_wave_shuffle_and_fill_up_call(operation, args)
        relative_operator = self.HLSL_WAVE_RELATIVE_SHUFFLE_OPERATORS.get(operation)
        if relative_operator is not None:
            # Match the Metal builtin path whether this arrived as a canonical wave
            # operation or as a function call shadowed by a user overload.
            value_type = (
                self.expression_result_type(args[0])
                or self.current_expression_expected_type
            )
            mapped_value_type = self.map_type(value_type)
            value = self.generate_expression_with_expected(
                args[0],
                self.hlsl_wave_shuffle_payload_type(value_type, mapped_value_type),
            )
            delta = self.generate_expression_with_expected(args[1], "uint")
            if self.software_subgroup_width is not None:
                result = self.hlsl_software_subgroup_call(
                    operation,
                    mapped_value_type,
                    value,
                    delta,
                )
            elif self.relative_wave_shuffle_out_of_range == "self":
                result = self.hlsl_defined_relative_wave_shuffle_call(
                    operation,
                    mapped_value_type,
                    value,
                    delta,
                )
            else:
                result = (
                    f"WaveReadLaneAt({value}, "
                    f"(WaveGetLaneIndex() {relative_operator} uint({delta})))"
                )
            return self.hlsl_wave_shuffle_result_conversion(operation, args, result)
        inclusive = self.HLSL_WAVE_INCLUSIVE_PREFIX_OPERATORS.get(operation)
        if inclusive is not None:
            exclusive_op, combine = inclusive
            value = self.generate_expression(args[0])
            # Inclusive scan = exclusive scan combined with the lane's own value.
            return f"({exclusive_op}({value}) {combine} ({value}))"
        args_str = ", ".join(self.generate_expression(arg) for arg in args)
        result = f"{operation}({args_str})"
        return self.hlsl_wave_shuffle_result_conversion(operation, args, result)

    def hlsl_byteaddress_interlocked_operations(self):
        return {
            "InterlockedAdd": ((2, 3), ((1, "value"),), 2),
            "InterlockedMin": ((2, 3), ((1, "value"),), 2),
            "InterlockedMax": ((2, 3), ((1, "value"),), 2),
            "InterlockedAnd": ((2, 3), ((1, "value"),), 2),
            "InterlockedOr": ((2, 3), ((1, "value"),), 2),
            "InterlockedXor": ((2, 3), ((1, "value"),), 2),
            "InterlockedExchange": ((2, 3), ((1, "value"),), 2),
            "InterlockedCompareExchange": (
                (4,),
                ((1, "compare value"), (2, "value")),
                3,
            ),
            "InterlockedCompareStore": (
                (3,),
                ((1, "compare value"), (2, "value")),
                None,
            ),
        }

    def hlsl_byteaddress_interlocked_uint_input_matches(self, expr):
        kind = self.scalar_expression_kind(expr)
        if kind == "uint":
            return True
        if kind == "int":
            value = self.literal_int_value(expr, self.literal_int_constants)
            return value is not None and value >= 0
        if kind is None:
            # A value whose numeric kind is unknown passes only when its type
            # cannot be resolved at all. A resolvable non-uint scalar such as
            # bool or int64_t is refused like float: DXC would otherwise
            # silently convert the bool or narrow the 64-bit value to uint.
            return self.type_name_string(self.expression_result_type(expr)) is None
        return False

    def generate_hlsl_byteaddress_interlocked_member_call(self, func_expr, args):
        if not isinstance(func_expr, MemberAccessNode):
            return None

        method_name = str(getattr(func_expr, "member", ""))
        operation_info = self.hlsl_byteaddress_interlocked_operations().get(method_name)
        if operation_info is None:
            return None

        object_expr = getattr(
            func_expr, "object", getattr(func_expr, "object_expr", None)
        )
        resource_type = self.hlsl_buffer_helper_resource_type(object_expr)
        if resource_type is None:
            return None

        resource_name = self.hlsl_resource_type_name(resource_type)
        byteaddress_resources = {
            "ByteAddressBuffer",
            "RWByteAddressBuffer",
            "RasterizerOrderedByteAddressBuffer",
        }
        if resource_name not in byteaddress_resources:
            raise DirectXRefusal(
                f"{self.hlsl_byteaddress_interlocked_error_prefix(method_name)} "
                "requires ByteAddressBuffer, RWByteAddressBuffer, or "
                "RasterizerOrderedByteAddressBuffer resource, got "
                f"{resource_type}"
            )
        if resource_name == "ByteAddressBuffer":
            raise DirectXRefusal(
                f"{self.hlsl_byteaddress_interlocked_error_prefix(method_name)} "
                "cannot write readonly ByteAddressBuffer"
            )
        if not self.allow_hlsl_byteaddress_interlocked_member_expression:
            raise DirectXRefusal(
                f"{self.hlsl_byteaddress_interlocked_error_prefix(method_name)} "
                "requires standalone statement context"
            )

        arities, value_roles, original_index = operation_info
        if len(args) not in arities:
            raise DirectXRefusal(
                f"{self.hlsl_byteaddress_interlocked_error_prefix(method_name)} "
                f"requires {self.hlsl_byteaddress_interlocked_arity_label(arities)} "
                f"argument(s), got {len(args)}"
            )

        self.validate_hlsl_byteaddress_interlocked_uint_input(
            method_name, args, 0, "address"
        )
        # An int value selects the signed IMin/IMax operation.  A non-negative
        # int literal still passes as uint: DXC reads InterlockedMax(0, 5) as
        # the unsigned UMax, so it cannot take an int original value.
        signed = (
            method_name in {"InterlockedMin", "InterlockedMax"}
            and self.scalar_expression_kind(args[1]) == "int"
            and not self.hlsl_byteaddress_interlocked_uint_input_matches(args[1])
        )
        if not signed:
            for index, role in value_roles:
                self.validate_hlsl_byteaddress_interlocked_uint_input(
                    method_name, args, index, role
                )
        if original_index is not None and len(args) > original_index:
            self.validate_hlsl_byteaddress_interlocked_original(
                method_name, args[original_index], signed
            )

        receiver = self.generate_expression(object_expr)
        rendered_args = [
            self.generate_expression_with_expected(
                arg, "int" if signed and index == 1 else "uint"
            )
            for index, arg in enumerate(args)
        ]
        return f"{receiver}.{method_name}({', '.join(rendered_args)})"

    def collect_comparison_resources(self, root):
        texture_names = set()
        sampler_names = set()
        visited = set()
        global_resource_types = self.collect_global_texture_types(root)
        comparison_funcs = self.comparison_texture_function_names()

        def visit(value):
            if value is None or isinstance(value, (str, int, float, bool)):
                return
            if isinstance(value, dict):
                for item in value.values():
                    visit(item)
                return
            if isinstance(value, (list, tuple, set)):
                for item in value:
                    visit(item)
                return
            value_id = id(value)
            if value_id in visited:
                return
            visited.add(value_id)

            if isinstance(value, FunctionCallNode):
                func_expr = getattr(value, "function", getattr(value, "name", None))
                func_name = self.expression_name(func_expr)
                args = getattr(value, "arguments", getattr(value, "args", []))
                if func_name in comparison_funcs and len(args) >= 3:
                    texture_name = self.expression_name(args[0])
                    texture_type = global_resource_types.get(texture_name)
                    if self.texture_call_is_diagnostic_only(func_name, texture_type):
                        return
                    if texture_name:
                        texture_names.add(texture_name)
                    if len(args) >= 4:
                        sampler_name = self.expression_name(args[1])
                        if sampler_name:
                            sampler_names.add(sampler_name)

            if hasattr(value, "__dict__"):
                for child in vars(value).values():
                    visit(child)

        visit(root)
        return texture_names, sampler_names

    def hlsl_analyze_workgroup_pointer_function(
        self,
        function,
        pointer_parameters,
        pointer_parameter_indices,
        calls,
        intervals=None,
    ):
        function_name = getattr(function, "name", None)
        constants = self.visible_literal_int_constants(function)
        intervals = dict(intervals or {})
        direct_ranges = {}
        direct_observations = set()
        initial_aliases = {
            name: dict(binding)
            for name, binding in self.hlsl_global_groupshared_pointer_aliases.items()
        }
        for parameter in (
            getattr(function, "parameters", getattr(function, "params", [])) or []
        ):
            initial_aliases.pop(getattr(parameter, "name", None), None)
        for parameter in pointer_parameters.get(function_name, []):
            element_type = self.hlsl_workgroup_pointer_element_type(parameter)
            initial_aliases[parameter.name] = {
                "kind": "workgroup-pointer",
                "root": parameter.name,
                "root_kind": "parameter",
                "offset": 0,
                "array_size": None,
                "element_type": element_type,
                "source_element_type": element_type,
                "resource_type": "groupshared",
                "access": "read_write",
            }

        def record_observation(binding, source, parameter_name=None):
            if binding is None or binding.get("kind") != "workgroup-pointer":
                return
            if binding.get("selected_null"):
                self.hlsl_workgroup_pointer_null_observation_error(
                    function_name=function_name,
                    parameter_name=(
                        parameter_name or binding.get("root") or "workgroup_pointer"
                    ),
                    node=source,
                )
            if binding.get("root_kind") == "parameter":
                root = binding.get("root")
                if root is not None:
                    direct_observations.add(root)

        def record_access(binding, index_expression, source, *, required=False):
            if binding is None or binding.get("kind") != "workgroup-pointer":
                return
            record_observation(binding, source)
            offset_range = self.hlsl_private_pointer_interval(
                binding.get("offset", 0), intervals, constants
            )
            index_range = self.hlsl_private_pointer_interval(
                index_expression, intervals, constants
            )
            if offset_range is None or index_range is None:
                if required:
                    raise self.hlsl_workgroup_pointer_error(
                        "DirectX cannot prove the expanded record copy range",
                        function_name=function_name,
                        parameter_name=binding.get("root"),
                        reason="unprovable-view-offset",
                        node=source,
                    )
                return
            access_range = (
                offset_range[0] + index_range[0],
                offset_range[1] + index_range[1],
            )
            if binding.get("root_kind") == "concrete":
                self.hlsl_workgroup_pointer_access_error(
                    function_name=function_name,
                    parameter_name=binding.get("root"),
                    binding=binding,
                    access_range=access_range,
                    node=source,
                )
                return
            if binding.get("root_kind") != "parameter":
                return
            root = binding.get("root")
            current = direct_ranges.get(root)
            direct_ranges[root] = (
                access_range
                if current is None
                else (
                    min(current[0], access_range[0]),
                    max(current[1], access_range[1]),
                )
            )

        def visit(value, aliases, control_depth=0):
            if value is None or isinstance(value, (str, int, float, bool)):
                return
            if isinstance(value, (list, tuple)):
                for item in value:
                    visit(item, aliases, control_depth)
                return
            if isinstance(value, BlockNode):
                block_aliases = dict(aliases)
                for statement in getattr(value, "statements", []) or []:
                    visit(statement, block_aliases, control_depth)
                return
            if isinstance(value, UnaryOpNode) and value.op == "&":
                operand = value.operand
                if isinstance(operand, ArrayAccessNode):
                    visit(operand.index, aliases, control_depth)
                    return
            if isinstance(value, UnaryOpNode) and value.op == "*":
                binding = self.hlsl_static_workgroup_pointer_binding(
                    value.operand, aliases, constants
                )
                if binding is not None:
                    record_observation(binding, value)
                    return
            if isinstance(value, ArrayAccessNode):
                binding = self.hlsl_static_workgroup_pointer_binding(
                    value.array, aliases, constants
                )
                record_access(binding, value.index, value)
                visit(value.array, aliases, control_depth)
                visit(value.index, aliases, control_depth)
                return
            if isinstance(value, VariableNode):
                initial_value = getattr(value, "initial_value", None)
                initial_binding = self.hlsl_static_workgroup_pointer_binding(
                    initial_value, aliases, constants
                )
                storage_binding = (
                    self.hlsl_groupshared_pointer_bindings_by_declaration.get(id(value))
                )
                pointer_declaration = self.hlsl_workgroup_pointer_declaration(
                    value, aliases
                )
                if storage_binding is None and not pointer_declaration:
                    visit(initial_value, aliases, control_depth)
                aliases.pop(value.name, None)
                if storage_binding is not None:
                    aliases[value.name] = dict(storage_binding)
                    return
                if not pointer_declaration:
                    return
                if initial_binding is None:
                    raise self.hlsl_workgroup_pointer_error(
                        "DirectX cannot lower workgroup pointer alias "
                        f"'{value.name}' without a concrete shared-array backing "
                        "object",
                        function_name=function_name,
                        parameter_name=value.name,
                        reason="backing-unresolved",
                        node=value,
                    )
                if (
                    initial_binding.get("root_kind") == "concrete"
                    and initial_binding.get("array_size") is None
                ):
                    raise self.hlsl_workgroup_pointer_error(
                        "DirectX cannot lower workgroup pointer alias "
                        f"'{value.name}' because its backing array extent is "
                        "unresolved",
                        function_name=function_name,
                        parameter_name=value.name,
                        reason="missing-concrete-size",
                        node=value,
                    )
                declared_type = self.hlsl_workgroup_pointer_element_type(value)
                if declared_type and declared_type != initial_binding.get(
                    "element_type"
                ):
                    raise self.hlsl_workgroup_pointer_error(
                        "DirectX workgroup pointer alias "
                        f"'{value.name}' changes element type",
                        function_name=function_name,
                        parameter_name=value.name,
                        reason="element-type-mismatch",
                        node=value,
                    )
                aliases[value.name] = {
                    **initial_binding,
                    "element_type": (
                        declared_type or initial_binding.get("element_type")
                    ),
                    # Reassignments in the declaring straight-line scope keep
                    # one statically known offset per iteration of any
                    # enclosing loop; only nested control flow is path-dependent.
                    "declaration_scope": id(aliases),
                }
                return
            if isinstance(value, AssignmentNode):
                target = getattr(value, "target", getattr(value, "left", None))
                assigned = getattr(value, "value", getattr(value, "right", None))
                transfer = self.hlsl_workgroup_record_transfer(target, assigned)
                if transfer is not None:
                    destination, source = transfer
                    destination._hlsl_record_transfer_bounds_checked = False
                    binding = self.hlsl_static_workgroup_pointer_binding(
                        destination.expression, aliases, constants
                    )
                    if binding is not None:
                        extent, _ = self.hlsl_workgroup_record_layout(source, constants)
                        layout = scalar_storage_layout(binding.get("element_type"))
                        if (
                            layout is not None
                            and layout.bit_width == 32
                            and extent % 4 == 0
                        ):
                            record_access(binding, 0, destination, required=True)
                            record_access(
                                binding, extent // 4 - 1, destination, required=True
                            )
                            destination._hlsl_record_transfer_bounds_checked = True
                if not isinstance(target, (str, IdentifierNode, VariableNode)):
                    visit(assigned, aliases, control_depth)
                    visit(target, aliases, control_depth)
                    return
                target_name = self.expression_name(target)
                current = aliases.get(target_name)
                if current is None or current.get("kind") != "workgroup-pointer":
                    visit(assigned, aliases, control_depth)
                    visit(target, aliases, control_depth)
                    return
                operator = self.map_operator(
                    getattr(value, "operator", getattr(value, "op", "="))
                )
                if control_depth > 0 and current.get("declaration_scope") != id(
                    aliases
                ):
                    raise self.hlsl_workgroup_pointer_error(
                        "DirectX cannot preserve workgroup pointer reassignment "
                        f"for '{target_name}' across nested control flow",
                        function_name=function_name,
                        parameter_name=target_name,
                        reason="dynamic-control-flow-reassignment",
                        node=value,
                    )
                if operator in {"+=", "-="}:
                    delta = assigned if operator == "+=" else UnaryOpNode("-", assigned)
                    aliases[target_name] = {
                        **current,
                        "offset": self.hlsl_static_workgroup_pointer_offset_sum(
                            current.get("offset"), delta
                        ),
                    }
                    return
                if operator != "=":
                    raise self.hlsl_workgroup_pointer_error(
                        "DirectX cannot lower workgroup pointer operator "
                        f"'{operator}' for '{target_name}'",
                        function_name=function_name,
                        parameter_name=target_name,
                        reason="assignment-operator-unsupported",
                        node=value,
                    )
                replacement = self.hlsl_static_workgroup_pointer_binding(
                    assigned, aliases, constants
                )
                if replacement is None:
                    raise self.hlsl_workgroup_pointer_error(
                        "DirectX workgroup pointer assignment has no concrete "
                        f"shared-array backing object: {target_name}",
                        function_name=function_name,
                        parameter_name=target_name,
                        reason="assignment-backing-unresolved",
                        node=value,
                    )
                if replacement.get("root") != current.get("root"):
                    raise self.hlsl_workgroup_pointer_error(
                        "DirectX cannot dynamically change the backing array for "
                        f"workgroup pointer '{target_name}'",
                        function_name=function_name,
                        parameter_name=target_name,
                        reason="dynamic-backing-reassignment",
                        node=value,
                    )
                if replacement.get("element_type") != current.get("element_type"):
                    raise self.hlsl_workgroup_pointer_error(
                        "DirectX workgroup pointer assignment changes element type "
                        f"for '{target_name}'",
                        function_name=function_name,
                        parameter_name=target_name,
                        reason="element-type-mismatch",
                        node=value,
                    )
                aliases[target_name] = {
                    **replacement,
                    "declaration_scope": current.get("declaration_scope"),
                }
                return
            if isinstance(value, FunctionCallNode):
                arguments = list(
                    getattr(value, "arguments", getattr(value, "args", [])) or []
                )
                callee_name = self.hlsl_resolved_function_call_name(
                    value, pointer_parameters
                )
                pointer_argument_indices = set()
                if callee_name is not None:
                    bindings = {}
                    for index, parameter_name in pointer_parameter_indices[
                        callee_name
                    ].items():
                        pointer_argument_indices.add(index)
                        argument = arguments[index] if index < len(arguments) else None
                        parameter = next(
                            parameter
                            for parameter in pointer_parameters[callee_name]
                            if parameter.name == parameter_name
                        )
                        expected_type = self.hlsl_workgroup_pointer_element_type(
                            parameter
                        )
                        binding = self.hlsl_static_workgroup_pointer_binding(
                            argument, aliases, constants
                        )
                        if binding is None and self.hlsl_null_pointer_expression(
                            argument
                        ):
                            binding = self.hlsl_null_workgroup_pointer_binding(
                                expected_type, offset=0
                            )
                        if binding is None:
                            raise self.hlsl_workgroup_pointer_error(
                                "DirectX workgroup pointer argument for "
                                f"'{callee_name}.{parameter_name}' has no concrete "
                                "shared-array backing object",
                                function_name=callee_name,
                                parameter_name=parameter_name,
                                reason="call-backing-unresolved",
                                node=argument or value,
                            )
                        if binding.get("element_type") != expected_type:
                            raise self.hlsl_workgroup_pointer_error(
                                "DirectX workgroup pointer argument for "
                                f"'{callee_name}.{parameter_name}' changes element "
                                "type",
                                function_name=callee_name,
                                parameter_name=parameter_name,
                                reason="element-type-mismatch",
                                node=argument or value,
                            )
                        bindings[parameter_name] = binding
                    calls.append(
                        {
                            "caller": function_name,
                            "callee": callee_name,
                            "node": value,
                            "bindings": bindings,
                            "constants": dict(constants),
                            "intervals": dict(intervals),
                        }
                    )
                for index, argument in enumerate(arguments):
                    if index in pointer_argument_indices:
                        continue
                    visit(argument, aliases, control_depth)
                return
            if isinstance(value, IfNode):
                condition = getattr(value, "condition", None)
                visit(condition, aliases, control_depth)
                condition_value = self.literal_int_value(condition, constants)
                if condition_value is not None:
                    selected = (
                        getattr(value, "then_branch", None)
                        if condition_value
                        else getattr(value, "else_branch", None)
                    )
                    visit(selected, dict(aliases), control_depth + 1)
                    return
                visit(
                    getattr(value, "then_branch", None),
                    dict(aliases),
                    control_depth + 1,
                )
                visit(
                    getattr(value, "else_branch", None),
                    dict(aliases),
                    control_depth + 1,
                )
                return
            if isinstance(value, TernaryOpNode):
                binding = self.hlsl_static_workgroup_pointer_binding(
                    value, aliases, constants
                )
                if binding is not None:
                    record_observation(binding, value)
                    visit(value.condition, aliases, control_depth)
                    return
                condition_value = self.literal_int_value(value.condition, constants)
                visit(value.condition, aliases, control_depth)
                if condition_value is not None:
                    selected = value.true_expr if condition_value else value.false_expr
                    visit(selected, aliases, control_depth)
                    return
                visit(value.true_expr, aliases, control_depth)
                visit(value.false_expr, aliases, control_depth)
                return
            if isinstance(value, ForNode):
                loop_aliases = dict(aliases)
                visit(getattr(value, "init", None), loop_aliases, control_depth + 1)
                saved_intervals = dict(intervals)
                saved_constants = dict(constants)
                init = getattr(value, "init", None)
                loop_name = getattr(init, "name", None)
                initial = self.hlsl_private_pointer_interval(
                    getattr(init, "initial_value", None), intervals, constants
                )
                if loop_name:
                    constants.pop(loop_name, None)
                    intervals.pop(loop_name, None)
                    if initial is not None:
                        intervals[loop_name] = initial
                    _, loop_interval = self.hlsl_private_pointer_loop_interval(
                        value, intervals, constants
                    )
                    condition_names = {
                        self.expression_name(item)
                        for item in self.walk_ast(getattr(value, "condition", None))
                        if isinstance(item, (IdentifierNode, VariableNode))
                    }
                    for item in self.walk_ast(getattr(value, "body", None)):
                        changed = None
                        if isinstance(item, AssignmentNode):
                            changed = self.expression_name(item.target)
                        elif isinstance(item, UnaryOpNode) and item.op in {
                            "++",
                            "--",
                            "&",
                        }:
                            changed = self.expression_name(item.operand)
                        elif isinstance(item, VariableNode):
                            changed = item.name
                        elif isinstance(item, FunctionCallNode):
                            if any(
                                self.expression_name(argument) in condition_names
                                for argument in getattr(item, "arguments", []) or []
                            ):
                                loop_interval = None
                        if changed in condition_names:
                            loop_interval = None
                    if loop_interval is None:
                        intervals.pop(loop_name, None)
                    else:
                        intervals[loop_name] = loop_interval
                visit(
                    getattr(value, "condition", None),
                    loop_aliases,
                    control_depth + 1,
                )
                visit(getattr(value, "body", None), loop_aliases, control_depth + 1)
                visit(getattr(value, "update", None), loop_aliases, control_depth + 1)
                intervals.clear()
                intervals.update(saved_intervals)
                constants.clear()
                constants.update(saved_constants)
                return
            if isinstance(value, (ForInNode, WhileNode, DoWhileNode, LoopNode)):
                loop_aliases = dict(aliases)
                visit(
                    getattr(value, "iterable", None),
                    loop_aliases,
                    control_depth + 1,
                )
                visit(
                    getattr(value, "condition", None),
                    loop_aliases,
                    control_depth + 1,
                )
                visit(getattr(value, "body", None), loop_aliases, control_depth + 1)
                return
            if isinstance(value, UnaryOpNode) and value.op in {"++", "--"}:
                operand_name = self.expression_name(value.operand)
                binding = aliases.get(operand_name)
                if binding is not None and binding.get("kind") == "workgroup-pointer":
                    raise self.hlsl_workgroup_pointer_error(
                        "DirectX cannot lower workgroup pointer operator "
                        f"'{value.op}' for '{operand_name}'",
                        function_name=function_name,
                        parameter_name=operand_name,
                        reason="assignment-operator-unsupported",
                        node=value,
                    )
            if isinstance(value, IdentifierNode):
                binding = self.hlsl_static_workgroup_pointer_binding(
                    value, aliases, constants
                )
                if binding is not None:
                    record_observation(binding, value)
                return
            if not hasattr(value, "__dict__"):
                return
            for key, child in vars(value).items():
                if key in {"parent", "annotations", "name", "args"}:
                    continue
                visit(child, aliases, control_depth)

        visit(getattr(function, "body", []), initial_aliases)
        return direct_ranges, direct_observations

    def collect_private_pointer_array_size_hints(self, ast):
        functions = self.collect_functions(ast)
        functions_by_name = {
            getattr(function, "name", None): function
            for function in functions
            if not generic_function_parameters(function)
        }
        functions_by_name.pop(None, None)
        self.hlsl_private_pointer_fixed_array_parameters = (
            self.collect_hlsl_private_pointer_fixed_array_parameters(functions_by_name)
        )

        pointer_parameters = {}
        pointer_parameter_indices = {}
        pointer_base_names = {}
        for function_name, function in functions_by_name.items():
            parameters = list(getattr(function, "parameters", []) or [])
            private_parameters = [
                parameter
                for parameter in parameters
                if self.hlsl_private_pointer_parameter(parameter, function_name)
                and self.hlsl_entry_resource_parameter_global_type(parameter, function)
                is None
            ]
            if not private_parameters:
                continue
            pointer_parameters[function_name] = private_parameters
            pointer_parameter_indices[function_name] = {
                index: parameter.name
                for index, parameter in enumerate(parameters)
                if self.hlsl_private_pointer_parameter(parameter, function_name)
                and self.hlsl_entry_resource_parameter_global_type(parameter, function)
                is None
            }
            used_names = {getattr(parameter, "name", None) for parameter in parameters}
            used_names.update(self.collect_hlsl_function_identifier_names(function))
            base_names = {}
            for parameter in private_parameters:
                base_name = self.hlsl_unique_local_identifier(
                    f"{parameter.name}_base", used_names
                )
                used_names.add(base_name)
                base_names[parameter.name] = base_name
            pointer_base_names[function_name] = base_names

        self.function_private_pointer_parameters = pointer_parameters
        self.function_private_pointer_parameter_indices = pointer_parameter_indices
        self.function_private_pointer_base_names = pointer_base_names
        if not pointer_parameters:
            return {}

        local_array_extents = {
            function_name: self.hlsl_private_pointer_local_array_extents(function)
            for function_name, function in functions_by_name.items()
        }
        self.function_private_pointer_local_array_extents = local_array_extents
        self.function_private_pointer_local_backing_types = {
            function_name: self.hlsl_private_pointer_local_backing_types(function)
            for function_name, function in functions_by_name.items()
        }

        direct_spans, view_calls = self.hlsl_analyze_private_pointer_functions(
            functions_by_name, pointer_parameters
        )

        scalar_parameters = {
            function_name: set() for function_name in pointer_parameters
        }
        changed = True
        while changed:
            changed = False
            for caller_name, calls in view_calls.items():
                caller_scalar_parameters = scalar_parameters.get(caller_name, set())
                for call in calls:
                    for parameter_name, binding in call["bindings"].items():
                        is_scalar = binding.get("root_kind") == "scalar"
                        if binding.get("root_kind") == "parameter":
                            is_scalar = binding.get("root") in caller_scalar_parameters
                        if is_scalar and parameter_name not in scalar_parameters.get(
                            call["callee"], set()
                        ):
                            scalar_parameters[call["callee"]].add(parameter_name)
                            changed = True
        self.function_private_pointer_scalar_parameters = scalar_parameters
        aliased = hlsl_aliased_scalar_view(
            view_calls, scalar_parameters
        ) or aliased_row_view(view_calls)
        if aliased is None:
            aliased = hlsl_aliased_writable_view(
                view_calls,
                hlsl_writable_pointer_parameters(self, pointer_parameters),
            )
        if aliased is not None:
            call, first, second, root = aliased
            raise DirectXPrivatePointerParameterError(
                f"DirectX private pointer call to '{call['callee']}' binds "
                f"parameters '{first}' and '{second}' to the same backing object "
                f"'{root}'; HLSL inout copy-out cannot preserve aliased views",
                function_name=call["callee"],
                parameter_name=second,
                reason="aliased-view-backing",
                source_location=getattr(call.get("node"), "source_location", None),
            )

        for calls in view_calls.values():
            for call in calls:
                for parameter_name, binding in call["bindings"].items():
                    if parameter_name not in scalar_parameters.get(
                        call["callee"], set()
                    ):
                        continue
                    binding_is_scalar = binding.get("root_kind") == "scalar"
                    if binding.get("root_kind") == "parameter":
                        binding_is_scalar = binding.get(
                            "root"
                        ) in scalar_parameters.get(call["caller"], set())
                    if binding_is_scalar:
                        continue
                    raise DirectXPrivatePointerParameterError(
                        "DirectX private pointer parameter "
                        f"'{call['callee']}.{parameter_name}' is called with both "
                        "scalar and array backing objects",
                        function_name=call["callee"],
                        parameter_name=parameter_name,
                        reason="mixed-scalar-array-backing",
                        source_location=getattr(
                            call.get("node"), "source_location", None
                        ),
                    )

        for calls in view_calls.values():
            for call in calls:
                full_span_parameters = (
                    self.function_private_pointer_full_span_parameters.get(
                        call["callee"], set()
                    )
                )
                for parameter_name in full_span_parameters:
                    binding = call["bindings"].get(parameter_name, {})
                    if binding.get("root_kind") in {
                        "local",
                        "local_word_view",
                    } and binding.get("offset") == (0, 0):
                        direct_spans[call["callee"]][parameter_name] = max(
                            direct_spans[call["callee"]][parameter_name],
                            binding["extent"],
                        )

        required_spans = {
            function_name: dict(spans) for function_name, spans in direct_spans.items()
        }
        self.function_private_pointer_required_spans = required_spans
        max_iterations = max(
            1,
            sum(len(parameters) for parameters in pointer_parameters.values())
            * max(len(functions_by_name), 1),
        )
        for _iteration in range(max_iterations):
            changed = False
            for caller_name, calls in view_calls.items():
                caller_spans = required_spans.get(caller_name, {})
                for call in calls:
                    callee_spans = required_spans.get(call["callee"], {})
                    for parameter_name, binding in call["bindings"].items():
                        span = callee_spans.get(parameter_name, 0)
                        if span <= 0 or binding.get("root_kind") != "parameter":
                            continue
                        offset = self.hlsl_private_pointer_binding_interval(
                            call,
                            parameter_name,
                            binding,
                            required_span=span,
                        )
                        needed = offset[1] + span
                        root_name = binding["root"]
                        if needed > caller_spans.get(root_name, 0):
                            caller_spans[root_name] = needed
                            changed = True
            if not changed:
                break
        else:
            raise DirectXPrivatePointerParameterError(
                "DirectX private pointer view requirements do not converge",
                reason="recursive-view-span",
            )

        for function_name, parameters in pointer_parameters.items():
            for parameter in parameters:
                if required_spans[function_name].get(parameter.name, 0) > 0:
                    continue
                raise DirectXPrivatePointerParameterError(
                    "DirectX private pointer parameter "
                    f"'{function_name}.{parameter.name}' has no provable bounded span",
                    function_name=function_name,
                    parameter_name=parameter.name,
                    reason="missing-fixed-array-extent",
                    source_location=getattr(parameter, "source_location", None),
                )

        self.function_private_pointer_required_spans = required_spans
        self.function_private_pointer_view_calls = view_calls
        variants = {
            function_name: {
                tuple(
                    required_spans[function_name][parameter.name]
                    for parameter in parameters
                )
            }
            for function_name, parameters in pointer_parameters.items()
        }

        for calls in view_calls.values():
            for call in calls:
                callee_scalar_parameters = scalar_parameters.get(call["callee"], set())
                if not all(
                    binding.get("root_kind") in {"local", "local_word_view"}
                    or binding.get("root_kind") == "scalar"
                    and parameter_name in callee_scalar_parameters
                    for parameter_name, binding in call["bindings"].items()
                ):
                    continue
                variants[call["callee"]].add(
                    tuple(
                        (
                            call["bindings"][parameter.name]["word_view"]
                            if call["bindings"][parameter.name].get("root_kind")
                            == "local_word_view"
                            else call["bindings"][parameter.name]["extent"]
                        )
                        for parameter in pointer_parameters[call["callee"]]
                    )
                )

        changed = True
        while changed:
            changed = False
            for caller_name, calls in view_calls.items():
                caller_parameters = pointer_parameters.get(caller_name, [])
                caller_parameter_positions = {
                    parameter.name: index
                    for index, parameter in enumerate(caller_parameters)
                }
                caller_variants = variants.get(caller_name, {()})
                for caller_variant in tuple(caller_variants):
                    for call in calls:
                        callee_variant = []
                        for parameter in pointer_parameters[call["callee"]]:
                            binding = call["bindings"][parameter.name]
                            if binding.get("root_kind") == "local":
                                extent = binding["extent"]
                                variant_value = extent
                            elif binding.get("root_kind") == "local_word_view":
                                extent = binding["extent"]
                                variant_value = binding["word_view"]
                            elif binding.get(
                                "root_kind"
                            ) == "scalar" and parameter.name in scalar_parameters.get(
                                call["callee"], set()
                            ):
                                extent = 1
                                variant_value = extent
                            elif binding.get("root_kind") == "parameter":
                                position = caller_parameter_positions[binding["root"]]
                                variant_value = caller_variant[position]
                                extent = self.hlsl_private_pointer_variant_extent(
                                    variant_value
                                )
                            else:
                                self.hlsl_private_pointer_backing_error(
                                    call, parameter.name, binding
                                )
                            self.hlsl_validate_private_pointer_call_binding(
                                caller_name,
                                caller_variant,
                                call,
                                parameter.name,
                                binding,
                                extent,
                                caller_parameter_positions,
                            )
                            callee_variant.append(variant_value)
                        callee_variant = tuple(callee_variant)
                        if callee_variant not in variants[call["callee"]]:
                            variants[call["callee"]].add(callee_variant)
                            changed = True

        for function_name, parameters in pointer_parameters.items():
            incoming_calls = [
                call
                for calls in view_calls.values()
                for call in calls
                if call["callee"] == function_name
            ]
            if incoming_calls and all(
                all(
                    call["bindings"][parameter.name].get("root_kind")
                    == "local_word_view"
                    for parameter in parameters
                )
                for call in incoming_calls
            ):
                variants[function_name].discard(
                    tuple(
                        required_spans[function_name][parameter.name]
                        for parameter in parameters
                    )
                )
        variants = self.hlsl_private_pointer_emitted_variants(
            functions_by_name,
            pointer_parameters,
            scalar_parameters,
            required_spans,
            view_calls,
            variants,
        )

        self.function_private_pointer_backing_variants = {
            function_name: tuple(
                sorted(
                    function_variants,
                    key=lambda variant: tuple(
                        (0, value) if isinstance(value, int) else (1, repr(value))
                        for value in variant
                    ),
                )
            )
            for function_name, function_variants in variants.items()
        }
        return {
            function_name: {
                parameter.name: str(
                    self.hlsl_private_pointer_variant_extent(
                        self.function_private_pointer_backing_variants[function_name][
                            0
                        ][index]
                    )
                )
                for index, parameter in enumerate(parameters)
            }
            for function_name, parameters in pointer_parameters.items()
        }

    def hlsl_analyze_private_pointer_function(
        self, function, functions_by_name, direct_spans, calls
    ):
        function_name = getattr(function, "name", None)
        parameter_names = set(direct_spans)
        local_extents = self.function_private_pointer_local_array_extents.get(
            function_name, {}
        )
        local_types = self.function_private_pointer_local_backing_types.get(
            function_name, {}
        )
        constants = self.initial_literal_int_constants(function)
        signed_interval_types = {
            "int",
            "int16_t",
            "int64_t",
            "min12int",
            "min16int",
        }
        condition_interval_names = set()
        unsafe_condition_names = {
            getattr(parameter, "name", None)
            for parameter in getattr(function, "parameters", []) or []
        }
        for node in self.walk_ast(getattr(function, "body", [])):
            if not isinstance(node, VariableNode):
                continue
            name = getattr(node, "name", None)
            qualifiers = {
                str(qualifier).lower()
                for qualifier in getattr(node, "qualifiers", []) or []
            }
            if getattr(node, "is_mutable", True) and not qualifiers.intersection(
                {"const", "constant", "constexpr"}
            ):
                condition_interval_names.discard(name)
                unsafe_condition_names.add(name)
                continue
            type_info = self.hlsl_integer_arithmetic_type_info(
                getattr(node, "var_type", getattr(node, "vtype", None))
            )
            if (
                type_info is not None
                and type_info["width"] == 1
                and type_info["base_type"] in signed_interval_types
                and name not in unsafe_condition_names
            ):
                condition_interval_names.add(name)
            else:
                condition_interval_names.discard(name)
                unsafe_condition_names.add(name)
        condition_interval_names.update(
            name
            for name, value in constants.items()
            if not isinstance(value, _UnsignedLiteralInt)
            and name not in unsafe_condition_names
        )
        condition_interval_names.discard(None)
        self.current_hlsl_private_pointer_condition_interval_names = (
            condition_interval_names
        )
        intervals = {}
        for name, value in constants.items():
            literal = self.literal_int_value(value, constants)
            if literal is not None:
                intervals[name] = (literal, literal)
        # Integer parameters every call bounds; see
        # hlsl_private_pointer_call_intervals.
        intervals.update(
            self.hlsl_private_pointer_parameter_seeds.get(function_name, {})
        )
        offset_keys = tuple(
            (HLSL_PRIVATE_POINTER_OFFSET_KEY, name) for name in sorted(parameter_names)
        )
        for key in offset_keys:
            intervals[key] = (0, 0)
        statement_assignment_ids = self.hlsl_private_pointer_statement_assignment_ids(
            function
        )
        rebase_state = {"unsafe_depth": 0, "simulation_steps": 0}

        def contains_rebase(node):
            return any(
                self.hlsl_private_pointer_rebase_target(child, parameter_names)
                is not None
                for child in self.walk_ast(node)
            )

        def reject_rebase(node, detail):
            self.hlsl_private_pointer_rejected_rebases.setdefault(id(node), detail)

        def visit_rebase(node, rebased_name, active_intervals):
            assigned = getattr(node, "value", getattr(node, "right", None))
            visit(assigned, active_intervals)
            key = (HLSL_PRIVATE_POINTER_OFFSET_KEY, rebased_name)
            current = active_intervals.get(key)
            if id(node) not in statement_assignment_ids:
                return reject_rebase(node, "the rebased pointer value is used")
            if rebase_state["unsafe_depth"]:
                return reject_rebase(
                    node,
                    "the rebase is not in straight-line code or in a loop with "
                    "statically enumerable iterations",
                )
            if self.hlsl_private_pointer_expression_has_side_effects(assigned):
                return reject_rebase(node, "the rebase offset has side effects")
            delta = self.hlsl_private_pointer_interval(
                assigned, active_intervals, constants
            )
            if delta is None or current is None:
                return reject_rebase(
                    node, "the rebase offset is not statically bounded"
                )
            if self.map_operator(getattr(node, "operator", "=")) == "-=":
                delta = (-delta[1], -delta[0])
            replacement = (current[0] + delta[0], current[1] + delta[1])
            if replacement[0] < 0:
                return reject_rebase(
                    node, f"the view offset may become negative ({replacement[0]})"
                )
            if replacement[1] > HLSL_PRIVATE_POINTER_BASE_MAX:
                return reject_rebase(
                    node, "the view offset may overflow the 32-bit view base"
                )
            self.hlsl_private_pointer_rebase_ids.add(id(node))
            if delta != (0, 0):
                self.hlsl_private_pointer_nonzero_rebase_ids.add(id(node))
            active_intervals[key] = replacement
            return None

        def visit_unsafe_rebase_scope(node, callback):
            unsafe = contains_rebase(node)
            if unsafe:
                rebase_state["unsafe_depth"] += 1
            try:
                callback()
            finally:
                if unsafe:
                    rebase_state["unsafe_depth"] -= 1

        def visit_enumerated_rebase_loop(node, loop_intervals):
            if rebase_state["unsafe_depth"]:
                return None
            iteration_values = self.hlsl_private_pointer_rebase_loop_iteration_values(
                node, loop_intervals, constants, functions_by_name
            )
            if iteration_values is None or (
                rebase_state["simulation_steps"] + len(iteration_values)
                > HLSL_PRIVATE_POINTER_MAX_REBASE_SIMULATION_STEPS
            ):
                return None
            rebase_state["simulation_steps"] += len(iteration_values)
            loop_name = getattr(getattr(node, "init", None), "name", None)
            states = [{key: loop_intervals.get(key) for key in offset_keys}]
            for iteration_value in iteration_values:
                loop_intervals[loop_name] = (iteration_value, iteration_value)
                visit(getattr(node, "condition", None), loop_intervals)
                visit(getattr(node, "body", None), loop_intervals)
                visit(getattr(node, "update", None), loop_intervals)
                states.append({key: loop_intervals.get(key) for key in offset_keys})
            return states

        def record_access(binding, index_expression, active_intervals, source):
            if binding is None or binding.get("root_kind") != "parameter":
                return
            index = self.hlsl_private_pointer_interval(
                index_expression, active_intervals, constants
            )
            offset = binding.get("offset")
            if index is None or offset is None:
                self.function_private_pointer_full_span_parameters.setdefault(
                    function_name, set()
                ).add(binding["root"])
                return
            lower = offset[0] + index[0]
            upper = offset[1] + index[1]
            if lower < 0:
                raise DirectXPrivatePointerParameterError(
                    "DirectX private pointer access "
                    f"'{function_name}.{binding['root']}' may use negative index "
                    f"{lower}",
                    function_name=function_name,
                    parameter_name=binding["root"],
                    reason="negative-view-access",
                    source_location=getattr(source, "source_location", None),
                )
            if lower == 0 and upper == 0:
                self.hlsl_private_pointer_zero_index_access_ids.add(id(source))
            direct_spans[binding["root"]] = max(
                direct_spans[binding["root"]], upper + 1
            )

        def merge_intervals(left, right):
            return {
                name: (
                    min(left[name][0], right[name][0]),
                    max(left[name][1], right[name][1]),
                )
                for name in left.keys() & right.keys()
            }

        def visit(value, active_intervals):
            if value is None or isinstance(value, (str, int, float, bool)):
                return
            if isinstance(value, (list, tuple)):
                for item in value:
                    visit(item, active_intervals)
                return
            if isinstance(value, BlockNode):
                block_intervals = dict(active_intervals)
                statements = list(getattr(value, "statements", []) or [])
                outer_names = set(active_intervals)
                missing = object()
                shadowed_outer_intervals = {}
                for statement in statements:
                    declared_name = (
                        getattr(statement, "name", None)
                        if isinstance(statement, (VariableNode, ArrayNode))
                        else None
                    )
                    if (
                        declared_name in outer_names
                        and declared_name not in shadowed_outer_intervals
                    ):
                        shadowed_outer_intervals[declared_name] = block_intervals.get(
                            declared_name, missing
                        )
                        block_intervals.pop(declared_name, None)
                    visit(statement, block_intervals)
                for name in outer_names:
                    interval = shadowed_outer_intervals.get(
                        name, block_intervals.get(name, missing)
                    )
                    if interval is not missing:
                        active_intervals[name] = interval
                    else:
                        active_intervals.pop(name, None)
                return
            if isinstance(value, VariableNode):
                initial_value = getattr(value, "initial_value", None)
                visit(initial_value, active_intervals)
                interval = self.hlsl_private_pointer_interval(
                    initial_value, active_intervals, constants
                )
                if interval is None:
                    active_intervals.pop(value.name, None)
                else:
                    active_intervals[value.name] = interval
                return
            if isinstance(value, AssignmentNode):
                rebased_name = self.hlsl_private_pointer_rebase_target(
                    value, parameter_names
                )
                if rebased_name is not None:
                    visit_rebase(value, rebased_name, active_intervals)
                    return
                target = getattr(value, "target", getattr(value, "left", None))
                assigned = getattr(value, "value", getattr(value, "right", None))
                visit(target, active_intervals)
                visit(assigned, active_intervals)
                target_name = self.expression_name(target)
                if target_name and isinstance(
                    target, (str, IdentifierNode, VariableNode)
                ):
                    assigned_interval = self.hlsl_private_pointer_interval(
                        assigned, active_intervals, constants
                    )
                    operator = getattr(value, "operator", "=")
                    current = active_intervals.get(target_name)
                    if operator == "=":
                        replacement = assigned_interval
                    elif operator == "+=" and current and assigned_interval:
                        replacement = (
                            current[0] + assigned_interval[0],
                            current[1] + assigned_interval[1],
                        )
                    elif operator == "-=" and current and assigned_interval:
                        replacement = (
                            current[0] - assigned_interval[1],
                            current[1] - assigned_interval[0],
                        )
                    else:
                        replacement = None
                    if replacement is None:
                        active_intervals.pop(target_name, None)
                    else:
                        active_intervals[target_name] = replacement
                return
            if isinstance(value, ArrayAccessNode):
                binding = self.hlsl_private_pointer_static_binding(
                    value.array,
                    parameter_names,
                    local_extents,
                    active_intervals,
                    constants,
                    local_types,
                )
                record_access(binding, value.index, active_intervals, value)
                visit(value.array, active_intervals)
                visit(value.index, active_intervals)
                return
            if isinstance(value, UnaryOpNode):
                if value.op == "*":
                    binding = self.hlsl_private_pointer_static_binding(
                        value.operand,
                        parameter_names,
                        local_extents,
                        active_intervals,
                        constants,
                        local_types,
                    )
                    record_access(binding, 0, active_intervals, value)
                visit(value.operand, active_intervals)
                return
            if isinstance(value, FunctionCallNode):
                callee_name = self.hlsl_private_pointer_call_callee(value)
                arguments = list(getattr(value, "arguments", []) or [])
                parameter_indices = self.function_private_pointer_parameter_indices.get(
                    callee_name, {}
                )
                if parameter_indices:
                    bindings = {}
                    for index, parameter_name in parameter_indices.items():
                        argument = arguments[index] if index < len(arguments) else None
                        binding = self.hlsl_private_pointer_static_binding(
                            argument,
                            parameter_names,
                            local_extents,
                            active_intervals,
                            constants,
                            local_types,
                        )
                        if binding is None:
                            binding = {
                                "root_kind": None,
                                "root": self.expression_name(argument),
                                "offset": None,
                                "side_effecting": self.hlsl_private_pointer_expression_has_side_effects(
                                    argument
                                ),
                            }
                        bindings[parameter_name] = binding
                    calls.append(
                        {
                            "caller": function_name,
                            "callee": callee_name,
                            "node": value,
                            "bindings": bindings,
                            "argumentIntervals": hlsl_private_pointer_call_intervals.call_argument_intervals(
                                self.hlsl_private_pointer_signed_slots(
                                    functions_by_name.get(callee_name),
                                    parameter_indices,
                                ),
                                arguments,
                                lambda argument: self.hlsl_private_pointer_interval(
                                    argument, active_intervals, constants
                                ),
                            ),
                        }
                    )
                for argument in arguments:
                    visit(argument, active_intervals)
                return
            if isinstance(value, IfNode):
                condition_node = getattr(value, "condition", None)
                condition = self.hlsl_private_pointer_condition_value(
                    condition_node, active_intervals, constants
                )
                visit(condition_node, active_intervals)
                if condition is not None:
                    selected_branch = (
                        getattr(value, "then_branch", None)
                        if condition
                        else getattr(value, "else_branch", None)
                    )
                    skipped_branch = (
                        getattr(value, "else_branch", None)
                        if condition
                        else getattr(value, "then_branch", None)
                    )
                    # A proven-dead branch never moves the view; its rebases
                    # may still be emitted when HLSL keeps the branch.
                    for child in self.walk_ast(skipped_branch):
                        if (
                            self.hlsl_private_pointer_rebase_target(
                                child, parameter_names
                            )
                            is not None
                            and id(child) in statement_assignment_ids
                        ):
                            self.hlsl_private_pointer_rebase_ids.add(id(child))
                            self.hlsl_private_pointer_nonzero_rebase_ids.add(id(child))
                    visit(selected_branch, active_intervals)
                    return
                then_intervals = dict(active_intervals)
                else_intervals = dict(active_intervals)
                visit(getattr(value, "then_branch", None), then_intervals)
                visit(getattr(value, "else_branch", None), else_intervals)
                active_intervals.clear()
                active_intervals.update(merge_intervals(then_intervals, else_intervals))
                return
            if isinstance(value, ForNode):
                loop_intervals = dict(active_intervals)
                visit(getattr(value, "init", None), loop_intervals)
                has_rebase = contains_rebase(value)
                if has_rebase:
                    states = visit_enumerated_rebase_loop(value, loop_intervals)
                    if states is not None:
                        # The loop may stop after any enumerated iteration, so
                        # the view start after it is the hull of every state.
                        for key in offset_keys:
                            bounds = [state[key] for state in states]
                            active_intervals[key] = (
                                min(bound[0] for bound in bounds),
                                max(bound[1] for bound in bounds),
                            )
                        return
                    rebase_state["unsafe_depth"] += 1
                try:
                    loop_name, loop_interval = self.hlsl_private_pointer_loop_interval(
                        value, loop_intervals, constants
                    )
                    if loop_name:
                        if loop_interval is None:
                            loop_intervals.pop(loop_name, None)
                        else:
                            loop_intervals[loop_name] = loop_interval
                    visit(getattr(value, "condition", None), loop_intervals)
                    visit(getattr(value, "body", None), loop_intervals)
                finally:
                    if has_rebase:
                        rebase_state["unsafe_depth"] -= 1
                return
            if isinstance(value, ForInNode):

                def visit_for_in():
                    loop_intervals = dict(active_intervals)
                    iterable = getattr(value, "iterable", None)
                    start = self.hlsl_private_pointer_interval(
                        getattr(iterable, "start", None), loop_intervals, constants
                    )
                    end = self.hlsl_private_pointer_interval(
                        getattr(iterable, "end", None), loop_intervals, constants
                    )
                    pattern = getattr(value, "pattern", None)
                    if pattern and start and end:
                        maximum = end[1] - (
                            0 if getattr(iterable, "inclusive", False) else 1
                        )
                        loop_intervals[pattern] = (start[0], maximum)
                    elif pattern:
                        loop_intervals.pop(pattern, None)
                    visit(getattr(value, "body", None), loop_intervals)

                visit_unsafe_rebase_scope(value, visit_for_in)
                return
            if isinstance(value, (WhileNode, DoWhileNode, LoopNode)):

                def visit_loop():
                    loop_intervals = dict(active_intervals)
                    visit(getattr(value, "condition", None), loop_intervals)
                    visit(getattr(value, "body", None), loop_intervals)

                visit_unsafe_rebase_scope(value, visit_loop)
                return
            if not hasattr(value, "__dict__"):
                return

            def visit_children():
                for key, child in vars(value).items():
                    if key in {
                        "parent",
                        "annotations",
                        "array",
                        "index",
                        "args",
                        "name",
                    }:
                        continue
                    visit(child, active_intervals)

            if isinstance(value, (SwitchNode, MatchNode)):
                visit_unsafe_rebase_scope(value, visit_children)
                return
            visit_children()

        visit(getattr(function, "body", []), intervals)
        self.current_hlsl_private_pointer_condition_interval_names = set()

    def hlsl_matrix_aggregate_initializer(self, expr, mapped_type):
        """Lower ``{...}`` for a CrossGL matrix.

        Nested lists are columns (GLSL 4.60 section 4.1.11) and a flat list fills the columns in
        order; missing trailing elements are zero. HLSL constructors fill rows, so the columns are
        built as the transposed matrix, which keeps the elements in evaluation order.
        """
        dimensions = self.hlsl_matrix_dimensions(mapped_type)
        if dimensions is None:
            return None
        if self.hlsl_is_zero_aggregate_initializer(expr):
            return self.hlsl_value_initialized_expression(expr, mapped_type)

        component_type, rows, columns = dimensions
        elements = list(getattr(expr, "elements", []) or [])
        nested_count = sum(
            isinstance(element, ArrayLiteralNode) for element in elements
        )
        if nested_count not in {0, len(elements)}:
            self.hlsl_aggregate_error(
                expr,
                expected_type=mapped_type,
                reason="matrix-shape-mismatch",
                detail="matrix columns cannot mix nested and flat initializer elements",
            )

        rendered = []
        if nested_count or not elements:
            if len(elements) > columns:
                self.hlsl_aggregate_error(
                    expr,
                    expected_type=mapped_type,
                    reason="element-count-mismatch",
                    expected_element_count=columns,
                )
            for column in elements:
                column_elements = list(getattr(column, "elements", []) or [])
                if len(column_elements) > rows:
                    self.hlsl_aggregate_error(
                        column,
                        expected_type=mapped_type,
                        reason="element-count-mismatch",
                        expected_element_count=rows,
                    )
                rendered.extend(
                    self.generate_expression_with_expected(element, component_type)
                    for element in column_elements
                )
                rendered.extend(
                    self.hlsl_value_initialized_expression(column, component_type)
                    for _index in range(len(column_elements), rows)
                )
            rendered.extend(
                self.hlsl_value_initialized_expression(expr, component_type)
                for _index in range(len(elements) * rows, rows * columns)
            )
        else:
            expected_count = rows * columns
            if len(elements) > expected_count:
                self.hlsl_aggregate_error(
                    expr,
                    expected_type=mapped_type,
                    reason="element-count-mismatch",
                    expected_element_count=expected_count,
                )
            rendered.extend(
                self.generate_expression_with_expected(element, component_type)
                for element in elements
            )
            rendered.extend(
                self.hlsl_value_initialized_expression(expr, component_type)
                for _index in range(len(elements), expected_count)
            )
        transposed_type = f"{component_type}{columns}x{rows}"
        return f"transpose({transposed_type}({', '.join(rendered)}))"

    def generate_texture_call(self, func_name, args, *, call_node=None):
        if not func_name:
            return None
        texel_buffer_call = self.hlsl_texel_buffers.generate_call(func_name, args)
        if texel_buffer_call is not None:
            return texel_buffer_call
        if is_texture_lod_clamp_operation(func_name):
            return hlsl_texture_lod_clamp_call(self, func_name, args, call_node)
        if func_name == TEXTURE_RESIDENCY_OPERATION:
            return hlsl_texture_residency_call(self, args, call_node)
        if func_name == TEXTURE_RESIDENT_OPERATION:
            return hlsl_texture_resident_call(self, args, call_node)

        feedback_call = self.generate_hlsl_feedback_write_call(func_name, args)
        if feedback_call is not None:
            return feedback_call
        if (
            func_name != "textureSamplePosition"
            and func_name not in self.texture_resource_operation_names()
        ):
            return None

        self.validate_texture_call_arity(func_name, args)
        self.validate_image_resource_argument(func_name, args)
        self.validate_image_access_argument(func_name, args)
        self.validate_texture_resource_argument(func_name, args)
        self.validate_image_multisample_arguments(func_name, args)
        self.validate_image_atomic_format_argument(func_name, args)
        self.validate_integer_coordinate_argument(func_name, args)
        self.validate_coordinate_dimension_argument(func_name, args)
        self.validate_query_lod_coordinate_argument(func_name, args)
        self.validate_compare_argument(func_name, args)
        self.validate_lod_argument(func_name, args)
        self.validate_bias_argument(func_name, args)
        self.validate_sample_index_argument(func_name, args)
        self.validate_mip_level_argument(func_name, args)
        self.validate_gradient_dimension_arguments(func_name, args)
        self.validate_offset_dimension_argument(func_name, args)
        self.validate_gather_component_argument(func_name, args)

        if is_resource_size_query_operation(func_name) and args:
            lod_arg = args[1] if len(args) > 1 else None
            return self.texture_query_size_expression(args[0], lod_arg)

        if is_texture_query_levels_operation(func_name) and args:
            return self.texture_query_levels_expression(args[0])

        if is_resource_samples_query_operation(func_name) and args:
            return self.texture_samples_expression(args[0])

        if func_name == "textureSamplePosition":
            if len(args) != 2:
                return self.refuse_lowering(
                    "texture sample-position query: textureSamplePosition requires "
                    "texture and sample-index arguments",
                    "float2(0.0, 0.0)",
                )
            return self.texture_sample_position_expression(args[0], args[1])

        if is_texture_query_lod_operation(func_name) and len(args) >= 2:
            parts = self.texture_call_parts(args, func_name)
            if parts is None:
                return None
            texture_name, sampler_name, coord, _ = parts
            texture_type = self.texture_resource_type(args[0])
            if self.is_multisample_texture_resource_type(texture_type):
                return self.unsupported_multisample_texture_query_lod_call(texture_type)
            if self.is_storage_image_resource_type(texture_type):
                return self.unsupported_texture_query_lod_call(texture_type)
            if not self.is_explicit_sampler_argument(args):
                sampler_name = self.generate_implicit_query_lod_sampler_argument(
                    args[0]
                )
            coord = self.texture_query_lod_coordinate(texture_type, coord)
            return (
                f"float2({texture_name}.CalculateLevelOfDetail({sampler_name}, {coord}), "
                f"{texture_name}.CalculateLevelOfDetailUnclamped({sampler_name}, {coord}))"
            )

        if is_image_atomic_operation(func_name):
            return self.image_atomic_expression(func_name, args)

        if func_name == "imageLoad" and len(args) >= 2:
            swizzle_extent = self.hlsl_image_load_swizzle_extent
            self.hlsl_image_load_swizzle_extent = None
            image_name = self.generate_expression(args[0])
            coord = self.generate_expression(args[1])
            texture_type = self.texture_resource_type(args[0])
            if len(args) >= 3 and self.is_multisample_storage_image_resource_type(
                texture_type
            ):
                sample = self.generate_expression(args[2])
                load_expr = f"{image_name}.Load({coord}, {sample})"
            else:
                load_expr = f"{image_name}[{coord}]"
            image_format = self.image_resource_format(args[0])
            self.validate_image_load_result_type(texture_type, image_format)
            # A CrossGL texel has four components; the HLSL element may have fewer.
            component_kind = self.image_load_component_kind(texture_type, image_format)
            element_count = self.hlsl_storage_image_element_count(texture_type)
            target_count = self.expected_component_count() or TEXEL_COMPONENTS
            if swizzle_extent is not None and element_count is not None:
                # A swizzle within the element reads it as it is.
                target_count = (
                    element_count
                    if swizzle_extent <= element_count
                    else TEXEL_COMPONENTS
                )
            return reshaped_texel_expression(
                component_kind,
                component_kind,
                load_expr,
                element_count,
                target_count,
            )

        if func_name == "imageStore" and len(args) >= 3:
            image_name = self.generate_expression(args[0])
            coord = self.generate_expression(args[1])
            texture_type = self.texture_resource_type(args[0])
            if len(args) >= 4 and self.is_multisample_storage_image_resource_type(
                texture_type
            ):
                value_arg = args[3]
            else:
                value_arg = args[2]
                store_target = f"{image_name}[{coord}]"
            image_format = self.image_resource_format(args[0])
            self.validate_image_store_value_type(texture_type, image_format, value_arg)
            if self.is_multisample_storage_image_resource_type(texture_type):
                return self.unsupported_multisample_image_store_call(texture_type)
            value = self.generate_expression_with_expected(
                value_arg,
                self.image_store_expected_value_type(
                    texture_type, image_format, value_arg
                ),
            )
            element_count = self.hlsl_storage_image_element_count(texture_type)
            value_count = self.expression_component_count(value_arg)
            if (
                element_count is not None
                and value_count is not None
                and value_count > 1
                and value_count != element_count
            ):
                # Keep the leading components the element holds, or fill the
                # components a narrower value lacks.
                component_kind = self.image_load_component_kind(
                    texture_type, image_format
                )
                if value_count > element_count:
                    value = f"({value})"
                value = reshaped_texel_expression(
                    component_kind,
                    component_kind,
                    value,
                    value_count,
                    element_count,
                )
                return f"{store_target} = {value}"
            four_component_constructor = self.four_component_image_store_constructor(
                texture_type
            )
            if four_component_constructor and self.is_scalar_value_type(
                self.expression_result_type(value_arg)
            ):
                if self.hlsl_expression_is_repeatable(value_arg):
                    component_count = self.value_component_count(
                        four_component_constructor
                    )
                    value = (
                        f"{four_component_constructor}"
                        f"({', '.join([value] * (component_count or 4))})"
                    )
                else:
                    value = self.hlsl_scalar_splat_cast(
                        four_component_constructor, value
                    )
            elif (
                four_component_constructor
                and self.expression_component_count(value_arg) == 3
            ):
                constructor_info = self.three_component_image_store_constructor(
                    texture_type
                )
                if constructor_info is not None:
                    constructor, zero_value = constructor_info
                    value = f"{constructor}({value}, {zero_value})"
            two_component_constructor = self.two_component_image_store_constructor(
                texture_type
            )
            if two_component_constructor and self.is_scalar_value_type(
                self.expression_result_type(value_arg)
            ):
                # A stored scalar is splatted (``storage_image_texels``).
                constructor, _zero_value = two_component_constructor
                if self.hlsl_expression_is_repeatable(value_arg):
                    value = f"{constructor}({value}, {value})"
                else:
                    value = self.hlsl_scalar_splat_cast(constructor, value)
            return f"{store_target} = {value}"

        if len(args) < 2:
            return None

        texture_type = self.texture_resource_type(args[0])
        storage_image_operation = self.storage_image_texture_operation_expression(
            func_name, texture_type
        )
        if storage_image_operation is not None:
            return storage_image_operation

        if is_texture_compare_operation(func_name):
            return self.generate_texture_compare_call(
                func_name, args, call_node=call_node
            )

        if is_texture_gather_compare_operation(func_name):
            return self.generate_texture_gather_compare_call(func_name, args)

        if is_texture_gather_operation(func_name):
            return self.generate_texture_gather_call(func_name, args)

        explicit_shadow_call = self.generate_explicit_shadow_sample_call(
            func_name, args
        )
        if explicit_shadow_call is not None:
            return explicit_shadow_call

        if is_texture_sample_offset_operation(func_name):
            return self.generate_texture_sample_offset_call(
                func_name, args, call_node=call_node
            )

        if is_projected_texture_operation(func_name):
            return self.generate_texture_projected_call(
                func_name, args, call_node=call_node
            )

        if is_texture_sample_operation(func_name):
            parts = self.texture_call_parts(args, func_name)
            if parts is None:
                return None
            texture_name, sampler_name, coord, extra_args = parts
            texture_type = self.texture_resource_type(args[0])
            if self.is_multisample_texture_resource_type(texture_type):
                return self.unsupported_multisample_texture_call(
                    func_name, texture_type
                )
            if is_texture_sample_basic_operation(func_name) and len(extra_args) == 1:
                bias = self.generate_expression(extra_args[0])
                return f"{texture_name}.SampleBias({sampler_name}, {coord}, {bias})"
            if is_texture_sample_basic_operation(func_name):
                method = "Sample"
            elif is_texture_sample_lod_operation(func_name):
                method = "SampleLevel"
            elif is_texture_sample_grad_operation(func_name):
                method = "SampleGrad"
            else:
                return None
            mapped_args = [coord] + [
                self.generate_expression(arg) for arg in extra_args
            ]
            return f"{texture_name}.{method}({sampler_name}, {', '.join(mapped_args)})"

        if is_texel_fetch_basic_operation(func_name) and len(args) >= 3:
            texture_name = self.generate_expression(args[0])
            coord = self.generate_expression(args[1])
            lod = self.generate_expression(args[2])
            texture_type = self.texture_resource_type(args[0])
            shape_type = self.sampled_texture_shape_type(texture_type)
            if self.is_cube_texture_resource_type(texture_type):
                return self.unsupported_cube_texel_fetch_call(func_name, texture_type)
            if self.is_multisample_texture_resource_type(texture_type):
                return f"{texture_name}.Load({coord}, {lod})"
            if shape_type == "Texture1D":
                load_coord_type = "int2"
            elif shape_type in {"Texture2DArray", "Texture3D"}:
                load_coord_type = "int4"
            else:
                load_coord_type = "int3"
            return f"{texture_name}.Load({load_coord_type}({coord}, {lod}))"

        if is_texel_fetch_offset_operation(func_name) and len(args) >= 4:
            texture_name = self.generate_expression(args[0])
            coord = self.generate_expression(args[1])
            lod = self.generate_expression(args[2])
            offset = self.generate_expression(args[3])
            texture_type = self.resource_base_type(self.texture_resource_type(args[0]))
            shape_type = self.sampled_texture_shape_type(texture_type)
            if self.is_cube_texture_resource_type(texture_type):
                return self.unsupported_cube_texel_fetch_call(func_name, texture_type)
            if self.is_multisample_texture_resource_type(texture_type):
                return unsupported_multisample_texel_fetch_offset_expression("DirectX")
            if shape_type == "Texture2DArray":
                coord_xy = self.vector_component(coord, "xy")
                layer = self.vector_component(coord, "z")
                return (
                    f"{texture_name}.Load("
                    f"int4(({coord_xy} + {offset}), {layer}, {lod}))"
                )
            if shape_type == "Texture3D":
                return f"{texture_name}.Load(int4(({coord} + {offset}), {lod}))"
            if shape_type == "Texture1D":
                return f"{texture_name}.Load(int2(({coord} + {offset}), {lod}))"
            if shape_type == "Texture1DArray":
                coord_x = self.vector_component(coord, "x")
                layer = self.vector_component(coord, "y")
                return (
                    f"{texture_name}.Load(int3(({coord_x} + {offset}), {layer}, {lod}))"
                )
            return f"{texture_name}.Load(int3(({coord} + {offset}), {lod}))"

        return None

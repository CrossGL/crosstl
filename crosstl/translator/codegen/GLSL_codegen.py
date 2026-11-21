"""CrossGL-to-GLSL code generator."""

import re
from copy import deepcopy
from types import SimpleNamespace

from ..arithmetic_conversions import (
    ArithmeticScalarKind,
    narrow_integer_shape,
    target_arithmetic_type,
)
from ..ast import (
    ArrayAccessNode,
    ArrayLiteralNode,
    ArrayNode,
    AssignmentNode,
    BinaryOpNode,
    BlockNode,
    BreakNode,
    CastNode,
    ConstructorNode,
    ContinueNode,
    CooperativeMatrixOpNode,
    CooperativeMatrixType,
    DoWhileNode,
    ExpressionStatementNode,
    ForInNode,
    ForNode,
    FunctionCallNode,
    IdentifierNode,
    IfNode,
    LiteralNode,
    LoopNode,
    MatchNode,
    MemberAccessNode,
    MeshOpNode,
    PointerAccessNode,
    PointerReinterpretNode,
    PointerType,
    PreprocessorNode,
    PrimitiveType,
    RangeNode,
    RayQueryOpNode,
    RayTracingOpNode,
    ReferenceType,
    ReturnNode,
    StructNode,
    SwitchNode,
    SwizzleNode,
    TernaryOpNode,
    UnaryOpNode,
    VariableNode,
    WaveOpNode,
    WhileNode,
)
from ..matrix_shape import (
    crossgl_linear_algebra_product_type,
    crossgl_matrix_componentwise_type,
)
from ..refusals import OpenGLRefusal
from ..source_licenses import source_license_comments
from ..structure_conversions import StructureConversionKind
from ..validation import expression_debug_name
from ..vulkan_memory_scope import declares_vulkan_memory_model
from .array_utils import (
    collect_literal_int_constants,
    collect_struct_member_types,
    evaluate_literal_int_expression,
    fold_array_type_counts,
    format_c_style_array_declaration,
    split_array_type_suffix,
)
from .assignment_values import assignment_value_expression
from .builtin_result_types import GLSL_FAMILY, component_access_type
from .call_effects import builtin_argument_write_indices
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
    infer_enum_constructor_type,
    sanitize_type_name,
)
from .generic_concrete_call_results import concrete_overload_results_differ
from .generic_emission_order import mark_generic_function_emitted
from .generic_function_utils import (
    generic_function_call_name,
    generic_function_emission_list,
    generic_function_parameters,
    numeric_trait_method_result_type,
    prepare_generic_function_specializations,
)
from .generic_global_types import (
    collect_constant_and_cbuffer_member_types,
    glsl_preliminary_global_variable_types,
)
from .generic_struct_utils import (
    collect_generic_struct_definitions,
    collect_generic_struct_specialization_member_types,
    collect_generic_struct_specializations,
    infer_struct_constructor_type,
)
from .glsl_builtin_extensions import require_builtin_extensions
from .glsl_call_helpers import emit_glsl_call_helpers
from .glsl_compute_derivative_groups import compute_derivative_group_layout
from .glsl_descriptor_sets import GLSLDescriptorSetBindingsMixin
from .glsl_entry_resource_names import glsl_entry_resource_parameter_names
from .glsl_exact_lookups import glsl_call_names_resolving_to
from .glsl_float_atomic_storage import FloatAtomicStorage
from .glsl_fragment_interlock import plan_fragment_interlock, wrap_fragment_interlock
from .glsl_guarded_offsets import call_offset_relation
from .glsl_helper_stage_guards import (
    glsl_helper_stage_macros,
    wrap_glsl_helper_stage_guard,
)
from .glsl_hlsl_cbuffer import GLSLHlslConstantBuffers
from .glsl_hlsl_matrix import GLSLEmulatedMatrices
from .glsl_hlsl_precise import GLSLHlslPreciseMixin
from .glsl_hlsl_resources import GLSLHlslResourceMixin
from .glsl_hlsl_structured import GLSLHlslStructuredBuffers
from .glsl_hlsl_values import GLSLHlslValueMixin
from .glsl_invariant_builtins import invariant_builtin_redeclarations
from .glsl_narrow_vectors import GLSLNarrowVectorMixin, narrow_scalars_promote
from .glsl_quad_lane_reads import glsl_quad_lane_read_call, quad_lane_read_needs_helper
from .glsl_ray_queries import (
    glsl_ray_desc_declaration,
    glsl_ray_query_constant_expression,
)
from .glsl_refusals import input_attachment_refusal_reason
from .glsl_shadow_lod import require_shadow_lod_extension
from .glsl_specialization_dependents import (
    constant_source_name,
    partition_constants_by_specialization_dependency,
)
from .glsl_stage_distances import distance_output_epilogue
from .glsl_stage_entries import refuse_repeated_stage_entries
from .glsl_stage_input_values import stage_input_struct_value_declarations
from .glsl_stage_output_values import (
    flattened_stage_output_assignment,
    refuse_unwritten_entry_return,
    render_flattened_stage_output_store,
    render_stage_struct_value_return,
)
from .glsl_target_builtins import OPENGL_UNAVAILABLE_BUILTINS
from .glsl_union_storage import UnionStorage
from .glsl_vulkan_resources import VulkanResourceForms
from .image_access_contracts import (
    collect_function_image_access_requirements,
    collect_function_parameter_names,
    image_access_satisfies_requirement,
    is_image_atomic_operation,
    is_projected_texture_basic_offset_operation,
    is_projected_texture_grad_offset_operation,
    is_projected_texture_lod_offset_operation,
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
    is_texture_sample_offset_operation,
    is_texture_sampling_operation,
    record_explicit_image_metadata,
    texture_sample_offset_capability_error,
)
from .index_range_contracts import IntegerRange
from .loop_guard_bounds import guarded_additive_loop_entries
from .loop_induction import (
    fixed_stride_loop_terminates,
    geometric_loop_terminates,
    loop_constant_bindings,
    metadata_bound_key,
    metadata_loop_bindings,
    modular_stride_loops,
    static_loop_trip_count,
    walk_control_scopes,
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
from .opengl_aggregate_assignments import OpenGLAggregateAssignmentMixin
from .opengl_call_arguments import OpenGLCallArgumentMixin
from .opengl_call_expressions import OpenGLCallExpressionMixin
from .opengl_compound_statements import OpenGLCompoundStatementMixin
from .opengl_cooperative_matrix import OpenGLCooperativeMatrixMixin
from .opengl_expression_forms import OpenGLExpressionFormMixin
from .opengl_fixed_array_parameters import OpenGLFixedArrayParameterMixin
from .opengl_function_intervals import OpenGLFunctionIntervalMixin
from .opengl_function_signatures import OpenGLFunctionSignatureMixin
from .opengl_global_initializers import OpenGLGlobalInitializerMixin
from .opengl_image_operations import OpenGLImageOperationMixin
from .opengl_interface_blocks import OpenGLInterfaceBlockMixin
from .opengl_local_identifiers import OpenGLLocalIdentifierMixin
from .opengl_mesh_layouts import OpenGLMeshLayoutMixin
from .opengl_metal_builtins import OpenGLMetalBuiltinMixin
from .opengl_metal_math import OpenGLMetalMathMixin
from .opengl_numeric_conversions import OpenGLNumericConversionMixin
from .opengl_numeric_helpers import OpenGLNumericHelperMixin
from .opengl_private_pointer_bindings import OpenGLPrivatePointerBindingMixin
from .opengl_private_pointer_intervals import OpenGLPrivatePointerIntervalMixin
from .opengl_private_pointer_layouts import OpenGLPrivatePointerLayoutMixin
from .opengl_program_options import OpenGLProgramOptionMixin
from .opengl_program_structure import OpenGLProgramStructureMixin
from .opengl_resource_bindings import OpenGLResourceBindingMixin
from .opengl_resource_specializations import OpenGLResourceSpecializationMixin
from .opengl_specialization_constants import OpenGLSpecializationConstantMixin
from .opengl_stage_functions import OpenGLStageFunctionMixin
from .opengl_stage_io import OpenGLStageIoMixin
from .opengl_stage_members import OpenGLStageMemberMixin
from .opengl_storage_pointer_bindings import OpenGLStoragePointerBindingMixin
from .opengl_storage_pointer_declarations import OpenGLStoragePointerDeclarationMixin
from .opengl_storage_reinterpretation import OpenGLStorageReinterpretationMixin
from .opengl_structured_buffers import OpenGLStructuredBufferMixin
from .opengl_subgroup_control_flow import OpenGLSubgroupControlFlowMixin
from .opengl_tables import OpenGLTableMixin
from .opengl_texture_calls import OpenGLTextureCallMixin
from .opengl_values import (  # noqa: F401  (also imported from this module)
    GLSL_CONVERTED_FRAGMENT_BUILTIN_TYPES,
    GLSL_FLOAT_TO_INTEGER_RULE_HELPERS,
    GLSL_SPIRV_GUARD,
    OpenGLAggregateInitializerError,
    OpenGLArithmeticConversionError,
    OpenGLAtomicFenceLoweringError,
    OpenGLBooleanCompoundAssignmentError,
    OpenGLBooleanOrderedIntrinsicError,
    OpenGLCompileTimeGlobalError,
    OpenGLComplexArithmeticError,
    OpenGLCompoundAssignmentError,
    OpenGLCooperativeMatrixError,
    OpenGLCopySignError,
    OpenGLEntryPointSelectionError,
    OpenGLFixedArrayResourceError,
    OpenGLFloatingRemainderError,
    OpenGLForInIterableError,
    OpenGLGlobalInitializerError,
    OpenGLIndexTypeError,
    OpenGLIsFiniteError,
    OpenGLMappedOverloadError,
    OpenGLMetalMathError,
    OpenGLPreciseQualifierError,
    OpenGLPrivatePointerParameterError,
    OpenGLRecursiveFunctionError,
    OpenGLReferenceParameterError,
    OpenGLResourceMemoryQualifierError,
    OpenGLScalarConversionError,
    OpenGLSignBitError,
    OpenGLSoftwareSubgroupError,
    OpenGLSpecializationConstantError,
    OpenGLStoragePointerError,
    OpenGLStructConstructionError,
    OpenGLSubgroupWidthError,
    OpenGLTemplateTypeError,
    OpenGLTrailingZeroBuiltinError,
    OpenGLWorkgroupPointerError,
    OpenGLWorkgroupSizeError,
    _opengl_cooperative_matrix_contract_type,
    _opengl_cooperative_matrix_detail_value,
    _opengl_cooperative_matrix_fragment_contract,
    _opengl_cooperative_matrix_fragment_contract_message,
)
from .opengl_wave_operations import OpenGLWaveOperationMixin
from .opengl_workgroup_pointers import OpenGLWorkgroupPointerMixin
from .pointer_reinterpret import PointerReinterpretationError, scalar_storage_layout
from .private_vector_views import lower_private_vector_views
from .reference_aliases import lower_reference_aliases
from .resource_aggregates import lower_resource_aggregates
from .resource_array_bounds import DynamicIndexSizes
from .resource_arrays import (
    collect_resource_array_size_hints,
    is_private_pointer_parameter,
)
from .sampler_pairs import COMBINED_SAMPLER_PAIRS, lower_combined_sampler_pairs
from .scalar_return_bounds import aggregate_call_intervals
from .specialized_workgroup_size import has_specialized_workgroup_size
from .stage_io_semantic_locations import SEMANTIC_LOCATIONS
from .stage_utils import (
    collect_stage_local_structs,
    collect_stage_local_variables,
    compute_local_size,
    deduplicate_named_declarations,
    function_entry_execution_config,
    function_stage_name,
    normalize_stage_name,
    order_functions_by_dependencies,
    requested_stage_name,
    should_emit_qualified_function,
    stage_matches,
)
from .subgroup_control_flow import (
    converge_subgroup_bounded_loops,
    converge_subgroup_conditional_calls,
    converge_subgroup_divergent_barriers,
    converge_subgroup_early_exits,
    converge_subgroup_guarded_returns,
)
from .subgroup_loop_carried import loop_carried_names
from .subgroup_reassignments import ReassignmentFlow
from .uniform_returns import UniformReturnAnalysis, storage_load_call
from .workgroup_size_extents import glsl_workgroup_size_declaration_layout


class GLSLCodeGen(
    OpenGLTableMixin,
    OpenGLProgramOptionMixin,
    OpenGLSubgroupControlFlowMixin,
    OpenGLProgramStructureMixin,
    OpenGLFunctionIntervalMixin,
    OpenGLResourceSpecializationMixin,
    OpenGLFixedArrayParameterMixin,
    OpenGLSpecializationConstantMixin,
    OpenGLMeshLayoutMixin,
    OpenGLStageFunctionMixin,
    OpenGLStageIoMixin,
    OpenGLStageMemberMixin,
    OpenGLLocalIdentifierMixin,
    OpenGLStoragePointerDeclarationMixin,
    OpenGLStoragePointerBindingMixin,
    OpenGLStorageReinterpretationMixin,
    OpenGLWorkgroupPointerMixin,
    OpenGLGlobalInitializerMixin,
    OpenGLCooperativeMatrixMixin,
    OpenGLNumericHelperMixin,
    OpenGLNumericConversionMixin,
    OpenGLAggregateAssignmentMixin,
    OpenGLCompoundStatementMixin,
    OpenGLMetalBuiltinMixin,
    OpenGLMetalMathMixin,
    OpenGLWaveOperationMixin,
    OpenGLFunctionSignatureMixin,
    OpenGLCallArgumentMixin,
    OpenGLTextureCallMixin,
    OpenGLImageOperationMixin,
    OpenGLPrivatePointerLayoutMixin,
    OpenGLPrivatePointerIntervalMixin,
    OpenGLPrivatePointerBindingMixin,
    OpenGLStructuredBufferMixin,
    OpenGLResourceBindingMixin,
    OpenGLInterfaceBlockMixin,
    GLSLNarrowVectorMixin,
    GLSLDescriptorSetBindingsMixin,
    LoweringRefusalMixin,
    GLSLEmulatedMatrices,
    GLSLHlslStructuredBuffers,
    GLSLHlslConstantBuffers,
    GLSLHlslPreciseMixin,
    GLSLHlslValueMixin,
    GLSLHlslResourceMixin,
    OpenGLExpressionFormMixin,
    OpenGLCallExpressionMixin,
):
    """Emit GLSL source from the shared CrossGL translator AST."""

    # Desktop GLSL converts integer literals to float and uint implicitly.
    GLSL_IMPLICIT_LITERAL_CONVERSIONS = True
    GLSL_UNAVAILABLE_BUILTINS = OPENGL_UNAVAILABLE_BUILTINS
    GLSL_WAVE_ADDITIONAL_EXTENSION_REQUIREMENTS = {
        "WaveMatch",
        "WaveMultiPrefixSum",
        "WaveMultiPrefixCountBits",
        "WaveMultiPrefixProduct",
        "WaveMultiPrefixBitAnd",
        "WaveMultiPrefixBitOr",
        "WaveMultiPrefixBitXor",
    }
    GLSL_WAVE_HELPER_VALUE_TYPES = (
        "bool",
        "bvec2",
        "bvec3",
        "bvec4",
        "int",
        "ivec2",
        "ivec3",
        "ivec4",
        "uint",
        "uvec2",
        "uvec3",
        "uvec4",
        "float",
        "vec2",
        "vec3",
        "vec4",
        "double",
        "dvec2",
        "dvec3",
        "dvec4",
    )

    def __init__(
        self,
        *,
        cooperative_matrix_software_lowering=False,
        software_subgroup_width=None,
        private_pointer_out_of_bounds_read="error",
        cooperative_matrix_mma_profile=None,
    ):
        """Initialize GLSL type maps and per-generation stage/resource state."""
        self.float_atomic_storage = FloatAtomicStorage(self)
        self.metadata_preconditions = []
        self.immutable_storage = []
        self.glsl_immutable_storage_names = frozenset()
        self.set_cooperative_matrix_software_lowering(
            cooperative_matrix_software_lowering
        )
        self.set_private_pointer_out_of_bounds_read(private_pointer_out_of_bounds_read)
        self.set_cooperative_matrix_mma_profile(cooperative_matrix_mma_profile)
        self.software_subgroup_width = None
        self.set_software_subgroup_width(software_subgroup_width)
        self.required_glsl_software_subgroup_helpers = set()
        self.glsl_software_subgroup_entry_function_id = None
        self.glsl_software_subgroup_entry_function_name = None
        self.glsl_software_subgroup_helper_function_ids = set()
        self.glsl_software_subgroup_candidate_helper_function_ids = set()
        self.glsl_software_subgroup_omitted_helper_function_ids = set()
        self.glsl_software_subgroup_uniform_for_node_ids = set()
        self.glsl_software_subgroup_zero_trip_for_node_ids = set()
        self.glsl_software_subgroup_workgroup_size = None
        self.glsl_software_subgroup_workgroup_invocation_count = None
        self.glsl_software_subgroup_count = None
        self.glsl_software_subgroup_masked_if_plans = {}
        self.glsl_software_subgroup_strided_for_plans = {}
        self.glsl_software_subgroup_id_parameter_names = set()
        self.glsl_software_subgroup_guard_names = set()
        self.glsl_software_subgroup_guard_constant_names = set()
        self.glsl_software_subgroup_result_overrides = {}
        self.glsl_cooperative_matrix_fragment_contracts = {}
        self.glsl_cooperative_matrix_fragment_contracts_by_key = {}
        self.required_glsl_cooperative_matrix_helpers = set()
        self.current_glsl_exact_subgroup_width = None
        self.sampler_variables = set()
        self.current_sampler_parameters = set()
        self.glsl_resource_binding_records = []
        self.glsl_combined_sampler_uses = []
        self.glsl_sampler_registers = {}
        self.glsl_float_remainder_types = set()
        self.glsl_precise_float_remainder_types = set()
        self.glsl_integer_dot_types = set()
        self.glsl_vulkan_resources = VulkanResourceForms()
        self.glsl_vulkan_resource_exclusions = frozenset()
        self.reset_glsl_emulated_matrices()
        self.reset_glsl_hlsl_structured_buffers()
        self.reset_glsl_hlsl_cbuffers()
        self.glsl_image_usages = {}
        self.glsl_unknown_format_images = set()
        self.glsl_formatted_image_load_names = set()
        self.glsl_write_only_unknown_format_images = set()
        self.texture_variable_types = {}
        self.resource_variable_array_sizes = {}
        self.current_texture_parameters = {}
        self.current_resource_aliases = {}
        self.current_glsl_workgroup_pointer_aliases = {}
        self.current_glsl_storage_pointer_aliases = {}
        self.glsl_global_workgroup_pointer_aliases = {}
        self.glsl_global_storage_pointer_aliases = {}
        self.glsl_storage_pointer_access_requirement_cache = {}
        self.glsl_storage_pointer_access_requirement_active = set()
        self.current_glsl_synthetic_local_names = set()
        self.current_glsl_scalar_array_view_plans = {}
        self.current_glsl_scalar_array_views = {}
        self.glsl_fixed_array_vector_pointer_view_helper_ids = set()
        self.glsl_fixed_array_vector_pointer_view_calls = {}
        self.glsl_workgroup_pointer_scope_depth = 0
        self.image_variable_formats = {}
        self.current_image_format_parameters = {}
        self.image_variable_accesses = {}
        self.current_image_access_parameters = {}
        self.structured_buffer_variable_accesses = {}
        self.current_structured_buffer_access_parameters = {}
        self.glsl_buffer_block_variable_names = set()
        self.glsl_buffer_block_variable_types = {}
        self.glsl_buffer_block_variable_accesses = {}
        self.glsl_uniform_block_variable_types = {}
        self.glsl_interface_value_struct_names = set()
        self.glsl_interface_block_names_by_qualifier = {}
        self.global_variable_types = {}
        self.glsl_global_identifier_names = set()
        self.glsl_module_identifier_names = {}
        self.glsl_module_generated_identifier_names = {}
        self.glsl_module_used_identifier_names = set()
        self.glsl_type_identifier_names = {}
        self.glsl_generic_enum_constructor_names = {}
        self.glsl_compile_time_global_node_ids = set()
        self.glsl_compile_time_bitcast_substitution_node_ids = set()
        self.glsl_compile_time_bitcast_substitutions = {}
        self.glsl_active_compile_time_bitcast_substitutions = set()
        self.glsl_runtime_global_initializer_node_ids = set()
        self.glsl_stage_runtime_global_initializers = {}
        self.global_type_aliases = {}
        self.current_type_aliases = {}
        self.function_sampler_parameter_indices = {}
        self.function_parameter_names = {}
        self.function_parameter_types = {}
        self.function_parameter_infos = {}
        self.function_return_types = {}
        self.function_image_access_requirements = {}
        self.function_structured_buffer_access_requirements = {}
        self.function_definitions = {}
        self.glsl_function_overloads_by_name = {}
        self.glsl_function_target_names = {}
        self.current_glsl_function_key = None
        self.current_glsl_source_function_id = None
        self.current_glsl_source_function_name = None
        self.glsl_function_stage_contexts = {}
        self.glsl_function_call_graph = {}
        self.glsl_function_call_locations = {}
        self.glsl_function_call_source_edges = {}
        self.glsl_metal_simd_group_placeholder_functions = set()
        self.glsl_resource_function_specializations = {}
        self.glsl_resource_function_call_names = {}
        self.glsl_resource_specialized_source_names = set()
        self.glsl_generic_helper_refusals = {}
        self.glsl_generic_helper_code = {}
        self.glsl_resource_function_call_plans = {}
        self.glsl_resource_function_semantic_call_plans = {}
        self.current_glsl_resource_function_call_semantic_locations = {}
        self.emitted_glsl_resource_specialization_names = set()
        self.unsupported_structured_buffer_array_functions = {}
        self.resource_array_size_hints = {}
        self.function_resource_array_size_hints = {}
        self.function_private_pointer_array_size_hints = {}
        self.function_private_pointer_required_spans = {}
        self.function_private_pointer_parameters = {}
        self.function_private_pointer_parameter_indices = {}
        self.function_private_pointer_element_types = {}
        self.function_private_pointer_read_only_parameters = {}
        self.function_private_pointer_backing_variants = {}
        self.function_private_pointer_local_arrays = {}
        self.function_private_pointer_view_calls = {}
        self.function_private_pointer_base_names = {}
        self.function_private_pointer_source_names = {}
        self.function_private_pointer_full_span_parameters = {}
        self.function_private_pointer_scalar_access_violations = {}
        self.glsl_private_pointer_robust_zero_read_variants = {}
        self.glsl_private_pointer_zero_index_access_ids = set()
        self.glsl_private_pointer_noop_rebinding_ids = set()
        self.glsl_private_pointer_base_rebinding_parameters = {}
        self.current_glsl_private_pointer_variant = {}
        self.current_glsl_private_pointer_base_names = {}
        self.current_glsl_private_pointer_function_name = None
        self.literal_int_constants = {}
        self.glsl_specialization_default_int_constants = {}
        self.glsl_constant_types = {}
        self.glsl_specialization_constant_names = set()
        self.glsl_global_specialization_constants = []
        self.glsl_specialization_constant_declarations = []
        self.glsl_runtime_specialization_dependent_ids = set()
        self.literal_int_vector_constants = {}
        self.literal_int_vector_array_constants = {}
        self.current_compile_time_int_constants = {}
        self.current_compile_time_int_vector_constants = {}
        self.current_compile_time_int_vector_array_constants = {}
        self.current_stage_output = None
        self.current_stage_inputs = {}
        self.current_stage_outputs = {}
        self.current_stage_output_member_map = {}
        self.current_stage_parameter_aliases = {}
        self.stage_parameter_output_aliases = {}
        self.stage_entry_functions_by_stage = {}
        self.stage_entry_resource_parameter_aliases = {}
        self.stage_entry_storage_pointer_bindings = {}
        self.stage_entry_constant_value_parameter_aliases = {}
        self.stage_entry_resource_declaration_names = set()
        self.stage_entry_constant_value_cbuffer_names = set()
        self.current_identifier_aliases = {}
        self.current_glsl_local_identifier_names = {}
        self.glsl_hoisted_shared_declarations = []
        self.glsl_hoisted_shared_aliases_by_declaration = {}
        self.glsl_hoisted_shared_declaration_ids = set()
        self.glsl_hoisted_shared_alias_names = set()
        self.glsl_hoisted_shared_bindings_by_declaration = {}
        self.glsl_shared_dimension_constants = {}
        self.current_mesh_output_parameters = {}
        self.current_mesh_output_topology = None
        self.current_glsl_extensions = set()
        self.glsl_vulkan_memory_model = False
        self.current_glsl_version_line = None
        self.current_glsl_resource_binding_layouts_supported = True
        self.glsl_emission_requirements = []
        self.glsl_raised_version_line = None
        self.glsl_required_version_line = None
        self.current_mesh_output_count_limits = {}
        self.glsl_program_has_mesh_stage = False
        self.task_payload_shared_variables = []
        self.current_target_stage = None
        self.emit_stage_guards = False
        self.stage_io_used_locations = {}
        self.stage_io_used_components = {}
        self.stage_io_declarations = {}
        self.flattened_stage_variables = set()
        self.structs_by_name = {}
        self.vertex_input_struct_names = set()
        self.vertex_output_struct_names = set()
        self.default_clip_position_member_ids = set()
        self.stage_io_reserved_value_names = set()
        self.combined_vertex_output_member_names = set()
        self.fragment_input_struct_names = set()
        self.fragment_input_member_names = set()
        self.fragment_output_struct_names = set()
        self.fragment_output_member_name_maps = {}
        self.fragment_output_member_layout_maps = {}
        self.fragment_direct_output_name_maps = {}
        self.vertex_input_member_names = set()
        self.current_function_return_type = None
        self.current_glsl_precise_return_type = None
        self.current_glsl_precise_function = False
        self.current_glsl_precise_captures = None
        self.current_glsl_ieee_operator_arithmetic = False
        self.current_glsl_exactness_test_local_ids = {}
        self.current_glsl_precise_value_writes = {}
        self.current_stage_return_type = None
        self.current_stage_entry_type = None
        self.glsl_atomic_fence_stage = None
        self.function_fragment_only_requirements = {}
        self.current_expression_expected_type = None
        self.current_discarded_expression = None
        self.current_generic_function_substitutions = {}
        self.match_temp_variable_index = 0
        self.local_variable_types = {}
        self.local_variable_source_types = {}
        self.index_range_assertions = ()
        self.workgroup_access_assertions = ()
        self.required_glsl_complex64_helpers = set()
        self.required_glsl_boolean_order_helpers = set()
        self.required_glsl_metal_math_helpers = set()
        self.glsl_exponent_helpers = {}
        self.glsl_quad_lane_read_helpers = {}
        self.glsl_half_helper_names = {}
        self.glsl_half_bitcast_helper_names = {}
        self.glsl_bfloat_helper_names = {}
        self.glsl_bfloat_integer_helper_name = None
        self.glsl_bfloat_wide_integer_helper_required = False
        self.glsl_float_selection_helper_names = {}
        self.glsl_storage_structure_copy_helper_names = {}
        self.glsl_integer_float_helper_names = {}
        self.glsl_signed_remainder_helper_names = {}
        self.glsl_floating_remainder_helper_names = {}
        self.glsl_generating_global_initializer = False
        self.required_glsl_trailing_zero_helpers = set()
        self.glsl_expected_compare_helpers = {}
        self.glsl_trailing_zero_helper_names = {}
        self.glsl_trailing_zero_reserved_names = set()
        self.current_glsl_disabled_extensions = set()
        self.current_structured_buffer_array_parameters = {}
        self.current_structured_buffer_counter_parameters = {}
        self.struct_member_types = {}
        self.union_storage = UnionStorage(self)
        self.struct_member_name_maps = {}
        # Read by type-name scans that run before the program's enums are collected.
        self.struct_payload_enums = []
        self.generic_struct_definitions = {}
        self.generic_struct_specializations = {}
        self.generic_function_definitions = {}
        self.generic_function_specializations = {}
        self.generic_function_specialized_names = {}
        self.current_generic_function_substitutions = {}
        self.enforce_concrete_glsl_types = False
        self.generic_type_parameter_names = set()
        self.structured_buffer_instance_members = {}
        self.structured_buffer_counter_members = {}
        self.structured_buffer_counter_instances = {}
        self.glsl_buffer_block_struct_names = set()
        self.glsl_buffer_block_read_validation_suppression = 0
        self.semantic_map = {
            "gl_VertexID": "gl_VertexID",
            "gl_VertexIndex": "gl_VertexID",
            "SV_VertexID": "gl_VertexID",
            "SV_VertexId": "gl_VertexID",
            "sv_vertex_id": "gl_VertexID",
            "sv_vertexid": "gl_VertexID",
            "gl_InstanceID": "gl_InstanceID",
            "SV_InstanceID": "gl_InstanceID",
            "SV_InstanceId": "gl_InstanceID",
            "sv_instance_id": "gl_InstanceID",
            "sv_instanceid": "gl_InstanceID",
            "gl_IsFrontFace": "gl_FrontFacing",
            "gl_PrimitiveID": "gl_PrimitiveID",
            "SV_CullPrimitive": "gl_CullPrimitiveEXT",
            **{
                semantic: f"layout(location = {location})"
                for semantic, location in SEMANTIC_LOCATIONS.items()
            },
            # Vertex outputs
            "gl_Position": "gl_Position",
            "gl_PointSize": "gl_PointSize",
            "gl_ClipDistance": "gl_ClipDistance",
            "gl_CullDistance": "gl_CullDistance",
            # Fragment outputs
            "gl_FragColor": "layout(location = 0)",
            "gl_FragColor0": "layout(location = 0)",
            "gl_FragColor1": "layout(location = 1)",
            "gl_FragColor2": "layout(location = 2)",
            "gl_FragColor3": "layout(location = 3)",
            "gl_FragColor4": "layout(location = 4)",
            "gl_FragColor5": "layout(location = 5)",
            "gl_FragColor6": "layout(location = 6)",
            "gl_FragColor7": "layout(location = 7)",
            "gl_FragDepth": "gl_FragDepth",
            "gl_FragStencilRefEXT": "gl_FragStencilRefARB",
            "gl_SampleMask": "gl_SampleMask",
            "SV_Coverage": "gl_SampleMask",
            "sample_mask": "gl_SampleMask",
            # Additional fragment inputs
            "gl_FragCoord": "gl_FragCoord",
            "gl_FrontFacing": "gl_FrontFacing",
            "gl_PointCoord": "gl_PointCoord",
            "gl_SampleID": "gl_SampleID",
            "SV_SampleIndex": "gl_SampleID",
            "sv_sample_index": "gl_SampleID",
            "sv_sampleindex": "gl_SampleID",
            "sample_id": "gl_SampleID",
            "sample_index": "gl_SampleID",
            "gl_SamplePosition": "gl_SamplePosition",
            "sample_position": "gl_SamplePosition",
            "gl_SampleMaskIn": "gl_SampleMaskIn",
            "sample_mask_in": "gl_SampleMaskIn",
            # GL_NV_conservative_raster_underestimation spells SPIR-V's FullyCoveredEXT this way.
            "gl_FullyCoveredEXT": "gl_FragFullyCoveredNV",
            # Compute shader specific
            "gl_GlobalInvocationID": "gl_GlobalInvocationID",
            "SV_DispatchThreadID": "gl_GlobalInvocationID",
            "SV_DispatchThreadId": "gl_GlobalInvocationID",
            "sv_dispatch_thread_id": "gl_GlobalInvocationID",
            "sv_dispatchthreadid": "gl_GlobalInvocationID",
            "gl_LocalInvocationID": "gl_LocalInvocationID",
            "SV_GroupThreadID": "gl_LocalInvocationID",
            "SV_GroupThreadId": "gl_LocalInvocationID",
            "sv_group_thread_id": "gl_LocalInvocationID",
            "sv_groupthreadid": "gl_LocalInvocationID",
            "gl_WorkGroupID": "gl_WorkGroupID",
            "SV_GroupID": "gl_WorkGroupID",
            "SV_GroupId": "gl_WorkGroupID",
            "sv_group_id": "gl_WorkGroupID",
            "sv_groupid": "gl_WorkGroupID",
            "gl_LocalInvocationIndex": "gl_LocalInvocationIndex",
            "SV_GroupIndex": "gl_LocalInvocationIndex",
            "sv_group_index": "gl_LocalInvocationIndex",
            "sv_groupindex": "gl_LocalInvocationIndex",
            "gl_QuadGroupID": "gl_QuadGroupID",
            "gl_QuadGroupInvocationID": "gl_QuadGroupInvocationID",
            "gl_WorkGroupSize": "gl_WorkGroupSize",
            "gl_NumWorkGroups": "gl_NumWorkGroups",
        }

        self.type_mapping = {
            "vec2": "vec2",
            "vec3": "vec3",
            "vec4": "vec4",
            "float2": "vec2",
            "float3": "vec3",
            "float4": "vec4",
            "packed_float2": "vec2",
            "packed_float3": "vec3",
            "packed_float4": "vec4",
            "simd_float2": "vec2",
            "simd_float3": "vec3",
            "simd_float4": "vec4",
            "half": "float",
            "bfloat": "float",
            "bfloat16": "float",
            "bfloat16_t": "float",
            "bfloat2": "vec2",
            "bfloat3": "vec3",
            "bfloat4": "vec4",
            "bfloat16vec2": "vec2",
            "bfloat16vec3": "vec3",
            "bfloat16vec4": "vec4",
            "half2": "vec2",
            "half3": "vec3",
            "half4": "vec4",
            "packed_half2": "vec2",
            "packed_half3": "vec3",
            "packed_half4": "vec4",
            "atomic<int>": "int",
            "atomic<uint>": "uint",
            "float16": "float",
            "float16_t": "float",
            "f16": "float",
            "f32": "float",
            "f64": "double",
            "f16vec2": "vec2",
            "f16vec3": "vec3",
            "f16vec4": "vec4",
            "min16float": "float",
            "min10float": "float",
            "min16float2": "vec2",
            "min16float3": "vec3",
            "min16float4": "vec4",
            "min10float2": "vec2",
            "min10float3": "vec3",
            "min10float4": "vec4",
            "double2": "dvec2",
            "double3": "dvec3",
            "double4": "dvec4",
            "str": "int",
            "ivec2": "ivec2",
            "ivec3": "ivec3",
            "ivec4": "ivec4",
            "char": "int",
            "signed char": "int",
            "i8": "int",
            "i16": "int",
            "i32": "int",
            "i64": "int64_t",
            "int8": "int",
            "int8_t": "int",
            "uchar": "uint",
            "unsigned char": "uint",
            "u8": "uint",
            "u16": "uint",
            "u32": "uint",
            "u64": "uint64_t",
            "uint8": "uint",
            "uint8_t": "uint",
            "short": "int",
            "signed short": "int",
            "ushort": "uint",
            "unsigned short": "uint",
            "int16": "int",
            "int16_t": "int",
            "int32_t": "int",
            "int64": "int64_t",
            "int64_t": "int64_t",
            "long": "int64_t",
            "signed long": "int64_t",
            "ptrdiff_t": "int64_t",
            "uint16": "uint",
            "uint16_t": "uint",
            "uint32_t": "uint",
            "uint64": "uint64_t",
            "uint64_t": "uint64_t",
            "ulong": "uint64_t",
            "unsigned long": "uint64_t",
            "size_t": "uint64_t",
            "int2": "ivec2",
            "int3": "ivec3",
            "int4": "ivec4",
            "packed_int2": "ivec2",
            "packed_int3": "ivec3",
            "packed_int4": "ivec4",
            "simd_int2": "ivec2",
            "simd_int3": "ivec3",
            "simd_int4": "ivec4",
            "char2": "ivec2",
            "char3": "ivec3",
            "char4": "ivec4",
            "i8vec2": "ivec2",
            "i8vec3": "ivec3",
            "i8vec4": "ivec4",
            "i16vec2": "ivec2",
            "i16vec3": "ivec3",
            "i16vec4": "ivec4",
            "uchar2": "uvec2",
            "uchar3": "uvec3",
            "uchar4": "uvec4",
            "u8vec2": "uvec2",
            "u8vec3": "uvec3",
            "u8vec4": "uvec4",
            "u16vec2": "uvec2",
            "u16vec3": "uvec3",
            "u16vec4": "uvec4",
            "short2": "ivec2",
            "short3": "ivec3",
            "short4": "ivec4",
            "ushort2": "uvec2",
            "ushort3": "uvec3",
            "ushort4": "uvec4",
            "min16int": "int",
            "min12int": "int",
            "min16int2": "ivec2",
            "min16int3": "ivec3",
            "min16int4": "ivec4",
            "min12int2": "ivec2",
            "min12int3": "ivec3",
            "min12int4": "ivec4",
            "uint2": "uvec2",
            "uint3": "uvec3",
            "uint4": "uvec4",
            "packed_uint2": "uvec2",
            "packed_uint3": "uvec3",
            "packed_uint4": "uvec4",
            "simd_uint2": "uvec2",
            "simd_uint3": "uvec3",
            "simd_uint4": "uvec4",
            "min16uint": "uint",
            "min16uint2": "uvec2",
            "min16uint3": "uvec3",
            "min16uint4": "uvec4",
            "bool2": "bvec2",
            "bool3": "bvec3",
            "bool4": "bvec4",
            "mat2": "mat2",
            "mat3": "mat3",
            "mat4": "mat4",
            "float2x2": "mat2",
            "float2x3": "mat3x2",
            "float2x4": "mat4x2",
            "float3x2": "mat2x3",
            "float3x3": "mat3",
            "float3x4": "mat4x3",
            "float4x2": "mat2x4",
            "float4x3": "mat3x4",
            "float4x4": "mat4",
            "simd_float2x2": "mat2",
            "simd_float2x3": "mat3x2",
            "simd_float2x4": "mat4x2",
            "simd_float3x2": "mat2x3",
            "simd_float3x3": "mat3",
            "simd_float3x4": "mat4x3",
            "simd_float4x2": "mat2x4",
            "simd_float4x3": "mat3x4",
            "simd_float4x4": "mat4",
            "half2x2": "mat2",
            "half2x3": "mat3x2",
            "half2x4": "mat4x2",
            "half3x2": "mat2x3",
            "half3x3": "mat3",
            "half3x4": "mat4x3",
            "half4x2": "mat2x4",
            "half4x3": "mat3x4",
            "half4x4": "mat4",
            "f16mat2": "mat2",
            "f16mat3": "mat3",
            "f16mat4": "mat4",
            "f16mat2x2": "mat2",
            "f16mat2x3": "mat2x3",
            "f16mat2x4": "mat2x4",
            "f16mat3x2": "mat3x2",
            "f16mat3x3": "mat3",
            "f16mat3x4": "mat3x4",
            "f16mat4x2": "mat4x2",
            "f16mat4x3": "mat4x3",
            "f16mat4x4": "mat4",
            "min16float2x2": "mat2",
            "min16float2x3": "mat3x2",
            "min16float2x4": "mat4x2",
            "min16float3x2": "mat2x3",
            "min16float3x3": "mat3",
            "min16float3x4": "mat4x3",
            "min16float4x2": "mat2x4",
            "min16float4x3": "mat3x4",
            "min16float4x4": "mat4",
            "min10float2x2": "mat2",
            "min10float2x3": "mat3x2",
            "min10float2x4": "mat4x2",
            "min10float3x2": "mat2x3",
            "min10float3x3": "mat3",
            "min10float3x4": "mat4x3",
            "min10float4x2": "mat2x4",
            "min10float4x3": "mat3x4",
            "min10float4x4": "mat4",
            "double2x2": "dmat2",
            "double2x3": "dmat3x2",
            "double2x4": "dmat4x2",
            "double3x2": "dmat2x3",
            "double3x3": "dmat3",
            "double3x4": "dmat4x3",
            "double4x2": "dmat2x4",
            "double4x3": "dmat3x4",
            "double4x4": "dmat4",
            "float": "float",
            "int": "int",
            "uint": "uint",
            "bool": "bool",
            "double": "double",
            "void": "void",
            "sampler": "sampler",
            "comparison_sampler": "sampler",
            "texture1D": "texture1D",
            "texture1DArray": "texture1DArray",
            "texture2D": "texture2D",
            "texture3D": "texture3D",
            "textureCube": "textureCube",
            "texture2DArray": "texture2DArray",
            "textureCubeArray": "textureCubeArray",
            "sampler1D": "sampler1D",
            "sampler1DArray": "sampler1DArray",
            "sampler2D": "sampler2D",
            "sampler3D": "sampler3D",
            "samplerCube": "samplerCube",
            "sampler2DArray": "sampler2DArray",
            "samplerCubeArray": "samplerCubeArray",
            "sampler2DMS": "sampler2DMS",
            "sampler2DMSArray": "sampler2DMSArray",
            "samplerExternalOES": "samplerExternalOES",
            "__samplerExternal2DY2YEXT": "__samplerExternal2DY2YEXT",
            "sampler1DShadow": "sampler1DShadow",
            "sampler1DArrayShadow": "sampler1DArrayShadow",
            "sampler2DShadow": "sampler2DShadow",
            "sampler2DArrayShadow": "sampler2DArrayShadow",
            "samplerCubeShadow": "samplerCubeShadow",
            "samplerCubeArrayShadow": "samplerCubeArrayShadow",
            "isampler1D": "isampler1D",
            "isampler1DArray": "isampler1DArray",
            "isampler2D": "isampler2D",
            "isampler2DArray": "isampler2DArray",
            "isampler3D": "isampler3D",
            "isamplerCube": "isamplerCube",
            "isamplerCubeArray": "isamplerCubeArray",
            "isampler2DRect": "isampler2DRect",
            "isamplerBuffer": "isamplerBuffer",
            "isampler2DMS": "isampler2DMS",
            "isampler2DMSArray": "isampler2DMSArray",
            "usampler1D": "usampler1D",
            "usampler1DArray": "usampler1DArray",
            "usampler2D": "usampler2D",
            "usampler2DArray": "usampler2DArray",
            "usampler3D": "usampler3D",
            "usamplerCube": "usamplerCube",
            "usamplerCubeArray": "usamplerCubeArray",
            "usampler2DRect": "usampler2DRect",
            "usamplerBuffer": "usamplerBuffer",
            "usampler2DMS": "usampler2DMS",
            "usampler2DMSArray": "usampler2DMSArray",
            "iimage1D": "iimage1D",
            "iimage1DArray": "iimage1DArray",
            "iimage2D": "iimage2D",
            "iimage3D": "iimage3D",
            "iimageCube": "iimageCube",
            "iimageCubeArray": "iimageCubeArray",
            "iimage2DArray": "iimage2DArray",
            "iimage2DMS": "iimage2DMS",
            "iimage2DMSArray": "iimage2DMSArray",
            "uimage1D": "uimage1D",
            "uimage1DArray": "uimage1DArray",
            "uimage2D": "uimage2D",
            "uimage3D": "uimage3D",
            "uimageCube": "uimageCube",
            "uimageCubeArray": "uimageCubeArray",
            "uimage2DArray": "uimage2DArray",
            "uimage2DMS": "uimage2DMS",
            "uimage2DMSArray": "uimage2DMSArray",
            "samplerBuffer": "samplerBuffer",
            "imageBuffer": "imageBuffer",
            "iimageBuffer": "iimageBuffer",
            "uimageBuffer": "uimageBuffer",
            "image1D": "image1D",
            "image1DArray": "image1DArray",
            "image2D": "image2D",
            "image3D": "image3D",
            "imageCube": "imageCube",
            "imageCubeArray": "imageCubeArray",
            "image2DArray": "image2DArray",
            "image2DMS": "image2DMS",
            "image2DMSArray": "image2DMSArray",
            "accelerationStructureEXT": "accelerationStructureEXT",
        }

        self.function_map = {
            # HLSL's non-uniform resource index is GLSL's GL_EXT_nonuniform_qualifier.
            "NonUniformResourceIndex": "nonuniformEXT",
            "atan2": "atan",
            "lerp": "mix",
            "frac": "fract",
            "saturate": "clamp",
            "tex2D": "texture",
            "tex2Dproj": "textureProj",
            "tex2Dlod": "textureLod",
            "tex2Dbias": "texture",
            "tex2Dgrad": "textureGrad",
            "tex2Doffset": "textureOffset",
            "texCUBE": "texture",
            "texCUBElod": "textureLod",
            "texCUBEbias": "texture",
            "texCUBEgrad": "textureGrad",
            "textureOffset": "textureOffset",
            "textureProj": "textureProj",
            "textureGatherOffset": "textureGatherOffset",
            "textureGatherOffsets": "textureGatherOffsets",
            "textureQueryLevels": "textureQueryLevels",
            "textureQueryLod": "textureQueryLod",
            "texelFetch": "texelFetch",
            "interpolate_at_centroid": "interpolateAtCentroid",
            "interpolate_at_sample": "interpolateAtSample",
            "interpolate_at_offset": "interpolateAtOffset",
            "interpolateAtCentroid": "interpolateAtCentroid",
            "interpolateAtSample": "interpolateAtSample",
            "interpolateAtOffset": "interpolateAtOffset",
            "TraceRay": "traceRayEXT",
            "ReportHit": "reportIntersectionEXT",
            "ReportIntersection": "reportIntersectionEXT",
            "CallShader": "executeCallableEXT",
            "ExecuteCallable": "executeCallableEXT",
            "imageAtomicAdd": "imageAtomicAdd",
            "imageAtomicMin": "imageAtomicMin",
            "imageAtomicMax": "imageAtomicMax",
            "imageAtomicAnd": "imageAtomicAnd",
            "imageAtomicOr": "imageAtomicOr",
            "imageAtomicXor": "imageAtomicXor",
            "imageAtomicExchange": "imageAtomicExchange",
            "imageAtomicCompSwap": "imageAtomicCompSwap",
            "atomicCounterIncrement": "atomicCounterIncrement",
            "atomicCounterDecrement": "atomicCounterDecrement",
            "atomicCounter": "atomicCounter",
            "atomicCounterAdd": "atomicCounterAdd",
            "ddx": "dFdx",
            "ddx_fine": "dFdxFine",
            "ddx_coarse": "dFdxCoarse",
            "ddy": "dFdy",
            "ddy_fine": "dFdyFine",
            "ddy_coarse": "dFdyCoarse",
            "fwidth_fine": "fwidthFine",
            "fwidth_coarse": "fwidthCoarse",
            "rsqrt": "inversesqrt",
            "printf": "debugPrintfEXT",
            "inverseSqrt": "inversesqrt",
            "metal::rint": "roundEven",
            "metal_u3a_u3arint": "roundEven",
            "rint": "roundEven",
            "sincos": "sin_cos",  # Custom function needed
            "log2": "log2",
            "exp2": "exp2",
            "pow": "pow",
            "sqrt": "sqrt",
            "abs": "abs",
            "sign": "sign",
            "floor": "floor",
            "ceil": "ceil",
            "round": "round",
            "trunc": "trunc",
            "min": "min",
            "max": "max",
            "clamp": "clamp",
            "step": "step",
            "smoothstep": "smoothstep",
            "length": "length",
            "distance": "distance",
            "dot": "dot",
            "cross": "cross",
            "normalize": "normalize",
            "reflect": "reflect",
            "refract": "refract",
            "all": "all",
            "any": "any",
            "sin": "sin",
            "cos": "cos",
            "tan": "tan",
            "asin": "asin",
            "acos": "acos",
            "atan": "atan",
            "sinh": "sinh",
            "cosh": "cosh",
            "tanh": "tanh",
            "isfinite": "isfinite",
            "metal::isfinite": "isfinite",
            "metal_u3a_u3aisfinite": "isfinite",
        }

    def glsl_software_subgroup_operation_records(
        self,
        root,
        *,
        discover_zero_trip=False,
    ):
        records = []
        for node in self.glsl_software_subgroup_reachable_nodes(
            root,
            discover_zero_trip=discover_zero_trip,
        ):
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
                name = self.function_call_name(node)
                if (
                    name in self.GLSL_SOFTWARE_SUBGROUP_BARRIERS
                    and name not in self.function_return_types
                ):
                    records.append((name, node))
                    continue
                operation = self.glsl_wave_operation_name(name)
            if operation in self.GLSL_WAVE_INTRINSIC_ARITIES:
                records.append((operation, node))
        return records

    def glsl_software_subgroup_uses_system_inputs(self, root):
        """Whether ``root`` reads a subgroup lane, ID, size or count input."""
        inputs = self.SOFTWARE_SUBGROUP_SYSTEM_INPUTS
        return any(
            self.semantic_from_node(node) in inputs
            or (isinstance(node, IdentifierNode) and node.name in inputs)
            for node in self.walk_ast(root)
        )

    def glsl_software_subgroup_uniform_top_level_call_ids(self, body):
        call_ids = set()
        for statement in self.glsl_software_subgroup_statement_list(body):
            call = self.glsl_software_subgroup_top_level_call(statement)
            if call is not None:
                call_ids.add(id(call))
            if isinstance(statement, BlockNode):
                call_ids.update(
                    self.glsl_software_subgroup_uniform_top_level_call_ids(statement)
                )
            if (
                isinstance(statement, ForNode)
                and id(statement) in self.glsl_software_subgroup_uniform_for_node_ids
            ):
                call_ids.update(
                    self.glsl_software_subgroup_uniform_top_level_call_ids(
                        getattr(statement, "body", None)
                    )
                )
            if (
                isinstance(statement, IfNode)
                and id(statement) in self.glsl_software_subgroup_uniform_if_node_ids
            ):
                fixed = getattr(self, "glsl_software_subgroup_constant_guards", {}).get(
                    id(statement)
                )
                for branch in (
                    [statement.if_body if fixed else statement.else_body]
                    if fixed is not None
                    else [
                        statement.if_body,
                        *(getattr(statement, "else_if_bodies", []) or []),
                        getattr(statement, "else_body", None),
                    ]
                ):
                    call_ids.update(
                        self.glsl_software_subgroup_uniform_top_level_call_ids(branch)
                    )
        return call_ids

    def validate_glsl_software_subgroup_helpers(
        self,
        ast,
        entry_function,
        helper_records,
        workgroup_size,
        target_stage=None,
    ):
        self.glsl_software_subgroup_helper_function_names = set()
        self.glsl_software_subgroup_helper_function_ids = set()
        self.glsl_software_subgroup_omitted_helper_function_ids = set()
        self.glsl_software_subgroup_loop_constants = {}
        self.glsl_software_subgroup_loop_metadata = {}
        self.glsl_software_subgroup_function_metadata = {}
        self.glsl_software_subgroup_modular_stride_loops = set()

        functions = self.glsl_software_subgroup_functions(ast)
        for function in functions:
            self.glsl_software_subgroup_modular_stride_loops.update(
                modular_stride_loops(function, self.type_name_string, self.walk_ast)
            )
        functions_by_id = {id(function): function for function in functions}
        functions_by_name = {}
        for function in functions:
            name = getattr(function, "name", None)
            if name:
                functions_by_name.setdefault(name, []).append(function)

        direct_records = {}
        for operation, node, function in helper_records:
            name = getattr(function, "name", None)
            if not name:
                raise self.glsl_software_subgroup_error(
                    "OpenGL software subgroup helpers require a stable source "
                    "function identity",
                    workgroup_size=workgroup_size,
                    operation=operation,
                    reason="helper-identity-invalid",
                    source_location=getattr(node, "source_location", None),
                )
            direct_records.setdefault(id(function), []).append((operation, node))

        def source_overload_count(name, arity=None):
            return len(
                {
                    self.glsl_source_function_signature(function)
                    for function in functions_by_name.get(name, ())
                    if arity is None or len(function.parameters or []) == arity
                }
            )

        def resolve_call(function, call):
            name = self.function_call_name(call)
            if name not in functions_by_name:
                return None
            arguments = list(
                getattr(call, "arguments", getattr(call, "args", [])) or []
            )
            resolved = self.glsl_resolve_function_overload_in_scope(
                function,
                name,
                arguments,
                call_node=call,
            )
            if resolved is not None:
                exactly_resolved_calls.add(id(call))
                return resolved
            candidates = [
                function
                for function in functions_by_name[name]
                if getattr(function, "body", None) is not None
            ]
            signatures = {
                self.glsl_source_function_signature(function) for function in candidates
            }
            if len(signatures) == 1 and candidates:
                return candidates[0]
            return None

        exactly_resolved_calls = set()
        edges = {id(function): [] for function in functions}
        unresolved_edges = {id(function): [] for function in functions}
        for function in functions:
            body = getattr(function, "body", None)
            for node in self.glsl_software_subgroup_reachable_nodes(body):
                if not isinstance(node, FunctionCallNode):
                    continue
                try:
                    target = resolve_call(function, node)
                except OpenGLMappedOverloadError as error:
                    unresolved_edges[id(function)].append((node, error))
                    continue
                if target is not None and id(target) in functions_by_id:
                    edges[id(function)].append((node, id(target)))

        reaches_operation = set(direct_records)
        changed = True
        while changed:
            changed = False
            for function_id, call_edges in edges.items():
                if function_id in reaches_operation:
                    continue
                if any(
                    target_id in reaches_operation for _call, target_id in call_edges
                ):
                    reaches_operation.add(function_id)
                    changed = True

        def first_operation(function_id):
            pending = [function_id]
            visited = set()
            while pending:
                current_id = pending.pop(0)
                if current_id in visited:
                    continue
                visited.add(current_id)
                records = direct_records.get(current_id)
                if records:
                    return records[0]
                pending.extend(
                    target_id
                    for _call, target_id in edges.get(current_id, ())
                    if target_id in reaches_operation
                )
            operation, node, _function = helper_records[0]
            return operation, node

        def ambiguous_edge_error(call, error):
            name = self.function_call_name(call)
            candidate_ids = {
                id(function) for function in functions_by_name.get(name, ())
            }
            relevant_ids = candidate_ids & reaches_operation
            if not relevant_ids:
                return None
            operation, operation_node = first_operation(next(iter(relevant_ids)))
            return self.glsl_software_subgroup_error(
                "OpenGL software subgroup helper call cannot resolve one exact "
                f"source overload for '{name}'",
                workgroup_size=workgroup_size,
                operation=operation,
                reason="helper-identity-ambiguous",
                source_location=getattr(call, "source_location", None)
                or getattr(operation_node, "source_location", None)
                or getattr(error, "source_location", None),
            )

        self.glsl_software_subgroup_candidate_helper_function_ids = (
            reaches_operation - {id(entry_function)}
        )
        self.glsl_software_subgroup_resolved_call_targets = {
            id(call): target_id
            for call_edges in edges.values()
            for call, target_id in call_edges
        }
        calls = [
            node
            for function in functions
            for node in self.walk_ast(getattr(function, "body", None))
            if isinstance(node, FunctionCallNode)
        ]
        storage_loads = set()
        for call in calls:
            name = self.function_call_name(call)
            if name in functions_by_name or not isinstance(
                call.function, (str, IdentifierNode)
            ):
                continue
            if (
                self.glsl_software_subgroup_exit_constructor(call)
                or (name in {"min", "max"} and len(call.arguments) == 2)
                or (name == "clamp" and len(call.arguments) == 3)
            ):
                self.glsl_software_subgroup_uniform_call_ids.add(id(call))
            elif storage_load_call(name, call.arguments):
                storage_loads.add(id(call))
        analysis = UniformReturnAnalysis(
            {
                call_id: functions_by_id[target_id]
                for call_id, target_id in (
                    self.glsl_software_subgroup_resolved_call_targets.items()
                )
            },
            self.glsl_software_subgroup_uniform_call_ids,
            self.structs_by_name,
            storage_loads,
        )
        self.glsl_software_subgroup_uniform_returns = analysis
        constant_source_types = {
            **self.glsl_constant_types,
            **{
                node.name: node.var_type
                for node in ast.global_variables
                if isinstance(node, VariableNode)
                and set(node.qualifiers) & {"const", "constant"}
                and "volatile" not in node.qualifiers
            },
        }
        for function in functions:
            lexical = self.glsl_scalar_array_view_lexical_bindings(function)
            declared_names = {
                binding["name"] for binding in lexical["declarations"].values()
            }
            constant_types = {
                name: type_
                for name, type_ in constant_source_types.items()
                if name not in declared_names
            }
            for binding in lexical["declarations"].values():
                declaration = binding["node"]
                if not isinstance(getattr(declaration, "var_type", None), PointerType):
                    continue
                roots = analysis.pointer_argument_write_roots(
                    declaration.initial_value,
                    {**constant_types, **binding.get("initializer_source_types", {})},
                )
                if roots is not None:
                    self.glsl_software_subgroup_call_mutations[id(declaration)] = roots
            for call in self.walk_ast(getattr(function, "body", None)):
                if not isinstance(call, FunctionCallNode):
                    continue
                name = self.function_call_name(call)
                target = functions_by_id.get(
                    self.glsl_software_subgroup_resolved_call_targets.get(id(call))
                )
                arguments = list(call.arguments or [])
                builtin_writes = (
                    builtin_argument_write_indices(name, len(arguments))
                    if name not in functions_by_name
                    and isinstance(call.function, (str, IdentifierNode))
                    else None
                )
                pure = id(call) in self.glsl_software_subgroup_uniform_call_ids
                written_roots = set()
                if analysis.dependencies(call) is not None:
                    mutations = []
                elif target is not None:
                    parameters = target.parameters or []
                    mutations = []
                    for index, argument in enumerate(arguments):
                        if analysis.read_only_resource_parameter(target, index):
                            continue
                        if index >= len(parameters):
                            mutations.append(argument)
                        elif isinstance(parameters[index].param_type, PointerType):
                            roots = analysis.pointer_argument_write_roots(
                                argument, lexical["call_source_types"].get(id(call), {})
                            )
                            if roots is None:
                                mutations.append(argument)
                            else:
                                written_roots.update(roots)
                        elif isinstance(
                            parameters[index].param_type, ReferenceType
                        ) or set(self.glsl_parameter_qualifiers(parameters[index])) & {
                            "out",
                            "inout",
                        }:
                            # Retain destination writes, not value-only selectors.
                            # Nested argument effects remain separate call records.
                            roots = self.glsl_mutation_target_names(argument)
                            if roots:
                                written_roots.update(roots)
                            else:
                                mutations.append(argument)
                elif builtin_writes is not None:
                    mutations = [arguments[index] for index in builtin_writes]
                elif pure or (
                    name not in functions_by_name
                    and self.glsl_wave_operation_name(name)
                    in self.GLSL_WAVE_INTRINSIC_ARITIES
                ):
                    mutations = []
                else:
                    mutations = arguments
                self.glsl_software_subgroup_call_mutations[id(call)] = (
                    written_roots
                    | self.glsl_software_subgroup_expression_identifier_names(mutations)
                )

        entry_seeds = self.glsl_software_subgroup_uniform_seed_names(
            ast, entry_function, target_stage
        )
        self.glsl_software_subgroup_immutable_uniform_names = (
            self.glsl_software_subgroup_immutable_entry_names(
                entry_function, entry_seeds
            )
        )
        entry_arguments = self.glsl_software_subgroup_uniform_arguments(
            entry_function, entry_seeds, entry=True
        )
        self.glsl_software_subgroup_counter_loops = (
            self.glsl_software_subgroup_entry_counter_loops(entry_function)
        )
        # Uniform loop proofs must see transitive collective calls as well as leaves.
        self.glsl_software_subgroup_analyze_uniform_statements(
            getattr(entry_function, "body", None),
            self.glsl_software_subgroup_uniform_seed_names(ast, entry_function),
        )
        uniform_entry_call_ids = self.glsl_software_subgroup_uniform_top_level_call_ids(
            getattr(entry_function, "body", None)
        )
        entry_id = id(entry_function)

        for call, error in unresolved_edges.get(entry_id, ()):
            relevant = ambiguous_edge_error(call, error)
            if relevant is not None:
                raise relevant

        # Calls in branches a constant guard rules out never run.
        dead_ids = getattr(self, "glsl_software_subgroup_dead_node_ids", set())
        roots = set()
        for call, target_id in edges.get(entry_id, ()):
            if target_id not in reaches_operation or id(call) in dead_ids:
                continue
            operation, operation_node = first_operation(target_id)
            target = functions_by_id[target_id]
            name = getattr(target, "name", None)
            # Typed overload resolution selects one root by identity; only the
            # name-based fallback cannot distinguish same-arity overloads.
            if (
                source_overload_count(name, len(target.parameters or [])) != 1
                and id(call) not in exactly_resolved_calls
            ):
                raise self.glsl_software_subgroup_error(
                    "OpenGL software subgroup root helpers cannot use same-arity overloaded "
                    f"source function name '{name}'",
                    workgroup_size=workgroup_size,
                    operation=operation,
                    reason="helper-identity-ambiguous",
                    source_location=getattr(call, "source_location", None)
                    or getattr(operation_node, "source_location", None),
                )
            if (
                id(call) not in uniform_entry_call_ids
                or id(call) not in entry_arguments
            ):
                raise self.glsl_software_subgroup_error(
                    "OpenGL software subgroup helper "
                    f"'{name}' must be called directly from the compute entry "
                    "point as a statically uniform top-level statement",
                    workgroup_size=workgroup_size,
                    operation=operation,
                    reason="helper-call-not-uniform",
                    source_location=getattr(call, "source_location", None)
                    or getattr(operation_node, "source_location", None),
                )
            roots.add(target_id)

        if (
            helper_records
            and not roots
            and not self.glsl_software_subgroup_operation_records(
                getattr(entry_function, "body", None)
            )
        ):
            operation, node, function = helper_records[0]
            raise self.glsl_software_subgroup_error(
                "OpenGL software subgroup helper "
                f"'{getattr(function, 'name', '<unknown>')}' is not reached by "
                "a statically uniform top-level entry-point call",
                workgroup_size=workgroup_size,
                operation=operation,
                reason="helper-call-unproven",
                source_location=getattr(node, "source_location", None),
            )

        approved = set(roots)
        pending = list(roots)
        while pending:
            function_id = pending.pop(0)
            for call, error in unresolved_edges.get(function_id, ()):
                relevant = ambiguous_edge_error(call, error)
                if relevant is not None:
                    raise relevant
            for call, target_id in edges.get(function_id, ()):
                if target_id not in reaches_operation or id(call) in dead_ids:
                    continue
                if target_id not in approved:
                    approved.add(target_id)
                    pending.append(target_id)

        incoming = {function_id: 0 for function_id in approved}
        for function_id in approved:
            for _call, target_id in edges.get(function_id, ()):
                if target_id in incoming:
                    incoming[target_id] += 1
        pending = [function_id for function_id, count in incoming.items() if count == 0]
        visited = set()
        ordered = []
        while pending:
            function_id = pending.pop()
            visited.add(function_id)
            ordered.append(function_id)
            for _call, target_id in edges.get(function_id, ()):
                if target_id not in incoming:
                    continue
                incoming[target_id] -= 1
                if incoming[target_id] == 0:
                    pending.append(target_id)
        if visited != approved:
            operation, node = first_operation(next(iter(approved - visited)))
            raise self.glsl_software_subgroup_error(
                "OpenGL software subgroup helpers cannot form a recursive call chain",
                workgroup_size=workgroup_size,
                operation=operation,
                reason="helper-call-recursive",
                source_location=getattr(node, "source_location", None),
            )

        # Intersect caller facts before inspecting a callee. A parameter is not
        # uniform merely because one invocation supplies a constant argument.
        uniform_parameters = {}
        # Checked metadata bounds every call passes to a helper parameter.
        parameter_bounds = {}

        def propagate(function_id, arguments):
            bounds = self.glsl_software_subgroup_function_metadata.get(function_id)
            for call, target_id in edges.get(function_id, ()):
                if target_id not in approved or id(call) in dead_ids:
                    continue
                parameter_bounds[target_id] = join_parameter_bounds(
                    parameter_bounds.get(target_id),
                    call_parameter_bounds(
                        getattr(functions_by_id[target_id], "parameters", ()) or (),
                        getattr(call, "arguments", ()) or (),
                        bounds,
                        self.type_name_string,
                    ),
                )
                if id(call) not in arguments:
                    operation, operation_node = first_operation(target_id)
                    raise self.glsl_software_subgroup_error(
                        "OpenGL software subgroup helpers require a statically "
                        "uniform call",
                        workgroup_size=workgroup_size,
                        operation=operation,
                        reason="helper-call-not-uniform",
                        source_location=getattr(call, "source_location", None)
                        or getattr(operation_node, "source_location", None),
                    )
                facts = arguments[id(call)]
                if target_id not in uniform_parameters:
                    uniform_parameters[target_id] = set(facts)
                else:
                    uniform_parameters[target_id].intersection_update(facts)

        propagate(entry_id, entry_arguments)
        for function_id in ordered:
            function = functions_by_id[function_id]
            seeds = self.glsl_software_subgroup_uniform_seed_names(ast, None)
            seeds.difference_update(parameter.name for parameter in function.parameters)
            seeds.update(
                parameter.name
                for index, parameter in enumerate(function.parameters)
                if index in uniform_parameters.get(function_id, set())
                and not isinstance(parameter.param_type, ReferenceType)
                and not set(self.glsl_parameter_qualifiers(parameter))
                & {"out", "inout"}
                and (
                    analysis.value_type(parameter.param_type)
                    or analysis.resource_type(parameter.param_type, parameter)
                )
            )
            self.glsl_software_subgroup_function_uniform_seeds[function_id] = seeds
            arguments = self.glsl_software_subgroup_uniform_arguments(
                function,
                seeds,
                entry=False,
                inherited_bounds=parameter_bounds.get(function_id),
            )
            propagate(function_id, arguments)

        self.glsl_software_subgroup_helper_function_ids = approved
        self.glsl_software_subgroup_helper_function_names = {
            getattr(functions_by_id[function_id], "name", None)
            for function_id in approved
            if source_overload_count(
                getattr(functions_by_id[function_id], "name", None)
            )
            == 1
        }
        self.glsl_software_subgroup_helper_function_names.discard(None)
        self.glsl_software_subgroup_omitted_helper_function_ids = (
            reaches_operation - approved - {entry_id}
        )

    def validate_glsl_software_subgroup_contract(self, ast, target_stage=None):
        self.required_glsl_software_subgroup_helpers = set()
        self.glsl_software_subgroup_loop_constants = {}
        self.glsl_software_subgroup_loop_metadata = {}
        self.glsl_software_subgroup_function_metadata = {}
        self.glsl_software_subgroup_modular_stride_loops = set()
        self.glsl_software_subgroup_call_mutations = {}
        self.glsl_software_subgroup_uniform_call_ids = set()
        self.glsl_software_subgroup_uniform_returns = UniformReturnAnalysis({}, set())
        self.glsl_software_subgroup_resource_names = set()
        self.glsl_software_subgroup_function_uniform_seeds = {}
        self.glsl_software_subgroup_resolved_call_targets = {}
        self.glsl_software_subgroup_uniform_if_node_ids = set()
        self.glsl_software_subgroup_immutable_uniform_names = set()
        self.glsl_software_subgroup_counter_loops = {}
        self.glsl_software_subgroup_entry_function_id = None
        self.glsl_software_subgroup_entry_function_name = None
        self.glsl_software_subgroup_helper_function_names = set()
        self.glsl_software_subgroup_helper_function_ids = set()
        self.glsl_software_subgroup_candidate_helper_function_ids = set()
        self.glsl_software_subgroup_omitted_helper_function_ids = set()
        self.glsl_software_subgroup_uniform_for_node_ids = set()
        self.glsl_software_subgroup_zero_trip_for_node_ids = set()
        self.glsl_software_subgroup_workgroup_size = None
        self.glsl_software_subgroup_workgroup_invocation_count = None
        self.glsl_software_subgroup_count = None
        self.glsl_software_subgroup_masked_if_plans = {}
        self.glsl_software_subgroup_strided_for_plans = {}
        self.glsl_software_subgroup_id_parameter_names = set()
        self.glsl_software_subgroup_guard_names = set()
        self.glsl_software_subgroup_guard_constant_names = set()
        self.glsl_software_subgroup_result_overrides = {}
        self.glsl_software_subgroup_constant_guards = {}
        self.glsl_software_subgroup_dead_node_ids = set()
        if self.software_subgroup_width is None:
            return
        for function in self.glsl_software_subgroup_functions(ast):
            self.glsl_software_subgroup_modular_stride_loops.update(
                modular_stride_loops(function, self.type_name_string, self.walk_ast)
            )
        self.prepare_glsl_software_subgroup_constant_guards(ast)

        entries = self.glsl_software_subgroup_stage_entries(ast, target_stage)
        if len(entries) != 1 or entries[0][0] != "compute":
            raise self.glsl_software_subgroup_error(
                "OpenGL software subgroup lowering requires exactly one compute "
                "entry point and no other emitted stages",
                reason="entry-point-contract-invalid",
            )
        _stage_name, entry_function, execution_config = entries[0]
        self.glsl_software_subgroup_entry_function_id = id(entry_function)
        self.glsl_software_subgroup_entry_function_name = getattr(
            entry_function, "name", None
        )
        self.glsl_software_subgroup_id_parameter_names = {
            parameter.name
            for parameter in getattr(entry_function, "parameters", []) or []
            if getattr(parameter, "name", None)
            and self.map_semantic(self.semantic_from_node(parameter)) == "gl_SubgroupID"
        }
        self.glsl_software_subgroup_guard_names = (
            self.glsl_software_subgroup_entry_unwritten_names(
                entry_function, self.glsl_software_subgroup_id_parameter_names
            )
        )
        self.glsl_software_subgroup_guard_constant_names = (
            self.glsl_software_subgroup_entry_unwritten_names(
                entry_function, self.initial_literal_int_constants(entry_function)
            )
        )

        if has_specialized_workgroup_size(execution_config):
            raise self.glsl_software_subgroup_error(
                "OpenGL software subgroup lowering requires a workgroup size fixed "
                "in the shader, not one the pipeline specializes",
                reason="workgroup-size-specialized",
                source_location=getattr(entry_function, "source_location", None),
            )
        raw_workgroup_size = tuple(compute_local_size(execution_config))
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
        valid_workgroup_layout = (
            len(concrete_workgroup_size) == 3
            and all(value > 0 for value in concrete_workgroup_size)
            and invocation_count <= 1024
        )
        if not valid_workgroup_layout:
            raise self.glsl_software_subgroup_error(
                "OpenGL software subgroup lowering requires concrete positive "
                "local dimensions and at most 1024 invocations",
                workgroup_size=raw_workgroup_size,
                reason="workgroup-size-mismatch",
                source_location=getattr(entry_function, "source_location", None),
            )
        self.glsl_software_subgroup_workgroup_size = concrete_workgroup_size
        self.glsl_software_subgroup_workgroup_invocation_count = invocation_count
        self.glsl_software_subgroup_count = (
            invocation_count + self.software_subgroup_width - 1
        ) // self.software_subgroup_width

        # Refuses a WaveSize other than the software subgroup width.
        self.glsl_exact_subgroup_width(entry_function, "compute")

        all_records = self.glsl_software_subgroup_operation_records(
            ast,
            discover_zero_trip=True,
        )
        lane_local_matrix_nodes = self.glsl_software_subgroup_lane_local_matrix_nodes(
            ast
        )
        if lane_local_matrix_nodes and invocation_count % self.software_subgroup_width:
            raise self.glsl_software_subgroup_error(
                "OpenGL matrix fills and transfers require complete logical subgroups",
                workgroup_size=concrete_workgroup_size,
                operation="CooperativeMatrixTransfer",
                reason="incomplete-matrix-subgroup",
                source_location=getattr(
                    lane_local_matrix_nodes[0], "source_location", None
                ),
            )
        if not all_records and (
            lane_local_matrix_nodes
            or self.glsl_software_subgroup_uses_system_inputs(ast)
        ):
            # Subgroup system inputs, matrix fills and matrix transfers
            # without collective operations need only the logical split of
            # gl_LocalInvocationIndex into subgroup and lane IDs, which is
            # exact under any control flow.
            return
        if not all_records:
            raise self.glsl_software_subgroup_error(
                "OpenGL software subgroup lowering was requested but no subgroup "
                "operation is present",
                workgroup_size=concrete_workgroup_size,
                reason="operation-set-empty",
                source_location=getattr(entry_function, "source_location", None),
            )
        for operation, node in all_records:
            if (
                operation == "CooperativeMatrixMultiplyAccumulate"
                and invocation_count % self.software_subgroup_width
            ):
                raise self.glsl_software_subgroup_error(
                    "OpenGL matrix products require complete logical subgroups",
                    workgroup_size=concrete_workgroup_size,
                    operation=operation,
                    reason="incomplete-matrix-subgroup",
                    source_location=getattr(node, "source_location", None),
                )
            if operation not in self.GLSL_SOFTWARE_SUBGROUP_OPERATIONS:
                raise self.glsl_software_subgroup_error(
                    f"OpenGL software subgroup lowering does not support '{operation}'",
                    workgroup_size=concrete_workgroup_size,
                    operation=operation,
                    reason="operation-unsupported",
                    source_location=getattr(node, "source_location", None),
                )

        self.prepare_glsl_software_subgroup_masked_if_plans(entry_function)
        entry_records = self.glsl_software_subgroup_operation_records(
            getattr(entry_function, "body", None)
        )
        entry_node_ids = {id(node) for _operation, node in entry_records}
        helper_records = []
        for function in self.glsl_software_subgroup_functions(ast):
            if id(function) == id(entry_function):
                continue
            for operation, node in self.glsl_software_subgroup_operation_records(
                getattr(function, "body", None)
            ):
                if id(node) not in entry_node_ids:
                    helper_records.append((operation, node, function))
        self.glsl_software_subgroup_candidate_helper_function_ids = {
            id(function) for _operation, _node, function in helper_records
        }

        # Exempt only actual calls (and their callee identifiers), not raw
        # hardware names used as values. Declared source functions keep ownership.
        execution_barrier_call_nodes = set()
        for node in self.walk_ast(ast):
            if (
                isinstance(node, FunctionCallNode)
                and self.function_call_name(node)
                in self.GLSL_SOFTWARE_SUBGROUP_BARRIERS
            ):
                execution_barrier_call_nodes.update((id(node), id(node.function)))
        for node in self.walk_ast(ast):
            name = None
            if isinstance(node, IdentifierNode):
                name = node.name
            elif isinstance(node, FunctionCallNode):
                name = self.function_call_name(node)
            semantic = self.semantic_from_node(node)
            mapped_semantic = self.map_semantic(semantic) if semantic else None
            raw_hardware_name = (
                id(node) not in execution_barrier_call_nodes
                and isinstance(name, str)
                and (
                    name == "gl_NumSubgroups"
                    or name.startswith("gl_Subgroup")
                    or name.startswith("subgroup")
                )
            )
            unsupported_mapped_semantic = (
                mapped_semantic in self.GLSL_SUBGROUP_BASIC_BUILTINS
                and mapped_semantic not in self.GLSL_SOFTWARE_SUBGROUP_BUILTIN_ALIASES
            )
            if raw_hardware_name or unsupported_mapped_semantic:
                raise self.glsl_software_subgroup_error(
                    "OpenGL software subgroup lowering cannot preserve raw "
                    "hardware subgroup builtins",
                    workgroup_size=concrete_workgroup_size,
                    reason="subgroup-builtin-unsupported",
                    source_location=getattr(node, "source_location", None),
                )

        uniform_names = self.glsl_software_subgroup_uniform_seed_names(
            ast,
            entry_function,
            target_stage,
        )
        self.glsl_software_subgroup_immutable_uniform_names = (
            self.glsl_software_subgroup_immutable_entry_names(
                entry_function, uniform_names
            )
        )
        self.prepare_glsl_software_subgroup_strided_for_plans(
            entry_function,
            uniform_names,
        )
        self.glsl_software_subgroup_counter_loops = (
            self.glsl_software_subgroup_entry_counter_loops(entry_function)
        )
        self.glsl_software_subgroup_analyze_uniform_statements(
            getattr(entry_function, "body", None),
            uniform_names,
        )
        self.validate_glsl_software_subgroup_helpers(
            ast,
            entry_function,
            helper_records,
            tuple(concrete_workgroup_size),
            target_stage=target_stage,
        )
        for function in self.glsl_software_subgroup_functions(ast):
            if (
                function is entry_function
                or id(function) in self.glsl_software_subgroup_helper_function_ids
            ):
                self.validate_glsl_software_subgroup_exits(
                    getattr(function, "body", None),
                    operation=all_records[0][0],
                    **self.glsl_software_subgroup_exit_facts(
                        function,
                        (
                            self.glsl_software_subgroup_immutable_uniform_names
                            if function is entry_function
                            else self.glsl_software_subgroup_function_uniform_seeds.get(
                                id(function), set()
                            )
                        ),
                        function is entry_function,
                    ),
                )

    def glsl_software_subgroup_exit_uniform_expression(self, node, names, components):
        if isinstance(node, LiteralNode):
            return True
        if isinstance(node, IdentifierNode):
            return node.name in names
        load = self.glsl_software_subgroup_uniform_returns.storage_load(node)
        if isinstance(node, ArrayAccessNode) or load is not None:
            base, index = load or (node.array_expr, node.index_expr)
            if (
                not isinstance(base, IdentifierNode)
                or base.name not in self.glsl_software_subgroup_resource_names
            ):
                return False
            children = [base, index]
        elif isinstance(node, (MemberAccessNode, SwizzleNode)):
            base = (
                node.object_expr
                if isinstance(node, MemberAccessNode)
                else node.vector_expr
            )
            member = (
                node.member if isinstance(node, MemberAccessNode) else node.components
            )
            if isinstance(base, IdentifierNode) and base.name in components:
                positions = {
                    letter: index
                    for alphabet in ("xyzw", "rgba")
                    for index, letter in enumerate(alphabet)
                }
                return bool(member) and all(
                    positions.get(letter) in components[base.name] for letter in member
                )
            children = [base]
        elif isinstance(node, FunctionCallNode):
            children = self.glsl_software_subgroup_uniform_returns.uniform_arguments(
                node, self.glsl_software_subgroup_resource_names
            )
            if children is None:
                return False
        elif isinstance(node, ConstructorNode):
            children = [*node.arguments, *node.named_arguments.values()]
        elif isinstance(node, CastNode):
            if isinstance(node.target_type, (PointerType, ReferenceType)):
                return False
            children = [node.expression]
        elif isinstance(node, BinaryOpNode):
            children = [node.left, node.right]
        elif isinstance(node, UnaryOpNode) and self.map_operator(node.operator) in {
            "+",
            "-",
            "!",
            "~",
        }:
            children = [node.operand]
        elif isinstance(node, TernaryOpNode):
            children = [node.condition, node.true_expr, node.false_expr]
        else:
            return False
        return all(
            self.glsl_software_subgroup_exit_uniform_expression(
                child, names, components
            )
            for child in children
        )

    def glsl_software_subgroup_uniform_arguments(
        self, function, seeds, *, entry, inherited_bounds=None
    ):
        facts = self.glsl_software_subgroup_exit_facts(function, seeds, entry)
        metadata_bounds = (
            metadata_loop_bindings(
                function,
                self.metadata_preconditions,
                facts["mutable_names"],
                self.walk_ast,
            )
            if entry
            else inherited_metadata_bindings(
                function, inherited_bounds, facts["mutable_names"], self.walk_ast
            )
        )
        self.glsl_software_subgroup_function_metadata[id(function)] = metadata_bounds
        self.glsl_software_subgroup_loop_metadata.update(
            (id(node), metadata_bounds)
            for node in self.walk_ast(function.body)
            if isinstance(node, ForNode)
        )
        self.glsl_software_subgroup_loop_constants.update(
            loop_constant_bindings(
                function,
                self.initial_literal_int_constants(function),
                facts["mutable_names"],
                self.type_name_string,
            )
        )
        arguments = {}
        analysis = self.glsl_software_subgroup_uniform_returns
        followed = facts["followed_names"]
        guards = getattr(self, "glsl_software_subgroup_constant_guards", {})

        def declares_uniform(statement, names, components):
            return (
                (
                    statement.name not in facts["mutable_names"]
                    or statement.name in followed
                    or statement.name in facts["redeclared_names"]
                )
                and (
                    analysis.value_type(statement.var_type)
                    or statement.name in self.glsl_software_subgroup_resource_names
                )
                and self.glsl_software_subgroup_exit_uniform_expression(
                    statement.initial_value, names, components
                )
            )

        flow = ReassignmentFlow(
            followed,
            walk=self.walk_ast,
            map_operator=self.map_operator,
            uniform_expression=self.glsl_software_subgroup_exit_uniform_expression,
            declares_uniform=declares_uniform,
            offsets=facts["offset_names"],
            carried=facts["carried_names"],
            trip_uniform=lambda loop, names, components: (
                self.glsl_software_subgroup_uniform_for(loop, names)
            ),
            counter_loop=self.glsl_software_subgroup_uniform_counter,
        )

        def visit(body, names, components):
            previous = None
            for statement in self.glsl_software_subgroup_statement_list(body):
                if previous is not None:
                    flow.follow(previous, names, components)
                previous = statement
                if isinstance(statement, VariableNode):
                    names.discard(statement.name)
                    components.pop(statement.name, None)
                    if declares_uniform(statement, names, components):
                        names.add(statement.name)
                if isinstance(statement, BlockNode) or hasattr(statement, "statements"):
                    visit(statement, set(names), dict(components))
                    continue
                if isinstance(statement, IfNode):
                    fixed = guards.get(id(statement))
                    if fixed is not None:
                        # Every invocation takes the branch the constant selects.
                        self.glsl_software_subgroup_uniform_if_node_ids.add(
                            id(statement)
                        )
                        visit(
                            statement.if_body if fixed else statement.else_body,
                            set(names),
                            dict(components),
                        )
                        continue
                    conditions = [
                        statement.condition,
                        *(getattr(statement, "else_if_conditions", []) or []),
                    ]
                    if all(
                        self.glsl_software_subgroup_exit_uniform_expression(
                            condition, names, components
                        )
                        for condition in conditions
                    ):
                        self.glsl_software_subgroup_uniform_if_node_ids.add(
                            id(statement)
                        )
                        for branch in [
                            statement.if_body,
                            *(getattr(statement, "else_if_bodies", []) or []),
                            getattr(statement, "else_body", None),
                        ]:
                            visit(branch, set(names), dict(components))
                    continue
                if isinstance(statement, ForNode):
                    if self.glsl_software_subgroup_uniform_for(statement, names):
                        self.glsl_software_subgroup_uniform_for_node_ids.add(
                            id(statement)
                        )
                        # A proven initializer-free counter loop has no init.
                        loop_name = getattr(statement.init, "name", None)
                        trip_names, _kept = flow.carried_trip(
                            statement, names, components
                        )
                        nested_names = trip_names | ({loop_name} - {None})
                        nested_components = dict(components)
                        nested_components.pop(loop_name, None)
                        visit(statement.body, nested_names, nested_components)
                    elif (
                        entry
                        and id(statement)
                        not in self.glsl_software_subgroup_strided_for_plans
                        and self.glsl_software_subgroup_trip_uniform_for(
                            statement, names
                        )
                    ):
                        # Every invocation runs the same trips, but the counter
                        # differs between subgroups.
                        self.glsl_software_subgroup_uniform_for_node_ids.add(
                            id(statement)
                        )
                        nested_components = dict(components)
                        nested_components.pop(statement.init.name, None)
                        visit(
                            statement.body,
                            set(names)
                            - {statement.init.name}
                            - loop_carried_names(
                                statement,
                                facts["carried_names"],
                                self.walk_ast,
                                self.map_operator,
                            ),
                            nested_components,
                        )
                    continue
                if isinstance(
                    statement,
                    (
                        WhileNode,
                        DoWhileNode,
                        ForInNode,
                        LoopNode,
                        SwitchNode,
                        MatchNode,
                    ),
                ):
                    continue
                for call in self.walk_ast(statement):
                    if not isinstance(call, FunctionCallNode):
                        continue
                    guarded = any(
                        (
                            isinstance(parent, TernaryOpNode)
                            or (
                                isinstance(parent, BinaryOpNode)
                                and self.map_operator(parent.op) in {"&&", "||"}
                            )
                        )
                        and any(child is call for child in self.walk_ast(parent))
                        for parent in self.walk_ast(statement)
                    )
                    if not guarded:
                        arguments[id(call)] = {
                            index
                            for index, argument in enumerate(call.arguments)
                            if self.glsl_software_subgroup_exit_uniform_expression(
                                argument, names, components
                            )
                        }

        visit(function.body, facts["uniform_names"], facts["uniform_components"])
        return arguments

    def validate_glsl_software_subgroup_exits(
        self,
        body,
        later_work=False,
        *,
        operation,
        uniform_names=None,
        uniform_components=None,
        mutable_names=None,
        offset_names=None,
        followed_names=None,
        carried_names=None,
        redeclared_names=None,
    ):
        # Branches a constant guard rules out never run.
        dead_ids = getattr(self, "glsl_software_subgroup_dead_node_ids", set())
        guards = getattr(self, "glsl_software_subgroup_constant_guards", {})

        def contains_work(root):
            return any(
                id(record[1]) not in dead_ids
                for record in self.glsl_software_subgroup_operation_records(root)
            ) or any(
                isinstance(node, FunctionCallNode)
                and id(node) not in dead_ids
                and self.glsl_software_subgroup_candidate_helper_call(node)
                for node in self.glsl_software_subgroup_reachable_nodes(root)
            )

        statements = self.glsl_software_subgroup_statement_list(body)
        uniform_names = set(uniform_names or ())
        uniform_components = dict(uniform_components or {})
        mutable_names = set(mutable_names or ())
        offset_names = set(offset_names or ())
        followed_names = set(followed_names or ())

        def declares_uniform(statement, names, components):
            return (
                (
                    statement.name not in mutable_names
                    or statement.name in followed_names
                    or statement.name in (redeclared_names or ())
                )
                and (
                    self.glsl_software_subgroup_uniform_returns.value_type(
                        statement.var_type
                    )
                    or statement.name in self.glsl_software_subgroup_resource_names
                )
                and self.glsl_software_subgroup_exit_uniform_expression(
                    statement.initial_value, names, components
                )
            )

        # Offsets of immutable entry storage and followed locals' writes.
        flow = ReassignmentFlow(
            followed_names,
            walk=self.walk_ast,
            map_operator=self.map_operator,
            uniform_expression=self.glsl_software_subgroup_exit_uniform_expression,
            declares_uniform=declares_uniform,
            offsets=offset_names,
            carried=carried_names or (),
            trip_uniform=lambda loop, names, components: (
                self.glsl_software_subgroup_uniform_for(loop, names)
            ),
            counter_loop=self.glsl_software_subgroup_uniform_counter,
        )
        for index, statement in enumerate(statements):
            if index and (offset_names or followed_names):
                flow.follow(statements[index - 1], uniform_names, uniform_components)
            if isinstance(statement, VariableNode):
                uniform_names.discard(statement.name)
                uniform_components.pop(statement.name, None)
                if declares_uniform(statement, uniform_names, uniform_components):
                    uniform_names.add(statement.name)
            subsequent_work = later_work or any(
                contains_work(item) for item in statements[index + 1 :]
            )
            has_work = contains_work(statement)
            has_exit = any(
                isinstance(node, (ReturnNode, BreakNode, ContinueNode))
                and id(node) not in dead_ids
                for node in self.glsl_software_subgroup_reachable_nodes(statement)
            )
            fixed = guards.get(id(statement)) if isinstance(statement, IfNode) else None
            if fixed is not None:
                # Every invocation takes the branch the constant selects.
                self.validate_glsl_software_subgroup_exits(
                    statement.if_body if fixed else statement.else_body,
                    subsequent_work,
                    operation=operation,
                    uniform_names=uniform_names,
                    uniform_components=uniform_components,
                    mutable_names=mutable_names,
                    offset_names=offset_names,
                    followed_names=followed_names,
                    carried_names=carried_names,
                    redeclared_names=redeclared_names,
                )
            elif isinstance(statement, IfNode):
                conditions = [
                    statement.condition,
                    *(getattr(statement, "else_if_conditions", []) or []),
                ]
                if (
                    has_exit
                    and (subsequent_work or has_work)
                    and not all(
                        self.glsl_software_subgroup_exit_uniform_expression(
                            condition, uniform_names, uniform_components
                        )
                        for condition in conditions
                    )
                ):
                    raise self.glsl_software_subgroup_error(
                        "OpenGL software subgroup barriers cannot follow an "
                        "unproven divergent return, break, or continue",
                        reason="potentially-divergent-control-flow",
                        operation=operation,
                        source_location=getattr(statement, "source_location", None),
                    )
                for branch in [
                    statement.if_body,
                    *(getattr(statement, "else_if_bodies", []) or []),
                    getattr(statement, "else_body", None),
                ]:
                    self.validate_glsl_software_subgroup_exits(
                        branch,
                        subsequent_work,
                        operation=operation,
                        uniform_names=uniform_names,
                        uniform_components=uniform_components,
                        mutable_names=mutable_names,
                        offset_names=offset_names,
                        followed_names=followed_names,
                        carried_names=carried_names,
                        redeclared_names=redeclared_names,
                    )
            elif isinstance(
                statement,
                (
                    ForNode,
                    WhileNode,
                    DoWhileNode,
                    ForInNode,
                    LoopNode,
                    SwitchNode,
                    MatchNode,
                ),
            ):
                if has_exit and (subsequent_work or has_work):
                    raise self.glsl_software_subgroup_error(
                        "OpenGL software subgroup barriers require proven "
                        "participation across loop and switch exits",
                        reason="potentially-divergent-control-flow",
                        operation=operation,
                        source_location=getattr(statement, "source_location", None),
                    )
            elif isinstance(statement, BlockNode) or hasattr(statement, "statements"):
                self.validate_glsl_software_subgroup_exits(
                    statement,
                    subsequent_work,
                    operation=operation,
                    uniform_names=uniform_names,
                    uniform_components=uniform_components,
                    mutable_names=mutable_names,
                    offset_names=offset_names,
                    followed_names=followed_names,
                    carried_names=carried_names,
                    redeclared_names=redeclared_names,
                )

    def glsl_software_subgroup_uniform_for(self, node, uniform_names=None):
        if self.software_subgroup_width is None:
            return True
        initializer = getattr(node, "init", None)
        if initializer is None:
            return self.glsl_software_subgroup_counter_loop_uniform(node, uniform_names)
        if not isinstance(initializer, VariableNode):
            return False
        loop_name = getattr(initializer, "name", None)
        loop_type = self.map_type(getattr(initializer, "var_type", None))
        if not loop_name or loop_type not in {"int", "uint", "int16_t", "uint16_t"}:
            return False
        constants = dict(
            getattr(self, "glsl_software_subgroup_loop_constants", {}).get(
                id(node), self.glsl_active_literal_int_constants()
            )
        )
        constants.pop(loop_name, None)
        metadata_bounds = getattr(self, "glsl_software_subgroup_loop_metadata", {}).get(
            id(node), {}
        )
        if uniform_names is not None:
            constants = {
                name: value
                for name, value in constants.items()
                if name in uniform_names
            }
        if self.glsl_software_subgroup_static_for(node, constants):
            return True
        initial_expression = getattr(initializer, "initial_value", None)
        initial_value = evaluate_literal_int_expression(initial_expression, constants)
        if initial_value is None and (
            uniform_names is None
            or not self.glsl_software_subgroup_uniform_expression(
                initial_expression,
                uniform_names,
            )
        ):
            return False

        condition = getattr(node, "condition", None)
        if not isinstance(condition, BinaryOpNode):
            return False
        operator = self.map_operator(
            getattr(condition, "op", getattr(condition, "operator", None))
        )
        if operator not in {"<", "<=", ">", ">="}:
            return False
        left_name = self.expression_name(condition.left)
        right_name = self.expression_name(condition.right)
        if left_name == loop_name and right_name != loop_name:
            bound = condition.right
            loop_on_left = True
        elif right_name == loop_name and left_name != loop_name:
            bound = condition.left
            loop_on_left = False
        else:
            return False
        literal_bound = evaluate_literal_int_expression(bound, constants)
        if literal_bound is None and (
            uniform_names is None
            or not self.glsl_software_subgroup_uniform_expression(
                bound,
                set(uniform_names) | {loop_name},
            )
        ):
            return False

        update = getattr(node, "update", None)
        halving_update = False
        additive_update = False
        geometric_update = False
        modular_update = False
        if isinstance(update, UnaryOpNode):
            update_operator = self.map_operator(
                getattr(update, "op", getattr(update, "operator", None))
            )
            if update_operator not in {"++", "--"}:
                return False
            if self.expression_name(getattr(update, "operand", None)) != loop_name:
                return False
            increasing = update_operator == "++"
        elif isinstance(update, AssignmentNode):
            update_operator = self.map_operator(
                getattr(update, "op", getattr(update, "operator", None))
            )
            if (
                update_operator not in {"/=", ">>=", "+=", "-=", "*=", "<<="}
                or self.glsl_software_subgroup_assignment_target_name(update)
                != loop_name
            ):
                return False
            update_value = getattr(update, "value", getattr(update, "right", None))
            concrete_update_value = evaluate_literal_int_expression(
                update_value,
                constants,
            )
            if concrete_update_value is None:
                # A modular update visits the same values in every invocation
                # for any uniform operand, so it needs no operand range.
                if (
                    update_operator != "+="
                    or uniform_names is None
                    or id(node)
                    not in getattr(
                        self, "glsl_software_subgroup_modular_stride_loops", ()
                    )
                    or not self.glsl_software_subgroup_uniform_expression(
                        update_value, set(uniform_names) - {loop_name}
                    )
                ):
                    return False
                increasing = True
                modular_update = True
            elif update_operator in {"+=", "-="}:
                step = concrete_update_value * (1 if update_operator == "+=" else -1)
                increasing = step > 0
                additive_update = True
            elif update_operator in {"*=", "<<="}:
                increasing = True
                geometric_update = True
            else:
                if (
                    update_operator == "/="
                    and concrete_update_value < 2
                    or update_operator == ">>="
                    and concrete_update_value < 1
                ):
                    return False
                increasing = False
                halving_update = True
        else:
            return False
        normalized_operator = (
            operator
            if loop_on_left
            else {
                "<": ">",
                "<=": ">=",
                ">": "<",
                ">=": "<=",
            }[operator]
        )
        if halving_update:
            # Integer division or right shift is only a termination proof for
            # canonical positive-to-zero reduction loops.  Integral
            # ``value > 0`` and ``value >= 1`` are equivalent; admitting both
            # preserves source spellings such as Metal's
            # ``for (ushort off = ...; off >= 1; off >>= 1)`` without opening
            # wider bounds that can skip required reduction rounds.  Other
            # bounds can converge to a fixed point that still satisfies the
            # condition.
            canonical_positive_bound = (
                normalized_operator == ">" and literal_bound == 0
            ) or (normalized_operator == ">=" and literal_bound == 1)
            if not canonical_positive_bound:
                return False
        elif increasing != (normalized_operator in {"<", "<="}):
            return False
        if not modular_update and (
            additive_update
            or geometric_update
            or metadata_bound_key(bound) in metadata_bounds
            or (
                isinstance(update, UnaryOpNode)
                and narrow_integer_shape(self.type_name_string(initializer.var_type))
                is not None
            )
        ):
            proof = (
                geometric_loop_terminates
                if geometric_update
                else fixed_stride_loop_terminates
            )
            if any(
                self.glsl_software_subgroup_uniform_returns._qualified(
                    value, {"volatile"}
                )
                for value in (initializer, initializer.var_type)
            ) or not proof(
                self.type_name_string(initializer.var_type),
                initial_expression,
                bound,
                update,
                constants,
                normalized_operator,
                **(
                    {
                        "metadata_bounds": metadata_bounds,
                        "any_start": initial_value is None,
                    }
                    if proof is fixed_stride_loop_terminates
                    else {}
                ),
            ):
                return False

        bound_names = self.glsl_software_subgroup_dependency_names(bound)
        stride_names = (
            self.glsl_software_subgroup_dependency_names(update.value)
            if additive_update or geometric_update or modular_update
            else set()
        )
        return self.glsl_software_subgroup_loop_controls_preserved(
            node, bound_names | stride_names | {loop_name}
        )

    def glsl_software_subgroup_static_for(self, node, constants):
        """Accept a loop whose controls read only literal constants.

        MLX's ``for (int step = 1; (1 << step) < C; step++)`` runs the same
        checked trips in every invocation when its body leaves the counter
        and the constants alone.
        """
        initializer = getattr(node, "init", None)
        if not isinstance(initializer, VariableNode) or any(
            self.glsl_software_subgroup_uniform_returns._qualified(value, {"volatile"})
            for value in (initializer, initializer.var_type)
        ):
            return False
        trips = static_loop_trip_count(
            self.type_name_string(initializer.var_type),
            initializer.initial_value,
            getattr(node, "condition", None),
            getattr(node, "update", None),
            initializer.name,
            constants,
        )
        if trips is None:
            return False
        controls = {initializer.name}
        for part in (node.condition, node.update):
            controls |= self.glsl_software_subgroup_expression_identifier_names(part)
        return self.glsl_software_subgroup_loop_controls_preserved(node, controls)

    def glsl_software_subgroup_loop_controls_preserved(self, node, control_names):
        """Whether ``node``'s body leaves its counter, bound and stride alone."""
        for child, visible in walk_control_scopes(
            getattr(node, "body", None), control_names
        ):
            if isinstance(child, (BreakNode, ContinueNode, ReturnNode)):
                return False
            if isinstance(child, VariableNode):
                if child.name in visible:
                    return False
                if (
                    isinstance(child.var_type, (PointerType, ReferenceType))
                    and self.glsl_software_subgroup_call_mutations.get(
                        id(child),
                        self.glsl_software_subgroup_expression_identifier_names(
                            child.initial_value
                        ),
                    )
                    & visible
                ):
                    return False
            if isinstance(child, AssignmentNode):
                target_name = self.glsl_software_subgroup_assignment_target_name(child)
                if target_name in visible:
                    return False
            if isinstance(child, UnaryOpNode) and self.map_operator(
                getattr(child, "op", getattr(child, "operator", None))
            ) in {"++", "--", "&"}:
                target_name = self.expression_name(getattr(child, "operand", None))
                if target_name in visible:
                    return False
            if isinstance(child, FunctionCallNode):
                observed = self.glsl_software_subgroup_call_mutations.get(
                    id(child),
                    (
                        set()
                        if self.glsl_software_subgroup_exit_constructor(child)
                        else self.glsl_software_subgroup_expression_identifier_names(
                            child
                        )
                    ),
                )
                if observed & visible:
                    return False
        return True

    def glsl_wave_extension_lines(self, ast, target_stage=None):
        uses_matrix_multiply_accumulate = (
            self.glsl_software_cooperative_matrix_uses_multiply_accumulate(
                ast, target_stage
            )
        )
        if self.software_subgroup_width is not None:
            return []
        operations = self.glsl_wave_operations(ast, target_stage)
        lines = []
        if (
            self.glsl_stage_requires_exact_subgroup_width(ast, target_stage)
            or (self.uses_glsl_subgroup_basic_builtin(ast, target_stage))
            or any(
                operation in self.GLSL_WAVE_EXTENSION_REQUIREMENTS
                for operation in operations
            )
        ):
            lines.append(self.GLSL_SUBGROUP_BASIC_EXTENSION_LINE)
        for operation in self.GLSL_WAVE_INTRINSIC_ARITIES:
            if operation not in operations:
                continue
            required = self.GLSL_WAVE_EXTENSION_REQUIREMENTS.get(operation) or ""
            for line in required.splitlines():
                if line not in lines:
                    lines.append(line)
            if operation in self.GLSL_WAVE_ADDITIONAL_EXTENSION_REQUIREMENTS:
                shuffle_line = "#extension GL_KHR_shader_subgroup_shuffle : require"
                if shuffle_line not in lines:
                    lines.append(shuffle_line)
        if uses_matrix_multiply_accumulate:
            if self.GLSL_SUBGROUP_BASIC_EXTENSION_LINE not in lines:
                lines.append(self.GLSL_SUBGROUP_BASIC_EXTENSION_LINE)
            shuffle_line = "#extension GL_KHR_shader_subgroup_shuffle : require"
            if shuffle_line not in lines:
                lines.append(shuffle_line)
        return lines

    def glsl_wave_size_attribute(self, attribute):
        name = str(getattr(attribute, "name", "")).strip().lower()
        if name.startswith("hlsl_"):
            name = name[len("hlsl_") :]
        return name == "wavesize"

    def _generate_program_attempt(self, ast, target_stage, early_exits):
        ast = lower_resource_aggregates(
            ast,
            storage_pointer_parameters=True,
            workgroup_access_assertions=self.workgroup_access_assertions,
        )
        ast = lower_reference_aliases(ast)
        ast = lower_private_vector_views(ast, "opengl")
        separate_ast = ast
        ast, self.combined_sampler_records = lower_combined_sampler_pairs(
            ast, self.LOWERING_REFUSAL_TYPE.label
        )
        self.combined_sampler_pairs = (getattr(ast, "annotations", None) or {}).get(
            COMBINED_SAMPLER_PAIRS, []
        )
        self.glsl_vulkan_resources = VulkanResourceForms.plan(self, separate_ast, ast)
        self.glsl_narrow_scalars_promote = narrow_scalars_promote(ast)
        ast = self.with_glsl_builtin_option_prelude(ast)
        if self.software_subgroup_width is not None:
            ast = converge_subgroup_divergent_barriers(
                ast,
                self.glsl_software_subgroup_stage_entries(ast, target_stage),
                self.walk_ast,
                self.map_operator,
            )
            if early_exits:
                ast = converge_subgroup_early_exits(
                    ast,
                    self.glsl_software_subgroup_stage_entries(ast, target_stage),
                    self.walk_ast,
                    self.map_operator,
                )
            ast = converge_subgroup_guarded_returns(
                ast, self.walk_ast, self.map_operator
            )
            ast = converge_subgroup_bounded_loops(
                ast,
                self.glsl_software_subgroup_stage_entries(ast, target_stage),
                self.walk_ast,
                self.map_operator,
            )
            ast = converge_subgroup_conditional_calls(
                ast,
                self.glsl_software_subgroup_stage_entries(ast, target_stage),
                self.walk_ast,
                self.map_operator,
            )
        target_stage = requested_stage_name(target_stage)
        self.glsl_stage_reachable_function_names = self.stage_reachable_function_names(
            ast, target_stage
        )
        self.global_type_aliases = self.collect_glsl_type_aliases(ast)
        self.current_type_aliases = dict(self.global_type_aliases)
        self.glsl_builtin_option_available = self.glsl_ast_declares_type(ast, "Option")
        self.validate_stage_main_helper_conflict(ast, target_stage)
        refuse_repeated_stage_entries(
            ast, target_stage, self.combined_stage_entry_types()
        )

        self.sampler_variables = set()
        self.current_sampler_parameters = set()
        self.glsl_resource_binding_records = []
        self.glsl_combined_sampler_uses = []
        self.glsl_sampler_registers = {}
        self.glsl_float_remainder_types = set()
        self.glsl_precise_float_remainder_types = set()
        self.glsl_integer_dot_types = set()
        self.reset_glsl_emulated_matrices(ast)
        self.reset_glsl_hlsl_structured_buffers()
        self.reset_glsl_hlsl_cbuffers()
        self.glsl_image_usages = self.collect_glsl_image_usages(ast)
        self.glsl_unknown_format_images = set()
        self.glsl_formatted_image_load_names = set()
        self.glsl_write_only_unknown_format_images = set()
        self.texture_variable_types = {}
        self.resource_variable_array_sizes = {}
        self.current_texture_parameters = {}
        self.current_resource_aliases = {}
        self.current_glsl_workgroup_pointer_aliases = {}
        self.current_glsl_storage_pointer_aliases = {}
        self.glsl_global_workgroup_pointer_aliases = {}
        self.glsl_global_storage_pointer_aliases = {}
        self.glsl_storage_pointer_access_requirement_cache = {}
        self.glsl_storage_pointer_access_requirement_active = set()
        self.current_glsl_synthetic_local_names = set()
        self.current_glsl_scalar_array_view_plans = {}
        self.current_glsl_scalar_array_views = {}
        self.glsl_fixed_array_vector_pointer_view_helper_ids = set()
        self.glsl_fixed_array_vector_pointer_view_calls = {}
        self.glsl_workgroup_pointer_scope_depth = 0
        self.image_variable_formats = {}
        self.current_image_format_parameters = {}
        self.image_variable_accesses = {}
        self.current_image_access_parameters = {}
        self.structured_buffer_variable_accesses = {}
        self.current_structured_buffer_access_parameters = {}
        self.glsl_buffer_block_variable_names = set()
        self.glsl_buffer_block_variable_types = {}
        self.glsl_buffer_block_variable_accesses = {}
        self.glsl_uniform_block_variable_types = {}
        self.glsl_interface_value_struct_names = set()
        self.glsl_interface_block_names_by_qualifier = {}
        self.global_variable_types = {}
        self.glsl_global_identifier_names = self.collect_glsl_global_identifier_names(
            ast, target_stage
        )
        self.glsl_module_identifier_names = {}
        self.glsl_module_generated_identifier_names = {}
        self.glsl_module_used_identifier_names = set()
        self.glsl_type_identifier_names = {}
        self.glsl_generic_enum_constructor_names = {}
        self.glsl_compile_time_global_node_ids = set()
        self.glsl_compile_time_bitcast_substitution_node_ids = set()
        self.glsl_compile_time_bitcast_substitutions = {}
        self.glsl_active_compile_time_bitcast_substitutions = set()
        self.glsl_runtime_global_initializer_node_ids = set()
        self.glsl_stage_runtime_global_initializers = {}
        self.glsl_unresolved_call_targets = {}
        self.function_sampler_parameter_indices = (
            self.collect_function_sampler_parameter_indices(ast)
        )
        functions = self.collect_functions(ast)
        self.glsl_metal_simd_group_placeholder_functions = (
            self.collect_metal_simd_group_placeholder_functions(functions)
        )
        self.preserved_sampler_variables = set()
        self.preserved_sampler_parameters = {}
        self.function_return_types = {
            func.name: self.type_name_string(getattr(func, "return_type", "void"))
            for func in functions
            if getattr(func, "name", None)
        }
        self.function_parameter_names = collect_function_parameter_names(functions)
        self.function_parameter_types = self.collect_function_parameter_types(functions)
        self.function_parameter_infos = self.collect_function_parameter_infos(functions)
        self.function_definitions = {
            func.name: func for func in functions if getattr(func, "name", None)
        }
        self.validate_glsl_atomic_fences_before_resource_qualifiers(ast, target_stage)
        self.validate_typed_resource_memory_qualifier_contracts(ast, target_stage)
        self.glsl_function_overloads_by_name = {}
        self.glsl_function_target_names = {}
        self.current_glsl_function_key = None
        self.current_glsl_source_function_id = None
        self.current_glsl_source_function_name = None
        self.glsl_function_stage_contexts = {}
        self.glsl_function_call_graph = {}
        self.glsl_function_call_locations = {}
        self.glsl_function_call_source_edges = {}
        self.glsl_generic_helper_refusals = {}
        self.glsl_generic_helper_code = {}
        self.collect_preserved_nonuniform_sampler_resources(ast, functions)
        self.validate_glsl_tessellation_control_barriers(ast, target_stage, functions)
        self.glsl_resource_function_specializations = {}
        self.glsl_resource_function_call_names = {}
        self.glsl_resource_specialized_source_names = set()
        self.glsl_resource_function_call_plans = {}
        self.glsl_resource_function_semantic_call_plans = {}
        self.current_glsl_resource_function_call_semantic_locations = {}
        self.emitted_glsl_resource_specialization_names = set()
        self.function_private_pointer_array_size_hints = {}
        self.function_private_pointer_required_spans = {}
        self.function_private_pointer_parameters = {}
        self.function_private_pointer_parameter_indices = {}
        self.function_private_pointer_element_types = {}
        self.function_private_pointer_read_only_parameters = {}
        self.function_private_pointer_backing_variants = {}
        self.function_private_pointer_local_arrays = {}
        self.function_private_pointer_view_calls = {}
        self.function_private_pointer_base_names = {}
        self.function_private_pointer_source_names = {}
        self.function_private_pointer_full_span_parameters = {}
        self.function_private_pointer_scalar_access_violations = {}
        self.glsl_private_pointer_robust_zero_read_variants = {}
        self.glsl_private_pointer_zero_index_access_ids = set()
        self.glsl_private_pointer_noop_rebinding_ids = set()
        self.glsl_private_pointer_base_rebinding_parameters = {}
        self.current_glsl_private_pointer_variant = {}
        self.current_glsl_private_pointer_base_names = {}
        self.current_glsl_private_pointer_function_name = None
        self.glsl_global_specialization_constants = []
        self.glsl_specialization_constant_declarations = []
        self.glsl_runtime_specialization_dependent_ids = set()
        self.function_image_access_requirements = (
            collect_function_image_access_requirements(
                functions,
                self.function_parameter_names,
                self.walk_ast,
                self.function_call_name,
                self.expression_name,
            )
        )
        self.function_structured_buffer_access_requirements = (
            self.collect_function_structured_buffer_access_requirements(functions)
        )
        constants = list(getattr(ast, "constants", []) or [])
        self.glsl_constant_types = {
            node.name: self.type_name_string(
                self.glsl_specialization_constant_type(node)
            )
            for node in constants
            if getattr(node, "name", None)
        }
        self.glsl_specialization_constant_names = {
            node.name
            for node in constants + list(getattr(ast, "global_variables", []) or [])
            if getattr(node, "name", None)
            and self.glsl_specialization_constant_attributes(node)
        }
        # Constants derived from a specialization default remain runtime-dependent.
        self.literal_int_constants = collect_literal_int_constants(
            [
                node
                for node in constants
                if getattr(node, "name", None)
                not in self.glsl_specialization_constant_names
            ]
        )
        # The same constants with every specialization constant at its default value, for checks whose answer
        # cannot depend on the value.
        self.glsl_specialization_default_int_constants = (
            collect_literal_int_constants(constants)
            if self.glsl_specialization_constant_names
            else self.literal_int_constants
        )
        self.literal_int_vector_constants = (
            self.collect_literal_int_vector_constants_from_variables(
                getattr(ast, "global_variables", []),
                self.literal_int_constants,
            )
        )
        self.literal_int_vector_array_constants = (
            self.collect_literal_int_vector_array_constants_from_variables(
                getattr(ast, "global_variables", []),
                self.literal_int_constants,
                self.literal_int_vector_constants,
            )
        )
        self.current_compile_time_int_constants = dict(self.literal_int_constants)
        self.current_compile_time_int_vector_constants = dict(
            self.literal_int_vector_constants
        )
        self.current_compile_time_int_vector_array_constants = dict(
            self.literal_int_vector_array_constants
        )
        self.current_stage_output = None
        self.current_stage_inputs = {}
        self.current_stage_outputs = {}
        self.current_stage_output_member_map = {}
        self.current_stage_parameter_aliases = {}
        self.stage_entry_functions_by_stage = {}
        self.stage_entry_resource_parameter_aliases = {}
        self.stage_entry_storage_pointer_bindings = {}
        self.stage_entry_constant_value_parameter_aliases = {}
        self.stage_entry_resource_declaration_names = set()
        self.stage_entry_constant_value_cbuffer_names = set()
        self.current_identifier_aliases = {}
        self.current_glsl_local_identifier_names = {}
        self.glsl_hoisted_shared_declarations = []
        self.glsl_hoisted_shared_aliases_by_declaration = {}
        self.glsl_hoisted_shared_declaration_ids = set()
        self.glsl_hoisted_shared_alias_names = set()
        self.glsl_shared_dimension_constants = {}
        self.current_target_stage = target_stage
        self.glsl_tessellation_input_extents = {}
        self.current_glsl_extensions = self.glsl_preprocessor_extensions(ast)
        self.glsl_fragment_interlock_plan = plan_fragment_interlock(
            self, ast, target_stage
        )
        self.glsl_vulkan_memory_model = declares_vulkan_memory_model(
            getattr(ast, "preprocessors", None), self.current_glsl_extensions
        )
        self.glsl_shadow_lod_extension_used = False
        self.glsl_stage_builtin_extensions = []
        stage_names = {
            stage_name
            for stage_name in self.glsl_stage_names(ast, target_stage)
            if stage_name is not None
        }
        self.current_glsl_stage_names = stage_names
        if target_stage is not None:
            self.current_global_stage_io_stage = target_stage
        elif len(stage_names) == 1:
            self.current_global_stage_io_stage = next(iter(stage_names))
        else:
            self.current_global_stage_io_stage = None
        self.emit_stage_guards = self.should_emit_stage_guards(ast, target_stage)
        self.task_payload_shared_variables = []
        self.stage_io_used_locations = {}
        self.stage_io_used_components = {}
        self.stage_io_declarations = {}
        self.flattened_stage_variables = set()
        self.fragment_output_member_layout_maps = {}
        self.fragment_direct_output_name_maps = {}
        self.fragment_blend_support_layout_parts = []
        self.current_function_return_type = None
        self.current_glsl_precise_return_type = None
        self.current_glsl_precise_function = False
        self.current_glsl_precise_captures = None
        self.current_glsl_ieee_operator_arithmetic = False
        self.current_glsl_exactness_test_local_ids = {}
        self.current_glsl_precise_value_writes = {}
        self.current_stage_return_type = None
        self.current_stage_entry_type = None
        fence_stage_names = self.glsl_stage_names(ast, target_stage) - {None}
        self.glsl_atomic_fence_stage = (
            next(iter(fence_stage_names)) if len(fence_stage_names) == 1 else None
        )
        self.function_fragment_only_requirements = {}
        self.current_expression_expected_type = None
        self.current_discarded_expression = None
        self.match_temp_variable_index = 0
        self.local_variable_types = {}
        self.local_variable_source_types = {}
        self.required_glsl_complex64_helpers = set()
        self.required_glsl_boolean_order_helpers = set()
        self.required_glsl_metal_math_helpers = set()
        self.glsl_exponent_helpers = {}
        self.glsl_half_helper_names = {}
        self.glsl_half_bitcast_helper_names = {}
        self.glsl_bfloat_helper_names = {}
        self.glsl_bfloat_integer_helper_name = None
        self.glsl_bfloat_wide_integer_helper_required = False
        self.glsl_float_selection_helper_names = {}
        self.glsl_storage_structure_copy_helper_names = {}
        self.glsl_integer_float_helper_names = {}
        self.glsl_signed_remainder_helper_names = {}
        self.glsl_floating_remainder_helper_names = {}
        self.glsl_generating_global_initializer = False
        self.required_glsl_trailing_zero_helpers = set()
        self.glsl_expected_compare_helpers = {}
        self.glsl_trailing_zero_helper_names = {}
        self.glsl_trailing_zero_reserved_names = set()
        self.required_glsl_software_subgroup_helpers = set()
        self.glsl_software_subgroup_entry_function_id = None
        self.glsl_software_subgroup_entry_function_name = None
        self.glsl_software_subgroup_helper_function_ids = set()
        self.glsl_software_subgroup_candidate_helper_function_ids = set()
        self.glsl_software_subgroup_omitted_helper_function_ids = set()
        self.glsl_software_subgroup_uniform_for_node_ids = set()
        self.glsl_software_subgroup_zero_trip_for_node_ids = set()
        self.glsl_software_subgroup_workgroup_size = None
        self.glsl_software_subgroup_workgroup_invocation_count = None
        self.glsl_software_subgroup_count = None
        self.glsl_software_subgroup_masked_if_plans = {}
        self.glsl_software_subgroup_strided_for_plans = {}
        self.glsl_software_subgroup_id_parameter_names = set()
        self.glsl_software_subgroup_guard_names = set()
        self.glsl_software_subgroup_guard_constant_names = set()
        self.glsl_software_subgroup_result_overrides = {}
        self.current_glsl_disabled_extensions = set()
        self.glsl_cooperative_matrix_fragment_contracts = {}
        self.glsl_cooperative_matrix_fragment_contracts_by_key = {}
        self.required_glsl_cooperative_matrix_helpers = set()
        self.current_glsl_exact_subgroup_width = None
        self.current_structured_buffer_array_parameters = {}
        self.current_structured_buffer_counter_parameters = {}
        self.current_structured_buffer_access_parameters = {}
        self.structured_buffer_instance_members = {}
        self.structured_buffer_counter_members = {}
        self.structured_buffer_counter_instances = {}
        self.glsl_buffer_block_struct_names = set()
        self.enforce_concrete_glsl_types = False
        self.generic_type_parameter_names = self.collect_generic_type_parameter_names(
            ast
        )
        structs = deduplicate_named_declarations(
            list(getattr(ast, "structs", []) or [])
            + collect_stage_local_structs(ast, target_stage),
            "struct",
        )
        self.structs_by_name = {
            node.name: node for node in structs if self.is_struct_declaration_node(node)
        }
        self.prepare_opengl_descriptor_set_plan()
        self.struct_member_types = collect_struct_member_types(
            structs, self.type_name_string
        )
        self.remove_glsl_static_struct_member_types(structs)
        self.union_storage = UnionStorage(self, structs)
        self.struct_member_name_maps = {}
        self.generic_enum_struct_definitions = collect_generic_enum_struct_definitions(
            structs
        )
        self.generic_struct_definitions = collect_generic_struct_definitions(
            structs,
            excluded_names=set(self.generic_enum_struct_definitions),
        )
        self.generic_struct_definitions = self.glsl_instance_generic_struct_definitions(
            self.generic_struct_definitions
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
        # Generic calls bind type parameters from global arguments; the complete
        # table, built once declarations are lowered, replaces this one.
        self.global_variable_types = glsl_preliminary_global_variable_types(
            self, ast, target_stage
        )
        generic_collection_global_types = {
            **collect_constant_and_cbuffer_member_types(
                self,
                ast,
                cbuffers=self.glsl_cbuffer_nodes(ast, target_stage),
            ),
            **self.glsl_constant_types,
            **self.global_variable_types,
        }
        generic_function_specializations = prepare_generic_function_specializations(
            self,
            functions,
            global_types=generic_collection_global_types,
            ast=ast,
        )
        if generic_function_specializations:
            additional_generic_enum_specializations = (
                collect_generic_enum_specializations(
                    list(generic_function_specializations.values()),
                    self.generic_enum_struct_definitions,
                    self.type_name_string,
                )
            )
            if additional_generic_enum_specializations:
                self.generic_enum_specializations.update(
                    additional_generic_enum_specializations
                )
                self.struct_member_types.update(
                    collect_generic_enum_specialization_member_types(
                        self,
                        additional_generic_enum_specializations,
                    )
                )
            additional_generic_struct_specializations = (
                collect_generic_struct_specializations(
                    list(generic_function_specializations.values()),
                    self.generic_struct_definitions,
                    self.type_name_string,
                )
            )
            if additional_generic_struct_specializations:
                self.generic_struct_specializations.update(
                    additional_generic_struct_specializations
                )
                self.struct_member_types.update(
                    collect_generic_struct_specialization_member_types(
                        self,
                        additional_generic_struct_specializations,
                    )
                )
        self.function_return_types.update(
            {
                func.name: self.type_name_string(getattr(func, "return_type", "void"))
                for func in generic_function_specializations.values()
            }
        )
        if generic_function_specializations:
            self.function_definitions.update(
                {
                    func.name: func
                    for func in generic_function_specializations.values()
                    if getattr(func, "name", None)
                }
            )
            self.function_parameter_names.update(
                collect_function_parameter_names(
                    generic_function_specializations.values()
                )
            )
            self.function_parameter_types.update(
                self.collect_function_parameter_types(
                    generic_function_specializations.values()
                )
            )
            self.function_parameter_infos.update(
                self.collect_function_parameter_infos(
                    generic_function_specializations.values()
                )
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
        self.prepare_glsl_specialization_constants(ast)
        (
            self.resource_array_size_hints,
            self.function_resource_array_size_hints,
        ) = self.collect_resource_array_size_hints(ast)
        self.unsupported_structured_buffer_array_functions = (
            self.collect_unsupported_structured_buffer_array_functions(functions)
        )
        self.function_fragment_only_requirements = (
            self.collect_fragment_only_function_requirements(
                self.function_definitions.values()
            )
        )
        emitted_functions = self.glsl_emitted_functions_for_shared_lowering(
            ast, target_stage
        )
        identifier_functions = [
            func for func in emitted_functions if not generic_function_parameters(func)
        ] + list(generic_function_specializations.values())
        self.prepare_glsl_program_identifier_names(
            ast,
            structs,
            identifier_functions,
            target_stage,
        )
        self.prepare_glsl_trailing_zero_helper_names(functions)
        self.glsl_numeric_helper_reserved_names = {
            self.glsl_sanitized_identifier_base(name)
            for function in functions
            for name in self.collect_glsl_function_local_identifier_names(function)
        }
        self.prepare_glsl_enum_identifier_names()
        self.prepare_glsl_mapped_function_overloads(
            identifier_functions,
            reserved_names=self.glsl_overload_reserved_names(ast),
        )
        self.prepare_glsl_fixed_array_vector_pointer_views(
            ast,
            functions + list(generic_function_specializations.values()),
        )
        self.validate_global_resource_shadows(ast)
        self.validate_glsl_software_subgroup_contract(ast, target_stage)
        self.enforce_concrete_glsl_types = True
        code = ""
        preprocessors = getattr(ast, "preprocessors", []) or []
        version_line = None
        extra_lines = []
        for directive in preprocessors:
            if not self.should_emit_glsl_preprocessor_directive(directive):
                continue
            if isinstance(directive, PreprocessorNode):
                if directive.directive == "precision":
                    line = (
                        f"precision {directive.content};"
                        if directive.content
                        else "precision;"
                    )
                else:
                    line = f"#{directive.directive} {directive.content}".strip()
            else:
                line = str(directive).strip()
            if line.startswith("#version") and version_line is None:
                version_line = line
            elif line and self.should_emit_glsl_preprocessor_line(line):
                extra_lines.append(line)
        if version_line is None:
            version_line = self.default_glsl_version_line(ast, target_stage)
        required_version_line = getattr(self, "glsl_required_version_line", None)
        if required_version_line is not None:
            version_line = required_version_line
        self.glsl_raised_version_line = None
        self.glsl_emission_requirements = []
        self.glsl_descriptor_binding_macro_used = False
        self.glsl_spirv_location_macro_used = False
        self.current_glsl_version_line = version_line
        self.current_glsl_disabled_extensions = {
            match.group(1)
            for line in extra_lines
            if (
                match := re.match(
                    r"^\s*#extension\s+([A-Za-z_][A-Za-z0-9_]*)\s*:\s*disable\b",
                    line,
                )
            )
        }
        self.current_glsl_resource_binding_layouts_supported = (
            self.glsl_resource_binding_layouts_supported(version_line)
        )
        self.glsl_vulkan_resources.activate(self, version_line)
        exact_subgroup_width = self.glsl_stage_exact_subgroup_width(ast, target_stage)
        self.current_glsl_exact_subgroup_width = exact_subgroup_width
        code += f"{version_line}\n"
        code += source_license_comments(ast, "opengl")
        for line in self.glsl_stage_extension_lines(ast, target_stage):
            if line not in extra_lines:
                code += f"{line}\n"
        code += self.prepare_glsl_runtime_specialization_dependents(ast, version_line)
        if extra_lines:
            code += "\n".join(extra_lines) + "\n"
        if exact_subgroup_width is not None:
            code += (
                f"#define {self.GLSL_REQUIRED_SUBGROUP_WIDTH_MACRO} "
                f"{exact_subgroup_width}u\n"
            )
        if self.software_subgroup_width is not None:
            code += (
                f"#define {self.GLSL_SOFTWARE_SUBGROUP_WIDTH_MACRO} "
                f"{self.software_subgroup_width}u\n"
            )
        cooperative_matrix_support_insertion_index = len(code)
        code += self.generate_glsl_enum_constants(
            self.plain_enums + self.struct_payload_enums,
            qualifier="const",
        )
        code += self.generate_glsl_generic_enum_constants(
            self.generic_enum_struct_definitions,
            qualifier="const",
        )
        leading_constants, struct_dependent_constants = (
            partition_constants_by_struct_dependency(
                getattr(ast, "constants", []) or [], structs
            )
        )
        leading_constants, specialization_dependent_constants = (
            partition_constants_by_specialization_dependency(
                leading_constants,
                {
                    name
                    for node in self.glsl_specialization_constant_declarations
                    if (name := constant_source_name(node)) is not None
                },
            )
        )
        code += self.generate_constants(ast, leading_constants)
        code += self.generate_glsl_global_specialization_constants()
        code += self.generate_constants(ast, specialization_dependent_constants)
        code += self.generate_glsl_wave_helpers(ast, target_stage)

        all_global_vars = [
            node
            for node in list(getattr(ast, "global_variables", []) or [])
            if not self.is_elided_glsl_type_alias_global(node)
        ]
        for node in all_global_vars:
            input_attachment_reason = input_attachment_refusal_reason(node)
            if input_attachment_reason is not None:
                self.refuse_lowering(input_attachment_reason)
        specialization_global_ids = {
            id(node) for node in self.glsl_global_specialization_constants
        }
        global_vars = [
            node
            for node in all_global_vars
            if id(node) not in specialization_global_ids
        ]
        empty_struct_global_ids = self.collect_glsl_empty_struct_global_ids(global_vars)
        runtime_initialized_global_vars = [
            node for node in global_vars if id(node) not in empty_struct_global_ids
        ]
        stage_local_resource_vars = collect_stage_local_variables(
            ast, target_stage, self.is_stage_local_resource_variable
        )
        stage_local_interface_vars = self.deduplicate_stage_interface_declarations(
            collect_stage_local_variables(
                ast, target_stage, self.is_stage_local_interface_variable
            )
        )
        self.fragment_blend_support_layout_parts = (
            self.glsl_fragment_output_blend_support_layout_parts(
                global_vars + stage_local_interface_vars
            )
        )
        self.task_payload_shared_variables = (
            self.glsl_task_payload_shared_variable_infos(
                global_vars + stage_local_interface_vars
            )
        )
        self.glsl_interface_block_structs_by_name = {
            node.name: node
            for node in structs
            if isinstance(node, StructNode)
            and self.is_glsl_interface_block_struct(node)
        }
        stage_resource_params = self.collect_stage_entry_resource_parameters(
            ast,
            target_stage,
            global_vars + stage_local_resource_vars,
        )
        resource_declaration_nodes = self.deduplicate_resource_declaration_nodes(
            global_vars + stage_local_resource_vars + stage_resource_params
        )
        self.global_variable_types = self.collect_global_variable_types(
            self.glsl_global_specialization_constants
            + global_vars
            + stage_local_resource_vars
            + stage_local_interface_vars
            + stage_resource_params
        )
        for cbuffer in self.glsl_cbuffer_nodes(ast, target_stage):
            for member in getattr(cbuffer, "members", []) or []:
                member_name = getattr(member, "name", None)
                member_type = getattr(
                    member, "member_type", getattr(member, "vtype", None)
                )
                if not member_name or member_type is None:
                    continue
                type_name = self.type_name_string(member_type)
                if isinstance(member, ArrayNode):
                    element_type = getattr(member, "element_type", member_type)
                    size = getattr(member, "size", None)
                    type_name = f"{self.type_name_string(element_type)}[{size or ''}]"
                if type_name:
                    self.global_variable_types.setdefault(member_name, type_name)
        self.prepare_glsl_compile_time_globals(ast, global_vars)
        self.function_private_pointer_array_size_hints = (
            self.collect_private_pointer_array_size_hints(ast)
        )
        if self.glsl_private_pointer_robust_zero_read_variants:
            code += (
                "// CrossTL policy: statically out-of-bounds reads from "
                "const private-pointer views return zero.\n"
                "#define CROSSTL_PRIVATE_POINTER_OOB_READ_ZERO 1\n"
            )
        self.glsl_buffer_block_struct_names = (
            self.collect_glsl_buffer_block_struct_names(resource_declaration_nodes)
        )
        self.layout_bound_struct_uniform_names = (
            self.collect_layout_bound_struct_uniform_names(resource_declaration_nodes)
        )
        self.glsl_interface_value_struct_names = (
            self.collect_glsl_interface_value_struct_names(
                ast,
                structs,
                global_vars,
                functions,
            )
        )
        self.prepare_glsl_interface_block_names(ast, target_stage, structs)
        self.vertex_input_struct_names = self.stage_parameter_struct_names(
            ast, "vertex"
        )
        self.vertex_output_struct_names = self.stage_return_struct_names(ast, "vertex")
        self.default_clip_position_member_ids = set()
        self.default_clip_position_member_ids = (
            self.glsl_default_clip_position_member_ids()
        )
        # Stage inputs and outputs become globals in GLSL, where they share one namespace with variables,
        # functions and the members of unnamed blocks (GLSL 4.60 section 4.2.7). The whole program is consulted so
        # that stages translated one at a time name their varyings alike.
        self.stage_io_reserved_value_names = (
            self.collect_glsl_global_identifier_names(ast, None)
            | set(self.function_return_types)
            | glsl_entry_resource_parameter_names(self, ast)
        )
        self.fragment_input_struct_names = self.stage_parameter_struct_names(
            ast, "fragment"
        )
        self.fragment_output_struct_names = self.stage_return_struct_names(
            ast, "fragment"
        )
        self.vertex_input_member_names = self.struct_member_names(
            self.vertex_input_struct_names
        )
        self.vertex_input_member_names.update(
            self.stage_value_parameter_input_names(ast, "vertex")
        )
        self.combined_vertex_output_member_names = (
            self.vertex_output_declaration_names()
        )
        self.fragment_input_member_names = self.fragment_input_declaration_names(
            self.fragment_input_struct_names
        )
        self.fragment_input_member_names.update(
            self.stage_value_parameter_input_names(ast, "fragment")
        )
        self.fragment_output_member_name_maps = self.fragment_output_member_maps(ast)
        self.fragment_output_member_layout_maps = (
            self.fragment_output_member_layout_maps_for_outputs()
        )
        self.stage_entry_functions_by_stage = {
            stage_name: stage_functions[0]
            for stage_name in (
                "vertex",
                "fragment",
                "compute",
                "geometry",
                "tessellation_control",
                "tessellation_evaluation",
                "mesh",
                "task",
                "amplification",
                "object",
                "ray_generation",
                "ray_intersection",
                "ray_closest_hit",
                "ray_any_hit",
                "ray_miss",
                "ray_callable",
            )
            for stage_functions in [self.stage_functions(ast, stage_name)]
            if stage_functions
        }
        self.prepare_glsl_runtime_global_initializers(
            ast,
            runtime_initialized_global_vars,
            functions,
            target_stage,
        )
        emit_vertex_io = target_stage in {None, "vertex"}
        emit_fragment_io = target_stage in {None, "fragment"}
        emit_graphics_io = target_stage in {None, "vertex", "fragment"}
        # VSInput, VSOutput and PSInput name stage interfaces by convention only when the program has that stage.
        program_stage_names = {
            normalize_stage_name(stage_type)
            for stage_type in getattr(ast, "stages", {})
        } | {function_stage_name(func) for func in functions} - {None}
        legacy_vertex_names = not program_stage_names or "vertex" in program_stage_names
        self.glsl_program_has_mesh_stage = "mesh" in program_stage_names
        legacy_fragment_names = (
            not program_stage_names or "fragment" in program_stage_names
        )
        data_struct_nodes = []
        # Stage inputs and outputs of structure type name the data structures, so they follow them.
        stage_io_code = ""
        for node in structs:
            if isinstance(node, StructNode):
                if node.name in self.generic_enum_struct_definitions:
                    continue
                if node.name in self.generic_struct_definitions:
                    continue
                if self.is_glsl_interface_block_struct(node):
                    code += self.generate_glsl_interface_block_declaration(node)
                    continue
                if (
                    node.name in self.glsl_buffer_block_struct_names
                    and node.name not in self.glsl_interface_value_struct_names
                ):
                    continue
                if (
                    node.name in self.layout_bound_struct_uniform_names
                    and node.name not in self.glsl_interface_value_struct_names
                ):
                    continue
                elif (
                    node.name == "VSInput"
                    and legacy_vertex_names
                    or node.name in self.vertex_input_struct_names
                ):
                    if emit_vertex_io:
                        stage_io_code += self.wrap_stage_guard(
                            self.generate_stage_input_declarations(node),
                            "vertex",
                        )
                        if self.stage_io_struct_referenced_outside_stage_entries(
                            ast, node.name
                        ):
                            data_struct_nodes.append(node)
                    elif not emit_graphics_io:
                        data_struct_nodes.append(node)
                elif node.name == "VSOutput" and (
                    legacy_vertex_names or legacy_fragment_names
                ):
                    emitted_io = False
                    if node.name in self.vertex_output_struct_names and emit_vertex_io:
                        stage_io_code += self.wrap_stage_guard(
                            self.generate_vertex_output_declarations(node),
                            "vertex",
                        )
                        emitted_io = True
                    if (
                        node.name in self.fragment_input_struct_names
                        and emit_fragment_io
                    ):
                        stage_io_code += self.wrap_stage_guard(
                            self.generate_fragment_input_declarations(node),
                            "fragment",
                        )
                        emitted_io = True
                    if (
                        node.name in self.fragment_output_struct_names
                        and emit_fragment_io
                    ):
                        stage_io_code += self.wrap_stage_guard(
                            self.generate_fragment_output_declarations(node),
                            "fragment",
                        )
                        emitted_io = True
                    if not emitted_io:
                        if emit_vertex_io and legacy_vertex_names:
                            stage_io_code += self.wrap_stage_guard(
                                self.generate_legacy_output_declarations(node),
                                "vertex",
                            )
                            if self.stage_io_struct_referenced_outside_stage_entries(
                                ast, node.name
                            ):
                                data_struct_nodes.append(node)
                        else:
                            data_struct_nodes.append(node)
                    elif self.should_emit_flattened_stage_io_struct(
                        ast, node.name, emitted_io, emit_graphics_io
                    ):
                        data_struct_nodes.append(node)
                elif node.name in self.vertex_output_struct_names:
                    emitted_io = False
                    if emit_vertex_io:
                        stage_io_code += self.wrap_stage_guard(
                            self.generate_vertex_output_declarations(node),
                            "vertex",
                        )
                        emitted_io = True
                    if (
                        node.name in self.fragment_input_struct_names
                        and emit_fragment_io
                    ):
                        stage_io_code += self.wrap_stage_guard(
                            self.generate_fragment_input_declarations(node),
                            "fragment",
                        )
                        emitted_io = True
                    if (
                        node.name in self.fragment_output_struct_names
                        and emit_fragment_io
                    ):
                        stage_io_code += self.wrap_stage_guard(
                            self.generate_fragment_output_declarations(node),
                            "fragment",
                        )
                        emitted_io = True
                    if self.should_emit_flattened_stage_io_struct(
                        ast, node.name, emitted_io, emit_graphics_io
                    ):
                        data_struct_nodes.append(node)
                elif node.name == "PSInput" and legacy_fragment_names:
                    if emit_fragment_io:
                        stage_io_code += self.wrap_stage_guard(
                            self.generate_fragment_input_declarations(node),
                            "fragment",
                        )
                    elif not emit_graphics_io:
                        data_struct_nodes.append(node)
                elif node.name in self.fragment_input_struct_names:
                    if emit_fragment_io:
                        stage_io_code += self.wrap_stage_guard(
                            self.generate_fragment_input_declarations(node),
                            "fragment",
                        )
                        if node.name in self.fragment_output_struct_names:
                            stage_io_code += self.wrap_stage_guard(
                                self.generate_fragment_output_declarations(node),
                                "fragment",
                            )
                    if node.name in self.fragment_output_struct_names:
                        data_struct_nodes.append(node)
                    elif not emit_graphics_io or (
                        emit_fragment_io
                        and self.stage_io_struct_referenced_outside_stage_entries(
                            ast, node.name
                        )
                    ):
                        data_struct_nodes.append(node)
                elif node.name in self.fragment_output_struct_names:
                    if emit_fragment_io:
                        stage_io_code += self.wrap_stage_guard(
                            self.generate_fragment_output_declarations(node),
                            "fragment",
                        )
                    if node.name != "PSOutput" or not emit_fragment_io:
                        data_struct_nodes.append(node)
                elif node.name == "PSOutput":
                    if emit_fragment_io:
                        stage_io_code += self.wrap_stage_guard(
                            self.generate_fragment_output_declarations(node),
                            "fragment",
                        )
                    else:
                        data_struct_nodes.append(node)
                else:
                    data_struct_nodes.append(node)

        code += glsl_workgroup_size_declaration_layout(self, ast, target_stage)
        code += glsl_ray_desc_declaration(self, ast, data_struct_nodes)
        code += self.generate_ordered_data_struct_declarations(data_struct_nodes)
        code += self.generate_constants(ast, struct_dependent_constants)
        code += stage_io_code
        code += invariant_builtin_redeclarations(self, target_stage)
        code += self.generate_glsl_enum_constructor_functions(self.struct_payload_enums)
        code += self.generate_glsl_generic_enum_constructor_functions(
            self.generic_enum_specializations,
        )

        code += self.generate_stage_parameter_input_declarations(ast, target_stage)
        code += self.generate_stage_parameter_output_declarations(ast, target_stage)

        resource_binding_cursors = {}
        used_resource_bindings = {}
        self.reserve_explicit_global_resource_bindings(
            resource_declaration_nodes, used_resource_bindings
        )
        for index, node in enumerate(resource_declaration_nodes):
            vtype, array_size, array_suffix, resource_count = (
                self.resource_declaration_shape(node)
            )

            if hasattr(node, "name"):
                source_var_name = node.name
            elif hasattr(node, "variable_name"):
                source_var_name = node.variable_name
            else:
                source_var_name = f"var{index}"
            var_name = self.glsl_global_declaration_identifier_name(
                node, source_var_name
            )

            if self.is_glsl_buffer_block_variable(node, vtype):
                self.record_glsl_buffer_block_variable(var_name, node, vtype)
                if self.is_shader_record_buffer_block(node):
                    if self.explicit_resource_binding_index(node) is not None:
                        raise OpenGLRefusal(
                            "GLSL shaderRecordEXT buffer blocks cannot declare "
                            "binding layout qualifiers"
                        )
                    declaration = self.glsl_buffer_block_declaration(
                        node, vtype, var_name, None, array_suffix
                    )
                    if declaration is not None:
                        code += declaration
                        continue

                binding_namespace = "buffer binding"
                resource_binding = self.opengl_resource_binding_for_declaration(
                    node,
                    used_resource_bindings,
                    resource_binding_cursors,
                    binding_namespace,
                    resource_count,
                )
                declaration = self.glsl_buffer_block_declaration(
                    node, vtype, var_name, resource_binding, array_suffix
                )
                if declaration is not None:
                    self.reserve_resource_binding_range(
                        used_resource_bindings,
                        "OpenGL",
                        binding_namespace,
                        resource_binding,
                        resource_count,
                        var_name,
                    )
                    self.record_glsl_resource_binding(
                        node,
                        source_var_name,
                        var_name,
                        "buffer-block",
                        binding_namespace,
                        resource_binding,
                        resource_count,
                    )
                    code += declaration
                    self.advance_resource_binding(
                        resource_binding_cursors,
                        binding_namespace,
                        resource_binding,
                        resource_count,
                    )
                    continue

            mapped_type = self.map_resource_type_with_format(vtype, node)
            if mapped_type == "sampler":
                self.explicit_resource_binding_index(node)
                self.sampler_variables.add(var_name)
                self.glsl_sampler_registers[var_name] = (
                    source_var_name,
                    self.glsl_source_register(node),
                )
                if self.should_preserve_sampler_resource(source_var_name):
                    binding_namespace = "sampler binding"
                    resource_binding = self.opengl_resource_binding_for_declaration(
                        node,
                        used_resource_bindings,
                        resource_binding_cursors,
                        binding_namespace,
                        resource_count,
                    )
                    self.reserve_resource_binding_range(
                        used_resource_bindings,
                        "OpenGL",
                        binding_namespace,
                        resource_binding,
                        resource_count,
                        var_name,
                    )
                    self.record_glsl_resource_binding(
                        node,
                        source_var_name,
                        var_name,
                        "sampler",
                        binding_namespace,
                        resource_binding,
                        resource_count,
                    )
                    declaration = format_c_style_array_declaration(
                        f"{mapped_type}{array_suffix}", var_name
                    )
                    layout = self.glsl_resource_layout_prefix(
                        binding=resource_binding, node=node
                    )
                    prefix = f"{layout} " if layout else ""
                    code += f"{prefix}uniform {declaration};\n"
                    self.advance_resource_binding(
                        resource_binding_cursors,
                        binding_namespace,
                        resource_binding,
                        resource_count,
                    )
                continue
            if self.is_structured_buffer_type(vtype):
                self.record_structured_buffer_access_metadata(var_name, vtype, node)
                if source_var_name != var_name:
                    self.record_structured_buffer_access_metadata(
                        source_var_name, vtype, node
                    )
                binding_namespace = "buffer binding"
                resource_binding = self.opengl_resource_binding_for_declaration(
                    node,
                    used_resource_bindings,
                    resource_binding_cursors,
                    binding_namespace,
                    resource_count,
                )
                self.reserve_resource_binding_range(
                    used_resource_bindings,
                    "OpenGL",
                    binding_namespace,
                    resource_binding,
                    resource_count,
                    var_name,
                )
                code += self.structured_buffer_block_declaration(
                    vtype, var_name, resource_binding, array_size, node
                )
                self.record_glsl_resource_binding(
                    node,
                    source_var_name,
                    var_name,
                    "structured-buffer",
                    binding_namespace,
                    resource_binding,
                    resource_count,
                )
                self.advance_resource_binding(
                    resource_binding_cursors,
                    binding_namespace,
                    resource_binding,
                    resource_count,
                )
                if self.structured_buffer_requires_counter(vtype):
                    counter_binding = self.next_available_resource_binding(
                        used_resource_bindings,
                        resource_binding_cursors,
                        binding_namespace,
                        resource_count,
                    )
                    counter_name = self.structured_buffer_counter_resource_name(
                        var_name
                    )
                    self.reserve_resource_binding_range(
                        used_resource_bindings,
                        "OpenGL",
                        binding_namespace,
                        counter_binding,
                        resource_count,
                        counter_name,
                    )
                    code += self.structured_buffer_counter_block_declaration(
                        var_name, counter_binding, array_size
                    )
                    self.glsl_resource_binding_records.append(
                        {
                            "name": counter_name,
                            "glslName": counter_name,
                            "kind": "structured-buffer-counter",
                            "namespace": "shader-storage-buffer",
                            "binding": int(counter_binding),
                            "count": (
                                max(resource_count, 1)
                                if isinstance(resource_count, int)
                                else 1
                            ),
                            "register": None,
                            "of": source_var_name,
                        }
                    )
                    self.advance_resource_binding(
                        resource_binding_cursors,
                        binding_namespace,
                        counter_binding,
                        resource_count,
                    )
                continue
            if self.is_constant_buffer_type(vtype):
                self.glsl_uniform_block_variable_types[var_name] = (
                    self.constant_buffer_element_type(vtype)
                )
                if source_var_name != var_name:
                    self.glsl_uniform_block_variable_types[source_var_name] = (
                        self.constant_buffer_element_type(vtype)
                    )
                binding_namespace = "uniform buffer binding"
                resource_binding = self.opengl_resource_binding_for_declaration(
                    node,
                    used_resource_bindings,
                    resource_binding_cursors,
                    binding_namespace,
                    resource_count,
                )
                self.reserve_resource_binding_range(
                    used_resource_bindings,
                    "OpenGL",
                    binding_namespace,
                    resource_binding,
                    resource_count,
                    var_name,
                )
                code += self.glsl_constant_buffer_block_declaration(
                    vtype,
                    var_name,
                    resource_binding,
                    array_suffix,
                    node=node,
                )
                self.record_glsl_resource_binding(
                    node,
                    source_var_name,
                    var_name,
                    "constant-buffer",
                    binding_namespace,
                    resource_binding,
                    resource_count,
                )
                self.advance_resource_binding(
                    resource_binding_cursors,
                    binding_namespace,
                    resource_binding,
                    resource_count,
                )
                continue
            if self.is_opaque_resource_type(mapped_type):
                self.texture_variable_types[var_name] = mapped_type
                if source_var_name != var_name:
                    self.texture_variable_types[source_var_name] = mapped_type
                fixed_array_size = self.glsl_resource_array_size_value(array_size)
                if fixed_array_size is not None:
                    self.resource_variable_array_sizes[var_name] = fixed_array_size
                    if source_var_name != var_name:
                        self.resource_variable_array_sizes[source_var_name] = (
                            fixed_array_size
                        )
                record_explicit_image_metadata(
                    var_name,
                    node,
                    self.attribute_value_to_string,
                    image_formats=self.image_variable_formats,
                    image_accesses=self.image_variable_accesses,
                )
                if source_var_name != var_name:
                    record_explicit_image_metadata(
                        source_var_name,
                        node,
                        self.attribute_value_to_string,
                        image_formats=self.image_variable_formats,
                        image_accesses=self.image_variable_accesses,
                    )
            storage_type = self.float_atomic_storage.declaration(
                f"{mapped_type}{array_suffix}", var_name
            )
            code += self.float_atomic_storage.take_definitions()
            declaration = format_c_style_array_declaration(storage_type, var_name)
            if id(node) in empty_struct_global_ids:
                code += self.generate_glsl_empty_struct_global_declaration(
                    node,
                    declaration,
                    f"{mapped_type}{array_suffix}",
                )
                continue
            if self.is_opaque_resource_type(mapped_type):
                binding_namespace = (
                    "image binding"
                    if self.is_storage_image_type(vtype)
                    else "texture binding"
                )
                resource_binding = self.opengl_resource_binding_for_declaration(
                    node,
                    used_resource_bindings,
                    resource_binding_cursors,
                    binding_namespace,
                    resource_count,
                )
                self.reserve_resource_binding_range(
                    used_resource_bindings,
                    "OpenGL",
                    binding_namespace,
                    resource_binding,
                    resource_count,
                    var_name,
                )
                layout = self.opaque_resource_layout(
                    mapped_type, resource_binding, node
                )
                memory_qualifiers = self.resource_memory_qualifiers(node)
                if (
                    var_name in self.glsl_write_only_unknown_format_images
                    or source_var_name in self.glsl_write_only_unknown_format_images
                ) and "writeonly" not in memory_qualifiers.split():
                    memory_qualifiers = " ".join(
                        part for part in (memory_qualifiers, "writeonly") if part
                    )
                precision_qualifiers = self.resource_precision_qualifiers(node)
                qualifier_parts = []
                if layout:
                    qualifier_parts.append(layout)
                if memory_qualifiers:
                    qualifier_parts.append(memory_qualifiers)
                qualifier_parts.append("uniform")
                if precision_qualifiers:
                    qualifier_parts.append(precision_qualifiers)
                code += self.glsl_vulkan_resources.declaration(
                    source_var_name,
                    var_name,
                    " ".join(qualifier_parts),
                    declaration,
                    mapped_type,
                    array_suffix,
                )
                self.record_glsl_resource_binding(
                    node,
                    source_var_name,
                    var_name,
                    ("image" if binding_namespace == "image binding" else "texture"),
                    binding_namespace,
                    resource_binding,
                    resource_count,
                )
                self.advance_resource_binding(
                    resource_binding_cursors,
                    binding_namespace,
                    resource_binding,
                    resource_count,
                )
            elif self.is_layout_bound_struct_uniform(node, vtype):
                self.glsl_uniform_block_variable_types[var_name] = str(
                    self.resource_base_type(vtype)
                )
                if source_var_name != var_name:
                    self.glsl_uniform_block_variable_types[source_var_name] = str(
                        self.resource_base_type(vtype)
                    )
                binding_namespace = "uniform buffer binding"
                resource_binding = self.opengl_resource_binding_for_declaration(
                    node,
                    used_resource_bindings,
                    resource_binding_cursors,
                    binding_namespace,
                    resource_count,
                )
                self.reserve_resource_binding_range(
                    used_resource_bindings,
                    "OpenGL",
                    binding_namespace,
                    resource_binding,
                    resource_count,
                    var_name,
                )
                code += self.glsl_struct_uniform_block_declaration(
                    vtype,
                    var_name,
                    resource_binding,
                    array_suffix,
                    node=node,
                )
                self.record_glsl_resource_binding(
                    node,
                    source_var_name,
                    var_name,
                    "constant-buffer",
                    binding_namespace,
                    resource_binding,
                    resource_count,
                )
                self.advance_resource_binding(
                    resource_binding_cursors,
                    binding_namespace,
                    resource_binding,
                    resource_count,
                )
            else:
                code += self.generate_global_variable_declaration(
                    node,
                    declaration,
                    vtype,
                    mapped_type_for_layout=f"{mapped_type}{array_suffix}",
                )

        interface_var_stages = {
            id(variable): normalize_stage_name(stage_type)
            for stage_type, stage in getattr(ast, "stages", {}).items()
            for variable in getattr(stage, "local_variables", []) or []
        }
        for node in stage_local_interface_vars:
            # A stage's own in/out variables exist only in that stage's section of a combined program, and
            # take their locations in that stage's interface.
            owner = interface_var_stages.get(id(node))
            previous_stage_io_stage = self.current_global_stage_io_stage
            if owner is not None:
                self.current_global_stage_io_stage = owner
            try:
                declaration = self.generate_stage_local_interface_variable_declaration(
                    node
                )
            finally:
                self.current_global_stage_io_stage = previous_stage_io_stage
            code += (
                declaration
                if owner is None
                else self.wrap_stage_guard(declaration, owner)
            )

        cbuffers = self.glsl_cbuffer_nodes(ast, target_stage)
        if cbuffers:
            code += "// Constant Buffers\n"
            code += self.generate_cbuffers(ast, target_stage)

        self.collect_glsl_function_local_shared_declarations(ast, target_stage)
        self.glsl_global_workgroup_pointer_aliases = (
            self.collect_glsl_global_shared_bindings(global_vars)
        )
        self.glsl_global_storage_pointer_aliases = (
            self.collect_glsl_storage_pointer_resource_bindings(
                global_vars + stage_local_resource_vars
            )
        )
        for node, alias in self.glsl_hoisted_shared_declarations:
            code += self.generate_glsl_hoisted_shared_declaration(node, alias)

        combined_stage_entry_names = self.combined_stage_entry_names(ast, target_stage)
        previous_enforce_concrete_glsl_types = self.enforce_concrete_glsl_types
        self.enforce_concrete_glsl_types = False
        try:
            self.prepare_glsl_resource_function_specializations(ast, target_stage)
        finally:
            self.enforce_concrete_glsl_types = previous_enforce_concrete_glsl_types
        code += self.glsl_vulkan_resources.sampler_declarations(self)
        generated_helper_insertion_index = len(code)
        functions = getattr(ast, "functions", [])
        helper_stage_macros = glsl_helper_stage_macros(self, ast)
        deferred_top_level_helpers_by_stage = (
            self.collect_stage_deferred_top_level_helpers(ast, functions, target_stage)
        )
        deferred_top_level_helper_ids = {
            id(func)
            for helpers in deferred_top_level_helpers_by_stage.values()
            for func in helpers
        }
        called_function_names = self.collect_called_function_names(ast)
        top_level_prototype_sources = []
        for func in functions:
            if self.should_elide_glsl_function(func):
                continue
            if id(func) in deferred_top_level_helper_ids:
                continue
            if self.should_skip_unreachable_lowered_member_helper(func):
                continue
            if self.should_skip_unresolved_top_level_helper(
                func,
                called_function_names,
            ):
                continue
            qualifier_name = function_stage_name(func)
            if qualifier_name or not should_emit_qualified_function(
                target_stage, qualifier_name
            ):
                continue
            if getattr(func, "name", None) not in called_function_names:
                continue
            if self.glsl_has_resource_function_specializations(
                getattr(func, "name", None)
            ) and self.glsl_function_has_structured_buffer_parameter(func):
                continue
            top_level_prototype_sources.append(func)
        code += self.generate_glsl_resource_function_declarations()
        code += self.generate_glsl_function_declarations(
            self.glsl_concrete_helper_functions(top_level_prototype_sources)
        )

        for func in functions:
            if self.should_elide_glsl_function(func):
                continue
            if id(func) in deferred_top_level_helper_ids:
                continue
            if self.should_skip_unreachable_lowered_member_helper(func):
                continue
            if self.should_skip_unresolved_top_level_helper(
                func,
                called_function_names,
            ):
                continue
            qualifier_name = function_stage_name(func)

            if not should_emit_qualified_function(target_stage, qualifier_name):
                continue
            if getattr(func, "body", []) is None:
                continue

            resource_specializations = self.glsl_resource_function_emission_list(
                getattr(func, "name", None)
            )
            helper_stage_context = (
                target_stage
                if target_stage is not None and not qualifier_name
                else None
            )
            if resource_specializations and qualifier_name not in {
                "vertex",
                "fragment",
                "compute",
                "geometry",
                "tessellation_control",
                "tessellation_evaluation",
                "mesh",
                "task",
                "amplification",
                "object",
                "ray_generation",
                "ray_intersection",
                "ray_closest_hit",
                "ray_any_hit",
                "ray_miss",
                "ray_callable",
            }:
                specialized_code = "".join(
                    self.generate_function(
                        specialized_func,
                        stage_context=helper_stage_context,
                    )
                    for specialized_func in resource_specializations
                )
                code += wrap_glsl_helper_stage_guard(
                    self, specialized_code, func, helper_stage_macros
                )
            if self.glsl_has_resource_function_specializations(
                getattr(func, "name", None)
            ) and self.glsl_function_has_structured_buffer_parameter(func):
                # GLSL cannot pass storage blocks; every call uses a
                # specialization bound to its global buffer.
                continue

            if generic_function_parameters(func):
                mark_generic_function_emitted(self, func)
                specialized_code = ""
                for specialized_func in generic_function_emission_list(self, func):
                    for (
                        resource_specialized_func
                    ) in self.glsl_resource_function_emission_list(
                        getattr(specialized_func, "name", None)
                    ):
                        specialized_code += self.generate_function(
                            resource_specialized_func,
                            stage_context=helper_stage_context,
                        )
                    if not self.glsl_function_has_target_pointer_parameter(
                        specialized_func
                    ):
                        specialized_code += self.generate_function(
                            specialized_func,
                            stage_context=helper_stage_context,
                        )
                if qualifier_name in self.GLSL_STAGE_GUARD_MACROS:
                    code += self.wrap_stage_guard(specialized_code, qualifier_name)
                else:
                    code += wrap_glsl_helper_stage_guard(
                        self, specialized_code, func, helper_stage_macros
                    )
                continue

            if (
                self.glsl_function_has_target_pointer_parameter(func)
                and not qualifier_name
            ):
                continue

            if qualifier_name == "vertex":
                stage_code = "// Vertex Shader\n"
                stage_code += self.generate_function(
                    func,
                    shader_type="vertex",
                    entry_name=combined_stage_entry_names.get(id(func)),
                )
                code += self.wrap_stage_guard(stage_code, "vertex")
            elif qualifier_name == "fragment":
                stage_code = "// Fragment Shader\n"
                stage_code += self.generate_function(
                    func,
                    shader_type="fragment",
                    entry_name=combined_stage_entry_names.get(id(func)),
                )
                code += self.wrap_stage_guard(stage_code, "fragment")
            elif qualifier_name == "compute":
                stage_code = "// Compute Shader\n"
                stage_code += self.generate_function(
                    func,
                    shader_type="compute",
                    execution_config=function_entry_execution_config(func),
                    entry_name=combined_stage_entry_names.get(id(func)),
                )
                code += self.wrap_stage_guard(stage_code, "compute")
            else:
                self.refuse_stageless_glsl_entry(func)
                function_code = self.generate_function(
                    func,
                    stage_context=helper_stage_context,
                )
                if getattr(func, "name", None) in self.glsl_generic_helper_refusals:
                    self.glsl_generic_helper_code[func.name] = function_code
                code += wrap_glsl_helper_stage_guard(
                    self, function_code, func, helper_stage_macros
                )

        if hasattr(ast, "stages") and ast.stages:
            for stage_type, stage in ast.stages.items():
                stage_name = normalize_stage_name(stage_type)
                if not stage_matches(target_stage, stage_name):
                    continue

                stage_code = ""
                stage_local_functions = list(
                    getattr(stage, "local_functions", []) or []
                )
                deferred_top_level_helpers = deferred_top_level_helpers_by_stage.get(
                    stage_name, []
                )
                ordered_stage_helpers = order_functions_by_dependencies(
                    stage_local_functions + deferred_top_level_helpers,
                    self.walk_ast,
                    self.function_call_name,
                    FunctionCallNode,
                )
                stage_code += self.generate_glsl_function_declarations(
                    self.glsl_concrete_helper_functions(
                        [
                            func
                            for func in ordered_stage_helpers
                            if not self.should_elide_glsl_function(func)
                            and getattr(func, "name", None) in called_function_names
                        ]
                    ),
                    stage_context=stage_name,
                )
                for func in ordered_stage_helpers:
                    if self.should_elide_glsl_function(func):
                        continue
                    if self.should_skip_unreachable_lowered_member_helper(func):
                        continue
                    if getattr(func, "body", []) is None:
                        continue
                    resource_specializations = (
                        self.glsl_resource_function_emission_list(
                            getattr(func, "name", None)
                        )
                    )
                    if resource_specializations:
                        for specialized_func in resource_specializations:
                            stage_code += self.generate_function(
                                specialized_func,
                                stage_context=stage_name,
                            )

                    if generic_function_parameters(func):
                        mark_generic_function_emitted(self, func)
                        for specialized_func in generic_function_emission_list(
                            self,
                            func,
                        ):
                            for (
                                resource_specialized_func
                            ) in self.glsl_resource_function_emission_list(
                                getattr(specialized_func, "name", None)
                            ):
                                stage_code += self.generate_function(
                                    resource_specialized_func,
                                    stage_context=stage_name,
                                )
                            if not self.glsl_function_has_target_pointer_parameter(
                                specialized_func
                            ):
                                stage_code += self.generate_function(
                                    specialized_func,
                                    stage_context=stage_name,
                                )
                    elif self.glsl_function_has_target_pointer_parameter(func):
                        continue
                    else:
                        stage_code += self.generate_function(
                            func,
                            stage_context=stage_name,
                        )

                if hasattr(stage, "entry_point"):
                    stage_code += self.generate_function(
                        stage.entry_point,
                        shader_type=stage_name,
                        execution_config=getattr(stage, "execution_config", None),
                        stage_layout_qualifiers=getattr(
                            stage, "layout_qualifiers", None
                        ),
                        entry_name=combined_stage_entry_names.get(
                            id(stage.entry_point)
                        ),
                        stage_context=stage_name,
                    )

                if stage_code:
                    stage_code = f"// {stage_name.title()} Shader\n" + stage_code
                    code += self.wrap_stage_guard(stage_code, stage_name)

        generated_helpers = (
            self.generate_glsl_software_subgroup_support()
            + self.generate_glsl_trailing_zero_helpers()
            + self.generate_glsl_boolean_order_helpers()
            + self.generate_glsl_metal_math_helpers()
            + emit_glsl_call_helpers(self)
            + self.generate_glsl_half_helpers()
            + self.generate_glsl_half_bitcast_helpers()
            + self.generate_glsl_bfloat_helpers()
            + self.generate_glsl_bfloat_integer_helpers()
            + self.generate_glsl_float_selection_helpers()
            + self.generate_glsl_storage_structure_copy_helpers()
            + self.generate_glsl_integer_float_helpers()
            + self.generate_glsl_signed_remainder_helpers()
            + self.generate_glsl_floating_remainder_helpers()
            + self.generate_glsl_complex64_helpers()
            + self.generate_glsl_float_remainder_helpers()
            + self.generate_glsl_integer_dot_helpers()
            + self.generate_glsl_hlsl_structured_helpers()
            + self.generate_glsl_expected_compare_helpers()
            + self.float_atomic_storage.helper_definitions()
            + self.union_storage.helper_definitions()
        )
        if generated_helpers:
            code = (
                code[:generated_helper_insertion_index]
                + generated_helpers
                + code[generated_helper_insertion_index:]
            )
        cooperative_matrix_support = (
            self.generate_glsl_cooperative_matrix_software_support()
            + self.generate_glsl_emulated_matrix_support()
        )
        if cooperative_matrix_support:
            code = (
                code[:cooperative_matrix_support_insertion_index]
                + cooperative_matrix_support
                + code[cooperative_matrix_support_insertion_index:]
            )
        code = self.glsl_vulkan_resources.with_prelude(
            code,
            cooperative_matrix_support_insertion_index,
            self.glsl_descriptor_binding_macro_used,
            self.glsl_spirv_location_macro_used,
        )
        code = self.finish_generic_resource_helper_refusals(code)
        self.validate_glsl_function_call_graph()
        code = self.ensure_glsl_version_supports_generated_layouts(code)
        if getattr(self, "glsl_shadow_lod_extension_used", False):
            code = require_shadow_lod_extension(code)
        code = self.ensure_glsl_formatted_image_load_support(code)
        code = self.ensure_glsl_precise_support(code)
        code = require_builtin_extensions(
            code, getattr(self, "glsl_stage_builtin_extensions", ())
        )
        return self.ensure_glsl_int64_support(code)

    def annotate_glsl_control_flow_intervals(self, function):
        constants = self.initial_literal_int_constants(function)
        lexical = self.glsl_scalar_array_view_lexical_bindings(function)
        replay_depth = 0
        replay_budget = 4096
        replay_intervals = {}
        pointer_read_intervals = {}
        reached_statements = set()
        unreachable_tails = set()

        for node in self.walk_ast(getattr(function, "body", [])):
            if isinstance(node, ArrayAccessNode):
                node._glsl_storage_pointer_read_interval = None

        def pointer_array_extent(expression):
            if not isinstance(expression, IdentifierNode):
                return None
            declaration = lexical["declarations"].get(
                lexical["reference_bindings"].get(id(expression))
            )
            if declaration is not None:
                contract = self.glsl_storage_pointer_array_contract(declaration["node"])
            else:
                contract = getattr(function, "_glsl_storage_pointer_aliases", {}).get(
                    expression.name
                )
                if (
                    contract is not None
                    and contract.get("kind") != "storage-pointer-array"
                ):
                    contract = None
            return contract.get("extent") if contract is not None else None

        def lexical_source_types(expression):
            source_types = {}
            for node in self.walk_ast(expression):
                if not isinstance(node, IdentifierNode):
                    continue
                binding = lexical["declarations"].get(
                    lexical["reference_bindings"].get(id(node))
                )
                if binding is not None:
                    if binding["source_type"] is None:
                        return None
                    source_types[node.name] = binding["source_type"]
            return source_types

        def checked_interval(expression, intervals, destination_type=None):
            source_types = lexical_source_types(expression)
            if source_types is None:
                return None
            with self.glsl_lexical_source_type_scope(source_types):
                interval = self.glsl_checked_private_pointer_interval(
                    expression, intervals, constants
                )
                if interval is None and destination_type is not None:
                    # Unsigned wrap is exact modulo the destination width.
                    interval = self.glsl_congruent_storage_interval(
                        expression, intervals, constants, destination_type
                    )
                return interval

        def checked_components(expression, vector_type, intervals):
            source_types = lexical_source_types(expression)
            if source_types is None:
                return None
            with self.glsl_lexical_source_type_scope(source_types):
                components = self.glsl_static_integer_vector_component_intervals(
                    expression, vector_type, intervals, constants
                )
                if components is None:
                    return None
                return tuple(
                    self.glsl_integer_storage_interval(
                        bounds, vector_type, component=True
                    )
                    for bounds in components
                )

        def expression_source_type(expression):
            source_types = lexical_source_types(expression)
            if source_types is None:
                return None
            with self.glsl_lexical_source_type_scope(source_types):
                return self.glsl_source_expression_type(expression)

        def aggregate_updates(expression, intervals, *, additive=False):
            source_types = lexical_source_types(expression)
            if source_types is None:
                return None
            has_aggregate = False
            for argument in expression.arguments or ():
                source_type = expression_source_type(argument)
                if not self.glsl_struct_member_types(
                    self.type_name_string(source_type)
                ):
                    continue
                has_aggregate = True
                if (
                    not isinstance(argument, IdentifierNode)
                    or stable_lexical_type(argument) is None
                    or argument.name in escaped_names
                ):
                    return None
            if not has_aggregate:
                return None
            with self.glsl_lexical_source_type_scope(source_types):
                return aggregate_call_intervals(
                    self, expression, intervals, constants, additive=additive
                )

        escaped_names = set()
        local_storage_pointer_names = set()
        for node in self.walk_ast(getattr(function, "body", [])):
            if (
                isinstance(node, VariableNode)
                and self.glsl_storage_pointer_element_type(node) is not None
            ):
                local_storage_pointer_names.add(node.name)
            if isinstance(node, UnaryOpNode) and self.map_operator(node.op) == "&":
                escaped_names.update(self.glsl_interval_dependency_names(node.operand))
            elif isinstance(node, VariableNode) and (
                self.type_name_string(getattr(node, "var_type", None)) or ""
            ).endswith("&"):
                escaped_names.update(
                    self.glsl_interval_dependency_names(node.initial_value)
                )

        def stable_lexical_type(expression):
            owner = (
                expression.object
                if isinstance(expression, MemberAccessNode)
                else expression
            )
            if not isinstance(owner, IdentifierNode):
                return None
            binding = lexical["declarations"].get(
                lexical["reference_bindings"].get(id(owner))
            )
            if binding is None or binding["source_type"] is None:
                return None
            qualifiers = {
                str(qualifier).lower()
                for qualifier in getattr(binding["node"], "qualifiers", ()) or ()
            }
            source_type = binding["source_type"]
            metadata = getattr(self, "glsl_metadata_entry_bounds", {}).get(
                id(function), {}
            )
            if (
                owner.name in metadata
                and any(
                    binding["node"] is parameter for parameter in function.parameters
                )
                and getattr(source_type, "referenced_type", None) is not None
            ):
                # These source-owned immutable scalar references have already
                # passed metadata validation; their loaded values are integers.
                source_type = source_type.referenced_type
            type_name = self.type_name_string(source_type) or ""
            if (
                "&" in type_name
                or "volatile" in type_name.replace("*", " ").split()
                or qualifiers.intersection({"volatile", "out", "inout"})
            ):
                return None
            if isinstance(expression, MemberAccessNode):
                field_type = self.glsl_struct_member_types(type_name).get(
                    str(expression.member)
                )
                if field_type is not None:
                    return field_type
                component = self.glsl_index_component_index(expression.member)
                info = self.glsl_value_type_info(source_type)
                if (
                    info is None
                    or component is None
                    or not 2 <= info["width"] <= 4
                    or component >= info["width"]
                ):
                    return None
                return expression_source_type(expression)
            return source_type

        def lexical_integer_domain(expression):
            source_type = stable_lexical_type(expression)
            if source_type is None or "*" in (self.type_name_string(source_type) or ""):
                return None
            owner = (
                expression.object
                if isinstance(expression, MemberAccessNode)
                else expression
            )
            with self.glsl_lexical_source_type_scope(
                {owner.name: stable_lexical_type(owner)}
            ):
                return self.glsl_integer_type_domain(expression, allow_wide=True)

        def unit_increment(operator):
            operator = self.map_operator(operator)
            if operator in {"++", "PRE_INCREMENT", "POST_INCREMENT"}:
                return 1
            if operator in {"--", "PRE_DECREMENT", "POST_DECREMENT"}:
                return -1
            return None

        def checked_integer_interval(expression, intervals):
            source_types = {}
            for child in self.walk_ast(expression):
                if not isinstance(child, IdentifierNode):
                    continue
                binding = lexical["declarations"].get(
                    lexical["reference_bindings"].get(id(child))
                )
                if binding is not None:
                    source_type = stable_lexical_type(child)
                    if source_type is None or "*" in (
                        self.type_name_string(source_type) or ""
                    ):
                        return None
                    source_types[child.name] = source_type
            with self.glsl_lexical_source_type_scope(source_types):
                bounds = self.glsl_checked_private_pointer_interval(
                    expression, intervals, constants, require_arithmetic_types=True
                )
                domain = self.glsl_integer_type_domain(expression, allow_wide=True)
            if bounds is None or domain is None:
                return None
            return (
                bounds
                if domain.minimum <= bounds[0] <= bounds[1] <= domain.maximum
                else None
            )

        def refine_guard(condition, intervals, expected):
            source_types = {}
            expressions = []
            cast_names = {
                id(node.function)
                for node in self.walk_ast(condition)
                if isinstance(node, FunctionCallNode)
                and self.glsl_integer_scalar_cast_call(node) is not None
            }
            for node in self.walk_ast(condition):
                if isinstance(node, MemberAccessNode):
                    expressions.append(node)
                    continue
                if not isinstance(node, IdentifierNode) or id(node) in cast_names:
                    continue
                binding = lexical["declarations"].get(
                    lexical["reference_bindings"].get(id(node))
                )
                if binding is None or binding["source_type"] is None:
                    return dict(intervals)
                source_types[node.name] = binding["source_type"]
                expressions.append(node)
            with self.glsl_lexical_source_type_scope(source_types):
                guarded = dict(intervals)
                seeded_domains = {}
                for node in expressions:
                    owner = node
                    while isinstance(owner, MemberAccessNode):
                        if self.glsl_declared_struct_member_interval_key(owner) is None:
                            break
                        owner = owner.object
                    if not isinstance(owner, IdentifierNode):
                        continue
                    if owner.name in escaped_names:
                        continue
                    if isinstance(node, MemberAccessNode):
                        binding = lexical["declarations"][
                            lexical["reference_bindings"][id(owner)]
                        ]
                        owner_type = self.type_name_string(binding["source_type"]) or ""
                        if (
                            "*" in owner_type
                            or "&" in owner_type
                            or set(
                                self.glsl_parameter_qualifiers(binding["node"])
                            ).intersection({"out", "inout"})
                        ):
                            continue
                        name = self.glsl_declared_struct_member_interval_key(node)
                    else:
                        name = node.name
                    domain = self.glsl_integer_type_domain(node, allow_wide=True)
                    if domain is not None:
                        current = guarded.get(name)
                        if current is None or not (
                            domain.minimum <= current[0] <= current[1] <= domain.maximum
                        ):
                            guarded[name] = (domain.minimum, domain.maximum)
                            seeded_domains[name] = guarded[name]
                refined = self.glsl_private_pointer_refined_condition_intervals(
                    condition,
                    guarded,
                    constants,
                    expected,
                    excluded_names=escaped_names,
                    check_source_types=True,
                )
                if refined is not None:
                    for name, domain in seeded_domains.items():
                        if refined.get(name) == domain:
                            refined.pop(name, None)
                return refined

        def merge_intervals(left, right):
            return {
                name: (
                    min(left[name][0], right[name][0]),
                    max(left[name][1], right[name][1]),
                )
                for name in left.keys() & right.keys()
            }

        def annotate_expression(expression, intervals):
            if (
                isinstance(expression, ArrayAccessNode)
                and pointer_array_extent(expression.array) is not None
            ):
                # A recurrence can read a different initialized slot on every
                # visit. Join the values read, not the intersection of slots.
                current = checked_interval(expression, intervals)
                if id(expression) in pointer_read_intervals:
                    previous = pointer_read_intervals[id(expression)]
                    current = (
                        (min(previous[0], current[0]), max(previous[1], current[1]))
                        if previous is not None and current is not None
                        else None
                    )
                pointer_read_intervals[id(expression)] = current
            if hasattr(expression, "__dict__"):
                bounds = dict(intervals)
                if replay_depth:
                    previous = replay_intervals.get(id(expression))
                    if previous is not None:
                        bounds = merge_intervals(previous, bounds)
                    replay_intervals[id(expression)] = bounds
                expression._glsl_control_flow_intervals = bounds

        def target_interval_keys(target):
            if isinstance(target, UnaryOpNode) and self.map_operator(target.op) == "&":
                return target_interval_keys(target.operand)
            field_key = self.glsl_declared_struct_member_interval_key(target)
            if field_key is not None:
                return {field_key}
            if isinstance(target, ArrayAccessNode):
                component_index = self.literal_int_value(target.index, constants)
                array_name = self.expression_name(target.array)
                if pointer_array_extent(target.array) is not None:
                    array_key = self.glsl_pointer_array_interval_key(
                        array_name, component_index
                    )
                    return {array_key} if array_key is not None else {array_name}
                if (
                    isinstance(component_index, int)
                    and not isinstance(component_index, bool)
                    and component_index in range(4)
                ):
                    key = self.glsl_index_component_interval_key(
                        target.array,
                        component_index,
                    )
                    return {key} if key is not None else set()
                return {
                    key
                    for component_index in range(4)
                    if (
                        key := self.glsl_index_component_interval_key(
                            target.array,
                            component_index,
                        )
                    )
                    is not None
                }
            if isinstance(target, MemberAccessNode):
                component_indices = tuple(
                    self.glsl_index_component_index(component)
                    for component in str(target.member)
                )
                if component_indices and all(
                    component_index is not None for component_index in component_indices
                ):
                    return {
                        key
                        for component_index in component_indices
                        if (
                            key := self.glsl_index_component_interval_key(
                                target.object,
                                component_index,
                            )
                        )
                        is not None
                    }
                object_name = expression_debug_name(target.object)
                return {f"{object_name}.{target.member}"} if object_name else set()
            name = self.expression_name(target)
            if not name:
                return set()
            keys = {name}
            keys.update(
                self.glsl_index_component_interval_key(name, component_index)
                for component_index in range(4)
            )
            keys.discard(None)
            return keys

        def clear_interval_targets(active_intervals, target_keys):
            if not target_keys:
                return
            target_keys = set(target_keys)
            for name, binding in getattr(
                function, "_glsl_storage_pointer_aliases", {}
            ).items():
                if name in target_keys and isinstance(binding.get("offset"), str):
                    target_keys.add(binding["offset"])
            for interval_key in list(active_intervals):
                if self.glsl_interval_key_sets_overlap(
                    {interval_key},
                    target_keys,
                ):
                    active_intervals.pop(interval_key, None)

        def call_argument_mutation_keys(
            function_name,
            arguments,
            argument_index,
            argument,
        ):
            candidates = list(
                self.glsl_function_overloads_by_name.get(function_name, ())
            )
            fallback = self.function_definitions.get(function_name)
            if fallback is not None and fallback not in candidates:
                candidates.append(fallback)
            argument_name = expression_debug_name(argument)
            result = set()
            for candidate in candidates:
                parameters = list(
                    getattr(
                        candidate,
                        "parameters",
                        getattr(candidate, "params", []),
                    )
                    or []
                )
                if argument_index >= len(parameters):
                    continue
                parameter = parameters[argument_index]
                if self.glsl_storage_pointer_array_contract(parameter) is not None:
                    if getattr(candidate, "body", None) is None or (
                        self.glsl_storage_pointer_array_parameter_is_mutated(
                            candidate, getattr(parameter, "name", None)
                        )
                    ):
                        result.update(target_interval_keys(argument))
                    continue
                if not set(self.glsl_parameter_qualifiers(parameter)).intersection(
                    {"out", "inout"}
                ):
                    continue
                parameter_name = getattr(parameter, "name", None)
                if not parameter_name or getattr(candidate, "body", None) is None:
                    return target_interval_keys(argument)
                mutations = self.glsl_function_mutated_interval_keys(candidate)
                relevant = {
                    key
                    for key in mutations
                    if key == parameter_name or key.startswith(f"{parameter_name}.")
                }
                if not argument_name and relevant:
                    return target_interval_keys(argument)
                for key in relevant:
                    result.add(f"{argument_name}{key[len(parameter_name) :]}")
            return result

        def mutation_interval_keys(value, declaration=None):
            mutated = set()

            def relevant(target):
                if declaration is None:
                    return True
                name = lexical["declarations"][declaration]["node"].name
                return any(
                    isinstance(child, IdentifierNode)
                    and child.name == name
                    and lexical["reference_bindings"].get(id(child))
                    in {None, declaration}
                    for child in self.walk_ast(target)
                )

            def collect(node):
                if node is None or isinstance(node, (str, int, float, bool)):
                    return
                if isinstance(node, dict):
                    for child in node.values():
                        collect(child)
                    return
                if isinstance(node, (list, tuple, set)):
                    for child in node:
                        collect(child)
                    return
                if isinstance(node, AssignmentNode):
                    target = getattr(
                        node,
                        "target",
                        getattr(node, "left", None),
                    )
                    if relevant(target):
                        mutated.update(target_interval_keys(target))
                elif isinstance(node, UnaryOpNode) and (
                    self.map_operator(node.op) in {"++", "--"}
                    or node.op in {
                        "PRE_INCREMENT",
                        "PRE_DECREMENT",
                        "POST_INCREMENT",
                        "POST_DECREMENT",
                    }
                ):
                    if relevant(node.operand):
                        mutated.update(target_interval_keys(node.operand))
                elif isinstance(node, FunctionCallNode):
                    function_name = self.function_call_name(node)
                    candidates = list(
                        self.glsl_function_overloads_by_name.get(function_name, ())
                    )
                    fallback = self.function_definitions.get(function_name)
                    if fallback is not None and fallback not in candidates:
                        candidates.append(fallback)
                    for argument_index, argument in enumerate(node.arguments or []):
                        if not relevant(argument):
                            continue
                        mutated.update(
                            call_argument_mutation_keys(
                                function_name,
                                list(node.arguments or []),
                                argument_index,
                                argument,
                            )
                        )
                if hasattr(node, "child_nodes"):
                    for child in node.child_nodes():
                        collect(child)
                    return
                if hasattr(node, "__dict__"):
                    for field, child in vars(node).items():
                        if field not in {
                            "annotations",
                            "parent",
                            "source_location",
                            "_glsl_control_flow_intervals",
                        }:
                            collect(child)

            collect(value)
            return mutated

        def visit_sequence(value, intervals):
            if isinstance(value, BlockNode):
                return visit(value, intervals)
            statements = getattr(value, "statements", value)
            if not isinstance(statements, (list, tuple)):
                statements = [statements]
            for index, statement in enumerate(statements):
                if visit(statement, intervals) is False:
                    unreachable_tails.update(
                        id(item) for item in statements[index + 1 :]
                    )
                    return False
            return True

        def refuted_branch(body):
            # The checked intervals refute this branch's guard, so its
            # statements are unreachable unless another replay reaches them.
            statements = getattr(body, "statements", body)
            if statements is None:
                return
            if not isinstance(statements, (list, tuple)):
                statements = [statements]
            unreachable_tails.update(id(item) for item in statements)

        bounded_zero_trip = set()
        possibly_entered_loops = set()

        def bounded_loop_bounds(node, loop_intervals, mutated):
            init, condition, update = node.init, node.condition, node.update
            if not isinstance(init, VariableNode):
                return None

            def counter_operand(expression):
                if (
                    isinstance(expression, IdentifierNode)
                    and expression.name == init.name
                ):
                    return expression
                if (
                    isinstance(expression, BinaryOpNode)
                    and self.map_operator(expression.op) == "+"
                ):
                    for counter, offset in (
                        (expression.left, expression.right),
                        (expression.right, expression.left),
                    ):
                        if (
                            isinstance(counter, IdentifierNode)
                            and counter.name == init.name
                            and init.name not in self.glsl_interval_dependency_names(
                                offset
                            )
                        ):
                            return counter
                return None

            conditions = [condition]
            for candidate in conditions:
                if (
                    isinstance(candidate, BinaryOpNode)
                    and self.map_operator(candidate.op) == "&&"
                ):
                    conditions.extend((candidate.left, candidate.right))
            condition = next(
                (
                    candidate
                    for candidate in conditions
                    if isinstance(candidate, BinaryOpNode)
                    and isinstance(candidate.left, IdentifierNode)
                    and candidate.left.name == init.name
                    and self.map_operator(candidate.op) in {"<", "<="}
                ),
                None,
            )
            if condition is None:
                condition = next(
                    (
                        candidate
                        for candidate in conditions
                        if isinstance(candidate, BinaryOpNode)
                        and self.map_operator(candidate.op) in {"<", "<="}
                        and counter_operand(candidate.left) is not None
                    ),
                    None,
                )
            counter = counter_operand(condition.left) if condition is not None else None
            increment = None
            if (
                isinstance(update, UnaryOpNode)
                and isinstance(update.operand, IdentifierNode)
                and update.operand.name == init.name
                and unit_increment(update.op) == 1
            ):
                increment = LiteralNode(1, PrimitiveType("int"))
            elif (
                isinstance(update, AssignmentNode)
                and isinstance(update.target, IdentifierNode)
                and update.target.name == init.name
                and self.map_operator(update.operator) == "+="
            ):
                increment = update.value
            if (
                not isinstance(init, VariableNode)
                or getattr(init, "qualifiers", None)
                or not isinstance(condition, BinaryOpNode)
                or counter is None
                or self.map_operator(condition.op) not in {"<", "<="}
                or increment is None
                or not self.glsl_side_effect_free_expression(increment)
                or mutation_interval_keys(increment)
                or init.name in escaped_names
                or not self.glsl_side_effect_free_expression(init.initial_value)
                or mutation_interval_keys(init.initial_value)
            ):
                return None
            domain = lexical_integer_domain(counter)
            start = loop_intervals.get(init.name)
            dependencies = self.glsl_interval_dependency_names(condition)
            dependencies.discard(init.name)
            dependencies.update(self.glsl_interval_dependency_names(increment))
            for child in self.walk_ast(node.condition):
                if isinstance(child, AssignmentNode) or (
                    isinstance(child, UnaryOpNode)
                    and unit_increment(child.op) is not None
                ):
                    return None
                if isinstance(child, FunctionCallNode) and (
                    self.glsl_private_pointer_expression_has_side_effects(child)
                ):
                    # A return summary proves effects only for the analyzed
                    # arguments. They must remain invariant on later calls.
                    if (
                        self.glsl_interval_key_sets_overlap(
                            self.glsl_interval_dependency_names(child),
                            mutated | escaped_names | {init.name},
                        )
                        or checked_integer_interval(child, loop_intervals) is None
                    ):
                        return None
            if (
                domain is None
                or start is None
                or not 0 <= start[0] <= start[1] <= domain.maximum
                or checked_integer_interval(init.initial_value, loop_intervals) != start
                or self.glsl_interval_key_sets_overlap(
                    dependencies, mutated | escaped_names | {init.name}
                )
                or init.name in mutation_interval_keys(node.body, id(init))
                or init.name in self.glsl_interval_dependency_names(condition.right)
                or mutation_interval_keys(node.condition)
            ):
                return None
            declarations = {
                child.name
                for child in self.walk_ast(node.body)
                if isinstance(child, (VariableNode, ArrayNode))
            }
            if declarations & dependencies:
                return None
            bound = checked_integer_interval(condition.right, loop_intervals)
            if bound is None or bound[0] < 0:
                return None
            initial_left = checked_integer_interval(condition.left, loop_intervals)
            if initial_left is None:
                return None
            inclusive = self.map_operator(condition.op) == "<="
            step = checked_integer_interval(increment, loop_intervals)
            if step is None or step[0] <= 0:
                return None
            distance = max(0, bound[1] - initial_left[0] + int(inclusive))
            count = (distance + step[0] - 1) // step[0]
            terminal = start[1] + count * step[1]
            if terminal > domain.maximum:
                return None
            counter_bounds = (start[0], start[1] + max(0, count - 1) * step[1])
            if count:
                update_bounds = checked_integer_interval(
                    BinaryOpNode(counter, "+", increment),
                    {**loop_intervals, init.name: counter_bounds},
                )
                if (
                    update_bounds is None
                    or not domain.minimum
                    <= update_bounds[0]
                    <= update_bounds[1]
                    <= domain.maximum
                ):
                    return None
            if condition.left is not counter:
                for call in self.walk_ast(node.body):
                    if not isinstance(call, FunctionCallNode):
                        continue
                    name = self.function_call_name(call)
                    if (
                        not self.glsl_side_effect_free_expression(call)
                        and name not in self.function_definitions
                        and name not in self.glsl_function_overloads_by_name
                        and not (
                            name in {"buffer_load", "buffer_store"}
                            and name not in self.function_return_types
                        )
                    ):
                        return None
                # Include the terminal condition evaluation. Source additions and
                # usual arithmetic conversions must preserve these integer values.
                frame = {**loop_intervals, init.name: (start[0], terminal)}
                source_types = {}
                for child in self.walk_ast(condition):
                    if isinstance(child, IdentifierNode):
                        source_type = stable_lexical_type(child)
                        if source_type is not None:
                            source_types[child.name] = source_type
                with self.glsl_lexical_source_type_scope(source_types):
                    for expression in self.walk_ast(condition):
                        if not isinstance(expression, BinaryOpNode):
                            continue
                        plan = self.glsl_arithmetic_conversion_plan(
                            self.glsl_source_expression_type(expression.left),
                            self.glsl_source_expression_type(expression.right),
                            self.map_operator(expression.op),
                        )
                        if plan is None:
                            return None
                        for operand, target in (
                            (expression.left, plan.left_target_type),
                            (expression.right, plan.right_target_type),
                        ):
                            bounds = checked_integer_interval(operand, frame)
                            info = target_arithmetic_type(target)
                            if (
                                bounds is None
                                or info is None
                                or info.lanes != 1
                                or info.kind not in {
                                    ArithmeticScalarKind.SIGNED_INTEGER,
                                    ArithmeticScalarKind.UNSIGNED_INTEGER,
                                }
                            ):
                                return None
                            signed = info.kind == ArithmeticScalarKind.SIGNED_INTEGER
                            minimum = -(1 << (info.bits - 1)) if signed else 0
                            maximum = (1 << (info.bits - int(signed))) - 1
                            if not minimum <= bounds[0] <= bounds[1] <= maximum:
                                return None
            return count, counter_bounds

        def additive_loop_intervals(node, incoming, mutated, count):
            """Bound one unconditional, invariant additive update per iteration."""
            statements = list(getattr(node.body, "statements", node.body) or [])
            pointer_names = (
                set(
                    getattr(self, "glsl_entry_pointer_parameters", {}).get(
                        id(function), ()
                    )
                )
                | set(getattr(function, "_glsl_storage_pointer_aliases", {}))
                | local_storage_pointer_names
            )
            declared = {
                child.name
                for child in self.walk_ast(node.body)
                if isinstance(child, (VariableNode, ArrayNode))
            }
            entries, exits = {}, {}
            for index, statement in enumerate(statements):
                update = (
                    statement.expression
                    if isinstance(statement, ExpressionStatementNode)
                    else statement
                )
                call_step = None
                if isinstance(update, FunctionCallNode):
                    deltas = aggregate_updates(update, incoming, additive=True)
                    if not deltas or len(deltas) != 1:
                        continue
                    name, call_step = next(iter(deltas.items()))
                    owner_name, _, member = name.rpartition(".")
                    owners = [
                        argument
                        for argument in update.arguments
                        if isinstance(argument, IdentifierNode)
                        and argument.name == owner_name
                    ]
                    if len(owners) != 1:
                        continue
                    target = MemberAccessNode(owners[0], member)
                    assigned = [
                        argument
                        for argument in update.arguments
                        if argument is not owners[0]
                    ]
                    operator = "+="
                elif isinstance(update, AssignmentNode):
                    target = getattr(update, "target", getattr(update, "left", None))
                    assigned = getattr(update, "value", getattr(update, "right", None))
                    operator = self.map_operator(update.operator)
                else:
                    continue
                if not isinstance(
                    target, (IdentifierNode, MemberAccessNode)
                ) or operator not in {
                    "+=",
                    "-=",
                }:
                    continue
                if isinstance(target, MemberAccessNode):
                    if stable_lexical_type(target) is None:
                        continue
                    with self.glsl_lexical_source_type_scope(
                        {target.object.name: stable_lexical_type(target.object)}
                    ):
                        name = self.glsl_declared_struct_member_interval_key(
                            target
                        ) or self.glsl_index_component_interval_key(
                            target.object,
                            self.glsl_index_component_index(target.member),
                        )
                else:
                    name = target.name
                initial = incoming.get(name)
                dependencies = self.glsl_interval_dependency_names(assigned)
                other_statements = statements[:index] + statements[index + 1 :]
                if (
                    initial is None
                    or self.glsl_interval_key_sets_overlap(
                        {name}, escaped_names | declared
                    )
                    or self.glsl_interval_key_sets_overlap(
                        dependencies, mutated | escaped_names | declared
                    )
                    or self.glsl_interval_key_sets_overlap(
                        {name}, mutation_interval_keys(other_statements)
                    )
                    or any(
                        self.glsl_private_pointer_expression_has_side_effects(item)
                        for item in (assigned if call_step is not None else [assigned])
                    )
                ):
                    continue
                domain = lexical_integer_domain(target)
                source_type = stable_lexical_type(target)
                if (
                    domain is None
                    and name in pointer_names
                    and source_type is not None
                    and (
                        "*" in self.type_name_string(source_type)
                        or self.is_structured_buffer_type(source_type)
                    )
                ):
                    domain = IntegerRange(0, (1 << 31) - 1)
                step = (
                    call_step
                    if call_step is not None
                    else checked_integer_interval(assigned, incoming)
                )
                if domain is None or step is None:
                    continue
                if operator == "-=":
                    step = (-step[1], -step[0])
                exit_interval = (
                    initial[0] + min(0, count * step[0]),
                    initial[1] + max(0, count * step[1]),
                )
                if (
                    not domain.minimum
                    <= exit_interval[0]
                    <= exit_interval[1]
                    <= domain.maximum
                ):
                    continue
                entry_interval = (
                    initial[0] + min(0, (count - 1) * step[0]),
                    initial[1] + max(0, (count - 1) * step[1]),
                )
                if call_step is not None:
                    # Recheck promoted arithmetic over every possible entry,
                    # not just the zero-based probe used to obtain the step.
                    after = aggregate_updates(
                        update, {**incoming, name: entry_interval}
                    )
                    bounds = after.get(name) if after is not None else None
                    if (
                        bounds is None
                        or not exit_interval[0]
                        <= bounds[0]
                        <= bounds[1]
                        <= exit_interval[1]
                    ):
                        continue
                entries[name] = entry_interval
                exits[name] = exit_interval
            return entries, exits

        def visit(value, intervals):
            nonlocal replay_depth, replay_budget
            if value is None or isinstance(value, (str, int, float, bool)):
                return
            if isinstance(value, dict):
                for child in value.values():
                    visit(child, intervals)
                return
            if isinstance(value, (list, tuple, set)):
                return visit_sequence(list(value), intervals)

            reached_statements.add(id(value))
            annotate_expression(value, intervals)
            if isinstance(value, ReturnNode):
                visit(value.value, intervals)
                return False
            if isinstance(value, (VariableNode, ArrayNode)):
                initial_value = getattr(value, "initial_value", None)
                visit(initial_value, intervals)
                name = getattr(value, "name", None)
                if name:
                    variable_type = (
                        lexical["declarations"].get(id(value), {}).get("source_type")
                    )
                    clear_interval_targets(intervals, {name})
                    interval = self.glsl_integer_storage_interval(
                        checked_interval(initial_value, intervals, variable_type),
                        variable_type,
                    )
                    if (
                        interval is None
                        and self.glsl_storage_pointer_element_type(value) is not None
                    ):
                        asserted = self.glsl_index_asserted_range(
                            initial_value,
                            function_name=(
                                getattr(function, "_glsl_resource_source_name", None)
                                or getattr(function, "_generic_source_name", None)
                                or getattr(function, "name", None)
                            ),
                        )
                        if asserted is not None:
                            interval = (asserted.minimum, asserted.maximum)
                    if interval is None:
                        intervals.pop(name, None)
                    else:
                        intervals[name] = interval
                    component_intervals = checked_components(
                        initial_value,
                        variable_type,
                        intervals,
                    )
                    for component_index, component_interval in enumerate(
                        component_intervals or ()
                    ):
                        key = self.glsl_index_component_interval_key(
                            name,
                            component_index,
                        )
                        if key is not None:
                            intervals[key] = component_interval
                    member_intervals = (
                        self.glsl_simple_struct_constructor_member_intervals(
                            initial_value,
                            intervals,
                            constants,
                        )
                    )
                    for member, member_interval in member_intervals.items():
                        intervals[f"{name}.{member}"] = member_interval
                return
            if isinstance(value, AssignmentNode):
                target = getattr(value, "target", getattr(value, "left", None))
                assigned = getattr(value, "value", getattr(value, "right", None))
                visit(assigned, intervals)
                assigned_members = self.glsl_simple_struct_constructor_member_intervals(
                    assigned, intervals, constants
                )
                target_name = self.expression_name(target)
                if target_name and isinstance(
                    target, (str, IdentifierNode, VariableNode)
                ):
                    target_type = expression_source_type(target)
                    assigned_interval = checked_interval(
                        assigned, intervals, target_type
                    )
                    assigned_components = (
                        checked_components(
                            assigned,
                            target_type,
                            intervals,
                        )
                        if isinstance(assigned, IdentifierNode)
                        and target_type is not None
                        and self.type_name_string(target_type)
                        == self.type_name_string(expression_source_type(assigned))
                        else None
                    )
                    operator = self.map_operator(
                        getattr(value, "operator", getattr(value, "op", "="))
                    )
                    current = intervals.get(target_name)
                    if operator == "=":
                        replacement = assigned_interval
                    elif operator in {"+=", "-="} and current and assigned_interval:
                        replacement = checked_interval(
                            BinaryOpNode(target, operator[0], assigned),
                            intervals,
                            target_type,
                        )
                    else:
                        replacement = None
                    replacement = self.glsl_integer_storage_interval(
                        replacement, target_type
                    )
                    clear_interval_targets(intervals, {target_name})
                    if replacement is not None:
                        intervals[target_name] = replacement
                    if operator == "=":
                        for component, bounds in enumerate(assigned_components or ()):
                            key = self.glsl_index_component_interval_key(
                                target_name, component
                            )
                            if key is not None:
                                intervals[key] = bounds
                        for member, interval in assigned_members.items():
                            intervals[f"{target_name}.{member}"] = interval
                elif isinstance(target, MemberAccessNode):
                    clear_interval_targets(
                        intervals,
                        target_interval_keys(target),
                    )
                    component_index = self.glsl_index_component_index(target.member)
                    key = self.glsl_index_component_interval_key(
                        target.object,
                        component_index,
                    )
                    field_key = self.glsl_declared_struct_member_interval_key(target)
                    if field_key is not None:
                        key = field_key
                        component_index = None
                    if key is None:
                        key = expression_debug_name(target)
                    if key is not None:
                        assigned_interval = checked_interval(assigned, intervals)
                        operator = self.map_operator(
                            getattr(value, "operator", getattr(value, "op", "="))
                        )
                        current = getattr(
                            value,
                            "_glsl_control_flow_intervals",
                            {},
                        ).get(key)
                        if operator == "=":
                            replacement = assigned_interval
                        elif operator in {"+=", "-="} and current and assigned_interval:
                            replacement = checked_interval(
                                BinaryOpNode(target, operator[0], assigned),
                                {**intervals, key: current},
                            )
                        else:
                            replacement = None
                        target_type = expression_source_type(
                            target.object if component_index is not None else target
                        )
                        target_info = self.glsl_value_type_info(target_type)
                        replacement = (
                            self.glsl_integer_storage_interval(
                                replacement,
                                target_type,
                                component=component_index is not None,
                            )
                            if target_info is not None
                            and target_info["family"] in {"int", "uint"}
                            else None
                        )
                        if replacement is None:
                            intervals.pop(key, None)
                        else:
                            intervals[key] = replacement
                        if operator == "=":
                            for member, interval in assigned_members.items():
                                intervals[f"{key}.{member}"] = interval
                elif isinstance(target, ArrayAccessNode):
                    index_bounds = checked_interval(target.index, intervals)
                    array_name = self.expression_name(target.array)
                    source_bounds = checked_interval(target, intervals)
                    operator = self.map_operator(value.operator)
                    assigned_bounds = checked_integer_interval(assigned, intervals)
                    replacement = None
                    extent = pointer_array_extent(target.array)
                    if operator == "=" and extent is not None:
                        replacement = checked_interval(assigned, intervals)
                    if source_bounds is not None and assigned_bounds is not None:
                        if operator == "+=":
                            replacement = (
                                source_bounds[0] + assigned_bounds[0],
                                source_bounds[1] + assigned_bounds[1],
                            )
                        elif operator == "-=":
                            replacement = (
                                source_bounds[0] - assigned_bounds[1],
                                source_bounds[1] - assigned_bounds[0],
                            )
                    key = (
                        self.glsl_pointer_array_interval_key(
                            array_name, index_bounds[0]
                        )
                        if index_bounds is not None
                        and index_bounds[0] == index_bounds[1]
                        and extent is not None
                        and 0 <= index_bounds[0] < extent
                        else None
                    )
                    known_element = key is not None and (
                        key in intervals or operator == "="
                    )
                    clear_interval_targets(
                        intervals,
                        {key} if key is not None else target_interval_keys(target),
                    )
                    if (
                        known_element
                        and replacement is not None
                        and (0 <= replacement[0] <= replacement[1] < (1 << 31))
                    ):
                        intervals[key] = replacement
                visit(target, intervals)
                return
            if isinstance(value, TernaryOpNode):
                visit(value.condition, intervals)
                true_intervals = dict(intervals)
                false_intervals = dict(intervals)
                visit(value.true_expr, true_intervals)
                visit(value.false_expr, false_intervals)
                intervals.clear()
                intervals.update(merge_intervals(true_intervals, false_intervals))
                return
            if isinstance(value, BinaryOpNode) and self.map_operator(value.op) in {
                "&&",
                "||",
            }:
                visit(value.left, intervals)
                left_intervals = dict(intervals)
                right_intervals = dict(left_intervals)
                visit(value.right, right_intervals)
                intervals.clear()
                intervals.update(merge_intervals(left_intervals, right_intervals))
                return
            if isinstance(value, FunctionCallNode):
                function_expr = getattr(value, "function", None)
                visit(function_expr, intervals)
                arguments = list(value.arguments or [])
                for argument in arguments:
                    visit(argument, intervals)
                argument_mutations = (
                    [mutation_interval_keys(argument) for argument in arguments]
                    if any(
                        isinstance(argument, UnaryOpNode)
                        and self.map_operator(argument.op) == "&"
                        for argument in arguments
                    )
                    else []
                )
                for argument_index, argument in enumerate(arguments):
                    if pointer_array_extent(argument) is not None:
                        # An array argument refers to the current slots, including
                        # effects from evaluating other arguments of this call.
                        annotate_expression(argument, intervals)
                    elif (
                        isinstance(argument, UnaryOpNode)
                        and self.map_operator(argument.op) == "&"
                    ):
                        other_mutations = set().union(
                            *(
                                mutations
                                for index, mutations in enumerate(argument_mutations)
                                if index != argument_index
                            )
                        )
                        argument._glsl_pointer_argument_mutated = (
                            self.glsl_interval_key_sets_overlap(
                                target_interval_keys(argument.operand), other_mutations
                            )
                        )
                updates = aggregate_updates(value, intervals)
                function_name = self.function_call_name(value)
                for argument_index, argument in enumerate(arguments):
                    clear_interval_targets(
                        intervals,
                        call_argument_mutation_keys(
                            function_name,
                            arguments,
                            argument_index,
                            argument,
                        ),
                    )
                if updates is not None:
                    intervals.update(updates)
                return
            if isinstance(value, UnaryOpNode) and self.map_operator(value.op) == "&":
                visit(value.operand, intervals)
                clear_interval_targets(
                    intervals,
                    target_interval_keys(value.operand),
                )
                return
            if isinstance(value, UnaryOpNode) and (
                self.map_operator(value.op) in {"++", "--"}
                or value.op in {
                    "PRE_INCREMENT",
                    "PRE_DECREMENT",
                    "POST_INCREMENT",
                    "POST_DECREMENT",
                }
            ):
                visit(value.operand, intervals)
                name = getattr(value.operand, "name", None)
                current = intervals.get(name)
                domain = lexical_integer_domain(value.operand)
                replacement = None
                if (
                    current is not None
                    and domain is not None
                    and name not in escaped_names
                    and domain.minimum <= current[0] <= current[1] <= domain.maximum
                ):
                    step = unit_increment(value.op)
                    candidate = (current[0] + step, current[1] + step)
                    if domain.minimum <= candidate[0] <= candidate[1] <= domain.maximum:
                        replacement = candidate
                clear_interval_targets(
                    intervals,
                    target_interval_keys(value.operand),
                )
                if replacement is not None:
                    intervals[name] = replacement
                return
            if isinstance(value, BlockNode):
                outer_keys = set(intervals)
                block_intervals = dict(intervals)
                shadowed_values = {}
                block_names = set()
                statements = list(getattr(value, "statements", []) or [])
                continuing = True
                for index, statement in enumerate(statements):
                    if isinstance(statement, (VariableNode, ArrayNode)):
                        name = getattr(statement, "name", None)
                        if name:
                            block_names.add(name)
                            for key in outer_keys:
                                if key == name or key.startswith(f"{name}."):
                                    shadowed_values[key] = block_intervals.get(key)
                    if visit(statement, block_intervals) is False:
                        unreachable_tails.update(
                            id(item) for item in statements[index + 1 :]
                        )
                        continuing = False
                        break
                for key in outer_keys:
                    value_after_block = shadowed_values.get(
                        key,
                        block_intervals.get(key),
                    )
                    if value_after_block is not None:
                        intervals[key] = value_after_block
                    else:
                        intervals.pop(key, None)
                for key, bounds in block_intervals.items():
                    owner, separator, _ = key.partition(".[")
                    if separator and owner not in block_names:
                        intervals[key] = bounds
                return continuing
            if isinstance(value, IfNode):
                condition = getattr(
                    value, "condition", getattr(value, "if_condition", None)
                )
                visit(condition, intervals)
                static_known, static_body = self.glsl_static_if_body(value)
                if static_known:
                    return visit_sequence(static_body, intervals)
                branch_intervals = []
                then_intervals = refine_guard(
                    condition,
                    intervals,
                    True,
                )
                if then_intervals is not None:
                    if visit_sequence(value.if_body, then_intervals) is not False:
                        branch_intervals.append(then_intervals)
                else:
                    refuted_branch(value.if_body)
                fallthrough_intervals = refine_guard(
                    condition,
                    intervals,
                    False,
                )
                for else_if_condition, else_if_body in zip(
                    getattr(value, "else_if_conditions", []) or [],
                    getattr(value, "else_if_bodies", []) or [],
                ):
                    if fallthrough_intervals is None:
                        refuted_branch(else_if_body)
                        continue
                    visit(else_if_condition, fallthrough_intervals)
                    candidate = refine_guard(
                        else_if_condition,
                        fallthrough_intervals,
                        True,
                    )
                    if candidate is not None:
                        if visit_sequence(else_if_body, candidate) is not False:
                            branch_intervals.append(candidate)
                    else:
                        refuted_branch(else_if_body)
                    fallthrough_intervals = refine_guard(
                        else_if_condition,
                        fallthrough_intervals,
                        False,
                    )
                if fallthrough_intervals is not None:
                    else_intervals = dict(fallthrough_intervals)
                    if (
                        visit_sequence(
                            getattr(value, "else_body", None), else_intervals
                        )
                        is not False
                    ):
                        branch_intervals.append(else_intervals)
                else:
                    refuted_branch(getattr(value, "else_body", None))
                if not branch_intervals:
                    return False
                merged = branch_intervals[0]
                for candidate in branch_intervals[1:]:
                    merged = merge_intervals(merged, candidate)
                intervals.clear()
                intervals.update(merged)
                return
            if isinstance(value, (SwitchNode, MatchNode)):
                visit(getattr(value, "expression", None), intervals)
                branches = list(
                    getattr(value, "cases", getattr(value, "arms", [])) or []
                )
                default_case = getattr(value, "default_case", None)
                mutated = mutation_interval_keys([branches, default_case])
                branch_base = dict(intervals)
                clear_interval_targets(branch_base, mutated)
                for branch in branches:
                    branch_intervals = dict(branch_base)
                    visit(branch, branch_intervals)
                if default_case is not None:
                    default_intervals = dict(branch_base)
                    visit(default_case, default_intervals)
                clear_interval_targets(intervals, mutated)
                return
            if isinstance(value, ForNode):
                counter_name = getattr(value.init, "name", None)
                counter_before = {
                    key: bounds
                    for key, bounds in intervals.items()
                    if key == counter_name or key.startswith(f"{counter_name}.")
                }

                def restore_counter(frame):
                    if not isinstance(value.init, VariableNode):
                        return
                    clear_interval_targets(frame, {counter_name})
                    frame.update(counter_before)

                loop_intervals = dict(intervals)
                mutated = mutation_interval_keys(
                    [
                        getattr(value, "init", None),
                        getattr(value, "condition", None),
                        getattr(value, "body", None),
                        getattr(value, "update", None),
                    ]
                )
                clear_interval_targets(loop_intervals, mutated)
                visit(getattr(value, "init", None), loop_intervals)
                if isinstance(getattr(value, "init", None), list):
                    # for (int a = 0, b = n; ...) declares no single induction counter for the bounds below, so a
                    # declared variable the loop writes has no proven range inside the loop.
                    clear_interval_targets(loop_intervals, mutated)
                bounded = bounded_loop_bounds(value, loop_intervals, mutated)
                count = bounded[0] if bounded is not None else None
                if count == 0:
                    bounded_zero_trip.add(id(value))
                    return
                possibly_entered_loops.add(id(value))
                pointer_array_loop = any(
                    isinstance(child, ArrayAccessNode)
                    and pointer_array_extent(child.array) is not None
                    for child in self.walk_ast(value.body)
                )
                aggregate_loop_names = {
                    child.name
                    for child in self.walk_ast(value.body)
                    if isinstance(child, IdentifierNode)
                    and child.name not in escaped_names
                    and self.glsl_struct_member_types(
                        self.type_name_string(stable_lexical_type(child))
                    )
                    and any(key.startswith(f"{child.name}.") for key in intervals)
                }
                annotate_expression(value.condition, loop_intervals)
                exact_values = (
                    self.glsl_storage_pointer_array_loop_iteration_values(
                        value,
                        counter_mutated=value.init.name in mutation_interval_keys(
                            value.body, id(value.init)
                        ),
                    )
                    if count is not None
                    and (pointer_array_loop or aggregate_loop_names)
                    else None
                )
                if (
                    exact_values is not None
                    and len(exact_values) == count
                    and count <= replay_budget
                ):
                    # Replay proven iterations on source state, joining facts at
                    # each repeated occurrence rather than keeping the last visit.
                    replay = dict(intervals)
                    visit(value.init, replay)
                    replay_budget -= count
                    replay_depth += 1
                    try:
                        for counter in exact_values:
                            replay[value.init.name] = (counter, counter)
                            visit(value.condition, replay)
                            visit_sequence(value.body, replay)
                            visit(value.update, replay)
                    finally:
                        replay_depth -= 1
                    restore_counter(replay)
                    for key in list(intervals):
                        if key in replay:
                            intervals[key] = replay[key]
                        else:
                            intervals.pop(key, None)
                    for key, bounds in replay.items():
                        if ".[" in key or key.partition(".")[0] in aggregate_loop_names:
                            intervals[key] = bounds
                    return
                entries, exits = (
                    additive_loop_intervals(value, intervals, mutated, count)
                    if count is not None
                    else ({}, {})
                )
                entries.update(
                    guarded_additive_loop_entries(
                        self,
                        function,
                        value,
                        intervals,
                        entries,
                        count,
                        constants=constants,
                        escaped=escaped_names,
                        mutated=mutated,
                        lexical=lexical,
                        checked_integer=checked_integer_interval,
                        checked_components=checked_components,
                        mutation_keys=mutation_interval_keys,
                    )
                )
                loop_intervals.update(entries)
                loop_name, loop_interval = self.glsl_private_pointer_loop_interval(
                    value, loop_intervals, constants
                )
                if loop_interval is None and count is not None:
                    loop_name = value.init.name
                    # An upper trip bound establishes entry values, not an exact
                    # iteration sequence for pointer rebasing or exit state.
                    loop_interval = bounded[1]
                if loop_name in mutation_interval_keys(
                    [
                        getattr(value, "condition", None),
                        getattr(value, "body", None),
                    ],
                    (
                        id(value.init)
                        if isinstance(value.init, VariableNode)
                        and value.init.name == loop_name
                        else None
                    ),
                ):
                    loop_interval = None
                if loop_name:
                    if loop_interval is None:
                        loop_intervals.pop(loop_name, None)
                    else:
                        loop_intervals[loop_name] = loop_interval
                visit(getattr(value, "condition", None), loop_intervals)
                guarded = refine_guard(value.condition, loop_intervals, True)
                if guarded is not None:
                    visit_sequence(getattr(value, "body", None), guarded)
                    visit(getattr(value, "update", None), guarded)
                clear_interval_targets(intervals, mutated)
                intervals.update(exits)
                restore_counter(intervals)
                return
            if isinstance(value, ForInNode):
                loop_intervals = dict(intervals)
                mutated = mutation_interval_keys(
                    [
                        getattr(value, "iterable", None),
                        getattr(value, "body", None),
                    ]
                )
                clear_interval_targets(loop_intervals, mutated)
                iterable = getattr(value, "iterable", None)
                pattern_interval = self.glsl_for_in_pattern_interval(
                    value,
                    loop_intervals,
                    constants,
                )

                pattern = getattr(value, "pattern", None)
                if pattern:
                    loop_intervals.pop(pattern, None)
                    self.glsl_clear_index_component_intervals(
                        loop_intervals,
                        pattern,
                    )
                    if pattern_interval is not None:
                        loop_intervals[pattern] = pattern_interval
                visit_sequence(getattr(value, "body", None), loop_intervals)
                clear_interval_targets(intervals, mutated)
                return
            if isinstance(value, (WhileNode, DoWhileNode, LoopNode)):
                loop_intervals = dict(intervals)
                mutated = mutation_interval_keys(
                    [
                        getattr(value, "condition", None),
                        getattr(value, "body", None),
                    ]
                )
                clear_interval_targets(loop_intervals, mutated)
                if isinstance(value, DoWhileNode):
                    visit_sequence(getattr(value, "body", None), loop_intervals)
                    visit(getattr(value, "condition", None), loop_intervals)
                elif isinstance(value, WhileNode):
                    visit(getattr(value, "condition", None), loop_intervals)
                    guarded = refine_guard(value.condition, loop_intervals, True)
                    if guarded is not None:
                        visit_sequence(getattr(value, "body", None), guarded)
                else:
                    visit_sequence(getattr(value, "body", None), loop_intervals)
                clear_interval_targets(intervals, mutated)
                return
            if hasattr(value, "child_nodes"):
                for child in value.child_nodes():
                    visit(child, intervals)
                return
            if hasattr(value, "__dict__"):
                for field, child in vars(value).items():
                    if field not in {
                        "annotations",
                        "parent",
                        "source_location",
                        "_glsl_control_flow_intervals",
                    }:
                        visit(child, intervals)

        visit_sequence(
            getattr(function, "body", []),
            self.glsl_workgroup_proof_initial_intervals(function),
        )
        for node in self.walk_ast(getattr(function, "body", [])):
            if isinstance(node, ArrayAccessNode):
                node._glsl_storage_pointer_read_interval = pointer_read_intervals.get(
                    id(node)
                )
            # Replayed nodes are unreachable only if no iteration reaches them.
            if id(node) in unreachable_tails or hasattr(
                node, "_glsl_checked_unreachable"
            ):
                node._glsl_checked_unreachable = (
                    id(node) in unreachable_tails and id(node) not in reached_statements
                )
            if isinstance(node, ForNode):
                node._glsl_checked_zero_trip = (
                    id(node) in bounded_zero_trip
                    and id(node) not in possibly_entered_loops
                ) or self.glsl_control_flow_zero_trip_for(node, lexical)
            elif isinstance(node, IfNode):
                node._glsl_checked_static_if = self.glsl_control_flow_static_if_body(
                    node, lexical
                )

    def glsl_workgroup_pointer_direct_access_intervals(
        self,
        function,
        parameter_intervals,
    ):
        parameters = list(
            getattr(function, "parameters", getattr(function, "params", [])) or []
        )
        element_types = {
            parameter.name: self.glsl_workgroup_pointer_element_type(parameter)
            for parameter in parameters
            if self.glsl_workgroup_pointer_element_type(parameter) is not None
            and getattr(parameter, "name", None)
        }
        if not element_types:
            return {}

        constants = self.initial_literal_int_constants(function)
        intervals = dict(parameter_intervals)
        for name, value in constants.items():
            literal = self.literal_int_value(value, constants)
            if literal is not None:
                intervals[name] = (literal, literal)
        aliases = {
            name: {
                "kind": "workgroup-pointer",
                "root": name,
                "offset": 0,
                "offset_interval": (0, 0),
                "element_type": element_type,
            }
            for name, element_type in element_types.items()
        }
        access_intervals = {}
        mutated_interval_names = self.glsl_function_mutated_interval_names(function)

        def retain_refinable_offset(binding):
            if binding is None:
                return None
            binding = dict(binding)
            dependencies = self.glsl_interval_dependency_names(binding.get("offset", 0))
            binding["_glsl_workgroup_proof_offset_refinable"] = not (
                dependencies & mutated_interval_names
            )
            return binding

        def merge_intervals(left, right):
            return {
                name: (
                    min(left[name][0], right[name][0]),
                    max(left[name][1], right[name][1]),
                )
                for name in left.keys() & right.keys()
            }

        def clear_interval_names(active_intervals, names):
            for name in names:
                active_intervals.pop(name, None)
                self.glsl_clear_index_component_intervals(active_intervals, name)

        def alias_offset_interval(binding, active_intervals):
            if binding is None or binding.get(
                "_glsl_workgroup_proof_offset_unprovable"
            ):
                return None
            return self.glsl_workgroup_pointer_offset_interval(
                binding,
                active_intervals,
            )

        def poison_alias(binding):
            poisoned = dict(binding)
            poisoned["offset"] = IdentifierNode("__crosstl_unprovable_workgroup_offset")
            poisoned["offset_interval"] = None
            poisoned["_glsl_workgroup_proof_offset_refinable"] = False
            poisoned["_glsl_workgroup_proof_offset_unprovable"] = True
            return poisoned

        def alias_state_key(binding, active_intervals):
            if binding is None:
                return None
            return (
                binding.get("kind"),
                binding.get("root"),
                binding.get("element_type"),
                alias_offset_interval(binding, active_intervals),
                bool(binding.get("_glsl_workgroup_proof_offset_unprovable")),
            )

        def changed_alias_names(
            before_aliases,
            before_intervals,
            after_aliases,
            after_intervals,
        ):
            return {
                name
                for name, binding in before_aliases.items()
                if alias_state_key(binding, before_intervals)
                != alias_state_key(after_aliases.get(name), after_intervals)
            }

        def merge_alias_states(target_aliases, branch_states):
            for name, original in list(target_aliases.items()):
                bindings = [aliases.get(name) for aliases, _ in branch_states]
                if any(binding is None for binding in bindings):
                    target_aliases[name] = poison_alias(original)
                    continue
                if all(binding == original for binding in bindings):
                    continue
                signatures = {
                    (
                        binding.get("kind"),
                        binding.get("root"),
                        binding.get("element_type"),
                    )
                    for binding in bindings
                }
                offset_intervals = [
                    alias_offset_interval(binding, state_intervals)
                    for binding, (_, state_intervals) in zip(
                        bindings,
                        branch_states,
                    )
                ]
                if len(signatures) != 1 or any(
                    interval is None for interval in offset_intervals
                ):
                    target_aliases[name] = poison_alias(original)
                    continue
                merged = dict(bindings[0])
                merged["offset_interval"] = (
                    min(interval[0] for interval in offset_intervals),
                    max(interval[1] for interval in offset_intervals),
                )
                merged["_glsl_workgroup_proof_offset_refinable"] = False
                merged.pop("_glsl_workgroup_proof_offset_unprovable", None)
                target_aliases[name] = merged

        def propagate_outer_state(
            active_aliases,
            active_intervals,
            original_aliases,
            original_intervals,
            local_aliases,
            local_intervals,
            *,
            shadow_names=(),
        ):
            shadow_names = set(shadow_names)
            for name, original in original_aliases.items():
                if name in shadow_names:
                    active_aliases[name] = dict(original)
                elif name in local_aliases:
                    active_aliases[name] = dict(local_aliases[name])
                else:
                    active_aliases[name] = poison_alias(original)
            for name in list(active_aliases):
                if name not in original_aliases:
                    active_aliases.pop(name, None)

            for key, original in original_intervals.items():
                shadowed = any(
                    key == name or key.startswith(f"{name}.") for name in shadow_names
                )
                if shadowed:
                    active_intervals[key] = original
                elif key in local_intervals:
                    active_intervals[key] = local_intervals[key]
                else:
                    active_intervals.pop(key, None)
            for key in list(active_intervals):
                if key not in original_intervals:
                    active_intervals.pop(key, None)

        def exact_classic_for_iteration_values(node):
            values = self.glsl_storage_pointer_array_loop_iteration_values(node)
            if values is None:
                return None
            condition = getattr(node, "condition", None)
            init = getattr(node, "init", None)
            if not isinstance(condition, BinaryOpNode) or init is None:
                return None
            loop_name = getattr(init, "name", None)
            if self.expression_name(condition.left) == loop_name:
                bound = condition.right
            elif self.expression_name(condition.right) == loop_name:
                bound = condition.left
            else:
                return None
            if self.literal_int_value(bound, constants) is None:
                return None
            return values

        def exact_for_in_iteration_values(node):
            iterable = getattr(node, "iterable", None)
            if isinstance(iterable, RangeNode):
                start = self.literal_int_value(iterable.start, constants)
                end = self.literal_int_value(iterable.end, constants)
                if start is None or end is None:
                    return None
                stop = end + (1 if getattr(iterable, "inclusive", False) else 0)
                values = tuple(range(start, stop))
            else:
                count = self.literal_int_value(iterable, constants)
                if count is None or count < 0:
                    return None
                values = tuple(range(count))
            return values if len(values) <= 4096 else None

        def record_access(pointer, index, active_aliases, active_intervals):
            binding = self.glsl_workgroup_pointer_binding(
                pointer,
                active_aliases,
                active_intervals,
            )
            if binding is None or binding.get("root") not in element_types:
                return
            root = binding["root"]
            base_interval = alias_offset_interval(
                binding,
                active_intervals,
            )
            if binding.get("_glsl_workgroup_proof_offset_refinable"):
                refined_interval = self.glsl_private_pointer_interval(
                    binding.get("offset", 0),
                    active_intervals,
                    constants,
                )
                if (
                    refined_interval is not None
                    and base_interval is not None
                    and refined_interval[0] >= base_interval[0]
                    and refined_interval[1] <= base_interval[1]
                ):
                    base_interval = refined_interval
            index_interval = self.glsl_private_pointer_interval(
                index,
                active_intervals,
                constants,
            )
            if base_interval is None or index_interval is None:
                access_intervals[root] = None
                return
            candidate = (
                base_interval[0] + index_interval[0],
                base_interval[1] + index_interval[1],
            )
            current = access_intervals.get(root)
            if current is None and root in access_intervals:
                return
            if current is None:
                access_intervals[root] = candidate
            else:
                access_intervals[root] = (
                    min(current[0], candidate[0]),
                    max(current[1], candidate[1]),
                )

        def visit_sequence(value, active_aliases, active_intervals):
            statements = getattr(value, "statements", value)
            if not isinstance(statements, (list, tuple)):
                statements = [statements]
            for statement in statements:
                visit(statement, active_aliases, active_intervals)

        def visit_scoped_sequence(value, active_aliases, active_intervals):
            statements = getattr(value, "statements", value)
            if not isinstance(statements, (list, tuple)):
                statements = [statements]
            outer_alias_names = set(active_aliases)
            outer_interval_keys = set(active_intervals)
            scoped_aliases = dict(active_aliases)
            scoped_intervals = dict(active_intervals)
            shadowed_aliases = {}
            shadowed_intervals = {}
            for statement in statements:
                if isinstance(statement, (VariableNode, ArrayNode)):
                    name = getattr(statement, "name", None)
                    if name:
                        if name in outer_alias_names and name not in shadowed_aliases:
                            shadowed_aliases[name] = dict(scoped_aliases[name])
                        for key in outer_interval_keys:
                            if (
                                key == name or key.startswith(f"{name}.")
                            ) and key not in shadowed_intervals:
                                shadowed_intervals[key] = scoped_intervals.get(key)
                visit(statement, scoped_aliases, scoped_intervals)
            for name in outer_alias_names:
                if name in shadowed_aliases:
                    active_aliases[name] = shadowed_aliases[name]
                elif name in scoped_aliases:
                    active_aliases[name] = scoped_aliases[name]
                else:
                    active_aliases[name] = poison_alias(active_aliases[name])
            for key in outer_interval_keys:
                if key in shadowed_intervals:
                    value_after_scope = shadowed_intervals[key]
                else:
                    value_after_scope = scoped_intervals.get(key)
                if value_after_scope is None:
                    active_intervals.pop(key, None)
                else:
                    active_intervals[key] = value_after_scope

        def visit(value, active_aliases, active_intervals):
            if value is None or isinstance(value, (str, int, float, bool)):
                return
            if isinstance(value, dict):
                for child in value.values():
                    visit(child, active_aliases, active_intervals)
                return
            if isinstance(value, (list, tuple, set)):
                visit_sequence(list(value), active_aliases, active_intervals)
                return
            if isinstance(value, (VariableNode, ArrayNode)):
                initial_value = getattr(value, "initial_value", None)
                visit(initial_value, active_aliases, active_intervals)
                name = getattr(value, "name", None)
                if not name:
                    return
                binding = self.glsl_workgroup_pointer_binding(
                    initial_value,
                    active_aliases,
                    active_intervals,
                )
                if self.glsl_workgroup_pointer_declaration(value, active_aliases):
                    if binding is None:
                        active_aliases.pop(name, None)
                    else:
                        active_aliases[name] = retain_refinable_offset(binding)
                else:
                    active_aliases.pop(name, None)
                interval = self.glsl_private_pointer_interval(
                    initial_value,
                    active_intervals,
                    constants,
                )
                if interval is None:
                    active_intervals.pop(name, None)
                else:
                    active_intervals[name] = interval
                return
            if isinstance(value, AssignmentNode):
                target = getattr(value, "target", getattr(value, "left", None))
                assigned = getattr(value, "value", getattr(value, "right", None))
                visit(assigned, active_aliases, active_intervals)
                target_name = self.expression_name(target)
                pointer_target = isinstance(target, (str, IdentifierNode, VariableNode))
                current = active_aliases.get(target_name) if pointer_target else None
                operator = self.map_operator(
                    getattr(value, "operator", getattr(value, "op", "="))
                )
                if current is not None and current.get("kind") == "workgroup-pointer":
                    if operator == "=":
                        replacement = self.glsl_workgroup_pointer_binding(
                            assigned,
                            active_aliases,
                            active_intervals,
                        )
                    elif operator in {"+=", "-="}:
                        delta = (
                            assigned if operator == "+=" else UnaryOpNode("-", assigned)
                        )
                        replacement = self.glsl_workgroup_pointer_binding(
                            BinaryOpNode(target, "+", delta),
                            active_aliases,
                            active_intervals,
                        )
                    else:
                        replacement = None
                    if replacement is None:
                        active_aliases[target_name] = poison_alias(current)
                    else:
                        active_aliases[target_name] = retain_refinable_offset(
                            replacement
                        )
                    return
                if target_name and isinstance(
                    target, (str, IdentifierNode, VariableNode)
                ):
                    assigned_interval = self.glsl_private_pointer_interval(
                        assigned,
                        active_intervals,
                        constants,
                    )
                    current_interval = active_intervals.get(target_name)
                    if operator == "=":
                        replacement_interval = assigned_interval
                    elif operator == "+=" and current_interval and assigned_interval:
                        replacement_interval = (
                            current_interval[0] + assigned_interval[0],
                            current_interval[1] + assigned_interval[1],
                        )
                    elif operator == "-=" and current_interval and assigned_interval:
                        replacement_interval = (
                            current_interval[0] - assigned_interval[1],
                            current_interval[1] - assigned_interval[0],
                        )
                    else:
                        replacement_interval = None
                    if replacement_interval is None:
                        active_intervals.pop(target_name, None)
                    else:
                        active_intervals[target_name] = replacement_interval
                visit(target, active_aliases, active_intervals)
                return
            if isinstance(value, ArrayAccessNode):
                record_access(
                    value.array,
                    value.index,
                    active_aliases,
                    active_intervals,
                )
                visit(value.index, active_aliases, active_intervals)
                return
            if isinstance(value, PointerAccessNode):
                record_access(
                    value.pointer_expr,
                    0,
                    active_aliases,
                    active_intervals,
                )
                return
            if isinstance(value, MemberAccessNode):
                record_access(
                    value.object,
                    0,
                    active_aliases,
                    active_intervals,
                )
            if isinstance(value, UnaryOpNode) and self.map_operator(value.op) == "&":
                for name in self.glsl_mutation_target_names(value.operand):
                    active_intervals.pop(name, None)
                    self.glsl_clear_index_component_intervals(
                        active_intervals,
                        name,
                    )
                return
            if isinstance(value, UnaryOpNode) and self.map_operator(value.op) == "*":
                reinterpretation = self.glsl_storage_struct_reinterpret_node(value)
                if reinterpretation is not None:
                    word_count = self.glsl_workgroup_byte_array_copy_word_count(
                        reinterpretation,
                        active_aliases,
                        active_intervals,
                    )
                    if word_count is not None:
                        record_access(
                            reinterpretation.expression,
                            0,
                            active_aliases,
                            active_intervals,
                        )
                        record_access(
                            reinterpretation.expression,
                            word_count - 1,
                            active_aliases,
                            active_intervals,
                        )
                        return
                record_access(
                    value.operand,
                    0,
                    active_aliases,
                    active_intervals,
                )
                return
            if isinstance(value, UnaryOpNode) and (
                self.map_operator(value.op) in {"++", "--"}
                or value.op in {
                    "PRE_INCREMENT",
                    "PRE_DECREMENT",
                    "POST_INCREMENT",
                    "POST_DECREMENT",
                }
            ):
                target_name = self.expression_name(value.operand)
                current_alias = active_aliases.get(target_name)
                if (
                    current_alias is not None
                    and current_alias.get("kind") == "workgroup-pointer"
                ):
                    delta = 1 if self.map_operator(value.op) == "++" else -1
                    replacement = self.glsl_workgroup_pointer_binding(
                        BinaryOpNode(value.operand, "+", delta),
                        active_aliases,
                        active_intervals,
                    )
                    active_aliases[target_name] = (
                        retain_refinable_offset(replacement)
                        if replacement is not None
                        else poison_alias(current_alias)
                    )
                    return
                current_interval = active_intervals.get(target_name)
                if current_interval is None:
                    active_intervals.pop(target_name, None)
                else:
                    delta = 1 if self.map_operator(value.op) == "++" else -1
                    active_intervals[target_name] = (
                        current_interval[0] + delta,
                        current_interval[1] + delta,
                    )
                self.glsl_clear_index_component_intervals(
                    active_intervals,
                    target_name,
                )
                return
            if isinstance(value, TernaryOpNode):
                visit(value.condition, active_aliases, active_intervals)
                base_aliases = dict(active_aliases)
                branch_states = []
                for selected, branch in (
                    (True, value.true_expr),
                    (False, value.false_expr),
                ):
                    branch_intervals = (
                        self.glsl_private_pointer_refined_condition_intervals(
                            value.condition,
                            active_intervals,
                            constants,
                            selected,
                            excluded_names=mutated_interval_names,
                        )
                    )
                    if branch_intervals is None:
                        continue
                    branch_aliases = dict(base_aliases)
                    visit(branch, branch_aliases, branch_intervals)
                    branch_states.append((branch_aliases, branch_intervals))
                if branch_states:
                    merged = branch_states[0][1]
                    for _, branch_intervals in branch_states[1:]:
                        merged = merge_intervals(merged, branch_intervals)
                    active_intervals.clear()
                    active_intervals.update(merged)
                    active_aliases.clear()
                    active_aliases.update(base_aliases)
                    merge_alias_states(active_aliases, branch_states)
                return
            if isinstance(value, BinaryOpNode) and self.map_operator(value.op) in {
                "&&",
                "||",
            }:
                visit(value.left, active_aliases, active_intervals)
                left_aliases = dict(active_aliases)
                left_intervals = dict(active_intervals)
                right_aliases = dict(left_aliases)
                right_intervals = dict(left_intervals)
                visit(value.right, right_aliases, right_intervals)
                active_intervals.clear()
                active_intervals.update(
                    merge_intervals(left_intervals, right_intervals)
                )
                active_aliases.clear()
                active_aliases.update(left_aliases)
                merge_alias_states(
                    active_aliases,
                    [
                        (left_aliases, left_intervals),
                        (right_aliases, right_intervals),
                    ],
                )
                return
            if isinstance(value, BlockNode):
                visit_scoped_sequence(value, active_aliases, active_intervals)
                return
            if isinstance(value, IfNode):
                condition = getattr(
                    value,
                    "condition",
                    getattr(value, "if_condition", None),
                )
                visit(
                    condition,
                    active_aliases,
                    active_intervals,
                )
                static_known, static_body = self.glsl_static_if_body(value)
                if static_known:
                    visit_scoped_sequence(
                        static_body,
                        active_aliases,
                        active_intervals,
                    )
                    return
                base_aliases = dict(active_aliases)
                branch_states = []
                then_aliases = dict(base_aliases)
                then_intervals = self.glsl_private_pointer_refined_condition_intervals(
                    condition,
                    active_intervals,
                    constants,
                    True,
                    excluded_names=mutated_interval_names,
                )
                if then_intervals is not None:
                    visit_scoped_sequence(
                        value.if_body,
                        then_aliases,
                        then_intervals,
                    )
                    branch_states.append((then_aliases, then_intervals))
                fallthrough_aliases = dict(base_aliases)
                fallthrough_intervals = (
                    self.glsl_private_pointer_refined_condition_intervals(
                        condition,
                        active_intervals,
                        constants,
                        False,
                        excluded_names=mutated_interval_names,
                    )
                )
                for else_if_condition, else_if_body in zip(
                    getattr(value, "else_if_conditions", []) or [],
                    getattr(value, "else_if_bodies", []) or [],
                ):
                    if fallthrough_intervals is None:
                        break
                    visit(
                        else_if_condition,
                        fallthrough_aliases,
                        fallthrough_intervals,
                    )
                    candidate_aliases = dict(fallthrough_aliases)
                    candidate = self.glsl_private_pointer_refined_condition_intervals(
                        else_if_condition,
                        fallthrough_intervals,
                        constants,
                        True,
                        excluded_names=mutated_interval_names,
                    )
                    if candidate is not None:
                        visit_scoped_sequence(
                            else_if_body,
                            candidate_aliases,
                            candidate,
                        )
                        branch_states.append((candidate_aliases, candidate))
                    fallthrough_intervals = (
                        self.glsl_private_pointer_refined_condition_intervals(
                            else_if_condition,
                            fallthrough_intervals,
                            constants,
                            False,
                            excluded_names=mutated_interval_names,
                        )
                    )
                if fallthrough_intervals is not None:
                    else_aliases = dict(fallthrough_aliases)
                    else_intervals = dict(fallthrough_intervals)
                    visit_scoped_sequence(
                        getattr(value, "else_body", None),
                        else_aliases,
                        else_intervals,
                    )
                    branch_states.append((else_aliases, else_intervals))
                if branch_states:
                    merged_intervals = branch_states[0][1]
                    for _, candidate in branch_states[1:]:
                        merged_intervals = merge_intervals(
                            merged_intervals,
                            candidate,
                        )
                    active_intervals.clear()
                    active_intervals.update(merged_intervals)
                    active_aliases.clear()
                    active_aliases.update(base_aliases)
                    merge_alias_states(active_aliases, branch_states)
                return
            if isinstance(value, (SwitchNode, MatchNode)):
                visit(
                    getattr(value, "expression", None), active_aliases, active_intervals
                )
                base_aliases = dict(active_aliases)
                base_intervals = dict(active_intervals)
                branches = list(
                    getattr(value, "cases", getattr(value, "arms", [])) or []
                )
                default_case = getattr(value, "default_case", None)
                branch_states = []
                for branch in branches:
                    branch_aliases = dict(base_aliases)
                    branch_intervals = dict(base_intervals)
                    visit(
                        getattr(branch, "value", getattr(branch, "guard", None)),
                        branch_aliases,
                        branch_intervals,
                    )
                    visit_scoped_sequence(
                        getattr(branch, "statements", getattr(branch, "body", None)),
                        branch_aliases,
                        branch_intervals,
                    )
                    branch_states.append((branch_aliases, branch_intervals))
                if default_case is not None:
                    default_aliases = dict(base_aliases)
                    default_intervals = dict(base_intervals)
                    visit_scoped_sequence(
                        getattr(default_case, "statements", default_case),
                        default_aliases,
                        default_intervals,
                    )
                    branch_states.append((default_aliases, default_intervals))
                else:
                    branch_states.append((dict(base_aliases), dict(base_intervals)))
                changed_aliases = set()
                for branch_aliases, branch_intervals in branch_states:
                    changed_aliases.update(
                        changed_alias_names(
                            base_aliases,
                            base_intervals,
                            branch_aliases,
                            branch_intervals,
                        )
                    )
                switch_mutations = self.glsl_function_mutated_interval_names(
                    SimpleNamespace(body=[branches, default_case])
                )
                if changed_aliases or switch_mutations:
                    conservative_aliases = dict(base_aliases)
                    for name in changed_aliases:
                        conservative_aliases[name] = poison_alias(
                            conservative_aliases[name]
                        )
                    conservative_intervals = dict(base_intervals)
                    clear_interval_names(conservative_intervals, switch_mutations)
                    for branch in branches:
                        visit_scoped_sequence(
                            getattr(
                                branch,
                                "statements",
                                getattr(branch, "body", None),
                            ),
                            dict(conservative_aliases),
                            dict(conservative_intervals),
                        )
                    if default_case is not None:
                        visit_scoped_sequence(
                            getattr(default_case, "statements", default_case),
                            dict(conservative_aliases),
                            dict(conservative_intervals),
                        )
                merged_intervals = branch_states[0][1]
                for _, candidate in branch_states[1:]:
                    merged_intervals = merge_intervals(
                        merged_intervals,
                        candidate,
                    )
                clear_interval_names(merged_intervals, switch_mutations)
                active_intervals.clear()
                active_intervals.update(merged_intervals)
                active_aliases.clear()
                active_aliases.update(base_aliases)
                merge_alias_states(active_aliases, branch_states)
                for name in changed_aliases:
                    active_aliases[name] = poison_alias(active_aliases[name])
                return
            if isinstance(value, ForNode):
                original_aliases = dict(active_aliases)
                original_intervals = dict(active_intervals)
                loop_aliases = dict(active_aliases)
                loop_intervals = dict(active_intervals)
                init = getattr(value, "init", None)
                visit(init, loop_aliases, loop_intervals)
                entry_aliases = dict(loop_aliases)
                entry_intervals = dict(loop_intervals)
                loop_name, loop_interval = self.glsl_private_pointer_loop_interval(
                    value,
                    loop_intervals,
                    constants,
                )
                loop_mutations = self.glsl_function_mutated_interval_names(
                    SimpleNamespace(
                        body=[
                            getattr(value, "condition", None),
                            getattr(value, "body", None),
                            getattr(value, "update", None),
                        ]
                    )
                )
                clear_interval_names(loop_intervals, loop_mutations)
                body_mutations = self.glsl_function_mutated_interval_names(
                    SimpleNamespace(
                        body=[
                            getattr(value, "condition", None),
                            getattr(value, "body", None),
                        ]
                    )
                )
                if loop_name in body_mutations:
                    loop_interval = None
                if loop_name:
                    if loop_interval is None:
                        loop_intervals.pop(loop_name, None)
                    else:
                        loop_intervals[loop_name] = loop_interval
                visit(
                    getattr(value, "condition", None),
                    loop_aliases,
                    loop_intervals,
                )
                visit_scoped_sequence(
                    getattr(value, "body", None),
                    loop_aliases,
                    loop_intervals,
                )
                visit(
                    getattr(value, "update", None),
                    loop_aliases,
                    loop_intervals,
                )
                changed = changed_alias_names(
                    entry_aliases,
                    entry_intervals,
                    loop_aliases,
                    loop_intervals,
                )
                exact_values = (
                    exact_classic_for_iteration_values(value) if changed else None
                )
                if exact_values is not None:
                    loop_aliases = dict(entry_aliases)
                    loop_intervals = dict(entry_intervals)
                    for iteration_value in exact_values:
                        if loop_name:
                            loop_intervals[loop_name] = (
                                iteration_value,
                                iteration_value,
                            )
                        visit(
                            getattr(value, "condition", None),
                            loop_aliases,
                            loop_intervals,
                        )
                        visit_scoped_sequence(
                            getattr(value, "body", None),
                            loop_aliases,
                            loop_intervals,
                        )
                        visit(
                            getattr(value, "update", None),
                            loop_aliases,
                            loop_intervals,
                        )
                    visit(
                        getattr(value, "condition", None),
                        loop_aliases,
                        loop_intervals,
                    )
                    post_aliases = loop_aliases
                    post_intervals = loop_intervals
                else:
                    if changed:
                        repeat_aliases = dict(loop_aliases)
                        for name in changed:
                            repeat_aliases[name] = poison_alias(
                                repeat_aliases.get(name, entry_aliases[name])
                            )
                        repeat_intervals = dict(loop_intervals)
                        visit(
                            getattr(value, "condition", None),
                            repeat_aliases,
                            repeat_intervals,
                        )
                        visit_scoped_sequence(
                            getattr(value, "body", None),
                            repeat_aliases,
                            repeat_intervals,
                        )
                        visit(
                            getattr(value, "update", None),
                            repeat_aliases,
                            repeat_intervals,
                        )
                        loop_aliases = repeat_aliases
                        loop_intervals = repeat_intervals
                    post_aliases = dict(entry_aliases)
                    merge_alias_states(
                        post_aliases,
                        [
                            (entry_aliases, entry_intervals),
                            (loop_aliases, loop_intervals),
                        ],
                    )
                    post_intervals = merge_intervals(
                        entry_intervals,
                        loop_intervals,
                    )
                    clear_interval_names(post_intervals, loop_mutations)
                init_name = (
                    getattr(init, "name", None)
                    if isinstance(init, (VariableNode, ArrayNode))
                    else None
                )
                propagate_outer_state(
                    active_aliases,
                    active_intervals,
                    original_aliases,
                    original_intervals,
                    post_aliases,
                    post_intervals,
                    shadow_names={init_name} if init_name else (),
                )
                return
            if isinstance(value, ForInNode):
                visit(
                    getattr(value, "iterable", None),
                    active_aliases,
                    active_intervals,
                )
                original_aliases = dict(active_aliases)
                original_intervals = dict(active_intervals)
                loop_aliases = dict(active_aliases)
                loop_intervals = dict(active_intervals)
                iterable = getattr(value, "iterable", None)
                pattern_interval = self.glsl_for_in_pattern_interval(
                    value,
                    loop_intervals,
                    constants,
                )
                pattern = getattr(value, "pattern", None)
                if pattern:
                    loop_aliases.pop(pattern, None)
                    loop_intervals.pop(pattern, None)
                    self.glsl_clear_index_component_intervals(
                        loop_intervals,
                        pattern,
                    )
                    if pattern_interval is not None:
                        loop_intervals[pattern] = pattern_interval
                body_mutations = self.glsl_function_mutated_interval_names(
                    SimpleNamespace(body=getattr(value, "body", None))
                )
                body_mutations.discard(pattern)
                clear_interval_names(loop_intervals, body_mutations)
                exact_values = exact_for_in_iteration_values(value)
                if exact_values == ():
                    post_aliases = dict(original_aliases)
                    post_intervals = dict(original_intervals)
                else:
                    visit_scoped_sequence(
                        getattr(value, "body", None),
                        loop_aliases,
                        loop_intervals,
                    )
                    changed = changed_alias_names(
                        {
                            name: binding
                            for name, binding in original_aliases.items()
                            if name != pattern
                        },
                        original_intervals,
                        loop_aliases,
                        loop_intervals,
                    )
                    if changed and exact_values is not None:
                        loop_aliases = dict(original_aliases)
                        if pattern:
                            loop_aliases.pop(pattern, None)
                        loop_intervals = dict(original_intervals)
                        if pattern:
                            loop_intervals.pop(pattern, None)
                            self.glsl_clear_index_component_intervals(
                                loop_intervals,
                                pattern,
                            )
                        for iteration_value in exact_values:
                            if pattern:
                                loop_intervals[pattern] = (
                                    iteration_value,
                                    iteration_value,
                                )
                            visit_scoped_sequence(
                                getattr(value, "body", None),
                                loop_aliases,
                                loop_intervals,
                            )
                        post_aliases = loop_aliases
                        post_intervals = loop_intervals
                    else:
                        if changed:
                            repeat_aliases = dict(loop_aliases)
                            for name in changed:
                                repeat_aliases[name] = poison_alias(
                                    repeat_aliases.get(
                                        name,
                                        original_aliases[name],
                                    )
                                )
                            repeat_intervals = dict(loop_intervals)
                            visit_scoped_sequence(
                                getattr(value, "body", None),
                                repeat_aliases,
                                repeat_intervals,
                            )
                            loop_aliases = repeat_aliases
                            loop_intervals = repeat_intervals
                        post_aliases = dict(original_aliases)
                        merge_alias_states(
                            post_aliases,
                            [
                                (original_aliases, original_intervals),
                                (loop_aliases, loop_intervals),
                            ],
                        )
                        post_intervals = merge_intervals(
                            original_intervals,
                            loop_intervals,
                        )
                        clear_interval_names(post_intervals, body_mutations)
                propagate_outer_state(
                    active_aliases,
                    active_intervals,
                    original_aliases,
                    original_intervals,
                    post_aliases,
                    post_intervals,
                    shadow_names={pattern} if pattern else (),
                )
                return
            if isinstance(value, (WhileNode, DoWhileNode, LoopNode)):
                original_aliases = dict(active_aliases)
                original_intervals = dict(active_intervals)
                condition = getattr(value, "condition", None)
                condition_literal = (
                    evaluate_literal_int_expression(condition, constants)
                    if condition is not None
                    else None
                )
                if isinstance(value, WhileNode) and condition_literal == 0:
                    visit(condition, active_aliases, active_intervals)
                    return
                loop_aliases = dict(active_aliases)
                loop_intervals = dict(active_intervals)
                loop_mutations = self.glsl_function_mutated_interval_names(
                    SimpleNamespace(body=[condition, getattr(value, "body", None)])
                )
                clear_interval_names(loop_intervals, loop_mutations)
                if isinstance(value, DoWhileNode):
                    visit_scoped_sequence(
                        getattr(value, "body", None),
                        loop_aliases,
                        loop_intervals,
                    )
                    visit(condition, loop_aliases, loop_intervals)
                elif isinstance(value, WhileNode):
                    visit(condition, loop_aliases, loop_intervals)
                    visit_scoped_sequence(
                        getattr(value, "body", None),
                        loop_aliases,
                        loop_intervals,
                    )
                else:
                    visit_scoped_sequence(
                        getattr(value, "body", None),
                        loop_aliases,
                        loop_intervals,
                    )
                changed = changed_alias_names(
                    original_aliases,
                    original_intervals,
                    loop_aliases,
                    loop_intervals,
                )
                repeats = not (
                    isinstance(value, DoWhileNode) and condition_literal == 0
                )
                if changed and repeats:
                    repeat_aliases = dict(loop_aliases)
                    for name in changed:
                        repeat_aliases[name] = poison_alias(
                            repeat_aliases.get(name, original_aliases[name])
                        )
                    repeat_intervals = dict(loop_intervals)
                    if isinstance(value, DoWhileNode):
                        visit_scoped_sequence(
                            getattr(value, "body", None),
                            repeat_aliases,
                            repeat_intervals,
                        )
                        visit(condition, repeat_aliases, repeat_intervals)
                    elif isinstance(value, WhileNode):
                        visit(condition, repeat_aliases, repeat_intervals)
                        visit_scoped_sequence(
                            getattr(value, "body", None),
                            repeat_aliases,
                            repeat_intervals,
                        )
                    else:
                        visit_scoped_sequence(
                            getattr(value, "body", None),
                            repeat_aliases,
                            repeat_intervals,
                        )
                    loop_aliases = repeat_aliases
                    loop_intervals = repeat_intervals
                if isinstance(value, DoWhileNode) and condition_literal == 0:
                    post_aliases = loop_aliases
                    post_intervals = loop_intervals
                else:
                    post_aliases = dict(original_aliases)
                    merge_alias_states(
                        post_aliases,
                        [
                            (original_aliases, original_intervals),
                            (loop_aliases, loop_intervals),
                        ],
                    )
                    post_intervals = merge_intervals(
                        original_intervals,
                        loop_intervals,
                    )
                    clear_interval_names(post_intervals, loop_mutations)
                propagate_outer_state(
                    active_aliases,
                    active_intervals,
                    original_aliases,
                    original_intervals,
                    post_aliases,
                    post_intervals,
                )
                return
            if hasattr(value, "child_nodes"):
                for child in value.child_nodes():
                    visit(child, active_aliases, active_intervals)

        # The body is analyzed outside its own generation scope, so its parameters
        # must shadow same-named caller locals when helper calls are summarized.
        parameter_types = {
            parameter.name: getattr(
                parameter, "param_type", getattr(parameter, "vtype", None)
            )
            for parameter in parameters
            if getattr(parameter, "name", None)
        }
        with self.glsl_lexical_source_type_scope(parameter_types):
            visit_sequence(getattr(function, "body", []), aliases, intervals)
        return access_intervals

    def glsl_resource_function_specialization_key(
        self,
        func_name,
        args,
        aliases,
        *,
        lexical_source_types=None,
    ):
        callee = self.glsl_resource_function_source(
            func_name,
            args,
            aliases,
            lexical_source_types=lexical_source_types,
        )
        if callee is None:
            return None, None

        params = list(getattr(callee, "parameters", getattr(callee, "params", [])))
        # Argument intervals depend on the caller's lexical types, as the binding does.
        with self.glsl_lexical_source_type_scope(lexical_source_types):
            workgroup_parameter_intervals = (
                self.glsl_workgroup_call_parameter_intervals(callee, args)
            )
        access_requirements = self.function_image_access_requirements.get(func_name, {})
        bindings = {}
        key_parts = []
        for index, (param, arg) in enumerate(zip(params, args or [])):
            pointer_array_contract = self.glsl_storage_pointer_array_contract(param)
            if pointer_array_contract is not None:
                param_name = getattr(param, "name", None)
                pointer_array_contract = {
                    **pointer_array_contract,
                    "parameter_name": param_name,
                }
                extent = pointer_array_contract.get("extent")
                if not param_name or not isinstance(extent, int):
                    raise OpenGLStoragePointerError(
                        "OpenGL fixed storage-pointer-array specialization "
                        "requires a named positive compile-time extent",
                        function_name=getattr(callee, "name", None),
                        parameter_name=param_name,
                        reason="pointer-array-extent-unresolved",
                        source_location=getattr(param, "source_location", None),
                    )
                binding = self.glsl_storage_pointer_array_binding(
                    arg,
                    aliases,
                    pointer_array_contract,
                )
                if binding is None:
                    return None, None
                if binding.get("extent") != extent:
                    raise OpenGLStoragePointerError(
                        "OpenGL fixed storage-pointer-array argument extent "
                        f"{binding.get('extent')} does not match "
                        f"'{getattr(callee, 'name', None)}.{param_name}' extent "
                        f"{extent}",
                        function_name=getattr(callee, "name", None),
                        parameter_name=param_name,
                        reason="pointer-array-extent-mismatch",
                        source_location=getattr(arg, "source_location", None),
                    )
                if binding.get("element_type") != pointer_array_contract.get(
                    "element_type"
                ):
                    raise OpenGLStoragePointerError(
                        "OpenGL fixed storage-pointer-array argument changes "
                        f"element type for '{getattr(callee, 'name', None)}."
                        f"{param_name}'",
                        function_name=getattr(callee, "name", None),
                        parameter_name=param_name,
                        reason="element-type-mismatch",
                        source_location=getattr(arg, "source_location", None),
                    )
                required_access = pointer_array_contract.get("required_access")
                if not image_access_satisfies_requirement(
                    required_access, binding.get("access")
                ):
                    raise OpenGLStoragePointerError(
                        "OpenGL fixed storage-pointer-array argument for "
                        f"'{getattr(callee, 'name', None)}.{param_name}' cannot "
                        f"provide {required_access} access from "
                        f"{binding.get('access') or 'unspecified'} storage",
                        function_name=getattr(callee, "name", None),
                        parameter_name=param_name,
                        reason="access-mismatch",
                        source_location=getattr(arg, "source_location", None),
                    )
                offset_direction = (
                    "inout"
                    if self.glsl_storage_pointer_array_parameter_is_mutated(
                        callee, param_name
                    )
                    else None
                )
                binding = {
                    **binding,
                    **pointer_array_contract,
                    "kind": "storage-pointer-array",
                    "offset_direction": offset_direction,
                    "resource_root": False,
                }
                with self.glsl_lexical_source_type_scope(lexical_source_types):
                    offset_info = self.glsl_value_type_info(
                        self.glsl_source_expression_type(binding.get("offset", 0))
                    )
                    if (
                        offset_info is not None
                        and offset_info["width"] == 1
                        and offset_info["bits"] > 32
                    ):
                        binding["wide_offset"] = True
                    if (
                        isinstance(arg, UnaryOpNode)
                        and self.map_operator(arg.op) == "&"
                    ):
                        scalar = self.glsl_storage_pointer_binding(arg.operand, aliases)
                        bounds = self.glsl_checked_private_pointer_interval(
                            arg.operand,
                            getattr(arg, "_glsl_control_flow_intervals", {}),
                            self.glsl_active_literal_int_constants(),
                        )
                        offset_range = (
                            IntegerRange(*bounds)
                            if bounds is not None
                            else (
                                self.glsl_storage_pointer_offset_range(scalar, arg)
                                if scalar is not None
                                else None
                            )
                        )
                        source_intervals = (
                            (
                                (offset_range.minimum, offset_range.maximum)
                                if offset_range is not None
                                and 0
                                <= offset_range.minimum
                                <= offset_range.maximum
                                < (1 << 31)
                                else None
                            ),
                        )
                    else:
                        flow = getattr(arg, "_glsl_control_flow_intervals", {})
                        source_intervals = tuple(
                            flow.get(
                                self.glsl_pointer_array_interval_key(
                                    self.expression_name(arg), element_index
                                )
                            )
                            for element_index in range(extent)
                        )
                if (
                    "pointer_reinterpretation" in binding
                    or "byte_offset" in binding
                    or getattr(arg, "_glsl_pointer_argument_mutated", False)
                ):
                    source_intervals = (None,) * extent
                binding["source_offset_intervals"] = source_intervals
                bindings[index] = (param_name, binding)
                key_parts.append(
                    (
                        index,
                        "storage-pointer-array",
                        binding.get("root"),
                        binding.get("element_type"),
                        extent,
                        required_access,
                        offset_direction,
                        source_intervals,
                        binding.get("wide_offset", False),
                    )
                )
                continue
            fixed_array_contract = self.glsl_fixed_array_parameter_contract(
                callee, param
            )
            if fixed_array_contract is not None:
                binding = self.glsl_fixed_array_storage_binding(
                    callee,
                    param,
                    arg,
                    aliases,
                    fixed_array_contract,
                )
                if binding is not None:
                    param_name = getattr(param, "name", None)
                    extent = fixed_array_contract.get("extent")
                    required_access = fixed_array_contract.get("required_access")
                    actual_access = binding.get("access")
                    if extent is None:
                        self.raise_glsl_fixed_array_resource_error(
                            "OpenGL fixed-array storage-resource specialization "
                            "requires a positive compile-time extent for "
                            f"'{getattr(callee, 'name', None)}.{param_name}'",
                            function=callee,
                            parameter=param,
                            argument=arg,
                            binding=binding,
                            fixed_extent=(
                                fixed_array_contract.get("extent_text") or None
                            ),
                            required_access=required_access,
                            actual_access=actual_access,
                            reason="unknown-fixed-extent",
                        )
                    parameter_element_type = fixed_array_contract.get("element_type")
                    if (
                        parameter_element_type is None
                        or binding.get("element_type") != parameter_element_type
                    ):
                        self.raise_glsl_fixed_array_resource_error(
                            "OpenGL fixed-array storage-resource specialization "
                            "cannot preserve the parameter element type for "
                            f"'{getattr(callee, 'name', None)}.{param_name}'",
                            function=callee,
                            parameter=param,
                            argument=arg,
                            binding=binding,
                            fixed_extent=extent,
                            required_access=required_access,
                            actual_access=actual_access,
                            reason="element-type-conflict",
                        )
                    if not image_access_satisfies_requirement(
                        required_access, actual_access
                    ):
                        conflict = (
                            "writable-access-conflict"
                            if required_access in {"write", "read_write"}
                            else "readable-access-conflict"
                        )
                        self.raise_glsl_fixed_array_resource_error(
                            "OpenGL fixed-array helper "
                            f"'{getattr(callee, 'name', None)}' requires "
                            f"{required_access} access for '{param_name}', but "
                            f"storage resource '{binding.get('root')}' is "
                            f"{actual_access or 'unspecified'}",
                            function=callee,
                            parameter=param,
                            argument=arg,
                            binding=binding,
                            fixed_extent=extent,
                            required_access=required_access,
                            actual_access=actual_access,
                            reason=conflict,
                        )
                    binding = {
                        **binding,
                        "kind": "fixed-array-storage",
                        "parameter_element_type": parameter_element_type,
                        "fixed_extent": extent,
                        "required_access": required_access,
                    }
                    bindings[index] = (param_name, binding)
                    key_parts.append(
                        (
                            index,
                            "fixed-array-storage",
                            binding.get("root"),
                            parameter_element_type,
                            extent,
                            required_access,
                            actual_access,
                        )
                    )
                    continue
            workgroup_element_type = self.glsl_workgroup_pointer_element_type(param)
            if workgroup_element_type is not None:
                param_name = getattr(param, "name", None)
                if not param_name:
                    return None, None
                binding = self.glsl_workgroup_pointer_binding(arg, aliases)
                if binding is None:
                    if not self.glsl_workgroup_pointer_null_expression(arg):
                        return None, None
                    if self.glsl_workgroup_pointer_parameter_has_reachable_use(
                        callee, param_name
                    ):
                        return None, None
                    binding = {
                        "kind": "workgroup-pointer-unused",
                        "element_type": workgroup_element_type,
                        "parameter_element_type": workgroup_element_type,
                    }
                    bindings[index] = (param_name, binding)
                    key_parts.append(
                        (
                            index,
                            "workgroup-pointer-unused",
                            workgroup_element_type,
                        )
                    )
                    continue
                if binding.get("array_size") is None:
                    return None, None
                if binding.get("element_type") != workgroup_element_type:
                    return None, None
                binding = {
                    **binding,
                    "kind": "workgroup-pointer",
                    "parameter_element_type": workgroup_element_type,
                }
                self.validate_glsl_workgroup_pointer_specialization(
                    callee,
                    param,
                    arg,
                    binding,
                    workgroup_parameter_intervals,
                )
                bindings[index] = (param_name, binding)
                key_parts.append(
                    (
                        index,
                        "workgroup-pointer",
                        binding["root"],
                        binding["array_size"],
                        workgroup_element_type,
                        binding.get("entry_owner"),
                        binding.get("offset_interval"),
                    )
                )
                continue
            storage_element_type = self.glsl_storage_pointer_element_type(param)
            if storage_element_type is not None:
                param_name = getattr(param, "name", None)
                if not param_name:
                    return None, None
                binding = self.glsl_storage_pointer_binding(arg, aliases)
                if binding is None:
                    if not self.glsl_workgroup_pointer_null_expression(arg):
                        return None, None
                    if self.glsl_storage_pointer_parameter_has_reachable_use(
                        callee, param_name
                    ):
                        return None, None
                    binding = {
                        "kind": "storage-pointer-unused",
                        "element_type": storage_element_type,
                        "parameter_element_type": storage_element_type,
                    }
                    bindings[index] = (param_name, binding)
                    key_parts.append(
                        (
                            index,
                            "storage-pointer-unused",
                            storage_element_type,
                        )
                    )
                    continue
                if binding.get("element_type") != storage_element_type:
                    return None, None
                required_access = self.glsl_storage_pointer_parameter_required_access(
                    callee, param
                )
                offset_direction = self.glsl_storage_pointer_offset_parameter_direction(
                    param
                )
                if not image_access_satisfies_requirement(
                    required_access, binding.get("access")
                ):
                    return None, None
                binding = {
                    **binding,
                    "kind": "storage-pointer",
                    "parameter_element_type": storage_element_type,
                    "required_access": required_access,
                    "offset_direction": offset_direction,
                }
                with self.glsl_lexical_source_type_scope(lexical_source_types):
                    offset_range = self.glsl_storage_pointer_offset_range(binding, arg)
                    offset_info = self.glsl_value_type_info(
                        self.glsl_source_expression_type(binding.get("offset", 0))
                    )
                if (
                    offset_info is not None
                    and offset_info["width"] == 1
                    and offset_info["bits"] > 32
                ):
                    binding["wide_offset"] = True
                binding["source_offset_interval"] = (
                    (offset_range.minimum, offset_range.maximum)
                    if offset_range is not None
                    and 0 <= offset_range.minimum <= offset_range.maximum < (1 << 31)
                    else None
                )
                # A base that may lie before the buffer keeps its relation to
                # the callee's scalar arguments for guarded subscripts.
                binding.pop("guarded_offset_relation", None)
                if binding["source_offset_interval"] is None:
                    with self.glsl_lexical_source_type_scope(lexical_source_types):
                        relation = call_offset_relation(
                            self,
                            callee,
                            params,
                            args,
                            index,
                            binding,
                            local_names=lexical_source_types,
                        )
                    if relation is not None:
                        binding["guarded_offset_relation"] = relation
                bindings[index] = (param_name, binding)
                key_parts.append(
                    (
                        index,
                        "storage-pointer",
                        binding["root"],
                        storage_element_type,
                        required_access,
                        offset_direction,
                        binding.get("source_offset_interval"),
                        binding.get("wide_offset", False),
                        self.glsl_storage_pointer_specialization_view_key(binding),
                    )
                    + (
                        (binding["guarded_offset_relation"],)
                        if "guarded_offset_relation" in binding
                        else ()
                    )
                )
                continue
            param_type = self.type_name_string(
                getattr(param, "param_type", getattr(param, "vtype", None))
            )
            if self.structured_buffer_type_name(param_type) in {
                "StructuredBuffer",
                "RWStructuredBuffer",
            }:
                param_name = getattr(param, "name", None)
                binding = self.glsl_storage_pointer_binding(arg, aliases)
                if param_name and binding is not None:
                    storage_element_type = self.map_type(
                        self.structured_buffer_element_type(param_type)
                    )
                    if binding.get("element_type") != storage_element_type:
                        return None, None
                    required_access = (
                        "read"
                        if self.structured_buffer_type_name(param_type)
                        == "StructuredBuffer"
                        else "read_write"
                    )
                    if not image_access_satisfies_requirement(
                        required_access, binding.get("access")
                    ):
                        return None, None
                    binding = {
                        **binding,
                        "kind": "storage-pointer",
                        "parameter_element_type": storage_element_type,
                        "required_access": required_access,
                    }
                    bindings[index] = (param_name, binding)
                    key_parts.append(
                        (
                            index,
                            "storage-pointer",
                            binding["root"],
                            storage_element_type,
                            required_access,
                            binding.get("wide_offset", False),
                            self.glsl_storage_pointer_specialization_view_key(binding),
                        )
                    )
                    continue
            if not self.is_storage_image_type(param_type):
                continue
            binding = self.glsl_resource_binding_info(arg, aliases)
            if binding is None:
                continue
            param_name = getattr(param, "name", None)
            if not param_name:
                continue
            array_size = self.glsl_resource_parameter_array_size(
                func_name, param_name, param_type
            )
            if array_size is not None:
                binding = {**binding, "array_size": array_size}
            self.validate_storage_image_parameter_contract(
                func_name,
                param,
                arg,
                index,
                self.storage_image_binding_contract(binding),
            )
            required_access = access_requirements.get(param_name)
            if required_access is not None and not image_access_satisfies_requirement(
                required_access,
                binding.get("access"),
            ):
                return None, None
            if not binding.get("specializable", True):
                continue
            bindings[index] = (param_name, binding)
            key_parts.append((index, binding["expression"]))

        if not bindings:
            return None, None
        # Every transferred scalar fact can affect bounds or reachability in
        # the clone, including helpers whose only resource is a storage buffer.
        if workgroup_parameter_intervals or any(
            binding.get("kind")
            in {
                "workgroup-pointer",
                "workgroup-pointer-unused",
            }
            for _parameter_name, binding in bindings.values()
        ):
            key_parts.append(
                (
                    "workgroup-proof-intervals",
                    tuple(sorted(workgroup_parameter_intervals.items())),
                )
            )
        source_signature = self.glsl_source_function_signature(callee)
        return (func_name, source_signature, tuple(key_parts)), bindings

    def ensure_glsl_resource_function_specialization(
        self,
        func_name,
        args,
        aliases,
        *,
        lexical_source_types=None,
    ):
        key, bindings = self.glsl_resource_function_specialization_key(
            func_name,
            args,
            aliases,
            lexical_source_types=lexical_source_types,
        )
        if key is None:
            return None
        if key in self.glsl_resource_function_specializations:
            return self.glsl_resource_function_specializations[key]

        source_func = self.glsl_resource_function_source(
            func_name,
            args,
            aliases,
            lexical_source_types=lexical_source_types,
        )
        if source_func is None:
            return None

        with self.glsl_lexical_source_type_scope(lexical_source_types):
            proof_parameter_intervals = self.glsl_workgroup_call_parameter_intervals(
                source_func,
                args,
            )
        copied_nodes = {}
        clone = deepcopy(source_func, copied_nodes)
        # Resource specialization changes bindings, not the proven control flow.
        # Copy only facts attached to these exact source nodes, never by name.
        for proven in (
            self.glsl_software_subgroup_uniform_for_node_ids,
            self.glsl_software_subgroup_uniform_if_node_ids,
            self.glsl_software_subgroup_zero_trip_for_node_ids,
        ):
            proven.update(
                id(copied_nodes[node_id])
                for node_id in tuple(proven)
                if node_id in copied_nodes
            )
        null_storage_pointer_parameters = {
            parameter_name
            for parameter_name, binding in bindings.values()
            if binding.get("kind") == "storage-pointer-unused"
        }
        if null_storage_pointer_parameters:
            self.glsl_replace_null_pointer_parameter_references(
                clone,
                null_storage_pointer_parameters,
            )
        null_workgroup_pointer_parameters = {
            parameter_name
            for parameter_name, binding in bindings.values()
            if binding.get("kind") == "workgroup-pointer-unused"
        }
        if null_workgroup_pointer_parameters:
            self.glsl_replace_null_pointer_parameter_references(
                clone,
                null_workgroup_pointer_parameters,
                error_type=OpenGLWorkgroupPointerError,
                pointer_kind="workgroup",
            )
        self.glsl_fold_null_workgroup_pointer_locals(clone)
        clone.name = self.glsl_resource_specialization_name(func_name, bindings)
        clone.parameters = [
            param
            for index, param in enumerate(
                getattr(clone, "parameters", getattr(clone, "params", []))
            )
            if index not in bindings
        ]
        remaining_param_names = {
            getattr(param, "name", None)
            for param in getattr(clone, "parameters", getattr(clone, "params", []))
        }
        occupied_local_names = set(remaining_param_names)
        occupied_local_names.update(
            getattr(node, "name", None)
            for node in self.walk_ast(getattr(clone, "body", []))
            if isinstance(node, (VariableNode, ArrayNode))
            and getattr(node, "name", None)
        )
        dynamic_call_arguments = []
        for _, (_, binding) in sorted(bindings.items()):
            if binding.get("kind") in {
                "workgroup-pointer",
                "workgroup-pointer-unused",
            }:
                continue
            for dynamic_param in binding.get("dynamic_parameters") or []:
                param_name = dynamic_param.get("name")
                if not param_name or param_name in remaining_param_names:
                    continue
                remaining_param_names.add(param_name)
                clone.parameters.append(
                    SimpleNamespace(
                        name=param_name,
                        param_type=dynamic_param.get("type", "int"),
                        vtype=dynamic_param.get("type", "int"),
                        semantic=None,
                        attributes=[],
                        qualifiers=[],
                    )
                )
                dynamic_call_arguments.append(
                    dynamic_param.get("expression", param_name)
                )
        workgroup_pointer_aliases = {}
        workgroup_pointer_bound_indices = []
        for index, (param_name, binding) in sorted(bindings.items()):
            if binding.get("kind") != "workgroup-pointer":
                continue
            offset_name = self.glsl_synthetic_local_identifier(
                f"{sanitize_type_name(param_name)}_offset",
                occupied_local_names,
            )
            remaining_param_names.add(offset_name)
            clone.parameters.append(
                SimpleNamespace(
                    name=offset_name,
                    param_type="int",
                    vtype="int",
                    semantic=None,
                    attributes=[],
                    qualifiers=[],
                )
            )
            workgroup_pointer_bound_indices.append(index)
            workgroup_pointer_aliases[param_name] = {
                **binding,
                "offset": offset_name,
            }
        storage_pointer_aliases = {}
        storage_pointer_bound_indices = []
        storage_pointer_byte_offset_indices = []
        storage_pointer_offset_directions = {}
        storage_pointer_parameter_names = {}
        for index, (param_name, binding) in sorted(bindings.items()):
            if binding.get("kind") not in {
                "storage-pointer",
                "fixed-array-storage",
            }:
                continue
            offset_name = self.glsl_synthetic_local_identifier(
                f"{sanitize_type_name(param_name)}_offset",
                occupied_local_names,
            )
            remaining_param_names.add(offset_name)
            offset_direction = binding.get("offset_direction")
            if binding.get("source_offset_interval") is not None:
                proof_parameter_intervals[offset_name] = binding[
                    "source_offset_interval"
                ]
            clone.parameters.append(
                SimpleNamespace(
                    name=offset_name,
                    param_type="int",
                    vtype="int",
                    semantic=None,
                    attributes=[],
                    qualifiers=[offset_direction] if offset_direction else [],
                )
            )
            byte_offset_name = None
            if binding.get("pointer_reinterpretation") is not None or str(
                binding.get("byte_offset", "0")
            ) not in {"0", "0u"}:
                byte_offset_name = self.glsl_synthetic_local_identifier(
                    f"{sanitize_type_name(param_name)}_byte_offset",
                    occupied_local_names,
                )
                remaining_param_names.add(byte_offset_name)
                clone.parameters.append(
                    SimpleNamespace(
                        name=byte_offset_name,
                        param_type="int",
                        vtype="int",
                        semantic=None,
                        attributes=[],
                        qualifiers=[],
                    )
                )
                storage_pointer_byte_offset_indices.append(index)
            storage_pointer_bound_indices.append(index)
            storage_pointer_offset_directions[index] = offset_direction
            storage_pointer_parameter_names[index] = param_name
            storage_pointer_aliases[param_name] = {
                **binding,
                "specialization_kind": binding.get("kind"),
                "kind": "storage-pointer",
                "offset": offset_name,
                "resource_root": False,
            }
            if byte_offset_name is not None:
                storage_pointer_aliases[param_name]["byte_offset"] = byte_offset_name
        storage_pointer_array_bound_indices = []
        storage_pointer_array_offset_directions = {}
        storage_pointer_array_parameter_names = {}
        for index, (param_name, binding) in sorted(bindings.items()):
            if binding.get("kind") != "storage-pointer-array":
                continue
            extent = binding.get("extent")
            offset_direction = binding.get("offset_direction")
            offset_names = []
            for element_index in range(extent):
                offset_name = self.glsl_synthetic_local_identifier(
                    f"{sanitize_type_name(param_name)}_offset_{element_index}",
                    occupied_local_names,
                )
                remaining_param_names.add(offset_name)
                clone.parameters.append(
                    SimpleNamespace(
                        name=offset_name,
                        param_type="int",
                        vtype="int",
                        semantic=None,
                        attributes=[],
                        qualifiers=[offset_direction] if offset_direction else [],
                    )
                )
                offset_names.append(offset_name)
            storage_pointer_array_bound_indices.append(index)
            storage_pointer_array_offset_directions[index] = offset_direction
            storage_pointer_array_parameter_names[index] = param_name
            storage_pointer_aliases[param_name] = {
                **binding,
                "kind": "storage-pointer-array",
                "offset_array": None,
                "offset_expressions": offset_names,
                "resource_root": False,
            }
        clone._glsl_resource_source_name = func_name
        clone._glsl_resource_source_function_id = (
            getattr(source_func, "_glsl_resource_source_function_id", None)
            or getattr(source_func, "_generic_source_function_id", None)
            or id(source_func)
        )
        clone._glsl_resource_specialization_key = key
        clone._glsl_resource_bound_indices = set(bindings)
        clone._glsl_resource_dynamic_call_arguments = dynamic_call_arguments
        clone._glsl_resource_aliases = {
            param_name: binding
            for param_name, binding in bindings.values()
            if binding.get("kind")
            not in {
                "storage-pointer-unused",
                "fixed-array-storage",
                "storage-pointer",
                "storage-pointer-array",
                "workgroup-pointer",
                "workgroup-pointer-unused",
            }
        }
        clone._glsl_workgroup_pointer_aliases = workgroup_pointer_aliases
        clone._glsl_workgroup_pointer_bound_indices = tuple(
            workgroup_pointer_bound_indices
        )
        clone._glsl_storage_pointer_aliases = storage_pointer_aliases
        clone._glsl_storage_pointer_bound_indices = tuple(storage_pointer_bound_indices)
        clone._glsl_storage_pointer_byte_offset_indices = frozenset(
            storage_pointer_byte_offset_indices
        )
        clone._glsl_storage_pointer_offset_directions = (
            storage_pointer_offset_directions
        )
        clone._glsl_storage_pointer_parameter_names = storage_pointer_parameter_names
        clone._glsl_storage_pointer_array_bound_indices = tuple(
            storage_pointer_array_bound_indices
        )
        clone._glsl_storage_pointer_array_offset_directions = (
            storage_pointer_array_offset_directions
        )
        clone._glsl_storage_pointer_array_parameter_names = (
            storage_pointer_array_parameter_names
        )
        clone._glsl_workgroup_proof_parameter_intervals = {
            name: interval
            for name, interval in proof_parameter_intervals.items()
            if name in remaining_param_names
            or any(
                name.startswith(f"{parameter_name}.")
                for parameter_name in remaining_param_names
                if parameter_name
            )
        }

        target_name = self.glsl_generated_module_identifier(
            ("resource-specialization", key),
            clone.name,
        )
        clone._glsl_workgroup_materialization_name = target_name
        for binding in workgroup_pointer_aliases.values():
            binding["materialization_name"] = target_name
        self.glsl_function_target_names[id(clone)] = target_name
        self.register_glsl_resource_specialization_private_pointer_metadata(
            source_func,
            clone,
            target_name,
        )
        self.register_glsl_function_target_metadata(target_name, clone)

        self.glsl_resource_function_specializations[key] = clone
        self.glsl_resource_function_call_names[key] = target_name
        self.glsl_resource_specialized_source_names.add(func_name)
        return clone

    def generate_function(
        self,
        func,
        indent=0,
        shader_type=None,
        execution_config=None,
        entry_name=None,
        stage_layout_qualifiers=None,
        stage_context=None,
        _private_pointer_variant=None,
    ):
        """Render a function or GLSL ``main`` stage entry point."""
        if shader_type is None and (
            self.should_elide_glsl_function(func)
            or self.glsl_resource_specializations_replace(func)
        ):
            return ""
        function_key = self.glsl_function_declaration_name(func)
        variants = self.function_private_pointer_backing_variants.get(function_key)
        if _private_pointer_variant is None and variants and shader_type is None:
            previous_variant = self.current_glsl_private_pointer_variant
            previous_base_names = self.current_glsl_private_pointer_base_names
            parameters = self.function_private_pointer_parameters[function_key]
            code = ""
            try:
                for variant in variants:
                    self.current_glsl_private_pointer_variant = {
                        parameter.name: variant[index]
                        for index, parameter in enumerate(parameters)
                    }
                    self.current_glsl_private_pointer_base_names = dict(
                        self.function_private_pointer_base_names[function_key]
                    )
                    code += self.generate_function(
                        func,
                        indent=indent,
                        shader_type=shader_type,
                        execution_config=execution_config,
                        entry_name=entry_name,
                        stage_layout_qualifiers=stage_layout_qualifiers,
                        stage_context=stage_context,
                        _private_pointer_variant=variant,
                    )
            finally:
                self.current_glsl_private_pointer_variant = previous_variant
                self.current_glsl_private_pointer_base_names = previous_base_names
            return code

        previous_private_pointer_function_name = (
            self.current_glsl_private_pointer_function_name
        )
        self.current_glsl_private_pointer_function_name = function_key
        code = ""
        code += "  " * indent

        param_list = getattr(func, "parameters", getattr(func, "params", []))
        for parameter in param_list:
            if self.glsl_fixed_array_parameter_shape(parameter) is not None:
                self.validate_glsl_fixed_array_parameter_pointer_escape(
                    func,
                    parameter,
                )
        if shader_type is not None:
            unresolved_workgroup_parameters = [
                parameter
                for parameter in param_list
                if self.glsl_workgroup_pointer_declaration(parameter)
            ]
            if unresolved_workgroup_parameters:
                parameter = unresolved_workgroup_parameters[0]
                raise OpenGLWorkgroupPointerError(
                    "OpenGL stage entry workgroup pointer parameter "
                    f"'{parameter.name}' requires a concrete shared-array extent",
                    function_name=getattr(func, "name", None),
                    parameter_name=parameter.name,
                    reason="missing-concrete-size",
                    source_location=getattr(parameter, "source_location", None),
                )
        parameter_source_names = {
            getattr(parameter, "name", None)
            for parameter in param_list
            if getattr(parameter, "name", None)
        }
        params = []
        sampler_parameters = set()
        texture_parameters = {}
        image_format_parameters = {}
        image_access_parameters = {}
        resource_aliases = getattr(func, "_glsl_resource_aliases", {}) or {}
        workgroup_pointer_aliases = (
            getattr(func, "_glsl_workgroup_pointer_aliases", {}) or {}
        )
        storage_pointer_aliases = {
            **self.glsl_global_storage_pointer_aliases,
            **(getattr(func, "_glsl_storage_pointer_aliases", {}) or {}),
            **self.stage_entry_storage_pointer_bindings.get(id(func), {}),
        }
        if shader_type is not None:
            for parameter in param_list:
                if (
                    self.stage_entry_metal_scalar_reference_element_type(parameter)
                    is not None
                ):
                    # Updating a bound reference writes its value, not a pointer offset.
                    storage_pointer_aliases.pop(parameter.name, None)
        stage_entry_resource_aliases = (
            self.stage_entry_resource_parameter_aliases.get(id(func), {})
            if shader_type is not None
            else {}
        )
        stage_entry_constant_value_aliases = (
            self.stage_entry_constant_value_parameter_aliases.get(id(func), {})
            if shader_type is not None
            else {}
        )
        unsupported_buffer_array_info = (
            self.unsupported_structured_buffer_array_functions.get(func.name, {})
        )
        unsupported_buffer_array_indices = unsupported_buffer_array_info.get(
            "indices", set()
        )
        previous_function_return_type = self.current_function_return_type
        previous_precise_return_type = self.current_glsl_precise_return_type
        previous_precise_function = self.current_glsl_precise_function
        previous_precise_captures = self.current_glsl_precise_captures
        previous_ieee_operator_arithmetic = self.current_glsl_ieee_operator_arithmetic
        previous_exactness_test_local_ids = self.current_glsl_exactness_test_local_ids
        previous_precise_value_writes = getattr(
            self, "current_glsl_precise_value_writes", {}
        )
        previous_local_variable_types = self.local_variable_types
        previous_local_variable_source_types = self.local_variable_source_types
        previous_identifier_aliases = self.current_identifier_aliases
        previous_local_identifier_names = self.current_glsl_local_identifier_names
        previous_synthetic_local_names = self.current_glsl_synthetic_local_names
        previous_scalar_array_view_plans = getattr(
            self, "current_glsl_scalar_array_view_plans", {}
        )
        previous_scalar_array_views = getattr(
            self, "current_glsl_scalar_array_views", {}
        )
        previous_resource_call_semantic_locations = getattr(
            self,
            "current_glsl_resource_function_call_semantic_locations",
            {},
        )
        previous_tail_recursive_call_ids = getattr(
            self, "current_glsl_tail_recursive_call_ids", set()
        )
        previous_type_aliases = self.current_type_aliases
        previous_generic_function_substitutions = (
            self.current_generic_function_substitutions
        )
        previous_structured_buffer_array_parameters = (
            self.current_structured_buffer_array_parameters
        )
        previous_structured_buffer_counter_parameters = (
            self.current_structured_buffer_counter_parameters
        )
        previous_structured_buffer_access_parameters = (
            self.current_structured_buffer_access_parameters
        )
        self.local_variable_types = {}
        self.local_variable_source_types = {}
        self.current_glsl_precise_return_type = None
        self.current_glsl_ieee_operator_arithmetic = (
            self.glsl_is_metal_free_arithmetic_operator(func)
        )
        self.current_glsl_exactness_test_local_ids = self.glsl_exactness_test_local_ids(
            func
        )
        self.current_glsl_precise_function = self.glsl_has_precise_qualifier(func)
        self.current_glsl_precise_captures = (
            [] if self.current_glsl_precise_function else None
        )
        self.current_glsl_precise_value_writes = (
            self.glsl_precise_value_writes(getattr(func, "body", None))
            if self.current_glsl_precise_function
            and getattr(func, "body", None) is not None
            else {}
        )
        precise_parameter_copies = []
        written_value_names = (
            self.glsl_written_value_names(getattr(func, "body", None))
            if self.current_glsl_precise_function
            and shader_type is None
            and getattr(func, "body", None) is not None
            else set()
        )
        self.current_identifier_aliases = dict(self.current_identifier_aliases)
        self.prepare_glsl_local_identifier_names(func)
        self.current_glsl_synthetic_local_names = set(
            self.current_glsl_local_identifier_names.values()
        )
        self.current_type_aliases = dict(self.global_type_aliases)
        self.current_generic_function_substitutions = (
            getattr(func, "_generic_substitutions", {}) or {}
        )
        self.current_glsl_scalar_array_views = {}
        self.current_glsl_scalar_array_view_plans = self.glsl_scalar_array_view_locals(
            func
        )
        self.current_glsl_tail_recursive_call_ids = (
            self.glsl_tail_recursive_void_call_ids(func)
        )
        self.current_structured_buffer_array_parameters = {}
        self.current_structured_buffer_counter_parameters = {}
        self.current_structured_buffer_access_parameters = {}
        self.current_identifier_aliases.update(stage_entry_resource_aliases)
        self.current_identifier_aliases.update(stage_entry_constant_value_aliases)
        for alias_name, binding in resource_aliases.items():
            self.local_variable_types[alias_name] = binding.get("type")
        for alias_name, binding in workgroup_pointer_aliases.items():
            pointer_type = f"{binding.get('element_type', 'float')}*"
            self.local_variable_types[alias_name] = pointer_type
            source_element = binding.get("source_element_type")
            self.local_variable_source_types[alias_name] = (
                f"{source_element}*"
                if self.glsl_narrow_float_width(source_element) is not None
                else pointer_type
            )
        for alias_name, binding in storage_pointer_aliases.items():
            if binding.get("resource_root"):
                continue
            pointer_type = f"{binding.get('element_type', 'float')}*"
            self.local_variable_types[alias_name] = pointer_type
            source_element = binding.get("source_element_type")
            self.local_variable_source_types[alias_name] = (
                f"{source_element}*"
                if self.glsl_narrow_float_width(source_element) is not None
                else pointer_type
            )
        for index, p in enumerate(param_list):
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
            parameter_expression_type = (
                self.stage_entry_parameter_expression_type(p, raw_param_type)
                if shader_type is not None
                else self.type_name_string(raw_param_type)
            )
            metadata_param_type = (
                self.stage_entry_resource_parameter_type(p)
                if shader_type is not None
                else None
            )
            if metadata_param_type is None:
                metadata_param_type = raw_param_type
            self.local_variable_types[p.name] = parameter_expression_type
            self.local_variable_source_types[p.name] = self.type_name_string(
                raw_param_type
            )
            stage_resource_alias = stage_entry_resource_aliases.get(p.name)
            if stage_resource_alias:
                self.local_variable_types[stage_resource_alias] = (
                    parameter_expression_type
                )
                self.local_variable_source_types[stage_resource_alias] = (
                    self.type_name_string(raw_param_type)
                )
            stage_constant_value_alias = stage_entry_constant_value_aliases.get(p.name)
            if stage_constant_value_alias:
                self.local_variable_types[stage_constant_value_alias] = (
                    parameter_expression_type
                )
                self.local_variable_source_types[stage_constant_value_alias] = (
                    self.type_name_string(raw_param_type)
                )
            self.validate_resource_access_metadata_operands(p)
            self.record_structured_buffer_access_metadata(
                p.name, metadata_param_type, p, parameter=True
            )
            if stage_resource_alias:
                self.record_structured_buffer_access_metadata(
                    stage_resource_alias, metadata_param_type, p, parameter=True
                )

            if index in unsupported_buffer_array_indices:
                continue

            if self.is_sampler_type(raw_param_type):
                self.explicit_resource_binding_index(p)
                sampler_parameters.add(p.name)
                if not self.should_preserve_sampler_parameter(
                    getattr(func, "name", None), p.name
                ):
                    continue

            buffer_array = self.structured_buffer_array_parameter_info(
                raw_param_type, p.name, getattr(func, "name", None)
            )
            if buffer_array is not None:
                expanded_names = [
                    self.glsl_synthetic_local_identifier(f"{p.name}_{index}")
                    for index in range(buffer_array["count"])
                ]
                buffer_info = {
                    **buffer_array,
                    "source_name": p.name,
                    "expanded_names": expanded_names,
                }
                if self.structured_buffer_requires_counter(buffer_array["base_type"]):
                    counter_names = [
                        self.glsl_synthetic_local_identifier(f"{p.name}Counter_{index}")
                        for index in range(buffer_array["count"])
                    ]
                    buffer_info["counter_expanded_names"] = counter_names
                self.current_structured_buffer_array_parameters[p.name] = buffer_info
                for expanded_name in expanded_names:
                    self.local_variable_types[expanded_name] = (
                        f"{buffer_array['element_type']}[]"
                    )
                    params.append(
                        format_c_style_array_declaration(
                            f"{buffer_array['element_type']}[]",
                            expanded_name,
                        )
                    )
                for counter_name in buffer_info.get("counter_expanded_names", []):
                    self.local_variable_types[counter_name] = "uint[]"
                    self.local_variable_source_types[counter_name] = "uint[]"
                    params.append(
                        format_c_style_array_declaration("uint[]", counter_name)
                    )
                continue

            param_type = self.glsl_private_pointer_parameter_array_type(p, function_key)
            if param_type is None:
                param_type = self.map_resource_parameter_type_with_hint(
                    raw_param_type, p, getattr(func, "name", None)
                )
            if self.is_opaque_resource_type(
                self.map_resource_type_with_format(
                    self.resource_base_type(raw_param_type), p
                )
            ):
                texture_parameters[p.name] = self.map_resource_type_with_format(
                    self.resource_base_type(raw_param_type), p
                )
                record_explicit_image_metadata(
                    p.name,
                    p,
                    self.attribute_value_to_string,
                    image_formats=image_format_parameters,
                    image_accesses=image_access_parameters,
                )

            self.semantic_from_node(p)

            parameter_name = (
                self.glsl_parameter_identifier_name(p.name, parameter_source_names)
                if shader_type is None
                else p.name
            )
            declaration = format_c_style_array_declaration(param_type, parameter_name)
            if self.is_storage_image_type(param_type):
                memory_qualifiers = self.resource_memory_qualifiers(p)
                if memory_qualifiers:
                    declaration = f"{memory_qualifiers} {declaration}"
            parameter_qualifiers = self.glsl_parameter_qualifiers(p)
            precise_parameter = self.glsl_precise_parameter_form(
                p,
                raw_param_type,
                param_type,
                parameter_qualifiers,
                shader_type,
                written_value_names,
            )
            if precise_parameter == "copy":
                input_name = self.glsl_precise_parameter_input_name(parameter_name)
                precise_parameter_copies.append(
                    (param_type, parameter_name, input_name)
                )
                declaration = format_c_style_array_declaration(param_type, input_name)
            if parameter_qualifiers:
                declaration = f"{' '.join(parameter_qualifiers)} {declaration}"
            if precise_parameter == "qualifier":
                declaration = f"precise {declaration}"
            params.append(declaration)
            if is_private_pointer_parameter(p):
                private_pointer_shape = self.current_glsl_private_pointer_variant.get(
                    p.name
                )
                if private_pointer_shape in {0, "0"}:
                    continue
                base_name = self.current_glsl_private_pointer_base_names.get(p.name)
                if base_name is None:
                    base_name = self.function_private_pointer_base_names.get(
                        function_key, {}
                    ).get(p.name)
                if base_name is not None:
                    params.append(f"int {base_name}")
                    self.local_variable_types[base_name] = "int"
                    self.local_variable_source_types[base_name] = "int"
            if self.structured_buffer_requires_counter(raw_param_type):
                counter_name = self.structured_buffer_counter_parameter_name(p.name)
                self.current_structured_buffer_counter_parameters[p.name] = counter_name
                self.local_variable_types[counter_name] = "uint[]"
                self.local_variable_source_types[counter_name] = "uint[]"
                params.append(format_c_style_array_declaration("uint[]", counter_name))

        if shader_type == "compute":
            self.validate_compute_builtin_parameter_types(param_list)
        elif shader_type is not None:
            self.validate_graphics_builtin_parameter_types(param_list, shader_type)

        params_str = ", ".join(params)
        declaration_name = self.glsl_function_declaration_name(func)
        if getattr(func, "body", []) is None and shader_type is None:
            return_type = self.map_type(
                self.type_name_string(getattr(func, "return_type", None)) or "void"
            )
            self.current_function_return_type = previous_function_return_type
            self.current_glsl_precise_return_type = previous_precise_return_type
            self.current_glsl_precise_function = previous_precise_function
            self.current_glsl_precise_captures = previous_precise_captures
            self.current_glsl_ieee_operator_arithmetic = (
                previous_ieee_operator_arithmetic
            )
            self.current_glsl_exactness_test_local_ids = (
                previous_exactness_test_local_ids
            )
            self.current_glsl_precise_value_writes = previous_precise_value_writes
            self.local_variable_types = previous_local_variable_types
            self.local_variable_source_types = previous_local_variable_source_types
            self.current_identifier_aliases = previous_identifier_aliases
            self.current_glsl_local_identifier_names = previous_local_identifier_names
            self.current_glsl_synthetic_local_names = previous_synthetic_local_names
            self.current_glsl_scalar_array_view_plans = previous_scalar_array_view_plans
            self.current_glsl_scalar_array_views = previous_scalar_array_views
            self.current_glsl_tail_recursive_call_ids = previous_tail_recursive_call_ids
            self.current_type_aliases = previous_type_aliases
            self.current_generic_function_substitutions = (
                previous_generic_function_substitutions
            )
            self.current_structured_buffer_array_parameters = (
                previous_structured_buffer_array_parameters
            )
            self.current_structured_buffer_counter_parameters = (
                previous_structured_buffer_counter_parameters
            )
            self.current_structured_buffer_access_parameters = (
                previous_structured_buffer_access_parameters
            )
            self.current_glsl_private_pointer_function_name = (
                previous_private_pointer_function_name
            )
            return f"{code}{return_type} {declaration_name}({params_str});\n\n"

        stage_entry_types = {
            "vertex",
            "fragment",
            "compute",
            "geometry",
            "tessellation_control",
            "tessellation_evaluation",
            "mesh",
            "task",
            "amplification",
            "object",
            "ray_generation",
            "ray_intersection",
            "ray_closest_hit",
            "ray_any_hit",
            "ray_miss",
            "ray_callable",
        }
        previous_glsl_function_key = self.current_glsl_function_key
        previous_glsl_source_function_id = self.current_glsl_source_function_id
        previous_glsl_source_function_name = self.current_glsl_source_function_name
        emitted_function_name = (
            (entry_name or "main")
            if shader_type in stage_entry_types
            else declaration_name
        )
        self.register_glsl_function_graph_scope(
            func,
            stage_context=stage_context,
        )
        self.current_glsl_function_key = self.glsl_function_graph_key(
            func,
            emitted_function_name,
            stage_context=stage_context,
        )
        self.current_glsl_source_function_id = (
            getattr(func, "_glsl_resource_source_function_id", None)
            or getattr(func, "_generic_source_function_id", None)
            or id(func)
        )
        self.current_glsl_source_function_name = (
            getattr(func, "_glsl_resource_source_name", None)
            or getattr(func, "_generic_source_name", None)
            or getattr(func, "name", None)
        )
        self.glsl_function_call_graph.setdefault(
            self.current_glsl_function_key,
            set(),
        )

        if shader_type is not None:
            self.validate_function_return_semantic(func, shader_type)
            if shader_type == "tessellation_control":
                self.validate_glsl_tessellation_control_entry_barriers(func)
        elif stage_context == "tessellation_control":
            barrier_name = self.glsl_tessellation_control_first_native_barrier(
                getattr(func, "body", [])
            )
            if barrier_name is not None:
                raise OpenGLRefusal(
                    "GLSL tessellation_control barrier() may only be emitted "
                    "directly in main(); helper function "
                    f"'{func.name}' reaches '{barrier_name}'"
                )

        mesh_output_parameters = self.mesh_output_parameter_infos(
            func, shader_type, stage_layout_qualifiers
        )
        mesh_output_topology = (
            self.glsl_mesh_stage_output_topology_name(func, stage_layout_qualifiers)
            if shader_type == "mesh"
            else None
        )
        stage_output = self.stage_return_output(func, shader_type)
        if stage_output and stage_output["declaration"]:
            code += f"{stage_output['declaration']}\n"
        if mesh_output_parameters:
            code += self.generate_mesh_output_parameter_declarations(
                mesh_output_parameters
            )
        if shader_type == "mesh":
            code += self.generate_glsl_mesh_payload_parameter_declaration(func)

        stage_layout = self.generate_stage_layout(
            shader_type, func, stage_layout_qualifiers
        )
        if stage_layout:
            code += stage_layout

        is_native_stage_entry = (entry_name or "main") == "main"
        if shader_type in {"compute", "mesh", "task"} and is_native_stage_entry:
            code += self.generate_compute_layout(execution_config)
            code += compute_derivative_group_layout(
                self, shader_type, stage_layout_qualifiers, execution_config
            )

        if shader_type in stage_entry_types:
            code += f"void {entry_name or 'main'}() {{\n"
            interlock_body_start = len(code)
            self.current_function_return_type = "void"
            self.current_glsl_precise_return_type = None
            if shader_type == "compute" and is_native_stage_entry:
                code += self.generate_glsl_exact_subgroup_width_guard(func, shader_type)
        else:
            raw_return_type = self.type_name_string(getattr(func, "return_type", None))
            self.current_function_return_type = raw_return_type or "void"
            return_type = self.map_type(self.current_function_return_type)
            self.current_glsl_precise_return_type = self.glsl_precise_return_local_type(
                func, return_type
            )
            code += f"{return_type} {declaration_name}({params_str}) {{\n"
            interlock_body_start = len(code)

        previous_sampler_parameters = self.current_sampler_parameters
        previous_texture_parameters = self.current_texture_parameters
        previous_resource_aliases = self.current_resource_aliases
        previous_workgroup_pointer_aliases = self.current_glsl_workgroup_pointer_aliases
        previous_storage_pointer_aliases = self.current_glsl_storage_pointer_aliases
        previous_workgroup_pointer_scope_depth = self.glsl_workgroup_pointer_scope_depth
        previous_image_format_parameters = self.current_image_format_parameters
        previous_image_access_parameters = self.current_image_access_parameters
        previous_stage_output = self.current_stage_output
        previous_stage_inputs = self.current_stage_inputs
        previous_stage_outputs = self.current_stage_outputs
        previous_stage_output_member_map = self.current_stage_output_member_map
        previous_stage_parameter_aliases = self.current_stage_parameter_aliases
        previous_mesh_output_parameters = self.current_mesh_output_parameters
        previous_mesh_output_topology = self.current_mesh_output_topology
        previous_mesh_output_count_limits = self.current_mesh_output_count_limits
        previous_flattened_stage_variables = self.flattened_stage_variables
        previous_stage_return_type = self.current_stage_return_type
        previous_stage_entry_type = self.current_stage_entry_type
        previous_local_variable_types = self.local_variable_types
        previous_compile_time_int_constants = self.current_compile_time_int_constants
        previous_compile_time_int_vector_constants = (
            self.current_compile_time_int_vector_constants
        )
        previous_compile_time_int_vector_array_constants = (
            self.current_compile_time_int_vector_array_constants
        )
        self.current_sampler_parameters = sampler_parameters
        self.current_texture_parameters = {
            **texture_parameters,
            **{
                alias_name: binding.get("type")
                for alias_name, binding in resource_aliases.items()
            },
        }
        self.current_resource_aliases = resource_aliases
        self.current_glsl_workgroup_pointer_aliases = (
            self.glsl_function_workgroup_pointer_aliases(
                func,
                {
                    **self.glsl_global_workgroup_pointer_aliases,
                    **workgroup_pointer_aliases,
                },
            )
        )
        self.current_glsl_storage_pointer_aliases = {
            name: dict(binding) for name, binding in storage_pointer_aliases.items()
        }
        previous_emitted_function = getattr(self, "current_glsl_emitted_function", None)
        self.current_glsl_emitted_function = func
        self.current_glsl_synthetic_local_names = set(
            self.collect_glsl_function_local_identifier_names(func)
        )
        self.current_glsl_synthetic_local_names.update(
            self.current_glsl_local_identifier_names.values()
        )
        self.current_glsl_synthetic_local_names.update(
            input_name for _, _, input_name in precise_parameter_copies
        )
        self.glsl_workgroup_pointer_scope_depth = 0
        self.current_image_format_parameters = {
            **image_format_parameters,
            **{
                alias_name: binding.get("format")
                for alias_name, binding in resource_aliases.items()
                if binding.get("format") is not None
            },
        }
        self.current_image_access_parameters = {
            **image_access_parameters,
            **{
                alias_name: binding.get("access")
                for alias_name, binding in resource_aliases.items()
                if binding.get("access") is not None
            },
        }
        self.current_stage_output = stage_output
        self.current_stage_inputs = self.stage_input_member_maps(func, shader_type)
        self.current_stage_output_member_map = self.stage_output_member_map(
            func, shader_type
        )
        self.current_stage_outputs = self.stage_output_member_maps(func, shader_type)
        self.current_stage_parameter_aliases = self.stage_parameter_aliases(
            func, shader_type
        )
        self.apply_stage_context_entry_aliases(func, shader_type, stage_context)
        self.current_mesh_output_parameters = mesh_output_parameters
        self.current_mesh_output_topology = mesh_output_topology
        self.current_mesh_output_count_limits = (
            self.glsl_mesh_output_count_limits(func, stage_layout_qualifiers)
            if shader_type == "mesh"
            else {}
        )
        self.flattened_stage_variables = set(self.current_stage_outputs)
        self.current_stage_return_type = self.type_node_name(
            getattr(func, "return_type", None)
        )
        self.current_stage_entry_type = shader_type or stage_context
        self.current_identifier_aliases = dict(self.current_identifier_aliases)
        self.current_identifier_aliases.update(
            self.fragment_legacy_color_output_aliases(func, shader_type, stage_output)
        )
        self.current_compile_time_int_constants = dict(self.literal_int_constants)
        self.current_compile_time_int_vector_constants = dict(
            self.literal_int_vector_constants
        )
        self.current_compile_time_int_vector_array_constants = dict(
            self.literal_int_vector_array_constants
        )
        for param in param_list:
            name = getattr(param, "name", None)
            self.current_compile_time_int_constants.pop(name, None)
            self.current_compile_time_int_vector_constants.pop(name, None)
            self.current_compile_time_int_vector_array_constants.pop(name, None)
        body = getattr(func, "body", [])
        self.current_glsl_resource_function_call_semantic_locations = (
            self.glsl_resource_function_call_semantic_locations(func)
        )
        for copy_type, copy_name, input_name in precise_parameter_copies:
            declaration = format_c_style_array_declaration(copy_type, copy_name)
            code += f"    precise {declaration} = {input_name};\n"
        precise_capture_insertion_index = len(code)
        code += self.generate_glsl_mutable_storage_pointer_root_declarations(func, 1)
        for declaration in self.stage_builtin_parameter_local_alias_declarations(
            func, shader_type
        ):
            code += f"    {declaration}\n"
        for declaration in stage_input_struct_value_declarations(self, func):
            code += f"    {declaration}\n"
        if shader_type in stage_entry_types:
            code += self.generate_glsl_hlsl_cbuffer_loads(1)
            code += self.generate_glsl_runtime_global_initializers(func, 1)
        simd_placeholder_body = (
            self.glsl_unsupported_metal_simd_placeholder_function_body(func, 1)
        )
        if simd_placeholder_body is not None:
            code += simd_placeholder_body
        elif unsupported_buffer_array_info:
            code += self.unsupported_structured_buffer_array_function_body(
                func, unsupported_buffer_array_info, 1
            )
        elif self.current_glsl_tail_recursive_call_ids:
            code += "    while (true) {\n"
            if hasattr(body, "statements"):
                for stmt in body.statements:
                    code += self.generate_statement(stmt, 2)
            elif isinstance(body, list):
                for stmt in body:
                    code += self.generate_statement(stmt, 2)
            code += "        break;\n"
            code += "    }\n"
        elif hasattr(body, "statements"):
            for stmt in body.statements:
                code += self.generate_statement(stmt, 1)
        elif isinstance(body, list):
            for stmt in body:
                code += self.generate_statement(stmt, 1)
        if self.current_glsl_precise_captures:
            declarations = "".join(
                f"    precise {format_c_style_array_declaration(capture_type, name)};\n"
                for capture_type, name in self.current_glsl_precise_captures
            )
            code = (
                code[:precise_capture_insertion_index]
                + declarations
                + code[precise_capture_insertion_index:]
            )
        self.current_glsl_precise_captures = previous_precise_captures
        self.current_sampler_parameters = previous_sampler_parameters
        self.current_texture_parameters = previous_texture_parameters
        self.current_resource_aliases = previous_resource_aliases
        self.current_glsl_workgroup_pointer_aliases = previous_workgroup_pointer_aliases
        self.current_glsl_storage_pointer_aliases = previous_storage_pointer_aliases
        self.current_glsl_emitted_function = previous_emitted_function
        self.current_glsl_synthetic_local_names = previous_synthetic_local_names
        self.current_glsl_scalar_array_view_plans = previous_scalar_array_view_plans
        self.current_glsl_scalar_array_views = previous_scalar_array_views
        self.current_glsl_resource_function_call_semantic_locations = (
            previous_resource_call_semantic_locations
        )
        self.current_glsl_tail_recursive_call_ids = previous_tail_recursive_call_ids
        self.glsl_workgroup_pointer_scope_depth = previous_workgroup_pointer_scope_depth
        self.current_image_format_parameters = previous_image_format_parameters
        self.current_image_access_parameters = previous_image_access_parameters
        self.current_stage_output = previous_stage_output
        self.current_stage_inputs = previous_stage_inputs
        self.current_stage_outputs = previous_stage_outputs
        self.current_stage_output_member_map = previous_stage_output_member_map
        self.current_stage_parameter_aliases = previous_stage_parameter_aliases
        self.current_mesh_output_parameters = previous_mesh_output_parameters
        self.current_mesh_output_topology = previous_mesh_output_topology
        self.current_mesh_output_count_limits = previous_mesh_output_count_limits
        self.flattened_stage_variables = previous_flattened_stage_variables
        self.current_stage_return_type = previous_stage_return_type
        self.current_stage_entry_type = previous_stage_entry_type
        self.local_variable_types = previous_local_variable_types
        self.current_identifier_aliases = previous_identifier_aliases
        self.current_glsl_local_identifier_names = previous_local_identifier_names
        self.current_type_aliases = previous_type_aliases
        self.current_compile_time_int_constants = previous_compile_time_int_constants
        self.current_compile_time_int_vector_constants = (
            previous_compile_time_int_vector_constants
        )
        self.current_compile_time_int_vector_array_constants = (
            previous_compile_time_int_vector_array_constants
        )
        self.current_function_return_type = previous_function_return_type
        self.current_glsl_precise_return_type = previous_precise_return_type
        self.current_glsl_precise_function = previous_precise_function
        self.current_glsl_ieee_operator_arithmetic = previous_ieee_operator_arithmetic
        self.current_glsl_exactness_test_local_ids = previous_exactness_test_local_ids
        self.current_glsl_precise_value_writes = previous_precise_value_writes
        self.local_variable_types = previous_local_variable_types
        self.local_variable_source_types = previous_local_variable_source_types
        self.current_generic_function_substitutions = (
            previous_generic_function_substitutions
        )
        self.current_structured_buffer_array_parameters = (
            previous_structured_buffer_array_parameters
        )
        self.current_structured_buffer_counter_parameters = (
            previous_structured_buffer_counter_parameters
        )
        self.current_structured_buffer_access_parameters = (
            previous_structured_buffer_access_parameters
        )
        self.current_glsl_function_key = previous_glsl_function_key
        self.current_glsl_source_function_id = previous_glsl_source_function_id
        self.current_glsl_source_function_name = previous_glsl_source_function_name
        self.current_glsl_private_pointer_function_name = (
            previous_private_pointer_function_name
        )

        code = wrap_fragment_interlock(
            self, shader_type, entry_name, code, interlock_body_start
        )
        code += "}\n\n"
        return code

    def generate_statement(self, stmt, indent=0):
        """Render a single CrossGL AST statement as GLSL source."""
        if getattr(stmt, "_glsl_checked_unreachable", False):
            return ""
        indent_str = "    " * indent

        if isinstance(stmt, VariableNode):
            if self.is_glsl_type_alias_declaration(stmt):
                self.current_type_aliases[stmt.name] = getattr(stmt, "var_type", None)
                return ""
            if stmt.name in self.flattened_stage_variables:
                if getattr(stmt, "initial_value", None) is None:
                    return ""
                return render_flattened_stage_output_store(
                    self, stmt.name, stmt.initial_value, indent
                )
            scalar_array_view = self.generate_glsl_scalar_array_view_declaration(
                stmt, indent
            )
            if scalar_array_view is not None:
                return scalar_array_view
            if (
                getattr(stmt, "_glsl_storage_pointer_array_binding", None) is not None
                or self.glsl_storage_pointer_array_contract(stmt) is not None
            ):
                self.current_identifier_aliases.pop(stmt.name, None)
                self.current_glsl_storage_pointer_aliases.pop(stmt.name, None)
                return self.generate_glsl_storage_pointer_array_alias_declaration(
                    stmt, indent
                )
            if self.glsl_workgroup_pointer_declaration(
                stmt, self.glsl_workgroup_pointer_aliases()
            ):
                self.current_identifier_aliases.pop(stmt.name, None)
                self.current_glsl_workgroup_pointer_aliases.pop(stmt.name, None)
                return self.generate_glsl_workgroup_pointer_alias_declaration(
                    stmt, indent
                )
            if self.glsl_storage_pointer_declaration(
                stmt, self.glsl_storage_pointer_aliases()
            ):
                self.current_identifier_aliases.pop(stmt.name, None)
                self.current_glsl_storage_pointer_aliases.pop(stmt.name, None)
                return self.generate_glsl_storage_pointer_alias_declaration(
                    stmt, indent
                )
            if self.glsl_hoisted_shared_alias_for_declaration(stmt) is not None:
                # Workgroup-shared locals are emitted once at global scope; here
                # we only register the alias so references resolve and drop the
                # private local declaration entirely.
                self.current_identifier_aliases.pop(stmt.name, None)
                self.current_glsl_workgroup_pointer_aliases.pop(stmt.name, None)
                self.activate_glsl_hoisted_shared_declaration(stmt)
                return ""
            raw_local_type = getattr(
                stmt,
                "var_type",
                getattr(stmt, "vtype", None),
            )
            if isinstance(raw_local_type, (PointerType, ReferenceType)):
                alias_kind = (
                    "pointer"
                    if isinstance(raw_local_type, PointerType)
                    else "reference"
                )
                raise OpenGLPrivatePointerParameterError(
                    f"OpenGL cannot emit local private {alias_kind} "
                    f"'{stmt.name}' without a concrete array-view lowering",
                    function_name=getattr(
                        self,
                        "current_glsl_source_function_name",
                        None,
                    ),
                    parameter_name=stmt.name,
                    reason=f"unsupported-local-{alias_kind}",
                    source_location=getattr(stmt, "source_location", None),
                )
            var_type = self.local_variable_declared_type(stmt)
            stage_output_alias = self.current_identifier_aliases.get(stmt.name)
            if (
                self.current_stage_output is not None
                and stage_output_alias == self.current_stage_output["name"]
            ):
                self.local_variable_types[stmt.name] = var_type
                self.local_variable_source_types[stmt.name] = self.type_name_string(
                    var_type
                )
                initial_value = getattr(stmt, "initial_value", None)
                if initial_value is None:
                    return ""
                init_expr = self.generate_expression_with_expected(
                    initial_value, var_type
                )
                return f"{indent_str}{stage_output_alias} = {init_expr};\n"
            if (
                self.current_stage_output is not None
                and stmt.name == self.current_stage_output["name"]
            ):
                initial_value = getattr(stmt, "initial_value", None)
                if initial_value is None:
                    return ""
                init_expr = self.generate_expression_with_expected(
                    initial_value, var_type
                )
                return f"{indent_str}{stmt.name} = {init_expr};\n"
            self.current_identifier_aliases.pop(stmt.name, None)
            self.current_glsl_workgroup_pointer_aliases.pop(stmt.name, None)
            self.current_glsl_storage_pointer_aliases.pop(stmt.name, None)
            local_name = self.glsl_local_identifier_name(stmt.name)
            self.local_variable_types[stmt.name] = var_type
            self.local_variable_source_types[stmt.name] = self.type_name_string(
                var_type
            )
            self.update_compile_time_offset_constants(stmt, var_type)

            declared_type_node = getattr(stmt, "var_type", None)
            mapped_var_type = (
                declared_type_node
                if isinstance(declared_type_node, CooperativeMatrixType)
                else var_type
            )
            declaration = format_c_style_array_declaration(
                self.map_type(mapped_var_type), local_name
            )
            declaration = f"{self.local_variable_qualifier(stmt)}{declaration}"
            initial_value = getattr(stmt, "initial_value", None)
            if initial_value is not None and self.type_contains_opaque_resource(
                var_type
            ):
                self.current_identifier_aliases[stmt.name] = (
                    self.generate_expression_with_expected(initial_value, var_type)
                )
                return ""
            if isinstance(initial_value, MatchNode):
                code = f"{indent_str}{declaration};\n"
                code += generate_match_expression_assignment(
                    self,
                    initial_value,
                    local_name,
                    var_type,
                    indent,
                    "GLSL",
                )
                return code
            if initial_value is not None:
                storage_aggregate = (
                    self.glsl_storage_struct_reinterpret_materialization(
                        initial_value,
                        var_type,
                    )
                )
                if storage_aggregate is not None:
                    return f"{indent_str}{declaration} = {storage_aggregate};\n"
                dynamic_init = (
                    self.generate_glsl_dynamic_resource_call_assignment_statement(
                        initial_value,
                        local_name,
                        var_type,
                        indent,
                    )
                )
                if dynamic_init is not None and "const" not in (
                    self.local_variable_qualifier(stmt).split()
                ):
                    return f"{indent_str}{declaration};\n{dynamic_init}"
                compile_time_initializer = (
                    self.glsl_narrow_compile_time_integer_initializer(stmt, var_type)
                )
                init_expr = compile_time_initializer or (
                    self.generate_expression_with_expected(initial_value, var_type)
                )
                return f"{indent_str}{declaration} = {init_expr};\n"
            else:
                return f"{indent_str}{declaration};\n"
        elif isinstance(stmt, ArrayNode):
            return self.generate_array_declaration(stmt, indent)
        elif isinstance(stmt, AssignmentNode):
            storage_pointer_array_assignment = (
                self.generate_glsl_storage_pointer_array_assignment(stmt, indent)
            )
            if storage_pointer_array_assignment is not None:
                return storage_pointer_array_assignment
            workgroup_pointer_assignment = (
                self.generate_glsl_workgroup_pointer_assignment(stmt, indent)
            )
            if workgroup_pointer_assignment is not None:
                return workgroup_pointer_assignment
            storage_pointer_assignment = self.generate_glsl_storage_pointer_assignment(
                stmt, indent
            )
            if storage_pointer_assignment is not None:
                return storage_pointer_assignment
            mesh_assignment = self.generate_mesh_output_assignment_statement(
                stmt, indent
            )
            if mesh_assignment is not None:
                return mesh_assignment
            dynamic_assignment = self.generate_glsl_dynamic_resource_assignment_node(
                stmt,
                indent,
            )
            if dynamic_assignment is not None:
                return dynamic_assignment
            return self.generate_glsl_statement_code(
                self.generate_assignment(stmt, statement_context=True), indent
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
            if getattr(stmt, "value", None) is None:
                return f"{indent_str}return;\n"
            dynamic_resource_return = (
                self.generate_glsl_dynamic_resource_call_return_statement(
                    stmt.value,
                    indent,
                )
            )
            if dynamic_resource_return is not None:
                return dynamic_resource_return
            return_value_name = self.expression_name(stmt.value)
            if return_value_name in self.flattened_stage_variables:
                return (
                    distance_output_epilogue(self, indent_str)
                    + f"{indent_str}return;\n"
                )
            if (
                self.current_stage_output is not None
                and return_value_name == self.current_stage_output["name"]
            ):
                return f"{indent_str}return;\n"
            if (
                self.current_stage_output is not None
                and self.current_identifier_aliases.get(return_value_name)
                == self.current_stage_output["name"]
            ):
                return f"{indent_str}return;\n"
            stage_struct_return = self.generate_stage_struct_constructor_return(
                stmt.value, indent
            )
            if stage_struct_return is not None:
                return stage_struct_return
            if self.current_stage_output is not None:
                if isinstance(stmt.value, list):
                    values = ", ".join(
                        self.generate_expression(val) for val in stmt.value
                    )
                    value = values
                else:
                    # The entry is void in GLSL; its value converts to the output's type, as the source's
                    # return converted it to the return type (an HLSL float returned as float4 splats).
                    value = self.generate_expression_with_expected(
                        stmt.value,
                        self.current_stage_output.get("type")
                        or self.current_function_return_type,
                    )
                return (
                    f"{indent_str}{self.current_stage_output['name']} = {value};\n"
                    f"{indent_str}return;\n"
                )
            if self.current_function_return_type == "void":
                struct_value_return = render_stage_struct_value_return(
                    self, stmt.value, indent
                )
                if struct_value_return is not None:
                    return (
                        struct_value_return
                        + distance_output_epilogue(self, indent_str)
                        + f"{indent_str}return;\n"
                    )
                refuse_unwritten_entry_return(self, stmt.value)
                value = self.generate_expression_with_expected(stmt.value, None)
                return (
                    self.generate_glsl_statement_code(value, indent)
                    + f"{indent_str}return;\n"
                )
            if isinstance(stmt.value, list):
                value = ", ".join(self.generate_expression(val) for val in stmt.value)
            else:
                value = self.generate_expression_with_expected(
                    stmt.value, self.current_function_return_type
                )
            return self.generate_glsl_return_value_statement(value, indent)
        elif hasattr(stmt, "__class__") and "ExpressionStatementNode" in str(
            type(stmt)
        ):
            tail_return = self.generate_tail_expression_statement(stmt, indent)
            if tail_return is not None:
                return tail_return
            mesh_assignment = self.generate_mesh_output_assignment_statement(
                getattr(stmt, "expression", None), indent
            )
            if mesh_assignment is not None:
                return mesh_assignment
            mesh_intrinsic = self.generate_mesh_intrinsic_statement(
                getattr(stmt, "expression", None), indent
            )
            if mesh_intrinsic is not None:
                return mesh_intrinsic
            mesh_helper = self.generate_mesh_output_helper_call_statement(
                getattr(stmt, "expression", None), indent
            )
            if mesh_helper is not None:
                return mesh_helper
            expression = getattr(stmt, "expression", None)
            flattened_output = flattened_stage_output_assignment(self, expression)
            if flattened_output is not None:
                return render_flattened_stage_output_store(
                    self, flattened_output, expression.value, indent
                )
            if isinstance(expression, AssignmentNode):
                storage_pointer_array_assignment = (
                    self.generate_glsl_storage_pointer_array_assignment(
                        expression,
                        indent,
                    )
                )
                if storage_pointer_array_assignment is not None:
                    return storage_pointer_array_assignment
                workgroup_pointer_assignment = (
                    self.generate_glsl_workgroup_pointer_assignment(
                        expression,
                        indent,
                    )
                )
                if workgroup_pointer_assignment is not None:
                    return workgroup_pointer_assignment
                storage_pointer_assignment = (
                    self.generate_glsl_storage_pointer_assignment(
                        expression,
                        indent,
                    )
                )
                if storage_pointer_assignment is not None:
                    return storage_pointer_assignment
                dynamic_assignment = (
                    self.generate_glsl_dynamic_resource_assignment_node(
                        expression,
                        indent,
                    )
                )
                if dynamic_assignment is not None:
                    return dynamic_assignment
                return self.generate_glsl_statement_code(
                    self.generate_assignment(expression, statement_context=True),
                    indent,
                )
            tail_recursive_call = self.generate_glsl_tail_recursive_call_statement(
                expression,
                indent,
            )
            if tail_recursive_call is not None:
                return tail_recursive_call
            dynamic_resource_statement = (
                self.generate_glsl_dynamic_resource_call_expression_statement(
                    expression,
                    indent,
                )
            )
            if dynamic_resource_statement is not None:
                return dynamic_resource_statement
            subgroup_barrier = self.generate_glsl_software_subgroup_barrier_statement(
                expression, indent
            )
            if subgroup_barrier is not None:
                return subgroup_barrier
            clip_statement = self.generate_clip_statement(expression, indent)
            if clip_statement is not None:
                return clip_statement
            expr_code = self.generate_expression_statement(stmt)
            return f"{indent_str}{expr_code};\n"
        else:
            tail_recursive_call = self.generate_glsl_tail_recursive_call_statement(
                stmt,
                indent,
            )
            if tail_recursive_call is not None:
                return tail_recursive_call
            mesh_intrinsic = self.generate_mesh_intrinsic_statement(stmt, indent)
            if mesh_intrinsic is not None:
                return mesh_intrinsic
            mesh_helper = self.generate_mesh_output_helper_call_statement(stmt, indent)
            if mesh_helper is not None:
                return mesh_helper
            dynamic_resource_statement = (
                self.generate_glsl_dynamic_resource_call_expression_statement(
                    stmt,
                    indent,
                )
            )
            if dynamic_resource_statement is not None:
                return dynamic_resource_statement
            subgroup_barrier = self.generate_glsl_software_subgroup_barrier_statement(
                stmt, indent
            )
            if subgroup_barrier is not None:
                return subgroup_barrier
            clip_statement = self.generate_clip_statement(stmt, indent)
            if clip_statement is not None:
                return clip_statement
            expr_result = self.generate_expression(stmt)
            if expr_result.strip():
                return f"{indent_str}{expr_result};\n"
            else:
                return f"{indent_str}// Unhandled statement: {type(stmt).__name__}\n"

    def glsl_storage_to_workgroup_byte_array_copy_contract(
        self,
        left_expression,
        right_expression,
        operator,
        *,
        statement_context,
    ):
        destination = self.glsl_storage_struct_reinterpret_node(left_expression)
        source = self.glsl_storage_struct_reinterpret_node(right_expression)
        if destination is None or source is None:
            return None

        destination_pointer_type = getattr(destination, "target_type", None)
        source_pointer_type = getattr(source, "target_type", None)
        destination_address_space = str(
            getattr(destination_pointer_type, "address_space", None) or ""
        ).lower()
        source_address_space = str(
            getattr(source_pointer_type, "address_space", None) or ""
        ).lower()
        if destination_address_space not in {"threadgroup", "workgroup"} or (
            source_address_space not in {"device", "constant", "storage", "global"}
        ):
            return None

        destination_target = self.glsl_normalized_source_type(
            getattr(destination_pointer_type, "pointee_type", None)
        )
        source_target = self.glsl_normalized_source_type(
            getattr(source_pointer_type, "pointee_type", None)
        )
        if (
            not destination_target
            or scalar_storage_layout(destination_target) is not None
            or self.is_vector_value_type(destination_target)
        ):
            return None
        if self.map_type(destination_target) != self.map_type(source_target):
            self.glsl_storage_struct_reinterpret_error(
                source,
                source_type=None,
                target_type=source_target,
                reason="storage-workgroup-aggregate-target-mismatch",
                detail=(
                    f"the destination aggregate target is '{destination_target}', not "
                    f"'{source_target}'"
                ),
            )

        destination_binding = self.glsl_workgroup_pointer_binding(
            destination.expression,
            self.glsl_workgroup_pointer_aliases(),
        )
        if destination_binding is None:
            raise OpenGLWorkgroupPointerError(
                "OpenGL cannot resolve the destination aggregate copy to one "
                "shared-array backing object",
                reason="storage-workgroup-aggregate-backing-unresolved",
                source_location=getattr(destination, "source_location", None),
            )
        source_binding = self.glsl_storage_pointer_binding(
            source.expression,
            self.glsl_storage_pointer_aliases(),
        )
        source_type = self.strip_metal_parameter_indirection(
            self.type_name_string(
                (source_binding or {}).get("source_element_type")
                or (source_binding or {}).get("element_type")
            )
        )
        if source_binding is None or source_binding.get("domain") != "storage-buffer":
            self.glsl_storage_struct_reinterpret_error(
                source,
                source_type=source_type,
                target_type=source_target,
                reason="storage-workgroup-aggregate-backing-unresolved",
                detail="the source pointer has no single proven storage-buffer backing",
            )
        if source_binding.get("pointer_reinterpretation") is not None or str(
            source_binding.get("byte_offset", "0")
        ) not in {"0", "0u"}:
            self.glsl_storage_struct_reinterpret_error(
                source,
                source_type=source_type,
                target_type=source_target,
                reason="unsupported-storage-workgroup-aggregate-source-layout",
                detail="the source pointer is already a reinterpreted byte view",
            )
        if destination_binding.get("pointer_reinterpretation") is not None or str(
            destination_binding.get("byte_offset", "0")
        ) not in {"0", "0u"}:
            self.glsl_storage_struct_reinterpret_error(
                destination,
                source_type=self.glsl_normalized_source_type(
                    destination_binding.get("element_type")
                ),
                target_type=destination_target,
                reason="unsupported-storage-workgroup-aggregate-destination-layout",
                detail="the destination pointer is already a reinterpreted byte view",
            )

        destination_type = self.strip_metal_parameter_indirection(
            self.type_name_string(
                destination_binding.get("source_element_type")
                or destination_binding.get("element_type")
            )
        )
        destination_layout = scalar_storage_layout(destination_type)
        source_layout = scalar_storage_layout(source_type)
        layout = self.glsl_storage_struct_byte_array_layout(
            source_target,
            source,
            source_type,
        )
        if (
            source_layout is None
            or destination_layout is None
            or source_layout.bit_width not in {16, 32}
            or source_layout != destination_layout
        ):
            self.glsl_storage_struct_reinterpret_error(
                source,
                source_type=source_type,
                target_type=source_target,
                reason="storage-workgroup-aggregate-backing-mismatch",
                detail=(
                    "source and destination backings are not the same supported "
                    "16- or 32-bit scalar layout"
                ),
                alignment=layout["alignment"],
            )
        if layout["alignment"] > destination_layout.byte_width:
            self.glsl_storage_struct_reinterpret_error(
                source,
                source_type=source_type,
                target_type=source_target,
                reason="unsupported-storage-workgroup-aggregate-alignment",
                detail=(
                    f"the aggregate requires {layout['alignment']}-byte alignment, "
                    f"but scalar backing elements guarantee only "
                    f"{destination_layout.byte_width}-byte alignment"
                ),
                alignment=layout["alignment"],
            )
        if layout["byte_extent"] % destination_layout.byte_width != 0:
            self.glsl_storage_struct_reinterpret_error(
                source,
                source_type=source_type,
                target_type=source_target,
                reason="unsupported-storage-workgroup-aggregate-extent",
                detail=(
                    f"the {layout['byte_extent']}-byte aggregate does not span a "
                    "whole number of backing scalar elements"
                ),
                alignment=layout["alignment"],
            )
        if not image_access_satisfies_requirement("read", source_binding.get("access")):
            self.glsl_storage_struct_reinterpret_error(
                source,
                source_type=source_type,
                target_type=source_target,
                reason="storage-workgroup-aggregate-source-not-readable",
                detail="the source storage-buffer backing is not readable",
                alignment=layout["alignment"],
            )
        if not bool(getattr(destination_pointer_type, "is_mutable", True)):
            self.glsl_storage_struct_reinterpret_error(
                destination,
                source_type=destination_type,
                target_type=destination_target,
                reason="storage-workgroup-aggregate-destination-read-only",
                detail="the workgroup destination pointer is not mutable",
                access="write",
                alignment=layout["alignment"],
            )
        if operator != "=":
            self.glsl_storage_struct_reinterpret_error(
                source,
                source_type=source_type,
                target_type=source_target,
                reason="storage-workgroup-aggregate-assignment-operator-unsupported",
                detail=f"assignment operator '{operator}' is not supported",
                alignment=layout["alignment"],
            )
        if not statement_context:
            self.glsl_storage_struct_reinterpret_error(
                source,
                source_type=source_type,
                target_type=source_target,
                reason="storage-workgroup-aggregate-expression-unsupported",
                detail="the expanded aggregate copy must be a standalone statement",
                alignment=layout["alignment"],
            )

        destination_offset_node = destination_binding.get("offset", 0)
        source_offset_node = source_binding.get("offset", 0)
        for offset_node, role in (
            (destination_offset_node, "destination"),
            (source_offset_node, "source"),
        ):
            if (
                offset_node is None
                or self.glsl_private_pointer_expression_has_side_effects(offset_node)
            ):
                self.glsl_storage_struct_reinterpret_error(
                    source if role == "source" else destination,
                    source_type=(source_type if role == "source" else destination_type),
                    target_type=source_target,
                    reason=f"storage-workgroup-aggregate-{role}-offset-unprovable",
                    detail=f"the {role} scalar-element offset is unresolved or side effecting",
                    alignment=layout["alignment"],
                )

        word_count = layout["byte_extent"] // destination_layout.byte_width
        source_range = self.glsl_storage_pointer_offset_range(
            source_binding, source.expression
        )
        source_span = (
            IntegerRange(
                source_range.minimum,
                source_range.maximum + word_count - 1,
                source_range.status,
                source_range.provenance,
            )
            if source_range is not None
            else None
        )
        # Every expanded access is emitted in signed target-index arithmetic.
        # The destination extent cannot establish a range for the source buffer.
        accepted_span = IntegerRange(0, (1 << 31) - 1)
        if source_span is None or not source_span.is_within(accepted_span):
            self.glsl_index_type_error(
                source_offset_node,
                source.expression,
                self.glsl_source_expression_type(source_offset_node),
                "int",
                (
                    "index-range-unproven"
                    if source_span is None
                    else (
                        "constant-index-out-of-range"
                        if source_range.is_exact
                        else "index-range-out-of-target-range"
                    )
                ),
                range_status=("unproven" if source_span is None else "out-of-range"),
                source_range=source_span,
                accepted_range=accepted_span,
            )
        proven_interval = destination_binding.get(
            "_glsl_validated_workgroup_access_interval"
        )
        if proven_interval is None:
            offset_interval = self.glsl_workgroup_pointer_offset_interval(
                destination_binding,
                getattr(destination.expression, "_glsl_control_flow_intervals", {}),
            )
            if offset_interval is not None:
                proven_interval = (
                    offset_interval[0],
                    offset_interval[1] + word_count - 1,
                )
        backing_extent = destination_binding.get("array_size")
        if (
            proven_interval is None
            or not isinstance(backing_extent, int)
            or proven_interval[0] < 0
            or proven_interval[1] >= backing_extent
        ):
            rendered_offset = self.glsl_workgroup_pointer_offset_expression(
                destination_binding
            )
            raise OpenGLWorkgroupPointerError(
                "OpenGL cannot prove the expanded storage-to-workgroup aggregate "
                "copy stays within its shared-array backing",
                backing_name=destination_binding.get("root"),
                offset_expression=rendered_offset,
                reason="storage-workgroup-aggregate-out-of-bounds",
                source_location=getattr(destination, "source_location", None),
            )

        destination_offset = self.glsl_workgroup_pointer_offset_expression(
            destination_binding
        )
        source_offset = self.glsl_workgroup_pointer_offset_expression(source_binding)
        if not destination_offset or not source_offset:
            self.glsl_storage_struct_reinterpret_error(
                source,
                source_type=source_type,
                target_type=source_target,
                reason="storage-workgroup-aggregate-offset-unresolved",
                detail="a scalar-element offset cannot be rendered",
                alignment=layout["alignment"],
            )
        return {
            "destination_binding": destination_binding,
            "source_binding": source_binding,
            "destination_offset": destination_offset,
            "source_offset": source_offset,
            "word_count": word_count,
        }

    def global_variable_qualifier(self, node):
        qualifier_prefix = self.glsl_variable_qualifier_prefix(node)
        qualifiers = {str(q).lower() for q in getattr(node, "qualifiers", []) or []}
        runtime_storage_qualifiers = {
            "buffer",
            "device",
            "global",
            "groupshared",
            "in",
            "inout",
            "input",
            "out",
            "output",
            "shared",
            "storage",
            "threadgroup",
            "uniform",
            "workgroup",
        }
        is_compile_time = (
            "const" in qualifiers or id(node) in self.glsl_compile_time_global_node_ids
        ) and not qualifiers & runtime_storage_qualifiers
        if qualifier_prefix:
            storage_qualifiers = {
                "in",
                "out",
                "shared",
                "taskPayloadSharedEXT",
                "rayPayloadEXT",
                "rayPayloadInEXT",
                "hitAttributeEXT",
                "callableDataEXT",
                "callableDataInEXT",
            }
            if is_compile_time and not (
                set(qualifier_prefix.split()) & storage_qualifiers
            ):
                return f"const {qualifier_prefix} "
            return f"{qualifier_prefix} "

        if qualifiers & {"shared", "groupshared", "workgroup", "threadgroup"}:
            return "shared "
        if is_compile_time:
            return "const "
        if qualifiers & {"private", "static", "thread"}:
            # Static and thread-address-space globals are private to each
            # invocation, which is what a GLSL global without a storage
            # qualifier is.
            return ""
        return "uniform "

    def generate_glsl_cooperative_matrix_rounding_support(self):
        from ..fused_math import binary32_fma_support
        from ..lexer import Lexer
        from ..parser import Parser

        source = (
            "shader MatrixRounding {\n"
            + binary32_fma_support(self.glsl_cooperative_matrix_rounding_names())
            + "\n}"
        )
        ast = Parser(Lexer(source).tokens).parse()
        generator = GLSLCodeGen()
        # Initialize a separate module so source declarations cannot capture helper calls.
        generator.generate(ast)
        return source_license_comments(ast, "opengl") + "".join(
            generator.generate_function(function)
            for function in generator.collect_functions(ast)
        )

    def glsl_expected_type_conversion(self, expr, generated, expected_type):
        if expected_type is None or not isinstance(generated, str):
            return generated
        registered_binding = self.glsl_registered_scalar_structure_contract(
            expected_type
        )
        if registered_binding is not None:
            _contract, _source_expected_type, mapped_expected_type = registered_binding
            actual_type = self.glsl_source_expression_type(expr)
            is_destination_type = (
                actual_type is not None
                and self.map_type(actual_type) == mapped_expected_type
            )
            if is_destination_type:
                return generated
            if self.glsl_is_explicit_type_constructor_expression(
                generated, mapped_expected_type
            ):
                return generated
            conversion_kind = StructureConversionKind.CONTEXTUAL_SCALAR_CONVERSION.value
            if actual_type is None:
                self.glsl_struct_construction_error(
                    expr,
                    mapped_expected_type,
                    None,
                    conversion_kind,
                    "source-type-unknown",
                )
            actual = self.glsl_value_type_info(actual_type)
            if actual is not None and actual["width"] == 1:
                converted = self.glsl_registered_scalar_to_struct_conversion(
                    expr,
                    generated,
                    actual_type,
                    mapped_expected_type,
                    conversion_kind,
                )
                if converted is not None:
                    return converted
                self.glsl_struct_construction_error(
                    expr,
                    mapped_expected_type,
                    actual_type,
                    conversion_kind,
                    "scalar-source-kind-unsupported",
                )
            mapped_actual_type = self.map_type(actual_type)
            reason = (
                "structure-conversion-unsupported"
                if self.glsl_struct_node(mapped_actual_type) is not None
                else "source-shape-unsupported"
            )
            self.glsl_struct_construction_error(
                expr,
                mapped_expected_type,
                actual_type,
                conversion_kind,
                reason,
            )
        expected = self.glsl_value_type_info(expected_type)
        if expected is None:
            return generated
        actual_type = self.glsl_source_expression_type(expr)
        actual = self.glsl_value_type_info(actual_type)
        if actual is None:
            if expected["narrow"] is not None:
                converted = generated
                if not self.glsl_is_explicit_type_constructor_expression(
                    converted, expected["mapped"]
                ):
                    converted = f"{expected['mapped']}({converted})"
                return self.glsl_apply_narrow_integer_contract(
                    converted, expected["source"]
                )
            return generated

        narrow = expected["narrow"]
        if narrow is not None:
            carried = self.glsl_narrow_vector_carrier_form(
                generated, expected["source"]
            )
            if carried is not None:
                return carried
        exact_source_type = self.glsl_source_type_signature(
            actual["source"]
        ) == self.glsl_source_type_signature(expected["source"])
        if (
            exact_source_type
            and narrow is not None
            and not isinstance(expr, BinaryOpNode)
            and (
                not self.glsl_is_source_constructor_expression(expr)
                or generated.startswith("bitfieldExtract(")
            )
        ):
            return generated
        if (
            narrow is None
            and expected["width"] == 1
            and self.is_numeric_literal_expression(expr)
            and actual["family"] in {"int", "uint"}
            and expected["family"] in {"int", "uint", "float"}
            and isinstance(getattr(expr, "value", expr), (int, float))
            and getattr(expr, "value", expr) >= 0
        ):
            if (
                self.GLSL_IMPLICIT_LITERAL_CONVERSIONS
                or expected["family"] == actual["family"]
            ):
                return generated
            # GLSL ES has no implicit conversions (ESSL 3.00 section 4.1.10).
            return self.glsl_explicit_literal_conversion(generated, expected)
        if (
            isinstance(expr, FunctionCallNode)
            and self.function_call_name(expr) == "imageLoad"
        ):
            return generated
        if narrow is None and self.glsl_is_explicit_type_constructor_expression(
            generated, expected["mapped"]
        ):
            return generated
        if expected["mapped"] == actual["mapped"]:
            converted = generated
        elif expected["family"] == "bool":
            converted = self.glsl_bool_conversion_expression(
                generated, actual, expected, expr
            )
        else:
            if expected["width"] == 1 and actual["width"] != 1:
                self.glsl_scalar_conversion_error(
                    expr,
                    actual["source"],
                    expected["source"],
                    "vector-to-scalar",
                )
            if expected["width"] > 1 and actual["width"] not in {
                1,
                expected["width"],
            }:
                self.glsl_scalar_conversion_error(
                    expr,
                    actual["source"],
                    expected["source"],
                    "vector-width-mismatch",
                )
            integer = (
                self.glsl_float_to_integer_conversion_info(expr)
                if expected["family"] == "float" and narrow is None
                else None
            )
            converted = (
                self.generate_glsl_integer_float_conversion(
                    expr, generated, integer, expected["mapped"]
                )
                if integer is not None
                else None
            ) or f"{expected['mapped']}({generated})"

        if narrow is not None:
            return self.glsl_apply_narrow_integer_contract(
                converted, expected["source"]
            )
        return converted

    def glsl_outer_array_type(self, type_name):
        base_type, array_suffix = split_array_type_suffix(str(type_name or ""))
        if not array_suffix:
            return None
        match = re.fullmatch(r"\[([^\]]*)\](.*)", array_suffix)
        if match is None:
            return None
        extent_text = match.group(1).strip()
        remaining_suffix = match.group(2)
        element_type = f"{base_type}{remaining_suffix}"
        return extent_text, element_type

    def glsl_padded_array_constructor_arguments(self, func_name, call_args, expr):
        """Pad a short GLSL array constructor to one argument per element.

        GLSL 4.60 section 5.4.4 requires one argument per array element, so
        ``float[3](0.0)`` is invalid. A declaration's initializer is rewritten
        to an initializer list before codegen and zero-filled there, but an
        assignment or a call argument keeps the call spelling, and glslang
        rejects it. Pad the same way here: the given elements first, then the
        element type's value initializer for each missing element; a missing
        element of an array of arrays is the zero inner array, as in
        ``float[2][2](float[2](1.0, 2.0), float[2](0.0, 0.0))``. An extent named
        by a compile-time integer constant, ``float[N](x)``, is read as the
        constant's value; a specialization constant is not one. A constructor
        with more arguments than elements is refused, since that section allows
        no extra argument. A constructor that has exactly enough arguments, or
        an extent that is unsized or not a compile-time integer at any level,
        is left exactly as it was.
        """
        if func_name is None:
            func_name = self.glsl_array_of_arrays_constructor_name(
                getattr(expr, "function", None)
            )
        # An array constructor's name ends with its extent; a generic call with an
        # array type argument, such as ``pick<float[2]>``, is not one.
        if not isinstance(func_name, str) or not func_name.rstrip().endswith("]"):
            return call_args
        array_type = self.glsl_outer_array_type(
            fold_array_type_counts(func_name, self.glsl_active_literal_int_constants())
        )
        if array_type is None:
            return call_args
        extent_text, element_type = array_type
        if "[]" in element_type.replace(" ", ""):
            return call_args
        try:
            extent = int(extent_text)
        except ValueError:
            return call_args
        if extent > 0 and len(call_args) > extent:
            raise OpenGLAggregateInitializerError(
                f"OpenGL array constructor '{func_name}' has {len(call_args)} "
                f"arguments for {extent} elements; GLSL requires exactly one "
                "argument per array element (GLSL 4.60 section 5.4.4)",
                expected_type=func_name,
                element_count=len(call_args),
                expected_element_count=extent,
                reason="element-count-mismatch",
                source_location=getattr(expr, "source_location", None),
            )
        if extent <= 0 or len(call_args) >= extent:
            return call_args
        padded = list(call_args)
        padded.extend(
            self.glsl_value_initialized_expression(expr, element_type)
            for _ in range(extent - len(padded))
        )
        return padded

    @staticmethod
    def glsl_array_of_arrays_constructor_name(callee):
        """``float[2][3]`` for the callee of ``float[2][3](...)``, else ``None``.

        The CrossGL parser keeps a one-dimensional constructor's extent in the
        callee's name, ``float[3]``, but reads the callee of an array of arrays
        as its base type subscripted once per extent, outermost first:
        ``float[2][3]`` is ``float`` indexed by 2, then by 3. Integer literal
        extents and extents named by an identifier, ``float[N][2]``, are read;
        the caller resolves a name against the compile-time constants.
        """
        extents = []
        while isinstance(callee, ArrayAccessNode):
            index = callee.index_expr
            if isinstance(index, LiteralNode) and type(index.value) is int:
                extents.append(index.value)
            elif (
                isinstance(index, IdentifierNode)
                and isinstance(index.name, str)
                and index.name.isidentifier()
            ):
                extents.append(index.name)
            else:
                return None
            callee = callee.array_expr
        if not extents or not isinstance(callee, IdentifierNode):
            return None
        if not isinstance(callee.name, str) or "[" in callee.name:
            return None
        return callee.name + "".join(f"[{extent}]" for extent in reversed(extents))

    def generate_glsl_matrix_aggregate_initializer(self, expr, mapped_type):
        """Lower ``{...}`` for a matrix: nested lists are columns (GLSL 4.60 section 4.1.11)
        and a flat list fills the columns in order, as a constructor does."""
        dimensions = self.glsl_matrix_dimensions(mapped_type)
        if dimensions is None:
            return None

        columns, rows, component_type, column_type = dimensions
        elements = list(getattr(expr, "elements", []) or [])
        nested_columns = all(
            isinstance(element, ArrayLiteralNode) for element in elements
        )
        if nested_columns:
            if len(elements) != columns:
                self.glsl_aggregate_error(
                    expr,
                    expected_type=mapped_type,
                    reason="element-count-mismatch",
                    expected_element_count=columns,
                )
            column_values = [list(element.elements) for element in elements]
            for column_node, column in zip(elements, column_values):
                if len(column) != rows:
                    self.glsl_aggregate_error(
                        column_node,
                        expected_type=mapped_type,
                        reason="element-count-mismatch",
                        expected_element_count=rows,
                    )
        else:
            expected_count = columns * rows
            if len(elements) != expected_count:
                self.glsl_aggregate_error(
                    expr,
                    expected_type=mapped_type,
                    reason="element-count-mismatch",
                    expected_element_count=expected_count,
                )
            column_values = [
                elements[column * rows : (column + 1) * rows]
                for column in range(columns)
            ]

        rendered_columns = []
        for values in column_values:
            rendered_values = [
                self.generate_expression_with_expected(value, component_type)
                for value in values
            ]
            rendered_columns.append(f"{column_type}({', '.join(rendered_values)})")
        return f"{mapped_type}({', '.join(rendered_columns)})"

    def expression_result_type(self, expr):
        if expr is None:
            return None
        if is_comma_expression(expr):
            return self.expression_result_type(comma_result_operand(expr))
        emulated_matrix_type = self.glsl_emulated_matrix_expression_type(expr)
        if emulated_matrix_type is not None:
            return emulated_matrix_type
        if isinstance(expr, PointerReinterpretNode):
            target_type = getattr(expr.target_type, "pointee_type", None)
            return self.map_type(self.type_name_string(target_type))
        if self.is_bool_ray_query_call(expr):
            return "bool"
        if isinstance(expr, VariableNode):
            name = getattr(expr, "name", None)
            return (
                self.local_variable_types.get(name)
                or self.glsl_buffer_block_variable_types.get(name)
                or self.global_variable_types.get(name)
                or self.glsl_constant_types.get(name)
                or self.glsl_builtin_identifier_result_type(name)
            )
        if hasattr(expr, "__class__") and "IdentifierNode" in str(type(expr)):
            name = getattr(expr, "name", None)
            return (
                self.local_variable_types.get(name)
                or self.glsl_buffer_block_variable_types.get(name)
                or self.global_variable_types.get(name)
                or self.glsl_constant_types.get(name)
                or self.glsl_builtin_identifier_result_type(name)
            )
        if isinstance(expr, (int, float)):
            return "float" if isinstance(expr, float) else "int"
        if isinstance(expr, BinaryOpNode):
            operator = self.map_operator(
                getattr(expr, "op", getattr(expr, "operator", None))
            )
            if operator == ",":
                return self.expression_result_type(expr.right)
            if operator in self.GLSL_VECTOR_RELATIONAL_FUNCTIONS:
                left_type = self.expression_result_type(expr.left)
                right_type = self.expression_result_type(expr.right)
                vector_type = self.glsl_relational_vector_type(left_type, right_type)
                if vector_type is not None:
                    return self.glsl_bool_vector_type(vector_type)
                return "bool"
            if operator in {"&&", "||"}:
                return "bool"
            left_type = self.expression_result_type(expr.left)
            right_type = self.expression_result_type(expr.right)
            if self.is_glsl_complex64_type(left_type) or self.is_glsl_complex64_type(
                right_type
            ):
                return "complex64_t"
            if operator == "*":
                product_type = crossgl_linear_algebra_product_type(
                    left_type, right_type
                )
                if product_type is not None:
                    return product_type
            matrix_type = crossgl_matrix_componentwise_type(
                operator, left_type, right_type
            )
            if matrix_type is not None:
                return matrix_type
            arithmetic_type = self.glsl_common_arithmetic_type(
                left_type, right_type, operator
            )
            if arithmetic_type is not None:
                return arithmetic_type
            if self.is_vector_value_type(left_type):
                return left_type
            if self.is_vector_value_type(right_type):
                return right_type
            if left_type == "float" or right_type == "float":
                return "float"
            return left_type or right_type
        if isinstance(expr, UnaryOpNode):
            if self.map_operator(expr.op) == "*":
                binding = self.glsl_workgroup_pointer_binding(
                    expr.operand,
                    self.glsl_workgroup_pointer_aliases(),
                )
                if binding is not None:
                    return binding.get("element_type")
                binding = self.glsl_storage_pointer_binding(
                    expr.operand,
                    self.glsl_storage_pointer_aliases(),
                )
                if binding is not None:
                    return binding.get("element_type")
            return self.expression_result_type(expr.operand)
        if isinstance(expr, TernaryOpNode) or (
            hasattr(expr, "__class__") and "TernaryOp" in str(expr.__class__)
        ):
            true_type = self.expression_result_type(getattr(expr, "true_expr", None))
            false_type = self.expression_result_type(getattr(expr, "false_expr", None))
            arithmetic_type = self.glsl_common_arithmetic_type(
                true_type, false_type, "?:"
            )
            if arithmetic_type is not None:
                return arithmetic_type
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
        if isinstance(expr, ArrayAccessNode):
            binding = self.glsl_workgroup_pointer_binding(
                expr.array,
                self.glsl_workgroup_pointer_aliases(),
            )
            if binding is not None:
                return binding.get("element_type")
            binding = self.glsl_storage_pointer_binding(
                expr.array,
                self.glsl_storage_pointer_aliases(),
            )
            if binding is not None:
                return binding.get("element_type")
            array_type = self.type_name_string(self.expression_result_type(expr.array))
            if array_type and "[" in array_type and "]" in array_type:
                outer_array = self.glsl_outer_array_type(array_type)
                if outer_array is not None:
                    _extent_text, element_type = outer_array
                    return element_type
            # Indexing a pointer/reference (e.g. a `device T*` / `thread T*`
            # kernel parameter lowered to `T*`) yields the pointee type. Without
            # this, subscripting a pointer parameter (common in materialized MLX
            # helpers such as layer_norm's `threadgroup_sum(thread float* x)`)
            # left the element type unresolved, so numeric validations (e.g. the
            # wave-intrinsic argument check for simd_sum -> subgroupAdd) fell back
            # to a diagnostic comment instead of real GLSL.
            if array_type and array_type.rstrip().endswith("*"):
                return array_type.rstrip()[:-1].strip()
            if self.is_structured_buffer_type(array_type):
                return self.structured_buffer_source_element_type(array_type)
            component_type = self.vector_component_type(
                array_type
            ) or component_access_type(array_type, GLSL_FAMILY)
            if component_type is not None:
                return component_type
            return array_type
        if isinstance(expr, MemberAccessNode):
            object_type = self.expression_result_type(expr.object)
            member = str(expr.member)
            pointer_binding = self.glsl_workgroup_pointer_binding(
                expr.object,
                self.glsl_workgroup_pointer_aliases(),
            )
            if pointer_binding is None:
                pointer_binding = self.glsl_storage_pointer_binding(
                    expr.object,
                    self.glsl_storage_pointer_aliases(),
                )
            if pointer_binding is not None:
                pointer_type = pointer_binding.get("element_type")
                member_type = self.struct_member_types.get(
                    self.type_name_string(pointer_type), {}
                ).get(member)
                if member_type:
                    return member_type
            if object_type and all(ch in "xyzwrgba" for ch in member):
                component_type = self.vector_component_type(object_type)
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
            binding = self.glsl_workgroup_pointer_binding(
                expr.pointer_expr,
                self.glsl_workgroup_pointer_aliases(),
            )
            if binding is None:
                binding = self.glsl_storage_pointer_binding(
                    expr.pointer_expr,
                    self.glsl_storage_pointer_aliases(),
                )
            if binding is None:
                return None
            return self.struct_member_types.get(
                self.type_name_string(binding.get("element_type")), {}
            ).get(str(expr.member))
        if isinstance(expr, ConstructorNode):
            return infer_enum_constructor_type(
                self, expr
            ) or infer_struct_constructor_type(self, expr)
        if isinstance(expr, MatchNode):
            return infer_match_expression_result_type(self, expr)
        if isinstance(expr, WaveOpNode):
            return self.glsl_wave_result_type(expr.operation, expr.arguments)
        if isinstance(expr, FunctionCallNode):
            func_expr = getattr(expr, "function", None) or getattr(expr, "name", None)
            func_name = getattr(func_expr, "name", func_expr)
            args = getattr(expr, "arguments", getattr(expr, "args", []))
            unified_type = self.glsl_unified_builtin_result_type(func_name, args)
            if unified_type is not None:
                return unified_type
            buffer_result_type = self.glsl_buffer_call_result_type(func_name, args)
            if buffer_result_type is not None:
                return buffer_result_type
            simd_result_type = self.glsl_plain_simd_call_result_type(func_name, args)
            if simd_result_type is not None:
                return simd_result_type
            wave_operation = self.glsl_wave_operation_name(func_name)
            if wave_operation in self.GLSL_WAVE_INTRINSIC_ARITIES:
                return self.glsl_wave_result_type(wave_operation, args)
            if (
                self.is_metal_simd_group_helper_name(func_name)
                or self.is_mangled_metal_simd_group_helper_name(func_name)
            ) and args:
                return self.expression_result_type(args[0])
            if is_resource_size_query_operation(func_name) and args:
                return self.texture_size_result_type(args[0])
            numeric_result_type = numeric_trait_method_result_type(self, expr)
            if numeric_result_type:
                return numeric_result_type
            if func_name in self.GLSL_COPYSIGN_NAMES:
                resolved_copysign = self.resolve_glsl_function_overload(
                    func_name,
                    args,
                    call_node=expr,
                )
                if resolved_copysign is not None:
                    return self.type_name_string(
                        getattr(resolved_copysign, "return_type", None)
                    )
                if args:
                    return self.expression_result_type(args[0])
            if func_name in {"fmod", "metal::fmod", "metal_u3a_u3afmod"}:
                resolved_remainder = self.resolve_glsl_function_overload(
                    func_name, args, call_node=expr
                )
                if resolved_remainder is not None:
                    return self.type_name_string(
                        getattr(resolved_remainder, "return_type", None)
                    )
                if len(args) == 2 and self.is_vector_value_type(
                    self.expression_result_type(args[1])
                ):
                    return self.expression_result_type(args[1])
                if args:
                    return self.expression_result_type(args[0])
            if func_name in self.GLSL_SIGNBIT_NAMES:
                resolved_signbit = self.resolve_glsl_function_overload(
                    func_name,
                    args,
                    call_node=expr,
                )
                if resolved_signbit is not None:
                    return self.type_name_string(
                        getattr(resolved_signbit, "return_type", None)
                    )
                signbit_result_type = self.glsl_signbit_result_type(func_name, args)
                if signbit_result_type is not None:
                    return signbit_result_type
            if func_name in {"normalize", "reflect"} and args:
                return self.expression_result_type(args[0])
            if (
                func_name in {"max", "min", "fabs", "fmin", "fmax", "select"}
                and args
                and func_name not in self.function_return_types
            ):
                return self.expression_result_type(args[0])
            if func_name == "dot" and args:
                integer_dot = self.glsl_integer_dot_plan(list(args))
                if integer_dot is not None:
                    return integer_dot[0]
                return (
                    self.vector_component_type(self.expression_result_type(args[0]))
                    or "float"
                )
            if func_name == "cross" and args:
                return self.expression_result_type(args[0])
            if func_name in {"mix", "lerp"} and len(args) >= 2:
                left_type = self.expression_result_type(args[0])
                right_type = self.expression_result_type(args[1])
                if self.is_vector_value_type(left_type):
                    return left_type
                if self.is_vector_value_type(right_type):
                    return right_type
                return left_type or right_type
            if (
                func_name == "saturate"
                and args
                and func_name not in self.function_return_types
            ):
                return self.expression_result_type(args[0])
            if (
                func_name == "rcp"
                and args
                and func_name not in self.function_return_types
            ):
                return self.expression_result_type(args[0])
            trailing_zero_result_type = self.glsl_trailing_zero_result_type(
                func_name, args
            )
            if trailing_zero_result_type is not None:
                return trailing_zero_result_type
            isfinite_result_type = self.glsl_isfinite_result_type(func_name, args)
            if isfinite_result_type is not None:
                return isfinite_result_type
            componentwise_boolean_type = self.glsl_componentwise_boolean_result_type(
                func_name, args
            )
            if componentwise_boolean_type is not None:
                return componentwise_boolean_type
            if (
                func_name in self.GLSL_BITCAST_FUNCTIONS
                and args
                and func_name not in self.function_return_types
            ):
                return self.glsl_bitcast_result_type(func_name, args[0])
            metal_as_type_target = self.metal_as_type_target(func_name)
            if self.glsl_bfloat_width(metal_as_type_target) == 1:
                return "bfloat16_t"
            if (
                metal_as_type_target is not None
                and func_name not in self.function_return_types
            ):
                return self.map_type(metal_as_type_target)
            specialized_func_name = generic_function_call_name(self, func_name, args)
            if specialized_func_name in self.function_return_types:
                return self.function_return_types[specialized_func_name]
            resolved_overload = self.resolve_glsl_function_overload(
                func_name,
                args,
                call_node=expr,
            )
            if resolved_overload is not None:
                return self.type_name_string(
                    getattr(resolved_overload, "return_type", None)
                )
            if getattr(
                self, "collecting_generic_function_specializations", False
            ) and concrete_overload_results_differ(
                self,
                func_name,
                lambda generator, type_name: generator.map_type(type_name),
            ):
                # Generic deduction must not bind a type parameter to a guessed overload;
                # it resolves the overload from the argument types it tracks.
                return None
            if func_name in self.function_return_types:
                return self.function_return_types[func_name]
            componentwise_type = self.glsl_componentwise_unary_result_type(
                func_name, args
            )
            if componentwise_type is not None:
                return componentwise_type
            if func_name == "atomicCompareExchangeWeak":
                return "bool"
            if func_name == "imageLoad" and args:
                return self.image_load_result_type(args[0])
            if (
                func_name in {
                    "texture",
                    "textureLod",
                    "textureGrad",
                    "textureProj",
                    "textureProjLod",
                    "textureProjGrad",
                    "textureOffset",
                    "textureLodOffset",
                    "textureGradOffset",
                    "texelFetch",
                    "texelFetchOffset",
                    "textureGather",
                    "textureGatherOffset",
                    "textureGatherOffsets",
                }
                and args
            ):
                return self.texture_sampling_result_type(args[0], func_name)
            if (
                func_name in glsl_call_names_resolving_to(
                    self,
                    "derivative-aliases",
                    self.GLSL_DERIVATIVE_FUNCTIONS,
                    self.GLSL_DERIVATIVE_FUNCTION_ALIASES,
                )
                and args
            ):
                return self.expression_result_type(args[0])
            constructor = self.glsl_constructor_type(func_name)
            if constructor:
                return func_name
        if hasattr(expr, "__class__") and "Literal" in str(expr.__class__):
            literal_type = getattr(getattr(expr, "literal_type", None), "name", None)
            if literal_type:
                return literal_type
        if hasattr(expr, "__class__") and "Identifier" in str(expr.__class__):
            name = getattr(expr, "name", None)
            return (
                self.local_variable_types.get(name)
                or self.glsl_buffer_block_variable_types.get(name)
                or self.global_variable_types.get(name)
            )
        return None

    def generate_expression(self, expr, is_main=False):
        """Render a CrossGL AST expression into GLSL expression syntax."""
        union = self.union_storage.read(expr)
        if union is not None:
            return union
        storage_read = self.float_atomic_storage.read(expr, is_main)
        if storage_read is not None:
            return storage_read
        if expr is None:
            return ""
        ray_query_constant = glsl_ray_query_constant_expression(self, expr)
        if ray_query_constant is not None:
            return ray_query_constant
        if isinstance(expr, PointerReinterpretNode):
            target_type = self.type_name_string(
                getattr(getattr(expr, "target_type", None), "pointee_type", None)
            )
            target_layout = scalar_storage_layout(target_type)
            raise PointerReinterpretationError(
                "OpenGL cannot emit a pointer reinterpretation as a first-class "
                "GLSL expression",
                target_type=target_type,
                address_space=str(
                    getattr(getattr(expr, "target_type", None), "address_space", None)
                    or "unknown"
                ),
                alignment=target_layout.byte_width if target_layout else None,
                target_backend="opengl",
                reason="bare-pointer-reinterpretation",
                source_location=getattr(expr, "source_location", None),
            )
        if isinstance(expr, CooperativeMatrixOpNode):
            if self.cooperative_matrix_software_lowering:
                return self.generate_glsl_cooperative_matrix_operation(expr)
            operation = getattr(expr, "operation", None)
            matrix_type = getattr(expr, "result_type", None)
            fragment_contract = _opengl_cooperative_matrix_fragment_contract(
                matrix_type
            )
            fragment_contract_message = (
                _opengl_cooperative_matrix_fragment_contract_message(fragment_contract)
            )
            raise OpenGLCooperativeMatrixError(
                "OpenGL cooperative-matrix lowering is not available for "
                f"operation '{operation}'; emitting an ordinary GLSL call would "
                "change distributed matrix semantics; "
                f"{fragment_contract_message}",
                operation=operation,
                matrix_type=matrix_type,
                **fragment_contract,
                reason="unsupported-operation",
                source_location=getattr(expr, "source_location", None),
            )
        if isinstance(expr, str):
            return self.generate_glsl_raw_string_expression(expr)
        elif isinstance(expr, (int, float, bool)):
            if isinstance(expr, bool):
                return "true" if expr else "false"
            return str(expr)
        elif hasattr(expr, "__class__") and "VariableNode" in str(type(expr)):
            return self.generate_glsl_variable_expression(expr)
        elif hasattr(expr, "__class__") and "IdentifierNode" in str(type(expr)):
            return self.generate_glsl_identifier_node_expression(expr)
        elif isinstance(expr, ArrayLiteralNode):
            return self.generate_glsl_aggregate_initializer(expr)
        elif is_comma_expression(expr):
            return c_family_comma_expression(
                expr,
                lambda operand: self.generate_expression(operand, is_main),
                self.generate_discarded_expression,
            )
        elif hasattr(expr, "__class__") and "LiteralNode" in str(type(expr)):
            return self.generate_glsl_literal_expression(expr)
        elif hasattr(expr, "__class__") and "BinaryOpNode" in str(type(expr)):
            return self.generate_glsl_binary_expression(expr)
        elif hasattr(expr, "__class__") and "AssignmentNode" in str(type(expr)):
            return assignment_value_expression(
                self.generate_assignment(expr),
                discarded=expr is self.current_discarded_expression,
            )
        elif hasattr(expr, "__class__") and "UnaryOpNode" in str(type(expr)):
            return self.generate_glsl_unary_expression(expr)
        elif isinstance(expr, WaveOpNode):
            return self.generate_glsl_wave_op_expression(expr)
        elif isinstance(expr, RayTracingOpNode):
            args = [self.generate_expression(arg) for arg in expr.arguments]
            return self.map_ray_tracing_intrinsic(expr.operation, args)
        elif isinstance(expr, MeshOpNode):
            return self.generate_glsl_mesh_operation_expression(expr)
        elif isinstance(expr, RayQueryOpNode):
            return self.map_ray_query_intrinsic(
                expr.operation, expr.query_expr, expr.arguments
            )
        elif hasattr(expr, "__class__") and "ArrayAccessNode" in str(type(expr)):
            return self.generate_glsl_array_access_expression(expr)
        elif isinstance(expr, ConstructorNode):
            return self.generate_glsl_constructor_expression(expr)
        elif hasattr(expr, "__class__") and "FunctionCallNode" in str(type(expr)):
            return self.generate_glsl_function_call_expression(expr)
        elif hasattr(expr, "__class__") and "MemberAccessNode" in str(type(expr)):
            return self.generate_glsl_member_access_expression(expr)
        elif isinstance(expr, PointerAccessNode):
            return self.generate_glsl_pointer_access_expression(expr)
        elif hasattr(expr, "__class__") and "TernaryOpNode" in str(type(expr)):
            return self.generate_glsl_ternary_expression(expr)
        else:
            return str(expr)

    def generate_glsl_metal_math_call(self, func_name, args, *, call_node=None):
        arities = {"fabs": 1, "fmin": 2, "fmax": 2, "select": 3}
        if func_name not in arities or func_name in self.function_return_types:
            return None
        types = tuple(self.glsl_source_expression_type(arg) for arg in args)

        def reject(reason):
            raise OpenGLMetalMathError(
                f"OpenGL cannot preserve '{func_name}' for operand types {types}: {reason}",
                operation=func_name,
                operand_types=tuple(self.type_name_string(item) for item in types),
                target_profile=self.current_glsl_version_line,
                reason=reason,
                source_location=getattr(call_node, "source_location", None),
            )

        if len(args) != arities[func_name]:
            reject("invalid-arity")
        if any(item is None for item in types):
            reject("unresolved-operand-type")
        infos = [self.glsl_value_type_info(item) for item in types]
        if any(info is None or info["width"] not in {1, 2, 3, 4} for info in infos):
            reject("unsupported-operand-type")
        version = re.match(r"#version\s+(\d+)\b", self.current_glsl_version_line or "")
        if self.GLSL_TARGET_DISPLAY_NAME != "OpenGL" or (
            version is not None and int(version.group(1)) < 400
        ):
            reject("unsupported-profile")
        value = infos[0]
        values = infos[:2] if func_name != "fabs" else infos
        if any(
            info["family"] != "bool" and info["bits"] not in {32, 64} for info in values
        ):
            reject("unsupported-operand-width")
        if func_name != "select" and any(info["family"] != "float" for info in values):
            reject("non-floating-operand")
        if any(info["mapped"] != value["mapped"] for info in values):
            reject("operand-type-mismatch")
        condition_type = ""
        if func_name == "select":
            condition = infos[2]
            if condition["family"] != "bool":
                reject("non-boolean-condition")
            if condition["width"] not in {1, value["width"]}:
                reject("condition-shape-mismatch")
            condition_type = condition["mapped"]
        builtin = "abs" if func_name == "fabs" else "isnan"
        if func_name != "select" and (
            builtin in self.function_return_types
            or builtin in self.global_variable_types
            or builtin in self.local_variable_types
        ):
            reject("target-builtin-shadowed")
        rendered = [self.generate_expression(arg) for arg in args]
        if func_name == "fabs":
            return f"abs({rendered[0]})"

        # Helper parameters retain eager, single evaluation of both source values.
        identity = ("metal-math", func_name, value["mapped"], condition_type)
        helper_name = self.glsl_module_generated_identifier_names.get(identity)
        if helper_name is None:
            used_names = self.glsl_module_used_identifier_names | getattr(
                self, "glsl_numeric_helper_reserved_names", set()
            )
            helper_name = self.glsl_unique_identifier(
                f"crossgl_{func_name}_{value['mapped']}_{condition_type}".rstrip("_"),
                used_names,
            )
            self.glsl_module_used_identifier_names.add(helper_name)
            self.glsl_module_generated_identifier_names[identity] = helper_name
        self.required_glsl_metal_math_helpers.add(
            (helper_name, func_name, value["mapped"], condition_type, value["width"])
        )
        return f"{helper_name}({', '.join(rendered)})"

    def generate_glsl_wave_operation(
        self,
        operation,
        arguments,
        diagnostic_operation=None,
        source_location=None,
    ):
        if self.software_subgroup_width is not None:
            return self.generate_glsl_software_subgroup_operation(
                operation,
                arguments,
                source_location=source_location,
            )
        diagnostic_operation = diagnostic_operation or operation
        expected_arity = self.GLSL_WAVE_INTRINSIC_ARITIES.get(operation)
        if expected_arity is None:
            return self.glsl_wave_diagnostic_expression(
                diagnostic_operation, "is not recognized by the OpenGL backend"
            )

        actual_arity = len(arguments)
        accepted_arities = self.GLSL_WAVE_INTRINSIC_ALTERNATE_ARITIES.get(
            operation, (expected_arity,)
        )
        if actual_arity not in accepted_arities:
            expected = " or ".join(str(arity) for arity in accepted_arities)
            return self.glsl_wave_diagnostic_expression(
                diagnostic_operation,
                f"expects {expected} arguments, got {actual_arity}",
            )

        floating_bitwise = self.glsl_floating_bitwise_wave_operation(
            operation, arguments
        )
        if floating_bitwise is not None:
            return floating_bitwise

        type_diagnostic = self.glsl_wave_type_diagnostic(
            operation, arguments, diagnostic_operation
        )
        if type_diagnostic is not None:
            return type_diagnostic

        if operation == "WaveGetLaneCount":
            return "gl_SubgroupSize"
        if operation == "WaveGetLaneIndex":
            return "gl_SubgroupInvocationID"
        if operation == "WaveIsFirstLane":
            return "subgroupElect()"
        if operation == "QuadActiveSum":
            value = self.generate_expression(arguments[0])
            return f"subgroupClusteredAdd({value}, 4u)"

        if operation == "WaveActiveCountBits":
            predicate = self.generate_expression(arguments[0])
            return f"subgroupBallotBitCount(subgroupBallot({predicate}))"
        if operation == "WavePrefixCountBits":
            predicate = self.generate_expression(arguments[0])
            return f"subgroupBallotExclusiveBitCount(subgroupBallot({predicate}))"
        if operation == "WaveMatch":
            value = self.generate_expression(arguments[0])
            return f"crossglWaveMatch({value})"
        if operation == "WaveShuffleAndFillUp":
            value_type = self.expression_result_type(arguments[0])
            value = self.generate_expression_with_expected(arguments[0], value_type)
            fill = self.generate_expression_with_expected(arguments[1], value_type)
            unsigned_arguments = []
            for argument in arguments[2:]:
                argument_type = self.expression_result_type(argument)
                generated_argument = self.generate_expression_with_expected(
                    argument, None
                )
                if argument_type is None or self.map_type(argument_type) != "uint":
                    generated_argument = f"uint({generated_argument})"
                unsigned_arguments.append(generated_argument)
            delta = unsigned_arguments[0]
            modulo = (
                unsigned_arguments[1]
                if len(unsigned_arguments) == 2
                else "gl_SubgroupSize"
            )
            return f"crossglWaveShuffleAndFillUp({value}, {fill}, {delta}, {modulo})"
        if operation in self.GLSL_WAVE_MULTI_PREFIX_OPERATIONS:
            value = self.generate_expression(arguments[0])
            mask = self.generate_expression(arguments[1])
            helper = self.GLSL_WAVE_MULTI_PREFIX_HELPERS[operation]
            return f"{helper}({value}, {mask})"

        if quad_lane_read_needs_helper(operation, arguments):
            return glsl_quad_lane_read_call(self, arguments)

        mapped = self.GLSL_WAVE_DIRECT_MAPPINGS.get(operation)
        if mapped is None:
            return self.glsl_wave_diagnostic_expression(
                diagnostic_operation, "is not recognized by the OpenGL backend"
            )

        args = ", ".join(self.generate_expression(arg) for arg in arguments)
        return f"{mapped}({args})"

    def texture_call_parts(self, args):
        explicit_sampler = self.is_explicit_sampler_argument(args)
        coord_index = 2 if explicit_sampler else 1
        if len(args) <= coord_index:
            return None

        texture_name = self.generate_expression(args[0])
        coord = self.generate_expression(args[coord_index])
        extra_args = args[coord_index + 1 :]
        return texture_name, coord, extra_args

    def generate_texture_call(self, func_name, args):
        if not func_name or func_name not in self.texture_resource_operation_names():
            return None

        texture_type = self.texture_resource_type(args[0]) if args else None
        if is_texel_fetch_basic_operation(func_name) and self.resource_base_type(
            texture_type
        ) in {"samplerBuffer", "isamplerBuffer", "usamplerBuffer"}:
            if len(args) != 2:
                raise OpenGLRefusal(
                    f"{self.GLSL_TARGET_DISPLAY_NAME} texture-buffer operation "
                    f"'{func_name}' requires 2 argument(s), got {len(args)}"
                )
            texture_name = self.generate_expression(args[0])
            # A 32-bit texel-buffer coordinate is a texel coordinate, as for
            # imageLoad: every in-range unsigned value converts to int
            # unchanged. Other widths need the index range proof.
            coordinate_info = self.glsl_value_type_info(
                self.glsl_source_expression_type(args[1])
            )
            if (
                coordinate_info is not None
                and coordinate_info["width"] == 1
                and coordinate_info["bits"] == 32
                and coordinate_info["narrow"] is None
            ):
                coordinate = self.glsl_signed_coordinate_expression(args[1])
            else:
                coordinate = self.glsl_index_expression(
                    args[1], args[0], target_signed=True
                )
            return f"texelFetch({texture_name}, {coordinate})"

        self.validate_texture_call_arity(func_name, args)
        self.validate_image_resource_argument(func_name, args)
        self.validate_image_access_argument(func_name, args)
        self.validate_texture_resource_argument(func_name, args)
        self.validate_texture_size_argument_count(func_name, args)
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

        if is_texture_query_levels_operation(func_name) and args:
            return self.texture_query_levels_expression(args[0])

        if is_resource_samples_query_operation(func_name) and args:
            return self.texture_samples_expression(func_name, args[0])

        if is_resource_size_query_operation(func_name) and args:
            return self.texture_size_query_expression(args[0])

        if len(args) < 2:
            return None

        if func_name == "imageLoad" and len(args) >= 2:
            image_name = self.generate_expression(args[0])
            coord = self.glsl_signed_coordinate_expression(args[1])
            texture_type = self.texture_resource_type(args[0])
            if len(args) >= 3 and self.is_multisample_storage_image_type(texture_type):
                sample = self.generate_expression(args[2])
                load_expr = f"imageLoad({image_name}, {coord}, {sample})"
            else:
                load_expr = f"imageLoad({image_name}, {coord})"
            image_format = self.image_resource_format(args[0])
            self.validate_image_load_result_type(texture_type, image_format)
            return f"{load_expr}{self.image_load_component_suffix(texture_type, image_format)}"

        if func_name == "imageStore" and len(args) >= 3:
            image_name = self.generate_expression(args[0])
            coord = self.glsl_signed_coordinate_expression(args[1])
            texture_type = self.texture_resource_type(args[0])
            if len(args) >= 4 and self.is_multisample_storage_image_type(texture_type):
                sample = self.generate_expression(args[2])
                value_arg = args[3]
            else:
                sample = None
                value_arg = args[2]
            image_format = self.image_resource_format(args[0])
            self.validate_image_store_value_type(texture_type, image_format, value_arg)
            expected_value_type = self.image_store_expected_value_type(
                texture_type, image_format, value_arg
            )
            value = self.generate_expression_with_expected(
                value_arg,
                expected_value_type,
            )
            value = self.glsl_precise_stored_value(
                None,
                value_arg,
                value,
                expected_value_type or self.expression_result_type(value_arg),
            )
            value = self.image_store_value_expression(
                texture_type,
                image_format,
                value,
                self.expression_result_type(value_arg),
            )
            if sample is not None:
                return f"imageStore({image_name}, {coord}, {sample}, {value})"
            return f"imageStore({image_name}, {coord}, {value})"

        if is_image_atomic_operation(func_name) and args:
            return self.generate_image_atomic_call(func_name, args)

        texture_type = self.texture_resource_type(args[0])
        storage_image_operation = self.storage_image_texture_operation_expression(
            func_name, texture_type
        )
        if storage_image_operation is not None:
            return storage_image_operation

        if is_texture_compare_operation(func_name):
            return self.generate_texture_compare_call(func_name, args)

        if func_name == "textureGatherCompareOffsets":
            return self.generate_texture_gather_compare_offsets_call(func_name, args)

        if is_texture_gather_compare_operation(func_name):
            return self.generate_texture_gather_compare_call(func_name, args)

        if is_texture_gather_operation(func_name):
            return self.generate_texture_gather_call(func_name, args)

        if is_texture_query_lod_operation(func_name):
            parts = self.texture_call_parts(args)
            if parts is None:
                return None
            texture_name, coord, _ = parts
            texture_type = self.texture_resource_type(args[0])
            if self.is_multisample_texture_resource_type(texture_type):
                return self.unsupported_multisample_texture_query_lod_call(texture_type)
            if self.is_storage_image_type(texture_type):
                return self.unsupported_texture_query_lod_call(texture_type)
            coord = self.texture_query_lod_coordinate(texture_type, coord)
            return f"textureQueryLod({texture_name}, {coord})"

        if is_texel_fetch_offset_operation(func_name) and len(args) >= 4:
            texture_type = self.texture_resource_type(args[0])
            if self.is_cube_texture_resource_type(texture_type):
                return self.unsupported_cube_texel_fetch_call(func_name, texture_type)
            if self.is_multisample_texture_resource_type(texture_type):
                return self.unsupported_multisample_texel_fetch_offset_call(
                    texture_type
                )
            if not self.glsl_texture_call_offsets_are_compile_time_constants(
                func_name, args
            ):
                return self.unsupported_dynamic_texel_fetch_offset_call(
                    func_name, texture_type
                )
            return None

        if is_texel_fetch_basic_operation(func_name) and len(args) >= 3:
            texture_type = self.texture_resource_type(args[0])
            if self.is_cube_texture_resource_type(texture_type):
                return self.unsupported_cube_texel_fetch_call(func_name, texture_type)
            if any(
                self.glsl_expression_is_unsigned(argument) for argument in args[1:3]
            ):
                rendered = [self.generate_expression(args[0])]
                rendered.extend(
                    self.glsl_signed_coordinate_expression(argument)
                    for argument in args[1:3]
                )
                rendered.extend(self.generate_expression(arg) for arg in args[3:])
                return f"{func_name}({', '.join(rendered)})"
            return None

        if is_projected_texture_operation(func_name) and args:
            texture_type = self.resource_base_type(self.texture_resource_type(args[0]))
            if texture_type in {"samplerCube", "isamplerCube", "usamplerCube"}:
                return self.generate_projected_cube_texture_call(func_name, args)
            if texture_type in {
                "samplerCubeArray",
                "isamplerCubeArray",
                "usamplerCubeArray",
            }:
                return self.unsupported_texture_projected_call(
                    func_name,
                    "requires 1D, 2D, 2D-array, or 3D projection coordinates",
                    texture_type,
                )

        if is_texture_sampling_operation(func_name):
            texture_type = self.texture_resource_type(args[0])
            if self.is_multisample_texture_resource_type(texture_type):
                return self.unsupported_multisample_texture_call(
                    func_name, texture_type
                )
            if is_texture_sample_offset_operation(func_name):
                if not self.texture_sample_offset_supported(texture_type):
                    return self.unsupported_texture_sample_offset_call(
                        func_name, texture_sample_offset_capability_error("GLSL")
                    )
                if not self.glsl_texture_call_offsets_are_compile_time_constants(
                    func_name, args
                ):
                    return self.unsupported_dynamic_texel_offset_sample_call(func_name)
            if (
                is_projected_texture_basic_offset_operation(func_name)
                or is_projected_texture_lod_offset_operation(func_name)
                or is_projected_texture_grad_offset_operation(func_name)
            ):
                if not self.texture_sample_offset_supported(texture_type):
                    return self.unsupported_texture_projected_call(
                        func_name,
                        texture_sample_offset_capability_error("GLSL"),
                        texture_type,
                    )
                if not self.glsl_texture_call_offsets_are_compile_time_constants(
                    func_name, args
                ):
                    return self.unsupported_dynamic_texel_offset_projected_call(
                        func_name, texture_type
                    )

        if not is_texture_sampling_operation(
            func_name
        ) or not self.is_explicit_sampler_argument(args):
            return None

        parts = self.texture_call_parts(args)
        if parts is None:
            return None
        texture_name, coord, extra_args = parts
        sampler_name = self.generate_expression(args[1])
        declared_texture_type = self.texture_declared_resource_type(args[0])
        if self.is_separate_texture_object_type(declared_texture_type):
            sampled_type = self.sampled_image_type(declared_texture_type)
            sampled_type = self.resource_base_type(sampled_type)
            texture_name = f"{sampled_type}({texture_name}, {sampler_name})"
        else:
            self.record_glsl_combined_sampler_use(args[0], args[1])
            texture_name = self.glsl_vulkan_resources.sampled(
                args, texture_name, sampler_name
            )
        mapped_args = [texture_name, coord] + [
            self.generate_expression(arg) for arg in extra_args
        ]
        return f"{func_name}({', '.join(mapped_args)})"

    def collect_resource_array_size_hints(self, ast):
        texture_globals = self.collect_unsized_sampled_texture_globals(ast)
        texture_parameters = self.collect_unsized_sampled_texture_parameters(ast)
        buffer_globals = self.collect_unsized_structured_buffer_globals(ast)
        functions = self.collect_functions(ast)
        dynamic_sizes = DynamicIndexSizes(
            global_arrays=set(texture_globals) | set(buffer_globals),
            shared_globals=texture_globals,
            function_arrays=texture_parameters,
            functions=functions,
            constants=self.visible_literal_int_constants,
            target="opengl",
        )
        global_hints, function_hints = collect_resource_array_size_hints(
            global_arrays=texture_globals,
            function_arrays=texture_parameters,
            fixed_global_array_sizes=self.collect_fixed_resource_global_sizes(ast),
            fixed_function_array_sizes=self.collect_fixed_resource_parameter_sizes(ast),
            functions=functions,
            walk_nodes=self.walk_ast,
            expression_name=self.expression_name,
            literal_int_value=self.literal_int_value,
            visible_literal_int_constants=self.visible_literal_int_constants,
            function_call_name=self.function_call_name,
            initial_size=0,
            format_size=lambda size: str(size) if size > 1 else "",
            initial_literal_int_constants=self.initial_literal_int_constants,
            required_sizes=dynamic_sizes,
        )
        global_hints.update(
            self.collect_structured_buffer_global_array_size_hints(ast, function_hints)
        )
        buffer_sizes = dynamic_sizes.result()
        for name in sorted(set(buffer_globals) & set(buffer_sizes)):
            hint = str(global_hints.get(name) or "")
            if buffer_sizes[name] > (int(hint) if hint.isdigit() else 0):
                global_hints[name] = str(buffer_sizes[name])
        return global_hints, function_hints

    def collect_private_pointer_array_size_hints(self, ast):
        functions = []
        seen_function_ids = set()
        for function in [
            *self.collect_functions(ast),
            *getattr(self, "generic_function_specializations", {}).values(),
        ]:
            if id(function) in seen_function_ids or generic_function_parameters(
                function
            ):
                continue
            seen_function_ids.add(id(function))
            functions.append(function)

        functions_by_key = {}
        source_candidates = {}
        pointer_parameters = {}
        pointer_parameter_indices = {}
        pointer_element_types = {}
        pointer_read_only_parameters = {}
        pointer_base_names = {}
        source_names = {}
        for function in functions:
            function_name = getattr(function, "name", None)
            if not function_name:
                continue
            function_key = self.glsl_function_declaration_name(function)
            functions_by_key[function_key] = function
            source_candidates.setdefault(function_name, []).append(function_key)
            parameters = list(getattr(function, "parameters", []) or [])
            private_parameters = [
                parameter
                for parameter in parameters
                if is_private_pointer_parameter(parameter)
            ]
            if not private_parameters:
                continue
            pointer_parameters[function_key] = private_parameters
            pointer_parameter_indices[function_key] = {
                index: parameter.name
                for index, parameter in enumerate(parameters)
                if is_private_pointer_parameter(parameter)
            }
            pointer_element_types[function_key] = {
                parameter.name: self.glsl_private_pointer_parameter_element_type(
                    parameter, function_key
                )
                for parameter in private_parameters
            }
            pointer_read_only_parameters[function_key] = {
                parameter.name
                for parameter in private_parameters
                if {
                    str(qualifier).lower()
                    for qualifier in getattr(parameter, "qualifiers", []) or []
                }.intersection({"const", "constant", "readonly"})
                or not bool(
                    getattr(
                        getattr(
                            parameter,
                            "param_type",
                            getattr(parameter, "vtype", None),
                        ),
                        "is_mutable",
                        True,
                    )
                )
            }
            used_names = self.collect_glsl_function_local_identifier_names(function)
            base_names = {}
            for parameter in private_parameters:
                base_name = self.glsl_unique_identifier(
                    f"{parameter.name}_base",
                    used_names,
                    extra_reserved=self.GLSL_ALIAS_TARGET_LOCAL_IDENTIFIERS,
                )
                used_names.add(base_name)
                base_names[parameter.name] = base_name
            pointer_base_names[function_key] = base_names
            source_names[function_key] = function_name

        self.function_private_pointer_parameters = pointer_parameters
        self.function_private_pointer_parameter_indices = pointer_parameter_indices
        self.function_private_pointer_element_types = pointer_element_types
        self.function_private_pointer_read_only_parameters = (
            pointer_read_only_parameters
        )
        self.function_private_pointer_base_names = pointer_base_names
        self.function_private_pointer_source_names = source_names
        self.glsl_private_pointer_robust_zero_read_variants = {}
        if not pointer_parameters:
            self.function_private_pointer_required_spans = {}
            self.function_private_pointer_backing_variants = {}
            self.function_private_pointer_local_arrays = {}
            self.function_private_pointer_view_calls = {}
            self.function_private_pointer_full_span_parameters = {}
            self.function_private_pointer_scalar_access_violations = {}
            self.glsl_private_pointer_zero_index_access_ids = set()
            self.glsl_private_pointer_noop_rebinding_ids = set()
            self.glsl_private_pointer_base_rebinding_parameters = {}
            return {}

        local_arrays = {
            function_key: self.glsl_private_pointer_local_arrays(function)
            for function_key, function in functions_by_key.items()
        }
        self.function_private_pointer_local_arrays = local_arrays
        self.function_private_pointer_full_span_parameters = {
            function_key: set() for function_key in pointer_parameters
        }
        self.function_private_pointer_scalar_access_violations = {
            function_key: set() for function_key in pointer_parameters
        }
        self.glsl_private_pointer_zero_index_access_ids = set()
        self.glsl_private_pointer_noop_rebinding_ids = set()
        self.glsl_private_pointer_base_rebinding_parameters = {}

        propagated_intervals = self.glsl_private_pointer_propagated_parameter_intervals(
            functions_by_key,
            source_candidates,
            pointer_parameters,
        )
        required_spans = {}
        for function_key, parameters in pointer_parameters.items():
            function = functions_by_key[function_key]
            spans = {parameter.name: 0 for parameter in parameters}
            self.glsl_collect_private_pointer_static_direct_spans(
                function,
                function_key,
                spans,
                parameter_intervals=propagated_intervals.get(function_key),
            )
            required_spans[function_key] = spans

        view_calls = {function_key: [] for function_key in functions_by_key}
        for caller_key, function in functions_by_key.items():
            caller_element_types = pointer_element_types.get(caller_key, {})
            constants = self.visible_literal_int_constants(function)
            caller_local_arrays = local_arrays.get(caller_key, {})
            view_calls[caller_key] = self.glsl_collect_private_pointer_view_calls(
                function,
                caller_key,
                source_candidates,
                caller_element_types,
                caller_local_arrays,
                constants,
                parameter_intervals=propagated_intervals.get(caller_key),
            )

        self.function_private_pointer_required_spans = required_spans
        self.function_private_pointer_view_calls = view_calls
        for caller_key, calls in view_calls.items():
            if not calls or caller_key not in pointer_parameters:
                continue
            function = functions_by_key[caller_key]
            self.glsl_attach_private_pointer_correlated_view_contracts(
                function,
                caller_key,
                calls,
                self.visible_literal_int_constants(function),
            )
        for calls in view_calls.values():
            for call in calls:
                seen_backing_roots = {}
                for parameter_name, binding in call["bindings"].items():
                    if binding.get("root_kind") not in {
                        "local",
                        "parameter",
                        "scalar",
                        "byte_view",
                    }:
                        self.glsl_private_pointer_backing_error(
                            call, parameter_name, binding
                        )
                    if binding.get("root_kind") == "local" and not isinstance(
                        binding.get("extent"), int
                    ):
                        self.glsl_private_pointer_backing_error(
                            call, parameter_name, binding
                        )
                    self.glsl_private_pointer_binding_offset_interval(
                        call,
                        parameter_name,
                        binding,
                        require_exact=binding.get("root_kind") == "scalar",
                    )
                    expected_type = pointer_element_types[call["callee"]][
                        parameter_name
                    ]
                    actual_type = binding.get("element_type")
                    if actual_type != expected_type:
                        display_name = self.glsl_private_pointer_display_name(
                            call["callee"]
                        )
                        raise OpenGLPrivatePointerParameterError(
                            "OpenGL private pointer view "
                            f"'{display_name}.{parameter_name}' has incompatible "
                            f"element type '{actual_type or 'unknown'}'; expected "
                            f"'{expected_type}'",
                            function_name=display_name,
                            parameter_name=parameter_name,
                            reason="incompatible-element-type",
                            source_location=getattr(
                                call.get("node"), "source_location", None
                            ),
                        )
                    backing_key = (binding.get("root_kind"), binding.get("root"))
                    previous_parameter = seen_backing_roots.get(backing_key)
                    if previous_parameter is not None:
                        display_name = self.glsl_private_pointer_display_name(
                            call["callee"]
                        )
                        backing_name = binding.get("root") or "unknown"
                        raise OpenGLPrivatePointerParameterError(
                            "OpenGL private pointer call to "
                            f"'{display_name}' binds parameters "
                            f"'{previous_parameter}' and '{parameter_name}' to "
                            f"the same backing object '{backing_name}'; GLSL "
                            "inout copy-out cannot preserve aliased views",
                            function_name=display_name,
                            parameter_name=parameter_name,
                            reason="aliased-view-backing",
                            source_location=getattr(
                                call.get("node"), "source_location", None
                            ),
                        )
                    seen_backing_roots[backing_key] = parameter_name

        changed = True
        while changed:
            changed = False
            for caller_key, calls in view_calls.items():
                caller_full_span = (
                    self.function_private_pointer_full_span_parameters.get(
                        caller_key, set()
                    )
                )
                for call in calls:
                    callee_full_span = (
                        self.function_private_pointer_full_span_parameters.get(
                            call["callee"], set()
                        )
                    )
                    for parameter_name in tuple(callee_full_span):
                        binding = call["bindings"].get(parameter_name, {})
                        if binding.get("root_kind") != "parameter":
                            continue
                        root_name = binding.get("root")
                        if root_name not in caller_full_span:
                            caller_full_span.add(root_name)
                            changed = True

        for calls in view_calls.values():
            for call in calls:
                full_span_parameters = (
                    self.function_private_pointer_full_span_parameters.get(
                        call["callee"], set()
                    )
                )
                for parameter_name in full_span_parameters:
                    binding = call["bindings"].get(parameter_name, {})
                    if binding.get("root_kind") == "local" and binding.get(
                        "offset_interval"
                    ) == (0, 0):
                        required_spans[call["callee"]][parameter_name] = max(
                            required_spans[call["callee"]][parameter_name],
                            binding["extent"],
                        )

        max_iterations = max(
            1,
            sum(len(parameters) for parameters in pointer_parameters.values())
            * max(len(functions_by_key), 1),
        )
        for _iteration in range(max_iterations):
            changed = False
            for caller_key, calls in view_calls.items():
                caller_spans = required_spans.get(caller_key, {})
                for call in calls:
                    callee_spans = required_spans[call["callee"]]
                    for parameter_name, binding in call["bindings"].items():
                        span = callee_spans.get(parameter_name, 0)
                        if span <= 0 or binding.get("root_kind") != "parameter":
                            continue
                        if binding.get("correlated_access_contract") is not None:
                            continue
                        _lower, upper = (
                            self.glsl_private_pointer_binding_offset_interval(
                                call, parameter_name, binding
                            )
                        )
                        needed = upper + span
                        root_name = binding["root"]
                        if needed > caller_spans.get(root_name, 0):
                            caller_spans[root_name] = needed
                            changed = True
            if not changed:
                break
        else:
            raise OpenGLPrivatePointerParameterError(
                "OpenGL private pointer view requirements do not converge",
                reason="recursive-view-span",
            )

        for function_key, parameters in pointer_parameters.items():
            for parameter in parameters:
                if required_spans[function_key].get(parameter.name, 0) > 0:
                    continue
                display_name = self.glsl_private_pointer_display_name(function_key)
                raise OpenGLPrivatePointerParameterError(
                    "OpenGL private pointer parameter "
                    f"'{display_name}.{parameter.name}' has no provable bounded span",
                    function_name=display_name,
                    parameter_name=parameter.name,
                    reason="missing-fixed-array-extent",
                    source_location=getattr(parameter, "source_location", None),
                )

        variants = {function_key: set() for function_key in pointer_parameters}

        def propagate_variants():
            changed = True
            while changed:
                changed = False
                for caller_key, calls in view_calls.items():
                    caller_parameters = pointer_parameters.get(caller_key, [])
                    caller_parameter_positions = {
                        parameter.name: index
                        for index, parameter in enumerate(caller_parameters)
                    }
                    caller_variants = tuple(variants.get(caller_key, ()))
                    for call in calls:
                        requires_caller_variant = any(
                            binding.get("root_kind") == "parameter"
                            for binding in call["bindings"].values()
                        )
                        if requires_caller_variant and not caller_variants:
                            continue
                        source_variants = (
                            caller_variants if requires_caller_variant else ((),)
                        )
                        for caller_variant in source_variants:
                            callee_variant = []
                            for parameter in pointer_parameters[call["callee"]]:
                                binding = call["bindings"][parameter.name]
                                if binding["root_kind"] == "byte_view":
                                    extent = binding["byte_view"]
                                elif binding["root_kind"] == "local":
                                    extent = binding["extent"]
                                elif binding["root_kind"] == "scalar":
                                    extent = 0
                                else:
                                    root_name = binding["root"]
                                    position = caller_parameter_positions.get(root_name)
                                    if position is None:
                                        self.glsl_private_pointer_backing_error(
                                            call, parameter.name, binding
                                        )
                                    extent = caller_variant[position]
                                self.glsl_validate_private_pointer_call_binding(
                                    call,
                                    parameter.name,
                                    binding,
                                    self.glsl_private_pointer_variant_extent(extent),
                                )
                                callee_variant.append(extent)
                            callee_variant = tuple(callee_variant)
                            if callee_variant not in variants[call["callee"]]:
                                variants[call["callee"]].add(callee_variant)
                                changed = True

        propagate_variants()
        for function_key, parameters in pointer_parameters.items():
            if variants[function_key]:
                continue
            variants[function_key].add(
                tuple(
                    required_spans[function_key][parameter.name]
                    for parameter in parameters
                )
            )
        propagate_variants()

        self.function_private_pointer_backing_variants = {
            function_key: tuple(
                sorted(
                    function_variants,
                    key=self.glsl_private_pointer_variant_sort_key,
                )
            )
            for function_key, function_variants in variants.items()
        }
        return {
            function_key: {
                parameter.name: str(
                    self.glsl_private_pointer_variant_extent(
                        self.function_private_pointer_backing_variants[function_key][0][
                            index
                        ]
                    )
                )
                for index, parameter in enumerate(parameters)
            }
            for function_key, parameters in pointer_parameters.items()
        }

    def glsl_collect_private_pointer_view_calls(
        self,
        function,
        caller_key,
        source_candidates,
        caller_element_types,
        caller_local_arrays,
        constants,
        parameter_intervals=None,
    ):
        calls = []
        calls_by_node = {}
        initial_intervals = dict(parameter_intervals or {})
        for name, value in constants.items():
            literal = self.literal_int_value(value, constants)
            if literal is not None:
                initial_intervals[name] = (literal, literal)
        initial_bindings = {
            name: {
                "root_kind": "parameter",
                "root": name,
                "offset": 0,
                "offset_interval": (0, 0),
                "side_effecting": False,
                "element_type": element_type,
            }
            for name, element_type in caller_element_types.items()
        }
        for parameter in getattr(function, "parameters", []) or []:
            name = getattr(parameter, "name", None)
            if not name or name in initial_bindings:
                continue
            info = self.glsl_private_pointer_local_scalar_node_info(parameter)
            if info is not None:
                initial_bindings[name] = {
                    "root_kind": "scalar_value",
                    "root": name,
                    "extent": 1,
                    "offset": 0,
                    "offset_interval": (0, 0),
                    "side_effecting": False,
                    "element_type": info["element_type"],
                    "source_type": info["source_type"],
                }

        def shadow_binding(node):
            name = getattr(node, "name", None)
            if not name:
                return None, None
            info = self.glsl_private_pointer_local_array_node_info(node, constants)
            if info is not None:
                return name, {
                    "root_kind": "local",
                    "root": name,
                    "extent": info["extent"],
                    "offset": 0,
                    "offset_interval": (0, 0),
                    "side_effecting": False,
                    "element_type": info["element_type"],
                }
            scalar_info = self.glsl_private_pointer_local_scalar_node_info(node)
            if scalar_info is not None:
                return name, {
                    "root_kind": "scalar_value",
                    "root": name,
                    "extent": 1,
                    "offset": 0,
                    "offset_interval": (0, 0),
                    "side_effecting": False,
                    "element_type": scalar_info["element_type"],
                    "source_type": scalar_info["source_type"],
                }
            return name, {
                "root_kind": None,
                "root": name,
                "offset": None,
                "offset_interval": None,
                "side_effecting": False,
                "element_type": None,
            }

        def merge_intervals(left, right):
            return {
                name: (
                    min(left[name][0], right[name][0]),
                    max(left[name][1], right[name][1]),
                )
                for name in left.keys() & right.keys()
            }

        def modified_names(value):
            return self.glsl_function_mutated_interval_names(
                SimpleNamespace(body=value)
            )

        def record_call(node, active_bindings, active_intervals):
            callee_key = self.glsl_private_pointer_call_target_key(
                node,
                source_candidates,
            )
            parameter_indices = self.function_private_pointer_parameter_indices.get(
                callee_key, {}
            )
            arguments = list(
                getattr(node, "arguments", getattr(node, "args", [])) or []
            )
            for index, argument in enumerate(arguments):
                if index in parameter_indices:
                    continue
                escaped_binding = self.glsl_private_pointer_static_binding(
                    argument,
                    caller_element_types,
                    caller_local_arrays,
                    constants,
                    active_bindings,
                    active_intervals,
                )
                if (
                    escaped_binding is not None
                    and escaped_binding.get("root_kind") == "parameter"
                ):
                    display_name = self.glsl_private_pointer_display_name(caller_key)
                    parameter_name = escaped_binding["root"]
                    callee_name = self.function_call_name(node) or "<unknown>"
                    raise OpenGLPrivatePointerParameterError(
                        "OpenGL private pointer parameter "
                        f"'{display_name}.{parameter_name}' escapes through call "
                        f"'{callee_name}'",
                        function_name=display_name,
                        parameter_name=parameter_name,
                        reason="pointer-escape",
                        source_location=getattr(node, "source_location", None),
                    )
            if not parameter_indices:
                return
            bindings = {}
            for index, parameter_name in parameter_indices.items():
                argument = arguments[index] if index < len(arguments) else None
                binding = self.glsl_private_pointer_static_binding(
                    argument,
                    caller_element_types,
                    caller_local_arrays,
                    constants,
                    active_bindings,
                    active_intervals,
                )
                if binding is None:
                    binding = {
                        "root_kind": None,
                        "root": self.expression_name(argument),
                        "offset": None,
                        "offset_interval": None,
                        "side_effecting": (
                            self.glsl_private_pointer_expression_has_side_effects(
                                argument
                            )
                        ),
                        "element_type": None,
                        "ambiguous_shape": isinstance(argument, TernaryOpNode),
                    }
                bindings[parameter_name] = binding
            existing = calls_by_node.get(id(node))
            if existing is not None:
                for parameter_name, binding in bindings.items():
                    existing["bindings"][parameter_name] = (
                        self.glsl_merge_private_pointer_static_bindings(
                            [existing["bindings"].get(parameter_name), binding]
                        )
                    )
                return
            call = {
                "caller": caller_key,
                "callee": callee_key,
                "node": node,
                "bindings": bindings,
            }
            calls_by_node[id(node)] = call
            calls.append(call)

        def visit(value, active_bindings, active_intervals):
            if value is None or isinstance(value, (str, int, float, bool)):
                return
            if isinstance(value, (list, tuple)):
                for item in value:
                    visit(item, active_bindings, active_intervals)
                return
            if isinstance(value, BlockNode):
                block_bindings = dict(active_bindings)
                block_intervals = dict(active_intervals)
                statements = list(getattr(value, "statements", []) or [])
                outer_names = set(active_bindings)
                propagated_bindings = {
                    name: active_bindings.get(name) for name in outer_names
                }
                propagated_intervals = {
                    name: active_intervals.get(name) for name in outer_names
                }
                shadowed_names = set()
                for statement in statements:
                    declared_name = (
                        getattr(statement, "name", None)
                        if isinstance(statement, (VariableNode, ArrayNode))
                        else None
                    )
                    visit(statement, block_bindings, block_intervals)
                    if declared_name:
                        shadowed_names.add(declared_name)
                    for name in outer_names - shadowed_names:
                        propagated_bindings[name] = block_bindings.get(name)
                        propagated_intervals[name] = block_intervals.get(name)
                for name in outer_names:
                    binding = propagated_bindings.get(name)
                    if binding is None:
                        active_bindings.pop(name, None)
                    else:
                        active_bindings[name] = dict(binding)
                    interval = propagated_intervals.get(name)
                    if interval is None:
                        active_intervals.pop(name, None)
                    else:
                        active_intervals[name] = interval
                return
            if isinstance(value, (VariableNode, ArrayNode)):
                initial_value = getattr(value, "initial_value", None)
                visit(initial_value, active_bindings, active_intervals)
                name, binding = shadow_binding(value)
                if name:
                    interval = self.glsl_private_pointer_affine_interval(
                        initial_value, active_intervals, constants
                    )
                    if interval is None:
                        active_intervals.pop(name, None)
                    else:
                        active_intervals[name] = interval
                    active_bindings[name] = binding
                return
            if isinstance(value, AssignmentNode):
                target = getattr(value, "target", getattr(value, "left", None))
                assigned = getattr(value, "value", getattr(value, "right", None))
                rebased_parameter = (
                    self.glsl_private_pointer_base_rebinding_parameters.get(id(value))
                )
                if rebased_parameter is not None:
                    visit(assigned, active_bindings, active_intervals)
                    target_name = self.expression_name(target)
                    current = active_bindings.get(target_name)
                    delta_interval = self.glsl_private_pointer_affine_interval(
                        assigned,
                        active_intervals,
                        constants,
                    )
                    operator = self.map_operator(
                        getattr(value, "operator", getattr(value, "op", "="))
                    )
                    replacement = (
                        self.glsl_private_pointer_rebased_static_binding(
                            current,
                            delta_interval,
                            operator,
                        )
                        if current is not None
                        else None
                    )
                    if replacement is None:
                        display_name = self.glsl_private_pointer_display_name(
                            caller_key
                        )
                        raise OpenGLPrivatePointerParameterError(
                            "OpenGL cannot prove a forwarded private pointer base "
                            "after loop-carried rebinding",
                            function_name=display_name,
                            parameter_name=rebased_parameter,
                            reason="pointer-parameter-rebinding",
                            source_location=getattr(value, "source_location", None),
                        )
                    active_bindings[target_name] = replacement
                    return
                if id(value) in self.glsl_private_pointer_noop_rebinding_ids:
                    visit(assigned, active_bindings, active_intervals)
                    return
                visit(target, active_bindings, active_intervals)
                visit(assigned, active_bindings, active_intervals)
                target_name = self.expression_name(target)
                if target_name and isinstance(
                    target, (str, IdentifierNode, VariableNode)
                ):
                    assigned_interval = self.glsl_private_pointer_affine_interval(
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
            if isinstance(value, FunctionCallNode):
                record_call(value, active_bindings, active_intervals)
                arguments = list(
                    getattr(value, "arguments", getattr(value, "args", [])) or []
                )
                for argument in arguments:
                    visit(argument, active_bindings, active_intervals)
                function_name = self.function_call_name(value)
                candidates = list(
                    self.glsl_function_overloads_by_name.get(function_name, ())
                )
                fallback = self.function_definitions.get(function_name)
                if fallback is not None and fallback not in candidates:
                    candidates.append(fallback)
                for argument_index, argument in enumerate(arguments):
                    if not any(
                        argument_index < len(
                            parameters := list(
                                getattr(
                                    candidate,
                                    "parameters",
                                    getattr(candidate, "params", []),
                                )
                                or []
                            )
                        )
                        and set(
                            self.glsl_parameter_qualifiers(parameters[argument_index])
                        ).intersection({"out", "inout"})
                        for candidate in candidates
                    ):
                        continue
                    for name in self.glsl_mutation_target_names(argument):
                        active_intervals.pop(name, None)
                        self.glsl_clear_index_component_intervals(
                            active_intervals,
                            name,
                        )
                return
            if isinstance(value, IfNode):
                condition_node = getattr(value, "condition", None)
                condition = self.glsl_private_pointer_condition_value(
                    condition_node, active_intervals, constants
                )
                visit(
                    condition_node,
                    active_bindings,
                    active_intervals,
                )
                if condition is not None:
                    selected_branch = (
                        getattr(value, "then_branch", None)
                        if condition
                        else getattr(value, "else_branch", None)
                    )
                    visit(selected_branch, active_bindings, active_intervals)
                    return
                then_intervals = dict(active_intervals)
                else_intervals = dict(active_intervals)
                visit(
                    getattr(value, "then_branch", None),
                    dict(active_bindings),
                    then_intervals,
                )
                visit(
                    getattr(value, "else_branch", None),
                    dict(active_bindings),
                    else_intervals,
                )
                active_intervals.clear()
                active_intervals.update(merge_intervals(then_intervals, else_intervals))
                return
            if isinstance(value, ForNode):
                loop_bindings = dict(active_bindings)
                loop_intervals = dict(active_intervals)
                visit(getattr(value, "init", None), loop_bindings, loop_intervals)
                rebinding_ids = {
                    id(child)
                    for child in self.walk_ast(getattr(value, "body", None))
                    if isinstance(child, AssignmentNode)
                    and id(child) in self.glsl_private_pointer_base_rebinding_parameters
                }
                if rebinding_ids:
                    iteration_values = (
                        self.glsl_private_pointer_rebase_loop_iteration_values(
                            value,
                            loop_intervals,
                            constants,
                        )
                    )
                    if not iteration_values:
                        display_name = self.glsl_private_pointer_display_name(
                            caller_key
                        )
                        raise OpenGLPrivatePointerParameterError(
                            "OpenGL cannot reproduce a bounded private pointer "
                            "rebase while collecting forwarded views",
                            function_name=display_name,
                            reason="pointer-parameter-rebinding",
                            source_location=getattr(value, "source_location", None),
                        )
                    loop_name = getattr(getattr(value, "init", None), "name", None)
                    binding_states = [dict(loop_bindings)]
                    for iteration_value in iteration_values:
                        if loop_name:
                            loop_intervals[loop_name] = (
                                iteration_value,
                                iteration_value,
                            )
                        visit(
                            getattr(value, "condition", None),
                            loop_bindings,
                            loop_intervals,
                        )
                        visit(
                            getattr(value, "body", None),
                            loop_bindings,
                            loop_intervals,
                        )
                        visit(
                            getattr(value, "update", None),
                            loop_bindings,
                            loop_intervals,
                        )
                        binding_states.append(dict(loop_bindings))
                    init_name = getattr(getattr(value, "init", None), "name", None)
                    for name in active_bindings:
                        if name == init_name:
                            continue
                        merged = self.glsl_merge_private_pointer_static_bindings(
                            [state.get(name) for state in binding_states]
                        )
                        if merged is not None:
                            active_bindings[name] = merged
                    changed_names = modified_names(getattr(value, "body", None))
                    changed_names.update(modified_names(getattr(value, "update", None)))
                    for name in changed_names:
                        active_intervals.pop(name, None)
                    return
                loop_name, loop_interval = self.glsl_private_pointer_loop_interval(
                    value, loop_intervals, constants
                )
                if loop_name and loop_name in modified_names(
                    getattr(value, "body", None)
                ):
                    loop_interval = None
                if loop_name:
                    if loop_interval is None:
                        loop_intervals.pop(loop_name, None)
                    else:
                        loop_intervals[loop_name] = loop_interval
                visit(
                    getattr(value, "condition", None),
                    loop_bindings,
                    loop_intervals,
                )
                visit(getattr(value, "body", None), loop_bindings, loop_intervals)
                visit(getattr(value, "update", None), loop_bindings, loop_intervals)
                changed_names = modified_names(getattr(value, "body", None))
                changed_names.update(modified_names(getattr(value, "update", None)))
                for name in changed_names:
                    active_intervals.pop(name, None)
                return
            if isinstance(value, ForInNode):
                loop_bindings = dict(active_bindings)
                loop_intervals = dict(active_intervals)
                iterable = getattr(value, "iterable", None)
                visit(iterable, loop_bindings, loop_intervals)
                pattern_interval = self.glsl_for_in_pattern_interval(
                    value,
                    loop_intervals,
                    constants,
                    self.glsl_private_pointer_affine_interval,
                )
                pattern = getattr(value, "pattern", None)
                if pattern:
                    loop_intervals.pop(pattern, None)
                    self.glsl_clear_index_component_intervals(
                        loop_intervals,
                        pattern,
                    )
                    if pattern_interval is not None:
                        loop_intervals[pattern] = pattern_interval
                    loop_bindings[pattern] = {
                        "root_kind": None,
                        "root": pattern,
                        "offset": None,
                        "offset_interval": None,
                        "side_effecting": False,
                        "element_type": None,
                    }
                visit(getattr(value, "body", None), loop_bindings, loop_intervals)
                changed_names = modified_names(getattr(value, "body", None))
                changed_names.discard(pattern)
                for name in changed_names:
                    active_intervals.pop(name, None)
                    self.glsl_clear_index_component_intervals(
                        active_intervals,
                        name,
                    )
                return
            if isinstance(value, (WhileNode, DoWhileNode, LoopNode)):
                loop_bindings = dict(active_bindings)
                loop_intervals = {
                    name: interval
                    for name, interval in active_intervals.items()
                    if self.literal_int_value(name, constants) is not None
                }
                visit(
                    getattr(value, "condition", None),
                    loop_bindings,
                    loop_intervals,
                )
                visit(getattr(value, "body", None), loop_bindings, loop_intervals)
                changed_names = modified_names(getattr(value, "condition", None))
                changed_names.update(modified_names(getattr(value, "body", None)))
                for name in changed_names:
                    active_intervals.pop(name, None)
                return
            if isinstance(value, UnaryOpNode):
                visit(value.operand, active_bindings, active_intervals)
                if self.map_operator(value.op) == "&":
                    for name in self.glsl_mutation_target_names(value.operand):
                        active_intervals.pop(name, None)
                        self.glsl_clear_index_component_intervals(
                            active_intervals,
                            name,
                        )
                elif value.op in {"++", "--"}:
                    name = self.expression_name(value.operand)
                    if name:
                        active_intervals.pop(name, None)
                return
            if not hasattr(value, "__dict__"):
                return
            for key, child in vars(value).items():
                if key in {"parent", "annotations", "args", "name"}:
                    continue
                visit(child, active_bindings, active_intervals)

        visit(getattr(function, "body", []), initial_bindings, initial_intervals)
        return calls

    def glsl_attach_private_pointer_correlated_view_contracts(
        self,
        function,
        function_key,
        calls,
        constants,
    ):
        """Attach same-backing access witnesses to otherwise symbolic views."""

        pointer_roots = set(
            self.function_private_pointer_element_types.get(function_key, {})
        )
        full_span_roots = self.function_private_pointer_full_span_parameters.get(
            function_key, set()
        )
        if not pointer_roots or not full_span_roots:
            return

        intervals = {}
        forms = {}
        value_keys = {}
        for name, value in constants.items():
            literal = self.literal_int_value(value, constants)
            if literal is None:
                continue
            intervals[name] = (literal, literal)
            forms[name] = (literal, ())
            value_keys[name] = ("literal", literal)
        for parameter in getattr(function, "parameters", []) or []:
            name = getattr(parameter, "name", None)
            if not name or name in pointer_roots:
                continue
            symbol = f"value:{name}"
            forms[name] = (0, ((symbol, 1),))
            value_keys[name] = ("value", name)

        calls_by_id = {id(call["node"]): call for call in calls}
        call_footprints = []
        access_footprints = []

        def modified_names(value):
            names = set()
            for child in self.walk_ast(value):
                if isinstance(child, AssignmentNode):
                    target = getattr(child, "target", getattr(child, "left", None))
                    if isinstance(target, (str, IdentifierNode, VariableNode)):
                        name = self.expression_name(target)
                        if name:
                            names.add(name)
                elif isinstance(child, UnaryOpNode) and child.op in {"++", "--"}:
                    name = self.expression_name(child.operand)
                    if name:
                        names.add(name)
            return names

        def set_name_value(
            name,
            expression,
            operator,
            active_forms,
            active_value_keys,
            active_intervals,
            source,
        ):
            affine = self.glsl_private_pointer_affine_form(
                expression, active_forms, constants
            )
            value_key = self.glsl_private_pointer_contract_expression_key(
                expression, active_value_keys, constants
            )
            interval = self.glsl_private_pointer_interval(
                expression, active_intervals, constants
            )
            if operator != "=":
                scale = 1 if operator == "+=" else -1 if operator == "-=" else None
                if scale is None:
                    affine = None
                    interval = None
                    value_key = None
                else:
                    affine = self.glsl_private_pointer_combine_affine_forms(
                        active_forms.get(name), affine, scale
                    )
                    current_interval = active_intervals.get(name)
                    if current_interval is None or interval is None:
                        interval = None
                    elif scale > 0:
                        interval = (
                            current_interval[0] + interval[0],
                            current_interval[1] + interval[1],
                        )
                    else:
                        interval = (
                            current_interval[0] - interval[1],
                            current_interval[1] - interval[0],
                        )
                    current_key = active_value_keys.get(name)
                    value_key = (
                        None
                        if current_key is None or value_key is None
                        else ("binary", operator[0], current_key, value_key)
                    )
            if affine is None:
                active_forms.pop(name, None)
            else:
                active_forms[name] = affine
            if interval is None:
                active_intervals.pop(name, None)
            else:
                active_intervals[name] = interval
            active_value_keys[name] = (
                value_key if value_key is not None else ("unknown", id(source), name)
            )

        def record_call(
            node,
            active_forms,
            active_loops,
            active_predicates,
            active_pointer_roots,
        ):
            call = calls_by_id.get(id(node))
            if call is None:
                return
            arguments = list(
                getattr(node, "arguments", getattr(node, "args", [])) or []
            )
            parameter_indices = self.function_private_pointer_parameter_indices.get(
                call["callee"], {}
            )
            for index, parameter_name in parameter_indices.items():
                binding = call["bindings"].get(parameter_name, {})
                if (
                    binding.get("root_kind") != "parameter"
                    or binding.get("offset_interval") is not None
                    or binding.get("side_effecting")
                    or binding.get("root") not in full_span_roots
                    or parameter_name
                    in self.function_private_pointer_full_span_parameters.get(
                        call["callee"], set()
                    )
                ):
                    continue
                argument = arguments[index] if index < len(arguments) else None
                symbolic = self.glsl_private_pointer_symbolic_offset(
                    argument,
                    active_pointer_roots,
                    active_forms,
                    constants,
                )
                if symbolic is None or symbolic[0] != binding.get("root"):
                    continue
                call_footprints.append(
                    {
                        "binding": binding,
                        "call": call,
                        "parameter": parameter_name,
                        "root": symbolic[0],
                        "form": symbolic[1],
                        "loops": dict(active_loops),
                        "predicates": tuple(active_predicates),
                    }
                )

        def record_access(
            node,
            active_forms,
            active_loops,
            active_predicates,
            active_pointer_roots,
        ):
            symbolic = self.glsl_private_pointer_symbolic_offset(
                node.array,
                active_pointer_roots,
                active_forms,
                constants,
            )
            index = self.glsl_private_pointer_affine_form(
                node.index, active_forms, constants
            )
            if symbolic is None or index is None or symbolic[0] not in full_span_roots:
                return
            access_footprints.append(
                {
                    "root": symbolic[0],
                    "form": self.glsl_private_pointer_combine_affine_forms(
                        symbolic[1], index
                    ),
                    "loops": dict(active_loops),
                    "predicates": tuple(active_predicates),
                    "source_location": getattr(node, "source_location", None),
                }
            )

        def visit(
            value,
            active_forms,
            active_value_keys,
            active_intervals,
            active_loops,
            active_predicates,
            active_pointer_roots,
        ):
            if value is None or isinstance(value, (str, int, float, bool)):
                return
            if isinstance(value, (list, tuple)):
                for item in value:
                    visit(
                        item,
                        active_forms,
                        active_value_keys,
                        active_intervals,
                        active_loops,
                        active_predicates,
                        active_pointer_roots,
                    )
                return
            if isinstance(value, BlockNode):
                block_forms = dict(active_forms)
                block_value_keys = dict(active_value_keys)
                block_intervals = dict(active_intervals)
                block_roots = set(active_pointer_roots)
                for statement in getattr(value, "statements", []) or []:
                    visit(
                        statement,
                        block_forms,
                        block_value_keys,
                        block_intervals,
                        active_loops,
                        active_predicates,
                        block_roots,
                    )
                return
            if isinstance(value, (VariableNode, ArrayNode)):
                initial_value = getattr(value, "initial_value", None)
                visit(
                    initial_value,
                    active_forms,
                    active_value_keys,
                    active_intervals,
                    active_loops,
                    active_predicates,
                    active_pointer_roots,
                )
                name = getattr(value, "name", None)
                if name:
                    set_name_value(
                        name,
                        initial_value,
                        "=",
                        active_forms,
                        active_value_keys,
                        active_intervals,
                        value,
                    )
                    active_pointer_roots.discard(name)
                return
            if isinstance(value, AssignmentNode):
                target = getattr(value, "target", getattr(value, "left", None))
                assigned = getattr(value, "value", getattr(value, "right", None))
                visit(
                    target,
                    active_forms,
                    active_value_keys,
                    active_intervals,
                    active_loops,
                    active_predicates,
                    active_pointer_roots,
                )
                visit(
                    assigned,
                    active_forms,
                    active_value_keys,
                    active_intervals,
                    active_loops,
                    active_predicates,
                    active_pointer_roots,
                )
                if isinstance(target, (str, IdentifierNode, VariableNode)):
                    name = self.expression_name(target)
                    if name:
                        set_name_value(
                            name,
                            assigned,
                            getattr(value, "operator", "="),
                            active_forms,
                            active_value_keys,
                            active_intervals,
                            value,
                        )
                return
            if isinstance(value, FunctionCallNode):
                record_call(
                    value,
                    active_forms,
                    active_loops,
                    active_predicates,
                    active_pointer_roots,
                )
                for argument in (
                    getattr(value, "arguments", getattr(value, "args", [])) or []
                ):
                    visit(
                        argument,
                        active_forms,
                        active_value_keys,
                        active_intervals,
                        active_loops,
                        active_predicates,
                        active_pointer_roots,
                    )
                return
            if isinstance(value, ArrayAccessNode):
                record_access(
                    value,
                    active_forms,
                    active_loops,
                    active_predicates,
                    active_pointer_roots,
                )
                visit(
                    value.array,
                    active_forms,
                    active_value_keys,
                    active_intervals,
                    active_loops,
                    active_predicates,
                    active_pointer_roots,
                )
                visit(
                    value.index,
                    active_forms,
                    active_value_keys,
                    active_intervals,
                    active_loops,
                    active_predicates,
                    active_pointer_roots,
                )
                return
            if isinstance(value, IfNode):
                condition = getattr(value, "condition", None)
                condition_value = self.glsl_private_pointer_condition_value(
                    condition, active_intervals, constants
                )
                visit(
                    condition,
                    active_forms,
                    active_value_keys,
                    active_intervals,
                    active_loops,
                    active_predicates,
                    active_pointer_roots,
                )
                if condition_value is not None:
                    selected_branch = (
                        getattr(value, "then_branch", None)
                        if condition_value
                        else getattr(value, "else_branch", None)
                    )
                    visit(
                        selected_branch,
                        active_forms,
                        active_value_keys,
                        active_intervals,
                        active_loops,
                        active_predicates,
                        active_pointer_roots,
                    )
                    for name in modified_names(selected_branch):
                        active_forms.pop(name, None)
                        active_intervals.pop(name, None)
                        active_value_keys[name] = ("unknown", id(value), name)
                    return
                condition_key = self.glsl_private_pointer_contract_expression_key(
                    condition, active_value_keys, constants
                )
                if condition_key is None:
                    condition_key = ("unknown-condition", id(value))
                visit(
                    getattr(value, "then_branch", None),
                    dict(active_forms),
                    dict(active_value_keys),
                    dict(active_intervals),
                    active_loops,
                    (*active_predicates, (condition_key, True)),
                    set(active_pointer_roots),
                )
                visit(
                    getattr(value, "else_branch", None),
                    dict(active_forms),
                    dict(active_value_keys),
                    dict(active_intervals),
                    active_loops,
                    (*active_predicates, (condition_key, False)),
                    set(active_pointer_roots),
                )
                for name in modified_names(value):
                    active_forms.pop(name, None)
                    active_intervals.pop(name, None)
                    active_value_keys[name] = ("unknown", id(value), name)
                return
            if isinstance(value, ForNode):
                loop_forms = dict(active_forms)
                loop_value_keys = dict(active_value_keys)
                loop_intervals = dict(active_intervals)
                loop_roots = set(active_pointer_roots)
                visit(
                    getattr(value, "init", None),
                    loop_forms,
                    loop_value_keys,
                    loop_intervals,
                    active_loops,
                    active_predicates,
                    loop_roots,
                )
                loop_contract = self.glsl_private_pointer_correlated_loop_contract(
                    value,
                    loop_intervals,
                    loop_value_keys,
                    constants,
                    active_loops,
                )
                loop_name = getattr(getattr(value, "init", None), "name", None)
                nested_loops = dict(active_loops)
                if loop_contract is not None and loop_name:
                    symbol = loop_contract["symbol"]
                    loop_forms[loop_name] = (0, ((symbol, 1),))
                    loop_value_keys[loop_name] = ("loop", loop_contract["key"])
                    nested_loops[symbol] = loop_contract
                elif loop_name:
                    loop_forms.pop(loop_name, None)
                    loop_value_keys[loop_name] = ("unknown-loop", id(value))
                    loop_intervals.pop(loop_name, None)
                visit(
                    getattr(value, "condition", None),
                    loop_forms,
                    loop_value_keys,
                    loop_intervals,
                    nested_loops,
                    active_predicates,
                    loop_roots,
                )
                visit(
                    getattr(value, "body", None),
                    loop_forms,
                    loop_value_keys,
                    loop_intervals,
                    nested_loops,
                    active_predicates,
                    loop_roots,
                )
                visit(
                    getattr(value, "update", None),
                    loop_forms,
                    loop_value_keys,
                    loop_intervals,
                    nested_loops,
                    active_predicates,
                    loop_roots,
                )
                for name in modified_names(value):
                    active_forms.pop(name, None)
                    active_intervals.pop(name, None)
                    active_value_keys[name] = ("unknown", id(value), name)
                return
            if isinstance(value, (ForInNode, WhileNode, DoWhileNode, LoopNode)):
                loop_forms = {
                    name: form
                    for name, form in active_forms.items()
                    if name not in modified_names(value)
                }
                visit(
                    getattr(value, "iterable", None),
                    loop_forms,
                    dict(active_value_keys),
                    dict(active_intervals),
                    active_loops,
                    active_predicates,
                    set(active_pointer_roots),
                )
                visit(
                    getattr(value, "condition", None),
                    loop_forms,
                    dict(active_value_keys),
                    dict(active_intervals),
                    active_loops,
                    active_predicates,
                    set(active_pointer_roots),
                )
                visit(
                    getattr(value, "body", None),
                    loop_forms,
                    dict(active_value_keys),
                    dict(active_intervals),
                    active_loops,
                    active_predicates,
                    set(active_pointer_roots),
                )
                return
            if isinstance(value, UnaryOpNode):
                visit(
                    value.operand,
                    active_forms,
                    active_value_keys,
                    active_intervals,
                    active_loops,
                    active_predicates,
                    active_pointer_roots,
                )
                if value.op in {"++", "--"}:
                    name = self.expression_name(value.operand)
                    if name:
                        active_forms.pop(name, None)
                        active_intervals.pop(name, None)
                        active_value_keys[name] = ("unknown", id(value), name)
                return
            if not hasattr(value, "__dict__"):
                return
            for key, child in vars(value).items():
                if key in {
                    "parent",
                    "annotations",
                    "args",
                    "name",
                    "left",
                    "right",
                    "array",
                    "index",
                    "if_condition",
                    "if_body",
                    "else_body",
                }:
                    continue
                visit(
                    child,
                    active_forms,
                    active_value_keys,
                    active_intervals,
                    active_loops,
                    active_predicates,
                    active_pointer_roots,
                )

        visit(
            getattr(function, "body", []),
            forms,
            value_keys,
            intervals,
            {},
            (),
            pointer_roots,
        )

        for footprint in call_footprints:
            if not self.glsl_private_pointer_offset_is_nonnegative(footprint):
                continue
            for access in access_footprints:
                covered_span = self.glsl_private_pointer_access_covered_span(
                    footprint, access
                )
                if covered_span is not None:
                    footprint["binding"]["correlated_access_contract"] = {
                        "span": covered_span,
                        "source_location": access["source_location"],
                    }
                    break

    def glsl_collect_private_pointer_static_direct_spans(
        self,
        function,
        function_key,
        spans,
        parameter_intervals=None,
    ):
        parameter_element_types = self.function_private_pointer_element_types.get(
            function_key, {}
        )
        constants = self.initial_literal_int_constants(function)
        loop_parameter_intervals = dict(
            getattr(function, "_glsl_workgroup_proof_parameter_intervals", {}) or {}
        )
        loop_parameter_intervals.update(parameter_intervals or {})
        intervals = {}
        for name, value in constants.items():
            literal = self.literal_int_value(value, constants)
            if literal is not None:
                intervals[name] = (literal, literal)
        pointer_bindings = {
            name: {
                "root_kind": "parameter",
                "root": name,
                "offset": 0,
                "offset_interval": (0, 0),
                "side_effecting": False,
                "element_type": element_type,
            }
            for name, element_type in parameter_element_types.items()
        }
        discarded_assignment_ids = self.glsl_private_pointer_discarded_assignment_ids(
            function
        )

        def shadow_binding(node):
            name = getattr(node, "name", None)
            if not name:
                return None, None
            info = self.glsl_private_pointer_local_array_node_info(node, constants)
            if info is not None:
                return name, {
                    "root_kind": "local",
                    "root": name,
                    "extent": info["extent"],
                    "offset": 0,
                    "offset_interval": (0, 0),
                    "side_effecting": False,
                    "element_type": info["element_type"],
                }
            return name, {
                "root_kind": None,
                "root": name,
                "offset": None,
                "offset_interval": None,
                "side_effecting": False,
                "element_type": None,
            }

        def pointer_parameter_binding(expression, active_bindings):
            if not isinstance(expression, (str, IdentifierNode, VariableNode)):
                return None
            return pointer_value_binding(expression, active_bindings)

        def pointer_value_binding(expression, active_bindings):
            binding = self.glsl_private_pointer_static_binding(
                expression,
                parameter_element_types,
                {},
                constants,
                active_bindings,
            )
            if binding is not None and binding.get("root_kind") == "parameter":
                return binding
            return None

        def reject_pointer_rebinding(binding, source):
            display_name = self.glsl_private_pointer_display_name(function_key)
            parameter_name = binding["root"]
            raise OpenGLPrivatePointerParameterError(
                "OpenGL cannot rebase private pointer parameter "
                f"'{display_name}.{parameter_name}' because GLSL inout "
                "parameters cannot model pointer reassignment",
                function_name=display_name,
                parameter_name=parameter_name,
                reason="pointer-parameter-rebinding",
                source_location=getattr(source, "source_location", None),
            )

        def reject_pointer_escape(binding, source):
            display_name = self.glsl_private_pointer_display_name(function_key)
            parameter_name = binding["root"]
            raise OpenGLPrivatePointerParameterError(
                "OpenGL private pointer parameter "
                f"'{display_name}.{parameter_name}' escapes its helper contract",
                function_name=display_name,
                parameter_name=parameter_name,
                reason="pointer-escape",
                source_location=getattr(source, "source_location", None),
            )

        def record_access(
            pointer_expression,
            index_expression,
            active_intervals,
            active_bindings,
            source,
        ):
            binding = self.glsl_private_pointer_static_binding(
                pointer_expression,
                parameter_element_types,
                {},
                constants,
                active_bindings,
            )
            if binding is None or binding.get("root_kind") != "parameter":
                return
            index = self.glsl_private_pointer_interval(
                index_expression, active_intervals, constants
            )
            offset_interval = self.glsl_private_pointer_static_binding_interval(binding)
            if index is None or offset_interval is None:
                self.function_private_pointer_full_span_parameters.setdefault(
                    function_key, set()
                ).add(binding["root"])
                return
            if offset_interval != (0, 0) or index != (0, 0):
                self.function_private_pointer_scalar_access_violations.setdefault(
                    function_key, set()
                ).add(binding["root"])
            lower = offset_interval[0] + index[0]
            upper = offset_interval[1] + index[1]
            if lower < 0:
                display_name = self.glsl_private_pointer_display_name(function_key)
                raise OpenGLPrivatePointerParameterError(
                    "OpenGL private pointer access "
                    f"'{display_name}.{binding['root']}' may use negative index "
                    f"{lower}",
                    function_name=display_name,
                    parameter_name=binding["root"],
                    reason="negative-view-access",
                    source_location=getattr(source, "source_location", None),
                )
            if lower == 0 and upper == 0:
                self.glsl_private_pointer_zero_index_access_ids.add(id(source))
            spans[binding["root"]] = max(spans[binding["root"]], upper + 1)

        def merge_intervals(left, right):
            return {
                name: (
                    min(left[name][0], right[name][0]),
                    max(left[name][1], right[name][1]),
                )
                for name in left.keys() & right.keys()
            }

        bounded_rebase_depth = 0
        conditional_rebase_depth = 0
        unsafe_rebase_depth = 0

        def value_contains_pointer_rebase(value):
            for child in self.walk_ast(value):
                if not isinstance(child, AssignmentNode):
                    continue
                target = getattr(child, "target", getattr(child, "left", None))
                operator = self.map_operator(
                    getattr(child, "operator", getattr(child, "op", "="))
                )
                if (
                    isinstance(target, (str, IdentifierNode, VariableNode))
                    and self.expression_name(target) in parameter_element_types
                    and operator in {"+=", "-="}
                ):
                    return True
            return False

        def loop_contains_pointer_rebase(value):
            return value_contains_pointer_rebase(getattr(value, "body", None))

        def visit(value, active_intervals, active_bindings):
            nonlocal bounded_rebase_depth
            nonlocal conditional_rebase_depth
            nonlocal unsafe_rebase_depth
            if value is None or isinstance(value, (str, int, float, bool)):
                return
            if isinstance(value, (list, tuple)):
                for item in value:
                    visit(item, active_intervals, active_bindings)
                return
            if isinstance(value, BlockNode):
                block_bindings = dict(active_bindings)
                block_intervals = dict(active_intervals)
                statements = list(getattr(value, "statements", []) or [])
                # Bindings identify mutable names; intervals also contain constants.
                outer_binding_names = set(active_bindings)
                propagated_intervals = {
                    name: active_intervals.get(name) for name in outer_binding_names
                }
                propagated_bindings = {
                    name: active_bindings.get(name) for name in outer_binding_names
                }
                shadowed_names = set()
                for statement in statements:
                    declared_name = (
                        getattr(statement, "name", None)
                        if isinstance(statement, (VariableNode, ArrayNode))
                        else None
                    )
                    visit(statement, block_intervals, block_bindings)
                    if declared_name:
                        shadowed_names.add(declared_name)
                    for name in outer_binding_names - shadowed_names:
                        propagated_intervals[name] = block_intervals.get(name)
                        propagated_bindings[name] = block_bindings.get(name)
                for name, interval in propagated_intervals.items():
                    if interval is not None:
                        active_intervals[name] = interval
                    else:
                        active_intervals.pop(name, None)
                for name, binding in propagated_bindings.items():
                    if binding is None:
                        active_bindings.pop(name, None)
                    else:
                        active_bindings[name] = dict(binding)
                return
            if isinstance(value, (VariableNode, ArrayNode)):
                initial_value = getattr(value, "initial_value", None)
                node_type = getattr(value, "var_type", getattr(value, "vtype", None))
                initial_pointer_binding = pointer_value_binding(
                    initial_value, active_bindings
                )
                if isinstance(node_type, PointerType) and initial_pointer_binding:
                    reject_pointer_escape(initial_pointer_binding, value)
                visit(initial_value, active_intervals, active_bindings)
                name = getattr(value, "name", None)
                if name:
                    interval = self.glsl_private_pointer_interval(
                        initial_value, active_intervals, constants
                    )
                    if interval is None:
                        active_intervals.pop(name, None)
                    else:
                        active_intervals[name] = interval
                    _name, binding = shadow_binding(value)
                    active_bindings[name] = binding
                return
            if isinstance(value, AssignmentNode):
                target = getattr(value, "target", getattr(value, "left", None))
                assigned = getattr(value, "value", getattr(value, "right", None))
                target_binding = pointer_parameter_binding(target, active_bindings)
                if target_binding is not None:
                    operator = self.map_operator(
                        getattr(value, "operator", getattr(value, "op", "="))
                    )
                    assigned_interval = self.glsl_private_pointer_interval(
                        assigned,
                        active_intervals,
                        constants,
                    )
                    direct_parameter_target = (
                        isinstance(target, (str, IdentifierNode, VariableNode))
                        and self.expression_name(target) == target_binding["root"]
                    )
                    if (
                        id(value) in discarded_assignment_ids
                        and operator in {"+=", "-="}
                        and assigned_interval == (0, 0)
                        and direct_parameter_target
                        and not self.glsl_private_pointer_expression_has_side_effects(
                            assigned
                        )
                    ):
                        visit(assigned, active_intervals, active_bindings)
                        if (
                            id(value)
                            not in self.glsl_private_pointer_base_rebinding_parameters
                        ):
                            self.glsl_private_pointer_noop_rebinding_ids.add(id(value))
                        return
                    if (
                        id(value) in discarded_assignment_ids
                        and operator in {"+=", "-="}
                        and direct_parameter_target
                        and bounded_rebase_depth > 0
                        and conditional_rebase_depth == 0
                        and unsafe_rebase_depth == 0
                        and assigned_interval is not None
                        and not self.glsl_private_pointer_expression_has_side_effects(
                            assigned
                        )
                    ):
                        replacement = self.glsl_private_pointer_rebased_static_binding(
                            target_binding,
                            assigned_interval,
                            operator,
                        )
                        if replacement is None:
                            reject_pointer_rebinding(target_binding, value)
                        visit(assigned, active_intervals, active_bindings)
                        active_bindings[self.expression_name(target)] = replacement
                        self.glsl_private_pointer_base_rebinding_parameters[
                            id(value)
                        ] = target_binding["root"]
                        self.glsl_private_pointer_noop_rebinding_ids.discard(id(value))
                        return
                    reject_pointer_rebinding(target_binding, value)
                assigned_pointer_binding = pointer_value_binding(
                    assigned, active_bindings
                )
                if assigned_pointer_binding is not None:
                    reject_pointer_escape(assigned_pointer_binding, value)
                visit(target, active_intervals, active_bindings)
                visit(assigned, active_intervals, active_bindings)
                target_name = self.expression_name(target)
                if target_name and isinstance(
                    target, (str, IdentifierNode, VariableNode)
                ):
                    assigned_interval = self.glsl_private_pointer_interval(
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
            if isinstance(value, ReturnNode):
                returned = getattr(value, "value", None)
                returned_pointer_binding = pointer_value_binding(
                    returned, active_bindings
                )
                if returned_pointer_binding is not None:
                    reject_pointer_escape(returned_pointer_binding, value)
                visit(returned, active_intervals, active_bindings)
                return
            if isinstance(value, ArrayAccessNode):
                record_access(
                    value.array,
                    value.index,
                    active_intervals,
                    active_bindings,
                    value,
                )
                visit(value.array, active_intervals, active_bindings)
                visit(value.index, active_intervals, active_bindings)
                return
            if isinstance(value, UnaryOpNode):
                if value.op in {"++", "--"}:
                    operand_binding = pointer_parameter_binding(
                        value.operand, active_bindings
                    )
                    if operand_binding is not None:
                        reject_pointer_rebinding(operand_binding, value)
                if value.op == "*":
                    record_access(
                        value.operand,
                        0,
                        active_intervals,
                        active_bindings,
                        value,
                    )
                visit(value.operand, active_intervals, active_bindings)
                return
            if isinstance(value, PointerAccessNode):
                record_access(
                    value.pointer_expr,
                    0,
                    active_intervals,
                    active_bindings,
                    value,
                )
                visit(value.pointer_expr, active_intervals, active_bindings)
                return
            if isinstance(value, IfNode):
                condition_node = getattr(value, "condition", None)
                condition = self.glsl_private_pointer_condition_value(
                    condition_node, active_intervals, constants
                )
                visit(
                    condition_node,
                    active_intervals,
                    active_bindings,
                )
                else_if_conditions = list(
                    getattr(value, "else_if_conditions", []) or []
                )
                else_if_bodies = list(getattr(value, "else_if_bodies", []) or [])
                if condition is True:
                    visit(
                        getattr(value, "then_branch", None),
                        active_intervals,
                        active_bindings,
                    )
                    return
                if condition is False and not else_if_bodies:
                    visit(
                        getattr(value, "else_branch", None),
                        active_intervals,
                        active_bindings,
                    )
                    return
                branch_specs = []
                if condition is None:
                    branch_specs.append((None, getattr(value, "then_branch", None)))
                branch_specs.extend(zip(else_if_conditions, else_if_bodies))
                branch_specs.append((None, getattr(value, "else_branch", None)))
                branch_intervals = []
                conditional_rebase_depth += 1
                try:
                    for branch_condition, branch_body in branch_specs:
                        candidate_intervals = dict(active_intervals)
                        candidate_bindings = dict(active_bindings)
                        visit(
                            branch_condition,
                            candidate_intervals,
                            candidate_bindings,
                        )
                        visit(
                            branch_body,
                            candidate_intervals,
                            candidate_bindings,
                        )
                        branch_intervals.append(candidate_intervals)
                finally:
                    conditional_rebase_depth -= 1
                merged = branch_intervals[0]
                for candidate_intervals in branch_intervals[1:]:
                    merged = merge_intervals(merged, candidate_intervals)
                active_intervals.clear()
                active_intervals.update(merged)
                return
            if isinstance(value, ForNode):
                loop_intervals = dict(active_intervals)
                loop_bindings = dict(active_bindings)
                visit(getattr(value, "init", None), loop_intervals, loop_bindings)
                loop_bound_intervals = dict(loop_parameter_intervals)
                loop_bound_intervals.update(loop_intervals)
                if loop_contains_pointer_rebase(value):
                    iteration_values = (
                        self.glsl_private_pointer_rebase_loop_iteration_values(
                            value,
                            loop_bound_intervals,
                            constants,
                        )
                    )
                    if iteration_values:
                        loop_name = getattr(getattr(value, "init", None), "name", None)
                        entry_bindings = dict(loop_bindings)
                        binding_states = [dict(entry_bindings)]
                        bounded_rebase_depth += 1
                        try:
                            for iteration_value in iteration_values:
                                if loop_name:
                                    loop_intervals[loop_name] = (
                                        iteration_value,
                                        iteration_value,
                                    )
                                visit(
                                    getattr(value, "condition", None),
                                    loop_intervals,
                                    loop_bindings,
                                )
                                visit(
                                    getattr(value, "body", None),
                                    loop_intervals,
                                    loop_bindings,
                                )
                                visit(
                                    getattr(value, "update", None),
                                    loop_intervals,
                                    loop_bindings,
                                )
                                binding_states.append(dict(loop_bindings))
                        finally:
                            bounded_rebase_depth -= 1
                        init_name = getattr(getattr(value, "init", None), "name", None)
                        for name in active_bindings:
                            if name == init_name:
                                continue
                            merged = self.glsl_merge_private_pointer_static_bindings(
                                [state.get(name) for state in binding_states]
                            )
                            if merged is not None:
                                active_bindings[name] = merged
                        return
                has_unsafe_rebase = loop_contains_pointer_rebase(value)
                if has_unsafe_rebase:
                    unsafe_rebase_depth += 1
                try:
                    loop_name, loop_interval = self.glsl_private_pointer_loop_interval(
                        value,
                        loop_bound_intervals,
                        constants,
                    )
                    if loop_name:
                        if loop_interval is None:
                            loop_intervals.pop(loop_name, None)
                        else:
                            loop_intervals[loop_name] = loop_interval
                    visit(
                        getattr(value, "condition", None),
                        loop_intervals,
                        loop_bindings,
                    )
                    visit(getattr(value, "body", None), loop_intervals, loop_bindings)
                    visit(
                        getattr(value, "update", None),
                        loop_intervals,
                        loop_bindings,
                    )
                finally:
                    if has_unsafe_rebase:
                        unsafe_rebase_depth -= 1
                return
            if isinstance(value, ForInNode):
                loop_intervals = dict(active_intervals)
                loop_bindings = dict(active_bindings)
                loop_bound_intervals = dict(loop_parameter_intervals)
                loop_bound_intervals.update(loop_intervals)
                pattern_interval = self.glsl_for_in_pattern_interval(
                    value,
                    loop_bound_intervals,
                    constants,
                )
                pattern = getattr(value, "pattern", None)
                if pattern:
                    loop_intervals.pop(pattern, None)
                    self.glsl_clear_index_component_intervals(
                        loop_intervals,
                        pattern,
                    )
                    if pattern_interval is not None:
                        loop_intervals[pattern] = pattern_interval
                    loop_bindings[pattern] = {
                        "root_kind": None,
                        "root": pattern,
                        "offset": None,
                        "side_effecting": False,
                        "element_type": None,
                    }
                has_unsafe_rebase = loop_contains_pointer_rebase(value)
                if has_unsafe_rebase:
                    unsafe_rebase_depth += 1
                try:
                    visit(getattr(value, "body", None), loop_intervals, loop_bindings)
                finally:
                    if has_unsafe_rebase:
                        unsafe_rebase_depth -= 1
                body_mutations = self.glsl_function_mutated_interval_names(
                    SimpleNamespace(body=getattr(value, "body", None))
                )
                body_mutations.discard(pattern)
                for name in body_mutations:
                    active_intervals.pop(name, None)
                    self.glsl_clear_index_component_intervals(
                        active_intervals,
                        name,
                    )
                return
            if isinstance(value, (WhileNode, DoWhileNode, LoopNode)):
                loop_intervals = dict(active_intervals)
                loop_bindings = dict(active_bindings)
                has_unsafe_rebase = loop_contains_pointer_rebase(value)
                if has_unsafe_rebase:
                    unsafe_rebase_depth += 1
                try:
                    visit(
                        getattr(value, "condition", None),
                        loop_intervals,
                        loop_bindings,
                    )
                    visit(getattr(value, "body", None), loop_intervals, loop_bindings)
                finally:
                    if has_unsafe_rebase:
                        unsafe_rebase_depth -= 1
                return
            if isinstance(value, (SwitchNode, MatchNode)):
                has_unsafe_rebase = value_contains_pointer_rebase(value)
                if has_unsafe_rebase:
                    unsafe_rebase_depth += 1
                try:
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
                        visit(child, active_intervals, active_bindings)
                finally:
                    if has_unsafe_rebase:
                        unsafe_rebase_depth -= 1
                return
            if not hasattr(value, "__dict__"):
                return
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
                visit(child, active_intervals, active_bindings)

        visit(getattr(function, "body", []), intervals, pointer_bindings)

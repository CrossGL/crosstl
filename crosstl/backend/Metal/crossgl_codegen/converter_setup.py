"""Construction phases of ``MetalToCrossGLConverter``: option checks, the cooperative-matrix fragment mapping, and the initial per-translation state."""

from ....translator.cooperative_matrix import (
    has_cooperative_matrix_fragment_mapping,
)
from ....translator.host_workgroup_size import (
    normalize_threadgroup_size_option,
)
from ..MetalParser import (
    MetalSourceRefusal,
)
from ..preprocessor import (
    DEFAULT_EXPLICIT_TEMPLATE_SPECIALIZATION_LIMIT,
)


class MetalConverterSetupMixin:
    """Members of ``MetalToCrossGLConverter``: the phases ``__init__`` runs in order."""

    def configure_dispatch_and_profile_options(
        self,
        binary32_fma_profile,
        binary32_division_profile,
        binary16_remainder_profile,
        binary16_widening_profile,
        binary32_comparison_profile,
        binary32_atan2_profile,
        binary32_log_profile,
        binary32_power_operand_profile,
        binary32_power_accuracy_profile,
        floating_contraction_profile,
        preserve_resource_origins,
        threadgroup_size,
    ):
        """Check and store the threadgroup size, the resource-origin flag, and the contraction, power, FMA, division, binary16, comparison, atan2 and log profiles."""
        # The host picks a Metal kernel's threadgroup size at dispatch time; this option names it.
        self.threadgroup_size = normalize_threadgroup_size_option(threadgroup_size)
        if type(preserve_resource_origins) is not bool:
            raise MetalSourceRefusal("preserve_resource_origins must be a boolean")
        self.preserve_resource_origins = preserve_resource_origins
        if floating_contraction_profile not in (None, "rne-gradual", "rne-flush"):
            raise MetalSourceRefusal(
                "floating_contraction_profile must be 'rne-gradual', 'rne-flush', "
                "or None"
            )
        # Metal compiles with expression-level contraction even under
        # -fno-fast-math: a product feeding an addition in the same expression
        # becomes one fused operation. The profile selects how binary32
        # subnormals are treated by the portable single-rounding lowering.
        self.floating_contraction_profile = floating_contraction_profile
        if binary32_power_accuracy_profile not in (None, "portable-finite"):
            raise MetalSourceRefusal(
                "binary32_power_accuracy_profile must be 'portable-finite' or None"
            )
        self.binary32_power_accuracy_profile = binary32_power_accuracy_profile
        if binary32_power_operand_profile not in (None, "flush-subnormals"):
            raise MetalSourceRefusal(
                "binary32_power_operand_profile must be 'flush-subnormals' or None"
            )
        self.binary32_power_operand_profile = binary32_power_operand_profile
        if binary32_fma_profile not in (None, "rne-gradual", "rne-flush"):
            raise MetalSourceRefusal(
                "binary32_fma_profile must be 'rne-gradual', 'rne-flush', or None"
            )
        self.binary32_fma_profile = binary32_fma_profile
        if binary32_division_profile not in (None, "rne-gradual", "rne-flush"):
            raise MetalSourceRefusal(
                "binary32_division_profile must be 'rne-gradual', 'rne-flush', or None"
            )
        self.binary32_division_profile = binary32_division_profile
        if binary16_remainder_profile not in (None, "binary32-quotient"):
            raise MetalSourceRefusal(
                "binary16_remainder_profile must be 'binary32-quotient' or None"
            )
        self.binary16_remainder_profile = binary16_remainder_profile
        if binary16_widening_profile not in (None, "canonical-nan"):
            raise MetalSourceRefusal(
                "binary16_widening_profile must be 'canonical-nan' or None"
            )
        self.binary16_widening_profile = binary16_widening_profile
        if binary32_comparison_profile not in (
            None,
            "preserve-subnormals",
            "flush-subnormals",
        ):
            raise MetalSourceRefusal(
                "binary32_comparison_profile must be 'preserve-subnormals', "
                "'flush-subnormals', or None"
            )
        self.binary32_comparison_profile = binary32_comparison_profile
        if binary32_atan2_profile not in (
            None,
            "preserve-subnormals",
            "flush-subnormals",
        ):
            raise MetalSourceRefusal(
                "binary32_atan2_profile must be 'preserve-subnormals', "
                "'flush-subnormals', or None"
            )
        self.binary32_atan2_profile = binary32_atan2_profile
        if binary32_log_profile not in (
            None,
            "preserve-subnormals",
            "flush-subnormals",
        ):
            raise MetalSourceRefusal(
                "binary32_log_profile must be 'preserve-subnormals', "
                "'flush-subnormals', or None"
            )
        self.binary32_log_profile = binary32_log_profile

    def configure_log2_root_and_arithmetic_profiles(
        self,
        binary32_remainder_profile,
        binary32_additive_profile,
        binary32_multiplication_profile,
        binary32_log2_operand_profile,
        binary32_log2_accuracy_profile,
        binary32_sqrt_profile,
        binary32_rsqrt_profile,
    ):
        """Check and store the log2, square root, reciprocal square root, remainder, additive and multiplication profiles."""
        if binary32_log2_operand_profile not in (
            None,
            "preserve-subnormals",
            "flush-subnormals",
        ):
            raise MetalSourceRefusal(
                "binary32_log2_operand_profile must be 'preserve-subnormals', "
                "'flush-subnormals', or None"
            )
        if binary32_log2_accuracy_profile not in (None, "portable-finite"):
            raise MetalSourceRefusal(
                "binary32_log2_accuracy_profile must be 'portable-finite' or None"
            )
        self.binary32_log2_operand_profile = binary32_log2_operand_profile
        self.binary32_log2_accuracy_profile = binary32_log2_accuracy_profile
        if binary32_sqrt_profile not in (
            None,
            "preserve-subnormals",
            "flush-subnormals",
        ):
            raise MetalSourceRefusal(
                "binary32_sqrt_profile must be 'preserve-subnormals', "
                "'flush-subnormals', or None"
            )
        self.binary32_sqrt_profile = binary32_sqrt_profile
        if binary32_rsqrt_profile not in (
            None,
            "preserve-subnormals",
            "flush-subnormals",
        ):
            raise MetalSourceRefusal(
                "binary32_rsqrt_profile must be 'preserve-subnormals', "
                "'flush-subnormals', or None"
            )
        self.binary32_rsqrt_profile = binary32_rsqrt_profile
        if binary32_remainder_profile not in (
            None,
            "preserve-subnormals",
            "flush-arithmetic-subnormals",
        ):
            raise MetalSourceRefusal(
                "binary32_remainder_profile must be 'preserve-subnormals', "
                "'flush-arithmetic-subnormals', or None"
            )
        self.binary32_remainder_profile = binary32_remainder_profile
        if binary32_additive_profile not in (
            None,
            "rne-gradual",
            "rne-flush",
            "native-precise",
        ):
            raise MetalSourceRefusal(
                "binary32_additive_profile must be 'rne-gradual', 'rne-flush', "
                "'native-precise', or None"
            )
        self.binary32_additive_profile = binary32_additive_profile
        if binary32_multiplication_profile not in (None, "rne-gradual", "rne-flush"):
            raise MetalSourceRefusal(
                "binary32_multiplication_profile must be 'rne-gradual', 'rne-flush', or None"
            )
        self.binary32_multiplication_profile = binary32_multiplication_profile

    def configure_pointer_and_cooperative_matrix_options(
        self,
        cooperative_matrix_fragment_mapping,
        cooperative_matrix_fragment_mapping_provenance,
        preserve_pointer_pointee_const,
        resolve_standard_remove_cv_aliases,
    ):
        """Check and store the pointer-constness and ``remove_cv`` flags and the cooperative-matrix fragment mapping with its provenance."""
        if not isinstance(preserve_pointer_pointee_const, bool):
            raise MetalSourceRefusal("preserve_pointer_pointee_const must be a boolean")
        self.preserve_pointer_pointee_const = preserve_pointer_pointee_const
        if not isinstance(resolve_standard_remove_cv_aliases, bool):
            raise MetalSourceRefusal(
                "resolve_standard_remove_cv_aliases must be a boolean"
            )
        self.resolve_standard_remove_cv_aliases = resolve_standard_remove_cv_aliases
        mapping_is_configured = cooperative_matrix_fragment_mapping is not None
        provenance_is_configured = (
            cooperative_matrix_fragment_mapping_provenance is not None
        )
        if mapping_is_configured != provenance_is_configured:
            raise MetalSourceRefusal(
                "cooperative_matrix_fragment_mapping and "
                "cooperative_matrix_fragment_mapping_provenance must be "
                "configured together"
            )
        if mapping_is_configured:
            if (
                not isinstance(cooperative_matrix_fragment_mapping, str)
                or not cooperative_matrix_fragment_mapping.strip()
            ):
                raise MetalSourceRefusal(
                    "cooperative_matrix_fragment_mapping must be a non-empty string"
                )
            if (
                not isinstance(cooperative_matrix_fragment_mapping_provenance, str)
                or not cooperative_matrix_fragment_mapping_provenance.strip()
            ):
                raise MetalSourceRefusal(
                    "cooperative_matrix_fragment_mapping_provenance must be a "
                    "non-empty string"
                )
            cooperative_matrix_fragment_mapping = (
                cooperative_matrix_fragment_mapping.strip()
            )
            cooperative_matrix_fragment_mapping_provenance = (
                cooperative_matrix_fragment_mapping_provenance.strip()
            )
            if not has_cooperative_matrix_fragment_mapping(
                cooperative_matrix_fragment_mapping
            ):
                raise MetalSourceRefusal(
                    "Unknown cooperative-matrix fragment mapping profile "
                    f"'{cooperative_matrix_fragment_mapping}'"
                )

        self.cooperative_matrix_fragment_mapping = cooperative_matrix_fragment_mapping
        self.cooperative_matrix_fragment_mapping_provenance = (
            cooperative_matrix_fragment_mapping_provenance
        )

    def initialize_declaration_state(
        self,
    ):
        """Start empty tables of type aliases, alias templates, variables, structs, templates, the current function and constructors."""
        self.type_aliases = {}
        self.type_alias_qualifiers = {}
        self.type_alias_pointee_qualifiers = {}
        self.alias_template_declarations = {}
        self.alias_template_plain_declarations = {}
        self.scalar_alias_bindings = {}
        self.scalar_alias_names = set()
        self.parameter_type_contexts = {}
        self.type_alias_enum_declarations = {}
        self.alias_template_cache = {}
        self.alias_template_resolution_stack = []
        self.alias_template_structs = []
        self.alias_template_structs_by_qualified_name = {}
        self.alias_template_using_namespaces = []
        self.alias_template_using_declarations = []
        self.struct_member_alias_resolution_stack = []
        self.current_type_resolution_context = None
        self.callable_type_aliases = {}
        self.metal_later_declarations_visible = False
        self.callable_alias_declarations = {}
        self.global_variable_types = {}
        self.current_variable_types = {}
        self.global_variable_type_qualifiers = {}
        self.current_variable_type_qualifiers = {}
        self.metal_source_overload_groups = {}
        self.metal_source_overload_output_names = {}
        self.out_of_line_call_operator_replacements = {}
        self.suppressed_out_of_line_call_operator_ids = set()
        self.storage_texture_declaration_ids = set()
        self.comparison_sampler_ids = set()
        self.global_storage_texture_names = set()
        self.current_storage_texture_names = set()
        self.global_structured_buffer_names = set()
        self.current_structured_buffer_names = set()
        self.current_stage_entry_resource_parameter_ids = set()
        self.global_sampler_names = set()
        self.suppress_structured_buffer_index_lowering = False
        self.structured_buffer_lvalue_roots = set()
        self.struct_member_types = {}
        self.struct_member_name_maps = {}
        self.resolved_struct_member_types = {}
        self.resolved_struct_member_qualifiers = {}
        self.struct_declarations = {}
        self.struct_name_map = {}
        self.ambiguous_struct_names = set()
        self.nested_struct_types_by_owner = {}
        self.struct_static_constants = {}
        self.struct_static_constant_members = {}
        self.struct_static_constant_owner_candidates = {}
        self.equivalent_struct_static_constants = {}
        self.struct_static_constant_resolution_stack = []
        self.struct_static_constexpr_member_keys = set()
        self.current_struct_static_constant_owner = None
        self.struct_template_parameters = {}
        self.local_type_alias_names = set()
        self.unresolved_local_type_aliases = {}
        self.local_struct_type_aliases = {}
        self.local_integral_constant_bindings = {}
        self.global_integral_constant_bindings = {}
        self.local_aggregate_declarations = []
        self.local_aggregate_declaration_keys = {}
        self.local_aggregate_reserved_names = set()
        self.integral_constant_bindings = []
        self.template_value_bindings = []
        self.template_type_bindings = []
        self.template_binding_shadow_scopes = []
        self.value_template_functions = {}
        self.constexpr_value_template_functions = {}
        self.constexpr_helper_values = {}
        self.constexpr_helper_resolution_stack = []
        self.default_value_template_bindings = {}
        self.value_template_specializations = {}
        self.pending_value_template_specializations = []
        self.value_template_specialization_dependencies = {}
        self.current_function_specialization_key = None
        self.max_template_specializations = (
            DEFAULT_EXPLICIT_TEMPLATE_SPECIALIZATION_LIMIT
        )
        self.template_specialization_limit_source = "max_template_specializations"
        self.materialized_template_specialization_count = 0
        self.preserve_unmaterialized_template_calls = False
        self.suppressed_value_template_function_ids = set()
        self.metal_atomic_fence_transport_declaration_ids = set()
        self.current_function_name = None
        self.current_function_return_type = None
        self.current_function = None
        self.current_function_specialization = None
        self.current_function_materialization_bindings = {}
        self.materialized_constexpr_expression_contexts = []
        self.constructor_contracts_by_owner = {}
        self.constructor_factory_names = {}
        self.pending_constructor_factories = []
        self.current_constructor_scope_index = None

    def initialize_helper_state(
        self,
    ):
        """Start empty tables of enum, vector, precise-math, arithmetic-profile and cooperative-matrix helpers, and of identifier scopes."""
        self.metal_enum_arithmetic_types = {}
        self.metal_enum_member_types = {}
        self.small_vector_index_types = {}
        self.small_vector_index_operations = set()
        self.small_vector_resource_index_operations = {}
        self.wide_vector_types = {}
        self.wide_vector_binary_helpers = set()
        self.wide_vector_compound_helpers = set()
        self.wide_vector_reserved_names = set()
        self.metal_precise_math_helper_names = {}
        self.required_metal_precise_acos_widths = set()
        self.required_metal_precise_asin_widths = set()
        self.required_metal_precise_acosh_widths = set()
        self.required_metal_precise_atan_widths = set()
        self.required_metal_precise_atan2_widths = set()
        self.required_metal_power_widths = set()
        self.required_metal_precise_power_widths = set()
        self.required_metal_round_widths = set()
        self.required_metal_precise_exp_widths = set()
        self.required_metal_precise_log_widths = set()
        self.required_metal_log2_widths = set()
        self.required_metal_precise_sqrt_widths = set()
        self.required_metal_precise_rsqrt_widths = set()
        self.required_metal_precise_trig_widths = {"sin": set(), "cos": set()}
        self.required_metal_fma_widths = set()
        self.required_metal_fmuladd_widths = set()
        self.required_metal_division_widths = set()
        self.required_metal_half_remainder_widths = set()
        self.required_metal_half_widening_widths = set()
        self.required_metal_comparisons = set()
        self.required_metal_remainder_widths = set()
        self.required_metal_additive_operations = set()
        self.metal_additive_assignments = {}
        self.required_metal_multiplication_widths = set()
        self.metal_multiplication_assignments = {}
        self.metal_division_assignments = {}
        self.cooperative_matrix_fragment_helpers = {}
        self.storage_wrapper_read_helpers = {}
        self.cooperative_matrix_fragment_helper_names = set()
        self.cooperative_matrix_fragment_type_contracts = {}
        self.cooperative_matrix_fragment_type_replacements = {}
        self.identifier_maps = [{}]
        self.used_identifier_names = [set()]
        self.template_name_spellings = {}

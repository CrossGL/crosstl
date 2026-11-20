"""Preparation phases of ``GLSLCodeGen._generate_program_attempt``: AST lowering, the reset of per-program state, and the function, constant, struct, generic and identifier facts that emission reads."""

from ..vulkan_memory_scope import (
    declares_vulkan_memory_model,
)
from .array_utils import (
    collect_literal_int_constants,
    collect_struct_member_types,
)
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
)
from .generic_function_utils import (
    generic_function_parameters,
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
)
from .glsl_narrow_vectors import (
    narrow_scalars_promote,
)
from .glsl_stage_entries import (
    refuse_repeated_stage_entries,
)
from .glsl_union_storage import (
    UnionStorage,
)
from .image_access_contracts import (
    collect_function_image_access_requirements,
    collect_function_parameter_names,
)
from .private_vector_views import (
    lower_private_vector_views,
)
from .reference_aliases import (
    lower_reference_aliases,
)
from .resource_aggregates import (
    lower_resource_aggregates,
)
from .sampler_pairs import (
    COMBINED_SAMPLER_PAIRS,
    lower_combined_sampler_pairs,
)
from .stage_utils import (
    collect_stage_local_structs,
    deduplicate_named_declarations,
    requested_stage_name,
)
from .subgroup_control_flow import (
    converge_subgroup_bounded_loops,
    converge_subgroup_conditional_calls,
    converge_subgroup_divergent_barriers,
    converge_subgroup_early_exits,
    converge_subgroup_guarded_returns,
)


class OpenGLProgramPreparationMixin:
    """Preparation phases of ``GLSLCodeGen._generate_program_attempt``."""

    def lower_glsl_program_ast(
        self,
        ast,
        target_stage,
        early_exits,
    ):
        """Apply the AST lowerings that GLSL emission expects and set up software subgroup emulation."""
        ast = lower_resource_aggregates(
            ast,
            storage_pointer_parameters=True,
            workgroup_access_assertions=self.workgroup_access_assertions,
        )
        ast = lower_reference_aliases(ast)
        ast = lower_private_vector_views(ast, "opengl")
        ast, self.combined_sampler_records = lower_combined_sampler_pairs(
            ast, self.LOWERING_REFUSAL_TYPE.label
        )
        self.combined_sampler_pairs = (getattr(ast, "annotations", None) or {}).get(
            COMBINED_SAMPLER_PAIRS, []
        )
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
        return ast

    def reset_glsl_program_state(
        self,
        ast,
        target_stage,
    ):
        """Reset the per-program resource, alias, identifier and initializer state."""
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
        return target_stage

    def collect_glsl_function_and_constant_facts(
        self,
        ast,
        target_stage,
    ):
        """Collect function signatures, private-pointer state and compile-time constant values."""
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
        return functions

    def reset_glsl_stage_state(
        self,
        ast,
        target_stage,
    ):
        """Reset the stage interface, helper and software subgroup state for this program."""
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

    def collect_glsl_struct_and_generic_facts(
        self,
        ast,
        target_stage,
        functions,
    ):
        """Collect struct members and generic struct, enum and function specializations."""
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
        return structs, generic_function_specializations

    def collect_glsl_enum_and_specialization_facts(
        self,
        generic_function_specializations,
        structs,
    ):
        """Collect enum facts, specialization constants and the functions the shared lowering emits."""
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

    def prepare_glsl_program_identifiers(
        self,
        ast,
        target_stage,
        functions,
        generic_function_specializations,
        structs,
    ):
        """Choose identifier names and run the program-wide validations that precede emission."""
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

"""Preparation phases of ``HLSLCodeGen._generate_program_attempt``: AST lowering, the reset of per-program state, and the specialization-constant, struct, generic, enum and stage facts that emission reads."""

from ..ast import (
    StructNode,
)
from .array_utils import (
    collect_literal_int_constants,
    collect_struct_member_types,
)
from .buffer_length import (
    HlslBufferLength,
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
    prepare_generic_function_specializations,
)
from .generic_struct_utils import (
    collect_generic_struct_definitions,
    collect_generic_struct_specialization_member_types,
    collect_generic_struct_specializations,
)
from .glsl_buffer_layout import (
    collect_lowered_glsl_buffer_blocks,
)
from .helper_buffer_parameters import (
    lower_helper_buffer_parameters,
)
from .hlsl_array_value_arguments import (
    HlslArrayValueArguments,
)
from .hlsl_atomic_resources import (
    HLSLAtomicResources,
)
from .hlsl_byteaddress_typed_access import (
    hlsl_byteaddress_component_refusal,
)
from .hlsl_function_body_scans import (
    HlslFunctionBodyScans,
)
from .hlsl_groupshared_atomics import (
    hlsl_groupshared_variable_names,
)
from .hlsl_half_private_storage import (
    lower_half_private_storage,
)
from .hlsl_half_storage import (
    HLSLHalfStorage,
)
from .hlsl_matrix_orientation import (
    HLSLMatrixOrientation,
)
from .hlsl_mesh_semantics import (
    collect_hlsl_mesh_output_struct_offsets,
)
from .hlsl_program_functions import (
    HlslProgramFunctions,
)
from .hlsl_tall_matrix_rows import (
    HlslTallMatrixRows,
)
from .hlsl_texel_buffers import (
    HLSLTexelBuffers,
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
    canonicalize_separate_textures,
)
from .stage_utils import (
    collect_stage_local_structs,
    deduplicate_named_declarations,
    requested_stage_name,
    stage_matches,
)
from .storage_parameter_texels import (
    StorageParameterTexels,
)
from .subgroup_control_flow import (
    converge_subgroup_bounded_loops,
    converge_subgroup_conditional_calls,
    converge_subgroup_divergent_barriers,
    converge_subgroup_early_exits,
    converge_subgroup_guarded_returns,
)
from .union_lanes import (
    hlsl_lane_helpers,
)


class DirectXProgramPreparationMixin:
    """Preparation phases of ``HLSLCodeGen._generate_program_attempt``."""

    def lower_hlsl_program_ast(
        self,
        ast,
        target_stage,
        early_exits,
        owned,
    ):
        """Apply the AST lowerings that HLSL emission expects and set up software subgroup emulation."""
        self.hlsl_function_body_scans = HlslFunctionBodyScans(self)
        self.hlsl_program_functions = None
        ast = lower_resource_aggregates(
            ast,
            workgroup_access_assertions=getattr(
                self, "workgroup_access_assertions", ()
            ),
        )
        ast = lower_reference_aliases(ast, copy=not owned)
        ast = lower_private_vector_views(ast, "directx")
        ast = canonicalize_separate_textures(ast)
        # HLSL resource parameters cannot carry a caller's pointer rebase, so
        # Metal helpers whose pointer parameters merely carry [[buffer(n)]]
        # attributes take the storage-pointer offset ABI instead.
        ast = lower_helper_buffer_parameters(ast, self.stage_entry_types())
        self.hlsl_type_aliases = {
            node.name: getattr(node, "var_type", None)
            for node in getattr(ast, "global_variables", []) or []
            if getattr(node, "is_type_alias", False)
            and node.name not in self.METAL_TYPE_ALIAS_GLOBALS
        }
        # Typed binary16 loads, stores and selections may change NaN payloads.
        ast = lower_half_private_storage(ast, self.map_type)
        self.hlsl_groupshared_variable_names = hlsl_groupshared_variable_names(ast)
        target_stage = requested_stage_name(target_stage)
        self.directx_cooperative_matrix_lowerings = {}
        self.required_hlsl_cooperative_matrix_mma = False
        self.required_hlsl_cooperative_matrix_fragment_offset = False
        self.required_hlsl_cooperative_matrix_diagonals = {}
        self.hlsl_builtin_option_available = False
        ast = self.with_hlsl_builtin_option_prelude(ast)
        if self.software_subgroup_width is not None:
            ast = converge_subgroup_divergent_barriers(
                ast,
                self.hlsl_software_subgroup_stage_entries(ast, target_stage),
                self.walk_ast,
                self.map_operator,
            )
            if early_exits:
                ast = converge_subgroup_early_exits(
                    ast,
                    self.hlsl_software_subgroup_stage_entries(ast, target_stage),
                    self.walk_ast,
                    self.map_operator,
                )
            ast = converge_subgroup_guarded_returns(
                ast, self.walk_ast, self.map_operator
            )
            ast = converge_subgroup_bounded_loops(
                ast,
                self.hlsl_software_subgroup_stage_entries(ast, target_stage),
                self.walk_ast,
                self.map_operator,
            )
            ast = converge_subgroup_conditional_calls(
                ast,
                self.hlsl_software_subgroup_stage_entries(ast, target_stage),
                self.walk_ast,
                self.map_operator,
            )
        return ast, target_stage

    def reset_hlsl_program_state(
        self,
    ):
        """Reset the per-program type, resource, identifier and helper state."""

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
        self.function_parameter_types = {}
        self.hlsl_function_name_aliases = {}
        self.hlsl_function_overloads_by_name = {}
        self.hlsl_function_target_names = {}
        self.hlsl_function_target_names_by_signature = {}
        self.hlsl_mapped_overload_names = set()
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
        self.function_image_access_requirements = {}
        self.hlsl_pixel_only_feedback_function_names = {}
        self.hlsl_interpolation_function_names = {}
        self.hlsl_synchronization_function_names = {}
        self.hlsl_derivative_function_names = {}
        self.unsupported_glsl_buffer_block_functions = {}
        self.unsupported_glsl_buffer_block_struct_names = set()
        self.required_hlsl_inverse_helpers = set()
        self.required_hlsl_fragment_shading_rate_helper = False
        self.directx_resource_register_overrides = {}
        self.directx_resource_register_allocator = None
        self.directx_specialization_constant_records = []
        self.directx_specialization_constant_global_ids = set()
        self.directx_specialization_constant_types = {}
        self.directx_compile_time_global_node_ids = set()
        self.directx_omitted_compile_time_global_node_ids = set()
        self.current_identifier_aliases = {}
        self.current_identifier_reserved_names = set()
        self.current_function_name = None
        self.current_function_return_type = None
        self.current_hlsl_precise_return_type = None
        self.current_hlsl_parameter_epilogue = ()
        self.current_hlsl_tail_recursive_call_ids = set()
        self.current_hlsl_tail_recursive_function = None
        self.current_expression_expected_type = None
        self.hlsl_texture_offset_global_constants = {}
        self.hlsl_texture_offset_parameter_constants = {}
        self.current_hlsl_texture_offset_constants = None
        self.current_hlsl_stage_output_lowering = None
        self.current_hlsl_semantic_stage = None
        self.allow_hlsl_byteaddress_interlocked_member_expression = False
        self.local_variable_types = {}
        self.local_variable_source_types = {}
        self.global_variable_types = {}
        self.hlsl_union_layouts = {}
        self.hlsl_struct_buffer_resource_types = {}
        self.hlsl_promoted_entry_resource_parameter_names = {}
        self.hlsl_promoted_entry_constant_members = {}
        self.current_hlsl_resource_pointer_offsets = {}
        self.current_hlsl_resource_pointer_aliases = {}
        self.current_hlsl_bounded_loop_indices = {}
        self.current_hlsl_parameter_resource_types = {}
        self.current_global_resource_declaration_nodes = None
        self.standard_math_constant_shadow_names = set()
        self.hlsl_fixed_array_return_structs = {}
        self.hlsl_fixed_array_return_struct_nodes = {}
        self.hlsl_fixed_array_return_struct_metadata = {}
        self.required_hlsl_fixed_array_read_helpers = set()
        self.hlsl_struct_constructor_names = {}

    def reset_hlsl_stage_state(
        self,
        ast,
    ):
        """Reset the struct, stage interface and specialization state for this program."""
        self.hlsl_struct_constructor_reserved_names = {
            node.name
            for node in self.walk_ast(ast)
            if isinstance(getattr(node, "name", None), str)
        }
        self.required_hlsl_complex64_helpers = set()
        self.required_hlsl_bfloat16_helpers = set()
        self.hlsl_bfloat16_storage_resource_source_types = {}
        self.hlsl_half_storage = HLSLHalfStorage(self, ast)
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
        self.current_hlsl_available_functions = {}
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
        self.vertex_entry_output_struct_names = set()
        self.fragment_entry_input_struct_names = set()
        self.geometry_entry_vertex_struct_names = set()
        self.hlsl_mesh_output_struct_offsets = {}
        self.hlsl_temp_variable_index = 0
        self.current_glsl_buffer_block_parameters = {}
        self.unsupported_glsl_buffer_block_variables = set()
        self.unsupported_glsl_buffer_block_variable_types = {}
        self.current_unsupported_glsl_buffer_block_parameters = set()
        self.current_unsupported_glsl_buffer_block_local_variables = set()
        self.current_glsl_buffer_block_parameter_failures = {}
        self.current_glsl_buffer_block_parameter_struct_failures = {}
        self.directx_specialization_constant_records = (
            self.collect_directx_specialization_constants(ast)
        )

    def collect_hlsl_specialization_constant_facts(
        self,
        ast,
    ):
        """Collect the specialization constants and the texture-offset constants they feed."""
        self.directx_specialization_constant_global_ids = {
            id(record["node"])
            for record in self.directx_specialization_constant_records
            if record["is_global"]
        }
        self.directx_specialization_constant_types = {
            record["name"]: self.type_name_string(record["type"])
            for record in self.directx_specialization_constant_records
        }
        self.literal_int_constants = collect_literal_int_constants(
            getattr(ast, "constants", [])
        )
        for record in self.directx_specialization_constant_records:
            if record["mapped_type"] not in {"int", "uint"}:
                continue
            value = self.literal_int_value(record["value"], self.literal_int_constants)
            if value is not None:
                self.literal_int_constants[record["name"]] = value
        self.literal_int_constants.update(
            self.referenced_hlsl_glsl_builtin_int_constants(ast)
        )
        self.hlsl_texture_offset_global_constants = (
            self.collect_hlsl_texture_offset_global_constants(
                getattr(ast, "constants", [])
            )
        )
        self.literal_bool_constants = self.collect_hlsl_literal_bool_constants(
            getattr(ast, "constants", [])
        )
        for record in self.directx_specialization_constant_records:
            if record["mapped_type"] != "bool":
                continue
            value = self.hlsl_literal_bool_value(
                record["value"], self.literal_bool_constants
            )
            if value is not None:
                self.literal_bool_constants[record["name"]] = value
        self.current_hlsl_visible_int_constants = None

    def collect_hlsl_struct_and_enum_facts(
        self,
        ast,
        target_stage,
    ):
        """Collect structs, generic struct and enum specializations, enum facts and shadowed math constants."""
        structs = deduplicate_named_declarations(
            list(getattr(ast, "structs", []) or [])
            + collect_stage_local_structs(ast, target_stage),
            "struct",
        )
        self.hlsl_stage_output_lowerings = self.collect_hlsl_stage_output_lowerings(
            ast, target_stage, structs
        )
        if any(
            lowering.get("struct_name") is not None
            for lowering in self.hlsl_stage_output_lowerings.values()
        ):
            structs = [
                struct
                for struct in structs
                if not self.hlsl_should_omit_lowered_spirv_pervertex_struct(struct)
            ]
        structs = deduplicate_named_declarations(
            structs
            + [
                struct
                for lowering in self.hlsl_stage_output_lowerings.values()
                for struct in lowering["structs"]
            ],
            "struct",
        )
        self.structs_by_name = {
            node.name: node for node in structs if isinstance(node, StructNode)
        }
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
        self.standard_math_constant_shadow_names = {
            getattr(node, "name", None)
            for node in getattr(ast, "constants", []) or []
            if getattr(node, "name", None)
        }
        self.standard_math_constant_shadow_names.update(
            record["name"] for record in self.directx_specialization_constant_records
        )
        for import_node in getattr(ast, "imports", []) or []:
            alias = getattr(import_node, "alias", None)
            if alias:
                self.standard_math_constant_shadow_names.add(alias)
            self.standard_math_constant_shadow_names.update(
                item for item in getattr(import_node, "items", []) or [] if item
            )
        self.standard_math_constant_shadow_names.update(
            getattr(variant, "name", None)
            for enum in self.plain_enums + self.struct_payload_enums
            for variant in getattr(enum, "variants", []) or []
            if getattr(variant, "name", None)
        )
        cbuffers = self.hlsl_cbuffer_nodes(ast, target_stage)
        global_vars = self.global_resource_declaration_nodes(ast, target_stage)
        cbuffers = cbuffers + self.hlsl_stage_entry_constant_cbuffers(
            ast, target_stage, cbuffers, global_vars
        )
        self.current_global_resource_declaration_nodes = global_vars
        self.hlsl_struct_buffer_resource_types = (
            self.collect_hlsl_struct_buffer_resource_types(global_vars)
        )
        self.global_variable_types = self.collect_global_variable_types(global_vars)
        for constant in getattr(ast, "constants", []) or []:
            constant_type = self.type_name_string(getattr(constant, "const_type", None))
            if getattr(constant, "name", None) and constant_type:
                self.global_variable_types.setdefault(constant.name, constant_type)
        self.global_variable_types.update(self.directx_specialization_constant_types)
        self.global_variable_types.update(self.collect_cbuffer_member_types(cbuffers))
        functions = self.collect_functions(ast)
        self.hlsl_program_functions = HlslProgramFunctions(ast, functions)
        return structs, cbuffers, global_vars, functions

    def collect_hlsl_stage_struct_facts(
        self,
        ast,
        target_stage,
        cbuffers,
        functions,
        global_vars,
        structs,
    ):
        """Collect stage input and output structs, buffer-block structs and struct member types."""
        self.hlsl_compute_num_workgroups_uniform = (
            self.hlsl_compute_num_workgroups_uniform_names(
                ast, target_stage, cbuffers, global_vars, functions
            )
        )
        self.collect_hlsl_function_local_groupshared_declarations(ast, target_stage)
        self.hlsl_function_name_aliases = self.collect_hlsl_function_name_aliases(
            functions
        )
        self.prepare_hlsl_trailing_zero_helper_names(functions)
        self.hlsl_wide_integer_reserved_names = self.hlsl_helper_reserved_names(
            functions
        )
        self.prepare_hlsl_inverse_hyperbolic_helper_names(functions)
        self.prepare_hlsl_atan2_helper_names(functions)
        self.prepare_hlsl_half_helper_names(functions)
        self.prepare_hlsl_unsigned_float_helper_names(functions)
        self.prepare_hlsl_half_selection_helper_names(functions)
        self.prepare_hlsl_physical_subgroup_id_helper_names(functions)
        self.hlsl_float_atomic_reserved_names = self.hlsl_helper_reserved_names(
            functions
        )
        self.vertex_entry_output_struct_names = (
            self.collect_hlsl_vertex_entry_output_struct_names(ast, target_stage)
        )
        self.vertex_entry_output_struct_names.update(
            {
                lowering["struct_name"]
                for lowering in self.hlsl_stage_output_lowerings.values()
                if lowering["stage_name"] == "vertex"
                and lowering.get("struct_name") is not None
            }
        )
        self.vertex_entry_input_struct_names = (
            self.collect_hlsl_vertex_entry_input_struct_names(ast, target_stage)
        )
        self.vertex_entry_input_struct_names.update(
            {
                lowering["input_struct_name"]
                for lowering in self.hlsl_stage_output_lowerings.values()
                if lowering["stage_name"] == "vertex"
                and lowering.get("input_struct_name") is not None
            }
        )
        self.fragment_entry_input_struct_names = (
            self.collect_hlsl_fragment_entry_input_struct_names(ast, target_stage)
        )
        self.fragment_entry_output_struct_names = (
            self.collect_hlsl_fragment_entry_output_struct_names(ast, target_stage)
        )
        self.geometry_entry_vertex_struct_names = (
            self.collect_hlsl_geometry_entry_vertex_struct_names(ast, target_stage)
        )
        self.hlsl_mesh_output_struct_offsets = (
            collect_hlsl_mesh_output_struct_offsets(self, ast)
            if stage_matches(target_stage, "mesh")
            else {}
        )
        self.function_return_types = {
            func.name: self.hlsl_function_return_type_name(
                self.type_name_string(getattr(func, "return_type", "void"))
            )
            for func in functions
            if getattr(func, "name", None)
        }
        self.glsl_buffer_block_struct_names = (
            self.collect_glsl_buffer_block_struct_names(
                list(global_vars) + self.collect_function_parameters(functions)
            )
        )
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
            target_type_key="hlsl_type",
            unsupported_type_message=(
                "type is not supported by ByteAddressBuffer lowering"
            ),
            component_refusal=lambda info: hlsl_byteaddress_component_refusal(
                self, info
            ),
        )
        self.record_unsupported_glsl_buffer_block_layout_failures(
            global_vars,
            self.lowered_glsl_buffer_blocks,
            self.glsl_buffer_block_lowering_failures,
            self.glsl_buffer_block_struct_lowering_failures,
        )
        self.lowered_glsl_buffer_block_struct_names = {
            block["type_name"] for block in self.lowered_glsl_buffer_blocks.values()
        }
        self.unsupported_glsl_buffer_block_struct_names = set(
            self.glsl_buffer_block_struct_lowering_failures
        )
        self.struct_member_types = collect_struct_member_types(
            structs, self.type_name_string
        )
        self.remove_hlsl_static_struct_member_types(structs)
        self.hlsl_union_layouts = self.collect_hlsl_union_layouts(structs)
        self.hlsl_union_lanes = hlsl_lane_helpers(self)
        self.refresh_hlsl_fixed_array_return_struct_member_types()
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
        self.infer_hlsl_global_types(global_vars)
        self.prepare_directx_compile_time_globals(ast, global_vars)
        self.hlsl_lowered_struct_resource_members = (
            self.collect_hlsl_lowered_struct_resource_members(structs, functions)
        )

    def prepare_hlsl_generic_functions(
        self,
        ast,
        functions,
    ):
        """Prepare generic function specializations and the sampler struct members."""
        generic_function_specializations = prepare_generic_function_specializations(
            self,
            functions,
            global_types=self.global_variable_types,
            ast=ast,
        )
        if generic_function_specializations:
            specialized_functions = list(generic_function_specializations.values())
            self.collect_hlsl_generic_specializations_from_function_types(
                specialized_functions
            )
            self.generic_enum_specializations.update(
                collect_generic_enum_specializations(
                    specialized_functions,
                    self.generic_enum_struct_definitions,
                    self.type_name_string,
                )
            )
            self.generic_struct_specializations.update(
                collect_generic_struct_specializations(
                    specialized_functions,
                    self.generic_struct_definitions,
                    self.type_name_string,
                )
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
        self.function_return_types.update(
            {
                func.name: self.hlsl_function_return_type_name(
                    self.type_name_string(getattr(func, "return_type", "void"))
                )
                for func in generic_function_specializations.values()
            }
        )
        self.comparison_sampler_parameters = self.collect_comparison_sampler_parameters(
            ast
        )
        self.regular_sampler_parameters = self.collect_regular_sampler_parameters(ast)
        self.comparison_sampler_struct_members = self.collect_sampler_struct_members(
            ast,
            self.comparison_sampler_parameters,
            self.comparison_texture_function_names(),
        )
        self.regular_sampler_struct_members = self.collect_sampler_struct_members(
            ast,
            self.regular_sampler_parameters,
            self.regular_texture_function_names(),
        )
        self.function_parameter_names = collect_function_parameter_names(functions)
        self.function_parameter_types = self.collect_function_parameter_types(functions)
        self.function_overloads = self.collect_function_overloads(functions)
        return generic_function_specializations

    def prepare_hlsl_function_facts(
        self,
        ast,
        target_stage,
        functions,
        generic_function_specializations,
    ):
        """Prepare function overloads, fragment builtins and resource access requirements."""
        if generic_function_specializations:
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
            for name, overloads in self.collect_function_overloads(
                generic_function_specializations.values()
            ).items():
                self.function_overloads.setdefault(name, []).extend(overloads)
        resource_pointer_functions = list(functions)
        resource_pointer_functions.extend(generic_function_specializations.values())
        self.prepare_hlsl_mapped_function_overloads(
            resource_pointer_functions,
            reserved_names=self.hlsl_mapped_overload_reserved_names(ast),
        )
        self.collect_hlsl_resource_pointer_parameters(resource_pointer_functions)
        self.function_stage_parameter_dependencies = (
            self.collect_hlsl_function_stage_parameter_dependencies(ast, target_stage)
        )
        self.function_hlsl_fragment_builtin_dependencies = (
            self.collect_hlsl_function_fragment_builtin_dependencies(ast, target_stage)
        )
        self.function_hlsl_fragment_builtin_parameter_names = (
            self.collect_hlsl_fragment_builtin_parameter_names(
                ast,
                self.function_hlsl_fragment_builtin_dependencies,
            )
        )
        self.function_image_access_requirements = (
            collect_function_image_access_requirements(
                functions,
                self.function_parameter_names,
                self.walk_ast,
                self.function_call_name,
                self.expression_name,
            )
        )
        self.hlsl_pixel_only_feedback_function_names = (
            self.collect_hlsl_pixel_only_feedback_function_names(functions)
        )
        self.hlsl_interpolation_function_names = (
            self.collect_hlsl_interpolation_function_names(functions)
        )
        self.hlsl_synchronization_function_names = (
            self.collect_hlsl_synchronization_function_names(functions)
        )
        self.hlsl_derivative_function_names = (
            self.collect_hlsl_derivative_function_names(functions)
        )
        (
            self.resource_array_size_hints,
            self.function_resource_array_size_hints,
        ) = self.collect_resource_array_size_hints(ast)
        self.function_private_pointer_array_size_hints = (
            self.collect_private_pointer_array_size_hints(ast)
        )
        self.collect_hlsl_workgroup_pointer_variants(ast, target_stage)
        _, _, parameter_struct_failures = (
            self.collect_lowered_glsl_buffer_block_parameters(
                self.collect_function_parameters(functions)
            )
        )
        self.unsupported_glsl_buffer_block_struct_names.update(
            parameter_struct_failures
        )
        self.unsupported_glsl_buffer_block_functions = (
            self.collect_unsupported_glsl_buffer_block_functions(functions)
        )
        self.prepare_hlsl_software_subgroup_contract(ast, target_stage)
        self.validate_global_resource_shadows(ast)
        self.global_mixed_sampler_names = self.collect_global_mixed_sampler_names(ast)

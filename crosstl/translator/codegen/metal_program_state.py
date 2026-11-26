"""Per-program generator state that ``MetalCodeGen.render_rewritten_program`` sets up before it writes any code: the reset resource and helper tables, the function, type and global-variable tables, GLSL buffer-block layouts and struct member types, and generic function specializations with the dependency tables derived from them."""

from ..ast import (
    StructNode,
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
    metal_generic_global_types,
)
from .generic_struct_utils import (
    collect_generic_struct_definitions,
    collect_generic_struct_specialization_member_types,
    collect_generic_struct_specializations,
)
from .glsl_buffer_layout import (
    collect_lowered_glsl_buffer_blocks,
)
from .image_access_contracts import (
    collect_function_image_access_requirements,
    collect_function_parameter_names,
)
from .metal_array_arguments import (
    MetalArrayArguments,
)
from .metal_buffer_component_stores import (
    metal_glsl_buffer_component_refusal,
)
from .metal_buffer_resource_types import metal_generic_buffer_resource_types
from .metal_grid_quads import collect_metal_grid_quad_widths
from .metal_matrix_builtins import (
    MetalMatrixBuiltins,
)
from .metal_restrict import metal_restrict_suffix
from .metal_signed_builtins import (
    collect_metal_signed_builtin_members,
)
from .metal_specialization_constants import (
    block_member_array_types,
    flattened_stage_io_array_types,
    specialization_constants_sizing_arrays,
)
from .metal_texel_buffers import (
    MetalTexelBuffers,
)
from .metal_threadgroup_forwarding import (
    METAL_GROUPSHARED_ENTRY_STAGES,
    collect_metal_forwarded_threadgroup_storage,
)
from .stage_utils import (
    collect_stage_local_structs,
    collect_stage_local_variables,
    deduplicate_named_declarations,
)


class MetalProgramStateMixin:
    """State-preparation members of ``MetalCodeGen.render_rewritten_program``."""

    def reset_metal_program_resource_state(
        self,
        ast,
    ):
        """Reset the per-program resource, binding and stage-struct tables, and index the cbuffers."""
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

    def collect_metal_program_function_tables(
        self,
        ast,
    ):
        """Index the program's functions: overloads, parameters, return types, image access and cbuffer references."""
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
        return all_functions

    def reset_metal_program_parameter_state(
        self,
        ast,
    ):
        """Reset the texture, sampler, buffer-block, mesh and helper tables that function generation fills."""
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
        self.metal_grid_quad_widths = {}
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

    def collect_metal_program_type_tables(
        self,
        ast,
        target_stage,
    ):
        """Index the program's structs, stage-interface structs, generic definitions and enums."""
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
        return structs

    def collect_metal_program_global_tables(
        self,
        ast,
        target_stage,
        all_functions,
    ):
        """Collect the program-scope variables, lower threadgroup and thread-private globals, and record buffer-length dependencies."""
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
        return global_vars

    def reset_metal_program_generation_state(
        self,
    ):
        """Reset the current-function context and the helper requirements that generation records."""
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

    def collect_metal_buffer_block_layouts(
        self,
        ast,
        all_functions,
        global_vars,
        structs,
    ):
        """Lower GLSL buffer blocks, record their layout failures and specialization macros, and collect struct member types and address spaces."""
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
                block["restrict"] = metal_restrict_suffix(
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

    def prepare_metal_generic_function_specializations(
        self,
        ast,
        all_functions,
        global_vars,
        structs,
    ):
        """Specialize generic functions and collect the cbuffer, wave-lane, mesh-dispatch and stage-IO tables over all functions."""
        generic_function_specializations = prepare_generic_function_specializations(
            self,
            all_functions,
            {
                **getattr(self, "metal_program_scope_constant_types", {}),
                **getattr(self, "metal_cbuffer_member_types", {}),
                **metal_generic_global_types(self, global_vars),
                **metal_generic_buffer_resource_types(self, global_vars),
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
        self.metal_grid_quad_widths = collect_metal_grid_quad_widths(
            self, ast, wave_dependency_functions
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
        return generic_function_specializations

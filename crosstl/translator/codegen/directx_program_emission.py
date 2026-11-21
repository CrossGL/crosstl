"""Emission phases of ``HLSLCodeGen._generate_program_attempt``: the header and constants, groupshared and constant-buffer declarations, function bodies, stage entries and the final support passes."""

from ..ast import (
    FunctionCallNode,
)
from ..source_licenses import (
    source_license_comments,
)
from .bitfield_builtins import (
    emit_hlsl_bitfield_helpers,
)
from .constant_ordering import (
    partition_constants_by_struct_dependency,
)
from .enum_utils import (
    generate_enum_constants,
    generate_enum_constructor_functions,
    generate_generic_enum_constants,
    generate_generic_enum_constructor_functions,
)
from .extended_arithmetic_builtins import (
    emit_hlsl_extended_arithmetic_helpers,
)
from .generic_emission_order import (
    mark_generic_function_emitted,
)
from .generic_function_utils import (
    generic_function_emission_list,
    generic_function_parameters,
)
from .hlsl_byte_address_atomics import (
    emit_hlsl_byte_address_atomic_helpers,
)
from .hlsl_exact_float_builtins import (
    emit_hlsl_exact_float_helpers,
)
from .hlsl_ray_queries import (
    hlsl_ray_query_helpers,
)
from .hlsl_wave_extended import (
    emit_hlsl_wave_extension_helpers,
)
from .pack_builtins import (
    emit_hlsl_pack_helpers,
)
from .stage_utils import (
    collect_stage_local_variables,
    function_entry_execution_config,
    function_stage_name,
    normalize_stage_name,
    order_functions_by_dependencies,
    should_emit_qualified_function,
    stage_matches,
)


class DirectXProgramEmissionMixin:
    """Emission phases of ``HLSLCodeGen._generate_program_attempt``."""

    def generate_hlsl_program_header(
        self,
        ast,
        global_vars,
        structs,
    ):
        """Emit preprocessor lines, enum constants, global constants and the generic enum constructors."""
        self.global_mixed_sampler_aliases = {
            name: self.hlsl_comparison_sampler_alias_name(name)
            for name in self.global_mixed_sampler_names
        }
        self.validate_explicit_sampler_role_conflicts(ast)
        self.hlsl_program_functions = None
        code = "\n"
        code += source_license_comments(ast, "directx")
        preprocessors = getattr(ast, "preprocessors", []) or []
        for directive in preprocessors:
            line = self.generate_preprocessor_directive(directive)
            if line:
                code += f"{line}\n"
        if self.hlsl_requires_rwtexture_cube_alias(global_vars):
            code += "#define RWTextureCube RWTexture2DArray\n"
        half_helper_offset = len(code)

        code += generate_enum_constants(
            self, self.plain_enums + self.struct_payload_enums
        )
        code += generate_generic_enum_constants(
            self,
            self.generic_enum_struct_definitions,
        )
        leading_constants, struct_dependent_constants = (
            partition_constants_by_struct_dependency(
                getattr(ast, "constants", []) or [], structs
            )
        )
        code += self.generate_constants(ast, leading_constants)
        code += self.generate_directx_specialization_constant_globals()
        code += self.generate_hlsl_glsl_builtin_constant_declarations(ast)
        code += self.generate_hlsl_ordered_struct_declarations(structs)
        storage_helper_offset = len(code)
        code += self.generate_constants(ast, struct_dependent_constants)
        code += generate_enum_constructor_functions(self, self.struct_payload_enums)
        code += generate_generic_enum_constructor_functions(
            self,
            self.generic_enum_specializations,
        )
        return code, half_helper_offset, storage_helper_offset

    def collect_hlsl_global_sampler_facts(
        self,
        ast,
        global_vars,
    ):
        """Collect the comparison, explicit and implicit sampler names that the global declarations need."""

        self.hlsl_atomic_resources.prepare(ast)
        self.storage_parameter_texels.prepare(ast)
        comparison_texture_names, comparison_sampler_names = (
            self.collect_comparison_resources(ast)
        )
        self.implicit_texture_sampler_parameters = (
            self.collect_implicit_texture_sampler_parameters(ast)
        )
        comparison_sampler_names |= self.collect_comparison_sampler_arguments(
            ast, self.comparison_sampler_parameters
        )
        comparison_texture_names |= self.collect_implicit_comparison_texture_arguments(
            ast, self.implicit_texture_sampler_parameters
        )
        sampler_parameter_names = self.collect_sampler_parameter_names(ast)
        declared_sampler_names = set()
        explicit_sampler_names = set()
        for node in global_vars:
            var_type = getattr(node, "var_type", getattr(node, "vtype", "float"))
            var_name = getattr(node, "name", getattr(node, "variable_name", None))
            if var_name and self.is_sampler_type(var_type):
                declared_sampler_names.add(var_name)
                explicit_sampler_names.add(var_name)
        comparison_sampler_names |= {
            f"{texture_name}Sampler"
            for texture_name in comparison_texture_names
            if f"{texture_name}Sampler" in declared_sampler_names
        }
        global_implicit_sampler_texture_names = (
            self.collect_global_implicit_sampler_texture_names(
                ast,
                self.collect_global_texture_names(ast),
                declared_sampler_names | sampler_parameter_names,
                self.implicit_texture_sampler_parameters,
            )
        )
        global_implicit_regular_sampler_texture_names = (
            self.collect_global_implicit_regular_sampler_texture_names(
                ast,
                self.collect_global_texture_names(ast),
                declared_sampler_names | sampler_parameter_names,
                self.implicit_texture_sampler_parameters,
            )
        )
        global_implicit_query_lod_texture_names = (
            self.collect_global_implicit_query_lod_texture_names(
                ast,
                self.collect_global_texture_names(ast),
                declared_sampler_names | sampler_parameter_names,
                self.implicit_texture_sampler_parameters,
            )
        )
        implicit_member_samplers = self.collect_implicit_member_texture_samplers(ast)
        return (
            comparison_sampler_names,
            comparison_texture_names,
            explicit_sampler_names,
            global_implicit_sampler_texture_names,
            global_implicit_regular_sampler_texture_names,
            global_implicit_query_lod_texture_names,
            implicit_member_samplers,
        )

    def generate_hlsl_global_storage(
        self,
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
    ):
        """Emit implicit samplers, lowered struct resources, groupshared variables and constant buffers."""
        code += self.generate_implicit_member_texture_samplers(
            implicit_member_samplers,
            used_resource_registers,
            sampler_registers,
        )
        code += self.generate_hlsl_lowered_struct_resource_member_globals(
            used_resource_registers,
            texture_registers,
            sampler_registers,
            uav_registers,
        )

        groupshared_vars = self.deduplicate_hlsl_groupshared_declarations(
            collect_stage_local_variables(
                ast,
                target_stage,
                self.hlsl_declaration_has_groupshared_qualifier,
            ),
        )
        for node, alias in self.hlsl_groupshared_declarations_to_emit(groupshared_vars):
            code += self.generate_hlsl_hoisted_groupshared_declaration(node, alias)
        for node in groupshared_vars:
            # Stage-scope group-shared variables are visible to every function of the stage.
            declared_type = self.type_name_string(
                self.hlsl_groupshared_declaration_type(node)
            )
            if getattr(node, "name", None) and declared_type:
                self.global_variable_types.setdefault(node.name, declared_type)

        if cbuffers:
            code += "// Constant Buffers\n"
            code += self.generate_cbuffers(
                ast,
                cbuffers,
                target_stage,
                used_resource_registers,
                texture_registers,
                cbuffer_registers,
            )
        compute_num_workgroups_cbuffer = (
            self.generate_hlsl_compute_num_workgroups_cbuffer(
                used_resource_registers,
                cbuffer_registers,
            )
        )
        if compute_num_workgroups_cbuffer:
            if not cbuffers:
                code += "// Constant Buffers\n"
            code += compute_num_workgroups_cbuffer

        stage_entry_names = self.stage_entry_names(ast, target_stage)
        self.current_hlsl_hull_output_control_points = (
            self.hlsl_program_hull_output_control_points(ast)
        )
        self.current_hlsl_hull_output_element_type = (
            self.hlsl_program_hull_output_element_type(ast)
        )
        self.current_hlsl_hull_domain = self.hlsl_program_hull_domain(ast)
        self.current_hlsl_mesh_payload_types = self.hlsl_program_mesh_payload_types(ast)
        self.current_hlsl_dispatch_mesh_payload_types = (
            self.hlsl_program_dispatch_mesh_payload_types(ast)
        )
        self.current_hlsl_has_amplification_stage = (
            self.hlsl_program_has_amplification_stage(ast)
        )

        self.hlsl_texture_offset_parameter_constants = (
            self.collect_hlsl_texture_offset_parameter_constants(ast, target_stage)
        )
        functions = getattr(ast, "functions", [])
        global_functions_by_name = {
            func.name: func
            for func in functions
            if getattr(func, "name", None)
            and func.name not in self.hlsl_omitted_function_names
        }
        self.current_hlsl_available_functions = global_functions_by_name
        functions_code = ""
        function_declaration_sources = []
        ordered_functions = order_functions_by_dependencies(
            functions,
            self.walk_ast,
            self.hlsl_dependency_function_call_name,
            FunctionCallNode,
            include_overloads=True,
        )
        return (
            code,
            stage_entry_names,
            global_functions_by_name,
            functions_code,
            function_declaration_sources,
            ordered_functions,
        )

    def generate_hlsl_program_functions(
        self,
        target_stage,
        function_declaration_sources,
        functions_code,
        ordered_functions,
        stage_entry_names,
    ):
        """Emit the program's function definitions."""
        for func in ordered_functions:
            if getattr(func, "name", None) in self.hlsl_omitted_function_names:
                continue
            qualifier_name = function_stage_name(func)

            if not should_emit_qualified_function(target_stage, qualifier_name):
                continue

            if generic_function_parameters(func):
                mark_generic_function_emitted(self, func)
                specialized_functions = generic_function_emission_list(self, func)
                for specialized_func in specialized_functions:
                    if (
                        getattr(specialized_func, "name", None)
                        in self.hlsl_omitted_function_names
                    ):
                        continue
                    function_declaration_sources.append(specialized_func)
                    functions_code += self.generate_function(specialized_func)
                continue

            if qualifier_name == "vertex":
                functions_code += self.generate_function(
                    func,
                    shader_type="vertex",
                    entry_name=stage_entry_names.get(id(func)),
                )
            elif qualifier_name == "fragment":
                functions_code += self.generate_function(
                    func,
                    shader_type="fragment",
                    entry_name=stage_entry_names.get(id(func)),
                )
            elif qualifier_name == "compute":
                functions_code += self.generate_function(
                    func,
                    shader_type="compute",
                    execution_config=function_entry_execution_config(func),
                    entry_name=stage_entry_names.get(id(func)),
                )
            else:
                functions_code += self.generate_function(
                    func,
                    entry_name=stage_entry_names.get(id(func)),
                )
                if qualifier_name is None:
                    function_declaration_sources.append(func)
        return functions_code

    def generate_hlsl_program_stages(
        self,
        ast,
        target_stage,
        function_declaration_sources,
        functions_code,
        global_functions_by_name,
        stage_entry_names,
    ):
        """Emit the stage entry functions."""
        if hasattr(ast, "stages") and ast.stages:
            for stage_type, stage in ast.stages.items():
                stage_name = normalize_stage_name(stage_type)
                if not stage_matches(target_stage, stage_name):
                    continue

                local_functions = getattr(stage, "local_functions", []) or []
                stage_functions_by_name = dict(global_functions_by_name)
                stage_functions_by_name.update(
                    {
                        func.name: func
                        for func in local_functions
                        if getattr(func, "name", None)
                        and func.name not in self.hlsl_omitted_function_names
                    }
                )

                for func in order_functions_by_dependencies(
                    local_functions,
                    self.walk_ast,
                    self.hlsl_dependency_function_call_name,
                    FunctionCallNode,
                    include_overloads=True,
                ):
                    if getattr(func, "name", None) in self.hlsl_omitted_function_names:
                        continue
                    if generic_function_parameters(func):
                        mark_generic_function_emitted(self, func)
                        specialized_functions = generic_function_emission_list(
                            self, func
                        )
                        for specialized_func in specialized_functions:
                            if (
                                getattr(specialized_func, "name", None)
                                in self.hlsl_omitted_function_names
                            ):
                                continue
                            function_declaration_sources.append(specialized_func)
                            functions_code += self.generate_function(specialized_func)
                    else:
                        function_declaration_sources.append(func)
                        functions_code += self.generate_function(func)

                if hasattr(stage, "entry_point"):
                    previous_available_functions = self.current_hlsl_available_functions
                    self.current_hlsl_available_functions = {
                        name: func
                        for name, func in stage_functions_by_name.items()
                        if func is not stage.entry_point
                    }
                    try:
                        functions_code += self.generate_function(
                            stage.entry_point,
                            shader_type=stage_name,
                            execution_config=getattr(stage, "execution_config", None),
                            entry_name=stage_entry_names.get(id(stage.entry_point)),
                            stage_output_lowering=self.hlsl_stage_output_lowerings.get(
                                id(stage)
                            ),
                            stage_node=stage,
                        )
                    finally:
                        self.current_hlsl_available_functions = (
                            previous_available_functions
                        )
        return functions_code

    def finish_hlsl_program(
        self,
        code,
        function_declaration_sources,
        functions_code,
        half_helper_offset,
        storage_helper_offset,
    ):
        """Insert generated helpers and apply the final support passes."""

        function_declarations_code = self.generate_hlsl_function_declarations(
            function_declaration_sources
        )
        code += self.generate_texture_query_helpers()
        code += self.hlsl_texel_buffers.code()
        code += self.hlsl_buffer_length.code()
        code += self.generate_image_atomic_helpers()
        code += self.generate_byteaddress_atomic_helpers()
        code += emit_hlsl_byte_address_atomic_helpers(self)
        code += self.generate_hlsl_float_atomic_helpers()
        code += self.generate_hlsl_expected_compare_helpers()
        code += self.generate_glsl_buffer_aggregate_load_helpers()
        code += self.generate_hlsl_inverse_helpers()
        code += self.hlsl_matrix_orientation.helper_code()
        code += self.generate_hlsl_fragment_shading_rate_helper()
        code += self.generate_hlsl_wave_shuffle_and_fill_up_helpers()
        code += emit_hlsl_wave_extension_helpers(self)
        code += emit_hlsl_pack_helpers(self)
        code += emit_hlsl_exact_float_helpers(self)
        code += emit_hlsl_bitfield_helpers(self)
        code += emit_hlsl_extended_arithmetic_helpers(self)
        code += self.generate_hlsl_trailing_zero_helpers()
        code += self.generate_hlsl_wide_integer_compound_helpers()
        code += self.generate_hlsl_inverse_hyperbolic_helpers()
        code += self.generate_hlsl_precise_hyperbolic_helpers()
        code += self.generate_hlsl_precise_difference_helpers()
        code += self.generate_hlsl_atan2_helpers()
        code += self.generate_hlsl_complex64_helpers()
        code += self.generate_hlsl_software_subgroup_helpers()
        code += self.generate_hlsl_defined_relative_wave_shuffle_helpers()
        code += self.generate_hlsl_physical_subgroup_id_helper()
        code += self.generate_hlsl_bfloat16_helpers()
        code += self.generate_hlsl_union_storage_helpers()
        code += self.generate_hlsl_fixed_array_return_helpers()
        code += self.generate_hlsl_private_pointer_word_view_helpers()
        code += self.generate_hlsl_resource_pointer_postincrement_helpers()
        code += function_declarations_code
        code += functions_code

        code = (
            code[:storage_helper_offset]
            + self.hlsl_half_storage.declarations()
            + self.hlsl_array_value_arguments.declarations()
            + self.generate_hlsl_struct_constructor_helpers()
            + self.hlsl_tall_matrix_rows.declarations()
            + hlsl_ray_query_helpers(self)
            + code[storage_helper_offset:]
        )
        return (
            self.hlsl_half_storage.header()
            + code[:half_helper_offset]
            + self.generate_hlsl_explicit_bitcast_helpers()
            + self.generate_hlsl_half_helpers()
            + self.generate_hlsl_unsigned_float_helpers()
            + self.generate_hlsl_half_selection_helpers()
            + code[half_helper_offset:]
        )

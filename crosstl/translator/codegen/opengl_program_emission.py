"""Emission phases of ``GLSLCodeGen._generate_program_attempt``: the version and extension header, global classification, interface declarations, function bodies, stage entries and the final support passes."""

import re

from ..ast import (
    ArrayNode,
    FunctionCallNode,
    PreprocessorNode,
    StructNode,
)
from ..source_licenses import (
    source_license_comments,
)
from .constant_ordering import (
    partition_constants_by_struct_dependency,
)
from .exponent_builtins import (
    emit_glsl_exponent_helpers,
)
from .generic_emission_order import (
    mark_generic_function_emitted,
)
from .generic_function_utils import (
    generic_function_emission_list,
    generic_function_parameters,
)
from .glsl_builtin_extensions import (
    require_builtin_extensions,
)
from .glsl_entry_resource_names import (
    glsl_entry_resource_parameter_names,
)
from .glsl_helper_stage_guards import (
    glsl_helper_stage_macros,
    wrap_glsl_helper_stage_guard,
)
from .glsl_refusals import (
    input_attachment_refusal_reason,
)
from .glsl_shadow_lod import (
    require_shadow_lod_extension,
)
from .glsl_specialization_dependents import (
    constant_source_name,
    partition_constants_by_specialization_dependency,
)
from .glsl_vulkan_descriptor_sets import (
    descriptor_binding_macro_definition,
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


class OpenGLProgramEmissionMixin:
    """Emission phases of ``GLSLCodeGen._generate_program_attempt``."""

    def generate_glsl_program_header(
        self,
        ast,
        target_stage,
        structs,
    ):
        """Emit the version line, extensions, enum constants, global constants and wave helpers."""
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
        return (
            code,
            cooperative_matrix_support_insertion_index,
            struct_dependent_constants,
        )

    def classify_glsl_program_globals(
        self,
        ast,
        target_stage,
        structs,
    ):
        """Split the global variables into resource, interface, runtime-initialized and specialization groups."""
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
        return (
            global_vars,
            empty_struct_global_ids,
            runtime_initialized_global_vars,
            stage_local_resource_vars,
            stage_local_interface_vars,
            resource_declaration_nodes,
        )

    def collect_glsl_stage_interface_facts(
        self,
        ast,
        target_stage,
        code,
        functions,
        global_vars,
        resource_declaration_nodes,
        runtime_initialized_global_vars,
        structs,
    ):
        """Collect buffer-block, interface-block and stage input and output struct facts."""
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
        return (
            emit_vertex_io,
            emit_fragment_io,
            emit_graphics_io,
            legacy_vertex_names,
            legacy_fragment_names,
            data_struct_nodes,
            stage_io_code,
            code,
        )

    def generate_glsl_interface_and_prototypes(
        self,
        ast,
        target_stage,
        code,
        global_vars,
        stage_local_interface_vars,
        stage_local_resource_vars,
    ):
        """Emit stage-local interface variables, cbuffers, hoisted shared declarations and function prototypes."""
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
        return (
            combined_stage_entry_names,
            generated_helper_insertion_index,
            functions,
            helper_stage_macros,
            deferred_top_level_helpers_by_stage,
            deferred_top_level_helper_ids,
            called_function_names,
            code,
        )

    def generate_glsl_program_functions(
        self,
        target_stage,
        called_function_names,
        code,
        combined_stage_entry_names,
        deferred_top_level_helper_ids,
        functions,
        helper_stage_macros,
    ):
        """Emit the program's function definitions."""
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
        return code

    def generate_glsl_program_stages(
        self,
        ast,
        target_stage,
        called_function_names,
        code,
        combined_stage_entry_names,
        deferred_top_level_helpers_by_stage,
    ):
        """Emit the stage entry functions."""
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
        return code

    def finish_glsl_program(
        self,
        code,
        cooperative_matrix_support_insertion_index,
        generated_helper_insertion_index,
    ):
        """Insert generated helpers and support declarations, and apply the final version and extension passes."""
        generated_helpers = (
            self.generate_glsl_software_subgroup_support()
            + self.generate_glsl_trailing_zero_helpers()
            + self.generate_glsl_boolean_order_helpers()
            + self.generate_glsl_metal_math_helpers()
            + emit_glsl_exponent_helpers(self)
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
        if self.glsl_descriptor_binding_macro_used:
            code = (
                code[:cooperative_matrix_support_insertion_index]
                + descriptor_binding_macro_definition()
                + code[cooperative_matrix_support_insertion_index:]
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

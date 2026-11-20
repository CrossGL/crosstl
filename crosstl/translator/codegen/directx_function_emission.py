"""Phases of ``HLSLCodeGen.generate_function``: pointer and resource-access specializations, the per-function state reset, entry-parameter facts, required built-in parameters, return types, the stage header and the body."""

from ..ast import (
    PointerType,
)
from ..refusals import (
    DirectXRefusal,
)
from ..unhandled import (
    UNHANDLED,
)
from .constant_guards import (
    constant_branch_guards,
)
from .directx_function_state import (
    HLSL_FUNCTION_RESOURCE_PARAMETER_ATTRIBUTES,
    restore_hlsl_function_state,
    save_hlsl_function_state,
)
from .stage_utils import (
    normalize_stage_name,
    requests_early_fragment_tests,
)


class HLSLFunctionEmissionMixin:
    """Function-emission phases of ``HLSLCodeGen.generate_function``."""

    def generate_hlsl_function_variants(
        self,
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
    ):
        """Render one specialization per private-pointer, workgroup-pointer or resource-access variant of a function, or return ``UNHANDLED`` when the call already renders a single variant."""
        variants = self.function_private_pointer_backing_variants.get(function_name)
        if _private_pointer_variant is None and variants:
            previous_variant = self.current_hlsl_private_pointer_variant
            previous_base_names = self.current_hlsl_private_pointer_base_names
            parameters = self.function_private_pointer_parameters[function_name]
            code = ""
            try:
                for variant in variants:
                    self.current_hlsl_private_pointer_variant = {
                        parameter.name: variant[index]
                        for index, parameter in enumerate(parameters)
                    }
                    self.current_hlsl_private_pointer_base_names = dict(
                        self.function_private_pointer_base_names[function_name]
                    )
                    code += self.generate_function(
                        func,
                        indent=indent,
                        shader_type=shader_type,
                        execution_config=execution_config,
                        entry_name=entry_name,
                        stage_output_lowering=stage_output_lowering,
                        stage_node=stage_node,
                        _private_pointer_variant=variant,
                        _workgroup_pointer_variant=_workgroup_pointer_variant,
                    )
            finally:
                self.current_hlsl_private_pointer_variant = previous_variant
                self.current_hlsl_private_pointer_base_names = previous_base_names
            return code

        workgroup_variants = self.function_hlsl_workgroup_pointer_backing_variants.get(
            function_name
        )
        if _workgroup_pointer_variant is None and workgroup_variants:
            previous_variant = self.current_hlsl_workgroup_pointer_variant
            previous_base_names = self.current_hlsl_workgroup_pointer_base_names
            parameters = self.function_hlsl_workgroup_pointer_parameters[function_name]
            code = ""
            try:
                for variant in workgroup_variants:
                    self.current_hlsl_workgroup_pointer_variant = {
                        parameter.name: variant[index]
                        for index, parameter in enumerate(parameters)
                    }
                    self.current_hlsl_workgroup_pointer_base_names = dict(
                        self.function_hlsl_workgroup_pointer_base_names[function_name]
                    )
                    code += self.generate_function(
                        func,
                        indent=indent,
                        shader_type=shader_type,
                        execution_config=execution_config,
                        entry_name=entry_name,
                        stage_output_lowering=stage_output_lowering,
                        stage_node=stage_node,
                        _private_pointer_variant=_private_pointer_variant,
                        _workgroup_pointer_variant=variant,
                    )
            finally:
                self.current_hlsl_workgroup_pointer_variant = previous_variant
                self.current_hlsl_workgroup_pointer_base_names = previous_base_names
            return code

        access_variants = self.hlsl_resource_access_function_variants(func)
        if _resource_access_variant is None and access_variants:
            previous_promotions = self.current_hlsl_resource_access_promotions
            code = ""
            try:
                for variant in (frozenset(), *access_variants):
                    self.current_hlsl_resource_access_promotions = (
                        self.hlsl_resource_access_promotions_for(func, variant)
                    )
                    code += self.generate_function(
                        func,
                        indent=indent,
                        shader_type=shader_type,
                        execution_config=execution_config,
                        entry_name=entry_name,
                        stage_output_lowering=stage_output_lowering,
                        stage_node=stage_node,
                        _private_pointer_variant=_private_pointer_variant,
                        _workgroup_pointer_variant=_workgroup_pointer_variant,
                        _resource_access_variant=variant,
                    )
            finally:
                self.current_hlsl_resource_access_promotions = previous_promotions
            return code
        return UNHANDLED

    def reset_hlsl_function_emission_state(
        self,
        func,
        entry_name,
        stage_output_lowering,
        param_list,
    ):
        """Rebind the per-function buffer-block, alias, local-type, resource-pointer and stage-output state before a function is emitted."""
        (
            self.current_glsl_buffer_block_parameters,
            self.current_glsl_buffer_block_parameter_failures,
            self.current_glsl_buffer_block_parameter_struct_failures,
        ) = self.collect_lowered_glsl_buffer_block_parameters(param_list)
        self.current_unsupported_glsl_buffer_block_parameters = (
            self.collect_unsupported_glsl_buffer_block_parameter_names(param_list)
        )
        self.current_unsupported_glsl_buffer_block_local_variables = set()
        self.current_generic_function_substitutions = (
            getattr(func, "_generic_substitutions", {}) or {}
        )
        self.current_function_name = getattr(func, "name", None) or entry_name
        self.current_hlsl_tail_recursive_call_ids = set()
        self.current_hlsl_tail_recursive_function = None
        self.local_variable_types = {}
        self.local_variable_source_types = {}
        self.current_identifier_aliases = {}
        self.current_hlsl_resource_pointer_offsets = {}
        self.current_hlsl_resource_pointer_aliases = {
            name: dict(binding)
            for name, binding in self.hlsl_global_groupshared_pointer_aliases.items()
        }
        self.current_hlsl_bounded_loop_indices = {}
        self.current_hlsl_parameter_resource_types = {}
        # Name of the SV_GroupIndex parameter an explicit @gl_SubgroupID compute
        # parameter needs injected, populated lazily while emitting parameters and
        # consumed by the post-loop compute-builtin parameter pass.
        self.current_hlsl_compute_group_index_param = None
        self.current_hlsl_compute_subgroup_id_local = None
        self.current_hlsl_compute_subgroup_id_group_index = None
        self.current_hlsl_stage_output_lowering = stage_output_lowering
        self.current_identifier_reserved_names = (
            self.collect_hlsl_function_identifier_names(func)
        )
        if stage_output_lowering is not None:
            self.current_identifier_aliases.update(stage_output_lowering["aliases"])
            for local_name_key in ("local_name", "input_local_name"):
                local_name = stage_output_lowering.get(local_name_key)
                if local_name is not None:
                    self.current_identifier_reserved_names.add(local_name)

    def hlsl_entry_parameter_facts(
        self,
        func,
        effective_shader_type,
        param_list,
        parameter_prologue_statements,
    ):
        """Return the ids of the entry parameters promoted to resource globals and the default semantics of vertex-stage parameters."""
        promoted_entry_resource_parameter_ids = set()
        if normalize_stage_name(effective_shader_type) in self.stage_entry_types():
            promoted_entry_resource_parameter_ids = {
                id(parameter)
                for parameter in param_list
                if self.hlsl_entry_resource_parameter_global_type(parameter, func)
                is not None
            }
            promoted_entry_resource_parameter_ids.update(
                id(parameter)
                for parameter in param_list
                if id(parameter) in self.hlsl_promoted_entry_constant_members
            )
            mutable_resource_offsets = (
                self.hlsl_mutable_resource_pointer_argument_names(func)
                | self.hlsl_nested_entry_resource_rebase_names(func)
            )
            self.apply_hlsl_promoted_entry_resource_parameter_aliases(
                param_list,
                promoted_entry_resource_parameter_ids,
                func,
                mutable_resource_offsets=mutable_resource_offsets,
                parameter_prologue_statements=parameter_prologue_statements,
            )
        default_vertex_parameter_semantics = {}
        if effective_shader_type == "vertex":
            default_vertex_parameter_semantics = (
                self.hlsl_default_vertex_parameter_semantics(param_list, func)
            )
        return promoted_entry_resource_parameter_ids, default_vertex_parameter_semantics

    def append_hlsl_required_builtin_parameters(
        self,
        func,
        execution_config,
        effective_shader_type,
        param_list,
        params_str,
        signature,
    ):
        """Append the stage, fragment and compute built-in parameters a function needs to its parameter list, and return the list with the software-subgroup entry index."""
        software_subgroup_entry_index = None
        if effective_shader_type is None:
            params_str = self.append_required_hlsl_stage_parameter_parameters(
                params_str, getattr(func, "name", None), signature.param_names
            )
            params_str = self.append_required_hlsl_fragment_builtin_parameters(
                params_str, getattr(func, "name", None), signature.param_names
            )
            for parameter in self.required_hlsl_stage_parameters(
                getattr(func, "name", None)
            ):
                name = getattr(parameter, "name", None)
                if name and name not in self.local_variable_types:
                    self.local_variable_types[name] = self.type_name_string(
                        self.hlsl_parameter_raw_type(parameter)
                    )
            for builtin_name in self.required_hlsl_fragment_builtin_names(
                getattr(func, "name", None)
            ):
                name = self.hlsl_fragment_builtin_parameter_name(
                    getattr(func, "name", None),
                    builtin_name,
                )
                param_type = self.hlsl_fragment_builtin_parameter_type(builtin_name)
                if name and param_type:
                    self.current_identifier_aliases[builtin_name] = name
                    self.local_variable_types[name] = param_type
        elif effective_shader_type == "compute":
            for (
                builtin_name,
                name,
                param_type,
                semantic,
                alias,
            ) in self.required_hlsl_compute_builtin_parameters(func, execution_config):
                if alias is not None:
                    self.current_identifier_aliases[builtin_name] = alias
                    continue
                if not name:
                    continue
                self.current_identifier_aliases[builtin_name] = name
                if name in signature.param_names:
                    continue
                declaration = f"{param_type} {name}"
                if semantic:
                    declaration += f" : {semantic}"
                params_str = self.append_hlsl_parameter_declaration(
                    params_str, declaration
                )
                signature.param_names.add(name)
                self.local_variable_types[name] = param_type
            if self.hlsl_software_subgroup_invocation_variable is not None:
                software_subgroup_entry_index = (
                    self.hlsl_compute_group_index_dependency_name(param_list)
                )
            # Logical subgroup derivations require SV_GroupIndex even when the
            # source exposes only a partial local ID or no local ID at all.
            group_index_param = self.current_hlsl_compute_group_index_param
            if group_index_param and group_index_param not in signature.param_names:
                params_str = self.append_hlsl_parameter_declaration(
                    params_str, f"uint {group_index_param} : SV_GroupIndex"
                )
                signature.param_names.add(group_index_param)
                self.local_variable_types[group_index_param] = "uint"
            subgroup_id_local = self.current_hlsl_compute_subgroup_id_local
            if subgroup_id_local is not None:
                helper_name = self.hlsl_physical_subgroup_id_helper_name()
                group_index_name = self.current_hlsl_compute_subgroup_id_group_index
                signature.parameter_prologue_statements.insert(
                    0,
                    f"uint {subgroup_id_local} = "
                    f"{helper_name}({group_index_name});",
                )
        elif effective_shader_type == "vertex":
            explicit_stage_builtins = self.explicit_hlsl_vertex_builtin_parameters(
                param_list
            )
            for (
                name,
                param_type,
                semantic,
            ) in self.required_hlsl_vertex_builtin_parameters(func):
                if not name or name in signature.param_names:
                    continue
                # An input that already carries the system value delivers it; a second one would overlap it.
                existing = explicit_stage_builtins.get(semantic.lower())
                if existing:
                    self.current_identifier_aliases[name] = existing
                    continue
                declaration = f"{param_type} {name}"
                if semantic:
                    declaration += f" : {semantic}"
                params_str = self.append_hlsl_parameter_declaration(
                    params_str, declaration
                )
                signature.param_names.add(name)
                self.local_variable_types[name] = param_type
        elif effective_shader_type == "fragment":
            explicit_stage_builtins = self.explicit_hlsl_fragment_builtin_parameters(
                param_list
            )
            for builtin_name, parameter_name in explicit_stage_builtins.items():
                if parameter_name:
                    self.current_identifier_aliases[builtin_name] = parameter_name
            reserved_builtin_names = set(signature.param_names)
            reserved_builtin_names.update(self.function_scope_variable_types(func))
            for (
                builtin_name,
                name,
                param_type,
                semantic,
                prologue,
            ) in self.required_hlsl_fragment_builtin_parameters(
                func, reserved_builtin_names, explicit_stage_builtins
            ):
                declaration = f"{param_type} {name}"
                if semantic:
                    declaration += f" : {semantic}"
                params_str = self.append_hlsl_parameter_declaration(
                    params_str, declaration
                )
                signature.param_names.add(name)
                self.local_variable_types[name] = param_type
                if prologue is not None:
                    alias_name, alias_type, statement = prologue
                    signature.parameter_prologue_statements.append(statement)
                    self.local_variable_types[alias_name] = alias_type
                    self.current_identifier_aliases[builtin_name] = alias_name
                else:
                    self.current_identifier_aliases[builtin_name] = name
        return software_subgroup_entry_index, params_str

    def hlsl_function_return_types(
        self,
        func,
        stage_output_lowering,
    ):
        """Resolve and record the source, HLSL and declared return types of a function, refusing pointer returns and array returns that cannot be wrapped."""
        if isinstance(getattr(func, "return_type", None), PointerType):
            # HLSL has no pointer types; DXC rejects "T *f()" ("pointers are unsupported in HLSL").
            self.refuse_lowering(
                f"pointer return: '{func.name}' returns a pointer, which HLSL has no type for"
            )
        if hasattr(func, "return_type"):
            raw_return_type = self.type_name_string(func.return_type)
            if "[" in raw_return_type:
                fixed_array_return_type = self.hlsl_fixed_array_return_type_name(
                    raw_return_type
                )
                if fixed_array_return_type is None:
                    raise DirectXRefusal(
                        "DirectX output does not support array return types; "
                        "wrap the array in a struct or use an output parameter"
                    )
                raw_return_type = fixed_array_return_type
                return_type = fixed_array_return_type
            else:
                return_type = self.map_type(raw_return_type)
        else:
            raw_return_type = "void"
            return_type = "void"
        if (
            stage_output_lowering is not None
            and stage_output_lowering.get("struct_name") is not None
            and self.map_type(return_type) == "void"
        ):
            raw_return_type = stage_output_lowering["struct_name"]
            return_type = stage_output_lowering["struct_name"]
        self.current_function_return_type = raw_return_type
        self.current_hlsl_precise_return_type = (
            return_type
            if return_type != "void"
            and (
                self.hlsl_has_precise_modifier(func)
                or self.hlsl_is_metal_free_arithmetic_operator(func, return_type)
            )
            else None
        )
        # ``precise`` marks a value; a function that returns none keeps it on its locals and stores instead.
        declaration_return_type = (
            f"{self.hlsl_precise_modifier_prefix(func) if return_type != 'void' else ''}"
            f"{return_type}"
        )
        return declaration_return_type, raw_return_type, return_type

    def generate_hlsl_function_header(
        self,
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
    ):
        """Render the attributes and signature line of a stage entry point or an ordinary function."""
        if effective_shader_type is not None:
            self.validate_hlsl_function_return_semantic(
                func, effective_shader_type, raw_return_type, return_semantic
            )
            self.validate_hlsl_stage_return_semantics(
                func, effective_shader_type, return_semantic
            )
            self.validate_hlsl_feedback_texture_stage_calls(func, effective_shader_type)
            self.validate_hlsl_interpolation_stage_calls(func, effective_shader_type)
            self.validate_hlsl_synchronization_stage_calls(func, effective_shader_type)
            self.validate_hlsl_derivative_stage_calls(func, effective_shader_type)
        waveops_helper_lanes_attribute = (
            self.generate_hlsl_waveops_include_helper_lanes_attribute(
                func, effective_shader_type
            )
        )
        wave_size_attribute = self.generate_hlsl_wave_size_attribute(
            func, effective_shader_type
        )
        root_signature_attribute = self.generate_hlsl_root_signature_attribute(func)

        if effective_shader_type in shader_map:
            return_semantic_attr = self.map_semantic(return_semantic)
            code += f"// {effective_shader_type.capitalize()} Shader\n"
            code += root_signature_attribute
            self.validate_hlsl_stage_parameter_requirements(func, effective_shader_type)
            if effective_shader_type == "compute":
                code += self.generate_compute_numthreads(execution_config)
            if (
                stage_output_lowering is not None
                and stage_output_lowering.get("input_struct_name") is not None
            ):
                input_name = stage_output_lowering["input_local_name"]
                input_declaration = (
                    f"{stage_output_lowering['input_struct_name']} {input_name}"
                )
                params_str = self.append_hlsl_parameter_declaration(
                    params_str, input_declaration
                )
                self.local_variable_types[input_name] = stage_output_lowering[
                    "input_struct_name"
                ]
            code += wave_size_attribute
            code += waveops_helper_lanes_attribute
            if effective_shader_type == "fragment" and requests_early_fragment_tests(
                func, stage_node
            ):
                code += "[earlydepthstencil]\n"
            function_name = entry_name or shader_map[effective_shader_type]
            code += f"{declaration_return_type} {function_name}({params_str}){return_semantic_attr} {{\n"
        else:
            shader_attr = shader_attr_map.get(effective_shader_type)
            self.validate_hlsl_stage_requirements(
                func,
                effective_shader_type,
                stage_node=stage_node,
            )
            code += root_signature_attribute
            code += self.generate_stage_numthreads(
                func, effective_shader_type, execution_config
            )
            code += self.generate_hlsl_stage_attributes(
                func,
                effective_shader_type,
                stage_node=stage_node,
            )
            code += wave_size_attribute
            code += waveops_helper_lanes_attribute
            if shader_attr:
                code += f'[shader("{shader_attr}")]\n'
            function_name = self.hlsl_function_declaration_name(func, entry_name)
            if _workgroup_pointer_variant is not None:
                function_name = self.hlsl_workgroup_pointer_specialization_name(
                    getattr(func, "name", None),
                    _workgroup_pointer_variant,
                    function_name,
                )
            return_semantic_attr = self.map_semantic(return_semantic)
            code += f"{declaration_return_type} {function_name}({params_str}){return_semantic_attr} {{\n"
        return code, function_name

    def generate_hlsl_function_body(
        self,
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
    ):
        """Render the body of a function with its parameter prologue and epilogue and fall-through return, then restore the per-function state."""
        saved_resource_parameter_state = save_hlsl_function_state(
            self, HLSL_FUNCTION_RESOURCE_PARAMETER_ATTRIBUTES
        )
        self.current_sampler_parameters = signature.sampler_parameters
        self.current_texture_parameters = signature.texture_parameters
        self.current_image_access_parameters = signature.image_access_parameters
        self.current_image_format_parameters = signature.image_format_parameters
        self.current_implicit_texture_samplers = {
            texture_name: sampler_info["sampler_name"]
            for texture_name, sampler_info in (
                signature.implicit_texture_samplers.items()
            )
        }
        self.current_implicit_texture_regular_samplers = {
            texture_name: self.implicit_regular_sampler_name(texture_name, sampler_info)
            for texture_name, sampler_info in (
                signature.implicit_texture_samplers.items()
            )
            if sampler_info.get("regular") and sampler_info.get("comparison")
        }
        self.current_implicit_texture_query_lod_samplers = {
            texture_name: self.implicit_query_lod_sampler_name(
                texture_name, sampler_info
            )
            for texture_name, sampler_info in (
                signature.implicit_texture_samplers.items()
            )
            if sampler_info.get("query_lod")
        }
        previous_hlsl_visible_int_constants = self.current_hlsl_visible_int_constants
        self.current_hlsl_visible_int_constants = self.hlsl_initial_int_constants(func)
        previous_texture_offset_constants = self.current_hlsl_texture_offset_constants
        self.current_hlsl_texture_offset_constants = (
            self.hlsl_initial_texture_offset_constants(func)
        )
        for parameter_name, value in self.hlsl_texture_offset_parameter_constants.get(
            id(func), {}
        ).items():
            self.current_hlsl_texture_offset_constants[parameter_name] = value
        body_terminates = self.statement_body_terminates(body)
        needs_fallthrough_return = return_type != "void" and not body_terminates
        compound_declaration_offset = len(code)
        previous_compound_indices = self.hlsl_wide_integer_compound_indices
        self.hlsl_wide_integer_compound_indices = []
        previous_parameter_epilogue = self.current_hlsl_parameter_epilogue
        self.current_hlsl_parameter_epilogue = tuple(
            signature.parameter_epilogue_statements
        )
        previous_constant_guards = getattr(
            self, "current_hlsl_constant_branch_guards", ({}, set())
        )
        # Branches that compile-time constants rule out are not emitted: HLSL
        # type-checks them where the source's implicit conversions do not apply.
        self.current_hlsl_constant_branch_guards = constant_branch_guards(
            self, func, self.hlsl_initial_bool_constants(func)
        )
        previous_precise_function_state = self.enter_hlsl_precise_function(func, body)
        try:
            if stage_output_lowering is not None:
                output_name = stage_output_lowering.get("local_name")
                output_type = stage_output_lowering.get("struct_name")
                if output_name is not None and output_type is not None:
                    code += f"{'    ' * (indent + 1)}{output_type} {output_name};\n"
                    self.local_variable_types[output_name] = output_type
            for statement in signature.parameter_prologue_statements:
                code += f"{'    ' * (indent + 1)}{statement}\n"
            code += self.generate_hlsl_glsl_block_loads(
                function_name,
                shader_type is not None or entry_name is not None,
                indent + 1,
            )
            if (
                effective_shader_type == "compute"
                and self.hlsl_software_subgroup_invocation_variable is not None
            ):
                # HLSL static globals are private to each shader invocation.
                variable = self.hlsl_software_subgroup_invocation_variable
                code += (
                    f"{'    ' * (indent + 1)}{variable} = "
                    f"uint({software_subgroup_entry_index});\n"
                )
            self.hlsl_tall_matrix_rows.mark_writes(body)
            if self.current_hlsl_tail_recursive_call_ids:
                code += f"{'    ' * (indent + 1)}while (true) {{\n"
                code += self.generate_statement_body(body, indent + 2)
                code += f"{'    ' * (indent + 2)}break;\n"
                code += f"{'    ' * (indent + 1)}}}\n"
            else:
                code += self.generate_statement_body(body, indent + 1)
            declarations = "".join(
                f"{'    ' * (indent + 1)}{vtype} {name};\n"
                for vtype, name in self.hlsl_wide_integer_compound_indices + list(
                    self.hlsl_precise_captures or ()
                )
            )
            code = (
                code[:compound_declaration_offset]
                + declarations
                + code[compound_declaration_offset:]
            )
        finally:
            self.leave_hlsl_precise_function(previous_precise_function_state)
            self.current_hlsl_constant_branch_guards = previous_constant_guards
            self.current_hlsl_parameter_epilogue = previous_parameter_epilogue
            self.hlsl_wide_integer_compound_indices = previous_compound_indices
            self.current_hlsl_visible_int_constants = (
                previous_hlsl_visible_int_constants
            )
            self.current_hlsl_texture_offset_constants = (
                previous_texture_offset_constants
            )
        code = self.append_hlsl_function_epilogue(
            indent,
            stage_output_lowering,
            code,
            raw_return_type,
            signature,
            body_terminates,
            needs_fallthrough_return,
        )
        restore_hlsl_function_state(self, saved_resource_parameter_state)
        restore_hlsl_function_state(self, saved_function_state)
        restore_hlsl_function_state(self, saved_function_body_state)

        code += "  " * indent + "}\n\n"
        return code

    def append_hlsl_function_epilogue(
        self,
        indent,
        stage_output_lowering,
        code,
        raw_return_type,
        signature,
        body_terminates,
        needs_fallthrough_return,
    ):
        """Append the parameter writeback statements of a body that can fall through, and the default return of a non-void function whose body does not end in a return."""
        if not body_terminates:
            for statement in signature.parameter_epilogue_statements:
                code += f"{'    ' * (indent + 1)}{statement}\n"
        if needs_fallthrough_return:
            if (
                stage_output_lowering is not None
                and stage_output_lowering.get("local_name") is not None
            ):
                fallback = stage_output_lowering["local_name"]
                code += self.generate_hlsl_return_value_statement(fallback, indent + 1)
            else:
                struct_fallback = self.render_hlsl_default_struct_return(
                    raw_return_type, indent + 1
                )
                if struct_fallback is not None:
                    code += struct_fallback
                else:
                    fallback = self.hlsl_default_value_expression(raw_return_type)
                    code += self.generate_hlsl_return_value_statement(
                        fallback, indent + 1
                    )
        return code

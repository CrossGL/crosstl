"""Parts of a Slang function that ``SlangCodeGen.generate_function`` renders in turn: the return type with its semantic and stage rewrites, the parameter declarations with their local aliases, and the signature with the body."""

from .array_utils import (
    format_c_style_array_declaration,
)
from .slang_constant_references import (
    slang_constant_reference_parameter,
)
from .slang_resource_memory import (
    slang_resource_parameter_memory_prefix,
)
from .slang_varying_locations import (
    entry_parameter_location_prefix,
)
from .slang_wave_shuffles import (
    slang_subgroup_parameter_alias,
)
from .stage_utils import (
    is_fragment_output_parameter,
)


class SlangFunctionPartsMixin:
    """Function-rendering members of ``SlangCodeGen.generate_function``."""

    def slang_function_return_plan(
        self,
        node,
        shader_type,
        stage_role,
        extra_parameters,
    ):
        """Choose the return type and semantic, collect the body statements and parameters, and pick the fragment, hull and built-in return rewrites the stage needs."""
        if hasattr(node, "return_type"):
            ret_type_name = self.convert_type_node_to_string(node.return_type)
            ret_type = self.convert_type(ret_type_name)
        else:
            ret_type_name = "void"
            ret_type = "void"

        semantic = self.function_return_semantic(node)
        if shader_type is not None and semantic is None:
            semantic = self.default_slang_stage_return_semantic(shader_type, ret_type)
        body = getattr(node, "body", [])
        body_statements = self.get_statements(body)
        param_list = getattr(node, "parameters", getattr(node, "params", []))
        param_list = self.slang_merge_function_parameters(
            param_list, extra_parameters, shader_type
        )
        fragment_output_rewrite = None
        if self.convert_type(ret_type_name) == "void" and semantic is None:
            for parameter in param_list:
                parameter_semantic = self.semantic_from_node(parameter)
                if is_fragment_output_parameter(
                    shader_type,
                    parameter,
                    parameter_semantic,
                ):
                    ret_type_name = self.slang_parameter_type_name(parameter)
                    ret_type = self.convert_type(ret_type_name)
                    semantic = parameter_semantic
                    fragment_output_rewrite = {
                        "parameter": parameter,
                        "local_name": getattr(parameter, "name", None),
                        "return_type_name": ret_type_name,
                    }
                    break
        if fragment_output_rewrite is not None:
            output_parameter = fragment_output_rewrite["parameter"]
            param_list = [
                parameter
                for parameter in param_list
                if parameter is not output_parameter
            ]
        self.validate_slang_return_semantic(
            shader_type, semantic, stage_role, ret_type_name
        )
        self.validate_slang_struct_return_semantics(
            shader_type, ret_type_name, stage_role=stage_role
        )
        hull_output_rewrite = None
        if stage_role != "patch_constant":
            hull_output_rewrite = self.slang_stage_hull_output_rewrite(
                body_statements, shader_type, ret_type_name, semantic, param_list
            )
        if hull_output_rewrite is not None:
            ret_type_name = hull_output_rewrite["return_type_name"]
            ret_type = self.convert_type(ret_type_name)
            semantic = None

        builtin_return_rewrite = self.slang_stage_builtin_return_rewrite(
            body_statements, shader_type, ret_type_name, semantic
        )
        if builtin_return_rewrite is not None:
            ret_type_name = builtin_return_rewrite["return_type_name"]
            ret_type = self.convert_type(ret_type_name)
            semantic = builtin_return_rewrite["semantic"]
        return (
            semantic,
            body_statements,
            param_list,
            fragment_output_rewrite,
            hull_output_rewrite,
            builtin_return_rewrite,
            ret_type_name,
            ret_type,
        )

    def slang_function_parameter_declarations(
        self,
        node,
        shader_type,
        stage_role,
        body_statements,
        effective_param_list,
    ):
        """Declare the parameters, with the local aliases some stage parameters need."""
        params = []
        local_alias_declarations = []
        emitted_param_names = set()
        if effective_param_list:
            if effective_param_list and hasattr(effective_param_list[0], "name"):
                for param in effective_param_list:
                    if hasattr(param, "param_type"):
                        param_type_name = self.convert_type_node_to_string(
                            param.param_type
                        )
                        self.register_variable_type(param.name, param_type_name, param)
                        param_type = self.map_resource_type_with_format(
                            param_type_name, param
                        )
                    elif hasattr(param, "vtype"):
                        param_type_name = param.vtype
                        self.register_variable_type(param.name, param_type_name, param)
                        param_type = self.map_resource_type_with_format(
                            param_type_name, param
                        )
                    else:
                        param_type_name = "float"
                        param_type = "float"
                    semantic = self.semantic_from_node(param)
                    dispatch_thread_id_bridge = (
                        self.slang_compute_global_invocation_id_parameter_bridge(
                            param,
                            param_type,
                            semantic,
                            emitted_param_names,
                            effective_param_list,
                        )
                        if shader_type == "compute"
                        else None
                    )
                    intrinsic_alias = self.slang_num_workgroups_parameter_alias(
                        param, param_type, semantic, shader_type
                    )
                    if intrinsic_alias is None:
                        intrinsic_alias = self.slang_helper_invocation_parameter_alias(
                            param, param_type, semantic, shader_type
                        )
                    if intrinsic_alias is None:
                        intrinsic_alias = slang_subgroup_parameter_alias(
                            self, param, param_type, semantic
                        )
                    if intrinsic_alias is not None:
                        local_alias_declarations.append(intrinsic_alias)
                        emitted_param_names.add(param.name)
                        continue
                    if dispatch_thread_id_bridge is not None:
                        declaration, local_alias, native_name = (
                            dispatch_thread_id_bridge
                        )
                        self.register_variable_type(native_name, "uint3", param)
                        self.register_variable_type(param.name, param_type_name, param)
                        params.append(declaration)
                        local_alias_declarations.append(local_alias)
                        emitted_param_names.add(native_name)
                        continue

                    declaration = format_c_style_array_declaration(
                        param_type, param.name
                    )
                    declaration = (
                        slang_resource_parameter_memory_prefix(self, param_type, param)
                        + declaration
                    )
                    ray_declaration = self.slang_ray_stage_parameter_declaration(
                        declaration, param, shader_type
                    )
                    if ray_declaration is not None:
                        params.append(ray_declaration)
                        continue
                    mesh_payload_declaration = (
                        self.slang_mesh_payload_parameter_declaration(
                            declaration, param, shader_type
                        )
                    )
                    if mesh_payload_declaration is not None:
                        params.append(mesh_payload_declaration)
                        continue
                    mesh_output_declaration = (
                        self.slang_mesh_output_parameter_declaration(
                            declaration, param, shader_type
                        )
                    )
                    if mesh_output_declaration is not None:
                        params.append(mesh_output_declaration)
                        continue
                    constant_reference = (
                        slang_constant_reference_parameter(self, param)
                        if shader_type is not None
                        else None
                    )
                    if constant_reference is not None:
                        declaration, value_type = constant_reference
                        self.register_variable_type(param.name, value_type, param)
                        params.append(declaration)
                        emitted_param_names.add(param.name)
                        continue
                    if shader_type is not None and str(param_type).rstrip().endswith(
                        "&"
                    ):
                        # A Metal ``constant T& x [[buffer(n)]]`` argument; ``T &x`` is no Slang entry parameter.
                        self.refuse_lowering(
                            f"Slang entry parameter '{param.name}' has the reference type "
                            f"{param_type}, which a Slang entry point cannot take"
                        )
                    declaration = (
                        entry_parameter_location_prefix(
                            self, param, semantic, shader_type
                        )
                        + self.slang_stage_parameter_qualifier_prefix(
                            param, shader_type
                        )
                        + declaration
                    )
                    declaration = self.apply_slang_resource_binding_decorations(
                        declaration, param, param_type_name
                    )
                    params.append(
                        declaration + self.semantic_suffix(semantic, shader_type)
                    )
                    emitted_param_names.add(param.name)
            else:
                for param_type, param_name in effective_param_list:
                    self.register_variable_type(param_name, param_type)
                    params.append(
                        f"{self.map_resource_type_with_format(param_type)} {param_name}"
                    )

        for (
            param_type,
            param_name,
            semantic,
        ) in self.slang_implicit_stage_parameters(
            body_statements, shader_type, effective_param_list, stage_role=stage_role
        ):
            self.register_variable_type(param_name, param_type)
            params.append(f"{self.convert_type(param_type)} {param_name} : {semantic}")

        if shader_type is None:
            existing_param_names = {
                getattr(param, "name", None) for param in effective_param_list or []
            }
            for parameter in self.required_slang_stage_parameters(
                getattr(node, "name", None)
            ):
                name = getattr(parameter, "name", None)
                if not name or name in existing_param_names:
                    continue
                existing_param_names.add(name)
                self.register_variable_type(
                    name, self.slang_parameter_type_name(parameter), parameter
                )
                params.append(
                    self.slang_stage_parameter_dependency_declaration(parameter)
                )
        return params, local_alias_declarations

    def slang_function_definition_text(
        self,
        node,
        shader_type,
        execution_config,
        entry_name,
        emit_stage_decorations,
        body_statements,
        builtin_return_rewrite,
        fragment_output_rewrite,
        hull_output_rewrite,
        local_alias_declarations,
        params_str,
        ret_type,
        semantic_str,
    ):
        """Write the stage decorations, the signature and the body or the declaration."""
        result = ""
        if (
            builtin_return_rewrite is not None
            and builtin_return_rewrite["mode"] == "struct"
        ):
            result += self.generate_slang_builtin_output_struct(builtin_return_rewrite)
            result += "\n\n"
        if shader_type and emit_stage_decorations:
            self.validate_slang_stage_attributes(node, shader_type)
            result += self.generate_slang_stage_numthreads(
                node, shader_type, execution_config
            )
            result += self.generate_slang_stage_attributes(node, shader_type)
            shader_stage = self.slang_shader_stage_name(shader_type)
            result += f'[shader("{shader_stage}")]\n'
        function_name = entry_name or node.name
        precise_prefix = "precise " if self.slang_function_is_precise(node) else ""
        signature = (
            f"{precise_prefix}{ret_type} {function_name}({params_str}){semantic_str}"
        )
        if getattr(node, "body", None) is None and not shader_type:
            # A prototype stays a forward declaration: an empty body would define the function a second time,
            # which slangc rejects once the definition follows.
            result += f"{signature};"
        else:
            previous_precise_state = self.enter_slang_precise_function(
                node, body_statements
            )
            try:
                result += self.emit_slang_function_body(
                    signature,
                    body_statements,
                    builtin_return_rewrite,
                    hull_output_rewrite,
                    fragment_output_rewrite,
                    local_alias_declarations,
                )
            finally:
                self.leave_slang_precise_function(previous_precise_state)
        return result

"""Parts of CUDA function rendering: the function qualifiers, the parameter declarations with their value-array copies and resource metadata parameters, the stage comments, and the function body."""

from .c_reference_parameters import (
    is_output_parameter,
    reference_declarator,
)
from .value_array_parameters import (
    fixed_array_dimensions,
    plan_value_array_copies,
    value_array_copy_lines,
)


class CudaFunctionPartsMixin:
    """Function-rendering members of ``CudaCodeGen.visit_FunctionNode``."""

    def cuda_function_qualifiers(
        self,
        node,
    ):
        """Choose ``__global__`` or ``__device__`` from the function's stage qualifiers."""
        qualifiers = []
        stage_qualifiers = self.function_stage_qualifier_names(node)

        if stage_qualifiers:
            for qualifier_name in stage_qualifiers:
                if qualifier_name == "compute":
                    qualifiers.append("__global__")
                elif qualifier_name in ["vertex", "fragment"]:
                    qualifiers.append("__device__")
                else:
                    qualifiers.append("__device__")
        else:
            qualifiers.append("__device__")
        return qualifiers

    def cuda_function_parameter_declarations(
        self,
        node,
        param_list,
        stage_name,
    ):
        """Check the stage parameters, then declare the function's parameters, with value-array copies and the metadata, length and counter parameters of resources."""
        params = []
        self.validate_cuda_stage_parameter_semantics(stage_name, param_list)
        if stage_name == "geometry":
            self.validate_cuda_geometry_stage(node, param_list)
        if stage_name in {"tessellation_control", "tessellation_evaluation"}:
            self.validate_cuda_tessellation_stage(node, stage_name)
        if stage_name in self.cuda_mesh_task_stage_names():
            self.validate_cuda_mesh_task_stage(node, stage_name)

        value_array_copies = (
            {}
            if stage_name is not None
            else plan_value_array_copies(
                node,
                param_list,
                self.type_name_string,
                self.function_parameters_by_name,
                self.value_array_struct_names,
            )
        )
        value_array_copy_statements = []
        for param in param_list:
            if hasattr(param, "param_type"):
                param_type = param.param_type
            elif hasattr(param, "vtype"):
                param_type = param.vtype
            else:
                param_type = "void"

            declaration_type = self.resource_type_with_access(param_type, param)
            param_name = getattr(param, "name", getattr(param, "param_name", None))
            builtin_role = self.cuda_compute_builtin_parameter_role(stage_name, param)
            if param_name and builtin_role is not None:
                self.register_variable_type(param_name, declaration_type, param)
                self.stage_builtin_aliases[param_name] = builtin_role
                self.stage_builtin_alias_types[param_name] = declaration_type
                continue

            geometry_stream_declaration = self.format_cuda_geometry_stream_parameter(
                declaration_type,
                param_name,
            )
            if geometry_stream_declaration is not None:
                self.register_variable_type(
                    param_name,
                    self.cuda_geometry_stream_mapped_type(declaration_type),
                    param,
                )
                params.append(geometry_stream_declaration)
            elif param_name in value_array_copies:
                incoming = value_array_copies[param_name]
                self.register_variable_type(param_name, declaration_type, param)
                params.append(self.format_typed_declarator(declaration_type, incoming))
                value_array_copy_statements.extend(
                    value_array_copy_lines(
                        self.format_typed_declarator(declaration_type, param_name),
                        param_name,
                        incoming,
                        fixed_array_dimensions(self.type_name_string(declaration_type)),
                    )
                )
            else:
                self.register_variable_type(param_name, declaration_type, param)
                declarator = self.format_typed_declarator(declaration_type, param_name)
                if stage_name is None and is_output_parameter(param):
                    # out and inout copy back to the caller; a C++ reference binds it.
                    declarator = reference_declarator(declarator, param_name)
                declarator = self.apply_restrict_qualifier_to_declaration(
                    declarator, param_name, param
                )
                declarator = self.apply_volatile_qualifier_to_parameter(
                    declarator, param
                )
                params.append(declarator)
            metadata_param = self.query_metadata_parameter(param_name, declaration_type)
            if metadata_param:
                params.append(metadata_param)
            length_param = self.structured_buffer_length_parameter(
                node.name, param_name, declaration_type
            )
            if length_param:
                self.current_structured_buffer_length_parameters[param_name] = (
                    self.structured_buffer_length_name(param_name)
                )
                params.append(length_param)
            counter_param = self.structured_buffer_counter_parameter(
                param_name, declaration_type
            )
            if counter_param:
                params.append(counter_param)

        param_str = ", ".join(params)
        return value_array_copy_statements, param_str

    def cuda_function_stage_comments(
        self,
        node,
        param_list,
        return_semantic,
        stage_name,
    ):
        """Emit the comments that record the function's stage, return semantic and parameter semantics."""
        if stage_name in self.cuda_ray_stage_names():
            self.emit(
                "// CrossGL ray stage: "
                f"{self.cuda_ray_stage_metadata_name(stage_name)}"
            )
        if stage_name == "geometry":
            self.emit_generated_code(
                self.generate_cuda_geometry_stage_comments(node, param_list)
            )
        if stage_name in {"tessellation_control", "tessellation_evaluation"}:
            self.emit_generated_code(
                self.generate_cuda_tessellation_stage_comments(node, stage_name)
            )
        if stage_name in self.cuda_mesh_task_stage_names():
            self.emit_generated_code(
                self.generate_cuda_mesh_task_stage_comments(node, stage_name)
            )
        if return_semantic:
            self.emit(
                f"// CrossGL return semantic: {self.map_semantic(return_semantic)}"
            )
        if stage_name:
            for param in param_list:
                param_semantic = self.semantic_from_node(param)
                if param_semantic:
                    self.emit(
                        f"// CrossGL parameter semantic: {param.name}: "
                        f"{self.map_semantic(param_semantic)}"
                    )

    def cuda_function_body(
        self,
        node,
        body,
        fragment_output_param,
        function_name,
        param_str,
        return_type,
        signature_prefix,
        value_array_copy_statements,
    ):
        """Emit the function's signature and body."""
        self.emit(f"{signature_prefix} {return_type} {function_name}({param_str}) {{")

        self.indent_level += 1
        for statement in value_array_copy_statements:
            self.emit(statement)

        fragment_output_name = None
        if fragment_output_param is not None:
            fragment_output_name = getattr(
                fragment_output_param,
                "name",
                getattr(fragment_output_param, "param_name", None),
            )
            fragment_output_type = self.resource_type_with_access(
                self.get_parameter_type(fragment_output_param),
                fragment_output_param,
            )
            self.register_variable_type(
                fragment_output_name,
                fragment_output_type,
                fragment_output_param,
            )
            self.emit(
                f"{self.format_typed_declarator(fragment_output_type, fragment_output_name)};"
            )

        for local_var in getattr(self, "current_stage_local_variables", []) or []:
            if self.is_cuda_shared_variable(local_var) and id(local_var) not in getattr(
                self, "file_scope_shared_variable_ids", ()
            ):
                self.visit(local_var)

        precise_state = self.enter_precise_evaluation(node)
        try:
            self.emit_body(body)
        finally:
            self.leave_precise_evaluation(precise_state)
        if fragment_output_name is not None and not self.statement_body_terminates(
            body
        ):
            self.emit(f"return {fragment_output_name};")

        self.indent_level -= 1
        self.emit("}")

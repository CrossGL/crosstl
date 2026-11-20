"""The definition part of ``MetalToCrossGLConverter.generate_function``: parameters, attributes, signature and body."""

import re

from ....translator.resource_identity import source_resource_attribute


class MetalFunctionDefinitionMixin:
    """Members of ``MetalToCrossGLConverter``: the part of ``generate_function`` inside its scope guard."""

    def render_function_definition(
        self,
        func,
        indent,
        stage_entry,
        output_name,
        specialization_key,
        active_type_bindings,
        active_value_bindings,
        code,
        function_body,
        linkage_annotations,
        out_of_line_replacement,
    ):
        """Render the parameters, attributes, signature and body of one function, after ``generate_function`` has set up its scopes. Returns ``code`` with the function appended."""
        parameter_declarations, parameter_names = self.render_function_parameters(
            func, stage_entry, output_name
        )
        if out_of_line_replacement is not None:
            for definition_name, helper_name in out_of_line_replacement[
                "parameter_aliases"
            ]:
                self.current_variable_types[definition_name] = (
                    self.current_variable_types.get(helper_name)
                )
                self.current_variable_type_qualifiers[definition_name] = (
                    self.current_variable_type_qualifiers.get(helper_name, ())
                )
                self.identifier_maps[-1][definition_name] = self.render_identifier(
                    helper_name
                )
        fn_semantic = self.map_semantic(self.function_semantic_attributes(func))
        annotations = " ".join(
            item
            for item in [
                fn_semantic,
                *linkage_annotations,
                self.metal_mesh_function_annotation(func),
            ]
            if item
        )
        suffix = f" {annotations}" if annotations else ""
        function_name = self.sanitize_identifier(
            output_name or self.function_output_name(func)
        )
        self.current_function_specialization = function_name
        self.current_function_materialization_bindings = {
            **active_type_bindings,
            **active_value_bindings,
        }
        self.current_function_specialization_key = specialization_key
        return_type = self.map_function_return_type(
            func.return_type,
            qualifiers=(
                getattr(func, "return_qualifiers", None)
                or getattr(func, "declaration_qualifiers", None)
            ),
        )
        generic_prefix = (
            ""
            if getattr(func, "qualifier", None)
            else self.format_generic_prefix(
                func,
                bound_parameter_names={
                    *active_type_bindings,
                    *active_value_bindings,
                },
            )
        )
        value_param_decls = (
            ""
            if generic_prefix
            else self.format_value_template_parameter_declarations(
                func,
                indent + 1,
                body=function_body,
                bound_value_names=active_value_bindings,
            )
        )
        if function_body is None and getattr(func, "is_template_prototype", False):
            # A declaration-only primary that explicit specializations
            # complete; generators select them per deduced argument tuple.
            params = ", ".join(
                declaration for declaration in parameter_declarations if declaration
            )
            return (
                code + f"{generic_prefix}{return_type} {function_name}({params});\n\n"
            )
        body = (
            value_param_decls
            + self.metal_mesh_body_prologue(func, indent + 1)
            + self.metal_object_body_prologue(func, indent + 1)
            + self.generate_function_body(
                function_body,
                indent=indent + 1 + self.metal_object_body_indent(func),
            )
            + self.metal_object_body_epilogue(func, indent + 1)
            + self.metal_mesh_body_epilogue(func, indent + 1)
        )
        for index, parameter in enumerate(func.params):
            if (
                parameter.name in self.current_elided_parameter_uses
                and "@maybe_unused" not in parameter_declarations[index]
                and not re.search(rf"\b{re.escape(parameter_names[index])}\b", body)
            ):
                parameter_declarations[index] += " @maybe_unused"
        params = ", ".join(
            declaration for declaration in parameter_declarations if declaration
        )
        code += (
            f"{generic_prefix}{return_type} {function_name}({params})" f"{suffix} {{\n"
        )
        code += body
        code += "    }\n\n"
        return code

    def render_function_parameters(self, func, stage_entry, output_name):
        """Record the parameter types of ``func`` in the current scope and render its parameter declarations.

        Returns the declarations, with their buffer-layout and resource-origin attributes, and the rendered names.
        """
        for param in func.params:
            self.current_variable_types[param.name] = (
                self.metal_declaration_expression_type(param)
            )
            self.current_variable_type_qualifiers[param.name] = (
                self.metal_declaration_type_qualifiers(param)
            )
            if id(param) in self.storage_texture_declaration_ids:
                self.current_storage_texture_names.add(param.name)
            if self.structured_buffer_pointer_type(param):
                self.current_structured_buffer_names.add(param.name)
        semantic_context = {
            "kind": "parameter",
            "function_qualifier": getattr(func, "qualifier", None),
        }
        parameter_declarations = [
            self.format_parameter_decl(p, index, semantic_context=semantic_context)
            for index, p in enumerate(func.params)
        ]
        parameter_names = [self.render_identifier(p.name) for p in func.params]
        for index, parameter in enumerate(func.params):
            layout = self.structured_buffer_parameter_layout(func, parameter)
            if layout is None and stage_entry:
                layout = self.metal_constant_reference_layout(
                    parameter,
                    f"constant parameter {parameter.name} of "
                    f"{getattr(func, 'name', 'a function')}",
                )
            if layout is not None:
                parameter_declarations[index] = (
                    f"@buffer_layout({layout}) " + parameter_declarations[index]
                )
        if stage_entry and self.preserve_resource_origins:
            for index, parameter in enumerate(func.params):
                if self.is_stage_entry_buffer_resource_parameter(parameter):
                    parameter_declarations[index] = (
                        source_resource_attribute(
                            "metal",
                            output_name or self.function_output_name(func),
                            parameter.name,
                            index,
                        )
                        + parameter_declarations[index]
                    )
        return parameter_declarations, parameter_names

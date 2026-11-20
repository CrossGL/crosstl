"""Parts of HIP function rendering: the function qualifiers, the parameter declarations with their value-array copies and resource metadata parameters, the stage comments, and the function body."""

from ..ast import (
    FunctionNode,
)
from .c_reference_parameters import (
    is_output_parameter,
    reference_declarator,
)
from .stage_utils import (
    normalize_stage_name,
)
from .value_array_parameters import (
    fixed_array_dimensions,
    plan_value_array_copies,
    value_array_copy_lines,
)


class HipFunctionPartsMixin:
    """Function-rendering members of ``HipCodeGen.visit_FunctionNode``."""

    def hip_function_qualifiers(
        self,
        node: FunctionNode,
    ):
        """Choose ``__global__`` or ``__device__`` from the function's stage qualifiers and attributes."""
        qualifiers = []
        source_qualifiers = list(getattr(node, "qualifiers", []) or [])
        stage_attribute_names = {
            "compute",
            "fragment",
            "geometry",
            "kernel",
            "task",
            "mesh",
            "vertex",
        } | self.hip_ray_stage_names()
        source_qualifiers.extend(
            attribute_name
            for attribute_name in (
                getattr(attribute, "name", None)
                for attribute in getattr(node, "attributes", []) or []
            )
            if normalize_stage_name(attribute_name) in stage_attribute_names
        )
        source_qualifiers = [
            qualifier for qualifier in source_qualifiers if qualifier is not None
        ]
        if source_qualifiers:
            for qualifier in source_qualifiers:
                qualifier_name = normalize_stage_name(qualifier)
                if qualifier_name in {"kernel", "compute"}:
                    qualifiers.append("__global__")
                elif qualifier_name == "device":
                    qualifiers.append("__device__")
                else:
                    qualifiers.append("__device__")
        elif hasattr(node, "qualifier") and node.qualifier:
            qualifier_name = normalize_stage_name(node.qualifier)
            if qualifier_name in {"kernel", "compute"}:
                qualifiers.append("__global__")
            elif qualifier_name == "device":
                qualifiers.append("__device__")
            else:
                qualifiers.append("__device__")
        else:
            qualifiers.append("__device__")
        return qualifiers

    def hip_function_parameter_declarations(
        self,
        node: FunctionNode,
        param_list,
        stage_name,
    ):
        """Declare the function's parameters, with value-array copies and the metadata, length and counter parameters of resources."""
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
        param_declarations = []
        for param in param_list:
            param_type = self.get_parameter_type(param)
            param_name = getattr(param, "name", getattr(param, "param_name", None))
            builtin_role = self.hip_compute_builtin_parameter_role(stage_name, param)
            if param_name and builtin_role is not None:
                self.register_variable_type(param_name, param_type, param)
                self.stage_builtin_aliases[param_name] = builtin_role
                self.stage_builtin_alias_types[param_name] = param_type
                continue

            if param_name in value_array_copies:
                incoming = value_array_copies[param_name]
                declaration_type = self.resource_type_with_access(param_type, param)
                self.register_variable_type(param_name, declaration_type, param)
                param_declarations.append(
                    self.apply_resource_memory_qualifiers_to_declaration(
                        self.format_typed_declarator(declaration_type, incoming),
                        incoming,
                        param,
                    )
                )
                value_array_copy_statements.extend(
                    value_array_copy_lines(
                        self.apply_resource_memory_qualifiers_to_declaration(
                            self.format_typed_declarator(declaration_type, param_name),
                            param_name,
                            param,
                        ),
                        param_name,
                        incoming,
                        fixed_array_dimensions(self.type_name_string(declaration_type)),
                    )
                )
            else:
                declaration = self.visit_parameter(param)
                if stage_name is None and is_output_parameter(param):
                    # out and inout copy back to the caller; a C++ reference binds it.
                    declaration = reference_declarator(declaration, param_name)
                param_declarations.append(declaration)
            metadata_param = self.query_metadata_parameter(param_name, param_type)
            if metadata_param:
                param_declarations.append(metadata_param)
            length_param = self.structured_buffer_length_parameter(
                node.name, param_name, param_type
            )
            if length_param:
                self.current_structured_buffer_length_parameters[param_name] = (
                    self.structured_buffer_length_name(param_name)
                )
                param_declarations.append(length_param)
            counter_param = self.structured_buffer_counter_parameter(
                param_name, param_type
            )
            if counter_param:
                param_declarations.append(counter_param)
        params = ", ".join(param_declarations)
        return value_array_copy_statements, params

    def hip_function_stage_comments(
        self,
        node: FunctionNode,
        param_list,
        return_semantic,
        stage_name,
    ):
        """Emit the comments that record the function's stage, return semantic and parameter semantics."""
        if stage_name in self.hip_ray_stage_names():
            self.add_line(
                "// CrossGL ray stage: "
                f"{self.hip_ray_stage_metadata_name(stage_name)}"
            )
        if stage_name == "geometry":
            self.add_generated_code(
                self.generate_hip_geometry_stage_comments(node, param_list)
            )
        if stage_name in {"tessellation_control", "tessellation_evaluation"}:
            self.add_generated_code(
                self.generate_hip_tessellation_stage_comments(node, stage_name)
            )
        if stage_name in self.hip_mesh_task_stage_names():
            self.add_generated_code(
                self.generate_hip_mesh_task_stage_comments(node, stage_name)
            )
        if return_semantic:
            self.add_line(
                f"// CrossGL return semantic: {self.map_semantic(return_semantic)}"
            )
        if stage_name:
            for param in param_list:
                param_semantic = self.semantic_from_node(param)
                if param_semantic:
                    param_name = getattr(
                        param, "name", getattr(param, "param_name", "param")
                    )
                    self.add_line(
                        f"// CrossGL parameter semantic: {param_name}: "
                        f"{self.map_semantic(param_semantic)}"
                    )

    def hip_function_body(
        self,
        node: FunctionNode,
        body,
        fragment_output_param,
        signature,
        value_array_copy_statements,
    ):
        """Emit the function's declaration, or its signature and body."""
        if body is None:
            self.add_line(f"{signature};")
        else:
            self.add_line(signature)
            self.add_line("{")
            self.indent_level += 1
            precise_state = self.enter_precise_evaluation(node)
            for line in self.precise_body_prologue():
                self.add_line(line)
            for statement in value_array_copy_statements:
                self.add_line(statement)
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
                self.add_line(
                    f"{self.format_typed_declarator(fragment_output_type, fragment_output_name)};"
                )
            for local_var in getattr(self, "current_stage_local_variables", []) or []:
                if self.is_hip_shared_variable(local_var):
                    self.visit(local_var)
            try:
                self.emit_body(body)
            finally:
                self.leave_precise_evaluation(precise_state)
            if fragment_output_name is not None and not self.statement_body_terminates(
                body
            ):
                self.add_line(f"return {fragment_output_name};")

            self.indent_level -= 1
            self.add_line("}")

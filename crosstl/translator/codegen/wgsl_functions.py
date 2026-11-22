"""Function declarations of the WGSL generator: match fallback returns, signatures, parameters, implicit stage built-in parameters and helper-function reference checks."""

import re

from ..ast import (
    BlockNode,
    FunctionCallNode,
    IdentifierNode,
    IfNode,
    ReturnNode,
    VariableNode,
)
from ..refusals import WGSLRefusal
from .wgsl_function_behaviors import end_is_reachable
from .wgsl_stage_io import (
    interpolation_qualifiers,
    with_interpolation,
)
from .wgsl_subgroup_operations import wgsl_wave_builtin_inputs


class WGSLFunctionMixin:
    """Function members of ``WGSLCodeGen``."""

    def generate_function(self, func):
        # A helper only one stage's entry points reach is written in that stage's context (wgsl_helper_stages).
        previous_stage_name = self._current_stage_name
        if previous_stage_name is None:
            self._current_stage_name = self._helper_stage_names.get(id(func))
        try:
            return self.generate_function_in_stage(func)
        finally:
            self._current_stage_name = previous_stage_name

    def generate_function_in_stage(self, func):
        self.validate_helper_function_references(func)
        previous_function_return_type = self._current_function_return_type
        self._current_function_return_type = getattr(func, "return_type", None)
        self.push_identifier_scope(
            getattr(param, "name", "") for param in getattr(func, "parameters", [])
        )
        self.register_parameter_value_types(func)
        self.push_pointer_identifier_scope(self.buffer_pointer_parameter_names(func))
        try:
            self.push_parameter_plan(func)
            try:
                signature = self.generate_function_signature(func)
                body = self.generate_block(func.body, indent=0)
                body = self.add_generated_match_fallback_return(
                    body,
                    getattr(func, "body", None),
                    getattr(func, "return_type", None),
                )
                body = self.prepend_parameter_copies(body, func)
            finally:
                self.pop_parameter_plan()
        finally:
            self.pop_pointer_identifier_scope()
            self.pop_identifier_scope()
            self._current_function_return_type = previous_function_return_type
        return f"{signature} {body}"

    def add_generated_match_fallback_return(self, body_text, body_node, return_type):
        if not self.generated_match_fallback_return_required(body_node, return_type):
            return body_text
        lines = body_text.splitlines()
        if not lines or lines[-1].strip() != "}":
            return body_text
        fallback = self.default_value_expression_for_type(return_type)
        lines.insert(-1, f"    return {fallback};")
        return "\n".join(lines)

    def generated_match_fallback_return_required(self, body_node, return_type):
        if return_type is None or self.is_void_type(return_type):
            return False
        statements = self.block_statements(body_node)
        if not statements:
            return False
        if not self.contains_generated_rust_match_flag(statements):
            # WGSL rejects a value-returning body whose end it cannot rule out, as after a loop that returns.
            return end_is_reachable(statements)
        return not self.statement_definitely_returns(statements[-1])

    def block_statements(self, body_node):
        """The statements of a body; a braceless body is its one statement."""
        if body_node is None:
            return []
        if hasattr(body_node, "statements"):
            return list(getattr(body_node, "statements", []) or [])
        if isinstance(body_node, (list, tuple)):
            return list(body_node)
        return [body_node]

    def contains_generated_rust_match_flag(self, statements):
        return any(
            isinstance(statement, VariableNode)
            and re.fullmatch(r"_rust_match_matched_\d+", str(statement.name))
            for statement in statements
        )

    def statement_definitely_returns(self, statement):
        if isinstance(statement, ReturnNode):
            return True
        if isinstance(statement, BlockNode):
            statements = self.block_statements(statement)
            return bool(statements) and self.statement_definitely_returns(
                statements[-1]
            )
        if isinstance(statement, IfNode):
            return (
                statement.else_branch is not None
                and self.statement_definitely_returns(statement.then_branch)
                and self.statement_definitely_returns(statement.else_branch)
            )
        return False

    def generate_function_signature(
        self,
        func,
        name=None,
        return_attributes=(),
        leading_parameters=(),
        skip_parameter_ids=(),
        parameter_attributes=False,
    ):
        function_name = name or self.function_declaration_identifier_name(func)
        skip_parameter_ids = set(skip_parameter_ids or ())
        parameters = ", ".join(
            list(leading_parameters)
            + [
                self.generate_planned_parameter(param, parameter_attributes)
                for param in func.parameters
                if id(param) not in skip_parameter_ids
            ]
        )
        return_type = self.type_name_string(func.return_type)
        if return_type == "void":
            return f"fn {function_name}({parameters})"
        return_prefix = " ".join(return_attributes)
        if return_prefix:
            return f"fn {function_name}({parameters}) -> {return_prefix} {return_type}"
        return f"fn {function_name}({parameters}) -> {return_type}"

    def generate_parameter(self, node, include_attributes=True):
        attributes = (
            self.wgsl_attributes(node.attributes, direction="in")
            if include_attributes
            else ""
        )
        implicit_location = (
            self._entry_parameter_locations.get(id(node))
            if include_attributes
            else None
        )
        if implicit_location is not None:
            attributes = " ".join(
                part for part in (f"@location({implicit_location})", attributes) if part
            )
        if "@location(" in attributes:
            attributes = with_interpolation(
                attributes,
                self.type_name_string(node.param_type),
                interpolation_qualifiers(self, node.attributes),
                f"entry-point parameter '{node.name}'",
            )
        prefix = f"{attributes} " if attributes else ""
        parameter_name = self.identifier_name(node.name)
        if self.is_glsl_buffer_block_parameter(node):
            raise WGSLRefusal(
                "WGSL target does not support GLSL buffer block parameters yet; "
                "use module-scope storage bindings"
            )
        unsupported_storage_type = self.unsupported_storage_buffer_type_name(
            node.param_type
        )
        if unsupported_storage_type is not None:
            raise WGSLRefusal(
                "WGSL target does not support "
                f"{unsupported_storage_type} resources yet"
            )
        if self.structured_buffer_element_type(node.param_type) is not None:
            raise WGSLRefusal(
                "WGSL target does not support StructuredBuffer parameters yet; "
                "declare them as module-scope storage resources"
            )
        sampled_texture_array = self.sampled_texture_array_info(node.param_type)
        if sampled_texture_array is not None:
            resource_type = self.type_display_name(
                sampled_texture_array["element_type"]
            )
            raise WGSLRefusal(
                "WGSL target cannot lower sampled texture resource array "
                f"parameter {node.name} of type {resource_type}; WebGPU/WGSL "
                "requires project-level binding expansion to pass individual "
                "module-scope texture and sampler bindings"
            )
        sampled_texture_type = self.sampled_texture_type(node.param_type)
        if sampled_texture_type is not None:
            return (
                f"{prefix}{parameter_name}: {sampled_texture_type}, "
                f"{self.texture_sampler_name(parameter_name)}: "
                f"{self.companion_sampler_type(node.param_type)}"
            )
        multisampled_texture_type = self.multisampled_texture_type(node.param_type)
        if multisampled_texture_type is not None:
            return f"{prefix}{parameter_name}: {multisampled_texture_type}"
        if self.is_sampler_type(node.param_type):
            return f"{prefix}{parameter_name}: {self.sampler_type(node.param_type)}"
        if self.is_buffer_pointer_type(node.param_type, node.qualifiers):
            return (
                f"{prefix}{parameter_name}: "
                f"{self.buffer_pointer_parameter_type(node.param_type)}"
            )
        parameter = (
            f"{prefix}{parameter_name}: {self.type_name_string(node.param_type)}"
        )
        resource_parameters = self.resource_member_parameter_declarations(
            node.name, node.param_type
        )
        if resource_parameters:
            parameter += ", " + ", ".join(resource_parameters)
        return parameter

    def stage_implicit_builtin_parameters(self, function, stage_name):
        referenced = self.stage_direct_builtin_references(function)
        workgroup_size_references = self.direct_workgroup_size_references(function)
        if not referenced and not workgroup_size_references:
            return ()

        if workgroup_size_references and stage_name != "compute":
            raise WGSLRefusal(
                "WGSL target does not support gl_WorkGroupSize outside compute stages"
            )

        existing = self.existing_parameter_builtin_names(function)

        unsupported = sorted(
            original
            for original, builtin in referenced.items()
            if builtin not in self.INPUT_BUILTIN_TYPE_MAP
        )
        if unsupported:
            raise WGSLRefusal(
                "WGSL target does not support implicit builtin identifier(s): "
                + ", ".join(unsupported)
                + "; model them as entry-point parameters or return values"
            )

        stage_allowed = self.STAGE_INPUT_BUILTINS.get(stage_name, set())
        invalid_for_stage = sorted(
            original
            for original, builtin in referenced.items()
            if builtin in self.INPUT_BUILTIN_TYPE_MAP and builtin not in stage_allowed
        )
        if invalid_for_stage:
            raise WGSLRefusal(
                "WGSL target does not support implicit builtin identifier(s) "
                f"in {stage_name} stage: "
                + ", ".join(invalid_for_stage)
                + "; model them as entry-point parameters or return values"
            )

        parameter_names = {
            getattr(param, "name", "") for param in getattr(function, "parameters", [])
        }
        colliding_builtins = sorted(
            builtin
            for builtin in set(referenced.values())
            if builtin in parameter_names and builtin not in existing
        )
        if colliding_builtins:
            builtin = colliding_builtins[0]
            raise WGSLRefusal(
                "WGSL target cannot inject builtin "
                f"{builtin} because an entry parameter already uses that name "
                f"without @builtin({builtin})"
            )

        parameters = []
        for builtin, builtin_type in self.INPUT_BUILTIN_TYPE_MAP.items():
            if builtin not in referenced.values() or builtin in existing:
                continue
            parameters.append(f"@builtin({builtin}) {builtin}: {builtin_type}")
        return tuple(parameters)

    def validate_helper_function_references(self, function):
        builtin_references = self.stage_direct_builtin_references(function)
        if builtin_references:
            raise WGSLRefusal(
                "WGSL target does not support direct builtin identifier(s) "
                f"in helper function {function.name}: "
                + ", ".join(sorted(builtin_references))
                + "; pass builtin values through entry-point parameters instead"
            )

        workgroup_size_references = self.direct_workgroup_size_references(function)
        if workgroup_size_references:
            raise WGSLRefusal(
                "WGSL target does not support gl_WorkGroupSize inside helper "
                f"function {function.name}; keep it directly in compute entry points"
            )

        barrier_calls = self.function_barrier_call_names(function)
        if barrier_calls and self._current_stage_name != "compute":
            raise WGSLRefusal(
                "WGSL target does not support barrier() inside helper function "
                f"{function.name}, which is not reached only from compute entry "
                "points; WGSL rejects a barrier a vertex or fragment entry point reaches"
            )

    def direct_workgroup_size_references(self, function):
        body = getattr(function, "body", None)
        if body is None or not hasattr(body, "walk"):
            return ()
        references = []
        for node in body.walk():
            if (
                isinstance(node, IdentifierNode)
                and node.name in self.WORKGROUP_SIZE_IDENTIFIER_ALIASES
            ):
                references.append(node.name)
        return tuple(references)

    def function_barrier_call_names(self, function):
        body = getattr(function, "body", None)
        if body is None or not hasattr(body, "walk"):
            return ()
        calls = []
        for node in body.walk():
            if not isinstance(node, FunctionCallNode):
                continue
            function_name = self.expression_name(node.function)
            if self.semantic_key(function_name) in self.BARRIER_FUNCTION_NAMES:
                calls.append(function_name)
        return tuple(calls)

    def stage_direct_builtin_references(self, function):
        body = getattr(function, "body", None)
        if body is None or not hasattr(body, "walk"):
            return {}
        referenced = {}
        for node in body.walk():
            wave_builtin = wgsl_wave_builtin_inputs(node)
            if wave_builtin:
                # WaveGetLaneIndex() and the operations written with it read the built-in input.
                referenced[f"{node.operation}()"] = wave_builtin
            if not isinstance(node, IdentifierNode):
                continue
            builtin = self.BUILTIN_IDENTIFIER_ALIASES.get(node.name)
            if builtin:
                referenced[node.name] = builtin
        return referenced

    def existing_parameter_builtin_names(self, function):
        builtin_names = set()
        for param in function.parameters:
            for attr in getattr(param, "attributes", []) or []:
                key = self.semantic_key(str(getattr(attr, "name", attr)))
                builtin = self.BUILTIN_SEMANTICS.get(key)
                if builtin:
                    builtin_names.add(builtin)
        return builtin_names

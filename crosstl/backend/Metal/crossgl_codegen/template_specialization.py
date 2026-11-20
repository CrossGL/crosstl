"""Function template argument deduction, specialization keys and template text substitution."""

import re

from ..call_template_deduction import (
    TemplateDeductionConflict,
    deduce_omitted_template_arguments,
)
from ..preprocessor import MetalTemplateSpecializationError
from ..template_text import substitute_template_identifiers
from .support import MetalTemplateArgumentResolutionError


class MetalTemplateSpecializationMixin:
    """Members of ``MetalToCrossGLConverter``: function template argument deduction, specialization keys and template text substitution."""

    def bind_concrete_function_template_arguments(
        self,
        function,
        explicit_arguments,
        *,
        selected_call,
        deduced_arguments=None,
    ):
        parameters = [
            entry
            for entry in getattr(function, "template_parameters", None) or []
            if isinstance(entry, (tuple, list)) and len(entry) >= 2 and entry[1]
        ]
        defaults = getattr(function, "template_parameter_defaults", {}) or {}
        deduced_arguments = dict(deduced_arguments or {})
        explicit_arguments = list(explicit_arguments or [])
        if len(explicit_arguments) > len(parameters):
            raise MetalTemplateArgumentResolutionError(
                function.name,
                "<arguments>",
                ", ".join(str(argument) for argument in explicit_arguments),
                selected_call,
                "supplies more arguments than the selected template declares",
                "explicit",
                getattr(function, "template_source_location", None)
                or getattr(function, "source_location", None),
            )

        all_bindings = {}
        value_bindings = {}
        type_bindings = {}
        for index, (kind, name) in enumerate(parameters):
            explicit = index < len(explicit_arguments)
            deduced = not explicit and name in deduced_arguments
            if explicit:
                expression = explicit_arguments[index]
            elif deduced:
                expression = deduced_arguments[name]
            else:
                expression = defaults.get(name)
            if expression is None:
                raise MetalTemplateArgumentResolutionError(
                    function.name,
                    name,
                    None,
                    selected_call,
                    "was not supplied and has no declaration default",
                    "missing",
                    getattr(function, "template_source_location", None)
                    or getattr(function, "source_location", None),
                )
            if kind == "value":
                resolved = self.resolve_value_template_constant(
                    expression,
                    all_bindings,
                    function,
                    name,
                    selected_call,
                    is_default=not (explicit or deduced),
                )
                value_bindings[name] = resolved
            else:
                resolved = self.resolve_concrete_template_type_argument(
                    expression,
                    all_bindings,
                    function,
                    name,
                    selected_call,
                    is_default=not (explicit or deduced),
                )
                type_bindings[name] = resolved
            all_bindings[name] = resolved
        return value_bindings, type_bindings

    def deduce_omitted_call_template_arguments(
        self,
        function,
        explicit_arguments,
        call_arguments,
        *,
        selected_call,
    ):
        """Deduce the template arguments an explicit call list leaves out."""
        template_parameters = [
            (entry[0], entry[1])
            for entry in getattr(function, "template_parameters", None) or []
            if isinstance(entry, (tuple, list)) and len(entry) >= 2 and entry[1]
        ]
        explicit_count = len(explicit_arguments or [])
        if explicit_count >= len(template_parameters):
            return {}
        parameters = getattr(function, "params", None) or []
        argument_types = [
            self.expression_metal_type(argument) for argument in call_arguments
        ]
        try:
            return deduce_omitted_template_arguments(
                template_parameters,
                explicit_count,
                [getattr(parameter, "vtype", None) for parameter in parameters],
                argument_types,
            )
        except TemplateDeductionConflict as conflict:
            raise MetalTemplateArgumentResolutionError(
                function.name,
                conflict.parameter_name,
                ", ".join(conflict.values),
                selected_call,
                str(conflict),
                "deduced",
                getattr(function, "template_source_location", None)
                or getattr(function, "source_location", None),
            ) from conflict

    def resolve_concrete_template_type_argument(
        self,
        expression,
        earlier_bindings,
        function,
        parameter_name,
        selected_call,
        *,
        is_default,
    ):
        text = self.substitute_template_value_text(
            str(expression),
            bindings=earlier_bindings,
            honor_shadowing=False,
        ).strip()
        active_type_bindings = {
            name: value
            for scope in self.template_type_bindings
            for name, value in scope.items()
        }
        type_bindings = dict(active_type_bindings)
        type_bindings.update(earlier_bindings)
        for name in sorted(type_bindings, key=len, reverse=True):
            text = re.sub(
                rf"\b{re.escape(name)}\b",
                str(type_bindings[name]),
                text,
            )
        text = re.sub(r"^typename\s+", "", text).strip()
        text = self.resolve_local_type_aliases(text)
        if text in self.local_struct_type_aliases:
            text = self.local_struct_type_aliases[text]
        text = self.resolve_type_alias(text)

        dependent_names = {
            name
            for candidate in (self.current_function, function)
            for _kind, name in getattr(candidate, "template_parameters", None) or []
            if name
        }
        identifiers = set(re.findall(r"\b[A-Za-z_]\w*\b", str(text)))
        unresolved = sorted(identifiers.intersection(dependent_names))
        if not text or unresolved:
            reason = (
                f"remains dependent on {', '.join(unresolved)}"
                if unresolved
                else "does not identify a concrete type"
            )
            raise MetalTemplateArgumentResolutionError(
                function.name,
                parameter_name,
                str(expression),
                selected_call,
                reason,
                "default_type" if is_default else "explicit_type",
                getattr(function, "template_source_location", None)
                or getattr(function, "source_location", None),
            )
        return re.sub(r"\s+", " ", str(text)).strip()

    def concrete_template_specialization_key(
        self,
        function,
        value_bindings,
        type_bindings,
    ):
        return (
            id(function),
            tuple(
                (
                    kind,
                    (value_bindings[name] if kind == "value" else type_bindings[name]),
                )
                for kind, name in getattr(function, "template_parameters", None) or []
                if name
            ),
        )

    def concrete_template_specialization_arguments(
        self,
        function,
        value_bindings,
        type_bindings,
    ):
        return [
            value_bindings[name] if kind == "value" else type_bindings[name]
            for kind, name in getattr(function, "template_parameters", None) or []
            if name
        ]

    def value_template_specialization_identifier(
        self,
        function,
        value_bindings,
        type_bindings,
    ):
        values = self.concrete_template_specialization_arguments(
            function,
            value_bindings,
            type_bindings,
        )
        parts = [function.name, *values]
        identifier = "_".join(
            part
            for part in (
                re.sub(r"[^A-Za-z0-9_]+", "_", str(value)).strip("_") for value in parts
            )
            if part
        )
        identifier = identifier or function.name
        if len(self.value_template_functions.get(function.name, [])) > 1:
            signature = "_".join(self.metal_function_source_signature(function))
            if signature:
                identifier = f"{identifier}_{signature}"
        return self.sanitize_identifier(identifier)

    def reserve_value_template_specialization_identifier(
        self,
        function,
        value_bindings,
        type_bindings,
    ):
        base = self.value_template_specialization_identifier(
            function,
            value_bindings,
            type_bindings,
        )
        candidate = base
        suffix = 2
        while candidate in self.existing_function_names:
            candidate = f"{base}_{suffix}"
            suffix += 1
        self.existing_function_names.add(candidate)
        return candidate

    def materialized_value_template_call_name(
        self,
        name,
        call_arguments,
        source_location=None,
    ):
        if self.preserve_unmaterialized_template_calls:
            return name
        parsed = self.split_value_template_call_name(name)
        if parsed is None:
            return name
        function_name, arguments = parsed
        try:
            function = self.resolve_value_template_function(
                function_name,
                call_arguments,
                explicit_template_arguments=arguments,
            )
            if function is None:
                return name
            value_bindings, type_bindings = (
                self.bind_concrete_function_template_arguments(
                    function,
                    arguments,
                    selected_call=str(name),
                    deduced_arguments=self.deduce_omitted_call_template_arguments(
                        function,
                        arguments,
                        call_arguments,
                        selected_call=str(name),
                    ),
                )
            )
        except MetalTemplateArgumentResolutionError as error:
            raise self.contextualize_template_call_resolution_error(
                error,
                name,
                source_location,
            ) from error
        key = self.concrete_template_specialization_key(
            function,
            value_bindings,
            type_bindings,
        )
        self.record_value_template_specialization_dependency(key)
        existing = self.value_template_specializations.get(key)
        if existing is not None:
            return existing

        requested_arguments = tuple(
            str(argument)
            for argument in self.concrete_template_specialization_arguments(
                function,
                value_bindings,
                type_bindings,
            )
        )
        next_count = self.materialized_template_specialization_count + 1
        if next_count > self.max_template_specializations:
            requested_signature = f"{function.name}<{', '.join(requested_arguments)}>"
            raise MetalTemplateSpecializationError(
                "Metal template specialization limit exceeded while "
                f"materializing '{requested_signature}'; {next_count} unique "
                f"concrete signatures requested, limit "
                f"{self.max_template_specializations} from "
                f"{self.template_specialization_limit_source}.",
                limit=self.max_template_specializations,
                limit_source=self.template_specialization_limit_source,
                unique_specialization_count=next_count,
                requested_signature=requested_signature,
                source_location=source_location,
                caller_specialization=self.current_function_specialization,
                callee_template=function.name,
                requested_arguments=requested_arguments,
            )
        specialized_name = self.reserve_value_template_specialization_identifier(
            function,
            value_bindings,
            type_bindings,
        )
        self.materialized_template_specialization_count = next_count
        self.value_template_specializations[key] = specialized_name
        self.suppressed_value_template_function_ids.add(id(function))
        self.pending_value_template_specializations.append(
            (
                function,
                value_bindings,
                type_bindings,
                specialized_name,
                key,
            )
        )
        self.user_function_names.add(specialized_name)
        self.user_function_overloads_by_name.setdefault(specialized_name, []).append(
            function
        )
        return specialized_name

    def contextualize_template_call_resolution_error(
        self,
        error,
        call_name,
        source_location,
    ):
        if (
            getattr(error, "enclosing_specialization", None)
            or self.current_function_specialization_key is None
        ):
            return error
        return MetalTemplateArgumentResolutionError(
            error.function_name,
            error.parameter_name,
            error.argument_expression,
            error.selected_call,
            error.reason,
            error.argument_kind,
            source_location or error.source_location,
            requested_specialization=(
                getattr(error, "requested_specialization", None) or error.selected_call
            ),
            enclosing_function=self.current_function_name,
            enclosing_specialization=self.current_function_specialization,
            nested_helper=str(call_name),
            enclosing_context="template call",
        )

    def record_value_template_specialization_dependency(self, callee_key):
        caller_key = self.current_function_specialization_key
        if caller_key is None:
            return
        dependencies = self.value_template_specialization_dependencies.setdefault(
            caller_key,
            [],
        )
        if callee_key not in dependencies:
            dependencies.append(callee_key)

    def ordered_value_template_specialization_keys(self, entries):
        states = {}
        ordered = []
        stack = []

        def visit(key):
            state = states.get(key, 0)
            if state == 2:
                return
            if state == 1:
                cycle_start = stack.index(key)
                cycle = stack[cycle_start:] + [key]
                cycle_names = [
                    self.value_template_specializations.get(item, "<specialization>")
                    for item in cycle
                ]
                function, value_bindings, type_bindings, _name = entries[key]
                requested_arguments = tuple(
                    str(argument)
                    for argument in self.concrete_template_specialization_arguments(
                        function,
                        value_bindings,
                        type_bindings,
                    )
                )
                raise MetalTemplateSpecializationError(
                    "Metal template specialization graph is recursive: "
                    + " -> ".join(cycle_names),
                    requested_signature=(
                        f"{function.name}<{', '.join(requested_arguments)}>"
                    ),
                    caller_specialization=cycle_names[-2],
                    callee_template=function.name,
                    requested_arguments=requested_arguments,
                )

            states[key] = 1
            stack.append(key)
            for dependency in self.value_template_specialization_dependencies.get(
                key,
                [],
            ):
                if dependency in entries:
                    visit(dependency)
            stack.pop()
            states[key] = 2
            ordered.append(key)

        for key in entries:
            visit(key)
        return ordered

    def template_value_binding_is_shadowed(self, name):
        return any(
            name in scope for scope in reversed(self.template_binding_shadow_scopes)
        )

    def template_value_binding(self, name):
        if self.template_value_binding_is_shadowed(name):
            return None
        for bindings in reversed(self.template_value_bindings):
            if name in bindings:
                return bindings[name]
        return None

    def substitute_template_value_text(
        self, text, *, bindings=None, honor_shadowing=True
    ):
        text = str(text)
        if bindings is None:
            bindings = {
                name: value
                for scope in self.template_value_bindings
                for name, value in scope.items()
                if not honor_shadowing
                or not self.template_value_binding_is_shadowed(name)
            }
        if not bindings:
            return text
        return substitute_template_identifiers(text, bindings)

    def substitute_template_type_text(self, text, *, bindings=None):
        text = str(text)
        if bindings is None:
            bindings = {
                name: value
                for scope in self.template_type_bindings
                for name, value in scope.items()
            }
        if not bindings:
            return text
        return substitute_template_identifiers(text, bindings)

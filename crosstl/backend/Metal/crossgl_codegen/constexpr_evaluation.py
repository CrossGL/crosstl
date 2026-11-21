"""Constexpr helper calls and struct static constexpr values."""

import re

from ....translator.codegen.array_utils import evaluate_literal_int_expression
from ..MetalAst import (
    AssignmentNode,
    ReturnNode,
    VariableNode,
)
from ..MetalLexer import MetalLexer
from ..MetalParser import MetalParser
from .support import MetalTemplateArgumentResolutionError


class MetalConstexprMixin:
    """Members of ``MetalToCrossGLConverter``: constexpr helper calls and struct static constexpr values."""

    def function_is_constexpr(self, function):
        return "constexpr" in {
            str(qualifier).lower()
            for qualifier in getattr(function, "declaration_qualifiers", []) or []
        }

    def push_materialized_constexpr_expression_context(
        self,
        expression,
        *,
        required,
        allow_concrete_without_bindings=False,
    ):
        if (
            not self.current_function_materialization_bindings
            and not allow_concrete_without_bindings
        ):
            return False
        self.materialized_constexpr_expression_contexts.append(
            {
                "function": self.current_function_name,
                "specialization": self.current_function_specialization,
                "required": required,
                "source_location": (
                    getattr(expression, "source_location", None)
                    or getattr(self.current_function, "source_location", None)
                ),
            }
        )
        return True

    def render_constexpr_helper_call(self, call, is_main=False):
        owner = self.current_struct_static_constant_owner
        resolving_static_member = bool(
            owner is not None and self.struct_static_constant_resolution_stack
        )
        materialization_context = (
            self.materialized_constexpr_expression_contexts[-1]
            if self.materialized_constexpr_expression_contexts
            else None
        )
        if not resolving_static_member and materialization_context is None:
            return None

        call_name = str(getattr(call, "name", ""))
        parsed = self.split_value_template_call_name(
            call_name,
            function_index=self.constexpr_value_template_functions,
        )
        if parsed is None:
            function_name = call_name.rsplit("::", 1)[-1]
            if function_name not in self.constexpr_value_template_functions:
                return None
            explicit_arguments = None
        else:
            function_name, explicit_arguments = parsed

        try:
            function = self.resolve_value_template_function(
                function_name,
                getattr(call, "args", []) or [],
                explicit_template_arguments=explicit_arguments,
                function_index=self.constexpr_value_template_functions,
                prefer_constexpr_non_template=True,
            )
            if function is None:
                return None
            rendered = self.evaluate_struct_static_constexpr_helper(
                function,
                explicit_arguments,
                getattr(call, "args", []) or [],
                selected_call=call_name,
                is_main=is_main,
            )
        except MetalTemplateArgumentResolutionError as error:
            if (
                materialization_context is not None
                and not materialization_context["required"]
            ):
                return None
            if materialization_context is None or getattr(
                error, "enclosing_specialization", None
            ):
                raise
            raise self.contextualize_materialized_constexpr_resolution_error(
                error,
                call,
                materialization_context,
            ) from error
        if resolving_static_member:
            self.struct_static_constexpr_member_keys.add(
                self.struct_static_constant_resolution_stack[-1]
            )
        return rendered

    def contextualize_materialized_constexpr_resolution_error(
        self,
        error,
        call,
        context,
    ):
        return MetalTemplateArgumentResolutionError(
            error.function_name,
            error.parameter_name,
            error.argument_expression,
            error.selected_call,
            error.reason,
            error.argument_kind,
            getattr(call, "source_location", None)
            or error.source_location
            or context["source_location"],
            requested_specialization=(
                getattr(error, "requested_specialization", None) or error.selected_call
            ),
            enclosing_function=context["function"],
            enclosing_specialization=context["specialization"],
            nested_helper=str(getattr(call, "name", error.function_name)),
        )

    def evaluate_struct_static_constexpr_helper(
        self,
        function,
        explicit_arguments,
        call_arguments,
        *,
        selected_call,
        is_main,
    ):
        (
            template_bindings,
            value_bindings,
            type_bindings,
        ) = self.bind_struct_static_constexpr_template_arguments(
            function,
            explicit_arguments,
            selected_call=selected_call,
        )
        runtime_bindings = self.bind_struct_static_constexpr_call_arguments(
            function,
            call_arguments,
            selected_call=selected_call,
            is_main=is_main,
        )
        requested_specialization = self.constexpr_helper_specialization_name(
            function,
            template_bindings,
        )
        cache_key = (
            id(function),
            tuple(
                template_bindings.get(name)
                for _kind, name in getattr(function, "template_parameters", None) or []
                if name
            ),
            tuple(
                runtime_bindings.get(getattr(parameter, "name", None))
                for parameter in getattr(function, "params", []) or []
            ),
        )
        cached = self.constexpr_helper_values.get(cache_key)
        if cached is not None:
            return cached
        if cache_key in self.constexpr_helper_resolution_stack:
            raise MetalTemplateArgumentResolutionError(
                function.name,
                "<return>",
                requested_specialization,
                selected_call,
                "has a cyclic constexpr helper dependency",
                "constexpr_body",
                getattr(function, "template_source_location", None)
                or getattr(function, "source_location", None),
                requested_specialization=requested_specialization,
            )

        body = list(getattr(function, "body", None) or [])
        local_declarations = body[:-1]
        if (
            not body
            or not isinstance(body[-1], ReturnNode)
            or any(
                not self.is_supported_constexpr_local_declaration(statement)
                for statement in local_declarations
            )
        ):
            raise MetalTemplateArgumentResolutionError(
                function.name,
                "<return>",
                requested_specialization,
                selected_call,
                "contains statements outside the supported constexpr local "
                "declaration and return subset",
                "constexpr_body",
                getattr(function, "template_source_location", None)
                or getattr(function, "source_location", None),
                requested_specialization=requested_specialization,
            )

        return_type = self.substitute_template_value_text(
            getattr(function, "return_type", ""),
            bindings=type_bindings,
            honor_shadowing=False,
        )
        if not self.is_integral_constexpr_type(return_type):
            raise MetalTemplateArgumentResolutionError(
                function.name,
                "<return>",
                return_type,
                selected_call,
                "does not have an integral or boolean return type",
                "constexpr_body",
                getattr(function, "template_source_location", None)
                or getattr(function, "source_location", None),
                requested_specialization=requested_specialization,
            )

        previous_type_aliases = dict(self.type_aliases)
        previous_shadow_scopes = self.template_binding_shadow_scopes
        self.type_aliases.update(type_bindings)
        active_value_bindings = dict(value_bindings)
        active_value_bindings.update(runtime_bindings)
        self.template_value_bindings.append(active_value_bindings)
        self.template_binding_shadow_scopes = []
        self.constexpr_helper_resolution_stack.append(cache_key)
        try:
            for declaration in local_declarations:
                local_name = declaration.left.name
                local_expression = self.generate_expression(
                    declaration.right,
                    is_main,
                )
                local_value = self.normalize_struct_static_constexpr_expression(
                    local_expression,
                    function,
                    local_name,
                    selected_call,
                    requested_specialization,
                )
                active_value_bindings[local_name] = local_value
            rendered = self.generate_expression(body[-1].value, is_main)
        finally:
            self.constexpr_helper_resolution_stack.pop()
            self.template_binding_shadow_scopes = previous_shadow_scopes
            self.template_value_bindings.pop()
            self.type_aliases = previous_type_aliases

        result = self.normalize_struct_static_constexpr_expression(
            rendered,
            function,
            "<return>",
            selected_call,
            requested_specialization,
        )

        self.constexpr_helper_values[cache_key] = result
        return result

    def is_supported_constexpr_local_declaration(self, statement):
        if (
            not isinstance(statement, AssignmentNode)
            or getattr(statement, "operator", None) != "="
            or not isinstance(getattr(statement, "left", None), VariableNode)
            or not getattr(statement.left, "name", None)
        ):
            return False
        qualifiers = {
            str(qualifier).lower()
            for qualifier in getattr(statement.left, "qualifiers", []) or []
        }
        return "constexpr" in qualifiers and self.is_integral_constexpr_type(
            getattr(statement.left, "vtype", None)
        )

    def normalize_struct_static_constexpr_expression(
        self,
        rendered,
        function,
        expression_name,
        selected_call,
        requested_specialization,
    ):
        folded = self.evaluate_value_template_constant_expression(rendered)
        if folded is not None:
            return str(folded)

        dependencies = set(re.findall(r"\b[A-Za-z_]\w*\b", rendered))
        owner_parameters = self.current_struct_template_value_parameters()
        allowed = set(owner_parameters)
        allowed.update(self.constexpr_integral_type_names())
        allowed.update({"true", "false"})
        unresolved = sorted(dependencies - allowed)
        if unresolved or not dependencies.intersection(owner_parameters):
            reason = (
                f"remains dependent on {', '.join(unresolved)}"
                if unresolved
                else "cannot be evaluated as a supported integral constant expression"
            )
            raise MetalTemplateArgumentResolutionError(
                function.name,
                expression_name,
                rendered,
                selected_call,
                reason,
                "constexpr_body",
                getattr(function, "template_source_location", None)
                or getattr(function, "source_location", None),
                requested_specialization=requested_specialization,
            )
        return f"({rendered})"

    def bind_struct_static_constexpr_template_arguments(
        self,
        function,
        explicit_arguments,
        *,
        selected_call,
    ):
        parameters = getattr(function, "template_parameters", None) or []
        defaults = getattr(function, "template_parameter_defaults", {}) or {}
        explicit_arguments = list(explicit_arguments or [])
        template_bindings = {}
        value_bindings = {}
        type_bindings = {}
        argument_index = 0

        for kind, name in parameters:
            if not name:
                continue
            explicit = argument_index < len(explicit_arguments)
            if explicit:
                expression = explicit_arguments[argument_index]
                argument_index += 1
                argument_kind = "explicit" if kind == "value" else "explicit_type"
            else:
                expression = defaults.get(name)
                argument_kind = "default" if kind == "value" else "default_type"
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
                resolved = self.resolve_struct_static_constexpr_value_argument(
                    expression,
                    template_bindings,
                    function,
                    name,
                    selected_call,
                    argument_kind,
                )
                value_bindings[name] = resolved
            else:
                resolved = self.resolve_struct_static_constexpr_type_argument(
                    expression,
                    template_bindings,
                    function,
                    name,
                    selected_call,
                    argument_kind,
                )
                type_bindings[name] = resolved
            template_bindings[name] = resolved

        if argument_index != len(explicit_arguments):
            return {}, {}, {}
        return template_bindings, value_bindings, type_bindings

    def resolve_struct_static_constexpr_value_argument(
        self,
        expression,
        earlier_bindings,
        function,
        parameter_name,
        selected_call,
        argument_kind,
    ):
        text = self.struct_static_constexpr_argument_text(
            expression,
            earlier_bindings,
        )
        normalized = re.sub(r"\s+", " ", text).strip()
        if normalized in {"true", "false"}:
            return normalized
        integer = re.fullmatch(r"([-+]?\d+)(?:[uUlL]+)?", normalized)
        if integer is not None:
            return integer.group(1)
        value = self.evaluate_value_template_constant_expression(normalized)
        if value is None:
            value = self.fold_constexpr_helper_calls_in_constant_text(normalized)
        if value is not None:
            return str(value)

        identifiers = sorted(set(re.findall(r"\b[A-Za-z_]\w*\b", normalized)))
        unresolved = sorted(
            set(identifiers) - self.current_struct_template_value_parameters()
        )
        if identifiers and not unresolved:
            return normalized
        reason = (
            f"remains dependent on {', '.join(unresolved or identifiers)}"
            if identifiers
            else "cannot be evaluated as a supported integral constant expression"
        )
        raise MetalTemplateArgumentResolutionError(
            function.name,
            parameter_name,
            str(expression),
            selected_call,
            reason,
            argument_kind,
            getattr(function, "template_source_location", None)
            or getattr(function, "source_location", None),
        )

    def resolve_struct_static_constexpr_type_argument(
        self,
        expression,
        earlier_bindings,
        function,
        parameter_name,
        selected_call,
        argument_kind,
    ):
        text = self.struct_static_constexpr_argument_text(
            expression,
            earlier_bindings,
        )
        identifiers = set(re.findall(r"\b[A-Za-z_]\w*\b", text))
        helper_parameters = {
            name
            for _kind, name in getattr(function, "template_parameters", None) or []
            if name
        }
        owner_parameters = set(
            self.struct_template_parameters.get(
                self.current_struct_static_constant_owner,
                {},
            )
        )
        unresolved = sorted((identifiers & helper_parameters) - owner_parameters)
        if unresolved:
            raise MetalTemplateArgumentResolutionError(
                function.name,
                parameter_name,
                str(expression),
                selected_call,
                f"remains dependent on {', '.join(unresolved)}",
                argument_kind,
                getattr(function, "template_source_location", None)
                or getattr(function, "source_location", None),
            )
        return text.strip()

    def struct_static_constexpr_argument_text(self, expression, earlier_bindings):
        if self.materialized_constexpr_expression_contexts:
            bindings = {
                name: value
                for scope in self.template_value_bindings
                for name, value in scope.items()
                if not self.template_value_binding_is_shadowed(name)
            }
        else:
            bindings = dict(self.struct_static_owner_member_bindings())
            for scope in self.template_value_bindings:
                bindings.update(scope)
        bindings.update(earlier_bindings)
        return self.substitute_template_value_text(
            str(expression),
            bindings=bindings,
            honor_shadowing=False,
        )

    def struct_static_owner_member_bindings(self):
        owner = self.current_struct_static_constant_owner
        return {
            member_name: value
            for (
                candidate_owner,
                member_name,
            ), value in self.struct_static_constants.items()
            if candidate_owner == owner
        }

    def current_struct_template_value_parameters(self):
        return {
            name
            for name, kind in self.struct_template_parameters.get(
                self.current_struct_static_constant_owner,
                {},
            ).items()
            if kind == "value"
        }

    def bind_struct_static_constexpr_call_arguments(
        self,
        function,
        call_arguments,
        *,
        selected_call,
        is_main,
    ):
        bindings = {}
        allowed = self.current_struct_template_value_parameters()
        for parameter, argument in zip(
            getattr(function, "params", []) or [],
            call_arguments,
        ):
            rendered = self.generate_expression(argument, is_main)
            folded = self.evaluate_value_template_constant_expression(rendered)
            if folded is not None:
                rendered = str(folded)
            else:
                identifiers = set(re.findall(r"\b[A-Za-z_]\w*\b", rendered))
                unresolved = sorted(identifiers - allowed)
                if unresolved or not identifiers:
                    reason = (
                        f"remains dependent on {', '.join(unresolved)}"
                        if unresolved
                        else "cannot be evaluated as a supported integral "
                        "constant expression"
                    )
                    raise MetalTemplateArgumentResolutionError(
                        function.name,
                        getattr(parameter, "name", None) or "<argument>",
                        rendered,
                        selected_call,
                        reason,
                        "call",
                        getattr(argument, "source_location", None)
                        or getattr(function, "source_location", None),
                    )
            bindings[getattr(parameter, "name", None)] = rendered
        return bindings

    def constexpr_helper_specialization_name(self, function, bindings):
        parameters = [
            name
            for _kind, name in getattr(function, "template_parameters", None) or []
            if name
        ]
        if not parameters:
            return function.name
        arguments = ",".join(str(bindings[name]) for name in parameters)
        return f"{function.name}<{arguments}>"

    def is_integral_constexpr_type(self, type_name):
        resolved = self.normalized_metal_type(self.resolve_type_alias(type_name))
        return resolved in self.constexpr_integral_type_names()

    @staticmethod
    def constexpr_integral_type_names():
        return {
            "bool",
            "char",
            "uchar",
            "short",
            "ushort",
            "int",
            "uint",
            "long",
            "ulong",
            "size_t",
            "ptrdiff_t",
            "int8_t",
            "uint8_t",
            "int16_t",
            "uint16_t",
            "int32_t",
            "uint32_t",
            "int64_t",
            "uint64_t",
        }

    def bind_value_template_arguments(
        self,
        function,
        explicit_arguments,
        *,
        selected_call,
        require_defaults=True,
    ):
        parameters = getattr(function, "template_parameters", None) or []
        defaults = getattr(function, "template_parameter_defaults", {}) or {}
        explicit_arguments = list(explicit_arguments or [])
        bindings = {}
        argument_index = 0
        for entry in parameters:
            if not isinstance(entry, (tuple, list)) or len(entry) < 2:
                continue
            kind, name = entry[0], entry[1]
            if not name:
                continue
            explicit = argument_index < len(explicit_arguments)
            if explicit:
                argument = explicit_arguments[argument_index]
                argument_index += 1
                if kind != "value":
                    raise MetalTemplateArgumentResolutionError(
                        function.name,
                        name,
                        argument,
                        selected_call,
                        (
                            "cannot be preserved by the value-template "
                            "specialization identity"
                        ),
                        "explicit_type",
                        getattr(function, "template_source_location", None)
                        or getattr(function, "source_location", None),
                    )
                bindings[name] = self.resolve_value_template_constant(
                    argument,
                    bindings,
                    function,
                    name,
                    selected_call,
                    is_default=False,
                )
                continue
            if kind != "value":
                continue
            default = defaults.get(name)
            if default is None:
                if require_defaults:
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
                return None
            bindings[name] = self.resolve_value_template_constant(
                default,
                bindings,
                function,
                name,
                selected_call,
                is_default=True,
            )

        if argument_index != len(explicit_arguments):
            return None
        value_names = self.value_template_parameter_names(function)
        if any(name not in bindings for name in value_names):
            return None
        return bindings

    def resolve_value_template_constant(
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
            str(expression), bindings=earlier_bindings, honor_shadowing=False
        ).strip()
        active_bindings = {
            name: value
            for scope in self.template_value_bindings
            for name, value in scope.items()
            if not self.template_value_binding_is_shadowed(name)
        }
        if active_bindings:
            text = self.substitute_template_value_text(
                text, bindings=active_bindings, honor_shadowing=False
            ).strip()
        normalized = re.sub(r"\s+", " ", text)
        if normalized in {"true", "false"}:
            return normalized
        integer = re.fullmatch(r"([-+]?\d+)(?:[uUlL]+)?", normalized)
        if integer is not None:
            return integer.group(1)

        value = self.evaluate_value_template_constant_expression(normalized)
        if value is None:
            value = self.fold_constexpr_helper_calls_in_constant_text(normalized)
        if value is not None:
            return str(value)

        identifiers = sorted(set(re.findall(r"\b[A-Za-z_]\w*\b", normalized)))
        dependency = (
            f"remains dependent on {', '.join(identifiers)}"
            if identifiers
            else "cannot be evaluated as a supported integral constant expression"
        )
        raise MetalTemplateArgumentResolutionError(
            function.name,
            parameter_name,
            str(expression),
            selected_call,
            dependency,
            "default" if is_default else "explicit",
            getattr(function, "template_source_location", None)
            or getattr(function, "source_location", None),
        )

    def fold_constexpr_helper_calls_in_constant_text(self, text):
        """Evaluate a template argument that calls concrete constexpr helpers.

        A non-type template argument is a constant expression ([temp.arg.nontype]),
        so a call such as ``work_per_thread_float()`` (a constexpr helper the
        preprocessor materialized from a default argument) names one integral
        value. The helper body is evaluated with the constexpr helper evaluator;
        ``None`` keeps the caller's fail-closed diagnostic when any helper is
        unknown, unsupported or not provably constant.
        """

        helpers = self.constexpr_value_template_functions
        identifiers = set(re.findall(r"\b[A-Za-z_]\w*\b", text))
        if not identifiers.intersection(helpers) and not identifiers.intersection(
            {"sizeof", "alignof"}
        ):
            return None
        try:
            lexer = MetalLexer(text, preprocess=False)
            parser = MetalParser(lexer.tokenize())
            node = parser.parse_expression()
            if parser.current_token[0] != "EOF":
                return None
        except (SyntaxError, ValueError, TypeError):
            return None
        self.push_materialized_constexpr_expression_context(
            node,
            required=False,
            allow_concrete_without_bindings=True,
        )
        try:
            rendered = self.generate_expression(node, False)
        except MetalTemplateArgumentResolutionError:
            return None
        finally:
            self.materialized_constexpr_expression_contexts.pop()
        return self.evaluate_value_template_constant_expression(str(rendered))

    def evaluate_value_template_constant_expression(
        self, expression, *, constants=None
    ):
        try:
            lexer = MetalLexer(expression, preprocess=False)
            parser = MetalParser(lexer.tokenize())
            node = parser.parse_expression()
            if parser.current_token[0] != "EOF":
                return None
        except (SyntaxError, ValueError, TypeError):
            return None
        return evaluate_literal_int_expression(node, constants)

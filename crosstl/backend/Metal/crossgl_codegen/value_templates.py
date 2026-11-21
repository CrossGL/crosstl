"""Callable type aliases and value template functions."""

import re

from ..MetalAst import (
    CallableTypeAliasNode,
    FunctionCallNode,
    FunctionNode,
    TypeAliasNode,
)
from ..preprocessor import DEFAULT_EXPLICIT_TEMPLATE_SPECIALIZATION_LIMIT
from .support import (
    MetalCallableAliasLoweringError,
    MetalTemplateArgumentResolutionError,
)


class MetalValueTemplateMixin:
    """Members of ``MetalToCrossGLConverter``: callable type aliases and value template functions."""

    def collect_callable_type_aliases(self, aliases):
        alias_nodes = {
            alias.name: alias
            for alias in aliases or []
            if isinstance(alias, TypeAliasNode)
            and getattr(alias, "name", None)
            and not self.is_template_alias_declaration(alias)
        }
        callable_aliases = {
            name: alias
            for name, alias in alias_nodes.items()
            if isinstance(alias, CallableTypeAliasNode)
            or getattr(alias, "is_function_type", False)
        }

        changed = True
        while changed:
            changed = False
            for name, alias in alias_nodes.items():
                if name in callable_aliases:
                    continue
                referenced_name = self.plain_type_alias_reference(alias.alias_type)
                if referenced_name not in callable_aliases:
                    continue
                callable_aliases[name] = callable_aliases[referenced_name]
                changed = True

        self.callable_type_aliases = callable_aliases
        self.callable_alias_declarations = {
            name: alias_nodes[name] for name in callable_aliases
        }

    def plain_type_alias_reference(self, type_name):
        text = str(type_name or "").strip()
        while text.endswith(("*", "&")):
            text = text[:-1].rstrip()
        qualifier_pattern = (
            r"^(?:(?:const|volatile|thread|threadgroup|device|constant|"
            r"restrict|__restrict|__restrict__)\s+)+"
        )
        text = re.sub(qualifier_pattern, "", text).strip()
        if self.crossgl_identifier_pattern.fullmatch(text):
            return text
        return None

    def validate_runtime_callable_alias_usage(self, ast):
        if not self.callable_type_aliases:
            return

        for node in self.iter_runtime_ast_nodes(ast):
            for attribute in ("vtype", "return_type", "target_type", "vector_type"):
                type_name = getattr(node, attribute, None)
                alias_name = self.callable_alias_reference(type_name)
                if alias_name is None:
                    continue
                alias = self.callable_type_aliases[alias_name]
                declaration = self.callable_alias_declarations[alias_name]
                raise MetalCallableAliasLoweringError(
                    alias_name,
                    self.format_callable_alias_signature(alias, alias_name),
                    self.callable_alias_usage(node, attribute),
                    "runtime callable values are not representable in CrossGL; "
                    "materialize the callback as a non-type template argument "
                    "or use a supported Metal function-table resource",
                    source_location=(
                        getattr(node, "source_location", None)
                        or getattr(declaration, "source_location", None)
                        or getattr(alias, "source_location", None)
                    ),
                )

    def iter_runtime_ast_nodes(self, ast):
        pending = [
            value
            for name, value in getattr(ast, "__dict__", {}).items()
            if name not in {"typedefs", "includes"}
        ]
        seen = set()
        while pending:
            node = pending.pop()
            if node is None or isinstance(node, (str, int, float, bool)):
                continue
            if isinstance(node, dict):
                pending.extend(node.values())
                continue
            if isinstance(node, (list, tuple, set)):
                pending.extend(node)
                continue
            node_id = id(node)
            if node_id in seen:
                continue
            seen.add(node_id)
            if isinstance(node, TypeAliasNode):
                continue
            yield node
            pending.extend(self.iter_ast_children(node))

    def callable_alias_reference(self, type_name):
        text = str(type_name or "").strip()
        if not text:
            return None

        carrier = text
        while carrier.endswith(("*", "&")):
            carrier = carrier[:-1].rstrip()
        base_name, generic_args = self.generic_type_parts(carrier)
        if base_name in {
            "visible_function_table",
            "intersection_function_table",
        } and any(
            self.plain_type_alias_reference(argument) in self.callable_type_aliases
            for argument in generic_args
        ):
            return None

        for alias_name in sorted(self.callable_type_aliases, key=len, reverse=True):
            if re.search(
                rf"(?<![A-Za-z0-9_]){re.escape(alias_name)}(?![A-Za-z0-9_])",
                text,
            ):
                return alias_name
        return None

    def callable_alias_usage(self, node, attribute):
        name = getattr(node, "name", None)
        node_name = type(node).__name__
        if name:
            return f"{node_name} '{name}' {attribute}"
        return f"{node_name} {attribute}"

    def format_callable_alias_signature(self, alias, alias_name):
        return_type = str(
            getattr(alias, "return_type", None) or alias.alias_type
        ).strip()
        qualifiers = [
            str(value).strip()
            for value in getattr(alias, "qualifiers", []) or []
            if str(value).strip()
        ]
        if qualifiers:
            return_type = " ".join(qualifiers + [return_type])

        parameters = getattr(alias, "parameters", None)
        if parameters is None:
            parameter_text = "..."
        else:
            rendered_parameters = []
            for parameter in parameters:
                parts = [
                    str(value).strip()
                    for value in getattr(parameter, "qualifiers", []) or []
                    if str(value).strip()
                ]
                parts.append(str(getattr(parameter, "vtype", "")).strip())
                rendered_parameters.append(" ".join(part for part in parts if part))
            parameter_text = ", ".join(rendered_parameters)

        indirection = str(getattr(alias, "indirection", "") or "")
        declarator = f"({indirection}{alias_name})" if indirection else alias_name
        return f"{return_type} {declarator}({parameter_text})"

    def unbound_template_parameter_names(
        self, function, value_bindings=None, type_bindings=None
    ):
        """Named template parameters of ``function`` that no binding gives a value."""
        bound = {*(value_bindings or {}), *(type_bindings or {})}
        return [
            entry[1]
            for entry in getattr(function, "template_parameters", None) or []
            if isinstance(entry, (tuple, list))
            and len(entry) >= 2
            and entry[1]
            and entry[1] not in bound
        ]

    def value_template_parameter_names(self, function):
        return [
            entry[1]
            for entry in getattr(function, "template_parameters", None) or []
            if isinstance(entry, (tuple, list))
            and len(entry) >= 2
            and entry[0] == "value"
            and entry[1]
        ]

    def split_value_template_call_name(self, name, function_index=None):
        text = str(name).strip()
        if "<" not in text or not text.endswith(">"):
            return None
        angle = text.find("<")
        base = text[:angle].strip()
        if not base:
            return None
        arguments = self.split_generic_arguments(text[angle + 1 : -1])
        function_name = base.rsplit("::", 1)[-1]
        functions = (
            self.value_template_functions if function_index is None else function_index
        )
        if function_name not in functions:
            return None
        return function_name, arguments

    def map_generic_function_call_name(self, name):
        """``helper<int>`` for a call that names a user type-generic function with explicit type arguments.

        The function is emitted as a CrossGL ``generic`` declaration, so its call keeps the explicit arguments and
        CrossGL deduces the rest; sanitizing ``helper<int>`` into one identifier would name no function.
        """
        text = str(name).strip()
        if "<" not in text or not text.endswith(">"):
            return None
        compact = text.replace(" ", "")
        if any(
            str(function_name).replace(" ", "") == compact
            for function_name in self.user_function_names
        ):
            # An explicit specialization (``template<> uint f<uint>(...)``) is its own function.
            return None
        angle = text.find("<")
        base = text[:angle].strip()
        candidates = self.user_function_overloads_by_name.get(
            base.rsplit("::", 1)[-1], []
        )
        if not candidates or any(
            not getattr(candidate, "template_parameters", None)
            or any(
                kind not in {"typename", "class"}
                for kind, _parameter in candidate.template_parameters
            )
            for candidate in candidates
        ):
            return None
        arguments = self.split_generic_arguments(text[angle + 1 : -1])
        if not arguments or len(arguments) > min(
            len(candidate.template_parameters) for candidate in candidates
        ):
            return None
        mapped = [
            self.map_type(self.normalized_metal_type(self.resolve_type_alias(argument)))
            for argument in arguments
        ]
        return f"{self.sanitize_identifier(base)}<{', '.join(mapped)}>"

    def value_template_overload_signature(self, function):
        parameters = ", ".join(self.metal_function_source_signature(function))
        template_parameters = ", ".join(
            name
            for _kind, name in getattr(function, "template_parameters", None) or []
            if name
        )
        return f"{function.name}<{template_parameters}>({parameters})"

    def resolve_value_template_function(
        self,
        function_name,
        call_arguments,
        explicit_template_arguments=None,
        function_index=None,
        prefer_constexpr_non_template=False,
    ):
        functions = (
            self.value_template_functions if function_index is None else function_index
        )
        value_candidates = [
            function
            for function in functions.get(function_name, [])
            if len(getattr(function, "params", []) or []) == len(call_arguments)
        ]
        if not value_candidates:
            return None

        explicit_call = explicit_template_arguments is not None
        if explicit_call:
            explicit_count = len(explicit_template_arguments)
            value_candidates = [
                function
                for function in value_candidates
                if explicit_count
                <= len(getattr(function, "template_parameters", None) or [])
            ]
            overloads = value_candidates
        else:
            overloads = [
                function
                for function in self.user_function_overloads_by_name.get(
                    function_name, []
                )
                if len(getattr(function, "params", []) or []) == len(call_arguments)
            ]

        if not overloads:
            return None
        if len(overloads) == 1:
            return overloads[0] if overloads[0] in value_candidates else None

        argument_types = [
            self.expression_mapped_type(argument) for argument in call_arguments
        ]
        matching = []
        if all(argument_type is not None for argument_type in argument_types):
            matching = [
                function
                for function in overloads
                if self.metal_function_mapped_signature(function)
                == tuple(argument_types)
            ]
        if len(matching) == 1:
            return matching[0] if matching[0] in value_candidates else None

        # A non-template exact match wins over a same-signature function
        # template in C++. Ordinary call materialization leaves that overload
        # unchanged; constexpr evaluation can select it directly.
        non_template_matches = [
            function
            for function in matching
            if not self.value_template_parameter_names(function)
        ]
        if len(non_template_matches) == 1:
            if (
                prefer_constexpr_non_template
                and non_template_matches[0] in value_candidates
            ):
                return non_template_matches[0]
            return None

        ambiguous = matching or overloads
        ambiguous_value_candidates = [
            function for function in ambiguous if function in value_candidates
        ]
        if len(ambiguous_value_candidates) <= 1 and not explicit_call:
            # Unknown argument types plus an unrelated overload are not enough
            # evidence to select the template. Preserve the call unchanged.
            return None
        signatures = ", ".join(
            self.value_template_overload_signature(function)
            for function in ambiguous_value_candidates
        )
        representative = ambiguous_value_candidates[0]
        raise MetalTemplateArgumentResolutionError(
            function_name,
            "<overload>",
            signatures,
            f"{function_name}(...)",
            "does not identify one unique value-template declaration",
            "overload",
            getattr(representative, "template_source_location", None)
            or getattr(representative, "source_location", None),
        )

    def prepare_value_template_materializations(self, ast, functions):
        self.value_template_functions = {}
        for function in functions:
            if (
                isinstance(function, FunctionNode)
                and getattr(function, "name", None)
                and self.value_template_parameter_names(function)
            ):
                self.value_template_functions.setdefault(function.name, []).append(
                    function
                )
        self.constexpr_value_template_functions = {
            name: [
                function
                for function in candidates
                if self.function_is_constexpr(function)
            ]
            for name, candidates in self.value_template_functions.items()
            if any(self.function_is_constexpr(function) for function in candidates)
        }
        for function in functions:
            if (
                isinstance(function, FunctionNode)
                and getattr(function, "name", None)
                and not getattr(function, "template_parameters", None)
                and self.function_is_constexpr(function)
            ):
                self.constexpr_value_template_functions.setdefault(
                    function.name, []
                ).append(function)
        self.constexpr_helper_values = {}
        self.constexpr_helper_resolution_stack = []
        self.default_value_template_bindings = {}
        self.value_template_specializations = {}
        self.pending_value_template_specializations = []
        self.value_template_specialization_dependencies = {}
        self.current_function_specialization_key = None
        self.max_template_specializations = getattr(
            ast,
            "max_template_specializations",
            DEFAULT_EXPLICIT_TEMPLATE_SPECIALIZATION_LIMIT,
        )
        self.template_specialization_limit_source = getattr(
            ast,
            "template_specialization_limit_source",
            "max_template_specializations",
        )
        self.materialized_template_specialization_count = 0
        self.preserve_unmaterialized_template_calls = False
        self.suppressed_value_template_function_ids = set()
        self.existing_function_names = {
            function.name
            for function in functions
            if isinstance(function, FunctionNode) and getattr(function, "name", None)
        }
        if not self.value_template_functions:
            return

        omitted_calls = []
        seen = set()

        def visit(node):
            if node is None or isinstance(node, (str, int, float, bool)):
                return
            if not isinstance(node, (dict, list, tuple, set)):
                node_id = id(node)
                if node_id in seen:
                    return
                seen.add(node_id)
            if isinstance(node, FunctionCallNode):
                call_name = str(getattr(node, "name", ""))
                unscoped_name = call_name.rsplit("::", 1)[-1]
                if unscoped_name in self.value_template_functions:
                    omitted_calls.append((unscoped_name, node.args))
            for child in self.iter_ast_children(node):
                visit(child)

        visit(ast)
        for function_name, call_arguments in omitted_calls:
            function = self.resolve_value_template_function(
                function_name, call_arguments
            )
            if function is None or id(function) in self.default_value_template_bindings:
                continue
            bindings = self.bind_value_template_arguments(
                function,
                [],
                selected_call=f"{function_name}(...)",
                require_defaults=False,
            )
            if bindings is None:
                continue
            self.default_value_template_bindings[id(function)] = bindings
            self.suppressed_value_template_function_ids.add(id(function))
            if all(
                kind == "value"
                for kind, name in getattr(function, "template_parameters", None) or []
                if name
            ):
                key = self.concrete_template_specialization_key(
                    function,
                    bindings,
                    {},
                )
                self.value_template_specializations[key] = function.name

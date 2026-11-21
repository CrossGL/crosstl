"""Evaluation of constexpr function calls, static initializer dependencies and concrete sizeof replacement."""

import re
from dataclasses import replace
from typing import (
    Dict,
    FrozenSet,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
)

from ..function_specialization import deducible_template_parameters
from ..type_layout import (
    metal_type_layout,
    metal_type_size,
)
from .support import (
    _CONSTEXPR_CALL_SCAN_STOPS,
    IDENTIFIER_RE,
    _MetalConstexprCall,
    _MetalConstexprFunction,
    _MetalDataMember,
    _MetalStructDefinition,
    _MetalTemplateStruct,
    _MetalTypeAliasBinding,
)


class MetalStaticConstexprCallMixin:
    """Members of ``MetalPreprocessor``: evaluation of constexpr function calls, static initializer dependencies and concrete sizeof replacement."""

    def _static_constexpr_function_index(
        self, code: str
    ) -> Dict[str, List[_MetalConstexprFunction]]:
        # The index depends only on the source text; callers filter or read the
        # records but never mutate them, so share them and copy the containers.
        analysis = self._source_analysis(code)
        if analysis.static_constexpr_functions is None:
            analysis.static_constexpr_functions = (
                self._scan_static_constexpr_function_index(code)
            )
        return {
            name: list(candidates)
            for name, candidates in analysis.static_constexpr_functions.items()
        }

    def _scan_static_constexpr_function_index(
        self, code: str
    ) -> Dict[str, List[_MetalConstexprFunction]]:
        functions: Dict[str, List[_MetalConstexprFunction]] = {}
        namespace_visibility = self._metal_namespace_visibility(code)
        namespace_spans = list(namespace_visibility.namespace_spans)
        template_functions = self._find_template_functions(code)
        for function in template_functions:
            body_start = self._find_next_top_level_char(function.source, 0, "{")
            if body_start is None:
                continue
            header = function.source[:body_start]
            if re.search(r"\b(?:constexpr|consteval)\b", header) is None:
                continue
            functions.setdefault(function.name, []).append(
                _MetalConstexprFunction(
                    name=function.name,
                    span=function.span,
                    source=function.source,
                    namespace=function.namespace,
                    is_template=True,
                    template_parameters=list(function.template_parameters),
                    variadic_template_parameters=set(
                        function.variadic_template_parameters
                    ),
                    template_parameter_defaults=dict(
                        function.template_parameter_defaults
                    ),
                    template_parameter_types=dict(function.template_parameter_types),
                    namespace_visibility=namespace_visibility,
                    type_template_parameters=self._declared_type_template_parameters(
                        code,
                        function.span[0],
                    ),
                )
            )

        template_spans = self._find_template_declaration_spans(code)
        # Helper materialization appends concrete specializations after their
        # callers, so assertion evaluation must index those generated definitions.
        for definition in self._find_non_template_function_definitions(
            code,
            template_spans,
        ):
            source = code[definition.span[0] : definition.span[1]]
            body_start = self._find_next_top_level_char(source, 0, "{")
            if body_start is None:
                continue
            header = source[:body_start]
            if re.search(r"\b(?:constexpr|consteval)\b", header) is None:
                continue
            functions.setdefault(definition.name, []).append(
                _MetalConstexprFunction(
                    name=definition.name,
                    span=definition.span,
                    source=source,
                    namespace=self._namespace_at(
                        namespace_spans,
                        definition.span[0],
                    ),
                    namespace_visibility=namespace_visibility,
                )
            )
        return functions

    def _materialization_constexpr_function_index(
        self,
        code: str,
    ) -> Dict[str, List[_MetalConstexprFunction]]:
        functions = self._static_constexpr_function_index(code)
        return {
            name: [
                function
                for function in candidates
                if function.is_template
                or function.name in self._materialized_function_names
            ]
            for name, candidates in functions.items()
            if any(
                function.is_template
                or function.name in self._materialized_function_names
                for function in candidates
            )
        }

    def _resolve_static_constexpr_calls(
        self,
        expression: str,
        functions: Dict[str, List[_MetalConstexprFunction]],
        position: int,
    ) -> Tuple[str, str, Tuple[str, ...]]:
        if not functions:
            return expression, expression, ()
        calls = self._find_static_constexpr_calls(expression, set(functions))
        if not calls:
            return expression, expression, ()

        resolved_replacements: List[Tuple[int, int, str]] = []
        evaluation_replacements: List[Tuple[int, int, str]] = []
        unresolved: Set[str] = set()
        for index, call in enumerate(calls):
            value = self._evaluate_static_constexpr_call(
                call,
                functions,
                position,
            )
            if value is not None:
                resolved_replacements.append((*call.span, value))
                evaluation_replacements.append((*call.span, value))
                continue
            unresolved.add(call.name)
            placeholder = f"__crosstl_unresolved_constexpr_{index}_{call.name}"
            while placeholder in expression:
                placeholder += "_"
            evaluation_replacements.append((*call.span, placeholder))

        resolved = self._apply_text_replacements(expression, resolved_replacements)
        evaluation = self._apply_text_replacements(
            expression,
            evaluation_replacements,
        )
        return resolved, evaluation, tuple(sorted(unresolved))

    def _fold_proven_explicit_static_constexpr_calls(self, code: str) -> str:
        """Fold proven ``helper<...>(...)`` calls outside template declarations.

        Concrete struct/member materialization can copy a constexpr helper call
        out of its original template after the initial explicit-call pass.  The
        copied call is safe to replace only when the constexpr evaluator selects
        one lexically visible candidate and proves an integral result.  Calls
        that are overloaded, runtime-dependent, recursive, or otherwise
        unsupported remain untouched for the normal residual-template diagnostic.

        Inside a concrete struct body, a call without a template argument list
        folds too when every candidate is a template none of whose parameters
        the call could deduce: such a call names the specialization formed from
        the default template arguments ([temp.arg.explicit]/4).  Member
        lowering leaves static data member initializers and constructor member
        initializers there, where no later helper pass binds the call, so once
        the template declaration is removed the call would name nothing.  Calls
        in function bodies stay for the signature-aware helper materializer.
        """

        functions = self._materialization_constexpr_function_index(code)
        if not functions:
            return code
        excluded_spans = self._find_template_declaration_spans(code)
        replacements: List[Tuple[int, int, str]] = []
        struct_spans: Optional[List[Tuple[int, int]]] = None
        for call in self._find_static_constexpr_calls(code, set(functions)):
            if call.template_arguments is None:
                if not all(
                    function.is_template
                    and not deducible_template_parameters(self, function)
                    for function in functions.get(call.name, ())
                ):
                    continue
                if struct_spans is None:
                    struct_spans = [
                        struct.span
                        for struct in self._find_concrete_struct_definitions(code)
                    ]
                if self._containing_span(call.span[0], struct_spans) is None:
                    continue
            if self._containing_span(call.span[0], excluded_spans) is not None:
                continue
            value = self._evaluate_static_constexpr_call(
                call,
                functions,
                call.span[0],
            )
            if value is not None:
                replacements.append((*call.span, value))
        return self._apply_text_replacements(code, replacements)

    def _find_static_constexpr_calls(
        self,
        expression: str,
        function_names: Set[str],
    ) -> List[_MetalConstexprCall]:
        calls: List[_MetalConstexprCall] = []
        if not any(name in expression for name in function_names):
            return calls
        search = _CONSTEXPR_CALL_SCAN_STOPS.search
        i = 0
        while True:
            # Jump to the next character the scan acts on: a quote, a slash
            # that may open a comment, or a possible identifier start.
            stop = search(expression, i)
            if stop is None:
                break
            i = stop.start()
            character = expression[i]
            if character in "\"'":
                _literal, consumed = self._read_string(expression, i)
                i += consumed
                continue
            if character == "/":
                if expression.startswith("//", i):
                    end = expression.find("\n", i)
                    i = len(expression) if end == -1 else end + 1
                elif expression.startswith("/*", i):
                    end = expression.find("*/", i + 2)
                    i = len(expression) if end == -1 else end + 2
                else:
                    i += 1
                continue
            if not (character.isalpha() or character == "_"):
                i += 1
                continue

            name, consumed = self._read_identifier(expression, i)
            if name not in function_names:
                i += consumed
                continue
            name_end = i + consumed
            cursor = name_end
            while cursor < len(expression) and expression[cursor].isspace():
                cursor += 1
            template_arguments: Optional[List[str]] = None
            if cursor < len(expression) and expression[cursor] == "<":
                angle_end = self._find_matching_angle(expression, cursor)
                if angle_end is None:
                    i += consumed
                    continue
                template_arguments = self._split_top_level_commas(
                    expression[cursor + 1 : angle_end]
                )
                cursor = angle_end + 1
                while cursor < len(expression) and expression[cursor].isspace():
                    cursor += 1
            if cursor >= len(expression) or expression[cursor] != "(":
                i += consumed
                continue
            arguments_end = self._find_matching_delimiter(
                expression,
                cursor,
                "(",
                ")",
            )
            if arguments_end is None:
                i += consumed
                continue
            argument_text = expression[cursor + 1 : arguments_end]
            arguments = (
                self._split_top_level_commas(argument_text)
                if argument_text.strip()
                else []
            )
            call_start = self._scoped_identifier_start(expression, i)
            qualified_name = re.sub(
                r"\s*::\s*",
                "::",
                expression[call_start:name_end].strip(),
            )
            calls.append(
                _MetalConstexprCall(
                    name=name,
                    qualified_name=qualified_name,
                    template_arguments=template_arguments,
                    arguments=arguments,
                    span=(call_start, arguments_end + 1),
                    argument_open=cursor,
                )
            )
            i = arguments_end + 1
        return calls

    def _evaluate_static_constexpr_call(
        self,
        call: _MetalConstexprCall,
        functions: Dict[str, List[_MetalConstexprFunction]],
        position: int,
    ) -> Optional[str]:
        candidates = [
            function
            for function in functions.get(call.name, [])
            if (not function.is_template or function.span[0] <= position)
            and self._static_constexpr_namespace_matches(function, call, position)
            and self._static_constexpr_candidate_accepts_call(function, call)
        ]
        if len(candidates) != 1:
            return None
        function = candidates[0]
        template_bindings = self._bind_static_constexpr_template_arguments(
            function,
            list(call.template_arguments or []),
            functions,
            position,
        )
        if template_bindings is None:
            return None
        type_parameters = self._static_constexpr_type_template_parameters(function)
        if type_parameters:
            # Body, return and parameter types name the type parameters; spell
            # the bound concrete types so the integral evaluator sees them.
            function = replace(
                function,
                source=self._replace_identifiers(
                    function.source,
                    {name: template_bindings[name] for name in type_parameters},
                ),
            )
        value_bindings = {
            name: value
            for name, value in template_bindings.items()
            if name not in type_parameters
        }
        runtime_bindings = self._bind_static_constexpr_call_arguments(
            function,
            call.arguments,
            value_bindings,
            functions,
            position,
        )
        if runtime_bindings is None:
            return None

        cache_key = (
            function.span,
            tuple(template_bindings[name] for name in function.template_parameters),
            tuple(runtime_bindings.values()),
        )
        if cache_key in self._static_constexpr_helper_values:
            return self._static_constexpr_helper_values[cache_key]
        if cache_key in self._static_constexpr_helper_resolution_stack:
            return None

        self._static_constexpr_helper_resolution_stack.append(cache_key)
        try:
            bindings = dict(value_bindings)
            bindings.update(runtime_bindings)
            value = self._evaluate_static_constexpr_function_body(
                function,
                bindings,
                functions,
            )
        finally:
            self._static_constexpr_helper_resolution_stack.pop()
        self._static_constexpr_helper_values[cache_key] = value
        return value

    def _static_constexpr_namespace_matches(
        self,
        function: _MetalConstexprFunction,
        call: _MetalConstexprCall,
        position: int,
    ) -> bool:
        return self._metal_namespace_declaration_visible(
            function.namespace,
            self._metal_call_reference_namespace(call.qualified_name),
            position,
            function.namespace_visibility,
        )

    def _static_constexpr_candidate_accepts_call(
        self,
        function: _MetalConstexprFunction,
        call: _MetalConstexprCall,
    ) -> bool:
        if function.variadic_template_parameters:
            return False
        template_arguments = list(call.template_arguments or [])
        if len(template_arguments) > len(function.template_parameters):
            return False
        type_parameters = self._static_constexpr_type_template_parameters(function)
        if any(
            name not in type_parameters
            and (
                name not in function.template_parameter_types
                or self._static_constexpr_integral_type(
                    function.template_parameter_types[name]
                )
                is None
            )
            for name in function.template_parameters
        ):
            return False
        # A type parameter is viable only for a concrete builtin type argument
        # and an integral parameter never for a type, so a helper overloaded on
        # `template <typename T>` and `template <int N>` keeps one candidate.
        for name, argument in zip(function.template_parameters, template_arguments):
            is_type = (
                metal_type_size(self._normalize_template_argument_text(argument))
                is not None
            )
            if is_type != (name in type_parameters):
                return False
        if not self._template_arguments_satisfy_parameters(
            function,
            template_arguments,
        ):
            return False
        parameters = self._static_constexpr_function_parameters(function)
        if parameters is None:
            return False
        required_parameters = sum(
            default is None for _name, _type, default in parameters
        )
        return required_parameters <= len(call.arguments) <= len(parameters)

    def _bind_static_constexpr_template_arguments(
        self,
        function: _MetalConstexprFunction,
        arguments: List[str],
        functions: Dict[str, List[_MetalConstexprFunction]],
        position: int,
    ) -> Optional[Dict[str, str]]:
        bindings: Dict[str, str] = {}
        defaults = function.template_parameter_defaults
        type_parameters = self._static_constexpr_type_template_parameters(function)
        type_bindings: Dict[str, str] = {}
        value_bindings: Dict[str, str] = {}
        for index, name in enumerate(function.template_parameters):
            expression = (
                arguments[index] if index < len(arguments) else defaults.get(name)
            )
            if expression is None:
                return None
            if type_bindings:
                expression = self._replace_identifiers(expression, type_bindings)
            if name in type_parameters:
                type_text = self._normalize_template_argument_text(expression)
                if metal_type_size(type_text) is None:
                    return None
                bindings[name] = type_bindings[name] = type_text
                continue
            expression = self._substitute_template_argument_static_constants(
                expression,
                value_bindings,
            )
            value_type = self._static_constexpr_integral_type(
                function.template_parameter_types.get(name, "")
            )
            value = self._evaluate_static_constexpr_integral_value(
                expression,
                value_type,
                functions,
                position,
            )
            if value is None:
                return None
            bindings[name] = value_bindings[name] = value
        return bindings

    def _static_constexpr_type_template_parameters(
        self,
        function: _MetalConstexprFunction,
    ) -> FrozenSet[str]:
        return (
            function.type_template_parameters if function.is_template else frozenset()
        )

    def _declared_type_template_parameters(
        self,
        code: str,
        template_start: int,
    ) -> FrozenSet[str]:
        """Names declared ``typename T`` or ``class T`` by the template header.

        Packs, template template parameters and non-type parameters are
        excluded, so only a plain type parameter binds one concrete type.
        """

        header = re.compile(r"template\s*<").match(code, template_start)
        if header is None:
            return frozenset()
        close = self._find_matching_angle(code, header.end() - 1)
        if close is None:
            return frozenset()
        return frozenset(
            parameter.name
            for parameter in self._parse_template_parameter_list(
                code[header.end() : close]
            )
            if parameter.is_type_parameter
            and not parameter.is_variadic
            and parameter.name is not None
        )

    def _bind_static_constexpr_call_arguments(
        self,
        function: _MetalConstexprFunction,
        arguments: List[str],
        template_bindings: Dict[str, str],
        functions: Dict[str, List[_MetalConstexprFunction]],
        position: int,
    ) -> Optional[Dict[str, str]]:
        parameters = self._static_constexpr_function_parameters(function)
        if parameters is None:
            return None
        bindings: Dict[str, str] = {}
        visible = dict(template_bindings)
        for index, (name, value_type, default) in enumerate(parameters):
            expression = arguments[index] if index < len(arguments) else default
            if expression is None:
                return None
            expression = self._substitute_template_argument_static_constants(
                expression,
                visible,
            )
            value = self._evaluate_static_constexpr_integral_value(
                expression,
                value_type,
                functions,
                position,
            )
            if value is None:
                return None
            bindings[name] = value
            visible[name] = value
        return bindings

    def _static_constexpr_function_parameters(
        self,
        function: _MetalConstexprFunction,
    ) -> Optional[List[Tuple[str, str, Optional[str]]]]:
        body_start = self._find_next_top_level_char(function.source, 0, "{")
        if body_start is None:
            return None
        header = function.source[:body_start]
        open_paren = self._function_parameter_start(header)
        if open_paren is None:
            return None
        close_paren = self._find_matching_delimiter(header, open_paren, "(", ")")
        if close_paren is None:
            return None
        parameter_text = header[open_paren + 1 : close_paren]
        if not parameter_text.strip() or parameter_text.strip() == "void":
            return []

        parameters: List[Tuple[str, str, Optional[str]]] = []
        for parameter in self._split_top_level_commas(parameter_text):
            declaration, default = self._split_top_level_assignment(parameter)
            name = self._declared_local_name(declaration)
            if name is None:
                return None
            value_type = self._static_constexpr_integral_type(
                self._declared_local_type(declaration, name) or ""
            )
            if value_type is None:
                return None
            parameters.append((name, value_type, default))
        return parameters

    def _evaluate_static_constexpr_function_body(
        self,
        function: _MetalConstexprFunction,
        bindings: Dict[str, str],
        functions: Dict[str, List[_MetalConstexprFunction]],
    ) -> Optional[str]:
        body_start = self._find_next_top_level_char(function.source, 0, "{")
        if body_start is None:
            return None
        body_end = self._find_matching_brace(function.source, body_start)
        if body_end is None:
            return None
        statements = self._split_static_constexpr_body_statements(
            function.source[body_start + 1 : body_end - 1]
        )
        if not statements:
            return None
        return_type = self._static_constexpr_function_return_type(function)
        if return_type is None:
            return None

        active_bindings = dict(bindings)
        for index, statement in enumerate(statements):
            stripped = self._strip_template_argument_comments(statement).strip()
            return_match = re.fullmatch(r"return\s+(.+)", stripped, re.DOTALL)
            if return_match is not None:
                if index != len(statements) - 1:
                    return None
                expression = self._substitute_template_argument_static_constants(
                    return_match.group(1),
                    active_bindings,
                )
                return self._evaluate_static_constexpr_integral_value(
                    expression,
                    return_type,
                    functions,
                    function.span[0],
                )

            name = self._declared_local_name(stripped)
            member = (
                self._parse_ordered_data_member(name, stripped)
                if name is not None
                else None
            )
            if member is None or member.default is None:
                return None
            qualifiers = set(IDENTIFIER_RE.findall(member.type_text))
            value_type = self._static_constexpr_integral_type(member.type_text)
            if (
                "constexpr" not in qualifiers
                or value_type is None
                or member.array_suffix
            ):
                return None
            initializer_bindings = dict(active_bindings)
            initializer_bindings.pop(name, None)
            initializer = self._substitute_template_argument_static_constants(
                member.default,
                initializer_bindings,
            )
            value = self._evaluate_static_constexpr_integral_value(
                initializer,
                value_type,
                functions,
                function.span[0],
            )
            if value is None:
                return None
            active_bindings[name] = value
        return None

    def _split_static_constexpr_body_statements(
        self,
        body: str,
    ) -> Optional[List[str]]:
        statements: List[str] = []
        start = 0
        paren_depth = 0
        bracket_depth = 0
        i = 0
        while i < len(body):
            if body[i] in "\"'":
                _literal, consumed = self._read_string(body, i)
                i += consumed
                continue
            if body.startswith("//", i):
                end = body.find("\n", i)
                i = len(body) if end == -1 else end + 1
                continue
            if body.startswith("/*", i):
                end = body.find("*/", i + 2)
                if end == -1:
                    return None
                i = end + 2
                continue
            token = body[i]
            if token == "(":
                paren_depth += 1
            elif token == ")":
                paren_depth = max(0, paren_depth - 1)
            elif token == "[":
                bracket_depth += 1
            elif token == "]":
                bracket_depth = max(0, bracket_depth - 1)
            elif token in "{}":
                return None
            elif token == ";" and paren_depth == bracket_depth == 0:
                statement = body[start:i].strip()
                if statement:
                    statements.append(statement)
                start = i + 1
            i += 1
        if self._strip_template_argument_comments(body[start:]).strip():
            return None
        return statements

    def _static_constexpr_function_return_type(
        self,
        function: _MetalConstexprFunction,
    ) -> Optional[str]:
        body_start = self._find_next_top_level_char(function.source, 0, "{")
        if body_start is None:
            return None
        header = function.source[:body_start]
        open_paren = self._function_parameter_start(header)
        if open_paren is None:
            return None
        name_match = re.search(
            rf"\b{re.escape(function.name)}\s*$",
            header[:open_paren],
        )
        if name_match is None:
            return None
        return self._static_constexpr_integral_type(header[: name_match.start()])

    def _evaluate_static_constexpr_integral_value(
        self,
        expression: str,
        value_type: Optional[str],
        functions: Dict[str, List[_MetalConstexprFunction]],
        position: int,
    ) -> Optional[str]:
        if value_type is None:
            return None
        _resolved, evaluation, _unresolved = self._resolve_static_constexpr_calls(
            expression,
            functions,
            position,
        )
        return self._proven_integral_constant_value(value_type, evaluation)

    def _static_constexpr_integral_type(self, type_text: str) -> Optional[str]:
        if any(token in type_text for token in ("*", "&", "[", "]")):
            return None
        normalized = self._normalize_template_argument_text(type_text)
        normalized = re.sub(
            r"\b(?:const|constant|constexpr|consteval|device|inline|static|thread|threadgroup)\b",
            " ",
            normalized,
        )
        normalized = re.sub(r"\s+", " ", normalized).strip()
        normalized = re.sub(r"^(?:::)?metal::", "", normalized)
        return normalized if normalized in self._METAL_INTEGRAL_SCALAR_TYPES else None

    def _substitute_struct_scoped_alias_owners(
        self,
        source: str,
        struct: _MetalStructDefinition,
    ) -> str:
        if not struct.type_aliases:
            return source
        ignored_spans = self._find_comment_and_literal_spans(source)
        replacements: List[Tuple[int, int, str]] = []
        for alias in sorted(struct.type_aliases, key=len, reverse=True):
            target = self._canonicalize_struct_scoped_type(alias, struct, None)
            if not target or target == alias:
                continue
            for match in re.finditer(
                rf"\b{re.escape(alias)}\b(?=\s*::)",
                source,
            ):
                if self._containing_span(match.start(), ignored_spans) is None:
                    replacements.append((match.start(), match.end(), target))
        return self._apply_text_replacements(source, replacements)

    _STATIC_INITIALIZER_MAX_DEPENDENCY_DEPTH = 128

    def _static_initializer_dependencies(
        self, initializer: str, member_names: List[str]
    ) -> List[str]:
        # Reuse the body substitution scanner to identify only BARE dependency
        # references. Names in comments, strings, calls, or member accesses do
        # not form initializer-map edges.
        dependencies: List[str] = []
        for name in member_names:
            probe = self._substitute_bare_member_references(
                initializer, {name: f"({name})"}
            )
            if probe != initializer:
                dependencies.append(name)
        return dependencies

    def _fold_static_integral_initializer(
        self, member: _MetalDataMember, initializer: str
    ) -> str:
        # Fold only a small, integer-only expression language. Calls, casts,
        # names, floating-point values, and unsupported operators leave the
        # original initializer untouched and are still substituted verbatim.
        type_tokens = IDENTIFIER_RE.findall(member.type_text)
        base_type = next(
            (
                token
                for token in reversed(type_tokens)
                if token in self._METAL_INTEGRAL_SCALAR_TYPES
            ),
            None,
        )
        if base_type is None:
            return initializer
        value = self._proven_integral_constant_value(base_type, initializer)
        return initializer if value is None else value

    def _proven_integral_constant_value(
        self, base_type: str, initializer: str
    ) -> Optional[str]:
        if base_type not in self._METAL_INTEGRAL_SCALAR_TYPES:
            return None
        folded, value = self._evaluate_static_integral_expression(initializer)
        if not folded or value is None:
            return None
        if base_type == "bool":
            return "true" if bool(value) else "false"

        bits = self._METAL_SCALAR_TYPE_SIZES[base_type] * 8
        if base_type in self._METAL_SIGNED_SCALAR_TYPES:
            minimum = -(1 << (bits - 1))
            maximum = (1 << (bits - 1)) - 1
        else:
            minimum = 0
            maximum = (1 << bits) - 1
        integer_value = int(value)
        if not minimum <= integer_value <= maximum:
            return None
        return str(integer_value)

    _STATIC_FOLD_MIN_INT = -(1 << 31)
    _STATIC_FOLD_MAX_INT = (1 << 31) - 1
    _MAX_CONCRETE_AGGREGATE_SIZE = (1 << 63) - 1

    def _exact_template_type_id(
        self,
        type_text: str,
    ) -> Optional[Tuple[str, List[str]]]:
        text = self._normalize_template_argument_text(type_text)
        if not text or len(text) > 512:
            return None
        match = re.match(
            r"(?P<name>(?:::)?[A-Za-z_]\w*(?:::[A-Za-z_]\w*)*)<",
            text,
        )
        if match is None:
            return None
        angle_start = match.end() - 1
        angle_end = self._find_matching_angle(text, angle_start)
        if angle_end is None or angle_end != len(text) - 1:
            return None
        argument_text = text[angle_start + 1 : angle_end]
        arguments = (
            self._split_top_level_commas(argument_text) if argument_text.strip() else []
        )
        return match.group("name"), arguments

    def _concrete_template_struct_layout(
        self,
        *,
        template_name: str,
        template_arguments: List[str],
        structs: List[_MetalStructDefinition],
        template_structs: Sequence[_MetalTemplateStruct],
        type_aliases: Dict[str, List[_MetalTypeAliasBinding]],
        position: int,
        resolving: Set[str],
        explicit_template_specializations: Set[Tuple[str, Tuple[str, ...]]],
    ) -> Optional[Tuple[int, int]]:
        requested_name = template_name.lstrip(":")
        candidates = [
            template
            for template in template_structs
            if requested_name
            in {
                template.name,
                (
                    f"{template.namespace}::{template.name}"
                    if template.namespace
                    else template.name
                ).lstrip(":"),
            }
        ]
        # Multiple primary/partial declarations, or selecting a partial
        # specialization directly, requires C++ specialization ordering that
        # this layout-only proof deliberately does not approximate.
        if len(candidates) != 1:
            return None
        template = candidates[0]
        if self._template_struct_specialization_arguments(template) is not None:
            return None
        if template.variadic_template_parameters:
            return None
        if any(
            specialization_name == template.name
            for specialization_name, _arguments in explicit_template_specializations
        ):
            return None
        if len(template_arguments) > len(template.template_parameters):
            return None

        concrete_arguments = self._template_arguments_with_resolved_defaults(
            template,
            template_arguments,
        )
        if concrete_arguments is None or len(concrete_arguments) != len(
            template.template_parameters
        ):
            return None
        substitutions = dict(zip(template.template_parameters, concrete_arguments))
        for parameter, declared_type in template.template_parameter_types.items():
            resolved_declared_type = self._replace_identifiers(
                declared_type,
                substitutions,
            )
            integral_type = self._static_constexpr_integral_type(resolved_declared_type)
            if integral_type is None:
                return None
            value = self._proven_integral_constant_value(
                integral_type,
                self._replace_identifiers(substitutions[parameter], substitutions),
            )
            if value is None:
                return None
            substitutions[parameter] = value

        concrete_arguments = [
            self._normalize_template_argument_text(substitutions[parameter])
            for parameter in template.template_parameters
        ]
        if any(
            not argument
            or self._type_references_template_parameter(
                argument,
                template.template_parameters,
            )
            for argument in concrete_arguments
        ):
            return None

        qualified_name = (
            f"{template.namespace}::{template.name}"
            if template.namespace
            else template.name
        )
        specialization_key = self._template_specialization_signature(
            qualified_name,
            concrete_arguments,
        )
        if specialization_key in resolving:
            return None
        active = set(resolving)
        active.add(specialization_key)

        probe_name = self._template_specialization_identifier(
            f"__crosstl_layout_{qualified_name}",
            concrete_arguments,
        )
        materialized = self._materialize_template_struct_with_name(
            template,
            concrete_arguments,
            probe_name,
        )
        if not materialized:
            return None
        materialized_structs = self._find_concrete_struct_definitions(materialized)
        if len(materialized_structs) != 1:
            return None
        probe = materialized_structs[0]
        if probe.name != probe_name:
            return None
        # The synthetic source starts at offset zero, but aliases in its member
        # declarations obey the original template declaration's lexical scope.
        probe.span = template.span
        return self._concrete_aggregate_type_layout(
            probe_name,
            [*structs, probe],
            type_aliases,
            template.span[0],
            active,
            template_structs=template_structs,
            explicit_template_specializations=explicit_template_specializations,
        )

    def _concrete_aggregate_type_layout(
        self,
        type_text: str,
        structs: List[_MetalStructDefinition],
        type_aliases: Dict[str, List[_MetalTypeAliasBinding]],
        position: int,
        resolving: Optional[Set[str]] = None,
        *,
        template_structs: Optional[Sequence[_MetalTemplateStruct]] = None,
        explicit_template_specializations: Optional[
            Set[Tuple[str, Tuple[str, ...]]]
        ] = None,
    ) -> Optional[Tuple[int, int]]:
        resolved_type = self._resolve_type_aliases_at(
            type_text,
            type_aliases,
            position,
        )
        layout = metal_type_layout(resolved_type)
        if layout is not None:
            return layout

        normalized = self._normalize_inferred_type(resolved_type)
        normalized = re.sub(r"^(?:struct|class|union)\s+", "", normalized)
        template_id = self._exact_template_type_id(normalized)
        if template_id is not None:
            if not template_structs:
                return None
            template_name, template_arguments = template_id
            return self._concrete_template_struct_layout(
                template_name=template_name,
                template_arguments=template_arguments,
                structs=structs,
                template_structs=template_structs,
                type_aliases=type_aliases,
                position=position,
                resolving=set(resolving or ()),
                explicit_template_specializations=set(
                    explicit_template_specializations or ()
                ),
            )

        normalized = normalized.lstrip(":")
        if (
            re.fullmatch(
                r"[A-Za-z_]\w*(?:::[A-Za-z_]\w*)*",
                normalized,
            )
            is None
        ):
            return None

        candidates = [
            struct
            for struct in structs
            if normalized
            in {
                struct.name,
                struct.qualified_name.lstrip(":"),
            }
        ]
        if len(candidates) != 1:
            return None
        struct = candidates[0]
        if not struct.layout_supported or struct.base_clause:
            return None

        aggregate_name = struct.qualified_name or struct.name
        active = set(resolving or ())
        if aggregate_name in active:
            return None
        active.add(aggregate_name)

        name_counts: Dict[str, int] = {}
        for definition in structs:
            name_counts[definition.name] = name_counts.get(definition.name, 0) + 1
        structs_by_name = {
            definition.name: definition
            for definition in structs
            if name_counts[definition.name] == 1
        }

        offset = 0
        aggregate_alignment = 1
        for member in struct.data_members:
            qualifiers = set(IDENTIFIER_RE.findall(member.type_text))
            if "static" in qualifiers:
                continue
            if "*" in member.type_text or "&" in member.type_text:
                return None

            member_type = self._canonicalize_struct_scoped_type(
                member.type_text,
                struct,
                structs_by_name,
            )
            member_layout = self._concrete_aggregate_type_layout(
                member_type,
                structs,
                type_aliases,
                struct.span[0],
                active,
                template_structs=template_structs,
                explicit_template_specializations=explicit_template_specializations,
            )
            if member_layout is None:
                return None
            member_size, member_alignment = member_layout
            array_extents = self._concrete_array_extents(member.array_suffix)
            if array_extents is None:
                return None
            for extent in array_extents:
                stride = self._align_concrete_aggregate_offset(
                    member_size,
                    member_alignment,
                )
                if stride > self._MAX_CONCRETE_AGGREGATE_SIZE // extent:
                    return None
                member_size = stride * extent

            if struct.aggregate_kind == "union":
                offset = max(offset, member_size)
            else:
                offset = self._align_concrete_aggregate_offset(
                    offset,
                    member_alignment,
                )
                offset += member_size
            if offset > self._MAX_CONCRETE_AGGREGATE_SIZE:
                return None
            aggregate_alignment = max(aggregate_alignment, member_alignment)

        aggregate_size = self._align_concrete_aggregate_offset(
            offset,
            aggregate_alignment,
        )
        if aggregate_size == 0:
            aggregate_size = 1
        if aggregate_size > self._MAX_CONCRETE_AGGREGATE_SIZE:
            return None
        return aggregate_size, aggregate_alignment

    def _concrete_array_extents(self, suffix: str) -> Optional[List[int]]:
        if not suffix:
            return []
        extents: List[int] = []
        cursor = 0
        for match in re.finditer(r"\[\s*([^\[\]]+?)\s*\]", suffix):
            if suffix[cursor : match.start()].strip():
                return None
            folded, value = self._evaluate_static_integral_expression(match.group(1))
            if not folded or value is None or value <= 0:
                return None
            extents.append(int(value))
            cursor = match.end()
        if not extents or suffix[cursor:].strip():
            return None
        return extents

    @staticmethod
    def _align_concrete_aggregate_offset(offset: int, alignment: int) -> int:
        return ((offset + alignment - 1) // alignment) * alignment

    def _enter_sizeof_source(self, code: str) -> str:
        """Make ``code`` the active sizeof source if it defines aggregates.

        Returns the previous source for the caller to restore. A fragment
        without aggregate definitions keeps the enclosing source.
        """
        previous = self._active_sizeof_source
        if "sizeof" in code and re.search(r"\b(?:struct|class|union)\b", code):
            self._active_sizeof_source = code
        return previous

    def _active_sizeof_layout(self, type_text: str) -> Optional[Tuple[int, int]]:
        # [expr.sizeof]/2: a class operand yields its object size. Only names
        # visible at file scope of the active source are resolved, and the
        # lookup fails closed on ambiguous or unsupported layouts.
        source = self._active_sizeof_source
        if not source:
            return None
        context = self._sizeof_context_cache.get(source)
        if context is None:
            template_spans = self._find_template_declaration_spans(source)
            context = (
                self._find_concrete_struct_definitions(source),
                self._collect_local_type_alias_bindings(
                    source,
                    [(0, len(source))],
                    skip_spans=template_spans,
                ),
                self._find_template_structs(source),
            )
            self._sizeof_context_cache = {source: context}
        structs, type_aliases, template_structs = context
        # The last offset lies in the file scope, after every declaration.
        return self._concrete_aggregate_type_layout(
            type_text,
            list(structs),
            type_aliases,
            len(source) - 1,
            template_structs=template_structs,
        )

    def _replace_concrete_sizeof_expressions(
        self,
        expression: str,
        type_resolver=None,
        layout_resolver=None,
    ) -> str:
        resolver = type_resolver or (lambda type_text: type_text)

        def replace(match):
            operand = self._normalize_template_argument_text(match.group("operand"))
            resolved_operand = resolver(operand)
            layout = (
                layout_resolver(resolved_operand)
                if layout_resolver is not None
                else metal_type_layout(resolved_operand)
            )
            if layout is None and layout_resolver is None:
                layout = self._active_sizeof_layout(resolved_operand)
            size = layout[0] if layout is not None else None
            return str(size) if size is not None else match.group(0)

        return re.sub(
            r"\bsizeof\s*\(\s*(?P<operand>[^()]+?)\s*\)",
            replace,
            expression,
        )

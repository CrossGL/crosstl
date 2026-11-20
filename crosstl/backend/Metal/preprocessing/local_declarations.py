"""Local integral constants, data members and type aliases, struct-scoped template arguments and struct body parsing."""

import re
from dataclasses import replace
from heapq import (
    heappop,
    heappush,
)
from itertools import count as sequence_numbers
from typing import (
    Callable,
    Dict,
    Iterator,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
)

from ..auto_integral_constants import deduced_integral_auto_type
from ..conditional_types import (
    select_conditional_type,
    select_nested_conditional_types,
)
from ..integral_binding_visibility import (
    IntegralBindingTable,
    indexed_visible_bindings,
)
from ..local_decltypes import (
    complete_declared_type,
    substitute_local_decltypes_in,
)
from ..member_declarator_lists import split_member_declarator_list
from ..reopened_namespaces import ReopenedNamespaceIndex
from ..signed_type_spellings import fold_integer_type_spellings
from ..text_scanners import split_top_level_assignment
from .support import (
    _UNPARSED,
    IDENTIFIER_RE,
    LOCAL_DATA_MEMBER_MEMO_LIMIT,
    SOURCE_ANALYSIS_CACHE_LIMIT,
    _MetalConstexprFunction,
    _MetalConstructor,
    _MetalDataMember,
    _MetalIntegralConstantBinding,
    _MetalStructDefinition,
    _MetalStructMethod,
    _MetalTypeAliasBinding,
)


class MetalLocalDeclarationMixin:
    """Members of ``MetalPreprocessor``: local integral constants, data members and type aliases, struct-scoped template arguments and struct body parsing."""

    def _collect_local_integral_constant_bindings(
        self,
        code: str,
        owner_spans: List[Tuple[int, int]],
        type_aliases: Optional[Dict[str, List[_MetalTypeAliasBinding]]] = None,
        constexpr_functions: Optional[Dict[str, List[_MetalConstexprFunction]]] = None,
        constexpr_position_offset: int = 0,
        include_file_scope: bool = False,
        static_member_constants: Optional[Callable[[str, Dict[str, str]], str]] = None,
    ) -> Dict[str, List[_MetalIntegralConstantBinding]]:
        # Initializers such as `16 / sizeof(S)` fold only when the concrete
        # struct `S` of this source is visible to the sizeof evaluator. A source
        # fragment without struct definitions keeps the enclosing context.
        previous_sizeof_source = self._enter_sizeof_source(code)
        try:
            return self._collect_local_integral_constant_bindings_impl(
                code,
                owner_spans,
                type_aliases,
                constexpr_functions,
                constexpr_position_offset,
                include_file_scope,
                static_member_constants,
            )
        finally:
            self._active_sizeof_source = previous_sizeof_source

    def _collect_local_integral_constant_bindings_impl(
        self,
        code: str,
        owner_spans: List[Tuple[int, int]],
        type_aliases: Optional[Dict[str, List[_MetalTypeAliasBinding]]],
        constexpr_functions: Optional[Dict[str, List[_MetalConstexprFunction]]],
        constexpr_position_offset: int,
        include_file_scope: bool,
        static_member_constants: Optional[Callable[[str, Dict[str, str]], str]],
    ) -> Dict[str, List[_MetalIntegralConstantBinding]]:
        raw_declarations: List[Tuple[int, str, _MetalDataMember, bool]] = []
        parameter_bindings: List[Tuple[int, str, int, int]] = []
        seen_declarations: Set[Tuple[int, str]] = set()
        lexical_scopes = self._find_lexical_brace_scopes(code)
        selected_owner_spans = sorted(owner_spans)
        scan_spans = [(0, len(code))] if include_file_scope else selected_owner_spans
        file_scopes = {(0, len(code))}
        if include_file_scope:
            file_scopes.update(
                (start, end) for start, end, _name in self._find_namespace_spans(code)
            )
        for owner_start, owner_end in scan_spans:
            for (
                declaration_position,
                name,
                member,
            ) in self._local_data_member_declarations(code, owner_start, owner_end):
                file_scope_only = False
                if include_file_scope:
                    declaration_scope = self._innermost_lexical_scope(
                        lexical_scopes,
                        declaration_position,
                        len(code),
                    )
                    selected_owner = self._containing_span(
                        declaration_position,
                        selected_owner_spans,
                    )
                    if selected_owner is None and declaration_scope not in file_scopes:
                        continue
                    file_scope_only = selected_owner is None
                declaration_key = (declaration_position, name)
                if declaration_key in seen_declarations:
                    continue
                seen_declarations.add(declaration_key)
                raw_declarations.append(
                    (declaration_position, name, member, file_scope_only)
                )

        template_spans = self._find_template_declaration_spans(code)
        for function in self._find_non_template_function_definitions(
            code,
            template_spans,
        ):
            if (
                self._containing_span(
                    function.body_span[0],
                    selected_owner_spans,
                )
                is None
            ):
                continue
            header = code[function.span[0] : function.body_span[0] - 1]
            parameter_span = self._function_parameter_list_span(header)
            if parameter_span is None:
                continue
            parameter_start, parameter_end = parameter_span
            for parameter in self._split_top_level_commas(
                header[parameter_start + 1 : parameter_end]
            ):
                declaration = self._strip_metal_attributes(parameter)
                name = self._declared_data_member_name(declaration)
                if name and name != "void":
                    parameter_bindings.append(
                        (
                            function.body_span[0],
                            name,
                            function.body_span[0],
                            function.body_span[1],
                        )
                    )

        if not raw_declarations and not parameter_bindings:
            return {}
        bindings = IntegralBindingTable()
        ordered_parameters = iter(sorted(parameter_bindings))
        pending_parameter = next(ordered_parameters, None)

        def record_parameters_through(position: int) -> None:
            nonlocal pending_parameter
            while pending_parameter is not None and pending_parameter[0] <= position:
                declaration_position, name, scope_start, scope_end = pending_parameter
                bindings.add(
                    name,
                    _MetalIntegralConstantBinding(
                        declaration_position=declaration_position,
                        scope_start=scope_start,
                        scope_end=scope_end,
                        value=None,
                    ),
                )
                pending_parameter = next(ordered_parameters, None)

        ignored_type_tokens = {
            "const",
            "constant",
            "constexpr",
            "device",
            "register",
            "static",
            "thread",
            "threadgroup",
            "volatile",
        }
        reopened_index = ReopenedNamespaceIndex(self._find_namespace_spans(code), code)
        reopened_namespaces = reopened_index.has_reopened_namespace
        integral_types: Dict[int, str] = {}
        # ``(block start, order, name, binding)`` of namespace members recorded for the later blocks of their
        # namespace, each declared where its block opens so it hides enclosing declarations made in between.
        pending_reopened: List[Tuple[int, int, str, _MetalIntegralConstantBinding]] = []
        reopened_order = sequence_numbers()

        def record_pending_through(position: int) -> None:
            while pending_reopened and pending_reopened[0][0] <= position:
                block_start, _order, reopened_name, reopened = heappop(pending_reopened)
                record_parameters_through(block_start)
                bindings.add(reopened_name, reopened)
            record_parameters_through(position)

        for declaration_position, name, member, file_scope_only in sorted(
            raw_declarations
        ):
            record_pending_through(declaration_position)
            scope_start, scope_end = self._innermost_lexical_scope(
                lexical_scopes, declaration_position, len(code)
            )
            value: Optional[str] = None
            type_tokens = IDENTIFIER_RE.findall(member.type_text)
            value_type_tokens = fold_integer_type_spellings(
                [token for token in type_tokens if token not in ignored_type_tokens]
            )
            auto_type = None
            if (
                value_type_tokens == ["auto"]
                and member.default is not None
                and "constexpr" in type_tokens
            ):
                auto_type = deduced_integral_auto_type(
                    member.default,
                    {
                        visible_name: integral_types[id(binding)]
                        for visible_name, binding in (
                            self._local_integral_constant_bindings_at(
                                bindings, declaration_position
                            ).items()
                        )
                        if id(binding) in integral_types
                    },
                )
                if auto_type is not None:
                    value_type_tokens = [auto_type]
            is_constexpr_integral = (
                member.default is not None
                and "constexpr" in type_tokens
                and len(value_type_tokens) == 1
                and value_type_tokens[0] in self._METAL_INTEGRAL_SCALAR_TYPES
                and not member.array_suffix
                and "*" not in member.type_text
                and "&" not in member.type_text
            )
            if file_scope_only and not is_constexpr_integral:
                continue
            if is_constexpr_integral:
                visible_constants = self._local_integral_constants_at(
                    bindings, declaration_position
                )
                # The declared name is already in scope in its own initializer;
                # leaving it unresolved prevents an outer binding from making a
                # self-reference look like a valid constant expression.
                visible_constants.pop(name, None)
                initializer = member.default
                if static_member_constants is not None:
                    initializer = static_member_constants(
                        initializer,
                        self._visible_local_declared_types(
                            self._local_integral_constant_bindings_at(
                                bindings, declaration_position
                            )
                        ),
                    )
                initializer = self._substitute_template_argument_static_constants(
                    initializer,
                    visible_constants,
                )
                if type_aliases is not None:
                    initializer = self._replace_concrete_sizeof_expressions(
                        initializer,
                        lambda type_text: self._resolve_type_aliases_at(
                            type_text,
                            type_aliases,
                            declaration_position,
                        ),
                    )
                else:
                    initializer = self._replace_concrete_sizeof_expressions(initializer)
                initializer = self._fold_initializer_type_traits(
                    initializer, type_aliases, declaration_position
                )
                if constexpr_functions:
                    _resolved_initializer, initializer, _unresolved = (
                        self._resolve_static_constexpr_calls(
                            initializer,
                            constexpr_functions,
                            declaration_position + constexpr_position_offset,
                        )
                    )
                value = self._proven_integral_constant_value(
                    value_type_tokens[0], initializer
                )
            bindings.add(
                name,
                _MetalIntegralConstantBinding(
                    declaration_position=declaration_position,
                    scope_start=scope_start,
                    scope_end=scope_end,
                    value=value,
                    declared_type=(
                        None
                        if file_scope_only
                        else auto_type
                        or complete_declared_type(member.type_text, member.array_suffix)
                    ),
                    type_alias=bool(type_tokens)
                    and type_tokens[0] in {"using", "typedef"},
                ),
            )
            if is_constexpr_integral and value is not None:
                # A proven constant's type, for an ``auto`` that copies it.
                integral_types[id(bindings[name][-1])] = value_type_tokens[0]
            if reopened_namespaces:
                declared = bindings[name][-1]
                for block_start, block_end in reopened_index.unrecorded_later_blocks(
                    (scope_start, scope_end), declaration_position, name
                ):
                    heappush(
                        pending_reopened,
                        (
                            block_start,
                            next(reopened_order),
                            name,
                            replace(
                                declared,
                                declaration_position=block_start,
                                scope_start=block_start,
                                scope_end=block_end,
                            ),
                        ),
                    )
        record_pending_through(len(code))
        return bindings

    def _local_data_member_declarations(
        self, code: str, owner_start: int, owner_end: int
    ) -> Tuple[Tuple[int, str, _MetalDataMember], ...]:
        """Simple data declarations in ``code[owner_start:owner_end]``, in source order.

        Each entry is ``(position, name, member)``. The scan depends only on the
        source text and span, and the integral-constant collector repeats it for
        every materialization pass, so it is cached per source.
        """
        cache = self._source_analysis(code).local_data_member_declarations
        key = (owner_start, owner_end)
        cached = cache.pop(key, None)
        if cached is None:
            cached = tuple(
                self._scan_local_data_member_declarations(code, owner_start, owner_end)
            )
            cache = self._source_analysis(code).local_data_member_declarations
        cache[key] = cached
        if len(cache) > SOURCE_ANALYSIS_CACHE_LIMIT * 8:
            del cache[next(iter(cache))]
        return cached

    def _scan_local_data_member_declarations(
        self, code: str, owner_start: int, owner_end: int
    ) -> Iterator[Tuple[int, str, _MetalDataMember]]:
        owner_text = code[owner_start:owner_end]
        search_start = 0
        for statement in self._iter_simple_declarations(owner_text):
            relative_start = owner_text.find(statement, search_start)
            if relative_start == -1:
                continue
            search_start = relative_start + len(statement)
            declaration_offset = 0
            # Preserved ``#include``/``#pragma`` lines do not end in a
            # semicolon, so the simple-declaration iterator can return them
            # as a prefix of the next real declaration.  Ignore only leading
            # blank/directive lines and retain the exact declaration offset;
            # directives embedded after source tokens remain fail-closed.
            for line in statement.splitlines(keepends=True):
                stripped_line = line.strip()
                if not stripped_line or line.lstrip().startswith("#"):
                    declaration_offset += len(line)
                    continue
                break
            declaration_text = statement[declaration_offset:]
            stripped = declaration_text.strip()
            if not stripped:
                continue
            declaration_position = (
                owner_start
                + relative_start
                + declaration_offset
                + len(declaration_text)
                - len(declaration_text.lstrip())
            )
            # Most statements survive each materialization rewrite unchanged, and
            # their parse depends only on the text, so memoize it by statement.
            memo = self._local_data_member_memo
            parsed = memo.get(stripped, _UNPARSED)
            if parsed is _UNPARSED:
                name = self._declared_local_name(stripped)
                member = (
                    None
                    if name is None
                    else self._parse_ordered_data_member(name, stripped)
                )
                parsed = None if member is None else (name, member)
                if len(memo) >= LOCAL_DATA_MEMBER_MEMO_LIMIT:
                    memo.clear()
                memo[stripped] = parsed
            if parsed is None:
                continue
            yield declaration_position, parsed[0], parsed[1]

    def _substitute_local_integral_constant_array_extents(self, code: str) -> str:
        template_spans = self._find_template_declaration_spans(code)
        owner_spans = [
            function.body_span
            for function in self._find_non_template_function_definitions(
                code,
                template_spans,
            )
            if function.name in self._materialized_function_names
        ]
        if not owner_spans:
            return code

        type_aliases = self._collect_local_type_alias_bindings(code, owner_spans)
        constexpr_functions = self._materialization_constexpr_function_index(code)
        constants = self._collect_local_integral_constant_bindings(
            code,
            owner_spans,
            type_aliases,
            constexpr_functions,
            include_file_scope=True,
        )
        replacements: List[Tuple[int, int, str]] = []
        for owner_start, owner_end in sorted(owner_spans):
            owner_text = code[owner_start:owner_end]
            search_start = 0
            for statement in self._iter_simple_declarations(owner_text):
                relative_start = owner_text.find(statement, search_start)
                if relative_start == -1:
                    continue
                search_start = relative_start + len(statement)
                stripped = statement.strip()
                if not stripped:
                    continue
                name = self._declared_local_name(stripped)
                if name is None:
                    continue
                member = self._parse_ordered_data_member(name, stripped)
                if member is None or not member.array_suffix:
                    continue

                declaration_position = (
                    owner_start
                    + relative_start
                    + len(statement)
                    - len(statement.lstrip())
                )
                visible_constants = self._local_integral_constants_at(
                    constants,
                    declaration_position,
                )
                if not visible_constants:
                    continue
                resolved_suffix = self._substitute_template_argument_static_constants(
                    member.array_suffix,
                    visible_constants,
                )
                concrete_extents = self._concrete_array_extents(resolved_suffix)
                if concrete_extents is None:
                    continue
                concrete_suffix = "".join(f"[{extent}]" for extent in concrete_extents)
                if concrete_suffix == member.array_suffix:
                    continue

                lhs, _default = self._split_top_level_assignment(stripped)
                if not lhs.endswith(member.array_suffix):
                    continue
                suffix_start = (
                    declaration_position + len(lhs) - len(member.array_suffix)
                )
                replacements.append(
                    (
                        suffix_start,
                        suffix_start + len(member.array_suffix),
                        concrete_suffix,
                    )
                )
        return self._apply_text_replacements(code, replacements)

    @staticmethod
    def _local_integral_constant_bindings_at(
        bindings: Dict[str, List[_MetalIntegralConstantBinding]],
        position: int,
        *,
        names: Optional[Set[str]] = None,
    ) -> Dict[str, _MetalIntegralConstantBinding]:
        indexed = indexed_visible_bindings(bindings, position, names)
        if indexed is not None:
            return indexed
        visible: Dict[str, _MetalIntegralConstantBinding] = {}
        binding_items = (
            bindings.items()
            if names is None
            else ((name, bindings[name]) for name in sorted(names) if name in bindings)
        )
        for name, candidates in binding_items:
            best: Optional[_MetalIntegralConstantBinding] = None
            for binding in candidates:
                if binding.declaration_position > position:
                    break
                if not (binding.scope_start <= position < binding.scope_end):
                    continue
                if (
                    best is None
                    or binding.declaration_position > best.declaration_position
                ):
                    best = binding
            if best is not None:
                visible[name] = best
        return visible

    def _resolve_template_argument_types_at(
        self,
        arguments: Sequence[str],
        type_aliases: Dict[str, List[_MetalTypeAliasBinding]],
        position: int,
        declared_types: Dict[str, str],
    ) -> List[str]:
        """Resolve local type aliases and ``decltype(<local>)`` in template arguments.

        An alias such as ``using tile_t = decltype(tile);`` brings the ``decltype`` in only once it is resolved, and a
        declared type may itself name an alias, so each argument alternates the two until neither changes it.
        """
        resolved: List[str] = []
        for argument in arguments:
            current = argument
            for _ in range(8):
                substituted = self._resolve_type_aliases_at(
                    substitute_local_decltypes_in((current,), declared_types)[0],
                    type_aliases,
                    position,
                )
                if substituted == current:
                    break
                current = substituted
            resolved.append(current)
        return resolved

    def _select_conditional_template_arguments(
        self, arguments: Sequence[str]
    ) -> List[str]:
        """Replace each argument spelled ``conditional_t<C, A, B>`` by ``A`` or ``B`` once ``C`` folds."""
        return [
            select_conditional_type(argument, self._evaluate_static_integral_expression)
            or select_nested_conditional_types(
                argument, self._evaluate_static_integral_expression
            )
            for argument in arguments
        ]

    @staticmethod
    def _visible_local_declared_types(
        visible: Dict[str, _MetalIntegralConstantBinding],
    ) -> Dict[str, str]:
        return {
            name: binding.declared_type
            for name, binding in visible.items()
            if binding.declared_type is not None
        }

    @classmethod
    def _local_integral_constants_at(
        cls,
        bindings: Dict[str, List[_MetalIntegralConstantBinding]],
        position: int,
        *,
        names: Optional[Set[str]] = None,
    ) -> Dict[str, str]:
        return {
            name: binding.value
            for name, binding in cls._local_integral_constant_bindings_at(
                bindings,
                position,
                names=names,
            ).items()
            if binding.value is not None
        }

    def _strip_function_parameter_defaults(self, parameters: str) -> str:
        declarations: List[str] = []
        for parameter in self._split_top_level_commas(parameters):
            if not parameter.strip():
                continue
            declaration, _default = self._split_top_level_assignment(parameter)
            declarations.append(declaration.strip())
        return ", ".join(declarations)

    def _canonicalize_struct_scoped_local_declarations(
        self,
        body: str,
        struct: _MetalStructDefinition,
        structs_by_name: Optional[Dict[str, _MetalStructDefinition]],
    ) -> str:
        replacements: List[Tuple[int, int, str]] = []
        local_aliases = self._local_type_alias_shadow_scopes(body)
        search_start = 0
        for statement in self._iter_simple_declarations(body):
            statement_start = body.find(statement, search_start)
            if statement_start == -1:
                continue
            search_start = statement_start + len(statement)
            if self._declaration_is_type_alias(statement):
                continue
            name = self._declared_local_name(statement)
            if name is None:
                continue

            leading = len(statement) - len(statement.lstrip())
            declarator = self._strip_top_level_default_value(statement[leading:])
            paren = self._function_parameter_start(declarator)
            if paren is not None:
                declarator = declarator[:paren].rstrip()
            while declarator.endswith("]"):
                open_bracket = declarator.rfind("[")
                if open_bracket == -1:
                    break
                declarator = declarator[:open_bracket].rstrip()
            name_match = re.search(rf"\b{re.escape(name)}\s*$", declarator)
            if name_match is None:
                continue

            type_start = statement_start + leading
            type_end = type_start + name_match.start()
            type_text = body[type_start:type_end]
            shadowed_aliases = self._shadowed_type_aliases_at(local_aliases, type_start)
            canonical = self._canonicalize_struct_scoped_type(
                type_text,
                struct,
                structs_by_name,
                excluded_aliases=shadowed_aliases,
            )
            if canonical and canonical != type_text.strip():
                replacements.append((type_start, type_end, f"{canonical} "))
        return self._apply_text_replacements(body, replacements)

    @staticmethod
    def _shadowed_type_aliases_at(
        local_aliases: Dict[str, List[Tuple[int, int, int]]], position: int
    ) -> Set[str]:
        return {
            alias
            for alias, bindings in local_aliases.items()
            if any(
                declaration_position <= position and scope_start <= position < scope_end
                for declaration_position, scope_start, scope_end in bindings
            )
        }

    def _canonicalize_struct_scoped_cast_types(
        self,
        body: str,
        struct: _MetalStructDefinition,
        structs_by_name: Optional[Dict[str, _MetalStructDefinition]],
    ) -> str:
        alias_names = set(struct.type_aliases)
        if not alias_names or not (set(IDENTIFIER_RE.findall(body)) & alias_names):
            return body

        ignored_spans = self._find_comment_and_literal_spans(body)
        local_aliases = self._local_type_alias_shadow_scopes(body)
        replacements: List[Tuple[int, int, str]] = []

        def canonical_type(type_start: int, type_end: int) -> Optional[str]:
            type_text = body[type_start:type_end]
            if not (set(IDENTIFIER_RE.findall(type_text)) & alias_names):
                return None
            canonical = self._canonicalize_struct_scoped_type(
                type_text,
                struct,
                structs_by_name,
                excluded_aliases=self._shadowed_type_aliases_at(
                    local_aliases, type_start
                ),
            )
            if not canonical or canonical == type_text.strip():
                return None
            return canonical

        def add_replacement(type_start: int, type_end: int) -> None:
            canonical = canonical_type(type_start, type_end)
            if canonical is not None:
                replacements.append((type_start, type_end, canonical))

        named_cast_pattern = re.compile(
            r"\b(?:static_cast|reinterpret_cast|const_cast)\s*<"
        )
        for match in named_cast_pattern.finditer(body):
            if self._containing_span(match.start(), ignored_spans) is not None:
                continue
            angle_start = body.find("<", match.start(), match.end())
            angle_end = self._find_matching_angle(body, angle_start)
            if angle_end is not None:
                add_replacement(angle_start + 1, angle_end)

        # Function-style and braced casts name the type directly. Restrict this
        # to unqualified owner aliases so another struct's member alias and a
        # block-local alias keep their own scope.
        alias_pattern = "|".join(
            re.escape(alias) for alias in sorted(alias_names, key=len, reverse=True)
        )
        constructor_pattern = re.compile(
            rf"(?<![A-Za-z0-9_:])(?P<type>{alias_pattern})\s*(?=[({{])"
        )
        for match in constructor_pattern.finditer(body):
            if self._containing_span(match.start(), ignored_spans) is not None:
                continue
            prefix = body[: match.start()].rstrip()
            if prefix.endswith(("::", ".", "->")):
                continue
            if match.group("type") in self._shadowed_type_aliases_at(
                local_aliases, match.start()
            ):
                continue
            add_replacement(match.start("type"), match.end("type"))

        # Parenthesized expressions overlap C-style cast syntax. Only rewrite a
        # span that resolves to a pointer/reference type and has a concrete cast
        # operand; ambiguous scalar groups remain unchanged.
        allowed_c_style_identifiers = {
            "atomic",
            "const",
            "constant",
            "device",
            "packed",
            "restrict",
            "thread",
            "threadgroup",
            "threadgroup_imageblock",
            "typename",
            "volatile",
            "read",
            "write",
            "read_write",
            "__restrict",
            "__restrict__",
        }
        for alias_target in struct.type_aliases.values():
            allowed_c_style_identifiers.update(IDENTIFIER_RE.findall(alias_target))
        c_style_cast_pattern = re.compile(
            rf"\((?P<type>[^(){{}};]*\b(?:{alias_pattern})\b[^(){{}};]*)\)"
        )
        for match in c_style_cast_pattern.finditer(body):
            if self._containing_span(match.start(), ignored_spans) is not None:
                continue
            type_text = match.group("type")
            identifiers = set(IDENTIFIER_RE.findall(type_text))
            if not (identifiers & alias_names):
                continue
            if re.search(r"[=;{}?!|^/%+]", type_text):
                continue
            canonical = canonical_type(match.start("type"), match.end("type"))
            if canonical is None or not ({"*", "&"} & set(canonical)):
                continue
            if set(IDENTIFIER_RE.findall(canonical)) - allowed_c_style_identifiers:
                continue
            operand_start = match.end()
            while operand_start < len(body) and body[operand_start].isspace():
                operand_start += 1
            if operand_start >= len(body) or not (
                body[operand_start].isalnum()
                or body[operand_start] in {"_", "(", "{", "*", "&"}
            ):
                continue
            replacements.append((match.start("type"), match.end("type"), canonical))

        return self._apply_text_replacements(body, replacements)

    def _canonicalize_struct_scoped_template_arguments(
        self,
        body: str,
        struct: _MetalStructDefinition,
        method: _MetalStructMethod,
        structs_by_name: Optional[Dict[str, _MetalStructDefinition]],
    ) -> str:
        alias_names = set(struct.type_aliases)
        if not alias_names or not (set(IDENTIFIER_RE.findall(body)) & alias_names):
            return body

        ignored_spans = self._find_comment_and_literal_spans(body)
        local_aliases = self._local_type_alias_shadow_scopes(body)
        value_shadowed_aliases = alias_names & (
            set(method.parameter_names) | self._local_variable_names(body)
        )
        # Lowered methods leave the struct's type scope before explicit helper
        # templates are materialized and type traits are folded, so resolve
        # only their argument lists here.
        replacements: List[Tuple[int, int, str]] = []
        index = 0
        while index < len(body):
            ignored = self._containing_span(index, ignored_spans)
            if ignored is not None:
                index = ignored[1]
                continue
            if not (body[index].isalpha() or body[index] == "_"):
                index += 1
                continue

            identifier, consumed = self._read_identifier(body, index)
            identifier_end = index + consumed
            angle_start = identifier_end
            while angle_start < len(body) and body[angle_start].isspace():
                angle_start += 1
            if angle_start >= len(body) or body[angle_start] != "<":
                index = identifier_end
                continue
            angle_end = self._find_matching_angle(body, angle_start)
            if angle_end is None:
                index = identifier_end
                continue
            call_start = angle_end + 1
            while call_start < len(body) and body[call_start].isspace():
                call_start += 1
            # A type trait's operands are types in every position, e.g. the
            # owner alias in `is_same_v<scalar_T, float>`, so its argument list
            # leaves the struct's type scope too even though no call follows.
            is_type_trait = (
                identifier.removesuffix("_v") in self._METAL_TYPE_TRAIT_ARITIES
            )
            if not is_type_trait and (
                call_start >= len(body) or body[call_start] != "("
            ):
                index = identifier_end
                continue

            arguments = body[angle_start + 1 : angle_end]
            if not (set(IDENTIFIER_RE.findall(arguments)) & alias_names):
                index = angle_end + 1
                continue
            excluded_aliases = value_shadowed_aliases | self._shadowed_type_aliases_at(
                local_aliases, angle_start
            )
            canonical = self._canonicalize_struct_scoped_type(
                arguments,
                struct,
                structs_by_name,
                excluded_aliases=excluded_aliases,
            )
            if canonical and canonical != arguments.strip():
                replacements.append((angle_start + 1, angle_end, canonical))
            index = angle_end + 1

        return self._apply_text_replacements(body, replacements)

    def _split_struct_body(
        self,
        struct_name: str,
        body: str,
        body_offset: int,
        member_prototype_names: Optional[Set[str]] = None,
    ) -> Tuple[
        Set[str],
        Dict[str, str],
        List[_MetalStructMethod],
        List[_MetalStructMethod],
        List[_MetalDataMember],
        List[_MetalConstructor],
        List[Tuple[int, int]],
        bool,
    ]:
        # Walk a struct body separating DATA members from METHOD definitions.
        # A method is a declarator followed by `(params)` then `{...}`; everything
        # else terminated by `;` (or an access-specifier label) is data. Template
        # member functions (`template <...> ...`) are collected SEPARATELY: they
        # are instantiated on demand from their call sites rather than emitted up
        # front (a template method has no single concrete signature).
        data_member_names: Set[str] = set()
        data_member_types: Dict[str, str] = {}
        methods: List[_MetalStructMethod] = []
        template_methods: List[_MetalStructMethod] = []
        # Ordered data members and constructors are needed by the pointer-member
        # scalar-replacement path (declaration order and ctor mapping matter).
        ordered_members: List[_MetalDataMember] = []
        constructors: List[_MetalConstructor] = []
        member_function_body_spans: List[Tuple[int, int]] = []
        layout_supported = True
        if member_prototype_names is None:
            member_prototype_names = set()
        i = 0
        n = len(body)
        while i < n:
            ch = body[i]
            if ch.isspace():
                i += 1
                continue
            if body.startswith("//", i):
                end = body.find("\n", i)
                i = n if end == -1 else end + 1
                continue
            if body.startswith("/*", i):
                end = body.find("*/", i + 2)
                i = n if end == -1 else end + 2
                continue
            # Access specifiers (public:/private:/protected:) are labels, not
            # members; consume up to and including the colon.
            label = re.match(r"(public|private|protected)\s*:", body[i:])
            if label:
                i += label.end()
                continue

            # A `template <...>` member function: capture its definition so a
            # CALLED instance can be instantiated from its call-site argument
            # types. A bare prototype (no body) has nothing to lower and is
            # skipped; only definitions with a body are recorded.
            if re.match(r"template\s*<", body[i:]):
                angle_start = body.find("<", i)
                angle_end = self._find_matching_template_param_angle(body, angle_start)
                if angle_end is None:
                    layout_supported = False
                    break
                method_body_start = self._find_next_top_level_char(
                    body, angle_end + 1, "{"
                )
                semicolon = self._find_next_top_level_char(body, angle_end + 1, ";")
                if method_body_start is not None and (
                    semicolon is None or method_body_start < semicolon
                ):
                    method_body_end = self._find_matching_brace(body, method_body_start)
                    if method_body_end is not None:
                        member_function_body_spans.append(
                            (
                                body_offset + method_body_start + 1,
                                body_offset + method_body_end - 1,
                            )
                        )
                        template_method = self._parse_struct_template_method(
                            struct_name,
                            body,
                            i,
                            angle_start,
                            angle_end,
                            method_body_start,
                            method_body_end,
                            body_offset,
                        )
                        if template_method is not None:
                            if template_method.is_template:
                                template_methods.append(template_method)
                            else:
                                methods.append(template_method)
                        else:
                            constructor = self._parse_struct_constructor(
                                struct_name,
                                body,
                                angle_end + 1,
                                method_body_start,
                                method_body_end,
                                body_offset,
                            )
                            if constructor is not None:
                                parameter_text = body[angle_start + 1 : angle_end]
                                constructor.template_parameters = (
                                    self._template_parameter_names(parameter_text)
                                )
                                constructor.template_parameter_defaults = (
                                    self._template_parameter_defaults(parameter_text)
                                )
                                constructor.span = (
                                    body_offset + i,
                                    body_offset + method_body_end,
                                )
                                constructors.append(constructor)
                    if method_body_end is None:
                        layout_supported = False
                        i = n
                    else:
                        i = method_body_end
                elif semicolon is not None:
                    prototype_name = self._struct_member_prototype_name(
                        body[i:semicolon]
                    )
                    if prototype_name is not None:
                        member_prototype_names.add(prototype_name)
                    i = semicolon + 1
                else:
                    layout_supported = False
                    break
                continue

            # Find the next statement boundary: either a `{` (method body or
            # in-struct initializer braces) or a `;` (data member / declaration).
            brace = self._find_next_top_level_char(
                body,
                i,
                "{",
                track_angles=False,
            )
            semicolon = self._find_next_top_level_char(
                body,
                i,
                ";",
                track_angles=False,
            )
            if brace is not None and (semicolon is None or brace < semicolon):
                method = self._parse_struct_method(struct_name, body, i, brace)
                brace_end = self._find_matching_brace(body, brace)
                if brace_end is None:
                    layout_supported = False
                    break
                if method is not None:
                    member_function_body_spans.append(
                        (
                            body_offset + brace + 1,
                            body_offset + brace_end - 1,
                        )
                    )
                    method.span = (body_offset + i, body_offset + brace_end)
                    methods.append(method)
                    i = brace_end
                    # An optional trailing `;` after a method body.
                    j = i
                    while j < n and body[j].isspace():
                        j += 1
                    if j < n and body[j] == ";":
                        i = j + 1
                    continue
                # `_parse_struct_method` declined this brace-delimited construct.
                # It is either an OUT-OF-SCOPE method definition (constructor,
                # destructor, conversion/comparison operator, ...) or a data
                # member with a brace initializer (`size_t n_{0};`). A
                # method-shaped construct has a top-level parameter list before
                # the body and is left in place untouched so the parser's
                # existing struct-method skipping drops it; a brace-initialized
                # member is recorded as a data member.
                if self._brace_construct_is_method_definition(body[i:brace]):
                    member_function_body_spans.append(
                        (
                            body_offset + brace + 1,
                            body_offset + brace_end - 1,
                        )
                    )
                    # A constructor is captured (for pointer-member scalar
                    # replacement); other declined constructs (destructor,
                    # conversion/comparison operators) are simply left in place.
                    constructor = self._parse_struct_constructor(
                        struct_name, body, i, brace, brace_end, body_offset
                    )
                    if constructor is not None:
                        constructors.append(constructor)
                    i = brace_end
                    # Consume an optional trailing `;`.
                    j = i
                    while j < n and body[j].isspace():
                        j += 1
                    if j < n and body[j] == ";":
                        i = j + 1
                    continue
                decl_semicolon = self._find_next_top_level_char(body, brace_end, ";")
                if decl_semicolon is None:
                    layout_supported = False
                    i = brace_end
                    continue
                declaration = body[i:decl_semicolon]
                declaration_prefix = body[i:brace].strip()
                trailing_declarator = body[brace_end:decl_semicolon].strip()
                nested_type = re.match(
                    r"^(?:struct|class|union|enum)\s+[A-Za-z_]\w*\b",
                    declaration_prefix,
                )
                if nested_type is None or trailing_declarator:
                    layout_supported &= self._record_data_member(
                        declaration,
                        data_member_names,
                        data_member_types,
                        ordered_members,
                        declaration_span=(
                            body_offset + i,
                            body_offset + decl_semicolon + 1,
                        ),
                    )
                i = decl_semicolon + 1
                continue
            if semicolon is None:
                layout_supported = False
                break
            # A declaration terminated by `;`. It may still be a method
            # PROTOTYPE (declarator + params + ;) with no body — those have no
            # definition to lower, so record any data member name otherwise.
            declaration = body[i:semicolon]
            if self._declaration_is_method_prototype(declaration):
                prototype_name = self._struct_member_prototype_name(declaration)
                if prototype_name is not None:
                    member_prototype_names.add(prototype_name)
                if re.search(
                    r"\b(?:alignas|__attribute__|__declspec)\s*\(", declaration
                ):
                    layout_supported = False
                if re.search(r"\(\s*[*&]", declaration):
                    layout_supported = False
            elif not self._declaration_is_type_alias(declaration):
                type_declaration = re.fullmatch(
                    r"\s*(?:struct|class|union|enum)\s+[A-Za-z_]\w*\s*",
                    declaration,
                )
                if type_declaration is None:
                    layout_supported &= self._record_data_member(
                        declaration,
                        data_member_names,
                        data_member_types,
                        ordered_members,
                        declaration_span=(
                            body_offset + i,
                            body_offset + semicolon + 1,
                        ),
                    )
            i = semicolon + 1
        self._assign_receiver_overload_free_names(struct_name, methods)
        return (
            data_member_names,
            data_member_types,
            methods,
            template_methods,
            ordered_members,
            constructors,
            member_function_body_spans,
            layout_supported,
        )

    def _record_data_member(
        self,
        declaration: str,
        data_member_names: Set[str],
        data_member_types: Dict[str, str],
        ordered_members: List[_MetalDataMember],
        declaration_span: Tuple[int, int],
    ) -> bool:
        # Record a data member into the name set, the (normalized) type map, and
        # the ordered list with its FULL declared type text / default / array
        # suffix (used by the pointer-member scalar-replacement path). Strip both
        # assignment and direct-list defaults before deriving the member name:
        # otherwise `uint index{0}` ends in `}` and is silently omitted.
        members = split_member_declarator_list(declaration)
        if members is not None:
            recorded = [
                self._record_data_member(
                    member,
                    data_member_names,
                    data_member_types,
                    ordered_members,
                    declaration_span,
                )
                for member in members
            ]
            return all(recorded)
        declarator, _default, _initializer_style = self._split_data_member_initializer(
            declaration
        )
        name = self._declared_data_member_name(declarator)
        if not name:
            return False
        data_member_names.add(name)
        member = self._parse_ordered_data_member(
            name,
            declaration,
            declaration_span,
        )
        if member is not None:
            data_member_types[name] = self._normalize_template_argument_text(
                f"{member.type_text}{member.array_suffix}"
            )
            ordered_members.append(member)
            return True
        return False

    def _parse_ordered_data_member(
        self,
        name: str,
        declaration: str,
        declaration_span: Optional[Tuple[int, int]] = None,
    ) -> Optional[_MetalDataMember]:
        # Split `[qualifiers] Type name [array] [= default|{default}]` into its
        # full type text (preserving address space / cv / pointer), trailing array
        # suffix, and default initializer. Returns None when the type cannot be
        # isolated.
        lhs, default, initializer_style = self._split_data_member_initializer(
            declaration
        )
        lhs = lhs.strip()
        # Peel trailing array extents (`data[N][M]`) off the declarator.
        array_suffix = ""
        while lhs.endswith("]"):
            open_bracket = lhs.rfind("[")
            if open_bracket == -1:
                break
            array_suffix = lhs[open_bracket:] + array_suffix
            lhs = lhs[:open_bracket].rstrip()
        # Remove the trailing member name to leave the (full) type text.
        type_text = re.sub(rf"\b{re.escape(name)}\s*$", "", lhs).strip()
        type_text = re.sub(r"\s+", " ", type_text)
        if not type_text:
            return None
        return _MetalDataMember(
            name=name,
            type_text=type_text,
            default=default,
            initializer_style=initializer_style,
            array_suffix=array_suffix,
            span=declaration_span,
        )

    def _split_data_member_initializer(
        self,
        declaration: str,
    ) -> Tuple[str, Optional[str], str]:
        """Split a data declarator from an assignment or direct-list default.

        This deliberately recognizes only a single trailing top-level `{...}`
        group whose prefix has a data-declarator shape. Method/constructor bodies,
        aggregate type definitions, and malformed/trailing constructs remain
        unsplit so their existing fail-closed classification is preserved.
        """
        lhs, default = self._split_top_level_assignment(declaration)
        if default is not None:
            return lhs, default, "="

        text = declaration.strip()
        brace = self._find_next_top_level_char(text, 0, "{")
        if brace is None:
            return text, None, "="
        close = self._find_matching_delimiter(text, brace, "{", "}")
        if close is None or text[close + 1 :].strip():
            return text, None, "="

        declarator = text[:brace].rstrip()
        if not declarator or self._function_parameter_start(declarator) is not None:
            return text, None, "="
        if re.fullmatch(
            r"(?:struct|class|union|enum|namespace)\s+[A-Za-z_]\w*",
            declarator,
        ):
            return text, None, "="

        name_region = declarator
        while name_region.endswith("]"):
            open_bracket = name_region.rfind("[")
            if open_bracket == -1:
                return text, None, "="
            name_region = name_region[:open_bracket].rstrip()
        if (
            len(IDENTIFIER_RE.findall(name_region)) < 2
            or re.search(r"[A-Za-z_][A-Za-z0-9_]*\s*$", name_region) is None
        ):
            return text, None, "="
        return declarator, text[brace + 1 : close].strip(), "direct-list"

    def _split_top_level_assignment(self, text: str) -> Tuple[str, Optional[str]]:
        # Split at the first top-level `=` that is a plain assignment (not part of
        # `==`/`<=`/`>=`/`!=` or a compound operator), returning (lhs, rhs) with
        # rhs None when there is no such assignment. Used to peel a data member's
        # default initializer from its declarator.
        return split_top_level_assignment(text, self._read_string)

    def _parse_struct_constructor(
        self,
        struct_name: str,
        body: str,
        decl_start: int,
        brace: int,
        brace_end: int,
        body_offset: int = 0,
    ) -> Optional[_MetalConstructor]:
        # Parse a constructor `[macros] Name(params) : init_list { body }`.
        # `Name` is a bare identifier with NO return type (that is what makes it a
        # constructor rather than a method). Materialized template constructors
        # are renamed only after this parser has identified their source spans.
        # Destructors and operators are rejected. Returns None for any
        # non-constructor construct.
        header = body[decl_start:brace]
        paren_start = self._function_parameter_start(header)
        if paren_start is None:
            return None
        paren_end = self._find_matching_delimiter(header, paren_start, "(", ")")
        if paren_end is None:
            return None
        before = header[:paren_start].rstrip()
        is_constexpr = bool(re.search(r"\b(?:constexpr|C10_METAL_CONSTEXPR)\b", before))
        name_region = self._strip_function_qualifier_macros(before)
        name_region = re.sub(r"\b(?:constexpr|explicit|inline)\b", " ", name_region)
        name_region = re.sub(r"\s+", " ", name_region).strip()
        # A genuine constructor declarator is a single identifier: no return type
        # tokens, no `~` destructor, no `operator`.
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name_region or ""):
            return None
        if name_region == "operator":
            return None
        params_text = header[paren_start + 1 : paren_end]
        param_names = self._parameter_identifier_names(params_text)
        param_types = self._parameter_declared_types(params_text)
        # The member initializer list sits between `)` and `{`, introduced by `:`.
        between = header[paren_end + 1 :].strip()
        init_text = ""
        init_map: Dict[str, str] = {}
        colon = self._find_next_top_level_char(between, 0, ":")
        receiver_text = between if colon is None else between[:colon]
        if colon is not None:
            init_text = between[colon + 1 :].strip()
            init_map = self._parse_constructor_init_list(init_text)
        receiver_address_spaces = self._receiver_address_spaces(receiver_text)
        constructor_body = body[brace + 1 : brace_end - 1]
        # `prefix` is everything up to and including `(` (leading macros + the
        # constructor name), so the pointer-member promotion path can rebuild the
        # constructor header after dropping pointer parameters.
        prefix = header[: paren_start + 1]
        return _MetalConstructor(
            param_names=param_names,
            init_map=init_map,
            body=constructor_body,
            span=(body_offset + decl_start, body_offset + brace_end),
            prefix=prefix,
            params_text=params_text,
            init_text=init_text,
            param_types=param_types,
            is_constexpr=is_constexpr,
            receiver_address_spaces=receiver_address_spaces,
        )

    def _parameter_declared_types(self, params_text: str) -> List[str]:
        # Return the FULL declared type text of each parameter (in order),
        # preserving address space / cv / pointer (e.g. ``const device float2*``)
        # and dropping the parameter name / default. Mirrors
        # ``_parameter_identifier_names`` but keeps the type rather than the name,
        # so a pointer parameter can be recognized by its ``*``.
        types: List[str] = []
        for parameter in self._split_top_level_commas(params_text):
            parameter = parameter.strip()
            if not parameter:
                continue
            # Drop any default value.
            lhs, _default = self._split_top_level_assignment(parameter)
            lhs = self._strip_trailing_metal_attributes(lhs.strip())
            # Peel trailing array extents that belong to the declarator.
            while lhs.endswith("]"):
                open_bracket = lhs.rfind("[")
                if open_bracket == -1:
                    break
                lhs = lhs[:open_bracket].rstrip()
            name_match = re.search(r"([A-Za-z_][A-Za-z0-9_]*)\s*$", lhs)
            if name_match is None:
                types.append(re.sub(r"\s+", " ", lhs).strip())
                continue
            type_text = lhs[: name_match.start()].strip()
            # A bare-type parameter (`threadgroup float*`) leaves nothing after
            # stripping the "name": keep the whole text as the type instead.
            if not type_text:
                type_text = lhs
            types.append(re.sub(r"\s+", " ", type_text).strip())
        return types

    def _parse_constructor_init_list(self, init_text: str) -> Dict[str, str]:
        # Parse `member(expr), member2{expr2}, ...` into {member: expr}. Each
        # entry initializes one member; the expression usually names a constructor
        # parameter (`in(in_)`) but any expression is captured verbatim.
        init_map: Dict[str, str] = {}
        for item in self._split_top_level_commas(init_text):
            item = item.strip()
            if not item:
                continue
            match = re.match(r"([A-Za-z_][A-Za-z0-9_]*)\s*([\(\{])", item)
            if match is None:
                continue
            member = match.group(1)
            open_char = match.group(2)
            close_char = ")" if open_char == "(" else "}"
            open_index = match.end() - 1
            close_index = self._find_matching_delimiter(
                item, open_index, open_char, close_char
            )
            if close_index is None:
                continue
            init_map[member] = item[open_index + 1 : close_index].strip()
        return init_map

    def _parse_struct_template_method(
        self,
        struct_name: str,
        body: str,
        decl_start: int,
        angle_start: int,
        angle_end: int,
        method_body_start: int,
        method_body_end: int,
        body_offset: int,
    ) -> Optional[_MetalStructMethod]:
        # Parse a `template <...> RetType name(params) { body }` member function.
        # The declarator AFTER the template-parameter list is parsed with the
        # same machinery as a non-template method; the template parameter names
        # are recorded so the method can be instantiated from a call site.
        parameter_text = body[angle_start + 1 : angle_end]
        template_parameters = self._template_parameter_names(parameter_text)
        if not template_parameters:
            method = self._parse_struct_method(
                struct_name, body, angle_end + 1, method_body_start
            )
            if method is None:
                return None
            method.span = (body_offset + decl_start, body_offset + method_body_end)
            return method
        method = self._parse_struct_method(
            struct_name, body, angle_end + 1, method_body_start
        )
        if method is None:
            return None
        method.template_parameters = template_parameters
        method.template_parameter_types = {
            parameter.name: parameter.declared_type
            for parameter in self._parse_template_parameter_list(parameter_text)
            if parameter.name is not None
            and not parameter.is_type_parameter
            and parameter.declared_type is not None
        }
        method.variadic_template_parameters = self._variadic_template_parameter_names(
            parameter_text
        )
        method.template_parameter_defaults = self._template_parameter_defaults(
            parameter_text
        )
        method.template_constraints = self._template_parameter_constraints(
            parameter_text
        )
        # A SECOND SFINAE layer can hide on the RETURN TYPE, e.g. Sum/Min/Max:
        #   metal::enable_if_t<metal::is_integral_v<T>, T> simd_reduce_impl(T val)
        # Unwrap it so the emitted free function returns the real value type while
        # the wrapped constraint joins the overload-selection set.
        return_constraint, unwrapped = self._split_return_type_sfinae(
            method.return_type
        )
        if return_constraint is not None:
            method.return_type = unwrapped
            method.return_type_constraint = return_constraint
            method.template_constraints = [
                *method.template_constraints,
                return_constraint,
            ]
        # The span covers the whole `template <...> ... { ... }` definition so the
        # data-only struct removes it cleanly.
        method.span = (body_offset + decl_start, body_offset + method_body_end)
        return method

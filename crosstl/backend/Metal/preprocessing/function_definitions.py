"""Out-of-line member definitions, function definition scans, function references and explicit template function materialization."""

import re
from bisect import (
    bisect_left,
    bisect_right,
)
from heapq import (
    heappop,
    heappush,
)
from typing import (
    Dict,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
)

from ..member_address_types import merge_receiver_variable_types
from ..scope_qualifier_keywords import NON_SCOPE_KEYWORDS
from ..template_argument_text import normalize_template_argument_text
from .support import (
    CONTAINING_SPAN_CACHE_LIMIT,
    SCAN_STOP_RE,
    SOURCE_ANALYSIS_CACHE_LIMIT,
    MetalTemplateSpecializationError,
    _MetalContainingSpanLookup,
    _MetalFunctionDefinition,
    _MetalOutOfLineMemberCallSite,
    _MetalOutOfLineMemberDefinition,
    _MetalReachabilityTypeContext,
    _MetalReceiverContract,
    _MetalStructDefinition,
    _MetalStructMethod,
)


class MetalFunctionDefinitionMixin:
    """Members of ``MetalPreprocessor``: out-of-line member definitions, function definition scans, function references and explicit template function materialization."""

    def _out_of_line_member_definition(
        self,
        code: str,
        function: _MetalFunctionDefinition,
        struct_names: Set[str],
    ) -> Optional[_MetalOutOfLineMemberDefinition]:
        header = code[function.span[0] : function.body_span[0] - 1]
        owner = self._qualified_function_owner(header)
        if owner is None:
            return None
        owner_name = owner.rsplit("::", 1)[-1]
        if owner_name not in struct_names:
            return None

        parameter_span = self._function_parameter_list_span(header)
        if parameter_span is None:
            return None
        operator_call = re.search(r"\boperator\s*\(\s*\)", header)
        if operator_call is not None:
            method_start = operator_call.start()
        else:
            method_match = re.search(
                r"[A-Za-z_][A-Za-z0-9_]*\s*$",
                header[: parameter_span[0]],
            )
            if method_match is None:
                return None
            method_start = method_match.start()

        scope_end = method_start
        while scope_end > 0 and header[scope_end - 1].isspace():
            scope_end -= 1
        if scope_end < 2 or header[scope_end - 2 : scope_end] != "::":
            return None
        owner_end = scope_end - 2
        owner_match = re.search(
            r"(?P<owner>[A-Za-z_][A-Za-z0-9_]*"
            r"(?:\s*::\s*[A-Za-z_][A-Za-z0-9_]*)*)\s*$",
            header[:owner_end],
        )
        if owner_match is None:
            return None
        parsed_owner = re.sub(r"\s*::\s*", "::", owner_match.group("owner"))
        if parsed_owner != owner:
            return None

        dequalified_header = header[: owner_match.start("owner")] + header[scope_end:]
        body_start = function.body_span[0] - 1
        method_source = dequalified_header + code[body_start : function.span[1]]
        method = self._parse_struct_method(
            owner_name,
            method_source,
            0,
            len(dequalified_header),
        )
        if method is None:
            return None
        method.span = function.span
        return _MetalOutOfLineMemberDefinition(
            function=function,
            owner_name=owner_name,
            method=method,
        )

    def _reachability_type_context(
        self,
        code: str,
        excluded_spans: List[Tuple[int, int]],
        structs: Sequence[_MetalStructDefinition],
    ) -> _MetalReachabilityTypeContext:
        structs_by_name = {struct.name: struct for struct in structs}
        struct_names = set(structs_by_name)
        scan_skip_spans = sorted(
            set(excluded_spans + [struct.span for struct in structs])
        )
        struct_type_aliases = self._collect_struct_type_aliases(
            code,
            struct_names,
            scan_skip_spans,
            structs_by_name,
        )
        source_type_aliases = self._collect_local_type_alias_bindings(
            code,
            [(0, len(code))],
            skip_spans=excluded_spans,
        )
        receiver_variable_types = self._collect_struct_variable_types(
            code,
            struct_names,
            scan_skip_spans,
            include_indirect=True,
        )
        aliased_variable_types = self._collect_aliased_struct_variable_types(
            code,
            struct_type_aliases,
            scan_skip_spans,
        )
        for name, entries in aliased_variable_types.items():
            existing = receiver_variable_types.setdefault(name, [])
            existing.extend(entry for entry in entries if entry not in existing)
            existing.sort(key=lambda item: item[0])

        buffer_element_types = self._collect_buffer_element_types(
            code,
            scan_skip_spans,
        )
        local_variable_types = self._collect_local_variable_types(
            code,
            scan_skip_spans,
        )
        merge_receiver_variable_types(
            code, local_variable_types, receiver_variable_types
        )
        self._collect_function_parameter_types(
            code,
            scan_skip_spans,
            buffer_element_types,
            local_variable_types,
        )
        self._collect_auto_local_variable_types(
            code,
            scan_skip_spans,
            buffer_element_types,
            local_variable_types,
            structs_by_name,
            type_aliases=source_type_aliases,
            local_integral_constants=self._collect_local_integral_constant_bindings(
                code,
                [(0, len(code))],
                source_type_aliases,
                self._active_static_constexpr_functions,
                include_file_scope=True,
            ),
        )
        return _MetalReachabilityTypeContext(
            buffer_element_types=buffer_element_types,
            local_variable_types=local_variable_types,
            receiver_variable_types=receiver_variable_types,
            structs_by_name=structs_by_name,
            struct_type_aliases=struct_type_aliases,
            source_type_aliases=source_type_aliases,
        )

    def _reachable_out_of_line_member_functions(
        self,
        code: str,
        function: _MetalFunctionDefinition,
        owner_name: str,
        candidates: Sequence[_MetalOutOfLineMemberDefinition],
        context: _MetalReachabilityTypeContext,
    ) -> List[_MetalOutOfLineMemberDefinition]:
        call_sites = self._out_of_line_member_call_sites(
            code,
            function,
            owner_name,
            context,
        )
        if not call_sites:
            return []

        owner = context.structs_by_name.get(owner_name)
        selected: Dict[Tuple[int, int], _MetalOutOfLineMemberDefinition] = {}
        for call_site in call_sites:
            method_candidates = [
                candidate
                for candidate in candidates
                if candidate.method.name == call_site.method_name
            ]
            if not method_candidates:
                continue
            receiver_candidates = [
                candidate
                for candidate in method_candidates
                if call_site.receiver_contract is None
                or not self._receiver_candidate_mismatches(
                    candidate.method,
                    call_site.receiver_contract,
                )
            ]
            if not receiver_candidates:
                continue
            argument_types = self._reachable_member_call_argument_types(
                code,
                call_site.argument_open,
                context,
            )
            if argument_types is None:
                for candidate in receiver_candidates:
                    selected[candidate.function.span] = candidate
                continue

            exact_matches = [
                candidate
                for candidate in receiver_candidates
                if self._concrete_method_matches_call(
                    candidate.method,
                    code,
                    call_site.argument_open,
                    context.buffer_element_types,
                    context.local_variable_types,
                    context.receiver_variable_types,
                    context.structs_by_name,
                )
            ]
            if len(exact_matches) > 1 and call_site.receiver_contract is not None:
                ranked = [
                    (
                        self._receiver_candidate_preference_rank(
                            candidate.method,
                            call_site.receiver_contract,
                        ),
                        candidate,
                    )
                    for candidate in exact_matches
                ]
                best_rank = min(rank for rank, _candidate in ranked)
                exact_matches = [
                    candidate for rank, candidate in ranked if rank == best_rank
                ]
            if exact_matches:
                for candidate in exact_matches:
                    selected[candidate.function.span] = candidate
                continue

            template_fallback = owner is not None and any(
                method.name == call_site.method_name
                and self._callable_accepts_argument_count(
                    method.parameters,
                    len(argument_types),
                )
                for method in owner.template_methods
            )
            for candidate in receiver_candidates:
                if not self._callable_accepts_argument_count(
                    candidate.method.parameters,
                    len(argument_types),
                ):
                    continue
                if template_fallback and self._out_of_line_member_mismatch_is_proven(
                    candidate.method,
                    argument_types,
                    call_site.argument_open,
                    context,
                ):
                    continue
                selected[candidate.function.span] = candidate

        return list(selected.values())

    def _out_of_line_member_mismatch_is_proven(
        self,
        method: _MetalStructMethod,
        argument_types: Sequence[str],
        argument_position: int,
        context: _MetalReachabilityTypeContext,
    ) -> bool:
        parameters = [
            parameter
            for parameter in self._split_top_level_commas(method.parameters)
            if parameter.strip()
            and parameter.strip() != "void"
            and "..." not in parameter
        ]
        if len(argument_types) > len(parameters):
            return True

        declared_types = [
            self._function_parameter_value_type(parameter)
            for parameter in parameters[: len(argument_types)]
        ]
        canonical_declared = [
            self._canonicalize_type_aliases_at(
                declared_type,
                context.source_type_aliases,
                method.span[0],
            )
            or declared_type
            for declared_type in declared_types
        ]
        canonical_arguments = [
            self._canonicalize_type_aliases_at(
                argument_type,
                context.source_type_aliases,
                argument_position,
            )
            or argument_type
            for argument_type in argument_types
        ]
        known_struct_types = set(context.structs_by_name)
        known_struct_types.update(
            struct.qualified_name
            for struct in context.structs_by_name.values()
            if struct.qualified_name
        )

        def is_known_value_type(type_name: str) -> bool:
            normalized = self._normalize_inferred_type(type_name)
            return self._is_metal_scalar_or_vector_type(normalized) or (
                normalized in known_struct_types
            )

        if not all(
            is_known_value_type(type_name)
            for type_name in (*canonical_declared, *canonical_arguments)
        ):
            return False
        return [
            self._normalize_inferred_type(type_name) for type_name in canonical_declared
        ] != [
            self._normalize_inferred_type(type_name)
            for type_name in canonical_arguments
        ]

    def _out_of_line_member_call_sites(
        self,
        code: str,
        function: _MetalFunctionDefinition,
        owner_name: str,
        context: _MetalReachabilityTypeContext,
    ) -> List[_MetalOutOfLineMemberCallSite]:
        call_sites: Dict[
            Tuple[str, int],
            _MetalOutOfLineMemberCallSite,
        ] = {}
        body_start, body_end = function.body_span
        cursor = body_start
        while cursor < body_end:
            if code[cursor] in "\"'":
                _literal, consumed = self._read_string(code, cursor)
                cursor += consumed
                continue
            if code.startswith("//", cursor):
                line_end = code.find("\n", cursor, body_end)
                cursor = body_end if line_end == -1 else line_end + 1
                continue
            if code.startswith("/*", cursor):
                comment_end = code.find("*/", cursor + 2, body_end)
                cursor = body_end if comment_end == -1 else comment_end + 2
                continue
            if not (code[cursor].isalpha() or code[cursor] == "_"):
                cursor += 1
                continue

            ident, consumed = self._read_identifier(code, cursor)
            ident_end = cursor + consumed
            after = ident_end
            while after < body_end and code[after].isspace():
                after += 1

            if ident == owner_name:
                temporary = self._temporary_functor_call(code, after)
                if (
                    temporary is not None
                    and temporary.argument_close < body_end
                    and self._temporary_functor_is_candidate(
                        code,
                        cursor,
                        temporary,
                    )
                ):
                    call_sites[("operator()", temporary.argument_open)] = (
                        _MetalOutOfLineMemberCallSite(
                            method_name="operator()",
                            argument_open=temporary.argument_open,
                            receiver_contract=_MetalReceiverContract(
                                struct_name=owner_name,
                                value_category="rvalue",
                                source_type=f"thread {owner_name}",
                            ),
                        )
                    )
                elif code[after : after + 2] == "::":
                    member_start = after + 2
                    while member_start < body_end and code[member_start].isspace():
                        member_start += 1
                    member, member_length = self._read_identifier(code, member_start)
                    if member:
                        suffix = self._member_template_call_suffix(
                            code,
                            member_start + member_length,
                        )
                        if suffix is not None and suffix[0] < body_end:
                            call_sites[(member, suffix[0])] = (
                                _MetalOutOfLineMemberCallSite(
                                    method_name=member,
                                    argument_open=suffix[0],
                                    receiver_contract=None,
                                )
                            )

            declaration, receiver_type = self._receiver_struct_type_at(
                code,
                ident,
                cursor,
                set(context.structs_by_name),
                context.struct_type_aliases,
            )
            if receiver_type is None:
                inferred_receiver_type = self._resolve_declared_type_at(
                    context.local_variable_types,
                    ident,
                    cursor,
                )
                if inferred_receiver_type in context.structs_by_name:
                    receiver_type = inferred_receiver_type
            if receiver_type == owner_name:
                receiver_contract = (
                    self._receiver_contract_from_type_text(
                        declaration.raw_type_text,
                        owner_name,
                    )
                    if declaration is not None
                    else _MetalReceiverContract(
                        struct_name=owner_name,
                        source_type=f"thread {owner_name}",
                    )
                )
                if after < body_end and code[after] == "(":
                    call_sites[("operator()", after)] = _MetalOutOfLineMemberCallSite(
                        method_name="operator()",
                        argument_open=after,
                        receiver_contract=receiver_contract,
                    )
                elif after < body_end and (
                    code[after] == "." or code[after : after + 2] == "->"
                ):
                    member_start = after + (2 if code[after] == "-" else 1)
                    while member_start < body_end and code[member_start].isspace():
                        member_start += 1
                    member, member_length = self._read_identifier(code, member_start)
                    if member:
                        suffix = self._member_template_call_suffix(
                            code,
                            member_start + member_length,
                        )
                        if suffix is not None and suffix[0] < body_end:
                            call_sites[(member, suffix[0])] = (
                                _MetalOutOfLineMemberCallSite(
                                    method_name=member,
                                    argument_open=suffix[0],
                                    receiver_contract=receiver_contract,
                                )
                            )

            cursor = ident_end
        return list(call_sites.values())

    def _reachable_member_call_argument_types(
        self,
        code: str,
        arg_open: int,
        context: _MetalReachabilityTypeContext,
    ) -> Optional[List[str]]:
        arg_close = self._find_matching_delimiter(code, arg_open, "(", ")")
        if arg_close is None:
            return None
        arguments = [
            argument
            for argument in self._split_top_level_commas(code[arg_open + 1 : arg_close])
            if argument.strip()
        ]
        buffer_view = self._flatten_types_at(
            context.buffer_element_types,
            arg_open,
        )
        local_view = self._flatten_types_at(
            context.local_variable_types,
            arg_open,
        )
        field_types = self._struct_field_types_at(
            context.receiver_variable_types,
            context.structs_by_name,
            arg_open,
        )
        inferred_types: List[str] = []
        for argument in arguments:
            inferred = self._infer_argument_type(
                argument,
                buffer_view,
                local_view,
                field_types,
                context.structs_by_name,
            )
            if inferred is None:
                return None
            inferred_types.append(self._normalize_inferred_type(inferred))
        return inferred_types

    def _qualified_function_owner(self, header: str) -> Optional[str]:
        parameter_span = self._function_parameter_list_span(header)
        if parameter_span is None:
            return None
        before_parameters = header[: parameter_span[0]].rstrip()
        operator_call = re.search(
            r"(?P<owner>[A-Za-z_][A-Za-z0-9_]*"
            r"(?:\s*::\s*[A-Za-z_][A-Za-z0-9_]*)*)"
            r"\s*::\s*operator\s*\(\s*\)\s*$",
            before_parameters,
        )
        if operator_call is not None:
            return re.sub(r"\s+", "", operator_call.group("owner"))
        method = re.search(
            r"(?P<owner>[A-Za-z_][A-Za-z0-9_]*"
            r"(?:\s*::\s*[A-Za-z_][A-Za-z0-9_]*)*)"
            r"\s*::\s*(?:~?[A-Za-z_][A-Za-z0-9_]*|operator\s*[^\s]+)\s*$",
            before_parameters,
        )
        if method is None:
            return None
        return re.sub(r"\s+", "", method.group("owner"))

    def _unreachable_function_spans(
        self,
        code: str,
        excluded_spans: List[Tuple[int, int]],
    ) -> List[Tuple[int, int]]:
        functions = self._find_non_template_function_definitions(code, excluded_spans)
        reachable = self._reachable_function_spans(code, excluded_spans)
        if reachable is None:
            return []
        reachable_set = set(reachable)
        return [
            function.span
            for function in functions
            if function.span not in reachable_set
        ]

    def _find_non_template_function_definitions(
        self,
        code: str,
        excluded_spans: List[Tuple[int, int]],
    ) -> List[_MetalFunctionDefinition]:
        key = tuple(excluded_spans)
        cache = self._source_analysis(code).function_definitions
        cached = cache.pop(key, None)
        if cached is None:
            cached = tuple(
                self._scan_non_template_function_definitions(code, list(key))
            )
            cache = self._source_analysis(code).function_definitions
        cache[key] = cached
        if len(cache) > SOURCE_ANALYSIS_CACHE_LIMIT:
            del cache[next(iter(cache))]
        return list(cached)

    def _scan_non_template_function_definitions(
        self,
        code: str,
        excluded_spans: List[Tuple[int, int]],
    ) -> List[_MetalFunctionDefinition]:
        functions: List[_MetalFunctionDefinition] = []
        namespace_body_starts = {
            start - 1 for start, _end, _name in self._find_namespace_spans(code)
        }
        pos = 0
        while True:
            body_start = self._find_next_top_level_char(code, pos, "{")
            if body_start is None:
                break
            excluded = self._containing_span(body_start, excluded_spans)
            if excluded is not None:
                pos = excluded[1]
                continue
            body_end = self._find_matching_brace(code, body_start)
            if body_end is None:
                break

            declaration_start = self._function_declaration_start(code, body_start)
            header = code[declaration_start:body_start]
            function_name = self._function_name_from_header(header)
            if function_name is not None:
                functions.append(
                    _MetalFunctionDefinition(
                        name=function_name,
                        span=(declaration_start, body_end),
                        body_span=(body_start + 1, body_end - 1),
                        is_entry=self._header_declares_entry(header),
                    )
                )
                pos = body_end
                continue
            if body_start in namespace_body_starts:
                pos = body_start + 1
                continue
            pos = body_end
        return functions

    def _function_declaration_start(self, code: str, body_start: int) -> int:
        ignored_spans = self._find_comment_and_literal_spans(code)
        previous_semicolon = self._previous_unignored_char(
            code,
            ";",
            body_start,
            ignored_spans,
        )
        previous_block = self._previous_unignored_char(
            code,
            "}",
            body_start,
            ignored_spans,
        )
        return max(previous_semicolon, previous_block) + 1

    def _previous_unignored_char(
        self,
        code: str,
        target: str,
        end: int,
        ignored_spans: List[Tuple[int, int]],
    ) -> int:
        position = code.rfind(target, 0, end)
        while position >= 0:
            ignored = self._containing_span(position, ignored_spans)
            if ignored is None:
                return position
            position = code.rfind(target, 0, ignored[0])
        return -1

    def _find_function_references(
        self,
        code: str,
        function_names: Set[str],
        member_owner_names: Optional[Set[str]] = None,
    ) -> Set[str]:
        references: Set[str] = set()
        if not function_names:
            return references
        i = 0
        while i < len(code):
            if code[i] in "\"'":
                _literal, consumed = self._read_string(code, i)
                i += consumed
                continue
            if code.startswith("//", i):
                end = code.find("\n", i)
                if end == -1:
                    break
                i = end + 1
                continue
            if code.startswith("/*", i):
                end = code.find("*/", i + 2)
                if end == -1:
                    break
                i = end + 2
                continue
            if code[i].isalpha() or code[i] == "_":
                ident, consumed = self._read_identifier(code, i)
                j = i + consumed
                while j < len(code) and code[j].isspace():
                    j += 1
                if (
                    ident in function_names
                    and j < len(code)
                    and code[j] == "("
                    and not self._function_reference_is_member_call(
                        code,
                        i,
                        member_owner_names or set(),
                    )
                ):
                    references.add(ident)
                i += consumed
                continue
            i = self._next_scan_stop(code, i + 1)
        return references

    def _function_reference_is_member_call(
        self,
        code: str,
        identifier_start: int,
        member_owner_names: Set[str],
    ) -> bool:
        previous = identifier_start - 1
        while previous >= 0 and code[previous].isspace():
            previous -= 1
        if previous < 0:
            return False
        if code[previous] == ".":
            return True
        if previous >= 1 and code[previous - 1 : previous + 1] == "->":
            return True
        if previous < 1 or code[previous - 1 : previous + 1] != "::":
            return False

        owner_end = previous - 1
        owner_start = owner_end
        while owner_start > 0 and (
            code[owner_start - 1].isalnum() or code[owner_start - 1] == "_"
        ):
            owner_start -= 1
        return code[owner_start:owner_end] in member_owner_names

    def _find_identifier_references(
        self,
        code: str,
        identifiers: Set[str],
    ) -> Set[str]:
        references: Set[str] = set()
        if not identifiers:
            return references
        i = 0
        while i < len(code):
            if code[i] in "\"'":
                _literal, consumed = self._read_string(code, i)
                i += consumed
                continue
            if code.startswith("//", i):
                end = code.find("\n", i)
                if end == -1:
                    break
                i = end + 1
                continue
            if code.startswith("/*", i):
                end = code.find("*/", i + 2)
                if end == -1:
                    break
                i = end + 2
                continue
            if code[i].isalpha() or code[i] == "_":
                ident, consumed = self._read_identifier(code, i)
                if ident in identifiers:
                    references.add(ident)
                i += consumed
                continue
            i = self._next_scan_stop(code, i + 1)
        return references

    def _find_template_declaration_spans(self, code: str) -> List[Tuple[int, int]]:
        analysis = self._source_analysis(code)
        if analysis.template_declaration_spans is None:
            analysis.template_declaration_spans = self._scan_template_declaration_spans(
                code
            )
        return analysis.template_declaration_spans

    def _scan_template_declaration_spans(self, code: str) -> List[Tuple[int, int]]:
        spans: List[Tuple[int, int]] = []
        pos = 0
        while True:
            match = re.search(r"\btemplate\s*<", code[pos:])
            if match is None:
                break
            start = pos + match.start()
            angle_start = code.find("<", start)
            angle_end = self._find_matching_template_param_angle(code, angle_start)
            if angle_end is None:
                pos = start + len("template")
                continue

            declaration_start = angle_end + 1
            body_start = self._find_next_top_level_char(
                code, declaration_start, "{", stop_at=";"
            )
            semicolon = self._find_next_top_level_char(
                code, declaration_start, ";", stop_at="{"
            )
            if body_start is not None and (semicolon is None or body_start < semicolon):
                body_end = self._find_matching_brace(code, body_start)
                if body_end is None:
                    pos = body_start + 1
                    continue
                spans.append((start, body_end))
                pos = body_end
                continue
            if semicolon is not None:
                spans.append((start, semicolon + 1))
                pos = semicolon + 1
                continue
            pos = declaration_start
        return spans

    def _find_explicit_template_function_specializations(
        self, code: str, *, include_namespaced: bool = False
    ) -> Dict[
        Tuple[str, Tuple[str, ...], Tuple[str, ...]],
        Dict[str, object],
    ]:
        """Return exact global free-function specializations with concrete bodies.

        With ``include_namespaced``, a specialization declared directly in a
        namespace body counts too, and its entry records the namespace; the
        first key component is then the namespace-qualified name, so equal
        signatures in two namespaces stay distinct. Every other entry records
        the namespace ``""``.

        Direct Metal preprocessing intentionally preserves explicit specializations
        as source C++. Entry-scoped project translation targets template-hostile
        backends, however, and must select the exact body once call-site inference
        supplies all template arguments. Only unqualified declarations at global
        scope are admitted here. Namespaced declarations and out-of-line member
        specializations retain owner semantics that this free-function materializer
        cannot preserve, so they remain available to their owner-aware paths.

        C++ permits distinct overloaded primaries to have explicit specializations
        with the same template arguments. The concrete parameter signature is
        therefore part of the identity; keying only by template arguments can bind
        a valid body to the wrong overload or reject two valid specializations.
        """
        specializations: Dict[
            Tuple[str, Tuple[str, ...], Tuple[str, ...]],
            Dict[str, object],
        ] = {}
        masked_code = self._mask_comments_and_literals(code)
        namespace_spans = self._find_namespace_spans(code) if include_namespaced else []
        scope_cursor = 0
        scope_depth = 0
        for match in re.finditer(r"\btemplate\s*<\s*>\s*", masked_code):
            for character in masked_code[scope_cursor : match.start()]:
                if character == "{":
                    scope_depth += 1
                elif character == "}":
                    scope_depth = max(0, scope_depth - 1)
            scope_cursor = match.start()
            namespace = ""
            if scope_depth != 0:
                # Admitted only when every enclosing brace opens a namespace body.
                enclosing = [
                    span
                    for span in namespace_spans
                    if span[0] <= match.start() < span[1]
                ]
                if not include_namespaced or len(enclosing) != scope_depth:
                    continue
                namespace = self._namespace_at(namespace_spans, match.start())

            declaration_start = match.end()
            body_start = self._find_next_top_level_char(
                masked_code, declaration_start, "{"
            )
            semicolon = self._find_next_top_level_char(
                masked_code, declaration_start, ";"
            )
            if body_start is None or (semicolon is not None and semicolon < body_start):
                continue
            body_end = self._find_matching_brace(code, body_start)
            if body_end is None:
                continue

            header = masked_code[declaration_start:body_start]
            paren_index = self._function_parameter_start(header)
            if paren_index is None:
                continue
            before_params = header[:paren_index].rstrip()
            angle_end: Optional[int] = None
            angle_start: Optional[int] = None
            if before_params.endswith(">"):
                angle_end = len(before_params) - 1
                angle_start = self._matching_open_angle(before_params, angle_end)
            if angle_start is not None and angle_end is not None:
                name_source = before_params[:angle_start]
                arguments = self._split_top_level_commas(
                    before_params[angle_start + 1 : angle_end]
                )
                relative_name_end = angle_end + 1
            else:
                # Standard C++ permits an explicit function specialization to
                # omit its template argument list when all arguments can be
                # deduced from the concrete function type.
                name_source = before_params
                arguments = []
                relative_name_end = -1
            name_match = re.search(
                r"([A-Za-z_][A-Za-z0-9_:]*)\s*$",
                name_source,
            )
            if name_match is None:
                continue
            qualified_name = re.sub(r"\s*::\s*", "::", name_match.group(1))
            name_prefix = before_params[: name_match.start(1)].rstrip()
            if "::" in qualified_name or name_prefix.endswith("::"):
                continue
            function_name = qualified_name
            if relative_name_end < 0:
                relative_name_end = name_match.end(1)
            specialization_key = self._template_specialization_key(
                function_name,
                arguments,
            )
            parameter_types = tuple(self._function_parameter_type_texts(header))
            key = (
                (
                    f"{namespace}::{specialization_key[0]}"
                    if namespace
                    else specialization_key[0]
                ),
                tuple(specialization_key[1]),
                parameter_types,
            )
            source_location = self._source_location_for_offsets(
                code,
                match.start(),
                body_end,
            )
            if key in specializations:
                requested_signature = self._template_specialization_signature(
                    function_name,
                    list(specialization_key[1]),
                )
                concrete_signature = (
                    f"{requested_signature}({', '.join(parameter_types)})"
                )
                raise MetalTemplateSpecializationError(
                    "Metal explicit free-function template specialization is "
                    f"defined more than once for '{concrete_signature}'",
                    requested_signature=requested_signature,
                    source_location=source_location,
                    callee_template=function_name,
                    requested_arguments=tuple(specialization_key[1]),
                )

            source = code[declaration_start:body_end]
            relative_name_start = name_match.start(1)
            specializations[key] = {
                "name": function_name,
                "arguments": tuple(specialization_key[1]),
                "templateArgumentsExplicit": bool(arguments),
                "parameterTypes": parameter_types,
                "header": header,
                "span": (match.start(), body_end),
                "sourceLocation": source_location,
                "source": source,
                "nameSpan": (relative_name_start, relative_name_end),
                "namespace": namespace,
            }
        return specializations

    def _materialize_explicit_template_function_specialization(
        self,
        specialization: Dict[str, object],
        function_identifier: str,
    ) -> str:
        """Render one parsed explicit specialization as a concrete function."""
        source = str(specialization["source"])
        name_start, name_end = specialization["nameSpan"]
        materialized = (
            source[: int(name_start)] + function_identifier + source[int(name_end) :]
        )
        if not materialized.endswith("\n"):
            materialized += "\n"
        self._materialized_function_names.add(function_identifier)
        return materialized

    def _find_explicit_template_specialization_keys(
        self, code: str
    ) -> Set[Tuple[str, Tuple[str, ...]]]:
        keys: Set[Tuple[str, Tuple[str, ...]]] = set()
        for match in re.finditer(r"\btemplate\s*<\s*>\s*", code):
            declaration_start = match.end()
            body_start = self._find_next_top_level_char(
                code, declaration_start, "{", stop_at=";"
            )
            semicolon = self._find_next_top_level_char(
                code, declaration_start, ";", stop_at="{"
            )
            declaration_end = body_start
            if declaration_end is None or (
                semicolon is not None and semicolon < declaration_end
            ):
                declaration_end = semicolon
            if declaration_end is None:
                continue
            header = code[declaration_start:declaration_end]
            paren_index = header.find("(")
            if paren_index == -1:
                continue
            before_params = header[:paren_index].rstrip()
            angle_end = before_params.rfind(">")
            if angle_end == -1:
                continue
            angle_start = before_params.rfind("<", 0, angle_end)
            if angle_start == -1:
                continue
            name_match = re.search(
                r"([A-Za-z_][A-Za-z0-9_:]*)\s*$",
                before_params[:angle_start],
            )
            if name_match is None:
                continue
            function_name = name_match.group(1).split("::")[-1]
            args = self._split_top_level_commas(
                before_params[angle_start + 1 : angle_end]
            )
            keys.add(self._template_specialization_key(function_name, args))
        return keys

    def _containing_span(
        self, position: int, spans: List[Tuple[int, int]]
    ) -> Optional[Tuple[int, int]]:
        if not spans:
            return None

        first = spans[0]
        last = spans[-1]
        cache = self._containing_span_cache
        key = id(spans)
        cached = cache.get(key)
        if (
            cached is None
            or cached.spans is not spans
            or cached.span_count != len(spans)
            or cached.first_span != first
            or cached.last_span != last
        ):
            cached = self._build_containing_span_lookup(spans)
            cache[key] = cached
            if len(cache) > CONTAINING_SPAN_CACHE_LIMIT:
                del cache[next(iter(cache))]

        starts = cached.starts
        if starts is not None:
            index = bisect_right(starts, position) - 1
            if index >= 0:
                start, end = spans[index]
                if position < end:
                    return start, end
            return None

        boundaries = cached.boundaries
        containing_spans = cached.containing_spans
        if boundaries is not None and containing_spans is not None:
            boundary_index = bisect_right(boundaries, position) - 1
            if boundary_index >= 0:
                return containing_spans[boundary_index]
        return None

    def _next_scan_stop(
        self, code: str, start: int, *span_lists: Optional[List[Tuple[int, int]]]
    ) -> int:
        """First position at or after ``start`` where a source scanner can act.

        That is a quote, ``/``, an identifier character (any non-ASCII
        character counts) or a boundary of one of ``span_lists``. A scanner that
        only advances one character at every other position may jump here.
        """
        match = SCAN_STOP_RE.search(code, start)
        stop = len(code) if match is None else match.start()
        for spans in span_lists:
            if not spans:
                continue
            boundaries = self._containing_span_boundaries(spans)
            index = bisect_left(boundaries, start)
            if index < len(boundaries) and boundaries[index] < stop:
                stop = boundaries[index]
        return stop

    def _containing_span_boundaries(self, spans: List[Tuple[int, int]]) -> List[int]:
        cache = self._containing_span_cache
        key = id(spans)
        cached = cache.get(key)
        if (
            cached is None
            or cached.spans is not spans
            or cached.span_count != len(spans)
            or cached.first_span != spans[0]
            or cached.last_span != spans[-1]
        ):
            cached = self._build_containing_span_lookup(spans)
            cache[key] = cached
            if len(cache) > CONTAINING_SPAN_CACHE_LIMIT:
                del cache[next(iter(cache))]
        if cached.all_boundaries is None:
            cached.all_boundaries = sorted(
                {position for span in spans for position in span}
            )
        return cached.all_boundaries

    @staticmethod
    def _build_containing_span_lookup(
        spans: List[Tuple[int, int]],
    ) -> _MetalContainingSpanLookup:
        is_sorted_non_overlapping = all(
            spans[index - 1][0] <= spans[index][0]
            and spans[index - 1][1] <= spans[index][0]
            for index in range(1, len(spans))
        )
        if is_sorted_non_overlapping:
            return _MetalContainingSpanLookup(
                spans=spans,
                span_count=len(spans),
                first_span=spans[0],
                last_span=spans[-1],
                starts=[start for start, _end in spans],
                boundaries=None,
                containing_spans=None,
            )

        starts_by_position: Dict[int, List[int]] = {}
        ends_by_position: Dict[int, List[int]] = {}
        for index, (start, end) in enumerate(spans):
            if start >= end:
                continue
            starts_by_position.setdefault(start, []).append(index)
            ends_by_position.setdefault(end, []).append(index)

        active = [False] * len(spans)
        active_by_order: List[int] = []
        boundaries: List[int] = []
        containing_spans: List[Optional[Tuple[int, int]]] = []
        for boundary in sorted(starts_by_position.keys() | ends_by_position.keys()):
            for index in ends_by_position.get(boundary, ()):
                active[index] = False
            for index in starts_by_position.get(boundary, ()):
                active[index] = True
                heappush(active_by_order, index)
            while active_by_order and not active[active_by_order[0]]:
                heappop(active_by_order)
            boundaries.append(boundary)
            containing_spans.append(
                spans[active_by_order[0]] if active_by_order else None
            )

        return _MetalContainingSpanLookup(
            spans=spans,
            span_count=len(spans),
            first_span=spans[0],
            last_span=spans[-1],
            starts=None,
            boundaries=boundaries,
            containing_spans=containing_spans,
        )

    def _scoped_identifier_start(self, code: str, identifier_start: int) -> int:
        start = identifier_start
        while True:
            separator_end = start
            while separator_end > 0 and code[separator_end - 1].isspace():
                separator_end -= 1
            if separator_end < 2 or code[separator_end - 2 : separator_end] != "::":
                break
            name_end = separator_end - 2
            while name_end > 0 and code[name_end - 1].isspace():
                name_end -= 1
            name_start = name_end
            while name_start > 0 and (
                code[name_start - 1].isalnum() or code[name_start - 1] == "_"
            ):
                name_start -= 1
            if (
                name_start == name_end
                or code[name_start:name_end] in NON_SCOPE_KEYWORDS
            ):
                # A global qualifier, as in `return ::max(a, b)`.
                start = separator_end - 2
                break
            start = name_start
        return start

    def _template_specialization_identifier(
        self, function_name: str, template_arguments: List[str]
    ) -> str:
        parts = [function_name, *template_arguments]
        identifier = "_".join(
            part
            for part in (
                re.sub(r"[^A-Za-z0-9_]+", "_", str(value)).strip("_") for value in parts
            )
            if part
        )
        if not identifier:
            identifier = function_name
        if identifier[0].isdigit():
            identifier = f"{function_name}_{identifier}"
        return identifier

    def _canonical_materialized_struct_arguments(
        self,
        arguments: List[str],
        materialized_names: Dict[Tuple[str, Tuple[str, ...]], str],
    ) -> List[str]:
        """Spell each argument naming a materialized struct by its concrete name.

        Only a whole argument of the form ``Name<...>`` whose specialization
        key, with its own arguments canonicalized the same way, is already
        materialized is replaced.
        """
        canonical = []
        for argument in arguments:
            match = re.fullmatch(
                r"\s*([A-Za-z_]\w*)\s*<(.*)>\s*", str(argument), re.DOTALL
            )
            if match is not None:
                inner = self._canonical_materialized_struct_arguments(
                    self._split_top_level_commas(match.group(2)), materialized_names
                )
                concrete = materialized_names.get(
                    self._template_specialization_key(match.group(1), inner)
                )
                if concrete is not None:
                    argument = concrete
            canonical.append(argument)
        return canonical

    def _unique_struct_specialization_identifier(
        self, struct_name: str, key: Tuple[str, Tuple[str, ...]]
    ) -> str:
        """Name ``key``'s concrete struct without taking another key's name.

        The readable identifier drops punctuation, so distinct arguments such
        as ``!flag`` and ``flag`` give the same text. A later key takes the
        first numbered suffix that no other specialization owns.
        """
        preferred = self._template_specialization_identifier(struct_name, list(key[1]))
        owner = (key[0], tuple(key[1]))
        candidate, suffix = preferred, 2
        while self._materialized_struct_specializations.get(candidate, owner) != owner:
            candidate = f"{preferred}_{suffix}"
            suffix += 1
        return candidate

    def _template_specialization_key(
        self, function_name: str, template_arguments: List[str]
    ) -> Tuple[str, Tuple[str, ...]]:
        return (
            function_name,
            tuple(
                self._normalize_template_argument_text(argument)
                for argument in template_arguments
            ),
        )

    def _template_specialization_signature(
        self, function_name: str, template_arguments: List[str]
    ) -> str:
        return (
            f"{function_name}<"
            + ", ".join(
                self._normalize_template_argument_text(argument)
                for argument in template_arguments
            )
            + ">"
        )

    def _normalize_template_argument_text(self, value: str) -> str:
        return normalize_template_argument_text(
            value,
            self._strip_template_argument_comments,
            self._find_matching_delimiter,
        )

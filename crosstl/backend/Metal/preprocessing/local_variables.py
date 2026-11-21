"""Local variable scopes and declarations, struct member call-site rewriting and concrete type aliases."""

import re
from typing import (
    Dict,
    List,
    Optional,
    Set,
    Tuple,
)

from .support import (
    IDENTIFIER_RE,
    SIMPLE_DECLARATION_EVENT_RE,
    _keyword_before,
    _MetalIntegralConstantBinding,
    _MetalStructDefinition,
    _MetalStructMethod,
    _MetalTypeAliasBinding,
)


class MetalLocalVariableMixin:
    """Members of ``MetalPreprocessor``: local variable scopes and declarations, struct member call-site rewriting and concrete type aliases."""

    def _local_variable_shadow_scopes(
        self,
        body: str,
    ) -> Dict[str, List[Tuple[int, int, int]]]:
        shadows: Dict[str, List[Tuple[int, int, int]]] = {}
        lexical_scopes = self._find_lexical_brace_scopes(body)
        search_start = 0
        for statement in self._iter_simple_declarations(body):
            statement_start = body.find(statement, search_start)
            if statement_start == -1:
                continue
            search_start = statement_start + len(statement)
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

            declaration_position = statement_start + leading + name_match.start()
            scope_start, scope_end = self._innermost_lexical_scope(
                lexical_scopes,
                declaration_position,
                len(body),
            )
            shadows.setdefault(name, []).append(
                (declaration_position, scope_start, scope_end)
            )
        return shadows

    def _identifier_is_declaration_or_call(
        self, body: str, start: int, end: int
    ) -> bool:
        # A member name is left untouched when it is used as a function name
        # (`member(...)`) — a data member of struct type is not callable in our
        # model, but a same-named free function might be — to stay conservative.
        j = end
        while j < len(body) and body[j].isspace():
            j += 1
        return j < len(body) and body[j] == "("

    def _local_variable_names(self, body: str) -> Set[str]:
        # Collect identifiers introduced as locals inside a method body so member
        # references shadowed by a local are not rewritten to `self.x`. This is a
        # best-effort scan of `Type name ...;` / `Type name = ...;` declarations
        # at any brace depth, plus simple `for (Type name ...)` headers.
        names: Set[str] = set()
        for statement in self._iter_simple_declarations(body):
            name = self._declared_local_name(statement)
            if name:
                names.add(name)
        return names

    def _iter_simple_declarations(self, body: str) -> List[str]:
        # Yield candidate declaration statements: the text since the previous
        # statement/scope boundary up to each top-level (relative) `;` or the
        # initializer clauses of `for (...)` headers.
        statements: List[str] = []
        i = 0
        n = len(body)
        segment_start = 0
        paren_depth = 0
        # Only these characters change the scanner state; skip everything else.
        find_event = SIMPLE_DECLARATION_EVENT_RE.search
        while i < n:
            event = find_event(body, i)
            if event is None:
                break
            i = event.start()
            ch = body[i]
            if ch in "\"'":
                _literal, consumed = self._read_string(body, i)
                i += consumed
                continue
            if body.startswith("//", i):
                end = body.find("\n", i)
                i = n if end == -1 else end + 1
                segment_start = i
                continue
            if body.startswith("/*", i):
                end = body.find("*/", i + 2)
                i = n if end == -1 else end + 2
                segment_start = i
                continue
            if ch == "(":
                # Capture `for (` initializer declarations specially.
                if _keyword_before(body, segment_start, i, "for") is not None:
                    close = self._find_matching_delimiter(body, i, "(", ")")
                    if close is not None:
                        header = body[i + 1 : close]
                        first_clause = header.split(";", 1)[0]
                        statements.append(first_clause)
                        i = close + 1
                        segment_start = i
                        continue
                paren_depth += 1
                i += 1
                continue
            if ch == ")":
                paren_depth = max(0, paren_depth - 1)
                i += 1
                continue
            if paren_depth == 0 and ch == "{":
                _declarator, initializer = self._split_top_level_assignment(
                    body[segment_start:i]
                )
                if initializer is not None:
                    close = self._find_matching_delimiter(body, i, "{", "}")
                    if close is not None:
                        i = close + 1
                        continue
            if paren_depth == 0 and ch in ";{}":
                statements.append(body[segment_start:i])
                i += 1
                segment_start = i
                continue
            i += 1
        if segment_start < n:
            statements.append(body[segment_start:n])
        return statements

    def _declared_local_name(self, statement: str) -> Optional[str]:
        # Recognize `Type name`, `Type name = ...`, `Type name(...)` where the
        # leading token(s) look like a type. Reject pure expressions/assignments.
        text = statement.strip()
        if not text:
            return None
        # Strip an initializer to isolate the declarator.
        declarator = self._strip_top_level_default_value(text)
        # A call/paren-initialized declarator: `Type name(args)`.
        paren = self._function_parameter_start(declarator)
        if paren is not None:
            declarator = declarator[:paren].rstrip()
        # Drop trailing array extents.
        while declarator.endswith("]"):
            open_bracket = declarator.rfind("[")
            if open_bracket == -1:
                break
            declarator = declarator[:open_bracket].rstrip()
        tokens = IDENTIFIER_RE.findall(declarator)
        # Need at least a type token and a name token; a single token is an
        # expression (e.g. `i++` stripped) rather than a declaration.
        if len(tokens) < 2:
            return None
        # Reject obvious non-declarations (control-flow keywords as the leader).
        if tokens[0] in {"return", "if", "else", "while", "for", "switch", "do"}:
            return None
        name_match = re.search(r"([A-Za-z_][A-Za-z0-9_]*)\s*$", declarator)
        if name_match is None:
            return None
        return name_match.group(1)

    def _declared_local_type(self, statement: str, name: str) -> Optional[str]:
        declarator = self._strip_top_level_default_value(statement.strip())
        paren = self._function_parameter_start(declarator)
        if paren is not None:
            declarator = declarator[:paren].rstrip()
        while declarator.endswith("]"):
            open_bracket = declarator.rfind("[")
            if open_bracket == -1:
                break
            declarator = declarator[:open_bracket].rstrip()
        type_text = re.sub(rf"\b{re.escape(name)}\s*$", "", declarator).strip()
        normalized = self._normalize_inferred_type(type_text)
        return normalized or None

    def _rewrite_struct_member_call_sites(
        self,
        code: str,
        struct_names: Set[str],
        methods_by_struct: Dict[str, Dict[str, _MetalStructMethod]],
        template_methods_by_struct: Dict[str, Dict[str, List[_MetalStructMethod]]],
        operator_call_structs: Set[str],
        struct_spans: List[Tuple[int, int]],
        structs_by_name: Dict[str, _MetalStructDefinition],
        instantiated_template_functions: Dict[str, str],
        field_structs_by_name: Optional[Dict[str, _MetalStructDefinition]] = None,
        all_struct_spans: Optional[List[Tuple[int, int]]] = None,
        all_struct_names: Optional[Set[str]] = None,
    ) -> List[Tuple[int, int, str]]:
        # Rewrite call sites across the source (outside the struct definitions we
        # are replacing):
        #   var.m(args)  -> S__m(var, args)            (instance method)
        #   var(args)    -> S__operator_call(var, ...)  (var has operator())
        #   S::m(args)   -> S__m(args)                  (qualified static call)
        # A template member method call is instantiated from its argument types
        # and rewritten to a concrete free function. Local variable struct types
        # are tracked per the declarations that introduce them; scalar/vector
        # local types and buffer element types feed template-argument inference.
        # All struct/union spans and type names are needed so member-access field
        # resolution sees method-less carriers too; fall back to the method-driven
        # views when the caller does not provide them.
        field_structs_by_name = field_structs_by_name or structs_by_name
        all_struct_spans = (
            all_struct_spans if all_struct_spans is not None else struct_spans
        )
        all_struct_names = (
            all_struct_names if all_struct_names is not None else struct_names
        )
        # A materializer leaves the ORIGINAL template declarations in place after
        # emitting their concrete instantiations. Those residual template bodies
        # still reference unbound template parameters (`U a`, `static_cast<U>(b)`),
        # so their aliases and call sites must not participate in concrete
        # inference. Concrete and template spans are disjoint.
        scan_skip_spans = list(all_struct_spans)
        template_declaration_spans = self._find_template_declaration_spans(code)
        if template_declaration_spans:
            scan_skip_spans = sorted(all_struct_spans + template_declaration_spans)
        scan_skip_spans = sorted(
            scan_skip_spans + self._unreachable_function_spans(code, scan_skip_spans)
        )
        # Member specialization uses the same proven lexical constants as nested
        # struct materialization; unresolved or runtime bindings remain absent.
        local_constant_owner_spans = [
            function.body_span
            for function in self._find_non_template_function_definitions(
                code,
                scan_skip_spans,
            )
        ]
        local_constant_type_aliases = self._collect_local_type_alias_bindings(
            code,
            local_constant_owner_spans,
        )
        local_integral_constants = self._collect_local_integral_constant_bindings(
            code,
            local_constant_owner_spans,
            local_constant_type_aliases,
            self._active_static_constexpr_functions,
            include_file_scope=True,
        )
        struct_type_aliases = self._collect_struct_type_aliases(
            code,
            all_struct_names,
            scan_skip_spans,
            field_structs_by_name,
        )

        variable_types = self._collect_struct_variable_types(
            code, struct_names, scan_skip_spans
        )
        # Struct-typed locals for EVERY struct/union type (used for `obj.member`
        # inference), resolved over the full struct span set so member-access
        # works for data carriers without methods.
        field_variable_types = self._collect_struct_variable_types(
            code,
            all_struct_names,
            scan_skip_spans,
            include_indirect=True,
        )
        aliased_variable_types = self._collect_aliased_struct_variable_types(
            code, struct_type_aliases, scan_skip_spans
        )
        for name, entries in aliased_variable_types.items():
            field_entries = field_variable_types.setdefault(name, [])
            field_entries.extend(
                entry for entry in entries if entry not in field_entries
            )
            field_entries.sort(key=lambda item: item[0])

            method_entries = variable_types.setdefault(name, [])
            method_entries.extend(
                entry
                for entry in entries
                if entry[1] in struct_names and entry not in method_entries
            )
            method_entries.sort(key=lambda item: item[0])
        buffer_element_types = self._collect_buffer_element_types(code, scan_skip_spans)
        local_variable_types = self._collect_local_variable_types(
            code,
            scan_skip_spans,
            self._source_type_alias_bindings,
            list(field_structs_by_name.values()),
        )
        for name, entries in field_variable_types.items():
            local_entries = local_variable_types.setdefault(name, [])
            for entry in entries:
                if entry not in local_entries:
                    local_entries.append(entry)
            local_entries.sort(key=lambda item: item[0])
        # Fold the enclosing functions' parameters into the same position-ordered
        # maps so a call argument that is a parameter (or a subscript of one) is
        # inferable.
        self._collect_function_parameter_types(
            code,
            scan_skip_spans,
            buffer_element_types,
            local_variable_types,
        )
        # Type `auto` locals from their initializer LAST, so the inference can use
        # every buffer/parameter/scalar/struct local collected above (and the
        # tracked struct set for construction / functor-call initializers).
        self._collect_auto_local_variable_types(
            code,
            scan_skip_spans,
            buffer_element_types,
            local_variable_types,
            field_structs_by_name,
            type_aliases=self._source_type_alias_bindings,
            local_integral_constants=local_integral_constants,
        )

        def scan(start: int, limit: int) -> List[Tuple[int, int, str]]:
            replacements: List[Tuple[int, int, str]] = []
            i = start
            while i < limit:
                ch = code[i]
                if ch in "\"'":
                    _literal, consumed = self._read_string(code, i)
                    i += consumed
                    continue
                if code.startswith("//", i):
                    end = code.find("\n", i)
                    i = limit if end == -1 else end + 1
                    continue
                if code.startswith("/*", i):
                    end = code.find("*/", i + 2)
                    i = limit if end == -1 else end + 2
                    continue
                span = self._containing_span(i, scan_skip_spans)
                if span is not None:
                    i = span[1]
                    continue
                if ch.isalpha() or ch == "_":
                    ident, consumed = self._read_identifier(code, i)
                    ident_end = i + consumed
                    rewrite = self._try_rewrite_call_at(
                        code,
                        i,
                        ident,
                        ident_end,
                        struct_names,
                        methods_by_struct,
                        template_methods_by_struct,
                        operator_call_structs,
                        variable_types,
                        structs_by_name,
                        buffer_element_types,
                        local_variable_types,
                        instantiated_template_functions,
                        field_variable_types=field_variable_types,
                        field_structs_by_name=field_structs_by_name,
                        struct_type_aliases=struct_type_aliases,
                        local_integral_constants=local_integral_constants,
                        type_aliases=self._source_type_alias_bindings,
                    )
                    if rewrite is not None and rewrite[0] <= limit:
                        end, replacement = rewrite
                        replacements.append((i, end, replacement))
                        i = end
                        continue
                    i = ident_end
                    continue
                i += 1
            return replacements

        def rewrite_range(start: int, limit: int) -> str:
            nested = scan(start, limit)
            if not nested:
                return code[start:limit]
            return self._apply_text_replacements(
                code[start:limit],
                [(begin - start, end - start, text) for begin, end, text in nested],
            )

        self._call_argument_rewriters.append((code, rewrite_range))
        try:
            return scan(0, len(code))
        finally:
            self._call_argument_rewriters.pop()

    def _rewritten_call_arguments(
        self, code: str, arg_open: int, arg_close: int
    ) -> Optional[str]:
        """Arguments of the call at ``arg_open`` with their own member calls rewritten.

        A template member call is rewritten as a whole once its argument types
        are inferred, so the call-site scan never revisits its arguments. They
        are rewritten here first, in their original lexical scope, so an
        argument such as ``tile.frag_at(0)`` is typed and emitted as the
        lowered expression. ``None`` when no scan of ``code`` is active or
        nothing in the arguments was rewritten.
        """
        if not self._call_argument_rewriters:
            return None
        source, rewrite = self._call_argument_rewriters[-1]
        if source is not code:
            return None
        rewritten = rewrite(arg_open + 1, arg_close)
        return None if rewritten == code[arg_open + 1 : arg_close] else rewritten

    def _conversion_value_type(
        self,
        type_text: str,
        position: int,
        type_aliases: Dict[str, List[_MetalTypeAliasBinding]],
    ) -> Optional[str]:
        if (
            IDENTIFIER_RE.fullmatch(type_text)
            and type_text not in self._METAL_SCALAR_VECTOR_TYPES
            and type_text not in type_aliases
        ):
            return None
        canonical = self._canonicalize_type_aliases_at(
            type_text, type_aliases, position
        )
        if canonical is None:
            return None
        normalized = self._normalize_inferred_type(canonical)
        normalized = re.sub(r"^(?:::)?metal::", "", normalized)
        if normalized in self._METAL_SCALAR_VECTOR_TYPES:
            return normalized
        angle = normalized.find("<")
        if (
            angle < 0
            or normalized[:angle].strip() not in {"vec", "vector"}
            or self._find_matching_angle(normalized, angle) != len(normalized) - 1
        ):
            return None
        arguments = self._split_top_level_commas(normalized[angle + 1 : -1])
        if len(arguments) != 2:
            return None
        element, width = (argument.strip() for argument in arguments)
        if element not in {
            "float",
            "half",
            "bfloat",
            "int",
            "uint",
            "short",
            "ushort",
            "char",
            "uchar",
            "long",
            "ulong",
            "bool",
        } or width not in {"2", "3", "4"}:
            return None
        return f"{element}{width}"

    def _aliased_constructor_type_at(
        self,
        expression: str,
        position: int,
        type_aliases: Dict[str, List[_MetalTypeAliasBinding]],
        local_integral_constants: Dict[str, List[_MetalIntegralConstantBinding]],
    ) -> Optional[str]:
        constructor = re.match(r"(?P<name>[A-Za-z_]\w*)\s*([({])", expression)
        if constructor is None:
            return None
        opening = constructor.end() - 1
        delimiter = expression[opening]
        closing = self._find_matching_delimiter(
            expression, opening, delimiter, ")" if delimiter == "(" else "}"
        )
        if closing != len(expression) - 1:
            return None
        return self._concrete_type_alias_at(
            constructor.group("name"), position, type_aliases, local_integral_constants
        )

    def _concrete_type_alias_at(
        self,
        type_name: str,
        position: int,
        type_aliases: Dict[str, List[_MetalTypeAliasBinding]],
        local_integral_constants: Dict[str, List[_MetalIntegralConstantBinding]],
    ) -> Optional[str]:
        source_type = self._canonicalize_type_aliases_at(
            type_name, type_aliases, position
        )
        conditional = re.sub(r"^typename\s+", "", str(source_type or "").strip())
        type_accessor = conditional.endswith("::type")
        if type_accessor:
            conditional = conditional[: -len("::type")].strip()
        angle_open = conditional.find("<")
        angle_close = (
            self._find_matching_angle(conditional, angle_open)
            if angle_open >= 0
            else None
        )
        conditional_name = re.sub(
            r"^(?:::)?(?:metal::)?", "", conditional[:angle_open]
        ).strip()
        if (
            angle_open > 0
            and angle_close == len(conditional) - 1
            and conditional_name in {"conditional_t", "conditional"}
            and (conditional_name == "conditional_t" or type_accessor)
        ):
            arguments = self._split_top_level_commas(
                conditional[angle_open + 1 : angle_close]
            )
            if len(arguments) == 3:
                visible_constants = self._local_integral_constants_at(
                    local_integral_constants,
                    position,
                    names=set(IDENTIFIER_RE.findall(arguments[0])),
                )
                condition = self._substitute_template_argument_static_constants(
                    arguments[0], visible_constants
                )
                folded, value = self._evaluate_static_integral_expression(condition)
                if folded and value is not None:
                    return self._canonicalize_type_aliases_at(
                        arguments[1 if value else 2], type_aliases, position
                    )
        return source_type

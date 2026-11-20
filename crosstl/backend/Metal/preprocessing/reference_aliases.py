"""Function reference alias bindings, direct reference accessors, contextual receivers and lexical scope scans."""

import re
from bisect import bisect_right
from heapq import (
    heappop,
    heappush,
)
from typing import (
    Callable,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
)

from ..conditional_types import select_conditional_type
from ..forwarding_reference_accessors import (
    forwarded_storage_expression,
    selected_constexpr_body,
)
from ..function_parameter_lists import parameter_list_start
from ..method_arity import named_parameter_arguments
from ..reference_alias_arguments import call_argument_designates
from .support import (
    IDENTIFIER_RE,
    LEXICAL_SCOPE_CACHE_LIMIT,
    METAL_COMMENT_LITERAL_RE,
    METAL_RECEIVER_DECLARATION_RE,
    METAL_SCOPE_DELIMITER_RE,
    MetalStructMethodError,
    _MetalLexicalScopeLookup,
    _MetalReceiverContract,
    _MetalReceiverDeclaration,
    _MetalStructDefinition,
    _MetalStructMethod,
    _MetalTypeAliasBinding,
)


class MetalReferenceAliasMixin:
    """Members of ``MetalPreprocessor``: function reference alias bindings, direct reference accessors, contextual receivers and lexical scope scans."""

    def _rewrite_function_reference_alias_bindings(
        self,
        code: str,
        structs: List[_MetalStructDefinition],
        excluded_spans: List[Tuple[int, int]],
    ) -> str:
        """Bind proven accessor aliases in entry points and free functions.

        Struct method bodies bind theirs while they are lowered; this covers
        the remaining function bodies with the same storage contract.
        """
        if "auto" not in code or "&" not in code:
            return code
        structs_by_name = {struct.name: struct for struct in structs}
        rewrites: List[Tuple[int, int, str]] = []
        masked = self._mask_comments_and_literals(code)
        for function in self._find_non_template_function_definitions(
            code, excluded_spans
        ):
            # Include the braces so the body is its own lexical scope, and the
            # parameter list so parameters are visible receivers.
            end = function.body_span[1] + 1
            start = parameter_list_start(
                masked, function.span[0], function.body_span[0] - 1
            )
            body = code[start:end]
            rewritten = self._rewrite_local_reference_alias_bindings(
                body,
                {},
                structs_by_name,
                const_receivers=set(),
                resolve_type_aliases=True,
            )
            if rewritten != body:
                rewrites.append((start, end, rewritten))
        if not rewrites:
            return code
        return self._apply_text_replacements(code, rewrites)

    def _rewrite_local_reference_alias_bindings(
        self,
        code: str,
        variable_types: Dict[str, List[Tuple[int, str]]],
        structs_by_name: Dict[str, _MetalStructDefinition],
        *,
        const_receivers: Set[str],
        implicit_receiver_contract: Optional[_MetalReceiverContract] = None,
        resolve_type_aliases: bool = False,
    ) -> str:
        """Replace proven local aliases with their captured storage locations."""
        if "auto" not in code or "&" not in code:
            return code
        pattern = re.compile(
            r"\b(?P<qualifiers>(?:(?:thread|const)\s+)+)"
            r"auto\s*&\s*(?P<alias>[A-Za-z_]\w*)\s*=\s*"
        )
        working = code
        while True:
            masked = self._mask_comments_and_literals(working)
            bindings = list(pattern.finditer(masked))
            if not bindings:
                return working
            scopes = self._find_lexical_brace_scopes(working)
            # Local `using` aliases name the receiver types of entry points.
            type_aliases = (
                self._collect_struct_type_aliases(
                    working, set(structs_by_name), [], structs_by_name
                )
                if resolve_type_aliases
                else None
            )
            call_argument_spans = self._potential_call_argument_spans(
                working,
                masked,
                0,
                len(working),
            )
            rewrites: List[Tuple[int, int, str]] = []
            for binding in bindings:
                qualifiers = binding.group("qualifiers").split()
                if qualifiers.count("const") > 1 or qualifiers.count("thread") > 1:
                    continue
                mutable_alias = "const" not in qualifiers
                boundary = max(
                    masked.rfind(token, 0, binding.start()) for token in (";", "{", "}")
                )
                if masked[boundary + 1 : binding.start()].strip():
                    continue
                initializer = self._parse_dotted_reference_alias_initializer(
                    working,
                    masked,
                    binding.end(),
                )
                if initializer is None:
                    continue
                receiver_parts, method_name, arg_open, arg_close, statement_end = (
                    initializer
                )
                if receiver_parts == ["self"]:
                    # Accessors of the implicit object stay with method lowering.
                    continue
                receiver_declaration, receiver_type = self._receiver_struct_type_at(
                    working,
                    receiver_parts[0],
                    binding.start(),
                    set(structs_by_name),
                    type_aliases,
                )
                if receiver_declaration is None:
                    receiver_type = self._resolve_declared_type_at(
                        variable_types, receiver_parts[0], binding.start()
                    )
                receiver_info = self._nested_member_receiver_info(
                    receiver_type,
                    structs_by_name,
                )
                if receiver_info is None or receiver_info[1] != ".":
                    continue
                current_struct = structs_by_name.get(receiver_info[0])
                if current_struct is None:
                    continue
                receiver_contract = (
                    implicit_receiver_contract
                    if receiver_parts[0] == "self"
                    else self._receiver_contract_for_named_value(
                        working,
                        current_struct.name,
                        receiver_parts[0],
                        binding.start(),
                        type_aliases=type_aliases,
                    )
                )
                if receiver_contract is None:
                    continue
                if (
                    receiver_contract.address_spaces != ("thread",)
                    or receiver_contract.is_volatile
                ):
                    continue
                # A const alias of a writable receiver may observe later writes
                # and index changes, so it captures its storage at the binding
                # exactly like a mutable alias, without accepting writes.
                captures_storage = mutable_alias or (
                    receiver_parts[0] not in const_receivers
                    and not receiver_contract.is_readonly
                )

                value_chain_is_valid = True
                for member_name in receiver_parts[1:]:
                    member = next(
                        (
                            item
                            for item in current_struct.data_members
                            if item.name == member_name
                        ),
                        None,
                    )
                    if (
                        member is None
                        or member.array_suffix
                        or "*" in member.type_text
                        or "&" in member.type_text
                    ):
                        value_chain_is_valid = False
                        break
                    member_type = self._normalize_inferred_type(member.type_text)
                    receiver_contract = self._member_receiver_contract(
                        receiver_contract, current_struct, member_name, member_type
                    )
                    current_struct = structs_by_name.get(member_type)
                    if current_struct is None:
                        value_chain_is_valid = False
                        break
                if not value_chain_is_valid:
                    continue
                if receiver_contract is None or receiver_contract.is_volatile:
                    continue

                reference_methods = self._concrete_reference_methods(
                    current_struct,
                    method_name,
                    working,
                    arg_open,
                )
                if not reference_methods:
                    continue
                method = self._select_direct_reference_method(
                    working,
                    current_struct,
                    reference_methods,
                    arg_open,
                    receiver_is_const=receiver_contract.is_readonly,
                )
                declarations = ""
                if captures_storage:
                    captured = self._capture_reference_accessor_storage(
                        working,
                        current_struct,
                        method,
                        ".".join(receiver_parts),
                        arg_open,
                        structs_by_name,
                    )
                    if captured is None:
                        continue
                    declarations, storage = captured
                else:
                    _call_end, storage = self._direct_reference_accessor_rewrite(
                        working,
                        current_struct,
                        method,
                        receiver=".".join(receiver_parts),
                        arg_open=arg_open,
                    )

                scope_start, scope_end = self._innermost_lexical_scope(
                    scopes,
                    binding.start(),
                    len(working),
                )
                if not (scope_start <= binding.start() < statement_end <= scope_end):
                    continue
                alias_replacements = self._reference_alias_use_replacements(
                    working,
                    masked,
                    binding.group("alias"),
                    statement_end,
                    scope_end,
                    storage,
                    allow_writes=mutable_alias and not receiver_contract.is_readonly,
                )
                if captures_storage:
                    captured_names = {receiver_parts[0]}
                    if any(
                        self._declared_local_name(statement) in captured_names
                        for statement in self._iter_simple_declarations(
                            working[statement_end:scope_end]
                        )
                    ):
                        continue
                if alias_replacements is None or (
                    not captures_storage
                    and not self._reference_alias_capture_is_stable(
                        working,
                        masked,
                        call_argument_spans,
                        scope_start,
                        statement_end,
                        scope_end,
                        storage,
                        {
                            match.group(0)
                            for argument in self._split_top_level_commas(
                                working[arg_open + 1 : arg_close]
                            )
                            for match in re.finditer(r"\b[A-Za-z_]\w*\b", argument)
                            if not self._is_member_identifier_context(
                                argument, match.start()
                            )
                        },
                    )
                ):
                    continue
                erased_binding = "".join(
                    "\n" if char == "\n" else " "
                    for char in working[binding.start() : statement_end]
                )
                rewrites.extend(
                    [
                        (binding.start(), statement_end, declarations + erased_binding),
                        *alias_replacements,
                    ]
                )
                # Process one binding at a time; a later alias may use this one.
                break
            if not rewrites:
                return working
            working = self._apply_text_replacements(working, rewrites)

    def _parse_dotted_reference_alias_initializer(
        self,
        code: str,
        masked: str,
        start: int,
    ) -> Optional[Tuple[List[str], str, int, int, int]]:
        parts: List[str] = []
        cursor = start
        arg_open = -1
        while True:
            while cursor < len(masked) and masked[cursor].isspace():
                cursor += 1
            part, consumed = self._read_identifier(masked, cursor)
            if not part:
                return None
            parts.append(part)
            cursor += consumed
            while cursor < len(masked) and masked[cursor].isspace():
                cursor += 1
            if cursor < len(masked) and masked[cursor] == ".":
                cursor += 1
                continue
            if cursor < len(masked) and masked[cursor] == "(":
                arg_open = cursor
                break
            return None
        # `receiver.accessor(...)`: the receiver is a named value, optionally
        # reached through a chain of value members.
        if len(parts) < 2:
            return None
        arg_close = self._find_matching_delimiter(code, arg_open, "(", ")")
        if arg_close is None:
            return None
        statement_end = arg_close + 1
        while statement_end < len(masked) and masked[statement_end].isspace():
            statement_end += 1
        if statement_end >= len(masked) or masked[statement_end] != ";":
            return None
        return parts[:-1], parts[-1], arg_open, arg_close, statement_end + 1

    def _reference_alias_use_replacements(
        self,
        code: str,
        masked: str,
        alias: str,
        lifetime_start: int,
        lifetime_end: int,
        storage: str,
        *,
        allow_writes: bool = False,
    ) -> Optional[List[Tuple[int, int, str]]]:
        lifetime = code[lifetime_start:lifetime_end]
        if any(
            self._declared_local_name(statement) == alias
            for statement in self._iter_simple_declarations(lifetime)
        ):
            return None
        replacements: List[Tuple[int, int, str]] = []
        alias_pattern = re.compile(rf"\b{re.escape(alias)}\b")
        for match in alias_pattern.finditer(masked, lifetime_start, lifetime_end):
            if self._is_member_identifier_context(masked, match.start()):
                continue
            cursor = match.end()
            while cursor < lifetime_end and masked[cursor].isspace():
                cursor += 1
            if cursor >= lifetime_end or masked[cursor] != "[":
                return None
            while cursor < lifetime_end and masked[cursor] == "[":
                close = self._find_matching_delimiter(code, cursor, "[", "]")
                if close is None or close >= lifetime_end:
                    return None
                cursor = close + 1
                while cursor < lifetime_end and masked[cursor].isspace():
                    cursor += 1
            previous = match.start() - 1
            while previous >= lifetime_start and masked[previous].isspace():
                previous -= 1
            if previous >= lifetime_start and masked[previous] == "&":
                return None
            if not allow_writes and masked[
                max(lifetime_start, previous - 1) : previous + 1
            ] in {"++", "--"}:
                return None
            statement_boundary = max(
                masked.rfind(token, lifetime_start, match.start())
                for token in (";", "{", "}")
            )
            prefix = masked[statement_boundary + 1 : match.start()]
            if re.search(r"&\s*[A-Za-z_]\w*\s*=\s*$", prefix):
                return None
            if not allow_writes and re.match(
                r"(?:\+\+|--|(?:(?:<<|>>|[+\-*/%&|^])?=(?!=)))",
                masked[cursor:],
            ):
                return None
            replacements.append((match.start(), match.end(), storage))
        return replacements or None

    def _capture_reference_accessor_storage(
        self,
        code: str,
        struct: _MetalStructDefinition,
        method: _MetalStructMethod,
        receiver: str,
        arg_open: int,
        structs_by_name: Dict[str, _MetalStructDefinition],
    ) -> Optional[Tuple[str, str]]:
        """Capture parameter conversions and index evaluation at the binding."""
        arg_close = self._find_matching_delimiter(code, arg_open, "(", ")")
        if arg_close is None:
            return None
        arguments = self._split_top_level_commas(code[arg_open + 1 : arg_close])
        parameters = self._split_top_level_commas(method.parameters)
        if not code[arg_open + 1 : arg_close].strip():
            arguments = []
        if not method.parameters.strip():
            parameters = []
        if len(arguments) != len(parameters):
            return None
        names = set(IDENTIFIER_RE.findall(code))
        declarations = []

        def fresh_name(kind: str) -> str:
            index = 0
            while f"crosstl_reference_{kind}_{index}" in names:
                index += 1
            name = f"crosstl_reference_{kind}_{index}"
            names.add(name)
            return name

        captured_arguments = []
        for parameter, argument in zip(parameters, arguments):
            if "*" in parameter or "&" in parameter or "[" in parameter:
                return None
            value_type = self._function_parameter_value_type(parameter)
            if value_type is None:
                return None
            value_type = self._canonicalize_struct_scoped_type(
                value_type, struct, structs_by_name
            )
            scalar = self._scalar_and_width(value_type)
            if (
                scalar is None
                or scalar[1] != 1
                or self._is_integral_concrete_type(value_type) is not True
            ):
                return None
            name = fresh_name("argument")
            declarations.append(f"{value_type} {name} = ({argument});")
            captured_arguments.append(name)
        _, storage = self._direct_reference_accessor_rewrite(
            code,
            struct,
            method,
            receiver=receiver,
            arg_open=arg_open,
            argument_values=captured_arguments,
        )
        replacements = []
        cursor = storage.find("[")
        while cursor >= 0:
            end = self._find_matching_delimiter(storage, cursor, "[", "]")
            if end is None:
                return None
            name = fresh_name("index")
            declarations.append(f"const auto {name} = ({storage[cursor + 1:end]});")
            replacements.append((cursor + 1, end, name))
            cursor = storage.find("[", end + 1)
        return "\n".join(declarations) + "\n", self._apply_text_replacements(
            storage, replacements
        )

    def _reference_alias_capture_is_stable(
        self,
        code: str,
        masked: str,
        call_argument_spans: List[Tuple[int, int]],
        capture_scope_start: int,
        lifetime_start: int,
        lifetime_end: int,
        storage: str,
        argument_captures: Set[str],
    ) -> bool:
        captured = {
            match.group(0)
            for match in re.finditer(r"\b[A-Za-z_]\w*\b", storage)
            if not self._is_member_identifier_context(storage, match.start())
        }
        lifetime = code[lifetime_start:lifetime_end]
        if any(
            self._declared_local_name(statement) in captured
            for statement in self._iter_simple_declarations(lifetime)
        ):
            return False
        for name in captured:
            name_pattern = re.compile(rf"\b{re.escape(name)}\b")
            for match in name_pattern.finditer(
                masked,
                capture_scope_start,
                lifetime_end,
            ):
                if self._is_member_identifier_context(masked, match.start()):
                    continue
                previous = match.start() - 1
                while previous >= capture_scope_start and masked[previous].isspace():
                    previous -= 1
                statement_boundary = max(
                    masked.rfind(token, capture_scope_start, match.start())
                    for token in (";", "{", "}")
                )
                prefix = masked[statement_boundary + 1 : match.start()]
                reference_escape = (
                    previous >= capture_scope_start and masked[previous] == "&"
                ) or re.search(r"&\s*[A-Za-z_]\w*\s*=\s*$", prefix)
                if reference_escape and (
                    match.start() >= lifetime_start or name in argument_captures
                ):
                    return False
                if match.start() < lifetime_start:
                    continue
                if previous >= lifetime_start - 1 and masked[
                    max(lifetime_start, previous - 1) : previous + 1
                ] in {"++", "--"}:
                    return False
                if name in argument_captures and any(
                    call_argument_designates(
                        masked,
                        (start, end),
                        match.start(),
                        match.end(),
                        lambda access_start, limit: self._reference_capture_access_end(
                            code, masked, access_start, limit
                        ),
                    )
                    for start, end in call_argument_spans
                    if lifetime_start <= start < match.start() < end <= lifetime_end
                ):
                    return False
                cursor = self._reference_capture_access_end(
                    code,
                    masked,
                    match.end(),
                    lifetime_end,
                )
                if re.match(
                    r"(?:\+\+|--|(?:(?:<<|>>|[+\-*/%&|^])?=(?!=)))",
                    masked[cursor:lifetime_end],
                ):
                    return False
        return True

    def _reference_capture_access_end(
        self,
        code: str,
        masked: str,
        start: int,
        limit: int,
    ) -> int:
        cursor = start
        while cursor < limit:
            while cursor < limit and masked[cursor].isspace():
                cursor += 1
            if masked.startswith("->", cursor):
                cursor += 2
            elif masked[cursor : cursor + 1] == ".":
                cursor += 1
            elif masked[cursor : cursor + 1] == "[":
                close = self._find_matching_delimiter(code, cursor, "[", "]")
                if close is None or close >= limit:
                    return cursor
                cursor = close + 1
                continue
            else:
                break
            while cursor < limit and masked[cursor].isspace():
                cursor += 1
            _member, consumed = self._read_identifier(masked, cursor)
            if not consumed:
                break
            cursor += consumed
        while cursor < limit and masked[cursor].isspace():
            cursor += 1
        return cursor

    def _potential_call_argument_spans(
        self,
        code: str,
        masked: str,
        start: int,
        end: int,
    ) -> List[Tuple[int, int]]:
        pattern = re.compile(
            r"\b(?P<callee>[A-Za-z_]\w*" r"(?:\s*(?:\.|->|::)\s*[A-Za-z_]\w*)*)\s*\("
        )
        spans: List[Tuple[int, int]] = []
        control_names = {"if", "for", "while", "switch", "sizeof", "decltype"}
        for match in pattern.finditer(masked, start, end):
            callee_names = IDENTIFIER_RE.findall(match.group("callee"))
            if not callee_names or callee_names[-1] in control_names:
                continue
            open_paren = match.end() - 1
            close_paren = self._find_matching_delimiter(code, open_paren, "(", ")")
            if close_paren is not None and close_paren < end:
                spans.append((open_paren, close_paren))
        return spans

    def _mask_comments_and_literals(self, code: str) -> str:
        masked = list(code)
        for start, end in self._find_comment_and_literal_spans(code):
            for index in range(start, end):
                if masked[index] != "\n":
                    masked[index] = " "
        return "".join(masked)

    def _build_direct_reference_accessor_rewrite(
        self,
        code: str,
        struct: _MetalStructDefinition,
        method: _MetalStructMethod,
        *,
        receiver: str,
        arg_open: int,
        call_start: int,
        enclosing_return_type: Optional[str] = None,
    ) -> Tuple[int, str]:
        arg_close = self._find_matching_delimiter(code, arg_open, "(", ")")
        if arg_close is None:
            self._reject_reference_returning_method_call(
                code, struct, method, arg_open, arg_open
            )
            raise AssertionError("reference-return rejection must raise")

        if self._reference_call_binds_or_escapes(
            code,
            call_start,
            enclosing_return_type=enclosing_return_type,
        ):
            self._reject_reference_returning_method_call(
                code, struct, method, arg_open, arg_close
            )

        return self._direct_reference_accessor_rewrite(
            code,
            struct,
            method,
            receiver=receiver,
            arg_open=arg_open,
        )

    def _direct_reference_accessor_rewrite(
        self,
        code: str,
        struct: _MetalStructDefinition,
        method: _MetalStructMethod,
        *,
        receiver: str,
        arg_open: int,
        argument_values: Optional[List[str]] = None,
        depth: int = 0,
    ) -> Tuple[int, str]:
        arg_close = self._find_matching_delimiter(code, arg_open, "(", ")")
        if arg_close is None:
            self._reject_reference_returning_method_call(
                code, struct, method, arg_open, arg_open
            )
            raise AssertionError("reference-return rejection must raise")

        expression = self._direct_reference_return_expression(struct, method, depth)
        if expression is None:
            self._reject_reference_returning_method_call(
                code, struct, method, arg_open, arg_close
            )

        match = re.fullmatch(
            r"self\s*\.\s*(?P<member>[A-Za-z_]\w*)"
            r"(?P<indices>(?:\s*\[[^\[\]]+\])*)",
            expression,
        )
        if match is None:
            self._reject_reference_returning_method_call(
                code, struct, method, arg_open, arg_close
            )

        member = next(
            (
                item
                for item in struct.data_members
                if item.name == match.group("member")
            ),
            None,
        )
        member_qualifiers = (
            set(IDENTIFIER_RE.findall(member.type_text))
            if member is not None
            else set()
        )
        indices = re.findall(r"\[\s*([^\[\]]+)\s*\]", match.group("indices"))
        if (
            member is None
            or "*" in member.type_text
            or "&" in member.type_text
            or "static" in member_qualifiers
            or (member.array_suffix and not indices)
            or (not member.array_suffix and indices)
            or len(indices) != member.array_suffix.count("[")
            or any(
                not self._direct_reference_expression_is_safe(index)
                for index in indices
            )
        ):
            self._reject_reference_returning_method_call(
                code, struct, method, arg_open, arg_close
            )

        raw_arguments = [
            argument.strip()
            for argument in self._split_top_level_commas(code[arg_open + 1 : arg_close])
            if argument.strip()
        ]
        if argument_values is not None:
            raw_arguments = argument_values
        elif len(raw_arguments) != len(method.parameter_names):
            # Arguments for unnamed tag parameters are not read by the accessor (method_arity.py).
            raw_arguments = (
                named_parameter_arguments(
                    method.parameters,
                    raw_arguments,
                    self._split_top_level_commas,
                    self._declared_data_member_name,
                )
                or raw_arguments
            )
        if len(raw_arguments) != len(method.parameter_names):
            self._reject_reference_returning_method_call(
                code, struct, method, arg_open, arg_close
            )

        replacements = {"self": receiver}
        for name, argument in zip(method.parameter_names, raw_arguments):
            if self._bare_identifier_use_count(
                expression, name
            ) != 1 or not self._direct_reference_expression_is_safe(argument):
                self._reject_reference_returning_method_call(
                    code, struct, method, arg_open, arg_close
                )
            replacements[name] = f"({argument})"

        return arg_close + 1, self._replace_identifiers(expression, replacements)

    def _direct_reference_return_expression(
        self,
        struct: _MetalStructDefinition,
        method: _MetalStructMethod,
        depth: int = 0,
    ) -> Optional[str]:
        # An instantiated `if constexpr (true) { return at(j, i); } else ...` executes one return.
        body = selected_constexpr_body(self, self._rewrite_method_body(struct, method))
        match = re.fullmatch(
            r"\s*return\s+(?P<expression>.+?)\s*;\s*",
            body,
            re.DOTALL,
        )
        if match is None:
            return None
        expression = self._strip_enclosing_parens(match.group("expression").strip())
        # `return at(j, i);` forwards another accessor's storage (forwarding_reference_accessors.py).
        forwarded = forwarded_storage_expression(
            self, struct, method, expression, depth
        )
        return expression if forwarded is None else forwarded

    @staticmethod
    def _direct_reference_expression_is_safe(expression: str) -> bool:
        text = expression.strip()
        if not text or re.search(r"\+\+|--|&&|\|\||[=?:,;{}\[\]]", text):
            return False
        if re.search(r"\b[A-Za-z_]\w*\s*\(", text):
            return False
        return re.fullmatch(r"[A-Za-z0-9_'\s.()+\-*/%&|^<>]+", text) is not None

    def _bare_identifier_use_count(self, text: str, name: str) -> int:
        return sum(
            1
            for match in re.finditer(rf"\b{re.escape(name)}\b", text)
            if not self._is_member_identifier_context(text, match.start())
        )

    def _reference_call_binds_or_escapes(
        self,
        code: str,
        call_start: int,
        *,
        enclosing_return_type: Optional[str],
    ) -> bool:
        boundary = max(code.rfind(token, 0, call_start) for token in (";", "{", "}"))
        prefix = code[boundary + 1 : call_start]
        if re.search(r"&\s*[A-Za-z_]\w*\s*=\s*$", prefix):
            return True
        if re.search(r"\breturn\s*$", prefix) is None:
            return False
        if enclosing_return_type is None:
            return True
        return bool(
            re.search(
                r"(?<!&)\&\s*$",
                self._normalize_template_argument_text(enclosing_return_type),
            )
        )

    def _build_instance_call_rewrite(
        self,
        code: str,
        receiver: str,
        method: _MetalStructMethod,
        arg_open: int,
        *,
        struct_name: Optional[object] = None,
    ) -> Optional[Tuple[int, str]]:
        arg_close = self._find_matching_delimiter(code, arg_open, "(", ")")
        if arg_close is None:
            return None
        self._reject_readonly_pointer_result_write(
            code,
            struct_name,
            method,
            arg_open,
            arg_close,
        )
        self._reject_reference_returning_method_call(
            code,
            struct_name,
            method,
            arg_open,
            arg_close,
        )
        args = code[arg_open + 1 : arg_close].strip()
        # Leave arguments in the scanner's input so nested member/functor calls
        # are rewritten in their original lexical scope as well.
        separator = ", " if args else ""
        return arg_open + 1, f"{method.free_name}({receiver}{separator}"

    def _reject_readonly_pointer_result_write(
        self,
        code: str,
        struct_name: Optional[object],
        method: _MetalStructMethod,
        arg_open: int,
        arg_close: int,
    ) -> None:
        struct = struct_name or method.free_name.split("__", 1)[0]
        owner = getattr(struct, "name", None) or str(struct)
        return_type = method.return_type
        if isinstance(struct, _MetalStructDefinition):
            return_type = self._canonicalize_struct_scoped_type(
                return_type,
                struct,
                None,
            )
        normalized_return = self._normalize_template_argument_text(return_type)
        if "*" not in normalized_return:
            return
        pointee_contract = normalized_return.split("*", 1)[0]
        if re.search(r"\bconst\b", pointee_contract) is None:
            return

        cursor = arg_close + 1
        while True:
            while cursor < len(code) and code[cursor].isspace():
                cursor += 1
            if cursor < len(code) and code[cursor] == "[":
                close = self._find_matching_delimiter(code, cursor, "[", "]")
                if close is None:
                    return
                cursor = close + 1
                continue
            if code.startswith("->", cursor):
                cursor += 2
            elif cursor < len(code) and code[cursor] == ".":
                cursor += 1
            else:
                break
            while cursor < len(code) and code[cursor].isspace():
                cursor += 1
            _member, consumed = self._read_identifier(code, cursor)
            if not consumed:
                return
            cursor += consumed

        suffix = code[cursor:].lstrip()
        call_start = code.rfind(method.name, 0, arg_open)
        if call_start < 0:
            call_start = arg_open
        prefix = code[:call_start].rstrip()
        writes_result = suffix.startswith(("++", "--")) or bool(
            re.match(r"^(?:(?:<<|>>|[+\-*/%&|^])?=)(?!=)", suffix)
        )
        writes_result = writes_result or prefix.endswith(("++", "--"))
        if not writes_result:
            return

        arguments = code[arg_open + 1 : arg_close].strip()
        signature = f"{owner}::{method.name}({arguments})"
        receiver_spaces = method.receiver_address_spaces or ("thread",)
        receiver_parts = []
        if method.is_const:
            receiver_parts.append("const")
        if method.is_volatile:
            receiver_parts.append("volatile")
        receiver_parts.extend(receiver_spaces)
        receiver_parts.append(f"{owner}&")
        receiver_type = " ".join(receiver_parts)
        raise MetalStructMethodError(
            "Cannot write through read-only pointer result from member "
            f"'{owner}::{method.name}'. Requested call: {signature}.",
            struct_name=owner,
            method_name=method.name,
            requested_signature=signature,
            suggested_action=(
                "call a source overload returning a writable pointer or keep the "
                "operation read-only"
            ),
            source_location=self._source_location_for_offsets(
                code,
                call_start,
                cursor,
            ),
            missing_capabilities=("metal.readonly-member-pointer",),
            reason="readonly-member-pointer-write",
            return_type=normalized_return,
            receiver_type=receiver_type,
            candidate_signatures=(self._concrete_method_candidate_signature(method),),
        )

    def _reject_reference_returning_method_call(
        self,
        code: str,
        struct_name: Optional[object],
        method: _MetalStructMethod,
        arg_open: int,
        arg_close: int,
    ) -> None:
        struct = struct_name or method.free_name.split("__", 1)[0]
        owner = getattr(struct, "name", None) or str(struct)
        return_type = method.return_type
        if isinstance(struct, _MetalStructDefinition):
            return_type = self._canonicalize_struct_scoped_type(
                return_type,
                struct,
                None,
            )
        arguments = code[arg_open + 1 : arg_close].strip()
        signature = f"{owner}::{method.name}({arguments})"
        call_start = code.rfind(method.name, 0, arg_open)
        if call_start < 0:
            call_start = arg_open
        # Most calls return values, and the location costs a count over the whole unit, so it is computed only
        # once the call is known to be refused.
        self._reject_selected_reference_return(
            struct,
            method,
            return_type,
            signature,
            source_location_factory=lambda: self._source_location_for_offsets(
                code,
                call_start,
                arg_close + 1,
            ),
        )

    def _reject_selected_reference_return(
        self,
        struct: object,
        method: _MetalStructMethod,
        return_type: str,
        signature: str,
        *,
        source_location: Optional[object] = None,
        source_location_factory: Optional[Callable[[], object]] = None,
    ) -> None:
        normalized_return_type = self._normalize_template_argument_text(return_type)
        if (
            re.search(r"&&?\s*$", normalized_return_type) is None
            and normalized_return_type != "decltype(auto)"
        ):
            return

        if source_location is None and source_location_factory is not None:
            source_location = source_location_factory()
        owner = getattr(struct, "name", None) or str(struct)
        raise MetalStructMethodError(
            "Cannot lower reference-returning struct method "
            f"'{owner}::{method.name}' without preserving the returned lvalue "
            f"identity. Requested call: {signature}.",
            struct_name=owner,
            method_name=method.name,
            requested_signature=signature,
            suggested_action=(
                "rewrite the accessor as a direct operation on the underlying "
                "storage, or avoid binding or assigning through its returned "
                "reference until reference identity lowering is available"
            ),
            source_location=source_location,
            missing_capabilities=("struct.reference-return",),
            reason="reference-return-identity-unsupported",
            return_type=normalized_return_type,
        )

    # ------------------------------------------------------------------ #
    # Pointer-member promotion: construction and call-site rewriting.     #
    # ------------------------------------------------------------------ #

    def _collect_struct_type_aliases(
        self,
        code: str,
        struct_names: Set[str],
        skip_spans: List[Tuple[int, int]],
        structs_by_name: Optional[Dict[str, _MetalStructDefinition]] = None,
    ) -> Dict[str, List[_MetalTypeAliasBinding]]:
        # Map aliases to concrete structs with their lexical lifetime. Bindings
        # are position-ordered because kernels commonly reuse the same alias in
        # separate functions or nested blocks. Both `using ALIAS = Struct;` and
        # `typedef Struct ALIAS;` are recognized, including chains through an
        # already visible alias.
        raw_aliases: List[Tuple[int, str, str]] = []
        ignored_spans = sorted(
            [*skip_spans, *self._find_comment_and_literal_spans(code)]
        )
        for match in re.finditer(
            r"\busing\s+(?P<alias>[A-Za-z_]\w*)\s*=\s*" r"(?P<target>[^;{}]+?)\s*;",
            code,
            re.DOTALL,
        ):
            if self._containing_span(match.start(), ignored_spans) is not None:
                continue
            alias, target = match.group("alias"), match.group("target")
            raw_aliases.append(
                (match.start(), alias, self._normalize_template_argument_text(target))
            )
        for match in re.finditer(
            r"\btypedef\s+([A-Za-z_]\w*)\s+([A-Za-z_]\w*)\s*;", code
        ):
            if self._containing_span(match.start(), ignored_spans) is not None:
                continue
            target, alias = match.group(1), match.group(2)
            raw_aliases.append((match.start(), alias, target))

        aliases: Dict[str, List[_MetalTypeAliasBinding]] = {}
        qualified_struct_names = {
            struct.qualified_name.lstrip(":"): struct.name
            for struct in (structs_by_name or {}).values()
            if struct.qualified_name
        }
        lexical_scopes = self._find_lexical_brace_scopes(code)
        for declaration_position, alias, target in sorted(raw_aliases):
            resolved_target = self._resolve_struct_type_alias_at(
                aliases, target, declaration_position
            )
            if resolved_target is None and target in struct_names:
                resolved_target = target
            if resolved_target is None:
                resolved_target = qualified_struct_names.get(target.lstrip(":"))
            if resolved_target is None and structs_by_name:
                qualified = re.fullmatch(
                    r"(?:typename\s+)?(?P<owner>[A-Za-z_]\w*)\s*::\s*"
                    r"(?P<member>[A-Za-z_]\w*)",
                    target,
                )
                if qualified is not None:
                    owner = qualified.group("owner")
                    resolved_owner = self._resolve_struct_type_alias_at(
                        aliases, owner, declaration_position
                    )
                    if resolved_owner is None and owner in struct_names:
                        resolved_owner = owner
                    owner_struct = structs_by_name.get(resolved_owner or "")
                    if owner_struct is not None:
                        canonical = self._canonicalize_struct_scoped_type(
                            qualified.group("member"),
                            owner_struct,
                            structs_by_name,
                        )
                        if IDENTIFIER_RE.fullmatch(canonical) and (
                            canonical in struct_names
                            or canonical in self._materialized_struct_specializations
                        ):
                            resolved_target = canonical
            if resolved_target is None:
                # `using L = conditional_t<C, A, B>;` names A or B once C folds.
                selected = select_conditional_type(
                    target, self._evaluate_static_integral_expression
                )
                if selected is not None:
                    resolved_target = self._resolve_struct_type_alias_at(
                        aliases, selected, declaration_position
                    )
                    if resolved_target is None and selected in struct_names:
                        resolved_target = selected
                    if resolved_target is None:
                        resolved_target = qualified_struct_names.get(
                            selected.lstrip(":")
                        )
            scope_start, scope_end = self._innermost_lexical_scope(
                lexical_scopes, declaration_position, len(code)
            )
            aliases.setdefault(alias, []).append(
                _MetalTypeAliasBinding(
                    declaration_position=declaration_position,
                    scope_start=scope_start,
                    scope_end=scope_end,
                    target=resolved_target,
                )
            )
        return aliases

    def _find_receiver_declarations(
        self, code: str
    ) -> Dict[str, List[_MetalReceiverDeclaration]]:
        analysis = self._source_analysis(code)
        if analysis.receiver_declarations is None:
            analysis.receiver_declarations = self._scan_receiver_declarations(code)
        return analysis.receiver_declarations

    def _register_contextual_receiver_parameters(
        self,
        code: str,
        parameters: str,
    ) -> Set[str]:
        declarations = self._scan_receiver_declarations(code)
        contextual_names: Set[str] = set()
        for parameter in self._split_top_level_commas(parameters):
            text = self._strip_metal_attributes(
                self._strip_top_level_default_value(parameter)
            ).strip()
            if not text:
                continue
            match = METAL_RECEIVER_DECLARATION_RE.search(f"{text},")
            if match is None or match.start() != 0:
                continue
            raw_type_text = "".join(
                (
                    match.group("leading") or "",
                    match.group("type"),
                    match.group("trailing") or "",
                    match.group("indirection") or "",
                    match.group("indirection_qualifiers") or "",
                )
            )
            name = match.group("name")
            contextual_names.add(name)
            declarations.setdefault(name, []).insert(
                0,
                _MetalReceiverDeclaration(
                    name=name,
                    declaration_position=-1,
                    declaration_end=0,
                    type_position=0,
                    scope_start=0,
                    scope_end=len(code),
                    declared_type=match.group("type"),
                    raw_type_text=raw_type_text,
                ),
            )
        self._source_analysis(code).receiver_declarations = declarations
        return contextual_names

    def _scan_receiver_declarations(
        self, code: str
    ) -> Dict[str, List[_MetalReceiverDeclaration]]:
        ignored_spans = self._find_comment_and_literal_spans(code)
        ordered_scopes = sorted(
            self._find_lexical_brace_scopes(code),
            key=lambda scope: (scope[0], -scope[1]),
        )
        declarations: Dict[str, List[_MetalReceiverDeclaration]] = {}
        active_scopes: List[Tuple[int, int]] = []
        scope_index = 0

        for match in METAL_RECEIVER_DECLARATION_RE.finditer(code):
            position = match.start()
            while (
                scope_index < len(ordered_scopes)
                and ordered_scopes[scope_index][0] <= position
            ):
                scope = ordered_scopes[scope_index]
                while active_scopes and not (
                    active_scopes[-1][0] <= scope[0] < active_scopes[-1][1]
                ):
                    active_scopes.pop()
                active_scopes.append(scope)
                scope_index += 1
            while active_scopes and not (
                active_scopes[-1][0] <= position < active_scopes[-1][1]
            ):
                active_scopes.pop()
            if self._containing_span(position, ignored_spans) is not None:
                continue

            scope_start, scope_end = (
                active_scopes[-1] if active_scopes else (0, len(code))
            )
            raw_type_text = "".join(
                (
                    match.group("leading") or "",
                    match.group("type"),
                    match.group("trailing") or "",
                    match.group("indirection") or "",
                    match.group("indirection_qualifiers") or "",
                )
            )
            name = match.group("name")
            declarations.setdefault(name, []).append(
                _MetalReceiverDeclaration(
                    name=name,
                    declaration_position=position,
                    declaration_end=match.end(),
                    type_position=match.start("type"),
                    scope_start=scope_start,
                    scope_end=scope_end,
                    declared_type=match.group("type"),
                    raw_type_text=raw_type_text,
                )
            )
        return declarations

    def _find_comment_and_literal_spans(self, code: str) -> List[Tuple[int, int]]:
        analysis = self._source_analysis(code)
        if analysis.comment_and_literal_spans is None:
            analysis.comment_and_literal_spans = self._scan_comment_and_literal_spans(
                code
            )
        return analysis.comment_and_literal_spans

    def _scan_comment_and_literal_spans(self, code: str) -> List[Tuple[int, int]]:
        spans: List[Tuple[int, int]] = []
        i = 0
        while i < len(code):
            match = METAL_COMMENT_LITERAL_RE.search(code, i)
            if match is None:
                break
            i = match.start()
            if code[i] in "\"'":
                _literal, consumed = self._read_string(code, i)
                spans.append((i, i + consumed))
                i += consumed
                continue
            if code.startswith("//", i):
                newline = code.find("\n", i + 2)
                end = len(code) if newline == -1 else newline
                spans.append((i, end))
                i = end
                continue
            if code.startswith("/*", i):
                comment_end = code.find("*/", i + 2)
                end = len(code) if comment_end == -1 else comment_end + 2
                spans.append((i, end))
                i = end
                continue
            i += 1
        return spans

    def _find_lexical_brace_scopes(self, code: str) -> List[Tuple[int, int]]:
        analysis = self._source_analysis(code)
        if analysis.lexical_brace_scopes is None:
            analysis.lexical_brace_scopes = self._scan_lexical_brace_scopes(code)
        return analysis.lexical_brace_scopes

    def _scan_lexical_brace_scopes(self, code: str) -> List[Tuple[int, int]]:
        scopes: List[Tuple[int, int]] = [(0, len(code))]
        brace_stack: List[int] = []
        i = 0
        while i < len(code):
            match = METAL_SCOPE_DELIMITER_RE.search(code, i)
            if match is None:
                break
            i = match.start()
            if code[i] in "\"'":
                _literal, consumed = self._read_string(code, i)
                i += consumed
                continue
            if code.startswith("//", i):
                newline = code.find("\n", i + 2)
                if newline == -1:
                    break
                i = newline + 1
                continue
            if code.startswith("/*", i):
                comment_end = code.find("*/", i + 2)
                if comment_end == -1:
                    break
                i = comment_end + 2
                continue
            if code[i] == "{":
                brace_stack.append(i)
            elif code[i] == "}" and brace_stack:
                scopes.append((brace_stack.pop() + 1, i))
            i += 1
        return scopes

    def _innermost_lexical_scope(
        self,
        scopes: List[Tuple[int, int]],
        position: int,
        source_length: int,
    ) -> Tuple[int, int]:
        if not scopes:
            return 0, source_length

        first_span = scopes[0]
        last_span = scopes[-1]
        cache = self._lexical_scope_lookup_cache
        key = id(scopes)
        lookup = cache.get(key)
        if (
            lookup is None
            or lookup.scopes is not scopes
            or lookup.source_length != source_length
            or lookup.span_count != len(scopes)
            or lookup.first_span != first_span
            or lookup.last_span != last_span
        ):
            lookup = self._build_lexical_scope_lookup(scopes, source_length)
            cache[key] = lookup
            if len(cache) > LEXICAL_SCOPE_CACHE_LIMIT:
                del cache[next(iter(cache))]

        boundary_index = bisect_right(lookup.boundaries, position) - 1
        if boundary_index >= 0:
            scope = lookup.innermost_scopes[boundary_index]
            if scope is not None:
                return scope
        return 0, source_length

    @staticmethod
    def _build_lexical_scope_lookup(
        scopes: List[Tuple[int, int]],
        source_length: int,
    ) -> _MetalLexicalScopeLookup:
        starts_by_position: Dict[int, List[int]] = {}
        ends_by_position: Dict[int, List[int]] = {}
        for index, (start, end) in enumerate(scopes):
            if start >= end:
                continue
            starts_by_position.setdefault(start, []).append(index)
            ends_by_position.setdefault(end, []).append(index)

        active = [False] * len(scopes)
        active_by_start: List[Tuple[int, int]] = []
        boundaries: List[int] = []
        innermost_scopes: List[Optional[Tuple[int, int]]] = []
        for boundary in sorted(starts_by_position.keys() | ends_by_position.keys()):
            for index in ends_by_position.get(boundary, ()):
                active[index] = False
            for index in starts_by_position.get(boundary, ()):
                active[index] = True
                heappush(active_by_start, (-scopes[index][0], index))
            while active_by_start and not active[active_by_start[0][1]]:
                heappop(active_by_start)
            boundaries.append(boundary)
            innermost_scopes.append(
                scopes[active_by_start[0][1]] if active_by_start else None
            )

        return _MetalLexicalScopeLookup(
            scopes=scopes,
            source_length=source_length,
            span_count=len(scopes),
            first_span=scopes[0],
            last_span=scopes[-1],
            boundaries=boundaries,
            innermost_scopes=innermost_scopes,
        )

    @staticmethod
    def _resolve_type_alias_binding_at(
        type_aliases: Dict[str, List[_MetalTypeAliasBinding]],
        name: str,
        position: int,
    ) -> Optional[_MetalTypeAliasBinding]:
        best: Optional[_MetalTypeAliasBinding] = None
        for binding in type_aliases.get(name, []):
            if binding.declaration_position > position:
                break
            if not (binding.scope_start <= position < binding.scope_end):
                continue
            if best is None or binding.declaration_position > best.declaration_position:
                best = binding
        return best

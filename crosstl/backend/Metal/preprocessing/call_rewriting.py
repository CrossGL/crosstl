"""Rewriting of implicit conversions and of member calls on struct receivers."""

import re
from typing import (
    Dict,
    List,
    Optional,
    Set,
    Tuple,
)

from ..receiver_visibility import visible_receiver_declaration
from .support import (
    IDENTIFIER_RE,
    MetalStructMethodError,
    _MetalIntegralConstantBinding,
    _MetalPositionedBufferTypes,
    _MetalReceiverContract,
    _MetalReceiverDeclaration,
    _MetalStructDefinition,
    _MetalStructMethod,
    _MetalTypeAliasBinding,
)


class MetalCallRewriteMixin:
    """Members of ``MetalPreprocessor``: rewriting of implicit conversions and of member calls on struct receivers."""

    def _try_rewrite_implicit_conversion_call(
        self,
        code: str,
        ident_start: int,
        target_type: str,
        arg_open: int,
        buffer_element_types: _MetalPositionedBufferTypes,
        local_variable_types: Dict[str, List[Tuple[int, str]]],
        field_variable_types: Dict[str, List[Tuple[int, str]]],
        field_structs_by_name: Dict[str, _MetalStructDefinition],
        type_aliases: Optional[Dict[str, List[_MetalTypeAliasBinding]]] = None,
        local_integral_constants: Optional[
            Dict[str, List[_MetalIntegralConstantBinding]]
        ] = None,
    ) -> Optional[Tuple[int, str]]:
        normalized_target = self._conversion_value_type(
            target_type, ident_start, type_aliases or {}
        )
        if normalized_target is None:
            return None
        arg_close = self._find_matching_delimiter(code, arg_open, "(", ")")
        if arg_close is None:
            return None
        arguments = [
            argument.strip()
            for argument in self._split_top_level_commas(code[arg_open + 1 : arg_close])
            if argument.strip()
        ]
        if len(arguments) != 1:
            return None
        argument = arguments[0]
        receiver_expression = self._strip_enclosing_parens(
            self._strip_template_argument_comments(argument).strip()
        )
        source_type = self._infer_argument_type(
            argument,
            self._flatten_types_at(buffer_element_types, arg_open),
            self._flatten_types_at(local_variable_types, arg_open),
            self._struct_field_types_at(
                field_variable_types,
                field_structs_by_name,
                arg_open,
            ),
            field_structs_by_name,
        )
        if source_type is None and type_aliases:
            source_type = self._aliased_constructor_type_at(
                argument, arg_open, type_aliases, local_integral_constants or {}
            )
        if source_type is None and IDENTIFIER_RE.fullmatch(receiver_expression):
            declaration = self._visible_receiver_declaration_at(
                code, receiver_expression, ident_start
            )
            if declaration is not None:
                source_type = self._canonicalize_type_aliases_at(
                    declaration.raw_type_text,
                    type_aliases or {},
                    declaration.type_position,
                )
        source_type = self._canonicalize_type_aliases_at(
            source_type or "", type_aliases or {}, arg_open
        )
        source_type = (
            self._normalize_inferred_type(source_type or "").rstrip("&").strip()
        )
        struct = field_structs_by_name.get(source_type)
        if struct is None:
            return None
        candidates = [
            method
            for method in struct.methods
            if method.is_conversion_operator
            and self._conversion_value_type(
                self._canonicalize_struct_scoped_type(
                    method.return_type, struct, field_structs_by_name
                ),
                method.span[0],
                type_aliases or {},
            )
            == normalized_target
        ]
        if not candidates:
            return None
        receiver = self._receiver_contract_for_named_value(
            code,
            source_type,
            receiver_expression,
            ident_start,
            type_aliases=type_aliases,
        )
        if IDENTIFIER_RE.fullmatch(receiver_expression):
            declaration = self._visible_receiver_declaration_at(
                code, receiver_expression, ident_start
            )
            if declaration is not None:
                canonical_receiver = self._canonicalize_type_aliases_at(
                    declaration.raw_type_text,
                    type_aliases or {},
                    declaration.type_position,
                )
                if canonical_receiver is not None:
                    receiver = self._receiver_contract_from_type_text(
                        canonical_receiver, source_type
                    )
        if receiver is None:
            construction = re.match(r"[A-Za-z_]\w*\s*([({])", receiver_expression)
            if construction is not None:
                opening = construction.end() - 1
                delimiter = receiver_expression[opening]
                closing = self._find_matching_delimiter(
                    receiver_expression,
                    opening,
                    delimiter,
                    ")" if delimiter == "(" else "}",
                )
                if closing == len(receiver_expression) - 1:
                    receiver = self._receiver_contract_from_type_text(
                        source_type, source_type, value_category="rvalue"
                    )
            elif receiver_expression.startswith("*("):
                closing = self._find_matching_delimiter(
                    receiver_expression, 1, "(", ")"
                )
                if closing is not None:
                    pointer_type = receiver_expression[2:closing].strip()
                    if pointer_type.endswith("*"):
                        receiver = self._receiver_contract_from_type_text(
                            pointer_type[:-1], source_type
                        )
        records = []
        viable = []
        for candidate in candidates:
            mismatches = (
                self._receiver_candidate_mismatches(candidate, receiver)
                if receiver is not None
                else ("receiver cv/reference/address-space state is unavailable",)
            )
            records.append(
                {
                    "candidate": self._concrete_method_candidate_signature(candidate),
                    "receiver_qualifiers": self._method_receiver_qualifier_details(
                        candidate
                    ),
                    "mismatches": mismatches,
                    "viable": not mismatches,
                }
            )
            if not mismatches:
                viable.append(candidate)
        if len(viable) > 1:
            best = min(
                self._receiver_candidate_preference_rank(candidate, receiver)
                for candidate in viable
            )
            viable = [
                candidate
                for candidate in viable
                if self._receiver_candidate_preference_rank(candidate, receiver) == best
            ]
        if len(viable) != 1:
            raise MetalStructMethodError(
                f"Cannot select a unique conversion from '{source_type}' to '{target_type}'.",
                struct_name=source_type,
                method_name=f"operator {target_type}",
                requested_signature=f"{target_type}({argument})",
                suggested_action="preserve receiver qualifiers and provide a unique source conversion",
                source_location=self._source_location_for_offsets(
                    code, ident_start, arg_close + 1
                ),
                missing_capabilities=("metal.conversion-operator-selection",),
                reason=(
                    "conversion-operator-ambiguous"
                    if viable
                    else "conversion-operator-no-viable"
                ),
                receiver_type=(
                    receiver.source_type if receiver is not None else "<unknown>"
                ),
                candidate_signatures=tuple(
                    self._concrete_method_candidate_signature(candidate)
                    for candidate in candidates
                ),
                candidate_mismatches=tuple(records),
            )
        method = viable[0]
        if not self._conversion_operator_receiver_is_readonly(struct, method) and (
            IDENTIFIER_RE.fullmatch(receiver_expression) is None
        ):
            raise MetalStructMethodError(
                "Cannot lower a mutating conversion operator on a temporary "
                f"'{source_type}' value.",
                struct_name=source_type,
                method_name=method.name,
                requested_signature=f"{target_type}({argument})",
                suggested_action=(
                    "materialize the source aggregate in a named local or make "
                    "the conversion operator receiver read-only"
                ),
                source_location=self._source_location_for_offsets(
                    code, ident_start, arg_close + 1
                ),
                missing_capabilities=("metal.conversion-operator-temporary",),
                reason="conversion-temporary-receiver-mutation-unsupported",
                receiver_type=source_type,
            )
        return arg_close + 1, f"{method.free_name}({argument})"

    def _try_rewrite_nested_member_call_at(
        self,
        code: str,
        ident_start: int,
        ident: str,
        ident_end: int,
        methods_by_struct: Dict[str, Dict[str, _MetalStructMethod]],
        template_methods_by_struct: Dict[str, Dict[str, List[_MetalStructMethod]]],
        operator_call_structs: Set[str],
        variable_types: Dict[str, List[Tuple[int, str]]],
        structs_by_name: Dict[str, _MetalStructDefinition],
        buffer_element_types: _MetalPositionedBufferTypes,
        local_variable_types: Dict[str, List[Tuple[int, str]]],
        instantiated_template_functions: Dict[str, str],
        field_variable_types: Dict[str, List[Tuple[int, str]]],
        field_structs_by_name: Dict[str, _MetalStructDefinition],
        local_integral_constants: Optional[
            Dict[str, List[_MetalIntegralConstantBinding]]
        ],
        struct_type_aliases: Optional[Dict[str, List[_MetalTypeAliasBinding]]],
        type_aliases: Optional[Dict[str, List[_MetalTypeAliasBinding]]],
        type_alias_fallback_position: Optional[int],
        implicit_receiver_contract: Optional[_MetalReceiverContract],
    ) -> Optional[Tuple[int, str]]:
        """Rewrite a method call whose receiver is a nested struct field."""
        if self._is_member_identifier_context(code, ident_start):
            return None

        receiver_declaration, resolved_receiver_type = self._receiver_struct_type_at(
            code,
            ident,
            ident_start,
            set(field_structs_by_name),
            struct_type_aliases,
        )
        if receiver_declaration is not None:
            if resolved_receiver_type is None:
                return None
            receiver_contract = self._receiver_contract_from_type_text(
                receiver_declaration.raw_type_text,
                resolved_receiver_type,
            )
            declared_type = f"{resolved_receiver_type}{receiver_contract.indirection}"
        else:
            declared_type = self._resolve_declared_type_at(
                field_variable_types, ident, ident_start
            ) or self._resolve_declared_type_at(variable_types, ident, ident_start)
        receiver_info = self._nested_member_receiver_info(
            declared_type, field_structs_by_name
        )
        if receiver_info is None:
            return None
        current_struct_name, required_access = receiver_info
        current_receiver_contract = (
            implicit_receiver_contract
            if ident == "self" and implicit_receiver_contract is not None
            else self._receiver_contract_for_named_value(
                code,
                current_struct_name,
                ident,
                ident_start,
                type_aliases=struct_type_aliases,
            )
        )

        receiver_end = ident_end
        cursor = ident_end
        while True:
            while cursor < len(code) and code[cursor].isspace():
                cursor += 1
            if code.startswith("->", cursor):
                access = "->"
                cursor += 2
            elif cursor < len(code) and code[cursor] == ".":
                access = "."
                cursor += 1
            else:
                return None
            if access != required_access:
                return None

            while cursor < len(code) and code[cursor].isspace():
                cursor += 1
            member, consumed = self._read_identifier(code, cursor)
            if not member:
                return None
            cursor += consumed
            explicit_template_marker = member == "template"
            if explicit_template_marker:
                while cursor < len(code) and code[cursor].isspace():
                    cursor += 1
                member, consumed = self._read_identifier(code, cursor)
                if not member:
                    return None
                cursor += consumed

            call_suffix = self._member_template_call_suffix(code, cursor)
            if call_suffix is not None:
                arg_open, explicit_template_arguments = call_suffix
                receiver = code[ident_start:receiver_end].strip()
                if access == "->":
                    receiver = f"{receiver}[0]"
                owner_struct = structs_by_name.get(current_struct_name)
                reference_methods = (
                    self._concrete_reference_methods(
                        owner_struct, member, code, arg_open
                    )
                    if owner_struct is not None
                    else []
                )
                if (
                    reference_methods
                    and explicit_template_arguments is None
                    and re.fullmatch(r"[A-Za-z_]\w*(?:\s*\.\s*[A-Za-z_]\w*)*", receiver)
                    and access == "."
                ):
                    method = self._select_direct_reference_method(
                        code,
                        owner_struct,
                        reference_methods,
                        arg_open,
                        receiver_is_const=(
                            None
                            if current_receiver_contract is not None
                            and "constant" in current_receiver_contract.address_spaces
                            else (
                                current_receiver_contract.is_readonly
                                if current_receiver_contract is not None
                                else self._simple_receiver_constness(
                                    code,
                                    current_struct_name,
                                    ident,
                                    ident_start,
                                    type_aliases=struct_type_aliases,
                                )
                            )
                        ),
                    )
                    return self._build_direct_reference_accessor_rewrite(
                        code,
                        owner_struct,
                        method,
                        receiver=receiver,
                        arg_open=arg_open,
                        call_start=ident_start,
                    )
                template_overloads = template_methods_by_struct.get(
                    current_struct_name, {}
                ).get(member)
                method = None
                if owner_struct is not None and explicit_template_arguments is None:
                    method = self._select_concrete_method_for_call(
                        code,
                        owner_struct,
                        member,
                        arg_open,
                        buffer_element_types,
                        local_variable_types,
                        field_variable_types,
                        field_structs_by_name,
                        allow_template_fallback=bool(template_overloads),
                        template_overloads=template_overloads,
                        require_static=False,
                        type_alias_fallback_position=type_alias_fallback_position,
                        receiver_contract=current_receiver_contract,
                        call_start=ident_start,
                    )
                if method is not None and explicit_template_arguments is None:
                    return self._build_instance_call_rewrite(
                        code,
                        receiver,
                        method,
                        arg_open,
                        struct_name=owner_struct,
                    )
                if template_overloads:
                    struct = structs_by_name.get(current_struct_name)
                    if struct is None:
                        return None
                    return self._instantiate_template_member_call(
                        code,
                        struct,
                        template_overloads,
                        receiver=receiver,
                        receiver_is_const=(
                            None
                            if current_receiver_contract is not None
                            and "constant" in current_receiver_contract.address_spaces
                            else (
                                current_receiver_contract.is_readonly
                                if current_receiver_contract is not None
                                else self._simple_receiver_constness(
                                    code,
                                    current_struct_name,
                                    ident,
                                    ident_start,
                                    type_aliases=struct_type_aliases,
                                )
                            )
                        ),
                        arg_open=arg_open,
                        buffer_element_types=buffer_element_types,
                        local_variable_types=local_variable_types,
                        instantiated_template_functions=instantiated_template_functions,
                        structs_by_name=field_structs_by_name,
                        variable_types=field_variable_types,
                        template_methods_by_struct=template_methods_by_struct,
                        methods_by_struct=methods_by_struct,
                        operator_call_structs=operator_call_structs,
                        rewrite_structs_by_name=structs_by_name,
                        explicit_template_arguments=explicit_template_arguments,
                        local_integral_constants=local_integral_constants,
                        argument_type_aliases=type_aliases,
                        argument_type_fallback_position=type_alias_fallback_position,
                    )
                return None
            if explicit_template_marker:
                return None

            current_struct = field_structs_by_name.get(current_struct_name)
            if current_struct is None:
                return None
            field_type = current_struct.data_member_types.get(member)
            field = next(
                (item for item in current_struct.data_members if item.name == member),
                None,
            )
            if field is not None and field.array_suffix:
                for _ in range(field.array_suffix.count("[")):
                    cursor = self._skip_cpp_trivia(code, cursor)
                    if cursor >= len(code) or code[cursor] != "[":
                        return None
                    closing = self._find_matching_delimiter(code, cursor, "[", "]")
                    if closing is None:
                        return None
                    cursor = closing + 1
                field_type = field.type_text
            receiver_info = self._nested_member_receiver_info(
                field_type, field_structs_by_name
            )
            if receiver_info is None:
                return None
            next_struct_name, required_access = receiver_info
            current_receiver_contract = self._member_receiver_contract(
                current_receiver_contract,
                current_struct,
                member,
                next_struct_name,
            )
            current_struct_name = next_struct_name
            receiver_end = cursor

    def _nested_member_receiver_info(
        self,
        type_text: Optional[str],
        structs_by_name: Dict[str, _MetalStructDefinition],
    ) -> Optional[Tuple[str, str]]:
        """Return a concrete receiver struct and its required access operator."""
        normalized = self._normalize_inferred_type(type_text or "")
        if not normalized:
            return None
        indirection_match = re.search(r"(?P<indirection>[*&]+)\s*$", normalized)
        indirection = (
            indirection_match.group("indirection")
            if indirection_match is not None
            else ""
        )
        if indirection not in {"", "*", "&", "&&"}:
            return None
        struct_name = (
            normalized[: indirection_match.start()].strip()
            if indirection_match is not None
            else normalized
        )
        if struct_name not in structs_by_name:
            return None
        return struct_name, "->" if indirection == "*" else "."

    def _receiver_contract_for_named_value(
        self,
        code: str,
        struct_name: str,
        receiver: str,
        call_start: int,
        *,
        type_aliases: Optional[Dict[str, List[_MetalTypeAliasBinding]]] = None,
    ) -> Optional[_MetalReceiverContract]:
        if re.fullmatch(r"[A-Za-z_]\w*", receiver) is None:
            return None
        declaration = self._visible_receiver_declaration_at(
            code,
            receiver,
            call_start,
        )
        if declaration is None:
            return None
        resolved_type = declaration.declared_type
        if resolved_type != struct_name and type_aliases is not None:
            resolved_type = self._resolve_struct_type_alias_at(
                type_aliases,
                declaration.declared_type,
                declaration.type_position,
            )
        if resolved_type != struct_name:
            resolved_type = self._qualified_receiver_struct(
                code, declaration, {struct_name}
            )
        if resolved_type != struct_name:
            return None

        return self._receiver_contract_from_type_text(
            declaration.raw_type_text,
            struct_name,
        )

    def _visible_receiver_declaration_at(
        self,
        code: str,
        receiver: str,
        position: int,
    ) -> Optional[_MetalReceiverDeclaration]:
        candidates = self._find_receiver_declarations(code).get(receiver)
        if not candidates:
            return None
        return visible_receiver_declaration(
            self._source_analysis(code).receiver_sweeps,
            candidates,
            receiver,
            position,
        )

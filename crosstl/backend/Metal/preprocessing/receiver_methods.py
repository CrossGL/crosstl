"""Receiver struct types and contracts, and concrete method overload selection."""

import re
from bisect import bisect_right
from dataclasses import replace
from typing import (
    Dict,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
)

from ..overload_ranking import shared_template_parameter_positions
from ..reference_binding import argument_binds_parameter
from .member_typedef_receivers import member_typedef_target, names_library_template
from .qualified_receivers import qualified_struct_receiver
from .support import (
    IDENTIFIER_RE,
    METAL_BUILTIN_MEMBER_RECEIVER_TEMPLATE_RE,
    METAL_RECEIVER_DECLARATOR_PREFIX_RE,
    MetalStructMethodError,
    _MetalPositionedBufferTypes,
    _MetalReceiverContract,
    _MetalReceiverDeclaration,
    _MetalStructDefinition,
    _MetalStructMethod,
    _MetalTypeAliasBinding,
)


class MetalReceiverMethodMixin:
    """Members of ``MetalPreprocessor``: receiver struct types and contracts, and concrete method overload selection."""

    def _receiver_struct_type_at(
        self,
        code: str,
        receiver: str,
        position: int,
        struct_names: Set[str],
        type_aliases: Optional[Dict[str, List[_MetalTypeAliasBinding]]],
    ) -> Tuple[Optional[_MetalReceiverDeclaration], Optional[str]]:
        declaration = self._visible_receiver_declaration_at(
            code,
            receiver,
            position,
        )
        if declaration is None:
            return None, None
        resolved_type = declaration.declared_type
        if resolved_type not in struct_names and type_aliases is not None:
            resolved_type = self._resolve_struct_type_alias_at(
                type_aliases,
                declaration.declared_type,
                declaration.type_position,
            )
        if resolved_type not in struct_names:
            resolved_type = self._qualified_receiver_struct(
                code, declaration, struct_names
            )
        if resolved_type not in struct_names:
            return declaration, None
        return declaration, resolved_type

    def _qualified_receiver_struct(
        self,
        code: str,
        declaration: _MetalReceiverDeclaration,
        struct_names: Set[str],
    ) -> Optional[str]:
        """The struct a receiver declared with a namespace-qualified type names (:mod:`.qualified_receivers`)."""
        if "::" not in (declaration.declared_type or ""):
            return None
        definitions = [
            (definition.name, definition.qualified_name or definition.name)
            for definition in self._find_concrete_struct_definitions(code)
        ]
        lexical_namespace = self._namespace_at(
            self._find_namespace_spans(code), max(declaration.type_position, 0)
        )
        return qualified_struct_receiver(
            declaration.declared_type,
            struct_names,
            definitions,
            lexical_namespace,
        )

    def _receiver_has_builtin_member_type_at(
        self,
        code: str,
        receiver: str,
        position: int,
        receiver_declaration: Optional[_MetalReceiverDeclaration],
    ) -> bool:
        """Whether ``receiver`` is visibly declared as a Metal library template.

        The receiver scan only reads plain type names, so a declaration such as
        ``simdgroup_matrix<float, 8, 8> G`` or a ``texture2d<float> tex``
        parameter is invisible to it. Such a receiver's member calls are
        library operations the frontend lowers, not struct methods. The nearest
        visible declaration wins, so a later plain declaration that shadows the
        library-typed one still fails closed.
        """
        ignored_spans = self._find_comment_and_literal_spans(code)
        scopes = self._find_lexical_brace_scopes(code)
        name_re = re.compile(rf"{re.escape(receiver)}\b\s*(?=[,;)=\[\{{(])")
        nearest = None
        # The template references that end by ``position`` are exactly the ones
        # a scan of ``code[:position]`` finds: each ends at its ``<``.
        references, ends = self._builtin_member_receiver_template_references(code)
        for start, end in references[: bisect_right(ends, position)]:
            if self._containing_span(start, ignored_spans) is not None:
                continue
            close = self._find_matching_angle(code, end - 1)
            if close is None or close >= position:
                continue
            prefix = METAL_RECEIVER_DECLARATOR_PREFIX_RE.match(code, close + 1)
            cursor = prefix.end() if prefix is not None else close + 1
            declared = name_re.match(code, cursor)
            if declared is None:
                if not self._builtin_declarator_list_declares(
                    code, cursor, receiver, position
                ):
                    continue
            elif declared.end() > position:
                continue
            scope_start, scope_end = self._innermost_lexical_scope(
                scopes, start, len(code)
            )
            if scope_start <= position < scope_end:
                nearest = start
        if nearest is None:
            return False
        return (
            receiver_declaration is None
            or receiver_declaration.declaration_position < nearest
        )

    def _receiver_member_typedef_names_library_type(
        self,
        receiver_declaration: Optional[_MetalReceiverDeclaration],
        structs_by_name: Dict[str, _MetalStructDefinition],
    ) -> bool:
        """Whether ``receiver_declaration`` spells a struct member alias of a Metal library template.

        ``Frag::mat_type val_mat;`` where ``Frag`` declares ``typedef metal::simdgroup_matrix<float, 8, 8>
        mat_type;`` declares a library object (:mod:`.member_typedef_receivers`), so its member calls are left to
        the frontend like those of a receiver spelled with the template itself.
        """
        if receiver_declaration is None:
            return False
        target = member_typedef_target(
            receiver_declaration.declared_type, structs_by_name
        )
        return target is not None and names_library_template(
            target, METAL_BUILTIN_MEMBER_RECEIVER_TEMPLATE_RE
        )

    def _builtin_member_receiver_template_references(
        self, code: str
    ) -> Tuple[List[Tuple[int, int]], List[int]]:
        """Spans of the Metal library template references in ``code``, and their ends."""
        analysis = self._source_analysis(code)
        if analysis.builtin_member_receiver_template_references is None:
            spans = [
                match.span()
                for match in METAL_BUILTIN_MEMBER_RECEIVER_TEMPLATE_RE.finditer(code)
            ]
            analysis.builtin_member_receiver_template_references = (
                spans,
                [end for _start, end in spans],
            )
        return analysis.builtin_member_receiver_template_references

    def _builtin_declarator_list_declares(
        self, code: str, cursor: int, receiver: str, position: int
    ) -> bool:
        """Whether a later declarator of one declaration names ``receiver``.

        ``simdgroup_matrix<float, 8, 8> A, B[2], C = D;`` declares ``B`` and
        ``C`` with the library type too. Anything other than plain
        declarators, array extents and initializers ends the scan, so a
        parameter list or an unparsed initializer fails closed.
        """
        identifier = re.compile(r"[A-Za-z_]\w*")
        index = cursor
        while index < position:
            if index != cursor:
                prefix = METAL_RECEIVER_DECLARATOR_PREFIX_RE.match(code, index)
                if prefix is not None:
                    index = prefix.end()
            name = identifier.match(code, index)
            if name is None:
                return False
            if name.group(0) == receiver:
                return name.end() <= position
            index = name.end()
            while True:
                while index < len(code) and code[index].isspace():
                    index += 1
                if index >= len(code) or code[index] != "[":
                    break
                close = self._find_matching_delimiter(code, index, "[", "]")
                if close is None:
                    return False
                index = close + 1
            if index < len(code) and code[index] in "({":
                close = self._find_matching_delimiter(
                    code, index, code[index], ")" if code[index] == "(" else "}"
                )
                if close is None:
                    return False
                index = close + 1
            elif index < len(code) and code[index] == "=":
                index += 1
                depth = 0
                while index < len(code):
                    char = code[index]
                    if char in "([{":
                        depth += 1
                    elif char in ")]}":
                        if depth == 0:
                            return False
                        depth -= 1
                    elif char == "<" and depth == 0:
                        close = self._find_matching_angle(code, index)
                        if close is None:
                            return False
                        index = close
                    elif char in ",;" and depth == 0:
                        break
                    index += 1
            while index < len(code) and code[index].isspace():
                index += 1
            if index >= len(code) or code[index] != ",":
                return False
            index += 1
            while index < len(code) and code[index].isspace():
                index += 1
        return False

    def _receiver_contract_from_type_text(
        self,
        type_text: str,
        struct_name: str,
        *,
        inherited_address_spaces: Optional[Tuple[str, ...]] = None,
        value_category: str = "lvalue",
    ) -> _MetalReceiverContract:
        normalized = self._normalize_template_argument_text(type_text or struct_name)
        indirection_match = re.search(
            r"(?P<indirection>\*++|&&|&)\s*+"
            r"(?:(?:__restrict__|__restrict|restrict|const|volatile)\s*+)*+$",
            normalized,
        )
        indirection = (
            indirection_match.group("indirection")
            if indirection_match is not None
            else ""
        )
        object_qualifier_text = (
            normalized[: indirection_match.start()]
            if indirection_match is not None
            else normalized
        )
        qualifier_words = set(IDENTIFIER_RE.findall(object_qualifier_text))
        address_spaces = self._receiver_address_spaces(object_qualifier_text)
        if not address_spaces:
            address_spaces = inherited_address_spaces or ("thread",)
        is_const = "const" in qualifier_words
        is_volatile = "volatile" in qualifier_words
        display_qualifiers = []
        if is_const:
            display_qualifiers.append("const")
        if is_volatile:
            display_qualifiers.append("volatile")
        display_qualifiers.extend(address_spaces)
        display_qualifiers.append(f"{struct_name}{indirection}")
        return _MetalReceiverContract(
            struct_name=struct_name,
            is_const=is_const,
            is_volatile=is_volatile,
            address_spaces=address_spaces,
            indirection=indirection,
            value_category=value_category,
            source_type=" ".join(display_qualifiers),
        )

    def _member_receiver_contract(
        self,
        parent: Optional[_MetalReceiverContract],
        owner: _MetalStructDefinition,
        member_name: str,
        member_struct_name: str,
    ) -> Optional[_MetalReceiverContract]:
        member = next(
            (item for item in owner.data_members if item.name == member_name),
            None,
        )
        if member is None:
            return None
        inherited_spaces = parent.address_spaces if parent is not None else None
        contract = self._receiver_contract_from_type_text(
            member.type_text,
            member_struct_name,
            inherited_address_spaces=inherited_spaces,
        )
        is_indirect = bool(contract.indirection)
        member_words = set(IDENTIFIER_RE.findall(member.type_text))
        if parent is not None and not is_indirect:
            inherited_const = parent.is_const and "mutable" not in member_words
            contract = replace(
                contract,
                is_const=contract.is_const or inherited_const,
                is_volatile=contract.is_volatile or parent.is_volatile,
            )
            display_qualifiers = []
            if contract.is_const:
                display_qualifiers.append("const")
            if contract.is_volatile:
                display_qualifiers.append("volatile")
            display_qualifiers.extend(contract.address_spaces)
            display_qualifiers.append(f"{contract.struct_name}{contract.indirection}")
            contract = replace(
                contract,
                source_type=" ".join(display_qualifiers),
            )
        return contract

    def _concrete_method_matches_call(
        self,
        method: _MetalStructMethod,
        code: str,
        arg_open: int,
        buffer_element_types: _MetalPositionedBufferTypes,
        local_variable_types: Dict[str, List[Tuple[int, str]]],
        variable_types: Dict[str, List[Tuple[int, str]]],
        structs_by_name: Dict[str, _MetalStructDefinition],
        type_alias_fallback_position: Optional[int] = None,
        shared_positions: Optional[Set[int]] = None,
    ) -> bool:
        arg_close = self._find_matching_delimiter(code, arg_open, "(", ")")
        if arg_close is None:
            return False
        raw_args = code[arg_open + 1 : arg_close]
        call_arguments = [
            argument
            for argument in self._split_top_level_commas(raw_args)
            if argument.strip()
        ]
        declared_parameters = [
            parameter
            for parameter in self._split_top_level_commas(method.parameters)
            if parameter.strip() and parameter.strip() != "void"
        ]
        if not self._callable_accepts_argument_count(
            method.parameters, len(call_arguments)
        ):
            return False
        fixed_parameters = [
            parameter for parameter in declared_parameters if "..." not in parameter
        ]
        if len(call_arguments) > len(fixed_parameters):
            return False
        declared_types = [
            self._normalize_inferred_type(
                self._normalize_function_parameter_type_text(parameter)
            )
            for parameter in fixed_parameters[: len(call_arguments)]
        ]
        if self._source_type_alias_bindings:
            declared_types = [
                self._canonicalize_type_aliases_at(
                    declared_type,
                    self._source_type_alias_bindings,
                    method.span[0],
                )
                or declared_type
                for declared_type in declared_types
            ]
        buffer_view = self._flatten_types_at(buffer_element_types, arg_open)
        local_view = self._flatten_types_at(local_variable_types, arg_open)
        struct_field_types = self._struct_field_types_at(
            variable_types,
            structs_by_name,
            arg_open,
        )
        inferred_types = []
        for position, argument in enumerate(call_arguments):
            if shared_positions and position in shared_positions:
                # Every competing template converts this argument to the same
                # declared type, so the conversion does not rank candidates.
                inferred_types.append(declared_types[position])
                continue
            inferred = self._infer_argument_type(
                argument,
                buffer_view,
                local_view,
                struct_field_types,
                structs_by_name,
            )
            if inferred is None:
                return False
            inferred_type = self._normalize_inferred_type(inferred)
            if self._source_type_alias_bindings:
                inferred_type = (
                    self._canonicalize_type_aliases_at(
                        inferred_type,
                        self._source_type_alias_bindings,
                        arg_open,
                    )
                    or inferred_type
                )
                if (
                    type_alias_fallback_position is not None
                    and inferred_type == self._normalize_inferred_type(inferred)
                ):
                    inferred_type = (
                        self._canonicalize_type_aliases_at(
                            inferred_type,
                            self._source_type_alias_bindings,
                            type_alias_fallback_position,
                        )
                        or inferred_type
                    )
            inferred_types.append(inferred_type)
        # A reference parameter binds an argument of its referenced type
        # (reference_binding.py), although the two spellings differ.
        return all(
            argument_binds_parameter(
                self,
                declared_type,
                inferred_type,
                self._normalize_function_parameter_type_text(parameter),
                argument,
                local_view,
                buffer_view,
                structs_by_name,
            )
            for declared_type, inferred_type, parameter, argument in zip(
                declared_types, inferred_types, fixed_parameters, call_arguments
            )
        )

    def _select_concrete_method_for_call(
        self,
        code: str,
        struct: _MetalStructDefinition,
        method_name: str,
        arg_open: int,
        buffer_element_types: _MetalPositionedBufferTypes,
        local_variable_types: Dict[str, List[Tuple[int, str]]],
        variable_types: Dict[str, List[Tuple[int, str]]],
        structs_by_name: Dict[str, _MetalStructDefinition],
        *,
        allow_template_fallback: bool,
        template_overloads: Optional[Sequence[_MetalStructMethod]] = None,
        require_static: Optional[bool] = None,
        type_alias_fallback_position: Optional[int] = None,
        receiver_contract: Optional[_MetalReceiverContract] = None,
        call_start: Optional[int] = None,
    ) -> Optional[_MetalStructMethod]:
        arg_close = self._find_matching_delimiter(code, arg_open, "(", ")")
        if arg_close is None:
            return None
        raw_args = code[arg_open + 1 : arg_close].strip()
        argument_count = len(
            [
                argument
                for argument in self._split_top_level_commas(raw_args)
                if argument.strip()
            ]
        )
        candidates = [
            method
            for method in struct.methods
            if method.name == method_name
            and (require_static is None or method.is_static is require_static)
            and self._callable_accepts_argument_count(method.parameters, argument_count)
        ]
        if not candidates:
            return None
        exact_argument_match = {
            id(method): self._concrete_method_matches_call(
                method,
                code,
                arg_open,
                buffer_element_types,
                local_variable_types,
                variable_types,
                structs_by_name,
                type_alias_fallback_position,
                shared_positions=(
                    shared_template_parameter_positions(
                        self, method, template_overloads, argument_count
                    )
                    if len(candidates) == 1
                    else None
                ),
            )
            for method in candidates
        }
        require_exact_arguments = len(candidates) > 1 or allow_template_fallback
        receiver_contract_required = any(not method.is_static for method in candidates)

        mismatch_records = []
        receiver_viable = []
        viable = []
        for method in candidates:
            receiver_mismatches = []
            if not method.is_static:
                if receiver_contract is None and receiver_contract_required:
                    receiver_mismatches.append(
                        "receiver cv/reference/address-space state is unavailable"
                    )
                elif receiver_contract is not None:
                    receiver_mismatches.extend(
                        self._receiver_candidate_mismatches(
                            method,
                            receiver_contract,
                        )
                    )
            if not receiver_mismatches:
                receiver_viable.append(method)
            mismatches = []
            if require_exact_arguments and not exact_argument_match[id(method)]:
                mismatches.append("explicit argument types are not an exact match")
            mismatches.extend(receiver_mismatches)
            record = {
                "candidate": self._concrete_method_candidate_signature(method),
                "receiver_qualifiers": self._method_receiver_qualifier_details(method),
                "mismatches": tuple(mismatches),
                "viable": not mismatches,
            }
            mismatch_records.append(record)
            if not mismatches:
                viable.append(method)

        if (
            not viable
            and allow_template_fallback
            and not any(exact_argument_match.values())
        ):
            return None
        if len(viable) > 1 and receiver_contract is not None:
            ranked = [
                (
                    self._receiver_candidate_preference_rank(
                        method,
                        receiver_contract,
                    ),
                    method,
                )
                for method in viable
            ]
            best_rank = min(rank for rank, _method in ranked)
            viable = [method for rank, method in ranked if rank == best_rank]
        if len(viable) == 1:
            return viable[0]

        reason = "no source-viable concrete overload matches"
        diagnostic_reason = "concrete-overload-no-viable"
        ambiguous_conversions = False
        if len(viable) > 1:
            reason = "multiple equally preferred concrete overloads match"
            diagnostic_reason = "concrete-overload-ambiguous"
        elif (
            not viable
            and len(receiver_viable) > 1
            and not any(exact_argument_match.values())
        ):
            reason = "multiple receiver-compatible conversion candidates remain"
            diagnostic_reason = "concrete-overload-ambiguous"
            ambiguous_conversions = True
        requested_signature = f"{struct.name}::{method_name}({raw_args})"
        receiver_type = (
            receiver_contract.source_type
            if receiver_contract is not None
            else "<unknown>"
        )
        candidate_signatures = tuple(
            self._concrete_method_candidate_signature(method) for method in candidates
        )
        location_start = (
            arg_open if ambiguous_conversions or call_start is None else call_start
        )
        raise MetalStructMethodError(
            "Cannot lower concrete member overload "
            f"'{struct.name}::{method_name}' for receiver '{receiver_type}': "
            f"{reason}. Requested call: {requested_signature}.",
            struct_name=struct.name,
            method_name=method_name,
            requested_signature=requested_signature,
            suggested_action=(
                "preserve the receiver declaration qualifiers and make the "
                "source overload contract unique"
            ),
            source_location=self._source_location_for_offsets(
                code,
                location_start,
                arg_close + 1,
            ),
            reason=diagnostic_reason,
            receiver_type=receiver_type,
            candidate_signatures=candidate_signatures,
            candidate_mismatches=tuple(mismatch_records),
            missing_capabilities=("metal.member-overload-resolution",),
        )

    @staticmethod
    def _method_receiver_qualifier_details(
        method: _MetalStructMethod,
    ) -> Dict[str, object]:
        return {
            "const": method.is_const,
            "volatile": method.is_volatile,
            "reference": method.ref_qualifier or None,
            "address_spaces": method.receiver_address_spaces or ("thread",),
        }

    def _concrete_method_candidate_signature(
        self,
        method: _MetalStructMethod,
    ) -> str:
        trailing = " ".join(method.trailing_qualifiers)
        suffix = f" {trailing}" if trailing else ""
        return f"{method.return_type} {method.name}({method.parameters}){suffix}"

    def _receiver_candidate_mismatches(
        self,
        method: _MetalStructMethod,
        receiver: _MetalReceiverContract,
    ) -> Tuple[str, ...]:
        mismatches = []
        method_spaces = method.receiver_address_spaces or ("thread",)
        if len(receiver.address_spaces) != 1:
            mismatches.append(
                "receiver has multiple or unprovable Metal address spaces"
            )
        if len(method_spaces) != 1:
            mismatches.append(
                "candidate has multiple or unprovable Metal address spaces"
            )
        if (
            len(receiver.address_spaces) == 1
            and len(method_spaces) == 1
            and receiver.address_spaces[0] != method_spaces[0]
        ):
            mismatches.append(
                "receiver address space "
                f"'{receiver.address_spaces[0]}' does not match candidate "
                f"'{method_spaces[0]}'"
            )
        if receiver.is_readonly and not method.is_const:
            mismatches.append("read-only receiver cannot call a mutating candidate")
        if receiver.is_volatile and not method.is_volatile:
            mismatches.append("volatile receiver cannot call a non-volatile candidate")
        if method.ref_qualifier == "&" and receiver.value_category != "lvalue":
            mismatches.append("lvalue-qualified candidate requires an lvalue receiver")
        if method.ref_qualifier == "&&" and receiver.value_category != "rvalue":
            mismatches.append("rvalue-qualified candidate requires an rvalue receiver")

        semantic_qualifiers = {
            "const",
            "volatile",
            "constant",
            "device",
            "thread",
            "threadgroup",
            "threadgroup_imageblock",
            "&",
            "&&",
            "noexcept",
            "override",
            "final",
        }
        unknown = tuple(
            qualifier
            for qualifier in method.trailing_qualifiers
            if qualifier not in semantic_qualifiers
        )
        if unknown:
            mismatches.append(
                "candidate has unsupported trailing qualifiers: " + ", ".join(unknown)
            )
        return tuple(mismatches)

    @staticmethod
    def _receiver_candidate_preference_rank(
        method: _MetalStructMethod,
        receiver: _MetalReceiverContract,
    ) -> Tuple[int, int]:
        cv_additions = int(method.is_const and not receiver.is_const)
        cv_additions += int(method.is_volatile and not receiver.is_volatile)
        expected_reference = "&" if receiver.value_category == "lvalue" else "&&"
        reference_penalty = int(
            bool(method.ref_qualifier) and method.ref_qualifier != expected_reference
        )
        if not method.ref_qualifier:
            reference_penalty = 1
        return cv_additions, reference_penalty

    @staticmethod
    def _concrete_method_overload_ordinal(
        struct: _MetalStructDefinition,
        method: _MetalStructMethod,
    ) -> Optional[int]:
        overloads = [
            candidate
            for candidate in struct.methods
            if candidate.free_name == method.free_name
        ]
        if len(overloads) < 2:
            return None
        return next(
            index
            for index, candidate in enumerate(overloads, start=1)
            if candidate is method
        )

"""Promotable type tokens, promoted function forwarding and declared variable, field and parameter types."""

import re
from bisect import bisect_right
from typing import (
    Dict,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
)

from ..array_element_receivers import (
    array_extents_at,
    array_subscript_element_type,
)
from ..auto_placeholders import deduced_auto_type
from ..text_scanners import find_named_type_declarations
from .support import (
    IDENTIFIER_RE,
    MetalStructMethodError,
    _declaration_position,
    _DeclaredTypeT,
    _MetalAddressableValueType,
    _MetalFunctionDefinition,
    _MetalIntegralConstantBinding,
    _MetalPositionedBufferTypes,
    _MetalStructDefinition,
    _MetalStructMethod,
    _MetalSubscriptableType,
    _MetalTypeAliasBinding,
    _PointerPromotionPlan,
    _PromotedFunctionDefinition,
    _PromotedFunctionParameter,
    _PromotedPointerBinding,
)


class MetalDeclaredTypeMixin:
    """Members of ``MetalPreprocessor``: promotable type tokens, promoted function forwarding and declared variable, field and parameter types."""

    def _resolve_promotable_type_token(
        self,
        token: str,
        position: int,
        struct_names: Set[str],
        type_aliases: Dict[str, List[_MetalTypeAliasBinding]],
    ) -> Optional[str]:
        # Resolve an identifier used as a type to the promotable struct it names:
        # directly when it is a struct name, otherwise through the nearest
        # preceding alias at or before `position`. Returns None when the token is
        # neither.
        resolved = self._resolve_struct_type_alias_at(type_aliases, token, position)
        if resolved is None and token in struct_names:
            resolved = token
        return resolved if resolved in struct_names else None

    def _scan_pointer_struct_constructions(
        self,
        code: str,
        promotion_plans: Dict[str, _PointerPromotionPlan],
        skip_spans: List[Tuple[int, int]],
    ) -> Tuple[
        Dict[str, List[_PromotedPointerBinding]],
        List[Tuple[int, int, str]],
        Set[str],
    ]:
        # Find every construction of a promotion-planned struct and, for each,
        # record the receiver's pointer expressions and a replacement that drops
        # the pointer arguments from the constructor call. A struct is promoted
        # only when it is constructed at least once AND every one of its
        # constructions is a clean `S v = S(args)` / `S v(args)` we can rewrite;
        # any other construction (default `S v;`, unknown initializer, argument
        # count mismatch) drops the struct from promotion so it falls back to the
        # ordinary lowering unchanged.
        #
        # Returns (pointer_args, construction_replacements, promoted_names) where
        # pointer_args maps a receiver name to a position-ordered list of
        # (construction_offset, struct_name, [pointer_expr, ...]).
        struct_names = set(promotion_plans)
        # A promotable struct is often constructed through a local `using` alias
        # (`using read_writer_t = ReadWriter_...;` then
        # `read_writer_t rw = read_writer_t(...)`), so resolve such aliases too.
        type_aliases = self._collect_struct_type_aliases(code, struct_names, skip_spans)
        pointer_args: Dict[str, List[_PromotedPointerBinding]] = {}
        lexical_scopes = self._find_lexical_brace_scopes(code)
        # Each pending replacement is tagged with its struct so replacements for a
        # struct that ends up NOT promoted can be discarded (its constructor keeps
        # its pointer arguments).
        pending_replacements: List[Tuple[str, Tuple[int, int, str]]] = []
        constructed: Set[str] = set()
        failed: Set[str] = set()
        i = 0
        n = len(code)
        while i < n:
            ch = code[i]
            if ch in "\"'":
                _literal, consumed = self._read_string(code, i)
                i += consumed
                continue
            if code.startswith("//", i):
                end = code.find("\n", i)
                i = n if end == -1 else end + 1
                continue
            if code.startswith("/*", i):
                end = code.find("*/", i + 2)
                i = n if end == -1 else end + 2
                continue
            span = self._containing_span(i, skip_spans)
            if span is not None:
                i = span[1]
                continue
            if ch.isalpha() or ch == "_":
                ident, consumed = self._read_identifier(code, i)
                ident_end = i + consumed
                target = self._resolve_promotable_type_token(
                    ident, i, struct_names, type_aliases
                )
                if target is not None and not self._is_member_identifier_context(
                    code, i
                ):
                    outcome = self._match_pointer_struct_construction(
                        code, ident, ident_end, promotion_plans[target]
                    )
                    if outcome is not None:
                        kind = outcome[0]
                        if kind == "construction":
                            _, next_index, receiver, ptr_exprs, replacement = outcome
                            constructed.add(target)
                            scope_start, scope_end = self._innermost_lexical_scope(
                                lexical_scopes, i, len(code)
                            )
                            pointer_args.setdefault(receiver, []).append(
                                _PromotedPointerBinding(
                                    declaration_position=i,
                                    scope_start=scope_start,
                                    scope_end=scope_end,
                                    struct_name=target,
                                    pointer_exprs=tuple(ptr_exprs),
                                )
                            )
                            if replacement is not None:
                                pending_replacements.append((target, replacement))
                            i = next_index
                            continue
                        if kind == "fail":
                            failed.add(target)
                            i = outcome[1]
                            continue
                i = ident_end
                continue
            i += 1

        promoted_names = constructed - failed
        # Keep only what belongs to a struct we will actually promote.
        pointer_args = {
            receiver: [
                entry for entry in entries if entry.struct_name in promoted_names
            ]
            for receiver, entries in pointer_args.items()
        }
        pointer_args = {
            receiver: entries for receiver, entries in pointer_args.items() if entries
        }
        construction_replacements = [
            replacement
            for struct_name, replacement in pending_replacements
            if struct_name in promoted_names
        ]
        for entries in pointer_args.values():
            entries.sort(key=lambda item: item.declaration_position)
        return pointer_args, construction_replacements, promoted_names

    def _match_pointer_struct_construction(
        self,
        code: str,
        struct_name: str,
        after_name: int,
        plan: _PointerPromotionPlan,
    ):
        # Classify a `StructName ...` occurrence that begins at a declaration
        # boundary. Returns one of:
        #   ("construction", next_index, receiver, [ptr_expr,...], replacement)
        #   ("fail", next_index)      -- a construction we cannot rewrite
        #   None                      -- not a construction (plain type use)
        # where `replacement` is a (start, end, text) tuple rewriting the
        # constructor call to drop its pointer arguments.
        n = len(code)
        j = after_name
        while j < n and code[j].isspace():
            j += 1
        if j >= n:
            return None
        # A declaration names a receiver identifier after the type.
        if not (code[j].isalpha() or code[j] == "_"):
            return None
        receiver, consumed = self._read_identifier(code, j)
        if not receiver:
            return None
        after_receiver = j + consumed
        k = after_receiver
        while k < n and code[k].isspace():
            k += 1
        if k >= n:
            return None
        if code[k] == "(":
            # Direct initialization: `StructName receiver(args)`.
            return self._build_pointer_construction(
                code, struct_name, receiver, k, plan
            )
        if code[k] == "=":
            # Copy initialization: `StructName receiver = StructName(args)`.
            r = k + 1
            while r < n and code[r].isspace():
                r += 1
            rhs_ident, rhs_consumed = self._read_identifier(code, r)
            if rhs_ident != struct_name:
                # Initialized from something other than the constructor -> we
                # cannot source the pointer expressions.
                return ("fail", after_receiver)
            after_rhs = r + rhs_consumed
            while after_rhs < n and code[after_rhs].isspace():
                after_rhs += 1
            if after_rhs >= n or code[after_rhs] != "(":
                return ("fail", after_receiver)
            return self._build_pointer_construction(
                code, struct_name, receiver, after_rhs, plan
            )
        if code[k] == ";":
            # Default construction cannot supply the pointer members.
            return ("fail", k + 1)
        return None

    def _build_pointer_construction(
        self,
        code: str,
        struct_name: str,
        receiver: str,
        arg_open: int,
        plan: _PointerPromotionPlan,
    ):
        # Parse the constructor-call argument list at `arg_open` (the `(`), map
        # each pointer member to its argument expression, and build a replacement
        # that keeps only the non-pointer arguments.
        arg_close = self._find_matching_delimiter(code, arg_open, "(", ")")
        if arg_close is None:
            return ("fail", arg_open + 1)
        raw_args = code[arg_open + 1 : arg_close]
        arguments = [
            argument.strip()
            for argument in self._split_top_level_commas(raw_args)
            if argument.strip()
        ]
        constructor = plan.constructor
        if len(arguments) != len(constructor.param_names):
            # A default argument or variadic call breaks the positional mapping.
            return ("fail", arg_close + 1)
        pointer_exprs: List[str] = []
        for index in plan.pointer_ctor_arg_indices:
            if index >= len(arguments):
                return ("fail", arg_close + 1)
            pointer_exprs.append(arguments[index])
        drop = plan.dropped_pointer_ctor_arg_indices
        kept = [
            argument for index, argument in enumerate(arguments) if index not in drop
        ]
        # Only the parenthesized argument group is rewritten (dropping the pointer
        # arguments); this is correct for BOTH `S v(args)` (direct init) and
        # `S v = S(args)` (copy init), where the struct name / receiver preceding
        # the arguments is left in place.
        replacement_text = f"({', '.join(kept)})"
        replacement = (arg_open, arg_close + 1, replacement_text)
        return ("construction", arg_close + 1, receiver, pointer_exprs, replacement)

    def _plan_promoted_function_forwarding(
        self,
        code: str,
        promoted_structs: List[_MetalStructDefinition],
        promotion_plans: Dict[str, _PointerPromotionPlan],
        pointer_args: Dict[str, List[_PromotedPointerBinding]],
        struct_spans: List[Tuple[int, int]],
    ) -> List[Tuple[int, int, str]]:
        """Thread promoted pointer roots through ordinary helper functions.

        A promoted struct may be constructed in an entry point and then passed as
        a normal helper parameter. The residual scalar struct cannot recover the
        pointer members by itself, so every such parameter receives one explicit
        companion pointer per promoted member. Calls forward either construction
        roots or the current helper parameter's companions. Bindings are lexical,
        which keeps same-named parameters and nested constructions independent.
        """

        promoted_by_name = {struct.name: struct for struct in promoted_structs}
        excluded_spans = sorted(
            set(struct_spans + self._find_template_declaration_spans(code))
        )
        functions = self._find_non_template_function_definitions(code, excluded_spans)
        if not functions:
            return []

        replacements: List[Tuple[int, int, str]] = []
        definitions: List[_PromotedFunctionDefinition] = []
        all_arities: Dict[Tuple[str, int], int] = {}
        for function in functions:
            header_start = function.span[0]
            header_end = function.body_span[0] - 1
            header = code[header_start:header_end]
            parameter_span = self._function_parameter_list_span(header)
            if parameter_span is None:
                continue
            parameter_open, parameter_close = parameter_span
            parameter_spans = self._top_level_comma_item_spans(
                header, parameter_open + 1, parameter_close
            )
            if len(parameter_spans) == 1:
                only_start, only_end = parameter_spans[0]
                if header[only_start:only_end].strip() == "void":
                    parameter_spans = []
            parameters = [header[start:end].strip() for start, end in parameter_spans]
            all_arities[(function.name, len(parameters))] = (
                all_arities.get((function.name, len(parameters)), 0) + 1
            )

            occupied_names = set(
                IDENTIFIER_RE.findall(code[function.span[0] : function.span[1]])
            )
            promoted_parameters: List[_PromotedFunctionParameter] = []
            for index, ((start, end), parameter) in enumerate(
                zip(parameter_spans, parameters)
            ):
                absolute_start = header_start + start
                declaration, default = self._split_top_level_assignment(parameter)
                value_type = self._function_parameter_value_type(declaration)
                canonical_type = self._canonicalize_type_aliases_at(
                    value_type,
                    self._source_type_alias_bindings,
                    absolute_start,
                )
                normalized_type = self._normalize_inferred_type(
                    canonical_type or value_type
                )
                normalized_type = re.sub(r"^(?:struct|class)\s+", "", normalized_type)
                if normalized_type not in promoted_by_name:
                    continue
                if default is not None or "..." in declaration:
                    raise MetalStructMethodError(
                        "Metal promoted pointer members cannot be forwarded "
                        "through a defaulted or variadic struct parameter.",
                        struct_name=normalized_type,
                        method_name=function.name,
                        requested_signature=header.strip(),
                        suggested_action=(
                            "pass the promoted struct explicitly through a "
                            "non-variadic helper parameter"
                        ),
                        source_location=self._source_location_for_offsets(
                            code, absolute_start, header_start + end
                        ),
                        missing_capabilities=("metal.promoted-pointer-forwarding",),
                        reason="promoted-pointer-forwarding-unsupported-parameter",
                    )
                parameter_name = self._declared_data_member_name(declaration)
                if parameter_name is None or parameter_name == "void":
                    raise MetalStructMethodError(
                        "Metal promoted pointer forwarding requires a named "
                        "struct parameter.",
                        struct_name=normalized_type,
                        method_name=function.name,
                        requested_signature=header.strip(),
                        suggested_action="give the promoted struct parameter a name",
                        source_location=self._source_location_for_offsets(
                            code, absolute_start, header_start + end
                        ),
                        missing_capabilities=("metal.promoted-pointer-forwarding",),
                        reason="promoted-pointer-forwarding-unnamed-parameter",
                    )

                struct = promoted_by_name[normalized_type]
                plan = promotion_plans[normalized_type]
                companion_names: List[str] = []
                companion_declarations: List[str] = []
                for member in plan.pointer_members:
                    base_name = f"crosstl_ptr_{parameter_name}_{member.name}"
                    companion_name = base_name
                    suffix = 2
                    while companion_name in occupied_names:
                        companion_name = f"{base_name}_{suffix}"
                        suffix += 1
                    occupied_names.add(companion_name)
                    member_type = self._canonicalize_struct_scoped_type(
                        member.type_text, struct, None
                    )
                    companion_names.append(companion_name)
                    companion_declarations.append(f"{member_type} {companion_name}")
                promoted_parameters.append(
                    _PromotedFunctionParameter(
                        index=index,
                        name=parameter_name,
                        struct_name=normalized_type,
                        companion_names=tuple(companion_names),
                        companion_declarations=tuple(companion_declarations),
                    )
                )

            if not promoted_parameters:
                continue
            promoted_by_index = {
                parameter.index: parameter for parameter in promoted_parameters
            }
            expanded_parameters: List[str] = []
            for index, parameter in enumerate(parameters):
                expanded_parameters.append(parameter)
                promoted = promoted_by_index.get(index)
                if promoted is not None:
                    expanded_parameters.extend(promoted.companion_declarations)
            replacements.append(
                (
                    header_start + parameter_open + 1,
                    header_start + parameter_close,
                    ", ".join(expanded_parameters),
                )
            )
            definition = _PromotedFunctionDefinition(
                function=function,
                parameter_count=len(parameters),
                promoted_parameters=tuple(promoted_parameters),
            )
            definitions.append(definition)
            for parameter in promoted_parameters:
                pointer_args.setdefault(parameter.name, []).append(
                    _PromotedPointerBinding(
                        declaration_position=function.body_span[0],
                        scope_start=function.body_span[0],
                        scope_end=function.body_span[1],
                        struct_name=parameter.struct_name,
                        pointer_exprs=parameter.companion_names,
                    )
                )

        if not definitions:
            return replacements
        for entries in pointer_args.values():
            entries.sort(key=lambda binding: binding.declaration_position)
        replacements.extend(
            self._rewrite_promoted_function_calls(
                code,
                functions,
                definitions,
                all_arities,
                pointer_args,
            )
        )
        return replacements

    def _rewrite_promoted_function_calls(
        self,
        code: str,
        functions: List[_MetalFunctionDefinition],
        definitions: List[_PromotedFunctionDefinition],
        all_arities: Dict[Tuple[str, int], int],
        pointer_args: Dict[str, List[_PromotedPointerBinding]],
    ) -> List[Tuple[int, int, str]]:
        by_name_and_arity: Dict[Tuple[str, int], List[_PromotedFunctionDefinition]] = {}
        for definition in definitions:
            by_name_and_arity.setdefault(
                (definition.function.name, definition.parameter_count), []
            ).append(definition)

        replacements: List[Tuple[int, int, str]] = []
        for caller in functions:
            cursor = caller.body_span[0]
            body_end = caller.body_span[1]
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

                name, consumed = self._read_identifier(code, cursor)
                name_end = cursor + consumed
                if self._is_member_identifier_context(code, cursor):
                    cursor = name_end
                    continue
                argument_open = name_end
                while argument_open < body_end and code[argument_open].isspace():
                    argument_open += 1
                if argument_open >= body_end or code[argument_open] != "(":
                    cursor = name_end
                    continue
                argument_close = self._find_matching_delimiter(
                    code, argument_open, "(", ")"
                )
                if argument_close is None or argument_close > body_end:
                    cursor = name_end
                    continue
                argument_spans = self._top_level_comma_item_spans(
                    code, argument_open + 1, argument_close
                )
                key = (name, len(argument_spans))
                candidates = by_name_and_arity.get(key, ())
                if not candidates:
                    cursor = name_end
                    continue

                viable_plans: List[Tuple[Tuple[int, Tuple[str, ...]], ...]] = []
                for candidate in candidates:
                    plan: List[Tuple[int, Tuple[str, ...]]] = []
                    viable = True
                    for parameter in candidate.promoted_parameters:
                        if parameter.index >= len(argument_spans):
                            viable = False
                            break
                        argument_start, argument_end = argument_spans[parameter.index]
                        expression = self._strip_static_outer_parentheses(
                            code[argument_start:argument_end]
                        )
                        if re.fullmatch(r"[A-Za-z_]\w*", expression) is None:
                            viable = False
                            break
                        resolved = self._resolve_promoted_receiver(
                            pointer_args, expression, cursor
                        )
                        if resolved is None or resolved[0] != parameter.struct_name:
                            viable = False
                            break
                        plan.append((argument_end, tuple(resolved[1])))
                    if viable:
                        viable_plans.append(tuple(plan))

                unique_plans = list(dict.fromkeys(viable_plans))
                if not unique_plans:
                    if all_arities.get(key, 0) == len(candidates):
                        candidate = candidates[0]
                        promoted = candidate.promoted_parameters[0]
                        call_text = code[cursor : argument_close + 1]
                        raise MetalStructMethodError(
                            "Metal cannot resolve the promoted pointer roots "
                            f"needed by helper call '{call_text}'.",
                            struct_name=promoted.struct_name,
                            method_name=name,
                            requested_signature=call_text,
                            suggested_action=(
                                "pass a directly constructed promoted object or "
                                "another promoted helper parameter"
                            ),
                            source_location=self._source_location_for_offsets(
                                code, cursor, argument_close + 1
                            ),
                            missing_capabilities=("metal.promoted-pointer-forwarding",),
                            reason="promoted-pointer-forwarding-unresolved",
                        )
                    cursor = name_end
                    continue
                if len(unique_plans) != 1:
                    call_text = code[cursor : argument_close + 1]
                    raise MetalStructMethodError(
                        "Metal promoted pointer forwarding is ambiguous for "
                        f"helper call '{call_text}'.",
                        method_name=name,
                        requested_signature=call_text,
                        suggested_action=(
                            "disambiguate the helper overload or its promoted "
                            "struct argument"
                        ),
                        source_location=self._source_location_for_offsets(
                            code, cursor, argument_close + 1
                        ),
                        missing_capabilities=("metal.promoted-pointer-forwarding",),
                        reason="promoted-pointer-forwarding-ambiguous",
                    )
                for insertion_position, pointer_exprs in unique_plans[0]:
                    replacements.append(
                        (
                            insertion_position,
                            insertion_position,
                            ", " + ", ".join(pointer_exprs),
                        )
                    )
                cursor = name_end
        return replacements

    def _top_level_comma_item_spans(
        self, text: str, start: int, end: int
    ) -> List[Tuple[int, int]]:
        """Return trimmed top-level comma-delimited spans in ``text[start:end]``."""

        spans: List[Tuple[int, int]] = []
        item_start = start
        paren_depth = bracket_depth = brace_depth = angle_depth = 0
        cursor = start
        while cursor < end:
            if text[cursor] in "\"'":
                _literal, consumed = self._read_string(text, cursor)
                cursor += consumed
                continue
            if text.startswith("//", cursor):
                line_end = text.find("\n", cursor, end)
                cursor = end if line_end == -1 else line_end + 1
                continue
            if text.startswith("/*", cursor):
                comment_end = text.find("*/", cursor + 2, end)
                cursor = end if comment_end == -1 else comment_end + 2
                continue
            character = text[cursor]
            if character == "(":
                paren_depth += 1
            elif character == ")":
                paren_depth = max(0, paren_depth - 1)
            elif character == "[":
                bracket_depth += 1
            elif character == "]":
                bracket_depth = max(0, bracket_depth - 1)
            elif character == "{":
                brace_depth += 1
            elif character == "}":
                brace_depth = max(0, brace_depth - 1)
            elif character == "<":
                angle_depth += 1
            elif character == ">":
                angle_depth = max(0, angle_depth - 1)
            elif (
                character == ","
                and paren_depth == 0
                and bracket_depth == 0
                and brace_depth == 0
                and angle_depth == 0
            ):
                item_end = cursor
                while item_start < item_end and text[item_start].isspace():
                    item_start += 1
                while item_end > item_start and text[item_end - 1].isspace():
                    item_end -= 1
                if item_start < item_end:
                    spans.append((item_start, item_end))
                item_start = cursor + 1
            cursor += 1
        item_end = end
        while item_start < item_end and text[item_start].isspace():
            item_start += 1
        while item_end > item_start and text[item_end - 1].isspace():
            item_end -= 1
        if item_start < item_end:
            spans.append((item_start, item_end))
        return spans

    def _rewrite_promoted_call_sites(
        self,
        code: str,
        promoted_structs: List[_MetalStructDefinition],
        promotion_plans: Dict[str, _PointerPromotionPlan],
        pointer_args: Dict[str, List[_PromotedPointerBinding]],
        skip_spans: List[Tuple[int, int]],
    ) -> List[Tuple[int, int, str]]:
        # Rewrite `receiver.method(args)` for a promoted-struct receiver to
        # `S__method(receiver, <pointer exprs>, args)`, forwarding the pointer
        # expressions captured at the receiver's construction.
        methods_by_struct: Dict[str, Dict[str, _MetalStructMethod]] = {}
        structs_by_name = {struct.name: struct for struct in promoted_structs}
        for struct in promoted_structs:
            methods_by_struct[struct.name] = {
                method.name: method for method in struct.methods
            }
        replacements: List[Tuple[int, int, str]] = []
        i = 0
        n = len(code)
        while i < n:
            ch = code[i]
            if ch in "\"'":
                _literal, consumed = self._read_string(code, i)
                i += consumed
                continue
            if code.startswith("//", i):
                end = code.find("\n", i)
                i = n if end == -1 else end + 1
                continue
            if code.startswith("/*", i):
                end = code.find("*/", i + 2)
                i = n if end == -1 else end + 2
                continue
            span = self._containing_span(i, skip_spans)
            if span is not None:
                i = span[1]
                continue
            if ch.isalpha() or ch == "_":
                ident, consumed = self._read_identifier(code, i)
                ident_end = i + consumed
                if not self._is_member_identifier_context(code, i):
                    resolved = self._resolve_promoted_receiver(pointer_args, ident, i)
                    if resolved is not None:
                        struct_name, pointer_exprs = resolved
                        rewrite = self._try_rewrite_promoted_instance_call(
                            code,
                            ident,
                            ident_end,
                            structs_by_name[struct_name],
                            pointer_exprs,
                            methods_by_struct.get(struct_name, {}),
                        )
                        if rewrite is not None:
                            end, replacement = rewrite
                            replacements.append((i, end, replacement))
                            i = end
                            continue
                i = ident_end
                continue
            i += 1
        return replacements

    def _try_rewrite_promoted_instance_call(
        self,
        code: str,
        receiver: str,
        ident_end: int,
        struct: _MetalStructDefinition,
        pointer_exprs: List[str],
        methods_by_name: Dict[str, _MetalStructMethod],
    ) -> Optional[Tuple[int, str]]:
        n = len(code)
        j = ident_end
        while j < n and code[j].isspace():
            j += 1
        if j >= n or code[j] != ".":
            return None
        k = j + 1
        while k < n and code[k].isspace():
            k += 1
        member, consumed = self._read_identifier(code, k)
        if not member:
            return None
        after = k + consumed
        while after < n and code[after].isspace():
            after += 1
        if after >= n or code[after] != "(":
            return None
        method = methods_by_name.get(member)
        if method is None or method.is_static:
            return None
        arg_open = after
        arg_close = self._find_matching_delimiter(code, arg_open, "(", ")")
        if arg_close is None:
            return None
        self._reject_reference_returning_method_call(
            code,
            struct,
            method,
            arg_open,
            arg_close,
        )
        args = code[arg_open + 1 : arg_close].strip()
        pieces = [receiver]
        pieces.extend(pointer_exprs)
        if args:
            pieces.append(args)
        replacement = f"{method.free_name}({', '.join(pieces)})"
        return arg_close + 1, replacement

    def _resolve_promoted_receiver(
        self,
        pointer_args: Dict[str, List[_PromotedPointerBinding]],
        name: str,
        position: int,
    ) -> Optional[Tuple[str, List[str]]]:
        # Resolve only a declaration that is both preceding and lexically visible
        # at the use site. This prevents same-named helper parameters or nested
        # constructions from leaking pointer roots into another function/scope.
        entries = pointer_args.get(name)
        if not entries:
            return None
        best: Optional[_PromotedPointerBinding] = None
        for binding in entries:
            if binding.declaration_position > position:
                break
            if not (binding.scope_start <= position < binding.scope_end):
                continue
            if (
                best is None
                or binding.declaration_position > best.declaration_position
                or (
                    binding.declaration_position == best.declaration_position
                    and binding.scope_start > best.scope_start
                )
            ):
                best = binding
        if best is None:
            return None
        return best.struct_name, list(best.pointer_exprs)

    def _collect_struct_variable_types(
        self,
        code: str,
        struct_names: Set[str],
        struct_spans: List[Tuple[int, int]],
        *,
        include_indirect: bool = False,
    ) -> Dict[str, List[Tuple[int, str]]]:
        # Map each variable name to the POSITION-ORDERED list of struct
        # declarations that introduce it outside struct definitions. Direct
        # declarations are always recorded; pointer and reference declarations
        # are included only for field-carrier inference. Returning every
        # declaration with its source offset makes call-site resolution
        # deterministic and selects the nearest preceding declaration.
        declarations: Dict[str, List[Tuple[int, str]]] = {}
        # One position-ordered scan for every struct name keeps each per-name
        # list sorted by offset without depending on set iteration order.
        for struct_name, match in find_named_type_declarations(code, struct_names):
            if self._containing_span(match.start(), struct_spans) is not None:
                continue
            # Skip a match that is itself a member access (`x.S var` cannot
            # happen, but guard scoped `Ns::S` by checking the preceding char).
            preceding = match.start() - 1
            while preceding >= 0 and code[preceding].isspace():
                preceding -= 1
            if preceding >= 0 and code[preceding] in ".>":
                continue
            indirection = match.group("indirection") or ""
            if indirection and not include_indirect:
                continue
            declarations.setdefault(match.group("name"), []).append(
                (match.start(), f"{struct_name}{indirection}")
            )
        return declarations

    def _collect_aliased_struct_variable_types(
        self,
        code: str,
        type_aliases: Dict[str, List[_MetalTypeAliasBinding]],
        struct_spans: List[Tuple[int, int]],
    ) -> Dict[str, List[Tuple[int, str]]]:
        declarations: Dict[str, List[Tuple[int, str]]] = {}
        for alias, match in find_named_type_declarations(code, type_aliases):
            if self._containing_span(match.start(), struct_spans) is not None:
                continue
            resolved = self._resolve_struct_type_alias_at(
                type_aliases, alias, match.start()
            )
            if resolved is None:
                continue
            indirection = match.group("indirection") or ""
            declarations.setdefault(match.group("name"), []).append(
                (match.start(), f"{resolved}{indirection}")
            )
        return declarations

    def _resolve_declared_type_at(
        self,
        declarations: Dict[str, List[Tuple[int, _DeclaredTypeT]]],
        name: str,
        position: int,
    ) -> Optional[_DeclaredTypeT]:
        # Resolve the nearest preceding binding whose recorded scope is active.
        # A later declaration cannot type an earlier use;
        # accepting it can leak a same-named parameter from another function
        # into template deduction. Deterministic regardless of hash seed.
        entries = declarations.get(name)
        if not entries:
            return None
        # Every declaration list is position-ordered, so the candidates are the
        # prefix up to ``position``; scanning it backwards stops at the nearest
        # active binding instead of visiting every earlier declaration of a
        # name that recurs in each function of a large translation unit.
        for index in range(
            bisect_right(entries, position, key=_declaration_position) - 1, -1, -1
        ):
            declared_type = entries[index][1]
            if (
                isinstance(declared_type, _MetalAddressableValueType)
                and declared_type.scope is not None
                and not declared_type.scope[0] <= position < declared_type.scope[1]
            ):
                continue
            return declared_type
        return None

    def _flatten_types_at(
        self,
        declarations: Dict[str, List[Tuple[int, _DeclaredTypeT]]],
        position: int,
    ) -> Dict[str, _DeclaredTypeT]:
        # Flatten a position-ordered declaration map to a name -> type view valid
        # at `position` (nearest preceding declaration wins per name).
        flattened: Dict[str, _DeclaredTypeT] = {}
        for name in declarations:
            resolved = self._resolve_declared_type_at(declarations, name, position)
            if resolved is not None:
                if isinstance(resolved, _MetalAddressableValueType) and (
                    resolved.scope is None
                    or not resolved.scope[0] <= position < resolved.scope[1]
                ):
                    resolved = str(resolved)
                flattened[name] = resolved
        return flattened

    def _struct_field_types_at(
        self,
        variable_types: Optional[Dict[str, List[Tuple[int, str]]]],
        structs_by_name: Optional[Dict[str, _MetalStructDefinition]],
        position: int,
    ) -> Dict[str, Dict[str, str]]:
        # Build a call-site view of each struct carrier's field types. Pointer
        # carriers use an `obj->` key; objects and references use `obj`, allowing
        # member inference to require the matching access operator. Multi-level
        # indirection is intentionally excluded.
        if not variable_types or not structs_by_name:
            return {}
        field_types: Dict[str, Dict[str, str]] = {}
        for name in variable_types:
            struct_type = self._resolve_declared_type_at(variable_types, name, position)
            if struct_type is None:
                continue
            indirection_match = re.search(r"(?P<indirection>[*&]+)\s*$", struct_type)
            indirection = (
                indirection_match.group("indirection")
                if indirection_match is not None
                else ""
            )
            if indirection not in {"", "*", "&"}:
                continue
            base_type = (
                struct_type[: indirection_match.start()].strip()
                if indirection_match is not None
                else struct_type
            )
            struct = structs_by_name.get(self._normalize_inferred_type(base_type))
            if struct is None or not struct.data_member_types:
                continue
            access_key = f"{name}->" if indirection == "*" else name
            field_types[access_key] = {
                member: self._canonicalize_struct_scoped_type(
                    declared_type, struct, structs_by_name
                )
                for member, declared_type in struct.data_member_types.items()
            }
        return field_types

    # ------------------------------------------------------------------ #
    # Conservative call-argument type inference (template member methods). #
    # ------------------------------------------------------------------ #
    # The rules below NEVER guess: each returns None for any expression shape it
    # does not recognize, and the caller turns a single un-inferable argument
    # into a clean translation failure rather than emitting a dangling call.

    def _collect_buffer_element_types(
        self, code: str, struct_spans: List[Tuple[int, int]]
    ) -> _MetalPositionedBufferTypes:
        # Map each subscriptable name to the POSITION-ORDERED list of declarations
        # that give it an ELEMENT type, so `name[expr]` can be element-typed. Three
        # declaration shapes contribute:
        #   * pointer/buffer parameters and locals: `device [const] T* buf`,
        #     `constant T* p`, `thread U* q` -> element T (function parameters AND
        #     local pointer variables).
        #   * array locals/parameters: `U totals[4]`, `threadgroup U sh[32]`,
        #     `const float vals[N]` -> element U/float (the dominant shape in MLX
        #     reduce, where reduction accumulators are stack arrays).
        # Struct bodies are excluded so member declarations do not pollute the map
        # (struct fields are typed separately via `data_member_types`).
        element_types: _MetalPositionedBufferTypes = {}

        def record(
            name: str,
            element_type: str,
            position: int,
            pointer_type: Optional[str] = None,
        ) -> None:
            normalized = self._normalize_inferred_type(element_type)
            if not normalized:
                return
            normalized_pointer = self._normalize_known_address_space_pointer_type(
                pointer_type or ""
            )
            element_types.setdefault(name, []).append(
                (
                    position,
                    _MetalSubscriptableType(normalized, normalized_pointer),
                )
            )

        pointer_pattern = re.compile(
            r"\b(?P<type>(?:(?:const|constexpr|volatile)\s+)*"
            r"(?:device|constant|threadgroup|thread)\b"
            r"[^;,()\[\]{}]*?)\*\s*"
            r"(?P<name>[A-Za-z_][A-Za-z0-9_]*)"
        )
        for match in pointer_pattern.finditer(code):
            if self._containing_span(match.start(), struct_spans) is not None:
                continue
            record(
                match.group("name"),
                match.group("type"),
                match.start(),
                f"{match.group('type')}*",
            )

        # Array declarations: `<type> name[extent]` for locals and parameters.
        # The leading anchor keeps consecutive declarations from cannibalizing
        # each other; the type group allows address-space/cv qualifiers. Only a
        # RECOGNIZED element type (scalar/vector or a known struct/union) is
        # accepted so a non-declaration that happens to match the shape — e.g.
        # `return totals[i]` (type token `return`) or `else block[i]` — never
        # records a bogus element type (the never-guess contract).
        recognized_aggregates = self._aggregate_type_names(code, struct_spans)
        function_body_spans = [
            function.body_span
            for function in self._find_non_template_function_definitions(
                code, struct_spans
            )
        ]
        array_pattern = re.compile(
            r"(?:(?<=[;{}(,])|^)\s*"
            r"(?P<type>(?:const\s+|constexpr\s+|volatile\s+|thread\s+|"
            r"threadgroup\s+|device\s+|constant\s+)*"
            r"[A-Za-z_][A-Za-z0-9_]*)\s+"
            r"(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*\[(?!\[)",
            re.MULTILINE,
        )
        for match in array_pattern.finditer(code):
            if self._containing_span(match.start(), struct_spans) is not None:
                continue
            normalized = self._normalize_inferred_type(match.group("type"))
            if (
                normalized not in self._METAL_SCALAR_VECTOR_TYPES
                and normalized not in recognized_aggregates
            ):
                # A stack-array declaration may use a source alias whose unique
                # visible target is a retained aggregate (for example MLX's
                # `complex64_t totals[N]`). Resolve it through the same lexical,
                # cycle-detecting alias contract used for overload matching. A
                # forward, shadowed, cyclic, or still-unrecognized target stays
                # untyped rather than being guessed from the call site.
                resolved = self._canonicalize_type_aliases_at(
                    normalized,
                    getattr(self, "_source_type_alias_bindings", {}),
                    match.start(),
                )
                normalized = self._normalize_inferred_type(resolved or "")
                if (
                    normalized not in self._METAL_SCALAR_VECTOR_TYPES
                    and normalized not in recognized_aggregates
                ):
                    continue
            extents = array_extents_at(code, match.end() - 1)
            pointer_type = f"{match.group('type')}*"
            if (
                self._normalize_known_address_space_pointer_type(pointer_type) is None
                and self._containing_span(match.start(), function_body_spans)
                is not None
            ):
                pointer_type = f"thread {match.group('type')}*"
            record(
                match.group("name"),
                array_subscript_element_type(normalized, extents),
                match.start(),
                pointer_type,
            )

        for entries in element_types.values():
            entries.sort(key=lambda item: item[0])
        return element_types

    def _collect_local_variable_types(
        self,
        code: str,
        struct_spans: List[Tuple[int, int]],
        type_aliases: Optional[Dict[str, List[_MetalTypeAliasBinding]]] = None,
        structs: Optional[Sequence[_MetalStructDefinition]] = None,
    ) -> Dict[str, List[Tuple[int, str]]]:
        # Map each bare local/parameter name to the POSITION-ORDERED list of its
        # declared types. Two declaration shapes contribute:
        #   * scalar/vector locals: `T name;` / `T name = ...;` where T is a
        #     recognized Metal scalar/vector type.
        #   * union/struct locals: `bool4_or_uint update;` and any `Name var;`
        #     whose type names a struct/union declared in this source — recorded
        #     so a bare such local is inferable (and its members are resolvable).
        # Position ordering makes resolution deterministic and per-scope (nearest
        # preceding declaration wins), so same-named locals in sibling kernels do
        # not collapse to one PYTHONHASHSEED-dependent type.
        local_types: Dict[str, List[Tuple[int, str]]] = {}
        recognized_aggregates = self._aggregate_type_names(code, struct_spans)
        lexical_scopes = self._find_lexical_brace_scopes(code)
        function_body_spans = [
            function.body_span
            for function in self._find_non_template_function_definitions(
                code, struct_spans
            )
        ]
        # A zero-width leading anchor (start-of-string or a statement/scope
        # boundary) keeps consecutive declarations like `float acc=...; float
        # x=...;` from cannibalizing each other's anchor.
        pattern = re.compile(
            r"(?:(?<=[;{}()])|^)\s*"
            r"(?P<type>(?:const\s+|constexpr\s+|thread\s+|threadgroup\s+)*"
            r"(?:::)?[A-Za-z_][A-Za-z0-9_]*(?:::[A-Za-z_][A-Za-z0-9_]*)*"
            r"(?:\s*<[^;\n={}()]+>)?)\s+"
            r"(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*(?=[=;])",
            re.MULTILINE,
        )
        for match in pattern.finditer(code):
            if self._containing_span(match.start(), struct_spans) is not None:
                continue
            raw_type = match.group("type")
            normalized = self._normalize_inferred_type(raw_type)
            if (
                not self._is_metal_scalar_or_vector_type(normalized)
                and normalized not in recognized_aggregates
                and type_aliases
            ):
                # A local alias may name a struct-scoped type, as in
                # `using frag_t = typename Frag::frag_type; frag_t value;`.
                resolved = self._resolve_type_aliases_at(
                    raw_type, type_aliases, match.start()
                )
                if structs:
                    resolved = self._canonicalize_qualified_struct_type_aliases(
                        resolved, list(structs)
                    )
                    # `typename Owner::template alias<U>` names a member
                    # alias template; expand it to the type it denotes.
                    resolved = self._canonicalize_qualified_struct_alias_templates(
                        resolved, list(structs)
                    )
                if resolved != raw_type:
                    raw_type = resolved
                    normalized = self._normalize_inferred_type(resolved)
            if (
                not self._is_metal_scalar_or_vector_type(normalized)
                and normalized not in recognized_aggregates
            ):
                continue
            pointer_type = None
            scope = self._containing_span(match.start(), function_body_spans)
            if scope is not None:
                scope = self._innermost_lexical_scope(
                    lexical_scopes, match.start(), len(code)
                )
                storage_type = re.sub(r"\bconstexpr\b", "const", raw_type)
                if not set(IDENTIFIER_RE.findall(storage_type)).intersection(
                    {"thread", "threadgroup"}
                ):
                    storage_type = f"thread {storage_type}"
                pointer_type = self._normalize_known_address_space_pointer_type(
                    f"{storage_type}*"
                )
            local_types.setdefault(match.group("name"), []).append(
                (
                    match.start(),
                    _MetalAddressableValueType(normalized, pointer_type, scope),
                )
            )
        for entries in local_types.values():
            entries.sort(key=lambda item: item[0])
        return local_types

    def _aggregate_type_names(
        self, code: str, struct_spans: List[Tuple[int, int]]
    ) -> Set[str]:
        # Names of `struct`/`class`/`union` types DEFINED in this source. A bare
        # local of such a type is inferable, and (for structs) its members are
        # resolvable via `data_member_types`. Unions are included so reduce's
        # `bool4_or_uint update;` typed local is recognized.
        names: Set[str] = set()
        for match in re.finditer(
            r"\b(?:struct|class|union)\s+([A-Za-z_][A-Za-z0-9_]*)", code
        ):
            names.add(match.group(1))
        return names

    def _collect_function_parameter_types(
        self,
        code: str,
        struct_spans: List[Tuple[int, int]],
        buffer_element_types: _MetalPositionedBufferTypes,
        local_variable_types: Dict[str, List[Tuple[int, str]]],
    ) -> None:
        # Augment the (position-ordered) type maps with each FUNCTION PARAMETER,
        # so a call argument that is a parameter (or a subscript of one) resolves
        # via the parameter's declared type. Pointer/array parameters are already
        # captured by `_collect_buffer_element_types` (they match its patterns in
        # the header text); this adds SCALAR/VECTOR and struct/union parameters,
        # which the local-declaration scanner misses because a parameter is
        # comma-terminated rather than `=`/`;`-terminated. Each parameter is
        # recorded at the function BODY-START offset so nearest-preceding
        # resolution scopes it to that body. Struct method bodies are excluded
        # (their parameters are handled by the member-method lowering itself).
        recognized_aggregates = self._aggregate_type_names(code, struct_spans)
        for function in self._find_non_template_function_definitions(
            code, struct_spans
        ):
            header = code[function.span[0] : function.body_span[0] - 1]
            parameter_span = self._function_parameter_list_span(header)
            if parameter_span is None:
                continue
            paren_start, paren_end = parameter_span
            parameter_text = header[paren_start + 1 : paren_end]
            body_start = function.body_span[0]
            for parameter in self._split_top_level_commas(parameter_text):
                if not parameter.strip():
                    continue
                parameter_without_attributes = self._strip_metal_attributes(parameter)
                name = self._declared_data_member_name(parameter_without_attributes)
                if not name:
                    continue
                subscriptable = self._pointer_or_array_parameter_type(parameter)
                if subscriptable is not None:
                    buffer_element_types.setdefault(name, []).append(
                        (body_start, subscriptable)
                    )
                    continue
                scalar = self._function_parameter_value_type(
                    parameter_without_attributes
                )
                if (
                    self._is_metal_scalar_or_vector_type(scalar)
                    or scalar in recognized_aggregates
                ):
                    local_variable_types.setdefault(name, []).append(
                        (body_start, scalar)
                    )
        for entries in buffer_element_types.values():
            entries.sort(key=lambda item: item[0])
        for entries in local_variable_types.values():
            entries.sort(key=lambda item: item[0])

    def _collect_auto_local_variable_types(
        self,
        code: str,
        struct_spans: List[Tuple[int, int]],
        buffer_element_types: _MetalPositionedBufferTypes,
        local_variable_types: Dict[str, List[Tuple[int, str]]],
        structs_by_name: Optional[Dict[str, "_MetalStructDefinition"]],
        *,
        type_aliases: Optional[Dict[str, List[_MetalTypeAliasBinding]]] = None,
        local_integral_constants: Optional[
            Dict[str, List[_MetalIntegralConstantBinding]]
        ] = None,
    ) -> None:
        # Type `auto name = <initializer>;` locals from the inferred type of their
        # initializer, so a later call argument that uses such a local resolves.
        # MLX's complex unary ops build their template-member-call arguments from
        # `auto` temporaries (e.g. `auto i = complex64_t{0.0, 1.0};`), which the
        # scalar/aggregate scanner cannot type because the declarator names no
        # type. Declarations are processed in source order and each inferred
        # binding is recorded immediately, so one `auto` local may depend on an
        # earlier one. An unresolved initializer still shadows older bindings.
        # `auto*` and `const auto*` take a pointer initializer's type
        # ([dcl.type.auto.deduct]); `auto&` declarators are skipped.
        lexical_scopes = self._find_lexical_brace_scopes(code)
        pattern = re.compile(
            r"(?:(?<=[;{}()])|^)\s*(?P<specifier>(?:(?:static|constexpr|const)\s+)*auto(?=\s)"
            r"|(?:const\s+)?auto\s*\*)"
            r"\s*(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*=(?!=)",
            re.MULTILINE,
        )
        for match in pattern.finditer(code):
            if self._containing_span(match.start(), struct_spans) is not None:
                continue
            initializer_start = match.end()
            statement_end = self._statement_end(code, initializer_start)
            if statement_end is None:
                continue
            initializer = code[initializer_start : statement_end - 1].strip()
            if not initializer:
                continue
            position = match.start()
            buffer_view = self._flatten_types_at(buffer_element_types, position)
            local_view = self._flatten_types_at(local_variable_types, position)
            struct_field_types = self._struct_field_types_at(
                local_variable_types, structs_by_name, position
            )
            inferred = self._infer_argument_type(
                initializer,
                buffer_view,
                local_view,
                struct_field_types,
                structs_by_name,
            )
            if inferred is None and type_aliases:
                inferred = self._aliased_constructor_type_at(
                    initializer,
                    position,
                    type_aliases,
                    local_integral_constants or {},
                )
                if inferred not in (structs_by_name or {}) and not (
                    self._is_metal_scalar_or_vector_type(inferred or "")
                ):
                    inferred = None
            if inferred and type_aliases:
                inferred = self._concrete_type_alias_at(
                    inferred, position, type_aliases, local_integral_constants or {}
                )
            if match.group("specifier") != "auto":
                inferred = deduced_auto_type(match.group("specifier"), inferred)
                if inferred == "auto":
                    inferred = None
            normalized = self._normalize_inferred_expression_type(inferred or "")
            entries = local_variable_types.setdefault(match.group("name"), [])
            scope = self._innermost_lexical_scope(
                lexical_scopes, match.start("name"), len(code)
            )
            entries.append(
                (position, _MetalAddressableValueType(normalized or "", scope=scope))
            )
            entries.sort(key=lambda item: item[0])

    def _pointer_or_array_parameter_element_type(self, parameter: str) -> Optional[str]:
        # Element type of a pointer (`device T* p`) or array (`T p[N]`) parameter,
        # or None when the parameter is neither (so the caller treats it as a
        # plain value parameter). Conservative: returns None unless a `*` or `[`
        # declarator is present.
        subscriptable = self._pointer_or_array_parameter_type(parameter)
        return subscriptable.element_type if subscriptable is not None else None

    def _pointer_or_array_parameter_type(
        self, parameter: str
    ) -> Optional[_MetalSubscriptableType]:
        # Preserve the full pointer spelling only when one Metal address space is
        # explicit. Value inference still consumes the normalized element type;
        # address-of inference separately requires the proven pointer spelling.
        text = self._strip_top_level_default_value(parameter)
        text = self._strip_metal_attributes(text).strip()
        if not text:
            return None
        # `thread T (&vals)[N]` binds a reference to an array: subscripts yield
        # T exactly as for the array declarator `thread T vals[N]`.
        reference = re.fullmatch(
            r"(?P<type>[^()\[\]]+?)\s*\(\s*&\s*[A-Za-z_][A-Za-z0-9_]*\s*\)"
            r"\s*(?P<extents>(?:\[[^\[\]]*\]\s*)+)",
            text,
        )
        type_text = None
        if reference is not None:
            type_text = reference.group("type").strip()
            extents = array_extents_at(reference.group("extents"), 0)
        elif "[" in text:
            type_text = text[: text.find("[")]
            extents = array_extents_at(text, text.find("["))
            type_text = re.sub(r"\b[A-Za-z_][A-Za-z0-9_]*\s*$", "", type_text).strip()
        if type_text is not None:
            element_type = self._normalize_inferred_type(type_text)
            if not element_type:
                return None
            return _MetalSubscriptableType(
                array_subscript_element_type(element_type, extents),
                self._normalize_known_address_space_pointer_type(f"{type_text}*"),
            )
        star = text.rfind("*")
        if star != -1:
            type_text = text[:star]
            element_type = self._normalize_inferred_type(type_text)
            if not element_type:
                return None
            return _MetalSubscriptableType(
                element_type,
                self._normalize_known_address_space_pointer_type(text[: star + 1]),
            )
        return None

    # Recognized Metal scalar / vector / matrix element types whose declarations
    # are reliable enough to type a bare local variable.
    _METAL_SCALAR_VECTOR_TYPES: Set[str] = {
        base + suffix
        for base in (
            "float",
            "half",
            "bfloat",
            "double",
            "int",
            "uint",
            "short",
            "ushort",
            "char",
            "uchar",
            "long",
            "ulong",
            "bool",
        )
        for suffix in ("", "2", "3", "4")
    } | {
        # C stdint / sized scalar aliases used pervasively across MLX kernels
        # (e.g. `int32_t values[N]`, `CumSum<int32_t>`). They have no Metal vector
        # spelling (vectors use `int2`/`uint4`/... instead) and already carry
        # known sizes in `_METAL_SCALAR_TYPE_SIZES`, so recognizing the scalar
        # forms lets array/subscript/local/cast inference type them — which in
        # turn lets a SFINAE member method like `simd_scan<T>` bind and resolve
        # its `sizeof(T)` overload from the concrete element type.
        "int8_t",
        "uint8_t",
        "int16_t",
        "uint16_t",
        "int32_t",
        "uint32_t",
        "int64_t",
        "uint64_t",
        "float16_t",
        "bfloat16_t",
        "size_t",
    }

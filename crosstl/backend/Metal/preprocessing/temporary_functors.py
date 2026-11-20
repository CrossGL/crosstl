"""Temporary functor calls, temporary operator wrappers and template member call instantiation."""

import re
from dataclasses import replace
from typing import (
    Dict,
    List,
    Optional,
    Set,
    Tuple,
)

from ..deduction_arguments import deduction_argument_type
from ..integral_constant_arithmetic import fold_integral_constant_parameter_arithmetic
from ..overload_deduction import deduction_pattern_mismatch
from ..static_member_constants import fold_argument_static_members
from ..template_reference_accessors import _TemplateReferenceAccessor
from .support import (
    IDENTIFIER_RE,
    MetalStructMethodError,
    _MetalConstructor,
    _MetalIntegralConstantBinding,
    _MetalPositionedBufferTypes,
    _MetalStructDefinition,
    _MetalStructMethod,
    _MetalTemporaryFunctorCall,
    _MetalTypeAliasBinding,
)


class MetalTemporaryFunctorMixin:
    """Members of ``MetalPreprocessor``: temporary functor calls, temporary operator wrappers and template member call instantiation."""

    def _temporary_functor_call(
        self, code: str, constructor_start: int
    ) -> Optional[_MetalTemporaryFunctorCall]:
        if constructor_start >= len(code):
            return None
        if code[constructor_start] == "{":
            constructor_close = self._find_matching_delimiter(
                code,
                constructor_start,
                "{",
                "}",
            )
        elif code[constructor_start] == "(":
            constructor_close = self._find_matching_delimiter(
                code,
                constructor_start,
                "(",
                ")",
            )
        else:
            return None
        if constructor_close is None:
            return None
        arg_open = constructor_close + 1
        while arg_open < len(code) and code[arg_open].isspace():
            arg_open += 1
        if arg_open >= len(code) or code[arg_open] != "(":
            return None
        arg_close = self._find_matching_delimiter(code, arg_open, "(", ")")
        if arg_close is None:
            return None
        return _MetalTemporaryFunctorCall(
            constructor_open=constructor_start,
            constructor_close=constructor_close,
            argument_open=arg_open,
            argument_close=arg_close,
        )

    # Keywords that may be followed by a parenthesized expression. In
    # `return (float)(x)` the parentheses are a cast and its operand, not a
    # temporary `return(float)` called with `(x)`.
    _KEYWORDS_BEFORE_EXPRESSIONS = frozenset(
        {
            "case",
            "co_return",
            "co_yield",
            "do",
            "else",
            "for",
            "if",
            "return",
            "sizeof",
            "alignof",
            "switch",
            "throw",
            "while",
        }
    )

    def _temporary_functor_is_candidate(
        self,
        code: str,
        identifier_start: int,
        temporary: _MetalTemporaryFunctorCall,
    ) -> bool:
        identifier, _consumed = self._read_identifier(code, identifier_start)
        if identifier in self._KEYWORDS_BEFORE_EXPRESSIONS:
            return False
        constructor_text = code[
            temporary.constructor_open + 1 : temporary.constructor_close
        ].strip()
        # Function-pointer declarations share the `Type(...)(...)` delimiter
        # shape with a constructed callable. Exclude both named pointer
        # declarators (`(*Fn)`) and anonymous alias declarators (`(*)`).
        if re.fullmatch(
            r"(?:(?:[A-Za-z_]\w*)\s*::\s*)?\*\s*" r"(?:const\s*)?(?:[A-Za-z_]\w*)?",
            constructor_text,
        ):
            return False

        # A direct function-type typedef has no `*`: `typedef void (Fn)(...)`.
        # Limit this exclusion to the current declaration so an expression such
        # as `Missing(state)(value)` remains fail-closed.
        boundary = max(
            code.rfind(token, 0, identifier_start) for token in (";", "{", "}")
        )
        declaration_prefix = self._mask_comments_and_literals(
            code[boundary + 1 : identifier_start]
        )
        if re.search(r"\btypedef\b", declaration_prefix):
            return False

        return True

    def _temporary_functor_is_stateless_default(
        self,
        code: str,
        struct: _MetalStructDefinition,
        temporary: _MetalTemporaryFunctorCall,
    ) -> bool:
        constructor_arguments = code[
            temporary.constructor_open + 1 : temporary.constructor_close
        ]
        if constructor_arguments.strip() or struct.data_member_names:
            return False
        if not struct.constructors:
            return True

        viable_default_constructors: List[_MetalConstructor] = []
        for constructor in struct.constructors:
            parameters = [
                parameter
                for parameter in self._split_top_level_commas(constructor.params_text)
                if parameter.strip()
            ]
            if all(
                self._split_top_level_assignment(parameter)[1] is not None
                for parameter in parameters
            ):
                viable_default_constructors.append(constructor)
        if len(viable_default_constructors) != 1:
            return False
        constructor = viable_default_constructors[0]
        return not constructor.init_text.strip() and not constructor.body.strip()

    def _raise_temporary_functor_error(
        self,
        code: str,
        struct_name: str,
        call_start: int,
        temporary: _MetalTemporaryFunctorCall,
        *,
        reason: str,
        detail: str,
    ) -> None:
        call_end = temporary.argument_close + 1
        requested_signature = re.sub(r"\s+", " ", code[call_start:call_end]).strip()
        raise MetalStructMethodError(
            f"Cannot lower temporary function object '{requested_signature}': "
            f"{detail}.",
            struct_name=struct_name,
            method_name="operator()",
            requested_signature=requested_signature,
            suggested_action=(
                "materialize the function object as a typed local before the call, "
                "or provide a concrete stateless functor with a uniquely resolvable "
                "operator() overload"
            ),
            source_location=self._source_location_for_offsets(
                code, call_start, call_end
            ),
            reason=reason,
        )

    def _rewrite_nested_temporary_functor_calls(
        self,
        code: str,
        start: int,
        end: int,
        struct_names: Set[str],
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
    ) -> str:
        replacements: List[Tuple[int, int, str]] = []
        cursor = start
        while cursor < end:
            if code[cursor] in "\"'":
                _literal, consumed = self._read_string(code, cursor)
                cursor += consumed
                continue
            if code.startswith("//", cursor):
                line_end = code.find("\n", cursor)
                cursor = end if line_end == -1 else min(end, line_end + 1)
                continue
            if code.startswith("/*", cursor):
                comment_end = code.find("*/", cursor + 2)
                cursor = end if comment_end == -1 else min(end, comment_end + 2)
                continue
            if not (code[cursor].isalpha() or code[cursor] == "_"):
                cursor += 1
                continue

            ident, consumed = self._read_identifier(code, cursor)
            ident_end = cursor + consumed
            constructor_start = ident_end
            while constructor_start < end and code[constructor_start].isspace():
                constructor_start += 1
            nested = self._temporary_functor_call(code, constructor_start)
            if nested is None or nested.argument_close >= end:
                cursor = ident_end
                continue

            rewrite = self._try_rewrite_call_at(
                code,
                cursor,
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
                type_aliases=type_aliases,
                type_alias_fallback_position=type_alias_fallback_position,
            )
            if rewrite is None:
                self._raise_temporary_functor_error(
                    code,
                    ident,
                    cursor,
                    nested,
                    reason="temporary-functor-overload-unresolved",
                    detail="the nested operator() call could not be materialized",
                )
            rewrite_end, replacement = rewrite
            if rewrite_end > end:
                cursor = ident_end
                continue
            replacements.append((cursor, rewrite_end, replacement))
            cursor = rewrite_end

        if not replacements:
            return code[start:end]
        parts: List[str] = []
        position = start
        for replacement_start, replacement_end, replacement in replacements:
            parts.append(code[position:replacement_start])
            parts.append(replacement)
            position = replacement_end
        parts.append(code[position:end])
        return "".join(parts)

    def _build_temporary_operator_call_rewrite(
        self,
        code: str,
        struct: _MetalStructDefinition,
        method: _MetalStructMethod,
        arg_open: int,
        instantiated_template_functions: Dict[str, str],
        *,
        rewritten_arguments: Optional[str] = None,
    ) -> Optional[Tuple[int, str]]:
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
        overload_ordinal = self._concrete_method_overload_ordinal(struct, method)
        wrapper_name = (
            f"{method.free_name}__metal_overload_{overload_ordinal}__temporary"
            if overload_ordinal is not None
            else f"{method.free_name}__temporary"
        )
        if wrapper_name not in instantiated_template_functions:
            helper_source = self._emit_free_function(struct, method)
            wrapper = self._emit_temporary_operator_wrapper(
                struct.name,
                method.free_name,
                wrapper_name,
                helper_source,
            )
            if wrapper is None:
                return None
            instantiated_template_functions[wrapper_name] = wrapper
        return self._build_template_call_rewrite(
            code,
            None,
            wrapper_name,
            arg_open,
            arg_close,
            raw_arguments=rewritten_arguments,
        )

    def _instantiate_temporary_template_operator_call(
        self,
        code: str,
        struct: _MetalStructDefinition,
        overloads: List[_MetalStructMethod],
        arg_open: int,
        buffer_element_types: _MetalPositionedBufferTypes,
        local_variable_types: Dict[str, List[Tuple[int, str]]],
        instantiated_template_functions: Dict[str, str],
        structs_by_name: Optional[Dict[str, _MetalStructDefinition]] = None,
        variable_types: Optional[Dict[str, List[Tuple[int, str]]]] = None,
        template_methods_by_struct: Optional[
            Dict[str, Dict[str, List[_MetalStructMethod]]]
        ] = None,
        methods_by_struct: Optional[Dict[str, Dict[str, _MetalStructMethod]]] = None,
        operator_call_structs: Optional[Set[str]] = None,
        rewrite_structs_by_name: Optional[Dict[str, _MetalStructDefinition]] = None,
        local_integral_constants: Optional[
            Dict[str, List[_MetalIntegralConstantBinding]]
        ] = None,
        argument_type_aliases: Optional[Dict[str, List[_MetalTypeAliasBinding]]] = None,
        argument_type_fallback_position: Optional[int] = None,
        rewritten_arguments: Optional[str] = None,
    ) -> Optional[Tuple[int, str]]:
        arg_close = self._find_matching_delimiter(code, arg_open, "(", ")")
        if arg_close is None:
            return None
        raw_args = code[arg_open + 1 : arg_close]
        call_arguments = [
            argument
            for argument in self._split_top_level_commas(raw_args)
            if argument.strip()
        ]
        representative = overloads[0]
        signature = self._template_member_call_signature(
            struct,
            representative,
            None,
            raw_args,
        )
        call_start = code.rfind(struct.name, 0, arg_open)
        if call_start < 0:
            call_start = arg_open

        # The location costs a count over the whole unit, so it is computed only for a refusal.
        def call_location() -> Dict[str, int]:
            return self._source_location_for_offsets(code, call_start, arg_close + 1)

        buffer_view = self._flatten_types_at(buffer_element_types, arg_open)
        local_view = self._flatten_types_at(local_variable_types, arg_open)
        struct_field_types = self._struct_field_types_at(
            variable_types,
            structs_by_name,
            arg_open,
        )
        local_constant_values = self._local_integral_constants_at(
            local_integral_constants or {},
            arg_open,
            names=set(IDENTIFIER_RE.findall(raw_args)),
        )
        concrete_argument_types: List[str] = []
        for argument in call_arguments:
            inference_argument = self._substitute_template_argument_static_constants(
                argument,
                local_constant_values,
            )
            inference_argument = fold_argument_static_members(
                self, inference_argument, structs_by_name
            )
            inferred = self._infer_argument_type(
                inference_argument,
                buffer_view,
                local_view,
                struct_field_types,
                structs_by_name,
            )
            if inferred is None:
                raise MetalStructMethodError(
                    "Cannot lower template member method "
                    f"'{struct.name}::{representative.name}': the type of call "
                    f"argument '{argument.strip()}' could not be inferred "
                    f"conservatively. Requested call: {signature}.",
                    struct_name=struct.name,
                    method_name=representative.name,
                    requested_signature=signature,
                    suggested_action=(
                        "pass the argument as a buffer-element access, a typed "
                        "local variable, a literal, or an explicit cast so its "
                        "type can be inferred, or specialize the method manually"
                    ),
                    source_location=call_location(),
                )
            concrete_argument_types.append(inferred)

        try:
            free_name = self._instantiate_template_member_overload(
                struct,
                overloads,
                concrete_argument_types,
                signature,
                instantiated_template_functions,
                template_methods_by_struct,
                methods_by_struct=methods_by_struct,
                operator_call_structs=operator_call_structs,
                rewrite_structs_by_name=rewrite_structs_by_name,
                argument_type_aliases=argument_type_aliases,
                argument_type_position=arg_open,
                argument_type_fallback_position=argument_type_fallback_position,
            )
        except MetalStructMethodError as exc:
            if exc.source_location is None:
                exc.source_location = call_location()
            raise
        helper_source = instantiated_template_functions.get(free_name)
        if helper_source is None:
            return None
        wrapper_name = f"{free_name}__temporary"
        if wrapper_name not in instantiated_template_functions:
            wrapper = self._emit_temporary_operator_wrapper(
                struct.name,
                free_name,
                wrapper_name,
                helper_source,
            )
            if wrapper is None:
                return None
            instantiated_template_functions[wrapper_name] = wrapper
        selection = self._instantiated_template_member_calls.get(free_name)
        if selection is not None:
            self._instantiated_template_member_calls[wrapper_name] = selection
        return self._build_template_call_rewrite(
            code,
            None,
            wrapper_name,
            arg_open,
            arg_close,
            raw_arguments=rewritten_arguments,
        )

    def _emit_temporary_operator_wrapper(
        self,
        struct_name: str,
        free_name: str,
        wrapper_name: str,
        helper_source: str,
    ) -> Optional[str]:
        match = re.match(
            rf"\s*(?:static\s+)?(?P<return>.*?)\s+{re.escape(free_name)}\s*"
            r"\((?P<params>.*?)\)\s*\{",
            helper_source,
            re.DOTALL,
        )
        if match is None:
            return None
        return_type = re.sub(r"\s+", " ", match.group("return")).strip()
        parameters = [
            parameter.strip()
            for parameter in self._split_top_level_commas(match.group("params"))
            if parameter.strip()
        ]
        if not parameters:
            return None
        if self._declared_data_member_name(parameters[0]) != "self":
            return None
        wrapper_parameters = parameters[1:]
        wrapper_parameter_text = ", ".join(wrapper_parameters)
        argument_names = self._parameter_identifier_names(wrapper_parameter_text)
        call_arguments = ", ".join(["self", *argument_names])
        call = f"{free_name}({call_arguments})"
        if return_type == "void":
            body = f"{{ {struct_name} self; {call}; }}"
        else:
            body = f"{{ {struct_name} self; return {call}; }}"
        return f"static inline {return_type} {wrapper_name}({wrapper_parameter_text}) {body}"

    def _instantiate_template_member_call(
        self,
        code: str,
        struct: _MetalStructDefinition,
        overloads: List[_MetalStructMethod],
        receiver: Optional[str],
        arg_open: int,
        buffer_element_types: _MetalPositionedBufferTypes,
        local_variable_types: Dict[str, List[Tuple[int, str]]],
        instantiated_template_functions: Dict[str, str],
        structs_by_name: Optional[Dict[str, _MetalStructDefinition]] = None,
        variable_types: Optional[Dict[str, List[Tuple[int, str]]]] = None,
        template_methods_by_struct: Optional[
            Dict[str, Dict[str, List[_MetalStructMethod]]]
        ] = None,
        methods_by_struct: Optional[Dict[str, Dict[str, _MetalStructMethod]]] = None,
        operator_call_structs: Optional[Set[str]] = None,
        rewrite_structs_by_name: Optional[Dict[str, _MetalStructDefinition]] = None,
        explicit_template_arguments: Optional[List[str]] = None,
        local_integral_constants: Optional[
            Dict[str, List[_MetalIntegralConstantBinding]]
        ] = None,
        argument_type_aliases: Optional[Dict[str, List[_MetalTypeAliasBinding]]] = None,
        argument_type_fallback_position: Optional[int] = None,
        receiver_is_const: Optional[bool] = None,
    ) -> Optional[Tuple[int, str]]:
        # Instantiate a CALLED template member method from its call-site argument
        # types and rewrite the call to the concrete free function. `overloads`
        # holds every method sharing the called name; the unique overload whose
        # SFINAE constraints are satisfied by the inferred type bindings is
        # selected. A call that cannot be resolved (un-inferable argument,
        # non-binding template parameters, or ambiguous/unrecognized constraints)
        # raises MetalStructMethodError — a clean translation failure — rather
        # than leaving a dangling call or mis-selecting an overload.
        arg_close = self._find_matching_delimiter(code, arg_open, "(", ")")
        if arg_close is None:
            return None
        raw_args = code[arg_open + 1 : arg_close]
        # All overloads of one name share a declared-parameter shape; bind using a
        # representative for argument typing and the diagnostic signature.
        representative = overloads[0]
        signature = self._template_member_call_signature(
            struct, representative, receiver, raw_args
        )
        rewritten_args = self._rewritten_call_arguments(code, arg_open, arg_close)
        if rewritten_args is not None:
            raw_args = rewritten_args
        call_arguments = [
            argument
            for argument in self._split_top_level_commas(raw_args)
            if argument.strip()
        ]
        call_start = code.rfind(representative.name, 0, arg_open)
        if call_start < 0:
            call_start = arg_open

        # The location costs a count over the whole unit, so it is computed only for a refusal.
        def call_location() -> Dict[str, int]:
            return self._source_location_for_offsets(code, call_start, arg_close + 1)

        # Resolve the position-ordered declaration maps to the FLAT views valid at
        # this call site (nearest preceding declaration wins). Passing flat dicts
        # keeps `_infer_argument_type` a pure, unit-testable function while the
        # per-call-site resolution stays deterministic and scope-correct.
        buffer_view = self._flatten_types_at(buffer_element_types, arg_open)
        local_view = self._flatten_types_at(local_variable_types, arg_open)
        struct_field_types = self._struct_field_types_at(
            variable_types, structs_by_name, arg_open
        )
        local_constant_values = self._local_integral_constants_at(
            local_integral_constants or {},
            arg_open,
            names=set(
                IDENTIFIER_RE.findall(
                    " ".join(
                        [
                            raw_args,
                            *(explicit_template_arguments or ()),
                        ]
                    )
                )
            ),
        )

        # Infer the concrete type of every call argument; one un-inferable
        # argument is a clean failure. The source call stays unchanged; only the
        # specialization identity and emitted helper signature use proven values.
        concrete_argument_types: List[str] = []
        for argument in call_arguments:
            inference_argument = self._substitute_template_argument_static_constants(
                argument,
                local_constant_values,
            )
            inference_argument = fold_argument_static_members(
                self, inference_argument, structs_by_name
            )
            inferred = self._infer_argument_type(
                inference_argument,
                buffer_view,
                local_view,
                struct_field_types,
                structs_by_name,
            )
            if inferred is None:
                raise MetalStructMethodError(
                    "Cannot lower template member method "
                    f"'{struct.name}::{representative.name}': the type of call "
                    f"argument '{argument.strip()}' could not be inferred "
                    f"conservatively. Requested call: {signature}.",
                    struct_name=struct.name,
                    method_name=representative.name,
                    requested_signature=signature,
                    suggested_action=(
                        "pass the argument as a buffer-element access, a typed "
                        "local variable, a literal, or an explicit cast so its "
                        "type can be inferred, or specialize the method manually"
                    ),
                    source_location=call_location(),
                )
            concrete_argument_types.append(inferred)

        resolved_explicit_template_arguments = (
            [
                self._substitute_template_argument_static_constants(
                    argument,
                    local_constant_values,
                )
                for argument in explicit_template_arguments
            ]
            if explicit_template_arguments is not None
            else None
        )

        try:
            free_name = self._instantiate_template_member_overload(
                struct,
                overloads,
                concrete_argument_types,
                signature,
                instantiated_template_functions,
                template_methods_by_struct,
                methods_by_struct=methods_by_struct,
                operator_call_structs=operator_call_structs,
                rewrite_structs_by_name=rewrite_structs_by_name,
                explicit_template_arguments=resolved_explicit_template_arguments,
                argument_type_aliases=argument_type_aliases,
                argument_type_position=arg_open,
                argument_type_fallback_position=argument_type_fallback_position,
                inline_reference=True,
                receiver_is_const=receiver_is_const,
            )
        except _TemplateReferenceAccessor as accessor:
            try:
                return self._build_direct_reference_accessor_rewrite(
                    code,
                    struct,
                    accessor.method,
                    receiver=receiver if receiver is not None else "self",
                    arg_open=arg_open,
                    call_start=call_start,
                )
            except MetalStructMethodError as refusal:
                if (
                    getattr(refusal, "reason", None)
                    != "reference-return-identity-unsupported"
                ):
                    raise
                # Report the refusal against the call as written, as the template path always has.
                self._reject_selected_reference_return(
                    struct,
                    accessor.method,
                    accessor.method.return_type,
                    signature,
                    source_location_factory=call_location,
                )
                raise
        except MetalStructMethodError as exc:
            if exc.source_location is None:
                exc.source_location = call_location()
            raise
        # Static-ness comes from the overload set; all overloads of one name agree
        # on static-ness in the patterns we handle, so the representative decides.
        receiver_name = None if representative.is_static else receiver
        return self._build_template_call_rewrite(
            code,
            receiver_name,
            free_name,
            arg_open,
            arg_close,
            raw_arguments=raw_args,
        )

    @staticmethod
    def _source_location_for_offsets(code: str, start: int, end: int) -> Dict[str, int]:
        start = max(0, min(start, len(code)))
        end = max(start, min(end, len(code)))
        line = code.count("\n", 0, start) + 1
        end_line = code.count("\n", 0, end) + 1
        previous_newline = code.rfind("\n", 0, start)
        end_previous_newline = code.rfind("\n", 0, end)
        return {
            "line": line,
            "column": start - previous_newline,
            "offset": start,
            "length": end - start,
            "endLine": end_line,
            "endColumn": end - end_previous_newline,
            "endOffset": end,
        }

    def _instantiate_template_member_overload(
        self,
        struct: _MetalStructDefinition,
        overloads: List[_MetalStructMethod],
        concrete_argument_types: List[Optional[str]],
        signature: str,
        instantiated_template_functions: Dict[str, str],
        template_methods_by_struct: Optional[
            Dict[str, Dict[str, List[_MetalStructMethod]]]
        ],
        methods_by_struct: Optional[Dict[str, Dict[str, _MetalStructMethod]]] = None,
        operator_call_structs: Optional[Set[str]] = None,
        rewrite_structs_by_name: Optional[Dict[str, _MetalStructDefinition]] = None,
        explicit_template_arguments: Optional[List[str]] = None,
        argument_type_aliases: Optional[Dict[str, List[_MetalTypeAliasBinding]]] = None,
        argument_type_position: Optional[int] = None,
        argument_type_fallback_position: Optional[int] = None,
        inline_reference: bool = False,
        receiver_is_const: Optional[bool] = None,
    ) -> str:
        # Bind, select the enabled overload, materialize it (recursively lowering
        # any internal template-member calls in its body), and return the
        # concrete free-function name. Raises MetalStructMethodError on any
        # unresolved/ambiguous case.
        representative = overloads[0]
        candidates: List[Tuple[_MetalStructMethod, Dict[str, str]]] = []
        deduced: List[Tuple[_MetalStructMethod, Dict[str, str]]] = []
        for overload in overloads:
            failures: List[str] = []
            bindings = self._bind_template_method_parameters(
                overload,
                concrete_argument_types,
                explicit_template_arguments=explicit_template_arguments,
                owner_struct=struct,
                structs_by_name=rewrite_structs_by_name,
                argument_type_aliases=argument_type_aliases,
                argument_type_position=argument_type_position,
                argument_type_fallback_position=argument_type_fallback_position,
                deduction_failures=failures,
            )
            if bindings is not None:
                candidates.append((overload, bindings))
                if not failures:
                    deduced.append((overload, bindings))
        if len(candidates) > 1 and deduced:
            # An overload whose deduction fails for some argument is not a
            # candidate ([temp.deduct.call]/4), so it cannot make a call ambiguous.
            candidates = deduced
        if not candidates:
            raise MetalStructMethodError(
                "Cannot lower template member method "
                f"'{struct.name}::{representative.name}': its template parameters "
                f"{representative.template_parameters} did not bind consistently "
                f"from the inferred argument types {concrete_argument_types}. "
                f"Requested call: {signature}.",
                struct_name=struct.name,
                method_name=representative.name,
                requested_signature=signature,
                suggested_action=(
                    "ensure each template parameter appears in a parameter type "
                    "that matches the inferred argument types, or specialize the "
                    "method manually"
                ),
            )

        if len(candidates) > 1 and receiver_is_const is not None:
            # A const receiver binds only const overloads and a non-const one prefers the non-const overload
            # ([over.match.funcs]), so const and non-const twins are not ambiguous.
            same_constness = [
                candidate
                for candidate in candidates
                if candidate[0].is_const == receiver_is_const
            ]
            if same_constness:
                candidates = same_constness
        selected_candidate = self._select_constrained_overload(
            candidates,
            structs=list((rewrite_structs_by_name or {}).values()),
            type_aliases=self._source_type_alias_bindings,
            position=(
                argument_type_position
                if argument_type_position is not None
                else struct.span[0]
            ),
        )
        if selected_candidate is None:
            raise MetalStructMethodError(
                "Cannot lower template member method "
                f"'{struct.name}::{representative.name}': no unique overload is "
                "compatible with the inferred argument types and enabled by its "
                "SFINAE constraints (zero or several matched, or a constraint "
                "is unsupported). "
                f"Requested call: {signature}.",
                struct_name=struct.name,
                method_name=representative.name,
                requested_signature=signature,
                suggested_action=(
                    "specialize the method manually, or restrict the call to a "
                    "type whose constraint is recognized (sizeof / is_integral_v)"
                ),
            )
        selected, bindings = selected_candidate
        resolved_return_type = self._replace_identifiers(
            selected.return_type,
            bindings,
        )
        resolved_return_type = self._canonicalize_struct_scoped_type(
            resolved_return_type,
            struct,
            rewrite_structs_by_name,
        )
        if inline_reference and self._method_returns_lvalue_reference(
            replace(selected, return_type=resolved_return_type)
        ):
            # A reference accessor names storage; the caller inlines the instantiated method
            # (template_reference_accessors.py).
            raise _TemplateReferenceAccessor(
                self._instantiated_template_member_method(
                    struct,
                    selected,
                    bindings,
                    self._template_member_free_name(
                        struct,
                        selected,
                        [bindings[name] for name in selected.template_parameters],
                    ),
                    rewrite_structs_by_name,
                )
            )
        self._reject_selected_reference_return(
            struct,
            selected,
            resolved_return_type,
            signature,
        )
        ordered_arguments = [bindings[name] for name in selected.template_parameters]
        free_name = self._template_member_free_name(struct, selected, ordered_arguments)
        previous = self._instantiated_template_member_calls.get(free_name)
        if previous is not None and (
            previous[1].parameters,
            previous[1].body,
        ) != (selected.parameters, selected.body):
            # Overloads that differ only in their function parameters, such as a
            # fragment `mma` forwarding to a matrix `mma` with the same template
            # arguments, would share one name and read as a self call.
            free_name = f"{free_name}__overload_{overloads.index(selected)}"
        self._instantiated_template_member_calls[free_name] = (
            struct,
            selected,
            bindings,
            rewrite_structs_by_name,
        )
        if free_name not in instantiated_template_functions:
            # Reserve the name first so a (pathological) self-recursive method does
            # not loop forever; the body is filled in below.
            instantiated_template_functions[free_name] = ""
            instantiated_template_functions[free_name] = (
                self._emit_template_member_free_function(
                    struct,
                    selected,
                    bindings,
                    free_name,
                    instantiated_template_functions=instantiated_template_functions,
                    template_methods_by_struct=template_methods_by_struct,
                    methods_by_struct=methods_by_struct,
                    operator_call_structs=operator_call_structs,
                    rewrite_structs_by_name=rewrite_structs_by_name,
                )
            )
        return free_name

    def _template_member_call_signature(
        self,
        struct: _MetalStructDefinition,
        method: _MetalStructMethod,
        receiver: Optional[str],
        raw_args: str,
    ) -> str:
        target = receiver if receiver is not None else struct.name
        if method.is_operator_call:
            return f"{target}({raw_args.strip()})"
        return f"{target}.{method.name}({raw_args.strip()})"

    def _bind_template_method_parameters(
        self,
        method: _MetalStructMethod,
        concrete_argument_types: List[Optional[str]],
        *,
        explicit_template_arguments: Optional[List[str]] = None,
        owner_struct: Optional[_MetalStructDefinition] = None,
        structs_by_name: Optional[Dict[str, _MetalStructDefinition]] = None,
        argument_type_aliases: Optional[Dict[str, List[_MetalTypeAliasBinding]]] = None,
        argument_type_position: Optional[int] = None,
        argument_type_fallback_position: Optional[int] = None,
        deduction_failures: Optional[List[str]] = None,
    ) -> Optional[Dict[str, str]]:
        # Match each declared method-parameter type (which may contain the
        # template parameters) against the inferred concrete argument type and
        # collect consistent bindings. Returns None if any template parameter is
        # left unbound or a parameter binds inconsistently. Parameter types whose
        # deduction must fail (overload_deduction.py) are appended to
        # `deduction_failures` when it is given.
        template_parameter_set = set(method.template_parameters)
        source_type_aliases = self._source_type_alias_bindings

        def canonicalize_declared_aliases(
            type_text: str,
            *,
            excluded_aliases: Optional[Set[str]] = None,
        ) -> Optional[str]:
            if not source_type_aliases:
                return self._normalize_template_argument_text(type_text)
            return self._canonicalize_type_aliases_at(
                type_text,
                source_type_aliases,
                method.span[0],
                excluded_aliases=excluded_aliases,
            )

        def canonicalize_argument_aliases(type_text: str) -> Optional[str]:
            canonical = self._normalize_template_argument_text(type_text)
            primary_aliases = (
                source_type_aliases
                if argument_type_aliases is None
                else argument_type_aliases
            )
            if primary_aliases and argument_type_position is not None:
                canonical = self._canonicalize_type_aliases_at(
                    canonical,
                    primary_aliases,
                    argument_type_position,
                )
                if canonical is None:
                    return None
            if (
                primary_aliases is not source_type_aliases
                and source_type_aliases
                and argument_type_fallback_position is not None
            ):
                canonical = self._canonicalize_type_aliases_at(
                    canonical,
                    source_type_aliases,
                    argument_type_fallback_position,
                )
            return canonical

        # `method.parameters` is the already-extracted parameter list (the
        # `operator()` declarator is handled at parse time), so split it directly
        # rather than reparsing a synthesized header — a synthesized
        # `operator()` header would confuse the first-paren parameter finder.
        parameters = [
            parameter
            for parameter in self._split_top_level_commas(method.parameters)
            if parameter.strip()
        ]
        if len(concrete_argument_types) > len(parameters):
            return None
        for parameter in parameters[len(concrete_argument_types) :]:
            _declaration, default = self._split_top_level_assignment(parameter)
            if default is None or not default.strip():
                return None
        declared_parameter_types: List[str] = []
        declared_parameter_is_indirect: List[bool] = []
        declared_parameter_is_array: List[bool] = []
        declared_pointer_types: List[Optional[str]] = []
        for parameter in parameters[: len(concrete_argument_types)]:
            subscriptable = self._pointer_or_array_parameter_type(parameter)
            declared = (
                # Template arguments bind through the innermost element of a
                # multidimensional array parameter, as through its decayed pointer.
                re.sub(r"(?:\s*\[[^\[\]]*\])+$", "", subscriptable.element_type)
                if subscriptable is not None
                else self._function_parameter_value_type(parameter)
            )
            if owner_struct is not None:
                declared = self._canonicalize_struct_scoped_type(
                    declared, owner_struct, structs_by_name
                )
            normalized = self._normalize_inferred_type(declared)
            if normalized and normalized != "void":
                declared_parameter_types.append(normalized)
                declared_parameter_is_indirect.append(subscriptable is not None)
                declarator = self._strip_top_level_default_value(
                    self._strip_metal_attributes(parameter)
                ).strip()
                declared_parameter_is_array.append(declarator.endswith("]"))
                declared_pointer_types.append(
                    subscriptable.pointer_type if subscriptable is not None else None
                )
        if len(declared_parameter_types) != len(concrete_argument_types):
            return None
        # Bind each parameter into its OWN dict and merge with explicit conflict
        # detection. `_infer_template_parameter_bindings_from_type` silently
        # SKIPS a parameter whose new value conflicts with an existing binding;
        # that is the wrong behavior here — a template parameter that the call
        # site forces to two different concrete types (`pick(float, int)` for
        # `T pick(T, T)`) must clean-fail, not silently keep the first guess.
        explicit_template_arguments = explicit_template_arguments or []
        if len(explicit_template_arguments) > len(method.template_parameters):
            return None
        bindings: Dict[str, str] = {
            name: self._normalize_template_binding_type(argument)
            for name, argument in zip(
                method.template_parameters, explicit_template_arguments
            )
        }
        if any(not value for value in bindings.values()):
            return None
        explicitly_bound = set(bindings)
        indirect_pointee_types: List[Tuple[str, str]] = []
        for (
            declared_type,
            is_indirect,
            is_array,
            declared_pointer_type,
            concrete_type,
        ) in zip(
            declared_parameter_types,
            declared_parameter_is_indirect,
            declared_parameter_is_array,
            declared_pointer_types,
            concrete_argument_types,
        ):
            if concrete_type is None:
                continue
            local_bindings: Dict[str, str] = {}
            concrete_value_type = (
                self._normalize_template_binding_type(concrete_type).rstrip("&").strip()
            )
            if is_indirect and concrete_value_type.endswith("*"):
                canonical_declared_pointer = (
                    canonicalize_declared_aliases(
                        declared_pointer_type,
                        excluded_aliases=template_parameter_set,
                    )
                    if declared_pointer_type is not None
                    else None
                )
                canonical_concrete_pointer = canonicalize_argument_aliases(
                    concrete_value_type
                )
                if (
                    canonical_declared_pointer is None
                    or canonical_concrete_pointer is None
                    or not (
                        self._pointer_argument_matches_declared_type(
                            canonical_declared_pointer,
                            canonical_concrete_pointer,
                        )
                    )
                ):
                    return None
                concrete_value_type = self._pointer_pointee_value_type(
                    concrete_value_type
                )
                if concrete_value_type is None:
                    return None
                indirect_pointee_types.append((declared_type, concrete_value_type))
            elif is_array:
                # Internal method-body inference retains the element type for
                # local arrays. It is still a concrete pointee for validation.
                indirect_pointee_types.append((declared_type, concrete_value_type))
            concrete_value_type = deduction_argument_type(
                self, declared_type, concrete_value_type, structs_by_name
            )
            self._infer_template_parameter_bindings_from_type(
                declared_type,
                concrete_value_type,
                template_parameter_set,
                local_bindings,
            )
            if (
                deduction_failures is not None
                and not local_bindings
                and deduction_pattern_mismatch(
                    declared_type,
                    concrete_value_type,
                    template_parameter_set.difference(explicitly_bound),
                )
            ):
                deduction_failures.append(declared_type)
            for name, value in local_bindings.items():
                existing = bindings.get(name)
                if existing is not None and existing != value:
                    return None
                bindings[name] = value
        # Apply declared defaults only after explicit and inferred bindings so a
        # dependent default can reference an earlier template parameter.
        for name in method.template_parameters:
            if name not in bindings:
                default = method.template_parameter_defaults.get(name)
                if default is None:
                    return None
                resolved_default = self._replace_identifiers(default, bindings)
                normalized_default = self._normalize_template_binding_type(
                    resolved_default
                )
                if not normalized_default:
                    return None
                bindings[name] = normalized_default
        # Pointer deduction above validates address-space/cv compatibility and
        # binds template names. Once explicit/default bindings are also known,
        # validate the complete pointee pattern so concrete portions cannot be
        # ignored merely because another parameter supplied every binding.
        for declared_type, concrete_value_type in indirect_pointee_types:
            resolved_declared_type = self._replace_identifiers(declared_type, bindings)
            if owner_struct is not None:
                resolved_declared_type = self._canonicalize_struct_scoped_type(
                    resolved_declared_type, owner_struct, structs_by_name
                )
            resolved_declared_type = canonicalize_declared_aliases(
                resolved_declared_type
            )
            concrete_value_type = canonicalize_argument_aliases(concrete_value_type)
            if resolved_declared_type is None or concrete_value_type is None:
                return None
            resolved_declared_type = self._canonical_template_binding_pointee_type(
                resolved_declared_type, context=concrete_value_type
            )
            concrete_value_type = self._canonical_template_binding_pointee_type(
                concrete_value_type
            )
            # A materialized specialization such as `S_float16_t` expands to the
            # argument spelling it was named from, so canonicalize both sides'
            # aliases again after the expansion.
            resolved_declared_type = canonicalize_declared_aliases(
                resolved_declared_type
            )
            concrete_value_type = canonicalize_argument_aliases(concrete_value_type)
            if (
                resolved_declared_type is None
                or concrete_value_type is None
                or resolved_declared_type != concrete_value_type
            ):
                return None
        return bindings

    def _template_member_free_name(
        self,
        struct: _MetalStructDefinition,
        method: _MetalStructMethod,
        ordered_arguments: List[str],
    ) -> str:
        # Concrete free-function name for an instantiated template member method:
        # `S__m__<binding-suffix>` (or `S__operator_call__<suffix>`).
        base = self._struct_member_free_name(
            struct.name, method.name, method.is_operator_call
        )
        suffix = "_".join(
            self._template_member_name_component(value) for value in ordered_arguments
        ).strip("_")
        if not suffix:
            return base
        return f"{base}__{suffix}"

    @staticmethod
    def _template_member_name_component(value: str) -> str:
        text = str(value)
        text = re.sub(r"(?<![A-Za-z0-9_])-(?=\s*[0-9])", " negative ", text)
        text = text.replace("*", " ptr ").replace("&", " ref ")
        return re.sub(r"[^A-Za-z0-9_]+", "_", text).strip("_")

    def _instantiated_template_member_method(
        self,
        struct: _MetalStructDefinition,
        method: _MetalStructMethod,
        bindings: Dict[str, str],
        free_name: str,
        rewrite_structs_by_name: Optional[Dict[str, _MetalStructDefinition]] = None,
    ) -> _MetalStructMethod:
        """The concrete method a template member method's ``bindings`` instantiate."""
        instantiated_return = self._replace_identifiers(method.return_type, bindings)
        instantiated_return = re.sub(r"\s+", " ", instantiated_return).strip()
        instantiated_parameters = self._replace_identifiers(method.parameters, bindings)
        instantiated_body = self._replace_identifiers(method.body, bindings)
        source_body = instantiated_body
        instantiated_body = self._specialize_concrete_method_body(
            struct, method, instantiated_body, rewrite_structs_by_name
        )
        instantiated_body = self._substitute_integral_constant_parameter_values(
            instantiated_parameters, instantiated_body
        )
        instantiated_body = fold_integral_constant_parameter_arithmetic(
            self, instantiated_parameters, instantiated_body
        )
        concrete_parameters = self._strip_function_parameter_defaults(
            instantiated_parameters
        )
        concrete_parameters = self._preserve_parameter_usage(
            concrete_parameters, source_body, instantiated_body
        )
        concrete_method = _MetalStructMethod(
            name=method.name,
            free_name=free_name,
            is_static=method.is_static,
            is_operator_call=method.is_operator_call,
            return_type=instantiated_return,
            parameters=concrete_parameters,
            parameter_names=self._parameter_identifier_names(concrete_parameters),
            body=instantiated_body,
            span=method.span,
            is_const=method.is_const,
            is_volatile=method.is_volatile,
            ref_qualifier=method.ref_qualifier,
            receiver_address_spaces=method.receiver_address_spaces,
            trailing_qualifiers=method.trailing_qualifiers,
        )
        return concrete_method

    def _emit_template_member_free_function(
        self,
        struct: _MetalStructDefinition,
        method: _MetalStructMethod,
        bindings: Dict[str, str],
        free_name: str,
        instantiated_template_functions: Optional[Dict[str, str]] = None,
        template_methods_by_struct: Optional[
            Dict[str, Dict[str, List[_MetalStructMethod]]]
        ] = None,
        methods_by_struct: Optional[Dict[str, Dict[str, _MetalStructMethod]]] = None,
        operator_call_structs: Optional[Set[str]] = None,
        rewrite_structs_by_name: Optional[Dict[str, _MetalStructDefinition]] = None,
    ) -> str:
        # Instantiate the template member method by substituting the bound
        # template parameters into copies of its return type, parameters and
        # body, then emit a concrete free function reusing the non-template
        # lowering machinery (member references resolved to `self.x`).
        concrete_method = self._instantiated_template_member_method(
            struct, method, bindings, free_name, rewrite_structs_by_name
        )
        # Qualify owner fields before resolving nested calls, as for ordinary
        # methods. The instantiated signature supplies the lexical parameters.
        return self._emit_free_function(
            struct,
            concrete_method,
            instantiated_template_functions=instantiated_template_functions,
            template_methods_by_struct=template_methods_by_struct,
            methods_by_struct=methods_by_struct,
            operator_call_structs=operator_call_structs,
            structs_by_name=rewrite_structs_by_name,
        )

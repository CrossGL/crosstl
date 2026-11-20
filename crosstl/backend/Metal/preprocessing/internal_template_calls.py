"""Lowering of internal and runtime-value template member calls."""

import re
from typing import (
    Dict,
    List,
    Optional,
    Set,
    Tuple,
)

from ..method_arity import accepts_argument_count
from ..static_member_constants import fold_argument_static_members
from ..template_reference_accessors import inline_template_reference_accessor
from .support import (
    IDENTIFIER_RE,
    MetalStructMethodError,
    _MetalBufferTypeView,
    _MetalIntegralConstantBinding,
    _MetalStructDefinition,
    _MetalStructMethod,
    _MetalSubscriptableType,
    _MetalTypeAliasBinding,
)


class MetalInternalTemplateCallMixin:
    """Members of ``MetalPreprocessor``: lowering of internal and runtime-value template member calls."""

    def _lower_internal_template_member_calls(
        self,
        struct: _MetalStructDefinition,
        method: _MetalStructMethod,
        instantiated_parameters: str,
        instantiated_body: str,
        instantiated_template_functions: Dict[str, str],
        template_methods_by_struct: Dict[str, Dict[str, List[_MetalStructMethod]]],
        methods_by_struct: Optional[Dict[str, Dict[str, _MetalStructMethod]]] = None,
        operator_call_structs: Optional[Set[str]] = None,
        rewrite_structs_by_name: Optional[Dict[str, _MetalStructDefinition]] = None,
    ) -> str:
        # Rewrite receiver-less calls to OTHER template member methods of the same
        # struct inside an already-substituted method body. Argument types are
        # inferred from the (now concrete) parameter types and the body's local
        # declarations; the existing argument-inference + overload selection is
        # reused, so an un-inferable / ambiguous internal call clean-fails exactly
        # like a top-level call. The instance receiver is `self` (the free
        # function's first parameter); a static sibling takes no receiver.
        sibling_overloads = template_methods_by_struct.get(struct.name, {})
        methods_by_struct = methods_by_struct or {}
        operator_call_structs = operator_call_structs or set()
        rewrite_structs_by_name = rewrite_structs_by_name or {}
        self._register_contextual_receiver_parameters(
            instantiated_body,
            instantiated_parameters,
        )
        instantiated_body = self._lower_runtime_value_template_member_calls(
            struct,
            method,
            instantiated_body,
            sibling_overloads,
            instantiated_template_functions,
            template_methods_by_struct,
            methods_by_struct,
            operator_call_structs,
            rewrite_structs_by_name,
        )
        contextual_receiver_names = self._register_contextual_receiver_parameters(
            instantiated_body,
            instantiated_parameters,
        )
        rewrite_struct_names = set(rewrite_structs_by_name)
        struct_type_aliases = self._collect_struct_type_aliases(
            instantiated_body,
            rewrite_struct_names,
            [],
            rewrite_structs_by_name,
        )
        for alias, target in struct.type_aliases.items():
            resolved_target = self._resolve_struct_type_alias_at(
                struct_type_aliases, target, 0
            )
            if resolved_target is None and target in rewrite_struct_names:
                resolved_target = target
            if resolved_target is None:
                continue
            struct_type_aliases.setdefault(alias, []).insert(
                0,
                _MetalTypeAliasBinding(
                    declaration_position=-1,
                    scope_start=0,
                    scope_end=len(instantiated_body),
                    target=resolved_target,
                ),
            )
        qualified_identifiers = set(
            re.findall(r"\b([A-Za-z_]\w*)\s*::", instantiated_body)
        )
        has_external_qualified_call = bool(
            qualified_identifiers & (rewrite_struct_names | set(struct_type_aliases))
        )
        has_external_member_call = bool(
            (methods_by_struct or template_methods_by_struct)
            and any(
                re.search(
                    rf"\b{re.escape(receiver)}\s*(?:\.|->)",
                    instantiated_body,
                )
                for receiver in {"self", *contextual_receiver_names}
            )
        )
        if not has_external_member_call and (
            methods_by_struct or template_methods_by_struct
        ):
            # A body-local object of a method-bearing struct is a receiver
            # too: a method that only calls `tile.fill(x)` on its own local
            # must not keep that call once the local's struct is lowered to a
            # data-only carrier.
            has_external_member_call = any(
                re.search(
                    rf"\b{re.escape(receiver)}\s*(?:\.|->)\s*"
                    r"(?:template\s+)?[A-Za-z_]\w*\s*[<(]",
                    instantiated_body,
                )
                for receiver in sorted(
                    self._method_body_struct_receiver_locals(
                        struct,
                        instantiated_body,
                        methods_by_struct,
                        template_methods_by_struct,
                        operator_call_structs,
                        rewrite_structs_by_name,
                    )
                )
            )
        has_concrete_sibling_call = any(
            re.search(
                rf"(?<![\w.:>]){re.escape(sibling.name)}\s*\(",
                instantiated_body,
            )
            for sibling in struct.methods
        )
        # A method whose body has no sibling-template or external struct call
        # candidate is returned unchanged.
        if (
            not sibling_overloads
            and not operator_call_structs
            and not has_external_qualified_call
            and not has_external_member_call
            and not has_concrete_sibling_call
        ):
            return instantiated_body
        # Build flat name->type views for the body scope from the concrete
        # parameters plus body-local declarations.
        local_view: Dict[str, str] = {}
        buffer_view: _MetalBufferTypeView = {}
        for parameter in self._split_top_level_commas(instantiated_parameters):
            if not parameter.strip():
                continue
            name = self._declared_data_member_name(parameter)
            if not name:
                continue
            subscriptable = self._pointer_or_array_parameter_type(parameter)
            if subscriptable is not None:
                buffer_view[name] = subscriptable
                continue
            scalar = self._function_parameter_value_type(parameter)
            if scalar:
                # A parameter may be spelled with a struct-scoped alias such as
                # `thread const frag_type& v`; type it like a body local.
                local_view[name] = self._canonicalize_struct_scoped_type(
                    scalar, struct, rewrite_structs_by_name
                )
        for statement in self._iter_simple_declarations(instantiated_body):
            name = self._declared_local_name(statement)
            if not name:
                continue
            element_type = self._declared_local_type(statement, name)
            if element_type:
                element_type = self._canonicalize_struct_scoped_type(
                    element_type, struct, rewrite_structs_by_name
                )
                local_view.setdefault(name, element_type)
        # Body-local pointers (`threadgroup const float* view = ...;`) are
        # subscriptable like pointer parameters, so a template member call that
        # receives `view + 1` deduces its element type the same way.
        for name, subscriptable in self._method_body_local_pointer_types(
            instantiated_body
        ).items():
            buffer_view.setdefault(name, subscriptable)
        local_view["self"] = struct.name
        positioned_buffer_types = {
            name: [(0, value)] for name, value in buffer_view.items()
        }
        positioned_local_types = {
            name: [(0, value)] for name, value in local_view.items()
        }
        local_constant_owner_spans = [(0, len(instantiated_body))]
        local_constant_type_aliases = self._collect_local_type_alias_bindings(
            instantiated_body,
            local_constant_owner_spans,
        )
        local_integral_constants = self._collect_local_integral_constant_bindings(
            instantiated_body,
            local_constant_owner_spans,
            local_constant_type_aliases,
            self._active_static_constexpr_functions,
            method.span[0],
        )
        reference_body = self._rewrite_local_reference_alias_bindings(
            instantiated_body,
            positioned_local_types,
            rewrite_structs_by_name,
            const_receivers={"self"} if method.is_const else set(),
            implicit_receiver_contract=self._method_body_receiver_contract(
                struct, method
            ),
        )
        if reference_body != instantiated_body:
            # Captured indices add declarations and change lexical offsets.
            # Rebuild the positioned type/alias views before resolving calls.
            return self._lower_internal_template_member_calls(
                struct,
                method,
                instantiated_parameters,
                reference_body,
                instantiated_template_functions,
                template_methods_by_struct,
                methods_by_struct,
                operator_call_structs,
                rewrite_structs_by_name,
            )
        instantiated_body = self._rewrite_internal_direct_reference_accessor_calls(
            struct,
            method,
            instantiated_body,
        )
        result: List[str] = []
        i = 0
        n = len(instantiated_body)
        while i < n:
            ch = instantiated_body[i]
            if ch in "\"'":
                literal, consumed = self._read_string(instantiated_body, i)
                result.append(literal)
                i += consumed
                continue
            if instantiated_body.startswith("//", i):
                end = instantiated_body.find("\n", i)
                if end == -1:
                    result.append(instantiated_body[i:])
                    break
                result.append(instantiated_body[i:end])
                i = end
                continue
            if instantiated_body.startswith("/*", i):
                end = instantiated_body.find("*/", i + 2)
                if end == -1:
                    result.append(instantiated_body[i:])
                    break
                result.append(instantiated_body[i : end + 2])
                i = end + 2
                continue
            if ch.isalpha() or ch == "_":
                ident, consumed = self._read_identifier(instantiated_body, i)
                ident_end = i + consumed
                rewrite = self._try_rewrite_internal_template_call(
                    struct,
                    method,
                    ident,
                    ident_end,
                    instantiated_body,
                    sibling_overloads,
                    local_view,
                    buffer_view,
                    instantiated_template_functions,
                    template_methods_by_struct,
                    local_integral_constants,
                    local_constant_type_aliases,
                    methods_by_struct=methods_by_struct,
                    operator_call_structs=operator_call_structs,
                    rewrite_structs_by_name=rewrite_structs_by_name,
                )
                if rewrite is not None:
                    end, replacement = rewrite
                    result.append(replacement)
                    i = end
                    continue
                rewrite = self._try_rewrite_call_at(
                    instantiated_body,
                    i,
                    ident,
                    ident_end,
                    rewrite_struct_names,
                    methods_by_struct,
                    template_methods_by_struct,
                    operator_call_structs,
                    positioned_local_types,
                    rewrite_structs_by_name,
                    positioned_buffer_types,
                    positioned_local_types,
                    instantiated_template_functions,
                    field_variable_types=positioned_local_types,
                    field_structs_by_name=rewrite_structs_by_name,
                    struct_type_aliases=struct_type_aliases,
                    local_integral_constants=local_integral_constants,
                    type_aliases=local_constant_type_aliases,
                    type_alias_fallback_position=method.span[0],
                    implicit_receiver_contract=self._method_body_receiver_contract(
                        struct,
                        method,
                    ),
                )
                if rewrite is not None:
                    end, replacement = rewrite
                    result.append(replacement)
                    i = end
                    continue
                result.append(ident)
                i = ident_end
                continue
            result.append(ch)
            i += 1
        return "".join(result)

    def _method_body_local_pointer_types(
        self, body: str
    ) -> Dict[str, _MetalSubscriptableType]:
        """Element types of the single-declarator pointer locals of a method body.

        Only a declaration whose one declarator is ``[qualifiers] T* name`` with an
        explicit Metal address space is typed; arrays, references, multiple
        declarators and pointers without an address space stay untyped, so call
        argument inference keeps refusing them.
        """
        pointers: Dict[str, _MetalSubscriptableType] = {}
        for statement in self._iter_simple_declarations(body):
            name = self._declared_local_name(statement)
            if not name or name in pointers:
                continue
            declarator = self._strip_top_level_default_value(statement).strip()
            declarator = declarator.rstrip(";").strip()
            if len(self._split_top_level_commas(declarator)) != 1 or not re.fullmatch(
                r"(?:(?:const|volatile)\s+)*"
                r"(?:device|constant|threadgroup|thread)\b[^;=\[\](){}&]*\*\s*"
                + re.escape(name),
                declarator,
            ):
                continue
            subscriptable = self._pointer_or_array_parameter_type(declarator)
            if subscriptable is not None:
                pointers[name] = subscriptable
        return pointers

    def _method_body_struct_receiver_locals(
        self,
        struct: _MetalStructDefinition,
        body: str,
        methods_by_struct: Dict[str, Dict[str, _MetalStructMethod]],
        template_methods_by_struct: Dict[str, Dict[str, List[_MetalStructMethod]]],
        operator_call_structs: Set[str],
        rewrite_structs_by_name: Dict[str, _MetalStructDefinition],
    ) -> Set[str]:
        """Names of body-local objects whose struct declares member functions."""
        method_structs = set(operator_call_structs)
        method_structs.update(
            name for name, methods in methods_by_struct.items() if methods
        )
        method_structs.update(
            name for name, methods in template_methods_by_struct.items() if methods
        )
        if not method_structs:
            return set()
        receivers: Set[str] = set()
        for statement in self._iter_simple_declarations(body):
            name = self._declared_local_name(statement)
            if not name:
                continue
            declared_type = self._declared_local_type(statement, name)
            if not declared_type:
                continue
            declared_type = self._canonicalize_struct_scoped_type(
                declared_type, struct, rewrite_structs_by_name
            )
            words = IDENTIFIER_RE.findall(declared_type)
            if words and words[-1] in method_structs:
                receivers.add(name)
        return receivers

    def _rewrite_internal_direct_reference_accessor_calls(
        self,
        struct: _MetalStructDefinition,
        enclosing_method: _MetalStructMethod,
        body: str,
    ) -> str:
        """Inline proven implicit accessors before outer-call type deduction.

        Internal-call lowering normally scans left to right. For a call such as
        ``Frag::load(frag_at(i, j), src)``, that visits the outer static template
        member before the nested ``frag_at`` sibling. Template deduction must
        see the accessor's backing lvalue rather than guess a return type, so
        rewrite only concrete, direct reference accessors in a preliminary pass.
        The established accessor proof still rejects binding/escaping results,
        side-effecting arguments, ambiguous constness, and non-direct storage.
        """

        reference_methods_by_name: Dict[str, List[_MetalStructMethod]] = {}
        for candidate in struct.methods:
            if self._method_returns_lvalue_reference(candidate):
                reference_methods_by_name.setdefault(candidate.name, []).append(
                    candidate
                )
        if not reference_methods_by_name:
            return body

        result: List[str] = []
        cursor = 0
        while cursor < len(body):
            if body[cursor] in "\"'":
                literal, consumed = self._read_string(body, cursor)
                result.append(literal)
                cursor += consumed
                continue
            if body.startswith("//", cursor):
                end = body.find("\n", cursor)
                if end == -1:
                    result.append(body[cursor:])
                    break
                result.append(body[cursor:end])
                cursor = end
                continue
            if body.startswith("/*", cursor):
                end = body.find("*/", cursor + 2)
                if end == -1:
                    result.append(body[cursor:])
                    break
                result.append(body[cursor : end + 2])
                cursor = end + 2
                continue
            if not (body[cursor].isalpha() or body[cursor] == "_"):
                result.append(body[cursor])
                cursor += 1
                continue

            name, consumed = self._read_identifier(body, cursor)
            name_end = cursor + consumed
            methods = reference_methods_by_name.get(name)
            previous = cursor - 1
            while previous >= 0 and body[previous].isspace():
                previous -= 1
            is_qualified = previous >= 0 and (
                body[previous] == "."
                or (previous >= 1 and body[previous - 1 : previous + 1] in {"->", "::"})
            )
            call_suffix = (
                self._member_template_call_suffix(body, name_end)
                if methods and not is_qualified
                else None
            )
            if call_suffix is not None:
                arg_open, explicit_template_arguments = call_suffix
                if explicit_template_arguments is None:
                    selected = self._select_direct_reference_method(
                        body,
                        struct,
                        methods,
                        arg_open,
                        receiver_is_const=enclosing_method.is_const,
                    )
                    call_end, replacement = (
                        self._build_direct_reference_accessor_rewrite(
                            body,
                            struct,
                            selected,
                            receiver="self",
                            arg_open=arg_open,
                            call_start=cursor,
                            enclosing_return_type=enclosing_method.return_type,
                        )
                    )
                    result.append(replacement)
                    cursor = call_end
                    continue

            result.append(name)
            cursor = name_end

        return "".join(result)

    def _lower_runtime_value_template_member_calls(
        self,
        struct: _MetalStructDefinition,
        enclosing_method: _MetalStructMethod,
        body: str,
        sibling_overloads: Dict[str, List[_MetalStructMethod]],
        instantiated_template_functions: Dict[str, str],
        template_methods_by_struct: Dict[str, Dict[str, List[_MetalStructMethod]]],
        methods_by_struct: Dict[str, Dict[str, _MetalStructMethod]],
        operator_call_structs: Set[str],
        structs_by_name: Dict[str, _MetalStructDefinition],
    ) -> str:
        result: List[str] = []
        i = 0
        while i < len(body):
            if body[i] in "\"'":
                literal, consumed = self._read_string(body, i)
                result.append(literal)
                i += consumed
                continue
            if body.startswith("//", i):
                end = body.find("\n", i)
                if end == -1:
                    result.append(body[i:])
                    break
                result.append(body[i:end])
                i = end
                continue
            if body.startswith("/*", i):
                end = body.find("*/", i + 2)
                if end == -1:
                    result.append(body[i:])
                    break
                result.append(body[i : end + 2])
                i = end + 2
                continue
            if body[i].isalpha() or body[i] == "_":
                ident, consumed = self._read_identifier(body, i)
                ident_end = i + consumed
                rewrite = self._try_lower_runtime_value_template_member_call(
                    struct,
                    enclosing_method,
                    ident,
                    ident_end,
                    body,
                    sibling_overloads,
                    instantiated_template_functions,
                    template_methods_by_struct,
                    methods_by_struct,
                    operator_call_structs,
                    structs_by_name,
                )
                if rewrite is not None:
                    end, replacement = rewrite
                    result.append(replacement)
                    i = end
                    continue
                result.append(ident)
                i = ident_end
                continue
            result.append(body[i])
            i += 1
        return "".join(result)

    def _try_lower_runtime_value_template_member_call(
        self,
        struct: _MetalStructDefinition,
        enclosing_method: _MetalStructMethod,
        ident: str,
        ident_end: int,
        body: str,
        sibling_overloads: Dict[str, List[_MetalStructMethod]],
        instantiated_template_functions: Dict[str, str],
        template_methods_by_struct: Dict[str, Dict[str, List[_MetalStructMethod]]],
        methods_by_struct: Dict[str, Dict[str, _MetalStructMethod]],
        operator_call_structs: Set[str],
        structs_by_name: Dict[str, _MetalStructDefinition],
    ) -> Optional[Tuple[int, str]]:
        overloads = sibling_overloads.get(ident)
        if not overloads:
            return None
        previous = ident_end - len(ident) - 1
        while previous >= 0 and body[previous].isspace():
            previous -= 1
        if previous >= 0 and (
            body[previous] == "."
            or (previous >= 1 and body[previous - 1 : previous + 1] in {"->", "::"})
        ):
            return None
        call_suffix = self._member_template_call_suffix(body, ident_end)
        if call_suffix is None:
            return None
        arg_open, explicit_arguments = call_suffix
        if explicit_arguments is None:
            return None
        arg_close = self._find_matching_delimiter(body, arg_open, "(", ")")
        if arg_close is None or body[arg_open + 1 : arg_close].strip():
            return None

        candidates = [
            method
            for method in overloads
            if self._runtime_value_template_method_is_safe(method, explicit_arguments)
            and (method.is_const if enclosing_method.is_const else True)
        ]
        if not enclosing_method.is_const:
            non_const = [method for method in candidates if not method.is_const]
            if non_const:
                candidates = non_const
        if len(candidates) != 1:
            return None
        selected = candidates[0]
        # A reference accessor names storage, so it is inlined rather than lowered to a by-value function
        # (template_reference_accessors.py).
        inlined = inline_template_reference_accessor(
            self,
            struct,
            enclosing_method,
            selected,
            body,
            ident_end - len(ident),
            arg_open,
            explicit_arguments,
        )
        if inlined is not None:
            return inlined
        self._reject_selected_reference_return(
            struct,
            selected,
            self._canonicalize_struct_scoped_type(
                selected.return_type,
                struct,
                structs_by_name,
            ),
            f"{struct.name}::{selected.name}<" f"{', '.join(explicit_arguments)}>()",
            source_location_factory=lambda: self._source_location_for_offsets(
                body,
                ident_end - len(ident),
                arg_close + 1,
            ),
        )
        parameter_types = [
            selected.template_parameter_types[name]
            for name in selected.template_parameters
        ]
        type_suffix = "_".join(
            re.sub(r"[^A-Za-z0-9_]+", "_", value).strip("_")
            for value in parameter_types
        )
        free_name = (
            f"{self._struct_member_free_name(struct.name, ident, False)}"
            f"__runtime_values__{type_suffix}"
        )
        if selected.is_const:
            free_name += "__const"
        if free_name not in instantiated_template_functions:
            parameters = ", ".join(
                f"{type_name} {name}"
                for name, type_name in zip(
                    selected.template_parameters, parameter_types
                )
            )
            runtime_method = _MetalStructMethod(
                name=selected.name,
                free_name=free_name,
                is_static=selected.is_static,
                is_operator_call=selected.is_operator_call,
                return_type=selected.return_type,
                parameters=parameters,
                parameter_names=list(selected.template_parameters),
                body=selected.body,
                span=selected.span,
                is_const=selected.is_const,
                is_volatile=selected.is_volatile,
                ref_qualifier=selected.ref_qualifier,
                receiver_address_spaces=selected.receiver_address_spaces,
                trailing_qualifiers=selected.trailing_qualifiers,
            )
            instantiated_template_functions[free_name] = self._emit_free_function(
                struct,
                runtime_method,
                instantiated_template_functions=instantiated_template_functions,
                template_methods_by_struct=template_methods_by_struct,
                methods_by_struct=methods_by_struct,
                operator_call_structs=operator_call_structs,
                structs_by_name=structs_by_name,
            )
        call_arguments = ", ".join(explicit_arguments)
        if selected.is_static:
            replacement = f"{free_name}({call_arguments})"
        else:
            replacement = f"{free_name}(self, {call_arguments})"
        return arg_close + 1, replacement

    def _runtime_value_template_method_is_safe(
        self,
        method: _MetalStructMethod,
        explicit_arguments: List[str],
    ) -> bool:
        if method.parameters.strip() not in {"", "void"}:
            return False
        if len(method.template_parameters) != len(explicit_arguments):
            return False
        if set(method.template_parameter_types) != set(method.template_parameters):
            return False
        if (
            method.template_constraints
            or method.template_parameter_defaults
            or method.variadic_template_parameters
        ):
            return False
        for name in method.template_parameters:
            type_name = method.template_parameter_types[name]
            if self._is_integral_concrete_type(type_name) is not True:
                return False
            if re.search(rf"\b{re.escape(name)}\b", method.return_type):
                return False
        match = re.fullmatch(r"\s*return\s+(?P<expr>[^;]+);\s*", method.body)
        if match is None:
            return False
        expression = match.group("expr")
        if re.search(r"[<>{}?:,=]", expression):
            return False
        if re.search(r"\b(?:sizeof|alignof|decltype)\s*\(", expression):
            return False
        if re.search(r"[A-Za-z_][A-Za-z0-9_]*\s*\(", expression):
            return False
        return all(
            re.search(rf"\b{re.escape(name)}\b", expression) is not None
            for name in method.template_parameters
        )

    def _try_rewrite_internal_template_call(
        self,
        struct: _MetalStructDefinition,
        method: _MetalStructMethod,
        ident: str,
        ident_end: int,
        body: str,
        sibling_overloads: Dict[str, List[_MetalStructMethod]],
        local_view: Dict[str, str],
        buffer_view: _MetalBufferTypeView,
        instantiated_template_functions: Dict[str, str],
        template_methods_by_struct: Dict[str, Dict[str, List[_MetalStructMethod]]],
        local_integral_constants: Dict[str, List[_MetalIntegralConstantBinding]],
        local_type_aliases: Dict[str, List[_MetalTypeAliasBinding]],
        *,
        methods_by_struct: Dict[str, Dict[str, _MetalStructMethod]],
        operator_call_structs: Set[str],
        rewrite_structs_by_name: Dict[str, _MetalStructDefinition],
    ) -> Optional[Tuple[int, str]]:
        # A receiver-less `name(args)` where `name` is a sibling member method
        # (template OR concrete) is lowered to its concrete free function with an
        # implicit `self` receiver. A member access (`x.name(...)`) is excluded.
        # Skip a `.name(`/`->name(` member access — handled by the normal call
        # rewriter at the top level, not as an implicit-this call here.
        previous = ident_end - len(ident) - 1
        while previous >= 0 and body[previous].isspace():
            previous -= 1
        # `X::template name<A>(...)` and `x.template name<A>(...)` name a
        # member of `X`/`x`: the `template` disambiguator precedes the name.
        if body[max(0, previous - 7) : previous + 1] == "template" and (
            previous < 8
            or not (body[previous - 8].isalnum() or body[previous - 8] == "_")
        ):
            previous -= 8
            while previous >= 0 and body[previous].isspace():
                previous -= 1
        if previous >= 0 and (
            body[previous] == "."
            or (previous >= 1 and body[previous - 1 : previous + 1] in {"->", "::"})
        ):
            return None

        # An implicit-this `operator()(args)` call (e.g. the sizeof==8 reduction
        # combining elements). The declarator is `operator` then an empty `()`
        # then the real argument list.
        if ident == "operator":
            after_operator = ident_end
            while after_operator < len(body) and body[after_operator].isspace():
                after_operator += 1
            if (
                after_operator < len(body)
                and body[after_operator] == "("
                and body[after_operator + 1 : after_operator + 2] == ")"
            ):
                return self._rewrite_internal_operator_call(
                    struct,
                    method,
                    after_operator,
                    body,
                    sibling_overloads,
                    local_view,
                    buffer_view,
                    instantiated_template_functions,
                    template_methods_by_struct,
                    local_integral_constants,
                    local_type_aliases,
                    methods_by_struct=methods_by_struct,
                    operator_call_structs=operator_call_structs,
                    rewrite_structs_by_name=rewrite_structs_by_name,
                )
            return None

        template_overloads = sibling_overloads.get(ident)
        concrete_siblings = [m for m in struct.methods if m.name == ident]
        if not template_overloads and not concrete_siblings:
            return None
        call_suffix = self._member_template_call_suffix(body, ident_end)
        if call_suffix is None:
            return None
        arg_open, explicit_template_arguments = call_suffix
        arg_close = self._find_matching_delimiter(body, arg_open, "(", ")")
        if arg_close is None:
            return None
        raw_args = body[arg_open + 1 : arg_close]
        args = raw_args.strip()

        # A CONCRETE sibling has a fixed signature; lower it directly (no overload
        # selection / argument typing needed). When both a concrete and a template
        # sibling share the name, prefer the concrete one only if there is exactly
        # one — otherwise fall through to the template path / clean-fail.
        if (
            concrete_siblings
            and not template_overloads
            and explicit_template_arguments is None
        ):
            reference_methods = [
                candidate
                for candidate in concrete_siblings
                if self._method_returns_lvalue_reference(candidate)
            ]
            if reference_methods:
                concrete = self._select_direct_reference_method(
                    body,
                    struct,
                    reference_methods,
                    arg_open,
                    receiver_is_const=method.is_const,
                )
                return self._build_direct_reference_accessor_rewrite(
                    body,
                    struct,
                    concrete,
                    receiver="self",
                    arg_open=arg_open,
                    call_start=ident_end - len(ident),
                    enclosing_return_type=method.return_type,
                )
            if len(concrete_siblings) != 1:
                return None
            concrete = concrete_siblings[0]
            self._reject_reference_returning_method_call(
                body,
                struct,
                concrete,
                arg_open,
                arg_close,
            )
            if concrete.is_static:
                replacement = (
                    f"{concrete.free_name}({args})"
                    if args
                    else f"{concrete.free_name}()"
                )
            elif args:
                replacement = f"{concrete.free_name}(self, {args})"
            else:
                replacement = f"{concrete.free_name}(self)"
            return arg_close + 1, replacement

        overloads = template_overloads
        call_arguments = [
            argument
            for argument in self._split_top_level_commas(raw_args)
            if argument.strip()
        ]
        representative = overloads[0]
        signature = f"{struct.name}::{method.name} -> {ident}({raw_args.strip()})"
        local_constant_values = self._local_integral_constants_at(
            local_integral_constants,
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
        concrete_argument_types: List[Optional[str]] = []
        for argument_index, argument in enumerate(call_arguments):
            inference_argument = self._substitute_template_argument_static_constants(
                argument,
                local_constant_values,
            )
            inference_argument = fold_argument_static_members(
                self, inference_argument, rewrite_structs_by_name
            )
            inferred = self._infer_argument_type(
                inference_argument,
                buffer_view,
                local_view,
                {},
            )
            if inferred is None:
                if not self._template_member_argument_requires_inference(
                    representative,
                    argument_index,
                    explicit_template_arguments,
                ):
                    concrete_argument_types.append(None)
                    continue
                raise MetalStructMethodError(
                    "Cannot lower template member method "
                    f"'{struct.name}::{method.name}': internal call to "
                    f"'{ident}' has argument '{argument.strip()}' whose type "
                    f"could not be inferred. Requested call: {signature}.",
                    struct_name=struct.name,
                    method_name=method.name,
                    requested_signature=signature,
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

        free_name = self._instantiate_template_member_overload(
            struct,
            overloads,
            concrete_argument_types,
            signature,
            instantiated_template_functions,
            template_methods_by_struct,
            explicit_template_arguments=resolved_explicit_template_arguments,
            argument_type_aliases=local_type_aliases,
            argument_type_position=arg_open,
            argument_type_fallback_position=method.span[0],
            methods_by_struct=methods_by_struct,
            operator_call_structs=operator_call_structs,
            rewrite_structs_by_name=rewrite_structs_by_name,
        )
        args = self._expanded_template_member_call_arguments(free_name, raw_args)
        if representative.is_static:
            replacement = f"{free_name}({args})" if args else f"{free_name}()"
        elif args:
            replacement = f"{free_name}(self, {args})"
        else:
            replacement = f"{free_name}(self)"
        return arg_close + 1, replacement

    def _template_member_argument_requires_inference(
        self,
        method: _MetalStructMethod,
        argument_index: int,
        explicit_template_arguments: Optional[List[str]] = None,
    ) -> bool:
        parameters = [
            parameter
            for parameter in self._split_top_level_commas(method.parameters)
            if parameter.strip()
        ]
        if argument_index >= len(parameters):
            return True

        parameter_type = self._normalize_function_parameter_type_text(
            parameters[argument_index]
        )
        referenced = {
            name
            for name in method.template_parameters
            if re.search(rf"\b{re.escape(name)}\b", parameter_type)
        }
        if not referenced:
            return False

        explicit_count = len(explicit_template_arguments or [])
        available_without_inference = set(method.template_parameters[:explicit_count])
        return not referenced.issubset(available_without_inference)

    def _member_template_call_suffix(
        self, code: str, name_end: int
    ) -> Optional[Tuple[int, Optional[List[str]]]]:
        """Return a member call's argument opener and explicit template args."""
        index = name_end
        while index < len(code) and code[index].isspace():
            index += 1

        explicit_template_arguments: Optional[List[str]] = None
        if index < len(code) and code[index] == "<":
            angle_end = self._find_matching_angle(code, index)
            if angle_end is None:
                return None
            argument_source = code[index + 1 : angle_end]
            if self._find_next_top_level_char(argument_source, 0, ";") is not None:
                return None
            explicit_template_arguments = [
                argument.strip()
                for argument in self._split_top_level_commas(
                    code[index + 1 : angle_end]
                )
                if argument.strip()
            ]
            index = angle_end + 1
            while index < len(code) and code[index].isspace():
                index += 1

        if index >= len(code) or code[index] != "(":
            return None
        return index, explicit_template_arguments

    def _rewrite_internal_operator_call(
        self,
        struct: _MetalStructDefinition,
        method: _MetalStructMethod,
        empty_paren_open: int,
        body: str,
        sibling_overloads: Dict[str, List[_MetalStructMethod]],
        local_view: Dict[str, str],
        buffer_view: _MetalBufferTypeView,
        instantiated_template_functions: Dict[str, str],
        template_methods_by_struct: Dict[str, Dict[str, List[_MetalStructMethod]]],
        local_integral_constants: Dict[str, List[_MetalIntegralConstantBinding]],
        local_type_aliases: Dict[str, List[_MetalTypeAliasBinding]],
        *,
        methods_by_struct: Dict[str, Dict[str, _MetalStructMethod]],
        operator_call_structs: Set[str],
        rewrite_structs_by_name: Dict[str, _MetalStructDefinition],
    ) -> Optional[Tuple[int, str]]:
        # Lower an implicit-this `operator()(args)` call. `empty_paren_open` is the
        # `(` of the empty `()` after `operator`; the real argument list follows.
        empty_close = self._find_matching_delimiter(body, empty_paren_open, "(", ")")
        if empty_close is None:
            return None
        arg_open = empty_close + 1
        while arg_open < len(body) and body[arg_open].isspace():
            arg_open += 1
        if arg_open >= len(body) or body[arg_open] != "(":
            return None
        arg_close = self._find_matching_delimiter(body, arg_open, "(", ")")
        if arg_close is None:
            return None
        raw_args = body[arg_open + 1 : arg_close]
        args = raw_args.strip()
        # The caller has not yet emitted the `operator` token, so the returned
        # replacement fully replaces `operator()(args)` from that token onward;
        # only the end offset is needed.
        template_overloads = sibling_overloads.get("operator()")
        positioned_buffer_types = {
            name: [(0, value)] for name, value in buffer_view.items()
        }
        positioned_local_types = {
            name: [(0, value)] for name, value in local_view.items()
        }
        concrete = self._select_concrete_method_for_call(
            body,
            struct,
            "operator()",
            arg_open,
            positioned_buffer_types,
            positioned_local_types,
            positioned_local_types,
            {struct.name: struct},
            allow_template_fallback=bool(template_overloads),
            template_overloads=template_overloads,
            require_static=False,
            type_alias_fallback_position=method.span[0],
            receiver_contract=self._method_body_receiver_contract(struct, method),
            call_start=max(0, empty_paren_open - len("operator")),
        )
        if concrete is not None:
            self._reject_reference_returning_method_call(
                body,
                struct,
                concrete,
                arg_open,
                arg_close,
            )
            replacement = (
                f"{concrete.free_name}(self, {args})"
                if args
                else f"{concrete.free_name}(self)"
            )
            return arg_close + 1, replacement
        if not template_overloads:
            return None
        call_arguments = [
            argument
            for argument in self._split_top_level_commas(raw_args)
            if argument.strip()
        ]
        signature = f"{struct.name}::{method.name} -> operator()({args})"
        local_constant_values = self._local_integral_constants_at(
            local_integral_constants,
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
                self, inference_argument, rewrite_structs_by_name
            )
            inferred = self._infer_argument_type(
                inference_argument,
                buffer_view,
                local_view,
                {},
            )
            if inferred is None:
                raise MetalStructMethodError(
                    "Cannot lower template member method "
                    f"'{struct.name}::{method.name}': internal call to "
                    f"'operator()' has argument '{argument.strip()}' whose type "
                    f"could not be inferred. Requested call: {signature}.",
                    struct_name=struct.name,
                    method_name=method.name,
                    requested_signature=signature,
                )
            concrete_argument_types.append(inferred)
        free_name = self._instantiate_template_member_overload(
            struct,
            template_overloads,
            concrete_argument_types,
            signature,
            instantiated_template_functions,
            template_methods_by_struct,
            argument_type_aliases=local_type_aliases,
            argument_type_position=arg_open,
            argument_type_fallback_position=method.span[0],
            methods_by_struct=methods_by_struct,
            operator_call_structs=operator_call_structs,
            rewrite_structs_by_name=rewrite_structs_by_name,
        )
        args = self._expanded_template_member_call_arguments(free_name, raw_args)
        replacement = f"{free_name}(self, {args})" if args else f"{free_name}(self)"
        return arg_close + 1, replacement

    def _build_template_call_rewrite(
        self,
        code: str,
        receiver: Optional[str],
        free_name: str,
        arg_open: int,
        arg_close: int,
        *,
        raw_arguments: Optional[str] = None,
    ) -> Tuple[int, str]:
        args = self._expanded_template_member_call_arguments(
            free_name,
            (
                code[arg_open + 1 : arg_close]
                if raw_arguments is None
                else raw_arguments
            ),
        )
        if receiver is None:
            # Static template member call: no `self` receiver.
            replacement = f"{free_name}({args})" if args else f"{free_name}()"
        elif args:
            replacement = f"{free_name}({receiver}, {args})"
        else:
            replacement = f"{free_name}({receiver})"
        return arg_close + 1, replacement

    def _expanded_template_member_call_arguments(
        self,
        free_name: str,
        raw_arguments: str,
    ) -> str:
        arguments = [
            argument.strip()
            for argument in self._split_top_level_commas(raw_arguments)
            if argument.strip()
        ]
        selection = self._instantiated_template_member_calls.get(free_name)
        if selection is None:
            return raw_arguments.strip()
        struct, method, bindings, structs_by_name = selection
        parameters = [
            parameter
            for parameter in self._split_top_level_commas(method.parameters)
            if parameter.strip()
        ]
        if len(arguments) >= len(parameters):
            return raw_arguments.strip()
        for parameter in parameters[len(arguments) :]:
            declaration, default = self._split_top_level_assignment(parameter)
            if default is None or not default.strip():
                return raw_arguments.strip()
            resolved_default = self._replace_identifiers(default, bindings).strip()
            resolved_default = self._substitute_static_data_member_initializers(
                struct, method, resolved_default
            )
            resolved_default = self._replace_identifiers(
                resolved_default, bindings
            ).strip()
            if resolved_default == "{}":
                type_name = self._function_parameter_value_type(declaration)
                type_name = self._replace_identifiers(type_name, bindings)
                type_name = self._canonicalize_struct_scoped_type(
                    type_name, struct, structs_by_name
                )
                resolved_default = f"{type_name}{{}}"
            arguments.append(resolved_default)
        return ", ".join(arguments)

    def _method_accepts_arguments(self, method: _MetalStructMethod, count: int) -> bool:
        # parameter_names omits unnamed parameters (tag types such as
        # metal::bool_constant<B>), so count the declared parameters instead.
        return accepts_argument_count(
            method.parameters,
            count,
            self._split_top_level_commas,
            self._split_top_level_assignment,
        )

    def _method_returns_lvalue_reference(self, method: _MetalStructMethod) -> bool:
        return bool(
            re.search(
                r"(?<!&)\&\s*$",
                self._normalize_template_argument_text(method.return_type),
            )
        )

    def _concrete_reference_methods(
        self,
        struct: _MetalStructDefinition,
        method_name: str,
        code: Optional[str] = None,
        arg_open: Optional[int] = None,
    ) -> List[_MetalStructMethod]:
        methods = [
            method
            for method in struct.methods
            if method.name == method_name
            and self._method_returns_lvalue_reference(method)
        ]
        if not methods or code is None or arg_open is None:
            return methods
        # A call no concrete reference overload can take (for example one that
        # passes a tag argument to a member template) is resolved by the
        # template path instead of being refused here.
        arg_close = self._find_matching_delimiter(code, arg_open, "(", ")")
        if arg_close is None:
            return methods
        count = len(
            [
                argument
                for argument in self._split_top_level_commas(
                    code[arg_open + 1 : arg_close]
                )
                if argument.strip()
            ]
        )
        if any(self._method_accepts_arguments(method, count) for method in methods):
            return methods
        return []

    def _select_direct_reference_method(
        self,
        code: str,
        struct: _MetalStructDefinition,
        methods: List[_MetalStructMethod],
        arg_open: int,
        *,
        receiver_is_const: Optional[bool],
    ) -> _MetalStructMethod:
        arg_close = self._find_matching_delimiter(code, arg_open, "(", ")")
        representative = methods[0]
        if arg_close is None:
            self._reject_reference_returning_method_call(
                code, struct, representative, arg_open, arg_open
            )
            raise AssertionError("reference-return rejection must raise")

        arguments = [
            argument
            for argument in self._split_top_level_commas(code[arg_open + 1 : arg_close])
            if argument.strip()
        ]
        if receiver_is_const is True:
            candidates = [
                method
                for method in methods
                if not method.is_static
                and method.is_const
                and re.search(r"\bconst\b", method.return_type) is not None
                and self._method_accepts_arguments(method, len(arguments))
            ]
        else:
            candidates = [
                method
                for method in methods
                if not method.is_static
                and not method.is_const
                and re.search(r"\bconst\b", method.return_type) is None
                and self._method_accepts_arguments(method, len(arguments))
            ]
        if receiver_is_const is not None and len(candidates) == 1:
            candidate = candidates[0]
            normalized_parameters = self._normalize_template_argument_text(
                candidate.parameters
            )
            competing_overloads = [
                method
                for method in [*struct.methods, *struct.template_methods]
                if method is not candidate
                and method.name == candidate.name
                and self._method_accepts_arguments(method, len(arguments))
            ]
            matching_counterparts = [
                method
                for method in competing_overloads
                if method.is_const != candidate.is_const
                and not method.is_static
                and any(method is concrete for concrete in struct.methods)
                and self._normalize_template_argument_text(method.parameters)
                == normalized_parameters
            ]
            if len(matching_counterparts) <= 1 and len(matching_counterparts) == len(
                competing_overloads
            ):
                return candidate

        self._reject_reference_returning_method_call(
            code,
            struct,
            candidates[0] if candidates else representative,
            arg_open,
            arg_close,
        )
        raise AssertionError("reference-return rejection must raise")

    def _simple_receiver_constness(
        self,
        code: str,
        struct_name: str,
        receiver: str,
        call_start: int,
        *,
        type_aliases: Optional[Dict[str, List[_MetalTypeAliasBinding]]] = None,
    ) -> Optional[bool]:
        contract = self._receiver_contract_for_named_value(
            code,
            struct_name,
            receiver,
            call_start,
            type_aliases=type_aliases,
        )
        if contract is None or "constant" in contract.address_spaces:
            return None
        return contract.is_readonly

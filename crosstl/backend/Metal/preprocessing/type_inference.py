"""Expression type inference for pointer arithmetic, braced construction, operator calls and binary arithmetic."""

import re
from typing import (
    Dict,
    List,
    Optional,
    Set,
    Tuple,
)

from ..arithmetic_conversions import builtin_arithmetic_conversion
from ..array_element_receivers import array_element_receiver_end
from .support import (
    IDENTIFIER_RE,
    _MetalBufferTypeView,
    _MetalStructDefinition,
    _MetalStructMethod,
    _MetalSubscriptableType,
)


class MetalTypeInferenceMixin:
    """Members of ``MetalPreprocessor``: expression type inference for pointer arithmetic, braced construction, operator calls and binary arithmetic."""

    def _normalize_inferred_type(self, type_text: str) -> str:
        # Collapse whitespace and strip leading cv/address-space qualifiers that
        # do not change the value type used for template binding.
        text = self._normalize_template_argument_text(type_text or "")
        if not text:
            return ""
        tokens = text.split(" ")
        dropped = {
            "const",
            "constexpr",
            "device",
            "constant",
            "thread",
            "threadgroup",
            "volatile",
            "mutable",
        }
        kept = [token for token in tokens if token not in dropped]
        return " ".join(kept).strip()

    def _normalize_known_address_space_pointer_type(
        self, type_text: str
    ) -> Optional[str]:
        # Address-of inference may expose a pointer only when its complete Metal
        # address space is known. Multi-level pointers and conflicting address
        # spaces stay unsupported because reconstructing either would be a guess.
        normalized = self._normalize_template_argument_text(type_text)
        if (
            not normalized.endswith("*")
            or normalized.count("*") != 1
            or "&" in normalized
        ):
            return None
        address_spaces = [
            token
            for token in IDENTIFIER_RE.findall(normalized[:-1])
            if token in {"device", "constant", "thread", "threadgroup"}
        ]
        if len(address_spaces) != 1:
            return None
        pointee_type = self._normalize_inferred_type(normalized[:-1])
        if not pointee_type or pointee_type.endswith("*"):
            return None
        return normalized

    def _normalize_inferred_expression_type(self, type_text: str) -> Optional[str]:
        # Value expressions discard top-level cv/address-space qualifiers and
        # references during template binding: a named `thread U& total` has value
        # type U when passed to a by-value parameter. Pointer expressions cannot
        # discard pointee qualifiers or address spaces, so after reference decay
        # preserve one fully qualified pointer and reject every other
        # pointer-bearing spelling rather than normalizing it to `T*`.
        normalized = self._normalize_template_argument_text(type_text or "")
        if not normalized:
            return None
        reference = re.search(r"\s*(?:&&|&)\s*$", normalized)
        if reference is not None:
            normalized = normalized[: reference.start()].strip()
            if not normalized:
                return None
        if "*" in normalized:
            return self._normalize_known_address_space_pointer_type(normalized)
        value_type = self._normalize_inferred_type(normalized)
        return value_type or None

    def _record_known_member_function_return_type(
        self, function_name: str, return_type: str
    ) -> None:
        normalized = self._normalize_inferred_expression_type(return_type)
        if normalized is None:
            self._known_member_function_return_types.pop(function_name, None)
            return
        self._known_member_function_return_types[function_name] = normalized

    def _normalize_template_binding_type(self, type_text: str) -> str:
        return self._normalize_inferred_expression_type(type_text) or ""

    def _pointer_pointee_value_type(self, pointer_type: str) -> Optional[str]:
        normalized = self._normalize_known_address_space_pointer_type(pointer_type)
        if normalized is None:
            return None
        pointee = self._normalize_inferred_type(normalized[:-1])
        return pointee or None

    def _canonical_template_binding_pointee_type(
        self, type_text: str, *, context: Optional[str] = None
    ) -> str:
        normalized = self._normalize_inferred_type(type_text)
        specialization = self._materialized_struct_specializations.get(normalized)
        primary = self._materialized_struct_primary_templates.get(
            normalized
            if specialization is not None
            else self._normalize_inferred_type(context or "")
        )
        if specialization is not None:
            source_name, source_arguments = specialization
            normalized = f"{source_name}<{', '.join(source_arguments)}>"
        template_id = self._exact_template_type_id(normalized)
        if primary is not None and template_id is not None:
            name, arguments = template_id
            qualified_name = (
                f"{primary.namespace}::{primary.name}"
                if primary.namespace
                else primary.name
            )
            if name in {primary.name, qualified_name}:
                if not primary.variadic_template_parameters and len(arguments) > len(
                    primary.template_parameters
                ):
                    return self._normalize_template_argument_text(normalized)
                # Deduction must compare the primary's argument slots, even when
                # materialization selected a partial specialization's body.
                resolved = self._template_arguments_with_resolved_defaults(
                    primary, arguments
                )
                if resolved is not None:
                    normalized = f"{primary.name}<{', '.join(resolved)}>"
        return self._normalize_template_argument_text(normalized)

    @staticmethod
    def _pointer_argument_matches_declared_type(
        declared_pointer_type: str, concrete_pointer_type: str
    ) -> bool:
        address_space_tokens = {"device", "constant", "thread", "threadgroup"}
        declared_tokens = set(IDENTIFIER_RE.findall(declared_pointer_type))
        concrete_tokens = set(IDENTIFIER_RE.findall(concrete_pointer_type))
        declared_spaces = declared_tokens & address_space_tokens
        concrete_spaces = concrete_tokens & address_space_tokens
        if len(declared_spaces) != 1 or declared_spaces != concrete_spaces:
            return False
        # Pointee qualification may be added by the declared parameter, but it
        # may not be discarded from the concrete argument.
        for qualifier in ("const", "volatile"):
            if qualifier in concrete_tokens and qualifier not in declared_tokens:
                return False
        return True

    @staticmethod
    def _buffer_element_type(
        buffer_element_types: _MetalBufferTypeView, name: str
    ) -> Optional[str]:
        entry = buffer_element_types.get(name)
        if isinstance(entry, _MetalSubscriptableType):
            return entry.element_type
        return entry if isinstance(entry, str) and entry else None

    def _buffer_pointer_type(
        self, buffer_element_types: _MetalBufferTypeView, name: str
    ) -> Optional[str]:
        entry = buffer_element_types.get(name)
        if isinstance(entry, _MetalSubscriptableType):
            return self._normalize_known_address_space_pointer_type(
                entry.pointer_type or ""
            )
        return None

    def _buffer_pointer_expression_type(
        self, buffer_element_types: _MetalBufferTypeView, name: str
    ) -> Optional[str]:
        # Structured entries distinguish a proven pointer from an element type
        # whose address space is unknown. Plain strings remain supported for the
        # legacy unit-level element maps used by callers of this private helper.
        entry = buffer_element_types.get(name)
        if isinstance(entry, _MetalSubscriptableType):
            return self._normalize_known_address_space_pointer_type(
                entry.pointer_type or ""
            )
        if not isinstance(entry, str) or not entry or "*" in entry:
            return None
        return entry

    def _is_metal_scalar_or_vector_type(self, type_text: str) -> bool:
        return self._scalar_and_width(type_text) is not None

    def _configure_inferred_bitcast_ownership(self, code: str) -> None:
        references = self._find_static_constexpr_calls(code, {"as_type"})
        self._inferred_bitcast_shadowed = False
        if not references:
            return
        template_spans = self._find_template_declaration_spans(code)
        functions = self._find_non_template_function_definitions(code, template_spans)
        templates = self._find_template_functions(code)
        declarations = self._find_free_function_overload_declarations(
            code,
            references,
            excluded_spans=[f.body_span for f in functions]
            + [(t.body_start, t.span[1]) for t in templates],
            template_spans=template_spans,
            namespace_visibility=self._metal_namespace_visibility(code),
        )
        self._inferred_bitcast_shadowed = bool(declarations)

    def _vector_local_component_type(
        self, base: str, local_variable_types: Dict[str, str]
    ) -> Optional[str]:
        """Component type of ``v[i]`` for a local or parameter ``v`` of Metal vector type."""
        if IDENTIFIER_RE.fullmatch(base) is None or base not in local_variable_types:
            return None
        declared = local_variable_types[base]
        declared = getattr(declared, "value_type", declared)
        normalized = self._normalize_inferred_expression_type(declared or "")
        if not normalized or "*" in normalized:
            return None
        parts = self._scalar_and_width(normalized)
        if parts is None or parts[1] not in (2, 3, 4):
            return None
        return parts[0]

    def _infer_array_element_member_type(
        self,
        expr: str,
        buffer_element_types: _MetalBufferTypeView,
        local_variable_types: Dict[str, str],
        struct_field_types: Optional[Dict[str, Dict[str, str]]],
        structs_by_name: Optional[Dict[str, "_MetalStructDefinition"]],
    ) -> Optional[str]:
        """Type ``items[i].member...`` as ``member...`` of the element struct.

        The element of a tracked array or pointer whose element type is a
        concrete struct is an object of that struct ([expr.sub]), so the rest
        of the expression is typed through a stand-in receiver carrying the
        struct's field types.
        """
        if structs_by_name is None:
            return None
        match = re.match(r"(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*\[", expr)
        if match is None:
            return None
        element_end = array_element_receiver_end(expr, match.end() - 1, 1)
        if element_end is None:
            return None
        element_type = self._buffer_element_type(
            buffer_element_types, match.group("name")
        )
        if not element_type or re.search(r"[*&\[]", element_type):
            return None
        struct = structs_by_name.get(self._normalize_inferred_type(element_type))
        if struct is None or not struct.data_member_types:
            return None
        stand_in = "__crosstl_array_element"
        field_types = dict(struct_field_types or {})
        field_types[stand_in] = {
            member: self._canonicalize_struct_scoped_type(
                declared_type, struct, structs_by_name
            )
            for member, declared_type in struct.data_member_types.items()
        }
        return self._infer_argument_type(
            stand_in + expr[element_end:],
            buffer_element_types,
            local_variable_types,
            field_types,
            structs_by_name,
        )

    def _infer_addressed_buffer_element_pointer_type(
        self,
        expr: str,
        buffer_element_types: _MetalBufferTypeView,
        local_variable_types: Dict[str, str],
        struct_field_types: Optional[Dict[str, Dict[str, str]]],
        structs_by_name: Optional[Dict[str, "_MetalStructDefinition"]],
    ) -> Optional[str]:
        # Accepted shape: `&base[index]` with optional enclosing parentheses
        # around the subscript. `base` must be a bare, tracked subscriptable whose
        # declaration retained one explicit Metal address space. The index must
        # be a side-effect-free integral expression; calls, assignment,
        # increment/decrement, comma/conditional expressions, nested subscripts,
        # and precedence-ambiguous forms are rejected.
        if not expr.startswith("&") or expr.startswith("&&"):
            return None
        operand = self._strip_enclosing_parens(expr[1:].strip())
        bracket = operand.find("[")
        if bracket <= 0 or not operand.endswith("]"):
            return None
        close = self._find_matching_delimiter(operand, bracket, "[", "]")
        if close != len(operand) - 1:
            return None
        base = operand[:bracket].strip()
        if IDENTIFIER_RE.fullmatch(base) is None:
            return None
        pointer_type = self._buffer_pointer_type(buffer_element_types, base)
        if pointer_type is None:
            return None
        # `&rows[i]` of `T rows[N][M]` points to a row array, not to a `T`.
        if "[" in (self._buffer_element_type(buffer_element_types, base) or ""):
            return None
        index = operand[bracket + 1 : close].strip()
        if not self._direct_reference_expression_is_safe(
            index
        ) or self._static_integral_expression_has_ambiguous_precedence(index):
            return None
        index_type = self._infer_argument_type(
            index,
            buffer_element_types,
            local_variable_types,
            struct_field_types,
            structs_by_name,
        )
        if index_type is None or not self._is_integral_concrete_type(index_type):
            return None
        return pointer_type

    def _infer_known_member_function_call_type(self, expr: str) -> Optional[str]:
        match = re.match(r"(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*\(", expr)
        if match is None:
            return None
        return_type = self._known_member_function_return_types.get(match.group("name"))
        if return_type is None:
            return None
        paren_start = expr.find("(", match.start("name") + len(match.group("name")))
        paren_end = self._find_matching_delimiter(expr, paren_start, "(", ")")
        if paren_end != len(expr) - 1:
            return None
        return self._normalize_inferred_expression_type(return_type)

    def _infer_pointer_arithmetic_type(
        self,
        expr: str,
        buffer_element_types: _MetalBufferTypeView,
        local_variable_types: Dict[str, str],
        struct_field_types: Optional[Dict[str, Dict[str, str]]],
        structs_by_name: Optional[Dict[str, "_MetalStructDefinition"]],
    ) -> Optional[str]:
        expression = self._strip_enclosing_parens(expr)
        if IDENTIFIER_RE.fullmatch(expression):
            if expression in buffer_element_types:
                return self._buffer_pointer_expression_type(
                    buffer_element_types, expression
                )
            if expression in local_variable_types:
                return self._normalize_known_address_space_pointer_type(
                    local_variable_types[expression]
                )
            return None
        split = self._split_top_level_binary_arithmetic(expression)
        if split is None:
            return None
        left_expr, operator, right_expr = split
        if operator not in {"+", "-"}:
            return None
        left_expr = left_expr.strip()
        right_expr = right_expr.strip()
        left_pointer = self._infer_pointer_arithmetic_type(
            left_expr,
            buffer_element_types,
            local_variable_types,
            struct_field_types,
            structs_by_name,
        )
        right_pointer = self._infer_pointer_arithmetic_type(
            right_expr,
            buffer_element_types,
            local_variable_types,
            struct_field_types,
            structs_by_name,
        )
        if left_pointer is not None and right_pointer is None:
            right_type = self._infer_argument_type(
                right_expr,
                buffer_element_types,
                local_variable_types,
                struct_field_types,
                structs_by_name,
            )
            if right_type is not None and self._is_integral_concrete_type(right_type):
                return left_pointer
        if operator == "+" and right_pointer is not None and left_pointer is None:
            left_type = self._infer_argument_type(
                left_expr,
                buffer_element_types,
                local_variable_types,
                struct_field_types,
                structs_by_name,
            )
            if left_type is not None and self._is_integral_concrete_type(left_type):
                return right_pointer
        return None

    def _contains_bare_pointer_operand(
        self,
        expr: str,
        buffer_element_types: _MetalBufferTypeView,
    ) -> bool:
        expression = self._strip_enclosing_parens(expr)
        if IDENTIFIER_RE.fullmatch(expression):
            return expression in buffer_element_types
        split = self._split_top_level_binary_arithmetic(expression)
        if split is None:
            return False
        left_expr, _operator, right_expr = split
        return self._contains_bare_pointer_operand(
            left_expr, buffer_element_types
        ) or self._contains_bare_pointer_operand(right_expr, buffer_element_types)

    def _inferred_aggregate_type(
        self,
        type_name: str,
        structs_by_name: Optional[Dict[str, "_MetalStructDefinition"]],
    ) -> Optional[str]:
        """Resolve one aggregate spelling without guessing across alias scopes.

        Selected-entry preprocessing materializes aliases such as MLX's
        ``complex64_t = complex_t<float>`` to a concrete struct name before
        member-call rewriting. An ``auto`` initializer can still retain the
        source alias spelling, however. Admit that spelling only when every
        collected binding resolves to the same concrete retained struct; local
        shadowing with different targets therefore remains fail-closed.
        """

        if not structs_by_name:
            return None
        normalized = self._normalize_inferred_type(type_name)
        if normalized in structs_by_name:
            return normalized
        bindings = getattr(self, "_source_type_alias_bindings", {}).get(normalized, ())
        targets = {
            self._normalize_inferred_type(binding.target)
            for binding in bindings
            if getattr(binding, "target", None)
        }
        if len(targets) != 1:
            return None
        target = next(iter(targets))
        return target if target in structs_by_name else None

    def _infer_braced_construction_type(
        self,
        expr: str,
        structs_by_name: Optional[Dict[str, "_MetalStructDefinition"]],
    ) -> Optional[str]:
        # `T{...}` (brace-init construction, no trailing call) has type T when T
        # is a recognized scalar/vector type or a tracked struct/union. The brace
        # group must span the whole expression, so a functor construction-and-call
        # `T{...}(args)` (handled earlier) is not misread as a bare construction.
        match = re.match(r"(?P<name>[A-Za-z_][A-Za-z0-9_]*)", expr)
        if match is None:
            return None
        index = match.end()
        while index < len(expr) and expr[index].isspace():
            index += 1
        type_name = match.group("name")
        is_template_id = False
        if index < len(expr) and expr[index] == "<":
            angle_close = self._find_matching_angle(expr, index)
            if angle_close is None:
                return None
            type_name = self._normalize_template_argument_text(expr[: angle_close + 1])
            index = angle_close + 1
            while index < len(expr) and expr[index].isspace():
                index += 1
            is_template_id = True
        if index >= len(expr) or expr[index] != "{":
            return None
        brace_open = index
        brace_close = self._find_matching_delimiter(expr, brace_open, "{", "}")
        if brace_close != len(expr) - 1:
            return None
        if is_template_id:
            return type_name
        if type_name in self._METAL_SCALAR_VECTOR_TYPES:
            return self._normalize_inferred_type(type_name)
        aggregate_type = self._inferred_aggregate_type(type_name, structs_by_name)
        if aggregate_type is not None:
            return aggregate_type
        return None

    def _infer_functor_construction_call_type(
        self,
        expr: str,
        buffer_element_types: _MetalBufferTypeView,
        local_variable_types: Dict[str, str],
        struct_field_types: Optional[Dict[str, Dict[str, str]]],
        structs_by_name: Optional[Dict[str, "_MetalStructDefinition"]],
    ) -> Optional[str]:
        # A default-constructed functor temporary that is immediately CALLED:
        # `F{}(args)`, `F{...}(args)` or `F()(args)`. Its type is the result type
        # of `F::operator()` for the (recursively inferred) argument types. The
        # shape is a leading identifier, a construction group (`{...}` or `()`),
        # then a call group `(args)` that closes the whole expression.
        if not structs_by_name:
            return None
        match = re.match(r"(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*(?P<open>[{(])", expr)
        if match is None:
            return None
        struct = structs_by_name.get(match.group("name"))
        if struct is None:
            return None
        open_char = match.group("open")
        open_pos = match.start("open")
        close_char = "}" if open_char == "{" else ")"
        constructor_close = self._find_matching_delimiter(
            expr, open_pos, open_char, close_char
        )
        if constructor_close is None:
            return None
        call_open = constructor_close + 1
        while call_open < len(expr) and expr[call_open].isspace():
            call_open += 1
        if call_open >= len(expr) or expr[call_open] != "(":
            return None
        call_close = self._find_matching_delimiter(expr, call_open, "(", ")")
        if call_close != len(expr) - 1:
            return None
        raw_args = expr[call_open + 1 : call_close]
        argument_exprs = [
            argument
            for argument in self._split_top_level_commas(raw_args)
            if argument.strip()
        ]
        return self._infer_functor_operator_call_result_type(
            struct,
            argument_exprs,
            buffer_element_types,
            local_variable_types,
            struct_field_types,
            structs_by_name,
        )

    def _infer_functor_operator_call_result_type(
        self,
        struct: "_MetalStructDefinition",
        argument_exprs: List[str],
        buffer_element_types: _MetalBufferTypeView,
        local_variable_types: Dict[str, str],
        struct_field_types: Optional[Dict[str, Dict[str, str]]],
        structs_by_name: Optional[Dict[str, "_MetalStructDefinition"]],
    ) -> Optional[str]:
        # Conservatively resolve the result type of `struct::operator()(args)`.
        concrete_ops = [method for method in struct.methods if method.is_operator_call]
        template_ops = [
            method for method in struct.template_methods if method.is_operator_call
        ]
        if not concrete_ops and not template_ops:
            return None

        argument_types = [
            self._infer_argument_type(
                argument,
                buffer_element_types,
                local_variable_types,
                struct_field_types,
                structs_by_name,
            )
            for argument in argument_exprs
        ]
        normalized_args = [
            self._normalize_inferred_type(argument) if argument is not None else None
            for argument in argument_types
        ]

        # 1. A concrete operator() overload whose declared parameter types match
        #    the inferred argument types -> its concrete return type (e.g. the
        #    `complex64_t operator()(complex64_t)` overload of Sqrt/Log/Abs).
        if all(argument is not None for argument in normalized_args):
            for method in concrete_ops:
                declared = self._operator_call_parameter_types(method)
                if declared == normalized_args:
                    result = self._normalize_inferred_expression_type(
                        method.return_type
                    )
                    if result is not None:
                        return result

        # 2. A template operator() whose parameters bind from the inferred
        #    argument types -> its return type with those bindings applied. Covers
        #    the identity `T operator()(T x)` (result == argument type) and any
        #    return whose template parameters are all deducible from the call.
        if argument_types and all(argument is not None for argument in argument_types):
            for method in template_ops:
                bindings = self._bind_template_method_parameters(
                    method,
                    list(argument_types),
                    owner_struct=struct,
                    structs_by_name=structs_by_name,
                )
                if bindings is None:
                    continue
                substituted = self._normalize_inferred_expression_type(
                    self._replace_identifiers(method.return_type, bindings)
                )
                if (
                    substituted is not None
                    and not self._type_references_template_parameter(
                        substituted, method.template_parameters
                    )
                ):
                    return substituted

        # 3. Argument-INDEPENDENT result: when EVERY operator() overload returns
        #    the same concrete (non-template-dependent) type, that type is the
        #    result regardless of which overload applies (e.g. Real/Imag always
        #    return float). Bails for an identity-like functor whose operator()
        #    return type is one of its own template parameters.
        fixed_returns: Set[str] = set()
        for method in [*concrete_ops, *template_ops]:
            result = self._normalize_inferred_expression_type(method.return_type)
            if result is None or self._type_references_template_parameter(
                result, method.template_parameters
            ):
                return None
            fixed_returns.add(result)
        if len(fixed_returns) == 1:
            return next(iter(fixed_returns))
        return None

    def _operator_call_parameter_types(self, method: "_MetalStructMethod") -> List[str]:
        # The normalized declared parameter types of an operator() overload, in
        # the same shape produced for inferred call-argument types so the two can
        # be compared directly.
        return [
            normalized
            for normalized in (
                self._normalize_inferred_type(
                    self._normalize_function_parameter_type_text(parameter)
                )
                for parameter in self._split_top_level_commas(method.parameters)
            )
            if normalized and normalized != "void"
        ]

    def _type_references_template_parameter(
        self, type_text: str, template_parameters: List[str]
    ) -> bool:
        # True when `type_text` still mentions one of `template_parameters` (i.e.
        # the type stayed template-dependent after binding substitution).
        if not template_parameters:
            return False
        parameters = set(template_parameters)
        return any(
            match.group(0) in parameters for match in IDENTIFIER_RE.finditer(type_text)
        )

    def _infer_binary_arithmetic_type(
        self,
        expr: str,
        buffer_element_types: _MetalBufferTypeView,
        local_variable_types: Dict[str, str],
        struct_field_types: Optional[Dict[str, Dict[str, str]]],
        structs_by_name: Optional[Dict[str, "_MetalStructDefinition"]],
    ) -> Optional[str]:
        # Split a compound expression at its lowest-precedence top-level binary
        # arithmetic operator (`+ - * / %`), infer both operands, and combine them
        # under a conservative model of the usual arithmetic conversion.
        split = self._split_top_level_binary_arithmetic(expr)
        if split is None:
            return None
        left_expr, arithmetic_operator, right_expr = split
        left_expr = left_expr.strip()
        right_expr = right_expr.strip()
        if not left_expr or not right_expr:
            return None
        left_type = self._infer_argument_type(
            left_expr,
            buffer_element_types,
            local_variable_types,
            struct_field_types,
            structs_by_name,
        )
        right_type = self._infer_argument_type(
            right_expr,
            buffer_element_types,
            local_variable_types,
            struct_field_types,
            structs_by_name,
        )
        if left_type is None or right_type is None:
            return None
        integral_constant_type = self._integral_constant_binary_result_type(
            left_type, right_type, arithmetic_operator
        )
        if integral_constant_type is not None:
            return integral_constant_type
        integral_type = self._integral_arithmetic_operand_result_type(
            left_type, right_type
        )
        if integral_type is not None:
            return integral_type
        decided, converted = builtin_arithmetic_conversion(
            self, left_type, right_type, arithmetic_operator
        )
        if decided or arithmetic_operator == "%":
            return converted
        return self._combine_binary_arithmetic_operand_types(
            left_type, right_type, left_expr, right_expr
        )

    def _integral_arithmetic_operand_result_type(
        self, left_type: str, right_type: str
    ) -> Optional[str]:
        """Apply integral promotions without guessing user-defined operators."""
        left = self._scalar_and_width(left_type)
        right = self._scalar_and_width(right_type)
        if left is None or right is None:
            return None
        left_base, left_width = left
        right_base, right_width = right
        if (
            left_base not in self._METAL_INTEGRAL_SCALAR_TYPES
            or right_base not in self._METAL_INTEGRAL_SCALAR_TYPES
        ):
            return None
        if left_width != 1 or right_width != 1:
            if left == right and left_width in {2, 3, 4} and left_base != "bool":
                return f"{left_base}{left_width}"
            return None

        promoted_left = self._promote_small_integral_scalar(left_base)
        promoted_right = self._promote_small_integral_scalar(right_base)
        if promoted_left is None or promoted_right is None:
            return None
        if promoted_left == promoted_right:
            return promoted_left
        left_size = self._METAL_SCALAR_TYPE_SIZES[promoted_left]
        right_size = self._METAL_SCALAR_TYPE_SIZES[promoted_right]
        left_signed = promoted_left in self._METAL_SIGNED_SCALAR_TYPES
        right_signed = promoted_right in self._METAL_SIGNED_SCALAR_TYPES
        size = max(left_size, right_size)
        # At equal rank the unsigned type wins; a wider signed type represents
        # every value of the narrower unsigned type supported by Metal.
        signed = (
            left_signed and right_signed
            if left_size == right_size
            else left_signed if left_size > right_size else right_signed
        )
        return {
            (4, True): "int",
            (4, False): "uint",
            (8, True): "long",
            (8, False): "ulong",
        }.get((size, signed))

    def _integral_constant_binary_result_type(
        self, left_type: str, right_type: str, arithmetic_operator: str
    ) -> Optional[str]:
        if arithmetic_operator not in self._integral_constant_binary_operators:
            return None
        left = self._integral_constant_type_parts(left_type)
        right = self._integral_constant_type_parts(right_type)
        if left is None or right is None:
            return None
        left_base, left_value = left
        right_base, right_value = right
        # The MLX-style Int alias and the underlying operator overload both use
        # int operands. Wider and mixed signedness cases require the complete C++
        # usual-arithmetic-conversion contract and remain deliberately unresolved.
        if left_base != "int" or right_base != "int":
            return None
        folded, result = self._evaluate_static_integral_expression(
            f"({left_value}) {arithmetic_operator} ({right_value})"
        )
        if not folded or result is None:
            return None
        return self._normalize_template_argument_text(
            f"integral_constant<int, {result}>"
        )

    def _integral_constant_type_parts(
        self, type_text: str
    ) -> Optional[Tuple[str, int]]:
        normalized = self._normalize_template_argument_text(type_text)
        match = re.fullmatch(
            r"Int\s*<(?P<value>.+)>",
            normalized,
        )
        if match is not None and self._int_alias_contract_verified:
            folded, value = self._evaluate_static_integral_expression(
                match.group("value")
            )
            return ("int", value) if folded and value is not None else None

        match = re.fullmatch(
            r"integral_constant\s*<(?P<arguments>.+)>",
            normalized,
        )
        if match is None or not self._integral_constant_contract_verified:
            return None
        arguments = [
            argument.strip()
            for argument in self._split_top_level_commas(match.group("arguments"))
        ]
        if len(arguments) != 2:
            return None
        base_type = self._normalize_inferred_type(arguments[0])
        if self._is_integral_concrete_type(base_type) is not True:
            return None
        folded, value = self._evaluate_static_integral_expression(arguments[1])
        if not folded or value is None:
            return None
        return base_type, value

    def _combine_binary_arithmetic_operand_types(
        self, left_type: str, right_type: str, left_expr: str, right_expr: str
    ) -> Optional[str]:
        # Conservative usual-arithmetic-conversion result of two operand types:
        #   * identical types -> that type;
        #   * a concrete operand combined with a numeric literal:
        #       - an AGGREGATE concrete (struct/complex/etc.) absorbs the literal
        #         (its arithmetic operators are defined to return the aggregate);
        #       - a SCALAR/VECTOR concrete keeps its type UNLESS a floating-point
        #         literal meets an integer scalar (C++ then widens to floating and
        #         the target width is unknown), in which case we bail;
        #   * anything else (two different concrete non-literal types, or two
        #     differing literals) -> ambiguous, so bail rather than guess.
        # Pointer arithmetic has a dedicated, address-space-aware path. Reaching
        # this scalar fallback with any pointer means that path rejected the
        # expression, so normalizing qualifiers here would reopen it as `T*`.
        if "*" in left_type or "*" in right_type:
            return None
        left = self._normalize_inferred_type(left_type)
        right = self._normalize_inferred_type(right_type)
        if not left or not right:
            return None
        if left == right:
            return left
        promoted_left = self._promote_small_integral_scalar(left)
        promoted_right = self._promote_small_integral_scalar(right)
        if (
            promoted_left is not None
            and promoted_right is not None
            and promoted_left == promoted_right
        ):
            return promoted_left
        left_literal = self._infer_literal_type(self._strip_enclosing_parens(left_expr))
        right_literal = self._infer_literal_type(
            self._strip_enclosing_parens(right_expr)
        )
        left_is_literal = left_literal is not None
        right_is_literal = right_literal is not None
        if left_is_literal == right_is_literal:
            # Both literals (differing types) or both concrete (differing types):
            # ambiguous under this conservative model.
            return None
        if left_is_literal:
            concrete, literal_type = right, self._normalize_inferred_type(left_literal)
        else:
            concrete, literal_type = left, self._normalize_inferred_type(right_literal)
        # A struct/aggregate value's operators define the result type as the
        # aggregate (e.g. `complex64_t (+|-|*|/) float` -> complex64_t).
        if concrete not in self._METAL_SCALAR_VECTOR_TYPES:
            return concrete
        # Scalar/vector concrete: only trust the concrete type when the literal
        # does not out-rank it (a floating literal meeting an integer scalar
        # promotes to a floating type of unknown width -> bail).
        if self._numeric_literal_outranks_scalar(literal_type, concrete):
            return None
        return concrete

    def _promote_small_integral_scalar(self, type_text: str) -> Optional[str]:
        decomposed = self._scalar_and_width(type_text)
        if decomposed is None:
            return None
        base_type, width = decomposed
        if width != 1 or base_type not in self._METAL_INTEGRAL_SCALAR_TYPES:
            return None
        if (
            self._METAL_SCALAR_TYPE_SIZES[base_type]
            < self._METAL_SCALAR_TYPE_SIZES["int"]
        ):
            return "int"
        return base_type

    _INTEGER_SCALAR_BASES: Set[str] = {
        "bool",
        "char",
        "uchar",
        "short",
        "ushort",
        "int",
        "uint",
        "long",
        "ulong",
    }

    def _numeric_literal_outranks_scalar(
        self, literal_type: Optional[str], scalar_type: str
    ) -> bool:
        # True when a numeric literal of `literal_type` would promote an INTEGER
        # `scalar_type` to a wider (floating) type under the usual arithmetic
        # conversions -- i.e. a floating literal (`1.0`, `2.5h`) meeting an integer
        # scalar/vector. Integer literals never out-rank (they promote to the
        # concrete's own family), so those keep the concrete type.
        if literal_type not in {"float", "half"}:
            return False
        base = re.sub(r"[0-9]+$", "", scalar_type or "")
        return base in self._INTEGER_SCALAR_BASES

    def _strip_enclosing_parens(self, expr: str) -> str:
        expr = expr.strip()
        while (
            expr.startswith("(")
            and self._find_matching_delimiter(expr, 0, "(", ")") == len(expr) - 1
        ):
            expr = expr[1:-1].strip()
        return expr

    def _split_top_level_binary_arithmetic(
        self, expr: str
    ) -> Optional[Tuple[str, str, str]]:
        # Locate the operator at which to split `expr` for arithmetic inference:
        # the LOWEST-precedence top-level `+ - * / %` (additive below
        # multiplicative), rightmost among equal precedence so the split mirrors
        # C++'s left-associative grouping. Operators inside (), [], {} or string
        # literals are skipped, as are unary signs, `->`, `++`/`--`, compound
        # assignments and floating-point exponent signs. Returns (left, op, right)
        # or None when no top-level binary arithmetic operator is present.
        precedence = {"+": 0, "-": 0, "*": 1, "/": 1, "%": 1}
        depth = 0
        best_index: Optional[int] = None
        best_precedence: Optional[int] = None
        i = 0
        n = len(expr)
        while i < n:
            ch = expr[i]
            if ch in "\"'":
                _literal, consumed = self._read_string(expr, i)
                i += consumed
                continue
            if ch in "([{":
                depth += 1
                i += 1
                continue
            if ch in ")]}":
                depth = max(0, depth - 1)
                i += 1
                continue
            if (
                depth == 0
                and ch in precedence
                and self._is_binary_operator_position(expr, i)
            ):
                prec = precedence[ch]
                if best_precedence is None or prec <= best_precedence:
                    best_precedence = prec
                    best_index = i
            i += 1
        if best_index is None:
            return None
        return expr[:best_index], expr[best_index], expr[best_index + 1 :]

    def _is_binary_operator_position(self, expr: str, index: int) -> bool:
        # True when the `+ - * / %` at `index` is a BINARY operator rather than a
        # unary sign, a `->` arrow, an increment/decrement, a compound assignment,
        # or a floating-point exponent sign.
        ch = expr[index]
        nxt = expr[index + 1] if index + 1 < len(expr) else ""
        if ch == "-" and nxt == ">":
            return False
        if nxt == ch and ch in "+-":
            return False
        if nxt == "=":
            return False
        j = index - 1
        while j >= 0 and expr[j].isspace():
            j -= 1
        if j < 0:
            return False
        prev = expr[j]
        if not (prev.isalnum() or prev == "_" or prev in ")]}."):
            return False
        # `1e+5` / `2.0E-3`: a +/- right after an exponent marker that follows a
        # digit or dot belongs to the number, not a binary operator.
        if ch in "+-" and prev in "eE":
            k = j - 1
            if k >= 0 and (expr[k].isdigit() or expr[k] == "."):
                return False
        return True

"""Compile-time scalar expression evaluation and the visibility of namespace-scope template calls."""

import math
import re
import struct as binary_struct
from bisect import bisect_right
from typing import (
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

from .support import (
    IDENTIFIER_RE,
    _MetalConstexprAggregateValue,
    _MetalConstexprBinaryOperator,
    _MetalConstexprScalarValue,
    _MetalNamespaceVisibility,
    _MetalStructDefinition,
    _MetalTemplateFunction,
    _MetalUsingNamespaceDirective,
)


class MetalStaticScalarMixin:
    """Members of ``MetalPreprocessor``: compile-time scalar expression evaluation and the visibility of namespace-scope template calls."""

    def _static_constexpr_binary_operators(
        self,
        code: str,
    ) -> List[_MetalConstexprBinaryOperator]:
        template_matches = list(re.finditer(r"\btemplate\s*<", code))
        template_starts = [match.start() for match in template_matches]
        operators: List[_MetalConstexprBinaryOperator] = []
        for operator_match in re.finditer(
            r"\boperator\s*(?P<operator>[+\-*/])\s*\(",
            code,
        ):
            template_index = (
                bisect_right(
                    template_starts,
                    operator_match.start(),
                )
                - 1
            )
            if template_index < 0:
                continue
            template_match = template_matches[template_index]
            angle_start = code.find("<", template_match.start(), template_match.end())
            angle_end = self._find_matching_template_param_angle(code, angle_start)
            if angle_end is None or angle_end >= operator_match.start():
                continue
            declaration_prefix = code[angle_end + 1 : operator_match.start()]
            if any(character in declaration_prefix for character in ";{}"):
                continue
            if (
                re.search(
                    r"\b(?:constexpr|consteval)\b",
                    declaration_prefix,
                )
                is None
            ):
                continue

            open_paren = operator_match.end() - 1
            close_paren = self._find_matching_delimiter(
                code,
                open_paren,
                "(",
                ")",
            )
            if close_paren is None:
                continue
            body_start = close_paren + 1
            while body_start < len(code) and code[body_start].isspace():
                body_start += 1
            if body_start >= len(code) or code[body_start] != "{":
                continue
            body_end = self._find_matching_brace(code, body_start)
            if body_end is None:
                continue
            body = self._strip_template_argument_comments(
                code[body_start + 1 : body_end - 1]
            ).strip()
            return_match = re.fullmatch(
                r"return\s+(?P<expression>.+?)\s*;",
                body,
                re.DOTALL,
            )
            if return_match is None:
                continue

            parameter_text = code[open_paren + 1 : close_paren]
            parameter_names = self._parameter_identifier_names(parameter_text)
            parameter_types = self._parameter_declared_types(parameter_text)
            if len(parameter_names) != 2 or len(parameter_types) != 2:
                continue
            template_parameters = self._template_parameter_names(
                code[angle_start + 1 : angle_end]
            )
            if not template_parameters:
                continue
            return_type = re.sub(
                r"\b(?:constexpr|consteval|inline|static|METAL_FUNC)\b",
                " ",
                declaration_prefix,
            )
            return_type = self._normalize_template_argument_text(return_type)
            if not return_type:
                continue
            operators.append(
                _MetalConstexprBinaryOperator(
                    operator=operator_match.group("operator"),
                    span=(template_match.start(), body_end),
                    template_parameters=tuple(template_parameters),
                    return_type=return_type,
                    parameter_names=(parameter_names[0], parameter_names[1]),
                    parameter_types=(parameter_types[0], parameter_types[1]),
                    return_expression=return_match.group("expression").strip(),
                )
            )
        return operators

    def _canonical_static_aggregate_type(self, type_name: str) -> str:
        normalized = self._normalize_inferred_type(type_name)
        specialization = self._materialized_struct_specializations.get(normalized)
        if specialization is None:
            return normalized
        source_name, source_arguments = specialization
        return self._normalize_template_argument_text(
            f"{source_name}<{', '.join(source_arguments)}>"
        )

    def _substitute_static_type_bindings(
        self,
        type_text: str,
        bindings: Mapping[str, str],
    ) -> str:
        return self._normalize_template_argument_text(
            self._replace_identifiers(type_text, dict(bindings))
        )

    def _evaluate_static_scalar_expression(
        self,
        expression: str,
        *,
        expected_type: Optional[str] = None,
        scalar_bindings: Optional[Mapping[str, _MetalConstexprScalarValue]] = None,
        aggregate_bindings: Optional[
            Mapping[str, _MetalConstexprAggregateValue]
        ] = None,
    ) -> Optional[_MetalConstexprScalarValue]:
        text = self._strip_static_outer_parentheses(
            self._strip_template_argument_comments(expression)
        )
        if not text or len(text) > 512:
            return None
        scalar_bindings = scalar_bindings or {}
        aggregate_bindings = aggregate_bindings or {}

        def finish(
            value: Optional[_MetalConstexprScalarValue],
        ) -> Optional[_MetalConstexprScalarValue]:
            if value is None or expected_type is None:
                return value
            return self._cast_static_constexpr_scalar(value, expected_type)

        if text.startswith("static_cast"):
            angle_start = text.find("<", len("static_cast"))
            if angle_start != -1:
                angle_end = self._find_matching_angle(text, angle_start)
                cursor = angle_end + 1 if angle_end is not None else -1
                while cursor >= 0 and cursor < len(text) and text[cursor].isspace():
                    cursor += 1
                if cursor >= 0 and cursor < len(text) and text[cursor] == "(":
                    close_paren = self._find_matching_delimiter(
                        text,
                        cursor,
                        "(",
                        ")",
                    )
                    if close_paren == len(text) - 1:
                        cast_type = self._static_constexpr_scalar_type(
                            text[angle_start + 1 : angle_end]
                        )
                        value = self._evaluate_static_scalar_expression(
                            text[cursor + 1 : close_paren],
                            scalar_bindings=scalar_bindings,
                            aggregate_bindings=aggregate_bindings,
                        )
                        if cast_type is not None and value is not None:
                            return finish(
                                self._cast_static_constexpr_scalar(value, cast_type)
                            )
            return None

        cast_match = re.match(
            r"(?P<type>(?:::)?[A-Za-z_][A-Za-z0-9_]*"
            r"(?:::[A-Za-z_][A-Za-z0-9_]*)*)\s*\(",
            text,
        )
        if cast_match is not None:
            open_paren = cast_match.end() - 1
            close_paren = self._find_matching_delimiter(
                text,
                open_paren,
                "(",
                ")",
            )
            if close_paren == len(text) - 1:
                cast_type = self._static_constexpr_scalar_type(cast_match.group("type"))
                value = self._evaluate_static_scalar_expression(
                    text[open_paren + 1 : close_paren],
                    scalar_bindings=scalar_bindings,
                    aggregate_bindings=aggregate_bindings,
                )
                if cast_type is not None and value is not None:
                    return finish(self._cast_static_constexpr_scalar(value, cast_type))
                return None

        operation = self._split_top_level_binary_arithmetic(text)
        if operation is not None:
            left_text, binary_operator, right_text = operation
            left = self._evaluate_static_scalar_expression(
                left_text,
                scalar_bindings=scalar_bindings,
                aggregate_bindings=aggregate_bindings,
            )
            right = self._evaluate_static_scalar_expression(
                right_text,
                scalar_bindings=scalar_bindings,
                aggregate_bindings=aggregate_bindings,
            )
            return finish(
                self._evaluate_static_scalar_binary_operation(
                    binary_operator,
                    left,
                    right,
                )
            )

        member_match = re.fullmatch(
            r"([A-Za-z_][A-Za-z0-9_]*)\s*\.\s*" r"([A-Za-z_][A-Za-z0-9_]*)",
            text,
        )
        if member_match is not None:
            aggregate = aggregate_bindings.get(member_match.group(1))
            return finish(
                None if aggregate is None else aggregate.member(member_match.group(2))
            )
        if IDENTIFIER_RE.fullmatch(text):
            if text == "true":
                return finish(_MetalConstexprScalarValue("bool", True))
            if text == "false":
                return finish(_MetalConstexprScalarValue("bool", False))
            return finish(scalar_bindings.get(text))
        if text[0] in "+-":
            operand = self._evaluate_static_scalar_expression(
                text[1:],
                scalar_bindings=scalar_bindings,
                aggregate_bindings=aggregate_bindings,
            )
            if operand is None or isinstance(operand.value, bool):
                return None
            value = operand.value if text[0] == "+" else -operand.value
            return finish(
                self._cast_static_constexpr_scalar(
                    _MetalConstexprScalarValue(operand.type_name, value),
                    operand.type_name,
                )
            )
        return finish(self._parse_static_constexpr_scalar_literal(text))

    def _evaluate_static_scalar_binary_operation(
        self,
        binary_operator: str,
        left: Optional[_MetalConstexprScalarValue],
        right: Optional[_MetalConstexprScalarValue],
    ) -> Optional[_MetalConstexprScalarValue]:
        if left is None or right is None:
            return None
        result_type = self._static_constexpr_common_scalar_type(
            left.type_name,
            right.type_name,
        )
        if result_type is None:
            return None
        converted_left = self._cast_static_constexpr_scalar(left, result_type)
        converted_right = self._cast_static_constexpr_scalar(right, result_type)
        if converted_left is None or converted_right is None:
            return None
        left_value = converted_left.value
        right_value = converted_right.value
        if isinstance(left_value, bool) or isinstance(right_value, bool):
            return None
        try:
            if binary_operator == "+":
                value = left_value + right_value
            elif binary_operator == "-":
                value = left_value - right_value
            elif binary_operator == "*":
                value = left_value * right_value
            elif binary_operator == "/":
                if right_value == 0:
                    return None
                if result_type in self._METAL_INTEGRAL_SCALAR_TYPES:
                    quotient = abs(int(left_value)) // abs(int(right_value))
                    value = (
                        -quotient if (left_value < 0) != (right_value < 0) else quotient
                    )
                else:
                    value = left_value / right_value
            else:
                return None
        except (ArithmeticError, OverflowError, ValueError):
            return None
        return self._cast_static_constexpr_scalar(
            _MetalConstexprScalarValue(result_type, value),
            result_type,
        )

    def _parse_static_constexpr_scalar_literal(
        self,
        text: str,
    ) -> Optional[_MetalConstexprScalarValue]:
        match = re.fullmatch(
            r"(?P<number>(?:\d+\.\d*|\.\d+|\d+)(?:[eE][+-]?\d+)?)"
            r"(?P<suffix>[fFhH]|[uUlL]*)",
            text,
        )
        if match is None:
            return None
        number = match.group("number")
        suffix = match.group("suffix")
        is_floating = (
            "." in number
            or "e" in number.lower()
            or suffix.lower()
            in {
                "f",
                "h",
            }
        )
        try:
            if is_floating:
                if suffix and suffix.lower() not in {"f", "h"}:
                    return None
                type_name = {"f": "float", "h": "half"}.get(
                    suffix.lower(),
                    "double",
                )
                value: Union[bool, int, float] = float(number)
            else:
                if suffix and re.fullmatch(r"[uUlL]+", suffix) is None:
                    return None
                type_name = "uint" if "u" in suffix.lower() else "int"
                value = int(number, 10)
        except (OverflowError, ValueError):
            return None
        return self._cast_static_constexpr_scalar(
            _MetalConstexprScalarValue(type_name, value),
            type_name,
        )

    def _static_constexpr_scalar_type(self, type_text: str) -> Optional[str]:
        if any(token in type_text for token in ("*", "&", "[", "]")):
            return None
        normalized = self._normalize_template_argument_text(type_text)
        normalized = re.sub(
            r"\b(?:const|constant|constexpr|consteval|device|inline|static|"
            r"thread|threadgroup|volatile)\b",
            " ",
            normalized,
        )
        normalized = re.sub(r"\s+", " ", normalized).strip()
        normalized = re.sub(r"^(?:::)?metal::", "", normalized)
        return self._METAL_SCALAR_BY_SPELLING.get(normalized)

    def _static_constexpr_common_scalar_type(
        self,
        left_type: str,
        right_type: str,
    ) -> Optional[str]:
        left = self._static_constexpr_scalar_type(left_type)
        right = self._static_constexpr_scalar_type(right_type)
        if left is None or right is None:
            return None
        if left == right:
            return left
        floating_rank = {
            "half": 1,
            "float16_t": 1,
            "bfloat16_t": 1,
            "float": 2,
            "double": 3,
        }
        left_floating = floating_rank.get(left)
        right_floating = floating_rank.get(right)
        if left_floating is not None and right_floating is not None:
            if left_floating == right_floating == 1 and {left, right} not in (
                {"half", "float16_t"},
            ):
                return None
            return left if left_floating >= right_floating else right
        if left_floating is not None and right in self._METAL_INTEGRAL_SCALAR_TYPES:
            return left
        if right_floating is not None and left in self._METAL_INTEGRAL_SCALAR_TYPES:
            return right
        if (
            left in self._METAL_INTEGRAL_SCALAR_TYPES
            and right in self._METAL_INTEGRAL_SCALAR_TYPES
        ):
            left_size = self._METAL_SCALAR_TYPE_SIZES[left]
            right_size = self._METAL_SCALAR_TYPE_SIZES[right]
            left_signed = left in self._METAL_SIGNED_SCALAR_TYPES
            right_signed = right in self._METAL_SIGNED_SCALAR_TYPES
            if left_size == right_size and left_signed != right_signed:
                return None
            return left if left_size >= right_size else right
        return None

    def _cast_static_constexpr_scalar(
        self,
        value: _MetalConstexprScalarValue,
        target_type: str,
    ) -> Optional[_MetalConstexprScalarValue]:
        target = self._static_constexpr_scalar_type(target_type)
        if target is None:
            return None
        raw_value = value.value
        if target == "bool":
            return _MetalConstexprScalarValue(target, bool(raw_value))
        if target in self._METAL_INTEGRAL_SCALAR_TYPES:
            if isinstance(raw_value, float) and not math.isfinite(raw_value):
                return None
            try:
                integer_value = int(raw_value)
            except (OverflowError, TypeError, ValueError):
                return None
            bits = self._METAL_SCALAR_TYPE_SIZES[target] * 8
            if target in self._METAL_SIGNED_SCALAR_TYPES:
                minimum = -(1 << (bits - 1))
                maximum = (1 << (bits - 1)) - 1
            else:
                minimum = 0
                maximum = (1 << bits) - 1
            if not minimum <= integer_value <= maximum:
                return None
            return _MetalConstexprScalarValue(target, integer_value)

        try:
            floating_value = float(raw_value)
        except (OverflowError, TypeError, ValueError):
            return None
        if not math.isfinite(floating_value):
            return None
        try:
            if target in {"half", "float16_t"}:
                floating_value = binary_struct.unpack(
                    ">e",
                    binary_struct.pack(">e", floating_value),
                )[0]
            elif target == "bfloat16_t":
                bits = int.from_bytes(
                    binary_struct.pack(">f", floating_value),
                    "big",
                )
                rounded = bits + 0x7FFF + ((bits >> 16) & 1)
                floating_value = binary_struct.unpack(
                    ">f",
                    ((rounded >> 16) << 16).to_bytes(4, "big"),
                )[0]
            elif target == "float":
                floating_value = binary_struct.unpack(
                    ">f",
                    binary_struct.pack(">f", floating_value),
                )[0]
            elif target != "double":
                return None
        except (OverflowError, ValueError, binary_struct.error):
            return None
        if not math.isfinite(floating_value):
            return None
        return _MetalConstexprScalarValue(target, floating_value)

    def _strip_static_outer_parentheses(self, expression: str) -> str:
        text = expression.strip()
        while text.startswith("("):
            close_paren = self._find_matching_delimiter(text, 0, "(", ")")
            if close_paren != len(text) - 1:
                break
            text = text[1:close_paren].strip()
        return text

    def _split_static_top_level_operator(
        self,
        expression: str,
        operators: Sequence[str],
    ) -> Optional[Tuple[str, str, str]]:
        ordered_operators = sorted(operators, key=len, reverse=True)
        paren_depth = 0
        bracket_depth = 0
        brace_depth = 0
        angle_depth = 0
        found: Optional[Tuple[int, str]] = None
        index = 0
        while index < len(expression):
            if expression[index] in "\"'":
                _literal, consumed = self._read_string(expression, index)
                index += consumed
                continue
            if expression.startswith("//", index):
                end = expression.find("\n", index)
                index = len(expression) if end == -1 else end + 1
                continue
            if expression.startswith("/*", index):
                end = expression.find("*/", index + 2)
                index = len(expression) if end == -1 else end + 2
                continue
            character = expression[index]
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
            elif character == ">" and angle_depth:
                angle_depth -= 1
            elif paren_depth == bracket_depth == brace_depth == angle_depth == 0:
                matched = next(
                    (
                        operator_text
                        for operator_text in ordered_operators
                        if expression.startswith(operator_text, index)
                    ),
                    None,
                )
                if matched is not None:
                    if found is not None:
                        return None
                    found = (index, matched)
                    index += len(matched)
                    continue
            index += 1
        if found is None:
            return None
        operator_index, operator_text = found
        left = expression[:operator_index].strip()
        right = expression[operator_index + len(operator_text) :].strip()
        if not left or not right:
            return None
        return left, operator_text, right

    def _metal_namespace_visibility(self, code: str) -> _MetalNamespaceVisibility:
        analysis = self._source_analysis(code)
        if analysis.namespace_visibility is None:
            analysis.namespace_visibility = self._scan_metal_namespace_visibility(code)
        return analysis.namespace_visibility

    def _scan_metal_namespace_visibility(self, code: str) -> _MetalNamespaceVisibility:
        namespace_spans = tuple(self._find_namespace_spans(code))
        lexical_scopes = self._find_lexical_brace_scopes(code)
        ignored_spans = self._find_comment_and_literal_spans(code)
        directives: List[_MetalUsingNamespaceDirective] = []
        pattern = re.compile(
            r"(?<![A-Za-z0-9_])using\s+namespace\s+"
            r"(?P<name>(?:::)?\s*[A-Za-z_]\w*"
            r"(?:\s*::\s*[A-Za-z_]\w*)*)\s*;"
        )
        for match in pattern.finditer(code):
            if self._containing_span(match.start(), ignored_spans) is not None:
                continue
            raw_target = re.sub(r"\s*::\s*", "::", match.group("name").strip())
            globally_qualified = raw_target.startswith("::")
            target = raw_target.lstrip(":")
            if not target:
                continue
            scope_start, scope_end = self._innermost_lexical_scope(
                lexical_scopes,
                match.start(),
                len(code),
            )
            directives.append(
                _MetalUsingNamespaceDirective(
                    target=target,
                    globally_qualified=globally_qualified,
                    declaration_namespace=self._namespace_at(
                        list(namespace_spans),
                        match.start(),
                    ),
                    declaration_position=match.start(),
                    scope_start=scope_start,
                    scope_end=scope_end,
                )
            )
        return _MetalNamespaceVisibility(
            source_length=len(code),
            namespace_spans=namespace_spans,
            using_namespace_directives=tuple(directives),
        )

    @staticmethod
    def _metal_enclosing_namespaces(namespace: str) -> Tuple[str, ...]:
        components = [part for part in str(namespace or "").split("::") if part]
        return tuple(
            "::".join(components[:length]) for length in range(len(components), -1, -1)
        )

    @classmethod
    def _metal_namespace_reference_candidates(
        cls,
        reference: str,
        context_namespace: str,
        *,
        globally_qualified: bool,
    ) -> Tuple[str, ...]:
        normalized = re.sub(r"\s*::\s*", "::", str(reference or "").strip())
        normalized = normalized.lstrip(":")
        if globally_qualified:
            return (normalized,)
        candidates: List[str] = []
        for prefix in cls._metal_enclosing_namespaces(context_namespace):
            candidate = (
                f"{prefix}::{normalized}"
                if prefix and normalized
                else (prefix or normalized)
            )
            if candidate not in candidates:
                candidates.append(candidate)
        return tuple(candidates)

    @staticmethod
    def _metal_call_reference_namespace(qualified_name: str) -> Optional[str]:
        normalized = re.sub(
            r"\s*::\s*",
            "::",
            str(qualified_name or "").strip(),
        )
        if "::" not in normalized:
            return None
        namespace = normalized.rsplit("::", 1)[0]
        if normalized.startswith("::"):
            return f"::{namespace.lstrip(':')}"
        return namespace

    def _metal_namespace_declaration_visible(
        self,
        declaration_namespace: str,
        reference_namespace: Optional[str],
        position: int,
        visibility: Optional[_MetalNamespaceVisibility],
    ) -> bool:
        declaration = re.sub(
            r"\s*::\s*",
            "::",
            str(declaration_namespace or "").strip(),
        ).lstrip(":")
        if visibility is None:
            if reference_namespace is None:
                return declaration == ""
            requested = str(reference_namespace)
            return declaration == requested.lstrip(":")

        call_namespace = self._namespace_at(
            list(visibility.namespace_spans),
            position,
        )
        if reference_namespace is not None:
            requested = str(reference_namespace)
            return declaration in self._metal_namespace_reference_candidates(
                requested,
                call_namespace,
                globally_qualified=requested.startswith("::"),
            )

        if declaration in self._metal_enclosing_namespaces(call_namespace):
            return True
        for directive in visibility.using_namespace_directives:
            if not (
                directive.declaration_position <= position
                and directive.scope_start <= position < directive.scope_end
            ):
                continue
            if declaration in self._metal_namespace_reference_candidates(
                directive.target,
                directive.declaration_namespace,
                globally_qualified=directive.globally_qualified,
            ):
                return True
        return False

    def _metal_template_function_call_visible(
        self,
        template: _MetalTemplateFunction,
        qualified_name: str,
        position: int,
        visibility: _MetalNamespaceVisibility,
        owner: Optional[_MetalStructDefinition],
    ) -> bool:
        reference_scope = self._metal_call_reference_namespace(qualified_name)
        if owner is None:
            return self._metal_namespace_declaration_visible(
                template.namespace,
                reference_scope,
                position,
                visibility,
            )

        # A qualifier on a member-template call names a type scope, not the
        # declaration namespace.  Treating `Tile::make<T>()` as a request for a
        # namespace called `Tile` incorrectly hides the template before member
        # lowering can canonicalize owner-scoped aliases in its signature.  Admit
        # only the exact concrete owner visible from the call's lexical namespace;
        # an unrelated namespace, type alias, or same-named owner remains closed.
        if reference_scope is not None:
            call_namespace = self._namespace_at(
                list(visibility.namespace_spans),
                position,
            )
            owner_candidates = self._metal_namespace_reference_candidates(
                reference_scope,
                call_namespace,
                globally_qualified=reference_scope.startswith("::"),
            )
            return (owner.qualified_name or owner.name).lstrip(":") in owner_candidates

        # An unqualified member template is visible only inside its declaring
        # concrete owner.  Namespace visibility alone is insufficient because a
        # same-named free call outside the class must not bind to a member.
        return owner.body_span[0] <= position < owner.body_span[1]

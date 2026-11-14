"""Return-type SFINAE enablers of Metal function templates.

A function template can constrain itself through its return type::

    template <typename T>
    ::metal::enable_if_t<::metal::is_floating_point_v<T>, T> max(T a, T b);

``enable_if_t<B, T>`` names ``T`` when ``B`` is true and names no type otherwise, so deduction fails and the
template leaves the overload set (C++14 [meta.trans.other] Table 57, [temp.deduct]/8). The Metal standard library
provides ``metal::enable_if`` and ``metal::enable_if_t`` with the same meaning (MSL 4.0 section 6.1, "Type Traits").

* :func:`return_type_enabler_condition` gives the condition of a template's return-type enabler. It joins the
  template's constraints, so overload selection evaluates it as it evaluates an ``enable_if_t`` template parameter.
* :func:`fold_concrete_enablers` rewrites the enablers that concrete code keeps, outside every template
  declaration: once type traits are folded, a materialized ``enable_if_t<true, float>`` is ``float`` and
  ``enable_if_t<true>`` is ``void``. A condition that is false, or that does not fold to a constant, is refused:
  the selected instantiation would have no return type, and no target can spell the enabler itself.
"""

import re
from typing import Any, List, Optional, Tuple

# ``[typename] [::][metal::]enable_if_t<`` or ``[typename] [::][metal::]enable_if<``, not part of a longer
# qualified name.
_ENABLER_START_RE = re.compile(
    r"(?<![\w:])(?:typename\s+)?(?:::\s*)?(?:metal\s*::\s*)?enable_if(?P<alias>_t)?\s*<"
)
_TYPE_MEMBER_RE = re.compile(r"\s*::\s*type\b")
# Declarations of a source-defined ``enable_if`` or ``enable_if_t``; its meaning is then the source's own.
_USER_ENABLER_RE = re.compile(r"\b(?:struct|class|using)\s+enable_if(?:_t)?\b(?!\s*::)")
_CONDITION_TOKEN_RE = re.compile(
    r"\s*(?:(?P<bool>true|false)\b|(?P<int>\d+)[uUlL]*\b|(?P<op>&&|\|\||==|!=|<=|>=|<|>|!|\(|\)))"
)


class _NotConstant(Exception):
    """The condition is not a constant expression this module evaluates."""


def return_type_enabler_condition(
    preprocessor: Any, header: str, function_name: str
) -> Optional[str]:
    """The condition of the ``enable_if_t`` that is the whole return type in ``header``, or None.

    ``header`` is the declaration text between the template parameter list and the body. Only a return type that
    is exactly one enabler counts, as in ``_split_return_type_sfinae`` for member templates.
    """
    parameter_start = preprocessor._function_parameter_start(header)
    if parameter_start is None:
        return None
    name_match = re.search(
        rf"\b{re.escape(function_name)}\s*$", header[:parameter_start]
    )
    if name_match is None:
        return None
    return_type = re.sub(r"\[\[[^\]]*\]\]", " ", header[: name_match.start()])
    return_type = re.sub(
        r"\b(?:constexpr|inline|static|METAL_FUNC)\b", " ", return_type
    )
    return_type = preprocessor._strip_function_qualifier_macros(return_type).strip()
    if not return_type:
        return None
    condition, _value_type = preprocessor._split_return_type_sfinae(return_type)
    return condition


def fold_concrete_enablers(preprocessor: Any, code: str) -> str:
    """``code`` with every enabler outside a template declaration replaced by the type it names."""
    if "enable_if" not in code or _USER_ENABLER_RE.search(
        preprocessor._mask_comments_and_literals(code)
    ):
        return code
    ignored = sorted(
        [
            *preprocessor._find_template_declaration_spans(code),
            *preprocessor._find_comment_and_literal_spans(code),
        ]
    )
    replacements: List[Tuple[int, int, str]] = []
    for match in _ENABLER_START_RE.finditer(code):
        if preprocessor._containing_span(match.start(), ignored) is not None:
            continue
        angle_start = match.end() - 1
        angle_end = preprocessor._find_matching_template_param_angle(code, angle_start)
        if angle_end is None:
            continue
        end = angle_end + 1
        if match.group("alias") is None:
            member = _TYPE_MEMBER_RE.match(code, end)
            if member is None:
                # ``enable_if<B, T>`` itself is a structure type, not the type it names.
                continue
            end = member.end()
        arguments = preprocessor._split_template_parameter_list(
            code[angle_start + 1 : angle_end]
        )
        expression = re.sub(r"\s+", " ", code[match.start() : end]).strip()
        if len(arguments) not in (1, 2) or not arguments[0].strip():
            _refuse(preprocessor, code, match.start(), end, expression, "malformed")
        try:
            enabled = _evaluate_condition(arguments[0])
        except _NotConstant:
            _refuse(preprocessor, code, match.start(), end, expression, "unresolved")
        if not enabled:
            _refuse(preprocessor, code, match.start(), end, expression, "false")
        value_type = arguments[1].strip() if len(arguments) == 2 else "void"
        replacements.append((match.start(), end, value_type))
    return preprocessor._apply_text_replacements(code, replacements)


def _refuse(
    preprocessor: Any, code: str, start: int, end: int, expression: str, reason: str
) -> None:
    from .preprocessor import MetalTypeTraitError

    details = {
        "malformed": "it does not have one condition and at most one type",
        "unresolved": "its condition does not fold to a constant",
        "false": "its condition is false, so the selected declaration has no type",
    }
    suggested_action = (
        "make the condition depend only on concrete types and integral constants, "
        "or give the call a concrete overload"
    )
    raise MetalTypeTraitError(
        f"Metal enabler '{expression}' in concrete code cannot be resolved: {details[reason]}. "
        f"Suggested action: {suggested_action}.",
        trait="enable_if",
        expression=expression,
        operands=(),
        resolved_operands=(),
        reason=f"enabler-{reason}",
        owner=preprocessor._concrete_type_trait_owner(code, start),
        source_location=preprocessor._source_location_for_offsets(code, start, end),
        suggested_action=suggested_action,
    )


def _evaluate_condition(text: str) -> bool:
    """The value of a condition made of ``true``, ``false``, integer literals, ``!``, ``&&``, ``||``,
    comparisons and parentheses; raises :class:`_NotConstant` for anything else."""
    tokens: List[Tuple[str, str]] = []
    position = 0
    stripped = text.rstrip()
    while position < len(stripped):
        match = _CONDITION_TOKEN_RE.match(stripped, position)
        if match is None:
            raise _NotConstant(text)
        kind = match.lastgroup
        tokens.append((kind, match.group(kind)))
        position = match.end()
    parser = _ConditionParser(tokens)
    value = parser.disjunction()
    if parser.index != len(tokens):
        raise _NotConstant(text)
    if isinstance(value, bool):
        return value
    raise _NotConstant(text)


class _ConditionParser:
    """Recursive descent over C++ precedence: ``||`` < ``&&`` < equality < relational < unary."""

    def __init__(self, tokens: List[Tuple[str, str]]) -> None:
        self.tokens = tokens
        self.index = 0

    def peek(self) -> Optional[str]:
        return self.tokens[self.index][1] if self.index < len(self.tokens) else None

    def take(self) -> Tuple[str, str]:
        if self.index >= len(self.tokens):
            raise _NotConstant("truncated")
        token = self.tokens[self.index]
        self.index += 1
        return token

    def disjunction(self):
        value = self.conjunction()
        while self.peek() == "||":
            self.take()
            right = self.conjunction()
            value = _as_bool(value) or _as_bool(right)
        return value

    def conjunction(self):
        value = self.equality()
        while self.peek() == "&&":
            self.take()
            right = self.equality()
            value = _as_bool(value) and _as_bool(right)
        return value

    def equality(self):
        value = self.relational()
        while self.peek() in ("==", "!="):
            operator = self.take()[1]
            right = self.relational()
            if isinstance(value, bool) != isinstance(right, bool):
                raise _NotConstant("mixed equality")
            value = (value == right) if operator == "==" else (value != right)
        return value

    def relational(self):
        value = self.unary()
        while self.peek() in ("<", "<=", ">", ">="):
            operator = self.take()[1]
            right = self.unary()
            if isinstance(value, bool) or isinstance(right, bool):
                raise _NotConstant("relational on bool")
            value = {
                "<": value < right,
                "<=": value <= right,
                ">": value > right,
                ">=": value >= right,
            }[operator]
        return value

    def unary(self):
        kind, text = self.take()
        if text == "!":
            return not _as_bool(self.unary())
        if text == "(":
            value = self.disjunction()
            if self.take()[1] != ")":
                raise _NotConstant("unbalanced")
            return value
        if kind == "bool":
            return text == "true"
        if kind == "int":
            return int(text)
        raise _NotConstant(text)


def _as_bool(value) -> bool:
    if not isinstance(value, bool):
        raise _NotConstant("integer used as a condition")
    return value

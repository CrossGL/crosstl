"""Fold arithmetic between integral-constant values in materialized Metal bodies.

A Metal library may define ``integral_constant<T, v>`` together with free operator templates such as::

    template <typename T, T tv, typename U, U uv>
    constexpr auto operator*(integral_constant<T, tv>, integral_constant<U, uv>) {
      constexpr auto res = tv * uv;
      return integral_constant<decltype(res), res>{};
    }

When both operands of ``a * b`` are integral constants, overload resolution selects that template, because it is an
exact match while the built-in candidate needs a user-defined conversion on each operand ([over.match.oper],
[over.ics.rank]). The expression therefore has the type ``integral_constant<int, a * b>`` and no run-time value. The
preprocessor verifies the struct and operator shapes once per source (``_configure_integral_constant_contracts``).
This module rewrites such expressions in the bodies of materialized functions, after template parameters were
substituted, into a value-initialized construction of the result type, so later passes materialize that type and the
code generators never see an operator applied to two aggregates.

Operands are function parameters whose declared type is an integral constant and that the body does not redeclare,
integral-constant constructions such as ``Int<4>{}``, and the results of earlier folds. A candidate ``A op B`` is
folded only when it is a complete subexpression under C++ precedence and associativity, so ``x + a * b`` folds
``a * b`` and ``a * b * x`` folds ``a * b``, while ``x * a * b`` and ``-a * b`` stay unchanged. Every operand is
free of side effects, so replacing the operator call by its constant result preserves the program's behaviour.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .preprocessor import MetalPreprocessor

_PRECEDENCE = {
    "*": 13,
    "/": 13,
    "%": 13,
    "+": 12,
    "-": 12,
    "<<": 11,
    ">>": 11,
    "<": 10,
    "<=": 10,
    ">": 10,
    ">=": 10,
    "==": 9,
    "!=": 9,
    "&": 8,
    "^": 7,
    "|": 6,
    "&&": 5,
    "||": 4,
}
# Operator spellings, longest first, so ``<<=`` is never read as ``<<``.
_OPERATOR_TOKENS = (
    "<<=",
    ">>=",
    "&&",
    "||",
    "<<",
    ">>",
    "<=",
    ">=",
    "==",
    "!=",
    "+=",
    "-=",
    "*=",
    "/=",
    "%=",
    "&=",
    "|=",
    "^=",
    "++",
    "--",
    "->",
    "::",
    "+",
    "-",
    "*",
    "/",
    "%",
    "<",
    ">",
    "&",
    "^",
    "|",
    "!",
    "~",
    "=",
    "?",
    ":",
)
_BOUNDARY_CHARACTERS = "([{,;}?:"
# Keywords that precede an expression, so an operator after them is unary.
_OPERAND_KEYWORDS = {"return", "case", "else", "do", "throw", "co_return", "co_yield"}
_CONSTRUCTION = re.compile(
    r"(?P<type>(?:[A-Za-z_]\w*\s*::\s*)*(?:Int|integral_constant)\s*<[^<>;{}]*>)"
    r"\s*\{\s*\}"
)
_QUALIFIER = re.compile(r"^(?P<qualifier>(?:[A-Za-z_]\w*\s*::\s*)*)")
_MAX_FOLDS = 256


def fold_integral_constant_parameter_arithmetic(
    preprocessor: MetalPreprocessor, parameters: str, body: str
) -> str:
    """Return ``body`` with integral-constant arithmetic between its operands folded."""
    operators = preprocessor._integral_constant_binary_operators
    if not operators or not preprocessor._integral_constant_contract_verified:
        return body
    parameter_types = _integral_constant_parameter_types(preprocessor, parameters, body)
    if not parameter_types and _CONSTRUCTION.search(body) is None:
        return body
    for _ in range(_MAX_FOLDS):
        folded = _fold_once(preprocessor, body, parameter_types, operators)
        if folded is None:
            return body
        body = folded
    return body


def fold_integral_constant_function_arithmetic(
    preprocessor: MetalPreprocessor, function_text: str
) -> str:
    """Fold integral-constant arithmetic inside one function definition."""
    if not preprocessor._integral_constant_binary_operators:
        return function_text
    open_paren = preprocessor._function_parameter_start(function_text)
    if open_paren is None:
        return function_text
    close_paren = preprocessor._find_matching_delimiter(
        function_text, open_paren, "(", ")"
    )
    if close_paren is None:
        return function_text
    body_open = preprocessor._find_next_top_level_char(
        function_text, close_paren + 1, "{"
    )
    if body_open is None or body_open < 0:
        return function_text
    body_close = preprocessor._find_matching_delimiter(
        function_text, body_open, "{", "}"
    )
    if body_close is None:
        return function_text
    body = function_text[body_open + 1 : body_close]
    folded = fold_integral_constant_parameter_arithmetic(
        preprocessor, function_text[open_paren + 1 : close_paren], body
    )
    if folded == body:
        return function_text
    parameters = function_text[open_paren + 1 : close_paren]
    preserved = preprocessor._preserve_parameter_usage(parameters, body, folded)
    return (
        function_text[: open_paren + 1]
        + preserved
        + function_text[close_paren : body_open + 1]
        + folded
        + function_text[body_close:]
    )


def _integral_constant_parameter_types(
    preprocessor: MetalPreprocessor, parameters: str, body: str
) -> dict[str, str]:
    shadowed = preprocessor._local_variable_names(body)
    types: dict[str, str] = {}
    for parameter in preprocessor._split_top_level_commas(parameters):
        name = preprocessor._declared_data_member_name(parameter)
        if not name or name in shadowed:
            continue
        if preprocessor._lambda_binds_identifier(body, name):
            continue
        parameter_type = preprocessor._function_parameter_value_type(parameter)
        if _integral_constant_parts(preprocessor, parameter_type) is None:
            continue
        types[name] = parameter_type
    return types


def _integral_constant_parts(
    preprocessor: MetalPreprocessor, type_text: str
) -> tuple[str, int] | None:
    unqualified = _QUALIFIER.sub("", type_text.strip(), count=1)
    parts = preprocessor._integral_constant_type_parts(unqualified)
    if parts is not None:
        return parts
    specialization = preprocessor._materialized_struct_specializations.get(
        preprocessor._normalize_inferred_type(type_text)
    )
    if specialization is None:
        return None
    source_name, source_arguments = specialization
    return preprocessor._integral_constant_type_parts(
        f"{source_name.rsplit('::', 1)[-1]}<{', '.join(source_arguments)}>"
    )


def _fold_once(
    preprocessor: MetalPreprocessor,
    body: str,
    parameter_types: dict[str, str],
    operators,
) -> str | None:
    masked = preprocessor._mask_comments_and_literals(body)
    ungrouped = _strip_grouping_parentheses(preprocessor, body, masked)
    if ungrouped is not None:
        return ungrouped
    atoms = _atoms(preprocessor, masked, parameter_types)
    for left, right in zip(atoms, atoms[1:]):
        operator_start = _skip_space(masked, left[1])
        operator = _operator_at(masked, operator_start)
        if operator not in operators or operator not in _PRECEDENCE:
            continue
        if _skip_space(masked, operator_start + len(operator)) != right[0]:
            continue
        if not _left_context_allows(masked, left[0], operator):
            continue
        if not _right_context_allows(masked, right[1], operator):
            continue
        result_type = preprocessor._integral_constant_binary_result_type(
            _QUALIFIER.sub("", _canonical_type(preprocessor, left[2]), count=1),
            _QUALIFIER.sub("", _canonical_type(preprocessor, right[2]), count=1),
            operator,
        )
        if result_type is None:
            continue
        replacement = _result_spelling(preprocessor, left[2], result_type)
        if replacement is None:
            continue
        return body[: left[0]] + replacement + "{}" + body[right[1] :]
    return None


def _strip_grouping_parentheses(
    preprocessor: MetalPreprocessor, body: str, masked: str
) -> str | None:
    """Drop ``( C )`` around one integral-constant construction ``C``.

    Parentheses around a primary expression do not change its value or type ([expr.prim.paren]). Parentheses that
    follow an identifier, ``)`` or ``]`` belong to a call or a cast and are kept.
    """
    for match in re.finditer(r"\(\s*", masked):
        construction = _CONSTRUCTION.match(masked, match.end())
        if construction is None:
            continue
        if _integral_constant_parts(preprocessor, construction.group("type")) is None:
            continue
        close = _skip_space(masked, construction.end())
        if close >= len(masked) or masked[close] != ")":
            continue
        previous = match.start() - 1
        while previous >= 0 and masked[previous].isspace():
            previous -= 1
        if previous >= 0 and masked[previous] in ")]>":
            continue
        if previous >= 0 and (masked[previous].isalnum() or masked[previous] == "_"):
            word = re.search(r"\w+$", masked[: previous + 1]).group(0)
            if word not in _OPERAND_KEYWORDS:
                continue
        return (
            body[: match.start()]
            + body[construction.start() : construction.end()]
            + body[close + 1 :]
        )
    return None


def _atoms(
    preprocessor: MetalPreprocessor, masked: str, parameter_types: dict[str, str]
) -> list[tuple[int, int, str]]:
    atoms: list[tuple[int, int, str]] = []
    for match in _CONSTRUCTION.finditer(masked):
        if _is_member_or_scope_operand(masked, match.start()):
            continue
        if _integral_constant_parts(preprocessor, match.group("type")) is None:
            continue
        atoms.append((match.start(), match.end(), match.group("type")))
    if parameter_types:
        names = "|".join(re.escape(name) for name in sorted(parameter_types))
        for match in re.finditer(rf"(?<![\w.])(?:{names})\b", masked):
            if _is_member_or_scope_operand(masked, match.start()):
                continue
            atoms.append((match.start(), match.end(), parameter_types[match.group(0)]))
    atoms.sort()
    return atoms


def _canonical_type(preprocessor: MetalPreprocessor, type_text: str) -> str:
    specialization = preprocessor._materialized_struct_specializations.get(
        preprocessor._normalize_inferred_type(type_text)
    )
    if specialization is None:
        return type_text
    source_name, source_arguments = specialization
    return f"{source_name.rsplit('::', 1)[-1]}<{', '.join(source_arguments)}>"


def _result_spelling(
    preprocessor: MetalPreprocessor, operand_type: str, result_type: str
) -> str | None:
    """Spell the folded type the way the left operand's type is spelled."""
    if preprocessor._normalize_inferred_type(
        operand_type
    ) in preprocessor._materialized_struct_specializations and not re.search(
        r"<", operand_type
    ):
        source_name, arguments = _split_template_id(preprocessor, result_type)
        for name, (
            source,
            source_arguments,
        ) in preprocessor._materialized_struct_specializations.items():
            if (
                source.rsplit("::", 1)[-1] == source_name
                and [
                    preprocessor._normalize_template_argument_text(argument)
                    for argument in source_arguments
                ]
                == arguments
            ):
                return name
        return None
    qualifier = _QUALIFIER.match(operand_type.strip()).group("qualifier")
    unqualified = operand_type.strip()[len(qualifier) :]
    _base, value = preprocessor._integral_constant_type_parts(result_type)
    literal = preprocessor._static_integral_literal_text(value)
    if re.match(r"Int\s*<", unqualified):
        return f"{qualifier}Int<{literal}>"
    return f"{qualifier}integral_constant<int, {literal}>"


def _split_template_id(
    preprocessor: MetalPreprocessor, type_text: str
) -> tuple[str, list[str]]:
    name, _, arguments = type_text.partition("<")
    return name.strip(), [
        preprocessor._normalize_template_argument_text(argument)
        for argument in preprocessor._split_top_level_commas(arguments[:-1])
    ]


def _skip_space(text: str, index: int) -> int:
    while index < len(text) and text[index].isspace():
        index += 1
    return index


def _operator_at(text: str, index: int) -> str | None:
    for token in _OPERATOR_TOKENS:
        if text.startswith(token, index):
            return token
    return None


def _operator_ending_at(text: str, end: int) -> tuple[str, int] | None:
    for token in _OPERATOR_TOKENS:
        start = end - len(token)
        if start >= 0 and text[start:end] == token:
            return token, start
    return None


def _ends_operand(text: str, index: int) -> bool:
    previous = index - 1
    while previous >= 0 and text[previous].isspace():
        previous -= 1
    if previous < 0:
        return False
    if text[previous].isalnum() or text[previous] == "_":
        word = re.search(r"\w+$", text[: previous + 1]).group(0)
        return word not in _OPERAND_KEYWORDS
    return text[previous] in ")]}"


def _is_member_or_scope_operand(text: str, start: int) -> bool:
    previous = start - 1
    while previous >= 0 and text[previous].isspace():
        previous -= 1
    return text[max(0, previous - 1) : previous + 1] in ("->", "::") or (
        previous >= 0 and text[previous] == "."
    )


def _left_context_allows(text: str, start: int, operator: str) -> bool:
    previous = start - 1
    while previous >= 0 and text[previous].isspace():
        previous -= 1
    if previous < 0 or text[previous] in _BOUNDARY_CHARACTERS:
        return previous < 0 or text[previous] != ":" or text[previous - 1] != ":"
    if text[previous].isalnum() or text[previous] == "_":
        word = re.search(r"[A-Za-z_]\w*$", text[: previous + 1])
        return word is not None and word.group(0) == "return"
    found = _operator_ending_at(text, previous + 1)
    if found is None:
        return False
    token, token_start = found
    if token not in _PRECEDENCE:
        # Assignment, conditional and comma bind more loosely than any folded
        # operator; increments, member access and unary operators bind tighter.
        return token in ("=", "+=", "-=", "*=", "/=", "%=", "&=", "|=", "^=")
    if not _ends_operand(text, token_start):
        return False
    return _PRECEDENCE[token] < _PRECEDENCE[operator]


def _right_context_allows(text: str, end: int, operator: str) -> bool:
    following = _skip_space(text, end)
    if following >= len(text) or text[following] in ")],;}?:":
        return following >= len(text) or not text.startswith("::", following)
    token = _operator_at(text, following)
    if token is None or token not in _PRECEDENCE:
        return False
    return _PRECEDENCE[token] <= _PRECEDENCE[operator]

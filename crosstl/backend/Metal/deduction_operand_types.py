"""Operand shapes that type a call argument for template argument deduction in project materialization.

Project materialization deduces a function template's arguments from the types of the call's arguments
([temp.deduct.call]). In ``template <typename T> T outer(T x) { return erf(-x) + erf(x * M_SQRT1_2_F); }``
materialized for ``float``, deducing ``erf``'s ``T`` needs the types of three shapes:

* a unary sign, ``-x``: the promoted operand type ([expr.unary.op]/7);
* a predefined Metal constant, ``M_SQRT1_2_F``: the type the MSL standard library gives it (MSL section 6.6);
* a one-argument floating-point math function, ``exp(-x)``: the argument's type, since each such function is
  overloaded for every floating-point scalar and vector type with the same result type (MSL section 6.5).

:func:`operand_expression_type` types these shapes, or returns ``None`` so the caller keeps failing closed.
"""

from __future__ import annotations

import re
from typing import Callable, Optional

from .narrow_float_constants import metal_predefined_constant_type
from .operator_argument_types import infer_unary_sign_type

__all__ = ["operand_expression_type", "is_unary_sign_position"]

Infer = Callable[[str], Optional[str]]

# One-argument math functions whose result has their floating-point argument's type (MSL section 6.5).
_SAME_TYPE_FLOATING_FUNCTIONS = frozenset(
    {
        "acos",
        "acosh",
        "asin",
        "asinh",
        "atan",
        "atanh",
        "ceil",
        "cos",
        "cosh",
        "cospi",
        "exp",
        "exp10",
        "exp2",
        "fabs",
        "floor",
        "fract",
        "log",
        "log10",
        "log2",
        "rint",
        "round",
        "rsqrt",
        "saturate",
        "sin",
        "sinh",
        "sinpi",
        "sqrt",
        "tan",
        "tanh",
        "tanpi",
        "trunc",
    }
)
_FLOATING_SCALARS = frozenset({"float", "half"})
_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_CALL = re.compile(r"(?:(?:metal|precise|fast)\s*::\s*)*(?P<name>[A-Za-z_]\w*)\s*\(")
# A ``+`` or ``-`` after one of these, or at the start, is a sign rather than a binary operator.
_SIGN_PREDECESSORS = frozenset("+-*/%&|^!~=(,?:[{<>")


def is_unary_sign_position(expression: str, index: int) -> bool:
    """Whether the ``+`` or ``-`` at ``index`` of ``expression`` is a sign, an update or an arrow."""
    following = expression[index + 1 : index + 2]
    if following in {">", expression[index]}:
        return True
    previous = expression[:index].rstrip()
    if not previous:
        return True
    if previous[-1] in _SIGN_PREDECESSORS:
        # ``x++ - y`` and ``x-- + y``: an update ends an operand.
        return not previous.endswith(("++", "--")) or previous[-3:-2] in {"+", "-"}
    return False


def _floating_function_type(preprocessor, text: str, infer: Infer) -> str | None:
    call = _CALL.match(text)
    if call is None or call.group("name") not in _SAME_TYPE_FLOATING_FUNCTIONS:
        return None
    open_index = call.end() - 1
    if (
        preprocessor._find_matching_delimiter(text, open_index, "(", ")")
        != len(text) - 1
    ):
        return None
    arguments = preprocessor._split_top_level_commas(text[open_index + 1 : -1])
    if len(arguments) != 1 or not arguments[0].strip():
        return None
    argument_type = infer(arguments[0].strip())
    if argument_type is None or "*" in argument_type:
        return None
    normalized = preprocessor._normalize_inferred_type(argument_type)
    decomposed = preprocessor._scalar_and_width(normalized or "")
    if decomposed is None or decomposed[0] not in _FLOATING_SCALARS:
        return None
    return normalized


def operand_expression_type(preprocessor, text: str, infer: Infer) -> str | None:
    """The type of a unary sign, a predefined constant or a floating math call, or ``None``."""
    text = text.strip()
    if not text:
        return None
    if _IDENTIFIER.fullmatch(text):
        return metal_predefined_constant_type(text)
    signed = infer_unary_sign_type(preprocessor, text, infer)
    if signed is not None:
        return signed
    return _floating_function_type(preprocessor, text, infer)

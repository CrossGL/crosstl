"""Types of conditional and unary-sign call arguments in Metal sources.

Template argument deduction and overload selection in the Metal importer type each call argument with
``MetalPreprocessor._infer_argument_type``, which recognizes a fixed set of expression shapes and returns ``None``
for the rest, so the call is refused. Two common shapes had no rule:

* A conditional expression ``c ? a : b``. When both arms have one type the result has that type; when both are
  arithmetic, the usual arithmetic conversions give the common type (C++17 [expr.cond]/7). So
  ``m ? input[i] * scale : 0`` is a ``float`` and ``u < p ? 1u : 0u`` a ``uint``.
* A unary sign ``-x`` or ``+x`` on a value that is not a literal. The result has the operand's type after integral
  promotion (C++17 [expr.unary.op]/7-8), so ``-pad`` is an ``int`` for an ``int``, ``short`` or ``bool`` operand.
  Metal vector operands keep their type.

Both rules type their operands with the same inference, and fail closed: an operand without an inferred type, a
pointer, an assignment, mixed scalar and vector arms, arms of two different non-arithmetic types, and two floating
types of equal size (``half`` and ``bfloat``) all yield ``None``. A struct operand of a unary sign yields ``None``,
since a user-defined ``operator-`` chooses its own result type.
"""

from __future__ import annotations

from typing import Callable, Optional

__all__ = ["infer_conditional_type", "infer_unary_sign_type"]

Infer = Callable[[str], Optional[str]]

# Floating types by conversion rank; two different types of equal rank have no common type.
_FLOATING_RANKS = {"half": 1, "float16_t": 1, "bfloat16_t": 1, "float": 2, "double": 3}


def infer_conditional_type(preprocessor, expr: str, infer: Infer) -> str | None:
    """The type of ``expr`` when it is a conditional expression whose arms both have known types."""
    question = preprocessor._top_level_conditional_question(expr)
    if question is None:
        return None
    colon = preprocessor._matching_conditional_colon(expr, question)
    if colon is None:
        return None
    parts = (expr[:question], expr[question + 1 : colon], expr[colon + 1 :])
    if any(not part.strip() or _has_top_level_assignment(part) for part in parts):
        return None
    true_type = infer(parts[1].strip())
    false_type = infer(parts[2].strip())
    if true_type is None or false_type is None:
        return None
    return _common_arm_type(preprocessor, true_type, false_type)


def infer_unary_sign_type(preprocessor, expr: str, infer: Infer) -> str | None:
    """The type of ``-x`` or ``+x`` for an operand of arithmetic scalar or vector type."""
    if not expr or expr[0] not in "+-" or expr[:2] in {"++", "--"}:
        return None
    operand = expr[1:].strip()
    if not operand or operand[0] == "=":
        return None
    operand_type = infer(operand)
    if operand_type is None or "*" in operand_type:
        return None
    normalized = preprocessor._normalize_inferred_type(operand_type)
    decomposed = preprocessor._scalar_and_width(normalized)
    if decomposed is None:
        return None
    base, width = decomposed
    if width != 1:
        return None if base == "bool" else normalized
    if base in preprocessor._METAL_INTEGRAL_SCALAR_TYPES:
        return preprocessor._promote_small_integral_scalar(normalized)
    if base in _FLOATING_RANKS:
        return normalized
    return None


def _common_arm_type(preprocessor, true_type: str, false_type: str) -> str | None:
    if "*" in true_type or "*" in false_type:
        return None
    left = preprocessor._normalize_inferred_type(true_type)
    right = preprocessor._normalize_inferred_type(false_type)
    if not left or not right:
        return None
    if left == right:
        return left
    left_parts = preprocessor._scalar_and_width(left)
    right_parts = preprocessor._scalar_and_width(right)
    if left_parts is None or right_parts is None:
        return None
    if left_parts[1] != 1 or right_parts[1] != 1:
        return None
    left_base, right_base = left_parts[0], right_parts[0]
    integral = preprocessor._METAL_INTEGRAL_SCALAR_TYPES
    if left_base in integral and right_base in integral:
        return preprocessor._integral_arithmetic_operand_result_type(left, right)
    left_rank = _FLOATING_RANKS.get(left_base)
    right_rank = _FLOATING_RANKS.get(right_base)
    if left_rank is not None and right_base in integral:
        return left
    if right_rank is not None and left_base in integral:
        return right
    if left_rank is None or right_rank is None or left_rank == right_rank:
        return None
    return left if left_rank > right_rank else right


def _has_top_level_assignment(text: str) -> bool:
    """Whether ``text`` holds ``=`` or a compound assignment outside brackets."""
    depth = 0
    for index, char in enumerate(text):
        if char in "([{":
            depth += 1
        elif char in ")]}":
            depth = max(depth - 1, 0)
        elif char == "=" and depth == 0:
            before = text[index - 1] if index else ""
            after = text[index + 1] if index + 1 < len(text) else ""
            if after == "=" or before == "=":
                continue
            if before == "!":
                continue
            if (
                before
                and before in "<>"
                and text[index - 2 : index] not in {"<<", ">>"}
            ):
                continue
            return True
    return False

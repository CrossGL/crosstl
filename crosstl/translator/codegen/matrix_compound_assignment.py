"""Explicit products for CrossGL ``*=`` with a matrix right operand.

In CrossGL, as in GLSL 4.60 section 5.9, ``m *= n`` is the linear-algebra product ``m = m * n`` and ``v *= m`` is the
row-vector product ``v = v * m``. HLSL and Slang define ``*=`` on two matrices component-wise and reject a matrix on
the right of a vector's ``*=``; MSL has no ``*=`` that takes two matrices. Generators for those targets rewrite the
assignment as ``target = target * value`` and let their binary-product lowering choose ``mul(...)`` or ``*``.

The rewrite reads the target a second time, so it is only applied to targets whose evaluation has no side effects.
Any other target is refused rather than evaluated twice.
"""

from __future__ import annotations

from ..ast import AssignmentNode, BinaryOpNode
from ..diagnostics import CODE_PREFIX, TranslationDiagnostic
from ..semantics.rewrite import is_pure


class MatrixCompoundAssignmentError(TranslationDiagnostic):
    """A matrix ``*=`` whose target cannot be read twice without repeating a side effect."""

    def __init__(self, target, *, source_location=None):
        self.source_location = source_location
        super().__init__(
            "a '*=' with a matrix right operand is the product 'x = x * m' (GLSL 4.60 section 5.9); "
            "the target has side effects, so it cannot be read again for the explicit product",
            target=target,
            code=f"{CODE_PREFIX}matrix-compound-assignment-unsupported",
        )


def _repeatable(target):
    if isinstance(target, str):
        return target.isidentifier()
    return is_pure(target)


def matrix_product_assignment(
    target, value, op, *, type_of, is_matrix, is_vector, backend, target_type_of=None
):
    """Return ``target = target * value`` for a matrix ``*=``, or ``None`` when the assignment is not one.

    ``type_of`` (and ``target_type_of`` for the target, when it differs) infer the generator's types, and
    ``is_matrix`` and ``is_vector`` classify them. Types are only inferred for ``*=``, and the target's only once the
    right operand is a matrix. A scalar target is left alone: no language defines it, and the generator reports it
    in its usual way.
    """
    if op != "*=" or not is_matrix(type_of(value)):
        return None
    target_type = (target_type_of or type_of)(target)
    if not (is_matrix(target_type) or is_vector(target_type)):
        return None
    if not _repeatable(target):
        raise MatrixCompoundAssignmentError(
            backend, source_location=getattr(target, "source_location", None)
        )
    product = BinaryOpNode(target, "*", value)
    location = getattr(value, "source_location", None)
    if location is not None:
        product.source_location = location
    return AssignmentNode(target, product, "=")

"""Explicit remainders for a floating-point CrossGL ``%=``.

CrossGL ``%`` on floating-point operands is the truncated remainder ``x - y * trunc(x / y)``, whose sign is the
dividend's: HLSL ``%`` and ``fmod``, MSL ``fmod``, C ``fmod``, WGSL ``%`` and SPIR-V ``OpFRem`` all compute it.
GLSL has no floating ``%`` and MSL and C++ have no floating ``%=``, so generators for those targets rewrite
``x %= y`` as ``x = x % y`` and let their binary remainder lowering choose ``fmod`` or a helper.

The rewrite reads the target a second time, so it is only applied to targets whose evaluation has no side effects.
Any other target is refused rather than evaluated twice.
"""

from __future__ import annotations

from ..ast import AssignmentNode, BinaryOpNode
from ..diagnostics import CODE_PREFIX, TranslationDiagnostic
from ..semantics.rewrite import is_pure

__all__ = ["FloatRemainderAssignmentError", "float_remainder_assignment"]


class FloatRemainderAssignmentError(TranslationDiagnostic):
    """A floating ``%=`` whose target cannot be read twice without repeating a side effect."""

    def __init__(self, target, *, source_location=None):
        self.source_location = source_location
        super().__init__(
            "a floating-point '%=' is the remainder 'x = x % y'; the target has side effects, so it cannot be "
            "read again for the explicit remainder",
            target=target,
            code=f"{CODE_PREFIX}float-remainder-assignment-unsupported",
        )


def _repeatable(target):
    if isinstance(target, str):
        return target.isidentifier()
    return is_pure(target)


def float_remainder_assignment(target, value, op, *, type_of, is_floating, backend):
    """Return ``target = target % value`` for a floating ``%=``, or ``None`` when the assignment is not one.

    ``type_of`` infers the generator's type of an operand and ``is_floating`` says whether a type is a
    floating-point scalar or vector. The assignment is a floating remainder when either operand is.
    """
    if op != "%=":
        return None
    if not (is_floating(type_of(value)) or is_floating(type_of(target))):
        return None
    if not _repeatable(target):
        raise FloatRemainderAssignmentError(
            backend, source_location=getattr(target, "source_location", None)
        )
    remainder = BinaryOpNode(target, "%", value)
    location = getattr(value, "source_location", None)
    if location is not None:
        remainder.source_location = location
    return AssignmentNode(target, remainder, "=")

"""Scalar operands of Metal vector operations.

MSL converts the scalar operand of a binary operation with a vector to the vector's component type: ``half3 * f``
with a ``float`` ``f`` is a ``half3`` computed in ``half``, and ``int3 / u`` with a ``uint`` ``u`` is a signed
``int3`` division (MSL section 2.12, "Implicit type conversions"; confirmed with ``is_same`` in the Metal compiler).
CrossGL follows GLSL and HLSL instead, where both operands take their common type, so the same expression imported
from Metal computed ``float3(h) * f`` and an unsigned ``uint3`` division, and storing the ``float3`` back into a
``half3`` failed to compile on Metal (``assigning to 'half3' from incompatible type 'float3'``).

:func:`convert_metal_vector_scalar_operands` makes the MSL rule explicit for a program imported from Metal: in a
binary operation or compound assignment of a vector and a scalar whose type ranks above the vector's component
type, the scalar is converted to the component type, ``h * half(f)`` and ``v / int(u)``. A scalar that ranks below
the component type gives the same result under both rules and is left alone. MSL rejects a floating-point scalar
with an integer vector, so a source never holds one.
"""

from __future__ import annotations

from .. import ast as A
from .hub_types import HubType
from .rewrite import call
from .scoped_walk import ScopedWalker

__all__ = ["convert_metal_vector_scalar_operands"]

# Operators that convert their operands to a common type.
_CONVERTING_OPERATORS = frozenset(
    {"+", "-", "*", "/", "%", "<", ">", "<=", ">=", "==", "!=", "&", "|", "^"}
)
_COMPOUND_OPERATORS = frozenset({"+=", "-=", "*=", "/=", "%=", "&=", "|=", "^="})
_KIND_RANKS = {"i": 0, "u": 1, "f": 2}
_SCALAR_NAMES = {
    ("f", 16): "half",
    ("f", 32): "float",
    ("f", 64): "double",
    ("i", 8): "int8_t",
    ("i", 16): "int16_t",
    ("i", 32): "int",
    ("i", 64): "int64_t",
    ("u", 8): "uint8_t",
    ("u", 16): "uint16_t",
    ("u", 32): "uint",
    ("u", 64): "uint64_t",
}


def _rank(value: HubType):
    """C++ conversion rank of a scalar: floating point above integers, then width, then unsigned above signed."""
    if value.kind == "f":
        return (2, value.bits, 0)
    return (1, value.bits, _KIND_RANKS[value.kind])


def _conversion(vector: HubType | None, scalar: HubType | None):
    """The component spelling ``scalar`` is converted to beside ``vector``, or ``None``."""
    if vector is None or scalar is None or vector.lanes == 1 or scalar.lanes != 1:
        return None
    component = vector.scalar
    if component == scalar or "b" in (component.kind, scalar.kind):
        return None
    if component.kind != "f" and scalar.kind == "f":
        return None
    if _rank(scalar) <= _rank(component):
        return None
    return _SCALAR_NAMES.get((component.kind, component.bits))


class _VectorScalarOperands(ScopedWalker):
    def __init__(self, program):
        super().__init__(program)
        self.converted = 0

    def _convert(self, node, vector_slot, scalar_slot):
        name = _conversion(
            self.scope.infer(getattr(node, vector_slot)),
            self.scope.infer(getattr(node, scalar_slot)),
        )
        if name is None:
            return False
        setattr(node, scalar_slot, call(name, [getattr(node, scalar_slot)]))
        self.converted += 1
        return True

    def visit(self, node):
        if isinstance(node, A.BinaryOpNode) and node.op in _CONVERTING_OPERATORS:
            if not self._convert(node, "left", "right"):
                self._convert(node, "right", "left")
        elif (
            isinstance(node, A.AssignmentNode) and node.operator in _COMPOUND_OPERATORS
        ):
            if self._convert(node, "target", "value"):
                node.left, node.right = node.target, node.value


def convert_metal_vector_scalar_operands(program, source) -> int:
    """Convert the scalar operands of a Metal program's vector operations in place; returns how many."""
    if not isinstance(program, A.ShaderNode) or str(source or "").lower() != "metal":
        return 0
    walker = _VectorScalarOperands(program)
    walker.run_program()
    return walker.converted

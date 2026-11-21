"""Shared pieces of the passes that compute wave operations from ``WaveReadLaneAt`` reads of single lanes.

:mod:`.partitioned_waves` and :mod:`.subgroup_rotations` write CrossGL helper functions, one per operation and operand
type, whose reads go through ``WaveReadLaneAt``. Every subgroup target implements that operation as an arbitrary-lane
shuffle, but Direct3D and Metal take numeric values only and CUDA takes scalars only, so a ``bool`` travels as a
``uint`` and a vector is read one component at a time.
"""

from __future__ import annotations

from .. import ast as A
from .hub_types import HubType

__all__ = [
    "LANE_READ_TYPES",
    "lane_read",
    "parse_helper_function",
    "type_spelling",
    "wave_operand_type",
]

COMPONENTS = "xyzw"
_SCALAR_NAMES = {
    ("f", 16): "half",
    ("f", 32): "float",
    ("f", 64): "double",
    ("i", 32): "int",
    ("u", 32): "uint",
    ("b", 1): "bool",
}
_VECTOR_PREFIXES = {
    ("f", 16): "f16vec",
    ("f", 32): "vec",
    ("f", 64): "dvec",
    ("i", 32): "ivec",
    ("u", 32): "uvec",
}
# The (kind, bits) pairs whose scalars helpers can spell; vectors exist for every pair but bool.
LANE_READ_TYPES = frozenset(_SCALAR_NAMES)
# Wave operations whose result has the type of their first argument, for operands that are themselves wave calls.
_SAME_TYPE_PREFIXES = (
    "WaveActive",
    "WavePrefix",
    "WaveRead",
    "QuadRead",
    "WaveShuffle",
    "WaveClustered",
    "WaveRotate",
    "WaveMulti",
)
_OTHER_TYPE_OPERATIONS = frozenset(
    {
        "WaveActiveAllTrue",
        "WaveActiveAnyTrue",
        "WaveActiveAllEqual",
        "WaveActiveBallot",
        "WaveActiveCountBits",
        "WavePrefixCountBits",
    }
)


def type_spelling(value: HubType) -> str | None:
    """The CrossGL spelling of ``value``'s type, or ``None`` when a helper cannot take it."""
    key = (value.kind, value.bits)
    if value.lanes == 1:
        return _SCALAR_NAMES.get(key)
    prefix = _VECTOR_PREFIXES.get(key)
    return f"{prefix}{value.lanes}" if prefix else None


def lane_read(value: HubType, spelling: str, operand="value", lane="lane") -> str:
    """CrossGL source for ``operand`` as lane ``lane`` holds it."""
    if value.kind == "b":
        return f"WaveReadLaneAt({operand} ? 1u : 0u, {lane}) != 0u"
    if value.lanes == 1:
        return f"WaveReadLaneAt({operand}, {lane})"
    reads = ", ".join(
        f"WaveReadLaneAt({operand}.{c}, {lane})" for c in COMPONENTS[: value.lanes]
    )
    return f"{spelling}({reads})"


def wave_operand_type(scope, expr) -> HubType | None:
    """The type of ``expr``, looking through wave operations whose result has their first argument's type."""
    while (
        isinstance(expr, A.WaveOpNode)
        and expr.arguments
        and expr.operation not in _OTHER_TYPE_OPERATIONS
        and expr.operation.startswith(_SAME_TYPE_PREFIXES)
    ):
        expr = expr.arguments[0]
    return scope.infer(expr)


def parse_helper_function(source):
    """Parse the CrossGL source of one function."""
    from ..lexer import Lexer
    from ..parser import Parser

    shader = Parser(Lexer(f"shader crossgl_lane_reads {{\n{source}}}\n").tokens).parse()
    (function,) = shader.functions
    return function

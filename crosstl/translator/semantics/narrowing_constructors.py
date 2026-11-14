"""Narrowing constructors of vector values, for targets whose casts reject them.

CrossGL constructors follow GLSL. A scalar constructor given a vector uses the
vector's first component (GLSL 4.60 section 5.4.1, "Conversion and Scalar
Constructors"), and a vector constructor given a single wider vector uses its
leading components (section 5.4.2, "Vector and Matrix Constructors":
``vec3(vec4)`` drops the fourth component).

HLSL and Metal reject both spellings. DXC reports "too many elements in vector
initialization" for ``float(v)`` or ``float2(v)`` with ``v`` a ``float4``, and
Metal has no conversion from a vector to a scalar or to a narrower vector. The
storage-image texel a CrossGL ``imageLoad`` returns is four-wide, so sources
such as ``float(imageLoad(img, p))`` reach this case.

:func:`select_narrowing_constructor_components` rewrites the argument of such a
constructor to its leading components, ``float(v.x)`` or ``float2(v.xy)``, in
place.
"""

from __future__ import annotations

from .. import ast as A
from .hub_types import call_name, type_from_name
from .rewrite import child_slots, get_child, set_child
from .scoped_walk import ScopedWalker

__all__ = ["select_narrowing_constructor_components"]

_LEADING_SWIZZLES = {1: "x", 2: "xy", 3: "xyz"}


def _constructor_lanes(node):
    """Lanes of the scalar or vector type a constructor call names, or ``None``."""
    if isinstance(node, A.FunctionCallNode):
        name = call_name(node)
    elif isinstance(node, A.ConstructorNode):
        name = getattr(node.constructor_type, "name", None)
    else:
        return None
    hub = type_from_name(name)
    if hub is None or hub.lanes not in _LEADING_SWIZZLES:
        return None
    return hub.lanes


def _arguments(node):
    if isinstance(node, A.FunctionCallNode):
        return node.arguments or []
    return getattr(node, "arguments", None) or []


class _NarrowingConstructors(ScopedWalker):
    def visit(self, node):
        lanes = _constructor_lanes(node)
        if lanes is None:
            return
        arguments = _arguments(node)
        if len(arguments) != 1:
            return
        operand = arguments[0]
        operand_type = self.scope.infer(operand)
        if operand_type is None or operand_type.lanes <= max(lanes, 1):
            return
        leading = A.MemberAccessNode(
            operand,
            _LEADING_SWIZZLES[lanes],
            source_location=getattr(operand, "source_location", None),
        )
        for field, key in list(child_slots(node)):
            if get_child(node, field, key) is operand:
                set_child(node, field, key, leading)
                break


def select_narrowing_constructor_components(program):
    """Give each narrowing constructor of a vector the vector's leading components."""
    _NarrowingConstructors(program).run_program()
    return program

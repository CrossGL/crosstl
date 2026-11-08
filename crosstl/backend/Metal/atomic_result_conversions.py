"""Implicit conversions of Metal atomic results to the type that receives them.

``uint slot = atomic_fetch_add_explicit(&counter, 1, memory_order_relaxed);`` with an ``atomic_int`` counter is
valid C++: the ``int`` result converts implicitly ([conv.integral]). CrossGL's atomic intrinsics return the atomic
object's element type and require the receiving context to have that type, so the importer used to emit
``uint slot = atomicAdd(counter, 1);`` and the DirectX generator refused it. The conversion Metal applies
implicitly is made explicit instead: ``uint slot = uint(atomicAdd(counter, 1));``.
"""

from __future__ import annotations

from typing import Any

from .MetalAst import FunctionCallNode

__all__ = ["convert_atomic_result", "is_atomic_result_call"]

_SCALAR_RESULTS = frozenset({"int", "uint", "float"})
_RECEIVING_SCALARS = frozenset(
    {"int", "uint", "float", "half", "int64_t", "uint64_t", "bool"}
)


def _atomic_call_name(converter: Any, expr: Any):
    if not isinstance(expr, FunctionCallNode):
        return None
    name = str(getattr(expr, "name", "") or "").split("::")[-1]
    if name in getattr(converter, "user_function_names", ()):
        return None
    if name in converter.metal_atomic_intrinsics or name == "atomic_fetch_sub_explicit":
        return name
    return None


def is_atomic_result_call(converter: Any, expr: Any) -> bool:
    """Whether ``expr`` calls a Metal atomic read-modify-write function for its original value."""
    return _atomic_call_name(converter, expr) is not None


def convert_atomic_result(converter: Any, expr: Any, value: str, expected_type):
    """``value`` converted to ``expected_type`` when ``expr`` is an atomic call of another scalar type."""
    if not expected_type or _atomic_call_name(converter, expr) is None:
        return value
    if converter.is_plain_metal_auto_type(expected_type):
        return value
    result_type = converter.expression_metal_type(expr)
    if not result_type:
        return value
    result = converter.map_type(converter.resolve_type_alias(result_type))
    expected = converter.map_type(converter.resolve_type_alias(expected_type))
    if result not in _SCALAR_RESULTS or expected not in _RECEIVING_SCALARS:
        return value
    if result == expected:
        return value
    return f"{expected}({value})"

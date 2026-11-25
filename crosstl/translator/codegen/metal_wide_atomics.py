"""Metal output of CrossGL's 64-bit unsigned atomic minimum and maximum.

Metal has one 64-bit atomic type, ``atomic_ulong``, and two operations on it: ``atomic_max_explicit`` and
``atomic_min_explicit`` with ``memory_order_relaxed`` on device memory (MSL 4.1, section 6.15.4.7 "Atomic Modify
Functions (64 Bits)"). Both return nothing, where CrossGL's ``atomicMax`` and ``atomicMin`` return the original value.
A call whose result is discarded therefore becomes the Metal operation on an ``atomic_ulong`` view of the
``ulong`` element; a call whose result is used, a threadgroup target and a signed 64-bit target have no Metal form
and are refused.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "METAL_WIDE_ATOMIC_OPERATIONS",
    "METAL_WIDE_ATOMIC_TYPES",
    "discarded_wide_atomic_call",
    "is_metal_wide_atomic",
    "metal_wide_atomic_call",
]

# The buffer atomic operation of ``atomicMax`` and ``atomicMin`` -> the Metal function.
METAL_WIDE_ATOMIC_OPERATIONS = {
    "fetch_max": "atomic_max_explicit",
    "fetch_min": "atomic_min_explicit",
}
# Metal spellings of CrossGL's ``uint64_t``.
METAL_WIDE_ATOMIC_TYPES = frozenset({"ulong", "uint64_t"})


def is_metal_wide_atomic(operation: str, mapped_type: str | None) -> bool:
    """Whether a buffer atomic ``operation`` on ``mapped_type`` is Metal's 64-bit minimum or maximum."""
    return (
        operation in METAL_WIDE_ATOMIC_OPERATIONS
        and mapped_type in METAL_WIDE_ATOMIC_TYPES
    )


def discarded_wide_atomic_call(codegen: Any, expr: Any) -> str | None:
    """A discarded ``atomicMax``/``atomicMin`` call on a 64-bit target as Metal, or ``None`` for other expressions."""
    name = codegen.function_call_name(expr) if hasattr(expr, "args") else None
    if name not in {"atomicMax", "atomicMin"} or len(getattr(expr, "args", ())) != 2:
        return None
    if name in codegen.user_function_names:
        return None
    mapped_type = codegen.map_type(codegen.expression_result_type(expr.args[0]))
    if mapped_type not in METAL_WIDE_ATOMIC_TYPES:
        return None
    # A GLSL buffer-block member lowers by byte offset, not by reinterpret
    # cast, so try that path before the buffer-resource path.
    block_call = codegen.generate_glsl_buffer_block_atomic_call(
        name, expr.args, discarded=True
    )
    if block_call is not None:
        return block_call
    return codegen.generate_metal_buffer_resource_atomic_call(
        name, expr.args, wide_result_discarded=True
    )


def metal_wide_atomic_call(
    codegen: Any,
    func_name: str,
    operation: str,
    target: Any,
    target_expr: str,
    address_space: str,
    value_arg: Any,
    *,
    discarded: bool,
) -> str:
    """Metal's ``atomic_max_explicit``/``atomic_min_explicit`` for a 64-bit buffer atomic, or a refusal."""
    if not discarded:
        return codegen.unsupported_metal_buffer_resource_atomic_call(
            func_name,
            target,
            "Metal's 64-bit atomic min/max return no value, "
            "so the result cannot be used",
        )
    if address_space != "device":
        return codegen.unsupported_metal_buffer_resource_atomic_call(
            func_name,
            target,
            "on a 64-bit target requires device memory, the only memory Metal's 64-bit atomics take",
        )
    value = codegen.generate_expression_with_expected(value_arg, "ulong")
    function = METAL_WIDE_ATOMIC_OPERATIONS[operation]
    return (
        f"{function}(reinterpret_cast<device atomic_ulong*>({target_expr}), "
        f"{value}, memory_order_relaxed)"
    )

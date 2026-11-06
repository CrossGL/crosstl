"""HLSL emission of CrossGL byte-address buffer atomics.

CrossGL spells an atomic on a byte-address buffer as a call whose first argument is the buffer and whose second is
the byte offset of a 32-bit word: ``atomicAdd(buffer, offset, value)``, and likewise ``atomicMin``, ``atomicMax``,
``atomicAnd``, ``atomicOr``, ``atomicXor`` and ``atomicExchange``. Compare-exchange is
``atomicCompareExchange(buffer, offset, compare, value)``, with ``atomicCompSwap`` as a synonym. Each call returns
the word's previous value.

HLSL has these operations only as ``RWByteAddressBuffer`` methods (``InterlockedAdd`` ... ``InterlockedExchange``
and ``InterlockedCompareExchange``), which report the previous value through an optional trailing ``out`` argument,
plus ``InterlockedCompareStore`` for a compare-exchange whose previous value is not wanted. So:

- a standalone call becomes the method statement, ``buffer.InterlockedAdd(offset, value);``;
- a call whose value is used calls a small helper that passes a local as the ``out`` argument and returns it, and so
  does a standalone ``atomicExchange``, because ``InterlockedExchange`` always takes the ``out`` argument.

The word is unsigned. ``InterlockedMin`` and ``InterlockedMax`` choose the signed operation from an ``int`` value
argument, so an ``int`` value keeps that overload and the result is ``int``. A non-negative ``int`` literal is the
unsigned form, as DXC reads it. The other operations act on the bits, so an ``int`` value goes through ``asuint``.
A floating-point, bool, 64-bit or vector value is refused, as is a write to a read-only ``ByteAddressBuffer``.
"""

from __future__ import annotations

from ..ast import FunctionCallNode
from ..diagnostics import CODE_PREFIX, TranslationDiagnostic

__all__ = [
    "ByteAddressAtomicUnsupportedError",
    "emit_hlsl_byte_address_atomic_helpers",
    "hlsl_byte_address_atomic_expression",
    "hlsl_byte_address_atomic_statement",
]

# CrossGL name: HLSL RWByteAddressBuffer method.
_METHODS = {
    "atomicAdd": "InterlockedAdd",
    "atomicMin": "InterlockedMin",
    "atomicMax": "InterlockedMax",
    "atomicAnd": "InterlockedAnd",
    "atomicOr": "InterlockedOr",
    "atomicXor": "InterlockedXor",
    "atomicExchange": "InterlockedExchange",
    "atomicCompareExchange": "InterlockedCompareExchange",
    "atomicCompSwap": "InterlockedCompareExchange",
}
_COMPARE = "InterlockedCompareExchange"
_EXCHANGE = "InterlockedExchange"
_SIGNED_METHODS = frozenset({"InterlockedMin", "InterlockedMax"})
# Writable byte-address resource: helper name prefix.
_HELPER_PREFIXES = {
    "RWByteAddressBuffer": "__crossgl_bab_atomic",
    "RasterizerOrderedByteAddressBuffer": "__crossgl_rov_bab_atomic",
}
_READ_ONLY = "ByteAddressBuffer"


class ByteAddressAtomicUnsupportedError(TranslationDiagnostic):
    """A byte-address buffer atomic that HLSL cannot perform as written."""

    def __init__(self, func_name, reason):
        super().__init__(
            f"byte-address buffer atomic '{func_name}' {reason}",
            target="directx",
            code=f"{CODE_PREFIX}byte-address-atomic-unsupported",
        )


def _call_name(expr):
    func_expr = getattr(expr, "function", getattr(expr, "name", None))
    if isinstance(func_expr, str):
        return func_expr
    name = getattr(func_expr, "name", None)
    return name if isinstance(name, str) else None


def _resource_name(codegen, buffer):
    resource_type = codegen.hlsl_buffer_helper_resource_type(buffer)
    if resource_type is None:
        return None
    return codegen.hlsl_resource_type_name(resource_type)


def _type_label(codegen, expr):
    return codegen.type_name_string(codegen.expression_result_type(expr))


def _is_unsigned_literal(codegen, expr):
    value = codegen.literal_int_value(expr, codegen.literal_int_constants)
    return value is not None and value >= 0


def _operand(codegen, func_name, expr, role, *, signed=False, bits=False):
    """``expr`` rendered as the uint (or, when ``signed``, int) operand of the method."""
    kind = codegen.scalar_expression_kind(expr)
    if kind == "uint" or (kind == "int" and _is_unsigned_literal(codegen, expr)):
        return codegen.generate_expression_with_expected(expr, "uint")
    if kind == "int":
        if signed:
            return codegen.generate_expression_with_expected(expr, "int")
        rendered = codegen.generate_expression(expr)
        return f"asuint({rendered})" if bits else f"uint({rendered})"
    type_label = _type_label(codegen, expr)
    if kind is None and type_label is None:
        # An operand whose type cannot be resolved passes unchanged, as for the HLSL method form.
        return codegen.generate_expression(expr)
    raise ByteAddressAtomicUnsupportedError(
        func_name,
        f"{role} must be a 32-bit int or uint scalar, got {type_label or kind}",
    )


def _atomic_parts(codegen, func_name, args):
    """``(method, resource, buffer, offset, values, signed)`` for a byte-address atomic, or ``None``."""
    method = _METHODS.get(func_name)
    if (
        method is None
        or not args
        or func_name in getattr(codegen, "function_return_types", {})
    ):
        return None
    resource = _resource_name(codegen, args[0])
    if resource == _READ_ONLY:
        raise ByteAddressAtomicUnsupportedError(
            func_name, "cannot write the read-only ByteAddressBuffer"
        )
    if resource not in _HELPER_PREFIXES:
        return None

    roles = ("compare value", "value") if method == _COMPARE else ("value",)
    if len(args) != 2 + len(roles):
        raise ByteAddressAtomicUnsupportedError(
            func_name,
            f"takes the buffer, a byte offset and {' and '.join(roles)}; "
            f"got {len(args)} argument(s)",
        )
    signed = (
        method in _SIGNED_METHODS
        and codegen.scalar_expression_kind(args[2]) == "int"
        and not _is_unsigned_literal(codegen, args[2])
    )
    buffer = codegen.generate_expression(args[0])
    offset = _operand(codegen, func_name, args[1], "byte offset")
    values = [
        _operand(codegen, func_name, arg, role, signed=signed, bits=True)
        for arg, role in zip(args[2:], roles)
    ]
    return method, resource, buffer, offset, values, signed


def hlsl_byte_address_atomic_statement(codegen, expr):
    """The HLSL method statement (without ``;``) for a standalone byte-address atomic call, or ``None``."""
    if not isinstance(expr, FunctionCallNode):
        return None
    func_name = _call_name(expr)
    args = list(getattr(expr, "arguments", getattr(expr, "args", [])) or [])
    parts = _atomic_parts(codegen, func_name, args)
    if parts is None:
        return None
    method, _resource, buffer, offset, values, _signed = parts
    if method == _EXCHANGE:
        # InterlockedExchange always takes the out argument, so a standalone exchange still calls the helper.
        return hlsl_byte_address_atomic_expression(codegen, func_name, args)
    if method == _COMPARE:
        method = "InterlockedCompareStore"
    return f"{buffer}.{method}({', '.join([offset, *values])})"


def hlsl_byte_address_atomic_expression(codegen, func_name, args):
    """A helper call returning the previous word for a byte-address atomic whose value is used, or ``None``."""
    parts = _atomic_parts(codegen, func_name, list(args or []))
    if parts is None:
        return None
    method, resource, buffer, offset, values, signed = parts
    value_type = "int" if signed else "uint"
    operation = (
        "compare_exchange"
        if method == _COMPARE
        else method[len("Interlocked") :].lower()
    )
    name = f"{_HELPER_PREFIXES[resource]}_{operation}_{value_type}"
    helpers = codegen.__dict__.setdefault("hlsl_byte_address_atomic_helpers", {})
    if name not in helpers:
        parameters = (
            ["uint compareValue", f"{value_type} value"]
            if method == _COMPARE
            else [f"{value_type} value"]
        )
        arguments = ["compareValue", "value"] if method == _COMPARE else ["value"]
        helpers[name] = (
            f"{value_type} {name}({resource} buffer, uint offset, "
            f"{', '.join(parameters)}) {{\n"
            f"    {value_type} original;\n"
            f"    buffer.{method}(offset, {', '.join(arguments)}, original);\n"
            "    return original;\n"
            "}\n\n"
        )
    return f"{name}({', '.join([buffer, offset, *values])})"


def emit_hlsl_byte_address_atomic_helpers(codegen):
    """The byte-address atomic helpers the program calls, in the order they were first needed."""
    return "".join(
        codegen.__dict__.get("hlsl_byte_address_atomic_helpers", {}).values()
    )

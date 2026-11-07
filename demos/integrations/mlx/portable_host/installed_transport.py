"""Packed storage transport for the installed plain unary and binary families.

``HostRuntime.dispatch`` runs the installed ``float32``, ``int32`` and
``uint32`` unary and binary packages through word lists, which costs a Python
object per element at upload and at readback. When the host keeps none of an
entry's values, those families carry each buffer's own little-endian storage
bytes instead (``crosstl.project.packed_storage``): every target holds these
three types unchanged, so the packed packet is the buffer itself.

Their compact records hold no operand or output values, only a binary
family's output guard, so the record written here is byte-identical to the
one the word-list path writes. Float operands keep their exact bits, NaN
payloads included, where the word list would canonicalize them.
"""

import ctypes
import struct
import sys

from demos.integrations.mlx.portable_host.element_storage import (
    buffer_storage,
    packed_packet,
)
from demos.integrations.mlx.portable_host.trace_records import (
    retains_values,
    write_trace,
)

__all__ = ("VALUE_CODES", "finish", "packed_operation", "packed_value")

# The little-endian ``struct`` format of each plain dtype's logical values.
VALUE_CODES = {"float32": "f", "int32": "i", "uint32": "I"}


def packed_operation(host, entry, supplied, plain):
    """Whether this dispatch of ``entry`` carries packed storage.

    ``plain`` says ``entry`` is an installed plain unary or binary family.
    Every supplied buffer must hold one of ``VALUE_CODES``' types, and the
    host must keep neither the entry's values nor its native modules, whose
    records and audits read the word lists.
    """
    return (
        plain
        and sys.byteorder == "little"
        and not host.retain_native_modules
        and not retains_values(host, entry)
        and all(
            buffer.dtype.decode("ascii") in VALUE_CODES for buffer in supplied.values()
        )
    )


def _pack(dtype, values):
    return struct.pack(f"<{len(values)}{VALUE_CODES[dtype]}", *values)


def packed_value(buffer, dtype, target, guard):
    """The packed packet of a buffer; an output holds zeros, then ``guard``.

    ``guard`` lists the logical values the word-list path appends to a
    guarded output, or is empty.
    """
    if buffer.output:
        width = struct.calcsize("<" + VALUE_CODES[dtype])
        data = bytes(buffer.count * width) + _pack(dtype, guard)
    else:
        data = buffer_storage(buffer, dtype)
    return packed_packet(dtype, target, data)


def _read_back(output, expected, buffer, guard):
    """Check one packed readback, store its elements and return its guard."""
    if output.get("dtype") != expected["dtype"] or output.get("shape") != (
        expected["shape"]
    ):
        raise RuntimeError("Native readback layout does not match the output")
    if output.get("encoding") != expected.get("encoding"):
        raise RuntimeError("Native readback storage encoding does not match the output")
    data = output.get("values")
    initial = expected["values"]
    if type(data) is not bytes or len(data) != len(initial):
        raise RuntimeError("Native readback size does not match the output")
    dtype = buffer.dtype.decode("ascii")
    stored = buffer.count * struct.calcsize("<" + VALUE_CODES[dtype])
    tail = data[stored:]
    if tail != initial[stored:]:
        raise RuntimeError("Native operation changed the output buffer guard")
    ctypes.memmove(buffer.data, data, stored)
    return list(struct.unpack(f"<{len(guard)}{VALUE_CODES[dtype]}", tail))


def finish(host, event, result, outputs, destinations, guard, binary):
    """Store a packed dispatch's outputs and write its record.

    ``event`` holds the record's leading keys, which every family's record
    shares; a binary family's record adds its output guard as read back.
    """
    guard_values = []
    for name, (buffer, _) in destinations.items():
        guard_values = _read_back(result.outputs[name], outputs[name], buffer, guard)
    write_trace(
        host,
        {**event, **({"binaryGuardValues": guard_values} if binary else {})},
    )
    host.dispatch_count = getattr(host, "dispatch_count", 0) + 1

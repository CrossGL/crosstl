"""Recompute DivMod results from the operands a native dispatch retained.

Integer results follow ``FloorDivide`` and ``Remainder`` in ``binary_ops.h``
exactly. Floating results use correctly rounded division and MLX's floored
``fmod``; binary16 remainders follow Metal's binary32-quotient evaluation,
which is the profile the translated kernels pin. The audit covers zero and
normal finite values only, so an event that reaches subnormals, infinities or
NaNs fails instead of passing on an unmodelled flush.
"""

import ctypes
import hashlib
import math
import struct

from demos.integrations.mlx.portable_host.divmod_dispatch import (
    CTYPES,
    GUARD_COUNT,
    GUARDS,
    logical_operands,
    packet,
)
from demos.integrations.mlx.portable_host.divmod_packages import signature
from demos.integrations.mlx.portable_host.gather_evidence import (
    audit_native_execution,
)

FLOATS = {"float16": (16, 5, 10), "bfloat16": (16, 8, 7), "float32": (32, 8, 23)}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def float_value(word, dtype):
    if dtype == "float32":
        return struct.unpack("<f", struct.pack("<I", word))[0]
    if dtype == "float16":
        return struct.unpack("<e", struct.pack("<H", word))[0]
    return struct.unpack("<f", struct.pack("<I", word << 16))[0]


def float_word(value, dtype):
    """Round a binary64 value to ``dtype`` with round-to-nearest-even."""
    if dtype == "float32":
        return struct.unpack("<I", struct.pack("<f", value))[0]
    if dtype == "float16":
        return struct.unpack("<H", struct.pack("<e", value))[0]
    bits = struct.unpack("<I", struct.pack("<f", value))[0]
    return (bits + 0x7FFF + ((bits >> 16) & 1)) >> 16


def normal(word, dtype):
    """True for signed zeros and normal finite values of ``dtype``."""
    width, exponent_bits, fraction_bits = FLOATS[dtype]
    magnitude = word & ((1 << (width - 1)) - 1)
    exponent = magnitude >> fraction_bits
    return magnitude == 0 or 0 < exponent < (1 << exponent_bits) - 1


def round_to(value, dtype):
    word = float_word(value, dtype)
    require(normal(word, dtype), "DivMod evidence covers normal binary values only")
    return word


def binary32(value):
    return struct.unpack("<f", struct.pack("<f", value))[0]


def float_divmod(x, y, dtype):
    require(
        normal(x, dtype) and normal(y, dtype),
        "DivMod evidence covers normal binary operands only",
    )
    a, b = float_value(x, dtype), float_value(y, dtype)
    require(b != 0, "DivMod evidence requires nonzero floating divisors")
    # Binary64 holds every intermediate exactly or rounds it innocuously
    # (at least 2p + 2 bits), so one final rounding matches the target type.
    quotient = float_value(round_to(a / b, dtype), dtype)
    floor_quotient = math.copysign(float(math.floor(quotient)), quotient)
    if dtype == "float16":
        wide = binary32(a / b)
        integral = math.copysign(float(math.trunc(wide)), wide)
        remainder = float_value(
            round_to(binary32(a - binary32(integral * b)), dtype), dtype
        )
    else:
        remainder = math.fmod(a, b)
    if remainder != 0 and (remainder < 0) != (b < 0):
        remainder += b
    return round_to(floor_quotient, dtype), round_to(remainder, dtype)


def integer_divmod(x, y, dtype):
    require(y != 0, "DivMod evidence requires nonzero integer divisors")
    bits = 8 * ctypes.sizeof(CTYPES[dtype])
    quotient = abs(x) // abs(y) * (1 if (x < 0) == (y < 0) else -1)
    remainder = x - quotient * y
    if remainder != 0 and (x < 0) != (y < 0):
        quotient -= 1
    if remainder != 0 and (remainder < 0) != (y < 0):
        remainder += y
    signed = dtype.startswith("int")
    for value in (quotient, remainder):
        require(
            (
                -(1 << (bits - 1)) <= value < (1 << (bits - 1))
                if signed
                else 0 <= value < (1 << bits)
            ),
            "DivMod evidence result overflows its integer type",
        )
    return quotient, remainder


def audit_event(event, *, native=True):
    """Check one DivMod trace event against its retained operands.

    ``native`` also verifies the retained native modules, the artifact
    identity and the request geometry, as for source-specialized indexing.
    """
    kernel, dtype = signature(event["entry"])
    metadata = event["divmodMetadata"]
    require(
        metadata["kernel"] == kernel and metadata["dtype"] == dtype,
        "DivMod event metadata does not match its entry",
    )
    operands = event["operandValues"]
    target = event["target"]
    expected_inputs = {
        name: packet(dtype, target, operands[name] + [GUARDS[dtype]] * GUARD_COUNT)
        for name in "ab"
    }
    uploads = [*event["inputs"].values()]
    for name, value in expected_inputs.items():
        require(value in uploads, f"DivMod upload of {name} differs from its operands")
    threads = event["threads"]
    require(
        math.prod(metadata["shape"]) == threads
        and all(len(event["outputValues"][name]) == threads for name in "cd")
        and all(
            event["outputGuards"][name] == [GUARDS[dtype]] * GUARD_COUNT
            for name in "cd"
        ),
        "DivMod readback extent or guard differs",
    )
    pairs = logical_operands(metadata, operands["a"], operands["b"])
    compute = float_divmod if dtype in FLOATS else integer_divmod
    for index, (x, y) in enumerate(pairs):
        quotient, remainder = compute(x, y, dtype)
        require(
            event["outputValues"]["c"][index] == quotient
            and event["outputValues"]["d"][index] == remainder,
            f"DivMod result {index} of {event['entry']} differs from its operands",
        )
    payload = b"".join(
        bytes((CTYPES[dtype] * threads)(*event["outputValues"][name])) for name in "cd"
    )
    require(
        hashlib.sha256(payload).hexdigest() == event.get("outputHash"),
        "DivMod host output hash changed",
    )
    if native:
        request = event["details"]["request"]
        expected_entry = {"opengl": "main", "directx": "CSMain"}.get(
            target, event["entry"]
        )
        require(
            request["target"] == target
            and request["entryPoint"] == expected_entry
            and request["dispatch"]["workgroupCount"] == event["workgroupCount"]
            and request["dispatch"]["workgroupSize"]
            == event["workgroupSize"]
            == [1, 1, 1]
            and event["dispatchVersion"] == 3,
            "DivMod native request changed",
        )
        audit_native_execution(event)
    return threads

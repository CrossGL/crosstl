"""Recompute lazily translated casts and copies from their retained values.

General copies move storage words, so each destination element must equal the
source element its strides select and every other element must keep its
uploaded word. Casts follow ``cast_to`` in the pinned ``utils.h``: Booleans
test for nonzero storage (subnormals and NaNs are true, and a complex value is
true when either part is), ``complex64_t`` converts to and from its real part
as a float32 with a +0 imaginary part (``complex.h``), floating results are
rounded to nearest even with gradual underflow, integers wrap modulo their
width and floating values truncate toward zero. The audit refuses what no
target defines or the pinned profiles leave open (NaN or out-of-range values
converted to integers, binary32 subnormals widened to bfloat16 or the reverse,
and integers that need two roundings to reach a 16-bit float) instead of
passing on an unmodelled result.
"""

import ctypes
import hashlib
from fractions import Fraction

from demos.integrations.mlx.portable_host import copy_layout
from demos.integrations.mlx.portable_host.copy_packages import VECTOR, signature
from demos.integrations.mlx.portable_host.element_storage import (
    COMPLEX,
    CTYPES,
    FLOAT32_MASK,
    FLOATING,
    guard_words,
    packet,
)
from demos.integrations.mlx.portable_host.gather_evidence import (
    audit_native_execution,
    require,
)
from demos.integrations.mlx.portable_host.lazy_package_evidence import (
    require_unchanged,
)

# (width, exponent bits, fraction bits)
FLOATS = {"float16": (16, 5, 10), "bfloat16": (16, 8, 7), "float32": (32, 8, 23)}
NAN = "nan"


def float_value(word, dtype):
    """``(value, negative, subnormal)``; value is a Fraction, "inf" or NAN."""
    width, exponent_bits, fraction_bits = FLOATS[dtype]
    bias = (1 << (exponent_bits - 1)) - 1
    negative = bool(word >> (width - 1))
    exponent = (word >> fraction_bits) & ((1 << exponent_bits) - 1)
    fraction = word & ((1 << fraction_bits) - 1)
    if exponent == (1 << exponent_bits) - 1:
        return (NAN if fraction else "inf"), negative, False
    if exponent == 0:
        value = Fraction(fraction, 1 << (fraction_bits + bias - 1))
    else:
        value = Fraction((1 << fraction_bits) | fraction) * Fraction(2) ** (
            exponent - bias - fraction_bits
        )
    return (-value if negative else value), negative, exponent == 0 and fraction != 0


def float_word(value, dtype, negative=False):
    """Round an exact value to ``dtype``: nearest even, gradual underflow."""
    width, exponent_bits, fraction_bits = FLOATS[dtype]
    bias = (1 << (exponent_bits - 1)) - 1
    sign = int(value < 0 or (value == 0 and negative)) << (width - 1)
    magnitude = abs(Fraction(value))
    if magnitude == 0:
        return sign
    exponent = magnitude.numerator.bit_length() - magnitude.denominator.bit_length()
    if Fraction(2) ** exponent > magnitude:
        exponent -= 1
    exponent = max(exponent, 1 - bias)
    significand = round(magnitude / Fraction(2) ** (exponent - fraction_bits))
    if significand == 1 << (fraction_bits + 1):
        significand, exponent = significand >> 1, exponent + 1
    if exponent > bias:
        return sign | (((1 << exponent_bits) - 1) << fraction_bits)
    biased = exponent + bias if significand >> fraction_bits else 0
    return sign | (biased << fraction_bits) | (significand & ((1 << fraction_bits) - 1))


def is_nan(word, dtype):
    return float_value(word, dtype)[0] == NAN


def integer_bounds(dtype):
    bits = 8 * ctypes.sizeof(CTYPES[dtype])
    return (
        (-(1 << (bits - 1)), (1 << (bits - 1)) - 1)
        if dtype.startswith("int")
        else (0, (1 << bits) - 1)
    )


def wrap(value, dtype):
    low, high = integer_bounds(dtype)
    return (value - low) % (high - low + 1) + low


def cast_word(word, source, destination):
    """The destination storage word of one ``cast_to``; NAN for any NaN."""
    if source == COMPLEX:
        real, imag = word & FLOAT32_MASK, word >> 32
        if destination == "bool_":
            return int(
                cast_word(real, "float32", "bool_")
                or cast_word(imag, "float32", "bool_")
            )
        return cast_word(real, "float32", destination)
    if destination == COMPLEX:
        # complex_t(U x) stores static_cast<float>(x) and a +0 imaginary part.
        return cast_word(word, source, "float32")
    if destination == "bool_":
        if source in FLOATING:
            width = FLOATS[source][0]
            return int(word & ((1 << (width - 1)) - 1) != 0)
        return int(word != 0)
    if source in FLOATING:
        value, negative, subnormal = float_value(word, source)
    else:
        value, negative, subnormal = Fraction(word), False, False
    if destination in FLOATING:
        if value == NAN:
            return NAN
        width, exponent_bits, fraction_bits = FLOATS[destination]
        if value == "inf":
            return (int(negative) << (width - 1)) | (
                ((1 << exponent_bits) - 1) << fraction_bits
            )
        require(
            not (subnormal and {source, destination} == {"float32", "bfloat16"}),
            "Cast evidence does not model binary32 subnormals in bfloat16 casts",
        )
        if source not in FLOATING and destination != "float32":
            single = float_value(float_word(value, "float32"), "float32")[0]
            require(
                single == value,
                "Cast evidence requires integers exact in binary32 for 16-bit floats",
            )
        return float_word(value, destination, negative)
    if source in FLOATING:
        require(
            value not in (NAN, "inf"),
            "Cast evidence refuses NaN or infinite values converted to integers",
        )
        truncated = int(value)
        low, high = integer_bounds(destination)
        require(
            low <= truncated <= high,
            "Cast evidence refuses floating values outside the integer range",
        )
        return truncated
    return wrap(int(value), destination)


def expected_copy(metadata, source, initial):
    """Every destination word after a general copy of ``source``."""
    result = list(initial)
    reads = copy_layout.destination_indices(
        {
            "shape": metadata["shape"],
            "destinationStrides": metadata["sourceStrides"],
            "destinationOffset": metadata["sourceOffset"],
        }
    )
    for read, write in zip(reads, copy_layout.destination_indices(metadata)):
        result[write] = source[read]
    return result


def matches(actual, expected, dtype):
    if actual == expected or expected != NAN:
        return actual == expected
    if dtype == COMPLEX:
        return actual >> 32 == 0 and is_nan(actual & FLOAT32_MASK, "float32")
    return is_nan(actual, dtype)


def audit_event(event, cache=None, *, native=True):
    """Check one cast or copy dispatch against its retained values.

    ``cache`` derives the event's package again from the pinned source;
    ``native`` also verifies the retained native modules and the request.
    Returns the number of destination elements checked.
    """
    entry, target = event["entry"], event["target"]
    kernel, source, destination = signature(entry)
    metadata = event["copyMetadata"]
    require(
        (metadata["kernel"], metadata["source"], metadata["destination"])
        == (kernel, source, destination),
        "Copy event metadata does not match its entry",
    )
    values, initial = event["sourceValues"], event["initialValues"]
    outputs = event["outputValues"]
    uploads = [*event["inputs"].values()]
    for value in (
        packet(source, target, values),
        packet(destination, target, initial + guard_words(destination)),
    ):
        require(value in uploads, "Copy upload differs from its retained values")
    threads = event["threads"]
    if kernel == VECTOR:
        require(
            len(values) == len(initial) == len(outputs) == threads,
            "Cast readback extent differs",
        )
        expected = [cast_word(word, source, destination) for word in values]
    else:
        require(
            metadata["sourceCount"] == len(values)
            and metadata["destinationCount"] == len(initial) == len(outputs),
            "Copy readback extent differs",
        )
        expected = expected_copy(metadata, values, initial)
    for index, (actual, wanted) in enumerate(zip(outputs, expected)):
        require(
            matches(actual, wanted, destination),
            f"Copy result {index} of {entry} differs from its retained values",
        )
    payload = bytes((CTYPES[destination] * len(outputs))(*outputs))
    require(
        hashlib.sha256(payload).hexdigest() == event.get("outputHash"),
        "Copy host output hash changed",
    )
    if cache is not None:
        descriptor, directory = cache.get(entry)
        require_unchanged(descriptor, directory)
        require(
            descriptor["artifact"] == event["artifact"]
            and str(directory) == event["packageRoot"],
            "Copy dispatch does not name its cached package",
        )
    if native:
        request = event["details"]["request"]
        require(
            request["target"] == target
            and request["entryPoint"]
            == {"opengl": "main", "directx": "CSMain"}.get(target, entry)
            and request["dispatch"]["workgroupCount"] == event["workgroupCount"]
            and request["dispatch"]["workgroupSize"]
            == event["workgroupSize"]
            == [1, 1, 1]
            and event["dispatchVersion"] == 3,
            "Copy native request changed",
        )
        audit_native_execution(event)
    return len(outputs)

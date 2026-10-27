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

from crosstl.project.runtime_values import exact_types_within
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


def _float_parts(np, words, dtype):
    """Integer ``(negative, significand, exponent, nan, infinite, subnormal)``.

    A finite word's value is ``significand * 2**exponent``, exactly.
    """
    width, exponent_bits, fraction_bits = FLOATS[dtype]
    bias = (1 << (exponent_bits - 1)) - 1
    negative = (words >> (width - 1)) & 1
    exponent = (words >> fraction_bits) & ((1 << exponent_bits) - 1)
    fraction = words & ((1 << fraction_bits) - 1)
    special = exponent == (1 << exponent_bits) - 1
    normal = exponent != 0
    significand = np.where(normal, fraction | (1 << fraction_bits), fraction)
    scale = np.where(normal, exponent, 1) - bias - fraction_bits
    subnormal = ~normal & (fraction != 0)
    return (
        negative,
        significand,
        scale,
        special & (fraction != 0),
        special & (fraction == 0),
        subnormal,
    )


def _float_words(np, negative, significand, scale, dtype):
    """``float_word`` of each ``significand * 2**scale``, for nonnegative
    significands below ``2**53``: nearest even, gradual underflow, overflow to
    infinity.
    """
    width, exponent_bits, fraction_bits = FLOATS[dtype]
    bias = (1 << (exponent_bits - 1)) - 1
    # floor(log2(significand)) is exact through float64 for these widths.
    top = np.frexp(significand.astype(np.float64))[1].astype(np.int64) - 1
    exponent = np.maximum(top + scale, 1 - bias)
    drop = np.clip(exponent - fraction_bits - scale, -62, 62)
    left = np.where(drop < 0, significand << np.maximum(-drop, 0), significand)
    shift = np.maximum(drop, 0)
    kept = left >> shift
    rest = left - (kept << shift)
    half = np.where(shift > 0, np.int64(1) << np.maximum(shift - 1, 0), 0)
    up = (shift > 0) & ((rest > half) | ((rest == half) & ((kept & 1) == 1)))
    kept = kept + up
    carry = kept == (1 << (fraction_bits + 1))
    kept = np.where(carry, kept >> 1, kept)
    exponent = exponent + carry
    biased = np.where(kept >> fraction_bits, exponent + bias, 0)
    fraction = kept & ((1 << fraction_bits) - 1)
    infinite = (biased >= (1 << exponent_bits) - 1) & (significand != 0)
    biased = np.where(infinite, (1 << exponent_bits) - 1, biased)
    fraction = np.where(infinite | (significand == 0), 0, fraction)
    biased = np.where(significand == 0, 0, biased)
    return (negative << (width - 1)) | (biased << fraction_bits) | fraction


def float_casts(np, words, source, destination):
    """``cast_word`` of each floating ``words`` to a floating ``destination``.

    Returns ``(words, nan)``: the destination words as an int64 array and the
    mask of results that ``cast_word`` gives as ``NAN``.
    """
    words = np.asarray(words, dtype=np.int64)
    negative, significand, scale, nan, infinite, subnormal = _float_parts(
        np, words, source
    )
    require(
        not ({source, destination} == {"float32", "bfloat16"} and subnormal.any()),
        "Cast evidence does not model binary32 subnormals in bfloat16 casts",
    )
    width, exponent_bits, fraction_bits = FLOATS[destination]
    finite = np.where(nan | infinite, 0, significand)
    result = _float_words(np, negative, finite, scale, destination)
    result = np.where(
        infinite,
        (negative << (width - 1)) | (((1 << exponent_bits) - 1) << fraction_bits),
        result,
    )
    return np.where(nan, 0, result), nan


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


def strided_indices(np, shape, strides, offset):
    """``copy_layout.destination_indices`` as one int64 array, in its order."""
    shape = list(shape)
    if any(size <= 0 for size in shape):
        return np.zeros(0, dtype=np.int64)
    strides = list(strides)[: len(shape)]
    strides += [0] * (len(shape) - len(strides))
    index = np.array([offset], dtype=np.int64)
    for size, stride in zip(shape, strides):
        steps = np.arange(size, dtype=np.int64) * stride
        index = (index[:, None] + steps).reshape(-1)
    return index


_WORD_TYPES = frozenset((int,))


def _require_general_copy(entry, metadata, values, initial, outputs, destination):
    """Check a general copy whole; return its output bytes, or ``None``.

    The result must be ``expected_copy``'s, word for word. ``None`` leaves
    lists that are not all ``int`` words, and destinations whose strides do
    not prove each write lands on its own element, to the element-wise check,
    which applies overlapping writes in order.
    """
    import numpy as np

    shape = list(metadata["shape"])
    strides = list(metadata["destinationStrides"])[: len(shape)]
    strides += [0] * (len(shape) - len(strides))
    if not copy_layout.distinct_by_strides(shape, strides) or not all(
        exact_types_within(words, _WORD_TYPES) for words in (values, initial, outputs)
    ):
        return None
    dtype = np.dtype(CTYPES[destination])
    try:
        source = np.asarray(values, dtype=dtype)
        result = np.array(initial, dtype=dtype)
        actual = np.asarray(outputs, dtype=dtype)
    except OverflowError:
        require(False, f"Copy values of {entry} are not {destination} words")
    reads = strided_indices(
        np, shape, metadata["sourceStrides"], metadata["sourceOffset"]
    )
    writes = strided_indices(np, shape, strides, metadata["destinationOffset"])
    require(
        reads.size == 0
        or (
            reads.min() >= 0
            and reads.max() < source.size
            and writes.min() >= 0
            and writes.max() < result.size
        ),
        f"Copy indices of {entry} leave the retained values",
    )
    result[writes] = source[reads]
    wrong = actual != result
    if wrong.any():
        index = int(np.argmax(wrong))
        require(
            False, f"Copy result {index} of {entry} differs from its retained values"
        )
    return actual.tobytes()


def matches(actual, expected, dtype):
    if actual == expected or expected != NAN:
        return actual == expected
    if dtype == COMPLEX:
        return actual >> 32 == 0 and is_nan(actual & FLOAT32_MASK, "float32")
    return is_nan(actual, dtype)


def _require_float_casts(entry, values, outputs, source, destination):
    """Check a cast between floating types whole; return its output bytes.

    Each result must be ``cast_word`` of its source word, or any NaN of the
    destination type where that is ``NAN``.
    """
    import numpy as np

    wanted, nan = float_casts(np, values, source, destination)
    actual = np.asarray(outputs, dtype=np.int64)
    width = FLOATS[destination][0]
    nan_result = _float_parts(np, actual, destination)[3]
    wrong = (
        ((actual >> width) != 0)
        | (actual < 0)
        | np.where(nan, ~nan_result, actual != wanted)
    )
    if wrong.any():
        index = int(np.argmax(wrong))
        require(
            False, f"Copy result {index} of {entry} differs from its retained values"
        )
    return actual.astype(f"=u{width // 8}").tobytes()


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
    payload = None
    if kernel == VECTOR:
        require(
            len(values) == len(initial) == len(outputs) == threads,
            "Cast readback extent differs",
        )
        if source in FLOATS and destination in FLOATS:
            payload = _require_float_casts(entry, values, outputs, source, destination)
            expected = ()
        else:
            expected = [cast_word(word, source, destination) for word in values]
    else:
        require(
            metadata["sourceCount"] == len(values)
            and metadata["destinationCount"] == len(initial) == len(outputs),
            "Copy readback extent differs",
        )
        payload = _require_general_copy(
            entry, metadata, values, initial, outputs, destination
        )
        expected = (
            () if payload is not None else expected_copy(metadata, values, initial)
        )
    for index, (actual, wanted) in enumerate(zip(outputs, expected)):
        require(
            matches(actual, wanted, destination),
            f"Copy result {index} of {entry} differs from its retained values",
        )
    if payload is None:
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

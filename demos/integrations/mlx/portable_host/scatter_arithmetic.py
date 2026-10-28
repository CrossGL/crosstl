"""Element arithmetic of MLX's scatter reductions for the lane-packed types.

An audit recomputes a scatter from its uploads with the operation MLX's
atomic update applies to one lane (``reduction/ops.h``): replacement, the sum
or product in the element type, and the minimum or maximum that replaces a
lane only when ``Op::condition(update, current)`` holds, so a NaN update never
replaces a lane and a NaN lane is never replaced. The 8- and 16-bit integers
wrap modulo their width; ``float16`` rounds each step to nearest even, as
NumPy's binary16 does, and ``bfloat16`` computes in float32 and rounds the
result as MLX's ``_MLX_BFloat16(float)`` does, to nearest even with every NaN
becoming ``0x7FC0``. Both roundings are single: float32 holds every exact sum
and product of two 16-bit floats closely enough that rounding twice cannot
differ from rounding once.
"""

import ctypes

from demos.integrations.mlx.portable_host.element_storage import CTYPES

INTEGERS = frozenset({"uint8", "int8", "uint16", "int16"})
BFLOAT16_NAN = 0x7FC0


def bfloat16_bits(value):
    """``_MLX_BFloat16(float)`` of a Python or NumPy float."""
    bits = ctypes.c_uint32.from_buffer_copy(ctypes.c_float(float(value))).value
    if (bits & 0x7FFFFFFF) > 0x7F800000:
        return BFLOAT16_NAN
    return ((bits + ((bits >> 16) & 1) + 0x7FFF) & 0xFFFFFFFF) >> 16


def bfloat16_value(word):
    """The float32 value of a bfloat16 storage word."""
    return ctypes.c_float.from_buffer_copy(ctypes.c_uint32(word << 16)).value


def values(np, dtype, words):
    """Logical storage words as the values the operation combines."""
    words = np.asarray(words)
    if dtype in INTEGERS:
        return words.astype(dtype)
    if dtype == "float16":
        return words.astype(np.uint16).view(np.float16)
    if dtype == "bfloat16":
        return (words.astype(np.uint32) << 16).view(np.float32)
    raise ValueError(f"Unsupported lane-packed scatter type: {dtype!r}")


def storage_words(np, dtype, array):
    """Values back to their exact storage words, keeping NaN payloads.

    A bfloat16 value is the float32 of its word, so its word is the high half
    of those bits; a value with low bits set was never rounded and is refused.
    """
    array = np.ascontiguousarray(array).reshape(-1)
    if dtype in INTEGERS:
        return [int(value) for value in array.tolist()]
    if dtype == "float16":
        return array.astype(np.float16).view(np.uint16).tolist()
    bits = array.astype(np.float32).view(np.uint32)
    if (bits & 0xFFFF).any():
        raise ValueError("A bfloat16 scatter value was not rounded to its storage")
    return (bits >> 16).tolist()


def combine(np, dtype, operation, current, update):
    """One lane update, as MLX's atomic update applies it."""
    if operation == "none":
        return update
    if operation in ("min", "max"):
        replace = update < current if operation == "min" else update > current
        return update if replace else current
    if operation not in ("sum", "prod"):
        raise ValueError(f"Unsupported scatter operation: {operation!r}")
    if dtype in INTEGERS:
        left, right = int(current), int(update)
        result = left + right if operation == "sum" else left * right
        return np.array(CTYPES[dtype](result).value, dtype=dtype)[()]
    with np.errstate(all="ignore"):
        if dtype == "float16":
            left, right = np.float16(current), np.float16(update)
            return left + right if operation == "sum" else left * right
        left, right = np.float32(current), np.float32(update)
        result = left + right if operation == "sum" else left * right
    return np.float32(bfloat16_value(bfloat16_bits(result)))

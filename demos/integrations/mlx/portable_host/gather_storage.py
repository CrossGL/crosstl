"""Exact element storage for indexing values outside the original word types.

General and axis gathers first carried float32, the 32- and 64-bit integers
and Booleans as their own runtime words (``gather_layout.WORD_TYPES``). Every
other MLX storage type an indexing kernel instantiates (``float16``,
``bfloat16``, ``complex64`` and the 8- and 16-bit integers) travels as MLX's
logical storage words through ``element_storage``: packed into the carrier the
target reflects, initialised and trailed with ``element_storage`` guard words,
and decoded exactly on readback. Indices and metadata keep their word path.
"""

import hashlib

from demos.integrations.mlx.portable_host.element_storage import (
    CTYPES,
    GUARD_COUNT,
    buffer_words,
    guard_words,
    packet,
    reflected_layout_matches,
    unpack,
)
from demos.integrations.mlx.portable_host.gather_layout import WORD_TYPES

VALUES = frozenset({"src", "out"})
LAYOUT_KEYS = ("dtype", "encoding", "shape")
# NumPy storage types that hold each extended dtype's logical words.
NUMPY_WORDS = {
    "float16": "uint16",
    "bfloat16": "uint16",
    "complex64": "uint64",
    "int8": "int8",
    "int16": "int16",
    "uint8": "uint8",
    "uint16": "uint16",
}


def extended(dtype):
    """Whether ``dtype``'s values travel as ``element_storage`` words."""
    if dtype in WORD_TYPES:
        return False
    if dtype not in NUMPY_WORDS:
        raise ValueError(f"Unsupported indexing storage dtype: {dtype!r}")
    return True


def value_packet(dtype, target, buffer, output):
    """The runtime value for one value buffer; outputs start as guards."""
    if output:
        words = guard_words(dtype, buffer.count) + guard_words(dtype)
    else:
        words = buffer_words(buffer, dtype)
    return packet(dtype, target, words)


def bind_value(dtype, target, buffer, layout, output):
    value = value_packet(dtype, target, buffer, output)
    if not reflected_layout_matches(layout, value, dtype):
        raise ValueError("Native and reflected gather layouts disagree")
    return value


def readback(dtype, target, expected, returned, threads):
    """Decode a native output exactly and require its trailing guard."""
    if any(returned.get(key) != expected.get(key) for key in LAYOUT_KEYS) or (
        not isinstance(returned.get("values"), list)
    ):
        raise RuntimeError("Native gather readback layout does not match")
    try:
        words = unpack(dtype, target, returned["values"])
    except ValueError as error:
        raise RuntimeError("Native gather readback is outside its storage type") from (
            error
        )
    if len(words) != threads + GUARD_COUNT:
        raise RuntimeError("Native gather readback layout does not match")
    if words[threads:] != guard_words(dtype):
        raise RuntimeError("Native gather changed the output buffer guard")
    return words


def native_words(dtype, words):
    """``words`` as the host's exact storage array for ``dtype``."""
    return (CTYPES[dtype] * len(words))(*words)


def output_hash(dtype, words):
    return hashlib.sha256(bytes(native_words(dtype, words))).hexdigest()


def decoded(np, dtype, target, value):
    """A retained value packet's logical words as a NumPy array."""
    return np.array(unpack(dtype, target, value["values"]), dtype=NUMPY_WORDS[dtype])

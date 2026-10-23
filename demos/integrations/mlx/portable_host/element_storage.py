"""Exact transport of MLX element storage through native buffer carriers.

Host buffers hold MLX's logical storage words: IEEE bits for the floating
types, two's-complement values for the integers, canonical zero or one bytes
for Booleans, and for ``complex64`` one 64-bit word whose low half holds the
real part's float32 bits and whose high half holds the imaginary part's. Each
target reflects its own physical carrier. Direct3D and OpenGL keep Booleans
and the narrow integers they lack in 32-bit words, OpenGL keeps 16-bit floats
in float32 carriers, Direct3D keeps bfloat16 bits in uint16 words, and every
target reflects ``complex64_t`` as a tightly packed ``{real, imag}`` float32
struct. Packing and unpacking are exact: a carrier that does not hold a value
of its logical type is refused, never rounded or wrapped.
"""

import ctypes

from crosstl.project.runtime_value_encoding import (
    BFLOAT16_BITS,
    FLOAT16_BITS,
    FLOAT32_BITS,
)
from demos.integrations.mlx.portable_host import bfloat_storage, half_storage

DTYPES = (
    "bool_",
    "uint8",
    "uint16",
    "uint32",
    "uint64",
    "int8",
    "int16",
    "int32",
    "int64",
    "float16",
    "float32",
    "bfloat16",
)
FLOATING = frozenset({"float16", "float32", "bfloat16"})
COMPLEX = "complex64"
# Every MLX dtype a translated kernel can carry exactly; float64 has no GPU path.
STORAGE_DTYPES = (*DTYPES, COMPLEX)
INEXACT = FLOATING | {COMPLEX}
# Floating types are carried as their unsigned storage words.
CTYPES = {
    "bool_": ctypes.c_uint8,
    "float16": ctypes.c_uint16,
    "bfloat16": ctypes.c_uint16,
    "float32": ctypes.c_uint32,
    "uint8": ctypes.c_uint8,
    "uint16": ctypes.c_uint16,
    "uint32": ctypes.c_uint32,
    "uint64": ctypes.c_uint64,
    "int8": ctypes.c_int8,
    "int16": ctypes.c_int16,
    "int32": ctypes.c_int32,
    "int64": ctypes.c_int64,
    COMPLEX: ctypes.c_uint64,
}
# Booleans and the integers a target lacks widen to 32-bit storage words.
WIDENED = {
    "metal": {},
    "directx": {"bool_": "uint32", "uint8": "uint32", "int8": "int32"},
    "opengl": {
        "bool_": "uint32",
        "uint8": "uint32",
        "uint16": "uint32",
        "int8": "int32",
        "int16": "int32",
    },
}
PHYSICAL_SIZES = {
    "bool": 1,
    "uint8": 1,
    "int8": 1,
    "float16": 2,
    "bfloat16": 2,
    "uint16": 2,
    "int16": 2,
    "float32": 4,
    "uint32": 4,
    "int32": 4,
    "uint64": 8,
    "int64": 8,
}
GUARD_COUNT = 32
GUARDS = {
    "bool_": 1,
    "float16": 0x3555,
    "bfloat16": 0x3EAB,
    "float32": 0x6A15BEEF,
    "uint8": 0xA5,
    "uint16": 0xA5A5,
    "uint32": 0x6A15BEEF,
    "uint64": 0x6A15BEEF6A15BEEF,
    "int8": -91,
    "int16": 0x3555,
    "int32": 0x6A15BEEF,
    "int64": -0x6A15BEEF,
    COMPLEX: 0x3EAB35556A15BEEF,
}
# How every target reflects MLX's complex64_t storage element.
COMPLEX_LAYOUT = {
    "elementType": "float32",
    "physicalType": "complex_t_float",
    "componentCount": 2,
    "elementSizeBytes": 8,
    "elementStrideBytes": 8,
    "alignmentBytes": 4,
    "structMembers": [
        {"name": "real", "offsetBytes": 0, "physicalType": "float"},
        {"name": "imag", "offsetBytes": 4, "physicalType": "float"},
    ],
}
FLOAT32_MASK = 0xFFFFFFFF


def require_dtype(dtype):
    if dtype not in CTYPES:
        raise ValueError(f"Unsupported storage dtype: {dtype!r}")
    return dtype


def guard_words(dtype, count=GUARD_COUNT):
    """A fill that a kernel writing past its extent changes; Booleans alternate."""
    require_dtype(dtype)
    if dtype == "bool_":
        return [int(index % 2 == 0) for index in range(count)]
    return [GUARDS[dtype]] * count


def buffer_words(buffer, dtype):
    """Read a native buffer's logical storage words, refusing invalid Booleans."""
    ctype = CTYPES[require_dtype(dtype)]
    words = list(
        ctypes.cast(buffer.data, ctypes.POINTER(ctype * buffer.count)).contents
    )
    if dtype == "bool_" and any(word > 1 for word in words):
        raise ValueError("Boolean storage must contain canonical zero or one values")
    return words


def packet(dtype, target, words):
    """The runtime value that carries ``words`` in ``target``'s storage."""
    require_dtype(dtype)
    if dtype == COMPLEX:
        if any(type(word) is not int or not 0 <= word < 1 << 64 for word in words):
            raise ValueError("Complex storage must contain 64-bit words")
        return {
            "dtype": "float32",
            "shape": [2 * len(words)],
            "encoding": FLOAT32_BITS,
            "values": [
                part for word in words for part in (word & FLOAT32_MASK, word >> 32)
            ],
        }
    if dtype == "float32":
        return {
            "dtype": "float32",
            "shape": [len(words)],
            "encoding": FLOAT32_BITS,
            "values": list(words),
        }
    if dtype == "float16":
        physical = "float32" if target == "opengl" else "float16"
        return {
            "dtype": physical,
            "shape": [len(words)],
            "encoding": half_storage.encoding(target),
            "values": half_storage.pack(words, target),
        }
    if dtype == "bfloat16":
        words = bfloat_storage.pack(words, target)
        if target == "directx":
            return {"dtype": "uint16", "shape": [len(words)], "values": words}
        return {
            "dtype": "bfloat16" if target == "metal" else "float32",
            "shape": [len(words)],
            "encoding": BFLOAT16_BITS if target == "metal" else FLOAT32_BITS,
            "values": words,
        }
    if dtype == "bool_":
        if any(word not in (0, 1) or type(word) not in (bool, int) for word in words):
            raise ValueError(
                "Boolean storage must contain canonical zero or one values"
            )
        if target == "metal":
            return {
                "dtype": "bool",
                "shape": [len(words)],
                "values": [bool(w) for w in words],
            }
        return {
            "dtype": "uint32",
            "shape": [len(words)],
            "values": [int(w) for w in words],
        }
    return {
        "dtype": WIDENED[target].get(dtype, dtype),
        "shape": [len(words)],
        "values": [int(word) for word in words],
    }


def unpack(dtype, target, words):
    """Decode a native readback into logical storage words, exactly."""
    require_dtype(dtype)
    if dtype == COMPLEX:
        if len(words) % 2 or any(
            type(word) is not int or not 0 <= word <= FLOAT32_MASK for word in words
        ):
            raise ValueError("Native readback returned invalid complex storage")
        return [real | imag << 32 for real, imag in zip(words[::2], words[1::2])]
    if dtype == "float16":
        return half_storage.unpack(words, target)
    if dtype == "bfloat16":
        return bfloat_storage.unpack(words, target)
    if dtype == "bool_":
        expected = bool if target == "metal" else int
        if any(type(word) is not expected or word not in (0, 1) for word in words):
            raise ValueError("Native readback returned a non-canonical Boolean")
        return [int(word) for word in words]
    ctype = CTYPES[dtype]
    if any(type(word) is not int or ctype(word).value != word for word in words):
        raise ValueError("Native readback returned invalid scalar storage")
    return list(words)


def reflected_layout_matches(scalar, value, dtype=None):
    """The reflected element type must carry the packet's exact storage.

    A reflected struct element matches only a ``complex64`` packet, and only
    as the packed ``{real, imag}`` float32 pair ``complex64_t`` declares.
    """
    if dtype == COMPLEX or "componentCount" in scalar or "structMembers" in scalar:
        return (
            dtype == COMPLEX
            and all(scalar.get(key) == item for key, item in COMPLEX_LAYOUT.items())
            and value["dtype"] == "float32"
            and value.get("encoding") == FLOAT32_BITS
            and len(value["shape"]) == 1
            and value["shape"][0] % 2 == 0
        )
    storage = scalar.get("storageEncoding")
    if storage is not None:
        return (
            value["dtype"] == "float16"
            and scalar["elementType"] == "uint16"
            and scalar["elementStrideBytes"] == 2
            and storage == {"encoding": FLOAT16_BITS, "logicalElementType": "float16"}
        )
    return scalar["elementType"] == value["dtype"] and scalar[
        "elementStrideBytes"
    ] == PHYSICAL_SIZES.get(value["dtype"])


def supplied_buffers(buffers, count, label):
    """Index the callback's native buffers by name, refusing duplicates."""
    supplied = {}
    for index in range(count):
        buffer = buffers[index]
        if not buffer.name or not buffer.dtype:
            raise ValueError(f"{label} buffer identity is missing")
        name = buffer.name.decode("ascii")
        if name in supplied:
            raise ValueError(f"{label} buffer names must be unique")
        supplied[name] = buffer
    return supplied

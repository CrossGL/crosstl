"""Packed atomic words of general scatters into 8- and 16-bit elements.

MLX's ``mlx_atomic<T>`` for a ``T`` narrower than 32 bits is one ``uint`` word
of ``packing_size<T> = sizeof(uint) / sizeof(T)`` lanes (``atomic.h``,
``uint_or_packed``): an update of element ``offset`` compares and swaps word
``offset / packing_size`` with lane ``offset % packing_size`` replaced. Every
target reflects that output as a struct of one ``uint`` word, whatever the
lane type. The host therefore uploads the destination's logical storage words
little-endian into lanes, fills the unused lanes of the last word with the
lane guard and trails the words with ``GUARD_COUNT`` guard words; a readback
must keep that padding and those guards, and its lanes decode exactly into
the logical words of the element type. Updates keep their ``element_storage``
transport.
"""

import ctypes

from demos.integrations.mlx.portable_host.element_storage import (
    CTYPES,
    GUARD_COUNT,
    GUARDS,
    buffer_words,
)

# Lane width in bytes of every element type MLX packs into atomic words.
LANE_BYTES = {
    "uint8": 1,
    "int8": 1,
    "uint16": 2,
    "int16": 2,
    "float16": 2,
    "bfloat16": 2,
}
GUARD_WORD = 0x6A15BEEF
WORD_MASK = 0xFFFFFFFF
ATOMIC_MEMBERS = [{"name": "val", "offsetBytes": 0, "physicalType": "uint"}]


def packed(dtype):
    """Whether a scatter into ``dtype`` updates lanes of packed atomic words."""
    return dtype in LANE_BYTES


def _lanes(dtype):
    if not packed(dtype):
        raise ValueError(f"Scatter storage type {dtype!r} is not lane-packed")
    width = LANE_BYTES[dtype]
    return width, 4 // width, (1 << 8 * width) - 1


def word_count(dtype, count):
    """Words that hold ``count`` lanes, the last one possibly partial."""
    _, lanes, _ = _lanes(dtype)
    return -(-count // lanes)


def lane_guard(dtype):
    """The unsigned lane value that fills a final word's unused lanes."""
    _, _, mask = _lanes(dtype)
    return GUARDS[dtype] & mask


def pack(dtype, logical):
    """``logical`` storage words as atomic words, padded and guard-trailed."""
    width, lanes, mask = _lanes(dtype)
    ctype = CTYPES[dtype]
    if any(type(word) is not int or ctype(word).value != word for word in logical):
        raise ValueError("Packed scatter storage must hold its logical words")
    padding = [lane_guard(dtype)] * (-len(logical) % lanes)
    unsigned = [word & mask for word in logical] + padding
    words = [
        sum(lane << 8 * width * index for index, lane in enumerate(group))
        for group in zip(*[iter(unsigned)] * lanes)
    ]
    return words + [GUARD_WORD] * GUARD_COUNT


def split(dtype, words, count):
    """Decode ``count`` logical words and the unsigned padding lanes and guards."""
    width, lanes, mask = _lanes(dtype)
    if (
        not isinstance(words, list)
        or len(words) != word_count(dtype, count) + GUARD_COUNT
        or any(type(word) is not int or not 0 <= word <= WORD_MASK for word in words)
    ):
        raise ValueError("Packed scatter storage is not a list of its atomic words")
    body = word_count(dtype, count)
    unsigned = [
        (word >> 8 * width * index) & mask
        for word in words[:body]
        for index in range(lanes)
    ]
    ctype = CTYPES[dtype]
    logical = [ctype(lane).value for lane in unsigned[:count]]
    return logical, unsigned[count:] + words[body:]


def tail(dtype, count):
    """The padding lanes and guard words that ``pack`` appends."""
    _, lanes, _ = _lanes(dtype)
    return [lane_guard(dtype)] * (-count % lanes) + [GUARD_WORD] * GUARD_COUNT


def output_value(dtype, buffer):
    """The runtime value that uploads an initialized scatter destination."""
    words = pack(dtype, buffer_words(buffer, dtype))
    return {"dtype": "uint32", "shape": [len(words), 1], "values": words}


def layout_matches(layout):
    """Every target reflects the packed atomic as one 4-byte ``uint`` member."""
    return (
        layout.get("elementType") == "uint32"
        and layout.get("elementStrideBytes") == 4
        and layout.get("componentCount") == 1
        and layout.get("structMembers") == ATOMIC_MEMBERS
    )


def bind_output(dtype, buffer, layout):
    if not layout_matches(layout):
        raise ValueError("Native scatter atomic storage layout does not match")
    return output_value(dtype, buffer)


def readback(dtype, expected, returned, count):
    """Exact logical words of a native readback; padding and guards must hold."""
    if any(returned.get(key) != expected.get(key) for key in ("dtype", "shape")) or (
        "encoding" in returned
    ):
        raise RuntimeError("Native scatter readback layout does not match")
    try:
        logical, trailing = split(dtype, returned.get("values"), count)
    except ValueError as error:
        raise RuntimeError("Native scatter readback is outside its storage type") from (
            error
        )
    if trailing != tail(dtype, count):
        raise RuntimeError("Native scatter changed the output padding or guard")
    return logical, trailing


def native_words(dtype, logical):
    """``logical`` as the host's exact storage array for ``dtype``."""
    return (CTYPES[dtype] * len(logical))(*logical)


def element_size(dtype):
    return ctypes.sizeof(CTYPES[dtype])

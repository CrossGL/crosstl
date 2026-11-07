"""Exactness of the vectorized element-storage readback and buffer checks."""

import ctypes
import itertools
import random
from types import SimpleNamespace

import pytest

from demos.integrations.mlx.portable_host import element_storage
from demos.integrations.mlx.portable_host.element_storage import (
    CTYPES,
    buffer_words,
    native_bytes,
    scalar_words_fit,
    unpack,
    word_range,
)

INTEGER_DTYPES = tuple(
    dtype for dtype in element_storage.DTYPES if dtype not in element_storage.FLOATING
)
WORD_CTYPES = tuple(dict.fromkeys(CTYPES.values()))


def reference_fit(words, ctype):
    """The element-wise definition the vectorized check replaces."""
    return all(type(word) is int and ctype(word).value == word for word in words)


class Wide(int):
    """An ``int`` subclass, which exact storage refuses."""


def candidate_lists(ctype):
    low, high = word_range(ctype)
    edges = (low, low + 1, -1, 0, 1, high - 1, high, low - 1, high + 1)
    rng = random.Random(f"{ctype.__name__}")
    yield []
    for edge in edges:
        yield [edge]
        yield [0, edge, 1]
    for left, right in itertools.product(edges, repeat=2):
        yield [left, right]
    for _ in range(64):
        yield [rng.randint(low, high) for _ in range(rng.randint(1, 40))]
    for bad in (True, False, 1.0, None, "1", Wide(1), 2**64, -(2**63) - 1):
        yield [0, bad]
        yield [bad]


@pytest.mark.parametrize("ctype", WORD_CTYPES, ids=lambda item: item.__name__)
def test_scalar_words_fit_agrees_with_the_element_wise_definition(ctype):
    for words in candidate_lists(ctype):
        assert scalar_words_fit(words, ctype) is reference_fit(words, ctype), words


@pytest.mark.parametrize("ctype", WORD_CTYPES, ids=lambda item: item.__name__)
def test_word_range_is_the_unchanged_ctype_range(ctype):
    low, high = word_range(ctype)
    assert ctype(low).value == low and ctype(high).value == high
    assert ctype(low - 1).value != low - 1 and ctype(high + 1).value != high + 1


@pytest.mark.parametrize("target", ("metal", "directx", "opengl"))
@pytest.mark.parametrize("dtype", tuple(d for d in INTEGER_DTYPES if d != "bool_"))
def test_unpack_refuses_exactly_the_words_its_ctype_changes(dtype, target):
    low, high = word_range(CTYPES[dtype])
    assert unpack(dtype, target, [low, 0, high]) == [low, 0, high]
    for bad in ([low - 1], [high + 1], [True], [1.0], [Wide(3)], [0, None]):
        with pytest.raises(ValueError, match="invalid scalar storage"):
            unpack(dtype, target, bad)


def native(ctype, words):
    storage = (ctype * len(words))(*words)
    return SimpleNamespace(data=ctypes.addressof(storage), count=len(words)), storage


@pytest.mark.parametrize("dtype", element_storage.DTYPES)
def test_buffer_words_reads_exact_ints_of_every_storage_type(dtype):
    ctype = CTYPES[dtype]
    low, high = word_range(ctype)
    words = [0, 1] if dtype == "bool_" else [low, 0, 1, high]
    buffer, storage = native(ctype, words)
    read = buffer_words(buffer, dtype)
    assert read == words and all(type(word) is int for word in read)
    empty, empty_storage = native(ctype, [])
    assert buffer_words(empty, dtype) == []
    assert bytes(storage) and not bytes(empty_storage)


def test_buffer_words_refuses_non_canonical_booleans():
    buffer, storage = native(ctypes.c_uint8, [0, 1, 2])
    with pytest.raises(ValueError, match="canonical zero or one"):
        buffer_words(buffer, "bool_")
    assert bytes(storage) == b"\0\1\2"


def constructor_bytes(ctype, count, words):
    try:
        return bytes((ctype * count)(*words))
    except Exception as error:  # noqa: BLE001 - compared by type below
        return type(error)


@pytest.mark.parametrize("ctype", WORD_CTYPES, ids=lambda item: item.__name__)
def test_native_bytes_is_the_ctypes_constructor(ctype):
    low, high = word_range(ctype)
    cases = [
        (0, []),
        (4, [low, 0, 1, high]),
        (3, [True, False, 7]),
        (2, [low - 1, high + 1]),
        (2, [2**70, -(2**70)]),
        (5, [1, 2, 3]),
        (2, [1, 2, 3]),
        (2, [1.0, 2]),
        (1, ["1"]),
        (2, (1, 2)),
        (2, [Wide(5), 6]),
    ]
    rng = random.Random(ctype.__name__)
    cases += [(n, [rng.randint(low, high) for _ in range(n)]) for n in (1, 31, 4097)]
    for count, words in cases:
        expected = constructor_bytes(ctype, count, words)
        if isinstance(expected, bytes):
            assert native_bytes(ctype, count, words) == expected, (count, words)
        else:
            with pytest.raises(expected):
                native_bytes(ctype, count, words)

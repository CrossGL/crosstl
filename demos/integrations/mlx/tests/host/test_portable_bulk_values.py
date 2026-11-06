"""Bulk host value paths agree with the element-wise definitions they replace."""

import ctypes
import itertools
import math
import random

import pytest

from demos.integrations.mlx.portable_host import (
    copy_layout,
    element_storage,
    quantization_dispatch,
    random_layout,
    runtime,
    trace_records,
)


class Number(float):
    pass


def element_indices(metadata):
    for coordinates in itertools.product(*(range(size) for size in metadata["shape"])):
        yield metadata["destinationOffset"] + sum(
            coordinate * stride
            for coordinate, stride in zip(coordinates, metadata["destinationStrides"])
        )


def test_destination_indices_match_the_coordinate_definition():
    rng = random.Random(7)
    for _ in range(2000):
        rank = rng.randint(0, 5)
        metadata = {
            "shape": [rng.randint(0, 4) for _ in range(rank)],
            # Strides may be shorter or longer than the shape, and zero or negative.
            "destinationStrides": [
                rng.randint(-6, 6) for _ in range(rng.randint(0, 6))
            ],
            "destinationOffset": rng.randint(-4, 40),
        }
        assert list(copy_layout.destination_indices(metadata)) == list(
            element_indices(metadata)
        ), metadata


def test_wire_values_match_element_wire_values():
    cases = [
        [],
        [1, -2, 3],
        [True, False],
        [0.5, -1.25],
        [0.5, math.nan, math.inf, -math.inf],
        [Number(math.inf), 1.0],
        [1, 2.5],
    ]
    for values in cases:
        assert runtime.wire_values(values) == [
            runtime.wire_value(value) for value in values
        ]


def test_distinct_by_strides_proves_only_one_to_one_layouts():
    rng = random.Random(11)
    proved = 0
    for _ in range(3000):
        rank = rng.randint(1, 4)
        shape = [rng.randint(1, 4) for _ in range(rank)]
        strides = [rng.choice([-1, 1]) * rng.randint(0, 20) for _ in range(rank)]
        metadata = {
            "shape": shape,
            "destinationStrides": strides,
            "destinationOffset": 0,
        }
        indices = list(copy_layout.destination_indices(metadata))
        if copy_layout.distinct_by_strides(shape, strides):
            proved += 1
            assert len(set(indices)) == len(indices), metadata
    assert proved > 300
    # Row-major and transposed layouts, and axes of size one with any stride.
    assert copy_layout.distinct_by_strides([4, 8], [8, 1])
    assert copy_layout.distinct_by_strides([4, 8], [1, 4])
    assert copy_layout.distinct_by_strides([1, 3, 1], [0, -2, 0])
    # Broadcast and overlapping layouts decide nothing.
    assert not copy_layout.distinct_by_strides([2, 3], [0, 1])
    assert not copy_layout.distinct_by_strides([3, 3], [2, 1])


def test_ctypes_buffers_read_as_their_items():
    for ctype, items in (
        (ctypes.c_float, [0.5, -1.25, math.inf, 3.0e38]),
        (ctypes.c_double, [1e300, -0.0]),
        (ctypes.c_bool, [True, False]),
        (ctypes.c_uint8, [0, 255]),
        (ctypes.c_int16, [-32768, 7]),
        (ctypes.c_uint32, [0, 2**32 - 1]),
        (ctypes.c_int64, [-(2**63), 2**63 - 1]),
        (ctypes.c_uint64, [2**64 - 1]),
    ):
        array = (ctype * len(items))(*items)
        values = element_storage.array_values(array)
        assert values == list(array)
        assert [type(value) for value in values] == [type(v) for v in list(array)]
    assert element_storage.array_values((ctypes.c_float * 0)()) == []


def test_wire_values_of_ctypes_buffers_match_their_items():
    floats = (ctypes.c_float * 4)(0.5, math.nan, math.inf, -math.inf)
    assert runtime.wire_values(floats) == [0.5, "nan", "+infinity", "-infinity"]
    finite = (ctypes.c_float * 2)(0.5, -2.0)
    assert runtime.wire_values(finite) == [0.5, -2.0]
    # A sum past the float64 range still reads every finite value unchanged.
    large = (ctypes.c_double * 2)(1e308, 1e308)
    assert runtime.wire_values(large) == [1e308, 1e308]
    for ctype, items in ((ctypes.c_uint16, [0, 65535]), (ctypes.c_int32, [-3, 4])):
        array = (ctype * len(items))(*items)
        assert runtime.wire_values(array) == [
            runtime.wire_value(value) for value in array
        ]


@pytest.mark.parametrize("dtype", ["bool", "uint32"])
def test_boolean_values_accept_only_canonical_storage(dtype):
    canonical = [True, False, True] if dtype == "bool" else [1, 0, 1]
    converted = runtime.boolean_values(canonical, dtype)
    assert converted == canonical and converted is not canonical
    assert [type(value) for value in converted] == [type(canonical[0])] * 3
    other = [1, 0] if dtype == "bool" else [True, False]
    for invalid in ([2], [-1], other, [1.0]):
        with pytest.raises(ValueError, match="canonical"):
            runtime.boolean_values(invalid, dtype)


def test_random_unpack_matches_masked_signed_bytes():
    layout = random_layout.RandomOutputLayout(3, 5)
    values = [((index * 37) % 256) - 128 for index in range(layout.native_byte_count)]
    raw = bytes(value & 255 for value in values)
    expected = b"".join(
        raw[key * layout.native_bytes_per_key :][: layout.bytes_per_key]
        for key in range(layout.key_count)
    )
    assert layout.unpack(values) == expected
    for invalid in ([128], [-129], [True], [1.0]):
        bad = list(values)
        bad[-1:] = invalid
        with pytest.raises(ValueError, match="signed byte storage"):
            layout.unpack(bad)


def test_quantization_words_keep_their_range_checks():
    assert quantization_dispatch.encode([0, 255], "uint8", "uint8")["values"] == [
        0,
        255,
    ]
    for invalid in ([256], [-1], [True], [1.0]):
        with pytest.raises(ValueError, match="storage word"):
            quantization_dispatch.encode(invalid, "uint8", "uint8")


def test_trace_compaction_still_accepts_scalar_subclasses():
    values = [Number(0.5)] * (trace_records.COMPACT_TRACE_LIMIT + 1)
    compacted = trace_records.compact_trace_values({"values": values})
    assert compacted["values"]["compactedValues"]["length"] == len(values)

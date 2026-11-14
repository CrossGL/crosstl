"""Bulk host value paths agree with the element-wise definitions they replace."""

import itertools
import math
import random

import pytest

from demos.integrations.mlx.portable_host import (
    copy_layout,
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

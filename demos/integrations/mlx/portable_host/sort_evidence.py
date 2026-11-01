"""Audit lazily translated MLX merge-sort dispatches.

Every dispatch must name the package its sort cache derives again from the
pinned source at the same threadgroup width, and its retained native modules
and request must match the event. The metadata and launch are validated again
against ``gpu_merge_sort``. MLX sorts stably under ``LessThan`` (NaN after
every ordered value, complex values by real then imaginary part), so every
stage's result is a function of its inputs: a record that keeps its values
must have uploaded exactly those words, and its outputs are recomputed and
compared exactly. A record compacted past ``RETAINED_LIMIT`` values keeps the
lengths and digests of its long lists; its extents, guard uploads and
metadata are checked, and the unchanged upstream tests compare the sorted
result itself.
"""

import functools
import re
import struct
from pathlib import Path

from demos.integrations.mlx.portable_host import sort_layout
from demos.integrations.mlx.portable_host.element_storage import guard_words
from demos.integrations.mlx.portable_host.gather_evidence import (
    audit_native_execution,
    require,
)
from demos.integrations.mlx.portable_host.lazy_package_evidence import (
    require_unchanged,
)
from demos.integrations.mlx.portable_host.sort_dispatch import (
    ELEMENTS,
    RETAINED_LIMIT,
    inputs_of,
    output_hash,
    uploads,
)
from demos.integrations.mlx.portable_host.trace_records import compact_trace_values

__all__ = ("audit_event", "expected_outputs")

GEOMETRY = ("workgroupCount", "workgroupSize")
FLOAT32_MASK = 0xFFFFFFFF
DIGEST = re.compile(r"[0-9a-f]{64}")


def _float32(word):
    return struct.unpack("<f", struct.pack("<I", word & FLOAT32_MASK))[0]


def decode(dtype, word):
    """The value ``LessThan`` compares for one storage word."""
    if dtype == "float32":
        return _float32(word)
    if dtype == "float16":
        return struct.unpack("<e", struct.pack("<H", word))[0]
    if dtype == "bfloat16":
        return _float32(word << 16)
    if dtype == "complex64":
        return (_float32(word), _float32(word >> 32))
    return word


def _is_nan(value):
    if isinstance(value, tuple):
        return value[0] != value[0] or value[1] != value[1]
    return value != value


def less_than(left, right):
    """MLX ``LessThan``: NaN sorts after every ordered value."""
    left_nan, right_nan = _is_nan(left), _is_nan(right)
    if left_nan or right_nan:
        return not left_nan and right_nan
    return left < right


def _compare(left, right):
    return -1 if less_than(left[0], right[0]) else int(less_than(right[0], left[0]))


_KEY = functools.cmp_to_key(_compare)


def stable_order(keys):
    """The positions of ``keys`` in stable ``LessThan`` order."""
    return [item[1] for item in sorted(zip(keys, range(len(keys))), key=_KEY)]


def merge_partition(a_keys, b_keys, diagonal):
    """``merge_partition``: how many of the first ``diagonal`` merged come from A."""
    start, end = max(0, diagonal - len(b_keys)), min(diagonal, len(a_keys))
    while start < end:
        middle = start + (end - start) // 2
        if less_than(b_keys[diagonal - 1 - middle], a_keys[middle]):
            end = middle
        else:
            start = middle + 1
    return end


def elem_to_loc(element, shape, strides):
    location = 0
    for extent, stride in zip(reversed(shape), reversed(strides)):
        location += element % extent * stride
        element //= extent
    return location


def _row_offsets(metadata, rows, strides):
    if metadata["nc_dim"] == 0:
        return [0]
    return [elem_to_loc(row, metadata["nc_shape"], strides) for row in range(rows)]


def _block(entry, metadata, values, counts, rows):
    _, layout, argsort, dtype, _ = sort_layout.signature(entry)
    size = metadata["size_sorted_axis"]
    inp = values["inp"]
    out_type = sort_layout.operand_types(entry)["out"]
    out = guard_words(out_type, counts["out"])
    if layout == "c":
        in_rows = [row * metadata["in_stride_segment_axis"] for row in range(rows)]
        out_rows = [row * metadata["out_stride_segment_axis"] for row in range(rows)]
    else:
        in_rows = _row_offsets(metadata, rows, metadata["in_nc_strides"])
        out_rows = _row_offsets(metadata, rows, metadata["out_nc_strides"])
    in_stride = metadata["in_stride_sorted_axis"]
    out_stride = metadata["out_stride_sorted_axis"]
    for in_row, out_row in zip(in_rows, out_rows):
        words = [inp[in_row + i * in_stride] for i in range(size)]
        order = stable_order([decode(dtype, word) for word in words])
        for position, index in enumerate(order):
            out[out_row + position * out_stride] = index if argsort else words[index]
    return {"out": out}


def _tile_sort(entry, metadata, values, rows):
    _, _, _, dtype, bn = sort_layout.signature(entry)
    size, stride = metadata["size_sorted_axis"], metadata["stride_sorted_axis"]
    tile = bn * sort_layout.N_PER_THREAD
    inp = values["inp"]
    out_vals, out_idxs = [], []
    for row in _row_offsets(metadata, rows, metadata["nc_strides"]):
        for start in range(0, size, tile):
            indices = range(start, min(size, start + tile))
            words = [inp[row + index * stride] for index in indices]
            order = stable_order([decode(dtype, word) for word in words])
            out_vals.extend(words[position] for position in order)
            out_idxs.extend(start + position for position in order)
    return {"out_vals": out_vals, "out_idxs": out_idxs}


def _runs(size, tile, merge_tiles, group):
    span = tile * merge_tiles
    start = span * group
    a_start, a_end = min(size, start), min(size, start + span // 2)
    return a_start, a_end, a_end, min(size, a_end + span // 2)


def _partition(entry, metadata, values, counts, rows):
    _, _, _, dtype, bn = sort_layout.signature(entry)
    size, merge_tiles = metadata["size_sorted_axis"], metadata["merge_tiles"]
    n_blocks = metadata["n_blocks"]
    tile = bn * sort_layout.N_PER_THREAD
    width = sort_layout.width(entry, n_blocks)
    out = guard_words(sort_layout.INDEX_TYPE, counts["block_partitions"])
    for row in range(rows):
        keys = [decode(dtype, word) for word in values["dev_vals"][row * size :][:size]]
        for index in range(n_blocks + 1):
            group, lane = divmod(index, merge_tiles)
            a_start, a_end, b_start, b_end = _runs(size, tile, merge_tiles, group)
            diagonal = min(b_end - a_start, tile * lane)
            split = merge_partition(keys[a_start:a_end], keys[b_start:b_end], diagonal)
            out[row * width + index] = a_start + split
    return {"block_partitions": out}


def _merge(entry, metadata, values, rows):
    _, _, _, dtype, bn = sort_layout.signature(entry)
    size, merge_tiles = metadata["size_sorted_axis"], metadata["merge_tiles"]
    tile = bn * sort_layout.N_PER_THREAD
    out_vals, out_idxs = [], []
    for row in range(rows):
        vals = values["dev_vals_in"][row * size :][:size]
        idxs = values["dev_idxs_in"][row * size :][:size]
        keys = [decode(dtype, word) for word in vals]
        for group in range(-(-size // (tile * merge_tiles))):
            a_start, a_end, b_start, b_end = _runs(size, tile, merge_tiles, group)
            for start, end in ((a_start, a_end), (b_start, b_end)):
                require(
                    all(
                        not less_than(keys[i + 1], keys[i])
                        for i in range(start, end - 1)
                    ),
                    "Sort merge input runs are not sorted",
                )
            # A stable sort of A then B is the merge that takes A on ties.
            order = stable_order(keys[a_start:b_end])
            out_vals.extend(vals[a_start + position] for position in order)
            out_idxs.extend(idxs[a_start + position] for position in order)
    return {"dev_vals_out": out_vals, "dev_idxs_out": out_idxs}


def expected_outputs(entry, metadata, values, counts, logical_size):
    """Every output word one stage writes, recomputed from its inputs."""
    stage = sort_layout.signature(entry)[0]
    rows = logical_size // metadata["size_sorted_axis"]
    if stage == "block":
        return _block(entry, metadata, values, counts, rows)
    if stage == "sort":
        return _tile_sort(entry, metadata, values, rows)
    if stage == "partition":
        return _partition(entry, metadata, values, counts, rows)
    return _merge(entry, metadata, values, rows)


def _extent(value):
    if isinstance(value, list):
        return len(value)
    compacted = value.get("compactedValues") if isinstance(value, dict) else None
    require(
        isinstance(compacted, dict)
        and type(compacted.get("length")) is int
        and DIGEST.fullmatch(str(compacted.get("sha256"))),
        "Sort trace value list is neither retained nor compacted",
    )
    return compacted["length"]


def audit_event(event, cache):
    """Check one sort dispatch; returns whether its outputs were recomputed."""
    entry, target, threads = event["entry"], event["target"], event["threads"]
    execution = {key: event[key] for key in GEOMETRY}
    require("threadGridSize" not in event, "Sort launch names an exact thread grid")
    values, outputs = event["operandValues"], event["outputValues"]
    written = sort_layout.outputs(entry)
    require(
        isinstance(values, dict)
        and set(values) == set(inputs_of(entry))
        and isinstance(outputs, dict)
        and set(outputs) == set(written),
        "Sort trace does not name its element buffers",
    )
    counts = {name: _extent(value) for name, value in {**values, **outputs}.items()}
    metadata = event["sortMetadata"]
    sort_layout.validate_metadata(entry, metadata, counts, threads, execution)
    require(event["dispatchVersion"] == 3, "Sort dispatch version differs")
    retained = all(
        isinstance(value, list) for value in [*values.values(), *outputs.values()]
    )
    require(
        retained == (threads <= RETAINED_LIMIT),
        "Sort trace retention does not match its extent",
    )
    output_counts = {name: counts[name] for name in written}
    if retained:
        expected = uploads(entry, target, values, output_counts, metadata)
    else:
        placeholders = {name: [0] * counts[name] for name in inputs_of(entry)}
        expected = compact_trace_values(
            uploads(entry, target, placeholders, output_counts, metadata)
        )
    bound = list(event["inputs"].values())
    require(len(bound) == len(expected), "Sort uploads differ from its buffers")
    for name, packet in expected.items():
        if retained or name not in ELEMENTS or name in written:
            require(packet in bound, f"Sort upload of {name} differs from its values")
        else:
            require(
                any(
                    {k: v for k, v in item.items() if k != "values"}
                    == {k: v for k, v in packet.items() if k != "values"}
                    and _extent(item["values"]) == _extent(packet["values"])
                    for item in bound
                ),
                f"Sort upload of {name} differs from its extent",
            )
    if retained:
        require(
            output_hash(entry, outputs) == event["outputHash"],
            "Sort host output hash changed",
        )
        model = expected_outputs(entry, metadata, values, output_counts, threads)
        require(model == outputs, "Sort outputs differ from the stable LessThan order")
    else:
        require(
            DIGEST.fullmatch(str(event["outputHash"])), "Sort output hash is missing"
        )
    descriptor, directory = cache.get(
        sort_layout.variant(entry, execution["workgroupSize"][0])
    )
    require_unchanged(descriptor, directory)
    require(
        event["artifact"] == descriptor["artifact"]
        and Path(event["packageRoot"]).resolve() == Path(directory).resolve(),
        "Sort dispatch does not name its cached package",
    )
    request = event["details"]["request"]
    require(
        request["target"] == target
        and request["entryPoint"] == entry
        and all(request["dispatch"][key] == event[key] for key in GEOMETRY),
        "Sort native request changed",
    )
    audit_native_execution(event)
    return retained

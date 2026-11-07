"""Wide (``v2``) vectors: exact 2-D launches of MLX's own wide kernels.

``backend.cpp`` runs a vector of more than 65535 elements through the pinned
``unary_v2``, ``binary_vv2``/``sv2``/``vs2`` and ``ternary_v2`` kernels, over
the grid ``wide_vectors`` derives from its int64 size. These tests check the
split, the dispatch contract on every target through word lists and packed
storage, the audit and the host source.
"""

import ctypes
import json
import re
from types import SimpleNamespace

import pytest

from demos.integrations.mlx.portable_host import (
    element_storage,
    elementwise_dispatch,
    elementwise_evidence,
    elementwise_packages,
    prepare,
    runtime,
    trace_records,
    wide_vectors,
)
from demos.integrations.mlx.tests.host.test_portable_elementwise import (
    TARGETS,
    bit_words,
    native_buffers,
    run,
)

GROUPS = wide_vectors.MAX_GROUPS


@pytest.mark.parametrize(
    ("elements", "per_thread", "expected"),
    [
        (65536, 2, (32768, 1)),
        (131077, 2, (65535, 2)),
        (131070, 1, (65535, 2)),
        (65535, 1, (65535, 1)),
        (GROUPS * 16 * 8, 8, (65535, 16)),
        (GROUPS * 16 * 8 - 7, 8, (65535, 16)),
        (3, 4, (1, 1)),
    ],
)
def test_grid_covers_every_element_once(elements, per_thread, expected):
    columns, rows = wide_vectors.grid(elements, per_thread)
    assert (columns, rows) == expected
    # The kernel's offset N * (x + columns * y): every element is read by
    # exactly one invocation, and an invocation past the end reads none.
    threads = columns * rows
    assert (
        threads * per_thread
        >= elements
        > (threads - columns) * per_thread - (per_thread if rows == 1 else 0)
    )
    assert -(-elements // per_thread) <= threads


@pytest.mark.parametrize(
    ("elements", "per_thread", "message"),
    [
        (65536, 1, "whole rows"),
        (0, 2, "exceeds"),
        (GROUPS * 16 * 2 + 1, 2, "exceeds"),
        (True, 2, "exceeds"),
        (8, 3, "one, two, four or eight"),
    ],
)
def test_grid_refuses_launches_outside_the_contract(elements, per_thread, message):
    with pytest.raises(ValueError, match=message):
        wide_vectors.grid(elements, per_thread)


@pytest.mark.parametrize("per_thread", [1, 2, 4, 8])
@pytest.mark.parametrize(
    "count", [1, 65535, 65536, 131070, 131077, 2_000_003, 8 * GROUPS * 16 + 5]
)
def test_launches_split_a_vector_exactly(count, per_thread):
    launches = wide_vectors.launches(count, per_thread)
    assert launches[0][0] == 0
    assert all(
        first + size == following
        for (first, size, _), (following, _, _) in zip(launches, launches[1:])
    )
    assert sum(size for _, size, _ in launches) == count
    for index, (_, size, wide) in enumerate(launches):
        if wide:
            columns, rows = wide_vectors.grid(size, per_thread)
            assert rows <= wide_vectors.MAX_ROWS
            assert per_thread != 1 or size % GROUPS == 0
        else:
            # A narrow launch is the last one, of at most 65535 elements.
            assert size <= GROUPS and index == len(launches) - 1
    if count <= GROUPS:
        assert launches == [(0, count, False)]
    # Without the wide kernels, every launch is a narrow one.
    narrow = wide_vectors.launches(count, per_thread, wide=False)
    assert [size for _, size, _ in narrow] == [
        min(GROUPS, count - first) for first in range(0, count, GROUPS)
    ]
    assert not any(wide for *_, wide in narrow)


def test_launches_cut_the_rope_test_by_two_orders_of_magnitude():
    # test_fast.test_rope_large_input's 8 x 32 x 8192 x 32 float32 vectors.
    count = 8 * 32 * 8192 * 32
    assert len(wide_vectors.launches(count, 2, wide=False)) == 1025
    assert len(wide_vectors.launches(count, 2)) == 33


def test_next_elements_requires_a_wide_remainder():
    with pytest.raises(ValueError, match="more than 65535"):
        wide_vectors.next_elements(GROUPS, 2)
    assert wide_vectors.next_elements(GROUPS + 1, 2) == GROUPS + 1
    assert wide_vectors.next_elements(GROUPS * 3 - 1, 1) == GROUPS * 2


WIDE_CASES = (
    ("v2_Negativeint8int8", {"in": [-128, -1, 0, 1, 127]}, 65536 + 3),
    ("vv2_Addint64", {"a": [1 << 62, -3, 9], "b": [5, -(1 << 62), 0]}, 2 * GROUPS),
    ("sv2_Subtractint16", {"a": [-7], "b": [1, 2, -32768, 32767]}, 65536 + 1),
    ("vs2_LeftShiftuint32", {"a": [1, 0xFFFFFFFF, 3], "b": [31]}, 65536 + 5),
    (
        "v2_Selectint16",
        {"a": [1, 0, 1], "b": [1, -2, 3], "c": [4, 5, -6]},
        65536 + 2,
    ),
    ("vv2_Lessuint8", {"a": [1, 9, 255, 0], "b": [2, 3, 255, 1]}, 65536 + 7),
    ("v2_Absfloat32float32", {"in": [0x3F800000, 0x7FC12345, 0x80000000]}, 65537),
)


def operands_of(entry, operands, length):
    kernel = elementwise_packages.signature(entry)[0]
    scalar = elementwise_dispatch.SCALARS.get(kernel)
    return {
        name: values if name == scalar else (values * length)[:length]
        for name, values in operands.items()
    }


@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.parametrize("case", WIDE_CASES, ids=lambda case: case[0])
def test_wide_vectors_dispatch_their_exact_grid(monkeypatch, tmp_path, target, case):
    entry, operands, length = case
    operands = operands_of(entry, operands, length)
    floating = not elementwise_evidence.recomputable(entry)
    words, event, host = run(
        monkeypatch,
        tmp_path,
        target,
        entry,
        operands,
        compute=bit_words if floating else None,
    )
    per_thread = elementwise_packages.work_per_thread(entry)
    columns, rows = wide_vectors.grid(length, per_thread)
    assert host.dispatch_count == 1 and len(words) == length
    assert event["workgroupCount"] == [columns, rows, 1]
    assert event["threads"] == columns * rows
    assert event["elementwiseMetadata"]["elements"] == length
    assert event["elementwiseMetadata"]["workPerThread"] == per_thread
    size = next(
        value for name, value in event["inputs"].items() if name.startswith("size")
    )
    assert size["dtype"] == "int64" and size["values"] == [length]
    assert elementwise_evidence.audit_event(event, native=False) == length
    if not floating:
        kernel, operation, source, destination = elementwise_packages.signature(entry)
        values = elementwise_dispatch.broadcast(kernel, operands, length)
        assert words == elementwise_evidence.expected_words(
            kernel, operation, source, destination, values
        )


@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.parametrize("case", WIDE_CASES, ids=lambda case: case[0])
def test_wide_packed_storage_writes_the_word_list_record(
    monkeypatch, tmp_path, target, case
):
    entry, operands, length = case
    operands = operands_of(entry, operands, length)
    floating = not elementwise_evidence.recomputable(entry)
    packed = elementwise_dispatch.packed_transport(
        SimpleNamespace(target=target, trace_values=False), entry
    )
    results = {}
    for transport in ("packed", "words"):
        if transport == "words":
            monkeypatch.setattr(
                elementwise_dispatch, "packed_transport", lambda host, entry: False
            )
        work = tmp_path / transport
        work.mkdir()
        results[transport] = run(
            monkeypatch,
            work,
            target,
            entry,
            operands,
            compute=bit_words if floating else None,
            trace_values=False,
        )
        monkeypatch.undo()
    assert results["packed"][0] == results["words"][0]
    assert (tmp_path / "packed/dispatch.jsonl").read_bytes() == (
        tmp_path / "words/dispatch.jsonl"
    ).read_bytes()
    event = results["packed"][1]
    assert event["elementwiseMetadata"]["elements"] == length
    assert all(
        "compactedValues" in event[key] for key in ("initialValues", "outputValues")
    )
    assert length > trace_records.COMPACT_TRACE_LIMIT
    assert packed == all(
        element_storage.packed_layout(dtype, target) is not None
        for dtype in elementwise_packages.operand_types(entry).values()
    )


def supplied(entry, operands, *, size=None):
    array, memory, threads = native_buffers(entry, operands)
    buffers = {name: array[index] for index, name in enumerate(memory)}
    if size is not None:
        memory["size"] = (element_storage.CTYPES[size[0]] * 1)(size[1])
        buffers["size"] = runtime.Buffer(
            b"size", size[0].encode(), ctypes.addressof(memory["size"]), 1, 0
        )
    return buffers, memory, threads


def test_wide_launch_rejections():
    validate = elementwise_dispatch.validate
    entry = "vv2_Addint32"
    length = 65536 + 9
    operands = operands_of(entry, {"a": [1, 2], "b": [3, 4]}, length)
    buffers, memory, _ = supplied(entry, operands)
    launch = elementwise_dispatch.wide_launch(entry, length)
    assert launch == {"workgroupCount": [32773, 1, 1], "workgroupSize": [1, 1, 1]}
    metadata = validate(entry, buffers, 32773, launch)
    assert metadata["elements"] == length and metadata["workPerThread"] == 2
    with pytest.raises(ValueError, match="does not match its vector"):
        validate(entry, buffers, 32773, {**launch, "workgroupCount": [32773, 2, 1]})
    with pytest.raises(ValueError, match="does not match the launch"):
        validate(entry, buffers, 32772, launch)
    with pytest.raises(ValueError, match="does not match its vector"):
        validate(
            entry,
            buffers,
            32773,
            {"workgroupCount": [32773, 1, 1], "workgroupSize": [2, 1, 1]},
        )
    # A wide launch passes an int64 size; a narrow one a uint32 size.
    # Each case keeps its native memory alive while validate reads it.
    narrow, kept, _ = supplied(entry, operands, size=("uint32", length))
    with pytest.raises(ValueError, match="type, extent"):
        validate(entry, narrow, 32773, launch)
    small = operands_of("vv_Addint32", {"a": [1], "b": [2]}, 9)
    wide_size, kept_small, _ = supplied("vv_Addint32", small, size=("int64", 9))
    with pytest.raises(ValueError, match="type, extent"):
        validate(
            "vv_Addint32",
            wide_size,
            9,
            {"workgroupCount": [9, 1, 1], "workgroupSize": [1, 1, 1]},
        )
    # The size must equal every vector's extent.
    shorter, kept_short, _ = supplied(entry, operands, size=("int64", length - 1))
    with pytest.raises(ValueError, match="type, extent"):
        validate(
            entry, shorter, 32772, elementwise_dispatch.wide_launch(entry, length - 1)
        )
    # One-element kernels launch whole rows only.
    whole = operands_of("vv2_Addint64", {"a": [1], "b": [2]}, 65536)
    rows, kept_rows, _ = supplied("vv2_Addint64", whole)
    with pytest.raises(ValueError, match="whole rows"):
        validate(
            "vv2_Addint64",
            rows,
            65536,
            {"workgroupCount": [65535, 2, 1], "workgroupSize": [1, 1, 1]},
        )
    # A narrow launch still covers at most 65535 elements.
    with pytest.raises(ValueError, match="host contract"):
        validate(
            "vv_Addint32",
            buffers,
            65536,
            {"workgroupCount": [65535, 1, 1], "workgroupSize": [1, 1, 1]},
        )
    assert all(
        kept_memory for kept_memory in (memory, kept, kept_small, kept_short, kept_rows)
    )


@pytest.mark.parametrize(
    "fault", ["elements", "per-thread", "grid", "threads", "value"]
)
def test_wide_audit_requires_its_exact_launch(monkeypatch, tmp_path, fault):
    entry, operands, length = WIDE_CASES[0]
    _, event, _ = run(
        monkeypatch, tmp_path, "metal", entry, operands_of(entry, operands, length)
    )
    assert elementwise_evidence.audit_event(event, native=False) == length
    tampered = json.loads(json.dumps(event))
    if fault == "elements":
        tampered["elementwiseMetadata"]["elements"] += 1
    elif fault == "per-thread":
        tampered["elementwiseMetadata"]["workPerThread"] = 1
    elif fault == "grid":
        tampered["workgroupCount"] = [8194, 1, 1]
    elif fault == "threads":
        tampered["threads"] -= 1
    else:
        tampered["outputValues"][65538] ^= 1
    with pytest.raises(ValueError, match="Elementwise"):
        elementwise_evidence.audit_event(tampered, native=False)


def test_backend_runs_long_vectors_through_the_wide_kernels():
    source = (prepare.HERE / "backend.cpp").read_text(encoding="utf-8")
    assert f"constexpr uint64_t wide_max_groups = {wide_vectors.MAX_GROUPS};" in source
    assert f"constexpr uint64_t wide_max_rows = {wide_vectors.MAX_ROWS};" in source
    helper = source.split("std::vector<VectorLaunch> vector_launches(", 1)[1].split(
        "\n}\n", 1
    )[0]
    assert "wide && remaining > wide_max_groups" in helper
    assert "launch.elements -= launch.elements % wide_max_groups" in helper
    sites = {
        "void dispatch_unary(": (
            'std::string("v2_") + operation + dtype + out_type',
            "8 / in.itemsize()",
            "vector_launches(in.data_size(), per_thread, entry_available(wide))",
        ),
        "void dispatch_binary(": (
            'std::string(sv ? "sv2_" : vs ? "vs2_" : "vv2_") + operation + dtype',
            "8 / a.itemsize()",
            "vector_launches(count, per_thread, entry_available(wide))",
        ),
        "void Select::eval_gpu(": (
            'std::string("v2_Select") + dtype',
            "8 / out.itemsize()",
            "vector_launches(out.size(), per_thread, entry_available(wide))",
        ),
    }
    for start, texts in sites.items():
        body = source.split(start, 1)[1].split("\n}\n", 1)[0]
        for text in (*texts, "vector_size_buffer(launch, size, wide_size)"):
            assert text in body, (start, text)
        assert not re.search(r"first \+= 65535|std::min<uint64_t>\(65535", body)
    # The narrow size is uint32 and the wide one int64, as the kernels declare.
    assert '{"size", "int64", &wide, 1, 0}' in source
    assert '{"size", "uint32", &narrow, 1, 0}' in source

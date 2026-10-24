"""Wide small-column reductions mirror the pinned instantiations, check each
launch against strided_reduce_small's plan and recompute integer columns."""

import ctypes
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from demos.integrations.mlx.portable_host import (
    column_small_layout,
    column_small_packages,
    element_storage,
    lazy_package_evidence,
    runtime,
    wide_column_dispatch,
    wide_column_evidence,
)

ROOT = Path(__file__).resolve().parents[5]
CONTRACTS = ROOT / "demos/integrations/mlx/contracts"
TARGETS = ("metal", "directx", "opengl")
INT64_MAX, UINT64_MAX = 2**63 - 1, 2**64 - 1
# The (3, 3) diagonal view mx.trace reduces over a (3, 3, 3) array's first two
# axes: strides (1, 3), so its last axis is a column of three rows.
COLUMNS = {
    "reduction_size": [3],
    "reduction_stride": [3],
    "shape": [0],
    "strides": [0],
    "ndim": [0],
    "reduce_shape": [3],
    "reduce_strides": [3],
    "reduce_ndim": [1],
    "non_col_reductions": [1],
}
LAUNCH = {"workgroupCount": [1, 1, 1], "workgroupSize": [1, 3, 1]}


def test_entries_are_pinned_and_disjoint_from_the_installed_small_columns():
    path = CONTRACTS / "reduce.current-tree.translation.json"
    pinned = set(json.loads(path.read_text(encoding="utf-8"))["entries"])
    entries = set(column_small_layout.WIDE_SMALL_COLUMN_ENTRIES)
    # Three template ranks of sixteen wide operations and four 16-bit float
    # extrema, less the two float16 and bfloat16 partition sums installed.
    assert len(entries) == 58 and entries <= pinned
    assert not entries & set(column_small_layout.ALL_SMALL_COLUMN_ENTRIES)
    assert set(column_small_layout.LAUNCH_ENTRIES) == entries | set(
        column_small_layout.ALL_SMALL_COLUMN_ENTRIES
    )
    assert "col_reduce_small_1_reduce_sumfloat16" not in entries
    assert "col_reduce_small_2_reduce_sumfloat16" in entries
    assert wide_column_dispatch.signature("col_reduce_small_5_reduce_maxbfloat16") == (
        "max",
        "bfloat16",
    )
    assert wide_column_dispatch.signature("col_reduce_small_2_reduce_produint64") == (
        "prod",
        "uint64",
    )
    for entry in (
        "col_reduce_small_1_reduce_sumfloat32",
        "col_reduce_small_1_reduce_sumfloat16",
        "row_reduce_small_1_reduce_sumint64",
        None,
    ):
        with pytest.raises(ValueError, match="Unknown wide small-column"):
            wide_column_dispatch.signature(entry)
    assert lazy_package_evidence.family("col_reduce_small_1_reduce_sumint64") is None


def test_small_column_cache_keys_wide_variants():
    key = column_small_layout.variant(
        "col_reduce_small_1_reduce_sumcomplex64", (1, 3, 1)
    )
    assert key == "col_reduce_small_1_reduce_sumcomplex64@1x3"
    cache = column_small_packages.SmallColumnPackageCache.__new__(
        column_small_packages.SmallColumnPackageCache
    )
    assert cache.source_entry(key) == "col_reduce_small_1_reduce_sumcomplex64"
    assert cache.workgroup_size(key) == (1, 3, 1)
    with pytest.raises(ValueError, match="Unsupported small column"):
        column_small_layout.variant("col_reduce_small_1_reduce_sumint8", (1, 3, 1))


def native_buffers(entry, words, outputs, metadata):
    """A callback's eleven buffers: input, output and upstream's column metadata."""
    types = wide_column_dispatch.operand_types(entry)
    ctype = element_storage.CTYPES[types["in"]]
    memory = {"in": (ctype * len(words))(*words), "out": (ctype * outputs)()}
    for name, values in metadata.items():
        memory[name] = (element_storage.CTYPES[types[name]] * len(values))(*values)
    array = (runtime.Buffer * len(memory))(
        *(
            runtime.Buffer(
                name.encode(),
                types[name].encode(),
                ctypes.addressof(memory[name]),
                len(memory[name]),
                int(name == "out"),
            )
            for name in memory
        )
    )
    return array, memory


class FakeCache:
    """The small-column cache: one package per entry and threadgroup size."""

    def __init__(self, target):
        self.target = target
        self.requests = []

    def get(self, key):
        self.requests.append(key)
        entry, _ = column_small_layout.split_variant(key)
        types = wide_column_dispatch.operand_types(entry)
        prefix = "" if self.target == "metal" else entry + "_"
        bindings = []
        for name in types:
            value = element_storage.packet(types[name], self.target, [])
            scalar = (
                dict(element_storage.COMPLEX_LAYOUT)
                if types[name] == "complex64"
                else {
                    "elementType": value["dtype"],
                    "elementStrideBytes": element_storage.PHYSICAL_SIZES[
                        value["dtype"]
                    ],
                }
            )
            if self.target == "directx" and types[name] == "float16":
                scalar = {
                    "elementType": "uint16",
                    "elementStrideBytes": 2,
                    "storageEncoding": {
                        "encoding": "ieee754-binary16",
                        "logicalElementType": "float16",
                    },
                }
            member = {"in": "in_", "out": "out_"}.get(name, prefix + name)
            bindings.append(
                {
                    "name": name + "Binding",
                    "scalarLayout": {**scalar, "memberName": member},
                }
            )
        return {"bindings": bindings, "artifact": {"id": key}}, Path("/packages")


def outputs_of(metadata):
    outer = 1
    for size in metadata["shape"][: metadata["ndim"][0]]:
        outer *= size
    return outer * metadata["reduction_stride"][0]


def run(
    monkeypatch,
    tmp_path,
    target,
    entry,
    words,
    *,
    metadata=COLUMNS,
    launch=LAUNCH,
    threads=9,
    fault=None,
):
    operation, dtype = wide_column_dispatch.signature(entry)
    outputs = outputs_of(metadata)
    array, memory = native_buffers(entry, words, outputs, metadata)
    host = SimpleNamespace(
        small_columns=FakeCache(target),
        target=target,
        trace=tmp_path / "dispatch.jsonl",
        trace_values=True,
        dispatch_count=0,
    )

    def execute(_, request):
        inputs = request["inputs"]
        values = element_storage.unpack(dtype, target, inputs["inBinding"]["values"])
        initial = element_storage.unpack(dtype, target, inputs["outBinding"]["values"])
        assert (
            inputs["reduction_strideBinding"]["values"] == metadata["reduction_stride"]
        )
        assert request["execution"] == launch
        if dtype in element_storage.INEXACT:
            # Inexact columns echo their first inputs; identity checks cover them.
            result = values[:outputs]
        else:
            result = wide_column_evidence.expected_words(
                operation, dtype, values, metadata
            )
        result += initial[outputs:]
        if fault == "guard":
            result[-1] ^= 1
        return SimpleNamespace(
            status="ok",
            outputs={"outBinding": element_storage.packet(dtype, target, result)},
            details={"request": {}},
        )

    monkeypatch.setattr(
        wide_column_dispatch,
        "build_native_loader_dispatch_request",
        lambda descriptor, directory, inputs, outputs, execution, **_: {
            "inputs": inputs,
            "outputs": outputs,
            "execution": execution,
        },
    )
    monkeypatch.setattr(wide_column_dispatch, "execute", execute)
    native = runtime.Launch(
        (ctypes.c_uint32 * 3)(*launch["workgroupCount"]),
        (ctypes.c_uint32 * 3)(*launch["workgroupSize"]),
        (ctypes.c_uint32 * 3)(0, 0, 0),
    )
    wide_column_dispatch.dispatch(host, entry, array, len(array), threads, native)
    event = json.loads(host.trace.read_text().splitlines()[-1])
    return list(memory["out"]), event, host


def audit(monkeypatch, event, target):
    """Audit an event against fake packages, skipping native module checks."""
    checked = []
    monkeypatch.setattr(wide_column_evidence, "require_unchanged", lambda *_: None)
    monkeypatch.setattr(
        wide_column_evidence, "audit_native_execution", lambda e: checked.append(target)
    )
    event = {
        **event,
        "details": {
            "request": {
                "target": target,
                "entryPoint": event["entry"],
                "dispatch": {key: event[key] for key in LAUNCH},
            }
        },
    }
    count = wide_column_evidence.audit_event(event, FakeCache(target))
    assert checked == [target]
    return count


@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.parametrize("dtype", ("int64", "uint64"))
def test_integer_columns_wrap_and_recompute(monkeypatch, tmp_path, target, dtype):
    high = INT64_MAX if dtype == "int64" else UINT64_MAX
    words = [high, 2, 3, 4, 5, 6, 7, 8, 9]
    wrapped = -(2**63) + 10 if dtype == "int64" else 10
    for operation, expected in (("sum", [wrapped, 15, 18]), ("min", [4, 2, 3])):
        entry = f"col_reduce_small_1_reduce_{operation}{dtype}"
        out, event, host = run(monkeypatch, tmp_path, target, entry, words)
        assert out == expected and host.dispatch_count == 1
        assert host.small_columns.requests == [entry + "@1x3"]
        assert event["columnMetadata"] == COLUMNS
        assert audit(monkeypatch, event, target) == 3
        tampered = {
            **event,
            "outputValues": [expected[0], expected[1] ^ 1, expected[2]],
        }
        with pytest.raises(ValueError, match="differs from its inputs"):
            audit(monkeypatch, tampered, target)
        with pytest.raises(ValueError, match="upload differs"):
            audit(
                monkeypatch,
                {**event, "operandValues": [high, 2, 4, *words[3:]]},
                target,
            )


def test_outer_and_non_column_positions_fold_every_column():
    # A (2, 3, 4, 5) array reduced over axes 1 and 2 with its last axis kept:
    # two outer blocks of five columns, each folding twelve positions.
    metadata = {
        "reduction_size": [4],
        "reduction_stride": [5],
        "shape": [2],
        "strides": [60],
        "ndim": [1],
        "reduce_shape": [3, 4],
        "reduce_strides": [20, 5],
        "reduce_ndim": [2],
        "non_col_reductions": [3],
    }
    words = list(range(120))
    expected = wide_column_evidence.expected_words("sum", "int64", words, metadata)
    assert expected == [
        sum(words[o * 60 + i * 20 + j * 5 + c] for i in range(3) for j in range(4))
        for o in range(2)
        for c in range(5)
    ]
    assert wide_column_evidence.expected_words("max", "uint64", words, metadata)[
        :2
    ] == [
        55,
        56,
    ]
    with pytest.raises(ValueError, match="integer type"):
        wide_column_evidence.expected_words("sum", "complex64", words, metadata)
    with pytest.raises(ValueError, match="cover its rows"):
        wide_column_evidence.expected_words(
            "sum", "int64", words, {**metadata, "non_col_reductions": [2]}
        )


@pytest.mark.parametrize("target", TARGETS)
def test_two_axis_launches_follow_the_upstream_plan(monkeypatch, tmp_path, target):
    metadata = {
        "reduction_size": [4],
        "reduction_stride": [5],
        "shape": [2],
        "strides": [60],
        "ndim": [1],
        "reduce_shape": [3, 4],
        "reduce_strides": [20, 5],
        "reduce_ndim": [2],
        "non_col_reductions": [3],
    }
    # Two blocks of four columns along x and eight rows along y per output.
    launch = {"workgroupCount": [1, 2, 1], "workgroupSize": [2, 8, 1]}
    words = list(range(-60, 60))
    entry = "col_reduce_small_2_reduce_sumint64"
    out, event, _ = run(
        monkeypatch,
        tmp_path,
        target,
        entry,
        words,
        metadata=metadata,
        launch=launch,
        threads=120,
    )
    assert sum(out) == sum(words) and event["columnMetadata"] == metadata
    with pytest.raises(ValueError, match="launch does not match"):
        run(
            monkeypatch,
            tmp_path,
            target,
            entry,
            words,
            metadata=metadata,
            launch=LAUNCH,
            threads=120,
        )
    with pytest.raises(ValueError, match="template rank"):
        run(
            monkeypatch,
            tmp_path,
            target,
            "col_reduce_small_1_reduce_sumint64",
            words,
            metadata=metadata,
            launch=launch,
            threads=120,
        )


@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.parametrize(
    "entry",
    (
        "col_reduce_small_2_reduce_sumfloat16",
        "col_reduce_small_1_reduce_prodbfloat16",
        "col_reduce_small_1_reduce_maxfloat16",
        "col_reduce_small_1_reduce_sumcomplex64",
    ),
)
def test_inexact_columns_use_reflected_carriers(monkeypatch, tmp_path, target, entry):
    _, dtype = wide_column_dispatch.signature(entry)
    words = {
        "float16": [0x3C00, 0x4000, 0x4200] * 3,
        "bfloat16": [0x3F80, 0x4000, 0x4040] * 3,
        "complex64": [0x400000003F800000, 0x3F80000040000000, 0] * 3,
    }[dtype]
    metadata = COLUMNS
    if entry.startswith("col_reduce_small_2_"):
        # Rank-two templates serve two reduced axes: three rows of one.
        metadata = {
            **COLUMNS,
            "reduce_shape": [3, 1],
            "reduce_strides": [3, 3],
            "reduce_ndim": [2],
            "reduction_size": [1],
            "non_col_reductions": [3],
        }
    out, event, _ = run(monkeypatch, tmp_path, target, entry, words, metadata=metadata)
    assert out == words[:3]
    assert not wide_column_evidence.recomputable(entry)
    assert event["inputs"]["inBinding"] == element_storage.packet(dtype, target, words)
    assert audit(monkeypatch, event, target) == 3


def test_dispatch_refuses_changed_guards_buffers_and_plans(monkeypatch, tmp_path):
    entry = "col_reduce_small_1_reduce_sumint64"
    with pytest.raises(RuntimeError, match="guard"):
        run(monkeypatch, tmp_path, "opengl", entry, list(range(9)), fault="guard")
    # Off Metal, subnormal bfloat16 inputs stay refused, as in the base families.
    for target in ("directx", "opengl"):
        with pytest.raises(ValueError, match="crosstl#2000"):
            run(
                monkeypatch,
                tmp_path,
                target,
                "col_reduce_small_1_reduce_prodbfloat16",
                [0x3F80, 0x8001] + [0] * 7,
            )
    array, memory = native_buffers(entry, list(range(9)), 3, COLUMNS)
    buffers = wide_column_dispatch.supplied_buffers(array, 11, "Wide small-column")
    assert wide_column_dispatch.validate(entry, buffers, 9, LAUNCH)[:2] == (
        "sum",
        "int64",
    )
    with pytest.raises(ValueError, match="type, alignment"):
        wide_column_dispatch.validate(
            "col_reduce_small_1_reduce_sumuint64", buffers, 9, LAUNCH
        )
    with pytest.raises(ValueError, match="buffers do not match"):
        wide_column_dispatch.validate(entry, {"in": buffers["in"]}, 9, LAUNCH)
    with pytest.raises(ValueError, match="logical size"):
        wide_column_dispatch.validate(entry, buffers, 8, LAUNCH)
    # Thirty-two reduced elements per output belong to the looped pass.
    words = list(range(32 * 3))
    looped = {**COLUMNS, "reduction_size": [32], "reduce_shape": [32]}
    array, _ = native_buffers(entry, words, 3, looped)
    buffers = wide_column_dispatch.supplied_buffers(array, 11, "Wide small-column")
    with pytest.raises(ValueError, match="upstream plan"):
        wide_column_dispatch.validate(entry, buffers, 96, LAUNCH)
    host = SimpleNamespace(small_columns=None)
    with pytest.raises(ValueError, match="pinned sources"):
        wide_column_dispatch.dispatch(host, entry, array, 11, 96, None)


def test_audit_refuses_other_entries_and_changed_launches(monkeypatch, tmp_path):
    entry = "col_reduce_small_1_reduce_minint64"
    _, event, _ = run(monkeypatch, tmp_path, "directx", entry, list(range(9)))
    with pytest.raises(ValueError, match="not a wide small-column"):
        audit(
            monkeypatch,
            {**event, "entry": "col_reduce_small_1_reduce_minint32"},
            "directx",
        )
    with pytest.raises(ValueError, match="launch does not match"):
        audit(monkeypatch, {**event, "workgroupSize": [1, 2, 1]}, "directx")
    with pytest.raises(ValueError, match="exact thread grid"):
        audit(monkeypatch, {**event, "threadGridSize": [1, 3, 1]}, "directx")
    with pytest.raises(ValueError, match="output hash"):
        audit(monkeypatch, {**event, "outputHash": "0" * 64}, "directx")
    with pytest.raises(ValueError, match="cached package"):
        audit(monkeypatch, {**event, "artifact": {"id": "other"}}, "directx")


def test_detached_metadata_is_range_checked_before_the_plan():
    entry = "col_reduce_small_1_reduce_sumint64"
    assert (
        column_small_layout.validate_metadata(entry, COLUMNS, 9, 9, 3, LAUNCH)
        == COLUMNS
    )
    for name, value in (("ndim", [2**31]), ("reduction_size", [-1]), ("strides", [])):
        with pytest.raises(ValueError, match="metadata values"):
            column_small_layout.validate_metadata(
                entry, {**COLUMNS, name: value}, 9, 9, 3, LAUNCH
            )
    with pytest.raises(ValueError, match="layout"):
        column_small_layout.validate_metadata(entry, {"ndim": [0]}, 9, 9, 3, LAUNCH)
    with pytest.raises(ValueError, match="source span"):
        column_small_layout.validate_metadata(entry, COLUMNS, 9, 10, 3, LAUNCH)


def test_runtime_and_host_adapter_route_wide_columns():
    source = Path(runtime.__file__).read_text(encoding="utf-8")
    assert "if entry in wide_column_dispatch.ENTRIES:" in source
    adapter = (
        Path(runtime.__file__).with_name("backend.cpp").read_text(encoding="utf-8")
    )
    assert (
        "if (eight_byte && plan.type != ContiguousAllReduce && !small_rows && "
        "!small_columns) {"
        in adapter
    )
    assert "column_shape_size(plan.shape) < 32;" in adapter

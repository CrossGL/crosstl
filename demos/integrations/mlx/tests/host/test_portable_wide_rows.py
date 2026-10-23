"""Wide small-row reductions mirror the pinned instantiations, check each
launch against upstream's row plan and recompute integer rows exactly."""

import ctypes
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from demos.integrations.mlx.portable_host import (
    element_storage,
    lazy_package_evidence,
    region_dispatch,
    runtime,
    small_row_packages,
    wide_row_dispatch,
    wide_row_evidence,
)

ROOT = Path(__file__).resolve().parents[5]
CONTRACTS = ROOT / "demos/integrations/mlx/contracts"
TARGETS = ("metal", "directx", "opengl")
INT64_MAX, UINT64_MAX = 2**63 - 1, 2**64 - 1
# A contiguous (3, 3) array reduced over its last axis: one thread per row.
ROWS = {
    "row_size": [3],
    "non_row_reductions": [1],
    "shape": [3],
    "strides": [3],
    "ndim": [1],
    "reduce_shape": [0],
    "reduce_strides": [0],
    "reduce_ndim": [0],
}
SCALAR = {
    "workgroupCount": [1, 1, 1],
    "workgroupSize": [3, 1, 1],
    "threadGridSize": [3, 1, 1],
}


def test_entries_are_pinned_and_disjoint_from_the_installed_small_rows():
    path = CONTRACTS / "reduce.current-tree.translation.json"
    pinned = set(json.loads(path.read_text(encoding="utf-8"))["entries"])
    entries = set(small_row_packages.WIDE_SMALL_ROW_ENTRIES)
    assert len(entries) == 48 and entries <= pinned
    assert not entries & set(small_row_packages.SMALL_ROW_ENTRIES)
    assert set(small_row_packages.ALL_SMALL_ROW_ENTRIES) == entries | set(
        small_row_packages.SMALL_ROW_ENTRIES
    )
    assert wide_row_dispatch.signature("row_reduce_small_5_reduce_prodbfloat16") == (
        "prod",
        "bfloat16",
    )
    assert wide_row_dispatch.signature("row_reduce_small_2_reduce_maxuint64") == (
        "max",
        "uint64",
    )
    for entry in ("row_reduce_small_1_reduce_sumfloat32", "all_reduce_sumint64", None):
        with pytest.raises(ValueError, match="Unknown wide small-row"):
            wide_row_dispatch.signature(entry)


def test_small_row_cache_translates_wide_entries(monkeypatch, tmp_path):
    cache = small_row_packages.SmallRowPackageCache(tmp_path, tmp_path, "metal")
    built = []
    monkeypatch.setattr(cache, "_require_source", lambda: None)
    monkeypatch.setattr(
        cache,
        "_build",
        lambda entry, output, identity, region: built.append(entry)
        or output.joinpath("index.json").write_text("{}"),
    )
    monkeypatch.setattr(cache, "_load", lambda directory, identity: identity["entry"])
    entry = "row_reduce_small_1_reduce_sumcomplex64"
    assert cache.get(entry, thread_grid_size=(3, 1, 1), workgroup_size=(3, 1, 1)) == (
        entry,
    )
    assert built == [entry]
    with pytest.raises(ValueError, match="Unknown small-row"):
        cache.get(
            "row_reduce_small_1_reduce_sumint8",
            thread_grid_size=(3, 1, 1),
            workgroup_size=(3, 1, 1),
        )


def native_buffers(entry, words, rows, metadata):
    """A callback's ten buffers: input, output and upstream's row metadata."""
    types = wide_row_dispatch.operand_types(entry)
    ctype = element_storage.CTYPES[types["in"]]
    memory = {"in": (ctype * len(words))(*words), "out": (ctype * rows)()}
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
    """The small-row cache: one package on Metal, one per region elsewhere."""

    def __init__(self, target):
        self.target = target
        self.requests = []

    def get(self, entry, *, thread_grid_size, workgroup_size):
        self.requests.append((entry, list(thread_grid_size), list(workgroup_size)))
        types = wide_row_dispatch.operand_types(entry)
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
        descriptor = {"bindings": bindings, "artifact": {"id": entry}}
        return ((descriptor, Path("/packages")),)


def run(
    monkeypatch,
    tmp_path,
    target,
    entry,
    words,
    *,
    metadata=ROWS,
    launch=SCALAR,
    fault=None,
):
    operation, dtype = wide_row_dispatch.signature(entry)
    rows = 1
    for size in metadata["shape"][: metadata["ndim"][0]]:
        rows *= size
    array, memory = native_buffers(entry, words, rows, metadata)
    host = SimpleNamespace(
        small_rows=FakeCache(target),
        target=target,
        trace=tmp_path / "dispatch.jsonl",
        trace_values=True,
        dispatch_count=0,
    )
    calls = []

    def execute(inputs, outputs):
        values = element_storage.unpack(dtype, target, inputs["inBinding"]["values"])
        initial = element_storage.unpack(dtype, target, inputs["outBinding"]["values"])
        assert inputs["row_sizeBinding"]["values"] == metadata["row_size"]
        if dtype in element_storage.INEXACT:
            # Inexact rows echo their first inputs; identity checks cover them.
            result = values[:rows]
        else:
            result = wide_row_evidence.expected_words(
                operation, dtype, values, metadata
            )
        result += initial[rows:]
        if fault == "guard":
            result[-1] ^= 1
        return SimpleNamespace(
            status="ok",
            outputs={"outBinding": element_storage.packet(dtype, target, result)},
            details={"request": {}},
        )

    monkeypatch.setattr(
        wide_row_dispatch,
        "build_native_loader_dispatch_request",
        lambda descriptor, directory, inputs, outputs, execution, **_: {
            "inputs": inputs,
            "outputs": outputs,
            "execution": execution,
        },
    )
    monkeypatch.setattr(
        wide_row_dispatch,
        "execute",
        lambda _, request: calls.append("metal")
        or execute(request["inputs"], request["outputs"]),
    )
    monkeypatch.setattr(
        region_dispatch,
        "run_regions",
        lambda _, regions, inputs, outputs, execution: calls.append("regions")
        or execute(inputs, outputs),
    )
    native = runtime.Launch(
        (ctypes.c_uint32 * 3)(*launch["workgroupCount"]),
        (ctypes.c_uint32 * 3)(*launch["workgroupSize"]),
        (ctypes.c_uint32 * 3)(*launch["threadGridSize"]),
    )
    wide_row_dispatch.dispatch(host, entry, array, len(array), len(words), native)
    event = json.loads(host.trace.read_text().splitlines()[-1])
    assert calls == ["metal" if target == "metal" else "regions"]
    return list(memory["out"]), event, host


def audit(monkeypatch, event, target):
    """Audit an event against fake packages, skipping native module checks."""
    checked = []
    monkeypatch.setattr(wide_row_evidence, "require_unchanged", lambda *_: None)
    monkeypatch.setattr(
        wide_row_evidence, "audit_native_execution", lambda e: checked.append("metal")
    )
    monkeypatch.setattr(
        wide_row_evidence,
        "audit_regions",
        lambda e, packages, modules: checked.append("regions"),
    )
    if target == "metal":
        event = {
            **event,
            "details": {
                "request": {
                    "target": target,
                    "entryPoint": event["entry"],
                    "dispatch": {key: event[key] for key in SCALAR},
                }
            },
        }
    rows = wide_row_evidence.audit_event(event, FakeCache(target), "/modules")
    assert checked == ["metal" if target == "metal" else "regions"]
    return rows


@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.parametrize("dtype", ("int64", "uint64"))
def test_integer_rows_wrap_and_recompute(monkeypatch, tmp_path, target, dtype):
    high = INT64_MAX if dtype == "int64" else UINT64_MAX
    words = [high, 2, 3, 4, 5, 6, 7, 8, 9]
    wrapped = -(2**63) + 4 if dtype == "int64" else 4
    for operation, expected in (("sum", [wrapped, 15, 24]), ("max", [high, 6, 9])):
        entry = f"row_reduce_small_1_reduce_{operation}{dtype}"
        out, event, host = run(monkeypatch, tmp_path, target, entry, words)
        assert out == expected and host.dispatch_count == 1
        assert host.small_rows.requests == [(entry, [3, 1, 1], [3, 1, 1])]
        assert audit(monkeypatch, event, target) == 3
        tampered = {
            **event,
            "outputValues": [expected[0], expected[1] ^ 1, expected[2]],
        }
        tampered["outputHash"] = event["outputHash"]
        with pytest.raises(ValueError, match="differs from its inputs"):
            audit(monkeypatch, tampered, target)
        with pytest.raises(ValueError, match="upload differs"):
            audit(
                monkeypatch,
                {**event, "operandValues": [high, 2, 4, *words[3:]]},
                target,
            )


def test_strided_non_row_positions_fold_every_row():
    # Upstream's (12, 3, 10) array reduced over axes 0 and 2: three outputs,
    # each folding twelve rows of ten that start 30 elements apart.
    metadata = {
        "row_size": [10],
        "non_row_reductions": [12],
        "shape": [3],
        "strides": [10],
        "ndim": [1],
        "reduce_shape": [12],
        "reduce_strides": [30],
        "reduce_ndim": [1],
    }
    words = list(range(360))
    expected = wide_row_evidence.expected_words("sum", "int64", words, metadata)
    assert expected == [
        sum(words[r * 10 + p * 30 + c] for p in range(12) for c in range(10))
        for r in range(3)
    ]
    assert wide_row_evidence.expected_words("min", "uint64", words, metadata) == [
        0,
        10,
        20,
    ]
    assert wide_row_evidence.expected_words(
        "prod", "int64", [2**62, 2], {**ROWS, "row_size": [2], "shape": [1]}
    ) == [-(2**63)]
    with pytest.raises(ValueError, match="integer type"):
        wide_row_evidence.expected_words("sum", "complex64", [0], ROWS)


@pytest.mark.parametrize("target", TARGETS)
def test_simdgroup_launches_follow_the_upstream_plan(monkeypatch, tmp_path, target):
    metadata = {
        "row_size": [10],
        "non_row_reductions": [12],
        "shape": [3],
        "strides": [10],
        "ndim": [1],
        "reduce_shape": [12],
        "reduce_strides": [30],
        "reduce_ndim": [1],
    }
    launch = {
        "workgroupCount": [1, 3, 1],
        "workgroupSize": [32, 1, 1],
        "threadGridSize": [32, 3, 1],
    }
    words = list(range(-180, 180))
    out, event, _ = run(
        monkeypatch,
        tmp_path,
        target,
        "row_reduce_small_1_reduce_sumint64",
        words,
        metadata=metadata,
        launch=launch,
    )
    assert sum(out) == sum(words) and event["rowMetadata"] == metadata
    with pytest.raises(ValueError, match="launch does not match"):
        run(
            monkeypatch,
            tmp_path,
            target,
            "row_reduce_small_1_reduce_sumint64",
            words,
            metadata=metadata,
            launch=SCALAR,
        )


@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.parametrize("dtype", ("float16", "bfloat16", "complex64"))
def test_inexact_rows_use_reflected_carriers(monkeypatch, tmp_path, target, dtype):
    words = {
        "float16": [0x3C00, 0x4000, 0x4200] * 3,
        "bfloat16": [0x3F80, 0x4000, 0x4040] * 3,
        "complex64": [0x400000003F800000, 0x3F80000040000000, 0] * 3,
    }[dtype]
    entry = f"row_reduce_small_1_reduce_sum{dtype}"
    out, event, _ = run(monkeypatch, tmp_path, target, entry, words)
    assert out == words[:3]
    assert not wide_row_evidence.recomputable(entry)
    assert event["inputs"]["inBinding"] == element_storage.packet(dtype, target, words)
    assert audit(monkeypatch, event, target) == 3


def test_dispatch_refuses_changed_guards_buffers_and_metadata(monkeypatch, tmp_path):
    entry = "row_reduce_small_1_reduce_sumint64"
    with pytest.raises(RuntimeError, match="guard"):
        run(monkeypatch, tmp_path, "opengl", entry, list(range(9)), fault="guard")
    # Off Metal, subnormal bfloat16 inputs stay refused, as in the base families.
    for target in ("directx", "opengl"):
        with pytest.raises(ValueError, match="crosstl#2000"):
            run(
                monkeypatch,
                tmp_path,
                target,
                "row_reduce_small_1_reduce_sumbfloat16",
                [0x3F80, 0x8001] + [0] * 7,
            )
    array, memory = native_buffers(entry, list(range(9)), 3, ROWS)
    buffers = wide_row_dispatch.supplied_buffers(array, 10, "Wide small-row")
    assert wide_row_dispatch.validate(entry, buffers, 9, SCALAR)[:2] == ("sum", "int64")
    with pytest.raises(ValueError, match="type, alignment"):
        wide_row_dispatch.validate(
            "row_reduce_small_1_reduce_sumuint64", buffers, 9, SCALAR
        )
    with pytest.raises(ValueError, match="buffers do not match"):
        wide_row_dispatch.validate(entry, {"in": buffers["in"]}, 9, SCALAR)
    with pytest.raises(ValueError, match="logical size"):
        wide_row_dispatch.validate(entry, buffers, 8, SCALAR)
    memory["row_size"][0] = 65
    with pytest.raises(ValueError, match="1 to 64 elements"):
        wide_row_dispatch.validate(entry, buffers, 9, SCALAR)
    host = SimpleNamespace(small_rows=None)
    with pytest.raises(ValueError, match="pinned sources"):
        wide_row_dispatch.dispatch(host, entry, array, 10, 9, None)


def test_audit_refuses_other_entries_and_changed_launches(monkeypatch, tmp_path):
    entry = "row_reduce_small_1_reduce_minint64"
    _, event, _ = run(monkeypatch, tmp_path, "directx", entry, list(range(9)))
    with pytest.raises(ValueError, match="not a wide small-row"):
        audit(
            monkeypatch,
            {**event, "entry": "row_reduce_small_1_reduce_minint32"},
            "directx",
        )
    with pytest.raises(ValueError, match="launch does not match"):
        audit(monkeypatch, {**event, "workgroupSize": [32, 1, 1]}, "directx")
    with pytest.raises(ValueError, match="output hash"):
        audit(monkeypatch, {**event, "outputHash": "0" * 64}, "directx")
    with pytest.raises(ValueError, match="cached package"):
        audit(monkeypatch, {**event, "artifact": {"id": "other"}}, "directx")


def test_region_audit_accepts_the_wide_entries():
    entry = "row_reduce_small_1_reduce_sumcomplex64"
    with pytest.raises(ValueError, match="region coverage"):
        lazy_package_evidence.audit_regions(
            {"entry": entry, "details": {"regions": []}}, [object()], Path("/m")
        )
    lazy_package_evidence.audit_regions(
        {"entry": entry, "details": {"regions": []}}, [], Path("/m")
    )


def test_runtime_and_host_adapter_route_wide_rows():
    source = Path(runtime.__file__).read_text(encoding="utf-8")
    assert "if entry in wide_row_dispatch.ENTRIES:" in source
    adapter = (
        Path(runtime.__file__).with_name("backend.cpp").read_text(encoding="utf-8")
    )
    assert "if (wide && plan.type != ContiguousAllReduce && !small_rows) {" in adapter

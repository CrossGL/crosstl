"""MLX searchsorted through the portable host: check each pass's buffers and
launch, dispatch its package and audit the retained uploads, guards and
recomputed binary searches."""

import ctypes
import json
import random
import struct
from pathlib import Path
from types import SimpleNamespace

import pytest

from demos.integrations.mlx.portable_host import (
    element_storage,
    prepare,
    runtime,
    searchsorted_dispatch,
    searchsorted_evidence,
    searchsorted_layout,
    searchsorted_packages,
)

TARGETS = ("metal", "directx", "opengl")
NAN32 = 0x7FC00000


def f32(value):
    return struct.unpack("<I", struct.pack("<f", value))[0]


def words_of(dtype, values):
    if dtype == "float32":
        return [NAN32 if value != value else f32(value) for value in values]
    if dtype == "complex64":
        return [f32(real) | f32(imag) << 32 for real, imag in values]
    return list(values)


def native_buffers(entry, sequence, values, n, stride):
    types = searchsorted_layout.operand_types(entry)
    memory = {
        "a": (element_storage.CTYPES[types["a"]] * len(sequence))(*sequence),
        "v": (element_storage.CTYPES[types["v"]] * len(values))(*values),
        "out": (ctypes.c_uint32 * len(values))(),
        "n": (ctypes.c_uint32 * 1)(n),
        "a_stride": (ctypes.c_int64 * 1)(stride),
    }
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
    """The searchsorted cache: one package per entry and pass size."""

    def __init__(self, target):
        self.target = target
        self.requests = []

    def get(self, key):
        self.requests.append(key)
        entry, _ = searchsorted_layout.split_variant(key)
        prefix = "" if self.target == "metal" else entry + "_"
        bindings = []
        for name, kind in searchsorted_layout.operand_types(entry).items():
            value = element_storage.packet(kind, self.target, [0])
            if kind == "complex64":
                scalar = dict(element_storage.COMPLEX_LAYOUT)
            else:
                scalar = {
                    "elementType": value["dtype"],
                    "elementStrideBytes": element_storage.PHYSICAL_SIZES[
                        value["dtype"]
                    ],
                }
            member = {"out": "out_", "a": "a", "v": "v"}.get(name, prefix + name)
            bindings.append(
                {
                    "name": name + "Binding",
                    "scalarLayout": {**scalar, "memberName": member},
                }
            )
        return {"bindings": bindings, "artifact": {"id": key}}, Path("/packages")


def run(monkeypatch, tmp_path, target, entry, sequence, values, n, stride, fault=None):
    """Dispatch one pass through a fake package whose kernel is the model."""
    count = len(values)
    launch = searchsorted_layout.execution(count)
    array, memory = native_buffers(entry, sequence, values, n, stride)
    host = SimpleNamespace(
        searchsorted=FakeCache(target),
        target=target,
        trace=tmp_path / "dispatch.jsonl",
        trace_values=(searchsorted_dispatch.PREFIX,),
        dispatch_count=0,
    )
    types = searchsorted_layout.operand_types(entry)

    def execute(_, request):
        inputs = request["inputs"]
        assert request["execution"] == launch
        seen = {
            name: element_storage.unpack(
                types[name], target, inputs[name + "Binding"]["values"]
            )
            for name in ("a", "v")
        }
        assert seen == {"a": sequence, "v": values}
        initial = element_storage.unpack(
            "uint32", target, inputs["outBinding"]["values"]
        )
        assert initial == element_storage.guard_words(
            "uint32", count + element_storage.GUARD_COUNT
        )
        model = searchsorted_evidence.expected_outputs(
            entry, {"n": n, "aStride": stride}, seen
        )
        result = model + initial[count:]
        if fault == "guard":
            result[-1] ^= 1
        if fault == "value":
            result[0] ^= 1
        return SimpleNamespace(
            status="ok",
            outputs={"outBinding": element_storage.packet("uint32", target, result)},
            details={"request": {}},
        )

    monkeypatch.setattr(
        searchsorted_dispatch,
        "build_native_loader_dispatch_request",
        lambda descriptor, directory, inputs, outputs, execution, **_: {
            "inputs": inputs,
            "outputs": outputs,
            "execution": execution,
        },
    )
    monkeypatch.setattr(searchsorted_dispatch, "execute", execute)
    native = runtime.Launch(
        (ctypes.c_uint32 * 3)(*launch["workgroupCount"]),
        (ctypes.c_uint32 * 3)(*launch["workgroupSize"]),
        (ctypes.c_uint32 * 3)(0, 0, 0),
    )
    searchsorted_dispatch.dispatch(host, entry, array, len(array), count, native)
    event = json.loads(host.trace.read_text().splitlines()[-1])
    return list(memory["out"]), event, host


def audit(monkeypatch, event, target):
    """Audit an event against fake packages, skipping native module checks."""
    checked = []
    monkeypatch.setattr(searchsorted_evidence, "require_unchanged", lambda *_: None)
    monkeypatch.setattr(
        searchsorted_evidence,
        "audit_native_execution",
        lambda e: checked.append(target),
    )
    event = {
        **event,
        "details": {
            "request": {
                "target": target,
                "entryPoint": event["entry"],
                "dispatch": {key: event[key] for key in searchsorted_evidence.GEOMETRY},
            }
        },
    }
    retained = searchsorted_evidence.audit_event(event, FakeCache(target))
    assert checked == [target]
    return retained


def test_entries_are_the_pinned_instantiations():
    assert len(searchsorted_layout.ENTRIES) == 26
    assert set(searchsorted_layout.DTYPES) == set(element_storage.STORAGE_DTYPES)
    assert searchsorted_layout.signature("searchsorted_bool__left") == ("bool_", False)
    assert searchsorted_layout.signature("searchsorted_complex64_right") == (
        "complex64",
        True,
    )
    assert searchsorted_layout.operand_types("searchsorted_int64_left") == {
        "a": "int64",
        "v": "int64",
        "out": "uint32",
        "n": "uint32",
        "a_stride": "int64",
    }
    for entry in ("searchsorted_float64_left", "searchsorted_float32_middle", None):
        with pytest.raises(ValueError, match="Unknown searchsorted entry"):
            searchsorted_layout.signature(entry)


def test_variants_name_one_pass_size():
    key = searchsorted_layout.variant("searchsorted_float32_left", 257)
    assert key == "searchsorted_float32_left@257"
    assert searchsorted_layout.split_variant(key) == ("searchsorted_float32_left", 257)
    assert searchsorted_layout.execution(257) == {
        "workgroupCount": [257, 1, 1],
        "workgroupSize": [1, 1, 1],
    }
    for count in (0, 65536, True):
        with pytest.raises(ValueError, match="group limit"):
            searchsorted_layout.variant("searchsorted_float32_left", count)
    for key in (
        "searchsorted_float32_left",
        "searchsorted_float32_left@0257",
        "searchsorted_float32_left@1@2",
        "searchsorted_float64_left@3",
    ):
        with pytest.raises(ValueError):
            searchsorted_layout.split_variant(key)


@pytest.mark.parametrize("right", [False, True])
def test_binary_search_matches_numpy_with_nan_last(right):
    np = pytest.importorskip("numpy")
    generator = random.Random(3)
    side = "right" if right else "left"
    for n in (1, 2, 7, 8, 9, 33):
        sequence = sorted(
            generator.choice([-2.5, -0.0, 0.0, 1.0, 3.25]) for _ in range(n)
        )
        if n > 2:
            sequence[-1] = float("nan")
        values = [
            generator.choice([-3.0, -0.0, 1.0, 2.0, 9.0, float("nan")])
            for _ in range(9)
        ]
        expected = np.searchsorted(
            np.array(sequence, np.float32), np.array(values, np.float32), side=side
        ).tolist()
        entry = f"searchsorted_float32_{side}"
        got = searchsorted_evidence.expected_outputs(
            entry,
            {"n": n, "aStride": 1},
            {"a": words_of("float32", sequence), "v": words_of("float32", values)},
        )
        assert got == expected


def test_binary_search_orders_complex_lexicographically_and_strides():
    sequence = words_of("complex64", [(0, 5), (9, 9), (1, -1), (9, 9), (1, 2)])
    values = words_of("complex64", [(1, 0), (1, 2), (2, 0), (0, 0)])
    # The strided view holds (0, 5), (1, -1), (1, 2).
    assert searchsorted_evidence.expected_outputs(
        "searchsorted_complex64_left",
        {"n": 3, "aStride": 2},
        {"a": sequence, "v": values},
    ) == [2, 2, 3, 0]
    assert searchsorted_evidence.expected_outputs(
        "searchsorted_complex64_right",
        {"n": 3, "aStride": 2},
        {"a": sequence, "v": values},
    ) == [2, 3, 3, 0]


@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.parametrize(
    "entry,sequence,values,n,stride",
    [
        (
            "searchsorted_float32_left",
            [1.0, 2.0, 2.0, 4.0],
            [0.0, 1.0, 2.0, 3.0, 5.0],
            4,
            1,
        ),
        (
            "searchsorted_float32_right",
            [1.0, 9.0, 2.0, 9.0, 2.0, 9.0, 4.0],
            [2.0, 4.5],
            4,
            2,
        ),
        ("searchsorted_int64_left", [-(2**40), -3, 7, 2**40], [-3, 0, 2**41], 4, 1),
        ("searchsorted_bool__right", [0, 0, 1], [0, 1], 3, 1),
        (
            "searchsorted_complex64_left",
            [(0, 1), (1, 0), (1, 1)],
            [(1, 0), (1, 2)],
            3,
            1,
        ),
    ],
)
def test_dispatch_and_audit_recompute_each_search(
    monkeypatch, tmp_path, target, entry, sequence, values, n, stride
):
    dtype, _ = searchsorted_layout.signature(entry)
    sequence, values = words_of(dtype, sequence), words_of(dtype, values)
    out, event, host = run(
        monkeypatch, tmp_path, target, entry, sequence, values, n, stride
    )
    assert out == searchsorted_evidence.expected_outputs(
        entry, {"n": n, "aStride": stride}, {"a": sequence, "v": values}
    )
    assert host.searchsorted.requests == [f"{entry}@{len(values)}"]
    assert host.dispatch_count == 1
    assert event["searchMetadata"] == {"n": n, "aStride": stride}
    assert audit(monkeypatch, event, target) is True


@pytest.mark.parametrize("target", TARGETS)
def test_dispatch_refuses_guard_damage_and_audit_refuses_wrong_values(
    monkeypatch, tmp_path, target
):
    entry = "searchsorted_int32_left"
    sequence, values = [1, 3, 5, 7], [0, 4, 8]
    with pytest.raises(RuntimeError, match="guard"):
        run(monkeypatch, tmp_path, target, entry, sequence, values, 4, 1, fault="guard")
    (tmp_path / "dispatch.jsonl").unlink(missing_ok=True)
    _, event, _ = run(
        monkeypatch, tmp_path, target, entry, sequence, values, 4, 1, fault="value"
    )
    with pytest.raises(ValueError, match="binary search"):
        audit(monkeypatch, event, target)
    _, event, _ = run(monkeypatch, tmp_path, target, entry, sequence, values, 4, 1)
    changed = {**event, "operandValues": {**event["operandValues"], "v": [0, 4, 9]}}
    with pytest.raises(ValueError, match="upload of v"):
        audit(monkeypatch, changed, target)
    changed = {**event, "searchMetadata": {"n": 4, "aStride": 2}}
    with pytest.raises(ValueError, match="metadata"):
        audit(monkeypatch, changed, target)


def test_compacted_records_keep_extents(monkeypatch, tmp_path):
    entry = "searchsorted_uint32_left"
    sequence = list(range(0, 2 * 40000, 2))
    values = list(range(30000))
    out, event, _ = run(
        monkeypatch, tmp_path, "opengl", entry, sequence, values, len(sequence), 1
    )
    assert out[:4] == [0, 1, 1, 2]
    assert not isinstance(event["operandValues"]["a"], list)
    assert audit(monkeypatch, event, "opengl") is False


@pytest.mark.parametrize(
    "change,message",
    [
        ({"n": 0}, "metadata"),
        ({"stride": 0}, "metadata"),
        ({"stride": -1}, "metadata"),
        ({"span": 3}, "metadata"),
        ({"launch": 2}, "launch"),
        ({"overlap": True}, "overlaps"),
        ({"dtype": "int64"}, "type, extent or direction"),
    ],
)
def test_validate_refuses_inconsistent_passes(change, message):
    entry = "searchsorted_int32_left"
    n = change.get("n", 4)
    stride = change.get("stride", 1)
    sequence = [1, 3, 5, 7][: change.get("span", 4)]
    array, memory = native_buffers(entry, sequence, [0, 4, 8], n, stride)
    buffers = {buffer.name.decode(): buffer for buffer in array}
    if "dtype" in change:
        buffers["v"].dtype = change["dtype"].encode()
    if change.get("overlap"):
        buffers["out"].data = buffers["v"].data
    with pytest.raises(ValueError, match=message):
        searchsorted_layout.validate(
            entry, buffers, 3, searchsorted_layout.execution(change.get("launch", 3))
        )


def test_package_configs_carry_regions_and_opengl_index_bounds(tmp_path):
    key = "searchsorted_float32_left@257"
    metal = searchsorted_packages.project_config(tmp_path, "out", "metal", key)
    assert metal.entry_points == {
        searchsorted_layout.SOURCE: ("searchsorted_float32_left",)
    }
    assert not metal.index_range_assertions
    options = metal.source_options["metal"]
    assert options["preserve_resource_origins"] is True
    assert options["target_options"] == {"metal": {}}
    for target in ("directx", "opengl"):
        config = searchsorted_packages.project_config(tmp_path, "out", target, key)
        region = config.source_options["metal"]["target_options"][target]
        assert set(region) == {"dispatch_region"}
        assert region["dispatch_region"] == searchsorted_packages._region(257)
    opengl = searchsorted_packages.project_config(tmp_path, "out", "opengl", key)
    (assertion,) = opengl.index_range_assertions
    assert (
        assertion.source,
        assertion.expression,
        assertion.value_range.minimum,
        assertion.value_range.maximum,
    ) == (
        searchsorted_layout.SOURCE,
        searchsorted_packages.INDEX_EXPRESSION,
        0,
        searchsorted_layout.MAX_SPAN - 1,
    )
    cache = searchsorted_packages.SearchSortedPackageCache(tmp_path, tmp_path, "opengl")
    assert cache.workgroup_size(key) == (1, 1, 1)
    assert cache.source_entry(key) == "searchsorted_float32_left"


def test_runtime_and_host_adapter_route_searchsorted():
    source = Path(runtime.__file__).read_text(encoding="utf-8")
    assert "searchsorted_dispatch.dispatch(" in source
    assert "searchsorted_layout.ENTRIES and self.searchsorted is not None" in source
    adapter = (
        Path(runtime.__file__).with_name("backend.cpp").read_text(encoding="utf-8")
    )
    assert "void SearchSorted::eval_gpu(" in adapter
    assert '"searchsorted_"' in adapter
    assert '"SearchSorted",' in Path(prepare.__file__).read_text(encoding="utf-8")

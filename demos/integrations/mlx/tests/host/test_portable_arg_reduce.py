"""Arg reductions mirror the pinned instantiations, check each launch against
ArgReduce::eval_gpu's plan and recompute every index from the input."""

import ctypes
import json
import math
import struct
from pathlib import Path
from types import SimpleNamespace

import pytest

from demos.integrations.mlx.portable_host import (
    arg_reduce_dispatch,
    arg_reduce_evidence,
    arg_reduce_layout,
    arg_reduce_packages,
    element_storage,
    prepare,
    runtime,
)

ROOT = Path(__file__).resolve().parents[5]
CONTRACTS = ROOT / "demos/integrations/mlx/contracts"
TARGETS = ("metal", "directx", "opengl")
SHAPE = (10, 12, 13)
DTYPES = arg_reduce_layout.INSTANTIATED


def contiguous(shape):
    return arg_reduce_layout.contiguous_strides(list(shape))


def plan_for(shape, axis, strides=None):
    """ArgReduce::eval_gpu's metadata for ``shape`` reduced over ``axis``."""
    strides = list(strides or contiguous(shape))
    kept = list(shape)
    kept[axis] = 1
    out_strides = contiguous(kept)
    rest = [extent for index, extent in enumerate(shape) if index != axis]
    in_strides = [stride for index, stride in enumerate(strides) if index != axis]
    del out_strides[axis]
    ndim = len(rest)
    metadata = {
        "shape": rest or [0],
        "in_strides": in_strides or [0],
        "out_strides": out_strides or [0],
        "ndim": [ndim],
        "axis_stride": [strides[axis]],
        "axis_size": [shape[axis]],
    }
    rows = math.prod(rest)
    launch = {
        "workgroupCount": [1, rows, 1],
        "workgroupSize": [arg_reduce_layout.width(shape[axis]), 1, 1],
    }
    span = sum((extent - 1) * stride for extent, stride in zip(shape, strides)) + 1
    return metadata, launch, rows, span


def float_word(dtype, value):
    if dtype == "float32":
        return struct.unpack("<I", struct.pack("<f", value))[0]
    if dtype == "float16":
        return struct.unpack("<H", struct.pack("<e", value))[0]
    return struct.unpack("<I", struct.pack("<f", value))[0] >> 16


def sample_words(dtype, count):
    """Deterministic words of ``dtype`` with repeated values, so ties occur."""
    if dtype == "bool_":
        return [int(index * 7 % 5 < 2) for index in range(count)]
    if dtype in element_storage.FLOATING:
        return [
            float_word(dtype, ((index * 37) % 23) / 8 - 1) for index in range(count)
        ]
    size = 8 * ctypes.sizeof(element_storage.CTYPES[dtype])
    signed = dtype.startswith("int")
    span = 1 << min(size, 16)
    values = [(index * 7919) % span for index in range(count)]
    if signed:
        values = [value - span // 2 for value in values]
    return values


def reference(operation, dtype, words):
    """numpy's rule: the first NaN, else the first extreme value."""
    values = [arg_reduce_evidence._value(dtype, word) for word in words]
    for index, value in enumerate(values):
        if isinstance(value, float) and math.isnan(value):
            return index
    best = min(values) if operation == "argmin" else max(values)
    return values.index(best)


def reference_indices(operation, dtype, words, shape, axis, strides=None):
    strides = list(strides or contiguous(shape))
    rest = [index for index in range(len(shape)) if index != axis]
    indices = []
    for row in range(math.prod(shape[index] for index in rest)):
        offset, remainder = 0, row
        for index in reversed(rest):
            offset += (remainder % shape[index]) * strides[index]
            remainder //= shape[index]
        indices.append(
            reference(
                operation,
                dtype,
                [words[offset + i * strides[axis]] for i in range(shape[axis])],
            )
        )
    return indices


def native_buffers(entry, words, outputs, metadata):
    """A callback's eight buffers: input, index output and upstream metadata."""
    types = arg_reduce_layout.operand_types(entry)
    ctype = element_storage.CTYPES[types["in"]]
    memory = {
        "in": (ctype * len(words))(*words),
        "out": (ctypes.c_uint32 * outputs)(),
    }
    for name, values in metadata.items():
        memory[name] = (arg_reduce_layout.METADATA_TYPES[types[name]] * len(values))(
            *values
        )
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
    """The arg reduction cache: one package per entry and width."""

    def __init__(self, target):
        self.target = target
        self.requests = []

    def get(self, key):
        self.requests.append(key)
        entry, _ = arg_reduce_layout.split_variant(key)
        types = arg_reduce_layout.operand_types(entry)
        # Direct3D and OpenGL prefix the scalars with the entry, once.
        prefix = "" if self.target == "metal" else entry.rstrip("_") + "_"
        bindings = []
        for name, kind in types.items():
            value = element_storage.packet(kind, self.target, [])
            scalar = {
                "elementType": value["dtype"],
                "elementStrideBytes": element_storage.PHYSICAL_SIZES[value["dtype"]],
            }
            if self.target == "directx" and kind == "float16":
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


def run(monkeypatch, tmp_path, target, entry, words, shape, axis, *, fault=None):
    operation, dtype = arg_reduce_layout.signature(entry)
    metadata, launch, rows, span = plan_for(shape, axis)
    assert len(words) == span
    array, memory = native_buffers(entry, words, rows, metadata)
    host = SimpleNamespace(
        arg_reductions=FakeCache(target),
        target=target,
        trace=tmp_path / "dispatch.jsonl",
        trace_values=True,
        dispatch_count=0,
    )

    def execute(_, request):
        inputs = request["inputs"]
        values = element_storage.unpack(dtype, target, inputs["inBinding"]["values"])
        initial = element_storage.unpack(
            "uint32", target, inputs["outBinding"]["values"]
        )
        assert inputs["axis_sizeBinding"]["values"] == metadata["axis_size"]
        assert request["execution"] == launch
        result = arg_reduce_evidence.expected_indices(
            operation, dtype, values, metadata
        )
        result += initial[rows:]
        if fault == "guard":
            result[-1] ^= 1
        return SimpleNamespace(
            status="ok",
            outputs={"outBinding": element_storage.packet("uint32", target, result)},
            details={"request": {}},
        )

    monkeypatch.setattr(
        arg_reduce_dispatch,
        "build_native_loader_dispatch_request",
        lambda descriptor, directory, inputs, outputs, execution, **_: {
            "inputs": inputs,
            "outputs": outputs,
            "execution": execution,
        },
    )
    monkeypatch.setattr(arg_reduce_dispatch, "execute", execute)
    native = runtime.Launch(
        (ctypes.c_uint32 * 3)(*launch["workgroupCount"]),
        (ctypes.c_uint32 * 3)(*launch["workgroupSize"]),
        (ctypes.c_uint32 * 3)(0, 0, 0),
    )
    arg_reduce_dispatch.dispatch(
        host, entry, array, len(array), rows * shape[axis], native
    )
    event = json.loads(host.trace.read_text().splitlines()[-1])
    return list(memory["out"]), event, host


def audit(monkeypatch, event, target):
    """Audit an event against fake packages, skipping native module checks."""
    checked = []
    monkeypatch.setattr(arg_reduce_evidence, "require_unchanged", lambda *_: None)
    monkeypatch.setattr(
        arg_reduce_evidence, "audit_native_execution", lambda e: checked.append(target)
    )
    event = {
        **event,
        "details": {
            "request": {
                "target": target,
                "entryPoint": event["entry"],
                "dispatch": {key: event[key] for key in arg_reduce_evidence.GEOMETRY},
            }
        },
    }
    count = arg_reduce_evidence.audit_event(event, FakeCache(target))
    assert checked == [target]
    return count


def test_entries_are_the_pinned_instantiations():
    path = CONTRACTS / "arg_reduce.current-tree.translation.json"
    contract = json.loads(path.read_text(encoding="utf-8"))
    assert contract["provenance"]["commit"] == prepare.COMMIT
    assert set(arg_reduce_layout.ENTRIES) == set(contract["entries"])
    assert len(arg_reduce_layout.ENTRIES) == 24
    assert arg_reduce_layout.signature("argmax_bfloat16") == ("argmax", "bfloat16")
    for entry in ("argmin_complex64", "argmin_float64", "row_reduce_small", None):
        with pytest.raises(ValueError, match="Unsupported arg reduction"):
            arg_reduce_layout.signature(entry)


def test_widths_follow_upstream_and_key_each_variant():
    widths = {
        size: arg_reduce_layout.width(size) for size in (1, 128, 129, 1560, 10**6)
    }
    assert widths == {1: 32, 128: 32, 129: 64, 1560: 416, 10**6: 1024}
    for size in (0, -1, 2.0, True):
        with pytest.raises(ValueError, match="non-empty"):
            arg_reduce_layout.width(size)
    key = arg_reduce_layout.variant("argmin_int8", 416)
    assert key == "argmin_int8@416"
    assert arg_reduce_layout.split_variant(key) == ("argmin_int8", (416, 1, 1))
    cache = arg_reduce_packages.ArgReducePackageCache.__new__(
        arg_reduce_packages.ArgReducePackageCache
    )
    assert cache.source_entry(key) == "argmin_int8"
    assert cache.workgroup_size(key) == (416, 1, 1)
    for bad in ("argmin_int8@33", "argmin_int8@2048", "argmin_int8@", "argmin_int8"):
        with pytest.raises(ValueError):
            arg_reduce_layout.split_variant(bad)


@pytest.mark.parametrize("axis", (0, 1, 2))
def test_plan_accepts_each_axis_of_the_upstream_test_array(axis):
    metadata, launch, rows, span = plan_for(SHAPE, axis)
    expected = arg_reduce_layout.plan(metadata, span, rows, rows * SHAPE[axis])
    assert expected == launch
    assert launch["workgroupSize"] == [32, 1, 1]


def test_plan_accepts_the_flattened_array_with_placeholders():
    metadata, launch, rows, span = plan_for((1560,), 0)
    assert metadata["ndim"] == [0] and metadata["shape"] == [0] and rows == 1
    assert arg_reduce_layout.plan(metadata, span, 1, 1560) == {
        "workgroupCount": [1, 1, 1],
        "workgroupSize": [416, 1, 1],
    }
    with pytest.raises(ValueError, match="placeholders"):
        arg_reduce_layout.plan({**metadata, "in_strides": [1]}, span, 1, 1560)


def test_plan_refuses_views_outside_the_upstream_launch():
    metadata, _, rows, span = plan_for(SHAPE, 1)
    logical = rows * SHAPE[1]
    cases = (
        ({**metadata, "out_strides": [1, 10]}, span, rows, logical, "row contiguous"),
        (metadata, span + 1, rows, logical, "source span"),
        (metadata, span, rows - 1, logical, "rows"),
        (metadata, span, rows, logical + 1, "logical size"),
        ({**metadata, "ndim": [3]}, span, rows, logical, "ranks"),
        ({**metadata, "axis_size": [0]}, span, rows, logical, "bounds"),
        ({**metadata, "in_strides": [-156, 1]}, span, rows, logical, "bounds"),
    )
    for values, count, outputs, size, message in cases:
        with pytest.raises(ValueError, match=message):
            arg_reduce_layout.plan(values, count, outputs, size)
    # Two reads 2^24 apart: the pointer advanced past the 128 padded reads of
    # the 32 lanes reaches 2^31, beyond OpenGL's signed offsets.
    shape, strides = (2, 2), (1, 2**24)
    wide, _, rows, span = plan_for(shape, 1, strides)
    with pytest.raises(ValueError, match="signed 32-bit"):
        arg_reduce_layout.plan(wide, span, rows, rows * 2)
    many = {
        **metadata,
        "shape": [65536, 1],
        "in_strides": [1, 1],
        "out_strides": [1, 1],
    }
    with pytest.raises(ValueError, match="bounds"):
        arg_reduce_layout.plan(many, span, rows, logical)


@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.parametrize("dtype", DTYPES)
def test_dispatch_returns_and_audits_every_dtype(monkeypatch, tmp_path, target, dtype):
    words = sample_words(dtype, math.prod(SHAPE))
    for operation in arg_reduce_layout.OPERATIONS:
        entry = f"{operation}_{dtype}"
        for axis in (0, 2):
            out, event, host = run(
                monkeypatch, tmp_path, target, entry, words, SHAPE, axis
            )
            assert out == reference_indices(operation, dtype, words, SHAPE, axis)
            assert host.dispatch_count == 1
            assert host.arg_reductions.requests == [entry + "@32"]
            assert event["argReduceMetadata"] == plan_for(SHAPE, axis)[0]
            assert audit(monkeypatch, event, target) == len(out)


@pytest.mark.parametrize("target", ("metal", "directx"))
def test_multi_simdgroup_rows_run_where_waves_are_exact(monkeypatch, tmp_path, target):
    words = sample_words("float32", 1560)
    out, event, host = run(
        monkeypatch, tmp_path, target, "argmax_float32", words, (1560,), 0
    )
    assert out == [reference("argmax", "float32", words)]
    assert host.arg_reductions.requests == ["argmax_float32@416"]
    assert event["workgroupSize"] == [416, 1, 1]
    assert audit(monkeypatch, event, target) == 1


def test_opengl_refuses_rows_of_more_than_one_simdgroup(monkeypatch, tmp_path):
    words = sample_words("float32", 1560)
    with pytest.raises(ValueError, match="crosstl#2207"):
        run(monkeypatch, tmp_path, "opengl", "argmin_float32", words, (1560,), 0)
    words = sample_words("float32", 128)
    out, _, _ = run(monkeypatch, tmp_path, "opengl", "argmin_float32", words, (128,), 0)
    assert out == [reference("argmin", "float32", words)]


def test_indices_follow_the_kernel_nan_and_tie_rules():
    nan, inf = float("nan"), float("inf")
    meta = plan_for((6,), 0)[0]
    cases = (
        ("argmin", [3.0, 1.0, 1.0, nan, 0.5, nan], 3),
        ("argmax", [3.0, 1.0, 1.0, nan, 0.5, nan], 3),
        ("argmin", [2.0, 1.0, 1.0, 4.0, 1.0, 9.0], 1),
        ("argmax", [2.0, 9.0, 1.0, 9.0, 1.0, 9.0], 1),
        ("argmin", [inf] * 6, 0),
        ("argmax", [-inf] * 6, 0),
        ("argmax", [-inf, -inf, -0.0, 0.0, -1.0, -inf], 2),
    )
    for dtype in ("float32", "float16", "bfloat16"):
        for operation, values, index in cases:
            words = [float_word(dtype, value) for value in values]
            assert arg_reduce_evidence.expected_indices(
                operation, dtype, words, meta
            ) == [index], (dtype, operation, values)
    assert arg_reduce_evidence.expected_indices("argmin", "uint8", [255] * 6, meta) == [
        0
    ]
    assert arg_reduce_evidence.expected_indices(
        "argmax", "int64", [-(2**63)] * 5 + [-(2**63) + 1], meta
    ) == [5]
    assert arg_reduce_evidence.expected_indices("argmax", "bool_", [0] * 6, meta) == [0]
    assert arg_reduce_evidence.expected_indices(
        "argmin", "bool_", [1, 1, 0, 1, 0, 1], meta
    ) == [2]


def test_dispatch_and_audit_refuse_changed_results(monkeypatch, tmp_path):
    words = sample_words("int32", math.prod(SHAPE))
    with pytest.raises(RuntimeError, match="guard"):
        run(
            monkeypatch,
            tmp_path,
            "directx",
            "argmin_int32",
            words,
            SHAPE,
            1,
            fault="guard",
        )
    out, event, _ = run(
        monkeypatch, tmp_path, "directx", "argmin_int32", words, SHAPE, 1
    )
    tampered = {**event, "outputValues": [out[0] ^ 1, *out[1:]]}
    with pytest.raises(ValueError, match="differs from its inputs"):
        audit(monkeypatch, tampered, "directx")
    changed = {**event, "operandValues": [words[0] + 1, *words[1:]]}
    with pytest.raises(ValueError, match="upload differs"):
        audit(monkeypatch, changed, "directx")
    moved = {**event, "workgroupSize": [64, 1, 1]}
    with pytest.raises(ValueError, match="upstream plan"):
        audit(monkeypatch, moved, "directx")


def test_dispatch_refuses_incomplete_buffers(tmp_path):
    host = SimpleNamespace(arg_reductions=None, target="metal")
    with pytest.raises(ValueError, match="complete buffers"):
        arg_reduce_dispatch.dispatch(host, "argmin_float32", [object()], 8, 1, object())
    metadata, launch, rows, span = plan_for(SHAPE, 0)
    array, _ = native_buffers("argmin_float32", [0] * span, rows, metadata)
    buffers = {array[i].name.decode(): array[i] for i in range(len(array))}
    with pytest.raises(ValueError, match="buffer type"):
        arg_reduce_layout.validate("argmin_int32", buffers, rows * SHAPE[0], launch)


def test_package_configs_choose_exact_waves_or_software_subgroups(tmp_path):
    key = "argmin_float32@416"
    configs = {
        target: arg_reduce_packages.project_config(tmp_path, "out", target, key)
        for target in TARGETS
    }
    for target, config in configs.items():
        (rules,) = config.entry_workgroup_size_rules.values()
        assert {name: tuple(map(int, size)) for name, size in rules.items()} == {
            "argmin_float32": (416, 1, 1)
        }
        options = config.source_options["metal"]["target_options"]
        assert options == arg_reduce_packages.TARGET_OPTIONS
    assert {
        source: int(width)
        for source, width in configs["directx"].subgroup_width_rules.items()
    } == {arg_reduce_layout.SOURCE: 32}
    assert (
        "software_subgroup_width" not in arg_reduce_packages.TARGET_OPTIONS["directx"]
    )
    assert arg_reduce_packages.TARGET_OPTIONS["opengl"] == {
        "software_subgroup_width": 32
    }
    assert not configs["metal"].subgroup_width_rules
    assert not configs["opengl"].subgroup_width_rules
    assert [
        item.expression for item in configs["opengl"].index_range_assertions
    ] == list(arg_reduce_layout.index_assertions())
    assert not configs["metal"].index_range_assertions
    assert not configs["directx"].index_range_assertions
    assert len(arg_reduce_packages.verification_entries("metal")) == 44
    assert all(
        key.endswith("@32")
        for key in arg_reduce_packages.verification_entries("opengl")
    )


def test_runtime_and_host_adapter_route_arg_reductions():
    fake = SimpleNamespace(descriptors={}, copies=None, elementwise=None, aranges=None)
    fake.arg_reductions = object()
    assert runtime.HostRuntime._entry_available(fake, b"argmax_bfloat16") == 1
    fake.arg_reductions = None
    assert runtime.HostRuntime._entry_available(fake, b"argmax_bfloat16") == 0
    source = Path(runtime.__file__).read_text(encoding="utf-8")
    assert "if entry in arg_reduce_layout.ENTRIES:" in source
    adapter = (
        Path(runtime.__file__).with_name("backend.cpp").read_text(encoding="utf-8")
    )
    assert "void ArgReduce::eval_gpu(" in adapter
    assert '"ArgReduce",' in Path(prepare.__file__).read_text(encoding="utf-8")

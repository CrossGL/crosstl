"""RMS and layer normalization of the portable host: check every launch
against normalization.cpp's plan, dispatch each width's package and audit
the uploads, guards, weight placeholders and packages."""

import ctypes
import json
import math
import struct
from pathlib import Path
from types import SimpleNamespace

import pytest

from demos.integrations.mlx.portable_host import (
    element_storage,
    norm_dispatch,
    norm_evidence,
    norm_layout,
    norm_packages,
    prepare,
    runtime,
)

ROOT = Path(__file__).resolve().parents[5]
CONTRACTS = ROOT / "demos/integrations/mlx/contracts"
TARGETS = ("metal", "directx", "opengl")
CONTRACT_NAMES = {"rms": "rms_norm", "layer_norm": "layer_norm"}
EPS = 1e-5


def float_word(dtype, value):
    if dtype == "float32":
        return struct.unpack("<I", struct.pack("<f", value))[0]
    if dtype == "float16":
        return struct.unpack("<H", struct.pack("<e", value))[0]
    return struct.unpack("<I", struct.pack("<f", value))[0] >> 16


def word_value(dtype, word):
    if dtype == "float32":
        return struct.unpack("<f", struct.pack("<I", word))[0]
    if dtype == "float16":
        return struct.unpack("<e", struct.pack("<H", word))[0]
    return struct.unpack("<f", struct.pack("<I", word << 16))[0]


def sample_words(dtype, count, seed=0):
    return [
        float_word(dtype, (((index + seed) * 37) % 23) / 8 - 1)
        for index in range(count)
    ]


def _row_terms(entry, x):
    """One row's normalizer and centered values, as the kernels form them."""
    operation = norm_layout.signature(entry)[0]
    count = len(x)
    centered = x if operation == "rms" else [v - sum(x) / count for v in x]
    norm = 1 / math.sqrt(sum(v * v for v in centered) / count + EPS)
    return centered, norm


def reference(entry, values, counts, metadata):
    """Every output word a kernel writes, from float64 arithmetic."""
    operation, gradient, _, dtype, has_w = norm_layout.signature(entry)
    axis = metadata["axis_size"]
    decode = {
        name: [word_value(dtype, w) for w in words] for name, words in values.items()
    }
    x = decode["x"]
    weight = [decode["w"][i * metadata["w_stride"]] for i in range(axis)]
    rows = len(x) // axis
    if not gradient:
        bias = (
            [decode["b"][i * metadata["b_stride"]] for i in range(axis)]
            if "b" in decode
            else [0.0] * axis
        )
        out = []
        for row in range(rows):
            centered, norm = _row_terms(entry, x[row * axis : (row + 1) * axis])
            out += [
                float_word(dtype, weight[i] * (centered[i] * norm) + bias[i])
                for i in range(axis)
            ]
        return {"out": out}
    g = decode["g"]
    per_group = norm_layout.rows_per_group(entry, rows)
    groups = -(-rows // per_group)
    gx, gw = [], [[0.0] * axis for _ in range(groups)]
    for row in range(rows):
        xs, gs = x[row * axis : (row + 1) * axis], g[row * axis : (row + 1) * axis]
        centered, norm = _row_terms(entry, xs)
        if operation == "rms":
            mean_gwx = sum(xs[i] * weight[i] * gs[i] for i in range(axis)) / axis
            gx += [
                float_word(
                    dtype,
                    gs[i] * weight[i] * norm - xs[i] * mean_gwx * norm**3,
                )
                for i in range(axis)
            ]
            for i in range(axis):
                gw[row // per_group][i] += gs[i] * xs[i] * norm
        else:
            mean_wg = sum(weight[i] * gs[i] for i in range(axis)) / axis
            mean_wgxc = sum(weight[i] * gs[i] * centered[i] for i in range(axis)) / axis
            gx += [
                float_word(
                    dtype,
                    norm * (weight[i] * gs[i] - mean_wg)
                    - centered[i] * norm * mean_wgxc * norm * norm,
                )
                for i in range(axis)
            ]
            for i in range(axis):
                gw[row][i] = gs[i] * centered[i] * norm
    if not has_w:
        return {"gx": gx, "gw": element_storage.guard_words(dtype, 1)}
    return {"gx": gx, "gw": [float_word(dtype, v) for row in gw for v in row]}


def case(entry, axis, rows, w_stride=1, b_stride=1):
    """The values, counts and scalars ``eval_gpu`` binds for one launch."""
    _, gradient, _, dtype, has_w = norm_layout.signature(entry)
    types = norm_layout.operand_types(entry)
    span = (axis - 1) * w_stride + 1 if w_stride else 1
    values = {"x": sample_words(dtype, axis * rows), "w": sample_words(dtype, span, 3)}
    if "b" in types:
        values["b"] = sample_words(
            dtype, (axis - 1) * b_stride + 1 if b_stride else 1, 5
        )
    if "g" in types:
        values["g"] = sample_words(dtype, axis * rows, 7)
    per_group = norm_layout.rows_per_group(entry, rows)
    groups = -(-rows // per_group)
    counts = (
        {"out": axis * rows}
        if not gradient
        else {
            "gx": axis * rows,
            "gw": groups * axis if has_w else 1,
        }
    )
    metadata = {
        "eps": float_word("float32", EPS),
        "axis_size": axis,
        "w_stride": w_stride,
        "b_stride": b_stride,
        "n_rows": rows,
        "rows_per_group": per_group,
    }
    metadata = {name: value for name, value in metadata.items() if name in types}
    launch = {
        "workgroupCount": [groups, 1, 1],
        "workgroupSize": [norm_layout.width(entry, axis), 1, 1],
    }
    return values, counts, metadata, launch


def native_buffers(entry, values, counts, metadata):
    """The callback's buffers in source order."""
    types = norm_layout.operand_types(entry)
    written = norm_layout.outputs(entry)
    memory = {}
    for name, kind in types.items():
        if name in norm_layout.SCALAR_TYPES:
            memory[name] = (ctypes.c_uint32 * 1)(metadata[name])
        elif name in written:
            memory[name] = (element_storage.CTYPES[kind] * counts[name])()
        else:
            memory[name] = (element_storage.CTYPES[kind] * len(values[name]))(
                *values[name]
            )
    array = (runtime.Buffer * len(memory))(
        *(
            runtime.Buffer(
                name.encode(),
                types[name].encode(),
                ctypes.addressof(memory[name]),
                len(memory[name]),
                int(name in written),
            )
            for name in memory
        )
    )
    return array, memory


class FakeCache:
    """The normalization cache: one package per entry and width."""

    def __init__(self, target):
        self.target = target
        self.requests = []

    def get(self, key):
        self.requests.append(key)
        entry, _ = norm_layout.split_variant(key)
        # Direct3D and OpenGL prefix the scalars with the kernel, once.
        prefix = "" if self.target == "metal" else norm_layout.source_entry(entry) + "_"
        bindings = []
        for name, kind in norm_layout.operand_types(entry).items():
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
            if name in norm_layout.SCALAR_TYPES:
                member = prefix + name
            else:
                member = {"out": "out_"}.get(name, name)
            bindings.append(
                {
                    "name": name + "Binding",
                    "scalarLayout": {**scalar, "memberName": member},
                }
            )
        return {"bindings": bindings, "artifact": {"id": key}}, Path("/packages")


def run(monkeypatch, tmp_path, target, entry, axis, rows, *, fault=None, **strides):
    _, _, _, dtype, _ = norm_layout.signature(entry)
    values, counts, metadata, launch = case(entry, axis, rows, **strides)
    array, memory = native_buffers(entry, values, counts, metadata)
    host = SimpleNamespace(
        norms=FakeCache(target),
        target=target,
        trace=tmp_path / "dispatch.jsonl",
        trace_values=True,
        dispatch_count=0,
    )
    types = norm_layout.operand_types(entry)

    def execute(_, request):
        inputs = request["inputs"]
        for name, words in values.items():
            assert (
                element_storage.unpack(
                    dtype, target, inputs[name + "Binding"]["values"]
                )
                == words
            )
        for name, value in metadata.items():
            assert element_storage.unpack(
                types[name], target, inputs[name + "Binding"]["values"]
            ) == [value]
        assert request["execution"] == launch
        expected = reference(entry, values, counts, metadata)
        outputs = {}
        for name, words in expected.items():
            initial = element_storage.unpack(
                dtype, target, inputs[name + "Binding"]["values"]
            )
            assert len(initial) == counts[name] + element_storage.GUARD_COUNT
            words = words + initial[counts[name] :]
            if fault == "guard" and name == norm_layout.outputs(entry)[0]:
                words[-1] ^= 1
            if fault == "placeholder" and name == "gw":
                words[0] ^= 1
            outputs[name + "Binding"] = element_storage.packet(dtype, target, words)
        return SimpleNamespace(status="ok", outputs=outputs, details={"request": {}})

    monkeypatch.setattr(
        norm_dispatch,
        "build_native_loader_dispatch_request",
        lambda descriptor, directory, inputs, outputs, execution, **_: {
            "inputs": inputs,
            "outputs": outputs,
            "execution": execution,
        },
    )
    monkeypatch.setattr(norm_dispatch, "execute", execute)
    native = runtime.Launch(
        (ctypes.c_uint32 * 3)(*launch["workgroupCount"]),
        (ctypes.c_uint32 * 3)(*launch["workgroupSize"]),
        (ctypes.c_uint32 * 3)(0, 0, 0),
    )
    norm_dispatch.dispatch(host, entry, array, len(array), axis * rows, native)
    event = json.loads(host.trace.read_text().splitlines()[-1])
    written = {name: list(memory[name]) for name in norm_layout.outputs(entry)}
    return written, event, host, (values, counts, metadata)


def audit(monkeypatch, event, target):
    """Audit an event against fake packages, skipping native module checks."""
    checked = []
    monkeypatch.setattr(norm_evidence, "require_unchanged", lambda *_: None)
    monkeypatch.setattr(
        norm_evidence, "audit_native_execution", lambda e: checked.append(target)
    )
    event = {
        **event,
        "details": {
            "request": {
                "target": target,
                "entryPoint": norm_layout.source_entry(event["entry"]),
                "dispatch": {key: event[key] for key in norm_evidence.GEOMETRY},
            }
        },
    }
    retained = norm_evidence.audit_event(event, FakeCache(target))
    assert checked == [target]
    return retained


def test_entries_are_the_pinned_instantiations():
    for operation, source in norm_layout.SOURCES.items():
        path = CONTRACTS / f"{CONTRACT_NAMES[operation]}.current-tree.translation.json"
        contract = json.loads(path.read_text(encoding="utf-8"))
        assert contract["provenance"]["commit"] == prepare.COMMIT
        assert contract["provenance"]["source"] == source
        assert {
            norm_layout.source_entry(entry)
            for entry, signature in norm_layout.ENTRIES.items()
            if signature[0] == operation
        } == set(contract["entries"])
    assert len(norm_layout.ENTRIES) == 36
    assert all(entry.startswith(norm_layout.PREFIXES) for entry in norm_layout.ENTRIES)
    assert (
        norm_layout.source_entry("vjp_rms_loopedfloat16_now") == "vjp_rms_loopedfloat16"
    )
    assert norm_layout.constants("vjp_layer_normbfloat16_w") == {"has_w": True}
    assert norm_layout.constants("vjp_rmsfloat32_now") == {"has_w": False}
    assert norm_layout.constants("layer_norm_loopedfloat32") == {}


def test_kernels_and_widths_follow_upstream_and_key_each_variant():
    name = norm_layout.entry_name
    assert name("rms", False, "float32", 4096) == "rmsfloat32"
    assert name("rms", False, "float16", 4097) == "rms_loopedfloat16"
    assert name("layer_norm", False, "float32", 6656) == "layer_normfloat32"
    assert name("layer_norm", False, "float32", 6657) == "layer_norm_loopedfloat32"
    assert (
        name("layer_norm", True, "bfloat16", 8192, True) == "vjp_layer_normbfloat16_w"
    )
    assert name("layer_norm", True, "float32", 8193, False) == (
        "vjp_layer_norm_loopedfloat32_now"
    )
    assert name("rms", True, "float32", 4097, True) == "vjp_rms_loopedfloat32_w"
    for args in (("rms", True, "float32", 32), ("rms", False, "float32", 32, True)):
        with pytest.raises(ValueError):
            name(*args)
    width = norm_layout.width
    # RMS reads four values per thread, layer normalization eight.
    assert [width("rmsfloat32", n) for n in (1, 31, 128, 129, 4096)] == [
        32,
        32,
        32,
        64,
        1024,
    ]
    assert [width("layer_normfloat32", n) for n in (8, 256, 257, 4099, 6656)] == [
        32,
        32,
        64,
        544,
        832,
    ]
    assert width("vjp_layer_normfloat32_w", 8192) == 1024
    assert width("rms_loopedfloat32", 4097) == 1024
    for entry, size in (
        ("rmsfloat32", 4097),
        ("rms_loopedfloat32", 4096),
        ("rmsfloat32", 0),
    ):
        with pytest.raises(ValueError):
            width(entry, size)
    assert norm_layout.variant("rmsfloat32", 64) == "rmsfloat32@64"
    assert norm_layout.split_variant("vjp_rms_loopedfloat32_w@1024") == (
        "vjp_rms_loopedfloat32_w",
        (1024, 1, 1),
    )
    for key in ("rmsfloat32@48", "rms_loopedfloat32@512", "vjp_rmsfloat32@32", 7):
        with pytest.raises(ValueError):
            norm_layout.split_variant(key)
    # vjp_rms covers ceil(rows / 512) rows per threadgroup; the others one.
    assert norm_layout.rows_per_group("vjp_rmsfloat32_w", 800) == 2
    assert norm_layout.rows_per_group("vjp_rmsfloat32_w", 512) == 1
    assert norm_layout.rows_per_group("vjp_rms_loopedfloat32_w", 800) == 1
    assert norm_layout.rows_per_group("vjp_layer_normfloat32_w", 800) == 1


def test_plan_refuses_launches_outside_upstream(monkeypatch):
    plan = norm_layout.plan
    entry = "vjp_rmsfloat32_w"
    _, counts, metadata, launch = case(entry, 32, 800)
    counts = {**counts, "x": 32 * 800, "w": 32, "g": 32 * 800}
    counts.update({name: 1 for name in metadata})
    assert plan(entry, metadata, counts, 32 * 800) == {
        "workgroupCount": [400, 1, 1],
        "workgroupSize": [32, 1, 1],
    }
    refused = (
        ({**metadata, "rows_per_group": 1}, counts, 32 * 800),  # wrong grouping
        ({**metadata, "n_rows": 799}, counts, 32 * 800),  # wrong rows
        (metadata, {**counts, "gw": 800 * 32}, 32 * 800),  # gw is per group
        (metadata, {**counts, "x": 32 * 800 + 1}, 32 * 800 + 1),  # partial row
        (metadata, counts, 32 * 799),  # logical size differs
        (metadata, {**counts, "w": 31}, 32 * 800),  # weight misses its row
        ({**metadata, "w_stride": 2}, counts, 32 * 800),  # strided weight too short
        ({**metadata, "axis_size": 4097}, counts, 32 * 800),  # looped length
    )
    for case_metadata, case_counts, logical in refused:
        with pytest.raises(ValueError):
            plan(entry, case_metadata, case_counts, logical)
    # A broadcast weight binds one element at stride zero.
    assert plan(entry, {**metadata, "w_stride": 0}, {**counts, "w": 1}, 32 * 800)
    # Without a weight, the placeholder is one element and the stride zero.
    now = "vjp_rmsfloat32_now"
    with pytest.raises(ValueError):
        plan(now, metadata, counts, 32 * 800)
    assert plan(now, {**metadata, "w_stride": 0}, {**counts, "gw": 1, "w": 1}, 32 * 800)
    forward = "layer_normfloat32"
    _, counts, metadata, _ = case(forward, 8, 70000)
    counts = {**counts, "x": 8 * 70000, "w": 8, "b": 8, **{n: 1 for n in metadata}}
    with pytest.raises(ValueError, match="group limit"):
        plan(forward, metadata, counts, 8 * 70000)
    # The element bound keeps every offset signed; a wider bound would not.
    monkeypatch.setattr(norm_layout, "MAX_ELEMENTS", 2**33)
    looped = "rms_loopedfloat32"
    rows, axis = 2**4, 2**27
    counts = {"x": rows * axis, "w": axis, "out": rows * axis}
    counts.update({"eps": 1, "axis_size": 1, "w_stride": 1})
    metadata = {"eps": 0, "axis_size": axis, "w_stride": 1}
    with pytest.raises(ValueError, match="signed 32-bit"):
        plan(looped, metadata, counts, rows * axis)


@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.parametrize(
    "entry,axis,rows,strides",
    (
        ("rmsfloat32", 31, 3, {}),
        ("rmsfloat16", 32, 2, {"w_stride": 0}),
        ("rms_loopedbfloat16", 4099, 1, {}),
        ("layer_normfloat32", 33, 2, {"b_stride": 0}),
        ("layer_norm_loopedfloat16", 6660, 1, {"w_stride": 2}),
        ("vjp_rmsfloat32_w", 8, 1030, {}),
        ("vjp_rmsfloat32_now", 32, 4, {"w_stride": 0}),
        ("vjp_rms_loopedfloat32_w", 4100, 2, {}),
        ("vjp_layer_normfloat32_w", 8, 4, {}),
        ("vjp_layer_norm_loopedfloat32_now", 8200, 1, {"w_stride": 0}),
    ),
)
def test_dispatch_returns_and_audits_each_kernel(
    monkeypatch, tmp_path, target, entry, axis, rows, strides
):
    written, event, host, (values, counts, metadata) = run(
        monkeypatch, tmp_path, target, entry, axis, rows, **strides
    )
    assert written == reference(entry, values, counts, metadata)
    assert host.dispatch_count == 1
    width = norm_layout.width(entry, axis)
    assert host.norms.requests == [f"{entry}@{width}"]
    assert event["normMetadata"] == metadata
    assert audit(monkeypatch, event, target) is True


def test_long_launches_are_compacted_and_audited_by_extent(monkeypatch, tmp_path):
    for module in (norm_dispatch, norm_evidence):
        monkeypatch.setattr(module, "RETAINED_LIMIT", 64)
    entry = "vjp_layer_normfloat32_w"
    written, event, _, _ = run(monkeypatch, tmp_path, "opengl", entry, 300, 2)
    assert "compactedValues" in event["operandValues"]["x"]
    assert audit(monkeypatch, event, "opengl") is False
    resized = {**event, "normMetadata": {**event["normMetadata"], "axis_size": 200}}
    with pytest.raises(ValueError):
        audit(monkeypatch, resized, "opengl")
    monkeypatch.setattr(norm_evidence, "RETAINED_LIMIT", 10**6)
    with pytest.raises(ValueError, match="retention"):
        audit(monkeypatch, event, "opengl")


def test_dispatch_and_audit_refuse_changed_results(monkeypatch, tmp_path):
    entry = "vjp_rmsfloat32_now"
    with pytest.raises(RuntimeError, match="guard"):
        run(monkeypatch, tmp_path, "directx", entry, 32, 2, fault="guard", w_stride=0)
    with pytest.raises(RuntimeError, match="placeholder"):
        run(
            monkeypatch,
            tmp_path,
            "directx",
            entry,
            32,
            2,
            fault="placeholder",
            w_stride=0,
        )
    written, event, _, _ = run(
        monkeypatch, tmp_path, "directx", entry, 32, 2, w_stride=0
    )
    outputs = event["outputValues"]
    tampered = {
        **event,
        "outputValues": {**outputs, "gx": [outputs["gx"][0] ^ 1, *outputs["gx"][1:]]},
    }
    with pytest.raises(ValueError, match="output hash"):
        audit(monkeypatch, tampered, "directx")
    placeholder = {**event, "outputValues": {**outputs, "gw": [0]}}
    with pytest.raises(ValueError, match="placeholder"):
        audit(monkeypatch, placeholder, "directx")
    values = event["operandValues"]
    changed = {
        **event,
        "operandValues": {**values, "g": [values["g"][0] ^ 1, *values["g"][1:]]},
    }
    with pytest.raises(ValueError, match="upload of g differs"):
        audit(monkeypatch, changed, "directx")
    moved = {**event, "workgroupSize": [64, 1, 1]}
    with pytest.raises(ValueError, match="upstream plan"):
        audit(monkeypatch, moved, "directx")
    rescaled = {**event, "normMetadata": {**event["normMetadata"], "eps": 0}}
    with pytest.raises(ValueError, match="upload of eps differs"):
        audit(monkeypatch, rescaled, "directx")
    missing = {**event, "operandValues": {"x": values["x"]}}
    with pytest.raises(ValueError, match="element buffers"):
        audit(monkeypatch, missing, "directx")


def test_dispatch_refuses_incomplete_buffers_and_subnormal_bfloat(
    monkeypatch, tmp_path
):
    host = SimpleNamespace(norms=None, target="metal")
    with pytest.raises(ValueError, match="complete buffers"):
        norm_dispatch.dispatch(host, "rmsfloat32", [object()], 6, 1, object())
    values, counts, metadata, launch = case("rmsfloat32", 32, 2)
    array, _ = native_buffers("rmsfloat32", values, counts, metadata)
    buffers = {array[i].name.decode(): array[i] for i in range(len(array))}
    with pytest.raises(ValueError, match="buffer type"):
        norm_layout.validate("rmsfloat16", buffers, 64, launch)
    with pytest.raises(ValueError, match="upstream plan"):
        norm_layout.validate(
            "rmsfloat32", buffers, 64, {**launch, "workgroupSize": [64, 1, 1]}
        )
    entry = "rmsbfloat16"
    values, counts, metadata, launch = case(entry, 32, 1)
    values["x"][0] = 1
    array, _ = native_buffers(entry, values, counts, metadata)
    native = runtime.Launch(
        (ctypes.c_uint32 * 3)(*launch["workgroupCount"]),
        (ctypes.c_uint32 * 3)(*launch["workgroupSize"]),
        (ctypes.c_uint32 * 3)(0, 0, 0),
    )
    host = SimpleNamespace(norms=FakeCache("directx"), target="directx")
    with pytest.raises(ValueError, match="crosstl#2000"):
        norm_dispatch.dispatch(host, entry, array, len(array), 32, native)


def test_package_configs_choose_exact_waves_or_software_subgroups(tmp_path):
    for key in ("layer_normfloat16@544", "vjp_rms_loopedfloat32_now@1024"):
        entry, workgroup = norm_layout.split_variant(key)
        source = norm_layout.source(entry)
        kernel = norm_layout.source_entry(entry)
        configs = {
            target: norm_packages.project_config(tmp_path, "out", target, key)
            for target in TARGETS
        }
        for config in configs.values():
            assert list(config.include_patterns) == [source]
            assert dict(config.entry_points) == {source: (kernel,)}
            (rules,) = config.entry_workgroup_size_rules.values()
            assert {name: tuple(map(int, size)) for name, size in rules.items()} == {
                kernel: workgroup
            }
            assert dict(config.specialization_constants) == norm_layout.constants(entry)
            assert config.freeze_specialization_constants == bool(
                norm_layout.constants(entry)
            )
            options = config.source_options["metal"]["target_options"]
            assert options == norm_packages.TARGET_OPTIONS
        assert {
            path: int(width)
            for path, width in configs["directx"].subgroup_width_rules.items()
        } == {source: 32}
        assert not configs["metal"].subgroup_width_rules
        assert not configs["opengl"].subgroup_width_rules
        assert [
            item.expression for item in configs["opengl"].index_range_assertions
        ] == list(norm_layout.index_assertions(entry))
        assert not configs["metal"].index_range_assertions
        assert not configs["directx"].index_range_assertions
    keys = norm_packages.verification_entries()
    assert keys == sorted(set(keys))
    # The 4099-value rows run single-row layer normalization in 544 threads
    # and looped RMS normalization; 8192-value gradients of layer
    # normalization stay single-row at 1024 threads.
    assert {
        "layer_normfloat32@544",
        "rms_loopedfloat32@1024",
        "vjp_layer_normfloat32_w@1024",
        "vjp_layer_normfloat32_now@1024",
        "vjp_rmsfloat32_w@64",
    } <= set(keys)
    for key in keys:
        norm_layout.split_variant(key)


def test_runtime_and_host_adapter_route_normalization():
    fake = SimpleNamespace(
        descriptors={},
        copies=None,
        elementwise=None,
        aranges=None,
        arg_reductions=None,
        softmaxes=None,
        sorts=None,
    )
    fake.norms = object()
    assert (
        runtime.HostRuntime._entry_available(fake, b"vjp_layer_norm_loopedfloat16_w")
        == 1
    )
    fake.norms = None
    assert runtime.HostRuntime._entry_available(fake, b"rmsfloat32") == 0
    source = Path(runtime.__file__).read_text(encoding="utf-8")
    assert "if entry in norm_layout.ENTRIES:" in source
    here = Path(runtime.__file__).parent
    adapter = (here / "normalization.inc").read_text(encoding="utf-8")
    for primitive in ("RMSNorm", "RMSNormVJP", "LayerNorm", "LayerNormVJP"):
        assert f"void {primitive}::eval_gpu(" in adapter
    for primitive in ("RMSNorm", "LayerNorm"):
        assert f"bool {primitive}::use_fallback(Stream s) {{" in adapter
    # Column sums read any rank's dense rows as upstream's (rows, axis_size)
    # plan, so a batched bias gradient sums every row rather than axis 0 only.
    start = adapter.index("void sum_norm_rows(")
    body = adapter[start : adapter.index("\n}\n", start)]
    assert "rows.size() % axis_size != 0 || out.size() != size_t(axis_size)" in body
    assert "array view({int(count), int(axis_size)}" in body
    assert body.rstrip().endswith("Reduce(s, Reduce::Sum, {0}).eval_gpu({view}, out);")
    backend = (here / "backend.cpp").read_text(encoding="utf-8")
    # The normalization ports come first among the ports included at the end.
    assert backend.rstrip().endswith(
        '#include "mlx/backend/no_gpu/crosstl_normalization.inc"\n'
        "// So does the port of the compiled elementwise kernels.\n"
        '#include "mlx/backend/no_gpu/crosstl_compiled.inc"\n'
        "// So does the port of GatherMM's gathered products.\n"
        '#include "mlx/backend/no_gpu/crosstl_gather_mm.inc"'
    )
    original = (
        "namespace fast {\nNO_GPU_USE_FALLBACK(LayerNorm)\nNO_GPU_MULTI(LayerNormVJP)\n"
        "NO_GPU_USE_FALLBACK(RMSNorm)\nNO_GPU_MULTI(RMSNormVJP)\n} // namespace fast\n"
    )
    replaced = original
    for primitive in prepare.NORMALIZATION_PRIMITIVES:
        macro = (
            "NO_GPU_USE_FALLBACK"
            if primitive in prepare.FALLBACK_PRIMITIVES
            else "NO_GPU_MULTI"
        )
        replaced = prepare.replace_once(
            replaced,
            f"{macro}({primitive})",
            f"// {primitive} is implemented by the registered native dispatch backend.",
        )
    assert "NO_GPU" not in replaced

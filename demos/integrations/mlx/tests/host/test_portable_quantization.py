"""Affine host storage, complete-group launches and native output chaining."""

import ctypes
import json
import os
import struct
from pathlib import Path
from types import SimpleNamespace

import pytest

from demos.integrations.mlx.portable_host import quantization_dispatch as dispatch
from demos.integrations.mlx.portable_host import quantization_layout as layout
from demos.integrations.mlx.portable_host import quantization_packages, runtime
from demos.integrations.mlx.portable_host import (
    verify_upstream_quantization as upstream,
)
from demos.integrations.mlx.portable_host.quantization_packages import (
    QuantizationPackageCache,
)
from demos.integrations.mlx.portable_host.verify_quantization import packed_reference


def upstream_record(mode):
    return {
        "mode": mode,
        "tests": list(upstream.UPSTREAM_TESTS),
        "testsRun": 1,
        "success": True,
        "skipped": [],
        "errors": [],
        "failures": [],
        "nativeDispatches": 110 if mode == "native" else 0,
    }


def upstream_trace(target="opengl"):
    return [
        {"entry": f"affine_{operation}_float_gs_{group}_b_{bits}", "target": target}
        for operation in ("quantize", "dequantize")
        for group in (32, 64, 128)
        for bits in (2, 3, 4, 5, 6, 8)
        for _ in range(3 if (group, bits) == (32, 4) else 2)
    ] + [
        {"entry": "all_reduce_andbool_", "target": target, "threads": count}
        for count in (65536, 131072)
        for _ in range(18)
    ]


@pytest.mark.parametrize("mode", ["cpu", "native"])
@pytest.mark.parametrize(
    "fault",
    [
        None,
        "mode",
        "tests",
        "count",
        "success",
        "skipped",
        "errors",
        "failures",
        "dispatches",
        "dispatch-type",
    ],
)
def test_upstream_affine_results_require_the_complete_unchanged_test(mode, fault):
    record = upstream_record(mode)
    if fault in {"mode", "tests"}:
        record[fault] = "other"
    elif fault == "count":
        record["testsRun"] = True
    elif fault == "success":
        record["success"] = False
    elif fault in {"skipped", "errors", "failures"}:
        record[fault] = ["not passing"]
    elif fault == "dispatches":
        record["nativeDispatches"] = 0 if mode == "native" else 1
    elif fault == "dispatch-type":
        record["nativeDispatches"] = False
    if fault:
        with pytest.raises(RuntimeError, match="did not pass"):
            upstream.validate_result(record, mode)
    else:
        upstream.validate_result(record, mode)


@pytest.mark.parametrize("target", ["metal", "opengl", "directx"])
@pytest.mark.parametrize(
    "fault",
    [None, "missing", "extra", "target", "specialization", "slices", "large-assertion"],
)
def test_upstream_affine_trace_requires_both_input_families_and_slices(target, fault):
    trace = upstream_trace(target)
    count = len(trace)
    if fault == "missing":
        trace.pop()
    elif fault == "extra":
        trace.append(trace[0])
    elif fault == "target":
        trace[0]["target"] = "other"
    elif fault == "specialization":
        trace[0]["entry"] = "all_reduce_andbool_"
    elif fault == "slices":
        trace.pop(
            next(
                i
                for i, item in enumerate(trace)
                if item["entry"] == "affine_dequantize_float_gs_32_b_4"
            )
        )
        count -= 1
    elif fault == "large-assertion":
        trace[-1]["threads"] = 65535
    if fault:
        with pytest.raises(RuntimeError):
            upstream.validate_trace(trace, target, count)
    else:
        upstream.validate_trace(trace, target, count)


@pytest.mark.parametrize("fault", [None, "random", "cpu", "native", "source"])
def test_upstream_affine_runner_retains_workers_and_source_checks(
    tmp_path, monkeypatch, fault
):
    packages = tmp_path / "packages"
    packages.mkdir()
    (packages / "index.json").write_text(json.dumps({"target": "opengl"}))
    preparations = []

    def prepared(root):
        preparations.append(root)
        return {
            "source": (
                "changed" if fault == "source" and len(preparations) > 1 else "pinned"
            )
        }

    monkeypatch.setattr(upstream, "verify_prepared", prepared)
    monkeypatch.setattr(
        upstream, "upstream_test_sources", lambda *args: {"test_quantized.py": "pinned"}
    )
    commands = []

    def execute(command, **kwargs):
        mode = (
            command[command.index("--worker") + 1]
            if "--worker" in command
            else "random"
        )
        commands.append((mode, command))
        destination = Path(command[command.index("--output-dir") + 1])
        destination.mkdir()
        if mode != "random":
            upstream.write_json(destination / "results.json", upstream_record(mode))
            if mode == "native":
                (destination / "trace.jsonl").write_text(
                    "\n".join(json.dumps(item) for item in upstream_trace())
                )
        return SimpleNamespace(returncode=int(mode == fault))

    monkeypatch.setattr(upstream.subprocess, "run", execute)
    output = tmp_path / "evidence"
    if fault:
        with pytest.raises(RuntimeError, match="workers failed|sources changed"):
            upstream.run(tmp_path, packages, tmp_path / "reductions", output)
        assert not (output / "evidence.json").exists()
    else:
        record = upstream.run(tmp_path, packages, tmp_path / "reductions", output)
        assert (
            record["fullUpstreamSuite"] is False and record["sourceUnchanged"] is True
        )
    assert [mode for mode, _ in commands] == (
        ["random"] if fault == "random" else ["random", "cpu", "native"]
    )
    for mode, command in commands:
        assert (output / f"{mode}.command.json").exists()
        assert (
            command[command.index("--timeout-seconds") + 1]
            == {"random": "600", "cpu": "180", "native": "1800"}[mode]
        )
        if mode != "random":
            assert (
                "demos.integrations.mlx.portable_host.verify_upstream_quantization"
                in command
            )
    assert len(preparations) == 2


def test_upstream_affine_ci_reuses_existing_whole_reduction_jobs():
    import yaml

    root = Path(__file__).resolve().parents[5]
    job = yaml.safe_load(
        (root / ".github/workflows/demo-project-testing.yml").read_text()
    )["jobs"]["reductions"]
    cases = [
        item for item in job["strategy"]["matrix"]["include"] if item["family"] == "all"
    ]
    assert {item["target"] for item in cases} == {"metal", "opengl", "directx"}
    assert all(item["companion_args"] == "--upstream-quantization" for item in cases)
    assert len(job["strategy"]["matrix"]["include"]) == 9
    assert job["timeout-minutes"] == 360
    execution = next(
        step for step in job["steps"] if step.get("name") == "Execute MLX reductions"
    )
    assert "${{ matrix.companion_args }}" in execution["run"]
    assert "5400" in execution["run"] and "continue-on-error" not in execution


def buffers(entry, groups, data=None):
    memory, result = {}, {}
    for name, (dtype, count, output) in layout.buffer_contract(entry, groups).items():
        ctype = dispatch.WORD_TYPES[layout.ITEM_SIZES[dtype]]
        values = data[name] if data and name in data else [0] * count
        memory[name] = (ctype * count)(*values)
        result[name] = runtime.Buffer(
            name.encode(), dtype.encode(), ctypes.addressof(memory[name]), count, output
        )
    return result, memory


def launch(entry, groups):
    return runtime.Launch((groups, 1, 1), (layout.workgroup_size(entry), 1, 1))


@pytest.mark.parametrize("dtype", layout.SOURCE_TYPES.values())
@pytest.mark.parametrize("group", (32, 64, 128))
@pytest.mark.parametrize("bits", (2, 3, 4, 5, 6, 8))
@pytest.mark.parametrize("operation", ("quantize", "dequantize"))
def test_affine_layout_covers_all_pinned_specializations(dtype, group, bits, operation):
    entry = f"affine_{operation}_{dtype}_gs_{group}_b_{bits}"
    supplied, memory = buffers(entry, 5)
    assert layout.validate(
        entry, supplied, group * 5, launch(entry, 5).execution()
    ) == {"groupCount": 5, "elementCount": group * 5}
    assert memory


@pytest.mark.parametrize(
    "fault",
    (
        "missing",
        "dtype",
        "count",
        "direction",
        "null",
        "alignment",
        "overlap",
        "elements",
        "launch",
    ),
)
def test_affine_layout_rejects_invalid_storage_before_reading(fault):
    entry = "affine_quantize_float_gs_32_b_2"
    supplied, memory = buffers(entry, 5)
    geometry, elements = launch(entry, 5).execution(), 160
    if fault == "missing":
        supplied.pop("scales")
    elif fault == "dtype":
        supplied["w"].dtype = b"float16"
    elif fault == "count":
        supplied["out"].count -= 1
    elif fault == "direction":
        supplied["biases"].output = 0
    elif fault == "null":
        supplied["w"].data = None
    elif fault == "alignment":
        supplied["w"].data += 1
    elif fault == "overlap":
        supplied["scales"].data = supplied["biases"].data
    elif fault == "elements":
        elements -= 1
    else:
        geometry["workgroupSize"][0] = 1
    with pytest.raises(ValueError):
        layout.validate(entry, supplied, elements, geometry)
    assert memory


def test_affine_layout_accepts_upstream_sizes_not_elementwise_limit():
    entry = "affine_quantize_float_gs_32_b_2"
    supplied, memory = buffers(entry, 4096)
    assert (
        layout.validate(entry, supplied, 131072, launch(entry, 4096).execution())[
            "elementCount"
        ]
        == 131072
    )
    with pytest.raises(ValueError, match="complete groups"):
        layout.buffer_contract(entry, layout.MAX_GROUPS + 1)
    assert memory


@pytest.mark.parametrize(
    "logical,physical",
    (
        ("float16", "float16"),
        ("float16", "float32"),
        ("bfloat16", "float32"),
        ("bfloat16", "bfloat16"),
        ("bfloat16", "uint16"),
    ),
)
def test_affine_narrow_transport_preserves_all_words(logical, physical):
    values = list(range(65536))
    assert (
        dispatch.decode(dispatch.encode(values, logical, physical), logical, physical)
        == values
    )


@pytest.mark.parametrize(
    "logical,physical,word",
    (
        ("uint8", "uint32", 256),
        ("bfloat16", "float32", 0x3F800001),
        ("float16", "float32", 0x3F800001),
    ),
)
def test_affine_transport_does_not_round_invalid_native_outputs(
    logical, physical, word
):
    value = dispatch.encode([0], logical, physical)
    value["values"] = [word]
    with pytest.raises((ValueError, RuntimeError)):
        dispatch.decode(value, logical, physical)


@pytest.mark.parametrize(
    "entry",
    (
        "affine_quantize_float_gs_16_b_2",
        "affine_dequantize_float_gs_32_b_7",
        "affine_quantize_int_gs_32_b_2",
        "affine_quantize_float_gs_32_b_2_extra",
    ),
)
def test_affine_entries_reject_unvalidated_specializations(entry):
    with pytest.raises(ValueError, match="Unsupported"):
        layout.signature(entry)


@pytest.mark.parametrize(
    "fault", (None, "missing", "guard", "encoding", "shape", "range", "trace")
)
@pytest.mark.parametrize("dtype", ("float32", "float16", "bfloat16"))
@pytest.mark.parametrize("target", ("metal", "directx", "opengl"))
def test_affine_dispatch_commits_all_outputs_only_after_validation(
    tmp_path, monkeypatch, fault, dtype, target
):
    entry = f"affine_quantize_{layout.SOURCE_TYPES[dtype]}_gs_32_b_2"
    supplied, memory = buffers(entry, 1)
    descriptor = {"artifact": {}, "bindings": []}
    for name, buffer in supplied.items():
        kind = buffer.dtype.decode()
        physical = kind
        if kind == "uint8" and target != "metal":
            physical = "uint32"
        elif kind in {"float16", "bfloat16"} and target != "metal":
            physical = "uint16" if target == "directx" else "float32"
        scalar = {
            "elementType": physical,
            "elementStrideBytes": (
                {"uint16": 2, "uint32": 4}.get(
                    physical, layout.ITEM_SIZES.get(physical)
                )
            ),
        }
        if kind == "float16" and target == "directx":
            scalar.update(
                storageLayout="hlsl-structured-buffer",
                runtimeSized=True,
                storageEncoding={
                    "encoding": "ieee754-binary16",
                    "logicalElementType": "float16",
                },
            )
        descriptor["bindings"].append(
            {"name": name, "kind": "buffer", "scalarLayout": scalar}
        )
    one = {"float32": 0x3F800000, "float16": 0x3C00, "bfloat16": 0x3F80}[dtype]
    host = SimpleNamespace(
        target=target,
        quantization=SimpleNamespace(get=lambda name: (descriptor, tmp_path)),
        trace=tmp_path / "trace.jsonl",
        dispatch_count=0,
    )
    monkeypatch.setattr(
        dispatch,
        "build_native_loader_dispatch_request",
        lambda descriptor, directory, inputs, outputs, execution, **kwargs: outputs,
    )

    def execute(host, outputs):
        outputs = json.loads(json.dumps(outputs))
        for name, value in outputs.items():
            count = supplied[name].count
            logical = supplied[name].dtype.decode()
            value["values"][:count] = dispatch.encode(
                [47 if name == "out" else one] * count, logical, value["dtype"]
            )["values"]
        if fault == "missing":
            outputs.pop("biases")
        elif fault == "guard":
            outputs["biases"]["values"][-1] ^= 1
        elif fault == "encoding":
            outputs["biases"]["encoding"] = "other"
        elif fault == "shape":
            outputs["biases"]["shape"] = [1]
        elif fault == "range":
            outputs["out"]["values"][0] = 256
        return SimpleNamespace(status="ok", outputs=outputs, details={})

    monkeypatch.setattr(dispatch, "execute", execute)
    if fault == "trace":
        host.trace.mkdir()
    call = lambda: dispatch.dispatch(
        host, entry, (runtime.Buffer * 4)(*supplied.values()), 4, 32, launch(entry, 1)
    )
    if fault:
        with pytest.raises((ValueError, RuntimeError, OSError)):
            call()
        assert host.dispatch_count == 0
        assert all(not any(value) for value in memory.values())
    else:
        call()
        assert list(memory["out"]) == [47] * 8
        assert list(memory["scales"]) == list(memory["biases"]) == [one]
        assert host.dispatch_count == 1


def test_affine_reference_packing_has_explicit_bit_order():
    assert packed_reference([3, 3, 2, 0], 2) == b"\x2f"
    assert packed_reference(list(range(8)), 3) == bytes.fromhex("88c6fa")
    with pytest.raises(ValueError, match="incomplete"):
        packed_reference([1], 3)
    with pytest.raises(ValueError, match="Invalid"):
        packed_reference([8] * 8, 3)


@pytest.mark.parametrize("wrong_entry", (False, True))
def test_affine_cache_verifies_selected_source_entry(
    tmp_path, monkeypatch, wrong_entry
):
    entry = "affine_quantize_float_gs_32_b_2"
    identity = {"entry": entry, "sourceHash": "a" * 64}
    descriptor = {
        "target": "opengl",
        "stage": "compute",
        "entryPoint": {"name": "main"},
        "source": {"hash": {"algorithm": "sha256", "value": identity["sourceHash"]}},
    }
    (tmp_path / "index.json").write_text(
        json.dumps({"identity": identity, "descriptor": descriptor})
    )
    loader = {
        "success": True,
        "loadUnits": [
            {"entryPoint": {"source": "another_entry" if wrong_entry else entry}}
        ],
    }
    monkeypatch.setattr(
        quantization_packages, "build_runtime_loader_manifest", lambda path: loader
    )
    monkeypatch.setattr(
        quantization_packages,
        "build_native_loader_abi_descriptor",
        lambda payload: descriptor,
    )
    cache = QuantizationPackageCache(tmp_path, tmp_path, "opengl")
    if wrong_entry:
        with pytest.raises(ValueError, match="selected entry"):
            cache._load(tmp_path, identity)
    else:
        assert cache._load(tmp_path, identity) == (descriptor, tmp_path / "package")


def test_affine_ci_uses_existing_platform_jobs_and_retains_evidence():
    import yaml

    root = Path(__file__).resolve().parents[5]
    workflow = yaml.safe_load(
        (root / ".github/workflows/demo-project-testing.yml").read_text()
    )
    jobs = workflow["jobs"]
    job = jobs["half-host"]
    assert job["needs"] == "portable-host"
    assert job["env"]["MLX_COMMIT"] == jobs["portable-host"]["env"]["MLX_COMMIT"]
    assert not any(
        item.get("name") == "Execute affine quantization through MLX"
        for item in jobs["portable-host"]["steps"]
    )
    assert {item["target"] for item in job["strategy"]["matrix"]["include"]} == {
        "metal",
        "opengl",
        "directx",
    }
    step = next(
        item
        for item in job["steps"]
        if item.get("name") == "Execute affine quantization through MLX"
    )
    assert "if" not in step
    assert "portable_host.verify_quantization" in step["run"]
    assert "--timeout-seconds 1800" in step["run"]
    assert "test_portable_quantization.py" in step["run"]
    assert "--packages .mlx-portable-half/base/packages" in step["run"]
    assert "--output-dir .mlx-portable-half/affine" in step["run"]
    assert "tee .mlx-portable-half/affine.log" in step["run"]
    assert "continue-on-error" not in step
    names = [item.get("name") for item in job["steps"]]
    assert names.index("Build adapted upstream MLX") < names.index(step["name"])
    upload = next(
        item
        for item in job["steps"]
        if item.get("name") == "Retain half execution evidence"
    )
    assert upload["with"]["path"] == ".mlx-portable-half"
    assert upload["if"] == "always()"
    assert upload["with"]["include-hidden-files"] is True
    assert upload["with"]["if-no-files-found"] == "error"


@pytest.mark.parametrize("dtype", ("float32", "float16", "bfloat16"))
def test_affine_native_host_chains_large_quantize_dequantize(tmp_path, dtype):
    root = os.environ.get("CROSTL_MLX_ROOT")
    target = os.environ.get("CROSTL_MLX_AFFINE_TARGET")
    if not root or not target:
        if os.environ.get("CROSTL_REQUIRE_MLX_AFFINE_NATIVE") == "1":
            pytest.fail("Pinned source and affine native target are required")
        pytest.skip("Affine native source and target are not configured")
    from tests.test_translator.test_native_loader_dispatch_integration import _executor

    executor = _executor(target)
    host = SimpleNamespace(
        target=target,
        quantization=QuantizationPackageCache(root, tmp_path / "packages", target),
        executor=executor,
        trace=tmp_path / "trace.jsonl",
        dispatch_count=0,
    )
    source = layout.SOURCE_TYPES[dtype]
    quantize = f"affine_quantize_{source}_gs_32_b_2"
    dequantize = f"affine_dequantize_{source}_gs_32_b_2"
    groups, size = 4096, 131072

    def word(value):
        if dtype == "float16":
            return struct.unpack("<H", struct.pack("<e", value))[0]
        bits = struct.unpack("<I", struct.pack("<f", value))[0]
        return bits >> 16 if dtype == "bfloat16" else bits

    supplied, memory = buffers(
        quantize,
        groups,
        {"w": [word(value) for value in (0.0, 0.5, 1.5, 3.0)] * (size // 4)},
    )
    array = (runtime.Buffer * 4)(*supplied.values())
    try:
        runtime.HostRuntime.dispatch(
            host, quantize, array, 4, size, launch=launch(quantize, groups)
        )
        assert list(memory["out"]) == [47] * (size // 4)
        assert list(memory["scales"]) == [word(-1.0)] * groups
        assert list(memory["biases"]) == [word(3.0)] * groups
        inputs = {
            "w": list(memory["out"]),
            "scales": list(memory["scales"]),
            "biases": list(memory["biases"]),
        }
        supplied, decoded = buffers(dequantize, groups, inputs)
        runtime.HostRuntime.dispatch(
            host,
            dequantize,
            (runtime.Buffer * 4)(*supplied.values()),
            4,
            size,
            launch=launch(dequantize, groups),
        )
        assert list(decoded["out"]) == [
            word(value) for value in (0.0, 0.0, 1.0, 3.0)
        ] * (size // 4)
        assert host.dispatch_count == 2
        events = [json.loads(line) for line in host.trace.read_text().splitlines()]
        assert [event["entry"] for event in events] == [quantize, dequantize]
        assert all(
            event["quantizationMetadata"]["elementCount"] == size for event in events
        )
        assert set(events[0]["outputHashes"]) == {"out", "scales", "biases"}
    finally:
        close = getattr(executor.runtime_adapter.runtime, "close", None)
        if close:
            close()

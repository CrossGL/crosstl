"""Block scale storage, specialization selection and atomic native readback."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from demos.integrations.mlx.portable_host import quantization_dispatch as dispatch
from demos.integrations.mlx.portable_host import quantization_layout as layout
from demos.integrations.mlx.portable_host import quantization_packages, runtime
from demos.integrations.mlx.portable_host import (
    verify_block_quantization as verification,
)
from demos.integrations.mlx.portable_host.verify_block_quantization import (
    FORMATS,
    cases,
    reference,
)
from demos.integrations.mlx.tests.host.test_portable_quantization import buffers, launch


def entry_name(mode, operation, dtype, group, bits, global_scale):
    return (
        f"{mode}_{operation}_{layout.SOURCE_TYPES[dtype]}_gs_{group}_b_{bits}"
        f"_hgs_{str(global_scale).lower()}"
    )


@pytest.mark.parametrize("mode,group,bits,global_scale", FORMATS)
@pytest.mark.parametrize("dtype", layout.SOURCE_TYPES)
@pytest.mark.parametrize("operation", ("quantize", "dequantize"))
def test_block_layout_matches_every_pinned_specialization(
    mode, group, bits, global_scale, dtype, operation
):
    entry = entry_name(mode, operation, dtype, group, bits, global_scale)
    assert layout.signature(entry) == (operation, dtype, group, bits)
    assert layout.workgroup_size(entry) == (
        group if operation == "quantize" else group * bits // 8
    )
    contract = layout.buffer_contract(entry, 7)
    assert contract["scales"] == ("uint8", 7, int(operation == "quantize"))
    assert set(contract) == {"w", "out", "scales"} | (
        {"global_scale"} if global_scale else set()
    )
    if global_scale:
        assert contract["global_scale"] == ("float32", 1, 0)
    supplied, memory = buffers(entry, 7)
    assert layout.validate(
        entry, supplied, 7 * group, launch(entry, 7).execution()
    ) == {
        "groupCount": 7,
        "elementCount": 7 * group,
    }
    assert memory
    host = SimpleNamespace(descriptors={}, quantization=object())
    assert runtime.HostRuntime._entry_available(host, entry.encode()) == 1
    host.quantization = None
    assert runtime.HostRuntime._entry_available(host, entry.encode()) == 0


@pytest.mark.parametrize(
    "entry",
    (
        "mxfp4_quantize_float_gs_16_b_4_hgs_false",
        "mxfp4_quantize_float_gs_32_b_8_hgs_false",
        "mxfp8_dequantize_float_gs_32_b_8_hgs_true",
        "nvfp4_quantize_float_gs_32_b_4_hgs_false",
        "nvfp4_quantize_float_gs_16_b_4",
        "nvfp4_quantize_float_gs_16_b_4_hgs_True",
        "nvfp4_quantize_int_gs_16_b_4_hgs_true",
    ),
)
def test_block_entry_rejects_unvalidated_specializations(entry):
    with pytest.raises(ValueError, match="Unsupported"):
        layout.signature(entry)
    host = SimpleNamespace(descriptors={}, quantization=object())
    assert runtime.HostRuntime._entry_available(host, entry.encode()) == 0


@pytest.mark.parametrize(
    "fault",
    (
        "bias",
        "scale-dtype",
        "scale-count",
        "scale-direction",
        "global-direction",
        "global-count",
        "global-alignment",
        "overlap",
        "null",
        "incomplete",
        "geometry",
    ),
)
@pytest.mark.parametrize("operation", ("quantize", "dequantize"))
def test_block_binding_rejects_invalid_storage_before_dispatch(fault, operation):
    entry = entry_name("nvfp4", operation, "float32", 16, 4, True)
    supplied, memory = buffers(entry, 7)
    execution = launch(entry, 7).execution()
    count = 112
    if fault == "bias":
        supplied["biases"] = supplied["scales"]
    elif fault == "scale-dtype":
        supplied["scales"].dtype = b"float32"
    elif fault == "scale-count":
        supplied["scales"].count += 1
    elif fault == "scale-direction":
        supplied["scales"].output ^= 1
    elif fault == "global-direction":
        supplied["global_scale"].output = 1
    elif fault == "global-count":
        supplied["global_scale"].count = 2
    elif fault == "global-alignment":
        supplied["global_scale"].data += 1
    elif fault == "overlap":
        supplied["out"].data = supplied["global_scale"].data
    elif fault == "null":
        supplied["scales"].data = 0
    elif fault == "incomplete":
        count -= 1
    elif fault == "geometry":
        execution["workgroupSize"][0] *= 2
    with pytest.raises(ValueError):
        layout.validate(entry, supplied, count, execution)
    assert memory


@pytest.mark.parametrize("fault", (None, "missing", "guard", "range", "trace"))
@pytest.mark.parametrize("global_scale", (False, True))
def test_block_outputs_are_committed_together(
    tmp_path, monkeypatch, fault, global_scale
):
    entry = entry_name("nvfp4", "quantize", "float32", 16, 4, global_scale)
    supplied, memory = buffers(entry, 1)
    descriptor = {
        "artifact": {},
        "bindings": [
            {
                "name": name,
                "kind": "buffer",
                "scalarLayout": {
                    "elementType": "uint32" if value.dtype == b"uint8" else "float32",
                    "elementStrideBytes": 4,
                },
            }
            for name, value in supplied.items()
        ],
    }
    host = SimpleNamespace(
        target="opengl",
        dispatch_count=0,
        trace=tmp_path / "trace.jsonl",
        quantization=SimpleNamespace(get=lambda name: (descriptor, tmp_path)),
    )
    monkeypatch.setattr(
        dispatch,
        "build_native_loader_dispatch_request",
        lambda descriptor, directory, inputs, outputs, execution, **kw: outputs,
    )

    def execute(host, outputs):
        outputs = json.loads(json.dumps(outputs))
        outputs["out"]["values"][:8] = [0x21] * 8
        outputs["scales"]["values"][0] = 56
        if fault == "missing":
            outputs.pop("scales")
        elif fault == "guard":
            outputs["scales"]["values"][-1] ^= 1
        elif fault == "range":
            outputs["scales"]["values"][0] = 256
        return SimpleNamespace(status="ok", outputs=outputs, details={})

    monkeypatch.setattr(dispatch, "execute", execute)
    if fault == "trace":
        host.trace.mkdir()
    count = len(supplied)
    arguments = (
        host,
        entry,
        (runtime.Buffer * count)(*supplied.values()),
        count,
        16,
        launch(entry, 1),
    )
    if fault:
        with pytest.raises((ValueError, RuntimeError, OSError)):
            dispatch.dispatch(*arguments)
        assert host.dispatch_count == 0
        assert all(not any(value) for value in memory.values())
    else:
        dispatch.dispatch(*arguments)
        assert list(memory["out"]) == [0x21] * 8
        assert list(memory["scales"]) == [56]
        assert host.dispatch_count == 1
        assert not any(memory["w"])
        if global_scale:
            assert not any(memory["global_scale"])
        assert set(json.loads(host.trace.read_text())["outputHashes"]) == {
            "out",
            "scales",
        }


@pytest.mark.parametrize("target", ("metal", "opengl", "directx"))
@pytest.mark.parametrize("mode,group,bits,global_scale", FORMATS)
def test_block_package_recipe_uses_the_pinned_source_and_explicit_profiles(
    tmp_path, monkeypatch, target, mode, group, bits, global_scale
):
    class ObservedRecipe(Exception):
        pass

    entry = entry_name(mode, "quantize", "float32", group, bits, global_scale)

    def translate(config, **kwargs):
        assert config.include_patterns == [quantization_packages.BLOCK_SOURCE]
        assert config.entry_points == {quantization_packages.BLOCK_SOURCE: (entry,)}
        assert config.workgroup_size == (group, 1, 1)
        options = config.source_options["metal"]
        assert options["binary32_multiplication_profile"] == "rne-flush"
        assert options["binary32_log2_operand_profile"] == "flush-subnormals"
        assert options["binary32_log2_accuracy_profile"] == "portable-finite"
        if target != "metal":
            assert options["target_options"][target]["software_subgroup_width"] == 32
        assert bool(config.index_range_assertions) == (target == "opengl")
        raise ObservedRecipe

    monkeypatch.setattr(quantization_packages, "translate_project", translate)
    cache = quantization_packages.QuantizationPackageCache(tmp_path, tmp_path, target)
    with pytest.raises(ObservedRecipe):
        cache._build(entry, tmp_path)


def test_block_package_rejects_unsupported_target(tmp_path):
    with pytest.raises(ValueError, match="^Unsupported quantization target$"):
        quantization_packages.QuantizationPackageCache(tmp_path, tmp_path, "unknown")


def test_block_references_cover_every_host_dtype_and_global_scale():
    assert len(cases()) == 25
    assert sum(case["offset"] for case in cases()) == 13
    assert cases()[-1]["groups"] == 65536
    for case in cases():
        values, packed, scales = reference(case["mode"], 7, case["globalScale"])
        assert len(values) == len(scales) == 7
        assert len(packed) == 7 * case["group"] * case["bits"] // 8
        assert scales[-1] == 0 and values[-1] == [0.0] * case["group"]
        assert scales[0] != scales[1]
    assert reference("mxfp4")[1][:4] == bytes.fromhex("10325476")
    assert reference("mxfp8")[1][:4] == bytes((126, 254, 48, 176))
    assert reference("nvfp4", global_scale=True)[2][:5] == bytes((48, 56, 64, 72, 80))


@pytest.mark.parametrize(
    "fault", (None, "access", "kind", "dtype", "stride", "duplicate")
)
def test_inactive_global_scale_keeps_the_pinned_kernel_abi(
    tmp_path, monkeypatch, fault
):
    entry = entry_name("mxfp4", "quantize", "float32", 32, 4, False)
    supplied, memory = buffers(entry, 1)
    inactive = {
        "name": "global_scale",
        "kind": "buffer",
        "access": "read",
        "scalarLayout": {"elementType": "float32", "elementStrideBytes": 4},
    }
    if fault in {"access", "kind"}:
        inactive[fault] = "other"
    elif fault == "dtype":
        inactive["scalarLayout"]["elementType"] = "uint32"
    elif fault == "stride":
        inactive["scalarLayout"]["elementStrideBytes"] = 8
    descriptor = {"bindings": [inactive] * (2 if fault == "duplicate" else 1)}
    host = SimpleNamespace(
        quantization=SimpleNamespace(get=lambda name: (descriptor, tmp_path))
    )
    if fault:
        message = "Inactive global scale reflection"
    else:
        # The inactive parameter is accepted; the deliberately absent real bindings are not.
        message = "reflected buffers are incomplete"
    with pytest.raises(ValueError, match=message):
        dispatch.dispatch(
            host,
            entry,
            (runtime.Buffer * 3)(*supplied.values()),
            3,
            32,
            launch(entry, 1),
        )
    assert all(not any(value) for value in memory.values())


def test_block_ci_requires_native_execution_in_existing_platform_jobs():
    import yaml

    root = Path(__file__).resolve().parents[5]
    job = yaml.safe_load(
        (root / ".github/workflows/demo-project-testing.yml").read_text()
    )["jobs"]["half-host"]
    assert {item["target"] for item in job["strategy"]["matrix"]["include"]} == {
        "metal",
        "opengl",
        "directx",
    }
    assert job["timeout-minutes"] == 260
    steps = job["steps"]
    step = next(
        item
        for item in steps
        if item.get("name") == "Execute block quantization through MLX"
    )
    assert "if" not in step and "continue-on-error" not in step
    assert "--timeout-seconds 1800" in step["run"]
    assert "portable_host.verify_block_quantization" in step["run"]
    assert "--packages .mlx-portable-half/base/packages" in step["run"]
    assert "--output-dir .mlx-portable-half/block" in step["run"]
    assert step["env"]["CROSTL_DIRECTX_FORCE_WARP"] == "1"
    upload = next(
        item for item in steps if item.get("name") == "Retain half execution evidence"
    )
    assert upload["with"]["path"] == ".mlx-portable-half" and upload["if"] == "always()"


@pytest.mark.parametrize("target", ("metal", "opengl", "directx"))
@pytest.mark.parametrize("fault", (None, "compile", "empty", "source-change"))
def test_block_toolchains_require_exact_sources_and_nonempty_modules(
    tmp_path, monkeypatch, target, fault
):
    source = tmp_path / "source"
    source.write_text("void main() {}")
    descriptor = {
        "artifact": {
            "packagePath": source.name,
            "hash": {"value": hashlib.sha256(source.read_bytes()).hexdigest()},
        },
        "entryPoint": {"name": "CSMain" if target == "directx" else "main"},
    }
    calls = []

    def execute(command, **kwargs):
        calls.append(command)
        assert kwargs["timeout"] == 120 and kwargs["check"] is False
        if command[0] != "spirv-val":
            module = Path(
                command[command.index("-Fo" if target == "directx" else "-o") + 1]
            )
            if fault != "empty":
                module.write_bytes(b"compiled")
            if fault == "source-change":
                source.write_text("changed")
        return SimpleNamespace(
            returncode=int(fault == "compile"),
            stdout="output",
            stderr="error" if fault == "compile" else "",
        )

    monkeypatch.setattr(verification.subprocess, "run", execute)
    output = tmp_path / "compilation"
    if fault:
        with pytest.raises((ValueError, RuntimeError)):
            verification.compile_entry(target, "entry", descriptor, tmp_path, output)
    else:
        verification.compile_entry(target, "entry", descriptor, tmp_path, output)
    record = json.loads((output / "entry.json").read_text())
    assert record["status"] == ("failed" if fault else "passed")
    assert record["steps"][0]["stdout"] == "output"
    if target == "metal":
        assert "-Werror" in calls[0] and "-fno-fast-math" in calls[0]
    elif target == "directx":
        assert "-WX" in calls[0] and "cs_6_6" in calls[0] and "2021" in calls[0]
    elif fault != "compile":
        assert calls[0][0] == "glslangValidator" and calls[1][:3] == [
            "spirv-val",
            "--target-env",
            "spv1.3",
        ]


def test_block_spirv_validation_failure_is_not_a_compilation_pass(
    tmp_path, monkeypatch
):
    source = tmp_path / "source.glsl"
    source.write_text("void main() {}")
    descriptor = {
        "artifact": {
            "packagePath": source.name,
            "hash": {"value": hashlib.sha256(source.read_bytes()).hexdigest()},
        }
    }

    def execute(command, **kwargs):
        if command[0] == "glslangValidator":
            Path(command[-1]).write_bytes(b"invalid SPIR-V")
        return SimpleNamespace(
            returncode=int(command[0] == "spirv-val"), stdout="", stderr="validation"
        )

    monkeypatch.setattr(verification.subprocess, "run", execute)
    with pytest.raises(RuntimeError, match="validation failed"):
        verification.compile_entry(
            "opengl", "entry", descriptor, tmp_path, tmp_path / "compiled"
        )
    record = json.loads((tmp_path / "compiled/entry.json").read_text())
    assert record["status"] == "failed" and record["steps"][1]["returncode"] == 1

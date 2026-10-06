import hashlib
import importlib
import json
import subprocess
import sys
from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from demos.integrations.mlx.tests.corpus_evidence import (
    EVIDENCE_DIRECTORY,
    KEEP_EVIDENCE_ENV,
    corpus_workspace,
    run_compiler,
)


@pytest.mark.parametrize("keep", (False, True))
@pytest.mark.parametrize("fail", (False, True))
def test_workspace_retention_does_not_change_failure(keep, fail, tmp_path, monkeypatch):
    monkeypatch.setenv(KEEP_EVIDENCE_ENV, "1" if keep else "0")
    error = AssertionError("artifact mismatch")
    with pytest.raises(AssertionError) if fail else nullcontext():
        with corpus_workspace(
            tmp_path, family="copy", target="directx", entry_point="copy_float"
        ) as work_dir:
            assert work_dir.relative_to(tmp_path).parts
            (work_dir / "shader.hlsl").write_text("void CSMain() {}", encoding="utf-8")
            if fail:
                raise error
    assert work_dir.exists() is keep
    if keep:
        assert work_dir.parent == tmp_path / EVIDENCE_DIRECTORY
        assert (work_dir / "shader.hlsl").is_file()
        assert json.loads((work_dir / "case.json").read_text(encoding="utf-8")) == {
            "family": "copy",
            "target": "directx",
            "entryPoint": "copy_float",
        }


def test_retained_workspaces_do_not_overwrite_repeated_cases(tmp_path, monkeypatch):
    monkeypatch.setenv(KEEP_EVIDENCE_ENV, "1")
    paths = []
    for _ in range(2):
        with corpus_workspace(
            tmp_path, family="copy", target="directx", entry_point="copy_float"
        ) as work_dir:
            paths.append(work_dir)
    assert paths[0] != paths[1]
    assert all(path.is_dir() for path in paths)


@pytest.mark.parametrize("returncode", (0, 7))
def test_compiler_preserves_command_output_and_failure(returncode, tmp_path):
    command = [
        sys.executable,
        "-c",
        f"import sys; print('output'); print('diagnostic', file=sys.stderr); sys.exit({returncode})",
    ]
    result = run_compiler(command, work_dir=tmp_path, timeout=30)
    assert result.returncode == returncode
    assert json.loads((tmp_path / "compiler.json").read_text(encoding="utf-8")) == {
        "command": command,
        "timeoutSeconds": 30,
        "status": "completed",
        "returncode": returncode,
        "stdout": "output\n",
        "stderr": "diagnostic\n",
    }


@pytest.mark.parametrize("byte_output", (False, True))
def test_compiler_retains_timeout_output_and_reraises(
    byte_output, tmp_path, monkeypatch
):
    stdout = b"partial\xff" if byte_output else "partial"
    stderr = b"diagnostic" if byte_output else "diagnostic"
    error = subprocess.TimeoutExpired(["dxc"], 120, output=stdout, stderr=stderr)

    def timeout(*args, **kwargs):
        record = json.loads((tmp_path / "compiler.json").read_text(encoding="utf-8"))
        assert record["status"] == "running"
        assert kwargs == {
            "check": False,
            "capture_output": True,
            "text": True,
            "timeout": 120,
        }
        raise error

    monkeypatch.setattr(subprocess, "run", timeout)
    with pytest.raises(subprocess.TimeoutExpired) as caught:
        run_compiler(["dxc"], work_dir=tmp_path, timeout=120)
    assert caught.value is error
    record = json.loads((tmp_path / "compiler.json").read_text(encoding="utf-8"))
    assert record["status"] == "timed-out"
    assert record["stdout"] == ("partial\ufffd" if byte_output else "partial")
    assert record["stderr"] == "diagnostic"
    assert "returncode" not in record


def test_compiler_retains_launch_error_and_reraises(tmp_path, monkeypatch):
    error = OSError("compiler unavailable")

    def unavailable(*args, **kwargs):
        raise error

    monkeypatch.setattr(subprocess, "run", unavailable)
    with pytest.raises(OSError) as caught:
        run_compiler(["dxc"], work_dir=tmp_path, timeout=120)
    assert caught.value is error
    record = json.loads((tmp_path / "compiler.json").read_text(encoding="utf-8"))
    assert record["status"] == "launch-failed"
    assert record["error"] == "compiler unavailable"
    assert "returncode" not in record


@pytest.mark.parametrize("dispatch", (False, True))
def test_softmax_retains_report_before_artifact_checks(dispatch, tmp_path, monkeypatch):
    from demos.integrations.mlx.tests.kernels import test_softmax_native_loader as proof

    payload = {"summary": {"unitCount": 0}, "diagnostics": [{"message": "failed"}]}

    class Report:
        def to_json(self):
            return payload

        def write_json(self, path):
            path.write_text(json.dumps(payload), encoding="utf-8")

    monkeypatch.setenv(KEEP_EVIDENCE_ENV, "1")
    monkeypatch.setattr(proof, "_pinned_mlx_root", lambda: tmp_path)
    monkeypatch.setattr(proof, "load_project_config", lambda *args: None)
    monkeypatch.setattr(proof, "translate_project", lambda *args, **kwargs: Report())
    with pytest.raises(AssertionError):
        if dispatch:
            proof.test_pinned_mlx_softmax_translates_to_guarded_dispatch_artifacts(
                "directx"
            )
        else:
            proof._translate_selected_artifact(
                tmp_path,
                tmp_path,
                target="directx",
                workload_id="block-float32-axis-32-two-rows",
            )
    reports = list(tmp_path.rglob("portability-report.json"))
    assert len(reports) == 1
    assert json.loads(reports[0].read_text(encoding="utf-8")) == payload


@pytest.mark.parametrize("target", ("directx", "opengl"))
@pytest.mark.parametrize("returncode", (0, 7))
def test_native_compiler_evidence_keeps_results_and_artifact_bytes(
    target, returncode, tmp_path, monkeypatch
):
    from demos.integrations.mlx.tests import corpus_evidence as proof

    shader = tmp_path / ("shader.hlsl" if target == "directx" else "shader.glsl")
    shader.write_text("shader source", encoding="utf-8")
    module = tmp_path / "shader.dxil"
    command = ["dxc", "-Fo", str(module), str(shader)]
    if target == "directx":
        module.write_bytes(b"compiled module")
    else:
        command = ["glslangValidator", "-S", "comp", str(shader)]
    expected = subprocess.CompletedProcess(command, returncode, "output", "diagnostic")

    def compile_record(actual, **kwargs):
        assert actual == command
        assert kwargs == {"work_dir": tmp_path, "timeout": 120}
        return expected

    monkeypatch.setattr(proof, "run_compiler", compile_record)
    result = proof.native_compiler_runner(tmp_path)(command, input_text=None)
    assert result is expected
    record = json.loads((tmp_path / "compiler-artifacts.json").read_text())
    paths = [shader, module] if target == "directx" else [shader]
    assert len(record["files"]) == len(paths)
    for source, item in zip(paths, record["files"]):
        raw = source.read_bytes()
        assert (tmp_path / item["path"]).read_bytes() == raw
        assert item["sha256"] == hashlib.sha256(raw).hexdigest()
        assert item["sizeBytes"] == len(raw)


@pytest.mark.parametrize("family", ("softmax", "arg_reduce"))
@pytest.mark.parametrize("target", ("directx", "opengl"))
@pytest.mark.parametrize("failure", (None, "status", "readback"))
def test_selected_corpus_retains_native_inputs_and_results_without_suppressing_failures(
    family, target, failure, tmp_path, monkeypatch
):
    proof = importlib.import_module(
        f"demos.integrations.mlx.tests.kernels.test_{family}_native_loader"
    )
    dtype = "float32" if family == "softmax" else "uint32"
    values = [1.0] if family == "softmax" else [5, 7]

    monkeypatch.setenv(KEEP_EVIDENCE_ENV, "1")
    monkeypatch.setattr(proof, "_pinned_mlx_root", lambda: tmp_path)
    monkeypatch.setattr(
        proof, "_build_runtime_package", lambda *a, **kw: ({}, tmp_path)
    )
    fixture = {"inputs": [{"values": [2.0]}], "expectedOutputs": [{"values": [1.0]}]}
    plan = {"dispatch": {"workgroupSize": [32, 1, 1]}}
    request = SimpleNamespace(
        fixture=SimpleNamespace(to_json=lambda: fixture),
        execution_plan=SimpleNamespace(to_json=lambda: plan),
    )
    monkeypatch.setattr(
        proof, "_runtime_request", lambda *a, **kw: (request, "out", values)
    )
    result = SimpleNamespace(
        status="failed" if failure == "status" else "ok",
        outputs={
            "out": {
                "dtype": dtype,
                "shape": [len(values)],
                "values": [0] * len(values) if failure == "readback" else values,
            }
        },
        details={"device": "unit-test-only"},
        message=None,
    )
    executor = SimpleNamespace(
        is_available=lambda request: SimpleNamespace(available=True),
        run=lambda request: result,
    )
    monkeypatch.setattr(proof, "RuntimeParityExecutor", lambda *a, **kw: executor)
    test = getattr(
        proof, f"test_pinned_mlx_{family}_executes_through_{target}_native_loader"
    )
    with pytest.raises(AssertionError) if failure else nullcontext():
        if family == "softmax":
            test("block-float32-axis-32-two-rows")
        else:
            test()
    workspaces = list((tmp_path / EVIDENCE_DIRECTORY).iterdir())
    assert len(workspaces) == (2 if family == "arg_reduce" and not failure else 1)
    for work_dir in workspaces:
        saved_request = json.loads((work_dir / "request.json").read_text())
        assert saved_request["commit"] == proof.MLX_COMMIT
        assert saved_request["fixture"] == fixture
        assert saved_request["executionPlan"] == plan
        saved_result = json.loads((work_dir / "result.json").read_text())
        assert saved_result == vars(result)


@pytest.mark.parametrize("target", ("directx", "opengl"))
@pytest.mark.parametrize("failure", ("translation", "identity"))
def test_arg_reduce_retains_failed_translation_before_artifact_checks(
    target, failure, tmp_path, monkeypatch
):
    from demos.integrations.mlx.tests.kernels import (
        test_arg_reduce_native_loader as proof,
    )

    payload = {"summary": {"unitCount": 0}, "diagnostics": [{"message": "failed"}]}
    if failure == "identity":
        payload = {
            "summary": {"unitCount": 1, "translatedCount": 1, "failedCount": 0},
            "diagnostics": [],
            "artifacts": [
                {
                    "source": proof.MLX_ARG_REDUCE_SOURCE,
                    "sourceHash": {
                        "algorithm": "sha256",
                        "value": proof.MLX_ARG_REDUCE_SHA256,
                    },
                    "generatedHash": {"algorithm": "sha256", "value": "0" * 64},
                }
            ],
        }

    class Report:
        def to_json(self):
            return payload

        def write_json(self, path):
            path.write_text(json.dumps(payload), encoding="utf-8")

    monkeypatch.setenv(KEEP_EVIDENCE_ENV, "1")
    monkeypatch.setattr(proof, "_pinned_mlx_root", lambda: tmp_path)
    monkeypatch.setattr(proof, "load_project_config", lambda *args: None)
    monkeypatch.setattr(proof, "translate_project", lambda *args, **kwargs: Report())
    with pytest.raises(AssertionError):
        proof.test_pinned_mlx_arg_reduce_translates_to_bounded_artifact(
            target, "argmin-float32-axis-32-two-rows"
        )
    reports = list(tmp_path.rglob("portability-report.json"))
    assert len(reports) == 1
    assert json.loads(reports[0].read_text(encoding="utf-8")) == payload
    assert (reports[0].parent / "crosstl.toml").is_file()


@pytest.mark.parametrize(
    "family", ("rms_norm", "rms_norm_vjp", "layer_norm", "layer_norm_vjp")
)
@pytest.mark.parametrize("target", ("directx", "opengl"))
def test_normalization_retains_failed_translation_before_artifact_checks(
    family, target, tmp_path, monkeypatch
):
    proof = importlib.import_module(
        f"demos.integrations.mlx.tests.kernels.test_{family}_native_loader"
    )
    payload = {"summary": {"unitCount": 0}, "diagnostics": [{"message": "failed"}]}

    class Report:
        def to_json(self):
            return payload

        def write_json(self, path):
            path.write_text(json.dumps(payload), encoding="utf-8")

    monkeypatch.setenv(KEEP_EVIDENCE_ENV, "1")
    monkeypatch.setattr(proof, "_pinned_mlx_root", lambda: tmp_path)
    monkeypatch.setattr(proof, "load_project_config", lambda *args: None)
    monkeypatch.setattr(proof, "translate_project", lambda *args, **kwargs: Report())
    suffix = (
        "directx_native_loader_artifact"
        if target == "directx"
        else (
            "deferred_software_opengl"
            if family.endswith("_vjp")
            else "software_subgroup_opengl"
        )
    )
    with pytest.raises(AssertionError):
        getattr(proof, f"test_pinned_mlx_{family}_translates_to_{suffix}")()
    (report,) = tmp_path.rglob(f"{target}-portability-report.json")
    assert json.loads(report.read_text()) == payload
    assert (report.parent / "crosstl.toml").is_file()
    assert report.parent.parent == tmp_path / EVIDENCE_DIRECTORY


@pytest.mark.parametrize(
    "family", ("rms_norm", "rms_norm_vjp", "layer_norm", "layer_norm_vjp")
)
@pytest.mark.parametrize("target", ("directx", "opengl"))
@pytest.mark.parametrize("failure", (None, "status", "readback"))
def test_normalization_retains_native_results_without_suppressing_failures(
    family, target, failure, tmp_path, monkeypatch
):
    proof = importlib.import_module(
        f"demos.integrations.mlx.tests.kernels.test_{family}_native_loader"
    )
    vjp = family.endswith("_vjp")
    compilation = {"case": "unit-test-only"}
    deferred = (tmp_path, compilation) if target == "opengl" else None
    package = ({}, tmp_path, deferred) if vjp else ({}, tmp_path)
    monkeypatch.setenv(KEEP_EVIDENCE_ENV, "1")
    monkeypatch.setattr(proof, "_pinned_mlx_root", lambda: tmp_path)
    monkeypatch.setattr(proof, "_build_runtime_package", lambda *a: package)
    names = proof._expected_binding_names(target)
    if vjp:
        inputs, outputs, expected_gx, expected_gw = proof._runtime_values(target)
        expected = {names[3]: expected_gx, names[4]: expected_gw}
    else:
        inputs, outputs = {"case": "unit-test-only"}, {}
        output_binding = 2 if family == "rms_norm" else 3
        expected = {names[output_binding]: proof._workload()[-1]}
    fixture = {"inputs": inputs, "expectedOutputs": outputs}
    plan = {"dispatch": {"workgroupSize": [32, 1, 1]}}
    request = SimpleNamespace(
        fixture=SimpleNamespace(to_json=lambda: fixture),
        execution_plan=SimpleNamespace(to_json=lambda: plan),
    )
    if vjp:
        monkeypatch.setattr(
            proof,
            "_directx_dispatch_request",
            lambda *a: (request, expected_gx, expected_gw),
        )
    else:
        monkeypatch.setattr(
            proof,
            "_dispatch_request",
            lambda *a: (request, next(iter(expected.values()))),
        )
    result = SimpleNamespace(
        status="failed" if failure == "status" else "ok",
        outputs={
            name: {
                "dtype": "float32",
                "shape": [len(values)],
                "values": [1000.0] * len(values) if failure == "readback" else values,
            }
            for name, values in expected.items()
        },
        details={
            "nativeDeferredCompilation": {
                "success": True,
                "target": {"backend": target},
                "variant": {
                    "specializationValues": [{"id": 20, "name": "has_w", "value": True}]
                },
                "interface": {"status": "verified"},
                "cache": {"status": "published"},
            }
        },
        message=None,
    )
    executor = SimpleNamespace(
        is_available=lambda request: SimpleNamespace(available=True),
        run=lambda request: result,
    )
    monkeypatch.setattr(proof, "RuntimeParityExecutor", lambda *a, **kw: executor)
    if vjp:
        monkeypatch.setattr(
            proof,
            "execute_native_deferred_compilation_request",
            lambda *a, **kw: result,
        )
    with pytest.raises(AssertionError) if failure else nullcontext():
        getattr(
            proof, f"test_pinned_mlx_{family}_executes_through_{target}_native_loader"
        )()
    (work_dir,) = (tmp_path / EVIDENCE_DIRECTORY).iterdir()
    saved = json.loads((work_dir / "request.json").read_text())
    assert saved["commit"] == proof.MLX_COMMIT
    assert saved["workload"] == getattr(proof, "MLX_" + family.upper() + "_ENTRY")
    if vjp and target == "opengl":
        assert saved["compilationRequest"] == compilation
        assert saved["inputs"] == inputs and saved["outputs"] == outputs
        assert saved["expectedGx"] == expected_gx and saved["expectedGw"] == expected_gw
        assert saved["workgroups"] == [1, 1, 1]
    else:
        assert saved["fixture"] == fixture
        assert saved["executionPlan"] == plan
    assert json.loads((work_dir / "result.json").read_text()) == vars(result)


@pytest.mark.parametrize("target", ("directx", "opengl"))
@pytest.mark.parametrize("failure", ("translation", "identity"))
def test_attention_retains_failed_translation_before_artifact_checks(
    target, failure, tmp_path, monkeypatch
):
    from demos.integrations.mlx.tests.kernels import (
        test_scaled_dot_product_attention_native_loader as proof,
    )

    payload = {"summary": {"unitCount": 0}, "diagnostics": [{"message": "failed"}]}
    if failure == "identity":
        payload = {
            "summary": {
                "unitCount": 1,
                "translatedCount": 1,
                "failedCount": 0,
                "diagnosticCounts": {"error": 0},
            },
            "diagnostics": [],
            "artifacts": [
                {
                    "source": proof.MLX_ATTENTION_SOURCE,
                    "sourceHash": {
                        "algorithm": "sha256",
                        "value": proof.MLX_ATTENTION_SHA256,
                    },
                    "generatedHash": {"algorithm": "sha256", "value": "0" * 64},
                }
            ],
        }

    class Report:
        def to_json(self):
            return payload

        def write_json(self, path):
            path.write_text(json.dumps(payload), encoding="utf-8")

    monkeypatch.setenv(KEEP_EVIDENCE_ENV, "1")
    monkeypatch.setattr(proof, "_pinned_mlx_root", lambda: tmp_path)
    monkeypatch.setattr(proof, "load_project_config", lambda *args: None)
    monkeypatch.setattr(proof, "translate_project", lambda *args, **kwargs: Report())
    test = (
        proof.test_pinned_mlx_attention_translates_to_directx_native_loader_artifact
        if target == "directx"
        else proof.test_pinned_mlx_attention_translates_to_deferred_software_opengl
    )
    with pytest.raises(AssertionError):
        test()
    (report,) = tmp_path.rglob(f"{target}-portability-report.json")
    assert json.loads(report.read_text()) == payload
    assert (report.parent / "crosstl.toml").is_file()
    assert report.parent.parent == tmp_path / EVIDENCE_DIRECTORY


@pytest.mark.parametrize("target", ("directx", "opengl"))
@pytest.mark.parametrize("failure", (None, "status", "readback"))
def test_attention_retains_native_results_without_suppressing_failures(
    target, failure, tmp_path, monkeypatch
):
    from demos.integrations.mlx.tests.kernels import (
        test_scaled_dot_product_attention_native_loader as proof,
    )

    inputs, outputs, expected = proof._runtime_values(target)
    name = proof._expected_binding_names(target)[3]
    compilation = {"case": "unit-test-only"}
    deferred = (tmp_path, compilation) if target == "opengl" else None
    monkeypatch.setenv(KEEP_EVIDENCE_ENV, "1")
    monkeypatch.setattr(proof, "_pinned_mlx_root", lambda: tmp_path)
    monkeypatch.setattr(
        proof, "_build_runtime_package", lambda *a: ({}, tmp_path, deferred)
    )
    fixture = {"inputs": inputs, "expectedOutputs": outputs}
    plan = {"dispatch": {"workgroupSize": [1024, 1, 1]}}
    request = SimpleNamespace(
        fixture=SimpleNamespace(to_json=lambda: fixture),
        execution_plan=SimpleNamespace(to_json=lambda: plan),
    )
    monkeypatch.setattr(proof, "_dispatch_request", lambda *a: (request, expected))
    result = SimpleNamespace(
        status="failed" if failure == "status" else "ok",
        outputs={
            name: {
                "dtype": "float32",
                "shape": [proof.DIMENSION],
                "values": (
                    [1000.0] * proof.DIMENSION if failure == "readback" else expected
                ),
            }
        },
        details={
            "nativeDeferredCompilation": {
                "success": True,
                "target": {"backend": target},
                "variant": {
                    "specializationValues": [
                        {"id": key, "name": name, "value": False}
                        for key, name in proof.SPECIALIZATION_NAMES.items()
                    ]
                },
                "interface": {"status": "verified"},
                "cache": {"status": "published"},
            }
        },
        message=None,
    )
    executor = SimpleNamespace(
        is_available=lambda request: SimpleNamespace(available=True),
        run=lambda request: result,
    )
    monkeypatch.setattr(proof, "RuntimeParityExecutor", lambda *a, **kw: executor)
    monkeypatch.setattr(
        proof, "execute_native_deferred_compilation_request", lambda *a, **kw: result
    )
    test = getattr(
        proof, f"test_pinned_mlx_attention_executes_through_{target}_native_loader"
    )
    with pytest.raises(AssertionError) if failure else nullcontext():
        test()
    (work_dir,) = (tmp_path / EVIDENCE_DIRECTORY).iterdir()
    saved = json.loads((work_dir / "request.json").read_text())
    assert saved["commit"] == proof.MLX_COMMIT
    assert saved["workload"] == proof.MLX_ATTENTION_ENTRY
    if target == "directx":
        assert saved["fixture"] == fixture
        assert saved["executionPlan"] == plan
    else:
        assert saved["compilationRequest"] == compilation
        assert saved["inputs"] == inputs and saved["outputs"] == outputs
        assert saved["expectedValues"] == expected and saved["workgroups"] == [1, 1, 1]
    assert json.loads((work_dir / "result.json").read_text()) == vars(result)


@pytest.mark.parametrize(
    "family,target",
    [(name, "directx") for name in ("unary", "binary", "copy", "reduce")]
    + [(name, "metal") for name in ("unary", "binary", "reduce", "copy", "quantized")],
)
def test_corpus_retains_report_before_translation_assertions(
    family, target, tmp_path, monkeypatch
):
    suffix = "metal_roundtrip" if target == "metal" else target
    module = importlib.import_module(
        "demos.integrations.mlx.tests.kernels.test_unary_native_loader"
        if (family, target) == ("unary", "metal")
        else f"demos.integrations.mlx.tests.kernels.test_{family}_complete_{suffix}"
    )
    payload = {
        "summary": {"unitCount": 0},
        "diagnostics": [{"message": "test failure"}],
    }

    class Report:
        def to_json(self):
            return payload

        def write_json(self, path):
            path.write_text(json.dumps(payload), encoding="utf-8")

    monkeypatch.setattr(module, "load_project_config", lambda *args: None)
    monkeypatch.setattr(module, "translate_project", lambda *args, **kwargs: Report())
    workload = getattr(module, f"CURRENT_{family.upper()}_{target.upper()}_WORKLOADS")[
        0
    ]
    with pytest.raises(AssertionError):
        if (family, target) == ("unary", "metal"):
            module._translate_unary_artifact(tmp_path, tmp_path, target, workload)
        elif target == "metal":
            getattr(module, f"_translate_{family}_metal_artifact")(
                tmp_path, tmp_path, workload
            )
        else:
            module._translate_and_validate(tmp_path, tmp_path, workload)
    assert (
        json.loads((tmp_path / "portability-report.json").read_text(encoding="utf-8"))
        == payload
    )
    assert (tmp_path / "crosstl.toml").is_file()
    assert not (tmp_path / "compiler.json").exists()


@pytest.mark.parametrize(
    "damage", (None, "translation", "manifest", "entry_point", "resources")
)
@pytest.mark.parametrize("family", ("unary", "binary", "reduce", "copy", "quantized"))
def test_metal_bundle_export_requires_translation_and_host_interface(
    family, damage, tmp_path, monkeypatch
):
    module = importlib.import_module(
        "demos.integrations.mlx.tests.kernels.test_unary_native_loader"
        if family == "unary"
        else f"demos.integrations.mlx.tests.kernels.test_{family}_complete_metal_roundtrip"
    )
    workload = getattr(module, f"CURRENT_{family.upper()}_METAL_WORKLOADS")[0]
    monkeypatch.setattr(module, "_pinned_mlx_root", lambda: tmp_path)
    if family == "unary":
        resources = [
            {"name": name, "kind": kind, "binding": binding, "access": access}
            for name, kind, binding, access in (
                ("in_", "buffer", 0, "read"),
                ("out_", "buffer", 1, "read_write"),
                ("in_shape", "constant-buffer", 2, "read"),
                ("in_strides", "constant-buffer", 3, "read"),
                ("ndim", "buffer", 4, "read"),
            )
        ]
    elif family == "binary":
        resources = module._resources(
            module.BINARY_SHAPE_SPECS[workload.shape].resource_kind
        )
    elif family == "copy":
        resources = module.COPY_METAL_RESOURCES_BY_ENTRY[workload.entry_point]
    elif family == "quantized":
        resources = module.QUANTIZED_METAL_RESOURCE_CONTRACTS[workload.resources_sha256]
    else:
        resources = module._resources(
            module.EXPECTED_SHAPE_CONTRACTS[workload.shape]["templateName"],
            workload.input_type,
            workload.output_type,
        )
    manifest = {
        "success": damage != "manifest",
        "artifacts": [
            {
                "hostInterface": {
                    "status": "ready",
                    "entryPoints": [
                        {
                            "name": workload.entry_point,
                            "stage": "compute",
                            "executionConfig": {},
                        }
                    ],
                    "resources": [
                        dict(resource, metadata={"entryPoint": workload.entry_point})
                        for resource in resources
                    ],
                }
            }
        ],
    }
    if damage == "entry_point":
        manifest["artifacts"][0]["hostInterface"]["entryPoints"][0][
            "name"
        ] = "incorrect"
    if damage == "resources":
        manifest["artifacts"][0]["hostInterface"]["resources"] = []
    calls = []

    def translate(*args, **kwargs):
        assert kwargs == (
            {"defer_native_compilation": True}
            if family not in {"unary", "binary"}
            else {}
        )
        calls.append("translation")
        assert damage != "translation"
        if family == "unary":
            assert args[2:] == ("metal", workload)
            path = tmp_path / "report.json"
            path.write_text(json.dumps({"artifacts": [{"path": "source.metal"}]}))
            return path
        return tmp_path / "report.json", tmp_path / "source.metal"

    def reflect(*args):
        calls.append("reflection")
        return manifest

    def export(source, contract, entry, root):
        assert calls == ["translation", "reflection"]
        assert source == tmp_path / "source.metal"
        assert contract == getattr(module, f"{family.upper()}_METAL_CONTRACT_PATH")
        assert entry == workload.entry_point
        assert root == tmp_path / "bundle"
        calls.append("export")

    monkeypatch.setattr(
        module,
        (
            "_translate_unary_artifact"
            if family == "unary"
            else f"_translate_{family}_metal_artifact"
        ),
        translate,
    )
    monkeypatch.setattr(module, "build_runtime_artifact_manifest", reflect)
    monkeypatch.setattr(module, "write_bundle_entry", export)
    monkeypatch.setattr(
        module.shutil,
        "which",
        lambda *args: pytest.fail("Export attempted native compilation"),
    )
    with pytest.raises(AssertionError) if damage else nullcontext():
        getattr(module, f"_roundtrip_pinned_mlx_{family}_through_metal")(
            workload, bundle_root=tmp_path / "bundle"
        )
    assert ("export" in calls) is (damage is None)


@pytest.mark.parametrize("mode", ("native", "source"))
@pytest.mark.parametrize("family", ("unary", "binary", "reduce", "copy", "quantized"))
def test_metal_required_source_cannot_skip(family, mode, monkeypatch):
    module = importlib.import_module(
        "demos.integrations.mlx.tests.kernels.test_unary_native_loader"
        if family == "unary"
        else f"demos.integrations.mlx.tests.kernels.test_{family}_complete_metal_roundtrip"
    )
    native_flag = (
        module.REQUIRE_PROOF_ENVS["metal"]
        if family == "unary"
        else getattr(module, f"REQUIRE_{family.upper()}_METAL_ENV")
    )
    source_flag = getattr(module, f"REQUIRE_{family.upper()}_METAL_SOURCE_ENV")
    monkeypatch.delenv("CROSTL_MLX_ROOT", raising=False)
    monkeypatch.delenv(native_flag, raising=False)
    monkeypatch.delenv(source_flag, raising=False)
    flag = native_flag if mode == "native" else source_flag
    monkeypatch.setenv(flag, "1")
    with pytest.raises(
        pytest.fail.Exception, match="CROSTL_MLX_ROOT is not configured"
    ):
        module._pinned_mlx_root()


@pytest.mark.parametrize("available", [False, True])
@pytest.mark.parametrize(
    "damage",
    (
        None,
        "code",
        "severity",
        "target",
        "capability",
        "extra",
        "count",
        "missing",
        "status",
    ),
)
def test_deferred_metal_diagnostics_only_allow_missing_native_compiler(
    available, damage
):
    from demos.integrations.mlx.tests.corpus_evidence import (
        assert_deferred_metal_compiler_diagnostics,
    )

    diagnostic = {
        "severity": "warning",
        "code": "project.validate.toolchain-unavailable",
        "target": "metal",
        "missingCapabilities": ["toolchain.validation"],
    }
    payload = {
        "summary": {
            "diagnosticCounts": {"note": 0, "warning": int(not available), "error": 0}
        },
        "validation": {
            "toolchains": [
                {
                    "target": "metal",
                    "status": "available" if available else "unavailable",
                }
            ]
        },
        "diagnostics": [] if available else [diagnostic],
    }
    if damage is not None and available:
        payload["diagnostics"].append(diagnostic)
    if damage == "code":
        diagnostic["code"] = "project.source.unsupported"
    elif damage == "severity":
        diagnostic["severity"] = "error"
    elif damage == "target":
        diagnostic["target"] = "opengl"
    elif damage == "capability":
        diagnostic["missingCapabilities"] = ["source.translation"]
    elif damage == "extra":
        payload["diagnostics"].append(dict(diagnostic))
    elif damage == "count":
        payload["summary"]["diagnosticCounts"]["warning"] += 1
    elif damage == "missing" and not available:
        payload["diagnostics"] = []
    elif damage == "status":
        payload["validation"]["toolchains"][0]["status"] = "not-configured"
    with pytest.raises(AssertionError) if damage else nullcontext():
        assert_deferred_metal_compiler_diagnostics(payload)

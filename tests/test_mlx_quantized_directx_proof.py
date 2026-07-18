import copy
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PROOF_PATH = ROOT / "demos" / "integrations" / "mlx" / ("prove_quantized_directx.py")

EXPECTED_PINNED_HASHES = {
    "mlx/backend/metal/kernels/quantized.metal": (
        "292aab5a98e3fc047b8ed91343fc10b66e5a92e12c258cde168929520ab2abfd"
    ),
    "mlx/backend/metal/kernels/quantized.h": (
        "4da52bf4ee688165a65b84c52a5f4e82efcae7f69e8c74d9ee3e00bef463c99f"
    ),
}


def _load_proof():
    spec = importlib.util.spec_from_file_location(
        "mlx_quantized_directx_proof",
        PROOF_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _synthetic_checkout(module, tmp_path, monkeypatch):
    mlx_root = tmp_path / "mlx"
    hashes = {}
    for index, relative_path in enumerate(module.PINNED_FILE_SHA256):
        path = mlx_root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"pinned fixture {index}: {relative_path}\n", encoding="utf-8")
        hashes[relative_path] = hashlib.sha256(path.read_bytes()).hexdigest()
    monkeypatch.setattr(module, "PINNED_FILE_SHA256", hashes)
    monkeypatch.setattr(module, "_git_revision", lambda _root: module.MLX_COMMIT)
    return mlx_root


def _generated_hlsl():
    return """
RWStructuredBuffer<uint> out_ : register(u0);

[numthreads(1, 1, 1)]
void CSMain(uint3 index_dispatchThreadID : SV_DispatchThreadID) {
    uint64_t out_index = index_dispatchThreadID.x;
    uint output = 0;
    out_[uint((out_index + 4))] = uint(((output & 1095216660480ull) >> 32));
}
"""


def _translated_payload(module, mlx_root, work_dir, generated=None):
    artifact_path = work_dir / "artifacts" / "directx" / "quantized.hlsl"
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(generated or _generated_hlsl(), encoding="utf-8")
    artifact_hash = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
    payload = {
        "kind": "crosstl-project-portability-report",
        "summary": {
            "unitCount": 1,
            "targetCount": 1,
            "artifactCount": 1,
            "translatedCount": 1,
            "failedCount": 0,
            "skippedCount": 0,
            "diagnosticCounts": {"error": 0, "note": 0, "warning": 0},
        },
        "diagnostics": [],
        "artifacts": [
            {
                "source": module.MLX_QUANTIZED_SOURCE,
                "sourceBackend": "metal",
                "target": "directx",
                "status": "translated",
                "path": artifact_path.relative_to(mlx_root).as_posix(),
                "sourceHash": {
                    "algorithm": "sha256",
                    "value": module.PINNED_FILE_SHA256[module.MLX_QUANTIZED_SOURCE],
                },
                "provenance": {
                    "pipeline": "entry-scoped-translate",
                    "intermediate": "crossgl",
                },
                "entryPoint": {
                    "source": module.MLX_QUANTIZED_ENTRY_POINT,
                    "target": "CSMain",
                    "stage": "compute",
                },
                "requiredCapabilities": [],
                "templateMaterialization": {
                    "status": "materialized",
                    "specializationCount": 3,
                    "specializations": [
                        {
                            "name": "affine_quantize",
                            "hostName": module.MLX_QUANTIZED_ENTRY_POINT,
                            "materializedName": module.MLX_QUANTIZED_ENTRY_POINT,
                            "parameters": {
                                "T": "float",
                                "bits": "2",
                                "group_size": "32",
                            },
                            "source": "source-instantiation",
                        },
                        {
                            "name": "get_pack_factor",
                            "materializedName": "get_pack_factor_2_8",
                            "parameters": {"bits": "2", "wsize": "8"},
                            "source": "call-site",
                        },
                        {
                            "name": "get_bytes_per_pack",
                            "materializedName": "get_bytes_per_pack_2",
                            "parameters": {"bits": "2", "wsize": "8"},
                            "source": "call-site",
                        },
                    ],
                    "unsupported": [],
                    "accounting": {
                        "reachableSpecializationCount": 6,
                        "prunedCandidateCount": 110861,
                    },
                },
                "generatedHash": {
                    "algorithm": "sha256",
                    "value": artifact_hash,
                },
                "generatedSizeBytes": artifact_path.stat().st_size,
            }
        ],
    }
    return payload, artifact_path


def test_quantized_directx_proof_pins_revision_source_header_and_entry():
    module = _load_proof()

    assert module.MLX_COMMIT == "4367c73b60541ddd5a266ce4644fd93d20223b6e"
    assert module.PINNED_FILE_SHA256 == EXPECTED_PINNED_HASHES
    assert module.MLX_QUANTIZED_ENTRY_POINT == "affine_quantize_float_gs_32_b_2"
    assert module.DIRECTX_TARGET_PROFILE == "directx-12"
    assert module.TEMPLATE_SPECIALIZATION_LIMIT == 128
    assert module.MATERIALIZATION_WORK_LIMIT == 4096
    assert module.NON_RUNTIME_CLAIMS == {
        "runtimeExecution": False,
        "numericalParity": False,
        "mlxUnitTests": False,
        "fullMlxTestSuite": False,
    }


def test_quantized_directx_checkout_fails_closed_on_revision_and_file_drift(
    tmp_path,
    monkeypatch,
):
    module = _load_proof()
    mlx_root = _synthetic_checkout(module, tmp_path, monkeypatch)
    assert module._verify_checkout(mlx_root)["status"] == "passed"

    monkeypatch.setattr(module, "_git_revision", lambda _root: "0" * 40)
    with pytest.raises(module.MlxQuantizedDirectXProofError, match="must be pinned"):
        module._verify_checkout(mlx_root)

    monkeypatch.setattr(module, "_git_revision", lambda _root: module.MLX_COMMIT)
    header = mlx_root / module.MLX_QUANTIZED_HEADER
    header.write_text("drifted\n", encoding="utf-8")
    with pytest.raises(
        module.MlxQuantizedDirectXProofError,
        match=r"SHA-256 mismatch for .*quantized\.h",
    ):
        module._verify_checkout(mlx_root)


def test_quantized_directx_work_and_artifact_paths_must_stay_in_checkout(tmp_path):
    module = _load_proof()
    mlx_root = tmp_path / "mlx"
    mlx_root.mkdir()

    with pytest.raises(module.MlxQuantizedDirectXProofError, match="inside"):
        module._resolve_work_dir(mlx_root, str(mlx_root))
    with pytest.raises(module.MlxQuantizedDirectXProofError, match="inside"):
        module._resolve_work_dir(mlx_root, str(tmp_path / "outside"))

    work_dir = mlx_root / "proof"
    payload, _artifact_path = _translated_payload(module, mlx_root, work_dir)
    escaped = tmp_path / "escaped.hlsl"
    escaped.write_text(_generated_hlsl(), encoding="utf-8")
    payload["artifacts"][0]["path"] = "../escaped.hlsl"
    payload["artifacts"][0]["generatedHash"] = {
        "algorithm": "sha256",
        "value": hashlib.sha256(escaped.read_bytes()).hexdigest(),
    }
    payload["artifacts"][0]["generatedSizeBytes"] = escaped.stat().st_size

    with pytest.raises(module.MlxQuantizedDirectXProofError, match="outside"):
        module._translated_artifact(
            payload,
            mlx_root=mlx_root,
            work_dir=work_dir,
        )


def test_quantized_directx_provenance_precedes_cleanup_and_translation(
    tmp_path,
    monkeypatch,
):
    module = _load_proof()
    mlx_root = tmp_path / "mlx"
    mlx_root.mkdir()
    work_dir = mlx_root / "proof"
    work_dir.mkdir()
    marker = work_dir / "existing.txt"
    marker.write_text("preserve\n", encoding="utf-8")
    translated = False

    def reject_provenance(_root):
        raise module.MlxQuantizedDirectXProofError("provenance drift")

    def unexpected_translation(*_args, **_kwargs):
        nonlocal translated
        translated = True
        raise AssertionError("translation must not run")

    monkeypatch.setattr(module, "_verify_checkout", reject_provenance)
    monkeypatch.setattr(module, "_translate_report", unexpected_translation)

    with pytest.raises(module.MlxQuantizedDirectXProofError, match="provenance drift"):
        module.run_proof(mlx_root, work_dir)

    assert marker.read_text(encoding="utf-8") == "preserve\n"
    assert translated is False


def test_quantized_directx_project_config_selects_one_directx_12_entry(tmp_path):
    module = _load_proof()
    mlx_root = tmp_path / "mlx"
    mlx_root.mkdir()
    work_dir = mlx_root / "proof"

    config = module._project_config(mlx_root, work_dir)

    assert tuple(config.source_roots) == (module.MLX_KERNEL_ROOT,)
    assert tuple(config.include_patterns) == (module.MLX_QUANTIZED_SOURCE,)
    assert tuple(config.targets) == ("directx-12",)
    assert tuple(config.include_dirs) == (".",)
    assert config.source_overrides == {module.MLX_QUANTIZED_SOURCE: "metal"}
    assert config.entry_points == {
        module.MLX_QUANTIZED_SOURCE: module.MLX_QUANTIZED_ENTRY_POINT
    }
    assert config.source_options == {
        "metal": {
            "max_template_specializations": 128,
            "max_template_materialization_work": 4096,
        }
    }


def test_quantized_directx_translation_uses_public_project_api(
    tmp_path,
    monkeypatch,
):
    module = _load_proof()
    mlx_root = tmp_path / "mlx"
    mlx_root.mkdir()
    work_dir = mlx_root / "proof"
    report_path = work_dir / "report.json"
    config = module._project_config(mlx_root, work_dir)
    captured = {}
    expected = {"kind": "crosstl-project-portability-report", "value": 1}

    class FakeReport:
        def to_json(self):
            return expected

    def translate_project(config_arg, **kwargs):
        captured["config"] = config_arg
        captured["kwargs"] = kwargs
        return FakeReport()

    monkeypatch.setattr(module, "translate_project", translate_project)

    assert module._translate_report(config, report_path=report_path) == expected
    assert captured == {
        "config": config,
        "kwargs": {"format_output": False, "validate": False},
    }
    assert json.loads(report_path.read_text(encoding="utf-8")) == expected


@pytest.mark.parametrize(
    "mutate",
    [
        lambda payload: payload.update(kind="unexpected-report"),
        lambda payload: payload["summary"].update(unitCount=2),
        lambda payload: payload["summary"].update(artifactCount=2),
        lambda payload: payload["summary"]["diagnosticCounts"].update(warning=1),
        lambda payload: payload["diagnostics"].append(
            {"severity": "warning", "code": "project.translate.warning"}
        ),
        lambda payload: payload["artifacts"][0].update(source="other.metal"),
        lambda payload: payload["artifacts"][0].update(target="opengl"),
        lambda payload: payload["artifacts"][0]["entryPoint"].update(
            target="OtherMain"
        ),
        lambda payload: payload["artifacts"][0].update(
            requiredCapabilities=["directx.native-16bit-types"]
        ),
        lambda payload: payload["artifacts"][0]["templateMaterialization"].update(
            specializationCount=2
        ),
        lambda payload: payload["artifacts"][0]["templateMaterialization"][
            "accounting"
        ].update(reachableSpecializationCount=5),
        lambda payload: payload["artifacts"][0]["templateMaterialization"][
            "accounting"
        ].update(prunedCandidateCount=110860),
    ],
)
def test_quantized_directx_report_drift_fails_closed(tmp_path, mutate):
    module = _load_proof()
    mlx_root = tmp_path / "mlx"
    work_dir = mlx_root / "proof"
    payload, _artifact_path = _translated_payload(module, mlx_root, work_dir)
    mutated = copy.deepcopy(payload)
    mutate(mutated)

    with pytest.raises(module.MlxQuantizedDirectXProofError):
        module._translated_artifact(
            mutated,
            mlx_root=mlx_root,
            work_dir=work_dir,
        )


def test_quantized_directx_generated_hash_drift_fails_closed(tmp_path):
    module = _load_proof()
    mlx_root = tmp_path / "mlx"
    work_dir = mlx_root / "proof"
    payload, artifact_path = _translated_payload(module, mlx_root, work_dir)
    artifact_path.write_text(_generated_hlsl() + "// drift\n", encoding="utf-8")

    with pytest.raises(module.MlxQuantizedDirectXProofError, match="identity"):
        module._translated_artifact(
            payload,
            mlx_root=mlx_root,
            work_dir=work_dir,
        )


def test_quantized_directx_generated_contract_is_exact(tmp_path):
    module = _load_proof()
    artifact_path = tmp_path / "quantized.hlsl"
    artifact_path.write_text(_generated_hlsl(), encoding="utf-8")

    checks, compiler = module._validate_generated_hlsl(artifact_path)

    assert checks == {
        "status": "passed",
        "entryPoint": "CSMain",
        "native16BitTypes": "not-required",
        "staticAssertions": "absent",
        "minimumPrecisionTypes": "absent",
        "typedResourceStoreNarrowing": {
            "status": "passed",
            "resource": "out_",
            "resourceElementType": "uint",
            "sourceExpressionType": "uint64_t",
            "generatedStore": module.NARROWED_RESOURCE_STORE,
        },
    }
    assert compiler == {
        "entryPoint": "CSMain",
        "profile": "cs_6_0",
        "compilerArguments": [],
        "targetProfiles": ["directx-12"],
        "warningsAsErrors": True,
    }


@pytest.mark.parametrize(
    ("generated", "message"),
    [
        (_generated_hlsl() + "\nstatic_assert(true);\n", "static_assert"),
        (_generated_hlsl() + "\nmin16uint minimum_width_marker;\n", "min16"),
        (
            _generated_hlsl().replace(
                "uint(((output & 1095216660480ull) >> 32))",
                "((output & 1095216660480ull) >> 32)",
            ),
            "narrowed",
        ),
        (_generated_hlsl().replace("CSMain", "OtherMain"), "CSMain"),
        (_generated_hlsl() + "\nuint16_t native_width_marker;\n", "native 16-bit"),
    ],
)
def test_quantized_directx_generated_contract_rejects_output_drift(
    tmp_path,
    generated,
    message,
):
    module = _load_proof()
    artifact_path = tmp_path / "quantized.hlsl"
    artifact_path.write_text(generated, encoding="utf-8")

    with pytest.raises(module.MlxQuantizedDirectXProofError, match=message):
        module._validate_generated_hlsl(artifact_path)


def test_quantized_directx_fake_dxc_uses_derived_profile_and_flags(
    tmp_path,
    monkeypatch,
):
    module = _load_proof()
    mlx_root = tmp_path / "mlx"
    work_dir = mlx_root / "proof"
    log_dir = work_dir / "logs"
    artifact_path = work_dir / "artifacts" / "quantized.hlsl"
    artifact_path.parent.mkdir(parents=True)
    artifact_path.write_text(_generated_hlsl(), encoding="utf-8")
    _checks, contract = module._validate_generated_hlsl(artifact_path)
    captured = {}

    def run_command(name, command, *, log_dir, timeout_seconds=180):
        del timeout_seconds
        captured["name"] = name
        captured["command"] = list(command)
        output_path = Path(command[command.index("-Fo") + 1])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"DXIL")
        log_dir.mkdir(parents=True, exist_ok=True)
        stdout_path = log_dir / f"{name}.stdout"
        stderr_path = log_dir / f"{name}.stderr"
        stdout_path.write_text("", encoding="utf-8")
        stderr_path.write_text("", encoding="utf-8")
        return {
            "name": name,
            "command": list(command),
            "returncode": 0,
            "stdoutPath": stdout_path,
            "stderrPath": stderr_path,
        }

    monkeypatch.setattr(module, "_run_command", run_command)
    result = module._compile_directx_artifact(
        artifact_path,
        contract,
        dxc="C:/tools/dxc.exe",
        mlx_root=mlx_root,
        work_dir=work_dir,
        log_dir=log_dir,
        required=True,
    )

    assert captured["name"] == "compile-quantized-directx"
    assert captured["command"] == [
        "C:/tools/dxc.exe",
        "-WX",
        "-T",
        "cs_6_0",
        "-E",
        "CSMain",
        str(artifact_path),
        "-Fo",
        str(work_dir / "native" / "directx" / "affine_quantize_float_gs_32_b_2.dxil"),
    ]
    assert result["status"] == "compiled"
    assert result["compiledArtifactCount"] == 1
    assert result["targetProfiles"] == ["directx-12"]
    assert result["compilerArguments"] == []


def test_quantized_directx_toolchain_requirement_and_output_fail_closed(
    tmp_path,
    monkeypatch,
):
    module = _load_proof()
    mlx_root = tmp_path / "mlx"
    artifact_path = mlx_root / "quantized.hlsl"
    artifact_path.parent.mkdir()
    artifact_path.write_text(_generated_hlsl(), encoding="utf-8")
    _checks, contract = module._validate_generated_hlsl(artifact_path)
    work_dir = mlx_root / "proof"
    log_dir = work_dir / "logs"

    optional = module._compile_directx_artifact(
        artifact_path,
        contract,
        dxc=None,
        mlx_root=mlx_root,
        work_dir=work_dir,
        log_dir=log_dir,
        required=False,
    )
    assert optional["status"] == "not-required"
    assert optional["available"] is False
    assert optional["compiledArtifactCount"] == 0

    with pytest.raises(module.MlxQuantizedDirectXProofError, match="requires dxc"):
        module._compile_directx_artifact(
            artifact_path,
            contract,
            dxc=None,
            mlx_root=mlx_root,
            work_dir=work_dir,
            log_dir=log_dir,
            required=True,
        )

    def run_without_output(name, command, *, log_dir, timeout_seconds=180):
        del timeout_seconds
        log_dir.mkdir(parents=True, exist_ok=True)
        stdout_path = log_dir / f"{name}.stdout"
        stderr_path = log_dir / f"{name}.stderr"
        stdout_path.write_text("", encoding="utf-8")
        stderr_path.write_text("", encoding="utf-8")
        return {
            "name": name,
            "command": list(command),
            "returncode": 0,
            "stdoutPath": stdout_path,
            "stderrPath": stderr_path,
        }

    monkeypatch.setattr(module, "_run_command", run_without_output)
    with pytest.raises(module.MlxQuantizedDirectXProofError, match="nonempty"):
        module._compile_directx_artifact(
            artifact_path,
            contract,
            dxc="dxc",
            mlx_root=mlx_root,
            work_dir=work_dir,
            log_dir=log_dir,
            required=True,
        )


def test_quantized_directx_run_writes_deterministic_compile_only_summary(
    tmp_path,
    monkeypatch,
):
    module = _load_proof()
    mlx_root = _synthetic_checkout(module, tmp_path, monkeypatch)
    work_dir = mlx_root / "proof"

    def translate_report(_config, *, report_path):
        payload, _artifact_path = _translated_payload(module, mlx_root, work_dir)
        module._write_json(report_path, payload)
        return payload

    monkeypatch.setattr(module, "_translate_report", translate_report)
    monkeypatch.setattr(module.shutil, "which", lambda _name: None)

    summary = module.run_proof(
        mlx_root,
        work_dir,
        require_directx_toolchain=False,
    )
    summary_path = work_dir / "summary.json"

    assert summary["status"] == "passed"
    assert summary["scope"]["translation"] == {
        "source": module.MLX_QUANTIZED_SOURCE,
        "selectedEntryPoint": module.MLX_QUANTIZED_ENTRY_POINT,
        "sourceBackend": "metal",
        "sourceOverride": "metal",
        "includeDirectories": ["."],
        "target": "directx-12",
        "projectTranslationApi": "crosstl.project.translate_project",
        "materializationLimits": {
            "maxTemplateSpecializations": 128,
            "maxTemplateMaterializationWork": 4096,
        },
    }
    assert summary["scope"]["runtime"] == {
        "executionAttempted": False,
        "backendIntegrationAttempted": False,
        "mlxTestsRun": False,
    }
    assert summary["scope"]["numerical"] == {
        "comparisonAttempted": False,
        "parityClaimed": False,
    }
    assert summary["claims"] == {
        "projectTranslation": True,
        "nativeCompilation": False,
        "runtimeExecution": False,
        "numericalParity": False,
        "mlxUnitTests": False,
        "fullMlxTestSuite": False,
    }
    assert summary["compiler"]["status"] == "not-required"
    assert summary["translation"]["templateMaterialization"]["specializationCount"] == 3
    assert summary_path.read_text(encoding="utf-8") == (
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )


def test_quantized_directx_cli_writes_failure_summary_inside_work_dir(
    tmp_path,
    monkeypatch,
):
    module = _load_proof()
    mlx_root = tmp_path / "mlx"
    mlx_root.mkdir()

    def fail_run(*_args, **_kwargs):
        raise module.MlxQuantizedDirectXProofError("translation failed closed")

    monkeypatch.setattr(module, "run_proof", fail_run)

    result = module.main(
        [
            "--mlx-root",
            str(mlx_root),
            "--work-dir",
            "proof",
            "--require-directx-toolchain",
        ]
    )
    summary = json.loads(
        (mlx_root / "proof" / "summary.json").read_text(encoding="utf-8")
    )

    assert result == 1
    assert summary["status"] == "failed"
    assert summary["scope"]["compiler"]["required"] is True
    assert summary["scope"]["runtime"]["executionAttempted"] is False
    assert summary["scope"]["numerical"]["parityClaimed"] is False
    assert summary["claims"]["runtimeExecution"] is False
    assert summary["claims"]["numericalParity"] is False


def test_quantized_directx_cli_exposes_required_toolchain_flag():
    module = _load_proof()
    args = module.parse_args(
        [
            "--mlx-root",
            "/tmp/mlx",
            "--require-directx-toolchain",
            "--no-clean",
        ]
    )

    assert args.require_directx_toolchain is True
    assert args.no_clean is True

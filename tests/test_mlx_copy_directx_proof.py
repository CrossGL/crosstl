import copy
import hashlib
import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PROOF_PATH = ROOT / "demos" / "integrations" / "mlx" / "prove_copy_directx.py"
WORKFLOW_PATH = ROOT / ".github" / "workflows" / "mlx-project-porting.yml"

EXPECTED_PINNED_HASHES = {
    "mlx/backend/metal/kernels/copy.metal": (
        "ed8a579eb6fe6a14c36560d2c8b548baf99e66fa77d300fb4ad7554883820eba"
    ),
    "mlx/backend/metal/kernels/copy.h": (
        "faafc09afc5e190252f3544c966b333b503c30404be3836444abf645f615b1c8"
    ),
    "mlx/backend/metal/kernels/utils.h": (
        "c30223b42b71068321149eea4fcd319878a4004425fb7cc34cdd296a76fabbfc"
    ),
    "mlx/backend/metal/kernels/bf16.h": (
        "abd87446a310b77ac530ef52a324feae5cb285d03ec9613e3a88ebb71410fdcb"
    ),
    "mlx/backend/metal/kernels/bf16_math.h": (
        "1f374f8380f756eb89acf6a847741cb8fecbe642945e159fb6208d804cc06496"
    ),
    "mlx/backend/metal/kernels/complex.h": (
        "aa3d29a2a0bb31fc0071493e3ac917387f96ba059d10a8f371a0b6a41a216dd3"
    ),
    "mlx/backend/metal/kernels/defines.h": (
        "a2930dbd644c69c4b66a511a094034217f3c03f48e29a1613f601532150f9163"
    ),
    "mlx/backend/metal/kernels/logging.h": (
        "fae44781743dbc5eb727e505b090e8445adff5626c1179bcb193b6fe7bedac8f"
    ),
}


def _load_proof():
    spec = importlib.util.spec_from_file_location("mlx_copy_directx_proof", PROOF_PATH)
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
struct complex64_t {
    float real;
    float imag;
};
StructuredBuffer<complex64_t> src : register(t0);
RWStructuredBuffer<uint16_t> dst : register(u1);

uint __crossgl_bfloat16_from_float(float value) {
    return (asuint(value) >> 16u) & 0xffffu;
}

[numthreads(1, 1, 1)]
void CSMain(uint3 index_dispatchThreadID : SV_DispatchThreadID) {
    uint index = index_dispatchThreadID.x;
    if (index == 0) {
        int i = 0;
        dst[(index + i)] = uint16_t(__crossgl_bfloat16_from_float(float((src.Load(0)).real)));
    } else {
        int i = 0;
        dst[(index + i)] = uint16_t(__crossgl_bfloat16_from_float(float((src.Load(0)).real)));
    }
}
"""


def _translated_payload(module, mlx_root, work_dir, generated=None):
    artifact_path = work_dir / "artifacts" / "directx" / "copy.hlsl"
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(generated or _generated_hlsl(), encoding="utf-8")
    artifact_hash = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
    payload = {
        "kind": "crosstl-project-portability-report",
        "summary": {
            "unitCount": 1,
            "artifactCount": 1,
            "translatedCount": 1,
            "failedCount": 0,
            "diagnosticCounts": {"error": 0},
        },
        "diagnostics": [],
        "artifacts": [
            {
                "source": module.MLX_COPY_SOURCE,
                "sourceBackend": "metal",
                "target": "directx",
                "status": "translated",
                "path": artifact_path.relative_to(mlx_root).as_posix(),
                "sourceHash": {
                    "algorithm": "sha256",
                    "value": module.PINNED_FILE_SHA256[module.MLX_COPY_SOURCE],
                },
                "provenance": {
                    "pipeline": "entry-scoped-translate",
                    "intermediate": "crossgl",
                },
                "requiredCapabilities": ["directx.native-16bit-types"],
                "entryPoint": {
                    "source": module.MLX_COPY_ENTRY_POINT,
                    "target": "CSMain",
                    "stage": "compute",
                },
                "templateMaterialization": {
                    "status": "materialized",
                    "specializationCount": 1,
                    "specializations": [
                        {
                            "name": module.MLX_COPY_TEMPLATE,
                            "materializedName": module.MLX_COPY_ENTRY_POINT,
                            "parameters": module.MLX_COPY_TEMPLATE_ARGUMENTS,
                            "source": "source-instantiation",
                            "hostName": module.MLX_COPY_ENTRY_POINT,
                        }
                    ],
                    "unsupported": [],
                    "accounting": {
                        "reachableSpecializationCount": 1,
                        "prunedCandidateCount": 69915,
                    },
                },
                "bfloat16Lowering": {
                    "status": "exact",
                    "approximationUsed": False,
                    "registerRepresentation": "uint-low-16-bits",
                    "storageRepresentation": "native-uint16",
                    "roundingMode": "round-to-nearest-ties-to-even",
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


def test_copy_directx_proof_pins_revision_source_headers_and_entry():
    module = _load_proof()

    assert module.MLX_COMMIT == "4367c73b60541ddd5a266ce4644fd93d20223b6e"
    assert module.PINNED_FILE_SHA256 == EXPECTED_PINNED_HASHES
    assert module.MLX_COPY_ENTRY_POINT == "s_copycomplex64bfloat16"
    assert module.MLX_COPY_TEMPLATE_ARGUMENTS == {
        "T": "complex64_t",
        "U": "bfloat16_t",
        "N": "1",
    }
    assert module.NON_RUNTIME_CLAIMS == {
        "runtimeExecution": False,
        "numericalParity": False,
        "mlxUnitTests": False,
        "fullMlxTestSuite": False,
    }


def test_copy_directx_provenance_fails_closed_on_revision_and_header_drift(
    tmp_path,
    monkeypatch,
):
    module = _load_proof()
    mlx_root = _synthetic_checkout(module, tmp_path, monkeypatch)
    assert module._verify_checkout(mlx_root)["status"] == "passed"

    monkeypatch.setattr(module, "_git_revision", lambda _root: "0" * 40)
    with pytest.raises(module.MlxCopyDirectXProofError, match="must be pinned"):
        module._verify_checkout(mlx_root)

    monkeypatch.setattr(module, "_git_revision", lambda _root: module.MLX_COMMIT)
    header = mlx_root / "mlx/backend/metal/kernels/bf16.h"
    header.write_text("drifted\n", encoding="utf-8")
    with pytest.raises(
        module.MlxCopyDirectXProofError,
        match=r"SHA-256 mismatch for .*bf16\.h",
    ):
        module._verify_checkout(mlx_root)


def test_copy_directx_provenance_precedes_cleanup_and_translation(
    tmp_path,
    monkeypatch,
):
    module = _load_proof()
    mlx_root = tmp_path / "mlx"
    mlx_root.mkdir()
    work_dir = mlx_root / "proof"
    work_dir.mkdir()
    marker = work_dir / "existing.txt"
    marker.write_text("preserve on provenance failure\n", encoding="utf-8")
    translated = False

    def reject_provenance(_root):
        raise module.MlxCopyDirectXProofError("provenance drift")

    def unexpected_translation(*_args, **_kwargs):
        nonlocal translated
        translated = True
        raise AssertionError("translation must not run")

    monkeypatch.setattr(module, "_verify_checkout", reject_provenance)
    monkeypatch.setattr(module, "_translate_report", unexpected_translation)

    with pytest.raises(module.MlxCopyDirectXProofError, match="provenance drift"):
        module.run_proof(mlx_root, work_dir)

    assert marker.read_text(encoding="utf-8") == "preserve on provenance failure\n"
    assert translated is False


def test_copy_directx_project_config_selects_one_full_source_entry(tmp_path):
    module = _load_proof()
    mlx_root = tmp_path / "mlx"
    mlx_root.mkdir()
    work_dir = mlx_root / "proof"

    config = module._project_config(mlx_root, work_dir)

    assert tuple(config.targets) == ("directx",)
    assert tuple(config.include_patterns) == (module.MLX_COPY_SOURCE,)
    assert config.entry_points == {module.MLX_COPY_SOURCE: module.MLX_COPY_ENTRY_POINT}
    assert tuple(config.include_dirs) == (".",)
    assert config.source_overrides == {module.MLX_COPY_SOURCE: "metal"}
    assert config.source_options == {
        "metal": {
            "max_template_specializations": 16,
            "max_template_materialization_work": 64,
        }
    }


@pytest.mark.parametrize(
    "mutate",
    [
        lambda payload: payload.update(kind="unexpected-report"),
        lambda payload: payload["summary"].update(artifactCount=2),
        lambda payload: payload["diagnostics"].append(
            {"severity": "error", "code": "project.translate.failed"}
        ),
        lambda payload: payload["artifacts"][0].update(source="other.metal"),
        lambda payload: payload["artifacts"][0]["sourceHash"].update(value="0" * 64),
        lambda payload: payload["artifacts"][0].update(
            provenance={"pipeline": "other", "intermediate": "crossgl"}
        ),
        lambda payload: payload["artifacts"][0]["templateMaterialization"].update(
            specializationCount=2
        ),
        lambda payload: payload["artifacts"][0]["templateMaterialization"][
            "accounting"
        ].update(reachableSpecializationCount=2),
    ],
)
def test_copy_directx_report_and_provenance_drift_fail_closed(
    tmp_path,
    mutate,
):
    module = _load_proof()
    mlx_root = tmp_path / "mlx"
    work_dir = mlx_root / "proof"
    payload, _artifact_path = _translated_payload(module, mlx_root, work_dir)
    mutated = copy.deepcopy(payload)
    mutate(mutated)

    with pytest.raises(module.MlxCopyDirectXProofError):
        module._translated_artifact(
            mutated,
            mlx_root=mlx_root,
            work_dir=work_dir,
        )


def test_copy_directx_generated_hash_drift_fails_closed(tmp_path):
    module = _load_proof()
    mlx_root = tmp_path / "mlx"
    work_dir = mlx_root / "proof"
    payload, artifact_path = _translated_payload(module, mlx_root, work_dir)
    artifact_path.write_text(_generated_hlsl() + "// drift\n", encoding="utf-8")

    with pytest.raises(module.MlxCopyDirectXProofError, match="identity"):
        module._translated_artifact(
            payload,
            mlx_root=mlx_root,
            work_dir=work_dir,
        )


def test_copy_directx_conversion_projects_real_once_per_store(tmp_path):
    module = _load_proof()
    artifact_path = tmp_path / "copy.hlsl"
    artifact_path.write_text(_generated_hlsl(), encoding="utf-8")

    conversion, contract = module._validate_conversion(artifact_path)

    assert conversion == {
        "status": "passed",
        "sourceExpression": "static_cast<bfloat16_t>(src[0])",
        "generatedProjection": "src.Load(0).real",
        "conversionHelper": "__crossgl_bfloat16_from_float",
        "storeSiteCount": 2,
        "sourceValueEvaluationsPerStore": 1,
        "complexStructPassedDirectly": False,
    }
    assert contract == {
        "entryPoint": "CSMain",
        "stage": "compute",
        "profile": "cs_6_2",
        "compilerArguments": ["-enable-16bit-types"],
        "executionConfig": {"numthreads": [1, 1, 1]},
    }


@pytest.mark.parametrize(
    "replacement",
    [
        "uint16_t(__crossgl_bfloat16_from_float(float(src.Load(0))))",
        (
            "uint16_t(__crossgl_bfloat16_from_float("
            "float((src.Load(0)).real + (src.Load(0)).real)))"
        ),
        "uint16_t(float((src.Load(0)).real))",
    ],
)
def test_copy_directx_conversion_rejects_malformed_complex_lowering(
    tmp_path,
    replacement,
):
    module = _load_proof()
    safe = "uint16_t(__crossgl_bfloat16_from_float(float((src.Load(0)).real)))"
    artifact_path = tmp_path / "copy.hlsl"
    artifact_path.write_text(
        _generated_hlsl().replace(safe, replacement),
        encoding="utf-8",
    )

    with pytest.raises(module.MlxCopyDirectXProofError):
        module._validate_conversion(artifact_path)


def test_copy_directx_fake_dxc_uses_derived_contract(tmp_path, monkeypatch):
    module = _load_proof()
    mlx_root = tmp_path / "mlx"
    work_dir = mlx_root / "proof"
    log_dir = work_dir / "logs"
    artifact_path = work_dir / "artifacts" / "copy.hlsl"
    artifact_path.parent.mkdir(parents=True)
    artifact_path.write_text(_generated_hlsl(), encoding="utf-8")
    _conversion, contract = module._validate_conversion(artifact_path)
    captured = {}

    def run_command(name, command, *, log_dir, timeout_seconds=180):
        del timeout_seconds
        captured["name"] = name
        captured["command"] = list(command)
        output_path = Path(command[command.index("-Fo") + 1])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"DXIL")
        stdout_path = log_dir / f"{name}.stdout"
        stderr_path = log_dir / f"{name}.stderr"
        log_dir.mkdir(parents=True, exist_ok=True)
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

    assert captured["name"] == "compile-copy-directx"
    assert captured["command"] == [
        "C:/tools/dxc.exe",
        "-WX",
        "-T",
        "cs_6_2",
        "-enable-16bit-types",
        "-E",
        "CSMain",
        str(artifact_path),
        "-Fo",
        str(work_dir / "native" / "directx" / "s_copycomplex64bfloat16.dxil"),
    ]
    assert result["status"] == "compiled"
    assert result["required"] is True
    assert result["entryPoint"] == "CSMain"
    assert result["profile"] == "cs_6_2"
    assert result["compilerArguments"] == ["-enable-16bit-types"]
    assert result["compiledArtifactCount"] == 1


def test_copy_directx_toolchain_requirement_fails_closed(tmp_path):
    module = _load_proof()
    mlx_root = tmp_path / "mlx"
    artifact_path = mlx_root / "copy.hlsl"
    artifact_path.parent.mkdir()
    artifact_path.write_text(_generated_hlsl(), encoding="utf-8")
    _conversion, contract = module._validate_conversion(artifact_path)

    optional = module._compile_directx_artifact(
        artifact_path,
        contract,
        dxc=None,
        mlx_root=mlx_root,
        work_dir=mlx_root / "proof",
        log_dir=mlx_root / "proof" / "logs",
        required=False,
    )
    assert optional == {
        "required": False,
        "compiler": "dxc",
        "entryPoint": "CSMain",
        "profile": "cs_6_2",
        "compilerArguments": ["-enable-16bit-types"],
        "available": False,
        "status": "not-required",
        "reason": "dxc-unavailable",
        "compiledArtifactCount": 0,
    }

    with pytest.raises(module.MlxCopyDirectXProofError, match="requires dxc"):
        module._compile_directx_artifact(
            artifact_path,
            contract,
            dxc=None,
            mlx_root=mlx_root,
            work_dir=mlx_root / "proof",
            log_dir=mlx_root / "proof" / "logs",
            required=True,
        )


def test_copy_directx_structural_run_denies_runtime_and_numerical_claims(
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

    summary = module.run_proof(mlx_root, work_dir, require_toolchain=False)

    assert summary["status"] == "passed"
    assert summary["scope"]["fallbackUsed"] is False
    assert summary["scope"]["toolchainRequired"] is False
    assert summary["claims"] == {
        "projectTranslation": True,
        "structuralConversionValidation": True,
        "nativeCompilation": False,
        "runtimeExecution": False,
        "numericalParity": False,
        "mlxUnitTests": False,
        "fullMlxTestSuite": False,
    }
    assert summary["toolchain"]["status"] == "not-required"
    assert summary["translation"]["templateMaterialization"]["specializationCount"] == 1


def test_copy_directx_cli_exposes_fail_closed_toolchain_flag():
    module = _load_proof()
    args = module.parse_args(
        ["--mlx-root", "/tmp/mlx", "--require-toolchain", "--no-clean"]
    )

    assert args.require_toolchain is True
    assert args.no_clean is True


def test_mlx_workflow_requires_copy_directx_proof_on_windows():
    workflow = WORKFLOW_PATH.read_text(encoding="utf-8")

    assert "name: Prove pinned MLX complex copy DirectX lowering" in workflow
    assert "if: runner.os == 'Windows'" in workflow
    assert "python demos/integrations/mlx/prove_copy_directx.py" in workflow
    assert "--work-dir .crosstl-mlx-porting/copy-directx" in workflow
    assert "--require-toolchain" in workflow
    assert 'MLX_COMMIT: "4367c73b60541ddd5a266ce4644fd93d20223b6e"' in workflow

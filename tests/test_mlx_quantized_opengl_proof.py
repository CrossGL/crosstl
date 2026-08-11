import copy
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PROOF_PATH = ROOT / "demos" / "integrations" / "mlx" / "prove_quantized_opengl.py"

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
        "mlx_quantized_opengl_proof",
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


def _generated_glsl():
    return """#version 450 core
#extension GL_ARB_gpu_shader_int64 : require
#extension GL_KHR_shader_subgroup_basic : require
#extension GL_KHR_shader_subgroup_arithmetic : require
#extension GL_KHR_shader_subgroup_shuffle_relative : require
layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
void main() {
    float val_0 = w[uint((in_index + uint64_t(i)))];
    w_min = subgroupMin(w_min);
    w_max = subgroupMax(w_max);
    float scale = max(((w_max - w_min) / n_bins), eps);
    float q0 = round((edge / scale));
    scales[uint(gindex)] = float(scale);
    biases[uint(gindex)] = float(bias);
    uint val = bitfieldExtract(uint(min(round(((w_thread[i] - bias) / scale)), n_bins)), 0, 8);
    uint sval = subgroupShuffleDown(val, j);
    out_[uint((out_index / uint64_t(writes_per_reduce)))] = output_;
}
"""


def _translated_payload(module, mlx_root, work_dir, generated=None):
    artifact_path = work_dir / "artifacts" / "opengl" / "quantized.glsl"
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(generated or _generated_glsl(), encoding="utf-8")
    artifact_hash = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
    assertions = [dict(assertion) for assertion in module.INDEX_RANGE_ASSERTIONS]
    payload = {
        "kind": "crosstl-project-portability-report",
        "project": {
            "indexRangeAssertionCount": 3,
            "indexRangeAssertions": assertions,
        },
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
                "target": "opengl",
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
                "requiredCapabilities": [],
                "entryPoint": {
                    "source": module.MLX_QUANTIZED_ENTRY_POINT,
                    "target": "main",
                    "stage": "compute",
                },
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


def test_quantized_opengl_proof_pins_revision_files_entry_and_ranges():
    module = _load_proof()

    assert module.MLX_COMMIT == "4367c73b60541ddd5a266ce4644fd93d20223b6e"
    assert module.PINNED_FILE_SHA256 == EXPECTED_PINNED_HASHES
    assert module.MLX_QUANTIZED_ENTRY_POINT == "affine_quantize_float_gs_32_b_2"
    assert module.INDEX_RANGE_EXPRESSIONS == (
        "in_index + i",
        "gindex",
        "out_index / writes_per_reduce",
    )
    assert [dict(assertion) for assertion in module.INDEX_RANGE_ASSERTIONS] == [
        {
            "source": module.MLX_QUANTIZED_SOURCE,
            "expression": expression,
            "minimum": 0,
            "maximum": 2147483647,
        }
        for expression in module.INDEX_RANGE_EXPRESSIONS
    ]
    assert module.NON_RUNTIME_CLAIMS == {
        "runtimeExecution": False,
        "numericalParity": False,
        "mlxUnitTests": False,
        "fullMlxTestSuite": False,
    }


def test_quantized_opengl_checkout_fails_closed_on_revision_and_file_drift(
    tmp_path,
    monkeypatch,
):
    module = _load_proof()
    mlx_root = _synthetic_checkout(module, tmp_path, monkeypatch)
    assert module._verify_checkout(mlx_root)["status"] == "passed"

    monkeypatch.setattr(module, "_git_revision", lambda _root: "0" * 40)
    with pytest.raises(module.MlxQuantizedOpenGLProofError, match="must be pinned"):
        module._verify_checkout(mlx_root)

    monkeypatch.setattr(module, "_git_revision", lambda _root: module.MLX_COMMIT)
    header = mlx_root / module.MLX_QUANTIZED_HEADER
    header.write_text("drifted\n", encoding="utf-8")
    with pytest.raises(
        module.MlxQuantizedOpenGLProofError,
        match=r"SHA-256 mismatch for .*quantized\.h",
    ):
        module._verify_checkout(mlx_root)


def test_quantized_opengl_paths_and_provenance_fail_closed(tmp_path, monkeypatch):
    module = _load_proof()
    mlx_root = tmp_path / "mlx"
    mlx_root.mkdir()

    with pytest.raises(module.MlxQuantizedOpenGLProofError, match="inside"):
        module._resolve_work_dir(mlx_root, str(mlx_root))
    with pytest.raises(module.MlxQuantizedOpenGLProofError, match="inside"):
        module._resolve_work_dir(mlx_root, str(tmp_path / "outside"))

    work_dir = mlx_root / "proof"
    work_dir.mkdir()
    marker = work_dir / "existing.txt"
    marker.write_text("preserve\n", encoding="utf-8")
    translated = False

    def reject_provenance(_root):
        raise module.MlxQuantizedOpenGLProofError("provenance drift")

    def unexpected_translation(*_args, **_kwargs):
        nonlocal translated
        translated = True
        raise AssertionError("translation must not run")

    monkeypatch.setattr(module, "_verify_checkout", reject_provenance)
    monkeypatch.setattr(module, "_translate_report", unexpected_translation)
    with pytest.raises(module.MlxQuantizedOpenGLProofError, match="provenance drift"):
        module.run_proof(mlx_root, work_dir)

    assert marker.read_text(encoding="utf-8") == "preserve\n"
    assert translated is False


def test_quantized_opengl_project_config_uses_public_entry_and_range_contract(tmp_path):
    module = _load_proof()
    mlx_root = tmp_path / "mlx"
    mlx_root.mkdir()
    config = module._project_config(mlx_root, mlx_root / "proof")

    assert tuple(config.source_roots) == (module.MLX_KERNEL_ROOT,)
    assert tuple(config.include_patterns) == (module.MLX_QUANTIZED_SOURCE,)
    assert tuple(config.targets) == ("opengl",)
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
    assert [assertion.to_json() for assertion in config.index_range_assertions] == [
        dict(assertion) for assertion in module.INDEX_RANGE_ASSERTIONS
    ]


def test_quantized_opengl_translation_uses_public_project_api(tmp_path, monkeypatch):
    module = _load_proof()
    mlx_root = tmp_path / "mlx"
    mlx_root.mkdir()
    config = module._project_config(mlx_root, mlx_root / "proof")
    report_path = mlx_root / "proof" / "report.json"
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


def test_quantized_opengl_artifact_and_index_contract_reject_report_drift(tmp_path):
    module = _load_proof()
    mlx_root = tmp_path / "mlx"
    work_dir = mlx_root / "proof"
    payload, artifact_path = _translated_payload(module, mlx_root, work_dir)

    assert module._require_index_range_contract(payload)["assertionCount"] == 3
    artifact, resolved = module._translated_artifact(
        payload,
        mlx_root=mlx_root,
        work_dir=work_dir,
    )
    assert artifact["requiredCapabilities"] == []
    assert resolved == artifact_path
    assert module._validate_generated_glsl(artifact_path)["status"] == "passed"

    wrong_ranges = copy.deepcopy(payload)
    wrong_ranges["project"]["indexRangeAssertions"][0]["maximum"] = 2**32 - 1
    with pytest.raises(module.MlxQuantizedOpenGLProofError, match="index-range"):
        module._require_index_range_contract(wrong_ranges)

    wrong_source = copy.deepcopy(payload)
    wrong_source["artifacts"][0]["source"] = "other.metal"
    with pytest.raises(module.MlxQuantizedOpenGLProofError, match="provenance"):
        module._translated_artifact(
            wrong_source,
            mlx_root=mlx_root,
            work_dir=work_dir,
        )

    artifact_path.write_text(_generated_glsl() + "// drift\n", encoding="utf-8")
    with pytest.raises(module.MlxQuantizedOpenGLProofError, match="identity"):
        module._translated_artifact(
            payload,
            mlx_root=mlx_root,
            work_dir=work_dir,
        )


@pytest.mark.parametrize(
    "removed,message",
    [
        ("w_min = subgroupMin(w_min);", "quantization computation"),
        ("w[uint((in_index + uint64_t(i)))]", "in_index \\+ i"),
        ("#extension GL_KHR_shader_subgroup_arithmetic : require", "extension"),
    ],
)
def test_quantized_opengl_generated_contract_rejects_semantic_drift(
    tmp_path,
    removed,
    message,
):
    module = _load_proof()
    artifact_path = tmp_path / "quantized.glsl"
    artifact_path.write_text(_generated_glsl().replace(removed, ""), encoding="utf-8")

    with pytest.raises(module.MlxQuantizedOpenGLProofError, match=message):
        module._validate_generated_glsl(artifact_path)


def test_quantized_opengl_toolchain_targets_opengl_spirv13(tmp_path, monkeypatch):
    module = _load_proof()
    mlx_root = tmp_path / "mlx"
    work_dir = mlx_root / "proof"
    log_dir = work_dir / "logs"
    artifact_path = work_dir / "quantized.glsl"
    artifact_path.parent.mkdir(parents=True)
    artifact_path.write_text(_generated_glsl(), encoding="utf-8")
    commands = []

    def run_command(name, command, *, log_dir, timeout_seconds=180):
        del timeout_seconds
        commands.append(list(command))
        log_dir.mkdir(parents=True, exist_ok=True)
        stdout_path = log_dir / f"{name}.stdout"
        stderr_path = log_dir / f"{name}.stderr"
        stdout_path.write_text("", encoding="utf-8")
        stderr_path.write_text("", encoding="utf-8")
        if name == "compile-quantized-opengl":
            output_path = Path(command[command.index("-o") + 1])
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(b"SPIR-V 1.3")
        return {
            "name": name,
            "command": list(command),
            "returncode": 0,
            "stdoutPath": stdout_path,
            "stderrPath": stderr_path,
        }

    monkeypatch.setattr(module, "_run_command", run_command)
    result = module._compile_and_validate(
        artifact_path,
        glslang="/tools/glslangValidator",
        spirv_val="/tools/spirv-val",
        mlx_root=mlx_root,
        work_dir=work_dir,
        log_dir=log_dir,
        required=True,
    )
    output_path = (
        work_dir / "native" / "opengl" / ("affine_quantize_float_gs_32_b_2.spv")
    )

    assert commands == [
        [
            "/tools/glslangValidator",
            "--target-env",
            "opengl",
            "--target-env",
            "spirv1.3",
            "-S",
            "comp",
            str(artifact_path),
            "-o",
            str(output_path),
        ],
        ["/tools/spirv-val", "--target-env", "spv1.3", str(output_path)],
    ]
    assert result["status"] == "compiled-and-validated"
    assert result["compiledArtifactCount"] == 1
    assert result["compilerTarget"] == "OpenGL/SPIR-V 1.3"
    assert result["validatorTarget"] == "SPIR-V 1.3"


def test_quantized_opengl_toolchain_requirement_fails_closed(tmp_path):
    module = _load_proof()
    mlx_root = tmp_path / "mlx"
    mlx_root.mkdir()
    artifact_path = mlx_root / "quantized.glsl"
    artifact_path.write_text(_generated_glsl(), encoding="utf-8")

    optional = module._compile_and_validate(
        artifact_path,
        glslang=None,
        spirv_val=None,
        mlx_root=mlx_root,
        work_dir=mlx_root / "proof",
        log_dir=mlx_root / "proof" / "logs",
        required=False,
    )
    assert optional["status"] == "not-required"
    assert optional["missingTools"] == ["glslangValidator", "spirv-val"]

    with pytest.raises(
        module.MlxQuantizedOpenGLProofError, match="requires these tools"
    ):
        module._compile_and_validate(
            artifact_path,
            glslang=None,
            spirv_val=None,
            mlx_root=mlx_root,
            work_dir=mlx_root / "proof",
            log_dir=mlx_root / "proof" / "logs",
            required=True,
        )


def test_quantized_opengl_run_writes_deterministic_compile_only_summary(
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

    summary = module.run_proof(mlx_root, work_dir)
    summary_path = work_dir / "summary.json"

    assert summary["status"] == "passed"
    assert summary["scope"]["translation"]["indexRangeContract"] == {
        "status": "configured",
        "assertions": [dict(assertion) for assertion in module.INDEX_RANGE_ASSERTIONS],
        "assertionCount": 3,
        "contractKind": "explicit-host-runtime-portability-preconditions",
        "inferred": False,
        "runtimeEnforced": False,
    }
    assert summary["scope"]["runtime"] == {
        "executionAttempted": False,
        "backendIntegrationAttempted": False,
        "mlxTestsRun": False,
    }
    assert summary["claims"] == {
        "projectTranslation": True,
        "nativeCompilation": False,
        "spirvValidation": False,
        "runtimeExecution": False,
        "numericalParity": False,
        "mlxUnitTests": False,
        "fullMlxTestSuite": False,
    }
    assert summary["toolchain"]["status"] == "not-required"
    assert summary_path.read_text(encoding="utf-8") == (
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )


def test_quantized_opengl_cli_writes_failure_summary_inside_work_dir(
    tmp_path,
    monkeypatch,
):
    module = _load_proof()
    mlx_root = tmp_path / "mlx"
    mlx_root.mkdir()

    def fail_run(*_args, **_kwargs):
        raise module.MlxQuantizedOpenGLProofError("translation failed closed")

    monkeypatch.setattr(module, "run_proof", fail_run)
    result = module.main(
        [
            "--mlx-root",
            str(mlx_root),
            "--work-dir",
            "proof",
            "--require-opengl-toolchain",
        ]
    )
    summary = json.loads(
        (mlx_root / "proof" / "summary.json").read_text(encoding="utf-8")
    )

    assert result == 1
    assert summary["status"] == "failed"
    assert summary["scope"]["toolchain"]["required"] is True
    assert summary["scope"]["runtime"]["executionAttempted"] is False
    assert summary["scope"]["numerical"]["parityClaimed"] is False
    assert summary["claims"]["runtimeExecution"] is False
    assert summary["claims"]["numericalParity"] is False


def test_quantized_opengl_cli_exposes_required_toolchain_flag():
    module = _load_proof()
    args = module.parse_args(
        [
            "--mlx-root",
            "/tmp/mlx",
            "--require-opengl-toolchain",
            "--no-clean",
        ]
    )

    assert args.require_opengl_toolchain is True
    assert args.no_clean is True

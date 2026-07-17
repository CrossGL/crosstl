import hashlib
import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PROOF_PATH = ROOT / "demos" / "integrations" / "mlx" / "prove_copy_opengl.py"
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
    spec = importlib.util.spec_from_file_location("mlx_copy_opengl_proof", PROOF_PATH)
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


def _known_materialization_payload(module):
    message = (
        "Metal template materialization work budget exceeded while running "
        "explicit-template-materialization for target 'opengl'; 65 work items "
        "requested for reachable source entry 'g3_copybool_uint32', limit 64."
    )
    return {
        "summary": {
            "artifactCount": 1,
            "translatedCount": 0,
            "failedCount": 1,
            "diagnosticCounts": {"error": 1},
        },
        "diagnostics": [
            {
                "severity": "error",
                "code": module.MATERIALIZATION_DIAGNOSTIC,
                "message": message,
                "location": {"file": module.MLX_COPY_SOURCE},
                "target": "opengl",
                "sourceBackend": "metal",
                "missingCapabilities": ["template.specialization"],
                "details": {
                    "templateMaterialization": {
                        "limit": module.MATERIALIZATION_WORK_LIMIT,
                        "requiredWorkItems": module.MATERIALIZATION_WORK_LIMIT + 1,
                        "requestedSignature": (
                            "explicit-template-materialization: 65 work items for "
                            "reachable source entry 'g3_copybool_uint32'"
                        ),
                        "accounting": {
                            "reachableSpecializationCount": 65,
                            "dependencyDiscoveryWorkCount": 0,
                            "prunedCandidateCount": 69851,
                        },
                    }
                },
            }
        ],
        "artifacts": [
            {
                "source": module.MLX_COPY_SOURCE,
                "sourceBackend": "metal",
                "target": "opengl",
                "status": "failed",
                "error": message,
            }
        ],
    }


def _translated_payload(module, mlx_root, work_dir):
    wrapper_path = work_dir / "source" / module.COPY_WRAPPER_NAME
    wrapper_path.parent.mkdir(parents=True, exist_ok=True)
    wrapper_path.write_text(module.COPY_WRAPPER_SOURCE, encoding="utf-8")
    source = wrapper_path.relative_to(mlx_root).as_posix()
    artifact_path = work_dir / "artifacts" / "copy.glsl"
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(
        "#version 450 core\n"
        "layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;\n"
        "void main() {\n"
        "    dst[(index + uint(i))] = (src[0]).real;\n"
        "}\n",
        encoding="utf-8",
    )
    artifact_hash = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
    return (
        {
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
                    "source": source,
                    "sourceBackend": "metal",
                    "target": "opengl",
                    "status": "translated",
                    "path": artifact_path.relative_to(mlx_root).as_posix(),
                    "sourceHash": {
                        "algorithm": "sha256",
                        "value": module.COPY_WRAPPER_SHA256,
                    },
                    "generatedHash": {
                        "algorithm": "sha256",
                        "value": artifact_hash,
                    },
                    "entryPoint": {
                        "source": module.MLX_COPY_ENTRY_POINT,
                        "target": "main",
                        "stage": "compute",
                    },
                    "templateMaterialization": {
                        "status": "materialized",
                        "specializationCount": 1,
                        "specializations": [
                            {
                                "name": module.MLX_COPY_TEMPLATE,
                                "hostName": module.MLX_COPY_ENTRY_POINT,
                                "materializedName": module.MLX_COPY_ENTRY_POINT,
                                "parameters": module.MLX_COPY_TEMPLATE_ARGUMENTS,
                            }
                        ],
                        "unsupported": [],
                    },
                }
            ],
        },
        source,
        artifact_path,
    )


def test_copy_opengl_proof_pins_revision_source_headers_and_wrapper():
    module = _load_proof()

    assert module.MLX_COMMIT == "4367c73b60541ddd5a266ce4644fd93d20223b6e"
    assert module.PINNED_FILE_SHA256 == EXPECTED_PINNED_HASHES
    assert module.MLX_COPY_DECLARED_ENTRY_COUNT == 2496
    assert module.MLX_COPY_PREPROCESSED_INSTANTIATION_COUNT == 2497
    assert hashlib.sha256(module.COPY_WRAPPER_SOURCE.encode()).hexdigest() == (
        module.COPY_WRAPPER_SHA256
    )
    assert module.COPY_WRAPPER_SOURCE.splitlines() == [
        '#include "mlx/backend/metal/kernels/utils.h"',
        '#include "mlx/backend/metal/kernels/copy.h"',
        "",
        'instantiate_kernel("s_copycomplex64float32", copy_s, '
        "complex64_t, float, 1)",
    ]
    assert module.NON_RUNTIME_CLAIMS == {
        "runtimeExecution": False,
        "numericalParity": False,
        "mlxUnitTests": False,
        "fullMlxTestSuite": False,
    }


def test_copy_opengl_provenance_fails_closed_on_revision_drift(tmp_path, monkeypatch):
    module = _load_proof()
    mlx_root = _synthetic_checkout(module, tmp_path, monkeypatch)
    monkeypatch.setattr(module, "_git_revision", lambda _root: "0" * 40)

    with pytest.raises(module.MlxCopyOpenGLProofError, match="must be pinned"):
        module._verify_checkout(mlx_root)


def test_copy_opengl_provenance_fails_closed_on_header_drift(tmp_path, monkeypatch):
    module = _load_proof()
    mlx_root = _synthetic_checkout(module, tmp_path, monkeypatch)
    assert module._verify_checkout(mlx_root)["status"] == "passed"

    complex_header = mlx_root / "mlx/backend/metal/kernels/complex.h"
    complex_header.write_text("drifted\n", encoding="utf-8")

    with pytest.raises(
        module.MlxCopyOpenGLProofError,
        match=r"SHA-256 mismatch for .*complex\.h",
    ):
        module._verify_checkout(mlx_root)


def test_copy_opengl_provenance_precedes_cleanup_and_translation(tmp_path, monkeypatch):
    module = _load_proof()
    mlx_root = tmp_path / "mlx"
    mlx_root.mkdir()
    work_dir = mlx_root / "proof"
    work_dir.mkdir()
    marker = work_dir / "existing.txt"
    marker.write_text("preserve on provenance failure\n", encoding="utf-8")
    translated = False

    def reject_provenance(_root):
        raise module.MlxCopyOpenGLProofError("provenance drift")

    def unexpected_translation(*_args, **_kwargs):
        nonlocal translated
        translated = True
        raise AssertionError("translation must not run")

    monkeypatch.setattr(module, "_verify_checkout", reject_provenance)
    monkeypatch.setattr(module, "_translate_report", unexpected_translation)

    with pytest.raises(module.MlxCopyOpenGLProofError, match="provenance drift"):
        module.run_proof(mlx_root, work_dir)

    assert marker.read_text(encoding="utf-8") == ("preserve on provenance failure\n")
    assert translated is False


def test_copy_opengl_project_configs_select_one_entry_with_finite_limits(tmp_path):
    module = _load_proof()
    mlx_root = tmp_path / "mlx"
    mlx_root.mkdir()
    work_dir = mlx_root / "proof"

    real = module._project_config(
        mlx_root,
        work_dir,
        source=module.MLX_COPY_SOURCE,
        output_name="real",
    )
    wrapper_source = "proof/source/copy_complex64_float32.metal"
    wrapper = module._project_config(
        mlx_root,
        work_dir,
        source=wrapper_source,
        output_name="wrapper",
    )

    for config, source in (
        (real, module.MLX_COPY_SOURCE),
        (wrapper, wrapper_source),
    ):
        assert tuple(config.targets) == ("opengl",)
        assert tuple(config.include_patterns) == (source,)
        assert config.entry_points == {source: module.MLX_COPY_ENTRY_POINT}
        assert tuple(config.include_dirs) == (".",)
        assert config.source_overrides == {source: "metal"}
        assert config.source_options == {
            "metal": {
                "max_template_specializations": 16,
                "max_template_materialization_work": 64,
            }
        }


def test_copy_opengl_wrapper_fallback_requires_the_known_bounded_blocker():
    module = _load_proof()
    payload = _known_materialization_payload(module)

    result = module._classify_real_source_probe(payload)

    assert result["status"] == "bounded-materialization-blocked"
    assert result["fallbackRequired"] is True
    assert result["materializationWorkLimit"] == 64
    assert result["requiredWorkItems"] == 65

    payload["diagnostics"][0]["code"] = "project.translate.failed"
    with pytest.raises(
        module.MlxCopyOpenGLProofError,
        match="known materialization blocker",
    ):
        module._classify_real_source_probe(payload)


def test_copy_opengl_artifact_and_projection_drift_fail_closed(tmp_path):
    module = _load_proof()
    mlx_root = tmp_path / "mlx"
    work_dir = mlx_root / "proof"
    work_dir.mkdir(parents=True)
    payload, source, artifact_path = _translated_payload(module, mlx_root, work_dir)

    artifact, resolved_path = module._translated_artifact(
        payload,
        mlx_root=mlx_root,
        work_dir=work_dir,
        expected_source=source,
        expected_source_hash=module.COPY_WRAPPER_SHA256,
        require_single_materialization=True,
    )
    assert artifact["entryPoint"]["source"] == module.MLX_COPY_ENTRY_POINT
    assert resolved_path == artifact_path
    assert module._validate_real_projection(artifact_path)["singleEvaluation"] is True

    invalid_source = artifact_path.read_text(encoding="utf-8").replace(
        "(src[0]).real", "float(src[0])"
    )
    artifact_path.write_text(invalid_source, encoding="utf-8")
    with pytest.raises(
        module.MlxCopyOpenGLProofError,
        match="artifact hash does not match",
    ):
        module._translated_artifact(
            payload,
            mlx_root=mlx_root,
            work_dir=work_dir,
            expected_source=source,
            expected_source_hash=module.COPY_WRAPPER_SHA256,
            require_single_materialization=True,
        )
    with pytest.raises(
        module.MlxCopyOpenGLProofError,
        match=r"projection of src\[0\]\.real",
    ):
        module._validate_real_projection(artifact_path)


def test_copy_opengl_toolchain_targets_opengl_spirv13(tmp_path, monkeypatch):
    module = _load_proof()
    mlx_root = tmp_path / "mlx"
    work_dir = mlx_root / "proof"
    log_dir = work_dir / "logs"
    artifact_path = work_dir / "copy.glsl"
    artifact_path.parent.mkdir(parents=True)
    artifact_path.write_text("#version 450 core\n", encoding="utf-8")
    tools = {
        "glslangValidator": "/tools/glslangValidator",
        "spirv-val": "/tools/spirv-val",
    }
    commands = []

    monkeypatch.setattr(module.shutil, "which", tools.get)

    def run_command(name, command, *, log_dir, timeout_seconds=180):
        del timeout_seconds
        commands.append(list(command))
        stdout_path = log_dir / f"{name}.stdout"
        stderr_path = log_dir / f"{name}.stderr"
        log_dir.mkdir(parents=True, exist_ok=True)
        stdout_path.write_text("", encoding="utf-8")
        stderr_path.write_text("", encoding="utf-8")
        if name == "compile-copy-opengl":
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
        mlx_root=mlx_root,
        work_dir=work_dir,
        log_dir=log_dir,
    )

    assert commands[0] == [
        "/tools/glslangValidator",
        "--target-env",
        "opengl",
        "--target-env",
        "spirv1.3",
        "-S",
        "comp",
        str(artifact_path),
        "-o",
        str(work_dir / "toolchain" / "s_copycomplex64float32.spv"),
    ]
    assert commands[1] == [
        "/tools/spirv-val",
        "--target-env",
        "spv1.3",
        str(work_dir / "toolchain" / "s_copycomplex64float32.spv"),
    ]
    assert result["status"] == "compiled-and-validated"
    assert result["compilerTarget"] == "OpenGL/SPIR-V 1.3"
    assert result["validatorTarget"] == "SPIR-V 1.3"


def test_mlx_workflow_requires_copy_opengl_proof_on_linux():
    workflow = WORKFLOW_PATH.read_text(encoding="utf-8")

    assert "name: Prove pinned MLX complex copy OpenGL lowering" in workflow
    assert "if: runner.os == 'Linux'" in workflow
    assert "python demos/integrations/mlx/prove_copy_opengl.py" in workflow
    assert "--work-dir .crosstl-mlx-porting/copy-opengl" in workflow
    assert 'MLX_COMMIT: "4367c73b60541ddd5a266ce4644fd93d20223b6e"' in workflow

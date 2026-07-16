import json
import shutil
import subprocess
import textwrap

import pytest

import crosstl.project.pipeline as project_pipeline
from crosstl.project import (
    ProjectConfig,
    build_runtime_artifact_manifest,
    build_runtime_loader_manifest,
    build_runtime_package,
    translate_project,
    validate_project_report,
)

MULTI_ENTRY_COMPUTE = textwrap.dedent("""
    shader StandaloneDirectXCompute {
        const uint selected_stride = 2u;
        const uint selected_scale = selected_stride;
        const uint unrelated_stride = 9u;

        cbuffer SelectedConfig @set(2) @binding(2) {
            uint selected_bias;
        };

        cbuffer UnrelatedConfig @set(3) @binding(3) {
            uint unrelated_bias;
        };

        StructuredBuffer<uint> selected_input @set(1) @binding(0);
        RWStructuredBuffer<uint> selected_output @set(1) @binding(1);
        StructuredBuffer<uint> unrelated_input @set(3) @binding(4);
        RWStructuredBuffer<uint> unrelated_output @set(3) @binding(5);

        uint selected_helper(uint value) {
            return value * selected_scale + selected_bias;
        }

        uint unrelated_helper(uint value) {
            return value * unrelated_stride + unrelated_bias;
        }

        compute first {
            @numthreads(8, 1, 1)
            void main(uint index @gl_GlobalInvocationID) {
                unrelated_output[index] = unrelated_helper(unrelated_input[index]);
            }
        }

        compute second {
            @numthreads(8, 1, 1)
            void main(uint index @gl_GlobalInvocationID) {
                selected_output[index] = selected_helper(selected_input[index]);
            }
        }
    }
    """).strip()

AMBIGUOUS_COMPUTE = textwrap.dedent("""
    shader AmbiguousDirectXCompute {
        compute duplicate {
            void main(uint index @gl_GlobalInvocationID) {}
        }

        compute duplicate {
            void main(uint index @gl_GlobalInvocationID) {}
        }
    }
    """).strip()

VERTEX_ENTRY = textwrap.dedent("""
    shader UnsupportedDirectXStage {
        vertex vertex_entry {
            vec4 main(vec4 position @POSITION) @gl_Position {
                return position;
            }
        }
    }
    """).strip()


def _project_config(root, *, entry_points=None):
    return ProjectConfig(
        root=root,
        include_patterns=("kernels/multi.cgl",),
        targets=("directx",),
        output_dir="translated",
        entry_points=entry_points or {},
    )


def _write_source(tmp_path, source=MULTI_ENTRY_COMPUTE):
    root = tmp_path / "repo"
    source_path = root / "kernels" / "multi.cgl"
    source_path.parent.mkdir(parents=True)
    source_path.write_text(source, encoding="utf-8")
    return root


def _assert_hlsl_compute_compiles_if_available(shader_path, tmp_path):
    dxc = shutil.which("dxc")
    if dxc:
        result = subprocess.run(
            [
                dxc,
                "-T",
                "cs_6_0",
                "-E",
                "CSMain",
                str(shader_path),
                "-Fo",
                str(tmp_path / "selected.dxil"),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        return

    glslang = shutil.which("glslangValidator")
    if not glslang:
        return
    spirv_path = tmp_path / "selected.spv"
    result = subprocess.run(
        [
            glslang,
            "-D",
            "-V",
            "-S",
            "comp",
            "-e",
            "CSMain",
            str(shader_path),
            "-o",
            str(spirv_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr

    spirv_val = shutil.which("spirv-val")
    if spirv_val:
        validation = subprocess.run(
            [spirv_val, str(spirv_path)],
            capture_output=True,
            text=True,
            check=False,
        )
        assert validation.returncode == 0, validation.stdout + validation.stderr


@pytest.mark.parametrize(
    "entry_points",
    (
        pytest.param(
            {"kernels/*.cgl": "first", "kernels/multi.cgl": "second"},
            id="glob-before-exact",
        ),
        pytest.param(
            {"kernels/multi.cgl": "second", "kernels/*.cgl": "first"},
            id="exact-before-glob",
        ),
    ),
)
def test_selected_directx_compute_is_standalone_and_replayable(
    tmp_path,
    entry_points,
):
    root = _write_source(tmp_path)
    config = _project_config(root, entry_points=entry_points)

    report = translate_project(config, format_output=False, validate=True)
    payload = report.to_json()
    artifact = payload["artifacts"][0]
    shader_path = root / artifact["path"]
    generated = shader_path.read_text(encoding="utf-8")

    assert payload["summary"]["translatedCount"] == 1
    assert payload["summary"]["failedCount"] == 0
    assert artifact["path"] == "translated/directx/kernels/multi/second.hlsl"
    assert artifact["entryPoint"] == {
        "source": "second",
        "target": "CSMain",
        "stage": "compute",
    }
    assert artifact["provenance"]["pipeline"] == "entry-scoped-translate"
    assert artifact["sourceMap"]["generated"]["file"] == artifact["path"]
    assert artifact["sourceRemap"]["generatedFile"] == artifact["path"]
    assert artifact["sourceRemap"]["mappingCount"] == len(
        artifact["sourceMap"]["mappings"]
    )
    assert project_pipeline._directx_dxc_entry_profiles(  # noqa: SLF001
        shader_path,
        artifact=artifact,
    ) == (("CSMain", "cs_6_0"),)

    assert generated.count("void CSMain(") == 1
    assert "[numthreads(8, 1, 1)]" in generated
    assert "selected_helper" in generated
    assert "selected_stride" in generated
    assert "selected_scale" in generated
    assert "selected_input : register(t0, space1)" in generated
    assert "selected_output : register(u1, space1)" in generated
    assert "SelectedConfig : register(b2, space2)" in generated
    assert "unrelated_" not in generated
    assert "UnrelatedConfig" not in generated

    report_path = root / "portability-report.json"
    report.write_json(report_path)
    assert validate_project_report(report_path)["success"] is True

    manifest = build_runtime_artifact_manifest(report_path)
    assert manifest["success"] is True
    assert manifest["summary"]["artifactCount"] == 1
    runtime_artifact = manifest["artifacts"][0]
    assert runtime_artifact["entryPoints"] == [
        {
            "name": "CSMain",
            "stage": "compute",
            "executionConfig": {"numthreads": [8, 1, 1]},
        }
    ]
    assert {
        (
            binding["name"],
            binding["kind"],
            binding["set"],
            binding["binding"],
            binding["access"],
        )
        for binding in runtime_artifact["resourceBindings"]
    } == {
        ("selected_input", "buffer", 1, 0, "read"),
        ("selected_output", "buffer", 1, 1, "read_write"),
        ("SelectedConfig", "constant-buffer", 2, 2, "read"),
    }
    assert "unrelated_" not in json.dumps(runtime_artifact, sort_keys=True)

    runtime_manifest_path = root / "runtime-artifact-manifest.json"
    runtime_manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    package_dir = root / "runtime-package"
    package = build_runtime_package(runtime_manifest_path, package_dir)
    loader = build_runtime_loader_manifest(package_dir / "runtime-package.json")
    load_unit = loader["loadUnits"][0]
    load_step = load_unit["loadSteps"][0]
    validation_step = load_unit["loadSteps"][-1]

    assert package["success"] is True
    assert loader["success"] is True
    assert load_unit["hostInterface"]["entryPoints"] == [
        {
            "name": "CSMain",
            "stage": "compute",
            "executionConfig": {"numthreads": [8, 1, 1]},
        }
    ]
    assert load_step["metadata"]["entryProfiles"] == [
        {"entry": "CSMain", "profile": "cs_6_0"}
    ]
    assert validation_step["metadata"]["commandInput"]["entryPoint"] == "CSMain"
    assert validation_step["metadata"]["commandInput"]["shaderModel"] == "cs_6_0"

    repeated = translate_project(config, format_output=False).to_json()["artifacts"][0]
    assert repeated["path"] == artifact["path"]
    assert repeated["generatedHash"] == artifact["generatedHash"]
    _assert_hlsl_compute_compiles_if_available(shader_path, tmp_path)


def test_directx_project_preserves_aggregate_output_without_selection(tmp_path):
    root = _write_source(tmp_path)

    payload = translate_project(
        _project_config(root),
        format_output=False,
    ).to_json()
    artifact = payload["artifacts"][0]
    generated = (root / artifact["path"]).read_text(encoding="utf-8")

    assert artifact["path"] == "translated/directx/kernels/multi.hlsl"
    assert artifact["provenance"]["pipeline"] == "single-file-translate"
    assert "entryPoint" not in artifact
    assert generated.count("[numthreads(") == 2
    assert "selected_helper" in generated
    assert "unrelated_helper" in generated
    assert "selected_input" in generated
    assert "unrelated_input" in generated
    assert "SelectedConfig" in generated
    assert "UnrelatedConfig" in generated


@pytest.mark.parametrize(
    ("source", "selection", "reason", "available", "stage"),
    (
        pytest.param(
            MULTI_ENTRY_COMPUTE,
            "missing",
            "not-found",
            ["first", "second"],
            None,
            id="missing",
        ),
        pytest.param(
            AMBIGUOUS_COMPUTE,
            "duplicate",
            "ambiguous",
            ["duplicate"],
            None,
            id="ambiguous",
        ),
        pytest.param(
            VERTEX_ENTRY,
            "vertex_entry",
            "unsupported-stage",
            ["vertex_entry"],
            "vertex",
            id="unsupported-stage",
        ),
    ),
)
def test_directx_entry_selection_failures_are_structured_and_write_no_file(
    tmp_path,
    source,
    selection,
    reason,
    available,
    stage,
):
    root = _write_source(tmp_path, source)
    config = _project_config(
        root,
        entry_points={"kernels/multi.cgl": selection},
    )

    payload = translate_project(config, format_output=False).to_json()
    artifact = payload["artifacts"][0]

    assert artifact["status"] == "failed"
    assert not (root / artifact["path"]).exists()
    assert "generatedHash" not in artifact
    assert "sourceMap" not in artifact
    assert "sourceRemap" not in artifact
    diagnostic = next(
        item
        for item in payload["diagnostics"]
        if item["code"] == "project.translate.directx-entry-point-unavailable"
    )
    assert diagnostic["severity"] == "error"
    assert diagnostic["missingCapabilities"] == ["artifact.entry-point-selection"]
    assert diagnostic["details"]["sourcePath"] == "kernels/multi.cgl"
    assert diagnostic["details"]["targetArtifact"] == artifact["path"]
    selection_details = diagnostic["details"]["entryPointSelection"]
    assert selection_details["entryPoint"] == selection
    assert selection_details["availableEntryPoints"] == available
    assert selection_details["reason"] == reason
    if stage is None:
        assert "stage" not in selection_details
    else:
        assert selection_details["stage"] == stage

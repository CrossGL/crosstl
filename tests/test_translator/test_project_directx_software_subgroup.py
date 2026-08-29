import textwrap

import pytest

from crosstl.project import (
    ProjectConfig,
    build_runtime_artifact_manifest,
    translate_project,
    validate_project_report,
)

SOURCE = "shaders/software_wave.metal"
ENTRY = "software_wave"


def _write_fixture(repo):
    source_path = repo / SOURCE
    source_path.parent.mkdir(parents=True)
    source_path.write_text(
        textwrap.dedent(
            """
            #include <metal_stdlib>
            using namespace metal;

            kernel void software_wave(
                device float* output [[buffer(0)]],
                uint invocation [[thread_index_in_threadgroup]],
                uint subgroup_id [[simdgroup_index_in_threadgroup]],
                uint subgroup_lane [[thread_index_in_simdgroup]]) {
                float value = float(invocation + 1u);
                float shuffled = simd_shuffle_down(value, 1u);
                output[invocation] = shuffled
                    + float(subgroup_id) + float(subgroup_lane);
            }
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )


def _config(repo, *, width=32, workgroup_size=(32, 2, 1)):
    return ProjectConfig(
        root=repo,
        targets=("directx",),
        output_dir="out",
        entry_points={SOURCE: ENTRY},
        workgroup_size=workgroup_size,
        source_options={
            "metal": {
                "target_options": {
                    "directx": {
                        "relative_wave_shuffle_out_of_range": "self",
                        "software_subgroup_width": width,
                    },
                }
            }
        },
    )


def _diagnostic(payload):
    return next(
        item
        for item in payload["diagnostics"]
        if item["code"]
        == "project.translate.directx-software-subgroup-invalid"
    )


def test_project_directx_target_option_emits_logical_software_subgroups(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _write_fixture(repo)

    report = translate_project(
        _config(repo),
        format_output=False,
        validate=True,
    )
    payload = report.to_json()

    assert payload["project"]["sourceOptions"] == {
        "metal": {
            "target_options": {
                "directx": {
                    "relative_wave_shuffle_out_of_range": "self",
                    "software_subgroup_width": 32,
                },
            }
        }
    }
    assert payload["summary"]["translatedCount"] == 1
    assert payload["summary"]["failedCount"] == 0
    artifact = payload["artifacts"][0]
    assert artifact["provenance"] == {
        "pipeline": "entry-scoped-translate",
        "intermediate": "crossgl",
    }
    assert artifact["execution"]["workgroupSize"] == [32, 2, 1]
    assert "subgroupWidth" not in artifact["execution"]
    assert "subgroupWidthEnforcement" not in artifact["execution"]

    generated_path = repo / artifact["path"]
    generated = generated_path.read_text(encoding="utf-8")
    assert "groupshared float __crossgl_software_subgroup_scratch_float[64];" in generated
    assert "__crossgl_software_subgroup_shuffle_down_float" in generated
    assert generated.count("GroupMemoryBarrierWithGroupSync();") == 2
    assert "uint subgroup_id = (invocation / 32u);" in generated
    assert "uint subgroup_lane = (invocation % 32u);" in generated
    assert "WaveReadLaneAt" not in generated
    assert "WaveGetLaneIndex" not in generated
    assert "__crossgl_physical_subgroup" not in generated

    report_path = repo / "report.json"
    report.write_json(report_path)
    assert validate_project_report(report_path)["success"] is True
    runtime_manifest = build_runtime_artifact_manifest(report_path)
    assert runtime_manifest["success"] is True
    execution_config = runtime_manifest["artifacts"][0]["hostInterface"][
        "entryPoints"
    ][0]["executionConfig"]
    assert execution_config == {"numthreads": [32, 2, 1]}
    assert "subgroupWidth" not in execution_config


@pytest.mark.parametrize("width", [True, 16, 64, "32"])
def test_project_directx_rejects_invalid_software_subgroup_width(tmp_path, width):
    repo = tmp_path / "repo"
    repo.mkdir()
    _write_fixture(repo)

    payload = translate_project(
        _config(repo, width=width),
        format_output=False,
    ).to_json()

    assert payload["summary"]["translatedCount"] == 0
    assert payload["summary"]["failedCount"] == 1
    assert payload["artifacts"][0]["status"] == "failed"
    assert not (repo / payload["artifacts"][0]["path"]).exists()
    diagnostic = _diagnostic(payload)
    assert diagnostic["checkKind"] == "execution-specialization"
    assert diagnostic["missingCapabilities"] == [
        "directx.software-subgroup-lowering"
    ]
    specialization = diagnostic["details"]["executionSpecialization"]
    assert specialization["reason"] == "configured-width-invalid"
    assert specialization["softwareSubgroupWidth"] == width


def test_project_directx_rejects_software_subgroup_workgroup_mismatch(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _write_fixture(repo)

    payload = translate_project(
        _config(repo, workgroup_size=(16, 2, 1)),
        format_output=False,
    ).to_json()

    assert payload["summary"]["translatedCount"] == 0
    assert payload["summary"]["failedCount"] == 1
    diagnostic = _diagnostic(payload)
    specialization = diagnostic["details"]["executionSpecialization"]
    assert specialization == {
        "reason": "workgroup-size-mismatch",
        "softwareSubgroupWidth": 32,
        "sourceEntryPoints": [],
        "workgroupSize": ["16", "2", "1"],
    }

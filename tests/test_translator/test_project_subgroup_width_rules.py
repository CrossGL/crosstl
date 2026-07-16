import copy
import json
import textwrap
from dataclasses import replace

import pytest

import crosstl.project as project_api
import crosstl.project.pipeline as project_pipeline


def _write_fixture(repo):
    shader_dir = repo / "shaders"
    shader_dir.mkdir(parents=True)
    (shader_dir / "wave.metal").write_text(
        textwrap.dedent("""
            #include <metal_stdlib>
            using namespace metal;

            template <int WIDTH, int Y>
            [[kernel]] void wave(
                device uint* output [[buffer(0)]],
                uint index [[thread_position_in_grid]],
                uint lane [[thread_index_in_simdgroup]]) {
                output[index] = lane + uint(WIDTH + Y);
            }

            instantiate_kernel("wave32", wave, 32, 1)
            instantiate_kernel("wave64", wave, 64, 2)
            """).strip()
        + "\n",
        encoding="utf-8",
    )


def _config(repo, *, targets=("directx",), with_workgroup=False):
    kwargs = {}
    if with_workgroup:
        kwargs["workgroup_size_rules"] = {
            "shaders/wave.metal": ["WIDTH", "Y", "1"]
        }
    return project_api.ProjectConfig(
        root=repo,
        targets=targets,
        output_dir="out",
        subgroup_width_rules={"shaders/wave.metal": "WIDTH"},
        **kwargs,
    )


def _diagnostic(payload, code):
    return next(item for item in payload["diagnostics"] if item["code"] == code)


def _wrap_materialization(monkeypatch, mutate):
    original = project_pipeline._project_template_materialization_for_artifact

    def wrapped(**kwargs):
        result = original(**kwargs)
        if result is None:
            return None
        metadata = copy.deepcopy(result.metadata)
        mutate(metadata)
        return replace(result, metadata=metadata)

    monkeypatch.setattr(
        project_pipeline,
        "_project_template_materialization_for_artifact",
        wrapped,
    )


@pytest.mark.parametrize("with_workgroup", (False, True))
def test_subgroup_width_rules_emit_exact_directx_contract(tmp_path, with_workgroup):
    repo = tmp_path / "repo"
    repo.mkdir()
    _write_fixture(repo)

    report = project_api.translate_project(
        _config(repo, with_workgroup=with_workgroup),
        format_output=False,
        validate=True,
    )
    payload = report.to_json()

    assert payload["project"]["subgroupWidthRules"] == {
        "shaders/wave.metal": "WIDTH"
    }
    assert payload["project"]["subgroupWidthRuleCount"] == 1
    assert payload["summary"]["translatedCount"] == 1
    assert payload["summary"]["failedCount"] == 0
    artifact = payload["artifacts"][0]
    entries = artifact["execution"]["entryPoints"]
    assert [entry["sourceEntryPoint"] for entry in entries] == ["wave32", "wave64"]
    assert {
        entry["sourceEntryPoint"]: entry["subgroupWidth"] for entry in entries
    } == {"wave32": 32, "wave64": 64}
    assert all(
        entry["subgroupWidthRule"]
        == {
            "expression": "WIDTH",
            "path": 'project.subgroup_width_rules["shaders/wave.metal"]',
            "sourcePattern": "shaders/wave.metal",
        }
        for entry in entries
    )
    if with_workgroup:
        assert {
            entry["sourceEntryPoint"]: entry["workgroupSize"] for entry in entries
        } == {"wave32": [32, 1, 1], "wave64": [64, 2, 1]}

    enforcement = artifact["execution"]["subgroupWidthEnforcement"]
    assert enforcement["mechanism"] == "hlsl-wave-size-attribute"
    assert enforcement["minimumShaderModel"] == "6.6"
    assert {item["profile"] for item in enforcement["entryProfiles"]} == {"cs_6_6"}
    generated_path = repo / artifact["path"]
    generated = generated_path.read_text(encoding="utf-8")
    assert generated.count("[WaveSize(32)]") == 1
    assert generated.count("[WaveSize(64)]") == 1
    assert {
        profile for _entry, profile in project_pipeline._directx_dxc_entry_profiles(
            generated_path, artifact=artifact
        )
    } == {"cs_6_6"}

    report_path = repo / "report.json"
    report.write_json(report_path)
    assert project_api.validate_project_report(report_path)["success"] is True


def test_subgroup_width_rule_join_is_independent_of_record_order(
    tmp_path, monkeypatch
):
    repo = tmp_path / "repo"
    repo.mkdir()
    _write_fixture(repo)
    baseline = project_api.translate_project(_config(repo), format_output=False).to_json()

    def reverse_records(metadata):
        metadata["specializations"] = list(reversed(metadata["specializations"]))

    _wrap_materialization(monkeypatch, reverse_records)
    reordered = project_api.translate_project(
        _config(repo), format_output=False
    ).to_json()

    assert reordered["artifacts"][0]["execution"] == baseline["artifacts"][0][
        "execution"
    ]


@pytest.mark.parametrize(
    ("target", "reason"),
    [
        ("opengl", "opengl-enforcement-unavailable"),
        ("metal", "target-not-supported"),
        ("vulkan", "target-not-supported"),
        ("cuda", "target-not-supported"),
        ("hip", "target-not-supported"),
        ("mojo", "target-not-supported"),
        ("rust", "target-not-supported"),
        ("slang", "target-not-supported"),
        ("webgl", "target-not-supported"),
        ("wgsl", "target-not-supported"),
    ],
)
def test_subgroup_width_rules_fail_closed_without_target_enforcement(
    tmp_path, target, reason
):
    repo = tmp_path / "repo"
    repo.mkdir()
    _write_fixture(repo)

    payload = project_api.translate_project(
        _config(repo, targets=(target,)), format_output=False
    ).to_json()

    assert payload["summary"]["translatedCount"] == 0
    assert payload["summary"]["failedCount"] == 1
    artifact = payload["artifacts"][0]
    assert artifact["status"] == "failed"
    assert not (repo / artifact["path"]).exists()
    diagnostic = _diagnostic(
        payload, "project.translate.subgroup-width-enforcement-unsupported"
    )
    assert diagnostic["checkKind"] == "execution-specialization"
    assert diagnostic["details"]["executionSpecialization"]["reason"] == reason
    assert diagnostic["missingCapabilities"] == [
        "execution.subgroup-width-specialization"
    ]


@pytest.mark.parametrize(
    ("expression", "reason"),
    [
        ("UNKNOWN", "unknown-identifier"),
        ("WIDTH / 0", "division-by-zero"),
        ("0", "non-positive-result"),
        ("3", "target-width-unsupported"),
    ],
)
def test_subgroup_width_rules_reject_invalid_results(tmp_path, expression, reason):
    repo = tmp_path / "repo"
    repo.mkdir()
    _write_fixture(repo)
    config = project_api.ProjectConfig(
        root=repo,
        targets=["directx"],
        output_dir="out",
        subgroup_width_rules={"shaders/wave.metal": expression},
    )

    payload = project_api.translate_project(config, format_output=False).to_json()

    assert payload["summary"]["failedCount"] == 1
    diagnostic = next(
        item
        for item in payload["diagnostics"]
        if item["code"]
        in {
            "project.translate.subgroup-width-invalid",
            "project.translate.subgroup-width-rule-invalid",
        }
    )
    assert diagnostic["details"]["executionSpecialization"]["reason"] == reason
    assert not (repo / payload["artifacts"][0]["path"]).exists()


def test_subgroup_width_report_rejects_rehashed_materialization_tampering(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _write_fixture(repo)
    payload = project_api.translate_project(_config(repo), format_output=False).to_json()
    artifact = payload["artifacts"][0]
    execution = artifact["execution"]
    entry = execution["entryPoints"][0]
    entry["parameters"]["WIDTH"] = "64"
    entry["subgroupWidth"] = 64
    entry["identity"] = project_pipeline._workgroup_rule_entry_identity(
        source=artifact["source"],
        source_hash=artifact["sourceHash"],
        target=artifact["target"],
        variant=None,
        entry=entry,
    )
    execution["identity"] = project_pipeline._subgroup_rule_execution_identity(
        source=artifact["source"],
        source_hash=artifact["sourceHash"],
        target=artifact["target"],
        variant=None,
        execution=execution,
    )
    report_path = repo / "tampered.json"
    report_path.write_text(json.dumps(payload), encoding="utf-8")

    validation = project_api.validate_project_report(report_path)

    assert validation["success"] is False
    invalid = _diagnostic(validation, "project.validate.invalid-report")
    assert "must match its template materialization" in invalid["message"]


@pytest.mark.parametrize(
    "rules",
    [
        {"": "32"},
        {"shaders/wave.metal": ""},
        {"shaders/wave.metal": True},
    ],
)
def test_project_config_rejects_malformed_subgroup_width_rules(tmp_path, rules):
    repo = tmp_path / "repo"
    repo.mkdir()

    with pytest.raises(ValueError, match="subgroup_width_rules"):
        project_api.ProjectConfig(root=repo, subgroup_width_rules=rules)


def test_directx_exact_wave_size_profile_boundary():
    source = textwrap.dedent("""
        [WaveSize(32)]
        [numthreads(32, 1, 1)]
        void CSMain(uint3 tid : SV_DispatchThreadID) {}
        """)

    assert project_pipeline._directx_dxc_profile_for_source("cs_6_0", source) == (
        "cs_6_6"
    )
    assert project_pipeline._directx_dxc_profile_for_source("cs_6_5", source) == (
        "cs_6_6"
    )
    assert project_pipeline._directx_dxc_profile_for_source("cs_6_6", source) == (
        "cs_6_6"
    )

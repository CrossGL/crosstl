import copy
import json
import textwrap
from dataclasses import replace

import pytest

import crosstl.project as project_api
import crosstl.project.pipeline as project_pipeline


def _write_two_entry_fixture(repo):
    shader_dir = repo / "shaders"
    shader_dir.mkdir(parents=True)
    (shader_dir / "tiled.metal").write_text(
        textwrap.dedent("""
            #include <metal_stdlib>
            using namespace metal;

            template <int N>
            uint add_tile_width(uint value) {
                return value + uint(N);
            }

            template <typename T, int BM, int BN>
            [[kernel]] void tiled(
                device T* output [[buffer(0)]],
                uint index [[thread_position_in_grid]],
                uint3 group_size [[threads_per_threadgroup]]) {
                uint value = add_tile_width<BN>(
                    group_size.x + group_size.y + group_size.z);
                output[index] = T(value + uint(BM));
            }

            instantiate_kernel("tile_small", tiled, float, 1, 2)
            instantiate_kernel("tile_large", tiled, float, 2, 4)
            """).strip() + "\n",
        encoding="utf-8",
    )


def _rule_config(repo, *, targets=("directx", "opengl"), output_dir="out"):
    return project_api.ProjectConfig(
        root=repo,
        targets=targets,
        output_dir=output_dir,
        workgroup_size_rules={
            "shaders/tiled.metal": ["32", "BN", "BM"],
        },
    )


def _diagnostic(payload, code):
    return next(item for item in payload["diagnostics"] if item["code"] == code)


def _refresh_artifact_rollups(payload):
    artifacts = payload["artifacts"]
    summary = payload["summary"]
    summary["artifactCount"] = len(artifacts)
    summary["translatedCount"] = sum(
        artifact.get("status") == "translated" for artifact in artifacts
    )
    summary["failedCount"] = sum(
        artifact.get("status") == "failed" for artifact in artifacts
    )
    summary["artifactsBySourceBackend"] = (
        project_pipeline._artifact_counts_by_source_backend(artifacts)
    )
    summary["artifactsByVariant"] = project_pipeline._artifact_counts_by_variant(
        artifacts
    )
    summary["artifactsByTarget"] = project_pipeline._artifact_counts_by_target(
        artifacts
    )
    summary.update(project_pipeline._artifact_provenance_rollups(artifacts))
    summary.update(project_pipeline._source_map_rollups(artifacts))

    matrix = project_pipeline._expected_artifact_matrix_metadata(
        payload["project"], payload["units"], artifacts
    )
    inspected = project_pipeline._inspection_artifact_matrix_summary(
        matrix,
        artifacts,
        project=payload["project"],
        units=payload["units"],
    )
    payload["artifactMatrix"] = {
        field: inspected[field]
        for field in project_pipeline.REPORT_ARTIFACT_MATRIX_FIELDS
    }


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


def test_project_workgroup_rules_emit_directx_library_and_opengl_entries(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _write_two_entry_fixture(repo)
    (repo / "crosstl.toml").write_text(
        textwrap.dedent("""
            [project]
            source_roots = ["shaders"]
            targets = ["directx", "opengl"]
            output_dir = "out"

            [project.workgroup_size_rules]
            "shaders/tiled.metal" = ["32", "BN", "BM"]
            """).strip() + "\n",
        encoding="utf-8",
    )

    report = project_api.translate_project(
        project_api.load_project_config(repo),
        format_output=False,
        validate=True,
    )
    payload = report.to_json()

    assert payload["project"]["workgroupSizeRules"] == {
        "shaders/tiled.metal": ["32", "BN", "BM"]
    }
    assert payload["project"]["workgroupSizeRuleCount"] == 1
    assert payload["summary"]["translatedCount"] == 3
    assert payload["summary"]["failedCount"] == 0

    directx = next(
        artifact for artifact in payload["artifacts"] if artifact["target"] == "directx"
    )
    directx_entries = directx["execution"]["entryPoints"]
    assert [entry["sourceEntryPoint"] for entry in directx_entries] == [
        "tile_large",
        "tile_small",
    ]
    assert {
        entry["sourceEntryPoint"]: entry["workgroupSize"] for entry in directx_entries
    } == {
        "tile_large": [32, 4, 2],
        "tile_small": [32, 2, 1],
    }
    assert [entry["targetEntryPoint"] for entry in directx_entries] == [
        "CSMain_2",
        "CSMain",
    ]
    assert all(
        entry["parameters"]
        == {
            "BM": "2" if entry["sourceEntryPoint"] == "tile_large" else "1",
            "BN": "4" if entry["sourceEntryPoint"] == "tile_large" else "2",
            "T": "float",
        }
        for entry in directx_entries
    )
    assert all(
        set(entry["parameterSources"].values()) == {"source-instantiation"}
        for entry in directx_entries
    )
    assert any(
        "hostName" not in record
        for record in directx["templateMaterialization"]["specializations"]
    )
    directx_source = (repo / directx["path"]).read_text(encoding="utf-8")
    assert "[numthreads(32, 2, 1)]" in directx_source
    assert "[numthreads(32, 4, 2)]" in directx_source

    opengl = [
        artifact for artifact in payload["artifacts"] if artifact["target"] == "opengl"
    ]
    assert len(opengl) == 2
    assert {artifact["entryPoint"]["source"] for artifact in opengl} == {
        "tile_large",
        "tile_small",
    }
    assert all(
        artifact["entryPoint"]
        == {
            "source": artifact["execution"]["sourceEntryPoints"][0],
            "target": "main",
            "stage": "compute",
        }
        for artifact in opengl
    )
    assert all(len(artifact["execution"]["entryPoints"]) == 1 for artifact in opengl)
    assert all(
        artifact["execution"]["entryPoints"][0]["targetEntryPoint"] == "main"
        for artifact in opengl
    )
    assert all(
        {
            record["hostName"]
            for record in artifact["templateMaterialization"]["specializations"]
            if "hostName" in record
        }
        == {"tile_large", "tile_small"}
        for artifact in opengl
    )
    for artifact in opengl:
        generated = (repo / artifact["path"]).read_text(encoding="utf-8")
        assert "void main(" in generated
        size = artifact["execution"]["entryPoints"][0]["workgroupSize"]
        assert (
            "layout(local_size_x = "
            f"{size[0]}, local_size_y = {size[1]}, local_size_z = {size[2]}) in;"
        ) in generated

    report_path = repo / "report.json"
    report.write_json(report_path)
    assert project_api.validate_project_report(report_path)["success"] is True
    runtime = project_api.build_runtime_artifact_manifest(report_path)
    assert runtime["summary"]["artifactCount"] == 3
    assert (
        sum(artifact["dispatch"]["workgroupCount"] for artifact in runtime["artifacts"])
        == 4
    )
    assert all(artifact["execution"] is not None for artifact in runtime["artifacts"])


def test_project_workgroup_rule_report_rejects_missing_opengl_split(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _write_two_entry_fixture(repo)
    payload = project_api.translate_project(
        _rule_config(repo, targets=("opengl",)), format_output=False
    ).to_json()
    payload["artifacts"] = [
        artifact
        for artifact in payload["artifacts"]
        if artifact["entryPoint"]["source"] == "tile_small"
    ]
    _refresh_artifact_rollups(payload)
    report_path = repo / "missing-split.json"
    report_path.write_text(json.dumps(payload), encoding="utf-8")

    validation = project_api.validate_project_report(report_path)

    assert validation["success"] is False
    invalid = _diagnostic(validation, "project.validate.invalid-report")
    assert (
        "artifacts must include units[0].path shaders/tiled.metal target opengl"
        in invalid["message"]
    )


def test_project_workgroup_rule_reuses_one_materialized_ast_per_target(
    tmp_path, monkeypatch
):
    repo = tmp_path / "repo"
    repo.mkdir()
    _write_two_entry_fixture(repo)
    ast_calls = []
    original_ast = project_pipeline._crossgl_ast_for_project_target

    def record_ast(**kwargs):
        ast_calls.append((kwargs["target"], kwargs["source_is_materialized"]))
        return original_ast(**kwargs)

    def reject_second_materialization(**_kwargs):
        raise AssertionError("materialized project input was materialized again")

    monkeypatch.setattr(project_pipeline, "_crossgl_ast_for_project_target", record_ast)
    monkeypatch.setattr(
        project_pipeline,
        "materialize_metal_source_for_target",
        reject_second_materialization,
    )

    payload = project_api.translate_project(
        _rule_config(repo), format_output=False
    ).to_json()

    assert payload["summary"]["translatedCount"] == 3
    assert ast_calls == [("directx", True), ("opengl", True)]


def test_project_workgroup_rule_rejects_unsupported_target(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _write_two_entry_fixture(repo)

    report = project_api.translate_project(
        _rule_config(repo, targets=("directx", "vulkan")),
        format_output=False,
    )
    payload = report.to_json()

    assert payload["summary"]["translatedCount"] == 1
    assert payload["summary"]["failedCount"] == 1
    directx = next(
        artifact for artifact in payload["artifacts"] if artifact["target"] == "directx"
    )
    vulkan = next(
        artifact for artifact in payload["artifacts"] if artifact["target"] == "vulkan"
    )
    assert directx["status"] == "translated"
    assert vulkan["status"] == "failed"
    assert not (repo / vulkan["path"]).exists()

    diagnostic = _diagnostic(
        payload,
        "project.translate.workgroup-size-rule-unsupported-target",
    )
    assert diagnostic["checkKind"] == "execution-specialization"
    assert diagnostic["details"] == {
        "executionSpecialization": {
            "reason": "target-not-supported",
            "rule": {
                "components": ["32", "BN", "BM"],
                "path": 'project.workgroup_size_rules["shaders/tiled.metal"]',
                "sourcePattern": "shaders/tiled.metal",
                "supportedTargets": ["directx", "opengl"],
                "target": "vulkan",
            },
            "sourceEntryPoints": [],
        },
        "sourcePath": "shaders/tiled.metal",
        "targetArtifact": vulkan["path"],
    }

    report_path = repo / "report.json"
    report.write_json(report_path)
    validation = project_api.validate_project_report(report_path)
    assert validation["success"] is False
    assert (
        "project.validate.invalid-report" not in validation["diagnosticsByCode"]
    ), validation["diagnostics"]
    assert (
        validation["diagnosticsByCode"][
            "project.translate.workgroup-size-rule-unsupported-target"
        ]
        == 1
    )


def test_project_workgroup_rule_join_is_independent_of_record_order(
    tmp_path, monkeypatch
):
    repo = tmp_path / "repo"
    repo.mkdir()
    _write_two_entry_fixture(repo)
    baseline = project_api.translate_project(
        _rule_config(repo, targets=("directx",)),
        format_output=False,
    ).to_json()

    def reverse_records(metadata):
        metadata["specializations"] = list(reversed(metadata["specializations"]))

    _wrap_materialization(monkeypatch, reverse_records)
    reordered = project_api.translate_project(
        _rule_config(repo, targets=("directx",)),
        format_output=False,
    ).to_json()

    assert reordered["summary"]["translatedCount"] == 1
    assert (
        reordered["artifacts"][0]["execution"] == baseline["artifacts"][0]["execution"]
    )


def test_project_workgroup_rule_checks_overflow_without_changing_metal_folding():
    expression = "9223372036854775807 + 1 - 1"

    assert (
        project_pipeline._metal_evaluate_integer_constant_expression(expression, {})
        == "9223372036854775807"
    )
    with pytest.raises(
        project_pipeline._ProjectWorkgroupRuleExpressionError,
        match="overflowed signed 64-bit arithmetic",
    ):
        project_pipeline._evaluate_project_bounded_integer_expression(expression, {})


@pytest.mark.parametrize(
    ("expression", "reason"),
    [
        ("BN()", "call-unsupported"),
        ("BN.x", "member-access-unsupported"),
        ("UNKNOWN", "unknown-identifier"),
        ("1.5", "non-integral-expression"),
        ("9223372036854775807 + 1", "overflow"),
        ("BN / 0", "division-by-zero"),
        ("0", "non-positive-result"),
    ],
)
def test_project_workgroup_rule_rejects_invalid_expressions(
    tmp_path, expression, reason
):
    repo = tmp_path / "repo"
    repo.mkdir()
    _write_two_entry_fixture(repo)
    config = project_api.ProjectConfig(
        root=repo,
        targets=["directx"],
        output_dir="out",
        workgroup_size_rules={
            "shaders/tiled.metal": [expression, "1", "1"],
        },
    )

    payload = project_api.translate_project(config, format_output=False).to_json()

    assert payload["summary"]["failedCount"] == 1
    diagnostic = _diagnostic(payload, "project.translate.workgroup-size-rule-invalid")
    assert diagnostic["details"]["executionSpecialization"]["reason"] == reason
    assert not (repo / payload["artifacts"][0]["path"]).exists()


@pytest.mark.parametrize(
    ("case", "reason"),
    [
        ("missing", "entry-materialization-missing"),
        ("duplicate", "duplicate-entry-materialization"),
        ("helper-only", "helper-only-materialization"),
        ("conflicting", "conflicting-entry-materialization"),
        ("unmatched", "host-materialization-unmatched"),
    ],
)
def test_project_workgroup_rule_rejects_invalid_materialization_joins(
    tmp_path, monkeypatch, case, reason
):
    repo = tmp_path / "repo"
    repo.mkdir()
    _write_two_entry_fixture(repo)

    def mutate(metadata):
        records = metadata["specializations"]
        host_records = [record for record in records if "hostName" in record]
        helpers = [record for record in records if "hostName" not in record]
        by_host = {record["hostName"]: record for record in host_records}
        if case == "missing":
            metadata["specializations"] = [by_host["tile_small"], *helpers]
        elif case == "duplicate":
            metadata["specializations"] = [
                *records,
                copy.deepcopy(by_host["tile_small"]),
            ]
        elif case == "helper-only":
            helper_entry = copy.deepcopy(by_host["tile_small"])
            helper_entry.pop("hostName")
            metadata["specializations"] = [
                helper_entry,
                by_host["tile_large"],
                *helpers,
            ]
        elif case == "conflicting":
            shared = copy.deepcopy(by_host["tile_small"])
            shared["materializedName"] = "tile_large"
            metadata["specializations"] = [shared, *helpers]
        else:
            ghost = copy.deepcopy(by_host["tile_small"])
            ghost["hostName"] = "tile_ghost"
            ghost["materializedName"] = "tile_ghost"
            metadata["specializations"] = [*records, ghost]
        metadata["specializationCount"] = len(metadata["specializations"])

    _wrap_materialization(monkeypatch, mutate)
    payload = project_api.translate_project(
        _rule_config(repo, targets=("directx",)), format_output=False
    ).to_json()

    assert payload["summary"]["failedCount"] == 1
    diagnostic = _diagnostic(
        payload, "project.translate.workgroup-size-materialization-invalid"
    )
    assert diagnostic["details"]["executionSpecialization"]["reason"] == reason


def test_project_workgroup_rule_reports_target_limits(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _write_two_entry_fixture(repo)
    config = project_api.ProjectConfig(
        root=repo,
        targets=["directx"],
        output_dir="out",
        workgroup_size_rules={
            "shaders/tiled.metal": ["1024", "BN", "BM"],
        },
    )

    payload = project_api.translate_project(config, format_output=False).to_json()

    diagnostic = _diagnostic(payload, "project.translate.workgroup-size-invalid")
    assert diagnostic["details"]["executionSpecialization"]["reason"] == (
        "thread-count-limit-exceeded"
    )
    assert payload["summary"]["failedCount"] == 1


def test_project_workgroup_rule_report_rejects_rehashed_parameter_tampering(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _write_two_entry_fixture(repo)
    report = project_api.translate_project(
        _rule_config(repo, targets=("directx",)), format_output=False
    )
    payload = report.to_json()
    artifact = payload["artifacts"][0]
    execution = artifact["execution"]
    entry = execution["entryPoints"][0]
    entry["parameters"]["BN"] = "3"
    entry["workgroupSize"] = [32, 3, 2]
    entry["identity"] = project_pipeline._workgroup_rule_entry_identity(
        source=artifact["source"],
        source_hash=artifact["sourceHash"],
        target=artifact["target"],
        variant=None,
        entry=entry,
    )
    execution["identity"] = project_pipeline._workgroup_rule_execution_identity(
        source=artifact["source"],
        source_hash=artifact["sourceHash"],
        target=artifact["target"],
        variant=None,
        source_entry_points=execution["sourceEntryPoints"],
        entry_points=execution["entryPoints"],
        provenance=execution["provenance"],
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
        {"shaders/tiled.metal": ["32", "BN"]},
        {"shaders/tiled.metal": ["32", True, "BM"]},
        {"": ["32", "BN", "BM"]},
    ],
)
def test_project_config_rejects_malformed_workgroup_size_rules(tmp_path, rules):
    repo = tmp_path / "repo"
    repo.mkdir()

    with pytest.raises(ValueError, match="workgroup_size_rules"):
        project_api.ProjectConfig(root=repo, workgroup_size_rules=rules)

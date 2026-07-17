import copy
import json
import textwrap

import pytest

from crosstl.project import (
    DISPATCH_CONTRACT_KIND,
    DISPATCH_CONTRACT_SCHEMA_VERSION,
    DispatchArtifactPlan,
    DispatchArtifactPlanError,
    load_project_config,
    plan_dispatch_artifacts,
    scan_project,
    translate_project,
    validate_project_report,
)

SIMPLE_CROSSL = textwrap.dedent("""
    shader main {
        vertex {
            void main() {
            }
        }
    }
    """).strip()


def _dispatch_manifest(source="kernels/copy.cgl"):
    return {
        "kind": DISPATCH_CONTRACT_KIND,
        "schemaVersion": DISPATCH_CONTRACT_SCHEMA_VERSION,
        "inputs": [{"name": "elementCount", "role": "shape", "type": "integer"}],
        "workloads": [
            {"id": "count-64", "values": {"elementCount": 64}},
            {"id": "count-257", "values": {"elementCount": 257}},
        ],
        "capabilities": [],
        "devices": [{"id": "portable", "values": {}}],
        "contracts": [
            {
                "id": "copy-dispatch",
                "source": source,
                "entryPoint": "copy_float32",
                "branches": [
                    {
                        "id": "default",
                        "when": True,
                        "workgroupSize": [64, 1, 1],
                        "dispatch": {
                            "workgroupCount": [
                                {
                                    "op": "ceilDiv",
                                    "args": [{"input": "elementCount"}, 64],
                                },
                                1,
                                1,
                            ]
                        },
                    }
                ],
            }
        ],
    }


def _write_project(tmp_path, *, with_contract=True, contract_source=None):
    root = tmp_path / "repo"
    kernels = root / "kernels"
    kernels.mkdir(parents=True)
    for name in ("copy.cgl", "unrelated.cgl"):
        (kernels / name).write_text(SIMPLE_CROSSL + "\n", encoding="utf-8")

    project_lines = [
        "[project]",
        'include = ["kernels/*.cgl"]',
        'targets = ["cgl"]',
        'output_dir = "generated"',
    ]
    if with_contract:
        contracts = root / "contracts"
        contracts.mkdir()
        (contracts / "copy.json").write_text(
            json.dumps(
                _dispatch_manifest(contract_source or "kernels/copy.cgl"), indent=2
            )
            + "\n",
            encoding="utf-8",
        )
        project_lines.append('dispatch_contracts = ["contracts/copy.json"]')
    (root / "crosstl.toml").write_text(
        "\n".join(project_lines) + "\n", encoding="utf-8"
    )
    return root, load_project_config(root)


def _expected_plan(config, scan):
    return plan_dispatch_artifacts(
        config.dispatch_contract_evaluations,
        source_units=(unit.relative_path for unit in scan.units),
    )


def _write_report_payload(root, payload):
    report_path = root / "report.json"
    report_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return report_path


def _invalid_report_message(validation):
    diagnostic = next(
        item
        for item in validation["diagnostics"]
        if item["code"] == "project.validate.invalid-report"
    )
    return diagnostic["message"]


def test_scan_and_report_retain_the_source_scoped_dispatch_artifact_plan(tmp_path):
    _, config = _write_project(tmp_path)

    scan = scan_project(config)
    expected = _expected_plan(config, scan)
    report = scan.to_report()
    project = report.to_json()["project"]

    assert isinstance(scan.dispatch_artifact_plan, DispatchArtifactPlan)
    assert scan.dispatch_artifact_plan == expected
    assert report.dispatch_artifact_plan == expected
    assert project["dispatchArtifactCount"] == len(expected.artifacts) == 1
    assert project["dispatchArtifactPlan"] == expected.to_json()
    assert project["dispatchArtifactPlan"]["dispatchVariantCount"] == 2
    assert project["dispatchArtifactPlan"]["sourceUnitCount"] == 2


def test_translate_report_retains_dispatch_artifact_plan_metadata(tmp_path):
    _, config = _write_project(tmp_path)
    expected_scan = scan_project(config)
    expected = _expected_plan(config, expected_scan)

    report = translate_project(
        config,
        output_dir="alternate-generated",
        format_output=False,
    )
    project = report.to_json()["project"]

    assert report.dispatch_artifact_plan == expected
    assert report.config.dispatch_contracts == config.dispatch_contracts
    assert report.config.output_dir == "alternate-generated"
    assert project["dispatchArtifactCount"] == 1
    assert project["dispatchArtifactPlan"] == expected.to_json()


def test_project_without_dispatch_contracts_reports_a_valid_empty_plan(tmp_path):
    root, config = _write_project(tmp_path, with_contract=False)

    scan = scan_project(config)
    report = scan.to_report()
    payload = report.to_json()
    plan = report.dispatch_artifact_plan

    assert isinstance(plan, DispatchArtifactPlan)
    assert plan.source_units == ("kernels/copy.cgl", "kernels/unrelated.cgl")
    assert plan.artifacts == ()
    assert plan.dispatch_variants == ()
    assert payload["project"]["dispatchArtifactCount"] == 0
    assert payload["project"]["dispatchArtifactPlan"] == plan.to_json()
    assert payload["project"]["dispatchArtifactPlan"]["sourceUnitCount"] == 2
    assert payload["project"]["dispatchArtifactPlan"]["artifactCount"] == 0
    assert payload["project"]["dispatchArtifactPlan"]["dispatchVariantCount"] == 0

    report_path = _write_report_payload(root, payload)
    assert validate_project_report(report_path)["success"] is True


def test_project_report_validation_replays_dispatch_artifact_plan(tmp_path):
    root, config = _write_project(tmp_path)
    report = scan_project(config).to_report()
    report_path = _write_report_payload(root, report.to_json())

    validation = validate_project_report(report_path)

    assert validation["success"] is True
    assert "project.validate.invalid-report" not in {
        item["code"] for item in validation["diagnostics"]
    }


def _tamper_project_artifact_count(project):
    project["dispatchArtifactCount"] += 1


def _tamper_plan_artifact_count(project):
    project["dispatchArtifactPlan"]["artifactCount"] += 1


def _tamper_plan_source_unit_count(project):
    project["dispatchArtifactPlan"]["sourceUnitCount"] -= 1


def _tamper_artifact_record(project):
    project["dispatchArtifactPlan"]["artifacts"][0]["entryPoint"] = "other"


def _tamper_dispatch_variant_record(project):
    project["dispatchArtifactPlan"]["dispatchVariants"][0]["dispatch"] = {
        "workgroupCount": [99, 1, 1]
    }


@pytest.mark.parametrize(
    "mutation",
    (
        _tamper_project_artifact_count,
        _tamper_plan_artifact_count,
        _tamper_plan_source_unit_count,
        _tamper_artifact_record,
        _tamper_dispatch_variant_record,
    ),
)
def test_project_report_validation_rejects_tampered_dispatch_artifact_plan(
    tmp_path, mutation
):
    root, config = _write_project(tmp_path)
    payload = copy.deepcopy(scan_project(config).to_report().to_json())
    mutation(payload["project"])
    report_path = _write_report_payload(root, payload)

    validation = validate_project_report(report_path)

    assert validation["success"] is False
    assert "dispatchArtifact" in _invalid_report_message(validation)


def test_scan_rejects_dispatch_contract_source_absent_from_discovered_units(tmp_path):
    _, config = _write_project(
        tmp_path,
        contract_source="kernels/not-discovered.cgl",
    )

    with pytest.raises(DispatchArtifactPlanError) as exc_info:
        scan_project(config)

    assert exc_info.value.code == "dispatch-source-unmatched"
    assert exc_info.value.details == {
        "variantId": config.dispatch_contract_evaluations[0][0].variant_id,
        "source": "kernels/not-discovered.cgl",
        "availableSources": ["kernels/copy.cgl", "kernels/unrelated.cgl"],
    }

import copy
import json
import textwrap

import pytest

from crosstl.project import (
    DISPATCH_CONTRACT_EVALUATION_KIND,
    DISPATCH_CONTRACT_KIND,
    DISPATCH_CONTRACT_SCHEMA_VERSION,
    DispatchContractError,
    ProjectConfig,
    inspect_project_report,
    load_project_config,
    scan_project,
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


def _dispatch_manifest():
    return {
        "kind": DISPATCH_CONTRACT_KIND,
        "schemaVersion": DISPATCH_CONTRACT_SCHEMA_VERSION,
        "provenance": {"source": "host/dispatch.cpp"},
        "inputs": [{"name": "elementCount", "role": "shape", "type": "integer"}],
        "workloads": [
            {
                "id": "count-64",
                "values": {"elementCount": 64},
                "provenance": {"testCase": "bounded-import"},
            }
        ],
        "capabilities": [],
        "devices": [{"id": "portable", "values": {}}],
        "contracts": [
            {
                "id": "copy-dispatch",
                "source": "simple.cgl",
                "entryPoint": "main",
                "branches": [
                    {
                        "id": "default",
                        "when": True,
                        "workgroupSize": [64, 1, 1],
                        "specializationConstants": {"0": {"input": "elementCount"}},
                        "dispatch": {"workgroupCount": [1, 1, 1]},
                    }
                ],
            }
        ],
    }


def _write_project(tmp_path, *, config_contract_path="contracts/copy.json"):
    root = tmp_path / "repo"
    contract_path = root / "contracts" / "copy.json"
    contract_path.parent.mkdir(parents=True)
    (root / "simple.cgl").write_text(SIMPLE_CROSSL, encoding="utf-8")
    contract_path.write_text(
        json.dumps(_dispatch_manifest(), indent=2) + "\n", encoding="utf-8"
    )
    (root / "crosstl.toml").write_text(
        textwrap.dedent(f"""
            [project]
            include = ["simple.cgl"]
            targets = ["cgl"]
            output_dir = "generated"
            dispatch_contracts = ['{config_contract_path}']
            """).strip() + "\n",
        encoding="utf-8",
    )
    return root, contract_path


def _write_report(root, config):
    report = scan_project(config).to_report(targets=config.targets)
    report_path = root / "report.json"
    report.write_json(report_path)
    return report, report_path


def test_project_config_loads_and_evaluates_dispatch_contracts(tmp_path):
    root, contract_path = _write_project(tmp_path)

    config = load_project_config(root)

    assert config.dispatch_contracts == ["contracts/copy.json"]
    assert len(config.dispatch_contract_manifests) == 1
    assert config.dispatch_contract_manifests[0].source == str(contract_path.resolve())
    assert len(config.dispatch_contract_evaluations) == 1
    variant = config.dispatch_contract_evaluations[0][0]
    assert variant.entry_point == "main"
    assert variant.workgroup_size == (64, 1, 1)
    assert variant.specialization_constants == {"0": 64}


def test_project_config_normalizes_dispatch_contract_path_separators(tmp_path):
    root, _ = _write_project(tmp_path, config_contract_path=r"contracts\copy.json")

    config = load_project_config(root)

    assert config.dispatch_contracts == ["contracts/copy.json"]


def test_direct_project_config_rejects_duplicate_dispatch_contract_paths(tmp_path):
    root, _ = _write_project(tmp_path)

    with pytest.raises(
        ValueError,
        match="ProjectConfig.dispatch_contracts must not contain duplicate paths",
    ):
        ProjectConfig(
            root=root,
            dispatch_contracts=("contracts/copy.json", "contracts/copy.json"),
        )


def test_invalid_dispatch_contract_fails_during_project_config_load(tmp_path):
    root, contract_path = _write_project(tmp_path)
    payload = _dispatch_manifest()
    payload["schemaVersion"] = 999
    contract_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(DispatchContractError) as exc_info:
        load_project_config(root)

    assert exc_info.value.code == "project.dispatch-contract.schema-version-unsupported"


def test_project_report_embeds_replayable_dispatch_contract_metadata(tmp_path):
    root, contract_path = _write_project(tmp_path)
    config = load_project_config(root)

    report, report_path = _write_report(root, config)
    project = report.to_json()["project"]
    record = project["dispatchContracts"][0]

    assert project["dispatchContractFiles"] == ["contracts/copy.json"]
    assert project["dispatchContractCount"] == 1
    assert project["dispatchVariantCount"] == 1
    assert record["path"] == "contracts/copy.json"
    assert record["schemaVersion"] == DISPATCH_CONTRACT_SCHEMA_VERSION
    assert record["contentIdentity"] == (
        config.dispatch_contract_manifests[0].content_identity.to_json()
    )
    assert record["manifest"] == config.dispatch_contract_manifests[0].to_json()
    assert record["evaluation"]["kind"] == DISPATCH_CONTRACT_EVALUATION_KIND
    assert record["evaluation"]["manifestSource"] == str(contract_path.resolve())
    assert record["evaluation"]["variantCount"] == 1
    assert validate_project_report(report_path)["success"] is True

    inspection = inspect_project_report(report_path)
    assert inspection["report"]["project"]["dispatchContractFiles"] == [
        "contracts/copy.json"
    ]
    assert inspection["report"]["project"]["dispatchContractCount"] == 1
    assert inspection["report"]["project"]["dispatchVariantCount"] == 1


def test_project_report_validation_does_not_require_original_contract_file(tmp_path):
    root, contract_path = _write_project(tmp_path)
    _, report_path = _write_report(root, load_project_config(root))
    contract_path.unlink()

    assert validate_project_report(report_path)["success"] is True


@pytest.mark.parametrize(
    ("mutation", "expected_reason"),
    (
        (
            lambda record: record["contentIdentity"].update({"value": "0" * 64}),
            "contentIdentity must match the embedded manifest",
        ),
        (
            lambda record: record["manifest"]["workloads"][0]["values"].update(
                {"elementCount": 128}
            ),
            "contentIdentity must match the embedded manifest",
        ),
        (
            lambda record: record["evaluation"].update({"kind": "invalid"}),
            "evaluation.kind must be",
        ),
        (
            lambda record: record["evaluation"]["variants"][0].update(
                {"workgroupSize": [32, 1, 1]}
            ),
            "evaluation must match deterministic manifest replay",
        ),
    ),
)
def test_project_report_validation_rejects_tampered_dispatch_metadata(
    tmp_path, mutation, expected_reason
):
    root, _ = _write_project(tmp_path)
    report, report_path = _write_report(root, load_project_config(root))
    payload = copy.deepcopy(report.to_json())
    mutation(payload["project"]["dispatchContracts"][0])
    report_path.write_text(json.dumps(payload), encoding="utf-8")

    validation = validate_project_report(report_path)

    assert validation["success"] is False
    diagnostic = next(
        item
        for item in validation["diagnostics"]
        if item["code"] == "project.validate.invalid-report"
    )
    assert expected_reason in diagnostic["message"]


def test_project_report_validation_rejects_missing_dispatch_contract_file_record(
    tmp_path,
):
    root, _ = _write_project(tmp_path)
    report, report_path = _write_report(root, load_project_config(root))
    payload = copy.deepcopy(report.to_json())
    payload["project"]["dispatchContractFiles"] = []
    report_path.write_text(json.dumps(payload), encoding="utf-8")

    validation = validate_project_report(report_path)

    assert validation["success"] is False
    diagnostic = next(
        item
        for item in validation["diagnostics"]
        if item["code"] == "project.validate.invalid-report"
    )
    assert (
        "project.dispatchContracts paths must match project.dispatchContractFiles"
        in diagnostic["message"]
    )

import copy
import json
import textwrap

import pytest

from crosstl.project import (
    DISPATCH_CONTRACT_KIND,
    DISPATCH_CONTRACT_SCHEMA_VERSION,
    DispatchArtifactPlanError,
    load_project_config,
    translate_project,
    validate_project_report,
)


PLANNED_COMPUTE_SOURCE = textwrap.dedent("""
    shader PlannedCompute {
        bool enabled @function_constant(7);

        compute {
            void main() {
                uint width = gl_WorkGroupSize.x;
                bool selected = enabled && width > 0u;
            }
        }
    }
    """).strip()

UNRELATED_COMPUTE_SOURCE = textwrap.dedent("""
    shader UnrelatedCompute {
        compute {
            void main() {
                uint value = 1u;
            }
        }
    }
    """).strip()


def _dispatch_manifest(*, subgroup_width=None):
    workgroup_size = [32, 1, 1] if subgroup_width is not None else [8, 2, 1]
    branch = {
        "id": "default",
        "when": True,
        "workgroupSize": workgroup_size,
        "specializationConstants": {"7": True},
        "dispatch": {
            "workgroupCount": [
                {
                    "op": "ceilDiv",
                    "args": [{"input": "elementCount"}, workgroup_size[0]],
                },
                1,
                1,
            ]
        },
    }
    if subgroup_width is not None:
        branch["subgroupWidth"] = subgroup_width
    return {
        "kind": DISPATCH_CONTRACT_KIND,
        "schemaVersion": DISPATCH_CONTRACT_SCHEMA_VERSION,
        "inputs": [{"name": "elementCount", "role": "shape", "type": "integer"}],
        "workloads": [
            {"id": "count-64", "values": {"elementCount": 64}},
            {"id": "count-129", "values": {"elementCount": 129}},
        ],
        "capabilities": [],
        "devices": [{"id": "portable", "values": {}}],
        "contracts": [
            {
                "id": "planned-compute",
                "source": "kernels/planned.cgl",
                "entryPoint": "main",
                "branches": [branch],
            }
        ],
    }


def _write_project(tmp_path, *, workgroup_size=None, subgroup_width=None):
    root = tmp_path / "repo"
    kernels = root / "kernels"
    contracts = root / "contracts"
    kernels.mkdir(parents=True)
    contracts.mkdir()
    (kernels / "planned.cgl").write_text(
        PLANNED_COMPUTE_SOURCE + "\n", encoding="utf-8"
    )
    (kernels / "unrelated.cgl").write_text(
        UNRELATED_COMPUTE_SOURCE + "\n", encoding="utf-8"
    )
    (contracts / "planned.json").write_text(
        json.dumps(
            _dispatch_manifest(subgroup_width=subgroup_width), indent=2
        )
        + "\n",
        encoding="utf-8",
    )

    project_lines = [
        "[project]",
        'include = ["kernels/*.cgl"]',
        'targets = ["directx", "opengl"]',
        'output_dir = "generated"',
        'dispatch_contracts = ["contracts/planned.json"]',
    ]
    if workgroup_size is not None:
        project_lines.append(
            "workgroup_size = [{}]".format(
                ", ".join(str(component) for component in workgroup_size)
            )
        )
    (root / "crosstl.toml").write_text(
        "\n".join(project_lines) + "\n", encoding="utf-8"
    )
    return root, load_project_config(root)


def _artifacts_for(payload, source, *, target=None):
    return [
        artifact
        for artifact in payload["artifacts"]
        if artifact["source"] == source
        and (target is None or artifact["target"] == target)
    ]


def _write_report(root, payload, name="report.json"):
    report_path = root / name
    report_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return report_path


def test_directx_and_opengl_generate_only_source_scoped_dispatch_artifacts(tmp_path):
    root, config = _write_project(tmp_path)

    report = translate_project(config, format_output=False)
    payload = report.to_json()

    plan = payload["project"]["dispatchArtifactPlan"]
    assert plan["artifactCount"] == 1
    assert plan["dispatchVariantCount"] == 2
    planned_job = plan["artifacts"][0]
    planned_variant = planned_job["variant"]

    assert payload["summary"]["translatedCount"] == 4
    assert payload["summary"]["failedCount"] == 0
    assert len(payload["artifacts"]) == 4
    for target in ("directx", "opengl"):
        planned_artifacts = _artifacts_for(
            payload, "kernels/planned.cgl", target=target
        )
        unrelated_artifacts = _artifacts_for(
            payload, "kernels/unrelated.cgl", target=target
        )

        assert len(planned_artifacts) == 1
        planned_artifact = planned_artifacts[0]
        assert planned_artifact["variant"] == planned_variant
        assert planned_artifact["dispatchArtifact"] == planned_job
        assert planned_artifact["execution"]["workgroupSize"] == [8, 2, 1]
        assert planned_artifact["execution"]["provenance"] == {
            "kind": "host-dispatch-contract",
            "path": "project.dispatchArtifactPlan.artifacts[0]",
            "variant": planned_variant,
            "artifactId": planned_job["artifactId"],
        }

        constants = {
            record["id"]: record
            for record in planned_artifact["specializationConstants"]
        }
        assert constants[7]["concreteValue"] is True
        assert constants[7]["valueProvenance"] == planned_artifact["execution"][
            "provenance"
        ]

        assert len(unrelated_artifacts) == 1
        assert "variant" not in unrelated_artifacts[0]
        assert "dispatchArtifact" not in unrelated_artifacts[0]

    directx_artifact = _artifacts_for(
        payload, "kernels/planned.cgl", target="directx"
    )[0]
    directx_source = (root / directx_artifact["path"]).read_text(encoding="utf-8")
    assert "[numthreads(8, 2, 1)]" in directx_source
    assert "static const bool enabled = true;" in directx_source

    opengl_artifact = _artifacts_for(
        payload, "kernels/planned.cgl", target="opengl"
    )[0]
    opengl_source = (root / opengl_artifact["path"]).read_text(encoding="utf-8")
    assert (
        "layout(local_size_x = 8, local_size_y = 2, local_size_z = 1) in;"
        in opengl_source
    )

    matrix = payload["artifactMatrix"]
    assert matrix["variantMode"] == "source-scoped"
    assert matrix["variantCount"] == 1
    assert matrix["expectedArtifactCount"] == 4
    assert matrix["emittedArtifactCount"] == 4
    assert matrix["missingArtifactCount"] == 0
    assert matrix["extraArtifactCount"] == 0
    assert matrix["identityCoverageAvailable"] is True
    assert matrix["complete"] is True

    report_path = root / "generated" / "report.json"
    report.write_json(report_path)
    assert validate_project_report(report_path)["success"] is True


def test_report_validation_rejects_tampered_dispatch_artifact_metadata(tmp_path):
    root, config = _write_project(tmp_path)
    payload = copy.deepcopy(
        translate_project(config, format_output=False).to_json()
    )
    planned_artifact = _artifacts_for(
        payload, "kernels/planned.cgl", target="directx"
    )[0]
    planned_artifact["dispatchArtifact"]["workgroupSize"] = [4, 2, 1]
    report_path = _write_report(root, payload, "tampered-report.json")

    validation = validate_project_report(report_path)

    assert validation["success"] is False
    invalid_report = next(
        diagnostic
        for diagnostic in validation["diagnostics"]
        if diagnostic["code"] == "project.validate.invalid-report"
    )
    assert "dispatchArtifact" in invalid_report["message"]


@pytest.mark.parametrize(
    ("mutation", "expected_message"),
    (
        ("specialization", "specializationConstants"),
        ("execution", "execution"),
    ),
)
def test_report_validation_rejects_dispatch_emission_metadata_tampering(
    tmp_path, mutation, expected_message
):
    root, config = _write_project(tmp_path)
    payload = copy.deepcopy(
        translate_project(
            config,
            targets=["directx"],
            format_output=False,
        ).to_json()
    )
    artifact = _artifacts_for(
        payload, "kernels/planned.cgl", target="directx"
    )[0]
    assert "dispatchArtifact" in artifact
    if mutation == "specialization":
        constant = next(
            record
            for record in artifact["specializationConstants"]
            if record["id"] == 7
        )
        constant["concreteValue"] = False
    else:
        artifact.pop("execution")
    report_path = _write_report(root, payload, f"tampered-{mutation}-report.json")

    validation = validate_project_report(report_path)

    assert validation["success"] is False
    invalid_report = next(
        diagnostic
        for diagnostic in validation["diagnostics"]
        if diagnostic["code"] == "project.validate.invalid-report"
    )
    assert expected_message in invalid_report["message"]


def test_non_specialization_target_keeps_regular_per_source_artifacts(tmp_path):
    root, config = _write_project(tmp_path)

    report = translate_project(
        config,
        targets=["cgl"],
        output_dir="generated-cgl",
        format_output=False,
    )
    payload = report.to_json()

    assert payload["project"]["dispatchArtifactCount"] == 1
    assert payload["summary"]["translatedCount"] == 2
    assert payload["summary"]["failedCount"] == 0
    assert len(payload["artifacts"]) == 2
    assert {artifact["source"] for artifact in payload["artifacts"]} == {
        "kernels/planned.cgl",
        "kernels/unrelated.cgl",
    }
    assert all("variant" not in artifact for artifact in payload["artifacts"])
    assert all("dispatchArtifact" not in artifact for artifact in payload["artifacts"])
    assert payload["artifactMatrix"]["variantMode"] == "none"
    assert payload["artifactMatrix"]["expectedArtifactCount"] == 2
    assert payload["artifactMatrix"]["complete"] is True

    report_path = root / "generated-cgl" / "report.json"
    report.write_json(report_path)
    assert validate_project_report(report_path)["success"] is True


def test_dispatch_workgroup_conflict_fails_before_artifact_generation(tmp_path):
    root, config = _write_project(tmp_path, workgroup_size=(4, 1, 1))

    with pytest.raises(DispatchArtifactPlanError) as exc_info:
        translate_project(config, format_output=False)

    assert exc_info.value.code == "dispatch-workgroup-config-conflict"
    assert exc_info.value.details == {"source": "kernels/planned.cgl"}
    assert not (root / "generated").exists()


def test_directx_enforces_exact_dispatch_subgroup_width(tmp_path):
    root, config = _write_project(tmp_path, subgroup_width=32)

    report = translate_project(
        config,
        targets=["directx"],
        output_dir="generated-directx",
        format_output=False,
    )
    payload = report.to_json()

    assert payload["summary"]["translatedCount"] == 2
    assert payload["summary"]["failedCount"] == 0
    planned_job = payload["project"]["dispatchArtifactPlan"]["artifacts"][0]
    assert planned_job["subgroupWidth"] == 32
    artifact = _artifacts_for(
        payload, "kernels/planned.cgl", target="directx"
    )[0]
    assert artifact["status"] == "translated"
    assert artifact["dispatchArtifact"] == planned_job

    execution = artifact["execution"]
    assert execution["sourceEntryPoints"] == ["main"]
    assert execution["subgroupWidthProvenance"] == execution["provenance"]
    assert execution["entryPoints"][0]["workgroupSize"] == [32, 1, 1]
    assert execution["entryPoints"][0]["subgroupWidth"] == 32
    assert execution["subgroupWidthEnforcement"] == {
        "mechanism": "hlsl-wave-size-attribute",
        "minimumShaderModel": "6.6",
        "entryProfiles": [
            {
                "entryPoint": execution["entryPoints"][0]["targetEntryPoint"],
                "profile": "cs_6_6",
            }
        ],
    }

    generated_path = root / artifact["path"]
    generated = generated_path.read_text(encoding="utf-8")
    assert "[WaveSize(32)]" in generated
    assert "[numthreads(32, 1, 1)]" in generated

    report_path = root / "directx-report.json"
    report.write_json(report_path)
    assert validate_project_report(report_path)["success"] is True

    tampered = copy.deepcopy(payload)
    tampered_artifact = _artifacts_for(
        tampered, "kernels/planned.cgl", target="directx"
    )[0]
    tampered_artifact["execution"]["subgroupWidthEnforcement"][
        "minimumShaderModel"
    ] = "6.5"
    tampered_path = _write_report(root, tampered, "tampered-subgroup-report.json")
    validation = validate_project_report(tampered_path)
    assert validation["success"] is False
    invalid_report = next(
        diagnostic
        for diagnostic in validation["diagnostics"]
        if diagnostic["code"] == "project.validate.invalid-report"
    )
    assert "subgroupWidthEnforcement" in invalid_report["message"]


def test_opengl_fails_closed_for_exact_dispatch_subgroup_width(tmp_path):
    root, config = _write_project(tmp_path, subgroup_width=32)

    report = translate_project(
        config,
        targets=["opengl"],
        output_dir="generated-opengl",
        format_output=False,
    )
    payload = report.to_json()

    assert payload["summary"]["translatedCount"] == 1
    assert payload["summary"]["failedCount"] == 1
    planned_job = payload["project"]["dispatchArtifactPlan"]["artifacts"][0]
    assert planned_job["subgroupWidth"] == 32
    artifact = _artifacts_for(
        payload, "kernels/planned.cgl", target="opengl"
    )[0]
    assert artifact["status"] == "failed"
    assert artifact["dispatchArtifact"] == planned_job
    assert not (root / artifact["path"]).exists()

    diagnostic = next(
        item
        for item in payload["diagnostics"]
        if item["code"] == "project.translate.subgroup-width-invalid"
    )
    assert diagnostic["message"] == (
        "The target cannot enforce an exact host dispatch subgroup width."
    )
    assert diagnostic["target"] == "opengl"
    assert diagnostic["checkKind"] == "execution-specialization"
    assert diagnostic["missingCapabilities"] == [
        "execution.subgroup-width-specialization"
    ]
    assert diagnostic["details"]["executionSpecialization"] == {
        "reason": "target-not-enforceable",
        "sourceEntryPoints": ["main"],
        "subgroupWidth": 32,
    }

    report_path = root / "opengl-report.json"
    report.write_json(report_path)
    validation = validate_project_report(report_path)
    assert "project.validate.invalid-report" not in {
        item["code"] for item in validation["diagnostics"]
    }

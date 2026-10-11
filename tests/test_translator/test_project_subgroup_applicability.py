import copy
import json
import os
import sys
from dataclasses import replace
from pathlib import Path

import pytest

import crosstl.project as project
import crosstl.project.pipeline as pipeline

SOURCE = """#include <metal_stdlib>
using namespace metal;
kernel void collective(const device float* input [[buffer(0)]],
                       device float* output [[buffer(1)]],
                       uint index [[thread_position_in_grid]]) {
    output[index + 1u] = simd_sum(input[index]);
}
kernel void pointwise(const device float* input [[buffer(0)]],
                      device float* output [[buffer(1)]],
                      uint index [[thread_position_in_grid]]) {
    output[index + 1u] = input[index] * 2.0f + 1.0f;
}
"""


def _config(root, target, *, discovered=False, mode="when-used", width=32):
    root.mkdir()
    (root / "mixed.metal").write_text(SOURCE, encoding="utf-8")
    options = {"software_subgroup_width": width}
    if mode is not None:
        options["software_subgroup_applicability"] = mode
    if target == "directx":
        options["relative_wave_shuffle_out_of_range"] = "self"
    return project.ProjectConfig(
        root=root,
        targets=(target,),
        include_patterns=("*.metal",),
        output_dir="out",
        entry_points={} if discovered else {"mixed.metal": ("collective", "pointwise")},
        translate_discovered_entry_points=("mixed.metal",) if discovered else (),
        workgroup_size=(32, 1, 1),
        source_options=(
            {"metal": {"target_options": {target: options}}}
            if target != "metal"
            else {}
        ),
    )


def _check(payload, target, count=2):
    assert payload["summary"]["failedCount"] == 0, payload["diagnostics"]
    assert payload["summary"]["translatedCount"] == count
    for artifact in payload["artifacts"]:
        if target == "metal":
            assert "softwareSubgroupPolicy" not in artifact["provenance"]
            continue
        collective = artifact["entryPoint"]["source"] == "collective"
        assert artifact["provenance"]["softwareSubgroupPolicy"] == {
            "applicability": "when-used",
            "requestedWidth": 32,
            "effectiveWidth": 32 if collective else None,
            "requirements": ["operation:WaveActiveSum"] if collective else [],
        }
        if not collective:
            text = (Path(payload["project"]["root"]) / artifact["path"]).read_text()
            assert not any(
                token in text
                for token in (
                    "groupshared",
                    "shared ",
                    "GroupMemoryBarrier",
                    "barrier()",
                    "gl_Subgroup",
                    "WaveGet",
                    "crossglSoftwareSubgroup",
                )
            )


@pytest.mark.parametrize("target", ("directx", "opengl"))
@pytest.mark.parametrize("discovered", (False, True))
@pytest.mark.parametrize("workers", (1, 2))
def test_mixed_entries_resolve_policy_after_selection(
    tmp_path, target, discovered, workers
):
    config = _config(tmp_path / "repo", target, discovered=discovered)
    report = project.translate_project(config, format_output=False, max_workers=workers)
    _check(report.to_json(), target)
    path = tmp_path / "report.json"
    report.write_json(path)
    assert project.validate_project_report(path)["success"]
    manifest = project.build_runtime_artifact_manifest(path)
    assert manifest["success"], manifest
    assert len(manifest["artifacts"]) == 2
    for artifact in manifest["artifacts"]:
        source = next(
            item
            for item in report.to_json()["artifacts"]
            if item["entryPoint"] == artifact["entryPoint"]
        )
        assert (
            artifact["provenance"]["softwareSubgroupPolicy"]
            == source["provenance"]["softwareSubgroupPolicy"]
        )


@pytest.mark.parametrize("target", ("directx", "opengl"))
@pytest.mark.parametrize("mode", (None, "required"))
def test_explicit_required_policy_keeps_empty_operation_error(tmp_path, target, mode):
    config = _config(tmp_path / "repo", target, mode=mode)
    data = project.translate_project(config, format_output=False).to_json()
    assert data["summary"]["translatedCount"] == 1
    assert data["summary"]["failedCount"] == 1
    assert any(
        d.get("details", {}).get("executionSpecialization", {}).get("reason")
        == "operation-set-empty"
        for d in data["diagnostics"]
    )


@pytest.mark.parametrize("target", ("directx", "opengl"))
@pytest.mark.parametrize(
    "mode,width,reason",
    (
        ("typo", 32, "configured-applicability-invalid"),
        (True, 32, "configured-applicability-invalid"),
        ("when-used", 16, "configured-width-invalid"),
        ("when-used", True, "configured-width-invalid"),
        ("when-used", "32", "configured-width-invalid"),
    ),
)
def test_ordinary_entry_does_not_hide_invalid_policy(
    tmp_path, target, mode, width, reason
):
    config = _config(tmp_path / "repo", target, mode=mode, width=width)
    config = replace(config, entry_points={"mixed.metal": ("pointwise",)})
    data = project.translate_project(config, format_output=False).to_json()
    assert data["summary"]["failedCount"] == 1
    assert any(
        d.get("details", {}).get("executionSpecialization", {}).get("reason") == reason
        for d in data["diagnostics"]
    ), data["diagnostics"]


@pytest.mark.parametrize("target", ("directx", "opengl"))
def test_policy_provenance_survives_variants_and_checkpoint(
    tmp_path, target, monkeypatch
):
    config = _config(tmp_path / "repo", target)
    config = replace(config, variants={"first": {"MODE": "1"}, "second": {"MODE": "2"}})
    checkpoint = tmp_path / "checkpoint.json"
    reports = []

    def interrupt(self, payload):
        reports.append(payload)
        raise KeyboardInterrupt("before publication")

    with monkeypatch.context() as patch:
        patch.setattr(
            pipeline.ProjectTranslationCheckpointRecorder, "write_complete", interrupt
        )
        with pytest.raises(KeyboardInterrupt, match="before publication"):
            project.translate_project(
                config, format_output=False, max_workers=2, checkpoint_path=checkpoint
            )
    assert json.loads(checkpoint.read_text())["plan"]["completedCount"] == 4
    _check(reports[0], target, count=4)
    changed_options = copy.deepcopy(config.source_options)
    changed_options["metal"]["target_options"][target][
        "software_subgroup_applicability"
    ] = "required"
    with pytest.raises(pipeline.ProjectTranslationCheckpointError) as caught:
        project.translate_project(
            replace(config, source_options=changed_options),
            format_output=False,
            max_workers=2,
            checkpoint_path=checkpoint,
            resume=True,
        )
    assert caught.value.reason == "project-identity-mismatch"

    def unexpected(**kwargs):
        pytest.fail("Completed artifacts should not be regenerated")

    monkeypatch.setattr(
        pipeline, "_generate_project_target_from_crossgl_ast", unexpected
    )
    restored = project.translate_project(
        config,
        format_output=False,
        max_workers=2,
        checkpoint_path=checkpoint,
        resume=True,
    )
    assert restored.to_json()["artifacts"] == reports[0]["artifacts"]


@pytest.mark.parametrize("fault", ("remove", "width", "mode", "requirements", "extra"))
def test_report_rejects_inconsistent_subgroup_policy(tmp_path, fault):
    config = _config(tmp_path / "repo", "opengl")
    data = project.translate_project(config, format_output=False).to_json()
    _check(data, "opengl")
    provenance = data["artifacts"][0]["provenance"]
    policy = provenance["softwareSubgroupPolicy"]
    if fault == "remove":
        del provenance["softwareSubgroupPolicy"]
    elif fault == "width":
        policy["effectiveWidth"] = None
    elif fault == "mode":
        policy["applicability"] = "required"
    elif fault == "requirements":
        policy["requirements"] = []
    else:
        policy["unrecognized"] = True
    path = tmp_path / "report.json"
    path.write_text(json.dumps(data))
    validated = project.validate_project_report(path)
    assert not validated["success"]
    assert "softwareSubgroupPolicy" in json.dumps(validated["diagnostics"])


@pytest.mark.parametrize("target", ("directx", "opengl"))
@pytest.mark.parametrize(
    "body",
    (
        "if (index % 2u == 0u) { output[index] = simd_sum(input[index]); }",
        "output[index] = float(simd_ballot(input[index] > 0.0f));",
    ),
)
def test_when_used_keeps_collective_safety_diagnostics(tmp_path, target, body):
    config = _config(tmp_path / "repo", target)
    source = SOURCE.replace("output[index + 1u] = simd_sum(input[index]);", body)
    (config.root / "mixed.metal").write_text(source)
    config = replace(config, entry_points={"mixed.metal": ("collective",)})
    data = project.translate_project(config, format_output=False).to_json()
    assert data["summary"]["failedCount"] == 1, data
    assert data["summary"]["translatedCount"] == 0


def test_mixed_entry_policy_executes(tmp_path):
    if os.environ.get("CROSTL_REQUIRE_SUBGROUP_APPLICABILITY") != "1":
        pytest.skip(
            "set CROSTL_REQUIRE_SUBGROUP_APPLICABILITY=1 for native mixed-entry checks"
        )
    from tests.runtime_helpers import _validate
    from tests.test_translator.test_native_loader_dispatch_integration import _executor

    target = {"darwin": "metal", "linux": "opengl", "win32": "directx"}[sys.platform]
    config = _config(tmp_path / "repo", target)
    report = project.translate_project(config, format_output=False)
    _check(report.to_json(), target)
    report_path = tmp_path / "report.json"
    report.write_json(report_path)
    manifest = project.build_runtime_artifact_manifest(report_path)
    assert manifest["success"], manifest
    manifest_path = tmp_path / "artifacts.json"
    manifest_path.write_text(json.dumps(manifest))
    package = tmp_path / "package"
    assert project.build_runtime_package(manifest_path, package)["success"]
    loader = project.build_runtime_loader_manifest(package / "runtime-package.json")
    assert loader["success"] and len(loader["loadUnits"]) == 2
    executor = _executor(target)
    try:
        for unit in loader["loadUnits"]:
            descriptor = project.build_native_loader_abi_descriptor(
                loader, load_unit_id=unit["id"]
            )
            entry = unit["entryPoint"]["source"]
            work = tmp_path / entry
            work.mkdir()
            _validate(package / descriptor["artifact"]["packagePath"], work, target)
            count, guard = 96, 1234567.0
            inputs = [float((i * 7) % 31 - 15) for i in range(count)]
            expected = [
                (
                    sum(inputs[(i // 32) * 32 : (i // 32 + 1) * 32])
                    if entry == "collective"
                    else inputs[i] * 2 + 1
                )
                for i in range(count)
            ]
            bindings, outputs = {}, {}
            for binding in descriptor["bindings"]:
                assert binding["scalarLayout"]["elementType"] == "float32"
                writable = binding["access"] != "read"
                values = [guard] * (count + 9) if writable else inputs
                bindings[binding["name"]] = {
                    "dtype": "float32",
                    "shape": [len(values)],
                    "values": values,
                }
                if writable:
                    outputs[binding["name"]] = {
                        **bindings[binding["name"]],
                        "values": [guard] + expected + [guard] * 8,
                    }
            assert len(bindings) == 2 and len(outputs) == 1
            request = project.build_native_loader_dispatch_request(
                descriptor,
                package,
                bindings,
                outputs,
                {"workgroupCount": [3, 1, 1], "workgroupSize": [32, 1, 1]},
                expected_target=target,
            )
            assert executor.is_available(request).available
            result = executor.run(request)
            (work / "evidence.json").write_text(
                json.dumps(
                    {
                        "descriptor": descriptor,
                        "expected": outputs,
                        "status": result.status,
                        "outputs": result.outputs,
                        "details": result.details,
                    },
                    indent=2,
                )
            )
            assert result.status == "ok", result.details
            for name, output in outputs.items():
                assert result.outputs[name]["values"] == output["values"]
    finally:
        close = getattr(executor.runtime_adapter.runtime, "close", None)
        if close:
            close()


@pytest.mark.parametrize("target", ("directx", "opengl"))
@pytest.mark.parametrize(
    "semantic",
    (
        "thread_index_in_simdgroup",
        "threads_per_simdgroup",
        "simdgroup_index_in_threadgroup",
        "simdgroups_per_threadgroup",
    ),
)
def test_subgroup_system_inputs_do_not_silently_use_hardware(
    tmp_path, target, semantic
):
    config = _config(tmp_path / "repo", target)
    source = SOURCE.replace(
        "uint index [[thread_position_in_grid]]) {",
        f"uint index [[thread_position_in_grid]], uint subgroup [[{semantic}]]) {{",
    ).replace("input[index] * 2.0f + 1.0f", "float(subgroup)")
    (config.root / "mixed.metal").write_text(source)
    config = replace(config, entry_points={"mixed.metal": ("pointwise",)})
    data = project.translate_project(config, format_output=False).to_json()
    assert data["summary"]["failedCount"] == 1, data
    assert any(
        d.get("details", {}).get("executionSpecialization", {}).get("reason")
        == "operation-set-empty"
        for d in data["diagnostics"]
    ), data["diagnostics"]


@pytest.mark.parametrize("target", ("directx", "opengl"))
def test_source_scoped_policy_keeps_mixed_files_separate(tmp_path, target):
    config = _config(tmp_path / "repo", target)
    (config.root / "ordinary.metal").write_text(
        SOURCE.split("kernel void pointwise")[0].replace(
            "simd_sum(input[index])", "input[index]"
        )
    )
    settings = config.source_options["metal"]["target_options"][target]
    config = replace(
        config,
        entry_points={
            "mixed.metal": ("collective", "pointwise"),
            "ordinary.metal": "collective",
        },
        source_options={
            "metal": {
                "target_options": {
                    target: {"source_patterns": {"mixed.metal": settings}}
                }
            }
        },
    )
    report = project.translate_project(config, format_output=False, max_workers=2)
    data = report.to_json()
    assert data["summary"]["translatedCount"] == 3, data["diagnostics"]
    ordinary = next(a for a in data["artifacts"] if a["source"] == "ordinary.metal")
    assert "softwareSubgroupPolicy" not in ordinary["provenance"]
    path = tmp_path / "report.json"
    report.write_json(path)
    assert project.validate_project_report(path)["success"]


@pytest.mark.parametrize("target", ("directx", "opengl"))
def test_applicability_requires_width(tmp_path, target):
    config = _config(tmp_path / "repo", target)
    config.source_options["metal"]["target_options"][target].pop(
        "software_subgroup_width"
    )
    data = project.translate_project(config, format_output=False).to_json()
    assert data["summary"]["failedCount"] == 2
    assert all(
        d.get("details", {}).get("executionSpecialization", {}).get("reason")
        == "configured-width-missing"
        for d in data["diagnostics"]
        if d["severity"] == "error"
    )


def test_ci_requires_mixed_entry_execution():
    from tools import ci_coverage

    workflow = (
        Path(__file__).resolve().parents[2]
        / ".github/workflows/demo-project-testing.yml"
    ).read_text()
    step = ci_coverage.workflow_job_step_section(
        workflow, "portable-host", "Validate collective helper arguments"
    )
    selector = "tests/test_translator/test_project_subgroup_applicability.py"
    assert step.count(selector) == 1
    assert 'CROSTL_REQUIRE_SUBGROUP_APPLICABILITY: "1"' in step
    assert "--timeout-seconds 360" in step
    assert "continue-on-error" not in step and "if:" not in step


@pytest.mark.parametrize("target", ("directx", "opengl"))
def test_applicability_is_not_specific_to_metal_sources(tmp_path, target):
    for name, expression in (
        ("collective", "WaveActiveSum(invocation)"),
        ("pointwise", "invocation + 1u"),
    ):
        (tmp_path / f"{name}.cgl").write_text(f"""shader Mixed {{
            RWStructuredBuffer<uint> results @register(u0);
            compute {{
                @numthreads(32, 1, 1)
                void main(uint invocation @gl_LocalInvocationIndex) {{
                    results[invocation] = {expression};
                }}
            }}
        }}""")
    settings = {
        "software_subgroup_width": 32,
        "software_subgroup_applicability": "when-used",
    }
    if target == "directx":
        settings["relative_wave_shuffle_out_of_range"] = "self"
    config = project.ProjectConfig(
        root=tmp_path,
        targets=(target,),
        include_patterns=("*.cgl",),
        source_options={"crossgl": {"target_options": {target: settings}}},
    )
    report = project.translate_project(config, format_output=False)
    data = report.to_json()
    assert data["summary"]["translatedCount"] == 2, data["diagnostics"]
    assert data["summary"]["failedCount"] == 0
    for artifact in data["artifacts"]:
        width = 32 if artifact["source"] == "collective.cgl" else None
        assert (
            artifact["provenance"]["softwareSubgroupPolicy"]["effectiveWidth"] == width
        )
    path = tmp_path / "report.json"
    report.write_json(path)
    assert project.validate_project_report(path)["success"]

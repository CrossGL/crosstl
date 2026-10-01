"""Project packages retain exact-grid specialization through native loading."""

import copy
import hashlib
import json
import os
import sys
from dataclasses import replace

import pytest

from crosstl.project import (
    ProjectConfig,
    build_native_loader_abi_descriptor,
    build_native_loader_dispatch_request,
    build_runtime_artifact_manifest,
    build_runtime_loader_manifest,
    build_runtime_package,
    translate_project,
)
from crosstl.project.native_loader_abi import NativeLoaderABIError
from crosstl.project.native_loader_dispatch import NativeLoaderDispatchError
from crosstl.project.runtime_verification import (
    RuntimeAllocationView,
    RuntimeExecutionState,
)
from crosstl.translator.dispatch_regions import DispatchRegion, plan_dispatch_regions
from tests.test_translator.test_dispatch_regions import REQUIRE_ENV, _case
from tests.test_translator.test_exact_thread_grid_runtime import GRIDS, SOURCE
from tests.test_translator.test_metal_builtin_ownership import _compile
from tests.test_translator.test_native_loader_dispatch_integration import _executor


def _config(root, target, region):
    root.mkdir(parents=True, exist_ok=True)
    (root / "kernel.metal").write_text(SOURCE, encoding="utf-8")
    return ProjectConfig(
        root=root,
        include_patterns=("kernel.metal",),
        targets=(target,),
        output_dir="out",
        workgroup_size=region.workgroup_size,
        source_options={
            "metal": {
                "target_options": {
                    target: {
                        "dispatch_region": region.to_json(),
                        "software_subgroup_width": 32,
                        **(
                            {"relative_wave_shuffle_out_of_range": "self"}
                            if target == "directx"
                            else {}
                        ),
                    }
                }
            }
        },
    )


def _package(root, target, region):
    report = translate_project(_config(root, target, region), format_output=False)
    payload = report.to_json()
    assert payload["summary"]["failedCount"] == 0, json.dumps(
        payload["diagnostics"], indent=2
    )
    assert len(payload["artifacts"]) == 1
    assert set(payload["artifacts"][0]["provenance"]) == {
        "pipeline",
        "intermediate",
        "dispatchRegion",
    }
    assert payload["artifacts"][0]["provenance"]["dispatchRegion"] == region.to_json()
    report.write_json(root / "report.json")
    manifest = build_runtime_artifact_manifest(root / "report.json")
    assert manifest["success"], json.dumps(manifest, indent=2)
    (root / "artifacts.json").write_text(json.dumps(manifest))
    package = root / "package"
    assert build_runtime_package(root / "artifacts.json", package)["success"]
    loader = build_runtime_loader_manifest(package / "runtime-package.json")
    assert loader["success"] and len(loader["loadUnits"]) == 1, loader
    descriptor = build_native_loader_abi_descriptor(loader)
    assert descriptor["provenance"]["dispatchRegion"] == region.to_json()
    return descriptor, package, loader


def _values(descriptor, inputs, initial, expected):
    supplied, outputs = {}, {}
    for binding in descriptor["bindings"]:
        name = binding["name"]
        is_input = binding["coordinates"]["binding"] == 0
        values = inputs if is_input else initial
        supplied[name] = {"dtype": "uint32", "shape": [len(values)], "values": values}
        outputs[name] = {
            "dtype": "uint32",
            "shape": [len(values)],
            "values": inputs if is_input else expected,
        }
    return supplied, outputs


def _request(descriptor, package, region, inputs, initial, expected, **geometry):
    supplied, outputs = _values(descriptor, inputs, initial, expected)
    return build_native_loader_dispatch_request(
        descriptor,
        package,
        supplied,
        outputs,
        {
            "workgroupCount": list(region.workgroup_count),
            "workgroupSize": list(region.workgroup_size),
            **geometry,
        },
    )


@pytest.mark.parametrize("value", [None, [], {}, {"workgroupSize": [1]}, {"extra": 1}])
def test_dispatch_region_json_rejects_incomplete_contract(value):
    with pytest.raises(ValueError, match="exactly"):
        DispatchRegion.from_json(value)


def test_dispatch_region_json_roundtrip_is_independent():
    for region in plan_dispatch_regions([7, 5, 3], [4, 3, 2]):
        payload = region.to_json()
        assert DispatchRegion.from_json(payload) == region
        payload["threadGridSize"][0] = 1
        assert region.thread_grid_size == (7, 5, 3)


@pytest.mark.parametrize("target", ["directx", "opengl"])
def test_project_region_package_and_launch_contract(tmp_path, target):
    grid, size = (37, 1, 1), (32, 1, 1)
    region = plan_dispatch_regions(grid, size)[-1]
    descriptor, package, loader = _package(tmp_path, target, region)
    inputs, initial, expected = _case(grid, size)
    request = _request(descriptor, package, region, inputs, initial, expected)
    assert not request.execution_plan.diagnostics
    compact = _request(
        descriptor,
        package,
        region,
        inputs,
        initial,
        expected,
        workgroupCount=[1],
        workgroupSize=[5],
    )
    assert not compact.execution_plan.diagnostics
    _compile(request.artifact_path.read_text(), target, tmp_path)
    for geometry in (
        {"workgroupCount": [2, 1, 1]},
        {"threadGridSize": [5, 1, 1]},
    ):
        with pytest.raises(
            NativeLoaderDispatchError, match="dispatch-region-geometry-mismatch"
        ):
            _request(descriptor, package, region, inputs, initial, expected, **geometry)
    for contract in (
        None,
        {},
        {**region.to_json(), "workgroupSize": [4, 1, 1]},
        plan_dispatch_regions([37], [37])[0].to_json(),
    ):
        corrupted = copy.deepcopy(loader)
        corrupted["loadUnits"][0]["provenance"]["dispatchRegion"] = contract
        with pytest.raises(NativeLoaderABIError, match="dispatch-region-invalid"):
            build_native_loader_abi_descriptor(corrupted)


@pytest.mark.parametrize("target", ["directx", "opengl", "metal"])
def test_project_region_rejects_incompatible_execution(tmp_path, target):
    region = plan_dispatch_regions([37], [32])[-1]
    config = _config(tmp_path, target, region)
    config = replace(config, workgroup_size=(32, 1, 1))
    payload = translate_project(config, format_output=False).to_json()
    assert payload["summary"]["failedCount"] == 1
    assert "dispatch_region" in payload["artifacts"][0]["error"]
    assert not list((tmp_path / "out").rglob("*.hlsl"))
    assert not list((tmp_path / "out").rglob("*.glsl"))


@pytest.mark.parametrize("target", ["directx", "opengl"])
def test_region_identity_includes_source_grid_and_offset(tmp_path, target):
    first = plan_dispatch_regions([37], [32])[-1]
    second = plan_dispatch_regions([69], [32])[-1]
    a, _, _ = _package(tmp_path / "first", target, first)
    b, _, _ = _package(tmp_path / "second", target, second)
    assert a["source"]["hash"] == b["source"]["hash"]
    assert a["artifact"]["hash"] != b["artifact"]["hash"]
    assert a["entryPoint"]["executionConfig"] == b["entryPoint"]["executionConfig"]
    report_path = tmp_path / "first/report.json"
    payload = json.loads(report_path.read_text())
    for region in (None, second.to_json()):
        corrupted = copy.deepcopy(payload)
        if region is None:
            corrupted["artifacts"][0]["provenance"].pop("dispatchRegion")
        else:
            corrupted["artifacts"][0]["provenance"]["dispatchRegion"] = region
        report_path.write_text(json.dumps(corrupted))
        manifest = build_runtime_artifact_manifest(report_path)
        assert not manifest["success"]
        assert "dispatchRegion must match" in json.dumps(manifest["diagnostics"])


@pytest.mark.parametrize("scope", ["backend", "source", "target", "target-source"])
def test_region_source_option_resolution(tmp_path, scope):
    region = plan_dispatch_regions([37], [32])[-1]
    config = _config(tmp_path, "opengl", region)
    value = {"dispatch_region": region.to_json(), "software_subgroup_width": 32}
    if "source" in scope:
        value = {"source_patterns": {"*.metal": value}}
    if "target" in scope:
        value = {"target_options": {"opengl": value}}
    config = replace(config, source_options={"metal": value})
    report = translate_project(config, format_output=False)
    assert report.to_json()["summary"]["failedCount"] == 0
    report.write_json(tmp_path / "report.json")
    manifest = build_runtime_artifact_manifest(tmp_path / "report.json")
    assert manifest["success"], manifest


@pytest.mark.parametrize("target", ["directx", "opengl"])
def test_region_accepts_entry_workgroup_rules(tmp_path, target):
    region = plan_dispatch_regions([37], [32])[-1]
    config = replace(
        _config(tmp_path, target, region),
        workgroup_size=None,
        entry_workgroup_size_rules={"kernel.metal": {"*": region.workgroup_size}},
    )
    (tmp_path / "kernel.metal").write_text(
        """
#include <metal_stdlib>
using namespace metal;
template <typename T>
[[kernel]] void region_copy(device T* output [[buffer(0)]],
                          uint index [[thread_position_in_grid]],
                          uint3 size [[threads_per_threadgroup]]) {
    output[index] = T(simd_sum(index) + size.x);
}
instantiate_kernel("region_uint", region_copy, uint)
""",
        encoding="utf-8",
    )
    report = translate_project(config, format_output=False)
    assert report.to_json()["summary"]["failedCount"] == 0, json.dumps(
        report.to_json()["diagnostics"]
    )
    report.write_json(tmp_path / "report.json")
    manifest = build_runtime_artifact_manifest(tmp_path / "report.json")
    assert manifest["success"], manifest


def test_region_package_native_gate_is_required():
    from pathlib import Path

    from tools import ci_coverage

    workflow = (
        Path(__file__).resolve().parents[2] / ".github/workflows/mlx-portable-host.yml"
    ).read_text()
    step = ci_coverage.workflow_step_section(
        workflow, "Validate exact dispatch regions"
    )
    assert "test_dispatch_region_project.py" in step
    assert f'{REQUIRE_ENV}: "1"' in step
    assert "if:" not in step and "continue-on-error" not in step
    for event in ("push", "pull_request"):
        assert (
            "tests/test_translator/test_dispatch_region_project.py"
            in ci_coverage.workflow_event_path_filters(workflow, event)
        )


@pytest.mark.parametrize("grid,size", GRIDS)
def test_packaged_regions_execute_natively(tmp_path, grid, size):
    if sys.platform == "darwin":
        pytest.skip("Metal uses the separate exact-grid original/generated control")
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 to require native region packages")
    target = {"win32": "directx", "linux": "opengl"}[sys.platform]
    inputs, initial, expected = _case(grid, size)
    executor = _executor(target)
    requests, evidence = [], []
    for index, region in enumerate(plan_dispatch_regions(grid, size)):
        directory = tmp_path / str(index)
        descriptor, package, _ = _package(directory, target, region)
        request = _request(descriptor, package, region, inputs, initial, expected)
        assert not request.execution_plan.diagnostics
        _, module = _compile(request.artifact_path.read_text(), target, directory)
        assert module.is_file()
        state = RuntimeExecutionState(
            request=request,
            plan=request.execution_plan,
            loaded_artifact=(
                request.artifact_path.read_text()
                if target == "opengl"
                else module.read_bytes()
            ),
        )
        native = executor.runtime_adapter._prepare_dispatch_request(
            state, request.artifact_path, module
        )
        buffers = {
            name: replace(
                buffer,
                value=buffer.value if index == 0 else None,
                source=buffer.source if index == 0 else None,
                allocation=RuntimeAllocationView(allocation_id=name),
            )
            for name, buffer in native.buffers.items()
        }
        requests.append(replace(native, buffers=buffers))
        evidence.append(
            {
                "descriptor": descriptor,
                "region": region.to_json(),
                "artifactSha256": (
                    hashlib.sha256(request.artifact_path.read_bytes()).hexdigest()
                ),
                "moduleSha256": hashlib.sha256(module.read_bytes()).hexdigest(),
            }
        )
    result = executor.runtime_adapter.runtime.dispatch_sequence(None, None, requests)
    _, expected_outputs = _values(descriptor, inputs, initial, expected)
    (tmp_path / "evidence.json").write_text(
        json.dumps(
            {
                "grid": grid,
                "size": size,
                "sourceSha256": hashlib.sha256(SOURCE.encode()).hexdigest(),
                "regions": evidence,
                "inputs": inputs,
                "initial": initial,
                "expected": expected,
                "result": result,
            },
            indent=2,
        )
    )
    assert result == expected_outputs

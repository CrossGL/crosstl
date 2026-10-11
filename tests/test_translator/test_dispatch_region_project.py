"""Project packages retain exact-grid specialization through native loading."""

import copy
import hashlib
import json
import os
import sys
from dataclasses import replace
from types import SimpleNamespace

import pytest

from crosstl.project import (
    ProjectConfig,
    build_native_loader_abi_descriptor,
    build_native_loader_dispatch_request,
    build_runtime_artifact_manifest,
    build_runtime_loader_manifest,
    build_runtime_package,
    prepare_native_loader_dispatch_regions,
    select_native_loader_dispatch_regions,
    translate_project,
)
from crosstl.project.native_loader_abi import NativeLoaderABIError
from crosstl.project.native_loader_dispatch import NativeLoaderDispatchError
from crosstl.translator.dispatch_regions import DispatchRegion, plan_dispatch_regions
from tests.ci_helpers import assert_paths_covered
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


def _package(root, target, region, *, config=None):
    report = translate_project(
        config or _config(root, target, region), format_output=False
    )
    payload = report.to_json()
    assert payload["summary"]["failedCount"] == 0, json.dumps(
        payload["diagnostics"], indent=2
    )
    assert len(payload["artifacts"]) == 1
    assert set(payload["artifacts"][0]["provenance"]) == {
        "pipeline",
        "intermediate",
        "dispatchRegion",
        "dispatchRegionProgram",
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


@pytest.mark.parametrize("target", ["directx", "opengl"])
def test_region_selection_preserves_program_and_complete_coverage(tmp_path, target):
    regions = plan_dispatch_regions([37], [32])
    packages = [
        _package(tmp_path / str(i), target, region)[:2]
        for i, region in enumerate(regions)
    ]
    selected = select_native_loader_dispatch_regions(
        packages[::-1], thread_grid_size=[37], source_workgroup_size=[32]
    )
    assert selected == tuple(packages)
    programs = [
        descriptor["provenance"]["dispatchRegionProgram"] for descriptor, _ in selected
    ]
    assert programs[0] == programs[1]
    assert programs[0]["sourceEntryPoint"] != "main"
    for invalid in (packages[:1], packages + packages[:1], [packages[0], packages[0]]):
        with pytest.raises(NativeLoaderDispatchError, match="coverage-invalid"):
            select_native_loader_dispatch_regions(
                invalid, thread_grid_size=[37], source_workgroup_size=[32]
            )
    with pytest.raises(NativeLoaderDispatchError, match="coverage-invalid"):
        select_native_loader_dispatch_regions(
            packages, thread_grid_size=[69], source_workgroup_size=[32]
        )
    legacy = copy.deepcopy(packages)
    legacy[1][0]["provenance"].pop("dispatchRegionProgram")
    with pytest.raises(NativeLoaderDispatchError, match="program-missing"):
        select_native_loader_dispatch_regions(
            legacy, thread_grid_size=[37], source_workgroup_size=[32]
        )


@pytest.mark.parametrize("change", ["entry", "include", "policy", "implementation"])
def test_region_selection_rejects_mixed_programs(tmp_path, change, monkeypatch):
    from crosstl.translator import dispatch_region_identity

    source = """
#include <metal_stdlib>
#include "value.h"
using namespace metal;
kernel void first(device uint* output [[buffer(0)]], uint tid [[thread_position_in_grid]]) {
    output[tid] = VALUE;
}
kernel void second(device uint* output [[buffer(0)]], uint tid [[thread_position_in_grid]]) {
    output[tid] = VALUE + 1;
}
"""
    packages = []
    for i, region in enumerate(plan_dispatch_regions([37], [32])):
        root = tmp_path / str(i)
        config = replace(
            _config(root, "opengl", region),
            entry_points={
                "kernel.metal": "second" if i and change == "entry" else "first"
            },
        )
        config.source_options["metal"]["target_options"]["opengl"].pop(
            "software_subgroup_width"
        )
        (root / "kernel.metal").write_text(source)
        (root / "value.h").write_text(
            f"#define VALUE {2 if i and change == 'include' else 1}\n"
        )
        if i and change == "policy":
            config.source_options["metal"]["target_options"]["opengl"][
                "private_pointer_out_of_bounds_read"
            ] = "zero"
        if i and change == "implementation":
            monkeypatch.setattr(
                dispatch_region_identity,
                "translation_implementation_hash",
                lambda: "a" * 64,
            )
        packages.append(_package(root, "opengl", region, config=config)[:2])
    a, b = (pair[0] for pair in packages)
    assert a["source"]["hash"] == b["source"]["hash"]
    assert a["entryPoint"]["name"] == b["entryPoint"]["name"] == "main"
    assert (
        a["provenance"]["dispatchRegionProgram"]
        != b["provenance"]["dispatchRegionProgram"]
    )
    with pytest.raises(NativeLoaderDispatchError, match="program-mismatch"):
        select_native_loader_dispatch_regions(
            packages, thread_grid_size=[37], source_workgroup_size=[32]
        )


@pytest.mark.parametrize(
    "field,value",
    [
        ("hash", "0" * 64),
        ("sourceEntryPoint", "other"),
        ("target", "metal"),
        ("schemaVersion", True),
    ],
)
def test_region_program_metadata_is_validated(tmp_path, field, value):
    region = plan_dispatch_regions([37], [32])[-1]
    _, _, loader = _package(tmp_path, "opengl", region)
    loader["loadUnits"][0]["provenance"]["dispatchRegionProgram"][field] = value
    with pytest.raises(NativeLoaderABIError, match="program-invalid"):
        build_native_loader_abi_descriptor(loader)
    payload = json.loads((tmp_path / "report.json").read_text())
    payload["artifacts"][0]["provenance"]["dispatchRegionProgram"][field] = value
    (tmp_path / "report.json").write_text(json.dumps(payload))
    result = build_runtime_artifact_manifest(tmp_path / "report.json")
    assert not result["success"]
    assert "dispatchRegionProgram" in json.dumps(result["diagnostics"])


def test_region_selection_rejects_different_configured_subgroup_widths(tmp_path):
    packages = []
    for i, region in enumerate(plan_dispatch_regions([192], [128])):
        root = tmp_path / str(i)
        config = replace(
            _config(root, "directx", region),
            workgroup_size=None,
            entry_workgroup_size_rules={"kernel.metal": {"*": region.workgroup_size}},
            subgroup_width_rules={"kernel.metal": 32 if i == 0 else 64},
            entry_points={"kernel.metal": "subgroup_uint"},
        )
        config.source_options["metal"]["target_options"]["directx"].pop(
            "software_subgroup_width"
        )
        (root / "kernel.metal").write_text("""
#include <metal_stdlib>
using namespace metal;
template <typename T>
[[kernel]] void subgroup_copy(device T* output [[buffer(0)]], uint tid [[thread_position_in_grid]]) {
    output[tid] = T(simd_sum(tid));
}
instantiate_kernel("subgroup_uint", subgroup_copy, uint)
""")
        packages.append(_package(root, "directx", region, config=config)[:2])
    first, second = (
        pair[0]["provenance"]["dispatchRegionProgram"] for pair in packages
    )
    assert first["intermediateHash"] == second["intermediateHash"]
    assert first["settingsHash"] != second["settingsHash"]
    with pytest.raises(NativeLoaderDispatchError, match="program-mismatch"):
        select_native_loader_dispatch_regions(
            packages, thread_grid_size=[192], source_workgroup_size=[128]
        )


def test_region_preparation_checks_all_artifacts_before_compilation(tmp_path):
    packages = [
        _package(tmp_path / str(i), "opengl", region)[:2]
        for i, region in enumerate(plan_dispatch_regions([37], [32]))
    ]
    inputs, initial, expected = _case((37, 1, 1), (32, 1, 1))
    supplied, outputs = _values(packages[0][0], inputs, initial, expected)
    descriptor, root = packages[-1]
    (root / descriptor["artifact"]["packagePath"]).write_text("changed")
    adapter = SimpleNamespace(
        target="opengl",
        prepare_buffers=lambda state: pytest.fail(
            "must validate every artifact before compilation"
        ),
    )
    with pytest.raises(NativeLoaderDispatchError, match="artifact"):
        with prepare_native_loader_dispatch_regions(
            packages,
            supplied,
            outputs,
            thread_grid_size=[37],
            source_workgroup_size=[32],
            adapter=adapter,
        ):
            pytest.fail("must reject changed package")


@pytest.mark.parametrize("fail_index", [0, 1])
def test_region_preparation_cleans_up_after_compile_failure(tmp_path, fail_index):
    import tempfile
    from pathlib import Path

    packages = [
        _package(tmp_path / str(i), "opengl", region)[:2]
        for i, region in enumerate(plan_dispatch_regions([37], [32]))
    ]
    inputs, initial, expected = _case((37, 1, 1), (32, 1, 1))
    supplied, outputs = _values(packages[0][0], inputs, initial, expected)
    directories = []
    adapter = _executor("opengl").runtime_adapter

    def prepare(state):
        directory = tempfile.TemporaryDirectory()
        directories.append(Path(directory.name))
        state.temporary_directories.append(directory)
        if len(directories) == fail_index + 1:
            raise RuntimeError("compiler failed")
        state.loaded_artifact = state.request.artifact_path.read_text()
        return adapter._prepare_dispatch_request(
            state, state.request.artifact_path, state.request.artifact_path
        )

    with pytest.raises(RuntimeError, match="compiler failed"):
        with prepare_native_loader_dispatch_regions(
            packages,
            supplied,
            outputs,
            thread_grid_size=[37],
            source_workgroup_size=[32],
            adapter=SimpleNamespace(target="opengl", prepare_buffers=prepare),
        ):
            pytest.fail("must not dispatch")
    assert directories and not any(path.exists() for path in directories)


def test_region_preparation_retains_explicit_allocation_views(tmp_path):
    from crosstl.project.runtime_verification import RuntimeAllocationView

    packages = [
        _package(tmp_path / str(i), "opengl", region)[:2]
        for i, region in enumerate(plan_dispatch_regions([37], [32]))
    ]
    inputs, initial, expected = _case((37, 1, 1), (32, 1, 1))
    supplied, outputs = _values(packages[0][0], inputs, initial, expected)
    adapter = _executor("opengl").runtime_adapter
    allocations = {}

    def prepare(state):
        state.loaded_artifact = state.request.artifact_path.read_text()
        native = adapter._prepare_dispatch_request(
            state, state.request.artifact_path, state.request.artifact_path
        )
        buffers = {}
        for name, buffer in native.buffers.items():
            allocations[name] = RuntimeAllocationView(
                allocation_id=name, byte_offset=32, byte_length=4 * buffer.shape[0]
            )
            buffers[name] = replace(buffer, allocation=allocations[name])
        return replace(native, buffers=buffers)

    with prepare_native_loader_dispatch_regions(
        packages,
        supplied,
        outputs,
        thread_grid_size=[37],
        source_workgroup_size=[32],
        adapter=SimpleNamespace(target="opengl", prepare_buffers=prepare),
    ) as requests:
        for index, request in enumerate(requests):
            for name, buffer in request.buffers.items():
                assert buffer.allocation == allocations[name]
                assert (buffer.value is not None) == (index == 0)
                assert (buffer.upload_snapshot is not None) == (index == 0)


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
        Path(__file__).resolve().parents[2]
        / ".github/workflows/demo-project-testing.yml"
    ).read_text()
    step = ci_coverage.workflow_step_section(
        workflow, "Validate exact dispatch regions"
    )
    assert "test_dispatch_region_project.py" in step
    assert f'{REQUIRE_ENV}: "1"' in step
    assert "if:" not in step and "continue-on-error" not in step
    for event in ("push", "pull_request"):
        assert_paths_covered(
            ci_coverage.workflow_event_path_filters(workflow, event),
            "tests/test_translator/test_dispatch_region_project.py",
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
    packages, evidence = [], []
    for index, region in enumerate(plan_dispatch_regions(grid, size)):
        directory = tmp_path / str(index)
        descriptor, package, _ = _package(directory, target, region)
        artifact = package / descriptor["artifact"]["packagePath"]
        _, validated_module = _compile(artifact.read_text(), target, directory)
        assert validated_module.is_file()
        packages.append((descriptor, package))
    supplied, expected_outputs = _values(descriptor, inputs, initial, expected)
    with prepare_native_loader_dispatch_regions(
        packages[::-1],
        supplied,
        expected_outputs,
        thread_grid_size=grid,
        source_workgroup_size=size,
        adapter=executor.runtime_adapter,
    ) as requests:
        for index, (request, (descriptor, _)) in enumerate(zip(requests, packages)):
            assert request.module_path.is_file()
            for buffer in request.buffers.values():
                assert buffer.allocation is not None
                assert (buffer.value is not None) == (index == 0)
            # Retain the validated module before the preparation context cleans up.
            module = tmp_path / str(index) / request.module_path.name
            module.write_bytes(request.module_path.read_bytes())
            evidence.append(
                {
                    "descriptor": descriptor,
                    "region": descriptor["provenance"]["dispatchRegion"],
                    "artifactSha256": (
                        hashlib.sha256(request.artifact_path.read_bytes()).hexdigest()
                    ),
                    "moduleSha256": hashlib.sha256(module.read_bytes()).hexdigest(),
                    "moduleFile": module.name,
                }
            )
        result = executor.runtime_adapter.runtime.dispatch_sequence(
            None, None, requests
        )
    assert all(not request.module_path.exists() for request in requests)
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


CONSTANT_SOURCE = """#include <metal_stdlib>
using namespace metal;
kernel void update(device uint* values [[buffer(0)]],
                   constant uint& bias [[buffer(1)]],
                   uint3 tid [[thread_position_in_grid]],
                   uint3 gid [[threadgroup_position_in_grid]],
                   uint3 grid [[threads_per_grid]]) {
    uint index = tid.x + grid.x * (tid.y + grid.y * tid.z);
    values[16u + index] += bias + gid.x + 11u * gid.y + 101u * gid.z;
}
"""


@pytest.mark.parametrize(
    "grid,size", [((37, 1, 1), (32, 1, 1)), ((7, 5, 3), (4, 3, 2))]
)
def test_packaged_regions_reuse_source_constants(tmp_path, grid, size):
    if sys.platform == "darwin":
        pytest.skip("DirectX/OpenGL region sequences require their native platforms")
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 to require native region packages")
    target = {"win32": "directx", "linux": "opengl"}[sys.platform]
    count = grid[0] * grid[1] * grid[2]
    guard = 0x5A39E714
    bias = 37
    initial = [guard] * 16 + [3 * i + 1 for i in range(count)] + [guard] * 16
    expected = initial.copy()
    for z in range(grid[2]):
        for y in range(grid[1]):
            for x in range(grid[0]):
                index = x + grid[0] * (y + grid[1] * z)
                expected[16 + index] += (
                    bias + x // size[0] + 11 * (y // size[1]) + 101 * (z // size[2])
                )
    packages = []
    for index, region in enumerate(plan_dispatch_regions(grid, size)):
        directory = tmp_path / str(index)
        config = _config(directory, target, region)
        config.source_options["metal"]["target_options"][target].pop(
            "software_subgroup_width"
        )
        config.source_options["metal"]["target_options"][target].pop(
            "relative_wave_shuffle_out_of_range", None
        )
        (directory / "kernel.metal").write_text(CONSTANT_SOURCE, encoding="utf-8")
        descriptor, package, _ = _package(directory, target, region, config=config)
        artifact = package / descriptor["artifact"]["packagePath"]
        _, compiled = _compile(artifact.read_text(), target, directory)
        assert compiled.is_file()
        packages.append((descriptor, package))
    supplied, outputs = {}, {}
    for binding in descriptor["bindings"]:
        if "executionInput" in binding.get("provenance", {}):
            continue
        is_output = binding["coordinates"]["binding"] == 0
        values = initial if is_output else [bias]
        supplied[binding["name"]] = {
            "dtype": "uint32",
            "shape": [len(values)],
            "values": values,
        }
        if is_output:
            outputs[binding["name"]] = {
                "dtype": "uint32",
                "shape": [len(expected)],
                "values": expected,
            }
        else:
            constant_name = binding["name"]
    executor = _executor(target)
    evidence = []
    with prepare_native_loader_dispatch_regions(
        packages[::-1],
        supplied,
        outputs,
        thread_grid_size=grid,
        source_workgroup_size=size,
        adapter=executor.runtime_adapter,
    ) as requests:
        assert len(requests) > 1
        shared_constant = requests[0].buffers[constant_name].allocation
        derived_ids = set()
        for index, (request, (descriptor, _)) in enumerate(zip(requests, packages)):
            constant = request.buffers[constant_name]
            assert constant.allocation == shared_constant
            assert (constant.value is not None) == (index == 0)
            derived_names = {
                binding["name"]
                for binding in descriptor["bindings"]
                if "executionInput" in binding.get("provenance", {})
            }
            for name in derived_names:
                derived = request.buffers[name]
                assert derived.value is not None
                assert derived.allocation.allocation_id not in derived_ids
                assert derived.allocation != shared_constant
                derived_ids.add(derived.allocation.allocation_id)
            module = tmp_path / str(index) / request.module_path.name
            module.write_bytes(request.module_path.read_bytes())
            evidence.append(
                {
                    "descriptor": descriptor,
                    "artifactSha256": (
                        hashlib.sha256(request.artifact_path.read_bytes()).hexdigest()
                    ),
                    "moduleSha256": hashlib.sha256(module.read_bytes()).hexdigest(),
                    "moduleFile": module.name,
                    "sourceConstantAllocation": shared_constant.to_json(),
                    "derivedConstants": sorted(derived_names),
                }
            )
        result = executor.runtime_adapter.runtime.dispatch_sequence(
            None, None, requests
        )
    (tmp_path / "evidence.json").write_text(
        json.dumps(
            {
                "target": target,
                "grid": grid,
                "size": size,
                "sourceSha256": hashlib.sha256(CONSTANT_SOURCE.encode()).hexdigest(),
                "bias": bias,
                "initial": initial,
                "expected": expected,
                "result": result,
                "regions": evidence,
            },
            indent=2,
        )
    )
    assert result == outputs

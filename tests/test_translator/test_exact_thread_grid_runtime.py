"""Exact thread grids retain partial groups and source invocation coordinates."""

import hashlib
import itertools
import json
import math
import os
from dataclasses import replace
from types import SimpleNamespace

import pytest

from crosstl.project import (
    NativeLoaderDispatchError,
    build_native_loader_dispatch_request,
)
from crosstl.project.metal_runtime import MetalComputeRuntime
from crosstl.project.native_runtime_drivers import _workgroup_count
from crosstl.project.runtime_verification import (
    RuntimeAdapterSetupError,
    RuntimeDispatchGeometry,
    RuntimeVerificationError,
    _complete_runtime_dispatch_geometry,
    _merge_runtime_dispatch,
    _parse_runtime_dispatch_geometry,
)
from tests.test_translator.test_metal_builtin_ownership import _compile
from tests.test_translator.test_metal_native_runtime import (
    _execute_native,
    _native_request,
    _package,
    _request,
)
from tests.test_translator.test_native_loader_dispatch import (
    _inputs,
    _outputs,
    _write_descriptor,
)
from tests.test_translator.test_native_loader_dispatch_integration import _executor


def test_exact_grid_survives_runtime_contracts():
    value = {
        "workgroupCount": [2, 1, 1],
        "workgroupSize": [32, 1, 1],
        "threadGridSize": [37, 1, 1],
    }
    parsed = _parse_runtime_dispatch_geometry(value, field_name="dispatch")
    assert parsed.to_json() == value
    merged = _merge_runtime_dispatch(RuntimeDispatchGeometry(entry_point="run"), parsed)
    completed = _complete_runtime_dispatch_geometry(
        SimpleNamespace(dispatch=merged, entry_points=())
    )
    assert completed.thread_grid_size == (37, 1, 1)
    assert completed.global_size == (37, 1, 1)
    assert completed.workgroup_count == (2, 1, 1)
    assert completed.entry_point == "run"


@pytest.mark.parametrize("grid", [None, [], [0], [-1], [True], [1.5], [1, 1, 1, 1]])
def test_runtime_contract_cannot_discard_invalid_exact_grid(grid):
    with pytest.raises(RuntimeVerificationError, match="threadGridSize"):
        _parse_runtime_dispatch_geometry(
            {"threadGridSize": grid}, field_name="dispatch"
        )


@pytest.mark.parametrize("grid", [[1, 1, 1], [3, 1, 1], [4, 1, 1]])
def test_metal_exact_grid_preflight(tmp_path, grid):
    _, descriptor, package, inputs, outputs = _request(tmp_path)
    geometry = {
        "workgroupCount": [(grid[0] + 1) // 2, 1, 1],
        "workgroupSize": [2, 1, 1],
        "threadGridSize": grid,
    }
    request = build_native_loader_dispatch_request(
        descriptor, package, inputs, outputs, geometry
    )
    _, native = _native_request(request)
    payload, _ = MetalComputeRuntime()._prepare_request(native)
    assert payload["threadGridSize"] == grid
    assert native.dispatch.thread_grid_size == tuple(grid)
    assert native.dispatch.global_size == tuple(grid)
    assert native.dispatch.grid_size == tuple(grid)


@pytest.mark.parametrize(
    "grid", [[], None, False, 0, [0], [-1], [True], [1.5], [1, 1, 1, 1]]
)
def test_loader_rejects_invalid_exact_grid(tmp_path, grid):
    _, descriptor, package, inputs, outputs = _request(tmp_path)
    with pytest.raises(NativeLoaderDispatchError):
        build_native_loader_dispatch_request(
            descriptor,
            package,
            inputs,
            outputs,
            {
                "workgroupCount": [2, 2, 1],
                "workgroupSize": [2, 1, 1],
                "threadGridSize": grid,
            },
        )


@pytest.mark.parametrize(
    "extra",
    [
        {"threadGridSize": [2, 2, 1]},
        {"threadGridSize": [5, 2, 1]},
        {"threadGridSize": [3, 2, 1], "globalSize": [4, 2, 1]},
        {"threadGridSize": [3, 2, 1], "gridSize": [4, 2, 1]},
    ],
)
def test_loader_rejects_conflicting_exact_grid(tmp_path, extra):
    _, descriptor, package, inputs, outputs = _request(tmp_path)
    with pytest.raises(NativeLoaderDispatchError, match="dispatch-size-mismatch"):
        build_native_loader_dispatch_request(
            descriptor,
            package,
            inputs,
            outputs,
            {"workgroupCount": [2, 2, 1], "workgroupSize": [2, 1, 1], **extra},
        )


@pytest.mark.parametrize(
    "grid",
    [
        None,
        False,
        0,
        [],
        (True,),
        (0,),
        (-1,),
        (1.5,),
        "123",
        (2, 2, 1),
        (5, 2, 1),
        (1 << 32, 2, 1),
    ],
)
def test_native_metal_rechecks_exact_grid(tmp_path, grid):
    request, _, _, _, _ = _request(tmp_path)
    _, native = _native_request(request)
    native = replace(native, dispatch=replace(native.dispatch, thread_grid_size=grid))
    with pytest.raises(RuntimeAdapterSetupError):
        MetalComputeRuntime()._prepare_request(native)


@pytest.mark.parametrize("target", ["directx", "opengl"])
def test_unimplemented_targets_never_round_up_exact_grid(tmp_path, target):
    descriptor = _write_descriptor(tmp_path, target)
    with pytest.raises(
        NativeLoaderDispatchError, match="exact-thread-grid-unsupported"
    ):
        build_native_loader_dispatch_request(
            descriptor,
            tmp_path,
            _inputs(),
            _outputs(),
            {"workgroupCount": [2, 1, 1], "threadGridSize": [65, 1, 1]},
        )


@pytest.mark.parametrize("target", ["DirectX", "OpenGL", "Vulkan"])
def test_native_driver_cannot_bypass_exact_grid_requirement(target):
    request = SimpleNamespace(
        dispatch=RuntimeDispatchGeometry(
            workgroup_count=(2, 1, 1),
            workgroup_size=(32, 1, 1),
            thread_grid_size=(37, 1, 1),
        )
    )
    with pytest.raises(RuntimeAdapterSetupError) as failure:
        _workgroup_count(request, target=target)
    assert failure.value.details["reasonKind"] == "exact-thread-grid-unsupported"


SOURCE = """#include <metal_stdlib>
using namespace metal;
kernel void transform(device uint* inputWords [[buffer(0)]],
                      device uint* outputWords [[buffer(1)]],
                      uint3 tid [[thread_position_in_grid]],
                      uint3 gid [[threadgroup_position_in_grid]],
                      uint3 grid [[threads_per_grid]],
                      uint3 size [[threads_per_threadgroup]],
                      uint3 lid [[thread_position_in_threadgroup]],
                      uint3 groups [[threadgroups_per_grid]],
                      uint lane [[thread_index_in_simdgroup]],
                      uint sid [[simdgroup_index_in_threadgroup]],
                      uint sgcount [[simdgroups_per_threadgroup]]) {
    uint index = tid.x + grid.x * (tid.y + grid.y * tid.z);
    uint total = simd_sum(inputWords[index]);
    uint base = 17u + index * 22u;
    outputWords[base] = tid.x;
    outputWords[base + 1u] = tid.y;
    outputWords[base + 2u] = tid.z;
    outputWords[base + 3u] = gid.x;
    outputWords[base + 4u] = gid.y;
    outputWords[base + 5u] = gid.z;
    outputWords[base + 6u] = grid.x;
    outputWords[base + 7u] = grid.y;
    outputWords[base + 8u] = grid.z;
    outputWords[base + 9u] = size.x;
    outputWords[base + 10u] = size.y;
    outputWords[base + 11u] = size.z;
    outputWords[base + 12u] = lid.x;
    outputWords[base + 13u] = lid.y;
    outputWords[base + 14u] = lid.z;
    outputWords[base + 15u] = groups.x;
    outputWords[base + 16u] = groups.y;
    outputWords[base + 17u] = groups.z;
    outputWords[base + 18u] = lane;
    outputWords[base + 19u] = sid;
    outputWords[base + 20u] = sgcount;
    outputWords[base + 21u] = total;
}
"""
GRIDS = [((n, 1, 1), (32, 1, 1)) for n in (1, 5, 31, 32, 33, 37, 63, 65)] + [
    ((1025, 1, 1), (1024, 1, 1)),
    ((35, 3, 1), (32, 2, 1)),
    ((7, 5, 3), (4, 3, 2)),
    ((2, 5, 3), (4, 3, 2)),
]


def _coordinates(size):
    return (
        tuple(reversed(point))
        for point in itertools.product(*(range(n) for n in reversed(size)))
    )


@pytest.mark.parametrize("grid,size", GRIDS)
def test_metal_exact_thread_grid_native(tmp_path, grid, size):
    if os.environ.get("CROSTL_REQUIRE_METAL_PACKAGE_RUNTIME") != "1":
        pytest.skip("requires the macOS native package runtime gate")
    descriptor, package = _package(tmp_path, SOURCE)
    counts = tuple((n + w - 1) // w for n, w in zip(grid, size))
    padded = math.prod(n * w for n, w in zip(counts, size))
    input_words = [(i * 17 + 3) % 251 for i in range(padded + 17)]
    output_words = [0xFFFFFFFF] * (34 + padded * 22)
    expected = output_words.copy()
    for gid in _coordinates(counts):
        active = tuple(min(w, n - g * w) for n, w, g in zip(grid, size, gid))
        group = []
        for lid in _coordinates(active):
            tid = tuple(g * w + local for g, w, local in zip(gid, size, lid))
            index = tid[0] + grid[0] * (tid[1] + grid[1] * tid[2])
            group.append((tid, lid, index))
        for linear, (tid, lid, index) in enumerate(group):
            sid, lane = divmod(linear, 32)
            total = sum(
                input_words[item[2]] for item in group[sid * 32 : (sid + 1) * 32]
            )
            values = [
                *tid,
                *gid,
                *grid,
                *active,
                *lid,
                *counts,
                lane,
                sid,
                (len(group) + 31) // 32,
                total,
            ]
            expected[17 + index * 22 : 17 + (index + 1) * 22] = values
    inputs = {
        "inputWords": {
            "dtype": "uint32",
            "shape": [len(input_words)],
            "values": input_words,
        },
        "outputWords": {
            "dtype": "uint32",
            "shape": [len(output_words)],
            "values": output_words,
        },
    }
    outputs = {
        "inputWords": inputs["inputWords"],
        "outputWords": {**inputs["outputWords"], "values": expected},
    }
    geometry = {"workgroupCount": counts, "workgroupSize": size, "threadGridSize": grid}
    request = build_native_loader_dispatch_request(
        descriptor, package, inputs, outputs, geometry
    )
    result = _execute_native(request, tmp_path)
    executor = _executor("metal")
    original_directory = tmp_path / "original"
    original_directory.mkdir()
    _, original_library = _compile(SOURCE, "metal", original_directory)
    try:
        state, native = _native_request(request)
        original = executor.runtime_adapter.runtime.dispatch(
            None,
            state,
            replace(native, module_path=original_library, entry_point="transform"),
        )
    finally:
        executor.runtime_adapter.runtime.close()
    evidence = {
        "geometry": geometry,
        "inputs": inputs,
        "expected": outputs,
        "generated": result.outputs,
        "original": original,
        "sourceSha256": (
            hashlib.sha256((tmp_path / "sample.metal").read_bytes()).hexdigest()
        ),
        "artifactSha256": descriptor["artifact"]["hash"]["value"],
    }
    (tmp_path / "evidence.json").write_text(
        json.dumps(evidence, indent=2), encoding="utf-8"
    )
    for name in outputs:
        assert result.outputs[name]["values"] == outputs[name]["values"]
        assert original[name]["values"] == outputs[name]["values"]

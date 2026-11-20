"""Metal's lane-index intrinsic in helper functions (crosstl/backend/Metal/simdgroup_lane_queries.py).

A 64-thread group holds two SIMD-groups of 32 lanes, so thread ``i`` is lane ``i % 32`` on Apple GPUs and under
DirectX and OpenGL software subgroups alike.
"""

import json
import os
import sys

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
from crosstl.translator.refusals import TranslationRefusal
from tests.test_translator.test_boolean_buffer_runtime import _bound_values
from tests.test_translator.test_loop_updates import _execute
from tests.test_translator.test_metal_builtin_ownership import _compile

REQUIRE_ENV = "CROSTL_REQUIRE_MPP_MATMUL2D_RUNTIME"
TARGETS = ("directx", "opengl", "metal")
THREADS = 64
GUARD = -1.0
QUERY_ONLY = """#include <metal_stdlib>
using namespace metal;
static inline ushort lane_of_caller() { return __metal_get_thread_index_in_simdgroup(ushort()); }
kernel void lanes(device float* dst [[buffer(0)]], uint gid [[thread_position_in_grid]]) {
  dst[gid] = float(lane_of_caller());
}
"""
# The query decides which lane each shuffle reads, as in a rotation by one lane.
WITH_SHUFFLE = """#include <metal_stdlib>
using namespace metal;
static inline float rotated(float value) {
  const ushort lane = __metal_get_thread_index_in_simdgroup(ushort());
  return simd_shuffle(value, ushort((lane + 1) & 31));
}
kernel void lanes(device float* dst [[buffer(0)]], uint gid [[thread_position_in_grid]]) {
  const float neighbour = rotated(float(gid));
  dst[gid] = neighbour + 1000.0f * float(__metal_get_thread_index_in_simdgroup(uint()));
}
"""
EXPECTED = {
    "query": [float(i % 32) for i in range(THREADS)],
    "shuffle": [
        float((i // 32) * 32 + (i % 32 + 1) % 32) + 1000.0 * (i % 32)
        for i in range(THREADS)
    ],
}
SOURCES = {"query": QUERY_ONLY, "shuffle": WITH_SHUFFLE}


def _request(root, target, kind):
    (root / "lanes.metal").write_text(SOURCES[kind], encoding="utf-8")
    metal = {}
    if target != "metal":
        options = {"software_subgroup_width": 32}
        if target == "directx":
            options["relative_wave_shuffle_out_of_range"] = "self"
        metal["target_options"] = {target: options}
    report = translate_project(
        ProjectConfig(
            root=root,
            include_patterns=("lanes.metal",),
            targets=(target,),
            output_dir="out",
            workgroup_size=(THREADS, 1, 1),
            source_options={"metal": metal},
        ),
        format_output=False,
    )
    report.write_json(root / "report.json")
    assert report.to_json()["summary"]["failedCount"] == 0, json.dumps(report.to_json())
    manifest = build_runtime_artifact_manifest(root / "report.json")
    assert manifest["success"], manifest
    (root / "artifacts.json").write_text(json.dumps(manifest))
    package = root / "package"
    assert build_runtime_package(root / "artifacts.json", package)["success"]
    loader = build_runtime_loader_manifest(package / "runtime-package.json")
    descriptor = build_native_loader_abi_descriptor(
        loader, load_unit_id=loader["loadUnits"][0]["id"]
    )

    def values(data):
        return _bound_values(
            descriptor,
            {"dst": {"dtype": "float32", "shape": [THREADS], "values": data}},
        )

    request = build_native_loader_dispatch_request(
        descriptor,
        package,
        values([GUARD] * THREADS),
        values(EXPECTED[kind]),
        {"workgroupCount": [1, 1, 1], "workgroupSize": [THREADS, 1, 1]},
        expected_target=target,
    )
    assert not request.execution_plan.diagnostics
    return request, values(EXPECTED[kind])


@pytest.mark.parametrize("kind", sorted(SOURCES))
@pytest.mark.parametrize("target", TARGETS)
def test_lane_queries_translate_and_compile(tmp_path, target, kind):
    request, _ = _request(tmp_path, target, kind)
    generated = request.artifact_path.read_text()
    assert "__metal_get_thread_index_in_simdgroup" not in generated
    if target == "directx":
        assert "% 32u)" in generated and "WaveGetLaneIndex" not in generated
    elif target == "opengl":
        assert "gl_LocalInvocationIndex %" in generated
        assert "gl_SubgroupInvocationID" not in generated
    else:
        assert "[[thread_index_in_simdgroup]]" in generated
    _compile(generated, target, tmp_path)


@pytest.mark.parametrize("argument", ["5", "float()", "", "ushort(), ushort()"])
def test_other_argument_forms_are_refused(tmp_path, argument):
    source = QUERY_ONLY.replace(
        "__metal_get_thread_index_in_simdgroup(ushort())",
        f"__metal_get_thread_index_in_simdgroup({argument})",
    )
    path = tmp_path / "lanes.metal"
    path.write_text(source, encoding="utf-8")
    from crosstl._crosstl import translate

    with pytest.raises(TranslationRefusal, match="value-initialised integer type"):
        translate(str(path), backend="cgl")


@pytest.mark.parametrize("kind", sorted(SOURCES))
def test_lane_queries_match_the_hardware_lanes_natively(tmp_path, kind):
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 for required lane-query execution")
    target = {"darwin": "metal", "win32": "directx", "linux": "opengl"}[sys.platform]
    request, expected = _request(tmp_path, target, kind)
    _execute(
        request,
        expected,
        tmp_path,
        original_source=SOURCES[kind] if target == "metal" else None,
        original_entry="lanes",
    )

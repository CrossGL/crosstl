"""Software subgroups lower every relative shuffle, not only ``simd_shuffle_down``.

Metal's ``simd_shuffle_up`` reads lane ``lane - delta`` and leaves the lower ``delta`` lanes unchanged, and
``simd_shuffle_xor`` reads lane ``lane ^ mask`` (MSL specification, SIMD-group functions). Butterfly row
reductions such as MLX's ``BaseMMAFrag::row_reduce`` combine values with ``simd_shuffle_xor``. DirectX and
OpenGL software subgroup lowering accepted only shuffle-down and lane reads, and refused the rest with
``operation-unsupported``. Both now exchange the value through the shared scratch array and keep the calling
lane's value when the source lane is outside the active subgroup.
"""

import os
import sys

import pytest

from crosstl.project import build_native_loader_dispatch_request
from tests.test_translator.test_boolean_buffer_runtime import _bound_values
from tests.test_translator.test_metal_builtin_ownership import _compile
from tests.test_translator.test_software_subgroup_product import _package

REQUIRE_ENV = "CROSTL_REQUIRE_SOFTWARE_SUBGROUP_RELATIVE_SHUFFLES"
ENTRY = "relative_shuffles"
WIDTH = 32
WORKGROUP = (64, 1, 1)
GROUPS = 2
THREADS = GROUPS * WORKGROUP[0]
OUTPUTS = 6
GUARD = [0x6D2B79F5, -7, 0x1234567, -0x5A5A5A5]
VALUES = [((index * 2654435761) % 4001) - 2000 for index in range(THREADS)]

SOURCE = """#include <metal_stdlib>
using namespace metal;

[[kernel]] void relative_shuffles(
    const device int* values [[buffer(0)]],
    device int* results [[buffer(1)]],
    uint index [[thread_position_in_grid]]) {
  int v = values[index];
  float f = float(v) * 0.5f;
  // Butterfly maximum over pairs of lanes 8 apart, as in a fragment row reduction.
  float m = metal::max(f, simd_shuffle_xor(f, ushort(1)));
  m = metal::max(m, simd_shuffle_xor(m, ushort(8)));
  results[6 * index + 0] = simd_shuffle_xor(v, ushort(1));
  results[6 * index + 1] = simd_shuffle_xor(v, ushort(16));
  results[6 * index + 2] = simd_shuffle_up(v, ushort(3));
  results[6 * index + 3] = simd_shuffle_down(v, ushort(5));
  results[6 * index + 4] = int(m * 2.0f);
  results[6 * index + 5] = int(simd_shuffle_up(uint(v + 4096), ushort(31)));
}
"""


def _expected():
    def lane_value(index, source_lane):
        base = index - index % WIDTH
        return VALUES[base + source_lane]

    results = []
    for index in range(THREADS):
        lane = index % WIDTH
        own = VALUES[index]
        x1 = lane_value(index, lane ^ 1)
        f = [value * 0.5 for value in VALUES]
        base = index - lane

        def pair_max(source_lane):
            return max(f[base + source_lane], f[base + (source_lane ^ 1)])

        m = max(pair_max(lane), pair_max(lane ^ 8))
        results.extend(
            [
                x1,
                lane_value(index, lane ^ 16),
                lane_value(index, lane - 3) if lane >= 3 else own,
                lane_value(index, lane + 5) if lane + 5 < WIDTH else own,
                int(m * 2.0),
                lane_value(index, lane - 31) + 4096 if lane >= 31 else own + 4096,
            ]
        )
    return results + GUARD


def _request(root, target):
    source, descriptor, package = _package(
        root, target, "int", WORKGROUP, source=SOURCE
    )
    expected = _expected()
    initial = [-0x21524111] * (len(expected) - len(GUARD)) + GUARD
    inputs = _bound_values(
        descriptor,
        {
            "values": {"dtype": "int32", "shape": [len(VALUES)], "values": VALUES},
            "results": {"dtype": "int32", "shape": [len(initial)], "values": initial},
        },
    )
    outputs = _bound_values(
        descriptor,
        {"results": {"dtype": "int32", "shape": [len(expected)], "values": expected}},
    )
    request = build_native_loader_dispatch_request(
        descriptor,
        package,
        inputs,
        outputs,
        {"workgroupCount": [GROUPS, 1, 1], "workgroupSize": list(WORKGROUP)},
        expected_target=target,
    )
    assert not request.execution_plan.diagnostics
    return source, request, outputs


@pytest.mark.parametrize("target", ("directx", "opengl"))
def test_relative_shuffles_exchange_through_software_scratch(tmp_path, target):
    _, request, _ = _request(tmp_path, target)
    generated = request.artifact_path.read_text()
    # Every relative shuffle reads the shared scratch; none falls back to a native wave intrinsic.
    if target == "directx":
        assert "__crossgl_software_subgroup_shuffle_xor_int" in generated
        assert "__crossgl_software_subgroup_shuffle_xor_float" in generated
        assert "__crossgl_software_subgroup_shuffle_up_uint" in generated
        assert "(lane ^ delta) < 32u" in generated and "delta <= lane" in generated
        assert "WaveReadLaneAt" not in generated and "WaveGetLaneIndex" not in generated
    else:
        assert "crossglSoftwareSubgroupShuffleXor" in generated
        assert "crossglSoftwareSubgroupShuffleUp" in generated
        assert "(lane ^ delta) < 32u" in generated and "delta <= lane" in generated
        assert (
            "subgroupShuffle" not in generated
            and "GL_KHR_shader_subgroup" not in generated
        )
    _compile(generated, target, tmp_path)


@pytest.mark.parametrize("target", ("directx", "opengl"))
def test_partial_subgroup_xor_keeps_the_calling_lane(tmp_path, target):
    source = SOURCE.replace("ushort(16)", "ushort(24)")
    _, descriptor, package = _package(
        tmp_path, target, "int", (48, 1, 1), source=source
    )
    suffix = {"directx": ".hlsl", "opengl": ".glsl"}[target]
    (artifact,) = (tmp_path / "out").rglob(f"*{suffix}")
    generated = artifact.read_text()
    # The second logical subgroup holds 16 lanes, so lane ^ 24 must be checked against the active count.
    assert "activeCount = min(32u, 48u - subgroupBase)" in generated
    assert "(lane ^ delta) < activeCount" in generated
    _compile(generated, target, tmp_path)


def test_relative_shuffles_compile_on_metal(tmp_path):
    _, request, _ = _request(tmp_path, "metal")
    _compile(
        request.artifact_path.read_text(),
        "metal",
        tmp_path,
        metal_compile_flags=("-Wall", "-Wextra"),
    )


def test_relative_shuffles_execute(tmp_path):
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 for required native relative shuffles")
    from tests.test_translator.test_loop_updates import _execute

    target = {"darwin": "metal", "linux": "opengl", "win32": "directx"}[sys.platform]
    source, request, expected = _request(tmp_path, target)
    _execute(
        request,
        expected,
        tmp_path,
        original_source=source,
        original_entry=ENTRY,
        metal_compile_flags=("-Wall", "-Wextra"),
        compare_with_original=target == "metal",
    )


def test_relative_shuffles_are_required_on_all_native_platforms():
    from pathlib import Path

    from tests import workflow_text

    workflow = (
        Path(__file__).resolve().parents[2]
        / ".github/workflows/demo-project-testing.yml"
    ).read_text()
    step = workflow_text.workflow_job_step_section(
        workflow, "portable-host", "Validate collective helper arguments"
    )
    assert f'{REQUIRE_ENV}: "1"' in step
    assert "test_software_subgroup_relative_shuffles.py" in step
    assert "if:" not in step and "continue-on-error" not in step

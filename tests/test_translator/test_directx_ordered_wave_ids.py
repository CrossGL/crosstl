"""Physical DirectX waves are numbered by the first invocation each one holds.

Metal, Vulkan and OpenGL number subgroups by thread position, so subgroup 0
holds invocation 0, and kernels pair ``simdgroup_index_in_threadgroup == 0``
with ``thread_index_in_threadgroup == 0``. Direct3D numbers no waves. The
DirectX lowering ranks each wave by the smallest ``SV_GroupIndex`` it holds,
so the wave holding invocation 0 is subgroup 0 whatever order the waves run
in (CrossGL/crosstl#2120).
"""

import os
import re
import sys
from pathlib import Path

import pytest

from crosstl.project import (
    ProjectConfig,
    build_native_loader_dispatch_request,
    translate_project,
)
from tests.runtime_helpers import _prepare_native_package
from tests.test_translator.test_loop_updates import _execute
from tests.test_translator.test_metal_builtin_ownership import _compile

REQUIRE_ENV = "CROSTL_REQUIRE_ORDERED_WAVE_IDS"
# Thirteen 32-wide subgroups, and more waves wherever waves are narrower.
WORKGROUP = (416, 1, 1)
GROUPS = 4
SENTINEL = 0xDEADBEEF
FIRST_WAVE = """#include <metal_stdlib>
using namespace metal;

kernel void first_wave(device uint* out [[buffer(0)]],
                       uint group [[threadgroup_position_in_grid]],
                       uint lid [[thread_index_in_threadgroup]],
                       uint sg [[simdgroup_index_in_threadgroup]]) {
  if (lid == 0) {
    out[2 * group + 1] = sg;
  }
  if (sg != 0) {
    return;
  }
  if (lid == 0) {
    out[2 * group] = 1u;
  }
}
"""


def _report(root, target, workgroup=WORKGROUP):
    root.mkdir(parents=True, exist_ok=True)
    (root / "kernel.metal").write_text(FIRST_WAVE, encoding="utf-8")
    return translate_project(
        ProjectConfig(
            root=root,
            include_patterns=("kernel.metal",),
            targets=(target,),
            output_dir="out",
            workgroup_size=workgroup,
        ),
        format_output=False,
    )


def _artifact(report):
    data = report.to_json()
    assert [item["code"] for item in data["diagnostics"]] == []
    (artifact,) = (
        artifact for artifact in data["artifacts"] if artifact["target"] == "directx"
    )
    return (Path(data["project"]["root"]) / artifact["path"]).read_text()


def test_directx_ranks_waves_by_their_first_invocation(tmp_path):
    generated = _artifact(_report(tmp_path / "project", "directx"))

    helper = re.search(
        r"uint __crossgl_physical_subgroup_id\(uint groupIndex\)\s*\{(.*?)\n\}",
        generated,
        re.S,
    )
    assert helper is not None
    body = helper.group(1)
    assert "groupshared uint __crossgl_physical_subgroup_leaders[32];" in generated
    # Each wave publishes the smallest index it holds and counts the waves
    # whose smallest index is smaller; nothing depends on arrival order.
    assert "uint leader = WaveActiveMin(groupIndex);" in body
    assert (
        "InterlockedOr(__crossgl_physical_subgroup_leaders[leaderWord], leaderBit);"
        in body
    )
    assert "countbits(__crossgl_physical_subgroup_leaders[leaderWord] & " in body
    assert "InterlockedAdd" not in generated
    assert "WaveReadLaneFirst" not in body
    assert body.count("GroupMemoryBarrierWithGroupSync();") == 2
    assert "uint crossglPhysicalSubgroupID = " in generated
    _compile(generated, "directx", tmp_path)


@pytest.mark.parametrize("workgroup", [(1, 1, 1), (7, 1, 1), (32, 32, 1)])
def test_leader_words_cover_every_workgroup_size(tmp_path, workgroup):
    # The words a wave reads are below its leader's word, and the leader is
    # an invocation index, so the first min(size, 32) invocations clear them.
    generated = _artifact(_report(tmp_path / "project", "directx", workgroup))
    assert "if (groupIndex < 32u)" in generated
    assert f"[numthreads({', '.join(map(str, workgroup))})]" in generated
    _compile(generated, "directx", tmp_path)


def test_first_wave_owns_invocation_zero_executes(tmp_path):
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 for required ordered wave execution")
    target = {"darwin": "metal", "win32": "directx"}.get(sys.platform)
    if target is None:
        pytest.skip("native DirectX and Metal execution run on Windows and macOS")
    report = _report(tmp_path / "project", target)
    assert [item["code"] for item in report.to_json()["diagnostics"]] == []
    descriptor, package = _prepare_native_package(report, tmp_path)
    (binding,) = descriptor["bindings"]

    def words(values):
        return {"dtype": "uint32", "shape": [len(values)], "values": values}

    # Every group's first invocation is in subgroup 0, and subgroup 0 writes.
    outputs = {binding["name"]: words([1, 0] * GROUPS)}
    inputs = {binding["name"]: words([SENTINEL] * (2 * GROUPS))}
    request = build_native_loader_dispatch_request(
        descriptor,
        package,
        inputs,
        outputs,
        {"workgroupCount": [GROUPS, 1, 1], "workgroupSize": list(WORKGROUP)},
        expected_target=target,
    )
    assert not request.execution_plan.diagnostics
    _execute(
        request,
        outputs,
        tmp_path,
        original_source=FIRST_WAVE if target == "metal" else None,
        original_entry="first_wave",
    )


def test_ordered_wave_execution_is_required_in_existing_ci_job():
    from tests import workflow_text

    workflow = (
        Path(__file__).resolve().parents[2]
        / ".github/workflows/demo-project-testing.yml"
    ).read_text()
    step = workflow_text.workflow_step_section(
        workflow, "Validate collective helper arguments"
    )
    assert f'{REQUIRE_ENV}: "1"' in step
    assert "tests/test_translator/test_directx_ordered_wave_ids.py" in step
    assert "continue-on-error" not in step

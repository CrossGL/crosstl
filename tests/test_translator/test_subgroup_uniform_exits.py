"""Subgroup-uniform early exits keep reductions exact on software subgroups.

A kernel whose SIMD groups each own a tile of output rows may leave before
its reductions when its tile starts past the output, branch on the output
size and reduce only the rows inside it::

    if (first >= rows) return;
    if (rows < 8) {
        for (int row = 0; row < 4 && first + row < rows; row++) {
            totals[row] = simd_sum(totals[row]);
            ...
        }
    } else { ... }

Every condition is uniform within a SIMD group, so each software subgroup
runs the converged reductions with all of its lanes or publishes nothing.
"""

import math
import os
import sys

import pytest

from crosstl.project import (
    ProjectConfig,
    build_native_loader_dispatch_request,
    translate_project,
)
from tests.test_translator.test_boolean_buffer_runtime import _bound_values
from tests.test_translator.test_loop_updates import _execute
from tests.test_translator.test_metal_builtin_ownership import _compile
from tests.test_translator.test_software_subgroup_product import _package

REQUIRE_ENV = "CROSTL_REQUIRE_SUBGROUP_UNIFORM_EXITS"
TILE = 4
WORKGROUP = (32, 2, 1)
WIDTH = 64

# The exit, the size branch and the bounded reduction loop; each variant
# below makes one of them, or the helper's subgroup argument, vary by lane.
EXIT = "first >= rows"
BRANCH = "rows < 8"
BOUND = "first + row < rows"
SUBGROUP_ARGUMENT = "simd_index"


def _source(exit=EXIT, branch=BRANCH, bound=BOUND, simd_index=SUBGROUP_ARGUMENT):
    return f"""#include <metal_stdlib>
using namespace metal;

void accumulate_rows(
    const device uint* inputs,
    device uint* outputs,
    device uint* checks,
    const constant int& width,
    const constant int& rows,
    uint3 group,
    uint simd_index,
    uint lane) {{
  uint totals[{TILE}] = {{lane, lane, lane, lane}};
  const int first = int(group.x) * 8 + int(simd_index) * {TILE};
  const int shifted = min(rows - {TILE}, first);
  const uint check = (group.x * 2u + simd_index) * 32u + lane;
  if ({exit}) {{
    return;
  }}
  if ({branch}) {{
    inputs += lane;
    outputs += first;
    for (int k = 0; k < width; k += 32) {{
      for (int row = 0; row < {TILE} && first + row < rows; row++) {{
        totals[row] += inputs[(first + row) * width + k];
      }}
    }}
    for (int row = 0; row < {TILE} && {bound}; row++) {{
      totals[row] = simd_sum(totals[row]);
      if (lane == 0) {{
        outputs[row] = totals[row];
      }}
    }}
  }} else {{
    inputs += lane;
    outputs += shifted;
    for (int k = 0; k < width; k += 32) {{
      for (int row = 0; row < {TILE}; row++) {{
        totals[row] += inputs[(shifted + row) * width + k];
      }}
    }}
    for (int row = 0; row < {TILE}; row++) {{
      totals[row] = simd_sum(totals[row]);
      if (lane == 0) {{
        outputs[row] = totals[row];
      }}
    }}
  }}
  checks[check] = totals[{TILE - 1}];
}}

kernel void row_sums(
    const device uint* inputs [[buffer(0)]],
    device uint* outputs [[buffer(1)]],
    device uint* checks [[buffer(2)]],
    const constant int& width [[buffer(3)]],
    const constant int& rows [[buffer(4)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint simd_index [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {{
  accumulate_rows(inputs, outputs, checks, width, rows, group, {simd_index}, lane);
}}
"""


def _translate(root, target, source):
    (root / "rows.metal").write_text(source, encoding="utf-8")
    options = {"software_subgroup_width": 32}
    if target == "directx":
        options["relative_wave_shuffle_out_of_range"] = "self"
    report = translate_project(
        ProjectConfig(
            root=root,
            include_patterns=("rows.metal",),
            targets=(target,),
            output_dir="out",
            workgroup_size=WORKGROUP,
            source_options={"metal": {"target_options": {target: options}}},
        ),
        format_output=False,
    ).to_json()
    return report


@pytest.mark.parametrize("target", ["metal", "directx", "opengl"])
def test_uniform_exits_translate_and_compile(tmp_path, target):
    _, request, _ = _request(tmp_path, target, 13)
    generated = request.artifact_path.read_text()
    if target != "metal":
        assert "simd_sum" not in generated and "WaveActiveSum" not in generated
        # The bounded loop runs every trip; its running predicate and the
        # guarded update keep rows past the output untouched.
        assert "crossgl_subgroup_running" in generated
        assert "crossgl_subgroup_value" in generated
    _compile(generated, target, tmp_path)


@pytest.mark.parametrize("target", ["directx", "opengl"])
@pytest.mark.parametrize(
    "variant",
    [
        {"exit": "first + int(lane) >= rows + 31"},
        {"bound": "first + row + int(lane) < rows + 31"},
        {"branch": "rows < 8 + int(lane)"},
        {"simd_index": "lane / 16u"},
    ],
    ids=["exit", "bound", "branch", "helper-argument"],
)
def test_lane_varying_predicates_stay_diagnostic(tmp_path, target, variant):
    report = _translate(tmp_path, target, _source(**variant))
    assert report["summary"]["failedCount"] == 1, report["summary"]
    codes = {item["code"] for item in report["diagnostics"]}
    assert codes & {
        "project.translate.directx-software-subgroup-invalid",
        "project.translate.opengl-software-subgroup-invalid",
    }, codes


def _reference(rows):
    """Outputs and per-invocation checks of ``row_sums`` for ``rows`` rows."""
    words = [(index * 2654435761 >> 7) % 1009 for index in range(rows * WIDTH)]
    groups = math.ceil(rows / 8)
    outputs = [None] * rows
    checks = [None] * (groups * 64)
    for group in range(groups):
        for subgroup in range(2):
            first = group * 8 + subgroup * TILE
            if first >= rows:
                continue
            base = first if rows < 8 else min(rows - TILE, first)
            totals = []
            for row in range(TILE):
                inside = base + row < rows
                totals.append(
                    sum(
                        lane
                        + sum(
                            words[(base + row) * WIDTH + k + lane]
                            for k in range(0, WIDTH, 32)
                        )
                        for lane in range(32)
                    )
                    if inside
                    else None
                )
                if inside:
                    outputs[base + row] = totals[row]
            for lane in range(32):
                last = totals[TILE - 1]
                # Rows past the output keep each lane's own initial total.
                checks[(group * 2 + subgroup) * 32 + lane] = (
                    lane if last is None else last
                )
    assert None not in outputs
    return words, outputs, checks


def test_reference_covers_partial_tiles_exits_and_shifted_tiles():
    _, _, checks = _reference(3)
    assert checks[:32] == list(range(32)) and checks[32:] == [None] * 32
    _, _, checks = _reference(12)
    assert checks[96:] == [None] * 32 and None not in checks[:96]
    _, outputs, _ = _reference(13)
    assert len(outputs) == 13


SENTINEL = 0xDEADBEEF


def _request(root, target, rows):
    source = _source()
    source_text, descriptor, package = _package(
        root, target, "uint", WORKGROUP, source=source
    )
    words, outputs, checks = _reference(rows)
    guards = [SENTINEL] * 32
    wanted_checks = [SENTINEL if value is None else value for value in checks]

    def payload(data):
        return dict(dtype="uint32", shape=[len(data)], values=data)

    inputs = _bound_values(
        descriptor,
        {
            "inputs": payload(words),
            "outputs": payload([SENTINEL] * (rows + len(guards))),
            "checks": payload([SENTINEL] * (len(checks) + len(guards))),
        },
    )
    for suffix, value in (("width", WIDTH), ("rows", rows)):
        # DirectX prefixes constant members with the entry name.
        (name,) = (
            binding["name"]
            for binding in descriptor["bindings"]
            if binding["scalarLayout"]
            .get("memberName", binding["name"])
            .endswith(suffix)
        )
        inputs[name] = dict(dtype="int32", shape=[1], values=[value])
    expected = _bound_values(
        descriptor,
        {
            "outputs": payload(outputs + guards),
            "checks": payload(wanted_checks + guards),
        },
    )
    request = build_native_loader_dispatch_request(
        descriptor,
        package,
        inputs,
        expected,
        dict(workgroupCount=[math.ceil(rows / 8), 1, 1], workgroupSize=list(WORKGROUP)),
        observe_inputs=[item for item in inputs if item not in expected],
        expected_target=target,
    )
    assert not request.execution_plan.diagnostics
    return source_text, request, expected


@pytest.mark.parametrize("rows", [3, 6, 12, 13])
def test_uniform_exits_execute(tmp_path, rows):
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 for required subgroup-uniform exits")
    target = {"darwin": "metal", "win32": "directx"}.get(sys.platform, "opengl")
    source, request, expected = _request(tmp_path, target, rows)
    _execute(
        request,
        expected,
        tmp_path,
        original_source=source if target == "metal" else None,
        original_entry="row_sums",
    )


def test_uniform_exit_gate_is_required_on_every_native_target():
    from pathlib import Path

    from tests import workflow_text
    from tests.ci_helpers import assert_paths_covered

    root = Path(__file__).resolve().parents[2]
    workflow = (root / ".github/workflows/demo-project-testing.yml").read_text()
    step = workflow_text.workflow_step_section(
        workflow, "Validate subgroup-guarded returns"
    )
    assert "test_subgroup_uniform_exits.py" in step
    assert f'{REQUIRE_ENV}: "1"' in step
    assert "continue-on-error" not in step and "if:" not in step
    for event in ("pull_request", "push"):
        assert_paths_covered(
            workflow_text.workflow_event_path_filters(workflow, event),
            "tests/test_translator/test_subgroup_uniform_exits.py",
        )

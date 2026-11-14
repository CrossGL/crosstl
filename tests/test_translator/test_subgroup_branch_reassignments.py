"""Loop bounds clamped in uniform branches stay uniform for software subgroups.

``int limit = rows; if (rows > 2) { int reach = (group + 1) * 3; limit = min(rows, reach); }``
writes ``limit`` only outside loops, from values every invocation of a workgroup shares. A loop that
calls a subgroup helper up to ``limit`` is therefore uniform. MLX's causal steel attention bounds its
key-block loop this way, and clamps the first causal block with ``q_min = max(0, q_min)``.

DirectX followed such writes but not a local declared inside the branch, such as ``reach``, so it refused
the loop as potentially divergent. OpenGL did not follow reassigned locals at all when it proved helper
arguments, so it refused the helper call. Both now follow each plain write outside loops, scoping branch
declarations, and still refuse a write of a divergent value, a divergent branch or a write in a loop.
"""

import os
import sys

import pytest

from crosstl.project import (
    ProjectConfig,
    build_native_loader_dispatch_request,
    translate_project,
)
from tests.test_translator.test_boolean_buffer_runtime import _bound_values
from tests.test_translator.test_metal_builtin_ownership import _compile
from tests.test_translator.test_software_subgroup_product import _package

REQUIRE_ENV = "CROSTL_REQUIRE_SUBGROUP_BRANCH_REASSIGNMENTS"
ENTRY = "branch_bounds"
WIDTH = 32
GROUPS = 3
ROWS = 8
GUARD = [0x6D2B79F5, -7, 0x1234567, -0x5A5A5A5]
VALUES = [
    ((index * 2654435761) % 4001) - 2000 for index in range(GROUPS * WIDTH * ROWS)
]

CLAMP = """  if (rows > 2) {
    int reach = (int(group) + 1) * 3;
    limit = min(rows, reach);
  }"""

# The helper avoids HLSL intrinsic names: DXC binds a call to an intrinsic such as
# ``step(int, int)`` even when the source declares that exact overload (#2109).
TEMPLATE = """#include <metal_stdlib>
using namespace metal;

%(constants)s
static inline int shifted(int value, int offset) {
  return simd_shuffle_down(value + offset, ushort(1));
}

[[kernel]] void branch_bounds(
    const device int* values [[buffer(0)]],
    device int* results [[buffer(1)]],
    constant int& rows [[buffer(2)]],
    uint lid [[thread_index_in_threadgroup]],
    uint group [[threadgroup_position_in_grid]]) {
  int limit = rows;
%(clamp)s
  int first = int(group);
  first = max(0, first - 1);
  int total = 0;
  for (int i = first; i < limit; i++) {
    total += shifted(values[(group * 32 + lid) * 8 + uint(i)], i);
  }
  results[group * 32 + lid] = total;
}
"""


def _source(clamp=CLAMP, constants=""):
    return TEMPLATE % {"clamp": clamp, "constants": constants}


SOURCE = _source()


def _bounds(group):
    limit = min(ROWS, (group + 1) * 3)
    return max(0, group - 1), limit


def _expected():
    results = []
    for group in range(GROUPS):
        first, limit = _bounds(group)
        for lane in range(WIDTH):
            # Lane 31 has no lane below it, so shuffle-down keeps its own value.
            source = lane + 1 if lane + 1 < WIDTH else lane
            row = (group * WIDTH + source) * ROWS
            results.append(sum(VALUES[row + i] + i for i in range(first, limit)))
    return results + GUARD


def _request(root, target):
    source, descriptor, package = _package(
        root, target, "int", (WIDTH, 1, 1), source=SOURCE
    )
    expected = _expected()
    initial = [-0x21524111] * (len(expected) - len(GUARD)) + GUARD

    def payload(data):
        return {"dtype": "int32", "shape": [len(data)], "values": data}

    inputs = _bound_values(
        descriptor, {"values": payload(VALUES), "results": payload(initial)}
    )
    # DirectX prefixes constant members with the entry name.
    (name,) = (
        binding["name"]
        for binding in descriptor["bindings"]
        if binding["scalarLayout"].get("memberName", binding["name"]).endswith("rows")
    )
    inputs[name] = payload([ROWS])
    outputs = _bound_values(descriptor, {"results": payload(expected)})
    request = build_native_loader_dispatch_request(
        descriptor,
        package,
        inputs,
        outputs,
        {"workgroupCount": [GROUPS, 1, 1], "workgroupSize": [WIDTH, 1, 1]},
        observe_inputs=[item for item in inputs if item not in outputs],
        expected_target=target,
    )
    assert not request.execution_plan.diagnostics
    return source, request, outputs


def _translate(root, target, source, constants=None):
    (root / "bounds.metal").write_text(source, encoding="utf-8")
    options = {"software_subgroup_width": WIDTH}
    if target == "directx":
        options["relative_wave_shuffle_out_of_range"] = "self"
    report = translate_project(
        ProjectConfig(
            root=root,
            include_patterns=("bounds.metal",),
            targets=(target,),
            output_dir="out",
            workgroup_size=(WIDTH, 1, 1),
            specialization_constants=constants or {},
            freeze_specialization_constants=bool(constants),
            source_options={"metal": {"target_options": {target: options}}},
        ),
        format_output=False,
    ).to_json()
    errors = [item for item in report["diagnostics"] if item["severity"] == "error"]
    return report, errors


def test_reference_covers_clamped_and_shifted_bounds():
    # The first group's bound is clamped below ``rows``, the last is not, and
    # the clamped start moves only for later groups.
    assert [_bounds(group) for group in range(GROUPS)] == [(0, 3), (0, 6), (1, 8)]


@pytest.mark.parametrize("target", ("directx", "opengl"))
def test_branch_clamped_bounds_compile(tmp_path, target):
    _, request, _ = _request(tmp_path, target)
    generated = request.artifact_path.read_text()
    marker = (
        "__crossgl_software_subgroup_shuffle_down_int"
        if target == "directx"
        else "crossglSoftwareSubgroupShuffleDownInt"
    )
    assert marker in generated
    _compile(generated, target, tmp_path)


def test_branch_clamped_bounds_compile_on_metal(tmp_path):
    _, request, _ = _request(tmp_path, "metal")
    _compile(
        request.artifact_path.read_text(),
        "metal",
        tmp_path,
        metal_compile_flags=("-Wall", "-Wextra"),
    )


@pytest.mark.parametrize("target", ("directx", "opengl"))
def test_constant_guard_selects_the_clamping_branch(tmp_path, target):
    source = _source(
        clamp=CLAMP.replace("if (rows > 2)", "if (clamped)"),
        constants="constant bool clamped [[function_constant(0)]];\n",
    )
    report, errors = _translate(tmp_path, target, source, {"clamped": True})
    assert not errors, errors
    assert report["summary"]["translatedCount"] == 1


@pytest.mark.parametrize("target", ("directx", "opengl"))
@pytest.mark.parametrize(
    "clamp",
    (
        # The branch declares its bound from the invocation's own index.
        CLAMP.replace("(int(group) + 1) * 3", "int(lid) + 1"),
        # Invocations of one subgroup may take different branches.
        CLAMP.replace("rows > 2", "int(lid) > 2"),
        # A write inside a loop is not followed.
        """  for (int j = 0; j < 2; j++) {
    limit = rows - j;
  }""",
    ),
    ids=("divergent-value", "divergent-branch", "loop-write"),
)
def test_divergent_bounds_are_still_refused(tmp_path, target, clamp):
    report, errors = _translate(tmp_path, target, _source(clamp=clamp))
    assert report["summary"]["translatedCount"] == 0
    assert [item["code"] for item in errors][:1] == [
        f"project.translate.{target}-software-subgroup-invalid"
    ], errors


def test_branch_clamped_bounds_execute(tmp_path):
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 for required native branch reassignments")
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


def test_branch_reassignments_are_required_on_all_native_platforms():
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
    assert "test_subgroup_branch_reassignments.py" in step
    assert "if:" not in step and "continue-on-error" not in step

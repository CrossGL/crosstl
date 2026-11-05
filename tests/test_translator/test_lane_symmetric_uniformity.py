"""Lane-symmetric values and the workgroup-uniform collectives they feed.

With a workgroup of whole software subgroups, every subgroup computes the same
reduction of a lane-symmetric value, so a loop or helper that leaves on that
result leaves in every invocation together (CrossGL/crosstl#2546).
"""

import os
import sys
from itertools import accumulate
from pathlib import Path

import pytest

from crosstl.project import (
    ProjectConfig,
    build_native_loader_dispatch_request,
    translate_project,
)
from crosstl.translator import parse
from crosstl.translator.ast import (
    BinaryOpNode,
    CastNode,
    FunctionCallNode,
    IdentifierNode,
    IfNode,
    LiteralNode,
    PrimitiveType,
    ReturnNode,
)
from crosstl.translator.codegen.directx_codegen import DirectXSoftwareSubgroupError
from crosstl.translator.codegen.lane_symmetry import (
    SYMMETRIC,
    UNIFORM,
    VARYING,
    _may_fall_through,
)
from tests.runtime_helpers import _prepare_native_package
from tests.test_translator.test_directx_software_reductions import _codegen, _source
from tests.test_translator.test_loop_updates import _execute
from tests.test_translator.test_metal_builtin_ownership import _compile

REQUIRE_ENV = "CROSTL_REQUIRE_LANE_SYMMETRIC_UNIFORMITY"
TILE_ROWS = 16
SCHEDULE = """#include <metal_stdlib>
using namespace metal;

template <int BM>
METAL_FUNC bool schedule_row_tile(const device int* offsets, const int num_groups,
    const int M, const int tile, const uint simd_lane_id, thread int& row,
    thread int& group, thread short& rows) {
  int tiles_before = 0;
  for (int e = 0; e < num_groups; e += 32) {
    const int g = e + simd_lane_id;
    int start = M;
    int end = M;
    if (g < num_groups) {
      start = offsets[g];
      end = g + 1 < num_groups ? offsets[g + 1] : M;
    }
    const int n = (end - start + BM - 1) / BM;
    const int tiles_through = tiles_before + simd_prefix_inclusive_sum(n);
    const ushort owner_lane = OWNER;
    if (owner_lane < 32) {
      group = e + owner_lane;
      row = simd_shuffle(start, owner_lane) +
          (tile - simd_shuffle(tiles_through - n, owner_lane)) * BM;
      rows = short(min(BM, simd_shuffle(end, owner_lane) - row));
      return true;
    }
    tiles_before = simd_shuffle(tiles_through, 31);
  }
  return false;
}

[[kernel]] void tiles(
    const device int* offsets [[buffer(0)]],
    device int* out [[buffer(1)]],
    const constant int& M [[buffer(2)]],
    const constant int& E [[buffer(3)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]]) {
  int row;
  int group;
  short rows;
  if (!schedule_row_tile<16>(offsets, E, M, tid.y, simd_lane_id, row, group, rows)) {
    return;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  int total = 0;
  CONSUMER
  out[tid.y * 64 + lid] = row * 1000 + group * 10 + rows + total;
}
"""


OWNER = "ushort(simd_sum(int(tiles_through <= tile)))"
# Collectives in a loop the helper's reference results bound: every
# invocation must receive the same row count. total ends as zero.
ROWS_LOOP = "for (int r = 0; r < rows; r++) { total += simd_sum(1) - 32; }"


def _kernel(owner=OWNER, consumer=""):
    return SCHEDULE.replace("OWNER", owner).replace("CONSUMER", consumer)


def _report(
    root,
    target,
    source,
    *,
    workgroup=(32, 2, 1),
    immutable=True,
    bounded=True,
):
    root.mkdir(parents=True, exist_ok=True)
    (root / "kernel.metal").write_text(source, encoding="utf-8")
    return translate_project(
        ProjectConfig(
            root=root,
            include_patterns=("kernel.metal",),
            targets=(target,),
            output_dir="out",
            workgroup_size=workgroup,
            immutable_storage=(
                [{"entryPoint": "tiles", "parameter": "offsets"}] if immutable else []
            ),
            metadata_preconditions=(
                [
                    {
                        "source": "kernel.metal",
                        "entryPoint": "tiles",
                        "parameter": "E",
                        "minimum": 1,
                        "maximum": 4096,
                    }
                ]
                if bounded
                else []
            ),
            source_options={
                "metal": {
                    "preserve_resource_origins": True,
                    "target_options": {
                        "directx": {
                            "software_subgroup_width": 32,
                            "relative_wave_shuffle_out_of_range": "self",
                        },
                        "opengl": {"software_subgroup_width": 32},
                    },
                }
            },
        ),
        format_output=False,
    )


def _codes(report):
    return [item["code"] for item in report.to_json()["diagnostics"]]


def _reasons(report):
    return [
        item.get("details", {}).get("executionSpecialization", {}).get("reason")
        for item in report.to_json()["diagnostics"]
    ]


def test_directx_schedules_tiles_from_lane_symmetric_reductions(tmp_path):
    report = _report(tmp_path, "directx", _kernel())
    assert _codes(report) == [], report.to_json()["diagnostics"]
    (artifact,) = report.to_json()["artifacts"]
    generated = (tmp_path / artifact["path"]).read_text()
    # The helper keeps its software-subgroup reduction, shuffles and early
    # return; the entry leaves when it returns false.
    assert "bool schedule_row_tile_16(" in generated
    assert "__crossgl_software_subgroup_sum_int" in generated
    assert "__crossgl_software_subgroup_read_lane_at_int" in generated
    _compile(generated, "directx", tmp_path)


@pytest.mark.parametrize(
    ("owner", "options"),
    [
        # A partial subgroup sums fewer lanes than a full one.
        (OWNER, {"workgroup": (48, 1, 1)}),
        # The thread index differs between subgroups at the same lane.
        ("ushort(simd_sum(int(lid < 40u)))", {}),
        # A broadcast from a lane-dependent lane only stays lane-symmetric.
        ("ushort(simd_shuffle(int(tiles_through <= tile), simd_lane_id))", {}),
        # Loads from storage that is not declared immutable vary.
        (OWNER, {"immutable": False}),
    ],
)
def test_directx_refuses_exits_on_values_that_are_not_uniform(tmp_path, owner, options):
    report = _report(tmp_path, "directx", _kernel(owner), **options)
    assert _codes(report) == ["project.translate.directx-software-subgroup-invalid"]


def test_directx_receives_uniform_results_through_reference_parameters(tmp_path):
    # rows comes from a broadcast at owner_lane, a uniform unsigned local the
    # helper's condition bounds below the subgroup width.
    report = _report(tmp_path, "directx", _kernel(consumer=ROWS_LOOP))
    assert _codes(report) == [], report.to_json()["diagnostics"]
    (artifact,) = report.to_json()["artifacts"]
    _compile((tmp_path / artifact["path"]).read_text(), "directx", tmp_path)


@pytest.mark.parametrize(
    ("owner", "lane_type"),
    [
        # A signed lane could be negative, so its broadcast stays per lane.
        ("int(simd_sum(int(tiles_through <= tile)))", "int"),
        # Without a full-subgroup sum the lane is lane-symmetric only.
        ("ushort(simd_shuffle(int(tiles_through <= tile), simd_lane_id))", "ushort"),
    ],
)
def test_directx_refuses_reference_results_from_unproven_lanes(
    tmp_path, owner, lane_type
):
    source = _kernel(owner, ROWS_LOOP).replace(
        "const ushort owner_lane", f"const {lane_type} owner_lane"
    )
    report = _report(tmp_path, "directx", source)
    assert _codes(report) == ["project.translate.directx-software-subgroup-invalid"]


def test_opengl_still_requires_entry_level_collective_helpers(tmp_path):
    report = _report(tmp_path, "opengl", _kernel())
    assert _codes(report) == ["project.translate.opengl-software-subgroup-invalid"]
    assert _reasons(report) == ["helper-call-not-uniform"]


@pytest.mark.parametrize(
    ("threads", "accepted"), [((32, 4, 1), True), ((48, 1, 1), False)]
)
def test_full_subgroup_reduction_bounds_a_collective_loop(threads, accepted):
    body = (
        "uint stop = WaveActiveSum(1u);"
        "for (uint step = 0u; step < stop; ++step) {"
        " outputWords[index] = WaveActiveSum(index); }"
    )
    source = _source("uint", body).replace(
        "@numthreads(32, 4, 1)", "@numthreads({}, {}, {})".format(*threads)
    )
    if accepted:
        generated = _codegen().generate_stage(parse(source), "compute")
        assert "step < stop" in generated
    else:
        with pytest.raises(DirectXSoftwareSubgroupError):
            _codegen().generate_stage(parse(source), "compute")


def _literal(value):
    return LiteralNode(value, PrimitiveType("int"))


def test_fall_through_is_conservative():
    returned = ReturnNode(_literal(1))
    condition = IdentifierNode("flag")
    assert not _may_fall_through([returned])
    assert _may_fall_through([IfNode(condition, [returned])])
    assert not _may_fall_through([IfNode(condition, [returned], [returned])])
    assert _may_fall_through([])
    assert _may_fall_through(None)


def test_levels_are_ordered():
    assert VARYING < SYMMETRIC < UNIFORM


def test_constant_lanes_fold_only_value_preserving_conversions():
    from crosstl.translator.codegen.lane_symmetry import FunctionSymmetry

    class Rules:
        @staticmethod
        def constant_int(node):
            return node.value if isinstance(node, LiteralNode) else None

        @staticmethod
        def map_operator(operator):
            return operator

    folder = FunctionSymmetry.__new__(FunctionSymmetry)
    folder.rules = Rules()
    cast = FunctionCallNode(IdentifierNode("uint"), [_literal(31)])
    assert folder._constant_lane(BinaryOpNode(cast, "&", _literal(65535))) == 31
    assert folder._constant_lane(CastNode(_literal(7), PrimitiveType("ushort"))) == 7
    negative = FunctionCallNode(IdentifierNode("uint"), [_literal(-1)])
    assert folder._constant_lane(negative) is None
    assert (
        folder._constant_lane(FunctionCallNode(IdentifierNode("f"), [_literal(1)]))
        is None
    )


def _offsets_case():
    # Five groups, one empty and one spanning several 16-row tiles.
    offsets = [0, 5, 5, 40, 41]
    rows = 70
    return offsets, rows


def _expected_tiles(offsets, rows, tiles):
    values = []
    groups = len(offsets)
    for tile in range(tiles):
        result = None
        before = 0
        for first in range(0, groups, 32):
            lanes = []
            for lane in range(32):
                group = first + lane
                start = end = rows
                if group < groups:
                    start = offsets[group]
                    end = offsets[group + 1] if group + 1 < groups else rows
                lanes.append((start, end, (end - start + TILE_ROWS - 1) // TILE_ROWS))
            through = [before + total for total in accumulate(n for *_, n in lanes)]
            owner = sum(value <= tile for value in through)
            if owner < 32:
                start, end, count = lanes[owner]
                row = start + (tile - (through[owner] - count)) * TILE_ROWS
                result = row * 1000 + (first + owner) * 10 + min(TILE_ROWS, end - row)
                break
            before = through[31]
        values.extend([result] * 64)
    return values


def test_expected_tiles_cover_every_branch():
    offsets, rows = _offsets_case()
    values = _expected_tiles(offsets, rows, 8)
    owners = [values[tile * 64] for tile in range(8)]
    # Tile 1 skips the empty group; tile 3 is the last, partial tile of group
    # 2; tile 7 is past the last group's tiles.
    assert owners == [
        0 * 1000 + 0 * 10 + 5,
        5 * 1000 + 2 * 10 + 16,
        21 * 1000 + 2 * 10 + 16,
        37 * 1000 + 2 * 10 + 3,
        40 * 1000 + 3 * 10 + 1,
        41 * 1000 + 4 * 10 + 16,
        57 * 1000 + 4 * 10 + 13,
        None,
    ]
    assert all(
        values[tile * 64 : tile * 64 + 64] == [owners[tile]] * 64 for tile in range(8)
    )


@pytest.mark.parametrize("consumer", ["", ROWS_LOOP])
def test_lane_symmetric_schedule_executes(tmp_path, consumer):
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 for required lane-symmetric execution")
    target = {"darwin": "metal", "win32": "directx"}.get(sys.platform)
    if target is None:
        pytest.skip(
            "OpenGL software subgroups refuse this helper (CrossGL/crosstl#2546)"
        )
    source = _kernel(consumer=consumer)
    report = _report(tmp_path, target, source)
    assert _codes(report) == []
    descriptor, package = _prepare_native_package(report, tmp_path)
    names = {
        binding["provenance"]["sourceResource"]["parameter"]: binding["name"]
        for binding in descriptor["bindings"]
    }
    offsets, rows = _offsets_case()
    tiles = 8
    sentinel = -0x21524111
    expected = [
        sentinel if value is None else value
        for value in _expected_tiles(offsets, rows, tiles)
    ]

    def words(values):
        return {"dtype": "int32", "shape": [len(values)], "values": values}

    outputs = {names["out"]: words(expected)}
    inputs = {
        names["offsets"]: words(offsets),
        names["out"]: words([sentinel] * len(expected)),
        names["M"]: words([rows]),
        names["E"]: words([len(offsets)]),
    }
    request = build_native_loader_dispatch_request(
        descriptor,
        package,
        inputs,
        outputs,
        {"workgroupCount": [1, tiles, 1], "workgroupSize": [32, 2, 1]},
        expected_target=target,
    )
    assert not request.execution_plan.diagnostics
    _execute(
        request,
        outputs,
        tmp_path,
        original_source=source if target == "metal" else None,
        original_entry="tiles",
    )


def test_lane_symmetric_execution_is_required_in_existing_ci_job():
    from tests import workflow_text

    workflow = (
        Path(__file__).resolve().parents[2]
        / ".github/workflows/demo-project-testing.yml"
    ).read_text()
    step = workflow_text.workflow_step_section(
        workflow, "Validate collective helper arguments"
    )
    assert f'{REQUIRE_ENV}: "1"' in step
    assert "tests/test_translator/test_lane_symmetric_uniformity.py" in step
    assert "continue-on-error" not in step

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
    PointerType,
    PrimitiveType,
    ReturnNode,
)
from crosstl.translator.codegen.directx_codegen import DirectXSoftwareSubgroupError
from crosstl.translator.codegen.GLSL_codegen import (
    GLSLCodeGen,
    OpenGLSoftwareSubgroupError,
)
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


HELPER_CALLS = {
    "directx": ("bool schedule_row_tile_16(", "__crossgl_software_subgroup_"),
    "opengl": ("bool schedule_row_tile_16_", "crossglSoftwareSubgroup"),
}
TARGETS = tuple(HELPER_CALLS)


def _refused(report, target):
    return _codes(report) == [f"project.translate.{target}-software-subgroup-invalid"]


@pytest.mark.parametrize("target", TARGETS)
def test_schedules_tiles_from_lane_symmetric_reductions(tmp_path, target):
    report = _report(tmp_path, target, _kernel())
    assert _codes(report) == [], report.to_json()["diagnostics"]
    (artifact,) = report.to_json()["artifacts"]
    generated = (tmp_path / artifact["path"]).read_text()
    # The helper keeps its software-subgroup reduction, shuffles and early
    # return; the entry leaves when it returns false.
    helper, prefix = HELPER_CALLS[target]
    assert helper in generated
    for operation in ("sum_int", "read_lane_at_int"):
        if target == "opengl":
            operation = "".join(part.title() for part in operation.split("_"))
        assert f"{prefix}{operation}" in generated
    _compile(generated, target, tmp_path)


@pytest.mark.parametrize("target", TARGETS)
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
def test_refuses_exits_on_values_that_are_not_uniform(tmp_path, target, owner, options):
    report = _report(tmp_path, target, _kernel(owner), **options)
    assert _refused(report, target), report.to_json()["diagnostics"]


@pytest.mark.parametrize("target", TARGETS)
def test_receives_uniform_results_through_reference_parameters(tmp_path, target):
    # rows comes from a broadcast at owner_lane, a uniform unsigned local the
    # helper's condition bounds below the subgroup width.
    report = _report(tmp_path, target, _kernel(consumer=ROWS_LOOP))
    assert _codes(report) == [], report.to_json()["diagnostics"]
    (artifact,) = report.to_json()["artifacts"]
    _compile((tmp_path / artifact["path"]).read_text(), target, tmp_path)


@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.parametrize(
    ("owner", "lane_type"),
    [
        # A signed lane could be negative, so its broadcast stays per lane.
        ("int(simd_sum(int(tiles_through <= tile)))", "int"),
        # Without a full-subgroup sum the lane is lane-symmetric only.
        ("ushort(simd_shuffle(int(tiles_through <= tile), simd_lane_id))", "ushort"),
    ],
)
def test_refuses_reference_results_from_unproven_lanes(
    tmp_path, target, owner, lane_type
):
    source = _kernel(owner, ROWS_LOOP).replace(
        "const ushort owner_lane", f"const {lane_type} owner_lane"
    )
    report = _report(tmp_path, target, source)
    assert _refused(report, target), report.to_json()["diagnostics"]


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


GLSL_LOOP = """
shader LaneSymmetricLoopExits {
    cbuffer Dispatch @binding(0) {
        uint nRows;
    };

    compute {
        layout(local_size_x = THREADS, local_size_y = 1, local_size_z = 1) in;
        void main() {
            float total = 0.0;
            for (uint row = 0u; row < nRows; row++) {
                total += WaveActiveSum(float(gl_LocalInvocationIndex));
                EXIT
            }
            total += WaveActiveSum(1.0);
        }
    }
}
"""


def _glsl(exit_statement, threads=64):
    source = GLSL_LOOP.replace("THREADS", str(threads)).replace("EXIT", exit_statement)
    return GLSLCodeGen(software_subgroup_width=32).generate(parse(source))


@pytest.mark.parametrize(
    "exit_statement",
    [
        "break;",
        "continue;",
        "return;",
        # A condition every invocation evaluates alike.
        "if (nRows > 4u) { break; }",
        # A full-subgroup reduction of a uniform value.
        "if (WaveActiveSum(1u) > 16u) { break; }",
    ],
)
def test_opengl_loops_keep_exits_every_invocation_takes(tmp_path, exit_statement):
    generated = _glsl(exit_statement)
    assert "crossglSoftwareSubgroupSumFloat(" in generated
    _compile(generated, "opengl", tmp_path)


@pytest.mark.parametrize(
    ("exit_statement", "threads"),
    [
        ("if (gl_LocalInvocationIndex == 0u) { break; }", 64),
        # The subgroups reduce different lanes of the invocation index.
        ("if (WaveActiveSum(gl_LocalInvocationIndex) > 600u) { break; }", 64),
        # A partial subgroup sums fewer lanes than a full one.
        ("if (WaveActiveSum(1u) > 16u) { break; }", 48),
    ],
)
def test_opengl_loops_refuse_exits_that_may_diverge(exit_statement, threads):
    with pytest.raises(OpenGLSoftwareSubgroupError) as raised:
        _glsl(exit_statement, threads)
    assert raised.value.reason == "potentially-divergent-control-flow"


SCHEDULE_CALL = (
    "!schedule_row_tile<16>(offsets, E, M, tid.y, simd_lane_id, row, group, rows)"
)


@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.parametrize(
    "condition",
    [
        # Every invocation evaluates the right operand, or none does: the
        # left operand is workgroup-uniform. This is the shape of MLX's
        # gather_mm_rhs guard.
        f"M <= int(tid.x) || {SCHEDULE_CALL}",
        f"E > 0 && {SCHEDULE_CALL}",
        f"(E > 0 ? {SCHEDULE_CALL} : true)",
    ],
)
def test_helper_calls_under_uniform_short_circuits_are_proven(
    tmp_path, target, condition
):
    report = _report(tmp_path, target, _kernel().replace(SCHEDULE_CALL, condition))
    assert _codes(report) == [], report.to_json()["diagnostics"]
    (artifact,) = report.to_json()["artifacts"]
    _compile((tmp_path / artifact["path"]).read_text(), target, tmp_path)


@pytest.mark.parametrize(
    "condition",
    [
        # The right operand of a short-circuit operator runs only in the
        # invocations its varying left operand selects.
        f"lid > 0u && {SCHEDULE_CALL}",
        f"lid == 0u || {SCHEDULE_CALL}",
        f"(lid > 0u ? {SCHEDULE_CALL} : true)",
        # An else-if condition runs only where a varying first one failed.
        f"lid == 0u) {{ return; }} else if ({SCHEDULE_CALL}",
    ],
)
def test_opengl_refuses_helper_calls_a_condition_may_skip(tmp_path, condition):
    report = _report(tmp_path, "opengl", _kernel().replace(SCHEDULE_CALL, condition))
    assert _refused(report, "opengl"), report.to_json()["diagnostics"]
    assert _reasons(report) == ["helper-call-not-uniform"]


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


def test_local_names_shadow_module_constant_lanes():
    from crosstl.translator.codegen.lane_symmetry import FunctionSymmetry

    class Rules:
        @staticmethod
        def constant_int(node):
            # The module declares a constant LANE = 3.
            return 3 if getattr(node, "name", None) == "LANE" else None

        @staticmethod
        def walk(node):
            return [node]

    folder = FunctionSymmetry.__new__(FunctionSymmetry)
    folder.rules = Rules()
    folder.function = None
    folder.declared = set()
    assert folder._constant_lane(IdentifierNode("LANE")) == 3
    # A local LANE holds whatever the function stores in it.
    folder.declared = {"LANE"}
    assert folder._constant_lane(IdentifierNode("LANE")) is None


def _view_levels(pointer_type):
    """Levels after ``P view = table; int value = view[lane];``."""
    from crosstl.translator.ast import (
        ArrayAccessNode,
        FunctionNode,
        ParameterNode,
        VariableNode,
    )
    from crosstl.translator.codegen.lane_symmetry import (
        LaneSymmetryAnalysis,
        SymmetryRules,
    )

    def walk(node):
        pending = [node]
        while pending:
            current = pending.pop()
            if isinstance(current, (list, tuple)):
                pending.extend(current)
            elif current is not None:
                yield current
                pending.extend(current.child_nodes())

    rules = SymmetryRules(
        walk=walk,
        map_operator=lambda operator: operator,
        collective=lambda node: None,
        call_target=lambda call: None,
        pure_call=lambda call: False,
        call_dependencies=lambda call: None,
        storage_load=lambda node: None,
        constant_int=lambda node: None,
        call_writes=lambda call: set(),
        seeded_parameter=lambda parameter: True,
        volatile=lambda declaration: False,
        storage_parameter=lambda parameter: True,
        private_reference=lambda parameter: False,
        width=32,
        full_subgroups=True,
        reductions=frozenset(),
        broadcasts=frozenset(),
        lane_preserving=frozenset(),
    )
    table_type = PointerType(PrimitiveType("int"), False, "device", "read")
    function = FunctionNode(
        "f",
        PrimitiveType("void"),
        [
            ParameterNode("table", table_type),
            ParameterNode("lane", PrimitiveType("uint")),
        ],
        [
            VariableNode("view", pointer_type, IdentifierNode("table")),
            VariableNode(
                "value",
                PrimitiveType("int"),
                ArrayAccessNode(IdentifierNode("view"), IdentifierNode("lane")),
            ),
        ],
    )
    facts = LaneSymmetryAnalysis(rules).analyze(
        function, {"table": UNIFORM, "lane": SYMMETRIC}, True
    )
    return {name: facts.name_level(name) for name in ("table", "view", "value")}


def test_read_only_storage_views_keep_their_referent_level():
    # A read-only view cannot write the storage it reads, so lowering a
    # parameter to a view, as OpenGL resource references do, keeps loads
    # through it at the parameter's level.
    read_only = PointerType(PrimitiveType("int"), False, "device", "read")
    assert _view_levels(read_only) == {
        "table": UNIFORM,
        "view": UNIFORM,
        "value": SYMMETRIC,
    }
    # A writable view, or one into private memory, still fails closed.
    for writable in (
        PointerType(PrimitiveType("int"), True, "device", "read_write"),
        PointerType(PrimitiveType("int"), False, "thread", "read"),
    ):
        assert _view_levels(writable) == {
            "table": VARYING,
            "view": VARYING,
            "value": VARYING,
        }


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
    target = {"darwin": "metal", "win32": "directx", "linux": "opengl"}[sys.platform]
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

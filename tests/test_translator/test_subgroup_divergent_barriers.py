"""Workgroup barriers in possibly divergent branches run in every invocation."""

import os
import re
import sys

import pytest

from crosstl.project import build_native_loader_dispatch_request
from crosstl.translator import parse
from tests.test_translator.test_loop_updates import _execute
from tests.test_translator.test_metal_builtin_ownership import _compile
from tests.test_translator.test_software_subgroup_product import _package
from tests.test_translator.test_software_subgroup_votes import _codegen
from tests.test_translator.test_subgroup_early_exits import (
    _bits,
    _bound_values,
    _words,
)

REQUIRE_ENV = "CROSTL_REQUIRE_SUBGROUP_DIVERGENT_BARRIER_RUNTIME"
# OpenGL drops the reserved leading underscores.
TAKEN = re.compile(r"\bbool (_*crossgl_subgroup_taken\w*) = \(?(?P<condition>[^;]*);")
BARRIERS = {"directx": "GroupMemoryBarrierWithGroupSync();", "opengl": "barrier();"}


def _source(branch, condition="invocation < limit"):
    return f"""shader DivergentBarriers {{
        RWStructuredBuffer<float> results @register(u0);
        StructuredBuffer<float> inputs @register(t1);
        void columns(uint invocation, uint limit) {{
            float value = 0.0;
            if ({condition}) {{
                {branch}
            }}
            for (uint offset = 2u; offset >= 1u; offset >>= 1u) {{
                value += WaveShuffleDown(value, offset);
            }}
            if (invocation % 4u == 0u) {{
                results[invocation / 4u] = value;
            }}
        }}
        compute {{
            @numthreads(64, 1, 1)
            void main(uint invocation @gl_LocalInvocationIndex) {{
                columns(invocation, 37u);
            }}
        }}
    }}"""


ROUNDS = (
    "for (uint round = 0u; round < 2u; ++round) {"
    " workgroupBarrier(); value += inputs[invocation * 2u + round]; }"
)


def _helper(generated):
    helper = generated[generated.index("void columns(uint invocation, uint limit) {") :]
    return helper[: helper.index("\n}\n")]


@pytest.mark.parametrize("target", ["directx", "opengl"])
def test_divergent_branch_barriers_run_in_every_invocation(tmp_path, target):
    generated = _codegen(target).generate_stage(parse(_source(ROUNDS)), "compute")
    helper = _helper(generated)
    taken = TAKEN.search(helper)
    assert taken is not None and "invocation < limit" in taken.group("condition")
    barrier = helper.index(BARRIERS[target])
    # The rounds and their barrier run everywhere; only the loads are guarded.
    assert (
        helper.index("for (uint round")
        < barrier
        < helper.index(f"if ({taken.group(1)})", barrier)
    )
    assert "if ((invocation < limit))" not in helper
    assert "if (invocation < limit)" not in helper
    _compile(generated, target, tmp_path)


@pytest.mark.parametrize("target", ["directx", "opengl"])
def test_nested_divergent_barriers_converge_through_both_branches(tmp_path, target):
    nested = (
        "value = inputs[invocation]; "
        "if (invocation % 2u == 0u) { workgroupBarrier(); value += 1.0; }"
    )
    generated = _codegen(target).generate_stage(parse(_source(nested)), "compute")
    helper = _helper(generated)
    outer, inner = TAKEN.findall(helper)
    assert "invocation < limit" in outer[1]
    # The inner flag also requires the outer branch.
    assert outer[0] in inner[1] and "invocation % 2u" in inner[1]
    barrier = helper.index(BARRIERS[target])
    assert (
        helper.index(f"bool {inner[0]}")
        < barrier
        < helper.index(f"if ({inner[0]})", barrier)
    )
    _compile(generated, target, tmp_path)


@pytest.mark.parametrize("target", ["directx", "opengl"])
@pytest.mark.parametrize(
    "condition, branch",
    [
        # Every invocation takes the same branch.
        ("limit > 4u", ROUNDS),
        # A branch without barriers needs no convergence.
        ("invocation < limit", "value += inputs[invocation];"),
    ],
)
def test_uniform_or_barrier_free_branches_stay_unchanged(target, condition, branch):
    generated = _codegen(target).generate_stage(
        parse(_source(branch, condition)), "compute"
    )
    assert not TAKEN.search(generated)


@pytest.mark.parametrize("target", ["directx", "opengl"])
@pytest.mark.parametrize(
    "branch",
    [
        # Rounds whose trip count reads memory.
        (
            "for (uint round = 0u; round < uint(inputs[0]); ++round) "
            "{ workgroupBarrier(); value += inputs[round]; }"
        ),
        # A barrier under a nested branch with an alternative.
        (
            "if (invocation % 2u == 0u) { workgroupBarrier(); } "
            "else { value += 1.0; }"
        ),
        # A loop counter its body writes.
        (
            "for (uint round = 0u; round < 2u; ++round) "
            "{ workgroupBarrier(); round = 2u; }"
        ),
    ],
)
def test_unconvergeable_branches_keep_the_source(target, branch):
    generated = _codegen(target).generate_stage(parse(_source(branch)), "compute")
    assert not TAKEN.search(generated)


def _converged(source):
    from crosstl.translator.codegen.subgroup_control_flow import (
        converge_subgroup_divergent_barriers,
    )

    codegen = _codegen("directx")
    ast = parse(source)
    entries = codegen.hlsl_software_subgroup_stage_entries(ast, "compute")
    return ast, converge_subgroup_divergent_barriers(
        ast, entries, codegen.walk_ast, codegen.map_operator
    )


@pytest.mark.parametrize(
    "exit",
    [
        "",
        # A guard that only a mutable local decides may run.
        "bool skip = false; if (limit == 0u) { skip = true; } if (skip) { return; }",
        "if (limit == 0u) { return; }",
    ],
)
def test_branches_that_return_belong_to_early_exit_convergence(exit):
    if exit:
        source = _with_exit(_source(ROUNDS), exit)
    else:
        source = _source(f"{ROUNDS} return;")
    ast, converged = _converged(source)
    assert converged is ast


# Returns that boolean constants rule out, as folded template guards leave
# them; the function never exits, so its divergent barriers still converge.
DEAD_EXITS = [
    "if (false) { results[0] = 0.0; return; }",
    "const bool has_mask = false; if (has_mask) { results[0] = 0.0; return; }",
    "const bool has_mask = true; if (!has_mask) { return; }",
]


def _with_exit(source, exit):
    return source.replace("float value = 0.0;", f"float value = 0.0; {exit}", 1)


@pytest.mark.parametrize("target", ["directx", "opengl"])
@pytest.mark.parametrize("exit", DEAD_EXITS)
def test_returns_that_constants_rule_out_keep_barriers_convergent(
    tmp_path, target, exit
):
    generated = _codegen(target).generate_stage(
        parse(_with_exit(_source(ROUNDS), exit)), "compute"
    )
    helper = _helper(generated)
    taken = TAKEN.search(helper)
    assert taken is not None and "invocation < limit" in taken.group("condition")
    barrier = helper.index(BARRIERS[target])
    assert (
        helper.index("for (uint round")
        < barrier
        < helper.index(f"if ({taken.group(1)})", barrier)
    )
    _compile(generated, target, tmp_path)


# MLX's GEMV-T shape: four lanes per column and eight columns per SIMD group;
# columns past the matrix skip the row loop and its barrier. With the dead
# exit, like GEMV-T's output-mask exit, a template guard rules out the only
# return.
DEAD_TEMPLATE_EXIT = "if (has_mask) { if (blocks == 0) { return; } }"
COLUMNS = 21
ROW_BLOCKS = 3
METAL_SOURCE = """#include <metal_stdlib>
using namespace metal;

template <bool has_mask>
struct Columns {
  static void accumulate(const device float* values, device float* sums,
                         constant int& columns, constant int& blocks,
                         uint3 tid, uint simd_gid, uint simd_lid) {
    const int lane = simd_lid % 4;
    int column = int(tid.x) * 16 + int(simd_gid) * 8 + int(simd_lid / 4);
    float value = 0.0f;
    /*EXIT*/
    if (column < columns) {
      for (int block = 0; block < blocks; ++block) {
        threadgroup_barrier(mem_flags::mem_none);
        value += values[(block * 32 + column) * 4 + lane];
      }
    }
    for (ushort offset = 2; offset >= 1; offset >>= 1) {
      value += simd_shuffle_down(value, offset);
    }
    if (lane == 0 && column < columns) {
      sums[column] = value;
    }
  }
};

kernel void divergent_barrier_columns(
    const device float* values [[buffer(0)]],
    device float* sums [[buffer(1)]],
    constant int& columns [[buffer(2)]],
    constant int& blocks [[buffer(3)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  Columns<false>::accumulate(
      values, sums, columns, blocks, tid, simd_gid, simd_lid);
}
"""


def _request(root, target, exit=""):
    source = METAL_SOURCE.replace("/*EXIT*/", exit)
    original, descriptor, package = _package(
        root, target, "float", (64, 1, 1), source=source
    )
    inputs = [float((index * 5) % 11 - 5) for index in range(ROW_BLOCKS * 32 * 4)]
    marker = _bits(-999.0)
    outputs = [marker] * 32
    expected = list(outputs)
    for column in range(COLUMNS):
        expected[column] = _bits(
            sum(
                inputs[(block * 32 + column) * 4 + lane]
                for block in range(ROW_BLOCKS)
                for lane in range(4)
            )
        )
    bound_expected = _bound_values(descriptor, {"sums": _words(expected)})
    request = build_native_loader_dispatch_request(
        descriptor,
        package,
        _bound_values(
            descriptor,
            {
                "values": _words(_bits(value) for value in inputs),
                "sums": _words(outputs),
                "columns": {"dtype": "int32", "values": [COLUMNS], "shape": [1]},
                "blocks": {"dtype": "int32", "values": [ROW_BLOCKS], "shape": [1]},
            },
        ),
        bound_expected,
        {"workgroupCount": [2, 1, 1], "workgroupSize": [64, 1, 1]},
        expected_target=target,
    )
    assert not request.execution_plan.diagnostics
    return original, request, bound_expected


@pytest.mark.parametrize("exit", ["", DEAD_TEMPLATE_EXIT])
@pytest.mark.parametrize("target", ["directx", "opengl"])
def test_member_divergent_barrier_translates_through_projects(tmp_path, target, exit):
    _, request, _ = _request(tmp_path, target, exit)
    generated = request.artifact_path.read_text(encoding="utf-8")
    assert TAKEN.search(generated)
    _compile(generated, target, tmp_path)


@pytest.mark.parametrize("exit", ["", DEAD_TEMPLATE_EXIT])
def test_member_divergent_barrier_executes_natively(tmp_path, exit):
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 for required divergent-barrier execution")
    target = {"darwin": "metal", "win32": "directx", "linux": "opengl"}[sys.platform]
    original, request, expected = _request(tmp_path, target, exit)
    _execute(
        request,
        expected,
        tmp_path,
        original_source=original,
        original_entry="divergent_barrier_columns",
    )

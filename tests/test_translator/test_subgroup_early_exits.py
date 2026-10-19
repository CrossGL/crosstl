"""Early returns before software subgroup collectives become predicates."""

import os
import re
import struct
import sys

import pytest

from crosstl.project import build_native_loader_dispatch_request
from crosstl.translator import parse
from crosstl.translator.codegen.directx_codegen import DirectXSoftwareSubgroupError
from crosstl.translator.codegen.GLSL_codegen import OpenGLSoftwareSubgroupError
from tests.test_translator.test_loop_updates import _execute
from tests.test_translator.test_metal_builtin_ownership import _compile
from tests.test_translator.test_software_subgroup_product import _package
from tests.test_translator.test_software_subgroup_votes import _codegen

REQUIRE_ENV = "CROSTL_REQUIRE_SUBGROUP_EARLY_EXIT_RUNTIME"
# OpenGL drops the reserved leading underscores.
ACTIVE = re.compile(r"\b_*crossgl_subgroup_active\b")
ERRORS = {
    "directx": DirectXSoftwareSubgroupError,
    "opengl": OpenGLSoftwareSubgroupError,
}


def _source(helper_body, prelude="float value = float(invocation);"):
    return f"""shader EarlyExits {{
        RWStructuredBuffer<float> results @register(u0);
        StructuredBuffer<float> inputs @register(t1);
        void rows(uint invocation, uint limit) {{
            {prelude}
            {helper_body}
        }}
        compute {{
            @numthreads(64, 1, 1)
            void main(uint invocation @gl_LocalInvocationIndex) {{
                rows(invocation, 11u);
            }}
        }}
    }}"""


EXIT = "if (invocation / 4u >= limit) { return; }"
REDUCE = (
    "for (uint offset = 2u; offset >= 1u; offset >>= 1u) {"
    " value += WaveShuffleDown(value, offset); }"
)
STORE = "if (invocation % 4u == 0u) { results[invocation / 4u] = value; }"


@pytest.mark.parametrize("target", ["directx", "opengl"])
def test_early_exit_runs_reductions_in_every_invocation(tmp_path, target):
    generated = _codegen(target).generate_stage(
        parse(_source(f"{EXIT} {REDUCE} {STORE}")), "compute"
    )
    helper = generated[generated.index("void rows(uint invocation, uint limit) {") :]
    helper = helper[: helper.index("\n}\n")]
    declaration = re.search(r"bool (_*crossgl_subgroup_active) = \(?!", helper)
    assert declaration is not None
    # The reduction runs in every invocation; only the store is predicated.
    assert helper.index("offset >>=") < helper.index(f"if ({declaration.group(1)})")
    assert "return;" not in helper
    _compile(generated, target, tmp_path)


@pytest.mark.parametrize("target", ["directx", "opengl"])
def test_early_exit_keeps_uniform_programs_unchanged(target):
    uniform = _source(
        f"if (limit == 0u) {{ return; }} {REDUCE} {STORE}",
    )
    generated = _codegen(target).generate_stage(parse(uniform), "compute")
    assert not ACTIVE.search(generated)


EXIT_FLAG = re.compile(r"\b_*crossgl_subgroup_exit\b")
LATER_EXIT = "if (invocation % 4u == 1u) { results[invocation] = value; return; }"


def _helper(generated):
    helper = generated[generated.index("void rows(uint invocation, uint limit) {") :]
    return helper[: helper.index("\n}\n")]


@pytest.mark.parametrize("target", ["directx", "opengl"])
def test_later_exits_store_then_leave_the_active_set(tmp_path, target):
    """``if (c) { S; return; }`` after a collective runs ``S`` and clears active.

    The invocations taking the exit evaluate ``c`` and run ``S`` where they
    are still active, as in the source; every later statement without a
    collective then skips them. A relative shuffle that reads a lane which
    left is undefined in the source, so the value that lane still holds
    refines it.
    """
    generated = _codegen(target).generate_stage(
        parse(_source(f"{EXIT} {REDUCE} {LATER_EXIT} {REDUCE} {STORE}")), "compute"
    )
    helper = _helper(generated)
    assert "return;" not in helper
    active = ACTIVE.search(helper).group(0)
    taken = EXIT_FLAG.search(helper).group(0)
    assert f"bool {taken} = ({active} && ((invocation % 4u) == 1u));" in helper
    assert re.search(rf"\bif \({taken}\) {{\s+results\[invocation\] = value;", helper)
    assert re.search(rf"\b{active} = \({active} && \(?!{taken}\)?\);", helper)
    # Both reductions run in every invocation; the final store waits.
    first, second = (match.start() for match in re.finditer("offset >>=", helper))
    assert first < helper.index(f"bool {taken}") < second
    assert second < helper.rindex(f"if ({active})")
    _compile(generated, target, tmp_path)


@pytest.mark.parametrize("target", ["directx", "opengl"])
def test_exits_in_literal_branches_follow_their_condition(tmp_path, target):
    """``if (true)`` exits like its body; ``if (false)`` never runs.

    MLX's masked GEMV returns from inside ``if (has_output_mask) { ... }``
    once its template constant is folded. The mask it reads from memory is
    read only where ``active`` holds and is zero in invocations that left.
    """
    body = (
        f"{EXIT} if (false) {{ if (value > 2.0) {{ return; }} }} "
        "if (true) { float scale = inputs[invocation]; "
        "if (scale == 0.0) { results[invocation] = 0.0; return; } "
        f"value *= scale; }} {REDUCE} {STORE}"
    )
    generated = _codegen(target).generate_stage(parse(_source(body)), "compute")
    helper = _helper(generated)
    assert "return;" not in helper and "2.0" not in helper
    active = ACTIVE.search(helper).group(0)
    taken = EXIT_FLAG.search(helper).group(0)
    assert re.search(
        rf"float scale = 0;\s+if \({active}\) {{\s+scale = inputs\[invocation\];",
        helper,
    )
    assert f"bool {taken} = ({active} && (scale == 0.0));" in helper
    assert re.search(rf"if \({active}\) {{\s+value \*= scale;", helper)
    assert helper.index("value *= scale") < helper.index("offset >>=")
    _compile(generated, target, tmp_path)


@pytest.mark.parametrize("target", ["directx", "opengl"])
@pytest.mark.parametrize(
    "declaration",
    [
        "const float scale = inputs[invocation];",
        "bool keep = inputs[invocation] > 0.0;",
    ],
)
def test_predicated_constants_and_flags_start_from_zero(tmp_path, target, declaration):
    """A constant or ``bool`` read from memory becomes a zeroed variable."""
    name = declaration.split("=")[0].split()[-1]
    use = f"value *= {name};" if "scale" in name else f"if (!{name}) {{ value = 0.0; }}"
    body = f"{EXIT} {declaration} {REDUCE} {use} {STORE}"
    generated = _codegen(target).generate_stage(parse(_source(body)), "compute")
    helper = _helper(generated)
    zero = "false" if name == "keep" else "0"
    kind = "bool" if name == "keep" else "float"
    assert re.search(rf"(?<!const ){kind} {name} = {zero};", helper), helper
    _compile(generated, target, tmp_path)


@pytest.mark.parametrize("target", ["directx", "opengl"])
@pytest.mark.parametrize(
    "body",
    [
        # A collective whose result reaches memory in every invocation.
        f"{EXIT} results[invocation] = WaveShuffleDown(value, 1u);",
        # A collective reading memory that inactive invocations may not read.
        f"{EXIT} value += WaveShuffleDown(inputs[invocation], 1u); {STORE}",
        # A loop whose trip count reads memory.
        (
            f"{EXIT} for (uint offset = 0u; offset < uint(inputs[0]); ++offset) "
            f"{{ value += WaveShuffleDown(value, 1u); }} {STORE}"
        ),
        # A later exit whose statements before the return synchronize.
        (
            f"{EXIT} {REDUCE} if (invocation % 4u == 1u) "
            f"{{ value += WaveShuffleDown(value, 1u); return; }} {REDUCE} {STORE}"
        ),
        # A later exit inside a branch on a value that varies.
        (
            f"{EXIT} {REDUCE} if (invocation % 2u == 0u) "
            f"{{ if (value > 0.0) {{ return; }} }} {REDUCE} {STORE}"
        ),
        # A later exit with an alternative.
        (
            f"{EXIT} {REDUCE} if (invocation % 4u == 1u) {{ return; }} "
            f"else {{ value += 1.0; }} {REDUCE} {STORE}"
        ),
        # A later exit that returns from inside a loop.
        (
            f"{EXIT} {REDUCE} for (uint step = 0u; step < 2u; ++step) "
            f"{{ if (invocation % 4u == step) {{ return; }} }} {REDUCE} {STORE}"
        ),
        # A loop counter the body writes.
        (
            f"{EXIT} for (uint offset = 2u; offset >= 1u; offset >>= 1u) "
            f"{{ value += WaveShuffleDown(value, offset); offset = 0u; }} {STORE}"
        ),
        # A collective writing a value that is not private to the helper.
        f"{EXIT} limit += uint(WaveShuffleDown(value, 1u)); {STORE}",
        # Integer division by a value that may be zero or -1.
        f"{EXIT} for (uint offset = 2u / limit; offset >= 1u; offset >>= 1u) "
        f"{{ value += WaveShuffleDown(value, offset); }} {STORE}",
    ],
)
def test_early_exit_rejects_unsafe_inactive_work(target, body):
    with pytest.raises(ERRORS[target]):
        _codegen(target).generate_stage(parse(_source(body)), "compute")


@pytest.mark.parametrize("target", ["directx", "opengl"])
@pytest.mark.parametrize(
    "collective",
    [
        "WaveActiveSum(value)",
        "WavePrefixSum(value)",
        "WaveReadLaneAt(value, 0u)",
        "WaveReadLaneFirst(value)",
        "WaveActiveMax(value)",
    ],
)
def test_early_exit_converges_only_relative_shuffles(target, collective):
    """Lanes that left must not change a sum, scan or broadcast.

    A relative shuffle from a lane that left is undefined in the source, so
    any value refines it; a sum, scan, maximum or read of a fixed lane would
    include the inactive lanes' values, which the source excludes.
    """
    body = f"{EXIT} value += {collective}; {STORE}"
    with pytest.raises(ERRORS[target]):
        _codegen(target).generate_stage(parse(_source(body)), "compute")


# MLX's GEMV shape: four lanes per row and eight rows per SIMD group; rows
# past the matrix leave before their lanes reduce. A row's lanes share their
# exit, so each lane reads only lanes of its own row.
ROWS = 21
METAL_SOURCE = """#include <metal_stdlib>
using namespace metal;

struct Rows {
  static void reduce(const device float* values, device float* sums,
                     constant int& rows, uint3 tid, uint simd_gid,
                     uint simd_lid) {
    const int row_in_group = simd_lid / 4;
    const int lane = simd_lid % 4;
    int row = int(tid.x) * 16 + int(simd_gid) * 8 + row_in_group;
    if (row >= rows) {
      return;
    }
    float value = values[row * 4 + lane];
    for (ushort offset = 2; offset >= 1; offset >>= 1) {
      value += simd_shuffle_down(value, offset);
    }
    if (lane == 0) {
      sums[row] = value;
    }
  }
};

kernel void early_exit_rows(const device float* values [[buffer(0)]],
                            device float* sums [[buffer(1)]],
                            constant int& rows [[buffer(2)]],
                            uint3 tid [[threadgroup_position_in_grid]],
                            uint simd_gid [[simdgroup_index_in_threadgroup]],
                            uint simd_lid [[thread_index_in_simdgroup]]) {
  Rows::reduce(values, sums, rows, tid, simd_gid, simd_lid);
}
"""


def _bits(value):
    return struct.unpack("<I", struct.pack("<f", value))[0]


def _words(values, dtype="float32", encoding=None):
    from crosstl.project.runtime_value_encoding import FLOAT32_BITS

    values = list(values)
    return {
        "dtype": dtype,
        "encoding": encoding or FLOAT32_BITS,
        "values": values,
        "shape": [len(values)],
    }


def _bound_values(descriptor, values):
    """Bind values by source parameter; DirectX prefixes cbuffer members."""
    bound = {}
    for binding in descriptor["bindings"]:
        member = binding["scalarLayout"].get("memberName", binding["name"])
        for name, value in values.items():
            if member == name or member.endswith("_" + name):
                bound[binding["name"]] = value
    assert len(bound) == len(values), (sorted(bound), sorted(values))
    return bound


def _request(root, target):
    original, descriptor, package = _package(
        root, target, "float", (64, 1, 1), source=METAL_SOURCE
    )
    inputs = [float((index * 7) % 13 - 6) for index in range(32 * 4)]
    marker = _bits(-999.0)
    outputs = [marker] * 32
    expected = list(outputs)
    for row in range(ROWS):
        expected[row] = _bits(sum(inputs[row * 4 : row * 4 + 4]))
    bound_expected = _bound_values(descriptor, {"sums": _words(expected)})
    request = build_native_loader_dispatch_request(
        descriptor,
        package,
        _bound_values(
            descriptor,
            {
                "values": _words(_bits(value) for value in inputs),
                "sums": _words(outputs),
                "rows": {"dtype": "int32", "values": [ROWS], "shape": [1]},
            },
        ),
        bound_expected,
        {"workgroupCount": [2, 1, 1], "workgroupSize": [64, 1, 1]},
        expected_target=target,
    )
    assert not request.execution_plan.diagnostics
    return original, request, bound_expected


@pytest.mark.parametrize("target", ["directx", "opengl"])
def test_member_early_exit_translates_through_projects(tmp_path, target):
    _, request, _ = _request(tmp_path, target)
    generated = request.artifact_path.read_text(encoding="utf-8")
    assert ACTIVE.search(generated)
    _compile(generated, target, tmp_path)


def test_member_early_exit_executes_natively(tmp_path):
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 for required early-exit execution")
    target = {"darwin": "metal", "win32": "directx", "linux": "opengl"}[sys.platform]
    original, request, expected = _request(tmp_path, target)
    _execute(
        request,
        expected,
        tmp_path,
        original_source=original,
        original_entry="early_exit_rows",
    )

"""Early exits before halving loops and shuffles of struct values.

A workgroup reduction often folds each subgroup, publishes one value per
subgroup through shared memory, and lets only the first subgroup fold those
values again::

    if (subgroup != 0u) { return; }
    for (uint offset = 16u; offset > 0u; offset /= 2u) {
        pair = reduce(pair, shuffle_down(pair, offset));
    }

Software subgroups need every invocation of the workgroup at the shuffles'
barriers, so the exit is predicated. These tests cover the two forms that
reduction uses: a counter that halves through ``/=``, and a user overload that
shuffles each member of a struct and rebuilds it.
"""

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

REQUIRE_ENV = "CROSTL_REQUIRE_SUBGROUP_EXIT_VALUE_SHUFFLE_RUNTIME"
# OpenGL drops the reserved leading underscores.
ACTIVE = re.compile(r"\b_*crossgl_subgroup_active\b")
ERRORS = {
    "directx": DirectXSoftwareSubgroupError,
    "opengl": OpenGLSoftwareSubgroupError,
}


def _source(body, helpers="", value_type="float"):
    return f"""shader ExitShuffles {{
        struct Pair {{
            uint index;
            {value_type} value;
        }};
        RWStructuredBuffer<float> results @register(u0);
        StructuredBuffer<float> inputs @register(t1);
        {helpers}
        void rows(uint invocation, uint limit) {{
            float value = float(invocation);
            Pair pair = Pair(invocation, {value_type}(value));
            {body}
        }}
        compute {{
            @numthreads(64, 1, 1)
            void main(uint invocation @gl_LocalInvocationIndex) {{
                rows(invocation, 11u);
            }}
        }}
    }}"""


EXIT = "if (invocation / 4u >= limit) { return; }"
STORE = "if (invocation % 4u == 0u) { results[invocation / 4u] = value; }"
PAIR_STORE = (
    "if (invocation % 4u == 0u) "
    "{ results[invocation / 4u] = float(pair.value) + float(pair.index); }"
)
HALVING = (
    "for (uint offset = 2u; offset > 0u; offset /= 2u) "
    "{ value += WaveShuffleDown(value, offset); }"
)
PAIR_SHUFFLE = """Pair shuffle_pair(Pair pair, uint delta) {
            return Pair(
                WaveShuffleDown(pair.index, delta),
                WaveShuffleDown(pair.value, delta)
            );
        }"""
PAIR_REDUCE = (
    "for (uint offset = 2u; offset >= 1u; offset >>= 1u) {"
    " Pair other = shuffle_pair(pair, offset);"
    " pair = Pair(pair.index + other.index, pair.value + other.value); }"
)


def _helper(generated):
    helper = generated[generated.index("void rows(uint invocation, uint limit) {") :]
    return helper[: helper.index("\n}\n")]


@pytest.mark.parametrize("target", ["directx", "opengl"])
def test_halving_loop_runs_in_every_invocation(tmp_path, target):
    generated = _codegen(target).generate_stage(
        parse(_source(f"{EXIT} {HALVING} {STORE}")), "compute"
    )
    helper = _helper(generated)
    declaration = re.search(r"bool (_*crossgl_subgroup_active) = \(?!", helper)
    assert declaration is not None
    # Every invocation halves the counter and shuffles; only the store is
    # predicated, and nothing leaves before the shuffles' barriers.
    assert helper.index("offset > 0") < helper.index(f"if ({declaration.group(1)})")
    assert "return;" not in helper
    _compile(generated, target, tmp_path)


@pytest.mark.parametrize("target", ["directx", "opengl"])
@pytest.mark.parametrize(
    ("value_type", "shuffle"),
    [
        ("float", "WaveShuffleDown(pair.value, delta)"),
        # A bfloat16 member shuffles as float and converts back.
        ("bfloat16", "bfloat16(WaveShuffleDown(float(pair.value), delta))"),
    ],
)
def test_struct_shuffle_overload_runs_in_every_invocation(
    tmp_path, target, value_type, shuffle
):
    helpers = PAIR_SHUFFLE.replace("WaveShuffleDown(pair.value, delta)", shuffle)
    generated = _codegen(target).generate_stage(
        parse(
            _source(f"{EXIT} {PAIR_REDUCE} {PAIR_STORE}", helpers, value_type),
        ),
        "compute",
    )
    helper = _helper(generated)
    assert ACTIVE.search(helper)
    assert "return;" not in helper
    # The shuffle runs unpredicated; the combine and the store do not.
    active = ACTIVE.search(helper).group(0)
    shuffle = helper.index("shuffle_pair(pair, offset)")
    assert helper.index(f"if ({active})", shuffle) > shuffle
    _compile(generated, target, tmp_path)


@pytest.mark.parametrize("target", ["directx", "opengl"])
@pytest.mark.parametrize(
    ("body", "helpers"),
    [
        # A halving counter divided by a value that may be zero.
        (
            f"{EXIT} for (uint offset = 2u; offset > 0u; offset /= limit) "
            f"{{ value += WaveShuffleDown(value, offset); }} {STORE}",
            "",
        ),
        # A remainder update by a constant zero traps.
        (
            f"{EXIT} for (uint offset = 2u; offset > 0u; offset %= 0u) "
            f"{{ value += WaveShuffleDown(value, offset); }} {STORE}",
            "",
        ),
        # The body writes the halving counter.
        (
            f"{EXIT} for (uint offset = 2u; offset > 0u; offset /= 2u) "
            f"{{ value += WaveShuffleDown(value, offset); offset = 0u; }} {STORE}",
            "",
        ),
        # The overload reads memory that inactive invocations may not read.
        (
            f"{EXIT} {PAIR_REDUCE} {PAIR_STORE}",
            PAIR_SHUFFLE.replace(
                "WaveShuffleDown(pair.value, delta)",
                "WaveShuffleDown(inputs[pair.index], delta)",
            ),
        ),
        # The overload sums the subgroup, which the inactive lanes would join.
        (
            f"{EXIT} {PAIR_REDUCE} {PAIR_STORE}",
            PAIR_SHUFFLE.replace(
                "WaveShuffleDown(pair.value, delta)", "WaveActiveSum(pair.value)"
            ),
        ),
    ],
)
def test_exit_shuffles_reject_unsafe_inactive_work(target, body, helpers):
    with pytest.raises(ERRORS[target]):
        _codegen(target).generate_stage(parse(_source(body, helpers)), "compute")


# A two-level arg reduction: each 32-lane simdgroup folds its lanes'
# (index, value) pairs, writes one pair to threadgroup memory, and only the
# first simdgroup folds those pairs. Ties keep the lower index.
GROUP = 128
METAL_SOURCE = """#include <metal_stdlib>
using namespace metal;

struct IndexValue {
  uint index;
  float value;
};

IndexValue simd_shuffle_down(IndexValue data, ushort delta) {
  return IndexValue{simd_shuffle_down(data.index, delta),
                    simd_shuffle_down(data.value, delta)};
}

IndexValue larger(IndexValue best, IndexValue current) {
  if (best.value < current.value ||
      (best.value == current.value && best.index > current.index)) {
    return current;
  }
  return best;
}

kernel void group_argmax(const device float* values [[buffer(0)]],
                         device uint* indices [[buffer(1)]],
                         uint3 lid [[thread_position_in_threadgroup]],
                         uint3 lsize [[threads_per_threadgroup]],
                         uint3 tid [[threadgroup_position_in_grid]],
                         uint simd_size [[threads_per_simdgroup]],
                         uint simd_lane_id [[thread_index_in_simdgroup]],
                         uint simd_group_id [[simdgroup_index_in_threadgroup]]) {
  threadgroup IndexValue local_data[32];
  IndexValue best{lid.x, values[tid.x * lsize.x + lid.x]};
  for (uint offset = simd_size / 2; offset > 0; offset /= 2) {
    IndexValue neighbor = simd_shuffle_down(best, offset);
    best = larger(best, neighbor);
  }
  if (simd_lane_id == 0) {
    local_data[simd_group_id] = best;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simd_group_id != 0) {
    return;
  }
  uint simd_groups = (lsize.x + simd_size - 1) / simd_size;
  if (simd_lane_id < simd_groups) {
    best = local_data[simd_lane_id];
  }
  for (uint offset = simd_size / 2; offset > 0; offset /= 2) {
    IndexValue neighbor = simd_shuffle_down(best, offset);
    best = larger(best, neighbor);
  }
  if (lid.x == 0) {
    indices[tid.x] = best.index;
  }
}
"""


def _bits(value):
    return struct.unpack("<I", struct.pack("<f", value))[0]


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
    from crosstl.project.runtime_value_encoding import FLOAT32_BITS

    original, descriptor, package = _package(
        root, target, "float", (GROUP, 1, 1), source=METAL_SOURCE
    )
    groups = 2
    # Each group's maximum appears twice, so ties must keep the lower index,
    # and once in a lane past the first simdgroup.
    inputs = [float((index * 37) % 101) for index in range(GROUP * groups)]
    for group, (first, second) in enumerate(((77, 109), (40, 41))):
        inputs[group * GROUP + first] = inputs[group * GROUP + second] = 500.0
    expected = [77, 40]
    marker = 0xFFFFFFFF
    bound_expected = _bound_values(
        descriptor,
        {"indices": {"dtype": "uint32", "values": expected, "shape": [groups]}},
    )
    request = build_native_loader_dispatch_request(
        descriptor,
        package,
        _bound_values(
            descriptor,
            {
                "values": {
                    "dtype": "float32",
                    "encoding": FLOAT32_BITS,
                    "values": [_bits(value) for value in inputs],
                    "shape": [len(inputs)],
                },
                "indices": {
                    "dtype": "uint32",
                    "values": [marker] * groups,
                    "shape": [groups],
                },
            },
        ),
        bound_expected,
        {"workgroupCount": [groups, 1, 1], "workgroupSize": [GROUP, 1, 1]},
        expected_target=target,
    )
    assert not request.execution_plan.diagnostics
    return original, request, bound_expected


@pytest.mark.parametrize("target", ["directx", "opengl"])
def test_group_arg_reduction_translates_through_projects(tmp_path, target):
    _, request, _ = _request(tmp_path, target)
    generated = request.artifact_path.read_text(encoding="utf-8")
    assert ACTIVE.search(generated)
    _compile(generated, target, tmp_path)


def test_group_arg_reduction_executes_natively(tmp_path):
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 for required exit-shuffle execution")
    target = {"darwin": "metal", "win32": "directx", "linux": "opengl"}[sys.platform]
    original, request, expected = _request(tmp_path, target)
    _execute(
        request,
        expected,
        tmp_path,
        original_source=original,
        original_entry="group_argmax",
    )

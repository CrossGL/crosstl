"""Addresses of indexed elements leave a uniform loop counter uniform.

``&vals[i]`` designates an element of ``vals``. The counter ``i`` is only read to select the element, so it
can neither be written through the pointer nor diverge (C++ [expr.unary.op]/3). DirectX software subgroup validation treated every name read
by such an operand as address-taken, so a constant-trip loop that passed ``&vals[i * rows]`` to a helper
with a subgroup shuffle was refused as potentially divergent. MLX's ``MMATile::row_reduce`` passes
``&vals[i * kElemRows]`` to the butterfly ``BaseMMAFrag::row_reduce``.
"""

import os
import sys

import pytest

from crosstl.project import build_native_loader_dispatch_request
from tests.test_translator.test_boolean_buffer_runtime import _bound_values
from tests.test_translator.test_metal_builtin_ownership import _compile
from tests.test_translator.test_software_subgroup_product import _package

REQUIRE_ENV = "CROSTL_REQUIRE_SUBGROUP_LOOP_ELEMENT_ADDRESSES"
ENTRY = "loop_element_addresses"
WIDTH = 32
WORKGROUP = (32, 2, 1)
GROUPS = 2
THREADS = GROUPS * WORKGROUP[0] * WORKGROUP[1]
ROWS = 3
GUARD = [0x6D2B79F5, -7, 0x1234567, -0x5A5A5A5]
VALUES = [((index * 2654435761) % 4001) - 2000 for index in range(THREADS * ROWS * 2)]

SOURCE = """#include <metal_stdlib>
using namespace metal;

static inline void row_max(thread const int2& frag, thread int* reduced) {
  int own = metal::max(frag.x, frag.y);
  int pair = metal::max(own, simd_shuffle_xor(own, ushort(1)));
  int quad = metal::max(pair, simd_shuffle_xor(pair, ushort(8)));
  reduced[0] = metal::max(reduced[0], quad);
}

static inline void bump(thread int* slot, int amount) {
  slot[0] += simd_shuffle_down(amount, ushort(1));
}

[[kernel]] void loop_element_addresses(
    const device int* values [[buffer(0)]],
    device int* results [[buffer(1)]],
    uint index [[thread_index_in_threadgroup]],
    uint group [[threadgroup_position_in_grid]]) {
  uint global = group * 64 + index;
  int2 frags[3];
  for (short i = 0; i < 3; ++i) {
    frags[i] = int2(values[6 * global + 2 * i], values[6 * global + 2 * i + 1]);
  }
  int vals[3] = {-3000, 0, -3000};
  for (short i = 0; i < 3; ++i) {
    row_max(frags[i], &vals[i * 1]);
  }
  for (short i = 0; i < 2; ++i) {
    bump(&vals[i + 1], int(i) + 1);
  }
  for (int i = 0; i < 3; ++i) {
    results[3 * global + i] = vals[i];
  }
}
"""


def _expected():
    results = []
    for global_index in range(THREADS):
        lane = global_index % WIDTH
        base = global_index - lane

        def own(source_lane, row):
            start = 6 * (base + source_lane) + 2 * row
            return max(VALUES[start], VALUES[start + 1])

        def pair(source_lane, row):
            return max(own(source_lane, row), own(source_lane ^ 1, row))

        vals = [-3000, 0, -3000]
        for row in range(ROWS):
            vals[row] = max(vals[row], pair(lane, row), pair(lane ^ 8, row))
        for i in range(2):
            # Lane 31 has no lane below it, so shuffle-down keeps its own value.
            vals[i + 1] += i + 1
        results.extend(vals)
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
def test_element_addresses_keep_constant_loops_uniform(tmp_path, target):
    _, request, _ = _request(tmp_path, target)
    generated = request.artifact_path.read_text()
    # The shuffles are lowered through the logical subgroup scratch inside the loops.
    marker = (
        "__crossgl_software_subgroup_shuffle_xor_int"
        if target == "directx"
        else "crossglSoftwareSubgroupShuffleXorInt"
    )
    assert marker in generated
    _compile(generated, target, tmp_path)


def test_address_of_counter_still_refuses_the_loop():
    from crosstl.translator.codegen.directx_codegen import (
        DirectXSoftwareSubgroupError,
        HLSLCodeGen,
    )
    from crosstl.translator.lexer import Lexer
    from crosstl.translator.parser import Parser

    code = """
    shader AddressOfCounter {
        compute {
            void store(thread int* slot) { *slot = 3; }
            @ stage_entry
            @ numthreads(32, 1, 1)
            void main(RWStructuredBuffer<uint> output @buffer(0),
                      uint groupIndex @gl_LocalInvocationIndex) {
                uint total = 0u;
                for (int i = 0; i < 4; ++i) {
                    store(&i);
                    total += WaveShuffleDown(groupIndex, 1u);
                }
                output[groupIndex] = total;
            }
        }
    }
    """
    codegen = HLSLCodeGen(
        relative_wave_shuffle_out_of_range="self", software_subgroup_width=32
    )
    with pytest.raises(DirectXSoftwareSubgroupError) as excinfo:
        codegen.generate(Parser(Lexer(code).get_tokens()).parse())
    assert excinfo.value.reason == "potentially-divergent-control-flow"


def test_element_addresses_compile_on_metal(tmp_path):
    _, request, _ = _request(tmp_path, "metal")
    _compile(
        request.artifact_path.read_text(),
        "metal",
        tmp_path,
        metal_compile_flags=("-Wall", "-Wextra"),
    )


def test_element_addresses_execute(tmp_path):
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 for required native element addresses")
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


def test_element_addresses_are_required_on_all_native_platforms():
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
    assert "test_subgroup_loop_element_addresses.py" in step
    assert "if:" not in step and "continue-on-error" not in step

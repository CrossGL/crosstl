"""Metal evaluates a conditionally evaluated call that copies reference arguments in place.

CrossGL follows GLSL: the right operand of ``&&`` and ``||`` is evaluated only when the left one does not decide
the result, and only one arm of ``?:`` is evaluated (OpenGL Shading Language 4.60, section 5.9); an ``inout``
argument is copied in at its turn and copied back when the call returns (section 6.1.1). Metal lowers ``inout`` to
``thread T&``, and an argument it cannot bind directly is copied through a thread temporary. Copies made around the
whole statement would run whether or not the operand runs, so such a call is evaluated in place as a comma
expression whose temporaries are declared before the statement.
"""

from __future__ import annotations

import textwrap

import pytest

import crosstl
from crosstl.translator.codegen.metal_refusals import MetalLoweringRefused
from tests.metal_compute_dispatch import compile_metal_library, dispatch_word_buffer

REQUIRE_ENV = "CROSSTL_REQUIRE_REFERENCE_ALIASING_EXECUTION"

PROBE = """
#version 450
layout(local_size_x = 1) in;
layout(std430, binding = 0) buffer Outp { int o[]; };
shared int tile[4];
int bump(inout int t) { t = t * 3 + 1; return t - 2; }
int pick(inout int p, int q) { p = p + q; return p; }
bool below(inout int t, int limit) { t = t + 1; return t < limit; }
void main() {
    bool yes = o[30] != 0;
    bool no = o[31] != 0;
    int m = 7;
    o[0] = yes ? pick(m, bump(m)) : -1;
    o[1] = m;
    m = 7;
    o[2] = no ? pick(m, bump(m)) : -1;
    o[3] = m;
    tile[0] = 100;
    bool b = no && below(tile[0], 103);
    o[4] = (b ? 1000 : 0) + tile[0];
    b = yes && below(tile[0], 103);
    o[5] = (b ? 1000 : 0) + tile[0];
    b = yes || below(tile[0], 103);
    o[6] = (b ? 1000 : 0) + tile[0];
    b = no || below(tile[0], 102);
    o[7] = (b ? 1000 : 0) + tile[0];
    tile[1] = 7;
    o[8] = yes ? pick(tile[1], bump(tile[1])) : -1;
    o[9] = tile[1];
    int n = 0;
    tile[2] = 100;
    while (n < 3 && below(tile[2], 103)) { n = n + 1; }
    o[10] = n * 1000 + tile[2];
    tile[3] = 100;
    if (no ? below(tile[3], 103) : false) { n = n + 100; }
    o[11] = n * 1000 + tile[3];
}
"""

# Slots 30 and 31 hold the inputs `yes` (1) and `no` (0). bump(7) leaves 22 and returns 20.
WORDS = [0] * 32
WORDS[30] = 1
EXPECTED = [
    27,  # pick copies m (7) in before bump stores 22: 7 + 20, stored back over 22
    27,
    -1,  # the arm that is not taken copies nothing
    7,
    100,  # `no && ...` does not call below
    1101,  # `yes && ...` calls it once: 101 < 103
    1101,  # `yes || ...` does not call it
    102,  # `no || ...` calls it: 102 < 102 is false
    27,  # pick's copy of tile[1] is made before bump's, and stored back after it
    27,
    2103,  # the while condition calls below on each of its three tests
    2100,  # the if condition's untaken arm does not call below
]

BYTE_BLOCK = """
#version 450
layout(local_size_x = 1) in;
layout(std430, binding = 0) buffer Outp { int o[]; };
int bump(inout int t) { t = t * 3 + 1; return t - 2; }
void main() {
    bool yes = o[30] != 0;
    o[13] = yes ? bump(o[12]) : -1;
}
"""


def _translate(tmp_path, source, suffix="comp"):
    path = tmp_path / f"conditional.{suffix}"
    path.write_text(textwrap.dedent(source).strip() + "\n", encoding="utf-8")
    return crosstl.translate(str(path), backend="metal", format_output=False)


def _kernel_body(code):
    lines = code.splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith("kernel void"))
    end = next(i for i in range(start, len(lines)) if lines[i] == "}")
    return [line.strip() for line in lines[start + 1 : end] if line.strip()]


@pytest.fixture(scope="module")
def probe_metal(tmp_path_factory):
    return _translate(tmp_path_factory.mktemp("conditional"), PROBE)


@pytest.fixture(scope="module")
def probe_body(probe_metal):
    return _kernel_body(probe_metal)


def test_ternary_arm_copies_only_when_it_is_taken(probe_body):
    index = probe_body.index("int m = 7;")
    assert probe_body[index + 1 : index + 5] == [
        "int __crossgl_ref1;",
        "int __crossgl_arg2;",
        "int __crossgl_call3;",
        "(*reinterpret_cast<device int*>(outp + 0)) = yes ? (__crossgl_ref1 = m, "
        "__crossgl_arg2 = bump(m), __crossgl_call3 = pick(__crossgl_ref1, __crossgl_arg2), "
        "m = __crossgl_ref1, __crossgl_call3) : -1;",
    ]


def test_short_circuit_operand_copies_only_when_it_runs(probe_body):
    assert (
        "bool b = no && (__crossgl_ref7 = tile[0], __crossgl_call8 = "
        "below(__crossgl_ref7, 103), tile[0] = __crossgl_ref7, __crossgl_call8);"
    ) in probe_body


def test_inner_call_in_a_conditional_arm_is_sequenced_inside_the_outer_one(
    probe_body,
):
    line = next(line for line in probe_body if "outp + 32" in line)
    assert (
        "yes ? (__crossgl_ref17 = tile[1], __crossgl_arg18 = (__crossgl_ref15 = tile[1], "
        "__crossgl_call16 = bump(__crossgl_ref15), tile[1] = __crossgl_ref15, __crossgl_call16), "
        "__crossgl_call19 = pick(__crossgl_ref17, __crossgl_arg18), tile[1] = __crossgl_ref17, "
        "__crossgl_call19) : -1;"
    ) in line, line


def test_loop_guard_declares_its_temporaries_inside_the_loop(probe_body):
    index = probe_body.index("for (; ; ) {")
    assert probe_body[index + 1 : index + 4] == [
        "int __crossgl_ref21;",
        "bool __crossgl_call22;",
        "bool __crossgl_cond20 = n < 3 && (__crossgl_ref21 = tile[2], __crossgl_call22 = "
        "below(__crossgl_ref21, 103), tile[2] = __crossgl_ref21, __crossgl_call22);",
    ]


def test_hlsl_structured_buffer_element_in_a_short_circuit_operand(tmp_path):
    source = """
    RWStructuredBuffer<float> values : register(u0);
    bool Bump(inout float value) { value += 1.0; return value > 2.0; }
    [NumThreads(64, 1, 1)]
    void Execute(uint3 ThreadIds : SV_DispatchThreadID) {
        bool done = ThreadIds.x > 3 && Bump(values[ThreadIds.x]);
        values[0] = done ? 1.0 : 0.0;
    }
    """
    code = _translate(tmp_path, source, "hlsl")
    assert (
        "bool done = ThreadIds.x > 3 && (__crossgl_ref1 = values[ThreadIds.x], "
        "__crossgl_call2 = Bump(__crossgl_ref1), values[ThreadIds.x] = __crossgl_ref1, "
        "__crossgl_call2);"
    ) in _kernel_body(code)
    compile_metal_library(code, tmp_path, REQUIRE_ENV)


def test_byte_block_element_copied_in_a_conditional_arm_is_refused(tmp_path):
    with pytest.raises(MetalLoweringRefused, match="buffer block held as bytes"):
        _translate(tmp_path, BYTE_BLOCK)


def test_conditional_copies_match_glsl_on_metal(probe_metal, tmp_path):
    got = dispatch_word_buffer(probe_metal, tmp_path, WORDS, REQUIRE_ENV)
    assert got[: len(EXPECTED)] == EXPECTED

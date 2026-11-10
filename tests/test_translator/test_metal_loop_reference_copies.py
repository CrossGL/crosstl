"""Metal evaluates calls that copy reference arguments in a ``do``/``while`` condition or a ``for`` update in place.

CrossGL follows GLSL: an ``inout`` argument is copied in at its turn and copied back when the call returns (OpenGL
Shading Language 4.60, section 6.1.1), and a ``continue`` in the body of a ``do`` loop goes to its condition, and in
the body of a ``for`` loop to its update (section 6.4). Metal lowers ``inout`` to ``thread T&``, and an argument it
cannot bind directly is copied through a thread temporary. No statement runs before every evaluation of these two
headers, so each such call is evaluated in place as a comma expression whose temporaries are declared before the
loop.
"""

from __future__ import annotations

import textwrap

import pytest

import crosstl
from crosstl.translator.codegen.metal_refusals import MetalLoweringRefused
from tests.metal_compute_dispatch import compile_metal_library, dispatch_word_buffer
from tests.test_translator.test_metal_reference_argument_headers import HEADERS

REQUIRE_ENV = "CROSSTL_REQUIRE_REFERENCE_ALIASING_EXECUTION"

PROBE = """
#version 450
layout(local_size_x = 1) in;
layout(std430, binding = 0) buffer Outp { int o[]; };
shared int tile[4];
bool below(inout int t, int limit) { t = t + 1; return t < limit; }
int bump(inout int t) { t = t * 3 + 1; return t - 2; }
void main() {
    int n = 0;
    tile[0] = 100;
    do { n = n + 1; } while (below(tile[0], 104));
    o[0] = n * 1000 + tile[0];
    n = 0;
    tile[1] = 100;
    do {
        n = n + 1;
        if (n == 2) { continue; }
    } while (n < 10 && below(tile[1], 104));
    o[1] = n * 1000 + tile[1];
    tile[2] = 1;
    int s = 0;
    for (int i = 0; i < 3; s += bump(tile[2])) {
        i = i + 1;
        if (i == 2) { continue; }
    }
    o[2] = s;
    o[3] = tile[2];
    tile[3] = 0;
    int k = 0;
    for (int j = 0; j < 4; j++, below(tile[3], 100)) { k = k + j; }
    o[4] = k * 1000 + tile[3];
}
"""

EXPECTED = [
    4104,  # the condition copies tile[0] on each of its four evaluations
    4104,  # the continue at n == 2 still evaluates the condition; a guard in the body would give 5104
    51,  # the update runs after each of the three bodies, including the one that continues: 2 + 11 + 38
    40,
    6004,  # the discarded call in the update copies tile[3] back after each of the four bodies
]

DEVICE_DO_WHILE = """
RWStructuredBuffer<int> values : register(u0);
bool Below(inout int value, int limit) { value += 1; return value < limit; }
[numthreads(1, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    int n = 0;
    do { n += 1; } while (Below(values[id.x], 3));
    for (int i = 0; i < 2; i += 1, Below(values[id.x + 1], 9)) { n += 1; }
    values[2] = n;
}
"""


def _translate(tmp_path, source, suffix="comp"):
    path = tmp_path / f"loops.{suffix}"
    path.write_text(textwrap.dedent(source).strip() + "\n", encoding="utf-8")
    return crosstl.translate(str(path), backend="metal", format_output=False)


def _kernel_body(code):
    lines = code.splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith("kernel void"))
    end = next(i for i in range(start, len(lines)) if lines[i] == "}")
    return [line.strip() for line in lines[start + 1 : end] if line.strip()]


@pytest.fixture(scope="module")
def probe_metal(tmp_path_factory):
    return _translate(tmp_path_factory.mktemp("loops"), PROBE)


@pytest.fixture(scope="module")
def probe_body(probe_metal):
    return _kernel_body(probe_metal)


def test_do_while_condition_copies_on_every_evaluation(probe_body):
    index = probe_body.index("tile[1] = 100;")
    assert probe_body[index + 1 : index + 4] == [
        "int __crossgl_ref3;",
        "bool __crossgl_call4;",
        "do {",
    ]
    assert (
        "} while (n < 10 && (__crossgl_ref3 = tile[1], __crossgl_call4 = "
        "below(__crossgl_ref3, 104), tile[1] = __crossgl_ref3, __crossgl_call4));"
    ) in probe_body


def test_for_update_copies_after_every_body(probe_body):
    index = probe_body.index("int s = 0;")
    assert probe_body[index + 1 : index + 3] == [
        "int __crossgl_ref5;",
        "int __crossgl_call6;",
    ]
    assert probe_body[index + 3].startswith("for (int i = 0; i < 3; s = ")
    assert (
        "(__crossgl_ref5 = tile[2], __crossgl_call6 = bump(__crossgl_ref5), "
        "tile[2] = __crossgl_ref5, __crossgl_call6)"
    ) in probe_body[index + 3]


def test_discarded_update_call_keeps_no_result(probe_body):
    # A comma expression that ended with the unused result would not compile with -Werror (-Wunused-value).
    line = next(line for line in probe_body if line.startswith("for (int j = 0;"))
    assert line.endswith(
        ", (__crossgl_ref7 = tile[3], below(__crossgl_ref7, 100), tile[3] = __crossgl_ref7)) {"
    ), line
    assert "bool __crossgl_call8;" not in probe_body


def test_hlsl_structured_buffer_elements_in_loop_headers(tmp_path):
    code = _translate(tmp_path, DEVICE_DO_WHILE, "hlsl")
    body = _kernel_body(code)
    assert (
        "} while ((__crossgl_ref1 = values[id.x], __crossgl_call2 = Below(__crossgl_ref1, 3), "
        "values[id.x] = __crossgl_ref1, __crossgl_call2));"
    ) in body
    assert any(
        line.startswith("for (int i = 0; i < 2;")
        and line.endswith(
            "(__crossgl_index3 = id.x + 1, __crossgl_ref4 = values[__crossgl_index3], "
            "Below(__crossgl_ref4, 9), values[__crossgl_index3] = __crossgl_ref4)) {"
        )
        for line in body
    ), body
    compile_metal_library(code, tmp_path, REQUIRE_ENV)


def test_switch_selector_of_another_type_is_refused_with_a_type(tmp_path):
    path = tmp_path / "shader.cgl"
    path.write_text(HEADERS.replace("switch (take(tile[3]))", "switch (flip(tile[3]))"))
    with pytest.raises(MetalLoweringRefused, match="switch selector of type 'bool'"):
        crosstl.translate(str(path), backend="metal", format_output=False)


def test_loop_header_copies_match_glsl_on_metal(probe_metal, tmp_path):
    got = dispatch_word_buffer(probe_metal, tmp_path, [0] * 8, REQUIRE_ENV)
    assert got[: len(EXPECTED)] == EXPECTED

"""Metal copies a reference argument that another access during its call would alias.

CrossGL follows GLSL for ``out`` and ``inout`` parameters (OpenGL Shading Language 4.60, section 6.1.1): arguments
are evaluated left to right, an ``inout`` argument is copied into the parameter at its turn, and it is copied back
when the function returns. Metal lowers these parameters to ``thread T&``, so a thread local bound directly is the
parameter. When another argument writes the local, another reference parameter binds it, or the callee reaches it
as a forwarded module-scope variable, the argument goes through a thread temporary instead. When an enclosing call
reaches a copied argument's storage through an earlier argument, the inner call is sequenced in place as a comma
expression (C++14 [expr.comma]).
"""

from __future__ import annotations

import textwrap

import pytest

import crosstl
from tests.metal_compute_dispatch import compile_metal_library, dispatch_word_buffer

REQUIRE_ENV = "CROSSTL_REQUIRE_REFERENCE_ALIASING_EXECUTION"

PROBE = """
#version 450
layout(local_size_x = 1) in;
layout(std430, binding = 0) buffer Outp { int o[]; };
shared int tile[4];
int g;
int bump(inout int t) { t = t * 3 + 1; return t - 2; }
void scale(inout int p, int q) { p = p * 10 + q; }
void f(inout int p) { g = 5; p += 1; }
void h(inout int p, inout int q) { p = p * 10; q = q + 1; }
int readG(int v) { return g * 100 + v; }
int readTile(int v) { return tile[3] * 100 + v; }
void scaleR(int q, inout int p) { p = p * 10 + q; }
void main() {
    int x = 7;
    int m = x;
    scale(m, bump(m));
    o[0] = m;
    g = x;
    f(g);
    o[1] = g;
    m = x;
    h(m, m);
    o[2] = m;
    tile[0] = x;
    int before = atomicAdd(tile[0], bump(tile[0]));
    o[3] = before * 1000 + tile[0];
    tile[1] = x;
    scale(tile[1], bump(tile[1]));
    o[4] = tile[1];
    g = x;
    int r = readG(bump(g));
    o[5] = r;
    g = x;
    int s = readG(bump(g)) + g;
    o[6] = s;
    tile[3] = x;
    o[7] = readTile(bump(tile[3]));
    tile[2] = x;
    scaleR(bump(tile[2]), tile[2]);
    o[8] = tile[2];
    m = x;
    scaleR(bump(m), m);
    o[9] = m;
    g = x;
    scaleR(bump(g), g);
    o[10] = g;
}
"""

# The values GLSL's copy-in/copy-out rules give each slot. bump(7) leaves 22 and returns 20.
EXPECTED = [
    90,  # scale copies m (7) in before bump writes it: 7 * 10 + 20, stored back over bump's 22
    8,  # f's parameter is a copy of g: g = 5 inside f, then the copy (8) is stored back
    (70, 8),  # h(m, m): two copies; GLSL leaves the order of the copy-backs undefined
    22042,  # atomicAdd(tile[0], 20) runs after bump stored 22: returns 22, leaves 42
    90,  # as slot 0, for threadgroup storage
    2220,  # readG reads g after bump stored 22
    2242,  # the same, plus g read after the call
    2220,  # readTile reads tile[3] after bump stored 22
    240,  # scaleR copies tile[2] in after bump stored 22: 22 * 10 + 20
    240,  # the same for a thread local
    240,  # the same for a module-scope variable
]

KEEP = """
#version 450
layout(local_size_x = 1) in;
layout(std430, binding = 0) buffer Outp { int o[]; };
void scale(inout int p, int q) { p = p * 10 + q; }
void main() {
    int m = 7;
    scale(m, m + 1);
    o[0] = m;
}
"""

CONDITIONAL = """
#version 450
layout(local_size_x = 1) in;
layout(std430, binding = 0) buffer Outp { int o[]; };
int pick(inout int p, int q) { p = p + q; return p; }
int bump(inout int t) { t = t * 3 + 1; return t - 2; }
void main() {
    int m = 7;
    bool c = o[1] > 0;
    int q = c ? pick(m, bump(m)) : 0;
    o[2] = q;
}
"""

METAL_REFERENCES = """
#include <metal_stdlib>
using namespace metal;
void h(thread int& p, thread int& q) { p = p * 10; q = q + 1; }
kernel void k(device int* o [[buffer(0)]]) {
    int m = 7;
    h(m, m);
    o[0] = m;
}
"""


def _translate(tmp_path, source, suffix="comp"):
    path = tmp_path / f"aliasing.{suffix}"
    path.write_text(textwrap.dedent(source).strip() + "\n", encoding="utf-8")
    return crosstl.translate(str(path), backend="metal", format_output=False)


def _kernel_body(code):
    lines = code.splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith("kernel void"))
    end = next(i for i in range(start, len(lines)) if lines[i] == "}")
    return [line.strip() for line in lines[start + 1 : end] if line.strip()]


def _following(body, line, count):
    index = body.index(line)
    return body[index : index + count]


@pytest.fixture(scope="module")
def probe_metal(tmp_path_factory):
    return _translate(tmp_path_factory.mktemp("probe"), PROBE)


@pytest.fixture(scope="module")
def probe_body(probe_metal):
    return _kernel_body(probe_metal)


def test_argument_another_argument_writes_is_copied_in_first(probe_body):
    assert _following(probe_body, "int __crossgl_ref1 = m;", 3) == [
        "int __crossgl_ref1 = m;",
        "scale(__crossgl_ref1, bump(m));",
        "m = __crossgl_ref1;",
    ]


def test_forwarded_global_the_callee_writes_is_copied(probe_body):
    assert _following(probe_body, "int __crossgl_ref2 = g;", 3) == [
        "int __crossgl_ref2 = g;",
        "f(__crossgl_ref2, g);",
        "g = __crossgl_ref2;",
    ]


def test_local_bound_to_two_references_gets_two_copies(probe_body):
    assert "h(__crossgl_ref3, __crossgl_ref4);" in probe_body
    assert "h(m, m);" not in probe_body


def test_atomic_reaching_a_copied_argument_runs_after_its_copy_back(probe_body):
    atomic = next(line for line in probe_body if line.startswith("int before = "))
    sequenced = (
        "(__crossgl_ref5 = tile[0], __crossgl_call6 = bump(__crossgl_ref5), "
        "tile[0] = __crossgl_ref5, __crossgl_call6)"
    )
    assert sequenced in atomic, atomic


def test_enclosing_reference_copy_brackets_the_inner_copy_back(probe_body):
    assert _following(probe_body, "int __crossgl_ref9 = tile[1];", 3) == [
        "int __crossgl_ref9 = tile[1];",
        "scale(__crossgl_ref9, (__crossgl_ref7 = tile[1], "
        "__crossgl_call8 = bump(__crossgl_ref7), tile[1] = __crossgl_ref7, "
        "__crossgl_call8));",
        "tile[1] = __crossgl_ref9;",
    ]


def test_argument_others_only_read_binds_directly(tmp_path):
    body = _kernel_body(_translate(tmp_path, KEEP))
    assert any(line.startswith("scale(m, ") for line in body), body
    assert not any("__crossgl_ref" in line for line in body), body


def test_metal_source_references_keep_their_aliasing(tmp_path):
    body = _kernel_body(_translate(tmp_path, METAL_REFERENCES, "metal"))
    assert "h(m, m);" in body, body


def test_aliased_argument_in_a_conditional_expression_is_copied_in_place(tmp_path):
    body = _kernel_body(_translate(tmp_path, CONDITIONAL))
    assert (
        "int q = c ? (__crossgl_ref1 = m, __crossgl_arg2 = bump(m), "
        "__crossgl_call3 = pick(__crossgl_ref1, __crossgl_arg2), m = __crossgl_ref1, "
        "__crossgl_call3) : 0;"
    ) in body, body


def test_probe_compiles(probe_metal, tmp_path):
    compile_metal_library(probe_metal, tmp_path, REQUIRE_ENV)


def test_probe_matches_glsl_copy_in_copy_out_on_metal(probe_metal, tmp_path):
    words = [-559038737] * len(EXPECTED)
    got = dispatch_word_buffer(probe_metal, tmp_path, words, REQUIRE_ENV)
    for slot, (value, expected) in enumerate(zip(got, EXPECTED)):
        allowed = expected if isinstance(expected, tuple) else (expected,)
        assert value in allowed, (slot, got)

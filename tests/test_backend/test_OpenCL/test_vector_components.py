"""OpenCL C numbered and half component selections, 4-component cross and constant-expression array extents.

The importer copied ``.s01``, ``.hi`` and friends verbatim, which DXC rejected as "invalid format for vector
swizzle"; passed a 4-component ``cross`` to CrossGL's 3-component one; and turned ``float s[N * 2]`` into an
undeclared identifier. These were the OpenCL C input coverage cells numeric-components, vector-halves, cross and
constant-array-size.
"""

import re
import shutil
import subprocess

import pytest

import crosstl
from crosstl.backend.OpenCL.array_extents import (
    OpenCLArrayExtentUnsupportedError,
    fold_integer_constant_expression,
)
from crosstl.backend.OpenCL.OpenCLCrossGLCodeGen import OpenCLToCrossGLConverter
from crosstl.backend.OpenCL.OpenCLLexer import OpenCLLexer
from crosstl.backend.OpenCL.OpenCLParser import OpenCLParser
from crosstl.backend.OpenCL.vector_components import (
    OpenCLVectorComponentUnsupportedError,
)
from crosstl.translator.lexer import Lexer as CrossGLLexer
from crosstl.translator.parser import Parser as CrossGLParser

KERNEL = """
typedef struct {{ float lo; float hi; float s0; }} Pair;
__kernel void k(__global float* a, __global float4* q, __global int* n) {{
    uint i = get_global_id(0);
    float4 v = (float4)(a[i], 1.0f, 2.0f, 3.0f);
    float3 t = (float3)(a[i], 4.0f, 5.0f);
    float2 u = (float2)(a[i], 6.0f);
{body}
}}
"""


def to_crossgl(source):
    ast = OpenCLParser(OpenCLLexer(source).tokenize()).parse()
    crossgl = OpenCLToCrossGLConverter().generate(ast)
    CrossGLParser(CrossGLLexer(crossgl).get_tokens()).parse()
    return crossgl


def kernel_body(body):
    return to_crossgl(KERNEL.format(body=body)).split("// Kernel: k", 1)[1]


def flat(code):
    return re.sub(r"\s+", " ", code)


@pytest.mark.parametrize(
    "statement, expected",
    [
        ("float2 r = v.s01;", "var r: vec2<f32> = v.xy;"),
        ("float2 r = v.s32;", "var r: vec2<f32> = v.wz;"),
        ("float r = v.S2;", "var r: f32 = v.z;"),
        ("float4 r = v.s3210;", "var r: vec4<f32> = v.wzyx;"),
        ("float2 r = v.lo;", "var r: vec2<f32> = v.xy;"),
        ("float2 r = v.hi;", "var r: vec2<f32> = v.zw;"),
        ("float2 r = v.even;", "var r: vec2<f32> = v.xz;"),
        ("float2 r = v.odd;", "var r: vec2<f32> = v.yw;"),
        (
            "float r = u.lo + u.hi + u.even + u.odd;",
            "var r: f32 = (u.x + u.y + u.x + u.y);",
        ),
        # A 3-component vector's .lo and .even select defined components only.
        ("float2 r = t.lo + t.even;", "var r: vec2<f32> = (t.xy + t.xz);"),
        # A selection of a selection is one swizzle of the vector.
        ("float r = v.odd.hi;", "var r: f32 = v.w;"),
        ("float r = v.hi.s0;", "var r: f32 = v.z;"),
        ("float r = v.xyz.odd.x;", "var r: f32 = v.y;"),
        ("float r = t.odd.x + t.hi.x;", "var r: f32 = (t.y + t.z);"),
        ("float2 r = (v * 2.0f).s13;", "var r: vec2<f32> = (v * 2.0f).yw;"),
        ("float r = (v + v).hi.y;", "var r: f32 = (v + v).w;"),
        (
            "float r = q[i].s3 + convert_float4(n[i]).hi.x;",
            "var r: f32 = (q[i].w + vec4(n[i]).z);",
        ),
    ],
)
def test_numbered_and_half_selections_become_letter_swizzles(statement, expected):
    assert expected in flat(kernel_body(f"    {statement}\n    a[i] = 0.0f;"))


@pytest.mark.parametrize(
    "statement, expected",
    [
        ("v.lo = v.odd;", "v.xy = v.yw;"),
        ("v.even = (float2)(a[i], 5.0f);", "v.xz = vec2<f32>(a[i], 5.0f);"),
        ("v.s13 = u;", "v.yw = u;"),
        ("v.S2 += 4.0f;", "v.z += 4.0f;"),
        ("t.lo = u;", "t.xy = u;"),
        ("u.hi = a[i];", "u.y = a[i];"),
    ],
)
def test_selections_are_assignable(statement, expected):
    assert expected in flat(
        kernel_body(f"    {statement}\n    a[i] = v.x + t.x + u.x;")
    )


def test_structure_members_named_like_selections_keep_their_names():
    body = kernel_body("    Pair p;\n    p.lo = 1.0f;\n    a[i] = p.hi + p.s0;")
    assert "p.lo = 1.0f;" in body
    assert "(p.hi + p.s0)" in body


@pytest.mark.parametrize(
    "statement, message",
    [
        ("t.hi = u;", "undefined w component"),
        ("t.odd = u;", "undefined w component"),
        ("u = t.hi;", "undefined w component"),
        ("u = v.xyz.odd;", "undefined w component"),
        ("a[i] = v.s4;", "reads past the 4 components"),
        ("a[i] = t.s3;", "reads past the 3 components"),
        ("v.s00 = u;", "selects a component more than once"),
    ],
)
def test_selections_crossgl_cannot_write_are_refused(statement, message):
    with pytest.raises(OpenCLVectorComponentUnsupportedError, match=message):
        kernel_body(f"    {statement}")


def test_cross_of_four_component_vectors_crosses_xyz_and_returns_zero_w():
    body = flat(kernel_body("    q[i] = cross(v, q[i]);\n    float3 c = cross(t, t);"))
    assert "q[i] = vec4<f32>(cross(v.xyz, q[i].xyz), 0.0);" in body
    assert "var c: vec3<f32> = cross(t, t);" in body


def test_double_cross_writes_a_double_zero():
    body = flat(
        kernel_body(
            "    double4 d = (double4)(1.0);\n    d = cross(d, d);\n    a[i] = (float)d.z;"
        )
    )
    assert "d = vec4<f64>(cross(d.xyz, d.xyz), f64(0.0));" in body


@pytest.mark.parametrize(
    "text, value",
    [
        ("4 * 2", 8),
        ("(4 * 2) + 1", 9),
        ("0x10u >> 2", 4),
        ("010", 8),
        ("-7 / 2", -3),
        ("-7 % 2", -1),
        ("1 << 3 | 1", 9),
        ("N * 2", None),
        ("A * 2", None),
        ("4 / 0", None),
    ],
)
def test_integer_constant_extents_fold_with_c_arithmetic(text, value):
    assert fold_integer_constant_expression(text) == value


def test_macro_extent_folds_and_enumerator_extent_stays_an_expression():
    source = """
#define N 4
enum { K = 3 };
__kernel void k(__global float* a) {
    float s[N * 2];
    float r[K * 2];
    s[7] = 1.0f;
    r[5] = 2.0f;
    a[0] = s[7] + r[5];
}
"""
    body = flat(to_crossgl(source))
    assert "var s: array<f32, 8>;" in body
    assert "var r: array<f32, (K * 2)>;" in body
    assert "array_extent" not in body


def test_non_positive_extent_is_refused():
    source = "__kernel void k(__global float* a) { float s[2 - 2]; a[0] = 0.0f; }"
    with pytest.raises(OpenCLArrayExtentUnsupportedError, match="must be positive"):
        to_crossgl(source)


COMPILED = """
#define N 4
__kernel __attribute__((reqd_work_group_size(1, 1, 1)))
void k(__global float* a, __global float4* q) {
    uint i = get_global_id(0);
    float4 v = (float4)(a[i], 1.0f, 2.0f, 3.0f);
    float3 t = (float3)(a[i], 4.0f, 5.0f);
    float s[N * 2];
    for (int j = 0; j < N * 2; j++) { s[j] = (float)j; }
    v.lo = v.odd;
    v.s13 = t.lo;
    q[i] = cross(v, q[i]) + (float4)(v.s0, t.even, s[N * 2 - 1]);
}
"""


def _run(command, cwd):
    result = subprocess.run(
        command, cwd=cwd, capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_translation_compiles_as_hlsl(tmp_path):
    dxc = shutil.which("dxc")
    if dxc is None:
        pytest.skip("dxc is not on PATH")
    source = tmp_path / "kernel.cl"
    source.write_text(COMPILED, encoding="utf-8")
    code = crosstl.translate(str(source), backend="directx")
    (tmp_path / "kernel.hlsl").write_text(code, encoding="utf-8")
    entry = re.findall(r"void\s+(\w+)\s*\(", code)[-1]
    _run(
        [
            dxc,
            "-T",
            "cs_6_6",
            "-E",
            entry,
            "-HV",
            "2021",
            "kernel.hlsl",
            "-Fo",
            "k.bin",
        ],
        tmp_path,
    )


def test_translation_compiles_as_glsl(tmp_path):
    if shutil.which("glslangValidator") is None:
        pytest.skip("glslangValidator is not on PATH")
    source = tmp_path / "kernel.cl"
    source.write_text(COMPILED, encoding="utf-8")
    (tmp_path / "kernel.comp").write_text(
        crosstl.translate(str(source), backend="opengl"), encoding="utf-8"
    )
    _run(["glslangValidator", "kernel.comp"], tmp_path)

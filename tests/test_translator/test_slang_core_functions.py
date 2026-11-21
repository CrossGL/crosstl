"""Slang output calls the Slang core-module functions that share another language's built-in name and meaning.

Slang's core module (Slang v2026.20 ``hlsl.meta.slang`` and ``core.meta.slang``) declares C and MSL math
(``fmax``, ``copysign``, ``rint``, ``nextafter``, ``fdim``, ``divide``, ``fabs``), MSL's three-operand ``max3``,
``min3``, ``median3`` and their ``f`` forms, GLSL's ``bitfieldExtract``/``bitfieldInsert`` with ``uint`` offsets, the
float packing functions, and GLSL's unpacking functions as ``unpack...ToFloat``. Slang input reads the Slang-only
spellings back as the CrossGL built-ins they mean.
"""

from __future__ import annotations

import shutil
import subprocess

import pytest

import crosstl
from crosstl.translator.codegen.slang_codegen import SlangCodeGen
from crosstl.translator.codegen.slang_refusals import SlangLoweringRefused
from crosstl.translator.lexer import Lexer
from crosstl.translator.parser import Parser

COMPUTE = """shader Probe {{
    compute {{
        layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
        buffer float* f;
        buffer int* n;
        buffer uint* u;
        void main(uvec3 gid @builtin(global_invocation_id)) {{
            float x = f[0];
            float y = f[1];
            float z = f[2];
            vec2 a = vec2(x, y);
            vec2 b = vec2(y, z);
            int i = n[0];
            int j = n[1];
            int k = n[2];
            ivec2 v = ivec2(i, j);
            uint w = u[0];
            {statement}
        }}
    }}
}}
"""

FRAGMENT = """shader Probe {{
    struct FragmentInput {{
        vec2 a0;
    }}
    struct FragmentOutput {{
        vec4 color;
    }}
    fragment {{
        buffer vec2* outp;
        FragmentOutput main(FragmentInput input) {{
            FragmentOutput output;
            outp[0] = {value};
            output.color = vec4(0.0, 0.0, 0.0, 1.0);
            return output;
        }}
    }}
}}
"""


def generate(source):
    return SlangCodeGen().generate(Parser(Lexer(source).get_tokens()).parse())


def written(statement):
    slang = generate(COMPUTE.format(statement=statement))
    target = statement.split("=", 1)[0].strip()
    lines = [
        line.strip()
        for line in slang.splitlines()
        if line.strip().startswith(target + " =")
    ]
    assert len(lines) == 1, slang
    return lines[0], slang


@pytest.mark.parametrize(
    "statement, expected",
    [
        ("f[3] = fmax(x, y);", "f[3] = fmax(x, y);"),
        ("f[3] = fmin(x, y);", "f[3] = fmin(x, y);"),
        ("f[3] = copysign(x, y);", "f[3] = copysign(x, y);"),
        ("f[3] = rint(x);", "f[3] = rint(x);"),
        ("f[3] = nextafter(x, y);", "f[3] = nextafter(x, y);"),
        ("f[3] = fdim(x, y);", "f[3] = fdim(x, y);"),
        ("f[3] = divide(x, y);", "f[3] = divide(x, y);"),
        ("f[3] = fabs(x);", "f[3] = fabs(x);"),
        ("f[3] = fmax3(x, y, z);", "f[3] = fmax3(x, y, z);"),
        ("f[3] = fmin3(x, y, z);", "f[3] = fmin3(x, y, z);"),
        ("f[3] = fmedian3(x, y, z);", "f[3] = fmedian3(x, y, z);"),
        ("f[3] = max3(x, y, z);", "f[3] = max3(x, y, z);"),
        ("f[3] = min3(x, y, z);", "f[3] = min3(x, y, z);"),
        ("f[3] = median3(x, y, z);", "f[3] = median3(x, y, z);"),
        ("u[1] = packUnorm4x8(vec4(a, b));", "u[1] = packUnorm4x8(float4(a, b));"),
        ("u[1] = packSnorm4x8(vec4(a, b));", "u[1] = packSnorm4x8(float4(a, b));"),
        ("u[1] = packUnorm2x16(a);", "u[1] = packUnorm2x16(a);"),
        ("u[1] = packSnorm2x16(a);", "u[1] = packSnorm2x16(a);"),
        ("u[1] = packHalf2x16(a);", "u[1] = packHalf2x16(a);"),
    ],
)
def test_core_module_functions_keep_their_name(statement, expected):
    assert written(statement)[0] == expected


@pytest.mark.parametrize(
    "statement, expected",
    [
        ("f[3] = unpackUnorm4x8(w).x;", "f[3] = unpackUnorm4x8ToFloat(w).x;"),
        ("f[3] = unpackSnorm4x8(w).x;", "f[3] = unpackSnorm4x8ToFloat(w).x;"),
        ("f[3] = unpackUnorm2x16(w).x;", "f[3] = unpackUnorm2x16ToFloat(w).x;"),
        ("f[3] = unpackSnorm2x16(w).x;", "f[3] = unpackSnorm2x16ToFloat(w).x;"),
        ("f[3] = unpackHalf2x16(w).x;", "f[3] = unpackHalf2x16ToFloat(w).x;"),
    ],
)
def test_unpacking_functions_take_slangs_float_spelling(statement, expected):
    assert written(statement)[0] == expected


def test_bit_field_offsets_and_counts_are_uint():
    line, _ = written("n[3] = bitfieldExtract(i, 2, k) + bitfieldInsert(i, j, 1, 4);")
    assert line == (
        "n[3] = bitfieldExtract(i, uint(2), uint(k)) + bitfieldInsert(i, j, uint(1), uint(4));"
    )
    line, _ = written("u[1] = bitfieldExtract(w, w, 3u);")
    assert line == "u[1] = bitfieldExtract(w, w, 3u);"


def test_integer_three_operand_functions_use_max_and_min():
    # Slang declares max3, min3 and median3 for floating-point operands only; MSL's integer forms are max and min.
    line, slang = written(
        "n[3] = max3(i, j, k) + min3(i, j, k) + median3(i, j, k) + median3(v, v + 1, v - 1).x;"
    )
    assert line == (
        "n[3] = max(i, max(j, k)) + min(i, min(j, k)) + cgl_median3_int(i, j, k) "
        "+ cgl_median3_int2(v, v + 1, v - 1).x;"
    )
    assert "int cgl_median3_int(int x, int y, int z)" in slang
    assert "int2 cgl_median3_int2(int2 x, int2 y, int2 z)" in slang
    assert slang.count("return max(min(x, y), min(max(x, y), z));") == 2


def test_a_three_operand_call_with_two_operands_is_refused():
    with pytest.raises(
        SlangLoweringRefused, match="median3: requires 3 arguments, got 2"
    ):
        generate(COMPUTE.format(statement="f[3] = median3(x, y);"))


def test_interpolation_at_the_centroid_is_evaluate_attribute_at_centroid():
    slang = generate(FRAGMENT.format(value="interpolateAtCentroid(input.a0)"))
    assert "EvaluateAttributeAtCentroid(input.a0)" in slang


def test_interpolation_at_an_offset_stays_refused():
    # EvaluateAttributeSnapped takes an int2 offset in sixteenths of a pixel, not GLSL's vec2 offset in pixels.
    with pytest.raises(SlangLoweringRefused, match="interpolateAtOffset"):
        generate(
            FRAGMENT.format(value="interpolateAtOffset(input.a0, vec2(0.25, 0.25))")
        )


SLANG_SOURCE = """RWStructuredBuffer<float> f;
RWStructuredBuffer<uint> u;
RWStructuredBuffer<half> h;
[shader("compute")]
[numthreads(1, 1, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
    float4 q = unpackUnorm4x8ToFloat(u[0]) + unpackSnorm4x8ToFloat(u[1]);
    float2 r = unpackHalf2x16ToFloat(u[2]) + unpackSnorm2x16ToFloat(u[3]) + unpackUnorm2x16ToFloat(u[4]);
    half4 hv = unpackUnorm4x8ToHalf(u[5]);
    half2 hw = unpackSnorm2x16ToHalf(u[6]);
    float2 c = copysign_float(float2(f[0], f[1]), float2(f[2], f[3]));
    h[0] = hv.x + hw.y + copysign_half(half2(h[1], h[2]), half2(h[3], h[3])).x;
    f[4] = q.x + r.y + c.x + fmax(f[0], f[1]) + median3(f[0], f[1], f[2]);
    u[7] = packUnorm4x8(q) + bitfieldExtract(u[8], 1u, 3u);
}
"""


def _import(tmp_path, source=SLANG_SOURCE):
    path = tmp_path / "probe.slang"
    path.write_text(source, encoding="utf-8")
    return crosstl.translate(str(path), backend="cgl")


def _compact(text):
    return " ".join(text.split())


def test_slang_input_reads_slang_only_spellings_as_crossgl_built_ins(tmp_path):
    crossgl = _compact(_import(tmp_path))
    assert "unpackUnorm4x8(u[0]) + unpackSnorm4x8(u[1])" in crossgl
    assert (
        "unpackHalf2x16(u[2]) + unpackSnorm2x16(u[3]) + unpackUnorm2x16(u[4])"
        in crossgl
    )
    assert "f16vec4(unpackUnorm4x8(u[5]))" in crossgl
    assert "f16vec2(unpackSnorm2x16(u[6]))" in crossgl
    assert "copysign(vec2(f[0], f[1]), vec2(f[2], f[3]))" in crossgl
    assert "copysign(half2(h[1], h[2]), half2(h[3], h[3])).x" in crossgl
    assert "fmax(f[0], f[1]) + median3(f[0], f[1], f[2])" in crossgl
    assert "ToFloat" not in crossgl and "ToHalf" not in crossgl
    assert "copysign_" not in crossgl


def test_slang_input_reads_evaluate_attribute_at_centroid(tmp_path):
    source = """struct PSIn { float2 a0 : TEXCOORD0; };
RWStructuredBuffer<float2> outp;
[shader("fragment")]
float4 main(PSIn input) : SV_Target
{
    outp[0] = EvaluateAttributeAtCentroid(input.a0);
    return float4(0.0, 0.0, 0.0, 1.0);
}
"""
    assert "interpolateAtCentroid(input.a0)" in _compact(_import(tmp_path, source))


def _slangc(tmp_path, source, stage):
    slangc = shutil.which("slangc")
    if slangc is None:
        pytest.skip("slangc is not installed")
    path = tmp_path / f"{stage}.slang"
    path.write_text(source, encoding="utf-8")
    completed = subprocess.run(
        [
            slangc,
            str(path),
            "-target",
            "spirv",
            "-entry",
            "main",
            "-stage",
            stage,
            "-o",
            str(tmp_path / "out.spv"),
        ],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert completed.returncode == 0, (completed.stdout + completed.stderr)[-3000:]


def test_slang_output_compiles_with_slangc(tmp_path):
    statement = (
        "f[3] = fmax(x, y) + fmin(x, y) + copysign(x, y) + rint(x) + nextafter(x, y) + fdim(x, y) + divide(x, y)"
        " + fabs(x) + fmax3(x, y, z) + fmin3(x, y, z) + fmedian3(x, y, z) + max3(x, y, z) + min3(x, y, z)"
        " + median3(x, y, z) + unpackUnorm4x8(w).x + unpackSnorm4x8(w).y + unpackUnorm2x16(w).x"
        " + unpackSnorm2x16(w).y + unpackHalf2x16(w).x;"
        " n[3] = max3(i, j, k) + min3(i, j, k) + median3(v, v + 1, v - 1).x + bitfieldExtract(i, 2, k)"
        " + bitfieldInsert(i, j, 1, 4);"
        " u[1] = packUnorm4x8(vec4(a, b)) + packSnorm4x8(vec4(a, b)) + packUnorm2x16(a) + packSnorm2x16(a)"
        " + packHalf2x16(a) + bitfieldExtract(w, w, 3u);"
    )
    _slangc(tmp_path, generate(COMPUTE.format(statement=statement)), "compute")
    _slangc(
        tmp_path,
        generate(FRAGMENT.format(value="interpolateAtCentroid(input.a0)")),
        "fragment",
    )


def test_slang_round_trip_compiles_with_slangc(tmp_path):
    crossgl = tmp_path / "probe.cgl"
    crossgl.write_text(_import(tmp_path), encoding="utf-8")
    _slangc(tmp_path, crosstl.translate(str(crossgl), backend="slang"), "compute")

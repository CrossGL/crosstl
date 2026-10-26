"""WGSL output refuses numeric types WGSL cannot represent.

WGSL's scalar types are bool, i32, u32, f32 and f16 (WGSL section 6.2). A 64-bit float, a 64-bit integer or a
bfloat16 must not be narrowed to f32 or i32, which changes its range, precision and host layout, nor be written
verbatim, which no WGSL implementation accepts.
"""

import pytest

import crosstl
from crosstl.translator.codegen.wgsl_numeric_types import (
    WGSLNumericTypeUnsupportedError,
    unsupported_wgsl_numeric_family,
)
from crosstl.translator.diagnostics import TranslationDiagnostic

BUFFER = "layout(std430, binding = 0) buffer Data { float values[]; } data;"


def translate_wgsl(tmp_path, source):
    path = tmp_path / "shader.cgl"
    path.write_text(source, encoding="utf-8")
    return crosstl.translate(str(path), backend="wgsl")


def compute(body, helpers="", declarations=BUFFER):
    return f"""
shader NumericProbe {{
    {helpers}
    compute {{
        {declarations}
        layout(local_size_x = 1) in;
        void main() {{
            {body}
        }}
    }}
}}
"""


@pytest.mark.parametrize(
    "spelling,family",
    [
        ("double", "64-bit floating-point"),
        ("f64", "64-bit floating-point"),
        ("float64_t", "64-bit floating-point"),
        ("dvec3", "64-bit floating-point"),
        ("double4", "64-bit floating-point"),
        ("dmat2", "64-bit floating-point"),
        ("dmat3x4", "64-bit floating-point"),
        ("double2x3", "64-bit floating-point"),
        ("long", "64-bit integer"),
        ("ulong", "64-bit integer"),
        ("unsigned long", "64-bit integer"),
        ("int64_t", "64-bit integer"),
        ("uint64_t", "64-bit integer"),
        ("i64", "64-bit integer"),
        ("u64", "64-bit integer"),
        ("size_t", "64-bit integer"),
        ("long2", "64-bit integer"),
        ("ulong4", "64-bit integer"),
        ("i64vec2", "64-bit integer"),
        ("u64vec3", "64-bit integer"),
        ("int64_t2", "64-bit integer"),
        ("int64_t3x3", "64-bit integer"),
        ("bfloat16", "bfloat16"),
        ("bf16vec2", "bfloat16"),
    ],
)
def test_wide_and_bfloat16_spellings_are_classified(spelling, family):
    assert unsupported_wgsl_numeric_family(spelling) == family


@pytest.mark.parametrize(
    "spelling",
    [
        "float",
        "int",
        "uint",
        "bool",
        "half",
        "f16",
        "short",
        "uchar",
        "i16vec2",
        "vec4",
        "ivec2",
        "float3",
        "mat4",
        "float2x3",
        "Double",
        "longValue",
        "doubled",
    ],
)
def test_wgsl_representable_and_unrelated_spellings_are_not_refused(spelling):
    assert unsupported_wgsl_numeric_family(spelling) is None


@pytest.mark.parametrize(
    "case,source,spelling",
    [
        (
            "double-local",
            compute("double x = double(1.0); data.values[0] = float(x);"),
            "double",
        ),
        (
            "f64-generic-vector",
            compute("vec2<f64> v = vec2<f64>(1.0); data.values[0] = float(v.x);"),
            "f64",
        ),
        (
            "dvec-constructor",
            compute("data.values[0] = float(dvec2(1.0, 2.0).y);"),
            "dvec2",
        ),
        (
            "double-conversion",
            compute("float x = 1.0; data.values[0] = float(double(x) * 2.0);"),
            "double",
        ),
        (
            "dmat-local",
            compute("dmat2 m = dmat2(1.0); data.values[0] = float(m[0].x);"),
            "dmat2",
        ),
        (
            "long-local",
            compute("long x = long(1); data.values[0] = float(x);"),
            "long",
        ),
        (
            "long-c-cast",
            compute("float x = 1.0; data.values[0] = float((long)x);"),
            "long",
        ),
        (
            "ulong-local",
            compute("ulong x = ulong(1u); data.values[0] = float(x);"),
            "ulong",
        ),
        (
            "i64-generic-vector",
            compute("vec3<i64> v = vec3<i64>(1); data.values[0] = float(v.x);"),
            "i64",
        ),
        (
            "int64-storage-buffer",
            compute(
                "data.values[0] = 1;",
                declarations="layout(std430, binding = 0) buffer Data "
                "{ int64_t values[]; } data;",
            ),
            "int64_t",
        ),
        (
            "double-uniform",
            compute(
                "data.values[0] = float(scale);",
                declarations=BUFFER + " uniform double scale;",
            ),
            "double",
        ),
        (
            "double-struct-member",
            compute(
                "Pair p; p.f = 1.0; data.values[0] = p.f;",
                helpers="struct Pair { double d; float f; }",
            ),
            "double",
        ),
        (
            "double-helper-signature",
            compute(
                "data.values[0] = float(twice(1.0));",
                helpers="double twice(double a) { return a + a; }",
            ),
            "double",
        ),
        (
            "bfloat16-local",
            compute("bfloat16 x = bfloat16(1.0); data.values[0] = float(x);"),
            "bfloat16",
        ),
    ],
)
def test_unrepresentable_numeric_types_fail_closed(tmp_path, case, source, spelling):
    with pytest.raises(WGSLNumericTypeUnsupportedError) as raised:
        translate_wgsl(tmp_path, source)

    error = raised.value
    assert isinstance(error, TranslationDiagnostic)
    assert error.code == "project.translate.wgsl-numeric-type-unsupported"
    assert error.target == "wgsl"
    assert error.spelling == spelling
    assert f"type '{spelling}'" in str(error)
    assert "WGSL section 6.2" in str(error)


def test_struct_named_like_a_wide_vector_is_still_a_struct(tmp_path):
    code = translate_wgsl(
        tmp_path,
        compute(
            "long2 p; p.lo = 1.0; data.values[0] = p.lo;",
            helpers="struct long2 { float lo; float hi; }",
        ),
    )

    assert "struct long2 {" in code
    assert "var p: long2;" in code


def test_supported_scalars_still_translate(tmp_path):
    code = translate_wgsl(
        tmp_path,
        compute(
            "float f = 1.0; int i = 2; uint u = 3u; half h = half(0.5);"
            " data.values[0] = f + float(i) + float(u) + float(h);"
        ),
    )

    assert "var f: f32 = 1.0;" in code
    assert "var i: i32 = 2;" in code
    assert "var u: u32 = 3u;" in code

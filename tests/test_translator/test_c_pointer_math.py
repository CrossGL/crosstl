"""CUDA and HIP output write ``frexp``, ``modf``, ``sincos``, ``remquo`` and ``ldexp`` in the C functions' shapes.

CrossGL's GLSL-shaped calls take the part as an ``out`` argument and accept vectors; CUDA and HIP have only the
scalar C functions, which write the part through a pointer, so the call written as is did not compile.
"""

import re

import pytest

import crosstl
from crosstl.translator.codegen.lowering_refusals import LoweringRefused

BACKENDS = ("cuda", "hip")


def kernel(body, declarations=""):
    return f"""shader P {{
    struct Parts {{
        int e;
        vec2 w;
    }}
    {declarations}
    compute {{
        layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
        buffer float* outp;
        void main() {{
            uint i = gl_GlobalInvocationID.x;
            float a = outp[i];
            vec2 v = vec2(a, 1.0);
            {body}
        }}
    }}
}}
"""


def translate(tmp_path, source, backend, suffix=".cgl"):
    path = tmp_path / f"kernel{suffix}"
    path.write_text(source)
    return crosstl.translate(str(path), backend=backend, format_output=False)


def bare_calls(code):
    """Calls spelled with the GLSL names, which have no float or vector form in CUDA and HIP."""
    return re.findall(r"(?<![\w.])(?:frexp|modf|ldexp)\s*\(", code)


@pytest.mark.parametrize("backend", BACKENDS)
def test_scalar_parts_are_written_through_a_pointer(tmp_path, backend):
    code = translate(
        tmp_path,
        kernel(
            """int e;
            float m = frexp(a, e);
            float w;
            float f = modf(a, w);
            double d = double(a);
            int de;
            double dm = frexp(d, de);
            double dw;
            double df = modf(d, dw);
            Parts p;
            float pm = frexp(a, p.e);
            int arr[2];
            float am = frexp(a, arr[1]);
            ivec4 e4;
            float cm = frexp(a, e4.y);
            outp[i] = m + float(e) + f + w + float(dm + df + dw) + float(de) + pm + am + cm;"""
        ),
        backend,
    )
    assert "float m = frexpf(a, &(e));" in code
    assert "float f = modff(a, &(w));" in code
    assert "double dm = frexp(d, &(de));" in code
    assert "double df = modf(d, &(dw));" in code
    assert "frexpf(a, &(p.e))" in code
    assert "frexpf(a, &(arr[1]))" in code
    assert "frexpf(a, &(e4.y))" in code
    assert "cgl_frexp" not in code and "cgl_modf" not in code
    assert bare_calls(code) == ["frexp(", "modf("]


@pytest.mark.parametrize("backend", BACKENDS)
def test_vector_parts_call_a_reference_helper(tmp_path, backend):
    code = translate(
        tmp_path,
        kernel(
            """ivec2 e;
            vec2 m = frexp(v, e);
            vec2 w;
            vec2 f = modf(v, w);
            Parts p;
            vec2 g = modf(v, p.w);
            ivec2 e2;
            vec2 n = frexp(m, e2);
            outp[i] = m.x + float(e.y) + f.x + w.y + g.x + p.w.y + n.y + float(e2.x);"""
        ),
        backend,
    )
    assert bare_calls(code) == []
    assert "float2 m = cgl_frexp_float2_int2(v, e);" in code
    assert "float2 f = cgl_modf_float2_float2(v, w);" in code
    assert "cgl_modf_float2_float2(v, p.w)" in code
    assert (
        "__device__ inline float2 cgl_frexp_float2_int2(float2 x, int2& part)\n{\n"
        "    float2 result;\n"
        "    int native_part;\n"
        "    result.x = frexpf(x.x, &native_part);\n"
        "    part.x = native_part;\n"
        "    result.y = frexpf(x.y, &native_part);\n"
        "    part.y = native_part;\n"
        "    return result;\n}"
    ) in code
    assert "    result.y = modff(x.y, &native_part);\n" in code
    # One helper per value and part type, however many calls use it.
    assert code.count("inline float2 cgl_frexp_float2_int2(") == 1


@pytest.mark.parametrize("backend", BACKENDS)
def test_ldexp_vectors_and_exponent_types(tmp_path, backend):
    code = translate(
        tmp_path,
        kernel("""ivec2 e = ivec2(2, 3);
            uvec2 u = uvec2(1u, 2u);
            vec2 r1 = ldexp(v, e);
            vec2 r2 = ldexp(v, 3);
            vec2 r3 = ldexp(v, vec2(0.5, 1.5));
            float r4 = ldexp(a, 2);
            float r5 = ldexp(a, 2.5);
            vec2 r6 = ldexp(v, u);
            double r7 = ldexp(double(a), 4);
            outp[i] = r1.x + r2.y + r3.x + r4 + r5 + r6.y + float(r7);"""),
        backend,
    )
    assert "float2 r1 = cgl_ldexp_float2_int2(v, e);" in code
    assert "float2 r2 = cgl_ldexp_float2_int(v, 3);" in code
    assert "float r4 = ldexpf(a, 2);" in code
    # A floating-point exponent means x * 2^e, as DXC lowers HLSL's ldexp.
    assert "float r5 = (a * exp2f(2.5" in code
    assert "ldexp(double(a), 4)" in code or "ldexp((double)(a), 4)" in code
    assert "make_float2(ldexpf(x.x, e.x), ldexpf(x.y, e.y))" in code
    assert "make_float2(ldexpf(x.x, e), ldexpf(x.y, e))" in code
    assert "make_float2(x.x * exp2f(e.x), x.y * exp2f(e.y))" in code
    assert (
        "make_float2(ldexpf(x.x, static_cast<int>(e.x)), "
        "ldexpf(x.y, static_cast<int>(e.y)))"
    ) in code


@pytest.mark.parametrize("backend", BACKENDS)
def test_hlsl_part_types_are_converted(tmp_path, backend):
    hlsl = """RWStructuredBuffer<float> outp : register(u0);
[numthreads(1, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    float exponent;
    float m = frexp(outp[id.x], exponent);
    int whole;
    float f = modf(outp[id.x], whole);
    float2 exponents;
    float2 m2 = frexp(float2(outp[id.x], 2.0), exponents);
    outp[id.x] = m + exponent + f + whole + m2.y + exponents.x;
}
"""
    code = translate(tmp_path, hlsl, backend, suffix=".hlsl")
    assert "cgl_frexp_float_float(" in code
    assert "cgl_modf_float_int(" in code
    assert "cgl_frexp_float2_float2(" in code
    assert "    part = static_cast<float>(native_part);\n" in code
    assert "    part = static_cast<int>(native_part);\n" in code
    assert "    part.y = static_cast<float>(native_part);\n" in code


@pytest.mark.parametrize("backend", BACKENDS)
def test_helper_functions_and_reference_parameters(tmp_path, backend):
    code = translate(
        tmp_path,
        kernel(
            "ivec3 ve; float fe = 0.0; vec2 r = helper(vec3(v, a), ve, fe); "
            "outp[i] = r.y + float(ve.z) + fe;",
            declarations="""vec2 helper(vec3 x, out ivec3 ve, inout float fe) {
        vec3 s = frexp(x, ve);
        float g = frexp(x.x, fe);
        return s.xy + vec2(g, 0.0);
    }""",
        ),
        backend,
    )
    assert "cgl_frexp_float3_int3(x, ve)" in code
    assert "cgl_frexp_float_float(x.x, fe)" in code


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize(
    "body, reason",
    [
        (
            "ivec3 e; vec2 m = frexp(v, e); outp[i] = m.x;",
            "a 3-component second operand for a 2-component value",
        ),
        (
            "ivec2 e; float m = frexp(a, e); outp[i] = m;",
            "a 2-component second operand for a 1-component value",
        ),
        (
            "ivec4 e; vec2 m = frexp(v, e.xy); outp[i] = m.x;",
            "not a single assignable value",
        ),
        ("vec2 r = ldexp(v, ivec3(1, 2, 3)); outp[i] = r.x;", "3-component"),
        ("bvec2 b; vec2 r = modf(v, b); outp[i] = r.x;", "not a numeric"),
    ],
)
def test_calls_without_a_faithful_lowering_are_refused(tmp_path, backend, body, reason):
    with pytest.raises(LoweringRefused, match=re.escape(reason)):
        translate(tmp_path, kernel(body), backend)


@pytest.mark.parametrize("backend", BACKENDS)
def test_half_values_are_refused(tmp_path, backend):
    hlsl = """RWStructuredBuffer<float> outp : register(u0);
[numthreads(1, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    half h = half(outp[id.x]);
    int e;
    half m = frexp(h, e);
    outp[id.x] = float(m) + e;
}
"""
    with pytest.raises(LoweringRefused, match="frexp: the value is a half"):
        translate(tmp_path, hlsl, backend, suffix=".hlsl")


@pytest.mark.parametrize("backend", BACKENDS)
def test_sincos_and_remquo_write_every_part(tmp_path, backend):
    code = translate(
        tmp_path,
        kernel("""float s;
            float c;
            sincos(a, s, c);
            vec2 vs;
            vec2 vc;
            sincos(v, vs, vc);
            double ds;
            double dc;
            sincospi(double(a), ds, dc);
            float fs;
            float fc;
            __sincosf(a, fs, fc);
            sincospif(a, fs, fc);
            int q;
            float r = remquo(a, 2.0, q);
            ivec2 q2;
            vec2 r2 = remquo(v, vec2(3.0, 4.0), q2);
            float m = frexpf(a, q);
            float w;
            float f = modff(a, w);
            outp[i] = s + c + vs.x + vc.y + float(ds + dc) + fs + fc + r + float(q) + r2.x
                + float(q2.y) + m + f + w;"""),
        backend,
    )
    assert "sincosf(a, &(s), &(c));" in code
    assert "cgl_sincos_float2_float2_float2(v, vs, vc);" in code
    assert (
        "__device__ inline void cgl_sincos_float2_float2_float2("
        "float2 x, float2& sine, float2& cosine)\n{\n"
        "    float native_sine;\n"
        "    float native_cosine;\n"
        "    sincosf(x.x, &native_sine, &native_cosine);\n"
        "    sine.x = native_sine;\n"
        "    cosine.x = native_cosine;\n"
    ) in code
    assert "sincospi(" in code and "&(ds), &(dc));" in code
    assert "__sincosf(a, &(fs), &(fc));" in code
    assert "sincospif(a, &(fs), &(fc));" in code
    assert "float r = remquof(a, 2.0" in code and ", &(q));" in code
    assert "float2 r2 = cgl_remquo_float2_int2(v, make_float2(" in code
    assert "    result.y = remquof(x.y, y.y, &native_part);\n" in code
    assert "float m = frexpf(a, &(q));" in code
    assert "float f = modff(a, &(w));" in code
    assert bare_calls(code) == []


@pytest.mark.parametrize("backend", BACKENDS)
def test_float_spellings_take_float_values(tmp_path, backend):
    body = "double s; double c; __sincosf(double(a), s, c); outp[i] = float(s + c);"
    with pytest.raises(LoweringRefused, match="__sincosf takes a float value"):
        translate(tmp_path, kernel(body), backend)

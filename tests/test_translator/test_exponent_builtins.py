"""frexp, modf and ldexp with HLSL's operand types, lowered for GLSL and Metal.

Execution is covered by the ``g_frexp_ldexp_modf`` and ``h_frexp_ldexp_modf`` conformance cases; these tests check
the spelling each target receives, the typed refusals and the native checkers when they are installed.
"""

import re
import shutil
import subprocess

import pytest

import crosstl
from crosstl.translator import parse
from crosstl.translator.codegen.GLSL_codegen import GLSLCodeGen
from crosstl.translator.codegen.glsl_refusals import OpenGLLoweringRefused
from crosstl.translator.codegen.metal_codegen import MetalCodeGen
from crosstl.translator.codegen.metal_refusals import MetalLoweringRefused

GLSL_TYPES = """shader GlslTypes {
    RWStructuredBuffer<float> values @register(u0);
    RWStructuredBuffer<int> exponents @register(u1);
    compute {
        @numthreads(64, 1, 1)
        void main(uint i @gl_GlobalInvocationID) {
            int e;
            float m = frexp(values[i], e);
            ivec3 e3;
            vec3 m3 = frexp(vec3(values[i], 2.0, 3.0), e3);
            frexp(values[i + 1u], exponents[i]);
            float whole;
            float part = modf(values[i], whole);
            values[i] = m + m3.y + part + whole + ldexp(values[i], e) + ldexp(m3, e3).z
                + ldexp(m3, e).x;
        }
    }
}"""

HLSL_TYPES = """shader HlslTypes {
    RWStructuredBuffer<float> values @register(u0);
    compute {
        @numthreads(64, 1, 1)
        void main(uint i @gl_GlobalInvocationID) {
            float e;
            float m = frexp(values[i], e);
            vec3 e3;
            vec3 m3 = frexp(vec3(values[i], 2.0, 3.0), e3);
            int whole;
            float part = modf(values[i], whole);
            values[i] = m + e + m3.y + e3.z + part + float(whole) + ldexp(values[i], e)
                + ldexp(m3, 2.5).x + ldexp(values[i], 3u);
        }
    }
}"""

HLSL_SOURCE = """RWStructuredBuffer<float> values : register(u0);
[numthreads(64, 1, 1)]
void CSMain(uint3 id : SV_DispatchThreadID) {
    float e;
    float m = frexp(values[id.x], e);
    int whole;
    float part = modf(values[id.x], whole);
    values[id.x] = m + e + part + float(whole) + ldexp(values[id.x], e);
}
"""

HELPER_CALL = re.compile(r"\bcrosstl_(?:frexp|modf)_\w+\(")


def _check(command, tmp_path):
    result = subprocess.run(command, capture_output=True, text=True, cwd=tmp_path)
    if "unable to find utility" in result.stderr:
        pytest.skip("xcrun has no Metal compiler")
    assert result.returncode == 0, result.stdout + result.stderr


def _check_glsl(code, tmp_path):
    if shutil.which("glslangValidator"):
        (tmp_path / "unit.comp").write_text(code, encoding="utf-8")
        _check(["glslangValidator", "unit.comp"], tmp_path)


def _check_metal(code, tmp_path):
    if shutil.which("xcrun"):
        (tmp_path / "unit.metal").write_text(code, encoding="utf-8")
        _check(["xcrun", "-sdk", "macosx", "metal", "-c", "unit.metal"], tmp_path)


def test_glsl_keeps_native_calls_when_the_operands_have_glsl_types(tmp_path):
    code = GLSLCodeGen().generate(parse(GLSL_TYPES))

    assert not HELPER_CALL.search(code)
    assert "frexp(values[i], e)" in code
    assert "modf(values[i], whole)" in code
    # A scalar exponent next to a vector value is splatted to the value's width.
    assert "ldexp(m3, ivec3(e))" in code
    _check_glsl(code, tmp_path)


def test_glsl_converts_hlsl_typed_outputs_and_exponents(tmp_path):
    code = GLSLCodeGen().generate(parse(HLSL_TYPES))

    assert "float crosstl_frexp_float_float(float x, out float part)" in code
    assert "vec3 crosstl_frexp_vec3_vec3(vec3 x, out vec3 part)" in code
    assert "float crosstl_modf_float_int(float x, out int part)" in code
    assert "crosstl_frexp_float_float(values[i], e)" in code
    # HLSL's ldexp(x, e) is x * 2^e for a floating-point e, so a fractional exponent keeps its meaning.
    assert "(values[i] * exp2(float(e)))" in code
    assert "(m3 * exp2(vec3(2.5)))" in code
    assert "ldexp(values[i], int(3u))" in code
    _check_glsl(code, tmp_path)


def test_metal_writes_every_output_through_an_address_space_overload(tmp_path):
    code = MetalCodeGen().generate(parse(GLSL_TYPES))

    for space in ("thread", "device", "threadgroup"):
        assert (
            f"static inline float crosstl_frexp_float(float x, {space} Output& part)"
            in code
        )
    # The exponent buffer element is device memory, which frexp itself cannot bind.
    assert re.search(r"crosstl_frexp_float\(values\[i \+ 1u?\], exponents\[i\]\)", code)
    assert "ldexp(m3, int3(e))" in code
    _check_metal(code, tmp_path)


def test_metal_converts_hlsl_typed_outputs_and_exponents(tmp_path):
    code = MetalCodeGen().generate(parse(HLSL_TYPES))

    assert "crosstl_frexp_float3(float3(values[i], 2.0, 3.0), e3)" in code
    assert "crosstl_modf_float(values[i], whole)" in code
    assert "(values[i] * exp2(float(e)))" in code
    _check_metal(code, tmp_path)


def test_hlsl_sources_reach_glsl_and_metal_with_valid_calls(tmp_path):
    source = tmp_path / "exponents.hlsl"
    source.write_text(HLSL_SOURCE, encoding="utf-8")

    glsl = crosstl.translate(str(source), backend="opengl")
    metal = crosstl.translate(str(source), backend="metal")

    assert "crosstl_frexp_float_float(" in glsl
    assert "crosstl_modf_float_int(" in glsl
    assert "crosstl_frexp_float(" in metal
    _check_glsl(glsl, tmp_path)
    _check_metal(metal, tmp_path)


@pytest.mark.parametrize(
    "replacement",
    [
        ("frexp(vec3(values[i], 2.0, 3.0), e3)", "frexp(vec3(values[i], 2.0, 3.0), e)"),
        ("ldexp(m3, 2.5)", "ldexp(m3, vec2(2.5))"),
        ("modf(values[i], whole)", "modf(int(values[i]), whole)"),
    ],
)
@pytest.mark.parametrize(
    "codegen, refusal",
    [(GLSLCodeGen, OpenGLLoweringRefused), (MetalCodeGen, MetalLoweringRefused)],
)
def test_operands_neither_language_defines_are_refused(replacement, codegen, refusal):
    source = HLSL_TYPES.replace(*replacement, 1)
    with pytest.raises(refusal):
        codegen().generate(parse(source))


def test_glsl_refuses_a_double_value_with_a_floating_point_exponent():
    source = HLSL_TYPES.replace("ldexp(m3, 2.5).x", "float(ldexp(double(m3.x), 2.5))")
    with pytest.raises(OpenGLLoweringRefused, match="exp2 on doubles"):
        GLSLCodeGen().generate(parse(source))


def test_a_user_function_named_frexp_is_called_unchanged():
    source = GLSL_TYPES.replace(
        "    compute {",
        "    float frexp(float x, int e) { return x; }\n    compute {",
        1,
    ).replace("vec3 m3 = frexp(vec3(values[i], 2.0, 3.0), e3);", "vec3 m3 = vec3(1.0);")
    source = source.replace("frexp(values[i + 1u], exponents[i]);", "")
    for codegen in (GLSLCodeGen, MetalCodeGen):
        code = codegen().generate(parse(source))
        assert "crosstl_frexp" not in code
        assert re.search(r"float m = frexp\(values\[i\], e\);", code)

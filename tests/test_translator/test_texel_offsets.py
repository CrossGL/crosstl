"""Texel offsets the program determines become the constant expressions GLSL, HLSL, WGSL and SPIR-V require.

GLSL and HLSL take the offset of a texel lookup as a constant expression, WGSL as a const-expression and SPIR-V as a
``ConstOffset`` operand (a non-constant ``Offset`` is valid on gathers only, VUID-RuntimeSpirv-Offset-10213). MSL
takes any ``int2``, so Metal shaders keep offsets in locals and compute them in loops. Constant locals are forwarded
into the offsets and constant-trip loops that compute them are unrolled; anything else is still refused.
"""

import re
import shutil
import subprocess

import pytest

import crosstl
from crosstl.translator import ast as A
from crosstl.translator.lexer import Lexer
from crosstl.translator.parser import Parser
from crosstl.translator.semantics.texel_offsets import legalize_texel_offsets

GLSL_REFUSAL = "compile-time integer constants"
SPIRV_REFUSAL = "non-constant texel offset is valid only on gather operations"

LOCAL_OFFSET = """
shader LocalOffset {
    fragment {
        uniform sampler2D colorMap;
        vec4 main(vec2 uv @ TEXCOORD0) @ gl_FragColor {
            ivec2 offset = ivec2(1, -1);
            int step = 2;
            ivec2 wide = ivec2(step, -step) + offset;
            vec4 a = textureOffset(colorMap, uv, offset);
            vec4 b = textureLodOffset(colorMap, uv, 0.0, wide);
            vec4 c = texelFetchOffset(colorMap, ivec2(2, 3), 0, offset);
            return a + b + c;
        }
    }
}
"""

PCF_METAL = """
#include <metal_stdlib>
using namespace metal;
struct VOut { float4 position [[position]]; float2 uv; };
fragment float4 frag(VOut in [[stage_in]], texture2d<float> tex [[texture(0)]],
                     sampler s [[sampler(0)]]) {
    float4 sum = float4(0.0);
    for (int i = -1; i <= 1; i++) {
        for (int j = -1; j <= 1; ++j) {
            sum += tex.sample(s, in.uv, int2(i, j));
        }
    }
    return sum / 9.0;
}
"""


def _translate(tmp_path, source, backend, suffix="cgl"):
    path = tmp_path / f"shader.{suffix}"
    path.write_text(source)
    return crosstl.translate(str(path), backend=backend)


def _fragment(body, declarations=""):
    return f"""
shader Probe {{
    {declarations}
    fragment {{
        uniform sampler2D colorMap;
        uniform int k;
        vec4 main(vec2 uv @ TEXCOORD0) @ gl_FragColor {{
            vec4 sum = vec4(0.0);
            {body}
            return sum;
        }}
    }}
}}
"""


def _plain(code):
    """``code`` with negative literals unparenthesized, as some generators print them: ``ivec2((-1), 0)``."""
    return re.sub(r"\((-\d+)\)", r"\1", code)


def _offsets(code):
    return re.findall(r"textureOffset\([^;]*?(ivec2\(-?\d+, -?\d+\))\)", _plain(code))


def _validate_glsl(tmp_path, code):
    if shutil.which("glslangValidator") is None:
        return
    path = tmp_path / "shader.frag"
    path.write_text(code)
    result = subprocess.run(
        ["glslangValidator", str(path)], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stdout + result.stderr


def _validate_spirv(tmp_path, code):
    if shutil.which("spirv-as") is None or shutil.which("spirv-val") is None:
        return
    source, binary = tmp_path / "shader.spvasm", tmp_path / "shader.spv"
    source.write_text(code)
    for command in (
        ["spirv-as", "--target-env", "vulkan1.0", str(source), "-o", str(binary)],
        ["spirv-val", "--target-env", "vulkan1.0", str(binary)],
    ):
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        assert result.returncode == 0, result.stdout + result.stderr


# Constant locals ------------------------------------------------------------------------------------------------
def test_glsl_reads_constant_locals_as_offsets(tmp_path):
    code = _translate(tmp_path, LOCAL_OFFSET, "opengl")
    plain = _plain(code)
    assert "textureOffset(colorMap, uv, ivec2(1, -1))" in plain
    assert "textureLodOffset(colorMap, uv, 0.0, ivec2(3, -3))" in plain
    assert "texelFetchOffset(colorMap, ivec2(2, 3), 0, ivec2(1, -1))" in plain
    _validate_glsl(tmp_path, code)


def test_hlsl_reads_constant_locals_as_immediate_offsets(tmp_path):
    plain = _plain(_translate(tmp_path, LOCAL_OFFSET, "directx"))
    assert ".Sample(colorMapSampler, uv, int2(1, -1))" in plain
    assert ".SampleLevel(colorMapSampler, uv, 0.0, int2(3, -3))" in plain


def test_spirv_reads_constant_locals_as_const_offsets(tmp_path):
    code = _translate(tmp_path, LOCAL_OFFSET, "vulkan")
    assert len(re.findall(r"\bOpImage\w+ .*\bConstOffset\b", code)) == 3
    assert not re.search(r"(?<!Const)Offset %", code)
    assert "ImageGatherExtended" not in code
    _validate_spirv(tmp_path, code)


def test_wgsl_reads_constant_locals_and_offsets_texel_fetches(tmp_path):
    code = _translate(tmp_path, LOCAL_OFFSET, "wgsl")
    plain = _plain(code)
    assert "textureSample(colorMap, colorMap_sampler, uv, vec2<i32>(1, -1))" in plain
    assert "textureLoad(colorMap, (vec2<i32>(2, 3) + vec2<i32>(1, -1)), 0)" in plain
    assert (
        "textureSampleLevel(colorMap, colorMap_sampler, uv, 0.0, vec2<i32>(3, -3))"
        in plain
    )
    assert "texelFetchOffset" not in code
    if shutil.which("naga") is not None:
        path = tmp_path / "shader.wgsl"
        path.write_text(code)
        result = subprocess.run(
            ["naga", str(path)], capture_output=True, text=True, check=False
        )
        assert result.returncode == 0, result.stdout + result.stderr


def test_metal_keeps_the_local():
    program = Parser(Lexer(LOCAL_OFFSET).tokens).parse()
    assert legalize_texel_offsets(program, "metal") == 0


@pytest.mark.parametrize(
    "body",
    [
        "ivec2 o = ivec2(1, 0); o.x = k; sum += textureOffset(colorMap, uv, o);",
        "ivec2 o = ivec2(1, 0); o += ivec2(k); sum += textureOffset(colorMap, uv, o);",
        "int d = 1; d++; sum += textureOffset(colorMap, uv, ivec2(d, 0));",
        "ivec2 o = ivec2(1, 0); bump(o); sum += textureOffset(colorMap, uv, o);",
        "ivec2 o = ivec2(k, 0); sum += textureOffset(colorMap, uv, o);",
    ],
    ids=["member-write", "compound", "increment", "inout-argument", "uniform"],
)
def test_written_or_varying_locals_stay_refused(tmp_path, body):
    declarations = "void bump(inout ivec2 v) { v.x = v.x + 1; }"
    with pytest.raises(Exception, match=GLSL_REFUSAL):
        _translate(tmp_path, _fragment(body, declarations), "opengl")


def test_a_shadowing_declaration_hides_the_constant(tmp_path):
    body = """
            ivec2 o = ivec2(1, 0);
            {
                ivec2 o = ivec2(k, 0);
                sum += textureOffset(colorMap, uv, o);
            }
            sum += textureOffset(colorMap, uv, o);
    """
    with pytest.raises(Exception, match=GLSL_REFUSAL):
        _translate(tmp_path, _fragment(body), "opengl")
    program = Parser(Lexer(_fragment(body)).tokens).parse()
    legalize_texel_offsets(program, "opengl")
    entry = list(program.stages.values())[0].entry_point
    statements = (
        entry.body.statements if isinstance(entry.body, A.BlockNode) else entry.body
    )
    inner_call = statements[2].statements[1].expression.value
    outer_call = statements[3].expression.value
    assert isinstance(inner_call.arguments[2], A.IdentifierNode)
    assert isinstance(outer_call.arguments[2], A.FunctionCallNode)


# Loops ----------------------------------------------------------------------------------------------------------
def test_metal_pcf_loops_unroll_for_glsl(tmp_path):
    code = _translate(tmp_path, PCF_METAL, "opengl", suffix="metal")
    assert "for (" not in code
    offsets = _offsets(code)
    assert len(offsets) == 9
    assert offsets[0] == "ivec2(-1, -1)" and offsets[-1] == "ivec2(1, 1)"
    assert code.count("const int i = ") == 3 and code.count("const int j = ") == 9
    _validate_glsl(tmp_path, code)


def test_metal_pcf_loops_unroll_for_spirv(tmp_path):
    code = _translate(tmp_path, PCF_METAL, "vulkan", suffix="metal")
    assert len(re.findall(r"\bOpImageSample\w+ .*\bConstOffset\b", code)) == 9
    assert "OpLoopMerge" not in code
    _validate_spirv(tmp_path, code)


def test_metal_pcf_loops_unroll_for_hlsl(tmp_path):
    code = _translate(tmp_path, PCF_METAL, "directx", suffix="metal")
    assert "for (" not in code
    assert code.count(".Sample(") == 9


def test_metal_keeps_its_loops(tmp_path):
    code = _translate(tmp_path, PCF_METAL, "metal", suffix="metal")
    assert code.count("for (") == 2


def test_module_constant_bounds_and_steps_unroll(tmp_path):
    body = """
            for (int i = -TAPS; i <= TAPS; i += 2) {
                float w = float(i);
                sum += w * textureOffset(colorMap, uv, ivec2(i, 0));
            }
    """
    code = _translate(tmp_path, _fragment(body, "const int TAPS = 2;"), "opengl")
    assert _offsets(code) == ["ivec2(-2, 0)", "ivec2(0, 0)", "ivec2(2, 0)"]
    _validate_glsl(tmp_path, code)


def test_a_loop_that_runs_zero_times_disappears(tmp_path):
    body = """
            for (int i = 3; i < 3; i++) {
                sum += textureOffset(colorMap, uv, ivec2(i, 0));
            }
    """
    code = _translate(tmp_path, _fragment(body), "opengl")
    assert "textureOffset" not in code


@pytest.mark.parametrize(
    "loop",
    [
        "for (int i = 0; i < 2; i++) { if (k > i) { break; } sum += textureOffset(colorMap, uv, ivec2(i, 0)); }",
        "for (int i = 0; i < 2; i++) { if (k > i) { continue; } sum += textureOffset(colorMap, uv, ivec2(i, 0)); }",
        "for (int i = 0; i < 2; i++) { sum += textureOffset(colorMap, uv, ivec2(i, 0)); i += k; }",
        "for (int i = 0; i < k; i++) { sum += textureOffset(colorMap, uv, ivec2(i, 0)); }",
        "for (int i = 0; i < 65; i++) { sum += textureOffset(colorMap, uv, ivec2(i, 0)); }",
        "for (uint i = 2u; i >= 0u; i--) { sum += textureOffset(colorMap, uv, ivec2(int(i), 0)); }",
        "for (int i = 0; i < 2u; i++) { sum += textureOffset(colorMap, uv, ivec2(i, 0)); }",
        "for (int i = 0; i < 2; i++) { sum += textureOffset(colorMap, uv, ivec2(i, k)); }",
    ],
    ids=[
        "break",
        "continue",
        "body-writes-induction",
        "uniform-bound",
        "too-many-trips",
        "unsigned-wrap",
        "mixed-signedness",
        "varying-component",
    ],
)
def test_loops_that_cannot_unroll_stay_refused(tmp_path, loop):
    with pytest.raises(Exception, match=GLSL_REFUSAL):
        _translate(tmp_path, _fragment(loop), "opengl")


def test_breaks_of_nested_loops_and_switches_do_not_block_unrolling(tmp_path):
    body = """
            for (int i = 0; i < 2; i++) {
                for (int n = 0; n < k; n++) { if (n > 3) { break; } }
                switch (k) { case 0: sum.x += 1.0; break; default: break; }
                sum += textureOffset(colorMap, uv, ivec2(i, 0));
            }
    """
    code = _translate(tmp_path, _fragment(body), "opengl")
    assert _offsets(code) == ["ivec2(0, 0)", "ivec2(1, 0)"]
    _validate_glsl(tmp_path, code)


def test_gathers_do_not_unroll(tmp_path):
    body = """
            for (int i = 0; i < 2; i++) {
                sum += textureGatherOffset(colorMap, uv, ivec2(i, 0));
            }
    """
    code = _translate(tmp_path, _fragment(body), "vulkan")
    assert "OpLoopMerge" in code
    assert re.search(r"OpImageGather .* Offset %", code)
    assert "OpCapability ImageGatherExtended" in code


# SPIR-V refusal -------------------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    "call",
    [
        "textureOffset(colorMap, uv, ivec2(k, 0))",
        "textureLodOffset(colorMap, uv, 0.0, ivec2(k, 0))",
        "texelFetchOffset(colorMap, ivec2(1, 2), 0, ivec2(k, 0))",
    ],
)
def test_spirv_refuses_non_constant_offsets_outside_gathers(tmp_path, call):
    with pytest.raises(Exception, match=SPIRV_REFUSAL):
        _translate(tmp_path, _fragment(f"sum += {call};"), "vulkan")

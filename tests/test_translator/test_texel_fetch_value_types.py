"""Texel fetches as typed values: integer locations from HLSL, element conversions on Metal.

HLSL ``Load`` and the texture subscript operator convert a floating-point location implicitly
(``tex.Load(int3(pos.xy, 0))``, ``ms.Load(pos.xy, i)``), and the importer used to drop the conversion, so CrossGL
``texelFetch`` received a float coordinate that GLSL refuses. The Metal emitter knew the value type of a sample but
not of a fetch, so ``half4 h = tex.read(...)`` on a ``texture2d<float>`` lacked the explicit ``half4`` conversion
MSL requires between vector types (MSL 3.2 section 2.13).
"""

import re
import shutil
import subprocess
import textwrap

import pytest

import crosstl

HLSL = textwrap.dedent("""
    Texture2D<float4> albedo : register(t0);
    Texture2D<float2> velocity : register(t1);
    Texture2DMS<float4> resolved : register(t2);
    Texture1D<float4> ramp : register(t3);

    float4 main(float4 pos : SV_Position, float level : TEXCOORD0) : SV_Target
    {
        float4 a = albedo.Load(int3(pos.xy, 0));
        float4 b = albedo.Load(int3(pos.x, pos.y, level));
        float2 v = velocity[pos.xy];
        float4 m = resolved.Load(pos.xy, 1);
        float4 r = ramp.Load(int2(pos.x, 0));
        float3 location = float3(pos.xy, 0.0);
        float4 c = albedo.Load(location);
        return a + b + float4(v, 0.0, 0.0) + m + r + c;
    }
    """)

METAL = textwrap.dedent("""
    #include <metal_stdlib>
    using namespace metal;
    struct QuadInOut { float4 position [[position]]; };
    fragment half4 shade(QuadInOut in [[stage_in]],
                         texture2d<half> albedo [[texture(0)]],
                         sampler s [[sampler(0)]]) {
        uint2 texel = uint2(in.position.xy);
        half4 fetched = albedo.read(texel);
        half4 sampled = albedo.sample(s, float2(0.5));
        return fetched + sampled;
    }
    """)

# A plain read of a depth texture is a scalar in MSL that the emitter widens to the (d, 0, 0, 1) of GLSL and HLSL.
CGL_DEPTH = textwrap.dedent("""
    shader main {
        @ binding(0)
        texture2DShadow depthMap;
        fragment {
            vec4 main(vec4 pos @ gl_FragCoord) @ gl_FragColor {
                f16vec4 shadow = texelFetch(depthMap, ivec2(pos.xy), 0);
                return vec4(shadow);
            }
        }
    }
    """)


def _translate(tmp_path, source, suffix, backend):
    path = tmp_path / f"fetch{suffix}"
    path.write_text(source, encoding="utf-8")
    return crosstl.translate(str(path), backend=backend)


def _run(tmp_path, *commands):
    for command in commands:
        result = subprocess.run(
            command, capture_output=True, text=True, timeout=120, cwd=tmp_path
        )
        assert result.returncode == 0, result.stderr[-2000:]


def test_hlsl_import_converts_floating_point_texel_locations(tmp_path):
    code = _translate(tmp_path, HLSL, ".frag.hlsl", "cgl")
    assert "texelFetch(albedo, ivec2(pos.xy), 0)" in code
    assert "texelFetch(albedo, ivec2(pos.x, pos.y), int(level))" in code
    assert "texelFetch(velocity, ivec2(pos.xy), 0)" in code
    assert "texelFetch(resolved, ivec2(pos.xy), 1)" in code
    assert "texelFetch(ramp, int(pos.x), 0)" in code
    assert "texelFetch(albedo, ivec2(location.xy), int(location.z))" in code


def test_hlsl_import_keeps_integer_texel_locations(tmp_path):
    source = textwrap.dedent("""
        Texture2D<float4> albedo : register(t0);
        float4 main(float4 pos : SV_Position) : SV_Target
        {
            int2 texel = int2(pos.xy);
            uint mip = 0;
            return albedo.Load(int3(texel, mip)) + albedo[uint2(texel)];
        }
        """)
    code = _translate(tmp_path, source, ".frag.hlsl", "cgl")
    assert "texelFetch(albedo, texel, mip)" in code
    assert "ivec2(texel)" not in code


def test_hlsl_floating_point_locations_reach_glsl(tmp_path):
    code = _translate(tmp_path, HLSL, ".frag.hlsl", "opengl")
    assert "texelFetch(albedo, ivec2(gl_FragCoord.xy), 0)" in code
    if not shutil.which("glslangValidator"):
        pytest.skip("glslangValidator is not installed")
    (tmp_path / "a.frag").write_text(code, encoding="utf-8")
    _run(tmp_path, ["glslangValidator", "a.frag"])


def test_metal_converts_fetched_texels_to_half(tmp_path):
    code = _translate(tmp_path, METAL, ".metal", "metal")
    assert re.search(r"half4 fetched =\s*half4\(albedo\.read\(", code)
    assert re.search(r"half4 sampled =\s*half4\(albedo\.sample\(", code)
    _compile_metal(tmp_path, code)


def test_metal_converts_widened_depth_reads(tmp_path):
    code = _translate(tmp_path, CGL_DEPTH, ".cgl", "metal")
    assert re.search(r"half4 shadow =\s*half4\(float4\(depthMap\.read\(", code)


def _compile_metal(tmp_path, code):
    if not shutil.which("xcrun"):
        pytest.skip("the Metal toolchain is not installed")
    (tmp_path / "a.metal").write_text(code, encoding="utf-8")
    _run(tmp_path, ["xcrun", "-sdk", "macosx", "metal", "-c", "a.metal", "-o", "a.air"])

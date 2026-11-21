"""HLSL level-of-detail queries on array textures and ``IsHelperLane()`` outside pixel shaders.

``Texture2DArray.CalculateLevelOfDetail(s, uv)`` takes the coordinate within a layer, while CrossGL's
``textureQueryLod`` takes the texture's sampling coordinate with the layer, which every generator drops again. The
importer appends a layer of 0, so the round trip writes the query DXC accepts instead of refusing it.

``IsHelperLane()`` returns false outside pixel shaders (Shader Model 6.6). CrossGL's ``gl_HelperInvocation`` exists
only in fragment stages, so a call that only non-pixel entry points reach imports as ``false``.
"""

import re
import textwrap

import pytest

import crosstl
from crosstl.translator.semantics.builtin_values import BuiltinValueUnsupportedError

ARRAY_LOD_HLSL = textwrap.dedent("""
    Texture1DArray<float4> t1 : register(t0);
    Texture2DArray<float4> t2 : register(t1);
    TextureCubeArray<float4> tc : register(t2);
    SamplerState s : register(s0);

    float4 main(float4 uv : TEXCOORD0) : SV_Target {
        float a = t1.CalculateLevelOfDetail(s, uv.x);
        float b = t2.CalculateLevelOfDetailUnclamped(s, uv.xy);
        float c = tc.CalculateLevelOfDetail(s, uv.xyz);
        float d = t2.CalculateLevelOfDetail(s, uv.xyz);
        float e = t2.CalculateLevelOfDetail(s, uv.w);
        return float4(a, b, c, d + e);
    }
""")

PLAIN_LOD_HLSL = textwrap.dedent("""
    Texture2D<float4> t : register(t0);
    SamplerState s : register(s0);

    float4 main(float2 uv : TEXCOORD0) : SV_Target {
        return float4(t.CalculateLevelOfDetail(s, uv), 0.0, 0.0, 1.0);
    }
""")


def _translate(tmp_path, source, backend, name="probe.hlsl"):
    path = tmp_path / name
    path.write_text(source)
    return crosstl.translate(str(path), backend=backend)


def _lines_with(text, needle):
    return [line.strip() for line in text.splitlines() if needle in line]


def test_array_lod_queries_import_with_a_zero_layer(tmp_path):
    crossgl = _translate(tmp_path, ARRAY_LOD_HLSL, "cgl")
    queries = _lines_with(crossgl, "textureQueryLod")
    assert queries == [
        "float a = textureQueryLod(t1, s, vec2(uv.x, 0.0)).x;",
        "float b = textureQueryLod(t2, s, vec3(uv.xy, 0.0)).y;",
        "float c = textureQueryLod(tc, s, vec4(uv.xyz, 0.0)).x;",
        # HLSL truncates a wider coordinate and splats a scalar one.
        "float d = textureQueryLod(t2, s, vec3(uv.xyz.xy, 0.0)).x;",
        "float e = textureQueryLod(t2, s, vec3(vec2(uv.w), 0.0)).x;",
    ]


def test_array_lod_queries_round_trip_to_in_layer_coordinates(tmp_path):
    hlsl = re.sub(r"\s+", " ", _translate(tmp_path, ARRAY_LOD_HLSL, "directx"))
    assert "t1.CalculateLevelOfDetail(s, (float2(uv.x, 0.0)).x)" in hlsl
    assert "t2.CalculateLevelOfDetailUnclamped(s, (float3(uv.xy, 0.0)).xy)" in hlsl
    assert "tc.CalculateLevelOfDetail(s, (float4(uv.xyz, 0.0)).xyz)" in hlsl


def test_array_lod_queries_translate_to_glsl_in_layer_coordinates(tmp_path):
    glsl = re.sub(r"\s+", " ", _translate(tmp_path, ARRAY_LOD_HLSL, "opengl"))
    assert re.search(r"textureQueryLod\(\w+, \(?vec2\(uv\.x, 0\.0\)\)?\.x\)", glsl)
    assert re.search(r"textureQueryLod\(\w+, \(?vec3\(uv\.xy, 0\.0\)\)?\.xy\)", glsl)


def test_non_array_lod_query_keeps_its_coordinate(tmp_path):
    crossgl = _translate(tmp_path, PLAIN_LOD_HLSL, "cgl")
    assert _lines_with(crossgl, "textureQueryLod") == [
        "return vec4(textureQueryLod(t, s, uv).x, 0.0, 0.0, 1.0);"
    ]


HELPER_LANE_HLSL = textwrap.dedent("""
    RWStructuredBuffer<int> outp : register(u0);

    int computeOnly() {
        return IsHelperLane() ? 1 : 0;
    }

    [numthreads(1, 1, 1)]
    void main(uint3 id : SV_DispatchThreadID) {
        bool r = IsHelperLane();
        outp[id.x] = (r ? 10 : 0) + computeOnly();
    }
""")

PIXEL_HELPER_LANE_HLSL = textwrap.dedent("""
    int pixelHelper() {
        return IsHelperLane() ? 1 : 0;
    }

    float4 main() : SV_Target {
        bool r = IsHelperLane();
        return float4(r ? 1.0 : 0.0, float(pixelHelper()), 0.0, 1.0);
    }
""")


def test_is_helper_lane_outside_pixel_shaders_imports_as_false(tmp_path):
    crossgl = _translate(tmp_path, HELPER_LANE_HLSL, "cgl")
    assert "gl_HelperInvocation" not in crossgl
    assert _lines_with(crossgl, "? 1 : 0") == ["return false ? 1 : 0;"]
    assert _lines_with(crossgl, "bool r") == ["bool r = false;"]
    hlsl = _translate(tmp_path, HELPER_LANE_HLSL, "directx")
    assert "IsHelperLane" not in hlsl


def test_is_helper_lane_in_pixel_shaders_reads_the_helper_flag(tmp_path):
    crossgl = _translate(tmp_path, PIXEL_HELPER_LANE_HLSL, "cgl")
    assert _lines_with(crossgl, "bool r") == ["bool r = gl_HelperInvocation;"]
    assert _lines_with(crossgl, "? 1 : 0") == ["return gl_HelperInvocation ? 1 : 0;"]
    hlsl = _translate(tmp_path, PIXEL_HELPER_LANE_HLSL, "directx")
    assert hlsl.count("IsHelperLane()") == 2


def test_is_helper_lane_in_a_helper_both_kinds_of_stage_reach_fails_closed(tmp_path):
    # A helper that a pixel shader reaches keeps gl_HelperInvocation, and CrossGL checks every stage reaching it.
    source = textwrap.dedent("""
        RWStructuredBuffer<int> outp : register(u0);

        int laneFlag() {
            return IsHelperLane() ? 1 : 0;
        }

        [numthreads(1, 1, 1)]
        void cs_main(uint3 id : SV_DispatchThreadID) {
            outp[id.x] = laneFlag();
        }

        float4 ps_main() : SV_Target {
            return float4(float(laneFlag()), 0.0, 0.0, 1.0);
        }
    """)
    crossgl = _translate(tmp_path, source, "cgl")
    assert _lines_with(crossgl, "? 1 : 0") == ["return gl_HelperInvocation ? 1 : 0;"]
    with pytest.raises(BuiltinValueUnsupportedError, match="gl_HelperInvocation"):
        _translate(tmp_path, source, "directx")

"""HLSL texture samples return the texture's declared element type.

``Texture2D<float2>.Sample`` returns ``float2`` and ``Texture2D<float>.Sample`` returns ``float``
(HLSL ``Texture2D::Sample``, "Return Value": the texture's template type). CrossGL ``texture``
returns four components, as in GLSL. The importer narrows every sample to the element width.
Without it, a scalar assigned to ``float4`` read ``(r, 0, 0, 1)`` instead of HLSL's splat
``(r, r, r, r)``, and narrower element types produced initializers that GLSL refused and MSL
rejected.
"""

from __future__ import annotations

import re
import shutil
import subprocess

import pytest

from tests.test_translator.test_comparison_samplers import (
    _compile_hlsl,
    _compile_metal,
    _validate_glsl,
)
from tests.test_translator.test_spirv_separate_textures import _translate, _validate

HLSL_NARROW = """Texture2D<float2> velocity : register(t0);
Texture2D<float> depthTex : register(t1);
Texture2D plain : register(t2);
Texture2DArray<float3> layers : register(t3);
SamplerState samp : register(s0);
float4 main(float2 uv : TEXCOORD0) : SV_Target {
    float2 v = velocity.Sample(samp, uv);
    float d = depthTex.SampleLevel(samp, uv, 0);
    float vy = velocity.Sample(samp, uv).y;
    float2 vo = velocity.Sample(samp, uv, int2(1, 1));
    float2 vb = velocity.SampleBias(samp, uv, 0.5);
    float2 vg = velocity.SampleGrad(samp, uv, float2(0.1, 0.0), float2(0.0, 0.1));
    float4 splat = depthTex.Sample(samp, uv);
    float3 a = layers.SampleLevel(samp, float3(uv, 0), 0);
    float4 p = plain.Sample(samp, uv);
    float s = depthTex.Sample(samp, uv) * 2.0 + velocity.Sample(samp, uv).x;
    return splat + p + float4(a, v.x + d + vy + vo.x + vb.y + vg.x + s);
}
"""


def _crossgl_lines(tmp_path):
    crossgl = _translate(tmp_path, HLSL_NARROW, ".hlsl", "cgl")
    return {
        match.group(1): match.group(2)
        for match in re.finditer(r"^\s*\w+ (\w+) = (.+);$", crossgl, re.M)
    }


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("v", "texture(velocity, samp, uv).xy"),
        ("d", "textureLod(depthTex, samp, uv, 0).x"),
        ("vy", "texture(velocity, samp, uv).y"),
        ("vo", "textureOffset(velocity, samp, uv, ivec2(1, 1)).xy"),
        ("vg", "textureGrad(velocity, samp, uv, vec2(0.1, 0.0), vec2(0.0, 0.1)).xy"),
        ("splat", "texture(depthTex, samp, uv).x"),
        ("a", "textureLod(layers, samp, vec3(uv, 0), 0).xyz"),
        ("p", "texture(plain, samp, uv)"),
        (
            "s",
            "(texture(depthTex, samp, uv).x * 2.0) + texture(velocity, samp, uv).x",
        ),
    ],
)
def test_crossgl_samples_narrow_to_the_element_type(tmp_path, name, expected):
    assert _crossgl_lines(tmp_path)[name] == expected


def test_crossgl_bias_sample_narrows_to_the_element_type(tmp_path):
    assert _crossgl_lines(tmp_path)["vb"].endswith(").xy")


def test_scalar_sample_assigned_to_a_vector_splats_on_glsl(tmp_path):
    glsl = _translate(tmp_path, HLSL_NARROW, ".hlsl", "opengl")

    assert "vec4 splat = vec4(texture(depthTex, uv).x);" in glsl
    _validate_glsl(tmp_path, glsl)


def test_narrowed_samples_compile_on_metal(tmp_path):
    metal = _translate(tmp_path, HLSL_NARROW, ".hlsl", "metal")

    assert "float2 v = velocity.sample(samp, uv).xy;" in metal
    assert "float d = depthTex.sample(samp, uv, level(0)).x;" in metal
    _compile_metal(tmp_path, metal)


def test_narrowed_samples_validate_on_spirv(tmp_path):
    code = _translate(tmp_path, HLSL_NARROW, ".hlsl", "vulkan")

    assert "OpImageSampleImplicitLod" in code
    _validate(tmp_path, code)


def test_narrowed_samples_round_trip_through_hlsl(tmp_path):
    hlsl = _translate(tmp_path, HLSL_NARROW, ".hlsl", "directx")

    assert "float2 v = velocity.Sample(samp, uv).xy;" in hlsl
    _compile_hlsl(tmp_path, hlsl)


@pytest.mark.skipif(shutil.which("dxc") is None, reason="dxc is not installed")
def test_source_compiles_on_dxc(tmp_path):
    path = tmp_path / "source.hlsl"
    path.write_text(HLSL_NARROW, encoding="utf-8")
    result = subprocess.run(
        ["dxc", "-T", "ps_6_0", "-E", "main", str(path)],
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    assert result.returncode == 0, (result.stdout + result.stderr)[-2000:]

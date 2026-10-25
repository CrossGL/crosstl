"""Locals named after GLSL built-in functions that the translation calls are renamed for GLSL."""

import shutil
import subprocess

import crosstl

SOURCE = """Texture2D tex : register(t0);
SamplerState samp : register(s0);
float4 main(float4 pos : SV_Position, float2 uv : TEXCOORD0) : SV_Target
{
    float2 textureSize;
    tex.GetDimensions(textureSize.x, textureSize.y);
    float4 texture = tex.Sample(samp, uv + 1.0 / textureSize);
    float mod = fmod(texture.x, 2.0);
    return texture * mod;
}
"""


def test_hlsl_locals_do_not_hide_the_glsl_builtins(tmp_path):
    path = tmp_path / "shader.hlsl"
    path.write_text(SOURCE)
    glsl = crosstl.translate(str(path), backend="opengl", format_output=False)
    # A local named texture used to hide texture() in its scope ("can't use function syntax on variable").
    assert "vec2 textureSize_;" in glsl and "textureSize(tex, 0)" in glsl
    assert "vec4 texture_ = texture(tex, (uv + (1.0 / textureSize_)));" in glsl
    assert "float mod_ = " in glsl
    if shutil.which("glslangValidator"):
        (tmp_path / "out.frag").write_text(glsl)
        result = subprocess.run(
            ["glslangValidator", "-S", "frag", "out.frag"],
            cwd=tmp_path,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stdout + result.stderr

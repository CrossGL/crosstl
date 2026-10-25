"""Locals named after the resource-query helpers the HLSL generator defines are renamed."""

import shutil
import subprocess

import crosstl

SOURCE = """Texture2D tex : register(t0);
SamplerState samp : register(s0);
float4 main(float4 pos : SV_Position, float2 uv : TEXCOORD0) : SV_Target
{
    float2 textureSize;
    tex.GetDimensions(textureSize.x, textureSize.y);
    return tex.Sample(samp, uv + 1.0 / textureSize);
}
"""


def test_a_local_named_texture_size_does_not_hide_the_helper(tmp_path):
    path = tmp_path / "shader.frag.hlsl"
    path.write_text(SOURCE)
    hlsl = crosstl.translate(str(path), backend="directx", source_backend="directx")
    # The query lowers to a textureSize helper; the local of that name used to hide it ("type 'float2' does not
    # provide a call operator").
    assert "float2 textureSize_;" in hlsl
    assert "textureSize_.x = textureSize(tex, 0).x;" in hlsl
    if shutil.which("dxc"):
        (tmp_path / "out.hlsl").write_text(hlsl)
        result = subprocess.run(
            ["dxc", "-T", "ps_6_0", "-E", "PSMain", "out.hlsl"],
            cwd=tmp_path,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stdout + result.stderr

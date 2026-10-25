"""An HLSL fragment input structure's SV_Position member is the fragment coordinate."""

import re
import shutil
import subprocess

import crosstl

SOURCE = """[[vk::input_attachment_index(0)]][[vk::binding(1)]] SubpassInput positionDepth;
struct VSOutput
{
    float4 Pos : SV_POSITION;
    [[vk::location(0)]] float2 UV : TEXCOORD0;
};
float4 main(VSOutput input) : SV_TARGET
{
    float depth = positionDepth.SubpassLoad().a;
    return float4(input.UV, depth, input.Pos.z);
}
"""


def test_subpass_loads_read_the_input_position(tmp_path):
    path = tmp_path / "shader.frag.hlsl"
    path.write_text(SOURCE)
    hlsl = crosstl.translate(
        str(path), backend="directx", source_backend="directx", format_output=False
    )
    # A second SV_Position parameter used to be added, which DXC rejects ("overlapping semantic index").
    assert re.search(r"float4 PSMain\(VSOutput input\)\s*:\s*SV_TARGET", hlsl)
    assert "positionDepth.Load(int3(int2(input.Pos.xy), 0))" in hlsl
    assert hlsl.count("SV_Position") + hlsl.count("SV_POSITION") == 1
    if shutil.which("dxc"):
        (tmp_path / "out.hlsl").write_text(hlsl)
        result = subprocess.run(
            ["dxc", "-T", "ps_6_0", "-E", "PSMain", "out.hlsl"],
            cwd=tmp_path,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stdout + result.stderr

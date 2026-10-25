"""A structure member's name never selects an HLSL system value."""

import re
import shutil
import subprocess

import crosstl

SOURCE = """#include <metal_stdlib>
using namespace metal;
struct VOut { float4 position [[position]]; float3 view; float layer; float viewport; };
vertex VOut v(uint vid [[vertex_id]]) {
    VOut out;
    out.position = float4(float(vid), 0.0, 0.0, 1.0);
    out.view = float3(1.0, 2.0, 3.0);
    out.layer = 0.5;
    out.viewport = 0.25;
    return out;
}
fragment float4 f(VOut in [[stage_in]]) { return float4(in.view * in.layer, in.viewport); }
"""


def test_varyings_named_view_and_layer_are_user_varyings(tmp_path):
    # They used to become SV_ViewID, SV_RenderTargetArrayIndex and SV_ViewportArrayIndex by name alone.
    path = tmp_path / "shader.metal"
    path.write_text(SOURCE)
    hlsl = crosstl.translate(str(path), backend="directx", format_output=False)
    assert re.search(r"float3 view\s*:\s*TEXCOORD", hlsl)
    assert re.search(r"float layer\s*:\s*TEXCOORD", hlsl)
    assert re.search(r"float viewport\s*:\s*TEXCOORD", hlsl)
    assert "SV_ViewID" not in hlsl and "SV_RenderTargetArrayIndex" not in hlsl
    assert "SV_ViewportArrayIndex" not in hlsl
    if shutil.which("dxc"):
        (tmp_path / "out.hlsl").write_text(hlsl)
        for profile, entry in (("vs_6_0", "VSMain"), ("ps_6_0", "PSMain")):
            result = subprocess.run(
                ["dxc", "-T", profile, "-E", entry, "out.hlsl"],
                cwd=tmp_path,
                capture_output=True,
                text=True,
            )
            assert result.returncode == 0, result.stdout + result.stderr

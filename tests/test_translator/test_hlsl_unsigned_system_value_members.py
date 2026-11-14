"""Structure members carrying system values that DXIL delivers as uint are declared uint."""

import re
import shutil
import subprocess

import crosstl
from crosstl.translator.codegen.directx_codegen import HLSLCodeGen

SOURCE = """struct VSOutput { float4 Pos : SV_POSITION; };
struct GSOutput
{
    float4 Pos : SV_POSITION;
    int Layer : SV_RenderTargetArrayIndex;
    int Viewport : SV_ViewportArrayIndex;
};
[maxvertexcount(3)]
[instance(2)]
void main(triangle VSOutput input[3], uint InvocationID : SV_GSInstanceID,
          inout TriangleStream<GSOutput> outStream)
{
    for (int i = 0; i < 3; i++)
    {
        GSOutput output;
        output.Pos = input[i].Pos;
        output.Layer = InvocationID;
        output.Viewport = 0;
        outStream.Append(output);
    }
    outStream.RestartStrip();
}
"""


def test_int_layer_and_viewport_members_are_uint(tmp_path):
    # Vulkan HLSL accepts int; DXIL requires uint ("SV_RenderTargetArrayIndex must be uint").
    path = tmp_path / "shader.geom"
    path.write_text(SOURCE)
    hlsl = crosstl.translate(
        str(path), backend="directx", source_backend="directx", format_output=False
    )
    assert re.search(r"uint Layer\s*:\s*SV_RenderTargetArrayIndex;", hlsl)
    assert re.search(r"uint Viewport\s*:\s*SV_ViewportArrayIndex;", hlsl)
    if shutil.which("dxc"):
        (tmp_path / "out.hlsl").write_text(hlsl)
        result = subprocess.run(
            ["dxc", "-T", "gs_6_0", "-E", "GSMain", "out.hlsl"],
            cwd=tmp_path,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stdout + result.stderr


def test_only_int_members_of_those_system_values_change():
    codegen = HLSLCodeGen()
    assert codegen.hlsl_unsigned_system_value_type("int", "SV_PrimitiveID") == "uint"
    assert codegen.hlsl_unsigned_system_value_type("uint", "SV_PrimitiveID") == "uint"
    assert codegen.hlsl_unsigned_system_value_type("int", "TEXCOORD0") == "int"
    assert codegen.hlsl_unsigned_system_value_type("float4", "SV_Position") == "float4"


STENCIL_EXPORT = """
shader Probe {
    struct FragmentInput {
        vec2 uv;
    }
    struct FragmentOutput {
        vec4 color;
        int stencil @ gl_FragStencilRefARB;
    }
    fragment {
        FragmentOutput main(FragmentInput input) {
            FragmentOutput output;
            output.color = vec4(input.uv, 0.0, 1.0);
            output.stencil = int(input.uv.x * 255.0);
            return output;
        }
    }
}
"""


def test_int_stencil_export_member_is_uint(tmp_path):
    # GL_ARB_shader_stencil_export declares gl_FragStencilRefARB int; DXIL requires SV_StencilRef uint.
    path = tmp_path / "shader.cgl"
    path.write_text(STENCIL_EXPORT)
    hlsl = crosstl.translate(str(path), backend="directx", format_output=False)
    assert re.search(r"uint stencil\s*:\s*SV_StencilRef;", hlsl)
    assert HLSLCodeGen().hlsl_unsigned_system_value_type("int", "SV_StencilRef") == (
        "uint"
    )
    if shutil.which("dxc"):
        (tmp_path / "out.hlsl").write_text(hlsl)
        result = subprocess.run(
            ["dxc", "-T", "ps_6_0", "-E", "PSMain", "out.hlsl"],
            cwd=tmp_path,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stdout + result.stderr

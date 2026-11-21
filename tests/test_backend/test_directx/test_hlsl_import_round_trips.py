"""HLSL imports that lost a promise or a texel type on the way through CrossGL.

The coverage round trip (HLSL to CrossGL to HLSL) reported each of these as wrong. Where DXC is on ``PATH`` the
round-tripped HLSL is compiled too.
"""

import re
import shutil
import subprocess
import textwrap

import pytest

import crosstl


def _translate(tmp_path, source, backend):
    path = tmp_path / "probe.hlsl"
    path.write_text(textwrap.dedent(source).strip() + "\n", encoding="utf-8")
    return crosstl.translate(str(path), backend=backend)


def _compile(tmp_path, generated, profile, entry):
    compiler = shutil.which("dxc")
    if compiler is None:
        return
    source = tmp_path / "round_trip.hlsl"
    source.write_text(generated, encoding="utf-8")
    command = [compiler, "-T", profile, "-HV", "2021", "-E", entry, str(source)]
    command += ["-Fo", str(tmp_path / "round_trip.bin")]
    result = subprocess.run(
        command, check=False, capture_output=True, text=True, timeout=120
    )
    assert result.returncode == 0, result.stdout + result.stderr


CONSERVATIVE_DEPTH = """
struct FragmentOutput {
    float4 color : SV_Target;
    float depth : SEMANTIC;
};

FragmentOutput main(float2 uv : TEXCOORD0) {
    FragmentOutput output;
    output.color = float4(uv, 0.0, 1.0);
    output.depth = uv.x * 0.5 + 0.25;
    return output;
}
"""


@pytest.mark.parametrize(
    ("semantic", "layout"),
    [("SV_DepthGreaterEqual", "depth_greater"), ("SV_DepthLessEqual", "depth_less")],
)
def test_conservative_depth_semantic_keeps_its_direction(tmp_path, semantic, layout):
    source = CONSERVATIVE_DEPTH.replace("SEMANTIC", semantic)
    crossgl = _translate(tmp_path, source, "cgl")
    assert f"float depth @ gl_FragDepth @ {layout};" in crossgl

    generated = _translate(tmp_path, source, "directx")
    assert re.search(rf"float depth\s*:\s*{semantic}\s*;", generated)
    _compile(tmp_path, generated, "ps_6_0", "PSMain")


def test_plain_depth_semantic_has_no_layout(tmp_path):
    crossgl = _translate(
        tmp_path, CONSERVATIVE_DEPTH.replace("SEMANTIC", "SV_Depth"), "cgl"
    )
    assert "float depth @ gl_FragDepth;" in crossgl


def test_depth_return_semantic_keeps_its_direction(tmp_path):
    generated = _translate(
        tmp_path,
        """
        float main(float2 uv : TEXCOORD0) : SV_DepthLessEqual {
            return uv.x * 0.5;
        }
        """,
        "directx",
    )
    assert "SV_DepthLessEqual" in generated
    assert "SV_DEPTH" not in generated


@pytest.mark.parametrize("texture", ["Texture2D", "Texture2DArray"])
def test_gather_raw_texture_keeps_its_scalar_texel(tmp_path, texture):
    coordinate = "float2" if texture == "Texture2D" else "float3"
    source = f"""
        RWStructuredBuffer<float> fin : register(u0);
        RWStructuredBuffer<uint4> outp : register(u1);
        {texture}<uint> obj : register(t0);
        {texture}<uint> plain : register(t1);
        SamplerState smp : register(s0);
        [numthreads(1, 1, 1)]
        void main(uint3 id : SV_DispatchThreadID) {{
            uint i = id.x;
            {coordinate} a1 = {coordinate}(fin[i]);
            outp[i] = obj.GatherRaw(smp, a1) + plain.GatherRed(smp, a1);
        }}
    """
    crossgl = _translate(tmp_path, source, "cgl")
    # Only the GatherRaw receiver keeps the HLSL spelling; other textures stay CrossGL texture types.
    assert f"{texture}<uint> obj;" in crossgl
    assert f"{texture}<uint> plain;" not in crossgl

    generated = _translate(tmp_path, source, "directx")
    assert re.search(rf"{texture}<uint>\s+obj\s*:", generated)
    _compile(tmp_path, generated, "cs_6_7", "CSMain")


def test_nested_sign_operators_are_not_read_as_increments(tmp_path):
    # nested-unary: '-(-x)' was written to CrossGL as '--x', a decrement.
    crossgl = _translate(
        tmp_path,
        """
        RWStructuredBuffer<float> outp : register(u0);
        [numthreads(1, 1, 1)]
        void main(uint3 id : SV_DispatchThreadID) {
            float x = outp[id.x];
            float a = -(-x);
            float b = +(+x);
            float c = -(-(-x));
            outp[id.x] = a + b + c;
        }
        """,
        "cgl",
    )
    assert "float a = -(-x);" in crossgl
    assert "float b = +(+x);" in crossgl
    assert "float c = -(-(-x));" in crossgl
    assert "--x" not in crossgl and "++x" not in crossgl

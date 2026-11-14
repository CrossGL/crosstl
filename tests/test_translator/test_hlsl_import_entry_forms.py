"""HLSL entry forms that DXC accepts: a scalar returned as a vector, and [numthreads] on a helper."""

import shutil
import subprocess

import crosstl

RETURN_SPLAT = """Texture2D t : register(t0);
SamplerState s : register(s0);
float4 main(float2 uv : TEXCOORD0) : SV_TARGET
{
    float result = t.Sample(s, uv).r;
    int n = 2;
    return result / float(n);
}
"""
NUMTHREADS_HELPER = """RWStructuredBuffer<float> values : register(u0);
[numthreads(16, 1, 1)]
bool positive(float value)
{
    return value > 0.0;
}
[numthreads(16, 1, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
    if (positive(values[id.x]))
    {
        values[id.x] = 1.0;
    }
}
"""


def translate(tmp_path, source, backend, suffix):
    path = tmp_path / f"shader{suffix}"
    path.write_text(source)
    return crosstl.translate(
        str(path), backend=backend, source_backend="directx", format_output=False
    )


def _glslang(tmp_path, glsl, stage):
    if shutil.which("glslangValidator"):
        (tmp_path / f"out.{stage}").write_text(glsl)
        result = subprocess.run(
            ["glslangValidator", "-S", stage, f"out.{stage}"],
            cwd=tmp_path,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stdout + result.stderr


def test_a_scalar_returned_as_a_vector_splats_in_glsl(tmp_path):
    # HLSL converts the float to the float4 return type; GLSL needs the constructor ("cannot convert").
    glsl = translate(tmp_path, RETURN_SPLAT, "opengl", ".frag.hlsl")
    assert "fragColor = vec4((result / float(n)));" in glsl
    _glslang(tmp_path, glsl, "frag")


def test_numthreads_on_a_value_returning_helper_does_not_make_an_entry(tmp_path):
    # The helper used to become the compute entry (its parameters turned into uniform blocks).
    cgl = translate(tmp_path, NUMTHREADS_HELPER, "cgl", ".comp.hlsl")
    assert cgl.count("compute {") == 1
    assert (
        "bool positive(float value) {" in " ".join(cgl.split())
        or "bool positive(float value)" in cgl
    )
    glsl = translate(tmp_path, NUMTHREADS_HELPER, "opengl", ".comp.hlsl")
    assert "bool positive(float value)" in glsl and "positive_value" not in glsl
    _glslang(tmp_path, glsl, "comp")
    metal = translate(tmp_path, NUMTHREADS_HELPER, "metal", ".comp.hlsl")
    assert "numthreads" not in metal
    hlsl = translate(tmp_path, NUMTHREADS_HELPER, "directx", ".comp.hlsl")
    if shutil.which("dxc"):
        (tmp_path / "out.hlsl").write_text(hlsl)
        result = subprocess.run(
            ["dxc", "-T", "cs_6_0", "-E", "CSMain", "out.hlsl"],
            cwd=tmp_path,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stdout + result.stderr

"""Metal matrix shape conversions go through a helper that takes the matrix by value.

``(float3x3)mul(view, model)`` in HLSL keeps the upper-left 3x3 of the product. The Metal generator spelled the
conversion as ``float3x3(view * model[0].xyz, ...)``: the column index bound to ``model`` alone, which compiled to a
matrix-times-vector type error (sascha-vulkan ``instancing.vert``), and the argument was evaluated once per column.
"""

import re
import shutil
import subprocess

import pytest

import crosstl

HLSL = """
cbuffer UBO : register(b0)
{
    float4x4 projection;
    float4x4 modelview;
};

struct VSInput
{
    [[vk::location(0)]] float3 Pos : POSITION0;
    [[vk::location(1)]] float3 Normal : NORMAL0;
};

struct VSOutput
{
    float4 Pos : SV_POSITION;
    [[vk::location(0)]] float3 Normal : NORMAL0;
};

float4x4 rotation(float angle)
{
    float s = sin(angle);
    float c = cos(angle);
    return float4x4(c, 0.0, s, 0.0, 0.0, 1.0, 0.0, 0.0, -s, 0.0, c, 0.0, 0.0, 0.0, 0.0, 1.0);
}

VSOutput main(VSInput input)
{
    VSOutput output = (VSOutput)0;
    float4x4 model = rotation(input.Pos.x);
    output.Normal = mul((float3x3)mul(modelview, model), input.Normal);
    output.Normal += mul((float3x3)rotation(input.Pos.y), input.Normal);
    float4x3 narrowed = (float4x3)modelview;
    output.Pos = mul(projection, mul(modelview, float4(input.Pos, 1.0))) + float4(narrowed[3], 0.0);
    return output;
}
"""


def _translate(tmp_path):
    path = tmp_path / "shader.vert.hlsl"
    path.write_text(HLSL)
    return crosstl.translate(str(path), backend="metal", format_output=False)


def test_conversions_call_one_helper_per_shape(tmp_path):
    code = _translate(tmp_path)
    assert (
        "inline float3x3 __crossgl_resize_float4x4_to_float3x3(float4x4 m) {\n"
        "    return float3x3(m[0].xyz, m[1].xyz, m[2].xyz);\n}\n"
    ) in code
    assert re.search(
        r"__crossgl_resize_float4x4_to_float3x3\([^;]*modelview \* model\)", code
    )
    # The call converted to a 3x3 is evaluated once.
    assert len(re.findall(r"\brotation\(input\.Pos\.y\)", code)) == 1


def test_converted_programs_compile(tmp_path):
    code = _translate(tmp_path)
    if shutil.which("xcrun") is None:
        pytest.skip("the Metal toolchain is not installed")
    found = subprocess.run(
        ["xcrun", "-sdk", "macosx", "--find", "metal"], capture_output=True
    )
    if found.returncode != 0:
        pytest.skip("the Metal toolchain is not installed")
    artifact = tmp_path / "out.metal"
    artifact.write_text(code)
    result = subprocess.run(
        ["xcrun", "-sdk", "macosx", "metal", "-c", str(artifact)]
        + ["-o", str(tmp_path / "out.air")],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr

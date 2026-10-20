"""HLSL ``ConstantBuffer<T>`` resources reach SPIR-V and Metal as one constant buffer each."""

import re
import shutil
import subprocess

import pytest

import crosstl
from crosstl.translator.codegen.metal_refusals import MetalLoweringRefused
from crosstl.translator.codegen.spirv_refusals import SpirvLoweringRefused

CONSTANT_BUFFERS_HLSL = """
struct Camera { float4x4 projection; float4 eye; };
[[vk::binding(0, 0)]] ConstantBuffer<Camera> camera : register(b0);
struct Material { float4 tint; float roughness; };
[[vk::binding(1, 0)]] ConstantBuffer<Material> material : register(b1);
struct VSOutput { float4 Pos : SV_POSITION; float3 Tint : TEXCOORD0; float Rough : TEXCOORD1; };
VSOutput main(float3 pos : POSITION0) {
    VSOutput output = (VSOutput) 0;
    output.Pos = mul(camera.projection, float4(pos, 1.0)) + camera.eye;
    output.Tint = material.tint.xyz * material.tint.w;
    output.Rough = material.roughness;
    return output;
}
"""

CONSTANT_BUFFER_ARRAY_HLSL = """
struct Light { float4 color; };
ConstantBuffer<Light> lights[4] : register(b0);
float4 main(float2 uv : TEXCOORD0) : SV_Target { return lights[1].color; }
"""


def _translate(tmp_path, source, backend, name="shader.vert.hlsl"):
    path = tmp_path / name
    path.write_text(source)
    return crosstl.translate(str(path), backend=backend, format_output=False)


def _run(command):
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


def test_import_keeps_constant_buffers_out_of_the_program_constants(tmp_path):
    crossgl = _translate(tmp_path, CONSTANT_BUFFERS_HLSL, "cgl")

    assert "ConstantBuffer<Camera> camera;" in crossgl
    assert "hlsl_program_constant" not in crossgl


def test_spirv_declares_each_constant_buffer_as_a_uniform_block(tmp_path):
    spirv = _translate(tmp_path, CONSTANT_BUFFERS_HLSL, "vulkan")

    names = dict(re.findall(r'OpName %(\w+) "(\w+)"', spirv))
    variables = {
        names.get(var, var): pointer
        for var, pointer in re.findall(r"%(\w+) = OpVariable %(\w+) Uniform\b", spirv)
    }
    assert set(variables) == {"camera", "material"}
    decorations = {
        names.get(var, var): (int(dset), int(binding))
        for var, dset, binding in re.findall(
            r"OpDecorate %(\w+) DescriptorSet (\d+)\s+OpDecorate %\1 Binding (\d+)",
            spirv,
        )
    }
    assert decorations == {"camera": (0, 0), "material": (0, 1)}
    assert len(re.findall(r"OpDecorate %\w+ Block\b", spirv)) == 2
    assert "Unknown type" not in spirv

    if shutil.which("spirv-as") and shutil.which("spirv-val"):
        asm, binary = tmp_path / "out.spvasm", tmp_path / "out.spv"
        asm.write_text(spirv)
        _run(["spirv-as", "--target-env", "vulkan1.1", str(asm), "-o", str(binary)])
        _run(["spirv-val", "--target-env", "vulkan1.1", str(binary)])


def test_metal_binds_each_constant_buffer_as_a_constant_reference(tmp_path):
    metal = _translate(tmp_path, CONSTANT_BUFFERS_HLSL, "metal")

    assert re.search(r"constant Camera\s*&\s*camera \[\[buffer\(0\)\]\]", metal)
    assert re.search(r"constant Material\s*&\s*material \[\[buffer\(1\)\]\]", metal)
    assert "ConstantBuffer" not in metal
    assert "HlslProgramConstants" not in metal
    assert "material.roughness" in metal

    if shutil.which("xcrun") is None:
        return
    found = subprocess.run(
        ["xcrun", "-sdk", "macosx", "--find", "metal"], capture_output=True
    )
    if found.returncode != 0:
        return
    source = tmp_path / "out.metal"
    source.write_text(metal)
    _run(
        ["xcrun", "-sdk", "macosx", "metal", "-c", str(source)]
        + ["-o", str(tmp_path / "out.air")]
    )


def test_hlsl_round_trip_keeps_the_constant_buffer_declarations(tmp_path):
    hlsl = _translate(tmp_path, CONSTANT_BUFFERS_HLSL, "directx")

    assert "ConstantBuffer<Camera> camera : register(b0);" in hlsl
    assert "ConstantBuffer<Material> material : register(b1);" in hlsl


@pytest.mark.parametrize(
    "backend, error, reason",
    [
        (
            "vulkan",
            SpirvLoweringRefused,
            "array of constant buffers 'lights': the SPIR-V emitter does not declare descriptor arrays",
        ),
        (
            "metal",
            MetalLoweringRefused,
            "array of constant buffers 'lights': Metal binds an array of buffers only through an argument buffer",
        ),
    ],
)
def test_arrays_of_constant_buffers_are_refused(tmp_path, backend, error, reason):
    with pytest.raises(error, match=re.escape(reason)):
        _translate(tmp_path, CONSTANT_BUFFER_ARRAY_HLSL, backend, "shader.frag.hlsl")

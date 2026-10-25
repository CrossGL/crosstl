"""Arrays of constant buffers (descriptor arrays of uniform blocks) and non-uniform resource indices.

A GLSL array of uniform blocks and an HLSL ``ConstantBuffer<T>`` array bind one buffer per element. CrossGL spells
both as ``ConstantBuffer<T> name[N]``; the GLSL importer used to produce ``uniform T name[N]`` instead, which every
emitter reads as one buffer holding an array, and SPIR-V refused descriptor arrays of uniform blocks outright.
``NonUniformResourceIndex`` and ``nonuniformEXT`` are the same qualifier; the GLSL emitter used to copy the HLSL
spelling verbatim and the Metal emitter the GLSL one.
"""

from __future__ import annotations

import re
import shutil
import subprocess

import pytest

import crosstl

GLSL_BLOCK_ARRAY = """#version 460
#extension GL_EXT_nonuniform_qualifier : enable
layout(location = 0) flat in uint idx;
layout(location = 0) out vec4 o_color;
layout(binding = 4) uniform Weights
{
    vec4 data[4];
} weights[];
layout(set = 1, binding = 2) uniform Fixed
{
    vec4 bias;
} fixed_blocks[3];
void main() {
    o_color = weights[nonuniformEXT(idx)].data[1] + fixed_blocks[idx].bias;
}
"""

HLSL_CONSTANT_BUFFERS = """
struct W { float4 data[4]; };
ConstantBuffer<W> single : register(b4, space1);
ConstantBuffer<W> runtime[] : register(b0, space2);
ConstantBuffer<W> bounded[3] : register(b1, space0);
float4 main(nointerpolation uint idx : TEXCOORD0) : SV_Target {
    return single.data[0] + runtime[NonUniformResourceIndex(idx)].data[1] + bounded[idx].data[2];
}
"""

GLSL_TEXTURE_ARRAY = """#version 460
#extension GL_EXT_nonuniform_qualifier : enable
layout(location = 0) flat in uint idx;
layout(location = 1) in vec2 uv;
layout(location = 0) out vec4 o_color;
layout(binding = 0) uniform sampler2D tex[4];
void main() { o_color = texture(tex[nonuniformEXT(idx)], uv); }
"""

CROSSGL_TEXTURE_ARRAY = """
shader main {
    texture2D textures[4] @binding(0);
    sampler linearSampler @binding(4);
    fragment {
        vec4 main(uint idx @location(0) @flat) @location(0) {
            return sample(textures[NonUniformResourceIndex(idx)], linearSampler, vec2(0.5, 0.5));
        }
    }
}
"""


def _translate(tmp_path, name, source, backend):
    path = tmp_path / name
    path.write_text(source, encoding="utf-8")
    return crosstl.translate(str(path), backend=backend, format_output=False)


def _validate_spirv(tmp_path, assembly):
    if not (shutil.which("spirv-as") and shutil.which("spirv-val")):
        pytest.skip("SPIRV-Tools are not on PATH")
    source = tmp_path / "out.spvasm"
    binary = tmp_path / "out.spv"
    source.write_text(assembly, encoding="utf-8")
    for command in (
        ["spirv-as", "--target-env", "vulkan1.1", str(source), "-o", str(binary)],
        ["spirv-val", "--target-env", "vulkan1.1", str(binary)],
    ):
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        assert result.returncode == 0, result.stdout + result.stderr


def _named(assembly, name):
    match = re.search(rf'OpName %(\w+) "{re.escape(name)}"', assembly)
    assert match, assembly
    return match.group(1)


def _slot(assembly, name):
    variable = _named(assembly, name)
    descriptor_set = re.search(rf"OpDecorate %{variable} DescriptorSet (\d+)", assembly)
    binding = re.search(rf"OpDecorate %{variable} Binding (\d+)", assembly)
    assert descriptor_set and binding, assembly
    return int(descriptor_set.group(1)), int(binding.group(1))


def test_glsl_block_arrays_import_as_constant_buffer_arrays(tmp_path):
    crossgl = _translate(tmp_path, "blocks.frag", GLSL_BLOCK_ARRAY, "cgl")
    assert re.search(
        r"ConstantBuffer<Weights> weights\[\] @binding\(4\) @buffer_layout\(std140\);",
        crossgl,
    )
    assert re.search(
        r"ConstantBuffer<Fixed> fixed_blocks\[3\] @set\(1\) @binding\(2\) @buffer_layout\(std140\);",
        crossgl,
    )
    assert "uniform Weights" not in crossgl


def test_glsl_block_arrays_become_spirv_descriptor_arrays(tmp_path):
    assembly = _translate(tmp_path, "blocks.frag", GLSL_BLOCK_ARRAY, "vulkan")
    weights = _named(assembly, "weights")
    pointer = re.search(rf"%{weights} = OpVariable %(\w+) Uniform", assembly)
    assert pointer, assembly
    array = re.search(
        rf"%{pointer.group(1)} = OpTypePointer Uniform %(\w+)", assembly
    ).group(1)
    block = re.search(rf"%{array} = OpTypeRuntimeArray %(\w+)", assembly).group(1)
    assert f"OpDecorate %{block} Block" in assembly
    fixed = _named(assembly, "fixed_blocks")
    fixed_pointer = re.search(rf"%{fixed} = OpVariable %(\w+) Uniform", assembly)
    fixed_array = re.search(
        rf"%{fixed_pointer.group(1)} = OpTypePointer Uniform %(\w+)", assembly
    ).group(1)
    assert re.search(rf"%{fixed_array} = OpTypeArray %\w+ %\w+", assembly)
    assert _slot(assembly, "weights") == (0, 4)
    assert _slot(assembly, "fixed_blocks") == (1, 2)
    for capability in (
        "RuntimeDescriptorArray",
        "ShaderNonUniform",
        "UniformBufferArrayNonUniformIndexing",
    ):
        assert f"OpCapability {capability}" in assembly
    assert 'OpExtension "SPV_EXT_descriptor_indexing"' in assembly
    chain = re.search(rf"%(\w+) = OpAccessChain %\w+ %{weights} %(\w+)", assembly)
    assert chain, assembly
    assert f"OpDecorate %{chain.group(1)} NonUniform" in assembly
    _validate_spirv(tmp_path, assembly)


def test_glsl_block_arrays_keep_their_block_names_in_glsl(tmp_path):
    glsl = _translate(tmp_path, "blocks.frag", GLSL_BLOCK_ARRAY, "opengl")
    assert re.search(r"uniform Weights \{", glsl)
    assert re.search(r"\} weights\[\];", glsl)
    assert re.search(r"uniform Fixed \{", glsl)
    assert "struct Weights" not in glsl


def test_glsl_block_arrays_become_hlsl_constant_buffer_arrays(tmp_path):
    hlsl = _translate(tmp_path, "blocks.frag", GLSL_BLOCK_ARRAY, "directx")
    assert re.search(r"ConstantBuffer<Weights> weights\[\] : register\(b4\);", hlsl)
    assert "NonUniformResourceIndex(" in hlsl


def test_hlsl_constant_buffers_keep_their_registers_in_spirv(tmp_path):
    assembly = _translate(tmp_path, "cb.hlsl", HLSL_CONSTANT_BUFFERS, "vulkan")
    assert _slot(assembly, "single") == (1, 4)
    assert _slot(assembly, "runtime") == (2, 0)
    assert _slot(assembly, "bounded") == (0, 1)
    assert "OpCapability UniformBufferArrayNonUniformIndexing" in assembly
    _validate_spirv(tmp_path, assembly)


def test_hlsl_runtime_constant_buffer_arrays_round_trip(tmp_path):
    hlsl = _translate(tmp_path, "cb.hlsl", HLSL_CONSTANT_BUFFERS, "directx")
    assert re.search(r"ConstantBuffer<W> runtime\[\] : register\(b0, space2\);", hlsl)
    assert re.search(r"ConstantBuffer<W> bounded\[3\] : register\(b1\);", hlsl)
    assert "NonUniformResourceIndex(idx)" in hlsl


def test_hlsl_non_uniform_index_is_spelled_for_glsl(tmp_path):
    glsl = _translate(tmp_path, "cb.hlsl", HLSL_CONSTANT_BUFFERS, "opengl")
    assert "NonUniformResourceIndex" not in glsl
    assert "runtime[nonuniformEXT(idx)]" in glsl
    assert "#extension GL_EXT_nonuniform_qualifier : require" in glsl


def test_metal_drops_the_non_uniform_qualifier(tmp_path):
    metal = _translate(tmp_path, "tex.frag", GLSL_TEXTURE_ARRAY, "metal")
    assert "nonuniformEXT" not in metal
    assert "NonUniformResourceIndex" not in metal
    assert re.search(r"tex\[input\.idx\]\.sample\(", metal)


def test_crossgl_non_uniform_index_is_dropped_for_metal_and_kept_for_hlsl(tmp_path):
    metal = _translate(tmp_path, "tex.cgl", CROSSGL_TEXTURE_ARRAY, "metal")
    assert "NonUniformResourceIndex" not in metal
    hlsl = _translate(tmp_path, "tex.cgl", CROSSGL_TEXTURE_ARRAY, "directx")
    assert "NonUniformResourceIndex(" in hlsl


def test_metal_refuses_arrays_of_constant_buffers(tmp_path):
    with pytest.raises(Exception, match="argument buffer"):
        _translate(tmp_path, "blocks.frag", GLSL_BLOCK_ARRAY, "metal")

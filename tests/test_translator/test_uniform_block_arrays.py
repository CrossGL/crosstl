"""Arrays of uniform blocks bind one buffer per element in every direction.

GLSL ``uniform B { ... } name[N]`` and HLSL ``ConstantBuffer<B> name[N]`` both declare ``N`` uniform buffers, one
descriptor each. CrossGL spells them ``ConstantBuffer<B> name[N]``; SPIR-V declares a ``Uniform`` variable of an
array of ``Block`` structs, and an unsized array is a runtime descriptor array.
"""

import re
import shutil
import subprocess

import pytest

import crosstl
from crosstl.translator.codegen.metal_refusals import MetalLoweringRefused
from tests.spirv_friendly_names import friendly_spirv

GLSL_SIZED = """#version 450
layout(location = 0) flat in uint idx;
layout(location = 0) out vec4 o_color;
layout(set = 0, binding = 4) uniform Weights
{
    vec4 data[4];
} weights_arr[3];
void main() {
    o_color = weights_arr[idx].data[1] + weights_arr[2].data[0];
}
"""

GLSL_RUNTIME = """#version 450
#extension GL_EXT_nonuniform_qualifier : enable
layout(location = 0) flat in uint idx;
layout(location = 0) out vec4 o_color;
layout(set = 1, binding = 2) uniform Weights
{
    vec4 data[4];
} weights_arr[];
void main() {
    o_color = weights_arr[nonuniformEXT(idx)].data[1];
}
"""

HLSL_ARRAY = """struct Light { float4 color; float4 direction; };
ConstantBuffer<Light> lights[4] : register(b3, space1);
float4 main(nointerpolation uint idx : TEXCOORD0) : SV_Target {
    return lights[idx].color + lights[1].direction;
}
"""

HLSL_SINGLE = """struct Light { float4 color; };
ConstantBuffer<Light> light : register(b5);
float4 main() : SV_Target { return light.color; }
"""

HLSL_GRID = """struct Light { float4 color; };
ConstantBuffer<Light> lights[2][3] : register(b0);
float4 main(nointerpolation uint idx : TEXCOORD0) : SV_Target { return lights[1][idx].color; }
"""


def _translate(tmp_path, source, backend, name):
    path = tmp_path / name
    path.write_text(source)
    return crosstl.translate(str(path), backend=backend, format_output=False)


def _run(command, **kwargs):
    result = subprocess.run(command, capture_output=True, text=True, **kwargs)
    assert result.returncode == 0, result.stdout + result.stderr


def _validate_spirv(tmp_path, code):
    if not (shutil.which("spirv-as") and shutil.which("spirv-val")):
        return
    asm, binary = tmp_path / "out.spvasm", tmp_path / "out.spv"
    asm.write_text(code)
    _run(["spirv-as", "--target-env", "vulkan1.1", str(asm), "-o", str(binary)])
    _run(["spirv-val", "--target-env", "vulkan1.1", str(binary)])


def _validate_glsl(code, *flags):
    if shutil.which("glslangValidator") is None:
        return
    _run(["glslangValidator", *flags, "--stdin", "-S", "frag"], input=code)


def _validate_vulkan_glsl(tmp_path, code):
    # Vulkan rules make glslang write SPIR-V, so it goes to the test's directory.
    _validate_glsl(code, "-V", "-o", str(tmp_path / "frag.spv"))


def _spirv(tmp_path, source, name):
    return friendly_spirv(_translate(tmp_path, source, "vulkan", name))


def _variable(spirv, name):
    """``(pointer type, set, binding)`` of the ``Uniform`` variable ``name``."""
    pointer = re.search(rf"%{name} = OpVariable (%\w+) Uniform\b", spirv)
    assert pointer, spirv
    descriptor_set = re.search(rf"OpDecorate %{name} DescriptorSet (\d+)", spirv)
    binding = re.search(rf"OpDecorate %{name} Binding (\d+)", spirv)
    return pointer.group(1), int(descriptor_set.group(1)), int(binding.group(1))


def _pointee(spirv, pointer):
    match = re.search(rf"{pointer} = OpTypePointer Uniform (%\w+)", spirv)
    assert match, spirv
    return match.group(1)


def _assert_block(spirv, struct):
    assert re.search(rf"OpDecorate {struct} Block\b", spirv), spirv
    assert re.search(rf"OpMemberDecorate {struct} 0 Offset 0\b", spirv), spirv


@pytest.mark.parametrize(
    "source, declaration",
    [
        (
            GLSL_SIZED,
            "ConstantBuffer<Weights> weights_arr[3] @set(0) @binding(4) @buffer_layout(std140);",
        ),
        (
            GLSL_RUNTIME,
            "ConstantBuffer<Weights> weights_arr[] @set(1) @binding(2) @buffer_layout(std140);",
        ),
    ],
)
def test_glsl_import_declares_one_constant_buffer_per_element(
    tmp_path, source, declaration
):
    crossgl = _translate(tmp_path, source, "cgl", "shader.frag")

    assert declaration in crossgl
    assert "uniform Weights weights_arr" not in crossgl


def test_spirv_declares_a_sized_array_of_uniform_blocks(tmp_path):
    spirv = _spirv(tmp_path, GLSL_SIZED, "shader.frag")

    pointer, descriptor_set, binding = _variable(spirv, "weights_arr")
    assert (descriptor_set, binding) == (0, 4)
    array = _pointee(spirv, pointer)
    element = re.search(rf"{array} = OpTypeArray (%\w+) %int_3\b", spirv)
    assert element, spirv
    _assert_block(spirv, element.group(1))
    # The descriptor index selects the block; the member follows it directly, with no wrapper struct.
    assert re.search(r"OpAccessChain %\w+ %weights_arr %int_2\n", spirv), spirv
    assert "RuntimeDescriptorArray" not in spirv
    _validate_spirv(tmp_path, spirv)


def test_spirv_declares_a_runtime_array_of_uniform_blocks(tmp_path):
    spirv = _spirv(tmp_path, GLSL_RUNTIME, "shader.frag")

    pointer, descriptor_set, binding = _variable(spirv, "weights_arr")
    assert (descriptor_set, binding) == (1, 2)
    element = re.search(
        rf"{_pointee(spirv, pointer)} = OpTypeRuntimeArray (%\w+)\n", spirv
    )
    assert element, spirv
    _assert_block(spirv, element.group(1))
    for capability in (
        "RuntimeDescriptorArray",
        "ShaderNonUniform",
        "UniformBufferArrayNonUniformIndexing",
    ):
        assert f"OpCapability {capability}\n" in spirv
    assert 'OpExtension "SPV_EXT_descriptor_indexing"' in spirv
    chain = re.search(r"(%\w+) = OpAccessChain %\w+ %weights_arr (%\w+)\n", spirv)
    assert chain, spirv
    assert f"OpDecorate {chain.group(1)} NonUniform" in spirv
    assert f"OpDecorate {chain.group(2)} NonUniform" in spirv
    _validate_spirv(tmp_path, spirv)


def test_uniform_indexing_needs_no_non_uniform_capability(tmp_path):
    spirv = _spirv(tmp_path, GLSL_SIZED, "shader.frag")

    assert "NonUniform" not in spirv


def test_spirv_binds_hlsl_constant_buffer_arrays_at_their_register(tmp_path):
    spirv = _spirv(tmp_path, HLSL_ARRAY, "shader.frag.hlsl")

    pointer, descriptor_set, binding = _variable(spirv, "lights")
    assert (descriptor_set, binding) == (1, 3)
    element = re.search(
        rf"{_pointee(spirv, pointer)} = OpTypeArray (%\w+) %int_4\b", spirv
    )
    assert element, spirv
    _assert_block(spirv, element.group(1))
    assert re.search(rf"OpMemberDecorate {element.group(1)} 1 Offset 16\b", spirv)
    _validate_spirv(tmp_path, spirv)


def test_spirv_binds_a_single_hlsl_constant_buffer_at_its_register(tmp_path):
    spirv = _spirv(tmp_path, HLSL_SINGLE, "shader.frag.hlsl")

    assert _variable(spirv, "light")[1:] == (0, 5)
    _validate_spirv(tmp_path, spirv)


def test_spirv_flattens_arrays_of_arrays_of_constant_buffers(tmp_path):
    spirv = _spirv(tmp_path, HLSL_GRID, "shader.frag.hlsl")

    pointer, _, _ = _variable(spirv, "lights")
    assert re.search(rf"{_pointee(spirv, pointer)} = OpTypeArray %\w+ %int_6\b", spirv)
    # lights[1][idx] selects descriptor 1 * 3 + idx.
    assert re.search(r"OpIMul %uint %\w+ %uint_3\b", spirv), spirv
    _validate_spirv(tmp_path, spirv)


def test_glsl_round_trip_keeps_an_array_of_uniform_blocks(tmp_path):
    glsl = _translate(tmp_path, GLSL_SIZED, "opengl", "shader.frag")

    assert re.search(
        r"layout\(std140, binding = 4\) uniform \w+ \{\s*vec4 data\[4\];\s*\}\s*weights_arr\[3\];",
        glsl,
    ), glsl
    _validate_glsl(glsl)


def test_glsl_round_trip_keeps_a_runtime_array_of_uniform_blocks(tmp_path):
    glsl = _translate(tmp_path, GLSL_RUNTIME, "opengl", "shader.frag")

    assert re.search(r"\}\s*weights_arr\[\];", glsl), glsl
    assert "weights_arr[nonuniformEXT(idx)].data[1]" in glsl
    _validate_vulkan_glsl(tmp_path, glsl)


@pytest.mark.parametrize(
    "source, declaration",
    [
        (GLSL_SIZED, "ConstantBuffer<Weights> weights_arr[3] : register(b4);"),
        (GLSL_RUNTIME, "ConstantBuffer<Weights> weights_arr[] : register(b2, space1);"),
    ],
)
def test_hlsl_declares_glsl_uniform_block_arrays_as_constant_buffer_arrays(
    tmp_path, source, declaration
):
    hlsl = _translate(tmp_path, source, "directx", "shader.frag")

    assert declaration in hlsl


def test_glsl_declares_hlsl_constant_buffer_arrays_as_uniform_block_arrays(tmp_path):
    glsl = _translate(tmp_path, HLSL_ARRAY, "opengl", "shader.frag.hlsl")

    assert re.search(r"uniform \w+ \{[^}]*\}\s*lights\[4\];", glsl), glsl
    _validate_glsl(glsl)


def test_metal_refuses_arrays_of_uniform_blocks(tmp_path):
    with pytest.raises(MetalLoweringRefused, match="through an argument buffer"):
        _translate(tmp_path, GLSL_SIZED, "metal", "shader.frag")

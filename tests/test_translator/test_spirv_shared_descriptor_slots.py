"""One descriptor slot for the image and sampler halves of a combined image sampler.

Vulkan lets a separate sampled image (``OpTypeImage`` with Sampled 1) and a separate ``OpTypeSampler`` with the same
set and binding consume one ``VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER`` descriptor (the Vulkan specification,
"Shader Resource and Descriptor Type Correspondence"). DXC emits that layout for HLSL that gives a ``Texture2D`` and
a ``SamplerState`` one ``[[vk::binding]]``, and glslang for GLSL ``texture2D`` and ``sampler`` uniforms with one
``layout(binding)``. Any other pair of resources on one slot is still a duplicate.
"""

import re
import shutil
import subprocess

import pytest

import crosstl
from tests.spirv_friendly_names import friendly_spirv

HAS_SPIRV_TOOLS = bool(shutil.which("spirv-as") and shutil.which("spirv-val"))

# Source shape: Khronos Vulkan-Samples texture_loading/hlsl/texture.frag.hlsl.
HLSL_SHARED_SLOT = """
[[vk::binding(1, 0)]]
Texture2D textureColor : register(t1);
[[vk::binding(1, 0)]]
SamplerState samplerColor : register(s1);
float4 main([[vk::location(0)]] float2 uv : TEXCOORD0) : SV_TARGET
{
    return textureColor.Sample(samplerColor, uv);
}
"""

HLSL_TWO_TEXTURES_ONE_SLOT = """
[[vk::binding(1, 0)]]
Texture2D a : register(t1);
[[vk::binding(1, 0)]]
Texture2D b : register(t2);
[[vk::binding(2, 0)]]
SamplerState s : register(s1);
float4 main([[vk::location(0)]] float2 uv : TEXCOORD0) : SV_TARGET
{
    return a.Sample(s, uv) + b.Sample(s, uv);
}
"""

GLSL_SHARED_SLOT = """#version 450
layout(set = 0, binding = 1) uniform texture2D tex;
layout(set = 0, binding = 1) uniform sampler samp;
layout(location = 0) in vec2 uv;
layout(location = 0) out vec4 o;
void main() { o = texture(sampler2D(tex, samp), uv); }
"""

GLSL_TWO_SAMPLERS_ONE_SLOT = """#version 450
layout(set = 0, binding = 1) uniform texture2D tex;
layout(set = 0, binding = 1) uniform sampler samp;
layout(set = 0, binding = 1) uniform sampler other;
layout(location = 0) in vec2 uv;
layout(location = 0) out vec4 o;
void main() { o = texture(sampler2D(tex, samp), uv) + texture(sampler2D(tex, other), uv); }
"""


def _spirv(tmp_path, source, name):
    path = tmp_path / name
    path.write_text(source)
    return friendly_spirv(
        crosstl.translate(str(path), backend="vulkan", format_output=False)
    )


def _validate(tmp_path, spirv):
    if not HAS_SPIRV_TOOLS:
        return
    assembly = tmp_path / "module.spvasm"
    binary = tmp_path / "module.spv"
    assembly.write_text(spirv)
    for command in (
        ["spirv-as", "--target-env", "vulkan1.1", str(assembly), "-o", str(binary)],
        ["spirv-val", "--target-env", "vulkan1.1", str(binary)],
    ):
        result = subprocess.run(command, capture_output=True, text=True)
        assert result.returncode == 0, result.stdout + result.stderr


def _slot(spirv, name):
    descriptor_set = re.search(rf"OpDecorate %{name} DescriptorSet (\d+)", spirv)
    binding = re.search(rf"OpDecorate %{name} Binding (\d+)", spirv)
    assert descriptor_set and binding, name
    return int(descriptor_set.group(1)), int(binding.group(1))


@pytest.mark.parametrize(
    "source,name,image,sampler",
    [
        (HLSL_SHARED_SLOT, "shader.frag.hlsl", "textureColor", "samplerColor"),
        (GLSL_SHARED_SLOT, "shader.frag", "tex", "samp"),
    ],
    ids=["hlsl", "glsl"],
)
def test_image_and_sampler_share_one_binding(tmp_path, source, name, image, sampler):
    spirv = _spirv(tmp_path, source, name)

    assert _slot(spirv, image) == (0, 1)
    assert _slot(spirv, sampler) == (0, 1)
    assert re.search(rf"%{sampler} = OpVariable %\w+ UniformConstant", spirv)
    _validate(tmp_path, spirv)


@pytest.mark.parametrize(
    "source,name",
    [
        (HLSL_TWO_TEXTURES_ONE_SLOT, "shader.frag.hlsl"),
        (GLSL_TWO_SAMPLERS_ONE_SLOT, "shader.frag"),
    ],
    ids=["hlsl", "glsl"],
)
def test_other_resources_on_one_binding_stay_duplicates(tmp_path, source, name):
    with pytest.raises(
        ValueError, match="Duplicate SPIR-V resource binding set 0 binding 1"
    ):
        _spirv(tmp_path, source, name)

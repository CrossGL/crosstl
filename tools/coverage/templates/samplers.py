"""Shared-sampler probe: a kernel samples two 2D textures through one sampler object, as ``sampled-texture`` samples
one, and adds a component of each.

HLSL, Slang, Metal, WGSL, OpenCL C and rust-gpu always declare textures and samplers apart, so their probe declares
two textures and one sampler. GLSL combines a texture with its sampler in a ``sampler2D`` unless the program declares
a ``texture2D`` and a ``sampler`` and pairs them where it samples, as ``sampler2D(texa, smp)``, which only GLSL for
Vulkan has (GL_KHR_vulkan_glsl), so the GLSL probe is checked for Vulkan. CrossGL passes the texture and the sampler
to the sampling call. GLSL ES 3.00 has only combined samplers, a CUDA or HIP texture object carries its own
sampling state, and Mojo's kernels take no textures.
"""

from __future__ import annotations

from .base import ABSENT, Template

NAMES = ("shared-sampler",)

_PAIR = "outp[i] = {}.x + {}.y;"


# GLSL pairs the texture with the sampler in a ``sampler2D`` constructor; CrossGL passes both to the sampling call,
# the form HLSL, GLSL and Metal programs import as.
_GLSL_SAMPLE = "textureLod(sampler2D({}, smp), vec2(fin[i], 0.5), 0.0)"
_CROSSGL_SAMPLE = "textureLod({}, smp, vec2(fin[i], 0.5), 0.0)"


def _glsl(
    sample: str = _GLSL_SAMPLE,
    binding: str = "layout(binding = {}) ",
    target: str = "vulkan",
) -> dict[str, Template]:
    """The GLSL probe, checked for Vulkan, or CrossGL's, which binds nothing."""
    return {
        "shared-sampler": Template(
            _PAIR.format(sample.format("texa"), sample.format("texb")),
            f"{binding.format(3)}uniform texture2D texa;\n"
            f"{binding.format(4)}uniform texture2D texb;\n"
            f"{binding.format(5)}uniform sampler smp;",
            target=target,
        )
    }


GLSL = _glsl()
CROSSGL = _glsl(_CROSSGL_SAMPLE, binding="", target="")
WEBGL = dict.fromkeys(NAMES, ABSENT)

_HLSL_SAMPLE = "{}.SampleLevel(smp, float2(fin[i], 0.5), 0.0)"
HLSL = {
    "shared-sampler": Template(
        _PAIR.format(_HLSL_SAMPLE.format("texa"), _HLSL_SAMPLE.format("texb")),
        "Texture2D<float4> texa : register(t1);\nTexture2D<float4> texb : register(t2);\n"
        "SamplerState smp : register(s0);",
    )
}

_METAL_SAMPLE = "{}.sample(smp, float2(fin[i], 0.5), level(0.0))"
METAL = {
    "shared-sampler": Template(
        _PAIR.format(_METAL_SAMPLE.format("texa"), _METAL_SAMPLE.format("texb")),
        params="texture2d<float> texa [[texture(0)]]\ntexture2d<float> texb [[texture(1)]]\n"
        "sampler smp [[sampler(0)]]",
    )
}

_WGSL_SAMPLE = "textureSampleLevel({}, smp, vec2<f32>(fin[i], 0.5), 0.0)"
WGSL = {
    "shared-sampler": Template(
        _PAIR.format(_WGSL_SAMPLE.format("texa"), _WGSL_SAMPLE.format("texb")),
        "@group(0) @binding(4) var texa: texture_2d<f32>;\n"
        "@group(0) @binding(5) var texb: texture_2d<f32>;\n"
        "@group(0) @binding(6) var smp: sampler;",
    )
}

KERNELS = dict.fromkeys(NAMES, ABSENT)

_OPENCL_SAMPLE = "read_imagef({}, smp, (float2)(fin[i], 0.5f))"
OPENCL = {
    "shared-sampler": Template(
        _PAIR.format(_OPENCL_SAMPLE.format("texa"), _OPENCL_SAMPLE.format("texb")),
        "__constant sampler_t smp = CLK_NORMALIZED_COORDS_TRUE | CLK_ADDRESS_CLAMP_TO_EDGE | "
        "CLK_FILTER_LINEAR;",
        params="__read_only image2d_t texa\n__read_only image2d_t texb",
    )
}

_RUST_TEXTURE = (
    "#[spirv(descriptor_set = 0, binding = {})] {}: &Image!(2D, type=f32, sampled)"
)
_RUST_SAMPLE = "let {}: Vec4 = {}.sample_by_lod(*smp, Vec2::new(fin[i], 0.5), 0.0);"
RUST = {
    "shared-sampler": Template(
        f"{_RUST_SAMPLE.format('a', 'texa')}\n{_RUST_SAMPLE.format('b', 'texb')}\n"
        "outp[i] = a.x + b.y;",
        prelude="use spirv_std::{Image, Sampler};",
        params=f"{_RUST_TEXTURE.format(3, 'texa')}\n{_RUST_TEXTURE.format(4, 'texb')}\n"
        "#[spirv(descriptor_set = 0, binding = 5)] smp: &Sampler",
    )
}

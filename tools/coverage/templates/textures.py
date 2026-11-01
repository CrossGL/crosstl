"""One-dimensional, 3D, cube, array and buffer texture probes: the texture shapes other than the 2D texture of
``sampled-texture``.

Each probe is a compute kernel that reads one texel of a texture with that shape at an explicit level of detail and
stores its first component, as ``sampled-texture`` does for a 2D texture. GLSL and CrossGL use ``sampler1D``,
``sampler3D``, ``samplerCube``, ``sampler2DArray``, ``samplerCubeArray`` and ``samplerBuffer``, HLSL and Slang the
matching ``Texture*`` types and ``Buffer``, Metal ``texture1d``, ``texture3d``, ``texturecube``, ``texture2d_array``,
``texturecube_array`` and ``texture_buffer``, WGSL the ``texture_*`` types, and rust-gpu the ``Image!`` shapes. CUDA
and HIP read texture objects through ``tex1DLod``, ``tex3DLod``, ``texCubemapLod``, ``tex2DLayeredLod``,
``texCubemapLayeredLod`` and ``tex1Dfetch``, and OpenCL C reads ``image1d_t``, ``image3d_t``, ``image2d_array_t``
and ``image1d_buffer_t`` images.

A buffer texture is read with a texel fetch, since it has no sampler. Metal has no level of detail for a 1D texture,
and WGSL samples no 1D texture, so those probes read it without one and with ``textureLoad``. WGSL has no buffer
textures, OpenCL C no cube images, GLSL ES 3.00 none of 1D, cube-array or buffer textures, and Mojo's kernels take
no textures.
"""

from __future__ import annotations

from .base import ABSENT, Template

NAMES = (
    "texture-1d",
    "texture-3d",
    "texture-cube",
    "texture-2d-array",
    "texture-cube-array",
    "texel-buffer",
)

# Each GLSL probe's sampler type and the coordinate it samples at.
_GLSL_SHAPES = {
    "texture-1d": ("sampler1D", "fin[i]"),
    "texture-3d": ("sampler3D", "vec3(fin[i], 0.5, 0.5)"),
    "texture-cube": ("samplerCube", "vec3(fin[i], 0.5, 1.0)"),
    "texture-2d-array": ("sampler2DArray", "vec3(fin[i], 0.5, 1.0)"),
    "texture-cube-array": ("samplerCubeArray", "vec4(fin[i], 0.5, 1.0, 1.0)"),
}
_FETCH = "outp[i] = texelFetch(tex, int(fin[i])).x;"


def _glsl(precision: str = "", binding: str = "layout(binding = 3) ") -> dict:
    sampled = {
        name: Template(
            f"outp[i] = textureLod(tex, {coordinate}, 0.0).x;",
            f"{binding}uniform {precision}{sampler} tex;",
        )
        for name, (sampler, coordinate) in _GLSL_SHAPES.items()
    }
    return {
        **sampled,
        "texel-buffer": Template(
            _FETCH, f"{binding}uniform {precision}samplerBuffer tex;"
        ),
    }


GLSL = _glsl()
CROSSGL = _glsl(binding="")
WEBGL = {
    **dict.fromkeys(NAMES, ABSENT),
    **{
        name: template
        for name, template in _glsl(precision="highp ", binding="").items()
        if name in ("texture-3d", "texture-cube", "texture-2d-array")
    },
}

# Each HLSL probe's texture type and the coordinate it samples at.
_HLSL_SHAPES = {
    "texture-1d": ("Texture1D", "fin[i]"),
    "texture-3d": ("Texture3D", "float3(fin[i], 0.5, 0.5)"),
    "texture-cube": ("TextureCube", "float3(fin[i], 0.5, 1.0)"),
    "texture-2d-array": ("Texture2DArray", "float3(fin[i], 0.5, 1.0)"),
    "texture-cube-array": ("TextureCubeArray", "float4(fin[i], 0.5, 1.0, 1.0)"),
}
HLSL = {
    **{
        name: Template(
            f"outp[i] = tex.SampleLevel(smp, {coordinate}, 0.0).x;",
            f"{texture}<float4> tex : register(t1);\nSamplerState smp : register(s0);",
        )
        for name, (texture, coordinate) in _HLSL_SHAPES.items()
    },
    "texel-buffer": Template(
        "outp[i] = tex.Load(int(fin[i])).x;", "Buffer<float4> tex : register(t1);"
    ),
}

# Each Metal probe's texture type and the arguments of its sample call after the sampler.
_METAL_SHAPES = {
    "texture-1d": ("texture1d", "fin[i]"),
    "texture-3d": ("texture3d", "float3(fin[i], 0.5, 0.5), level(0.0)"),
    "texture-cube": ("texturecube", "float3(fin[i], 0.5, 1.0), level(0.0)"),
    "texture-2d-array": ("texture2d_array", "float2(fin[i], 0.5), 1, level(0.0)"),
    "texture-cube-array": (
        "texturecube_array",
        "float3(fin[i], 0.5, 1.0), 1, level(0.0)",
    ),
}
METAL = {
    **{
        name: Template(
            f"outp[i] = tex.sample(smp, {arguments}).x;",
            params=f"{texture}<float> tex [[texture(0)]]\nsampler smp [[sampler(0)]]",
        )
        for name, (texture, arguments) in _METAL_SHAPES.items()
    },
    "texel-buffer": Template(
        "outp[i] = tex.read(uint(fin[i])).x;",
        params="texture_buffer<float> tex [[texture(0)]]",
    ),
}

_WGSL_SAMPLER = "\n@group(0) @binding(5) var smp: sampler;"
# Each WGSL probe's texture type and the arguments of its textureSampleLevel call after the sampler.
_WGSL_SHAPES = {
    "texture-3d": ("texture_3d", "vec3<f32>(fin[i], 0.5, 0.5), 0.0"),
    "texture-cube": ("texture_cube", "vec3<f32>(fin[i], 0.5, 1.0), 0.0"),
    "texture-2d-array": ("texture_2d_array", "vec2<f32>(fin[i], 0.5), 1, 0.0"),
    "texture-cube-array": ("texture_cube_array", "vec3<f32>(fin[i], 0.5, 1.0), 1, 0.0"),
}
WGSL = {
    "texture-1d": Template(
        "outp[i] = textureLoad(tex, i32(fin[i]), 0).x;",
        "@group(0) @binding(4) var tex: texture_1d<f32>;",
    ),
    **{
        name: Template(
            f"outp[i] = textureSampleLevel(tex, smp, {arguments}).x;",
            f"@group(0) @binding(4) var tex: {texture}<f32>;{_WGSL_SAMPLER}",
        )
        for name, (texture, arguments) in _WGSL_SHAPES.items()
    },
    "texel-buffer": ABSENT,
}

# Each CUDA and HIP probe's texture read.
_CUDA_READS = {
    "texture-1d": "tex1DLod<float4>(tex, fin[i], 0.0f)",
    "texture-3d": "tex3DLod<float4>(tex, fin[i], 0.5f, 0.5f, 0.0f)",
    "texture-cube": "texCubemapLod<float4>(tex, fin[i], 0.5f, 1.0f, 0.0f)",
    "texture-2d-array": "tex2DLayeredLod<float4>(tex, fin[i], 0.5f, 1, 0.0f)",
    "texture-cube-array": (
        "texCubemapLayeredLod<float4>(tex, fin[i], 0.5f, 1.0f, 1, 0.0f)"
    ),
    "texel-buffer": "tex1Dfetch<float4>(tex, int(fin[i]))",
}
CUDA = {
    name: Template(f"outp[i] = {read}.x;", params="cudaTextureObject_t tex")
    for name, read in _CUDA_READS.items()
}
HIP = {
    name: Template(f"outp[i] = {read}.x;", params="hipTextureObject_t tex")
    for name, read in _CUDA_READS.items()
}

_OPENCL_SAMPLER = (
    "__constant sampler_t smp = CLK_NORMALIZED_COORDS_TRUE | CLK_ADDRESS_CLAMP_TO_EDGE | "
    "CLK_FILTER_LINEAR;"
)
# Each OpenCL C probe's image type and the coordinate it reads at; a 2D array image reads its layer as the third
# coordinate.
_OPENCL_SHAPES = {
    "texture-1d": ("image1d_t", "fin[i]"),
    "texture-3d": ("image3d_t", "(float4)(fin[i], 0.5f, 0.5f, 0.0f)"),
    "texture-2d-array": ("image2d_array_t", "(float4)(fin[i], 0.5f, 1.0f, 0.0f)"),
}
OPENCL = {
    **{
        name: Template(
            f"outp[i] = read_imagef(tex, smp, {coordinate}).x;",
            _OPENCL_SAMPLER,
            params=f"__read_only {image} tex",
        )
        for name, (image, coordinate) in _OPENCL_SHAPES.items()
    },
    "texture-cube": ABSENT,
    "texture-cube-array": ABSENT,
    "texel-buffer": Template(
        "outp[i] = read_imagef(tex, (int)fin[i]).x;",
        params="__read_only image1d_buffer_t tex",
    ),
}
MOJO = dict.fromkeys(NAMES, ABSENT)

_RUST_SAMPLER = "\n#[spirv(descriptor_set = 0, binding = 4)] sampler: &Sampler"
# Each rust-gpu probe's image type and the coordinate it samples at.
_RUST_SHAPES = {
    "texture-1d": ("1D, type=f32, sampled", "fin[i]"),
    "texture-3d": ("3D, type=f32, sampled", "Vec3::new(fin[i], 0.5, 0.5)"),
    "texture-cube": ("cube, type=f32, sampled", "Vec3::new(fin[i], 0.5, 1.0)"),
    "texture-2d-array": (
        "2D, type=f32, sampled, arrayed",
        "Vec3::new(fin[i], 0.5, 1.0)",
    ),
    "texture-cube-array": (
        "cube, type=f32, sampled, arrayed",
        "Vec4::new(fin[i], 0.5, 1.0, 1.0)",
    ),
}
# spirv-std fetches a texel only with a level of detail, which a buffer image does not take, so the probe writes the
# fetch in SPIR-V.
_RUST_FETCH = """let coordinate = fin[i] as i32;
let mut color = Vec4::ZERO;
unsafe {
    asm!(
        "%image = OpLoad _ {tex}",
        "%coordinate = OpLoad _ {coordinate}",
        "%color = OpImageFetch typeof*{color} %image %coordinate",
        "OpStore {color} %color",
        tex = in(reg) tex,
        coordinate = in(reg) &coordinate,
        color = in(reg) &mut color,
    );
}
outp[i] = color.x;"""
RUST = {
    **{
        name: Template(
            f"let color: Vec4 = tex.sample_by_lod(*sampler, {coordinate}, 0.0);\n"
            "outp[i] = color.x;",
            prelude="use spirv_std::{Image, Sampler};",
            params=f"#[spirv(descriptor_set = 0, binding = 3)] tex: &Image!({image})"
            + _RUST_SAMPLER,
        )
        for name, (image, coordinate) in _RUST_SHAPES.items()
    },
    "texel-buffer": Template(
        _RUST_FETCH,
        prelude="#![feature(asm_experimental_arch)]\nuse core::arch::asm;\nuse spirv_std::Image;",
        params="#[spirv(descriptor_set = 0, binding = 3)] tex: &Image!(buffer, type=f32, sampled)",
    ),
}

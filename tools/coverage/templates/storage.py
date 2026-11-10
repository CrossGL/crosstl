"""Storage-image shape probes: a kernel writes its input to a 1D, 3D or 2D array storage image, or to a storage texel
buffer, as ``storage-image`` writes it to a 2D storage image of 32-bit float texels.

GLSL and CrossGL declare ``image1D``, ``image3D``, ``image2DArray`` and ``imageBuffer``, HLSL and Slang
``RWTexture1D``, ``RWTexture3D``, ``RWTexture2DArray`` and ``RWBuffer``, Metal the ``access::write`` forms of
``texture1d``, ``texture3d``, ``texture2d_array`` and ``texture_buffer``, WGSL the ``texture_storage_*`` types, OpenCL
C ``__write_only`` images and rust-gpu unsampled ``Image!`` shapes. An array image is written at layer 1. CUDA and HIP
write surface objects through ``surf1Dwrite``, ``surf3Dwrite`` and ``surf2DLayeredwrite``; a surface has no buffer
shape, since a kernel writes linear memory through a pointer. WGSL has no storage texel buffers either, GLSL ES 3.00
has no storage images, and Mojo's kernels take no images.
"""

from __future__ import annotations

from .base import ABSENT, Template

NAMES = (
    "storage-image-1d",
    "storage-image-3d",
    "storage-image-array",
    "storage-texel-buffer",
)

# Each GLSL probe's image type and the coordinate it writes at.
_GLSL_SHAPES = {
    "storage-image-1d": ("image1D", "0"),
    "storage-image-3d": ("image3D", "ivec3(0, 0, 0)"),
    "storage-image-array": ("image2DArray", "ivec3(0, 0, 1)"),
    "storage-texel-buffer": ("imageBuffer", "0"),
}


def _glsl(binding: str = ", binding = 4") -> dict[str, Template]:
    """The GLSL probes; CrossGL's bind nothing."""
    return {
        name: Template(
            f"imageStore(img, {coordinate}, vec4(fin[i]));\noutp[i] = 0.0;",
            f"layout(rgba32f{binding}) uniform writeonly {image} img;",
        )
        for name, (image, coordinate) in _GLSL_SHAPES.items()
    }


GLSL = _glsl()
CROSSGL = _glsl(binding="")
WEBGL = dict.fromkeys(NAMES, ABSENT)

_HLSL_SHAPES = {
    "storage-image-1d": ("RWTexture1D", "0"),
    "storage-image-3d": ("RWTexture3D", "uint3(0, 0, 0)"),
    "storage-image-array": ("RWTexture2DArray", "uint3(0, 0, 1)"),
    "storage-texel-buffer": ("RWBuffer", "0"),
}
HLSL = {
    name: Template(
        f"img[{coordinate}] = float4(fin[i], fin[i], fin[i], fin[i]);\noutp[i] = 0.0;",
        f"{image}<float4> img : register(u3);",
    )
    for name, (image, coordinate) in _HLSL_SHAPES.items()
}

# Each Metal probe's texture type and the arguments of its write call after the value.
_METAL_SHAPES = {
    "storage-image-1d": ("texture1d", "uint(0)"),
    "storage-image-3d": ("texture3d", "uint3(0, 0, 0)"),
    "storage-image-array": ("texture2d_array", "uint2(0, 0), 1"),
    "storage-texel-buffer": ("texture_buffer", "uint(0)"),
}
METAL = {
    name: Template(
        f"img.write(float4(fin[i]), {arguments});\noutp[i] = 0.0;",
        params=f"{texture}<float, access::write> img [[texture(1)]]",
    )
    for name, (texture, arguments) in _METAL_SHAPES.items()
}

# Each WGSL probe's storage texture type and the arguments of its textureStore call before the value.
_WGSL_SHAPES = {
    "storage-image-1d": ("texture_storage_1d", "0i"),
    "storage-image-3d": ("texture_storage_3d", "vec3<i32>(0, 0, 0)"),
    "storage-image-array": ("texture_storage_2d_array", "vec2<i32>(0, 0), 1"),
}
WGSL = {
    **{
        name: Template(
            f"textureStore(img, {arguments}, vec4<f32>(fin[i]));\noutp[i] = 0.0;",
            f"@group(0) @binding(6) var img: {texture}<rgba32float, write>;",
        )
        for name, (texture, arguments) in _WGSL_SHAPES.items()
    },
    "storage-texel-buffer": ABSENT,
}

# Each CUDA and HIP probe's surface write, after the value and the surface.
_CUDA_WRITES = {
    "storage-image-1d": ("surf1Dwrite", "0"),
    "storage-image-3d": ("surf3Dwrite", "0, 0, 0"),
    "storage-image-array": ("surf2DLayeredwrite", "0, 0, 1"),
}


def _cuda(surface: str) -> dict[str, Template]:
    """The CUDA probes, or the HIP ones with HIP's surface type."""
    return {
        **{
            name: Template(
                f"{write}(make_float4(fin[i], fin[i], fin[i], fin[i]), img, {coordinate});\n"
                "outp[i] = 0.0f;",
                params=f"{surface} img",
            )
            for name, (write, coordinate) in _CUDA_WRITES.items()
        },
        "storage-texel-buffer": ABSENT,
    }


CUDA = _cuda("cudaSurfaceObject_t")
HIP = _cuda("hipSurfaceObject_t")

_OPENCL_SHAPES = {
    "storage-image-1d": ("image1d_t", "0"),
    "storage-image-3d": ("image3d_t", "(int4)(0, 0, 0, 0)"),
    "storage-image-array": ("image2d_array_t", "(int4)(0, 0, 1, 0)"),
    "storage-texel-buffer": ("image1d_buffer_t", "0"),
}
OPENCL = {
    name: Template(
        f"write_imagef(img, {coordinate}, (float4)(fin[i]));\noutp[i] = 0.0f;",
        params=f"__write_only {image} img",
    )
    for name, (image, coordinate) in _OPENCL_SHAPES.items()
}

MOJO = dict.fromkeys(NAMES, ABSENT)

# Each rust-gpu probe's image shape and the coordinate it writes at.
_RUST_SHAPES = {
    "storage-image-1d": ("1D", "0u32"),
    "storage-image-3d": ("3D", "UVec3::new(0, 0, 0)"),
    "storage-image-array": ("2D, arrayed", "UVec3::new(0, 0, 1)"),
    "storage-texel-buffer": ("buffer", "0u32"),
}
RUST = {
    name: Template(
        f"unsafe {{ img.write({coordinate}, Vec4::splat(fin[i])) }};\noutp[i] = 0.0;",
        prelude="use spirv_std::Image;",
        params="#[spirv(descriptor_set = 0, binding = 3)] img: "
        f"&Image!({shape}, format = rgba32f, sampled = false)",
    )
    for name, (shape, coordinate) in _RUST_SHAPES.items()
}

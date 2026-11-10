"""Storage-image format probes: a kernel writes its input to a storage image of 8-bit normalized RGBA texels
(``unorm-storage-image``) and to one of 32-bit unsigned integer texels (``integer-storage-image``), where
``storage-image`` writes 32-bit floats.

GLSL and CrossGL name the format in the image's layout (``rgba8``, ``r32ui``), WGSL in its type (``rgba8unorm``,
``r32uint``) and rust-gpu in its ``Image!`` (``format = rgba8``, ``format = r32ui``). HLSL and Slang give the texel
type: ``unorm float4`` holds a normalized value, ``uint`` an integer. Metal and OpenCL C take the format from the
texture the host binds and declare only the texel type, ``float`` or ``uint``. A CUDA or HIP surface stores the bytes
it is given, so the kernel converts its input to 8-bit normalized values itself. GLSL ES 3.00 has no storage images,
and Mojo's kernels take no images.
"""

from __future__ import annotations

from .base import ABSENT, Template

NAMES = ("unorm-storage-image", "integer-storage-image")

_GLSL_STORE = "imageStore(img, ivec2(0, 0), {});\noutp[i] = 0.0;"


def _glsl(binding: str = ", binding = 4") -> dict[str, Template]:
    """The GLSL probes; CrossGL's bind nothing."""
    return {
        "unorm-storage-image": Template(
            _GLSL_STORE.format("vec4(fin[i])"),
            f"layout(rgba8{binding}) uniform writeonly image2D img;",
        ),
        "integer-storage-image": Template(
            _GLSL_STORE.format("uvec4(uint(fin[i]))"),
            f"layout(r32ui{binding}) uniform writeonly uimage2D img;",
        ),
    }


GLSL = _glsl()
CROSSGL = _glsl(binding="")
WEBGL = dict.fromkeys(NAMES, ABSENT)

HLSL = {
    "unorm-storage-image": Template(
        "img[uint2(0, 0)] = float4(fin[i], fin[i], fin[i], fin[i]);\noutp[i] = 0.0;",
        "RWTexture2D<unorm float4> img : register(u3);",
    ),
    "integer-storage-image": Template(
        "img[uint2(0, 0)] = uint(fin[i]);\noutp[i] = 0.0;",
        "RWTexture2D<uint> img : register(u3);",
    ),
}

METAL = {
    "unorm-storage-image": Template(
        "img.write(float4(fin[i]), uint2(0, 0));\noutp[i] = 0.0;",
        params="texture2d<float, access::write> img [[texture(1)]]",
    ),
    "integer-storage-image": Template(
        "img.write(uint4(uint(fin[i])), uint2(0, 0));\noutp[i] = 0.0;",
        params="texture2d<uint, access::write> img [[texture(1)]]",
    ),
}

_WGSL_STORE = "textureStore(img, vec2<i32>(0, 0), {});\noutp[i] = 0.0;"
_WGSL_IMAGE = "@group(0) @binding(6) var img: texture_storage_2d<{}, write>;"
WGSL = {
    "unorm-storage-image": Template(
        _WGSL_STORE.format("vec4<f32>(fin[i])"), _WGSL_IMAGE.format("rgba8unorm")
    ),
    "integer-storage-image": Template(
        _WGSL_STORE.format("vec4<u32>(u32(fin[i]))"), _WGSL_IMAGE.format("r32uint")
    ),
}


def _cuda(surface: str) -> dict[str, Template]:
    """The CUDA probes, or the HIP ones with HIP's surface type."""
    return {
        "unorm-storage-image": Template(
            "unsigned char v = (unsigned char)(__saturatef(fin[i]) * 255.0f + 0.5f);\n"
            "surf2Dwrite(make_uchar4(v, v, v, v), img, 0, 0);\noutp[i] = 0.0f;",
            params=f"{surface} img",
        ),
        "integer-storage-image": Template(
            "surf2Dwrite((unsigned int)fin[i], img, 0, 0);\noutp[i] = 0.0f;",
            params=f"{surface} img",
        ),
    }


CUDA = _cuda("cudaSurfaceObject_t")
HIP = _cuda("hipSurfaceObject_t")

OPENCL = {
    "unorm-storage-image": Template(
        "write_imagef(img, (int2)(0, 0), (float4)(fin[i]));\noutp[i] = 0.0f;",
        params="__write_only image2d_t img",
    ),
    "integer-storage-image": Template(
        "write_imageui(img, (int2)(0, 0), (uint4)((uint)fin[i]));\noutp[i] = 0.0f;",
        params="__write_only image2d_t img",
    ),
}

MOJO = dict.fromkeys(NAMES, ABSENT)

_RUST_IMAGE = "#[spirv(descriptor_set = 0, binding = 3)] img: &Image!(2D, format = {}, sampled = false)"
RUST = {
    "unorm-storage-image": Template(
        "unsafe { img.write(UVec2::new(0, 0), Vec4::splat(fin[i])) };\noutp[i] = 0.0;",
        prelude="use spirv_std::Image;",
        params=_RUST_IMAGE.format("rgba8"),
    ),
    "integer-storage-image": Template(
        "unsafe { img.write(UVec2::new(0, 0), fin[i] as u32) };\noutp[i] = 0.0;",
        prelude="use spirv_std::Image;",
        params=_RUST_IMAGE.format("r32ui"),
    ),
}

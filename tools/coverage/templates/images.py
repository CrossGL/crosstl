"""Texture-in-helper, texture-argument, array-of-textures and read-write-image probes: a texture that a helper function
samples, an array of textures, and a storage image that a kernel reads and writes in place. Two more probes index the
array at run time: by a loop counter, which is dynamically uniform, and by a value that can differ between invocations.

Each texture probe samples a 2D texture at an explicit level of detail, as ``sampled-texture`` does. GLSL, HLSL, Slang,
WGSL and CrossGL declare textures at program scope, so the texture-in-helper probe's helper samples the texture by
name, and the texture-argument probe's helper takes it as a parameter, with its sampler in HLSL, Slang and WGSL. Metal,
CUDA, HIP, OpenCL C and rust-gpu bind textures to entry-point parameters, so both helpers take the texture as a
parameter there. The array-of-textures probe samples both textures of a two-element array at constant indices, and
the read-write-image probe adds its input to the texel at the origin of a one-channel float storage image. The
non-uniform index is ``nonuniformEXT`` (GL_EXT_nonuniform_qualifier) in GLSL and CrossGL and
``NonUniformResourceIndex`` in HLSL and Slang; Metal indexes an array of textures with any value and has no qualifier.

WGSL arrays hold plain types only, which textures are not (``binding_array`` is a wgpu extension). CUDA and HIP take
texture objects as kernel parameters, which cannot be arrays, and OpenCL C declares no arrays of images. GLSL ES 3.00
has no storage images and indexes an array of samplers with constant expressions only. rust-gpu cannot mark an index
non-uniform: spirv-std has no function for it, and a ``NonUniform`` decoration written in inline SPIR-V does not reach
the module. Mojo's kernels take no textures.
"""

from __future__ import annotations

from .base import ABSENT, Template

NAMES = (
    "texture-in-helper",
    "texture-argument",
    "array-of-textures",
    "read-write-image",
    "array-of-textures-loop-index",
    "array-of-textures-nonuniform-index",
)

_CALL = "outp[i] = lookup({}fin[i]) * 2.0;"
_PAIR = "outp[i] = {0}.x + {1}.y;"
# The loop counter is dynamically uniform; the non-uniform index differs between invocations.
_LOOP = "float acc = 0.0;\nfor (int k = 0; k < 2; k++) {{\n    acc += {}.x;\n}}\noutp[i] = acc;"
_NONUNIFORM = "{} k = {}(fin[i]) & 1{};\noutp[i] = {}.x;"


def _glsl(precision: str = "", binding: str = "layout(binding = 3) ") -> dict:
    """The GLSL probes; CrossGL's bind nothing, and GLSL ES 3.00's give their samplers a precision."""
    sampler = f"{binding}uniform {precision}sampler2D tex"
    sample = "textureLod({}, vec2({}, 0.5), 0.0)"
    helper = (
        "float lookup({}float x) {{\n    return " + sample.format("{}", "x") + ".x;\n}}"
    )
    image = "layout(r32f) " if not binding else "layout(r32f, binding = 4) "
    return {
        "texture-in-helper": Template(
            _CALL.format(""), f"{sampler};\n\n" + helper.format("", "tex")
        ),
        "texture-argument": Template(
            _CALL.format("tex, "),
            f"{sampler};\n\n" + helper.format(f"{precision}sampler2D t, ", "t"),
        ),
        "array-of-textures": Template(
            _PAIR.format(
                sample.format("tex[0]", "fin[i]"), sample.format("tex[1]", "fin[i]")
            ),
            f"{sampler}[2];",
        ),
        "read-write-image": Template(
            "imageStore(img, ivec2(0, 0), imageLoad(img, ivec2(0, 0)) + vec4(fin[i]));\noutp[i] = 0.0;",
            f"{image}uniform image2D img;",
        ),
        "array-of-textures-loop-index": Template(
            _LOOP.format(sample.format("tex[k]", "fin[i]")), f"{sampler}[2];"
        ),
        "array-of-textures-nonuniform-index": Template(
            _NONUNIFORM.format(
                "int", "int", "", sample.format("tex[nonuniformEXT(k)]", "fin[i]")
            ),
            f"{sampler}[2];",
            prelude=(
                "#extension GL_EXT_nonuniform_qualifier : require" if binding else ""
            ),
        ),
    }


GLSL = _glsl()
CROSSGL = _glsl(binding="")
WEBGL = {
    **_glsl(precision="highp ", binding=""),
    "read-write-image": ABSENT,
    "array-of-textures-loop-index": ABSENT,
    "array-of-textures-nonuniform-index": ABSENT,
}

_HLSL_TEXTURE = (
    "Texture2D<float4> tex{} : register(t1);\nSamplerState smp : register(s0);"
)
_HLSL_SAMPLE = "{}.SampleLevel({}, float2({}, 0.5), 0.0)"
_HLSL_HELPER = "float lookup({}float x) {{\n    return {}.x;\n}}"
HLSL = {
    "texture-in-helper": Template(
        _CALL.format(""),
        f"{_HLSL_TEXTURE.format('')}\n\n"
        + _HLSL_HELPER.format("", _HLSL_SAMPLE.format("tex", "smp", "x")),
    ),
    "texture-argument": Template(
        _CALL.format("tex, smp, "),
        f"{_HLSL_TEXTURE.format('')}\n\n"
        + _HLSL_HELPER.format(
            "Texture2D<float4> t, SamplerState s, ",
            _HLSL_SAMPLE.format("t", "s", "x"),
        ),
    ),
    "array-of-textures": Template(
        _PAIR.format(
            _HLSL_SAMPLE.format("tex[0]", "smp", "fin[i]"),
            _HLSL_SAMPLE.format("tex[1]", "smp", "fin[i]"),
        ),
        _HLSL_TEXTURE.format("[2]"),
    ),
    "read-write-image": Template(
        "img[uint2(0, 0)] = img[uint2(0, 0)] + fin[i];\noutp[i] = 0.0;",
        "RWTexture2D<float> img : register(u3);",
    ),
    "array-of-textures-loop-index": Template(
        _LOOP.format(_HLSL_SAMPLE.format("tex[k]", "smp", "fin[i]")),
        _HLSL_TEXTURE.format("[2]"),
    ),
    "array-of-textures-nonuniform-index": Template(
        _NONUNIFORM.format(
            "uint",
            "uint",
            "u",
            _HLSL_SAMPLE.format("tex[NonUniformResourceIndex(k)]", "smp", "fin[i]"),
        ),
        _HLSL_TEXTURE.format("[2]"),
    ),
}

_METAL_SAMPLE = "{}.sample(smp, float2({}, 0.5), level(0.0))"
_METAL_PARAMS = "{} tex [[texture(0)]]\nsampler smp [[sampler(0)]]"
# Metal binds a texture to an entry-point parameter, so either helper takes it and its sampler.
_METAL_HELPER = Template(
    _CALL.format("tex, smp, "),
    "float lookup(texture2d<float> t, sampler s, float x) {\n"
    "    return t.sample(s, float2(x, 0.5), level(0.0)).x;\n}",
    params=_METAL_PARAMS.format("texture2d<float>"),
)
METAL = {
    "texture-in-helper": _METAL_HELPER,
    "texture-argument": _METAL_HELPER,
    "array-of-textures": Template(
        _PAIR.format(
            _METAL_SAMPLE.format("tex[0]", "fin[i]"),
            _METAL_SAMPLE.format("tex[1]", "fin[i]"),
        ),
        params=_METAL_PARAMS.format("array<texture2d<float>, 2>"),
    ),
    "read-write-image": Template(
        "img.write(img.read(uint2(0, 0)) + float4(fin[i]), uint2(0, 0));\noutp[i] = 0.0;",
        params="texture2d<float, access::read_write> img [[texture(1)]]",
    ),
    "array-of-textures-loop-index": Template(
        _LOOP.format(_METAL_SAMPLE.format("tex[k]", "fin[i]")),
        params=_METAL_PARAMS.format("array<texture2d<float>, 2>"),
    ),
    # Metal indexes an array of textures with any value and has no non-uniform qualifier.
    "array-of-textures-nonuniform-index": Template(
        _NONUNIFORM.format(
            "uint", "uint", "u", _METAL_SAMPLE.format("tex[k]", "fin[i]")
        ),
        params=_METAL_PARAMS.format("array<texture2d<float>, 2>"),
    ),
}

_WGSL_TEXTURE = "@group(0) @binding(4) var tex: texture_2d<f32>;\n@group(0) @binding(5) var smp: sampler;"
_WGSL_HELPER = (
    "fn lookup({}x: f32) -> f32 {{\n"
    "    return textureSampleLevel({}, {}, vec2<f32>(x, 0.5), 0.0).x;\n}}"
)
WGSL = {
    "texture-in-helper": Template(
        _CALL.format(""), f"{_WGSL_TEXTURE}\n\n" + _WGSL_HELPER.format("", "tex", "smp")
    ),
    "texture-argument": Template(
        _CALL.format("tex, smp, "),
        f"{_WGSL_TEXTURE}\n\n"
        + _WGSL_HELPER.format("t: texture_2d<f32>, s: sampler, ", "t", "s"),
    ),
    "array-of-textures": ABSENT,
    "array-of-textures-loop-index": ABSENT,
    "array-of-textures-nonuniform-index": ABSENT,
    "read-write-image": Template(
        "textureStore(img, vec2<i32>(0, 0), textureLoad(img, vec2<i32>(0, 0)) + vec4<f32>(fin[i]));\n"
        "outp[i] = 0.0;",
        "@group(0) @binding(6) var img: texture_storage_2d<r32float, read_write>;",
    ),
}


def _cuda(texture: str, surface: str) -> dict:
    """The CUDA probes, or HIP's with its ``texture`` and ``surface`` object types."""
    helper = Template(
        "outp[i] = lookup(tex, fin[i]) * 2.0f;",
        f"__device__ float lookup({texture} t, float x) {{\n"
        "    return tex2DLod<float4>(t, x, 0.5f, 0.0f).x;\n}",
        params=f"{texture} tex",
    )
    return {
        "texture-in-helper": helper,
        "texture-argument": helper,
        "array-of-textures": ABSENT,
        "array-of-textures-loop-index": ABSENT,
        "array-of-textures-nonuniform-index": ABSENT,
        "read-write-image": Template(
            "float v;\nsurf2Dread(&v, img, 0, 0);\nsurf2Dwrite(v + fin[i], img, 0, 0);\noutp[i] = 0.0f;",
            params=f"{surface} img",
        ),
    }


CUDA = _cuda("cudaTextureObject_t", "cudaSurfaceObject_t")
HIP = _cuda("hipTextureObject_t", "hipSurfaceObject_t")

_OPENCL_SAMPLER = (
    "__constant sampler_t smp = CLK_NORMALIZED_COORDS_TRUE | CLK_ADDRESS_CLAMP_TO_EDGE | "
    "CLK_FILTER_LINEAR;"
)
_OPENCL_HELPER = Template(
    "outp[i] = lookup(tex, fin[i]) * 2.0f;",
    f"{_OPENCL_SAMPLER}\n\nfloat lookup(__read_only image2d_t t, float x) {{\n"
    "    return read_imagef(t, smp, (float2)(x, 0.5f)).x;\n}",
    params="__read_only image2d_t tex",
)
OPENCL = {
    "texture-in-helper": _OPENCL_HELPER,
    "texture-argument": _OPENCL_HELPER,
    "array-of-textures": ABSENT,
    "array-of-textures-loop-index": ABSENT,
    "array-of-textures-nonuniform-index": ABSENT,
    "read-write-image": Template(
        "write_imagef(img, (int2)(0, 0), read_imagef(img, (int2)(0, 0)) + (float4)(fin[i]));\n"
        "outp[i] = 0.0f;",
        params="__read_write image2d_t img",
    ),
}
MOJO = dict.fromkeys(NAMES, ABSENT)

_RUST_SAMPLED = "Image!(2D, type=f32, sampled)"
_RUST_PARAMS = (
    "#[spirv(descriptor_set = 0, binding = 3)] tex: &{}\n"
    "#[spirv(descriptor_set = 0, binding = 4)] sampler: &Sampler"
)
_RUST_PRELUDE = "use spirv_std::{Image, Sampler};"
_RUST_SAMPLE = "let {}: Vec4 = {}.sample_by_lod(*sampler, Vec2::new(fin[i], 0.5), 0.0);"
# rust-gpu binds a texture to an entry-point parameter, so either helper takes it and its sampler.
_RUST_HELPER = Template(
    "outp[i] = lookup(tex, *sampler, fin[i]) * 2.0;",
    f"fn lookup(t: &{_RUST_SAMPLED}, s: Sampler, x: f32) -> f32 {{\n"
    "    let color: Vec4 = t.sample_by_lod(s, Vec2::new(x, 0.5), 0.0);\n    color.x\n}",
    prelude=_RUST_PRELUDE,
    params=_RUST_PARAMS.format(_RUST_SAMPLED),
)
RUST = {
    "texture-in-helper": _RUST_HELPER,
    "texture-argument": _RUST_HELPER,
    "array-of-textures": Template(
        f"{_RUST_SAMPLE.format('a', 'tex[0]')}\n{_RUST_SAMPLE.format('b', 'tex[1]')}\n"
        "outp[i] = a.x + b.y;",
        prelude=_RUST_PRELUDE,
        params=_RUST_PARAMS.format(f"[{_RUST_SAMPLED}; 2]"),
    ),
    "array-of-textures-loop-index": Template(
        "let mut acc = 0.0;\nfor k in 0..2 {\n    "
        + _RUST_SAMPLE.format("color", "tex[k]")
        + "\n    acc += color.x;\n}\noutp[i] = acc;",
        prelude=_RUST_PRELUDE,
        params=_RUST_PARAMS.format(f"[{_RUST_SAMPLED}; 2]"),
    ),
    "array-of-textures-nonuniform-index": ABSENT,
    "read-write-image": Template(
        "let v: f32 = img.read(UVec2::new(0, 0));\n"
        "unsafe { img.write(UVec2::new(0, 0), v + fin[i]) };\noutp[i] = 0.0;",
        prelude="use spirv_std::Image;",
        params="#[spirv(descriptor_set = 0, binding = 3)] img: &Image!(2D, format = r32f, sampled = false)",
    ),
}

"""Texel buffers translate between HLSL, GLSL, MSL and SPIR-V as texel buffers.

A texel buffer is a one-dimensional array of formatted texels: HLSL ``Buffer<T>``/``RWBuffer<T>``, GLSL
``samplerBuffer``/``imageBuffer`` (Vulkan GLSL also ``textureBuffer``), MSL ``texture_buffer<T, access>`` and SPIR-V
``OpTypeImage`` with ``Dim`` ``Buffer``. CrossGL spells them ``samplerBuffer`` and ``imageBuffer`` (with the ``i``/``u``
prefix of the element family). Reads return four components filled as ``(r, g, 0, 1)`` and stores take four, as in
GLSL; HLSL reads are narrowed to the declared element width and stores widened, as for textures.

Before, HLSL texel buffers were imported as untyped ``Buffer<float4>`` declarations that GLSL emitted verbatim, Metal
needed a length sidecar for, and SPIR-V could not type; GLSL texel buffers had no Metal, HLSL or SPIR-V size query or
fetch; and Vulkan ``utextureBuffer`` declarations were folded into a uniform block.
"""

from __future__ import annotations

import re
import shutil
import subprocess

import pytest

from tests.test_translator.test_comparison_samplers import _compile_metal
from tests.test_translator.test_spirv_separate_textures import (
    _definitions,
    _id_of,
    _translate,
    _validate,
)

HLSL_TEXEL_BUFFERS = """Buffer<float2> pairs : register(t0);
Buffer<int> ints : register(t1);
Buffer<float4> colorsIn : register(t2);
RWBuffer<uint> counters : register(u0);
RWBuffer<int2> ipairs : register(u1);
RWBuffer<float4> colors : register(u2);

float2 readPair(Buffer<float2> b, uint i) { return b[i]; }

[numthreads(64, 1, 1)]
void CSMain(uint3 id : SV_DispatchThreadID)
{
    uint original;
    InterlockedAdd(counters[id.x], 1u, original);
    InterlockedMax(counters[0], original);
    uint swapped;
    InterlockedCompareExchange(counters[1], 0u, id.x, swapped);
    float2 p = readPair(pairs, id.x) + pairs.Load(id.x + 1);
    int k = ints[id.x];
    ipairs[id.x] = int2(k, k + 1);
    ipairs[id.x] += int2(1, 1);
    ipairs[id.x].y = 7;
    colors[id.x] = float4(p, 0.0, 1.0) + colorsIn[id.x];
    colors[id.x].x += 1.0;
    uint n;
    counters.GetDimensions(n);
    uint m;
    pairs.GetDimensions(m);
    colors[id.x + 1] = float4(float(n), float(original), float(swapped), float(m));
}
"""

GLSL_TEXEL_BUFFERS = """#version 450
layout(local_size_x = 64) in;
layout(binding = 0) uniform samplerBuffer inData;
layout(binding = 1) uniform usamplerBuffer idx;
layout(binding = 2, r32f) uniform imageBuffer outData;
layout(binding = 3, rgba32ui) uniform readonly uimageBuffer roData;
layout(binding = 4, r32ui) uniform uimageBuffer counters;
void main() {
    int w = textureSize(inData);
    uint i = texelFetch(idx, int(gl_GlobalInvocationID.x)).x;
    vec4 v = texelFetch(inData, int(i));
    int n = imageSize(outData);
    uvec4 r = imageLoad(roData, int(gl_GlobalInvocationID.x));
    imageStore(outData, int(gl_GlobalInvocationID.x), vec4(v.x + float(w + n) + float(r.x)));
    float o = imageLoad(outData, int(gl_GlobalInvocationID.x) + 1).x;
    imageStore(outData, int(gl_GlobalInvocationID.x) + 2, vec4(o));
    uint old = imageAtomicAdd(counters, int(i), 1u);
    imageAtomicCompSwap(counters, 0, old, 2u);
}
"""

VULKAN_GLSL_TEXEL_BUFFERS = """#version 450
layout(local_size_x = 64) in;
layout(set = 0, binding = 0) uniform textureBuffer tb;
layout(set = 0, binding = 1) uniform utextureBuffer utb;
layout(set = 0, binding = 2, r32ui) uniform uimageBuffer counters;
void main() {
    int i = int(gl_GlobalInvocationID.x);
    vec4 v = texelFetch(tb, i);
    uint u = texelFetch(utb, i).x;
    uint old = imageAtomicAdd(counters, i, u + uint(v.x));
}
"""

MSL_TEXEL_BUFFERS = """#include <metal_stdlib>
using namespace metal;
kernel void k(texture_buffer<float, access::read> inData [[texture(0)]],
              texture_buffer<uint, access::read> idx [[texture(1)]],
              texture_buffer<float, access::read_write> outData [[texture(2)]],
              uint gid [[thread_position_in_grid]]) {
    uint w = inData.get_width();
    uint i = idx.read(gid).x;
    float4 v = inData.read(i);
    outData.write(float4(v.x + float(w)), gid);
    float o = outData.read(gid + 1).x;
    outData.write(float4(o), gid + 2);
}
"""

SOURCES = {
    "hlsl": (HLSL_TEXEL_BUFFERS, ".hlsl", "CSMain"),
    "glsl": (GLSL_TEXEL_BUFFERS, ".comp", "main"),
    "vulkan-glsl": (VULKAN_GLSL_TEXEL_BUFFERS, ".comp", "main"),
    "msl": (MSL_TEXEL_BUFFERS, ".metal", "k"),
}


def _validate_compute_glsl(tmp_path, glsl):
    glslang = shutil.which("glslangValidator")
    if glslang is None:
        return
    path = tmp_path / "shader.comp"
    path.write_text(glsl, encoding="utf-8")
    result = subprocess.run(
        [glslang, str(path)], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stdout + result.stderr


def _compile_compute_hlsl(tmp_path, hlsl, entry):
    dxc = shutil.which("dxc")
    if dxc is None:
        return
    path = tmp_path / "shader.hlsl"
    path.write_text(hlsl, encoding="utf-8")
    result = subprocess.run(
        [dxc, "-T", "cs_6_0", "-E", entry, str(path)],
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    assert result.returncode == 0, (result.stdout + result.stderr)[-2000:]


def _image_operands(code, name):
    """``(component, Dim, Sampled, Format)`` of the image a ``UniformConstant`` variable points to."""
    definitions = _definitions(code)
    opcode, operands = definitions[_id_of(code, name)]
    assert opcode == "OpVariable" and operands[1] == "UniformConstant"
    pointer_opcode, pointer_operands = definitions[operands[0]]
    assert pointer_opcode == "OpTypePointer"
    image_opcode, image = definitions[pointer_operands[1]]
    assert image_opcode == "OpTypeImage", (name, image_opcode)
    component = {
        ("OpTypeFloat", ("32",)): "float",
        ("OpTypeInt", ("32", "1")): "int",
        ("OpTypeInt", ("32", "0")): "uint",
    }[(definitions[image[0]][0], tuple(definitions[image[0]][1]))]
    return component, image[1], image[5], image[6]


def _crossgl(tmp_path, source, suffix):
    return _translate(tmp_path, source, suffix, "cgl")


@pytest.mark.parametrize(
    ("declaration"),
    [
        "samplerBuffer pairs;",
        "isamplerBuffer ints;",
        "samplerBuffer colorsIn;",
        "uimageBuffer counters;",
        "iimageBuffer ipairs;",
        "imageBuffer colors;",
    ],
)
def test_hlsl_texel_buffers_import_as_crossgl_texel_buffers(tmp_path, declaration):
    crossgl = _crossgl(tmp_path, HLSL_TEXEL_BUFFERS, ".hlsl")
    assert re.search(rf"^\s*{re.escape(declaration)}$", crossgl, re.M), crossgl
    assert "Buffer<" not in crossgl


def test_hlsl_storage_texel_buffers_take_the_format_of_the_bound_view(tmp_path):
    crossgl = _crossgl(tmp_path, HLSL_TEXEL_BUFFERS, ".hlsl")
    for name in ("counters", "ipairs", "colors"):
        assert re.search(
            rf"@ format\(unknown\)\n\s*\w+ {name};", crossgl
        ), f"{name} lacks @ format(unknown)"


@pytest.mark.parametrize(
    "expected",
    [
        "return texelFetch(b, i).xy;",
        "vec2 p = readPair(pairs, id.x) + texelFetch(pairs, id.x + 1).xy;",
        "int k = texelFetch(ints, id.x).x;",
        "imageStore(ipairs, id.x, ivec4(ivec2(k, k + 1), 0, 1));",
        "original = imageAtomicAdd(counters, id.x, 1u);",
        "imageAtomicMax(counters, 0, original);",
        "swapped = imageAtomicCompSwap(counters, 1, 0u, id.x);",
        "n = uint(imageSize(counters));",
        "m = uint(textureSize(pairs));",
    ],
)
def test_hlsl_texel_buffer_accesses_import_as_texel_operations(tmp_path, expected):
    assert expected in _crossgl(tmp_path, HLSL_TEXEL_BUFFERS, ".hlsl")


def test_hlsl_buffer_parameters_import_as_texel_buffers(tmp_path):
    crossgl = _crossgl(tmp_path, HLSL_TEXEL_BUFFERS, ".hlsl")
    assert "vec2 readPair(samplerBuffer b, uint i)" in crossgl


def test_vulkan_glsl_texture_buffers_import_as_sampler_buffers(tmp_path):
    crossgl = _crossgl(tmp_path, VULKAN_GLSL_TEXEL_BUFFERS, ".comp")
    assert re.search(r"^\s*samplerBuffer tb\b", crossgl, re.M), crossgl
    assert re.search(r"^\s*usamplerBuffer utb\b", crossgl, re.M), crossgl
    assert "cbuffer" not in crossgl


def test_msl_texture_buffer_width_is_a_scalar_size(tmp_path):
    crossgl = _crossgl(tmp_path, MSL_TEXEL_BUFFERS, ".metal")
    assert "uint w = imageSize(inData);" in crossgl


@pytest.mark.parametrize("source", sorted(SOURCES))
def test_texel_buffers_compile_as_glsl(tmp_path, source):
    text, suffix, _entry = SOURCES[source]
    _validate_compute_glsl(tmp_path, _translate(tmp_path, text, suffix, "opengl"))


@pytest.mark.parametrize("source", sorted(SOURCES))
def test_texel_buffers_compile_as_metal(tmp_path, source):
    text, suffix, _entry = SOURCES[source]
    metal = _translate(tmp_path, text, suffix, "metal")
    assert "texture_buffer<" in metal
    _compile_metal(tmp_path, metal)


@pytest.mark.parametrize("source", sorted(SOURCES))
def test_texel_buffers_validate_as_spirv(tmp_path, source):
    text, suffix, _entry = SOURCES[source]
    _validate(tmp_path, _translate(tmp_path, text, suffix, "vulkan"))


@pytest.mark.parametrize("source", sorted(SOURCES))
def test_texel_buffers_compile_as_hlsl(tmp_path, source):
    text, suffix, _entry = SOURCES[source]
    hlsl = _translate(tmp_path, text, suffix, "directx")
    assert re.search(r"\bR?W?Buffer<", hlsl)
    entry = re.search(r"\[numthreads\([^)]*\)\]\s*void (\w+)", hlsl).group(1)
    _compile_compute_hlsl(tmp_path, hlsl, entry)


def test_spirv_texel_buffers_are_buffer_images_without_samplers(tmp_path):
    code = _translate(tmp_path, HLSL_TEXEL_BUFFERS, ".hlsl", "vulkan")
    assert _image_operands(code, "pairs") == ("float", "Buffer", "1", "Unknown")
    assert _image_operands(code, "ints") == ("int", "Buffer", "1", "Unknown")
    assert _image_operands(code, "colors") == ("float", "Buffer", "2", "Unknown")
    # Image atomics need a single-channel 32-bit integer format.
    assert _image_operands(code, "counters") == ("uint", "Buffer", "2", "R32ui")
    assert "OpTypeSampledImage" not in code
    for capability in ("SampledBuffer", "ImageBuffer", "ImageQuery"):
        assert f"OpCapability {capability}\n" in code
    fetches = re.findall(r"= OpImageFetch %\w+ %\w+ %\w+(.*)$", code, re.M)
    assert fetches and all(not operands.strip() for operands in fetches), fetches


def test_hlsl_declares_storage_texel_buffers_by_format_and_atomic_use(tmp_path):
    hlsl = _translate(tmp_path, GLSL_TEXEL_BUFFERS, ".comp", "directx")
    assert "Buffer<float4> inData : register(t0);" in hlsl
    assert "RWBuffer<float> outData" in hlsl
    assert "RWBuffer<uint4> roData" in hlsl
    assert "RWBuffer<uint> counters" in hlsl
    # A one-channel read fills (r, 0, 0, 1) and a store keeps the stored channel.
    assert re.search(r"float4\(outData\[[^\]]+\], 0\.0, 0\.0, 1\.0\)\.x", hlsl), hlsl
    assert re.search(r"outData\[[^\]]+\] =\s*\(float4\(o, o, o, o\)\)\.x;", hlsl), hlsl


def test_hlsl_unformatted_atomic_texel_buffer_is_single_channel(tmp_path):
    hlsl = _translate(tmp_path, HLSL_TEXEL_BUFFERS, ".hlsl", "directx")
    assert "RWBuffer<uint> counters : register(u0);" in hlsl
    assert "RWBuffer<int4> ipairs : register(u1);" in hlsl
    assert (
        "InterlockedCompareExchange(buffer[coord], value0, value1, original);" in hlsl
    )


def test_metal_texel_buffers_use_texture_buffer_members(tmp_path):
    metal = _translate(tmp_path, GLSL_TEXEL_BUFFERS, ".comp", "metal")
    assert "texture_buffer<float, access::read> inData" in metal
    assert "texture_buffer<uint, access::read> roData" in metal
    assert "int(inData.get_width())" in metal
    assert "counters.atomic_fetch_add(uint(" in metal
    assert "atomic_compare_exchange_weak(coord, &original, value)" in metal


def test_metal_refuses_a_store_to_a_read_only_texel_buffer(tmp_path):
    source = """#version 450
layout(local_size_x = 1) in;
layout(binding = 0, r32f) uniform readonly imageBuffer ro;
void main() { imageStore(ro, 0, vec4(1.0)); }
"""
    with pytest.raises(Exception, match="read-only|readonly|writable"):
        _translate(tmp_path, source, ".comp", "metal")


def test_glsl_es_storage_texel_buffers_need_es_320_or_the_extension(tmp_path):
    source = """#version 310 es
precision highp float;
precision highp imageBuffer;
layout(local_size_x = 1) in;
layout(binding = 0, r32f) uniform highp writeonly imageBuffer outData;
void main() { imageStore(outData, 0, vec4(1.0)); }
"""
    glsl = _translate(tmp_path, source, ".comp", "opengl")
    assert glsl.startswith("#version 320 es") or (
        "#extension GL_EXT_texture_buffer" in glsl
    )
    _validate_compute_glsl(tmp_path, glsl)


SINGLE_CHANNEL_IMAGES = """#version 450
layout(local_size_x = 8, local_size_y = 8) in;
layout(binding = 0, r32f) uniform image2D heights;
void main() {
    ivec2 p = ivec2(gl_GlobalInvocationID.xy);
    float h = imageLoad(heights, p).x;
    imageStore(heights, p, vec4(h + 1.0));
}
"""


def test_spirv_single_channel_image_reads_have_four_components(tmp_path):
    """Vulkan requires four-component ``OpImageRead`` results (VUID-StandaloneSpirv-Result-04780)."""
    code = _translate(tmp_path, SINGLE_CHANNEL_IMAGES, ".comp", "vulkan")
    float_id = re.search(r"(%\w+) = OpTypeFloat 32", code).group(1)
    vec4_id = re.search(rf"(%\w+) = OpTypeVector {float_id} 4\b", code).group(1)
    assert re.search(rf"= OpImageRead {vec4_id} ", code), code
    _validate(tmp_path, code)


def test_spirv_atomic_target_of_an_hlsl_typed_uav_is_r32ui(tmp_path):
    source = """RWTexture2D<uint> histogram : register(u0);
[numthreads(8, 8, 1)]
void CSMain(uint3 id : SV_DispatchThreadID)
{
    uint before = histogram[id.xy];
    InterlockedAdd(histogram[id.xy], before + 1u);
}
"""
    code = _translate(tmp_path, source, ".hlsl", "vulkan")
    assert _image_operands(code, "histogram") == ("uint", "2D", "2", "R32ui")
    _validate(tmp_path, code)


HLSL_TEXEL_COMPONENT_ACCESSES = """RWBuffer<float4> rb : register(u0);
RWTexture2D<float4> rt : register(u1);
Buffer<float4> sb : register(t0);
RWTexture2D<float4> images[2] : register(u2);
RWStructuredBuffer<float> o : register(u4);
[numthreads(1, 1, 1)]
void CSMain(uint3 t : SV_DispatchThreadID)
{
    float a = rb[t.x][2];
    float b = rt[t.xy][1];
    float c = sb[t.x][3];
    float d = images[1][t.xy][2];
    rt[t.xy][1] = a + b;
    rb[t.x][3] = c + d;
    o[0] = a + b + c + d;
}
"""


def test_hlsl_texel_component_subscripts_import_after_the_texel_load(tmp_path):
    """``image[p][i]`` is component ``i`` of the texel at ``p``, not image ``p``'s texel ``i``."""
    cgl = _crossgl(tmp_path, HLSL_TEXEL_COMPONENT_ACCESSES, ".hlsl")
    assert "float a = imageLoad(rb, t.x)[2];" in cgl
    assert "float b = imageLoad(rt, t.xy)[1];" in cgl
    assert "float c = texelFetch(sb, t.x)[3];" in cgl
    assert "float d = imageLoad(images[1], t.xy)[2];" in cgl
    # A component store reads, modifies and writes the whole texel.
    assert re.search(
        r"vec4 (\w+) = imageLoad\(rt, t\.xy\);\s*\1\.y = a \+ b;\s*"
        r"imageStore\(rt, t\.xy, \1\);",
        cgl,
    ), cgl
    assert re.search(
        r"vec4 (\w+) = imageLoad\(rb, t\.x\);\s*\1\.w = c \+ d;\s*"
        r"imageStore\(rb, t\.x, \1\);",
        cgl,
    ), cgl


def test_hlsl_texel_component_subscripts_round_trip(tmp_path):
    hlsl = _translate(tmp_path, HLSL_TEXEL_COMPONENT_ACCESSES, ".hlsl", "directx")
    # A subscripted load is not narrowed to the element's scalar context.
    assert "float b = rt[t.xy][1];" in hlsl
    assert "float d = images[1][t.xy][2];" in hlsl
    _compile_compute_hlsl(tmp_path, hlsl, "CSMain")
    _validate_compute_glsl(
        tmp_path, _translate(tmp_path, HLSL_TEXEL_COMPONENT_ACCESSES, ".hlsl", "opengl")
    )


def test_unsized_texel_buffer_arrays_are_resource_arrays(tmp_path):
    source = """shader main {
    samplerBuffer coefficients[][2];
    compute {
        @numthreads(1, 1, 1)
        void main(uvec3 t @gl_GlobalInvocationID) {
            vec4 v = texelFetch(coefficients[t.x][1], 0);
        }
    }
}
"""
    hlsl = _translate(tmp_path, source, ".cgl", "directx")
    assert "Buffer<float4> coefficients[][2] : register(t0);" in hlsl
    assert "coefficients[t.x][1].Load(0)" in hlsl

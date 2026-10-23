"""Storage-image texels keep GLSL's four-component shape on every backend.

CrossGL ``imageLoad`` returns four components whatever the image format, filling the channels a narrower format
lacks with ``(0, 0, 1)``, and ``imageStore`` takes four components and ignores those the format lacks (GLSL 4.60
section 8.12; Vulkan "Conversion to RGBA"; MSL ``read``/``write``; Direct3D typed UAV format conversion).

Before, Metal, HLSL and SPIR-V typed a load from an ``r32f`` or ``rg32f`` image with the format's width: HLSL
splatted ``float4(r, r, r, r)``, Metal and SPIR-V refused ``.w`` or ``.xy`` on such loads, and every backend but
GLSL refused to store a ``vec4`` into an ``rg32f`` image. HLSL also declared unformatted integer UAVs with a
single component, which dropped the other components of ``RWTexture2D<uint4>`` and ``RWTexture2D<int2>``.
"""

from __future__ import annotations

import re

import pytest

from tests.test_translator.test_comparison_samplers import _compile_metal
from tests.test_translator.test_spirv_separate_textures import (
    _definitions,
    _translate,
    _validate,
)
from tests.test_translator.test_texel_buffers import (
    _compile_compute_hlsl,
    _validate_compute_glsl,
)

GLSL_FORMATTED_IMAGES = """#version 450
layout(local_size_x = 8, local_size_y = 8) in;
layout(binding = 0, r32f) uniform image2D imgR;
layout(binding = 1, rg32f) uniform image2D imgRG;
layout(binding = 2, r32ui) uniform uimage2D imgU;
layout(binding = 3, rgba8) uniform image2D imgC;
layout(binding = 4, rg32i) uniform iimage2D imgI;
layout(std430, binding = 5) buffer OutBuf { vec4 o[]; };
void main() {
    ivec2 p = ivec2(gl_GlobalInvocationID.xy);
    vec4 a = imageLoad(imgR, p);
    vec4 b = imageLoad(imgRG, p);
    uvec4 c = imageLoad(imgU, p);
    float d = imageLoad(imgR, p).x;
    float w = imageLoad(imgR, p).w;
    vec2 e = imageLoad(imgRG, p).xy;
    vec2 g = imageLoad(imgR, p).xy;
    ivec4 f = imageLoad(imgI, p);
    o[p.x] = a + b + imageLoad(imgC, p) + vec4(d) + vec4(e, g) + vec4(f) + vec4(w);
    imageStore(imgRG, p, a);
    imageStore(imgU, p, c + 1u);
    imageStore(imgI, p, f * 2);
}
"""

HLSL_TYPED_UAVS = """RWTexture2D<uint4> wide : register(u0);
RWTexture2D<int2> pairs : register(u1);
RWTexture2D<float> scalars : register(u2);
RWTexture2D<uint> counters : register(u3);
[numthreads(8, 8, 1)]
void CSMain(uint3 id : SV_DispatchThreadID) {
    uint4 v = wide[id.xy];
    wide[id.xy] = v + uint4(1, 2, 3, 4);
    int2 w = pairs[id.xy];
    pairs[id.xy] = w * 2;
    float f = scalars[id.xy];
    scalars[id.xy] = f + 1.0;
    uint original;
    InterlockedAdd(counters[id.xy], 1u, original);
}
"""

SOURCES = {
    "glsl": (GLSL_FORMATTED_IMAGES, ".comp"),
    "hlsl": (HLSL_TYPED_UAVS, ".hlsl"),
}


def _statement(code, variable):
    match = re.search(rf"\b{variable} = ([^;]+);", code)
    assert match, (variable, code[-2000:])
    return re.sub(r"\s+", " ", match.group(1))


@pytest.mark.parametrize("source", sorted(SOURCES))
def test_storage_image_texels_compile_as_glsl(tmp_path, source):
    text, suffix = SOURCES[source]
    _validate_compute_glsl(tmp_path, _translate(tmp_path, text, suffix, "opengl"))


@pytest.mark.parametrize("source", sorted(SOURCES))
def test_storage_image_texels_compile_as_metal(tmp_path, source):
    text, suffix = SOURCES[source]
    _compile_metal(tmp_path, _translate(tmp_path, text, suffix, "metal"))


@pytest.mark.parametrize("source", sorted(SOURCES))
def test_storage_image_texels_validate_as_spirv(tmp_path, source):
    text, suffix = SOURCES[source]
    _validate(tmp_path, _translate(tmp_path, text, suffix, "vulkan"))


@pytest.mark.parametrize("source", sorted(SOURCES))
def test_storage_image_texels_compile_as_hlsl(tmp_path, source):
    text, suffix = SOURCES[source]
    _compile_compute_hlsl(
        tmp_path, _translate(tmp_path, text, suffix, "directx"), "CSMain"
    )


def test_hlsl_widens_narrow_format_loads_with_the_texel_fill(tmp_path):
    hlsl = _translate(tmp_path, GLSL_FORMATTED_IMAGES, ".comp", "directx")
    assert "RWTexture2D<float> imgR" in hlsl
    assert "RWTexture2D<float2> imgRG" in hlsl
    # (r, 0, 0, 1), not the HLSL scalar splat (r, r, r, r).
    assert _statement(hlsl, "a") == "float4(imgR[p], 0.0, 0.0, 1.0)"
    assert _statement(hlsl, "b") == "float4(imgRG[p], 0.0, 1.0)"
    assert _statement(hlsl, "c") == "uint4(imgU[p], 0u, 0u, 1u)"
    # A swizzle within the element reads the element as it is.
    assert _statement(hlsl, "d") == "imgR[p].x"
    assert _statement(hlsl, "e") == "imgRG[p].xy"
    assert _statement(hlsl, "w") == "float4(imgR[p], 0.0, 0.0, 1.0).w"
    assert _statement(hlsl, "g") == "float4(imgR[p], 0.0, 0.0, 1.0).xy"


def test_hlsl_stores_keep_the_components_the_element_holds(tmp_path):
    hlsl = _translate(tmp_path, GLSL_FORMATTED_IMAGES, ".comp", "directx")
    assert _statement(hlsl, r"imgRG\[p\]") == "(a).xy"
    assert _statement(hlsl, r"imgU\[p\]").endswith(".x")
    assert _statement(hlsl, r"imgI\[p\]").endswith(".xy")


def test_metal_reads_four_components_and_writes_four_wide_values(tmp_path):
    metal = _translate(tmp_path, GLSL_FORMATTED_IMAGES, ".comp", "metal")
    assert _statement(metal, "w") == "imgR.read(uint2(p)).w"
    assert _statement(metal, "g") == "imgR.read(uint2(p)).xy"
    assert _statement(metal, "a") == "imgR.read(uint2(p))"
    assert "imgRG.write(a, uint2(p))" in metal


def test_spirv_reads_four_component_texels(tmp_path):
    code = _translate(tmp_path, GLSL_FORMATTED_IMAGES, ".comp", "vulkan")
    definitions = _definitions(code)
    reads = re.findall(r"= OpImageRead (%\w+)", code)
    assert len(reads) == 9, reads
    for result_type in reads:
        opcode, operands = definitions[result_type]
        assert opcode == "OpTypeVector" and operands[1] == "4", (opcode, operands)
    # Rg32f and Rg32i are extended storage formats.
    assert "OpCapability StorageImageExtendedFormats\n" in code


def test_hlsl_unformatted_integer_uavs_keep_their_width(tmp_path):
    crossgl = _translate(tmp_path, HLSL_TYPED_UAVS, ".hlsl", "cgl")
    assert "uimage2D wide;" in crossgl and "iimage2D pairs;" in crossgl
    hlsl = _translate(tmp_path, HLSL_TYPED_UAVS, ".hlsl", "directx")
    assert "RWTexture2D<uint4> wide" in hlsl
    assert "RWTexture2D<int4> pairs" in hlsl
    assert "RWTexture2D<float4> scalars" in hlsl
    # An image atomic needs a single-component element.
    assert "RWTexture2D<uint> counters" in hlsl
    assert _statement(hlsl, "uint4 v") == "wide[id.xy]"
    assert _statement(hlsl, "int2 w") == "pairs[id.xy].xy"


def test_glsl_narrow_contexts_take_the_leading_components(tmp_path):
    crossgl = """shader S {
    compute {
        @ format(r32f) image2D r;
        @ format(rg32f) image2D rg;
        @ numthreads(1, 1, 1)
        void main() {
            float s = imageLoad(r, ivec2(0, 0));
            vec2 t = imageLoad(rg, ivec2(0, 0));
            imageStore(rg, ivec2(1, 1), vec4(t, s, 1.0));
        }
    }
}
"""
    glsl = _translate(tmp_path, crossgl, ".cgl", "opengl")
    assert _statement(glsl, "float s") == "imageLoad(r, ivec2(0, 0)).x"
    assert _statement(glsl, "vec2 t") == "imageLoad(rg, ivec2(0, 0)).xy"
    hlsl = _translate(tmp_path, crossgl, ".cgl", "directx")
    assert _statement(hlsl, "float s") == "r[int2(0, 0)]"
    assert _statement(hlsl, "float2 t") == "rg[int2(0, 0)]"
    metal = _translate(tmp_path, crossgl, ".cgl", "metal")
    assert _statement(metal, "float s") == "r.read(uint2(int2(0, 0))).x"


ATOMICS_THROUGH_PARAMETERS = """shader AtomicsThroughParameters {
    uimage2D counters;
    uimage2D joined;
    uimage2D plain;
    uimageBuffer bufferCounters;
    uimageBuffer bufferPlain;
    uint bump(uimage2D image, ivec2 pixel, uint value) {
        return imageAtomicAdd(image, pixel, value);
    }
    uint readBoth(uimage2D image, ivec2 pixel) {
        return imageLoad(image, pixel).x;
    }
    uint readPlain(uimage2D image, ivec2 pixel) {
        return imageLoad(image, pixel).y;
    }
    uint bumpBuffer(uimageBuffer buffer, int index, uint value) {
        return imageAtomicMax(buffer, index, value);
    }
    uint readBuffer(uimageBuffer buffer, int index) {
        return imageLoad(buffer, index).y;
    }
    compute {
        @ numthreads(1, 1, 1)
        void main() {
            uint a = bump(counters, ivec2(0, 1), 2u);
            uint b = readBoth(joined, ivec2(0, 1)) + readBoth(counters, ivec2(1, 1));
            uint c = readPlain(plain, ivec2(0, 1));
            uint d = bumpBuffer(bufferCounters, 0, a + b + c);
            imageStore(bufferPlain, 1, uvec4(d + readBuffer(bufferPlain, 2)));
        }
    }
}
"""


def test_hlsl_atomics_through_parameters_need_scalar_elements(tmp_path):
    hlsl = _translate(tmp_path, ATOMICS_THROUGH_PARAMETERS, ".cgl", "directx")
    # The atomic reaches counters through bump's parameter. HLSL passes a
    # resource only to a parameter of its type, so readBoth's parameter and
    # every resource passed to it share the scalar element.
    for declaration in (
        "RWTexture2D<uint> counters",
        "RWTexture2D<uint> joined",
        "uint bump(RWTexture2D<uint> image",
        "uint readBoth(RWTexture2D<uint> image",
        "RWBuffer<uint> bufferCounters",
        "uint bumpBuffer(RWBuffer<uint> buffer",
    ):
        assert declaration in hlsl, declaration
    for declaration in (
        "RWTexture2D<uint4> plain",
        "uint readPlain(RWTexture2D<uint4> image",
        "RWBuffer<uint4> bufferPlain",
        "uint readBuffer(RWBuffer<uint4> buffer",
    ):
        assert declaration in hlsl, declaration
    assert "imageLoad(" not in hlsl
    _compile_compute_hlsl(tmp_path, hlsl, "CSMain")


NARROWING_CONSTRUCTORS = """shader NarrowingConstructors {
    @ format(r32f) image2D img;
    compute {
        @ numthreads(1, 1, 1)
        void main() {
            vec4 v = vec4(1.0, 2.0, 3.0, 4.0);
            vec2 a = vec2(v);
            vec3 b = vec3(v);
            ivec2 i = ivec2(v);
            float c = float(v);
            vec2 d = vec2(imageLoad(img, ivec2(0, 0)));
            imageStore(img, ivec2(1, 1), vec4(a, b.x, c + d.y + float(i.y)));
        }
    }
}
"""


def test_narrowing_vector_constructors_take_the_leading_components(tmp_path):
    # GLSL 4.60 section 5.4.2: vec3(vec4) drops the fourth component. HLSL and
    # Metal reject a vector constructor of a wider vector.
    metal = _translate(tmp_path, NARROWING_CONSTRUCTORS, ".cgl", "metal")
    hlsl = _translate(tmp_path, NARROWING_CONSTRUCTORS, ".cgl", "directx")
    for code in (metal, hlsl):
        assert _statement(code, "float2 a") == "float2(v.xy)"
        assert _statement(code, "float3 b") == "float3(v.xyz)"
        assert _statement(code, "int2 i") == "int2(v.xy)"
        assert _statement(code, "float c") == "float(v.x)"
    # The r32f texel is (r, 0, 0, 1), so d is (r, 0) on every backend.
    assert _statement(metal, "float2 d") == "float2(img.read(uint2(int2(0, 0))).xy)"
    assert _statement(hlsl, "float2 d") == (
        "float2(float4(img[int2(0, 0)], 0.0, 0.0, 1.0).xy)"
    )
    _compile_metal(tmp_path, metal)
    _compile_compute_hlsl(tmp_path, hlsl, "CSMain")


THREE_COMPONENT_STORES = """shader ThreeComponentStores {
    @ format(rgba16f) image2D colors;
    compute {
        @ numthreads(1, 1, 1)
        void main() {
            vec3 shortColor = vec3(1.0, 0.5, 0.25);
            imageStore(colors, ivec2(0, 0), shortColor);
        }
    }
}
"""

HLSL_TEXEL_BUFFER_SWIZZLE = """RWBuffer<int2> pairs : register(u0);
[numthreads(1, 1, 1)]
void CSMain(uint3 id : SV_DispatchThreadID) {
    pairs[id.x] += int2(1, 1);
}
"""


def test_three_component_stores_take_the_alpha_fill(tmp_path):
    # GLSL 4.60 section 8.12 stores a gvec4; CrossGL fills a vec3 to
    # (v, 1) for a four-channel format, as a narrower format's load does.
    glsl = _translate(tmp_path, THREE_COMPONENT_STORES, ".cgl", "opengl")
    assert "imageStore(colors, ivec2(0, 0), vec4(shortColor, 1.0));" in glsl
    _validate(tmp_path, _translate(tmp_path, THREE_COMPONENT_STORES, ".cgl", "vulkan"))


def test_metal_texel_buffer_swizzles_keep_their_width(tmp_path):
    # HLSL addition wraps, so Metal adds the bit casts; the swizzle of a
    # texel-buffer read is two components wide.
    metal = _translate(tmp_path, HLSL_TEXEL_BUFFER_SWIZZLE, ".hlsl", "metal")
    assert "as_type<uint2>(pairs.read(uint(id.x)).xy)" in metal
    _compile_metal(tmp_path, metal)

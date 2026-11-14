"""MSL textures import as separate CrossGL textures, so SPIR-V declares them as sampled images.

An MSL ``texture2d<T>`` or ``depth2d<T>`` is bound apart from the ``sampler`` its ``sample`` calls take (MSL 3.2
sections 2.9 and 6.12), so its Vulkan descriptor is a ``SAMPLED_IMAGE`` paired with a ``SAMPLER`` at each sample,
not a ``COMBINED_IMAGE_SAMPLER``. Depth textures keep their depth flag (``Depth`` 1 in ``OpTypeImage``), reads and
size queries go to the image itself, and the other targets still declare the texture they declared before.
"""

from __future__ import annotations

import re
import shutil
import subprocess

import pytest

from crosstl.backend.Metal.texture_types import combined_texture_type
from tests.test_translator.test_metal_stage_varyings import metal_compiler
from tests.test_translator.test_spirv_separate_textures import (
    _definitions,
    _id_of,
    _image_shape,
    _translate,
    _validate,
    _variable_pointee,
)

MSL_SAMPLED = """#include <metal_stdlib>
using namespace metal;
struct VOut { float4 pos [[position]]; float2 uv; };
fragment float4 fmain(VOut in [[stage_in]],
                      texture2d<float> albedo [[texture(0)]],
                      sampler linearSampler [[sampler(0)]]) {
    return albedo.sample(linearSampler, in.uv);
}
"""

MSL_DEPTH = """#include <metal_stdlib>
using namespace metal;
struct VOut { float4 pos [[position]]; float2 uv; };
fragment float4 fmain(VOut in [[stage_in]],
                      depth2d<float> shadowMap [[texture(0)]],
                      sampler shadowSampler [[sampler(0)]]) {
    float lit = shadowMap.sample_compare(shadowSampler, in.uv, 0.5);
    return float4(lit, lit, lit, 1.0);
}
"""

MSL_FETCHED = """#include <metal_stdlib>
using namespace metal;
struct VOut { float4 pos [[position]]; };
fragment float4 fmain(VOut in [[stage_in]],
                      texture2d<uint> labels [[texture(0)]],
                      texture2d_ms<float> msColor [[texture(1)]],
                      texturecube<float> environment [[texture(2)]]) {
    uint label = labels.read(uint2(1, 2)).x;
    float4 ms = msColor.read(uint2(0, 0), 1);
    uint width = labels.get_width();
    uint samples = msColor.get_num_samples();
    uint levels = environment.get_num_mip_levels();
    return ms + float4(float(label), float(width), float(samples), float(levels));
}
"""

MSL_TEXTURE_TYPES = """#include <metal_stdlib>
using namespace metal;
struct VOut {{ float4 pos [[position]]; }};
fragment float4 fmain(VOut in [[stage_in]], {texture_type} tex [[texture(0)]]) {{
    return float4(0.0);
}}
"""


def _sampled_image_descriptors(code):
    """The opcodes of the types every ``UniformConstant`` pointer points to."""
    definitions = _definitions(code)
    return {
        definitions[pointee][0]
        for pointee in re.findall(r"OpTypePointer UniformConstant (%\w+)", code)
    }


def _sampled_from(code, texture, sampler):
    """Whether an ``OpSampledImage`` pairs loads of the named texture and sampler variables."""
    image = re.search(rf"(%\w+) = OpLoad %\w+ {_id_of(code, texture)}\b", code)
    state = re.search(rf"(%\w+) = OpLoad %\w+ {_id_of(code, sampler)}\b", code)
    if image is None or state is None:
        return False
    return bool(
        re.search(rf"= OpSampledImage %\w+ {image.group(1)} {state.group(1)}\b", code)
    )


def test_msl_texture_is_a_sampled_image_paired_at_the_sample(tmp_path):
    code = _translate(tmp_path, MSL_SAMPLED, ".metal", "vulkan")

    assert _image_shape(code, "albedo") == ("float", "2D", "0", "0", "0")
    assert _variable_pointee(code, "linearSampler")[0] == "OpTypeSampler"
    assert "OpTypeSampledImage" not in _sampled_image_descriptors(code)
    assert _sampled_from(code, "albedo", "linearSampler")
    assert "OpImageSampleImplicitLod" in code
    _validate(tmp_path, code)


def test_msl_depth_texture_is_a_depth_image_compared_through_its_sampler(tmp_path):
    code = _translate(tmp_path, MSL_DEPTH, ".metal", "vulkan")

    # Depth operand 1: the image holds depth, so only depth-comparison sampling reads it.
    assert _image_shape(code, "shadowMap") == ("float", "2D", "1", "0", "0")
    assert _variable_pointee(code, "shadowSampler")[0] == "OpTypeSampler"
    assert "OpTypeSampledImage" not in _sampled_image_descriptors(code)
    assert _sampled_from(code, "shadowMap", "shadowSampler")
    assert "OpImageSampleDrefImplicitLod" in code
    _validate(tmp_path, code)


def test_msl_texture_reads_and_queries_use_the_image_without_a_sampler(tmp_path):
    code = _translate(tmp_path, MSL_FETCHED, ".metal", "vulkan")

    assert _image_shape(code, "labels") == ("uint", "2D", "0", "0", "0")
    assert _image_shape(code, "msColor") == ("float", "2D", "0", "0", "1")
    assert _image_shape(code, "environment") == ("float", "Cube", "0", "0", "0")
    assert _sampled_image_descriptors(code) == {"OpTypeImage"}
    assert "OpSampledImage" not in code
    assert code.count("OpImageFetch") == 2
    for query in ("OpImageQuerySizeLod", "OpImageQuerySamples", "OpImageQueryLevels"):
        assert query in code, query
    _validate(tmp_path, code)


@pytest.mark.parametrize(
    ("texture_type", "crossgl_type"),
    [
        ("texture1d<float>", "texture1D"),
        ("texture1d_array<half>", "texture1DArray"),
        ("texture2d<int>", "itexture2D"),
        ("texture2d_array<uint>", "utexture2DArray"),
        ("texture2d_ms<float>", "texture2DMS"),
        ("texture2d_ms_array<int>", "itexture2DMSArray"),
        ("texture3d<uint>", "utexture3D"),
        ("texturecube<float>", "textureCube"),
        ("texturecube_array<float>", "textureCubeArray"),
        ("depth2d<float>", "texture2DShadow"),
        ("depth2d_array<float>", "texture2DArrayShadow"),
        ("depthcube<float>", "textureCubeShadow"),
        ("depthcube_array<float>", "textureCubeArrayShadow"),
        # depth2d_ms offers only read, so it has no comparison sampling to keep.
        ("depth2d_ms<float>", "texture2DMS"),
        ("depth2d_ms_array<float>", "texture2DMSArray"),
        # A texel buffer never takes a sampler.
        ("texture_buffer<float>", "samplerBuffer"),
    ],
)
def test_msl_texture_types_import_as_separate_textures(
    tmp_path, texture_type, crossgl_type
):
    source = MSL_TEXTURE_TYPES.format(texture_type=texture_type)
    crossgl = _translate(tmp_path, source, ".metal", "cgl")

    assert f"{crossgl_type} tex @texture(0)" in crossgl


@pytest.mark.parametrize(
    ("type_name", "combined"),
    [
        ("texture2D", "sampler2D"),
        ("utexture2DArray", "usampler2DArray"),
        ("itexture2DMSArray", "isampler2DMSArray"),
        ("textureCubeArrayShadow", "samplerCubeArrayShadow"),
        ("samplerBuffer", "samplerBuffer"),
        ("image2D", "image2D"),
    ],
)
def test_combined_texture_type_respells_only_separate_textures(type_name, combined):
    assert combined_texture_type(type_name) == combined


def test_glsl_combines_msl_textures_with_their_samplers(tmp_path):
    glsl = _translate(tmp_path, MSL_SAMPLED, ".metal", "opengl")
    shadow = _translate(tmp_path, MSL_DEPTH, ".metal", "opengl")

    assert "uniform sampler2D albedo;" in glsl
    assert "texture(albedo, uv)" in glsl
    assert "uniform sampler2DShadow shadowMap;" in shadow
    assert "texture(shadowMap, vec3(uv, 0.5))" in shadow
    glslang = shutil.which("glslangValidator")
    if glslang is None:
        return
    for name, code in (("color.frag", glsl), ("shadow.frag", shadow)):
        path = tmp_path / name
        path.write_text(code, encoding="utf-8")
        result = subprocess.run(
            [glslang, str(path)], capture_output=True, text=True, check=False
        )
        assert result.returncode == 0, result.stdout + result.stderr


def test_hlsl_declares_msl_textures_with_their_sampler_states(tmp_path):
    hlsl = _translate(tmp_path, MSL_SAMPLED, ".metal", "directx")
    shadow = _translate(tmp_path, MSL_DEPTH, ".metal", "directx")

    assert "Texture2D albedo : register(t0);" in hlsl
    assert "SamplerState linearSampler : register(s0);" in hlsl
    assert "albedo.Sample(linearSampler, " in hlsl
    assert "Texture2D shadowMap : register(t0);" in shadow
    assert "SamplerComparisonState shadowSampler : register(s0);" in shadow
    assert "shadowMap.SampleCmp(shadowSampler, " in shadow


@pytest.mark.parametrize("source", [MSL_SAMPLED, MSL_DEPTH, MSL_FETCHED])
def test_metal_round_trip_keeps_the_msl_texture_types(tmp_path, source):
    metal = _translate(tmp_path, source, ".metal", "metal")

    for texture_type in re.findall(r"(\w+<\w+>) \w+ \[\[texture", source):
        assert texture_type in metal, texture_type
    _compile_metal(tmp_path, metal)


CGL_INTEGER_TEXTURE_SIZES = """shader main {
    usampler2D labels;
    isampler2DArray offsets;
    utexture3D volume;
    itextureCube faces;
    fragment {
        vec4 main(vec4 position @ gl_FragCoord) @ gl_FragColor {
            ivec2 a = textureSize(labels, 0);
            ivec3 b = textureSize(offsets, 1);
            ivec3 c = textureSize(volume, 0);
            ivec2 d = textureSize(faces, 0);
            return vec4(a.x, b.z, c.y, d.x);
        }
    }
}
"""


def test_metal_queries_the_size_of_integer_textures(tmp_path):
    # The size query depends on the image shape alone; an integer texture must not fall back to a
    # free textureSize call, which MSL does not have.
    metal = _translate(tmp_path, CGL_INTEGER_TEXTURE_SIZES, ".cgl", "metal")

    assert "textureSize(" not in metal
    assert "int2(labels.get_width(uint(0)), labels.get_height(uint(0)))" in metal
    assert "offsets.get_array_size()" in metal
    assert "volume.get_depth(uint(0))" in metal
    assert "int2(faces.get_width(uint(0)), faces.get_height(uint(0)))" in metal
    _compile_metal(tmp_path, metal)


def _compile_metal(tmp_path, metal):
    if metal_compiler() is None:
        return
    path = tmp_path / "round_trip.metal"
    path.write_text(metal, encoding="utf-8")
    result = subprocess.run(
        [
            "xcrun",
            "-sdk",
            "macosx",
            "metal",
            "-c",
            str(path),
            "-o",
            str(tmp_path / "round_trip.air"),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr[-2000:]

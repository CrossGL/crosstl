"""Separate textures stay separate images in SPIR-V, and the HLSL importer spells them as separate textures.

An HLSL ``Texture2D`` is an SRV bound apart from its ``SamplerState``, so its Vulkan descriptor is a
``SAMPLED_IMAGE``: an ``OpTypeImage`` (``Sampled`` 1) variable that sampling pairs with the sampler through
``OpSampledImage``. Declaring it as a ``COMBINED_IMAGE_SAMPLER`` (a pointer to ``OpTypeSampledImage``) makes the
host bind the wrong descriptor type. Fetches and size/level/sample-count queries read the image itself. The other
targets declare a separate texture exactly as the combined texture they sampled through explicit samplers before.
"""

from __future__ import annotations

import re
import shutil
import subprocess

import pytest

import crosstl
from crosstl.translator.codegen.spirv_separate_textures import (
    separate_texture_type_info,
)

HLSL_SAMPLED = """Texture2D<float4> albedo : register(t0);
SamplerState linearSampler : register(s0);

float4 main(float4 position : SV_Position, float2 uv : TEXCOORD0) : SV_Target {
    return albedo.Sample(linearSampler, uv);
}
"""

HLSL_FETCHED = """Texture2D<float4> albedo : register(t0);
Texture2DMS<float4> msColor : register(t1);

float4 main(float4 position : SV_Position) : SV_Target {
    uint width, height, levels;
    albedo.GetDimensions(0, width, height, levels);
    uint msWidth, msHeight, samples;
    msColor.GetDimensions(msWidth, msHeight, samples);
    return albedo.Load(int3(1, 2, 0)) + msColor.Load(int2(0, 0), 1)
        + float4(width, height, levels, samples);
}
"""

HLSL_COMPARED = """Texture2D<float> shadowMap : register(t0);
SamplerComparisonState shadowSampler : register(s0);

float4 main(float4 position : SV_Position, float2 uv : TEXCOORD0) : SV_Target {
    float lit = shadowMap.SampleCmpLevelZero(shadowSampler, uv, 0.5);
    return float4(lit, lit, lit, 1.0);
}
"""

CGL_SEPARATE_TYPES = """shader main {
    texture2D colorTexture;
    utexture2D labels;
    itexture2DArray offsets;
    texture2DMS msColor;
    texture2DMSArray msLayers;
    texture1DArray rows;
    utexture3D volume;
    textureCube environment;
    texture2DShadow shadowMap;
    sampler linearSampler;
    fragment {
        vec4 main(vec4 position @ gl_FragCoord) @ gl_FragColor {
            uint label = texelFetch(labels, ivec2(0, 0), 0).x;
            ivec4 offset = texelFetch(offsets, ivec3(0, 0, 1), 0);
            vec4 ms = texelFetch(msColor, ivec2(0, 0), 1) + texelFetch(msLayers, ivec3(0, 0, 1), 2);
            vec4 row = textureLod(rows, linearSampler, vec2(0.5, 1.0), 0.0);
            uvec4 voxel = texelFetch(volume, ivec3(0, 0, 0), 0);
            vec4 sky = texture(environment, linearSampler, position.xyz);
            float lit = textureCompareLod(shadowMap, linearSampler, position.xy, 0.5, 0.0);
            vec4 gathered = textureGather(colorTexture, linearSampler, position.xy);
            ivec2 size = textureSize(msColor);
            int sampleCount = textureSamples(msColor);
            int levels = textureQueryLevels(environment);
            return vec4(label, offset.x, voxel.x, lit) + ms + row + sky + gathered
                + vec4(size, sampleCount, levels);
        }
    }
}
"""

# (sampled type, Dim, Depth, Arrayed, MS) of each separate texture above; Sampled is 1 and the format Unknown.
EXPECTED_IMAGES = {
    "colorTexture": ("float", "2D", "0", "0", "0"),
    "labels": ("uint", "2D", "0", "0", "0"),
    "offsets": ("int", "2D", "0", "1", "0"),
    "msColor": ("float", "2D", "0", "0", "1"),
    "msLayers": ("float", "2D", "0", "1", "1"),
    "rows": ("float", "1D", "0", "1", "0"),
    "volume": ("uint", "3D", "0", "0", "0"),
    "environment": ("float", "Cube", "0", "0", "0"),
    "shadowMap": ("float", "2D", "1", "0", "0"),
}


def _translate(tmp_path, source, suffix, backend):
    path = tmp_path / f"shader{suffix}"
    path.write_text(source, encoding="utf-8")
    return crosstl.translate(str(path), backend=backend, format_output=False)


def _definitions(code):
    """``{result id: (opcode, operands)}`` of every instruction that defines an id."""
    definitions = {}
    for match in re.finditer(r"^\s*(%\w+) = (Op\w+)(.*)$", code, re.M):
        definitions[match.group(1)] = (match.group(2), match.group(3).split())
    return definitions


def _id_of(code, name):
    """The id ``OpName`` gives ``name``."""
    return re.search(rf'OpName (%\w+) "{name}"', code).group(1)


def _variable_pointee(code, name):
    """The opcode and operands of the type a named ``UniformConstant`` variable points to."""
    definitions = _definitions(code)
    opcode, operands = definitions[_id_of(code, name)]
    assert opcode == "OpVariable" and operands[1] == "UniformConstant"
    pointer_opcode, pointer_operands = definitions[operands[0]]
    assert pointer_opcode == "OpTypePointer"
    return definitions[pointer_operands[1]]


def _image_shape(code, name):
    definitions = _definitions(code)
    opcode, operands = _variable_pointee(code, name)
    assert opcode == "OpTypeImage", (name, opcode)
    sampled_type = definitions[operands[0]]
    component = {
        ("OpTypeFloat", ("32",)): "float",
        ("OpTypeInt", ("32", "1")): "int",
        ("OpTypeInt", ("32", "0")): "uint",
    }[(sampled_type[0], tuple(sampled_type[1]))]
    dim, depth, arrayed, multisampled, sampled, image_format = operands[1:7]
    assert (sampled, image_format) == ("1", "Unknown"), name
    return component, dim, depth, arrayed, multisampled


def _validate(tmp_path, code):
    assembler, validator = shutil.which("spirv-as"), shutil.which("spirv-val")
    if assembler is None or validator is None:
        return
    asm = tmp_path / "shader.spvasm"
    asm.write_text(code, encoding="utf-8")
    binary = tmp_path / "shader.spv"
    subprocess.run(
        [assembler, "--target-env", "vulkan1.0", str(asm), "-o", str(binary)],
        capture_output=True,
        check=True,
    )
    result = subprocess.run(
        [validator, "--target-env", "vulkan1.0", str(binary)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr[-2000:]


def test_hlsl_texture_is_a_sampled_image_paired_at_the_sample(tmp_path):
    code = _translate(tmp_path, HLSL_SAMPLED, ".hlsl", "vulkan")

    assert _image_shape(code, "albedo") == ("float", "2D", "0", "0", "0")
    assert _variable_pointee(code, "linearSampler")[0] == "OpTypeSampler"
    # No combined image-sampler descriptor is declared.
    assert not any(
        _definitions(code)[pointee][0] == "OpTypeSampledImage"
        for pointee in re.findall(r"OpTypePointer UniformConstant (%\w+)", code)
    )
    image = re.search(rf"(%\w+) = OpLoad %\w+ {_id_of(code, 'albedo')}\b", code)
    sampler = re.search(
        rf"(%\w+) = OpLoad %\w+ {_id_of(code, 'linearSampler')}\b", code
    )
    image, sampler = image.group(1), sampler.group(1)
    assert re.search(rf"= OpSampledImage %\w+ {image} {sampler}\b", code)
    assert "OpImageSampleImplicitLod" in code
    assert " = OpImage " not in code
    _validate(tmp_path, code)


def test_hlsl_texture_fetches_and_queries_read_the_image(tmp_path):
    code = _translate(tmp_path, HLSL_FETCHED, ".hlsl", "vulkan")

    assert _image_shape(code, "albedo") == ("float", "2D", "0", "0", "0")
    assert _image_shape(code, "msColor") == ("float", "2D", "0", "0", "1")
    assert "OpSampledImage" not in code and " = OpImage " not in code
    assert code.count("OpImageFetch") == 2
    assert "OpImageQuerySizeLod" in code and "OpImageQueryLevels" in code
    assert "OpImageQuerySamples" in code and "OpImageQuerySize " in code
    _validate(tmp_path, code)


def test_separate_texture_types_declare_their_image_shapes(tmp_path):
    code = _translate(tmp_path, CGL_SEPARATE_TYPES, ".cgl", "vulkan")

    for name, shape in EXPECTED_IMAGES.items():
        assert _image_shape(code, name) == shape, name
    assert "OpImageSampleDrefExplicitLod" in code and "OpImageGather" in code
    _validate(tmp_path, code)


@pytest.mark.parametrize(
    "type_name",
    ["texture3DArray", "texture1DMS", "utexture2DShadow", "texture2DMSShadow", "foo"],
)
def test_image_shapes_spir_v_cannot_declare_are_not_separate_textures(type_name):
    assert separate_texture_type_info(type_name) is None


def test_compare_only_hlsl_texture_imports_as_a_depth_texture(tmp_path):
    crossgl = _translate(tmp_path, HLSL_COMPARED, ".hlsl", "cgl")
    assert "texture2DShadow shadowMap;" in crossgl

    spirv = _translate(tmp_path, HLSL_COMPARED, ".hlsl", "vulkan")
    assert _image_shape(spirv, "shadowMap") == ("float", "2D", "1", "0", "0")
    assert "OpImageSampleDrefExplicitLod" in spirv
    _validate(tmp_path, spirv)

    glsl = _translate(tmp_path, HLSL_COMPARED, ".hlsl", "opengl")
    assert re.search(r"uniform sampler2DShadow shadowMap;", glsl)
    hlsl = _translate(tmp_path, HLSL_COMPARED, ".hlsl", "directx")
    assert re.search(r"\bTexture2D\b(<float>)? shadowMap", hlsl)
    assert "SampleCmpLevelZero" in hlsl
    metal = _translate(tmp_path, HLSL_COMPARED, ".hlsl", "metal")
    assert "depth2d<float> shadowMap" in metal


SEPARATE_TEXTURE_PROGRAM = """shader main {
    TEXTURE_TYPE albedo;
    sampler linearSampler;
    fragment {
        vec4 main(vec4 position @ gl_FragCoord) @ gl_FragColor {
            ivec2 size = textureSize(albedo, 0);
            int width = textureSize(albedo, 0).x;
            return texture(albedo, linearSampler, position.xy)
                + texelFetch(albedo, ivec2(0, 0), 0) + vec4(size, width, 0);
        }
    }
}
"""


@pytest.mark.parametrize(
    "backend", ["directx", "metal", "opengl", "slang", "rust", "mojo", "cuda", "hip"]
)
def test_targets_declare_a_separate_texture_like_an_explicitly_sampled_one(
    tmp_path, backend
):
    separate = _translate(
        tmp_path,
        SEPARATE_TEXTURE_PROGRAM.replace("TEXTURE_TYPE", "texture2D"),
        ".cgl",
        backend,
    )
    combined = _translate(
        tmp_path,
        SEPARATE_TEXTURE_PROGRAM.replace("TEXTURE_TYPE", "sampler2D"),
        ".cgl",
        backend,
    )

    assert separate == combined
    assert not re.search(r"\btexture2D albedo\b", separate)


def test_cuda_texture_size_infers_its_vector_result(tmp_path):
    code = _translate(
        tmp_path,
        SEPARATE_TEXTURE_PROGRAM.replace("TEXTURE_TYPE", "sampler2D"),
        ".cgl",
        "cuda",
    )

    # The member read needs the query's vector type; reading it once raised a bare KeyError.
    assert re.search(
        r"int2 size = cgl_textureSize_sampler2D\(albedo_metadata, 0\);", code
    )
    assert re.search(
        r"int width = cgl_textureSize_sampler2D\(albedo_metadata, 0\)\.x;", code
    )

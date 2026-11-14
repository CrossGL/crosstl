"""Explicit samplers reach every target: one sampler state per (texture, sampler) pair.

HLSL, MSL and Vulkan GLSL sample a texture through a sampler object named at each call. A texture filtered by two
samplers must keep two sampler states: SPIR-V pairs the image with the operation's sampler (``OpSampledImage``),
and OpenGL GLSL, which has no sampler objects in the language, gives each pair its own combined uniform.
"""

import json
import re
import shutil
import subprocess

import pytest

import crosstl
from crosstl.translator.artifact_contracts import (
    COMBINED_SAMPLER_PREFIX,
    artifact_contract_headers,
)
from crosstl.translator.codegen.glsl_refusals import OpenGLLoweringRefused
from crosstl.translator.codegen.metal_refusals import MetalLoweringRefused

TWO_SAMPLERS_HLSL = """
Texture2D albedo : register(t0);
SamplerState pointSampler : register(s0);
SamplerState linearSampler : register(s1);
float4 main(float2 uv : TEXCOORD0) : SV_Target {
    return albedo.Sample(pointSampler, uv) + albedo.Sample(linearSampler, uv);
}
"""

COMPARE_AND_SAMPLE_HLSL = """
Texture2D shadowMap : register(t0);
SamplerComparisonState lessCompare : register(s0);
SamplerState pointSampler : register(s1);
float4 main(float2 uv : TEXCOORD0) : SV_Target {
    float lit = shadowMap.SampleCmp(lessCompare, uv, 0.5);
    float level0 = shadowMap.SampleCmpLevelZero(lessCompare, uv, 0.25);
    return float4(lit, level0, shadowMap.Sample(pointSampler, uv).r, 1.0);
}
"""

HELPERS_HLSL = """
Texture2D albedo : register(t0);
Texture2D<float> depthTex : register(t1);
SamplerState pointSampler : register(s0);
SamplerState linearSampler : register(s1);
SamplerComparisonState compareSampler : register(s2);
float4 fetch(Texture2D t, SamplerState s, float2 uv) { return t.Sample(s, uv); }
float4 nested(Texture2D t, SamplerState s, float2 uv) { return fetch(t, s, uv) * 2.0; }
float4 viaGlobal(SamplerState s, float2 uv) { return albedo.Sample(s, uv); }
float4 main(float2 uv : TEXCOORD0) : SV_Target {
    float lit = depthTex.SampleCmp(compareSampler, uv, 0.5);
    float raw = depthTex.Load(int3(0, 0, 0)).x;
    return fetch(albedo, pointSampler, uv) + nested(albedo, linearSampler, uv)
        + viaGlobal(pointSampler, uv) + float4(lit, raw, 0.0, 0.0);
}
"""

VULKAN_SEPARATE_GLSL = """#version 450
layout(set = 0, binding = 0) uniform texture2D depthTex;
layout(set = 0, binding = 1) uniform sampler compareSampler;
layout(set = 0, binding = 2) uniform sampler pointSampler;
layout(location = 0) in vec2 uv;
layout(location = 0) out vec4 outColor;
void main() {
    float lit = texture(sampler2DShadow(depthTex, compareSampler), vec3(uv, 0.5));
    outColor = vec4(lit) + texture(sampler2D(depthTex, pointSampler), uv);
}
"""

VULKAN_ONE_SAMPLER_GLSL = """#version 450
layout(set = 0, binding = 0) uniform texture2D albedo;
layout(set = 0, binding = 1) uniform sampler linearSampler;
layout(location = 0) in vec2 uv;
layout(location = 0) out vec4 outColor;
void main() {
    outColor = texture(sampler2D(albedo, linearSampler), uv);
}
"""

METAL_ENTRY_RESOURCES = """
#include <metal_stdlib>
using namespace metal;
fragment float4 fragment_main(float2 uv [[user(locn0)]],
                              texture2d<float> albedo [[texture(0)]],
                              sampler pointSampler [[sampler(0)]],
                              sampler linearSampler [[sampler(1)]]) {
  return albedo.sample(pointSampler, uv) + albedo.sample(linearSampler, uv);
}
"""


def _translate(tmp_path, source, suffix, backend):
    path = tmp_path / f"shader{suffix}"
    path.write_text(source)
    return crosstl.translate(str(path), backend=backend, format_output=False)


def _run(command, **kwargs):
    result = subprocess.run(command, capture_output=True, text=True, **kwargs)
    assert result.returncode == 0, result.stdout + result.stderr


def _validate_glsl(code, stage="frag"):
    if shutil.which("glslangValidator") is None:
        return
    _run(["glslangValidator", "--stdin", "-S", stage], input=code)


def _validate_spirv(tmp_path, code):
    if not (shutil.which("spirv-as") and shutil.which("spirv-val")):
        return
    asm, binary = tmp_path / "out.spvasm", tmp_path / "out.spv"
    asm.write_text(code)
    _run(["spirv-as", "--target-env", "vulkan1.0", str(asm), "-o", str(binary)])
    _run(["spirv-val", "--target-env", "vulkan1.0", str(binary)])


def _validate_metal(tmp_path, code):
    if shutil.which("xcrun") is None:
        return
    found = subprocess.run(
        ["xcrun", "-sdk", "macosx", "--find", "metal"], capture_output=True
    )
    if found.returncode != 0:
        return
    source = tmp_path / "out.metal"
    source.write_text(code)
    _run(
        ["xcrun", "-sdk", "macosx", "metal", "-c", str(source)]
        + ["-o", str(tmp_path / "out.air")]
    )


def _pairs(glsl):
    header = artifact_contract_headers(glsl)[COMBINED_SAMPLER_PREFIX]
    return json.loads(header[len(COMBINED_SAMPLER_PREFIX) :])["pairs"]


def _spirv_names(spirv):
    names = dict(re.findall(r'OpName %(\w+) "(\w+)"', spirv))
    return lambda value: names.get(value, value)


def _sampled_image_samplers(spirv):
    """The variable each ``OpSampledImage`` loaded its sampler from, by name."""
    name = _spirv_names(spirv)
    loads = dict(re.findall(r"%(\w+) = OpLoad %\w+ %(\w+)", spirv))
    return [
        name(loads.get(sampler))
        for sampler in re.findall(r"= OpSampledImage %\w+ %\w+ %(\w+)", spirv)
    ]


def test_spirv_samples_through_each_explicit_sampler(tmp_path):
    spirv = _translate(tmp_path, TWO_SAMPLERS_HLSL, ".hlsl", "vulkan")

    assert _sampled_image_samplers(spirv) == ["pointSampler", "linearSampler"]
    assert spirv.count("OpImageSampleImplicitLod") == 2
    _validate_spirv(tmp_path, spirv)


def test_spirv_depth_compares_through_the_comparison_sampler(tmp_path):
    spirv = _translate(tmp_path, COMPARE_AND_SAMPLE_HLSL, ".hlsl", "vulkan")

    assert _sampled_image_samplers(spirv) == [
        "lessCompare",
        "lessCompare",
        "pointSampler",
    ]
    assert "OpImageSampleDrefImplicitLod" in spirv
    assert "OpImageSampleDrefExplicitLod" in spirv
    _validate_spirv(tmp_path, spirv)


def test_spirv_texel_fetch_reads_the_image_without_a_sampler(tmp_path):
    spirv = _translate(tmp_path, HELPERS_HLSL, ".hlsl", "vulkan")

    fetch = next(line for line in spirv.splitlines() if "OpImageFetch" in line)
    assert "OpSampledImage" not in fetch
    _validate_spirv(tmp_path, spirv)


def test_opengl_gives_each_sampler_of_a_texture_its_own_unit(tmp_path):
    glsl = _translate(tmp_path, TWO_SAMPLERS_HLSL, ".hlsl", "opengl")

    assert "layout(binding = 0) uniform sampler2D albedo;" in glsl
    assert "uniform sampler2D albedo_linearSampler;" in glsl
    assert "texture(albedo, uv) + texture(albedo_linearSampler, uv)" in glsl
    assert _pairs(glsl) == [
        {
            "sampler": "pointSampler",
            "shadow": False,
            "texture": "albedo",
            "uniform": "albedo",
        },
        {
            "sampler": "linearSampler",
            "shadow": False,
            "texture": "albedo",
            "uniform": "albedo_linearSampler",
        },
    ]
    _validate_glsl(glsl)


def test_opengl_splits_compare_and_plain_samples_by_sampler_type(tmp_path):
    glsl = _translate(tmp_path, COMPARE_AND_SAMPLE_HLSL, ".hlsl", "opengl")

    assert "layout(binding = 0) uniform sampler2D shadowMap;" in glsl
    assert "uniform sampler2DShadow shadowMap_lessCompare;" in glsl
    assert "texture(shadowMap_lessCompare, vec3(uv, 0.5))" in glsl
    assert "textureLod(shadowMap_lessCompare, vec3(uv, 0.25), 0.0)" in glsl
    assert "texture(shadowMap, uv).r" in glsl
    _validate_glsl(glsl)


def test_opengl_pairs_reach_helpers_through_their_call_sites(tmp_path):
    glsl = _translate(tmp_path, HELPERS_HLSL, ".hlsl", "opengl")

    assert "vec4 fetch(sampler2D t, vec2 uv)" in glsl
    assert "fetch(albedo, uv)" in glsl
    assert "nested(albedo_linearSampler, uv)" in glsl
    assert re.search(
        r"vec4 viaGlobal\(vec2 uv\) \{\s*return texture\(albedo, uv\);", glsl
    )
    # The depth texture is texel-fetched and depth-compared: a plain and a shadow uniform.
    assert "layout(binding = 1) uniform sampler2D depthTex;" in glsl
    assert "uniform sampler2DShadow depthTex_compareSampler;" in glsl
    assert "texelFetch(depthTex, ivec2(0, 0), 0)" in glsl
    assert {pair["uniform"] for pair in _pairs(glsl)} == {
        "albedo",
        "albedo_linearSampler",
        "depthTex",
        "depthTex_compareSampler",
    }
    _validate_glsl(glsl)


def test_webgl_splits_pairs_without_a_leading_header(tmp_path):
    glsl = _translate(tmp_path, COMPARE_AND_SAMPLE_HLSL, ".hlsl", "webgl")

    assert glsl.startswith("#version 300 es")
    assert COMBINED_SAMPLER_PREFIX.strip() not in glsl
    assert "uniform sampler2DShadow shadowMap_lessCompare;" in glsl
    _validate_glsl(glsl)


@pytest.mark.parametrize(
    "source, reason",
    [
        (
            """
            Texture2D albedo : register(t0);
            SamplerState pointSampler : register(s0);
            SamplerState linearSampler : register(s1);
            float4 both(Texture2D t, float2 uv) {
                return t.Sample(pointSampler, uv) + t.Sample(linearSampler, uv);
            }
            float4 main(float2 uv : TEXCOORD0) : SV_Target { return both(albedo, uv); }
            """,
            "texture parameter 't' of 'both' is sampled through 2 sampler states",
        ),
        (
            """
            Texture2D albedo : register(t0);
            SamplerState pointSampler : register(s0);
            SamplerState linearSampler : register(s1);
            float4 viaGlobal(SamplerState s, float2 uv) { return albedo.Sample(s, uv); }
            float4 main(float2 uv : TEXCOORD0) : SV_Target {
                return viaGlobal(pointSampler, uv) + viaGlobal(linearSampler, uv);
            }
            """,
            "sampler parameter 's' of 'viaGlobal', which its callers bind to several samplers",
        ),
        (
            """
            Texture2D albedo : register(t0);
            SamplerState samplers[2] : register(s0);
            int index;
            float4 viaParameter(SamplerState s, float2 uv) { return albedo.Sample(s, uv); }
            float4 main(float2 uv : TEXCOORD0) : SV_Target {
                return viaParameter(samplers[index], uv);
            }
            """,
            "texture 'albedo' is sampled through a sampler selected at run time",
        ),
        (
            """
            Texture2D albedo[2] : register(t0);
            SamplerState samplers[2] : register(s0);
            int layer;
            int index;
            float4 main(float2 uv : TEXCOORD0) : SV_Target {
                return albedo[layer].Sample(samplers[index], uv);
            }
            """,
            "texture 'albedo' is sampled through a sampler selected at run time",
        ),
    ],
)
def test_opengl_refuses_pairs_it_cannot_bind_statically(tmp_path, source, reason):
    with pytest.raises(OpenGLLoweringRefused, match=re.escape(reason)):
        _translate(tmp_path, source, ".hlsl", "opengl")


RUNTIME_SELECTED_HLSL = """
Texture2D albedo : register(t0);
Texture2D detail : register(t1);
Texture2D shadowMap : register(t2);
SamplerState states[3] : register(s0);
SamplerComparisonState compares[2] : register(s3);
cbuffer Params : register(b0) { int which; int cmpIndex; };
float4 main(float2 uv : TEXCOORD0) : SV_Target {
    float4 a = albedo.Sample(states[which], uv);
    float4 b = albedo.Sample(states[1], uv) + detail.Sample(states[0], uv);
    float lit = shadowMap.SampleCmp(compares[cmpIndex], uv, 0.5);
    float raw = shadowMap.Load(int3(0, 0, 0)).x;
    return a + b + float4(lit, raw, 0.0, 0.0);
}
"""


def test_opengl_indexes_one_combined_uniform_per_runtime_selected_sampler(tmp_path):
    glsl = _translate(tmp_path, RUNTIME_SELECTED_HLSL, ".hlsl", "opengl")

    # Element i of each array is the texture combined with sampler i; arrays take free units after the source's.
    assert "layout(binding = 0) uniform sampler2D albedo;" in glsl
    assert "layout(binding = 2) uniform sampler2D shadowMap;" in glsl
    assert "layout(binding = 3) uniform sampler2D albedo_states[3];" in glsl
    assert "layout(binding = 6) uniform sampler2DShadow shadowMap_compares[2];" in glsl
    assert "texture(albedo_states[which], uv)" in glsl
    assert "texture(albedo, uv) + texture(detail, uv)" in glsl
    assert "texture(shadowMap_compares[cmpIndex], vec3(uv, 0.5))" in glsl
    assert "texelFetch(shadowMap, ivec2(0, 0), 0)" in glsl
    assert {
        (pair["uniform"], pair["sampler"], pair.get("samplerArray", False))
        for pair in _pairs(glsl)
    } == {
        ("albedo", "states", False),
        ("albedo_states", "states", True),
        ("shadowMap", None, False),
        ("shadowMap_compares", "compares", True),
    }
    _validate_glsl(glsl)


def test_webgl_selects_runtime_selected_samplers_with_constant_indices(tmp_path):
    glsl = _translate(tmp_path, RUNTIME_SELECTED_HLSL, ".hlsl", "webgl")

    # GLSL ES 3.00 indexes sampler arrays only with constant-index-expressions (section 4.1.7.1).
    assert "uniform sampler2D albedo_states[3];" in glsl
    assert "switch (which)" in glsl
    assert "texture(albedo_states[2], uv)" in glsl
    assert "switch (cmpIndex)" in glsl
    assert "texture(shadowMap_compares[1], vec3(uv, 0.5))" in glsl
    assert "states[" not in glsl.replace("albedo_states[", "")
    _validate_glsl(glsl)


def test_webgl_elementwise_texture_and_sampler_arrays_drop_the_sampler(tmp_path):
    source = """
    shader ElementwisePairs {
        sampler2D textures[4];
        sampler samplers[4];
        uniform int layer;
        fragment {
            vec4 main(vec2 uv @ TEXCOORD0) @ gl_FragColor {
                return texture(textures[layer], samplers[layer], uv);
            }
        }
    }
    """
    glsl = _translate(tmp_path, source, ".cgl", "webgl")

    assert "texture(textures[3], uv)" in glsl
    assert "samplers" not in glsl
    _validate_glsl(glsl)


def test_opengl_keeps_elementwise_texture_and_sampler_arrays(tmp_path):
    source = """
    shader ElementwisePairs {
        sampler2D textures[4];
        sampler samplers[4];
        uniform int layer;
        fragment {
            vec4 main(vec2 uv @ TEXCOORD0) @ gl_FragColor {
                return texture(textures[layer], samplers[layer], uv);
            }
        }
    }
    """
    glsl = _translate(tmp_path, source, ".cgl", "opengl")

    assert "uniform sampler2D textures[4];" in glsl
    assert "texture(textures[layer], uv)" in glsl
    assert COMBINED_SAMPLER_PREFIX.strip() not in glsl


def test_vulkan_glsl_separate_texture_lowers_to_combined_opengl(tmp_path):
    glsl = _translate(tmp_path, VULKAN_ONE_SAMPLER_GLSL, ".frag", "opengl")

    assert "layout(binding = 0) uniform sampler2D albedo;" in glsl
    assert "texture2D" not in glsl
    assert "linearSampler" not in glsl
    assert "texture(albedo, uv)" in glsl
    _validate_glsl(glsl)


def test_vulkan_glsl_shadow_constructor_imports_as_a_depth_compare(tmp_path):
    crossgl = _translate(tmp_path, VULKAN_SEPARATE_GLSL, ".frag", "cgl")
    assert "textureCompare(depthTex, compareSampler, input.uv, 0.5)" in crossgl
    assert "texture(depthTex, pointSampler, input.uv)" in crossgl

    glsl = _translate(tmp_path, VULKAN_SEPARATE_GLSL, ".frag", "opengl")
    assert "uniform sampler2DShadow depthTex_compareSampler;" in glsl
    assert "texture(depthTex_compareSampler, vec3(uv, 0.5))" in glsl
    _validate_glsl(glsl)

    spirv = _translate(tmp_path, VULKAN_SEPARATE_GLSL, ".frag", "vulkan")
    assert _sampled_image_samplers(spirv) == ["compareSampler", "pointSampler"]
    assert "OpImageSampleDrefImplicitLod" in spirv
    _validate_spirv(tmp_path, spirv)


@pytest.mark.parametrize("backend", ["metal", "directx"])
def test_separate_textures_emit_native_textures(tmp_path, backend):
    code = _translate(tmp_path, VULKAN_ONE_SAMPLER_GLSL, ".frag", backend)

    if backend == "metal":
        assert "texture2d<float> albedo [[texture(0)]]" in code
        assert "albedo.sample(linearSampler, " in code
        _validate_metal(tmp_path, code)
    else:
        assert "Texture2D albedo : register(t0);" in code
        assert "albedo.Sample(linearSampler, " in code


def test_metal_entry_textures_become_spirv_descriptors(tmp_path):
    spirv = _translate(tmp_path, METAL_ENTRY_RESOURCES, ".metal", "vulkan")

    name = _spirv_names(spirv)
    storage = {
        name(variable): storage_class
        for variable, storage_class in re.findall(
            r"%(\w+) = OpVariable %\w+ (\w+)", spirv
        )
    }
    for resource in ("albedo", "pointSampler", "linearSampler"):
        assert storage[resource] == "UniformConstant"
    # MSL's texture and sampler namespaces both start at 0; one Vulkan set takes them in order.
    bindings = {
        name(variable): int(binding)
        for variable, binding in re.findall(r"OpDecorate %(\w+) Binding (\d+)", spirv)
    }
    assert bindings == {"albedo": 0, "pointSampler": 1, "linearSampler": 2}
    assert _sampled_image_samplers(spirv) == ["pointSampler", "linearSampler"]
    _validate_spirv(tmp_path, spirv)


def test_metal_entry_textures_split_into_opengl_pairs(tmp_path):
    glsl = _translate(tmp_path, METAL_ENTRY_RESOURCES, ".metal", "opengl")

    assert "uniform sampler2D albedo;" in glsl
    assert "uniform sampler2D albedo_linearSampler;" in glsl
    assert "texture(albedo_linearSampler, uv)" in glsl
    _validate_glsl(glsl)


DEPTH_READS_CGL = """
shader DepthReads {
    struct FragmentInput {
        vec2 uv;
    }
    struct FragmentOutput {
        vec4 color;
    }
    sampler2D shadowMap;
    sampler lessCompare;
    sampler pointSampler;
    fragment {
        FragmentOutput main(FragmentInput input) {
            FragmentOutput output;
            float lit = textureCompare(shadowMap, lessCompare, input.uv, 0.5);
            vec4 raw = texture(shadowMap, pointSampler, input.uv);
            vec4 level0 = textureLod(shadowMap, pointSampler, input.uv, 0.0);
            vec4 quad = textureGather(shadowMap, pointSampler, input.uv);
            vec4 texel = texelFetch(shadowMap, ivec2(0, 0), 0);
            output.color = vec4(lit) + raw + level0 + quad + texel;
            return output;
        }
    }
}
"""


def test_metal_declares_compared_textures_as_depth_textures(tmp_path):
    metal = _translate(tmp_path, DEPTH_READS_CGL, ".cgl", "metal")

    # MSL's sample_compare exists only on depth textures; plain reads of one return the depth as a scalar,
    # which HLSL and GLSL read as (d, 0, 0, 1).
    assert "depth2d<float> shadowMap [[texture(0)]]" in metal
    assert "shadowMap.sample_compare(lessCompare, input.uv, 0.5)" in metal
    assert "float4(shadowMap.sample(pointSampler, input.uv), 0.0, 0.0, 1.0)" in metal
    assert (
        "float4(shadowMap.sample(pointSampler, input.uv, level(0.0)), 0.0, 0.0, 1.0)"
        in metal
    )
    assert "shadowMap.gather(pointSampler, input.uv)" in metal
    assert "float4(shadowMap.read(" in metal
    _validate_metal(tmp_path, metal)


def test_metal_vulkan_separate_shadow_texture_is_a_depth_texture(tmp_path):
    metal = _translate(tmp_path, VULKAN_SEPARATE_GLSL, ".frag", "metal")

    assert "depth2d<float> depthTex [[texture(0)]]" in metal
    assert "depthTex.sample_compare(compareSampler, " in metal
    assert "float4(depthTex.sample(pointSampler, " in metal
    _validate_metal(tmp_path, metal)


def test_metal_refuses_a_helper_shared_by_depth_and_colour_textures(tmp_path):
    source = """
    Texture2D shadowMap : register(t0);
    Texture2D colorMap : register(t1);
    SamplerComparisonState lessCompare : register(s0);
    SamplerState pointSampler : register(s1);
    float lit(Texture2D t, SamplerComparisonState c, float2 uv) { return t.SampleCmp(c, uv, 0.5); }
    float4 plain(Texture2D t, SamplerState s, float2 uv) { return t.Sample(s, uv); }
    float4 main(float2 uv : TEXCOORD0) : SV_Target {
        return lit(shadowMap, lessCompare, uv) + plain(shadowMap, pointSampler, uv)
            + plain(colorMap, pointSampler, uv);
    }
    """
    with pytest.raises(
        MetalLoweringRefused,
        match=re.escape(
            "texture parameter 't' of 'plain' receives both depth and colour textures"
        ),
    ):
        _translate(tmp_path, source, ".hlsl", "metal")

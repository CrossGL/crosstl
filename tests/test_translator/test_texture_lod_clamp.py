"""Minimum-LOD clamped sampling (CrossGL ``texture*Clamp``) is carried between HLSL, MSL, GLSL and SPIR-V.

Every importer maps its spelling of the clamp to the CrossGL operations (HLSL ``Sample(S, L, Offset, Clamp)``,
MSL ``min_lod_clamp``, GLSL ``texture*ClampARB``, SPIR-V ``MinLod``), every graphics emitter attaches it to the
sample it lowers to, and the targets that cannot express it refuse it with a typed diagnostic.
"""

from __future__ import annotations

import platform
import shutil
import subprocess

import pytest

import crosstl
from crosstl.backend.SPIRV.ray_tracing_import import SpirvInstructionUnsupportedError
from crosstl.translator.codegen.texture_lod_clamp import (
    TEXTURE_LOD_CLAMP_FORMS,
    TextureLodClampUnsupportedError,
    clamped_spirv_instruction,
    split_call_text,
)

CROSSGL_FRAGMENT = """
shader LodClamp {
    struct FragmentInput {
        vec2 uv;
    }

    struct FragmentOutput {
        vec4 color;
    }

    fragment {
        uniform sampler2D colorMap;
        uniform samplerCube skyMap;
        uniform sampler2DArray layers;
        uniform sampler2DShadow shadowMap;
        uniform float firstResidentMip;

        FragmentOutput main(FragmentInput input) {
            FragmentOutput output;
            float c = firstResidentMip;
            vec4 a = textureClamp(colorMap, input.uv, c);
            a += textureClamp(colorMap, input.uv, c, 0.5);
            a += textureOffsetClamp(colorMap, input.uv, ivec2(1, -1), c);
            a += textureOffsetClamp(colorMap, input.uv, ivec2(1, 1), c, 0.25);
            a += textureGradClamp(colorMap, input.uv, vec2(0.1), vec2(0.2), c);
            a += textureGradOffsetClamp(colorMap, input.uv, vec2(0.1), vec2(0.2), ivec2(1, 0), c);
            a += textureClamp(skyMap, vec3(input.uv, 1.0), c);
            a += textureGradClamp(skyMap, vec3(input.uv, 1.0), vec3(0.1), vec3(0.2), c);
            a += textureOffsetClamp(layers, vec3(input.uv, 1.0), ivec2(1, 1), c);
            float d = textureCompareClamp(shadowMap, input.uv, 0.5, c);
            d += textureCompareOffsetClamp(shadowMap, input.uv, 0.5, ivec2(1, 1), c);
            d += textureCompareGradClamp(shadowMap, input.uv, 0.5, vec2(0.1), vec2(0.2), c);
            d += textureCompareGradOffsetClamp(shadowMap, input.uv, 0.5, vec2(0.1), vec2(0.2), ivec2(1, 1), c);
            output.color = a + vec4(d);
            return output;
        }
    }
}
"""

MSL_FRAGMENT = """
#include <metal_stdlib>
using namespace metal;

fragment float4 shade(float4 position [[position]],
                      texture2d<float> colorMap [[texture(0)]],
                      texturecube<float> skyMap [[texture(1)]],
                      depth2d<float> shadowMap [[texture(2)]],
                      texture2d_array<float> layers [[texture(3)]],
                      sampler colorSampler [[sampler(0)]],
                      sampler shadowSampler [[sampler(1)]],
                      constant float& c [[buffer(0)]]) {
    float2 uv = position.xy;
    float4 a = colorMap.sample(colorSampler, uv, min_lod_clamp(c));
    a += colorMap.sample(colorSampler, uv, min_lod_clamp(c), int2(1, 1));
    a += colorMap.sample(colorSampler, uv, bias(0.5), min_lod_clamp(c));
    a += colorMap.sample(colorSampler, uv, bias(0.5), min_lod_clamp(c), int2(1, -1));
    a += colorMap.sample(colorSampler, uv, gradient2d(float2(0.1), float2(0.2)), min_lod_clamp(c));
    a += colorMap.sample(colorSampler, uv, gradient2d(float2(0.1), float2(0.2)), min_lod_clamp(c), int2(1, 0));
    a += skyMap.sample(colorSampler, float3(uv, 1.0), bias(0.5), min_lod_clamp(c));
    a += skyMap.sample(colorSampler, float3(uv, 1.0), gradientcube(float3(0.1), float3(0.2)), min_lod_clamp(c));
    a += layers.sample(colorSampler, uv, 1, min_lod_clamp(c), int2(1, 1));
    float d = shadowMap.sample_compare(shadowSampler, uv, 0.5, min_lod_clamp(c));
    d += shadowMap.sample_compare(shadowSampler, uv, 0.5, min_lod_clamp(c), int2(1, 1));
    d += shadowMap.sample_compare(shadowSampler, uv, 0.5, gradient2d(float2(0.1), float2(0.2)), min_lod_clamp(c));
    d += shadowMap.sample_compare(shadowSampler, uv, 0.5, gradient2d(float2(0.1), float2(0.2)), min_lod_clamp(c), int2(1, 1));
    return a + d;
}
"""

HLSL_FRAGMENT = """
Texture2D<float4> colorMap : register(t0);
TextureCube<float4> skyMap : register(t1);
Texture2D<float> shadowMap : register(t2);
Texture3D<float4> volume : register(t3);
SamplerState colorSampler : register(s0);
SamplerComparisonState shadowSampler : register(s1);
cbuffer Params : register(b0) { float c; };

float4 main(float4 position : SV_Position) : SV_Target {
    float2 uv = position.xy;
    float4 a = colorMap.Sample(colorSampler, uv, int2(1, 1), c);
    a += colorMap.SampleBias(colorSampler, uv, 0.5, int2(1, -1), c);
    a += colorMap.SampleGrad(colorSampler, uv, float2(0.1, 0.1), float2(0.2, 0.2), int2(1, 0), c);
    a += skyMap.Sample(colorSampler, float3(uv, 1.0), c);
    a += skyMap.SampleBias(colorSampler, float3(uv, 1.0), 0.5, c);
    a += skyMap.SampleGrad(colorSampler, float3(uv, 1.0), float3(0.1, 0.1, 0.1), float3(0.2, 0.2, 0.2), c);
    a += volume.Sample(colorSampler, float3(uv, 0.5), int3(1, 0, -1), c);
    float d = shadowMap.SampleCmp(shadowSampler, uv, 0.5, int2(1, 1), c);
    d += shadowMap.SampleCmpGrad(shadowSampler, uv, 0.5, float2(0.1, 0.1), float2(0.2, 0.2), int2(1, 1), c);
    return a + d;
}
"""

GLSL_FRAGMENT = """
#version 450
#extension GL_ARB_sparse_texture_clamp : require
layout(binding = 0) uniform sampler2D colorMap;
layout(binding = 1) uniform samplerCube skyMap;
layout(binding = 2) uniform sampler2DShadow shadowMap;
layout(binding = 3) uniform samplerCubeArrayShadow shadowCubes;
layout(binding = 4) uniform Params { float c; };
layout(location = 0) in vec2 uv;
layout(location = 0) out vec4 color;
void main() {
    vec4 a = textureClampARB(colorMap, uv, c);
    a += textureClampARB(colorMap, uv, c, 0.5);
    a += textureOffsetClampARB(colorMap, uv, ivec2(1, 1), c, 0.5);
    a += textureGradClampARB(colorMap, uv, vec2(0.1), vec2(0.2), c);
    a += textureGradOffsetClampARB(colorMap, uv, vec2(0.1), vec2(0.2), ivec2(1, -1), c);
    a += textureClampARB(skyMap, vec3(uv, 1.0), c);
    float d = textureClampARB(shadowMap, vec3(uv, 0.5), c);
    d += textureOffsetClampARB(shadowMap, vec3(uv, 0.5), ivec2(1, 1), c);
    d += textureGradClampARB(shadowMap, vec3(uv, 0.5), vec2(0.1), vec2(0.2), c);
    d += textureClampARB(shadowCubes, vec4(uv, 1.0, 2.0), 0.5, c);
    color = a + d;
}
"""

SPIRV_FRAGMENT = """
; SPIR-V
; Version: 1.0
OpCapability Shader
OpCapability MinLod
%glsl = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %uv %out_color
OpExecutionMode %main OriginUpperLeft
OpName %main "main"
OpName %color_tex "colorTex"
OpName %uv "uv"
OpName %out_color "outColor"
OpName %clamp_value "clampValue"
OpDecorate %color_tex DescriptorSet 0
OpDecorate %color_tex Binding 0
OpDecorate %uv Location 0
OpDecorate %out_color Location 0
%void = OpTypeVoid
%fn = OpTypeFunction %void
%float = OpTypeFloat 32
%int = OpTypeInt 32 1
%v2float = OpTypeVector %float 2
%v4float = OpTypeVector %float 4
%v2int = OpTypeVector %int 2
%image = OpTypeImage %float 2D 0 0 0 1 Unknown
%sampled = OpTypeSampledImage %image
%ptr_sampled = OpTypePointer UniformConstant %sampled
%ptr_input_v2float = OpTypePointer Input %v2float
%ptr_output_v4float = OpTypePointer Output %v4float
%ptr_private_float = OpTypePointer Private %float
%color_tex = OpVariable %ptr_sampled UniformConstant
%uv = OpVariable %ptr_input_v2float Input
%out_color = OpVariable %ptr_output_v4float Output
%float_half = OpConstant %float 0.5
%float_quarter = OpConstant %float 0.25
%int_one = OpConstant %int 1
%offset = OpConstantComposite %v2int %int_one %int_one
%clamp_value = OpVariable %ptr_private_float Private %float_quarter
%main = OpFunction %void None %fn
%label = OpLabel
%tex = OpLoad %sampled %color_tex
%coord = OpLoad %v2float %uv
%min_lod = OpLoad %float %clamp_value
%dx = OpDPdx %v2float %coord
%dy = OpDPdy %v2float %coord
%s0 = OpImageSampleImplicitLod %v4float %tex %coord MinLod %min_lod
%s1 = OpImageSampleImplicitLod %v4float %tex %coord Bias|ConstOffset|MinLod %float_half %offset %min_lod
%s2 = OpImageSampleExplicitLod %v4float %tex %coord Grad|MinLod %dx %dy %min_lod
%s3 = OpImageSampleExplicitLod %v4float %tex %coord Grad|ConstOffset|MinLod %dx %dy %offset %min_lod
%sum0 = OpFAdd %v4float %s0 %s1
%sum1 = OpFAdd %v4float %sum0 %s2
%sum2 = OpFAdd %v4float %sum1 %s3
OpStore %out_color %sum2
OpReturn
OpFunctionEnd
"""


def _translate(tmp_path, source, suffix, backend):
    path = tmp_path / f"shader{suffix}"
    path.write_text(source)
    return crosstl.translate(str(path), backend=backend, format_output=False)


def _run(command):
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


def _validate(tmp_path, backend, code):
    """Compile ``code`` with the target's native checker when it is installed."""
    if backend == "vulkan" and shutil.which("spirv-as") and shutil.which("spirv-val"):
        asm, binary = tmp_path / "out.spvasm", tmp_path / "out.spv"
        asm.write_text(code)
        _run(["spirv-as", "--target-env", "vulkan1.1", str(asm), "-o", str(binary)])
        _run(["spirv-val", "--target-env", "vulkan1.1", str(binary)])
    elif backend == "opengl" and shutil.which("glslangValidator"):
        path = tmp_path / "out.frag"
        path.write_text(code)
        _run(["glslangValidator", "-S", "frag", str(path)])
    elif (
        backend == "metal"
        and platform.system() == "Darwin"
        and shutil.which("xcrun")
        and subprocess.run(
            ["xcrun", "-sdk", "macosx", "-f", "metal"], capture_output=True
        ).returncode
        == 0
    ):
        path = tmp_path / "out.metal"
        path.write_text(code)
        _run(
            [
                "xcrun",
                "-sdk",
                "macosx",
                "metal",
                "-c",
                str(path),
                "-o",
                str(tmp_path / "out.air"),
            ]
        )
    elif backend == "directx" and shutil.which("dxc"):
        path = tmp_path / "out.hlsl"
        path.write_text(code)
        entry = "PSMain" if "PSMain(" in code else "main"
        # SampleCmpGrad needs Shader Model 6.8.
        _run(
            [
                "dxc",
                "-T",
                "ps_6_8",
                "-E",
                entry,
                str(path),
                "-Fo",
                str(tmp_path / "out.dxil"),
            ]
        )


def test_crossgl_clamp_operations_lower_to_hlsl_clamp_overloads(tmp_path):
    hlsl = _translate(tmp_path, CROSSGL_FRAGMENT, ".cgl", "directx")

    # A non-cube sample without an offset takes a zero offset, since HLSL's Clamp follows the offset.
    assert "colorMap.Sample(colorMapSampler, input.uv, int2(0, 0), c)" in hlsl
    assert "colorMap.SampleBias(colorMapSampler, input.uv, 0.5, int2(0, 0), c)" in hlsl
    assert "colorMap.Sample(colorMapSampler, input.uv, int2(1, -1), c)" in hlsl
    assert "colorMap.SampleBias(colorMapSampler, input.uv, 0.25, int2(1, 1), c)" in hlsl
    assert (
        "colorMap.SampleGrad(colorMapSampler, input.uv, float2(0.1, 0.1), float2(0.2, 0.2), "
        "int2(0, 0), c)"
    ) in hlsl
    assert "skyMap.Sample(skyMapSampler, float3(input.uv, 1.0), c)" in hlsl
    assert "layers.Sample(layersSampler, float3(input.uv, 1.0), int2(1, 1), c)" in hlsl
    assert "shadowMap.SampleCmp(shadowMapSampler, input.uv, 0.5, int2(0, 0), c)" in hlsl
    assert (
        "shadowMap.SampleCmpGrad(shadowMapSampler, input.uv, 0.5, float2(0.1, 0.1), "
        "float2(0.2, 0.2), int2(1, 1), c)"
    ) in hlsl
    assert "Clamp(" not in hlsl
    _validate(tmp_path, "directx", hlsl)


def test_crossgl_clamp_operations_lower_to_msl_min_lod_clamp(tmp_path):
    metal = _translate(tmp_path, CROSSGL_FRAGMENT, ".cgl", "metal")

    assert ", input.uv, min_lod_clamp(c));" in metal
    assert ", input.uv, bias(0.5), min_lod_clamp(c));" in metal
    # The clamp precedes the offset, which is always the last sample argument in MSL.
    assert ", input.uv, min_lod_clamp(c), int2(1, -1));" in metal
    assert ", input.uv, bias(0.25), min_lod_clamp(c), int2(1, 1));" in metal
    assert (
        "gradient2d(float2(0.1), float2(0.2)), min_lod_clamp(c), int2(1, 0));" in metal
    )
    assert "gradientcube(float3(0.1), float3(0.2)), min_lod_clamp(c));" in metal
    assert "shadowMap.sample_compare(" in metal
    assert ", input.uv, 0.5, min_lod_clamp(c), int2(1, 1));" in metal
    assert "Clamp(" not in metal
    _validate(tmp_path, "metal", metal)


def test_crossgl_clamp_operations_lower_to_arb_sparse_texture_clamp(tmp_path):
    glsl = _translate(tmp_path, CROSSGL_FRAGMENT, ".cgl", "opengl")

    assert "#extension GL_ARB_sparse_texture_clamp : require" in glsl
    assert "textureClampARB(colorMap, uv, c)" in glsl
    # The clamp precedes the bias, as in GL_ARB_sparse_texture_clamp.
    assert "textureClampARB(colorMap, uv, c, 0.5)" in glsl
    assert "textureOffsetClampARB(colorMap, uv, ivec2(1, 1), c, 0.25)" in glsl
    assert (
        "textureGradOffsetClampARB(colorMap, uv, vec2(0.1), vec2(0.2), ivec2(1, 0), c)"
        in glsl
    )
    assert "textureClampARB(shadowMap, vec3(uv, 0.5), c)" in glsl
    assert (
        "textureGradOffsetClampARB(shadowMap, vec3(uv, 0.5), vec2(0.1), vec2(0.2), ivec2(1, 1), c)"
        in glsl
    )
    _validate(tmp_path, "opengl", glsl)


def test_crossgl_clamp_operations_lower_to_spirv_min_lod(tmp_path):
    spirv = _translate(tmp_path, CROSSGL_FRAGMENT, ".cgl", "vulkan")

    assert "OpCapability MinLod" in spirv
    samples = [line for line in spirv.splitlines() if "OpImageSample" in line]
    assert len(samples) == 13
    assert all("MinLod" in line for line in samples)
    assert any("Bias|ConstOffset|MinLod" in line for line in samples)
    assert any("Grad|ConstOffset|MinLod" in line for line in samples)
    assert any(
        "OpImageSampleDrefExplicitLod" in line and "Grad|MinLod" in line
        for line in samples
    )
    _validate(tmp_path, "vulkan", spirv)


@pytest.mark.parametrize(
    ("backend", "reason"),
    [
        ("webgl", "GLSL ES 3.00 has no clamped sampling functions"),
        ("wgsl", "lodMinClamp"),
        ("cuda", "minMipmapLevelClamp"),
        ("hip", "minMipmapLevelClamp"),
        ("slang", "no minimum-LOD operand"),
        ("rust", "no minimum-LOD operand"),
        ("mojo", "no minimum-LOD operand"),
    ],
)
def test_targets_without_a_clamp_operand_refuse_it(tmp_path, backend, reason):
    with pytest.raises(TextureLodClampUnsupportedError) as error:
        _translate(tmp_path, CROSSGL_FRAGMENT, ".cgl", backend)

    assert error.value.operation == "textureClamp"
    assert reason in str(error.value)
    assert error.value.code.endswith(f"{backend}-texture-lod-clamp-unsupported")


def test_crossgl_clamp_with_wrong_operand_count_is_rejected(tmp_path):
    source = CROSSGL_FRAGMENT.replace(
        "textureGradClamp(colorMap, input.uv, vec2(0.1), vec2(0.2), c)",
        "textureGradClamp(colorMap, input.uv, vec2(0.1), vec2(0.2), c, 0.5)",
    )
    for backend in ("directx", "metal", "opengl"):
        with pytest.raises(ValueError, match="'textureGradClamp' accepts 5 argument"):
            _translate(tmp_path, source, ".cgl", backend)


def test_msl_min_lod_clamp_options_import_to_crossgl_clamp_operations(tmp_path):
    crossgl = _translate(tmp_path, MSL_FRAGMENT, ".metal", "cgl")

    assert "textureClamp(colorMap, colorSampler, uv, c)" in crossgl
    assert "textureOffsetClamp(colorMap, colorSampler, uv, ivec2(1, 1), c)" in crossgl
    assert "textureClamp(colorMap, colorSampler, uv, c, 0.5)" in crossgl
    assert (
        "textureOffsetClamp(colorMap, colorSampler, uv, ivec2(1, (-1)), c, 0.5)"
        in crossgl
    )
    assert (
        "textureGradClamp(colorMap, colorSampler, uv, vec2(0.1), vec2(0.2), c)"
        in crossgl
    )
    assert (
        "textureGradOffsetClamp(colorMap, colorSampler, uv, vec2(0.1), vec2(0.2), ivec2(1, 0), c)"
        in crossgl
    )
    assert "textureClamp(skyMap, colorSampler, vec3(uv, 1.0), c, 0.5)" in crossgl
    assert (
        "textureGradClamp(skyMap, colorSampler, vec3(uv, 1.0), vec3(0.1), vec3(0.2), c)"
        in crossgl
    )
    assert (
        "textureOffsetClamp(layers, colorSampler, vec3(uv, 1), ivec2(1, 1), c)"
        in crossgl
    )
    assert "textureCompareClamp(shadowMap, shadowSampler, uv, 0.5, c)" in crossgl
    assert (
        "textureCompareOffsetClamp(shadowMap, shadowSampler, uv, 0.5, ivec2(1, 1), c)"
        in crossgl
    )
    assert (
        "textureCompareGradClamp(shadowMap, shadowSampler, uv, 0.5, vec2(0.1), vec2(0.2), c)"
        in crossgl
    )
    assert "min_lod_clamp" not in crossgl


@pytest.mark.parametrize("backend", ["directx", "metal", "opengl"])
def test_msl_min_lod_clamp_translates_to_valid_targets(tmp_path, backend):
    code = _translate(tmp_path, MSL_FRAGMENT, ".metal", backend)

    clamp_marker = {
        "directx": "shade_c)",
        "metal": "min_lod_clamp(",
        "opengl": "ClampARB(",
    }
    assert code.count(clamp_marker[backend]) >= 13
    _validate(tmp_path, backend, code)


def test_hlsl_clamp_overloads_import_to_crossgl_clamp_operations(tmp_path):
    crossgl = _translate(tmp_path, HLSL_FRAGMENT, ".hlsl", "cgl")

    assert "textureOffsetClamp(colorMap, colorSampler, uv, ivec2(1, 1), c)" in crossgl
    # SampleBias(S, L, Bias, Offset, Clamp) reorders to CrossGL's (offset, clamp, bias).
    assert (
        "textureOffsetClamp(colorMap, colorSampler, uv, ivec2(1, -1), c, 0.5)"
        in crossgl
    )
    assert (
        "textureGradOffsetClamp(colorMap, colorSampler, uv, vec2(0.1, 0.1), vec2(0.2, 0.2), "
        "ivec2(1, 0), c)"
    ) in crossgl
    assert "textureClamp(skyMap, colorSampler, vec3(uv, 1.0), c)" in crossgl
    assert "textureClamp(skyMap, colorSampler, vec3(uv, 1.0), c, 0.5)" in crossgl
    assert "textureGradClamp(skyMap, colorSampler, vec3(uv, 1.0)" in crossgl
    assert (
        "textureOffsetClamp(volume, colorSampler, vec3(uv, 0.5), ivec3(1, 0, -1), c)"
        in crossgl
    )
    assert (
        "textureCompareOffsetClamp(shadowMap, shadowSampler, uv, 0.5, ivec2(1, 1), c)"
        in crossgl
    )
    assert "textureCompareGradOffsetClamp(shadowMap, shadowSampler, uv, 0.5" in crossgl


@pytest.mark.parametrize("backend", ["directx", "metal", "opengl", "vulkan"])
def test_hlsl_clamp_overloads_translate_to_valid_targets(tmp_path, backend):
    code = _translate(tmp_path, HLSL_FRAGMENT, ".hlsl", backend)

    if backend == "vulkan":
        assert code.count("MinLod %") == 9
    elif backend == "metal":
        assert code.count("min_lod_clamp(") == 9
    elif backend == "opengl":
        assert code.count("ClampARB(") == 9
    else:
        assert "colorMap.SampleBias(colorSampler, uv, 0.5, int2(1, -1), c)" in code
    _validate(tmp_path, backend, code)


def test_glsl_arb_clamp_functions_import_to_crossgl_clamp_operations(tmp_path):
    crossgl = _translate(tmp_path, GLSL_FRAGMENT, ".frag", "cgl")

    assert "textureClamp(colorMap, input.uv, c)" in crossgl
    assert "textureClamp(colorMap, input.uv, c, 0.5)" in crossgl
    assert "textureOffsetClamp(colorMap, input.uv, ivec2(1, 1), c, 0.5)" in crossgl
    assert (
        "textureGradOffsetClamp(colorMap, input.uv, vec2(0.1), vec2(0.2), ivec2(1, (-1)), c)"
        in crossgl
    )
    # A shadow sampler's packed reference becomes the compare operand.
    assert "textureCompareClamp(shadowMap, input.uv, 0.5, c)" in crossgl
    assert (
        "textureCompareOffsetClamp(shadowMap, input.uv, 0.5, ivec2(1, 1), c)" in crossgl
    )
    assert (
        "textureCompareGradClamp(shadowMap, input.uv, 0.5, vec2(0.1), vec2(0.2), c)"
        in crossgl
    )
    assert (
        "textureCompareClamp(shadowCubes, vec4(input.uv, 1.0, 2.0), 0.5, c)" in crossgl
    )
    assert "ClampARB" not in crossgl.replace("GL_ARB_sparse_texture_clamp", "")


@pytest.mark.parametrize("backend", ["directx", "metal", "opengl", "vulkan"])
def test_glsl_arb_clamp_functions_translate_to_valid_targets(tmp_path, backend):
    code = _translate(tmp_path, GLSL_FRAGMENT, ".frag", backend)

    expected = {
        "directx": ("dca", None),
        "metal": ("min_lod_clamp(", 10),
        "opengl": ("ClampARB(", 10),
        "vulkan": ("MinLod %", 10),
    }[backend]
    if expected[1] is not None:
        assert code.count(expected[0]) == expected[1]
    else:
        assert (
            "shadowCubes.SampleCmp(shadowCubesSampler, float4(input.uv, 1.0, 2.0), 0.5, c)"
            in code
        )
    if backend == "directx":
        # The ``#extension`` line of the GLSL source is not HLSL.
        assert "#extension" not in code
    _validate(tmp_path, backend, code)


def test_glsl_shadow_clamp_with_bias_is_refused(tmp_path):
    source = GLSL_FRAGMENT.replace(
        "float d = textureClampARB(shadowMap, vec3(uv, 0.5), c);",
        "float d = textureClampARB(shadowMap, vec3(uv, 0.5), c, 0.5);",
    )
    with pytest.raises(TextureLodClampUnsupportedError, match="no LOD bias"):
        _translate(tmp_path, source, ".frag", "cgl")


def test_spirv_min_lod_operand_imports_to_crossgl_clamp_operations(tmp_path):
    crossgl = _translate(tmp_path, SPIRV_FRAGMENT, ".spvasm", "cgl")

    assert "textureClamp(colorTex, uv, clampValue)" in crossgl
    # Bias follows the clamp in CrossGL.
    assert "textureOffsetClamp(colorTex, uv, int2(1, 1), clampValue, 0.5)" in crossgl
    assert "textureGradClamp(colorTex, uv, dFdx(uv), dFdy(uv), clampValue)" in crossgl
    assert (
        "textureGradOffsetClamp(colorTex, uv, dFdx(uv), dFdy(uv), int2(1, 1), clampValue)"
        in crossgl
    )
    assert "spirvTexture" not in crossgl


def test_spirv_min_lod_operand_round_trips_to_valid_spirv(tmp_path):
    code = _translate(tmp_path, SPIRV_FRAGMENT, ".spvasm", "vulkan")

    assert "OpCapability MinLod" in code
    assert code.count("MinLod %") == 4
    _validate(tmp_path, "vulkan", code)


def test_spirv_projective_min_lod_sample_is_refused(tmp_path):
    source = SPIRV_FRAGMENT.replace(
        "%s0 = OpImageSampleImplicitLod %v4float %tex %coord MinLod %min_lod",
        "%coord3 = OpCompositeConstruct %v4float %float_half %float_half %float_half %float_half\n"
        "%s0 = OpImageSampleProjImplicitLod %v4float %tex %coord3 MinLod %min_lod",
    )
    with pytest.raises(SpirvInstructionUnsupportedError, match="projective"):
        _translate(tmp_path, source, ".spvasm", "cgl")


def test_clamp_forms_name_their_unclamped_counterparts():
    assert {form.base for form in TEXTURE_LOD_CLAMP_FORMS.values()} == {
        "texture",
        "textureOffset",
        "textureGrad",
        "textureGradOffset",
        "textureCompare",
        "textureCompareOffset",
        "textureCompareGrad",
        "textureCompareGradOffset",
    }
    assert all(
        name == f"{form.base}Clamp" for name, form in TEXTURE_LOD_CLAMP_FORMS.items()
    )


def test_split_call_text_accepts_only_one_whole_call():
    assert split_call_text("t.Sample(s, f(a, b), int2(0, 0))") == (
        "t.Sample",
        ["s", "f(a, b)", "int2(0, 0)"],
    )
    assert split_call_text("arr[i].sample(sampler(a::b), uv)") == (
        "arr[i].sample",
        ["sampler(a::b)", "uv"],
    )
    assert split_call_text("a + t.Sample(s, uv)") is None
    assert split_call_text("t.Sample(s, uv).x") is None
    assert split_call_text("(t.Sample(s, uv))") is None


def test_spirv_min_lod_is_refused_on_explicit_lod_samples():
    def image_operands(*operands):
        return " ".join(
            ["|".join(o.split()[0] for o in operands)]
            + [w for o in operands for w in o.split()[1:]]
        )

    assert (
        clamped_spirv_instruction(
            "textureClamp",
            "%9 = OpImageSampleImplicitLod %4 %5 %6 Bias %7",
            8,
            image_operands,
        )
        == "%9 = OpImageSampleImplicitLod %4 %5 %6 Bias|MinLod %7 %8"
    )
    with pytest.raises(TextureLodClampUnsupportedError, match="implicit-LOD or Grad"):
        clamped_spirv_instruction(
            "textureClamp",
            "%9 = OpImageSampleExplicitLod %4 %5 %6 Lod %7",
            8,
            image_operands,
        )

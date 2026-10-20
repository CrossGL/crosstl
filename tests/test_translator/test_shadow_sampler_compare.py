"""Depth comparisons keep comparing in every target.

GLSL's sampling functions compare on shadow samplers (GLSL 4.60 section 8.9.2): ``texture(sampler2DShadow, P)`` tests
``P.z`` against the depth texels at ``P.xy``. CrossGL programs written with them must become comparisons for HLSL
(``SampleCmp*``), Metal (``sample_compare``), SPIR-V (``OpImageSampleDref*``) and Slang, not ordinary samples of the
depth texel. HLSL ``SampleCmpLevelZero`` compares at level 0, so it must not become an implicit-LOD compare, and GLSL
reaches level 0 on cube and array shadows, which have no core ``textureLod``, through zero-gradient ``textureGrad``.
"""

from __future__ import annotations

import shutil
import subprocess

import pytest

import crosstl
from crosstl.translator.semantics import ShadowSampleUnsupportedError

SHADOW_SAMPLES = """
shader ShadowSamples {
    sampler2DShadow shadowMap;
    samplerCubeShadow shadowCube;
    sampler2DArrayShadow shadowArray;

    struct FSInput {
        vec3 uvz @ TEXCOORD0;
        vec4 uvzq @ TEXCOORD1;
        vec4 dirRef @ TEXCOORD2;
        vec2 uv @ TEXCOORD3;
        float depth @ TEXCOORD4;
    };

    fragment {
        vec4 main(FSInput input) @ gl_FragColor {
            float a = texture(shadowMap, input.uvz);
            float b = texture(shadowMap, vec3(input.uv, input.depth));
            float c = textureLod(shadowMap, input.uvz, 0.0);
            float d = textureOffset(shadowMap, input.uvz, ivec2(1, 0));
            float e = texture(shadowCube, input.dirRef);
            float f = texture(shadowArray, input.dirRef);
            float g = textureProj(shadowMap, input.uvzq);
            vec4 h = textureGather(shadowMap, input.uv, input.depth);
            return vec4(a + b + c, d + e, f + g, h.x);
        }
    }
}
"""

EXPECTED = {
    "directx": [
        "float a = shadowMap.SampleCmp(shadowMapSampler, input.uvz.xy, input.uvz.z);",
        "float b = shadowMap.SampleCmp(shadowMapSampler, input.uv, input.depth);",
        "float c = shadowMap.SampleCmpLevelZero(shadowMapSampler, input.uvz.xy, input.uvz.z);",
        "float d = shadowMap.SampleCmp(shadowMapSampler, input.uvz.xy, input.uvz.z, int2(1, 0));",
        "float e = shadowCube.SampleCmp(shadowCubeSampler, input.dirRef.xyz, input.dirRef.w);",
        "float f = shadowArray.SampleCmp(shadowArraySampler, input.dirRef.xyz, input.dirRef.w);",
        "input.uvzq.z / input.uvzq.xyw.z);",
        "float4 h = shadowMap.GatherCmp(shadowMapSampler, input.uv, input.depth);",
    ],
    "metal": [
        "input.uvz.xy, input.uvz.z);",
        "input.uv, input.depth);",
        "input.uvz.xy, input.uvz.z, level(0.0));",
        "input.uvz.xy, input.uvz.z, int2(1, 0));",
        "input.dirRef.xyz, input.dirRef.w);",
        "input.uvzq.z / input.uvzq.xyw.z);",
        "float4 h = shadowMap.gather_compare(",
    ],
    "slang": [
        "float a = shadowMap.SampleCmp(input.uvz.xy, input.uvz.z);",
        "float b = shadowMap.SampleCmp(input.uv, input.depth);",
        "float c = shadowMap.SampleCmpLevel(input.uvz.xy, input.uvz.z, 0.0);",
        "float d = shadowMap.SampleCmp(input.uvz.xy, input.uvz.z, int2(1, 0));",
        "float e = shadowCube.SampleCmp(input.dirRef.xyz, input.dirRef.w);",
        "float f = shadowArray.SampleCmp(input.dirRef.xyz, input.dirRef.w);",
        "float4 h = shadowMap.GatherCmp(input.uv, input.depth);",
    ],
    "opengl": [
        "float a = texture(shadowMap, uvz);",
        "float b = texture(shadowMap, vec3(uv, depth));",
        "float c = textureLod(shadowMap, vec3(uvz.xy, uvz.z), 0.0);",
        "float d = textureOffset(shadowMap, uvz, ivec2(1, 0));",
        "float e = texture(shadowCube, dirRef);",
        "float f = texture(shadowArray, dirRef);",
        "float g = textureProj(shadowMap, uvzq);",
        "vec4 h = textureGather(shadowMap, uv, depth);",
    ],
}

EDGE = """
shader ShadowEdge {
    sampler2DShadow shadowMap;
    samplerCubeShadow shadowCube;
    samplerCubeArrayShadow shadowCubes;
    sampler2DArrayShadow shadowArray;

    struct FSInput {
        vec3 uvz @ TEXCOORD0;
        vec4 dirLayer @ TEXCOORD1;
        float depth @ TEXCOORD2;
        float lod @ TEXCOORD3;
    };

    vec3 jitter(vec3 p) { return p + vec3(0.001); }

    fragment {
        vec4 main(FSInput input) @ gl_FragColor {
            float a = CALL;
            return vec4(a);
        }
    }
}
"""

HLSL_LEVEL_ZERO = """
Texture2D<float> depthTex : register(t0);
TextureCube<float> cubeTex : register(t1);
Texture2DArray<float> arrayTex : register(t2);
SamplerComparisonState cmp : register(s0);
struct PSInput {
    float4 pos : SV_Position;
    float3 P : TEXCOORD0;
    float4 Q : TEXCOORD1;
};
float4 main(PSInput input) : SV_Target {
    float3 P = input.P;
    float4 Q = input.Q;
    float a = depthTex.SampleCmp(cmp, P.xy, P.z);
    float b = depthTex.SampleCmpLevelZero(cmp, P.xy, P.z);
    float c = depthTex.SampleCmpLevelZero(cmp, P.xy, P.z, int2(1, -1));
    float d = cubeTex.SampleCmpLevelZero(cmp, P, Q.w);
    float e = arrayTex.SampleCmpLevelZero(cmp, P, Q.w, int2(1, 0));
    return float4(a + b, c, d, e);
}
"""


def _translate(tmp_path, source, backend, name="shader.cgl"):
    path = tmp_path / name
    path.write_text(source)
    return crosstl.translate(str(path), backend=backend, format_output=False)


def _edge(tmp_path, call, backend):
    return _translate(tmp_path, EDGE.replace("CALL", call), backend)


def _flat(code):
    return " ".join(code.split())


def _run(command):
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("backend", sorted(EXPECTED))
def test_shadow_samples_become_comparisons(tmp_path, backend):
    code = _flat(_translate(tmp_path, SHADOW_SAMPLES, backend))
    for snippet in EXPECTED[backend]:
        assert snippet in code, snippet
    if backend == "directx":
        assert ".Sample(" not in code
    if backend == "metal":
        assert code.count(".sample_compare(") == 7


def test_spirv_shadow_samples_use_dref_instructions(tmp_path):
    code = _translate(tmp_path, SHADOW_SAMPLES, "vulkan")
    instructions = [
        line.split("=", 1)[-1].split()[0]
        for line in code.splitlines()
        if "OpImage" in line
    ]
    assert instructions.count("OpImageSampleDrefImplicitLod") == 6
    assert instructions.count("OpImageSampleDrefExplicitLod") == 1
    assert instructions.count("OpImageDrefGather") == 1
    assert "OpImageSampleImplicitLod" not in instructions
    if shutil.which("spirv-as") and shutil.which("spirv-val"):
        asm, binary = tmp_path / "out.spvasm", tmp_path / "out.spv"
        asm.write_text(code)
        _run(["spirv-as", "--target-env", "vulkan1.0", str(asm), "-o", str(binary)])
        _run(["spirv-val", "--target-env", "vulkan1.0", str(binary)])


def test_cube_array_shadow_sample_takes_the_compare_operand(tmp_path):
    call = "texture(shadowCubes, input.dirLayer, input.depth)"
    hlsl = _flat(_edge(tmp_path, call, "directx"))
    assert (
        "shadowCubes.SampleCmp(shadowCubesSampler, input.dirLayer, input.depth)" in hlsl
    )
    metal = _flat(_edge(tmp_path, call, "metal"))
    assert "input.dirLayer.xyz, uint(input.dirLayer.w), input.depth)" in metal
    slang = _flat(_edge(tmp_path, call, "slang"))
    assert "shadowCubes.SampleCmp(input.dirLayer, input.depth)" in slang


def test_metal_keeps_biased_shadow_samples(tmp_path):
    metal = _flat(_edge(tmp_path, "texture(shadowMap, input.uvz, 0.5)", "metal"))
    assert "input.uvz.xy, input.uvz.z, bias(0.5))" in metal


@pytest.mark.parametrize(
    "call, backend, reason",
    [
        (
            "texture(shadowMap, jitter(input.uvz))",
            "directx",
            "evaluated once for the coordinate",
        ),
        ("texture(shadowMap, input.uvz, 0.5)", "directx", "level-of-detail bias"),
        ("texture(shadowMap, input.uvz, 0.5)", "vulkan", "level-of-detail bias"),
        (
            "textureProj(shadowArray, input.dirLayer)",
            "directx",
            "only for 1D, 2D and rectangle",
        ),
        (
            "textureOffset(shadowCubes, input.dirLayer, ivec2(1, 0))",
            "metal",
            "cube-array shadow",
        ),
    ],
)
def test_shadow_samples_without_a_compare_form_are_refused(
    tmp_path, call, backend, reason
):
    with pytest.raises(ShadowSampleUnsupportedError) as error:
        _edge(tmp_path, call, backend)
    assert reason in str(error.value)
    assert error.value.code.endswith(f"{backend}-shadow-sample-unsupported")


def test_hlsl_level_zero_compares_keep_their_level(tmp_path):
    crossgl = _flat(_translate(tmp_path, HLSL_LEVEL_ZERO, "cgl", "s.hlsl"))
    assert "float a = textureCompare(depthTex, cmp, P.xy, P.z);" in crossgl
    assert "float b = textureCompareLod(depthTex, cmp, P.xy, P.z, 0.0);" in crossgl
    assert (
        "textureCompareLodOffset(depthTex, cmp, P.xy, P.z, 0.0, ivec2(1, -1))"
        in crossgl
    )
    assert "float d = textureCompareLod(cubeTex, cmp, P, Q.w, 0.0);" in crossgl

    hlsl = _flat(_translate(tmp_path, HLSL_LEVEL_ZERO, "directx", "s.hlsl"))
    assert "depthTex.SampleCmp(cmp, P.xy, P.z);" in hlsl
    assert "depthTex.SampleCmpLevelZero(cmp, P.xy, P.z);" in hlsl
    assert "depthTex.SampleCmpLevelZero(cmp, P.xy, P.z, int2(1, -1));" in hlsl
    assert "cubeTex.SampleCmpLevelZero(cmp, P, Q.w);" in hlsl

    metal = _flat(_translate(tmp_path, HLSL_LEVEL_ZERO, "metal", "s.hlsl"))
    assert "depthTex.sample_compare(cmp, P.xy, P.z, level(0.0));" in metal
    assert "cubeTex.sample_compare(cmp, P, Q.w, level(0.0));" in metal

    glsl = _flat(_translate(tmp_path, HLSL_LEVEL_ZERO, "opengl", "s.hlsl"))
    assert "textureLod(depthTex, vec3(P.xy, P.z), 0.0)" in glsl
    assert "textureLodOffset(depthTex, vec3(P.xy, P.z), 0.0, ivec2(1, (-1)))" in glsl
    # Cube and 2D-array shadows have no core textureLod: zero derivatives select the base level.
    assert "textureGrad(cubeTex, vec4(P, Q.w), vec3(0.0), vec3(0.0))" in glsl
    assert (
        "textureGradOffset(arrayTex, vec4(P, Q.w), vec2(0.0), vec2(0.0), ivec2(1, 0))"
        in glsl
    )
    assert "GL_EXT_texture_shadow_lod" not in glsl

    spirv = _translate(tmp_path, HLSL_LEVEL_ZERO, "vulkan", "s.hlsl")
    assert spirv.count("OpImageSampleDrefExplicitLod") == 4


@pytest.mark.parametrize("backend", ["opengl", "webgl"])
def test_glsl_level_zero_shadow_compares_compile(tmp_path, backend):
    if not shutil.which("glslangValidator"):
        pytest.skip("glslangValidator is not installed")
    source = HLSL_LEVEL_ZERO
    if backend == "webgl":
        # WebGL 2 has no cube or 2D-array textureGradOffset overload on shadows; keep the overloads it has.
        source = source.replace(
            "float e = arrayTex.SampleCmpLevelZero(cmp, P, Q.w, int2(1, 0));",
            "float e = 0.0;",
        )
    path = tmp_path / "out.frag"
    path.write_text(_translate(tmp_path, source, backend, "s.hlsl"))
    _run(["glslangValidator", str(path)])


def test_glsl_explicit_shadow_levels_require_the_shadow_lod_extension(tmp_path):
    calls = (
        "textureLod(shadowArray, input.dirLayer, input.lod) + textureLod(shadowCube, input.dirLayer, input.lod) + "
        "textureLod(shadowCubes, input.dirLayer, input.depth, input.lod) + "
        "textureLod(shadowArray, input.dirLayer, 0.0)"
    )
    glsl = _edge(tmp_path, calls, "opengl")
    assert "#extension GL_EXT_texture_shadow_lod : require" in glsl
    flat = _flat(glsl)
    assert "textureLod(shadowArray, vec4(dirLayer.xyz, dirLayer.w), lod)" in flat
    assert "textureLod(shadowCube, vec4(dirLayer.xyz, dirLayer.w), lod)" in flat
    assert "textureLod(shadowCubes, dirLayer, depth, lod)" in flat
    assert (
        "textureGrad(shadowArray, vec4(dirLayer.xyz, dirLayer.w), vec2(0.0), vec2(0.0))"
        in flat
    )
    if shutil.which("glslangValidator"):
        path = tmp_path / "ext.frag"
        path.write_text(glsl)
        _run(["glslangValidator", str(path)])


def test_webgl_refuses_explicit_levels_without_the_shadow_lod_extension(tmp_path):
    source = EDGE.replace("samplerCubeArrayShadow shadowCubes;", "").replace(
        "CALL", "textureLod(shadowArray, input.dirLayer, input.lod)"
    )
    with pytest.raises(Exception) as error:
        _translate(tmp_path, source, "webgl")
    assert "textureCompareLod explicit LOD requires 2D shadow samplers" in str(
        error.value
    )


def test_metal_output_compiles_with_the_metal_compiler(tmp_path):
    if not shutil.which("xcrun"):
        pytest.skip("xcrun is not installed")
    probe = subprocess.run(["xcrun", "--find", "metal"], capture_output=True, text=True)
    if probe.returncode != 0:
        pytest.skip("the Metal compiler is not installed")
    for name, source, file_name in (
        ("cgl", SHADOW_SAMPLES, "shader.cgl"),
        ("hlsl", HLSL_LEVEL_ZERO, "s.hlsl"),
    ):
        path = tmp_path / f"{name}.metal"
        path.write_text(_translate(tmp_path, source, "metal", file_name))
        _run(["xcrun", "metal", "-c", str(path), "-o", str(tmp_path / f"{name}.air")])


def test_hlsl_output_compiles_with_dxc(tmp_path):
    if not shutil.which("dxc"):
        pytest.skip("dxc is not installed")
    path = tmp_path / "out.hlsl"
    path.write_text(_translate(tmp_path, SHADOW_SAMPLES, "directx"))
    _run(["dxc", "-T", "ps_6_0", "-E", "main", str(path), "-Fo", str(tmp_path / "o")])

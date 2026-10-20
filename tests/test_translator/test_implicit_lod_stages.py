"""Implicit level-of-detail operations outside fragment stages (``crosstl.translator.semantics.implicit_lod``).

CrossGL follows GLSL: an implicit-LOD sample samples the base level outside fragment shaders. SPIR-V, HLSL, WGSL and
Slang sample implicitly only in fragment shaders, so those samples become explicit level-0 samples there; a helper
that both kinds of stage call is cloned for the base-level paths. Bias, derivative, LOD-query and implicit LOD-clamp
operations reached from a base-level stage are refused for every target.
"""

from __future__ import annotations

import shutil
import subprocess

import pytest

import crosstl
from crosstl.translator import ast as A
from crosstl.translator.lexer import Lexer
from crosstl.translator.parser import Parser
from crosstl.translator.semantics import (
    ImplicitLodStageUnsupportedError,
    apply_rule_carrying,
)
from crosstl.translator.semantics.implicit_lod import legalize_implicit_lod

VERTEX_SAMPLING = """
shader VertexTextureLod {
    struct VertexInput {
        vec3 position;
        vec2 texCoord;
    }

    struct VertexOutput {
        vec2 uv;
        vec4 position;
    }

    struct FragmentInput {
        vec2 uv;
    }

    struct FragmentOutput {
        vec4 color;
    }

    sampler2D heightMap;
    sampler2DShadow shadowMap;

    float sampleHeight(vec2 uv) {
        return texture(heightMap, uv).r;
    }

    vertex {
        VertexOutput main(VertexInput input) {
            VertexOutput output;
            float h = sampleHeight(input.texCoord);
            float s = textureCompare(shadowMap, input.texCoord, 0.5);
            float so = textureCompareOffset(shadowMap, input.texCoord, 0.5, ivec2(1, 0));
            vec4 o = textureOffset(heightMap, input.texCoord, ivec2(1, 0));
            vec4 p = textureProj(heightMap, vec3(input.texCoord, 2.0));
            output.uv = input.texCoord;
            output.position = vec4(input.position.xy, h + s + so + o.x + p.x, 1.0);
            return output;
        }
    }

    fragment {
        FragmentOutput main(FragmentInput input) {
            FragmentOutput output;
            float h = sampleHeight(input.uv);
            output.color = vec4(h, textureCompare(shadowMap, input.uv, 0.25), 0.0, 1.0);
            return output;
        }
    }
}
"""

VERTEX_OPERATION = """
shader VertexOperation {
    struct VertexInput {
        vec2 texCoord;
    }

    struct VertexOutput {
        vec4 position;
    }

    sampler2D heightMap;

    vertex {
        VertexOutput main(VertexInput input) {
            VertexOutput output;
            output.position = vec4(OPERATION);
            return output;
        }
    }
}
"""

SEPARATE_SAMPLER = """
shader SeparateSampler {
    struct VertexInput {
        vec2 texCoord;
    }

    struct VertexOutput {
        vec4 position;
    }

    sampler2D heightMap;
    sampler heightSampler;

    vertex {
        VertexOutput main(VertexInput input) {
            VertexOutput output;
            output.position = texture(heightMap, heightSampler, input.texCoord)
                + textureOffset(heightMap, heightSampler, input.texCoord, ivec2(1, 0));
            return output;
        }
    }
}
"""

COMPUTE_SAMPLING = """
shader ComputeSampling {
    sampler2D heightMap;
    buffer float values[];

    compute {
        void main() {
            values[0] = texture(heightMap, vec2(0.5)).r;
        }
    }
}
"""

COMBINED_AND_SEPARATE_TEXTURES = """
shader CombinedAndSeparate {
    struct FragmentInput {
        vec2 uv;
    }

    struct FragmentOutput {
        vec4 color;
    }

    sampler2D colorMap;
    texture2D detailTexture;
    sampler detailSampler;

    fragment {
        FragmentOutput main(FragmentInput input) {
            FragmentOutput output;
            output.color = texture(colorMap, input.uv) + texture(detailTexture, detailSampler, input.uv);
            return output;
        }
    }
}
"""

FRAGMENT_COMPARE_LOD = """
shader FragmentCompareLod {
    struct FragmentInput {
        vec2 uv;
    }

    struct FragmentOutput {
        vec4 color;
    }

    sampler2DShadow shadowMap;
    uniform float level;

    fragment {
        FragmentOutput main(FragmentInput input) {
            FragmentOutput output;
            float a = textureCompareLod(shadowMap, input.uv, 0.5, 0.0);
            float b = textureCompareLodOffset(shadowMap, input.uv, 0.5, 0.0, ivec2(1, 0));
            float c = textureCompareLod(shadowMap, input.uv, 0.5, level);
            output.color = vec4(a, b, c, 1.0);
            return output;
        }
    }
}
"""


def _parse(source):
    return Parser(Lexer(source).get_tokens()).parse()


def _translate(tmp_path, source, backend):
    path = tmp_path / "shader.cgl"
    path.write_text(source)
    return crosstl.translate(str(path), backend=backend, format_output=False)


def _run(command):
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


def _validate_spirv(tmp_path, code):
    if not (shutil.which("spirv-as") and shutil.which("spirv-val")):
        return
    asm, binary = tmp_path / "out.spvasm", tmp_path / "out.spv"
    asm.write_text(code)
    _run(["spirv-as", "--target-env", "vulkan1.1", str(asm), "-o", str(binary)])
    _run(["spirv-val", "--target-env", "vulkan1.1", str(binary)])


def _validate_hlsl(tmp_path, code, profile, entry):
    if not shutil.which("dxc"):
        return
    path = tmp_path / "out.hlsl"
    path.write_text(code)
    _run(
        ["dxc", "-T", profile, "-E", entry, str(path), "-Fo", str(tmp_path / "x.dxil")]
    )


def _stage(program, name):
    return next(stage for stage in program.stages.values() if stage.stage.value == name)


def _function_text(code, signature):
    start = code.index(signature)
    end = code.find("\n}\n", start)
    return code[start : end if end >= 0 else len(code)]


def test_hlsl_vertex_samples_take_level_zero_and_fragment_samples_stay_implicit(
    tmp_path,
):
    hlsl = _translate(tmp_path, VERTEX_SAMPLING, "directx")

    vertex = _function_text(hlsl, "VSMain(")
    assert "sampleHeight_base_lod(input.texCoord)" in vertex
    assert (
        "shadowMap.SampleCmpLevelZero(shadowMapSampler, input.texCoord, 0.5)" in vertex
    )
    assert (
        "shadowMap.SampleCmpLevelZero(shadowMapSampler, input.texCoord, 0.5, int2(1, 0))"
        in vertex
    )
    assert (
        "heightMap.SampleLevel(heightMapSampler, input.texCoord, 0.0, int2(1, 0))"
        in vertex
    )
    assert ".Sample(" not in vertex and "SampleCmp(" not in vertex

    clone = _function_text(hlsl, "float sampleHeight_base_lod(float2 uv) {")
    assert "heightMap.SampleLevel(heightMapSampler, uv, 0.0)" in clone
    original = _function_text(hlsl, "float sampleHeight(float2 uv) {")
    assert "heightMap.Sample(heightMapSampler, uv)" in original

    fragment = _function_text(hlsl, "PSMain(")
    assert "sampleHeight(input.uv)" in fragment
    assert "shadowMap.SampleCmp(shadowMapSampler, input.uv, 0.25)" in fragment

    _validate_hlsl(tmp_path, hlsl, "vs_6_0", "VSMain")
    _validate_hlsl(tmp_path, hlsl, "ps_6_0", "PSMain")


def test_spirv_vertex_samples_use_explicit_lod_and_validate(tmp_path):
    spirv = _translate(tmp_path, VERTEX_SAMPLING, "vulkan")

    vertex_start = spirv.index("OpEntryPoint Vertex")
    assert "OpImageSampleDrefExplicitLod" in spirv
    assert "Lod|ConstOffset" in spirv
    # The fragment entry and the original helper keep their implicit level of detail.
    assert spirv.count("OpImageSampleImplicitLod") == 1
    assert spirv.count("OpImageSampleDrefImplicitLod") == 1
    assert vertex_start >= 0
    _validate_spirv(tmp_path, spirv)


def test_slang_vertex_samples_take_level_zero(tmp_path):
    slang = _translate(tmp_path, VERTEX_SAMPLING, "slang")

    assert "heightMap.SampleLevel(uv, 0.0)" in slang
    assert "heightMap.Sample(uv)" in slang


@pytest.mark.parametrize("backend", ["opengl", "metal"])
def test_targets_with_base_level_implicit_sampling_keep_the_program(tmp_path, backend):
    code = _translate(tmp_path, VERTEX_SAMPLING, backend)

    assert "sampleHeight_base_lod" not in code


def test_separate_sampler_level_follows_the_coordinate():
    program = _parse(SEPARATE_SAMPLER)
    legalize_implicit_lod(program, "directx")

    calls = [
        node
        for node in _calls(_stage(program, "vertex").entry_point)
        if isinstance(node.function, A.IdentifierNode)
        and node.function.name.startswith("texture")
    ]
    assert [call.function.name for call in calls] == ["textureLod", "textureLodOffset"]
    for call in calls:
        assert isinstance(call.arguments[3], A.LiteralNode)
        assert call.arguments[3].value == 0.0


def test_separate_sampler_hlsl_validates(tmp_path):
    hlsl = _translate(tmp_path, SEPARATE_SAMPLER, "directx")

    assert "heightMap.SampleLevel(heightSampler, input.texCoord, 0.0)" in hlsl
    assert (
        "heightMap.SampleLevel(heightSampler, input.texCoord, 0.0, int2(1, 0))" in hlsl
    )
    _validate_hlsl(tmp_path, hlsl, "vs_6_0", "VSMain")


def test_compute_stages_are_left_to_the_emitters():
    program = _parse(COMPUTE_SAMPLING)
    legalize_implicit_lod(program, "directx")

    names = [
        node.function.name
        for node in _calls(_stage(program, "compute").entry_point)
        if isinstance(node.function, A.IdentifierNode)
    ]
    assert "texture" in names and "textureLod" not in names


def test_clone_name_avoids_existing_names(tmp_path):
    source = VERTEX_SAMPLING.replace(
        "    float sampleHeight(vec2 uv) {",
        "    float sampleHeight_base_lod(float x) {\n        return x;\n    }\n\n"
        "    float sampleHeight(vec2 uv) {",
    )
    hlsl = _translate(tmp_path, source, "directx")

    assert "sampleHeight_base_lod2(input.texCoord)" in hlsl
    assert "float sampleHeight_base_lod(float x)" in hlsl


def test_overloads_reached_from_a_base_level_stage_move_with_the_clone(tmp_path):
    source = VERTEX_SAMPLING.replace(
        "    float sampleHeight(vec2 uv) {",
        "    float sampleHeight(float u) {\n        return u;\n    }\n\n"
        "    float sampleHeight(vec2 uv) {",
    ).replace(
        "float h = sampleHeight(input.texCoord);",
        "float h = sampleHeight(input.texCoord) + sampleHeight(input.texCoord.x);",
    )
    hlsl = _translate(tmp_path, source, "directx")

    vertex = _function_text(hlsl, "VSMain(")
    assert "sampleHeight_base_lod(input.texCoord)" in vertex
    assert "sampleHeight_base_lod(input.texCoord.x)" in vertex
    assert "float sampleHeight_base_lod(float u)" in hlsl
    _validate_hlsl(tmp_path, hlsl, "vs_6_0", "VSMain")
    _validate_hlsl(tmp_path, hlsl, "ps_6_0", "PSMain")


@pytest.mark.parametrize(
    "operation, name",
    [
        ("texture(heightMap, input.texCoord, 1.0).x", "texture"),
        (
            "textureOffset(heightMap, input.texCoord, ivec2(1, 0), 1.0).x",
            "textureOffset",
        ),
        ("dFdx(input.texCoord).x", "dFdx"),
        ("fwidth(input.texCoord.x)", "fwidth"),
        ("textureQueryLod(heightMap, input.texCoord).x", "textureQueryLod"),
        ("textureClamp(heightMap, input.texCoord, 1.0).x", "textureClamp"),
    ],
)
@pytest.mark.parametrize("backend", ["directx", "vulkan", "opengl", "metal", "wgsl"])
def test_derivative_operations_in_a_vertex_stage_are_refused(
    tmp_path, operation, name, backend
):
    if backend == "wgsl" and name == "textureClamp":
        pytest.skip("WGSL refuses every minimum-LOD clamp before stage checks")
    source = VERTEX_OPERATION.replace("OPERATION", operation)

    with pytest.raises(ImplicitLodStageUnsupportedError) as error:
        _translate(tmp_path, source, backend)

    assert error.value.operation == name
    assert error.value.stage == "vertex"
    assert error.value.code.endswith("implicit-lod-stage-unsupported")


def test_crossgl_target_keeps_the_program():
    program = _parse(VERTEX_OPERATION.replace("OPERATION", "dFdx(input.texCoord).x"))
    apply_rule_carrying(program, "cgl", "cgl")


def test_spirv_declares_one_sampled_image_type_per_image_type(tmp_path):
    spirv = _translate(tmp_path, COMBINED_AND_SEPARATE_TEXTURES, "vulkan")

    assert spirv.count("OpTypeSampledImage") == 1
    _validate_spirv(tmp_path, spirv)


def test_hlsl_compare_at_literal_level_zero_needs_no_shader_model_6_7(tmp_path):
    hlsl = _translate(tmp_path, FRAGMENT_COMPARE_LOD, "directx")

    assert "shadowMap.SampleCmpLevelZero(shadowMapSampler, input.uv, 0.5)" in hlsl
    assert (
        "shadowMap.SampleCmpLevelZero(shadowMapSampler, input.uv, 0.5, int2(1, 0))"
        in hlsl
    )
    assert "shadowMap.SampleCmpLevel(shadowMapSampler, input.uv, 0.5, level)" in hlsl


def _calls(node):
    from crosstl.translator.semantics.rewrite import child_slots, get_child

    stack = [node]
    while stack:
        current = stack.pop()
        if isinstance(current, A.FunctionCallNode):
            yield current
        children = [get_child(current, f, k) for f, k in child_slots(current)]
        stack.extend(reversed(children))

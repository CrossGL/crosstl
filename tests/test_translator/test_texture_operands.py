"""CrossGL texture operands: compare bias, compare-gather component and sample position (#4796).

CrossGL keeps each operation as written. A target whose generator has not registered the feature refuses it with
TextureOperandUnsupportedError instead of dropping the operand or printing the call verbatim; a compare-gather
component of 0 is dropped, and one that is not an integer constant from 0 to 3 is refused for every target.
"""

import pytest

import crosstl
from crosstl.translator.semantics import (
    ShadowSampleUnsupportedError,
    TextureOperandError,
    TextureOperandUnsupportedError,
    texture_operands,
)
from crosstl.translator.semantics.implicit_lod import ImplicitLodStageUnsupportedError
from crosstl.translator.validation import (
    texture_bias_argument_index,
    texture_gather_component_argument_index,
    texture_intrinsic_max_argument_count,
)

FRAGMENT = """
shader TextureOperands {
    fragment {
        uniform sampler2DShadow shadowMap;
        uniform texture2D depth;
        uniform comparison_sampler cmp;
        uniform texture2DMS msTexture;
        uniform texture2DMSArray msArray;

        vec4 main(vec4 position @ gl_FragCoord) @ gl_FragColor {
            %s
        }
    }
}
"""

COMPARE_BIAS_CALLS = {
    "combined": "return vec4(textureCompare(shadowMap, position.xy, 0.5, 1.0));",
    "separate": "return vec4(textureCompare(depth, cmp, position.xy, 0.5, 1.0));",
    "offset": (
        "return vec4(textureCompareOffset(depth, cmp, position.xy, 0.5, ivec2(1, 0), 1.0));"
    ),
    "clamp": (
        "return vec4(textureCompareClamp(depth, cmp, position.xy, 0.5, 2.0, 1.0));"
    ),
    "offset-clamp": (
        "return vec4(textureCompareOffsetClamp(depth, cmp, position.xy, 0.5, ivec2(1, 0), 2.0, 1.0));"
    ),
}
GATHER_GREEN = "return textureGatherCompare(depth, cmp, position.xy, 0.5, 1);"
SAMPLE_POSITION = "return vec4(textureSamplePosition(msTexture, 1), textureSamplePosition(msArray, 2));"

TARGETS = ["directx", "metal", "opengl", "vulkan", "slang", "wgsl"]


def _write(tmp_path, body, name="probe.cgl"):
    path = tmp_path / name
    path.write_text(FRAGMENT % body)
    return str(path)


def test_compare_bias_and_gather_component_arities():
    assert texture_intrinsic_max_argument_count("textureCompare") == 4
    assert texture_intrinsic_max_argument_count("textureCompare", True) == 5
    assert texture_intrinsic_max_argument_count("textureCompareOffsetClamp", True) == 7
    assert texture_intrinsic_max_argument_count("textureGatherCompare", True) == 5
    assert texture_intrinsic_max_argument_count("textureSamplePosition") == 2
    # The bias follows the reference; an unbiased call has none.
    assert texture_bias_argument_index("textureCompare", False, 4) == 3
    assert texture_bias_argument_index("textureCompare", True, 5) == 4
    assert texture_bias_argument_index("textureCompare", True, 4) is None
    assert texture_bias_argument_index("textureCompareOffset", True, 6) == 5
    # The component follows the reference (and offsets); an unselected gather has none.
    assert texture_gather_component_argument_index("textureGatherCompare", True, 5) == 4
    assert (
        texture_gather_component_argument_index("textureGatherCompare", True, 4) is None
    )
    assert (
        texture_gather_component_argument_index("textureGatherCompareOffset", True, 6)
        == 5
    )
    assert (
        texture_gather_component_argument_index("textureGatherCompareOffsets", True, 6)
        == 5
    )
    assert (
        texture_gather_component_argument_index("textureGatherCompareOffsets", True, 9)
        == 8
    )
    assert (
        texture_gather_component_argument_index("textureGatherCompareOffsets", True, 8)
        is None
    )


@pytest.mark.parametrize(
    "body", [*COMPARE_BIAS_CALLS.values(), GATHER_GREEN, SAMPLE_POSITION]
)
def test_crossgl_output_keeps_the_operands(tmp_path, body):
    out = crosstl.translate(_write(tmp_path, body), backend="cgl")
    assert body.removeprefix("return ").removesuffix(";") in out


@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.parametrize("form", sorted(COMPARE_BIAS_CALLS))
def test_unregistered_targets_refuse_compare_bias(tmp_path, target, form):
    path = _write(tmp_path, COMPARE_BIAS_CALLS[form])
    with pytest.raises(TextureOperandUnsupportedError) as caught:
        crosstl.translate(path, backend=target)
    assert caught.value.feature == texture_operands.COMPARE_BIAS
    assert caught.value.missing_capabilities == (f"{target}.compare-bias",)
    assert "level-of-detail bias" in str(caught.value)


def test_wgsl_cites_its_missing_compare_bias(tmp_path):
    path = _write(tmp_path, COMPARE_BIAS_CALLS["separate"])
    with pytest.raises(
        TextureOperandUnsupportedError, match="textureSampleCompare takes no bias"
    ):
        crosstl.translate(path, backend="wgsl")


def test_unbiased_comparisons_are_unchanged(tmp_path):
    path = _write(
        tmp_path, "return vec4(textureCompare(depth, cmp, position.xy, 0.5));"
    )
    hlsl = crosstl.translate(path, backend="directx")
    assert "depth.SampleCmp(cmp, position.xy, 0.5)" in hlsl


@pytest.mark.parametrize("target", TARGETS)
def test_unregistered_targets_refuse_gather_compare_components(tmp_path, target):
    with pytest.raises(TextureOperandUnsupportedError) as caught:
        crosstl.translate(_write(tmp_path, GATHER_GREEN), backend=target)
    assert caught.value.feature == texture_operands.GATHER_COMPARE_COMPONENT
    assert caught.value.operation == "textureGatherCompare"


@pytest.mark.parametrize(
    "target, reason",
    [
        ("opengl", "refZ operand compares only the first component"),
        ("vulkan", "OpImageDrefGather has no component operand"),
        ("metal", "gather_compare compares only the first component"),
        ("wgsl", "textureGatherCompare compares only the first component"),
    ],
)
def test_first_component_targets_cite_their_gather(tmp_path, target, reason):
    with pytest.raises(TextureOperandUnsupportedError, match=reason):
        crosstl.translate(_write(tmp_path, GATHER_GREEN), backend=target)


def test_a_first_component_gather_drops_the_operand(tmp_path):
    body = "return textureGatherCompare(depth, cmp, position.xy, 0.5, 0);"
    hlsl = crosstl.translate(_write(tmp_path, body), backend="directx")
    assert "depth.GatherCmp(cmp, position.xy, 0.5)" in hlsl
    glsl = crosstl.translate(_write(tmp_path, body), backend="opengl")
    assert (
        "textureGather(" in glsl
        and ", 0)" not in glsl.split("textureGather(", 1)[1].split(";", 1)[0]
    )


@pytest.mark.parametrize("component", ["4", "-1", "position.x > 0.0 ? 1 : 2", "1.0"])
@pytest.mark.parametrize("target", ["directx", "opengl", "metal"])
def test_gather_compare_components_must_be_constants_from_0_to_3(
    tmp_path, component, target
):
    body = f"return textureGatherCompare(depth, cmp, position.xy, 0.5, {component});"
    with pytest.raises(TextureOperandError, match="integer constant from 0 to 3"):
        crosstl.translate(_write(tmp_path, body), backend=target)


def test_hlsl_writes_sample_positions(tmp_path):
    hlsl = crosstl.translate(_write(tmp_path, SAMPLE_POSITION), backend="directx")
    assert "msTexture.GetSamplePosition(1)" in hlsl
    assert "msArray.GetSamplePosition(2)" in hlsl


@pytest.mark.parametrize("target", ["metal", "opengl", "vulkan", "slang", "wgsl"])
def test_other_targets_refuse_sample_positions(tmp_path, target):
    with pytest.raises(TextureOperandUnsupportedError) as caught:
        crosstl.translate(_write(tmp_path, SAMPLE_POSITION), backend=target)
    assert caught.value.feature == texture_operands.SAMPLE_POSITION
    assert caught.value.operation == "textureSamplePosition"


def test_glsl_cites_gl_sample_position(tmp_path):
    with pytest.raises(TextureOperandUnsupportedError, match="gl_SamplePosition"):
        crosstl.translate(_write(tmp_path, SAMPLE_POSITION), backend="opengl")


def test_biased_shadow_samples_stay_refused_until_a_target_registers(tmp_path):
    body = "return vec4(texture(shadowMap, vec3(position.xy, 0.5), 1.0));"
    with pytest.raises(ShadowSampleUnsupportedError, match="level-of-detail bias"):
        crosstl.translate(_write(tmp_path, body), backend="directx")


def test_a_registered_target_receives_the_compare_form_with_its_bias(
    tmp_path, monkeypatch
):
    targets = {
        feature: set(names) for feature, names in texture_operands._TARGETS.items()
    }
    monkeypatch.setattr(texture_operands, "_TARGETS", targets)
    texture_operands.register_texture_operand_target(
        texture_operands.COMPARE_BIAS, "vulkan"
    )
    assert "vulkan" in texture_operands.texture_operand_targets(
        texture_operands.COMPARE_BIAS
    )
    from crosstl.translator import ast as A
    from crosstl.translator.lexer import Lexer
    from crosstl.translator.parser import Parser
    from crosstl.translator.semantics import legalize_shadow_samples

    body = "return vec4(texture(shadowMap, vec3(position.xy, 0.5), 1.0));"
    program = Parser(Lexer(FRAGMENT % body).get_tokens()).parse()
    legalize_shadow_samples(program, "vulkan")
    calls = [
        node
        for node in program.walk()
        if isinstance(node, A.FunctionCallNode)
        and getattr(node.function, "name", None) == "textureCompare"
    ]
    assert len(calls) == 1
    arguments = calls[0].arguments
    assert [type(argument).__name__ for argument in arguments][0] == "IdentifierNode"
    assert len(arguments) == 4
    assert arguments[2].value in (0.5, "0.5")
    assert arguments[3].value in (1.0, "1.0")


def test_registering_an_unknown_feature_is_an_error():
    with pytest.raises(ValueError, match="unknown texture operand feature"):
        texture_operands.register_texture_operand_target("compare-lod", "directx")


VERTEX_BIAS = """
shader BiasedVertexCompare {
    vertex {
        uniform texture2D depth;
        uniform comparison_sampler cmp;
        vec4 main(vec2 uv) @ gl_Position {
            return vec4(textureCompare(depth, cmp, uv, 0.5, 1.0));
        }
    }
}
"""


def test_a_compare_bias_outside_fragment_stages_has_no_meaning(tmp_path, monkeypatch):
    targets = {
        feature: set(names) for feature, names in texture_operands._TARGETS.items()
    }
    targets[texture_operands.COMPARE_BIAS].add("directx")
    monkeypatch.setattr(texture_operands, "_TARGETS", targets)
    path = tmp_path / "vertex.cgl"
    path.write_text(VERTEX_BIAS)
    with pytest.raises(ImplicitLodStageUnsupportedError):
        crosstl.translate(str(path), backend="directx")

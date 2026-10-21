"""The WebGL (GLSL ES 3.00) row: ANGLE's feature list, the fragment-shader probes and the version check."""

import pytest

from tools.coverage import probes, report, runner, validators
from tools.coverage.dialect import UNIFORM_ARRAY_LENGTH
from tools.coverage.features import webgl
from tools.coverage.model import Feature, Param, Signature, Type

ANGLE_TABLE = """
// The format of this file: comments and blank lines are ignored.
GROUP BEGIN Math {"queryFunction": true}
  GROUP BEGIN Common
    DEFAULT METADATA {"essl_level": "COMMON_BUILTINS"}
      genType abs(genType);
    DEFAULT METADATA {"essl_level": "ESSL3_BUILTINS"}
      genType modf(genType, out genType);
      genType trunc(genType);
    DEFAULT METADATA {"essl_level": "ESSL3_1_BUILTINS"}
      genType frexpLater(genType);
    DEFAULT METADATA {"essl_level": "ESSL3_BUILTINS", "essl_extension": "EXT_gpu_shader5"}
      genType fmaByExtension(genType, genType, genType);
  GROUP END Common
GROUP END Math

GROUP BEGIN Texture {"queryFunction": true}
  GROUP BEGIN LodVS {"shader_type": "VERTEX"}
    DEFAULT METADATA {"essl_level": "ESSL3_BUILTINS"}
      vec4 vertexOnly(sampler2D, vec2);
  GROUP END LodVS
  GROUP BEGIN Derivatives {"shader_type": "FRAGMENT"}
    DEFAULT METADATA {"essl_level": "ESSL3_BUILTINS"}
      genType dFdx(genType);
  GROUP END Derivatives
  DEFAULT METADATA {"essl_level": "ESSL3_BUILTINS"}
    gvec4 texelFetchOffset(gsampler2D, ivec2, int, ivec2);
    gvec4 texelFetchOffset(gsampler3D, ivec3, int, ivec3);
GROUP END Texture

GROUP BEGIN Unleveled
  genType noLevel(genType);
GROUP END Unleveled
"""

VEC3 = Type("float", (3,))


def by_name(features):
    return {f.name: f for f in features}


def test_webgl_keeps_glsl_es_300_functions_without_extensions():
    features = by_name(webgl.parse(ANGLE_TABLE))
    # Default metadata ends with its top-level group, so ``noLevel`` has no level and is left out.
    assert sorted(features) == ["abs", "dFdx", "modf", "texelFetchOffset", "trunc"]
    assert features["abs"].group == "math/common"
    assert features["dFdx"].group == "texture/derivatives"
    assert all(f.stage == "fragment" and f.crossgl == f.name for f in features.values())


def test_webgl_makes_generic_types_concrete():
    features = by_name(webgl.parse(ANGLE_TABLE))
    assert features["abs"].signature == Signature(VEC3, (Param(VEC3),))
    assert features["modf"].signature.params[1] == Param(VEC3, "out")
    fetch = features["texelFetchOffset"].signature
    # The probe uses the first overload; the texel offset must be a constant, the coordinate need not be.
    assert fetch.params[0].type == Type.other("sampler2D")
    assert fetch.params[1] == Param(Type("int", (2,)))
    assert fetch.params[3] == Param(Type("int", (2,)), "const")


def test_webgl_rejects_lines_it_cannot_read():
    with pytest.raises(ValueError, match="unrecognised line"):
        webgl.parse("GROUP BEGIN Math\n  this is not a prototype\nGROUP END Math\n")


def _function(name, result, *params):
    return Feature(
        "webgl",
        "function",
        name,
        Signature(result, tuple(Param(p) for p in params)),
        crossgl=name,
        stage="fragment",
    )


SAMPLE = _function(
    "texture", Type("float", (4,)), Type.other("sampler2D"), Type("float", (2,))
)


def test_webgl_probe_is_a_glsl_es_fragment_shader_without_buffers():
    source = probes.build(_function("sin", VEC3, VEC3), "webgl").source
    assert source.startswith("#version 300 es\nprecision highp float;\n")
    assert f"uniform float fin[{UNIFORM_ARRAY_LENGTH}];" in source
    assert "    vec3 outp[1];" in source
    assert "buffer" not in source and "layout(std430" not in source
    assert "vec3 a0 = vec3(fin[i + 0u], fin[i + 1u], fin[i + 2u]);" in source


def test_webgl_probe_declares_samplers_with_a_precision():
    source = probes.build(SAMPLE, "webgl").source
    assert "uniform highp sampler2D u0;" in source
    assert "binding" not in source


def test_crossgl_es_probe_is_a_fragment_stage_with_uniform_arrays():
    source = probes.build(_function("sin", VEC3, VEC3), "crossgl-es").source
    assert "    fragment {" in source
    assert f"        uniform float fin[{UNIFORM_ARRAY_LENGTH}];" in source
    assert "            vec3 outp[1];" in source
    assert "buffer" not in source and "compute" not in source


def test_webgl_probes_never_read_past_the_uniform_arrays():
    wide = _function("wide", Type("float"), *[Type("float", (4, 4))] * 3)
    for language in ("webgl", "crossgl-es"):
        with pytest.raises(probes.ProbeError, match="more than 32 elements"):
            probes.build(wide, language)


def test_webgl_has_no_double_or_explicit_width_types():
    with pytest.raises(probes.ProbeError, match="GLSL ES 3.00 has no double"):
        probes.build(_function("abs", Type("double"), Type("double")), "webgl")


def test_webgl_output_uses_the_crossgl_es_dialect_and_has_no_input():
    assert runner.CROSSGL_DIALECTS["webgl"] == "crossgl-es"
    assert runner.OUTPUT_BACKENDS["webgl"] == "webgl"
    assert "webgl" not in runner.INPUT_BACKENDS
    assert ("webgl", "input") in report.NOT_OFFERED


def test_webgl_validator_requires_glsl_es_300():
    desktop = "#version 460\nvoid main() {\n}\n"
    verdict = validators.validate("webgl", desktop, "fragment")
    assert not verdict.accepted
    assert "#version 300 es" in verdict.log


@pytest.mark.skipif(
    not validators.available("webgl"), reason="glslangValidator is not installed"
)
@pytest.mark.parametrize("stage", ["fragment", "vertex"])
def test_webgl_validator_tells_the_stage_from_the_suffix(stage):
    vertex = "#version 300 es\nin vec4 position;\nvoid main() {\n    gl_Position = position;\n}\n"
    verdict = validators.validate("webgl", vertex, stage)
    # gl_Position exists only in a vertex shader.
    assert verdict.accepted == (stage == "vertex"), verdict.log


@pytest.mark.skipif(
    not validators.available("webgl"), reason="glslangValidator is not installed"
)
@pytest.mark.parametrize(
    "probe_feature",
    [
        _function("sin", VEC3, VEC3),
        SAMPLE,
        _function("inverse", Type("float", (3, 3)), Type("float", (3, 3))),
    ],
    ids=lambda f: f.name,
)
def test_webgl_probes_are_valid_glsl_es(probe_feature):
    probe = probes.build(probe_feature, "webgl")
    verdict = validators.validate("webgl", probe.source, probe_feature.stage)
    assert verdict.accepted, verdict.log


@pytest.mark.skipif(
    not validators.available("webgl"), reason="glslangValidator is not installed"
)
def test_webgl_output_is_measured_end_to_end():
    result = runner.run_output(SAMPLE, "webgl")
    assert result.outcome == "supported", result.detail
    assert result.validator == "glslangValidator"

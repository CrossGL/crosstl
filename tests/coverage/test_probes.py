import pytest

from tools.coverage import probes, validators
from tools.coverage.model import Feature, Param, Signature, Type

VEC3 = Type("float", (3,))


def feature(name, result, *params, stage="compute", crossgl=None):
    signature = Signature(
        result, tuple(p if isinstance(p, Param) else Param(p) for p in params)
    )
    return Feature(
        "glsl",
        "function",
        name,
        signature,
        crossgl=name if crossgl is None else crossgl,
        stage=stage,
    )


def test_crossgl_probe_reads_buffers_and_stores_the_result():
    source = probes.build(feature("mix", VEC3, VEC3, VEC3, VEC3), "crossgl").source
    assert "buffer float* fin;" in source
    assert "buffer vec3* outp;" in source
    assert "vec3 a2 = vec3(fin[i + 6u], fin[i + 7u], fin[i + 8u]);" in source
    assert "vec3 r = mix(a0, a1, a2);" in source
    assert "outp[i] = r;" in source


def test_probe_binds_inout_parameters_to_the_output_buffer():
    source = probes.build(
        feature("atomicAdd", Type("uint"), Param(Type("uint"), "inout"), Type("uint")),
        "glsl",
    ).source
    assert "atomicAdd(outp[i], a1);" in source
    assert "buffer B1 { uint outp[]; }" in source


def test_probe_stores_booleans_as_integers():
    source = probes.build(
        feature("any", Type("bool"), Type("bool", (3,))), "hlsl"
    ).source
    assert "bool3 a0 = bool3((fin[i + 0u] > 0.0)" in source
    assert "outp[i] = r ? 1 : 0;" in source
    assert "RWStructuredBuffer<int> outp" in source


@pytest.mark.parametrize(
    ("language", "spelling"),
    [
        ("crossgl", "mat2x3"),
        ("glsl", "mat2x3"),
        ("hlsl", "float3x2"),
        ("metal", "float2x3"),
    ],
)
def test_matrix_spellings(language, spelling):
    assert probes.spell(Type("float", (2, 3)), language) == spelling


def test_probe_refuses_what_it_cannot_express():
    with pytest.raises(probes.ProbeError, match="sampler2D"):
        probes.build(
            feature("texture", Type("float", (4,)), Type.other("sampler2D")), "crossgl"
        )
    with pytest.raises(probes.ProbeError, match="fragment"):
        probes.build(feature("dFdx", VEC3, VEC3, stage="fragment"), "crossgl")
    with pytest.raises(probes.ProbeError, match="no spelling"):
        probes.build(feature("NMin", VEC3, VEC3, VEC3, crossgl=""), "crossgl")


SIGNATURES = [
    feature("mix", VEC3, VEC3, VEC3, VEC3),
    feature("determinant", Type("float"), Type("float", (3, 3))),
    feature("frexp", VEC3, VEC3, Param(Type("int", (3,)), "out")),
    feature("lessThan", Type("bool", (3,)), VEC3, VEC3),
    feature("atomicAdd", Type("uint"), Param(Type("uint"), "inout"), Type("uint")),
    feature("barrier", Type.other("void")),
    feature("packDouble2x32", Type("double"), Type("uint", (2,))),
]


@pytest.mark.skipif(
    not validators.available("glsl"), reason="glslangValidator is not installed"
)
@pytest.mark.parametrize("probe_feature", SIGNATURES, ids=lambda f: f.name)
def test_glsl_probes_are_valid_glsl(probe_feature):
    probe = probes.build(probe_feature, "glsl")
    verdict = validators.validate("glsl", probe.source)
    assert verdict.accepted, verdict.log

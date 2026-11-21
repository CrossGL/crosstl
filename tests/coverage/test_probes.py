import pytest

from tools.coverage import probes, spellings, validators
from tools.coverage.model import (
    Feature,
    Param,
    Signature,
    Type,
)

from .glsl_features import VEC3, feature


def test_crossgl_probe_reads_buffers_and_stores_the_result():
    source = probes.build(feature("mix", VEC3, VEC3, VEC3, VEC3), "crossgl").source
    assert "buffer float* fin;" in source
    assert "buffer vec3* outp;" in source
    assert "vec3 a2 = vec3(fin[i + 6u], fin[i + 7u], fin[i + 8u]);" in source
    assert "vec3 r = mix(a0, a1, a2);" in source
    assert "outp[i] = r;" in source


@pytest.mark.parametrize("language", ["glsl", "hlsl", "metal", "slang", "wgsl", "cuda"])
def test_a_call_without_arguments_declares_the_input_the_harness_binds(language):
    # The conformance harness binds an input buffer at slot 0, so where it runs the feature's language, the probe
    # declares one that the call never reads.
    lane = Feature(
        language,
        "function",
        "WaveGetLaneIndex",
        Signature(Type("uint"), ()),
        crossgl="WaveGetLaneIndex",
    )
    native = probes.build(lane, language).source
    crossgl = probes.build(lane, "crossgl").source
    if language == "cuda":
        assert "fin" not in native + crossgl
        return
    assert native.count("fin") == 1 and native.index("fin") < native.index("outp")
    assert "buffer float* fin;\n        buffer uint* outp;" in crossgl


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
    with pytest.raises(probes.ProbeError, match="samplerCubeShadow"):
        probes.build(
            feature("texture", Type("float"), Type.other("samplerCubeShadow")),
            "crossgl",
        )
    with pytest.raises(probes.ProbeError, match="no vertex-stage probe"):
        probes.build(feature("dFdx", VEC3, VEC3, stage="vertex"), "crossgl")
    with pytest.raises(probes.ProbeError, match="cuda has no fragment stage"):
        probes.build(feature("dFdx", VEC3, VEC3, stage="fragment"), "cuda")
    with pytest.raises(probes.NoSpelling, match="no spelling"):
        probes.build(feature("NMin", VEC3, VEC3, VEC3, crossgl=""), "crossgl")
    with pytest.raises(probes.ProbeError, match="double"):
        probes.build(feature("abs", Type("double"), Type("double")), "metal")


def test_an_operation_crossgl_writes_as_an_expression_fills_its_holes():
    clamp = feature("NClamp", VEC3, VEC3, VEC3, VEC3, crossgl="fmin(fmax({}, {}), {})")
    assert "vec3 r = fmin(fmax(a0, a1), a2);" in probes.build(clamp, "crossgl").source


def test_a_builtin_value_is_read_rather_than_called():
    # Slang's WaveGetNumWaves() is CrossGL's gl_NumSubgroups, and an argument indexes an arrayed value.
    waves = Feature(
        "slang",
        "function",
        "WaveGetNumWaves",
        Signature(Type("uint"), ()),
        crossgl="gl_NumSubgroups",
    )
    assert "uint r = gl_NumSubgroups;" in probes.build(waves, "crossgl").source
    assert "uint r = WaveGetNumWaves();" in probes.build(waves, "slang").source
    indexed = Feature(
        "slang",
        "function",
        "WaveGetLaneValue",
        Signature(VEC3, (Param(Type("uint")),)),
        crossgl="gl_LaneValues[{}]",
    )
    assert "vec3 r = gl_LaneValues[a0];" in probes.build(indexed, "crossgl").source


@pytest.mark.parametrize("language", ["hlsl", "slang"])
def test_select_passes_the_condition_last_in_crossgl(language):
    # HLSL's and Slang's select(c, t, f) is CrossGL's select(f, t, c), on the same inputs as the native probe, so
    # the two can be run against each other.
    ivec2 = Type("int", (2,))
    select = Feature(
        language,
        "function",
        "select",
        Signature(ivec2, (Param(Type("bool", (2,))), Param(ivec2), Param(ivec2))),
        crossgl=spellings.spelling(language, "select", set()),
    )
    native = probes.build(select, language).source
    crossgl = probes.build(select, "crossgl").source
    assert "int2 r = select(a0, a1, a2);" in native
    assert "ivec2 r = select(a2, a1, a0);" in crossgl
    assert "bvec2 a0 = bvec2(" in crossgl
    assert native.count("fin[") == crossgl.count("fin[")
    assert native.count("iin[") == crossgl.count("iin[")


def test_an_interpolant_is_read_straight_from_a_shader_input():
    vec2 = Type("float", (2,))
    centroid = feature("interpolateAtCentroid", vec2, vec2, stage="fragment")
    glsl = probes.build(centroid, "glsl").source
    assert "layout(location = 0) in vec2 a0;" in glsl
    assert "vec2 r = interpolateAtCentroid(a0);" in glsl
    crossgl = probes.build(centroid, "crossgl").source
    assert "        vec2 uv;\n        vec2 a0;\n    }" in crossgl
    assert "vec2 r = interpolateAtCentroid(input.a0);" in crossgl
    hlsl = Feature(
        "hlsl",
        "function",
        "EvaluateAttributeCentroid",
        centroid.signature,
        stage="fragment",
    )
    assert "float4 main(float2 a0 : TEXCOORD0) : SV_Target {" in (
        probes.build(hlsl, "hlsl").source
    )


def test_baseline_reads_the_same_arguments_without_the_call():
    mix = feature("mix", VEC3, VEC3, VEC3, VEC3)
    probe = probes.build(mix, "glsl").source
    baseline = probes.build(mix, "glsl", baseline=True).source
    assert "mix(" not in baseline
    assert "outp[i] = outp[i];" in baseline
    assert baseline.count("fin[") == probe.count("fin[")


def test_hlsl_stores_the_first_row_of_a_matrix_result():
    # float3x2 in HLSL: two columns of three rows, so a row has two components.
    source = probes.build(
        feature("transpose", Type("float", (2, 3)), Type("float", (3, 2))), "hlsl"
    ).source
    assert "RWStructuredBuffer<float2> outp" in source
    assert "float2x3 a0 = float2x3(float3(" in source


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


HLSL_SIGNATURES = [
    feature("lerp", VEC3, VEC3, VEC3, VEC3),
    feature("transpose", Type("float", (2, 3)), Type("float", (3, 2))),
    feature("any", Type("bool"), Type("bool", (3,))),
    feature(
        "InterlockedAdd", Type.other("void"), Param(Type("uint"), "inout"), Type("uint")
    ),
    feature("asdouble", Type("double", (2,)), Type("uint", (2,)), Type("uint", (2,))),
    feature("asint16", Type("int16", (2,)), Type("half", (2,))),
    feature("asuint16", Type("uint16"), Type("int16")),
    feature("unpack_s8s32", Type("int", (4,)), Type("int8x4_packed")),
    feature("pack_u8", Type("uint8x4_packed"), Type("int", (4,))),
]
HLSL_BY_NAME = {f.name: f for f in HLSL_SIGNATURES}


def test_16_bit_scalars_are_read_and_stored_through_32_bit_buffers():
    asint16, asuint16 = HLSL_BY_NAME["asint16"], HLSL_BY_NAME["asuint16"]
    source = probes.build(asint16, "hlsl").source
    assert "RWStructuredBuffer<int2> outp : register(u1);" in source
    assert "half2 a0 = half2((half)fin[i + 0u], (half)fin[i + 1u]);" in source
    assert "int16_t2 r = asint16(a0);" in source
    assert "outp[i] = (int2)r;" in source
    assert "int16_t a0 = (int16_t)iin[i + 0u];" in probes.build(asuint16, "hlsl").source
    assert "half2 a0" in probes.build(asint16, "slang").source
    crossgl = probes.build(asint16, "crossgl").source
    assert (
        "f16vec2 a0 = f16vec2(float16(fin[i + 0u]), float16(fin[i + 1u]));" in crossgl
    )
    assert "i16vec2 r = asint16(a0);" in crossgl
    assert "outp[i] = ivec2(r);" in crossgl
    # A language whose 16-bit spelling has not been checked leaves the feature untested.
    with pytest.raises(probes.ProbeError, match="16-bit"):
        probes.build(asint16, "glsl")


def test_packed_8_bit_integers_are_read_and_stored_as_32_bit_words():
    unpack, pack = HLSL_BY_NAME["unpack_s8s32"], HLSL_BY_NAME["pack_u8"]
    source = probes.build(unpack, "hlsl").source
    assert "int8_t4_packed a0 = (int8_t4_packed)uin[i + 0u];" in source
    assert "int4 r = unpack_s8s32(a0);" in source
    source = probes.build(pack, "hlsl").source
    assert "RWStructuredBuffer<uint> outp : register(u1);" in source
    assert "uint8_t4_packed r = pack_u8(a0);" in source
    assert "outp[i] = (uint)r;" in source
    # CrossGL's HLSL importer reads a packed word as uint.
    assert "uint r = pack_u8(a0);" in probes.build(pack, "crossgl").source
    with pytest.raises(probes.ProbeError, match="packed 8-bit"):
        probes.build(pack, "metal")


@pytest.mark.skipif(not validators.available("hlsl"), reason="dxc is not installed")
@pytest.mark.parametrize("probe_feature", HLSL_SIGNATURES, ids=lambda f: f.name)
@pytest.mark.parametrize("baseline", [False, True], ids=["probe", "baseline"])
def test_hlsl_probes_are_valid_hlsl(probe_feature, baseline):
    probe = probes.build(probe_feature, "hlsl", baseline=baseline)
    verdict = validators.validate("hlsl", probe.source, entry=probe.entry)
    assert verdict.accepted, verdict.log


def test_slang_probe_is_an_hlsl_kernel_with_a_shader_attribute():
    source = probes.build(HLSL_SIGNATURES[0], "slang").source
    assert source.startswith("RWStructuredBuffer<float> fin;\n")
    assert '[shader("compute")]\n[numthreads(1, 1, 1)]\nvoid main(' in source
    assert "register(" not in source
    assert probes.build(HLSL_SIGNATURES[0], "slang").suffix == ".slang"


@pytest.mark.skipif(not validators.available("slang"), reason="slangc is not installed")
@pytest.mark.parametrize("probe_feature", HLSL_SIGNATURES, ids=lambda f: f.name)
@pytest.mark.parametrize("baseline", [False, True], ids=["probe", "baseline"])
def test_slang_probes_are_valid_slang(probe_feature, baseline):
    probe = probes.build(probe_feature, "slang", baseline=baseline)
    verdict = validators.validate("slang", probe.source, entry=probe.entry)
    assert verdict.accepted, verdict.log

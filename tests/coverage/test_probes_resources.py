"""Probes for features that take textures, samplers, images, objects or constant arguments."""

import pytest

from tools.coverage import lowering, probes, validators
from tools.coverage.features.common import prototype
from tools.coverage.model import (
    STRING,
    VOID,
    Feature,
    Param,
    Signature,
    Type,
)

from .glsl_features import feature


def method(owner, name, result, *params):
    return Feature(
        "hlsl",
        "method",
        f"{owner}.{name}",
        Signature(result, tuple(Param(p) for p in params)),
        group=owner,
        crossgl=name,
        spelling=name,
    )


def test_64_bit_integers_are_read_through_32_bit_buffers_in_hlsl():
    exchange = method(
        "RWByteAddressBuffer",
        "InterlockedExchange64",
        Type.other("void"),
        Type("uint"),
        Type("int64"),
    )
    source = probes.build(exchange, "hlsl").source
    assert "int64_t a1 = (int64_t)iin[i + 0u];" in source
    assert "obj.InterlockedExchange64(a0, a1);" in source
    assert (
        "int64_t a1 = int64_t(iin[i + 0u]);" in probes.build(exchange, "crossgl").source
    )


SAMPLE_LEVEL = method(
    "Texture2D",
    "SampleLevel",
    Type("float", (4,)),
    Type.other("sampler"),
    Type("float", (2,)),
    Type("float"),
)


def test_method_probe_declares_its_object_and_sampler():
    source = probes.build(SAMPLE_LEVEL, "hlsl").source
    assert "Texture2D<float4> obj : register(t0);" in source
    assert "SamplerState smp : register(s0);" in source
    assert "float4 r = obj.SampleLevel(smp, a1, a2);" in source
    crossgl = probes.build(SAMPLE_LEVEL, "crossgl").source
    assert "        Texture2D<vec4> obj;" in crossgl
    assert "        SamplerState smp;" in crossgl
    assert "vec4 r = obj.SampleLevel(smp, a1, a2);" in crossgl


VEC4 = Type("float", (4,))
TEXTURE_LOD = feature(
    "textureLod", VEC4, Type.other("sampler2D"), Type("float", (2,)), Type("float")
)
IMAGE_ATOMIC = feature(
    "imageAtomicAdd",
    Type("uint"),
    Type.other("uimage2D"),
    Type("int", (2,)),
    Type("uint"),
)
TEXEL_OFFSET = feature(
    "textureLodOffset",
    VEC4,
    Type.other("sampler2D"),
    Type("float", (2,)),
    Type("float"),
    Param(Type("int", (2,)), "const"),
)
ATOMIC_COUNTER = feature(
    "atomicCounterAdd", Type("uint"), Type.other("atomic_uint"), Type("uint")
)
GATHER_OFFSETS = feature(
    "textureGatherOffsets",
    VEC4,
    Type.other("sampler2D"),
    Type("float", (2,)),
    Param(Type("int", (2,), count=4), "const"),
)


def test_function_probe_declares_the_samplers_and_images_it_passes():
    source = probes.build(TEXTURE_LOD, "glsl").source
    assert "layout(binding = 8) uniform sampler2D u0;" in source
    assert "vec4 r = textureLod(u0, a1, a2);" in source
    image = probes.build(IMAGE_ATOMIC, "glsl").source
    assert "layout(r32ui, binding = 8) uniform uimage2D u0;" in image
    crossgl = probes.build(IMAGE_ATOMIC, "crossgl").source
    assert "        layout(r32ui) uniform uimage2D u0;" in crossgl
    assert "uint r = imageAtomicAdd(u0, a1, a2);" in crossgl


def test_a_glsl_probe_passes_a_shadow_sampler_for_a_depth_comparison():
    shadow = feature(
        "texture", Type("float"), Type.other("sampler2DShadow"), Type("float", (3,))
    )
    source = probes.build(shadow, "glsl").source
    assert "layout(binding = 8) uniform sampler2DShadow u0;" in source
    assert "float r = texture(u0, a1);" in source


def test_a_subpass_input_names_its_input_attachment():
    subpass = feature("subpassLoad", VEC4, Type.other("subpassInput"), stage="fragment")
    source = probes.build(subpass, "glsl").source
    assert (
        "layout(input_attachment_index = 0, binding = 8) uniform subpassInput u0;"
        in source
    )
    assert "vec4 r = subpassLoad(u0);" in source
    crossgl = probes.build(subpass, "crossgl").source
    assert (
        "        layout(input_attachment_index = 0) uniform subpassInput u0;" in crossgl
    )


def test_atomic_counters_use_the_first_atomic_counter_binding():
    source = probes.build(ATOMIC_COUNTER, "glsl").source
    assert "layout(binding = 0) uniform atomic_uint u0;" in source
    assert "uint r = atomicCounterAdd(u0, a1);" in source
    assert (
        "        uniform atomic_uint u0;" in probes.build(
            ATOMIC_COUNTER, "crossgl"
        ).source
    )


def test_constant_vector_arguments_are_zero_vectors():
    source = probes.build(TEXEL_OFFSET, "glsl").source
    assert "textureLodOffset(u0, a1, a2, ivec2(0, 0))" in source


@pytest.mark.parametrize("language", ["slang", "crossgl"])
def test_constant_unsigned_arguments_are_one(language):
    # A cluster size must be a power of two, so zero, the other constants' value, is no valid count or size.
    rotate = feature(
        "WaveClusteredRotate",
        Type("float"),
        Type("float"),
        Type("uint"),
        Param(Type("uint"), "const"),
    )
    source = probes.build(rotate, language).source
    assert "float r = WaveClusteredRotate(a0, a1, 1u);" in source


@pytest.mark.parametrize("language", ["slang", "crossgl"])
def test_compile_time_conditions_are_true(language):
    # A false condition would fail the probe however CrossTL translates the call.
    check = feature(
        "static_assert", VOID, Param(Type("bool"), "condition"), Param(STRING, "const")
    )
    source = probes.build(check, language).source
    assert 'static_assert(true, "");' in source


@pytest.mark.parametrize("language", ["glsl", "crossgl"])
def test_constant_array_arguments_are_local_constants(language):
    source = probes.build(GATHER_OFFSETS, language).source
    zero = "ivec2(0, 0)"
    assert f"const ivec2 a2[4] = ivec2[4]({', '.join([zero] * 4)});" in source
    assert "vec4 r = textureGatherOffsets(u0, a1, a2);" in source


def test_array_arguments_that_need_not_be_constant_stay_untested():
    probe = feature("f", Type("float"), Param(Type("float", count=2)), crossgl="f")
    with pytest.raises(probes.ProbeError, match="array parameters"):
        probes.build(probe, "glsl")


@pytest.mark.skipif(
    not validators.available("glsl"), reason="glslangValidator is not installed"
)
@pytest.mark.parametrize(
    "probe_feature",
    [TEXTURE_LOD, IMAGE_ATOMIC, TEXEL_OFFSET, ATOMIC_COUNTER, GATHER_OFFSETS],
    ids=lambda f: f.name,
)
def test_glsl_texture_and_image_probes_are_valid_glsl(probe_feature):
    probe = probes.build(probe_feature, "glsl")
    verdict = validators.validate("glsl", probe.source)
    assert verdict.accepted, verdict.log


def test_ray_query_is_a_local_variable():
    probe = method("RayQuery", "CommittedRayT", Type("float"))
    source = probes.build(probe, "hlsl").source
    assert (
        "    RayQuery<RAY_FLAG_NONE> obj;\n    float r = obj.CommittedRayT();" in source
    )


def test_method_probe_needs_an_object_it_can_declare():
    with pytest.raises(probes.ProbeError, match="no probe for FutureObject objects"):
        probes.build(method("FutureObject", "IsValid", Type("bool")), "hlsl")
    with pytest.raises(probes.ProbeError, match="no probe for metal Texture2D"):
        probes.build(SAMPLE_LEVEL, "metal")


GATHER_RAW = method(
    "Texture2D",
    "GatherRaw",
    Type("uint", (4,)),
    Type.other("sampler"),
    Type("float", (2,)),
)
FEEDBACK_LEVEL = method(
    "FeedbackTexture2DArray",
    "WriteSamplerFeedbackLevel",
    Type.other("void"),
    Type.other("Texture2DArray"),
    Type.other("sampler"),
    Type("float", (3,)),
    Type("float"),
)


def test_gather_raw_reads_a_uint_texture():
    source = probes.build(GATHER_RAW, "hlsl").source
    assert "Texture2D<uint> obj : register(t0);" in source
    assert "Texture2D<uint> obj;" in probes.build(GATHER_RAW, "crossgl").source


def test_feedback_probe_declares_the_texture_it_records():
    source = probes.build(FEEDBACK_LEVEL, "hlsl").source
    assert (
        "FeedbackTexture2DArray<SAMPLER_FEEDBACK_MIN_MIP> obj : register(u8);" in source
    )
    assert "Texture2DArray<float4> tex : register(t0);" in source
    assert "obj.WriteSamplerFeedbackLevel(tex, smp, a2, a3);" in source


# DXC cannot lower a feedback texture to SPIR-V, so this probe is only checked as DXIL.
@pytest.mark.skipif(not validators.available("hlsl"), reason="dxc is not installed")
def test_feedback_probe_is_valid_hlsl():
    probe = probes.build(FEEDBACK_LEVEL, "hlsl")
    verdict = validators.validate("hlsl", probe.source, entry=probe.entry)
    assert verdict.accepted, verdict.log


HLSL_METHODS = [
    SAMPLE_LEVEL,
    GATHER_RAW,
    method(
        "TextureCUBE",
        "GatherCmp",
        Type("float", (4,)),
        Type.other("sampler_cmp"),
        Type("float", (3,)),
        Type("float"),
    ),
    method(
        "RWByteAddressBuffer",
        "Store2",
        Type.other("void"),
        Type("uint"),
        Type("uint", (2,)),
    ),
    method("StructuredBuffer", "Load", Type("float"), Type("int")),
    # DXC's table names this type Texture2DArrayMS; HLSL spells it Texture2DMSArray.
    method(
        "Texture2DArrayMS", "Load", Type("float", (4,)), Type("int", (3,)), Type("int")
    ),
    method("RayQuery", "CommittedRayT", Type("float")),
    method(
        "RWByteAddressBuffer",
        "InterlockedAdd64",
        Type.other("void"),
        Type("uint"),
        Type("uint64"),
    ),
]


@pytest.mark.skipif(not validators.available("hlsl"), reason="dxc is not installed")
@pytest.mark.parametrize("probe_feature", HLSL_METHODS, ids=lambda f: f.name)
def test_hlsl_method_probes_are_valid_hlsl(probe_feature):
    probe = probes.build(probe_feature, "hlsl")
    verdict = validators.validate("hlsl", probe.source, entry=probe.entry)
    assert verdict.accepted, verdict.log


HIT_OBJECT = Feature(
    "hlsl",
    "method",
    "DxHitObject.GetRayTMin",
    Signature(Type("float"), ()),
    group="DxHitObject",
    crossgl="GetRayTMin",
    spelling="GetRayTMin",
    stage="ray-generation",
)


def test_hit_object_methods_query_a_default_hit_object():
    source = probes.build(HIT_OBJECT, "hlsl").source
    assert '[shader("raygeneration")]' in source
    assert "    dx::HitObject obj;" in source
    assert "float r = obj.GetRayTMin();" in source


@pytest.mark.skipif(not validators.available("hlsl"), reason="dxc is not installed")
def test_hit_object_probe_is_valid_hlsl():
    probe = probes.build(HIT_OBJECT, "hlsl")
    verdict = validators.validate("hlsl", probe.source, HIT_OBJECT.stage, probe.entry)
    assert verdict.accepted, verdict.log


# A baseline only has to lower to SPIR-V, which is all the semantic layer does with it. The DXIL validator rejects a
# ray query baseline because it declares the query without using it.
@pytest.mark.skipif(
    not lowering.can_lower("hlsl"), reason="dxc or spirv-dis is not installed"
)
@pytest.mark.parametrize("probe_feature", HLSL_METHODS, ids=lambda f: f.name)
def test_hlsl_method_baselines_lower_to_spirv(probe_feature):
    probe = probes.build(probe_feature, "hlsl", baseline=True)
    assembly, log = lowering.lower("hlsl", probe.source, probe.entry)
    assert assembly, log


WGSL_SAMPLE_LEVEL = Feature(
    "wgsl",
    "function",
    "textureSampleLevel",
    Signature(
        VEC4,
        (
            Param(Type.other("texture_2d<f32>")),
            Param(Type.other("sampler")),
            Param(Type("float", (2,))),
            Param(Type("float")),
        ),
    ),
    crossgl="textureLod",
    counterpart=TEXTURE_LOD.signature,
)


def test_wgsl_probe_binds_a_texture_and_a_sampler_and_crossgl_calls_the_counterpart():
    source = probes.build(WGSL_SAMPLE_LEVEL, "wgsl").source
    assert "@group(0) @binding(8) var u0: texture_2d<f32>;" in source
    assert "@group(0) @binding(9) var u1: sampler;" in source
    assert "var r: vec4<f32> = textureSampleLevel(u0, u1, a2, a3);" in source
    crossgl = probes.build(WGSL_SAMPLE_LEVEL, "crossgl").source
    assert "uniform sampler2D u0;" in crossgl
    assert "vec4 r = textureLod(u0, a1, a2);" in crossgl


@pytest.mark.skipif(not validators.available("wgsl"), reason="naga is not installed")
def test_wgsl_texture_probe_is_valid_wgsl():
    probe = probes.build(WGSL_SAMPLE_LEVEL, "wgsl")
    verdict = validators.validate("wgsl", probe.source, entry=probe.entry)
    assert verdict.accepted, verdict.log


def test_a_counterpart_stands_in_for_a_signature_only_in_the_crossgl_probe():
    name, counterpart = prototype("vec3 frexp(vec3, out ivec3)")
    feature = Feature(
        "wgsl", "function", "frexp", None, crossgl=name, counterpart=counterpart
    )
    source = probes.build(feature, "crossgl").source
    assert "ivec3 a1;" in source
    assert "vec3 r = frexp(a0, a1);" in source
    with pytest.raises(probes.ProbeError, match="no signature"):
        probes.build(feature, "wgsl")

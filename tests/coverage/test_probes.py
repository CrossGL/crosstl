from dataclasses import replace

import pytest

from tools.coverage import probes, validators
from tools.coverage.features.common import prototype
from tools.coverage.model import (
    COOP_COLUMNS,
    STRING,
    VOID,
    Feature,
    Param,
    Signature,
    Type,
    coop_vector,
)

from .glsl_features import VEC3, feature


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


@pytest.mark.skipif(not validators.available("hlsl"), reason="dxc is not installed")
@pytest.mark.parametrize("probe_feature", HLSL_SIGNATURES, ids=lambda f: f.name)
@pytest.mark.parametrize("baseline", [False, True], ids=["probe", "baseline"])
def test_hlsl_probes_are_valid_hlsl(probe_feature, baseline):
    probe = probes.build(probe_feature, "hlsl", baseline=baseline)
    verdict = validators.validate("hlsl", probe.source, entry=probe.entry)
    assert verdict.accepted, verdict.log


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
    not validators.can_lower("hlsl"), reason="dxc or spirv-dis is not installed"
)
@pytest.mark.parametrize("probe_feature", HLSL_METHODS, ids=lambda f: f.name)
def test_hlsl_method_baselines_lower_to_spirv(probe_feature):
    probe = probes.build(probe_feature, "hlsl", baseline=True)
    assembly, log = validators.lower("hlsl", probe.source, probe.entry)
    assert assembly, log


WGSL_SIGNATURES = [
    feature("mix", VEC3, VEC3, VEC3, VEC3),
    feature("select", VEC3, VEC3, VEC3, Type("bool")),
    feature("determinant", Type("float"), Type("float", (3, 3))),
    feature("atomicAdd", Type("int"), Param(Type("int"), "inout"), Type("int")),
    feature("workgroupBarrier", Type.other("void")),
    feature(
        "insertBits",
        Type("int", (3,)),
        Type("int", (3,)),
        Type("int", (3,)),
        Type("uint"),
        Type("uint"),
    ),
]


@pytest.mark.skipif(not validators.available("wgsl"), reason="naga is not installed")
@pytest.mark.parametrize("probe_feature", WGSL_SIGNATURES, ids=lambda f: f.name)
@pytest.mark.parametrize("baseline", [False, True], ids=["probe", "baseline"])
def test_wgsl_probes_are_valid_wgsl(probe_feature, baseline):
    probe = probes.build(probe_feature, "wgsl", baseline=baseline)
    verdict = validators.validate("wgsl", probe.source, entry=probe.entry)
    assert verdict.accepted, verdict.log


def test_wgsl_probe_binds_atomics_through_a_pointer():
    source = probes.build(WGSL_SIGNATURES[3], "wgsl").source
    assert "var<storage, read_write> outp: array<atomic<i32>>;" in source
    assert "atomicAdd(&outp[i], a1)" in source
    # The baseline cannot read an atomic element directly, so it leaves the buffer alone.
    assert (
        "outp[i]" not in probes.build(WGSL_SIGNATURES[3], "wgsl", baseline=True).source
    )


def test_wgsl_has_no_output_parameters():
    frexp = feature("frexp", VEC3, VEC3, Param(Type("int", (3,)), "out"))
    with pytest.raises(probes.ProbeError, match="output parameters"):
        probes.build(frexp, "wgsl")


WORKGROUP_LOAD = Feature(
    "wgsl",
    "function",
    "workgroupUniformLoad",
    Signature(Type("float"), (Param(Type("float"), "workgroup"),)),
    crossgl="workgroupUniformLoad",
)


def test_workgroup_parameter_takes_a_shared_variable_the_probe_fills():
    source = probes.build(WORKGROUP_LOAD, "wgsl").source
    assert "var<workgroup> a0: f32;" in source
    assert "    a0 = fin[i + 0u];" in source
    assert "    var r: f32 = workgroupUniformLoad(&a0);" in source
    crossgl = probes.build(WORKGROUP_LOAD, "crossgl").source
    assert "        shared float a0;" in crossgl
    assert "            float r = workgroupUniformLoad(a0);" in crossgl


@pytest.mark.skipif(not validators.available("wgsl"), reason="naga is not installed")
def test_workgroup_parameter_probe_is_valid_wgsl():
    probe = probes.build(WORKGROUP_LOAD, "wgsl")
    verdict = validators.validate("wgsl", probe.source, entry=probe.entry)
    assert verdict.accepted, verdict.log


METAL_SIGNATURES = [
    feature("fma", VEC3, VEC3, VEC3, VEC3),
    feature("select", VEC3, VEC3, VEC3, Type("bool", (3,))),
    feature("frexp", VEC3, VEC3, Param(Type("int", (3,)), "out")),
    feature("simd_sum", Type("float"), Type("float")),
    feature(
        "atomic_fetch_add_explicit",
        Type("int"),
        Param(Type("int"), "inout"),
        Type("int"),
        Param(Type.other("memory_order"), "const"),
    ),
    feature(
        "atomic_max", Type.other("void"), Param(Type("uint64"), "inout"), Type("uint64")
    ),
    feature(
        "threadgroup_barrier",
        Type.other("void"),
        Param(Type.other("mem_flags"), "const"),
    ),
    feature(
        "atomic_compare_exchange_weak",
        Type("bool"),
        Param(Type("int"), "inout"),
        Param(Type("int"), "address"),
        Type("int"),
    ),
    feature("simd_ballot", Type.other("simd_vote"), Type("bool")),
    feature("quad_active_threads_mask", Type.other("quad_vote")),
    feature("pack_half_to_unorm4x8", Type("uint"), Type("half", (4,))),
    feature("unpack_unorm565_to_half", Type("half", (3,)), Type("uint")),
]


@pytest.mark.skipif(
    not validators.available("metal"), reason="the Metal toolchain is not installed"
)
@pytest.mark.parametrize("probe_feature", METAL_SIGNATURES, ids=lambda f: f.name)
def test_metal_probes_are_valid_metal(probe_feature):
    probe = probes.build(probe_feature, "metal")
    verdict = validators.validate("metal", probe.source, entry=probe.entry)
    assert verdict.accepted, verdict.log


def test_cuda_hip_and_crossgl_pass_the_address_of_a_local_for_a_pointer_to_values():
    norm = Feature(
        "cuda",
        "function",
        "normf",
        Signature(Type("float"), (Param(Type("int")), Param(Type("float"), "address"))),
        crossgl="norm",
    )
    for language in ("cuda", "hip"):
        source = probes.build(replace(norm, language=language), language).source
        assert "    float a1 = fin[i + 0u];" in source
        assert "    float r = normf(a0, &a1);" in source
    source = probes.build(norm, "crossgl").source
    assert "            float a1 = fin[i + 0u];" in source
    assert "            float r = norm(a0, &a1);" in source
    # A language without pointers to locals says so rather than spelling one.
    with pytest.raises(probes.ProbeError, match="no probe passes glsl a pointer"):
        probes.build(replace(norm, language="glsl"), "glsl")


def test_hip_and_crossgl_pass_the_empty_string_for_a_c_string():
    nan = Feature(
        "hip",
        "function",
        "nanf",
        Signature(Type("float"), (Param(STRING, "const"),)),
        crossgl="nan",
    )
    assert '    float r = nanf("");' in probes.build(nan, "hip").source
    assert '            float r = nan("");' in probes.build(nan, "crossgl").source
    # Languages without strings say so rather than spelling one.
    with pytest.raises(probes.ProbeError, match="parameters of type string"):
        probes.build(replace(nan, language="glsl"), "glsl")


def test_metal_compare_exchange_takes_the_address_of_the_expected_value():
    exchange = next(
        f for f in METAL_SIGNATURES if f.name == "atomic_compare_exchange_weak"
    )
    source = probes.build(exchange, "metal").source
    assert "    int a1 = iin[i + 0u];" in source
    assert "atomic_compare_exchange_weak(&outp[i], &a1, a2);" in source
    with pytest.raises(probes.ProbeError, match="no probe passes hlsl a pointer"):
        probes.build(exchange, "hlsl")


def test_metal_ballots_are_read_as_their_vote_integer():
    ballot = feature("simd_ballot", Type.other("simd_vote"), Type("bool"))
    source = probes.build(ballot, "metal").source
    # The 64-bit mask is stored as a 32-bit buffer element, as every 64-bit result is.
    assert "device uint* outp" in source
    assert "ulong r = static_cast<simd_vote::vote_t>(simd_ballot(a0));" in source
    assert "outp[i] = uint(r);" in source
    mask = feature("quad_active_threads_mask", Type.other("quad_vote"))
    source = probes.build(mask, "metal").source
    assert (
        "uint r = static_cast<quad_vote::vote_t>(quad_active_threads_mask());" in source
    )
    # CrossGL has no mask type to read, so the operation is an IR gap, not an untested probe.
    with pytest.raises(probes.NoSpelling, match="no type for Metal's quad_vote"):
        probes.build(mask, "crossgl")


def test_metal_half_arguments_are_read_from_float_buffers():
    pack, unpack = METAL_SIGNATURES[-2:]
    source = probes.build(pack, "metal").source
    assert "half4 a0 = half4(half(fin[i + 0u]), half(fin[i + 1u])" in source
    assert "uint r = pack_half_to_unorm4x8(a0);" in source
    source = probes.build(unpack, "metal").source
    assert "device float3* outp [[buffer(1)]]" in source
    assert "outp[i] = float3(r);" in source


def test_metal_atomic_probe_passes_the_output_element_and_a_relaxed_order():
    add = feature(
        "atomic_fetch_add_explicit",
        Type("int"),
        Param(Type("int"), "inout"),
        Type("int"),
        Param(Type.other("memory_order"), "const"),
    )
    source = probes.build(add, "metal").source
    assert "device atomic_int* outp [[buffer(1)]]" in source
    assert "atomic_fetch_add_explicit(&outp[i], a1, memory_order_relaxed);" in source


CUDA_SIGNATURES = [
    feature("fmaf", Type("float"), Type("float"), Type("float"), Type("float")),
    feature("frexpf", Type("float"), Type("float"), Param(Type("int"), "out")),
    feature("atomicAdd", Type("int"), Param(Type("int"), "inout"), Type("int")),
    feature("__popc", Type("int"), Type("uint")),
    feature("__syncthreads", Type.other("void")),
    feature("llabs", Type("int64"), Type("int64")),
    feature("__popcll", Type("int"), Type("uint64")),
]


def test_cuda_probe_passes_buffers_as_kernel_pointers():
    source = probes.build(CUDA_SIGNATURES[2], "cuda").source
    assert 'extern "C" __global__ void probe(int* iin, int* outp) {' in source
    assert "atomicAdd(&outp[i], a1);" in source
    frexp = probes.build(CUDA_SIGNATURES[1], "cuda").source
    assert "frexpf(a0, &" in frexp
    vector = probes.build(feature("normalize", VEC3, VEC3), "cuda").source
    assert "float3 a0 = make_float3(fin[i + 0u], fin[i + 1u], fin[i + 2u]);" in vector


def test_cuda_has_no_matrices_or_boolean_vectors():
    with pytest.raises(probes.ProbeError, match="matrix"):
        probes.build(
            feature("transpose", Type("float", (2, 2)), Type("float", (2, 2))), "cuda"
        )
    with pytest.raises(probes.ProbeError, match="bool vectors"):
        probes.build(feature("any", Type("bool"), Type("bool", (3,))), "cuda")


def test_64_bit_integers_are_read_and_stored_through_32_bit_buffers():
    llabs, popcll = CUDA_SIGNATURES[-2:]
    source = probes.build(llabs, "cuda").source
    assert 'extern "C" __global__ void probe(int* iin, int* outp) {' in source
    assert "long long a0 = (long long)iin[i + 0u];" in source
    assert "outp[i] = (int)r;" in source
    assert "unsigned long long a0 = (unsigned long long)uin[i + 0u];" in (
        probes.build(popcll, "cuda").source
    )
    crossgl = probes.build(llabs, "crossgl").source
    assert "int64_t a0 = int64_t(iin[i + 0u]);" in crossgl
    assert "outp[i] = int(r);" in crossgl
    # A language whose 64-bit spelling has not been checked leaves the feature untested.
    with pytest.raises(probes.ProbeError, match="64-bit integer"):
        probes.build(llabs, "glsl")


@pytest.mark.skipif(not validators.available("cuda"), reason="nvcc is not installed")
@pytest.mark.parametrize("probe_feature", CUDA_SIGNATURES, ids=lambda f: f.name)
@pytest.mark.parametrize("baseline", [False, True], ids=["probe", "baseline"])
def test_cuda_probes_are_valid_cuda(probe_feature, baseline):
    probe = probes.build(probe_feature, "cuda", baseline=baseline)
    verdict = validators.validate("cuda", probe.source, entry=probe.entry)
    assert verdict.accepted, verdict.log


def test_hip_probe_is_a_cuda_kernel_that_includes_the_hip_runtime():
    source = probes.build(CUDA_SIGNATURES[2], "hip").source
    assert source.startswith("#include <hip/hip_runtime.h>\n")
    assert 'extern "C" __global__ void probe(int* iin, int* outp) {' in source
    assert probes.build(CUDA_SIGNATURES[0], "hip").suffix == ".hip"


@pytest.mark.skipif(not validators.available("hip"), reason="hipcc is not installed")
@pytest.mark.parametrize("probe_feature", CUDA_SIGNATURES, ids=lambda f: f.name)
@pytest.mark.parametrize("baseline", [False, True], ids=["probe", "baseline"])
def test_hip_probes_are_valid_hip(probe_feature, baseline):
    probe = probes.build(probe_feature, "hip", baseline=baseline)
    verdict = validators.validate("hip", probe.source, entry=probe.entry)
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


@pytest.mark.skipif(not validators.available("slang"), reason="slangc is not installed")
def test_slang_validation_needs_an_entry_point():
    source = "RWStructuredBuffer<float> outp;\nvoid helper() { outp[0] = 1.0; }\n"
    assert not validators.validate("slang", source, entry="main").accepted


@pytest.mark.skipif(
    not validators.available("slang", "hlsl"), reason="slangc or dxc is not installed"
)
def test_slang_functions_spirv_lacks_compile_for_hlsl():
    tess = feature(
        "ProcessIsolineTessFactors",
        VOID,
        Type("float"),
        Type("float"),
        Param(Type("float"), "out"),
        Param(Type("float"), "out"),
        stage="tessellation-control",
    )
    tess = replace(tess, target="hlsl")
    probe = probes.build(tess, "slang")
    verdict = validators.validate(
        "slang", probe.source, tess.stage, probe.entry, "hlsl"
    )
    assert verdict.accepted, verdict.log
    # Without the HLSL target the probe compiles for SPIR-V, which has no such function.
    assert not validators.validate(
        "slang", probe.source, tess.stage, probe.entry
    ).accepted


COOP4 = coop_vector("float", 4)
COMPONENT = Param(Type.other("CoopVecComponentType"), "const")
COOP_MATMUL = replace(
    feature(
        "coopVecMatMul",
        COOP4,
        COOP4,
        COMPONENT,
        Type.other("RWByteAddressBuffer"),
        Type("int"),
        COMPONENT,
        Param(Type.other("CoopVecMatrixLayout"), "const"),
        Param(Type("bool"), "const"),
        Param(Type("uint"), "const"),
        crossgl="coopVecMatMul<float, 4, 4, float>",
    ),
    language="slang",
    spelling="coopVecMatMul<float, 4, 4, float>",
)
COOP_GROUPSHARED = replace(
    feature(
        "coopVecLoadGroupshared",
        COOP4,
        Param(Type("float", count=4), "workgroup"),
        Type("int"),
        crossgl="coopVecLoadGroupshared<4, float, 4>",
    ),
    language="slang",
    spelling="coopVecLoadGroupshared<4, float, 4>",
)
COOP_PACKED = replace(
    feature(
        "coopVecMatMulPacked",
        COOP4,
        COOP4,
        COMPONENT,
        Param(COOP_COLUMNS, "const"),
        Type.other("RWByteAddressBuffer"),
        Type("int"),
        COMPONENT,
        Param(Type.other("CoopVecMatrixLayout"), "const"),
        Param(Type("bool"), "const"),
        Param(Type("uint"), "const"),
        crossgl="coopVecMatMulPacked<float, 4, 4, float>",
    ),
    language="slang",
    spelling="coopVecMatMulPacked<float, 4, 4, float>",
)
COOP_REDUCE = replace(
    feature(
        "coopVecReduceSumAccumulate",
        VOID,
        COOP4,
        Type.other("RWByteAddressBuffer"),
        Type("int"),
    ),
    language="slang",
)


def test_slang_probe_fills_a_cooperative_vector_and_reads_its_first_element():
    source = probes.build(COOP_MATMUL, "slang").source
    assert "RWByteAddressBuffer u0;" in source
    assert (
        "CoopVec<float, 4> a0 = CoopVec<float, 4>(fin[i + 0u], fin[i + 1u], fin[i + 2u], fin[i + 3u]);"
        in source
    )
    assert (
        "float r = coopVecMatMul<float, 4, 4, float>(a0, CoopVecComponentType.Float32, u0, a3, "
        "CoopVecComponentType.Float32, CoopVecMatrixLayout.RowMajor, false, 1u)[0];"
        in source
    )
    # CrossGL spells the vector, its enumerations and the buffer as Slang does.
    crossgl = probes.build(COOP_MATMUL, "crossgl").source
    assert "uniform RWByteAddressBuffer u0;" in crossgl
    assert "CoopVec<float, 4> a0 = CoopVec<float, 4>(" in crossgl
    assert (
        "a3, CoopVecComponentType.Float32, CoopVecMatrixLayout.RowMajor, false, 1u)[0];"
        in crossgl
    )


def test_slang_probe_passes_the_vector_length_as_a_packed_column_count():
    for language in ("slang", "crossgl"):
        source = probes.build(COOP_PACKED, language).source
        assert "(a0, CoopVecComponentType.Float32, 4, u0, a4, " in source


def test_slang_probe_fills_a_workgroup_array_element_by_element():
    source = probes.build(COOP_GROUPSHARED, "slang").source
    assert "groupshared float a0[4];" in source
    assert "    a0[3] = fin[i + 3u];" in source
    assert "coopVecLoadGroupshared<4, float, 4>(a0, a1)[0]" in source
    assert "shared float a0[4];" in probes.build(COOP_GROUPSHARED, "crossgl").source


@pytest.mark.skipif(not validators.available("slang"), reason="slangc is not installed")
@pytest.mark.parametrize(
    "probe_feature",
    [COOP_MATMUL, COOP_PACKED, COOP_GROUPSHARED, COOP_REDUCE],
    ids=lambda f: f.name,
)
def test_slang_cooperative_vector_probes_are_valid_slang(probe_feature):
    probe = probes.build(probe_feature, "slang")
    verdict = validators.validate("slang", probe.source, entry=probe.entry)
    assert verdict.accepted, verdict.log


MOJO_SIGNATURES = [
    feature("fma", Type("float"), Type("float"), Type("float"), Type("float")),
    feature("isnan", Type("bool"), Type("float")),
    feature("pop_count", Type("int"), Type("int")),
    feature("ldexp", Type("double"), Type("double"), Type("int")),
    feature("gcd", Type("int64"), Type("int64"), Type("int64")),
    feature("frexp", Type.other("StaticTuple[Float32, 2]"), Type("float")),
    feature(
        "divmod", Type.other("Tuple[Float32, Float32]"), Type("float"), Type("float")
    ),
]


def test_mojo_probe_spells_the_word_sized_int_and_reads_the_first_of_a_pair():
    source = probes.build(MOJO_SIGNATURES[4], "mojo").source
    assert "    var a0: Int = Int(iin[i + 0])\n" in source
    assert "    var r: Int = gcd(a0, a1)\n    outp[i] = Int32(r)\n" in source
    source = probes.build(MOJO_SIGNATURES[5], "mojo").source
    assert "    var r: Float32 = frexp(a0)[0]\n    outp[i] = r\n" in source
    # Only Mojo reads a pair.
    with pytest.raises(probes.ProbeError, match="Tuple"):
        probes.build(MOJO_SIGNATURES[6], "cuda")


def test_mojo_probe_is_a_gpu_kernel_over_pointers():
    source = probes.build(MOJO_SIGNATURES[1], "mojo").source
    assert "from max.gpu import global_idx\n" in source
    assert (
        "def probe(fin: UnsafePointer[Float32, MutAnyOrigin], "
        "outp: UnsafePointer[Int32, MutAnyOrigin]):" in source
    )
    assert "    var a0: Float32 = fin[i + 0]\n    var r: Bool = isnan(a0)\n" in source
    assert "outp[i] = Int32(1) if r else Int32(0)" in source
    vector = probes.build(feature("normalize", VEC3, VEC3), "mojo").source
    assert (
        "var a0: SIMD[DType.float32, 3] = SIMD[DType.float32, 3](fin[i + 0]," in vector
    )
    assert probes.build(MOJO_SIGNATURES[0], "mojo").suffix == ".mojo"
    with pytest.raises(probes.ProbeError, match="matrix"):
        probes.build(
            feature("transpose", Type("float", (2, 2)), Type("float", (2, 2))), "mojo"
        )


@pytest.mark.skipif(
    not validators.available("mojo"), reason="the Mojo compiler is not installed"
)
@pytest.mark.parametrize("probe_feature", MOJO_SIGNATURES, ids=lambda f: f.name)
@pytest.mark.parametrize("baseline", [False, True], ids=["probe", "baseline"])
def test_mojo_probes_are_valid_mojo(probe_feature, baseline):
    probe = probes.build(probe_feature, "mojo", baseline=baseline)
    verdict = validators.validate("mojo", probe.source, entry=probe.entry)
    assert verdict.accepted, verdict.log


@pytest.mark.skipif(
    not validators.available("mojo"), reason="the Mojo compiler is not installed"
)
def test_mojo_validation_type_checks_functions_nothing_calls():
    source = "def unused() -> Float32:\n    return nosuch(1.0)\n"
    verdict = validators.validate("mojo", source)
    assert not verdict.accepted
    assert verdict.log.startswith("probe.mojo:2:12: error: use of unknown declaration")


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

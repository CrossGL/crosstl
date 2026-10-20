import pytest

from tools.coverage import probes, validators
from tools.coverage.model import VOID, Feature, Param, Signature, Type

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


DFDX = feature("dFdx", VEC3, VEC3, stage="fragment")


@pytest.mark.parametrize(
    ("language", "entry"),
    [
        ("crossgl", "FragmentOutput main(FragmentInput input) {"),
        ("glsl", "layout(location = 0) out vec4 color;"),
        ("hlsl", "float4 main() : SV_Target {"),
        ("metal", "fragment float4 probe(device float* fin [[buffer(0)]]"),
        ("rust", "#[spirv(fragment)]"),
        ("slang", '[shader("fragment")]\nfloat4 main() : SV_Target {'),
        ("wgsl", "@fragment\nfn main() -> @location(0) vec4<f32> {"),
    ],
)
def test_fragment_features_get_a_fragment_shader(language, entry):
    probe = probes.build(DFDX, language)
    assert entry in probe.source
    assert "compute" not in probe.source and "numthreads" not in probe.source
    assert probe.suffix == (
        ".frag" if language == "glsl" else probes.DIALECTS[language].suffix
    )


@pytest.mark.parametrize("language", ["glsl", "hlsl", "metal", "slang", "wgsl"])
def test_fragment_probes_are_valid(language):
    if not validators.available(language):
        pytest.skip(f"no {language} validator")
    name = {"hlsl": "ddx", "metal": "dfdx", "slang": "ddx", "wgsl": "dpdx"}.get(
        language, "dFdx"
    )
    spelled = Feature(
        language, "function", name, DFDX.signature, crossgl="dFdx", stage="fragment"
    )
    for baseline in (False, True):
        probe = probes.build(spelled, language, baseline=baseline)
        verdict = validators.validate(language, probe.source, "fragment", probe.entry)
        assert verdict.accepted, verdict.log


WORLD_RAY_ORIGIN = feature("WorldRayOrigin", VEC3, stage="closest-hit")
HIT_PARAMETERS = "void main(inout Payload payload, in BuiltInTriangleIntersectionAttributes attributes) {"


@pytest.mark.parametrize(
    ("language", "entry"),
    [
        (
            "crossgl",
            "    ray_closest_hit {\n        buffer vec3* outp;\n        void main(Payload payload @ payload, "
            "BuiltInTriangleIntersectionAttributes attributes @ hit_attribute) {",
        ),
        (
            "hlsl",
            f'RWStructuredBuffer<float3> outp : register(u0);\n[shader("closesthit")]\n{HIT_PARAMETERS}',
        ),
        (
            "slang",
            f'RWStructuredBuffer<float3> outp;\n[shader("closesthit")]\n{HIT_PARAMETERS}',
        ),
    ],
)
def test_raytracing_features_get_a_shader_of_their_stage(language, entry):
    probe = probes.build(WORLD_RAY_ORIGIN, language)
    assert entry in probe.source
    assert "numthreads" not in probe.source
    # Payload access qualifiers are required from HLSL shader model 6.7; Slang takes a plain struct.
    payload = {"hlsl": "struct [raypayload] Payload", "slang": "struct Payload\n{"}
    if language in payload:
        assert payload[language] in probe.source


def test_a_raytracing_shader_without_a_hit_takes_no_payload():
    reorder = feature("ReorderThread", VOID, Type("uint"), stage="ray-generation")
    source = probes.build(reorder, "hlsl").source
    assert '[shader("raygeneration")]\nvoid main() {' in source
    assert "Payload" not in source
    assert "    ray_generation {" in probes.build(reorder, "crossgl").source
    with pytest.raises(probes.ProbeError, match="no closest-hit-stage probe for glsl"):
        probes.build(WORLD_RAY_ORIGIN, "glsl")


@pytest.mark.parametrize("language", ["hlsl", "slang"])
def test_raytracing_probes_are_valid(language):
    if not validators.available(language):
        pytest.skip(f"no {language} validator")
    for stage, name in (("closest-hit", "WorldRayOrigin"), ("any-hit", "HitKind")):
        result = Type("uint") if name == "HitKind" else VEC3
        spelled = Feature(
            language, "function", name, Signature(result, ()), crossgl=name, stage=stage
        )
        for baseline in (False, True):
            probe = probes.build(spelled, language, baseline=baseline)
            verdict = validators.validate(language, probe.source, stage, probe.entry)
            assert verdict.accepted, verdict.log


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
]


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


def test_function_probe_declares_the_samplers_and_images_it_passes():
    source = probes.build(TEXTURE_LOD, "glsl").source
    assert "layout(binding = 8) uniform sampler2D u0;" in source
    assert "vec4 r = textureLod(u0, a1, a2);" in source
    image = probes.build(IMAGE_ATOMIC, "glsl").source
    assert "layout(r32ui, binding = 8) uniform uimage2D u0;" in image
    crossgl = probes.build(IMAGE_ATOMIC, "crossgl").source
    assert "        layout(r32ui) uniform uimage2D u0;" in crossgl
    assert "uint r = imageAtomicAdd(u0, a1, a2);" in crossgl


def test_constant_vector_arguments_are_zero_vectors():
    source = probes.build(TEXEL_OFFSET, "glsl").source
    assert "textureLodOffset(u0, a1, a2, ivec2(0, 0))" in source


@pytest.mark.skipif(
    not validators.available("glsl"), reason="glslangValidator is not installed"
)
@pytest.mark.parametrize(
    "probe_feature", [TEXTURE_LOD, IMAGE_ATOMIC, TEXEL_OFFSET], ids=lambda f: f.name
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
    with pytest.raises(probes.ProbeError, match="no probe for NodeOutput objects"):
        probes.build(method("NodeOutput", "IsValid", Type("bool")), "hlsl")
    with pytest.raises(probes.ProbeError, match="no probe for metal Texture2D"):
        probes.build(SAMPLE_LEVEL, "metal")


HLSL_METHODS = [
    SAMPLE_LEVEL,
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
]


@pytest.mark.skipif(not validators.available("hlsl"), reason="dxc is not installed")
@pytest.mark.parametrize("probe_feature", HLSL_METHODS, ids=lambda f: f.name)
def test_hlsl_method_probes_are_valid_hlsl(probe_feature):
    probe = probes.build(probe_feature, "hlsl")
    verdict = validators.validate("hlsl", probe.source, entry=probe.entry)
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
]


@pytest.mark.skipif(
    not validators.available("metal"), reason="the Metal toolchain is not installed"
)
@pytest.mark.parametrize("probe_feature", METAL_SIGNATURES, ids=lambda f: f.name)
def test_metal_probes_are_valid_metal(probe_feature):
    probe = probes.build(probe_feature, "metal")
    verdict = validators.validate("metal", probe.source, entry=probe.entry)
    assert verdict.accepted, verdict.log


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


MOJO_SIGNATURES = [
    feature("fma", Type("float"), Type("float"), Type("float"), Type("float")),
    feature("isnan", Type("bool"), Type("float")),
    feature("pop_count", Type("int"), Type("int")),
    feature("ldexp", Type("double"), Type("double"), Type("int")),
]


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

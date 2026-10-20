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
    with pytest.raises(probes.NoSpelling, match="no spelling"):
        probes.build(feature("NMin", VEC3, VEC3, VEC3, crossgl=""), "crossgl")
    with pytest.raises(probes.ProbeError, match="double"):
        probes.build(feature("abs", Type("double"), Type("double")), "metal")


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
]


@pytest.mark.skipif(
    not validators.available("metal"), reason="the Metal toolchain is not installed"
)
@pytest.mark.parametrize("probe_feature", METAL_SIGNATURES, ids=lambda f: f.name)
def test_metal_probes_are_valid_metal(probe_feature):
    probe = probes.build(probe_feature, "metal")
    verdict = validators.validate("metal", probe.source, entry=probe.entry)
    assert verdict.accepted, verdict.log


CUDA_SIGNATURES = [
    feature("fmaf", Type("float"), Type("float"), Type("float"), Type("float")),
    feature("frexpf", Type("float"), Type("float"), Param(Type("int"), "out")),
    feature("atomicAdd", Type("int"), Param(Type("int"), "inout"), Type("int")),
    feature("__popc", Type("int"), Type("uint")),
    feature("__syncthreads", Type.other("void")),
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

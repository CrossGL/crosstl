"""Probes in CUDA, HIP, Slang and Mojo."""

from dataclasses import replace

import pytest

from tools.coverage import probes, validators
from tools.coverage.model import (
    COOP_COLUMNS,
    VOID,
    Param,
    Type,
    coop_vector,
)

from .glsl_features import VEC3, feature

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
    feature("mulwide", Type("uint64"), Type("uint"), Type("uint")),
]


def test_mojo_probe_spells_the_word_sized_int_and_reads_the_first_of_a_pair():
    source = probes.build(MOJO_SIGNATURES[4], "mojo").source
    assert "    var a0: Int = Int(iin[i + 0])\n" in source
    assert "    var r: Int = gcd(a0, a1)\n    outp[i] = Int32(r)\n" in source
    source = probes.build(MOJO_SIGNATURES[5], "mojo").source
    assert "    var r: Float32 = frexp(a0)[0]\n    outp[i] = r\n" in source
    source = probes.build(MOJO_SIGNATURES[7], "mojo").source
    assert "    var r: UInt64 = mulwide(a0, a1)\n    outp[i] = UInt32(r)\n" in source
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

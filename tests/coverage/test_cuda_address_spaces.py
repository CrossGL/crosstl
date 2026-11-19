"""CUDA's address space conversions: how they are read from Clang's intrinsics header and probed."""

from dataclasses import replace

import pytest

from tools.coverage import probes, validators
from tools.coverage.features import cuda
from tools.coverage.model import GENERIC_POINTER, Param, Signature, Type

# The conversions as __clang_cuda_intrinsics.h defines them, under Clang's own names.
INTRINSICS = """
#if CUDA_VERSION >= 11000
extern "C" {
__device__ inline size_t __nv_cvta_generic_to_global_impl(const void *__ptr) {
  return (size_t)(void __attribute__((address_space(1))) *)__ptr;
}
__device__ inline size_t __nv_cvta_generic_to_shared_impl(const void *__ptr) {
  return (size_t)(void __attribute__((address_space(3))) *)__ptr;
}
__device__ inline size_t __nv_cvta_generic_to_constant_impl(const void *__ptr) {
  return (size_t)(void __attribute__((address_space(4))) *)__ptr;
}
__device__ inline size_t __nv_cvta_generic_to_local_impl(const void *__ptr) {
  return (size_t)(void __attribute__((address_space(5))) *)__ptr;
}
__device__ inline void *__nv_cvta_global_to_generic_impl(size_t __ptr) {
  return (void *)(void __attribute__((address_space(1))) *)__ptr;
}
__device__ inline void *__nv_cvta_shared_to_generic_impl(size_t __ptr) {
  return (void *)(void __attribute__((address_space(3))) *)__ptr;
}
__device__ inline void *__nv_cvta_constant_to_generic_impl(size_t __ptr) {
  return (void *)(void __attribute__((address_space(4))) *)__ptr;
}
__device__ inline void *__nv_cvta_local_to_generic_impl(size_t __ptr) {
  return (void *)(void __attribute__((address_space(5))) *)__ptr;
}
} // extern "C"
#endif
"""
SPACES = ("global", "shared", "constant", "local")


def conversions():
    return {f.name: f for f in cuda.parse("", "", frozenset(), INTRINSICS)}


def test_each_conversion_takes_a_pointer_into_the_space_it_names():
    features = conversions()
    assert sorted(features) == sorted(
        [f"__cvta_generic_to_{space}" for space in SPACES]
        + [f"__cvta_{space}_to_generic" for space in SPACES]
    )
    pointers = {
        "global": "buffer",
        "shared": "workgroup address",
        "constant": "constant address",
        "local": "address",
    }
    for space, direction in pointers.items():
        # size_t is 64 bits wide in device code on Linux.
        assert features[f"__cvta_generic_to_{space}"].signature == Signature(
            Type("uint64"), (Param(Type("float"), direction),)
        )
        assert features[f"__cvta_{space}_to_generic"].signature == Signature(
            Type.other(GENERIC_POINTER), (Param(Type("uint64")),)
        )


def lines(feature, language):
    return [
        line.strip() for line in probes.build(feature, language).source.splitlines()
    ]


def test_cuda_and_crossgl_pass_the_address_of_a_variable_in_each_space():
    features = conversions()
    shared = features["__cvta_generic_to_shared"]
    source = lines(shared, "cuda")
    # A __shared__ variable is declared before the kernel and filled from the input buffer.
    assert source.index("__shared__ float a0;") < source.index(
        'extern "C" __global__ void probe(float* fin, unsigned int* outp) {'
    )
    assert "a0 = fin[i + 0u];" in source
    assert "unsigned long long r = __cvta_generic_to_shared(&a0);" in source
    assert "outp[i] = (unsigned int)r;" in source
    source = lines(shared, "crossgl")
    assert "shared float a0;" in source
    assert "uint64_t r = __cvta_generic_to_shared(&a0);" in source
    assert "outp[i] = uint(r);" in source
    # Constant memory is set by the host, so the probe declares it without a value.
    constant = features["__cvta_generic_to_constant"]
    assert "__constant__ float a0;" in lines(constant, "cuda")
    assert "uniform float a0;" in lines(constant, "crossgl")
    for language in ("cuda", "crossgl"):
        assert "__cvta_generic_to_constant(&a0);" in " ".join(lines(constant, language))
        source = " ".join(lines(features["__cvta_generic_to_global"], language))
        assert "__cvta_generic_to_global(fin);" in source
        source = lines(features["__cvta_generic_to_local"], language)
        assert "float a0 = fin[i + 0u];" in source
        assert "__cvta_generic_to_local(&a0);" in " ".join(source)
    # HIP's shared memory is spelled as CUDA's.
    assert "__shared__ float a0;" in lines(replace(shared, language="hip"), "hip")


def test_a_generic_pointer_result_is_held_in_the_languages_pointer_type():
    shared = conversions()["__cvta_shared_to_generic"]
    source = lines(shared, "cuda")
    assert "unsigned long long a0 = (unsigned long long)uin[i + 0u];" in source
    assert "void* r = __cvta_shared_to_generic(a0);" in source
    source = lines(shared, "crossgl")
    assert "uint64_t a0 = uint64_t(uin[i + 0u]);" in source
    assert "ptr<void> r = __cvta_shared_to_generic(a0);" in source


def test_a_language_without_constant_memory_says_so():
    constant = conversions()["__cvta_generic_to_constant"]
    # With a 32-bit result, which GLSL probes can store, only the constant memory is missing.
    narrow = Signature(Type("uint"), constant.signature.params)
    with pytest.raises(
        probes.ProbeError, match="no probe declares glsl constant memory"
    ):
        probes.build(replace(constant, language="glsl", signature=narrow), "glsl")


@pytest.mark.skipif(not validators.available("cuda"), reason="nvcc is not installed")
@pytest.mark.parametrize("name", sorted(conversions()))
@pytest.mark.parametrize("baseline", [False, True], ids=["probe", "baseline"])
def test_conversion_probes_are_valid_cuda(name, baseline):
    probe = probes.build(conversions()[name], "cuda", baseline=baseline)
    verdict = validators.validate("cuda", probe.source, entry=probe.entry)
    assert verdict.accepted, verdict.log

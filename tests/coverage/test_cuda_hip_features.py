"""CUDA and HIP feature lists, read from the CUDA and HIP headers that ship with Clang."""

from tools.coverage.features import cuda, hip
from tools.coverage.model import (
    STRING,
    Param,
    Signature,
    Type,
)


def by_name(features):
    return {feature.name: feature for feature in features}


CUDA_MATH = """
#define __DEVICE__ static __device__ __forceinline__
__DEVICE__ double erf(double __a) { return __nv_erf(__a); }
__DEVICE__ float erff(float __a) { return __nv_erff(__a); }
__DEVICE__ double fabs(double __a) { return __nv_fabs(__a); }
__DEVICE__ float fabsf(float __a) { return __nv_fabsf(__a); }
__DEVICE__ double frexp(double __a, int *__b) { return __nv_frexp(__a, __b); }
__DEVICE__ float frexpf(float __a, int *__b) { return __nv_frexpf(__a, __b); }
__DEVICE__ long labs(long __a) { return llabs(__a); }
__DEVICE__ double powi(double __a, int __b) { return __nv_powi(__a, __b); }
__DEVICE__ double norm(int __dim, const double *__t) { return __nv_norm(__dim, __t); }
__DEVICE__ void *memcpy(void *__a, const void *__b, size_t __c) {
  return __builtin_memcpy(__a, __b, __c);
}
"""

CUDA_DEVICE = """
__DEVICE__ int __all(int __a) { return __nvvm_vote_all(__a); }
__DEVICE__ void __brkpt() { __asm__ __volatile__("brkpt;"); }
__DEVICE__ void __attribute__((overloadable)) __brkpt(int __a) { __brkpt(); }
__DEVICE__ float __fAtomicAdd(float *__p, float __v) {
  return __nvvm_atom_add_gen_f(__p, __v);
}
__DEVICE__ int __iAtomicAdd(int *__p, int __v) {
  return __nvvm_atom_add_gen_i(__p, __v);
}
__DEVICE__ unsigned int __uAtomicAdd_block(unsigned int *__p, unsigned int __v) {
  return __nvvm_atom_cta_add_gen_i((int *)__p, __v);
}
__DEVICE__ int __popc(unsigned int __a) { return __nv_popc(__a); }
__DEVICE__ int __isfinited(double __a) { return __nv_isfinited(__a); }
#ifdef _MSC_VER
__DEVICE__ int __isnanl(long double __a);
#endif
__DEVICE__ /* clock_t= */ int clock() { return __nvvm_read_ptx_sreg_clock(); }
__DEVICE_VOID__ void __threadfence(void) { __nvvm_membar_gl(); }
"""


def test_cuda_reads_device_functions_and_leaves_out_the_rest():
    features = by_name(cuda.parse(CUDA_MATH, CUDA_DEVICE, {"abs", "frexp"}))
    assert sorted(features) == [
        "__popc",
        "__threadfence",
        "atomicAdd",
        "atomicAdd_block",
        "clock",
        "erf",
        "erff",
        "fabs",
        "fabsf",
        "frexp",
        "frexpf",
        "labs",
        "norm",
    ]
    # ``long`` is 64 bits wide on Linux, where nvcc validates the probes.
    assert features["labs"].signature.params[0].type == Type("int64")
    assert features["__threadfence"].group == "intrinsics"
    assert features["fabsf"].group == "math"
    # The comment before clock's result type is not part of it.
    assert features["clock"].signature == Signature(Type("int"), ())
    # A pointer to constant values takes the address of a local.
    assert features["norm"].signature.params == (
        Param(Type("int")),
        Param(Type("double"), "address"),
    )


def test_cuda_atomics_take_the_int_helper_and_bind_the_pointer_inout():
    features = by_name(cuda.parse(CUDA_MATH, CUDA_DEVICE))
    add = features["atomicAdd"]
    assert add.group == "atomic"
    assert add.signature.params == (Param(Type("int"), "inout"), Param(Type("int")))
    assert features["atomicAdd_block"].signature.result == Type("uint")
    frexp = features["frexpf"].signature
    assert frexp.params[1] == Param(Type("int"), "out")


def test_cuda_spells_single_precision_functions_as_their_crossgl_overload():
    glsl_names = {"abs", "frexp", "erf", "bitCount", "memoryBarrier"}
    features = by_name(cuda.parse(CUDA_MATH, CUDA_DEVICE, glsl_names))
    assert features["fabsf"].crossgl == features["fabs"].crossgl == "abs"
    assert features["frexpf"].crossgl == "frexp"
    # erff and erf are one CrossGL function, told apart by argument type.
    assert features["erff"].crossgl == features["erf"].crossgl
    assert features["__popc"].crossgl == "bitCount"
    assert features["__threadfence"].crossgl == "memoryBarrier"
    # A rename applies only when GLSL has the target, so a CrossGL spelling is never invented.
    assert by_name(cuda.parse(CUDA_MATH, CUDA_DEVICE))["__popc"].crossgl == "__popc"


HIP_MATH = """
#define __DEVICE__ static __device__ inline __attribute__((always_inline))
#define __RETURN_TYPE bool
__DEVICE__
void __suppress_unused_warning(bool b){};
__DEVICE__
uint64_t __make_mantissa(const char *__tagp __attribute__((nonnull))) { return 0; }
__DEVICE__
double erf(double __x) { return __ocml_erf_f64(__x); }
__DEVICE__
float erff(float __x) { return __ocml_erf_f32(__x); }
__DEVICE__
float frexpf(float __x, int *__nptr) { return __builtin_frexpf(__x, __nptr); }
__DEVICE__
float nanf(const char *__tagp) { return __builtin_nanf(""); }
__DEVICE__
__RETURN_TYPE __isnanf(float __x) { return __builtin_isnan(__x); }
#if defined OCML_BASIC_ROUNDED_OPERATIONS
__DEVICE__
float __fadd_rd(float __x, float __y) { return __ocml_add_rtn_f32(__x, __y); }
__DEVICE__
float __fadd_rn(float __x, float __y) { return __ocml_add_rte_f32(__x, __y); }
#else
__DEVICE__
float __fadd_rn(float __x, float __y) { return __x + __y; }
#endif
template <class T> __DEVICE__ T min(T __arg1, T __arg2) { return __arg1 < __arg2 ? __arg1 : __arg2; }
__DEVICE__ int min(int __arg1, int __arg2) { return __arg1 < __arg2 ? __arg1 : __arg2; }
"""

HIP_ATOMIC = """
template<typename T, typename F> struct Cond_t<true, T, F> { using type = T; };
__device__
inline
int atomicAdd(int* address, int val) {
    return __hip_atomic_fetch_add(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}
__device__
inline
float atomicAdd(float* address, float val) { return unsafeAtomicAdd(address, val); }
"""

HIP_DEVICE = """
static inline __device__ void printf(const char* format, All... all) {}
__device__ static inline unsigned int __popc(unsigned int input) {
    return __builtin_popcount(input);
}
__device__ static int32_t __fns32(uint64_t mask, uint32_t base, int32_t offset) { return 0; }
__device__
inline
static void __syncthreads() { __barrier(__CLK_LOCAL_MEM_FENCE); }
"""


def test_hip_reads_the_math_atomic_and_device_headers():
    glsl_names = {"erf", "frexp", "isnan", "bitCount", "barrier", "min"}
    features = by_name(hip.parse(HIP_MATH, HIP_ATOMIC, HIP_DEVICE, glsl_names))
    assert sorted(features) == [
        "__fadd_rn",
        "__fns32",
        "__isnanf",
        "__popc",
        "__syncthreads",
        "atomicAdd",
        "erf",
        "erff",
        "frexpf",
        "min",
        "nanf",
    ]
    assert {f.group for f in features.values()} == {"math", "atomic", "device"}
    # Only the #else branch of OCML_BASIC_ROUNDED_OPERATIONS is compiled by default, so __fadd_rd is left out.
    assert "__fadd_rd" not in features
    # The first definition is the one probed: the int overloads of atomicAdd and min.
    add = features["atomicAdd"].signature
    assert add.params == (Param(Type("int"), "inout"), Param(Type("int")))
    assert features["min"].signature.result == Type("int")
    assert features["frexpf"].signature.params[1] == Param(Type("int"), "out")
    # A C string takes a string constant.
    assert features["nanf"].signature.params == (Param(STRING, "const"),)
    assert features["__isnanf"].signature.result == Type("bool")
    assert features["__fns32"].signature.params[0].type == Type("uint64")
    assert features["__fns32"].signature.params[1].type == Type("uint")
    assert features["erff"].crossgl == features["erf"].crossgl == "erf"
    assert features["__popc"].crossgl == "bitCount"
    assert features["__syncthreads"].crossgl == "barrier"


def test_hip_default_compile_drops_opt_in_branches_with_their_nested_blocks():
    header = "\n".join(
        [
            "int a();",
            "#if defined(OCML_BASIC_ROUNDED_OPERATIONS)",
            "#ifdef X",
            "int b();",
            "#endif",
            "int c();",
            "#endif",
            "#if defined OCML_BASIC_ROUNDED_OPERATIONS",
            "int d();",
            "#else",
            "int e();",
            "#endif",
            "#ifdef Y",
            "int f();",
            "#endif",
        ]
    )
    kept = [line for line in hip.default_compile(header).splitlines() if "(" in line]
    assert kept == ["int a();", "int e();", "int f();"]

from tools.coverage import validators
from tools.coverage.features import cuda, glsl, hlsl, metal, spirv, wgsl
from tools.coverage.model import Param, Type

GLSL_EXCERPT = """
== Common Functions

=== Abs
[source,glsl]
----
genFType abs(genFType x)
genIType abs(genIType x)
ifdef::ESSL[]
highp genFType esOnly(genFType x)
endif::ESSL[]
----

=== Mul Extended
[source,glsl]
----
void umulExtended(highp genUType x,
                  highp genUType y,
                  out highp genUType msb,
                  out highp genUType lsb)
----

=== Atan
[source,glsl]
----
genFType atan(genFType y, genFType x)
genFType atan(genFType y_over_x)
----

=== Min
[source,glsl]
----
genFType min(genFType x, genFType y)
----

=== FTransform
[source,glsl]
----
vec4 ftransform()
----

== Texture Functions

[source,glsl]
----
gvec4 texture(gsampler2D sampler, vec2 P [, float bias] )
----

[source,glsl]
----
float Example(float x)
{
    return x;
}
----

== Fragment Processing Functions

[source,glsl]
----
genFType dFdx(genFType p)
----
"""


def by_name(features):
    return {feature.name: feature for feature in features}


def test_glsl_reads_desktop_prototypes_only():
    features = by_name(glsl.parse(GLSL_EXCERPT))
    assert sorted(features) == [
        "abs",
        "atan",
        "dFdx",
        "min",
        "texture",
        "umulExtended",
    ]


def test_glsl_makes_generic_types_concrete():
    features = by_name(glsl.parse(GLSL_EXCERPT))
    abs_ = features["abs"].signature
    assert abs_.result == Type("float", (3,))
    assert features["umulExtended"].signature.params[2].direction == "out"
    assert features["umulExtended"].signature.params[0].type == Type("uint", (3,))


def test_glsl_records_section_stage_and_opaque_types():
    features = by_name(glsl.parse(GLSL_EXCERPT))
    assert features["dFdx"].stage == "fragment"
    assert features["abs"].group == "common-functions"
    assert features["texture"].signature.unsupported == "gvec4"
    assert len(features["texture"].signature.params) == 2


GRAMMAR = """{"instructions": [
 {"opname": "FAbs", "operands": [{"kind": "IdRef"}]},
 {"opname": "SAbs", "operands": [{"kind": "IdRef"}]},
 {"opname": "Atan", "operands": [{"kind": "IdRef"}]},
 {"opname": "Atan2", "operands": [{"kind": "IdRef"}, {"kind": "IdRef"}]},
 {"opname": "UMulExtended", "operands": [{"kind": "IdRef"}]},
 {"opname": "NMin", "operands": [{"kind": "IdRef"}, {"kind": "IdRef"}]}
]}"""


def test_spirv_maps_instructions_to_glsl_overloads():
    features = by_name(spirv.parse(GRAMMAR, GLSL_EXCERPT))
    assert features["FAbs"].crossgl == "abs"
    assert features["FAbs"].signature.result.scalar == "float"
    assert features["SAbs"].signature.params[0].type == Type("int", (3,))
    assert len(features["Atan"].signature.params) == 1
    assert len(features["Atan2"].signature.params) == 2


def test_spirv_explains_unreachable_and_unspelled_instructions():
    features = by_name(spirv.parse(GRAMMAR, GLSL_EXCERPT))
    assert features["NMin"].signature.params[1].type == Type("float", (3,))
    assert features["NMin"].crossgl == ""
    assert "NaN" in features["NMin"].note
    assert features["UMulExtended"].signature is None
    assert features["UMulExtended"].note


HLSL_TABLE = """
namespace Intrinsics {
double<> [[rn]] asdouble(in $match<0, 1> uint<> x, in $match<0, 2> uint<> y) : reinterpret_fuse_double;
$type1 [[rn]] lerp(in float_like<> a, in $type1 b, in $type1 s);
$type1 [[rn,unsigned_op=umad]] mad(in numeric<> a, in $type1 b, in $type1 c);
void [[]] InterlockedAdd(ref int32_only result, in uint value);
$type1 [[rn]] ddx(in float_like<> x);
$match<1, 1> any<c, r> [[rn]] transpose(in any<r, c> x);
uint   [[rn,min_sm=6.10]] GetGroupWaveIndex();
} namespace

namespace VkIntrinsics {
u64 [[]] ReadClock(in uint scope);
} namespace

namespace Texture2DMethods {
float [[ro]] CalculateLevelOfDetail(in any_sampler s, in float<2> x) : tex2d_t_calc_lod;
} namespace
"""


def test_hlsl_reads_intrinsics_and_methods_from_the_dxc_table():
    features = by_name(hlsl.parse(HLSL_TABLE, {"mix", "fma"}))
    assert "ReadClock" not in features
    assert features["Texture2D.CalculateLevelOfDetail"].category == "method"
    assert features["Texture2D.CalculateLevelOfDetail"].signature is None
    assert features["ddx"].stage == "fragment"
    assert features["GetGroupWaveIndex"].signature is None
    assert features["GetGroupWaveIndex"].note == "needs shader model 6.10"


def test_hlsl_resolves_generic_layouts_and_matched_types():
    features = by_name(hlsl.parse(HLSL_TABLE, {"mix", "fma"}))
    lerp = features["lerp"].signature
    assert lerp.result == Type("float", (2,))
    assert [p.type for p in lerp.params] == [Type("float", (2,))] * 3
    asdouble = features["asdouble"].signature
    assert asdouble.result == Type("double", (2,))
    assert asdouble.params[1].type == Type("uint", (2,))
    assert features["InterlockedAdd"].signature.params[0].direction == "inout"
    assert features["transpose"].signature.params[0].type == Type("float", (2, 2))


def test_hlsl_takes_crossgl_spellings_from_the_template_layer():
    features = by_name(hlsl.parse(HLSL_TABLE, {"mix", "fma"}))
    assert features["lerp"].crossgl == "mix"
    assert features["mad"].crossgl == "fma"
    # No GLSL atomicAdd in this excerpt, so the probe keeps the HLSL spelling.
    assert features["InterlockedAdd"].crossgl == "InterlockedAdd"


WGSL_DEF = """
match fa_f32_f16: fa | f32 | f16
match iu32: i32 | u32
match fiu32: f32 | i32 | u32

////////////////////////////////////////////////////////////////////////////////
// Builtin Functions                                                          //
////////////////////////////////////////////////////////////////////////////////
@must_use @const implicit(T: fa_f32_f16)
  fn atan2(T, T) -> T
@must_use @const implicit(T: fa_f32_f16, N: num)
  fn atan2(vec<N, T>, vec<N, T>) -> vec<N, T>
@must_use @const implicit(U: fiu32)
  fn bitcast<T: fiu32>(U) -> T
@must_use @stage("fragment")
  fn dpdx(f32) -> f32

//////////////////////////////////////////
// atomics
//////////////////////////////////////////
@stage("fragment", "compute") implicit(T: iu32, S: workgroup_or_storage)
  fn atomicAdd(ptr<S, atomic<T>, read_write>, T) -> T

//////////////////////////////////////////
// subgroup matrix
//////////////////////////////////////////
@must_use fn subgroupMatrixLoad(f32) -> f32

////////////////////////////////////////////////////////////////////////////////
// Value constructors                                                         //
////////////////////////////////////////////////////////////////////////////////
@must_use @const fn i32() -> i32
"""


def test_wgsl_reads_builtin_functions_only():
    features = by_name(wgsl.parse(WGSL_DEF, {"atan", "dFdx", "atomicAdd"}))
    assert sorted(features) == ["atan2", "atomicAdd", "bitcast", "dpdx"]
    assert features["atomicAdd"].group == "atomics"
    assert features["dpdx"].stage == "fragment"


def test_wgsl_makes_templates_concrete():
    features = by_name(wgsl.parse(WGSL_DEF, {"atan", "dFdx", "atomicAdd"}))
    assert features["atan2"].signature.result == Type("float", (3,))
    atomic = features["atomicAdd"].signature
    assert atomic.params[0] == Param(Type("int"), "inout")
    # An explicit template argument takes the unsigned member, so the cast changes the type.
    assert features["bitcast"].signature.result == Type("uint")
    assert features["bitcast"].spelling == "bitcast<u32>"


def test_wgsl_takes_crossgl_spellings_from_the_template_layer():
    features = by_name(wgsl.parse(WGSL_DEF, {"atan", "dFdx", "atomicAdd"}))
    assert features["atan2"].crossgl == "atan"
    assert features["dpdx"].crossgl == "dFdx"


METAL_LIST = """# Generated by python -m tools.coverage.features.metal; do not edit.
# toolchain: Apple metal version 1.0 (test)
# language: Metal Shading Language 9.9
metal_math fma float3(float3, float3, float3)
metal_math frexp float3(float3, out int3)
metal_matrix determinant float(float3x3)
metal_graphics dfdx float3(float3)
metal_atomic atomic_fetch_add_explicit -
"""


def test_metal_reads_the_generated_list():
    features = by_name(metal.parse(METAL_LIST, {"fma", "dFdx", "frexp", "determinant"}))
    assert features["frexp"].signature.params[1] == Param(Type("int", (3,)), "out")
    assert features["determinant"].signature.params[0].type == Type("float", (3, 3))
    assert features["dfdx"].stage == "fragment"
    assert features["dfdx"].crossgl == "dFdx"
    assert features["atomic_fetch_add_explicit"].signature is None
    assert (
        features["atomic_fetch_add_explicit"].note
        == "no overload takes only value types"
    )


def test_metal_list_records_its_toolchain():
    assert metal.source(METAL_LIST)["describes"] == (
        "Metal Shading Language 9.9 standard library, from Apple metal version 1.0 (test)"
    )
    # The committed list says which toolchain produced it.
    assert metal.LIST.read_text().count("# toolchain: Apple metal version") == 1
    # Probes compile against the language version the committed list was read from.
    version = validators.METAL_STANDARD.removeprefix("metal")
    assert f"# language: Metal Shading Language {version}\n" in metal.LIST.read_text()


CUDA_MATH = """
#define __DEVICE__ static __device__ __forceinline__
__DEVICE__ double erf(double __a) { return __nv_erf(__a); }
__DEVICE__ float erff(float __a) { return __nv_erff(__a); }
__DEVICE__ double fabs(double __a) { return __nv_fabs(__a); }
__DEVICE__ float fabsf(float __a) { return __nv_fabsf(__a); }
__DEVICE__ double frexp(double __a, int *__b) { return __nv_frexp(__a, __b); }
__DEVICE__ float frexpf(float __a, int *__b) { return __nv_frexpf(__a, __b); }
__DEVICE__ long labs(long __a) { return llabs(__a); }
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
__DEVICE_VOID__ void __threadfence(void) { __nvvm_membar_gl(); }
"""


def test_cuda_reads_device_functions_and_leaves_out_the_rest():
    features = by_name(cuda.parse(CUDA_MATH, CUDA_DEVICE, {"abs", "frexp"}))
    assert sorted(features) == [
        "__popc",
        "__threadfence",
        "atomicAdd",
        "atomicAdd_block",
        "erf",
        "erff",
        "fabs",
        "fabsf",
        "frexp",
        "frexpf",
        "labs",
    ]
    assert features["labs"].signature.params[0].type == Type.other("long")
    assert features["__threadfence"].group == "intrinsics"
    assert features["fabsf"].group == "math"


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

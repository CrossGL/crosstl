"""The Slang feature list, read from the core module sources."""

import pytest

from crosstl.translator.semantics.builtin_function_names import BUILTIN_FUNCTION_NAMES
from crosstl.translator.semantics.builtin_names import is_crossgl_gl_builtin
from tools.coverage import probes, spellings
from tools.coverage.features import slang
from tools.coverage.model import (
    COOP_COLUMNS,
    STRING,
    Param,
    Signature,
    Type,
    coop_vector,
    tuple_of,
)


def by_name(features):
    return {feature.name: feature for feature in features}


SLANG_HLSL = """
//@hidden:
__generic<T : __BuiltinFloatingPointType>
T __hiddenHelper(T x) { return x; }

//@public:
/// Sine.
__generic<T : __BuiltinFloatingPointType>
[__readNone]
[require(cpp_cuda_glsl_hlsl_metal_spirv_wgsl_llvm, sm_4_0_version)]
T sin(T x)
{
    __target_switch
    {
    case hlsl: __intrinsic_asm "sin";
    case spirv: return spirv_asm { OpExtInst $$T result glsl450 Sin $x };
    }
}

__generic<T : __BuiltinFloatingPointType, let N : int>
[require(cpp_cuda_glsl_hlsl_metal_spirv_wgsl_llvm, sm_4_0_version)]
vector<T, N> sin(vector<T, N> x)
{
    VECTOR_MAP_UNARY(T, N, sin, x);
}

[ForceInline]
__func_extension<T : __BuiltinFloatingPointType, let N : int>
fwd_diff(CoopVec<T, N>::__subscript::get)(
    DifferentialPair<CoopVec<T, N>> self,
    int index) -> DifferentialPair<T>
{
    return diffPair(self.p[index], reinterpret<T.Differential, T>(self.d[index]));
}

${
struct SlangAtomicOperationInfo
{
    const char* slangCallSuffix;
    const char* internalCallSuffix;
    const char* interface;
};
SlangAtomicOperationInfo slangAtomicOperationInfo[2] = {
    { "Add", "add", "IArithmeticAtomicable" },
    { "Exchange", "exchange", "IAtomicable" },
};
for (SlangAtomicOperationInfo atomicOp : slangAtomicOperationInfo)
{
$}
[ForceInline]
[require(cuda_glsl_hlsl_metal_spirv, atomic_glsl_hlsl_cuda_metal)]
void Interlocked$(atomicOp.slangCallSuffix)<T:$(atomicOp.interface)>(__ref T dest, T value)
{
    __atomic_$(atomicOp.internalCallSuffix)(dest, value);
}
${
}
$}

[require(cpp_cuda_glsl_hlsl_spirv, fragment)]
void clip(float x);

[require(hlsl, sm_5_0)]
void ProcessTriTessFactorsAvg(in float3 RawEdgeFactors, in float InsideScale, out float3 Rounded);

[require(cuda_glsl_hlsl_spirv, raytracing_anyhit_closesthit_intersection_miss)]
float3 WorldRayOrigin();

[require(cuda_glsl_hlsl_spirv, raytracing_anyhit_closesthit_intersection_miss)]
float3x4 ObjectToWorld3x4();

/// Alias for ObjectToWorld3x4.
[NonUniformReturn] float3x4 ObjectToWorld() { return ObjectToWorld3x4(); }

__generic<A>
[require(glsl_hlsl_spirv, raytracing_intersection)]
bool ReportHit(float tHit, uint hitKind, A attributes);

[require(glsl_spirv, GLSL_430_SPIRV_1_0_compute)]
[require(glsl_spirv, meshshading)]
uint3 WorkgroupCount()
{
    return uint3(0);
}

[require(optix_multilevel_traversal, raytracing_stages)]
uint64_t GetTraversableChild(uint64_t handle);

__generic<T : __BuiltinFloatingPointType>
T lerp(T x, T y, T s);

struct Hidden
{
    float member(float x) { return x; }
};
"""

SLANG_CORE = """
//@public:
__generic<T, U>
[__unsafeForceInlineEarly]
T bit_cast(U value);

__intrinsic_op($(kIROp_StaticAssert))
void static_assert(constexpr bool condition, NativeString errorMessage);
"""


def test_slang_reads_the_public_global_functions_of_the_core_module():
    glsl_names = {"sin", "mix"}
    features = by_name(slang.parse(SLANG_CORE, SLANG_HLSL, glsl_names))
    assert sorted(features) == [
        "GetTraversableChild",
        "InterlockedAdd",
        "InterlockedExchange",
        "ObjectToWorld",
        "ObjectToWorld3x4",
        "ProcessTriTessFactorsAvg",
        "ReportHit",
        "WorkgroupCount",
        "WorldRayOrigin",
        "bit_cast",
        "clip",
        "lerp",
        "sin",
        "static_assert",
    ]
    # A function extension, such as the derivative of CoopVec's subscript, declares no function of its own.
    assert "fwd_diff" not in features
    assert features["bit_cast"].group == "core"
    assert features["sin"].group == "hlsl"
    # Generic functions are probed with two-component vectors when they have a vector overload.
    vec2 = Type("float", (2,))
    assert features["sin"].signature.result == vec2
    assert features["sin"].signature.params == (Param(vec2),)
    add = features["InterlockedAdd"].signature
    assert add.params == (Param(Type("int"), "inout"), Param(Type("int")))
    assert features["lerp"].crossgl == "mix"
    assert features["WorkgroupCount"].signature.result == Type("uint", (3,))
    assert features["static_assert"].signature.params == (
        Param(Type("bool"), "condition"),
        Param(STRING, "const"),
    )


def test_slang_spells_its_coarse_and_fine_fwidth_as_crossgl_does():
    glsl_names = {"fwidthCoarse", "fwidthFine"}
    assert spellings.spelling("slang", "fwidth_coarse", glsl_names) == "fwidthCoarse"
    assert spellings.spelling("slang", "fwidth_fine", glsl_names) == "fwidthFine"
    # HLSL has no coarse or fine fwidth, and Slang's additions leave its table alone.
    assert "fwidth_coarse" not in spellings.RENAMES["hlsl"]
    assert spellings.RENAMES["slang"]["lerp"] == "mix"


def test_slang_functions_crossgl_names_otherwise_take_crossgl_names():
    glsl_names = {"unpackHalf2x16", "interpolateAtCentroid"}
    assert spellings.spelling("slang", "unpackHalf2x16ToFloat", glsl_names) == (
        "unpackHalf2x16"
    )
    assert spellings.spelling("slang", "EvaluateAttributeAtCentroid", glsl_names) == (
        "interpolateAtCentroid"
    )
    # A name GLSL lacks applies without the GLSL feature list.
    assert (
        spellings.spelling("slang", "WavePrefixMax", set()) == "WavePrefixExclusiveMax"
    )
    assert (
        spellings.spelling("slang", "WaveGetLaneEqMask", set()) == "gl_SubgroupEqMask"
    )
    assert spellings.spelling("wgsl", "quadSwapX", set()) == "QuadReadAcrossX"
    # A half-precision unpack returns half vectors, which GLSL's do not, so it converts GLSL's result as Slang does.
    assert spellings.spelling("slang", "unpackHalf2x16ToHalf", glsl_names) == (
        "f16vec2(unpackHalf2x16({}))"
    )
    # Slang's pack_s8 calls packInt4x8, and its WaveMultiPrefixSum is the exclusive scan.
    assert spellings.spelling("slang", "packInt4x8", set()) == "pack_s8"
    assert spellings.spelling("slang", "unpackUint4x8ToUint16", set()) == "unpack_u8u16"
    assert spellings.spelling("slang", "WaveMultiPrefixExclusiveSum", set()) == (
        "WaveMultiPrefixSum"
    )
    # Slang writes an inclusive multi-prefix scan for HLSL as the exclusive scan combined with the lane's value.
    assert spellings.spelling("slang", "WaveMultiPrefixInclusiveSum", set()) == (
        "(WaveMultiPrefixSum({0}, {1}) + {0})"
    )
    assert spellings.spelling("slang", "getRealtimeClockLow", set()) == (
        "clockRealtime2x32EXT().x"
    )


def test_other_names_are_names_crossgl_defines():
    for language, names in spellings.OTHER_NAMES.items():
        for name, spelled in names.items():
            used = (
                spellings.named(spelled)
                if spellings.is_expression(spelled)
                else [spelled]
            )
            assert used, (language, name)
            for base in used:
                if base.startswith("gl_"):
                    assert is_crossgl_gl_builtin(base), (language, name)
                else:
                    assert base in BUILTIN_FUNCTION_NAMES, (language, name)


def test_expression_spellings_name_what_they_call():
    assert spellings.is_expression("WaveActiveBallot(true)")
    assert spellings.is_expression("gl_HitTriangleVertexPositionsEXT[{}]")
    assert not spellings.is_expression("gl_NumSubgroups")
    assert spellings.named("f16vec2(unpackHalf2x16({}))") == ["unpackHalf2x16"]
    assert spellings.named("ivec3(gl_WorkGroupSize)") == ["gl_WorkGroupSize"]
    assert spellings.named("(WaveMultiPrefixBitXor({0}, {1}) ^ {0})") == [
        "WaveMultiPrefixBitXor"
    ]
    assert spellings.named("clockRealtime2x32EXT().x") == ["clockRealtime2x32EXT"]


def test_slang_stages_targets_and_explicit_generic_arguments():
    features = by_name(slang.parse(SLANG_CORE, SLANG_HLSL))
    assert features["clip"].stage == "fragment"
    assert features["WorldRayOrigin"].stage == "closest-hit"
    assert features["ReportHit"].stage == "intersection"
    # An alias without a require attribute takes the stage of the Direct3D intrinsic it is named after.
    assert features["ObjectToWorld"].stage == "closest-hit"
    # One require alternative works in compute, so no other stage is needed.
    assert features["WorkgroupCount"].stage == "compute"
    # SPIR-V lacks it, so it compiles for HLSL, in the hull stage Direct3D allows it in.
    tess = features["ProcessTriTessFactorsAvg"]
    assert (tess.target, tess.stage, tess.note) == ("hlsl", "tessellation-control", "")
    assert tess.signature.params[2] == Param(Type("float", (3,)), "out")
    assert features["clip"].target == ""
    # An OptiX capability implies the CUDA target, which is not probed.
    traversable = features["GetTraversableChild"]
    assert (traversable.signature, traversable.target) == (None, "")
    assert traversable.note == "needs a target other than SPIR-V and HLSL"
    # No argument fixes bit_cast's result type, so the probe names it.
    assert features["bit_cast"].spelling == "bit_cast<int>"
    assert features["sin"].spelling == ""


def test_slang_types():
    generics = {"T": "float", "N": "size"}
    assert slang.slang_type("vector<T, N>", generics) == Type("float", (2,))
    assert slang.slang_type("matrix<T, 3, 4>", generics) == Type("float", (4, 3))
    assert slang.slang_type("float3x4", {}) == Type("float", (4, 3))
    assert slang.slang_type("uint32_t4", {}) == Type("uint", (4,))
    assert slang.slang_type("half", {}) == Type("half")
    assert slang.slang_type("int16_t4", {}) == Type("int16", (4,))
    assert slang.slang_type("vector<uint16_t, 3>", {}) == Type("uint16", (3,))
    assert slang.slang_type("half*", {}) == Type.other("half*")
    assert slang.slang_type("uint64_t", {}) == Type("uint64")
    assert slang.slang_type("int64_t2", {}) == Type("int64", (2,))
    assert slang.slang_type("int8_t4_packed", {}) == Type("int8x4_packed")
    # A cooperative vector keeps its element type and length; a generic length is 4.
    assert slang.slang_type("CoopVec<T, N>", generics) == coop_vector("float", 4)
    assert slang.slang_type("CoopVec<half, 2>", {}) == coop_vector("half", 2)
    assert slang.slang_type("CoopVec<float, 8>", {}) == Type.other("CoopVec<float, 8>")


SLANG_COOP = """
//@public:
enum CoopVecComponentType { Float32 };
CoopVec<T, N> coopVecLoad<let N : int, T : __BuiltinArithmeticType>(ByteAddressBuffer buffer, int32_t offset = 0);
CoopVec<T, N> coopVecLoad<let N : int, T : __BuiltinArithmeticType, L : IBufferDataLayout = DefaultDataLayout>(
    StructuredBuffer<T, L> buffer, int32_t offset = 0);
CoopVec<T, N> coopVecLoadGroupshared<let N : int, T : __BuiltinArithmeticType, let M : int>(
    __constref groupshared const T[M] data, int32_t offset = 0);
__generic<T : __BuiltinArithmeticType, let M : int, let K : int, U : __BuiltinArithmeticType>
CoopVec<T, M> coopVecMatMul(CoopVec<U, K> input, constexpr CoopVecComponentType inputInterpretation,
    RWByteAddressBuffer matrix, int32_t matrixOffset, constexpr bool transpose, constexpr uint matrixStride);
void coopVecReduceSumAccumulate<T : __BuiltinArithmeticType, let N : int>(
    CoopVec<T, N> v, RWByteAddressBuffer buffer, int32_t offset);
__generic<T : __BuiltinArithmeticType, let M : int, let PackedK : int, U : __BuiltinArithmeticType>
CoopVec<T, M> coopVecMatMulPacked(CoopVec<U, PackedK> input, constexpr CoopVecComponentType inputInterpretation,
    constexpr int k, RWByteAddressBuffer matrix, int32_t matrixOffset);
"""


def test_slang_cooperative_vector_functions_spell_their_generic_arguments():
    features = by_name(slang.parse("", SLANG_COOP))
    load = features["coopVecLoad"]
    assert load.signature.result == coop_vector("float", 4)
    assert load.signature.params[0] == Param(Type.other("ByteAddressBuffer"))
    # slangc infers no generic argument from the result, so the probe passes them all, and CrossGL, which calls the
    # function by Slang's name, does too.
    assert load.spelling == load.crossgl == "coopVecLoad<4, float>"
    matmul = features["coopVecMatMul"]
    assert matmul.spelling == "coopVecMatMul<float, 4, 4, float>"
    assert matmul.signature.params[:2] == (
        Param(coop_vector("float", 4)),
        Param(Type.other("CoopVecComponentType"), "const"),
    )
    # A read-only workgroup array is a workgroup parameter of the generic length.
    shared = features["coopVecLoadGroupshared"]
    assert shared.signature.params[0] == Param(Type("float", count=4), "workgroup")
    assert shared.spelling == "coopVecLoadGroupshared<4, float, 4>"
    # The arguments fix every generic parameter of a function that returns nothing.
    assert features["coopVecReduceSumAccumulate"].spelling == ""
    assert (
        features["coopVecReduceSumAccumulate"].crossgl == "coopVecReduceSumAccumulate"
    )
    # A packed multiply's column count k is the input vector's length.
    packed = features["coopVecMatMulPacked"].signature.params
    assert packed[2] == Param(COOP_COLUMNS, "const")


def test_slang_strings_are_literals_and_parameter_packs_are_empty():
    text = """
//@public:
int getStringHash(String string);

[require(cpp_cuda_glsl_hlsl_spirv_llvm, printf)]
void printf<each T>(NativeString format, expand each T args);

__generic<each T : __BuiltinIntegerType>
[require(cuda_glsl_hlsl_spirv, raytracing_intersection)]
bool ReportHitOptix(float tHit, uint hitKind, expand each T attribs);
"""
    features = by_name(slang.parse(text, ""))
    constant = Param(STRING, "const")
    assert features["getStringHash"].signature.params == (constant,)
    assert features["printf"].signature.params == (constant,)
    assert features["ReportHitOptix"].signature.params == (
        Param(Type("float")),
        Param(Type("uint")),
    )
    assert probes.build(features["printf"], "slang").source.count('printf("");') == 1


def test_slang_type_packs_hold_one_int_and_tuples_are_read_through_their_first_member():
    text = """
//@public:
__intrinsic_op($(kIROp_MakeTuple))
Tuple<T> makeTuple<each T>(T v);

Tuple<T, U> concat<each T, each U>(Tuple<T> t, Tuple<U> u)
{
    return makeTuple(expand each t, expand each u);
}
"""
    features = by_name(slang.parse(text, ""))
    make = features["makeTuple"].signature
    assert make == Signature(tuple_of("int", 1), (Param(Type("int")),))
    concat = features["concat"].signature
    assert concat.result == tuple_of("int", 2)
    assert concat.params == (Param(tuple_of("int", 1)), Param(tuple_of("int", 1)))
    native = probes.build(features["concat"], "slang").source
    assert "Tuple<int> a0 = Tuple<int>(iin[i + 0u]);" in native
    assert "int r = concat(a0, a1)._0;" in native
    crossgl = probes.build(features["makeTuple"], "crossgl").source
    assert "int r = makeTuple(a0)._0;" in crossgl


def test_slang_pointers_are_uniforms_and_free_value_generics_are_spelled():
    text = """
//@public:
[ForceInline]
T loadAligned<int alignment, T>(T* ptr)
{
    return __load_aligned(ptr, __align_attr(alignment));
}

[require(spirv)]
void InterlockedAddF16x2(half2* dest, half2 value, out half2 originalValue);

[require(spirv, cooperative_vector, vk_mem_model)]
CoopVec<T, N> coopVecLoadCoherent<let N : int, T : __BuiltinArithmeticType>(T* buffer, int32_t byteOffset16ByteAligned = 0, constexpr MemoryScope memoryScope = MemoryScope::Device);

internal void enableVMMDeviceScopeCapabilityIfNeeded(constexpr MemoryScope memoryScope);
"""
    features = by_name(slang.parse(text, ""))
    assert "enableVMMDeviceScopeCapabilityIfNeeded" not in features
    load = features["loadAligned"]
    assert load.signature.params == (Param(Type("int"), "pointer"),)
    assert load.spelling == load.crossgl == "loadAligned<4>"
    assert features["InterlockedAddF16x2"].signature.params[0] == Param(
        Type("half", (2,)), "pointer"
    )
    native = probes.build(load, "slang").source
    assert "uniform int* a0;" in native
    assert "int r = loadAligned<4>(a0);" in native
    assert "uniform int* a0;" in probes.build(load, "crossgl").source
    coherent = probes.build(features["coopVecLoadCoherent"], "slang").source
    assert "uniform float* a0;" in coherent
    assert "coopVecLoadCoherent<4, float>(a0, a1, MemoryScope.Device)" in coherent
    with pytest.raises(probes.ProbeError, match="device pointer"):
        probes.build(load, "hlsl")


def test_slang_floating_point_cooperative_elements_are_probed_as_float():
    text = """
//@public:
__intrinsic_op($(kIROp_FloatCast))
T floatCast<T:IFloatingPointCoopElement, U:IFloatingPointCoopElement>(U val);
"""
    cast = by_name(slang.parse(text, ""))["floatCast"]
    assert cast.signature == Signature(Type("float"), (Param(Type("float")),))
    # No argument fixes the result type, so the probe names it.
    assert cast.spelling == "floatCast<float>"
    assert "float r = floatCast<float>(a0);" in probes.build(cast, "slang").source


def test_slang_generic_type_parameters_nothing_fixes_are_untested():
    text = """
//@public:
[ForceInline]
bool hasTrait<T : ITrait>()
{
    return false;
}
"""
    query = by_name(slang.parse(text, ""))["hasTrait"]
    assert (query.signature, query.spelling, query.target) == (None, "", "")
    assert query.note == "no probe type for the generic parameter T : ITrait"
    with pytest.raises(
        probes.ProbeError, match="no probe type for the generic parameter T"
    ):
        probes.build(query, "slang")


def test_slang_descriptor_functions_are_probed_with_a_float_buffer():
    text = """
//@public:
[ForceInline]
bool usesNativeSpirvDescriptorHandle<T : IOpaqueDescriptor>()
{
    return false;
}

T getDescriptorFromExplicitHeapIndex<T:IOpaqueDescriptor>(uint2 handle);

T defaultGetDescriptorFromHandle<T:IOpaqueDescriptor>(DescriptorHandle<T> handleValue, constexpr BindlessDescriptorOptions bindlessOptions = BindlessDescriptorOptions.VkMutable);

__intrinsic_op($(kIROp_NonUniformResourceIndex))
DescriptorHandle<T> nonuniform<T:IOpaqueDescriptor>(DescriptorHandle<T> ptr);
"""
    features = by_name(slang.parse(text, ""))
    handle = Type.other("DescriptorHandle<StructuredBuffer<float>>")
    # slangc infers no descriptor type from the result or from nothing, so the probe names it.
    query = features["usesNativeSpirvDescriptorHandle"]
    assert query.signature == Signature(Type("bool"), ())
    assert query.spelling == "usesNativeSpirvDescriptorHandle<StructuredBuffer<float>>"
    heap = features["getDescriptorFromExplicitHeapIndex"]
    assert (
        heap.spelling == "getDescriptorFromExplicitHeapIndex<StructuredBuffer<float>>"
    )
    source = probes.build(heap, "slang").source
    assert (
        "float r = getDescriptorFromExplicitHeapIndex<StructuredBuffer<float>>(a0)[0];"
        in source
    )
    # A handle is a uniform the probe declares, and a descriptor result is read through its first element.
    default = features["defaultGetDescriptorFromHandle"]
    assert default.signature.params[0].type == handle
    source = probes.build(default, "slang").source
    assert "uniform DescriptorHandle<StructuredBuffer<float>> u0;" in source
    assert (
        "float r = defaultGetDescriptorFromHandle(u0, BindlessDescriptorOptions.None)[0];"
        in source
    )
    assert features["nonuniform"].signature == Signature(handle, (Param(handle),))
    source = probes.build(features["nonuniform"], "crossgl").source
    assert "uniform DescriptorHandle<StructuredBuffer<float>> u0;" in source
    assert "float r = nonuniform(u0)[0];" in source

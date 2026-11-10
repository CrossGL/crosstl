"""The HLSL feature list read from DXC's intrinsic table."""

from tools.coverage import validators
from tools.coverage.features import hlsl
from tools.coverage.model import VOID, Param, Signature, Type


def by_name(features):
    return {feature.name: feature for feature in features}


HLSL_TABLE = """
namespace Intrinsics {
double<> [[rn]] asdouble(in $match<0, 1> uint<> x, in $match<0, 2> uint<> y) : reinterpret_fuse_double;
int16_t<> [[rn]] asint16(in $match<0,1> numeric16_only<> x) : reinterpret_int16;
float [[rn]] dot2add(in float16_t<2> a, in $type1 b, in float c);
$type1 [[rn]] lerp(in float_like<> a, in $type1 b, in $type1 s);
$type1 [[rn,unsigned_op=umad]] mad(in numeric<> a, in $type1 b, in $type1 c);
void [[]] InterlockedAdd(ref int32_only result, in uint value);
void [[]] InterlockedAdd(ref int64_only result, in u64 value);
int<4> [[rn]] unpack_s8s32(in p32i8 pk);
p32u8 [[rn]] pack_u8(in any_int16or32<4> v);
$type1 [[rn]] ddx(in float_like<> x);
$type1 [[rn]] GetAttributeAtVertex(in numeric<> value, in uint VertexID);
float<3> [[rn]] WorldRayOrigin();
void [[]] IgnoreHit();
uint [[]] GetRemainingRecursionLevels();
void [[]]   abort();
void [[]] source_mark();
void [[]] printf(in string Format, ...);
void [[]] Barrier(in uint MemoryTypeFlags, in uint SemanticFlags);
$type1 [[rn]] NonUniformResourceIndex(in any<> index) : nonuniform_resource_index;
$match<1, 1> any<c, r> [[rn]] transpose(in any<r, c> x);
uint   [[rn,min_sm=6.10]] GetGroupWaveIndex();
float_like<4> [[ro]] tex2D(in sampler2d s, in float_like<2> x) : tex2d;
float_like<4> [[ro]] texCUBElod(in sampler_cube s, in float_like<4> x) : texcube_lod;
} namespace

namespace VkIntrinsics {
u64 [[]] ReadClock(in uint scope);
} namespace

namespace Texture2DMethods {
float [[ro]] CalculateLevelOfDetail(in any_sampler s, in float<2> x) : tex2d_t_calc_lod;
$match<0, -1> void<4> [[ro]] GatherCmp(in sampler_cmp s, in float<2> x, in float compareValue) : tex2d_t_gather_comp;
$classT [[]] Load(in int<3> x, in int<2> o, out uint_only status) : tex2d_t_load_o_s;
$classT [[ro]] Load(in int<3> x) : tex2d_t_load;
$match<0, -1> void<4> [[ro]] GatherRaw(in sampler s, in float<2> x);
} namespace

namespace Texture2DArrayMethods {
$classT [[ro]] SampleGrad(in sampler s, in float<3> x, in $match<2, 2> float<2> ddx, in $match<2, 2> float<2> ddy) : tex2d_t_dd_array;
} namespace

namespace TextureCUBEArrayMethods {
float_like [[ro]] SampleCmpGrad(in sampler_cmp s, in float<4> x, in float compareValue, in $match<2, 2> float<3> ddx, in $match<2, 2> float<3> ddy) : texcube_t_comp_dd_array;
} namespace

namespace FeedbackTexture2DMethods {
void [[]] WriteSamplerFeedback(in Texture2D t, in sampler s, in float<2> x);
void [[]] WriteSamplerFeedbackLevel(in Texture2D t, in sampler s, in float<2> x, in float lod);
} namespace

namespace ByteAddressBufferMethods {
$funcT [[ro]] Load(in uint byteOffset) : byteaddress_load;
} namespace

namespace RWByteAddressBufferMethods {
void [[]] InterlockedExchange64(in uint byteOffset, in any_int64 value, out any_int64 original);
} namespace

namespace DxHitObjectMethods {
    DxHitObject [[static,class_prefix,min_sm=6.9]] MakeNop();
    bool [[rn,class_prefix,min_sm=6.9]] IsMiss();
} namespace

namespace NodeOutputMethods {
    bool [[]] IsValid();
} namespace

namespace FutureObjectMethods {
    bool [[]] IsValid();
} namespace

namespace StreamMethods {
void [[]] Append(in $match<-1, 1> void x) : stream_append;
void [[]] RestartStrip() : stream_restart;
} namespace

namespace DxIntrinsics {
    void [[min_sm=6.9]] MaybeReorderThread(in uint CoherenceHint, in uint NumCoherenceHintBitsFromLSB);
} namespace

namespace VkSubpassInputMethods {
$classT [[]] SubpassLoad() : subpassinput_load;
} namespace
"""


def test_hlsl_reads_intrinsics_and_methods_from_the_dxc_table():
    features = by_name(hlsl.parse(HLSL_TABLE, {"mix", "fma"}))
    assert "ReadClock" not in features
    assert "VkSubpassInput.SubpassLoad" not in features
    # DXC rejects every call to the shader model 2 and 3 sampling functions.
    assert "tex2D" not in features and "texCUBElod" not in features
    # Nor does it compile abort or source_mark for any target.
    assert "abort" not in features and "source_mark" not in features
    # A work graph's recursion depth exists only in its node shaders.
    assert features["GetRemainingRecursionLevels"].stage == "node"
    assert features["Texture2D.CalculateLevelOfDetail"].category == "method"
    assert features["ddx"].stage == "fragment"
    assert features["GetGroupWaveIndex"].signature is None
    assert features["GetGroupWaveIndex"].note == "needs shader model 6.10"
    # The probes compile to DXIL, where DXC has no printf.
    assert features["printf"].signature is None
    assert features["printf"].note.startswith("DXC compiles printf only to SPIR-V")
    # Ray-tracing intrinsics run in a closest-hit shader unless another stage alone has them.
    assert features["WorldRayOrigin"].stage == "closest-hit"
    assert features["IgnoreHit"].stage == "any-hit"
    # The probes compile for shader model 6.9, so its intrinsics are probed in the stage they belong to.
    reorder = features["dx::MaybeReorderThread"]
    assert (reorder.category, reorder.stage, reorder.note) == (
        "function",
        "ray-generation",
        "",
    )
    assert [p.type for p in reorder.signature.params] == [Type("uint")] * 2


def test_hlsl_methods_take_the_element_type_of_their_object():
    features = by_name(hlsl.parse(HLSL_TABLE))
    lod = features["Texture2D.CalculateLevelOfDetail"]
    # Implicit derivatives need a fragment shader, or a compute group of at least four threads.
    assert lod.stage == "fragment"
    assert lod.signature.params[0].type == Type.other("sampler")
    gather = features["Texture2D.GatherCmp"].signature
    assert gather.result == Type("float", (4,))
    assert gather.params[0].type == Type.other("sampler_cmp")
    # The overload without an offset or the tiled-resource status comes first.
    load = features["Texture2D.Load"]
    assert load.signature == Signature(Type("float", (4,)), (Param(Type("int", (3,))),))
    assert (load.group, load.spelling, load.crossgl) == ("Texture2D", "Load", "Load")
    assert features["ByteAddressBuffer.Load"].signature.result == Type("uint")


def test_hlsl_matched_types_keep_a_size_the_table_spells_out():
    features = by_name(hlsl.parse(HLSL_TABLE))
    # Array layers take no gradient: Texture2DArray.SampleGrad has float3 coordinates and float2 gradients.
    grad = features["Texture2DArray.SampleGrad"].signature
    assert [p.type for p in grad.params[1:]] == [
        Type("float", (3,)),
        Type("float", (2,)),
        Type("float", (2,)),
    ]
    compare = features["TextureCUBEArray.SampleCmpGrad"].signature
    assert [p.type for p in compare.params[1:]] == [
        Type("float", (4,)),
        Type("float"),
        Type("float", (3,)),
        Type("float", (3,)),
    ]


def test_hlsl_gather_raw_returns_the_raw_bits_of_a_uint_texture():
    features = by_name(hlsl.parse(HLSL_TABLE))
    assert features["Texture2D.GatherRaw"].signature.result == Type("uint", (4,))
    assert features["Texture2D.GatherCmp"].signature.result == Type("float", (4,))


def test_hlsl_feedback_methods_take_the_texture_they_record():
    features = by_name(hlsl.parse(HLSL_TABLE))
    implicit = features["FeedbackTexture2D.WriteSamplerFeedback"]
    level = features["FeedbackTexture2D.WriteSamplerFeedbackLevel"]
    assert implicit.signature.params[0].type == Type.other("Texture2D")
    # Only the implicit level of detail needs derivatives, so a fragment shader.
    assert (implicit.stage, level.stage) == ("fragment", "compute")


def test_hlsl_methods_of_types_no_probe_declares_are_untested():
    features = by_name(hlsl.parse(HLSL_TABLE))
    assert features["FutureObject.IsValid"].signature is None
    assert (
        features["FutureObject.IsValid"].note == "no probe for FutureObject objects yet"
    )


def test_hlsl_work_graph_objects_are_probed_in_a_node_shader():
    node_output = by_name(hlsl.parse(HLSL_TABLE))["NodeOutput.IsValid"]
    assert (node_output.stage, node_output.signature) == (
        "node",
        Signature(Type("bool"), ()),
    )


def test_hlsl_hit_object_methods_are_probed_in_a_ray_generation_shader():
    features = by_name(hlsl.parse(HLSL_TABLE))
    is_miss = features["DxHitObject.IsMiss"]
    assert (is_miss.stage, is_miss.signature.result) == ("ray-generation", Type("bool"))
    assert features["DxHitObject.MakeNop"].stage == "ray-generation"
    # A static method is called on its type, the others on the object.
    assert features["DxHitObject.MakeNop"].spelling == "dx::HitObject::MakeNop"
    assert features["DxHitObject.IsMiss"].spelling == "IsMiss"


def test_hlsl_stream_methods_are_probed_in_a_geometry_shader():
    features = by_name(hlsl.parse(HLSL_TABLE))
    append, restart = features["Stream.Append"], features["Stream.RestartStrip"]
    assert (append.stage, restart.stage) == ("geometry", "geometry")
    # Append takes a vertex of the stream's element type: the one the geometry probe fills in.
    assert append.signature.params == (Param(Type.other("Vertex")),)
    assert restart.signature == Signature(VOID, ())


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
    # The table declares the vertex index ``in``, but DXC requires a literal.
    vertex = features["GetAttributeAtVertex"]
    assert [p.direction for p in vertex.signature.params] == ["in", "const"]
    assert vertex.stage == "fragment"
    barrier = features["Barrier"].signature
    assert [p.direction for p in barrier.params] == ["const", "const"]
    # The table admits any type, but the function marks a resource array index: ``uint`` in the HLSL reference.
    assert features["NonUniformResourceIndex"].signature == Signature(
        Type("uint"), (Param(Type("uint"), "in"),)
    )


def test_hlsl_reads_16_bit_scalars():
    features = by_name(hlsl.parse(HLSL_TABLE, {"mix", "fma"}))
    asint16 = features["asint16"].signature
    assert asint16.result == Type("int16", (2,))
    assert asint16.params[0].type == Type("half", (2,))
    dot2add = features["dot2add"].signature
    assert [p.type for p in dot2add.params] == [Type("half", (2,))] * 2 + [
        Type("float")
    ]


def test_hlsl_reads_64_bit_and_packed_8_bit_integers():
    features = by_name(hlsl.parse(HLSL_TABLE, {"mix", "fma"}))
    exchange = features["RWByteAddressBuffer.InterlockedExchange64"].signature
    assert [p.type for p in exchange.params] == [Type("uint")] + [Type("int64")] * 2
    unpack = features["unpack_s8s32"].signature
    assert unpack.result == Type("int", (4,))
    assert unpack.params[0].type == Type("int8x4_packed")
    assert features["pack_u8"].signature.result == Type("uint8x4_packed")
    # The 32-bit overload of a function is probed rather than its 64-bit one.
    assert features["InterlockedAdd"].signature.params[0].type == Type("uint")


def test_hlsl_atomic_destinations_take_the_unsigned_instance_of_their_operand():
    table = (
        "namespace Intrinsics {\n"
        "void [[]] InterlockedCompareExchange(ref int32_only result, in uint compare, in uint value,"
        " out any_int32 original);\n"
        "void [[]] InterlockedMin(ref int32_only result, in any_int32 value) : interlockedmin;\n"
        "}\n"
    )
    features = by_name(hlsl.parse(table))
    exchange = features["InterlockedCompareExchange"].signature
    assert [p.type for p in exchange.params] == [Type("uint")] * 4
    # With no fixed unsigned operand, the signed instance stays.
    assert [p.type for p in features["InterlockedMin"].signature.params] == [
        Type("int")
    ] * 2


def test_hlsl_probes_compile_for_the_shader_model_the_feature_list_assumes():
    major, minor = hlsl.SHADER_MODEL
    assert validators.HLSL_PROFILES == {
        stage: f"{prefix}_{major}_{minor}"
        for stage, prefix in (
            ("compute", "cs"),
            ("fragment", "ps"),
            ("vertex", "vs"),
            ("geometry", "gs"),
            ("tessellation-control", "hs"),
            ("tessellation-evaluation", "ds"),
            ("task", "as"),
            ("mesh", "ms"),
            ("ray-generation", "lib"),
            ("intersection", "lib"),
            ("any-hit", "lib"),
            ("closest-hit", "lib"),
            ("miss", "lib"),
            ("callable", "lib"),
            ("node", "lib"),
        )
    }


def test_hlsl_takes_crossgl_spellings_from_the_template_layer():
    features = by_name(hlsl.parse(HLSL_TABLE, {"mix", "fma"}))
    assert features["lerp"].crossgl == "mix"
    assert features["mad"].crossgl == "fma"
    # No GLSL atomicAdd in this excerpt, so the probe keeps the HLSL spelling.
    assert features["InterlockedAdd"].crossgl == "InterlockedAdd"

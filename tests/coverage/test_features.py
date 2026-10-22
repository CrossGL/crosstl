import pytest

from tools.coverage import probes, validators
from tools.coverage.features import (
    cuda,
    glsl,
    hip,
    hlsl,
    metal,
    mojo,
    slang,
    spirv,
    wgsl,
)
from tools.coverage.model import (
    COOP_COLUMNS,
    STRING,
    VOID,
    Param,
    Signature,
    Type,
    coop_vector,
    tuple_of,
)

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
    assert features["texture"].signature.unsupported == "sampler2D"
    assert len(features["texture"].signature.params) == 2


TEXTURE_EXCERPT = """
== Texture Functions

[source,glsl]
----
int textureSize(gsampler1D sampler, int lod)
ivec2 textureSize(gsampler2D sampler, int lod)
gvec4 texture(gsampler2D sampler, vec2 P [, float bias] )
gvec4 textureLod(gsampler2D sampler, vec2 P, float lod)
gvec4 textureLodOffset(gsampler2D sampler, vec2 P, float lod, ivec2 offset)
gvec4 textureGatherOffsets(gsampler2D sampler, vec2 P, ivec2 offsets[4] [, int comp])
----

== Image Functions

[source,glsl]
----
gvec4 imageLoad(readonly IMAGE_PARAMS)
uint imageAtomicAdd(IMAGE_PARAMS, uint data)
int imageAtomicAdd(IMAGE_PARAMS, int data)
----

== Fragment Processing Functions

[source,glsl]
----
vec2 interpolateAtOffset(float interpolant, vec2 offset)
----

== Geometry Shader Functions

[source,glsl]
----
void EmitStreamVertex(int stream)
void EmitVertex()
----

== Subpass-Input Functions

=== SubpassLoad
[source,glsl]
----
gvec4 subpassLoad(gsubpassInput subpass)
gvec4 subpassLoad(gsubpassInputMS subpass, int sample)
----
"""


def test_glsl_probes_textures_and_images_through_their_2d_forms():
    features = by_name(glsl.parse(TEXTURE_EXCERPT))
    size = features["textureSize"].signature
    assert size.result == Type("int", (2,))
    assert size.params[0].type == Type.other("sampler2D")
    assert features["textureLod"].signature.result == Type("float", (4,))
    assert features["textureLod"].stage == "compute"
    load = features["imageLoad"].signature
    assert [p.type for p in load.params] == [
        Type.other("image2D"),
        Type("int", (2,)),
    ]
    atomic = features["imageAtomicAdd"].signature
    assert atomic.params[0].type == Type.other("uimage2D")
    assert atomic.params[2].type == Type("uint")


def test_glsl_marks_constant_offsets_arrays_and_implicit_lod():
    features = by_name(glsl.parse(TEXTURE_EXCERPT))
    offset = features["textureLodOffset"].signature.params[3]
    assert offset == Param(Type("int", (2,)), "const")
    # An interpolation offset is a float and need not be constant.
    assert features["interpolateAtOffset"].signature.params[1].direction == "in"
    gather = features["textureGatherOffsets"].signature
    assert gather.params[2] == Param(Type("int", (2,), count=4), "const")
    assert len(gather.params) == 3
    assert features["texture"].stage == "fragment"
    stream = features["EmitStreamVertex"]
    assert stream.stage == "geometry"
    assert stream.signature.params == (Param(Type("int"), "const"),)


def test_glsl_probes_subpass_loads_in_a_vulkan_fragment_shader():
    features = by_name(glsl.parse(TEXTURE_EXCERPT))
    subpass = features["subpassLoad"]
    assert subpass.signature.params == (Param(Type.other("subpassInput")),)
    assert (subpass.stage, subpass.target) == ("fragment", "vulkan")
    assert features["texture"].target == ""


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


WGSL_TEXTURES = """
match fiu32: f32 | i32 | u32
match iu32: i32 | u32

////////////////////////////////////////////////////////////////////////////////
// Builtin Functions                                                          //
////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////
// Textures
/////////////////////////////////////////
@must_use implicit(T: fiu32, C: iu32, L: iu32) fn textureLoad(texture: texture_1d<T>, coords: C, level: L) -> vec4<T>
@must_use implicit(T: fiu32, C: iu32, L: iu32) fn textureLoad(texture: texture_2d<T>, coords: vec2<C>, level: L) -> vec4<T>
@must_use fn textureSampleCompareLevel(texture: texture_depth_2d, sampler: sampler_comparison, coords: vec2<f32>,
                                       depth_ref: f32) -> f32
@must_use @stage("fragment") fn textureSampleBaseClampToEdge(texture: texture_2d<f32>, sampler: sampler,
                                                             coords: vec2<f32>) -> vec4<f32>
"""


def test_wgsl_texture_functions_take_a_2d_texture_and_call_glsl_from_crossgl():
    features = by_name(wgsl.parse(WGSL_TEXTURES))
    load = features["textureLoad"]
    assert load.signature.params[0].type == Type.other("texture_2d<T>")
    assert load.signature.result == Type("float", (4,))
    assert load.crossgl == "texelFetch"
    assert [p.type for p in load.counterpart.params] == [
        Type.other("sampler2D"),
        Type("int", (2,)),
        Type("int"),
    ]
    compare = features["textureSampleCompareLevel"].counterpart
    # WGSL always samples level 0, so the CrossGL probe passes the level as a constant.
    assert compare.params[-1] == Param(Type("float"), "const")
    assert features["textureSampleBaseClampToEdge"].counterpart is None
    # CrossGL has no clamp-to-edge sample, so the operation is a gap rather than an untested probe.
    assert features["textureSampleBaseClampToEdge"].crossgl == ""
    assert "half a texel" in features["textureSampleBaseClampToEdge"].note


WGSL_WORKGROUP = """
match iu32: i32 | u32

////////////////////////////////////////////////////////////////////////////////
// Builtin Functions                                                          //
////////////////////////////////////////////////////////////////////////////////
@must_use @stage("compute") implicit(T: __constructible)
  fn workgroupUniformLoad(ptr<workgroup, T, read_write>) -> T
@must_use @stage("compute") implicit(T: iu32)
  fn workgroupUniformLoad(ptr<workgroup, atomic<T>, read_write>) -> T
"""


def test_wgsl_workgroup_pointer_takes_a_workgroup_variable():
    load = by_name(wgsl.parse(WGSL_WORKGROUP))["workgroupUniformLoad"]
    assert load.signature == Signature(
        Type("float"), (Param(Type("float"), "workgroup"),)
    )


WGSL_ARRAY_LENGTH = """
////////////////////////////////////////////////////////////////////////////////
// Builtin Functions                                                          //
////////////////////////////////////////////////////////////////////////////////
@must_use implicit(AS: uniform_or_storage, T, R: read)
  fn arrayLength(ptr<AS, runtime_array<T>, R>) -> u32
"""


def test_wgsl_array_length_reads_the_probe_buffer_as_crossgl_buffer_dimensions():
    length = by_name(wgsl.parse(WGSL_ARRAY_LENGTH))["arrayLength"]
    assert length.signature == Signature(Type("uint"), (Param(Type("uint"), "buffer"),))
    assert length.crossgl == "buffer_dimensions"
    native = probes.build(length, "wgsl").source
    assert "var<storage, read_write> uin: array<u32>;" in native
    assert "var r: u32 = arrayLength(&uin);" in native
    crossgl = probes.build(length, "crossgl").source
    assert "uint r = buffer_dimensions(uin);" in crossgl


METAL_LIST = """# Generated by python -m tools.coverage.features.metal; do not edit.
# toolchain: Apple metal version 1.0 (test)
# language: Metal Shading Language 9.9
metal_math fma float3(float3, float3, float3)
metal_math frexp float3(float3, out int3)
metal_matrix determinant float(float3x3)
metal_graphics dfdx float3(float3)
metal_atomic atomic_fetch_add_explicit -
metal_pack pack_half_to_unorm4x8 uint(half4)
"""


def test_metal_reads_the_generated_list():
    features = by_name(metal.parse(METAL_LIST, {"fma", "dFdx", "frexp", "determinant"}))
    assert features["frexp"].signature.params[1] == Param(Type("int", (3,)), "out")
    pack = features["pack_half_to_unorm4x8"].signature
    assert pack == Signature(Type("uint"), (Param(Type("half", (4,))),))
    assert features["determinant"].signature.params[0].type == Type("float", (3, 3))
    assert features["dfdx"].stage == "fragment"
    assert features["dfdx"].crossgl == "dFdx"
    assert features["atomic_fetch_add_explicit"].signature is None
    assert (
        features["atomic_fetch_add_explicit"].note
        == "no overload takes only value types"
    )


def test_metal_lists_ballots_whose_mask_a_probe_reads(tmp_path):
    for header in metal.HEADERS:
        (tmp_path / header).write_text("")
    (tmp_path / "metal_simdgroup").write_text(
        "METAL_FUNC simd_vote simd_ballot(bool expr)\n{\n  return simd_vote(0);\n}\n"
        "METAL_FUNC bool simd_none(simd_vote vote)\n{\n  return false;\n}\n"
    )
    lines = metal.generate(tmp_path, "test", "9.9").splitlines()
    assert "metal_simdgroup simd_ballot simd_vote(bool)" in lines
    # A probe reads a mask a function returns, but has none to pass.
    assert "metal_simdgroup simd_none -" in lines
    ballot = by_name(metal.parse("metal_simdgroup simd_ballot simd_vote(bool)\n"))
    assert ballot["simd_ballot"].signature == Signature(
        Type.other("simd_vote"), (Param(Type("bool")),)
    )


METAL_SIMDGROUP_HEADER = """
template <typename T, typename _E = typename enable_if<__is_valid_simdgroup_type<T>::value>::type>
METAL_FUNC T simd_shuffle(T data METAL_MAYBE_UNDEF, ushort simd_lane_id)
{
  return __metal_simd_shuffle(data, simd_lane_id);
}
"""


def test_metal_header_parse_drops_attribute_macros():
    # METAL_MAYBE_UNDEF follows the parameter name; it is neither part of the type nor the name.
    assert list(metal.declarations("metal_simdgroup", METAL_SIMDGROUP_HEADER)) == [
        ("simd_shuffle", "float", ["float", "uint"])
    ]


METAL_MATRIX_HEADER = """
template <typename T, int Cols, int Rows>
METAL_FUNC METAL_CONSTEXPR_SINCE_MTL_410 matrix<T, Cols, Rows> operator*(T a, const device coherent(device) matrix<T, Cols, Rows> &b)
{
  return b * a;
}
template <typename T, int Cols, int Rows>
METAL_FUNC matrix<T, Cols, Rows> _matrix_scalar_product_impl(const device coherent(device) matrix<T, Cols, Rows> &a, T b)
{
}
template <typename T, int K>
METAL_FUNC METAL_CONSTEXPR_SINCE_MTL_410 device coherent(device) vec<T, K> &operator*=(device coherent(device) vec<T, K> &a, const thread  matrix<T, K, K> &b)
{
}
template <typename T, int Cols, int Rows>
METAL_FUNC matrix<T, Rows, Cols> transpose(matrix<T, Cols, Rows> x)
{
}
"""


def test_metal_header_parse_skips_operators_and_helpers():
    # Neither operator* nor an underscore helper is read as a function named after a parameter's coherent(device).
    assert list(metal.declarations("metal_matrix", METAL_MATRIX_HEADER)) == [
        ("transpose", "float3x3", ["float3x3"])
    ]


METAL_ATOMIC_HEADER = """
template <typename T, typename U, thread_scope S>
METAL_FUNC enable_if_t<_valid_fetch_add_sub_type<device T *>::value &&
                       is_convertible<T, U>::value, T>
atomic_fetch_add_explicit(volatile device _atomic<T, S> *object, U operand, memory_order order) METAL_CONST_ARG(order)
{
  return T(__metal_atomic_fetch_add_explicit(&object->__s, operand, int(order), int(S)));
}
template <typename T, typename U, thread_scope S>
METAL_FUNC enable_if_t<_valid_non_fetch_type<device T *>::value &&
                       is_convertible<T, U>::value>
atomic_store_explicit(volatile device _atomic<T, S> *object, U desired, memory_order order, mem_flags flags)
{
}
template <typename T, typename U>
METAL_FUNC enable_if_t<_valid_max_type<device T *>::value &&
                       is_convertible<T, U>::value>
atomic_max_explicit(volatile coherent(device) device _atomic<T> *object, U operand, memory_order order)
{
}
template <typename T, typename U, thread_scope S>
METAL_FUNC enable_if_t<_valid_non_fetch_type<device T *>::value, bool>
atomic_compare_exchange_weak_explicit(volatile device _atomic<T, S> *object, thread T *expected, U desired)
{
}
METAL_FUNC void atomic_thread_fence(mem_flags flags, memory_order order, thread_scope scope = thread_scope_device) METAL_CONST_ARG(flags)
METAL_FUNC void prefetch(const device float *p)
{
}
"""


def test_metal_header_parse_reads_atomic_functions():
    # The atomic object is the probe's output buffer element; trailing attribute macros and defaulted parameters are
    # not parameters, and enable_if_t without a type is void.
    assert list(metal.declarations("metal_atomic", METAL_ATOMIC_HEADER)) == [
        ("atomic_fetch_add_explicit", "int", ["inout int", "int", "memory_order"]),
        (
            "atomic_store_explicit",
            "void",
            ["inout int", "int", "memory_order", "mem_flags"],
        ),
        # Metal declares the atomic max and min without a fetch for 64-bit unsigned integers only.
        ("atomic_max_explicit", "void", ["inout ulong", "ulong", "memory_order"]),
        # A pointer to a thread-local value takes the address of a local.
        (
            "atomic_compare_exchange_weak_explicit",
            "bool",
            ["inout int", "address int", "int"],
        ),
        ("atomic_thread_fence", "void", ["mem_flags", "memory_order"]),
        # A pointer to anything else is not probed.
        ("prefetch", "void", ["float*"]),
    ]


METAL_ATOMIC_LIST = """metal_atomic atomic_fetch_add_explicit int(inout int, int, memory_order)
metal_atomic atomic_fetch_add int(inout int, int)
metal_atomic atomic_load_explicit int(inout int, memory_order)
metal_atomic atomic_max void(inout ulong, ulong)
metal_compute threadgroup_barrier void(mem_flags)
"""


def test_metal_atomic_functions_call_glsl_atomics_from_crossgl():
    features = by_name(metal.parse(METAL_ATOMIC_LIST, {"atomicAdd", "barrier"}))
    add = features["atomic_fetch_add_explicit"]
    assert add.signature.params == (
        Param(Type("int"), "inout"),
        Param(Type("int")),
        Param(Type.other("memory_order"), "const"),
    )
    # CrossGL's atomics take no memory order.
    assert add.crossgl == "atomicAdd"
    assert add.counterpart.params == (Param(Type("int"), "inout"), Param(Type("int")))
    assert features["atomic_fetch_add"].crossgl == "atomicAdd"
    assert features["atomic_fetch_add"].counterpart is None
    assert features["atomic_max"].signature.params[0] == Param(Type("uint64"), "inout")
    assert features["threadgroup_barrier"].crossgl == "barrier"
    assert features["threadgroup_barrier"].counterpart.params == ()
    # An operation CrossGL has only without a memory order is a gap, not an untested probe.
    load = features["atomic_load_explicit"]
    assert load.crossgl == ""
    assert "memory order" in load.note


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
    ]
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


MOJO_MATH_INIT = """
from std.utils.numerics import isnan, nan
from .math import (
    Floorable,
    floor,
    frexp,
    gcd,
    rsqrt,
    sqrt,
)
"""

MOJO_MATH = """
from std.utils.numerics import isnan


@always_inline
def floor[T: Floorable, //](value: T) -> T:
    \"\"\"Floor.\"\"\"
    return value.__floor__()


def _sqrt_nvvm(x: SIMD, out res: type_of(x)):
    pass


def sqrt[
    dtype: DType, width: SIMDLength, //
](x: SIMD[dtype, width]) -> SIMD[dtype, width]:
    comptime assert (
        dtype.is_numeric() or dtype == .bool
    ), "type must be arithmetic or boolean"

    comptime if dtype.is_integral():
        return x
    return x


def sqrt(x: Int) -> Int:
    return x


def rsqrt[
    dtype: DType, width: SIMDLength, //
](x: SIMD[dtype, width]) -> type_of(x) where dtype.is_floating_point():
    return x


def frexp[
    dtype: DType, width: SIMDLength, //
](x: SIMD[dtype, width]) -> StaticTuple[
    SIMD[dtype, width], 2
] where dtype.is_floating_point():
    return x


def gcd(m: Int, n: Int, /) -> Int:
    return m
"""

MOJO_NUMERICS = """
def nan[dtype: DType]() -> Scalar[dtype]:
    comptime assert dtype.is_floating_point(), "fp only"
    return 0


def isnan[
    dtype: DType, width: SIMDLength, //
](val: SIMD[dtype, width]) -> SIMD[.bool, width]:
    return False
"""

MOJO_BIT_INIT = """
from .bit import (
    pop_count,
    rotate_bits_left,
)
"""

MOJO_BIT = """
def pop_count[dtype: DType, width: SIMDLength, //](val: SIMD[dtype, width]) -> SIMD[dtype, width]:
    comptime assert dtype.is_integral(), "must be integral"
    return val


def rotate_bits_left[
    dtype: DType,
    width: SIMDLength,
    //,
    shift: Int,
](x: SIMD[dtype, width]) -> SIMD[dtype, width] where dtype.is_integral():
    return x
"""


def test_mojo_reads_the_functions_the_packages_export():
    glsl_names = {"floor", "inversesqrt", "bitCount", "isnan"}
    found = mojo.parse(
        MOJO_MATH_INIT, MOJO_MATH, MOJO_NUMERICS, MOJO_BIT_INIT, MOJO_BIT, glsl_names
    )
    features = by_name(found)
    # Traits and private helpers are not functions of the public API.
    assert sorted(features) == [
        "floor",
        "frexp",
        "gcd",
        "isnan",
        "nan",
        "pop_count",
        "rotate_bits_left",
        "rsqrt",
        "sqrt",
    ]
    assert {f.group for f in found} == {"math", "bit"}
    assert features["floor"].signature == Signature(
        Type("float"), (Param(Type("float")),)
    )
    # The first definition is probed, and a comptime if on the type does not constrain it.
    assert features["sqrt"].signature.params == (Param(Type("float")),)
    assert features["rsqrt"].signature.result == Type("float")
    assert features["rsqrt"].crossgl == "inversesqrt"
    assert features["isnan"].signature.result == Type("bool")
    assert features["pop_count"].signature.result == Type("int")
    assert features["pop_count"].crossgl == "bitCount"
    assert features["gcd"].signature == Signature(
        Type("int64"), (Param(Type("int64")), Param(Type("int64")))
    )
    # frexp returns a pair the probe reads the first value of, and CrossGL writes it the way GLSL does.
    assert features["frexp"].signature.result == Type.other("StaticTuple[Float32, 2]")
    assert features["frexp"].counterpart == Signature(
        Type("float"), (Param(Type("float")), Param(Type("int"), "out"))
    )


def test_mojo_names_the_parameters_no_argument_fixes():
    features = by_name(
        mojo.parse(MOJO_MATH_INIT, MOJO_MATH, MOJO_NUMERICS, MOJO_BIT_INIT, MOJO_BIT)
    )
    assert features["nan"].spelling == "nan[dtype=DType.float32]"
    assert features["nan"].signature == Signature(Type("float"), ())
    assert features["rotate_bits_left"].spelling == "rotate_bits_left[shift=1]"
    assert features["rotate_bits_left"].signature.params == (Param(Type("int")),)
    assert features["sqrt"].spelling == ""

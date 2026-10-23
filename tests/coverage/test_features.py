from tools.coverage import probes, validators
from tools.coverage.features import glsl, hlsl, spirv, wgsl
from tools.coverage.model import VOID, Param, Signature, Type

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

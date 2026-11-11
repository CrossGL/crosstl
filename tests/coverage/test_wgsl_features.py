"""The WGSL feature list read from Tint's ``wgsl.def``."""

from tools.coverage import probes, spellings
from tools.coverage.features import wgsl
from tools.coverage.model import Param, Signature, Type


def by_name(features):
    return {feature.name: feature for feature in features}


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


def test_wgsl_counts_zero_bits_with_the_crossgl_functions_metal_names():
    assert spellings.spelling("wgsl", "countLeadingZeros", set()) == "clz"
    assert spellings.spelling("wgsl", "countTrailingZeros", set()) == "ctz"
    assert spellings.spelling("mojo", "count_leading_zeros", set()) == "clz"


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

from tools.coverage.features import glsl, hlsl, spirv
from tools.coverage.model import Type

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

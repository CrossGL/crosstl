"""The GLSL feature list from the specification's prototypes, and the SPIR-V list built on it."""

import pytest

from tools.coverage.features import glsl, spirv
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


def test_spirv_explains_unreachable_instructions_and_spells_nan_aware_ones():
    features = by_name(spirv.parse(GRAMMAR, GLSL_EXCERPT))
    # NMin takes min's operands and is CrossGL's NaN-aware fmin.
    assert features["NMin"].signature.params[1].type == Type("float", (3,))
    assert features["NMin"].crossgl == "fmin"
    assert features["UMulExtended"].signature is None
    assert features["UMulExtended"].note


CORE_GRAMMAR = """{"instructions": [
 {"opname": "OpDPdx", "class": "Derivative"},
 {"opname": "OpFOrdLessThan", "class": "Relational_and_Logical"},
 {"opname": "OpSLessThan", "class": "Relational_and_Logical"},
 {"opname": "OpULessThan", "class": "Relational_and_Logical"}
]}"""
CORE_GLSL = """
== Vector Relational Functions

[source,glsl]
----
bvec lessThan(vec x, vec y)
bvec lessThan(ivec x, ivec y)
bvec lessThan(uvec x, uvec y)
----

== Fragment Processing Functions

[source,glsl]
----
genFType dFdx(genFType p)
----
"""
# A GL_KHR_shader_subgroup text that defines no function.
NO_SUBGROUP_FUNCTIONS = (
    "Additions to Chapter 8 of the OpenGL Shading Language Specification\n"
)
CORE_TABLE = {
    "OpDPdx": ("dFdx", ""),
    "OpFOrdLessThan": ("lessThan", "float"),
    "OpSLessThan": ("lessThan", "int"),
    "OpULessThan": ("lessThan", "uint"),
}


def test_spirv_core_instructions_take_the_overload_that_selects_them(monkeypatch):
    monkeypatch.setattr(spirv, "CORE", CORE_TABLE)
    features = by_name(spirv.core(CORE_GRAMMAR, CORE_GLSL, NO_SUBGROUP_FUNCTIONS))
    assert sorted(features) == sorted(CORE_TABLE)
    for name, scalar in (
        ("OpFOrdLessThan", "float"),
        ("OpSLessThan", "int"),
        ("OpULessThan", "uint"),
    ):
        feature = features[name]
        assert feature.signature.params[0].type == Type(scalar, (3,))
        assert (feature.crossgl, feature.spelling) == ("lessThan", "lessThan")
        assert feature.group == "core-relational-and-logical"
        assert spirv.is_core(feature)
    # A derivative is probed in the fragment stage, as its GLSL function is.
    assert (features["OpDPdx"].stage, features["OpDPdx"].group) == (
        "fragment",
        "core-derivative",
    )


def test_spirv_core_table_entries_must_exist_upstream(monkeypatch):
    monkeypatch.setattr(spirv, "CORE", {"OpIAdd": ("lessThan", "")})
    with pytest.raises(ValueError, match="OpIAdd is not an instruction"):
        spirv.core(CORE_GRAMMAR, CORE_GLSL, NO_SUBGROUP_FUNCTIONS)
    monkeypatch.setattr(spirv, "CORE", {"OpSLessThan": ("lessThan", "bool")})
    with pytest.raises(ValueError, match="no overload on bool"):
        spirv.core(CORE_GRAMMAR, CORE_GLSL, NO_SUBGROUP_FUNCTIONS)


def test_spirv_glsl_std_450_instructions_are_not_core():
    features = spirv.parse(GRAMMAR, GLSL_EXCERPT)
    assert features and not any(spirv.is_core(f) for f in features)

from tools.coverage.features import glsl, spirv
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
    assert sorted(features) == ["abs", "atan", "dFdx", "texture", "umulExtended"]


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


def test_spirv_explains_unreachable_instructions():
    features = by_name(spirv.parse(GRAMMAR, GLSL_EXCERPT))
    assert features["NMin"].signature is None
    assert "NaN" in features["NMin"].note
    assert features["UMulExtended"].signature is None
    assert features["UMulExtended"].note

"""The GLSL ES 3.x output row: the GLSL ES text of the GLSL specification source, its probes and its version check."""

import pytest

from tools.coverage import (
    chart,
    constructs,
    features,
    issues,
    probes,
    report,
    runner,
    templates,
    validators,
)
from tools.coverage.features import gles
from tools.coverage.model import LANGUAGES, Feature, Type
from tools.coverage.templates import ABSENT
from tools.coverage.templates import gles as gles_templates
from tools.coverage.upstream import SOURCES, cache_dir

SPECIFICATION = """
== Common Functions

=== Abs
[source,glsl]
----
genFType abs(genFType x)
ifdef::GLSL[]
genDType abs(genDType x)
endif::GLSL[]
ifdef::ESSL[]
genIType abs(genIType x)
endif::ESSL[]
----

ifdef::GLSL[]
=== Double Pack
[source,glsl]
----
double packDouble2x32(uvec2 v)
----
endif::GLSL[]

== Subpass-Input Functions

=== SubpassLoad
[source,glsl]
----
gvec4 subpassLoad(gsubpassInput subpass)
----

ifdef::GLSL[]
== Shader Invocation Group Functions
endif::GLSL[]

=== AnyInvocation
[source,glsl]
----
bool anyInvocation(bool value)
----
"""

VEC3 = Type("float", (3,))
IVEC3 = Type("int", (3,))


def test_gles_reads_the_glsl_es_text_of_the_specification_source():
    found = gles.overloads(SPECIFICATION)
    assert sorted(found) == ["abs", "subpassLoad"]
    # The ES text keeps abs(genIType) and drops abs(genDType); desktop GLSL keeps the other.
    assert [s.result for s in found["abs"][1]] == [VEC3, IVEC3]
    desktop = features.glsl.overloads(SPECIFICATION)
    assert sorted(desktop) == ["abs", "anyInvocation", "packDouble2x32", "subpassLoad"]
    assert [s.result for s in desktop["abs"][1]] == [VEC3, Type("double", (3,))]


def test_gles_leaves_out_a_function_the_es_text_files_under_another_section():
    # The source hides the Shader Invocation Group Functions heading from GLSL ES but not its subsections, which
    # the ES text would then list under Subpass-Input Functions.
    assert features.glsl.overloads(SPECIFICATION, essl_text=True)["anyInvocation"][
        0
    ] == ("subpass-input-functions")
    assert "anyInvocation" not in gles.overloads(SPECIFICATION)


def test_gles_functions_keep_glsl_stages_and_targets():
    listed = {f.name: f for f in gles.parse(SPECIFICATION)}
    assert listed["abs"].language == "gles"
    assert listed["abs"].stage == "compute"
    assert listed["subpassLoad"].stage == "fragment"
    assert listed["subpassLoad"].target == "vulkan"


def _cached(name):
    source = SOURCES[name]
    return (cache_dir() / "upstream" / source.commit / source.path).is_file()


@pytest.mark.skipif(
    not _cached("glsl"), reason="the GLSL specification source is not cached"
)
def test_gles_lists_the_glsl_es_3_20_functions():
    names = {f.name for f in features.load("gles", offline=True)}
    # GLSL ES 3.20 has geometry shaders, image atomics and interpolation functions, and no doubles, 1D textures or
    # vertex streams (GLSL ES 3.20, Built-In Functions).
    assert {"EmitVertex", "imageAtomicAdd", "interpolateAtCentroid"} <= names
    assert not names & {
        "packDouble2x32",
        "EmitStreamVertex",
        "textureQueryLod",
        "anyInvocation",
    }


def test_gles_is_an_output_row_with_its_own_name_and_tools():
    assert "gles" in LANGUAGES
    assert runner.OUTPUT_BACKENDS["gles"] == "gles"
    assert "gles" not in runner.INPUT_BACKENDS
    assert ("gles", "input") in report.NOT_OFFERED
    assert chart.name("gles") == "GLSL ES"
    assert validators.TOOLS["gles"] == "glslangValidator"
    assert issues.TOOLS["gles"] == "glslang"


def test_gles_probes_name_glsl_es_3_20_and_every_precision():
    kernel = probes.build(
        Feature(
            "gles",
            "function",
            "textureSize",
            features.glsl.parse_prototype("ivec2 textureSize(sampler2D s, int lod)")[1],
            crossgl="textureSize",
        ),
        "gles",
    ).source
    lines = kernel.splitlines()
    assert lines[:3] == [
        "#version 320 es",
        "precision highp float;",
        "precision highp int;",
    ]
    assert "precision highp sampler2D;" in lines
    assert "layout(binding = 8) uniform highp sampler2D u0;" in lines


def test_gles_programs_put_the_precisions_after_the_extensions():
    text = "#version 460\n#extension GL_EXT_a : require\nlayout(location = 0) in highp sampler3D t;\nvoid main() {}\n"
    assert gles_templates.program(text).splitlines()[:5] == [
        "#version 320 es",
        "#extension GL_EXT_a : require",
        "precision highp float;",
        "precision highp int;",
        "precision highp sampler3D;",
    ]
    with pytest.raises(ValueError, match="#version"):
        gles_templates.program("void main() {}\n")


def test_gles_templates_leave_out_what_glsl_es_lacks():
    table = templates.TEMPLATES["gles"]
    names = {construct.name for construct in constructs.CURATED} | {
        "double",
        "ray-generation",
        "mesh",
        "task",
        "intersection",
        "any-hit",
        "closest-hit",
        "miss",
        "callable",
    }
    assert set(gles_templates.MISSING) <= names
    assert all(table[name] is ABSENT for name in gles_templates.MISSING)
    assert all(
        template.program.startswith("#version 320 es\n")
        for template in table.values()
        if template is not ABSENT and template.program
    )
    # GLSL ES initializes arrays with constructors, as WebGL's probes do.
    assert "float[4](" in table["array-initializer"].body


def test_gles_version_check():
    verdict = validators.validate("gles", "#version 460\nvoid main() {}\n")
    assert not verdict.accepted
    assert "#version 300, 310 or 320 es" in verdict.log


@pytest.mark.skipif(
    not validators.available("gles"), reason="glslangValidator is not installed"
)
@pytest.mark.parametrize("version", ["310 es", "320 es"])
def test_gles_validator_accepts_glsl_es_compute_shaders(version):
    source = (
        f"#version {version}\nlayout(local_size_x = 1) in;\n"
        "layout(std430, binding = 0) buffer B { float x[]; };\n"
        "void main() { x[gl_GlobalInvocationID.x] = 1.0; }\n"
    )
    verdict = validators.validate("gles", source)
    assert verdict.accepted, verdict.log


@pytest.mark.skipif(
    not validators.available("gles") or not _cached("glsl"),
    reason="glslangValidator or the GLSL specification source is missing",
)
def test_gles_stage_probes_are_valid_glsl_es():
    rejected = {}
    for feature in features.load("gles", offline=True):
        if not constructs.is_construct(feature) or feature.category != "stage":
            continue
        probe = probes.build(feature, "gles")
        verdict = validators.validate(
            "gles", probe.source, feature.stage, probe.entry, feature.target
        )
        if not verdict.accepted:
            rejected[feature.name] = validators.first_error(verdict.log)
    assert not rejected


@pytest.mark.skipif(
    not validators.available("gles"), reason="glslangValidator is not installed"
)
def test_gles_output_is_measured_end_to_end():
    feature = Feature(
        "gles",
        "function",
        "abs",
        features.glsl.parse_prototype("genFType abs(genFType x)")[1],
        crossgl="abs",
    )
    result = runner.run_output(feature, "gles")
    assert result.outcome == "supported", result.detail
    assert result.validator == "glslangValidator"

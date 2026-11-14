"""The language-level feature list: anchors, selection per language, probe programs and native templates."""

import re
from pathlib import Path

import pytest

from tools.coverage import (
    constructs,
    crossgl,
    execution,
    lowering,
    probes,
    semantic,
    templates,
    validators,
)
from tools.coverage.model import Feature, Result
from tools.coverage.templates import relations
from tools.coverage.templates.base import ABSENT
from tools.coverage.templates.family import OPERATORS

from .construct_grammar import GRAMMAR, construct_feature, grammar


def test_stages_are_the_shader_execution_models():
    stages = [c.name for c in constructs.universe(GRAMMAR) if c.category == "stage"]
    assert stages == [
        "vertex",
        "tessellation-control",
        "tessellation-evaluation",
        "geometry",
        "fragment",
        "compute",
        "task",
        "mesh",
        "ray-generation",
        "intersection",
        "any-hit",
        "closest-hit",
        "miss",
        "callable",
    ]


def test_scalar_widths_come_from_capabilities():
    types = {c.name: c.anchors for c in constructs.universe(GRAMMAR)}
    assert types["int64"] == ("Capability.Int64",)
    assert types["half"] == ("Capability.Float16",)


def test_an_anchor_missing_upstream_fails_loudly():
    with pytest.raises(ValueError, match="OpSwitch"):
        constructs.universe(grammar(drop="OpSwitch"))


def test_a_language_lists_only_what_it_can_spell():
    wgsl = {f.name for f in constructs.features("wgsl", GRAMMAR)}
    assert {"switch", "half", "vertex"} <= wgsl
    assert not {"double", "int64", "geometry", "any-hit"} & wgsl
    assert "num-workgroups" not in {
        f.name for f in constructs.features("hlsl", GRAMMAR)
    }


def test_spirv_lists_every_construct_glsl_hlsl_or_slang_spells():
    spirv = {f.name for f in constructs.features("spirv", GRAMMAR)}
    spelled = {
        c.name
        for c in constructs.universe(GRAMMAR)
        if any(constructs.spells(c.name, lang) for lang in ("glsl", "hlsl", "slang"))
    }
    assert spirv == spelled
    assert not {"vector-halves", "numeric-components"} & spirv
    # Only Slang spells pointers among the three.
    assert {"pointer-local", "pointer-arithmetic"} <= spirv


def test_pointer_probes_are_absent_where_the_language_has_no_pointers():
    names = {"pointer-local", "pointer-arithmetic"}
    listing = {
        language: names & {f.name for f in constructs.features(language, GRAMMAR)}
        for language in templates.TEMPLATES
    }
    assert {lang for lang, found in listing.items() if found == names} == {
        "cuda",
        "hip",
        "metal",
        "mojo",
        "opencl",
        "slang",
        "spirv",
    }
    # WGSL and rust-gpu have pointers and references but no pointer arithmetic; GLSL and HLSL have neither.
    assert listing["wgsl"] == listing["rust"] == {"pointer-local"}
    assert not listing["glsl"] | listing["hlsl"] | listing["webgl"]
    local = probes.build(construct_feature("opencl", "pointer-local"), "opencl")
    assert "__global float* p = &outp[i];" in local.source
    arithmetic = probes.build(construct_feature("slang", "pointer-arithmetic"), "slang")
    assert "float* p = &outp[0] + i;" in arithmetic.source
    assert "q = q + i;" in arithmetic.source


def test_only_opencl_lists_its_vector_component_names():
    names = {"vector-halves", "numeric-components"}
    listing = {
        language
        for language in templates.TEMPLATES
        if names & {f.name for f in constructs.features(language, GRAMMAR)}
    }
    assert listing == {"opencl"}
    halves = probes.build(construct_feature("opencl", "vector-halves"), "opencl")
    assert "v.hi;" in halves.source and "v.even = (float2)(" in halves.source
    assert "v.odd.hi;" in halves.source and "* 16.0f" in halves.source
    numeric = probes.build(construct_feature("opencl", "numeric-components"), "opencl")
    assert "v.s32;" in numeric.source and "v.s13 = " in numeric.source
    assert "v.S2 = v.s0 + 4.0f;" in numeric.source
    # CrossGL writes the same components as swizzles.
    crossgl_probe = probes.build(
        construct_feature("opencl", "vector-halves"), "crossgl"
    ).source
    assert "v.zw;" in crossgl_probe and "v.xz = vec2(" in crossgl_probe


def test_constructs_share_a_semantic_id_across_languages():
    ids = {
        semantic.lower(construct_feature(lang, "switch"))[0]
        for lang in ("glsl", "wgsl", "spirv")
    }
    assert ids == {"construct:control-flow/switch"}


def test_every_curated_construct_has_a_crossgl_template():
    assert {c.name for c in constructs.CURATED} <= set(crossgl.CONSTRUCTS)


def test_a_construct_without_a_template_is_untested_not_absent(monkeypatch):
    feature = construct_feature("glsl", "tessellation-control")
    monkeypatch.delitem(templates.TEMPLATES["glsl"], "tessellation-control")
    with pytest.raises(probes.ProbeError, match="no glsl probe template"):
        probes.build(feature, "glsl")


def test_kernel_templates_fill_the_language_spelling_of_types():
    hlsl = probes.build(construct_feature("hlsl", "swizzle"), "hlsl").source
    assert "float4 v = float4(" in hlsl
    crossgl_probe = probes.build(construct_feature("glsl", "struct"), "crossgl").source
    assert crossgl_probe.index("struct Pair") < crossgl_probe.index("compute {")


def test_stage_programs_take_the_stage_suffix():
    probe = probes.build(construct_feature("glsl", "fragment"), "glsl")
    assert probe.suffix == ".frag"
    assert probes.build(construct_feature("glsl", "geometry"), "glsl").suffix == ".geom"
    assert probes.build(construct_feature("glsl", "discard"), "glsl").suffix == ".frag"
    assert (
        probes.build(construct_feature("glsl", "closest-hit"), "glsl").suffix
        == ".rchit"
    )
    tese = probes.build(construct_feature("glsl", "tessellation-evaluation"), "glsl")
    assert tese.suffix == ".tese"
    assert probes.build(construct_feature("glsl", "task"), "glsl").suffix == ".task"
    assert probes.build(construct_feature("glsl", "mesh"), "glsl").suffix == ".mesh"


def test_slang_names_the_hull_stage_on_its_entry_point():
    hull = probes.build(
        construct_feature("slang", "tessellation-control"), "slang"
    ).source
    assert '[shader("hull")]\n[domain("tri")]' in hull
    task = probes.build(construct_feature("slang", "task"), "slang").source
    assert '[shader("amplification")]\n[numthreads(1, 1, 1)]' in task


@pytest.mark.parametrize(
    ("stage", "air", "accepted"),
    [
        ("vertex", "define void @f() {\n}\n", False),
        ("vertex", "!air.vertex = !{!9}\n", True),
        ("compute", "!air.vertex = !{!9}\n", False),
        ("tessellation-evaluation", "!air.vertex = !{!9}\n", False),
        (
            "tessellation-evaluation",
            '!air.vertex = !{!9}\n!17 = !{!"air.patch", !"triangle"}\n',
            True,
        ),
        ("mesh", "!air.mesh = !{!9}\n", True),
        ("task", "!air.mesh = !{!9}\n", False),
        ("task", "!air.object = !{!9}\n", True),
        ("intersection", "!air.kernel = !{!9}\n", False),
        ("intersection", "!air.intersection = !{!9}\n", True),
    ],
)
def test_a_metal_probe_needs_an_entry_point_of_its_stage(
    monkeypatch, stage, air, accepted
):
    def run(arguments, cwd):
        Path(cwd, arguments[arguments.index("-o") + 1]).write_text(air)
        return validators.Verdict(True)

    monkeypatch.setattr(validators, "command", lambda name: (name,))
    monkeypatch.setattr(validators, "run", run)
    verdict = validators.validate("metal", "", stage)
    assert verdict.accepted is accepted
    assert accepted or f"no {stage} entry point" in verdict.log


def test_a_metal_intersection_probe_is_a_bounding_box_intersection_function():
    probe = probes.build(construct_feature("metal", "intersection"), "metal")
    assert "using namespace raytracing;" in probe.source
    assert "[[intersection(bounding_box)]]\nBoxResult probe(" in probe.source


def test_ray_tracing_and_mesh_stages_are_checked_for_vulkan(monkeypatch):
    commands = []

    def run(arguments, cwd):
        commands.append(arguments)
        return validators.Verdict(True)

    monkeypatch.setattr(validators, "command", lambda name: (name,))
    monkeypatch.setattr(validators, "run", run)
    validators.validate("glsl", "", "miss")
    validators.validate("glsl", "", "mesh")
    validators.validate("glsl", "", "fragment")
    validators.validate("glsl", "", "fragment", target="vulkan")
    assert commands == [
        ["glslangValidator", "--target-env", "vulkan1.2", "probe.rmiss"],
        ["glslangValidator", "--target-env", "vulkan1.2", "probe.mesh"],
        ["glslangValidator", "probe.frag"],
        ["glslangValidator", "--target-env", "vulkan1.1", "probe.frag"],
    ]


def test_entry_parameters_join_the_kernel_signature():
    wgsl = probes.build(construct_feature("wgsl", "local-invocation-id"), "wgsl").source
    assert "id: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>)" in wgsl
    metal = probes.build(construct_feature("metal", "sampled-texture"), "metal").source
    assert "texture2d<float> tex [[texture(0)]], sampler smp [[sampler(0)]]" in metal


@pytest.mark.parametrize(
    "generated, accepted",
    [
        ("void main() { x = ProcessTriTessFactorsAvg(); }", True),
        ("void main() { /* debugBreak() not currently supported for HLSL */; }", False),
    ],
)
def test_a_slang_probe_for_hlsl_is_written_as_hlsl_and_compiled_by_dxc(
    monkeypatch, generated, accepted
):
    commands = []

    def run(arguments, cwd, environment=None, stdout=True):
        commands.append(arguments)
        if arguments[0] == "slangc":
            Path(cwd, arguments[arguments.index("-o") + 1]).write_text(generated)
        return validators.Verdict(True)

    monkeypatch.setattr(validators, "command", lambda name: (name,))
    monkeypatch.setattr(validators, "run", run)
    verdict = validators.validate(
        "slang", '[shader("hull")]', "tessellation-control", "main", "hlsl"
    )
    assert verdict.accepted is accepted
    slangc = ["slangc", "probe.slang", "-entry", "main", "-target", "hlsl"]
    assert commands[0] == [*slangc, "-o", "probe.hlsl"]
    if accepted:
        profile = validators.HLSL_PROFILES["tessellation-control"]
        assert commands[1][:5] == ["dxc", "-T", profile, "-E", "main"]
        assert commands[1][-3:] == ["probe.hlsl", "-Fo", "probe.bin"]
    else:
        assert len(commands) == 1
        assert "debugBreak() not currently supported" in verdict.log


@pytest.mark.skipif(not validators.available("slang"), reason="slangc is not installed")
def test_slangc_validates_the_spirv_it_writes():
    # The half-precision atomic converts its pointer to an integer, which SPIR-V allows only for a physical storage
    # buffer pointer; slangc writes the conversion for a local's address and exits 0 unless told to validate.
    source = "\n".join(
        [
            "RWStructuredBuffer<half> outp;",
            '[shader("compute")]',
            "[numthreads(1, 1, 1)]",
            "void main(uint3 tid : SV_DispatchThreadID) {",
            "    half a0 = outp[tid.x];",
            "    half a1;",
            "    InterlockedAddF16Emulated(&a0, a0, a1);",
            "    outp[tid.x] = a1;",
            "}",
        ]
    )
    verdict = validators.validate("slang", source)
    assert not verdict.accepted
    assert verdict.log.startswith("error: line ")
    assert "PhysicalStorageBuffer" in verdict.log


def test_a_slang_probe_for_hlsl_needs_dxc(monkeypatch):
    monkeypatch.setattr(
        validators, "command", lambda name: (name,) if name == "slangc" else ()
    )
    assert validators.missing("slang") == ""
    assert validators.missing("slang", "hlsl") == "dxc"
    assert not validators.available("slang", "hlsl")


def test_hlsl_entry_falls_back_to_the_stage_convention():
    assert (
        validators.hlsl_entry("float4 PSMain() : SV_Target {}", "fragment") == "PSMain"
    )
    assert validators.hlsl_entry("float4 main() : SV_Target {}", "fragment") == "main"


def _templated(language: str) -> list[Feature]:
    return [
        f
        for f in constructs.features(language, GRAMMAR)
        if isinstance(templates.TEMPLATES[language].get(f.name), templates.Template)
    ]


@pytest.mark.parametrize(
    "language",
    [
        "cuda",
        "glsl",
        "hip",
        "hlsl",
        "metal",
        "mojo",
        "opencl",
        "rust",
        "slang",
        "webgl",
        "wgsl",
    ],
)
def test_native_templates_are_valid(language):
    if not validators.available(language):
        pytest.skip(f"{validators.TOOLS[language]} is not installed")
    rejected = []
    for feature in _templated(language):
        # A Slang template for the HLSL target also needs DXC; the runner reports such a probe as untested.
        if validators.missing(language, feature.target):
            continue
        probe = probes.build(feature, language)
        verdict = validators.validate(
            language, probe.source, feature.stage, probe.entry, feature.target
        )
        if not verdict.accepted:
            rejected.append(f"{feature.name}: {verdict.log[:200]}")
    assert not rejected


def _reads(source: str) -> set[str]:
    return set(re.findall(r"\b(fin|iin)\b", source))


def test_a_kernel_probe_declares_only_the_input_buffers_it_reads():
    # The harness that runs probes binds one input buffer at slot 0 and the output at slot 1.
    floats = probes.build(
        construct_feature("glsl", "conditional-expression"), "glsl"
    ).source
    assert _reads(floats) == {"fin"}
    assert "binding = 0) buffer B0 { float fin[]; };" in floats
    assert "binding = 1) buffer B1 { float outp[]; };" in floats
    ints = probes.build(construct_feature("glsl", "switch"), "glsl").source
    assert _reads(ints) == {"iin"} and "buffer B0 { int iin[]; };" in ints
    both = probes.build(construct_feature("glsl", "for-loop"), "glsl").source
    assert _reads(both) == {"fin", "iin"} and "buffer B2 { float outp[]; };" in both
    crossgl_probe = probes.build(construct_feature("glsl", "swizzle"), "crossgl").source
    assert _reads(crossgl_probe) == {"fin"}
    feature = construct_feature("hlsl", "swizzle")
    native = probes.build(feature, "hlsl").source
    assert execution.executable(
        Result(feature, "output", "supported", sources=(native, native))
    )


def test_a_kernel_template_can_give_the_output_element_type():
    # The output buffer holds float unless the template names another element type, in the language's spelling.
    vectors = probes.build(construct_feature("glsl", "vector-buffer"), "glsl").source
    assert "buffer B1 { vec4 outp[]; };" in vectors
    pairs = probes.build(construct_feature("wgsl", "struct-buffer"), "wgsl").source
    assert "array<Pair>" in pairs and "array<f32>" in pairs
    crossgl_probe = probes.build(
        construct_feature("glsl", "vector-buffer"), "crossgl"
    ).source
    assert "vec4* outp" in crossgl_probe


def test_reserved_name_probes_use_their_own_language_s_words_at_every_declaration_and_use():
    crossgl = probes.build(
        construct_feature("glsl", "reserved-type-name"), "crossgl"
    ).source
    assert crossgl.index("struct type {") < crossgl.index("compute {")
    assert "type make(float x)" in crossgl and "float weigh(type t)" in crossgl
    for language, declaration, use in (
        ("metal", "constant float var = 2.0;", "fin[i] * var + input[k]"),
        ("hlsl", "static const float loop = 2.0;", "fin[i] * loop + type[k]"),
        ("cuda", "__constant__ float var = 2.0f;", "mod[2 - k] * 1024.0f"),
        ("wgsl", "const half: f32 = 2.0;", "fin[i] * half + output[k]"),
        ("mojo", "comptime output: Array[Float32, 3]", "materialize[output]()"),
    ):
        source = probes.build(
            construct_feature(language, "reserved-global-name"), language
        ).source
        assert declaration in source and use in source
    opencl = probes.build(
        construct_feature("opencl", "reserved-type-name"), "opencl"
    ).source
    assert "} let;" in opencl and "let make(float x)" in opencl
    cuda = probes.build(
        construct_feature("cuda", "reserved-member-name"), "cuda"
    ).source
    assert (
        "__device__ float total(Rec r)" in cuda
        and "r.fn = (float)(iin[i] & 7);" in cuda
    )
    rust = probes.build(construct_feature("rust", "reserved-type-name"), "rust").source
    assert "pub struct var {" in rust and "var { a: x * 2.0, b: 3 }" in rust


def test_vector_probes_shift_by_scalars_and_resize_with_constructors():
    crossgl = probes.build(construct_feature("glsl", "vector-shift"), "crossgl").source
    assert "ivec2 r = a >> 1;" in crossgl and "uvec2 w = u << 3u;" in crossgl
    # WGSL shifts by an unsigned vector of the operand's width and has no scalar operand on a bitwise operator.
    wgsl = probes.build(construct_feature("wgsl", "vector-shift"), "wgsl").source
    assert "let r = a >> vec2<u32>(1u);" in wgsl
    wgsl = probes.build(construct_feature("wgsl", "vector-bitwise"), "wgsl").source
    assert "(a ^ vec2<i32>(9))" in wgsl and "~a & vec2<i32>(15)" in wgsl
    crossgl = probes.build(construct_feature("glsl", "matrix-resize"), "crossgl").source
    assert "mat2 a = mat2(m);" in crossgl and "mat2x3 n = mat2x3(m);" in crossgl
    # HLSL casts and indexes by rows; Slang casts only matrices; Metal takes swizzles and columns.
    hlsl = probes.build(construct_feature("hlsl", "matrix-resize"), "hlsl").source
    assert "float2x2 a = (float2x2)m;" in hlsl and "n[2][1] * 8192.0" in hlsl
    for language, line in (
        ("hlsl", "float2 a = (float2)v;"),
        ("slang", "float2 a = v.xy;"),
        ("metal", "int2 j = k.xy;"),
        ("rust", "let b = v.truncate();"),
    ):
        source = probes.build(
            construct_feature(language, "vector-truncation"), language
        ).source
        assert line in source, language
    metal = probes.build(construct_feature("metal", "matrix-resize"), "metal").source
    assert "float2x2 a = float2x2(m[0].xy, m[1].xy);" in metal
    rust = probes.build(construct_feature("rust", "matrix-resize"), "rust").source
    assert "Mat3::from_mat2(a * 2.0)" in rust and "n[1].z * 8192.0" in rust
    for language, absent in (
        ("cuda", {"vector-shift", "vector-truncation", "matrix-resize"}),
        ("opencl", {"matrix-resize"}),
        ("mojo", {"vector-truncation", "matrix-resize"}),
    ):
        names = {f.name for f in constructs.features(language, GRAMMAR)}
        assert not names & absent, language


def test_relation_probes_compare_integer_and_boolean_vectors_per_language():
    # CrossGL and the C family compare with operators; GLSL spells each comparison and negation as a function.
    crossgl = probes.build(
        construct_feature("glsl", "integer-vector-comparison"), "crossgl"
    ).source
    assert "bvec2 gt = u > uvec2(5u, 7u);" in crossgl and "if (all(ne)) {" in crossgl
    glsl = probes.build(
        construct_feature("glsl", "integer-vector-comparison"), "glsl"
    ).source
    assert "bvec2 gt = greaterThan(u, uvec2(5u, 7u));" in glsl
    glsl = probes.build(construct_feature("glsl", "bool-vector-logic"), "glsl").source
    assert "bvec2 n = not(b);" in glsl and "if (any(not(c))) {" in glsl
    crossgl = probes.build(
        construct_feature("glsl", "bool-vector-logic"), "crossgl"
    ).source
    assert "bvec2 e = b == c;" in crossgl and "if (any(!c)) {" in crossgl
    for language, line in (
        ("hlsl", "bool2 b = (bool2)k;"),
        ("metal", "bool2 b = bool2(k);"),
        ("wgsl", "let u = vec2<u32>(vec2<bool>(fin[i] > 1.0, true));"),
        ("rust", "let g = Vec2::select(b, Vec2::ONE, Vec2::ZERO);"),
        ("mojo", "var j = c.cast[DType.int32]()"),
    ):
        source = probes.build(
            construct_feature(language, "bool-vector-conversion"), language
        ).source
        assert line in source, language
    opencl = probes.build(
        construct_feature("opencl", "integer-vector-comparison"), "opencl"
    ).source
    assert "int2 gt = u > (uint2)(5u, 7u);" in opencl and "r += 128.0f;" in opencl
    rust = probes.build(construct_feature("rust", "bool-vector-logic"), "rust").source
    assert "let e = !(b ^ c);" in rust and "if (!c).any() {" in rust
    for language, absent in (
        ("cuda", set(relations.NAMES)),
        ("opencl", {"bool-vector-logic", "bool-vector-conversion"}),
    ):
        names = {f.name for f in constructs.features(language, GRAMMAR)}
        assert not names & absent, language


# glslang folds the swizzle probe's v.wzyx.x to one OpCompositeExtract.
_UNANCHORED_TYPES = {"swizzle"}


# GLSL has no float remainder with the dividend's sign, so its probe computes one through trunc, and no typedef,
# enumeration, member function or boolean bitwise operator; DXC lowers the HLSL probes of all five.
_HLSL_ANCHORED = {
    "float-remainder",
    "type-alias",
    "enum-constant",
    "struct-method",
    "bool-bitwise",
}


def _missing_anchors(language: str, names: set[str] | None = None) -> list[str]:
    """The anchors of the operator and type constructs (or of ``names``) that ``language``'s probe lowers without."""
    missing = []
    for construct in constructs.universe(GRAMMAR):
        if construct.category not in ("operator", "type"):
            continue
        if names is None and construct.name in _UNANCHORED_TYPES | _HLSL_ANCHORED:
            continue
        if names is not None and construct.name not in names:
            continue
        if not constructs.spells(construct.name, language):
            continue
        probe = probes.build(construct_feature(language, construct.name), language)
        assembly, log = lowering.lower(
            language, probe.source, probe.entry, "compute", raw_ids=False
        )
        assert assembly, log
        found = set(re.findall(r"\bOp\w+", assembly))
        missing += [
            f"{construct.name}: {a}"
            for a in construct.anchors
            if a.startswith("Op") and a not in found
        ]
    return missing


def test_hlsl_anchored_probes_compile_to_their_anchors():
    if not lowering.can_lower("hlsl"):
        pytest.skip("dxc or spirv-dis is not installed")
    assert not _missing_anchors("hlsl", _HLSL_ANCHORED)


def test_operator_and_type_probes_compile_to_their_anchors():
    # glslang lowers the GLSL probe of each operator and type to the instructions the construct is anchored to; it
    # writes a signed remainder as OpSMod, which the probes' non-negative operands make equal to OpSRem. Scalar
    # widths are anchored to capabilities, which this leaves out.
    if not lowering.can_lower("glsl"):
        pytest.skip("glslangValidator or spirv-dis is not installed")
    assert not _missing_anchors("glsl")


def test_c_operator_probes_cast_rather_than_construct():
    # CUDA, HIP and OpenCL C have no constructor casts, and CUDA spells uint as unsigned int.
    for language in ("cuda", "hip", "opencl"):
        table = templates.TEMPLATES[language]
        bodies = [table[name].body for name in OPERATORS if table[name] is not ABSENT]
        assert not [
            b for b in bodies if re.search(r"\b(float|int|uint)\(", b)
        ], language
    unsigned = templates.TEMPLATES["cuda"]["unsigned-division"].body
    assert unsigned.startswith("unsigned int a = (unsigned int)iin[i] + 7u;")

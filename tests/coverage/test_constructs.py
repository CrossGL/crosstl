import json
import re
from pathlib import Path

import pytest

from tools.coverage import (
    constructs,
    crossgl,
    execution,
    probes,
    semantic,
    templates,
    validators,
)
from tools.coverage.model import Feature, Result
from tools.coverage.templates.base import ABSENT, OPERATORS

_MODELS = [
    ("Vertex", ["Shader"]),
    ("TessellationControl", ["Tessellation"]),
    ("TessellationEvaluation", ["Tessellation"]),
    ("Geometry", ["Geometry"]),
    ("Fragment", ["Shader"]),
    ("GLCompute", ["Shader"]),
    ("Kernel", ["Kernel"]),
    ("TaskNV", ["MeshShadingNV"]),
    ("TaskEXT", ["MeshShadingEXT"]),
    ("MeshEXT", ["MeshShadingEXT"]),
    ("RayGenerationKHR", ["RayTracingNV", "RayTracingKHR"]),
    ("IntersectionKHR", ["RayTracingNV", "RayTracingKHR"]),
    ("AnyHitKHR", ["RayTracingNV", "RayTracingKHR"]),
    ("ClosestHitKHR", ["RayTracingNV", "RayTracingKHR"]),
    ("MissKHR", ["RayTracingNV", "RayTracingKHR"]),
    ("CallableKHR", ["RayTracingNV", "RayTracingKHR"]),
]


def _grammar(drop: str = "") -> str:
    """A grammar with exactly the items the constructs anchor to, less ``drop``."""
    anchors = {a for c in constructs._curated() for a in c.anchors}
    anchors |= {f"Capability.{name}" for name in constructs._WIDTHS}
    kinds: dict[str, list[dict]] = {
        "Capability": [],
        "StorageClass": [],
        "BuiltIn": [],
        "Decoration": [],
        "ExecutionModel": [],
    }
    instructions = []
    for anchor in sorted(anchors - {drop}):
        kind, _, name = anchor.rpartition(".")
        if kind:
            kinds[kind].append({"enumerant": name})
        else:
            instructions.append({"opname": name})
    kinds["ExecutionModel"] = [
        {"enumerant": name, "capabilities": caps} for name, caps in _MODELS
    ]
    return json.dumps(
        {
            "instructions": instructions,
            "operand_kinds": [
                {"kind": kind, "enumerants": values} for kind, values in kinds.items()
            ],
        }
    )


GRAMMAR = _grammar()


def _feature(language: str, name: str) -> Feature:
    (feature,) = (f for f in constructs.features(language, GRAMMAR) if f.name == name)
    return feature


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
        constructs.universe(_grammar(drop="OpSwitch"))


def test_a_language_lists_only_what_it_can_spell():
    wgsl = {f.name for f in constructs.features("wgsl", GRAMMAR)}
    assert {"switch", "half", "vertex"} <= wgsl
    assert not {"double", "int64", "geometry", "any-hit"} & wgsl
    assert "num-workgroups" not in {
        f.name for f in constructs.features("hlsl", GRAMMAR)
    }


def test_spirv_lists_every_construct():
    spirv = constructs.features("spirv", GRAMMAR)
    assert len(spirv) == len(constructs.universe(GRAMMAR))


def test_constructs_share_a_semantic_id_across_languages():
    ids = {
        semantic.lower(_feature(lang, "switch"))[0]
        for lang in ("glsl", "wgsl", "spirv")
    }
    assert ids == {"construct:control-flow/switch"}


def test_every_curated_construct_has_a_crossgl_template():
    assert {c.name for c in constructs._curated()} <= set(crossgl.CONSTRUCTS)


def test_a_construct_without_a_template_is_untested_not_absent(monkeypatch):
    feature = _feature("glsl", "tessellation-control")
    monkeypatch.delitem(templates.TEMPLATES["glsl"], "tessellation-control")
    with pytest.raises(probes.ProbeError, match="no glsl probe template"):
        probes.build(feature, "glsl")


def test_kernel_templates_fill_the_language_spelling_of_types():
    hlsl = probes.build(_feature("hlsl", "swizzle"), "hlsl").source
    assert "float4 v = float4(" in hlsl
    crossgl_probe = probes.build(_feature("glsl", "struct"), "crossgl").source
    assert crossgl_probe.index("struct Pair") < crossgl_probe.index("compute {")


def test_stage_programs_take_the_stage_suffix():
    probe = probes.build(_feature("glsl", "fragment"), "glsl")
    assert probe.suffix == ".frag"
    assert probes.build(_feature("glsl", "geometry"), "glsl").suffix == ".geom"
    assert probes.build(_feature("glsl", "discard"), "glsl").suffix == ".frag"
    assert probes.build(_feature("glsl", "closest-hit"), "glsl").suffix == ".rchit"
    tese = probes.build(_feature("glsl", "tessellation-evaluation"), "glsl")
    assert tese.suffix == ".tese"
    assert probes.build(_feature("glsl", "task"), "glsl").suffix == ".task"
    assert probes.build(_feature("glsl", "mesh"), "glsl").suffix == ".mesh"


def test_slang_names_the_hull_stage_on_its_entry_point():
    hull = probes.build(_feature("slang", "tessellation-control"), "slang").source
    assert '[shader("hull")]\n[domain("tri")]' in hull
    task = probes.build(_feature("slang", "task"), "slang").source
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
    monkeypatch.setattr(validators, "_run", run)
    verdict = validators.validate("metal", "", stage)
    assert verdict.accepted is accepted
    assert accepted or f"no {stage} entry point" in verdict.log


def test_a_metal_intersection_probe_is_a_bounding_box_intersection_function():
    probe = probes.build(_feature("metal", "intersection"), "metal")
    assert "using namespace raytracing;" in probe.source
    assert "[[intersection(bounding_box)]]\nBoxResult probe(" in probe.source


def test_ray_tracing_and_mesh_stages_are_checked_for_vulkan(monkeypatch):
    commands = []

    def run(arguments, cwd):
        commands.append(arguments)
        return validators.Verdict(True)

    monkeypatch.setattr(validators, "command", lambda name: (name,))
    monkeypatch.setattr(validators, "_run", run)
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
    wgsl = probes.build(_feature("wgsl", "local-invocation-id"), "wgsl").source
    assert "id: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>)" in wgsl
    metal = probes.build(_feature("metal", "sampled-texture"), "metal").source
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
    monkeypatch.setattr(validators, "_run", run)
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
        probe = probes.build(feature, language)
        verdict = validators.validate(
            language, probe.source, feature.stage, probe.entry
        )
        if not verdict.accepted:
            rejected.append(f"{feature.name}: {verdict.log[:200]}")
    assert not rejected


def _reads(source: str) -> set[str]:
    return set(re.findall(r"\b(fin|iin)\b", source))


def test_a_kernel_probe_declares_only_the_input_buffers_it_reads():
    # The harness that runs probes binds one input buffer at slot 0 and the output at slot 1.
    floats = probes.build(_feature("glsl", "conditional-expression"), "glsl").source
    assert _reads(floats) == {"fin"}
    assert "binding = 0) buffer B0 { float fin[]; };" in floats
    assert "binding = 1) buffer B1 { float outp[]; };" in floats
    ints = probes.build(_feature("glsl", "switch"), "glsl").source
    assert _reads(ints) == {"iin"} and "buffer B0 { int iin[]; };" in ints
    both = probes.build(_feature("glsl", "for-loop"), "glsl").source
    assert _reads(both) == {"fin", "iin"} and "buffer B2 { float outp[]; };" in both
    crossgl_probe = probes.build(_feature("glsl", "swizzle"), "crossgl").source
    assert _reads(crossgl_probe) == {"fin"}
    feature = _feature("hlsl", "swizzle")
    native = probes.build(feature, "hlsl").source
    assert execution.executable(
        Result(feature, "output", "supported", sources=(native, native))
    )


def test_a_kernel_template_can_give_the_output_element_type():
    # The output buffer holds float unless the template names another element type, in the language's spelling.
    vectors = probes.build(_feature("glsl", "vector-buffer"), "glsl").source
    assert "buffer B1 { vec4 outp[]; };" in vectors
    pairs = probes.build(_feature("wgsl", "struct-buffer"), "wgsl").source
    assert "array<Pair>" in pairs and "array<f32>" in pairs
    crossgl_probe = probes.build(_feature("glsl", "vector-buffer"), "crossgl").source
    assert "vec4* outp" in crossgl_probe


def test_a_uniform_float_array_reads_one_word_per_row_in_every_language():
    # std140 and a cbuffer give each float its own 16-byte row; languages that pack a float array declare vectors.
    padded = probes.build(_feature("glsl", "uniform-array"), "glsl").source
    assert "float w[4];" in padded and "+ w[k])" in padded
    for language, declaration in (
        ("metal", "float4 w[4];"),
        ("wgsl", "w: array<vec4<f32>, 4>,"),
        ("cuda", "float4 w[4];"),
        ("rust", "w: [Vec4; 4],"),
    ):
        source = probes.build(_feature(language, "uniform-array"), language).source
        assert declaration in source and "w[k].x" in source


def test_a_uniform_struct_is_declared_before_its_block_and_copied_into_a_mutable_local():
    glsl = probes.build(_feature("glsl", "uniform-struct"), "glsl").source
    assert glsl.index("struct Light {") < glsl.index("uniform Params {")
    assert "    Light light;\n    vec4 tint;" in glsl
    for language, copy in (
        ("wgsl", "var l = params.light;"),
        ("rust", "let mut l = params.light;"),
        ("mojo", "var l = params.light\n"),
        ("hlsl", "Light l = light;"),
    ):
        source = probes.build(
            _feature(language, "uniform-struct-copy"), language
        ).source
        assert copy in source
    # Mojo reads a vector's lanes by index.
    mojo = probes.build(_feature("mojo", "uniform-struct-array"), "mojo").source
    assert "params.lights[k].color[0]" in mojo and "var lights: Array[Light, 2]" in mojo


def test_reserved_name_probes_use_their_own_language_s_words_at_every_declaration_and_use():
    crossgl = probes.build(_feature("glsl", "reserved-type-name"), "crossgl").source
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
            _feature(language, "reserved-global-name"), language
        ).source
        assert declaration in source and use in source
    opencl = probes.build(_feature("opencl", "reserved-type-name"), "opencl").source
    assert "} let;" in opencl and "let make(float x)" in opencl
    cuda = probes.build(_feature("cuda", "reserved-member-name"), "cuda").source
    assert (
        "__device__ float total(Rec r)" in cuda
        and "r.fn = (float)(iin[i] & 7);" in cuda
    )
    rust = probes.build(_feature("rust", "reserved-type-name"), "rust").source
    assert "pub struct var {" in rust and "var { a: x * 2.0, b: 3 }" in rust


def test_uniform_matrix_probes_read_every_column_from_its_own_16_byte_row():
    crossgl = probes.build(_feature("glsl", "uniform-matrix3"), "crossgl").source
    assert "cbuffer Params @binding(3) {\n        mat3 m;\n        vec4 v;" in crossgl
    # Metal, WGSL and Rust give a 2x2 matrix an 8-byte column stride, so they build it from 16-byte vectors.
    for language, member, build in (
        (
            "metal",
            "float4 c[2];",
            "float2x2 m = float2x2(params.c[0].xy, params.c[1].xy);",
        ),
        (
            "wgsl",
            "c: array<vec4<f32>, 2>,",
            "let m = mat2x2<f32>(params.c[0].xy, params.c[1].xy);",
        ),
        (
            "rust",
            "c: [Vec4; 2],",
            "let m = Mat2::from_cols(Vec2::new(params.c[0].x, params.c[0].y)",
        ),
    ):
        source = probes.build(_feature(language, "uniform-matrix2"), language).source
        assert member in source and build in source, language
    rust = probes.build(_feature("rust", "uniform-matrix3"), "rust").source
    assert "c: [Vec4; 3],\n    v: Vec4," in rust and "m.z_axis.y * params.v.w" in rust
    # HLSL names CrossGL's mat2x3 float3x2 and swaps the indices.
    hlsl = probes.build(_feature("hlsl", "uniform-non-square-matrix"), "hlsl").source
    assert "float3x2 m;" in hlsl and "mul(v.xyz, m)" in hlsl and "m[2][1] * v.w" in hlsl
    wgsl = probes.build(_feature("wgsl", "uniform-non-square-matrix"), "wgsl").source
    assert "m: mat2x3<f32>," in wgsl and "params.m[1][2] * params.v.w" in wgsl
    for language in ("cuda", "hip", "opencl", "mojo"):
        assert not [
            f
            for f in constructs.features(language, GRAMMAR)
            if f.name.startswith("uniform-matrix")
            or f.name == "uniform-non-square-matrix"
        ], language


def test_uniform_bool_probes_store_each_boolean_in_a_32_bit_word():
    crossgl = probes.build(_feature("glsl", "uniform-bool-vector"), "crossgl").source
    assert (
        "cbuffer Params @binding(3) {\n        bvec4 f;\n        bvec2 g;\n        bool h;"
        in crossgl
    )
    hlsl = probes.build(_feature("hlsl", "uniform-bool-vector"), "hlsl").source
    assert "bool4 f;\n    bool2 g;" in hlsl and "bool4 c = f;" in hlsl
    # Metal's bool is one byte and the others cannot put a boolean in a host-written block, so they test a uint.
    for language, member, test in (
        ("metal", "uint b0;", "bool b0 = bool(params.b0);"),
        ("wgsl", "b0: u32,", "let b0 = bool(params.b0);"),
        ("cuda", "unsigned int b0;", "bool b0 = params.b0 != 0u;"),
        ("opencl", "uint b0;", "bool b0 = params.b0 != 0u;"),
        ("rust", "b0: u32,", "let b0 = params.b0 != 0;"),
        ("mojo", "var b0: UInt32", "var b0 = params.b0 != 0"),
    ):
        source = probes.build(_feature(language, "uniform-bool"), language).source
        assert member in source and test in source, language
    for language, member, test in (
        ("metal", "uint4 f;\n    uint2 g;", "bool4 f = bool4(params.f);"),
        ("wgsl", "f: vec4<u32>,\n    g: vec2<u32>,", "var c = vec4<bool>(params.f);"),
        ("rust", "f: UVec4,\n    g: UVec2,", "params.f.cmpne(UVec4::ZERO)"),
        ("mojo", "var f: SIMD[DType.uint32, 4]", "params.f.cast[DType.bool]()"),
    ):
        source = probes.build(
            _feature(language, "uniform-bool-vector"), language
        ).source
        assert member in source and test in source, language
    for language in ("cuda", "hip", "opencl"):
        names = {f.name for f in constructs.features(language, GRAMMAR)}
        assert "uniform-bool" in names and "uniform-bool-vector" not in names, language


def test_row_major_probes_store_each_matrix_row_on_its_own_16_byte_row():
    crossgl = probes.build(
        _feature("glsl", "uniform-row-major-non-square"), "crossgl"
    ).source
    assert (
        "cbuffer Params @binding(3) {\n        @ row_major mat2x3 m;\n        vec4 v;"
        in crossgl
    )
    glsl = probes.build(_feature("glsl", "uniform-row-major-array"), "glsl").source
    assert "layout(row_major) mat2 m[2];" in glsl and "m[0][1][0] * 8.0" in glsl
    # HLSL indexes by rows, so CrossGL's element in column 1 and row 2 is m[2][1].
    hlsl = probes.build(_feature("hlsl", "uniform-row-major-non-square"), "hlsl").source
    assert "row_major float3x2 m;\n    float4 v;" in hlsl and "m[2][1] * v.w" in hlsl
    # Metal, WGSL and Rust have no row-major matrices, so they transpose the matrix whose columns are the rows.
    for language, member, build in (
        (
            "metal",
            "float4 r[3];\n    float4 v;",
            "float3x3 m = transpose(float3x3(params.r[0].xyz,",
        ),
        (
            "wgsl",
            "r: array<vec4<f32>, 3>,\n    v: vec4<f32>,",
            "let m = transpose(mat3x3<f32>(params.r[0].xyz,",
        ),
        (
            "rust",
            "r: [Vec4; 3],\n    v: Vec4,",
            ".transpose();\n    let t = m * params.v",
        ),
    ):
        source = probes.build(
            _feature(language, "uniform-row-major-matrix"), language
        ).source
        assert member in source and build in source, language
    names = {f.name for f in constructs.features("rust", GRAMMAR)}
    assert "uniform-row-major-array" in names
    assert "uniform-row-major-non-square" not in names
    for language in ("cuda", "hip", "opencl", "mojo"):
        assert not [
            f
            for f in constructs.features(language, GRAMMAR)
            if f.name.startswith("uniform-row-major")
        ], language


def test_packing_probes_name_the_layout_rule_and_declare_its_offsets():
    for name, rule in (
        ("uniform-std140-vector", "std140"),
        ("uniform-cbuffer-vector", "hlsl_cbuffer"),
        ("uniform-std140-array", "std140"),
    ):
        crossgl = probes.build(_feature("glsl", name), "crossgl").source
        assert f"cbuffer Params @binding(3) @buffer_layout({rule}) {{" in crossgl, name
    # slangc lays out a plain cbuffer by std140 for SPIR-V, so HLSL and Slang name every member's offset.
    for language in ("hlsl", "slang"):
        std140 = probes.build(_feature(language, "uniform-std140-vector"), language)
        assert (
            "float3 b : packoffset(c1);\n    float c : packoffset(c1.w);"
            in std140.source
        )
        cbuffer = probes.build(_feature(language, "uniform-cbuffer-vector"), language)
        assert (
            "float3 b : packoffset(c0.y);\n    float c : packoffset(c1);"
            in cbuffer.source
        )
        array = probes.build(_feature(language, "uniform-std140-array"), language)
        assert "float d : packoffset(c2);" in array.source
    # No GLSL, WGSL or OpenCL C rule puts a three-component vector at offset 4, so those declare its components.
    for language, member, build in (
        ("glsl", "float b0;", "vec3 b = vec3(b0, b1, b2);"),
        ("wgsl", "b0: f32,", "let b = vec3<f32>(params.b0, params.b1, params.b2);"),
        (
            "opencl",
            "float b0;",
            "float3 b = (float3)(params.b0, params.b1, params.b2);",
        ),
    ):
        source = probes.build(_feature(language, "uniform-cbuffer-vector"), language)
        assert member in source.source and build in source.source, language
    # The languages without std140 rules declare its padding.
    for language, padded in (
        ("metal", "float pad2;\n    packed_float3 b;"),
        ("cuda", "float pad2;\n    float3 b;"),
        ("rust", "pad2: f32,\n    b: Vec3,"),
        ("mojo", "var pad2: Float32\n    var b0: Float32"),
    ):
        source = probes.build(_feature(language, "uniform-std140-vector"), language)
        assert padded in source.source and "pad3" in source.source, language


def test_uniform_integer_probes_keep_each_member_s_signedness_in_every_language():
    crossgl = probes.build(_feature("glsl", "uniform-integer"), "crossgl").source
    assert "cbuffer Params @binding(3) {\n        int n0;" in crossgl
    for language, member, use in (
        ("wgsl", "u1: u32,", "f32(params.u2 % 1000u + (params.u3 >> 24))"),
        ("cuda", "unsigned int u1;", "(float)(params.u2 % 1000u + (params.u3 >> 24))"),
        ("opencl", "uint u1;", "(float)(params.u2 % 1000u + (params.u3 >> 24))"),
        ("rust", "u1: u32,", "(params.u2 % 1000 + (params.u3 >> 24)) as f32"),
        ("mojo", "var u1: UInt32", "Float32(params.u2 % 1000 + (params.u3 >> 24))"),
    ):
        source = probes.build(_feature(language, "uniform-unsigned"), language).source
        assert member in source and use in source, language
    mojo = probes.build(_feature("mojo", "uniform-integer"), "mojo").source
    assert "params.n3 // 1000000" in mojo
    for language, local in (
        ("wgsl", "let v = params.iv;"),
        ("mojo", "var v = params.iv"),
    ):
        vector = probes.build(_feature(language, "uniform-integer-vector"), language)
        assert local in vector.source and "$" not in vector.source, language
    assert "(v[0] >> 20) + (v[1] >> 20) + (v[3] >> 24)" in vector.source


# Each HLSL matrix probe with its matrices written as CrossGL lists them, by columns, and transposed into HLSL's rows.
# The indexing probe keeps CrossGL's columns as HLSL rows instead, and indexes them as CrossGL does.
_HLSL_BY_COLUMNS = {
    "matrix-indexing": (
        [("int", "iin"), ("float", "outp")],
        [
            "float f = float(iin[i]);",
            "float2x2 m = float2x2(f, 1.0, 2.0, 3.0);",
            "float2 c = m[iin[i] & 1];",
            "outp[i] = m[1][0] * 100.0 + m[0].y * 10.0 + c.x + 0.5 * c.y;",
        ],
    ),
    "matrix-from-vectors": (
        [("int", "iin"), ("float", "outp")],
        [
            "float f = float(iin[i]);",
            "float2x2 m = transpose(float2x2(float2(f, 1.0), float2(2.0, 3.0)));",
            "outp[i] = m[1][0] * 100.0 + m[0][1] * 10.0 + m[0][0];",
        ],
    ),
    "matrix-vector-product": (
        [("float", "fin"), ("float", "outp")],
        [
            "float2x2 m = transpose(float2x2(fin[i], 1.0, 2.0, 3.0));",
            "float2 v = mul(m, float2(1.0, fin[i]));",
            "outp[i] = v.y;",
        ],
    ),
    "matrix-product": (
        [("int", "iin"), ("float", "outp")],
        [
            "float f = float(iin[i]);",
            "float2x2 m = transpose(float2x2(f, 1.0, 2.0, 3.0));",
            "float2x2 n = transpose(float2x2(1.0, f, 0.5, 2.0));",
            "float2 v = mul(mul(m, n), float2(1.0, f));",
            "outp[i] = v.x + 2.0 * v.y;",
        ],
    ),
    "vector-matrix-product": (
        [("int", "iin"), ("float", "outp")],
        [
            "float f = float(iin[i]);",
            "float2x2 m = transpose(float2x2(f, 1.0, 2.0, 3.0));",
            "float2 v = mul(float2(1.0, f), m);",
            "outp[i] = v.x + 2.0 * v.y;",
        ],
    ),
}


@pytest.mark.parametrize("name", sorted(_HLSL_BY_COLUMNS))
def test_the_hlsl_matrix_probes_compute_what_the_crossgl_probes_compute(name):
    # HLSL fills and indexes a matrix by rows and CrossGL by columns, so the probes must write them differently.
    if not validators.can_lower("hlsl"):
        pytest.skip("dxc or spirv-dis is not installed")
    feature = _feature("hlsl", name)
    native = probes.build(feature, "hlsl").source
    by_columns = probes.DIALECTS["hlsl"].kernel(*_HLSL_BY_COLUMNS[name])
    probe = Result(feature, "output", "supported", sources=(native, by_columns))
    [result], used = execution.run([probe], workers=1)
    if not used:
        pytest.skip("no Vulkan runtime")
    assert (result.outcome, result.runtime) == ("supported", used[0])


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
        probe = probes.build(_feature(language, construct.name), language)
        assembly, log = validators.lower(
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
    if not validators.can_lower("hlsl"):
        pytest.skip("dxc or spirv-dis is not installed")
    assert not _missing_anchors("hlsl", _HLSL_ANCHORED)


def test_operator_and_type_probes_compile_to_their_anchors():
    # glslang lowers the GLSL probe of each operator and type to the instructions the construct is anchored to; it
    # writes a signed remainder as OpSMod, which the probes' non-negative operands make equal to OpSRem. Scalar
    # widths are anchored to capabilities, which this leaves out.
    if not validators.can_lower("glsl"):
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

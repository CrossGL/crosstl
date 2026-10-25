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


def test_operator_and_type_probes_compile_to_their_anchors():
    # glslang lowers the GLSL probe of each operator and type to the instructions the construct is anchored to; it
    # writes a signed remainder as OpSMod, which the probes' non-negative operands make equal to OpSRem. Scalar
    # widths are anchored to capabilities, which this leaves out.
    if not validators.can_lower("glsl"):
        pytest.skip("glslangValidator or spirv-dis is not installed")
    missing = []
    for construct in constructs.universe(GRAMMAR):
        if construct.category not in ("operator", "type"):
            continue
        if construct.name in _UNANCHORED_TYPES:
            continue
        probe = probes.build(_feature("glsl", construct.name), "glsl")
        assembly, log = validators.lower(
            "glsl", probe.source, probe.entry, "compute", raw_ids=False
        )
        assert assembly, log
        found = set(re.findall(r"\bOp\w+", assembly))
        missing += [
            f"{construct.name}: {a}"
            for a in construct.anchors
            if a.startswith("Op") and a not in found
        ]
    assert not missing


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

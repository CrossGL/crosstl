import json

import pytest

from tools.coverage import constructs, crossgl, probes, semantic, templates, validators
from tools.coverage.model import Feature

_MODELS = [
    ("Vertex", ["Shader"]),
    ("TessellationControl", ["Tessellation"]),
    ("Geometry", ["Geometry"]),
    ("Fragment", ["Shader"]),
    ("GLCompute", ["Shader"]),
    ("Kernel", ["Kernel"]),
    ("TaskNV", ["MeshShadingNV"]),
    ("MeshEXT", ["MeshShadingEXT"]),
    ("AnyHitKHR", ["RayTracingNV", "RayTracingKHR"]),
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
        "geometry",
        "fragment",
        "compute",
        "mesh",
        "any-hit",
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


def test_a_construct_without_a_template_is_untested_not_absent():
    feature = _feature("glsl", "geometry")
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
    assert probes.build(_feature("glsl", "discard"), "glsl").suffix == ".frag"


def test_entry_parameters_join_the_kernel_signature():
    wgsl = probes.build(_feature("wgsl", "local-invocation-id"), "wgsl").source
    assert "id: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>)" in wgsl
    metal = probes.build(_feature("metal", "sampled-texture"), "metal").source
    assert "texture2d<float> tex [[texture(0)]], sampler smp [[sampler(0)]]" in metal


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
    "language", ["cuda", "glsl", "hip", "hlsl", "metal", "mojo", "slang", "wgsl"]
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

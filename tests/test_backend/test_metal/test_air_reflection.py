"""The host interface of Metal entry points read from AIR metadata (tools/coverage/metal_reflection.py)."""

from __future__ import annotations

import shutil
import subprocess

import pytest

from tools.coverage import metal_reflection as reflection

# The metadata the Metal compiler writes for a vertex function with two attributes, a constant buffer, an argument
# buffer of a texture, a sampler and a device pointer, and a vertex_id; and for a fragment function with two render
# targets, a depth output, a texture, a sampler and a device atomic buffer (Apple metal 32023, -std=metal3.1).
AIR = r"""
!air.vertex = !{!9}
!air.fragment = !{!25}
!air.compile_options = !{!38}
!9 = !{<{ <4 x float>, <2 x float>, float }> (<3 x float>, <2 x float>, %struct.U addrspace(2)*, i32, %struct.AB addrspace(2)*)* @vmain, !10, !14}
!10 = !{!11, !12, !13}
!11 = !{!"air.position", !"air.arg_type_name", !"float4", !"air.arg_name", !"pos"}
!12 = !{!"air.vertex_output", !"user(locn1)", !"air.arg_type_name", !"float2", !"air.arg_name", !"uv"}
!13 = !{!"air.vertex_output", !"generated(1kf)", !"air.arg_type_name", !"float", !"air.arg_name", !"k"}
!14 = !{!15, !16, !17, !19, !20}
!15 = !{i32 0, !"air.vertex_input", !"air.location_index", i32 0, i32 1, !"air.arg_type_name", !"float3", !"air.arg_name", !"p"}
!16 = !{i32 1, !"air.vertex_input", !"air.location_index", i32 3, i32 1, !"air.arg_type_name", !"float2", !"air.arg_name", !"uv"}
!17 = !{i32 2, !"air.buffer", !"air.buffer_size", i32 80, !"air.location_index", i32 1, i32 1, !"air.read", !"air.address_space", i32 2, !"air.struct_type_info", !18, !"air.arg_type_size", i32 80, !"air.arg_type_align_size", i32 16, !"air.arg_type_name", !"U", !"air.arg_name", !"u"}
!18 = !{i32 0, i32 64, i32 0, !"float4x4", !"m", i32 64, i32 4, i32 0, !"float", !"s"}
!19 = !{i32 3, !"air.vertex_id", !"air.arg_type_name", !"uint", !"air.arg_name", !"vid"}
!20 = !{i32 4, !"air.indirect_buffer", !"air.buffer_size", i32 24, !"air.location_index", i32 2, i32 1, !"air.read", !"air.address_space", i32 2, !"air.struct_type_info", !21, !"air.arg_type_size", i32 24, !"air.arg_type_align_size", i32 8, !"air.arg_type_name", !"AB", !"air.arg_name", !"ab"}
!21 = !{i32 0, i32 8, i32 0, !"texture2d<float, sample>", !"t", !"air.indirect_argument", !22, i32 8, i32 8, i32 0, !"sampler", !"s", !"air.indirect_argument", !23, i32 16, i32 8, i32 0, !"float", !"d", !"air.indirect_argument", !24}
!22 = !{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.sample", !"air.arg_type_name", !"texture2d<float, sample>", !"air.arg_name", !"t"}
!23 = !{i32 1, !"air.sampler", !"air.location_index", i32 1, i32 1, !"air.arg_type_name", !"sampler", !"air.arg_name", !"s"}
!24 = !{i32 2, !"air.buffer", !"air.location_index", i32 2, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"float", !"air.arg_name", !"d"}
!25 = !{<{ <4 x float>, <4 x float>, float }> (<4 x float>, <2 x float>, float, %struct._texture_2d_t addrspace(1)*, %struct._sampler_t addrspace(2)*, %"struct.metal::_atomic" addrspace(1)*)* @fmain, !26, !30}
!26 = !{!27, !28, !29}
!27 = !{!"air.render_target", i32 0, i32 0, !"air.arg_type_name", !"float4", !"air.arg_name", !"c"}
!28 = !{!"air.render_target", i32 2, i32 0, !"air.arg_type_name", !"float4", !"air.arg_name", !"c2"}
!29 = !{!"air.depth", !"air.depth_qualifier", !"air.any", !"air.arg_type_name", !"float", !"air.arg_name", !"d"}
!30 = !{!31, !32, !33, !34, !35, !36}
!31 = !{i32 0, !"air.position", !"air.center", !"air.no_perspective", !"air.arg_type_name", !"float4", !"air.arg_name", !"pos", !"air.arg_unused"}
!32 = !{i32 1, !"air.fragment_input", !"user(locn1)", !"air.center", !"air.perspective", !"air.arg_type_name", !"float2", !"air.arg_name", !"uv"}
!33 = !{i32 2, !"air.fragment_input", !"generated(1kf)", !"air.flat", !"air.arg_type_name", !"float", !"air.arg_name", !"k"}
!34 = !{i32 3, !"air.texture", !"air.location_index", i32 3, i32 1, !"air.sample", !"air.arg_type_name", !"texture2d<float, sample>", !"air.arg_name", !"t"}
!35 = !{i32 4, !"air.sampler", !"air.location_index", i32 1, i32 1, !"air.arg_type_name", !"sampler", !"air.arg_name", !"s"}
!36 = !{i32 5, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.struct_type_info", !37, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"metal::_atomic", !"air.arg_name", !"cnt"}
!37 = !{i32 0, i32 4, i32 0, !"uint", !"__s"}
!38 = !{!"air.compile.fast_math_enable"}
"""


def _renamed(air: str) -> str:
    """``air`` with other names for its functions, arguments, members and structures."""
    for old, new in (
        ("@vmain", "@vertex_main"),
        ("@fmain", "@fragment_main"),
        ('!"U"', '!"Uniforms"'),
        ('!"AB"', '!"Material"'),
        ('!"m", ', '!"model", '),
        ('!"cnt"', '!"counter"'),
        ('!"air.arg_name", !"uv"', '!"air.arg_name", !"texcoord"'),
        ("generated(1kf)", "generated(5shadef)"),
    ):
        air = air.replace(old, new)
    return air


def test_each_entry_point_has_its_kind_results_and_host_arguments():
    vertex, fragment = reflection.interfaces(AIR)
    assert (vertex.kind, fragment.kind) == ("vertex", "fragment")
    kinds = [argument[0] for argument in vertex.arguments]
    # The vertex_id built-in is no part of the host interface.
    assert sorted(kinds) == sorted(
        ["air.vertex_input", "air.vertex_input", "air.buffer", "air.indirect_buffer"]
    )
    assert ("air.render_target", 2, 0, ("type", "float4")) in fragment.results
    assert ("air.depth", "air.depth_qualifier", "air.any", ("type", "float")) in (
        fragment.results
    )


def test_argument_buffer_members_keep_their_offsets_types_and_indices():
    vertex, _fragment = reflection.interfaces(AIR)
    (indirect,) = (a for a in vertex.arguments if a[0] == "air.indirect_buffer")
    layout = next(value for value in indirect if isinstance(value, tuple) and value)
    offsets = [member[0] for member in layout]
    assert offsets == [0, 8, 16]
    assert layout[0][3] == "texture2d<float, sample>"
    assert layout[1][4][1:5] == ("air.sampler", "air.location_index", 1, 1)


def test_names_a_translation_chooses_do_not_count():
    assert reflection.differences(AIR, _renamed(AIR)) == []


@pytest.mark.parametrize(
    ("old", "new", "expected"),
    [
        (
            '!"air.texture", !"air.location_index", i32 3,',
            '!"air.texture", !"air.location_index", i32 4,',
            "the fragment entry's arguments lacks texture(3)",
        ),
        (
            '!"air.vertex_input", !"air.location_index", i32 3,',
            '!"air.vertex_input", !"air.location_index", i32 2,',
            "the vertex entry's arguments lacks vertex_input(3) float2 and adds vertex_input(2) float2",
        ),
        (
            '!{!"air.render_target", i32 2, i32 0,',
            '!{!"air.render_target", i32 1, i32 0,',
            "the fragment entry's results lacks render_target(2) float4",
        ),
        (
            '!"user(locn1)", !"air.center", !"air.perspective"',
            '!"user(locn1)", !"air.center", !"air.no_perspective"',
            "the fragment entry's arguments lacks fragment_input user(locn1) float2",
        ),
        # An argument buffer member at another offset.
        ('i32 16, i32 8, i32 0, !"float"', 'i32 24, i32 8, i32 0, !"float"', "lacks"),
        # A constant buffer of another size.
        ('i32 64, i32 4, i32 0, !"float"', 'i32 64, i32 8, i32 0, !"float2"', "lacks"),
    ],
)
def test_a_moved_or_changed_binding_is_a_difference(old, new, expected):
    assert old in AIR
    found = reflection.differences(AIR, AIR.replace(old, new))
    assert found and expected in found[0], found


def test_a_missing_entry_point_is_a_difference():
    vertex_only = AIR.replace("!air.fragment = !{!25}\n", "")
    assert reflection.differences(AIR, vertex_only) == [
        "0 fragment entry points where 1 are expected"
    ]


def test_an_unsized_clip_distance_is_an_array_of_one():
    sized = (
        "!air.vertex = !{!1}\n!1 = !{void ()* @v, !2, !4}\n!2 = !{!3}\n"
        '!3 = !{!"air.clip_distance", !"air.clip_distance_array_size", i32 1, '
        '!"air.arg_type_name", !"float", !"air.arg_name", !"clip"}\n!4 = !{}\n'
    )
    unsized = sized.replace('!"air.clip_distance_array_size", i32 1, ', "")
    assert reflection.differences(sized, unsized) == []
    assert reflection.differences(sized, sized.replace("i32 1,", "i32 2,"))


_METAL = (
    shutil.which("xcrun")
    and subprocess.run(
        ["xcrun", "-sdk", "macosx", "-f", "metal"], capture_output=True, check=False
    ).returncode
    == 0
)


def _air(tmp_path, source: str) -> str:
    (tmp_path / "probe.metal").write_text(source)
    subprocess.run(
        [
            *("xcrun", "-sdk", "macosx", "metal", "-std=metal3.1", "-S", "-emit-llvm"),
            *("probe.metal", "-o", "probe.ll"),
        ],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )
    return (tmp_path / "probe.ll").read_text()


RENDER = """
#include <metal_stdlib>
using namespace metal;
struct VIn { float3 p [[attribute(0)]]; float2 uv [[attribute(2)]]; };
struct VOut { float4 pos [[position]]; float2 uv [[user(locn0)]]; };
struct U { float4x4 m; float s; };
vertex VOut vs(VIn i [[stage_in]], constant U& u [[buffer(1)]]) {
  VOut o; o.pos = u.m * float4(i.p, 1.0); o.uv = i.uv * u.s; return o; }
fragment float4 fs(VOut i [[stage_in]], texture2d<float> t [[texture(3)]], sampler s [[sampler(1)]]) {
  return t.sample(s, i.uv); }
"""


@pytest.mark.skipif(not _METAL, reason="the Metal compiler is not installed")
def test_metal_input_round_trip_keeps_the_render_interface(tmp_path):
    import crosstl

    (tmp_path / "render.metal").write_text(RENDER)
    crossgl = crosstl.translate(str(tmp_path / "render.metal"), backend="cgl")
    (tmp_path / "render.cgl").write_text(crossgl)
    translated = crosstl.translate(str(tmp_path / "render.cgl"), backend="metal")
    native_air = _air(tmp_path, RENDER)
    assert reflection.differences(native_air, _air(tmp_path, translated)) == []
    moved = RENDER.replace("[[texture(3)]]", "[[texture(0)]]")
    assert reflection.differences(native_air, _air(tmp_path, moved))


def _round_trip(monkeypatch, translated: str, moved: str):
    """Run the coverage runner's Metal input check of ``sin`` with every tool accepting everything, the round trip
    writing ``translated`` and the reflection check reporting ``moved``."""
    from tools.coverage import runner, validators
    from tools.coverage.model import Feature, Param, Signature, Type

    monkeypatch.setattr(validators, "available", lambda language, target="": True)
    monkeypatch.setattr(
        runner, "_validated", lambda *args, **kwargs: validators.Verdict(True)
    )
    monkeypatch.setattr(
        runner, "_native_rejection", lambda feature, language: (feature, "")
    )
    monkeypatch.setattr(runner, "parse_crossgl", lambda source: None)
    monkeypatch.setattr(runner, "_runs", lambda feature, language: False)
    translations = {"cgl": "r = sin(a0);", "metal": translated, "hip": translated}
    monkeypatch.setattr(
        runner, "translate", lambda source, suffix, backend, _: translations[backend]
    )
    checked = []

    def difference(native, written):
        checked.append((native, written))
        return moved

    monkeypatch.setattr(reflection, "round_trip_difference", difference)
    signature = Signature(Type("float"), (Param(Type("float")),))
    return (
        runner,
        Feature("metal", "function", "sin", signature, crossgl="sin"),
        checked,
    )


def test_a_metal_round_trip_that_changes_the_host_interface_is_wrong(monkeypatch):
    moved = "the host interface changes: the fragment entry's arguments lacks texture(3) texture2d<float, sample>"
    runner, feature, checked = _round_trip(monkeypatch, "float r = sin(a0);", moved)
    result = runner.run_input(feature, "metal")
    assert (result.outcome, result.detail) == (
        "wrong",
        f"after the round trip: {moved}",
    )
    assert checked and checked[0][1] == "float r = sin(a0);"


def test_a_metal_round_trip_that_keeps_the_host_interface_is_supported(monkeypatch):
    runner, feature, checked = _round_trip(monkeypatch, "float r = sin(a0);", "")
    assert runner.run_input(feature, "metal").outcome == "supported"
    assert len(checked) == 1


@pytest.mark.parametrize("compiled", [("", "!air.vertex = !{}"), (AIR, "")])
def test_the_check_is_skipped_without_air_for_either_source(monkeypatch, compiled):
    # Without the Metal compiler (or when it rejects a source) there is no metadata to compare, so the round trip
    # keeps the verdict its validator gives it instead of failing.
    from tools.coverage import validators

    sources = dict(zip(("native", "translated"), compiled))
    monkeypatch.setattr(validators, "air", lambda source: sources[source])
    assert reflection.round_trip_difference("native", "translated") == ""


def _output(monkeypatch, stage: str, moved: str):
    """Run the coverage runner's Metal output check of a ``stage`` probe with every tool accepting everything and
    the reflection check reporting ``moved``."""
    from tools.coverage import probes, runner, validators
    from tools.coverage.model import VOID, Feature, Signature

    monkeypatch.setattr(validators, "available", lambda language, target="": True)
    monkeypatch.setattr(
        runner, "_validated", lambda *args, **kwargs: validators.Verdict(True)
    )
    monkeypatch.setattr(
        runner, "_native_rejection", lambda feature, language: (feature, "")
    )
    monkeypatch.setattr(runner, "parse_crossgl", lambda source: None)
    monkeypatch.setattr(runner, "_runs", lambda feature, language: False)
    monkeypatch.setattr(runner, "translate", lambda *args: "fragment float4 f();")
    monkeypatch.setattr(
        runner.decorations, "lost", lambda feature, language, translated: ""
    )
    checked = []

    def build(feature, dialect):
        checked.append(dialect)
        return type("Probe", (), {"source": f"{dialect} probe", "suffix": ".cgl"})()

    monkeypatch.setattr(probes, "build", build)
    differences = []

    def difference(native, written):
        differences.append((native, written))
        return moved

    monkeypatch.setattr(reflection, "round_trip_difference", difference)
    feature = Feature("metal", "stage", stage, Signature(VOID, ()), stage=stage)
    return runner.run_output(feature, "metal"), checked, differences


def test_a_graphics_translation_with_another_interface_than_the_native_probe_is_wrong(
    monkeypatch,
):
    moved = "the host interface changes: the fragment entry's results lacks render_target(1) float4"
    result, _, differences = _output(monkeypatch, "fragment", moved)
    assert (result.outcome, result.detail) == (
        "wrong",
        "the host interface differs from the native Metal probe's: the fragment entry's results lacks "
        "render_target(1) float4",
    )
    assert differences == [("metal probe", "fragment float4 f();")]


def test_a_graphics_translation_with_the_native_interface_is_supported(monkeypatch):
    result, _, differences = _output(monkeypatch, "vertex", "")
    assert result.outcome == "supported"
    assert len(differences) == 1


def test_a_compute_probe_is_compared_by_running_it_not_by_its_interface(monkeypatch):
    # The harness binds each compute probe's own resources and compares the words it writes.
    result, checked, differences = _output(monkeypatch, "compute", "changes")
    assert result.outcome == "supported"
    assert differences == [] and "metal" not in checked


def _metal_features():
    """The Metal feature list from the cached upstream sources; skips when they are not cached, as in a CI job that
    does not restore the coverage cache."""
    from tools.coverage import features
    from tools.coverage.upstream import UpstreamError

    try:
        return {
            feature.name: feature for feature in features.load("metal", offline=True)
        }
    except UpstreamError:
        pytest.skip("the Metal feature sources are not cached")


@pytest.mark.skipif(not _METAL, reason="the Metal compiler is not installed")
def test_metal_output_of_every_graphics_probe_keeps_the_native_interface():
    from tools.coverage import probes, runner

    checked = 0
    for feature in _metal_features().values():
        if feature.stage == "compute" or feature.name not in {
            "fragment",
            "vertex",
            "multiple-render-targets",
            "mesh",
            "intersection",
        }:
            continue
        probe = probes.build(feature, "crossgl")
        translated = runner.translate(probe.source, probe.suffix, "metal", "cgl")
        assert reflection.output_difference(feature, translated) == "", feature.name
        checked += 1
    assert checked == 5


@pytest.mark.skipif(not _METAL, reason="the Metal compiler is not installed")
def test_a_render_target_moved_by_the_translation_is_reported():
    from tools.coverage import probes

    feature = _metal_features()["multiple-render-targets"]
    native = probes.build(feature, "metal").source
    moved = native.replace("[[color(1)]]", "[[color(2)]]")
    assert moved != native
    found = reflection.output_difference(feature, moved)
    assert found.startswith("the host interface differs from the native Metal probe's:")
    assert "render_target" in found

"""``crosstl.translate_artifacts``: every artifact a target needs, with the source imported once."""

import pytest

import crosstl
from crosstl import _crosstl, translation_artifacts
from crosstl.project import glsl_entry_split
from crosstl.translator.codegen.stage_refusals import StageUnsupportedError

TWO_KERNELS = """
#include <metal_stdlib>
using namespace metal;

kernel void add_one(device uint* data [[buffer(0)]], uint i [[thread_position_in_grid]]) {
    data[i] += 1u;
}

kernel void double_it(device float* data [[buffer(1)]], uint i [[thread_position_in_grid]]) {
    data[i] *= 2.0f;
}
"""

ONE_KERNEL = """
#include <metal_stdlib>
using namespace metal;

kernel void add_one(device uint* data [[buffer(0)]], uint i [[thread_position_in_grid]]) {
    data[i] += 1u;
}
"""

VERTEX_AND_FRAGMENT = """
#include <metal_stdlib>
using namespace metal;

struct VertexOut {
    float4 position [[position]];
};

vertex VertexOut vertex_main(uint id [[vertex_id]]) {
    VertexOut out;
    out.position = float4(float(id), 0.0f, 0.0f, 1.0f);
    return out;
}

fragment float4 fragment_main(VertexOut in [[stage_in]]) {
    return float4(1.0f, 0.0f, 0.0f, 1.0f);
}
"""


def _source(tmp_path, text, name="unit.metal"):
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return str(path)


@pytest.fixture
def imports(monkeypatch):
    """The number of times a translation imports its source (reads, parses and rule-carries it)."""
    calls = []
    original = _crosstl._target_program

    def counting(*args, **kwargs):
        calls.append(kwargs.get("entry_point"))
        return original(*args, **kwargs)

    monkeypatch.setattr(_crosstl, "_target_program", counting)
    return calls


def test_repeated_stage_entries_are_translated_one_per_artifact_from_one_import(
    tmp_path, imports
):
    path = _source(tmp_path, TWO_KERNELS)

    artifacts = crosstl.translate_artifacts(path, "opengl", format_output=False)

    assert len(imports) == 1
    assert [(a.stage, a.entry_point) for a in artifacts] == [
        ("compute", "add_one"),
        ("compute", "double_it"),
    ]
    for artifact in artifacts:
        assert artifact.code.count("void main()") == 1
        assert artifact.code == crosstl.translate(
            path,
            backend="opengl",
            format_output=False,
            entry_point=artifact.entry_point,
        )


def test_artifacts_match_the_split_after_the_whole_file_refusal(tmp_path, monkeypatch):
    path = _source(tmp_path, TWO_KERNELS)
    split = crosstl.translate_artifacts(path, "opengl", format_output=False)

    # Without discovery, the whole file is translated, refused for its two compute entries, and then split.
    monkeypatch.setattr(
        translation_artifacts,
        "discovered_repeated_stage_entries",
        lambda *args, **kwargs: (),
    )
    calls = []
    original = _crosstl._target_program
    monkeypatch.setattr(
        _crosstl,
        "_target_program",
        lambda *a, **k: calls.append(k.get("entry_point")) or original(*a, **k),
    )
    refused_then_split = crosstl.translate_artifacts(
        path, "opengl", format_output=False
    )

    assert calls == [None, "add_one"]
    assert refused_then_split == split


def test_a_discovered_entry_the_program_lacks_falls_back_to_the_programs_entries(
    tmp_path, monkeypatch
):
    path = _source(tmp_path, TWO_KERNELS)
    expected = crosstl.translate_artifacts(path, "opengl", format_output=False)
    monkeypatch.setattr(
        translation_artifacts,
        "discovered_repeated_stage_entries",
        lambda *args, **kwargs: (("compute", "add_one"), ("compute", "missing")),
    )

    assert crosstl.translate_artifacts(path, "opengl", format_output=False) == expected


@pytest.mark.parametrize("backend", ["directx", "vulkan", "metal"])
def test_a_target_that_takes_the_whole_file_gets_one_artifact(
    tmp_path, imports, backend
):
    path = _source(tmp_path, TWO_KERNELS)

    artifacts = crosstl.translate_artifacts(path, backend, format_output=False)

    assert imports == [None]
    assert artifacts == (
        crosstl.TranslationArtifact(
            crosstl.translate(path, backend=backend, format_output=False)
        ),
    )


@pytest.mark.parametrize("text", [ONE_KERNEL, VERTEX_AND_FRAGMENT])
def test_entries_of_different_stages_stay_in_one_artifact(tmp_path, imports, text):
    path = _source(tmp_path, text)

    assert translation_artifacts.discovered_repeated_stage_entries(path, "opengl") == ()
    artifacts = crosstl.translate_artifacts(path, "opengl", format_output=False)

    assert imports == [None]
    assert len(artifacts) == 1 and artifacts[0].entry_point is None
    assert artifacts[0].code == crosstl.translate(
        path, backend="opengl", format_output=False
    )


def test_a_source_no_stage_of_the_target_supports_is_refused_before_it_is_imported(
    tmp_path, imports
):
    path = _source(tmp_path, TWO_KERNELS)

    with pytest.raises(StageUnsupportedError):
        crosstl.translate_artifacts(path, "webgl", format_output=False)
    assert imports == []


def test_one_entry_per_stage_targets_are_those_the_project_pipeline_splits():
    for target in ("opengl", "webgl", "gles", "glsl"):
        assert translation_artifacts.holds_one_entry_per_stage(target), target
    for target in ("directx", "metal", "vulkan", "cuda", "hip", "wgsl", "slang"):
        assert not translation_artifacts.holds_one_entry_per_stage(target), target
    assert (
        glsl_entry_split.holds_one_entry_per_stage
        is translation_artifacts.holds_one_entry_per_stage
    )

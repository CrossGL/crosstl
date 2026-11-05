"""WebGL stage refusals decided from a Metal source's entry points, before template materialization.

A Metal compute library cannot translate to WebGL 2, which has only vertex and fragment shaders, but the refusal came
from the WebGL generator after the Metal importer had materialized every template: minutes for the MLX and PyTorch
kernel libraries of the corpus, past its time limit. Entry discovery finds the same refusal in under a second.
"""

import pytest

import crosstl
import crosstl._crosstl as crosstl_module
import crosstl.project.pipeline as pipeline
from crosstl.translator.codegen.stage_refusals import StageUnsupportedError

COMPUTE = """#include <metal_stdlib>
using namespace metal;
template <typename T>
kernel void scale(device T* data [[buffer(0)]], uint id [[thread_position_in_grid]]) {
    data[id] = data[id] * T(2);
}
template [[host_name("scale_float")]] kernel void scale<float>(device float*, uint);
kernel void fill(device float* data [[buffer(0)]], uint id [[thread_position_in_grid]]) {
    data[id] = 1.0f;
}
"""

GRAPHICS_AND_COMPUTE = """#include <metal_stdlib>
using namespace metal;
vertex float4 vs(uint vid [[vertex_id]]) { return float4(float(vid), 0.0, 0.0, 1.0); }
fragment float4 fs() { return float4(1.0); }
kernel void fill(device float* data [[buffer(0)]], uint id [[thread_position_in_grid]]) {
    data[id] = 1.0f;
}
"""


def write(tmp_path, source, name="kernels.metal"):
    path = tmp_path / name
    path.write_text(source)
    return str(path)


def refusal(path):
    with pytest.raises(StageUnsupportedError) as caught:
        crosstl.translate(path, backend="webgl", format_output=False)
    return caught.value


def test_compute_only_metal_source_is_refused_before_materialization(
    tmp_path, monkeypatch
):
    def materialize(**_kwargs):
        raise AssertionError("the Metal source was materialized")

    monkeypatch.setattr(pipeline, "materialize_metal_source_for_target", materialize)
    error = refusal(write(tmp_path, COMPUTE))
    assert error.code == "project.translate.webgl-stage-unsupported"
    assert error.stages == ("compute",)
    assert error.missing_capabilities == ("webgl.stage.compute",)


def test_early_refusal_matches_the_generator_refusal(tmp_path, monkeypatch):
    path = write(tmp_path, COMPUTE)
    early = refusal(path)
    monkeypatch.setattr(
        crosstl_module, "refuse_infeasible_entry_stages", lambda *a, **k: None
    )
    late = refusal(path)
    assert str(early) == str(late)
    assert early.code == late.code
    assert early.stages == late.stages


class Materialized(Exception):
    pass


def materialize_marker(monkeypatch):
    def materialize(**_kwargs):
        raise Materialized

    monkeypatch.setattr(pipeline, "materialize_metal_source_for_target", materialize)


def test_a_selected_entry_point_takes_the_full_translation(tmp_path, monkeypatch):
    materialize_marker(monkeypatch)
    with pytest.raises(Materialized):
        crosstl.translate(
            write(tmp_path, COMPUTE),
            backend="webgl",
            format_output=False,
            entry_point="fill",
        )


def test_a_failed_discovery_takes_the_full_translation(tmp_path, monkeypatch):
    from crosstl.translator.source_registry import SOURCE_REGISTRY

    def fail(*_args, **_kwargs):
        raise RuntimeError("discovery failed")

    monkeypatch.setattr(
        type(SOURCE_REGISTRY.get("metal")), "discover_entry_points", fail
    )
    materialize_marker(monkeypatch)
    with pytest.raises(Materialized):
        crosstl.translate(
            write(tmp_path, COMPUTE), backend="webgl", format_output=False
        )


def test_a_source_with_a_vertex_or_fragment_entry_translates_as_before(tmp_path):
    # The generator keeps the stages WebGL declares; the precheck only refuses when there are none.
    code = crosstl.translate(
        write(tmp_path, GRAPHICS_AND_COMPUTE), backend="webgl", format_output=False
    )
    assert "#version 300 es" in code


def test_other_targets_are_not_prechecked(tmp_path):
    code = crosstl.translate(
        write(tmp_path, COMPUTE), backend="opengl", format_output=False
    )
    assert "local_size_x" in code or "void main" in code


def test_project_translation_reports_the_stage_refusal(tmp_path):
    # A compute kernel that also reinterprets a pointer: WebGL refuses it for its stage.
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "reinterpret.metal").write_text(
        "#include <metal_stdlib>\n"
        "using namespace metal;\n"
        "[[kernel]] void read_byte(const device uint32_t* words [[buffer(0)]],\n"
        "    device uint8_t* out_values [[buffer(1)]], uint gid [[thread_position_in_grid]]) {\n"
        "  const device uint8_t* bytes = (const device uint8_t*)words;\n"
        "  out_values[gid] = bytes[gid];\n"
        "}\n",
        encoding="utf-8",
    )
    payload = pipeline.translate_project(
        repo, targets=["webgl"], output_dir="out", format_output=False
    ).to_json()
    assert payload["summary"]["translatedCount"] == 0
    assert payload["summary"]["failedCount"] == 1
    codes = [item["code"] for item in payload["diagnostics"]]
    assert "project.translate.webgl-stage-unsupported" in codes

"""Generated helpers must not acquire source locations through token similarity."""

import json

import pytest

from crosstl.project import pipeline as project_pipeline
from crosstl.project import translate_project, validate_project_report

SOURCE = """#include <metal_stdlib>
// clang-format off
using namespace metal;
kernel void root_value(const device float* input [[buffer(0)]],
                       device float* output [[buffer(1)]],
                       uint index [[thread_position_in_grid]]) {
    output[index] = metal::precise::sqrt(input[index]);
}
"""


@pytest.mark.parametrize("target", ["metal", "directx", "opengl", "cgl"])
def test_generated_helper_records_file_provenance_without_guessed_lines(
    tmp_path, target
):
    (tmp_path / "root.metal").write_text(SOURCE, encoding="utf-8")
    report = translate_project(tmp_path, targets=[target], format_output=False)
    payload = report.to_json()
    (artifact,) = payload["artifacts"]
    assert artifact["status"] == "translated"
    source_map = artifact["sourceMap"]
    assert source_map["mappingGranularity"] == "file"
    assert source_map["mappings"] == [
        {
            "source": source_map["source"],
            "generated": source_map["generated"],
        }
    ]
    assert payload["summary"]["fineGrainedSourceMapCount"] == 0
    assert payload["summary"]["sourceMapsByGranularity"] == {"file": 1}
    assert artifact["sourceRemap"]["mappingGranularity"] == "file"
    remap = json.loads((tmp_path / artifact["sourceRemap"]["path"]).read_text())
    assert remap["mappings"] == [
        {
            "original": source_map["source"],
            "generated": source_map["generated"],
        }
    ]
    if target == "metal":
        generated = (tmp_path / artifact["path"]).read_text()
        assert "#pragma clang fp contract(off)" in generated
        assert "#pragma clang fp reassociate(off)" in generated
    path = tmp_path / "report.json"
    report.write_json(path)
    validation = validate_project_report(path)
    assert validation["success"] is True


def test_validation_rejects_reproducible_but_unproven_line_origin(tmp_path):
    (tmp_path / "root.metal").write_text(SOURCE, encoding="utf-8")
    payload = translate_project(
        tmp_path, targets=["metal"], format_output=False
    ).to_json()
    (artifact,) = payload["artifacts"]
    generated_path = tmp_path / artifact["path"]
    lines = generated_path.read_text().splitlines()
    source_map = artifact["sourceMap"]
    source_map["mappingGranularity"] = "line"
    source_spans = project_pipeline._line_spans(tmp_path / "root.metal", "root.metal")
    generated_spans = project_pipeline._line_spans(generated_path, artifact["path"])
    # These are the historical token-overlap matches, including unrelated helper lines.
    legacy_matches = [(1, 2), (7, 5), (2, 6), (2, 7), (4, 40), (4, 61), (7, 62)]
    assert lines[5] == "    #pragma clang fp contract(off)"
    assert lines[6] == "    #pragma clang fp reassociate(off)"
    source_map["mappings"] = [
        {
            "source": source_spans[source_line - 1].to_json(),
            "generated": generated_spans[generated_line - 1].to_json(),
        }
        for source_line, generated_line in legacy_matches
    ]
    remap_path = tmp_path / artifact["sourceRemap"]["path"]
    project_pipeline._write_source_remap_sidecar(
        remap_path, project_pipeline._source_remap_payload(source_map)
    )
    artifact["sourceRemap"].update(
        {
            "mappingGranularity": "line",
            "mappingCount": len(legacy_matches),
            "sizeBytes": remap_path.stat().st_size,
            "hash": project_pipeline._source_hash(remap_path),
        }
    )
    payload["summary"].update(
        project_pipeline._source_map_rollups(payload["artifacts"])
    )
    path = tmp_path / "report.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    validation = validate_project_report(path)
    assert validation["success"] is False
    (checked,) = validation["validation"]["artifacts"]
    assert checked["sourceHashStatus"] == checked["generatedHashStatus"] == "ok"
    assert checked["sourceRemapStatus"] == "ok"
    assert checked["sourceMapStatus"] == "mismatch"
    assert any(
        "line mappings require line-preserving source and generated files"
        in diagnostic["message"]
        for diagnostic in validation["diagnostics"]
    )

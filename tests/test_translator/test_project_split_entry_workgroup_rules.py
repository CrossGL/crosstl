"""Entry workgroup-size rules on a unit that a target translates one entry per artifact.

A GLSL compilation unit holds one compute entry (``main``), so ``crosstl.project.glsl_entry_split`` gives every
discovered entry its own artifact. An ``entry_workgroup_size_rules`` table names the entries it sizes, so each split
artifact resolves the rule of its own entry. An entry no rule names keeps the size its source or host gives it.
"""

import textwrap

import pytest

import crosstl.project as project_api

SOURCE = "shaders/rows.metal"


def _write_rows_fixture(repo):
    shader_dir = repo / "shaders"
    shader_dir.mkdir(parents=True)
    (shader_dir / "rows.metal").write_text(
        textwrap.dedent("""
            #include <metal_stdlib>
            using namespace metal;

            template <typename T, int N_READS>
            [[kernel]] void tile(
                device T* output [[buffer(0)]],
                uint lid [[thread_position_in_threadgroup]]) {
                output[lid * N_READS] = T(lid);
            }

            template <typename T, int N_READS>
            [[kernel]] void strided(
                device T* output [[buffer(0)]],
                uint lid [[thread_position_in_threadgroup]],
                uint lsize [[threads_per_threadgroup]]) {
                for (uint r = lid * N_READS; r < 4096; r += lsize * N_READS) {
                    output[r] = T(lsize);
                }
            }

            instantiate_kernel("tile_float", tile, float, 4)
            instantiate_kernel("strided_float", strided, float, 4)
            """).strip() + "\n",
        encoding="utf-8",
    )


def _translate(tmp_path, entry_rules, *, targets=("opengl",)):
    repo = tmp_path / "repo"
    _write_rows_fixture(repo)
    report = project_api.translate_project(
        project_api.ProjectConfig(
            root=repo,
            targets=targets,
            output_dir="out",
            entry_workgroup_size_rules={SOURCE: entry_rules},
        ),
        format_output=False,
        validate=True,
    )
    return repo, report


def _artifacts_by_entry(payload):
    return {
        artifact["entryPoint"]["source"]: artifact
        for artifact in payload["artifacts"]
        if artifact["target"] == "opengl"
    }


def _error_codes(payload):
    return sorted(
        {
            diagnostic["code"]
            for diagnostic in payload["diagnostics"]
            if diagnostic["severity"] == "error"
        }
    )


def test_entry_rules_size_the_named_entries_of_a_split_opengl_unit(tmp_path):
    repo, report = _translate(tmp_path, {"strided_*": ["1024", "1", "1"]})
    payload = report.to_json()

    assert payload["summary"]["failedCount"] == 0, payload["diagnostics"]
    artifacts = _artifacts_by_entry(payload)
    assert sorted(artifacts) == ["strided_float", "tile_float"]
    assert (
        artifacts["strided_float"]["path"]
        == "out/opengl/shaders/rows/strided_float.glsl"
    )
    assert [
        entry["workgroupSize"]
        for entry in artifacts["strided_float"]["execution"]["entryPoints"]
    ] == [[1024, 1, 1]]
    assert artifacts["strided_float"]["execution"]["entryPoints"][0]["rule"] == {
        "components": ["1024", "1", "1"],
        "sourcePattern": SOURCE,
        "entryPattern": "strided_*",
        "path": f'project.entry_workgroup_size_rules["{SOURCE}"]["strided_*"]',
    }
    strided = (repo / artifacts["strided_float"]["path"]).read_text(encoding="utf-8")
    assert (
        "layout(local_size_x = 1024, local_size_y = 1, local_size_z = 1) in;" in strided
    )
    assert "uint lsize = uint(gl_WorkGroupSize.x);" in strided

    # No rule names tile_float, and it does not read its workgroup size, so the host still chooses it.
    assert "execution" not in artifacts["tile_float"] or not artifacts["tile_float"][
        "execution"
    ].get("entryPoints")
    tile = (repo / artifacts["tile_float"]["path"]).read_text(encoding="utf-8")
    assert "local_size_x_id = 0" in tile
    assert "local_size_x = 1024" not in tile

    report_path = repo / "report.json"
    report.write_json(report_path)
    assert project_api.validate_project_report(report_path)["success"] is True


def test_split_entry_rules_still_reject_a_pattern_no_entry_matches(tmp_path):
    _repo, report = _translate(tmp_path, {"missing_*": ["1024", "1", "1"]})
    payload = report.to_json()

    # Coverage is checked against every entry of the unit, so each split artifact refuses the stray pattern.
    assert payload["summary"]["translatedCount"] == 0
    assert payload["summary"]["failedCount"] == 2
    assert "project.translate.workgroup-size-entry-rule-unmatched" in _error_codes(
        payload
    )


def test_split_entry_rules_leave_an_unnamed_consuming_entry_host_sized(tmp_path):
    repo, report = _translate(tmp_path, {"tile_*": ["64", "1", "1"]})
    payload = report.to_json()

    artifacts = _artifacts_by_entry(payload)
    assert artifacts["tile_float"]["status"] == "translated"
    assert [
        entry["workgroupSize"]
        for entry in artifacts["tile_float"]["execution"]["entryPoints"]
    ] == [[64, 1, 1]]
    # The rule names only tile_*, so strided_float keeps the host-chosen size: it reads
    # gl_WorkGroupSize from the CROSSGL_WORKGROUP_SIZE_* macros instead of taking 64.
    strided = artifacts["strided_float"]
    assert strided["status"] == "translated"
    assert strided.get("execution") is None
    code = (repo / strided["path"]).read_text(encoding="utf-8")
    assert "gl_WorkGroupSize" in code
    assert "CROSSGL_WORKGROUP_SIZE_X" in code
    assert "local_size_x = 64" not in code
    assert _error_codes(payload) == []


@pytest.mark.parametrize("target", ("directx", "metal"))
def test_multi_entry_artifacts_still_need_a_rule_for_every_entry(tmp_path, target):
    _repo, report = _translate(
        tmp_path, {"strided_*": ["1024", "1", "1"]}, targets=(target,)
    )
    payload = report.to_json()

    assert payload["summary"]["translatedCount"] == 0
    assert "project.translate.workgroup-size-entry-rule-unmatched" in _error_codes(
        payload
    )

from __future__ import annotations

import textwrap

import pytest

import crosstl.project.pipeline as project_pipeline
from crosstl.project import load_project_config, translate_project

SOURCE = "matrix.cgl"


@pytest.fixture(autouse=True)
def _declare_opengl_validation_tool_available(monkeypatch):
    """Keep these target-option assertions independent of host-installed tools."""
    monkeypatch.setattr(
        project_pipeline.shutil,
        "which",
        lambda tool: (
            "/test-tools/glslangValidator" if tool == "glslangValidator" else None
        ),
    )


def _write_fixture(repo) -> None:
    (repo / SOURCE).write_text(
        textwrap.dedent("""
            shader CooperativeMatrixProject {
                compute {
                    void main() {
                        CooperativeMatrix<
                            float, 8, 8, subgroup, unspecified, unspecified,
                            metal_thread_elements, 32, 2,
                            metal_thread_elements_reference_view,
                            tile_4x4_row_pair,
                            mlx_steel_BaseMMAFrag_get_coord
                        > left;
                        CooperativeMatrix<
                            float, 8, 8, subgroup, unspecified, unspecified,
                            metal_thread_elements, 32, 2,
                            metal_thread_elements_reference_view,
                            tile_4x4_row_pair,
                            mlx_steel_BaseMMAFrag_get_coord
                        > right;
                        cooperative_matrix_element(left, 0) = 1.0;
                        cooperative_matrix_element(right, 0) = 2.0;
                        CooperativeMatrix<
                            float, 8, 8, subgroup, unspecified, unspecified,
                            metal_thread_elements, 32, 2,
                            metal_thread_elements_reference_view,
                            tile_4x4_row_pair,
                            mlx_steel_BaseMMAFrag_get_coord
                        > sum = cooperative_matrix_add(left, right);
                        float value = cooperative_matrix_element(sum, 0);
                    }
                }
            }
            """).strip() + "\n",
        encoding="utf-8",
    )


def _write_config(repo, enabled) -> None:
    literal = (
        "true" if enabled is True else "false" if enabled is False else f'"{enabled}"'
    )
    (repo / "crosstl.toml").write_text(
        textwrap.dedent(f"""
            [project]
            source_roots = ["."]
            include = ["{SOURCE}"]
            targets = ["opengl"]
            output_dir = "out"

            [project.source_options.cgl.target_options.opengl]
            cooperative_matrix_software_lowering = {literal}
            """).strip() + "\n",
        encoding="utf-8",
    )


def _translate(repo, enabled):
    _write_fixture(repo)
    _write_config(repo, enabled)
    return translate_project(
        load_project_config(repo),
        format_output=False,
        validate=True,
        run_toolchains=False,
    ).to_json()


def test_project_opengl_target_option_enables_exact_lane_local_matrix_lowering(
    tmp_path,
):
    payload = _translate(tmp_path, True)

    assert payload["project"]["sourceOptions"] == {
        "cgl": {
            "target_options": {"opengl": {"cooperative_matrix_software_lowering": True}}
        }
    }
    assert payload["summary"]["translatedCount"] == 1
    assert payload["summary"]["failedCount"] == 0
    assert payload["diagnostics"] == []
    generated_path = tmp_path / payload["artifacts"][0]["path"]
    generated = generated_path.read_text(encoding="utf-8")
    assert "struct crossgl_cooperative_matrix_fragment_" in generated
    assert "float elements[2];" in generated
    assert "result.elements[0] = left.elements[0] + right.elements[0];" in generated
    for intrinsic in ("cooperative_matrix_element(", "cooperative_matrix_add("):
        assert intrinsic not in generated


def test_project_opengl_target_option_remains_strictly_opt_in(tmp_path):
    payload = _translate(tmp_path, False)

    assert payload["summary"]["translatedCount"] == 0
    assert payload["summary"]["failedCount"] == 1
    assert payload["summary"]["diagnosticsByCode"] == {
        "project.translate.opengl-cooperative-matrix-unsupported": 1,
        "project.validate.failed-artifact": 1,
    }
    assert payload["diagnostics"][0]["details"]["cooperativeMatrix"]["reason"] == (
        "unsupported-type"
    )
    assert not (tmp_path / payload["artifacts"][0]["path"]).exists()


@pytest.mark.parametrize("enabled", ["yes", "1"])
def test_project_opengl_target_option_rejects_non_boolean_values(tmp_path, enabled):
    payload = _translate(tmp_path, enabled)

    assert payload["summary"]["translatedCount"] == 0
    assert payload["summary"]["failedCount"] == 1
    assert "must be a boolean" in payload["artifacts"][0]["error"]
    assert not (tmp_path / payload["artifacts"][0]["path"]).exists()

from __future__ import annotations

import textwrap

import pytest

from crosstl.project import load_project_config, translate_project

SOURCE = "private_pointer.cgl"
SOURCE_TEXT = textwrap.dedent("""
    shader RobustPrivatePointerProject {
        float read_values(const thread float* values) {
            float result = 0.0;
            for (int i = 0; i < 3; ++i) {
                values += 2 * i;
                result += values[1];
            }
            return result;
        }

        compute {
            layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

            void main() {
                float backing[7];
                float observed = read_values(backing);
            }
        }
    }
    """).strip() + "\n"


def _translate(repo, *, policy: str, target: str = "opengl"):
    (repo / SOURCE).write_text(SOURCE_TEXT, encoding="utf-8")
    (repo / "crosstl.toml").write_text(
        textwrap.dedent(f"""
            [project]
            source_roots = ["."]
            include = ["{SOURCE}"]
            targets = ["{target}"]
            output_dir = "out"

            [project.source_options.cgl.target_options.{target}]
            private_pointer_out_of_bounds_read = "{policy}"
            """).strip() + "\n",
        encoding="utf-8",
    )
    return translate_project(
        load_project_config(repo),
        format_output=False,
        validate=True,
        run_toolchains=False,
    ).to_json()


def test_project_opengl_private_pointer_zero_read_policy_is_explicit_and_visible(
    tmp_path,
):
    payload = _translate(tmp_path, policy="zero")

    assert payload["project"]["sourceOptions"] == {
        "cgl": {
            "target_options": {"opengl": {"private_pointer_out_of_bounds_read": "zero"}}
        }
    }
    assert payload["summary"]["translatedCount"] == 1
    assert payload["summary"]["failedCount"] == 0
    generated = (tmp_path / payload["artifacts"][0]["path"]).read_text(encoding="utf-8")
    assert "#define CROSSTL_PRIVATE_POINTER_OOB_READ_ZERO 1" in generated
    assert "values_base += int((2 * i));" in generated
    assert "? values[(values_base + int(1))] : float(0)" in generated


def test_project_opengl_private_pointer_error_policy_still_rejects_oob(tmp_path):
    payload = _translate(tmp_path, policy="error")

    assert payload["summary"]["translatedCount"] == 0
    assert payload["summary"]["failedCount"] == 1
    assert "requires at least 8 elements" in payload["artifacts"][0]["error"]
    assert not (tmp_path / payload["artifacts"][0]["path"]).exists()


@pytest.mark.parametrize("policy", ["truncate", "", "ZERO"])
def test_project_opengl_private_pointer_policy_rejects_unknown_values(tmp_path, policy):
    payload = _translate(tmp_path, policy=policy)

    assert payload["summary"]["translatedCount"] == 0
    assert payload["summary"]["failedCount"] == 1
    assert "must be 'error' or 'zero'" in payload["artifacts"][0]["error"]


def test_project_private_pointer_zero_read_policy_rejects_non_opengl_target(
    tmp_path,
):
    payload = _translate(tmp_path, policy="zero", target="metal")

    assert payload["summary"]["translatedCount"] == 0
    assert payload["summary"]["failedCount"] == 1
    assert "supported only by the OpenGL target" in payload["artifacts"][0]["error"]


def test_project_metal_const_private_pointer_provenance_enables_zero_read_policy(
    tmp_path,
):
    source = "private_pointer.metal"
    (tmp_path / source).write_text(
        textwrap.dedent("""
            float read_values(const thread float* values) {
                float result = 0.0f;
                for (int i = 0; i < 3; ++i) {
                    values += 2 * i;
                    result += values[1];
                }
                return result;
            }

            kernel void robust_private_pointer(
                device float* output [[buffer(0)]]) {
                thread float backing[7];
                output[0] = read_values(backing);
            }
            """).strip() + "\n",
        encoding="utf-8",
    )
    (tmp_path / "crosstl.toml").write_text(
        textwrap.dedent(f"""
            [project]
            source_roots = ["."]
            include = ["{source}"]
            targets = ["opengl"]
            output_dir = "out"

            [project.source_options.metal.target_options.opengl]
            private_pointer_out_of_bounds_read = "zero"
            """).strip() + "\n",
        encoding="utf-8",
    )

    payload = translate_project(
        load_project_config(tmp_path),
        format_output=False,
        validate=True,
        run_toolchains=False,
    ).to_json()

    assert payload["summary"]["translatedCount"] == 1
    assert payload["summary"]["failedCount"] == 0
    assert payload["project"]["sourceOptions"] == {
        "metal": {
            "target_options": {"opengl": {"private_pointer_out_of_bounds_read": "zero"}}
        }
    }
    generated = (tmp_path / payload["artifacts"][0]["path"]).read_text(encoding="utf-8")
    assert "#define CROSSTL_PRIVATE_POINTER_OOB_READ_ZERO 1" in generated
    assert "values_base += int((2 * i));" in generated
    assert "? values[(values_base + int(1))] : float(0)" in generated


def test_project_metal_implicit_array_template_deduction_lowers_private_pointer_base(
    tmp_path,
):
    source = "array_decay.metal"
    (tmp_path / source).write_text(
        textwrap.dedent("""
            template <typename W>
            void decode(W values) {
                for (int i = 0; i < 2; ++i) {
                    values += i;
                    values[0] = float(i);
                }
            }

            kernel void array_decay(device float* result [[buffer(0)]]) {
                float values[8];
                decode(values);
                result[0] = values[0];
            }
            """).strip() + "\n",
        encoding="utf-8",
    )
    (tmp_path / "crosstl.toml").write_text(
        textwrap.dedent(f"""
            [project]
            source_roots = ["."]
            include = ["{source}"]
            targets = ["opengl"]
            output_dir = "out"
            """).strip() + "\n",
        encoding="utf-8",
    )

    payload = translate_project(
        load_project_config(tmp_path),
        format_output=False,
        validate=True,
        run_toolchains=False,
    ).to_json()

    assert payload["summary"]["translatedCount"] == 1
    assert payload["summary"]["failedCount"] == 0
    assert payload["artifacts"][0]["templateMaterialization"]["specializations"] == [
        {
            "name": "decode",
            "materializedName": "decode_thread_float",
            "parameters": {"W": "thread float*"},
            "parameterSources": {"W": "call-site"},
            "source": "call-site",
        }
    ]
    generated = (tmp_path / payload["artifacts"][0]["path"]).read_text(encoding="utf-8")
    assert (
        "void decode_thread_float(inout float values[8], int values_base)" in generated
    )
    assert "values_base += int(i);" in generated
    assert "values[(values_base + int(0))] = float(i);" in generated
    assert "decode_thread_float(values, 0);" in generated
    assert "values +=" not in generated

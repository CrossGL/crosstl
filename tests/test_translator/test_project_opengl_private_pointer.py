from __future__ import annotations

import textwrap

import pytest

from crosstl.project import load_project_config, translate_project
from tests.test_translator.test_codegen.test_directx_codegen import (
    assert_directx_warnings_clean_if_available,
)

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


def test_project_directx_implicit_array_template_deduction_lowers_private_pointer_base(
    tmp_path,
):
    # MLX's dequantize<U, N, bits>(w, scale, bias, w_dq) deduces its trailing
    # by-value W from a local array. As in C++, the array decays to a thread
    # pointer, which DirectX lowers to an array view with a movable base.
    source = "array_decay.metal"
    (tmp_path / source).write_text(
        textwrap.dedent("""
            template <typename U, int N, typename W>
            inline void decode_block(
                const device uint8_t* w, U scale, U bias, W w_local) {
              for (int i = 0; i < (N / 4); i++) {
                w_local += 4 * i;
                for (int j = 0; j < 4; j++) {
                  w_local[j] = scale * static_cast<U>(w[4 * i + j]) + bias;
                }
              }
            }

            kernel void decode_blocks(
                const device uint8_t* w [[buffer(0)]],
                device float* result [[buffer(1)]],
                uint tid [[thread_position_in_grid]]) {
              float w_dq[8];
              decode_block<float, 8>(w + tid * 8, 0.5f, -1.0f, w_dq);
              float total = 0.0f;
              for (int i = 0; i < 8; i++) {
                total += w_dq[i];
              }
              result[tid] = total;
            }
            """).strip() + "\n",
        encoding="utf-8",
    )
    (tmp_path / "crosstl.toml").write_text(
        textwrap.dedent(f"""
            [project]
            source_roots = ["."]
            include = ["{source}"]
            targets = ["directx"]
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
            "name": "decode_block",
            "materializedName": "decode_block_float_8_thread_float",
            "parameters": {"N": "8", "U": "float", "W": "thread float*"},
            "parameterSources": {
                "N": "call-site",
                "U": "call-site",
                "W": "call-site",
            },
            "source": "call-site",
        }
    ]
    generated = (tmp_path / payload["artifacts"][0]["path"]).read_text(encoding="utf-8")
    # Offsets 0 and 4 reach element 7: one view of all eight values.
    assert generated.count("inout float w_local[8], int w_local_base)") == 2
    assert "w_local_base += int((4 * i));" in generated
    assert "w_local[(w_local_base + j)] = " in generated
    assert "w_dq, 0);" in generated
    assert "w_local +=" not in generated
    assert "float[8]" not in generated
    assert_directx_warnings_clean_if_available(generated, tmp_path)

"""``m.columns[i]`` of a Metal matrix imports as the column ``m[i]``."""

import shutil
import subprocess

import pytest

import crosstl
from crosstl.backend.Metal.matrix_columns import MetalMatrixColumnsError

SOURCE = """#include <metal_stdlib>
using namespace metal;
struct Frame { float4x4 view; float4x4 inverse_view; };
struct Columns { float4 columns; };
kernel void k(device float* o [[buffer(0)]], constant Frame& frame [[buffer(1)]],
              uint3 id [[thread_position_in_grid]]) {
{body}
}
"""


def translate(tmp_path, body, backend):
    path = tmp_path / "kernel.metal"
    path.write_text(SOURCE.replace("{body}", body))
    return crosstl.translate(str(path), backend=backend, format_output=False)


def validate_spirv(tmp_path, assembly):
    if not (shutil.which("spirv-as") and shutil.which("spirv-val")):
        return
    (tmp_path / "out.spvasm").write_text(assembly)
    for command in (
        ["spirv-as", "--target-env", "vulkan1.1", "out.spvasm", "-o", "out.spv"],
        ["spirv-val", "--target-env", "vulkan1.1", "out.spv"],
    ):
        result = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True)
        assert result.returncode == 0, result.stderr


def test_matrix_columns_are_the_matrix_columns(tmp_path):
    body = (
        "    float3 eye = frame.view.columns[3].xyz;\n"
        "    float4x4 m = frame.inverse_view;\n"
        "    m.columns[0] = float4(eye, 1.0);\n"
        "    float3x3 n = float3x3(frame.view.columns[0].xyz, frame.view.columns[1].xyz,\n"
        "                          frame.view.columns[2].xyz);\n"
        "    Columns s;\n"
        "    s.columns = m.columns[1];\n"
        "    o[id.x] = eye.x + n[1].y + m.columns[0].x + s.columns.w;"
    )
    cgl = translate(tmp_path, body, "cgl")
    assert "columns[" not in cgl
    assert "vec3 eye = frame.view[3].xyz;" in cgl
    assert "m[0] = vec4(eye, 1.0);" in cgl
    assert "mat3(frame.view[0].xyz, frame.view[1].xyz, frame.view[2].xyz)" in cgl
    # A structure member named columns is an ordinary member.
    assert "s.columns = m[1];" in cgl
    validate_spirv(tmp_path, translate(tmp_path, body, "vulkan"))


def test_columns_element_reads_select_the_column_first(tmp_path):
    body = "    o[id.x] = frame.inverse_view.columns[3][1] * frame.view.columns[0][2];"
    cgl = translate(tmp_path, body, "cgl")
    assert "frame.inverse_view[3]" in cgl and "frame.view[0]" in cgl
    assert "columns" not in cgl.split("struct Columns", 1)[0]
    validate_spirv(tmp_path, translate(tmp_path, body, "vulkan"))


def test_the_columns_array_as_a_value_is_refused(tmp_path):
    body = (
        "    float4x4 m = frame.view;\n"
        "    thread const float4* columns = m.columns;\n"
        "    o[id.x] = columns[0].x;"
    )
    with pytest.raises(MetalMatrixColumnsError) as raised:
        translate(tmp_path, body, "cgl")
    assert raised.value.project_diagnostic_code == (
        "project.translate.metal-matrix-columns-unsupported"
    )

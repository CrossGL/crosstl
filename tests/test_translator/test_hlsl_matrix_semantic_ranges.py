"""A matrix varying takes one semantic index per column (row for row_major) in HLSL signatures."""

import shutil
import subprocess

import pytest

import crosstl
from crosstl.translator.codegen.interface_conflicts import (
    InterfaceLocationConflictError,
)

SOURCE = """struct VSOutput
{{
    float4 Pos : SV_POSITION;
    float4x4 view : MATRIX0;
    float4x4 projection : MATRIX{second};
}};
VSOutput main(uint id : SV_VertexID)
{{
    VSOutput output;
    output.Pos = float4(float(id), 0.0, 0.0, 1.0);
    output.view = (float4x4)0;
    output.projection = (float4x4)0;
    return output;
}}
"""


def translate(tmp_path, second):
    path = tmp_path / "shader.vert.hlsl"
    path.write_text(SOURCE.format(second=second))
    return crosstl.translate(
        str(path), backend="directx", source_backend="directx", format_output=False
    )


def test_overlapping_matrix_semantics_are_refused(tmp_path):
    # MATRIX0 covers MATRIX0-MATRIX3, so MATRIX1 overlaps it; DXC rejected the output ("Semantic 'MATRIX' overlap").
    with pytest.raises(
        InterfaceLocationConflictError, match="MATRIX1-MATRIX4 overlaps"
    ):
        translate(tmp_path, 1)


def test_disjoint_matrix_semantics_are_kept(tmp_path):
    hlsl = translate(tmp_path, 4)
    assert "MATRIX4" in hlsl
    if shutil.which("dxc"):
        (tmp_path / "out.hlsl").write_text(hlsl)
        result = subprocess.run(
            ["dxc", "-T", "vs_6_0", "-E", "VSMain", "out.hlsl"],
            cwd=tmp_path,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stdout + result.stderr

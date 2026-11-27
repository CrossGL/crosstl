"""DirectX and Slang generators type WaveReadLaneAt's lane index as an integer.

The lane argument of ``WaveReadLaneAt`` and ``QuadReadLaneAt`` must remain an
integer type regardless of whether the value being read is a scalar or a vector
(#4405).
"""

from __future__ import annotations

import shutil
import subprocess

import pytest

import crosstl

SHADER_TEMPLATE = """
shader Rotate {{
    compute {{
        layout(local_size_x = 32, local_size_y = 1, local_size_z = 1) in;
        buffer {val_type}* values;

        void main() {{
            uint i = gl_GlobalInvocationID.x;
            values[i] = {expr};
        }}
    }}
}}
"""


def _translate(tmp_path, val_type, expr, backend):
    source_path = tmp_path / f"shader_{backend}.cgl"
    source_path.write_text(SHADER_TEMPLATE.format(val_type=val_type, expr=expr))
    return crosstl.translate(str(source_path), backend=backend, format_output=False)


@pytest.mark.parametrize("val_type", ["vec2", "vec4"])
def test_directx_wave_read_lane_at_vector_value(tmp_path, val_type):
    expr = "WaveReadLaneAt(values[i], (WaveGetLaneIndex() + 1u) % WaveGetLaneCount())"
    hlsl = _translate(tmp_path, val_type, expr, "directx")

    assert (
        "WaveReadLaneAt(values[i], ((WaveGetLaneIndex() + 1u) % WaveGetLaneCount()))"
        in hlsl
    )

    dxc = shutil.which("dxc")
    if dxc:
        path = tmp_path / "out.hlsl"
        path.write_text(hlsl, encoding="utf-8")
        result = subprocess.run(
            [dxc, "-T", "cs_6_0", "-E", "CSMain", str(path), "-Fo", str(path) + ".o"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("val_type", ["vec2", "vec4"])
def test_slang_wave_read_lane_at_vector_value(tmp_path, val_type):
    expr = "WaveReadLaneAt(values[i], (WaveGetLaneIndex() + 1u) % WaveGetLaneCount())"
    slang = _translate(tmp_path, val_type, expr, "slang")

    assert (
        "WaveReadLaneAt(values[i], (WaveGetLaneIndex() + 1u) % WaveGetLaneCount())"
        in slang
    )

    slangc = shutil.which("slangc")
    if slangc:
        path = tmp_path / "out.slang"
        path.write_text(slang, encoding="utf-8")
        result = subprocess.run(
            [
                slangc,
                "-target",
                "spirv",
                "-stage",
                "compute",
                "-entry",
                "main",
                str(path),
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("val_type", ["float", "int"])
def test_slang_wave_read_lane_at_scalar_value_keeps_integer_lane(tmp_path, val_type):
    expr = "WaveReadLaneAt(values[i], (WaveGetLaneIndex() + 1u) % WaveGetLaneCount())"
    slang = _translate(tmp_path, val_type, expr, "slang")

    assert "float(WaveGetLaneIndex())" not in slang
    assert "float(WaveGetLaneCount())" not in slang
    assert "int(WaveGetLaneIndex())" not in slang
    assert "int(WaveGetLaneCount())" not in slang
    assert (
        "WaveReadLaneAt(values[i], (WaveGetLaneIndex() + 1u) % WaveGetLaneCount())"
        in slang
    )


@pytest.mark.parametrize("backend", ["directx", "slang"])
def test_nested_wave_read_lane_at(tmp_path, backend):
    expr = "WaveReadLaneAt(WaveReadLaneAt(values[i], 1u), WaveGetLaneIndex())"
    code = _translate(tmp_path, "vec2", expr, backend)

    assert "WaveReadLaneAt(WaveReadLaneAt(values[i], 1u), WaveGetLaneIndex())" in code


@pytest.mark.parametrize("backend", ["directx", "slang"])
def test_quad_read_lane_at_with_wave_expression(tmp_path, backend):
    expr = "QuadReadLaneAt(values[i], WaveGetLaneIndex() % 4u)"
    code = _translate(tmp_path, "vec2", expr, backend)

    assert "QuadReadLaneAt(values[i]" in code
    assert "WaveGetLaneIndex()" in code

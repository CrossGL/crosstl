"""``gl_PointCoord`` lowers to Metal's ``[[point_coord]]`` and is refused on HLSL.

Both generators used to emit the GLSL name verbatim, which left an undeclared
identifier. Metal fragment functions receive the point coordinate as a
``[[point_coord]]`` input. Direct3D 10 and later have no point sprites, so
HLSL has no counterpart and the use is refused.
"""

import shutil
import subprocess
import textwrap

import pytest

import crosstl

GLSL = textwrap.dedent("""
    #version 450
    layout(location = 0) in vec4 inColor;
    layout(location = 0) out vec4 outFragColor;
    void main() {
        vec2 centered = gl_PointCoord - vec2(0.5);
        float falloff = 1.0 - clamp(dot(centered, centered) * 4.0, 0.0, 1.0);
        outFragColor = inColor * falloff;
    }
    """)


def _translate(tmp_path, backend):
    path = tmp_path / "particle.frag"
    path.write_text(GLSL, encoding="utf-8")
    return crosstl.translate(str(path), backend=backend)


def test_metal_reads_the_point_coordinate_input(tmp_path):
    code = _translate(tmp_path, "metal")
    assert "float2 _crossglPointCoord [[point_coord]]" in code
    assert "gl_PointCoord" not in code
    if shutil.which("xcrun") is None:
        pytest.skip("xcrun is not installed")
    (tmp_path / "particle.metal").write_text(code, encoding="utf-8")
    result = subprocess.run(
        ["xcrun", "-sdk", "macosx", "metal", "-c", "particle.metal", "-o", "p.air"],
        capture_output=True,
        text=True,
        timeout=300,
        cwd=tmp_path,
    )
    if "unable to find utility" in result.stderr:
        pytest.skip("the Metal toolchain is not installed")
    assert result.returncode == 0, result.stderr[-2000:]


def test_hlsl_refuses_the_point_coordinate(tmp_path):
    with pytest.raises(ValueError, match="HLSL has no point coordinate input"):
        _translate(tmp_path, "directx")


@pytest.mark.parametrize(
    ("backend", "spelling"),
    [("opengl", "gl_PointCoord"), ("vulkan", "BuiltIn PointCoord")],
)
def test_glsl_and_spirv_keep_the_point_coordinate(tmp_path, backend, spelling):
    assert spelling in _translate(tmp_path, backend)

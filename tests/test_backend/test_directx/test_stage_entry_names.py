"""HLSL entry points that are not named ``main`` keep their body through CrossGL.

The importer writes each HLSL entry into its CrossGL stage block. CrossGL takes a stage function named ``main`` as
the entry and needs ``@ stage_entry`` on any other; without it a target synthesized an empty entry and wrote the
real one as an unused helper, so ``float4 frag(...) : SV_Target`` became ``void main() {}`` in GLSL.
"""

from __future__ import annotations

import shutil
import subprocess

import pytest

import crosstl

FRAGMENT = """
float4 frag(float4 c : COLOR0) : SV_Target { return c * 2.0; }
"""

VERTEX = """
float4 VertexMain(float4 p : POSITION) : SV_Position { return p * 0.5; }
"""


def _translate(tmp_path, source, backend):
    path = tmp_path / "entry.hlsl"
    path.write_text(source, encoding="utf-8")
    return crosstl.translate(str(path), backend=backend, format_output=False)


def test_fragment_entry_not_named_main_is_marked_stage_entry(tmp_path):
    crossgl = _translate(tmp_path, FRAGMENT, "cgl")
    assert "vec4 frag(vec4 c @ Color0) @ gl_FragColor @ stage_entry {" in crossgl


@pytest.mark.parametrize("source", (FRAGMENT, VERTEX), ids=("fragment", "vertex"))
def test_glsl_entry_holds_the_hlsl_entry_body(tmp_path, source):
    glsl = _translate(tmp_path, source, "opengl")
    assert "void main() {}" not in glsl
    assert "* 2.0" in glsl or "* 0.5" in glsl
    # The entry is the GLSL main itself, not a helper beside an empty main.
    assert "vec4 frag(" not in glsl
    assert "vec4 VertexMain(" not in glsl


def test_hlsl_round_trip_keeps_the_pixel_entry_body(tmp_path):
    hlsl = _translate(tmp_path, FRAGMENT, "directx")
    assert "return (c * 2.0);" in hlsl
    assert "void PSMain()" not in hlsl


@pytest.mark.skipif(
    shutil.which("glslangValidator") is None, reason="glslangValidator is not available"
)
def test_glslang_accepts_the_fragment_entry(tmp_path):
    shader = tmp_path / "entry.frag"
    shader.write_text(_translate(tmp_path, FRAGMENT, "opengl"), encoding="utf-8")
    result = subprocess.run(
        ["glslangValidator", str(shader)], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stdout + result.stderr

"""Slang entry points without ``[shader(...)]`` take their stage from the result semantic."""

import shutil
import subprocess

import pytest

import crosstl

FRAGMENT_ENTRY = "float4 main(float4 pos : SV_Position) : SV_Target { return pos; }"
VERTEX_ENTRY = "float4 main(float3 position : POSITION) : SV_Position { return float4(position, 1.0); }"
HELPER_ONLY = "float4 shade(float4 color) { return color; }"


def _translate(tmp_path, source, backend):
    path = tmp_path / "shader.slang"
    path.write_text(source, encoding="utf-8")
    return crosstl.translate(str(path), backend=backend, format_output=False)


def _validate_glsl(tmp_path, code, stage):
    validator = shutil.which("glslangValidator")
    if validator is None:
        return
    path = tmp_path / f"shader.{stage}"
    path.write_text(code, encoding="utf-8")
    result = subprocess.run(
        [validator, str(path)], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_sv_target_result_makes_a_fragment_entry(tmp_path):
    crossgl = _translate(tmp_path, FRAGMENT_ENTRY, "cgl")
    assert "fragment {" in crossgl
    assert "pos @ gl_FragCoord" in crossgl

    opengl = _translate(tmp_path, FRAGMENT_ENTRY, "opengl")
    assert "void main()" in opengl
    assert "fragColor = gl_FragCoord;" in opengl
    _validate_glsl(tmp_path, opengl, "frag")


def test_sv_position_result_makes_a_vertex_entry(tmp_path):
    crossgl = _translate(tmp_path, VERTEX_ENTRY, "cgl")
    assert "vertex {" in crossgl

    opengl = _translate(tmp_path, VERTEX_ENTRY, "opengl")
    assert "// Vertex Shader" in opengl
    assert "void main()" in opengl


@pytest.mark.parametrize(
    "source",
    [
        VERTEX_ENTRY,
        '[shader("vertex")]\n' + VERTEX_ENTRY,
        VERTEX_ENTRY.replace("SV_Position", "SV_POSITION"),
    ],
    ids=["by-result-semantic", "shader-attribute", "upper-case-semantic"],
)
def test_sv_position_result_writes_gl_position(tmp_path, source):
    # The clip position is the built-in, as the HLSL importer maps it; a
    # user varying left nothing rasterized at the intended position.
    crossgl = _translate(tmp_path, source, "cgl")
    assert "@ gl_Position" in crossgl
    assert "Out_Position" not in crossgl

    opengl = _translate(tmp_path, source, "opengl")
    assert "gl_Position = vec4(position, 1.0);" in opengl
    assert "vertexOutput" not in opengl
    _validate_glsl(tmp_path, opengl, "vert")


@pytest.mark.parametrize("stage", ["fragment {", "vertex {", "compute {"])
def test_function_without_result_semantic_gets_no_stage(tmp_path, stage):
    assert stage not in _translate(tmp_path, HELPER_ONLY, "cgl")

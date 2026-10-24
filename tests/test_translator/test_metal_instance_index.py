"""GLSL ``gl_InstanceIndex`` becomes a Metal ``[[instance_id]]`` stage input.

The Metal generator added a ``[[vertex_id]]`` parameter for ``gl_VertexIndex`` but none for ``gl_InstanceIndex``,
which reached MSL as an undeclared identifier (sascha-vulkan ``gears.vert``, ``deferred/mrt.vert`` and others).
Vulkan's ``InstanceIndex`` and MSL's ``instance_id`` both count from the base instance.
"""

import shutil
import subprocess

import pytest

import crosstl

VERTEX = """#version 450
layout (location = 0) in vec4 inPos;
layout (binding = 0) uniform UBO { mat4 model[4]; } ubo;
layout (location = 0) out vec3 outColor;
void main()
{
    outColor = vec3(float(gl_VertexIndex), float(gl_InstanceIndex), 0.0);
    gl_Position = ubo.model[gl_InstanceIndex] * inPos;
}
"""


def _translate(tmp_path, source, suffix):
    path = tmp_path / f"shader.{suffix}"
    path.write_text(source)
    return crosstl.translate(str(path), backend="metal", format_output=False)


def _compile(tmp_path, code):
    if shutil.which("xcrun") is None:
        pytest.skip("the Metal toolchain is not installed")
    found = subprocess.run(
        ["xcrun", "-sdk", "macosx", "--find", "metal"], capture_output=True
    )
    if found.returncode != 0:
        pytest.skip("the Metal toolchain is not installed")
    artifact = tmp_path / "out.metal"
    artifact.write_text(code)
    result = subprocess.run(
        ["xcrun", "-sdk", "macosx", "metal", "-c", str(artifact)]
        + ["-o", str(tmp_path / "out.air")],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_instance_index_is_the_instance_id_input(tmp_path):
    code = _translate(tmp_path, VERTEX, "vert")
    assert "uint _crossglInstanceID [[instance_id]]" in code
    assert "uint _crossglVertexID [[vertex_id]]" in code
    assert "gl_InstanceIndex" not in code
    assert "model[_crossglInstanceID]" in code
    _compile(tmp_path, code)

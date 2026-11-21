"""GLSL output: ``array.length()`` on an array of structures is the array's method, not a structure member."""

import shutil
import subprocess

import pytest

import crosstl
from crosstl.translator.codegen.lowering_refusals import LoweringRefused

SOURCE = """#version 450
layout (local_size_x = 16) in;
struct Light { vec4 position; vec3 color; };
layout (std430, binding = 0) buffer Lights { Light lights[]; };
layout (std430, binding = 1) buffer Out { vec4 values[]; };
void main() {
    Light fixedLights[3];
    vec4 sum = vec4(float(fixedLights.length()));
    for (int i = 0; i < lights.length(); i++) { sum += lights[i].position; }
    values[gl_GlobalInvocationID.x] = sum;
}
"""


def translate(tmp_path, text):
    path = tmp_path / "probe.comp"
    path.write_text(text)
    return crosstl.translate(str(path), backend="opengl", source_backend="opengl")


def validate(tmp_path, glsl):
    if shutil.which("glslangValidator") is None:
        return
    path = tmp_path / "out.comp"
    path.write_text(glsl)
    checked = subprocess.run(
        [
            "glslangValidator",
            "--target-env",
            "vulkan1.2",
            str(path),
            "-o",
            str(tmp_path / "out.spv"),
        ],
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )
    assert checked.returncode == 0, checked.stdout + checked.stderr


def test_length_of_a_runtime_array_of_structures_is_kept(tmp_path):
    glsl = translate(tmp_path, SOURCE)
    assert ".lights.length()" in glsl
    validate(tmp_path, glsl)


def test_a_member_the_structure_does_not_declare_is_still_refused(tmp_path):
    text = SOURCE.replace("sum += lights[i].position;", "sum += lights[i].length;")
    with pytest.raises(LoweringRefused, match="member 'length' is not declared"):
        translate(tmp_path, text)

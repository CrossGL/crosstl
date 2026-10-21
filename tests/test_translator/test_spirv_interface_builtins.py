"""SPIR-V stage interfaces: per-vertex built-ins and the locations structure members occupy.

The SPIR-V importer declares glslang's ``gl_PerVertex`` members as stage-level outputs, and a stage structure's
array or matrix member occupies one location per element or column. Both used to collide with user locations.
"""

import re
import shutil
import subprocess

import pytest

import crosstl

PER_VERTEX_CGL = """
shader main {
    vertex {
        vec3 inPos @input @location(0);
        vec2 outUV @output @location(0);
        vec4 gl_Position @output @gl_Position;
        float gl_PointSize @output @gl_PointSize;
        float gl_ClipDistance[2] @output @gl_ClipDistance;
        float gl_CullDistance[1] @output @gl_CullDistance;
        vec3 outLight @output @location(2);
        vec3 outNormal @output @location(1);
        void main() {
            outUV = inPos.xy;
            outNormal = inPos;
            outLight = -inPos;
            gl_Position = vec4(inPos, 1.0);
            gl_PointSize = 2.0;
            gl_ClipDistance[1] = inPos.x;
        }
    }
}
"""

MATRIX_MEMBER_CGL = """
shader main {
    struct VSOutput {
        vec4 Pos @ gl_Position;
        mat3 frame @ TexCoord0;
        vec2 uv @ TexCoord1;
    }
    vertex {
    VSOutput main(vec3 p @ Position) {
        VSOutput output;
        output.Pos = vec4(p, 1.0);
        output.frame = mat3(1.0);
        output.uv = p.xy;
        return output;
    }
    }
}
"""


def _translate(tmp_path, source):
    path = tmp_path / "shader.cgl"
    path.write_text(source)
    return crosstl.translate(str(path), backend="vulkan")


def _validate(tmp_path, code):
    if not (shutil.which("spirv-as") and shutil.which("spirv-val")):
        return
    asm, binary = tmp_path / "out.spvasm", tmp_path / "out.spv"
    asm.write_text(code)
    for command in (
        ["spirv-as", "--target-env", "vulkan1.0", str(asm), "-o", str(binary)],
        ["spirv-val", "--target-env", "vulkan1.0", str(binary)],
    ):
        result = subprocess.run(command, capture_output=True, text=True)
        assert result.returncode == 0, result.stdout + result.stderr


def _decorations(code, kind):
    # Without SPIRV-Tools the module keeps numeric ids; name them through OpName.
    names = dict(re.findall(r'OpName %(\w+) "(\w+)"', code))
    found = re.findall(rf"OpDecorate %(\w+) {kind} (\w+)", code)
    return {names.get(target, target): value for target, value in found}


def test_per_vertex_outputs_are_built_ins_not_locations(tmp_path):
    code = _translate(tmp_path, PER_VERTEX_CGL)

    assert _decorations(code, "BuiltIn") == {
        "gl_Position": "Position",
        "gl_PointSize": "PointSize",
        "gl_ClipDistance": "ClipDistance",
        "gl_CullDistance": "CullDistance",
    }
    assert _decorations(code, "Location") == {
        "inPos": "0",
        "outUV": "0",
        "outLight": "2",
        "outNormal": "1",
    }
    assert "OpCapability ClipDistance" in code
    assert "OpCapability CullDistance" in code
    _validate(tmp_path, code)


def test_clip_distance_must_be_a_float_array(tmp_path):
    source = PER_VERTEX_CGL.replace(
        "float gl_ClipDistance[2] @output", "int gl_ClipDistance[2] @output"
    ).replace("gl_ClipDistance[1] = inPos.x;", "gl_ClipDistance[1] = 1;")

    with pytest.raises(ValueError, match="sized float array"):
        _translate(tmp_path, source)


def test_structure_members_after_a_matrix_take_later_locations(tmp_path):
    code = _translate(tmp_path, MATRIX_MEMBER_CGL)

    locations = _decorations(code, "Location")
    assert locations["CrossGL_vertex_output_main_frame"] == "1"
    assert locations["CrossGL_vertex_output_main_uv"] == "4"
    _validate(tmp_path, code)

"""Zero values of GLSL arrays give one constructor argument per element (GLSL 4.60 section 5.4.4)."""

import shutil
import subprocess

import crosstl

SOURCE = """#version 450
layout(local_size_x = 1) in;
layout(std430, binding = 0) buffer O { float o[]; };
struct Inner { vec2 v[2]; int k; };
struct Data { float avg[3]; Inner inner[2]; };
Data data;
float table[4];
void main() {
    data.avg[0] = 1.0;
    o[0] = data.avg[1] + data.inner[1].v[0].x + table[2] + float(data.inner[0].k);
}
"""


def test_zero_initialized_globals_construct_every_element(tmp_path):
    path = tmp_path / "kernel.comp"
    path.write_text(SOURCE)
    glsl = " ".join(crosstl.translate(str(path), backend="opengl").split())
    # `float[3](0)` used to be emitted, which glslang rejects ("array constructor needs one argument per element").
    assert "float[3](0.0, 0.0, 0.0)" in glsl
    assert (
        "Inner[2](Inner(vec2[2](vec2(0.0), vec2(0.0)), 0), Inner(vec2[2](vec2(0.0), vec2(0.0)), 0))"
        in glsl
    )
    assert "float table[4] = float[4](0.0, 0.0, 0.0, 0.0);" in glsl
    assert "](0)" not in glsl
    if shutil.which("glslangValidator"):
        (tmp_path / "out.comp").write_text(
            crosstl.translate(str(path), backend="opengl")
        )
        result = subprocess.run(
            ["glslangValidator", "-S", "comp", "out.comp"],
            cwd=tmp_path,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stdout + result.stderr

"""A flattened member of a named uniform block keeps its meaning when a function uses the same name."""

import shutil
import subprocess

import crosstl

SOURCE = """#version 450
layout(location = 0) in vec2 inUV;
layout(location = 0) out vec4 outColor;
layout(push_constant) uniform Push { vec4 color; int mode; } push;
layout(binding = 0) uniform Scene { mat4 model; } scene;
vec4 tint(mat4 model, vec4 color) { return model * color; }
void main() {
    vec4 color = vec4(inUV, 0.0, 1.0);
    if (push.mode == 1) {
        color = push.color * 0.5;
    }
    outColor = tint(scene.model, color);
}
"""


def translate(tmp_path, backend):
    path = tmp_path / "shader.frag"
    path.write_text(SOURCE)
    return crosstl.translate(str(path), backend=backend)


def test_members_that_a_function_also_names_are_renamed(tmp_path):
    cgl = translate(tmp_path, "cgl")
    # push.color used to become `color`, which read the local instead of the block member.
    assert "vec4 Push_color;" in cgl and "color = (Push_color * 0.5);" in cgl
    assert "mat4 Scene_model;" in cgl and "tint(Scene_model, color)" in cgl
    # A member no function names keeps its name.
    assert "int mode;" in cgl and "(mode == 1)" in cgl


def test_hlsl_and_glsl_read_the_block_members(tmp_path):
    hlsl = translate(tmp_path, "directx")
    assert "(Push_color * 0.5)" in hlsl
    glsl = translate(tmp_path, "opengl")
    assert "Push_color" in glsl
    if shutil.which("glslangValidator"):
        (tmp_path / "out.frag").write_text(glsl)
        result = subprocess.run(
            ["glslangValidator", "-S", "frag", "out.frag"],
            cwd=tmp_path,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stdout + result.stderr
    if shutil.which("dxc"):
        (tmp_path / "out.hlsl").write_text(hlsl)
        result = subprocess.run(
            ["dxc", "-T", "ps_6_0", "-E", "PSMain", "out.hlsl"],
            cwd=tmp_path,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stdout + result.stderr

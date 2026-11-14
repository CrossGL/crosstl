"""A local named after a member of a named uniform block is renamed once the block is flattened."""

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


def test_locals_named_after_block_members_are_renamed(tmp_path):
    cgl = translate(tmp_path, "cgl")
    # push.color used to become `color`, which read the local instead of the block member. The members keep
    # their names (the host may bind them by name); the locals and parameters are renamed.
    assert "vec4 color;" in cgl and "mat4 model;" in cgl
    assert "vec4 color_2 = vec4(input.inUV, 0.0, 1.0);" in cgl
    assert "color_2 = (color * 0.5);" in cgl
    assert (
        "vec4 tint(mat4 model_1, vec4 color_1)" in cgl and "(model_1 * color_1)" in cgl
    )
    assert "tint(model, color_2)" in cgl and "(mode == 1)" in cgl


def test_hlsl_and_glsl_read_the_block_members(tmp_path):
    hlsl = translate(tmp_path, "directx")
    assert "(color * 0.5)" in hlsl and "color_2" in hlsl
    glsl = translate(tmp_path, "opengl")
    assert "(color * 0.5)" in glsl and "color_2" in glsl
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

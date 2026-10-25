"""``gl_InstanceIndex`` (Vulkan GLSL) for OpenGL, OpenGL ES and HLSL targets."""

import re
import shutil
import subprocess

import crosstl

SOURCE = """#version {version}
layout(location = 0) in vec4 inPos;
layout(binding = 0) uniform UBO {{ mat4 mvp; vec4 instancePos[3]; }} ubo;
layout(location = 0) out flat int outInstance;
void main() {{
    gl_Position = ubo.mvp * (inPos + ubo.instancePos[gl_InstanceIndex]);
    outInstance = gl_InstanceIndex + gl_VertexIndex;
}}
"""


def translate(tmp_path, backend, version=450):
    path = tmp_path / "shader.vert"
    path.write_text(SOURCE.format(version=version))
    return crosstl.translate(str(path), backend=backend, format_output=False)


def _validate_glsl(tmp_path, glsl):
    if shutil.which("glslangValidator"):
        (tmp_path / "out.vert").write_text(glsl)
        result = subprocess.run(
            ["glslangValidator", "-S", "vert", "out.vert"],
            cwd=tmp_path,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stdout + result.stderr


def test_opengl_adds_the_base_instance(tmp_path):
    # gl_InstanceID starts at 0 for every draw; gl_InstanceIndex starts at the draw's first instance.
    glsl = translate(tmp_path, "opengl")
    assert "#extension GL_ARB_shader_draw_parameters : require" in glsl
    assert "instancePos[(gl_InstanceID + gl_BaseInstanceARB)]" in glsl
    assert "gl_InstanceIndex" not in glsl
    _validate_glsl(tmp_path, glsl)


def test_glsl_460_uses_the_core_base_instance(tmp_path):
    glsl = translate(tmp_path, "opengl", version=460)
    assert "(gl_InstanceID + gl_BaseInstance)" in glsl
    assert "GL_ARB_shader_draw_parameters" not in glsl
    _validate_glsl(tmp_path, glsl)


def test_glsl_es_draws_have_no_base_instance(tmp_path):
    essl = translate(tmp_path, "webgl")
    assert "instancePos[gl_InstanceID]" in essl
    assert "gl_InstanceIndex" not in essl and "gl_BaseInstance" not in essl
    _validate_glsl(tmp_path, essl)


def test_hlsl_reads_the_instance_id_system_value(tmp_path):
    hlsl = translate(tmp_path, "directx")
    assert "uint gl_InstanceIndex : SV_InstanceID" in hlsl
    if shutil.which("dxc"):
        (tmp_path / "out.hlsl").write_text(hlsl)
        result = subprocess.run(
            ["dxc", "-T", "vs_6_0", "-E", "VSMain", "out.hlsl"],
            cwd=tmp_path,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stdout + result.stderr


def test_spirv_reads_the_instance_index(tmp_path):
    # SPIR-V's InstanceIndex includes the first instance, as gl_InstanceIndex does; it used to be refused.
    path = tmp_path / "shader.vert"
    path.write_text(SOURCE.format(version=450))
    spirv = crosstl.translate(str(path), backend="vulkan", format_output=False)
    assert re.search(r"OpDecorate %\w+ BuiltIn InstanceIndex\n", spirv)
    if shutil.which("spirv-as") and shutil.which("spirv-val"):
        (tmp_path / "out.spvasm").write_text(spirv)
        for command in (
            ["spirv-as", "--target-env", "vulkan1.1", "out.spvasm", "-o", "o.spv"],
            ["spirv-val", "--target-env", "vulkan1.1", "o.spv"],
        ):
            result = subprocess.run(
                command, cwd=tmp_path, capture_output=True, text=True
            )
            assert result.returncode == 0, result.stderr


def test_metal_keeps_the_base_instance(tmp_path):
    metal = translate(tmp_path, "metal")
    assert "[[instance_id]]" in metal

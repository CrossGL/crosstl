"""GLSL specialization constants compile both through SPIR-V and as OpenGL source.

``layout(constant_id = N)`` exists only when GLSL is compiled to SPIR-V (GLSL 4.60 section 4.4.5, GL_ARB_gl_spirv,
GL_KHR_vulkan_glsl): an OpenGL or OpenGL ES driver compiling the emitted source rejected it. The emitter now guards
the qualifier with ``GL_SPIRV``/``VULKAN`` and falls back to the default value, which is what the shader computes
when no specialization is supplied.
"""

from __future__ import annotations

import shutil
import subprocess

import pytest

import crosstl

SOURCE = """#version 450
layout(constant_id = 3) const int taps = 5;
layout(constant_id = 4) const float scale = 0.5;
layout(location = 0) out vec4 color;
void main() {
    float sum = 0.0;
    for (int i = 0; i < taps; i++) { sum += scale; }
    color = vec4(sum);
}
"""

GLSLANG = shutil.which("glslangValidator")


def translate(tmp_path, backend):
    path = tmp_path / "shader.frag"
    path.write_text(SOURCE)
    return crosstl.translate(str(path), backend=backend, source_backend="opengl")


def glslang(tmp_path, code, *flags):
    path = tmp_path / "out.frag"
    path.write_text(code)
    return subprocess.run(
        [GLSLANG, *flags, str(path)], capture_output=True, text=True, cwd=tmp_path
    )


def test_specialization_constants_are_guarded_for_spirv_compilation(tmp_path):
    glsl = translate(tmp_path, "opengl")
    assert "#if defined(GL_SPIRV) || defined(VULKAN)\n" in glsl
    assert (
        "layout(constant_id = 3) const int taps = 5;\n#else\nconst int taps = 5;\n#endif"
        in glsl
    )


@pytest.mark.skipif(GLSLANG is None, reason="glslangValidator is not installed")
def test_guarded_glsl_compiles_for_opengl_and_keeps_spec_ids_in_spirv(tmp_path):
    glsl = translate(tmp_path, "opengl")
    opengl = glslang(tmp_path, glsl)
    assert opengl.returncode == 0, opengl.stdout + opengl.stderr
    vulkan = glslang(tmp_path, glsl, "-V", "-H", "-o", str(tmp_path / "out.spv"))
    assert vulkan.returncode == 0, vulkan.stdout + vulkan.stderr
    assert "SpecId 3" in vulkan.stdout and "SpecId 4" in vulkan.stdout
    gl_spirv = glslang(tmp_path, glsl, "-G", "-o", str(tmp_path / "gl.spv"))
    assert gl_spirv.returncode == 0, gl_spirv.stdout + gl_spirv.stderr


@pytest.mark.skipif(GLSLANG is None, reason="glslangValidator is not installed")
def test_guarded_glsl_es_compiles(tmp_path):
    essl = translate(tmp_path, "webgl")
    assert "#else\nconst int taps = 5;\n#endif" in essl
    result = glslang(tmp_path, essl)
    assert result.returncode == 0, result.stdout + result.stderr


# Source shape: SaschaWillems/Vulkan glsl/ssao/ssao.frag, whose kernel uniform block is sized by a spec constant.
SIZED_BLOCK_SOURCE = """#version 450
layout(constant_id = 0) const int KERNEL_SIZE = 64;
layout(binding = 3) uniform UBOKernel { vec4 samples[KERNEL_SIZE]; } kernel;
layout(location = 0) out vec4 color;
void main() {
    vec4 sum = vec4(0.0);
    for (int i = 0; i < KERNEL_SIZE; i++) { sum += kernel.samples[i]; }
    color = sum;
}
"""

# std430 places a float array at stride 4 and the std140 declaration at stride 16: register storage would have to
# fix the array's count, so a spec-constant-sized array cannot be declared.
SIZED_STD430_BLOCK_CROSSGL = """
shader main {
    const int COUNT @constant_id(0) = 4;
    cbuffer Weights @binding(0) @buffer_layout(std430) { float weights[COUNT]; };
    fragment {
        vec4 main() @location(0) { return vec4(weights[1]); }
    }
}
"""


@pytest.mark.parametrize("backend", ["opengl", "webgl"])
def test_std140_block_array_sized_by_spec_constant_keeps_the_constant(
    tmp_path, backend
):
    path = tmp_path / "shader.frag"
    path.write_text(SIZED_BLOCK_SOURCE)
    glsl = crosstl.translate(str(path), backend=backend, source_backend="opengl")

    assert "vec4 samples[KERNEL_SIZE];" in glsl
    assert "#else\nconst int KERNEL_SIZE = 64;\n#endif" in glsl
    if GLSLANG is not None:
        result = glslang(tmp_path, glsl)
        assert result.returncode == 0, result.stdout + result.stderr


def test_non_std140_block_array_sized_by_spec_constant_is_refused(tmp_path):
    path = tmp_path / "shader.cgl"
    path.write_text(SIZED_STD430_BLOCK_CROSSGL)

    with pytest.raises(Exception, match="needs a constant size"):
        crosstl.translate(str(path), backend="opengl")

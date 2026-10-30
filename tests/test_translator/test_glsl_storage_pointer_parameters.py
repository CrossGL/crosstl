"""CrossGL ``buffer T*`` helper parameters reach GLSL as buffer-specialized helpers.

GLSL has no pointers, and an array parameter must be explicitly sized (GLSL 4.60 section 6.1), so a helper that
takes a storage pointer is specialized for the buffer each call passes and receives an element offset, as MSL
``device T*`` helper parameters already are.
"""

import shutil
import subprocess

import pytest

from crosstl.translator import parse
from crosstl.translator.codegen.GLSL_codegen import GLSLCodeGen

BUFFER_HELPERS = """
shader BufferHelpers {
    struct Item { float a; vec4 b; }
    float readAt(buffer float* values, int index) { return values[index]; }
    float readAlias(buffer float* values, int index) { return readAt(values, index); }
    void writeAt(buffer float* values, int index, float value) { values[index] = value; }
    float readItem(const buffer Item* q, int i) { return q[i].a + q[i].b.x; }
    void bumpNext(buffer float* p, int i) { p[i + 1] = p[i] + 1.0; }
    compute {
        layout(local_size_x = 64) in;
        buffer float* inputValues;
        buffer float* outputValues;
        buffer Item* items;
        void main(uint3 gid @ gl_GlobalInvocationID) {
            float value = readAlias(inputValues, int(gid.x));
            writeAt(outputValues, int(gid.x), value + readItem(items, int(gid.x)));
            bumpNext(outputValues + 2, 1);
            writeAt(inputValues, 0, readAt(outputValues, 1));
            return;
        }
    }
}
"""


def _glsl():
    return GLSLCodeGen().generate(parse(BUFFER_HELPERS))


def test_buffer_pointer_parameters_are_not_unsized_array_parameters():
    code = _glsl()
    for name in ("readAt", "readAlias", "writeAt", "readItem", "bumpNext"):
        assert f"{name}(float " not in code
        assert f"{name}(Item " not in code
    assert "values[]," not in code and "q[]," not in code and "p[]," not in code


def test_helpers_are_specialized_per_buffer_with_element_offsets():
    code = _glsl()
    assert (
        "float readAt_glsl_values_inputValues_float(int index, int values_offset) {\n"
        "    return inputValues[(values_offset + index)];" in code
    )
    assert "readAt_glsl_values_outputValues_float(" in code
    assert "writeAt_glsl_values_outputValues_float(" in code
    assert "writeAt_glsl_values_inputValues_float(" in code
    assert (
        "return readAt_glsl_values_inputValues_float(index, int(values_offset));" in (
            code
        )
    )
    assert "items[(q_offset + i)].a + items[(q_offset + i)].b.x" in code
    assert (
        "outputValues[(p_offset + (i + 1))] = (outputValues[(p_offset + i)] + 1.0);"
        in code
    )
    assert "bumpNext_glsl_p_outputValues_float(1, int(2));" in code


def test_output_is_deterministic():
    assert _glsl() == _glsl()


@pytest.mark.parametrize("target", ["vulkan1.1", "opengl"])
def test_generated_helpers_compile_with_glslang(tmp_path, target):
    compiler = shutil.which("glslangValidator")
    if compiler is None:
        pytest.skip("glslangValidator is not installed")
    source = tmp_path / "helpers.comp"
    source.write_text(_glsl(), encoding="utf-8")
    command = [compiler, "-S", "comp", str(source)]
    if target.startswith("vulkan"):
        command[1:1] = ["--target-env", target]
    # A Vulkan target writes comp.spv to the working directory.
    result = subprocess.run(
        command, capture_output=True, text=True, timeout=120, cwd=tmp_path
    )
    assert result.returncode == 0, result.stdout + result.stderr

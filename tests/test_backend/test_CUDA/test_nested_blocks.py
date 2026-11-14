"""CUDA, HIP and OpenCL C input keep nested ``{ ... }`` blocks and braced ``case`` bodies.

The parsers spliced a nested compound statement into the enclosing one, so a declaration in it redeclared an outer
name, or a name in another ``case``, and a use after the block read the inner variable. A nested block is now a
``BlockNode`` and imports as a CrossGL block (C++17 [basic.scope.block], C11 6.2.1).
"""

import shutil
import subprocess

import pytest

import crosstl
import crosstl.translator as cgl_translator
from crosstl.backend.common_ast import BlockNode
from crosstl.backend.CUDA.CudaLexer import CudaLexer
from crosstl.backend.CUDA.CudaParser import CudaParser

BODY = """
    float x = fin[i];
    float r = 0.0f;
    {
        float x = 2.0f;
        r += x;
    }
    switch (iin[i] & 1) {
        case 0: {
            float t = fin[i];
            r += t;
            break;
        }
        default: {
            int t = iin[i] * 4;
            r += (float)t;
            break;
        }
    }
    outp[i] = r * 100.0f + x;
"""

SOURCES = {
    "cu": (
        "__global__ void probe(const float* fin, const int* iin, float* outp) {\n"
        "    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;\n" + BODY + "}\n"
    ),
    "hip": (
        "#include <hip/hip_runtime.h>\n"
        "__global__ void probe(const float* fin, const int* iin, float* outp) {\n"
        "    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;\n" + BODY + "}\n"
    ),
    "cl": (
        "__kernel void probe(__global const float* fin, __global const int* iin, "
        "__global float* outp) {\n"
        "    uint i = get_global_id(0);\n" + BODY + "}\n"
    ),
}


def _crossgl(tmp_path, extension):
    source = tmp_path / f"probe.{extension}"
    source.write_text(SOURCES[extension])
    return crosstl.translate(str(source), backend="cgl")


def test_cuda_parser_keeps_nested_blocks():
    ast = CudaParser(CudaLexer(SOURCES["cu"]).tokenize()).parse()
    function = next(k for k in ast.kernels if k.name == "probe")
    blocks = [stmt for stmt in function.body if isinstance(stmt, BlockNode)]
    assert len(blocks) == 1 and len(blocks[0].statements) == 2


@pytest.mark.parametrize("extension", ["cu", "hip", "cl"])
def test_nested_blocks_import_as_crossgl_blocks(tmp_path, extension):
    code = _crossgl(tmp_path, extension)
    compact = " ".join(code.split())
    assert "BlockNode" not in code
    assert "{ var x: f32 = 2.0f; r += x; }" in compact
    assert "case 0: { var t: f32 = fin[i];" in compact
    assert "default: { var t: i32 = (iin[i] * 4);" in compact
    # The use after the block reads the outer x.
    assert compact.count("var x: f32") == 2
    cgl_translator.parse(code)


@pytest.mark.parametrize("extension", ["cu", "cl"])
def test_imported_blocks_reach_glsl(tmp_path, extension):
    crossgl = _crossgl(tmp_path, extension)
    path = tmp_path / "probe.cgl"
    path.write_text(crossgl)
    glsl = crosstl.translate(str(path), backend="opengl", format_output=False)
    if shutil.which("glslangValidator"):
        shader = tmp_path / "probe.comp"
        shader.write_text(glsl)
        result = subprocess.run(
            ["glslangValidator", str(shader)], capture_output=True, text=True
        )
        assert result.returncode == 0, result.stdout + result.stderr

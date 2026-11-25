"""An entry-scoped translation of a CUDA file skips the host code its kernel never reaches.

The CUDA, HIP and OpenCL C importers write a kernel as a module-level function with a stage attribute
(``@compute fn k(...)``) beside the file's host functions, whose calls to the CUDA runtime or the C library they
leave unresolved. The OpenGL generator selects such an entry in place and writes only the functions it reaches, so
the undeclared-name checks of an entry-scoped translation count these functions as entries too and skip the host
code. A whole-program translation still writes the host functions and so still refuses their undeclared calls.
"""

import shutil
import subprocess

import pytest

import crosstl
from crosstl.translator.lexer import Lexer
from crosstl.translator.parser import Parser
from crosstl.translator.semantics import UndeclaredFunctionError
from crosstl.translator.semantics.declared_types import (
    dead_function_ids,
    entry_dropped_function_ids,
)

HOST_CODE = """
#include <cstdlib>
__device__ float scale(float x) { return 2.0f * x; }
__global__ void k(float* out) { out[threadIdx.x] = scale(1.0f); }
void fill(float* a, int n) { for (int i = 0; i < n; i++) a[i] = rand() / (float)RAND_MAX; }
int main() { float* d; cudaMalloc(&d, 64); k<<<1, 64>>>(d); getLastCudaError("launch"); return 0; }
"""

REACHED_UNDECLARED = """
__device__ float scale(float x) { return nosuch(x); }
__global__ void k(float* out) { out[threadIdx.x] = scale(1.0f); }
"""

FUNCTION_FORM_PROGRAM = """
f32 scale(f32 x) {
    return (2.0f * x);
}

void fill(ptr<f32> a, i32 n) {
    for (var i: i32 = 0; (i < n); (i++)) {
        a[i] = (rand() / f32(RAND_MAX));
    }
}

@compute
@host_workgroup_size
fn k(
    @group(0) @binding(0) var<storage, read_write> out: array<f32>
) {
    out[gl_LocalInvocationID.x] = scale(1.0f);
}
"""

GLSL_TARGETS = ["opengl", "gles"]
WORKGROUP_SIZE = [
    "-DCROSSGL_WORKGROUP_SIZE_X=64",
    "-DCROSSGL_WORKGROUP_SIZE_Y=1",
    "-DCROSSGL_WORKGROUP_SIZE_Z=1",
]


def _cuda_file(tmp_path, source):
    path = tmp_path / "kernel.cu"
    path.write_text(source, encoding="utf-8")
    return path


def _program(source):
    return Parser(Lexer(source).get_tokens()).parse()


def _functions(program):
    return {function.name: function for function in program.functions}


def test_function_form_entries_drop_only_unreached_functions_for_a_selected_entry():
    program = _program(FUNCTION_FORM_PROGRAM)
    functions = _functions(program)

    dropped = entry_dropped_function_ids(program, "k")

    assert id(functions["fill"]) in dropped
    assert id(functions["scale"]) not in dropped
    assert id(functions["k"]) not in dropped
    # A whole-program translation writes every function, and the passes that model stage reachability keep
    # treating a program without stage blocks as a library.
    assert entry_dropped_function_ids(program) == set()
    assert dead_function_ids(program) == set()


@pytest.mark.parametrize("target", GLSL_TARGETS + ["directx", "metal"])
def test_entry_scoped_cuda_translation_skips_unreached_host_code(tmp_path, target):
    code = crosstl.translate(
        str(_cuda_file(tmp_path, HOST_CODE)), backend=target, entry_point="k"
    )

    assert "getLastCudaError" not in code
    assert "rand" not in code
    assert "cudaMalloc" not in code


@pytest.mark.parametrize("target", GLSL_TARGETS)
def test_entry_scoped_cuda_translation_to_glsl_compiles(tmp_path, target):
    code = crosstl.translate(
        str(_cuda_file(tmp_path, HOST_CODE)), backend=target, entry_point="k"
    )
    glslang = shutil.which("glslangValidator")
    if glslang is None:
        pytest.skip("glslangValidator is not installed")
    shader = tmp_path / "kernel.comp"
    shader.write_text(code, encoding="utf-8")
    result = subprocess.run(
        [glslang, "-l", *WORKGROUP_SIZE, str(shader)],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("target", GLSL_TARGETS)
def test_whole_program_cuda_translation_still_refuses_host_calls(tmp_path, target):
    with pytest.raises(UndeclaredFunctionError):
        crosstl.translate(str(_cuda_file(tmp_path, HOST_CODE)), backend=target)


@pytest.mark.parametrize("target", GLSL_TARGETS)
def test_entry_scoped_cuda_translation_refuses_a_reached_undeclared_call(
    tmp_path, target
):
    with pytest.raises(UndeclaredFunctionError):
        crosstl.translate(
            str(_cuda_file(tmp_path, REACHED_UNDECLARED)),
            backend=target,
            entry_point="k",
        )

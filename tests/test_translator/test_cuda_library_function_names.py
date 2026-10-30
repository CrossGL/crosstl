"""CUDA and HIP output rename a user function named like a device library function (issue #3326).

``cuda_runtime.h`` and ``hip/hip_runtime.h`` declare the device math library, so ``float rsqrt(float x)`` was a
redefinition for nvcc and ``float fabs(float x)`` one for hipcc. The generators also resolve a call by name alone,
so a user ``clamp(float)`` turned the built-in ``clamp(x, 0.0, 1.0)`` into a call of the user function with too many
arguments.
"""

import shutil
import subprocess

import pytest

import crosstl
from crosstl.translator.codegen.cuda_library_functions import shadows_cuda_library

SHADER = """
shader S {
    float lerp(float a, float b) { return a * 2.0 - b; }
    float rsqrt(float x) { return x + 0.25; }
    float fabs(float x) { return x * x; }
    float sinf(float x) { return x + 2.0; }
    float clamp(float x) { return x - 1.0; }
    float atomicInc(float x) { return x + 4.0; }
    float make_float2(float x) { return x * 5.0; }
    float transpose(float x) { return x * 6.0; }
    float helper(float x) { return x; }
    compute {
        layout(local_size_x = 1) in;
        void main() {
            float x = float(gl_GlobalInvocationID.x);
            float y = lerp(x, 1.0) + rsqrt(x) + fabs(x) + sinf(x) + sin(x) + clamp(x)
                + clamp(x, 0.0, 1.0) + atomicInc(x) + make_float2(x) + transpose(x) + helper(x);
        }
    }
}
"""

BACKENDS = ["cuda", "hip"]


def _translate(tmp_path, backend, source=SHADER):
    path = tmp_path / "names.cgl"
    path.write_text(source)
    return crosstl.translate(str(path), backend=backend, format_output=False)


def _flat(code):
    return " ".join(code.split())


def test_library_names_are_recognized():
    for name in ("rsqrt", "fabs", "sinf", "atomicInc", "make_float2", "__syncthreads"):
        assert shadows_cuda_library(name), name
    for name in ("lerp", "helper", "clamp", "saturate"):
        assert not shadows_cuda_library(name), name


@pytest.mark.parametrize("backend", BACKENDS)
def test_user_functions_named_like_library_functions_are_renamed(tmp_path, backend):
    code = _flat(_translate(tmp_path, backend))
    for name in ("rsqrt", "fabs", "sinf", "atomicInc", "make_float2", "transpose"):
        assert f"float {name}_fn(float x)" in code, name
        assert f"{name}_fn(x)" in code, name
        assert f"float {name}(float x)" not in code, name
    # Names the device library does not declare keep theirs, and the built-in sin still reaches sinf.
    assert "float lerp(float a, float b)" in code
    assert "lerp(x, 1.0f)" in code
    assert "float helper(float x)" in code
    assert "sinf(x)" in code


@pytest.mark.parametrize("backend", BACKENDS)
def test_a_call_only_a_builtin_accepts_reaches_the_builtin(tmp_path, backend):
    code = _flat(_translate(tmp_path, backend))
    assert "float clamp_fn(float x)" in code
    assert "clamp_fn(x)" in code
    assert "clamp(x, 0.0, 1.0)" not in code
    assert "fmaxf(0.0f, fminf(1.0f, x))" in code


@pytest.mark.parametrize("backend", BACKENDS)
def test_entry_points_and_other_targets_keep_their_names(tmp_path, backend):
    source = """
shader S {
    buffer float values[];
    float rsqrt(float x) { return x + 0.25; }
    compute {
        layout(local_size_x = 1) in;
        void main() {
            uint i = gl_GlobalInvocationID.x;
            values[i] = rsqrt(values[i]);
        }
    }
}
"""
    code = _flat(_translate(tmp_path, backend, source))
    assert "__global__ void compute_main(" in code
    assert "rsqrt_fn(values[i])" in code
    glsl = _flat(_translate(tmp_path, "opengl", source))
    assert "float rsqrt(float x)" in glsl


def _compile(tmp_path, tool, args, code, suffix):
    if shutil.which(tool) is None:
        pytest.skip(f"{tool} is not installed")
    source = tmp_path / f"names{suffix}"
    source.write_text(code)
    result = subprocess.run(
        [tool, *args, str(source), "-o", str(tmp_path / "names.o")],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_nvcc_compiles(tmp_path):
    code = _translate(tmp_path, "cuda")
    _compile(tmp_path, "nvcc", ["-std=c++17", "-c"], code, ".cu")


def test_hipcc_compiles(tmp_path):
    code = _translate(tmp_path, "hip")
    _compile(
        tmp_path, "hipcc", ["-c", "--offload-arch=gfx90a", "-x", "hip"], code, ".hip"
    )

"""CUDA and HIP declare the file-scope values device code reads where nvcc and hipcc let kernels and helpers name them."""

import shutil
import subprocess

import pytest

from crosstl.translator.codegen.cuda_codegen import CudaCodeGen
from crosstl.translator.codegen.hip_codegen import HipCodeGen
from crosstl.translator.lexer import Lexer
from crosstl.translator.parser import Parser
from crosstl.translator.refusals import CudaRefusal, HipRefusal

GENERATORS = {"cuda": (CudaCodeGen, CudaRefusal), "hip": (HipCodeGen, HipRefusal)}
BACKENDS = sorted(GENERATORS)

SOURCE = """
shader Globals {
    const int N = 3;
    const int M = N * 2 + 1;
    const float S = float(M) * 0.5;
    const vec2 V = vec2(1.0, 2.0) * S;
    const vec3 T = V.yxy;
    const float A[2] = {S, S * 2.0};
    const ivec2 Q = ivec2(7, -7) / 2;
    const uint U = 1u << 31u;
    const vec2 BIG = vec2(16777216.0) + 1.0;
    cbuffer Params @binding(3) {
        vec4 scale;
    };
    compute {
        layout(local_size_x = 2, local_size_y = 1, local_size_z = 1) in;
        buffer float* fin;
        buffer float* outp;
        device float acc[4] = {1.0, 2.0};
        shared float tile[2];
        shared float own[2];
        const vec2 STAGE = V + vec2(0.5);

        void stash(float x) {
            tile[0] = x;
            tile[1] = x * 0.5;
            acc[1] = acc[1] + x;
        }

        float exchange(float x) {
            stash(x);
            barrier();
            return tile[0] * 2.0 + tile[1];
        }

        void main() {
            uint i = gl_GlobalInvocationID.x;
            own[i & 1u] = fin[i];
            vec4 t = scale * fin[i];
            outp[i] = exchange(fin[i]) + own[0] + V.y + T.z + A[1] + float(Q.y)
                + float(U >> 31u) + BIG.x + STAGE.x + t.w + acc[1];
        }
    }
}
"""


def _generate(backend, source=SOURCE):
    generator, _ = GENERATORS[backend]
    return generator().generate(Parser(Lexer(source).tokens).parse())


def _lines(code):
    return [line.strip() for line in code.splitlines()]


@pytest.mark.parametrize("backend", BACKENDS)
def test_vector_and_array_constants_fold_to_static_brace_lists(backend):
    lines = _lines(_generate(backend))
    for declaration in (
        "__device__ const float2 V = {3.5f, 7.0f};",
        "__device__ const float3 T = {7.0f, 3.5f, 7.0f};",
        "__device__ const float A[2] = {3.5f, 7.0f};",
        # C++ integer division truncates toward zero.
        "__device__ const int2 Q = {3, -3};",
        # float arithmetic rounds each result to float: 2^24 + 1 is 2^24.
        "__device__ const float2 BIG = {16777216.0f, 16777216.0f};",
        "__device__ const float2 STAGE = {4.0f, 7.5f};",
    ):
        assert declaration in lines, declaration
    # Device code reads a const scalar with a constant initializer directly.
    assert "const float S = (float(M) * 0.5f);" in lines


@pytest.mark.parametrize("backend", BACKENDS)
def test_device_variable_is_in_device_memory(backend):
    assert "__device__ float acc[4] = {1.0f, 2.0f};" in _lines(_generate(backend))


@pytest.mark.parametrize("backend", BACKENDS)
def test_workgroup_variable_a_helper_uses_is_declared_at_file_scope(backend):
    code = _generate(backend)
    lines = _lines(code)
    assert lines.count("__shared__ float tile[2];") == 1
    # Declared before the helpers that name it, at file scope.
    assert code.index("__shared__ float tile[2];") < code.index("void stash(")
    assert "\n__shared__ float tile[2];" in code
    # A workgroup variable only the kernel names stays in the kernel.
    assert "\n__shared__ float own[2];" not in code
    assert lines.count("__shared__ float own[2];") == 1
    assert code.index("__shared__ float own[2];") > code.index("compute_main(")


def test_cuda_constant_buffer_vector_arithmetic_uses_the_vector_helper():
    # CUDA's float4 has no operator*; the member's type selects the helper.
    assert "float4 t = cgl_float4_mul_scalar(scale, fin[i]);" in _lines(
        _generate("cuda")
    )


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize(
    "declaration",
    [
        "const vec2 W = vec2(sin(1.0), 0.0);",
        "const float W[2] = {sin(1.0), 0.0};",
    ],
)
def test_vector_constant_without_a_static_initializer_is_refused(backend, declaration):
    source = f"""
shader Refused {{
    {declaration}
    compute {{
        buffer float* outp;
        void main() {{ outp[0] = W[0]; }}
    }}
}}
"""
    _, refusal = GENERATORS[backend]
    with pytest.raises(refusal, match="static initializer"):
        _generate(backend, source)


@pytest.mark.parametrize("backend", BACKENDS)
def test_device_variable_without_a_static_initializer_is_refused(backend):
    source = """
shader Refused {
    compute {
        buffer float* outp;
        device float g = sin(outp[0]);
        void main() { outp[0] = g; }
    }
}
"""
    _, refusal = GENERATORS[backend]
    with pytest.raises(refusal, match="static initializer"):
        _generate(backend, source)


@pytest.mark.parametrize(
    "declaration",
    [
        # Signed overflow and an over-wide shift are undefined in C++, so they do not fold.
        "const ivec2 W = ivec2(2147483647) + 1;",
        "const ivec2 W = ivec2(1) << 32;",
        "const vec2 W = vec2(1.0) / 0.0;",
    ],
)
def test_undefined_operations_do_not_fold(declaration):
    source = f"""
shader Refused {{
    {declaration}
    compute {{
        buffer float* outp;
        void main() {{ outp[0] = float(W.x); }}
    }}
}}
"""
    with pytest.raises(CudaRefusal, match="static initializer"):
        _generate("cuda", source)


def _compile(command, tmp_path, name, code):
    source = tmp_path / name
    source.write_text(code)
    result = subprocess.run(
        [*command, str(source), "-o", str(tmp_path / "globals.o")],
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_nvcc_compiles_the_file_scope_globals(tmp_path):
    nvcc = shutil.which("nvcc")
    if nvcc is None:
        pytest.skip("nvcc is not installed")
    _compile(
        [nvcc, "-arch=sm_80", "-std=c++17", "-c"],
        tmp_path,
        "globals.cu",
        _generate("cuda"),
    )


def test_hipcc_compiles_the_file_scope_globals(tmp_path):
    hipcc = shutil.which("hipcc")
    if hipcc is None:
        pytest.skip("hipcc is not installed")
    _compile(
        [hipcc, "--offload-arch=gfx1100", "-std=c++17", "-c"],
        tmp_path,
        "globals.hip",
        _generate("hip"),
    )

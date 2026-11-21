"""CUDA and HIP fold file-scope matrix constants, scalars that read device constants and FLT_MAX components.

Device code may read a const scalar only when its initializer is a constant expression, and a ``__device__``
variable only when it is statically initialized (CUDA C++ Programming Guide, "Initialization"; hipcc follows the
same rule). The matrix helpers' component constructor is constexpr, so a folded matrix constant is a static
``__device__`` initializer.
"""

import re
import shutil
import subprocess

import pytest

import crosstl
from crosstl.translator.codegen.cuda_codegen import CudaCodeGen
from crosstl.translator.codegen.hip_codegen import HipCodeGen
from crosstl.translator.lexer import Lexer
from crosstl.translator.parser import Parser
from crosstl.translator.refusals import CudaRefusal, HipRefusal

GENERATORS = {"cuda": (CudaCodeGen, CudaRefusal), "hip": (HipCodeGen, HipRefusal)}
BACKENDS = sorted(GENERATORS)

SOURCE = """
shader Constants {
    const vec3 BIG = vec3(3.402823466e+38);
    const float LOW[2] = {-3.402823466e+38, 0.0};
    const vec2 V = vec2(1.0, 2.0);
    const float Y = V.y;
    const int L = int(V.x) + 1;
    const mat2 M = mat2(1.0, 2.0, 3.0, 4.0);
    const mat2x3 R = mat2x3(1.0, 2.0, 3.0, 4.0, 5.0, 6.0);
    const dmat2 D = dmat2(2.0);
    const mat3 G = mat3(M);
    const mat2 P = mat2(V, V.yx);
    const vec3 C = R[1];
    const float E = M[1].y;
    compute {
        layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
        buffer float* outp;
        device vec3 TOP = vec3(3.402823466e+38);
        void main() {
            outp[0] = min(outp[0], BIG.x) + LOW[0] + Y + float(L)
                + (M * vec2(1.0, 1.0)).y + (R * vec2(1.0, 2.0)).z
                + float((D * dvec2(3.0, 1.0)).x) + (G * vec3(1.0, 1.0, 1.0)).z
                + (P * vec2(1.0, 1.0)).x + C.y + E + min(outp[1], TOP.z);
        }
    }
}
"""

HLSL_SOURCE = """
RWStructuredBuffer<float> outp : register(u0);
static const float2x2 M = float2x2(1.0, 2.0, 3.0, 4.0);

[numthreads(1, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    outp[0] = mul(M, float2(1.0, 1.0)).y;
}
"""


def _generate(backend, source=SOURCE):
    generator, _ = GENERATORS[backend]
    return generator().generate(Parser(Lexer(source).tokens).parse())


def _lines(code):
    return [" ".join(line.split()) for line in code.splitlines()]


@pytest.mark.parametrize("backend", BACKENDS)
def test_float_max_components_fold_without_overflow(backend):
    lines = _lines(_generate(backend))
    # 4 digits round FLT_MAX up to 3.403e+38, which is beyond the largest float; 8 digits read back exactly.
    big = "{3.4028235e+38f, 3.4028235e+38f, 3.4028235e+38f}"
    assert f"__device__ const float3 BIG = {big};" in lines
    assert "__device__ const float LOW[2] = {-3.4028235e+38f, 0.0f};" in lines
    assert f"__device__ float3 TOP = {big};" in lines


@pytest.mark.parametrize("backend", BACKENDS)
def test_scalar_constants_that_read_device_constants_fold(backend):
    lines = _lines(_generate(backend))
    assert "const float Y = 2.0f;" in lines
    assert "const int L = 2;" in lines
    # Column 1, row 1 of M, not the flat element 1.
    assert "const float E = 4.0f;" in lines


@pytest.mark.parametrize("backend", BACKENDS)
def test_matrix_constants_fold_to_static_device_initializers(backend):
    lines = _lines(_generate(backend))
    for declaration in (
        "__device__ const float2x2 M = float2x2(1.0f, 2.0f, 3.0f, 4.0f);",
        "__device__ const float2x3 R = float2x3(1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f);",
        "__device__ const double2x2 D = double2x2(2.0, 0.0, 0.0, 2.0);",
        # A smaller matrix is extended with the identity.
        "__device__ const float3x3 G = "
        "float3x3(1.0f, 2.0f, 0.0f, 3.0f, 4.0f, 0.0f, 0.0f, 0.0f, 1.0f);",
        "__device__ const float2x2 P = float2x2(1.0f, 2.0f, 2.0f, 1.0f);",
        "__device__ const float3 C = {4.0f, 5.0f, 6.0f};",
    ):
        assert declaration in lines, declaration


@pytest.mark.parametrize("backend", BACKENDS)
def test_matrix_component_constructor_is_constexpr(backend):
    code = " ".join(_generate(backend).split())
    assert (
        "__host__ __device__ constexpr float2x2(float v0, float v1, float v2, float v3) "
        ": m{v0, v1, v2, v3} { }"
    ) in code
    assert "constexpr double4x4(double v0," in code


@pytest.mark.parametrize("backend", BACKENDS)
def test_hlsl_static_const_matrix_is_a_device_constant(backend, tmp_path):
    source = tmp_path / "matrix.hlsl"
    source.write_text(HLSL_SOURCE)
    code = crosstl.translate(str(source), backend=backend)
    # HLSL float2x2(1, 2, 3, 4) lists rows; CrossGL stores columns.
    assert (
        "__device__ static const float2x2 M = float2x2(1.0f, 3.0f, 2.0f, 4.0f);"
        in _lines(code)
    )


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize(
    "declaration, pattern",
    [
        ("const float W = sin(1.0);", "constant initializer"),
        ("const float W = length(vec2(3.0, 4.0));", "constant initializer"),
        ("const mat2 W = mat2(sin(1.0));", "static initializer"),
        # Matrix arithmetic does not fold.
        ("const mat2 W = mat2(1.0) * 2.0;", "static initializer"),
    ],
)
def test_constants_without_a_constant_initializer_are_refused(
    backend, declaration, pattern
):
    source = f"""
shader Refused {{
    {declaration}
    compute {{
        buffer float* outp;
        void main() {{ outp[0] = float(W == W); }}
    }}
}}
"""
    _, refusal = GENERATORS[backend]
    with pytest.raises(refusal, match=re.escape(pattern)):
        _generate(backend, source)


@pytest.mark.parametrize("backend", BACKENDS)
def test_scalar_constant_expressions_keep_their_spelling(backend):
    source = """
shader Kept {
    const int N = 3;
    const float S = float(N) * 0.5;
    compute {
        buffer float* outp;
        void main() { outp[0] = S; }
    }
}
"""
    assert "const float S = (float(N) * 0.5f);" in _lines(_generate(backend, source))


def _compile(command, tmp_path, name, code):
    source = tmp_path / name
    source.write_text(code)
    result = subprocess.run(
        [*command, str(source), "-o", str(tmp_path / "constants.o")],
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def _hlsl(backend, tmp_path):
    source = tmp_path / "matrix.hlsl"
    source.write_text(HLSL_SOURCE)
    return crosstl.translate(str(source), backend=backend)


def test_nvcc_compiles_the_folded_constants(tmp_path):
    nvcc = shutil.which("nvcc")
    if nvcc is None:
        pytest.skip("nvcc is not installed")
    command = [nvcc, "-arch=sm_80", "-std=c++17", "-c"]
    _compile(command, tmp_path, "constants.cu", _generate("cuda"))
    _compile(command, tmp_path, "matrix.cu", _hlsl("cuda", tmp_path))


def test_hipcc_compiles_the_folded_constants(tmp_path):
    hipcc = shutil.which("hipcc")
    if hipcc is None:
        pytest.skip("hipcc is not installed")
    command = [hipcc, "--offload-arch=gfx1100", "-std=c++17", "-c"]
    _compile(command, tmp_path, "constants.hip", _generate("hip"))
    _compile(command, tmp_path, "matrix.hip", _hlsl("hip", tmp_path))

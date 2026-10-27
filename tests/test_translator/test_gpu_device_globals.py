"""CUDA and HIP place file-scope resources in device memory so that kernels and device functions can read them."""

import shutil
import subprocess

import pytest

from crosstl.translator.codegen.cuda_codegen import CudaCodeGen
from crosstl.translator.codegen.gpu_device_globals import (
    braced_vector_constructors,
    in_device_memory,
)
from crosstl.translator.codegen.hip_codegen import HipCodeGen
from crosstl.translator.lexer import Lexer
from crosstl.translator.parser import Parser

SOURCE = """
shader Resources {
    const float WEIGHTS[4] = {0.125, 0.25, 0.5, 1.0};
    const vec2 OFFSET = vec2(1.0, 2.0);
    const int COUNT = 3;
    compute {
        layout(local_size_x = 64) in;
        buffer float* values;
        buffer float outv[];
        RWStructuredBuffer<uint> counts;
        StructuredBuffer<float> weights;
        image2D target;
        uniform float scale;
        const float BIAS = 0.5;
        const vec3 TINT = vec3(1.0);
        shared float tile[64];

        float load(uint i) {
            float local = values[i] * scale;
            return local;
        }

        void main() {
            const float k = 3.0;
            uint i = gl_GlobalInvocationID.x;
            tile[gl_LocalInvocationID.x] = load(i);
            barrier();
            uint n;
            uint stride;
            weights.GetDimensions(n, stride);
            ivec2 size = imageSize(target);
            outv[i] = tile[gl_LocalInvocationID.x] * weights[i] + BIAS + k + float(n + uint(size.x));
            outv[i] += WEIGHTS[i & 3u] + OFFSET.y + TINT.z + float(COUNT);
            counts[i] = n;
        }
    }
}
"""

GENERATORS = {"cuda": CudaCodeGen, "hip": HipCodeGen}


def _generate(backend):
    ast = Parser(Lexer(SOURCE).tokens).parse()
    return GENERATORS[backend]().generate(ast)


def _declarations(code):
    return {line.strip() for line in code.splitlines() if line.strip().endswith(";")}


@pytest.mark.parametrize("backend", ["cuda", "hip"])
def test_storage_buffers_are_device_globals(backend):
    declarations = _declarations(_generate(backend))
    assert "__device__ float* values;" in declarations
    assert "__device__ float* outv;" in declarations
    # The element count GetDimensions reads is bound with the buffer.
    assert "__device__ const uint* weights_length;" in declarations
    assert any(
        line.startswith("__device__ ") and line.endswith("* counts;")
        for line in declarations
    )
    assert "float* values;" not in declarations


@pytest.mark.parametrize("backend", ["cuda", "hip"])
def test_images_and_their_query_metadata_are_device_globals(backend):
    declarations = _declarations(_generate(backend))
    surface = {"cuda": "cudaSurfaceObject_t", "hip": "hipSurfaceObject_t"}[backend]
    assert f"__device__ {surface} target;" in declarations
    assert "__device__ CglResourceQueryInfo target_metadata = {};" in declarations


@pytest.mark.parametrize("backend", ["cuda", "hip"])
def test_other_execution_spaces_and_locals_are_unchanged(backend):
    code = _generate(backend)
    declarations = _declarations(code)
    assert "__constant__ float scale;" in declarations
    assert "__shared__ float tile[64];" in declarations
    assert "float local = (values[i] * scale);" in declarations
    assert "__device__ __constant__" not in code
    assert "__device__ __shared__" not in code
    assert "__device__ float local" not in code


@pytest.mark.parametrize("backend", ["cuda", "hip"])
def test_stage_constants_stay_const(backend):
    declarations = _declarations(_generate(backend))
    # Device code may read a const scalar with a constant initializer, never a mutable host variable.
    assert "const float BIAS = 0.5f;" in declarations
    assert "const float k = 3.0f;" in declarations


@pytest.mark.parametrize("backend", ["cuda", "hip"])
def test_constant_arrays_and_vectors_are_static_device_globals(backend):
    declarations = _declarations(_generate(backend))
    assert (
        "__device__ const float WEIGHTS[4] = {0.125f, 0.25f, 0.5f, 1.0f};" in declarations
    )
    # make_float2 is not constexpr: a device variable needs the brace list to be initialized statically.
    assert "__device__ const float2 OFFSET = {1.0f, 2.0f};" in declarations
    assert "__device__ const float3 TINT = {1.0f, 1.0f, 1.0f};" in declarations
    assert "const int COUNT = 3;" in declarations


def test_only_whole_vector_constructors_become_brace_lists():
    assert braced_vector_constructors("make_float2(1.0, 2.0)") == "{1.0, 2.0}"
    assert (
        braced_vector_constructors("{make_int2(1, (2)), make_int2(3, 4)}")
        == "{{1, (2)}, {3, 4}}"
    )
    for unchanged in (
        "f(make_float2(1.0, 2.0).x)",
        "make_float2(1.0, 2.0).x",
        "make_float2(make_float2(1.0, 2.0).x, 3.0)",
        "make_half2(1.0, 2.0)",
        "1.0",
    ):
        assert braced_vector_constructors(unchanged) == unchanged


def test_declarations_that_name_an_execution_space_keep_it():
    assert in_device_memory("float* values") == "__device__ float* values"
    assert in_device_memory("__constant__ float scale") == "__constant__ float scale"
    assert in_device_memory("__shared__ float tile[4]") == "__shared__ float tile[4]"
    assert in_device_memory("__device__ int x") == "__device__ int x"


def _compile(command, tmp_path, name, code):
    source = tmp_path / name
    source.write_text(code)
    result = subprocess.run(
        [*command, str(source), "-o", str(tmp_path / "resources.o")],
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_nvcc_accepts_device_resources(tmp_path):
    nvcc = shutil.which("nvcc")
    if nvcc is None:
        pytest.skip("nvcc is not installed")
    _compile(
        [nvcc, "-arch=sm_80", "-std=c++17", "-c"],
        tmp_path,
        "resources.cu",
        _generate("cuda"),
    )


def test_hipcc_accepts_device_resources(tmp_path):
    hipcc = shutil.which("hipcc")
    if hipcc is None:
        pytest.skip("hipcc is not installed")
    _compile(
        [hipcc, "--offload-arch=gfx1100", "-std=c++17", "-c"],
        tmp_path,
        "resources.hip",
        _generate("hip"),
    )

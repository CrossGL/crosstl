"""CUDA and HIP output read a compute built-in's component with a runtime subscript.

``gl_GlobalInvocationID[d]`` with a runtime ``d`` translates for GLSL, HLSL, Metal and SPIR-V, but CUDA and HIP
output refused it, because their ``uint3`` and ``dim3`` have no subscript operator. Both generators now call a
helper with the index and the three component expressions, so the index is evaluated once. The helper is checked
here by compiling it as host C++ and running it, and nvcc and hipcc compile the whole translation when they are
installed.
"""

from __future__ import annotations

import re
import shutil
import subprocess

import pytest

import crosstl
from crosstl.translator.codegen.cuda_refusals import CudaLoweringRefused
from crosstl.translator.codegen.hip_refusals import HipLoweringRefused

RUNTIME_INDEX = """
shader Probe {
    compute {
        layout(local_size_x = 4, local_size_y = 2, local_size_z = 2) in;
        buffer uint* iin;
        buffer uint* outp;

        uint pick(uint k) {
            outp[1] = outp[1] + 1u;
            return k;
        }

        void main() {
            uint i = gl_GlobalInvocationID.x;
            uint d = iin[i];
            outp[i] = gl_GlobalInvocationID[d] + gl_LocalInvocationID[d] * 16u
                + gl_WorkGroupID[d] * 256u + gl_NumWorkGroups[pick(d)] * 4096u;
        }
    }
}
"""

REFUSED = {"cuda": CudaLoweringRefused, "hip": HipLoweringRefused}


def _translate(tmp_path, source, backend):
    path = tmp_path / "probe.cgl"
    path.write_text(source, encoding="utf-8")
    return crosstl.translate(str(path), backend=backend, format_output=False)


@pytest.mark.parametrize("backend", ["cuda", "hip"])
def test_a_runtime_subscript_reads_each_component_through_the_helper(tmp_path, backend):
    code = _translate(tmp_path, RUNTIME_INDEX, backend)

    assert (
        "cgl_builtin_component(d, (blockIdx.x * blockDim.x + threadIdx.x), "
        "(blockIdx.y * blockDim.y + threadIdx.y), (blockIdx.z * blockDim.z + threadIdx.z))"
    ) in code
    assert "cgl_builtin_component(d, threadIdx.x, threadIdx.y, threadIdx.z)" in code
    assert "cgl_builtin_component(d, blockIdx.x, blockIdx.y, blockIdx.z)" in code
    assert code.count("__device__ inline unsigned int cgl_builtin_component(") == 1


@pytest.mark.parametrize("backend", ["cuda", "hip"])
def test_the_index_expression_is_evaluated_once(tmp_path, backend):
    code = _translate(tmp_path, RUNTIME_INDEX, backend)

    calls = re.findall(
        r"cgl_builtin_component\(pick\(d\), gridDim\.x, gridDim\.y, gridDim\.z\)", code
    )
    assert len(calls) == 1
    assert code.count("pick(d)") == 1


@pytest.mark.parametrize("backend", ["cuda", "hip"])
def test_a_constant_subscript_still_names_its_component(tmp_path, backend):
    source = RUNTIME_INDEX.replace("gl_LocalInvocationID[d]", "gl_LocalInvocationID[1]")
    code = _translate(tmp_path, source, backend)

    assert "threadIdx.y * 16u" in code
    assert "cgl_builtin_component(d, threadIdx.x" not in code


@pytest.mark.parametrize("backend", ["cuda", "hip"])
def test_a_constant_subscript_outside_the_vector_is_refused(tmp_path, backend):
    source = RUNTIME_INDEX.replace("gl_LocalInvocationID[d]", "gl_LocalInvocationID[3]")

    with pytest.raises(REFUSED[backend], match="with index 3"):
        _translate(tmp_path, source, backend)


@pytest.mark.parametrize("backend", ["cuda", "hip"])
def test_a_subscript_of_the_scalar_local_index_is_refused(tmp_path, backend):
    source = RUNTIME_INDEX.replace(
        "gl_LocalInvocationID[d]", "gl_LocalInvocationIndex[d]"
    )

    with pytest.raises(REFUSED[backend], match="is a scalar"):
        _translate(tmp_path, source, backend)


def test_the_helper_returns_the_indexed_component(tmp_path):
    compiler = shutil.which("c++") or shutil.which("clang++") or shutil.which("g++")
    if compiler is None:
        pytest.skip("no host C++ compiler is installed")
    code = _translate(tmp_path, RUNTIME_INDEX, "cuda")
    helper = re.search(
        r"__device__ inline unsigned int cgl_builtin_component\(.*?\n\}", code, re.S
    ).group(0)
    program = tmp_path / "helper.cpp"
    program.write_text(
        "#include <cstdio>\n#define __device__\n" + helper + "\nint main() {\n"
        "  for (unsigned int i = 0; i < 3; ++i) {\n"
        '    std::printf("%u ", cgl_builtin_component(i, 10u, 20u, 30u));\n'
        "  }\n"
        "  return 0;\n"
        "}\n",
        encoding="utf-8",
    )
    binary = tmp_path / "helper"
    subprocess.run([compiler, str(program), "-o", str(binary)], check=True, timeout=120)
    result = subprocess.run(
        [str(binary)], capture_output=True, text=True, check=True, timeout=60
    )

    assert result.stdout.split() == ["10", "20", "30"]


@pytest.mark.parametrize(
    "backend, tool, flags",
    [
        ("cuda", "nvcc", ["-std=c++17", "-c"]),
        ("hip", "hipcc", ["--offload-arch=gfx1100", "-std=c++17", "-c"]),
    ],
)
def test_the_reference_compiler_accepts_the_translation(tmp_path, backend, tool, flags):
    compiler = shutil.which(tool)
    if compiler is None:
        pytest.skip(f"{tool} is not installed")
    source = tmp_path / ("probe.cu" if backend == "cuda" else "probe.hip")
    source.write_text(_translate(tmp_path, RUNTIME_INDEX, backend), encoding="utf-8")
    result = subprocess.run(
        [compiler, *flags, str(source), "-o", str(tmp_path / "probe.o")],
        capture_output=True,
        text=True,
        timeout=900,
        check=False,
    )
    assert result.returncode == 0, result.stderr[-2000:]

"""CUDA, HIP and OpenCL C no-contraction scopes import as CrossGL ``@precise`` functions.

See crosstl/backend/no_contraction.py. Execution evidence: the conformance cells u_fmul_rn and c_fp_contract_off.
"""

import re

import pytest

import crosstl
from crosstl.backend.no_contraction import (
    FpContractState,
    fp_contract_pragma_state,
    function_requires_no_contraction,
)


def import_crossgl(tmp_path, name, source):
    path = tmp_path / name
    path.write_text(source)
    return crosstl.translate(str(path), backend="cgl", format_output=False)


def precise_functions(crossgl):
    """Names of the functions that follow an ``@precise`` attribute line."""
    names = []
    lines = crossgl.splitlines()
    for index, line in enumerate(lines):
        if line.strip() != "@precise":
            continue
        for following in lines[index + 1 :]:
            match = re.search(r"\b(?:fn\s+)?(\w+)\s*\(", following)
            if match and not following.strip().startswith("@"):
                names.append(match.group(1))
                break
    return names


CUDA = """__device__ float product(float a, float b) { return __fmul_rn(a, b); }
__device__ float plain(float a, float b) { return a * b; }
__global__ void unfused(const float* a, float* o) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    o[i] = __fadd_rn(__fmul_rn(a[i], a[i]), -1.0f);
}
__global__ void contracted(const float* a, float* o) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    o[i] = a[i] * a[i] - 1.0f;
}
"""


def test_cuda_round_to_nearest_intrinsics_mark_their_functions_precise(tmp_path):
    crossgl = import_crossgl(tmp_path, "k.cu", CUDA)
    assert sorted(precise_functions(crossgl)) == ["product", "unfused"]


HIP = """#include <hip/hip_runtime.h>
__global__ void unfused(const double* a, double* o) {
    int i = threadIdx.x;
    o[i] = __dadd_rn(__dmul_rn(a[i], a[i]), 1.0);
}
__global__ void contracted(const float* a, float* o) {
    int i = threadIdx.x;
    o[i] = a[i] * a[i] + 1.0f;
}
"""


def test_hip_round_to_nearest_intrinsics_mark_their_kernels_precise(tmp_path):
    crossgl = import_crossgl(tmp_path, "k.hip", HIP)
    assert precise_functions(crossgl) == ["unfused"]


OPENCL_FILE_SCOPE = """#pragma OPENCL FP_CONTRACT OFF
float helper(float a) { return a * a - 1.0f; }
__kernel void unfused(__global const float* a, __global float* o) {
    int i = get_global_id(0);
    o[i] = helper(a[i]) + a[i] * a[i];
}
#pragma OPENCL FP_CONTRACT ON
__kernel void contracted(__global const float* a, __global float* o) {
    int i = get_global_id(0);
    o[i] = a[i] * a[i] - 1.0f;
}
"""


def test_opencl_file_scope_fp_contract_off_marks_functions_until_it_is_turned_on(
    tmp_path,
):
    crossgl = import_crossgl(tmp_path, "k.cl", OPENCL_FILE_SCOPE)
    assert sorted(precise_functions(crossgl)) == ["helper", "unfused"]


OPENCL_BLOCK_SCOPE = """float unfused(float a) {
#pragma OPENCL FP_CONTRACT OFF
    return a * a - 1.0f;
}
float contracted(float a) { return a * a - 1.0f; }
__kernel void k(__global const float* a, __global float* o) {
    int i = get_global_id(0);
    o[i] = unfused(a[i]) + contracted(a[i]);
}
"""


def test_opencl_fp_contract_off_inside_a_function_marks_only_that_function(tmp_path):
    crossgl = import_crossgl(tmp_path, "k.cl", OPENCL_BLOCK_SCOPE)
    assert precise_functions(crossgl) == ["unfused"]


class _Pragma:
    directive = "pragma"

    def __init__(self, content):
        self.content = content


@pytest.mark.parametrize(
    ("content", "state"),
    [
        ("OPENCL FP_CONTRACT OFF", "OFF"),
        ("STDC FP_CONTRACT ON", "ON"),
        ("OPENCL\tFP_CONTRACT DEFAULT", "DEFAULT"),
        ("OPENCL EXTENSION cl_khr_fp64 : enable", None),
        ("OPENCL FP_CONTRACT", None),
        ("OPENCL FP_CONTRACT FAST", None),
    ],
)
def test_fp_contract_pragma_state(content, state):
    assert fp_contract_pragma_state(_Pragma(content)) == state


def test_file_scope_state_follows_each_fp_contract_pragma():
    state = FpContractState()
    assert not state.update(object())
    assert state.update(_Pragma("OPENCL FP_CONTRACT OFF")) and state.off
    assert state.update(_Pragma("OPENCL FP_CONTRACT DEFAULT")) and not state.off


def test_a_function_without_a_body_needs_no_contraction_scope():
    class Declaration:
        body = None

    assert not function_requires_no_contraction(Declaration())
    assert function_requires_no_contraction(Declaration(), file_scope_off=True)

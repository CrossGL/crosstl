"""SPIR-V output of CUDA helper functions that take generic pointers.

CUDA input spells every device pointer ``ptr<T>``, since a CUDA pointer may address any memory. SPIR-V output now
gives each one the address space of the argument it receives (``generic_pointer_spaces``), as the HLSL, Metal and
Slang generators already did, so a helper taking a buffer is bound to that buffer, an offset pointer keeps its
offset, and a pointer into a ``__shared__`` array reads workgroup memory. It used to be refused as "Unknown type
ptr<f32>".
"""

from __future__ import annotations

import re
import shutil
import subprocess

from tests.spirv_friendly_names import translate_spirv

HELPERS = """
__device__ float get(const float* a, int i) { return a[i] * 2.0f; }
__device__ void put(int* p, int i, int v) { p[i] = v; }
__device__ float sum(const float* s, int n) {
    float t = 0.0f;
    for (int k = 0; k < n; ++k) t += s[k];
    return t;
}
__global__ void k(float* o, const float* a, int* c) {
    __shared__ float tile[64];
    tile[threadIdx.x] = a[threadIdx.x];
    __syncthreads();
    o[threadIdx.x] = get(a + 4, threadIdx.x) + sum(tile, 8);
    put(c + 1, threadIdx.x, 3);
}
"""


def _validate(tmp_path, code):
    assembler, validator = shutil.which("spirv-as"), shutil.which("spirv-val")
    if assembler is None or validator is None:
        return
    asm = tmp_path / "module.spvasm"
    asm.write_text(code, encoding="utf-8")
    binary = tmp_path / "module.spv"
    subprocess.run(
        [assembler, "--target-env", "vulkan1.1", str(asm), "-o", str(binary)],
        capture_output=True,
        check=True,
    )
    result = subprocess.run(
        [validator, "--target-env", "vulkan1.1", str(binary)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr[-2000:]


def test_helpers_taking_generic_pointers_read_the_memory_they_are_given(tmp_path):
    path = tmp_path / "helpers.cu"
    path.write_text(HELPERS, encoding="utf-8")
    code = translate_spirv(path)
    assert "ptr<" not in code
    # put(c + 1, i, 3) stores 3 into c[(0 + 1) + i].
    store = re.search(
        r"%(\w+) = OpIAdd %int %\w+ %\w+\s+%(\w+) = OpAccessChain %_ptr_\w+_int %c %int_0 %\1\s+"
        r"OpStore %\2 %int_3\b",
        code,
    )
    assert store, code
    # get(a + 4, i) reads a[(0 + 4) + i] and doubles it.
    load = re.search(
        r"%(\w+) = OpIAdd %int %\w+ %\w+\s+%(\w+) = OpAccessChain %_ptr_\w+_float %a %int_0 %\1\s+"
        r"%(\w+) = OpLoad %float %\2\s+%\w+ = OpFMul %float %\3 %float_2\b",
        code,
    )
    assert load, code
    # sum(tile, 8) reads the workgroup array.
    assert re.search(r"OpAccessChain %_ptr_Workgroup_float %tile ", code)
    _validate(tmp_path, code)

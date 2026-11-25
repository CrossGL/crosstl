"""GLSL output of 64-bit integer buffer and shared-memory atomics.

GLSL 4.60 section 8.11 declares the memory atomic functions for 32-bit ``int`` and ``uint`` memory only.
GL_NV_shader_atomic_int64 adds their ``int64_t`` and ``uint64_t`` overloads, which glslang declares from GLSL 4.40.
Their value operands have the memory type, like the 32-bit ones.
"""

from __future__ import annotations

import shutil
import subprocess

import pytest

import crosstl

GLSLANG = shutil.which("glslangValidator")
EXTENSION_LINE = "#extension GL_NV_shader_atomic_int64 : require"

BUFFER_AND_SHARED_CGL = """shader Atomics64 {
    compute {
        layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;
        groupshared uint64 total;
        groupshared int64 lows[4];
        void main(RWStructuredBuffer<uint64> outp @buffer(0), RWStructuredBuffer<int64> sout @buffer(1),
                  uvec3 id @gl_GlobalInvocationID) {
            uint i = id.x;
            int64 s = int64(i) - 3;
            atomicAdd(total, uint64(i));
            atomicMin(lows[i % 4u], s);
            uint64 seen = atomicLoad(outp[0]);
            atomicStore(outp[1], seen);
            int64 low = atomicLoad(sout[0]);
            atomicMax(outp[2], uint64(s));
            atomicXor(outp[3], i);
            atomicOr(sout[1], 5);
            atomicCompSwap(outp[4], seen, uint64(i));
            sout[2] = low;
        }
    }
}
"""

THIRTY_TWO_BIT_CGL = """shader Atomics32 {
    compute {
        layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;
        void main(RWStructuredBuffer<uint> outp @buffer(0), uvec3 id @gl_GlobalInvocationID) {
            atomicMax(outp[0], id.x);
        }
    }
}
"""

METAL_KERNEL = """#include <metal_stdlib>
using namespace metal;

kernel void probe(device uint* uin [[buffer(0)]], device atomic_ulong* outp [[buffer(1)]],
                  uint3 id [[thread_position_in_grid]]) {
    uint i = id.x;
    ulong value = ulong(uin[i]);
    atomic_{operation}_explicit(&outp[i], value, memory_order_relaxed);
}
"""

GLSL_430_BLOCK = """#version 430 core
#extension GL_ARB_gpu_shader_int64 : require
#extension GL_NV_shader_atomic_int64 : require
layout(std430, binding = 0) buffer B { uint64_t data[]; } b;
layout(local_size_x = 64) in;
void main() {
    atomicMax(b.data[gl_GlobalInvocationID.x], uint64_t(gl_GlobalInvocationID.x));
}
"""

CUDA_KERNEL = """__global__ void k(unsigned long long* out, const unsigned int* in) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    atomicMax(&out[0], (unsigned long long)in[i]);
    atomicAdd(&out[1], 1ull);
}
"""

METAL_STORE_MEMBER = """#include <metal_stdlib>
using namespace metal;
struct Counter { ulong before; atomic<ulong> value; ulong after; };

kernel void store_values(const device ulong* values [[buffer(0)]],
                         device Counter* results [[buffer(1)]],
                         device uint* counts [[buffer(2)]],
                         uint tid [[thread_position_in_grid]]) {
    uint position = tid + 2;
    uint value_index = tid;
    atomic_store_explicit(&results[position++].value, values[value_index++], memory_order_relaxed);
    counts[tid * 2] = position;
    counts[tid * 2 + 1] = value_index;
}
"""


def _translate(tmp_path, name, text):
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return crosstl.translate(str(path), backend="opengl", format_output=False)


def _assert_glslang_accepts(tmp_path, text):
    if GLSLANG is None:
        return
    path = tmp_path / "out.comp"
    path.write_text(text, encoding="utf-8")
    result = subprocess.run(
        [GLSLANG, str(path)], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_buffer_and_shared_64_bit_atomics_require_the_extension(tmp_path):
    translated = _translate(tmp_path, "atomics64.cgl", BUFFER_AND_SHARED_CGL)

    assert translated.count(EXTENSION_LINE) == 1
    assert "#extension GL_ARB_gpu_shader_int64 : require" in translated
    assert "atomicAdd(total, uint64_t(i));" in translated
    assert "atomicMin(lows[(i % 4u)], s);" in translated
    # atomicLoad is an atomicOr with a zero of the memory type.
    assert "uint64_t seen = atomicOr(outp[0], 0ul);" in translated
    assert "int64_t low = atomicOr(sout[0], 0l);" in translated
    assert "atomicExchange(outp[1], seen);" in translated
    # 32-bit value operands are widened to the 64-bit memory type.
    assert "atomicXor(outp[3], uint64_t(i));" in translated
    assert "atomicOr(sout[1], int64_t(5));" in translated
    assert "atomicCompSwap(outp[4], seen, uint64_t(i));" in translated
    _assert_glslang_accepts(tmp_path, translated)


def test_32_bit_atomics_need_no_64_bit_extension(tmp_path):
    translated = _translate(tmp_path, "atomics32.cgl", THIRTY_TWO_BIT_CGL)

    assert "GL_NV_shader_atomic_int64" not in translated
    assert "atomicMax(outp[0], gl_GlobalInvocationID.x);" in translated
    _assert_glslang_accepts(tmp_path, translated)


@pytest.mark.parametrize("operation", ["max", "min", "fetch_add"])
def test_metal_atomic_ulong_operations_compile_as_glsl(tmp_path, operation):
    translated = _translate(
        tmp_path, "probe.metal", METAL_KERNEL.replace("{operation}", operation)
    )

    assert EXTENSION_LINE in translated
    _assert_glslang_accepts(tmp_path, translated)


def test_glsl_source_version_rises_to_the_64_bit_atomic_floor(tmp_path):
    translated = _translate(tmp_path, "block.comp", GLSL_430_BLOCK)

    assert translated.startswith("#version 440 core\n")
    assert translated.count(EXTENSION_LINE) == 1
    assert "uint64_t(gl_GlobalInvocationID.x));" in translated
    _assert_glslang_accepts(tmp_path, translated)


def test_cuda_unsigned_long_long_atomics_compile_as_glsl(tmp_path):
    translated = _translate(tmp_path, "kernel.cu", CUDA_KERNEL)

    assert EXTENSION_LINE in translated
    assert "uint64_t(in_[i])" in translated
    _assert_glslang_accepts(tmp_path, translated)


def test_metal_atomic_ulong_member_store_is_one_atomic_exchange(tmp_path):
    translated = _translate(tmp_path, "store.metal", METAL_STORE_MEMBER)

    assert EXTENSION_LINE in translated
    # The store's index and value operands keep their side effects once each.
    assert (
        "atomicExchange(results[(position++)].value, values[(value_index++)]);"
        in translated
    )
    _assert_glslang_accepts(tmp_path, translated)

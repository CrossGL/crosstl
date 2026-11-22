"""CUDA and HIP input of system-scoped atomics, ``__syncwarp(mask)``, the device clocks and initialized ``__device__``
variables.

Each of these used to be refused at import. They now import as follows, and these tests pin both directions:

* a system-scoped atomic (``atomicAdd_system`` ...) is also atomic for the host and peer devices, which the
  device-scoped CrossGL atomics are not, so it keeps its CUDA name in CrossGL. CUDA and HIP output write it back (HIP
  declares no ``atomicInc_system``/``atomicDec_system``, so HIP output performs them with an ``atomicCAS_system``
  loop); every other generator refuses it with a typed diagnostic;
* CUDA ``__syncwarp(mask)`` stays a CUDA call too, which CUDA output writes back and
  ``crosstl.translator.semantics.warp`` refuses for the targets without CUDA warps unless the mask names the whole
  warp;
* CUDA ``clock64()`` and HIP ``clock()``/``clock64()`` read the cycle counter, CrossGL ``clockARB()``; CUDA ``clock()``
  reads its low 32 bits (PTX ``%clock``) as an int widened to ``clock_t``; HIP ``wall_clock64()`` reads the
  constant-rate counter, CrossGL ``clockRealtimeEXT()``;
* an initialized ``__device__`` variable, which the module load fills, becomes ``var<device> name: T = value;``.
"""

from __future__ import annotations

import re

import pytest

import crosstl
from crosstl.backend.import_refusals import CudaImportRefused, HipImportRefused
from crosstl.translator.diagnostics import TranslationDiagnostic
from crosstl.translator.lexer import Lexer as CrossGLLexer
from crosstl.translator.parser import Parser as CrossGLParser

OTHER_TARGETS = ("metal", "directx", "opengl", "vulkan", "wgsl", "slang")

ATOMIC_KERNEL = """
extern "C" __global__ void probe(unsigned int* uin, unsigned int* outp) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int old = OPERATION(&outp[i], uin[i]);
    outp[i + 1u] = old;
}
"""

CAS_KERNEL = """
extern "C" __global__ void probe(int* iin, int* outp) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    outp[i + 1u] = atomicCAS_system(&outp[i], iin[i], iin[i + 1u]);
}
"""

SYNCWARP_KERNEL = """
extern "C" __global__ void probe(unsigned int* uin, unsigned int* outp) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int mask = uin[i];
    __syncwarp(mask);
    outp[i] = mask;
}
"""

CUDA_CLOCK_KERNEL = """
extern "C" __global__ void probe(long long* outp, clock_t* ticks) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    long long start = clock64();
    clock_t low = clock();
    outp[i] = clock64() - start;
    ticks[i] = low;
}
"""

HIP_CLOCK_KERNEL = """
extern "C" __global__ void probe(long long* outp) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    long long cycles = clock64();
    long long other = clock();
    long long wall = wall_clock64();
    outp[i] = cycles + other + wall;
}
"""

DEVICE_GLOBAL_KERNEL = """
__device__ float acc[64] = {1.0f, 2.0f, 4.0f};
__device__ int counter = 7;
extern "C" __global__ void probe(float* fin, float* outp) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    acc[i & 63] = acc[i & 63] * 2.0f + fin[i];
    outp[i] = acc[i & 63] + (float)counter;
}
"""


def _translate(tmp_path, source, suffix, backend):
    path = tmp_path / f"kernel{suffix}"
    path.write_text(source, encoding="utf-8")
    return crosstl.translate(str(path), backend=backend, format_output=False)


def _flat(code):
    return re.sub(r"\s+", " ", code)


def _crossgl(tmp_path, source, suffix):
    crossgl = _translate(tmp_path, source, suffix, "cgl")
    CrossGLParser(CrossGLLexer(crossgl).tokens).parse()
    return crossgl


def _assert_typed_refusal(error):
    """A refusal is typed: a diagnostic class, or an error that names its project diagnostic code."""
    assert isinstance(error, TranslationDiagnostic) or getattr(
        error, "project_diagnostic_code", None
    ), repr(error)


def _assert_refused_by_other_targets(tmp_path, crossgl, construct):
    for backend in OTHER_TARGETS:
        with pytest.raises(Exception) as raised:
            _translate(tmp_path, crossgl, ".cgl", backend)
        _assert_typed_refusal(raised.value)
        assert construct in str(raised.value), (backend, str(raised.value))


CUDA_SYSTEM_ATOMICS = [
    "atomicAdd_system",
    "atomicSub_system",
    "atomicExch_system",
    "atomicMin_system",
    "atomicMax_system",
    "atomicInc_system",
    "atomicDec_system",
    "atomicAnd_system",
    "atomicOr_system",
    "atomicXor_system",
]


@pytest.mark.parametrize("suffix", [".cu", ".hip"])
@pytest.mark.parametrize("operation", CUDA_SYSTEM_ATOMICS)
def test_system_atomic_keeps_its_scope(tmp_path, suffix, operation):
    crossgl = _crossgl(tmp_path, ATOMIC_KERNEL.replace("OPERATION", operation), suffix)
    assert f"var old: u32 = {operation}(outp[i], uin[i]);" in crossgl

    cuda = _flat(_translate(tmp_path, crossgl, ".cgl", "cuda"))
    assert f"unsigned int old = {operation}(&outp[i], uin[i]);" in cuda

    hip = _flat(_translate(tmp_path, crossgl, ".cgl", "hip"))
    if operation in ("atomicInc_system", "atomicDec_system"):
        # HIP declares neither; a compare-and-swap loop on atomicCAS_system performs it.
        helper = f"cgl_hip_{operation}"
        assert f"unsigned int old = {helper}(&outp[i], uin[i]);" in hip
        assert f"__device__ inline unsigned int {helper}(unsigned int* address" in hip
        assert "old = atomicCAS_system(address, assumed," in hip
    else:
        assert f"unsigned int old = {operation}(&outp[i], uin[i]);" in hip

    _assert_refused_by_other_targets(tmp_path, crossgl, operation)


@pytest.mark.parametrize("suffix", [".cu", ".hip"])
def test_system_compare_and_swap_keeps_its_scope(tmp_path, suffix):
    crossgl = _crossgl(tmp_path, CAS_KERNEL, suffix)
    assert "atomicCAS_system(outp[i], iin[i], iin[(i + 1u)])" in crossgl
    for backend in ("cuda", "hip"):
        code = _flat(_translate(tmp_path, crossgl, ".cgl", backend))
        assert "atomicCAS_system(&outp[i], iin[i], iin[(i + 1u)])" in code
    _assert_refused_by_other_targets(tmp_path, crossgl, "atomicCAS_system")


def _evaluate_update(update, assumed, limit):
    """Evaluate a helper's C update expression ``(condition) ? a : b`` on small unsigned values."""
    condition, when_true, when_false = re.fullmatch(
        r"\((.*)\) \? (.*) : (.*)", update
    ).groups()

    def python(text):
        return re.sub(r"\b(\d+)u\b", r"\1", text).replace("||", " or ")

    names = {"assumed": assumed, "limit": limit}
    chosen = when_true if eval(python(condition), {}, names) else when_false
    return eval(python(chosen), {}, names)


def test_hip_wrapping_atomic_helpers_update_like_cuda():
    from crosstl.translator.codegen.hip_buffer_calls import (
        HIP_SYSTEM_WRAPPING_ATOMIC_UPDATES,
    )

    increment = HIP_SYSTEM_WRAPPING_ATOMIC_UPDATES["cgl_hip_atomicInc_system"]
    decrement = HIP_SYSTEM_WRAPPING_ATOMIC_UPDATES["cgl_hip_atomicDec_system"]
    # CUDA C++ Programming Guide, "Atomic Functions": atomicInc stores ((old >= val) ? 0 : (old+1)); atomicDec
    # stores (((old == 0) || (old > val)) ? val : (old-1)).
    for old, limit in [(0, 0), (0, 5), (4, 5), (5, 5), (9, 5), (3, 0)]:
        assert _evaluate_update(increment, old, limit) == (
            0 if old >= limit else old + 1
        )
        assert _evaluate_update(decrement, old, limit) == (
            limit if old == 0 or old > limit else old - 1
        )


def test_syncwarp_with_a_mask_stays_a_cuda_call(tmp_path):
    crossgl = _crossgl(tmp_path, SYNCWARP_KERNEL, ".cu")
    assert "__syncwarp(mask);" in crossgl
    cuda = _flat(_translate(tmp_path, crossgl, ".cgl", "cuda"))
    assert "__syncwarp(mask);" in cuda
    _assert_refused_by_other_targets(tmp_path, crossgl, "CUDA warp")


def test_syncwarp_with_a_mask_needs_a_full_warp_mask_elsewhere(tmp_path):
    # In a one-warp workgroup the warp pass emulates __syncwarp only for a mask that names every lane.
    crossgl = """
shader WarpProbe {
    compute probe {
        layout(local_size_x = 32, local_size_y = 1, local_size_z = 1) in;
        buffer uint* data;
        void main() {
            uint i = gl_GlobalInvocationID.x;
            uint m = data[i];
            __syncwarp(m);
            data[i] = m;
        }
    }
}
"""
    with pytest.raises(Exception, match="mask that is not a literal") as raised:
        _translate(tmp_path, crossgl, ".cgl", "metal")
    _assert_typed_refusal(raised.value)
    metal = _translate(
        tmp_path,
        crossgl.replace("__syncwarp(m);", "__syncwarp(0xFFFFFFFFu);"),
        ".cgl",
        "metal",
    )
    assert "threadgroup_barrier(" in metal


def test_syncwarp_with_two_arguments_is_refused(tmp_path):
    source = SYNCWARP_KERNEL.replace("__syncwarp(mask);", "__syncwarp(mask, 1);")
    with pytest.raises(
        CudaImportRefused, match=r"__syncwarp\(mask, 1\) takes one mask"
    ):
        _translate(tmp_path, source, ".cu", "cgl")


def test_cuda_clocks_read_the_cycle_counter(tmp_path):
    crossgl = _crossgl(tmp_path, CUDA_CLOCK_KERNEL, ".cu")
    flat = _flat(crossgl)
    assert "var<storage, read_write> ticks: array<i64>" in flat
    assert "var start: i64 = i64(clockARB());" in flat
    assert "var low: i64 = i64(i32(clock2x32ARB().x));" in flat

    cuda = _flat(_translate(tmp_path, crossgl, ".cgl", "cuda"))
    assert "long long start = static_cast<long long>(clock64());" in cuda
    glsl = _translate(tmp_path, crossgl, ".cgl", "opengl")
    assert "#extension GL_ARB_shader_clock : require" in glsl
    assert "clock2x32ARB()" in glsl
    spirv = _translate(tmp_path, crossgl, ".cgl", "vulkan")
    assert "OpReadClockKHR" in spirv
    for backend in ("metal", "directx"):
        with pytest.raises(Exception, match="clock") as raised:
            _translate(tmp_path, crossgl, ".cgl", backend)
        _assert_typed_refusal(raised.value)


def test_hip_clocks_read_the_cycle_and_wall_counters(tmp_path):
    crossgl = _crossgl(tmp_path, HIP_CLOCK_KERNEL, ".hip")
    flat = _flat(crossgl)
    assert "var cycles: i64 = i64(clockARB());" in flat
    assert "var other: i64 = i64(clockARB());" in flat
    assert "var wall: i64 = i64(clockRealtimeEXT());" in flat

    hip = _flat(_translate(tmp_path, crossgl, ".cgl", "hip"))
    assert "static_cast<long long>(clock64())" in hip
    assert "static_cast<long long>(wall_clock64())" in hip
    glsl = _translate(tmp_path, crossgl, ".cgl", "opengl")
    assert "#extension GL_EXT_shader_realtime_clock : require" in glsl


@pytest.mark.parametrize(
    ("source", "error"),
    [
        (
            CUDA_CLOCK_KERNEL.replace("clock64();\n", "clock64(1);\n", 1),
            CudaImportRefused,
        ),
        (
            HIP_CLOCK_KERNEL.replace("wall_clock64()", "wall_clock64(2)"),
            HipImportRefused,
        ),
    ],
)
def test_clock_with_arguments_is_refused(tmp_path, source, error):
    suffix = ".cu" if error is CudaImportRefused else ".hip"
    with pytest.raises(error, match="takes no arguments"):
        _translate(tmp_path, source, suffix, "cgl")


@pytest.mark.parametrize("suffix", [".cu", ".hip"])
def test_initialized_device_variable_is_a_device_address_space_variable(
    tmp_path, suffix
):
    crossgl = _crossgl(tmp_path, DEVICE_GLOBAL_KERNEL, suffix)
    assert "var<device> acc: array<f32, 64> = {1.0f, 2.0f, 4.0f};" in crossgl
    assert "var<device> counter: i32 = 7;" in crossgl
    # The kernel's buffers keep bindings 0 and 1: an initialized variable binds no buffer.
    assert "@group(0) @binding(0) var<storage, read_write> fin: array<f32>" in crossgl
    assert "@group(0) @binding(1) var<storage, read_write> outp: array<f32>" in crossgl

    for backend in ("cuda", "hip"):
        code = _flat(_translate(tmp_path, crossgl, ".cgl", backend))
        assert "__device__ float acc[64] = {1.0f, 2.0f, 4.0f};" in code
        assert "__device__ int counter = 7;" in code
    _assert_refused_by_other_targets(tmp_path, crossgl, "device variable 'acc'")


@pytest.mark.parametrize("suffix", [".cu", ".hip"])
def test_uninitialized_device_variable_still_binds_a_buffer(tmp_path, suffix):
    source = DEVICE_GLOBAL_KERNEL.replace(" = {1.0f, 2.0f, 4.0f}", "")
    crossgl = _crossgl(tmp_path, source, suffix)
    assert "var<device> counter: i32 = 7;" in crossgl
    assert re.search(
        r"@group\(0\) @binding\(\d\) var<storage, read_write> acc: array<f32, 64>;",
        crossgl,
    )

"""Metal output gives each entry its own argument table (#3629, #3664).

A Metal entry function takes its resources as parameters, and ``[[buffer(n)]]`` indexes that function's own
argument table, so two kernels of one file may bind different buffers at the same index. OpenCL C and CUDA input
hoist each kernel's parameters to program scope and mark them with their kernel; CrossGL declares a stage's own
resources in its stage block. Metal output passes such a declaration to its own entry only and checks its binding in
that entry's table. Before, it checked every binding program-wide and passed every resource to every kernel: it
refused two kernels whose ``<kernel>_Args`` blocks or buffers shared an index, and it bound the second kernel's
``out`` at the first kernel's index when two kernels named different buffers ``out``.
"""

import re
import shutil
import struct
import subprocess

import pytest

import crosstl

TWO_SCALAR_KERNELS = """
extern "C" __global__ void fill(float* x, unsigned int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { x[i] = 1.0f; }
}
extern "C" __global__ void scale(float* x, unsigned int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { x[i] = x[i] * 2.0f; }
}
"""

OPENCL_SHARED_INDEX = """
__kernel void k1(__global const float* a, __global float* b) {
    size_t i = get_global_id(0);
    b[i] = a[i];
}

__kernel void k2(__global float* x) {
    size_t i = get_global_id(0);
    x[i] = 1.0f;
}
"""

CUDA_SHARED_INDEX = """
__global__ void k1(const float* a, float* b) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    b[i] = a[i];
}

__global__ void k2(float* x) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    x[i] = 1.0f;
}
"""

# Both kernels name a buffer `out`, at different indices.
CUDA_SAME_NAME = """
__global__ void k1(const float* a, float* out) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    out[i] = a[i] + 1.0f;
}

__global__ void k2(float* out) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    out[i] = 7.0f;
}
"""

CROSSGL_STAGE_BUFFERS = """
shader S {
    compute {
        layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
        layout(binding = 0) buffer float* a;
        layout(binding = 1) buffer float* out;
        void main() {
            uint i = gl_GlobalInvocationID.x;
            out[i] = a[i] * 3.0;
        }
    }
    compute {
        layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
        layout(binding = 0) buffer float* out;
        void main() {
            uint i = gl_GlobalInvocationID.x;
            out[i] = 5.0;
        }
    }
}
"""

# A shader-scope helper that names `out` becomes a function of each stage that calls it, reading that stage's `out`.
CROSSGL_SHARED_HELPER = """
shader S {
    void store(uint i, float v) {
        out[i] = v;
    }
    compute {
        layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
        layout(binding = 1) buffer float* out;
        void main() {
            store(gl_GlobalInvocationID.x, 4.0);
        }
    }
    compute {
        layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
        layout(binding = 0) buffer float* out;
        void main() {
            store(gl_GlobalInvocationID.x, 9.0);
        }
    }
}
"""

# Two stages declare different functions named `helper`.
CROSSGL_STAGE_HELPERS = """
shader S {
    compute {
        layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
        layout(binding = 0) buffer float* a;
        float helper(float v) { return v * 2.0; }
        void main() {
            uint i = gl_GlobalInvocationID.x;
            a[i] = helper(a[i]);
        }
    }
    compute {
        layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
        layout(binding = 0) buffer float* b;
        float helper(float v) { return v + 1.0; }
        void main() {
            uint i = gl_GlobalInvocationID.x;
            b[i] = helper(b[i]);
        }
    }
}
"""


def _translate(tmp_path, name, source):
    path = tmp_path / name
    path.write_text(source)
    return crosstl.translate(str(path), backend="metal")


def _kernel(metal, name):
    """The signature and body of kernel ``name``, up to the next kernel."""
    start = metal.index(f"kernel void {name}(")
    end = metal.find("kernel void ", start + 1)
    text = metal[start:] if end < 0 else metal[start:end]
    # clang-format, when it is installed, writes `float* a`; without it the output
    # reads `float *a`. Compare one spelling.
    return re.sub(r"\s*([*&])\s*", r" \1", text)


def _compile(tmp_path, metal):
    if shutil.which("xcrun") is None:
        return
    source = tmp_path / "out.metal"
    source.write_text(metal)
    result = subprocess.run(
        [
            "xcrun",
            "-sdk",
            "macosx",
            "metal",
            "-c",
            str(source),
            "-o",
            str(tmp_path / "out.air"),
        ],
        capture_output=True,
        text=True,
    )
    if "unable to find utility" in result.stderr:
        return
    assert result.returncode == 0, result.stderr


def _device():
    metal = pytest.importorskip("Metal")
    device = metal.MTLCreateSystemDefaultDevice()
    if device is None:
        pytest.skip("no Metal device")
    return metal, device


def _run(source, kernel, buffers, count, constants=None):
    """Dispatch ``count`` threads of ``kernel`` with float ``buffers`` (index -> values); return every buffer."""
    metal, device = _device()
    options = metal.MTLCompileOptions.new()
    options.setMathMode_(metal.MTLMathModeSafe)
    library, error = device.newLibraryWithSource_options_error_(source, options, None)
    assert library is not None, error
    pipeline, error = device.newComputePipelineStateWithFunction_error_(
        library.newFunctionWithName_(kernel), None
    )
    assert pipeline is not None, error
    commands = device.newCommandQueue().commandBuffer()
    encoder = commands.computeCommandEncoder()
    encoder.setComputePipelineState_(pipeline)
    handles = {}
    for index, values in buffers.items():
        data = struct.pack(f"{len(values)}f", *values)
        handles[index] = device.newBufferWithBytes_length_options_(data, len(data), 0)
        encoder.setBuffer_offset_atIndex_(handles[index], 0, index)
    for index, data in (constants or {}).items():
        encoder.setBytes_length_atIndex_(data, len(data), index)
    encoder.dispatchThreads_threadsPerThreadgroup_(
        metal.MTLSizeMake(count, 1, 1), metal.MTLSizeMake(1, 1, 1)
    )
    encoder.endEncoding()
    commands.commit()
    commands.waitUntilCompleted()
    return {
        index: list(
            struct.unpack(
                f"{len(buffers[index])}f",
                handle.contents().as_buffer(4 * len(buffers[index])),
            )
        )
        for index, handle in handles.items()
    }


def test_two_cuda_kernels_take_their_own_args_block_at_the_same_index(tmp_path):
    metal = _translate(tmp_path, "k.cu", TWO_SCALAR_KERNELS)
    fill, scale = _kernel(metal, "kernel_fill"), _kernel(metal, "kernel_scale")
    assert "constant fill_Args &fill_Args [[buffer(1)]]" in fill
    assert "scale_Args" not in fill
    assert "constant scale_Args &scale_Args [[buffer(1)]]" in scale
    assert "fill_Args" not in scale
    assert "fill_Args.n" in fill and "scale_Args.n" in scale
    _compile(tmp_path, metal)


def test_two_cuda_kernels_with_args_blocks_compute_their_sources(tmp_path):
    metal = _translate(tmp_path, "k.cu", TWO_SCALAR_KERNELS)
    n = struct.pack("I", 3)
    filled = _run(metal, "kernel_fill", {0: [0.0] * 5}, 5, {1: n})[0]
    assert filled == [1.0, 1.0, 1.0, 0.0, 0.0]
    scaled = _run(metal, "kernel_scale", {0: [1.0, 2.0, 3.0, 4.0, 5.0]}, 5, {1: n})[0]
    assert scaled == [2.0, 4.0, 6.0, 4.0, 5.0]


@pytest.mark.parametrize(
    "name, source",
    [("k.cl", OPENCL_SHARED_INDEX), ("k.cu", CUDA_SHARED_INDEX)],
)
def test_kernels_bind_different_buffers_at_the_same_index(tmp_path, name, source):
    metal = _translate(tmp_path, name, source)
    k1, k2 = _kernel(metal, "kernel_k1"), _kernel(metal, "kernel_k2")
    assert "*a [[buffer(0)]]" in k1 and "*b [[buffer(1)]]" in k1
    assert "*x [[buffer(0)]]" in k2
    assert " a [[" not in k2 and "*a" not in k2 and "*b" not in k2
    assert "*x" not in k1
    _compile(tmp_path, metal)
    copied = _run(metal, "kernel_k1", {0: [1.0, 2.0, 3.0], 1: [0.0] * 3}, 3)
    assert copied[1] == [1.0, 2.0, 3.0]
    assert _run(metal, "kernel_k2", {0: [0.0] * 3}, 3)[0] == [1.0] * 3


def test_kernels_that_name_different_buffers_alike_bind_each_at_its_own_index(
    tmp_path,
):
    metal = _translate(tmp_path, "k.cu", CUDA_SAME_NAME)
    k1, k2 = _kernel(metal, "kernel_k1"), _kernel(metal, "kernel_k2")
    assert "*out [[buffer(1)]]" in k1
    # k2's `out` is its buffer 0; Metal output used to pass k1's `out` at buffer 1 to k2 as well.
    assert "[[buffer(0)]]" in k2 and "[[buffer(1)]]" not in k2
    _compile(tmp_path, metal)
    first = _run(metal, "kernel_k1", {0: [1.0, 2.0], 1: [0.0, 0.0]}, 2)
    assert first[1] == [2.0, 3.0]
    assert _run(metal, "kernel_k2", {0: [0.0, 0.0]}, 2)[0] == [7.0, 7.0]


def test_crossgl_stage_buffers_bind_in_their_own_entry(tmp_path):
    metal = _translate(tmp_path, "s.cgl", CROSSGL_STAGE_BUFFERS)
    first, second = _kernel(metal, "kernel_main"), _kernel(metal, "kernel_main_2")
    assert "*a [[buffer(0)]]" in first and "*out [[buffer(1)]]" in first
    assert "[[buffer(0)]]" in second and "[[buffer(1)]]" not in second
    _compile(tmp_path, metal)
    tripled = _run(metal, "kernel_main", {0: [1.0, 2.0], 1: [0.0, 0.0]}, 2)
    assert tripled[1] == [3.0, 6.0]
    assert _run(metal, "kernel_main_2", {0: [0.0, 0.0]}, 2)[0] == [5.0, 5.0]


def test_a_shared_helper_reads_the_buffer_of_the_entry_that_calls_it(tmp_path):
    metal = _translate(tmp_path, "s.cgl", CROSSGL_SHARED_HELPER)
    _compile(tmp_path, metal)
    first = _run(metal, "kernel_main", {1: [0.0, 0.0]}, 2)
    assert first[1] == [4.0, 4.0]
    second = _run(metal, "kernel_main_2", {0: [0.0, 0.0]}, 2)
    assert second[0] == [9.0, 9.0]


def test_stage_functions_of_one_name_in_two_stages_stay_two_functions(tmp_path):
    metal = _translate(tmp_path, "s.cgl", CROSSGL_STAGE_HELPERS)
    _compile(tmp_path, metal)
    assert _run(metal, "kernel_main", {0: [1.0, 2.0]}, 2)[0] == [2.0, 4.0]
    assert _run(metal, "kernel_main_2", {0: [1.0, 2.0]}, 2)[0] == [2.0, 3.0]

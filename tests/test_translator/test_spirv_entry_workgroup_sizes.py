"""SPIR-V sizes each compute entry point on its own and lets the pipeline set a size the host chooses."""

import re
import shutil
import subprocess

import pytest

from crosstl.translator.codegen.spirv_refusals import SpirvLoweringRefused
from tests.spirv_friendly_names import translate_spirv

HEADER = "#include <metal_stdlib>\nusing namespace metal;\n"

# The size of ``reduce_sum`` is the host's; ``scale`` does not depend on it and runs one-invocation groups.
HOST_AND_FIXED = HEADER + """
kernel void reduce_sum(device const float* src [[buffer(0)]],
                       device float* dst [[buffer(1)]],
                       uint tid [[thread_position_in_threadgroup]],
                       uint gid [[threadgroup_position_in_grid]],
                       uint tsize [[threads_per_threadgroup]]) {
    threadgroup float scratch[256];
    scratch[tid] = src[gid * tsize + tid];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float total = 0.0f;
        for (uint i = 0; i < tsize; ++i) { total += scratch[i]; }
        dst[gid] = total;
    }
}

kernel void scale(device float* data [[buffer(0)]], uint id [[thread_position_in_grid]]) {
    data[id] = data[id] * 2.0f;
}
"""

HOST_ONLY = HOST_AND_FIXED.split("kernel void scale", 1)[0]

TWO_HOST_SIZED = HOST_ONLY + HOST_ONLY.split(HEADER, 1)[1].replace(
    "reduce_sum", "reduce_max"
).replace("total += scratch[i]", "total = max(total, scratch[i])")

FUNCTION_CONSTANT = (
    HEADER
    + "constant int kMode [[function_constant(0)]];\n"
    + (
        HOST_ONLY.split(HEADER, 1)[1].replace(
            "dst[gid] = total;", "dst[gid] = kMode == 1 ? total : -total;"
        )
    )
)

TWO_SIZES = """
shader main {
    compute {
        layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;
        @ stage_entry
        void first(RWStructuredBuffer<uint> data @buffer(0), uint gid @gl_GlobalInvocationID,
                   uint size @gl_WorkGroupSize) {
            buffer_store(data, gid, size);
        }
    }
    compute {
        layout(local_size_x = 8, local_size_y = 1, local_size_z = 1) in;
        @ stage_entry
        void second(RWStructuredBuffer<uint> data @buffer(1), uint gid @gl_GlobalInvocationID,
                    uint size @gl_WorkGroupSize) {
            buffer_store(data, gid, size);
        }
    }
}
"""

SHARED_HELPER = """
shader main {
    uint group_width() {
        return gl_WorkGroupSize.x;
    }
    compute {
        layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;
        @ stage_entry
        void first(RWStructuredBuffer<uint> data @buffer(0), uint gid @gl_GlobalInvocationID) {
            buffer_store(data, gid, group_width());
        }
    }
    compute {
        layout(local_size_x = 8, local_size_y = 1, local_size_z = 1) in;
        @ stage_entry
        void second(RWStructuredBuffer<uint> data @buffer(1), uint gid @gl_GlobalInvocationID) {
            buffer_store(data, gid, group_width());
        }
    }
}
"""


def _translate(tmp_path, source, suffix=".metal"):
    path = tmp_path / f"kernel{suffix}"
    path.write_text(source)
    return translate_spirv(path)


def _lines(code, pattern):
    return [line.strip() for line in code.splitlines() if re.search(pattern, line)]


def _spec_ids(code):
    return sorted(int(value) for value in re.findall(r"SpecId (\d+)", code))


def _validate(tmp_path, code, environment="vulkan1.1", *flags):
    if not (shutil.which("spirv-as") and shutil.which("spirv-val")):
        pytest.skip("SPIRV-Tools are not installed")
    assembly, binary = tmp_path / "kernel.spvasm", tmp_path / "kernel.spv"
    assembly.write_text(code)
    version = re.search(r"^; Version: (\d+\.\d+)", code, re.M).group(1)
    for command in (
        ["spirv-as", "--target-env", f"spv{version}", str(assembly), "-o", str(binary)],
        ["spirv-val", "--target-env", environment, *flags, str(binary)],
    ):
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        assert result.returncode == 0, result.stderr + result.stdout


def test_entries_of_different_sizes_each_read_their_own_size(tmp_path):
    code = _translate(tmp_path, TWO_SIZES, ".cgl")

    # A module-wide WorkgroupSize built-in would resize both entries to the first one's size.
    assert "BuiltIn WorkgroupSize" not in code
    assert _lines(code, r"OpExecutionMode .* LocalSize ") == [
        "OpExecutionMode %2 LocalSize 64 1 1",
        "OpExecutionMode %4 LocalSize 8 1 1",
    ]
    composites = _lines(code, r"= OpConstantComposite %v3uint")
    assert composites == [
        "%gl_WorkGroupSize = OpConstantComposite %v3uint %uint_64 %uint_1 %uint_1",
        "%gl_WorkGroupSize_0 = OpConstantComposite %v3uint %uint_8 %uint_1 %uint_1",
    ]
    _validate(tmp_path, code)


def test_shared_function_of_entries_of_different_sizes_is_refused(tmp_path):
    with pytest.raises(SpirvLoweringRefused, match="different workgroup sizes"):
        _translate(tmp_path, SHARED_HELPER, ".cgl")


def test_host_sized_kernel_takes_its_size_from_the_pipeline(tmp_path):
    code = _translate(tmp_path, HOST_ONLY)

    assert _lines(code, r"OpExecutionMode .* LocalSize ") == [
        "OpExecutionMode %2 LocalSize 1 1 1"
    ]
    assert "OpDecorate %gl_WorkGroupSize BuiltIn WorkgroupSize" in code
    assert re.search(
        r"%gl_WorkGroupSize = OpSpecConstantComposite %v3uint "
        r"%crossgl_workgroup_size_x %crossgl_workgroup_size_y %crossgl_workgroup_size_z",
        code,
    )
    assert _spec_ids(code) == [0, 1, 2]
    assert "; Version: 1.0" in code
    _validate(tmp_path, code)


def test_host_sized_kernels_of_one_module_share_the_specialization_ids(tmp_path):
    code = _translate(tmp_path, TWO_HOST_SIZED)

    # Each pipeline specializes its own entry, so one composite sizes both.
    assert len(_lines(code, r"OpEntryPoint GLCompute")) == 2
    assert _lines(code, r"BuiltIn WorkgroupSize") == [
        "OpDecorate %gl_WorkGroupSize BuiltIn WorkgroupSize"
    ]
    assert _spec_ids(code) == [0, 1, 2]
    _validate(tmp_path, code)


def test_host_sized_kernel_next_to_a_fixed_one_uses_local_size_id(tmp_path):
    code = _translate(tmp_path, HOST_AND_FIXED)

    assert "BuiltIn WorkgroupSize" not in code
    assert re.search(
        r"OpExecutionModeId %2 LocalSizeId %crossgl_workgroup_size_x "
        r"%crossgl_workgroup_size_y %crossgl_workgroup_size_z",
        code,
    )
    assert "OpExecutionMode %5 LocalSize 1 1 1" in code
    assert _spec_ids(code) == [0, 1, 2]
    # LocalSizeId is SPIR-V 1.2; Vulkan accepts it with VK_KHR_maintenance4, core in Vulkan 1.3.
    assert "; Version: 1.2" in code
    _validate(tmp_path, code, "vulkan1.3")
    _validate(tmp_path, code, "vulkan1.1", "--allow-localsizeid")


def test_host_size_ids_avoid_the_function_constants(tmp_path):
    code = _translate(tmp_path, FUNCTION_CONSTANT)

    assert _spec_ids(code) == [0, 1, 2, 3]
    workgroup_ids = [
        int(spec_id)
        for name, spec_id in re.findall(r"OpDecorate %(\S+) SpecId (\d+)", code)
        if not name.startswith("kMode")
    ]
    assert workgroup_ids == [1, 2, 3]
    _validate(tmp_path, code)


def test_cuda_block_size_takes_its_size_from_the_pipeline(tmp_path):
    source = """
__global__ void block_sum(const float* in, float* out) {
    __shared__ float tile[64];
    tile[threadIdx.x] = in[blockIdx.x * blockDim.x + threadIdx.x];
    __syncthreads();
    if (threadIdx.x == 0) out[blockIdx.x] = tile[0] + tile[blockDim.x - 1];
}
"""
    code = _translate(tmp_path, source, ".cu")

    assert "OpDecorate %gl_WorkGroupSize BuiltIn WorkgroupSize" in code
    assert _spec_ids(code) == [0, 1, 2]
    _validate(tmp_path, code)


NARROW_SIZE_PARAMETERS = """
#include <metal_stdlib>
using namespace metal;
kernel void k(device float *o [[buffer(0)]], uint tid [[thread_position_in_threadgroup]],
              uint tptg [[threads_per_threadgroup]], ushort2 t2 [[threads_per_threadgroup]]) {
    threadgroup float shared[64];
    shared[tid] = o[tid];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    o[tid] = shared[(tid + 1) % tptg] + float(t2.y);
}
"""


@pytest.mark.parametrize("threadgroup_size", [None, (32, 2, 1)])
def test_narrower_size_parameters_take_the_leading_components(
    tmp_path, threadgroup_size
):
    """``ushort2 [[threads_per_threadgroup]]`` holds the first two components of the uint3 built-in."""
    path = tmp_path / "kernel.metal"
    path.write_text(NARROW_SIZE_PARAMETERS)
    options = (
        {"source_options": {"threadgroup_size": threadgroup_size}}
        if threadgroup_size
        else {}
    )
    code = translate_spirv(path, **options)
    assert "OpStore %t2 %gl_WorkGroupSize" not in code
    _validate(tmp_path, code)

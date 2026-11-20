"""CUDA, HIP and OpenCL kernels, and ``@ compute fn`` entries, keep the size their host or source gives them."""

import re
import shutil
import subprocess

import pytest

import crosstl
from crosstl.translator.host_workgroup_size import (
    HOST_WORKGROUP_SIZE_KEY,
    HostDefinedWorkgroupSizeError,
    function_entry_execution_config,
    require_static_workgroup_size,
)
from crosstl.translator.lexer import Lexer
from crosstl.translator.parser import Parser
from crosstl.translator.semantics.warp import CudaWarpIntrinsicUnsupportedError
from tests.spirv_friendly_names import translate_spirv

SIZES = {
    "vulkan": r"OpExecutionMode %\S+ LocalSize (\d+) (\d+) (\d+)",
    "directx": r"\[numthreads\((\d+), (\d+), (\d+)\)\]",
    "opengl": r"local_size_x = (\d+), local_size_y = (\d+), local_size_z = (\d+)",
}

CUDA_SHARED = """
__global__ void block_sum(const float* in, float* out, int n) {
    __shared__ float tile[64];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    tile[threadIdx.x] = i < n ? in[i] : 0.0f;
    __syncthreads();
    if (threadIdx.x == 0) {
        float s = 0.0f;
        for (int k = 0; k < blockDim.x; ++k) s += tile[k];
        out[blockIdx.x] = s;
    }
}
"""

CUDA_WARP = """
__global__ void warp_sum(const int* in, int* out) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    int v = in[i];
    for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffffu, v, off);
    if ((threadIdx.x & 31u) == 0u) out[i >> 5] = v;
}
"""

CUDA_GRID_STRIDE = """
__global__ void fill(float* data, int n) {
    for (int i = threadIdx.x + blockDim.x * blockIdx.x; i < n; i += blockDim.x * gridDim.x) {
        data[i] = 2.0f;
    }
}
"""

CUDA_BLOCK_INDEX = """
__global__ void tag(int* out) {
    out[blockIdx.x * blockDim.x + threadIdx.x] = blockIdx.x;
}
"""

OPENCL_LOCAL = """
__kernel void block_sum(__global const float* in, __global float* out) {
    __local float tile[64];
    size_t lid = get_local_id(0);
    tile[lid] = in[get_global_id(0)];
    barrier(CLK_LOCAL_MEM_FENCE);
    if (lid == 0) out[get_group_id(0)] = tile[0] + tile[get_local_size(0) - 1];
}
"""

OPENCL_REQUIRED = (
    "__kernel __attribute__((reqd_work_group_size(32, 1, 1)))"
    + OPENCL_LOCAL.split("__kernel", 1)[1]
)


def _translate(tmp_path, source, suffix, backend, format_output=True, **options):
    path = tmp_path / f"kernel{suffix}"
    path.write_text(source)
    if backend == "vulkan":
        # Friendly ids whether or not SPIRV-Tools reformat the module.
        return translate_spirv(path, source_options=options or None)
    return crosstl.translate(
        str(path),
        backend=backend,
        source_options=options or None,
        format_output=format_output,
    )


def _size(code, backend):
    match = re.search(SIZES[backend], code)
    assert match, code
    return tuple(int(value) for value in match.groups())


# SPIR-V, GLSL and WGSL specialize a size the host chooses instead (test_spirv_entry_workgroup_sizes.py,
# test_glsl_host_workgroup_size.py, test_wgsl_host_workgroup_size.py).
HOST_NUMTHREADS = (
    "[numthreads(CROSSGL_WORKGROUP_SIZE_X, CROSSGL_WORKGROUP_SIZE_Y, "
    "CROSSGL_WORKGROUP_SIZE_Z)]"
)


def _require_static_sizes(crossgl):
    """Refuse every entry of a CrossGL program as a target that fixes the size in the shader, such as Slang, does."""
    shader = Parser(Lexer(crossgl).get_tokens()).parse()
    for stage in (shader.stages or {}).values():
        require_static_workgroup_size(stage.execution_config)
    for function in shader.functions or []:
        require_static_workgroup_size(function_entry_execution_config(function))


@pytest.mark.parametrize(
    "source, suffix, reason",
    [
        pytest.param(CUDA_SHARED, ".cu", "workgroup variable tile", id="cuda-shared"),
        pytest.param(CUDA_SHARED, ".hip", "workgroup variable tile", id="hip-shared"),
        pytest.param(OPENCL_LOCAL, ".cl", "workgroup variable tile", id="opencl-local"),
    ],
)
def test_size_dependent_kernel_is_refused_without_the_host_size(
    tmp_path, source, suffix, reason
):
    """A literal-only size is refused; HLSL reads the size from macros the host defines."""
    with pytest.raises(HostDefinedWorkgroupSizeError, match=reason):
        _require_static_sizes(_translate(tmp_path, source, suffix, "cgl"))
    code = _translate(tmp_path, source, suffix, "directx", format_output=False)
    assert HOST_NUMTHREADS in code


def test_warp_shuffle_kernel_needs_the_host_size(tmp_path):
    """A CUDA warp is 32 consecutive threads of the block, so the shuffle's exchange needs the block size: HLSL's
    driver-defined wave cannot stand in for it, and without a size every target refuses the kernel.
    """
    with pytest.raises(HostDefinedWorkgroupSizeError, match="calls __shfl_down_sync"):
        _require_static_sizes(_translate(tmp_path, CUDA_WARP, ".cu", "cgl"))
    with pytest.raises(
        CudaWarpIntrinsicUnsupportedError, match="no declared workgroup size"
    ):
        _translate(tmp_path, CUDA_WARP, ".cu", "directx", format_output=False)
    code = _translate(
        tmp_path, CUDA_WARP, ".cu", "directx", threadgroup_size=[64, 1, 1]
    )
    assert _size(code, "directx") == (64, 1, 1)
    assert "Wave" not in code and "GroupMemoryBarrierWithGroupSync" in code


@pytest.mark.parametrize("backend", sorted(SIZES))
@pytest.mark.parametrize(
    "source, suffix",
    [
        pytest.param(CUDA_SHARED, ".cu", id="cuda"),
        pytest.param(CUDA_SHARED, ".hip", id="hip"),
        pytest.param(OPENCL_LOCAL, ".cl", id="opencl"),
    ],
)
def test_threadgroup_size_option_fixes_the_target_size(
    tmp_path, backend, source, suffix
):
    code = _translate(tmp_path, source, suffix, backend, threadgroup_size=[64, 1, 1])
    assert _size(code, backend) == (64, 1, 1)


def test_threadgroup_size_option_sizes_the_block_dimension(tmp_path):
    code = _translate(tmp_path, CUDA_SHARED, ".cu", "directx", threadgroup_size=64)
    assert "[numthreads(64, 1, 1)]" in code
    assert "uint3(64, 1, 1).x" in code


@pytest.mark.parametrize("backend", sorted(SIZES))
def test_required_opencl_size_is_kept(tmp_path, backend):
    code = _translate(tmp_path, OPENCL_REQUIRED, ".cl", backend, threadgroup_size=64)
    assert _size(code, backend) == (32, 1, 1)


@pytest.mark.parametrize("backend", sorted(SIZES))
@pytest.mark.parametrize(
    "source",
    [
        pytest.param(CUDA_GRID_STRIDE, id="grid-stride"),
        pytest.param(CUDA_BLOCK_INDEX, id="block-index"),
    ],
)
def test_grid_index_and_grid_stride_need_no_block_size(tmp_path, backend, source):
    """blockIdx * blockDim + threadIdx and blockDim * gridDim do not depend on how the grid is split."""
    if source is CUDA_BLOCK_INDEX:
        # blockIdx alone does depend on the split, so SPIR-V and GLSL let the pipeline set the size.
        if backend == "vulkan":
            code = _translate(tmp_path, source, ".cu", backend)
            assert "OpDecorate %gl_WorkGroupSize BuiltIn WorkgroupSize" in code
            assert "SpecId 0" in code
            return
        if backend == "opengl":
            code = _translate(tmp_path, source, ".cu", backend)
            assert (
                "local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2" in code
            )
            return
        # HLSL reads the size from macros the host defines when it compiles the shader.
        code = _translate(tmp_path, source, ".cu", backend, format_output=False)
        assert HOST_NUMTHREADS in code
        with pytest.raises(HostDefinedWorkgroupSizeError, match="gl_WorkGroupID"):
            _require_static_sizes(_translate(tmp_path, source, ".cu", "cgl"))
        return
    assert _size(_translate(tmp_path, source, ".cu", backend), backend) == (1, 1, 1)


@pytest.mark.parametrize("backend", sorted(SIZES))
@pytest.mark.parametrize("suffix", [".cu", ".hip"])
def test_named_block_size_is_kept_when_results_do_not_depend_on_it(
    tmp_path, backend, suffix
):
    """CUDA and HIP launches cover whole blocks, so the named block size keeps the host's group count."""
    code = _translate(
        tmp_path, CUDA_GRID_STRIDE, suffix, backend, threadgroup_size=[64, 1, 1]
    )
    assert _size(code, backend) == (64, 1, 1)
    crossgl = _translate(
        tmp_path, CUDA_GRID_STRIDE, suffix, "cgl", threadgroup_size=[64, 1, 1]
    )
    assert "@workgroup_size(64, 1, 1)" in crossgl
    assert "@host_workgroup_size" not in crossgl


@pytest.mark.parametrize("suffix", [".cu", ".hip"])
def test_grid_index_imports_as_the_global_invocation_id(tmp_path, suffix):
    crossgl = _translate(tmp_path, CUDA_GRID_STRIDE, suffix, "cgl")
    assert "gl_GlobalInvocationID.x" in crossgl
    assert "gl_LocalInvocationID" not in crossgl
    assert "@host_workgroup_size" in crossgl
    assert "@workgroup_size" not in crossgl
    assert "let thread_id" not in crossgl


@pytest.mark.parametrize("suffix", [".cu", ".hip", ".cl"])
def test_metal_target_keeps_the_size_with_the_host(tmp_path, suffix):
    source = OPENCL_LOCAL if suffix == ".cl" else CUDA_SHARED
    code = _translate(tmp_path, source, suffix, "metal")
    assert re.search(r"kernel void\s+\w*block_sum", code)


def _function_entry(source):
    shader = Parser(Lexer(source).get_tokens()).parse()
    (function,) = shader.functions
    return function


FUNCTION_ENTRY = """
@compute
{size}
fn scan(@group(0) @binding(0) var<storage, read_write> data: array<f32>) {{
    var<workgroup> tile: array<f32, 64>;
    tile[gl_LocalInvocationID.x] = data[gl_GlobalInvocationID.x];
    workgroupBarrier();
    data[gl_GlobalInvocationID.x] = tile[0];
}}
"""


def test_parser_records_function_entry_sizes():
    explicit = _function_entry(FUNCTION_ENTRY.format(size="@workgroup_size(4, 2, 1)"))
    assert explicit.execution_config == {"numthreads": ("4", "2", "1")}
    hosted = _function_entry(
        FUNCTION_ENTRY.format(size="@host_workgroup_size(8, 1, 1)")
    )
    assert hosted.execution_config == {"numthreads": ("8", "1", "1")}
    assert all(attr.name != "host_workgroup_size" for attr in hosted.attributes)
    unsized = _function_entry(FUNCTION_ENTRY.format(size="@host_workgroup_size"))
    assert unsized.execution_config[HOST_WORKGROUP_SIZE_KEY] == {
        "entry": "scan",
        "reason": "declares the workgroup variable tile",
    }
    both = _function_entry(
        FUNCTION_ENTRY.format(size="@workgroup_size(16, 1, 1)\n@host_workgroup_size")
    )
    assert both.execution_config == {"numthreads": ("16", "1", "1")}


def test_parser_drops_the_mark_from_partition_invariant_reads():
    source = """
@compute
@host_workgroup_size
fn copy(@group(0) @binding(0) var<storage, read_write> data: array<u32>) {
    var i: u32 = gl_LocalInvocationID.y + gl_WorkGroupSize.y * gl_WorkGroupID.y;
    data[i] = gl_NumWorkGroups.y * gl_WorkGroupSize.y;
}
"""
    assert _function_entry(source).execution_config == {}


@pytest.mark.parametrize("backend", sorted(SIZES) + ["wgsl"])
def test_function_entry_workgroup_size_reaches_every_target(tmp_path, backend):
    """A top-level ``@ workgroup_size`` used to be dropped by the HLSL, GLSL and SPIR-V generators."""
    path = tmp_path / "scan.cgl"
    path.write_text(FUNCTION_ENTRY.format(size="@workgroup_size(4, 2, 1)"))
    code = crosstl.translate(str(path), backend=backend)
    if backend == "wgsl":
        assert "@workgroup_size(4, 2, 1)" in code
    else:
        assert _size(code, backend) == (4, 2, 1)


@pytest.mark.skipif(
    shutil.which("spirv-as") is None or shutil.which("spirv-val") is None,
    reason="SPIRV-Tools is not installed",
)
def test_sized_cuda_kernel_spirv_validates(tmp_path):
    code = _translate(tmp_path, CUDA_SHARED, ".cu", "vulkan", threadgroup_size=64)
    assembly = tmp_path / "kernel.spvasm"
    binary = tmp_path / "kernel.spv"
    assembly.write_text(code)
    subprocess.run(
        ["spirv-as", "--target-env", "vulkan1.1", str(assembly), "-o", str(binary)],
        check=True,
    )
    subprocess.run(
        ["spirv-val", "--target-env", "vulkan1.1", str(binary)],
        check=True,
    )
    assert re.search(r"%gl_WorkGroupSize = OpConstantComposite \S+ %uint_64 ", code)

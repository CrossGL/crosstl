"""WGSL lets the WebGPU pipeline set a workgroup size that the source's host chooses at dispatch time."""

import re
import shutil
import subprocess

import pytest

import crosstl
from crosstl.translator.codegen.specialized_workgroup_size import (
    SpecializedWorkgroupSizeError,
)
from crosstl.translator.codegen.wgsl_workgroup_locals import WGSLWorkgroupLocalError

HEADER = "#include <metal_stdlib>\nusing namespace metal;\n"

ROTATE = """
kernel void NAME(device float* data [[buffer(0)]],
                 uint gid [[thread_position_in_grid]],
                 uint lid [[thread_position_in_threadgroup]],
                 uint count [[threads_per_threadgroup]]) {
    threadgroup float tile[256];
    tile[lid] = data[gid] * SCALE;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    data[gid] = tile[(lid + 1) % count];
}
"""

TWO_CONSTANTS = """
constant float first [[function_constant(0)]];
constant float second [[function_constant(2)]];
"""

OVERRIDE = re.compile(
    r"@id\((\d+)\)\s*override (crossgl_workgroup_size_[xyz])\s*:\s*u32;"
)
HOST_SIZE = (
    "@workgroup_size(crossgl_workgroup_size_x, crossgl_workgroup_size_y, "
    "crossgl_workgroup_size_z)"
)

SPECIALIZED_PAIR = """
shader main {
    compute {
        layout(local_size_x = 64, local_size_x_id = FIRST) in;
        @ stage_entry
        void first(RWStructuredBuffer<uint> data @buffer(0), uint gid @gl_GlobalInvocationID,
                   uint size @gl_WorkGroupSize) {
            buffer_store(data, gid, size);
        }
    }
    compute {
        layout(local_size_x = 32, local_size_x_id = SECOND) in;
        @ stage_entry
        void second(RWStructuredBuffer<uint> data @buffer(1), uint gid @gl_GlobalInvocationID,
                    uint size @gl_WorkGroupSize) {
            buffer_store(data, gid, size);
        }
    }
}
"""


def _kernel(name="rotate", scale="1.0"):
    return ROTATE.replace("NAME", name).replace("SCALE", scale)


def _translate(tmp_path, source, backend="wgsl", name="kernel.metal"):
    path = tmp_path / name
    path.write_text(source)
    return crosstl.translate(str(path), backend=backend, format_output=False)


def _naga(tmp_path, code):
    if not shutil.which("naga"):
        return
    shader = tmp_path / "out.wgsl"
    shader.write_text(code)
    result = subprocess.run(
        ["naga", str(shader)], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr or result.stdout


def test_host_sized_kernel_gets_overrides_the_pipeline_must_set(tmp_path):
    code = _translate(tmp_path, HEADER + _kernel())
    assert OVERRIDE.findall(code) == [
        ("0", "crossgl_workgroup_size_x"),
        ("1", "crossgl_workgroup_size_y"),
        ("2", "crossgl_workgroup_size_z"),
    ]
    assert HOST_SIZE in code
    # threads_per_threadgroup reads the same overrides.
    assert (
        "vec3<u32>(crossgl_workgroup_size_x, crossgl_workgroup_size_y, "
        "crossgl_workgroup_size_z)" in code
    )
    _naga(tmp_path, code)


def test_wgsl_spirv_and_glsl_give_the_host_the_same_ids(tmp_path):
    source = HEADER + TWO_CONSTANTS + _kernel(scale="(first + second)")
    wgsl = _translate(tmp_path, source)
    assert [spec_id for spec_id, _ in OVERRIDE.findall(wgsl)] == ["1", "3", "4"]
    glsl = _translate(tmp_path, source, backend="opengl")
    assert re.search(
        r"local_size_x_id = 1,\s*local_size_y_id = 3,\s*local_size_z_id = 4", glsl
    )
    _naga(tmp_path, wgsl)


def test_host_sized_kernels_of_one_module_share_the_overrides(tmp_path):
    source = HEADER + _kernel("first_kernel") + _kernel("second_kernel", "2.0")
    code = _translate(tmp_path, source)
    assert len(OVERRIDE.findall(code)) == 3
    assert code.count(HOST_SIZE) == 2
    _naga(tmp_path, code)


def test_explicit_threadgroup_size_keeps_a_fixed_size(tmp_path):
    path = tmp_path / "kernel.metal"
    path.write_text(HEADER + _kernel())
    code = crosstl.translate(
        str(path),
        backend="wgsl",
        format_output=False,
        source_options={"threadgroup_size": (64, 1, 1)},
    )
    assert "@workgroup_size(64, 1, 1)" in code
    assert "override crossgl_workgroup_size" not in code


def test_entries_specialized_through_one_id_share_its_override(tmp_path):
    source = SPECIALIZED_PAIR.replace("FIRST", "4").replace("SECOND", "4")
    source = source.replace("local_size_x = 32", "local_size_x = 64")
    code = _translate(tmp_path, source, name="shader.cgl")
    assert code.count("override crossgl_workgroup_size_x") == 1
    _naga(tmp_path, code)


@pytest.mark.parametrize("first, second, default", [("4", "5", "64"), ("4", "4", "32")])
def test_entries_that_disagree_on_an_override_are_refused(
    tmp_path, first, second, default
):
    source = SPECIALIZED_PAIR.replace("FIRST", first).replace("SECOND", second)
    source = source.replace("local_size_x = 32", f"local_size_x = {default}")
    with pytest.raises(SpecializedWorkgroupSizeError) as raised:
        _translate(tmp_path, source, name="shader.cgl")
    assert raised.value.reason == "conflicting-workgroup-size-override"


OPENCL_LOCAL = """
__kernel void block_sum(__global const float* in, __global float* out) {
    __local float tile[64];
    size_t lid = get_local_id(0);
    tile[lid] = in[get_global_id(0)];
    barrier(CLK_LOCAL_MEM_FENCE);
    if (lid == 0) out[get_group_id(0)] = tile[0] + tile[get_local_size(0) - 1];
}
"""


def test_size_dependent_opencl_kernel_takes_the_host_size(tmp_path):
    code = _translate(tmp_path, OPENCL_LOCAL, name="kernel.cl")
    assert HOST_SIZE in code
    assert re.search(r"^var<workgroup> tile: array<f32, 64>;$", code, re.M)
    _naga(tmp_path, code)


def test_threadgroup_locals_become_module_workgroup_variables(tmp_path):
    code = _translate(tmp_path, HEADER + _kernel())
    assert re.search(r"^var<workgroup> tile: array<f32, 256>;$", code, re.M)
    assert not re.search(r"^\s+var tile\b", code, re.M)
    # [[threads_per_threadgroup]] has no WGSL built-in: it reads the workgroup size.
    assert "count: u32)" not in code
    assert re.search(r"var count: u32 = u32\(vec3<u32>\(crossgl_workgroup_size_x", code)


def test_threadgroup_locals_of_two_kernels_get_their_own_variables(tmp_path):
    source = HEADER + _kernel("first_kernel") + _kernel("second_kernel", "2.0")
    code = _translate(tmp_path, source)
    assert re.findall(r"^var<workgroup> (\w+):", code, re.M) == ["tile_1", "tile_2"]
    assert "tile_1[lid]" in code and "tile_2[lid]" in code
    _naga(tmp_path, code)


def test_threadgroup_local_of_a_helper_is_refused(tmp_path):
    source = HEADER + """
float stage_value(float value, uint lid) {
    threadgroup float tile[64];
    tile[lid] = value;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return tile[63 - lid];
}
kernel void reverse(device float* data [[buffer(0)]],
                    uint lid [[thread_position_in_threadgroup]]) {
    data[lid] = stage_value(data[lid], lid);
}
"""
    path = tmp_path / "kernel.metal"
    path.write_text(source)
    with pytest.raises(WGSLWorkgroupLocalError) as raised:
        crosstl.translate(
            str(path),
            backend="wgsl",
            format_output=False,
            source_options={"threadgroup_size": (64, 1, 1)},
        )
    assert raised.value.refusal == "helper-function"

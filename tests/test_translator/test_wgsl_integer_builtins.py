"""Integer built-in inputs whose source type is not WGSL's ``vec3<u32>`` or ``u32`` are received in that type."""

import shutil
import subprocess

import pytest

import crosstl

METAL = """#include <metal_stdlib>
using namespace metal;
kernel void k(device float* o [[buffer(0)]], uint i [[thread_position_in_grid]],
              uint2 l [[thread_position_in_threadgroup]], uint g [[threadgroup_position_in_grid]],
              uint3 n [[threadgroups_per_grid]]) {
    uint i_global_invocation_id = 1u;
    i += i_global_invocation_id;
    o[i] = float(l.x + l.y + g + n.z);
}
"""
HLSL = """RWStructuredBuffer<float> o : register(u0);
[numthreads(64, 1, 1)]
void main(uint id : SV_DispatchThreadID, int2 gt : SV_GroupThreadID, uint3 g : SV_GroupID) {
    o[id] = float(gt.x + gt.y) + float(g.x);
}
"""


def translate(tmp_path, source, suffix):
    path = tmp_path / f"kernel{suffix}"
    path.write_text(source)
    output = crosstl.translate(
        str(path),
        backend="wgsl",
        format_output=False,
        source_options={"threadgroup_size": (64, 1, 1)} if suffix == ".metal" else None,
    )
    if shutil.which("naga"):
        (tmp_path / "out.wgsl").write_text(output)
        checked = subprocess.run(
            ["naga", "out.wgsl"], cwd=tmp_path, capture_output=True, text=True
        )
        assert checked.returncode == 0, checked.stdout + checked.stderr
    return output


def test_metal_scalar_and_two_component_thread_positions(tmp_path):
    wgsl = translate(tmp_path, METAL, ".metal")
    assert "@builtin(global_invocation_id) i_global_invocation_id_1: vec3<u32>" in wgsl
    assert "@builtin(local_invocation_id) l_local_invocation_id: vec3<u32>" in wgsl
    assert "@builtin(workgroup_id) g_workgroup_id: vec3<u32>" in wgsl
    # A vec3<u32> parameter is received as it is.
    assert "@builtin(num_workgroups) n: vec3<u32>" in wgsl
    assert "var i: u32 = u32(i_global_invocation_id_1.x);" in wgsl
    assert "var l: vec2<u32> = vec2<u32>(l_local_invocation_id.xy);" in wgsl
    assert "var g: u32 = u32(g_workgroup_id.x);" in wgsl
    # The source parameter is a local, which the body may write.
    assert "i += i_global_invocation_id;" in wgsl


VERTEX = """#include <metal_stdlib>
using namespace metal;
struct Out { float4 position [[position]]; };
vertex Out v(const device float4* positions [[buffer(0)]], uint vid [[vertex_id]],
             uint iid [[instance_id]]) {
    Out out;
    out.position = positions[vid] + float4(float(iid));
    return out;
}
"""


def test_signed_vertex_and_instance_indices(tmp_path):
    # CrossGL types gl_VertexID and gl_InstanceID int, as GLSL does; WGSL's built-ins are u32.
    wgsl = translate(tmp_path, VERTEX, ".metal")
    assert "@builtin(vertex_index) vid_vertex_index: u32" in wgsl
    assert "@builtin(instance_index) iid_instance_index: u32" in wgsl
    assert "var vid: i32 = i32(vid_vertex_index);" in wgsl


def test_hlsl_signed_and_scalar_dispatch_ids(tmp_path):
    wgsl = translate(tmp_path, HLSL, ".hlsl")
    assert "var id: u32 = u32(id_global_invocation_id.x);" in wgsl
    assert "var gt: vec2<i32> = vec2<i32>(gt_local_invocation_id.xy);" in wgsl
    assert "@builtin(workgroup_id) g: vec3<u32>" in wgsl


@pytest.mark.parametrize("backend", ["vulkan", "opengl"])
def test_other_targets_keep_the_source_parameters(tmp_path, backend):
    path = tmp_path / "kernel.hlsl"
    path.write_text(HLSL)
    output = crosstl.translate(str(path), backend=backend, format_output=False)
    assert "id_global_invocation_id" not in output

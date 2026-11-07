"""A library with a kernel whose size only the host chooses translates to every target that emits compute.

No target refuses such a kernel: Metal and CUDA leave the size to the dispatch, SPIR-V, GLSL and WGSL specialize
it, HLSL reads it from macros the host defines, and Slang reads those macros or specialization constants. The other
kernels of the library keep their own sizes.
"""

import pytest

import crosstl

HEADER = "#include <metal_stdlib>\nusing namespace metal;\n"

# ``scale`` runs one invocation per workgroup exactly; ``rotate`` shares threadgroup memory.
SCALE = """
kernel void scale(device float* data [[buffer(0)]], uint gid [[thread_position_in_grid]]) {
    data[gid] = data[gid] * 2.0;
}
"""
ROTATE = """
kernel void rotate(device float* data [[buffer(0)]],
                   uint gid [[thread_position_in_grid]],
                   uint3 lid [[thread_position_in_threadgroup]]) {
    threadgroup float tile[256];
    tile[lid.x] = data[gid];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    data[gid] = tile[(lid.x + 1) % 256];
}
"""
KERNELS = HEADER + SCALE + ROTATE

STAGE_BLOCK = """
shader main {
    struct FragmentOutput {
        vec4 color;
    }

    fragment {
        FragmentOutput main() {
            FragmentOutput output;
            output.color = vec4(1.0);
            return output;
        }
    }

    compute {
        @ host_workgroup_size
        @ stage_entry
        void rotate(RWStructuredBuffer<float> data @buffer(0), uint gid @gl_GlobalInvocationID) {
            threadgroup float[64] tile;
            tile[gid % 64u] = buffer_load(data, gid);
            workgroupBarrier();
            buffer_store(data, gid, tile[(gid + 1u) % 64u]);
        }
    }
}
"""


def _write(tmp_path, name, source):
    path = tmp_path / name
    path.write_text(source)
    return str(path)


@pytest.mark.parametrize(
    "backend", ["cuda", "directx", "metal", "opengl", "slang", "vulkan", "wgsl"]
)
@pytest.mark.parametrize(
    "name, source", [("rotate.cgl", STAGE_BLOCK), ("kernels.metal", KERNELS)]
)
def test_no_target_refuses_a_host_sized_kernel(tmp_path, backend, name, source):
    assert crosstl.translate(_write(tmp_path, name, source), backend=backend)


def test_selected_entry_is_left_to_its_generator(tmp_path):
    """An entry-scoped translation of the kernel that needs no size is not refused for the other kernel."""
    code = crosstl.translate(
        _write(tmp_path, "kernels.metal", KERNELS),
        backend="directx",
        entry_point="scale",
    )
    assert "[numthreads(1, 1, 1)]" in code
    assert "rotate" not in code

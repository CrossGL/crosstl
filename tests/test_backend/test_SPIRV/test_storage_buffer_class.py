"""Storage buffers in SPIR-V 1.4 modules use the ``StorageBuffer`` storage class and ``Block``.

The SPIR-V specification's Decoration table lists ``BufferBlock`` as missing after version 1.3. Mesh, task and
ray tracing modules must be SPIR-V 1.4 (SPV_EXT_mesh_shader, SPV_KHR_ray_tracing), yet the emitter declared
their storage buffers as ``Uniform`` + ``BufferBlock``, which ``spirv-val`` rejects. Modules for other stages keep
``Uniform`` + ``BufferBlock`` so they stay valid SPIR-V 1.0 for Vulkan 1.0 and 1.1.
"""

from __future__ import annotations

import re
import shutil
import subprocess

import pytest

import crosstl

RAY_GENERATION = """shader Probe {
    ray_generation {
        buffer float* outp;
        void main() {
            outp[0] = 1.0;
        }
    }
}
"""

RAY_GENERATION_ATOMIC = """shader Probe {
    ray_generation {
        buffer uint* counters;
        void main() {
            atomicAdd(counters[0], 1u);
        }
    }
}
"""

MESH = """shader Probe {
    struct Vertex {
        vec4 position @ gl_Position;
    }
    mesh {
        buffer vec4* positions;
        void main(@ vertices out Vertex verts[3], @ indices out uvec3 tris[1])
            @numthreads(1, 1, 1)
            @outputtopology(triangle) {
            SetMeshOutputCounts(3, 1);
            verts[0].position = positions[0];
            verts[1].position = positions[1];
            verts[2].position = positions[2];
            tris[0] = uvec3(0, 1, 2);
        }
    }
}
"""

COMPUTE = """shader Probe {
    compute {
        buffer float* outp;
        void main() @numthreads(1, 1, 1) {
            outp[0] = 1.0;
        }
    }
}
"""

GLSL_RAY_GENERATION = """#version 460
#extension GL_EXT_ray_tracing : require
layout(std430, set = 0, binding = 0) buffer Output { float values[]; } results;
void main() { results.values[gl_LaunchIDEXT.x] = 1.0; }
"""


def translate(tmp_path, source, suffix=".cgl", source_backend=None):
    path = tmp_path / f"shader{suffix}"
    path.write_text(source)
    kwargs = {"source_backend": source_backend} if source_backend else {}
    return crosstl.translate(str(path), backend="vulkan", format_output=False, **kwargs)


def spirv_val(tmp_path, code, env):
    if not (shutil.which("spirv-as") and shutil.which("spirv-val")):
        pytest.skip("SPIRV-Tools is not installed")
    (tmp_path / "out.spvasm").write_text(code)
    assembled = subprocess.run(
        ["spirv-as", "--target-env", env, "out.spvasm", "-o", "out.spv"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert assembled.returncode == 0, assembled.stderr
    validated = subprocess.run(
        ["spirv-val", "--target-env", env, "out.spv"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert validated.returncode == 0, validated.stderr


def buffer_variable_storage_classes(code):
    """The storage class of every descriptor-bound variable."""
    bound = set(re.findall(r"OpDecorate %(\w+) Binding \d+", code))
    return {
        name: storage
        for name, storage in re.findall(r"%(\w+) = OpVariable %\w+ (\w+)", code)
        if name in bound
    }


@pytest.mark.parametrize(
    "source",
    [RAY_GENERATION, RAY_GENERATION_ATOMIC, MESH],
    ids=["ray-generation", "ray-generation-atomic", "mesh"],
)
def test_spirv_1_4_modules_declare_storage_buffers_without_buffer_block(
    tmp_path, source
):
    spirv = translate(tmp_path, source)
    assert "; Version: 1.4" in spirv
    assert "BufferBlock" not in spirv
    assert re.search(r"OpDecorate %\w+ Block\b", spirv)
    assert set(buffer_variable_storage_classes(spirv).values()) == {"StorageBuffer"}
    spirv_val(tmp_path, spirv, "vulkan1.2")


def test_glsl_ray_generation_buffer_block_round_trips_to_valid_spirv(tmp_path):
    spirv = translate(tmp_path, GLSL_RAY_GENERATION, ".rgen", "opengl")
    assert "BufferBlock" not in spirv
    assert set(buffer_variable_storage_classes(spirv).values()) == {"StorageBuffer"}
    spirv_val(tmp_path, spirv, "vulkan1.2")


def test_compute_modules_keep_spirv_1_0_buffer_blocks(tmp_path):
    spirv = translate(tmp_path, COMPUTE)
    assert "; Version: 1.0" in spirv
    assert re.search(r"OpDecorate %\w+ BufferBlock\b", spirv)
    assert set(buffer_variable_storage_classes(spirv).values()) == {"Uniform"}
    spirv_val(tmp_path, spirv, "vulkan1.0")

"""Barrier semantics across importers and emitters: no translation may weaken a barrier.

CrossGL's barrier vocabulary follows GLSL 4.60 section 8.16: ``memoryBarrierShared`` orders shared memory and
``groupMemoryBarrier`` orders buffer, image and shared memory, both within the workgroup; ``deviceMemoryBarrier``
orders buffers and images at device scope (HLSL ``DeviceMemoryBarrier``); ``memoryBarrier``/``allMemoryBarrier``
order all three at device scope. SPIR-V ``OpControlBarrier``/``OpMemoryBarrier`` import to the CrossGL barrier that
covers the instruction's scope and storage classes.
"""

import shutil
import subprocess

import pytest

import crosstl
from crosstl.backend.SPIRV.barrier_import import (
    SpirvBarrierUnsupportedError,
    spirv_barrier_statements,
)

ALL_BARRIERS_CROSSGL = """
shader main {
    RWStructuredBuffer<float> o @binding(0);
    shared float s[4];
    compute {
        layout(local_size_x = 4) in;
        void main() {
            s[0] = 1.0;
            deviceMemoryBarrier();
            allMemoryBarrier();
            groupMemoryBarrier();
            memoryBarrierShared();
            workgroupBarrier();
            buffer_store(o, 0u, s[1]);
        }
    }
}
"""

GLSL_BARRIERS = """#version 450
layout(local_size_x = 4) in;
layout(std430, binding = 0) buffer B { float d[]; } b;
layout(binding = 1, r32f) uniform image2D img;
shared float s[4];
void main() {
    uint i = gl_LocalInvocationID.x;
    s[i] = b.d[i];
    memoryBarrierShared();
    groupMemoryBarrier();
    memoryBarrierBuffer();
    memoryBarrierImage();
    memoryBarrier();
    barrier();
    b.d[i] = s[(i + 1u) % 4u];
}
"""


def _names(statements):
    return [statement.name for statement in statements]


@pytest.mark.parametrize(
    "opcode,operands,expected",
    [
        # glslang barrier() in compute; DXC GroupMemoryBarrierWithGroupSync.
        ("OpControlBarrier", ["2", "2", "264"], ["workgroupBarrier"]),
        # No memory semantics: an execution barrier only.
        ("OpControlBarrier", ["2", "2", "0"], ["workgroupExecutionBarrier"]),
        # DXC AllMemoryBarrierWithGroupSync: fence, barrier, fence.
        (
            "OpControlBarrier",
            ["2", "1", "2376"],
            ["memoryBarrier", "workgroupBarrier", "memoryBarrier"],
        ),
        # DXC DeviceMemoryBarrierWithGroupSync.
        (
            "OpControlBarrier",
            ["2", "1", "2120"],
            ["deviceMemoryBarrier", "workgroupBarrier", "deviceMemoryBarrier"],
        ),
        # A workgroup-scope control barrier over buffer memory (Metal threadgroup_barrier(mem_device)).
        (
            "OpControlBarrier",
            ["2", "2", "72"],
            ["groupMemoryBarrier", "workgroupBarrier", "groupMemoryBarrier"],
        ),
        ("OpMemoryBarrier", ["2", "264"], ["memoryBarrierShared"]),
        # glslang memoryBarrierShared(): Device scope over workgroup memory only.
        ("OpMemoryBarrier", ["1", "264"], ["memoryBarrierShared"]),
        ("OpMemoryBarrier", ["2", "3400"], ["groupMemoryBarrier"]),
        ("OpMemoryBarrier", ["1", "3400"], ["memoryBarrier"]),
        ("OpMemoryBarrier", ["1", "72"], ["memoryBarrierBuffer"]),
        ("OpMemoryBarrier", ["1", "2056"], ["memoryBarrierImage"]),
        ("OpMemoryBarrier", ["1", "2120"], ["deviceMemoryBarrier"]),
        # QueueFamily is narrower than Device, so a device-scope fence covers it.
        ("OpMemoryBarrier", ["5", "72"], ["memoryBarrierBuffer"]),
        # Subgroup scope is covered by workgroup scope.
        ("OpMemoryBarrier", ["3", "264"], ["memoryBarrierShared"]),
        # No storage class: the barrier orders nothing.
        ("OpMemoryBarrier", ["1", "8"], []),
    ],
)
def test_spirv_barriers_import_to_covering_crossgl_barriers(opcode, operands, expected):
    assert _names(spirv_barrier_statements(opcode, operands, ["GLCompute"])) == expected


def test_spirv_tessellation_control_barrier_imports_as_patch_barrier():
    for semantics in ("0", "4096"):
        statements = spirv_barrier_statements(
            "OpControlBarrier", ["2", "4", semantics], ["TessellationControl"]
        )
        assert _names(statements) == ["barrier"]


@pytest.mark.parametrize(
    "opcode,operands",
    [
        ("OpControlBarrier", ["3", "3", "264"]),  # subgroup execution scope
        ("OpMemoryBarrier", ["0", "72"]),  # CrossDevice memory scope
        ("OpControlBarrier", ["2", "2", "semantics_spec"]),  # not a constant
    ],
)
def test_spirv_barriers_without_crossgl_equivalent_are_refused(opcode, operands):
    with pytest.raises(SpirvBarrierUnsupportedError):
        spirv_barrier_statements(opcode, operands, ["GLCompute"])


def _translate(tmp_path, source, backend, name):
    path = tmp_path / name
    path.write_text(source)
    return crosstl.translate(str(path), backend=backend)


def test_glsl_emits_device_fences_as_memory_barrier(tmp_path):
    glsl = _translate(tmp_path, ALL_BARRIERS_CROSSGL, "opengl", "shader.cgl")

    assert "deviceMemoryBarrier" not in glsl
    assert "allMemoryBarrier" not in glsl
    assert glsl.count("memoryBarrier();") == 2
    assert "groupMemoryBarrier();" in glsl
    if shutil.which("glslangValidator"):
        path = tmp_path / "out.comp"
        path.write_text(glsl)
        result = subprocess.run(
            ["glslangValidator", str(path)], capture_output=True, text=True
        )
        assert result.returncode == 0, result.stdout + result.stderr


def test_group_memory_barrier_is_not_narrowed_to_shared_memory(tmp_path):
    hlsl = _translate(tmp_path, ALL_BARRIERS_CROSSGL, "directx", "shader.cgl")
    metal = _translate(tmp_path, ALL_BARRIERS_CROSSGL, "metal", "shader.cgl")
    spirv = _translate(tmp_path, ALL_BARRIERS_CROSSGL, "vulkan", "shader.cgl")

    # groupMemoryBarrier, allMemoryBarrier -> AllMemoryBarrier; memoryBarrierShared -> GroupMemoryBarrier.
    assert hlsl.count("AllMemoryBarrier();") == 2
    assert hlsl.count("GroupMemoryBarrier();") == 1
    assert hlsl.count("DeviceMemoryBarrier();") == 1
    all_flags = (
        "threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup | "
        "mem_flags::mem_texture);"
    )
    assert " ".join(metal.split()).count(all_flags) == 2
    assert "threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_texture);" in (
        " ".join(metal.split())
    )
    # Workgroup scope with uniform, workgroup and image memory, as glslang emits groupMemoryBarrier.
    assert "OpMemoryBarrier %uint_2 %uint_2376" in spirv
    assert "OpMemoryBarrier %uint_1 %uint_2120" in spirv


def test_hlsl_group_memory_barrier_imports_as_shared_memory_fence(tmp_path):
    hlsl = """
RWStructuredBuffer<float> o : register(u0);
groupshared float s[4];
[numthreads(4,1,1)] void CSMain(uint3 t : SV_GroupThreadID) {
    s[t.x] = 1.0;
    GroupMemoryBarrier();
    o[t.x] = s[(t.x + 1) % 4];
}
"""
    crossgl = _translate(tmp_path, hlsl, "cgl", "shader.comp.hlsl")
    regenerated = _translate(tmp_path, hlsl, "directx", "shader.comp.hlsl")

    assert "memoryBarrierShared();" in crossgl
    assert "groupMemoryBarrier" not in crossgl
    assert "GroupMemoryBarrier();" in regenerated


@pytest.mark.skipif(
    not (shutil.which("glslangValidator") and shutil.which("spirv-dis")),
    reason="glslangValidator and spirv-dis are required",
)
def test_glslang_barriers_round_trip_through_spirv_to_every_target(tmp_path):
    source = tmp_path / "b.comp"
    source.write_text(GLSL_BARRIERS)
    binary = tmp_path / "b.spv"
    subprocess.run(
        ["glslangValidator", "-V", str(source), "-o", str(binary)],
        check=True,
        capture_output=True,
    )
    disassembly = subprocess.run(
        ["spirv-dis", str(binary)], check=True, capture_output=True, text=True
    ).stdout

    crossgl = _translate(tmp_path, disassembly, "cgl", "b.spvasm")
    body = crossgl[crossgl.index("void main") :]
    order = [
        line.strip()
        for line in body.splitlines()
        if line.strip().endswith("();") and "arrier" in line
    ]
    assert order == [
        "memoryBarrierShared();",
        "groupMemoryBarrier();",
        "memoryBarrierBuffer();",
        "memoryBarrierImage();",
        "memoryBarrier();",
        "workgroupBarrier();",
    ]
    for backend in ("opengl", "directx", "metal", "vulkan"):
        output = _translate(tmp_path, disassembly, backend, "b.spvasm")
        assert "spirvControlBarrier" not in output
        assert "spirvMemoryBarrier" not in output

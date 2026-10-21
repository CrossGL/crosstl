"""Barrier semantics across importers and emitters: no translation may weaken a barrier.

CrossGL's barrier vocabulary follows GLSL 4.60 section 8.16: ``memoryBarrierShared`` orders shared memory and
``groupMemoryBarrier`` orders buffer, image and shared memory, both within the workgroup; ``deviceMemoryBarrier``
orders buffers and images at device scope (HLSL ``DeviceMemoryBarrier``); ``memoryBarrier``/``allMemoryBarrier``
order all three at device scope. SPIR-V ``OpControlBarrier``/``OpMemoryBarrier`` import to the CrossGL barrier that
covers the instruction's scope and storage classes.
"""

import re
import shutil
import subprocess

import pytest

import crosstl
from crosstl.backend.Metal.barrier_import import metal_barrier_statements
from crosstl.backend.SPIRV.barrier_import import (
    SpirvBarrierUnsupportedError,
    spirv_barrier_statements,
)
from crosstl.translator.codegen.barrier_peephole import (
    HLSL_EXECUTION_BARRIER_PLACEHOLDER,
    merge_adjacent_hlsl_barriers,
    merge_adjacent_metal_barriers,
)
from tests.spirv_friendly_names import friendly_spirv

ALL_BARRIERS_CROSSGL = """
shader main {
    RWStructuredBuffer<float> o @binding(0);
    shared float s[4];
    compute {
        layout(local_size_x = 4) in;
        void main() {
            s[0] = 1.0;
            deviceMemoryBarrier();
            s[1] = 2.0;
            allMemoryBarrier();
            s[2] = 3.0;
            groupMemoryBarrier();
            s[3] = 4.0;
            memoryBarrierShared();
            s[0] = 5.0;
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
        # DXC DeviceMemoryBarrierWithGroupSync: no workgroup memory, so an execution barrier between the fences.
        (
            "OpControlBarrier",
            ["2", "1", "2120"],
            [
                "deviceMemoryBarrier",
                "workgroupExecutionBarrier",
                "deviceMemoryBarrier",
            ],
        ),
        # A workgroup-scope control barrier over buffer memory (Metal threadgroup_barrier(mem_device)).
        (
            "OpControlBarrier",
            ["2", "2", "72"],
            ["groupMemoryBarrier", "workgroupExecutionBarrier", "groupMemoryBarrier"],
        ),
        # glslang subgroupBarrier() and a subgroup execution barrier.
        ("OpControlBarrier", ["3", "3", "2376"], ["subgroupBarrier"]),
        ("OpControlBarrier", ["3", "3", "0"], ["subgroupExecutionBarrier"]),
        # A subgroup control barrier ordering buffers at device scope keeps device-scope fences.
        (
            "OpControlBarrier",
            ["3", "1", "72"],
            ["memoryBarrierBuffer", "subgroupExecutionBarrier", "memoryBarrierBuffer"],
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
        # glslang subgroupMemoryBarrier*() fences.
        ("OpMemoryBarrier", ["3", "264"], ["subgroupMemoryBarrierShared"]),
        ("OpMemoryBarrier", ["3", "72"], ["subgroupMemoryBarrierBuffer"]),
        ("OpMemoryBarrier", ["3", "2056"], ["subgroupMemoryBarrierImage"]),
        ("OpMemoryBarrier", ["3", "2376"], ["subgroupMemoryBarrier"]),
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
        ("OpControlBarrier", ["1", "1", "264"]),  # device execution scope
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
    if backend == "vulkan":
        return friendly_spirv(
            crosstl.translate(str(path), backend=backend, format_output=False)
        )
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


@pytest.mark.parametrize(
    "name,flags,expected",
    [
        ("threadgroup_barrier", {"mem_none"}, ("workgroupExecutionBarrier",)),
        ("threadgroup_barrier", {"mem_threadgroup"}, ("workgroupBarrier",)),
        (
            "threadgroup_barrier",
            {"mem_device"},
            ("memoryBarrierBuffer", "workgroupExecutionBarrier", "memoryBarrierBuffer"),
        ),
        (
            "threadgroup_barrier",
            {"mem_texture"},
            ("memoryBarrierImage", "workgroupExecutionBarrier", "memoryBarrierImage"),
        ),
        (
            "threadgroup_barrier",
            {"mem_device", "mem_threadgroup"},
            ("memoryBarrierBuffer", "workgroupBarrier", "memoryBarrierBuffer"),
        ),
        (
            "threadgroup_barrier",
            {"mem_device", "mem_threadgroup", "mem_texture"},
            ("deviceMemoryBarrier", "workgroupBarrier", "deviceMemoryBarrier"),
        ),
        ("simdgroup_barrier", {"mem_none"}, ("subgroupExecutionBarrier",)),
        ("simdgroup_barrier", {"mem_threadgroup"}, ("subgroupBarrier",)),
        ("simdgroup_barrier", {"mem_device", "mem_texture"}, ("subgroupBarrier",)),
    ],
)
def test_metal_barriers_import_to_covering_crossgl_barriers(name, flags, expected):
    assert metal_barrier_statements(name, flags) == expected


@pytest.mark.parametrize(
    "flags",
    [None, {"mem_threadgroup_imageblock"}, {"mem_object_data", "mem_device"}],
)
def test_metal_barriers_without_crossgl_equivalent_are_refused(flags):
    assert isinstance(metal_barrier_statements("threadgroup_barrier", flags), str)


METAL_DEVICE_BARRIER = """
#include <metal_stdlib>
using namespace metal;
kernel void k(device uint* d [[buffer(0)]], uint i [[thread_position_in_grid]]) {
    d[i] = i;
    threadgroup_barrier(mem_flags::mem_device);
    d[i] = d[i ^ 1u];
    simdgroup_barrier(mem_flags::mem_threadgroup);
}
"""


def _barrier_lines(code):
    return [
        line.strip()
        for line in code.splitlines()
        if re.match(r"\s*\w*[Bb]arrier\w*\(", line)
    ]


def test_metal_device_barrier_keeps_its_execution_barrier_on_every_target(tmp_path):
    crossgl = _translate(tmp_path, METAL_DEVICE_BARRIER, "cgl", "k.metal")
    glsl = _translate(tmp_path, METAL_DEVICE_BARRIER, "opengl", "k.metal")
    metal = _translate(tmp_path, METAL_DEVICE_BARRIER, "metal", "k.metal")
    spirv = _translate(tmp_path, METAL_DEVICE_BARRIER, "vulkan", "k.metal")

    assert _barrier_lines(crossgl) == [
        "memoryBarrierBuffer();",
        "workgroupExecutionBarrier();",
        "memoryBarrierBuffer();",
        "subgroupBarrier();",
    ]
    # threadgroup_barrier(mem_device) used to become memoryBarrierBuffer() alone, a fence without the barrier.
    assert _barrier_lines(glsl) == [
        "memoryBarrierBuffer();",
        "barrier();",
        "memoryBarrierBuffer();",
        "subgroupBarrier();",
    ]
    assert _barrier_lines(metal)[0] == "threadgroup_barrier(mem_flags::mem_device);"
    assert "OpControlBarrier %uint_2 %uint_2 %uint_0" in spirv
    # Subgroup execution and memory scope, acquire-release over uniform, workgroup and image memory.
    assert "OpControlBarrier %uint_3 %uint_3 %uint_2376" in spirv
    if shutil.which("spirv-as") and shutil.which("spirv-val"):
        assembly = tmp_path / "k.spvasm"
        assembly.write_text(spirv)
        binary = tmp_path / "k.spv"
        subprocess.run(
            ["spirv-as", "--target-env", "vulkan1.1", str(assembly), "-o", str(binary)],
            check=True,
            capture_output=True,
        )
        result = subprocess.run(
            ["spirv-val", "--target-env", "vulkan1.1", str(binary)],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr


def test_metal_device_barrier_lowers_to_hlsl_device_barrier_with_group_sync(tmp_path):
    source = METAL_DEVICE_BARRIER.replace(
        "    simdgroup_barrier(mem_flags::mem_threadgroup);\n", ""
    )
    hlsl = _translate(tmp_path, source, "directx", "k.metal")

    assert _barrier_lines(hlsl) == ["DeviceMemoryBarrierWithGroupSync();"]


def test_metal_simdgroup_barrier_is_refused_for_hlsl(tmp_path):
    # HLSL has no wave-scope barrier, and a group barrier needs every thread of the group.
    with pytest.raises(Exception, match="subgroup execution barriers"):
        _translate(tmp_path, METAL_DEVICE_BARRIER, "directx", "k.metal")


def test_metal_imageblock_barrier_is_refused(tmp_path):
    source = """
#include <metal_stdlib>
using namespace metal;
kernel void k(device uint* d [[buffer(0)]], uint i [[thread_position_in_grid]]) {
    d[i] = i;
    threadgroup_barrier(mem_flags::mem_threadgroup_imageblock);
}
"""
    with pytest.raises(Exception, match="mem_threadgroup_imageblock"):
        _translate(tmp_path, source, "cgl", "k.metal")


GLSL_SUBGROUP_BARRIERS = """#version 450
#extension GL_KHR_shader_subgroup_basic : require
layout(local_size_x = 64) in;
layout(std430, binding = 0) buffer Data { uint d[]; };
shared uint s[64];
void main() {
    uint i = gl_LocalInvocationID.x;
    s[i] = d[i];
    subgroupMemoryBarrierShared();
    d[i] = s[i ^ 1u];
    subgroupMemoryBarrierBuffer();
    d[i] = d[i] + 1u;
    subgroupMemoryBarrierImage();
    d[i] = d[i] + 2u;
    subgroupMemoryBarrier();
    d[i] = d[i] + 3u;
    subgroupBarrier();
    d[i] = s[i];
}
"""


def test_glsl_subgroup_barriers_lower_to_subgroup_scope_barriers(tmp_path):
    metal = _translate(tmp_path, GLSL_SUBGROUP_BARRIERS, "metal", "s.comp")
    spirv = _translate(tmp_path, GLSL_SUBGROUP_BARRIERS, "vulkan", "s.comp")
    glsl = _translate(tmp_path, GLSL_SUBGROUP_BARRIERS, "opengl", "s.comp")

    all_flags = (
        "simdgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup | "
        "mem_flags::mem_texture);"
    )
    flat_metal = " ".join(metal.split())
    assert "simdgroup_barrier(mem_flags::mem_threadgroup);" in flat_metal
    assert "simdgroup_barrier(mem_flags::mem_device);" in flat_metal
    assert "simdgroup_barrier(mem_flags::mem_texture);" in flat_metal
    assert flat_metal.count(all_flags) == 2
    assert "subgroupMemoryBarrier" not in metal and "subgroupBarrier" not in metal
    for semantics in ("264", "72", "2056", "2376"):
        assert f"OpMemoryBarrier %uint_3 %uint_{semantics}" in spirv
    assert "OpControlBarrier %uint_3 %uint_3 %uint_2376" in spirv
    assert "GL_KHR_shader_subgroup_basic" in glsl
    assert _barrier_lines(glsl) == [
        "subgroupMemoryBarrierShared();",
        "subgroupMemoryBarrierBuffer();",
        "subgroupMemoryBarrierImage();",
        "subgroupMemoryBarrier();",
        "subgroupBarrier();",
    ]


def test_glsl_subgroup_fences_lower_to_hlsl_group_fences(tmp_path):
    source = GLSL_SUBGROUP_BARRIERS.replace("    subgroupBarrier();\n", "")
    hlsl = _translate(tmp_path, source, "directx", "s.comp")

    assert _barrier_lines(hlsl) == [
        "GroupMemoryBarrier();",
        "DeviceMemoryBarrier();",
        "DeviceMemoryBarrier();",
        "AllMemoryBarrier();",
    ]


@pytest.mark.parametrize(
    "lines,expected",
    [
        (
            [
                "    threadgroup_barrier(mem_flags::mem_device);",
                "    threadgroup_barrier(mem_flags::mem_threadgroup);",
                "    threadgroup_barrier(mem_flags::mem_device);",
            ],
            [
                "    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);"
            ],
        ),
        (
            [
                "    threadgroup_barrier(mem_flags::mem_none);",
                "    threadgroup_barrier(mem_flags::mem_none);",
            ],
            ["    threadgroup_barrier(mem_flags::mem_none);"],
        ),
        # A statement between the barriers, a different block or a different barrier kind keeps them apart.
        (
            [
                "    threadgroup_barrier(mem_flags::mem_device);",
                "    x = 1;",
                "    threadgroup_barrier(mem_flags::mem_device);",
            ],
            None,
        ),
        (
            [
                "    threadgroup_barrier(mem_flags::mem_device);",
                "        threadgroup_barrier(mem_flags::mem_device);",
            ],
            None,
        ),
        (
            [
                "    threadgroup_barrier(mem_flags::mem_device);",
                "    simdgroup_barrier(mem_flags::mem_device);",
            ],
            None,
        ),
    ],
)
def test_metal_adjacent_barriers_merge(lines, expected):
    code = "\n".join(["{", *lines, "}"])
    merged = merge_adjacent_metal_barriers(code)
    assert merged == ("\n".join(["{", *expected, "}"]) if expected else code)


@pytest.mark.parametrize(
    "lines,expected",
    [
        (
            [
                "    DeviceMemoryBarrier();",
                f"    {HLSL_EXECUTION_BARRIER_PLACEHOLDER}();",
                "    DeviceMemoryBarrier();",
            ],
            ["    DeviceMemoryBarrierWithGroupSync();"],
        ),
        (
            ["    GroupMemoryBarrier();", "    GroupMemoryBarrierWithGroupSync();"],
            ["    GroupMemoryBarrierWithGroupSync();"],
        ),
        (
            ["    DeviceMemoryBarrier();", "    GroupMemoryBarrier();"],
            ["    AllMemoryBarrier();"],
        ),
        (
            [f"    {HLSL_EXECUTION_BARRIER_PLACEHOLDER}();"],
            ["    GroupMemoryBarrierWithGroupSync();"],
        ),
        (
            ["    DeviceMemoryBarrier();", "    x = 1;", "    DeviceMemoryBarrier();"],
            ["    DeviceMemoryBarrier();", "    x = 1;", "    DeviceMemoryBarrier();"],
        ),
    ],
)
def test_hlsl_adjacent_barriers_merge(lines, expected):
    code = "\n".join(["{", *lines, "}"])
    assert merge_adjacent_hlsl_barriers(code) == "\n".join(["{", *expected, "}"])

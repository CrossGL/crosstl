"""GLSL output carries CrossGL ``atomicThreadFence`` with the GLSL memory barrier that covers it.

GLSL output lowered only ``atomicThreadFence(mem_threadgroup, memory_order_acq_rel, thread_scope_threadgroup)`` and
refused every other fence, so each Metal atomic function without ``_explicit`` (sequentially consistent, which the
Metal importer writes between two device-scope fences) was refused. See
:mod:`crosstl.translator.codegen.glsl_atomic_fences`.
"""

from __future__ import annotations

import shutil
import subprocess

import pytest

import crosstl
from crosstl.translator.codegen.glsl_atomic_fences import (
    GLSLAtomicFenceRefusal,
    glsl_atomic_fence_barrier,
)
from crosstl.translator.codegen.opengl_values import OpenGLAtomicFenceLoweringError

GLSLANG = shutil.which("glslangValidator")

ALL_METAL_MEMORY = frozenset(
    {
        "mem_device",
        "mem_threadgroup",
        "mem_threadgroup_imageblock",
        "mem_texture",
        "mem_object_data",
    }
)

METAL_ATOMICS = """#include <metal_stdlib>
using namespace metal;

kernel void probe(device int* values [[buffer(0)]],
                  device atomic_int* counters [[buffer(1)]],
                  device atomic_uint* flags [[buffer(2)]],
                  uint3 id [[thread_position_in_grid]]) {
    uint i = id.x;
    int operand = values[i];
    atomic_fetch_add(&counters[i], operand);
    int previous = atomic_exchange(&counters[i], operand + 1);
    int expected = operand;
    bool swapped = atomic_compare_exchange_weak(&counters[i], &expected, previous);
    atomic_fetch_or(&flags[i], 4u);
    values[i] = previous + expected + int(swapped);
}
"""

FENCE = (
    "atomicThreadFence(mem_device | mem_threadgroup | mem_threadgroup_imageblock | mem_texture | "
    "mem_object_data, memory_order_seq_cst, thread_scope_device);"
)


def _write(tmp_path, name, source):
    path = tmp_path / name
    path.write_text(source)
    return str(path)


def _glslang(tmp_path, glsl, stage, *arguments):
    if GLSLANG is None:
        pytest.skip("glslangValidator is not on PATH")
    path = tmp_path / f"translated.{stage}"
    path.write_text(glsl)
    result = subprocess.run(
        [GLSLANG, *arguments, str(path)], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stdout + result.stderr + glsl


def _cgl(body, stage="compute", globals_=""):
    layout = "layout(local_size_x = 1) in;" if stage == "compute" else ""
    signature = (
        "void main()"
        if stage == "compute"
        else "vec4 main(vec4 position @ gl_Position)"
    )
    result = "" if stage == "compute" else "return position;"
    return (
        f"shader Fences {{\n{globals_}\n    {stage} {{\n        {layout}\n"
        f"        {signature} {{\n            {body}\n            {result}\n        }}\n    }}\n}}\n"
    )


@pytest.mark.parametrize(
    ("flags", "scope", "stage", "expected"),
    [
        # The fences around a Metal atomic without _explicit.
        (ALL_METAL_MEMORY, "thread_scope_device", "compute", "memoryBarrier()"),
        ({"mem_device"}, "thread_scope_device", "compute", "memoryBarrierBuffer()"),
        ({"mem_texture"}, "thread_scope_device", "compute", "memoryBarrierImage()"),
        (
            {"mem_device", "mem_texture"},
            "thread_scope_device",
            "compute",
            "memoryBarrier()",
        ),
        # Shared memory is only visible to the workgroup, so a device-scope fence on it is a workgroup one.
        (
            {"mem_threadgroup"},
            "thread_scope_device",
            "compute",
            "memoryBarrierShared()",
        ),
        (
            {"mem_threadgroup"},
            "thread_scope_threadgroup",
            "compute",
            "memoryBarrierShared()",
        ),
        (
            {"mem_threadgroup", "mem_threadgroup_imageblock"},
            "thread_scope_simdgroup",
            "compute",
            "memoryBarrierShared()",
        ),
        ({"mem_none"}, "thread_scope_device", "compute", "memoryBarrierShared()"),
        (
            {"mem_device"},
            "thread_scope_threadgroup",
            "compute",
            "groupMemoryBarrier()",
        ),
        (ALL_METAL_MEMORY, "thread_scope_thread", "mesh", None),
        # Stages without workgroups have no shared memory and use the device-wide barriers.
        (
            {"mem_device"},
            "thread_scope_threadgroup",
            "fragment",
            "memoryBarrierBuffer()",
        ),
        (
            {"mem_device", "mem_threadgroup"},
            "thread_scope_device",
            "vertex",
            "memoryBarrierBuffer()",
        ),
        ({"mem_texture"}, "thread_scope_simdgroup", "fragment", "memoryBarrierImage()"),
        (ALL_METAL_MEMORY, "thread_scope_device", "fragment", "memoryBarrier()"),
        ({"mem_threadgroup"}, "thread_scope_device", None, "memoryBarrier()"),
    ],
)
def test_each_fence_becomes_the_narrowest_covering_barrier(
    flags, scope, stage, expected
):
    if expected is None:
        with pytest.raises(GLSLAtomicFenceRefusal) as refusal:
            glsl_atomic_fence_barrier(
                frozenset(flags), "memory_order_seq_cst", scope, stage
            )
        assert refusal.value.reason == "unsupported-object-data-memory"
        return
    for order in (
        "memory_order_relaxed",
        "memory_order_acq_rel",
        "memory_order_seq_cst",
    ):
        assert (
            glsl_atomic_fence_barrier(frozenset(flags), order, scope, stage) == expected
        )


def test_atomic_counter_memory_widens_a_device_memory_fence():
    assert (
        glsl_atomic_fence_barrier(
            frozenset({"mem_device"}),
            "memory_order_seq_cst",
            "thread_scope_device",
            "compute",
            has_atomic_counters=True,
        )
        == "memoryBarrier()"
    )


@pytest.mark.parametrize(
    ("flags", "scope", "stage", "reason"),
    [
        (
            {"mem_device"},
            "thread_scope_system",
            "compute",
            "unsupported-system-thread-scope",
        ),
        (
            {"mem_threadgroup"},
            "thread_scope_threadgroup",
            "fragment",
            "unsupported-shader-stage",
        ),
        (
            {"mem_threadgroup"},
            "thread_scope_simdgroup",
            None,
            "unsupported-shader-stage",
        ),
        (
            {"mem_object_data"},
            "thread_scope_device",
            "amplification",
            "unsupported-object-data-memory",
        ),
    ],
)
def test_fences_glsl_has_no_barrier_for_are_refused(flags, scope, stage, reason):
    with pytest.raises(GLSLAtomicFenceRefusal) as refusal:
        glsl_atomic_fence_barrier(
            frozenset(flags), "memory_order_acq_rel", scope, stage
        )
    assert refusal.value.reason == reason


def test_sequentially_consistent_metal_atomics_translate_between_device_barriers(
    tmp_path,
):
    path = _write(tmp_path, "atomics.metal", METAL_ATOMICS)
    assert crosstl.translate(path, backend="cgl").count(FENCE) == 8
    glsl = crosstl.translate(path, backend="opengl")
    assert "atomicThreadFence" not in glsl
    assert glsl.count("memoryBarrier();") == 8
    body = " ".join(glsl.split())
    assert "memoryBarrier(); atomicAdd(counters[i], operand); memoryBarrier();" in body
    _glslang(tmp_path, glsl, "comp")


def test_metal_atomics_translate_for_glsl_es(tmp_path):
    path = _write(tmp_path, "atomics.metal", METAL_ATOMICS)
    gles = crosstl.translate(path, backend="gles")
    assert gles.count("memoryBarrier();") == 8
    _glslang(tmp_path, gles, "comp")


@pytest.mark.parametrize(
    ("fence", "barrier"),
    [
        (
            "atomicThreadFence(mem_device, memory_order_release, thread_scope_threadgroup);",
            "groupMemoryBarrier();",
        ),
        (
            "atomicThreadFence(mem_texture, memory_order_acquire, thread_scope_device);",
            "memoryBarrierImage();",
        ),
        (
            "atomicThreadFence(mem_none, memory_order_relaxed, thread_scope_device);",
            "memoryBarrierShared();",
        ),
    ],
)
def test_compute_fences_validate(tmp_path, fence, barrier):
    path = _write(tmp_path, "fence.cgl", _cgl(fence))
    glsl = crosstl.translate(path, backend="opengl")
    assert barrier in glsl
    assert "atomicThreadFence" not in glsl
    _glslang(tmp_path, glsl, "comp")


def test_a_device_fence_in_a_vertex_stage_orders_buffer_memory(tmp_path):
    fence = "atomicThreadFence(mem_device | mem_threadgroup, memory_order_seq_cst, thread_scope_device);"
    path = _write(tmp_path, "fence.cgl", _cgl(fence, stage="vertex"))
    glsl = crosstl.translate(path, backend="opengl")
    assert "memoryBarrierBuffer();" in glsl
    assert "memoryBarrierShared" not in glsl
    _glslang(tmp_path, glsl, "vert")


def test_a_device_memory_fence_beside_atomic_counters_orders_them_too(tmp_path):
    counters = "    layout(binding = 0, offset = 0) uniform atomic_uint hits;\n"
    fence = (
        "atomicCounterIncrement(hits);\n"
        "            atomicThreadFence(mem_device, memory_order_seq_cst, thread_scope_device);"
    )
    path = _write(tmp_path, "fence.cgl", _cgl(fence, globals_=counters))
    glsl = crosstl.translate(path, backend="opengl")
    assert "memoryBarrier();" in glsl
    assert "memoryBarrierBuffer" not in glsl
    _glslang(tmp_path, glsl, "comp")


def test_a_system_scope_fence_is_still_refused(tmp_path):
    fence = "atomicThreadFence(mem_device, memory_order_seq_cst, thread_scope_system);"
    path = _write(tmp_path, "fence.cgl", _cgl(fence))
    with pytest.raises(OpenGLAtomicFenceLoweringError) as refusal:
        crosstl.translate(path, backend="opengl")
    assert refusal.value.reason == "unsupported-system-thread-scope"

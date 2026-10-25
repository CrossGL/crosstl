"""Metal ``atomic_fetch_sub_explicit`` imports as an atomic addition of the negated operand.

CrossGL, GLSL and HLSL have no atomic subtraction. Metal integer atomics wrap modulo 2^N (MSL 3.2 section
6.15.4.5), so ``atomicAdd(object, -operand)`` stores and returns the same values. The importer used to emit the
Metal call verbatim into CrossGL.
"""

import shutil
import subprocess

import pytest

import crosstl
from crosstl.backend.Metal.MetalCrossGLCodeGen import (
    MetalAtomicFetchSubLoweringError,
)

KERNEL = """#include <metal_stdlib>
using namespace metal;
struct Bin { uint indexCount; atomic_int instanceCount; };
kernel void k(device Bin* bins [[buffer(0)]],
              device atomic_uint* counters [[buffer(1)]],
              device int* out [[buffer(2)]],
              uint id [[thread_position_in_grid]]) {
    int slot = atomic_fetch_add_explicit(&(bins[id].instanceCount), 1, memory_order_relaxed);
    if (slot > 4) {
        atomic_fetch_sub_explicit(&(bins[id].instanceCount), 1, memory_order_relaxed);
    }
    uint before = atomic_fetch_sub_explicit(&counters[id], 2u, memory_order_relaxed);
    atomic_fetch_sub_explicit(&counters[id + 1], 1, memory_order_relaxed);
    out[id] = slot + int(before);
}
"""


def _translate(tmp_path, source, backend):
    path = tmp_path / "kernel.metal"
    path.write_text(source, encoding="utf-8")
    return crosstl.translate(str(path), backend=backend, format_output=False)


def _run(command):
    return subprocess.run(command, capture_output=True, text=True, check=False)


def test_atomic_fetch_sub_imports_as_atomic_add_of_the_negation(tmp_path):
    crossgl = _translate(tmp_path, KERNEL, "cgl")

    assert "atomic_fetch_sub_explicit" not in crossgl
    assert "atomicAdd(buffer_load(bins, id).instanceCount, -(1));" in crossgl
    assert "uint before = atomicAdd(counters[id], 0u - 2u);" in crossgl
    assert "atomicAdd(counters[id + 1], 0u - uint(1));" in crossgl


def test_atomic_fetch_sub_result_keeps_the_atomic_element_type(tmp_path):
    source = KERNEL.replace(
        "uint before = atomic_fetch_sub_explicit(&counters[id], 2u, memory_order_relaxed);",
        "auto before = atomic_fetch_sub_explicit(&counters[id], 2u, memory_order_relaxed);",
    )
    crossgl = _translate(tmp_path, source, "cgl")

    assert "uint before = atomicAdd(counters[id], 0u - 2u);" in crossgl


def test_atomic_fetch_sub_on_a_non_atomic_object_is_refused(tmp_path):
    source = """#include <metal_stdlib>
using namespace metal;
kernel void k(device int* values [[buffer(0)]], uint id [[thread_position_in_grid]]) {
    atomic_fetch_sub_explicit(&values[id], 1, memory_order_relaxed);
}
"""
    with pytest.raises(Exception) as raised:
        _translate(tmp_path, source, "cgl")
    error = raised.value
    while error is not None and not isinstance(error, MetalAtomicFetchSubLoweringError):
        error = error.__cause__ or error.__context__
    assert isinstance(error, MetalAtomicFetchSubLoweringError), raised.value
    assert "atomic_int, atomic_uint or atomic_float" in str(error)


def test_atomic_fetch_sub_translations_compile(tmp_path):
    source = KERNEL.replace(
        "struct Bin { uint indexCount; atomic_int instanceCount; };",
        "struct Bin { uint indexCount; int instanceCount; };",
    ).replace(
        "    int slot = atomic_fetch_add_explicit(&(bins[id].instanceCount), 1, memory_order_relaxed);\n"
        "    if (slot > 4) {\n"
        "        atomic_fetch_sub_explicit(&(bins[id].instanceCount), 1, memory_order_relaxed);\n"
        "    }\n",
        "    int slot = bins[id].instanceCount;\n",
    )

    hlsl = _translate(tmp_path, source, "directx")
    assert "InterlockedAdd(counters[id], (0u - 2u), before);" in hlsl

    xcrun = shutil.which("xcrun")
    if xcrun is not None:
        metal = tmp_path / "out.metal"
        metal.write_text(_translate(tmp_path, source, "metal"), encoding="utf-8")
        result = _run(
            [xcrun, "-sdk", "macosx", "metal", "-c", str(metal)]
            + ["-o", str(tmp_path / "out.air")]
        )
        if "unable to find utility" not in result.stderr:
            assert result.returncode == 0, result.stderr

    glslang = shutil.which("glslangValidator")
    if glslang is not None:
        glsl = tmp_path / "out.comp"
        glsl.write_text(_translate(tmp_path, source, "opengl"), encoding="utf-8")
        result = _run([glslang, "--target-env", "opengl", str(glsl)])
        assert result.returncode == 0, result.stdout + result.stderr

    spirv_as, spirv_val = shutil.which("spirv-as"), shutil.which("spirv-val")
    if spirv_as is not None and spirv_val is not None:
        assembly = tmp_path / "out.spvasm"
        assembly.write_text(_translate(tmp_path, source, "vulkan"), encoding="utf-8")
        binary = tmp_path / "out.spv"
        result = _run(
            [spirv_as, "--target-env", "vulkan1.1", str(assembly), "-o", str(binary)]
        )
        assert result.returncode == 0, result.stderr
        result = _run([spirv_val, "--target-env", "vulkan1.1", str(binary)])
        assert result.returncode == 0, result.stdout + result.stderr

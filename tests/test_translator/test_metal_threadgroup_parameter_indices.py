"""Metal threadgroup-memory parameters keep their ``[[threadgroup(index)]]`` slot.

The host sizes each threadgroup argument with ``setThreadgroupMemoryLength:atIndex:`` (Metal Shading Language
Specification 5.2.1), so the index is ABI. The Metal importer records it as ``@threadgroup(index)``; Metal output
used to drop it, which let the compiler renumber the slots in declaration order.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

import crosstl
from crosstl.translator.codegen.metal_refusals import MetalLoweringRefused
from crosstl.translator.semantics.binding_constants import BindingIndexUnresolvedError

ROOT = Path(__file__).resolve().parents[2]
REQUIRE_ENV = "CROSSTL_REQUIRE_THREADGROUP_BINDING_EXECUTION"

METAL_SOURCE = """\
#include <metal_stdlib>
using namespace metal;

kernel void reduce_pair(device const float* input [[buffer(0)]],
                        device float* output [[buffer(1)]],
                        threadgroup float* scratch [[threadgroup(0)]],
                        threadgroup float* other [[threadgroup(2)]],
                        uint lid [[thread_position_in_threadgroup]],
                        uint gid [[thread_position_in_grid]],
                        uint tgid [[threadgroup_position_in_grid]]) {
    scratch[lid] = input[gid];
    other[lid] = 1.0;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid == 0) {
        output[tgid] = scratch[0] + other[0];
    }
}
"""

CROSSGL_SOURCE = """\
shader main {
    compute {
        @ host_workgroup_size
        @ stage_entry
        void tile_copy(StructuredBuffer<float> input @buffer(0), RWStructuredBuffer<float> output @buffer(1),
                       threadgroup float* tile TILE_ATTRIBUTES, uint lid @gl_LocalInvocationID,
                       uint gid @gl_GlobalInvocationID) {
            tile[lid] = buffer_load(input, gid);
            workgroupBarrier();
            buffer_store(output, gid, tile[lid]);
        }
    }
}
"""


def translate(tmp_path, source, suffix):
    path = tmp_path / f"shader{suffix}"
    path.write_text(textwrap.dedent(source), encoding="utf-8")
    return crosstl.translate(str(path), backend="metal")


def crossgl(attributes):
    return CROSSGL_SOURCE.replace("TILE_ATTRIBUTES", attributes)


def kernel_parameters(metal):
    match = re.search(r"kernel void \w+\((.*?)\) \{", metal, re.S)
    assert match is not None, metal
    # clang-format, when installed, binds ``*`` to the name; without it the
    # emitter binds it to the type. Compare one canonical spelling.
    return re.sub(r"\s*\*\s*", " *", " ".join(match.group(1).split()))


def metal_compiler():
    xcrun = shutil.which("xcrun")
    if xcrun is None:
        return None
    probe = subprocess.run(
        [xcrun, "-sdk", "macosx", "-f", "metal"], capture_output=True, text=True
    )
    return xcrun if probe.returncode == 0 else None


def run(command):
    completed = subprocess.run(
        [str(part) for part in command], capture_output=True, text=True
    )
    assert completed.returncode == 0, completed.stderr[-2000:]
    return completed.stdout


def compile_library(tmp_path, xcrun, metal):
    shader = tmp_path / "threadgroup.metal"
    shader.write_text(metal, encoding="utf-8")
    air = tmp_path / "threadgroup.air"
    library = tmp_path / "threadgroup.metallib"
    run([xcrun, "-sdk", "macosx", "metal", "-Werror", "-c", shader, "-o", air])
    run([xcrun, "-sdk", "macosx", "metallib", air, "-o", library])
    return library


def test_metal_round_trip_keeps_each_threadgroup_index(tmp_path):
    parameters = kernel_parameters(translate(tmp_path, METAL_SOURCE, ".metal"))

    assert "threadgroup float *scratch [[threadgroup(0)]]" in parameters
    # A gap in the slots is kept rather than renumbered.
    assert "threadgroup float *other [[threadgroup(2)]]" in parameters


def test_crossgl_threadgroup_index_becomes_the_metal_attribute(tmp_path):
    parameters = kernel_parameters(
        translate(tmp_path, crossgl("@threadgroup(3)"), ".cgl")
    )

    assert "threadgroup float *tile [[threadgroup(3)]]" in parameters


def test_threadgroup_parameter_without_an_index_gets_no_attribute(tmp_path):
    parameters = kernel_parameters(translate(tmp_path, crossgl(""), ".cgl"))

    assert "threadgroup float *tile," in parameters
    assert "[[threadgroup(" not in parameters


def test_threadgroup_index_named_by_a_constant_is_folded(tmp_path):
    source = crossgl("@threadgroup(TILE_SLOT)").replace(
        "shader main {", "shader main {\n    const int TILE_SLOT = 4;", 1
    )
    parameters = kernel_parameters(translate(tmp_path, source, ".cgl"))

    assert "threadgroup float *tile [[threadgroup(4)]]" in parameters


def test_threadgroup_index_naming_no_constant_is_refused(tmp_path):
    with pytest.raises(BindingIndexUnresolvedError, match="@threadgroup\\(slot\\)"):
        translate(tmp_path, crossgl("@threadgroup(slot)"), ".cgl")


def test_negative_threadgroup_index_is_refused(tmp_path):
    with pytest.raises(
        MetalLoweringRefused, match="not one non-negative integer literal"
    ):
        translate(tmp_path, crossgl("@threadgroup(-1)"), ".cgl")


def test_two_threadgroup_indices_on_one_parameter_are_refused(tmp_path):
    with pytest.raises(MetalLoweringRefused, match="carries 2 \\[\\[threadgroup"):
        translate(tmp_path, crossgl("@threadgroup(0) @threadgroup(1)"), ".cgl")


def test_round_tripped_kernel_compiles(tmp_path):
    xcrun = metal_compiler()
    if xcrun is None:
        pytest.skip("the Metal toolchain is not available")
    compile_library(tmp_path, xcrun, translate(tmp_path, METAL_SOURCE, ".metal"))


def test_compiled_pipeline_reflects_the_source_threadgroup_slots(tmp_path):
    if os.environ.get(REQUIRE_ENV) != "1" or sys.platform != "darwin":
        pytest.skip(f"set {REQUIRE_ENV}=1 on macOS to require the Metal reflection")
    xcrun = metal_compiler()
    assert xcrun is not None, "the Metal toolchain is required"
    library = compile_library(
        tmp_path, xcrun, translate(tmp_path, METAL_SOURCE, ".metal")
    )
    runner = tmp_path / "bindings"
    run(
        [
            "swiftc",
            ROOT
            / "tests/fixtures/runtime_verification/metal_threadgroup_bindings.swift",
            "-o",
            runner,
        ]
    )

    assert run([runner, library, "reduce_pair"]).split("\n")[:-1] == [
        "other 2",
        "scratch 0",
    ]

"""CUDA and HIP input count ``__syncthreads_count``, ``__syncthreads_and`` and ``__syncthreads_or`` per block.

The importers refused the three votes because CrossGL has no barrier that reduces a predicate. Each now calls one
CrossGL function that counts the predicate in a workgroup variable between barriers (:mod:`crosstl.backend.CUDA.
block_votes`). The translations are compiled by each target's reference compiler where it is installed, and on a
Mac the Metal translation runs on the GPU for several block sizes and predicate patterns and is compared with the
CUDA definition of the votes.
"""

from __future__ import annotations

import json
import os
import random
import shutil
import struct
import subprocess
import sys
from pathlib import Path

import pytest

import crosstl
from crosstl.backend.import_refusals import CudaImportRefused, HipImportRefused
from tests.metal_compute_dispatch import compile_metal_library

ROOT = Path(__file__).resolve().parents[3]

REQUIRE_ENV = "CROSSTL_REQUIRE_METAL_BLOCK_VOTES"
RUNNER = ROOT / "tests/fixtures/runtime_verification/metal_raw_buffers.swift"

VOTES = """
__global__ void votes(const int* in, int* out) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    int p = in[i];
    int c = __syncthreads_count(p);
    int a = __syncthreads_and(p > 2);
    int o = __syncthreads_or(p);
    out[i * 3 + 0] = c;
    out[i * 3 + 1] = a;
    out[i * 3 + 2] = o;
}
"""

SOURCES = {
    "cuda": ("kernel.cu", VOTES),
    "hip": ("kernel.hip", "#include <hip/hip_runtime.h>\n" + VOTES),
}
REFUSED = {"cuda": CudaImportRefused, "hip": HipImportRefused}


def _translate(tmp_path, language, backend, source=None):
    name, default = SOURCES[language]
    path = tmp_path / name
    path.write_text(default if source is None else source, encoding="utf-8")
    return crosstl.translate(str(path), backend=backend, format_output=False)


@pytest.mark.parametrize("language", ["cuda", "hip"])
def test_the_votes_count_in_one_workgroup_variable(tmp_path, language):
    crossgl = _translate(tmp_path, language, "cgl")

    assert crossgl.count("var<workgroup> crossgl_block_votes: u32;") == 1
    assert crossgl.count("u32 crossgl_block_count(bool vote, u32 thread_index) {") == 1
    assert "i32(crossgl_block_count((p != 0), gl_LocalInvocationIndex))" in crossgl
    assert (
        "i32(crossgl_block_count(!(((p > 2) != 0)), gl_LocalInvocationIndex) == 0u)"
        in crossgl
    )
    assert (
        "i32(crossgl_block_count((p != 0), gl_LocalInvocationIndex) != 0u)" in crossgl
    )
    assert "__syncthreads_" not in crossgl


@pytest.mark.parametrize("language", ["cuda", "hip"])
def test_the_counter_function_orders_memory_like_syncthreads(tmp_path, language):
    crossgl = _translate(tmp_path, language, "cgl")
    body = crossgl.split("u32 crossgl_block_count(bool vote, u32 thread_index) {", 1)[1]
    lines = [
        line.strip() for line in body.split("\n}\n", 1)[0].splitlines() if line.strip()
    ]

    # The ordered barrier __syncthreads() imports as, the clear, a barrier, the atomic count, a barrier, the read.
    assert lines == [
        "groupMemoryBarrier();",
        "workgroupBarrier();",
        "groupMemoryBarrier();",
        "if (thread_index == 0u) {",
        "crossgl_block_votes = 0u;",
        "}",
        "workgroupBarrier();",
        "if (vote) {",
        "atomicAdd(crossgl_block_votes, 1u);",
        "}",
        "workgroupBarrier();",
        "return crossgl_block_votes;",
    ]


@pytest.mark.parametrize("language", ["cuda", "hip"])
def test_a_program_without_votes_declares_no_counter(tmp_path, language):
    source = SOURCES[language][1].split("__global__")[0] + (
        "__global__ void copy(const int* in, int* out) { out[threadIdx.x] = in[threadIdx.x]; }\n"
    )
    crossgl = _translate(tmp_path, language, "cgl", source)

    assert "crossgl_block" not in crossgl


@pytest.mark.parametrize("language", ["cuda", "hip"])
def test_the_names_avoid_the_program_identifiers(tmp_path, language):
    source = (
        SOURCES[language][1].replace("int* out", "int* vote").replace("out[", "vote[")
    )
    source = (
        "__device__ int crossgl_block_count(int v) { return v; }\n"
        + source.replace("int p = in[i];", "int p = crossgl_block_count(in[i]);")
    )
    crossgl = _translate(tmp_path, language, "cgl", source)

    assert "u32 crossgl_block_count_(bool vote_, u32 thread_index) {" in crossgl
    assert "var<workgroup> crossgl_block_votes: u32;" in crossgl


@pytest.mark.parametrize("language", ["cuda", "hip"])
def test_a_vote_with_two_arguments_is_refused(tmp_path, language):
    source = SOURCES[language][1].replace(
        "__syncthreads_count(p)", "__syncthreads_count(p, 1)"
    )

    with pytest.raises(REFUSED[language], match="takes one predicate"):
        _translate(tmp_path, language, "cgl", source)


@pytest.mark.parametrize("language", ["cuda", "hip"])
@pytest.mark.parametrize(
    "backend", ["cuda", "hip", "metal", "directx", "opengl", "vulkan"]
)
def test_every_target_translates_the_votes(tmp_path, language, backend):
    output = _translate(tmp_path, language, backend)

    assert "crossgl_block_votes" in output
    assert "__syncthreads_count" not in output


@pytest.mark.parametrize(
    "language, backend, tool, flags",
    [
        ("cuda", "cuda", "nvcc", ["-std=c++17", "-c"]),
        ("hip", "hip", "hipcc", ["--offload-arch=gfx1100", "-std=c++17", "-c"]),
    ],
)
def test_the_round_trip_compiles(tmp_path, language, backend, tool, flags):
    compiler = shutil.which(tool)
    if compiler is None:
        pytest.skip(f"{tool} is not installed")
    source = tmp_path / ("round_trip.cu" if backend == "cuda" else "round_trip.hip")
    source.write_text(_translate(tmp_path, language, backend), encoding="utf-8")
    result = subprocess.run(
        [compiler, *flags, str(source), "-o", str(tmp_path / "round_trip.o")],
        capture_output=True,
        text=True,
        timeout=900,
        check=False,
    )
    assert result.returncode == 0, result.stderr[-2000:]


def _predicates(group, groups, rng):
    values = []
    for index in range(groups):
        for thread in range(group):
            values.append(
                [
                    0,
                    5,
                    7 if thread == group - 1 else 0,
                    0 if thread == 0 else 3,
                    rng.choice((0, 1, 2, 3, -4)),
                    rng.randrange(-3, 9),
                ][index % 6]
            )
    return values


@pytest.mark.parametrize("language", ["cuda", "hip"])
def test_the_metal_translation_computes_the_votes(tmp_path, language):
    if sys.platform != "darwin" or shutil.which("swiftc") is None:
        if os.environ.get(REQUIRE_ENV) == "1":
            pytest.fail(f"{REQUIRE_ENV}=1 needs macOS with swiftc")
        pytest.skip("the Metal dispatch needs macOS with swiftc")
    metal = _translate(tmp_path, language, "metal")
    library = compile_metal_library(metal, tmp_path, REQUIRE_ENV)
    entry = "kernel_votes" if "kernel_votes(" in metal else "votes"
    runner = tmp_path / "runner"
    subprocess.run(["swiftc", str(RUNNER), "-o", str(runner)], check=True, timeout=600)
    rng = random.Random(4851)
    groups = 12
    for group in (32, 64, 256):
        values = _predicates(group, groups, rng)
        inputs, outputs = tmp_path / f"in-{group}.bin", tmp_path / f"out-{group}.bin"
        inputs.write_bytes(struct.pack(f"<{len(values)}i", *values))
        outputs.write_bytes(bytes(12 * len(values)))
        request = tmp_path / f"request-{group}.json"
        request.write_text(
            json.dumps(
                {
                    "buffers": [str(inputs), str(outputs)],
                    "workgroupCount": [groups, 1, 1],
                    "workgroupSize": [group, 1, 1],
                    "simdWidth": 32,
                }
            ),
            encoding="utf-8",
        )
        result_dir = tmp_path / f"result-{group}"
        subprocess.run(
            [str(runner), str(library), entry, str(request), str(result_dir)],
            check=True,
            capture_output=True,
            timeout=300,
        )
        got = struct.unpack(
            f"<{3 * len(values)}i",
            (result_dir / "buffer-1.bin").read_bytes()[: 12 * len(values)],
        )
        for index in range(groups):
            block = values[index * group : (index + 1) * group]
            want = (
                sum(value != 0 for value in block),
                int(all(value > 2 for value in block)),
                int(any(value != 0 for value in block)),
            )
            for thread in range(group):
                offset = 3 * (index * group + thread)
                assert got[offset : offset + 3] == want, (group, index, thread)

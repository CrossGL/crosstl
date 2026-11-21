"""Quad operations in compute thread groups whose quads are 2x2 blocks of threads (#4840).

HLSL Shader Model 6.6 forms the quads of a thread group with even x and y dimensions from 2x2 blocks of threads.
A Metal quad-group is four consecutive SIMD-group threads, so Metal output used to read ``gi ^ 2`` for
``QuadReadAcrossY`` in an 8x8 thread group where HLSL reads ``gi ^ 8``. ``metal_grid_quads`` remaps those reads
to SIMD-group shuffles, or refuses thread groups whose quads the shuffles cannot reach.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

import crosstl
from crosstl.translator.codegen.lowering_refusals import LoweringRefused
from crosstl.translator.codegen.metal_grid_quads import grid_quad_row_width
from crosstl.translator.refusals import TranslationRefusal

ROOT = Path(__file__).resolve().parents[2]
REQUIRE_ENV = "CROSTL_REQUIRE_METAL_GRID_QUADS"
OUTPUTS = 8

KERNEL = """\
RWStructuredBuffer<uint> fin : register(u0);
RWStructuredBuffer<uint> outp : register(u1);

uint second_row(uint v) {
    return QuadReadLaneAt(v, 2) + QuadReadAcrossY(v) * 3u;
}

[numthreads(@X@, @Y@, @Z@)]
void main(uint gi : SV_GroupIndex) {
    uint v = fin[gi];
    outp[gi * 8u + 0u] = QuadReadAcrossX(v);
    outp[gi * 8u + 1u] = QuadReadAcrossY(v);
    outp[gi * 8u + 2u] = QuadReadAcrossDiagonal(v);
    outp[gi * 8u + 3u] = QuadReadLaneAt(v, 0);
    outp[gi * 8u + 4u] = QuadReadLaneAt(v, 3);
    outp[gi * 8u + 5u] = (QuadAny(v % 7u == 0u) ? 1u : 0u) + (QuadAll(v % 3u != 0u) ? 2u : 0u);
    uint vote = 4u;
    if (v % 5u != 1u) {
        vote = (QuadAny(v % 4u == 0u) ? 1u : 0u) + (QuadAll(v % 2u == 0u) ? 2u : 0u);
    }
    outp[gi * 8u + 6u] = vote;
    outp[gi * 8u + 7u] = second_row(v);
}
"""


def _kernel(size):
    source = KERNEL
    for name, value in zip(("@X@", "@Y@", "@Z@"), size):
        source = source.replace(name, str(value))
    return source


def _translate(tmp_path, source, suffix=".hlsl", name="kernel"):
    path = tmp_path / f"{name}{suffix}"
    path.write_text(source, encoding="utf-8")
    return crosstl.translate(str(path), backend="metal")


@pytest.mark.parametrize(
    ("size", "width"),
    [
        ((8, 8, 1), 8),
        ((4, 4, 2), 4),
        ((16, 2, 1), 16),
        ((2, 8, 1), 2),
        ((64, 1, 1), 2),
        ((8, 1, 1), 2),
        ((7, 4, 1), 2),
        ((32, 2, 1), 0),
        ((6, 2, 1), 0),
        ((12, 4, 1), 0),
    ],
)
def test_row_width_follows_the_hlsl_quad_rule(size, width):
    assert grid_quad_row_width(size) == width


def test_quad_reads_of_a_2d_thread_group_use_simd_lanes(tmp_path):
    metal = _translate(tmp_path, _kernel((8, 8, 1)))
    # The thread at (x, y) reads (x ^ 1, y), (x, y ^ 1) and (x ^ 1, y ^ 1): lanes gi ^ 1, gi ^ 8 and gi ^ 9.
    assert "quad_shuffle_xor(v, ushort(1))" in metal
    assert "simd_shuffle_xor(v, ushort(8))" in metal
    assert "simd_shuffle_xor(v, ushort(9))" in metal
    assert "quad_shuffle_xor(v, ushort(2))" not in metal
    assert "quad_broadcast" not in metal
    assert "simd_shuffle(v, ushort((crossglWaveLaneIndex & ~9u) | 9u))" in metal
    assert "[[thread_index_in_simdgroup]]" in metal
    # The helper takes the lane from its caller.
    assert "uint second_row(uint v, uint crossglWaveLaneIndex)" in metal
    assert "simd_active_threads_mask()" in metal
    assert "quad_any" not in metal and "quad_all" not in metal


@pytest.mark.parametrize("size", [(64, 1, 1), (2, 8, 1), (7, 4, 1)])
def test_quads_metal_already_forms_keep_the_quad_lane_reads(tmp_path, size):
    metal = _translate(tmp_path, _kernel(size))
    assert "quad_shuffle_xor(v, ushort(2))" in metal
    assert "quad_broadcast(v, ushort(3))" in metal
    assert "simd_shuffle" not in metal
    # Votes count the active threads of the four consecutive lanes.
    assert "simd_vote::vote_t(15u) << (crossglWaveLaneIndex & ~3u)" in metal
    assert "quad_any(" not in metal and "quad_all(" not in metal


def test_crossgl_compute_stage_follows_the_same_rule(tmp_path):
    source = """
shader QuadRows {
    compute {
        layout(local_size_x = 4, local_size_y = 2, local_size_z = 1) in;
        buffer uint* fin;
        buffer uint* outp;
        void main() {
            uint i = gl_LocalInvocationIndex;
            outp[i] = QuadReadAcrossDiagonal(fin[i]);
        }
    }
}
"""
    metal = _translate(tmp_path, source, suffix=".cgl")
    assert "simd_shuffle_xor(" in metal and "ushort(5))" in metal


def test_metal_quad_functions_are_not_remapped(tmp_path):
    source = """#include <metal_stdlib>
using namespace metal;
kernel void k(device const int* fin [[buffer(0)]], device int* outp [[buffer(1)]],
              uint i [[thread_position_in_grid]]) {
    outp[i] = quad_shuffle_xor(fin[i], ushort(2)) + quad_broadcast(fin[i], ushort(1));
}
"""
    metal = _translate(tmp_path, source, suffix=".metal")
    assert "quad_shuffle_xor(" in metal and "quad_broadcast(" in metal
    assert "simd_shuffle" not in metal


@pytest.mark.parametrize("size", [(32, 2, 1), (6, 2, 1), (12, 4, 1)])
def test_unreachable_2d_quads_are_refused(tmp_path, size):
    with pytest.raises(TranslationRefusal, match="2x2 blocks of threads"):
        _translate(tmp_path, _kernel(size))


def test_runtime_quad_lane_is_refused_in_a_2d_thread_group(tmp_path):
    source = _kernel((8, 8, 1)).replace(
        "QuadReadLaneAt(v, 3)", "QuadReadLaneAt(v, v & 3u)"
    )
    with pytest.raises(LoweringRefused, match="constant quad lane index"):
        _translate(tmp_path, source)


def test_helper_shared_by_stages_with_different_quads_is_refused(tmp_path):
    source = """
shader SharedQuads {
    float swap_rows(float v) {
        return QuadReadAcrossY(v);
    }
    compute {
        layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;
        buffer float* data;
        void main() {
            uint i = gl_LocalInvocationIndex;
            data[i] = swap_rows(data[i]);
        }
    }
    fragment {
        input vec4 color @ TEXCOORD0;
        output vec4 result @ gl_FragColor;
        void main() {
            result = vec4(swap_rows(color.x));
        }
    }
}
"""
    with pytest.raises(TranslationRefusal, match="quads are formed differently"):
        _translate(tmp_path, source, suffix=".cgl")


def test_lane_reads_of_linear_quads_may_be_shared_with_a_fragment_stage(tmp_path):
    source = """
shader SharedQuads {
    float swap_columns(float v) {
        return QuadReadAcrossX(v);
    }
    compute {
        layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;
        buffer float* data;
        void main() {
            uint i = gl_LocalInvocationIndex;
            data[i] = swap_columns(data[i]);
        }
    }
    fragment {
        input vec4 color @ TEXCOORD0;
        output vec4 result @ gl_FragColor;
        void main() {
            result = vec4(swap_columns(color.x));
        }
    }
}
"""
    metal = _translate(tmp_path, source, suffix=".cgl")
    assert "quad_shuffle_xor(v, ushort(1))" in metal


def _run(command, cwd):
    return subprocess.run(
        command, cwd=cwd, check=True, capture_output=True, text=True, timeout=180
    ).stdout


def _compile(tmp_path, name, source):
    shader = tmp_path / f"{name}.metal"
    shader.write_text(source, encoding="utf-8")
    air, library = tmp_path / f"{name}.air", tmp_path / f"{name}.metallib"
    _run(
        ["xcrun", "-sdk", "macosx", "metal", "-c", str(shader), "-o", str(air)],
        tmp_path,
    )
    _run(
        ["xcrun", "-sdk", "macosx", "metallib", str(air), "-o", str(library)], tmp_path
    )
    return library


SIZES = [(8, 8, 1), (4, 4, 2), (16, 2, 1), (2, 8, 1), (12, 1, 1), (64, 1, 1)]


@pytest.mark.skipif(
    sys.platform != "darwin" or shutil.which("xcrun") is None,
    reason="the Metal compiler needs macOS and xcrun",
)
@pytest.mark.parametrize("size", SIZES)
def test_translated_kernels_compile(tmp_path, size):
    _compile(tmp_path, "kernel", _translate(tmp_path, _kernel(size)))


def _hlsl_quads(values, size):
    """``outp`` of ``KERNEL`` by the HLSL rule: 2x2 quads for even x and y, otherwise four consecutive indices."""
    x_size, y_size, _z_size = size
    count = len(values)

    def members(gi):
        if x_size % 2 == 0 and y_size % 2 == 0:
            x, y, z = gi % x_size, gi // x_size % y_size, gi // (x_size * y_size)
            return [
                (x & ~1) + (lane & 1) + x_size * ((y & ~1) + (lane >> 1) + y_size * z)
                for lane in range(4)
            ], (x & 1) | (y & 1) << 1
        return [(gi & ~3) + lane for lane in range(4)], gi & 3

    def votes(quad, any_test, all_test, active):
        live = [values[m] for m in quad if active(values[m])]
        return (1 if any(map(any_test, live)) else 0) + (
            2 if all(map(all_test, live)) else 0
        )

    expected = []
    for gi in range(count):
        quad, lane = members(gi)
        v = values[gi]
        diverged = v % 5 != 1
        expected += [
            values[quad[lane ^ 1]],
            values[quad[lane ^ 2]],
            values[quad[lane ^ 3]],
            values[quad[0]],
            values[quad[3]],
            votes(quad, lambda a: a % 7 == 0, lambda a: a % 3 != 0, lambda a: True),
            (
                votes(
                    quad,
                    lambda a: a % 4 == 0,
                    lambda a: a % 2 == 0,
                    lambda a: a % 5 != 1,
                )
                if diverged
                else 4
            ),
            (values[quad[2]] + values[quad[lane ^ 2]] * 3) & 0xFFFFFFFF,
        ]
    return expected


@pytest.mark.parametrize("size", SIZES)
def test_translated_quads_compute_the_hlsl_values_on_the_gpu(tmp_path, size):
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 to require native Metal execution")
    assert sys.platform == "darwin", "Native Metal execution requires macOS"
    count = size[0] * size[1] * size[2]
    values = [(index * 2654435761 >> 7) % 1000 for index in range(count)]
    runner = tmp_path / "readback"
    fixture = ROOT / "tests/fixtures/runtime_verification/metal_uint32_buffers.swift"
    _run(["xcrun", "swiftc", str(fixture), "-o", str(runner)], tmp_path)
    library = _compile(tmp_path, "kernel", _translate(tmp_path, _kernel(size)))
    request = tmp_path / "request.json"
    request.write_text(
        json.dumps(
            {
                "inputs": [values],
                "outputCount": count * OUTPUTS,
                "threadCount": 1,
                "threadgroupSize": list(size),
            }
        ),
        encoding="utf-8",
    )
    readback = json.loads(
        _run([str(runner), str(library), "kernel_main", str(request)], tmp_path)
    )
    assert readback["threadExecutionWidth"] == 32
    assert readback["values"] == _hlsl_quads(values, size)

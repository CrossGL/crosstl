"""Execute the CUDA and HIP wave helpers on a host model of a warp, with every lane pattern that matters.

No CUDA or HIP device runs in CI, so the helper bodies of :mod:`crosstl.translator.codegen.warp_wave_helpers` are
compiled as host C++ against a small SIMT model: each lane is a thread, ``__activemask()`` and ``__ballot(1)`` return
the lanes that call the helper, and a shuffle publishes every caller's value and reads the source lane's after all
callers have published, as ``__shfl_sync`` does. The results are compared with the HLSL definitions of the wave
intrinsics over the active lanes, including masks without lane 0 and single-lane masks.
"""

from __future__ import annotations

import random
import shutil
import subprocess

import pytest

from crosstl.translator.codegen.warp_wave_helpers import (
    CUDA_WARP,
    HIP_WAVEFRONT,
    wave_first_lane_test,
    wave_helper_body,
)

COMPILER = shutil.which("c++") or shutil.which("clang++") or shutil.which("g++")
pytestmark = pytest.mark.skipif(COMPILER is None, reason="no host C++ compiler")

MODEL = r"""
#include <condition_variable>
#include <cstdio>
#include <mutex>
#include <thread>
#include <vector>
#define __device__
typedef unsigned int uint;
struct uint4 { uint x, y, z, w; };
static uint4 make_uint4(uint x, uint y, uint z, uint w) { return {x, y, z, w}; }
static int warpSize = WIDTH;
typedef unsigned long long mask_t;
static mask_t g_active;
static thread_local int t_lane;
static std::mutex g_mutex;
static std::condition_variable g_cv;
static int g_arrived = 0;
static unsigned long g_generation = 0;
static unsigned long long g_slots[64];

static void model_barrier() {
    std::unique_lock<std::mutex> lock(g_mutex);
    unsigned long generation = g_generation;
    if (++g_arrived == __builtin_popcountll(g_active)) {
        g_arrived = 0;
        ++g_generation;
        g_cv.notify_all();
    } else {
        g_cv.wait(lock, [&] { return g_generation != generation; });
    }
}
template <typename T> static T exchange(T value, int source) {
    g_slots[t_lane] = (unsigned long long)value;
    model_barrier();
    unsigned long long read = g_slots[source];
    model_barrier();
    if (!((g_active >> source) & 1ull)) { std::printf("READ-INACTIVE %d\n", source); }
    return (T)read;
}
static mask_t model_ballot(bool predicate) {
    g_slots[t_lane] = predicate ? 1ull : 0ull;
    model_barrier();
    mask_t bits = 0;
    for (int lane = 0; lane < WIDTH; ++lane) {
        if (((g_active >> lane) & 1ull) && g_slots[lane]) { bits |= 1ull << lane; }
    }
    model_barrier();
    return bits;
}
// CUDA
static unsigned int __activemask() { return (unsigned int)g_active; }
template <typename T> static T __shfl_sync(unsigned int, T value, int source) { return exchange(value, source); }
template <typename T> static T __shfl_xor_sync(unsigned int, T value, int m) { return exchange(value, t_lane ^ m); }
static unsigned int __ballot_sync(unsigned int, bool p) { return (unsigned int)model_ballot(p); }
static int __all_sync(unsigned int, bool p) { return model_ballot(p) == g_active; }
static int __any_sync(unsigned int, bool p) { return model_ballot(p) != 0; }
static int __ffs(int x) { return __builtin_ffs(x); }
static int __popc(unsigned int x) { return __builtin_popcount(x); }
// HIP
static mask_t __ballot(int p) { return model_ballot(p != 0); }
template <typename T> static T __shfl(T value, int source) { return exchange(value, source); }
template <typename T> static T __shfl_xor(T value, int m) { return exchange(value, t_lane ^ m); }
static int __all(int p) { return model_ballot(p != 0) == g_active; }
static int __any(int p) { return model_ballot(p != 0) != 0; }
static int __ffsll(unsigned long long x) { return __builtin_ffsll((long long)x); }
static int __popcll(unsigned long long x) { return __builtin_popcountll(x); }
#define LANE ((uint)t_lane)
"""

# (operation, result type, value type, parameters, call)
OPERATIONS = [
    ("WaveActiveSum", "uint", "uint", "uint value", "(uint)(lane * 3 + 1)"),
    ("WaveActiveProduct", "uint", "uint", "uint value", "(uint)(lane % 3 + 1)"),
    ("WaveActiveMin", "int", "int", "int value", "(int)(lane * 7 % 11) - 5"),
    ("WaveActiveMax", "int", "int", "int value", "(int)(lane * 7 % 11) - 5"),
    ("WaveActiveBitXor", "uint", "uint", "uint value", "(uint)(lane * 2654435761u)"),
    ("WavePrefixSum", "uint", "uint", "uint value", "(uint)(lane + 1)"),
    ("WaveActiveAllEqual", "bool", "uint", "uint value", "(uint)(lane / 64)"),
    ("WaveActiveCountBits", "uint", "bool", "bool predicate", "lane % 3 == 0"),
    ("WavePrefixCountBits", "uint", "bool", "bool predicate", "lane % 3 == 0"),
    ("WaveReadLaneFirst", "uint", "uint", "uint value", "(uint)(lane + 100)"),
    ("WaveMatch", "uint4", "uint", "uint value", "(uint)(lane % 4)"),
    ("WaveActiveBallot", "uint4", "bool", "bool predicate", "lane % 2 == 1"),
]


def _program(dialect, width):
    helpers, calls = [], []
    for index, (operation, result, value_type, parameter, argument) in enumerate(
        OPERATIONS
    ):
        body = wave_helper_body(dialect, operation, result, value_type, "LANE")
        assert body is not None, operation
        helpers.append(f"static {result} h{index}({parameter}) {{\n{body}}}\n")
        low, high = ("r.x", "r.y") if result == "uint4" else ("r", "0")
        calls.append(
            f"    {{ {result} r = h{index}({argument}); "
            f'std::printf("%d %d %llu %llu\\n", {index}, lane, '
            f"(unsigned long long)({low}), (unsigned long long)({high})); }}\n"
        )
    first = wave_first_lane_test(dialect, "LANE")
    calls.append(
        f'    std::printf("%d %d %d 0\\n", {len(OPERATIONS)}, lane, (int){first});\n'
    )
    main = (
        "static std::mutex g_print;\n"
        "static void run_lane(int lane) {\n"
        "    t_lane = lane;\n" + "".join(calls) + "}\n"
        "int main(int argc, char **argv) {\n"
        "    g_active = std::stoull(argv[1]);\n"
        "    std::vector<std::thread> threads;\n"
        f"    for (int lane = 0; lane < {width}; ++lane) {{\n"
        "        if ((g_active >> lane) & 1ull) { threads.emplace_back(run_lane, lane); }\n"
        "    }\n"
        "    for (auto &thread : threads) { thread.join(); }\n"
        "}\n"
    )
    return (
        "#include <string>\n"
        + MODEL.replace("WIDTH", str(width))
        + "".join(helpers)
        + main
    )


def _expected(index, lane, lanes):
    operation = OPERATIONS[index][0] if index < len(OPERATIONS) else "first"
    below = [other for other in lanes if other < lane]
    if operation == "WaveActiveSum":
        return sum(other * 3 + 1 for other in lanes) % 2**32, 0
    if operation == "WaveActiveProduct":
        product = 1
        for other in lanes:
            product = product * (other % 3 + 1) % 2**32
        return product, 0
    if operation in {"WaveActiveMin", "WaveActiveMax"}:
        values = [other * 7 % 11 - 5 for other in lanes]
        value = min(values) if operation == "WaveActiveMin" else max(values)
        return value % 2**64, 0
    if operation == "WaveActiveBitXor":
        value = 0
        for other in lanes:
            value ^= other * 2654435761 % 2**32
        return value, 0
    if operation == "WavePrefixSum":
        return sum(other + 1 for other in below), 0
    if operation == "WaveActiveAllEqual":
        return 1, 0
    if operation == "WaveActiveCountBits":
        return sum(other % 3 == 0 for other in lanes), 0
    if operation == "WavePrefixCountBits":
        return sum(other % 3 == 0 for other in below), 0
    if operation == "WaveReadLaneFirst":
        return min(lanes) + 100, 0
    if operation in {"WaveMatch", "WaveActiveBallot"}:
        if operation == "WaveMatch":
            bits = sum(1 << other for other in lanes if other % 4 == lane % 4)
        else:
            bits = sum(1 << other for other in lanes if other % 2 == 1)
        return bits & 0xFFFFFFFF, bits >> 32
    return int(lane == min(lanes)), 0


@pytest.mark.parametrize(
    "dialect, width", [(CUDA_WARP, 32), (HIP_WAVEFRONT, 64)], ids=["cuda", "hip"]
)
def test_wave_helpers_combine_exactly_the_active_lanes(dialect, width, tmp_path):
    source = tmp_path / "model.cpp"
    binary = tmp_path / "model"
    source.write_text(_program(dialect, width))
    subprocess.run(
        [COMPILER, "-std=c++17", "-O1", "-pthread", str(source), "-o", str(binary)],
        check=True,
        capture_output=True,
        timeout=300,
    )
    rng = random.Random(14)
    full = (1 << width) - 1
    masks = [
        full,
        sum(1 << lane for lane in range(0, width, 2)),
        sum(1 << lane for lane in range(1, width, 2)),
        1 << (width - 1),
        1 << 5,
        full & ~1,
        rng.getrandbits(width) | (1 << 9),
    ]
    for mask in masks:
        lanes = [lane for lane in range(width) if (mask >> lane) & 1]
        result = subprocess.run(
            [str(binary), str(mask)],
            check=True,
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert "READ-INACTIVE" not in result.stdout, hex(mask)
        rows = [line.split() for line in result.stdout.splitlines()]
        assert len(rows) == len(lanes) * (len(OPERATIONS) + 1)
        for index, lane, low, high in (
            (int(a), int(b), int(c), int(d)) for a, b, c, d in rows
        ):
            name = OPERATIONS[index][0] if index < len(OPERATIONS) else "first lane"
            assert (low, high) == _expected(index, lane, lanes), (name, lane, hex(mask))

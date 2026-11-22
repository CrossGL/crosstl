"""HIP output of CUDA warp intrinsics (``crosstl.translator.codegen.hip_cuda_warps``).

CrossGL keeps CUDA's warp intrinsics under their CUDA names. HIP output computes each over the caller's 32-lane
segment of a 32- or 64-lane wavefront. These tests pin the emitted helpers, check that the shuffles they call pick
the source lane PTX ``shfl.sync`` picks on both wavefront sizes, and compile the output with hipcc when it is
installed.
"""

from __future__ import annotations

import re
import shutil
import subprocess

import pytest

import crosstl
from crosstl.translator.codegen.hip_cuda_warps import CUDA_WARP_INTRINSIC_ARITIES

KERNEL = """
__global__ void warps(const float* in, const double* wide, float* out, unsigned* bits, int* flags) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    float v = in[i];
    v += __shfl_sync(0xffffffffu, v, 3);
    v += __shfl_up_sync(0xffffffffu, v, 2, 16);
    v += __shfl_down_sync(0xffffffffu, v, 4);
    v += __shfl_xor_sync(0xffffffffu, v, 1, 8);
    double w = __shfl_xor_sync(0xffffffffu, wide[i], 5);
    int lane = (int)(i & 31u);
    int pred = 0;
    unsigned active = __activemask();
    bits[i] = __ballot_sync(active, v > 1.0f) + __match_any_sync(active, lane / 4)
        + __match_all_sync(0xffffffffu, lane, &pred);
    flags[i] = __any_sync(0xffffffffu, v > 2.0f) + __all_sync(0xffffffffu, v > 0.0f)
        + __uni_sync(0xffffffffu, lane > 7) + pred;
    unsigned int u = (unsigned int)lane;
    bits[i] += __reduce_add_sync(0xffffffffu, u) + __reduce_and_sync(0xffffffffu, u)
        + __reduce_or_sync(0xffffffffu, u) + __reduce_xor_sync(0xffffffffu, u);
    flags[i] += __reduce_min_sync(0xffffffffu, lane) + __reduce_max_sync(0xffffffffu, lane);
    __syncwarp();
    out[i] = v + (float)w;
}
"""


@pytest.fixture(scope="module")
def hip_source(tmp_path_factory):
    path = tmp_path_factory.mktemp("hip_cuda_warps") / "warps.cu"
    path.write_text(KERNEL, encoding="utf-8")
    return crosstl.translate(str(path), backend="hip")


def _kernel_body(hip):
    return hip[hip.index("__global__ void warps") :]


def test_every_cuda_warp_intrinsic_becomes_a_hip_helper(hip_source):
    body = _kernel_body(hip_source)
    for name in CUDA_WARP_INTRINSIC_ARITIES:
        assert not re.search(rf"\b{name}\(", body), name
    for helper in (
        "cgl_cuda_shfl_sync",
        "cgl_cuda_shfl_up_sync",
        "cgl_cuda_shfl_down_sync",
        "cgl_cuda_shfl_xor_sync",
        "cgl_cuda_activemask",
        "cgl_cuda_ballot_sync",
        "cgl_cuda_match_any_sync",
        "cgl_cuda_match_all_sync",
        "cgl_cuda_any_sync",
        "cgl_cuda_all_sync",
        "cgl_cuda_uni_sync",
        "cgl_cuda_reduce_add_sync",
        "cgl_cuda_reduce_and_sync",
        "cgl_cuda_reduce_or_sync",
        "cgl_cuda_reduce_xor_sync",
        "cgl_cuda_reduce_min_sync",
        "cgl_cuda_reduce_max_sync",
        "cgl_cuda_syncwarp",
    ):
        assert f"{helper}(" in body, helper
        definitions = re.findall(rf"\b{helper}\([^;{{]*\)\s*{{", hip_source)
        assert len(definitions) == 1, helper
    assert "HIP_ENABLE_WARP_SYNC_BUILTINS" not in hip_source
    assert "cgl_cuda_match_all_sync(4294967295u, lane, &pred)" in body


def test_helpers_are_defined_before_the_helpers_that_call_them(hip_source):
    def definition(name):
        match = re.search(rf"\b{name}\([^;{{]*\)\s*{{", hip_source)
        assert match, name
        return match.start()

    assert definition("cgl_cuda_warp_bits") < definition("cgl_cuda_ballot_sync")
    assert definition("cgl_cuda_ballot_sync") < definition("cgl_cuda_any_sync")
    assert definition("cgl_cuda_warp_value_bits") < definition(
        "cgl_cuda_match_any_sync"
    )
    assert definition("cgl_cuda_match_any_sync") < definition("cgl_cuda_match_all_sync")


def test_votes_take_the_callers_32_bits_of_the_wavefront_ballot(hip_source):
    assert "return static_cast<unsigned int>(wave_bits >> (__lane_id() & ~31u));" in (
        hip_source
    )
    assert "return cgl_cuda_warp_bits(__ballot(predicate)) & mask;" in hip_source
    assert (
        "return cgl_cuda_ballot_sync(mask, predicate) == cgl_cuda_ballot_sync(mask, 1);"
        in hip_source
    )
    # __syncwarp orders the wavefront's memory; its lanes execute together.
    assert '__builtin_amdgcn_fence(__ATOMIC_RELEASE, "wavefront");' in hip_source
    assert '__builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "wavefront");' in hip_source


# PTX shfl.sync (PTX ISA, "Data Movement and Conversion Instructions: shfl.sync"), with the c operand CUDA's
# sm_30_intrinsics.hpp passes: ((32 - width) << 8) | 0x1f, and (32 - width) << 8 for the up mode.
def _ptx_source_lane(mode, lane, b, width):
    segmask = (32 - width) & 31
    cval = 0 if mode == "up" else 0x1F
    bval = b & 31
    max_lane = (lane & segmask) | (cval & ~segmask & 31)
    min_lane = lane & segmask
    if mode == "up":
        j, valid = lane - bval, lane - bval >= max_lane
    elif mode == "down":
        j, valid = lane + bval, lane + bval <= max_lane
    elif mode == "bfly":
        j, valid = lane ^ bval, (lane ^ bval) <= max_lane
    else:
        j = min_lane | (bval & ~segmask & 31)
        valid = j <= max_lane
    return j if valid else lane


# HIP's wavefront shuffles (hip/amd_detail/amd_warp_functions.h, ROCm 5.7), read through ds_bpermute, whose
# address selects a lane modulo the wavefront size.
def _hip_source_lane(function, self, operand, width, wave):
    if function == "__shfl":
        index = (operand & (width - 1)) + (self & ~(width - 1))
    elif function == "__shfl_up":
        index = self - operand
        index = self if index < (self & ~(width - 1)) else index
    elif function == "__shfl_down":
        index = self + operand
        index = self if (self & (width - 1)) + operand >= width else index
    else:
        index = self ^ operand
        index = self if index >= ((self + width) & ~(width - 1)) else index
    return index % wave


_MODES = {
    "cgl_cuda_shfl_sync": "idx",
    "cgl_cuda_shfl_up_sync": "up",
    "cgl_cuda_shfl_down_sync": "down",
    "cgl_cuda_shfl_xor_sync": "bfly",
}


@pytest.mark.parametrize("helper", sorted(_MODES))
def test_shuffle_helpers_read_the_lane_ptx_shfl_sync_reads(hip_source, helper):
    match = re.search(
        rf"{helper}\(unsigned int mask, T value,\s*(?:unsigned )?int lane,\s*int width = 32\) {{\s*"
        r"\(void\)mask;\s*return (__shfl\w*)\(value, lane & (31u?), width\);",
        hip_source,
    )
    assert match, helper
    function, lane_mask = match.group(1), int(match.group(2).rstrip("u"))
    mode = _MODES[helper]
    for wave in (32, 64):
        for width in (1, 2, 4, 8, 16, 32):
            for b in range(80):
                for self in range(wave):
                    base = self & ~31
                    expected = base + _ptx_source_lane(mode, self & 31, b, width)
                    actual = _hip_source_lane(
                        function, self, b & lane_mask, width, wave
                    )
                    assert actual == expected, (wave, width, b, self)


@pytest.mark.parametrize("arch", ["gfx1100", "gfx90a"])
def test_hip_output_compiles_for_32_and_64_lane_wavefronts(hip_source, tmp_path, arch):
    hipcc = shutil.which("hipcc")
    if hipcc is None:
        pytest.skip("hipcc is not installed")
    source = tmp_path / "warps.hip"
    source.write_text(hip_source, encoding="utf-8")
    result = subprocess.run(
        [
            hipcc,
            "-std=c++17",
            f"--offload-arch={arch}",
            "-c",
            str(source),
            "-o",
            str(tmp_path / "warps.o"),
        ],
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )
    assert result.returncode == 0, (result.stdout + result.stderr)[-4000:]

"""CUDA warp votes, reductions, matches and ``__syncwarp`` on targets without CUDA warps (S29).

A warp is 32 threads of consecutive linear index; a vote, reduction or match combines the lanes of the caller's warp
(CUDA C++ Programming Guide, "Warp Vote Functions", "Warp Reduce Functions", "Warp Match Functions"). Wave and
subgroup operations act on a subgroup of implementation-defined size instead (4 or 8 lanes on lavapipe, 4 on WARP),
so the legalizer folds each warp's slots of a workgroup exchange. Execution evidence is the ``x_warp_vote``
conformance case; these tests check the emitted shape, the refusals and, where the tools are installed, validity.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

import crosstl
from crosstl.translator.semantics.warp import CudaWarpIntrinsicUnsupportedError

CASE = Path(__file__).parents[1] / "conformance" / "cases" / "x_warp_vote" / "src.cgl"
EMULATING = ("metal", "directx", "opengl", "vulkan", "slang", "wgsl")
WAVE_OPERATIONS = (
    "Wave",
    "subgroup",
    "simd_",
    "OpGroupNonUniform",
    "__ballot",
    "__any",
    "__all",
    "__reduce",
    "__match",
    "__syncwarp",
)


def program(body, lanes=32):
    return f"""
shader WarpVotes {{
    compute probe {{
        layout(local_size_x = {lanes}, local_size_y = 1, local_size_z = 1) in;
        uniform int count;
        buffer int* data;
        void main() {{
            uint i = gl_GlobalInvocationID.x;
            int v = data[i];
            {body}
            data[i] = v;
        }}
    }}
}}
"""


def translate(tmp_path, source, backend, **options):
    path = tmp_path / "probe.cgl"
    path.write_text(source)
    return crosstl.translate(str(path), backend=backend, format_output=False, **options)


def refusal(tmp_path, source, backend="metal"):
    with pytest.raises(CudaWarpIntrinsicUnsupportedError) as raised:
        translate(tmp_path, source, backend)
    assert (
        raised.value.code
        == f"project.translate.{backend}-cuda-warp-intrinsic-unsupported"
    )
    return raised.value.reason_code


@pytest.mark.parametrize("backend", EMULATING)
def test_votes_reductions_and_matches_fold_each_warp(tmp_path, backend):
    generated = translate(tmp_path, CASE.read_text(), backend)
    for operation in WAVE_OPERATIONS:
        assert operation not in generated, operation
    assert "cgl_warp_exchange_0" in generated
    assert "cgl_warp_result_9" in generated


def _run(command):
    completed = subprocess.run(command, capture_output=True, text=True)
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_emulation_passes_the_native_checkers(tmp_path):
    source = CASE.read_text()
    checked = 0
    if shutil.which("glslangValidator"):
        path = tmp_path / "vote.comp"
        path.write_text(translate(tmp_path, source, "opengl"))
        _run(["glslangValidator", str(path)])
        checked += 1
    if shutil.which("spirv-as") and shutil.which("spirv-val"):
        text, binary = tmp_path / "vote.spvasm", tmp_path / "vote.spv"
        text.write_text(translate(tmp_path, source, "vulkan"))
        _run(["spirv-as", "--target-env", "vulkan1.1", str(text), "-o", str(binary)])
        _run(["spirv-val", "--target-env", "vulkan1.1", str(binary)])
        checked += 1
    if shutil.which("naga"):
        path = tmp_path / "vote.wgsl"
        path.write_text(translate(tmp_path, source, "wgsl"))
        _run(["naga", str(path)])
        checked += 1
    if not checked:
        pytest.skip("no GLSL, SPIR-V or WGSL checker is installed")


def test_partial_last_warp_folds_only_its_lanes(tmp_path):
    body = "v = __reduce_min_sync(0xFFFFFFFF, v);"
    generated = translate(tmp_path, program(body, 48), "opengl")
    # Warp k is local indices [32 k, 32 k + 32); the second warp of 48 invocations has 16 lanes.
    assert "(gl_LocalInvocationIndex - (gl_LocalInvocationIndex & 31u))" in generated
    assert "min(32u, (48u - cgl_warp_lane_0))" in generated
    assert (
        "min(cgl_warp_result_0, cgl_warp_exchange_0[(cgl_warp_lane_0 + cgl_warp_lane_1)])"
        in generated
    )


def test_one_warp_workgroup_folds_its_invocations(tmp_path):
    body = "v = int(__ballot_sync(0x0000FFFF, v > 0));"
    generated = translate(tmp_path, program(body, 16), "opengl")
    # A 16-invocation workgroup is one warp of 16 lanes; a mask naming those lanes reaches every invocation.
    assert "(cgl_warp_lane_0 < 16u)" in generated
    assert "(cgl_warp_exchange_0[cgl_warp_lane_0] << cgl_warp_lane_0)" in generated


def test_signed_sum_wraps_through_unsigned_arithmetic(tmp_path):
    generated = translate(
        tmp_path, program("v = __reduce_add_sync(0xFFFFFFFF, v);"), "metal"
    )
    # CUDA's sum wraps modulo 2^32; MSL leaves signed overflow undefined, so the fold adds unsigned values.
    assert "uint cgl_warp_value_0 = uint(v);" in generated
    assert "int(cgl_warp_result_0)" in generated


def test_syncwarp_is_a_workgroup_barrier_at_uniform_points(tmp_path):
    body = "data[i] = v; __syncwarp(); v = data[i ^ 1u];"
    for lanes in (32, 64):
        generated = translate(tmp_path, program(body, lanes), "opengl")
        assert "__syncwarp" not in generated
        assert "subgroupBarrier" not in generated
        assert generated.count("barrier();") == 1
        # __syncwarp orders the lanes' buffer accesses too, which barrier() alone does not.
        assert re.search(
            r"groupMemoryBarrier\(\);\s*barrier\(\);\s*groupMemoryBarrier\(\);",
            generated,
        )


def test_a_vote_result_is_not_workgroup_uniform(tmp_path):
    # The two warps of 48 invocations see different ballots (32 and 16 lanes), so a shuffle under it may be reached
    # by one warp only.
    body = """
            uint bits = __ballot_sync(0xFFFFFFFF, count > 0);
            if (bits == 4294967295u) { v = __shfl_sync(0xFFFFFFFF, v, 0); }
"""
    assert refusal(tmp_path, program(body, 48)) == "non-uniform-control-flow"


def test_syncwarp_requires_uniform_control_flow_for_every_warp_intrinsic(tmp_path):
    # CUDA lets a warp's lanes meet at different __syncwarp calls, which no workgroup barrier can express.
    body = "if (v > 0) { v = __any_sync(0xFFFFFFFF, v > 3); } __syncwarp();"
    assert refusal(tmp_path, program(body)) == "non-uniform-control-flow"
    without = "if (v > 0) { v = __any_sync(0xFFFFFFFF, v > 3); }"
    assert "cgl_warp_exchange_0" in translate(tmp_path, program(without), "metal")


def test_locals_that_share_a_name_are_told_apart(tmp_path):
    # The first j differs between invocations; the second loop's j is a different, uniform variable.
    body = """
            for (int j = int(i); j < count; j += 64) { v += 1; }
            for (int j = 16; j > 0; j >>= 1) { v += __shfl_down_sync(0xFFFFFFFF, v, 1); }
"""
    generated = translate(tmp_path, program(body, 64), "opengl")
    assert "__shfl" not in generated
    shadowed = """
            int k = 4;
            if (count > 0) { int k = int(i); v += k; }
            for (int j = 0; j < k; j++) { v += __shfl_down_sync(0xFFFFFFFF, v, 1); }
"""
    assert "__shfl" not in translate(tmp_path, program(shadowed, 64), "opengl")
    reused = """
            int k = 4;
            if (count > 0) { k = int(i); }
            for (int j = 0; j < k; j++) { v += __shfl_down_sync(0xFFFFFFFF, v, 1); }
"""
    assert refusal(tmp_path, program(reused, 64)) == "non-uniform-control-flow"


def test_invocations_reconverge_after_a_divergent_loop(tmp_path):
    body = """
            for (uint j = i; j < uint(count); j += 64u) { if (v > 3) { break; } v += 1; }
            v = __reduce_add_sync(0xFFFFFFFF, v);
"""
    for backend in ("metal", "opengl", "directx", "vulkan"):
        assert "cgl_warp_result" in translate(tmp_path, program(body, 64), backend)
    # WGSL's uniformity analysis (section 15.2) is kept conservative after a loop with a divergent exit.
    assert refusal(tmp_path, program(body, 64), "wgsl") == "non-uniform-control-flow"


def test_a_divergent_return_in_a_loop_still_taints_what_follows(tmp_path):
    body = """
            for (uint j = i; j < uint(count); j += 64u) { if (v > 3) { return; } }
            v = __reduce_add_sync(0xFFFFFFFF, v);
"""
    assert refusal(tmp_path, program(body, 64)) == "non-uniform-control-flow"


def test_a_divergent_break_reaches_the_next_iteration(tmp_path):
    # Invocations that broke out in one iteration do not reach the next iteration's reduction.
    body = """
            for (int j = 0; j < 4; j++) { v = __reduce_add_sync(0xFFFFFFFF, v); if (v > 3) { break; } }
"""
    assert refusal(tmp_path, program(body, 64)) == "non-uniform-control-flow"


CUDA_VOTES = """
__global__ void votes(const int *inp, int *outp) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    int v = inp[i];
    unsigned int full = 0xffffffffu;
    int total = __reduce_add_sync(0xffffffffu, v);
    unsigned int positive = __ballot_sync(full, v > 0);
    __syncwarp();
    outp[i] = total + (int)positive + __all_sync(0xffffffff, v != 0);
}
"""


@pytest.mark.parametrize("backend", ("metal", "directx", "opengl", "vulkan", "wgsl"))
def test_cuda_source_votes_are_emulated_for_the_dispatched_block(tmp_path, backend):
    # The importer used to lower these to wave operations, which act on a subgroup of another size on lavapipe,
    # WARP and most GPUs; the block of 64 threads holds two warps.
    path = tmp_path / "votes.cu"
    path.write_text(CUDA_VOTES)
    generated = crosstl.translate(
        str(path),
        backend=backend,
        format_output=False,
        source_options={"threadgroup_size": [64, 1, 1]},
    )
    for operation in WAVE_OPERATIONS:
        assert operation not in generated, operation
    assert "cgl_warp_result_2" in generated


def test_cuda_source_votes_without_a_block_size_are_refused(tmp_path):
    path = tmp_path / "votes.cu"
    path.write_text(CUDA_VOTES)
    with pytest.raises(CudaWarpIntrinsicUnsupportedError) as raised:
        crosstl.translate(str(path), backend="vulkan", format_output=False)
    assert raised.value.reason_code == "workgroup-size"

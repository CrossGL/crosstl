"""CrossGL wave (subgroup) operations written by the Mojo generator as loops over a MAX warp's active lanes.

The helpers name the active lanes (NVVM ``activemask`` on NVIDIA, MAX's ``vote`` of ``True`` elsewhere) and read
each of them with ``shuffle_idx`` under that mask, so every active lane folds the same lanes in the same order.
``subgroupBarrier`` is a warp barrier between two fences, and CUDA's ``__match_any_sync`` matches bit patterns
among the lanes of the caller's 32-lane CUDA warp that its mask names. Quad operations stay refused.
"""

from __future__ import annotations

import re
import subprocess

import pytest

import crosstl
from crosstl.translator.refusals import MojoRefusal
from tests.test_translator.test_codegen.test_mojo_codegen import find_mojo_compiler

KERNEL = """shader Probe {{
    compute {{
        layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;
        buffer float* fin;
        buffer uint* uin;
        buffer int* iin;
        buffer double* din;
        buffer float* outp;
        buffer uint* uout;
        buffer int* iout;
        void main() {{
            uint i = gl_GlobalInvocationID.x;
            float f = fin[i];
            uint u = uin[i];
            int s = iin[i];
            double d = din[i];
            bool p = f > 0.5;
            vec4 v = vec4(f, f + 1.0, f * 2.0, 0.0);
            ivec3 iv = ivec3(s, s + 1, s * 2);
            uvec4 mask = uvec4(u, u, 0u, 0u);
            if (i % 3u != 0u) {{
                {body}
            }}
        }}
    }}
}}
"""

# One kernel per family of wave operations; each calls the operation inside a divergent branch.
CASES = {
    "lane": (
        "uout[i] = WaveGetLaneIndex() + WaveGetLaneCount(); if (WaveIsFirstLane()) { uout[i] = 7u; }"
    ),
    "sum": (
        "outp[i] = WaveActiveSum(f); iout[i] = WaveActiveSum(s); uout[i] = WaveActiveSum(u);"
    ),
    "product": "outp[i] = WaveActiveProduct(f); iout[i] = WaveActiveProduct(s);",
    "minmax": (
        "outp[i] = WaveActiveMin(f) + WaveActiveMax(f); iout[i] = WaveActiveMin(s) - WaveActiveMax(s);"
        " uout[i] = WaveActiveMin(u);"
    ),
    "bits": (
        "uout[i] = WaveActiveBitAnd(u) | WaveActiveBitOr(u) ^ WaveActiveBitXor(u);"
        " iout[i] = WaveActiveBitOr(s);"
    ),
    "boolbits": (
        "bool a = WaveActiveBitAnd(p); bool o = WaveActiveBitOr(p); bool x = WaveActiveBitXor(p);"
        " if (a || o || x) { uout[i] = 1u; }"
    ),
    "vectors": (
        "vec4 r = WaveActiveSum(v) + WaveActiveMax(v) + WaveReadLaneAt(v, 3u); outp[i] = r.y;"
        " ivec3 q = WaveActiveMin(iv) + WavePrefixSum(iv); iout[i] = q.z;"
    ),
    "votes": (
        "if (WaveActiveAllTrue(p) || WaveActiveAnyTrue(p)) {"
        " uout[i] = WaveActiveCountBits(p) + WavePrefixCountBits(p); }"
    ),
    "ballot": "uvec4 b = WaveActiveBallot(p); uout[i] = b.x + b.y;",
    "allequal": (
        "if (WaveActiveAllEqual(f) && WaveActiveAllEqual(u) && WaveActiveAllEqual(p)) { uout[i] = 1u; }"
    ),
    "reads": (
        "outp[i] = WaveReadLaneAt(f, 0u) + WaveReadLaneFirst(f); uout[i] = WaveReadLaneAt(u, i % 4u);"
        " if (WaveReadLaneFirst(p)) { iout[i] = WaveReadLaneAt(s, 1u); }"
    ),
    "shuffles": (
        "outp[i] = subgroupShuffleDown(f, 1u) + subgroupShuffleUp(f, 2u) + subgroupShuffleXor(f, 4u);"
        " uout[i] = subgroupShuffle(u, 3u) + subgroupBroadcast(u, 0u);"
    ),
    "prefix": (
        "outp[i] = WavePrefixSum(f) + WavePrefixProduct(f) + subgroupInclusiveAdd(f) + subgroupInclusiveMul(f);"
        " uout[i] = subgroupExclusiveMin(u) + subgroupInclusiveMax(u) + subgroupExclusiveAnd(u)"
        " + subgroupInclusiveOr(u) + subgroupExclusiveXor(u);"
    ),
    "boolscan": (
        "if (subgroupInclusiveAnd(p) || subgroupExclusiveOr(p) || subgroupInclusiveXor(p)) { uout[i] = 1u; }"
    ),
    "clustered": (
        "outp[i] = subgroupClusteredAdd(f, 4u) + subgroupClusteredMax(f, 8u);"
        " uout[i] = subgroupClusteredAnd(u, 2u) + subgroupClusteredXor(u, 16u);"
        " if (subgroupClusteredOr(p, 4u)) { iout[i] = 1; }"
    ),
    "match": (
        "uvec4 m = WaveMatch(f); uvec4 n = WaveMatch(iv); uvec4 o = WaveMatch(p); uout[i] = m.x + n.x + o.y;"
    ),
    "multiprefix": (
        "outp[i] = WaveMultiPrefixSum(f, mask) + WaveMultiPrefixProduct(f, mask);"
        " uout[i] = WaveMultiPrefixBitAnd(u, mask) + WaveMultiPrefixBitOr(u, mask)"
        " + WaveMultiPrefixBitXor(u, mask) + WaveMultiPrefixCountBits(p, mask);"
    ),
    "ballotbits": (
        "uvec4 b = subgroupBallot(p); uout[i] = subgroupBallotBitCount(b) + subgroupBallotInclusiveBitCount(b)"
        " + subgroupBallotExclusiveBitCount(b) + subgroupBallotFindLSB(b) + subgroupBallotFindMSB(b);"
        " if (subgroupBallotBitExtract(b, 5u) || subgroupInverseBallot(b)) { iout[i] = 1; }"
    ),
    "rotate": "outp[i] = subgroupRotate(f, 1u) + subgroupClusteredRotate(f, 1u, 4u);",
    "partitioned": (
        "outp[i] = subgroupPartitionedAddNV(f, mask) + subgroupPartitionedInclusiveMaxNV(f, mask);"
        " uout[i] = subgroupPartitionedExclusiveAddNV(u, mask);"
    ),
    "barriers": (
        "subgroupBarrier(); subgroupMemoryBarrier(); subgroupMemoryBarrierShared(); uout[i] = 1u;"
    ),
    "cuda_match": (
        "uout[i] = __match_any_sync(4294967295u, f) + __match_any_sync(65535u, d)"
        " + __match_any_sync(u, s);"
    ),
}


BUILTIN_CALL = re.compile(
    r"(?<![_A-Za-z])(?:Wave|subgroup|Quad)[A-Z]\w*\(|__match_any_sync"
)


def translate(tmp_path, body, name="probe"):
    source = tmp_path / f"{name}.cgl"
    source.write_text(KERNEL.format(body=body))
    return crosstl.translate(str(source), backend="mojo")


def helper(code, name):
    """The source of the Mojo helper ``name`` in ``code``."""
    start = code.index(f"def {name}(")
    end = code.find("\n\n\n", start)
    return code[start:end]


@pytest.mark.parametrize("name", sorted(CASES))
def test_every_wave_family_translates_without_a_stand_in(tmp_path, name):
    code = translate(tmp_path, CASES[name], name)

    # No call of a CrossGL or CUDA warp built-in is left; CrossGL's own helpers (crossgl_Wave...) may stay.
    assert not BUILTIN_CALL.search(code), BUILTIN_CALL.search(code)


@pytest.mark.parametrize("name", sorted(CASES))
def test_every_wave_family_builds_with_mojo(tmp_path, name):
    mojo = find_mojo_compiler()
    code = translate(tmp_path, CASES[name], name)
    source = tmp_path / f"{name}.mojo"
    source.write_text(code)

    result = subprocess.run(
        [
            mojo,
            "build",
            "--emit",
            "llvm",
            "--disable-warnings",
            str(source),
            "-o",
            str(tmp_path / f"{name}.ll"),
        ],
        capture_output=True,
        text=True,
        timeout=300,
    )

    assert result.returncode == 0, (result.stdout + result.stderr)[-2000:]


def test_reductions_fold_every_active_lane_in_lane_order(tmp_path):
    code = translate(
        tmp_path, "outp[i] = WaveActiveSum(f); uout[i] = WaveActiveMin(u);"
    )

    total = helper(code, "_crossgl_wave_active_sum_f32")
    assert "var active = _crossgl_active_lanes()" in total
    assert "var source = _crossgl_ctz(remaining)" in total
    assert "_crossgl_lane_read(active, value, UInt32(source))" in total
    assert "remaining &= remaining - 1" in total
    # The NVIDIA active mask comes from activemask, never from a full-warp constant.
    assert 'llvm_intrinsic["llvm.nvvm.activemask", UInt32]()' in code
    assert "0xFFFFFFFF)" not in total


def test_scans_choose_their_lanes_by_lane_index(tmp_path):
    code = translate(
        tmp_path,
        "outp[i] = WavePrefixSum(f) + subgroupInclusiveAdd(f) + WaveActiveMin(f);",
    )

    assert "source < lane" in helper(code, "_crossgl_wave_prefix_sum_f32")
    assert "source <= lane" in helper(code, "_crossgl_wave_prefix_inclusive_sum_f32")
    # A lane that combines no lane gets the operator's identity: +inf for a floating-point min.
    assert "Float32.MAX" in helper(code, "_crossgl_wave_active_min_f32")


def test_boolean_votes_use_the_ballot_of_the_active_lanes(tmp_path):
    code = translate(
        tmp_path, "if (WaveActiveAllTrue(p)) { uout[i] = WaveActiveCountBits(p); }"
    )

    all_true = helper(code, "_crossgl_wave_active_all_true")
    assert "return _crossgl_lane_ballot(active, predicate) == active" in all_true
    assert "_crossgl_pop_count" in helper(code, "_crossgl_wave_active_count_bits")


def test_subgroup_barrier_fences_both_sides_of_a_warp_barrier(tmp_path):
    code = translate(tmp_path, CASES["barriers"])

    barrier = helper(code, "_crossgl_subgroup_barrier")
    assert barrier.splitlines()[1:] == [
        "    _crossgl_fence()",
        "    _crossgl_syncwarp(Int(_crossgl_active_lanes()))",
        "    _crossgl_fence()",
    ]
    assert helper(code, "_crossgl_subgroup_memory_barrier").splitlines()[1:] == [
        "    _crossgl_fence()"
    ]
    assert "from std.atomic import fence as _crossgl_fence" in code
    body = code.split("def compute_main", 1)[1]
    assert body.count("_crossgl_subgroup_memory_barrier()") == 2


def test_cuda_match_any_compares_bits_within_the_callers_cuda_warp(tmp_path):
    code = translate(tmp_path, CASES["cuda_match"])

    match = helper(code, "_crossgl_cuda_match_any_f32")
    assert match.startswith(
        "def _crossgl_cuda_match_any_f32(mask: UInt32, value: Float32) -> UInt32:"
    )
    assert "var warp = UInt64(lane_id()) >> 5" in match
    assert "(source >> 5) == warp and (mask & bit) != 0" in match
    assert (
        "_crossgl_bitcast[DType.uint32](other) == _crossgl_bitcast[DType.uint32](value)"
        in match
    )
    assert "_crossgl_bitcast[DType.uint64]" in helper(
        code, "_crossgl_cuda_match_any_f64"
    )
    assert "_crossgl_cuda_match_any_i32(UInt32(u), s)" in code


@pytest.mark.parametrize(
    ("body", "message"),
    [
        ("uout[i] = __match_any_sync(1u, v);", "__match_any_sync: CUDA matches"),
        ("uout[i] = __match_any_sync(1u);", "__match_any_sync requires 2 arguments"),
        ("uout[i] = QuadReadAcrossX(u);", "QuadReadAcrossX: .*does not lower quad"),
        ("outp[i] = WaveActiveBitAnd(f);", "requires an integer or bool operand"),
        ("bool q = WaveActiveSum(p);", "requires a numeric operand, got bool"),
    ],
)
def test_wave_operations_mojo_cannot_express_are_refused(tmp_path, body, message):
    with pytest.raises(MojoRefusal, match=message):
        translate(tmp_path, body)


def test_a_program_function_named_subgroup_barrier_stays_a_call(tmp_path):
    source = tmp_path / "user.cgl"
    source.write_text("""shader Probe {
    void subgroupBarrier() {}
    compute {
        layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;
        buffer uint* uout;
        void main() {
            subgroupBarrier();
            uout[gl_GlobalInvocationID.x] = 1u;
        }
    }
}
""")

    code = crosstl.translate(str(source), backend="mojo")

    assert "_crossgl_subgroup_barrier" not in code
    assert "_crossgl_fence" not in code


def test_wave_helpers_are_written_once_in_first_use_order(tmp_path):
    body = "outp[i] = WaveActiveSum(f) + WaveActiveMax(f) + WaveActiveSum(f);"
    first = translate(tmp_path, body, "first")
    second = translate(tmp_path, body, "second")

    assert first == second
    assert first.count("def _crossgl_wave_active_sum_f32(") == 1
    assert first.index("def _crossgl_wave_active_sum_f32(") < first.index(
        "def _crossgl_wave_active_max_f32("
    )

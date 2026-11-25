"""CrossGL wave operations in the rust-gpu output (:mod:`crosstl.translator.codegen.rust_gpu_subgroups`).

Each operation is written as the ``spirv_std::arch`` subgroup function of its SPIR-V ``OpGroupNonUniform*``
instruction, chosen by the operand's scalar type, and every generated crate compiles with rust-gpu. These were the
85 Rust output coverage cells refused as "subgroup operations: not lowered to rust-gpu's spirv_std".
"""

from __future__ import annotations

import re

import pytest

from crosstl.translator.codegen.rust_refusals import RustLoweringRefused

from .test_codegen.test_rust_codegen import (
    assert_generated_rust_smoke_compiles,
    generate_code,
    parse_code,
    tokenize_code,
)


def _rust(source):
    return generate_code(parse_code(tokenize_code(source)))


def _kernel(body, layout="", helpers=""):
    return f"""
        shader Probe {{
            compute {{
                layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;
                {layout}
                buffer float* fin;
                buffer int* iin;
                buffer uint* uin;
                buffer float* outp;
                {helpers}
                void main() {{
                    uint i = gl_GlobalInvocationID.x;
                    float x = fin[i];
                    int a = iin[i];
                    uint p = uin[i];
                    bool b = x > 0.0;
                    vec3 v = vec3(fin[i], fin[i + 1u], fin[i + 2u]);
                    ivec2 n = ivec2(a, a + 1);
                    bvec3 c = bvec3(b, !b, b);
                    uvec4 m = WaveActiveBallot(b);
                    {body}
                }}
            }}
        }}
        """


_A = "spirv_std::arch::"

SPELLINGS = [
    ("float r = WaveActiveSum(x);", f"{_A}subgroup_f_add(x)"),
    ("int r = WaveActiveSum(a);", f"{_A}subgroup_i_add(a)"),
    ("uint r = WaveActiveProduct(p);", f"{_A}subgroup_i_mul(p)"),
    ("int r = WaveActiveMin(a);", f"{_A}subgroup_s_min(a)"),
    ("uint r = WaveActiveMax(p);", f"{_A}subgroup_u_max(p)"),
    ("float r = WaveActiveMin(x);", f"{_A}subgroup_f_min(x)"),
    ("uint r = WaveActiveBitAnd(p);", f"{_A}subgroup_and(p)"),
    ("bool r = WaveActiveBitXor(b);", f"{_A}subgroup_logical_xor(b)"),
    ("vec3 r = WaveActiveSum(v);", f"{_A}subgroup_f_add(v)"),
    ("ivec2 r = WaveActiveMax(n);", f"{_A}subgroup_s_max(n)"),
    ("bvec3 r = WaveActiveBitOr(c);", f"{_A}subgroup_logical_or(c)"),
    ("float r = WavePrefixSum(x);", f"{_A}subgroup_exclusive_f_add(x)"),
    ("int r = WavePrefixProduct(a);", f"{_A}subgroup_exclusive_i_mul(a)"),
    ("float r = WavePrefixInclusiveSum(x);", f"{_A}subgroup_inclusive_f_add(x)"),
    ("uint r = WavePrefixExclusiveMin(p);", f"{_A}subgroup_exclusive_u_min(p)"),
    ("int r = WavePrefixInclusiveBitXor(a);", f"{_A}subgroup_inclusive_xor(a)"),
    (
        "bool r = WavePrefixExclusiveBitAnd(b);",
        f"{_A}subgroup_exclusive_logical_and(b)",
    ),
    (
        "float r = WaveClusteredSum(x, 4u);",
        f"unsafe {{ {_A}subgroup_clustered_f_add::<4, _>(x) }}",
    ),
    (
        "uint r = WaveClusteredMax(p, 8u);",
        f"unsafe {{ {_A}subgroup_clustered_u_max::<8, _>(p) }}",
    ),
    (
        "bool r = WaveClusteredBitAnd(b, 2u);",
        f"unsafe {{ {_A}subgroup_clustered_logical_and::<2, _>(b) }}",
    ),
    ("bool r = WaveActiveAnyTrue(b);", f"{_A}subgroup_any(b)"),
    ("bool r = WaveActiveAllTrue(b);", f"{_A}subgroup_all(b)"),
    ("bool r = WaveActiveAllEqual(v);", f"{_A}subgroup_all_equal(v)"),
    ("bool r = WaveIsFirstLane();", f"{_A}subgroup_elect()"),
    ("uvec4 r = WaveActiveBallot(b);", f"{_A}subgroup_ballot(b)"),
    (
        "uint r = WaveActiveCountBits(b);",
        f"{_A}subgroup_ballot_bit_count({_A}subgroup_ballot(b))",
    ),
    (
        "uint r = WavePrefixCountBits(b);",
        f"{_A}subgroup_ballot_exclusive_bit_count({_A}subgroup_ballot(b))",
    ),
    ("uint r = WaveBallotBitCount(m);", f"{_A}subgroup_ballot_bit_count(m)"),
    (
        "uint r = WaveBallotInclusiveBitCount(m);",
        f"{_A}subgroup_ballot_inclusive_bit_count(m)",
    ),
    ("uint r = WaveBallotFindMSB(m);", f"{_A}subgroup_ballot_find_msb(m)"),
    ("bool r = WaveBallotBitExtract(m, p);", f"{_A}subgroup_ballot_bit_extract(m, p)"),
    ("bool r = WaveInverseBallot(m);", f"unsafe {{ {_A}subgroup_inverse_ballot(m) }}"),
    ("float r = WaveReadLaneFirst(x);", f"{_A}subgroup_broadcast_first(x)"),
    ("float r = WaveReadLaneAt(x, p);", f"{_A}subgroup_shuffle(x, p)"),
    # A lane index of another integer type is converted to the u32 spirv-std takes.
    ("vec3 r = WaveReadLaneAt(v, a);", f"{_A}subgroup_shuffle(v, ((a) as u32))"),
    ("int r = WaveShuffleXor(a, 1u);", f"{_A}subgroup_shuffle_xor(a, 1)"),
    ("float r = WaveShuffleUp(x, 2u);", f"{_A}subgroup_shuffle_up(x, 2)"),
    ("float r = WaveShuffleDown(x, p);", f"{_A}subgroup_shuffle_down(x, p)"),
    ("float r = QuadReadLaneAt(x, 1u);", f"{_A}subgroup_quad_broadcast(x, 1)"),
    ("float r = QuadReadAcrossX(x);", f"{_A}subgroup_quad_swap::<0, _>(x)"),
    ("float r = QuadReadAcrossY(x);", f"{_A}subgroup_quad_swap::<1, _>(x)"),
    ("vec3 r = QuadReadAcrossDiagonal(v);", f"{_A}subgroup_quad_swap::<2, _>(v)"),
]


def _spaced(text):
    return " ".join(text.replace("{", " { ").replace("}", " } ").split())


@pytest.mark.parametrize("statement, spelling", SPELLINGS)
def test_wave_operations_are_spirv_std_subgroup_calls(statement, spelling):
    code = _rust(_kernel(statement + "\n outp[i] = 0.0;"))
    assert _spaced(spelling) in _spaced(code), code


def test_lane_queries_bind_the_subgroup_built_ins_in_helpers_too():
    code = _rust(
        _kernel(
            "outp[i] = float(lane() + WaveGetLaneCount());",
            helpers="uint lane() { return WaveGetLaneIndex(); }",
        )
    )
    code = re.sub(r"\s*:\s*", ": ", code)
    assert (
        "#[spirv(subgroup_local_invocation_id)] gl_SubgroupInvocationID: u32" in code
    ), code
    assert "#[spirv(subgroup_size)] gl_SubgroupSize: u32" in code, code
    assert "pub fn lane(gl_SubgroupInvocationID: u32) -> u32" in code, code
    assert "WaveGetLane" not in code


ALL_OPERATIONS = _kernel(
    "\n".join(
        statement.replace("r =", f"r{index} =")
        for index, (statement, _) in enumerate(SPELLINGS)
    )
    + "\n uint lanes = WaveGetLaneIndex() + WaveGetLaneCount();\n outp[i] = r0 + float(lanes);",
)


def test_every_lowering_compiles_with_rust_gpu(tmp_path):
    assert_generated_rust_smoke_compiles(_rust(ALL_OPERATIONS), tmp_path)


@pytest.mark.parametrize(
    "statement",
    [
        # spirv-std has no subgroup add of booleans, and no quad vote.
        "bool r = WaveActiveSum(b);",
        "bool r = QuadAny(b);",
        # A clustered reduction needs a constant power-of-two cluster size.
        "float r = WaveClusteredSum(x, p);",
        "float r = WaveClusteredSum(x, 3u);",
        # Ballot arithmetic takes a uvec4 mask, and a lane index is an integer.
        "uint r = WaveBallotBitCount(p);",
        "float r = WaveReadLaneAt(x, x);",
    ],
)
def test_operations_without_a_spirv_std_lowering_are_refused(statement):
    with pytest.raises(
        RustLoweringRefused, match="has no spirv_std lowering for these operands"
    ):
        _rust(_kernel(statement + "\n outp[i] = 0.0;"))


def test_quad_operations_of_a_declared_derivative_group_are_refused():
    with pytest.raises(RustLoweringRefused, match="QuadReadAcrossX"):
        _rust(
            _kernel(
                "float r = QuadReadAcrossX(x);\n outp[i] = r;",
                layout="layout(derivative_group_linearKHR) in;",
            )
        )

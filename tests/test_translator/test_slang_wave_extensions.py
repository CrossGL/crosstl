"""Slang output writes CrossGL's wave operations beyond Shader Model 6.0 with the wave functions of Slang's core module.

Slang v2026.20 declares QuadAll/QuadAny, WaveActiveAllEqual, WaveActiveCountBits, WavePrefixCountBits, the exclusive
scans WavePrefix{Min,Max,BitAnd,BitOr,BitXor} and the clustered reductions WaveClustered*. An inclusive scan is the
exclusive scan combined with the lane's own value; WaveMultiPrefixCountBits, which Slang declares for HLSL and CUDA
only, counts the ballot under the partition and the lower lanes.
"""

from __future__ import annotations

import re
import shutil
import subprocess

import pytest

from crosstl.translator.codegen.slang_codegen import SlangCodeGen
from crosstl.translator.codegen.slang_refusals import SlangLoweringRefused
from crosstl.translator.lexer import Lexer
from crosstl.translator.parser import Parser

COMPUTE = """shader Probe {{
    compute {{
        layout(local_size_x = 32, local_size_y = 1, local_size_z = 1) in;
        buffer float* f;
        buffer uint* u;
        buffer int* n;
        void main(uvec3 gid @builtin(global_invocation_id)) {{
            float x = f[gid.x];
            uint k = u[gid.x];
            int i = n[gid.x];
            vec2 v = vec2(x, x);
            bool b = k > 2u;
            bvec2 bb = bvec2(b, !b);
            {statement}
        }}
    }}
}}
"""


def generate(statement):
    return SlangCodeGen().generate(
        Parser(Lexer(COMPUTE.format(statement=statement)).get_tokens()).parse()
    )


def written(statement):
    slang = generate(statement)
    target = statement.split("=", 1)[0].strip()
    lines = [
        line.strip()
        for line in slang.splitlines()
        if line.strip().startswith(target + " =")
    ]
    assert len(lines) == 1, slang
    return lines[0], slang


@pytest.mark.parametrize(
    "statement, expected",
    [
        ("bool q = QuadAll(b); u[1] = q ? 1u : 0u;", "bool q = QuadAll(b);"),
        ("bool q = QuadAny(b); u[1] = q ? 1u : 0u;", "bool q = QuadAny(b);"),
        (
            "bool q = WaveActiveAllEqual(v); u[1] = q ? 1u : 0u;",
            "bool q = WaveActiveAllEqual(v);",
        ),
        ("u[1] = WaveActiveCountBits(b);", "u[1] = WaveActiveCountBits(b);"),
        ("u[1] = WavePrefixCountBits(b);", "u[1] = WavePrefixCountBits(b);"),
        ("f[1] = WavePrefixExclusiveMax(x);", "f[1] = WavePrefixMax(x);"),
        ("f[1] = WavePrefixExclusiveMin(x);", "f[1] = WavePrefixMin(x);"),
        ("u[1] = WavePrefixExclusiveBitAnd(k);", "u[1] = WavePrefixBitAnd(k);"),
        ("u[1] = WavePrefixExclusiveBitOr(k);", "u[1] = WavePrefixBitOr(k);"),
        ("n[1] = WavePrefixExclusiveBitXor(i);", "n[1] = WavePrefixBitXor(i);"),
        ("f[1] = WaveClusteredSum(x, 4u);", "f[1] = WaveClusteredSum(x, 4u);"),
        ("f[1] = WaveClusteredProduct(x, 2);", "f[1] = WaveClusteredProduct(x, 2u);"),
        ("n[1] = WaveClusteredMin(i, 8u);", "n[1] = WaveClusteredMin(i, 8u);"),
        ("n[1] = WaveClusteredMax(i, 16u);", "n[1] = WaveClusteredMax(i, 16u);"),
        ("u[1] = WaveClusteredBitAnd(k, 4u);", "u[1] = WaveClusteredBitAnd(k, 4u);"),
        ("u[1] = WaveClusteredBitOr(k, 4u);", "u[1] = WaveClusteredBitOr(k, 4u);"),
        ("u[1] = WaveClusteredBitXor(k, 4u);", "u[1] = WaveClusteredBitXor(k, 4u);"),
    ],
)
def test_operations_slang_declares_are_called(statement, expected):
    assert written(statement)[0] == expected


@pytest.mark.parametrize(
    "operation, scan, combined",
    [
        ("WavePrefixInclusiveSum", "WavePrefixSum", "WavePrefixSum(value) + value"),
        (
            "WavePrefixInclusiveProduct",
            "WavePrefixProduct",
            "WavePrefixProduct(value) * value",
        ),
        ("WavePrefixInclusiveMin", "WavePrefixMin", "min(WavePrefixMin(value), value)"),
        ("WavePrefixInclusiveMax", "WavePrefixMax", "max(WavePrefixMax(value), value)"),
    ],
)
def test_inclusive_scans_combine_the_exclusive_scan_with_the_lane_value(
    operation, scan, combined
):
    line, slang = written(f"f[1] = {operation}(x);")
    assert line == f"f[1] = cgl_{operation}_float(x);"
    assert f"float cgl_{operation}_float(float value)" in slang
    assert f"    return {combined};" in slang


def test_inclusive_bitwise_scans_of_integers():
    line, slang = written(
        "u[1] = WavePrefixInclusiveBitAnd(k) | WavePrefixInclusiveBitOr(k) | WavePrefixInclusiveBitXor(k);"
    )
    assert line == (
        "u[1] = cgl_WavePrefixInclusiveBitAnd_uint(k) | cgl_WavePrefixInclusiveBitOr_uint(k) "
        "| cgl_WavePrefixInclusiveBitXor_uint(k);"
    )
    assert "    return WavePrefixBitAnd(value) & value;" in slang
    assert "    return WavePrefixBitOr(value) | value;" in slang
    assert "    return WavePrefixBitXor(value) ^ value;" in slang


def test_bool_bitwise_operations_are_the_logical_operators():
    line, slang = written(
        "u[1] = (WavePrefixExclusiveBitAnd(b) || WaveClusteredBitXor(b, 4u) "
        "|| WavePrefixInclusiveBitOr(b) || WavePrefixExclusiveBitXor(bb).y) ? 1u : 0u;"
    )
    assert "(WavePrefixBitAnd(uint(b)) != uint(0))" in line
    assert "(WaveClusteredBitXor(uint(b), 4u) != uint(0))" in line
    assert "cgl_WavePrefixInclusiveBitOr_bool(b)" in line
    assert "(WavePrefixBitXor(uint2(bb)) != uint2(0)).y" in line
    assert "    uint bits = uint(value);" in slang
    assert "    return ((WavePrefixBitOr(bits) | bits) != uint(0));" in slang


def test_multi_prefix_count_bits_counts_the_lower_lanes_of_the_partition():
    line, slang = written("u[1] = WaveMultiPrefixCountBits(b, WaveMatch(k));")
    assert line == "u[1] = cgl_WaveMultiPrefixCountBits(b, WaveMatch(k));"
    assert "uint4 bits = WaveActiveBallot(value) & mask & WaveGetLaneLtMask();" in slang
    assert (
        "return countbits(bits.x) + countbits(bits.y) + countbits(bits.z) + countbits(bits.w);"
        in slang
    )


@pytest.mark.parametrize(
    "statement, reason",
    [
        ("f[1] = WaveClusteredSum(x, 3u);", "constant power-of-two cluster size"),
        ("f[1] = WaveClusteredSum(x, k);", "constant power-of-two cluster size"),
        (
            "f[1] = WavePrefixExclusiveBitAnd(x);",
            "integer or bool scalar or vector, got float",
        ),
        ("u[1] = WaveActiveCountBits(k);", "predicate must be scalar bool, got uint"),
        (
            "u[1] = WavePrefixExclusiveMax(b) ? 1u : 0u;",
            "value must be numeric, got bool",
        ),
        (
            "u[1] = WaveMultiPrefixCountBits(b, k);",
            "partition mask must be uint4, got uint",
        ),
        ("f[1] = WaveClusteredMax(x);", "expects 2 arguments, got 1"),
    ],
)
def test_operations_without_a_slang_form_are_refused(statement, reason):
    with pytest.raises(SlangLoweringRefused, match=re.escape(reason)):
        generate(statement)


def test_slang_output_compiles_with_slangc(tmp_path):
    slangc = shutil.which("slangc")
    if slangc is None:
        pytest.skip("slangc is not installed")
    statement = (
        "f[1] = WavePrefixExclusiveMax(x) + WavePrefixExclusiveMin(x) + WavePrefixInclusiveMax(x)"
        " + WavePrefixInclusiveMin(x) + WavePrefixInclusiveSum(x) + WavePrefixInclusiveProduct(x)"
        " + WaveClusteredSum(x, 4u) + WaveClusteredProduct(v, 2u).y;"
        " u[1] = WavePrefixExclusiveBitAnd(k) | WavePrefixInclusiveBitOr(k) | WaveClusteredBitXor(k, 8u)"
        " | WaveActiveCountBits(b) | WavePrefixCountBits(b) | WaveMultiPrefixCountBits(b, WaveMatch(k));"
        " n[1] = WaveClusteredMin(i, 4u) + WaveClusteredMax(i, 4u) + WavePrefixInclusiveBitXor(i);"
        " u[2] = (QuadAll(b) && QuadAny(b) && WaveActiveAllEqual(v) && WavePrefixInclusiveBitAnd(b)"
        " && WaveClusteredBitOr(bb, 4u).x) ? 1u : 0u;"
    )
    source = tmp_path / "waves.slang"
    source.write_text(generate(statement), encoding="utf-8")
    completed = subprocess.run(
        [
            slangc,
            str(source),
            "-target",
            "spirv",
            "-entry",
            "main",
            "-stage",
            "compute",
            "-o",
            str(tmp_path / "waves.spv"),
        ],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert completed.returncode == 0, (completed.stdout + completed.stderr)[-3000:]

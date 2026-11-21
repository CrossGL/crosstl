"""Explicit conversions of Metal ballot masks to their ``vote_t`` integers import through ``WaveActiveBallot``.

The importer wrote ``static_cast<simd_vote::vote_t>(simd_ballot(p))`` as ``(vote_t)WaveActiveBallot(p)``, which no
CrossGL program declares, so the four ballot probes of the Metal input row were refused. The conversion now calls a
helper that assembles the mask from the ``uvec4`` ballot (see :mod:`crosstl.backend.Metal.vote_masks`).
"""

import shutil
import subprocess

import pytest

import crosstl
from crosstl.backend.import_refusals import MetalImportRefused
from crosstl.backend.Metal.MetalCrossGLCodeGen import MetalToCrossGLConverter
from crosstl.backend.Metal.MetalLexer import MetalLexer
from crosstl.backend.Metal.MetalParser import MetalParser
from crosstl.translator.lexer import Lexer as CrossGLLexer
from crosstl.translator.parser import Parser as CrossGLParser

KERNEL = """#include <metal_stdlib>
using namespace metal;
HELPERS
kernel void probe(device float* fin [[buffer(0)]], device ulong* outp [[buffer(1)]],
                  uint3 id [[thread_position_in_grid]]) {
    uint i = id.x;
    bool p = fin[i] > 0.0;
    outp[i] = ulong(VALUE);
}
"""

SIMD = "__crossgl_metal_simd_vote_bits"
QUAD = "__crossgl_metal_quad_vote_bits"


def _crossgl(value, helpers=""):
    source = KERNEL.replace("HELPERS", helpers).replace("VALUE", value)
    program = MetalParser(MetalLexer(source).tokenize()).parse()
    crossgl = MetalToCrossGLConverter().generate(program)
    assert CrossGLParser(CrossGLLexer(crossgl).get_tokens()).parse() is not None
    return crossgl


@pytest.mark.parametrize(
    "value, call",
    [
        (
            "static_cast<simd_vote::vote_t>(simd_ballot(p))",
            f"{SIMD}(WaveActiveBallot(p))",
        ),
        ("simd_vote::vote_t(metal::simd_ballot(p))", f"{SIMD}(WaveActiveBallot(p))"),
        ("(ulong)simd_ballot(p)", f"{SIMD}(WaveActiveBallot(p))"),
        ("ulong(simd_ballot(p))", f"{SIMD}(WaveActiveBallot(p))"),
        ("uint64_t(simd_ballot(p))", f"{SIMD}(WaveActiveBallot(p))"),
        (
            "static_cast<uint64_t>(simd_active_threads_mask())",
            f"{SIMD}(WaveActiveBallot(true))",
        ),
        (
            "static_cast<quad_vote::vote_t>(quad_ballot(p))",
            f"{QUAD}(WaveActiveBallot(p), WaveGetLaneIndex())",
        ),
        (
            "ushort(quad_active_threads_mask())",
            f"{QUAD}(WaveActiveBallot(true), WaveGetLaneIndex())",
        ),
        (
            "(metal::quad_vote::vote_t)quad_ballot(p)",
            f"{QUAD}(WaveActiveBallot(p), WaveGetLaneIndex())",
        ),
    ],
)
def test_ballot_conversion_imports_as_a_mask_helper(value, call):
    crossgl = _crossgl(value)

    assert call in crossgl
    assert "vote_t" not in crossgl


def test_simd_helper_joins_the_two_low_ballot_words():
    crossgl = _crossgl("simd_vote::vote_t(simd_ballot(p))")

    assert f"uint64 {SIMD}(uvec4 ballot) {{" in crossgl
    assert "return uint64(ballot.x) | (uint64(ballot.y) << 32u);" in crossgl
    assert QUAD not in crossgl


def test_quad_helper_takes_the_four_bits_of_the_callers_quad():
    crossgl = _crossgl("quad_vote::vote_t(quad_ballot(p))")

    assert f"uint16 {QUAD}(uvec4 ballot, uint lane) {{" in crossgl
    assert "uint first = lane & 0xfffffffcu;" in crossgl
    assert "uint word = first < 32u ? ballot.x : ballot.y;" in crossgl
    assert "return uint16((word >> (first & 31u)) & 15u);" in crossgl
    assert SIMD not in crossgl


def test_helper_name_skips_a_program_function_of_that_name():
    crossgl = _crossgl(
        "simd_vote::vote_t(simd_ballot(p)) + ulong(__crossgl_metal_simd_vote_bits(1u))",
        helpers="uint __crossgl_metal_simd_vote_bits(uint x) { return x * 2u; }\n",
    )

    assert f"uint64 {SIMD}_(uvec4 ballot) {{" in crossgl
    assert f"{SIMD}_(WaveActiveBallot(p))" in crossgl


def test_program_without_a_ballot_conversion_has_no_helper():
    crossgl = _crossgl("fin[i] > 1.0 ? 1u : 0u")

    assert SIMD not in crossgl
    assert QUAD not in crossgl


def test_conversion_of_a_stored_mask_is_refused():
    source = KERNEL.replace("HELPERS", "").replace(
        "outp[i] = ulong(VALUE);",
        "simd_vote vote = simd_ballot(p);\n    outp[i] = simd_vote::vote_t(vote);",
    )
    program = MetalParser(MetalLexer(source).tokenize()).parse()

    with pytest.raises(
        MetalImportRefused, match="simd_vote value to simd_vote::vote_t"
    ):
        MetalToCrossGLConverter().generate(program)


@pytest.mark.parametrize(
    "value",
    [
        "static_cast<simd_vote::vote_t>(simd_ballot(p))",
        "quad_vote::vote_t(quad_ballot(p))",
    ],
)
def test_ballot_conversion_round_trips_to_metal(tmp_path, value):
    path = tmp_path / "probe.metal"
    path.write_text(
        KERNEL.replace("HELPERS", "").replace("VALUE", value), encoding="utf-8"
    )

    output = crosstl.translate(str(path), backend="metal")

    assert "simd_ballot(" in output
    if shutil.which("xcrun"):
        target = tmp_path / "out.metal"
        target.write_text(output, encoding="utf-8")
        result = subprocess.run(
            [
                "xcrun",
                "-sdk",
                "macosx",
                "metal",
                "-c",
                str(target),
                "-o",
                str(tmp_path / "out.air"),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    "value, reduction",
    [
        ("simd_ballot(p).any()", "WaveActiveAnyTrue(p)"),
        ("metal::quad_ballot(p).any()", "QuadAny(p)"),
        ("simd_active_threads_mask().any()", "true"),
    ],
)
def test_any_of_a_ballot_imports_as_the_any_reduction(value, reduction):
    crossgl = _crossgl(f"{value} ? 1u : 0u")

    assert f"{reduction} ? 1u : 0u" in crossgl
    assert ".any()" not in crossgl


@pytest.mark.parametrize(
    "value", ["simd_ballot(p).all()", "quad_active_threads_mask().all()"]
)
def test_all_of_a_ballot_is_refused(value):
    # all() is false in a group with an inactive thread, unlike simd_all and quad_all (see the module docstring).
    source = KERNEL.replace("HELPERS", "").replace("VALUE", f"{value} ? 1u : 0u")
    program = MetalParser(MetalLexer(source).tokenize()).parse()

    with pytest.raises(MetalImportRefused, match=r"vote::all\(\) is false"):
        MetalToCrossGLConverter().generate(program)

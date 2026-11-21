"""Explicit conversions of Metal ballot masks to their ``vote_t`` integers import through ``WaveActiveBallot``.

The importer wrote ``static_cast<simd_vote::vote_t>(simd_ballot(p))`` as ``(vote_t)WaveActiveBallot(p)``, which no
CrossGL program declares, so the four ballot probes of the Metal input row were refused. The conversion now calls a
helper that assembles the mask from the ``uvec4`` ballot (see :mod:`crosstl.backend.Metal.vote_masks`).
"""

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

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


def _crossgl_body(body, helpers=""):
    source = KERNEL.replace("HELPERS", helpers).replace("outp[i] = ulong(VALUE);", body)
    program = MetalParser(MetalLexer(source).tokenize()).parse()
    crossgl = MetalToCrossGLConverter().generate(program)
    assert CrossGLParser(CrossGLLexer(crossgl).get_tokens()).parse() is not None
    return crossgl


def _refusal(body):
    source = KERNEL.replace("HELPERS", "").replace("outp[i] = ulong(VALUE);", body)
    program = MetalParser(MetalLexer(source).tokenize()).parse()
    with pytest.raises(MetalImportRefused) as refusal:
        MetalToCrossGLConverter().generate(program)
    return str(refusal.value)


@pytest.mark.parametrize(
    "body, declaration, read",
    [
        (
            "simd_vote vote = simd_ballot(p);\n    outp[i] = simd_vote::vote_t(vote);",
            "uvec4 vote = WaveActiveBallot(p);",
            f"{SIMD}(vote)",
        ),
        (
            # A ballot local declared auto used to import as bool, and its conversion as a bool cast.
            "const auto vote = simd_ballot(p);\n    outp[i] = ulong(vote.operator unsigned long());",
            "const uvec4 vote = WaveActiveBallot(p);",
            f"{SIMD}(vote)",
        ),
        (
            "auto active = simd_active_threads_mask();\n    outp[i] = static_cast<uint64_t>(active);",
            "uvec4 active = WaveActiveBallot(true);",
            f"{SIMD}(active)",
        ),
        (
            "metal::quad_vote q = quad_ballot(p);\n    outp[i] = ulong(static_cast<ushort>(q));",
            "uvec4 q = WaveActiveBallot(p);",
            f"{QUAD}(q, WaveGetLaneIndex())",
        ),
        (
            "auto q = quad_active_threads_mask();\n    outp[i] = ulong(quad_vote::vote_t(q));",
            "uvec4 q = WaveActiveBallot(true);",
            f"{QUAD}(q, WaveGetLaneIndex())",
        ),
    ],
)
def test_stored_mask_imports_as_its_uvec4_ballot(body, declaration, read):
    crossgl = _crossgl_body(body)

    assert declaration in crossgl
    assert read in crossgl
    assert re.search(r"\b(?:simd|quad)_vote\b", crossgl) is None
    assert "bool vote" not in crossgl


def test_mask_parameter_and_return_import_as_uvec4():
    crossgl = _crossgl_body(
        "outp[i] = bits(pick(p));",
        helpers=(
            "static simd_vote pick(bool p) { return simd_ballot(p); }\n"
            "static ulong bits(simd_vote v) { return static_cast<simd_vote::vote_t>(v); }\n"
        ),
    )

    assert "uvec4 pick(bool p)" in crossgl
    assert "return WaveActiveBallot(p);" in crossgl
    assert "uint64 bits(uvec4 v)" in crossgl
    assert f"return {SIMD}(v);" in crossgl


def test_default_constructed_mask_is_a_zero_ballot():
    # simd_vote(vote_t v = 0) is the default constructor (metal_simdgroup header).
    crossgl = _crossgl_body("simd_vote vote;\n    outp[i] = simd_vote::vote_t(vote);")

    assert "uvec4 vote = uvec4(0u);" in crossgl


def test_vote_t_of_an_integer_is_an_integer_conversion():
    crossgl = _crossgl_body("outp[i] = simd_vote::vote_t(i) + quad_vote::vote_t(i);")

    assert "uint64(i) + uint16(i)" in crossgl
    assert SIMD not in crossgl and QUAD not in crossgl


@pytest.mark.parametrize(
    "value, comparison",
    [
        ("simd_vote vote = simd_ballot(p);", f"({SIMD}(vote) != uint64(0u))"),
        (
            "quad_vote vote = quad_ballot(p);",
            f"({QUAD}(vote, WaveGetLaneIndex()) != uint16(0u))",
        ),
    ],
)
def test_any_of_a_stored_mask_compares_its_vote_t_with_zero(value, comparison):
    crossgl = _crossgl_body(f"{value}\n    outp[i] = vote.any() ? 1u : 0u;")

    assert f"{comparison} ? 1u : 0u" in crossgl


@pytest.mark.parametrize(
    "body, message",
    [
        (
            "simd_vote vote = simd_vote(5ul);\n    outp[i] = simd_vote::vote_t(vote);",
            "constructs a simd_vote from its vote_t",
        ),
        (
            "simd_vote vote = simd_ballot(p);\n    outp[i] = vote.all() ? 1u : 0u;",
            r"simd_vote::all() is false",
        ),
        (
            "simd_vote vote = simd_ballot(p);\n    outp[i] = static_cast<quad_vote::vote_t>(vote);",
            "converts a simd_vote only to simd_vote::vote_t",
        ),
        (
            (
                "simd_vote votes[2];\n    votes[0] = simd_ballot(p);\n"
                "    outp[i] = simd_vote::vote_t(votes[0]);"
            ),
            "mask array votes",
        ),
    ],
)
def test_mask_construct_without_a_crossgl_form_is_refused(body, message):
    assert message in _refusal(body)


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


REQUIRE_ENV = "CROSTL_REQUIRE_METAL_VOTE_MASKS"
ROOT = Path(__file__).resolve().parents[3]
PARITY_KERNEL = """#include <metal_stdlib>
using namespace metal;
static uint low(simd_vote v) { return uint(static_cast<simd_vote::vote_t>(v)); }
kernel void kernel_main(device const uint* fin [[buffer(0)]], device uint* outp [[buffer(1)]],
                        uint i [[thread_position_in_grid]]) {
    for (uint k = 0u; k < 6u; ++k) {
        outp[i * 6u + k] = 0u;
    }
    if (fin[i] % 3u != 0u) {
        bool p = (fin[i] & 1u) != 0u;
        simd_vote vote = simd_ballot(p);
        const auto active = simd_active_threads_mask();
        quad_vote q = quad_ballot(p);
        auto qa = quad_active_threads_mask();
        simd_vote none;
        outp[i * 6u + 0u] = low(vote);
        outp[i * 6u + 1u] = uint(active.operator unsigned long());
        outp[i * 6u + 2u] = uint(static_cast<ushort>(q));
        outp[i * 6u + 3u] = uint(quad_vote::vote_t(qa));
        outp[i * 6u + 4u] = (vote.any() ? 1u : 0u) + (q.any() ? 2u : 0u) + (none.any() ? 4u : 0u);
        outp[i * 6u + 5u] = uint(static_cast<simd_vote::vote_t>(vote) >> 32);
    }
}
"""


def _run(command, cwd):
    return subprocess.run(
        command, cwd=cwd, check=True, capture_output=True, text=True, timeout=180
    ).stdout


def _readback(tmp_path, name, source, runner, request):
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
    readback = _run([str(runner), str(library), "kernel_main", str(request)], tmp_path)
    return json.loads(readback)["values"]


def test_stored_masks_compute_the_native_values_on_the_gpu(tmp_path):
    # Two SIMD-groups of 32 threads in which a third of the threads skip the ballots, so every mask is partial.
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 to require native Metal execution")
    assert sys.platform == "darwin", "Native Metal execution requires macOS"
    native = tmp_path / "native.metal"
    native.write_text(PARITY_KERNEL, encoding="utf-8")
    translated = crosstl.translate(str(native), backend="metal")
    assert "simd_ballot(" in translated and "simd_vote vote" not in translated
    runner = tmp_path / "readback"
    fixture = ROOT / "tests/fixtures/runtime_verification/metal_uint32_buffers.swift"
    _run(["xcrun", "swiftc", str(fixture), "-o", str(runner)], tmp_path)
    request = tmp_path / "request.json"
    words = [(index * 2654435761) >> 16 & 0xFFFF for index in range(64)]
    request.write_text(
        json.dumps(
            {
                "inputs": [words],
                "outputCount": 64 * 6,
                "threadCount": 1,
                "threadsPerThreadgroup": 64,
            }
        )
    )

    expected = _readback(tmp_path, "native", PARITY_KERNEL, runner, request)
    actual = _readback(tmp_path, "translated", translated, runner, request)

    assert any(
        expected[index * 6] for index in range(64)
    ), "the probe set no ballot bit"
    assert actual == expected

"""Exact binary32 division independently checked on each native runtime."""

import functools
import json
import os
import random
import sys
from fractions import Fraction
from pathlib import Path

import pytest

from crosstl import translate
from crosstl.translator.division_math import binary32_division_support
from crosstl.translator.lexer import Lexer
from crosstl.translator.parser import Parser
from tests.test_translator import test_fused_math as native_math
from tests.test_translator.test_fused_math import _dispatch
from tests.test_translator.test_metal_builtin_ownership import _compile

REQUIRE_ENV = "CROSTL_REQUIRE_DIVISION_MATH"
GUARD = 0x1937A5C3


def _source():
    return (
        "shader DivisionReference {\n" + binary32_division_support("divide_bits") + """
    StructuredBuffer<uint> values @binding(0);
    RWStructuredBuffer<uint> results @binding(1);
    compute {
        layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
        void computeMain(uvec3 tid @ gl_GlobalInvocationID) @ stage_entry {
            uint i = tid.x;
            uint a = values[2u * i];
            uint b = values[2u * i + 1u];
            results[4u + 2u * i] = divide_bits(a, b, false);
            results[5u + 2u * i] = divide_bits(a, b, true);
        }
    }
}
"""
    )


def _translate(tmp_path, target):
    path = tmp_path / "division.cgl"
    path.write_text(_source(), encoding="utf-8")
    return translate(str(path), backend=target, format_output=False)


def test_division_helper_has_caller_owned_name_and_private_linkage():
    code = "shader Division {" + binary32_division_support("private_divide") + "}"
    ast = Parser(Lexer(code).get_tokens()).parse()
    assert len(ast.functions) == 1
    assert ast.functions[0].name == "private_divide"
    assert ast.functions[0].linkage == "internal"


def _fraction(bits):
    exponent = (bits >> 23) & 255
    significand = (bits & 0x7FFFFF) | (0x800000 if exponent else 0)
    power = exponent - 150 if exponent else -149
    return Fraction(significand) * Fraction(2) ** power


def _oracle(a, b, flush=False):
    if flush:
        a, b = (w & 0x80000000 if w & 0x7F800000 == 0 else w for w in (a, b))
    sign = (a ^ b) & 0x80000000
    ma, mb = a & 0x7FFFFFFF, b & 0x7FFFFFFF
    if ma > 0x7F800000 or mb > 0x7F800000 or ma == mb == 0 or ma == mb == 0x7F800000:
        return 0x7FC00000
    if ma == 0x7F800000 or mb == 0:
        return sign | 0x7F800000
    if ma == 0 or mb == 0x7F800000:
        return sign
    exact = _fraction(a) / _fraction(b)
    if flush and exact < Fraction(2) ** -126:
        return sign
    # Binary-search ordered positive float encodings, independently of long division.
    low, high = 0, 0x7F800000
    while low + 1 < high:
        middle = (low + high) // 2
        if _fraction(middle) <= exact:
            low = middle
        else:
            high = middle
    # Infinity uses the next ideal finite value when deciding the overflow tie.
    upper = Fraction(2) ** 128 if high == 0x7F800000 else _fraction(high)
    midpoint = (_fraction(low) + upper) / 2
    return sign | (high if exact > midpoint or (exact == midpoint and low & 1) else low)


@pytest.mark.parametrize(
    "a,b,gradual,flushed",
    [
        (0x3F800000, 0x40400000, 0x3EAAAAAB, 0x3EAAAAAB),
        (0x3F800000, 0x7E970000, 0x006C80D9, 0),
        (0x3F800000, 0x7EF90000, 0x0041CC98, 0),
        (0x3F800000, 0x7F4D0000, 0x0027F602, 0),
        (0xBF800000, 0x7F4D0000, 0x8027F602, 0x80000000),
        (0x00800000, 0x3F800001, 0x007FFFFF, 0),
        (0x00800000, 0x3F800000, 0x00800000, 0x00800000),
        (0x00800000, 0x4B800000, 0, 0),
        (0x00800001, 0x4B800000, 1, 0),
        (0x00800003, 0x40000000, 0x00400002, 0),
        (0x00FFFFFF, 0x40000000, 0x00800000, 0),
        (0x80FFFFFF, 0x40000000, 0x80800000, 0x80000000),
        (1, 0x3F800000, 1, 0),
        (3, 0x40000000, 2, 0),
        (0x7F7FFFFF, 0x3F7FFFFF, 0x7F800000, 0x7F800000),
        (0x80000000, 0x3F800000, 0x80000000, 0x80000000),
        (0, 0, 0x7FC00000, 0x7FC00000),
        (0xFF800000, 0x40000000, 0xFF800000, 0xFF800000),
        (0xBF800000, 0x7F800000, 0x80000000, 0x80000000),
        (0x7F800000, 0xFF800000, 0x7FC00000, 0x7FC00000),
        (0x7F800001, 0x3F800000, 0x7FC00000, 0x7FC00000),
        (0x3F800000, 0xFFC00001, 0x7FC00000, 0x7FC00000),
        (1, 1, 0x3F800000, 0x7FC00000),
    ],
)
def test_division_oracle_boundaries(a, b, gradual, flushed):
    assert _oracle(a, b) == gradual
    assert _oracle(a, b, flush=True) == flushed


def _pairs():
    edges = (
        0,
        1,
        2,
        3,
        0x007FFFFF,
        0x00800000,
        0x00800001,
        0x00FFFFFF,
        0x3F000000,
        0x3F7FFFFF,
        0x3F800000,
        0x3F800001,
        0x40000000,
        0x40400000,
        0x4B800000,
        0x7E970000,
        0x7EF90000,
        0x7F4D0000,
        0x7F7FFFFF,
        0x7F800000,
        0x7F800001,
        0x7FC00000,
    )
    signed = [word | sign for word in edges for sign in (0, 0x80000000)]
    pairs = [(a, b) for a in signed for b in signed]
    # Exercise every exponent with adjacent mantissas on both sides of one.
    pairs.extend(
        ((exponent << 23) | fraction | sign, denominator)
        for exponent in range(255)
        for fraction in (0, 1, 0x7FFFFE, 0x7FFFFF)
        for sign in (0, 0x80000000)
        for denominator in (0x3F7FFFFF, 0x3F800000, 0x3F800001)
    )
    source = random.Random(2086)
    pairs.extend((source.getrandbits(32), source.getrandbits(32)) for _ in range(4096))
    return pairs


@pytest.mark.parametrize("target", ["directx", "opengl", "metal"])
def test_integer_division_helper_compiles(tmp_path, target):
    generated = _translate(tmp_path, target)
    assert "divide_bits(" in generated
    assert "uint64" not in generated and "double" not in generated
    assert "$bits" not in generated
    _compile(generated, target, tmp_path)


def test_integer_division_helper_executes(tmp_path):
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 for native division rounding")
    target = {"win32": "directx", "linux": "opengl", "darwin": "metal"}[sys.platform]
    pairs = _pairs()
    expected = (
        [GUARD] * 4
        + [_oracle(*pair, flush=flush) for pair in pairs for flush in (False, True)]
        + [GUARD] * 4
    )
    (tmp_path / "expected.json").write_text(json.dumps(expected), encoding="utf-8")
    actual, evidence = _dispatch(
        tmp_path,
        target,
        _translate(tmp_path, target),
        pairs,
        len(expected),
        initial_output=[GUARD] * len(expected),
    )
    mismatches = [
        {"index": i, "expected": want, "actual": got}
        for i, (want, got) in enumerate(zip(expected, actual))
        if want != got
    ]
    evidence.update(
        profiles=["rne-gradual", "rne-flush-before-rounding"],
        oracle="exact rational, ordered-encoding binary search",
        guardCount=8,
        mismatchCount=len(mismatches),
        mismatches=mismatches,
        sourceEnvironmentEquivalence=False,
    )
    (tmp_path / "evidence.json").write_text(
        json.dumps(evidence, indent=2), encoding="utf-8"
    )
    assert len(actual) == len(expected)
    assert not mismatches, mismatches[:10]


def test_original_metal_division_flush_profile(tmp_path, monkeypatch):
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 for native division rounding")
    if sys.platform != "darwin":
        pytest.skip("The original Metal control requires macOS")
    source = """#include <metal_stdlib>
using namespace metal;
kernel void divide_control(const device uint* values [[buffer(0)]],
                           device uint* results [[buffer(1)]],
                           uint i [[thread_position_in_grid]]) {
    float a = as_type<float>(values[2u * i]);
    float b = as_type<float>(values[2u * i + 1u]);
    results[4u + i] = as_type<uint>(a / b);
}
"""
    flags = ("-std=metal3.1", "-fno-fast-math")
    monkeypatch.setattr(
        native_math, "_compile", functools.partial(_compile, metal_compile_flags=flags)
    )
    pairs = _pairs()
    expected = (
        [GUARD] * 4 + [_oracle(*pair, flush=True) for pair in pairs] + [GUARD] * 4
    )
    (tmp_path / "expected.json").write_text(json.dumps(expected), encoding="utf-8")
    actual, evidence = _dispatch(
        tmp_path,
        "metal",
        source,
        pairs,
        len(expected),
        entry="divide_control",
        initial_output=[GUARD] * len(expected),
    )
    mismatches = []
    for i, (want, got) in enumerate(zip(expected, actual)):
        payload = 4 <= i < len(expected) - 4
        both_nan = (want & 0x7FFFFFFF) > 0x7F800000 and (got & 0x7FFFFFFF) > 0x7F800000
        if got != want and not (payload and both_nan):
            mismatches.append({"index": i, "expected": want, "actual": got})
    evidence.update(
        profile="rne-flush-before-rounding",
        compilerFlags=list(flags),
        originalSource=True,
        nanComparison="classification-only",
        guardCount=8,
        mismatchCount=len(mismatches),
        mismatches=mismatches,
    )
    (tmp_path / "evidence.json").write_text(
        json.dumps(evidence, indent=2), encoding="utf-8"
    )
    assert len(actual) == len(expected)
    assert not mismatches, mismatches[:10]


def test_ci_requires_native_division_without_new_runners():
    from tools import ci_coverage

    workflow = (
        Path(__file__).resolve().parents[2]
        / ".github/workflows/demo-project-testing.yml"
    ).read_text()
    step = ci_coverage.workflow_job_step_section(
        workflow, "portable-host", "Validate binary32 arithmetic"
    )
    assert "if:" not in step
    assert f'{REQUIRE_ENV}: "1"' in step
    assert "tests/test_translator/test_division_math.py" in step
    assert "tests/test_translator/test_fused_math.py" in step
    assert "tests/test_translator/test_metal_fma.py" in step
    assert "--basetemp=" in step and "--junitxml=" in step
    assert "pytest -q -n auto" in step


def test_ci_does_not_duplicate_native_division_checks():
    workflow = (
        Path(__file__).resolve().parents[2]
        / ".github/workflows/demo-project-testing.yml"
    ).read_text()
    assert workflow.count("tests/test_translator/test_division_math.py") == 1
    assert workflow.count(f'{REQUIRE_ENV}: "1"') == 1

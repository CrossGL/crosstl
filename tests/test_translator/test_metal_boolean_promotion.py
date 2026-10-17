"""Preserve Metal scalar boolean integral promotion before target lowering."""

import json
import math
import os
import struct
import sys

import pytest

from crosstl import translate
from tests.test_translator.test_metal_builtin_ownership import _compile
from tests.test_translator.test_metal_precise_trig import GUARD, _bits, _execute

REQUIRE_ENV = "CROSTL_REQUIRE_METAL_BOOLEAN_PROMOTION"
SOURCE = """#include <metal_stdlib>
using namespace metal;
bool record(thread uint& count, bool value) { count += 1; return value; }
float classify(float x) { return (x > 0.0f) - (x < 0.0f); }
kernel void promoted_values(device const uint* values [[buffer(0)]],
                            device uint* results [[buffer(1)]],
                            uint i [[thread_position_in_grid]]) {
    bool a = (values[i] & 1u) != 0u;
    bool b = (values[i] & 2u) != 0u;
    uint count = 0;
    int difference = record(count, a) - record(count, b);
    auto nested = (a + b) * (a - b);
    bool converted = a - b;
    bool compound = a;
    compound -= b;
    results[11 * i] = as_type<uint>(difference);
    results[11 * i + 1] = as_type<uint>(int(a + b));
    results[11 * i + 2] = as_type<uint>(int(nested));
    results[11 * i + 3] = as_type<uint>(int(a << b));
    results[11 * i + 4] = as_type<uint>(int(a | b));
    results[11 * i + 5] = as_type<uint>(int(a ^ b));
    results[11 * i + 6] = as_type<uint>(int(a & b));
    results[11 * i + 7] = uint(converted);
    results[11 * i + 8] = uint(compound);
    results[11 * i + 9] = as_type<uint>(classify(as_type<float>(values[i])));
    results[11 * i + 10] = count;
}
"""


def _translate(tmp_path, source, target):
    path = tmp_path / "promotion.metal"
    path.write_text(source, encoding="utf-8")
    return translate(str(path), backend=target, format_output=False)


@pytest.mark.parametrize("target", ["directx", "opengl", "metal"])
def test_metal_boolean_promotion_compiles(tmp_path, target):
    generated = _translate(tmp_path, SOURCE, target)
    _compile(generated, target, tmp_path)


@pytest.mark.parametrize(
    "operator", ["+", "-", "*", "/", "%", "&", "|", "^", "<<", ">>"]
)
def test_scalar_boolean_operands_are_promoted(tmp_path, operator):
    source = f"int calculate(bool a, bool b) {{ return a {operator} b; }}"
    generated = _translate(tmp_path, source, "crossgl")
    assert f"return int(a) {operator} int(b);" in generated


@pytest.mark.parametrize("operator", ["&&", "||", "==", "!="])
def test_logical_and_comparison_operands_are_not_promoted(tmp_path, operator):
    generated = _translate(
        tmp_path,
        f"bool calculate(bool a, bool b) {{ return a {operator} b; }}",
        "crossgl",
    )
    assert f"return a {operator} b;" in generated
    assert "int(a)" not in generated


def test_promotion_preserves_source_overload_and_alias(tmp_path):
    generated = _translate(
        tmp_path,
        """
        using Flag = bool;
        struct Value { int x; };
        int operator-(Value a, bool b) { return a.x + int(b); }
        int own(Value a, bool b) { return a - b; }
        int builtin(Flag a, Flag b) { return a - b; }
        float mixed(bool a, float b) { return a - b; }
        uint unsigned_mix(bool a, uint b) { return a - b; }
    """,
        "crossgl",
    )
    assert "return int(a) - int(b);" in generated
    assert generated.count("return int(a) - b;") == 2
    assert "return a - int(b);" not in generated


def _inputs():
    # Low mantissa bits cover all boolean pairs without subnormal comparisons.
    values = [-100.0, -0.75, -0.5, -0.125, -0.0, 0.0, 0.125, 0.5, 0.75, 100.0]
    words = [_bits(value) for value in values]
    for value in (-1.0, 1.0):
        words.extend(_bits(value) + low for low in range(4))
    return words + [0x7F800000, 0xFF800000, 0x7FC12345, 0xFFC12345]


def _expected(words):
    results = []
    for word in words:
        a, b = int(bool(word & 1)), int(bool(word & 2))
        value = struct.unpack("<f", struct.pack("<I", word))[0]
        sign = 0.0 if math.isnan(value) else float((value > 0) - (value < 0))
        row = [
            a - b,
            a + b,
            (a + b) * (a - b),
            a << b,
            a | b,
            a ^ b,
            a & b,
            int(bool(a - b)),
            int(bool(a - b)),
            _bits(sign),
            2,
        ]
        results.extend(number & 0xFFFFFFFF for number in row)
    return results + GUARD


def _check(actual, expected):
    assert actual == expected, "promoted arithmetic or output guard changed"
    return 0


@pytest.mark.parametrize("index", [0, 1, 9, 10, -1])
def test_boolean_promotion_verifier_rejects_corruption(index):
    expected = _expected(_inputs())
    actual = list(expected)
    actual[index] ^= 1
    with pytest.raises(AssertionError):
        _check(actual, expected)


def test_metal_boolean_promotion_executes(tmp_path):
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 for required promotion execution")
    target = {"win32": "directx", "linux": "opengl", "darwin": "metal"}[sys.platform]
    inputs = _inputs()
    expected = _expected(inputs)
    (tmp_path / "inputs.bin").write_bytes(struct.pack(f"<{len(inputs)}I", *inputs))
    (tmp_path / "expected.bin").write_bytes(
        struct.pack(f"<{len(expected)}I", *expected)
    )
    generated = _translate(tmp_path, SOURCE, target)
    records = {
        "generated": _execute(
            tmp_path / "generated",
            target,
            generated,
            inputs,
            expected,
            metal_entry="promoted_values",
            check_outputs=_check,
        )
    }
    if target == "metal":
        records["original"] = _execute(
            tmp_path / "original",
            target,
            SOURCE,
            inputs,
            expected,
            metal_entry="promoted_values",
            check_outputs=_check,
            metal_compile_flags=("-fno-fast-math",),
        )
    (tmp_path / "evidence.json").write_text(
        json.dumps(
            {
                "target": target,
                "inputCount": len(inputs),
                "records": records,
            },
            indent=2,
        )
    )

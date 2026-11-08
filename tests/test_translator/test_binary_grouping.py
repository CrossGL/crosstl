"""Arithmetic grouping survives source conversion and native target generation."""

import json
import os
import struct
import sys
from pathlib import Path

import pytest

import crosstl.translator
from crosstl import translate
from crosstl.translator.codegen.metal_codegen import MetalCodeGen
from tests.test_backend.test_metal.test_codegen import (
    convert,
    crossgl_expression_tree,
    crossgl_local_initializers,
    find_crossgl_function,
)
from tests.test_translator.test_metal_builtin_ownership import _compile
from tests.test_translator.test_metal_precise_trig import GUARD, _execute

REQUIRE_ENV = "CROSTL_REQUIRE_BINARY_GROUPING"
CASES = (
    (0x3F800000, 0x4B800000, 0xCB800000),
    (0x3F800001, 0x3F7FFFFF, 0x3F7FFFFF),
    (0xBF800000, 0xCB800000, 0x4B800000),
    (0xBF800001, 0x3F7FFFFF, 0x3F7FFFFF),
    (0x80000000, 0x80000000, 0x80000000),
    (0, 0x80000000, 0x3F800000),
)
EXPRESSIONS = ("a + (b + c)", "(a + b) + c", "a * (b * c)", "(a * b) * c")


@pytest.mark.parametrize(
    "dtype,operator",
    [
        (dtype, operator)
        for dtype in ("float", "float2", "int", "uint")
        for operator in ("+", "*", "&", "|", "^")
        if dtype in {"int", "uint"} or operator in {"+", "*"}
    ],
)
def test_source_conversion_preserves_arithmetic_trees(dtype, operator):
    source = f"""{dtype} grouped({dtype} a, {dtype} b, {dtype} c) {{
        {dtype} right = a {operator} (b {operator} c);
        {dtype} left = (a {operator} b) {operator} c;
        return right;
    }}"""
    parsed = crosstl.translator.parse(convert(source))
    values = crossgl_local_initializers(find_crossgl_function(parsed, "grouped"))
    assert crossgl_expression_tree(values["right"]) == (
        "binary",
        operator,
        "a",
        ("binary", operator, "b", "c"),
    )
    assert crossgl_expression_tree(values["left"]) == (
        "binary",
        operator,
        ("binary", operator, "a", "b"),
        "c",
    )


@pytest.mark.parametrize(
    "dtype,operator",
    [
        (dtype, operator)
        for dtype in ("float", "vec2", "int", "uint")
        for operator in ("+", "*", "&", "|", "^")
        if dtype in {"int", "uint"} or operator in {"+", "*"}
    ],
)
def test_metal_target_preserves_arithmetic_trees(dtype, operator):
    source = f"""shader Grouped {{
        {dtype} right({dtype} a, {dtype} b, {dtype} c) {{
            return a {operator} (b {operator} c);
        }}
        {dtype} left({dtype} a, {dtype} b, {dtype} c) {{
            return (a {operator} b) {operator} c;
        }}
    }}"""
    generated = MetalCodeGen().generate(crosstl.translator.parse(source))
    assert f"return a {operator} (b {operator} c);" in generated
    assert f"return a {operator} b {operator} c;" in generated


def _source(width):
    dtype = "float" if width == 1 else f"float{width}"
    statements = []
    for index, name in enumerate("abc"):
        components = [
            f"as_type<float>(values[{3 * width}u * i + {3 * lane + index}u])"
            for lane in range(width)
        ]
        value = components[0] if width == 1 else f"{dtype}({', '.join(components)})"
        statements.append(f"    {dtype} {name} = {value};")
    for index, expression in enumerate(EXPRESSIONS):
        statements.append(f"    {dtype} result{index} = {expression};")
        for lane in range(width):
            component = f".{'xyzw'[lane]}" if width > 1 else ""
            statements.append(
                f"    results[{4 * width}u * i + {index * width + lane}u] = "
                f"as_type<uint>(result{index}{component});"
            )
    return f"""#include <metal_stdlib>
using namespace metal;
kernel void grouping(device const uint* values [[buffer(0)]],
                     device uint* results [[buffer(1)]],
                     uint i [[thread_position_in_grid]]) {{
    if (i >= {len(CASES)}u) return;
{chr(10).join(statements)}
}}
"""


def _bits(value):
    return struct.unpack("<I", struct.pack("<f", value))[0]


def _float(word):
    return struct.unpack("<f", struct.pack("<I", word))[0]


def _expected(triple):
    a, b, c = map(_float, triple)
    # Python binary64 exactly represents these binary32 sums and products;
    # explicitly round each source operation to binary32 before its parent.
    return (
        _bits(a + _float(_bits(b + c))),
        _bits(_float(_bits(a + b)) + c),
        _bits(a * _float(_bits(b * c))),
        _bits(_float(_bits(a * b)) * c),
    )


def _inputs_and_expected(width):
    inputs, expected = [], []
    for index in range(len(CASES)):
        lanes = [CASES[(index + lane) % len(CASES)] for lane in range(width)]
        inputs.extend(word for triple in lanes for word in triple)
        results = [_expected(triple) for triple in lanes]
        expected.extend(
            results[lane][result] for result in range(4) for lane in range(width)
        )
    return inputs, expected + GUARD


def _translate_saved(root, source, target):
    path = root / "original.metal"
    path.write_text(source, encoding="utf-8")
    saved = root / "saved.cgl"
    saved.write_text(
        translate(str(path), backend="cgl", format_output=False), encoding="utf-8"
    )
    generated = translate(str(path), backend=target, format_output=False)
    assert generated == translate(str(saved), backend=target, format_output=False)
    return source, generated


@pytest.mark.parametrize("target", ["metal", "directx", "opengl"])
@pytest.mark.parametrize("width", [1, 2, 3, 4])
def test_grouped_arithmetic_translates_and_compiles(tmp_path, target, width):
    _source_text, generated = _translate_saved(tmp_path, _source(width), target)
    _compile(
        generated,
        target,
        tmp_path,
        metal_compile_flags=("-fno-fast-math",),
        directx_compile_flags=("-Gis",),
    )


def _check(actual, expected):
    assert actual == expected, [
        (index, hex(found), hex(wanted))
        for index, (found, wanted) in enumerate(zip(actual, expected))
        if found != wanted
    ][:20]
    return 0


def _run_native(
    tmp_path, source, inputs, expected, *, metal_compile_flags=("-fno-fast-math",)
):
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 for required native binary grouping")
    target = {"darwin": "metal", "win32": "directx", "linux": "opengl"}[sys.platform]
    source, generated = _translate_saved(tmp_path, source, target)
    for name, words in (("inputs", inputs), ("expected", expected)):
        (tmp_path / f"{name}.bin").write_bytes(struct.pack(f"<{len(words)}I", *words))
    records = {}
    for name, code in (("generated", generated), ("original", source)):
        if name == "original" and target != "metal":
            continue
        records[name] = _execute(
            tmp_path / name,
            target,
            code,
            inputs,
            expected,
            metal_entry="grouping",
            check_outputs=_check,
            metal_compile_flags=metal_compile_flags,
            directx_compile_flags=("-Gis",),
        )
    (tmp_path / "evidence.json").write_text(
        json.dumps(
            {
                "target": target,
                "inputCount": len(inputs),
                "resultCount": len(expected) - len(GUARD),
                "guardCount": len(GUARD),
                "records": records,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


@pytest.mark.parametrize("width", [1, 2, 3, 4])
def test_grouped_arithmetic_executes_natively(tmp_path, width):
    inputs, expected = _inputs_and_expected(width)
    _run_native(tmp_path, _source(width), inputs, expected)


def _mixed_bitwise_source(operator):
    return f"""#include <metal_stdlib>
using namespace metal;
kernel void grouping(device const uint* values [[buffer(0)]],
                     device uint* results [[buffer(1)]],
                     uint i [[thread_position_in_grid]]) {{
    if (i != 0u) return;
    ulong a = ulong(values[0]) << 32u;
    int b = as_type<int>(values[1]);
    uint c = values[2];
    ulong right = a {operator} (b {operator} c);
    ulong left = (a {operator} b) {operator} c;
    results[0] = uint(right);
    results[1] = uint(right >> 32u);
    results[2] = uint(left);
    results[3] = uint(left >> 32u);
}}
"""


@pytest.mark.parametrize("operator", ["|", "^"])
@pytest.mark.parametrize("target", ["metal", "directx", "opengl"])
def test_mixed_bitwise_grouping_compiles(tmp_path, operator, target):
    _source_text, generated = _translate_saved(
        tmp_path, _mixed_bitwise_source(operator), target
    )
    _compile(generated, target, tmp_path)


@pytest.mark.parametrize("operator,high_word", [("|", 0xFFFFFFFF), ("^", 0xFFFFFFFE)])
def test_mixed_bitwise_grouping_retains_conversion_order(tmp_path, operator, high_word):
    expected = [0xFFFFFFFF, 1, 0xFFFFFFFF, high_word] + GUARD
    _run_native(tmp_path, _mixed_bitwise_source(operator), [1, 0xFFFFFFFF, 0], expected)


# Groupings C++ precedence already implies but clang's -Wall rejects under
# -Werror (-Wlogical-op-parentheses, -Wbitwise-op-parentheses, -Wparentheses),
# and a conditional condition, which `?:`'s right associativity regroups.
WARNED_GROUPINGS = (
    ("bool", "bool", "a || (b && c)"),
    ("bool", "bool", "(a && b) || c"),
    ("int", "int", "a | (b & c)"),
    ("int", "int", "(a & b) | c"),
    ("int", "int", "a | (b ^ c)"),
    ("int", "int", "(a ^ b) | c"),
    ("int", "int", "a ^ (b & c)"),
    ("int", "int", "(a & b) ^ c"),
    ("int", "bool", "(a == b) & c"),
    ("int", "bool", "c | (a < b)"),
    ("int", "bool", "(a != b) ^ c"),
    ("int", "int", "(a ? b : 0) ? 5 : 6"),
)


@pytest.mark.parametrize("operand,result,expression", WARNED_GROUPINGS)
def test_metal_target_keeps_groupings_clang_warns_about(operand, result, expression):
    third = "bool" if result == "bool" and operand != result else operand
    source = f"""shader Grouped {{
        {result} grouped({operand} a, {operand} b, {third} c) {{
            return {expression};
        }}
    }}"""
    generated = MetalCodeGen().generate(crosstl.translator.parse(source))
    assert f"return {expression};" in generated


def test_metal_target_omits_redundant_logical_and_arithmetic_parentheses():
    source = """shader Grouped {
        bool logical(bool a, bool b, bool c) { return (a || b) || (c && a); }
        int arithmetic(int a, int b, int c) { return (a * b) + (c << 1); }
        int nested(int a, int b, int c) { return a ? (b ? 7 : 8) : (c ? 9 : 10); }
    }"""
    generated = MetalCodeGen().generate(crosstl.translator.parse(source))
    assert "return a || b || (c && a);" in generated
    assert "return a * b + (c << 1);" in generated
    assert "return a ? b ? 7 : 8 : c ? 9 : 10;" in generated


LOGICAL_CASES = (
    (0, 0, 0),
    (1, 0, 4),
    (2, 3, 4),
    (0, 3, 4),
    (1, 5, 0),
    (2, 0, 0),
    (0xFFFFFFFF, 0x0F0F0F0F, 0x00FF00FF),
    (0, 7, 4),
)
MASK = 0xFFFFFFFF
LOGICAL_EXPRESSIONS = (
    ("(x ? y : 0u) ? 5u : 6u", lambda x, y, z: 5 if (y if x else 0) else 6),
    ("x ? (y ? 7u : 8u) : 9u", lambda x, y, z: (7 if y else 8) if x else 9),
    (
        "(x > 1u || (y == 3u && z == 4u)) ? 1u : 0u",
        lambda x, y, z: int(x > 1 or (y == 3 and z == 4)),
    ),
    (
        "((x > 1u || y == 3u) && z == 4u) ? 1u : 0u",
        lambda x, y, z: int((x > 1 or y == 3) and z == 4),
    ),
    ("(x ^ y) | (y & z)", lambda x, y, z: (x ^ y) | (y & z)),
    ("x ^ (y & z)", lambda x, y, z: x ^ (y & z)),
    ("(x | y) & z", lambda x, y, z: (x | y) & z),
    ("uint((x == y) | (z != 0u))", lambda x, y, z: int(x == y or z != 0)),
    ("(x + y) << (z & 3u)", lambda x, y, z: ((x + y) << (z & 3)) & MASK),
)


def _logical_source():
    count = len(LOGICAL_EXPRESSIONS)
    statements = "\n".join(
        f"    results[{count}u * i + {index}u] = {expression};"
        for index, (expression, _) in enumerate(LOGICAL_EXPRESSIONS)
    )
    return f"""#include <metal_stdlib>
using namespace metal;
kernel void grouping(device const uint* values [[buffer(0)]],
                     device uint* results [[buffer(1)]],
                     uint i [[thread_position_in_grid]]) {{
    if (i >= {len(LOGICAL_CASES)}u) return;
    uint x = values[3u * i];
    uint y = values[3u * i + 1u];
    uint z = values[3u * i + 2u];
{statements}
}}
"""


def _logical_inputs_and_expected():
    inputs = [word for case in LOGICAL_CASES for word in case]
    expected = [
        evaluate(*case) & MASK
        for case in LOGICAL_CASES
        for _, evaluate in LOGICAL_EXPRESSIONS
    ]
    return inputs, expected + GUARD


@pytest.mark.parametrize("target", ["metal", "directx", "opengl"])
def test_logical_and_bitwise_grouping_compiles_warning_clean(tmp_path, target):
    _source_text, generated = _translate_saved(tmp_path, _logical_source(), target)
    if target == "metal":
        assert "(x ? y : 0u) ? 5u : 6u" in generated
        assert "x > 1u || (y == 3u && z == 4u)" in generated
        assert "(x ^ y) | (y & z)" in generated
    _compile(
        generated,
        target,
        tmp_path,
        metal_compile_flags=("-fno-fast-math", "-Wall", "-Wextra"),
    )


def test_logical_and_bitwise_grouping_executes_natively(tmp_path):
    inputs, expected = _logical_inputs_and_expected()
    _run_native(
        tmp_path,
        _logical_source(),
        inputs,
        expected,
        metal_compile_flags=("-fno-fast-math", "-Wall", "-Wextra"),
    )


def test_logical_oracle_distinguishes_regrouped_conditions():
    _inputs, expected = _logical_inputs_and_expected()
    count = len(LOGICAL_EXPRESSIONS)
    # x=1, y=0: the regrouped `x ? y : (0 ? 5 : 6)` would yield 0.
    assert expected[count * 1] == 6
    # x=2, z=0: `(x > 1 || y == 3) && z == 4` would yield 0.
    assert expected[count * 5 + 2] == 1 and expected[count * 5 + 3] == 0


def test_grouping_oracle_distinguishes_associations():
    assert _expected(CASES[0])[:2] == (0x3F800000, 0)
    assert _expected(CASES[1])[2:] == (0x3F800000, 0x3F7FFFFF)
    assert _expected(CASES[2])[:2] == (0xBF800000, 0)
    assert _expected(CASES[3])[2:] == (0xBF800000, 0xBF7FFFFF)
    assert _expected(CASES[4]) == (0x80000000,) * 4
    assert _expected(CASES[5]) == (0x3F800000, 0x3F800000, 0x80000000, 0x80000000)


@pytest.mark.parametrize("index", [0, 1, 2, 3, -1])
def test_grouping_oracle_rejects_value_and_guard_changes(index):
    _inputs, expected = _inputs_and_expected(1)
    actual = list(expected)
    actual[index] ^= 1
    with pytest.raises(AssertionError):
        _check(actual, expected)


def test_ci_requires_binary_grouping_without_an_additional_runner():
    workflow = (
        Path(__file__).resolve().parents[2]
        / ".github/workflows/demo-project-testing.yml"
    ).read_text()
    step = workflow.split("      - name: Validate Metal builtin ownership\n", 1)[1]
    step = step.split("      - name:", 1)[0]
    assert "if:" not in step and "continue-on-error" not in step
    assert f'{REQUIRE_ENV}: "1"' in step
    assert "tests/test_translator/test_binary_grouping.py" in step
    assert "--timeout-seconds 360" in step
    assert "pytest -q -n auto" in step

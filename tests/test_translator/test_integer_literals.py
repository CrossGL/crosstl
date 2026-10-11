"""Integer source types survive canonical translation and native execution."""

import os
import sys
from pathlib import Path

import pytest

from crosstl.backend.Metal.MetalCrossGLCodeGen import MetalToCrossGLConverter
from crosstl.project import build_native_loader_dispatch_request
from crosstl.translator import parse
from crosstl.translator.ast import LiteralNode
from crosstl.translator.codegen.directx_codegen import HLSLCodeGen
from crosstl.translator.integer_literals import (
    IntegerLiteralError,
    integer_literal_parts,
)
from crosstl.translator.lexer import Lexer
from tests.test_translator.test_boolean_buffer_runtime import _bound_values
from tests.test_translator.test_loop_updates import _execute
from tests.test_translator.test_metal_builtin_ownership import _compile
from tests.test_translator.test_software_subgroup_product import _package

REQUIRE_ENV = "CROSTL_REQUIRE_INTEGER_LITERAL_RUNTIME"
LITERAL_CASES = [
    ("0", "int"),
    ("2147483647", "int"),
    ("2147483648", "int64_t"),
    ("4294967295", "int64_t"),
    ("4294967296u", "uint64_t"),
    ("9223372036854775807", "int64_t"),
    ("18446744073709551615ul", "uint64_t"),
    ("0x7fffffff", "int"),
    ("0x80000000", "uint"),
    ("0xffffffff", "uint"),
    ("0x100000000", "int64_t"),
    ("0x8000000000000000", "uint64_t"),
    ("0xffffffffffffffffL", "uint64_t"),
    ("0b1", "int"),
    ("0o17", "int"),
]
SUFFIXES = (
    "l",
    "L",
    "ll",
    "LL",
    "ul",
    "uL",
    "Ul",
    "UL",
    "lu",
    "lU",
    "Lu",
    "LU",
    "ull",
    "uLL",
    "Ull",
    "ULL",
    "llu",
    "llU",
    "LLu",
    "LLU",
)


@pytest.mark.parametrize("spelling,type_name", LITERAL_CASES)
def test_integer_literal_types_follow_radix_and_representability(spelling, type_name):
    digits, actual = integer_literal_parts(spelling)
    assert actual == type_name
    ast = parse(f"shader Literal {{ void main() {{ inspect({spelling}); }} }}")
    literal = next(node for node in ast.walk() if isinstance(node, LiteralNode))
    assert literal.literal_type.name == type_name
    assert literal.value == int(
        digits, 0 if digits.startswith(("0x", "0b", "0o")) else 10
    )


@pytest.mark.parametrize("suffix", SUFFIXES)
@pytest.mark.parametrize("digits", ("1", "0x1", "0b1", "0o1"))
def test_wide_integer_suffixes_are_single_typed_tokens(digits, suffix):
    spelling = digits + suffix
    tokens = Lexer(spelling).tokens
    assert tokens[0][1] == spelling
    expected = "uint64_t" if "u" in suffix.lower() else "int64_t"
    assert integer_literal_parts(spelling)[1] == expected


@pytest.mark.parametrize(
    "spelling",
    (
        "1uu",
        "1lL",
        "1Ll",
        "1lll",
        "1ulu",
        "9223372036854775808l",
        "18446744073709551616ul",
        "0x10000000000000000",
        "9223372036854775808",
    ),
)
def test_invalid_or_unrepresentable_integer_literals_are_rejected(spelling):
    with pytest.raises(IntegerLiteralError):
        parse(f"shader Literal {{ void main() {{ inspect({spelling}); }} }}")


@pytest.mark.parametrize(
    "spelling,canonical,type_name",
    (
        ("1LL", "1l", "int64_t"),
        ("1LU", "1ul", "uint64_t"),
        ("2'147'483'648", "2147483648l", "int64_t"),
        ("0xffffffff", "0xffffffffu", "uint"),
        ("037777777777", "0o37777777777u", "uint"),
        ("0xffffffffffffffffl", "0xfffffffffffffffful", "uint64_t"),
    ),
)
def test_metal_literal_normalization_and_overload_typing_agree(
    spelling, canonical, type_name
):
    converter = MetalToCrossGLConverter()
    assert converter.normalize_literal_string(spelling) == canonical
    assert converter.metal_literal_string_type(spelling) == type_name


@pytest.mark.parametrize("spelling", ("08", "09ul", "1lL", "1uu"))
def test_metal_rejects_invalid_integer_spellings(spelling):
    converter = MetalToCrossGLConverter()
    with pytest.raises(IntegerLiteralError):
        converter.normalize_literal_string(spelling)


CASES = {
    "signed-width": ("(1l << 40) + 4294967297l", 2**40 + 4294967297),
    "unsigned-width": ("(1ul << 40) + 4294967297ul", 2**40 + 4294967297),
    "signed-shift": ("(-8LL >> 2) + 4l", 2),
    "unsigned-shift": ("0xffffffffffffffffUL >> 60", 15),
    "decimal-type": ("2147483648 >> 31", 1),
    "hex-type": ("0xffffffff >> 31", 1),
    "octal-type": ("037777777777 >> 31", 1),
    "explicit-conversion": ("long(4294967297ul) + 1l", 4294967298),
    "signed-overload": ("select_width(1l)", 64),
    "unsigned-overload": ("select_width(1ul)", 65),
    "decimal-overload": ("select_width(2147483648)", 64),
    "hex-overload": ("select_width(0xffffffff)", 33),
    "dynamic-signed-width": ("(1l << shifts[0]) + 4294967297l", 2**40 + 4294967297),
    "dynamic-unsigned-width": ("(1ul << shifts[0]) + 4294967297ul", 2**40 + 4294967297),
    "dynamic-unsigned-wrap": ("0xffffffff + shifts[1]", 0),
    "binary-width": ("0b1ULL << 40", 2**40),
    "separated-decimal": ("4'294'967'297l + 1l", 4294967298),
    "unsigned-limit": ("numeric_limits<ulong>::max()", 2**64 - 1),
    "signed-limit": ("numeric_limits<long>::lowest()", 2**63),
    "signed-maximum": ("numeric_limits<long>::max()", 2**63 - 1),
    "unsigned-negation-shift": ("(-9223372036854775808ul >> 63)", 1),
    "unsigned-negation-comparison": ("(-9223372036854775808ul < 0)", 0),
}


def _source(case):
    expression, _ = CASES[case]
    parameter = (
        ", constant uint* shifts [[buffer(1)]]" if "shifts[" in expression else ""
    )
    return f"""#include <metal_stdlib>
using namespace metal;
ulong select_width(int value) {{ return ulong(value - value) + 32ul; }}
ulong select_width(uint value) {{ return ulong(value - value) + 33ul; }}
ulong select_width(long value) {{ return ulong(value - value) + 64ul; }}
ulong select_width(ulong value) {{ return value - value + 65ul; }}
kernel void integer_widths(device ulong* results [[buffer(0)]]{parameter}) {{
    results[1] = ulong({expression});
}}
"""


def _request(root, target, case):
    _, descriptor, package = _package(
        root, target, "uint", (1, 1, 1), source=_source(case), software_subgroups=False
    )
    expected = [123456789, CASES[case][1], 123456789]
    inputs = {
        "results": {
            "dtype": "uint64",
            "shape": [3],
            "values": [123456789, 0, 123456789],
        }
    }
    outputs = {"results": {"dtype": "uint64", "shape": [3], "values": expected}}
    if "shifts[" in CASES[case][0]:
        inputs["shifts"] = {"dtype": "uint32", "shape": [2], "values": [40, 1]}
    request = build_native_loader_dispatch_request(
        descriptor,
        package,
        _bound_values(descriptor, inputs),
        _bound_values(descriptor, outputs),
        {"workgroupCount": [1, 1, 1], "workgroupSize": [1, 1, 1]},
        expected_target=target,
    )
    assert not request.execution_plan.diagnostics
    return request, _bound_values(descriptor, outputs)


@pytest.mark.parametrize("target", ("metal", "opengl", "directx"))
@pytest.mark.parametrize("case", CASES)
def test_integer_widths_package_and_compile(tmp_path, target, case):
    request, _ = _request(tmp_path, target, case)
    _compile(request.artifact_path.read_text(), target, tmp_path)


@pytest.mark.parametrize("case", CASES)
def test_integer_widths_execute(tmp_path, case):
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 for required literal execution")
    target = {"linux": "opengl", "darwin": "metal", "win32": "directx"}[sys.platform]
    request, expected = _request(tmp_path, target, case)
    options = (
        {
            "original_source": _source(case),
            "original_entry": "integer_widths",
            "metal_compile_flags": ("-Wall", "-Wextra", "-Werror"),
        }
        if target == "metal"
        else {}
    )
    _execute(request, expected, tmp_path, **options)


def test_integer_widths_are_required_in_native_ci():
    workflow = Path(".github/workflows/demo-project-testing.yml").read_text()
    step = workflow.split("- name: Validate collective helper arguments", 1)[1].split(
        "- name:", 1
    )[0]
    assert f'{REQUIRE_ENV}: "1"' in step
    assert "tests/test_translator/test_integer_literals.py" in step


@pytest.mark.parametrize("operator", ["<<", ">>"])
@pytest.mark.parametrize("count_type", ["int", "uint", "int64_t", "uint64_t"])
@pytest.mark.parametrize("operand", ["1", "-8", "~0", "(5 + 3)", "(flag ? 8 : 16)"])
def test_hlsl_literal_shift_operands_have_independent_types(
    operator, count_type, operand
):
    source = f"shader Test {{ int shift({count_type} count, bool flag) {{ return {operand} {operator} count; }} }}"
    generated = HLSLCodeGen().generate(parse(source))
    assert f") {operator} count)" in generated
    assert "int(" in generated
    assert f"{count_type}(count)" not in generated
    assert "uint(1)" not in generated and "1u" not in generated


@pytest.mark.parametrize("count_type", ["int", "uint", "int64_t", "uint64_t"])
def test_hlsl_shift_result_keeps_left_signedness_in_wider_arithmetic(count_type):
    source = f"shader Test {{ int64_t shift(int value, {count_type} count) {{ return (value >> count) / 3l; }} }}"
    generated = HLSLCodeGen().generate(parse(source))
    assert "int64_t((value >> count))" in generated
    assert "/ 3ll" in generated


SHIFT_EXPRESSIONS = (
    "1 << (count % 31u)",
    "1 >> count",
    "-8 >> count",
    "~0 >> count",
    "(5 + 3) >> count",
    "(tid % 2u == 0u ? -8 : -16) >> count",
    "tid & ((1 << (count % 31u)) - 1)",
    "(-8 >> count) / 3l",
    "(value >> count) / 3l",
    "(value >> count) < 0",
    "0xffffffffu >> count",
    "1l << count",
    "1ul << count",
    "1 << int(count % 31u)",
    "1 << long(count % 31u)",
    "1 << ulong(count % 31u)",
    "-8 >> long(count)",
    "-8 >> ulong(count)",
    "1 << (counter++ % 31u)",
    "counter",
)


def _shift_source():
    stores = "\n".join(
        f"    results[5u + tid * {len(SHIFT_EXPRESSIONS)}u + {index}u] = as_type<uint>(int({expression}));"
        for index, expression in enumerate(SHIFT_EXPRESSIONS)
    )
    return f"""#include <metal_stdlib>
using namespace metal;
kernel void literal_shifts(device uint* results [[buffer(0)]],
                           constant uint* counts [[buffer(1)]],
                           uint tid [[thread_position_in_grid]]) {{
    if (tid >= 32u) {{ return; }}
    uint count = counts[tid];
    uint counter = count;
    int value = -8;
{stores}
}}
"""


def _shift_values():
    guard = 0xDEADBEEF
    expected = [guard] * 5
    for count in range(32):
        power = 2 ** (count % 31)
        signed = -8 // (2**count)
        quotient = -(abs(signed) // 3)
        values = [
            power,
            1 >> count,
            signed,
            -1,
            8 >> count,
            (-8 if count % 2 == 0 else -16) // (2**count),
            count & (power - 1),
            quotient,
            quotient,
            1,
            (2**32 - 1) >> count,
            2**count,
            2**count,
            power,
            power,
            power,
            signed,
            signed,
            power,
            count + 1,
        ]
        assert len(values) == len(SHIFT_EXPRESSIONS)
        expected.extend(value & 0xFFFFFFFF for value in values)
    expected.extend([guard] * 7)
    inputs = {
        "counts": {"dtype": "uint32", "shape": [32], "values": list(range(32))},
        "results": {
            "dtype": "uint32",
            "shape": [len(expected)],
            "values": [guard] * len(expected),
        },
    }
    outputs = {
        "results": {"dtype": "uint32", "shape": [len(expected)], "values": expected}
    }
    return inputs, outputs


def _shift_request(root, target):
    source, descriptor, package = _package(
        root,
        target,
        "uint",
        (32, 1, 1),
        source=_shift_source(),
        software_subgroups=False,
    )
    inputs, outputs = _shift_values()
    outputs = _bound_values(descriptor, outputs)
    request = build_native_loader_dispatch_request(
        descriptor,
        package,
        _bound_values(descriptor, inputs),
        outputs,
        {"workgroupCount": [2, 1, 1], "workgroupSize": [32, 1, 1]},
        expected_target=target,
    )
    assert not request.execution_plan.diagnostics
    return source, request, outputs


@pytest.mark.parametrize("target", ["metal", "directx", "opengl"])
def test_literal_shifts_compile_with_runtime_counts(tmp_path, target):
    _, request, _ = _shift_request(tmp_path, target)
    generated = request.artifact_path.read_text()
    assert generated.count("counter++") == 1
    _compile(generated, target, tmp_path)


def test_literal_shifts_execute_with_source_signedness(tmp_path):
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 for required literal execution")
    target = {"linux": "opengl", "darwin": "metal", "win32": "directx"}[sys.platform]
    source, request, outputs = _shift_request(tmp_path, target)
    _execute(
        request,
        outputs,
        tmp_path,
        original_source=source if target == "metal" else None,
        original_entry="literal_shifts",
    )

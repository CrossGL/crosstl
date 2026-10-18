"""Nested unary expressions preserve values, evaluation counts and updates."""

import json
import os
import struct
import sys
from pathlib import Path

import pytest

from crosstl import translate
from tests.test_translator.test_float_negation import _inputs as _float_inputs
from tests.test_translator.test_metal_builtin_ownership import _compile
from tests.test_translator.test_metal_precise_trig import GUARD, _execute

REQUIRE_ENV = "CROSTL_REQUIRE_UNARY_GROUPING"
SIGNS = (
    ("-(-value)", False),
    ("+(+value)", False),
    ("-(+value)", True),
    ("+(-value)", True),
    ("-(-(-value))", True),
    ("+(+(+value))", False),
    ("-(+(-value))", False),
    ("+(-(+value))", True),
)
UPDATES = (
    ("-(++value)", 1, 1, True, 0),
    ("-(--value)", -1, -1, True, 0),
    ("-(value++)", 0, 1, True, 0),
    ("-(value--)", 0, -1, True, 0),
    ("+(++value)", 1, 1, False, 0),
    ("+(--value)", -1, -1, False, 0),
    ("+(value++)", 0, 1, False, 0),
    ("+(value--)", 0, -1, False, 0),
    ("-(-(++value))", 1, 1, False, 0),
    ("+(+(--value))", -1, -1, False, 0),
    ("-(-(value++))", 0, 1, False, 0),
    ("+(+(value--))", 0, -1, False, 0),
    ("-(-record(count, value))", 0, 0, False, 1),
    ("+(+record(count, value))", 0, 0, False, 1),
    ("-(-TYPE(value++))", 0, 1, False, 0),
    ("+(+TYPE(--value))", -1, -1, False, 0),
)


def _prefix_source(dtype, width):
    value_type = dtype if width == 1 else f"{dtype}{width}"
    stride = (len(SIGNS) + 1) * width
    initial = "values[i]" if dtype == "uint" else f"as_type<{dtype}>(values[i])"
    lines = []
    for index, (expression, _negative) in enumerate(SIGNS):
        lines.append(f"    {value_type} result{index} = {expression};")
    for index, name in enumerate(
        [f"result{index}" for index in range(len(SIGNS))] + ["value"]
    ):
        for lane in range(width):
            access = name + (f".{'xyzw'[lane]}" if width > 1 else "")
            lines.append(
                f"    results[{stride}u * i + {index * width + lane}u] = as_type<uint>({access});"
            )
    return f"""#include <metal_stdlib>
using namespace metal;
kernel void unary_values(device const uint* values [[buffer(0)]],
                         device uint* results [[buffer(1)]],
                         uint i [[thread_position_in_grid]]) {{
    {value_type} value = {value_type}({initial});
{chr(10).join(lines)}
}}
"""


def _prefix_expected(inputs, dtype, width):
    expected = []
    for word in inputs:
        opposite = word ^ 0x80000000 if dtype == "float" else (-word) & 0xFFFFFFFF
        for _expression, negative in SIGNS:
            expected.extend([opposite if negative else word] * width)
        expected.extend([word] * width)
    return expected + GUARD


def _update_source(dtype):
    lines = []
    stride = 3 * len(UPDATES)
    for index, (expression, *_rest) in enumerate(UPDATES):
        expression = expression.replace("TYPE", dtype)
        expression = expression.replace("value", f"value{index}")
        expression = expression.replace("count", f"count{index}")
        lines.extend(
            [
                f"    {dtype} value{index} = {dtype}(int(values[i] % 257u) - 128);",
                f"    uint count{index} = 0u;",
                f"    {dtype} result{index} = {expression};",
                f"    results[{stride}u * i + {3 * index}u] = as_type<uint>(result{index});",
                f"    results[{stride}u * i + {3 * index + 1}u] = as_type<uint>(value{index});",
                f"    results[{stride}u * i + {3 * index + 2}u] = count{index};",
            ]
        )
    return f"""#include <metal_stdlib>
using namespace metal;
{dtype} record(thread uint& count, {dtype} value) {{ count += 1; return value; }}
kernel void unary_updates(device const uint* values [[buffer(0)]],
                          device uint* results [[buffer(1)]],
                          uint i [[thread_position_in_grid]]) {{
{chr(10).join(lines)}
}}
"""


def _update_expected(inputs, dtype):
    def bits(value):
        if dtype == "float":
            return struct.unpack("<I", struct.pack("<f", value))[0]
        return value & 0xFFFFFFFF

    expected = []
    for word in inputs:
        initial = word % 257 - 128
        for _expression, result_delta, final_delta, negative, count in UPDATES:
            value = initial + result_delta
            if dtype == "float":
                value = float(value)
            expected.extend(
                [
                    bits(-value if negative else value),
                    bits(initial + final_delta),
                    count,
                ]
            )
    return expected + GUARD


def _check(actual, expected):
    assert actual == expected, [
        (index, hex(wanted), hex(found))
        for index, (wanted, found) in enumerate(zip(expected, actual))
        if wanted != found
    ][:20]
    return 0


def _translate_saved(root, source, target):
    original = root / "original.metal"
    original.write_text(source, encoding="utf-8")
    intermediate = root / "saved.cgl"
    intermediate.write_text(
        translate(str(original), backend="cgl", format_output=False), encoding="utf-8"
    )
    generated = translate(str(original), backend=target, format_output=False)
    assert (
        translate(str(intermediate), backend=target, format_output=False) == generated
    )
    return generated


@pytest.mark.parametrize("target", ["metal", "directx", "opengl"])
@pytest.mark.parametrize("dtype", ["int", "uint", "float"])
@pytest.mark.parametrize("width", [1, 2, 3, 4])
def test_nested_signs_translate_and_compile(tmp_path, target, dtype, width):
    source = _prefix_source(dtype, width)
    generated = _translate_saved(tmp_path, source, target)
    assert "++value" not in generated and "--value" not in generated
    _compile(generated, target, tmp_path)


@pytest.mark.parametrize("target", ["metal", "directx", "opengl"])
@pytest.mark.parametrize("dtype", ["int", "uint", "float"])
def test_nested_updates_translate_and_compile(tmp_path, target, dtype):
    generated = _translate_saved(tmp_path, _update_source(dtype), target)
    assert "+++value" not in generated and "---value" not in generated
    _compile(generated, target, tmp_path)


def _run_native(root, source, inputs, expected, entry):
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 for required native unary grouping")
    target = {"darwin": "metal", "win32": "directx", "linux": "opengl"}[sys.platform]
    generated = _translate_saved(root, source, target)
    for name, words in (("inputs", inputs), ("expected", expected)):
        (root / f"{name}.bin").write_bytes(struct.pack(f"<{len(words)}I", *words))
    records = {}
    for name, code in (("generated", generated), ("original", source)):
        if name == "original" and target != "metal":
            continue
        records[name] = _execute(
            root / name,
            target,
            code,
            inputs,
            expected,
            metal_entry=entry,
            check_outputs=_check,
            metal_compile_flags=("-fno-fast-math",),
        )
    (root / "evidence.json").write_text(
        json.dumps(
            {
                "target": target,
                "inputCount": len(inputs),
                "outputCount": len(expected),
                "records": records,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


@pytest.mark.parametrize("dtype", ["int", "uint", "float"])
@pytest.mark.parametrize("width", [1, 2, 3, 4])
def test_nested_signs_execute_without_mutating_operands(tmp_path, dtype, width):
    inputs = _float_inputs()
    if dtype == "int":
        # Negating INT_MIN is undefined in the original signed Metal expression.
        inputs.remove(0x80000000)
    _run_native(
        tmp_path,
        _prefix_source(dtype, width),
        inputs,
        _prefix_expected(inputs, dtype, width),
        "unary_values",
    )


@pytest.mark.parametrize("dtype", ["int", "uint", "float"])
def test_nested_updates_execute_once(tmp_path, dtype):
    inputs = list(range(257))
    _run_native(
        tmp_path,
        _update_source(dtype),
        inputs,
        _update_expected(inputs, dtype),
        "unary_updates",
    )


@pytest.mark.parametrize("index", [0, 1, 2, -1])
def test_unary_oracle_rejects_value_operand_counter_and_guard_changes(index):
    expected = _update_expected([129], "int")
    actual = list(expected)
    actual[index] ^= 1
    with pytest.raises(AssertionError):
        _check(actual, expected)


def test_ci_requires_unary_grouping_without_an_additional_runner():
    workflow = (
        Path(__file__).resolve().parents[2]
        / ".github/workflows/demo-project-testing.yml"
    ).read_text()
    step = workflow.split("      - name: Validate Metal builtin ownership\n", 1)[1]
    step = step.split("      - name:", 1)[0]
    assert "if:" not in step and "continue-on-error" not in step
    assert f'{REQUIRE_ENV}: "1"' in step
    assert "tests/test_translator/test_unary_grouping.py" in step
    assert "--timeout-seconds 120" in step
    assert "pytest -q -n auto" in step

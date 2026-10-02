"""Retain Metal vector element conversions through native project packages."""

import os
import sys
from pathlib import Path

import pytest

from crosstl import translate
from tests.test_translator.test_software_subgroup_product import _package
from tests.test_translator.test_wide_integer_arithmetic import (
    REQUIRE_ENV,
    _execute_integer_case,
    _quotient,
)

OPERATORS = ("/", "%", "&", "|", "^", "<", "<=", ">", ">=", "==", "!=")
TYPE_PAIRS = (
    ("uint", "long"),
    ("int", "ulong"),
    ("long", "uint"),
    ("ulong", "int"),
    ("int", "uint"),
    ("uint", "int"),
    ("long", "ulong"),
    ("ulong", "long"),
)


def _integer(value, kind):
    bits = 64 if kind in {"long", "ulong"} else 32
    value %= 1 << bits
    if kind in {"int", "long"} and value >= 1 << (bits - 1):
        value -= 1 << bits
    return value


@pytest.mark.parametrize("target", ["metal", "directx", "opengl"])
@pytest.mark.parametrize("vector_kind,scalar_kind", TYPE_PAIRS)
def test_vector_scalar_conversions_survive_saved_crossgl(
    tmp_path, target, vector_kind, scalar_kind
):
    source = tmp_path / "original.metal"
    source.write_text(f"""#include <metal_stdlib>
using namespace metal;
using Scalar = {scalar_kind};
using Vector = {vector_kind}2;
Scalar scalar_value(const Scalar value) {{ return value; }}
Vector vector_value(const Vector value) {{ return value; }}
kernel void products(device long* numerators [[buffer(0)]],
                     device long* denominators [[buffer(1)]],
                     device long* outputs [[buffer(2)]],
                     uint tid [[thread_position_in_grid]]) {{
    const Scalar scalar = Scalar(numerators[tid]);
    Vector value = Vector({vector_kind}(denominators[tid]));
    auto divided = vector_value(value) / scalar_value(scalar);
    auto nested = divided / scalar;
    auto inverse = scalar_value(scalar) / vector_value(value);
    value /= scalar_value(scalar);
    outputs[tid * 4u] = long(divided.x);
    outputs[tid * 4u + 1u] = long(nested.y);
    outputs[tid * 4u + 2u] = long(inverse.x);
    outputs[tid * 4u + 3u] = long(value.y);
}}
""")
    intermediate = tmp_path / "saved.cgl"
    intermediate.write_text(translate(str(source), backend="cgl", format_output=False))
    element = {"long": "int64", "ulong": "uint64"}.get(vector_kind, vector_kind)
    assert f"vector_value(value) / {element}(scalar_value(scalar))" in (
        intermediate.read_text()
    )
    assert translate(
        str(intermediate), backend=target, format_output=False
    ) == translate(str(source), backend=target, format_output=False)


@pytest.mark.parametrize("vector_kind,scalar_kind", TYPE_PAIRS)
@pytest.mark.parametrize("width", [2, 3, 4])
def test_vector_scalar_conversions_execute_before_arithmetic(
    tmp_path, vector_kind, scalar_kind, width
):
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 for required Metal vector arithmetic")
    target = {"darwin": "metal", "win32": "directx", "linux": "opengl"}[sys.platform]
    vector = f"{vector_kind}{width}"
    statements = []
    output_count = 0

    def save(expression):
        nonlocal output_count
        statements.append(f"outputs[base + {output_count}u] = {expression};")
        output_count += 1

    for operator in OPERATORS:
        for reverse in (False, True):
            left, right = ("scalar", "value") if reverse else ("value", "scalar")
            name = f"result{output_count}"
            statements.append(f"auto {name} = {left} {operator} {right};")
            for component in "xyzw"[:width]:
                save(f"{name}.{component}")
    statements.extend(
        [
            f"{vector} pairs[2] = {{value, value}};",
            "uint index = 0u;",
            f"{scalar_kind} updated = scalar;",
            "pairs[index++] /= updated++;",
        ]
    )
    for component in "xyzw"[:width]:
        save(f"pairs[0].{component}")
    save("index")
    save("updated")
    statements.extend(
        [
            f"{vector} remainder = value;",
            "updated = scalar;",
            "remainder %= updated++;",
        ]
    )
    for component in "xyzw"[:width]:
        save(f"remainder.{component}")
    save("updated")
    statements.append("auto nested = (value / scalar) / scalar;")
    for component in "xyzw"[:width]:
        save(f"nested.{component}")
    source = f"""#include <metal_stdlib>
using namespace metal;
kernel void products(device long* numerators [[buffer(0)]],
                     device long* denominators [[buffer(1)]],
                     device long* outputs [[buffer(2)]],
                     uint tid [[thread_position_in_grid]]) {{
    {scalar_kind} scalar = {scalar_kind}(numerators[tid]);
    {vector} value = {vector}({vector_kind}(denominators[tid]));
    uint base = tid * {output_count}u;
    {chr(10).join(statements)}
}}
"""
    source, descriptor, package = _package(
        tmp_path, target, "long", (1, 1, 1), source=source, software_subgroups=False
    )
    numerators = [-94, -128, -257, 94, 2**31, -(2**31), 2**32 + 1, -(2**40) + 1]
    denominators = [128, 128, 128, 128, 2**31 - 1, 2**31 - 1, 2**32 - 1, 3]
    wanted = []

    def evaluate(left, right, operator):
        if operator == "/":
            return _quotient(left, right)
        if operator == "%":
            return left - _quotient(left, right) * right
        return {
            "&": lambda: left & right,
            "|": lambda: left | right,
            "^": lambda: left ^ right,
            "<": lambda: int(left < right),
            "<=": lambda: int(left <= right),
            ">": lambda: int(left > right),
            ">=": lambda: int(left >= right),
            "==": lambda: int(left == right),
            "!=": lambda: int(left != right),
        }[operator]()

    for raw_scalar, raw_vector in zip(numerators, denominators):
        scalar = _integer(raw_scalar, scalar_kind)
        converted = _integer(scalar, vector_kind)
        value = _integer(raw_vector, vector_kind)
        assert converted != 0 and value != 0
        for operator in OPERATORS:
            for reverse in (False, True):
                left, right = (converted, value) if reverse else (value, converted)
                result = evaluate(left, right, operator)
                wanted.extend([_integer(result, "long")] * width)
        quotient = _quotient(value, converted)
        remainder = value - quotient * converted
        incremented = _integer(_integer(scalar + 1, scalar_kind), "long")
        wanted.extend([_integer(quotient, "long")] * width + [1, incremented])
        wanted.extend([_integer(remainder, "long")] * width + [incremented])
        wanted.extend([_integer(_quotient(quotient, converted), "long")] * width)
    assert len(wanted) == output_count * len(numerators)
    _execute_integer_case(
        tmp_path,
        source,
        descriptor,
        package,
        target,
        numerators,
        denominators,
        wanted,
        "int64",
        "int64",
        # Original comparisons intentionally exercise mixed signedness. Generated
        # targets keep the strict default warning policy and explicit conversions.
        original_metal_compile_flags=("-Wno-sign-compare",),
    )


def test_vector_scalar_native_gate_is_required_on_every_target():
    from tools import ci_coverage

    root = Path(__file__).resolve().parents[2]
    workflow = (root / ".github/workflows/mlx-portable-host.yml").read_text()
    step = ci_coverage.workflow_step_section(
        workflow, "Validate mixed-width integer arithmetic"
    )
    assert f'{REQUIRE_ENV}: "1"' in step
    assert "test_metal_vector_scalar_arithmetic.py" in step
    assert "if:" not in step and "continue-on-error" not in step
    assert "--timeout-seconds 180" in step and "--junitxml" in step
    for event in ("pull_request", "push"):
        assert (
            "tests/test_translator/test_metal_vector_scalar_arithmetic.py"
            in ci_coverage.workflow_event_path_filters(workflow, event)
        )

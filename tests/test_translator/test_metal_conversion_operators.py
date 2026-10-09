"""Concrete conversion operators retain lexical source types and mutations."""

import os
import shutil
import sys
from pathlib import Path

import pytest

from crosstl import translate
from tests.test_translator.test_boolean_buffer_runtime import _bound_values, _request
from tests.test_translator.test_loop_updates import _execute
from tests.test_translator.test_metal_builtin_ownership import _compile
from tests.test_translator.test_software_subgroup_product import _package

REQUIRE_ENV = "CROSTL_REQUIRE_SCALAR_ALIASES"


def source(alias):
    return f"""#include <metal_stdlib>
using namespace metal;
struct First {{
    float value;
    First(float x) : value(x + 8.0f) {{}}
    operator float() const {{ return value + 1.0f; }}
}};
struct Second {{
    float value;
    Second(float x) : value(x + 16.0f) {{}}
    operator float() const {{ return value + 2.0f; }}
}};
float helper(float x) {{ auto item = First(x); return float(item); }}
template<bool choice> float2 select_value(float x) {{
    using Base = {alias};
    using Value = Base;
    auto item = Value(x++);
    auto copied = item;
    return float2(float(copied), x);
}}
float nested(float x) {{
    auto item = First(x);
    float result = float(item);
    {{ auto item = Second(x); result += float(item); }}
    return result + float(item);
}}
float alias_scope(float x) {{
    using Value = First;
    auto item = Value(x);
    {{ using Value = Second; return float(item); }}
}}
kernel void convert_values(const device float* src [[buffer(0)]],
                           device float* results [[buffer(1)]],
                           uint i [[thread_position_in_grid]]) {{
    float value = src[i];
    float2 first = select_value<true>(value);
    float2 second = select_value<false>(value);
    uint base = 4u + 7u * i;
    results[base] = helper(value);
    results[base + 1u] = first.x;
    results[base + 2u] = first.y;
    results[base + 3u] = second.x;
    results[base + 4u] = second.y;
    results[base + 5u] = nested(value);
    results[base + 6u] = alias_scope(value);
}}
"""


ALIASES = (
    "metal::conditional_t<choice, First, Second>",
    "typename metal::conditional<choice, First, Second>::type",
)


@pytest.mark.parametrize("target", ("metal", "opengl", "directx"))
@pytest.mark.parametrize("alias", ALIASES)
def test_concrete_conversion_helpers_compile(tmp_path, target, alias):
    path = tmp_path / "source.metal"
    path.write_text(source(alias))
    generated = translate(str(path), backend=target, format_output=False)
    tool = {"metal": "xcrun", "opengl": "glslangValidator", "directx": "dxc"}[target]
    if shutil.which(tool):
        _compile(generated, target, tmp_path)


@pytest.mark.parametrize("alias", ALIASES)
def test_concrete_conversion_helpers_execute(tmp_path, alias):
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 for required conversion execution")
    target = {"darwin": "metal", "win32": "directx", "linux": "opengl"}[sys.platform]
    original = source(alias)
    _, descriptor, package = _package(
        tmp_path, target, "float", (1, 1, 1), source=original, software_subgroups=False
    )
    values = [-4096.5, -7.25, -2.0, -1.0, -0.0, 0.0, 0.25, 1.0, 3.0, 4096.5]
    guard = -123456.0
    expected = [guard] * 4
    for value in values:
        expected.extend(
            [
                value + 9,
                value + 9,
                value + 1,
                value + 18,
                value + 1,
                3 * value + 36,
                value + 9,
            ]
        )
    expected.extend([guard] * 4)
    inputs = {
        "src": {"dtype": "float32", "shape": [len(values)], "values": values},
        "results": {
            "dtype": "float32",
            "shape": [len(expected)],
            "values": [guard] * len(expected),
        },
    }
    outputs = {"results": {**inputs["results"], "values": expected}}
    request = _request(descriptor, package, inputs, outputs, len(values))
    _execute(
        request,
        _bound_values(descriptor, outputs),
        tmp_path,
        original_source=original,
        original_entry="convert_values",
    )


def test_conversion_execution_reuses_required_alias_gate():
    from tools import ci_coverage

    workflow = (
        Path(__file__).resolve().parents[2]
        / ".github/workflows/demo-project-testing.yml"
    ).read_text()
    step = ci_coverage.workflow_step_section(
        workflow, "Validate scalar alias resolution"
    )
    assert f'{REQUIRE_ENV}: "1"' in step
    assert "tests/test_translator/test_metal_conversion_operators.py" in step
    assert "--timeout-seconds 120" in step and "-n auto" in step
    assert "continue-on-error" not in step and "if:" not in step

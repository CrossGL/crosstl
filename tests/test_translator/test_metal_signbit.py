"""Qualified sign classification preserves source ownership and payload signs."""

import os
import shutil
import sys
from pathlib import Path

import pytest

from crosstl import translate
from tests.test_translator.test_boolean_buffer_runtime import _bound_values, _request
from tests.test_translator.test_loop_updates import _execute
from tests.test_translator.test_metal_builtin_ownership import REQUIRE_ENV, _compile
from tests.test_translator.test_software_subgroup_product import _package


def source(dtype):
    value = (
        "as_type<float>(values[cursor++])"
        if dtype == "float"
        else "as_type<half>(ushort(values[cursor++]))"
    )
    opposite = (
        "as_type<float>(values[i] ^ 0x80000000u)"
        if dtype == "float"
        else "as_type<half>(ushort(values[i] ^ 0x8000u))"
    )
    return f"""#include <metal_stdlib>
using namespace metal;
bool signbit(float x) {{ return x > 0.0f; }}
namespace custom {{ bool signbit(float x) {{ return x < 0.0f; }} }}
namespace metal {{ bool signbit(int x) {{ return x >= 0; }} }}
kernel void classify(const device uint* values [[buffer(0)]],
                     device uint* results [[buffer(1)]],
                     uint i [[thread_position_in_grid]]) {{
    uint cursor = i;
    auto scalar = metal::signbit({value});
    {dtype} value = {value.replace("cursor++", "i")};
    auto flags = metal::signbit({dtype}4(value, {opposite}, {dtype}(0.0f), -{dtype}(0.0f)));
    results[9u * i + 1u] = uint(scalar);
    results[9u * i + 2u] = uint(flags.x);
    results[9u * i + 3u] = uint(flags.y);
    results[9u * i + 4u] = uint(flags.z);
    results[9u * i + 5u] = uint(flags.w);
    results[9u * i + 6u] = cursor - i;
    results[9u * i + 7u] = uint(::signbit(1.0f));
    results[9u * i + 8u] = uint(custom::signbit(1.0f));
    results[9u * i + 9u] = uint(metal::signbit(-1));
}}
"""


@pytest.mark.parametrize(
    "target,dtype",
    [("metal", "float"), ("metal", "half"), ("opengl", "float"), ("directx", "float")],
)
def test_qualified_signbit_infers_boolean_results_and_compiles(tmp_path, target, dtype):
    path = tmp_path / "source.metal"
    path.write_text(source(dtype))
    intermediate = translate(str(path), backend="crossgl", format_output=False)
    assert "bool scalar = signbit(" in intermediate
    assert "bvec4 flags = signbit(" in intermediate
    assert "signbit__metal_overload_" in intermediate
    generated = translate(str(path), backend=target, format_output=False)
    assert "metal_u3a_u3asignbit" not in generated
    tool = {"metal": "xcrun", "opengl": "glslangValidator", "directx": "dxc"}[target]
    if shutil.which(tool):
        _, module = _compile(generated, target, tmp_path)
        assert module.is_file() and module.stat().st_size


@pytest.mark.parametrize("width", (1, 2, 3, 4))
def test_qualified_signbit_does_not_bind_local_shadow(tmp_path, width):
    dtype = "float" + (str(width) if width != 1 else "")
    result = "signbit" if width == 1 else "all(signbit)"
    path = tmp_path / "shadow.metal"
    path.write_text(f"""#include <metal_stdlib>
using namespace metal;
kernel void shadow(device uint* results [[buffer(0)]]) {{
    auto signbit = metal::signbit({dtype}(-1.0f));
    results[0] = uint({result});
}}
""")
    generated = translate(str(path), backend="metal", format_output=False)
    assert "metal::signbit(" in generated
    if shutil.which("xcrun"):
        _compile(generated, "metal", tmp_path)


@pytest.mark.parametrize("dtype,start", (("float", 0), ("half", 0), ("half", 32768)))
def test_signbit_executes_with_exact_signs(tmp_path, dtype, start):
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 for required sign classification execution")
    target = {"darwin": "metal", "win32": "directx", "linux": "opengl"}[sys.platform]
    if dtype == "half" and target != "metal":
        pytest.skip(
            "Native half sign classification is specific to the Metal roundtrip"
        )
    words = (
        list(range(start, start + 32768))
        if dtype == "half"
        else [
            sign | (exponent << 23) | mantissa
            for sign in (0, 0x80000000)
            for exponent in range(256)
            for mantissa in (0, 1, 0x3FFFFF, 0x400000, 0x7FFFFE, 0x7FFFFF)
        ]
    )
    original = source(dtype)
    _, descriptor, package = _package(
        tmp_path, target, "uint", (1, 1, 1), source=original, software_subgroups=False
    )
    guard = 0x6A15BEEF
    expected = [guard]
    for word in words:
        sign = word >> (15 if dtype == "half" else 31)
        expected.extend((sign, sign, 1 - sign, 0, 1, 1, 1, 0, 0))
    expected.append(guard)
    inputs = {
        "values": {"dtype": "uint32", "shape": [len(words)], "values": words},
        "results": {
            "dtype": "uint32",
            "shape": [len(expected)],
            "values": [guard] * len(expected),
        },
    }
    outputs = {"results": {**inputs["results"], "values": expected}}
    request = _request(descriptor, package, inputs, outputs, len(words))
    _execute(
        request,
        _bound_values(descriptor, outputs),
        tmp_path,
        original_source=original,
        original_entry="classify",
    )


def test_signbit_native_gate_uses_existing_runners():
    from tools import ci_coverage

    workflow = (
        Path(__file__).resolve().parents[2]
        / ".github/workflows/demo-project-testing.yml"
    ).read_text()
    step = ci_coverage.workflow_step_section(
        workflow, "Validate Metal builtin ownership"
    )
    assert f'{REQUIRE_ENV}: "1"' in step
    assert "tests/test_translator/test_metal_signbit.py" in step
    assert "--timeout-seconds 120" in step
    assert "continue-on-error" not in step

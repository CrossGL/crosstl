"""Preserve floating-point operands and lazy evaluation in conditional selection."""

import json
import os
import struct
import sys
from pathlib import Path

import pytest

from crosstl import translate
from crosstl.translator import parse
from crosstl.translator.codegen.GLSL_codegen import (
    GLSLCodeGen,
    OpenGLScalarConversionError,
)
from tests.ci_helpers import assert_paths_covered
from tests.test_translator.test_metal_builtin_ownership import _compile
from tests.test_translator.test_metal_precise_trig import GUARD, _execute

REQUIRE_ENV = "CROSTL_REQUIRE_METAL_FLOAT_SELECTION"
WORDS = [
    0,
    0x80000000,
    0,
    0x3F800000,
    0xBF800000,
    0x7F800000,
    0xFF800000,
    0x7FC12345,
    0xFFC54321,
]
SOURCE = """#include <metal_stdlib>
using namespace metal;
float smaller(float x, float y) { return x < y ? x : y; }
float larger(float x, float y) { return x > y ? x : y; }
bool condition(thread uint& count, bool value) { count += 10; return value; }
float record(thread uint& count, float value) { count += 1; return value; }
kernel void selection_values(device const uint* values [[buffer(0)]],
                             device uint* results [[buffer(1)]],
                             uint i [[thread_position_in_grid]]) {
    float x = as_type<float>(values[i]);
    float y = as_type<float>(values[(i + 1) % 9]);
    bool choose = (i & 1u) != 0u;
    uint count = 0;
    float2 a2 = float2(x, y), b2 = float2(y, x);
    float3 a3 = float3(x, y, x), b3 = float3(y, x, y);
    float4 a4 = float4(x, y, x, y), b4 = float4(y, x, y, x);
    float2 selected2 = choose ? a2 : b2;
    float3 selected3 = choose ? a3 : b3;
    float4 selected4 = choose ? a4 : b4;
    float lazy = condition(count, choose) ? record(count, x) : record(count, y);
    results[13 * i] = as_type<uint>(smaller(x, y));
    results[13 * i + 1] = as_type<uint>(larger(x, y));
    results[13 * i + 2] = as_type<uint>(selected2.x);
    results[13 * i + 3] = as_type<uint>(selected2.y);
    results[13 * i + 4] = as_type<uint>(selected3.x);
    results[13 * i + 5] = as_type<uint>(selected3.y);
    results[13 * i + 6] = as_type<uint>(selected3.z);
    results[13 * i + 7] = as_type<uint>(selected4.x);
    results[13 * i + 8] = as_type<uint>(selected4.y);
    results[13 * i + 9] = as_type<uint>(selected4.z);
    results[13 * i + 10] = as_type<uint>(selected4.w);
    results[13 * i + 11] = as_type<uint>(lazy);
    results[13 * i + 12] = count;
}
"""


def _translate(tmp_path, target):
    path = tmp_path / "selection.metal"
    path.write_text(SOURCE, encoding="utf-8")
    return translate(str(path), backend=target, format_output=False)


def _shader(body, helpers=""):
    return parse(f"""shader FloatSelection {{
        {helpers}
        RWStructuredBuffer<float> output @ binding(0);
        compute {{
            layout(local_size_x = 1) in;
            void main() {{ {body} }}
        }}
    }}""")


def _expected():
    result = []
    for i, x in enumerate(WORDS):
        y = WORDS[(i + 1) % len(WORDS)]
        a, b = (struct.unpack("<f", struct.pack("<I", word))[0] for word in (x, y))
        selected = [x, y, x, y] if i & 1 else [y, x, y, x]
        result.extend(
            [
                x if a < b else y,
                x if a > b else y,
                *selected[:2],
                *selected[:3],
                *selected,
                x if i & 1 else y,
                11,
            ]
        )
    return result + GUARD


def _check(actual, expected):
    assert (
        actual == expected
    ), "conditional operand bits, evaluation count or guard changed"
    return 0


@pytest.mark.parametrize("target", ["directx", "opengl", "metal"])
def test_float_selection_compiles(tmp_path, target):
    _compile(_translate(tmp_path, target), target, tmp_path)


@pytest.mark.parametrize("dtype", ["float", "vec2", "vec3", "vec4"])
def test_float_selection_helpers_preserve_types(tmp_path, dtype):
    generated = GLSLCodeGen().generate(
        _shader(
            f"{dtype} a = {dtype}(output[0]); {dtype} b = {dtype}(output[1]); "
            f"{dtype} selected = output[2] < 0.0 ? a : b;"
        )
    )
    assert f"{dtype} crossgl_select_bits_{dtype}(bool condition" in generated
    assert f"crossgl_select_bits_{dtype}((output_[2] < 0.0), a, b)" in generated
    _compile(generated, "opengl", tmp_path)


def test_float_selection_helper_names_and_state(tmp_path):
    generator = GLSLCodeGen()
    generated = generator.generate(
        _shader(
            "float crossgl_select_bits_float = output[0]; float b = output[1]; "
            "output[2] = crossgl_select_bits_float < b ? crossgl_select_bits_float : b;"
        )
    )
    assert "float crossgl_select_bits_float_2(bool condition" in generated
    _compile(generated, "opengl", tmp_path)
    assert "crossgl_select_bits" not in generator.generate(_shader("output[0] = 1.0;"))


def test_float_selection_keeps_lazy_arms_and_constant_initializers(tmp_path):
    generated = GLSLCodeGen().generate(
        _shader(
            "int i = 0; float a = output[0]; "
            "float b = i++ == 0 ? a : 0.0; "
            "output[0] = i == 0 ? output[i++] : output[0];",
            "const float initial = true ? 1.0 : 2.0;",
        )
    )
    assert "crossgl_select_bits" not in generated
    _compile(generated, "opengl", tmp_path)


@pytest.mark.parametrize("builtin", ["floatBitsToUint", "uintBitsToFloat"])
def test_float_selection_rejects_shadowed_builtins(builtin):
    with pytest.raises(OpenGLScalarConversionError) as error:
        GLSLCodeGen().generate(
            _shader(
                "float a = output[0]; float b = output[1]; output[2] = a < b ? a : b;",
                f"float {builtin}(float value) {{ return value; }}",
            )
        )
    assert error.value.reason == "conditional-target-builtin-shadowed"


def test_float_selection_is_required_in_native_workflows():
    from tools import ci_coverage

    root = Path(__file__).resolve().parents[2]
    workflow = (root / ".github/workflows/demo-project-testing.yml").read_text()
    for job in ("mlx-metal-porting", "metal-host", "portable-host"):
        step = ci_coverage.workflow_job_step_section(
            workflow, job, "Validate pinned comparison arithmetic"
        )
        assert "test_metal_float_selection.py" in step
        assert f'{REQUIRE_ENV}: "1"' in step
        assert "-n auto" in step and "continue-on-error" not in step
        assert "--timeout-seconds" in step and "--junitxml" in step
        for event in ("pull_request", "push"):
            assert_paths_covered(
                ci_coverage.workflow_event_path_filters(workflow, event),
                "tests/test_translator/test_metal_float_selection.py",
            )


@pytest.mark.parametrize("index", [0, 1, 2, 6, 11, 12, -1])
def test_float_selection_verifier_rejects_corruption(index):
    expected = _expected()
    actual = list(expected)
    actual[index] ^= 0x80000000 if index != 12 else 1
    with pytest.raises(AssertionError):
        _check(actual, expected)


def test_float_selection_executes(tmp_path):
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 for required native float selection")
    target = {"win32": "directx", "linux": "opengl", "darwin": "metal"}[sys.platform]
    expected = _expected()
    (tmp_path / "inputs.bin").write_bytes(struct.pack(f"<{len(WORDS)}I", *WORDS))
    (tmp_path / "expected.bin").write_bytes(
        struct.pack(f"<{len(expected)}I", *expected)
    )
    records = {
        "generated": _execute(
            tmp_path / "generated",
            target,
            _translate(tmp_path, target),
            WORDS,
            expected,
            metal_entry="selection_values",
            check_outputs=_check,
            metal_compile_flags=("-fno-fast-math",),
        )
    }
    if target == "metal":
        records["original"] = _execute(
            tmp_path / "original",
            target,
            SOURCE,
            WORDS,
            expected,
            metal_entry="selection_values",
            check_outputs=_check,
            metal_compile_flags=("-fno-fast-math",),
        )
    (tmp_path / "evidence.json").write_text(
        json.dumps({"target": target, "records": records}, indent=2)
    )

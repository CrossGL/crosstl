"""Preserve floating-point operands and lazy evaluation in conditional selection."""

import json
import os
import struct
import sys
from pathlib import Path

import pytest

from crosstl import translate
from crosstl.translator import parse
from crosstl.translator.codegen.directx_codegen import (
    DirectXContextualConversionError,
    HLSLCodeGen,
)
from crosstl.translator.codegen.GLSL_codegen import (
    GLSLCodeGen,
    OpenGLScalarConversionError,
)
from tests.ci_helpers import assert_paths_covered
from tests.test_translator.test_boolean_buffer_runtime import _bound_values, _request
from tests.test_translator.test_half_buffer_runtime import _validate_half
from tests.test_translator.test_half_copy_identity import _widen
from tests.test_translator.test_loop_updates import _execute as _execute_package
from tests.test_translator.test_metal_builtin_ownership import _compile
from tests.test_translator.test_metal_precise_trig import GUARD, _execute
from tests.test_translator.test_software_subgroup_product import _package

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


@pytest.mark.parametrize("width", (1, 2, 3, 4))
def test_half_selection_uses_integer_masks(tmp_path, width):
    dtype = "float16_t" + (str(width) if width > 1 else "")
    yes = ", ".join(["output[0]"] * width)
    no = ", ".join(["output[1]"] * width)
    generated = HLSLCodeGen().generate(
        _shader(
            f"{dtype} a = {dtype}({yes}); {dtype} b = {dtype}({no}); "
            f"{dtype} selected = output[2] < 0.0 ? a : b;"
        )
    )
    assert f"{dtype} __crossgl_select_half_bits{width}(bool condition" in generated
    assert f"__crossgl_select_half_bits{width}((output[2] < 0.0), a, b)" in generated
    assert "0u - uint(condition)" in generated
    assert "asuint16(yes)" in generated and "asuint16(no)" in generated
    _compile(
        generated, "directx", tmp_path, directx_compile_flags=("-enable-16bit-types",)
    )


def test_half_selection_helper_names_and_state(tmp_path):
    generator = HLSLCodeGen()
    generated = generator.generate(
        _shader(
            "float16_t __crossgl_select_half_bits1 = float16_t(output[0]); "
            "float16_t b = float16_t(output[1]); "
            "float16_t result = output[2] < 0.0 ? __crossgl_select_half_bits1 : b;"
        )
    )
    assert "float16_t __crossgl_select_half_bits1_(bool condition" in generated
    _compile(
        generated, "directx", tmp_path, directx_compile_flags=("-enable-16bit-types",)
    )
    assert "__crossgl_select_half_bits" not in generator.generate(
        _shader("output[0] = 1.0;")
    )


def test_half_selection_keeps_lazy_evaluation_and_constant_initializers(tmp_path):
    generated = HLSLCodeGen().generate(
        _shader(
            "int i = 0; float16_t a = float16_t(output[0]); float16_t b = a; "
            "float16_t selected = i++ == 0 ? a : b; "
            "output[0] = i == 0 ? record(i, a) : record(i, b);",
            "const float16_t initial = true ? float16_t(1.0) : float16_t(2.0); "
            "float16_t record(inout int i, float16_t value) { i++; return value; }",
        )
    )
    assert "__crossgl_select_half_bits" not in generated
    assert generated.count("record(i,") == 2
    _compile(
        generated, "directx", tmp_path, directx_compile_flags=("-enable-16bit-types",)
    )


@pytest.mark.parametrize("builtin", ("asuint16", "asfloat16"))
def test_half_selection_rejects_shadowed_target_intrinsics(builtin):
    with pytest.raises(DirectXContextualConversionError) as error:
        HLSLCodeGen().generate(
            _shader(
                "float16_t a = float16_t(1.0); float16_t b = float16_t(2.0); "
                "float16_t selected = output[0] < 0.0 ? a : b;",
                f"float16_t {builtin}(float16_t value) {{ return value; }}",
            )
        )
    assert error.value.reason == "half-selection-intrinsic-shadowed"


@pytest.mark.parametrize("predicate", ("isnan", "isinf", "isfinite"))
def test_half_selection_folds_safe_early_returns(tmp_path, predicate):
    generated = HLSLCodeGen().generate(
        _shader(
            "",
            f"float16_t choose(float16_t a, float16_t b) {{ if ({predicate}(a)) return a; return a < b ? a : b; }}",
        )
    )
    assert (
        f"return __crossgl_select_half_bits1({predicate}(a), a, __crossgl_select_half_bits1((a < b), a, b));"
        in generated
    )
    assert f"if ({predicate}(a))" not in generated
    _compile(
        generated, "directx", tmp_path, directx_compile_flags=("-enable-16bit-types",)
    )


@pytest.mark.parametrize(
    "body,helper",
    [
        ("if (take) return a; return output[0] < 0.0 ? a : b;", ""),
        (
            "if (take) return a; return record(a);",
            "float16_t record(float16_t x) { return x; }",
        ),
        ("if (take) { a = b; return a; } return b;", ""),
        ("if (take) return a; else return b; return a;", ""),
        (
            "if (isnan(a)) return a; return b;",
            "bool isnan(float16_t x) { return x < float16_t(0.0); }",
        ),
    ],
)
def test_half_selection_keeps_unsafe_or_owned_early_returns(body, helper):
    generated = HLSLCodeGen().generate(
        _shader(
            "",
            f"{helper} float16_t choose(float16_t a, float16_t b, bool take) {{ {body} }}",
        )
    )
    assert "if (" in generated
    assert "return __crossgl_select_half_bits1(take, a," not in generated


@pytest.mark.parametrize("width", (1, 2, 3, 4))
@pytest.mark.parametrize("predicate", ("isnan", "isinf", "isfinite"))
def test_hlsl_classification_result_type_retains_width_and_ownership(width, predicate):
    from crosstl.translator.ast import FunctionCallNode, IdentifierNode

    generator = HLSLCodeGen()
    generator.local_variable_types["x"] = "float16_t" + (
        str(width) if width > 1 else ""
    )
    expression = FunctionCallNode(IdentifierNode(predicate), [IdentifierNode("x")])
    assert generator.expression_result_type(expression) == "bool" + (
        str(width) if width > 1 else ""
    )
    generator.function_return_types[predicate] = "uint"
    assert generator.expression_result_type(expression) == "uint"


def _half_source():
    source = """#include <metal_stdlib>
using namespace metal;
half smaller(half x, half y) { return x < y ? x : y; }
half larger(half x, half y) { return x > y ? x : y; }
half minimum(half x, half y) { if (isnan(x)) return x; return x < y ? x : y; }
half maximum(half x, half y) { if (isnan(x)) return x; return x > y ? x : y; }
kernel void half_selection(const device half* values [[buffer(0)]],
                           device half* results [[buffer(1)]],
                           uint i [[thread_position_in_grid]]) {
    half x = values[2 * i];
    half y = values[2 * i + 1];
    bool choose = (i & 1u) != 0u;
"""
    expressions = ["smaller(x, y)", "larger(x, y)", "minimum(x, y)", "maximum(x, y)"]
    for width in (2, 3, 4):
        yes = ", ".join("xy"[i % 2] for i in range(width))
        no = ", ".join("yx"[i % 2] for i in range(width))
        source += f"    half{width} a{width} = half{width}({yes}), b{width} = half{width}({no});\n"
        source += f"    half{width} selected{width} = choose ? a{width} : b{width};\n"
        expressions.extend(f"selected{width}.{lane}" for lane in "xyzw"[:width])
    for index, expression in enumerate(expressions):
        source += f"    results[13 * i + {index}] = {expression};\n"
    return source + "}\n"


def _half_pairs(start):
    pairs = [
        (word, (word * 40503 + 71) & 65535) for word in range(start, start + 32768)
    ]
    if start == 0:
        edges = (
            0,
            0x8000,
            1,
            0x8001,
            0x3C00,
            0xBC00,
            0x7C00,
            0xFC00,
            0x7C01,
            0xFC01,
            0x7E55,
            0xFE55,
        )
        pairs.extend((x, y) for x in edges for y in edges)
    return pairs


def _half_expected(inputs):
    result = []
    for i, (x, y) in enumerate(inputs):
        a, b = (struct.unpack("<e", struct.pack("<H", word))[0] for word in (x, y))
        less, greater = (x if a < b else y), (x if a > b else y)
        selected = [x, y, x, y] if i & 1 else [y, x, y, x]
        result.extend(
            [
                less,
                greater,
                x if x & 0x7FFF > 0x7C00 else less,
                x if x & 0x7FFF > 0x7C00 else greater,
                *selected[:2],
                *selected[:3],
                *selected,
            ]
        )
    return result + [0x3555] * 32


def test_half_selection_reference_covers_words_payloads_and_ties():
    pairs = _half_pairs(0) + _half_pairs(32768)
    assert len(pairs) == 65680
    assert {x for x, _ in pairs} == set(range(65536))
    assert {y for _, y in pairs} == set(range(65536))
    for x, y in ((0, 0x8000), (0x8000, 0), (0x7C01, 0xFC01), (0, 0xFC01)):
        assert (x, y) in pairs
        expected = _half_expected([(x, y)])
        assert expected[:2] == [y, y]
        assert expected[2:4] == ([x, x] if x & 0x7FFF > 0x7C00 else [y, y])
        assert expected[-32:] == [0x3555] * 32


@pytest.mark.parametrize("target", ("metal", "directx", "opengl"))
def test_half_selection_compiles(tmp_path, target):
    source = tmp_path / "half.metal"
    source.write_text(_half_source(), encoding="utf-8")
    generated = translate(str(source), backend=target, format_output=False)
    if target == "directx":
        assert all(
            f"__crossgl_select_half_bits{width}(" in generated for width in range(1, 5)
        )
    _compile(
        generated, target, tmp_path, directx_compile_flags=("-enable-16bit-types",)
    )


@pytest.mark.parametrize("start", (0, 32768))
def test_half_selection_preserves_every_word(tmp_path, start):
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 for required native half selection")
    target = {"win32": "directx", "linux": "opengl", "darwin": "metal"}[sys.platform]
    inputs = _half_pairs(start)
    expected = _half_expected(inputs)
    source = _half_source()
    _, descriptor, package = _package(
        tmp_path, target, "uint", (1, 1, 1), source=source, software_subgroups=False
    )

    def payload(words):
        return {
            "dtype": "float32" if target == "opengl" else "float16",
            "encoding": (
                "ieee754-binary32" if target == "opengl" else "ieee754-binary16"
            ),
            "shape": [len(words)],
            "values": [_widen(word) for word in words] if target == "opengl" else words,
        }

    pairs = [word for pair in inputs for word in pair]
    bindings = {"values": payload(pairs), "results": payload([0x3555] * len(expected))}
    outputs = {"results": payload(expected)}
    request = _request(descriptor, package, bindings, outputs, len(inputs))
    (tmp_path / "case.json").write_text(
        json.dumps(
            {
                "target": target,
                "start": start,
                "inputCount": len(inputs),
                "valueCount": len(expected) - 32,
                "guardCount": 32,
            },
            indent=2,
        )
    )
    _execute_package(
        request,
        _bound_values(descriptor, outputs),
        tmp_path,
        original_source=source,
        original_entry="half_selection",
        validate=_validate_half,
    )

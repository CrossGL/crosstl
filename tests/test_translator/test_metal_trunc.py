"""Metal truncation keeps wrapper types, source ownership and exact results."""

import os
import random
import re
import shutil
import sys
from pathlib import Path

import pytest

from crosstl import translate
from crosstl.backend.Metal.MetalCrossGLCodeGen import (
    MetalStandardLibraryWrapperLoweringError,
)
from tests.test_backend.test_metal.test_codegen import (
    convert_without_preprocessing,
    normalize,
)
from tests.test_translator.test_boolean_buffer_runtime import _bound_values, _request
from tests.test_translator.test_half_buffer_runtime import _validate_half
from tests.test_translator.test_half_copy_identity import _widen
from tests.test_translator.test_loop_updates import _execute
from tests.test_translator.test_metal_builtin_ownership import REQUIRE_ENV, _compile
from tests.test_translator.test_software_subgroup_product import _package


@pytest.mark.parametrize("namespace", ("metal", "metal::fast", "metal::precise"))
@pytest.mark.parametrize(
    "dtype,canonical",
    (
        ("float", "float"),
        ("float4", "vec4"),
        ("half", "float16"),
        ("bfloat", "bfloat16"),
    ),
)
def test_materialized_trunc_preserves_result_type(namespace, dtype, canonical):
    opening = " ".join(f"namespace {part} {{" for part in namespace.split("::"))
    closing = "}" * len(namespace.split("::"))
    operand = "float(value)" if dtype == "bfloat" else "value"
    source = f"""typedef {dtype} Value;
    {opening}
    METAL_FUNC Value trunc(Value value) {{ return Value(__metal_trunc({operand})); }}
    {closing}
    Value evaluate(Value value) {{
      auto result = {namespace}::trunc(value++);
      return result;
    }}"""
    generated = normalize(convert_without_preprocessing(source))
    expression = (
        "bfloat16(trunc(float(value++)))" if dtype == "bfloat" else "trunc(value++)"
    )
    assert f"{canonical} result = {expression};" in generated
    assert generated.count("value++") == 1
    assert "__metal_" not in generated and "<unknown>" not in generated


@pytest.mark.parametrize("intrinsic", ("__metal_round", "__metal_unknown"))
def test_trunc_rejects_mismatched_materialized_wrapper(intrinsic):
    with pytest.raises(MetalStandardLibraryWrapperLoweringError):
        convert_without_preprocessing(f"""
        namespace metal {{
        METAL_FUNC float trunc(float value) {{ return {intrinsic}(value); }}
        }}
        float evaluate(float value) {{ return metal::trunc(value); }}
        """)


@pytest.mark.parametrize(
    "body",
    (
        "return __metal_trunc(value) + 1.0f;",
        "return __metal_trunc(value + 1.0f);",
        "value += 1.0f; return __metal_trunc(value);",
        "return __metal_trunc(value++);",
        "return __metal_trunc(half(value));",
        "return __metal_trunc(value, value++ > 0.0f);",
    ),
)
def test_trunc_does_not_discard_additional_wrapper_computation(body):
    with pytest.raises(MetalStandardLibraryWrapperLoweringError):
        convert_without_preprocessing(f"""
        namespace metal {{ METAL_FUNC float trunc(float value) {{ {body} }} }}
        float evaluate(float value) {{ return metal::trunc(value); }}
        """)


@pytest.mark.parametrize(
    "expression",
    (
        "__metal_trunc(value)",
        "float(__metal_trunc(value))",
        "static_cast<float>(__metal_trunc(value))",
    ),
)
def test_trunc_accepts_identity_return_conversions(expression):
    generated = normalize(convert_without_preprocessing(f"""
    namespace metal {{ METAL_FUNC float trunc(float value) {{ return {expression}; }} }}
    float evaluate(float value) {{ return metal::trunc(value); }}
    """))
    assert "return trunc(value);" in generated


@pytest.mark.parametrize(
    "mode",
    (
        "true",
        "false",
        "__METAL_MAYBE_FAST_MATH__",
        "__METAL_FAST_MATH__",
        "__METAL_PRECISE_MATH__",
    ),
)
def test_trunc_accepts_header_math_mode_metadata(mode):
    generated = normalize(convert_without_preprocessing(f"""
    typedef bfloat bfloat16_t;
    namespace metal {{
    METAL_FUNC bfloat16_t trunc(bfloat16_t x) {{
      return (bfloat16_t)__metal_trunc((float)x, {mode});
    }}
    }}
    bfloat16_t evaluate(bfloat16_t value) {{ return metal::trunc(value); }}
    """))
    assert "return bfloat16(trunc(float(value)));" in generated
    assert "__METAL_" not in generated


@pytest.mark.parametrize("namespace", ("metal", "metal::fast", "metal::precise", ""))
def test_trunc_infers_bare_vector_and_aliased_scalar(namespace):
    qualifier = namespace + "::" if namespace else ""
    generated = normalize(convert_without_preprocessing(f"""
    using Value = float;
    float4 evaluate(Value value, float4 values) {{
      auto scalar = {qualifier}trunc(value);
      auto vector = {qualifier}trunc(values);
      return vector + float4(scalar);
    }}"""))
    assert "float scalar = trunc(value);" in generated
    assert "vec4 vector = trunc(values);" in generated


def _source(dtype):
    return f"""#include <metal_stdlib>
using namespace metal;
typedef {dtype} Value;
float trunc(float value) {{ return value + 8.0f; }}
namespace custom {{ float trunc(float value) {{ return value - 8.0f; }} }}
uint payload(float value) {{
  return metal::isnan(value) ? 0x7fc00000u : as_type<uint>(value);
}}
kernel void truncation(const device uint* values [[buffer(0)]],
                       device uint* results [[buffer(1)]],
                       uint i [[thread_position_in_grid]]) {{
    uint cursor = i;
    Value value = Value(as_type<float>(values[cursor++]));
    auto standard = metal::trunc(value);
    auto fast = metal::fast::trunc(value);
    auto precise_result = metal::precise::trunc(value);
    float4 vector = metal::trunc(float4(float(value), -float(value), 1.75f, -1.75f));
    results[10u * i + 1u] = payload(float(standard));
    results[10u * i + 2u] = payload(float(fast));
    results[10u * i + 3u] = payload(float(precise_result));
    results[10u * i + 4u] = payload(vector.x);
    results[10u * i + 5u] = payload(vector.y);
    results[10u * i + 6u] = payload(vector.z);
    results[10u * i + 7u] = payload(vector.w);
    results[10u * i + 8u] = cursor - i;
    results[10u * i + 9u] = payload(::trunc(1.5f));
    results[10u * i + 10u] = payload(custom::trunc(1.5f));
}}
"""


def _words(dtype, start):
    if dtype == "half":
        return [_widen(word) for word in range(start, start + 32768)]
    if dtype == "bfloat":
        return [word << 16 for word in range(start, start + 32768)]
    words = [
        sign | (exponent << 23) | mantissa
        for sign in (0, 0x80000000)
        for exponent in range(256)
        for mantissa in (0, 1, 0x3FFFFF, 0x400000, 0x7FFFFE, 0x7FFFFF)
    ]
    rng = random.Random(2091)
    words.extend(rng.getrandbits(32) for _ in range(256))
    return words


def _truncated(word):
    exponent = (word >> 23) & 0xFF
    if exponent == 0xFF:
        return 0x7FC00000 if word & 0x7FFFFF else word
    shift = exponent - 127
    if shift < 0:
        return word & 0x80000000
    if shift < 23:
        return word & ~((1 << (23 - shift)) - 1)
    return word


@pytest.mark.parametrize("target", ("metal", "opengl", "directx"))
@pytest.mark.parametrize("dtype", ("float", "half", "bfloat"))
def test_qualified_trunc_translates_and_compiles(tmp_path, target, dtype):
    source = tmp_path / "source.metal"
    source.write_text(_source(dtype), encoding="utf-8")
    generated = translate(str(source), backend=target, format_output=False)
    assert "metal_overload_" in generated
    assert not re.search(r"\b__metal_", generated) and "<unknown>" not in generated
    tool = {"metal": "xcrun", "opengl": "glslangValidator", "directx": "dxc"}[target]
    if not shutil.which(tool):
        pytest.skip(f"{tool} is unavailable")
    _, module = _compile(
        generated,
        target,
        tmp_path,
        metal_compile_flags=("-std=metal3.1", "-fno-fast-math"),
        directx_compile_flags=("-enable-16bit-types",),
    )
    assert module.is_file() and module.stat().st_size


@pytest.mark.parametrize(
    "dtype,start",
    (("float", 0), ("half", 0), ("half", 32768), ("bfloat", 0), ("bfloat", 32768)),
)
def test_trunc_executes_with_exact_finite_bits(tmp_path, dtype, start):
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 for required truncation execution")
    target = {"darwin": "metal", "win32": "directx", "linux": "opengl"}[sys.platform]
    source = _source(dtype)
    _, descriptor, package = _package(
        tmp_path, target, "uint", (1, 1, 1), source=source, software_subgroups=False
    )
    words = _words(dtype, start)
    guard = 0x6A15BEEF
    expected = [guard]
    for word in words:
        expected.extend([_truncated(word)] * 4)
        expected.extend(
            (
                _truncated(word ^ 0x80000000),
                0x3F800000,
                0xBF800000,
                1,
                0x41180000,
                0xC0D00000,
            )
        )
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
        original_source=source,
        original_entry="truncation",
        metal_compile_flags=("-std=metal3.1",),
        validate=_validate_half,
    )


def test_trunc_native_gate_is_required():
    from tools import ci_coverage

    workflow = Path(".github/workflows/demo-project-testing.yml").read_text()
    step = ci_coverage.workflow_step_section(
        workflow, "Validate Metal builtin ownership"
    )
    assert f'{REQUIRE_ENV}: "1"' in step
    assert "tests/test_translator/test_metal_trunc.py" in step

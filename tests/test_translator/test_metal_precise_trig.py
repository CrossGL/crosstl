"""Precise sine/cosine over binary32 exponents, including native readbacks."""

import hashlib
import json
import math
import os
import random
import struct
import sys
from decimal import ROUND_HALF_EVEN, Decimal, localcontext
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace

import pytest

from crosstl import translate
from crosstl.backend.Metal.MetalCrossGLCodeGen import (
    MetalPreciseMathLoweringError,
    MetalToCrossGLConverter,
)
from crosstl.backend.Metal.MetalLexer import MetalLexer
from crosstl.backend.Metal.MetalParser import MetalParser
from crosstl.project import MetalComputeRuntime
from crosstl.project.host_reflection import reflect_target_host_interface
from crosstl.project.native_runtime_drivers import (
    DirectXComputeRuntime,
    OpenGLComputeRuntime,
)
from crosstl.project.runtime_verification import (
    NativeRuntimeBufferBinding,
    NativeRuntimeDispatchRequest,
    RuntimeDispatchGeometry,
    RuntimeResourceBinding,
)
from tests.test_translator.test_metal_builtin_ownership import _compile

REQUIRE_ENV = "CROSTL_REQUIRE_METAL_PRECISE_TRIG"
GUARD = [0x6A15BEEF] * 32
LANE_SIGNS = (0, 0, 1, 0, 1, 0, 0, 1, 0, 1)


def _source():
    source = """#include <metal_stdlib>
using namespace metal;
float record(thread uint& count, float value) { count += 1; return value; }
kernel void trigonometry(device const uint* values [[buffer(0)]],
                         device uint* results [[buffer(1)]],
                         uint i [[thread_position_in_grid]]) {
    float x = as_type<float>(values[i]);
"""
    for index, operation in enumerate(("sin", "cos")):
        source += f"""
    uint count_{operation} = 0;
    float2 pair_{operation} = metal::precise::{operation}(float2(x, -x));
    float3 triple_{operation} = precise::{operation}(float3(x, -x, x));
    float4 quad_{operation} = metal::precise::{operation}(float4(record(count_{operation}, x), -x, x, -x));
"""
        outputs = [f"metal::precise::{operation}(x)"]
        outputs += [f"pair_{operation}.{lane}" for lane in "xy"]
        outputs += [f"triple_{operation}.{lane}" for lane in "xyz"]
        outputs += [f"quad_{operation}.{lane}" for lane in "xyzw"]
        for lane, expression in enumerate(outputs):
            source += f"    results[22 * i + {11 * index + lane}] = as_type<uint>({expression});\n"
        source += f"    results[22 * i + {11 * index + 10}] = count_{operation};\n"
    return source + "}\n"


def _translate(tmp_path, source, target):
    path = tmp_path / "trigonometry.metal"
    path.write_text(source, encoding="utf-8")
    return translate(str(path), backend=target, format_output=False)


@pytest.mark.parametrize("target", ["directx", "opengl", "metal"])
def test_precise_trig_helpers_compile(tmp_path, target):
    generated = _translate(tmp_path, _source(), target)
    for operation in ("sin", "cos"):
        for suffix in ("", "2", "3", "4"):
            assert f"metal_precise_{operation}_float{suffix}(" in generated
    assert "Arm Limited" in generated
    assert "Permission is hereby granted" in generated
    assert "uint64" not in generated and "double" not in generated
    _compile(generated, target, tmp_path)


@pytest.mark.parametrize("operation", ["sin", "cos"])
def test_precise_trig_preserves_other_modes_and_user_functions(tmp_path, operation):
    generated = _translate(
        tmp_path,
        f"""
        float {operation}(float x) {{ return x + 3.0f; }}
        float explicit_mode(float x) {{ return metal::precise::{operation}(x); }}
        float imported_mode(float x) {{ return precise::{operation}(x); }}
        float normal_mode(float x) {{ return metal::{operation}(x); }}
        float fast_mode(float x) {{ return metal::fast::{operation}(x); }}
        float own_function(float x) {{ return ::{operation}(x); }}
    """,
        "crossgl",
    )
    assert generated.count(f"return __crossgl_metal_precise_{operation}_float(x);") == 2
    assert generated.count(f"return {operation}(x);") == 2
    assert f"return {operation}__metal_overload_1(x);" in generated
    other = "sin" if operation == "cos" else "cos"
    assert f"__crossgl_metal_precise_{other}_" not in generated
    assert "Copyright (c) 2018-2024, Arm Limited." in generated
    assert "Permission is hereby granted" in generated


def test_precise_trig_names_do_not_collide(tmp_path):
    generated = _translate(
        tmp_path,
        """
        float __crossgl_metal_precise_sin_float(float x) { return x; }
        float __crossgl_metal_trig_evaluate(float x) { return x; }
        float __crossgl_metal_trig_word(float x) { return x; }
        float2 evaluate(float2 x) { return metal::precise::sin(x); }
    """,
        "crossgl",
    )
    assert "float __crossgl_metal_precise_sin_float_(float value)" in generated
    assert "uint __crossgl_metal_trig_word_(uint index)" in generated
    assert "return __crossgl_metal_trig_evaluate_(value, false);" in generated


def test_precise_trig_generation_state_does_not_leak():
    converter = MetalToCrossGLConverter()
    for expression, required in (
        ("metal::precise::sin(x)", True),
        ("metal::sin(x)", False),
    ):
        source = f"float evaluate(float x) {{ return {expression}; }}"
        ast = MetalParser(MetalLexer(source).tokenize()).parse()
        generated = converter.generate(ast)
        assert ("@source_license(arm_optimized)" in generated) == required
        assert ("__crossgl_metal_trig_evaluate" in generated) == required


@pytest.mark.parametrize("operation", ["sin", "cos"])
@pytest.mark.parametrize("operand", ["int", "double", "half", "Payload"])
def test_precise_trig_rejects_unrepresentable_operands(tmp_path, operation, operand):
    source = "struct Payload { float value; };\n" + (
        f"{operand} evaluate({operand} x) {{ return metal::precise::{operation}(x); }}"
    )
    with pytest.raises(MetalPreciseMathLoweringError) as error:
        _translate(tmp_path, source, "crossgl")
    assert error.value.operation == operation
    assert (
        error.value.project_diagnostic_code
        == "project.translate.metal-precise-math-unsupported"
    )


def _float(bits):
    return struct.unpack("<f", struct.pack("<I", bits))[0]


def _bits(value):
    return struct.unpack("<I", struct.pack("<f", value))[0]


@lru_cache(maxsize=1)
def _pi():
    # Machin's identity is independent of the implementation's 4/pi table.
    with localcontext() as context:
        context.prec = 170

        def arctangent(inverse):
            x = Decimal(1) / inverse
            power = x
            total = x
            for index in range(1, 1000):
                power *= -x * x
                term = power / (2 * index + 1)
                total += term
                if abs(term) < Decimal("1e-170"):
                    return total
            raise AssertionError("arctangent series did not converge")

        return 16 * arctangent(Decimal(5)) - 4 * arctangent(Decimal(239))


def _round_decimal(value):
    negative = value.is_signed()
    magnitude = abs(value)
    approximate = _bits(float(magnitude))
    candidates = range(max(0, approximate - 1), approximate + 2)
    rounded = min(
        candidates,
        key=lambda word: (abs(Decimal.from_float(_float(word)) - magnitude), word & 1),
    )
    return rounded | (0x80000000 if negative else 0)


@lru_cache(maxsize=None)
def _oracle(bits):
    magnitude = bits & 0x7FFFFFFF
    if magnitude >= 0x7F800000:
        return 0x7FC00000, 0x7FC00000
    if magnitude == 0:
        return bits, 0x3F800000
    with localcontext() as context:
        context.prec = 160
        value = Decimal.from_float(_float(magnitude))
        half_pi = _pi() / 2
        quotient = (value / half_pi).to_integral_value(rounding=ROUND_HALF_EVEN)
        reduced = value - quotient * half_pi
        sine = sin_term = reduced
        cosine = cos_term = Decimal(1)
        for index in range(1, 100):
            sin_term *= -reduced * reduced / ((2 * index) * (2 * index + 1))
            cos_term *= -reduced * reduced / ((2 * index - 1) * (2 * index))
            sine += sin_term
            cosine += cos_term
            if max(abs(sin_term), abs(cos_term)) < Decimal("1e-155"):
                break
        else:
            raise AssertionError("trigonometric series did not converge")
        quadrant = int(quotient) & 3
        sine, cosine = (
            (sine, cosine),
            (cosine, -sine),
            (-sine, -cosine),
            (-cosine, sine),
        )[quadrant]
        if bits >> 31:
            sine = -sine
        return _round_decimal(sine), _round_decimal(cosine)


def _inputs():
    words = {0, 1, 0x007FFFFF, 0x7F7FFFFF, 0x7F800000, 0x7FC12345}
    random_source = random.Random(1964)
    for exponent in range(1, 255):
        for mantissa in (0, 1, 0x3FFFFF, 0x7FFFFE, 0x7FFFFF):
            words.add((exponent << 23) | mantissa)
        for _ in range(8):
            words.add((exponent << 23) | random_source.getrandbits(23))
    for index in range(257):
        words.add(_bits(index / 32))
    for boundary in (0x39800000, 0x3F490FDB, 0x40000000):
        words.update(range(boundary - 4, boundary + 5))
    # Neighbors of multiples of pi/2 at every large exponent, not only random
    # significands. The oracle reduces the actual represented input exactly.
    with localcontext() as context:
        context.prec = 160
        for exponent in range(128):
            half_pi = _pi() / 2
            quotient = (Decimal(2) ** exponent / half_pi).to_integral_value()
            center = _bits(float(max(1, quotient) * half_pi))
            words.update(range(center - 4, center + 5))
    for value in (1e8, 1e9, 1e10, 1e20, 1e30):
        words.add(_bits(value))
    return sorted(words | {word | 0x80000000 for word in words})


def _expected(inputs):
    output = []
    for word in inputs:
        for operation in range(2):
            output.extend(
                _oracle(word ^ (sign << 31))[operation] for sign in LANE_SIGNS
            )
            output.append(1)
    return output + GUARD


def _check(actual, expected):
    assert len(actual) == len(expected), "output size"
    assert actual[-len(GUARD) :] == GUARD, "output guard"
    maximum = 0
    for index, (got, want) in enumerate(
        zip(actual[: -len(GUARD)], expected[: -len(GUARD)])
    ):
        if index % 11 == 10:
            assert got == 1, "operand evaluation count"
        elif want & 0x7FFFFFFF > 0x7F800000:
            assert got & 0x7FFFFFFF > 0x7F800000, "NaN classification"
        elif want & 0x7FFFFFFF == 0:
            assert got == want, "zero sign"
        else:
            assert got >> 31 == want >> 31, "result sign"
            error = abs(got - want)
            assert error <= 4, (index, hex(got), hex(want), error)
            maximum = max(maximum, error)
    return maximum


def test_precise_trig_decimal_oracle():
    assert str(_pi()).startswith("3.14159265358979323846264338327950288419716939937510")
    for word in (0, 0x80000000, 1, 0x7F7FFFFF, _bits(1e30), _bits(math.pi)):
        sine, cosine = _oracle(word)
        assert sine == _bits(math.sin(_float(word)))
        assert cosine == _bits(math.cos(_float(word)))


def test_precise_trig_corpus_covers_every_exponent_and_both_signs():
    inputs = _inputs()
    words = set(inputs)
    assert len(inputs) < 65536
    assert {(word >> 23) & 255 for word in inputs} == set(range(256))
    assert all(word ^ 0x80000000 in words for word in inputs)
    for value in (0.0, -0.0, 1e8, 1e9, 1e10, 1e20, 1e30):
        assert _bits(value) in words


@pytest.mark.parametrize(
    "corruption", ["value", "size", "guard", "count", "zero", "nan"]
)
def test_precise_trig_verifier_rejects_corruption(corruption):
    expected = _expected([0x80000000, _bits(1e20), 0x7F800000])
    actual = expected.copy()
    if corruption == "size":
        actual.pop()
    elif corruption == "guard":
        actual[-1] ^= 1
    elif corruption == "count":
        actual[10] = 2
    elif corruption == "zero":
        actual[0] = 0
    elif corruption == "nan":
        actual[44] = 0
    else:
        actual[22] += 5
    with pytest.raises(AssertionError):
        _check(actual, expected)


def _execute(directory, target, source, inputs, expected):
    directory.mkdir()
    artifact, module = _compile(source, target, directory)
    assert module.is_file(), "native compiler required"
    layouts = (
        {
            resource["name"]: resource["scalarLayout"]
            for resource in reflect_target_host_interface(artifact, target=target)[
                "resources"
            ]
        }
        if target == "metal"
        else {}
    )
    initial = [0xDEADBEEF] * (len(expected) - len(GUARD)) + GUARD
    buffers = {}
    for slot, (name, words) in enumerate((("values", inputs), ("results", initial))):
        buffers[name] = NativeRuntimeBufferBinding(
            name=name,
            binding=RuntimeResourceBinding(
                name=name,
                kind="buffer",
                set=0,
                binding=slot,
                type_name=("RW" if slot else "") + "StructuredBuffer<uint>",
                access="read_write" if slot else "read",
                metadata={"scalarLayout": layouts[name]} if layouts else {},
            ),
            source="expectedOutput" if slot else "input",
            dtype="uint32",
            shape=(len(words),),
            value=words,
        )
    entry = {"directx": "CSMain", "opengl": "main", "metal": "trigonometry"}[target]
    request = NativeRuntimeDispatchRequest(
        target=target,
        artifact={"target": target},
        artifact_path=artifact,
        module_path=module,
        loaded_artifact=source if target == "opengl" else module.read_bytes(),
        buffers=buffers,
        constants={},
        entry_point=entry,
        dispatch=RuntimeDispatchGeometry(
            entry_point=entry,
            workgroup_size=(1, 1, 1),
            workgroup_count=(len(inputs), 1, 1),
        ),
    )
    runtime = {
        "directx": DirectXComputeRuntime,
        "metal": MetalComputeRuntime,
        "opengl": lambda: OpenGLComputeRuntime(context_backends=("egl",)),
    }[target]()
    state = SimpleNamespace(details={})
    actual = runtime.dispatch(None, state, request)["results"]["values"]
    (directory / "readback.bin").write_bytes(struct.pack(f"<{len(actual)}I", *actual))
    if target == "metal":
        assert (
            state.details["metalRuntime"]["librarySHA256"]
            == hashlib.sha256(module.read_bytes()).hexdigest()
        )
    return {
        "maxUlpError": _check(actual, expected),
        "artifactSha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
        "moduleSha256": hashlib.sha256(module.read_bytes()).hexdigest(),
        "runtime": state.details,
    }


def test_precise_trig_executes(tmp_path):
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 for required native trigonometry")
    target = {"win32": "directx", "linux": "opengl", "darwin": "metal"}[sys.platform]
    inputs = _inputs()
    expected = _expected(inputs)
    (tmp_path / "inputs.bin").write_bytes(struct.pack(f"<{len(inputs)}I", *inputs))
    (tmp_path / "expected.bin").write_bytes(
        struct.pack(f"<{len(expected)}I", *expected)
    )
    generated = _translate(tmp_path, _source(), target)
    records = {
        "generated": _execute(
            tmp_path / "generated", target, generated, inputs, expected
        )
    }
    if target == "metal":
        records["original"] = _execute(
            tmp_path / "original", target, _source(), inputs, expected
        )
    (tmp_path / "evidence.json").write_text(
        json.dumps(
            {
                "target": target,
                "inputCount": len(inputs),
                "outputCount": len(expected),
                "oracle": "160-digit Decimal Machin pi and Taylor series, binary32 RNE",
                "records": records,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def test_ci_requires_precise_trig_execution():
    workflow = (
        Path(__file__).resolve().parents[2]
        / ".github/workflows/mlx-project-porting.yml"
    ).read_text()
    step = workflow.split("      - name: Validate Metal builtin ownership\n", 1)[
        1
    ].split("      - name:", 1)[0]
    assert "if:" not in step and "continue-on-error" not in step
    assert f'{REQUIRE_ENV}: "1"' in step
    assert "tests/test_translator/test_metal_precise_trig.py" in step
    assert "pytest -q -n auto" in step

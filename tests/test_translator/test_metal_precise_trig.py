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
@pytest.mark.parametrize("operand", ["int", "double", "Payload"])
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


def _execute(
    directory,
    target,
    source,
    inputs,
    expected,
    *,
    metal_entry="trigonometry",
    check_outputs=_check,
    metal_compile_flags=(),
    directx_compile_flags=(),
):
    directory.mkdir()
    artifact, module = _compile(
        source,
        target,
        directory,
        metal_compile_flags=metal_compile_flags,
        directx_compile_flags=directx_compile_flags,
    )
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
    entry = {"directx": "CSMain", "opengl": "main", "metal": metal_entry}[target]
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
        "maxUlpError": check_outputs(actual, expected),
        "artifactSha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
        "moduleSha256": hashlib.sha256(module.read_bytes()).hexdigest(),
        "metalCompileFlags": list(metal_compile_flags) if target == "metal" else [],
        "directxCompileFlags": (
            list(directx_compile_flags) if target == "directx" else []
        ),
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


PROMOTED_OPERATIONS = ("sin", "cos", "asin", "acos", "atan", "acosh")
PROMOTED_SHAPES = (("half", 1), ("bfloat", 1))
PROMOTED_INPUTS = (
    0,
    0x8000,
    1,
    0x8001,
    0x3555,
    0x3C00,
    0x3F80,
    0x4000,
    0x4200,
    0x7BFF,
    0x7C00,
    0x7D01,
    0x7F80,
    0x7FC1,
    0xFBFF,
    0xFFFF,
)


def _promotion_source(operand, width):
    suffix = str(width) if width > 1 else ""
    stride = 2 * width * len(PROMOTED_OPERATIONS) + 1
    source = f"""#include <metal_stdlib>
using namespace metal;
using Narrow = {operand}{suffix};
Narrow record(thread uint& count, Narrow value) {{ count += 1; return value; }}
kernel void promote_precise(const device uint* values [[buffer(0)]],
                            device uint* results [[buffer(1)]],
                            uint i [[thread_position_in_grid]]) {{
    Narrow value = Narrow(as_type<{operand}>(ushort(values[i])));
    uint count = 0;
"""
    for index, operation in enumerate(PROMOTED_OPERATIONS):
        source += f"""
    auto implicit_{operation} = metal::precise::{operation}(record(count, value));
    auto explicit_{operation} = metal::precise::{operation}(float{suffix}(value));
"""
        for lane in range(width):
            component = "." + "xyzw"[lane] if width > 1 else ""
            offset = 2 * (index * width + lane)
            source += f"    results[{stride} * i + {offset}] = as_type<uint>(implicit_{operation}{component});\n"
            source += f"    results[{stride} * i + {offset + 1}] = as_type<uint>(explicit_{operation}{component});\n"
    return source + f"    results[{stride} * i + {stride - 1}] = count;\n}}\n"


def _promotion_expected(operand, width):
    results = []
    for word in PROMOTED_INPUTS:
        value = (
            struct.unpack("<e", struct.pack("<H", word))[0]
            if operand == "half"
            else _float(word << 16)
        )
        for operation in PROMOTED_OPERATIONS:
            try:
                result = getattr(math, operation)(value)
            except ValueError:
                result = math.nan
            results.extend([_bits(result)] * (2 * width))
        results.append(len(PROMOTED_OPERATIONS))
    return results + GUARD


def _check_promotions(actual, expected, width, *, original=False):
    assert len(actual) == len(expected), "output size"
    assert actual[-len(GUARD) :] == GUARD, "output guard"
    stride = 2 * width * len(PROMOTED_OPERATIONS) + 1
    maximum = 0
    for base in range(0, len(expected) - len(GUARD), stride):
        assert actual[base + stride - 1] == len(
            PROMOTED_OPERATIONS
        ), "operand evaluation count"
        for offset in range(0, stride - 1, 2):
            got, explicit = actual[base + offset : base + offset + 2]
            want = expected[base + offset]
            assert got == explicit, "implicit and explicit promotion differ"
            # Keep the established original-Metal atan control separate from
            # the stricter generated-output contract (MSL 8.1/8.5).
            operation = PROMOTED_OPERATIONS[offset // (2 * width)]
            if original and operation == "atan":
                from tests.test_translator.test_metal_precise_atan import (
                    _check_original_word,
                )

                maximum = max(maximum, _check_original_word(got, want))
                continue
            if want & 0x7FFFFFFF > 0x7F800000:
                assert got & 0x7FFFFFFF > 0x7F800000, "NaN classification"
            elif want & 0x7FFFFFFF in (0, 0x7F800000):
                assert got == want, "zero or infinity"
            else:
                assert got >> 31 == want >> 31, "result sign"
                error = abs(got - want)
                assert error <= 4, (base, offset, hex(got), hex(want), error)
                maximum = max(maximum, error)
    return maximum


@pytest.mark.parametrize("target", ("directx", "opengl", "metal"))
@pytest.mark.parametrize("operand,width", PROMOTED_SHAPES)
def test_precise_math_promotes_narrow_operands(tmp_path, target, operand, width):
    generated = _translate(tmp_path, _promotion_source(operand, width), target)
    suffix = str(width) if width > 1 else ""
    for operation in PROMOTED_OPERATIONS:
        assert f"metal_precise_{operation}_float{suffix}(" in generated


@pytest.mark.parametrize("operation", ("sin", "cos", "atan", "acosh"))
@pytest.mark.parametrize("width", (2, 3, 4))
def test_precise_math_rejects_implicit_narrow_vector_conversion(
    tmp_path, operation, width
):
    source = f"float{width} evaluate(half{width} value) {{ return metal::precise::{operation}(value); }}"
    with pytest.raises(MetalPreciseMathLoweringError):
        _translate(tmp_path, source, "crossgl")


@pytest.mark.parametrize(
    "corruption", ("pair", "value", "size", "guard", "count", "zero", "nan")
)
def test_precise_promotion_verifier_rejects_corruption(corruption):
    expected = _promotion_expected("half", 1)
    actual = expected.copy()
    if corruption == "pair":
        actual[2] ^= 1
    elif corruption == "value":
        actual[2:4] = [expected[2] + 5] * 2
    elif corruption == "size":
        actual.pop()
    elif corruption == "guard":
        actual[-1] ^= 1
    elif corruption == "count":
        actual[12] = 0
    elif corruption == "zero":
        actual[:2] = [0x80000000] * 2
    else:
        actual[10:12] = [0] * 2
    with pytest.raises(AssertionError):
        _check_promotions(actual, expected, 1)


def test_precise_promotion_original_control_does_not_relax_generated_checks():
    expected = _promotion_expected("bfloat", 1)
    actual = expected.copy()
    for index in (1, 2, 3):
        actual[13 * index + 8 : 13 * index + 10] = [0, 0]
    _check_promotions(actual, expected, 1, original=True)
    with pytest.raises(AssertionError):
        _check_promotions(actual, expected, 1)


@pytest.mark.parametrize("operand,width", PROMOTED_SHAPES)
def test_precise_math_promotions_execute_natively(tmp_path, operand, width):
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 for required native precise promotions")
    target = {"win32": "directx", "linux": "opengl", "darwin": "metal"}[sys.platform]
    source = _promotion_source(operand, width)
    generated = _translate(tmp_path, source, target)
    expected = _promotion_expected(operand, width)
    (tmp_path / "inputs.bin").write_bytes(
        struct.pack(f"<{len(PROMOTED_INPUTS)}I", *PROMOTED_INPUTS)
    )
    (tmp_path / "expected.bin").write_bytes(
        struct.pack(f"<{len(expected)}I", *expected)
    )
    records = {}
    for label, code in (("generated", generated), ("original", source)):
        if label == "original" and target != "metal":
            continue
        records[label] = _execute(
            tmp_path / label,
            target,
            code,
            PROMOTED_INPUTS,
            expected,
            metal_entry="promote_precise",
            check_outputs=lambda actual, reference: _check_promotions(
                actual, reference, width, original=label == "original"
            ),
            metal_compile_flags=("-fno-fast-math",),
            directx_compile_flags=("-enable-16bit-types",),
        )
    (tmp_path / "evidence.json").write_text(
        json.dumps(
            {
                "target": target,
                "operand": operand,
                "width": width,
                "oracle": (
                    "binary32-rounded Python math with at most four ULP error; exact implicit/explicit pairs"
                ),
                "originalMetalControl": (
                    "The existing original atan control permits subnormal flushing and either zero sign; generated checks remain exact."
                ),
                "records": records,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def test_ci_requires_precise_trig_execution():
    workflow = (
        Path(__file__).resolve().parents[2]
        / ".github/workflows/demo-project-testing.yml"
    ).read_text()
    step = workflow.split("      - name: Validate Metal builtin ownership\n", 1)[
        1
    ].split("      - name:", 1)[0]
    assert "if:" not in step and "continue-on-error" not in step
    assert f'{REQUIRE_ENV}: "1"' in step
    assert "tests/test_translator/test_metal_precise_trig.py" in step
    assert "pytest -q -n auto" in step


NARROW_INPUTS = tuple(
    sorted(
        set(PROMOTED_INPUTS)
        | {
            (center + offset) ^ sign
            for center in (0x32B3, 0x4578, 0x2CAF, 0x2F4C, 0x3094, 0x7D29)
            for offset in (-1, 0, 1)
            for sign in (0, 0x8000)
        }
    )
)


def _narrow_word(word, operand):
    if operand == "half":
        return struct.unpack("<H", struct.pack("<e", _float(word)))[0]
    return (word + 0x7FFF + ((word >> 16) & 1)) >> 16


def _narrow_source(operand):
    source = f"""#include <metal_stdlib>
using namespace metal;
using Narrow = {operand};
kernel void narrow_precise(const device uint* values [[buffer(0)]],
                          device uint* results [[buffer(1)]],
                          uint i [[thread_position_in_grid]]) {{
    Narrow value = as_type<{operand}>(ushort(values[i]));
"""
    stride = 2 * len(PROMOTED_OPERATIONS)
    for index, operation in enumerate(PROMOTED_OPERATIONS):
        source += f"""
    float wide_{operation} = metal::precise::{operation}(value);
    Narrow narrow_{operation} = Narrow(wide_{operation});
    results[{stride} * i + {2 * index}] = as_type<uint>(wide_{operation});
    results[{stride} * i + {2 * index + 1}] = uint(as_type<ushort>(narrow_{operation}));
"""
    return source + "}\n"


def _narrow_expected(operand):
    expected = []
    for word in NARROW_INPUTS:
        value = (
            struct.unpack("<e", struct.pack("<H", word))[0]
            if operand == "half"
            else _float(word << 16)
        )
        for operation in PROMOTED_OPERATIONS:
            try:
                result = getattr(math, operation)(value)
            except ValueError:
                result = math.nan
            wide = _bits(result)
            expected.extend((wide, _narrow_word(wide, operand)))
    return expected + GUARD


def _check_narrowing(actual, expected, operand, *, original=False):
    assert len(actual) == len(expected), "output size"
    assert actual[-len(GUARD) :] == GUARD, "output guard"
    maximum = 0
    for offset in range(0, len(expected) - len(GUARD), 2):
        wide, narrow = actual[offset : offset + 2]
        want = expected[offset]
        operation = PROMOTED_OPERATIONS[(offset // 2) % len(PROMOTED_OPERATIONS)]
        if original and operation == "atan":
            from tests.test_translator.test_metal_precise_atan import (
                _check_original_word,
            )

            maximum = max(maximum, _check_original_word(wide, want))
        elif want & 0x7FFFFFFF > 0x7F800000:
            assert wide & 0x7FFFFFFF > 0x7F800000, "NaN classification"
        elif want & 0x7FFFFFFF < 0x00800000 or want & 0x7FFFFFFF == 0x7F800000:
            assert wide == want, "zero, subnormal or infinity"
        else:
            assert wide >> 31 == want >> 31, "result sign"
            error = abs(wide - want)
            assert error <= 4, (offset, hex(wide), hex(want), error)
            maximum = max(maximum, error)

        assert 0 <= narrow <= 0xFFFF, "narrow storage width"
        if wide & 0x7FFFFFFF > 0x7F800000:
            infinity = 0x7C00 if operand == "half" else 0x7F80
            assert narrow & 0x7FFF > infinity, "narrow NaN classification"
        else:
            # The float operation has its own accuracy bound. Its actual result
            # must still be narrowed exactly, including near rounding midpoints.
            assert narrow == _narrow_word(wide, operand), "narrow conversion"
    return maximum


@pytest.mark.parametrize("operand", ("half", "bfloat"))
@pytest.mark.parametrize("target", ("directx", "opengl", "metal"))
def test_precise_narrowing_compiles(tmp_path, operand, target):
    generated = _translate(tmp_path, _narrow_source(operand), target)
    _compile(
        generated,
        target,
        tmp_path,
        directx_compile_flags=("-enable-16bit-types",),
    )


@pytest.mark.parametrize(
    "operand,wide,expected",
    (
        ("half", 0x3E54D000, 0x32A6),
        ("half", 0x3E54D001, 0x32A7),
        ("half", 0x3E54CFFF, 0x32A6),
        ("bfloat", 0xBED28000, 0xBED2),
        ("bfloat", 0xBED28001, 0xBED3),
        ("bfloat", 0xBED27FFF, 0xBED2),
        ("half", 0x80000000, 0x8000),
        ("bfloat", 0x80000000, 0x8000),
    ),
)
def test_precise_narrowing_reference_rounds_midpoints(operand, wide, expected):
    assert _narrow_word(wide, operand) == expected


def test_precise_narrowing_original_policy_does_not_relax_generated_checks():
    expected = _narrow_expected("bfloat")
    actual = expected.copy()
    offset = 2 * (len(PROMOTED_OPERATIONS) * NARROW_INPUTS.index(1) + 4)
    actual[offset : offset + 2] = [0, 0]
    _check_narrowing(actual, expected, "bfloat", original=True)
    with pytest.raises(AssertionError):
        _check_narrowing(actual, expected, "bfloat")


def test_precise_narrowing_checks_the_observed_float_result():
    expected = _narrow_expected("half")
    actual = expected.copy()
    offset = 2 * len(PROMOTED_OPERATIONS) * NARROW_INPUTS.index(0x32B3)
    assert expected[offset : offset + 2] == [0x3E54D000, 0x32A6]
    actual[offset : offset + 2] = [0x3E54D001, 0x32A7]
    assert _check_narrowing(actual, expected, "half") == 1
    actual[offset + 1] = expected[offset + 1]
    with pytest.raises(AssertionError, match="narrow conversion"):
        _check_narrowing(actual, expected, "half")


@pytest.mark.parametrize("operand", ("half", "bfloat"))
@pytest.mark.parametrize(
    "corruption", ("value", "narrow", "size", "guard", "width", "zero", "nan")
)
def test_precise_narrowing_verifier_rejects_corruption(operand, corruption):
    expected = _narrow_expected(operand)
    _check_narrowing(expected, expected, operand)
    actual = expected.copy()
    if corruption == "value":
        actual[2] += 5
    elif corruption == "narrow":
        actual[3] += 1
    elif corruption == "size":
        actual.pop()
    elif corruption == "guard":
        actual[-1] ^= 1
    elif corruption == "width":
        actual[3] |= 0x10000
    elif corruption == "zero":
        actual[:2] = [0x80000000, 0x8000]
    else:
        actual[11] = 0
    with pytest.raises(AssertionError):
        _check_narrowing(actual, expected, operand)


@pytest.mark.parametrize("operand", ("half", "bfloat"))
def test_precise_narrowing_executes_natively(tmp_path, operand):
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 for required native precise narrowing")
    target = {"win32": "directx", "linux": "opengl", "darwin": "metal"}[sys.platform]
    source = _narrow_source(operand)
    generated = _translate(tmp_path, source, target)
    expected = _narrow_expected(operand)
    (tmp_path / "inputs.bin").write_bytes(
        struct.pack(f"<{len(NARROW_INPUTS)}I", *NARROW_INPUTS)
    )
    (tmp_path / "expected.bin").write_bytes(
        struct.pack(f"<{len(expected)}I", *expected)
    )
    records = {}
    for label, code in (("generated", generated), ("original", source)):
        if label == "original" and target != "metal":
            continue
        records[label] = _execute(
            tmp_path / label,
            target,
            code,
            NARROW_INPUTS,
            expected,
            metal_entry="narrow_precise",
            check_outputs=lambda actual, reference: _check_narrowing(
                actual, reference, operand, original=label == "original"
            ),
            metal_compile_flags=("-fno-fast-math",),
            directx_compile_flags=("-enable-16bit-types",),
        )
    (tmp_path / "evidence.json").write_text(
        json.dumps(
            {
                "target": target,
                "operand": operand,
                "inputCount": len(NARROW_INPUTS),
                "operations": PROMOTED_OPERATIONS,
                "oracle": (
                    "Existing four-ULP binary32 bound; exact narrowing of each observed binary32 result"
                ),
                "originalMetalControl": (
                    "Separate original atan zero/subnormal policy; no claim of bitwise transcendental parity"
                ),
                "records": records,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

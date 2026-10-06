"""Binary16 value semantics with float32 OpenGL storage."""

import hashlib
import json
import math
import os
import random
import shutil
import struct
import subprocess
import sys
from pathlib import Path

import pytest

from crosstl import translate
from crosstl.project import ProjectConfig, translate_project
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
from crosstl.translator import parse
from crosstl.translator.ast import FunctionCallNode
from crosstl.translator.codegen.GLSL_codegen import (
    GLSLCodeGen,
    OpenGLScalarConversionError,
)
from tests.runtime_helpers import compile_metal, run
from tests.test_translator.test_directx_float_atomics import _compile as compile_directx

REQUIRE_ENV = "CROSTL_REQUIRE_HALF_CONVERSION_RUNTIME"
TARGET_ENV = "CROSTL_HALF_CONVERSION_TARGET"
ROOT = Path(__file__).resolve().parents[2]
GUARD = struct.pack("<I", 0x6A15BEEF) * 32
SOURCE = """
#include <metal_stdlib>
using namespace metal;
kernel void narrow_half(const device float* input [[buffer(0)]],
                        device float* output [[buffer(1)]],
                        uint i [[thread_position_in_grid]]) {
    BODY
}
"""


def _compile(generated, directory):
    artifact, module = directory / "kernel.comp", directory / "kernel.spv"
    artifact.write_text(generated, encoding="utf-8")
    tools = [shutil.which(name) for name in ("glslangValidator", "spirv-val")]
    if not all(tools):
        assert os.environ.get(REQUIRE_ENV) != "1", "Native validation tools required"
        return artifact, module
    for command in (
        [
            tools[0],
            "-G",
            "--target-env",
            "opengl",
            "-S",
            "comp",
            str(artifact),
            "-o",
            str(module),
        ],
        [tools[1], "--target-env", "opengl4.5", str(module)],
    ):
        result = subprocess.run(command, capture_output=True, text=True, timeout=60)
        assert result.returncode == 0, result.stdout + result.stderr
        assert "WARNING" not in result.stdout + result.stderr
    return artifact, module


def _shader(body, helpers=""):
    return parse(f"""shader HalfConversion {{
        {helpers}
        RWStructuredBuffer<float> output @ binding(0);
        compute {{
            layout(local_size_x = 1) in;
            void main() {{ {body} }}
        }}
    }}""")


@pytest.mark.parametrize(
    "expression", ["half(input[i])", "static_cast<half>(input[i])", "(half)input[i]"]
)
def test_public_metal_half_cast_rounds_before_widening(tmp_path, expression):
    source = tmp_path / "kernel.metal"
    source.write_text(
        SOURCE.replace("BODY", f"output[i] = float({expression});"), encoding="utf-8"
    )
    generated = translate(str(source), backend="opengl", format_output=False)
    assert "float(crossgl_round_half1(float(input_[i])))" in generated
    _compile(generated, tmp_path)


@pytest.mark.parametrize(
    "dtype", ["half", "float16_t", "f16", "half2", "half3", "half4", "f16vec2"]
)
def test_half_constructor_and_initialization_compile(tmp_path, dtype):
    width = int(dtype[-1]) if dtype[-1] in "234" else 1
    mapped = "float" if width == 1 else f"vec{width}"
    generated = GLSLCodeGen().generate(
        _shader(f"{dtype} x = {mapped}(output[0]); {dtype} y = {dtype}(output[1]);")
    )
    assert f"crossgl_round_half{width}(" in generated
    _compile(generated, tmp_path)


def test_half_boundaries_and_arithmetic_compile(tmp_path):
    generated = GLSLCodeGen().generate(
        _shader(
            """
        half x = 1.00048828125;
        x = 1.00146484375;
        x += 0.00048828125;
        x++;
        ++x;
        half y = half(1.0);
        output[0] = float(x + y);
        output[1] = widen(1.00048828125);
        output[2] = float(narrow(1.00146484375));
    """,
            "half narrow(float x) { return x; } float widen(half x) { return float(x); }",
        )
    )
    assert "return crossgl_round_half1(x);" in generated
    assert "widen(1.0)" in generated
    assert "float(crossgl_round_half1(" in generated
    _compile(generated, tmp_path)


@pytest.mark.parametrize("operator", ("+", "-", "*", "/", "?:", "=="))
@pytest.mark.parametrize(
    "left,right,expected",
    (
        ("half", "int", "half"),
        ("uint", "float16", "half"),
        ("half", "bool", "half"),
        ("int64_t", "half", "half"),
        ("half2", "int", "half2"),
        ("uint", "half3", "half3"),
        ("half", "half4", "half4"),
        ("f16vec2", "half2", "half2"),
        ("half", "float16_t", "half"),
        ("half", "float", "float"),
        ("float", "half4", "vec4"),
        ("double", "half", "double"),
        ("half", "bfloat", "float"),
    ),
)
def test_half_common_arithmetic_preserves_source_precision(
    operator, left, right, expected
):
    generator = GLSLCodeGen()
    assert generator.glsl_common_arithmetic_type(left, right, operator) == expected


@pytest.mark.parametrize("dtype", ("half", "half2", "half3", "half4"))
def test_nested_half_integer_arithmetic_compiles(tmp_path, dtype):
    width = int(dtype[-1]) if dtype[-1] in "234" else 1
    component = ".x" if width > 1 else ""
    source = tmp_path / "kernel.metal"
    source.write_text(
        SOURCE.replace(
            "BODY",
            f"""{dtype} x = {dtype}(input[i]);
            int offset = 2049;
            auto nested = (offset - x) / (1 + x);
            output[i] = float(nested{component});""",
        ),
        encoding="utf-8",
    )
    generated = translate(str(source), backend="opengl", format_output=False)
    assert "crossgl_round_half1(float(offset))" in generated
    assert f"crossgl_round_half{width}((1.0 + x))" in generated
    _compile(generated, tmp_path)


def test_half_helper_names_and_generator_reuse(tmp_path):
    generator = GLSLCodeGen()
    first = generator.generate(
        _shader("float crossgl_round_half1 = output[0]; half x = crossgl_round_half1;")
    )
    assert "float crossgl_round_half1_2(float value)" in first
    _compile(first, tmp_path)
    assert "crossgl_round_half" not in generator.generate(_shader("output[0] = 1.0;"))


@pytest.mark.parametrize(
    "body,reason",
    [
        ("half x = double(1.0);", "half-double-rounding"),
        ("output[0] = float(half(double(1.0)));", "half-double-rounding"),
        ("half3 x = half3(dvec2(1.0));", "half-double-rounding"),
        ("half x = 1.0; output[0] = float(x++);", "narrow-postfix-result"),
        ("half x[2]; int i = 0; x[i++] += 1.0;", "narrow-update-lvalue-side-effects"),
    ],
)
def test_half_unproven_conversions_fail_closed(body, reason):
    with pytest.raises(OpenGLScalarConversionError) as error:
        GLSLCodeGen().generate(_shader(body))
    assert error.value.reason == reason


@pytest.mark.parametrize("version", ["#version 330 core", "#version 300 es"])
def test_half_conversion_rejects_unproven_profiles(version):
    generator = GLSLCodeGen()
    generator.current_glsl_version_line = version
    ast = _shader("output[0] = half(1.0);")
    call = next(node for node in ast.walk() if isinstance(node, FunctionCallNode))
    call.source_location = {"line": 4, "column": 21}
    with pytest.raises(OpenGLScalarConversionError) as error:
        generator.generate_expression(call)
    assert error.value.reason == "half-unsupported-profile"
    assert error.value.source_location == call.source_location


@pytest.mark.parametrize("builtin", ["floatBitsToUint", "uintBitsToFloat", "findMSB"])
def test_half_conversion_rejects_shadowed_builtins(builtin):
    with pytest.raises(OpenGLScalarConversionError) as error:
        GLSLCodeGen().generate(
            _shader("half x = output[0];", f"float {builtin}(float x) {{ return x; }}")
        )
    assert error.value.reason == "half-target-builtin-shadowed"


def test_half_literal_globals_remain_constant_expressions(tmp_path):
    generated = GLSLCodeGen().generate(
        _shader(
            "output[0] = float(narrowed) + float(pair.x);",
            "const half narrowed = half(0.1); const half2 pair = half2(1.00048828125, -0.0);",
        )
    )
    assert "const float narrowed = 0.0999755859375;" in generated
    assert "const vec2 pair = vec2(1.0, -0.0);" in generated
    assert "crossgl_round_half" not in generated
    _compile(generated, tmp_path)


def test_half_unfolded_global_has_a_diagnostic():
    with pytest.raises(OpenGLScalarConversionError) as error:
        GLSLCodeGen().generate(_shader("", "const half value = half(1.0 + 0.1);"))
    assert error.value.reason == "half-global-initializer"


def _float(word):
    return struct.unpack("<f", struct.pack("<I", word))[0]


def _word(value):
    return struct.unpack("<I", struct.pack("<f", value))[0]


def _half(value):
    try:
        return struct.unpack("<e", struct.pack("<e", value))[0]
    except OverflowError:
        return math.copysign(math.inf, value)


def _rounding_words():
    words = {
        (
            ((bits & 0x8000) << 16) | 0x7FC00000
            if bits & 0x7C00 == 0x7C00 and bits & 0x3FF
            else _word(struct.unpack("<e", struct.pack("<H", bits))[0])
        )
        for bits in range(65536)
    }
    # Every finite positive half interval, both midpoint neighbors, and both signs.
    for bits in range(0x7BFF):
        left = struct.unpack("<e", struct.pack("<H", bits))[0]
        right = struct.unpack("<e", struct.pack("<H", bits + 1))[0]
        midpoint = _word((left + right) / 2)
        for neighbor in (midpoint - 1, midpoint, midpoint + 1):
            words.update((neighbor, neighbor | 0x80000000))
    for value in (65520.0, 2**-25):
        midpoint = _word(value)
        for neighbor in (midpoint - 1, midpoint, midpoint + 1):
            words.update((neighbor, neighbor | 0x80000000))
    random_source = random.Random(1992)
    words.update(random_source.getrandbits(32) for _ in range(4096))
    return sorted(words)


def test_half_rounding_dataset_is_platform_independent():
    words = _rounding_words()
    assert len(words) == 258052
    payload = struct.pack(f"<{len(words)}I", *words)
    assert hashlib.sha256(payload).hexdigest() == (
        "3434e4637b75ba6f60db3dcc1bd7732b9627f396b57dc4840b99856c2b57ec71"
    )


def _check(data, expected):
    assert len(data) == len(expected) * 4 + len(GUARD), "size"
    assert data[len(expected) * 4 :] == GUARD, "guard"
    for index, ((word,), reference) in enumerate(
        zip(struct.iter_unpack("<I", data[: -len(GUARD)]), expected)
    ):
        if math.isnan(reference):
            assert math.isnan(_float(word)), f"NaN at {index}"
        else:
            assert word == _word(
                reference
            ), f"value at {index}: {_float(word)!r} != {reference!r}"


@pytest.mark.parametrize("corruption", ["value", "guard", "size", "zero-sign"])
def test_half_verifier_rejects_corruption(corruption):
    data = struct.pack("<2f", 1.0, -0.0) + GUARD
    if corruption == "value":
        data = struct.pack("<f", 1.00048828125) + data[4:]
    elif corruption == "guard":
        data = data[:-1] + b"\x00"
    elif corruption == "size":
        data = data[:-4]
    else:
        data = data[:4] + struct.pack("<f", 0.0) + data[8:]
    with pytest.raises(AssertionError):
        _check(data, [1.0, -0.0])


@pytest.mark.parametrize(
    "mode", ["rounding", "boundaries", "vectors", "nested-arithmetic"]
)
def test_half_rounding_native(tmp_path, mode, *, compile_only=False):
    if not compile_only and os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 for native half conversion checks")
    target = os.environ.get(TARGET_ENV, "opengl")
    assert target in {"opengl", "metal", "directx"}
    if compile_only:
        assert target == "directx"
    else:
        assert (
            sys.platform
            == {"metal": "darwin", "opengl": "linux", "directx": "win32"}[target]
        )
    words = _rounding_words()
    expected = [_half(_float(word)) for word in words]
    body = f"if (i < {len(words)}u) output[i] = float(half(input[i]));"
    text = SOURCE
    if mode == "boundaries":
        values = [
            1.00048828125,
            1.00146484375,
            -1.00048828125,
            -1.00146484375,
            0.0000000298023223876953125,
            65519.0,
        ]
        words = [_word(value) for value in values]
        expected = []
        for value in values:
            narrowed = _half(value)
            expected.extend(
                [narrowed] * 4
                + [_half(narrowed + 0.00048828125)] * 2
                + [narrowed, _half(-value), 1.0]
            )
        text = SOURCE.replace(
            "kernel void",
            """
half narrow_value(float value) { return value; }
float widen_value(half value) { return float(value); }
float record_value(thread float& calls, float value) { calls += 1.0f; return value; }
struct Record { half value; };
kernel void""",
        )
        body = f"""
        if (i >= {len(words)}u) return;
        half value = input[i];
        output[9 * i] = float(value);
        output[9 * i + 1] = float(narrow_value(input[i]));
        output[9 * i + 2] = widen_value(input[i]);
        Record item;
        item.value = input[i];
        output[9 * i + 3] = float(item.value);
        value += 0.00048828125f;
        output[9 * i + 4] = float(value);
        output[9 * i + 5] = float(half(input[i]) + half(0.00048828125f));
        float calls = 0.0f;
        half2 pair = half2(record_value(calls, input[i]), -input[i]);
        output[9 * i + 6] = float(pair.x);
        output[9 * i + 7] = float(pair.y);
        output[9 * i + 8] = calls;
        """
    if mode == "vectors":
        values = [
            2**-24,
            2**-25,
            3 * 2**-25,
            2**-14,
            1.00048828125,
            1.00146484375,
            -0.0,
            65519.0,
            65520.0,
            math.inf,
            math.nan,
        ]
        words = [_word(value) for value in values]
        expected = [
            lane for value in values for lane in [_half(value), _half(-value)] * 2
        ]
        body = f"""
        if (i >= {len(words)}u) return;
        half4 narrowed = half4(input[i], -input[i], input[i], -input[i]);
        float4 widened = float4(float2(narrowed.xy), float(narrowed.z), float(narrowed.w));
        output[4 * i] = widened.x;
        output[4 * i + 1] = widened.y;
        output[4 * i + 2] = widened.z;
        output[4 * i + 3] = widened.w;
        """
    if mode == "nested-arithmetic":
        words = [
            _word(struct.unpack("<e", struct.pack("<H", bits))[0])
            for bits in range(65536)
        ]

        def reciprocal(value):
            denominator = _half(1.0 + value)
            if denominator == 0.0:
                return math.copysign(math.inf, denominator)
            return _half(1.0 / denominator)

        expected = []
        for index, word in enumerate(words):
            value = _float(word)
            nested = reciprocal(value)
            expected.extend(
                (
                    nested,
                    nested,
                    nested,
                    reciprocal(-value),
                    nested,
                    1.0,
                    _half(value + 2048.0),
                    _half(2048.0 - value),
                    float(value == 2048.0),
                    value if index & 1 else 2048.0,
                    _half(value + 2048.0),
                    _half(-value + 2048.0),
                    value if index & 1 else 2048.0,
                    float(index & 1),
                )
            )
        text = SOURCE.replace(
            "kernel void",
            """typedef half Value;
Value record_half(thread uint& calls, Value value) { calls += 1u; return value; }
kernel void""",
        )
        body = f"""
        if (i >= {len(words)}u) return;
        Value value = Value(input[i]);
        auto nested = 1 / (1 + value);
        Value denominator = 1 + value;
        auto pair = 1 / (1 + half2(value, -value));
        uint calls = 0u;
        auto observed = 1 / (1 + record_half(calls, value));
        int offset = 2049;
        auto shifted = half2(value, -value) + offset;
        uint selected_calls = 0u;
        auto selected = (i & 1u) != 0u ? record_half(selected_calls, value) : offset;
        output[14u * i] = float(nested);
        output[14u * i + 1u] = float(1 / denominator);
        output[14u * i + 2u] = float(pair.x);
        output[14u * i + 3u] = float(pair.y);
        output[14u * i + 4u] = float(observed);
        output[14u * i + 5u] = float(calls);
        output[14u * i + 6u] = float(value + offset);
        output[14u * i + 7u] = float(offset - value);
        output[14u * i + 8u] = float(value == offset);
        output[14u * i + 9u] = float((i & 1u) != 0u ? value : offset);
        output[14u * i + 10u] = float(shifted.x);
        output[14u * i + 11u] = float(shifted.y);
        output[14u * i + 12u] = float(selected);
        output[14u * i + 13u] = float(selected_calls);
        """
    source = tmp_path / "kernel.metal"
    source.write_text(text.replace("BODY", body), encoding="utf-8")
    report = translate_project(
        ProjectConfig(
            root=tmp_path,
            targets=(target,),
            include_patterns=("kernel.metal",),
            output_dir=Path("translated"),
            workgroup_size=(64, 1, 1),
        ),
        format_output=False,
    )
    report.write_json(tmp_path / "report.json")
    assert not report.to_json()["diagnostics"]
    inputs = struct.pack(f"<{len(words)}I", *words) + GUARD
    initial = struct.pack("<I", 0xDEADBEEF) * len(expected) + GUARD
    for slot, data in enumerate((inputs, initial)):
        (tmp_path / f"input-{slot}.bin").write_bytes(data)
    (tmp_path / "expected.bin").write_bytes(
        struct.pack(f"<{len(expected)}f", *expected)
    )
    groups = [(len(words) + 63) // 64, 1, 1]
    extension = {"metal": "metal", "opengl": "glsl", "directx": "hlsl"}[target]
    generated = next((tmp_path / "translated").rglob(f"*.{extension}"))
    records = []
    if target == "metal":
        executable = tmp_path / "readback"
        run(
            [
                "swiftc",
                ROOT / "tests/fixtures/runtime_verification/metal_raw_buffers.swift",
                "-o",
                executable,
            ],
            tmp_path,
            "runner",
        )
        request = tmp_path / "request.json"
        request.write_text(
            json.dumps(
                {
                    "buffers": [
                        str(tmp_path / f"input-{slot}.bin") for slot in range(2)
                    ],
                    "workgroupCount": groups,
                    "workgroupSize": [64, 1, 1],
                    "simdWidth": 32,
                }
            ),
            encoding="utf-8",
        )
        for label, artifact in (("original", source), ("generated", generated)):
            directory = tmp_path / label
            directory.mkdir()
            flags = (
                ("-std=metal3.1", "-fno-fast-math")
                if mode == "nested-arithmetic"
                else ()
            )
            module = compile_metal(artifact, tmp_path, directory, flags=flags)
            run(
                [executable, module, "narrow_half", request, directory],
                directory,
                "execute",
            )
            assert (directory / "buffer-0.bin").read_bytes() == inputs
            data = (directory / "buffer-1.bin").read_bytes()
            _check(data, expected)
            records.append(
                {
                    "path": label,
                    "values": len(expected),
                    "sha256": hashlib.sha256(data).hexdigest(),
                }
            )
    else:
        if target == "directx":
            assert shutil.which("dxc"), "DXC is required"
            artifact, module = compile_directx(
                generated.read_text(encoding="utf-8"),
                tmp_path,
                profile="cs_6_2",
                flags=("-enable-16bit-types",),
            )
            if compile_only:
                (tmp_path / "evidence.json").write_text(
                    json.dumps({"execution": "not-tested", "compilation": "passed"}),
                    encoding="utf-8",
                )
                return
        else:
            artifact, module = _compile(generated.read_text(encoding="utf-8"), tmp_path)
        buffers = {}
        for slot, (name, data) in enumerate((("input", inputs), ("output", initial))):
            payload = [word for (word,) in struct.iter_unpack("<I", data)]
            buffers[name] = NativeRuntimeBufferBinding(
                name=name,
                binding=RuntimeResourceBinding(
                    name=name,
                    kind="buffer",
                    set=0,
                    binding=slot,
                    type_name=(
                        f"{'RW' if slot else ''}StructuredBuffer<float>"
                        if target == "directx"
                        else "float"
                    ),
                    access="read" if slot == 0 else "read_write",
                    metadata={"byteStride": 4},
                ),
                dtype="uint32",
                shape=(len(payload),),
                value=payload,
                source="input" if slot == 0 else "expectedOutput",
            )
        entry = "CSMain" if target == "directx" else "main"
        request = NativeRuntimeDispatchRequest(
            target=target,
            artifact={"target": target},
            artifact_path=artifact,
            module_path=module,
            loaded_artifact=artifact.read_text(encoding="utf-8"),
            buffers=buffers,
            constants={},
            entry_point=entry,
            dispatch=RuntimeDispatchGeometry(
                entry_point=entry,
                workgroup_size=(64, 1, 1),
                workgroup_count=tuple(groups),
            ),
        )
        runtime = (
            DirectXComputeRuntime()
            if target == "directx"
            else OpenGLComputeRuntime(context_backends=("egl",))
        )
        outputs = runtime.dispatch(None, None, request)
        assert set(outputs) == {"output"}
        data = struct.pack(
            f"<{len(outputs['output']['values'])}I", *outputs["output"]["values"]
        )
        (tmp_path / "buffer-1.bin").write_bytes(data)
        _check(data, expected)
        records.append(
            {
                "path": target,
                "values": len(expected),
                "sha256": hashlib.sha256(data).hexdigest(),
            }
        )
    (tmp_path / "evidence.json").write_text(
        json.dumps(
            {
                "execution": "passed",
                "sourceSha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "generatedSha256": hashlib.sha256(generated.read_bytes()).hexdigest(),
                "records": records,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

"""Runtime integer narrowing rounds directly to bfloat without a float32 step."""

import os
import random
import shutil
import struct
from pathlib import Path

import pytest

from crosstl import translate
from crosstl.translator.codegen.directx_codegen import HLSLCodeGen
from tests.test_translator.test_boolean_buffer_runtime import _bound_values, _request
from tests.test_translator.test_half_buffer_runtime import _validate_half
from tests.test_translator.test_loop_updates import _execute
from tests.test_translator.test_opengl_bfloat_conversion import REQUIRE_ENV, TARGET
from tests.test_translator.test_opengl_half_conversion import _shader
from tests.test_translator.test_software_subgroup_product import _package


def _round_integer(value, precision):
    magnitude = abs(value)
    shift = max(0, magnitude.bit_length() - precision)
    retained, remainder = divmod(magnitude, 1 << shift)
    if shift:
        midpoint = 1 << (shift - 1)
        retained += remainder > midpoint or (remainder == midpoint and retained % 2)
    rounded = retained << shift
    return -rounded if value < 0 else rounded


def _payload(value):
    rounded = _round_integer(value, 8)
    return struct.unpack("<I", struct.pack("<f", rounded))[0] >> 16


def _values(signed):
    values = [0, 1, 127, 128, 255, 256]
    for exponent in range(8, 63 if signed else 64):
        spacing = 1 << (exponent - 7)
        values.extend(
            significand * spacing + spacing // 2 + delta
            for significand in range(128, 256)
            for delta in (-1, 0, 1)
        )
    if signed:
        values.extend(-value for value in values[1:].copy())
        values.extend((-(1 << 63), (1 << 63) - 1))
    else:
        values.append((1 << 64) - 1)
    rng = random.Random(6408)
    values.extend(
        rng.randrange(-(1 << 63), 1 << 63) if signed else rng.randrange(1 << 64)
        for _ in range(256)
    )
    return values


def _source(signed):
    dtype = "long" if signed else "ulong"
    return f"""#include <metal_stdlib>
using namespace metal;
bfloat narrow({dtype} value) {{ return value; }}
kernel void wide_bfloat(const device {dtype}* values [[buffer(0)]],
                        device uint* results [[buffer(1)]],
                        uint i [[thread_position_in_grid]]) {{
    uint cursor = i;
    bfloat direct = bfloat(values[cursor++]);
    bfloat cast_value = static_cast<bfloat>(values[i]);
    bfloat returned = narrow(values[i]);
    bfloat mediated = bfloat(float(values[i]));
    results[5u * i + 1u] = uint(as_type<ushort>(direct));
    results[5u * i + 2u] = uint(as_type<ushort>(cast_value));
    results[5u * i + 3u] = uint(as_type<ushort>(returned));
    results[5u * i + 4u] = uint(as_type<ushort>(mediated));
    results[5u * i + 5u] = cursor - i;
}}
"""


@pytest.mark.parametrize("signed", (False, True))
def test_wide_integer_bfloat_reference_covers_midpoints_and_limits(signed):
    values = _values(signed)
    assert len(values) < 65536
    assert max(values) == (1 << (63 if signed else 64)) - 1
    assert min(values) == (-(1 << 63) if signed else 0)
    assert _payload(16842753) == 0x4B81
    assert _payload(-16842753) == 0xCB81
    assert _payload(9259400833873739777) == 0x5F01
    assert _payload(_round_integer(9259400833873739777, 24)) == 0x5F00
    assert _payload(-(1 << 63)) == 0xDF00
    assert _payload((1 << 64) - 1) == 0x5F80
    assert any(
        _payload(value) != _payload(_round_integer(value, 24)) for value in values
    )


@pytest.mark.parametrize("dtype", ("int64_t", "uint64_t"))
@pytest.mark.parametrize(
    "body",
    (
        "bfloat value = bfloat(seed);",
        "bfloat value = seed;",
        "bfloat value = bfloat(0); value = seed;",
        "bfloat value = bfloat16(seed);",
    ),
)
def test_hlsl_wide_integer_bfloat_uses_integer_rounding(dtype, body):
    generator = HLSLCodeGen()
    generated = generator.generate(
        _shader(f"{dtype} seed = {dtype}(output[0]); {body} output[0] = float(value);")
    )
    helper = "from_int64" if dtype == "int64_t" else "from_uint64"
    assert f"__crossgl_bfloat16_{helper}({dtype}(seed))" in generated
    assert "__crossgl_bfloat16_from_float" not in generated
    assert generated.count("uint __crossgl_bfloat16_from_uint64(uint64_t value) {") == 1
    assert "firstbithigh(high)" in generated
    assert "uint64_t remainder" in generated
    assert "__crossgl_bfloat16_from_uint64" not in generator.generate(
        _shader("output[0] = 1.0;")
    )


@pytest.mark.parametrize("signed", (False, True))
def test_metal_wide_integer_bfloat_translates_and_compiles(tmp_path, signed):
    path = tmp_path / "wide.metal"
    path.write_text(_source(signed), encoding="utf-8")
    generated = translate(str(path), backend="directx", format_output=False)
    helper = "from_int64" if signed else "from_uint64"
    assert f"__crossgl_bfloat16_{helper}" in generated
    assert "__crossgl_bfloat16_from_float" in generated
    assert generated.count("cursor++") == 1
    artifact = tmp_path / "generated.hlsl"
    artifact.write_text(generated, encoding="utf-8")
    if shutil.which("dxc") is None:
        if TARGET == "directx" and os.environ.get(REQUIRE_ENV) == "1":
            pytest.fail("DXC is required for native DirectX bfloat conversion")
        pytest.skip("DXC is unavailable")
    _validate_half(artifact, tmp_path, "directx")


@pytest.mark.parametrize("signed", (False, True))
def test_wide_integer_bfloat_executes_natively(tmp_path, signed):
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 for required bfloat conversions")
    if TARGET == "opengl":
        pytest.skip("OpenGL wide-integer bfloat conversion remains a diagnostic")
    values = _values(signed)
    source = _source(signed)
    _, descriptor, package = _package(
        tmp_path, TARGET, "uint", (1, 1, 1), source=source, software_subgroups=False
    )
    guard = 0xC0FFEE
    expected = [guard]
    for value in values:
        direct = _payload(value)
        expected.extend(
            (direct, direct, direct, _payload(_round_integer(value, 24)), 1)
        )
    expected.append(guard)
    inputs = {
        "values": {
            "dtype": "int64" if signed else "uint64",
            "shape": [len(values)],
            "values": values,
        },
        "results": {
            "dtype": "uint32",
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
        original_source=source,
        original_entry="wide_bfloat",
        validate=_validate_half,
    )


def test_wide_integer_bfloat_native_gates_are_required():
    from tools import ci_coverage

    workflow = Path(".github/workflows/demo-project-testing.yml").read_text()
    for name in (
        "Validate indexed DirectX gather and resource aggregates",
        "Validate Metal byte and vector storage",
    ):
        step = ci_coverage.workflow_step_section(workflow, name)
        assert f'{REQUIRE_ENV}: "1"' in step
        assert "tests/test_translator/test_wide_integer_bfloat.py" in step

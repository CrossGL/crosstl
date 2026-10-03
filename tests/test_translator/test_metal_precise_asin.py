"""Portable precise arcsine, including native scalar/vector numerical contracts."""

import hashlib
import json
import math
import os
import struct
import sys
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import pytest

from crosstl import translate
from crosstl.backend.Metal.MetalCrossGLCodeGen import MetalPreciseMathLoweringError
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
from crosstl.translator import parse
from crosstl.translator.source_licenses import source_license_comments
from tests.test_translator.test_metal_builtin_ownership import _compile

REQUIRE_ENV = "CROSTL_REQUIRE_METAL_PRECISE_ASIN"
SOURCE = """#include <metal_stdlib>
using namespace metal;
float record(thread float& count, float value) { count += 1.0f; return value; }
kernel void arcsine(device const float* values [[buffer(0)]],
                    device float* results [[buffer(1)]],
                    uint i [[thread_position_in_grid]]) {
    float x = values[i];
    float2 pair = metal::precise::asin(float2(x, -x));
    float3 triple = metal::precise::asin(float3(x, -x, x));
    float count = 0.0f;
    float4 quad = metal::precise::asin(float4(record(count, x), -x, x, -x));
    results[11 * i] = metal::precise::asin(x);
    results[11 * i + 1] = pair.x;
    results[11 * i + 2] = pair.y;
    results[11 * i + 3] = triple.x;
    results[11 * i + 4] = triple.y;
    results[11 * i + 5] = triple.z;
    results[11 * i + 6] = quad.x;
    results[11 * i + 7] = quad.y;
    results[11 * i + 8] = quad.z;
    results[11 * i + 9] = quad.w;
    results[11 * i + 10] = count;
}
"""


def _translate(tmp_path, source, target):
    path = tmp_path / "arcsine.metal"
    path.write_text(source, encoding="utf-8")
    return translate(str(path), backend=target, format_output=False)


@pytest.mark.parametrize("target", ["directx", "opengl", "metal"])
def test_precise_asin_helpers_compile(tmp_path, target):
    generated = _translate(tmp_path, SOURCE, target)
    for suffix in ("", "2", "3", "4"):
        assert f"metal_precise_asin_float{suffix}(" in generated
    assert "asin(x)" not in generated
    assert "Copyright (C) 1993 by Sun" in generated
    _compile(generated, target, tmp_path)


def test_precise_asin_keeps_source_overloads_and_other_modes(tmp_path):
    source = """#include <metal_stdlib>
    float asin(float value) { return value + 3.0f; }
    float explicit_precise(float x) { return metal::precise::asin(x); }
    float imported_precise(float x) { return precise::asin(x); }
    float default_mode(float x) { return metal::asin(x); }
    float fast_mode(float x) { return metal::fast::asin(x); }
    float source_function(float x) { return ::asin(x); }
    """
    generated = _translate(tmp_path, source, "crossgl")
    assert generated.count("return __crossgl_metal_precise_asin_float(x);") == 2
    assert generated.count("return asin(x);") == 2
    assert "return asin__metal_overload_1(x);" in generated


def test_precise_asin_helper_names_do_not_collide(tmp_path):
    source = """
    float __crossgl_metal_precise_asin_float(float x) { return x; }
    float2 __crossgl_metal_precise_asin_float2(float2 x) { return x; }
    float2 evaluate(float2 x) { return metal::precise::asin(x); }
    """
    generated = _translate(tmp_path, source, "crossgl")
    assert "float __crossgl_metal_precise_asin_float_(float value)" in generated
    assert "return __crossgl_metal_precise_asin_float2_(x);" in generated


@pytest.mark.parametrize("operand", ["int", "Payload"])
def test_precise_asin_rejects_unrepresentable_operands(tmp_path, operand):
    source = "struct Payload { float value; };\n" + (
        f"{operand} evaluate({operand} x) {{ return metal::precise::asin(x); }}"
    )
    with pytest.raises(MetalPreciseMathLoweringError) as error:
        _translate(tmp_path, source, "crossgl")
    assert error.value.operation == "asin"
    assert error.value.project_diagnostic_code == (
        "project.translate.metal-precise-math-unsupported"
    )


@pytest.mark.parametrize(
    "arguments", ["", "unknown", "fdlibm, fdlibm", "123", '"fdlibm"']
)
def test_source_license_rejects_unknown_or_ambiguous_metadata(arguments):
    with pytest.raises(SyntaxError, match="source_license"):
        parse(
            f"shader S {{ @source_license({arguments}) float f() {{ return 0.0; }} }}"
        )


@pytest.mark.parametrize(
    "target,prefix",
    [
        ("directx", "//"),
        ("opengl", "//"),
        ("metal", "//"),
        ("mojo", "#"),
        ("vulkan", ";"),
    ],
)
def test_source_license_metadata_retains_names_and_deduplicates(target, prefix):
    ast = parse("""shader S {
        @source_license(fdlibm) float first() { return 1.0; }
        @source_license(fdlibm) float second() { return 2.0; }
    }""")
    assert [function.name for function in ast.functions] == ["first", "second"]
    notice = source_license_comments(ast, target)
    assert notice.count("Copyright") == 1
    assert all(line.startswith(prefix) for line in notice.splitlines())
    assert "provided that this notice is preserved" in notice


def test_ci_requires_native_precise_asin_on_all_three_platforms():
    from pathlib import Path

    workflow = (
        Path(__file__).resolve().parents[2]
        / ".github/workflows/mlx-project-porting.yml"
    ).read_text()
    step = workflow.split("      - name: Validate Metal builtin ownership\n", 1)[
        1
    ].split("      - name:", 1)[0]
    assert "if:" not in step
    assert f'{REQUIRE_ENV}: "1"' in step
    assert "tests/test_translator/test_metal_precise_asin.py" in step
    assert "--timeout-seconds 120" in step
    assert "pytest -q -n auto" in step


def _float32(value):
    return struct.unpack("<f", struct.pack("<f", value))[0]


def _bits(value):
    return struct.unpack("<I", struct.pack("<f", value))[0]


def _write_values(path, values):
    portable = [
        (
            "nan"
            if math.isnan(value)
            else (
                "+infinity"
                if value == math.inf
                else "-infinity" if value == -math.inf else value
            )
        )
        for value in values
    ]
    path.write_text(json.dumps(portable, allow_nan=False), encoding="utf-8")


def _inputs():
    values = [index / 2048 for index in range(-2048, 2049)]
    for bits in (0x39800000, 0x3F000000, 0x3F800000):
        for offset in range(-4, 5):
            value = struct.unpack("<f", struct.pack("<I", bits + offset))[0]
            values.extend((value, -value))
    for exponent in range(-120, -1):
        values.extend((2.0**exponent, -(2.0**exponent)))
    values.extend((-0.0, 0.0, -math.inf, math.inf, math.nan))
    return values


def _check_outputs(inputs, outputs):
    assert len(outputs) == 11 * len(inputs)
    max_ulp_error = 0
    for index, value in enumerate(inputs):
        arguments = (
            value,
            value,
            -value,
            value,
            -value,
            value,
            value,
            -value,
            value,
            -value,
        )
        for lane, argument in enumerate(arguments):
            actual = outputs[11 * index + lane]
            if math.isnan(argument) or abs(argument) > 1:
                assert math.isnan(actual), (argument, actual)
                continue
            expected = _float32(math.asin(argument))
            assert math.isfinite(actual), (argument, actual)
            if expected == 0:
                assert _bits(actual) == _bits(expected), (argument, actual)
            else:
                assert math.copysign(1, actual) == math.copysign(1, expected)
                error = abs(_bits(actual) - _bits(expected))
                assert error <= 4, (argument, actual, expected, error)
                max_ulp_error = max(max_ulp_error, error)
        assert outputs[11 * index + 10] == 1, "An operand was evaluated repeatedly"
    return max_ulp_error


def test_precise_asin_executes(tmp_path):
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 to require native precise arcsine")
    target = {"win32": "directx", "linux": "opengl", "darwin": "metal"}[sys.platform]
    generated = _translate(tmp_path, SOURCE, target)
    artifact, module = _compile(generated, target, tmp_path)
    assert module.is_file(), "A native compiler must produce the module"
    inputs = _inputs()
    _write_values(tmp_path / "inputs.json", inputs)
    layouts = {}
    if target == "metal":
        layouts = {
            resource["name"]: resource["scalarLayout"]
            for resource in reflect_target_host_interface(artifact, target=target)[
                "resources"
            ]
        }
    buffers = {}
    for name, slot, value, count in (
        ("values", 0, inputs, len(inputs)),
        ("results", 1, None, 11 * len(inputs)),
    ):
        output = value is None
        buffers[name] = NativeRuntimeBufferBinding(
            name=name,
            binding=RuntimeResourceBinding(
                name=name,
                kind="buffer",
                set=0,
                binding=slot,
                type_name=("RW" if output else "") + "StructuredBuffer<float>",
                access="read_write" if output else "read",
                metadata={"scalarLayout": layouts[name]} if layouts else {},
            ),
            source="expectedOutput" if output else "input",
            dtype="float32",
            shape=(count,),
            value=value,
        )
    entry = {"directx": "CSMain", "opengl": "main", "metal": "arcsine"}[target]
    request = NativeRuntimeDispatchRequest(
        target=target,
        artifact={"target": target},
        artifact_path=artifact,
        module_path=module,
        loaded_artifact=generated if target == "opengl" else module.read_bytes(),
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
        "opengl": lambda: OpenGLComputeRuntime(context_backends=("egl",)),
        "metal": MetalComputeRuntime,
    }[target]()
    state = SimpleNamespace(details={})
    outputs = runtime.dispatch(None, state, request)["results"]["values"]
    _write_values(tmp_path / "readback.json", outputs)
    error = _check_outputs(inputs, outputs)
    evidence = {
        "target": target,
        "inputCount": len(inputs),
        "outputCount": len(outputs),
        "maxUlpError": error,
        "artifactSHA256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
        "moduleSHA256": hashlib.sha256(module.read_bytes()).hexdigest(),
        "runtime": deepcopy(state.details),
    }
    if target == "metal":
        original_dir = tmp_path / "original"
        original_dir.mkdir()
        original, library = _compile(SOURCE, target, original_dir)
        original_request = replace(
            request,
            artifact_path=original,
            module_path=library,
            loaded_artifact=library.read_bytes(),
        )
        original_outputs = runtime.dispatch(None, state, original_request)["results"][
            "values"
        ]
        _write_values(original_dir / "readback.json", original_outputs)
        evidence["originalMaxUlpError"] = _check_outputs(inputs, original_outputs)
        evidence["originalArtifactSHA256"] = hashlib.sha256(
            original.read_bytes()
        ).hexdigest()
        evidence["originalModuleSHA256"] = hashlib.sha256(
            library.read_bytes()
        ).hexdigest()
        evidence["originalRuntime"] = deepcopy(state.details)
        assert (
            evidence["runtime"]["metalRuntime"]["librarySHA256"]
            == evidence["moduleSHA256"]
        )
        assert (
            evidence["originalRuntime"]["metalRuntime"]["librarySHA256"]
            == evidence["originalModuleSHA256"]
        )
    (tmp_path / "evidence.json").write_text(
        json.dumps(evidence, indent=2), encoding="utf-8"
    )

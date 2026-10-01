"""Retain qualified Metal builtins alongside colliding source functions."""

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from crosstl import translate
from crosstl.project.native_runtime_drivers import (
    DirectXComputeRuntime,
    OpenGLComputeRuntime,
)
from crosstl.project.runtime_verification import (
    NativeRuntimeBufferBinding,
    NativeRuntimeDispatchRequest,
    RuntimeArtifactSelector,
    RuntimeDispatchGeometry,
    RuntimeExecutionRequest,
    RuntimeFixture,
    RuntimeResourceBinding,
)

REQUIRE_ENV = "CROSTL_REQUIRE_METAL_BUILTIN_OWNERSHIP"
ROOT = Path(__file__).resolve().parents[2]

SOURCE = """
#include <metal_stdlib>
using namespace metal;

float fmin(float left, float right) { return left + right; }
float fmin(int left, int right) { return float(left - right); }
namespace custom {
float fmin(float left, float right) { return ::fmin(left, right) * 2.0f; }
float sin(float value) { return value + 10.0f; }
}
namespace metal {
float fmin(int left, int right) { return float(left + right + 1); }
}

kernel void ownership(device float* results [[buffer(0)]]) {
    results[0] = metal::fmin(2.0f, 3.0f);
    results[1] = ::fmin(2.0f, 3.0f);
    results[2] = ::fmin(5, 2);
    results[3] = custom::fmin(2.0f, 3.0f);
    results[4] = ::metal::fmin(3.0f, 2.0f);
    results[5] = metal::fmin(2, 3);
    results[6] = metal::sin(0.0f);
    results[7] = sin(0.0f);
    results[8] = custom::sin(0.0f);
}
"""


def _translate(tmp_path, source, target):
    path = tmp_path / "ownership.metal"
    path.write_text(source, encoding="utf-8")
    return translate(str(path), backend=target, format_output=False)


def _run(command):
    result = subprocess.run(command, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "warning:" not in (result.stdout + result.stderr).lower(), (
        result.stdout + result.stderr
    )
    return result.stdout


def _compile(generated, target, tmp_path, *, metal_compile_flags=()):
    suffix, module_suffix, tool = {
        "directx": ("hlsl", "dxil", "dxc"),
        "opengl": ("comp", "spv", "glslangValidator"),
        "metal": ("metal", "metallib", "xcrun"),
    }[target]
    artifact = tmp_path / f"translated.{suffix}"
    module = tmp_path / f"translated.{module_suffix}"
    artifact.write_text(generated, encoding="utf-8")
    if not shutil.which(tool):
        return artifact, module
    if target == "directx":
        _run(
            [
                tool,
                "-T",
                "cs_6_6",
                "-E",
                "CSMain",
                "-WX",
                str(artifact),
                "-Fo",
                str(module),
            ]
        )
    elif target == "metal":
        air = tmp_path / "translated.air"
        _run(
            [
                tool,
                "--sdk",
                "macosx",
                "metal",
                "-Werror",
                *metal_compile_flags,
                "-c",
                str(artifact),
                "-o",
                str(air),
            ]
        )
        _run([tool, "--sdk", "macosx", "metallib", str(air), "-o", str(module)])
    else:
        _run([tool, "-G", "-S", "comp", str(artifact), "-o", str(module)])
        if shutil.which("spirv-val"):
            _run(["spirv-val", "--target-env", "opengl4.5", str(module)])
    return artifact, module


@pytest.mark.parametrize("target", ["directx", "opengl", "metal"])
def test_qualified_metal_builtin_and_source_overloads_compile(tmp_path, target):
    generated = _translate(tmp_path, SOURCE, target)
    assert "metal_overload_" in generated
    _compile(generated, target, tmp_path)


@pytest.mark.parametrize(
    "operation,arguments,parameters",
    [
        ("fabs", "-2.0f", "float value"),
        ("fmin", "2.0f, 3.0f", "float value, float other"),
        ("fmax", "2.0f, 3.0f", "float value, float other"),
        ("copysign", "2.0f, -3.0f", "float value, float other"),
        ("select", "2.0f, 3.0f, false", "float value, float other, bool condition"),
        ("sin", "2.0f", "float value"),
    ],
)
def test_qualified_math_keeps_source_calls_distinct(
    tmp_path, operation, arguments, parameters
):
    source = f"""#include <metal_stdlib>
    float {operation}({parameters}) {{ return value + 10.0f; }}
    kernel void ownership(device float* results [[buffer(0)]]) {{
        results[0] = metal::{operation}({arguments});
        results[1] = {operation}({arguments});
    }}"""
    intermediate = _translate(tmp_path, source, "crossgl")
    assert f"float {operation}__metal_overload_1(" in intermediate
    assert f"buffer_store(results, 0, {operation}(" in intermediate
    assert f"buffer_store(results, 1, {operation}__metal_overload_1(" in intermediate


def test_builtin_transport_avoids_source_identifiers(tmp_path):
    source = """#include <metal_stdlib>
    float fmin(float x, float y) { return x + y; }
    float fmin__metal_overload_1(float x) { return x; }
    kernel void ownership(device float* results [[buffer(0)]]) {
        float fmin__metal_overload_1_2 = 2.0f;
        results[0] = metal::fmin(fmin__metal_overload_1_2, 3.0f);
        results[1] = ::fmin(2.0f, 3.0f);
    }"""
    intermediate = _translate(tmp_path, source, "crossgl")
    assert "float fmin__metal_overload_1_3(float x, float y)" in intermediate
    assert "fmin__metal_overload_1_3(2.0f, 3.0f)" in intermediate


def test_qualified_math_does_not_change_unrelated_source_helpers(tmp_path):
    source = """#include <metal_stdlib>
    float fmin(float x, float y) { return x + y; }
    kernel void ownership(device float* results [[buffer(0)]]) {
        results[0] = ::fmin(2.0f, 3.0f);
    }"""
    intermediate = _translate(tmp_path, source, "crossgl")
    assert "float fmin(float x, float y)" in intermediate
    assert "metal_overload" not in intermediate


def test_imported_builtin_does_not_bind_unrelated_namespace_helper(tmp_path):
    intermediate = _translate(tmp_path, SOURCE, "crossgl")
    assert "buffer_store(results, 6, sin(0.0f))" in intermediate
    assert "buffer_store(results, 7, sin(0.0f))" in intermediate
    assert "buffer_store(results, 8, sin__metal_overload_1(0.0f))" in intermediate


def test_qualified_math_preserves_materialized_template_helpers(tmp_path):
    source = """#include <metal_stdlib>
    template <typename T> T fmin(T x, T y) { return x + y; }
    kernel void ownership(device float* results [[buffer(0)]]) {
        results[0] = metal::fmin(2.0f, 3.0f);
        results[1] = fmin<float>(2.0f, 3.0f);
    }"""
    intermediate = _translate(tmp_path, source, "crossgl")
    assert "buffer_store(results, 0, fmin(2.0f, 3.0f))" in intermediate
    assert "buffer_store(results, 1, fmin_float(2.0f, 3.0f))" in intermediate
    for target in ("directx", "opengl", "metal"):
        _compile(_translate(tmp_path, source, target), target, tmp_path)


def test_metal_builtin_ownership_executes(tmp_path):
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 to require native builtin ownership checks")
    target = {"win32": "directx", "darwin": "metal", "linux": "opengl"}[sys.platform]
    generated = _translate(tmp_path, SOURCE, target)
    artifact, module = _compile(generated, target, tmp_path)
    assert module.is_file(), "The native compiler must produce a module"
    expected = [2.0, 5.0, 3.0, 10.0, 2.0, 6.0, 0.0, 0.0, 10.0]
    if target == "metal":
        runner = tmp_path / "readback"
        _run(
            [
                "swiftc",
                str(
                    ROOT
                    / "tests/fixtures/runtime_verification/metal_float_readback.swift"
                ),
                "-o",
                str(runner),
            ]
        )
        original = tmp_path / "original.air"
        original_library = tmp_path / "original.metallib"
        _run(
            [
                "xcrun",
                "--sdk",
                "macosx",
                "metal",
                "-Werror",
                "-c",
                str(tmp_path / "ownership.metal"),
                "-o",
                str(original),
            ]
        )
        _run(
            [
                "xcrun",
                "--sdk",
                "macosx",
                "metallib",
                str(original),
                "-o",
                str(original_library),
            ]
        )
        entry = re.search(r"\bkernel\s+void\s+(\w+)", generated).group(1)
        outputs = {}
        for label, library, name in (
            ("source", original_library, "ownership"),
            ("translated", module, entry),
        ):
            outputs[label] = json.loads(
                _run([str(runner), str(library), name, str(len(expected))])
            )
        (tmp_path / "outputs.json").write_text(
            json.dumps(outputs, indent=2), encoding="utf-8"
        )
        assert outputs["source"]["values"] == expected
        assert outputs["translated"]["values"] == expected
        return
    entry = "CSMain" if target == "directx" else "main"
    runtime = (
        DirectXComputeRuntime()
        if target == "directx"
        else OpenGLComputeRuntime(context_backends=("egl",))
    )
    request = NativeRuntimeDispatchRequest(
        target=target,
        artifact={"target": target},
        artifact_path=artifact,
        module_path=module,
        loaded_artifact=module.read_bytes() if target == "directx" else generated,
        buffers={
            "results": NativeRuntimeBufferBinding(
                name="results",
                binding=RuntimeResourceBinding(
                    name="results",
                    kind="buffer",
                    type_name="RWStructuredBuffer<float>",
                    set=0,
                    binding=0,
                    access="read_write",
                ),
                source="expectedOutput",
                dtype="float32",
                shape=(len(expected),),
                value=[-1234.0] * len(expected),
            )
        },
        constants={},
        entry_point=entry,
        dispatch=RuntimeDispatchGeometry(
            entry_point=entry, workgroup_size=(1, 1, 1), workgroup_count=(1, 1, 1)
        ),
    )
    availability = runtime.is_available(
        None,
        RuntimeExecutionRequest(
            fixture=RuntimeFixture(
                id="metal-builtin-ownership",
                selector=RuntimeArtifactSelector(target=target),
                entry_point=entry,
            ),
            artifact=request.artifact,
            artifact_path=artifact,
            project_root=tmp_path,
        ),
    )
    assert availability.available, availability.reason
    outputs = runtime.dispatch(None, None, request)
    (tmp_path / "outputs.json").write_text(
        json.dumps(outputs, indent=2), encoding="utf-8"
    )
    assert outputs["results"]["values"] == expected

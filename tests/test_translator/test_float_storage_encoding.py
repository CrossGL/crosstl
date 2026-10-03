"""Lossless float32 transport through typed native runtime contracts."""

import hashlib
import json
import os
import struct
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from crosstl.project import (
    build_native_loader_abi_descriptor,
    build_native_loader_dispatch_request,
    build_runtime_artifact_manifest,
    build_runtime_loader_manifest,
    build_runtime_package,
    compare_runtime_outputs,
    load_project_config,
    parse_runtime_verification_fixtures,
    translate_project,
)
from crosstl.project.native_loader_dispatch import NativeLoaderDispatchError
from crosstl.project.native_runtime_drivers import (
    _buffer_readback,
    _pack_values,
    _prepare_directx_buffers,
    _prepare_opengl_buffers,
    _unpack_values,
)
from crosstl.project.runtime_value_encoding import FLOAT32_BITS
from crosstl.project.runtime_verification import (
    RuntimeAllocationView,
    RuntimeExecutionState,
    RuntimeTolerance,
    RuntimeValue,
    _compare_runtime_value,
    _parse_runtime_value,
)
from tests.test_translator.test_boolean_buffer_runtime import _bound_values
from tests.test_translator.test_metal_builtin_ownership import _compile
from tests.test_translator.test_metal_native_runtime import _native_request
from tests.test_translator.test_native_loader_dispatch import _build
from tests.test_translator.test_native_loader_dispatch_integration import _executor

REQUIRE_ENV = "CROSTL_REQUIRE_FLOAT_STORAGE_RUNTIME"
SOURCE = """#include <metal_stdlib>
using namespace metal;
kernel void float_storage(device float* state [[buffer(0)]],
                          const device float* additions [[buffer(1)]],
                          device uint* observed [[buffer(2)]],
                          uint i [[thread_position_in_grid]]) {
    uint index = i * 3 + 9;
    state[index] += additions[i];
    observed[i] = as_type<uint>(state[index]);
}
"""

WORDS = [
    0,
    0x80000000,
    1,
    0x80000001,
    0x007FFFFF,
    0x807FFFFF,
    0x7F800000,
    0xFF800000,
    0x7FC00000,
    0x7FC12345,
    0xFFC12345,
    0x7FA00001,
    0xFFA00001,
    0x7F7FFFFF,
    0xFF7FFFFF,
    0x3F800000,
]


@pytest.mark.parametrize("target", ["Metal", "OpenGL", "DirectX", "Vulkan"])
def test_storage_packing_never_converts_through_float(target):
    payload = _pack_values(
        [WORDS[:8], WORDS[8:]],
        "float32",
        expected_count=len(WORDS),
        target=target,
        encoding=FLOAT32_BITS,
    )
    assert payload == struct.pack("<16I", *WORDS)
    assert (
        _unpack_values(payload, "float32", target=target, encoding=FLOAT32_BITS)
        == WORDS
    )
    readback = _buffer_readback(
        payload, "float32", [16], target=target, encoding=FLOAT32_BITS
    )
    assert readback == {
        "dtype": "float32",
        "shape": [16],
        "encoding": FLOAT32_BITS,
        "values": WORDS,
    }
    assert json.loads(json.dumps(readback, allow_nan=False)) == readback


@pytest.mark.parametrize("target", ["directx", "opengl"])
@pytest.mark.parametrize(
    "input_encoded,output_encoded", [(True, True), (True, False), (False, True)]
)
def test_public_dispatch_keeps_upload_and_readback_encodings_independent(
    tmp_path, target, input_encoded, output_encoded
):
    inputs = {
        "input_values": {
            "dtype": "float32",
            "shape": [4],
            "values": WORDS[8:12] if input_encoded else [1.0] * 4,
        }
    }
    outputs = {"output_values": {"dtype": "float32", "shape": [4]}}
    if input_encoded:
        inputs["input_values"]["encoding"] = FLOAT32_BITS
    if output_encoded:
        outputs["output_values"]["encoding"] = FLOAT32_BITS
    request = _build(tmp_path, target, input_values=inputs, output_values=outputs)
    _, native = _native_request(request)
    prepare = (
        _prepare_directx_buffers if target == "directx" else _prepare_opengl_buffers
    )
    buffers = {item.name: item for item in prepare(native.buffers)}
    assert buffers["input_values"].payload == (
        struct.pack("<4I", *WORDS[8:12])
        if input_encoded
        else struct.pack("<4f", *([1.0] * 4))
    )
    assert buffers["output_values"].readback_encoding == (
        FLOAT32_BITS if output_encoded else None
    )
    assert native.buffers["input_values"].to_json().get("encoding") == (
        FLOAT32_BITS if input_encoded else None
    )


@pytest.mark.parametrize(
    "fault",
    [
        "unknown-encoding",
        "wrong-dtype",
        "bool-word",
        "negative-word",
        "large-word",
        "float-word",
        "string-word",
        "object-word",
        "count",
        "conflicting-values",
        "encoding-object",
    ],
)
@pytest.mark.parametrize("role", ["input", "output"])
def test_public_dispatch_rejects_invalid_storage(tmp_path, fault, role):
    value = {
        "dtype": "float32",
        "shape": [4],
        "encoding": FLOAT32_BITS,
        "values": WORDS[:4],
    }
    if fault == "unknown-encoding":
        value["encoding"] = "native-endian"
    elif fault == "wrong-dtype":
        value["dtype"] = "uint32"
    elif fault == "count":
        value["values"] = [1]
    elif fault == "conflicting-values":
        value["value"] = [1] * 4
    elif fault == "encoding-object":
        value["encoding"] = {"format": FLOAT32_BITS}
    else:
        invalid = {
            "bool-word": True,
            "negative-word": -1,
            "large-word": 2**32,
            "float-word": 1.0,
            "string-word": "nan",
            "object-word": {"bits": 1},
        }
        value["values"][0] = invalid[fault]
    key, name = (
        ("input_values", "input_values")
        if role == "input"
        else ("output_values", "output_values")
    )
    with pytest.raises(NativeLoaderDispatchError) as caught:
        _build(tmp_path, **{key: {name: value}})
    assert caught.value.code.endswith(
        (".value-encoding-invalid", ".value-size-mismatch", ".value-schema-invalid")
    )


def test_runtime_value_serialization_retains_exact_storage():
    value = RuntimeValue(
        name="values", dtype="float32", shape=(16,), values=WORDS, encoding=FLOAT32_BITS
    )
    document = json.loads(json.dumps(value.to_json(), allow_nan=False))
    assert _parse_runtime_value(document, field_name="buffer") == value
    assert value.values == WORDS


def test_public_fixture_and_comparison_preserve_storage_contract():
    value = {
        "name": "out",
        "dtype": "float32",
        "shape": [len(WORDS)],
        "values": WORDS,
        "encoding": FLOAT32_BITS,
    }
    (fixture,) = parse_runtime_verification_fixtures(
        {
            "fixtures": [
                {
                    "id": "float-storage",
                    "selector": {"source": "storage.metal", "target": "metal"},
                    "inputs": [value],
                    "expectedOutputs": [value],
                }
            ]
        }
    )
    assert fixture.to_json()["inputs"][0]["encoding"] == FLOAT32_BITS
    actual = {"out": dict(value)}
    (comparison,) = compare_runtime_outputs(fixture.expected_outputs, actual)
    assert comparison["status"] == "passed" and comparison["comparison"] == "bitwise"
    actual["out"]["values"] = list(WORDS)
    actual["out"]["values"][9] ^= 1
    (comparison,) = compare_runtime_outputs(
        fixture.expected_outputs, actual, default_tolerance={"absolute": 1e30}
    )
    assert comparison["status"] == "comparison-failed"


@pytest.mark.parametrize("target", ["directx", "opengl"])
def test_encoded_payload_cannot_reinterpret_reflected_storage(tmp_path, target):
    descriptor, package = _package(tmp_path, target)
    inputs, outputs = _values(1)
    inputs["observed"]["dtype"] = "float32"
    inputs["observed"]["encoding"] = FLOAT32_BITS
    with pytest.raises(NativeLoaderDispatchError) as caught:
        build_native_loader_dispatch_request(
            descriptor,
            package,
            _bound_values(descriptor, inputs),
            _bound_values(descriptor, outputs),
            {"workgroupCount": [16, 1, 1], "workgroupSize": [1, 1, 1]},
            expected_target=target,
        )
    assert "layout" in caught.value.code or "dtype" in caught.value.code


@pytest.mark.parametrize(
    "change",
    [
        None,
        "nan-payload",
        "zero-sign",
        "one-bit",
        "missing-encoding",
        "wrong-word-type",
    ],
)
def test_storage_comparison_is_exact_even_with_large_numeric_tolerance(change):
    expected = RuntimeValue(
        name="out",
        dtype="float32",
        shape=(16,),
        values=WORDS,
        encoding=FLOAT32_BITS,
        tolerance=RuntimeTolerance(1e30, 1e30),
    )
    actual = replace(expected, values=list(WORDS))
    if change == "nan-payload":
        actual.values[9] = 0x7FC00000
    elif change == "zero-sign":
        actual.values[0] = 0x80000000
    elif change == "one-bit":
        actual.values[-1] ^= 1
    elif change == "missing-encoding":
        actual = replace(actual, encoding=None)
    elif change == "wrong-word-type":
        actual.values[-1] = float(actual.values[-1])
    result = _compare_runtime_value(
        expected, actual, default_tolerance=RuntimeTolerance(1e30, 1e30)
    )
    assert result["status"] == ("passed" if change is None else "comparison-failed")
    assert result["comparison"] == "bitwise"
    assert result["tolerance"] == {"absolute": 0.0, "relative": 0.0}


def _package(root, target):
    root.mkdir(parents=True, exist_ok=True)
    (root / "storage.metal").write_text(SOURCE, encoding="utf-8")
    (root / "crosstl.toml").write_text(
        '[project]\nsource_roots = ["."]\ninclude = ["storage.metal"]\n'
        f'targets = ["{target}"]\noutput_dir = "out"\n'
        "workgroup_size = [1, 1, 1]\n",
        encoding="utf-8",
    )
    report = translate_project(load_project_config(root), format_output=False)
    report.write_json(root / "report.json")
    assert report.to_json()["summary"]["failedCount"] == 0, report.to_json()
    manifest = build_runtime_artifact_manifest(root / "report.json")
    assert manifest["success"], manifest
    (root / "artifacts.json").write_text(json.dumps(manifest), encoding="utf-8")
    package = root / "package"
    assert build_runtime_package(root / "artifacts.json", package)["success"]
    loader = build_runtime_loader_manifest(package / "runtime-package.json")
    assert loader["success"] and len(loader["loadUnits"]) == 1, loader
    descriptor = build_native_loader_abi_descriptor(
        loader, load_unit_id=loader["loadUnits"][0]["id"]
    )
    (root / "descriptor.json").write_text(
        json.dumps(descriptor, indent=2), encoding="utf-8"
    )
    return descriptor, package


def _values(steps):
    guard = [0x42F60000] * 8
    initial = (
        guard
        + [
            word
            for i, value in enumerate(WORDS)
            for word in (value, 0x3F800000, WORDS[-i - 1])
        ]
        + guard
    )
    expected = list(initial)
    result_word = {1: 0x3FC00000, 2: 0x40000000}[steps]
    for i in range(len(WORDS)):
        expected[i * 3 + 9] = result_word

    def typed(values, dtype="float32", encoding=FLOAT32_BITS):
        return {
            "dtype": dtype,
            "shape": [len(values)],
            "values": values,
            **({"encoding": encoding} if encoding else {}),
        }

    observed = [0xDEADBEEF] * (len(WORDS) + 8)
    return (
        {
            "state": typed(initial),
            "additions": typed([0x3F000000] * len(WORDS)),
            "observed": typed(observed, "uint32", None),
        },
        {
            "state": typed(expected),
            "observed": typed(
                [result_word] * len(WORDS) + observed[-8:], "uint32", None
            ),
        },
    )


@pytest.mark.parametrize("target", ["metal", "directx", "opengl"])
def test_float_storage_package_retains_float_arithmetic(tmp_path, target):
    descriptor, package = _package(tmp_path, target)
    inputs, outputs = _values(1)
    request = build_native_loader_dispatch_request(
        descriptor,
        package,
        _bound_values(descriptor, inputs),
        _bound_values(descriptor, outputs),
        {"workgroupCount": [len(WORDS), 1, 1], "workgroupSize": [1, 1, 1]},
        expected_target=target,
    )
    assert not request.execution_plan.diagnostics
    bindings = {
        item["scalarLayout"].get("memberName", item["name"]): item
        for item in descriptor["bindings"]
    }
    assert bindings["state"]["scalarLayout"]["elementType"] == "float32"
    assert bindings["additions"]["scalarLayout"]["elementType"] == "float32"
    _compile(request.artifact_path.read_text(), target, tmp_path)


@pytest.mark.parametrize("steps", [1, 2])
def test_float_storage_executes_natively(tmp_path, steps):
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 for required native float storage")
    target = {"darwin": "metal", "win32": "directx", "linux": "opengl"}[sys.platform]
    if target == "metal" and steps == 2:
        pytest.skip("Metal does not expose the shared-allocation sequence API")
    descriptor, package = _package(tmp_path, target)
    inputs, outputs = _values(steps)
    expected = _bound_values(descriptor, outputs)
    request = build_native_loader_dispatch_request(
        descriptor,
        package,
        _bound_values(descriptor, inputs),
        expected,
        {"workgroupCount": [len(WORDS), 1, 1], "workgroupSize": [1, 1, 1]},
        expected_target=target,
    )
    executor = _executor(target)
    state = RuntimeExecutionState(request=request, plan=request.execution_plan)
    records = {}
    try:
        available = executor.is_available(request)
        assert available.available, available
        validation = tmp_path / "validation"
        validation.mkdir()
        _, validated_module = _compile(
            request.artifact_path.read_text(), target, validation
        )
        assert validated_module.is_file() and validated_module.stat().st_size
        native = executor.runtime_adapter.prepare_buffers(state)
        retained = tmp_path / native.module_path.name
        retained.write_bytes(native.module_path.read_bytes())
        if steps == 1:
            requests = (native,)
            actual = executor.runtime_adapter.dispatch(state, native)
        else:
            bindings = {
                name: replace(
                    value,
                    allocation=RuntimeAllocationView(
                        name, 0, value.shape[0] * 4, value.shape[0] * 4
                    ),
                )
                for name, value in native.buffers.items()
            }
            first = replace(
                native,
                buffers={
                    name: replace(value, expected_output=None)
                    for name, value in bindings.items()
                },
            )
            second = replace(
                native,
                buffers={
                    name: replace(value, value=None) for name, value in bindings.items()
                },
            )
            requests = (first, second)
            actual = executor.runtime_adapter.runtime.dispatch_sequence(
                None, state, requests
            )
        assert actual == expected
        records["generated"] = {
            "outputs": actual,
            "details": state.details,
            "adapterSteps": [step.to_json() for step in state.adapter_steps],
            "requests": [item.to_json() for item in requests],
            "moduleFile": retained.name,
            "moduleKind": "source" if target == "opengl" else "binary",
            "moduleSha256": hashlib.sha256(retained.read_bytes()).hexdigest(),
            "validationModuleFile": str(validated_module.relative_to(tmp_path)),
            "validationModuleSha256": (
                hashlib.sha256(validated_module.read_bytes()).hexdigest()
            ),
        }
        if target == "metal":
            original = tmp_path / "original"
            original.mkdir()
            source, module = _compile(
                SOURCE, "metal", original, metal_compile_flags=("-fno-fast-math",)
            )
            control = replace(
                native,
                artifact_path=source,
                module_path=module,
                entry_point="float_storage",
                loaded_artifact=None,
            )
            original_outputs = executor.runtime_adapter.runtime.dispatch(
                None, state, control
            )
            assert original_outputs == expected
            records["originalMetal"] = {
                "outputs": original_outputs,
                "moduleFile": str(module.relative_to(tmp_path)),
                "moduleSha256": hashlib.sha256(module.read_bytes()).hexdigest(),
            }
        (tmp_path / "evidence.json").write_text(
            json.dumps(
                {
                    "target": target,
                    "steps": steps,
                    "inputs": inputs,
                    "expected": expected,
                    "sourceSha256": hashlib.sha256(SOURCE.encode()).hexdigest(),
                    "artifactSha256": (
                        hashlib.sha256(request.artifact_path.read_bytes()).hexdigest()
                    ),
                    "records": records,
                },
                indent=2,
                allow_nan=False,
            ),
            encoding="utf-8",
        )
    finally:
        for directory in state.temporary_directories:
            directory.cleanup()
        close = getattr(executor.runtime_adapter.runtime, "close", None)
        if close:
            close()


def test_float_storage_native_gate_is_required():
    from tools import ci_coverage

    workflow = (
        Path(__file__).resolve().parents[2] / ".github/workflows/mlx-portable-host.yml"
    ).read_text()
    step = ci_coverage.workflow_step_section(
        workflow, "Validate lossless float storage"
    )
    assert "test_float_storage_encoding.py" in step
    assert f'{REQUIRE_ENV}: "1"' in step
    assert "-n auto" in step and "--junitxml" in step and "--timeout-seconds" in step
    assert "if:" not in step and "continue-on-error" not in step
    for event in ("push", "pull_request"):
        assert (
            "tests/test_translator/test_float_storage_encoding.py"
            in ci_coverage.workflow_event_path_filters(workflow, event)
        )

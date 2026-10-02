"""Boolean storage through project packages and native target runtimes."""

import base64
import copy
import hashlib
import json
import os
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from crosstl.project import (
    MetalComputeRuntime,
    NativeLoaderDispatchError,
    build_native_loader_abi_descriptor,
    build_native_loader_dispatch_request,
    build_runtime_artifact_manifest,
    build_runtime_loader_manifest,
    build_runtime_package,
    load_project_config,
    translate_project,
)
from crosstl.project.native_runtime_drivers import (
    _normalize_dtype,
    _pack_values,
    _unpack_values,
)
from crosstl.project.runtime_verification import (
    RuntimeAdapterDispatchError,
    RuntimeAdapterSetupError,
    RuntimeAllocationView,
    RuntimeExecutorUnavailable,
    RuntimeTolerance,
    RuntimeValue,
    _values_match,
)
from tests.test_translator.test_metal_builtin_ownership import _compile
from tests.test_translator.test_metal_native_runtime import _native_request
from tests.test_translator.test_native_loader_dispatch_integration import _executor

REQUIRE_ENV = "CROSTL_REQUIRE_BOOLEAN_BUFFER_RUNTIME"
SOURCE = """#include <metal_stdlib>
using namespace metal;
template <typename T>
kernel void boolean_storage(const device T* values [[buffer(0)]],
                            const device bool* masks [[buffer(1)]],
                            device bool* results [[buffer(2)]],
                            device bool* state [[buffer(3)]],
                            device uint* observed [[buffer(4)]],
                            uint i [[thread_position_in_grid]]) {
    uint j = i + 3;
    results[j] = masks[j] && values[i] < 0.0f;
    state[j] = state[j] != masks[j];
    observed[i] = masks[j] ? 1u : 0u;
}
template [[host_name("boolean_storage_float")]] [[kernel]]
decltype(boolean_storage<float>) boolean_storage<float>;
"""


def _package(root, target, source=SOURCE):
    root.mkdir(parents=True, exist_ok=True)
    (root / "boolean.metal").write_text(source, encoding="utf-8")
    (root / "crosstl.toml").write_text(
        '[project]\nsource_roots = ["."]\ninclude = ["boolean.metal"]\n'
        f'targets = ["{target}"]\noutput_dir = "out"\n'
        '[project.entry_workgroup_size_rules."boolean.metal"]\n'
        '"boolean_storage_float" = [1, 1, 1]\n',
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
    assert loader["success"], loader
    assert len(loader["loadUnits"]) == 1
    descriptor = build_native_loader_abi_descriptor(
        loader, load_unit_id=loader["loadUnits"][0]["id"]
    )
    (root / "descriptor.json").write_text(
        json.dumps(descriptor, indent=2), encoding="utf-8"
    )
    return descriptor, package


def _values(target, count):
    dtype = "bool" if target == "metal" else "uint32"

    def boolean(values):
        return {
            "dtype": dtype,
            "shape": [len(values)],
            "values": values if dtype == "bool" else [int(value) for value in values],
        }

    values = [float(i % 7 - 3) for i in range(count)]
    masks = [i % 3 != 0 for i in range(count + 8)]
    initial = [i % 2 == 0 for i in range(count + 8)]
    results, state = list(initial), list(initial)
    for i, value in enumerate(values):
        results[i + 3] = masks[i + 3] and value < 0
        state[i + 3] = initial[i + 3] != masks[i + 3]
    inputs = {
        "values": {"dtype": "float32", "shape": [count], "values": values},
        "masks": boolean(masks),
        "results": boolean(initial),
        "state": boolean(initial),
        "observed": {"dtype": "uint32", "shape": [count], "values": [99] * count},
    }
    outputs = {
        "results": boolean(results),
        "state": boolean(state),
        "observed": {
            "dtype": "uint32",
            "shape": [count],
            "values": [int(value) for value in masks[3 : count + 3]],
        },
    }
    return inputs, outputs


def _request(descriptor, package, inputs, outputs, count):
    return build_native_loader_dispatch_request(
        descriptor,
        package,
        _bound_values(descriptor, inputs),
        _bound_values(descriptor, outputs),
        {"workgroupCount": [count, 1, 1], "workgroupSize": [1, 1, 1]},
        expected_target=descriptor["target"],
    )


def _bound_values(descriptor, values):
    names = {
        binding["scalarLayout"].get("memberName", binding["name"]): binding["name"]
        for binding in descriptor["bindings"]
    }
    return {
        names[name]: (
            replace(value, name=names[name])
            if isinstance(value, RuntimeValue)
            else value
        )
        for name, value in values.items()
    }


@pytest.mark.parametrize("target", ["metal", "directx", "opengl"])
def test_boolean_package_uses_target_physical_storage(tmp_path, target):
    descriptor, package = _package(tmp_path, target)
    inputs, outputs = _values(target, 7)
    request = _request(descriptor, package, inputs, outputs, 7)
    assert not request.execution_plan.diagnostics
    layouts = {
        binding["scalarLayout"].get("memberName", binding["name"]): binding[
            "scalarLayout"
        ]
        for binding in descriptor["bindings"]
    }
    for name in ("masks", "results", "state"):
        layout = layouts[name]
        assert layout["elementType"] == ("bool" if target == "metal" else "uint32")
        assert (
            layout["elementSizeBytes"]
            == layout["elementStrideBytes"]
            == (1 if target == "metal" else 4)
        )
    _compile(request.artifact_path.read_text(), target, tmp_path)
    if target == "metal":
        _, native = _native_request(request)
        payload, readbacks = MetalComputeRuntime()._prepare_request(native)
        assert readbacks["results"] == ("bool", (15,), 15, None)
        assert sorted(item["length"] for item in payload["allocations"]) == [
            15,
            15,
            15,
            28,
            28,
        ]


@pytest.mark.parametrize("target", ["directx", "opengl"])
def test_boolean_byte_payload_is_not_a_32_bit_storage_payload(tmp_path, target):
    descriptor, package = _package(tmp_path, target)
    inputs, outputs = _values("metal", 7)
    with pytest.raises(NativeLoaderDispatchError, match="resource-layout-mismatch"):
        _request(descriptor, package, inputs, outputs, 7)


@pytest.mark.parametrize("value", [0, 1, 2, -1, "true", None, 1.0])
def test_boolean_storage_does_not_coerce_nonboolean_values(tmp_path, value):
    descriptor, package = _package(tmp_path, "metal")
    inputs, outputs = _values("metal", 7)
    inputs["masks"]["values"][0] = value
    with pytest.raises(NativeLoaderDispatchError, match="value-"):
        _request(descriptor, package, inputs, outputs, 7)
    reason = "count does not match shape" if value is None else "true or false"
    with pytest.raises(RuntimeExecutorUnavailable, match=reason):
        _pack_values([value], "bool", expected_count=1, target="Metal")


@pytest.mark.parametrize("size", [14, 16])
def test_boolean_loader_rejects_payload_shape_mismatch(tmp_path, size):
    descriptor, package = _package(tmp_path, "metal")
    inputs, outputs = _values("metal", 7)
    inputs["masks"]["values"] = [False] * size
    with pytest.raises(NativeLoaderDispatchError, match="value-size-mismatch"):
        _request(descriptor, package, inputs, outputs, 7)


@pytest.mark.parametrize("byte", [2, 127, 128, 255])
def test_boolean_readback_rejects_noncanonical_bytes(byte):
    with pytest.raises(RuntimeAdapterDispatchError, match="noncanonical"):
        _unpack_values(bytes([0, byte, 1]), "bool", target="Metal")


def test_boolean_packing_has_exact_byte_width():
    assert _normalize_dtype("boolean", target="Metal") == "bool"
    assert (
        _pack_values([True, False, True], "bool", expected_count=3, target="Metal")
        == b"\1\0\1"
    )
    assert _unpack_values(b"\1\0\1", "bool", target="Metal") == [True, False, True]
    for target in ("DirectX", "OpenGL", "Vulkan"):
        with pytest.raises(RuntimeExecutorUnavailable):
            _normalize_dtype("bool", target=target)


@pytest.mark.parametrize(
    "field,value",
    [
        ("elementStrideBytes", 4),
        ("elementSizeBytes", 4),
        ("alignmentBytes", 4),
        ("memberOffsetBytes", 1),
    ],
)
def test_boolean_loader_rejects_incompatible_layout(tmp_path, field, value):
    descriptor, package = _package(tmp_path, "metal")
    descriptor = copy.deepcopy(descriptor)
    binding = next(
        binding for binding in descriptor["bindings"] if binding["name"] == "masks"
    )
    binding["scalarLayout"][field] = value
    recorded = next(
        binding
        for binding in descriptor["scalarLayout"]["bindings"]
        if binding["binding"] == "masks"
    )
    recorded["layout"][field] = value
    inputs, outputs = _values("metal", 7)
    with pytest.raises(NativeLoaderDispatchError, match="resource-layout-"):
        _request(descriptor, package, inputs, outputs, 7)


def test_boolean_offset_views_preserve_exact_upload_bytes(tmp_path):
    descriptor, package = _package(tmp_path, "metal")
    inputs, outputs = _values("metal", 7)
    request = _request(descriptor, package, inputs, outputs, 7)
    _, native = _native_request(request)
    binding = native.buffers["masks"]
    native = replace(
        native,
        buffers={
            **native.buffers,
            "masks": replace(
                binding, allocation=RuntimeAllocationView("offset-mask", 3, 15, 23)
            ),
        },
    )
    payload, _ = MetalComputeRuntime()._prepare_request(native)
    allocation = next(
        item for item in payload["allocations"] if item["id"] == "offset-mask"
    )
    assert allocation["length"] == 23
    assert allocation["uploads"][0]["offset"] == 3
    assert base64.b64decode(allocation["uploads"][0]["data"]) == bytes(
        inputs["masks"]["values"]
    )


@pytest.mark.parametrize(
    "offset,length,backing", [(-1, 15, 23), (3, 14, 23), (3, 15, 17), (True, 15, 23)]
)
def test_boolean_invalid_views_fail_before_native_execution(
    tmp_path, offset, length, backing
):
    descriptor, package = _package(tmp_path, "metal")
    inputs, outputs = _values("metal", 7)
    _, native = _native_request(_request(descriptor, package, inputs, outputs, 7))
    binding = native.buffers["masks"]
    native = replace(
        native,
        buffers={
            **native.buffers,
            "masks": replace(
                binding,
                allocation=RuntimeAllocationView(
                    "invalid-mask", offset, length, backing
                ),
            ),
        },
    )
    with pytest.raises(RuntimeAdapterSetupError, match="allocation view"):
        MetalComputeRuntime()._prepare_request(native)


def _check_outputs(actual, expected):
    assert set(actual) == set(expected)
    for name, wanted in expected.items():
        observed = actual[name]
        assert observed["dtype"] == wanted["dtype"]
        assert observed["shape"] == wanted["shape"]
        assert observed["values"] == wanted["values"], name
        expected_type = bool if wanted["dtype"] == "bool" else int
        assert all(type(value) is expected_type for value in observed["values"])


@pytest.mark.parametrize(
    "expected,actual", [(True, 1), (False, 0), (1, True), (0, False)]
)
def test_boolean_comparison_does_not_accept_numeric_values(expected, actual):
    assert not _values_match(expected, actual, RuntimeTolerance())[0]


@pytest.mark.parametrize("index", [0, 3, -1])
def test_boolean_verifier_rejects_corrupt_values_and_guards(index):
    _, expected = _values("metal", 7)
    actual = copy.deepcopy(expected)
    actual["results"]["values"][index] = not actual["results"]["values"][index]
    with pytest.raises(AssertionError):
        _check_outputs(actual, expected)


def test_boolean_storage_is_required_in_native_workflows():
    from tools import ci_coverage

    root = Path(__file__).resolve().parents[2]
    for name in (
        "mlx-portable-host.yml",
        "mlx-metal-host.yml",
        "mlx-project-porting.yml",
    ):
        workflow = (root / ".github/workflows" / name).read_text()
        step = ci_coverage.workflow_step_section(
            workflow, "Validate pinned comparison arithmetic"
        )
        assert "test_boolean_buffer_runtime.py" in step
        assert f'{REQUIRE_ENV}: "1"' in step
        assert "-n auto" in step and "continue-on-error" not in step
        assert "--timeout-seconds" in step and "--junitxml" in step
        for event in ("pull_request", "push"):
            assert (
                "tests/test_translator/test_boolean_buffer_runtime.py"
                in ci_coverage.workflow_event_path_filters(workflow, event)
            )


@pytest.mark.parametrize("reference", [False, True])
def test_metal_boolean_constant_buffer_executes(tmp_path, reference):
    if os.environ.get(REQUIRE_ENV) != "1" or sys.platform != "darwin":
        pytest.skip("requires the native Metal boolean storage gate")
    source = SOURCE.replace("const device bool* masks", "constant bool* masks")
    inputs, outputs = _values("metal", 7)
    if reference:
        source = source.replace("constant bool* masks", "constant bool& masks").replace(
            "masks[j]", "masks"
        )
        inputs["masks"] = {"dtype": "bool", "shape": [1], "values": [True]}
        outputs["results"]["values"][3:10] = [
            value < 0 for value in inputs["values"]["values"]
        ]
        outputs["state"]["values"][3:10] = [
            not value for value in inputs["state"]["values"][3:10]
        ]
        outputs["observed"]["values"] = [1] * 7
    descriptor, package = _package(tmp_path, "metal", source)
    request = _request(descriptor, package, inputs, outputs, 7)
    executor = _executor("metal")
    try:
        assert executor.is_available(request).available
        result = executor.run(request)
        assert result.status == "ok", result
        _check_outputs(result.outputs, outputs)
        (tmp_path / "constant-evidence.json").write_text(
            json.dumps(
                {
                    "reference": reference,
                    "inputs": inputs,
                    "expected": outputs,
                    "outputs": result.outputs,
                    "details": result.details,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    finally:
        executor.runtime_adapter.runtime.close()


@pytest.mark.parametrize("count,offset", [(1, 0), (7, 0), (257, 0), (7, 3)])
def test_boolean_package_executes_on_device(tmp_path, count, offset):
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 for required native boolean storage")
    target = {"darwin": "metal", "win32": "directx", "linux": "opengl"}[sys.platform]
    if offset and target != "metal":
        offset = 256
    descriptor, package = _package(tmp_path, target)
    inputs, outputs = _values(target, count)
    if offset:
        for values in (inputs, outputs):
            for name in ("masks", "results", "state"):
                if name not in values:
                    continue
                value = values[name]
                size = value["shape"][0] * (1 if target == "metal" else 4)
                values[name] = RuntimeValue(
                    name=name,
                    dtype=value["dtype"],
                    shape=tuple(value["shape"]),
                    values=value["values"],
                    allocation=RuntimeAllocationView(
                        name, offset, size, offset + size + 16
                    ),
                )
    request = _request(descriptor, package, inputs, outputs, count)
    compiled = tmp_path / "compiled"
    compiled.mkdir()
    _, module = _compile(
        request.artifact_path.read_text(),
        target,
        compiled,
        metal_compile_flags=("-fno-fast-math",),
    )
    assert (
        module.is_file() and module.stat().st_size
    ), "native gate requires the target compiler"
    validation_hash = hashlib.sha256(module.read_bytes()).hexdigest()
    expected = _bound_values(
        descriptor,
        {
            name: (
                {
                    "dtype": value.dtype,
                    "shape": list(value.shape),
                    "values": value.values,
                }
                if isinstance(value, RuntimeValue)
                else value
            )
            for name, value in outputs.items()
        },
    )
    executor = _executor(target)
    records = {}
    try:
        availability = executor.is_available(request)
        assert availability.available, availability
        result = executor.run(request)
        assert result.status == "ok", result
        _check_outputs(result.outputs, expected)
        records["generated"] = {"outputs": result.outputs, "details": result.details}
        if target == "metal":
            original = tmp_path / "original"
            original.mkdir()
            source, module = _compile(
                SOURCE, target, original, metal_compile_flags=("-fno-fast-math",)
            )
            state, native = _native_request(request)
            native = replace(
                native,
                artifact_path=source,
                module_path=module,
                entry_point="boolean_storage_float",
            )
            actual = executor.runtime_adapter.runtime.dispatch(None, state, native)
            _check_outputs(actual, expected)
            records["original"] = {
                "outputs": actual,
                "details": state.details,
                "sourceSha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "moduleSha256": hashlib.sha256(module.read_bytes()).hexdigest(),
            }
    finally:
        close = getattr(executor.runtime_adapter.runtime, "close", None)
        if close:
            close()
    (tmp_path / "evidence.json").write_text(
        json.dumps(
            {
                "target": target,
                "count": count,
                "offset": offset,
                "inputs": {
                    name: value.to_json() if isinstance(value, RuntimeValue) else value
                    for name, value in inputs.items()
                },
                "expected": expected,
                "artifactSha256": (
                    hashlib.sha256(request.artifact_path.read_bytes()).hexdigest()
                ),
                "compilerValidationSha256": validation_hash,
                "records": records,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

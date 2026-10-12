"""Read-only observations preserve resource access and compare exact upload bytes."""

import hashlib
import json
import os
import struct
import sys
from dataclasses import replace
from types import SimpleNamespace

import pytest

from crosstl.project.native_loader_dispatch import (
    NativeLoaderDispatchError,
    build_native_loader_dispatch_request,
)
from crosstl.project.native_runtime_drivers import (
    VulkanComputeRuntime,
    _input_observation_readback,
    _prepare_directx_buffers,
    _prepare_opengl_buffers,
    _snapshot_native_buffer_binding,
)
from crosstl.project.runtime_verification import (
    NativeRuntimeBufferBinding,
    NativeRuntimeParityAdapter,
    RuntimeAdapterDispatchError,
    RuntimeAdapterSetupError,
    RuntimeExecutionState,
    RuntimeResourceBinding,
)
from tests.test_translator.test_loop_updates import _execute
from tests.test_translator.test_native_loader_dispatch import (
    _inputs,
    _outputs,
    _write_descriptor,
)
from tests.test_translator.test_runtime_input_ownership import _request

REQUIRE_ENV = "CROSTL_REQUIRE_INPUT_OBSERVATIONS"


@pytest.mark.parametrize("target", ("directx", "opengl"))
def test_loader_observations_do_not_create_outputs(tmp_path, target):
    descriptor = _write_descriptor(tmp_path, target, constants=[])
    names = ["input_values"]
    request = build_native_loader_dispatch_request(
        descriptor, tmp_path, _inputs(), _outputs(), [1, 1, 1], observe_inputs=names
    )
    names.clear()
    assert request.observed_inputs == ("input_values",)
    assert [value.name for value in request.fixture.expected_outputs] == [
        "output_values"
    ]
    state = RuntimeExecutionState(request, request.execution_plan)
    adapter = NativeRuntimeParityAdapter()
    adapter.target = target
    bindings = adapter._native_buffer_bindings(state)
    observed = [binding for binding in bindings.values() if binding.observe_input]
    assert len(observed) == 1
    binding = observed[0]
    assert binding.binding.access == "read" and binding.expected_output is None
    assert binding.to_json()["observeInput"] is True
    prepare = (
        _prepare_directx_buffers if target == "directx" else _prepare_opengl_buffers
    )
    prepared = next(value for value in prepare(bindings) if value.observe_input)
    assert not prepared.readback and not prepared.writable
    if target == "directx":
        assert prepared.namespace == "srv"


@pytest.mark.parametrize(
    "names", [None, "input_values", [1], [True], [[]], [""], ["input_values"] * 2]
)
def test_loader_rejects_invalid_observation_names(tmp_path, names):
    descriptor = _write_descriptor(tmp_path, constants=[])
    with pytest.raises(NativeLoaderDispatchError) as error:
        build_native_loader_dispatch_request(
            descriptor, tmp_path, _inputs(), _outputs(), [1, 1, 1], observe_inputs=names
        )
    assert error.value.code.endswith("observation-names-invalid")


@pytest.mark.parametrize("name", ["unknown", "output_values"])
def test_loader_rejects_non_input_observations(tmp_path, name):
    descriptor = _write_descriptor(tmp_path, constants=[])
    with pytest.raises(NativeLoaderDispatchError) as error:
        build_native_loader_dispatch_request(
            descriptor,
            tmp_path,
            _inputs(),
            _outputs(),
            [1, 1, 1],
            observe_inputs=[name],
        )
    assert error.value.code.endswith("observation-binding-invalid")


def _observed_binding():
    return _snapshot_native_buffer_binding(
        NativeRuntimeBufferBinding(
            name="input",
            binding=RuntimeResourceBinding(name="input", kind="buffer", access="read"),
            value=[0, 0x80000000, 0x7FC00123],
            dtype="float32",
            shape=(3,),
            encoding="ieee754-binary32",
            observe_input=True,
        ),
        target="directx",
    )


@pytest.mark.parametrize(
    "damage",
    [
        None,
        "bit",
        "zero-sign",
        "missing",
        "truncated",
        "extra",
        "invalid",
        "field",
        "type",
    ],
)
def test_adapter_requires_exact_observation_bytes(damage):
    binding = _observed_binding()
    expected = binding.upload_snapshot.payload
    data = bytearray(expected)
    if damage == "bit":
        data[-1] ^= 1
    elif damage == "zero-sign":
        data[7] ^= 128
    elif damage == "truncated":
        data.pop()
    elif damage == "extra":
        data.append(0)
    packet = {"observationBytes": data.hex()}
    if damage == "invalid":
        packet["observationBytes"] = "z" * len(data.hex())
    elif damage == "type":
        packet["observationBytes"] = bytes(data)
    elif damage == "field":
        packet["unchanged"] = True
    result = {"output": {"values": [42]}, "input": packet}
    if damage == "missing":
        del result["input"]
    state = SimpleNamespace(details={})
    adapter = NativeRuntimeParityAdapter()
    if damage:
        with pytest.raises(RuntimeAdapterDispatchError):
            adapter._collect_input_observations(state, {"input": binding}, result)
    else:
        assert adapter._collect_input_observations(
            state, {"input": binding}, result
        ) == {"output": {"values": [42]}}
        receipt = state.details["inputObservations"]["input"]
        assert bytes.fromhex(receipt["observedBytes"]) == expected
        assert receipt["observedSHA256"] == hashlib.sha256(expected).hexdigest()
        assert receipt["unchanged"] is True
    assert "input" in result or damage == "missing"
    if damage in {"bit", "zero-sign"}:
        assert state.details["inputObservations"]["input"]["unchanged"] is False


@pytest.mark.parametrize("builtin", (False, True))
def test_observations_fail_before_dispatch_on_unsupported_driver(builtin):
    def unexpected(*args):
        pytest.fail("An unsupported observation reached dispatch")

    runtime = (
        VulkanComputeRuntime() if builtin else SimpleNamespace(dispatch=unexpected)
    )
    adapter = NativeRuntimeParityAdapter(runtime=runtime)
    native = SimpleNamespace(buffers={"input": _observed_binding()})
    with pytest.raises(RuntimeAdapterSetupError) as error:
        adapter.dispatch(SimpleNamespace(details={}), native)
    assert error.value.details["reasonKind"] == "input-observation-unsupported"


@pytest.mark.parametrize("size", [0, 3, 5, 16])
def test_observations_reject_truncated_or_extra_native_bytes(size):
    with pytest.raises(RuntimeAdapterDispatchError):
        _input_observation_readback(bytes(size), 4, 4)


def test_observations_exclude_uniform_alignment_padding():
    data = struct.pack("<I", 17)
    assert _input_observation_readback(data + bytes(12), 16, 4) == {
        "observationBytes": data.hex()
    }


@pytest.mark.parametrize(
    "names",
    [
        ("missing",),
        ("output_values",),
        ["input_values"],
        ("input_values", "input_values"),
    ],
)
def test_direct_request_observation_selection_is_preflighted(tmp_path, names):
    descriptor = _write_descriptor(tmp_path, constants=[])
    request = build_native_loader_dispatch_request(
        descriptor, tmp_path, _inputs(), _outputs(), [1, 1, 1]
    )
    request = replace(request, observed_inputs=names)
    adapter = NativeRuntimeParityAdapter()
    adapter.target = "directx"
    with pytest.raises(RuntimeAdapterSetupError):
        adapter._native_buffer_bindings(
            RuntimeExecutionState(request, request.execution_plan)
        )


@pytest.mark.parametrize("target", ("directx", "opengl"))
@pytest.mark.parametrize("damage", ("access", "value", "flag", "output"))
def test_driver_observations_preserve_access_constraints(target, damage):
    binding = _observed_binding()
    resource = replace(binding.binding, binding=0, set=0)
    binding = replace(binding, binding=resource)
    if damage == "access":
        binding = replace(binding, binding=replace(resource, access="read_write"))
    elif damage == "value":
        binding = replace(binding, value=None)
    elif damage == "flag":
        binding = replace(binding, observe_input=1)
    else:
        binding = replace(binding, source="expectedOutput")
    prepare = (
        _prepare_directx_buffers if target == "directx" else _prepare_opengl_buffers
    )
    with pytest.raises(RuntimeAdapterSetupError) as error:
        prepare({"input": binding})
    assert error.value.details["reasonKind"].startswith("input-observation-")


def test_input_observations_are_required_in_native_ci():
    from pathlib import Path

    from tools import ci_coverage

    workflow = (
        Path(__file__).resolve().parents[2]
        / ".github/workflows/demo-project-testing.yml"
    ).read_text()
    job = ci_coverage.workflow_job_text(workflow, "portable-host")
    step = ci_coverage.workflow_step_section(
        job, "Validate collective helper arguments"
    )
    assert f'{REQUIRE_ENV}: "1"' in step
    assert "tests/test_translator/test_runtime_input_observations.py" in step
    assert "continue-on-error" not in step and "if:" not in step
    assert "--timeout-seconds 360" in step


@pytest.mark.parametrize("case", ("scalar", "record", "offset-view"))
def test_readonly_inputs_execute_with_observations(tmp_path, case):
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 for required native input observations")
    target = {"darwin": "metal", "linux": "opengl", "win32": "directx"}[sys.platform]
    request, expected = _request(tmp_path, target, case, False)
    observed = tuple(
        (resource.initial_value or resource.value).name
        for resource in request.execution_plan.resource_bindings
        if resource.binding.access == "read"
    )
    assert len(observed) == 2
    request = replace(request, observed_inputs=observed)
    _execute(request, expected, tmp_path)
    evidence = json.loads((tmp_path / "evidence.json").read_text())
    receipts = evidence["records"]["generated"]["details"]["inputObservations"]
    assert set(receipts) == set(observed)
    for receipt in receipts.values():
        assert receipt["binding"]["access"] == "read"
        assert receipt["unchanged"] is True
        data = bytes.fromhex(receipt["observedBytes"])
        assert hashlib.sha256(data).hexdigest() == receipt["uploadSnapshot"]["sha256"]
    if case == "offset-view":
        assert any(
            receipt.get("allocation", {}).get("byteOffset") == 256
            for receipt in receipts.values()
        )

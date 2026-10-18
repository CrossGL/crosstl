"""Prepared native upload bytes cannot drift with mutable fixture values."""

import copy
import hashlib
import struct
from dataclasses import FrozenInstanceError, replace

import pytest

from crosstl.project.native_runtime_drivers import (
    _native_buffer_element_count,
    _native_buffer_payload,
    _prepare_directx_buffers,
    _prepare_opengl_buffers,
    _prepare_vulkan_buffers,
    _snapshot_native_buffer_binding,
)
from crosstl.project.runtime_verification import (
    NativeRuntimeBufferBinding,
    RuntimeAdapterSetupError,
    RuntimeExecutionState,
    RuntimeResourceBinding,
)


@pytest.mark.parametrize("target", ("metal", "directx", "opengl", "vulkan"))
@pytest.mark.parametrize("shape", ((), (4,), (2, 2)))
@pytest.mark.parametrize("encoded", (False, True))
def test_upload_snapshot_pins_exact_values(target, shape, encoded):
    words = [0, 0x80000000, 0x7FC12345, 0x7F800000]
    values = [words[:2], words[2:]] if shape == (2, 2) else words[:]
    binding = NativeRuntimeBufferBinding(
        name="values",
        binding=RuntimeResourceBinding(name="values", kind="buffer"),
        value=values,
        dtype="float32" if encoded else "uint32",
        shape=shape,
        encoding="ieee754-binary32" if encoded else None,
    )
    captured = _snapshot_native_buffer_binding(binding, target=target)
    values[:] = ["not an upload value"]
    assert _native_buffer_element_count(captured) == 4
    expected = struct.pack("<4I", *words)
    assert (
        _native_buffer_payload(
            captured, captured.dtype, expected_count=4, target=target
        )
        == expected
    )
    assert _snapshot_native_buffer_binding(captured, target=target) is captured
    assert captured.to_json()["uploadSnapshot"] == {
        "dtype": captured.dtype,
        "shape": list(shape),
        "encoding": captured.encoding,
        "elementCount": 4,
        "sizeBytes": 16,
        "sha256": hashlib.sha256(expected).hexdigest(),
    }
    with pytest.raises(FrozenInstanceError):
        captured.upload_snapshot.payload = b"changed"
    assert "payload" not in captured.to_json()["uploadSnapshot"]


@pytest.mark.parametrize("target", ("metal", "directx"))
@pytest.mark.parametrize(
    "dtype,encoding,values,format",
    (
        ("float16", "ieee754-binary16", [0, 0x8000, 0x7E01, 0x7C00], "H"),
        ("int64", None, [0, -(1 << 63), (1 << 63) - 1, -1], "q"),
        ("uint64", None, [0, 1 << 63, (1 << 64) - 1, 0x123456789ABCDEF0], "Q"),
    ),
)
def test_upload_snapshot_preserves_wide_and_encoded_half_bits(
    target, dtype, encoding, values, format
):
    expected = struct.pack("<4" + format, *values)
    binding = NativeRuntimeBufferBinding(
        name="values",
        binding=RuntimeResourceBinding(name="values", kind="buffer"),
        value=values[:],
        dtype=dtype,
        shape=(4,),
        encoding=encoding,
    )
    captured = _snapshot_native_buffer_binding(binding, target=target)
    binding.value[:] = [0] * 4
    assert (
        _native_buffer_payload(captured, dtype, expected_count=4, target=target)
        == expected
    )


@pytest.mark.parametrize(
    "fault", ("dtype", "shape", "encoding", "count", "payload-type", "payload-size")
)
def test_upload_snapshot_rejects_incompatible_binding(fault):
    binding = NativeRuntimeBufferBinding(
        name="values",
        binding=RuntimeResourceBinding(name="values", kind="buffer"),
        value=[1, 2],
        dtype="uint32",
        shape=(2,),
    )
    captured = _snapshot_native_buffer_binding(binding, target="opengl")
    dtype, count = "uint32", 2
    if fault == "dtype":
        dtype = "int32"
    elif fault == "shape":
        captured = replace(captured, shape=(1, 2))
    elif fault == "encoding":
        captured = replace(captured, encoding="ieee754-binary32")
    elif fault == "count":
        count = 3
    else:
        payload = bytearray(8) if fault == "payload-type" else b"short"
        captured = replace(
            captured, upload_snapshot=replace(captured.upload_snapshot, payload=payload)
        )
    with pytest.raises(RuntimeAdapterSetupError) as error:
        _native_buffer_payload(captured, dtype, expected_count=count, target="opengl")
    assert error.value.details["reasonKind"] == "upload-snapshot-mismatch"


@pytest.mark.parametrize(
    "target,prepare",
    (
        ("directx", _prepare_directx_buffers),
        ("opengl", _prepare_opengl_buffers),
        ("vulkan", _prepare_vulkan_buffers),
    ),
)
def test_native_buffer_preparation_consumes_snapshot(target, prepare):
    binding = NativeRuntimeBufferBinding(
        name="values",
        binding=RuntimeResourceBinding(
            name="values", kind="buffer", binding=0, set=0, access="read"
        ),
        value=[1, 2],
        dtype="uint32",
        shape=(2,),
    )
    captured = _snapshot_native_buffer_binding(binding, target=target)
    binding.value[:] = [99, 100]
    (prepared,) = prepare({"values": captured})
    assert prepared.payload == struct.pack("<2I", 1, 2)
    (legacy,) = prepare({"values": binding})
    assert legacy.payload == struct.pack("<2I", 99, 100)


def test_uninitialized_output_has_no_upload_snapshot():
    binding = NativeRuntimeBufferBinding(
        name="out",
        binding=RuntimeResourceBinding(name="out", kind="buffer"),
        value=None,
        dtype="uint32",
        shape=(4,),
    )
    assert _snapshot_native_buffer_binding(binding, target="opengl") is binding
    assert "uploadSnapshot" not in binding.to_json()


@pytest.mark.parametrize("target", ("metal", "directx", "opengl"))
def test_adapter_snapshot_owns_reflected_metadata(tmp_path, target):
    from tests.test_translator.test_native_loader_dispatch_integration import _executor
    from tests.test_translator.test_runtime_input_ownership import _request

    request, _ = _request(tmp_path, target, "record", False)
    state = RuntimeExecutionState(request=request, plan=request.execution_plan)
    prepared = _executor(target).runtime_adapter._native_buffer_bindings(state)
    expected = {
        name: copy.deepcopy(binding.to_json()) for name, binding in prepared.items()
    }
    for resource in state.plan.resource_bindings:
        resource.binding.metadata.clear()
        value = resource.initial_value or resource.value
        value.values[:] = [999]
    assert {name: binding.to_json() for name, binding in prepared.items()} == expected
    for binding in prepared.values():
        snapshot = binding.upload_snapshot
        assert snapshot is not None
        assert (
            _native_buffer_payload(
                binding,
                snapshot.dtype,
                expected_count=snapshot.element_count,
                target=target,
            )
            == snapshot.payload
        )

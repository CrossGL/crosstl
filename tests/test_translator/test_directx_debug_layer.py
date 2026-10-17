"""Optional native validation retains bounded messages without hiding failures."""

import struct
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from crosstl.project import directx_runtime
from crosstl.project.runtime_verification import (
    RuntimeAdapterDispatchError,
    RuntimeAdapterSetupError,
)
from tests.test_translator.test_directx_buffer_views import (
    _device,
    _plan,
    _request,
    _response,
    _runtime_packet,
)
from tests.test_translator.test_directx_buffer_views import (
    protocol_worker as protocol_worker,
)


def _messages(*, severity=2, discarded=0, truncated=0, text=b"Validation detail"):
    return (
        struct.pack("<IQQI", 0x31475844, 1, discarded, 1)
        + struct.pack("<5I", 882, 6, severity, truncated, len(text))
        + text
    )


@pytest.mark.parametrize("value", [0, 1, None, "false", [], {}])
def test_debug_layer_requires_boolean_before_preparation(monkeypatch, value):
    def unexpected(*args, **kwargs):
        pytest.fail("Invalid debug option reached request preparation")

    monkeypatch.setattr(directx_runtime, "encode_dispatches", unexpected)
    with pytest.raises(RuntimeAdapterSetupError) as error:
        directx_runtime.execute_buffer_views(
            None, None, None, None, device=_device(), debug_layer=value
        )
    assert error.value.details["reasonKind"] == "debug-layer-invalid"


@pytest.mark.parametrize("severity", range(5))
@pytest.mark.parametrize("truncated", [0, 1])
def test_debug_message_receipt_preserves_severity_and_limits(
    tmp_path, severity, truncated
):
    path = tmp_path / "messages.bin"
    path.write_bytes(_messages(severity=severity, discarded=17, truncated=truncated))
    assert directx_runtime._worker_debug_messages(path) == {
        "enabled": True,
        "discardedMessages": 17,
        "messages": [
            {
                "id": 882,
                "category": 6,
                "severity": severity,
                "description": "Validation detail",
                "truncated": bool(truncated),
            }
        ],
    }


@pytest.mark.parametrize(
    "payload",
    [
        b"",
        b"x" * 270000,
        struct.pack("<IQQI", 0, 0, 0, 0),
        struct.pack("<IQQI", 0x31475844, 65, 0, 65),
        struct.pack("<IQQI", 0x31475844, 1, 0, 0),
        struct.pack("<IQQI", 0x31475844, 0, 0, 1),
        _messages()[:-1],
        _messages() + b"\0",
        _messages(text=b"has\0nul"),
        _messages(severity=5),
        _messages(truncated=2),
        _messages(text=b"x" * 4097),
    ],
    ids=range(12),
)
def test_debug_message_receipt_rejects_invalid_packets(tmp_path, payload):
    path = tmp_path / "messages.bin"
    path.write_bytes(payload)
    with pytest.raises(RuntimeAdapterSetupError) as error:
        directx_runtime._worker_debug_messages(path)
    assert error.value.details["reasonKind"] == "worker-output-invalid"


@pytest.mark.parametrize("failure", [None, "dispatch", "timeout"])
@pytest.mark.parametrize(
    "receipt", ["warning", "error", "discarded", "missing", "broken"]
)
def test_debug_messages_preserve_original_execution_failure(
    tmp_path, monkeypatch, failure, receipt
):
    nodes, allocations, keys = _plan((_request(tmp_path),))
    monkeypatch.setattr(directx_runtime, "_worker", lambda: Path("worker.exe"))

    def execute(command, **kwargs):
        assert command[-1] == "--debug-layer"
        assert "--warp-tool-debug" not in command
        Path(command[2]).write_bytes(_response(allocations))
        Path(command[2] + ".runtime").write_bytes(_runtime_packet())
        if receipt != "missing":
            Path(command[2] + ".debug").write_bytes(
                b"broken"
                if receipt == "broken"
                else _messages(
                    severity=1 if receipt == "error" else 2,
                    discarded=int(receipt == "discarded"),
                )
            )
        if failure == "timeout":
            raise RuntimeAdapterSetupError(
                "Worker timed out", details={"reasonKind": "worker-timeout"}
            )
        return subprocess.CompletedProcess(command, int(bool(failure)), b"", b"failed")

    monkeypatch.setattr(directx_runtime, "_run", execute)
    state = SimpleNamespace(details={})
    arguments = (nodes, allocations, keys, state)
    if failure or receipt != "warning":
        with pytest.raises(
            (RuntimeAdapterSetupError, RuntimeAdapterDispatchError)
        ) as error:
            directx_runtime.execute_buffer_views(
                *arguments, device=_device(), debug_layer=True
            )
        assert error.value.details["reasonKind"] == {
            "dispatch": "native-buffer-view-failed",
            "timeout": "worker-timeout",
        }.get(
            failure,
            {
                "error": "debug-layer-error",
                "discarded": "debug-layer-incomplete",
            }.get(receipt, "worker-output-invalid"),
        )
    else:
        directx_runtime.execute_buffer_views(
            *arguments, device=_device(), debug_layer=True
        )
    details = state.details["directxRuntime"]
    assert details["requestedDebugLayer"] is True
    if receipt in {"missing", "broken"}:
        assert "debugLayerError" in details and "debugLayer" not in details
    else:
        assert details["debugLayer"]["enabled"] is True
        assert (
            details["debugLayer"]["messages"][0]["description"] == "Validation detail"
        )


@pytest.mark.parametrize(
    "options,valid",
    [
        (["--debug-layer"], True),
        (["--debug-layer", "--warp-tool-debug"], True),
        (["--warp-tool-debug", "--debug-layer"], True),
        (["--debug-layer", "--debug-layer"], False),
        (["--warp-tool-debug", "--warp-tool-debug"], False),
        (["--debug-layer", "--unknown"], False),
    ],
)
def test_native_worker_debug_options_are_explicit(
    tmp_path, protocol_worker, options, valid
):
    nodes, allocations, keys = _plan((_request(tmp_path),))
    request = tmp_path / "request.bin"
    request.write_bytes(directx_runtime.encode_dispatches(nodes, allocations, keys)[0])
    result = subprocess.run(
        [
            str(protocol_worker),
            str(request),
            "--validate",
            "adapter",
            *(["0"] * 5),
            *options,
        ],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == (0 if valid else 1)
    if not valid:
        assert "invalid pipeline debug option" in result.stderr

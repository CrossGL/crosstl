import importlib.util
import sys
from pathlib import Path

import pytest

from crosstl.project.runtime_verification import RuntimeExecutorAvailability

ROOT = Path(__file__).resolve().parents[1]
SMOKE_PATH = ROOT / "tools" / "directx_runtime_smoke.py"


def _load_smoke_module():
    spec = importlib.util.spec_from_file_location("directx_runtime_smoke", SMOKE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class _ProbeRuntime:
    def __init__(self, availability):
        self.availability = availability

    def is_available(self, adapter, request):
        return self.availability


def test_directx_runtime_smoke_reports_skip_for_missing_adapter(tmp_path, monkeypatch):
    module = _load_smoke_module()
    summary = tmp_path / "summary.md"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))
    runtime = _ProbeRuntime(
        RuntimeExecutorAvailability(
            False,
            reason="No Direct3D 12 compute device is available.",
            details={"reasonKind": "device-unavailable"},
        )
    )

    assert module._probe_runtime(runtime, tmp_path) is False
    assert "Status: skipped" in summary.read_text(encoding="utf-8")


@pytest.mark.parametrize(
    "reason_kind",
    [
        "backend-unavailable",
        "dependency-unavailable",
        "device-selection-failed",
        "direct3d-runtime-unavailable",
        "platform-unavailable",
    ],
)
def test_directx_runtime_smoke_fails_closed_before_adapter_selection(
    tmp_path, reason_kind
):
    module = _load_smoke_module()
    runtime = _ProbeRuntime(
        RuntimeExecutorAvailability(
            False,
            reason="runtime probe failed",
            details={"reasonKind": reason_kind},
        )
    )

    with pytest.raises(RuntimeError, match="runtime probe failed"):
        module._probe_runtime(runtime, tmp_path)


def test_directx_runtime_smoke_translates_compiles_and_builds_dispatch(
    tmp_path, monkeypatch
):
    module = _load_smoke_module()

    def passing_dxc(command, **kwargs):
        _ = kwargs
        output_path = Path(command[command.index("-Fo") + 1])
        output_path.write_bytes(b"DXBC-smoke")
        return type("Result", (), {"returncode": 0, "stderr": "", "stdout": ""})()

    monkeypatch.setattr(module.subprocess, "run", passing_dxc)

    hlsl_path, dxil_path = module._translate_and_compile(tmp_path, "dxc")
    request = module._dispatch_request(hlsl_path, dxil_path)

    assert "RWStructuredBuffer<float> outputBuffer : register(u0);" in (
        hlsl_path.read_text(encoding="utf-8")
    )
    assert request.loaded_artifact == b"DXBC-smoke"
    assert request.entry_point == "CSMain"
    assert request.dispatch.workgroup_count == (4, 1, 1)


def test_directx_runtime_smoke_builds_union_storage_dispatch(tmp_path, monkeypatch):
    module = _load_smoke_module()

    def passing_dxc(command, **kwargs):
        _ = kwargs
        output_path = Path(command[command.index("-Fo") + 1])
        output_path.write_bytes(b"DXBC-union-smoke")
        return type("Result", (), {"returncode": 0, "stderr": "", "stdout": ""})()

    monkeypatch.setattr(module.subprocess, "run", passing_dxc)

    hlsl_path, dxil_path = module._translate_and_compile(
        tmp_path,
        "dxc",
        module.UNION_CASE,
    )
    request = module._dispatch_request(hlsl_path, dxil_path, module.UNION_CASE)
    generated = hlsl_path.read_text(encoding="utf-8")

    assert "uint2 CrossGLUnionStorage;" in generated
    assert "uint2 words;" not in generated
    assert "uint4 bytes[2];" not in generated
    assert request.loaded_artifact == b"DXBC-union-smoke"
    assert request.buffers["output"].dtype == "uint32"
    assert request.buffers["output"].shape == (4,)
    assert request.dispatch.workgroup_count == (1, 1, 1)
    assert list(module.UNION_CASE.expected_values) == [
        1,
        8,
        0x0C0B0A09,
        0x08070605,
    ]


def test_directx_runtime_smoke_fails_closed_for_dxc_errors(tmp_path, monkeypatch):
    module = _load_smoke_module()

    def failing_dxc(command, **kwargs):
        _ = command, kwargs
        return type(
            "Result",
            (),
            {"returncode": 1, "stderr": "invalid shader", "stdout": ""},
        )()

    monkeypatch.setattr(module.subprocess, "run", failing_dxc)

    with pytest.raises(RuntimeError, match="DXC compilation failed: invalid shader"):
        module._translate_and_compile(tmp_path, "dxc")

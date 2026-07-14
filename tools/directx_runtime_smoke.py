"""Run a native Direct3D 12 translation, compilation, and dispatch smoke test."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import crosstl
from crosstl.project.native_runtime_drivers import DirectXComputeRuntime
from crosstl.project.runtime_verification import (
    NativeRuntimeBufferBinding,
    NativeRuntimeDispatchRequest,
    RuntimeArtifactSelector,
    RuntimeDispatchGeometry,
    RuntimeExecutionRequest,
    RuntimeFixture,
    RuntimeResourceBinding,
)

SOURCE = """\
shader RuntimeSmoke {
    RWStructuredBuffer<float> outputBuffer;
    compute {
        layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
        void computeMain(uvec3 dispatchThreadID @ gl_GlobalInvocationID) @ stage_entry {
            uint index = dispatchThreadID.x;
            outputBuffer[index] = float(index) + 1.0;
        }
    }
}
"""
EXPECTED_VALUES = [1.0, 2.0, 3.0, 4.0]


def _write_summary(status: str, detail: str) -> None:
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if not summary_path:
        return
    with open(summary_path, "a", encoding="utf-8") as summary:
        summary.write("## Direct3D 12 Native Runtime Smoke\n")
        summary.write(f"- Status: {status}\n")
        summary.write(f"- Detail: {detail}\n")


def _probe_request(work_dir: Path) -> RuntimeExecutionRequest:
    return RuntimeExecutionRequest(
        fixture=RuntimeFixture(
            id="directx-native-runtime-smoke",
            selector=RuntimeArtifactSelector(target="directx"),
        ),
        artifact={"target": "directx", "path": "runtime-smoke.hlsl"},
        artifact_path=work_dir / "runtime-smoke.hlsl",
        project_root=work_dir,
    )


def _probe_runtime(runtime: DirectXComputeRuntime, work_dir: Path) -> bool:
    availability = runtime.is_available(None, _probe_request(work_dir))
    if availability.available:
        device = availability.details.get("device", "unnamed D3D12 adapter")
        print(f"Direct3D 12 adapter available: {device}")
        return True

    reason_kind = availability.details.get("reasonKind")
    if reason_kind != "device-unavailable":
        raise RuntimeError(
            "Direct3D 12 runtime probe failed before adapter selection: "
            f"{availability.reason or availability.details}"
        )

    detail = availability.reason or "No usable Direct3D 12 adapter exists."
    print(f"DIRECTX_RUNTIME_SMOKE_SKIPPED: {detail}")
    _write_summary("skipped", detail)
    return False


def _translate_and_compile(work_dir: Path, dxc: str) -> tuple[Path, Path]:
    source_path = work_dir / "runtime-smoke.cgl"
    hlsl_path = work_dir / "runtime-smoke.hlsl"
    dxil_path = work_dir / "runtime-smoke.dxil"
    source_path.write_text(SOURCE, encoding="utf-8")

    generated = crosstl.translate(
        str(source_path),
        backend="directx",
        save_shader=str(hlsl_path),
        format_output=False,
    )
    if not generated.strip() or not hlsl_path.is_file():
        raise RuntimeError("CrossTL DirectX translation did not emit HLSL.")

    result = subprocess.run(
        [dxc, "-T", "cs_6_0", "-E", "CSMain", "-Fo", str(dxil_path), str(hlsl_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        diagnostics = (result.stderr or result.stdout or "no DXC diagnostics").strip()
        raise RuntimeError(f"DXC compilation failed: {diagnostics}")
    if not dxil_path.is_file() or dxil_path.stat().st_size == 0:
        raise RuntimeError("DXC reported success without emitting a DXIL artifact.")
    return hlsl_path, dxil_path


def _dispatch_request(hlsl_path: Path, dxil_path: Path) -> NativeRuntimeDispatchRequest:
    return NativeRuntimeDispatchRequest(
        target="directx",
        artifact={"target": "directx", "path": str(hlsl_path)},
        artifact_path=hlsl_path,
        module_path=dxil_path,
        loaded_artifact=dxil_path.read_bytes(),
        buffers={
            "outputBuffer": NativeRuntimeBufferBinding(
                name="outputBuffer",
                binding=RuntimeResourceBinding(
                    name="outputBuffer",
                    kind="buffer",
                    type_name="RWStructuredBuffer<float>",
                    set=0,
                    binding=0,
                    access="read_write",
                ),
                source="expectedOutput",
                dtype="float32",
                shape=(len(EXPECTED_VALUES),),
                metadata={"runtimeValueName": "result"},
            )
        },
        constants={},
        dispatch=RuntimeDispatchGeometry(
            entry_point="CSMain",
            workgroup_count=(len(EXPECTED_VALUES), 1, 1),
        ),
        entry_point="CSMain",
    )


def main() -> int:
    if not sys.platform.startswith("win"):
        raise RuntimeError(
            "The Direct3D 12 native runtime smoke test requires Windows."
        )

    dxc = shutil.which("dxc")
    if dxc is None:
        raise RuntimeError(
            "DXC is required for the Direct3D 12 native runtime smoke test."
        )

    with tempfile.TemporaryDirectory(prefix="crosstl-directx-runtime-") as temp_dir:
        work_dir = Path(temp_dir)
        runtime = DirectXComputeRuntime()
        if not _probe_runtime(runtime, work_dir):
            return 0

        hlsl_path, dxil_path = _translate_and_compile(work_dir, dxc)
        outputs = runtime.dispatch(
            None,
            None,
            _dispatch_request(hlsl_path, dxil_path),
        )

    actual = outputs.get("result", {}).get("values")
    if actual != EXPECTED_VALUES:
        raise RuntimeError(
            "Direct3D 12 native runtime readback mismatch: "
            f"expected {EXPECTED_VALUES}, received {actual}"
        )

    detail = f"translated, compiled, dispatched, and read back {len(actual)} values"
    print(f"Direct3D 12 native runtime smoke passed: {detail}.")
    _write_summary("passed", detail)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

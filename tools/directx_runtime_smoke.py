"""Run a native Direct3D 12 translation, compilation, and dispatch smoke test."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
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

UNION_SOURCE = """\
#include <metal_stdlib>
using namespace metal;

union PackedWords {
    uint2 words;
    uchar4 bytes[2];
};

kernel void union_storage(device uint* output [[buffer(0)]]) {
    PackedWords value;
    value.words = uint2(0x04030201u, 0x08070605u);
    output[0] = value.bytes[0][0];
    output[1] = value.bytes[1][3];
    value.bytes[0] = uchar4(9u, 10u, 11u, 12u);
    output[2] = value.words.x;
    output[3] = value.words.y;
}
"""
UNION_EXPECTED_VALUES = [1, 8, 0x0C0B0A09, 0x08070605]

BOOLEAN_ORDER_SOURCE = """\
shader BooleanOrderRuntime {
    RWStructuredBuffer<uint> output;
    compute {
        layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
        void computeMain() @ stage_entry {
            output[0] = min(false, false) ? 1u : 0u;
            output[1] = min(false, true) ? 1u : 0u;
            output[2] = min(true, false) ? 1u : 0u;
            output[3] = min(true, true) ? 1u : 0u;
            output[4] = max(false, false) ? 1u : 0u;
            output[5] = max(false, true) ? 1u : 0u;
            output[6] = max(true, false) ? 1u : 0u;
            output[7] = max(true, true) ? 1u : 0u;

            bvec4 minimum = min(
                bvec4(false, false, true, true),
                bvec4(false, true, false, true)
            );
            bvec4 maximum = max(
                bvec4(false, false, true, true),
                bvec4(false, true, false, true)
            );
            output[8] = minimum.x ? 1u : 0u;
            output[9] = minimum.y ? 1u : 0u;
            output[10] = minimum.z ? 1u : 0u;
            output[11] = minimum.w ? 1u : 0u;
            output[12] = maximum.x ? 1u : 0u;
            output[13] = maximum.y ? 1u : 0u;
            output[14] = maximum.z ? 1u : 0u;
            output[15] = maximum.w ? 1u : 0u;
        }
    }
}
"""
BOOLEAN_ORDER_EXPECTED_VALUES = [
    0,
    0,
    0,
    1,
    0,
    1,
    1,
    1,
    0,
    0,
    0,
    1,
    0,
    1,
    1,
    1,
]


@dataclass(frozen=True)
class RuntimeSmokeCase:
    id: str
    source_name: str
    source: str
    source_backend: str | None
    buffer_name: str
    buffer_type: str
    dtype: str
    expected_values: tuple[float | int, ...]
    workgroup_count: tuple[int, int, int]


BASIC_CASE = RuntimeSmokeCase(
    id="runtime-smoke",
    source_name="runtime-smoke.cgl",
    source=SOURCE,
    source_backend=None,
    buffer_name="outputBuffer",
    buffer_type="RWStructuredBuffer<float>",
    dtype="float32",
    expected_values=tuple(EXPECTED_VALUES),
    workgroup_count=(len(EXPECTED_VALUES), 1, 1),
)
UNION_CASE = RuntimeSmokeCase(
    id="union-storage-smoke",
    source_name="union-storage-smoke.metal",
    source=UNION_SOURCE,
    source_backend="metal",
    buffer_name="output",
    buffer_type="RWStructuredBuffer<uint>",
    dtype="uint32",
    expected_values=tuple(UNION_EXPECTED_VALUES),
    workgroup_count=(1, 1, 1),
)
BOOLEAN_ORDER_CASE = RuntimeSmokeCase(
    id="boolean-order-smoke",
    source_name="boolean-order-smoke.cgl",
    source=BOOLEAN_ORDER_SOURCE,
    source_backend=None,
    buffer_name="output",
    buffer_type="RWStructuredBuffer<uint>",
    dtype="uint32",
    expected_values=tuple(BOOLEAN_ORDER_EXPECTED_VALUES),
    workgroup_count=(1, 1, 1),
)
SMOKE_CASES = (BASIC_CASE, UNION_CASE, BOOLEAN_ORDER_CASE)


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


def _translate_and_compile(
    work_dir: Path,
    dxc: str,
    case: RuntimeSmokeCase = BASIC_CASE,
) -> tuple[Path, Path]:
    source_path = work_dir / case.source_name
    hlsl_path = work_dir / f"{case.id}.hlsl"
    dxil_path = work_dir / f"{case.id}.dxil"
    source_path.write_text(case.source, encoding="utf-8")

    translate_options = {
        "backend": "directx",
        "save_shader": str(hlsl_path),
        "format_output": False,
    }
    if case.source_backend is not None:
        translate_options["source_backend"] = case.source_backend
    generated = crosstl.translate(str(source_path), **translate_options)
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


def _dispatch_request(
    hlsl_path: Path,
    dxil_path: Path,
    case: RuntimeSmokeCase = BASIC_CASE,
) -> NativeRuntimeDispatchRequest:
    return NativeRuntimeDispatchRequest(
        target="directx",
        artifact={"target": "directx", "path": str(hlsl_path)},
        artifact_path=hlsl_path,
        module_path=dxil_path,
        loaded_artifact=dxil_path.read_bytes(),
        buffers={
            case.buffer_name: NativeRuntimeBufferBinding(
                name=case.buffer_name,
                binding=RuntimeResourceBinding(
                    name=case.buffer_name,
                    kind="buffer",
                    type_name=case.buffer_type,
                    set=0,
                    binding=0,
                    access="read_write",
                ),
                source="expectedOutput",
                dtype=case.dtype,
                shape=(len(case.expected_values),),
                metadata={"runtimeValueName": "result"},
            )
        },
        constants={},
        dispatch=RuntimeDispatchGeometry(
            entry_point="CSMain",
            workgroup_count=case.workgroup_count,
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

        value_count = 0
        for case in SMOKE_CASES:
            hlsl_path, dxil_path = _translate_and_compile(work_dir, dxc, case)
            outputs = runtime.dispatch(
                None,
                None,
                _dispatch_request(hlsl_path, dxil_path, case),
            )
            actual = outputs.get("result", {}).get("values")
            expected = list(case.expected_values)
            if actual != expected:
                raise RuntimeError(
                    f"Direct3D 12 native runtime readback mismatch for {case.id}: "
                    f"expected {expected}, received {actual}"
                )
            value_count += len(actual)

    detail = (
        f"translated, compiled, dispatched, and read back {value_count} values "
        f"across {len(SMOKE_CASES)} cases"
    )
    print(f"Direct3D 12 native runtime smoke passed: {detail}.")
    _write_summary("passed", detail)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

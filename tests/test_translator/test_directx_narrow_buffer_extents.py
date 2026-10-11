"""Direct loads isolate narrow buffer extent handling from subgroup execution."""

import hashlib
import json
import math
import os
import struct
import sys
from pathlib import Path

import pytest

from crosstl.project import build_native_loader_dispatch_request
from crosstl.project.native_runtime_drivers import (
    DirectXComputeRuntime,
    _prepare_directx_buffers,
)
from crosstl.project.runtime_verification import RuntimeAllocationView, RuntimeValue
from tests.ci_helpers import assert_paths_covered
from tests.test_translator import test_directx_buffer_views as buffer_views
from tests.test_translator.test_boolean_buffer_runtime import _bound_values
from tests.test_translator.test_integer_subgroup_shuffles import KINDS
from tests.test_translator.test_loop_updates import _execute
from tests.test_translator.test_metal_builtin_ownership import _compile
from tests.test_translator.test_metal_native_runtime import _native_request
from tests.test_translator.test_native_runtime_drivers import (
    _directx_dispatch_request,
    _FakeCompushady,
    _FakeDirectXCompute,
)
from tests.test_translator.test_software_subgroup_product import _package

MODES = ("exact", "padded-allocation", "extended-view")
REQUIRE_ENV = "CROSTL_REQUIRE_DIRECTX_NARROW_BUFFERS"
GUARD = 0xBAD01234567890AB


def _request(root, kind, count, mode):
    source_type, dtype, _, signed = KINDS[kind]
    source = f"""shader NarrowBufferExtent {{
    StructuredBuffer<{source_type}> inputs @register(t0);
    RWStructuredBuffer<uint64_t> outputs @register(u0);
    compute {{
        @numthreads(7, 5, 1)
        void main(uint invocation @gl_LocalInvocationIndex,
                  uvec3 group @gl_WorkGroupID) {{
            uint index = group.x * 35u + invocation;
            if (index < {count}u) {{
                outputs[index + 4u] = uint64_t(inputs[index]) & 65535ul;
            }}
        }}
    }}
}}"""
    _, descriptor, package = _package(
        root,
        "directx",
        kind,
        (7, 5, 1),
        source=source,
        source_backend="crossgl",
        software_subgroups=False,
    )
    patterns = (0, 1, 0xFFFF, 0x8000, 0x7FFF)
    words = [patterns[index % len(patterns)] for index in range(count)]
    words[-1] = 0x7FFF
    input_words = words + ([0x3553] if mode == "extended-view" else [])
    values = [
        word - 65536 if signed and word >= 32768 else word for word in input_words
    ]
    size = len(input_words) * 2
    allocation_size = ((size + 3) // 4) * 4 + 4 if mode == "padded-allocation" else size
    payload = struct.pack(f"<{len(input_words)}H", *input_words).ljust(
        allocation_size, b"\0"
    )
    inputs = {
        "inputs": RuntimeValue(
            name="inputs",
            dtype=dtype,
            shape=(len(values),),
            values=values,
            allocation=RuntimeAllocationView("input-extent", 0, size, allocation_size),
        ),
        "outputs": {
            "dtype": "uint64",
            "shape": [count + 8],
            "values": [GUARD] * (count + 8),
        },
    }
    expected = _bound_values(
        descriptor,
        {
            "outputs": {
                "dtype": "uint64",
                "shape": [count + 8],
                "values": [GUARD] * 4 + words + [GUARD] * 4,
            }
        },
    )
    request = build_native_loader_dispatch_request(
        descriptor,
        package,
        _bound_values(descriptor, inputs),
        expected,
        {"workgroupCount": [math.ceil(count / 35), 1, 1], "workgroupSize": [7, 5, 1]},
        expected_target="directx",
    )
    assert not request.execution_plan.diagnostics
    return request, expected, payload


def _capture_inputs(monkeypatch, root):
    create = DirectXComputeRuntime._create_buffer_resource
    read_outputs = DirectXComputeRuntime._read_outputs
    records = {}

    def capture(module, device, resource, owned, phase):
        prepared = resource.prepared
        if prepared.namespace != "srv":
            return
        readback = module.Buffer(
            prepared.allocation_size, module.HEAP_READBACK, device=device
        )
        owned.append(readback)
        resource.device_buffer.copy_to(readback)
        payload = bytes(readback.readback(prepared.allocation_size))
        path = root / f"input-{phase}.bin"
        path.write_bytes(payload)
        records[phase] = {
            "file": path.name,
            "sha256": hashlib.sha256(payload).hexdigest(),
            "sizeBytes": len(payload),
            "viewBytes": prepared.size,
            "stride": prepared.stride,
        }
        (root / "input-readbacks.json").write_text(json.dumps(records, indent=2))

    def create_and_capture(self, module, device, prepared, owned):
        resource = create(self, module, device, prepared, owned)
        capture(module, device, resource, owned, "before")
        return resource

    def read_and_capture(self, module, device, resources, owned):
        for resource in resources:
            capture(module, device, resource, owned, "after")
        return read_outputs(self, module, device, resources, owned)

    monkeypatch.setattr(
        DirectXComputeRuntime, "_create_buffer_resource", create_and_capture
    )
    monkeypatch.setattr(DirectXComputeRuntime, "_read_outputs", read_and_capture)
    return records


@pytest.mark.parametrize("kind", ["short", "ushort"])
@pytest.mark.parametrize("count", [105, 106])
@pytest.mark.parametrize("mode", MODES)
def test_narrow_buffer_extent_package_and_compile(tmp_path, kind, count, mode):
    request, expected, payload = _request(tmp_path, kind, count, mode)
    _, native = _native_request(request)
    prepared = _prepare_directx_buffers(native.buffers)
    (input_buffer,) = (item for item in prepared if item.namespace == "srv")
    logical_count = count + (mode == "extended-view")
    assert input_buffer.stride == 2
    assert input_buffer.size == logical_count * 2
    assert input_buffer.allocation_size == len(payload)
    assert input_buffer.payload == payload[: input_buffer.size]
    assert input_buffer.byte_offset == 0
    assert next(iter(expected.values()))["values"][count + 3] == 0x7FFF
    generated = request.artifact_path.read_text()
    assert "Wave" not in generated and "groupshared" not in generated
    _compile(generated, "directx", tmp_path)


@pytest.mark.parametrize("mutate_device_input", [False, True])
def test_input_capture_reads_device_storage_without_modifying_bindings(
    tmp_path, monkeypatch, mutate_device_input
):
    if mutate_device_input:
        dispatch = _FakeDirectXCompute.dispatch

        def mutate_after_dispatch(self, *args):
            dispatch(self, *args)
            self.srv[0].payload[:] = struct.pack("<2f", -7, 8)

        monkeypatch.setattr(_FakeDirectXCompute, "dispatch", mutate_after_dispatch)
    records = _capture_inputs(monkeypatch, tmp_path)
    module = _FakeCompushady()
    runtime = DirectXComputeRuntime(
        module_loader=lambda name: module, platform_name="win32"
    )
    result = runtime.dispatch(None, None, _directx_dispatch_request(tmp_path))
    assert result["result"]["values"] == [3.0, 6.0]
    assert set(records) == {"before", "after"}
    for phase, record in records.items():
        values = (-7, 8) if phase == "after" and mutate_device_input else (1, 2)
        assert (tmp_path / record["file"]).read_bytes() == struct.pack("<2f", *values)
        assert record["viewBytes"] == 8 and record["stride"] == 4
    assert all(buffer.released for buffer in module.buffers)


@pytest.fixture
def retained_packets(tmp_path, monkeypatch):
    buffer_views.retain_native_packets.__wrapped__(tmp_path, monkeypatch)


@pytest.mark.parametrize("kind", ["short", "ushort"])
@pytest.mark.parametrize("count", [105, 106])
@pytest.mark.parametrize("mode", MODES)
def test_narrow_buffer_extent_execute(
    tmp_path, monkeypatch, retained_packets, kind, count, mode
):
    if sys.platform != "win32" or os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(
            f"set {REQUIRE_ENV}=1 on Windows for native narrow-buffer execution"
        )
    request, expected, payload = _request(tmp_path, kind, count, mode)
    (tmp_path / "input-expected.bin").write_bytes(payload)
    records = _capture_inputs(monkeypatch, tmp_path)
    _execute(request, expected, tmp_path)
    if mode == "padded-allocation":
        evidence = json.loads((tmp_path / "evidence.json").read_text())
        native = evidence["records"]["generated"]["details"]["directxRuntime"]
        assert native["runtime"] == "native-buffer-views"
        (input_allocation,) = (
            item
            for item in native["allocations"]
            if item["allocationId"] == "input-extent"
        )
        assert input_allocation["readbackSHA256"] == hashlib.sha256(payload).hexdigest()
        assert (tmp_path / "native-readbacks.bin").is_file()
        assert not records
    else:
        assert set(records) == {"before", "after"}
        for record in records.values():
            assert (tmp_path / record["file"]).read_bytes() == payload
            assert record["sha256"] == hashlib.sha256(payload).hexdigest()


def test_narrow_extent_controls_share_existing_native_gate():
    from tools import ci_coverage

    path = "tests/test_translator/test_directx_narrow_buffer_extents.py"
    root = Path(__file__).resolve().parents[2]
    workflow = (root / ".github/workflows/demo-project-testing.yml").read_text()
    step = ci_coverage.workflow_step_section(
        workflow, "Validate collective helper arguments"
    )
    assert f"{path}::test_narrow_buffer_extent_execute" in step
    assert f'{REQUIRE_ENV}: "1"' in step
    assert "--timeout-seconds 360" in step and "-n auto" in step
    assert "if:" not in step and "continue-on-error" not in step
    assert step.index(path) < step.index("test_subgroup_uniform_arguments.py")
    assert "--basetemp=" in step and "--junitxml=" in step
    for event in ("pull_request", "push"):
        assert_paths_covered(
            ci_coverage.workflow_event_path_filters(workflow, event), path
        )

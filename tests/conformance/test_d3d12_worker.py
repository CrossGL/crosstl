"""The D3D12 adapter's requests to CrossTL's buffer-view worker, checked against the worker's own parser.

The worker binds each view at the register the artifact declares with the descriptor HLSL requires: a raw view
(``R32_TYPELESS`` with the raw flag) for a ``ByteAddressBuffer``, a structured view of its stride for a
``StructuredBuffer`` and a constant-buffer view for a ``cbuffer``. Binding a byte-address buffer through a view
without the raw flag is invalid, and WARP then removes the device when the pipeline is created.
"""

import shutil
import struct

import pytest

from . import native
from .process import run
from .runtimes import REPOSITORY

SHADER = b"DXBC" + bytes(60)


def _job(tmp_path, resources, constants=None, input_bytes=512):
    (tmp_path / "input.bin").write_bytes(bytes(range(256)) * (input_bytes // 256))
    (tmp_path / "output.bin").write_bytes(b"\xef\xbe\xad\xde" * 40)
    job = {
        "directory": str(tmp_path),
        "threads": 100,
        "group": [64, 1, 1],
        "runtime": "d3d12-warp",
        "resources": resources,
    }
    if constants is not None:
        (tmp_path / native.CONSTANTS).write_bytes(constants)
        job["constants"] = {"slot": 2, "kind": "uniform"}
    return job


def _decode(payload):
    """Allocation sizes, dispatch groups and views of a request, parsed as the worker parses it."""
    magic, count = struct.unpack_from("<II", payload)
    assert magic == native.D3D12_REQUEST
    offset, sizes = 8, []
    for _ in range(count):
        (size,) = struct.unpack_from("<Q", payload, offset)
        sizes.append(size)
        offset += 8 + size
    dispatches, length = struct.unpack_from("<IQ", payload, offset)
    assert dispatches == 1
    offset += 12
    assert payload[offset : offset + length] == SHADER
    offset += length
    *groups, views = struct.unpack_from("<4I", payload, offset)
    offset += 16
    decoded = [
        struct.unpack_from("<4I2Q", payload, offset + 32 * index)
        for index in range(views)
    ]
    assert offset + 32 * views == len(payload)
    return sizes, groups, decoded


def test_byte_address_buffers_get_raw_views(tmp_path):
    job = _job(
        tmp_path,
        [
            {"register": "u0", "kind": "raw", "stride": 0},
            {"register": "u1", "kind": "raw", "stride": 0},
        ],
    )
    payload, files, sizes = native.d3d12_request(job, SHADER)
    assert files == ["input.bin", "output.bin"] and sizes == [512, 160]
    assert _decode(payload) == (
        [512, 160],
        [2, 1, 1],
        [
            (native.D3D12_UAV | native.D3D12_RAW, 0, 0, 0, 0, 512),
            (native.D3D12_UAV | native.D3D12_RAW, 1, 1, 0, 0, 160),
        ],
    )


def test_structured_buffers_and_constants_keep_their_registers(tmp_path):
    job = _job(
        tmp_path,
        [
            {"register": "t0", "kind": "structured", "stride": 12},
            {"register": "u1", "kind": "structured", "stride": 4},
        ],
        constants=bytes(20),
    )
    payload, files, sizes = native.d3d12_request(job, SHADER)
    # A structured view covers whole elements, as a view of size // stride elements does; the CBV is 256-aligned.
    assert sizes == [512, 160, 256]
    assert _decode(payload)[2] == [
        (native.D3D12_SRV, 0, 0, 12, 0, 504),
        (native.D3D12_UAV, 1, 1, 4, 0, 160),
        (native.D3D12_CBV, 2, 2, 0, 0, 256),
    ]
    assert files == ["input.bin", "output.bin"]


def _inputs(tmp_path, count):
    """Write the extra input files of a ``count``-input job, each its own size, and return every input's size."""
    sizes = [512]
    for slot in range(1, count):
        (tmp_path / f"input{slot}.bin").write_bytes(bytes([slot]) * 64 * slot)
        sizes.append(64 * slot)
    return sizes


@pytest.mark.parametrize("count", [2, 3])
def test_every_input_buffer_gets_its_own_allocation_and_register(tmp_path, count):
    inputs = [
        {"register": "u0", "kind": "structured", "stride": 4},
        {"register": "t1", "kind": "structured", "stride": 4},
        {"register": "u2", "kind": "raw", "stride": 0},
    ][:count]
    job = _job(
        tmp_path,
        [*inputs, {"register": f"u{count}", "kind": "structured", "stride": 4}],
        constants=bytes(20),
    )
    job["constants"]["slot"] = 0
    expected = _inputs(tmp_path, count) + [160]
    payload, files, sizes = native.d3d12_request(job, SHADER)
    assert files == list(native.buffer_files(count))
    assert sizes == [*expected, 256]
    kinds = [native.D3D12_UAV, native.D3D12_SRV, native.D3D12_UAV | native.D3D12_RAW]
    strides = [4, 4, 0]
    assert _decode(payload)[2] == [
        *(
            (kinds[slot], slot, slot, strides[slot], 0, expected[slot])
            for slot in range(count)
        ),
        (native.D3D12_UAV, count, count, 4, 0, 160),
        (native.D3D12_CBV, 0, count + 1, 0, 0, 256),
    ]


def test_a_resource_without_a_buffer_file_is_refused(tmp_path):
    job = _job(
        tmp_path,
        [
            {"register": "u0", "kind": "raw", "stride": 0},
            {"register": "u1", "kind": "raw", "stride": 0},
            {"register": "u2", "kind": "raw", "stride": 0},
        ],
    )
    with pytest.raises(FileNotFoundError, match="input1.bin"):
        native.d3d12_request(job, SHADER)


def test_readbacks_and_runtime_identity_decode():
    payload = (
        struct.pack("<II", native.D3D12_READBACKS, 2)
        + struct.pack("<QQ", 1, 4)
        + b"abcd"
        + struct.pack("<QQ", 2, 2)
        + b"ef"
    )
    assert native.d3d12_readbacks(payload, [4, 2]) == [b"abcd", b"ef"]
    with pytest.raises(RuntimeError, match="incompatible"):
        native.d3d12_readbacks(payload, [4])
    with pytest.raises(RuntimeError, match="truncated"):
        native.d3d12_readbacks(payload, [4, 3])
    warp = "C:\\runtime\\d3d10warp.dll"
    identity = struct.pack("<III", native.D3D12_IDENTITY, 7, 3)
    identity += struct.pack("<III", 1, 0, 3) + "a.b".encode("utf-16-le")
    identity += struct.pack("<III", 0, 0, 0)
    identity += struct.pack("<III", 1, 0, len(warp)) + warp.encode("utf-16-le")
    assert native.d3d12_loaded_libraries(identity) == {
        "d3d12.dll": "a.b",
        "D3D12Core.dll": None,
        "d3d10warp.dll": warp,
    }


@pytest.mark.parametrize(
    ("kind", "count"), [("raw", 1), ("structured", 1), ("structured", 3)]
)
def test_the_worker_accepts_the_request(tmp_path, kind, count):
    """The worker's request parser is portable C++; its ``--validate`` mode runs on any host."""
    compiler = shutil.which("c++") or shutil.which("clang++") or shutil.which("g++")
    if compiler is None:
        pytest.skip("no C++ compiler builds the worker's request parser")
    worker = tmp_path / "worker"
    built = run(
        [
            compiler,
            "-std=c++17",
            REPOSITORY / "crosstl/project/directx_runtime_worker.cpp",
            "-o",
            worker,
        ],
        timeout=180,
    )
    assert built.status == "accepted", built.log
    stride = {"raw": 0, "structured": 4}[kind]
    inputs = [
        {"register": f"t{slot}", "kind": kind, "stride": stride}
        for slot in range(count)
    ]
    job = _job(
        tmp_path,
        [*inputs, {"register": f"u{count}", "kind": kind, "stride": stride}],
        constants=bytes(48),
    )
    job["constants"]["slot"] = 0
    _inputs(tmp_path, count)
    request = tmp_path / "request.bin"
    request.write_bytes(native.d3d12_request(job, SHADER)[0])
    checked = run([worker, request, "--validate"])
    assert checked.status == "accepted", checked.log
    assert checked.log.strip() == f"{count + 2} allocations, 1 dispatches"

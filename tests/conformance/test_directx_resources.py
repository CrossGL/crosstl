"""The resources besides its buffers that the D3D12 harness binds for an HLSL kernel (:mod:`.directx_resources`).

The interface reads them from the kernel, the request encodes them as the worker's extended views, and the worker's
portable request parser (``--validate``) accepts every request the harness writes and refuses malformed views.
"""

import shutil
import struct

import pytest

from . import directx_resources, native
from .interface import InterfaceChanged, interface
from .process import run
from .runtimes import REPOSITORY

SHADER = b"DXBC" + bytes(60)
_BUFFERS = (
    "RWStructuredBuffer<float> fin : register(u0);\n"
    "RWStructuredBuffer<float4> outp : register(u1);\n"
)
_ENTRY = (
    "[numthreads(1, 1, 1)]\nvoid main(uint3 id : SV_DispatchThreadID) {{\n    {}\n}}\n"
)


def _kernel(declarations, body="outp[id.x] = 0.0;"):
    return _BUFFERS + declarations + "\n" + _ENTRY.format(body)


def _interface(tmp_path, text, name="a.hlsl"):
    path = tmp_path / name
    path.write_text(text)
    return interface(path, "hlsl", 1)


def test_the_interface_reads_textures_samplers_and_other_buffers(tmp_path):
    found = _interface(
        tmp_path,
        _kernel(
            "Texture2D<float4> tex : register(t1);\n"
            "SamplerState smp : register(s0);\n"
            "Buffer<uint4> typed : register(t3);\n"
            "RWByteAddressBuffer raw : register(u8);\n"
            "AppendStructuredBuffer<float> queue : register(u9);\n"
            "RWTexture2D<float> img[2] : register(u3);\n"
        ),
    )
    assert found["bindings"] == (0, 1)
    assert [resource["name"] for resource in found["resources"]] == ["fin", "outp"]
    extras = {(record["name"], record["element"]): record for record in found["extras"]}
    assert [record["register"] for record in found["extras"]] == [
        "s0",
        "t1",
        "t3",
        "u3",
        "u4",
        "u8",
        "u9",
    ]
    # Each register class numbers its resources in register order, counting array elements.
    assert [record["ordinal"] for record in found["extras"]] == [0, 0, 1, 0, 1, 2, 3]
    texture = extras["tex", 0]
    assert (texture["kind"], texture["format"], texture["texel"]) == ("texture", 2, 16)
    assert (texture["dimension"], texture["extent"]) == (3, (4, 4, 1))
    assert extras["typed", 0]["kind"] == "typed" and extras["typed", 0]["format"] == 3
    assert extras["raw", 0]["kind"] == "raw"
    assert extras["queue", 0]["counter"] == "append"
    assert extras["queue", 0]["stride"] == 4
    assert extras["img", 1]["format"] == 41 and extras["img", 1]["register"] == "u4"
    assert extras["smp", 0] == dict(extras["smp", 0], kind="sampler", comparison=0)


def test_a_kernel_without_other_resources_keeps_its_interface(tmp_path):
    found = _interface(tmp_path, _kernel(""))
    assert "extras" not in found
    assert found["resources"][1] == {
        "register": "u1",
        "name": "outp",
        "kind": "structured",
        "stride": 16,
    }


def test_a_translation_that_renumbers_registers_reads_the_same_contents(tmp_path):
    """A probe at t1 and u8 and its translation at t0 and u2 get the same texels and words."""
    native_records = _interface(
        tmp_path,
        _kernel(
            "Texture2DArray<float4> tex : register(t1);\n"
            "SamplerState smp : register(s0);\n"
            "RWByteAddressBuffer obj : register(u8);\n"
        ),
        "native.hlsl",
    )["extras"]
    translated = _interface(
        tmp_path,
        _kernel(
            "Texture2DArray tex : register(t0);\n"
            "SamplerState smp : register(s0);\n"
            "RWByteAddressBuffer obj : register(u2);\n"
        ),
        "translated.hlsl",
    )["extras"]
    assert [
        directx_resources.contents(record)
        for record in native_records
        if record["kind"] != "sampler"
    ] == [
        directx_resources.contents(record)
        for record in translated
        if record["kind"] != "sampler"
    ]


def test_contents_fill_every_texel_and_differ_between_resources(tmp_path):
    found = _interface(
        tmp_path,
        _kernel(
            "TextureCubeArray<float4> cubes : register(t0);\n"
            "Texture3D<uint2> volume : register(t1);\n"
            "RWTexture1D<unorm float4> line : register(u2);\n"
            "ConsumeStructuredBuffer<float2> queue : register(u3);\n"
        ),
    )["extras"]
    cubes, volume, line, queue = found
    assert len(directx_resources.contents(cubes)) == 16 * 4 * 4 * 12
    assert len(directx_resources.contents(volume)) == 8 * 4 * 4 * 4
    assert len(directx_resources.contents(line)) == 4 * 64
    words = struct.unpack("<128f", directx_resources.contents(queue))
    # Every element of a consume buffer is the same, so the order threads consume in does not matter.
    assert words[0::2] == (words[0],) * 64 and words[1::2] == (words[1],) * 64
    texels = struct.unpack("<768f", directx_resources.contents(cubes))
    assert all(texels) and len(set(texels)) > 200
    assert volume["ordinal"] == 1
    assert directx_resources.contents(volume) != directx_resources.contents(
        dict(volume, ordinal=0)
    )


def test_a_comparison_sampler_makes_float_textures_single_channel(tmp_path):
    found = _interface(
        tmp_path,
        _kernel(
            "Texture2D<float4> tex : register(t0);\n"
            "SamplerComparisonState cmp : register(s1);\n"
        ),
    )["extras"]
    sampler, texture = found
    assert sampler["comparison"] == 1
    assert (texture["format"], texture["texel"], texture["components"]) == (41, 4, 1)


@pytest.mark.parametrize(
    ("declarations", "body", "message"),
    [
        ("Texture2DMS<float4> obj : register(t0);", "", "binds no Texture2DMS"),
        ("Texture2D<float4> tex[] : register(t0);", "", "unsized array"),
        ("SamplerState smp[2] : register(s0);", "", "sampler array"),
        ("Texture2D<half4> tex : register(t0);", "", "no format"),
        ("RWTexture2D<float3> img : register(u2);", "", "no format"),
        ("Texture2D<float4> tex : register(u2);", "", "Register class u"),
        (
            "RWStructuredBuffer<float> obj : register(u2);",
            "outp[id.x] = obj.IncrementCounter();",
            "counter values of obj",
        ),
        (
            "RWByteAddressBuffer obj : register(u2);",
            "uint r; obj.InterlockedAdd(0, 1, r); outp[id.x] = r;",
            "InterlockedAdd returns the original value of obj",
        ),
        (
            "RWTexture2D<uint> img : register(u2);",
            "uint r; InterlockedMax(img[uint2(0, 0)], 1, r); outp[id.x] = r;",
            "InterlockedMax returns the original value of img",
        ),
    ],
)
def test_resources_without_repeatable_contents_are_refused(
    tmp_path, declarations, body, message
):
    with pytest.raises(InterfaceChanged, match=message):
        _interface(tmp_path, _kernel(declarations, body or "outp[id.x] = 0.0;"))


def test_interlocked_operations_without_an_original_value_run(tmp_path):
    found = _interface(
        tmp_path,
        _kernel(
            "RWByteAddressBuffer obj : register(u2);",
            "obj.InterlockedAdd(0, 1); uint r; InterlockedAdd(outp[0].x, 1, r);",
        ),
    )
    assert [record["name"] for record in found["extras"]] == ["obj"]


_ALL = (
    "Texture1D<float4> a : register(t0);\n"
    "Texture1DArray<float4> b : register(t1);\n"
    "Texture2D<float4> c[2] : register(t2);\n"
    "Texture2DArray<int4> d : register(t4);\n"
    "Texture3D<float> e : register(t5);\n"
    "TextureCube<float4> f : register(t6);\n"
    "TextureCubeArray<float4> g : register(t7);\n"
    "Buffer<float4> h : register(t8);\n"
    "StructuredBuffer<float> i : register(t9);\n"
    "ByteAddressBuffer j : register(t10);\n"
    "RWTexture1DArray<uint> k : register(u2);\n"
    "RWTexture3D<float4> l : register(u3);\n"
    "RWBuffer<int2> m : register(u4);\n"
    "RWStructuredBuffer<float4> n : register(u5);\n"
    "AppendStructuredBuffer<float> o : register(u6);\n"
    "ConsumeStructuredBuffer<uint> p : register(u7);\n"
    "RWByteAddressBuffer q : register(u8);\n"
    "SamplerState r : register(s0);\n"
    "SamplerComparisonState s : register(s1);\n"
)


def _worker(tmp_path):
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
    return worker


def _job(tmp_path, found):
    (tmp_path / "input.bin").write_bytes(bytes(256))
    (tmp_path / "output.bin").write_bytes(bytes(1024))
    return {
        "directory": str(tmp_path),
        "threads": 64,
        "group": [1, 1, 1],
        "runtime": "d3d12-warp",
        **found,
    }


def test_the_worker_accepts_every_resource_the_harness_binds(tmp_path):
    worker = _worker(tmp_path)
    found = _interface(tmp_path, _kernel(_ALL))
    payload, files, sizes = native.d3d12_request(_job(tmp_path, found), SHADER)
    assert files == ["input.bin", "output.bin"]
    # The two buffers, then one allocation per resource that is not a sampler, and a counter after each of the
    # three structured buffers that have one.
    assert len(sizes) == 2 + 18 + 3
    request = tmp_path / "request.bin"
    request.write_bytes(payload)
    checked = run([worker, request, "--validate"])
    assert checked.status == "accepted", checked.log
    assert checked.log.strip() == f"{len(sizes)} allocations, 1 dispatches"


def _request(allocations, views):
    parts = [struct.pack("<II", native.D3D12_REQUEST, len(allocations))]
    for allocation in allocations:
        parts += [struct.pack("<Q", len(allocation)), allocation]
    parts += [
        struct.pack("<IQ", 1, len(SHADER)),
        SHADER,
        struct.pack("<4I", 1, 1, 1, len(views)),
    ]
    return b"".join(parts + views)


def _view(kind, slot, allocation, stride, length, *trailer):
    view = struct.pack("<4I2Q", kind, slot, allocation, stride, 0, length)
    return (
        view + struct.pack("<8I", *trailer, *(0,) * (8 - len(trailer)))
        if kind & 8
        else view
    )


@pytest.mark.parametrize(
    ("views", "message"),
    [
        # A cube texture has no unordered-access view.
        ([_view(2 | 8, 0, 0, 0, 16 * 96, 2, 16, 6, 4, 4, 6)], "invalid texture view"),
        ([_view(1 | 8, 0, 0, 0, 16 * 15, 2, 16, 3, 4, 4, 1)], "length does not match"),
        ([_view(1 | 8, 0, 0, 0, 16 * 16, 2, 16, 3, 4, 4, 2)], "invalid texture view"),
        ([_view(3 | 8, 0, 1, 0, 0, 0)], "invalid sampler"),
        ([_view(3, 0, 0, 0, 0)], "invalid sampler"),
        ([_view(2 | 4 | 8, 0, 0, 0, 256, 0, 0, 0, 0, 0, 0, 2)], "invalid UAV counter"),
        ([_view(1 | 8, 0, 0, 0, 256, 2, 12)], "invalid typed-buffer view"),
        ([_view(1 | 16, 1, 0, 4, 256)], "continued descriptor range"),
        (
            [_view(1 | 8, 0, 0, 0, 256, 0, 0, 0, 0, 0, 0, 0, 7)],
            "invalid view extension",
        ),
    ],
)
def test_the_worker_refuses_malformed_resource_views(tmp_path, views, message):
    worker = _worker(tmp_path)
    request = tmp_path / "request.bin"
    request.write_bytes(_request([bytes(4096), bytes(16)], views))
    checked = run([worker, request, "--validate"])
    assert checked.status != "accepted"
    assert message in checked.log

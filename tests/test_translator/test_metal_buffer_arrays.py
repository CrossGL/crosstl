"""Arrays of buffers reach a Metal entry point as one ``[[buffer]]`` argument per element.

A Metal entry point rejects ``array<device T*, N> b [[buffer(K)]]``; the generator declares ``b_0`` .. ``b_{N-1}``
at slots ``K`` .. ``K + N - 1`` and rebuilds ``b`` as a local array, so HLSL register ranges and GLSL binding
ranges keep their consecutive slots.
"""

import json
import os
import re
import shutil
import struct
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

import crosstl

ROOT = Path(__file__).resolve().parents[2]
REQUIRE_ENV = "CROSSTL_REQUIRE_BUFFER_ARRAY_EXECUTION"

HLSL_ARRAYS = """
struct Item { float4 v; uint id; };
StructuredBuffer<Item> inputs[3] : register(t0);
RWStructuredBuffer<uint> totals[2] : register(u0);
AppendStructuredBuffer<uint> queues[2] : register(u2);
RWStructuredBuffer<float4> results : register(u4);
cbuffer Params : register(b0) { uint which; };

float4 pick(uint k, uint i) { return inputs[k][i].v; }

[numthreads(64, 1, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
    uint count, stride;
    inputs[which].GetDimensions(count, stride);
    if (id.x >= count)
        return;
    InterlockedAdd(totals[1][0], 1u);
    queues[id.x & 1].Append(inputs[0][id.x].id);
    results[id.x] = pick(id.x % 3, id.x) + inputs[which][id.x].v;
}
"""

GLSL_BLOCK_ARRAYS = """
#version 450
layout(local_size_x = 32) in;
layout(std430, binding = 0) readonly buffer Source { float data[]; } sources[3];
layout(std430, binding = 3) buffer Destination { float values[]; } destinations[2];

float fetch(uint k, uint i) { return sources[k].data[i]; }

void main()
{
    uint i = gl_GlobalInvocationID.x;
    if (i < uint(sources[1].data.length()))
        destinations[i & 1u].values[i] = fetch(0u, i) + sources[2].data[i];
}
"""

# HLSL packs Entry into 16 bytes (D3D11.3 section 5.3.7); Metal's natural layout aligns the uint3 to 16 and takes 32.
RUNTIME_HLSL = """
struct Entry { uint value; uint3 tag; };
StructuredBuffer<Entry> sources[3] : register(t0);
RWStructuredBuffer<uint> results : register(u0);

[numthreads(12, 1, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
    uint k = id.x % 3;
    uint count, stride;
    sources[k].GetDimensions(count, stride);
    uint j = id.x / 3;
    results[id.x] = j < count
        ? sources[k][j].value + sources[k][j].tag.z + 1000u * count + 100000u * stride
        : 0xFFFFFFFFu;
}
"""


def translate(tmp_path, source, suffix, backend="metal"):
    path = tmp_path / f"arrays.{suffix}"
    path.write_text(textwrap.dedent(source).strip() + "\n", encoding="utf-8")
    return crosstl.translate(str(path), backend=backend, format_output=False)


def metal_compiler():
    xcrun = shutil.which("xcrun")
    if xcrun is None:
        return None
    probe = subprocess.run(
        [xcrun, "-sdk", "macosx", "-f", "metal"], capture_output=True, text=True
    )
    return xcrun if probe.returncode == 0 else None


def run(command):
    completed = subprocess.run(
        [str(item) for item in command], capture_output=True, text=True, timeout=180
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


def compile_metal(source, tmp_path):
    """Compile ``source``; returns the library, or ``None`` without a Metal toolchain."""
    xcrun = metal_compiler()
    if xcrun is None:
        return None
    shader = tmp_path / "arrays.metal"
    shader.write_text(source, encoding="utf-8")
    air, library = tmp_path / "arrays.air", tmp_path / "arrays.metallib"
    run([xcrun, "-sdk", "macosx", "metal", "-Werror", "-c", shader, "-o", air])
    run([xcrun, "-sdk", "macosx", "metallib", air, "-o", library])
    return library


def kernel_slots(metal):
    kernel = re.search(r"kernel void (\w+)\((.*?)\) \{", metal, re.S)
    assert kernel is not None, metal
    slots = {
        name: int(index)
        for name, index in re.findall(r"(\w+) \[\[buffer\((\d+)\)\]\]", kernel.group(2))
    }
    return kernel.group(1), slots


def test_hlsl_structured_buffer_arrays_bind_one_slot_per_element(tmp_path):
    metal = translate(tmp_path, HLSL_ARRAYS, "hlsl")
    assert not re.search(r"array<[^>]*> \w+ \[\[buffer", metal), metal
    _, slots = kernel_slots(metal)
    inputs = [slots[f"inputs_{k}"] for k in range(3)]
    totals = [slots[f"totals_{k}"] for k in range(2)]
    queues = [slots[f"queues_{k}"] for k in range(2)]
    lengths = [slots[f"inputsLength_{k}"] for k in range(3)]
    counters = [slots[f"queuesCounter_{k}"] for k in range(2)]
    for group in (inputs, totals, queues, lengths, counters):
        assert group == list(range(group[0], group[0] + len(group))), slots
    assert len(set(slots.values())) == len(slots), slots
    assert (
        "array<const device Item*, 3> inputs = {inputs_0, inputs_1, inputs_2};" in metal
    )
    assert "array<device uint*, 2> totals = {totals_0, totals_1};" in metal
    assert (
        "array<constant uint*, 3> inputsLength = "
        "{inputsLength_0, inputsLength_1, inputsLength_2};"
    ) in metal
    assert (
        "array<device atomic_uint*, 2> queuesCounter = "
        "{queuesCounter_0, queuesCounter_1};"
    ) in metal
    assert "pick(id.x % 3, id.x, inputs, inputsLength)" in metal
    # GetDimensions(count, stride) reports the declared element stride.
    assert "stride = uint(sizeof(inputs[params.which][0]))" in metal
    compile_metal(metal, tmp_path)


def test_glsl_storage_block_arrays_bind_one_slot_per_element(tmp_path):
    metal = translate(tmp_path, GLSL_BLOCK_ARRAYS, "comp")
    assert not re.search(r"array<[^>]*> \w+ \[\[buffer", metal), metal
    _, slots = kernel_slots(metal)
    assert [slots[f"sources_{k}"] for k in range(3)] == [0, 1, 2], slots
    assert [slots[f"destinations_{k}"] for k in range(2)] == [3, 4], slots
    assert (
        "array<const device float*, 3> sources = {sources_0, sources_1, sources_2};"
        in metal
    )
    assert (
        "array<device float*, 2> destinations = {destinations_0, destinations_1};"
        in metal
    )
    compile_metal(metal, tmp_path)


def test_generated_element_names_avoid_program_identifiers(tmp_path):
    metal = translate(
        tmp_path,
        """
        StructuredBuffer<float> data[2] : register(t0);
        RWStructuredBuffer<float> results : register(u0);
        [numthreads(1, 1, 1)]
        void main(uint3 id : SV_DispatchThreadID)
        {
            float data_0 = data[0][id.x];
            results[id.x] = data_0 + data[1][id.x];
        }
        """,
        "hlsl",
    )
    _, slots = kernel_slots(metal)
    assert "data_0" not in slots, slots
    assert [slots["data_0_"], slots["data_1"]] == [0, 1], slots
    assert "array<const device float*, 2> data = {data_0_, data_1};" in metal
    compile_metal(metal, tmp_path)


def test_unsized_buffer_array_binds_the_extent_its_constant_indices_need(tmp_path):
    metal = translate(
        tmp_path,
        """
        #version 450
        layout(local_size_x = 1) in;
        layout(std430, binding = 2) buffer Block { uint count; float values[]; } blocks[];
        void main()
        {
            blocks[2].values[0] = float(blocks[1].count);
        }
        """,
        "comp",
    )
    _, slots = kernel_slots(metal)
    assert [slots[f"blocks_{k}"] for k in range(3)] == [2, 3, 4], slots
    assert "array<device uchar*, 3> blocks = {blocks_0, blocks_1, blocks_2};" in metal
    compile_metal(metal, tmp_path)


def test_buffer_arrays_execute_on_metal(tmp_path):
    if os.environ.get(REQUIRE_ENV) != "1" or sys.platform != "darwin":
        pytest.skip(f"set {REQUIRE_ENV}=1 on macOS to require the Metal dispatch")
    metal = translate(tmp_path, RUNTIME_HLSL, "hlsl")
    library = compile_metal(metal, tmp_path)
    assert library is not None, "the Metal toolchain is required"
    entry, slots = kernel_slots(metal)
    sources = [[10 * k + j for j in range(k + 2)] for k in range(3)]
    payloads = {"results": struct.pack("<I", 0xDEADBEEF) * 12}
    for k, values in enumerate(sources):
        words = [word for value in values for word in (value, 0, 0, 7 * value)]
        payloads[f"sources_{k}"] = struct.pack(f"<{len(words)}I", *words)
        payloads[f"sourcesLength_{k}"] = struct.pack("<I", len(values))
    assert sorted(slots) == sorted(payloads), slots
    assert sorted(slots.values()) == list(range(len(slots))), slots
    files = []
    for name, index in sorted(slots.items(), key=lambda item: item[1]):
        path = tmp_path / f"buffer-{index}.in"
        path.write_bytes(payloads[name])
        files.append(str(path))
    runner = tmp_path / "runner"
    run(
        [
            "swiftc",
            ROOT / "tests/fixtures/runtime_verification/metal_raw_buffers.swift",
            "-o",
            runner,
        ]
    )
    request = tmp_path / "request.json"
    request.write_text(
        json.dumps(
            {
                "buffers": files,
                "workgroupCount": [1, 1, 1],
                "workgroupSize": [12, 1, 1],
                "simdWidth": 32,
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "output"
    run([runner, library, entry, request, output])
    data = (output / f"buffer-{slots['results']}.bin").read_bytes()
    expected = []
    for index in range(12):
        k, j = index % 3, index // 3
        values = sources[k]
        expected.append(
            8 * values[j] + 1000 * len(values) + 100000 * 16
            if j < len(values)
            else 0xFFFFFFFF
        )
    assert list(struct.unpack("<12I", data)) == expected

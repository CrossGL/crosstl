"""Ordered mixed-type inputs, including readback of every input and the output guard."""

import json
import sys
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from . import native, runner, runtimes
from .cases import Case
from .interface import InterfaceChanged, interface
from .process import CommandResult

GLSL = """#version 450
layout(local_size_x = 1) in;
layout(std430, binding = 0) buffer B0 { float fin[]; };
layout(std430, binding = 1) buffer B1 { int iin[]; };
layout(std430, binding = 2) buffer B2 { float outp[]; };
void main() {
    uint i = gl_GlobalInvocationID.x;
    outp[i] = ldexp(fin[i], iin[i]);
}
"""
METAL = """#include <metal_stdlib>
using namespace metal;
kernel void probe(device float* fin [[buffer(0)]], device int* iin [[buffer(1)]],
                  device float* outp [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    outp[i] = ldexp(fin[i], iin[i]);
}
"""
SPIRV_INTERFACE = """OpEntryPoint GLCompute %main "main"
OpExecutionMode %main LocalSize 1 1 1
OpDecorate %fin DescriptorSet 0
OpDecorate %fin Binding 0
OpDecorate %iin DescriptorSet 0
OpDecorate %iin Binding 1
OpDecorate %outp DescriptorSet 0
OpDecorate %outp Binding 2
"""
SOURCES = {
    "glsl": (".comp", GLSL),
    "metal": (".metal", METAL),
    "spirv": (".spvasm", SPIRV_INTERFACE),
}


def make_case(tmp_path, language="glsl", input_count=2):
    suffix, text = SOURCES[language]
    if input_count == 3:
        text = text.replace(
            "outp[i] = ldexp(fin[i], iin[i]);",
            "outp[i] = ldexp(fin[i], iin[i]) + float(uin[i]);",
        )
        if language == "metal":
            text = text.replace(
                "device float* outp [[buffer(2)]]",
                "device uint* uin [[buffer(2)]], device float* outp [[buffer(3)]]",
            )
        elif language == "glsl":
            text = text.replace(
                "layout(std430, binding = 2) buffer B2 { float outp[]; };",
                "layout(std430, binding = 2) buffer B2 { uint uin[]; };\nlayout(std430, binding = 3) buffer B3 { float outp[]; };",
            )
    path = tmp_path / ("probe" + suffix)
    path.write_text(text)
    return Case(
        id="mixed-inputs",
        cls="harness",
        title="Scale floats by integer exponents",
        rule="ldexp",
        inputs=(
            np.array([0.5, -2, 3, 4], np.float32),
            np.array([1, 2, -1, 0], np.int32),
        )
        + ((np.arange(4, dtype=np.uint32),) if input_count == 3 else ()),
        out_dtype=np.float32,
        oracle=lambda data: np.ldexp(*data[:2])
        + (data[2].astype(np.float32) if len(data) == 3 else 0),
        threads=4,
        group=1,
        src=path,
        lang=language,
    )


def staged(tmp_path, runtime="opengl-llvmpipe", count=2):
    case = make_case(tmp_path, input_count=count)
    cell = runner.Cell(case, "glsl", "strict", tmp_path, {})
    cell.artifact, cell.binary = case.src, tmp_path / "probe.spv"
    cell.abi = {
        "entry": "main",
        "group": (1, 1, 1),
        "bindings": tuple(range(count + 1)),
        "constants": None,
    }
    path, size = runner.stage(cell, runtimes.RUNTIMES[runtime])
    return cell, path, size


def test_case_keeps_single_array_and_sequence_oracles(tmp_path):
    case = make_case(tmp_path)
    assert case.out_count == 4
    np.testing.assert_array_equal(case.expected(), [1, -8, 1.5, 4])
    listed = replace(case, inputs=list(case.inputs))
    np.testing.assert_array_equal(listed.expected(), case.expected())
    single = replace(case, inputs=case.inputs[0], oracle=lambda data: data * 2)
    assert single.input_buffers == (single.inputs,)
    np.testing.assert_array_equal(single.expected(), single.inputs * 2)
    for invalid in ((), [], [1, 2]):
        with pytest.raises(ValueError, match="non-empty sequence of arrays"):
            replace(case, inputs=invalid)


@pytest.mark.parametrize("language", SOURCES)
def test_interfaces_require_the_exact_number_of_storage_slots(tmp_path, language):
    case = make_case(tmp_path, language)
    assert interface(case.src, language, 1, input_count=2)["bindings"] == (0, 1, 2)
    for count in (1, 3):
        with pytest.raises(InterfaceChanged, match="slots"):
            interface(case.src, language, 1, input_count=count)
    suffix, text = SOURCES[language]
    old = {"glsl": "binding = 1", "metal": "buffer(1)", "spirv": "Binding 1"}[language]
    for slot in (0, 3):
        case.src.write_text(text.replace(old, old.replace("1", str(slot))))
        with pytest.raises(InterfaceChanged, match="slots"):
            interface(case.src, language, 1, input_count=2)


@pytest.mark.parametrize("count", [0, -1, True, 1.5])
def test_interface_rejects_invalid_input_counts(tmp_path, count):
    case = make_case(tmp_path)
    with pytest.raises(InterfaceChanged, match="positive input-buffer count"):
        interface(case.src, "glsl", 1, input_count=count)


def test_constants_do_not_replace_a_storage_binding(tmp_path):
    case = make_case(tmp_path, "metal")
    text = METAL.replace("uint i [[", "constant uint& scale [[buffer(3)]], uint i [[")
    case.src.write_text(text)
    assert (
        interface(case.src, "metal", 1, True, input_count=2)["constants"]["slot"] == 3
    )
    for slot in (0, 1, 2):
        case.src.write_text(text.replace("buffer(3)", f"buffer({slot})"))
        with pytest.raises(InterfaceChanged, match="shares a storage-buffer"):
            interface(case.src, "metal", 1, True, input_count=2)
    case = make_case(tmp_path, "spirv")
    block = """OpDecorate %constants DescriptorSet 0
OpDecorate %constants Binding 3
OpDecorate %block Block
%ptr = OpTypePointer Uniform %block
%constants = OpVariable %ptr Uniform
"""
    case.src.write_text(SPIRV_INTERFACE + block)
    assert (
        interface(case.src, "spirv", 1, True, input_count=2)["constants"]["slot"] == 3
    )
    case.src.write_text(SPIRV_INTERFACE + block.replace("Binding 3", "Binding 2"))
    with pytest.raises(InterfaceChanged, match="shares a storage-buffer"):
        interface(case.src, "spirv", 1, True, input_count=2)
    # OpenGL uniform blocks have their own binding namespace.
    case = make_case(tmp_path)
    case.src.write_text(
        GLSL + "layout(std140, binding = 2) uniform C { uint scale; };\n"
    )
    assert interface(case.src, "glsl", 1, True, input_count=2)["constants"]["slot"] == 2


@pytest.mark.parametrize(
    "corrupt", [None, "input.bin", "input1.bin", "input2.bin", "output.bin"]
)
def test_stage_and_evaluate_check_every_buffer(tmp_path, corrupt):
    cell, path, size = staged(tmp_path, count=3)
    job = json.loads(path.read_text())
    assert job["bindings"] == [0, 1, 2, 3]
    for filename, data in zip(native.buffer_files(3), cell.case.input_buffers):
        assert (path.parent / filename).read_bytes() == data.tobytes()
    output = path.parent / "output.bin"
    output.write_bytes(cell.case.expected().tobytes() + output.read_bytes()[size:])
    if corrupt:
        target = path.parent / corrupt
        data = bytearray(target.read_bytes())
        data[-1] ^= 1
        target.write_bytes(data)
    result = runner.evaluate(cell, path, size, {"status": "executed"})
    assert result["outcome"] == ("memory-corruption" if corrupt else "preserved")
    assert result["inputUnchanged"] == (corrupt in (None, "output.bin"))
    assert result["outputGuardIntact"] == (corrupt != "output.bin")


def test_prepare_passes_the_input_count_and_hashes_ordered_buffers(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(runner, "validate", lambda *args, **kwargs: ([], {}))
    case = make_case(tmp_path)

    def prepare(case, config):
        return runner.prepare(
            case,
            "glsl",
            "reference",
            config,
            case.src,
            {"status": "reference"},
            environment="vulkan1.1",
            profile="test",
        )

    original = prepare(case, "original")
    assert original.outcome is None and original.abi["bindings"] == (0, 1, 2)
    changed = replace(case, inputs=(case.inputs[0], case.inputs[1] + 1))
    swapped = replace(case, inputs=case.inputs[::-1])
    # Identical concatenated bytes with different buffer boundaries must not share a hash.
    joined = np.concatenate([data.view(np.uint8) for data in case.inputs])
    split = replace(case, inputs=(joined[:4], joined[4:]))
    hashes = {original.base["inputSHA256"]}
    for index, other in enumerate((changed, swapped, split)):
        hashes.add(prepare(other, f"changed{index}").base["inputSHA256"])
    assert len(hashes) == 4
    single = replace(case, inputs=case.inputs[0])
    assert prepare(single, "single").base["inputSHA256"] == runner.digest(
        single.inputs.tobytes()
    )


@pytest.mark.parametrize("count", [1, 2, 3])
def test_vulkan_passes_all_storage_files_and_constants(tmp_path, monkeypatch, count):
    files = native.buffer_files(count)
    (tmp_path / native.CONSTANTS).write_bytes(bytes(16))
    seen = []

    def vkrun(args, **kwargs):
        seen.append(args)
        return CommandResult(
            tuple(args),
            "accepted",
            0,
            "ok gpu_ns=1\ndriver=llvmpipe\ndevice=llvmpipe\n",
        )

    monkeypatch.setattr(native, "run", vkrun)
    job = dict(
        directory=str(tmp_path),
        bindings=list(range(count + 1)),
        binary="probe.spv",
        entry="main",
        threads=4,
        group=(1, 1, 1),
        runtime="vulkan-lavapipe",
    )
    for block in (
        None,
        {"slot": count + 1, "kind": "uniform"},
        {"slot": None, "kind": "push"},
    ):
        native.execute_vulkan(dict(job, constants=block))
        assert seen[-1][6 : 6 + len(files)] == [
            f"{slot}:inout:{tmp_path / filename}" for slot, filename in enumerate(files)
        ]
        if block:
            prefix = "push" if block["kind"] == "push" else f"{count + 1}:uniform"
            assert seen[-1][-1] == f"{prefix}:{tmp_path / native.CONSTANTS}"


def test_opengl_binds_and_reads_back_all_buffers(tmp_path, monkeypatch):
    cell, path, _ = staged(tmp_path, count=3)
    job = json.loads(path.read_text())
    context = MagicMock()
    context.info = {"GL_RENDERER": "llvmpipe", "GL_VERSION": "4.5"}
    buffers = []

    def buffer(data):
        value = MagicMock()
        value.read.return_value = data + b"changed"
        buffers.append(value)
        return value

    context.buffer.side_effect = buffer
    monkeypatch.setitem(
        sys.modules,
        "moderngl",
        SimpleNamespace(
            create_standalone_context=lambda **kwargs: context, Error=RuntimeError
        ),
    )
    native.execute_opengl(job)
    assert len(buffers) == 4
    for slot, (buffer, filename) in enumerate(zip(buffers, native.buffer_files(3))):
        buffer.bind_to_storage_buffer.assert_called_once_with(slot)
        buffer.read.assert_called_once_with()
        buffer.release.assert_called_once_with()
        assert (path.parent / filename).read_bytes().endswith(b"changed")
    context.compute_shader.return_value.run.assert_called_once_with(group_x=4)
    context.release.assert_called_once_with()


def test_metal_binds_and_reads_back_all_buffers(tmp_path, monkeypatch):
    cell, path, _ = staged(tmp_path, "metal-apple", count=3)
    job = json.loads(path.read_text())
    device = MagicMock()
    device.name.return_value = "Apple test"
    device.newLibraryWithSource_options_error_.return_value = (MagicMock(), None)
    pipeline = MagicMock()
    pipeline.maxTotalThreadsPerThreadgroup.return_value = 64
    device.newComputePipelineStateWithFunction_error_.return_value = (pipeline, None)
    command = device.newCommandQueue.return_value.commandBuffer.return_value
    command.error.return_value = None
    buffers = []

    def buffer(data, length, options):
        value = MagicMock()
        value.length.return_value = length
        value.contents.return_value.as_buffer.return_value = data + b"changed"
        buffers.append(value)
        return value

    device.newBufferWithBytes_length_options_.side_effect = buffer
    metal = MagicMock()
    metal.MTLCreateSystemDefaultDevice.return_value = device
    monkeypatch.setitem(sys.modules, "Metal", metal)
    native.execute_metal(job)
    encoder = command.computeCommandEncoder.return_value
    assert len(buffers) == encoder.setBuffer_offset_atIndex_.call_count == 4
    for slot, (buffer, filename) in enumerate(zip(buffers, native.buffer_files(3))):
        encoder.setBuffer_offset_atIndex_.assert_any_call(buffer, 0, slot)
        assert (path.parent / filename).read_bytes().endswith(b"changed")

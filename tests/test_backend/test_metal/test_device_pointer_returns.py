"""Device pointer results of Metal functions, and buffer handles as Metal results and locals (#2194).

The importer typed every function returning a ``device`` or ``constant`` pointer as a CrossGL buffer handle
(``StructuredBuffer<T>``/``RWStructuredBuffer<T>``). A handle carries no offset, so ``advance(values)[i]`` for
``values += 1; return values;`` read ``values[i]`` in HLSL output, and Metal output wrote the handle type back
as the function's result type, which the Metal compiler rejects. A function that returns an interior pointer now
keeps its Metal pointer type; one that only returns pointer parameters unchanged (a buffer selector) keeps the
handle type. Metal output writes a handle result or local as the pointer Metal passes for the buffer.
"""

import shutil
import struct
import subprocess

import pytest

import crosstl
from crosstl.translator.refusals import MetalRefusal

HEADER = "#include <metal_stdlib>\nusing namespace metal;\n"

ADVANCE = HEADER + """
const device int* advance(const device int* values) {
    values += 1;
    return values;
}
const device int* offset(const device int* values, uint k) { return values + k; }
device int* skip(device int* values) {
    ++values;
    return values;
}
kernel void probe(const device int* inputs [[buffer(0)]],
                  device int* results [[buffer(1)]],
                  uint tid [[thread_position_in_grid]]) {
    skip(results)[tid] = advance(inputs)[tid] + offset(inputs, 2u)[tid];
}
"""

SELECTOR = HEADER + """
const device int* same(const device int* v) { return v; }
const device int* choose(bool first, const device int* a, const device int* b) {
    return first ? a : b;
}
kernel void probe(const device int* inputs [[buffer(0)]],
                  device int* results [[buffer(1)]],
                  uint tid [[thread_position_in_grid]]) {
    results[tid] = same(inputs)[tid] + choose((tid & 1u) == 0u, inputs, results)[64u + (tid & 3u)];
}
"""

CROSSGL_HANDLES = """shader BufferHandles {
    StructuredBuffer<int> inputs;
    RWStructuredBuffer<int> results;
    StructuredBuffer<int> same(StructuredBuffer<int> v) { return v; }
    RWStructuredBuffer<int> out_of(RWStructuredBuffer<int> r) { return r; }
    compute {
        layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;
        void main() {
            uint i = gl_GlobalInvocationID.x;
            RWStructuredBuffer<int> o = out_of(results);
            StructuredBuffer<int> s = inputs;
            o[i] = same(s)[i] * 3 + 1;
        }
    }
}
"""

APPEND_RESULT = """shader AppendResult {
    struct P { float x; };
    AppendStructuredBuffer<P> outs;
    AppendStructuredBuffer<P> pick(AppendStructuredBuffer<P> a) { return a; }
    compute {
        layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
        void main() {
            P p;
            p.x = 1.0;
            pick(outs).Append(p);
        }
    }
}
"""

COUNT = 64
WORDS = [((i * 2654435761) & 0xFFFFFFFF) >> 9 for i in range(COUNT + 4)]


def _translate(tmp_path, source, name, backend):
    path = tmp_path / name
    path.write_text(source, encoding="utf-8")
    return crosstl.translate(str(path), backend=backend)


def _compile(tmp_path, source, stem):
    xcrun = shutil.which("xcrun")
    if xcrun is None:
        pytest.skip("xcrun is not installed")
    artifact = tmp_path / f"{stem}.metal"
    artifact.write_text(source, encoding="utf-8")
    result = subprocess.run(
        [xcrun, "--sdk", "macosx", "metal", "-Werror", "-c", str(artifact)]
        + ["-o", str(tmp_path / f"{stem}.air")],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0 and "unable to find utility" in result.stderr:
        pytest.skip("the Metal compiler is not installed")
    assert result.returncode == 0, result.stdout + result.stderr


def _run(source, entry="probe", results_first=False):
    """Dispatch ``COUNT`` threads with ``WORDS`` in the input buffer; return the result buffer's words."""
    metal = pytest.importorskip("Metal")
    device = metal.MTLCreateSystemDefaultDevice()
    if device is None:
        pytest.skip("no Metal device")
    library, error = device.newLibraryWithSource_options_error_(source, None, None)
    assert library is not None, error
    pipeline, error = device.newComputePipelineStateWithFunction_error_(
        library.newFunctionWithName_(entry), None
    )
    assert pipeline is not None, error
    data = struct.pack(f"<{len(WORDS)}i", *WORDS)
    inputs = device.newBufferWithBytes_length_options_(data, len(data), 0)
    results = device.newBufferWithBytes_length_options_(data, len(data), 0)
    commands = device.newCommandQueue().commandBuffer()
    encoder = commands.computeCommandEncoder()
    encoder.setComputePipelineState_(pipeline)
    encoder.setBuffer_offset_atIndex_(inputs, 0, 1 if results_first else 0)
    encoder.setBuffer_offset_atIndex_(results, 0, 0 if results_first else 1)
    encoder.dispatchThreads_threadsPerThreadgroup_(
        metal.MTLSizeMake(COUNT, 1, 1), metal.MTLSizeMake(32, 1, 1)
    )
    encoder.endEncoding()
    commands.commit()
    commands.waitUntilCompleted()
    raw = bytes(results.contents().as_buffer(len(data)))
    return list(struct.unpack(f"<{len(WORDS)}i", raw))


def test_metal_input_keeps_the_pointer_type_of_an_advanced_device_pointer(tmp_path):
    crossgl = _translate(tmp_path, ADVANCE, "advance.metal", "cgl")

    assert "const device int* advance(const device int* values)" in crossgl
    assert "const device int* offset(const device int* values, uint k)" in crossgl
    assert "device int* skip(device int* values)" in crossgl
    assert "StructuredBuffer<int> advance" not in crossgl


def test_metal_input_keeps_buffer_handles_for_selectors_of_whole_buffers(tmp_path):
    crossgl = _translate(tmp_path, SELECTOR, "selector.metal", "cgl")

    assert "StructuredBuffer<int> same(const device int* v)" in crossgl
    assert "StructuredBuffer<int> choose(bool first" in crossgl


@pytest.mark.parametrize(
    "source, name",
    [(ADVANCE, "advance.metal"), (SELECTOR, "selector.metal")],
    ids=["interior-pointers", "whole-buffer-selectors"],
)
def test_metal_round_trip_compiles_and_computes_the_source_results(
    tmp_path, source, name
):
    metal = _translate(tmp_path, source, name, "metal")

    assert "StructuredBuffer" not in metal
    _compile(tmp_path, metal, "round_trip")
    assert _run(metal) == _run(source)


def test_metal_output_writes_buffer_handle_results_and_locals_as_pointers(tmp_path):
    metal = _translate(tmp_path, CROSSGL_HANDLES, "handles.cgl", "metal")

    assert "const device int *same(const device int *v)" in metal
    assert "device int *out_of(device int *r)" in metal
    assert "device int *o = out_of(results);" in metal
    assert "const device int *s = inputs;" in metal
    assert "StructuredBuffer" not in metal
    _compile(tmp_path, metal, "handles")
    entry = next(
        line.split("kernel void ", 1)[1].split("(", 1)[0]
        for line in metal.splitlines()
        if "kernel void " in line
    )
    expected = list(WORDS)
    for i in range(COUNT):
        expected[i] = (WORDS[i] * 3 + 1 + 2**31) % 2**32 - 2**31
    assert _run(metal, entry) == expected


def test_metal_output_refuses_a_counted_buffer_result(tmp_path):
    with pytest.raises(MetalRefusal, match="AppendStructuredBuffer"):
        _translate(tmp_path, APPEND_RESULT, "append.cgl", "metal")

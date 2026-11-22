"""Metal output lays out a CrossGL cbuffer without ``@buffer_layout`` by std140, as the GLSL and SPIR-V outputs do.

MSL's natural layout of the same members differs (MSL 3.2 section 2.2 and table 2.4): ``float3`` takes 16 bytes,
``bool`` one, a ``float`` array has a 4-byte stride and a ``float2x2`` 8-byte columns. std140 (GLSL 4.60 section
7.6.2.2) packs a scalar after a ``vec3``, stores a boolean in four bytes, and pads every array element and matrix
column to 16 bytes. The host fills the block by std140, so the Metal output declares the padded storage.
"""

import re
import struct

import pytest

import crosstl

BLOCK = """
    cbuffer Params @binding(3) {
        vec3 a;
        float b;
        bool flag;
        float w[3];
        mat2 m;
        int64_t big[2];
        u64vec3 wide[2];
    };
"""

PROBE = "shader Probe {\n" + BLOCK + """
    compute {
        layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
        buffer float* outp;
        void main() {
            uint i = gl_GlobalInvocationID.x;
            float f = flag ? 1.0 : 0.0;
            outp[i] = a.x + a.y * 2.0 + a.z * 4.0 + b * 8.0 + f * 16.0 + w[i] * 32.0
                + m[1][0] * 64.0 + m[0][1] * 128.0 + float(big[1]) * 256.0
                + float(wide[1].z) * 512.0;
        }
    }
}
"""


def _translate(tmp_path, source):
    path = tmp_path / "probe.cgl"
    path.write_text(source)
    return crosstl.translate(str(path), backend="metal", format_output=False)


def _flat(code):
    return " ".join(code.split())


def _struct(code):
    return re.search(r"struct Params \{[^}]*\};", _flat(code)).group(0)


def test_unmarked_cbuffer_declares_std140_storage(tmp_path):
    declaration = _struct(_translate(tmp_path, PROBE))
    assert "packed_float3 a; float b; uint flag;" in declaration
    assert "float4 w[3];" in declaration
    assert "crosstl_hlsl_cbuffer_float2x2_padded m;" in declaration
    assert "long2 big[2];" in declaration
    assert "ulong4 wide[2];" in declaration


def test_unmarked_cbuffer_matches_a_std140_marked_one(tmp_path):
    marked = PROBE.replace("@binding(3)", "@binding(3) @buffer_layout(std140)")
    assert _flat(_translate(tmp_path, PROBE)) == _flat(_translate(tmp_path, marked))


def test_metal_marked_cbuffer_keeps_the_natural_layout(tmp_path):
    source = PROBE.replace("@binding(3)", "@binding(3) @buffer_layout(metal)")
    declaration = _struct(_translate(tmp_path, source))
    assert "float3 a; float b; bool flag; float w[3]; float2x2 m;" in declaration


def _std140_block(i_values):
    """``Params`` by std140: a at 0, b at 12, flag at 16, w at 32 (stride 16), m at 80 (columns 16 apart), big at
    112 (stride 16) and wide, aligned to 32, at 160 (stride 32)."""
    data = bytearray(224)
    struct.pack_into("<3f", data, 0, 1.0, 2.0, 3.0)
    struct.pack_into("<f", data, 12, 5.0)
    struct.pack_into("<I", data, 16, 7)  # any nonzero word is true
    for index, value in enumerate(i_values):
        struct.pack_into("<f", data, 32 + 16 * index, value)
    struct.pack_into("<2f", data, 80, 11.0, 13.0)  # column 0
    struct.pack_into("<2f", data, 96, 17.0, 19.0)  # column 1
    struct.pack_into("<q", data, 112, -3)
    struct.pack_into("<q", data, 128, 23)
    struct.pack_into("<3Q", data, 160, 1, 2, 3)
    struct.pack_into("<3Q", data, 192, 29, 31, 37)
    return bytes(data)


def _expected(w, i):
    return (
        1.0
        + 2.0 * 2.0
        + 3.0 * 4.0
        + 5.0 * 8.0
        + 16.0
        + w[i] * 32.0
        + 17.0 * 64.0
        + 13.0 * 128.0
        + 23.0 * 256.0
        + 37.0 * 512.0
    )


def test_metal_output_reads_the_std140_bytes_the_host_uploads(tmp_path):
    Metal = pytest.importorskip("Metal")
    device = Metal.MTLCreateSystemDefaultDevice()
    if device is None:
        pytest.skip("no Metal device")
    code = _translate(tmp_path, PROBE)
    slots = dict(re.findall(r"(\w+) \[\[buffer\((\d+)\)\]\]", code))
    library, error = device.newLibraryWithSource_options_error_(
        code, Metal.MTLCompileOptions.new(), None
    )
    assert library is not None, error
    entry = re.search(r"kernel void (\w+)", code).group(1)
    pipeline, error = device.newComputePipelineStateWithFunction_error_(
        library.newFunctionWithName_(entry), None
    )
    assert pipeline is not None, error
    w = (0.5, -1.25, 3.0)
    block = _std140_block(w)
    constants = device.newBufferWithBytes_length_options_(
        block, len(block), Metal.MTLResourceStorageModeShared
    )
    results = device.newBufferWithLength_options_(
        12, Metal.MTLResourceStorageModeShared
    )
    command = device.newCommandQueue().commandBuffer()
    encoder = command.computeCommandEncoder()
    encoder.setComputePipelineState_(pipeline)
    encoder.setBuffer_offset_atIndex_(constants, 0, int(slots["params"]))
    encoder.setBuffer_offset_atIndex_(results, 0, int(slots["outp"]))
    encoder.dispatchThreads_threadsPerThreadgroup_(
        Metal.MTLSizeMake(3, 1, 1), Metal.MTLSizeMake(1, 1, 1)
    )
    encoder.endEncoding()
    command.commit()
    command.waitUntilCompleted()
    assert command.error() is None, command.error()
    data = struct.unpack("<3f", bytes(results.contents().as_buffer(12)))
    assert data == tuple(_expected(w, i) for i in range(3))

"""Row-major members of unmarked constant blocks are stored as their transpose in Metal output."""

import re
import struct

import pytest

import crosstl
from crosstl.translator.semantics import RowMajorMemberError

PROBE = """shader P {
    cbuffer Params @binding(3) {
        @ row_major mat3 m;
        @ row_major mat2x3 n;
        @ row_major mat3 ms[2];
        vec4 v;
    };
    compute {
        layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
        buffer float* outp;
        float pick(mat3 a) { return a[2][1]; }
        void main() {
            uint i = gl_GlobalInvocationID.x;
            vec3 t = m * v.xyz;
            vec3 u = v.xyz * m;
            vec3 w = n * v.xy;
            vec3 c = ms[1][i];
            mat3 copy = m;
            outp[i] = t.x + t.y * 2.0 + t.z * 4.0 + u.x * 8.0 + w.z * 16.0 + c.y * 32.0
                + m[2][1] * 64.0 + n[1][2] * 128.0 + ms[0][1][2] * 256.0 + pick(m) * 512.0
                + copy[0][i] * 1024.0;
            {
                mat3 m = mat3(1.0);
                outp[i] += m[2][1];
            }
        }
    }
}
"""

# Row-major matrices by row; ``M[r][c]`` is the CrossGL element ``m[c][r]``.
M = ((1.0, 2.0, 3.0), (4.0, 5.0, 6.0), (7.0, 8.0, 10.0))
N = ((1.0, 2.0), (3.0, 4.0), (5.0, 6.0))
MS = (
    ((1.0, 2.0, 3.0), (4.0, 5.0, 6.0), (7.0, 8.0, 9.0)),
    ((10.0, 11.0, 12.0), (13.0, 14.0, 15.0), (16.0, 17.0, 19.0)),
)
V = (2.0, 3.0, 5.0, 7.0)


def _translate(tmp_path, source, backend):
    path = tmp_path / "shader.cgl"
    path.write_text(source)
    return crosstl.translate(str(path), backend=backend, format_output=False)


def _flat(code):
    return " ".join(code.split())


def test_metal_declares_the_transpose_and_rewrites_every_read(tmp_path):
    code = _flat(_translate(tmp_path, PROBE, "metal"))
    assert (
        "struct Params { float3x3 m; float3x2 n; float3x3 ms[2]; float4 v; };" in code
    )
    assert "float3 t = transpose(params.m) * params.v.xyz;" in code
    assert "float3 u = params.v.xyz * transpose(params.m);" in code
    assert "float3 w = transpose(params.n) * params.v.xy;" in code
    assert "float3 c = transpose(params.ms[1])[i];" in code
    assert "float3x3 copy = transpose(params.m);" in code
    assert "params.m[1][2] * 64.0" in code
    assert "params.n[2][1] * 128.0" in code
    assert "params.ms[0][2][1] * 256.0" in code
    assert "pick(transpose(params.m)) * 512.0" in code
    # A local that shadows the member is an ordinary column-major matrix.
    assert "float3x3 m = float3x3(1.0); outp[i] += m[2][1];" in code


def test_marked_blocks_keep_their_layout_plan(tmp_path):
    source = PROBE.replace(
        "cbuffer Params @binding(3)",
        "cbuffer Params @binding(3) @buffer_layout(std140)",
    )
    code = _flat(_translate(tmp_path, source, "metal"))
    assert "float3 t = transpose(params.m) * params.v.xyz;" in code
    assert "params.m[1][2]" not in code


@pytest.mark.parametrize(
    "backend, declaration",
    [
        ("directx", "row_major float3x2 n;"),
        ("opengl", "layout(row_major) mat2x3 n;"),
    ],
)
def test_targets_with_row_major_matrices_keep_the_qualifier(
    tmp_path, backend, declaration
):
    assert declaration in _flat(_translate(tmp_path, PROBE, backend))


def test_refuses_an_array_member_used_without_selecting_one_matrix(tmp_path):
    source = PROBE.replace("vec3 c = ms[1][i];", "mat3 c[2] = ms;").replace(
        "c.y * 32.0", "c[1][i].y * 32.0"
    )
    with pytest.raises(RowMajorMemberError, match="without selecting one matrix"):
        _translate(tmp_path, source, "metal")


def _block():
    """``Params`` in MSL's layout of the transposed members: rows of ``m`` and ``ms`` 16 bytes apart, of ``n`` 8."""
    words = [0.0] * 48
    for row in range(3):
        words[4 * row : 4 * row + 3] = M[row]  # float3x3 at byte 0
        words[12 + 2 * row : 14 + 2 * row] = N[row]  # float3x2 at byte 48
        for index in range(2):  # float3x3[2] at byte 80
            start = 20 + 12 * index + 4 * row
            words[start : start + 3] = MS[index][row]
    words[44:48] = V  # float4 at byte 176
    return struct.pack("<48f", *words)


def _expected(i):
    column = lambda matrix, c: [matrix[r][c] for r in range(len(matrix))]  # noqa: E731
    t = [sum(M[r][c] * V[c] for c in range(3)) for r in range(3)]
    u = [sum(V[r] * M[r][c] for r in range(3)) for c in range(3)]
    w = [sum(N[r][c] * V[c] for c in range(2)) for r in range(3)]
    c = column(MS[1], i)
    return (
        t[0]
        + t[1] * 2.0
        + t[2] * 4.0
        + u[0] * 8.0
        + w[2] * 16.0
        + c[1] * 32.0
        + M[1][2] * 64.0
        + N[2][1] * 128.0
        + MS[0][2][1] * 256.0
        + M[1][2] * 512.0
        + column(M, 0)[i] * 1024.0
    )


def test_metal_output_reads_the_rows_the_host_uploads(tmp_path):
    Metal = pytest.importorskip("Metal")
    device = Metal.MTLCreateSystemDefaultDevice()
    if device is None:
        pytest.skip("no Metal device")
    code = _translate(tmp_path, PROBE, "metal")
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
    block = _block()
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
    assert data == tuple(_expected(i) for i in range(3))

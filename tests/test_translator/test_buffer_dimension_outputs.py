"""``buffer_dimensions(b, out...)`` writes the outputs the buffer's own ``GetDimensions`` declares.

CrossGL's statement form stores a buffer's dimensions into its output arguments: ``buffer_dimensions(b, count)``
for the element count, ``buffer_dimensions(b, count, stride)`` for a structured buffer's count and stride, and
``buffer_dimensions(b, bytes)`` for a byte-address buffer's size. The HLSL and Slang generators printed the outputs
as ``b.GetDimensions(count)`` for every buffer, but ``StructuredBuffer::GetDimensions`` has only the
``(out uint count, out uint stride)`` overload, so DXC ("no matching member function") and slangc ("not enough
arguments to call") rejected the lone-count form. That form now assigns the element count; output lists that no
overload takes are refused with a diagnostic.
"""

from __future__ import annotations

import re
import shutil
import subprocess

import pytest

import crosstl
from crosstl.translator.codegen.buffer_length import BufferLengthUnsupportedError
from crosstl.translator.diagnostics import TranslationDiagnostic

SHADER = """
shader BufferDimensionOutputs {
    struct Particle {
        vec4 position;
        float mass;
    };

    RWStructuredBuffer<Particle> particles @binding(0);
    StructuredBuffer<float> weights @binding(1);
    RWByteAddressBuffer raw @binding(2);
    RWStructuredBuffer<uint> counts @binding(3);

    compute {
        layout(local_size_x = 1) in;
        void main(uint3 tid @ gl_GlobalInvocationID) {
            uint count;
            uint weightCount;
            uint weightStride;
            uint bytes;
            buffer_dimensions(particles, count);
            buffer_dimensions(weights, weightCount);
            buffer_dimensions(weights, weightCount, weightStride);
            buffer_dimensions(raw, bytes);
            buffer_store(counts, 0u, count + weightCount + weightStride + bytes);
        }
    }
}
"""


def _with_call(call):
    return SHADER.replace("buffer_dimensions(raw, bytes);", call)


def _translate(tmp_path, source, backend):
    path = tmp_path / "shader.cgl"
    path.write_text(source, encoding="utf-8")
    return crosstl.translate(str(path), backend=backend, format_output=False)


def _flat(code):
    return re.sub(r"\s+", " ", code)


def _run(command):
    completed = subprocess.run(command, capture_output=True, text=True, timeout=300)
    output = "\n".join(p for p in (completed.stdout, completed.stderr) if p.strip())
    assert completed.returncode == 0, output[-3000:]


def _tool(name):
    path = shutil.which(name)
    if path is None:
        pytest.skip(f"{name} is not installed")
    return path


def test_hlsl_lone_count_of_a_structured_buffer_is_assigned(tmp_path):
    code = _flat(_translate(tmp_path, SHADER, "directx"))

    assert "weightCount = cgl_buffer_count(weights);" in code
    assert "weights.GetDimensions(weightCount, weightStride);" in code
    assert "raw.GetDimensions(bytes);" in code
    assert ".GetDimensions(weightCount);" not in code
    assert "particles.GetDimensions(count);" not in code


def test_slang_lone_count_of_a_structured_buffer_is_assigned(tmp_path):
    code = _flat(_translate(tmp_path, SHADER, "slang"))

    assert (
        "count = cgl_bufferDimensions_RWStructuredBuffer_Particle(particles);" in code
    )
    assert "weightCount = cgl_bufferDimensions_StructuredBuffer_float(weights);" in code
    assert "weights.GetDimensions(weightCount, weightStride);" in code
    assert "raw.GetDimensions(bytes);" in code
    assert "particles.GetDimensions(count);" not in code
    assert "weights.GetDimensions(weightCount);" not in code


@pytest.mark.parametrize(
    ("backend", "call", "error"),
    [
        (
            "directx",
            "uint extra; buffer_dimensions(raw, bytes, extra);",
            BufferLengthUnsupportedError,
        ),
        (
            "directx",
            "uint a; uint b; buffer_dimensions(weights, bytes, a, b);",
            BufferLengthUnsupportedError,
        ),
        (
            "slang",
            "uint extra; buffer_dimensions(raw, bytes, extra);",
            TranslationDiagnostic,
        ),
        (
            "slang",
            "uint a; uint b; buffer_dimensions(weights, bytes, a, b);",
            TranslationDiagnostic,
        ),
    ],
)
def test_output_lists_no_overload_takes_are_refused(tmp_path, backend, call, error):
    with pytest.raises(error, match="buffer_dimensions"):
        _translate(tmp_path, _with_call(call), backend)


def test_hlsl_output_compiles_with_dxc(tmp_path):
    dxc = _tool("dxc")
    source = tmp_path / "shader.hlsl"
    source.write_text(_translate(tmp_path, SHADER, "directx"), encoding="utf-8")
    _run([dxc, "-T", "cs_6_0", "-E", "CSMain", str(source)])


def test_slang_output_compiles_with_slangc(tmp_path):
    slangc = _tool("slangc")
    source = tmp_path / "shader.slang"
    source.write_text(_translate(tmp_path, SHADER, "slang"), encoding="utf-8")
    _run(
        [
            slangc,
            str(source),
            "-target",
            "spirv",
            "-entry",
            "main",
            "-stage",
            "compute",
            "-o",
            str(tmp_path / "shader.spv"),
        ]
    )


APPEND_SHADER = """
shader AppendBufferDimensions {
    AppendStructuredBuffer<float> queue @binding(0);
    RWStructuredBuffer<uint> counts @binding(1);

    compute {
        layout(local_size_x = 1) in;
        void main(uint3 tid @ gl_GlobalInvocationID) {
            uint queueCount;
            uint queueStride;
            buffer_dimensions(queue, queueCount);
            buffer_dimensions(queue, queueCount, queueStride);
            buffer_store(counts, 0u, queueCount + queueStride);
        }
    }
}
"""


def test_hlsl_append_buffer_reports_count_and_stride(tmp_path):
    code = _flat(_translate(tmp_path, APPEND_SHADER, "directx"))

    # AppendStructuredBuffer::GetDimensions takes (out uint numStructs, out uint stride).
    assert "queueCount = cgl_buffer_count(queue);" in code
    assert "queue.GetDimensions(queueCount, queueStride);" in code
    assert "queue.GetDimensions(queueCount);" not in code


def test_hlsl_append_buffer_dimensions_compile_with_dxc(tmp_path):
    dxc = _tool("dxc")
    source = tmp_path / "append.hlsl"
    source.write_text(_translate(tmp_path, APPEND_SHADER, "directx"), encoding="utf-8")
    _run([dxc, "-T", "cs_6_0", "-E", "CSMain", str(source)])

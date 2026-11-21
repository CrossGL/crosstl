"""Array constructors in shader-scope initializers (#3713, #3714, #3715).

Metal builds an array only in an aggregate initializer. A GLSL array constructor such as ``float[2](1.0, 2.0)``
in the initializer of a program-scope variable, a stage-block constant, a named structure constructor
(``W{m: float[2](1.0, 2.0), s: 3.0}``), an unsized constructor (``float[](1.0, 2.0)``) or an indexed constructor
(``float[2](1.0, 2.0)[1]``) used to reach Metal output verbatim, which the Metal compiler rejects with "expected '('
for function-style cast or type construction". Each now becomes an initializer list, or the element it selects,
and computes the CrossGL values on the GPU; an array constructor that is not a member of an aggregate in such an
initializer is refused with a typed diagnostic.
"""

import os
import struct

import pytest

import crosstl
from crosstl.translator.semantics.array_values import ArrayConstructorUnsupportedError

STRUCTURE_INITIALIZERS = """
shader Probe {
    buffer float* outp;
    struct W { float m[2]; float s; };
    W G = {float[2](1.0, 2.0), 3.0};
    const W N = W{m: float[2](4.0, 5.0), s: 6.0};
    compute {
        const W S = {float[2](7.0, 8.0), 9.0};
        layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
        void main() {
            uint i = gl_GlobalInvocationID.x;
            outp[i] = G.m[1] + N.m[1] * 10.0 + S.m[1] * 100.0 + G.s + N.s * 10.0 + S.s * 100.0;
        }
    }
}
"""
# 2 + 50 + 800 + 3 + 60 + 900
STRUCTURE_VALUE = 1815.0

INDEXED_CONSTANT = """
shader Probe {
    buffer float* outp;
    const float X = float[2](1.0, 2.0)[1];
    const float Y = float[3](4.0, float[2](5.0, 6.0)[0], 7.0)[1] + 0.5;
    compute {
        layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
        void main() {
            uint i = gl_GlobalInvocationID.x;
            outp[i] = X * 10.0 + Y;
        }
    }
}
"""
INDEXED_VALUE = 25.5

UNSIZED_MEMBERS = """
shader Probe {
    buffer float* outp;
    struct W { float m[2]; float s; };
    const W C = {float[](1.0, 2.0), 3.0};
    compute {
        layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
        void main() {
            uint i = gl_GlobalInvocationID.x;
            W c = {float[](4.0, 5.0), 6.0};
            outp[i] = C.m[1] + C.s * 10.0 + c.m[0] * 100.0 + c.s * 1000.0;
        }
    }
}
"""
UNSIZED_VALUE = 6432.0

NON_MEMBER = """
shader Probe {
    buffer float* outp;
    float f(float a[2]) { return a[0]; }
    const float X = f(float[2](1.0, 2.0));
    compute {
        layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
        void main() {
            uint i = gl_GlobalInvocationID.x;
            outp[i] = X;
        }
    }
}
"""


def _metal(tmp_path, source):
    path = tmp_path / "probe.cgl"
    path.write_text(source, encoding="utf-8")
    return crosstl.translate(os.fspath(path), backend="metal", format_output=False)


def _run_on_gpu(metal_source, entry="kernel_main"):
    Metal = pytest.importorskip("Metal")
    device = Metal.MTLCreateSystemDefaultDevice()
    if device is None:
        pytest.skip("no Metal device")
    library, error = device.newLibraryWithSource_options_error_(
        metal_source, Metal.MTLCompileOptions.new(), None
    )
    assert library is not None, f"{error}\n{metal_source}"
    function = library.newFunctionWithName_(entry)
    assert function is not None, metal_source
    pipeline, error = device.newComputePipelineStateWithFunction_error_(function, None)
    assert pipeline is not None, error

    output = device.newBufferWithLength_options_(4, Metal.MTLResourceStorageModeShared)
    command = device.newCommandQueue().commandBuffer()
    encoder = command.computeCommandEncoder()
    encoder.setComputePipelineState_(pipeline)
    encoder.setBuffer_offset_atIndex_(output, 0, 0)
    encoder.dispatchThreads_threadsPerThreadgroup_(
        Metal.MTLSizeMake(1, 1, 1), Metal.MTLSizeMake(1, 1, 1)
    )
    encoder.endEncoding()
    command.commit()
    command.waitUntilCompleted()
    assert command.error() is None, command.error()
    return struct.unpack("<f", bytes(output.contents().as_buffer(4)))[0]


def test_structure_initializers_at_shader_scope_are_lists(tmp_path):
    metal = _metal(tmp_path, STRUCTURE_INITIALIZERS)
    assert "float[2](" not in metal, metal
    assert "constant W G = {{1.0, 2.0}, 3.0};" in metal, metal
    assert "constant W N = W{{4.0, 5.0}, 6.0};" in metal, metal
    assert "const W S = {{7.0, 8.0}, 9.0};" in metal, metal


def test_structure_initializers_at_shader_scope_compute_on_the_gpu(tmp_path):
    assert _run_on_gpu(_metal(tmp_path, STRUCTURE_INITIALIZERS)) == STRUCTURE_VALUE


def test_indexed_constructor_in_a_constant_is_its_element(tmp_path):
    metal = _metal(tmp_path, INDEXED_CONSTANT)
    assert "float[" not in metal, metal
    assert "constant float X = 2.0;" in metal, metal
    assert (
        "constant float Y = (5.0 + 0.5);" in metal
        or "constant float Y = 5.0 + 0.5;" in metal
    ), metal
    assert _run_on_gpu(metal) == INDEXED_VALUE


def test_unsized_constructor_members_are_lists(tmp_path):
    metal = _metal(tmp_path, UNSIZED_MEMBERS)
    assert "float[](" not in metal, metal
    assert "constant W C = {{1.0, 2.0}, 3.0};" in metal, metal
    assert "W c = {{4.0, 5.0}, 6.0};" in metal, metal
    assert _run_on_gpu(metal) == UNSIZED_VALUE


@pytest.mark.parametrize("backend", ["metal", "directx"])
def test_constructor_outside_an_aggregate_at_shader_scope_is_refused(tmp_path, backend):
    path = tmp_path / "probe.cgl"
    path.write_text(NON_MEMBER, encoding="utf-8")
    with pytest.raises(ArrayConstructorUnsupportedError) as raised:
        crosstl.translate(os.fspath(path), backend=backend, format_output=False)
    assert raised.value.reason_code == "array-constructor-at-shader-scope"
    assert "shader-scope constant 'X'" in str(raised.value)


def test_hlsl_refuses_a_named_constructor_with_an_array_member_at_shader_scope(
    tmp_path,
):
    """HLSL writes ``W{...}`` as a call, whose arguments cannot be initializer lists, so the member stays a
    constructor and is refused."""
    path = tmp_path / "probe.cgl"
    path.write_text(STRUCTURE_INITIALIZERS, encoding="utf-8")
    with pytest.raises(ArrayConstructorUnsupportedError) as raised:
        crosstl.translate(os.fspath(path), backend="directx", format_output=False)
    assert "shader-scope constant 'N'" in str(raised.value)

"""Enum variants after a computed initializer are numbered as C++ numbers them (#3703).

A variant without an initializer is one more than the previous variant. Before the fix the HLSL, Metal, Slang and
HIP generators restarted at 0 after an initializer that is not a plain literal (``-5``, ``1 << 3``, ``W::P + 3``),
and the Metal importer wrote a scoped enum's ``Q = P + 3`` without naming ``P``'s enum. Every output compiled and
computed with the wrong values.
"""

import os
import struct

import pytest

import crosstl
import crosstl.translator
from crosstl.translator.codegen.metal_codegen import MetalCodeGen

CROSSGL_PROBE = """
shader Probe {
    enum V { A = -5, B, C = 1 << 3, D }
    enum W { P = 2, Q = W::P + 3, R }
    buffer float values[];
    compute {
        layout(local_size_x = 1) in;
        void main() {
            uint i = gl_GlobalInvocationID.x;
            values[i] = float(V::B) + float(V::D) + float(W::R);
        }
    }
}
"""

METAL_PROBE = """
#include <metal_stdlib>
using namespace metal;

enum class W : int { P = 2, Q = P + 3, R };
enum V { A = -5, B, C = 1 << 3, D };

kernel void probe(device float* values [[buffer(0)]], uint i [[thread_position_in_grid]]) {
    values[i] = float(int(W::R)) + float(B) + float(D);
}
"""

# V::B = -4, V::D = 9, W::R = 6.
EXPECTED = -4.0 + 9.0 + 6.0

FOLLOWING_VALUES = (
    "int V_B = V_A + 1;",
    "int V_D = V_C + 1;",
    "int W_R = W_Q + 1;",
)


def _translate(tmp_path, source, extension, backend):
    path = tmp_path / f"probe{extension}"
    path.write_text(source)
    return crosstl.translate(os.fspath(path), backend=backend, format_output=False)


@pytest.mark.parametrize("backend", ["metal", "directx", "slang", "hip"])
def test_variants_after_a_computed_value_follow_the_previous_constant(
    tmp_path, backend
):
    generated = _translate(tmp_path, CROSSGL_PROBE, ".cgl", backend)
    for declaration in FOLLOWING_VALUES:
        assert declaration in generated, generated
    assert "int W_Q = " in generated and "W_P + 3" in generated, generated


def test_variants_after_literal_values_stay_numbers():
    metal = MetalCodeGen().generate(crosstl.translator.parse("""
shader Literals {
    enum E { X, Y = 4, Z }
    buffer int values[];
    compute {
        layout(local_size_x = 1) in;
        void main() { values[0] = E::X + E::Y + E::Z; }
    }
}
"""))
    assert "constant int E_X = 0;" in metal, metal
    assert "constant int E_Y = 4;" in metal, metal
    assert "constant int E_Z = 5;" in metal, metal


def test_metal_import_names_the_enum_of_an_earlier_enumerator(tmp_path):
    crossgl = _translate(tmp_path, METAL_PROBE, ".metal", "cgl")
    assert "Q = W::P + 3" in crossgl, crossgl

    metal = _translate(tmp_path, METAL_PROBE, ".metal", "metal")
    assert "constant int W_Q = W_P + 3;" in metal, metal
    assert "constant int W_R = W_Q + 1;" in metal, metal


def _run_on_gpu(metal_source, entry):
    Metal = pytest.importorskip("Metal")
    device = Metal.MTLCreateSystemDefaultDevice()
    if device is None:
        pytest.skip("no Metal device")
    library, error = device.newLibraryWithSource_options_error_(
        metal_source, Metal.MTLCompileOptions.new(), None
    )
    assert library is not None, error
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


def test_crossgl_enum_values_compute_on_the_gpu(tmp_path):
    metal = _translate(tmp_path, CROSSGL_PROBE, ".cgl", "metal")
    assert _run_on_gpu(metal, "kernel_main") == EXPECTED


def test_imported_metal_enum_values_compute_on_the_gpu(tmp_path):
    assert _run_on_gpu(METAL_PROBE, "probe") == EXPECTED
    metal = _translate(tmp_path, METAL_PROBE, ".metal", "metal")
    assert _run_on_gpu(metal, "probe") == EXPECTED

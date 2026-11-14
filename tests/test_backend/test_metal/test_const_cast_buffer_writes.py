"""Metal ``device const`` buffers that a kernel writes after casting the ``const`` away.

``device const char* dst [[buffer(1)]]`` binds the same buffer as ``device char* dst``. A kernel that stores through
``(device float*)((device char*)dst + offset)`` writes into it, so CrossGL must declare the buffer read-write, or
every target refuses (or drops) the store. A ``const`` buffer the kernel only reads stays read-only.
"""

import pytest

import crosstl
from crosstl.translator.codegen.directx_pointer_alias_errors import (
    DirectXResourcePointerAliasError,
)

PREFIX = "#include <metal_stdlib>\nusing namespace metal;\n"


def _translate(tmp_path, source, backend):
    path = tmp_path / "kernel.metal"
    path.write_text(PREFIX + source)
    return crosstl.translate(
        str(path), backend=backend, format_output=False, source_backend="metal"
    )


def _parameters(cgl):
    line = next(line for line in cgl.splitlines() if "void k(" in line)
    return line.split("(", 1)[1]


BYTE_VIEW_STORE = """
kernel void k(device const float* src [[buffer(0)]],
              device const char* dst [[buffer(1)]],
              uint i [[thread_position_in_grid]]) {
    device float* row = (device float*)((device char*)dst + i * 4);
    row[0] = src[i] * 2.0f;
}
"""

SAME_TYPE_STORE = """
kernel void k(device const float* src [[buffer(0)]],
              device const float* dst [[buffer(1)]],
              uint i [[thread_position_in_grid]]) {
    device float* out = (device float*)dst + i;
    out[0] = src[i] * 2.0f;
}
"""


def test_byte_view_store_through_const_cast_makes_the_buffer_writable(tmp_path):
    parameters = _parameters(_translate(tmp_path, BYTE_VIEW_STORE, "cgl"))
    assert "StructuredBuffer<float> src @buffer(0)" in parameters
    assert "RWStructuredBuffer<float> src" not in parameters
    assert "RWStructuredBuffer<int8> dst @buffer(1)" in parameters


def test_same_type_store_through_const_cast_reaches_every_target(tmp_path):
    parameters = _parameters(_translate(tmp_path, SAME_TYPE_STORE, "cgl"))
    assert "RWStructuredBuffer<float> dst @buffer(1)" in parameters

    hlsl = _translate(tmp_path, SAME_TYPE_STORE, "directx")
    assert "StructuredBuffer<float> src : register(t0);" in hlsl
    assert "RWStructuredBuffer<float> dst : register(u1);" in hlsl
    assert "dst[uint(out__offset)] = (src.Load(i) * 2.0);" in hlsl

    glsl = _translate(tmp_path, SAME_TYPE_STORE, "opengl")
    assert "readonly buffer srcBuffer" in glsl
    assert "layout(std430, binding = 1) buffer dstBuffer { float dst[]; };" in glsl


@pytest.mark.parametrize(
    "statement",
    [
        "((device float*)dst)[i] = 1.0f;",
        "*((device float*)dst + i) = 1.0f;",
        "device float* out = (device float*)dst; out[i] += 1.0f;",
        "device float* out = (device float*)dst; out[i]++;",
        "auto out = (device float*)dst; device float* next = out + 1; next[i] = 0.0f;",
        "device float* out = (device float*)(i > 4u ? dst : dst + 1); out[0] = 1.0f;",
        "device float* out; out = (device float*)dst; out[i] = 1.0f;",
    ],
)
def test_store_forms_through_const_cast_make_the_buffer_writable(tmp_path, statement):
    source = (
        "kernel void k(device const float* dst [[buffer(0)]],"
        " uint i [[thread_position_in_grid]]) {\n    " + statement + "\n}\n"
    )
    parameters = _parameters(_translate(tmp_path, source, "cgl"))
    assert "RWStructuredBuffer<float> dst @buffer(0)" in parameters


def test_const_cast_pointer_passed_to_a_call_makes_the_buffer_writable(tmp_path):
    source = """
void put(device float* p, uint i) { p[i] = 1.0f; }
kernel void k(device const float* dst [[buffer(0)]],
              uint i [[thread_position_in_grid]]) {
    put((device float*)dst, i);
}
"""
    parameters = _parameters(_translate(tmp_path, source, "cgl"))
    assert "RWStructuredBuffer<float> dst @buffer(0)" in parameters


@pytest.mark.parametrize(
    "statement",
    [
        "result[i] = dst[i];",
        "device const float* view = dst + 1; result[i] = view[0];",
        "device float* cast = (device float*)dst; result[i] = cast[0];",
        "const device float* view = (const device float*)dst; result[i] = view[0];",
        "device float* local = result; local[i] = dst[i];",
    ],
)
def test_const_buffer_that_is_only_read_stays_read_only(tmp_path, statement):
    source = (
        "kernel void k(device const float* dst [[buffer(0)]],"
        " device float* result [[buffer(1)]],"
        " uint i [[thread_position_in_grid]]) {\n    " + statement + "\n}\n"
    )
    parameters = _parameters(_translate(tmp_path, source, "cgl"))
    assert "StructuredBuffer<float> dst @buffer(0)" in parameters
    assert "RWStructuredBuffer<float> dst" not in parameters
    assert "RWStructuredBuffer<float> result @buffer(1)" in parameters


def test_constant_address_space_buffer_is_never_promoted(tmp_path):
    source = """
kernel void k(constant float* params [[buffer(0)]],
              device float* result [[buffer(1)]],
              uint i [[thread_position_in_grid]]) {
    result[i] = params[0];
}
"""
    parameters = _parameters(_translate(tmp_path, source, "cgl"))
    assert "RWStructuredBuffer<float> params" not in parameters


def test_store_through_an_alias_of_a_read_only_buffer_is_a_typed_refusal(tmp_path):
    shader = """
    shader ReadonlyPointerWrite {
        compute {
            layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

            void main(StructuredBuffer<float> sourceValues @buffer(0)) {
                device float* writer = sourceValues + 1;
                *writer = 2.0;
            }
        }
    }
    """
    path = tmp_path / "readonly.cgl"
    path.write_text(shader)
    with pytest.raises(DirectXResourcePointerAliasError) as raised:
        crosstl.translate(str(path), backend="directx", format_output=False)
    assert raised.value.reason == "access-mismatch"
    assert raised.value.pointer_name == "writer"
    assert raised.value.project_diagnostic_code == (
        "project.translate.directx-resource-pointer-alias-unsupported"
    )

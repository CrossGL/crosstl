"""Subscripts of multidimensional Metal arrays remove one dimension each.

``T a[N][M]`` gives ``a[i]`` the type ``T[M]`` and ``a[i][j]`` the type ``T`` ([dcl.array]/7, [expr.sub]/1). Argument
inference used to give every array subscript the innermost element type, so ``a[i][j]`` was untyped and ``a[i]`` was
typed as a ``T`` value.
"""

import pytest

import crosstl
from crosstl.backend.Metal.array_element_receivers import (
    array_extents_at,
    array_subscript_element_type,
)
from crosstl.backend.Metal.preprocessor import MetalPreprocessor

KERNEL = """#include <metal_stdlib>
using namespace metal;

template <typename TO>
struct Op {
  template <typename T = TO, enable_if_t<is_same_v<T, float>, bool> = true>
  static inline T finalize(float v, float p) {
    return v + p;
  }
};

template <typename OP>
kernel void k(
    device float* out [[buffer(0)]],
    uint2 tid [[thread_position_in_threadgroup]]) {
  threadgroup float shmem[4][8];
  shmem[tid.y][tid.x] = float(tid.x);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  out[tid.x] = OP::finalize(ARGUMENT, 1.0f);
}

template [[host_name("k_float")]] kernel void k<Op<float>>(device float*, uint2);
"""


def _translate(tmp_path, argument, backend):
    source = tmp_path / "kernel.metal"
    source.write_text(KERNEL.replace("ARGUMENT", argument))
    return crosstl.translate(
        str(source), backend=backend, source_backend="metal", format_output=False
    )


def _view(source, position_marker="CALL"):
    preprocessor = MetalPreprocessor()
    position = source.index(position_marker)
    declarations = preprocessor._collect_buffer_element_types(source, [])
    return preprocessor, preprocessor._flatten_types_at(declarations, position)


@pytest.mark.parametrize(
    "text,position,extents",
    (
        (" [4][8];", 0, ["4", "8"]),
        ("[TG_Y][ TG_X ]", 0, ["TG_Y", "TG_X"]),
        ("[N * 2];", 0, ["N * 2"]),
        ("[4][8", 0, []),
        ("= 1;", 0, []),
    ),
)
def test_array_extents_at(text, position, extents):
    assert array_extents_at(text, position) == extents


def test_array_subscript_element_type_keeps_inner_dimensions():
    assert array_subscript_element_type("float", ["4"]) == "float"
    assert array_subscript_element_type("float", ["4", "8"]) == "float[8]"
    assert array_subscript_element_type("half2", ["2", "3", "4"]) == "half2[3][4]"


@pytest.mark.parametrize(
    "expression,expected",
    (
        ("rows[0][i]", "float"),
        ("rows[i][i + 1]", "float"),
        ("cube[0][1][i]", "uint"),
        ("flat[i]", "float"),
        # A sub-array has the array type of its remaining dimensions, as a
        # member array does.
        ("rows[0]", "float[8]"),
        ("cube[0][1]", "uint[4]"),
        ("*rows", "float[8]"),
        # The address of a row points to an array, not to a `float`.
        ("&rows[0]", None),
        # A one-dimensional element address keeps its pointer type.
        ("&flat[i]", "threadgroup float*"),
    ),
)
def test_argument_inference_subscripts_one_dimension_at_a_time(expression, expected):
    source = (
        "kernel void k(uint i [[thread_index_in_threadgroup]]) {\n"
        "  threadgroup float rows[4][8];\n"
        "  thread uint cube[2][3][4];\n"
        "  threadgroup float flat[8];\n"
        "  CALL;\n"
        "}\n"
    )
    preprocessor, view = _view(source)
    assert view["rows"].element_type == "float[8]"
    assert view["cube"].element_type == "uint[3][4]"
    assert view["flat"].element_type == "float"
    assert (
        preprocessor._infer_argument_type(expression, view, {"i": "uint"}) == expected
    )


@pytest.mark.parametrize(
    "parameter,element_type",
    (
        ("thread float (&m)[4][8]", "float[8]"),
        ("thread float m[4][8]", "float[8]"),
        ("threadgroup half m[16]", "half"),
        ("thread float (&m)[4]", "float"),
    ),
)
def test_array_parameters_record_their_inner_dimensions(parameter, element_type):
    subscriptable = MetalPreprocessor()._pointer_or_array_parameter_type(parameter)
    assert subscriptable.element_type == element_type


def test_two_dimensional_element_argument_specializes_member_template(tmp_path):
    crossgl = _translate(tmp_path, "shmem[0][tid.x]", "cgl")
    assert "threadgroup float[4][8] shmem;" in crossgl
    assert "Op_float__finalize__float(shmem[0][tid.x], 1.0f)" in crossgl

    metal = _translate(tmp_path, "shmem[0][tid.x]", "metal")
    assert "threadgroup float shmem[4][8];" in metal
    assert "Op_float__finalize__float(shmem[0][tid.x], 1.0)" in metal

    glsl = _translate(tmp_path, "shmem[0][tid.x]", "opengl")
    assert "shared float k_float_shmem[4][8];" in glsl
    assert "Op_float_finalize_float(k_float_shmem[0][tid.x], 1.0)" in glsl

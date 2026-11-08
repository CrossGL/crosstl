"""Implicit template materialization does not type a call by another overload's return type."""

import pytest

import crosstl
from crosstl.project.pipeline import MetalSourceMaterializationError

SOURCE = """#include <metal_stdlib>
using namespace metal;
float f(float x) {{ return x * 2.0f; }}
{other} f({other_parameter} x) {{ return {other}(x); }}
template <typename T>
T g(T v) {{ return v + T(1); }}
kernel void k(device float* out [[buffer(0)]], uint i [[thread_position_in_grid]]) {{
  float x = out[i];
  out[i] = g(f(x));
}}
"""


def _translate(tmp_path, other, other_parameter, backend="directx"):
    path = tmp_path / "overloads.metal"
    path.write_text(SOURCE.format(other=other, other_parameter=other_parameter))
    return crosstl.translate(str(path), backend=backend, format_output=False)


@pytest.mark.parametrize("backend", ("directx", "vulkan", "opengl"))
def test_overloads_with_different_return_types_do_not_type_the_call(tmp_path, backend):
    # f(float) is selected, so g is g<float>. The later f(half) definition used to
    # type the call as half, which materialized g<half> and rounded the value.
    try:
        output = _translate(tmp_path, "half", "half", backend)
    except MetalSourceMaterializationError as error:
        assert "g missing T" in str(error)
    else:
        assert "g_half" not in output
        assert "g_float" in output


def test_overloads_with_one_return_type_still_type_the_call(tmp_path):
    output = _translate(tmp_path, "float", "half")
    assert "g_float(" in output
    assert "g_half" not in output


UNDEDUCED_AUTO = """#include <metal_stdlib>
using namespace metal;
template <typename T, typename U>
T smaller(T a, U b) { return a < T(b) ? a : T(b); }
kernel void k(device uint* out [[buffer(0)]], uint tid [[thread_position_in_grid]]) {
  const auto limit = undeclared_source_of_a_value();
  out[tid] = smaller(tid, limit);
}
"""


def test_an_undeduced_auto_local_is_not_a_template_argument():
    from crosstl.backend.Metal.preprocessor import MetalPreprocessor
    from crosstl.project.pipeline import _metal_expression_type

    preprocessor = MetalPreprocessor()
    environment = {"limit": "auto", "begin": "uint", "pointer": "const auto*"}
    assert _metal_expression_type(preprocessor, "limit", environment, {}) is None
    assert (
        _metal_expression_type(preprocessor, "begin + limit", environment, {}) is None
    )
    assert _metal_expression_type(preprocessor, "pointer", environment, {}) is None
    assert _metal_expression_type(preprocessor, "begin", environment, {}) == "uint"
    # Increments keep their operand's type; `+` alone does not.
    assert _metal_expression_type(preprocessor, "begin++", environment, {}) == "uint"
    assert _metal_expression_type(preprocessor, "--begin", environment, {}) == "uint"
    assert _metal_expression_type(preprocessor, "limit++", environment, {}) is None


ARITHMETIC = """#include <metal_stdlib>
using namespace metal;
template <typename T>
T g(T v) {{ return v; }}
kernel void k(device {out_type}* out [[buffer(0)]], constant long& big [[buffer(1)]],
              constant float4& v4 [[buffer(2)]], constant half& h [[buffer(3)]],
              uint tid [[thread_position_in_grid]]) {{
  uint u = tid;
  long l = big;
  float f = float(tid);
  short s = short(tid);
  out[tid] = {out_cast}g({argument});
}}
"""


@pytest.mark.parametrize(
    "argument,out_type,out_cast,specialization",
    (
        # uint + long is long; the first operand's uint used to be taken.
        ("u + l", "long", "", "g_long("),
        ("u - l", "long", "", "g_long("),
        ("v4 * f", "float4", "", "g_float4("),
        ("h * f", "float", "", "g_float("),
        ("s + s", "int", "", "g_int("),
        ("u + 1u", "uint", "", "g_uint("),
    ),
)
def test_arithmetic_arguments_take_the_usual_arithmetic_conversions(
    tmp_path, argument, out_type, out_cast, specialization
):
    path = tmp_path / "arithmetic.metal"
    path.write_text(
        ARITHMETIC.format(argument=argument, out_type=out_type, out_cast=out_cast)
    )
    output = crosstl.translate(str(path), backend="directx", format_output=False)
    assert specialization in output


def test_arithmetic_result_types():
    from crosstl.backend.Metal.preprocessor import MetalPreprocessor
    from crosstl.project.arithmetic_result_types import arithmetic_result_type

    preprocessor = MetalPreprocessor()
    cases = (
        (["uint", "long"], "+", "long"),
        (["int", "uint"], "+", "uint"),
        (["short", "short"], "*", "int"),
        (["half", "float"], "*", "float"),
        (["int", "half"], "*", "half"),
        (["half4", "float"], "*", "half4"),
        (["float4", "int"], "*", "float4"),
        (["int4", "float"], "*", None),
        (["half", "bfloat"], "+", None),
        (["float4", "half4"], "+", None),
        (["float", "int"], "%", None),
        (["uint", None], "+", None),
        (["Custom", "Custom"], "+", "Custom"),
        (["Custom", "float"], "*", None),
        (["int", "int", "long"], "+", "long"),
    )
    for operands, operator, expected in cases:
        assert (
            arithmetic_result_type(preprocessor, operands, operator) == expected
        ), operands


def test_pointer_arithmetic_result_types():
    from crosstl.backend.Metal.preprocessor import MetalPreprocessor
    from crosstl.project.arithmetic_result_types import arithmetic_result_type

    preprocessor = MetalPreprocessor()
    cases = (
        # A pointer plus an offset has the pointer's type; an offset of `+` is
        # integral even when it is untyped, because pointer + pointer is
        # ill-formed.
        (["threadgroup float*", None], "+", "threadgroup float*"),
        ([None, "device float*"], "+", "device float*"),
        (["device float*", "int", None], "+", "device float*"),
        (["device half* const&", "uint"], "+", "device half*"),
        (["device float*", "int"], "-", "device float*"),
        # `p - q` of two pointers is a ptrdiff_t, so a subtrahend must be typed.
        (["device float*", None], "-", None),
        (["device float*", "device float*"], "-", None),
        (["int", "device float*"], "-", None),
        (["device float*", "float"], "+", None),
        (["device float*", "device float*"], "+", None),
        (["device float*", "int"], "*", None),
    )
    for operands, operator, expected in cases:
        assert arithmetic_result_type(preprocessor, operands, operator) == expected, (
            operands,
            operator,
        )


MEMBER_POINTER_OFFSET = """#include <metal_stdlib>
using namespace metal;
template <typename T, int N, typename W>
inline void fill(T value, W out) {
  for (int i = 0; i < N; i++) {
    out[i] = value;
  }
}
template <typename T>
struct Loader {
  threadgroup T* dst;
  void load() const thread {
    for (int i = 0; i < 2; i++) {
      fill<T, 4>(T(1), dst + i * 4);
    }
  }
};
kernel void k(device float* out [[buffer(0)]], uint tid [[thread_position_in_grid]],
              uint lid [[thread_index_in_threadgroup]]) {
  threadgroup float scratch[64];
  Loader<float> loader{scratch + lid * 8};
  loader.load();
  threadgroup_barrier(mem_flags::mem_threadgroup);
  out[tid] = scratch[lid];
}
"""


@pytest.mark.parametrize("backend", ("directx", "vulkan"))
def test_a_member_pointer_offset_deduces_the_pointer_type(tmp_path, backend):
    # `dst + i * 4` is `self.dst + ...` inside the method; the loop variable is
    # untyped there, and the argument still has the member's pointer type.
    path = tmp_path / "member_pointer_offset.metal"
    path.write_text(MEMBER_POINTER_OFFSET)
    output = crosstl.translate(str(path), backend=backend, format_output=False)
    if backend == "directx":
        assert "fill_float_4_threadgroup_float(" in output
    else:
        assert "OpFunctionCall" in output

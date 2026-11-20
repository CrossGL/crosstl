"""Constexpr helpers templated on a type, used where a constant expression is required.

A non-type template argument is a converted constant expression ([temp.arg.nontype]/2), so
``template <typename T, uint W = work_per_thread<T>()>`` gives each specialization one integral ``W``:
``work_per_thread<float>()`` is ``div_ceil<8, sizeof(float)>()``, which is 2, and ``half`` gives 4. The importer
evaluates such helpers for builtin type arguments, whose ``sizeof`` the Metal specification fixes, and keeps
failing closed for a dependent or aggregate argument. Shapes are those of candle's ``affine.metal`` and
``ternary.metal``.
"""

import platform
import re
import shutil
import subprocess

import pytest

from crosstl.backend.Metal.preprocessor import MetalPreprocessor
from tests.stage_entry_helpers import joined_entries

PRELUDE = """
#include <metal_stdlib>
using namespace metal;

template<uint Y>
constexpr uint div_ceil(uint x) {
    return x / Y + (x % Y > 0);
}

template<uint X, uint Y>
constexpr uint div_ceil() {
    return X / Y + (X % Y > 0);
}

template<typename T>
constexpr uint work_per_thread() {
    return div_ceil<8, sizeof(T)>();
}
"""

EXPLICIT_INSTANTIATION = PRELUDE + """
template <typename T, int W = work_per_thread<T>()>
[[kernel]] void affine_kernel(
    constant size_t &dim,
    device const T *input,
    device T *output,
    uint tid [[thread_position_in_grid]]
) {
    const uint step = div_ceil<W>(dim);
    for (uint i = tid; i < dim; i += step) {
        output[i] = static_cast<T>(float(input[i]) * 2.0f);
    }
}

template [[host_name("affine_f32")]] [[kernel]] decltype(affine_kernel<float>) affine_kernel<float>;
template [[host_name("affine_f16")]] [[kernel]] decltype(affine_kernel<half>) affine_kernel<half>;
"""

CALL_SITE = PRELUDE + """
template<typename T, typename ID, uint W = work_per_thread<T>()>
METAL_FUNC void where_cond(constant size_t &numel, device const ID *ids, device const T *t,
                           device T *out, uint tid) {
    const uint step = div_ceil<W>(numel);
    for (uint i = tid; i < numel; i += step) {
        out[i] = ids[i] ? t[i] : T(0);
    }
}

kernel void where_u32_f16(constant size_t &numel, device const uint32_t *ids, device const half *t,
                          device half *out, uint i [[thread_position_in_grid]]) {
    where_cond<half, uint32_t>(numel, ids, t, out, i);
}

kernel void where_u8_f32(constant size_t &numel, device const uint8_t *ids, device const float *t,
                         device float *out, uint i [[thread_position_in_grid]]) {
    where_cond<float, uint8_t>(numel, ids, t, out, i);
}
"""

UNIQUE_CALLEE = """
#include <metal_stdlib>
using namespace metal;

template<typename T>
constexpr uint lanes() {
    return 16 / sizeof(T);
}

template<uint N>
uint scaled(uint x) {
    return x * N;
}

kernel void k(device uint* o [[buffer(0)]], uint tid [[thread_position_in_grid]]) {
    o[tid] = scaled<lanes<half>()>(tid);
    o[tid + 1] = scaled<lanes<packed_float3>()>(tid);
}
"""


def _translate(tmp_path, source, backend="metal"):
    path = tmp_path / "unit.metal"
    path.write_text(source)
    return joined_entries(path, backend, format_output=False, source_backend="metal")


def _compile_metal_if_available(code, tmp_path):
    xcrun = shutil.which("xcrun")
    if platform.system() != "Darwin" or xcrun is None:
        return
    shader = tmp_path / "out.metal"
    shader.write_text(code)
    subprocess.run(
        [xcrun, "-sdk", "macosx", "metal", "-std=metal3.1", "-c", str(shader)],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )


def _kernel_body(code, name):
    match = re.search(rf"kernel void {name}\(.*?\n}}", code, re.DOTALL)
    assert match is not None, code
    return match.group(0)


def _step_callee(body):
    match = re.search(r"const uint step = (\w+)\(", body)
    assert match is not None, body
    return match.group(1)


def test_default_argument_of_an_explicit_instantiation_takes_the_helper_value(tmp_path):
    code = _translate(tmp_path, EXPLICIT_INSTANTIATION)

    f32 = _step_callee(_kernel_body(code, "affine_f32"))
    f16 = _step_callee(_kernel_body(code, "affine_f16"))
    assert re.search(rf"uint {f32}\(uint x\) {{\s*return x / 2 \+", code), code
    assert re.search(rf"uint {f16}\(uint x\) {{\s*return x / 4 \+", code), code
    assert "sizeof(T)" not in code
    _compile_metal_if_available(code, tmp_path)


def test_default_argument_of_a_called_template_takes_the_helper_value(tmp_path):
    code = _translate(tmp_path, CALL_SITE)

    assert "template" not in code
    assert "sizeof(T)" not in code
    bodies = {
        name: re.search(rf"void {name}\(.*?const uint step = (\w+)\(", code, re.DOTALL)
        for name in ("where_cond_half_uint32_t", "where_cond_float_uint8_t")
    }
    assert all(bodies.values()), code
    half_step = bodies["where_cond_half_uint32_t"].group(1)
    float_step = bodies["where_cond_float_uint8_t"].group(1)
    assert re.search(rf"uint {half_step}\(uint x\) {{\s*return x / 4 \+", code), code
    assert re.search(rf"uint {float_step}\(uint x\) {{\s*return x / 2 \+", code), code
    _compile_metal_if_available(code, tmp_path)


@pytest.mark.parametrize("backend", ["directx", "opengl"])
def test_helper_values_reach_source_targets(tmp_path, backend):
    code = _translate(tmp_path, CALL_SITE, backend)

    assert "sizeof(T)" not in code
    assert re.search(r"x / 4\b", code), code
    assert re.search(r"x / 2\b", code), code


def test_helper_values_reach_spirv(tmp_path):
    code = _translate(tmp_path, CALL_SITE, "vulkan")

    assert code.count("OpEntryPoint GLCompute") == 2


def test_type_templated_helper_folds_in_a_unique_callee_argument(tmp_path):
    code = _translate(tmp_path, UNIQUE_CALLEE)

    body = _kernel_body(code, "k")
    first, second = re.findall(r"= (\w+)\(tid\)", body)
    # 16 / sizeof(half) is 8 and 16 / sizeof(packed_float3) is 1.
    assert re.search(rf"uint {first}\(uint x\) {{\s*return x \* 8;", code), code
    assert re.search(rf"uint {second}\(uint x\) {{\s*return x \* 1;", code), code
    assert "lanes" not in code
    _compile_metal_if_available(code, tmp_path)


HELPERS = """
template<int N>
constexpr int pick() { return N; }
template<typename T>
constexpr int pick() { return int(sizeof(T)) * 100; }
template<typename T, typename U = T>
constexpr uint ratio() { return sizeof(T) / sizeof(U); }
struct S { float a; };
template<typename T> constexpr uint lanes() { return 16 / sizeof(T); }
kernel void k() {}
"""


@pytest.mark.parametrize(
    "expression, expected",
    [
        # The argument's kind selects the overload: a value for `int N`, a type for `typename T`.
        ("pick<3>() + pick<float>()", "3 + 400"),
        ("pick<half2>()", "400"),
        ("ratio<float4, half>()", "8"),
        ("ratio<float>()", "1"),
    ],
)
def test_preprocessor_evaluates_type_template_helpers(expression, expected):
    preprocessor = MetalPreprocessor()
    functions = preprocessor._materialization_constexpr_function_index(HELPERS)
    resolved, _evaluation, unresolved = preprocessor._resolve_static_constexpr_calls(
        expression, functions, HELPERS.find("kernel")
    )
    assert (resolved, unresolved) == (expected, ())


@pytest.mark.parametrize(
    "expression", ["lanes<T>()", "lanes<S>()", "lanes<float, int>()"]
)
def test_dependent_aggregate_or_mismatched_type_arguments_stay_unresolved(expression):
    preprocessor = MetalPreprocessor()
    functions = preprocessor._materialization_constexpr_function_index(HELPERS)
    resolved, _evaluation, unresolved = preprocessor._resolve_static_constexpr_calls(
        expression, functions, HELPERS.find("kernel")
    )
    assert resolved == expression
    assert unresolved == ("lanes",)

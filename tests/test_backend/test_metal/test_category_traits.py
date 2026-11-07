"""Metal's is_scalar, is_vector and is_arithmetic traits select constrained overloads."""

import platform
import re
import shutil
import subprocess

import pytest

import crosstl
from crosstl.backend.Metal.category_traits import evaluate_category_trait
from crosstl.backend.Metal.preprocessor import MetalPreprocessor

# (operand, is_scalar, is_vector, is_arithmetic), as `xcrun metal -std=metal3.1` evaluates them.
DECIDED = (
    ("int", True, False, True),
    ("const int", True, False, True),
    ("bool", True, False, True),
    ("half", True, False, True),
    ("bfloat", True, False, True),
    ("float4", False, True, True),
    ("uchar2", False, True, True),
    ("vec<int, 3>", False, True, True),
    ("packed_float3", False, True, True),
    ("device float*", True, False, False),
    ("device int*", True, False, False),
    ("float2x2", False, False, False),
    ("thread int&", False, False, False),
    ("constant float&", False, False, False),
)


@pytest.mark.parametrize("operand,scalar,vector,arithmetic", DECIDED)
def test_category_traits_match_the_metal_library(operand, scalar, vector, arithmetic):
    preprocessor = MetalPreprocessor()
    assert evaluate_category_trait(preprocessor, "is_scalar", operand) is scalar
    assert evaluate_category_trait(preprocessor, "is_vector", operand) is vector
    assert evaluate_category_trait(preprocessor, "is_arithmetic", operand) is arithmetic


@pytest.mark.parametrize("operand", ("Custom", "E", "double", "atomic_int"))
def test_named_and_unsupported_operands_are_not_decided(operand):
    # An enumeration is a scalar and a class is not; the operand alone cannot tell.
    for trait in ("is_scalar", "is_vector", "is_arithmetic"):
        assert evaluate_category_trait(MetalPreprocessor(), trait, operand) is None


def _metal_compiler():
    if platform.system() != "Darwin" or not shutil.which("xcrun"):
        return None
    probe = subprocess.run(
        ["xcrun", "-sdk", "macosx", "metal", "--version"],
        capture_output=True,
        text=True,
        check=False,
    )
    return ["xcrun", "-sdk", "macosx", "metal"] if probe.returncode == 0 else None


@pytest.mark.skipif(
    _metal_compiler() is None, reason="the Metal compiler is not installed"
)
def test_the_metal_compiler_agrees(tmp_path):
    lines = ["#include <metal_stdlib>", "using namespace metal;"]
    for operand, scalar, vector, arithmetic in DECIDED:
        for trait, value in (
            ("is_scalar_v", scalar),
            ("is_vector_v", vector),
            ("is_arithmetic_v", arithmetic),
        ):
            expected = "true" if value else "false"
            lines.append(
                f'static_assert({trait}<{operand}> == {expected}, "{trait}<{operand}>");'
            )
    lines.append("kernel void probe_kernel() {}")
    source = tmp_path / "traits.metal"
    source.write_text("\n".join(lines) + "\n")
    compiled = subprocess.run(
        [
            *_metal_compiler(),
            "-std=metal3.1",
            "-c",
            str(source),
            "-o",
            str(tmp_path / "t.air"),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert compiled.returncode == 0, compiled.stderr[-2000:]


SAFE_MOD = """#include <metal_stdlib>
using namespace metal;
template <typename T>
constexpr constant bool is_scalar_integral_v =
    ::metal::is_integral_v<T> && ::metal::is_scalar_v<T>;
template <typename U, typename V>
using common_dtype = decltype(U(0) + V(0));
template <typename T, typename U,
          ::metal::enable_if_t<is_scalar_integral_v<T> && is_scalar_integral_v<U>, bool> = true>
inline common_dtype<T, U> safe_mod(T x, U y) { return x % y; }
kernel void k(device uint* out [[buffer(0)]], constant uint& size [[buffer(1)]],
              uint i [[thread_position_in_grid]]) {
  int linear_idx = int(i);
  out[i] = safe_mod(linear_idx, size);
}
"""

CATEGORY_OVERLOADS = """#include <metal_stdlib>
using namespace metal;
template <typename T, enable_if_t<is_scalar_v<T>, bool> = true>
inline float pick(T x) { return 1.0f; }
template <typename T, enable_if_t<is_vector_v<T>, bool> = true>
inline float pick(T x) { return 2.0f; }
kernel void k(device float* out [[buffer(0)]], uint i [[thread_position_in_grid]]) {
  out[i] = pick(float(i)) + pick(float4(1.0f)) * 10.0f;
}
"""


@pytest.mark.parametrize("backend", ("directx", "vulkan", "opengl"))
def test_user_trait_built_from_is_scalar_enables_an_overload(tmp_path, backend):
    path = tmp_path / "safe_mod.metal"
    path.write_text(SAFE_MOD)
    output = crosstl.translate(str(path), backend=backend, format_output=False)
    assert output


def test_scalar_and_vector_overloads_select_their_bodies(tmp_path):
    path = tmp_path / "pick.metal"
    path.write_text(CATEGORY_OVERLOADS)
    output = crosstl.translate(str(path), backend="directx", format_output=False)
    assert re.search(r"float pick_float\(float x\) \{\s*return 1\.0f?;", output)
    assert re.search(r"float pick_float4\(float4 x\) \{\s*return 2\.0f?;", output)
    assert "pick_float(" in output and "pick_float4(float4(" in output

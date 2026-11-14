"""Metal's half and bfloat predefined constants are imported as exact conversions."""

import platform
import shutil
import subprocess
from fractions import Fraction

import pytest

import crosstl
from crosstl.backend.Metal.narrow_float_constants import (
    _CONSTANTS,
    narrow_float_constant,
    round_to_format,
)


@pytest.mark.parametrize(
    "name,metal_type,value",
    (
        ("HALF_MAX", "half", Fraction(65504)),
        ("MAXHALF", "half", Fraction(65504)),
        ("HALF_MIN", "half", Fraction(1, 2**14)),
        ("HALF_EPSILON", "half", Fraction(1, 2**10)),
        ("M_PI_H", "half", Fraction("3.140625")),
        ("M_E_H", "half", Fraction("2.71875")),
        ("M_SQRT2_H", "half", Fraction("1.4140625")),
        ("BFLT_MAX", "bfloat", (2 - Fraction(1, 2**7)) * 2**127),
        ("BFLT_MIN", "bfloat", Fraction(1, 2**126)),
        ("BFLT_EPSILON", "bfloat", Fraction(1, 2**7)),
        ("M_PI_BF", "bfloat", Fraction("3.140625")),
    ),
)
def test_constants_have_their_rounded_values(name, metal_type, value):
    constant = narrow_float_constant(name)
    assert constant.metal_type == metal_type
    assert Fraction(float(constant.literal)) == value


def test_rounding_is_to_nearest_with_ties_to_even():
    # 2049 lies halfway between the half values 2048 and 2050; 2048 is even.
    assert round_to_format("2049", "half") == 2048
    assert round_to_format("2051", "half") == 2052
    assert round_to_format("2049.0001", "half") == 2050
    with pytest.raises(ValueError):
        round_to_format("65520", "half")


def test_integer_limits_are_int_constants():
    assert narrow_float_constant("HALF_DIG").literal == "3"
    assert narrow_float_constant("HALF_MIN_EXP").literal == "(-13)"
    assert narrow_float_constant("BFLT_MAX_EXP").metal_type == "int"
    assert narrow_float_constant("FLT_MAX") is None


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
    for name, constant in sorted(_CONSTANTS.items()):
        expected = (
            constant.literal
            if constant.metal_type == "int"
            else f"{constant.metal_type}({constant.literal}f)"
        )
        lines.append(f'static_assert({name} == {expected}, "{name}");')
    lines.append("kernel void probe_kernel() {}")
    source = tmp_path / "constants.metal"
    source.write_text("\n".join(lines) + "\n")
    compiled = subprocess.run(
        [
            *_metal_compiler(),
            "-std=metal3.1",
            "-c",
            str(source),
            "-o",
            str(tmp_path / "c.air"),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert compiled.returncode == 0, compiled.stderr[-2000:]


KERNEL = """#include <metal_stdlib>
using namespace metal;
kernel void k(device half* out [[buffer(0)]], device bfloat* wide [[buffer(1)]],
              device int* digits [[buffer(2)]], uint i [[thread_position_in_grid]]) {
  out[i] = HALF_MAX + M_PI_H + HALF_EPSILON + HALF_MIN;
  wide[i] = M_PI_BF * BFLT_EPSILON;
  digits[i] = HALF_DIG + BFLT_MAX_EXP;
}
"""

NAMES = (
    "HALF_MAX",
    "M_PI_H",
    "HALF_EPSILON",
    "HALF_MIN",
    "M_PI_BF",
    "BFLT_EPSILON",
    "HALF_DIG",
    "BFLT_MAX_EXP",
)


@pytest.mark.parametrize("backend", ("directx", "opengl", "metal", "cgl"))
def test_targets_receive_values_not_names(tmp_path, backend):
    path = tmp_path / "constants.metal"
    path.write_text(KERNEL)
    output = crosstl.translate(str(path), backend=backend, format_output=False)
    for name in NAMES:
        assert name not in output
    assert "65504.0" in output and "3.140625" in output


def test_hlsl_converts_to_the_narrow_type(tmp_path):
    path = tmp_path / "constants.metal"
    path.write_text(KERNEL)
    output = crosstl.translate(str(path), backend="directx", format_output=False)
    assert "float16_t(65504.0)" in output
    assert "float16_t(3.140625)" in output


def test_a_variable_of_the_same_name_is_not_replaced(tmp_path):
    # After #undef, the name is free for a variable.
    path = tmp_path / "shadow.metal"
    path.write_text("""#include <metal_stdlib>
using namespace metal;
#undef HALF_MAX
kernel void k(device float* out [[buffer(0)]], uint i [[thread_position_in_grid]]) {
  float HALF_MAX = out[i];
  out[i] = HALF_MAX * 2.0f;
}
""")
    output = crosstl.translate(str(path), backend="metal", format_output=False)
    assert "65504" not in output


@pytest.mark.parametrize(
    "expression,expected",
    (
        ("FLT_MIN", "float"),
        ("M_PI_F", "float"),
        ("INFINITY", "float"),
        ("HALF_MAX", "half"),
        ("M_PI_H", "half"),
        ("HALF_DIG", "int"),
        ("x * FLT_MAX", "float"),
        ("M_PI", None),  # C's double constant; Metal does not define it
    ),
)
def test_predefined_constants_type_call_arguments(expression, expected):
    from crosstl.backend.Metal.preprocessor import MetalPreprocessor

    inferred = MetalPreprocessor()._infer_argument_type(expression, {}, {"x": "float"})
    assert inferred == expected


def test_a_local_of_the_same_name_types_the_argument():
    from crosstl.backend.Metal.preprocessor import MetalPreprocessor

    assert (
        MetalPreprocessor()._infer_argument_type("FLT_MIN", {}, {"FLT_MIN": "half"})
        == "half"
    )


def test_constrained_call_with_a_predefined_constant_argument_resolves(tmp_path):
    path = tmp_path / "clamp.metal"
    path.write_text("""#include <metal_stdlib>
using namespace metal;
template <typename T, enable_if_t<is_floating_point_v<T>, bool> = true>
inline T floor_at(T value, T low) { return value < low ? low : value; }
kernel void k(device float* out [[buffer(0)]], uint i [[thread_position_in_grid]]) {
  float result = out[i];
  out[i] = floor_at(result, FLT_MIN);
}
""")
    output = crosstl.translate(str(path), backend="directx", format_output=False)
    assert (
        "floor_at_float(result, 1.17549435082228750796873653722224568e-38f)" in output
    )

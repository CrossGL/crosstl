"""Constructors type their owner's unqualified static constants.

A constructor's initializers and body may name the owner's static constexpr
members without qualification. Arithmetic profiles need the computation type
of every such expression; before the fix the static members of a lowered
constructor resolved to no type, so an integer division in an initializer
failed closed under a division profile while the same expression in a method
translated.
"""

import re

import pytest

from crosstl import translate
from tests.test_translator.test_metal_builtin_ownership import _compile

SOURCE = """#include <metal_stdlib>
using namespace metal;

template <typename T, short ROWS>
struct Walker {
  static constant constexpr const short pack = 2;
  static constant constexpr const short bytes = 1;
  const int ld;
  const int stride;
  float scale = 1.0f;
  Walker(const int ld_, const device float* factor) thread
      : ld(ld_), stride(ROWS * ld * bytes / pack) {
    scale = *factor / float(pack * ROWS);
  }
  int step() const thread { return ROWS * ld * bytes / pack; }
};

kernel void walk(const device float* factor [[buffer(0)]],
                 device float* y [[buffer(1)]],
                 uint i [[thread_position_in_grid]]) {
  Walker<float, 4> walker(int(i) + 3, factor);
  y[i] = float(walker.stride + walker.step()) / walker.scale;
}
"""


def _translate(tmp_path, target, profile):
    path = tmp_path / "walker.metal"
    path.write_text(SOURCE, encoding="utf-8")
    options = {"binary32_division_profile": profile} if profile else {}
    return translate(
        str(path), backend=target, format_output=False, source_options=options
    )


@pytest.mark.parametrize("target", ["directx", "opengl", "metal"])
@pytest.mark.parametrize("profile", [None, "rne-flush"])
def test_constructor_initializers_type_static_members(tmp_path, target, profile):
    generated = _translate(tmp_path, target, profile)
    # The integer stride stays an integer division in the constructor.
    (stride,) = re.findall(r"crosstl_ctor_value\.stride = (.*);", generated)
    assert "divide_float" not in stride
    assert re.search(r"\b4\b", stride) and re.search(r"\b2\b", stride), stride
    # The float divisions take the profile's helper, and only under it.
    helpers = re.findall(r"metal_divide_float\(", generated)
    assert bool(helpers) == (profile is not None)
    _compile(
        generated, target, tmp_path, directx_compile_flags=("-enable-16bit-types",)
    )

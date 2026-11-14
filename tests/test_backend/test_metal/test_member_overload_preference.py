"""A non-template member beats a member template when conversions tie.

[over.match.best]/2.4 prefers a non-template function over a function
template specialization when no argument converts better for either. A
conversion to a parameter that has the same non-dependent type in both
candidates, such as ``ushort`` to ``uint``, is identical for both, so it must
not hand the call to the template.
"""

import shutil

import pytest

import crosstl
from tests.test_translator.test_metal_builtin_ownership import _compile

_SOURCE = """#include <metal_stdlib>
using namespace metal;

struct Scaler {
  float factor;

  template <typename T>
  T apply(const device T* values, uint index) const {
    return values[index];
  }

  float apply(const device float* values, uint index) const {
    return values[index] * factor;
  }
};

kernel void scale(const device float* values [[buffer(0)]],
                  const device half* halves [[buffer(1)]],
                  device float* out [[buffer(2)]],
                  uint i [[thread_position_in_grid]]) {
  Scaler scaler{2.0f};
  ushort lane = ushort(i);
  out[i] = scaler.apply(values, lane);
  out[i + 64u] = float(scaler.apply(halves, lane));
}
"""

TOOLS = {"metal": "xcrun", "directx": "dxc", "opengl": "glslangValidator"}


def _translate(tmp_path, target):
    path = tmp_path / "scale.metal"
    path.write_text(_SOURCE, encoding="utf-8")
    return crosstl.translate(str(path), backend=target, format_output=False)


def test_non_template_member_wins_tied_conversions(tmp_path):
    crossgl = _translate(tmp_path, "cgl")
    # The float call uses the non-template; the half call still needs the template.
    assert "Scaler__apply(scaler, values, lane)" in crossgl
    assert "Scaler__apply__half(scaler, halves, lane)" in crossgl
    assert "Scaler__apply__float" not in crossgl


@pytest.mark.parametrize("target", ("metal", "directx", "opengl"))
def test_non_template_member_preference_compiles(tmp_path, target):
    generated = _translate(tmp_path, target)
    assert "factor" in generated
    if shutil.which(TOOLS[target]):
        _, module = _compile(generated, target, tmp_path)
        assert module.is_file() and module.stat().st_size

"""Template arguments deduce through explicit pointer conversions.

``reinterpret_cast<const device float*>(bits)`` and the C-style cast
``(const device float*)bits`` have the target pointer type
([expr.reinterpret.cast]/1, [expr.cast]/1). An ``auto`` or ``auto*`` local
initialized from one takes that type ([dcl.type.auto.deduct]), so a later
template call deduces ``T = float`` from it, for free functions and for member
templates alike. Casts that do not span the whole expression, multi-level
pointers and pointers without an explicit address space stay un-inferable.
"""

import pytest

from crosstl._crosstl import translate
from crosstl.backend.Metal.auto_placeholders import deduced_auto_type
from crosstl.backend.Metal.pointer_conversions import pointer_conversion_type
from crosstl.backend.Metal.preprocessor import MetalPreprocessor
from tests.test_translator.test_metal_builtin_ownership import _compile

FREE_SOURCE = """#include <metal_stdlib>
using namespace metal;

template <typename T>
T get(const device T* values, uint offset) {
  return values[offset];
}

kernel void load_view(const device uint* bits [[buffer(0)]],
                      device float* out [[buffer(1)]],
                      uint i [[thread_position_in_grid]]) {
  DECLARATION;
  out[i] = get(view, i) * 2.0f;
}
"""

MEMBER_SOURCE = """#include <metal_stdlib>
using namespace metal;

struct Reader {
  uint scale;

  template <typename T>
  T get(const device T* values, uint offset) const {
    return values[offset] * T(scale);
  }
};

kernel void load_view(const device uint* bits [[buffer(0)]],
                      device float* out [[buffer(1)]],
                      uint i [[thread_position_in_grid]]) {
  Reader reader{2u};
  DECLARATION;
  out[i] = reader.get(view, i);
  out[i + 64u] = reader.get(reinterpret_cast<const device float*>(bits), i);
}
"""

DECLARATIONS = (
    "auto view = reinterpret_cast<const device float*>(bits)",
    "auto* view = reinterpret_cast<const device float*>(bits)",
    "auto view = (const device float*)bits",
    "auto view = (const device float*)(bits)",
)


def _translate(tmp_path, source, backend):
    path = tmp_path / "pointer_conversion.metal"
    path.write_text(source, encoding="utf-8")
    return translate(
        str(path), backend=backend, source_backend="metal", format_output=False
    )


@pytest.mark.parametrize("declaration", DECLARATIONS)
def test_auto_pointer_conversion_deduces_free_template(tmp_path, declaration):
    source = FREE_SOURCE.replace("DECLARATION", declaration)
    generated = _translate(tmp_path, source, "opengl")
    assert "float get_float" in generated
    assert "uintBitsToFloat" in generated
    _compile(generated, "opengl", tmp_path)


@pytest.mark.parametrize("declaration", DECLARATIONS)
def test_auto_pointer_conversion_round_trips_to_metal(tmp_path, declaration):
    source = FREE_SOURCE.replace("DECLARATION", declaration)
    generated = _translate(tmp_path, source, "metal")
    assert "get_float(view, i)" in " ".join(generated.split())
    _compile(generated, "metal", tmp_path)


@pytest.mark.parametrize("declaration", DECLARATIONS)
def test_pointer_conversion_deduces_member_template(tmp_path, declaration):
    source = MEMBER_SOURCE.replace("DECLARATION", declaration)
    generated = _translate(tmp_path, source, "opengl")
    compact = " ".join(generated.split())
    assert compact.count("float Reader_get_float") == 2
    assert compact.count("= Reader_get_float") == 2
    _compile(generated, "opengl", tmp_path)


def test_member_pointer_conversion_round_trips_to_metal(tmp_path):
    source = MEMBER_SOURCE.replace("DECLARATION", DECLARATIONS[1])
    generated = _translate(tmp_path, source, "metal")
    _compile(generated, "metal", tmp_path)


@pytest.mark.parametrize(
    ("expression", "expected"),
    (
        ("reinterpret_cast<device Box<float>*>(out)", "device Box<float>*"),
        ("reinterpret_cast<const device float*>(bits)", "const device float*"),
        ("reinterpret_cast < threadgroup uint* > ( tile )", "threadgroup uint*"),
        ("(device float*)out", "device float*"),
        ("(const constant uint*)(bits)", "const constant uint*"),
        ("reinterpret_cast<device float*>(out) + 1", None),
        ("reinterpret_cast<device float*>(out)[0]", None),
        ("reinterpret_cast<device float*>()", None),
        ("reinterpret_cast<float*>(out)", None),
        ("reinterpret_cast<device float**>(out)", None),
        ("reinterpret_cast<device float&>(out)", None),
        ("reinterpret_cast<uint>(out)", None),
        ("(device float*)out + 1", None),
        ("static_cast<device float*>(out)", None),
    ),
)
def test_pointer_conversion_type_requires_whole_addressed_pointer_cast(
    expression, expected
):
    assert pointer_conversion_type(MetalPreprocessor(), expression) == expected


@pytest.mark.parametrize(
    ("declared", "initializer", "expected"),
    (
        ("auto", "float", "float"),
        ("auto", None, "auto"),
        ("auto*", "device float*", "device float*"),
        ("auto *", "const device float*", "const device float*"),
        ("const auto*", "device float*", "const device float*"),
        ("const auto*", "device const float*", "device const float*"),
        ("auto*", "float", "auto"),
        ("auto*", "device float**", "auto"),
        ("auto*", None, "auto"),
        ("auto&", "float", "auto"),
        ("device auto*", "device float*", "auto"),
    ),
)
def test_auto_placeholders_take_only_deducible_types(declared, initializer, expected):
    assert deduced_auto_type(declared, initializer) == expected

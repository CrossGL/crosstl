"""A type trait in a struct method resolves the owner's member type aliases.

Lowering a method to a free function leaves the struct's type scope, so every
type spelling in the body that names an owner alias must be resolved first.
Declarations, casts and explicit template-call arguments already were; the
operands of ``is_same_v<scalar_T, float>`` and ``is_same<scalar_T, half>::value``
were not, so the concrete trait could not be folded and the kernel was refused
with ``MetalTypeTraitError`` although every operand was known.
"""

import re

import pytest

import crosstl

_SOURCE = """
#include <metal_stdlib>
using namespace metal;

template <typename storage_T>
struct StorageTraits {
  using scalar_T = storage_T;
};

template <typename T>
struct StorageTraits<vec<T, 2>> {
  using scalar_T = T;
};

template <typename in_T, typename out_T>
struct IOTraits {
  using scalar_T = typename StorageTraits<in_T>::scalar_T;
};

template <typename in_T, typename out_T>
struct Writer {
  using scalar_T = typename IOTraits<in_T, out_T>::scalar_T;
  using complex_T = vec<scalar_T, 2>;
  int scale;

  METAL_FUNC complex_T write() const thread {
    complex_T value = {scalar_T(scale), -scalar_T(scale)};
    if constexpr (!metal::is_same_v<scalar_T, float>) {
      value = {scalar_T(37), -scalar_T(37)};
    }
    if constexpr (metal::is_same<scalar_T, half>::value) {
      value.y = scalar_T(59);
    }
    return value;
  }
};

kernel void write_float(device float2* out [[buffer(0)]],
                        uint i [[thread_position_in_grid]]) {
  Writer<float2, float2> writer = {1};
  out[i] = writer.write();
}

kernel void write_half(device half2* out [[buffer(0)]],
                       uint i [[thread_position_in_grid]]) {
  Writer<half2, half2> writer = {1};
  out[i] = writer.write();
}
"""


def _translate(tmp_path, backend):
    source = tmp_path / "writer.metal"
    source.write_text(_SOURCE)
    return crosstl.translate(
        str(source), backend=backend, source_backend="metal", format_output=False
    )


def _function_body(output, name):
    definition = re.search(rf"\w[\w<>]*\s+{name}\s*\([^)]*\)[^;{{]*\{{", output)
    assert definition is not None, f"no definition of {name}"
    open_brace = definition.end() - 1
    depth = 0
    for index in range(open_brace, len(output)):
        if output[index] == "{":
            depth += 1
        elif output[index] == "}":
            depth -= 1
            if depth == 0:
                return output[open_brace : index + 1]
    raise AssertionError(f"unterminated body for {name}")


@pytest.mark.parametrize("backend", ["metal", "cgl", "directx"])
def test_owner_alias_trait_operands_select_each_instantiation(tmp_path, backend):
    output = _translate(tmp_path, backend)

    assert "is_same" not in output
    float_body = _function_body(output, "Writer_float2_float2__write")
    half_body = _function_body(output, "Writer_half2_half2__write")
    # float: the `!is_same_v<float, float>` and `is_same<float, half>` branches are dropped.
    assert "37" not in float_body
    assert "59" not in float_body
    # half: both branches are kept, with the alias resolved to the concrete scalar.
    assert "37" in half_body
    assert "59" in half_body
    assert "scalar_T" not in float_body
    assert "scalar_T" not in half_body


def test_metal_output_names_no_owner_alias(tmp_path):
    output = _translate(tmp_path, "metal")

    assert "scalar_T" not in output
    assert "complex_T" not in output

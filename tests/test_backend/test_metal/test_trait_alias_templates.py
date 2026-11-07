"""Alias templates as type-trait operands.

``metal::is_same_v<StrY, Int<1>>`` with ``StrY`` deduced as ``Int<1>`` used to
be refused because ``Int`` is an alias template, not a known type. The alias
now expands to the template-id it denotes before the trait is evaluated.
"""

import re

import pytest

import crosstl
from crosstl.backend.Metal.preprocessor import MetalPreprocessor, MetalTypeTraitError
from crosstl.backend.Metal.trait_alias_templates import (
    expand_trait_alias_templates,
    find_trait_alias_templates,
)

ALIASES = """
template <typename T, T v>
struct integral_const {
  static constexpr constant T value = v;
};
template <int val>
using Int = integral_const<int, val>;
template <typename A, typename B>
using Swap = pair<B, A>;
namespace lib {
template <typename T>
using Pair2 = vec<T, 2>;
}
namespace other {
template <typename T>
using Pair2 = vec<T, 3>;
}
template <int... vs>
using Seq = seq<vs...>;
template <int v = 1>
using Defaulted = integral_const<int, v>;
struct Holder {
  template <typename T>
  using Self = T;
};
"""


def _expand(text):
    preprocessor = MetalPreprocessor()
    aliases = find_trait_alias_templates(preprocessor, ALIASES)
    return expand_trait_alias_templates(preprocessor, text, aliases)


def test_find_trait_alias_templates_keeps_plain_namespace_aliases_only():
    aliases = find_trait_alias_templates(MetalPreprocessor(), ALIASES)

    assert sorted(aliases) == ["Int", "Pair2", "Swap"]
    (int_alias,) = aliases["Int"]
    assert int_alias.parameters == ("val",)
    assert int_alias.target == "integral_const<int, val>"
    assert int_alias.namespace == ""
    assert [alias.namespace for alias in aliases["Pair2"]] == ["lib", "other"]


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Int<1>", "integral_const<int, 1>"),
        ("Int< 16 >", "integral_const<int, 16>"),
        ("Int<(4 > 2)>", "integral_const<int, (4 > 2)>"),
        ("Swap<Int<(1 >> 1)>, float>", "pair<float, integral_const<int, (1 >> 1)>>"),
        ("Swap<Int<2>, float>", "pair<float, integral_const<int, 2>>"),
        ("lib::Pair2<float>", "vec<float, 2>"),
        ("::other::Pair2<half>", "vec<half, 3>"),
        # Unqualified Pair2 is ambiguous between lib and other.
        ("Pair2<float>", "Pair2<float>"),
        ("third::Pair2<float>", "third::Pair2<float>"),
        ("Int<1, 2>", "Int<1, 2>"),
        ("Defaulted<2>", "Defaulted<2>"),
        ("Seq<1, 2>", "Seq<1, 2>"),
        ("Holder::Self<int>", "Holder::Self<int>"),
        ("MyInt<1>", "MyInt<1>"),
    ],
)
def test_expand_trait_alias_templates(text, expected):
    assert _expand(text) == expected


ALIAS_TRAIT_KERNEL = """
#include <metal_stdlib>
using namespace metal;

namespace lib {
template <typename T, T v>
struct integral_const {
  static constexpr constant T value = v;
};
template <int val>
using Int = integral_const<int, val>;
template <typename T>
using Pair2 = vec<T, 2>;
}

template <typename T, typename S>
float pick(const device float* src, T, S) {
  float r = 0.0f;
  if constexpr (metal::is_same_v<S, lib::Pair2<float>>) {
    r += 100.0f;
  }
  if constexpr (metal::is_same_v<T, lib::Int<1>>) {
    return r + src[0];
  } else {
    return r + src[T::value];
  }
}

kernel void k(const device float* src [[buffer(0)]],
              device float* out [[buffer(1)]],
              uint tid [[thread_position_in_grid]]) {
  out[tid] = pick(src, lib::Int<1>{}, float2(0.0f)) + pick(src, lib::Int<3>{}, half2(0.0h));
}
"""


def _function_body(output, name):
    # GLSL specializations append buffer suffixes to the name.
    definition = list(re.finditer(rf"float {name}\w*\([^)]*\)\s*\{{", output))[-1]
    end = output.find("\n}", definition.end())
    return re.sub(r"\s+", " ", output[definition.end() : end])


@pytest.mark.parametrize("backend", ["directx", "opengl"])
def test_alias_template_trait_operands_select_each_branch(tmp_path, backend):
    source = tmp_path / "alias_traits.metal"
    source.write_text(ALIAS_TRAIT_KERNEL)

    output = crosstl.translate(
        str(source), backend=backend, source_backend="metal", format_output=False
    )

    first = _function_body(output, "pick_lib_Int_1_float2")
    second = _function_body(output, "pick_lib_Int_3_half2")
    # Int<1> matches Int<1>, Int<3> does not; float2 is Pair2<float>, half2 is not.
    assert "r += 100.0;" in first
    assert re.search(r"src\[(?:uint\()?src_offset\)?\]", first)
    assert "100.0" not in second
    assert re.search(r"src\[(?:uint\()?\(src_offset \+ 3\)\)?\]", second)
    assert "is_same" not in output


def test_alias_template_with_default_argument_still_refused(tmp_path):
    source = tmp_path / "defaulted.metal"
    source.write_text("""
    #include <metal_stdlib>
    using namespace metal;
    template <typename T, T v>
    struct integral_const {
      static constexpr constant T value = v;
    };
    template <int v = 1>
    using Defaulted = integral_const<int, v>;

    template <typename T>
    float pick(const device float* src, T) {
      if constexpr (metal::is_same_v<T, Defaulted<1>>) {
        return src[0];
      } else {
        return src[1];
      }
    }

    kernel void k(const device float* src [[buffer(0)]], device float* out [[buffer(1)]]) {
      out[0] = pick(src, Defaulted<1>{});
    }
    """)

    with pytest.raises(MetalTypeTraitError) as caught:
        crosstl.translate(str(source), backend="directx", source_backend="metal")

    assert "Defaulted<1>" in str(caught.value)

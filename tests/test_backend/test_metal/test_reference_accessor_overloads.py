"""Reference accessors overloaded with unnamed (tag) parameters.

A member accessor that returns a reference into the receiver's storage is
inlined as that storage. Overload selection used to count only the *named*
parameters of each overload, so an overload with an unnamed tag parameter,
such as ``metal::bool_constant<B>`` or a plain ``int``, looked like it took
the same arguments as the two-index accessor and every call was refused.
"""

import re

import pytest

import crosstl
from crosstl.backend.Metal.method_arity import (
    accepts_argument_count,
    parameter_arity,
)
from crosstl.backend.Metal.preprocessor import MetalPreprocessor, MetalStructMethodError


def _compact(text):
    """Drop whitespace and parentheses so index arithmetic compares by value."""
    return re.sub(r"[\s()]", "", text)


def _arity(parameters):
    preprocessor = MetalPreprocessor()
    return parameter_arity(
        parameters,
        preprocessor._split_top_level_commas,
        preprocessor._split_top_level_assignment,
    )


def _accepts(parameters, count):
    preprocessor = MetalPreprocessor()
    return accepts_argument_count(
        parameters,
        count,
        preprocessor._split_top_level_commas,
        preprocessor._split_top_level_assignment,
    )


@pytest.mark.parametrize(
    "parameters, expected",
    [
        ("", (0, 0)),
        ("void", (0, 0)),
        ("const short i, const short j", (2, 2)),
        ("const short i, const short j, metal::bool_constant<transpose>", (3, 3)),
        ("short i, int", (2, 2)),
        ("int a, int b = 1, int c = 2", (1, 3)),
        ("vec<float, 2> v, metal::integral_constant<int, 3> n = {}", (1, 2)),
    ],
)
def test_parameter_arity_counts_unnamed_parameters_and_defaults(parameters, expected):
    assert _arity(parameters) == expected


def test_accepts_argument_count_honours_defaults_and_packs():
    assert _accepts("int a, int b = 1", 1)
    assert _accepts("int a, int b = 1", 2)
    assert not _accepts("int a, int b = 1", 3)
    assert not _accepts("const short i, const short j, metal::bool_constant<t>", 2)
    assert _accepts("int a, T... rest", 1)
    assert _accepts("int a, T... rest", 4)
    assert not _accepts("int a, T... rest", 0)


TAGGED_TEMPLATE_TILE = """
#include <metal_stdlib>
using namespace metal;

template <typename T, int R, int C>
struct Tile {
  static constant constexpr int kTileCols = C;
  typedef vec<T, 2> frag_type;
  frag_type val_frags[R * C];

  constexpr thread frag_type& frag_at(const short i, const short j) thread {
    return val_frags[i * kTileCols + j];
  }
  constexpr const thread frag_type& frag_at(const short i, const short j) const thread {
    return val_frags[i * kTileCols + j];
  }
  template <bool transpose>
  constexpr thread frag_type& frag_at(
      const short i, const short j, metal::bool_constant<transpose>) thread {
    if constexpr (transpose) {
      return frag_at(j, i);
    } else {
      return frag_at(i, j);
    }
  }
};

kernel void k(device float2* out [[buffer(0)]], uint tid [[thread_position_in_grid]]) {
  Tile<float, 2, 2> t;
  for (short i = 0; i < 4; i++) {
    t.val_frags[i] = float2(float(i), float(tid));
  }
  t.frag_at(1, 0) *= 2.0f;
  t.frag_at(0, 1)[1] += 1.0f;
  out[tid] = t.frag_at(1, 0) + t.frag_at(0, 1);
}
"""


def test_two_index_accessor_beside_tag_template_overload_inlines_storage():
    output = MetalPreprocessor().preprocess(TAGGED_TEMPLATE_TILE)
    kernel = _compact(output.split("kernel void k", 1)[1])

    assert "t.val_frags[1*2+0]*=2.0f;" in kernel
    assert "t.val_frags[0*2+1][1]+=1.0f;" in kernel
    assert "out[tid]=t.val_frags[1*2+0]+t.val_frags[0*2+1];" in kernel
    assert "frag_at" not in kernel


@pytest.mark.parametrize("backend", ["directx", "opengl", "metal"])
def test_two_index_accessor_beside_tag_template_overload_translates(tmp_path, backend):
    source = tmp_path / "tile.metal"
    source.write_text(TAGGED_TEMPLATE_TILE)

    output = crosstl.translate(
        str(source), backend=backend, source_backend="metal", format_output=False
    )

    body = _compact(output[output.rfind("val_frags[i]") :])
    assert "frag_at" not in body
    assert "t.val_frags[1*2+0]*=2.0" in body
    assert "t.val_frags[0*2+1]" in body


VALUE_TAG_OVERLOAD = """
#include <metal_stdlib>
using namespace metal;

struct Tile {
  float2 val_frags[4];
  thread float2& frag_at(const short i, const short j) thread {
    return val_frags[i * 2 + j];
  }
  const thread float2& frag_at(const short i, const short j) const thread {
    return val_frags[i * 2 + j];
  }
  float2 frag_at(const short i, const short j, int) const thread {
    return val_frags[j * 2 + i];
  }
};

struct Holder {
  Tile tile;
};

kernel void k(device float2* out [[buffer(0)]], uint tid [[thread_position_in_grid]]) {
  Holder h;
  for (short i = 0; i < 4; i++) {
    h.tile.val_frags[i] = float2(float(i), float(tid));
  }
  h.tile.frag_at(0, 1) += float2(1.0f);
  Tile t = h.tile;
  t.frag_at(1, 1) = t.frag_at(0, 1, 7);
  out[tid] = t.frag_at(1, 0) + t.frag_at(1, 1);
}
"""


def test_reference_accessor_and_value_tag_overload_each_keep_their_meaning(tmp_path):
    source = tmp_path / "value_tag.metal"
    source.write_text(VALUE_TAG_OVERLOAD)

    output = crosstl.translate(
        str(source), backend="directx", source_backend="metal", format_output=False
    )

    body = output[output.find("void CSMain") :]
    # The two-index calls name the storage element itself, including through a
    # member receiver; the three-argument call is the by-value overload, which
    # reads the transposed element.
    assert "h.tile.val_frags[((0 * 2) + 1)] += float2(1.0, 1.0);" in body
    assert "t.val_frags[((1 * 2) + 1)] = Tile__frag_at(t, " in body
    assert "out_[tid] = (t.val_frags[((1 * 2) + 0)] + t.val_frags[((1 * 2) + 1)]);" in (
        body
    )
    free_function = output[output.find("Tile__frag_at(") : output.find("void CSMain")]
    assert "val_frags[intj*2+inti]" in _compact(free_function)


def test_same_arity_reference_overloads_still_fail_closed():
    code = """
    struct Tile {
      float2 val_frags[4];
      thread float2& frag_at(short i, short j) thread { return val_frags[i * 2 + j]; }
      thread float2& frag_at(int i, int j) thread { return val_frags[j * 2 + i]; }
    };

    kernel void k(device float2* out [[buffer(0)]]) {
      Tile t;
      t.frag_at(1, 0) = float2(1.0f);
      out[0] = t.val_frags[2];
    }
    """

    with pytest.raises(MetalStructMethodError) as caught:
        MetalPreprocessor().preprocess(code)

    assert caught.value.reason == "reference-return-identity-unsupported"


TEMPLATE_INDEX_ACCESSOR = """
#include <metal_stdlib>
using namespace metal;

void bump(thread float2& value) { value += float2(1.0f, 2.0f); }

struct Tile {
  static constant constexpr int kCols = 2;
  float2 frags[4];
  template <int i, int j>
  thread float2& frag_at() thread { return frags[i * kCols + j]; }
  template <int i, int j>
  const thread float2& frag_at() const thread { return frags[i * kCols + j]; }

  void update() thread {
    frag_at<0, 1>() *= 3.0f;
    bump(frag_at<1, 0>());
  }
  float2 diagonal() const thread { return frag_at<0, 0>() + frag_at<1, 1>(); }
};

kernel void k(device float2* out [[buffer(0)]], uint tid [[thread_position_in_grid]]) {
  Tile t;
  for (short i = 0; i < 4; i++) {
    t.frags[i] = float2(float(i), float(tid));
  }
  t.update();
  out[tid] = t.diagonal() + t.frags[1] + t.frags[2];
}
"""


def test_member_template_reference_accessor_inlines_as_storage():
    # frag_at<i, j>() names an element of `frags`: a compound assignment and a
    # `thread float2&` argument must reach that element, not a returned copy.
    output = _compact(MetalPreprocessor().preprocess(TEMPLATE_INDEX_ACCESSOR))

    assert "self.frags[0*2+1]*=3.0f;" in output
    assert "bumpself.frags[1*2+0];" in output
    assert "returnself.frags[0*2+0]+self.frags[1*2+1];" in output
    assert "frag_at" not in output.split("structTile", 1)[1].split("kernelvoid", 1)[0]
    assert "__runtime_values__" not in output


@pytest.mark.parametrize("backend", ["directx", "opengl", "metal"])
def test_member_template_reference_accessor_translates(tmp_path, backend):
    source = tmp_path / "template_accessor.metal"
    source.write_text(TEMPLATE_INDEX_ACCESSOR)

    output = _compact(
        crosstl.translate(
            str(source), backend=backend, source_backend="metal", format_output=False
        )
    )

    assert "frag_at" not in output
    assert "frags[0*2+1]*=3.0" in output
    assert "bumpself.frags[1*2+0];" in output


def test_member_template_reference_accessor_returned_by_reference_fails_closed():
    code = """
    struct Tile {
      float2 frags[4];
      template <int i, int j>
      thread float2& frag_at() thread { return frags[i * 2 + j]; }
      thread float2& corner() thread { return frag_at<0, 1>(); }
    };

    kernel void k(device float2* out [[buffer(0)]]) {
      Tile t;
      t.corner() = float2(1.0f);
      out[0] = t.frags[1];
    }
    """

    with pytest.raises(MetalStructMethodError) as caught:
        MetalPreprocessor().preprocess(code)

    assert caught.value.reason == "reference-return-identity-unsupported"

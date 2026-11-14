"""Member templates selected by a library tag argument.

``A.frag_at(i, j, ta)`` with ``template <bool transpose> T& frag_at(short,
short, metal::bool_constant<transpose>)`` needs four things the importer did
not do: type ``metal::bool_constant<true>{}`` temporaries and ``constexpr
auto`` tag locals as call arguments, prefer the overload whose constness
matches the receiver over its const twin, inline a reference-returning
template member as the storage it names, and fold the instantiated
``if constexpr (transpose)`` dispatch to the accessor it forwards to.
"""

import re
import shutil
import subprocess

import pytest

import crosstl
from crosstl.backend.Metal.forwarding_reference_accessors import (
    selected_constexpr_body,
)
from crosstl.backend.Metal.method_arity import named_parameter_arguments
from crosstl.backend.Metal.preprocessor import MetalPreprocessor, MetalStructMethodError

TILE = """
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
  template <bool transpose>
  constexpr const thread frag_type& frag_at(
      const short i, const short j, metal::bool_constant<transpose>) const thread {
    if constexpr (transpose) {
      return frag_at(j, i);
    } else {
      return frag_at(i, j);
    }
  }
  template <bool transpose>
  float2 get(const short i, const short j, metal::bool_constant<transpose>) const thread {
    return transpose ? val_frags[j * C + i] : val_frags[i * C + j];
  }
};

template <class ATile, bool transpose_a>
float2 take(thread ATile& A, metal::bool_constant<transpose_a>) {
  constexpr auto ta = metal::bool_constant<transpose_a>{};
  A.frag_at(1, 0, ta) += float2(100.0f);
  return A.frag_at(0, 1, ta);
}

template <class ATile, bool transpose_a>
float2 peek(thread const ATile& A, metal::bool_constant<transpose_a>) {
  constexpr auto ta = metal::bool_constant<transpose_a>{};
  return A.frag_at(0, 1, ta) + A.get(1, 0, ta);
}

kernel void k(device float2* out [[buffer(0)]], uint tid [[thread_position_in_grid]]) {
  Tile<float, 2, 2> t;
  for (short i = 0; i < 4; i++) {
    t.val_frags[i] = float2(float(i), float(tid));
  }
  t.frag_at(1, 1, metal::bool_constant<true>()) *= 2.0f;
  out[tid] = take(t, metal::bool_constant<true>{})
      + peek(t, metal::bool_constant<false>{}) * 10.0f;
}
"""


def _compact(text):
    return re.sub(r"[\s()]", "", text)


def _body(output, name):
    definition = list(re.finditer(rf"\b{name}\w*\([^)]*\)\s*\{{", output))[-1]
    return _compact(output[definition.end() : output.find("\n}", definition.end())])


@pytest.mark.parametrize("backend", ["directx", "opengl"])
def test_tag_selected_reference_members_inline_the_transposed_storage(
    tmp_path, backend
):
    source = tmp_path / "tile.metal"
    source.write_text(TILE)

    output = crosstl.translate(
        str(source), backend=backend, source_backend="metal", format_output=False
    )

    take = _body(output, "take_Tile_float_2_2_true")
    peek = _body(output, "peek_Tile_float_2_2_false")
    # transpose = true swaps the indices: frag_at(1, 0) is element 0 * 2 + 1.
    assert "A.val_frags[0*2+1]+=" in take
    assert "returnA.val_frags[1*2+0];" in take
    # transpose = false keeps them; the by-value tag overload is a call.
    assert "A.val_frags[0*2+1]" in peek
    assert re.search(r"Tile_float_2_2_+get_+false\w*A,", peek)
    assert "t.val_frags[1*2+1]*=2.0" in _compact(output)
    assert "frag_at" not in output
    glslang = shutil.which("glslangValidator")
    if backend == "opengl" and glslang is not None:
        shader = tmp_path / "tile.comp"
        shader.write_text(output)
        result = subprocess.run(
            [glslang, "-S", "comp", str(shader)], capture_output=True, text=True
        )
        assert result.returncode == 0, result.stdout + result.stderr


def test_reference_template_member_that_names_no_storage_still_fails_closed():
    code = """
    struct State {
        template <typename T>
        thread T& element(thread T& value, metal::bool_constant<true>) { return value; }
    };
    void assign(thread State& state) {
        int value = 0;
        state.element(value, metal::bool_constant<true>{}) = 1;
    }
    """

    with pytest.raises(MetalStructMethodError) as caught:
        MetalPreprocessor().preprocess("using namespace metal;\n" + code)

    assert caught.value.reason == "reference-return-identity-unsupported"
    assert caught.value.requested_signature.startswith("state.element(")


def test_unnamed_argument_with_effects_is_not_dropped():
    preprocessor = MetalPreprocessor()

    def bind(arguments):
        return named_parameter_arguments(
            "short i, metal::bool_constant<t>",
            arguments,
            preprocessor._split_top_level_commas,
            preprocessor._declared_data_member_name,
        )

    assert bind(["1", "metal::bool_constant<true>{}"]) == ["1"]
    assert bind(["1", "ta"]) == ["1"]
    assert bind(["1", "make_tag(count)"]) is None
    assert bind(["1"]) is None


@pytest.mark.parametrize(
    "body, expected",
    [
        (
            "if constexpr (true) { return a(j, i); } else { return a(i, j); }",
            "return a(j, i);",
        ),
        (
            "if constexpr (1 > 2) { return a(j, i); } else { return a(i, j); }",
            "return a(i, j);",
        ),
        (
            "{ if constexpr (false) { return x; } else if constexpr (true) { return y; } }",
            "return y;",
        ),
        # Not decidable, or more than one statement: unchanged.
        ("if constexpr (B) { return x; } else { return y; }", None),
        ("if constexpr (true) { return x; } return y;", None),
    ],
)
def test_selected_constexpr_body(body, expected):
    result = selected_constexpr_body(MetalPreprocessor(), body)
    assert result.strip() == (expected if expected is not None else body)

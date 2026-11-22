"""Static member calls whose owner is a qualified struct member alias.

``Tile<float, 1, 2>::Frag_t::mma(...)`` calls the static member ``mma`` of the
struct that ``Frag_t`` names inside ``Tile<float, 1, 2>``. The call owner is
rewritten to that struct (``Frag::mma(...)``) so the static-member lowering
instantiates the template from the call's arguments.
"""

import re

import pytest

from crosstl._crosstl import translate
from crosstl.backend.Metal.preprocessor import MetalPreprocessor
from crosstl.backend.Metal.qualified_member_types import (
    resolve_qualified_struct_member_types,
)
from tests.test_translator.test_metal_builtin_ownership import _compile

PRELUDE = """#include <metal_stdlib>
using namespace metal;
struct Frag {
  static constant constexpr const short kElems = 8;
  template <typename U>
  using frag_t = typename metal::vec<U, kElems>;
  template <typename CType, typename AType, bool ta = false>
  static void mma(thread frag_t<CType>& C0, thread frag_t<CType>& C1,
                  const thread frag_t<AType>& A, metal::bool_constant<ta>) {
    for (short i = 0; i < kElems; i++) {
      C0[i] += ta ? CType(A[kElems - 1 - i]) : CType(A[i]);
      C1[i] += CType(2) * CType(A[i]);
    }
  }
};
template <typename T, short R, short Cc, class F = Frag>
struct Tile {
  using Frag_t = F;
  static constant constexpr const short kCols = Cc;
  typedef typename Frag_t::template frag_t<T> frag_type;
  frag_type vals[R * Cc];
  thread frag_type& frag_at(const short i, const short j) thread {
    return vals[i * kCols + j];
  }
  template <bool transpose>
  thread frag_type& frag_at(const short i, const short j,
                            metal::bool_constant<transpose>) thread {
    if constexpr (transpose) {
      return frag_at(j, i);
    } else {
      return frag_at(i, j);
    }
  }
};
template <class CTile, class ATile, bool ta>
void tile_mad(thread CTile& C, thread ATile& A, metal::bool_constant<ta>) {
  constexpr auto t = metal::bool_constant<ta>{};
  CTile::Frag_t::mma(C.frag_at(0, 0), C.frag_at(0, 1), A.frag_at(0, 0, t),
                     metal::bool_constant<ta>{});
}
"""

KERNEL = """kernel void k(device float* dst [[buffer(0)]],
              uint gid [[thread_position_in_grid]]) {{
  Tile<float, 1, 2> c;
  Tile<half, 1, 1> a;
  for (short i = 0; i < 8; i++) {{
    c.vals[0][i] = 0.0f;
    c.vals[1][i] = 1.0f;
    a.vals[0][i] = half(gid + i);
  }}
  {call}
  for (short i = 0; i < 8; i++) {{
    dst[gid * 16 + i] = c.vals[0][i];
    dst[gid * 16 + 8 + i] = c.vals[1][i];
  }}
}}
"""

CALLS = {
    "template-parameter-owner": "tile_mad(c, a, metal::bool_constant<true>{});",
    "concrete-owner": (
        "Tile<float, 1, 2>::Frag_t::mma(c.frag_at(0, 0), c.frag_at(0, 1), "
        "a.frag_at(0, 0), metal::bool_constant<true>{});"
    ),
    "local-alias-owner": (
        "using CT = Tile<float, 1, 2>;\n"
        "  CT::Frag_t::mma(c.frag_at(0, 0), c.frag_at(0, 1), a.frag_at(0, 0), "
        "metal::bool_constant<true>{});"
    ),
}

CASES = [
    (call, target)
    for call in CALLS
    for target in ("directx", "opengl", "metal")
    # The Metal target keeps function templates, and a static template member
    # call inside a retained function template is a separate, refused shape.
    if not (call == "template-parameter-owner" and target == "metal")
]


@pytest.mark.parametrize(("call", "target"), CASES)
def test_static_call_through_member_alias_instantiates_the_member(
    tmp_path, call, target
):
    path = tmp_path / "owners.metal"
    path.write_text(PRELUDE + KERNEL.format(call=CALLS[call]), encoding="utf-8")
    generated = translate(str(path), backend=target)
    assert "Frag_t::" not in generated
    assert re.search(r"\bFrag_+mma_+float_half_true\(", generated)
    # Every wide-vector type gets static splat and make helpers whether or not
    # the program calls them; only that dead-helper warning is not an error.
    _compile(
        generated,
        target,
        tmp_path,
        metal_compile_flags=("-Wno-unused-function",),
    )


def _resolve(code):
    preprocessor = MetalPreprocessor()
    templates = preprocessor._find_template_declaration_spans(code)
    aliases = preprocessor._collect_local_type_alias_bindings(
        code, [(0, len(code))], skip_spans=templates
    )
    structs = preprocessor._find_concrete_struct_definitions(code)
    return resolve_qualified_struct_member_types(
        preprocessor,
        code,
        structs,
        aliases,
        sorted(templates + [struct.span for struct in structs]),
    )


OWNERS = """struct Inner { static float f(float x) { return x; } };
struct Outer { using inner_t = Inner; static constant constexpr const int kN = 2; };
"""


def test_only_owners_that_name_a_struct_are_rewritten():
    code = OWNERS + (
        "void g(thread float& x) {\n"
        "  x = Outer::inner_t::f(x);\n"
        "  x = Outer::inner_t::template f<float>(x);\n"
        "  x = metal::precise::sqrt(x) + float(Outer::kN);\n"
        "}\n"
    )
    resolved = _resolve(code)
    assert "x = Inner::f(x);" in resolved
    assert "x = Inner::template f<float>(x);" in resolved
    assert "metal::precise::sqrt(x) + float(Outer::kN)" in resolved

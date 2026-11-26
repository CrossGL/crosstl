"""A struct's own member aliases inside the template arguments of its member types, on Metal input.

Unqualified lookup in a member declaration finds the struct's members first ([basic.lookup.unqual]), so with
``using scalar_T = int;`` the member ``vec<scalar_T, 2> w;`` is an ``int2`` and ``using complex_T = vec<scalar_T, 2>;
complex_T v;`` is one too. The importer resolved only a member whose whole type is an alias, and wrote
``vec<scalar_T, 2>`` for both, which every target refused as an undeclared type. A generic struct's members name the
alias targets in terms of its own parameters (``vec<T, 2>``).
"""

from __future__ import annotations

import re

import pytest

import crosstl
from tests.metal_compute_dispatch import dispatch_word_buffer

REQUIRE_ENV = "CROSSTL_REQUIRE_METAL_STRUCT_ALIAS_EXECUTION"

SOURCE = """#include <metal_stdlib>
using namespace metal;

template <typename T>
struct Pair {
  using scalar_T = T;
  using complex_T = vec<scalar_T, 2>;
  using same_T = complex_T;
  complex_T v;
  vec<scalar_T, 2> w;
  array<scalar_T, 2> a;
  same_T sum() const thread {
    return v + w + complex_T(a[0], a[1]);
  }
};

struct Fixed {
  using scalar_T = int;
  using wide_T = vec<scalar_T, 3>;
  wide_T u;
  vec<scalar_T, 2> t;
};

kernel void k(device int* words [[buffer(0)]], uint id [[thread_position_in_grid]]) {
  Pair<int> p;
  p.v = int2(words[0], words[1]);
  p.w = int2(words[2], words[3]);
  p.a[0] = words[4];
  p.a[1] = words[5];
  int2 s = p.sum();
  Fixed f;
  f.u = int3(words[0], words[2], words[4]);
  f.t = int2(words[1], words[3]);
  words[6] = s.x;
  words[7] = s.y;
  words[8] = f.u.x + f.u.y * 10 + f.u.z * 100 + f.t.x * 1000 + f.t.y * 10000;
}
"""

WORDS = [1, 2, 3, 4, 5, 6, 0, 0, 0, 0]
ALIASES = re.compile(r"\b(?:scalar_T|complex_T|same_T|wide_T)\b")


def _translate(tmp_path, backend):
    path = tmp_path / "aliases.metal"
    path.write_text(SOURCE, encoding="utf-8")
    return crosstl.translate(str(path), backend=backend, format_output=False)


def test_member_types_name_the_alias_targets(tmp_path):
    crossgl = _translate(tmp_path, "cgl")
    members = re.sub(r"\s+", " ", crossgl)

    assert "generic<T> struct Pair { vec<T, 2> v; vec<T, 2> w; T[2] a; }" in members
    assert "struct Pair_int { ivec2 v; ivec2 w; int[2] a; }" in members
    assert "struct Fixed { ivec3 u; ivec2 t; }" in members
    assert not ALIASES.search(crossgl)


def test_aliased_members_compute_like_the_source_on_metal(tmp_path):
    native_dir, translated_dir = tmp_path / "native", tmp_path / "translated"
    native_dir.mkdir()
    translated_dir.mkdir()
    metal = _translate(tmp_path, "metal")

    expected = dispatch_word_buffer(SOURCE, native_dir, WORDS, REQUIRE_ENV)
    assert expected[6:9] == [1 + 3 + 5, 2 + 4 + 6, 1 + 30 + 500 + 2000 + 40000]
    assert dispatch_word_buffer(metal, translated_dir, WORDS, REQUIRE_ENV) == expected


@pytest.mark.parametrize("backend", ["directx", "opengl", "vulkan", "cuda"])
def test_other_targets_translate_the_aliased_members(tmp_path, backend):
    code = _translate(tmp_path, backend)

    assert not ALIASES.search(code)

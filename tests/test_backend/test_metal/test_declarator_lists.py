"""Declarator lists of concrete struct types become one declaration per name."""

import re

import pytest

from crosstl.backend.Metal.declarator_lists import split_struct_declarator_lists
from crosstl.backend.Metal.preprocessor import MetalPreprocessor


def _split(code, names=("S",)):
    spans = MetalPreprocessor()._find_comment_and_literal_spans(code)
    return split_struct_declarator_lists(code, names, spans)


@pytest.mark.parametrize(
    ("code", "expected"),
    [
        ("void f() { S a, b, c; }", "void f() { S a; S b; S c; }"),
        (
            "void f() { const thread S a = {1, 2}, *b = &a, c[2]; }",
            "void f() { const thread S a = {1, 2}; const thread S *b = &a; "
            "const thread S c[2]; }",
        ),
        ("void f() { S a(1, 2), b{3, 4}; }", "void f() { S a(1, 2); S b{3, 4}; }"),
        (
            "void f() {\n  // a = x ; b = (y)\n  S a, b;\n}",
            "void f() {\n  // a = x ; b = (y)\n  S a; S b;\n}",
        ),
    ],
)
def test_struct_declarator_lists_split_into_single_declarations(code, expected):
    assert _split(code) == expected


@pytest.mark.parametrize(
    "code",
    [
        # A for init-statement must stay one statement.
        "void f() { for (S a, b; ;) {} }",
        # Expressions and declarations that only look like lists.
        "void f() { S::g(a, b); }",
        "void f() { S(x), y; }",
        "S make(int a, int b);",
        # A comma inside template arguments cannot be split exactly.
        "void f() { S a = cast<int, 2>(x), b; }",
        # Other types and single declarators are left alone.
        "void f() { T a, b; S c; }",
        "void f() { /* S a, b; */ }",
    ],
)
def test_lists_that_cannot_be_split_exactly_stay_unchanged(code):
    assert _split(code) == code


def test_alias_lists_split_only_where_the_predicate_names_a_struct():
    code = "void f() { A a, b; }\nvoid g() { A c, d; }"
    second = code.index("void g")
    split = split_struct_declarator_lists(
        code,
        ("S", "A"),
        (),
        lambda name, position: name == "S" or position < second,
    )
    assert split == "void f() { A a; A b; }\nvoid g() { A c, d; }"


ALIAS_LIST_KERNEL = """#include <metal_stdlib>
using namespace metal;
struct Acc {
  float v[4];
  void load(device const float* p) thread {
    for (int i = 0; i < 4; i++) v[i] = p[i];
  }
  float sum() const thread {
    float s = 0;
    for (int i = 0; i < 4; i++) s += v[i];
    return s;
  }
};
template <typename T, int N>
struct Tile {
  T v[N];
  void load(device const T* p) thread {
    for (int i = 0; i < N; i++) v[i] = p[i];
  }
  T sum() const thread {
    T s = 0;
    for (int i = 0; i < N; i++) s += v[i];
    return s;
  }
};
typedef Acc AccTypedef;
using AccUsing = Acc;
typedef Tile<float, 4> _Tile4;
template <typename T>
kernel void k(device const T* src [[buffer(0)]], device T* dst [[buffer(1)]],
              uint gid [[thread_position_in_grid]]) {
  AccTypedef a, b;
  AccUsing c, d;
  _Tile4 e, f;
  using LocalTile = Tile<T, 4>;
  LocalTile g, h;
  b.load(src + gid * 8);
  a.load(src + gid * 8 + 4);
  d.load(src + gid * 8);
  c.load(src + gid * 8 + 4);
  f.load(src + gid * 8);
  e.load(src + gid * 8 + 4);
  h.load(src + gid * 8);
  g.load(src + gid * 8 + 4);
  dst[gid] = a.sum() + b.sum() + c.sum() + d.sum() + e.sum() + f.sum() +
             g.sum() + h.sum();
}
template [[host_name("k_float")]] [[kernel]] decltype(k<float>) k<float>;
"""


@pytest.mark.parametrize("target", ["directx", "opengl", "metal"])
def test_struct_alias_declarator_lists_lower_every_receiver(tmp_path, target):
    from crosstl._crosstl import translate
    from tests.test_translator.test_metal_builtin_ownership import _compile

    path = tmp_path / "alias_lists.metal"
    path.write_text(ALIAS_LIST_KERNEL, encoding="utf-8")
    generated = translate(str(path), backend=target)
    for receiver in "abcdefgh":
        assert re.search(rf"\b\w+_load\w*\(\s*{receiver}\s*,", generated), receiver
        assert re.search(rf"\b\w+_sum\w*\(\s*{receiver}\s*\)", generated), receiver
    _compile(generated, target, tmp_path)

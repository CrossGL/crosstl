"""Distinct struct specializations keep distinct concrete names.

The readable specialization identifier drops punctuation, so ``Tile<T, 2,
!flip>`` and ``Tile<T, 2, flip>`` both read ``Tile_float_2_false`` once
``flip`` is ``false``. Each must still get its own struct, constructor and
methods; reusing one name merged the two and silently dropped the negation.
"""

import re

import pytest

from crosstl.project import ProjectConfig, translate_project

TILE = """
#include <metal_stdlib>
using namespace metal;

template <typename T, short N, bool reduce>
struct Tile {
  short n;
  Tile(short base) : n(reduce ? base + N : base - N) {}
  T scale(T value) const {
    return reduce ? value * T(N) : value + T(N);
  }
};
"""

DIRECT = TILE + """
template <typename T, short N, bool flip>
[[kernel]] void tiles(device T* out [[buffer(0)]],
                      uint index [[thread_position_in_grid]]) {
  thread Tile<T, N, !flip> a(short(index));
  thread Tile<T, N, flip> b(short(index));
  out[index] = a.scale(T(a.n)) * T(100) + b.scale(T(b.n));
}

template [[host_name("tiles_false")]] [[kernel]] void tiles<float, 2, false>(
    device float*, uint);
"""

MEMBER_ALIASES = TILE + """
template <typename T, short N, bool flip>
struct Pair {
  using a_t = Tile<T, N, !flip>;
  using b_t = Tile<T, N, flip>;
};

template <typename T, short N, bool flip>
[[kernel]] void tiles(device T* out [[buffer(0)]],
                      uint index [[thread_position_in_grid]]) {
  using pair = Pair<T, N, flip>;
  using a_t = typename pair::a_t;
  using b_t = typename pair::b_t;
  thread a_t a(short(index));
  thread b_t b(short(index));
  out[index] = a.scale(T(a.n)) * T(100) + b.scale(T(b.n));
}

template [[host_name("tiles_false")]] [[kernel]] void tiles<float, 2, false>(
    device float*, uint);
"""


def _translate(tmp_path, source, backend):
    (tmp_path / "tiles.metal").write_text(source)
    config = ProjectConfig(
        root=tmp_path,
        source_roots=(".",),
        include_patterns=("tiles.metal",),
        targets=(backend,),
        output_dir="out",
        entry_points={"tiles.metal": ("tiles_false",)},
        workgroup_size_rules={"tiles.metal": (4, 1, 1)},
    )
    payload = translate_project(config, format_output=False).to_json()
    assert payload["summary"]["translatedCount"] == 1, payload["diagnostics"]
    assert not payload["diagnostics"]
    (artifact,) = payload["artifacts"]
    return (tmp_path / artifact["path"]).read_text()


def _struct_names(code):
    return re.findall(r"\bstruct\s+(Tile\w*)\s*\{", code)


@pytest.mark.parametrize("source", [DIRECT, MEMBER_ALIASES], ids=["direct", "alias"])
@pytest.mark.parametrize("backend", ["metal", "directx", "opengl"])
def test_negated_argument_gets_its_own_struct(tmp_path, source, backend):
    code = _translate(tmp_path, source, backend)
    names = _struct_names(code)
    assert len(names) == len(set(names)) == 2, names


@pytest.mark.parametrize("source", [DIRECT, MEMBER_ALIASES], ids=["direct", "alias"])
def test_each_variable_keeps_its_specialization(tmp_path, source):
    code = _translate(tmp_path, source, "metal")
    names = _struct_names(code)
    assert len(set(names)) == 2, names
    reduced = {}
    for name in names:
        bodies = re.findall(
            rf"\b{name}__scale\b[^{{]*\{{\s*return ([^;]*);", code, flags=re.S
        )
        assert len(bodies) == 1, (name, bodies)
        reduced[name] = bodies[0].strip().startswith(("!false", "true"))
    assert sorted(reduced.values()) == [False, True]
    kernel = code[code.index("kernel void tiles_false") :]
    calls = dict(re.findall(r"\b(Tile\w*?)__scale\((\w+),", kernel))
    assert calls, kernel
    assert {calls[name] for name in calls} == {"a", "b"}
    assert all(reduced[name] == (variable == "a") for name, variable in calls.items())

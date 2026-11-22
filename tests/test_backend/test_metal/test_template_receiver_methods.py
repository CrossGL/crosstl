"""A function template called from a struct method lowers the member calls of its instantiation (#4986).

``tile_add(P, Q)`` in ``Block::run`` binds ``D`` in ``D.frag_at(m, n)`` to ``Tile<int, 1, 2>``, but the importer
instantiates a call in a member function body only after member lowering has moved ``frag_at`` out of
``Tile_int_1_2``. The instantiation then kept ``D.frag_at(m, n)``: GLSL output refused, HLSL output called an
undeclared member and Metal output refused the generic template. Member lowering now keeps a struct whole while a
pending template call still names it, and lowers it once the instantiation exists.
"""

from __future__ import annotations

import re
import shutil
import subprocess

import pytest

import crosstl
from tests.metal_compute_dispatch import compile_metal_library, dispatch_word_buffer

REQUIRE_ENV = "CROSSTL_REQUIRE_TEMPLATE_RECEIVER_EXECUTION"

TILE = """#include <metal_stdlib>
using namespace metal;
template <typename T, int M, int N>
struct Tile {
  T v[M * N];
  METAL_FUNC thread T& frag_at(const short i, const short j) { return v[i * N + j]; }
};
template <typename T, int M, int N>
METAL_FUNC void tile_add(thread Tile<T, M, N>& D, thread Tile<T, M, N>& A) {
  for (short m = 0; m < M; ++m)
    for (short n = 0; n < N; ++n)
      D.frag_at(m, n) += A.frag_at(m, n);
}
"""

# The reproducer of #4986 on one word buffer: the call sits in a method of a plain struct.
PLAIN_STRUCT = (
    TILE
    + """struct Block {
  METAL_FUNC int run(int x, int y) thread {
    Tile<int, 1, 2> P, Q;
    P.v[0] = x;
    P.v[1] = y;
    Q.v[0] = 10;
    Q.v[1] = 20;
    tile_add(P, Q);
    return P.v[0] * 100 + P.v[1];
  }
};
kernel void run(device int* words [[buffer(0)]], uint gid [[thread_position_in_grid]]) {
  Block b;
  words[0] = b.run(words[1], words[2]);
}
"""
)

# The method belongs to a struct template, used at two element types.
TEMPLATE_BLOCK = (
    TILE
    + """template <typename U>
struct Block {
  METAL_FUNC U run(U x, U y) thread {
    Tile<U, 1, 2> P, Q;
    P.v[0] = x;
    P.v[1] = y;
    Q.v[0] = U(10);
    Q.v[1] = U(20);
    tile_add(P, Q);
    return P.v[0] * U(100) + P.v[1];
  }
};
kernel void run(device int* words [[buffer(0)]], uint gid [[thread_position_in_grid]]) {
  Block<int> b;
  words[0] = b.run(words[1], words[2]);
  Block<float> f;
  words[3] = int(f.run(float(words[5]), 0.5f));
}
"""
)

# The tiles are data members of the struct whose method calls the template.
MEMBER_TILES = (
    TILE
    + """struct Block {
  Tile<int, 1, 2> P, Q;
  METAL_FUNC int run(int x, int y) thread {
    P.v[0] = x;
    P.v[1] = y;
    Q.v[0] = 10;
    Q.v[1] = 20;
    tile_add(P, Q);
    return P.v[0] * 100 + P.v[1];
  }
};
kernel void run(device int* words [[buffer(0)]], uint gid [[thread_position_in_grid]]) {
  Block b;
  words[0] = b.run(words[1], words[2]);
}
"""
)

# The method reaches the template through another function template that makes no member call itself.
CHAINED = (
    TILE
    + """template <typename T, int M, int N>
METAL_FUNC void tile_add_twice(thread Tile<T, M, N>& D, thread Tile<T, M, N>& A) {
  tile_add(D, A);
  tile_add(D, A);
}
struct Block {
  METAL_FUNC int run(int x, int y) thread {
    Tile<int, 1, 2> P, Q;
    P.v[0] = x;
    P.v[1] = y;
    Q.v[0] = 10;
    Q.v[1] = 20;
    tile_add_twice(P, Q);
    return P.v[0] * 100 + P.v[1];
  }
};
kernel void run(device int* words [[buffer(0)]], uint gid [[thread_position_in_grid]]) {
  Block b;
  words[0] = b.run(words[1], words[2]);
}
"""
)

# The same call in a free function, which already translated for HLSL and GLSL.
FREE_FUNCTION = (
    TILE
    + """METAL_FUNC int run_tiles(int x, int y) {
  Tile<int, 1, 2> P, Q;
  P.v[0] = x;
  P.v[1] = y;
  Q.v[0] = 10;
  Q.v[1] = 20;
  tile_add(P, Q);
  return P.v[0] * 100 + P.v[1];
}
kernel void run(device int* words [[buffer(0)]], uint gid [[thread_position_in_grid]]) {
  words[0] = run_tiles(words[1], words[2]);
}
"""
)

SOURCES = {
    "plain-struct": PLAIN_STRUCT,
    "template-block": TEMPLATE_BLOCK,
    "member-tiles": MEMBER_TILES,
    "chained": CHAINED,
    "free-function": FREE_FUNCTION,
}
WORDS = [0, 5, 2, 1, 0, 3, 2, 0]
# Worked by hand: (5 + 10) * 100 + (2 + 20) = 1522; chained adds Q twice, (5 + 20) * 100 + (2 + 40) = 2542; the float
# block gives (3 + 10) * 100 + (0.5 + 20) = 1320.5, written as 1320.
EXPECTED = {
    "plain-struct": {0: 1522},
    "template-block": {0: 1522, 3: 1320},
    "member-tiles": {0: 1522},
    "chained": {0: 2542},
    "free-function": {0: 1522},
}


def _write(tmp_path, name):
    source = tmp_path / f"{name}.metal"
    source.write_text(SOURCES[name], encoding="utf-8")
    return source


def _round_trip(tmp_path, name):
    crossgl = tmp_path / f"{name}.cgl"
    crossgl.write_text(
        crosstl.translate(str(_write(tmp_path, name)), backend="cgl"),
        encoding="utf-8",
    )
    return crosstl.translate(str(crossgl), backend="metal", format_output=False)


def _live_text(crossgl):
    # The residual generic template may stay for a struct template's own method body; drop it and keep the code
    # an entry reaches.
    return re.sub(
        r"generic<[^>]*>\s*void\s+tile_add\w*\([^)]*\)[^{]*\{.*?\n    \}\n",
        "",
        crossgl,
        flags=re.S,
    )


@pytest.mark.parametrize("name", sorted(SOURCES))
def test_instantiated_template_reaches_crossgl_without_member_calls(tmp_path, name):
    crossgl = crosstl.translate(str(_write(tmp_path, name)), backend="cgl")

    live = _live_text(crossgl)
    assert re.search(r"\bvoid tile_add_int_1_2\(", live), crossgl
    assert "frag_at" not in live, crossgl
    assert re.search(r"\bD\.v\[m \* 2 \+ n\] \+= A\.v\[m \* 2 \+ n\]", live), crossgl


@pytest.mark.parametrize("name", sorted(SOURCES))
def test_metal_output_compiles(tmp_path, name):
    for label, metal in (
        ("direct", crosstl.translate(str(_write(tmp_path, name)), backend="metal")),
        ("round-trip", _round_trip(tmp_path, name)),
    ):
        assert "frag_at" not in metal, metal
        (tmp_path / label).mkdir()
        compile_metal_library(metal, tmp_path / label, REQUIRE_ENV)


@pytest.mark.parametrize("name", sorted(SOURCES))
def test_translation_computes_like_the_source_on_metal(tmp_path, name):
    (tmp_path / "native").mkdir()
    (tmp_path / "translated").mkdir()
    native = dispatch_word_buffer(
        SOURCES[name], tmp_path / "native", WORDS, REQUIRE_ENV
    )
    translated = dispatch_word_buffer(
        _round_trip(tmp_path, name), tmp_path / "translated", WORDS, REQUIRE_ENV
    )
    assert translated == native


@pytest.mark.parametrize("name", sorted(SOURCES))
def test_translation_writes_the_expected_words(tmp_path, name):
    words = dispatch_word_buffer(
        _round_trip(tmp_path, name), tmp_path, WORDS, REQUIRE_ENV
    )
    expected = EXPECTED[name]
    assert {index: words[index] for index in expected} == expected


@pytest.mark.parametrize("backend", ["directx", "opengl"])
@pytest.mark.parametrize("name", sorted(SOURCES))
def test_other_targets_call_the_instantiation(tmp_path, name, backend):
    output = crosstl.translate(str(_write(tmp_path, name)), backend=backend)

    assert "frag_at" not in output, output
    assert re.search(
        r"\btile_add(?:_twice)?_int_1_2\((?:self\.)?P, (?:self\.)?Q\)", output
    ), output
    assert re.search(r"\bD\.v\[", output), output


@pytest.mark.parametrize("name", ["plain-struct", "template-block"])
def test_spirv_output_validates(tmp_path, name):
    spirv = crosstl.translate(str(_write(tmp_path, name)), backend="vulkan")
    assert "OpEntryPoint GLCompute" in spirv
    if shutil.which("spirv-as") is None or shutil.which("spirv-val") is None:
        pytest.skip("spirv-tools is not installed")
    (tmp_path / "tiles.spvasm").write_text(spirv, encoding="utf-8")
    for command in (
        ["spirv-as", "--target-env", "vulkan1.0", "tiles.spvasm", "-o", "t.spv"],
        ["spirv-val", "--target-env", "vulkan1.0", "t.spv"],
    ):
        result = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True)
        assert result.returncode == 0, result.stderr

"""Reference aliases bound to an accessor of a named local or parameter.

``thread auto& frag = tile.frag_at(0, k);`` binds ``frag`` to the element the
accessor returns. When ``tile`` is a local value or a ``thread`` parameter of
an entry point or free function, the binding captures the accessor's
arguments and every subscripted use of ``frag`` becomes a use of
``tile.val_frags[index]``. Writes through the alias reach ``tile``, and later
changes to ``k`` do not move the alias.
"""

import json
import re
import shutil
import sys

import pytest

from crosstl._crosstl import translate
from crosstl.backend.Metal.function_parameter_lists import parameter_list_start
from crosstl.backend.Metal.preprocessor import MetalPreprocessor, MetalStructMethodError
from tests.test_translator.test_metal_builtin_ownership import ROOT, _compile, _run

SOURCE = """#include <metal_stdlib>
using namespace metal;

template <typename T, short R, short C>
struct Tile {
  using frag_type = vec<T, 4>;
  frag_type val_frags[R * C];

  thread frag_type& frag_at(const short i, const short j) thread {
    return val_frags[i * C + j];
  }

  const thread frag_type& frag_at(const short i, const short j) const thread {
    return val_frags[i * C + j];
  }
};

void bump(thread Tile<float, 1, 2>& tile, short j) {
  thread auto& frag = tile.frag_at(0, j);
  frag[3] += 0.5f;
}

kernel void aliases(device float* results [[buffer(0)]]) {
  Tile<float, 1, 2> tile;
  for (short f = 0; f < 2; f++) {
    for (short i = 0; i < 4; i++) {
      tile.val_frags[f][i] = float(f * 4 + i);
    }
  }
  for (short k = 0; k < 2; k++) {
    thread auto& frag = tile.frag_at(0, k);
    for (short i = 0; i < 4; i++) {
      frag[i] = frag[i] * 2.0f + float(k);
    }
  }
  short k = 0;
  thread auto& first = tile.frag_at(0, k);
  k = 1;
  first[0] += 100.0f;
  bump(tile, k);
  const thread auto& second = tile.frag_at(0, k);
  for (short i = 0; i < 4; i++) {
    results[i] = tile.val_frags[0][i];
    results[4 + i] = second[i];
  }
}
"""

# Fragment 0 is doubled, then 100 is added to its first lane through `first`.
# Fragment 1 is doubled plus one, then `bump` adds 0.5 to its last lane.
EXPECTED = [100.0, 2.0, 4.0, 6.0, 9.0, 11.0, 13.0, 15.5]

TILE = """
#include <metal_stdlib>
using namespace metal;
struct Tile {
  float4 val_frags[2];
  thread float4& frag_at(const short i, const short j) thread {
    return val_frags[i * 2 + j];
  }
  const thread float4& frag_at(const short i, const short j) const thread {
    return val_frags[i * 2 + j];
  }
};
"""


def test_aliases_of_locals_and_parameters_use_the_receiver_storage():
    output = MetalPreprocessor().preprocess(SOURCE)

    assert re.search(r"\.\s*frag_at\s*\(", output) is None
    assert "auto&" not in output
    assert "tile.val_frags[crosstl_reference_index_0][3] += 0.5f;" in output
    # `first` captures `k` at its binding, so `k = 1;` does not move it.
    first = re.search(
        r"short (crosstl_reference_argument_\d+) = \(k\);\s*"
        r"const auto (crosstl_reference_index_\d+) = [^;]*\1[^;]*;\s*"
        r"k = 1;\s*tile\.val_frags\[\2\]\[0\] \+= 100\.0f;",
        output,
    )
    assert first is not None


@pytest.mark.parametrize("target", ["directx", "opengl", "metal"])
def test_aliases_translate_and_compile(tmp_path, target):
    path = tmp_path / "aliases.metal"
    path.write_text(SOURCE, encoding="utf-8")

    generated = translate(str(path), backend=target)

    assert "frag_at" not in generated
    _compile(generated, target, tmp_path)


@pytest.mark.skipif(
    sys.platform != "darwin" or not (shutil.which("xcrun") and shutil.which("swiftc")),
    reason="native Metal execution needs macOS with xcrun and swiftc",
)
def test_translated_metal_matches_the_source_natively(tmp_path):
    source = tmp_path / "aliases.metal"
    source.write_text(SOURCE, encoding="utf-8")
    generated = translate(str(source), backend="metal")
    _artifact, module = _compile(generated, "metal", tmp_path)
    runner = tmp_path / "readback"
    _run(
        [
            "swiftc",
            str(
                ROOT / "tests/fixtures/runtime_verification/metal_float_readback.swift"
            ),
            "-o",
            str(runner),
        ]
    )
    original_air = tmp_path / "original.air"
    original = tmp_path / "original.metallib"
    _run(
        ["xcrun", "--sdk", "macosx", "metal", "-Werror", "-c", str(source)]
        + ["-o", str(original_air)]
    )
    _run(
        ["xcrun", "--sdk", "macosx", "metallib", str(original_air), "-o", str(original)]
    )
    entry = re.search(r"\bkernel\s+void\s+(\w+)", generated).group(1)

    for library, name in ((original, "aliases"), (module, entry)):
        values = json.loads(_run([str(runner), str(library), name, str(len(EXPECTED))]))
        assert values["values"] == EXPECTED


REFUSED = {
    # A use without a subscript is not rewritten to the storage.
    "whole-alias-write": (
        "kernel void k(device float* r [[buffer(0)]]) {"
        " Tile tile; tile.val_frags[0] = float4(1);"
        " thread auto& frag = tile.frag_at(0, 0); frag += 1.0f;"
        " r[0] = tile.val_frags[0][0]; }"
    ),
    "address-escape": (
        "kernel void k(device float* r [[buffer(0)]]) {"
        " Tile tile; tile.val_frags[0] = float4(1);"
        " thread auto& frag = tile.frag_at(0, 0); thread float4* p = &frag;"
        " r[0] = (*p)[0]; }"
    ),
    # The inner `tile` would capture the alias's storage name.
    "shadowed-receiver": (
        "kernel void k(device float* r [[buffer(0)]]) {"
        " Tile tile; tile.val_frags[0] = float4(1);"
        " thread auto& frag = tile.frag_at(0, 0);"
        " { Tile tile; tile.val_frags[0] = float4(2); frag[0] = 3.0f;"
        " r[1] = tile.val_frags[0][0]; }"
        " r[0] = tile.val_frags[0][0]; }"
    ),
    "device-receiver": (
        "void f(device Tile& tile, device float* r) {"
        " thread auto& frag = tile.frag_at(0, 0); r[0] = frag[0]; }\n"
        "kernel void k(device Tile* t [[buffer(1)]], device float* r [[buffer(0)]])"
        " { f(t[0], r); }"
    ),
}


@pytest.mark.parametrize("name", sorted(REFUSED))
def test_unprovable_aliases_fail_closed(name):
    with pytest.raises(MetalStructMethodError) as excinfo:
        MetalPreprocessor().preprocess(TILE + REFUSED[name])

    error = excinfo.value
    assert error.missing_capabilities == ("struct.reference-return",)
    assert error.reason == "reference-return-identity-unsupported"
    assert error.requested_signature == "Tile::frag_at(0, 0)"


@pytest.mark.parametrize(
    ("header", "expected"),
    [
        ("void f(thread Tile& t, short j) {", "("),
        ("kernel void k(device float* r [[buffer(0)]]) {", "("),
        ("void f(int a /* ) */) {", "("),
        ("void f() const {", "{"),
    ],
)
def test_parameter_list_start(header, expected):
    masked = MetalPreprocessor()._mask_comments_and_literals(header)
    start = parameter_list_start(masked, 0, len(header) - 1)
    assert header[start] == expected

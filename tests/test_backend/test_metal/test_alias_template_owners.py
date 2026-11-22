"""Member alias templates named through a function-local alias of their owner.

``using CFrag = typename TileT::Frag_t;`` names a concrete struct once the
enclosing template is instantiated, so ``typename CFrag::template frag_t<T>``
denotes the type the member alias template names ([temp.alias]/2). The
importer must expand it, otherwise the argument types of calls such as
``CFrag::load(elems, ...)`` cannot be inferred and the template member method
cannot be lowered.
"""

import json
import re
import shutil
import sys

import pytest

from crosstl._crosstl import translate
from crosstl.backend.Metal.preprocessor import MetalPreprocessor
from tests.test_translator.test_metal_builtin_ownership import ROOT, _compile, _run

SOURCE = """#include <metal_stdlib>
using namespace metal;

struct Frag {
  static constant constexpr short kElems = 4;
  template <typename U>
  using frag_t = typename metal::vec<U, kElems>;

  template <typename T, typename P>
  static void load(thread frag_t<T>& dst, P src, int stride) {
    for (short i = 0; i < kElems; i++) {
      dst[i] = src[i * stride];
    }
  }
};

template <typename F>
struct Tile {
  using Frag_t = F;
};

template <typename T, typename TileT>
void gather(const device T* src, device float* out, int stride) {
  using CFrag = typename TileT::Frag_t;
  using cfrag_t = typename CFrag::template frag_t<T>;
  cfrag_t elems;
  CFrag::load(elems, src, stride);
  for (short i = 0; i < CFrag::kElems; i++) {
    out[i] = float(elems[i]) * 10.0f;
  }
}

kernel void owners(device float* results [[buffer(0)]]) {
  for (short i = 0; i < 8; i++) {
    results[i] = float(i);
  }
  gather<float, Tile<Frag>>(results, results + 8, 2);
}
"""

# `gather` reads every second input and scales it by ten.
EXPECTED = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 0.0, 20.0, 40.0, 60.0]

HEADER = """#include <metal_stdlib>
using namespace metal;
struct Frag {
  template <typename U>
  using frag_t = metal::vec<U, 4>;
};
struct Plain {
  int x;
};
"""

USE = "typename C::template frag_t<float> v = float4(1.0f); r[0] = v[0];"


def test_local_owner_alias_expands_the_member_alias_template():
    output = MetalPreprocessor().preprocess(SOURCE)

    assert re.search(r"\bfrag_t\b", output.split("void gather", 1)[1]) is None
    assert "using cfrag_t = metal::vec<float,4>;" in output
    assert "Frag__load__float_const_device_float_ptr(elems, src, stride);" in output


def test_function_local_owner_alias_outside_templates_expands():
    source = HEADER + (
        "kernel void k(device float* r [[buffer(0)]]) { using C = Frag; " + USE + " }\n"
    )

    output = MetalPreprocessor().preprocess(source)

    assert "metal::vec<float,4> v = float4(1.0f);" in output
    assert "frag_t<float>" not in output


@pytest.mark.parametrize("target", ["directx", "opengl", "metal"])
def test_owner_aliases_translate_and_compile(tmp_path, target):
    path = tmp_path / "owners.metal"
    path.write_text(SOURCE, encoding="utf-8")

    generated = translate(str(path), backend=target)

    assert re.search(r"\bfrag_t\b", generated) is None
    _compile(generated, target, tmp_path)


@pytest.mark.skipif(
    sys.platform != "darwin" or not (shutil.which("xcrun") and shutil.which("swiftc")),
    reason="native Metal execution needs macOS with xcrun and swiftc",
)
def test_translated_metal_matches_the_source_natively(tmp_path):
    source = tmp_path / "owners.metal"
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

    for library, name in ((original, "owners"), (module, entry)):
        values = json.loads(_run([str(runner), str(library), name, str(len(EXPECTED))]))
        assert values["values"] == EXPECTED


KEPT = {
    # The owner is concrete but declares no `frag_t`.
    "owner-without-member": "{ using C = Plain; " + USE + " }",
    # The alias of `f` is not visible in `k`.
    "alias-of-another-function": "{ " + USE + " }",
    # The alias ends with its block.
    "alias-of-a-closed-block": "{ { using C = Frag; } " + USE + " }",
}


@pytest.mark.parametrize("name", sorted(KEPT))
def test_owners_that_do_not_resolve_keep_their_spelling(name):
    source = HEADER + (
        "void f() { using C = Frag; }\n"
        "kernel void k(device float* r [[buffer(0)]]) " + KEPT[name] + "\n"
    )

    output = MetalPreprocessor().preprocess(source)

    assert "typename C::template frag_t<float> v" in output

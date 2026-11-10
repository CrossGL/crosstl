"""Implicit template materialization types a pointer offset argument by its pointer.

``p + n``, ``n + p`` and ``p - n`` have the type of the pointer ``p`` ([expr.add]). A
member function that passes ``dst + i * pack()`` for a deduced pointer parameter lowers
to ``self.dst + i * pack()``, whose offset (a loop counter times a call) is untyped, so
the arithmetic rule that needs every operand typed left the parameter unresolved and the
translation was refused.
"""

import shutil
import subprocess

import pytest

from crosstl.backend.Metal.preprocessor import MetalPreprocessor
from crosstl.project import ProjectConfig, translate_project
from crosstl.project.arithmetic_result_types import arithmetic_result_type
from crosstl.project.pipeline import _metal_expression_type

SOURCE = """#include <metal_stdlib>
using namespace metal;

template <int bits>
constexpr int packing() { return bits == 5 ? 8 : 4; }

template <typename U, int N, typename W>
inline void decode(const device uint8_t* w, U scale, W w_local) {
  for (int i = 0; i < N; i++) {
    w_local[i] = scale * U(w[i]);
  }
}

template <typename T, int bits>
struct Loader {
  static constant constexpr const short pack = packing<bits>();
  const short bi;
  threadgroup T* dst;
  const device uint8_t* src;
  Loader(threadgroup T* d, const device uint8_t* s, ushort lane)
      : bi(lane / 4), dst(d + bi * pack), src(s + bi) {}
  void load() const thread {
    if (bi >= 4) {
      return;
    }
    T scale = T(1);
    for (int i = 0; i < 2; i++) {
      decode<T, pack>(src + i * pack, scale, dst + i * pack);
    }
  }
  void next() {
    src += pack;
  }
};

template <typename T, int bits>
void run(const device uint8_t* w, device T* out, threadgroup T* tile, ushort lane) {
  Loader<T, bits> loader(tile, w, lane);
  for (int k = 0; k < 2; k++) {
    loader.load();
    loader.next();
  }
  out[lane] = tile[lane % 16];
}

template <typename T, int bits>
[[kernel]] void decode_tiles(
    device const uint8_t* w [[buffer(0)]],
    device T* out [[buffer(1)]],
    ushort lane [[thread_index_in_simdgroup]]) {
  threadgroup T tile[64];
  run<T, bits>(w, out, tile, lane);
}

template [[host_name("decode_tiles_float_5")]] [[kernel]]
decltype(decode_tiles<float, 5>) decode_tiles<float, 5>;
"""


def _translate(tmp_path, target):
    (tmp_path / "kernel.metal").write_text(SOURCE, encoding="utf-8")
    config = ProjectConfig(
        root=tmp_path,
        include_patterns=("kernel.metal",),
        targets=(target,),
        workgroup_size=(32, 1, 1),
        entry_points={"kernel.metal": ("decode_tiles_float_5",)},
    )
    payload = translate_project(config, format_output=False).to_json()
    assert payload["summary"]["translatedCount"] == 1, payload["diagnostics"]
    assert payload["diagnostics"] == []
    return (tmp_path / payload["artifacts"][0]["path"]).read_text(encoding="utf-8")


@pytest.mark.parametrize("target", ("metal", "directx"))
def test_member_pointer_offset_deduces_the_pointer_parameter(tmp_path, target):
    generated = _translate(tmp_path, target)
    # W is deduced from `dst + i * pack` as the threadgroup pointer, so the one
    # specialization takes the tile through a threadgroup float pointer.
    assert "decode_float_8_threadgroup_float(" in generated
    if target != "metal" or shutil.which("xcrun") is None:
        return
    source = tmp_path / "generated.metal"
    source.write_text(generated, encoding="utf-8")
    compiled = subprocess.run(
        ["xcrun", "-sdk", "macosx", "metal", "-std=metal3.1", "-Werror", "-c"]
        + [str(source), "-o", str(tmp_path / "generated.air")],
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )
    if "unable to find utility" in compiled.stderr:
        return
    assert compiled.returncode == 0, compiled.stderr


POINTER = "threadgroup float*"


@pytest.mark.parametrize(
    ("operand_types", "operator", "expected"),
    (
        # p + n and n + p keep the pointer type even for an untyped offset:
        # adding two pointers is ill-formed.
        ((POINTER, None), "+", POINTER),
        ((None, POINTER), "+", POINTER),
        ((POINTER, "int", None), "+", POINTER),
        ((POINTER, "uint"), "-", POINTER),
        # p - q is ptrdiff_t, so an untyped subtrahend may not be an offset.
        ((POINTER, None), "-", None),
        ((POINTER, POINTER), "-", None),
        (("int", POINTER), "-", None),
        ((POINTER, POINTER), "+", None),
        ((POINTER, "int"), "*", None),
        # Arithmetic without a pointer still needs every operand typed.
        (("uint", None), "+", None),
        (("uint", "long"), "+", "long"),
    ),
)
def test_pointer_offset_result_types(operand_types, operator, expected):
    preprocessor = MetalPreprocessor()
    assert arithmetic_result_type(preprocessor, operand_types, operator) == expected


def test_member_pointer_offset_expression_type():
    preprocessor = MetalPreprocessor()
    environment = {"self": "thread const Loader&", "self.dst": POINTER}
    assert (
        _metal_expression_type(
            preprocessor, "self.dst + i * (packing<5>())", environment, {}
        )
        == POINTER
    )
    assert (
        _metal_expression_type(preprocessor, "self.dst - offset", environment, {})
        is None
    )

"""The Metal importer predefines __METAL_VERSION__ and __METAL__, as the Metal compiler does.

Headers shared between host and shader code hide host-only declarations with ``#ifndef __METAL_VERSION__`` and choose
language features with ``#if __METAL_VERSION__ >= 230``. The importer defined neither macro, so it translated the
host branches: a host ``struct packed_float3`` that redefines Metal's type, or the older of two shader variants. It
now reads sources as MSL 3.1 (``__METAL_VERSION__`` 310) unless the caller defines another version.
"""

import crosstl
from crosstl.backend.Metal.preprocessor import METAL_PREDEFINED_MACROS

SOURCE = """
#include <metal_stdlib>
using namespace metal;

#ifndef __METAL_VERSION__
struct packed_float3 { float x; float y; float z; };
#endif

#if __METAL_VERSION__ >= 300
#define SCALE 3u
#elif defined(__METAL_VERSION__)
#define SCALE 2u
#else
#define SCALE 1u
#endif

#ifdef __METAL__
#define FROM_METAL 1u
#else
#define FROM_METAL 0u
#endif

kernel void scale(device uint *out [[buffer(0)]],
                  uint tid [[thread_position_in_grid]]) {
    out[tid] = tid * SCALE + FROM_METAL;
}
"""


def _translate(tmp_path, **options):
    path = tmp_path / "scale.metal"
    path.write_text(SOURCE, encoding="utf-8")
    return crosstl.translate(str(path), backend="cgl", **options)


def test_the_macros_are_those_of_msl_3_1():
    assert METAL_PREDEFINED_MACROS == {"__METAL_VERSION__": "310", "__METAL__": "1"}


def test_sources_take_their_metal_branches(tmp_path):
    crossgl = _translate(tmp_path)
    assert "packed_float3" not in crossgl
    assert "tid * 3u + 1u" in crossgl


def test_a_caller_defined_version_wins(tmp_path):
    crossgl = _translate(tmp_path, defines={"__METAL_VERSION__": "220"})
    assert "tid * 2u + 1u" in crossgl

"""Built-in results that convert, integer ``sign``, and arrays returned through a typedef, on Metal.

HLSL's ``int2 r = sign(v)`` of a ``float2`` imports as ``ivec2 r = sign(v)``: CrossGL's ``sign``, ``abs`` and
``step`` keep their operand's type, and the Metal emitter did not know those result types, so it wrote
``int2 r = sign(v)``, which Metal rejects (no conversion from ``float2``). ``sign`` of an integer operand had no Metal
overload at all: MSL declares ``sign`` for floating-point types only. ``typedef float Triple[3]; Triple f()``
returned ``float[3]``, which C++ cannot spell; the direct form already returned ``array<float, 3>``.
"""

from __future__ import annotations

import pytest

import crosstl
from tests.metal_compute_dispatch import dispatch_word_buffer

REQUIRE_ENV = "CROSSTL_REQUIRE_METAL_BUILTIN_RESULT_EXECUTION"

WORDS = [0, -3, 5, -7, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

SIGN_SOURCE = """RWStructuredBuffer<int> words : register(u0);
[numthreads(1, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    float2 a = float2(words[1], words[2]) * 0.5;
    int2 r = sign(a);
    int3 v = int3(words[1], words[2], words[3]);
    int3 s = sign(v);
    int k = sign(words[3]) + sign(words[0]);
    uint u = uint(words[4]);
    int su = sign(u) + sign(uint(words[0]));
    int2 m = abs(int2(words[1], words[3]));
    int2 st = step(float2(0.0, 0.0), a);
    words[5] = r.x * 10 + r.y;
    words[6] = s.x * 100 + s.y * 10 + s.z;
    words[7] = k;
    words[8] = su;
    words[9] = m.x * 100 + m.y;
    words[10] = st.x * 10 + st.y;
}
"""


def _sign(x):
    return (x > 0) - (x < 0)


def _expected_sign_words():
    words = list(WORDS)
    a = (WORDS[1] * 0.5, WORDS[2] * 0.5)
    words[5] = _sign(a[0]) * 10 + _sign(a[1])
    words[6] = _sign(WORDS[1]) * 100 + _sign(WORDS[2]) * 10 + _sign(WORDS[3])
    words[7] = _sign(WORDS[3]) + _sign(WORDS[0])
    words[8] = _sign(WORDS[4]) + _sign(WORDS[0])
    words[9] = abs(WORDS[1]) * 100 + abs(WORDS[3])
    words[10] = int(a[0] >= 0) * 10 + int(a[1] >= 0)
    return words


ARRAY_SOURCE = """RWStructuredBuffer<int> words : register(u0);
typedef float Triple[3];

Triple spread(float x) {
    Triple r = {x, x * 2.0, x + 1.0};
    return r;
}
float total(Triple t) { return t[0] + t[1] * 2.0 + t[2] * 4.0; }
[numthreads(1, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    Triple a = spread(float(words[2]));
    Triple b = {1.0, 2.0, 3.0};
    int k = words[4] % 3;
    words[5] = int(a[0] + a[1] * 16.0 + a[2] * 256.0);
    words[6] = int(spread(2.0)[k]);
    words[7] = int(total(a) + total(b) * 100.0);
    words[8] = int(total(spread(1.0)));
}
"""


def _expected_array_words():
    def spread(x):
        return [x, x * 2.0, x + 1.0]

    def total(t):
        return t[0] + t[1] * 2.0 + t[2] * 4.0

    words = list(WORDS)
    a = spread(float(WORDS[2]))
    words[5] = int(a[0] + a[1] * 16.0 + a[2] * 256.0)
    words[6] = int(spread(2.0)[WORDS[4] % 3])
    words[7] = int(total(a) + total([1.0, 2.0, 3.0]) * 100.0)
    words[8] = int(total(spread(1.0)))
    return words


def _metal(tmp_path, source, name="probe.hlsl"):
    path = tmp_path / name
    path.write_text(source, encoding="utf-8")
    return crosstl.translate(str(path), backend="metal", format_output=False)


def test_sign_abs_and_step_results_convert_to_the_declared_type(tmp_path):
    metal = _metal(tmp_path, SIGN_SOURCE)

    assert "int2 r = int2(sign(a));" in metal
    assert "int2 st = int2(step(" in metal


def test_integer_sign_goes_through_a_helper(tmp_path):
    metal = _metal(tmp_path, SIGN_SOURCE)

    assert "static inline int3 crosstl_sign_int3(int3 x) {" in metal
    assert "static inline uint crosstl_sign_uint(uint x) {" in metal
    assert "int3 s = crosstl_sign_int3(v);" in metal
    assert "sign(v)" not in metal


def test_builtin_results_compute_like_the_source_on_metal(tmp_path):
    metal = _metal(tmp_path, SIGN_SOURCE)

    assert (
        dispatch_word_buffer(metal, tmp_path, WORDS, REQUIRE_ENV)
        == _expected_sign_words()
    )


def test_typedef_array_return_is_a_metal_array(tmp_path):
    metal = _metal(tmp_path, ARRAY_SOURCE)

    assert "array<float, 3> spread(float x) {" in metal
    assert "array<float, 3> r = {" in metal
    assert "array<float, 3> a = spread(" in metal
    assert "total(a.data())" in metal
    assert "float[3]" not in metal


def test_typedef_array_return_computes_like_the_source_on_metal(tmp_path):
    metal = _metal(tmp_path, ARRAY_SOURCE)

    assert (
        dispatch_word_buffer(metal, tmp_path, WORDS, REQUIRE_ENV)
        == _expected_array_words()
    )


@pytest.mark.parametrize("backend", ["opengl", "directx"])
def test_other_targets_keep_translating_the_sources(tmp_path, backend):
    for index, source in enumerate((SIGN_SOURCE, ARRAY_SOURCE)):
        path = tmp_path / f"probe{index}.hlsl"
        path.write_text(source, encoding="utf-8")
        assert crosstl.translate(str(path), backend=backend)

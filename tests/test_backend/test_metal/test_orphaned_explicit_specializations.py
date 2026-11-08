"""Explicit function template specializations after project materialization.

``template <> void f<float>(...)`` replaces the body of the primary template ``f`` for ``float``; it is reached only
through calls of the primary ([temp.expl.spec]/1). Translation to a template-hostile target materializes each
reachable call into a concrete copy that takes the specialization's body, then removes the primary templates. The
specialization itself used to stay behind and reach the generator as a function of its own, with calls of the
removed generic helpers in its body.
"""

import crosstl
from crosstl.backend.Metal.orphaned_explicit_specializations import (
    prune_orphaned_explicit_specializations,
)
from crosstl.backend.Metal.preprocessor import MetalPreprocessor

PREFIX = "#include <metal_stdlib>\nusing namespace metal;\n"

ATOMIC_HELPERS = """
template <typename T>
constexpr constant bool is_atomic_t = is_same_v<T, int> || is_same_v<T, uint> || is_same_v<T, float>;

template <typename T, typename = void>
struct holder { atomic<uint> val; };

template <typename T>
struct holder<T, enable_if_t<is_atomic_t<T>>> { atomic<T> val; };

template <typename T, enable_if_t<is_atomic_t<T>, bool> = true>
METAL_FUNC T load_value(device holder<T>* o, size_t off) {
  return atomic_load_explicit(&(o[off].val), memory_order_relaxed);
}

template <typename T, enable_if_t<is_atomic_t<T>, bool> = true>
METAL_FUNC bool exchange_if(device holder<T>* o, thread T* e, T v, size_t off) {
  return atomic_compare_exchange_weak_explicit(
      &(o[off].val), e, v, memory_order_relaxed, memory_order_relaxed);
}

template <typename T, enable_if_t<is_atomic_t<T>, bool> = true>
METAL_FUNC void store_min(device holder<T>* o, T v, size_t off) {
  atomic_fetch_min_explicit(&(o[off].val), v, memory_order_relaxed);
}

template <>
METAL_FUNC void store_min<float>(device holder<float>* o, float v, size_t off) {
  float e = load_value(o, off);
  while (v < e) {
    if (exchange_if(o, &e, v, off)) {
      return;
    }
  }
}
"""


def _translate(tmp_path, source, backend="directx"):
    path = tmp_path / "kernel.metal"
    path.write_text(PREFIX + source)
    return crosstl.translate(
        str(path), backend=backend, format_output=False, source_backend="metal"
    )


def test_unreferenced_specialization_of_a_removed_primary_is_dropped(tmp_path):
    source = ATOMIC_HELPERS + """
kernel void k(device holder<float>* o [[buffer(0)]], uint i [[thread_position_in_grid]]) {
  atomic_store_explicit(&(o[i].val), 1.0f, memory_order_relaxed);
}
"""
    hlsl = _translate(tmp_path, source)
    assert "store_min" not in hlsl
    assert "load_value(" not in hlsl
    assert "&e" not in hlsl


def test_called_specialization_keeps_only_its_materialized_copy(tmp_path):
    source = ATOMIC_HELPERS + """
kernel void k(device holder<float>* o [[buffer(0)]], uint i [[thread_position_in_grid]]) {
  store_min(o, 1.0f, i);
}
"""
    hlsl = _translate(tmp_path, source)
    assert "void store_min_float(" in hlsl
    assert "store_min_float(o, int64_t(0), 1.0, i);" in hlsl
    assert "u3c" not in hlsl
    assert "load_value_float(o, int64_t(o_offset), off)" in hlsl


def test_explicit_argument_calls_reach_the_specialized_bodies(tmp_path):
    source = """
template <typename T> T max_value() { return T(0); }
template <> uint max_value<uint>() { return 4294967295u; }
template <> float max_value<float>() { return 3.0e38f; }
template <typename T> T twice(T x) { return x * 2; }
template <> float twice<float>(float x) { return x + x + 1.0f; }

kernel void k(device uint* o [[buffer(0)]], device float* f [[buffer(1)]],
              uint i [[thread_position_in_grid]]) {
  o[i] = max_value<uint>();
  float m = max_value<float>();
  f[i] = twice(m);
}
"""
    hlsl = _translate(tmp_path, source)
    assert "u3c" not in hlsl
    assert "o[i] = max_value_uint();" in hlsl
    assert "float m = max_value_float();" in hlsl
    assert "f[i] = twice_float(m);" in hlsl
    assert "return ((x + x) + 1.0);" in hlsl


def test_referenced_specialization_is_kept():
    code = """
template <typename T> T scale(T x) { return x; }
template <> float scale<float>(float x) { return x * 2.0f; }
template <> half scale<half>(half x) { return x; }
float use(float x) { return scale<float>(x); }
"""
    pruned = prune_orphaned_explicit_specializations(MetalPreprocessor(), code)
    assert pruned == code


def test_specializations_named_nowhere_else_are_removed():
    code = """
float scale_float(float x) { return x * 2.0f; }
template <> float scale<float>(float x) { return x * 2.0f; }
template <> half scale<half>(half x) { return x; }
float use(float x) { return scale_float(x); }
"""
    pruned = prune_orphaned_explicit_specializations(MetalPreprocessor(), code)
    assert "template <>" not in pruned
    assert "float scale_float(float x)" in pruned
    assert "return scale_float(x);" in pruned

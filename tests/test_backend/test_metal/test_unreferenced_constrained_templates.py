"""Constrained function templates that nothing in the program names.

A function template that is never named is never instantiated ([temp.inst]/4). CrossGL generics carry no
constraint, so an ``enable_if`` overload family that no code calls used to reach the importer as several
unconstrained generics with one signature, or made it refuse the source when a body needed the choice only the
constraint decides.
"""

import pytest

import crosstl
from crosstl.backend.Metal.preprocessor import MetalPreprocessor
from crosstl.backend.Metal.unreferenced_constrained_templates import (
    prune_unreferenced_constrained_templates,
)

PREFIX = "#include <metal_stdlib>\nusing namespace metal;\n"

OVERLOAD_FAMILY = """
template <typename T, enable_if_t<is_floating_point_v<T>, bool> = true>
METAL_FUNC T twice(T value) {
  return value * T(2);
}

template <typename T, enable_if_t<!is_floating_point_v<T>, bool> = true>
METAL_FUNC T twice(T value) {
  return value << 1;
}
"""

ATOMIC_FAMILY = """
template <typename T>
constexpr constant bool is_atom = is_same_v<T, int> || is_same_v<T, uint> || is_same_v<T, float>;

template <typename T, typename = void>
struct holder {
  atomic<uint> val;
};

template <typename T>
struct holder<T, enable_if_t<is_atom<T>>> {
  atomic<T> val;
};

template <typename T, enable_if_t<is_atom<T>, bool> = true>
METAL_FUNC T hload(device holder<T>* object, size_t offset) {
  return atomic_load_explicit(&(object[offset].val), memory_order_relaxed);
}

template <typename T, enable_if_t<!is_atom<T>, bool> = true>
METAL_FUNC T hload(device holder<T>* object, size_t offset) {
  uint bits = atomic_load_explicit(&(object[offset].val), memory_order_relaxed);
  return static_cast<T>(bits);
}

template <typename T, typename Op>
METAL_FUNC T hupdate(device holder<T>* object, size_t offset) {
  Op op;
  return op(hload(object, offset));
}

template <typename T, enable_if_t<is_atom<T>, bool> = true>
METAL_FUNC void hmin(device holder<T>* object, T value, size_t offset) {
  atomic_fetch_min_explicit(&(object[offset].val), value, memory_order_relaxed);
}

template <>
METAL_FUNC void hmin<float>(device holder<float>* object, float value, size_t offset) {
  float expected = hload(object, offset);
  if (value < expected) {
    atomic_store_explicit(&(object[offset].val), value, memory_order_relaxed);
  }
}
"""


def _translate(tmp_path, source, backend):
    path = tmp_path / "kernel.metal"
    path.write_text(PREFIX + source)
    return crosstl.translate(
        str(path), backend=backend, format_output=False, source_backend="metal"
    )


def _prune(code):
    return prune_unreferenced_constrained_templates(MetalPreprocessor(), code)


def test_unreferenced_overload_family_is_removed():
    code = OVERLOAD_FAMILY + """
template <typename T> T scale(T value) { return value * T(3); }
float use(float x) { return x; }
"""
    pruned = _prune(code)
    assert "twice" not in pruned
    # CrossGL can express an unconstrained template, so it stays even when unused.
    assert "template <typename T> T scale(T value)" in pruned
    assert "float use(float x) { return x; }" in pruned


@pytest.mark.parametrize(
    "use",
    [
        "float use(float x) { return twice(x); }",
        "inline float unused_helper(float x) { return twice(x); }",
        "template <typename T> T quad(T x) { return twice(twice(x)); }\n"
        "float use(float x) { return quad(x); }",
        "template <typename T, enable_if_t<is_floating_point_v<T>, bool> = true>\nT twice(T value);",
        "struct S { float m(float x) { return twice(x); } };",
    ],
    ids=[
        "concrete",
        "unreachable-concrete",
        "via-used-template",
        "prototype",
        "member",
    ],
)
def test_a_name_spelled_outside_unused_templates_keeps_the_family(use):
    code = OVERLOAD_FAMILY + use + "\n"
    assert _prune(code) == code


def test_templates_and_specializations_only_unused_code_needs_go_with_it():
    pruned = _prune(ATOMIC_FAMILY + "float use(float x) { return x; }\n")
    for name in ("hload", "hupdate", "hmin"):
        assert name not in pruned
    assert "struct holder<T, enable_if_t<is_atom<T>>>" in pruned
    assert "constexpr constant bool is_atom" in pruned


def test_unconstrained_template_unrelated_to_the_family_stays():
    code = ATOMIC_FAMILY + """
template <typename T> T first(device holder<T>* object) { return T(0); }
float use(float x) { return x; }
"""
    pruned = _prune(code)
    assert "hload" not in pruned
    assert "template <typename T> T first(device holder<T>* object)" in pruned


def test_unused_overload_family_does_not_reach_crossgl(tmp_path):
    source = OVERLOAD_FAMILY + """
kernel void k(device float* out [[buffer(0)]], uint i [[thread_position_in_grid]]) {
  out[i] = float(i);
}
"""
    cgl = _translate(tmp_path, source, "cgl")
    assert "twice" not in cgl
    assert "kernel" in cgl or "compute" in cgl


def test_called_member_of_a_family_binds_and_the_rest_is_removed(tmp_path):
    source = OVERLOAD_FAMILY + """
template <typename T, enable_if_t<is_integral_v<T>, bool> = true>
METAL_FUNC T thrice(T value) {
  return value * T(3);
}

kernel void k(device float* out [[buffer(0)]], uint i [[thread_position_in_grid]]) {
  out[i] = twice(float(i));
}
"""
    cgl = _translate(tmp_path, source, "cgl")
    assert "twice_float(" in cgl
    assert "generic" not in cgl
    assert "thrice" not in cgl
    metal = _translate(tmp_path, source, "metal")
    assert "twice_float(" in metal
    assert "thrice" not in metal


def test_unused_atomic_family_over_a_partial_specialization_imports(tmp_path):
    source = ATOMIC_FAMILY + """
kernel void k(device holder<float>* h [[buffer(0)]], uint i [[thread_position_in_grid]]) {
  atomic_store_explicit(&(h[i].val), 1.0f, memory_order_relaxed);
}
"""
    for backend in ("cgl", "metal"):
        output = _translate(tmp_path, source, backend)
        for name in ("hload", "hupdate", "hmin"):
            assert name not in output

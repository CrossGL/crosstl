"""Free operator templates are instantiated for the types that concrete static assertions use.

A namespace-scope or structure-scope ``static_assert`` is concrete code that the CrossGL importer emits, so its
condition needs the operator instantiations it calls even when no concrete function uses those types.
"""

import re

import crosstl
from crosstl.backend.Metal.preprocessor import MetalPreprocessor
from crosstl.backend.Metal.static_assertion_sources import (
    concrete_static_assertion_conditions,
)

PRELUDE = """#include <metal_stdlib>
using namespace metal;

template <typename T>
struct pair_t {
  T real;
  T imag;
};

template <typename T>
constexpr pair_t<T> operator*(pair_t<T> a, pair_t<T> b) {
  return {a.real * b.real - a.imag * b.imag, a.real * b.imag + a.imag * b.real};
}
"""

KERNEL = """
kernel void k(device float* out [[buffer(0)]], uint i [[thread_position_in_grid]]) {
  pair_t<float> a{out[i], 2.0f};
  pair_t<float> c = a * a;
  out[i] = c.real;
}
"""

HALF_ASSERTION = "static_assert((pair_t<half>{1.0h, 2.0h} * pair_t<half>{3.0h, 4.0h}).real == -5.0h);"
HALF_HELPER = "crosstl_metal_operator_multiply__pair_t_half__pair_t_half"
FLOAT_HELPER = "crosstl_metal_operator_multiply__pair_t_float__pair_t_float"


def _to_crossgl(tmp_path, source):
    path = tmp_path / "unit.metal"
    path.write_text(source)
    return crosstl.translate(
        str(path), backend="cgl", format_output=False, source_backend="metal"
    )


def _helper_definitions(code, helper):
    """How many times ``helper`` is defined: a definition follows its return type."""
    return len(re.findall(rf"\bpair_t_\w+\s+{helper}\(", code))


def test_namespace_scope_assertion_instantiates_its_operator(tmp_path):
    code = _to_crossgl(tmp_path, PRELUDE + HALF_ASSERTION + "\n" + KERNEL)

    assert _helper_definitions(code, HALF_HELPER) == 1
    assert _helper_definitions(code, FLOAT_HELPER) == 1
    assertion = next(line for line in code.splitlines() if "static_assert" in line)
    assert f"{HALF_HELPER}(" in assertion
    assert "pair_t<half>" not in assertion


def test_structure_scope_assertion_instantiates_its_operator(tmp_path):
    holder = "struct holder {\n  float x;\n  " + HALF_ASSERTION + "\n};\n"
    kernel = KERNEL.replace("out[i] = c.real;", "holder h{c.real};\n  out[i] = h.x;")
    code = _to_crossgl(tmp_path, PRELUDE + holder + kernel)

    assert _helper_definitions(code, HALF_HELPER) == 1
    assertion = next(line for line in code.splitlines() if "static_assert" in line)
    assert f"{HALF_HELPER}(" in assertion


def test_template_and_commented_assertions_instantiate_nothing(tmp_path):
    template = (
        "template <typename T>\nT unused(T v) {\n"
        "  static_assert((pair_t<short>{1, 2} * pair_t<short>{3, 4}).real == -5);\n"
        "  return v;\n}\n"
        "// static_assert((pair_t<int>{1, 2} * pair_t<int>{3, 4}).real == -5);\n"
    )
    code = _to_crossgl(tmp_path, PRELUDE + template + KERNEL)

    assert _helper_definitions(code, FLOAT_HELPER) == 1
    assert "operator_multiply__pair_t_short" not in code
    assert "operator_multiply__pair_t_int" not in code


def test_condition_scan_skips_templates_comments_and_literals():
    source = (
        'constant char* note = "static_assert(false)";\n'
        "/* static_assert(0 == 1); */\n"
        "template <typename T> void f() { static_assert(sizeof(T) == 4); }\n"
        'static_assert(sizeof(int) == 4, "int is (4) bytes");\n'
        "struct s { static_assert(f(g(1), 2) == 3); };\n"
    )
    preprocessor = MetalPreprocessor()
    template_spans = preprocessor._find_template_declaration_spans(source)

    assert concrete_static_assertion_conditions(
        preprocessor, source, template_spans
    ) == ['sizeof(int) == 4, "int is (4) bytes"', "f(g(1), 2) == 3"]


def test_condition_scan_without_assertions_is_empty():
    preprocessor = MetalPreprocessor()

    assert (
        concrete_static_assertion_conditions(preprocessor, "kernel void k() {}", [])
        == []
    )

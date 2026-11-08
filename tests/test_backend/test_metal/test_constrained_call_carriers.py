"""Calls of constrained templates in the bodies of unconstrained templates bind like any other call.

``wrap<T>(a, b) { return max(a, b); }`` calls ``max`` with dependent arguments. When ``max`` is an
``enable_if``-constrained template, the importer instantiates ``wrap`` for each reachable call and binds the
``max`` call in the instantiation to the template overload resolution selects. Leaving ``wrap`` to the CrossGL
generic path loses that choice: the call then reaches the Metal standard library ``max``, or a constrained template
whose constraint does not hold. Every source here compiles with the Metal compiler.
"""

import re

import pytest

import crosstl

BACKENDS = ["metal", "directx", "opengl"]

# A namespaced float-only max, called directly and through wrap.
DIRECT_AND_CARRIED = """#include <metal_stdlib>
namespace ns {
template <typename T, ::metal::enable_if_t<::metal::is_floating_point_v<T>, bool> = true>
T max(T a, T b) {
  return a > b ? a : b + T(100);
}
template <typename T>
T wrap(T a, T b) {
  return max(a, b);
}
} // namespace ns
kernel void k(device float* o [[buffer(0)]], uint i [[thread_position_in_grid]]) {
  o[i] = ns::max(o[i], 1.0f) + ns::wrap(o[i], 2.0f);
}
"""

# An integral-only max: for float, wrap's call resolves to metal::max.
DISABLED_FOR_FLOAT = """#include <metal_stdlib>
using namespace metal;
template <typename T, enable_if_t<is_integral_v<T>, bool> = true>
T max(T a, T b) {
  return a > b ? a : b + T(100);
}
template <typename T>
T wrap(T a, T b) {
  return max(a, b);
}
kernel void k(device float* o [[buffer(0)]], uint i [[thread_position_in_grid]]) {
  o[i] = wrap(o[i], 2.0f);
}
"""

# outer calls wrap, which calls the constrained max.
NESTED = """#include <metal_stdlib>
namespace ns {
template <typename T, ::metal::enable_if_t<::metal::is_floating_point_v<T>, bool> = true>
T max(T a, T b) {
  return a > b ? a : b + T(100);
}
template <typename T>
T wrap(T a, T b) {
  return max(a, b);
}
template <typename T>
T outer(T a) {
  return wrap(a, T(3));
}
} // namespace ns
kernel void k(device float* o [[buffer(0)]], uint i [[thread_position_in_grid]]) {
  o[i] = ns::outer(o[i]);
}
"""

# h is concrete but only a template calls it, so the pass finds it unreachable.
REACHED_THROUGH_TEMPLATE = """#include <metal_stdlib>
namespace ns {
template <typename T, ::metal::enable_if_t<::metal::is_floating_point_v<T>, bool> = true>
T max(T a, T b) {
  return a > b ? a : b + T(100);
}
float h(float a) {
  return max(a, 4.0f);
}
template <typename T>
T foo(T a) {
  return h(a);
}
} // namespace ns
kernel void k(device float* o [[buffer(0)]], uint i [[thread_position_in_grid]]) {
  o[i] = ns::max(o[i], 1.0f) + ns::foo(o[i]);
}
"""

# h is reached only through foo, and its int call disables the float-only max, so metal::max is meant.
DISABLED_THROUGH_TEMPLATE = """#include <metal_stdlib>
using namespace metal;
template <typename T, enable_if_t<is_floating_point_v<T>, bool> = true>
T max(T a, T b) {
  return a > b ? a : b + T(100);
}
int h(int a) {
  return max(a, 4);
}
template <typename T>
T foo(T a) {
  return T(h(int(a)));
}
kernel void k(device float* o [[buffer(0)]], uint i [[thread_position_in_grid]]) {
  o[i] = max(o[i], 1.0f) + foo(o[i]);
}
"""

# h is reached through two templates, the outer one called with explicit template arguments.
REACHED_THROUGH_EXPLICIT_TEMPLATES = """#include <metal_stdlib>
namespace ns {
template <typename T, ::metal::enable_if_t<::metal::is_floating_point_v<T>, bool> = true>
T max(T a, T b) {
  return a > b ? a : b + T(100);
}
float h(float a) {
  return max(a, 4.0f);
}
template <typename T>
T inner(T a) {
  return h(a);
}
template <typename T>
T outer(T a) {
  return inner(a) * T(2);
}
} // namespace ns
kernel void k(device float* o [[buffer(0)]], uint i [[thread_position_in_grid]]) {
  o[i] = ns::max(o[i], 1.0f) + ns::outer<float>(o[i]);
}
"""


def _translate(tmp_path, source, backend):
    path = tmp_path / "carrier.metal"
    path.write_text(source)
    return crosstl.translate(str(path), backend=backend, format_output=False)


def _body(code, name):
    match = re.search(rf"\b{name}\s*\([^;{{}}()]*\)\s*\{{([^}}]*)\}}", code)
    assert match, f"no definition of {name}"
    return " ".join(match.group(1).split())


@pytest.mark.parametrize("backend", BACKENDS)
def test_carried_call_binds_the_constrained_template(tmp_path, backend):
    code = _translate(tmp_path, DIRECT_AND_CARRIED, backend)

    assert re.search(r"\bmax_float\(a, b\)", _body(code, "wrap_float"))
    assert "100" in _body(code, "max_float")
    assert len(re.findall(r"\bmax_float\s*\([^;{}()]*\)\s*\{", code)) == 1


@pytest.mark.parametrize("backend", BACKENDS)
def test_carried_call_with_a_disabled_constraint_reaches_the_library(tmp_path, backend):
    code = _translate(tmp_path, DISABLED_FOR_FLOAT, backend)

    assert re.search(r"(?<![\w.])max\(a, b\)", _body(code, "wrap_float"))
    assert "100" not in code


@pytest.mark.parametrize("backend", BACKENDS)
def test_nested_carriers_bind_through_every_level(tmp_path, backend):
    code = _translate(tmp_path, NESTED, backend)

    assert re.search(r"\bwrap_float\(", _body(code, "outer_float"))
    assert re.search(r"\bmax_float\(a, b\)", _body(code, "wrap_float"))
    assert "100" in _body(code, "max_float")


def test_vulkan_carried_calls_keep_source_and_library_semantics(tmp_path):
    carried = _translate(tmp_path, DIRECT_AND_CARRIED, "vulkan")
    disabled = _translate(tmp_path, DISABLED_FOR_FLOAT, "vulkan")

    assert not re.search(r"\b[FN]Max\b", carried)
    assert re.search(r"\bFMax\b", disabled)


@pytest.mark.parametrize(
    "source",
    [DIRECT_AND_CARRIED, DISABLED_FOR_FLOAT, NESTED],
    ids=["direct", "disabled", "nested"],
)
def test_crossgl_keeps_no_unconstrained_copy_of_a_constrained_template(
    tmp_path, source
):
    code = _translate(tmp_path, source, "cgl")

    assert "generic<" not in code


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize(
    "source",
    [REACHED_THROUGH_TEMPLATE, REACHED_THROUGH_EXPLICIT_TEMPLATES],
    ids=["template", "explicit-templates"],
)
def test_function_reached_through_a_template_binds_the_constrained_template(
    tmp_path, backend, source
):
    """A concrete function only a template body calls is reached, so its calls bind before the template goes."""
    code = _translate(tmp_path, source, backend)

    callee = re.match(r"return (\w+)\(a, 4\.0f?\);", _body(code, "h"))
    assert callee and callee.group(1) != "max"
    assert "100" in _body(code, callee.group(1))
    definitions = re.findall(r"\b(max\w*)\s*\([^;{}()]*\)\s*\{", code)
    assert len(definitions) == len(set(definitions))


@pytest.mark.parametrize("backend", BACKENDS)
def test_function_reached_through_a_template_keeps_a_disabled_call_native(
    tmp_path, backend
):
    """max(int, int) disables the float-only template, so h keeps the library max.

    The kernel's max(float, float) calls the library too: beside ``using namespace metal;`` the library's
    non-template max(float, float) matches as well as the template and wins ([over.match.best]).
    """
    code = _translate(tmp_path, DISABLED_THROUGH_TEMPLATE, backend)

    assert re.fullmatch(r"return max\(a, 4\);", _body(code, "h"))
    assert "100" not in code
    assert not re.search(r"\bmax_\w+\(", code)


@pytest.mark.parametrize(
    "source",
    [
        REACHED_THROUGH_TEMPLATE,
        DISABLED_THROUGH_TEMPLATE,
        REACHED_THROUGH_EXPLICIT_TEMPLATES,
    ],
    ids=["template", "disabled", "explicit-templates"],
)
def test_crossgl_drops_constrained_templates_reached_through_templates(
    tmp_path, source
):
    code = _translate(tmp_path, source, "cgl")

    assert not re.search(r"generic<[^>]*>\s*\w+\s+max\b", code)


@pytest.mark.parametrize(
    "code, expected",
    [
        ("inline half twice(half v) {", True),
        ("float twice(float v) const [[visible]]\n{", True),
        ("  return twice(v);", False),
        ("  if (twice(v)) {", False),
        ("half twice(half v);", False),
    ],
)
def test_function_definition_declarators_are_no_calls(code, expected):
    from crosstl.backend.Metal.constrained_template_retention import defines_function

    reference_end = code.index(")") + 1
    assert defines_function(code, reference_end) is expected


def test_explicit_specialization_does_not_retain_its_constrained_primary():
    from crosstl.backend.Metal.preprocessor import MetalPreprocessor

    code = """
#include <metal_stdlib>
namespace util {
template <typename T>
metal::enable_if_t<metal::is_floating_point_v<T>, T> twice(T v) {
  return v + v;
}
template <>
inline half twice(half v) {
  return v * 3.0h;
}
}
kernel void k(device float* f [[buffer(0)]], device half* h [[buffer(1)]],
              uint id [[thread_position_in_grid]]) {
  f[id] = util::twice(f[id]);
  h[id] = util::twice(h[id]);
}
"""
    output = MetalPreprocessor().preprocess(code)
    assert "::util::twice_float(f[id])" in output
    assert "::util::twice_half(h[id])" in output
    assert "enable_if" not in output

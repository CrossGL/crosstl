"""Calls whose result type is named through an alias template or through the callee's template arguments.

A materialized instantiation of ``template <typename T, typename U> common_dtype<T, U> div(T, U)`` declares its
result as ``common_dtype<float, float>``, which is ``float`` ([temp.alias]/2), so ``auto q = div(b, a)`` declares a
``float``. A call to a function template has the return type of the specialization it selects: the declared return
type with the explicit template arguments, the deduced ones and the declaration defaults substituted
([temp.arg.explicit], [temp.deduct.call], [temp.deduct]/5). The callee's own ``T`` is not a type of the calling scope.
"""

import platform
import re
import shutil
import subprocess
from types import SimpleNamespace

import pytest

import crosstl
from crosstl.backend.Metal.alias_template_types import (
    expand_concrete_alias_template_references,
)
from crosstl.backend.Metal.generic_call_return_types import (
    deduce_call_return_type,
    same_arity_overloads,
)
from crosstl.backend.Metal.preprocessor import MetalPreprocessor

PRELUDE = "#include <metal_stdlib>\n"

KERNEL = """
kernel void k(device float* o [[buffer(0)]], uint i [[thread_position_in_grid]]) {
  o[i] = hyp(o[i], 2.0f);
}
"""

DEDUCED = PRELUDE + """
template <typename T>
inline T mul(const T x, const T y) {
  return x * y;
}
inline float hyp(float a, float b) {
  auto r = mul(a, b);
  auto s = ::metal::precise::sqrt(1 + r);
  return s;
}
""" + KERNEL

QUALIFIED = PRELUDE + """
namespace outer {
namespace inner {
template <typename T, typename U>
inline T mul(const T x, const U y) {
  return x * y;
}
inline float hyp(float a, float b) {
  auto r = outer::inner::mul(a, b);
  return ::metal::precise::sqrt(1 + r);
}
}
}
kernel void k(device float* o [[buffer(0)]], uint i [[thread_position_in_grid]]) {
  o[i] = outer::inner::hyp(o[i], 2.0f);
}
"""

# The shape of a numerically stable hypot: constrained overloads whose result is an alias template of a decltype.
ALIAS_RETURN = PRELUDE + """
namespace ns {
template <typename T>
constexpr constant bool is_pair_v = ::metal::is_same_v<T, float2>;
template <typename U, typename V>
using common_dtype = decltype(U(0) + V(0));
template <typename T, typename U, ::metal::enable_if_t<!is_pair_v<T>, bool> = true>
inline common_dtype<T, U> mul(const T x, const U y) {
  return x * y;
}
template <typename T, typename U, ::metal::enable_if_t<is_pair_v<T> && is_pair_v<U>, bool> = true>
inline common_dtype<T, U> mul(const T x, const U y) {
  return T(x.x * y.x - x.y * y.y, x.x * y.y + x.y * y.x);
}
template <typename T, typename U, ::metal::enable_if_t<!is_pair_v<T>, bool> = true>
inline common_dtype<T, U> div(const T x, const U y) {
  return x / y;
}
inline float hyp(float a, float b) {
  auto q = ns::div(b, a);
  auto r = ns::mul(q, q);
  return a * ::metal::precise::sqrt(1 + r);
}
}
kernel void k(device float* o [[buffer(0)]], uint i [[thread_position_in_grid]]) {
  o[i] = ns::hyp(o[i], 2.0f);
}
"""

CONCRETE_ALIAS = PRELUDE + """
template <typename U, typename V>
using common_dtype = decltype(U(0) + V(0));
inline common_dtype<float, int> f(float a, int b) { return a + b; }
kernel void k(device float* o [[buffer(0)]], uint i [[thread_position_in_grid]]) {
  common_dtype<half, int> h = half(o[i]);
  auto r = f(o[i], 2);
  o[i] = ::metal::precise::sqrt(r) + float(h);
}
"""

MIXED = PRELUDE + """
template <typename T, typename U>
U pick(T x, U y) { return y + U(x); }
template <typename R, typename T>
R conv(T x) { return R(x); }
template <typename T, typename R = float>
R widen(T x) { return R(x) * R(2); }
kernel void k(device float* o [[buffer(0)]], device half* h [[buffer(1)]],
              uint i [[thread_position_in_grid]]) {
  half hv = h[i];
  auto a = pick(hv, o[i]);
  auto b = conv<half>(o[i]);
  auto c = widen(hv);
  auto d = a + c;
  h[i] = b;
  o[i] = ::metal::precise::sqrt(d);
}
"""

ENCLOSING = PRELUDE + """
template <typename T>
T mul(T x, T y) { return x * y; }
template <typename T>
T twice_square(T x) {
  auto y = mul(x, x);
  return y + y;
}
kernel void k(device float* o [[buffer(0)]], device half* h [[buffer(1)]],
              uint i [[thread_position_in_grid]]) {
  o[i] = twice_square(o[i]);
  h[i] = twice_square(h[i]);
}
"""


def _translate(tmp_path, source, backend):
    path = tmp_path / "unit.metal"
    path.write_text(source)
    return crosstl.translate(
        str(path), backend=backend, format_output=False, source_backend="metal"
    )


def _compile_metal_if_available(code, tmp_path):
    xcrun = shutil.which("xcrun")
    if platform.system() != "Darwin" or xcrun is None:
        return
    shader = tmp_path / "out.metal"
    shader.write_text(code)
    subprocess.run(
        [xcrun, "-sdk", "macosx", "metal", "-std=metal3.1", "-c", str(shader)],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )


def _declared_type(code, name):
    match = re.search(rf"(\w+) {name} = ", code)
    assert match is not None, code
    return match.group(1)


@pytest.mark.parametrize("source", [DEDUCED, QUALIFIED, ALIAS_RETURN])
def test_auto_local_takes_the_call_result_type(tmp_path, source):
    crossgl = _translate(tmp_path, source, "cgl")
    assert _declared_type(crossgl, "r") == "float"
    assert not re.search(r"\bT r\b", crossgl)
    metal = _translate(tmp_path, source, "metal")
    assert _declared_type(metal, "r") == "float"
    _compile_metal_if_available(metal, tmp_path)


def test_materialized_alias_template_results_are_spelled_as_their_type(tmp_path):
    preprocessed = MetalPreprocessor().preprocess(ALIAS_RETURN)
    assert "common_dtype<float, float>" not in preprocessed
    assert re.search(r"inline float div_float_float\(", preprocessed)
    crossgl = _translate(tmp_path, ALIAS_RETURN, "cgl")
    assert _declared_type(crossgl, "q") == "float"
    assert "common_dtype" not in crossgl


def test_concrete_alias_template_references_name_their_type(tmp_path):
    crossgl = _translate(tmp_path, CONCRETE_ALIAS, "cgl")
    assert _declared_type(crossgl, "h") == "float16"
    assert _declared_type(crossgl, "r") == "float"
    assert "decltype" not in crossgl
    metal = _translate(tmp_path, CONCRETE_ALIAS, "metal")
    _compile_metal_if_available(metal, tmp_path)


def test_explicit_deduced_and_default_arguments_type_the_call(tmp_path):
    crossgl = _translate(tmp_path, MIXED, "cgl")
    assert _declared_type(crossgl, "a") == "float"
    assert _declared_type(crossgl, "b") == "float16"
    assert _declared_type(crossgl, "c") == "float"
    assert _declared_type(crossgl, "d") == "float"
    metal = _translate(tmp_path, MIXED, "metal")
    assert _declared_type(metal, "b") == "half"
    _compile_metal_if_available(metal, tmp_path)


def test_callee_parameter_bound_to_enclosing_parameter_keeps_its_spelling(tmp_path):
    crossgl = _translate(tmp_path, ENCLOSING, "cgl")
    assert _declared_type(crossgl, "y") == "T"
    metal = _translate(tmp_path, ENCLOSING, "metal")
    assert "float y = " in metal and "half y = " in metal
    _compile_metal_if_available(metal, tmp_path)


SCOPES = """namespace a {
template <typename U, typename V>
using sum_t = decltype(U(0) + V(0));
}
namespace b {
template <typename U, typename V>
using sum_t = decltype(U(0) * V(0) * 2.0f);
inline sum_t<int, int> g(int x) { return x; }
}
inline a::sum_t<half, int> f(half x) { return x; }
namespace c { inline sum_t<int, int> h(int x) { return x; } }
template <typename T> inline a::sum_t<T, int> d(T x) { return x; }
struct S {
  template <typename U, typename V>
  using pair_t = U;
  pair_t<int, float> m;
};
template <typename U, typename V>
using pair_t = decltype(U(0) + V(0));
inline pair_t<int, float> k(int x) { return x; }
"""


def test_alias_expansion_follows_lookup_and_leaves_hidden_names():
    expanded = expand_concrete_alias_template_references(MetalPreprocessor(), SCOPES)
    # An unqualified reference sees the alias of its enclosing namespace; a qualified one names its own.
    assert "inline float g(int x)" in expanded
    assert "inline half f(half x)" in expanded
    # No alias template sum_t is visible in namespace c.
    assert "inline sum_t<int, int> h(int x)" in expanded
    # Dependent arguments stay.
    assert "a::sum_t<T, int> d(T x)" in expanded
    # A member alias template of the same name hides the namespace-scope one inside its aggregate.
    assert "pair_t<int, float> m;" in expanded
    assert "inline pair_t<int, float> k(int x)" in expanded


class _Converter:
    """The converter services the deduction uses, over a fixed argument typing."""

    def __init__(self, argument_types, overloads=None):
        self.argument_types = argument_types
        self.overloads = overloads or {}

    def expression_metal_type(self, argument):
        return self.argument_types.get(argument)

    @staticmethod
    def metal_declaration_expression_type(parameter):
        return parameter.vtype

    @staticmethod
    def normalized_metal_type(text):
        return str(text).strip()

    @staticmethod
    def split_generic_arguments(text):
        return [part.strip() for part in text.split(",")]

    def metal_user_function_overloads(self, name):
        return list(self.overloads.get(name, []))


def _function(return_type, template_parameters, params, defaults=None, name="f"):
    return SimpleNamespace(
        name=name,
        return_type=return_type,
        template_parameters=template_parameters,
        template_parameter_defaults=defaults or {},
        params=[SimpleNamespace(name=n, vtype=t) for n, t in params],
    )


def _call(name, *args):
    return SimpleNamespace(name=name, args=list(args))


def test_bindings_take_explicit_then_deduced_then_default_arguments():
    converter = _Converter({"x": "half", "y": "int"})
    function = _function(
        "vec<R, N>",
        [("typename", "R"), ("typename", "T"), ("int", "N")],
        [("a", "T"), ("b", "R")],
        defaults={"N": "sizeof(R)"},
    )
    call = _call("f<float>", "x", "y")
    assert deduce_call_return_type(
        converter, function, call.args, "vec<R, N>", call
    ) == (
        "vec<float, sizeof(float)>",
        (),
        False,
    )


def test_explicit_arguments_of_another_callee_or_a_pack_bind_nothing():
    converter = _Converter({"x": "half"})
    function = _function("R", [("typename", "R"), ("typename", "T")], [("a", "T")])
    other = _call("g<float>", "x")
    assert deduce_call_return_type(converter, function, other.args, "R", other) == (
        None,
        ("R",),
        False,
    )
    pack = _function("R", [("typename", "R"), ("typename...", "Ts")], [("a", "Ts")])
    call = _call("f<float>", "x")
    assert deduce_call_return_type(converter, pack, call.args, "R", call)[0] is None


def test_conflicting_deduction_stays_dependent():
    converter = _Converter({"x": "float", "y": "half"})
    function = _function("T", [("typename", "T")], [("a", "T"), ("b", "T")])
    call = _call("f", "x", "y")
    assert deduce_call_return_type(converter, function, call.args, "T", call) == (
        None,
        (),
        True,
    )


def test_same_arity_overloads_look_through_explicit_template_arguments():
    template = _function("R", [("typename", "R")], [("a", "float")], name="conv")
    concrete = _function("float", [], [("a", "float")], name="conv")
    converter = _Converter({}, {"conv": [template, concrete]})
    assert same_arity_overloads(converter, _call("conv<half>", "x")) == [template]
    assert same_arity_overloads(converter, _call("conv", "x", "y")) == []

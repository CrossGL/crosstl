"""Function templates whose result type is an expression of their parameters.

A Metal function template can declare its result through ``decltype`` of its template or function parameters,
directly, through an alias template ([temp.alias]/2) or as a trailing return type ([dcl.fct]/2). CrossGL generic
functions declare types, not expressions, so every reachable call is materialized and the instantiated ``decltype``
of built-in operands is folded to the type it denotes ([dcl.type.decltype]/1.5, [expr.arith.conv]). A call that
cannot be materialized is refused instead of printing ``decltype`` as a type.
"""

import re
import textwrap

import pytest

import crosstl
from crosstl.backend.Metal.alias_template_types import (
    expand_alias_template_types,
    find_alias_templates,
    fold_expression_decltypes,
)
from crosstl.backend.Metal.dependent_result_templates import (
    dependent_result_template_names,
    has_dependent_decltype,
)
from crosstl.backend.Metal.preprocessor import MetalPreprocessor
from crosstl.translator.codegen.generic_result_types import GenericResultTypeError

PRELUDE = "#include <metal_stdlib>\nusing namespace metal;\n"

DIRECT = PRELUDE + textwrap.dedent("""\
    template <typename T, typename U>
    inline decltype(T(0) + U(0)) addv(const T x, const U y) { return x + y; }
    kernel void k(device float* out [[buffer(0)]], device int* n [[buffer(1)]],
                  uint i [[thread_position_in_grid]]) {
      auto q = addv(out[i], 2.0f);
      auto r = addv(n[i], 1.5f);
      out[i] = q + r;
    }
""")

ALIAS = PRELUDE + textwrap.dedent("""\
    template <typename U, typename V>
    using common_dtype = decltype(U(0) + V(0));
    template <typename T, typename U>
    inline common_dtype<T, U> addv(const T x, const U y) { return x + y; }
    kernel void k(device float* out [[buffer(0)]], device int* n [[buffer(1)]],
                  uint i [[thread_position_in_grid]]) {
      auto q = addv(out[i], 2.0f);
      auto r = addv(n[i], 1.5f);
      out[i] = q + r;
    }
""")

TRAILING_AND_CONSTRAINED = PRELUDE + textwrap.dedent("""\
    template <typename T>
    inline auto twice(const T x) -> decltype(x + x) { return x + x; }
    template <typename T, typename U>
    inline decltype(T(0) * U(0)) mulv(const T x, const U y) { return x * y; }
    template <typename T, metal::enable_if_t<metal::is_floating_point_v<T>, bool> = true>
    inline T halve(const T v) { return v * T(0.5); }
    kernel void k(device float* out [[buffer(0)]], device int* n [[buffer(1)]],
                  uint i [[thread_position_in_grid]]) {
      auto a = mulv(n[i], 2.0f);
      auto b = twice(n[i]);
      out[i] = halve(a) + float(b);
    }
""")

PROMOTIONS = PRELUDE + textwrap.dedent("""\
    template <typename T>
    inline auto twice(const T x) -> decltype(x + x) { return x + x; }
    template <typename T, typename U>
    inline decltype(T(0) + U(0)) addv(const T x, const U y) { return x + y; }
    template <typename T, typename U>
    inline decltype(T(0) - U(0)) unused_sub(const T x, const U y) { return x - y; }
    kernel void k(device int* out [[buffer(0)]], device short* s [[buffer(1)]],
                  device half* h [[buffer(2)]], uint i [[thread_position_in_grid]]) {
      auto a = twice(s[i]);
      auto b = addv(h[i], h[i]);
      out[i] = a + int(b);
    }
""")

CONCRETE_DECLTYPE = PRELUDE + textwrap.dedent("""\
    inline decltype(1.0f + 2) scale(float v) { return v * 2; }
    kernel void k(device float* out [[buffer(0)]], uint i [[thread_position_in_grid]]) {
      out[i] = scale(out[i]);
    }
""")

CONCRETE_TRAILING = PRELUDE + textwrap.dedent("""\
    inline auto twice_int(const int x) -> decltype(x + x) { return x + x; }
    kernel void k(device float* out [[buffer(0)]], device int* n [[buffer(1)]],
                  uint i [[thread_position_in_grid]]) {
      auto b = twice_int(n[i]);
      out[i] = float(b);
    }
""")

GENERIC_AUTO_LOCAL = PRELUDE + textwrap.dedent("""\
    template <typename T>
    inline T twice(const T x) { return x + x; }
    kernel void k(device float* out [[buffer(0)]], uint i [[thread_position_in_grid]]) {
      auto q = twice(out[i]);
      out[i] = q;
    }
""")

OVERLOADED = PRELUDE + textwrap.dedent("""\
    template <typename T, typename U>
    inline decltype(T(0) + U(0)) addv(const T x, const U y) { return x + y; }
    inline float addv(float x) { return x; }
    kernel void k(device float* out [[buffer(0)]], device int* n [[buffer(1)]],
                  uint i [[thread_position_in_grid]]) {
      out[i] = addv(n[i], 1.5f) + addv(out[i]);
    }
""")


def _translate(tmp_path, source, backend, suffix="metal"):
    path = tmp_path / f"shader.{suffix}"
    path.write_text(source)
    return crosstl.translate(str(path), backend=backend)


def _compact(code):
    return re.sub(r"\s+", " ", code)


@pytest.mark.parametrize(
    "text,expected",
    (
        ("decltype(float(0) + int(0)) f()", "float f()"),
        ("decltype(int(0) * float(0)) f()", "float f()"),
        ("decltype(half(0) + half(0)) f()", "half f()"),
        ("decltype(short(0) + short(0)) f()", "int f()"),
        ("decltype(uint(0) + int(0)) f()", "uint f()"),
        ("decltype(float2(0) * float(0)) f()", "float2 f()"),
        ("decltype(1.0f + 2) f()", "float f()"),
    ),
)
def test_decltype_of_builtin_operands_folds_to_its_type(text, expected):
    assert fold_expression_decltypes(MetalPreprocessor(), text) == expected


@pytest.mark.parametrize(
    "text",
    (
        "decltype(x + x) f(int x)",
        "decltype(T(0) + U(0)) f()",
        "decltype(g(1)) f()",
    ),
)
def test_decltype_that_names_anything_else_stays(text):
    assert fold_expression_decltypes(MetalPreprocessor(), text) == text


def test_alias_template_reference_expands_to_the_folded_type():
    preprocessor = MetalPreprocessor()
    aliases = find_alias_templates(preprocessor, ALIAS)
    assert sorted(aliases) == ["common_dtype"]
    header = (
        "inline common_dtype<int, float> addv_int_float(const int x, const float y)"
    )
    assert expand_alias_template_types(preprocessor, header, aliases) == (
        "inline float addv_int_float(const int x, const float y)"
    )
    unbound = "inline common_dtype<T, float> f(const T x)"
    assert expand_alias_template_types(preprocessor, unbound, aliases) == unbound


def test_dependent_decltype_names_a_parameter():
    assert has_dependent_decltype("decltype(T(0) + U(0))", ["T", "U"])
    assert has_dependent_decltype("auto  decltype(x + x)", ["T", "x"])
    assert not has_dependent_decltype("decltype(1.0f + 2)", ["T"])
    assert not has_dependent_decltype("T", ["T"])


@pytest.mark.parametrize(
    "source,expected",
    (
        (DIRECT, {"addv"}),
        (ALIAS, {"addv"}),
        (TRAILING_AND_CONSTRAINED, {"mulv", "twice"}),
        (GENERIC_AUTO_LOCAL, set()),
    ),
)
def test_dependent_result_templates_are_found(source, expected):
    preprocessor = MetalPreprocessor()
    templates = preprocessor._find_template_functions(source)
    assert dependent_result_template_names(preprocessor, source, templates) == expected


@pytest.mark.parametrize("source", (DIRECT, ALIAS), ids=("decltype", "alias"))
@pytest.mark.parametrize("backend", ("directx", "opengl", "metal"))
def test_each_call_gets_an_instantiation_with_a_folded_result(
    tmp_path, source, backend
):
    code = _compact(_translate(tmp_path, source, backend))
    assert "decltype" not in code
    assert "common_dtype" not in code
    assert re.search(
        r"float addv_float_float\((const )?float x, (const )?float y\)", code
    )
    assert re.search(r"float addv_int_float\((const )?int x, (const )?float y\)", code)
    assert re.search(r"float q = addv_float_float\(", code)
    assert re.search(r"float r = addv_int_float\(", code)


def test_trailing_decltype_of_function_parameters_and_constrained_calls(tmp_path):
    code = _compact(_translate(tmp_path, TRAILING_AND_CONSTRAINED, "directx"))
    assert "decltype" not in code
    assert "float mulv_int_float(int x, float y)" in code
    assert "int twice_int(int x)" in code
    assert "float halve_float(float v)" in code
    assert "float a = mulv_int_float(" in code
    assert "int b = twice_int(" in code


def test_results_follow_integral_promotion_and_half_arithmetic(tmp_path):
    code = _compact(_translate(tmp_path, PROMOTIONS, "directx"))
    assert "decltype" not in code
    # short + short is int ([conv.prom]); half + half stays half.
    assert "int twice_short(int16_t x)" in code
    assert "float16_t addv_half_half(float16_t x, float16_t y)" in code
    assert "int a = twice_short(" in code
    assert "float16_t b = addv_half_half(" in code
    # An unreachable dependent template has no instantiation to declare.
    assert "unused_sub" not in code


def test_concrete_decltype_return_type_is_a_function_definition(tmp_path):
    code = _compact(_translate(tmp_path, CONCRETE_DECLTYPE, "directx"))
    assert "float scale(float v)" in code
    assert "decltype" not in code


@pytest.mark.parametrize("backend", ("cgl", "directx"))
def test_trailing_decltype_of_a_concrete_function_types_auto_locals(tmp_path, backend):
    code = _compact(_translate(tmp_path, CONCRETE_TRAILING, backend))
    assert "decltype" not in code
    assert "int b = twice_int(" in code


def test_generic_call_types_an_auto_local_from_its_argument(tmp_path):
    code = _compact(_translate(tmp_path, GENERIC_AUTO_LOCAL, "cgl"))
    assert "generic<T> T twice(T x)" in code
    assert "float q = twice(" in code
    assert "T q" not in code


def test_overloaded_dependent_template_materializes_for_non_generic_targets(
    tmp_path,
):
    code = _compact(_translate(tmp_path, OVERLOADED, "directx"))
    assert "float addv_int_float(int x, float y)" in code
    assert "addv_int_float(n.Load(i), 1.5)" in code
    assert "decltype" not in code


def test_generic_specialization_with_an_expression_result_is_refused(tmp_path):
    with pytest.raises(GenericResultTypeError, match=r"decltype\(T\(0\) \+ U\(0\)\)"):
        _translate(tmp_path, OVERLOADED, "metal")

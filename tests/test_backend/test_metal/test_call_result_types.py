"""Constrained overload resolution types arguments that call source functions."""

import pytest

import crosstl
from crosstl.backend.Metal.call_result_types import (
    CallResultScope,
    collect_call_result_types,
    infer_call_result_type,
)
from crosstl.backend.Metal.preprocessor import MetalPreprocessor
from tests.stage_entry_helpers import joined_entries

PRELUDE = """#include <metal_stdlib>
using namespace metal;
"""


def _infer(source, expression, local_types=None):
    preprocessor = MetalPreprocessor()
    with preprocessor._call_result_scope.active(PRELUDE + source):
        return preprocessor._infer_argument_type(
            expression, {}, dict(local_types or {"i": "int"})
        )


@pytest.mark.parametrize(
    "source,expression,expected",
    (
        ("float lin(int i) { return float(i); }", "lin(i)", "float"),
        ("uint2 pair(int i) { return uint2(i); }", "pair(i)", "uint2"),
        # A template whose return type does not name a template parameter.
        (
            "template <typename I> inline float lin(I i) { return float(i); }",
            "lin(i)",
            "float",
        ),
        (
            "template <typename I> inline float lin(I i) { return float(i); }",
            "lin<int>(i)",
            "float",
        ),
        # Overloads agreeing on the return type.
        (
            "float lin(int i) { return float(i); }\n"
            "float lin(uint i) { return float(i); }\n"
            "template <typename I> float lin(I i, I j) { return float(i + j); }",
            "lin(i)",
            "float",
        ),
        (
            "namespace detail { half lin(int i) { return half(i); } }",
            "detail::lin(i)",
            "half",
        ),
        (
            "namespace a { namespace b { float lin(int i) { return float(i); } } }",
            "::a::b::lin(i)",
            "float",
        ),
    ),
)
def test_call_to_source_function_has_declared_return_type(source, expression, expected):
    assert _infer(source, expression) == expected


@pytest.mark.parametrize(
    "source,expression",
    (
        # Dependent, deduced, reference, pointer and void return types.
        ("template <typename T> T lin(T i) { return i; }", "lin(i)"),
        ("auto lin(int i) { return float(i); }", "lin(i)"),
        ("thread float& lin(thread float& i) { return i; }", "lin(i)"),
        ("void lin(int i) { }", "lin(i)"),
        # Overloads that disagree.
        (
            "float lin(int i) { return float(i); }\n" "half lin(half i) { return i; }",
            "lin(i)",
        ),
        (
            "float lin(int i) { return float(i); }\n"
            "template <typename T> T lin(T i, T j) { return i + j; }",
            "lin(i)",
        ),
        # A prototype whose return type the definition scan did not read.
        ("half lin(half i);\nfloat lin(int i) { return float(i); }", "lin(i)"),
        # Prototypes are not matched to definitions, so even one that agrees
        # leaves the name untyped.
        ("float lin(int i);\nfloat lin(int i) { return float(i); }", "lin(i)"),
        # Library names: the library's overloads are not in the source.
        ("float dot(int i) { return float(i); }", "dot(i)"),
        ("float length(int i) { return float(i); }", "length(i)"),
        ("float quad_reduce(int i) { return float(i); }", "quad_reduce(i)"),
        ("float mulhi(int i) { return float(i); }", "mulhi(i)"),
        # Not exactly one call.
        ("float lin(int i) { return float(i); }", "lin(i).x"),
        ("float lin(int i) { return float(i); }", "lin(i)(i)"),
        ("float lin(int i) { return float(i); }", "lin(i"),
        # A qualifier that is not a source namespace may name a struct.
        ("float lin(int i) { return float(i); }", "S::lin(i)"),
        # Undeclared callees.
        ("float lin(int i) { return float(i); }", "other(i)"),
    ),
)
def test_call_result_stays_untyped(source, expression):
    assert _infer(source, expression) is None


def test_local_variable_named_like_a_function_is_not_typed():
    source = "float lin(int i) { return float(i); }"
    assert _infer(source, "lin(i)", {"i": "int", "lin": "Functor"}) is None


def test_member_templates_are_not_free_candidates():
    source = (
        "struct S { template <typename T> half lin(T i) { return half(i); } };\n"
        "float lin(int i) { return float(i); }"
    )
    assert _infer(source, "lin(i)") == "float"


def test_inactive_scope_types_nothing():
    preprocessor = MetalPreprocessor()
    assert infer_call_result_type(preprocessor, "lin(i)", CallResultScope(), {}) is None


def test_scope_indexes_lazily_and_restores():
    preprocessor = MetalPreprocessor()
    scope = preprocessor._call_result_scope
    outer = PRELUDE + "float lin(int i) { return float(i); }"
    inner = PRELUDE + "half lin(int i) { return half(i); }"
    with scope.active(outer):
        with scope.active(inner):
            assert scope.types(preprocessor).types == {"lin": "half"}
        assert scope.types(preprocessor).types == {"lin": "float"}
    assert scope.types(preprocessor).types == {}


def test_collect_reports_source_namespaces():
    preprocessor = MetalPreprocessor()
    indexed = collect_call_result_types(
        preprocessor,
        PRELUDE
        + "namespace c10 { namespace metal { float lin(int i) { return i; } } }",
    )
    assert indexed.types == {"lin": "float"}
    assert {"c10", "metal"} <= indexed.namespaces


CONSTRAINED = PRELUDE + """
template <typename T, typename U, enable_if_t<is_same_v<U, T>, bool> = true>
inline T cast_to(const U from) { return from; }
template <typename T, typename U, enable_if_t<!is_same_v<U, T>, bool> = true>
inline T cast_to(const U from) { return static_cast<T>(from); }

template <typename I>
inline float lin(I i, constant float* v) { return v[0] * static_cast<float>(i); }

template <typename T>
kernel void k(device T* out [[buffer(0)]], constant float* v [[buffer(1)]],
              uint index [[thread_position_in_grid]]) {
  const int i = static_cast<int>(index);
  out[i] = cast_to<T>(lin(i, v));
}
template [[host_name("k_float")]] kernel void k<float>(device float*, constant float*, uint);
template [[host_name("k_half")]] kernel void k<half>(device half*, constant float*, uint);
"""


@pytest.mark.parametrize("backend", ("directx", "opengl"))
def test_constrained_call_with_helper_call_argument_selects_each_overload(
    tmp_path, backend
):
    path = tmp_path / "cast.metal"
    path.write_text(CONSTRAINED)
    output = joined_entries(path, backend, format_output=False)
    # float to float is the identity overload; float to half converts.
    assert "cast_to_float_float" in output
    assert "cast_to_half_float" in output


def test_constrained_call_with_helper_call_argument_reaches_spirv(tmp_path):
    path = tmp_path / "cast.metal"
    path.write_text(CONSTRAINED)
    output = crosstl.translate(str(path), backend="vulkan", format_output=False)
    assert output.count("OpEntryPoint GLCompute") == 2

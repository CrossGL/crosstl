"""CUDA and HIP device function templates import as CrossGL generic functions.

The importers dropped ``template <typename T>`` and wrote ``T twice(T x)``, so ``T`` was an undeclared type in every
target (the ``generic-function`` coverage cell of CUDA and HIP input), and HIP dropped a call's template arguments
(``widen<int, double>(2)`` became ``widen(2)``). Each test imports a source, checks the generic prefix and the
call's CrossGL template arguments, and translates the CrossGL to every target; the execution test runs the Metal
translation on this machine's GPU when it has one.
"""

from __future__ import annotations

import os
import sys

import pytest

import crosstl
from crosstl.backend.CUDA.function_templates import (
    TemplateParameter,
    parse_template_parameters,
)
from crosstl.backend.import_refusals import ImportRefused
from crosstl.translator.lexer import Lexer as CrossGLLexer
from crosstl.translator.parser import Parser as CrossGLParser

TARGETS = ("cuda", "hip", "metal", "directx", "opengl", "vulkan")
REQUIRE_ENV = "CROSSTL_REQUIRE_METAL_FUNCTION_TEMPLATES"

SOURCE = """
template <typename T>
__device__ T twice(T x) { return x + x; }
template <unsigned K>
__device__ int scaled(int x) { return x * (int)K; }
template <int K>
__device__ int offset(int x) { return x + K; }
template <typename T, typename U = int>
__device__ U widen(T x) { return (U)x + (U)1; }
__global__ void probe(int* data) {
    int x = data[0];
    data[1] = twice(x);
    data[2] = twice<int>(7);
    data[3] = scaled<3>(x);
    data[4] = offset<-2>(x);
    data[5] = widen<int>(x);
    data[6] = (int)widen<float, float>(2.5f);
    data[7] = (int)(twice(1.25f) * 4.0f);
}
"""


def _source(language):
    if language == "hip":
        return "#include <hip/hip_runtime.h>\n" + SOURCE
    return SOURCE


def _import(tmp_path, language, source=None):
    path = tmp_path / ("probe.hip" if language == "hip" else "probe.cu")
    path.write_text(source if source is not None else _source(language))
    return crosstl.translate(str(path), backend="cgl", format_output=False)


def test_parses_template_parameter_lists():
    assert parse_template_parameters("<typename T, class U = float>") == [
        TemplateParameter("typename", "T"),
        TemplateParameter("typename", "U", "float"),
    ]
    assert parse_template_parameters("<int N, unsigned int M = 4>") == [
        TemplateParameter("int", "N"),
        TemplateParameter("unsigned int", "M", "4"),
    ]
    assert parse_template_parameters("<typename... Ts>") == [
        TemplateParameter("typename", "Ts", pack=True)
    ]
    assert parse_template_parameters("<class...Ts>")[0].pack
    assert parse_template_parameters("<unsigned int>") == [
        TemplateParameter("unsigned int", None)
    ]
    assert parse_template_parameters("<template <typename> class C>")[0].kind == (
        "template"
    )
    assert parse_template_parameters("<>") == []


@pytest.mark.parametrize("language", ["cuda", "hip"])
def test_function_templates_import_as_generics(tmp_path, language):
    crossgl = _import(tmp_path, language)

    assert "generic<T> T twice(T x)" in crossgl
    assert "generic<K> i32 scaled(i32 x)" in crossgl
    assert "generic<T, U = i32> U widen(T x)" in crossgl
    assert "twice(x)" in crossgl
    assert "twice<i32>(7)" in crossgl
    # The value an unsigned parameter takes is unsigned ([temp.arg.nontype]/2).
    assert "scaled<3u>(x)" in crossgl
    assert "offset<-2>(x)" in crossgl
    assert "widen<i32>(x)" in crossgl
    assert "widen<f32, f32>(2.5f)" in crossgl

    path = tmp_path / "probe.cgl"
    path.write_text(crossgl)
    for target in TARGETS:
        assert crosstl.translate(str(path), backend=target, format_output=False)


@pytest.mark.parametrize("language", ["cuda", "hip"])
def test_a_value_parameter_default_is_refused(tmp_path, language):
    source = "template <int K = 3>\n__device__ int f(int x) { return x + K; }\n"
    if language == "hip":
        source = "#include <hip/hip_runtime.h>\n" + source
    with pytest.raises(
        ImportRefused, match="function template 'f' gives the value parameter 'int K'"
    ):
        _import(tmp_path, language, source)


@pytest.mark.parametrize("language", ["cuda", "hip"])
@pytest.mark.parametrize(
    "header", ["template <typename... Ts>", "template <template <typename> class C>"]
)
def test_parameters_without_a_generic_form_keep_the_function_as_it_was(
    tmp_path, language, header
):
    source = f"{header}\n__device__ int f(int x) {{ return x; }}\n"
    if language == "hip":
        source = "#include <hip/hip_runtime.h>\n" + source
    crossgl = _import(tmp_path, language, source)
    assert "generic<" not in crossgl
    assert "i32 f(i32 x)" in crossgl


@pytest.mark.parametrize("language", ["cuda", "hip"])
def test_a_dependent_member_type_keeps_the_function_parseable(tmp_path, language):
    # ``typename Box<T>::Type`` has no CrossGL generic spelling; a ``generic<T>`` prefix in front of a qualified
    # ``Box<T>::Type`` return type would make the CrossGL unparseable, so the function is written as before.
    source = (
        "template <typename T>\nstruct Box { using Type = T; };\n"
        "template <typename T>\n"
        "__device__ typename Box<T>::Type same(const T v) { return v; }\n"
    )
    if language == "hip":
        source = "#include <hip/hip_runtime.h>\n" + source
    crossgl = _import(tmp_path, language, source)
    CrossGLParser(CrossGLLexer(crossgl).tokens).parse()
    assert "generic<T> Box<T>::Type" not in crossgl


@pytest.mark.parametrize("language", ["cuda", "hip"])
def test_kernel_templates_are_not_generic(tmp_path, language):
    source = (
        "template <typename T>\n"
        "__global__ void scale(T* data, T factor) { data[threadIdx.x] *= factor; }\n"
    )
    if language == "hip":
        source = "#include <hip/hip_runtime.h>\n" + source
    crossgl = _import(tmp_path, language, source)
    assert "generic<" not in crossgl


def _expected(x):
    return [x, 2 * x, 14, 3 * x, x - 2, x + 1, 3, 10]


@pytest.mark.parametrize("language", ["cuda", "hip"])
def test_metal_translation_computes_what_the_templates_compute(tmp_path, language):
    if sys.platform != "darwin" and os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip("the Metal dispatch needs macOS")
    from tests.metal_compute_dispatch import dispatch_word_buffer

    crossgl = _import(tmp_path, language)
    path = tmp_path / "probe.cgl"
    path.write_text(crossgl)
    metal = crosstl.translate(str(path), backend="metal", format_output=False)
    words = dispatch_word_buffer(metal, tmp_path, [21] + [0] * 7, REQUIRE_ENV)
    assert words == _expected(21)

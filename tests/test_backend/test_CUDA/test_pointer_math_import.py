"""CUDA and HIP math functions that write results through pointers import with CrossGL ``out`` arguments.

The importers used to pass ``sincosf(a0, &a1, &a2)`` through verbatim, and the CUDA and HIP generators refused the
round trip because ``&a1`` is not an assignable value. Each pointer argument now becomes the value it addresses;
``frexpf`` and ``modff`` take CrossGL's portable names, and a ``sincosf`` statement becomes two assignments that keep
the libm error bound.
"""

from __future__ import annotations

import re
import shutil
import subprocess

import pytest

import crosstl
from crosstl.backend.CUDA.CudaAst import UnaryOpNode
from crosstl.backend.CUDA.CudaLexer import CudaLexer
from crosstl.backend.CUDA.CudaParser import CudaParser
from crosstl.backend.CUDA.pointer_math_import import is_pure_expression

KERNEL = """
extern "C" __global__ void probe(float* fin, float* outp) {{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    float x = fin[i];
    float y = fin[i + 1u];
    float s;
    float c;
    int e;
    float w;
    {body}
}}
"""


def _source(body, language):
    source = KERNEL.format(body=body)
    if language == "hip":
        return "#include <hip/hip_runtime.h>\n" + source.replace('extern "C" ', "")
    return source


def _translate(tmp_path, body, language="cuda", backend="cgl"):
    path = tmp_path / ("kernel.cu" if language == "cuda" else "kernel.hip")
    path.write_text(_source(body, language), encoding="utf-8")
    return crosstl.translate(str(path), backend=backend, format_output=False)


def _compact(text):
    return re.sub(r"\s+", " ", text)


@pytest.mark.parametrize("language", ["cuda", "hip"])
@pytest.mark.parametrize(
    "body, expected",
    [
        ("outp[i] = frexpf(x, &e) + (float)e;", "frexp(x, e)"),
        ("outp[i] = modff(x, &w) + w;", "modf(x, w)"),
        ("outp[i] = remquof(x, y, &e) + (float)e;", "remquof(x, y, e)"),
        ("sincospif(x, &s, &c); outp[i] = s + c;", "sincospif(x, s, c)"),
        ("sincosf(x, &outp[i], &outp[i + 1u]);", None),
    ],
)
def test_pointer_arguments_become_the_values_they_address(
    tmp_path, language, body, expected
):
    crossgl = _compact(_translate(tmp_path, body, language))

    assert "(&" not in crossgl
    if expected is not None:
        assert expected in crossgl


@pytest.mark.parametrize("language", ["cuda", "hip"])
def test_a_sincosf_statement_keeps_the_libm_bound(tmp_path, language):
    crossgl = _compact(
        _translate(tmp_path, "sincosf(x * 2.0f, &s, &c); outp[i] = s + c;", language)
    )

    assert "s = __cgl_libm_sin((x * 2.0f));" in crossgl
    assert "c = __cgl_libm_cos((x * 2.0f));" in crossgl
    assert "sincosf" not in crossgl


@pytest.mark.parametrize("language", ["cuda", "hip"])
def test_buffer_elements_and_pointer_casts_are_written_in_place(tmp_path, language):
    crossgl = _compact(
        _translate(tmp_path, "sincosf(x, &outp[i], (float*)&outp[i + 1u]);", language)
    )

    assert "outp[i] = __cgl_libm_sin(x);" in crossgl
    assert "outp[(i + 1u)] = __cgl_libm_cos(x);" in crossgl


@pytest.mark.parametrize("language", ["cuda", "hip"])
def test_fast_sincos_takes_the_precision_of_the_target(tmp_path, language):
    crossgl = _compact(
        _translate(tmp_path, "__sincosf(x, &s, &c); outp[i] = s + c;", language)
    )

    assert "s = sin(x);" in crossgl
    assert "c = cos(x);" in crossgl


@pytest.mark.parametrize("language", ["cuda", "hip"])
def test_an_operand_with_side_effects_is_evaluated_once(tmp_path, language):
    crossgl = _compact(
        _translate(tmp_path, "sincosf(x++, &s, &c); outp[i] = s + c + x;", language)
    )

    assert crossgl.count("x++") == 1
    assert "sincosf((x++), s, c)" in crossgl


def test_a_pointer_variable_is_dereferenced(tmp_path):
    source = """
__device__ void polar(float a, float* s, float* c) { sincosf(a, s, c); }
__device__ float exponent(float a, int* e) { return frexpf(a, e); }
extern "C" __global__ void probe(float* outp) {
    float s;
    float c;
    int e;
    polar(outp[0], &s, &c);
    outp[0] = s + c + exponent(outp[1], &e);
}
"""
    path = tmp_path / "kernel.cu"
    path.write_text(source, encoding="utf-8")
    crossgl = _compact(crosstl.translate(str(path), backend="cgl"))

    assert "(*s) = __cgl_libm_sin(a);" in crossgl
    assert "(*c) = __cgl_libm_cos(a);" in crossgl
    assert "frexp(a, (*e))" in crossgl


def test_a_user_function_with_a_math_name_is_called_as_written(tmp_path):
    source = """
__device__ float modff(float a, float* b) { *b = 1.0f; return a; }
extern "C" __global__ void probe(float* outp) {
    float w;
    outp[0] = modff(outp[0], &w) + w;
}
"""
    path = tmp_path / "kernel.cu"
    path.write_text(source, encoding="utf-8")
    crossgl = _compact(crosstl.translate(str(path), backend="cgl"))

    assert "modf(" not in crossgl.replace("modff(", "")


def _expression(text):
    source = f"__device__ void f(float x, float* p, int i) {{ float t = {text}; }}"
    program = CudaParser(CudaLexer(source).tokenize()).parse()
    return program.functions[0].body[0].value


@pytest.mark.parametrize(
    "text, pure",
    [
        ("x", True),
        ("-x * 2.0f + p[i]", True),
        ("(float)i", True),
        ("i > 0 ? x : -x", True),
        ("x++", False),
        ("sinf(x)", False),
        ("p[i++]", False),
        ("x = 1.0f", False),
    ],
)
def test_pure_expressions(text, pure):
    assert is_pure_expression(_expression(text)) is pure


def test_every_pointer_math_function_names_its_crossgl_function():
    from crosstl.backend.CUDA.pointer_math_import import (
        CROSSGL_NAMES,
        POINTER_OUTPUT_FUNCTIONS,
    )

    assert sorted(CROSSGL_NAMES) == sorted(POINTER_OUTPUT_FUNCTIONS)


def test_the_dereference_node_is_built_with_the_given_type():
    from crosstl.backend.CUDA.pointer_math_import import pointed_value

    node = pointed_value("p", UnaryOpNode)
    assert isinstance(node, UnaryOpNode)
    assert node.op == "*"
    assert node.operand == "p"


ROUND_TRIP_BODY = (
    "sincosf(x, &s, &c); outp[i] = s + c;"
    " outp[i] += frexpf(x, &e) + (float)e;"
    " outp[i] += modff(x, &w) + w;"
    " outp[i] += remquof(x, y, &e) + (float)e;"
    " sincospif(x, &s, &c); outp[i] += s + c;"
)


@pytest.mark.parametrize(
    "language, tool, arguments",
    [
        ("cuda", "nvcc", ["-std=c++17"]),
        ("hip", "hipcc", ["--offload-arch=gfx1100", "-std=c++17"]),
    ],
)
def test_the_round_trip_compiles(tmp_path, language, tool, arguments):
    compiler = shutil.which(tool)
    if compiler is None:
        pytest.skip(f"{tool} is not installed")
    suffix = ".cu" if language == "cuda" else ".hip"
    source = tmp_path / f"round_trip{suffix}"
    source.write_text(
        _translate(tmp_path, ROUND_TRIP_BODY, language, language), encoding="utf-8"
    )
    result = subprocess.run(
        [compiler, *arguments, "-c", str(source), "-o", str(tmp_path / "round_trip.o")],
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    assert result.returncode == 0, result.stderr[-2000:]

"""CUDA and HIP output keeps the ``float`` type of CrossGL floating literals."""

import math
import shutil
import subprocess

import pytest

from crosstl.translator.codegen.c_float_literals import c_float_literal_text
from crosstl.translator.codegen.cuda_codegen import CudaCodeGen
from crosstl.translator.codegen.hip_codegen import HipCodeGen
from crosstl.translator.lexer import Lexer
from crosstl.translator.parser import Parser

GENERATORS = {"cuda": CudaCodeGen, "hip": HipCodeGen}

LITERAL_SOURCE = """
shader FloatLiterals {
    struct Pair {
        float a;
        float b;
    };

    float remainder(float x) {
        return x * fmod(x, 2.0) + fmod(x, 3.0f) * 0.5;
    }

    float paired(float x) {
        Pair p = {x, 2.0};
        Pair q = {p.a * 3.0, 1e-3};
        return p.a + p.b + q.a + q.b;
    }

    double widened(double d) {
        return d * 0.25;
    }

    compute {
        layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

        void main() {
            float x = remainder(1.5) + paired(4.0);
            double d = widened(double(x));
        }
    }
}
"""


def generate(backend, source=LITERAL_SOURCE):
    ast = Parser(Lexer(source).tokens).parse()
    return GENERATORS[backend]().generate(ast)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (2.0, "2.0f"),
        (0.5, "0.5f"),
        (43758.5453, "43758.5453f"),
        (1e-07, "1e-07f"),
        (1e20, "1e+20f"),
        (-0.0, "-0.0f"),
    ],
)
def test_finite_values_are_written_with_the_float_suffix(value, expected):
    assert c_float_literal_text(value) == expected


@pytest.mark.parametrize("value", [math.inf, -math.inf, math.nan, 2, True, "2.0"])
def test_values_without_a_float_spelling_are_left_to_the_caller(value):
    assert c_float_literal_text(value) is None


@pytest.mark.parametrize("backend", sorted(GENERATORS))
def test_float_literals_keep_the_suffix(backend):
    code = generate(backend)

    assert "fmod(x, 2.0f)" in code
    assert "fmod(x, 3.0f)" in code
    assert "* 0.5f)" in code
    assert "fmod(x, 2.0)" not in code
    assert "* 0.5)" not in code


@pytest.mark.parametrize("backend", sorted(GENERATORS))
def test_brace_initializers_do_not_narrow_double_literals(backend):
    code = generate(backend)

    assert "Pair p = {x, 2.0f};" in code
    assert "Pair q = {(p.a * 3.0f), 0.001f};" in code


@pytest.mark.parametrize("backend", sorted(GENERATORS))
def test_float_literal_in_double_arithmetic_keeps_crossgl_type(backend):
    # A CrossGL floating literal is a float, as in GLSL, so it is widened in
    # double arithmetic rather than read as a double.
    code = generate(backend)

    assert "(d * 0.25f)" in code


@pytest.mark.parametrize("backend", sorted(GENERATORS))
def test_integer_literals_are_unchanged(backend):
    source = """
    shader IntegerLiterals {
        compute {
            layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

            void main() {
                int i = 3;
                uint u = 4u;
                float f = float(i) + float(u);
            }
        }
    }
    """
    code = generate(backend, source)

    assert "int i = 3;" in code
    assert "u = 4u;" in code
    assert "3f" not in code
    assert "4uf" not in code


def _compile(compiler, extra, code, tmp_path, suffix):
    source = tmp_path / f"literals{suffix}"
    source.write_text(code, encoding="utf-8")
    result = subprocess.run(
        [compiler, *extra, "-c", str(source), "-o", str(tmp_path / "literals.o")],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_cuda_output_compiles_with_nvcc(tmp_path):
    nvcc = shutil.which("nvcc")
    if nvcc is None:
        pytest.skip("nvcc is not installed")
    _compile(nvcc, ["-std=c++17"], generate("cuda"), tmp_path, ".cu")


def test_hip_output_compiles_with_hipcc(tmp_path):
    hipcc = shutil.which("hipcc")
    if hipcc is None:
        pytest.skip("hipcc is not installed")
    _compile(
        hipcc,
        ["-std=c++17", "-Werror=c++11-narrowing"],
        generate("hip"),
        tmp_path,
        ".hip",
    )

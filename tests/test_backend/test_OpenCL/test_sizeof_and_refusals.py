"""OpenCL C import: ``sizeof`` folds to the size of the declaration in scope, and unknown constructs are refused."""

import pytest

from crosstl.backend.OpenCL.OpenCLCrossGLCodeGen import (
    OpenCLAtomicFunctionUnsupportedError,
    OpenCLBlockUnsupportedError,
    OpenCLSizeofUnsupportedError,
    OpenCLToCrossGLConverter,
)
from crosstl.backend.OpenCL.OpenCLLexer import OpenCLLexer
from crosstl.backend.OpenCL.OpenCLParser import OpenCLParser
from crosstl.translator.lexer import Lexer as CrossGLLexer
from crosstl.translator.parser import Parser as CrossGLParser


def to_crossgl(source):
    ast = OpenCLParser(OpenCLLexer(source).tokenize()).parse()
    crossgl = OpenCLToCrossGLConverter().generate(ast)
    CrossGLParser(CrossGLLexer(crossgl).get_tokens()).parse()
    return crossgl


def stored(crossgl, index):
    """The value the program stores to ``o[index]``."""
    prefix = f"o[{index}] = "
    values = [
        line.strip()[len(prefix) : -1]
        for line in crossgl.splitlines()
        if line.strip().startswith(prefix)
    ]
    assert len(values) == 1, crossgl
    return values[0]


@pytest.mark.parametrize(
    "parameters, query, size",
    [
        ("__global uint* c", "sizeof(*c)", "4"),
        ("__global double* c", "sizeof(c[0])", "8"),
        ("__global uint* c", "sizeof(c)", "8"),
        ("__global atomic_uint* c", "sizeof(*c)", "4"),
        ("__global char4* c", "sizeof(*c)", "4"),
    ],
)
def test_sizeof_a_kernel_parameter_is_its_declared_size(parameters, query, size):
    # The importer had no sizes for parameters and wrote sizeof(*c) as a commented 0.
    crossgl = to_crossgl(
        f"__kernel void k({parameters}, __global int* o) {{ o[0] = {query}; }}"
    )
    assert stored(crossgl, 0) == size


def test_sizeof_an_array_parameter_is_a_pointer():
    # C17 6.7.6.3p7: a parameter declared as an array of T is a pointer to T.
    crossgl = to_crossgl(
        "void f(float a[4], __global int* o) { o[0] = sizeof(a); o[1] = sizeof(a[0]); }"
        "__kernel void k(__global int* o) { float b[4]; f(b, o); }"
    )
    assert (stored(crossgl, 0), stored(crossgl, 1)) == ("8", "4")


def test_sizeof_parameters_of_one_kernel_do_not_reach_the_next():
    crossgl = to_crossgl(
        "__kernel void k(__global double* c, __global int* o) { o[0] = sizeof(*c); }"
        "__kernel void k2(__global char* c, __global int* p) { p[0] = sizeof(*c); }"
    )
    assert stored(crossgl, 0) == "8"
    assert "p[0] = 1;" in crossgl


@pytest.mark.parametrize(
    "statement",
    [
        "{ char c = 1; o[1] = sizeof(c); }",
        "if (o[0] > 0) { char c = 1; o[1] = sizeof(c); }",
        "for (int i = 0; i < 2; i++) { char c = 1; o[1] = sizeof(c); }",
        "while (o[0] > 0) { char c = 1; o[1] = sizeof(c); }",
        "do { char c = 1; o[1] = sizeof(c); } while (o[0] > 0);",
        "for (char c = 0; c < 2; c++) { o[1] = sizeof(c); }",
        "switch (o[0]) { case 0: { char c = 1; o[1] = sizeof(c); break; } default: break; }",
    ],
)
def test_sizeof_sees_the_declaration_of_the_enclosing_block(statement):
    # C17 6.2.1p4, 6.8.4p3, 6.8.5p5: an inner declaration is in scope to the end of its block only. The inner
    # char c was sized for the rest of the function, so the outer double c read as 1 byte.
    crossgl = to_crossgl(
        "__kernel void k(__global int* o) { double c = 1.0; "
        f"{statement} o[2] = sizeof(c); }}"
    )
    assert (stored(crossgl, 1), stored(crossgl, 2)) == ("1", "8")


def test_sizeof_in_an_else_branch_does_not_see_the_then_branch():
    crossgl = to_crossgl(
        "__kernel void k(__global int* o) { double c = 1.0; "
        "if (o[0] > 0) { char c = 1; o[3] = 0; } else { o[1] = sizeof(c); } }"
    )
    assert stored(crossgl, 1) == "8"


def test_sizeof_an_unknown_type_is_refused():
    source = "__kernel void k(__global int* o) { o[0] = sizeof(struct missing); }"
    ast = OpenCLParser(OpenCLLexer(source).tokenize()).parse()
    with pytest.raises(OpenCLSizeofUnsupportedError, match=r"cannot determine sizeof"):
        OpenCLToCrossGLConverter().generate(ast)


@pytest.mark.parametrize("function", ["atomicAdd", "atomicCAS_system", "atomicExch"])
def test_cuda_atomic_names_are_refused(function):
    # OpenCL C 3.0, "Atomic Functions": these are CUDA and HIP functions, which the importer wrote unchanged.
    source = f"__kernel void k(__global int* c) {{ {function}(c, 1); }}"
    ast = OpenCLParser(OpenCLLexer(source).tokenize()).parse()
    with pytest.raises(
        OpenCLAtomicFunctionUnsupportedError,
        match=f"{function} is not an OpenCL C atomic function",
    ):
        OpenCLToCrossGLConverter().generate(ast)


def test_opencl_atomic_functions_are_mapped():
    crossgl = to_crossgl("__kernel void k(__global int* c) { atomic_add(c, 1); }")
    assert "atomicAdd(c, 1);" in crossgl


def test_block_literal_is_refused():
    # A block literal argument was written as `0 /* unsupported OpenCL block literal */`.
    source = """
    void apply(float (^op)(float)) { op(1.0f); }
    kernel void k() { apply(^(float x) { return x + 1.0f; }); }
    """
    ast = OpenCLParser(OpenCLLexer(source).tokenize()).parse()
    with pytest.raises(OpenCLBlockUnsupportedError, match="block literals have no"):
        OpenCLToCrossGLConverter().generate(ast)

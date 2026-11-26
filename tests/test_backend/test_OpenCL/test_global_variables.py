"""OpenCL C variables in the global address space preserve or refuse their sharing semantics."""

import pytest

from crosstl.backend.OpenCL.global_variables import (
    OpenCLGlobalVariableUnsupportedError,
)
from crosstl.backend.OpenCL.OpenCLCrossGLCodeGen import OpenCLToCrossGLConverter
from crosstl.backend.OpenCL.OpenCLLexer import OpenCLLexer
from crosstl.backend.OpenCL.OpenCLParser import OpenCLParser
from crosstl.translator.lexer import Lexer as CrossGLLexer
from crosstl.translator.parser import Parser as CrossGLParser

KERNEL = """
__kernel __attribute__((reqd_work_group_size(4, 1, 1)))
void k(__global int *out) {
    if (get_local_id(0) == 0) {
        WRITE;
    }
    barrier(CLK_GLOBAL_MEM_FENCE);
    out[get_local_id(0)] = READ;
}
"""


def to_crossgl(source):
    ast = OpenCLParser(OpenCLLexer(source).tokenize()).parse()
    return OpenCLToCrossGLConverter().generate(ast)


def kernel(declaration, write="counter = 7", read="counter"):
    return declaration + KERNEL.replace("WRITE", write).replace("READ", read)


@pytest.mark.parametrize(
    "declaration",
    [
        "int counter = 3;",
        "static int counter = 3;",
        "extern global int counter;",
    ],
)
def test_writable_program_scope_global_is_refused(declaration):
    # Every work-item shares one counter, so item 0's write is visible to the
    # others after the barrier; a per-invocation copy would read 3 there.
    with pytest.raises(
        OpenCLGlobalVariableUnsupportedError,
        match="program-scope variable 'counter' is in the global address space",
    ):
        to_crossgl(kernel(declaration))


@pytest.mark.parametrize(
    ("declaration", "expected"),
    [
        ("global int counter = 3;", "var<device> counter: i32 = 3;"),
        ("__global int counter = 3;", "var<device> counter: i32 = 3;"),
        ("__global int counter;", "var<device> counter: i32;"),
    ],
)
def test_explicit_program_scope_device_globals_keep_their_address_space(
    declaration, expected
):
    assert expected in to_crossgl(kernel(declaration))


def test_program_scope_global_array_keeps_its_address_space():
    crossgl = to_crossgl(
        kernel(
            "__global float acc[64] = {1.0f, 2.0f, 4.0f};",
            write="acc[0] = acc[0] * 2.0f",
            read="(int)acc[0]",
        )
    )

    assert "var<device> acc: array<f32, 64>" in crossgl


def test_program_scope_global_pointer_is_refused():
    with pytest.raises(OpenCLGlobalVariableUnsupportedError, match="'cursor'"):
        to_crossgl(
            kernel("global const int *cursor;", write="cursor = out", read="cursor[0]")
        )


def test_function_scope_static_variable_is_refused():
    source = """
    int bump(int v) { static int calls = 0; calls += v; return calls; }
    __kernel void k(__global int *out) { out[0] = bump(1); }
    """
    with pytest.raises(
        OpenCLGlobalVariableUnsupportedError,
        match="function-scope static variable 'calls'",
    ):
        to_crossgl(source)
    assert OpenCLGlobalVariableUnsupportedError.code.endswith(
        "opencl-global-variable-unsupported"
    )


@pytest.mark.parametrize(
    "declaration, read",
    [
        ("__constant int limit = 3;", "limit"),
        ("const __global int limit = 3;", "limit"),
        ("const float table[2] = {1.0f, 2.0f};", "(int)table[1]"),
        ("global int * const nowhere = 0;", "4"),
    ],
)
def test_read_only_program_scope_variables_keep_their_import(declaration, read):
    crossgl = to_crossgl(kernel(declaration, write="out[0] = 1", read=read))
    CrossGLParser(CrossGLLexer(crossgl).get_tokens()).parse()


def test_kernel_locals_and_global_pointers_keep_their_import():
    source = """
    __kernel void k(__global int *out) {
        static const int lim = 4;
        __global int *p = out;
        __local int tile[4];
        int x = lim;
        tile[get_local_id(0) & 3] = x;
        barrier(CLK_LOCAL_MEM_FENCE);
        p[get_global_id(0)] = tile[0];
    }
    """
    crossgl = to_crossgl(source)
    CrossGLParser(CrossGLLexer(crossgl).get_tokens()).parse()
    assert "var<workgroup> tile" in crossgl

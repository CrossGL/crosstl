"""Memory qualifiers of OpenCL C kernel pointer parameters.

The importer wrote ``__global volatile float* p`` and ``__global float* restrict p`` as plain CrossGL storage
buffers, so every target lost the volatile and restrict accesses (SPIR-V ``Decoration.Volatile`` and
``Decoration.Restrict``). It now writes ``@volatile`` for a volatile pointed-to type and ``@restrict`` for a
restrict pointer (OpenCL C 3.0 section 6.7, C11 6.7.3).
"""

import pytest

from crosstl.backend.OpenCL.OpenCLCrossGLCodeGen import OpenCLToCrossGLConverter
from crosstl.backend.OpenCL.OpenCLLexer import OpenCLLexer
from crosstl.backend.OpenCL.OpenCLParser import OpenCLParser
from crosstl.backend.OpenCL.pointer_qualifiers import opencl_pointer_memory_attributes
from crosstl.translator.lexer import Lexer as CrossGLLexer
from crosstl.translator.parser import Parser as CrossGLParser


def _crossgl(parameters):
    source = (
        "__kernel __attribute__((reqd_work_group_size(1, 1, 1))) "
        f"void probe({parameters}) {{\n"
        "    uint i = get_global_id(0);\n    outp[i] = fin[i] * 2.0f;\n}\n"
    )
    ast = OpenCLParser(OpenCLLexer(source).tokenize()).parse()
    crossgl = OpenCLToCrossGLConverter().generate(ast)
    return crossgl, CrossGLParser(CrossGLLexer(crossgl).get_tokens()).parse()


def _parameter_attributes(program, name):
    for function in program.functions:
        for parameter in function.parameters:
            if parameter.name == name:
                return {str(attribute.name) for attribute in parameter.attributes}
    raise AssertionError(f"no parameter {name}")


def test_volatile_pointed_to_type_becomes_a_volatile_buffer():
    crossgl, program = _crossgl("__global float* fin, __global volatile float* outp")

    assert "@group(0) @binding(0) var<storage, read_write> fin: array<f32>" in crossgl
    assert (
        "@volatile @group(0) @binding(1) var<storage, read_write> outp: array<f32>"
        in crossgl
    )
    assert "volatile" in _parameter_attributes(program, "outp")
    assert "volatile" not in _parameter_attributes(program, "fin")


@pytest.mark.parametrize("spelling", ["restrict", "__restrict", "__restrict__"])
def test_restrict_pointer_becomes_a_restrict_buffer(spelling):
    crossgl, program = _crossgl(
        f"__global const float* {spelling} fin, global float* {spelling} outp"
    )

    assert (
        "@restrict @group(0) @binding(0) var<storage, read> fin: array<f32>" in crossgl
    )
    assert (
        "@restrict @group(0) @binding(1) var<storage, read_write> outp: array<f32>"
        in crossgl
    )
    assert "restrict" in _parameter_attributes(program, "fin")
    assert "restrict" in _parameter_attributes(program, "outp")


def test_const_volatile_restrict_buffer_keeps_all_three():
    crossgl, program = _crossgl(
        "__global const volatile float* restrict fin, __global float* outp"
    )

    assert (
        "@volatile @restrict @group(0) @binding(0) var<storage, read> fin: array<f32>"
        in crossgl
    )
    assert {"volatile", "restrict"} <= _parameter_attributes(program, "fin")


def test_qualifiers_of_the_pointer_itself_leave_the_memory_plain():
    # ``float* volatile p`` and ``float* const p`` qualify only the parameter's copy of the address.
    crossgl, program = _crossgl(
        "__global float* volatile fin, __global float* const outp"
    )

    assert "@group(0) @binding(0) var<storage, read_write> fin: array<f32>" in crossgl
    assert "@group(0) @binding(1) var<storage, read_write> outp: array<f32>" in crossgl
    assert not {"volatile", "restrict"} & _parameter_attributes(program, "fin")
    assert not {"volatile", "restrict"} & _parameter_attributes(program, "outp")


@pytest.mark.parametrize(
    "raw_type, attributes",
    [
        ("__global__ float *", ""),
        ("__global__ volatile float *", "@volatile "),
        ("__global__ volatile float * volatile", "@volatile "),
        ("__global__ float * volatile", ""),
        ("__global__ float * __restrict__", "@restrict "),
        ("__global__ float * __restrict__ const", "@restrict "),
        ("__global__ const volatile int * __restrict__", "@volatile @restrict "),
        ("int", ""),
    ],
)
def test_pointer_memory_attributes(raw_type, attributes):
    assert opencl_pointer_memory_attributes(raw_type) == attributes

"""Vulkan GLSL expression statements keep their evaluation, and unknown nodes are refused.

The importer's statement loop used to write ``// Unhandled statement type: <Node>`` for an expression statement
whose root is a conditional, a member access or an index, so ``c ? bump() : reset();`` lost both calls.
"""

import pytest

from crosstl.backend.SPIRV.statement_import import SpirvStatementUnsupportedError
from crosstl.backend.SPIRV.VulkanCrossGLCodeGen import VulkanToCrossGLConverter
from crosstl.backend.SPIRV.VulkanLexer import VulkanLexer
from crosstl.backend.SPIRV.VulkanParser import VulkanParser

SOURCE = """
#version 450
layout(location = 0) out vec4 color;
int counter;
float values[2];
int bump() { counter += 1; return counter; }
int reset() { counter = 0; return counter; }
void main() {
    int a = 1;
    %s
    color = vec4(float(counter));
}
"""


def crossgl(statement):
    ast = VulkanParser(VulkanLexer(SOURCE % statement).tokenize()).parse()
    return VulkanToCrossGLConverter().generate(ast)


def main_body(code):
    body = code[code.index("int a = 1;") :]
    return [line.strip() for line in body.splitlines()[1:] if line.strip()][0]


def test_conditional_statement_keeps_both_calls():
    code = crossgl("(a > 0) ? bump() : reset();")
    assert "Unhandled" not in code
    assert main_body(code) == "((a > 0) ? bump() : reset());"


@pytest.mark.parametrize(
    "statement, expected",
    [
        ("values[0];", "values[0];"),
        ("values[bump()];", "values[bump()];"),
        ("color.x;", "color.x;"),
    ],
)
def test_access_statements_are_emitted(statement, expected):
    code = crossgl(statement)
    assert "Unhandled" not in code
    assert main_body(code) == expected


def test_unknown_statement_is_refused():
    with pytest.raises(SpirvStatementUnsupportedError, match="statement object"):
        VulkanToCrossGLConverter().generate_function_body([object()])


def test_unknown_expression_is_refused():
    with pytest.raises(SpirvStatementUnsupportedError, match="expression object"):
        VulkanToCrossGLConverter().generate_expression(object())

"""HLSL do/while loops import as CrossGL do/while loops.

A ``continue`` in a do/while body jumps to the loop condition (C11 6.8.6.2, which HLSL follows). Rewriting the loop
as ``while (true) { body if (!c) break; }`` makes the same ``continue`` skip the test, so the loop no longer ends.
"""

import re
import textwrap

import pytest

from crosstl.backend.DirectX import DirectxCrossGLCodeGen
from crosstl.backend.DirectX.DirectxLexer import HLSLLexer
from crosstl.backend.DirectX.DirectxParser import HLSLParser
from crosstl.translator.codegen.directx_codegen import HLSLCodeGen
from crosstl.translator.codegen.GLSL_codegen import GLSLCodeGen
from crosstl.translator.codegen.metal_codegen import MetalCodeGen
from crosstl.translator.lexer import Lexer as CrossGLLexer
from crosstl.translator.parser import Parser as CrossGLParser

DO_WHILE_CONTINUE_HLSL = textwrap.dedent("""
    RWStructuredBuffer<int> outp;

    [numthreads(1, 1, 1)]
    void main(uint3 id : SV_DispatchThreadID) {
        int k = 0;
        int n = 0;
        do {
            k++;
            if (k > 2) {
                continue;
            }
            n++;
        } while (k < 4);
        outp[id.x] = n * 100 + k;
    }
""")

NESTED_DO_WHILE_HLSL = textwrap.dedent("""
    RWStructuredBuffer<uint> outp;

    uint count(uint limit) {
        uint i = 0;
        uint total = 0;
        do {
            uint j = 0;
            do {
                j++;
                if ((j & 1) == 0) {
                    continue;
                }
                total += j;
            } while (j < limit);
            i++;
        } while (i < 3 && total < 100);
        return total;
    }

    [numthreads(1, 1, 1)]
    void main(uint3 id : SV_DispatchThreadID) {
        outp[id.x] = count(id.x + 4);
    }
""")


def _crossgl(code):
    ast = HLSLParser(HLSLLexer(code).tokenize()).parse()
    return DirectxCrossGLCodeGen.HLSLToCrossGLConverter().generate(ast)


def _parse_crossgl(code):
    return CrossGLParser(CrossGLLexer(code).get_tokens()).parse()


def _compact(text):
    return re.sub(r"\s+", " ", text)


def test_do_while_with_continue_keeps_the_loop_condition():
    output = _crossgl(DO_WHILE_CONTINUE_HLSL)
    compact = _compact(output)

    assert "while (true)" not in output
    assert re.search(
        r"do \{ k\+\+; if \(k > 2\) \{ continue; \} n\+\+; \} while \(\(?k < 4\)?\);",
        compact,
    ), output
    _parse_crossgl(output)


def test_nested_do_while_loops_keep_both_conditions():
    output = _crossgl(NESTED_DO_WHILE_HLSL)
    compact = _compact(output)

    assert "while (true)" not in output
    assert compact.count("do {") == 2, output
    assert re.search(r"\} while \(\(?j < limit\)?\);", compact), output
    assert re.search(
        r"\} while \(\(?\(?i < 3\)? && \(?total < 100\)?\)?\);", compact
    ), output
    _parse_crossgl(output)


@pytest.mark.parametrize(
    "generator, loop_open",
    [
        (GLSLCodeGen, r"do \{"),
        (MetalCodeGen, r"do \{"),
        (HLSLCodeGen, r"do \{"),
    ],
)
def test_imported_do_while_reaches_target_do_while(generator, loop_open):
    target = generator().generate(_parse_crossgl(_crossgl(DO_WHILE_CONTINUE_HLSL)))
    compact = _compact(target)

    assert re.search(loop_open, compact), target
    assert re.search(r"\} while \(+k < 4\)+;", compact), target
    assert "while (true)" not in target

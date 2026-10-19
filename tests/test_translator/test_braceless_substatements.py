"""Braceless bodies of selection and iteration statements keep their statement on every target.

Before, the CrossGL parser kept a braceless body as the bare statement, and consumers that expected a block lost
it: WGSL emitted ``if (c) return;`` as ``if (c) {}`` (and dropped braceless loop bodies and ``else if`` chains),
and GLSL crashed on a braceless ``for`` body. The parser now keeps such a body as a one-statement block, which is
how C++ ([stmt.select.general]/2, [stmt.iter.general]/2) and GLSL scope it.
"""

import re
import textwrap

import pytest

import crosstl
from crosstl.translator.ast import (
    BlockNode,
    DoWhileNode,
    ForNode,
    IfNode,
    ReturnNode,
    WhileNode,
)
from crosstl.translator.lexer import Lexer
from crosstl.translator.parser import Parser

# Each sentinel is stored by exactly one braceless body.
SENTINELS = (11, 12, 13, 14, 15, 16, 17, 18)

COMPUTE = textwrap.dedent("""
    shader BracelessBodies {
        compute {
            layout(std430, binding = 0) buffer int values[];
            [numthreads(64, 1, 1)]
            void main() {
                uint i = gl_GlobalInvocationID.x;
                if (i > 63u) return;
                if (i > 40u) values[i] = 11;
                else if (i > 20u) values[i] = 12;
                else values[i] = 13;
                for (int k = 0; k < 2; k++) values[i] += 14;
                while (values[i] > 1000) values[i] -= 15;
                do values[i] += 16; while (values[i] < 0);
                if (i == 0u) values[i] = 17; else values[i] = 18;
            }
        }
    }
    """)

FRAGMENT = textwrap.dedent("""
    shader BracelessFragment {
        struct FragmentOutput {
            vec4 color;
        }
        fragment {
            uniform int count;
            FragmentOutput main() {
                FragmentOutput output;
                int value = count;
                if (value > 40) value = 11;
                else if (value > 20) value = 12;
                else value = 13;
                for (int k = 0; k < 2; k++) value += 14;
                while (value > 1000) value -= 15;
                do value += 16; while (value < 0);
                if (value == 0) value = 17; else value = 18;
                output.color = vec4(float(value), 0.0, 0.0, 1.0);
                return output;
            }
        }
    }
    """)


# WGSL has no do-while statement (it refuses one), so its program leaves that loop out.
WGSL_COMPUTE = COMPUTE.replace(
    "            do values[i] += 16; while (values[i] < 0);\n", ""
)
PROGRAMS = {"compute": COMPUTE, "fragment": FRAGMENT, "wgsl-compute": WGSL_COMPUTE}


def parse(text):
    return Parser(Lexer(text).tokens).parse()


def entry_statements(ast):
    (stage,) = ast.stages.values()
    return stage.entry_point.body.statements


def test_braceless_bodies_parse_as_one_statement_blocks():
    statements = entry_statements(parse(COMPUTE))
    early_return, chain, loop, while_loop, do_loop, both = statements[1:]

    assert isinstance(early_return.then_branch, BlockNode)
    assert [type(s) for s in early_return.then_branch.statements] == [ReturnNode]

    # ``else if`` stays a chain of if statements; only its innermost bodies are wrapped.
    assert isinstance(chain.else_branch, IfNode)
    assert isinstance(chain.else_branch.then_branch, BlockNode)
    assert isinstance(chain.else_branch.else_branch, BlockNode)

    for node in (loop, while_loop, do_loop):
        assert isinstance(node, (ForNode, WhileNode, DoWhileNode))
        assert isinstance(node.body, BlockNode)
        assert len(node.body.statements) == 1
    assert isinstance(both.then_branch, BlockNode)
    assert isinstance(both.else_branch, BlockNode)


def test_braced_bodies_are_unchanged():
    braced = COMPUTE.replace("if (i > 63u) return;", "if (i > 63u) { return; }")
    early_return = entry_statements(parse(braced))[1]
    assert isinstance(early_return.then_branch, BlockNode)
    assert [type(s) for s in early_return.then_branch.statements] == [ReturnNode]


def sentinel_counts(backend, output):
    if backend == "vulkan":
        return {
            value: len(re.findall(rf"OpConstant %int {value}\b", output))
            for value in SENTINELS
        }
    return {
        value: len(re.findall(rf"(?<![\w.]){value}(?![\w.])", output))
        for value in SENTINELS
    }


@pytest.mark.parametrize(
    "backend, program",
    [
        ("opengl", "compute"),
        ("directx", "compute"),
        ("metal", "compute"),
        ("vulkan", "compute"),
        ("slang", "compute"),
        ("wgsl", "wgsl-compute"),
        ("webgl", "fragment"),
        ("opengl", "fragment"),
        ("directx", "fragment"),
        ("metal", "fragment"),
    ],
)
def test_every_braceless_body_reaches_the_output(tmp_path, backend, program):
    text = PROGRAMS[program]
    source = tmp_path / "braceless.cgl"
    source.write_text(text)
    output = crosstl.translate(str(source), backend=backend)
    expected = {value: 1 if f" {value};" in text else 0 for value in SENTINELS}
    assert sentinel_counts(backend, output) == expected
    assert not re.search(r"\)\s*\{\s*\}", output)
    assert not re.search(r"\belse\s*\{\s*\}", output)


def test_wgsl_keeps_an_early_return(tmp_path):
    source = tmp_path / "braceless.cgl"
    source.write_text(WGSL_COMPUTE)
    output = crosstl.translate(str(source), backend="wgsl")
    assert re.search(r"if \(\(i > 63u\)\)\s*\{\s*return;\s*\}", output)

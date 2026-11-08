"""WGSL closes a value-returning body whose end its behavior analysis cannot rule out (WGSL "Behavior Analysis").

``do { return x; } while (c);``, ``while (true) { if (d) { return x; } }`` and ``for (...) { return x; }`` end a
value-returning function in C-family sources. WGSL gives each of these loops the behavior ``Next``, so naga and Tint
rejected the emitted function ("Returning None where Some is expected"). The emitter now appends a ``return`` of the
zero value, which no defined execution of the source reaches (C++17 [stmt.return]/2).
"""

import shutil
import subprocess
import textwrap

import pytest

from crosstl.translator.ast import (
    BlockNode,
    BreakNode,
    ContinueNode,
    DoWhileNode,
    ExpressionStatementNode,
    ForNode,
    IdentifierNode,
    IfNode,
    LiteralNode,
    LoopNode,
    PrimitiveType,
    ReturnNode,
    WhileNode,
)
from crosstl.translator.codegen.wgsl_codegen import WGSLCodeGen
from crosstl.translator.codegen.wgsl_function_behaviors import (
    BREAK,
    CONTINUE,
    NEXT,
    RETURN,
    behaviors,
    end_is_reachable,
)
from crosstl.translator.lexer import Lexer
from crosstl.translator.parser import Parser

TRUE = LiteralNode(True, PrimitiveType("bool"))
FLAG = IdentifierNode("flag")


def ret():
    return ReturnNode(LiteralNode(1, PrimitiveType("int")))


def test_sequence_stops_at_a_statement_without_next():
    assert not end_is_reachable([ret(), ExpressionStatementNode(FLAG)])
    assert end_is_reachable([ExpressionStatementNode(FLAG)])
    assert end_is_reachable([])


def test_if_without_else_can_fall_through():
    assert behaviors(IfNode(FLAG, BlockNode([ret()]))) == {NEXT, RETURN}
    assert behaviors(IfNode(FLAG, BlockNode([ret()]), BlockNode([ret()]))) == {RETURN}


def test_loops_with_a_condition_or_a_break_have_next():
    body = BlockNode([ret()])
    assert behaviors(WhileNode(TRUE, body)) == {NEXT, RETURN}
    assert behaviors(ForNode(None, FLAG, None, body)) == {NEXT, RETURN}
    assert behaviors(DoWhileNode(body, FLAG)) == {NEXT, RETURN}
    assert behaviors(ForNode(None, None, None, body)) == {RETURN}
    assert behaviors(LoopNode(body)) == {RETURN}
    assert behaviors(LoopNode(BlockNode([IfNode(FLAG, BreakNode()), ret()]))) == {
        NEXT,
        RETURN,
    }


def test_loop_consumes_continue_and_break():
    body = BlockNode([IfNode(FLAG, ContinueNode()), BreakNode()])
    assert behaviors(body) == {CONTINUE, BREAK}
    assert behaviors(LoopNode(body)) == {NEXT}


def compute(helpers, call):
    source = textwrap.dedent("""
        shader Probe {
            compute {
                buffer int data[];
        %s
                [numthreads(1, 1, 1)]
                void main() {
                    data[0] = %s;
                }
            }
        }
        """) % (textwrap.indent(textwrap.dedent(helpers), " " * 8), call)
    return WGSLCodeGen().generate(Parser(Lexer(source).get_tokens()).parse())


HELPERS = """
int firstOdd(int n) {
    int i = 0;
    do {
        if ((i & 1) == 1) {
            return i;
        }
        i++;
    } while (i < n);
    return -1;
}
int spin(int n) {
    int i = n;
    while (true) {
        if (i > 8) {
            return i;
        }
        i += 3;
    }
}
int scan(int n) {
    for (int i = 0; i < n; i++) {
        return i;
    }
}
int pick(int n) {
    if (n > 0) {
        return 1;
    } else {
        return 2;
    }
}
"""


def function_text(code, name):
    start = code.index(f"fn {name}(")
    return code[start : code.index("\n}\n", start) + 3]


def test_loop_ending_a_value_returning_function_gets_a_closing_return():
    code = compute(HELPERS, "firstOdd(3) + spin(1) + scan(2) + pick(1)")
    assert function_text(code, "spin").rstrip().endswith("    return i32(0);\n}")
    assert function_text(code, "scan").rstrip().endswith("    return i32(0);\n}")
    for name in ("firstOdd", "pick"):
        assert "i32(0)" not in function_text(code, name)


def test_functions_ending_in_loops_validate(tmp_path):
    if shutil.which("naga") is None:
        pytest.skip("naga is not installed")
    path = tmp_path / "probe.wgsl"
    path.write_text(compute(HELPERS, "firstOdd(3) + spin(1) + scan(2) + pick(1)"))
    result = subprocess.run(["naga", str(path)], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr

"""An HLSL ``(void)e;`` statement keeps the evaluation of ``e`` and drops only the discarded value."""

import re

import pytest

import crosstl
from crosstl.backend.common_ast import CastNode, FunctionCallNode, VariableNode
from crosstl.backend.DirectX.discarded_values import discarded_cast_operand

HLSL = """
RWStructuredBuffer<int> data : register(u0);
static int counter = 0;
void bump() { counter = counter + 1; }
int twice(int v) { counter += v; return v * 2; }
[numthreads(1, 1, 1)]
void CSMain(uint3 id : SV_DispatchThreadID) {
    int x = data[1];
    (void)x;
    (void)(x + 1);
    (void)bump();
    (void)twice(x);
    (void)(void)bump();
    data[0] = counter + x;
}
"""


def _translate(tmp_path, backend):
    path = tmp_path / "discarded.hlsl"
    path.write_text(HLSL)
    return crosstl.translate(str(path), backend=backend)


def _main(output):
    return output.split("CSMain" if "CSMain" in output else "main", 1)[-1]


@pytest.mark.parametrize("backend", ["cgl", "opengl", "metal", "directx"])
def test_side_effects_of_a_discarded_cast_are_kept(tmp_path, backend):
    body = _main(_translate(tmp_path, backend))
    assert len(re.findall(r"\bbump\(", body)) == 2, body
    assert len(re.findall(r"\btwice\(", body)) == 1, body
    assert "Unhandled statement" not in body
    assert "x + 1" not in body
    assert not re.search(r"^\s*x;", body, re.MULTILINE)


def test_a_pure_operand_is_dropped():
    assert discarded_cast_operand(CastNode("void", "x")) is None


def test_an_effectful_operand_becomes_the_statement():
    call = FunctionCallNode("bump", [])
    assert discarded_cast_operand(CastNode("void", CastNode("void", call))) is call


def test_other_statements_are_unchanged():
    declaration = VariableNode("int", "x")
    assert discarded_cast_operand(declaration) is declaration
    cast = CastNode("float", FunctionCallNode("bump", []))
    assert discarded_cast_operand(cast) is cast

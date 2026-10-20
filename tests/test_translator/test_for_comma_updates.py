"""A ``for`` header with several comma-separated updates goes through every hub legalization pass.

The CrossGL parser keeps ``for (...; ...; ++i, ++j)``'s update as a list of expressions.  Rule carrying, binary16,
narrow-integer and CUDA-warp legalization each treated the update as one node, so the GLSL importer's output for
such a loop crashed with ``TypeError: vars() argument must have __dict__ attribute`` on every target.
"""

import pytest

import crosstl
from crosstl.translator import ast as A
from crosstl.translator.lexer import Lexer
from crosstl.translator.parser import Parser
from crosstl.translator.semantics import apply_rule_carrying
from crosstl.translator.semantics.rewrite import field_slots

GLSL_COMMA_UPDATE = """#version 450
layout(local_size_x = 1) in;
layout(std430, binding = 0) buffer Output { uint data[]; };
void main() {
    uint plane = 0u;
    for (uint j = 0u; j < 2u; ++j, ++plane) {
        data[plane] = j;
    }
}
"""

CROSSGL_COMMA_UPDATE = """shader S {
  compute {
    void main() {
      int total = 0;
      int k = 0;
      for (int i = 0; i < 4; ++i, k += 2) { total = total + i + k; }
    }
  }
}
"""


def for_loops(node, seen=None):
    seen = set() if seen is None else seen
    if id(node) in seen:
        return
    seen.add(id(node))
    if isinstance(node, A.ForNode):
        yield node
    values = node if isinstance(node, list) else None
    if values is None and isinstance(node, dict):
        values = list(node.values())
    if values is None and hasattr(node, "__dict__"):
        values = list(vars(node).values())
    for value in values or []:
        if isinstance(value, (A.ASTNode, list, dict)):
            yield from for_loops(value, seen)


def test_field_slots_enumerate_a_list_update():
    loop = next(for_loops(Parser(Lexer(CROSSGL_COMMA_UPDATE).get_tokens()).parse()))
    assert isinstance(loop.update, list)
    assert field_slots(loop, "update") == [("update", 0), ("update", 1)]
    assert field_slots(loop, "condition") == [("condition", None)]


@pytest.mark.parametrize(
    "source", ["glsl", "directx", "metal", "cuda", "vulkan", "cgl"]
)
@pytest.mark.parametrize("target", ["directx", "metal", "opengl", "vulkan", "wgsl"])
def test_rule_carrying_visits_every_comma_separated_update(source, target):
    program = Parser(Lexer(CROSSGL_COMMA_UPDATE).get_tokens()).parse()
    apply_rule_carrying(program, source, target)
    loop = next(for_loops(program))
    assert isinstance(loop.update, list) and len(loop.update) == 2


@pytest.mark.parametrize("backend", ["directx", "metal", "opengl", "vulkan"])
def test_glsl_comma_update_loop_translates(tmp_path, backend):
    path = tmp_path / "comma_update.comp"
    path.write_text(GLSL_COMMA_UPDATE, encoding="utf-8")
    code = crosstl.translate(str(path), backend=backend)
    assert "plane" in code

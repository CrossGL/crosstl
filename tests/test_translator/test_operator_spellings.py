"""The GLSL, HLSL and Metal emitters spell every CrossGL operator and refuse anything else.

Before, each emitter returned an operator it had no entry for unchanged, so a token name such as ``MOD`` or
``POST_DECREMENT`` reached the output verbatim and an operator CrossGL does not define was emitted as written.
"""

import textwrap

import pytest

from crosstl.translator.ast import BinaryOpNode, UnaryOpNode
from crosstl.translator.codegen.directx_codegen import HLSLCodeGen
from crosstl.translator.codegen.GLSL_codegen import GLSLCodeGen
from crosstl.translator.codegen.glsl_refusals import (
    OpenGLLoweringRefused,
    WebGLLoweringRefused,
)
from crosstl.translator.codegen.hlsl_refusals import DirectXLoweringRefused
from crosstl.translator.codegen.metal_codegen import MetalCodeGen
from crosstl.translator.codegen.metal_refusals import MetalLoweringRefused
from crosstl.translator.codegen.operator_spellings import (
    C_FAMILY_OPERATORS,
    OPERATOR_TOKEN_SPELLINGS,
    c_family_operator_spelling,
)
from crosstl.translator.codegen.webgl_codegen import WebGLCodeGen
from crosstl.translator.source_registry import SOURCE_REGISTRY, register_default_sources

EMITTERS = {
    "opengl": (GLSLCodeGen, OpenGLLoweringRefused),
    "webgl": (WebGLCodeGen, WebGLLoweringRefused),
    "directx": (HLSLCodeGen, DirectXLoweringRefused),
    "metal": (MetalCodeGen, MetalLoweringRefused),
}

PROGRAM = textwrap.dedent("""
    shader Operators {
        struct FragmentOutput {
            vec4 color;
        }
        fragment {
            uniform int count;
            FragmentOutput main() {
                FragmentOutput output;
                int a = count;
                int b = a % 3;
                int c = a + b;
                c--;
                output.color = vec4(float(c), 0.0, 0.0, 1.0);
                return output;
            }
        }
    }
    """)


def parse_cgl(text):
    register_default_sources()
    return SOURCE_REGISTRY.get("cgl").parse(text)


def nodes(node, kind, seen=None):
    """Every ``kind`` node reachable from ``node``, in a deterministic order."""
    seen = set() if seen is None else seen
    if id(node) in seen:
        return
    seen.add(id(node))
    if isinstance(node, kind):
        yield node
    if isinstance(node, dict):
        # Insertion order is deterministic; stage maps are keyed by enum members.
        children = list(node.values())
    elif isinstance(node, (list, tuple)):
        children = node
    elif hasattr(node, "__dict__"):
        children = [vars(node)[key] for key in sorted(vars(node)) if key != "parent"]
    else:
        return
    for child in children:
        if isinstance(child, (dict, list, tuple)) or hasattr(child, "__dict__"):
            yield from nodes(child, kind, seen)


def set_operator(node, op):
    """Replace a node's operator; nodes carry it as both ``op`` and ``operator``."""
    node.op = op
    node.operator = op


def generate(target, ast):
    emitter_type, _ = EMITTERS[target]
    return emitter_type().generate(ast)


def test_every_token_name_spells_a_c_family_operator():
    assert set(OPERATOR_TOKEN_SPELLINGS.values()) <= C_FAMILY_OPERATORS
    for token, spelling in OPERATOR_TOKEN_SPELLINGS.items():
        assert c_family_operator_spelling(token) == spelling
    for spelling in C_FAMILY_OPERATORS:
        assert c_family_operator_spelling(spelling) == spelling


@pytest.mark.parametrize("target", sorted(EMITTERS))
def test_emitters_agree_on_every_operator(target):
    emitter = EMITTERS[target][0]()
    for operator in sorted(C_FAMILY_OPERATORS | set(OPERATOR_TOKEN_SPELLINGS)):
        assert emitter.map_operator(operator) == c_family_operator_spelling(operator)


@pytest.mark.parametrize("target", sorted(EMITTERS))
@pytest.mark.parametrize("operator", ["<=>", "**", "^^", "and", "MATMUL", 3])
def test_an_undefined_operator_is_refused_outside_a_generation(target, operator):
    emitter_type, refusal_type = EMITTERS[target]
    with pytest.raises(refusal_type) as raised:
        emitter_type().map_operator(operator)
    assert type(raised.value) is refusal_type
    assert raised.value.code == refusal_type.code
    assert f"operator: {operator!r} has no" in str(raised.value)


@pytest.mark.parametrize("target", sorted(EMITTERS))
def test_a_node_without_an_operator_has_none(target):
    emitter = EMITTERS[target][0]()
    assert emitter.map_operator(None) is None
    assert emitter.map_operator("") == ""


@pytest.mark.parametrize("target", sorted(EMITTERS))
def test_an_undefined_binary_operator_refuses_the_program(target):
    ast = parse_cgl(PROGRAM)
    addition = next(node for node in nodes(ast, BinaryOpNode) if node.op == "+")
    set_operator(addition, "<=>")
    refusal_type = EMITTERS[target][1]
    with pytest.raises(refusal_type) as raised:
        generate(target, ast)
    assert type(raised.value) is refusal_type
    assert "operator: '<=>' has no" in str(raised.value)


@pytest.mark.parametrize("target", sorted(EMITTERS))
def test_token_named_operators_are_spelled_in_the_output(target):
    ast = parse_cgl(PROGRAM)
    modulo = next(node for node in nodes(ast, BinaryOpNode) if node.op == "%")
    set_operator(modulo, "MOD")
    decrement = next(node for node in nodes(ast, UnaryOpNode) if node.op == "--")
    set_operator(decrement, "POST_DECREMENT")
    output = generate(target, ast)
    assert "MOD" not in output
    assert "POST_DECREMENT" not in output
    assert "c--" in output
    assert output == generate(target, parse_cgl(PROGRAM))

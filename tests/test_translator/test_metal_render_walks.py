"""The Metal generator's recorded body walks: replayed within a render, fresh everywhere else."""

from collections import Counter
from contextlib import contextmanager

import pytest

from crosstl.translator import ast as cgl
from crosstl.translator.codegen import metal_render_walks
from crosstl.translator.codegen.ast_value_walk import iter_ast_values
from crosstl.translator.codegen.metal_codegen import MetalCodeGen
from crosstl.translator.codegen.metal_render_walks import (
    MetalRenderWalks,
    names_identifier_class,
    names_resource_reference_class,
)
from crosstl.translator.lexer import Lexer
from crosstl.translator.parser import Parser


def _call(name, *arguments):
    return cgl.FunctionCallNode(name, list(arguments))


def _body():
    return [
        cgl.VariableNode(
            "x", cgl.PrimitiveType("float"), _call("f", cgl.IdentifierNode("a"))
        ),
        cgl.ReturnNode(_call("g", _call("h", cgl.IdentifierNode("x")))),
    ]


def _counting_walks(monkeypatch):
    walked = Counter()

    def counting(root):
        walked[id(root)] += 1
        return iter_ast_values(root)

    monkeypatch.setattr(metal_render_walks, "iter_ast_values", counting)
    return walked


def test_walks_outside_a_window_are_fresh_and_see_edits():
    walks = MetalRenderWalks()
    body = _body()
    list(walks.walk(body))
    replacement = cgl.IdentifierNode("b")
    body[0].initial_value = replacement

    assert any(node is replacement for node in walks.walk(body))
    assert not walks.active


def test_a_window_walks_each_value_once_and_replays_the_same_nodes(monkeypatch):
    walked = _counting_walks(monkeypatch)
    walks = MetalRenderWalks()
    body = _body()
    expected = list(iter_ast_values(body))
    with walks.replaying():
        assert walks.active
        first = list(walks.walk(body))
        second = list(walks.walk(body))
        calls = walks.nodes_of_type(body, cgl.FunctionCallNode)
        calls_again = walks.nodes_of_type(body, cgl.FunctionCallNode)
    assert not walks.active

    assert first == second == expected
    assert all(left is right for left, right in zip(first, expected))
    assert walked[id(body)] == 1
    # A call's arguments are reachable as ``arguments`` and ``args``, so the walk produces nested calls twice.
    assert list(calls) == [
        node for node in expected if isinstance(node, cgl.FunctionCallNode)
    ]
    assert [call.name for call in calls][:3] == ["f", "g", "h"]
    assert calls is calls_again


def test_a_list_whose_items_changed_is_walked_again():
    walks = MetalRenderWalks()
    roots = [cgl.IdentifierNode("a")]
    with walks.replaying():
        assert [node.name for node in walks.walk(roots)] == ["a"]
        assert [
            node.name for node in walks.nodes_of_type(roots, cgl.IdentifierNode)
        ] == ["a"]
        roots.append(cgl.IdentifierNode("b"))
        assert [node.name for node in walks.walk(roots)] == ["a", "b"]
        assert [
            node.name for node in walks.nodes_of_type(roots, cgl.IdentifierNode)
        ] == ["a", "b"]
        roots[0] = cgl.IdentifierNode("c")
        assert [node.name for node in walks.walk(roots)] == ["c", "b"]


def test_suspended_and_nested_windows_never_replay_outer_records(monkeypatch):
    walked = _counting_walks(monkeypatch)
    walks = MetalRenderWalks()
    body = _body()
    with walks.replaying():
        list(walks.walk(body))
        with walks.suspended():
            assert not walks.active
            list(walks.walk(body))
            list(walks.walk(body))
        with walks.replaying():
            list(walks.walk(body))
            list(walks.walk(body))
        list(walks.walk(body))
    # Once for the outer window, twice while suspended, once for the nested window.
    assert walked[id(body)] == 4


def test_selections_keep_walk_order_and_subclasses():
    class SpecialCall(cgl.FunctionCallNode):
        pass

    body = [_call("outer", SpecialCall("inner", [])), cgl.IdentifierNode("z")]
    walks = MetalRenderWalks()
    for window in (contextmanager(lambda: (yield))(), walks.replaying()):
        with window:
            names = [
                node.name for node in walks.nodes_of_type(body, cgl.FunctionCallNode)
            ]
            references = walks.nodes_with_class(body, names_resource_reference_class)
            identifiers = walks.nodes_with_class(body, names_identifier_class)
        assert names == [
            node.name
            for node in iter_ast_values(body)
            if isinstance(node, cgl.FunctionCallNode)
        ]
        assert names[:2] == ["outer", "inner"]
        assert [type(node).__name__ for node in references] == [
            type(node).__name__
            for node in iter_ast_values(body)
            if isinstance(node, (cgl.FunctionCallNode, cgl.IdentifierNode))
        ]
        assert "SpecialCall" in {type(node).__name__ for node in references}
        assert [node.name for node in identifiers] == ["z"]


def test_identifier_classes_are_those_the_generator_always_matched():
    assert names_identifier_class(cgl.IdentifierNode)
    assert names_identifier_class(cgl.IdentifierPatternNode)
    assert not names_identifier_class(cgl.VariableNode)
    assert names_resource_reference_class(cgl.RayTracingOpNode)


def test_a_failing_walk_records_nothing_and_fails_again():
    looped = cgl.BlockNode([])
    looped.statements.append(looped)
    walks = MetalRenderWalks()
    with walks.replaying():
        for _ in range(2):
            with pytest.raises(RecursionError):
                list(walks.walk(looped))


def test_body_selections_treat_a_missing_body_as_empty():
    walks = MetalRenderWalks()
    with walks.replaying():
        assert walks.body_nodes_of_type(object(), cgl.VariableNode) == ()
        assert walks.body_nodes_with_class(object(), names_identifier_class) == ()


SOURCE = """
shader ReplayShader {
    struct Particle { vec4 position; vec4 velocity; };
    cbuffer Params { float dt; float damping; };

    float damp(float value) {
        float scaled = value * damping;
        return scaled;
    }

    vec4 advance(vec4 position, vec4 velocity) {
        vec4 step = velocity * damp(dt);
        return position + step;
    }

    compute {
        layout(std430, binding = 0) buffer Particles { Particle items[]; } particles;
        layout(local_size_x = 64) in;
        void main() {
            uint index = gl_GlobalInvocationID.x;
            Particle particle = particles.items[index];
            for (int i = 0; i < 4; i++) {
                particle.position = advance(particle.position, particle.velocity);
            }
            particles.items[index] = particle;
        }
    }
}
"""


class _FreshWalkMetalCodeGen(MetalCodeGen):
    """The generator with replay switched off: every walk is a fresh walk."""

    def render_program(self, ast, target_stage=None):
        with self.metal_render_walks.suspended():
            ast, target_stage = self.rewrite_rendered_program(ast, target_stage)
            return self.render_rewritten_program(ast, target_stage)


def _parse():
    return Parser(Lexer(SOURCE).get_tokens()).parse()


def test_replayed_render_matches_a_fresh_walk_render_and_walks_each_body_once(
    monkeypatch,
):
    fresh = _FreshWalkMetalCodeGen().generate(_parse())

    generator = MetalCodeGen()
    replayed_bodies = Counter()
    bodies = {}

    def counting(root):
        if generator.metal_render_walks.active and isinstance(
            root, (list, cgl.BlockNode)
        ):
            replayed_bodies[id(root)] += 1
            bodies[id(root)] = root
        return iter_ast_values(root)

    monkeypatch.setattr(metal_render_walks, "iter_ast_values", counting)
    replayed = generator.generate(_parse())

    assert replayed == fresh
    assert "advance(" in replayed and "damp(" in replayed
    assert (
        replayed_bodies
    ), "the render walked no function body inside its replay window"
    assert max(replayed_bodies.values()) == 1
    assert not generator.metal_render_walks.active

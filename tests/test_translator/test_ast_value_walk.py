"""The code generators' walk over AST values matches the recursive pre-order walk it replaced."""

import pytest

from crosstl.translator.codegen.ast_value_walk import iter_ast_values
from crosstl.translator.codegen.metal_codegen import MetalCodeGen
from crosstl.translator.lexer import Lexer
from crosstl.translator.parser import Parser


def _recursive_walk(node):
    """The recursive generator the iterative walk must reproduce."""
    if node is None or isinstance(node, (str, int, float, bool)):
        return
    if isinstance(node, (list, tuple, set)):
        for item in node:
            yield from _recursive_walk(item)
        return
    if not hasattr(node, "__dict__"):
        return
    yield node
    for key, value in vars(node).items():
        if key in {"parent", "annotations"}:
            continue
        yield from _recursive_walk(value)


class Node:
    def __init__(self, label, **attributes):
        self.label = label
        for key, value in attributes.items():
            setattr(self, key, value)

    def __repr__(self):
        return f"Node({self.label!r})"


class Slotted:
    __slots__ = ("child",)

    def __init__(self, child):
        self.child = child


def _labels(nodes):
    return [getattr(node, "label", type(node).__name__) for node in nodes]


def _nested_tree():
    leaf = Node("leaf", value=3, name="x", flag=True, ratio=0.5)
    shared = Node("shared")
    root = Node(
        "root",
        parent=Node("ignored-parent"),
        annotations=[Node("ignored-annotation")],
        body=[
            Node("first", operands=(leaf, None, "text", 7)),
            [Node("nested-list-item"), (Node("nested-tuple-item"),)],
            {"mapping": Node("inside-dict")},
            Slotted(Node("inside-slots")),
            shared,
        ],
        tail=shared,
        empty=[],
    )
    return root


def test_walk_matches_recursive_order_on_mixed_values():
    root = _nested_tree()
    assert _labels(iter_ast_values(root)) == _labels(_recursive_walk(root))
    assert _labels(iter_ast_values(root)) == [
        "root",
        "first",
        "leaf",
        "nested-list-item",
        "nested-tuple-item",
        "shared",
        "shared",
    ]


@pytest.mark.parametrize("root", [None, "text", 4, 2.5, True, {"key": Node("x")}])
def test_walk_skips_values_without_attributes(root):
    assert list(iter_ast_values(root)) == []


def test_walk_starts_from_sequences():
    roots = [Node("a", child=Node("b")), (Node("c"),), None]
    assert _labels(iter_ast_values(roots)) == ["a", "b", "c"]


def test_walk_reads_attributes_only_after_the_consumer_resumes():
    replacement = Node("replacement")
    root = Node("root", child=Node("original"))
    seen = []
    for node in iter_ast_values(root):
        seen.append(node.label)
        if node is root:
            root.child = replacement
    assert seen == ["root", "replacement"]


def test_walk_sees_items_appended_to_a_list_it_is_iterating():
    body = [Node("first")]
    root = Node("root", body=body)
    seen = []
    for node in iter_ast_values(root):
        seen.append(node.label)
        if node.label == "first":
            body.append(Node("appended"))
    assert seen == ["root", "first", "appended"]


def test_walk_handles_nesting_deeper_than_the_recursion_limit():
    import sys

    depth = sys.getrecursionlimit() * 2
    root = Node("0")
    current = root
    for index in range(1, depth):
        child = Node(str(index))
        current.child = [child]
        current = child
    labels = _labels(iter_ast_values(root))
    assert len(labels) == depth
    assert labels[:3] == ["0", "1", "2"] and labels[-1] == str(depth - 1)


def test_walk_raises_recursion_error_on_a_node_that_contains_itself():
    root = Node("root")
    middle = Node("middle", child=root)
    root.body = [middle]
    walk = iter_ast_values(root)
    assert _labels([next(walk), next(walk)]) == ["root", "middle"]
    with pytest.raises(RecursionError, match="cycle"):
        next(walk)


def test_walk_raises_recursion_error_on_a_list_that_contains_itself():
    body = [Node("item")]
    body.append(body)
    with pytest.raises(RecursionError, match="list value inside itself"):
        list(iter_ast_values(Node("root", body=body)))


def test_walk_allows_shared_values_that_are_not_their_own_ancestors():
    shared = Node("shared", child=Node("leaf"))
    root = Node("root", left=[shared], right=(shared,), again=shared)
    assert _labels(iter_ast_values(root)) == _labels(_recursive_walk(root))
    assert _labels(iter_ast_values(root)).count("leaf") == 3


SHADER = """
shader WalkOrder {
    struct Pair { float a; float b; }
    float combine(Pair pair, float scale) {
        float total = 0.0;
        for (int i = 0; i < 4; i++) {
            if (i > 1) { total += pair.a * scale; } else { total -= pair.b; }
        }
        return total;
    }
    compute {
        layout(local_size_x = 1) in;
        void main() {
            Pair pair;
            pair.a = 1.0;
            pair.b = 2.0;
            float value = combine(pair, 3.0);
        }
    }
}
"""


def test_metal_codegen_walk_matches_recursive_walk_on_a_parsed_shader():
    ast = Parser(Lexer(SHADER).get_tokens()).parse()
    # Shader and stage nodes refer to enumeration members, whose class refers back to them, so code generators walk
    # declarations and entry points rather than those nodes.
    roots = [
        ast.structs,
        ast.functions,
        [stage.entry_point for stage in ast.stages.values()],
    ]
    generator = MetalCodeGen()
    walked = list(generator.iter_ast_nodes(roots))
    expected = list(_recursive_walk(roots))
    assert len(walked) == len(expected) > 20
    assert all(left is right for left, right in zip(walked, expected))


def test_walk_reports_the_enumeration_cycle_of_a_shader_node_as_recursion_error():
    ast = Parser(Lexer(SHADER).get_tokens()).parse()
    with pytest.raises(RecursionError):
        list(_recursive_walk(ast))
    with pytest.raises(RecursionError, match="cycle"):
        list(iter_ast_values(ast))

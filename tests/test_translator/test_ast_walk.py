"""The iterative AST walk and the binding allocator keep their previous results.

``walk_ast_values`` replaced four copies of a recursive generator, and the GLSL
binding search replaced a rescan-until-stable loop. The references below are the
previous implementations, kept verbatim.
"""

import random

import pytest

from crosstl.translator.codegen.ast_walk import walk_ast_values
from crosstl.translator.codegen.directx_codegen import HLSLCodeGen
from crosstl.translator.codegen.GLSL_codegen import GLSLCodeGen
from crosstl.translator.codegen.slang_codegen import SlangCodeGen
from crosstl.translator.codegen.SPIRV_codegen import VulkanSPIRVCodeGen
from crosstl.translator.lexer import Lexer
from crosstl.translator.parser import Parser


def reference_walk(root):
    visited = set()

    def walk(value):
        if value is None or isinstance(value, (str, int, float, bool)):
            return
        if isinstance(value, dict):
            for item in value.values():
                yield from walk(item)
            return
        if isinstance(value, (list, tuple, set)):
            for item in value:
                yield from walk(item)
            return

        value_id = id(value)
        if value_id in visited:
            return
        visited.add(value_id)
        yield value

        if hasattr(value, "__dict__"):
            for child in vars(value).values():
                yield from walk(child)

    yield from walk(root)


def reference_next_binding(used_bindings, namespace, binding, count):
    count = max(count or 1, 1)
    ranges = used_bindings.get(namespace, [])
    while True:
        end = binding + count - 1
        conflict_end = None
        for used_start, used_end, _ in ranges:
            if binding <= used_end and used_start <= end:
                conflict_end = (
                    used_end if conflict_end is None else max(conflict_end, used_end)
                )
        if conflict_end is None:
            return binding
        binding = conflict_end + 1


class Node:
    def __init__(self, label):
        self.label = label


class Slotted:
    __slots__ = ("child",)

    def __init__(self, child):
        self.child = child


def random_graph(seed, size=60):
    rng = random.Random(seed)
    nodes = [Node(index) for index in range(size)]

    def value(depth):
        roll = rng.random()
        if depth > 3 or roll < 0.25:
            return rng.choice([None, 1, 2.5, True, "text", rng.choice(nodes)])
        if roll < 0.45:
            return [value(depth + 1) for _ in range(rng.randint(0, 3))]
        if roll < 0.55:
            return tuple(value(depth + 1) for _ in range(rng.randint(0, 3)))
        if roll < 0.65:
            return {f"k{i}": value(depth + 1) for i in range(rng.randint(0, 3))}
        if roll < 0.72:
            return {rng.choice(nodes) for _ in range(rng.randint(0, 3))}
        if roll < 0.8:
            return Slotted(value(depth + 1))
        return rng.choice(nodes)

    for node in nodes:
        for field in range(rng.randint(0, 4)):
            setattr(node, f"field{field}", value(0))
    return [nodes[0], nodes[1:5], {"rest": nodes[5:]}]


@pytest.mark.parametrize("seed", range(25))
def test_walk_matches_recursive_reference(seed):
    root = random_graph(seed)
    assert [id(value) for value in walk_ast_values(root)] == [
        id(value) for value in reference_walk(root)
    ]


def test_walk_reads_children_after_resuming():
    def pruned(walk):
        root = Node("root")
        root.keep = Node("keep")
        root.drop = Node("drop")
        labels = []
        for node in walk(root):
            labels.append(node.label)
            if node is root:
                del node.drop
        return labels

    assert pruned(walk_ast_values) == pruned(reference_walk) == ["root", "keep"]


def test_walk_handles_nesting_deeper_than_the_recursion_limit():
    root = node = Node(0)
    for index in range(1, 20000):
        node.next = Node(index)
        node = node.next
    assert [value.label for value in walk_ast_values(root)] == list(range(20000))


SHADER = """
shader WalkShader {
    struct Data { vec4 value; };
    cbuffer Params { float scale; };
    float helper(float x) { return x * scale + 1.0; }
    compute {
        @workgroup_size(8, 1, 1)
        void main() {
            float total = 0.0;
            for (int i = 0; i < 4; i++) { total += helper(float(i)); }
        }
    }
}
"""


@pytest.mark.parametrize(
    ("codegen", "method"),
    [
        (GLSLCodeGen, "walk_ast"),
        (HLSLCodeGen, "walk_ast"),
        (SlangCodeGen, "walk_ast"),
        (VulkanSPIRVCodeGen, "walk_ast_nodes"),
    ],
)
def test_codegen_walkers_match_reference_on_a_parsed_shader(codegen, method):
    ast = Parser(Lexer(SHADER).get_tokens()).parse()
    walker = getattr(codegen.__new__(codegen), method)
    assert [id(value) for value in walker(ast)] == [
        id(value) for value in reference_walk(ast)
    ]


@pytest.mark.parametrize("seed", range(40))
def test_next_binding_matches_rescanning_reference(seed):
    rng = random.Random(seed)
    used = {"space": []}
    for _ in range(rng.randint(0, 30)):
        start = rng.randint(0, 40)
        used["space"].append((start, start + rng.randint(-1, 6), "name"))
    generator = GLSLCodeGen.__new__(GLSLCodeGen)
    for binding in range(0, 50, 3):
        for count in (None, 0, 1, 2, 5, 9):
            for namespace in ("space", "other"):
                assert generator.next_available_resource_binding_from(
                    used, namespace, binding, count
                ) == reference_next_binding(used, namespace, binding, count)

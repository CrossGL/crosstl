"""``ASTNode.__deepcopy__`` builds the same object graph as ``copy.deepcopy``'s default protocol.

The reference is the default protocol itself, reached by hiding the method. Two copies are compared by a
canonical form that numbers every mutable object in first-visit order, so it records each value, each type, every
object reached twice (alias fields, ``parent`` back links) and every object the copy shares with the original.
"""

import copy
import enum

import pytest

from crosstl.translator import ast as A
from crosstl.translator.lexer import Lexer
from crosstl.translator.parser import Parser

PROGRAM = """
shader Copies {
    struct Pair { vec2 a; float b[2]; }
    const float SCALE = 2.0;

    float twice(float x) { return x * SCALE; }

    vertex {
        uniform mat4 transform;
        vec4 main(vec3 position) {
            Pair p;
            p.a = position.xy;
            p.b[0] = twice(position.z);
            int i = 0;
            for (int k = 0; k < 2; k++) { i += k; }
            if (i > 1) { p.a.x = -p.a.x; } else { p.a = vec2(0.0); }
            return transform * vec4(p.a, p.b[0], 1.0);
        }
    }

    fragment {
        vec4 main() { return vec4(1.0); }
    }
}
"""


class Colour(enum.Enum):
    RED = 1


class SelfCopying(A.ASTNode):
    """A node class with its own ``__deepcopy__``, which the fast path must still call."""

    def __init__(self, payload):
        super().__init__()
        self.payload = payload

    def __deepcopy__(self, memo):
        duplicate = SelfCopying(copy.deepcopy(self.payload, memo))
        duplicate.annotations["copied"] = True
        memo[id(self)] = duplicate
        return duplicate


def parse(source):
    return Parser(Lexer(source).get_tokens()).parse()


def canonical(root, original_ids):
    """Every reachable object in first-visit order: atoms by value, the rest by type, fields and back references."""
    numbers = {}
    out = []
    stack = [root]
    while stack:
        value = stack.pop()
        if type(value) in (str, int, float, bool, complex, bytes, type(None)):
            out.append(("atom", type(value).__name__, value))
            continue
        if id(value) in numbers:
            out.append(("ref", numbers[id(value)]))
            continue
        numbers[id(value)] = len(numbers)
        shared = id(value) in original_ids
        out.append(("obj", type(value).__qualname__, shared))
        if isinstance(value, A.StageMap):
            children = [item for pair in value.items() for item in pair]
        elif isinstance(value, dict):
            children = [item for pair in value.items() for item in pair]
        elif isinstance(value, (list, tuple)):
            children = list(value)
        elif isinstance(value, (set, frozenset)):
            children = sorted(value, key=repr)
        elif hasattr(value, "__dict__") and not isinstance(value, (type, enum.Enum)):
            out.append(("fields", tuple(vars(value))))
            children = list(vars(value).values())
        else:
            out.append(("opaque", repr(value)))
            children = []
        out.append(("len", len(children)))
        stack.extend(reversed(children))
    return out


def reachable_ids(root):
    return {
        id(value)
        for value in _reachable(root)
        if type(value) not in (str, int, float, bool, type(None))
    }


def _reachable(root):
    seen, stack = set(), [root]
    while stack:
        value = stack.pop()
        if id(value) in seen:
            continue
        seen.add(id(value))
        yield value
        if isinstance(value, dict):
            stack.extend(item for pair in value.items() for item in pair)
        elif isinstance(value, (list, tuple, set, frozenset)):
            stack.extend(value)
        elif hasattr(value, "__dict__") and not isinstance(value, (type, enum.Enum)):
            stack.extend(vars(value).values())


def default_deepcopy(value, monkeypatch):
    with monkeypatch.context() as patch:
        patch.setattr(A.ASTNode, "__deepcopy__", None)
        return copy.deepcopy(value)


def assert_same_copy(value, monkeypatch):
    original = reachable_ids(value)
    fast = copy.deepcopy(value)
    reference = default_deepcopy(value, monkeypatch)
    assert canonical(fast, original) == canonical(reference, original)
    return fast


def test_parsed_program_copies_like_the_default_protocol(monkeypatch):
    program = parse(PROGRAM)
    program.bind_parent_links()
    fast = assert_same_copy(program, monkeypatch)
    assert fast is not program
    assert isinstance(fast.stages, A.StageMap)


def test_alias_fields_and_back_links_stay_shared(monkeypatch):
    program = parse(PROGRAM)
    program.bind_parent_links()
    assignments = [
        node for node in program.walk() if isinstance(node, A.AssignmentNode)
    ]
    assert assignments
    fast = assert_same_copy(assignments[0], monkeypatch)
    # A subtree's parent link reaches the rest of the program, which the copy follows as the default does.
    assert fast.parent is not None and fast.parent is not assignments[0].parent
    if getattr(assignments[0], "left", None) is assignments[0].target:
        assert fast.left is fast.target


def test_mixed_field_values_copy_like_the_default_protocol(monkeypatch):
    shared = A.IdentifierNode("shared")
    node = A.IdentifierNode("outer")
    node.annotations = {"kind": Colour.RED, ("key", 1): [shared, shared]}
    node.alias_one = shared
    node.alias_two = shared
    node.items_tuple = (shared, 1, "text")
    node.atoms_tuple = (1, "text", None)
    node.names = {"a", "b"}
    node.frozen = frozenset({3, 4})
    node.nested = [[shared], {"inner": shared}]
    node.custom = SelfCopying([shared, 2])
    node.kind = Colour.RED
    node.text_subclass = Colour.RED.name
    node.cycle = node
    fast = assert_same_copy(node, monkeypatch)
    assert fast.alias_one is fast.alias_two is fast.nested[0][0]
    assert fast.cycle is fast
    assert fast.custom.annotations == {"copied": True}
    assert fast.custom.payload[0] is fast.alias_one
    assert fast.kind is Colour.RED


def test_shared_memo_with_an_outer_deepcopy(monkeypatch):
    """A node copied inside a larger structure is copied once and shared, as with the default protocol."""
    program = parse(PROGRAM)
    function = program.functions[0]
    outer = {"first": function, "second": [function, program]}
    fast = assert_same_copy(outer, monkeypatch)
    assert fast["first"] is fast["second"][0]


@pytest.mark.parametrize("value", [A.IdentifierNode("x"), A.LiteralNode(1, None)])
def test_copy_of_a_leaf_node_is_a_new_equal_node(value):
    duplicate = copy.deepcopy(value)
    assert duplicate is not value
    assert type(duplicate) is type(value)
    assert vars(duplicate).keys() == vars(value).keys()


def _pairs(original, duplicate):
    """Each mutable object reachable from ``original`` with the object at the same place in ``duplicate``."""
    seen, stack = set(), [(original, duplicate)]
    while stack:
        left, right = stack.pop()
        if type(left) in (str, int, float, bool, type(None)) or id(left) in seen:
            continue
        seen.add(id(left))
        yield left, right
        if isinstance(left, dict):
            stack.extend(zip(left.values(), right.values()))
        elif isinstance(left, list):
            stack.extend(zip(left, right))
        elif isinstance(left, A.ASTNode):
            stack.extend(zip(vars(left).values(), vars(right).values()))


def test_an_explicit_memo_maps_every_copied_object_to_its_copy():
    """Callers that pass their own memo look copies up in it (``legalize_hlsl_keyword_names``)."""
    program = parse(PROGRAM)
    program.bind_parent_links()
    memo = {}
    duplicate = copy.deepcopy(program, memo)
    pairs = list(_pairs(program, duplicate))
    assert len(pairs) > 100
    for original, copied in pairs:
        if isinstance(original, (A.ASTNode, list, dict)) and not isinstance(
            original, A.StageMap
        ):
            assert memo[id(original)] is copied
    # Every copied original stays alive in the memo, so no id in it can be reused.
    kept = {id(value) for value in memo[id(memo)]}
    assert all(
        id(original) in kept for original, _ in pairs if isinstance(original, A.ASTNode)
    )

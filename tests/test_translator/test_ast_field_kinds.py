"""Structural walks classify a field value once per type and keep the order and the alias rules they had.

``child_nodes``, :func:`~crosstl.translator.semantics.rewrite.child_slots` and the declared-name walks of
:mod:`~crosstl.translator.semantics.declared_types` dispatch on a per-type kind cache instead of a chain of
``isinstance`` checks. These tests pin what each walk yields for every kind of field, so the cache cannot change
a walk's order, its handling of aliases, or which containers it enters.
"""

import pytest

from crosstl.translator import ast as A
from crosstl.translator.semantics import declared_types as D
from crosstl.translator.semantics.rewrite import StageSlot, child_slots


class _ListSubclass(list):
    pass


class _DictSubclass(dict):
    pass


def _leaf(name):
    return A.IdentifierNode(name)


@pytest.mark.parametrize(
    ("value", "kind"),
    [
        (None, A.FIELD_LEAF),
        ("text", A.FIELD_LEAF),
        (3, A.FIELD_LEAF),
        (2.5, A.FIELD_LEAF),
        (True, A.FIELD_LEAF),
        (A.IdentifierNode("x"), A.FIELD_NODE),
        ([], A.FIELD_LIST),
        (_ListSubclass(), A.FIELD_LIST),
        (A.StageMap(), A.FIELD_STAGES),
        ({}, A.FIELD_DICT),
        (_DictSubclass(), A.FIELD_DICT),
        ((), A.FIELD_TUPLE),
        (set(), A.FIELD_SET),
        (frozenset(), A.FIELD_LEAF),
        (object(), A.FIELD_LEAF),
    ],
)
def test_field_kind_classifies_each_type(value, kind):
    assert A.field_kind(value) == kind
    # The second answer comes from the cache and agrees with the first.
    assert A.FIELD_KINDS[type(value)] == kind
    assert A.field_kind(value) == kind


def _node_with_every_field_kind():
    """A node whose fields hold every kind of value, with one child aliased under two names."""
    node = A.ASTNode()
    shared = _leaf("shared")
    node.first = shared
    node.alias = shared
    node.items = [_leaf("l0"), "skip", _leaf("l1"), [_leaf("nested")]]
    node.items_alias = node.items
    node.pair = (_leaf("t0"), _leaf("t1"))
    node.members = {"a": _leaf("d0"), "b": 7, "c": _leaf("d1")}
    node.unordered = {3}
    stages = A.StageMap()
    stages.append("vertex", _leaf("v0"))
    stages.append("vertex", _leaf("v1"))
    stages.append("fragment", _leaf("f0"))
    node.stages = stages
    node.count = 4
    node.label = "text"
    node.annotations = {"ignored": _leaf("annotation")}
    node.parent = _leaf("parent")
    return node


def _names(nodes):
    return [getattr(node, "name", None) for node in nodes]


def test_child_nodes_yield_each_child_object_once_in_field_order():
    node = _node_with_every_field_kind()
    assert _names(node.child_nodes()) == [
        "shared",
        "l0",
        "l1",
        "nested",
        "t0",
        "t1",
        "d0",
        "d1",
        "v0",
        "v1",
        "f0",
    ]


def test_walk_visits_the_nodes_child_nodes_yields():
    node = _node_with_every_field_kind()
    walked = list(node.walk())
    assert walked[0] is node
    assert _names(walked[1:]) == _names(node.child_nodes())


def test_child_slots_name_replaceable_slots_once_per_container():
    node = _node_with_every_field_kind()
    assert list(child_slots(node)) == [
        ("first", None),
        ("items", 0),
        ("items", 2),
        ("members", "a"),
        ("members", "c"),
        ("stages", StageSlot("vertex", 0)),
        ("stages", StageSlot("vertex", 1)),
        ("stages", StageSlot("fragment", 0)),
    ]


def test_declared_type_walk_children_keep_aliases_and_skip_sets():
    node = _node_with_every_field_kind()
    assert _names(D._children(node)) == [
        "shared",
        "shared",
        "l0",
        "l1",
        "l0",
        "l1",
        "t0",
        "t1",
        "d0",
        "d1",
        "v0",
        "v1",
        "f0",
    ]
    walked = list(D._nodes(node))
    assert walked[0] is node
    assert _names(walked[1:]) == [
        "shared",
        "l0",
        "l1",
        "t0",
        "t1",
        "d0",
        "d1",
        "v0",
        "v1",
        "f0",
    ]


def _program_with_dead_helper():
    helper = A.FunctionNode(
        "unused_helper", A.PrimitiveType("void"), [], A.BlockNode([])
    )
    entry = A.FunctionNode("main", A.PrimitiveType("void"), [], A.BlockNode([]))
    stage = A.StageNode(A.ShaderStage.COMPUTE, entry)
    program = A.ShaderNode(
        "P",
        A.ExecutionModel.COMPUTE_KERNEL,
        stages={A.ShaderStage.COMPUTE: stage},
        functions=[helper],
    )
    return program, helper


def test_shared_dead_functions_computes_once_and_returns_copies(monkeypatch):
    program, helper = _program_with_dead_helper()
    calls = []
    find = D._find_dead_functions

    def counting(program):
        calls.append(program)
        return find(program)

    monkeypatch.setattr(D, "_find_dead_functions", counting)
    with D.shared_dead_functions():
        first = D._dead_functions(program)
        assert first == {id(helper)}
        first.clear()
        assert D._dead_functions(program) == {id(helper)}
    assert len(calls) == 1
    # Outside the block every call computes the answer again.
    assert D._dead_functions(program) == {id(helper)}
    assert len(calls) == 2

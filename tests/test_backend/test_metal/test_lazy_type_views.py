"""Lazily resolved type views equal the views the Metal preprocessor builds eagerly."""

import copy
import pickle

import pytest

from crosstl.backend.Metal.preprocessing.lazy_type_views import LazyTypeView
from crosstl.backend.Metal.preprocessing.support import _MetalAddressableValueType
from crosstl.backend.Metal.preprocessor import MetalPreprocessor

NAMES = {"b": [1], "a": [2], "skip": [3], "c": [4]}


def _entry(name):
    return None if name == "skip" else (name, name.upper())


def _view(calls=None):
    def entry(name):
        if calls is not None:
            calls.append(name)
        return _entry(name)

    return LazyTypeView(NAMES, entry)


EAGER = {"b": "B", "a": "A", "c": "C"}


def test_a_lookup_resolves_only_the_name_it_reads():
    calls = []
    view = _view(calls)
    assert view["a"] == "A"
    assert view.get("c") == "C"
    assert "skip" not in view
    assert "missing" not in view
    assert view.get("missing", 7) == 7
    with pytest.raises(KeyError):
        view["skip"]
    assert calls == ["a", "c", "skip"]


@pytest.mark.parametrize(
    "operation",
    [
        list,
        len,
        bool,
        repr,
        lambda view: list(view.keys()),
        lambda view: list(view.values()),
        lambda view: list(view.items()),
        lambda view: list(reversed(view)),
        lambda view: dict(view),
        lambda view: {**view},
        lambda view: view.copy(),
        lambda view: copy.copy(view),
        lambda view: copy.deepcopy(view),
        lambda view: pickle.loads(pickle.dumps(view)),
        lambda view: view | {"d": "D"},
        lambda view: {"d": "D"} | view,
        lambda view: view == EAGER,
        lambda view: EAGER == view,
        lambda view: view != EAGER,
        lambda view: view == _view(),
    ],
)
def test_whole_view_operations_see_the_eager_view(operation):
    view = _view()
    view.get("c")  # a lookup first must not change the key order
    assert operation(view) == operation(dict(EAGER))


@pytest.mark.parametrize(
    "mutate",
    [
        lambda view: view.__setitem__("a", "new"),
        lambda view: view.__delitem__("a"),
        lambda view: view.pop("a"),
        lambda view: view.popitem(),
        lambda view: view.setdefault("d", "D"),
        lambda view: view.update({"a": "new"}),
        lambda view: view.__ior__({"e": "E"}),
        lambda view: view.clear(),
    ],
)
def test_mutations_act_on_the_eager_view(mutate):
    view = _view()
    eager = dict(EAGER)
    view.get("c")
    assert mutate(view) == mutate(eager)
    assert list(view.items()) == list(eager.items())
    assert dict(view) == eager


def test_a_key_that_differs_from_its_name():
    def entry(name):
        return {"p": ("p->", "P"), "q": ("q", "Q")}.get(name)

    def name_of(key):
        return key[:-2] if isinstance(key, str) and key.endswith("->") else key

    view = LazyTypeView({"p": [], "q": []}, entry, name_of)
    assert "p" not in view
    assert view["p->"] == "P"
    assert view.get("q->") is None
    assert dict(view) == {"p->": "P", "q": "Q"}


def _declarations():
    scoped = _MetalAddressableValueType("float", scope=(10, 20))
    return {
        "x": [(0, "int"), (12, scoped)],
        "y": [(5, "uint")],
        "late": [(50, "half")],
        "p": [(0, "Pair*")],
        "o": [(0, "Pair")],
        "r": [(0, "Pair**")],
    }


@pytest.mark.parametrize("position", [0, 6, 11, 15, 25, 60])
def test_the_preprocessor_views_equal_the_eager_views(position):
    preprocessor = MetalPreprocessor()
    declarations = _declarations()
    eager = preprocessor._flatten_types_at(declarations, position)
    lazy = preprocessor._lazy_flattened_types_at(declarations, position)
    for name in [*declarations, "absent"]:
        assert (name in lazy) == (name in eager)
        assert lazy.get(name) == eager.get(name)
    assert list(lazy.items()) == list(eager.items())


def test_the_struct_field_views_equal_the_eager_views():
    preprocessor = MetalPreprocessor()
    source = "struct Pair { float first; int second; };\n"
    structs = {
        struct.name: struct
        for struct in preprocessor._find_concrete_struct_definitions(source)
    }
    assert structs["Pair"].data_member_types
    declarations = _declarations()
    for position in (0, 6, 60):
        eager = preprocessor._struct_field_types_at(declarations, structs, position)
        lazy = preprocessor._lazy_struct_field_types_at(declarations, structs, position)
        assert {"p->", "o"} <= set(eager)
        for key in ["p", "p->", "o", "o->", "r", "r->", "x", "absent"]:
            assert lazy.get(key) == eager.get(key)
        assert list(lazy.items()) == list(eager.items())
    assert preprocessor._lazy_struct_field_types_at({}, structs, 0) == {}
    assert preprocessor._lazy_struct_field_types_at(declarations, {}, 0) == {}

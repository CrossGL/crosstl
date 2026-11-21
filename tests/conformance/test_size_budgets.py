import json

import pytest

from . import size_budgets
from .size_budgets import (
    FUNCTION_HEADROOM,
    FUNCTION_LIMIT,
    MODULE_LIMIT,
    budget_file,
    compare,
    function_lengths,
    line_count,
    load,
    lowered,
    measure,
    missing,
    resolved,
    write,
)

NESTED = """
import functools


@functools.cache
def top(a):
    def helper():
        return a

    return helper()


class Emitter:
    def emit(self):
        def visit(node):
            def leaf():
                return node

            return leaf

        return visit

    @property
    def name(self):
        return "a"

    @name.setter
    def name(self, value):
        self._name = value

    async def flush(self):
        class Local:
            def method(self):
                return 1

        return Local
"""


def budgets(modules=None, functions=None):
    return {"modules": modules or {}, "functions": functions or {}}


def test_functions_are_keyed_by_qualified_name_and_measured_from_def():
    assert function_lengths(NESTED) == {
        "top": 5,
        "top.<locals>.helper": 2,
        "Emitter.emit": 8,
        "Emitter.emit.<locals>.visit": 5,
        "Emitter.emit.<locals>.visit.<locals>.leaf": 2,
        "Emitter.name": 2,
        "Emitter.name#2": 2,
        "Emitter.flush": 6,
        "Emitter.flush.<locals>.Local.method": 2,
    }


def test_line_count_matches_wc():
    assert line_count("") == 0
    assert line_count("a\nb\n") == 2
    assert line_count("a\nb") == 2


def test_unrecorded_oversized_entries_and_growth_past_a_ceiling_fail():
    big = FUNCTION_LIMIT + 50
    functions = {"m.py::f": big, "m.py::g": big, "m.py::small": FUNCTION_LIMIT}
    modules = {"m.py": MODULE_LIMIT + 10, "n.py": MODULE_LIMIT}
    recorded = budgets({"m.py": MODULE_LIMIT + 10}, {"m.py::f": big})
    assert [line.split(":")[0] for line in compare(recorded, modules, functions)] == [
        "New oversized function m.py"
    ]
    grown = compare(
        recorded, {**modules, "m.py": MODULE_LIMIT + 11}, {"m.py::f": big + 1}
    )
    assert [line.split(" ", 3)[:3] for line in grown] == [
        ["Oversized", "module", "m.py"],
        ["Oversized", "function", "m.py::f"],
    ]
    assert "raise this ceiling" in grown[0]


def test_resolved_entries_are_reported_without_failing():
    recorded = budgets(
        {"gone.py": MODULE_LIMIT + 1, "m.py": MODULE_LIMIT + 1},
        {"m.py::split": FUNCTION_LIMIT + 1, "m.py::kept": FUNCTION_LIMIT + 1},
    )
    modules = {"m.py": MODULE_LIMIT}
    functions = {"m.py::split": 40, "m.py::kept": FUNCTION_LIMIT + 1}
    assert compare(recorded, modules, functions) == []
    assert resolved(recorded, modules, functions) == ["gone.py", "m.py", "m.py::split"]


def test_update_only_lowers_ceilings_and_never_admits():
    ceiling = FUNCTION_LIMIT + 200
    recorded = budgets(
        {},
        {
            "a.py::shrunk": ceiling,
            "a.py::near": ceiling,
            "a.py::split": ceiling,
            "b.py::other": ceiling,
        },
    )
    functions = {
        "a.py::shrunk": FUNCTION_LIMIT + 10,
        "a.py::near": ceiling - 1,
        "a.py::split": 12,
        "a.py::new": FUNCTION_LIMIT + 1,
        "b.py::other": FUNCTION_LIMIT + 1,
    }
    document = lowered(recorded, {}, functions, paths={"a.py"})
    assert document["functions"] == {
        "a.py::near": ceiling,
        "a.py::shrunk": FUNCTION_LIMIT + 10 + FUNCTION_HEADROOM,
        "b.py::other": ceiling,
    }
    assert lowered(recorded, {}, functions)["functions"]["b.py::other"] == (
        FUNCTION_LIMIT + 1 + FUNCTION_HEADROOM
    )
    assert compare(document, {}, functions)[0].startswith(
        "New oversized function a.py::new"
    )


def test_each_module_has_its_own_budget_file(tmp_path):
    document = budgets(
        {"crosstl/a.py": MODULE_LIMIT + 5},
        {
            "crosstl/a.py::A.f": FUNCTION_LIMIT + 1,
            "crosstl/sub/b.py::g": FUNCTION_LIMIT + 2,
        },
    )
    write(document, {"crosstl/a.py", "crosstl/sub/b.py"}, tmp_path)
    assert json.loads(budget_file("crosstl/a.py", tmp_path).read_text()) == {
        "module": MODULE_LIMIT + 5,
        "functions": {"A.f": FUNCTION_LIMIT + 1},
    }
    assert json.loads(budget_file("crosstl/sub/b.py", tmp_path).read_text()) == {
        "functions": {"g": FUNCTION_LIMIT + 2}
    }
    assert load(tmp_path) == document


def test_update_deletes_the_file_of_a_module_under_both_limits(tmp_path):
    recorded = budgets(
        {"crosstl/a.py": MODULE_LIMIT + 5},
        {"crosstl/a.py::f": FUNCTION_LIMIT + 1, "crosstl/b.py::g": FUNCTION_LIMIT + 1},
    )
    write(recorded, {"crosstl/a.py", "crosstl/b.py"}, tmp_path)
    modules = {"crosstl/a.py": MODULE_LIMIT, "crosstl/b.py": 10}
    functions = {"crosstl/a.py::f": 20, "crosstl/b.py::g": FUNCTION_LIMIT + 1}
    write(
        lowered(recorded, modules, functions, {"crosstl/a.py"}),
        {"crosstl/a.py"},
        tmp_path,
    )
    assert not budget_file("crosstl/a.py", tmp_path).exists()
    assert load(tmp_path) == budgets({}, {"crosstl/b.py::g": FUNCTION_LIMIT + 1})


def test_a_budget_file_for_a_missing_module_fails(tmp_path):
    recorded = budgets({}, {"crosstl/old.py::f": FUNCTION_LIMIT + 1})
    modules = {"crosstl/new.py": 10}
    assert missing(recorded, modules) == [
        "Budget file for missing module crosstl/old.py: "
        "move it with the module (git mv) or delete it"
    ]
    assert missing(recorded, {"crosstl/old.py": 10}) == []


def test_a_budget_file_with_unknown_keys_is_rejected(tmp_path):
    path = budget_file("crosstl/a.py", tmp_path)
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"modules": 1}))
    with pytest.raises(ValueError, match="unknown keys"):
        load(tmp_path)


def test_translator_modules_and_functions_stay_within_their_ceilings():
    assert not (size_budgets.HERE / "size_budgets.json").exists()
    document = load()
    modules, functions = measure()
    assert missing(document, modules) == []
    assert compare(document, modules, functions) == []
    for path in size_budgets.BUDGETS.rglob("*"):
        assert path.is_dir() or path.name.endswith(".py.json"), path

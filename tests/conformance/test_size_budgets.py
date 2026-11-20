import json

from . import size_budgets
from .size_budgets import (
    FUNCTION_HEADROOM,
    FUNCTION_LIMIT,
    MODULE_LIMIT,
    compare,
    function_lengths,
    line_count,
    lowered,
    measure,
    resolved,
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


def test_translator_modules_and_functions_stay_within_their_ceilings():
    document = json.loads(size_budgets.BUDGETS.read_text())
    assert set(document) == {"schemaVersion", "limits", "modules", "functions"}
    assert document["limits"] == {
        "moduleLines": MODULE_LIMIT,
        "functionLines": FUNCTION_LIMIT,
    }
    modules, functions = measure()
    assert compare(document, modules, functions) == []

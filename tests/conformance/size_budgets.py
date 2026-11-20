"""Static check that the translator's oversized modules and functions do not grow.

The engineering criterion is that no module under ``crosstl/translator`` or ``crosstl/backend`` exceeds
5,000 lines and no function exceeds 300 lines. The modules and functions that already exceed a limit are
recorded in ``size_budgets.json`` with a ceiling. The check fails when:

- a module or function that is not recorded exceeds its limit, so new code goes into new, small modules;
- a recorded module or function grows past its ceiling.

An entry whose module or function is gone or back within its limit is resolved debt. It does not fail the check,
because the pull request that resolves it may land after others that edit the same file; ``report`` counts these
entries and ``update`` drops them.

A function is keyed by its file and its qualified name (``Class.method``, ``outer.<locals>.inner``); a later
definition with the same qualified name in the same file gets a ``#2``, ``#3`` suffix. A function's length runs
from its ``def`` line to its last line, so decorators do not count; a module's length is its physical line count.

A ceiling sits a little above the size it was recorded at (100 lines for a module, 15 for a function), so that
a small bug fix does not have to edit this file, while new features have to land in new modules. Ceilings only
fall: ``update`` lowers each one to at most the current size plus that headroom and drops resolved entries. Give
it paths to touch only those modules, so that a split does not tighten modules that other pull requests are
editing. It never admits a new entry. A bug fix that has to grow a recorded module or function past its ceiling
raises that one ceiling by hand in the same pull request, and the review sees the number.

Usage: ``python -m tests.conformance.size_budgets [check|update [PATH ...]|report]``.
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parents[1]
BUDGETS = HERE / "size_budgets.json"
ROOTS = ("crosstl/translator", "crosstl/backend")
MODULE_LIMIT = 5000
FUNCTION_LIMIT = 300
MODULE_HEADROOM = 100
FUNCTION_HEADROOM = 15
SCHEMA = 1


def line_count(text):
    """Physical lines, as ``wc -l`` counts them for a file that ends in a newline."""
    return text.count("\n") + (1 if text and not text.endswith("\n") else 0)


def function_lengths(text, path="<string>"):
    """``{qualified name: lines}`` for every function in the module, nested ones included."""
    lengths = {}

    def visit(node, prefix):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                name = f"{prefix}{child.name}"
                key, ordinal = name, 1
                while key in lengths:
                    ordinal += 1
                    key = f"{name}#{ordinal}"
                lengths[key] = child.end_lineno - child.lineno + 1
                visit(child, f"{name}.<locals>.")
            elif isinstance(child, ast.ClassDef):
                visit(child, f"{prefix}{child.name}.")
            else:
                visit(child, prefix)

    visit(ast.parse(text, filename=path), "")
    return lengths


def measure(root=REPOSITORY):
    """``(modules, functions)``: every module's line count and every function's length, keyed by path."""
    modules, functions = {}, {}
    for top in ROOTS:
        for path in sorted((Path(root) / top).rglob("*.py")):
            relative = path.relative_to(root).as_posix()
            text = path.read_text(encoding="utf-8")
            modules[relative] = line_count(text)
            for name, length in function_lengths(text, relative).items():
                functions[f"{relative}::{name}"] = length
    return modules, functions


def oversized(modules, functions):
    """The entries a budget file has to record: everything over its limit."""
    return (
        {key: size for key, size in modules.items() if size > MODULE_LIMIT},
        {key: size for key, size in functions.items() if size > FUNCTION_LIMIT},
    )


def resolved(budgets, modules, functions):
    """Recorded entries whose module or function is gone or back within its limit."""
    return sorted(
        key
        for ceilings, sizes, limit in (
            (budgets["modules"], modules, MODULE_LIMIT),
            (budgets["functions"], functions, FUNCTION_LIMIT),
        )
        for key in ceilings
        if sizes.get(key, 0) <= limit
    )


def compare(budgets, modules, functions):
    """Failure messages for ``budgets`` (the parsed file) against the measured sizes."""
    failures = []
    kinds = (
        ("module", budgets["modules"], modules, MODULE_LIMIT),
        ("function", budgets["functions"], functions, FUNCTION_LIMIT),
    )
    for kind, ceilings, sizes, limit in kinds:
        for key in sorted(k for k, size in sizes.items() if size > limit):
            size, ceiling = sizes[key], ceilings.get(key)
            if ceiling is None:
                failures.append(
                    f"New oversized {kind} {key}: {size} lines, limit {limit}; "
                    "move the new logic into a new module or function"
                )
            elif size > ceiling:
                failures.append(
                    f"Oversized {kind} {key} grew to {size} lines, ceiling {ceiling}; "
                    "move the new logic out, or raise this ceiling in the same pull request for a bug fix"
                )
    return failures


def lowered(budgets, modules, functions, paths=None):
    """The budget file after ``update``: ceilings fall towards the current size, resolved entries go."""

    def touched(key):
        return paths is None or key.split("::", 1)[0] in paths

    def lower(ceilings, sizes, limit, headroom):
        result = {}
        for key, ceiling in ceilings.items():
            if not touched(key):
                result[key] = ceiling
            elif key in sizes and sizes[key] > limit:
                result[key] = min(ceiling, sizes[key] + headroom)
        return dict(sorted(result.items()))

    return {
        "schemaVersion": SCHEMA,
        "limits": {"moduleLines": MODULE_LIMIT, "functionLines": FUNCTION_LIMIT},
        "modules": lower(budgets["modules"], modules, MODULE_LIMIT, MODULE_HEADROOM),
        "functions": lower(
            budgets["functions"], functions, FUNCTION_LIMIT, FUNCTION_HEADROOM
        ),
    }


def report(budgets, modules, functions):
    """Totals that track the debt: how many entries remain and how many lines sit over the limits."""
    big_modules, big_functions = oversized(modules, functions)
    return {
        "oversizedModules": len(big_modules),
        "linesOverModuleLimit": sum(
            size - MODULE_LIMIT for size in big_modules.values()
        ),
        "oversizedFunctions": len(big_functions),
        "linesOverFunctionLimit": sum(
            size - FUNCTION_LIMIT for size in big_functions.values()
        ),
        "recordedEntries": len(budgets["modules"]) + len(budgets["functions"]),
        "resolvedEntries": len(resolved(budgets, modules, functions)),
    }


def write(document, path=BUDGETS):
    path.write_text(json.dumps(document, indent=1, sort_keys=True) + "\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument(
        "command", choices=("check", "update", "report"), nargs="?", default="check"
    )
    parser.add_argument(
        "paths", nargs="*", help="update: only these modules (repository-relative)"
    )
    args = parser.parse_args(argv)
    budgets = json.loads(BUDGETS.read_text())
    modules, functions = measure()
    if args.command == "report":
        print(json.dumps(report(budgets, modules, functions), indent=2))
        return 0
    if args.command == "update":
        paths = {Path(p).as_posix() for p in args.paths} or None
        document = lowered(budgets, modules, functions, paths)
        write(document)
        grown = compare(document, modules, functions)
        print(
            f"Recorded {len(document['modules'])} modules and "
            f"{len(document['functions'])} functions in {BUDGETS.name}"
        )
        if grown:
            print("\n".join(grown), file=sys.stderr)
            return 1
        return 0
    failures = compare(budgets, modules, functions)
    for failure in failures:
        print(failure, file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())

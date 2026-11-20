"""Static check that every refusal the translator raises is typed.

A typed refusal is an exception with a stable diagnostic code: a ``TranslationDiagnostic`` (which ``LoweringRefused``
and the importers' refusals derive from), or a class that defines ``project_diagnostic_code``. The project pipeline,
the corpus runner and the fuzzer report such a refusal by its code; any other exception is reported as a translator
error, whatever its message says.

An ``untyped-raise`` site is a ``raise`` of a built-in exception (``ValueError``, ``SyntaxError``, ``TypeError``,
``NotImplementedError``, ``RuntimeError``, ``KeyError``, ``IndexError``, ``LookupError``, ``Exception``), or of a
class under ``crosstl/translator`` or ``crosstl/backend`` that derives from one without being typed. A bare
``raise``, a raise of a variable, an ``AssertionError`` (an internal invariant, not a refusal) and a private class
(``_Unresolved``, a signal its module raises and catches) are not sites.

Sites are keyed and counted like ``fail_closed.py``'s, in ``typed_refusals_baseline.json``; the check fails when a
count rises, a new key appears, or a count falls without the baseline being lowered. The target is an empty
baseline.

Usage: ``python -m tests.conformance.typed_refusals [check|update|report]``.
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from collections import Counter
from pathlib import Path

from .fail_closed import REPOSITORY, ROOTS, compare

HERE = Path(__file__).resolve().parent
BASELINE = HERE / "typed_refusals_baseline.json"
KIND = "untyped-raise"
BUILTIN_EXCEPTIONS = frozenset(
    {
        "ValueError",
        "SyntaxError",
        "TypeError",
        "NotImplementedError",
        "RuntimeError",
        "KeyError",
        "IndexError",
        "LookupError",
        "Exception",
    }
)
TYPED_ROOTS = frozenset({"TranslationDiagnostic"})


def _name(node):
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _defines_diagnostic_code(class_node):
    for statement in class_node.body:
        targets = []
        if isinstance(statement, ast.Assign):
            targets = statement.targets
        elif isinstance(statement, ast.AnnAssign):
            targets = [statement.target]
        elif isinstance(statement, ast.FunctionDef):
            if statement.name == "project_diagnostic_code":
                return True
        if any(_name(target) == "project_diagnostic_code" for target in targets):
            return True
        if isinstance(statement, ast.FunctionDef) and statement.name == "__init__":
            for node in ast.walk(statement):
                if (
                    isinstance(node, ast.Attribute)
                    and node.attr == "project_diagnostic_code"
                    and isinstance(node.ctx, ast.Store)
                ):
                    return True
    return False


def exception_classes(sources):
    """``{class name: typed}`` for every exception class the sources define, by name across modules.

    A name defined more than once is typed only when every definition is.
    """
    bases, coded = {}, {}
    for text in sources:
        for node in ast.walk(ast.parse(text)):
            if isinstance(node, ast.ClassDef):
                names = [_name(base) for base in node.bases]
                bases.setdefault(node.name, []).append([n for n in names if n])
                coded.setdefault(node.name, []).append(_defines_diagnostic_code(node))

    memo = {}

    def classify(name, seen=()):
        """(is an exception, is typed) for ``name``."""
        if name in TYPED_ROOTS:
            return True, True
        if name in BUILTIN_EXCEPTIONS or name in {"AssertionError", "BaseException"}:
            return True, False
        if name not in bases or name in seen:
            return False, False
        if name in memo:
            return memo[name]
        exception, typed = False, True
        for index, definition in enumerate(bases[name]):
            results = [classify(base, seen + (name,)) for base in definition]
            is_exception = any(result[0] for result in results)
            exception |= is_exception
            typed &= coded[name][index] or any(result[1] for result in results)
        memo[name] = (exception, exception and typed)
        return memo[name]

    return {name: classify(name)[1] for name in sorted(bases) if classify(name)[0]}


def _internal(name):
    """A private exception class (``_Unresolved``) signals within its module and is caught there."""
    return name.startswith("_")


class _Visitor(ast.NodeVisitor):
    def __init__(self, untyped):
        self.untyped = untyped
        self.scope = []
        self.sites = Counter()

    def _scoped(self, node):
        self.scope.append(node.name)
        self.generic_visit(node)
        self.scope.pop()

    visit_FunctionDef = visit_AsyncFunctionDef = visit_ClassDef = _scoped

    def visit_Raise(self, node):
        raised = node.exc.func if isinstance(node.exc, ast.Call) else node.exc
        if raised is not None and _name(raised) in self.untyped:
            self.sites[(".".join(self.scope) or "<module>", KIND)] += 1
        self.generic_visit(node)


def _sources(root):
    for top in ROOTS:
        for path in sorted((Path(root) / top).rglob("*.py")):
            yield path.relative_to(root).as_posix(), path.read_text()


def untyped_names(sources):
    classes = exception_classes(sources)
    return BUILTIN_EXCEPTIONS | {
        name for name, typed in classes.items() if not typed and not _internal(name)
    }


def scan_source(text, untyped, path="<string>"):
    """Untyped raise sites of one module: {(function, kind): count}."""
    visitor = _Visitor(untyped)
    visitor.visit(ast.parse(text, filename=path))
    return visitor.sites


def scan(root=REPOSITORY):
    modules = list(_sources(root))
    untyped = untyped_names(text for _path, text in modules)
    sites = {}
    for relative, text in modules:
        for (function, kind), count in sorted(
            scan_source(text, untyped, relative).items()
        ):
            sites[f"{relative}::{function}::{kind}"] = count
    return dict(sorted(sites.items()))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument(
        "command", choices=("check", "update", "report"), nargs="?", default="check"
    )
    args = parser.parse_args(argv)
    current = scan()
    if args.command == "report":
        by_root = Counter()
        for key, count in current.items():
            by_root[key.split("/", 2)[1]] += count
        print(
            json.dumps(
                {KIND: sum(current.values()), **dict(sorted(by_root.items()))}, indent=2
            )
        )
        return 0
    baseline = json.loads(BASELINE.read_text())["sites"] if BASELINE.is_file() else {}
    if args.command == "update":
        grown = [f for f in compare(baseline, current) if f.startswith("New")]
        if baseline and grown:
            print("\n".join(grown), file=sys.stderr)
            print(
                "The baseline only shrinks; type the new refusals instead.",
                file=sys.stderr,
            )
            return 1
        document = {"schemaVersion": 1, "sites": current}
        BASELINE.write_text(json.dumps(document, indent=1, sort_keys=True) + "\n")
        print(
            f"Recorded {sum(current.values())} untyped raise sites in {BASELINE.name}"
        )
        return 0
    failures = [
        failure.replace("fail-open site", "untyped raise site").replace(
            "tests.conformance.fail_closed", "tests.conformance.typed_refusals"
        )
        for failure in compare(baseline, current)
    ]
    for failure in failures:
        print(failure, file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

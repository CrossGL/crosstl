"""Static check that the translator does not grow fail-open sites.

CrossTL must refuse what it cannot translate. Three code shapes break that rule silently:

- ``fallback-text``: a string literal that writes a loss marker into the output (``; WARNING``, ``// WARNING``,
  ``/* unsupported``, ``placeholder``, ``using default value``) instead of raising a diagnostic;
- ``identity-fallback``: ``mapping.get(key, key)``, which passes an unknown name through verbatim;
- ``empty-block-recovery``: an exception handler that replaces what failed to parse with an empty block.

Each site is keyed by file, enclosing function and kind, and counted. ``fail_closed_baseline.json`` records the
current counts; the check fails when a count rises or a new key appears, and also when a count falls without the
baseline being lowered, so that every removal is recorded and cannot be spent again. The baseline only shrinks:
the target is an empty baseline, enforced by this check.

Usage: ``python -m tests.conformance.fail_closed [check|update|report]``.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parents[1]
BASELINE = HERE / "fail_closed_baseline.json"
ROOTS = ("crosstl/translator", "crosstl/backend")
MARKERS = re.compile(
    r"; WARNING|// WARNING|/\*\s*unsupported|\bplaceholder\b|using default value", re.I
)
KINDS = ("fallback-text", "identity-fallback", "empty-block-recovery")


class _Visitor(ast.NodeVisitor):
    def __init__(self):
        self.scope = []
        self.sites = Counter()
        self.skip = set()

    def _add(self, kind):
        self.sites[(".".join(self.scope) or "<module>", kind)] += 1

    def _scoped(self, node):
        body = getattr(node, "body", None)
        if (
            body
            and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
        ):
            self.skip.add(
                id(body[0].value)
            )  # docstrings describe behaviour; they emit nothing
        self.scope.append(node.name)
        self.generic_visit(node)
        self.scope.pop()

    visit_FunctionDef = visit_AsyncFunctionDef = visit_ClassDef = _scoped

    def visit_Module(self, node):
        if (
            node.body
            and isinstance(node.body[0], ast.Expr)
            and isinstance(node.body[0].value, ast.Constant)
        ):
            self.skip.add(id(node.body[0].value))
        self.generic_visit(node)

    def visit_Raise(self, node):
        # Text inside a raised diagnostic reaches the user as an error, never the output.
        for child in ast.walk(node):
            self.skip.add(id(child))
        self.generic_visit(node)

    def visit_Constant(self, node):
        if (
            isinstance(node.value, str)
            and id(node) not in self.skip
            and MARKERS.search(node.value)
        ):
            self._add("fallback-text")

    def visit_Call(self, node):
        function = node.func
        if (
            isinstance(function, ast.Attribute)
            and function.attr == "get"
            and len(node.args) == 2
            and ast.dump(node.args[0]) == ast.dump(node.args[1])
        ):
            self._add("identity-fallback")
        self.generic_visit(node)

    def visit_ExceptHandler(self, node):
        for child in ast.walk(ast.Module(body=node.body, type_ignores=[])):
            if (
                isinstance(child, ast.Call)
                and _name(child.func).endswith(("Block", "BlockNode"))
                and _empty_arguments(child)
            ):
                self._add("empty-block-recovery")
        self.generic_visit(node)


def _name(node):
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return ""


def _empty_arguments(call):
    if not call.args and not call.keywords:
        return True
    return (
        len(call.args) == 1
        and isinstance(call.args[0], (ast.List, ast.Tuple))
        and not call.args[0].elts
    )


def scan_source(text, path="<string>"):
    """Fail-open sites of one module: {(function, kind): count}."""
    visitor = _Visitor()
    visitor.visit(ast.parse(text, filename=path))
    return visitor.sites


def scan(root=REPOSITORY):
    sites = {}
    for top in ROOTS:
        for path in sorted((Path(root) / top).rglob("*.py")):
            relative = path.relative_to(root).as_posix()
            for (function, kind), count in sorted(
                scan_source(path.read_text(), relative).items()
            ):
                sites[f"{relative}::{function}::{kind}"] = count
    return dict(sorted(sites.items()))


def compare(baseline, current):
    """Failures: new or grown sites, and removed sites the baseline still records."""
    failures = []
    for key in sorted(set(baseline) | set(current)):
        before, after = baseline.get(key, 0), current.get(key, 0)
        if after > before:
            failures.append(f"New fail-open site {key}: {before} -> {after}")
        elif after < before:
            failures.append(
                f"Removed fail-open site {key}: {before} -> {after}; lower the baseline with "
                "`python -m tests.conformance.fail_closed update`"
            )
    return failures


def totals(sites):
    counts = Counter()
    for key, count in sites.items():
        counts[key.rsplit("::", 1)[1]] += count
    return {kind: counts[kind] for kind in KINDS}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument(
        "command", choices=("check", "update", "report"), nargs="?", default="check"
    )
    args = parser.parse_args(argv)
    current = scan()
    if args.command == "report":
        print(json.dumps(totals(current), indent=2))
        return 0
    if args.command == "update":
        baseline = (
            json.loads(BASELINE.read_text())["sites"] if BASELINE.is_file() else {}
        )
        grown = [f for f in compare(baseline, current) if f.startswith("New")]
        if baseline and grown:
            print("\n".join(grown), file=sys.stderr)
            print(
                "The baseline only shrinks; fix the new sites instead.", file=sys.stderr
            )
            return 1
        document = {"schemaVersion": 1, "totals": totals(current), "sites": current}
        BASELINE.write_text(json.dumps(document, indent=1, sort_keys=True) + "\n")
        print(f"Recorded {sum(current.values())} fail-open sites in {BASELINE.name}")
        return 0
    failures = compare(json.loads(BASELINE.read_text())["sites"], current)
    for failure in failures:
        print(failure, file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

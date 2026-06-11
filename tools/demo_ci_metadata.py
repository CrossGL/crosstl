#!/usr/bin/env python3
"""Emit and validate CI inputs for the open-source porting demo."""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_METADATA_PATH = ROOT / "support" / "demo-ci-metadata.json"
SELECTOR_RE = re.compile(r"^[A-Za-z0-9_]+$")


class DemoCiMetadataError(ValueError):
    """Raised when demo CI metadata cannot be consumed safely."""


def _repo_path(path: str, root: Path = ROOT) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        raise DemoCiMetadataError(f"metadata path must be repository-relative: {path}")
    return root / candidate


def _test_function_names(test_file: Path) -> set[str]:
    try:
        tree = ast.parse(test_file.read_text(encoding="utf-8"))
    except OSError as exc:
        raise DemoCiMetadataError(f"unable to read pytest file {test_file}: {exc}") from exc
    return {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name.startswith("test_")
    }


def validation_errors(metadata: dict[str, Any], *, root: Path = ROOT) -> list[str]:
    errors: list[str] = []
    if metadata.get("schema_version") != 1:
        errors.append("schema_version must be 1")

    pytest_config = metadata.get("pytest")
    if not isinstance(pytest_config, dict):
        return errors + ["pytest must be an object"]

    test_file = pytest_config.get("test_file")
    if not isinstance(test_file, str) or not test_file:
        errors.append("pytest.test_file must be a non-empty string")
        test_names: set[str] = set()
    else:
        try:
            test_path = _repo_path(test_file, root)
            if not test_path.is_file():
                errors.append(f"pytest.test_file does not exist: {test_file}")
                test_names = set()
            else:
                test_names = _test_function_names(test_path)
        except DemoCiMetadataError as exc:
            errors.append(str(exc))
            test_names = set()

    cases = pytest_config.get("cases")
    if not isinstance(cases, list) or not cases:
        return errors + ["pytest.cases must be a non-empty list"]

    seen_ids: set[str] = set()
    seen_selectors: set[str] = set()
    for index, case in enumerate(cases):
        field = f"pytest.cases[{index}]"
        if not isinstance(case, dict):
            errors.append(f"{field} must be an object")
            continue

        case_id = case.get("id")
        if not isinstance(case_id, str) or not case_id:
            errors.append(f"{field}.id must be a non-empty string")
        elif case_id in seen_ids:
            errors.append(f"{field}.id is duplicated: {case_id}")
        else:
            seen_ids.add(case_id)

        selector = case.get("selector")
        if not isinstance(selector, str) or not selector:
            errors.append(f"{field}.selector must be a non-empty string")
            selector_matches: set[str] = set()
        elif not SELECTOR_RE.fullmatch(selector):
            errors.append(f"{field}.selector must be a pytest -k safe token: {selector}")
            selector_matches = set()
        elif selector in seen_selectors:
            errors.append(f"{field}.selector is duplicated: {selector}")
            selector_matches = set()
        else:
            seen_selectors.add(selector)
            selector_matches = {name for name in test_names if selector in name}

        targets = case.get("targets")
        if (
            not isinstance(targets, list)
            or not targets
            or not all(isinstance(target, str) and target for target in targets)
        ):
            errors.append(f"{field}.targets must be a non-empty string list")

        tests = case.get("tests")
        if (
            not isinstance(tests, list)
            or not tests
            or not all(isinstance(test, str) and test for test in tests)
        ):
            errors.append(f"{field}.tests must be a non-empty string list")
            continue

        missing = sorted(test for test in tests if test not in test_names)
        if missing:
            errors.append(f"{field}.tests are missing from {test_file}: {missing}")

        if selector_matches and sorted(tests) != sorted(selector_matches):
            errors.append(
                f"{field}.selector {selector!r} matches {sorted(selector_matches)}, "
                f"but metadata lists {sorted(tests)}"
            )
        elif not selector_matches:
            errors.append(f"{field}.selector does not match any test in {test_file}")

    return errors


def load_metadata(path: Path = DEFAULT_METADATA_PATH, *, root: Path = ROOT) -> dict[str, Any]:
    try:
        metadata = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise DemoCiMetadataError(f"unable to read demo CI metadata {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise DemoCiMetadataError(f"invalid demo CI metadata JSON {path}: {exc}") from exc
    if not isinstance(metadata, dict):
        raise DemoCiMetadataError("demo CI metadata must be a JSON object")

    errors = validation_errors(metadata, root=root)
    if errors:
        raise DemoCiMetadataError("\n".join(errors))
    return metadata


def pytest_files(metadata: dict[str, Any]) -> list[str]:
    return [metadata["pytest"]["test_file"]]


def pytest_selectors(metadata: dict[str, Any]) -> list[str]:
    return [case["selector"] for case in metadata["pytest"]["cases"]]


def pytest_selector_expression(metadata: dict[str, Any]) -> str:
    return " or ".join(pytest_selectors(metadata))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Emit and validate open-source porting demo CI inputs."
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=DEFAULT_METADATA_PATH,
        help="Path to demo CI metadata JSON.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("check", help="Validate the checked-in metadata.")
    subparsers.add_parser(
        "emit-pytest-files",
        help="Print pytest files, one per line, for the demo CI workflow.",
    )
    subparsers.add_parser(
        "emit-pytest-selector",
        help="Print the pytest -k selector for the demo CI workflow.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        metadata = load_metadata(args.metadata)
    except DemoCiMetadataError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    if args.command == "check":
        print("demo CI metadata is valid")
        return 0
    if args.command == "emit-pytest-files":
        print("\n".join(pytest_files(metadata)))
        return 0
    if args.command == "emit-pytest-selector":
        print(pytest_selector_expression(metadata))
        return 0

    parser.error(f"unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Summarize pytest JUnit failures into CI-friendly backend categories."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
GENERATOR = "tools/pytest_failure_summary.py"
SCHEMA_VERSION = 1
BACKEND_PATTERNS = {
    "directx": ("directx", "hlsl", "dxc"),
    "metal": ("metal", "msl"),
    "opengl": ("opengl", "glsl", "glslang"),
    "vulkan": ("spirv", "spv", "vulkan", "spir-v"),
    "cuda": ("cuda", "nvcc"),
    "hip": ("hip",),
    "mojo": ("mojo",),
    "rust": ("rust", "rustc"),
    "slang": ("slang", "slangc"),
}
COMPILER_MARKERS = (
    "compiler",
    "external_shader_validators",
    "glslang",
    "metallib",
    "mojo compiler",
    "spirv-as",
    "spirv-val",
    "dxc",
    "slangc",
    "nvcc",
    "rustc",
    "shader_validation",
    "xcrun",
    "validator",
)
COMPILER_VALIDATION_TEST_FILES = {
    "tests/test_translator/test_codegen/test_external_shader_validators.py",
    "tests/test_translator/test_codegen/test_shader_validation.py",
}
SUPPORT_TOOL_FILES = {
    "tools/ci_coverage.py",
    "tools/pytest_failure_summary.py",
    "tools/support_ci_summary.py",
    "tools/support_matrix.py",
    "tools/support_signals.py",
    "tools/sync_pr_issue_links.py",
    "tools/sync_support_issues.py",
}


def stable_json(data: Any) -> str:
    return json.dumps(data, indent=2, sort_keys=True) + "\n"


def display_path(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def load_error(path: Path, error_type: str, message: str) -> dict[str, Any]:
    return {
        "path": display_path(path),
        "load_error": {
            "type": error_type,
            "message": message,
        },
    }


def normalize_test_file(path: str | None, classname: str) -> str:
    if path:
        return path.replace("\\", "/")
    dotted = classname.replace(".", "/")
    if "/tests/" in dotted:
        dotted = dotted[dotted.index("/tests/") + 1 :]
    if dotted.startswith("tests/") and not dotted.endswith(".py"):
        return dotted + ".py"
    return dotted


def infer_backend(test_file: str, nodeid: str, message: str) -> str:
    haystack = " ".join((test_file, nodeid, message)).lower()
    for backend, patterns in BACKEND_PATTERNS.items():
        if any(pattern in haystack for pattern in patterns):
            return backend
    return "unknown"


def infer_category(test_file: str, nodeid: str, message: str) -> str:
    haystack = " ".join((test_file, nodeid, message)).lower()
    if (
        test_file.startswith("tests/test_support")
        or test_file
        in {
            "tests/test_ci_workflows.py",
            "tests/test_tool_cli.py",
            "tests/test_pr_issue_links.py",
        }
        or test_file in SUPPORT_TOOL_FILES
    ):
        return "support_automation"
    if test_file.startswith("tests/test_examples") or test_file.startswith("examples/"):
        return "examples"
    if test_file.startswith("tests/test_translator/test_codegen/"):
        if test_file in COMPILER_VALIDATION_TEST_FILES or any(
            marker in haystack for marker in COMPILER_MARKERS
        ):
            return "backend_compiler_validation"
        return "backend_codegen"
    if test_file.startswith("tests/test_backend/"):
        return "backend_frontend"
    if test_file.startswith("tests/test_translator/"):
        return "frontend_ir"
    return "unknown"


def testcase_nodeid(testcase: ET.Element) -> str:
    classname = testcase.get("classname", "")
    name = testcase.get("name", "")
    return "::".join(part for part in (classname, name) if part)


def failure_message(child: ET.Element) -> str:
    parts = [child.get("message", ""), child.text or ""]
    return "\n".join(part for part in parts if part).strip()


def junit_int_total(root: ET.Element, attribute: str) -> int:
    if root.get(attribute) is not None:
        return int(root.get(attribute, "0") or 0)
    return sum(
        int(testsuite.get(attribute, "0") or 0) for testsuite in root.iter("testsuite")
    )


def summarize_testcase(testcase: ET.Element) -> dict[str, Any] | None:
    failure = testcase.find("failure")
    error = testcase.find("error")
    child = failure if failure is not None else error
    if child is None:
        return None
    test_file = normalize_test_file(testcase.get("file"), testcase.get("classname", ""))
    nodeid = testcase_nodeid(testcase)
    message = failure_message(child)
    return {
        "nodeid": nodeid,
        "file": test_file,
        "name": testcase.get("name", ""),
        "kind": child.tag,
        "category": infer_category(test_file, nodeid, message),
        "backend": infer_backend(test_file, nodeid, message),
        "message": message[:1000],
    }


def parse_junit_report(path: Path) -> dict[str, Any]:
    if not path.exists():
        return load_error(path, "FileNotFoundError", "report does not exist")
    try:
        root = ET.parse(path).getroot()
    except (OSError, ET.ParseError) as exc:
        return load_error(path, type(exc).__name__, str(exc))

    failures = []
    for testcase in root.iter("testcase"):
        failure = summarize_testcase(testcase)
        if failure is not None:
            failures.append(failure)
    return {
        "path": display_path(path),
        "tests": junit_int_total(root, "tests"),
        "failures": junit_int_total(root, "failures"),
        "errors": junit_int_total(root, "errors"),
        "skipped": junit_int_total(root, "skipped"),
        "failed_testcases": failures,
    }


def counter_map(counter: Counter[str]) -> dict[str, int]:
    return dict(sorted(counter.items()))


def build_summary(paths: list[Path]) -> dict[str, Any]:
    reports = [parse_junit_report(path) for path in paths]
    failures = [
        failure for report in reports for failure in report.get("failed_testcases", [])
    ]
    by_category = Counter(failure["category"] for failure in failures)
    by_backend = Counter(failure["backend"] for failure in failures)
    load_errors = [report for report in reports if report.get("load_error")]
    loaded_reports = [report for report in reports if not report.get("load_error")]
    return {
        "schema_version": SCHEMA_VERSION,
        "generator": GENERATOR,
        "summary": {
            "report_count": len(reports),
            "load_error_count": len(load_errors),
            "testcase_count": sum(report["tests"] for report in loaded_reports),
            "failure_count": sum(report["failures"] for report in loaded_reports),
            "error_count": sum(report["errors"] for report in loaded_reports),
            "skipped_count": sum(report["skipped"] for report in loaded_reports),
            "failed_testcase_count": len(failures),
            "categories": counter_map(by_category),
            "backends": counter_map(by_backend),
        },
        "reports": reports,
        "failures": failures,
    }


def markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    def cell(value: Any) -> str:
        return str(value).replace("|", "\\|").replace("\n", " ").strip()

    lines = [
        "| " + " | ".join(cell(header) for header in headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(cell(value) for value in row) + " |")
    return "\n".join(lines)


def render_markdown(summary: dict[str, Any], sample_limit: int = 20) -> str:
    rows = [
        ["Reports", summary["summary"]["report_count"]],
        ["Load errors", summary["summary"]["load_error_count"]],
        ["Failed testcases", summary["summary"]["failed_testcase_count"]],
    ]
    lines = ["# Pytest Failure Summary", "", markdown_table(["Field", "Value"], rows)]
    if summary["summary"]["categories"]:
        lines.extend(
            [
                "",
                "## Categories",
                "",
                markdown_table(
                    ["Category", "Failures"],
                    [
                        [category, count]
                        for category, count in summary["summary"]["categories"].items()
                    ],
                ),
            ]
        )
    if summary["summary"]["backends"]:
        lines.extend(
            [
                "",
                "## Backends",
                "",
                markdown_table(
                    ["Backend", "Failures"],
                    [
                        [backend, count]
                        for backend, count in summary["summary"]["backends"].items()
                    ],
                ),
            ]
        )
    if summary["failures"]:
        lines.extend(["", "## Failure Samples", ""])
        for failure in summary["failures"][:sample_limit]:
            lines.append(
                "- `{}`: {} / {}".format(
                    failure["nodeid"],
                    failure["category"],
                    failure["backend"],
                )
            )
        omitted = len(summary["failures"]) - sample_limit
        if omitted > 0:
            lines.append("- Additional failures omitted: {}".format(omitted))
    return "\n".join(lines).rstrip() + "\n"


def write_output(path: Path | None, text: str) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    print("Wrote {}".format(display_path(path)))


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("junit_xml", nargs="+", type=Path)
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--markdown-output", type=Path)
    parser.add_argument("--sample-limit", type=int, default=20)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    summary = build_summary(args.junit_xml)
    json_text = stable_json(summary)
    markdown_text = render_markdown(summary, sample_limit=args.sample_limit)
    if args.json_output is None and args.markdown_output is None:
        print(markdown_text, end="")
    write_output(args.json_output, json_text)
    write_output(args.markdown_output, markdown_text)
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Report and validate CI workflow coverage.

This tool is intentionally narrow: it checks the workflow dimensions that are
easy to accidentally shrink while backend support work is active. The support
catalog remains the source of truth for backend identities and test paths.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import sys
from typing import Any

DEFAULT_ROOT = Path(__file__).resolve().parents[1]
ROOT = Path(os.environ.get("CROSSGL_CI_COVERAGE_ROOT", DEFAULT_ROOT)).resolve()
WORKFLOW_DIR = ROOT / ".github" / "workflows"
BACKENDS_PATH = ROOT / "support" / "backends.json"

PYTHON_VERSIONS = ["3.8", "3.9", "3.10", "3.11", "3.12", "3.13"]
RUNNER_OSES = ["ubuntu-latest", "windows-latest", "macOS-latest"]
EXAMPLES_PYTHON_VERSIONS = PYTHON_VERSIONS
TRANSLATOR_GENERAL_FRONTEND_TEST_COMMAND = (
    "pytest tests/test_translator --ignore=tests/test_translator/test_codegen"
)
FULL_SUITE_REQUIRED_TOOLS = [
    "glslangValidator",
    "spirv-as",
    "spirv-val",
    "dxc",
    "slangc",
    "nvcc",
]
FULL_SUITE_REQUIRED_MARKERS = [
    "compiler-smoke-linux:",
    "Compiler Smoke (Linux CUDA/DXC/SPIR-V/Slang)",
    "runs-on: ubuntu-24.04",
    "Jimver/cuda-toolkit@v0.2.35",
    'CUDA_VERSION: "13.2.0"',
    "SLANG_VERSION: v2026.9.1",
    "compiler-smoke-macos:",
    "Compiler Smoke (macOS Metal)",
    "runs-on: macOS-latest",
    "xcrun -sdk macosx -f metal",
    "sdk.lunarg.com/sdk/download/$vulkanSdkVersion/windows",
    "--accept-licenses --default-answer --confirm-command install copy_only=1",
]
SUPPORT_ISSUE_SYNC_REQUIRED_TESTS = [
    "tests/test_support_matrix.py",
    "tests/test_support_signals.py",
    "tests/test_support_issue_sync.py",
    "tests/test_pr_issue_links.py",
    "tests/test_ci_workflows.py",
    "tests/test_examples_test_script.py",
    "tests/test_tool_cli.py",
]
SUPPORT_ISSUE_SYNC_REQUIRED_PATH_FILTERS = [
    ".github/workflows/backend-tests.yml",
    ".github/workflows/docs.yml",
    ".github/workflows/examples-test.yml",
    ".github/workflows/full-tests.yml",
    ".github/workflows/support-matrix.yml",
    ".github/workflows/support-issue-sync.yml",
    ".github/workflows/translator-tests.yml",
    ".github/workflows/pr-issue-links.yml",
    "crosstl/backend/**",
    "crosstl/translator/ast.py",
    "crosstl/translator/codegen/**",
    "crosstl/translator/lexer.py",
    "crosstl/translator/parser.py",
    "crosstl/translator/validation.py",
    "docs/source/support-matrix.rst",
    "examples/test.py",
    "support/**",
    "tools/ci_coverage.py",
    "tools/support_matrix.py",
    "tools/support_signals.py",
    "tools/sync_support_issues.py",
    "tools/sync_pr_issue_links.py",
    "tests/test_backend/**",
    "tests/test_examples_test_script.py",
    "tests/test_translator/test_ast_ir_contracts.py",
    "tests/test_translator/test_backend_contract.py",
    "tests/test_translator/test_codegen/**",
    "tests/test_translator/test_frontend_*.py",
    "tests/test_translator/test_ir_legacy_alias_contracts.py",
    "tests/test_translator/test_lexer.py",
    "tests/test_translator/test_parser.py",
    "tests/test_translator/test_shader_validation.py",
    "tests/test_translator/test_translation_pipeline.py",
    "tests/test_ci_workflows.py",
    "tests/test_support_matrix.py",
    "tests/test_support_signals.py",
    "tests/test_support_issue_sync.py",
    "tests/test_pr_issue_links.py",
    "tests/test_tool_cli.py",
]
SUPPORT_MATRIX_REQUIRED_POLICIES = {
    "push_on_main": "push:",
    "pull_request_on_main": "pull_request:",
    "daily_schedule": 'cron: "17 3 * * *"',
    "workflow_dispatch": "workflow_dispatch:",
    "matrix_check": "python tools/support_matrix.py check",
    "docs_probe_job": "docs-probe:",
    "docs_probe_on_schedule": "github.event_name == 'schedule'",
    "docs_probe_on_dispatch": "github.event_name == 'workflow_dispatch'",
    "docs_probe_command": (
        "python tools/support_matrix.py docs --output "
        "support/generated/backend-docs-report.json"
    ),
    "docs_probe_artifact": "backend-docs-report",
}
PR_ISSUE_LINK_REQUIRED_POLICIES = {
    "pull_request_target": "pull_request_target:",
    "issues_write": "issues: write",
    "pull_requests_write": "pull-requests: write",
    "trusted_base_checkout": "Checkout trusted base",
    "sync_command": "python tools/sync_pr_issue_links.py",
    "support_traceability": "--check-support-traceability",
    "support_closure_sync": "--sync-support-closures",
}
DOCS_REQUIRED_POLICIES = {
    "push_on_main": "push:",
    "pull_request_on_main": "pull_request:",
    "workflow_dispatch": "workflow_dispatch:",
    "python_312": 'python-version: "3.12"',
    "install_doxygen": "sudo apt-get update && sudo apt-get install -y doxygen",
    "install_docs_requirements": "pip install -r docs/requirements.txt",
    "build_doxygen": "make -C docs doxygen",
    "build_sphinx_html": "make -C docs html",
}
EXAMPLES_REQUIRED_POLICIES = {
    "push_on_main": "push:",
    "pull_request_on_main": "pull_request:",
    "workflow_dispatch": "workflow_dispatch:",
    "comprehensive_test_script": "python test.py",
    "comprehensive_summary_json": "--summary-json",
    "comprehensive_regression_budget": 'summary["within_regression_budget"]',
    "backend_specific_job": "backend-specific:",
    "stability_test_job": "stability-test:",
    "fails_on_missing_output": 'echo "[ERROR] Output file not created: $OUTPUT_FILE"',
    "fails_on_small_output": (
        'echo "[ERROR] Output file is too small ($FILE_SIZE bytes)"'
    ),
}


class CiCoverageError(RuntimeError):
    """Raised when workflow coverage cannot be inspected."""


def configure_root(root: Path) -> None:
    global ROOT, WORKFLOW_DIR, BACKENDS_PATH
    ROOT = root.resolve()
    WORKFLOW_DIR = ROOT / ".github" / "workflows"
    BACKENDS_PATH = ROOT / "support" / "backends.json"


def stable_json(data: Any) -> str:
    return json.dumps(data, indent=2, sort_keys=True) + "\n"


def markdown_escape(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def relpath(path: Path) -> str:
    return os.path.relpath(str(path), str(ROOT)).replace(os.sep, "/")


def display_path(path: Path) -> str:
    try:
        return relpath(path)
    except ValueError:
        return str(path)


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def leading_spaces(line: str) -> int:
    return len(line) - len(line.lstrip(" "))


def nested_yaml_section(workflow: str, name: str, indent: int) -> list[str]:
    lines = workflow.splitlines()
    section_header = "{}{}:".format(" " * indent, name)
    for index, line in enumerate(lines):
        if line.rstrip() != section_header:
            continue
        section = []
        for section_line in lines[index + 1 :]:
            if section_line.strip() and leading_spaces(section_line) <= indent:
                break
            section.append(section_line)
        return section
    return []


def strip_yaml_scalar(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def yaml_list_values(section: list[str], key: str) -> list[str]:
    for index, line in enumerate(section):
        if line.strip() != "{}:".format(key):
            continue
        indent = leading_spaces(line)
        values = []
        for item_line in section[index + 1 :]:
            if item_line.strip() and leading_spaces(item_line) <= indent:
                break
            item = re.match(r"^\s*-\s+(.+?)\s*$", item_line)
            if item:
                values.append(strip_yaml_scalar(item.group(1)))
        return values
    return []


def pull_request_path_filters(workflow: str) -> list[str]:
    pull_request = nested_yaml_section(workflow, "pull_request", 2)
    return yaml_list_values(pull_request, "paths")


def load_backends() -> list[dict[str, Any]]:
    return json.loads(read_text(BACKENDS_PATH))["backends"]


def workflow_text(name: str) -> str:
    path = WORKFLOW_DIR / name
    if not path.exists():
        raise CiCoverageError("Workflow does not exist: {}".format(relpath(path)))
    return read_text(path)


def parse_matrix_values(raw: str) -> list[str]:
    raw = raw.strip().strip("[]")
    values = []
    for item in raw.split(","):
        value = item.strip().strip("\"'")
        if value:
            values.append(value)
    return values


def matrix_values(workflow: str, key: str) -> list[str]:
    inline = re.search(
        r"^\s*{}:\s*(\[[^\n]+\])\s*$".format(re.escape(key)),
        workflow,
        flags=re.MULTILINE,
    )
    if inline:
        return parse_matrix_values(inline.group(1))

    block = re.search(
        r"^\s*{}:\s*\n\s*\[\s*(.*?)\s*\]".format(re.escape(key)),
        workflow,
        flags=re.MULTILINE | re.DOTALL,
    )
    if block:
        return parse_matrix_values(block.group(1))
    raise CiCoverageError("Matrix key not found: {}".format(key))


def backend_test_matrix_name(backend: dict[str, Any]) -> str:
    prefix = "tests/test_backend/test_"
    for path in backend.get("tests", []):
        if path.startswith(prefix):
            return path[len(prefix) :].split("/", 1)[0]
    raise CiCoverageError("Backend '{}' has no backend test path".format(backend["id"]))


def translator_test_matrix_name(backend: dict[str, Any]) -> str:
    prefix = "tests/test_translator/test_codegen/test_"
    suffix = "_codegen.py"
    for path in backend.get("tests", []):
        if path.startswith(prefix) and path.endswith(suffix):
            return path[len(prefix) : -len(suffix)]
    raise CiCoverageError(
        "Backend '{}' has no translator codegen test path".format(backend["id"])
    )


def compare_sets(actual: list[str], expected: list[str]) -> dict[str, list[str]]:
    actual_set = set(actual)
    expected_set = set(expected)
    return {
        "actual": sorted(actual_set),
        "expected": sorted(expected_set),
        "missing": sorted(expected_set - actual_set),
        "extra": sorted(actual_set - expected_set),
    }


def dimension_ok(dimension: dict[str, list[str]]) -> bool:
    return not dimension["missing"] and not dimension["extra"]


def workflow_matrix_report(
    workflow_name: str,
    workflow: str,
    matrix_key: str,
    expected_components: list[str],
) -> dict[str, Any]:
    return {
        "workflow": workflow_name,
        "component_key": matrix_key,
        "components": compare_sets(
            matrix_values(workflow, matrix_key), expected_components
        ),
        "python_versions": compare_sets(
            matrix_values(workflow, "python-version"), PYTHON_VERSIONS
        ),
        "oses": compare_sets(matrix_values(workflow, "OS"), RUNNER_OSES),
        "fail_fast_false": "fail-fast: false" in workflow,
    }


def translator_workflow_report(
    workflow_name: str,
    workflow: str,
    matrix_key: str,
    expected_components: list[str],
) -> dict[str, Any]:
    report = workflow_matrix_report(
        workflow_name,
        workflow,
        matrix_key,
        expected_components,
    )
    report["general_frontend_suite"] = (
        TRANSLATOR_GENERAL_FRONTEND_TEST_COMMAND in workflow
    )
    return report


def full_suite_report(workflow: str) -> dict[str, Any]:
    required_markers = {
        marker: marker in workflow for marker in FULL_SUITE_REQUIRED_MARKERS
    }
    required_tools = {tool: tool in workflow for tool in FULL_SUITE_REQUIRED_TOOLS}
    return {
        "workflow": "full-tests.yml",
        "pytest_all_tests": (
            re.search(r"python\s+-m\s+pytest\s+tests\b", workflow) is not None
        ),
        "shader_validator_oses": compare_sets(
            matrix_values(workflow, "os"),
            RUNNER_OSES,
        ),
        "required_markers": required_markers,
        "required_tools": required_tools,
    }


def support_issue_sync_report(workflow: str) -> dict[str, Any]:
    path_filters = pull_request_path_filters(workflow)
    return {
        "workflow": "support-issue-sync.yml",
        "hourly_schedule": 'cron: "17 * * * *"' in workflow,
        "dry_run_on_pull_request": (
            "github.event_name == 'pull_request'" in workflow
            and "--dry-run" in workflow
        ),
        "mutates_outside_pull_request": (
            "github.event_name != 'pull_request'" in workflow
        ),
        "required_tests": {
            test: test in workflow for test in SUPPORT_ISSUE_SYNC_REQUIRED_TESTS
        },
        "required_path_filters": {
            path: path in path_filters
            for path in SUPPORT_ISSUE_SYNC_REQUIRED_PATH_FILTERS
        },
        "min_desired_issues": "--min-desired-issues 10" in workflow,
    }


def support_matrix_report(workflow: str) -> dict[str, Any]:
    return {
        "workflow": "support-matrix.yml",
        "required_policies": {
            name: marker in workflow
            for name, marker in SUPPORT_MATRIX_REQUIRED_POLICIES.items()
        },
        "uploads_docs_probe_artifact": (
            "actions/upload-artifact@v4" in workflow
            and "support/generated/backend-docs-report.json" in workflow
        ),
    }


def pr_issue_links_report(workflow: str) -> dict[str, Any]:
    return {
        "workflow": "pr-issue-links.yml",
        "required_policies": (
            {
                name: marker in workflow
                for name, marker in PR_ISSUE_LINK_REQUIRED_POLICIES.items()
            }
            | {
                "support_traceability_not_enforced": (
                    "--enforce-support-traceability" not in workflow
                ),
            }
        ),
    }


def docs_report(workflow: str) -> dict[str, Any]:
    return {
        "workflow": "docs.yml",
        "required_policies": {
            name: marker in workflow for name, marker in DOCS_REQUIRED_POLICIES.items()
        },
    }


def workflow_job_text(workflow: str, job_name: str) -> str:
    return "\n".join(nested_yaml_section(workflow, job_name, 2))


def example_backend_values(workflow: str) -> list[str]:
    return re.findall(r"\bbackend:\s*\"?([A-Za-z0-9_-]+)\"?", workflow)


def examples_report(workflow: str, expected_backends: list[str]) -> dict[str, Any]:
    backend_specific = workflow_job_text(workflow, "backend-specific")
    stability = workflow_job_text(workflow, "stability-test")
    return {
        "workflow": "examples-test.yml",
        "python_versions": compare_sets(
            matrix_values(workflow, "python-version"),
            EXAMPLES_PYTHON_VERSIONS,
        ),
        "oses": compare_sets(matrix_values(workflow, "os"), RUNNER_OSES),
        "backend_coverage": compare_sets(
            example_backend_values(workflow),
            expected_backends,
        ),
        "required_policies": {
            name: marker in workflow
            for name, marker in EXAMPLES_REQUIRED_POLICIES.items()
        },
        "backend_specific_strict": "continue-on-error: true" not in backend_specific,
        "stability_fails_on_regression": stability.count("raise SystemExit(1)") >= 2,
        "diagnostic_continue_on_error_count": workflow.count("continue-on-error: true"),
    }


def build_report() -> dict[str, Any]:
    backends = load_backends()
    backend_ids = sorted(backend["id"] for backend in backends)
    expected_backend_components = [
        backend_test_matrix_name(backend) for backend in backends
    ]
    expected_translator_components = [
        translator_test_matrix_name(backend) for backend in backends
    ] + ["general"]

    backend_workflow = workflow_text("backend-tests.yml")
    translator_workflow = workflow_text("translator-tests.yml")
    docs_workflow = workflow_text("docs.yml")
    examples_workflow = workflow_text("examples-test.yml")
    full_workflow = workflow_text("full-tests.yml")
    support_matrix_workflow = workflow_text("support-matrix.yml")
    support_issue_workflow = workflow_text("support-issue-sync.yml")
    pr_issue_links_workflow = workflow_text("pr-issue-links.yml")

    report = {
        "schema_version": 1,
        "generator": "tools/ci_coverage.py",
        "catalog": {
            "backend_count": len(backends),
            "backend_ids": backend_ids,
        },
        "workflows": {
            "backend_tests": workflow_matrix_report(
                "backend-tests.yml",
                backend_workflow,
                "backend",
                expected_backend_components,
            ),
            "translator_tests": translator_workflow_report(
                "translator-tests.yml",
                translator_workflow,
                "component",
                expected_translator_components,
            ),
            "docs": docs_report(docs_workflow),
            "examples": examples_report(examples_workflow, backend_ids),
            "full_tests": full_suite_report(full_workflow),
            "support_matrix": support_matrix_report(support_matrix_workflow),
            "support_issue_sync": support_issue_sync_report(support_issue_workflow),
            "pr_issue_links": pr_issue_links_report(pr_issue_links_workflow),
        },
    }
    report["summary"] = {
        "ok": not validation_errors(report),
        "errors": len(validation_errors(report)),
    }
    return report


def load_report(path: Path) -> dict[str, Any]:
    return json.loads(read_text(path))


def validation_errors(report: dict[str, Any]) -> list[str]:
    errors = []
    for key in ("backend_tests", "translator_tests"):
        workflow = report["workflows"][key]
        for dimension in ("components", "python_versions", "oses"):
            if not dimension_ok(workflow[dimension]):
                errors.append(
                    "{} {} mismatch: missing={}, extra={}".format(
                        workflow["workflow"],
                        dimension,
                        workflow[dimension]["missing"],
                        workflow[dimension]["extra"],
                    )
                )
        if not workflow["fail_fast_false"]:
            errors.append("{} must keep fail-fast: false".format(workflow["workflow"]))

    translator_tests = report["workflows"]["translator_tests"]
    if not translator_tests["general_frontend_suite"]:
        errors.append("translator-tests.yml must run the frontend general suite")

    docs = report["workflows"]["docs"]
    for policy, present in docs["required_policies"].items():
        if not present:
            errors.append("docs.yml missing policy: {}".format(policy))

    examples = report["workflows"]["examples"]
    for dimension in ("python_versions", "oses", "backend_coverage"):
        if not dimension_ok(examples[dimension]):
            errors.append(
                "examples-test.yml {} mismatch: missing={}, extra={}".format(
                    dimension,
                    examples[dimension]["missing"],
                    examples[dimension]["extra"],
                )
            )
    for policy, present in examples["required_policies"].items():
        if not present:
            errors.append("examples-test.yml missing policy: {}".format(policy))
    if not examples["backend_specific_strict"]:
        errors.append("examples-test.yml backend-specific job must fail on errors")
    if not examples["stability_fails_on_regression"]:
        errors.append("examples-test.yml stability job must fail on regression")
    if examples["diagnostic_continue_on_error_count"] > 1:
        errors.append(
            "examples-test.yml has too many continue-on-error steps: {}".format(
                examples["diagnostic_continue_on_error_count"]
            )
        )

    full_tests = report["workflows"]["full_tests"]
    if not full_tests["pytest_all_tests"]:
        errors.append("full-tests.yml must run python -m pytest tests")
    if not dimension_ok(full_tests["shader_validator_oses"]):
        errors.append(
            "full-tests.yml shader validator OS mismatch: missing={}, extra={}".format(
                full_tests["shader_validator_oses"]["missing"],
                full_tests["shader_validator_oses"]["extra"],
            )
        )
    for marker, present in full_tests["required_markers"].items():
        if not present:
            errors.append("full-tests.yml missing marker: {}".format(marker))
    for tool, present in full_tests["required_tools"].items():
        if not present:
            errors.append(
                "full-tests.yml missing compiler tool coverage: {}".format(tool)
            )

    support_matrix = report["workflows"]["support_matrix"]
    for policy, present in support_matrix["required_policies"].items():
        if not present:
            errors.append("support-matrix.yml missing policy: {}".format(policy))
    if not support_matrix["uploads_docs_probe_artifact"]:
        errors.append("support-matrix.yml missing docs probe artifact upload")

    pr_issue_links = report["workflows"]["pr_issue_links"]
    for policy, present in pr_issue_links["required_policies"].items():
        if not present:
            errors.append("pr-issue-links.yml missing policy: {}".format(policy))

    support_sync = report["workflows"]["support_issue_sync"]
    for field in (
        "hourly_schedule",
        "dry_run_on_pull_request",
        "mutates_outside_pull_request",
        "min_desired_issues",
    ):
        if not support_sync[field]:
            errors.append("support-issue-sync.yml missing {}".format(field))
    for test, present in support_sync["required_tests"].items():
        if not present:
            errors.append(
                "support-issue-sync.yml missing planner test: {}".format(test)
            )
    for path_filter, present in support_sync["required_path_filters"].items():
        if not present:
            errors.append(
                "support-issue-sync.yml missing path filter: {}".format(path_filter)
            )

    return errors


def compare_value_sets(
    baseline_values: list[str], current_values: list[str]
) -> dict[str, list[str]]:
    baseline_set = set(baseline_values)
    current_set = set(current_values)
    return {
        "baseline": sorted(baseline_set),
        "current": sorted(current_set),
        "removed": sorted(baseline_set - current_set),
        "added": sorted(current_set - baseline_set),
    }


def compare_bool_maps(
    baseline_values: dict[str, bool], current_values: dict[str, bool]
) -> dict[str, list[str]]:
    keys = set(baseline_values) | set(current_values)
    removed = []
    added = []
    for key in keys:
        baseline = bool(baseline_values.get(key))
        current = bool(current_values.get(key))
        if baseline and not current:
            removed.append(key)
        elif current and not baseline:
            added.append(key)
    return {
        "removed": sorted(removed),
        "added": sorted(added),
    }


def compare_bool_value(
    scope: str, name: str, baseline: bool, current: bool
) -> dict[str, Any] | None:
    if baseline == current:
        return None
    return {
        "scope": scope,
        "dimension": name,
        "removed": [name] if baseline and not current else [],
        "added": [name] if current and not baseline else [],
    }


def build_ci_coverage_comparison(
    baseline: dict[str, Any], current: dict[str, Any]
) -> dict[str, Any]:
    shrinks = []
    growth = []

    def add_set_change(
        scope: str,
        dimension: str,
        baseline_values: list[str],
        current_values: list[str],
    ) -> None:
        change = compare_value_sets(baseline_values, current_values)
        payload = {
            "scope": scope,
            "dimension": dimension,
            "removed": change["removed"],
            "added": change["added"],
        }
        if change["removed"]:
            shrinks.append(payload)
        if change["added"]:
            growth.append(payload)

    def add_bool_map_change(
        scope: str,
        dimension: str,
        baseline_values: dict[str, bool],
        current_values: dict[str, bool],
    ) -> None:
        change = compare_bool_maps(baseline_values, current_values)
        payload = {
            "scope": scope,
            "dimension": dimension,
            "removed": change["removed"],
            "added": change["added"],
        }
        if change["removed"]:
            shrinks.append(payload)
        if change["added"]:
            growth.append(payload)

    def add_bool_change(
        scope: str, dimension: str, baseline_value: bool, current_value: bool
    ) -> None:
        change = compare_bool_value(scope, dimension, baseline_value, current_value)
        if change is None:
            return
        if change["removed"]:
            shrinks.append(change)
        if change["added"]:
            growth.append(change)

    for workflow_key in ("backend_tests", "translator_tests"):
        baseline_workflow = baseline["workflows"][workflow_key]
        current_workflow = current["workflows"][workflow_key]
        scope = current_workflow["workflow"]
        for dimension in ("components", "python_versions", "oses"):
            add_set_change(
                scope,
                dimension,
                baseline_workflow[dimension]["actual"],
                current_workflow[dimension]["actual"],
            )
        add_bool_change(
            scope,
            "fail_fast_false",
            baseline_workflow["fail_fast_false"],
            current_workflow["fail_fast_false"],
        )
        if workflow_key == "translator_tests":
            add_bool_change(
                scope,
                "general_frontend_suite",
                baseline_workflow["general_frontend_suite"],
                current_workflow["general_frontend_suite"],
            )

    baseline_full = baseline["workflows"]["full_tests"]
    current_full = current["workflows"]["full_tests"]
    baseline_docs = baseline["workflows"]["docs"]
    current_docs = current["workflows"]["docs"]
    add_bool_map_change(
        "docs.yml",
        "required_policies",
        baseline_docs["required_policies"],
        current_docs["required_policies"],
    )

    baseline_examples = baseline["workflows"]["examples"]
    current_examples = current["workflows"]["examples"]
    for dimension in ("python_versions", "oses", "backend_coverage"):
        add_set_change(
            "examples-test.yml",
            dimension,
            baseline_examples[dimension]["actual"],
            current_examples[dimension]["actual"],
        )
    add_bool_map_change(
        "examples-test.yml",
        "required_policies",
        baseline_examples["required_policies"],
        current_examples["required_policies"],
    )
    for dimension in (
        "backend_specific_strict",
        "stability_fails_on_regression",
    ):
        add_bool_change(
            "examples-test.yml",
            dimension,
            baseline_examples[dimension],
            current_examples[dimension],
        )

    add_bool_change(
        "full-tests.yml",
        "pytest_all_tests",
        baseline_full["pytest_all_tests"],
        current_full["pytest_all_tests"],
    )
    add_set_change(
        "full-tests.yml",
        "shader_validator_oses",
        baseline_full["shader_validator_oses"]["actual"],
        current_full["shader_validator_oses"]["actual"],
    )
    add_bool_map_change(
        "full-tests.yml",
        "required_markers",
        baseline_full["required_markers"],
        current_full["required_markers"],
    )
    add_bool_map_change(
        "full-tests.yml",
        "required_tools",
        baseline_full["required_tools"],
        current_full["required_tools"],
    )

    baseline_matrix = baseline["workflows"]["support_matrix"]
    current_matrix = current["workflows"]["support_matrix"]
    add_bool_map_change(
        "support-matrix.yml",
        "required_policies",
        baseline_matrix["required_policies"],
        current_matrix["required_policies"],
    )
    add_bool_change(
        "support-matrix.yml",
        "uploads_docs_probe_artifact",
        baseline_matrix["uploads_docs_probe_artifact"],
        current_matrix["uploads_docs_probe_artifact"],
    )

    baseline_pr_links = baseline["workflows"]["pr_issue_links"]
    current_pr_links = current["workflows"]["pr_issue_links"]
    add_bool_map_change(
        "pr-issue-links.yml",
        "required_policies",
        baseline_pr_links["required_policies"],
        current_pr_links["required_policies"],
    )

    baseline_support = baseline["workflows"]["support_issue_sync"]
    current_support = current["workflows"]["support_issue_sync"]
    for dimension in (
        "hourly_schedule",
        "dry_run_on_pull_request",
        "mutates_outside_pull_request",
        "min_desired_issues",
    ):
        add_bool_change(
            "support-issue-sync.yml",
            dimension,
            baseline_support[dimension],
            current_support[dimension],
        )
    add_bool_map_change(
        "support-issue-sync.yml",
        "required_tests",
        baseline_support["required_tests"],
        current_support["required_tests"],
    )
    add_bool_map_change(
        "support-issue-sync.yml",
        "required_path_filters",
        baseline_support["required_path_filters"],
        current_support["required_path_filters"],
    )

    return {
        "schema_version": 1,
        "generator": "tools/ci_coverage.py compare",
        "summary": {
            "ok": not shrinks,
            "shrink_count": len(shrinks),
            "growth_count": len(growth),
        },
        "shrinks": shrinks,
        "growth": growth,
    }


def write_report(report: dict[str, Any], output: Path | None) -> None:
    text = stable_json(report)
    if output is None:
        print(text, end="")
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8")
    print("Wrote {}".format(display_path(output)))


def ok_text(value: bool) -> str:
    return "yes" if value else "no"


def dimension_summary(dimension: dict[str, list[str]]) -> str:
    if dimension_ok(dimension):
        return "{} / {}".format(len(dimension["actual"]), len(dimension["expected"]))
    details = []
    if dimension["missing"]:
        details.append("missing {}".format(", ".join(dimension["missing"])))
    if dimension["extra"]:
        details.append("extra {}".format(", ".join(dimension["extra"])))
    return "; ".join(details)


def markdown_table(headers: list[str], rows: list[list[Any]]) -> list[str]:
    lines = [
        "| " + " | ".join(markdown_escape(header) for header in headers) + " |",
        "| " + " | ".join("---" for _header in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(markdown_escape(cell) for cell in row) + " |")
    return lines


def render_markdown(report: dict[str, Any]) -> str:
    errors = validation_errors(report)
    backend_tests = report["workflows"]["backend_tests"]
    translator_tests = report["workflows"]["translator_tests"]
    docs = report["workflows"]["docs"]
    examples = report["workflows"]["examples"]
    full_tests = report["workflows"]["full_tests"]
    support_matrix = report["workflows"]["support_matrix"]
    pr_issue_links = report["workflows"]["pr_issue_links"]
    support_sync = report["workflows"]["support_issue_sync"]

    lines = [
        "# CI Coverage Report",
        "",
        "Generated by `tools/ci_coverage.py`.",
        "",
        "Status: **{}**".format("pass" if not errors else "fail"),
        "",
    ]
    lines.extend(
        markdown_table(
            ["Catalog", "Value"],
            [
                ["Backends", report["catalog"]["backend_count"]],
                ["Backend IDs", ", ".join(report["catalog"]["backend_ids"])],
            ],
        )
    )
    lines.extend(
        [
            "",
            "## Workflow Matrices",
            "",
        ]
    )
    lines.extend(
        markdown_table(
            [
                "Workflow",
                "Components",
                "Python",
                "OS",
                "Fail-fast disabled",
                "Frontend suite",
            ],
            [
                [
                    backend_tests["workflow"],
                    dimension_summary(backend_tests["components"]),
                    dimension_summary(backend_tests["python_versions"]),
                    dimension_summary(backend_tests["oses"]),
                    ok_text(backend_tests["fail_fast_false"]),
                    "n/a",
                ],
                [
                    translator_tests["workflow"],
                    dimension_summary(translator_tests["components"]),
                    dimension_summary(translator_tests["python_versions"]),
                    dimension_summary(translator_tests["oses"]),
                    ok_text(translator_tests["fail_fast_false"]),
                    ok_text(translator_tests["general_frontend_suite"]),
                ],
            ],
        )
    )
    lines.extend(
        [
            "",
            "## Documentation",
            "",
        ]
    )
    lines.extend(
        markdown_table(
            ["Check", "Status"],
            [
                [
                    "Required policies",
                    ok_text(all(docs["required_policies"].values())),
                ],
            ],
        )
    )
    lines.extend(
        [
            "",
            "## Examples",
            "",
        ]
    )
    lines.extend(
        markdown_table(
            ["Check", "Status"],
            [
                [
                    "Python matrix",
                    dimension_summary(examples["python_versions"]),
                ],
                ["OS matrix", dimension_summary(examples["oses"])],
                [
                    "Backend coverage",
                    dimension_summary(examples["backend_coverage"]),
                ],
                [
                    "Required policies",
                    ok_text(all(examples["required_policies"].values())),
                ],
                [
                    "Backend-specific failures are fatal",
                    ok_text(examples["backend_specific_strict"]),
                ],
                [
                    "Stability failures are fatal",
                    ok_text(examples["stability_fails_on_regression"]),
                ],
                [
                    "Diagnostic continue-on-error steps",
                    examples["diagnostic_continue_on_error_count"],
                ],
            ],
        )
    )
    lines.extend(
        [
            "",
            "## Full Test Suite",
            "",
        ]
    )
    lines.extend(
        markdown_table(
            ["Check", "Status"],
            [
                ["Runs all pytest tests", ok_text(full_tests["pytest_all_tests"])],
                [
                    "Shader validator OS matrix",
                    dimension_summary(full_tests["shader_validator_oses"]),
                ],
                [
                    "Compiler smoke markers",
                    ok_text(all(full_tests["required_markers"].values())),
                ],
                [
                    "Compiler tools",
                    ok_text(all(full_tests["required_tools"].values())),
                ],
            ],
        )
    )
    lines.extend(
        [
            "",
            "## Support Matrix",
            "",
        ]
    )
    lines.extend(
        markdown_table(
            ["Check", "Status"],
            [
                [
                    "Required policies",
                    ok_text(all(support_matrix["required_policies"].values())),
                ],
                [
                    "Documentation probe artifact",
                    ok_text(support_matrix["uploads_docs_probe_artifact"]),
                ],
            ],
        )
    )
    lines.extend(
        [
            "",
            "## PR Issue Links",
            "",
        ]
    )
    lines.extend(
        markdown_table(
            ["Check", "Status"],
            [
                [
                    "Required policies",
                    ok_text(all(pr_issue_links["required_policies"].values())),
                ],
            ],
        )
    )
    lines.extend(
        [
            "",
            "## Support Issue Sync",
            "",
        ]
    )
    lines.extend(
        markdown_table(
            ["Check", "Status"],
            [
                ["Hourly schedule", ok_text(support_sync["hourly_schedule"])],
                [
                    "PR dry-run mode",
                    ok_text(support_sync["dry_run_on_pull_request"]),
                ],
                [
                    "Issue mutation outside PRs",
                    ok_text(support_sync["mutates_outside_pull_request"]),
                ],
                ["Minimum desired issues", ok_text(support_sync["min_desired_issues"])],
                [
                    "Planner tests",
                    ok_text(all(support_sync["required_tests"].values())),
                ],
                [
                    "PR path filters",
                    ok_text(all(support_sync["required_path_filters"].values())),
                ],
            ],
        )
    )
    if errors:
        lines.extend(["", "## Errors", ""])
        lines.extend("- {}".format(error) for error in errors)
    return "\n".join(lines).rstrip() + "\n"


def write_markdown(report: dict[str, Any], output: Path | None) -> None:
    text = render_markdown(report)
    if output is None:
        print(text, end="")
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8")
    print("Wrote {}".format(display_path(output)))


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="Repository root to inspect. Defaults to this script's repository.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("report", "summary", "check"):
        subparser = subparsers.add_parser(command)
        subparser.add_argument("--output", type=Path, help="Optional output path")
    compare_parser = subparsers.add_parser("compare")
    compare_parser.add_argument("--baseline", type=Path, required=True)
    compare_parser.add_argument(
        "--current",
        type=Path,
        help="Current report JSON path. Defaults to the live workflow report.",
    )
    compare_parser.add_argument("--output", type=Path, help="Optional JSON output path")
    compare_parser.add_argument(
        "--fail-on-shrink",
        action="store_true",
        help="Exit non-zero when coverage was removed compared with the baseline",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    if args.root is not None:
        configure_root(args.root)
    try:
        if args.command == "compare":
            baseline_path = (
                args.baseline if args.baseline.is_absolute() else ROOT / args.baseline
            )
            current_report = (
                load_report(
                    args.current if args.current.is_absolute() else ROOT / args.current
                )
                if args.current
                else build_report()
            )
            comparison = build_ci_coverage_comparison(
                load_report(baseline_path),
                current_report,
            )
            if args.output is not None:
                output = (
                    args.output if args.output.is_absolute() else ROOT / args.output
                )
                write_report(comparison, output)
            else:
                write_report(comparison, None)
            if args.fail_on_shrink and comparison["summary"]["shrink_count"]:
                print("CI coverage comparison found removed coverage.", file=sys.stderr)
                return 1
            return 0

        report = build_report()
    except CiCoverageError as exc:
        print("CI coverage error: {}".format(exc), file=sys.stderr)
        return 2

    errors = validation_errors(report)
    if args.command == "summary":
        output = (
            args.output
            if args.output is None or args.output.is_absolute()
            else ROOT / args.output
        )
        write_markdown(report, output)
    elif args.output is not None:
        output = args.output if args.output.is_absolute() else ROOT / args.output
        write_report(report, output)
    elif args.command == "report":
        write_report(report, None)

    if args.command == "check":
        if errors:
            print("CI coverage check failed:", file=sys.stderr)
            for error in errors:
                print("- {}".format(error), file=sys.stderr)
            return 1
        print("CI coverage check passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

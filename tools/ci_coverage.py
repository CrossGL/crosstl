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
FULL_SUITE_FAILURE_SUMMARIES = {
    "complete": {
        "job": "pytest",
        "run_step": "Run complete test suite",
        "summary_step": "Write complete test failure summary",
        "upload_step": "Upload complete test failure summary",
        "junit": "support/generated/full-tests-pytest.xml",
        "json": "support/generated/full-tests-failure-summary.json",
        "markdown": "support/generated/full-tests-failure-summary.md",
        "artifact": "full-test-failure-summary",
    },
    "shader_validators": {
        "job": "shader-validators",
        "run_step": "Run external shader validator tests",
        "summary_step": "Write shader validator failure summary",
        "upload_step": "Upload shader validator failure summary",
        "junit": "support/generated/shader-validators-pytest.xml",
        "json": "support/generated/shader-validators-failure-summary.json",
        "markdown": "support/generated/shader-validators-failure-summary.md",
        "artifact": "shader-validator-failure-summary-${{ matrix.os }}",
    },
    "compiler_smoke_linux": {
        "job": "compiler-smoke-linux",
        "run_step": "Run external shader validator tests",
        "summary_step": "Write compiler smoke failure summary",
        "upload_step": "Upload compiler smoke failure summary",
        "junit": "support/generated/compiler-smoke-linux-pytest.xml",
        "json": "support/generated/compiler-smoke-linux-failure-summary.json",
        "markdown": "support/generated/compiler-smoke-linux-failure-summary.md",
        "artifact": "compiler-smoke-linux-failure-summary",
    },
    "compiler_smoke_macos": {
        "job": "compiler-smoke-macos",
        "run_step": "Run Metal compiler smoke tests",
        "summary_step": "Write Metal compiler smoke failure summary",
        "upload_step": "Upload Metal compiler smoke failure summary",
        "junit": "support/generated/compiler-smoke-macos-pytest.xml",
        "json": "support/generated/compiler-smoke-macos-failure-summary.json",
        "markdown": "support/generated/compiler-smoke-macos-failure-summary.md",
        "artifact": "compiler-smoke-macos-failure-summary",
    },
}
SUPPORT_ISSUE_SYNC_REQUIRED_TESTS = [
    "tests/test_support_matrix.py",
    "tests/test_support_signals.py",
    "tests/test_support_ci_summary.py",
    "tests/test_support_issue_sync.py",
    "tests/test_pr_issue_links.py",
    "tests/test_pytest_failure_summary.py",
    "tests/test_ci_workflows.py",
    "tests/test_examples_test_script.py",
    "tests/test_tool_cli.py",
]
SUPPORT_ISSUE_SYNC_REQUIRED_PATH_FILTERS = [
    ".github/workflows/backend-tests.yml",
    ".github/workflows/docs.yml",
    ".github/workflows/examples-test.yml",
    ".github/workflows/full-tests.yml",
    ".github/workflows/issue_assign.yml",
    ".github/workflows/support-matrix.yml",
    ".github/workflows/support-issue-sync.yml",
    ".github/workflows/stale-prs.yml",
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
    "tools/pytest_failure_summary.py",
    "tools/support_ci_summary.py",
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
    "tests/test_support_ci_summary.py",
    "tests/test_support_issue_sync.py",
    "tests/test_pr_issue_links.py",
    "tests/test_pytest_failure_summary.py",
    "tests/test_tool_cli.py",
]
SUPPORT_ISSUE_SYNC_PLANNED_ACTION_BUDGET_ARGS = [
    "--planned-action-budget-mode fail",
    "--max-planned-created 300",
    "--max-planned-updated 300",
    "--max-planned-closed 50",
    "--max-planned-attached 300",
    "--max-planned-total 600",
    "--max-planned-stale-parent-closures 0",
    "--max-planned-stale-backlog-closures 100",
    "--max-planned-stale-extracted-closures 100",
    "--max-planned-duplicate-marker-closures 25",
]
SUPPORT_MATRIX_REQUIRED_POLICIES = {
    "push_on_main": "push:",
    "pull_request_on_main": "pull_request:",
    "daily_schedule": 'cron: "17 3 * * *"',
    "workflow_dispatch": "workflow_dispatch:",
    "matrix_check": "python tools/support_matrix.py check",
    "matrix_check_report": "--output support/generated/support-matrix-check.json",
    "docs_probe_job": "docs-probe:",
    "docs_probe_on_schedule": "github.event_name == 'schedule'",
    "docs_probe_on_dispatch": "github.event_name == 'workflow_dispatch'",
    "docs_probe_command": (
        "python tools/support_matrix.py docs --output "
        "support/generated/backend-docs-report.json"
    ),
    "docs_probe_artifact": "backend-docs-report",
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
WORKFLOW_WRITE_PERMISSION_ALLOWLIST = {
    "issue_assign.yml": ["issues"],
    "pr-issue-links.yml": ["issues", "pull-requests"],
    "stale-prs.yml": ["issues", "pull-requests"],
    "support-issue-sync.yml": ["issues"],
}
MUTABLE_ACTION_REFS = {"head", "latest", "main", "master", "trunk"}
PULL_REQUEST_TARGET_WORKFLOW_ALLOWLIST = {"pr-issue-links.yml"}
PULL_REQUEST_HEAD_MARKERS = (
    "github.event.pull_request.head",
    "github.head_ref",
    "refs/pull/",
)


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


def workflow_step_section(workflow: str, name: str) -> str:
    pattern = r"^\s*-\s+name:\s*{}\s*$".format(re.escape(name))
    match = re.search(pattern, workflow, flags=re.MULTILINE)
    if not match:
        return ""
    next_step = re.search(r"^\s*-\s+name:\s*", workflow[match.end() :], re.MULTILINE)
    end = match.end() + next_step.start() if next_step else len(workflow)
    return workflow[match.start() : end]


def workflow_step_start(workflow: str, name: str) -> int:
    pattern = r"^\s*-\s+name:\s*{}\s*$".format(re.escape(name))
    match = re.search(pattern, workflow, flags=re.MULTILINE)
    return match.start() if match else -1


def workflow_step_after(workflow: str, later_name: str, earlier_name: str) -> bool:
    later_start = workflow_step_start(workflow, later_name)
    earlier_start = workflow_step_start(workflow, earlier_name)
    return later_start >= 0 and earlier_start >= 0 and later_start > earlier_start


def workflow_job_text(workflow: str, job_name: str) -> str:
    return "\n".join(nested_yaml_section(workflow, job_name, 2))


def workflow_job_step_section(workflow: str, job_name: str, step_name: str) -> str:
    return workflow_step_section(workflow_job_text(workflow, job_name), step_name)


def workflow_job_step_after(
    workflow: str, job_name: str, later_name: str, earlier_name: str
) -> bool:
    return workflow_step_after(
        workflow_job_text(workflow, job_name),
        later_name,
        earlier_name,
    )


def workflow_job_names(workflow: str) -> list[str]:
    jobs_section = nested_yaml_section(workflow, "jobs", 0)
    return [
        match.group(1)
        for line in jobs_section
        if (match := re.match(r"^  ([A-Za-z0-9_-]+):\s*$", line))
    ]


def workflow_job_timeout_minutes(workflow: str, job_name: str) -> int | None:
    job_text = workflow_job_text(workflow, job_name)
    match = re.search(r"^    timeout-minutes:\s*([0-9]+)\s*$", job_text, re.MULTILINE)
    return int(match.group(1)) if match else None


def workflow_runtime_report(workflows: dict[str, str]) -> dict[str, Any]:
    job_timeouts = {}
    missing_timeouts = {}
    invalid_timeouts = {}
    for workflow_name, workflow in workflows.items():
        timeouts = {
            job_name: workflow_job_timeout_minutes(workflow, job_name)
            for job_name in workflow_job_names(workflow)
        }
        job_timeouts[workflow_name] = timeouts
        missing = sorted(
            job_name for job_name, timeout in timeouts.items() if timeout is None
        )
        invalid = sorted(
            job_name
            for job_name, timeout in timeouts.items()
            if timeout is not None and timeout <= 0
        )
        if missing:
            missing_timeouts[workflow_name] = missing
        if invalid:
            invalid_timeouts[workflow_name] = invalid

    job_count = sum(len(timeouts) for timeouts in job_timeouts.values())
    jobs_with_timeouts = sum(
        timeout is not None
        for timeouts in job_timeouts.values()
        for timeout in timeouts.values()
    )
    return {
        "workflow_count": len(workflows),
        "job_count": job_count,
        "jobs_with_timeouts": jobs_with_timeouts,
        "job_timeouts": job_timeouts,
        "missing_job_timeouts": missing_timeouts,
        "invalid_job_timeouts": invalid_timeouts,
    }


def workflow_has_top_level_permissions(workflow: str) -> bool:
    return re.search(r"^permissions:\s*(?:\S.*)?$", workflow, re.MULTILINE) is not None


def workflow_top_level_permissions(workflow: str) -> dict[str, str]:
    inline = re.search(r"^permissions:\s*(\S.+?)\s*$", workflow, re.MULTILINE)
    section = nested_yaml_section(workflow, "permissions", 0)
    if inline and not section:
        return {"*": strip_yaml_scalar(inline.group(1))}

    permissions = {}
    for line in section:
        match = re.match(r"^  ([A-Za-z0-9_-]+):\s*([A-Za-z-]+)\s*$", line)
        if match:
            permissions[match.group(1)] = strip_yaml_scalar(match.group(2))
    return permissions


def workflow_write_permissions(permissions: dict[str, str]) -> list[str]:
    writes = []
    for permission, access in permissions.items():
        if access == "write" or access == "write-all":
            writes.append(permission)
    return sorted(writes)


def workflow_permissions_report(workflows: dict[str, str]) -> dict[str, Any]:
    explicit_permissions = {
        workflow_name: workflow_has_top_level_permissions(workflow)
        for workflow_name, workflow in workflows.items()
    }
    declared_permissions = {
        workflow_name: workflow_top_level_permissions(workflow)
        for workflow_name, workflow in workflows.items()
    }
    write_permissions = {
        workflow_name: workflow_write_permissions(permissions)
        for workflow_name, permissions in declared_permissions.items()
    }
    unexpected_writes = {}
    missing_required_writes = {}
    for workflow_name, writes in write_permissions.items():
        allowed = set(WORKFLOW_WRITE_PERMISSION_ALLOWLIST.get(workflow_name, []))
        actual = set(writes)
        unexpected = sorted(actual - allowed)
        missing = sorted(allowed - actual)
        if unexpected:
            unexpected_writes[workflow_name] = unexpected
        if missing:
            missing_required_writes[workflow_name] = missing

    return {
        "workflow_count": len(workflows),
        "explicit_permissions": explicit_permissions,
        "missing_explicit_permissions": sorted(
            workflow_name
            for workflow_name, explicit in explicit_permissions.items()
            if not explicit
        ),
        "declared_permissions": declared_permissions,
        "write_permissions": write_permissions,
        "unexpected_write_permissions": unexpected_writes,
        "missing_required_write_permissions": missing_required_writes,
    }


def workflow_action_refs(workflow: str) -> list[str]:
    refs = []
    for match in re.finditer(r"^\s*uses:\s+(.+?)\s*$", workflow, re.MULTILINE):
        action_ref = strip_yaml_scalar(match.group(1))
        if action_ref.startswith("./") or "@" not in action_ref:
            continue
        refs.append(action_ref)
    return refs


def action_ref_name(action_ref: str) -> str:
    return action_ref.rsplit("@", 1)[1].strip()


def workflow_actions_report(workflows: dict[str, str]) -> dict[str, Any]:
    action_refs = {
        workflow_name: workflow_action_refs(workflow)
        for workflow_name, workflow in workflows.items()
    }
    mutable_refs = {}
    for workflow_name, refs in action_refs.items():
        mutable = [
            action_ref
            for action_ref in refs
            if action_ref_name(action_ref).lower() in MUTABLE_ACTION_REFS
        ]
        if mutable:
            mutable_refs[workflow_name] = mutable
    return {
        "workflow_count": len(workflows),
        "action_refs": action_refs,
        "mutable_refs": mutable_refs,
    }


def workflow_has_pull_request_target(workflow: str) -> bool:
    return (
        re.search(r"^\s*pull_request_target:\s*$", workflow, re.MULTILINE) is not None
    )


def workflow_pr_head_markers(workflow: str) -> list[str]:
    return sorted(marker for marker in PULL_REQUEST_HEAD_MARKERS if marker in workflow)


def checkout_persists_credentials(checkout_step: str) -> bool:
    return "persist-credentials: false" not in checkout_step


def pull_request_target_report(workflows: dict[str, str]) -> dict[str, Any]:
    target_workflows = sorted(
        workflow_name
        for workflow_name, workflow in workflows.items()
        if workflow_has_pull_request_target(workflow)
    )
    unexpected = sorted(
        set(target_workflows) - set(PULL_REQUEST_TARGET_WORKFLOW_ALLOWLIST)
    )
    trusted_base_checkout = {}
    checkout_credentials_persist = {}
    head_context_markers = {}
    support_traceability = {}
    github_token_scoped_to_sync = {}

    for workflow_name in target_workflows:
        workflow = workflows[workflow_name]
        checkout_step = workflow_step_section(workflow, "Checkout trusted base")
        sync_step = workflow_step_section(workflow, "Sync PR issue links")
        trusted_base_checkout[workflow_name] = (
            "uses: actions/checkout@" in checkout_step
            and "ref:" not in checkout_step
            and "repository:" not in checkout_step
        )
        checkout_credentials_persist[workflow_name] = checkout_persists_credentials(
            checkout_step
        )
        markers = workflow_pr_head_markers(workflow)
        if markers:
            head_context_markers[workflow_name] = markers
        support_traceability[workflow_name] = (
            "python tools/sync_pr_issue_links.py" in sync_step
            and "--check-support-traceability" in sync_step
        )
        github_token_scoped_to_sync[workflow_name] = (
            "GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}" in sync_step
        )

    return {
        "allowlist": sorted(PULL_REQUEST_TARGET_WORKFLOW_ALLOWLIST),
        "workflows": target_workflows,
        "unexpected_workflows": unexpected,
        "trusted_base_checkout": trusted_base_checkout,
        "checkout_credentials_persist": checkout_credentials_persist,
        "head_context_markers": head_context_markers,
        "support_traceability": support_traceability,
        "github_token_scoped_to_sync": github_token_scoped_to_sync,
    }


def load_backends() -> list[dict[str, Any]]:
    return json.loads(read_text(BACKENDS_PATH))["backends"]


def workflow_text(name: str) -> str:
    path = WORKFLOW_DIR / name
    if not path.exists():
        raise CiCoverageError("Workflow does not exist: {}".format(relpath(path)))
    return read_text(path)


def all_workflow_texts() -> dict[str, str]:
    if not WORKFLOW_DIR.exists():
        raise CiCoverageError(
            "Workflow directory does not exist: {}".format(relpath(WORKFLOW_DIR))
        )
    workflows = {
        path.name: read_text(path) for path in sorted(WORKFLOW_DIR.glob("*.yml"))
    }
    if not workflows:
        raise CiCoverageError("No workflow files found in {}".format(WORKFLOW_DIR))
    return workflows


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


def full_suite_failure_summary_report(workflow: str) -> dict[str, dict[str, bool]]:
    summaries = {}
    for name, config in FULL_SUITE_FAILURE_SUMMARIES.items():
        job = config["job"]
        run_step = workflow_job_step_section(workflow, job, config["run_step"])
        summary_step = workflow_job_step_section(workflow, job, config["summary_step"])
        upload_step = workflow_job_step_section(workflow, job, config["upload_step"])
        summaries[name] = {
            "writes_junit": (
                "--junitxml {}".format(config["junit"]) in run_step
                or "--junitxml\n          {}".format(config["junit"]) in run_step
            ),
            "writes_failure_summary": (
                "python tools/pytest_failure_summary.py" in summary_step
                and config["junit"] in summary_step
                and "--json-output {}".format(config["json"]) in summary_step
                and "--markdown-output {}".format(config["markdown"]) in summary_step
            ),
            "appends_to_step_summary": (
                'cat {} >> "$GITHUB_STEP_SUMMARY"'.format(config["markdown"])
                in summary_step
            ),
            "summary_on_failure": "if: always()" in summary_step,
            "summary_after_run": workflow_job_step_after(
                workflow,
                job,
                config["summary_step"],
                config["run_step"],
            ),
            "uploads_failure_summary": (
                "actions/upload-artifact@v4" in upload_step
                and "name: {}".format(config["artifact"]) in upload_step
                and config["junit"] in upload_step
                and config["json"] in upload_step
                and config["markdown"] in upload_step
            ),
            "upload_on_failure": "if: always()" in upload_step,
            "upload_ignores_missing_files": "if-no-files-found: ignore" in upload_step,
            "upload_retention": "retention-days: 30" in upload_step,
            "upload_after_summary": workflow_job_step_after(
                workflow,
                job,
                config["upload_step"],
                config["summary_step"],
            ),
        }
    return summaries


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
        "failure_summaries": full_suite_failure_summary_report(workflow),
    }


def support_issue_sync_report(workflow: str) -> dict[str, Any]:
    path_filters = pull_request_path_filters(workflow)
    support_matrix_check_step = workflow_step_section(
        workflow, "Validate support matrix artifacts"
    )
    support_matrix_check_upload_step = workflow_step_section(
        workflow, "Upload support matrix check report"
    )
    ci_coverage_upload_step = workflow_step_section(
        workflow, "Upload CI coverage report"
    )
    support_signal_upload_step = workflow_step_section(
        workflow, "Upload support signal reports"
    )
    download_test_failure_step = workflow_step_section(
        workflow, "Download test failure summaries"
    )
    extract_signal_step = workflow_step_section(
        workflow, "Extract generated support signals"
    )
    dry_run_step = workflow_step_section(workflow, "Dry-run issue sync")
    plan_step = workflow_step_section(workflow, "Plan GitHub issue sync")
    sync_step = workflow_step_section(workflow, "Sync GitHub issues")
    summary_step = workflow_step_section(workflow, "Write support automation summary")
    issue_report_upload_step = workflow_step_section(
        workflow, "Upload support issue sync reports"
    )
    return {
        "workflow": "support-issue-sync.yml",
        "hourly_schedule": 'cron: "17 * * * *"' in workflow,
        "workflow_run_full_tests": (
            "workflow_run:" in workflow
            and "Complete Test Suite" in workflow
            and "types:" in workflow
            and "completed" in workflow
            and "branches:" in workflow
            and "main" in workflow
        ),
        "dry_run_on_pull_request": (
            "if: github.event_name == 'pull_request'" in dry_run_step
            and "--dry-run" in dry_run_step
        ),
        "mutates_outside_pull_request": (
            "if: github.event_name != 'pull_request'" in sync_step
            and "--dry-run" not in sync_step
            and "GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}" in sync_step
        ),
        "required_tests": {
            test: test in workflow for test in SUPPORT_ISSUE_SYNC_REQUIRED_TESTS
        },
        "required_path_filters": {
            path: path in path_filters
            for path in SUPPORT_ISSUE_SYNC_REQUIRED_PATH_FILTERS
        },
        "min_desired_issues": "--min-desired-issues 10" in workflow,
        "writes_support_matrix_check_report": (
            "python tools/support_matrix.py check --output support/generated/support-matrix-check.json"
            in support_matrix_check_step
        ),
        "uploads_support_matrix_check_report": (
            "actions/upload-artifact@v4" in support_matrix_check_upload_step
            and "name: support-matrix-check-report" in support_matrix_check_upload_step
            and "support/generated/support-matrix-check.json"
            in support_matrix_check_upload_step
        ),
        "uploads_support_matrix_check_report_on_failure": (
            "if: always()" in support_matrix_check_upload_step
        ),
        "support_matrix_check_report_ignores_missing_files": (
            "if-no-files-found: ignore" in support_matrix_check_upload_step
        ),
        "support_matrix_check_report_retention": (
            "retention-days: 30" in support_matrix_check_upload_step
        ),
        "support_matrix_check_upload_after_validate": workflow_step_after(
            workflow,
            "Upload support matrix check report",
            "Validate support matrix artifacts",
        ),
        "issue_sync_uses_support_matrix_check_report": (
            "--matrix-check-report support/generated/support-matrix-check.json"
            in dry_run_step
            and "--matrix-check-report support/generated/support-matrix-check.json"
            in plan_step
            and "--matrix-check-report support/generated/support-matrix-check.json"
            in sync_step
        ),
        "writes_support_automation_summary": (
            "python tools/support_ci_summary.py" in summary_step
            and "--matrix-check support/generated/support-matrix-check.json"
            in summary_step
            and "--issue-plan support/generated/support-issue-plan.json" in summary_step
            and "--sync-summary support/generated/support-issue-sync-summary.json"
            in summary_step
            and "--output support/generated/support-issue-ci-summary.md" in summary_step
        ),
        "support_automation_summary_on_failure": "if: always()" in summary_step,
        "appends_support_automation_summary_to_step_summary": (
            '--step-summary "$GITHUB_STEP_SUMMARY"' in summary_step
        ),
        "support_automation_summary_emits_annotations": (
            "--github-annotations" in summary_step
        ),
        "support_automation_summary_fails_on_attention": (
            "--fail-on-attention" in summary_step
        ),
        "support_automation_summary_after_issue_sync": (
            workflow_step_after(
                workflow,
                "Write support automation summary",
                "Dry-run issue sync",
            )
            and workflow_step_after(
                workflow,
                "Write support automation summary",
                "Plan GitHub issue sync",
            )
            and workflow_step_after(
                workflow,
                "Write support automation summary",
                "Sync GitHub issues",
            )
        ),
        "dry_run_writes_issue_plan": (
            "--plan-output support/generated/support-issue-plan.json" in dry_run_step
        ),
        "plans_issue_sync_before_mutation": (
            "if: github.event_name != 'pull_request'" in plan_step
            and "GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}" in plan_step
            and "--inspect-existing" in plan_step
            and "--plan-output support/generated/support-issue-plan.json" in plan_step
            and "--dry-run" in plan_step
            and workflow_step_after(
                workflow,
                "Sync GitHub issues",
                "Plan GitHub issue sync",
            )
        ),
        "checks_planned_action_budget": all(
            flag in plan_step for flag in SUPPORT_ISSUE_SYNC_PLANNED_ACTION_BUDGET_ARGS
        ),
        "sync_replans_before_mutation": (
            "--inspect-existing" in sync_step
            and "--plan-output support/generated/support-issue-plan.json" in sync_step
        ),
        "sync_checks_planned_action_budget": all(
            flag in sync_step for flag in SUPPORT_ISSUE_SYNC_PLANNED_ACTION_BUDGET_ARGS
        ),
        "sync_writes_issue_summary": (
            "--sync-summary-output support/generated/support-issue-sync-summary.json"
            in sync_step
        ),
        "uploads_ci_coverage_artifact_on_failure": (
            "if: always()" in ci_coverage_upload_step
        ),
        "ci_coverage_artifact_retention": (
            "retention-days: 30" in ci_coverage_upload_step
        ),
        "uploads_support_signal_artifact": (
            "actions/upload-artifact@v4" in support_signal_upload_step
            and "name: support-signal-reports" in support_signal_upload_step
            and "support/generated/backend-docs-report.json"
            in support_signal_upload_step
            and "support/generated/support-signals.json" in support_signal_upload_step
        ),
        "uploads_support_signal_artifact_on_failure": (
            "if: always()" in support_signal_upload_step
        ),
        "support_signal_artifact_ignores_missing_files": (
            "if-no-files-found: ignore" in support_signal_upload_step
        ),
        "support_signal_artifact_retention": (
            "retention-days: 30" in support_signal_upload_step
        ),
        "support_signal_upload_after_extract": workflow_step_after(
            workflow,
            "Upload support signal reports",
            "Extract generated support signals",
        ),
        "downloads_test_failure_summaries": (
            "actions/download-artifact@v4" in download_test_failure_step
            and "github-token: ${{ secrets.GITHUB_TOKEN }}"
            in download_test_failure_step
            and "run-id: ${{ github.event.workflow_run.id }}"
            in download_test_failure_step
            and 'pattern: "*failure-summary*"' in download_test_failure_step
            and "path: support/generated/pytest-failures" in download_test_failure_step
        ),
        "downloads_test_failure_summaries_on_workflow_run": (
            "if: github.event_name == 'workflow_run'" in download_test_failure_step
        ),
        "test_failure_summary_download_non_blocking": (
            "continue-on-error: true" in download_test_failure_step
        ),
        "support_signals_uses_pytest_failure_summaries": (
            "--pytest-failure-summary" in extract_signal_step
            and "support/generated/pytest-failures" in extract_signal_step
        ),
        "uploads_pytest_failure_summary_inputs": (
            "support/generated/pytest-failures/**" in support_signal_upload_step
        ),
        "uploads_issue_sync_report_artifact": (
            "actions/upload-artifact@v4" in issue_report_upload_step
            and "name: support-issue-sync-reports" in issue_report_upload_step
            and "support/generated/support-matrix-check.json"
            in issue_report_upload_step
            and "support/generated/support-issue-plan.json" in issue_report_upload_step
            and "support/generated/support-issue-sync-summary.json"
            in issue_report_upload_step
            and "support/generated/support-issue-ci-summary.md"
            in issue_report_upload_step
        ),
        "uploads_issue_sync_report_artifact_on_failure": (
            "if: always()" in issue_report_upload_step
        ),
        "issue_sync_report_artifact_ignores_missing_files": (
            "if-no-files-found: ignore" in issue_report_upload_step
        ),
        "issue_sync_report_artifact_retention": (
            "retention-days: 30" in issue_report_upload_step
        ),
        "issue_sync_report_upload_after_sync": (
            workflow_step_after(
                workflow,
                "Upload support issue sync reports",
                "Dry-run issue sync",
            )
            and workflow_step_after(
                workflow,
                "Upload support issue sync reports",
                "Plan GitHub issue sync",
            )
            and workflow_step_after(
                workflow,
                "Upload support issue sync reports",
                "Sync GitHub issues",
            )
            and workflow_step_after(
                workflow,
                "Upload support issue sync reports",
                "Write support automation summary",
            )
        ),
    }


def support_matrix_report(workflow: str) -> dict[str, Any]:
    check_upload_step = workflow_step_section(
        workflow, "Upload support matrix check report"
    )
    docs_probe_upload_step = workflow_step_section(
        workflow, "Upload documentation probe report"
    )
    return {
        "workflow": "support-matrix.yml",
        "required_policies": {
            name: marker in workflow
            for name, marker in SUPPORT_MATRIX_REQUIRED_POLICIES.items()
        },
        "uploads_check_report_artifact": (
            "actions/upload-artifact@v4" in check_upload_step
            and "support/generated/support-matrix-check.json" in check_upload_step
        ),
        "uploads_check_report_artifact_on_failure": "if: always()" in check_upload_step,
        "check_report_artifact_retention": "retention-days: 30" in check_upload_step,
        "check_report_upload_after_validate": workflow_step_after(
            workflow,
            "Upload support matrix check report",
            "Validate support matrix",
        ),
        "uploads_docs_probe_artifact": (
            "actions/upload-artifact@v4" in docs_probe_upload_step
            and "support/generated/backend-docs-report.json" in docs_probe_upload_step
        ),
        "uploads_docs_probe_artifact_on_failure": (
            "if: always()" in docs_probe_upload_step
        ),
        "docs_probe_artifact_retention": "retention-days: 30" in docs_probe_upload_step,
    }


def docs_report(workflow: str) -> dict[str, Any]:
    return {
        "workflow": "docs.yml",
        "required_policies": {
            name: marker in workflow for name, marker in DOCS_REQUIRED_POLICIES.items()
        },
    }


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

    workflow_texts = all_workflow_texts()
    backend_workflow = workflow_texts["backend-tests.yml"]
    translator_workflow = workflow_texts["translator-tests.yml"]
    docs_workflow = workflow_texts["docs.yml"]
    examples_workflow = workflow_texts["examples-test.yml"]
    full_workflow = workflow_texts["full-tests.yml"]
    support_matrix_workflow = workflow_texts["support-matrix.yml"]
    support_issue_workflow = workflow_texts["support-issue-sync.yml"]

    report = {
        "schema_version": 1,
        "generator": "tools/ci_coverage.py",
        "catalog": {
            "backend_count": len(backends),
            "backend_ids": backend_ids,
        },
        "workflows": {
            "runtime": workflow_runtime_report(workflow_texts),
            "permissions": workflow_permissions_report(workflow_texts),
            "actions": workflow_actions_report(workflow_texts),
            "pull_request_target": pull_request_target_report(workflow_texts),
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
    runtime = report["workflows"]["runtime"]
    for workflow_name, job_names in runtime["missing_job_timeouts"].items():
        errors.append(
            "{} missing timeout-minutes for jobs: {}".format(
                workflow_name, ", ".join(job_names)
            )
        )
    for workflow_name, job_names in runtime["invalid_job_timeouts"].items():
        errors.append(
            "{} has invalid timeout-minutes for jobs: {}".format(
                workflow_name, ", ".join(job_names)
            )
        )

    permissions = report["workflows"]["permissions"]
    for workflow_name in permissions["missing_explicit_permissions"]:
        errors.append("{} missing explicit permissions".format(workflow_name))
    for workflow_name, permission_names in permissions[
        "unexpected_write_permissions"
    ].items():
        errors.append(
            "{} has unexpected write permissions: {}".format(
                workflow_name, ", ".join(permission_names)
            )
        )
    for workflow_name, permission_names in permissions[
        "missing_required_write_permissions"
    ].items():
        errors.append(
            "{} missing required write permissions: {}".format(
                workflow_name, ", ".join(permission_names)
            )
        )

    actions = report["workflows"]["actions"]
    for workflow_name, action_refs in actions["mutable_refs"].items():
        errors.append(
            "{} has mutable action refs: {}".format(
                workflow_name, ", ".join(action_refs)
            )
        )

    pull_request_target = report["workflows"]["pull_request_target"]
    for workflow_name in pull_request_target["unexpected_workflows"]:
        errors.append(
            "{} uses pull_request_target but is not allowlisted".format(workflow_name)
        )
    for workflow_name, trusted in pull_request_target["trusted_base_checkout"].items():
        if not trusted:
            errors.append(
                "{} pull_request_target must checkout trusted base".format(
                    workflow_name
                )
            )
    for workflow_name, persists in pull_request_target[
        "checkout_credentials_persist"
    ].items():
        if persists:
            errors.append(
                "{} pull_request_target checkout must not persist credentials".format(
                    workflow_name
                )
            )
    for workflow_name, markers in pull_request_target["head_context_markers"].items():
        errors.append(
            "{} pull_request_target references PR head context: {}".format(
                workflow_name, ", ".join(markers)
            )
        )
    for workflow_name, enabled in pull_request_target["support_traceability"].items():
        if not enabled:
            errors.append(
                "{} pull_request_target must check support traceability".format(
                    workflow_name
                )
            )
    for workflow_name, scoped in pull_request_target[
        "github_token_scoped_to_sync"
    ].items():
        if not scoped:
            errors.append(
                "{} pull_request_target must scope GITHUB_TOKEN to sync step".format(
                    workflow_name
                )
            )

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
    for summary_name, summary_fields in full_tests["failure_summaries"].items():
        for field, present in summary_fields.items():
            if not present:
                errors.append(
                    "full-tests.yml missing pytest failure summary for {}: {}".format(
                        summary_name,
                        field,
                    )
                )

    support_matrix = report["workflows"]["support_matrix"]
    for policy, present in support_matrix["required_policies"].items():
        if not present:
            errors.append("support-matrix.yml missing policy: {}".format(policy))
    if not support_matrix["uploads_check_report_artifact"]:
        errors.append("support-matrix.yml missing check report artifact upload")
    if not support_matrix["uploads_check_report_artifact_on_failure"]:
        errors.append("support-matrix.yml check report artifact must upload on failure")
    if not support_matrix["check_report_artifact_retention"]:
        errors.append(
            "support-matrix.yml check report artifact must set retention-days"
        )
    if not support_matrix["check_report_upload_after_validate"]:
        errors.append(
            "support-matrix.yml check report upload must run after validation"
        )
    if not support_matrix["uploads_docs_probe_artifact"]:
        errors.append("support-matrix.yml missing docs probe artifact upload")
    if not support_matrix["uploads_docs_probe_artifact_on_failure"]:
        errors.append("support-matrix.yml docs probe artifact must upload on failure")
    if not support_matrix["docs_probe_artifact_retention"]:
        errors.append("support-matrix.yml docs probe artifact must set retention-days")

    support_sync = report["workflows"]["support_issue_sync"]
    for field in (
        "hourly_schedule",
        "workflow_run_full_tests",
        "dry_run_on_pull_request",
        "mutates_outside_pull_request",
        "min_desired_issues",
        "writes_support_matrix_check_report",
        "uploads_support_matrix_check_report",
        "uploads_support_matrix_check_report_on_failure",
        "support_matrix_check_report_ignores_missing_files",
        "support_matrix_check_report_retention",
        "support_matrix_check_upload_after_validate",
        "issue_sync_uses_support_matrix_check_report",
        "writes_support_automation_summary",
        "support_automation_summary_on_failure",
        "appends_support_automation_summary_to_step_summary",
        "support_automation_summary_emits_annotations",
        "support_automation_summary_fails_on_attention",
        "support_automation_summary_after_issue_sync",
        "dry_run_writes_issue_plan",
        "plans_issue_sync_before_mutation",
        "checks_planned_action_budget",
        "sync_replans_before_mutation",
        "sync_checks_planned_action_budget",
        "sync_writes_issue_summary",
        "uploads_ci_coverage_artifact_on_failure",
        "ci_coverage_artifact_retention",
        "uploads_support_signal_artifact",
        "uploads_support_signal_artifact_on_failure",
        "support_signal_artifact_ignores_missing_files",
        "support_signal_artifact_retention",
        "support_signal_upload_after_extract",
        "downloads_test_failure_summaries",
        "downloads_test_failure_summaries_on_workflow_run",
        "test_failure_summary_download_non_blocking",
        "support_signals_uses_pytest_failure_summaries",
        "uploads_pytest_failure_summary_inputs",
        "uploads_issue_sync_report_artifact",
        "uploads_issue_sync_report_artifact_on_failure",
        "issue_sync_report_artifact_ignores_missing_files",
        "issue_sync_report_artifact_retention",
        "issue_sync_report_upload_after_sync",
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


def workflow_timeout_presence(runtime: dict[str, Any]) -> dict[str, bool]:
    return {
        "{}:{}".format(workflow_name, job_name): timeout is not None and timeout > 0
        for workflow_name, timeouts in runtime["job_timeouts"].items()
        for job_name, timeout in timeouts.items()
    }


def workflow_required_write_presence(permissions: dict[str, Any]) -> dict[str, bool]:
    write_permissions = permissions["write_permissions"]
    presence = {}
    for (
        workflow_name,
        required_permissions,
    ) in WORKFLOW_WRITE_PERMISSION_ALLOWLIST.items():
        if workflow_name not in write_permissions:
            continue
        actual_permissions = set(write_permissions[workflow_name])
        for permission in required_permissions:
            presence["{}:{}".format(workflow_name, permission)] = (
                permission in actual_permissions
            )
    return presence


def workflow_write_policy_presence(permissions: dict[str, Any]) -> dict[str, bool]:
    return {
        workflow_name: not bool(
            permissions["unexpected_write_permissions"].get(workflow_name)
        )
        for workflow_name in permissions["explicit_permissions"]
    }


def workflow_action_policy_presence(actions: dict[str, Any]) -> dict[str, bool]:
    return {
        workflow_name: not bool(actions["mutable_refs"].get(workflow_name))
        for workflow_name in actions["action_refs"]
    }


def pull_request_target_policy_presence(report: dict[str, Any]) -> dict[str, bool]:
    presence = {}
    for workflow_name in report["workflows"]:
        presence["{}:allowlisted".format(workflow_name)] = (
            workflow_name not in report["unexpected_workflows"]
        )
    for workflow_name, trusted in report["trusted_base_checkout"].items():
        presence["{}:trusted_base_checkout".format(workflow_name)] = trusted
    for workflow_name, persists in report["checkout_credentials_persist"].items():
        presence["{}:no_persisted_checkout_credentials".format(workflow_name)] = (
            not persists
        )
    for workflow_name, markers in report["head_context_markers"].items():
        presence["{}:no_pr_head_context".format(workflow_name)] = not bool(markers)
    for workflow_name, enabled in report["support_traceability"].items():
        presence["{}:support_traceability".format(workflow_name)] = enabled
    for workflow_name, scoped in report["github_token_scoped_to_sync"].items():
        presence["{}:github_token_scoped_to_sync".format(workflow_name)] = scoped
    return presence


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

    baseline_runtime = baseline["workflows"]["runtime"]
    current_runtime = current["workflows"]["runtime"]
    add_bool_map_change(
        "workflows",
        "job_timeouts",
        workflow_timeout_presence(baseline_runtime),
        workflow_timeout_presence(current_runtime),
    )

    baseline_permissions = baseline["workflows"]["permissions"]
    current_permissions = current["workflows"]["permissions"]
    add_bool_map_change(
        "workflows",
        "explicit_permissions",
        baseline_permissions["explicit_permissions"],
        current_permissions["explicit_permissions"],
    )
    add_bool_map_change(
        "workflows",
        "required_write_permissions",
        workflow_required_write_presence(baseline_permissions),
        workflow_required_write_presence(current_permissions),
    )
    add_bool_map_change(
        "workflows",
        "write_permission_policy",
        workflow_write_policy_presence(baseline_permissions),
        workflow_write_policy_presence(current_permissions),
    )

    baseline_actions = baseline["workflows"]["actions"]
    current_actions = current["workflows"]["actions"]
    add_bool_map_change(
        "workflows",
        "action_ref_policy",
        workflow_action_policy_presence(baseline_actions),
        workflow_action_policy_presence(current_actions),
    )

    baseline_pull_request_target = baseline["workflows"]["pull_request_target"]
    current_pull_request_target = current["workflows"]["pull_request_target"]
    add_set_change(
        "workflows",
        "pull_request_target_workflows",
        baseline_pull_request_target["workflows"],
        current_pull_request_target["workflows"],
    )
    add_bool_map_change(
        "workflows",
        "pull_request_target_policy",
        pull_request_target_policy_presence(baseline_pull_request_target),
        pull_request_target_policy_presence(current_pull_request_target),
    )

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
    for summary_name in sorted(
        set(baseline_full["failure_summaries"]) | set(current_full["failure_summaries"])
    ):
        baseline_summary = baseline_full["failure_summaries"].get(summary_name, {})
        current_summary = current_full["failure_summaries"].get(summary_name, {})
        for field in sorted(set(baseline_summary) | set(current_summary)):
            add_bool_change(
                "full-tests.yml",
                "failure_summaries.{}.{}".format(summary_name, field),
                bool(baseline_summary.get(field)),
                bool(current_summary.get(field)),
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
        "uploads_check_report_artifact",
        baseline_matrix["uploads_check_report_artifact"],
        current_matrix["uploads_check_report_artifact"],
    )
    add_bool_change(
        "support-matrix.yml",
        "uploads_check_report_artifact_on_failure",
        baseline_matrix["uploads_check_report_artifact_on_failure"],
        current_matrix["uploads_check_report_artifact_on_failure"],
    )
    add_bool_change(
        "support-matrix.yml",
        "check_report_artifact_retention",
        baseline_matrix["check_report_artifact_retention"],
        current_matrix["check_report_artifact_retention"],
    )
    add_bool_change(
        "support-matrix.yml",
        "check_report_upload_after_validate",
        baseline_matrix["check_report_upload_after_validate"],
        current_matrix["check_report_upload_after_validate"],
    )
    add_bool_change(
        "support-matrix.yml",
        "uploads_docs_probe_artifact",
        baseline_matrix["uploads_docs_probe_artifact"],
        current_matrix["uploads_docs_probe_artifact"],
    )
    add_bool_change(
        "support-matrix.yml",
        "uploads_docs_probe_artifact_on_failure",
        baseline_matrix["uploads_docs_probe_artifact_on_failure"],
        current_matrix["uploads_docs_probe_artifact_on_failure"],
    )
    add_bool_change(
        "support-matrix.yml",
        "docs_probe_artifact_retention",
        baseline_matrix["docs_probe_artifact_retention"],
        current_matrix["docs_probe_artifact_retention"],
    )

    baseline_support = baseline["workflows"]["support_issue_sync"]
    current_support = current["workflows"]["support_issue_sync"]
    for dimension in (
        "hourly_schedule",
        "workflow_run_full_tests",
        "dry_run_on_pull_request",
        "mutates_outside_pull_request",
        "min_desired_issues",
        "writes_support_matrix_check_report",
        "uploads_support_matrix_check_report",
        "uploads_support_matrix_check_report_on_failure",
        "support_matrix_check_report_ignores_missing_files",
        "support_matrix_check_report_retention",
        "support_matrix_check_upload_after_validate",
        "issue_sync_uses_support_matrix_check_report",
        "writes_support_automation_summary",
        "support_automation_summary_on_failure",
        "appends_support_automation_summary_to_step_summary",
        "support_automation_summary_after_issue_sync",
        "dry_run_writes_issue_plan",
        "plans_issue_sync_before_mutation",
        "checks_planned_action_budget",
        "sync_replans_before_mutation",
        "sync_checks_planned_action_budget",
        "sync_writes_issue_summary",
        "uploads_ci_coverage_artifact_on_failure",
        "ci_coverage_artifact_retention",
        "uploads_support_signal_artifact",
        "uploads_support_signal_artifact_on_failure",
        "support_signal_artifact_ignores_missing_files",
        "support_signal_artifact_retention",
        "support_signal_upload_after_extract",
        "downloads_test_failure_summaries",
        "downloads_test_failure_summaries_on_workflow_run",
        "test_failure_summary_download_non_blocking",
        "support_signals_uses_pytest_failure_summaries",
        "uploads_pytest_failure_summary_inputs",
        "uploads_issue_sync_report_artifact",
        "uploads_issue_sync_report_artifact_on_failure",
        "issue_sync_report_artifact_ignores_missing_files",
        "issue_sync_report_artifact_retention",
        "issue_sync_report_upload_after_sync",
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


def comma_list(values: list[str]) -> str:
    return ", ".join(values) if values else "none"


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
    runtime = report["workflows"]["runtime"]
    permissions = report["workflows"]["permissions"]
    actions = report["workflows"]["actions"]
    pull_request_target = report["workflows"]["pull_request_target"]
    backend_tests = report["workflows"]["backend_tests"]
    translator_tests = report["workflows"]["translator_tests"]
    docs = report["workflows"]["docs"]
    examples = report["workflows"]["examples"]
    full_tests = report["workflows"]["full_tests"]
    support_matrix = report["workflows"]["support_matrix"]
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
            "## Workflow Runtime",
            "",
        ]
    )
    lines.extend(
        markdown_table(
            ["Workflow", "Jobs", "Timeouts", "Missing", "Invalid"],
            [
                [
                    workflow_name,
                    len(timeouts),
                    "{} / {}".format(
                        sum(timeout is not None for timeout in timeouts.values()),
                        len(timeouts),
                    ),
                    comma_list(runtime["missing_job_timeouts"].get(workflow_name, [])),
                    comma_list(runtime["invalid_job_timeouts"].get(workflow_name, [])),
                ]
                for workflow_name, timeouts in sorted(runtime["job_timeouts"].items())
            ],
        )
    )
    lines.extend(
        [
            "",
            "## Workflow Permissions",
            "",
        ]
    )
    lines.extend(
        markdown_table(
            [
                "Workflow",
                "Explicit",
                "Write permissions",
                "Unexpected writes",
                "Missing required writes",
            ],
            [
                [
                    workflow_name,
                    ok_text(permissions["explicit_permissions"][workflow_name]),
                    comma_list(permissions["write_permissions"].get(workflow_name, [])),
                    comma_list(
                        permissions["unexpected_write_permissions"].get(
                            workflow_name, []
                        )
                    ),
                    comma_list(
                        permissions["missing_required_write_permissions"].get(
                            workflow_name, []
                        )
                    ),
                ]
                for workflow_name in sorted(permissions["explicit_permissions"])
            ],
        )
    )
    lines.extend(
        [
            "",
            "## Workflow Actions",
            "",
        ]
    )
    lines.extend(
        markdown_table(
            ["Workflow", "Action refs", "Mutable refs"],
            [
                [
                    workflow_name,
                    len(actions["action_refs"].get(workflow_name, [])),
                    comma_list(actions["mutable_refs"].get(workflow_name, [])),
                ]
                for workflow_name in sorted(actions["action_refs"])
            ],
        )
    )
    lines.extend(
        [
            "",
            "## Pull Request Target",
            "",
        ]
    )
    lines.extend(
        markdown_table(
            [
                "Workflow",
                "Trusted checkout",
                "No persisted credentials",
                "No PR head context",
                "Traceability",
                "Token scoped",
            ],
            [
                [
                    workflow_name,
                    ok_text(
                        pull_request_target["trusted_base_checkout"].get(
                            workflow_name, False
                        )
                    ),
                    ok_text(
                        not pull_request_target["checkout_credentials_persist"].get(
                            workflow_name, True
                        )
                    ),
                    ok_text(
                        not pull_request_target["head_context_markers"].get(
                            workflow_name
                        )
                    ),
                    ok_text(
                        pull_request_target["support_traceability"].get(
                            workflow_name, False
                        )
                    ),
                    ok_text(
                        pull_request_target["github_token_scoped_to_sync"].get(
                            workflow_name, False
                        )
                    ),
                ]
                for workflow_name in pull_request_target["workflows"]
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
                [
                    "Pytest failure summaries",
                    ok_text(
                        all(
                            all(fields.values())
                            for fields in full_tests["failure_summaries"].values()
                        )
                    ),
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
                    "Support matrix check artifact",
                    ok_text(support_matrix["uploads_check_report_artifact"]),
                ],
                [
                    "Check artifact upload on failure",
                    ok_text(support_matrix["uploads_check_report_artifact_on_failure"]),
                ],
                [
                    "Check artifact retention",
                    ok_text(support_matrix["check_report_artifact_retention"]),
                ],
                [
                    "Check artifact upload after validation",
                    ok_text(support_matrix["check_report_upload_after_validate"]),
                ],
                [
                    "Documentation probe artifact",
                    ok_text(support_matrix["uploads_docs_probe_artifact"]),
                ],
                [
                    "Documentation probe upload on failure",
                    ok_text(support_matrix["uploads_docs_probe_artifact_on_failure"]),
                ],
                [
                    "Documentation probe artifact retention",
                    ok_text(support_matrix["docs_probe_artifact_retention"]),
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
                    "Support matrix check report",
                    ok_text(support_sync["writes_support_matrix_check_report"]),
                ],
                [
                    "Support matrix check artifact",
                    ok_text(support_sync["uploads_support_matrix_check_report"]),
                ],
                [
                    "Support matrix check upload on failure",
                    ok_text(
                        support_sync["uploads_support_matrix_check_report_on_failure"]
                    ),
                ],
                [
                    "Support matrix check ignores missing files",
                    ok_text(
                        support_sync[
                            "support_matrix_check_report_ignores_missing_files"
                        ]
                    ),
                ],
                [
                    "Support matrix check artifact retention",
                    ok_text(support_sync["support_matrix_check_report_retention"]),
                ],
                [
                    "Support matrix check upload after validation",
                    ok_text(support_sync["support_matrix_check_upload_after_validate"]),
                ],
                [
                    "Issue sync uses support matrix check",
                    ok_text(
                        support_sync["issue_sync_uses_support_matrix_check_report"]
                    ),
                ],
                [
                    "Support automation summary",
                    ok_text(support_sync["writes_support_automation_summary"]),
                ],
                [
                    "Support automation summary on failure",
                    ok_text(support_sync["support_automation_summary_on_failure"]),
                ],
                [
                    "Support automation summary in step summary",
                    ok_text(
                        support_sync[
                            "appends_support_automation_summary_to_step_summary"
                        ]
                    ),
                ],
                [
                    "Support automation summary after issue sync",
                    ok_text(
                        support_sync["support_automation_summary_after_issue_sync"]
                    ),
                ],
                [
                    "Dry-run writes issue plan",
                    ok_text(support_sync["dry_run_writes_issue_plan"]),
                ],
                [
                    "Issue sync planned before mutation",
                    ok_text(support_sync["plans_issue_sync_before_mutation"]),
                ],
                [
                    "Planned action budget guard",
                    ok_text(support_sync["checks_planned_action_budget"]),
                ],
                [
                    "Sync replans before mutation",
                    ok_text(support_sync["sync_replans_before_mutation"]),
                ],
                [
                    "Sync planned action budget guard",
                    ok_text(support_sync["sync_checks_planned_action_budget"]),
                ],
                [
                    "Issue sync writes summary",
                    ok_text(support_sync["sync_writes_issue_summary"]),
                ],
                [
                    "CI coverage artifact upload on failure",
                    ok_text(support_sync["uploads_ci_coverage_artifact_on_failure"]),
                ],
                [
                    "CI coverage artifact retention",
                    ok_text(support_sync["ci_coverage_artifact_retention"]),
                ],
                [
                    "Support signal artifact",
                    ok_text(support_sync["uploads_support_signal_artifact"]),
                ],
                [
                    "Support signal upload on failure",
                    ok_text(support_sync["uploads_support_signal_artifact_on_failure"]),
                ],
                [
                    "Support signal ignores missing files",
                    ok_text(
                        support_sync["support_signal_artifact_ignores_missing_files"]
                    ),
                ],
                [
                    "Support signal artifact retention",
                    ok_text(support_sync["support_signal_artifact_retention"]),
                ],
                [
                    "Support signal upload after extract",
                    ok_text(support_sync["support_signal_upload_after_extract"]),
                ],
                [
                    "Full-test workflow_run trigger",
                    ok_text(support_sync["workflow_run_full_tests"]),
                ],
                [
                    "Downloads test failure summaries",
                    ok_text(support_sync["downloads_test_failure_summaries"]),
                ],
                [
                    "Test failure download scoped to workflow_run",
                    ok_text(
                        support_sync["downloads_test_failure_summaries_on_workflow_run"]
                    ),
                ],
                [
                    "Test failure download non-blocking",
                    ok_text(support_sync["test_failure_summary_download_non_blocking"]),
                ],
                [
                    "Support signals consume pytest failures",
                    ok_text(
                        support_sync["support_signals_uses_pytest_failure_summaries"]
                    ),
                ],
                [
                    "Uploads pytest failure inputs",
                    ok_text(support_sync["uploads_pytest_failure_summary_inputs"]),
                ],
                [
                    "Issue sync report artifact",
                    ok_text(support_sync["uploads_issue_sync_report_artifact"]),
                ],
                [
                    "Issue sync report upload on failure",
                    ok_text(
                        support_sync["uploads_issue_sync_report_artifact_on_failure"]
                    ),
                ],
                [
                    "Issue sync report ignores missing files",
                    ok_text(
                        support_sync["issue_sync_report_artifact_ignores_missing_files"]
                    ),
                ],
                [
                    "Issue sync report artifact retention",
                    ok_text(support_sync["issue_sync_report_artifact_retention"]),
                ],
                [
                    "Issue sync report upload after sync",
                    ok_text(support_sync["issue_sync_report_upload_after_sync"]),
                ],
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

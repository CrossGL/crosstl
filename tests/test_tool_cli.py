import configparser
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TOOLS = [
    "tools/ci_coverage.py",
    "tools/pytest_failure_summary.py",
    "tools/support_matrix.py",
    "tools/support_signals.py",
    "tools/support_ci_summary.py",
    "tools/sync_pr_issue_links.py",
    "tools/sync_support_issues.py",
]


@pytest.mark.parametrize(
    ("args", "expected_stdout"),
    [
        (
            (),
            (
                "usage:",
                "translate",
                "scan",
                "translate-project",
                "validate-project",
                "inspect-report",
                "plan-runtime",
                "runtime-manifest",
                "package-runtime",
                "report",
            ),
        ),
        (
            ("translate",),
            (
                "usage:",
                "--backend",
                "--output",
                "--source-backend",
                "--include-dir",
                "--define",
            ),
        ),
        (
            ("scan",),
            (
                "usage:",
                "--target",
                "--output",
                "--source-root",
                "--include-dir",
                "--source-override",
                "--variant",
            ),
        ),
        (
            ("report",),
            (
                "usage:",
                "--target",
                "--output",
                "--source-root",
                "--include-dir",
                "--source-override",
                "--variant",
            ),
        ),
        (
            ("translate-project",),
            (
                "usage:",
                "--target",
                "--output-dir",
                "--report",
                "--validate",
                "--run-toolchains",
                "--no-format",
                "--variant",
            ),
        ),
        (
            ("validate-project",),
            ("usage:", "--format", "--output", "--run-toolchains"),
        ),
        (
            ("inspect-report",),
            (
                "usage:",
                "--format",
                "--output",
                "--max-diagnostics",
                "--max-artifact-matrix-artifacts",
                "--max-runtime-references",
                "--run-toolchains",
            ),
        ),
        (
            ("plan-runtime",),
            (
                "usage:",
                "--format",
                "--output",
                "--max-runtime-references",
            ),
        ),
        (
            ("runtime-manifest",),
            (
                "usage:",
                "--format",
                "--output",
            ),
        ),
        (
            ("package-runtime",),
            (
                "usage:",
                "--package-dir",
                "--format",
                "--output",
            ),
        ),
    ],
)
def test_crosstl_module_cli_help_surface(args, expected_stdout):
    result = subprocess.run(
        [sys.executable, "-m", "crosstl", *args, "--help"],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert result.stderr == ""
    for text in expected_stdout:
        assert text in result.stdout


def test_support_tools_expose_cli_help():
    for tool in TOOLS:
        result = subprocess.run(
            [sys.executable, tool, "--help"],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            check=True,
        )

        assert "usage:" in result.stdout


def test_support_tools_keep_main_entrypoint():
    for tool in TOOLS:
        text = (ROOT / tool).read_text(encoding="utf-8")

        assert text.startswith("#!/usr/bin/env python3")
        assert 'if __name__ == "__main__":' in text
        assert "sys.exit(main())" in text


def test_pre_commit_config_keeps_repo_wide_formatters_and_safe_support_checks():
    text = (ROOT / ".pre-commit-config.yaml").read_text(encoding="utf-8")

    assert "end-of-file-fixer" in text
    assert "mixed-line-ending" in text
    assert "trailing-whitespace" in text
    assert "ruff critical Python checks" in text
    assert "https://github.com/asottile/pyupgrade" in text
    assert "https://github.com/PyCQA/isort" in text
    assert "python tools/support_matrix.py check" in text
    assert "python tools/support_signals.py extract" in text
    assert "python tools/support_signals.py check" not in text
    assert "python tools/support_signals.py update" not in text
    assert "python tools/support_matrix.py update" not in text
    assert "stages: [manual]" in text


def test_pull_request_template_mentions_support_traceability_marker():
    text = (ROOT / ".github" / "PULL_REQUEST_TEMPLATE.md").read_text(encoding="utf-8")

    assert "Support issue traceability: no issue closed" in text


def test_local_worker_worktrees_are_ignored_by_developer_tooling():
    gitignore = (ROOT / ".gitignore").read_text(encoding="utf-8")
    pre_commit = (ROOT / ".pre-commit-config.yaml").read_text(encoding="utf-8")
    config = configparser.ConfigParser()
    config.read(ROOT / "setup.cfg", encoding="utf-8")

    assert "worktrees/" in gitignore
    assert "worktrees/" in pre_commit
    assert "worktrees" in config["tool:pytest"]["norecursedirs"].split()


def test_public_docs_reference_project_porting_workflow():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    quickstart = (ROOT / "docs" / "source" / "quickstart.rst").read_text(
        encoding="utf-8"
    )

    for text in (readme, quickstart):
        assert "scan" in text
        assert "translate-project" in text
        assert "validate-project" in text
        assert "inspect-report" in text
        assert "--format text" in text
        assert "project-porting" in text


def test_public_docs_use_root_package_cli_entrypoint():
    docs = {
        "README.md": (ROOT / "README.md").read_text(encoding="utf-8"),
        "docs/source/quickstart.rst": (
            (ROOT / "docs" / "source" / "quickstart.rst").read_text(encoding="utf-8")
        ),
        "docs/source/project-porting.rst": (
            (ROOT / "docs" / "source" / "project-porting.rst").read_text(
                encoding="utf-8"
            )
        ),
        "support/README.md": (
            (ROOT / "support" / "README.md").read_text(encoding="utf-8")
        ),
    }

    for path, text in docs.items():
        assert "python -m crosstl._crosstl" not in text, path
        assert "python -m crosstl " in text, path


def test_public_api_docs_include_project_api():
    public_api = (ROOT / "docs" / "source" / "api" / "public.rst").read_text(
        encoding="utf-8"
    )

    assert ".. automodule:: crosstl.project" in public_api


def test_project_porting_guide_references_python_project_api():
    guide = (ROOT / "docs" / "source" / "project-porting.rst").read_text(
        encoding="utf-8"
    )

    assert (
        "from crosstl.project import inspect_project_report, translate_project" in guide
    )
    assert "report.write_json(report_path)" in guide
    assert "inspect_project_report(report_path)" in guide

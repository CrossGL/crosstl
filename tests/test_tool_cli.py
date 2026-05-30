import subprocess
import sys
from pathlib import Path

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

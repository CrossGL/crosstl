import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_root_package_module_entrypoint_exposes_cli_help():
    result = subprocess.run(
        [sys.executable, "-m", "crosstl", "--help"],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert result.stderr == ""
    assert result.stdout.startswith("usage: crosstl ")
    for text in (
        "translate",
        "scan",
        "translate-project",
        "validate-project",
        "inspect-report",
        "report",
    ):
        assert text in result.stdout

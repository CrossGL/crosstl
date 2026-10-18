"""Build and run the MSTest descriptor fixture shared by the demo test modules."""

import importlib.util
import os
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FIXTURE_CONTRACT = ROOT / "tests" / "fixture" / "coverage.json"
FIXTURE_POLICY = ROOT / "tests" / "fixture" / "evidence_policy.json"


def load_module(name, path):
    """Import one of the demo scripts, which are not an installed package."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_tools(tmp_path_factory):
    """Build the discovery exporter and the fixture assembly with the .NET SDK."""
    dotnet = shutil.which("dotnet")
    if dotnet is None:
        if os.environ.get("CROSSTL_REQUIRE_DOTNET") == "1":
            pytest.fail("The discovery contract requires the .NET 10 SDK")
        pytest.skip("The discovery contract requires the .NET 10 SDK")
    output = tmp_path_factory.mktemp("discovery-build")
    for name, project in (
        ("tool", ROOT / "discovery" / "Discovery.csproj"),
        ("fixture", ROOT / "tests" / "fixture" / "Fixture.csproj"),
    ):
        subprocess.run(
            [
                dotnet,
                "build",
                str(project),
                "-c",
                "Release",
                "-o",
                str(output / name),
                "--artifacts-path",
                str(output / (name + "-artifacts")),
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=180,
        )
    return [
        dotnet,
        str(output / "tool" / "Discovery.dll"),
        str(output / "fixture" / "Fixture.dll"),
    ]


def export(tools, output, mode=None):
    """Export the fixture inventory; ``mode`` selects a fixture failure."""
    environment = os.environ.copy()
    environment.pop("DISCOVERY_FIXTURE_MODE", None)
    if mode:
        environment["DISCOVERY_FIXTURE_MODE"] = mode
    return subprocess.run(
        [*tools, str(output)],
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
    )


def run_fixture(tools, root):
    """Export the fixture and execute it with MSTest; return (discovery, trx)."""
    discovery = root / "discovery"
    result = export(tools, discovery)
    assert result.returncode == 0, result.stderr
    result = subprocess.run(
        [
            tools[0],
            "vstest",
            tools[2],
            "--Logger:trx;LogFileName=baseline.trx",
            "--ResultsDirectory:" + str(root),
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return discovery, root / "baseline.trx"

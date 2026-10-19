"""Build and run the MSTest descriptor fixture shared by the demo test modules."""

import contextlib
import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FIXTURE_CONTRACT = ROOT / "tests" / "fixture" / "coverage.json"
FIXTURE_POLICY = ROOT / "tests" / "fixture" / "evidence_policy.json"
PROGRAM_CONTRACT = ROOT / "tests" / "program_fixture" / "coverage.json"
HOST_CONTRACT = ROOT / "tests" / "host_fixture" / "coverage.json"


def load_module(name, path):
    """Import one of the demo scripts, which are not an installed package."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    # Dataclasses resolve their annotations through the registered module.
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _dotnet():
    dotnet = shutil.which("dotnet")
    if dotnet is None:
        if os.environ.get("CROSSTL_REQUIRE_DOTNET") == "1":
            pytest.fail("The discovery contract requires the .NET 10 SDK")
        pytest.skip("The discovery contract requires the .NET 10 SDK")
    return dotnet


def _build_output(tmp_path_factory, name):
    """Return a build directory that every xdist worker of the session shares.

    Each worker otherwise runs its own concurrent ``dotnet build`` of the same
    projects, which multiplies the build cost and lets the SDK's shared state
    (NuGet caches, build servers) race between workers.
    """
    root = tmp_path_factory.getbasetemp()
    if os.environ.get("PYTEST_XDIST_WORKER"):
        root = root.parent
    output = root / "computesharp-dotnet" / name
    output.mkdir(parents=True, exist_ok=True)
    return output


@contextlib.contextmanager
def _exclusive(path):
    """Hold an exclusive advisory lock on ``path`` across processes."""
    with open(path, "a+b") as handle:
        if os.name == "nt":
            import msvcrt

            handle.seek(0)
            # LK_LOCK retries for about ten seconds; wait out longer builds.
            while True:
                try:
                    msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
                    break
                except OSError:
                    continue
            try:
                yield
            finally:
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _build(dotnet, project, output, name):
    """Build ``project`` into ``output/name`` once per test session."""
    marker = output / (name + ".built")
    with _exclusive(output / (name + ".lock")):
        if marker.exists():
            return
        result = subprocess.run(
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
            capture_output=True,
            text=True,
            timeout=180,
        )
        if result.returncode != 0:
            log = (result.stdout + result.stderr).strip().splitlines()
            pytest.fail(
                f"dotnet build {project.name} exited with {result.returncode}:\n"
                + "\n".join(log[-40:]),
                pytrace=False,
            )
        marker.touch()


def build_tools(tmp_path_factory):
    """Build the discovery exporter and the fixture assembly with the .NET SDK."""
    dotnet = _dotnet()
    output = _build_output(tmp_path_factory, "discovery-build")
    for name, project in (
        ("tool", ROOT / "discovery" / "Discovery.csproj"),
        ("fixture", ROOT / "tests" / "fixture" / "Fixture.csproj"),
    ):
        _build(dotnet, project, output, name)
    return [
        dotnet,
        str(output / "tool" / "Discovery.dll"),
        str(output / "fixture" / "Fixture.dll"),
    ]


def build_program(tools, tmp_path_factory):
    """Build the top-level program fixture; return the discovery tools for it."""
    output = _build_output(tmp_path_factory, "program-build")
    _build(
        tools[0],
        ROOT / "tests" / "program_fixture" / "ProgramFixture.csproj",
        output,
        "program",
    )
    return [tools[0], tools[1], str(output / "program" / "ProgramFixture.dll")]


def build_host(tools, tmp_path_factory):
    """Build the host-only MSTest fixture; return the discovery tools for it."""
    output = _build_output(tmp_path_factory, "host-build")
    _build(
        tools[0],
        ROOT / "tests" / "host_fixture" / "HostFixture.csproj",
        output,
        "host",
    )
    return [tools[0], tools[1], str(output / "host" / "HostFixture.dll")]


def vstest(tools, root, name, test_filter=None):
    """Run the fixture assembly of ``tools`` with MSTest into ``root/name.trx``."""
    command = [
        tools[0],
        "vstest",
        tools[2],
        f"--Logger:trx;LogFileName={name}.trx",
        "--ResultsDirectory:" + str(root),
    ]
    if test_filter:
        command.append("--TestCaseFilter:" + test_filter)
    result = subprocess.run(command, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    return root / f"{name}.trx"


def export(tools, output, mode=None, options=()):
    """Export the fixture inventory; ``mode`` selects a fixture failure."""
    environment = os.environ.copy()
    environment.pop("DISCOVERY_FIXTURE_MODE", None)
    if mode:
        environment["DISCOVERY_FIXTURE_MODE"] = mode
    return subprocess.run(
        [*tools, str(output), *options],
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

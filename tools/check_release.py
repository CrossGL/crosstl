#!/usr/bin/env python3
"""Check release archives and exercise an isolated installed distribution."""

import argparse
import ast
import configparser
import email.parser
import importlib.metadata
import importlib.resources
import json
import re
import subprocess
import sys
import tarfile
import tempfile
import zipfile
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def package_metadata(root):
    with (root / "pyproject.toml").open("rb") as handle:
        return tomllib.load(handle)["project"]


def package_version(root):
    return package_metadata(root)["version"]


def release_notes(root, version):
    lines = (root / "CHANGELOG.md").read_text(encoding="utf-8").splitlines()
    header = re.compile(r"^## \[" + re.escape(version) + r"\] - \d{4}-\d{2}-\d{2}$")
    positions = [i for i, line in enumerate(lines) if header.fullmatch(line)]
    _require(len(positions) == 1, "Expected exactly one dated release section")
    result = []
    for line in lines[positions[0] + 1 :]:
        if line.startswith("## [") or line == "---":
            break
        result.append(line)
    _require(any(line.startswith("- ") for line in result), "Release notes are empty")
    return "\n".join(result).strip() + "\n"


def check_versions(root):
    version = package_version(root)
    _require(re.fullmatch(r"\d+\.\d+\.\d+", version), "Invalid stable release version")
    citation = (root / "CITATION.cff").read_text(encoding="utf-8")
    _require(
        f'version: "{version}"' in citation.splitlines(), "Citation version differs"
    )
    docs = ast.parse((root / "docs/source/conf.py").read_text(encoding="utf-8"))
    releases = [
        ast.literal_eval(node.value)
        for node in docs.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "release"
            for target in node.targets
        )
    ]
    _require(releases == [version], "Documentation version differs")
    release_notes(root, version)
    changelog = (root / "CHANGELOG.md").read_text(encoding="utf-8")
    date = next(
        line.split(" - ", 1)[1]
        for line in changelog.splitlines()
        if line.startswith(f"## [{version}] - ")
    )
    _require(
        f'date-released: "{date}"' in citation.splitlines(),
        "Citation release date differs",
    )
    return version


def _metadata(data, version):
    metadata = email.parser.BytesParser().parsebytes(data)
    for key, expected in (
        ("Name", "crosstl"),
        ("Version", version),
        ("Requires-Python", ">=3.8"),
        ("License-Expression", "Apache-2.0"),
        ("License-File", "LICENSE"),
    ):
        _require(metadata.get_all(key) == [expected], f"Incorrect {key} metadata")


def _package_files(root):
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in (root / "crosstl").rglob("*")
        if path.is_file()
        and "__pycache__" not in path.parts
        and path.suffix not in (".pyc", ".pyo")
    }


def _check_payload(files, expected):
    payload = {
        name: data for name, data in files.items() if name.startswith("crosstl/")
    }
    _require(payload.keys() == expected.keys(), "Packaged source inventory differs")
    for name, data in expected.items():
        _require(payload[name] == data, f"Packaged source differs: {name}")


def check_distributions(root, directory):
    version = check_versions(root)
    wheel_name = f"crosstl-{version}-py3-none-any.whl"
    source_name = f"crosstl-{version}.tar.gz"
    _require(
        {p.name for p in directory.iterdir()} == {wheel_name, source_name},
        "Expected only the release wheel and source distribution",
    )
    expected = _package_files(root)
    _require(
        "crosstl/project/directx_runtime_worker.cpp" in expected,
        "Native worker is missing",
    )
    with zipfile.ZipFile(directory / wheel_name) as archive:
        names = archive.namelist()
        _require(len(names) == len(set(names)), "Duplicate wheel members")
        files = {name: archive.read(name) for name in names if not name.endswith("/")}
    _check_payload(files, expected)
    info = f"crosstl-{version}.dist-info/"
    _require(
        all(name.startswith(("crosstl/", info)) for name in files),
        "Unexpected wheel payload",
    )
    _metadata(files[info + "METADATA"], version)
    _require(
        files.get(info + "licenses/LICENSE") == (root / "LICENSE").read_bytes(),
        "Packaged license differs",
    )
    entrypoints = configparser.ConfigParser()
    entrypoints.read_string(files[info + "entry_points.txt"].decode("utf-8"))
    _require(
        dict(entrypoints["console_scripts"]) == {"crosstl": "crosstl._crosstl:main"},
        "CLI entry point differs",
    )
    with tarfile.open(directory / source_name) as archive:
        members = archive.getmembers()
        _require(
            len(members) == len({m.name for m in members}), "Duplicate source members"
        )
        prefix = f"crosstl-{version}/"
        _require(
            all(m.isdir() or m.isfile() for m in members),
            "Unexpected source member type",
        )
        _require(
            all(m.name.startswith(prefix) for m in members if m.isfile()),
            "Unexpected source root",
        )
        files = {
            m.name[len(prefix) :]: archive.extractfile(m).read()
            for m in members
            if m.isfile()
        }
    _check_payload(files, expected)
    _metadata(files["PKG-INFO"], version)
    for name in (
        "pyproject.toml",
        "README.md",
        "LICENSE",
        "CITATION.cff",
        "CHANGELOG.md",
    ):
        _require(
            files.get(name) == (root / name).read_bytes(),
            f"Source distribution differs: {name}",
        )
    return {
        "version": version,
        "packageFiles": len(expected),
        "distributions": [wheel_name, source_name],
    }


def check_installation(version):
    import crosstl
    from crosstl import project
    from crosstl._crosstl import translate

    _require(
        importlib.metadata.version("crosstl") == version, "Installed version differs"
    )
    installed = Path(crosstl.__file__).resolve()
    _require(
        Path(sys.prefix).resolve() in installed.parents,
        "Imports are not from the isolated environment",
    )
    _require(callable(project.translate_project), "Project API is missing")
    worker = importlib.resources.read_binary(
        "crosstl.project", "directx_runtime_worker.cpp"
    )
    _require(
        b"D3D12CreateDevice" in worker, "Packaged native runtime source is missing"
    )
    source = """shader PackageSmoke {
        vertex {
            vec4 main(vec4 position @POSITION) @gl_Position { return position; }
        }
    }"""
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "smoke.cgl"
        path.write_text(source, encoding="utf-8")
        for backend in ("directx", "opengl", "metal"):
            result = translate(str(path), backend=backend, format_output=False)
            _require(
                isinstance(result, str) and "position" in result,
                f"Empty {backend} translation",
            )
        console = Path(sys.executable).with_name(
            "crosstl.exe" if sys.platform == "win32" else "crosstl"
        )
        for command in (
            [str(console)],
            [sys.executable, "-m", "crosstl"],
            [sys.executable, "-m", "crosstl._crosstl"],
        ):
            result = subprocess.run(
                command + ["--help"],
                cwd=directory,
                capture_output=True,
                text=True,
                timeout=30,
                check=True,
            )
            _require(
                "translate-project" in result.stdout and "scan" in result.stdout,
                "Installed CLI is incomplete",
            )
    return {
        "version": version,
        "installed": str(installed),
        "targets": ["directx", "opengl", "metal"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    version = commands.add_parser("version")
    version.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    distributions = commands.add_parser("distributions")
    distributions.add_argument("directory", type=Path)
    distributions.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    installed = commands.add_parser("installed")
    installed.add_argument("--version", required=True)
    args = parser.parse_args()
    if args.command == "version":
        print(package_version(args.root))
        return
    result = (
        check_installation(args.version)
        if args.command == "installed"
        else check_distributions(args.root, args.directory)
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

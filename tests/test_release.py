import io
import json
import os
import shutil
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

import pytest
import yaml

from tools.check_release import check_distributions, check_versions, release_notes

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github/workflows/release.yml"
TAG_ONLY = "github.event_name == 'push' && startsWith(github.ref, 'refs/tags/')"


def _workflow():
    return yaml.load(WORKFLOW.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)


def _step(name):
    return next(
        step for step in _workflow()["jobs"]["build"]["steps"] if step["name"] == name
    )


def test_release_versions_and_notes_match():
    version = check_versions(ROOT)
    notes = release_notes(ROOT, version)
    assert "### " in notes and "## [" not in notes and "[Unreleased]" not in notes


def test_release_validates_pull_requests_without_publishing():
    workflow = _workflow()
    assert workflow["on"]["push"] == {"tags": ["v*.*.*"]}
    assert workflow["on"]["pull_request"]["branches"] == ["main"]
    assert "workflow_dispatch" in workflow["on"]
    assert set(workflow["jobs"]) == {"build", "publish-pypi", "github-release"}
    for job in workflow["jobs"].values():
        assert job["runs-on"] == "ubuntu-latest" and "continue-on-error" not in job
        assert "strategy" not in job
    build = workflow["jobs"]["build"]
    assert "if" not in build
    assert build["permissions"] == {"contents": "read", "actions": "read"}
    for job, dependency in (
        ("publish-pypi", "build"),
        ("github-release", "publish-pypi"),
    ):
        assert workflow["jobs"][job]["if"] == TAG_ONLY
        assert workflow["jobs"][job]["needs"] == dependency
    for name in (
        "Verify tag belongs to main and matches package version",
        "Verify required main CI succeeded",
    ):
        assert _step(name)["if"] == TAG_ONLY
    for step in build["steps"]:
        assert "continue-on-error" not in step
        assert "secrets." not in json.dumps(step)
    paths = workflow["on"]["pull_request"]["paths"]
    for path in (
        "crosstl/**",
        "setup.py",
        "MANIFEST.in",
        "pyproject.toml",
        "tools/check_release.py",
        "tests/test_release.py",
    ):
        assert path in paths


def test_release_checks_and_publishes_the_same_distributions():
    jobs = _workflow()["jobs"]
    steps = jobs["build"]["steps"]
    names = [step["name"] for step in steps]
    assert names.index("Build wheel and source distribution") < names.index(
        "Validate distribution metadata"
    )
    assert names.index("Verify release metadata and package contents") < names.index(
        "Upload distributions"
    )
    assert names.index("Test isolated wheel and source installations") < names.index(
        "Upload distributions"
    )
    assert (
        _step("Validate distribution metadata")["run"]
        == "python -m twine check --strict dist/*"
    )
    smoke = _step("Test isolated wheel and source installations")["run"]
    assert "for package in dist/*.whl dist/*.tar.gz" in smoke
    assert 'cd "$RUNNER_TEMP"' in smoke and "env -u PYTHONPATH" in smoke
    artifact = _step("Upload distributions")["with"]
    for job in ("publish-pypi", "github-release"):
        download = next(
            s for s in jobs[job]["steps"] if s["name"] == "Download distributions"
        )
        assert download["with"]["name"] == artifact["name"]
        assert download["with"]["path"] == artifact["path"] == "dist/"
    publish = next(
        s
        for s in jobs["publish-pypi"]["steps"]
        if "pypa/gh-action" in s.get("uses", "")
    )
    assert publish["with"]["password"] == "${{ secrets.PYPI_TOKEN }}"
    assert publish["with"]["packages-dir"] == "dist/"
    release = jobs["github-release"]["steps"][-1]["run"]
    assert "--verify-tag" in release and "dist/*" in release


@pytest.fixture
def release_tree(tmp_path):
    root = tmp_path / "source"
    files = {
        "setup.py": 'setup(version="1.2.3")\n',
        "CITATION.cff": 'version: "1.2.3"\ndate-released: "2026-10-07"\n',
        "docs/source/conf.py": 'release = "1.2.3"\n',
        "CHANGELOG.md": (
            "## [Unreleased]\n\n---\n\n## [1.2.3] - 2026-10-07\n\n### Fixed\n\n- A change.\n\n---\n## [1.2.2] - 2026-10-01\n- Older.\n"
        ),
        "pyproject.toml": '[build-system]\nrequires = ["setuptools"]\n',
        "README.md": "Package readme\n",
        "LICENSE": "License\n",
        "crosstl/__init__.py": "",
        "crosstl/project/directx_runtime_worker.cpp": "D3D12CreateDevice\n",
    }
    for name, content in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    return root


def _archives(root, directory, mutation=None):
    files = {
        p.relative_to(root).as_posix(): p.read_bytes()
        for p in root.rglob("*")
        if p.is_file()
    }
    metadata = b"Name: crosstl\nVersion: 1.2.3\nRequires-Python: >=3.8\n\n"
    wheel = {name: data for name, data in files.items() if name.startswith("crosstl/")}
    info = "crosstl-1.2.3.dist-info/"
    wheel[info + "METADATA"] = metadata
    wheel[info + "entry_points.txt"] = (
        b"[console_scripts]\ncrosstl = crosstl._crosstl:main\n"
    )
    files["PKG-INFO"] = metadata
    if mutation:
        mutation(wheel, files)
    directory.mkdir()
    with zipfile.ZipFile(directory / "crosstl-1.2.3-py3-none-any.whl", "w") as archive:
        for name, data in wheel.items():
            archive.writestr(name, data)
    with tarfile.open(directory / "crosstl-1.2.3.tar.gz", "w:gz") as archive:
        for name, data in files.items():
            entry = tarfile.TarInfo("crosstl-1.2.3/" + name)
            entry.size = len(data)
            archive.addfile(entry, io.BytesIO(data))


def test_release_archive_checks_accept_complete_payload(release_tree, tmp_path):
    directory = tmp_path / "dist"
    _archives(release_tree, directory)
    result = check_distributions(release_tree, directory)
    assert result["version"] == "1.2.3" and result["packageFiles"] == 2
    assert release_notes(release_tree, "1.2.3") == "### Fixed\n\n- A change.\n"


@pytest.mark.parametrize(
    "archive,name,data",
    [
        ("wheel", "crosstl/project/directx_runtime_worker.cpp", None),
        ("wheel", "crosstl/__init__.py", b"stale source"),
        ("wheel", "demos/private.py", b"unexpected"),
        (
            "wheel",
            "crosstl-1.2.3.dist-info/METADATA",
            b"Name: crosstl\nVersion: 1.2.2\nRequires-Python: >=3.8\n",
        ),
        (
            "wheel",
            "crosstl-1.2.3.dist-info/entry_points.txt",
            b"[console_scripts]\ncrosstl = wrong:main\n",
        ),
        ("source", "crosstl/project/directx_runtime_worker.cpp", None),
        ("source", "crosstl/__init__.py", b"stale source"),
        ("source", "pyproject.toml", b"wrong build requirements"),
        (
            "source",
            "PKG-INFO",
            b"Name: other\nVersion: 1.2.3\nRequires-Python: >=3.8\n",
        ),
    ],
)
def test_release_archive_checks_reject_incomplete_or_wrong_payload(
    release_tree, tmp_path, archive, name, data
):
    def mutate(wheel, source):
        files = wheel if archive == "wheel" else source
        if data is None:
            files.pop(name)
        else:
            files[name] = data

    directory = tmp_path / "dist"
    _archives(release_tree, directory, mutate)
    with pytest.raises(ValueError):
        check_distributions(release_tree, directory)


@pytest.mark.parametrize(
    "name,text",
    [
        ("CITATION.cff", 'version: "0.1.0"\n'),
        ("CITATION.cff", 'version: "1.2.3"\ndate-released: "2026-10-01"\n'),
        ("docs/source/conf.py", 'release = "0.1.0"\n'),
        ("CHANGELOG.md", "## [1.2.30] - 2026-10-07\n- Wrong version.\n"),
        ("CHANGELOG.md", "## [1.2.3] - 2026-10-07\n\n---\n"),
    ],
)
def test_release_metadata_must_agree(release_tree, name, text):
    (release_tree / name).write_text(text, encoding="utf-8")
    with pytest.raises(ValueError):
        check_versions(release_tree)


@pytest.mark.skipif(
    os.name == "nt" or not shutil.which("bash"), reason="Release gates run on Ubuntu"
)
@pytest.mark.parametrize(
    "status,success",
    [
        ("completed\tsuccess\turl", True),
        ("", False),
        ("in_progress\t\turl", False),
        ("completed\tfailure\turl", False),
        ("completed\tcancelled\turl", False),
        ("completed\tskipped\turl", False),
    ],
)
def test_release_gate_requires_successful_main_push_at_exact_commit(
    tmp_path, status, success
):
    gh = tmp_path / "gh"
    log = tmp_path / "calls.jsonl"
    gh.write_text(
        f"#!{sys.executable}\nimport json, os, sys\n"
        "with open(os.environ['CALLS'], 'a') as output: output.write(json.dumps(sys.argv[1:]) + '\\n')\n"
        "print(os.environ['RUN_STATUS'])\n",
        encoding="utf-8",
    )
    gh.chmod(0o755)
    env = dict(
        os.environ,
        PATH=str(tmp_path) + os.pathsep + os.environ["PATH"],
        CALLS=str(log),
        RUN_STATUS=status,
        GITHUB_SHA="exact-commit",
        GITHUB_REPOSITORY="CrossGL/crosstl",
    )
    result = subprocess.run(
        [
            "bash",
            "-e",
            "-o",
            "pipefail",
            "-c",
            _step("Verify required main CI succeeded")["run"],
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert (result.returncode == 0) == success, result.stderr
    calls = [json.loads(line) for line in log.read_text().splitlines()]
    assert len(calls) == (5 if success else 1)
    for call in calls:
        assert call[call.index("--commit") + 1] == "exact-commit"
        assert call[call.index("--branch") + 1] == "main"
        assert call[call.index("--event") + 1] == "push"


@pytest.mark.skipif(
    os.name == "nt" or not shutil.which("bash"),
    reason="Release extraction runs on Ubuntu",
)
def test_release_notes_shell_extraction_matches_checked_section(tmp_path):
    shutil.copyfile(ROOT / "CHANGELOG.md", tmp_path / "CHANGELOG.md")
    version = check_versions(ROOT)
    steps = _workflow()["jobs"]["github-release"]["steps"]
    command = next(
        step["run"]
        for step in steps
        if step["name"] == "Extract release notes from changelog"
    )
    subprocess.run(
        ["bash", "-e", "-o", "pipefail", "-c", command],
        cwd=tmp_path,
        env=dict(os.environ, GITHUB_REF_NAME="v" + version),
        check=True,
        timeout=15,
    )
    assert (tmp_path / "release-notes.md").read_text().strip() == release_notes(
        ROOT, version
    ).strip()


@pytest.mark.skipif(
    os.name == "nt" or not shutil.which("bash"), reason="Release gates run on Ubuntu"
)
@pytest.mark.parametrize(
    "comparison,tag,success",
    [
        ("identical", "v1.2.3", True),
        ("ahead", "v1.2.3", True),
        ("behind", "v1.2.3", False),
        ("diverged", "v1.2.3", False),
        ("unknown", "v1.2.3", False),
        ("identical", "v1.2.4", False),
    ],
)
def test_release_tag_must_match_version_and_belong_to_main(
    tmp_path, comparison, tag, success
):
    gh = tmp_path / "gh"
    gh.write_text(
        f"#!{sys.executable}\nimport os, sys\n"
        "assert sys.argv[1:] == ['api', 'repos/CrossGL/crosstl/compare/tag-commit...main', '--jq', '.status']\n"
        "print(os.environ['COMPARISON'])\n",
        encoding="utf-8",
    )
    python = tmp_path / "python"
    python.write_text(
        f"#!{sys.executable}\nimport sys\n"
        "if sys.argv[1:] == ['setup.py', '--version']: print('1.2.3')\n"
        "else: assert sys.argv[1:] == ['-m', 'pip', 'install', 'setuptools==75.3.2']\n",
        encoding="utf-8",
    )
    for path in (gh, python):
        path.chmod(0o755)
    result = subprocess.run(
        [
            "bash",
            "-e",
            "-o",
            "pipefail",
            "-c",
            _step("Verify tag belongs to main and matches package version")["run"],
        ],
        cwd=tmp_path,
        env=dict(
            os.environ,
            PATH=str(tmp_path) + os.pathsep + os.environ["PATH"],
            COMPARISON=comparison,
            GITHUB_SHA="tag-commit",
            GITHUB_REF_NAME=tag,
            GITHUB_REPOSITORY="CrossGL/crosstl",
        ),
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert (result.returncode == 0) == success, result.stderr

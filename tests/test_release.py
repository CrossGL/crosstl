import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import tarfile
import urllib.error
import zipfile
from pathlib import Path

import pytest
import yaml

from tools import check_release
from tools.check_release import (
    check_distributions,
    check_published_distributions,
    check_versions,
    package_metadata,
    release_notes,
    tomllib,
)

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


def test_package_build_contract_retains_supported_python_and_resources():
    config = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert config["build-system"] == {
        "requires": ["flit_core==4.1.0"],
        "build-backend": "flit_core.buildapi",
    }
    metadata = package_metadata(ROOT)
    assert metadata["name"] == "crosstl"
    assert metadata["requires-python"] == ">=3.8"
    assert metadata["dependencies"] == ["gast", "tomli; python_version<'3.11'"]
    assert metadata["scripts"] == {"crosstl": "crosstl._crosstl:main"}
    assert metadata["license"] == "Apache-2.0"
    assert metadata["license-files"] == ["LICENSE"]
    assert set(config["tool"]["flit"]) == {"sdist"}
    assert not (ROOT / "setup.py").exists()
    assert not (ROOT / "MANIFEST.in").exists()
    source = config["tool"]["flit"]["sdist"]
    assert set(source["include"]) == {
        "CHANGELOG.md",
        "CITATION.cff",
        ".zenodo.json",
        "examples/",
        "support/",
        "docs/",
    }
    assert set(source["exclude"]) == {
        "docs/_build/",
        "docs/doxygen/build/",
        "examples/output/",
        "support/generated/",
        "**/__pycache__/",
        "**/*.py[cod]",
    }


def test_release_build_and_publisher_support_current_metadata():
    assert _step("Install release tooling")["run"] == (
        "python -m pip install build==1.2.2.post1 twine==7.0.0"
    )
    publisher = _workflow()["jobs"]["publish-pypi"]["steps"][-1]
    assert publisher["uses"] == (
        "pypa/gh-action-pypi-publish@dc37677b2e1c63e2034f94d8a5b11f265b73ba33"
    )
    assert _step("Set up minimum supported Python")["with"]["python-version"] == "3.8"
    smoke = _step("Test isolated wheel and source installations")["run"]
    assert "${{ steps.python-minimum.outputs.python-path }}" in smoke
    assert '"$(command -v python)"' in smoke
    assert '"$interpreter" -m venv "$environment"' in smoke
    assert "setup.py" not in WORKFLOW.read_text(encoding="utf-8")
    assert "setuptools" not in WORKFLOW.read_text(encoding="utf-8")


def test_release_version_command_reads_only_declarative_metadata(release_tree):
    (release_tree / "setup.py").write_text('raise RuntimeError("must not execute")\n')
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools/check_release.py"),
            "version",
            "--root",
            str(release_tree),
        ],
        cwd=release_tree,
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    assert result.stdout == "1.2.3\n" and not result.stderr


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
    assert publish["with"]["attestations"] == "false"
    release = jobs["github-release"]["steps"][-1]["run"]
    assert "--verify-tag" in release and "dist/*" in release
    steps = jobs["github-release"]["steps"]
    names = [step["name"] for step in steps]
    verify = names.index("Verify published PyPI archive identities")
    assert names.index("Download distributions") < verify
    assert verify < names.index("Create or update GitHub Release with distributions")
    assert steps[verify]["run"] == "python tools/check_release.py published dist"
    assert "if" not in steps[verify] and "continue-on-error" not in steps[verify]
    python = steps[names.index("Set up Python 3.12")]
    assert python["with"]["python-version"] == "3.12"


@pytest.fixture
def release_tree(tmp_path):
    root = tmp_path / "source"
    files = {
        "CITATION.cff": 'version: "1.2.3"\ndate-released: "2026-10-07"\n',
        "docs/source/conf.py": 'release = "1.2.3"\n',
        "CHANGELOG.md": (
            "## [Unreleased]\n\n---\n\n## [1.2.3] - 2026-10-07\n\n### Fixed\n\n- A change.\n\n---\n## [1.2.2] - 2026-10-01\n- Older.\n"
        ),
        "pyproject.toml": '[project]\nname = "crosstl"\nversion = "1.2.3"\n',
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
    metadata = (
        b"Name: crosstl\nVersion: 1.2.3\nRequires-Python: >=3.8\n"
        b"License-Expression: Apache-2.0\nLicense-File: LICENSE\n\n"
    )
    wheel = {name: data for name, data in files.items() if name.startswith("crosstl/")}
    info = "crosstl-1.2.3.dist-info/"
    wheel[info + "METADATA"] = metadata
    wheel[info + "licenses/LICENSE"] = files["LICENSE"]
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


@pytest.fixture
def published_release(release_tree, tmp_path):
    directory = tmp_path / "dist"
    _archives(release_tree, directory)
    metadata = {
        "info": {"name": "crosstl", "version": "1.2.3"},
        "urls": [
            {
                "filename": path.name,
                "size": path.stat().st_size,
                "digests": {"sha256": hashlib.sha256(path.read_bytes()).hexdigest()},
                "yanked": False,
            }
            for path in sorted(directory.iterdir())
        ],
    }
    return directory, metadata


def _pypi_responses(monkeypatch, responses):
    calls, sleeps = [], []
    pending = iter(responses)

    def open_url(request, *, timeout):
        assert request.full_url == "https://pypi.org/pypi/crosstl/1.2.3/json"
        assert request.get_header("Accept") == "application/json"
        assert timeout == 20
        calls.append(request.full_url)
        response = next(pending)
        if isinstance(response, Exception):
            raise response
        return io.BytesIO(json.dumps(response).encode("utf-8"))

    monkeypatch.setattr(check_release.urllib.request, "urlopen", open_url)
    monkeypatch.setattr(check_release.time, "sleep", sleeps.append)
    return calls, sleeps


def test_published_archives_match_local_validated_bytes(
    release_tree, published_release, monkeypatch
):
    directory, metadata = published_release
    calls, sleeps = _pypi_responses(monkeypatch, [metadata])
    result = check_published_distributions(release_tree, directory)
    assert len(calls) == 1 and sleeps == []
    assert result["version"] == "1.2.3"
    for item in metadata["urls"]:
        assert result["published"][item["filename"]] == {
            "sha256": item["digests"]["sha256"],
            "sizeBytes": item["size"],
        }


def test_published_command_checks_validated_distribution_identities(
    release_tree, published_release, monkeypatch, capsys
):
    directory, metadata = published_release
    _pypi_responses(monkeypatch, [metadata])
    monkeypatch.setattr(
        sys,
        "argv",
        ["check_release.py", "published", str(directory), "--root", str(release_tree)],
    )
    check_release.main()
    result = json.loads(capsys.readouterr().out)
    assert result["version"] == "1.2.3"
    assert set(result["published"]) == {item["filename"] for item in metadata["urls"]}


@pytest.mark.parametrize(
    "mutation",
    ["name", "version", "sha256", "size", "yanked", "duplicate", "unexpected"],
)
def test_published_archive_mismatches_fail_without_retry(
    release_tree, published_release, monkeypatch, mutation
):
    directory, metadata = published_release
    first = metadata["urls"][0]
    if mutation in ("name", "version"):
        metadata["info"][mutation] = "different"
    elif mutation == "sha256":
        first["digests"]["sha256"] = "0" * 64
    elif mutation == "size":
        first["size"] += 1
    elif mutation == "yanked":
        first["yanked"] = True
    elif mutation == "duplicate":
        metadata["urls"].append(first.copy())
    else:
        first["filename"] = "unexpected.whl"
    calls, sleeps = _pypi_responses(monkeypatch, [metadata])
    with pytest.raises(ValueError, match="PyPI"):
        check_published_distributions(release_tree, directory)
    assert len(calls) == 1 and sleeps == []


@pytest.mark.parametrize("code", [404, 429, 500, 503, "timeout", "network", "missing"])
def test_published_archives_retry_index_visibility_and_transient_errors(
    release_tree, published_release, monkeypatch, code
):
    directory, metadata = published_release
    if code == "missing":
        first = dict(metadata, urls=metadata["urls"][:1])
    elif code == "timeout":
        first = TimeoutError("timeout")
    elif code == "network":
        first = urllib.error.URLError("connection failed")
    else:
        first = urllib.error.HTTPError("url", code, "unavailable", None, io.BytesIO())
    calls, sleeps = _pypi_responses(monkeypatch, [first, metadata])
    assert check_published_distributions(release_tree, directory)["version"] == "1.2.3"
    assert len(calls) == 2 and sleeps == [10]


def test_published_archive_retry_limit_fails_closed(
    release_tree, published_release, monkeypatch
):
    directory, metadata = published_release
    incomplete = dict(metadata, urls=[])
    calls, sleeps = _pypi_responses(monkeypatch, [incomplete] * 6)
    with pytest.raises(ValueError, match="retry limit"):
        check_published_distributions(release_tree, directory)
    assert len(calls) == 6 and sleeps == [10] * 5


@pytest.mark.parametrize(
    "metadata",
    [
        None,
        {},
        {"info": []},
        {"info": {"name": "crosstl", "version": "1.2.3"}, "urls": None},
        {"info": {"name": "crosstl", "version": "1.2.3"}, "urls": [None]},
        {"info": {"name": "crosstl", "version": "1.2.3"}, "urls": [{}]},
    ],
)
def test_published_archive_malformed_metadata_fails_without_retry(
    release_tree, published_release, monkeypatch, metadata
):
    directory, _ = published_release
    calls, sleeps = _pypi_responses(monkeypatch, [metadata])
    with pytest.raises(ValueError, match="PyPI"):
        check_published_distributions(release_tree, directory)
    assert len(calls) == 1 and sleeps == []


@pytest.mark.parametrize("code", [400, 401, 403])
def test_published_archive_permanent_http_errors_fail_immediately(
    release_tree, published_release, monkeypatch, code
):
    directory, _ = published_release
    calls, sleeps = _pypi_responses(
        monkeypatch,
        [urllib.error.HTTPError("url", code, "rejected", None, io.BytesIO())],
    )
    with pytest.raises(urllib.error.HTTPError):
        check_published_distributions(release_tree, directory)
    assert len(calls) == 1 and sleeps == []


def test_published_archives_check_local_payload_before_network(
    release_tree, published_release, monkeypatch
):
    directory, metadata = published_release
    (directory / "unexpected.txt").write_text("unexpected", encoding="utf-8")
    calls, sleeps = _pypi_responses(monkeypatch, [metadata])
    with pytest.raises(ValueError, match="Expected only"):
        check_published_distributions(release_tree, directory)
    assert calls == sleeps == []


@pytest.mark.parametrize(
    "archive,name,data",
    [
        ("wheel", "crosstl/project/directx_runtime_worker.cpp", None),
        ("wheel", "crosstl/__init__.py", b"stale source"),
        ("wheel", "demos/private.py", b"unexpected"),
        ("wheel", "crosstl-1.2.3.dist-info/licenses/LICENSE", None),
        ("wheel", "crosstl-1.2.3.dist-info/licenses/LICENSE", b"wrong license"),
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
@pytest.mark.parametrize(
    "workflow",
    (
        "full-tests.yml",
        "native-host-loader.yml",
        "deferred-native-compilation.yml",
        "demo-project-testing.yml",
    ),
)
def test_release_gate_requires_successful_main_push_at_exact_commit(
    tmp_path, status, success, workflow
):
    required = (
        "full-tests.yml",
        "backend-tests.yml",
        "translator-tests.yml",
        "docs.yml",
        "examples-test.yml",
        "native-host-loader.yml",
        "deferred-native-compilation.yml",
        "demo-project-testing.yml",
    )
    gh = tmp_path / "gh"
    log = tmp_path / "calls.jsonl"
    gh.write_text(
        f"#!{sys.executable}\nimport json, os, sys\n"
        "with open(os.environ['CALLS'], 'a') as output: output.write(json.dumps(sys.argv[1:]) + '\\n')\n"
        "workflow = sys.argv[sys.argv.index('--workflow') + 1]\n"
        "print(os.environ['RUN_STATUS'] if workflow == os.environ['WORKFLOW'] else 'completed\\tsuccess\\turl')\n",
        encoding="utf-8",
    )
    gh.chmod(0o755)
    env = dict(
        os.environ,
        PATH=str(tmp_path) + os.pathsep + os.environ["PATH"],
        CALLS=str(log),
        RUN_STATUS=status,
        WORKFLOW=workflow,
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
    checked = required if success else required[: required.index(workflow) + 1]
    assert [call[call.index("--workflow") + 1] for call in calls] == list(checked)
    for call in calls:
        assert call[call.index("--commit") + 1] == "exact-commit"
        assert call[call.index("--branch") + 1] == "main"
        assert call[call.index("--event") + 1] == "push"


def test_native_release_gates_run_for_package_version_changes():
    for name in (
        "native-host-loader.yml",
        "deferred-native-compilation.yml",
        "demo-project-testing.yml",
    ):
        workflow = yaml.load(
            (ROOT / ".github" / "workflows" / name).read_text(encoding="utf-8"),
            Loader=yaml.BaseLoader,
        )
        for event in ("push", "pull_request"):
            assert "main" in workflow["on"][event]["branches"]
            assert "pyproject.toml" in workflow["on"][event]["paths"]


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
        "assert sys.argv[1:] == ['tools/check_release.py', 'version']\n"
        "print('1.2.3')\n",
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

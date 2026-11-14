import argparse
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.pytest_shard import parse_shard, shard_of  # noqa: E402


def test_parse_shard_reads_index_and_count():
    assert parse_shard("1/3") == (1, 3)
    assert parse_shard("3/3") == (3, 3)


@pytest.mark.parametrize("value", ["0/3", "4/3", "2", "a/3", "1/b"])
def test_parse_shard_rejects_malformed_values(value):
    with pytest.raises(argparse.ArgumentTypeError):
        parse_shard(value)


def test_shard_of_is_stable_and_in_range():
    paths = [f"tests/test_{index}.py" for index in range(50)]
    shards = [shard_of(path, 3) for path in paths]
    assert set(shards) == {1, 2, 3}
    assert shards == [shard_of(path, 3) for path in paths]


def _collect(project, *args):
    env = dict(os.environ, PYTHONPATH=str(ROOT))
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-p",
            "tools.pytest_shard",
            "-p",
            "no:cacheprovider",
        ]
        + ["--collect-only", "-q", *args],
        cwd=project,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode in (0, 5), result.stdout + result.stderr
    return {line for line in result.stdout.splitlines() if "::" in line}


def test_shards_run_every_test_once_and_split_large_files(tmp_path):
    for index in range(4):
        (tmp_path / f"test_mod{index}.py").write_text(
            "import pytest\n\n"
            "@pytest.mark.parametrize('n', range(30))\n"
            "def test_case(n):\n"
            "    pass\n",
            encoding="utf-8",
        )
    everything = _collect(tmp_path)
    assert len(everything) == 120

    shards = [_collect(tmp_path, "--shard", f"{index}/3") for index in (1, 2, 3)]
    assert set().union(*shards) == everything
    assert sum(len(shard) for shard in shards) == len(everything)
    for shard in shards:
        files = {nodeid.split("::", 1)[0] for nodeid in shard}
        assert len(files) == 4
        assert 20 <= len(shard) <= 60


def test_tests_sharing_a_module_fixture_stay_in_one_shard(tmp_path):
    (tmp_path / "test_shared.py").write_text(
        "import pytest\n\n"
        "@pytest.fixture(scope='module')\n"
        "def helper():\n"
        "    return object()\n\n"
        "@pytest.mark.parametrize('n', range(30))\n"
        "def test_uses_helper(helper, n):\n"
        "    pass\n\n"
        "@pytest.mark.parametrize('n', range(30))\n"
        "def test_alone(n):\n"
        "    pass\n",
        encoding="utf-8",
    )
    shards = [_collect(tmp_path, "--shard", f"{index}/3") for index in (1, 2, 3)]
    assert sum(len(shard) for shard in shards) == 60
    holders = [
        shard for shard in shards if any("test_uses_helper" in node for node in shard)
    ]
    assert len(holders) == 1
    assert sum("test_uses_helper" in node for node in holders[0]) == 30
    assert all(any("test_alone" in node for node in shard) for shard in shards)


def test_complete_suite_runs_one_job_per_shard():
    workflow = yaml.safe_load(
        (ROOT / ".github" / "workflows" / "full-tests.yml").read_text(encoding="utf-8")
    )
    job = workflow["jobs"]["pytest"]
    assert job["strategy"]["fail-fast"] is False
    shards = job["strategy"]["matrix"]["shard"]
    assert shards == list(range(1, len(shards) + 1))
    command = next(
        step["run"]
        for step in job["steps"]
        if step.get("name") == "Run complete test suite"
    )
    assert "-p tools.pytest_shard" in command
    assert '--shard "${{ matrix.shard }}/${{ strategy.job-total }}"' in command
    # Test times range from milliseconds to minutes; idle workers take queued tests.
    assert "-n auto --dist worksteal" in command


def test_translator_shard_legs_cover_every_shard():
    workflow = yaml.safe_load(
        (ROOT / ".github" / "workflows" / "translator-tests.yml").read_text(
            encoding="utf-8"
        )
    )
    job = workflow["jobs"]["test"]
    matrix = job["strategy"]["matrix"]
    legs = [(name, "ubuntu-latest") for name in matrix["component"]]
    legs += [(entry["component"], entry["OS"]) for entry in matrix["include"]]
    sharded = {}
    for name, runner in legs:
        if "-of-" in name:
            suite, _, spec = name.partition("-")
            index, _, count = spec.partition("-of-")
            sharded.setdefault((suite, runner), set()).add((int(index), int(count)))
    assert set(sharded) == {
        ("general", "ubuntu-latest"),
        ("all", "windows-latest"),
        ("all", "macOS-latest"),
    }
    for parts in sharded.values():
        counts = {count for _, count in parts}
        assert len(counts) == 1
        assert {index for index, _ in parts} == set(range(1, counts.pop() + 1))
    assert "general" not in matrix["component"]
    # A whole leg next to the shards of the same suite would run it twice.
    assert not {(name, runner) for name, runner in legs} & set(sharded)
    command = next(
        step["run"]
        for step in job["steps"]
        if step.get("name") == "Run translator tests"
    )
    assert "-p tools.pytest_shard" in command

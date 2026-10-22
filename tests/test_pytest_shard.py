import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.pytest_shard import TIMED_SECONDS, parse_shard, shard_of  # noqa: E402


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


def _collect_order(project, *args):
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
    return [line for line in result.stdout.splitlines() if "::" in line]


def _collect(project, *args):
    return set(_collect_order(project, *args))


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


def _run(project, *args):
    env = dict(os.environ, PYTHONPATH=str(ROOT))
    return subprocess.run(
        [sys.executable, "-m", "pytest", "-p", "tools.pytest_shard"]
        + ["-p", "no:cacheprovider", "-q", *args],
        cwd=project,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def _record(directory, name, tests, short_count, short_seconds):
    directory.mkdir(exist_ok=True)
    data = {"tests": tests, "short": {"count": short_count, "seconds": short_seconds}}
    (directory / name).write_text(json.dumps(data), encoding="utf-8")


def _parametrized(project, name, count, prelude=""):
    (project / f"{name}.py").write_text(
        "import pytest\n\n"
        f"{prelude}"
        f"@pytest.mark.parametrize('n', range({count}))\n"
        f"def {name}(n):\n"
        "    pass\n",
        encoding="utf-8",
    )


def test_recorded_durations_balance_the_shards_and_start_long_tests(tmp_path):
    _parametrized(tmp_path, "test_fast", 40)
    _parametrized(tmp_path, "test_slow", 4)
    recorded = tmp_path / "recorded"
    _record(recorded, "a.json", {"test_slow.py::test_slow[0]": 10.0}, 30, 0.3)
    _record(
        recorded,
        "b.json",
        {f"test_slow.py::test_slow[{n}]": 10.0 - n for n in (1, 2, 3)},
        10,
        0.1,
    )
    everything = _collect(tmp_path)
    shards = [
        _collect_order(
            tmp_path, "--shard", f"{index}/2", "--shard-durations", "recorded"
        )
        for index in (1, 2)
    ]
    assert set().union(*shards) == everything
    assert sum(len(shard) for shard in shards) == len(everything)
    # Longest first onto the shard with less work: 10 + 7 and 9 + 8 seconds.
    assert shards[0][:2] == ["test_slow.py::test_slow[0]", "test_slow.py::test_slow[3]"]
    assert shards[1][:2] == ["test_slow.py::test_slow[1]", "test_slow.py::test_slow[2]"]
    assert [len(shard) for shard in shards] == [22, 22]


def test_recorded_durations_keep_a_shared_fixture_file_together(tmp_path):
    (tmp_path / "test_shared.py").write_text(
        "import pytest\n\n"
        "@pytest.fixture(scope='module')\n"
        "def helper():\n"
        "    return object()\n\n"
        "@pytest.mark.parametrize('n', range(10))\n"
        "def test_uses_helper(helper, n):\n"
        "    pass\n\n"
        "@pytest.mark.parametrize('n', range(10))\n"
        "def test_alone(n):\n"
        "    pass\n",
        encoding="utf-8",
    )
    recorded = tmp_path / "recorded"
    tests = {f"test_shared.py::test_uses_helper[{n}]": 1 + n / 10 for n in range(10)}
    tests.update({f"test_shared.py::test_alone[{n}]": 2.0 for n in range(10)})
    _record(recorded, "all.json", tests, 0, 0.0)
    shards = [
        _collect_order(
            tmp_path, "--shard", f"{index}/2", "--shard-durations", "recorded"
        )
        for index in (1, 2)
    ]
    assert sum(len(shard) for shard in shards) == 20
    holders = [shard for shard in shards if any("helper" in node for node in shard)]
    assert len(holders) == 1
    # The file weighs 14.5 s, so one 2 s test joins it. That test moves ahead; the
    # file's own tests keep their order although each of them takes over 1 s.
    assert holders[0][0].startswith("test_shared.py::test_alone[")
    assert holders[0][1:] == [
        f"test_shared.py::test_uses_helper[{n}]" for n in range(10)
    ]


@pytest.mark.parametrize("workers", [None, "2"])
def test_recorded_durations_list_slow_tests_and_count_the_rest(tmp_path, workers):
    if workers:
        pytest.importorskip("xdist")
    (tmp_path / "test_timed.py").write_text(
        "import time\n\n"
        "def test_slow():\n"
        "    time.sleep(0.3)\n\n"
        "def test_fast_a():\n"
        "    pass\n\n"
        "def test_fast_b():\n"
        "    pass\n",
        encoding="utf-8",
    )
    out = tmp_path / "recorded" / "shard-1.json"
    extra = ["-n", workers] if workers else []
    result = _run(tmp_path, "--shard-durations-out", str(out), *extra)
    assert result.returncode == 0, result.stdout + result.stderr
    data = json.loads(out.read_text(encoding="utf-8"))
    assert list(data["tests"]) == ["test_timed.py::test_slow"]
    assert data["tests"]["test_timed.py::test_slow"] >= 0.3
    assert data["short"]["count"] == 2
    assert 0 <= data["short"]["seconds"] < 2 * TIMED_SECONDS
    order = _collect_order(
        tmp_path, "--shard", "1/1", "--shard-durations", str(out.parent)
    )
    assert order == [
        "test_timed.py::test_slow",
        "test_timed.py::test_fast_a",
        "test_timed.py::test_fast_b",
    ]


def test_recorded_durations_directory_must_hold_a_recording(tmp_path):
    _parametrized(tmp_path, "test_fast", 2)
    (tmp_path / "recorded").mkdir()
    result = _run(
        tmp_path, "--collect-only", "--shard", "1/2", "--shard-durations", "recorded"
    )
    assert result.returncode == pytest.ExitCode.USAGE_ERROR
    assert "no *.json files" in result.stdout + result.stderr


RECORDING_EVENTS = (
    "github.event_name == 'push' || github.event_name == 'workflow_dispatch'"
)


def _step(job, name):
    return next(step for step in job["steps"] if step.get("name") == name)


def test_complete_suite_balances_shards_by_the_durations_main_recorded():
    workflow = yaml.safe_load(
        (ROOT / ".github" / "workflows" / "full-tests.yml").read_text(encoding="utf-8")
    )
    jobs = workflow["jobs"]
    (find,) = jobs["durations"]["steps"]
    assert find["uses"].startswith("actions/cache/restore@")
    assert find["with"]["lookup-only"] is True
    assert find["with"]["restore-keys"] == "pytest-durations-"
    output = "${{ steps." + find["id"] + ".outputs.cache-matched-key }}"
    assert jobs["durations"]["outputs"]["key"] == output
    path = find["with"]["path"]

    shards = jobs["pytest"]
    assert shards["needs"] == "durations"
    names = [step.get("name") for step in shards["steps"]]
    restore = _step(shards, "Restore recorded test durations")
    assert names.index(restore["name"]) < names.index("Run complete test suite")
    assert restore["if"] == "needs.durations.outputs.key != ''"
    # Every shard restores the one key the durations job found, or fails.
    assert restore["with"] == {
        "path": path,
        "key": "${{ needs.durations.outputs.key }}",
        "fail-on-cache-miss": True,
    }
    run = _step(shards, "Run complete test suite")
    assert run["env"]["DURATIONS_KEY"] == "${{ needs.durations.outputs.key }}"
    assert f"balance=(--shard-durations {path})" in run["run"]
    assert '"${balance[@]}" --shard-durations-out' in run["run"]
    upload = _step(shards, "Upload recorded test durations")
    assert upload["if"] == RECORDING_EVENTS
    assert upload["with"]["name"] == "pytest-durations-${{ matrix.shard }}"

    save = jobs["save-durations"]
    assert save["needs"] == "pytest"
    assert save["if"] == RECORDING_EVENTS
    download, store = save["steps"]
    assert download["with"] == {
        "pattern": "pytest-durations-*",
        "path": path,
        "merge-multiple": True,
    }
    assert store["uses"].startswith("actions/cache/save@")
    assert store["with"]["path"] == path
    assert store["with"]["key"].startswith("pytest-durations-${{ github.run_id }}")


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
    assert set(sharded) == {("general", "ubuntu-latest"), ("all", "windows-latest")}
    for parts in sharded.values():
        counts = {count for _, count in parts}
        assert len(counts) == 1
        assert {index for index, _ in parts} == set(range(1, counts.pop() + 1))
    assert "general" not in matrix["component"]
    command = next(
        step["run"]
        for step in job["steps"]
        if step.get("name") == "Run translator tests"
    )
    assert "-p tools.pytest_shard" in command

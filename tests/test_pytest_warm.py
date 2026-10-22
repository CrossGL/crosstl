import os
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.pytest_warm import find_test_modules, split_by_size  # noqa: E402


def _write_modules(project: Path, sizes: list[int]) -> list[Path]:
    modules = []
    for index, padding in enumerate(sizes):
        module = project / "pkg" / f"test_mod{index}.py"
        module.parent.mkdir(exist_ok=True)
        module.write_text(
            f"# {'x' * padding}\n"
            "import pytest\n\n"
            "@pytest.mark.parametrize('n', range(3))\n"
            "def test_case(n):\n"
            "    assert n < 3\n",
            encoding="utf-8",
        )
        modules.append(module.resolve())
    (project / "pkg" / "helper.py").write_text("VALUE = 1\n", encoding="utf-8")
    (project / "pkg" / "check_test.py").write_text(
        "def test_suffix():\n    assert True\n", encoding="utf-8"
    )
    return modules + [(project / "pkg" / "check_test.py").resolve()]


def _pytest(project: Path, *args: str) -> subprocess.CompletedProcess:
    env = dict(os.environ, PYTHONPATH=str(ROOT))
    return subprocess.run(
        [sys.executable, "-m", *args],
        cwd=project,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def test_find_test_modules_matches_the_default_python_files(tmp_path):
    modules = _write_modules(tmp_path, [0, 0])
    assert find_test_modules([str(tmp_path)]) == sorted(modules)
    single = str(modules[0])
    assert find_test_modules([single]) == [modules[0]]


def test_split_by_size_is_a_balanced_partition(tmp_path):
    modules = _write_modules(tmp_path, [4000, 3000, 2000, 1000, 1000, 500, 0])
    slices = split_by_size(modules, 3)
    assert len(slices) == 3
    assert set().union(*slices) == set(modules)
    assert sum(len(part) for part in slices) == len(modules)
    totals = [sum(module.stat().st_size for module in part) for part in slices]
    assert max(totals) - min(totals) <= max(m.stat().st_size for m in modules)
    assert split_by_size(list(reversed(modules)), 3) == slices


def test_slices_collect_every_test_module_once(tmp_path):
    modules = _write_modules(tmp_path, [900, 600, 300, 0])
    collected = []
    for index in (1, 2, 3):
        result = _pytest(
            tmp_path,
            *("pytest", "-p", "tools.pytest_warm", "-p", "no:cacheprovider"),
            *("--collect-only", "-q", "--warm-slice", f"{index}/3", "pkg"),
        )
        assert result.returncode in (0, 5), result.stdout + result.stderr
        collected.append(
            {line.split("::")[0] for line in result.stdout.splitlines() if "::" in line}
        )
    names = {f"pkg/{module.name}" for module in modules}
    assert set().union(*collected) == names
    assert sum(len(part) for part in collected) == len(names)


def test_warm_writes_the_rewritten_module_of_every_test(tmp_path):
    modules = _write_modules(tmp_path, [900, 600, 300, 0])
    result = _pytest(tmp_path, "tools.pytest_warm", "-j", "2", "pkg")
    assert result.returncode == 0, result.stdout + result.stderr
    for module in modules:
        cached = list(
            (module.parent / "__pycache__").glob(f"{module.stem}.*-pytest-*.pyc")
        )
        assert cached, module.name
    assert not list((tmp_path / "pkg" / "__pycache__").glob("helper.*"))


def test_complete_suite_warms_the_cache_before_the_shard_runs():
    workflow = yaml.safe_load(
        (ROOT / ".github" / "workflows" / "full-tests.yml").read_text(encoding="utf-8")
    )
    names = [step.get("name") for step in workflow["jobs"]["pytest"]["steps"]]
    warm = names.index("Rewrite test modules")
    assert (
        names.index("Install dependencies")
        < warm
        < names.index("Run complete test suite")
    )

import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
import yaml

from demos.integrations.mlx.portable_host import lazy_package_evidence as lazy
from demos.integrations.mlx.portable_host import upstream_matrix_packages as warm
from demos.integrations.mlx.portable_host import verify_upstream_ops as ops

GEMM = (
    "steel_gemm_fused_nn_float32_float32_bm64_bn64_bk16_wm2_wn2_has_batch_n"
    "_use_out_source_n_do_axpby_n_align_M_n_align_N_n_align_K_t"
)
GEMV = "gemv_float32_bm1_bn1_sm8_sn4_tm4_tn4_nc0_axpby0"
OFFSETS = "gather_mm_offsets"


def test_recorded_entries_are_sorted_matrix_entries_of_known_groups(tmp_path):
    assert set(warm.ENTRIES) <= set(ops.GROUPS)
    assert set(warm.ORDER) == set(lazy.MATRIX_FAMILIES)
    caches = lazy.LazyPackages(tmp_path / "mlx", tmp_path / "packages", "metal").caches
    for entries in warm.ENTRIES.values():
        assert entries and list(entries) == sorted(set(entries))
        for entry in warm.ordered(entries):
            # Each entry parses as its family's cache would before translating.
            caches[lazy.family(entry)].signature(entry)
    assert not list(tmp_path.iterdir())


def test_entries_translate_slowest_family_first():
    assert warm.ordered([GEMV, OFFSETS, GEMM]) == [GEMM, OFFSETS, GEMV]
    with pytest.raises(ValueError, match="Not a lazily translated matrix entry"):
        warm.ordered([GEMM, "contig_scan_inclusive_sum_int32_int32"])
    with pytest.raises(ValueError, match="unique"):
        warm.ordered([GEMM, GEMM])


class Clock:
    def __init__(self, step):
        self.now, self.step = 0.0, step

    def monotonic(self):
        self.now += self.step
        return self.now


def warm_group(monkeypatch, tmp_path, entries, step, budget):
    built = []

    def build_one(root, packages, target, entry):
        assert Path(packages) == (tmp_path / "packages").resolve()
        built.append((target, entry))
        return {"entry": entry, "target": target}

    monkeypatch.setattr(warm, "ENTRIES", {"blas": entries})
    monkeypatch.setattr(warm, "build_one", build_one)
    monkeypatch.setattr(warm, "ProcessPoolExecutor", ThreadPoolExecutor)
    monkeypatch.setattr(warm, "time", Clock(step))
    records, deferred = warm.build_group(
        tmp_path / "mlx", tmp_path / "packages", "opengl", "blas", 1, budget
    )
    return built, records, deferred


def test_a_group_translates_every_entry_within_its_budget(monkeypatch, tmp_path):
    built, records, deferred = warm_group(
        monkeypatch, tmp_path, (GEMM, GEMV, OFFSETS), 1.0, 100.0
    )
    assert built == [("opengl", GEMM), ("opengl", OFFSETS), ("opengl", GEMV)]
    assert [record["entry"] for record in records] == sorted((GEMM, GEMV, OFFSETS))
    assert deferred == []


def test_entries_past_the_budget_are_left_to_translate_on_demand(monkeypatch, tmp_path):
    built, records, deferred = warm_group(
        monkeypatch, tmp_path, (GEMM, GEMV, OFFSETS), 10.0, 15.0
    )
    assert built == [("opengl", GEMM)]
    assert [record["entry"] for record in records] == [GEMM]
    assert deferred == [OFFSETS, GEMV]


@pytest.mark.parametrize(
    ("group", "jobs", "budget", "message"),
    [
        ("operations", 1, 1.0, "No matrix entries"),
        ("blas", 0, 1.0, "jobs and a positive budget"),
        ("blas", 1, 0.0, "jobs and a positive budget"),
    ],
)
def test_warming_refuses_unknown_groups_and_empty_budgets(
    tmp_path, group, jobs, budget, message
):
    with pytest.raises(ValueError, match=message):
        warm.build_group(tmp_path, tmp_path, "metal", group, jobs, budget)


def test_warming_refuses_entries_its_cache_would_refuse(monkeypatch, tmp_path):
    monkeypatch.setattr(warm, "ENTRIES", {"blas": (GEMM + "_unknown",)})
    with pytest.raises(ValueError):
        warm.build_group(tmp_path, tmp_path / "packages", "metal", "blas", 1, 1.0)


def test_only_the_blas_jobs_warm_their_matrix_kernels_before_the_tests():
    workflow = yaml.safe_load(
        Path(".github/workflows/demo-project-testing.yml").read_text()
    )
    job = workflow["jobs"]["upstream-ops-host"]
    names = [step.get("name") for step in job["steps"]]
    step = job["steps"][names.index("Translate the upstream BLAS matrix kernels")]
    assert step["if"] == "startsWith(matrix.group, 'blas')"
    blas = {group for group in ops.GROUPS if group.startswith("blas")}
    assert set(warm.ENTRIES) == blas == {"blas", "blas-wide", "blas-gather"}
    assert names.index(step["name"]) < names.index(
        "Execute unchanged upstream op tests"
    )
    run = step["run"]
    assert "set -euo pipefail" in run
    assert "portable_host.upstream_matrix_packages" in run
    assert '--group "${{ matrix.group }}"' in run
    tests = job["steps"][names.index("Execute unchanged upstream op tests")]["run"]
    packages = re.search(r"--packages (\S+)", run).group(1)
    assert f"--packages {packages} " in tests
    deadline = int(re.search(r"--timeout-seconds (\d+)", run).group(1))
    budget = float(re.search(r"--budget-seconds (\S+)", run).group(1))
    # The slowest CI translation takes under two minutes, so the entries
    # started within the budget finish inside the step's deadline.
    assert budget + 240 <= deadline

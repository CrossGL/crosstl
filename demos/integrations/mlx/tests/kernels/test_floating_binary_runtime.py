"""Coverage and descriptor selection for batched native binary checks."""

import pytest

from crosstl.project import load_project_config
from demos.integrations.mlx.tests.kernels import (
    test_current_additive,
    test_current_division,
    test_current_extrema,
)
from demos.integrations.mlx.tests.kernels.floating_binary_runtime import (
    MLX_BINARY_SOURCE,
    NATIVE_DTYPE_BATCHES,
    _batch_workloads,
    _binary_load_units,
    _project_config,
)


@pytest.mark.parametrize(
    "fixture,entry_count,pair_count",
    (
        (test_current_extrema, 6, 28032),
        (test_current_additive, 6, 34192),
        (test_current_division, 3, 31232),
    ),
)
def test_batches_preserve_every_case_and_source_profile(
    tmp_path, fixture, entry_count, pair_count
):
    native_name = fixture.__name__.rsplit(".", 1)[1] + "_native_parity"
    native_test = getattr(fixture, native_name)
    (parameters,) = (
        mark.args[1]
        for mark in native_test.pytestmark
        if mark.name == "parametrize" and mark.args[0] == "dtypes"
    )
    assert parameters == NATIVE_DTYPE_BATCHES
    all_cases = list(fixture._cases(("float16", "bfloat16", "float32")))
    batches = [list(fixture._cases(dtypes)) for dtypes in NATIVE_DTYPE_BATCHES]
    assert [case for batch in batches for case in batch] == all_cases
    assert len({case.entry for case in all_cases}) == entry_count
    assert sum(len(case.pairs) for case in all_cases) == pair_count
    for cases in batches:
        workloads = _batch_workloads(cases)
        entries = [case.entry for case in cases]
        path = tmp_path / "crosstl.toml"
        path.write_text(_project_config(workloads[0], entry_points=entries))
        batch = load_project_config(tmp_path, path)
        assert tuple(batch.entry_points[MLX_BINARY_SOURCE]) == tuple(entries)
        assert dict(batch.entry_workgroup_size_rules[MLX_BINARY_SOURCE]) == {
            "*": ("1", "1", "1")
        }
        for workload in workloads:
            path.write_text(_project_config(workload))
            single = load_project_config(tmp_path, path)
            assert single.source_options == batch.source_options
            assert single.index_range_assertions == batch.index_range_assertions


@pytest.mark.parametrize("failure", ("empty", "duplicate", "mixed-profile"))
def test_batch_rejects_incompatible_cases(failure):
    cases = list(test_current_extrema._cases(("float16",)))
    if failure == "empty":
        cases = []
    elif failure == "duplicate":
        cases.append(cases[0])
    else:
        cases.extend(test_current_extrema._cases(("float32",)))
    with pytest.raises(AssertionError):
        _batch_workloads(cases)


@pytest.mark.parametrize(
    "failure",
    (
        None,
        "missing",
        "duplicate",
        "duplicate-id",
        "unexpected",
        "wrong-source",
        "wrong-target",
        "blocked",
        "failed",
    ),
)
def test_batch_selects_exact_ready_load_units(failure):
    entries = ["vv_Minimumfloat16", "vv_Maximumfloat16"]
    units = [
        {
            "id": str(index),
            "entryPoint": {"source": entry},
            "target": "opengl",
            "source": MLX_BINARY_SOURCE,
            "validation": {"loadReady": True},
        }
        for index, entry in enumerate(reversed(entries))
    ]
    if failure == "missing":
        units.pop()
    elif failure == "duplicate":
        units[0] = units[1]
    elif failure == "duplicate-id":
        units[0]["id"] = units[1]["id"]
    elif failure == "unexpected":
        units[0]["entryPoint"]["source"] = "vv_Addfloat16"
    elif failure == "wrong-source":
        units[0]["source"] = "other.metal"
    elif failure == "wrong-target":
        units[0]["target"] = "directx"
    elif failure == "blocked":
        units[0]["validation"]["loadReady"] = False
    loader = {"success": failure != "failed", "loadUnits": units}
    if failure:
        with pytest.raises(AssertionError):
            _binary_load_units(loader, entries, "opengl")
    else:
        selected = _binary_load_units(loader, entries, "opengl")
        assert [selected[entry]["id"] for entry in entries] == ["1", "0"]

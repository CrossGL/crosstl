"""GitHub matrix expansion, and the MLX workflow's matrices under it."""

from pathlib import Path

import pytest
import yaml

from demos.integrations.mlx.tests.workflow_matrix import (
    expand_matrix,
    unrealised_includes,
)

WORKFLOW = (
    Path(__file__).resolve().parents[4] / ".github/workflows/demo-project-testing.yml"
)


def test_expansion_follows_the_documented_include_example():
    # The example in GitHub's "Expanding or adding matrix configurations".
    matrix = {
        "fruit": ["apple", "pear"],
        "animal": ["cat", "dog"],
        "include": [
            {"color": "green"},
            {"color": "pink", "animal": "cat"},
            {"fruit": "apple", "shape": "circle"},
            {"fruit": "banana"},
            {"fruit": "banana", "animal": "cat"},
        ],
    }
    assert expand_matrix(matrix) == [
        {"fruit": "apple", "animal": "cat", "color": "pink", "shape": "circle"},
        {"fruit": "apple", "animal": "dog", "color": "green", "shape": "circle"},
        {"fruit": "pear", "animal": "cat", "color": "pink"},
        {"fruit": "pear", "animal": "dog", "color": "green"},
        {"fruit": "banana"},
        {"fruit": "banana", "animal": "cat"},
    ]
    assert unrealised_includes(matrix) == []


def test_exclusions_apply_before_includes():
    matrix = {
        "os": ["linux", "windows"],
        "version": [1, 2],
        "exclude": [{"os": "windows", "version": 1}],
        "include": [{"os": "windows", "version": 1, "extra": True}],
    }
    assert expand_matrix(matrix) == [
        {"os": "linux", "version": 1},
        {"os": "linux", "version": 2},
        {"os": "windows", "version": 2},
        {"os": "windows", "version": 1, "extra": True},
    ]


def test_platform_entries_without_their_axis_collapse_to_the_last():
    matrix = {
        "test": ["a", "b"],
        "include": [
            {"os": "ubuntu", "target": "opengl"},
            {"os": "windows", "target": "directx"},
        ],
    }
    assert expand_matrix(matrix) == [
        {"test": "a", "os": "windows", "target": "directx"},
        {"test": "b", "os": "windows", "target": "directx"},
    ]
    assert unrealised_includes(matrix) == [{"os": "ubuntu", "target": "opengl"}]
    # With the target as an axis each entry only adds its runner.
    matrix = {
        "test": ["a", "b"],
        "target": ["opengl", "directx"],
        "include": [
            {"target": "opengl", "os": "ubuntu"},
            {"target": "directx", "os": "windows"},
        ],
    }
    assert len(expand_matrix(matrix)) == 4
    assert unrealised_includes(matrix) == []


def test_include_entries_alone_each_run_once():
    matrix = {"include": [{"os": "linux"}, {"os": "windows"}]}
    assert expand_matrix(matrix) == [{"os": "linux"}, {"os": "windows"}]
    assert unrealised_includes(matrix) == []


@pytest.mark.parametrize(
    "matrix", [{"test": "${{ fromJSON(inputs.tests) }}"}, {"include": ["a"]}]
)
def test_expansion_refuses_matrices_it_cannot_read(matrix):
    with pytest.raises(ValueError):
        expand_matrix(matrix)


def literal_matrices():
    jobs = yaml.safe_load(WORKFLOW.read_text())["jobs"]
    for name, job in sorted(jobs.items()):
        matrix = (job.get("strategy") or {}).get("matrix")
        if not isinstance(matrix, dict):
            continue
        axes = [
            value for key, value in matrix.items() if key not in ("include", "exclude")
        ]
        entries = [*(matrix.get("include") or ()), *(matrix.get("exclude") or ())]
        if all(isinstance(value, list) for value in axes) and all(
            isinstance(entry, dict) for entry in entries
        ):
            yield name, matrix


def test_every_include_entry_of_the_mlx_workflow_runs():
    matrices = dict(literal_matrices())
    assert "quantized-upstream-host" in matrices
    lost = {
        name: unrealised_includes(matrix)
        for name, matrix in matrices.items()
        if unrealised_includes(matrix)
    }
    assert lost == {}

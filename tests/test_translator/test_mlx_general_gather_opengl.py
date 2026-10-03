"""Execute pinned indexed gather with explicit bounded OpenGL workloads."""

import json
import os
from pathlib import Path

import pytest

from crosstl.project import build_native_loader_dispatch_request
from demos.integrations.mlx.portable_host.prepare import COMMIT
from tests.test_translator.test_boolean_buffer_runtime import _bound_values
from tests.test_translator.test_loop_updates import _execute
from tests.test_translator.test_mlx_current_gather import _validate
from tests.test_translator.test_mlx_general_gather import (
    LAYOUTS,
    _package,
    _verify_source,
    _workload,
)
from tests.test_translator.test_mlx_general_gather_directx import (
    INDEXED_SPECIALIZATIONS,
    _bound_inputs,
)

REQUIRE_ENV = "CROSTL_REQUIRE_MLX_OPENGL_GENERAL_GATHER"


def _assertions(count, ndim):
    maximum = max(
        len(value["values"]) - 1
        for layout in LAYOUTS
        for value in _workload(count, ndim, layout)["inputs"].values()
    )
    assert 0 <= maximum < 2**31
    return (
        {
            "source": "*",
            "expression": "reference.offset + index",
            "minimum": 0,
            "maximum": maximum,
        },
    )


@pytest.fixture(scope="module")
def upstream():
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 for required indexed OpenGL gather")
    root = Path(os.environ["CROSTL_MLX_CURRENT_ROOT"]).resolve()
    hashes = _verify_source(root)
    yield root, hashes
    assert _verify_source(root) == hashes


@pytest.fixture(
    scope="module",
    params=[
        pytest.param(
            spec,
            id=f"indices{spec[0]}-ndim{spec[1]}",
            marks=pytest.mark.xdist_group(f"gather-opengl-{spec[0]}-{spec[1]}"),
        )
        for spec in INDEXED_SPECIALIZATIONS
    ],
)
def package(request, upstream, tmp_path_factory):
    root, hashes = upstream
    count, ndim = request.param
    retained = tmp_path_factory.mktemp(f"gather-opengl-{count}-{ndim}")
    entry, source, descriptor, directory = _package(
        root,
        count,
        ndim,
        retained,
        "opengl",
        index_range_assertions=_assertions(count, ndim),
    )
    return hashes, count, ndim, entry, source, descriptor, directory


def _request(descriptor, directory, entry, workload):
    expected = _bound_values(descriptor, workload["outputs"])
    request = build_native_loader_dispatch_request(
        descriptor,
        directory,
        _bound_inputs(descriptor, entry, workload["inputs"]),
        expected,
        workload["grid"],
        expected_target="opengl",
    )
    assert not request.execution_plan.diagnostics
    return request, expected


def test_opengl_indexed_general_gather_compiles(package, tmp_path):
    _hashes, count, ndim, entry, _source, descriptor, directory = package
    for layout in LAYOUTS:
        _request(descriptor, directory, entry, _workload(count, ndim, layout))
    module = _validate(
        directory / descriptor["artifact"]["packagePath"], tmp_path, "opengl"
    )
    assert module.is_file() and module.stat().st_size


@pytest.mark.parametrize("layout", LAYOUTS)
def test_opengl_indexed_general_gather_executes_natively(package, tmp_path, layout):
    hashes, count, ndim, entry, source, descriptor, directory = package
    workload = _workload(count, ndim, layout)
    (tmp_path / "workload.json").write_text(
        json.dumps(
            {
                "commit": COMMIT,
                "headers": hashes,
                "entry": entry,
                "indexRangeAssertions": _assertions(count, ndim),
                **workload,
            },
            indent=2,
        )
    )
    (tmp_path / "source.metal").write_text(source)
    request, expected = _request(descriptor, directory, entry, workload)
    _execute(request, expected, tmp_path, validate=_validate)


def test_opengl_gather_gate_requires_native_execution():
    from tools import ci_coverage

    workflow = (
        Path(__file__).resolve().parents[2]
        / ".github/workflows/mlx-gather-roundtrip.yml"
    ).read_text()
    step = ci_coverage.workflow_step_section(
        workflow, "Validate indexed OpenGL gather and resource aggregates"
    )
    assert f'{REQUIRE_ENV}: "1"' in step
    assert 'CROSTL_REQUIRE_OPENGL_RESOURCE_AGGREGATES: "1"' in step
    assert "test_mlx_general_gather_opengl.py" in step
    assert "test_resource_aggregates_opengl.py" in step
    assert "--timeout-seconds 1200" in step and "--junitxml=" in step
    assert "--basetemp=" in step and "-n auto --dist loadgroup" in step
    assert "continue-on-error" not in workflow and "if:" not in step
    assert "runs-on: ubuntu-24.04" in workflow
    assert "LIBGL_ALWAYS_SOFTWARE" in step and "PYOPENGL_PLATFORM: egl" in step
    assert "include-hidden-files: true" in workflow and "if: always()" in workflow

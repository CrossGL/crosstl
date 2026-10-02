"""Execute unchanged indexed general-gather specializations on DirectX."""

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
    SPECIALIZATIONS,
    _package,
    _verify_source,
    _workload,
)

REQUIRE_ENV = "CROSTL_REQUIRE_MLX_DIRECTX_GENERAL_GATHER"
INDEXED_SPECIALIZATIONS = tuple(spec for spec in SPECIALIZATIONS if spec[0])


@pytest.fixture(scope="module")
def upstream():
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 for required indexed DirectX gather")
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
            marks=pytest.mark.xdist_group(f"gather-directx-{spec[0]}-{spec[1]}"),
        )
        for spec in INDEXED_SPECIALIZATIONS
    ],
)
def package(request, upstream, tmp_path_factory):
    root, hashes = upstream
    count, ndim = request.param
    retained = tmp_path_factory.mktemp(f"gather-directx-{count}-{ndim}")
    entry, source, descriptor, directory = _package(
        root, count, ndim, retained, "directx"
    )
    return root, hashes, count, ndim, entry, source, descriptor, directory


def test_indexed_general_gather_compiles(package, tmp_path):
    _root, _hashes, _count, _ndim, _entry, _source, descriptor, directory = package
    module = _validate(
        directory / descriptor["artifact"]["packagePath"], tmp_path, "directx"
    )
    assert module.is_file() and module.stat().st_size


@pytest.mark.parametrize("layout", LAYOUTS)
def test_indexed_general_gather_executes_natively(package, tmp_path, layout):
    root, hashes, count, ndim, entry, source, descriptor, directory = package
    workload = _workload(count, ndim, layout)
    (tmp_path / "workload.json").write_text(
        json.dumps(
            {
                "commit": COMMIT,
                "headers": hashes,
                "entry": entry,
                **workload,
            },
            indent=2,
        )
    )
    (tmp_path / "source.metal").write_text(source)
    expected = _bound_values(descriptor, workload["outputs"])
    request = build_native_loader_dispatch_request(
        descriptor,
        directory,
        _bound_values(descriptor, workload["inputs"]),
        expected,
        workload["grid"],
        expected_target="directx",
    )
    assert not request.execution_plan.diagnostics
    _execute(request, expected, tmp_path, validate=_validate)


def test_indexed_general_gather_retains_all_nonempty_specializations():
    assert INDEXED_SPECIALIZATIONS == ((1, 0), (1, 1), (1, 2), (2, 2), (2, 3))
    assert (0, 0) in SPECIALIZATIONS  # Zero-extent arrays retain their Metal gate.

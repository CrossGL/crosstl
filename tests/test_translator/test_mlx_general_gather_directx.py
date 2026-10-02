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
from tests.test_translator.test_software_subgroup_product import (
    _package as _control_package,
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
    _root, _hashes, count, ndim, entry, _source, descriptor, directory = package
    for layout in LAYOUTS:
        _request(descriptor, directory, entry, _workload(count, ndim, layout))
    module = _validate(
        directory / descriptor["artifact"]["packagePath"], tmp_path, "directx"
    )
    assert module.is_file() and module.stat().st_size


def _bound_inputs(descriptor, entry, inputs):
    bound, matched = {}, set()
    for binding in descriptor["bindings"]:
        if "executionInput" in binding.get("provenance", {}):
            continue
        layout = binding["scalarLayout"]
        member = layout.get("memberName", binding["name"])
        name = member.removeprefix(entry.rstrip("_") + "_")
        assert name in inputs and name not in matched, (name, binding)
        assert binding["name"] not in bound
        matched.add(name)
        value = inputs[name]
        if value["dtype"] == "bool":
            assert all(type(item) is bool for item in value["values"])
            value = {
                **value,
                "dtype": "uint32",
                "values": [int(item) for item in value["values"]],
            }
        assert value["dtype"] == layout["elementType"], (name, layout)
        bound[binding["name"]] = value
    assert matched == set(inputs)
    return bound


def _request(descriptor, directory, entry, workload):
    expected = _bound_values(descriptor, workload["outputs"])
    request = build_native_loader_dispatch_request(
        descriptor,
        directory,
        _bound_inputs(descriptor, entry, workload["inputs"]),
        expected,
        workload["grid"],
        expected_target="directx",
    )
    assert not request.execution_plan.diagnostics
    return request, expected


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
    request, expected = _request(descriptor, directory, entry, workload)
    _execute(request, expected, tmp_path, validate=_validate)


def test_directx_gather_request_preserves_constant_and_boolean_inputs(tmp_path):
    entry = "gather_bindings"
    source = """#include <metal_stdlib>
using namespace metal;
kernel void gather_bindings(const constant size_t& src_ndim [[buffer(4)]],
                            const device bool* idx_contigs [[buffer(9)]],
                            const constant int& idx_ndim [[buffer(10)]],
                            device uint* results [[buffer(11)]],
                            uint i [[thread_position_in_grid]]) {
    results[i] = uint(src_ndim) + uint(idx_ndim) + uint(idx_contigs[i]);
}
"""
    _, descriptor, directory = _control_package(
        tmp_path,
        "directx",
        "uint",
        (1, 1, 1),
        source=source,
        software_subgroups=False,
    )
    inputs = {
        "src_ndim": {"dtype": "uint64", "shape": [1], "values": [3]},
        "idx_ndim": {"dtype": "int32", "shape": [1], "values": [2]},
        "idx_contigs": {"dtype": "bool", "shape": [2], "values": [True, False]},
        "results": {"dtype": "uint32", "shape": [2], "values": [0, 0]},
    }
    bound = _bound_inputs(descriptor, entry, inputs)
    assert bound["idx_contigs"] == {
        "dtype": "uint32",
        "shape": [2],
        "values": [1, 0],
    }
    assert inputs["idx_contigs"]["dtype"] == "bool"
    assert all(type(item) is bool for item in inputs["idx_contigs"]["values"])
    constants = {
        binding["scalarLayout"]["memberName"].removeprefix(entry + "_"): bound[
            binding["name"]
        ]
        for binding in descriptor["bindings"]
        if binding["kind"] == "constant-buffer"
    }
    assert constants == {name: inputs[name] for name in ("src_ndim", "idx_ndim")}
    _request(
        descriptor,
        directory,
        entry,
        {
            "inputs": inputs,
            "outputs": {"results": {"dtype": "uint32", "shape": [2], "values": [6, 5]}},
            "grid": {"workgroupCount": [2, 1, 1], "workgroupSize": [1, 1, 1]},
        },
    )
    with pytest.raises(AssertionError):
        _bound_inputs(descriptor, entry, {**inputs, "unused": inputs["src_ndim"]})
    with pytest.raises(AssertionError):
        _bound_inputs(
            descriptor,
            entry,
            {name: value for name, value in inputs.items() if name != "idx_ndim"},
        )


def test_indexed_general_gather_retains_all_nonempty_specializations():
    assert INDEXED_SPECIALIZATIONS == ((1, 0), (1, 1), (1, 2), (2, 2), (2, 3))
    assert (0, 0) in SPECIALIZATIONS  # Zero-extent arrays retain their Metal gate.

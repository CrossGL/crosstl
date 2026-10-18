"""Checked two-pass column kernels preserve first-pass partial sums."""

import json
import os
import struct

import pytest

from crosstl.project import (
    ProjectConfig,
    build_native_loader_dispatch_request,
    translate_project,
)
from crosstl.translator.dispatch_regions import plan_dispatch_regions
from demos.integrations.mlx.tests.corpus_evidence import corpus_workspace
from demos.integrations.mlx.tests.kernels.test_current_arg_reduce import (
    _package_descriptor,
    _source_bindings,
)
from demos.integrations.mlx.tests.kernels.test_current_arg_reduce import (
    current_mlx as current_mlx,
)
from tests.runtime_helpers import _validate
from tests.test_translator.test_loop_updates import _execute

SOURCE = "mlx/backend/metal/kernels/reduce.metal"
REQUIRE_ENV = "CROSTL_REQUIRE_MLX_CHECKED_COLUMNS"
ENTRIES = tuple(
    f"col_reduce_2pass_large_1_32_32_reduce_sum{dtype}"
    for dtype in ("float16", "float32")
)
ROWS = (512, 1025)
COLUMNS = 33
METADATA = {
    "reduction_size": ("uint64", 512, 1025, False),
    "reduction_stride": ("int64", COLUMNS, COLUMNS, False),
    "shape": ("int32", 0, 0, True),
    "strides": ("int64", 0, 0, True),
    "ndim": ("int32", 0, 0, False),
    "reduce_shape": ("int32", 512, 1025, True),
    "reduce_strides": ("int64", COLUMNS, COLUMNS, True),
    "reduce_ndim": ("int32", 1, 1, False),
    "non_col_reductions": ("uint64", 1, 1, False),
    "out_size": ("uint64", 1, 1, False),
}


def _config(root, work, target, entry):
    options = {}
    if target != "metal":
        options["software_subgroup_width"] = 32
    if target == "directx":
        options["relative_wave_shuffle_out_of_range"] = "self"
    if target == "opengl":
        (region,) = plan_dispatch_regions((512, 32, 1), (256, 1, 1))
        options["dispatch_region"] = region.to_json()
    return ProjectConfig(
        root=root,
        source_roots=("mlx/backend/metal/kernels",),
        include_patterns=(SOURCE,),
        include_dirs=(".",),
        targets=(target,),
        output_dir=f"{work.relative_to(root).as_posix()}/out",
        entry_points={SOURCE: (entry,)},
        entry_workgroup_size_rules={SOURCE: {entry: (256, 1, 1)}},
        metadata_preconditions=[
            dict(
                source=SOURCE,
                entryPoint=entry,
                parameter=name,
                minimum=low,
                maximum=high,
                **({"element": 0} if array else {}),
            )
            for name, (_, low, high, array) in METADATA.items()
        ],
        source_options={
            "metal": dict(
                max_template_specializations=1024,
                max_template_materialization_work=1048576,
                preserve_resource_origins=True,
                target_options={target: options},
            )
        },
    )


@pytest.mark.parametrize("target", ["metal", "directx", "opengl"])
@pytest.mark.parametrize("entry", ENTRIES)
def test_column_ranges_are_checked_metadata_not_index_assertions(
    tmp_path, target, entry
):
    config = _config(tmp_path, tmp_path / "case", target, entry)
    assert not config.index_range_assertions
    assert len(config.metadata_preconditions) == 10
    assert config.entry_workgroup_size_rules[SOURCE][entry] == ("256", "1", "1")


def _reference_source(entry):
    scalar = "float16_t" if entry.endswith("float16") else "float"
    return f"""#include <metal_atomic>
#include <metal_simdgroup>
#include "mlx/backend/metal/kernels/defines.h"
#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/atomic.h"
#include "mlx/backend/metal/kernels/reduction/ops.h"
#include "mlx/backend/metal/kernels/reduce.h"
instantiate_kernel("{entry}", col_reduce_2pass,
                   {scalar}, {scalar}, Sum<{scalar}>, int64_t, 1, 32, 32)
"""


def _storage_payload(layout, values):
    result = dict(dtype=layout["elementType"], shape=[len(values)], values=values)
    encoding = layout.get("storageEncoding")
    if encoding is None:
        return result
    assert layout["elementType"] == "uint16"
    assert encoding == {"encoding": "ieee754-binary16", "logicalElementType": "float16"}
    result.update(
        dtype="float16",
        encoding="ieee754-binary16",
        values=[struct.unpack("<H", struct.pack("<e", value))[0] for value in values],
    )
    return result


def test_column_upload_uses_reflected_half_storage_encoding():
    values = [1.0, -1.0, 0.0625, 16384.0]
    for dtype in ("float16", "float32"):
        assert _storage_payload({"elementType": dtype}, values) == dict(
            dtype=dtype, shape=[4], values=values
        )
    assert _storage_payload(
        {
            "elementType": "uint16",
            "storageEncoding": {
                "encoding": "ieee754-binary16",
                "logicalElementType": "float16",
            },
        },
        values,
    ) == dict(
        dtype="float16",
        encoding="ieee754-binary16",
        shape=[4],
        values=[0x3C00, 0xBC00, 0x2C00, 0x7400],
    )


def _request(descriptor, package, rows, profile):
    values = [
        (
            1.0
            if profile == "ones"
            else float((row * COLUMNS + column) % 23 - 11)
            / (16 if profile == "fractions" else 1)
        )
        for row in range(rows)
        for column in range(COLUMNS)
    ]
    partials = [
        sum(
            values[row * COLUMNS + column]
            for row in range(rows)
            if (row // 32) % 32 == block
        )
        for block in range(32)
        for column in range(COLUMNS)
    ]
    guard = 16384.0
    expected = partials + [guard] * 32
    # Include inactive-lane and terminal pointer formation, not only loaded rows.
    allocation = COLUMNS * (max(ROWS) + 2 * 1024) + 64
    values += [1033.0] * (allocation - len(values))
    inputs, outputs, observations = {}, {}, []
    for binding in _source_bindings(descriptor):
        name = binding["name"]
        parameter = binding["provenance"]["sourceResource"]["parameter"]
        dtype = binding["scalarLayout"]["elementType"]
        if parameter == "in":
            data = values
        elif parameter == "out":
            data = [guard] * len(expected)
            outputs[name] = _storage_payload(binding["scalarLayout"], expected)
        else:
            source_dtype, minimum, _, _ = METADATA[parameter]
            assert dtype == source_dtype
            data = [
                rows if parameter in {"reduction_size", "reduce_shape"} else minimum
            ]
        inputs[name] = _storage_payload(binding["scalarLayout"], data)
        if parameter != "out":
            observations.append(name)
    assert len(inputs) == 12 and len(observations) == 11
    request = build_native_loader_dispatch_request(
        descriptor,
        package,
        inputs,
        outputs,
        dict(workgroupCount=[2, 32, 1], workgroupSize=[256, 1, 1]),
        observe_inputs=observations,
    )
    assert not request.execution_plan.diagnostics
    return request, outputs


@pytest.fixture(scope="module")
def checked_corpus(request):
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 for native column validation")
    assert os.environ.get("CROSTL_MLX_CURRENT_ROOT")
    assert os.environ.get("CROSTL_MLX_CURRENT_TARGET")
    return request.getfixturevalue("current_mlx")


@pytest.mark.parametrize("entry", ENTRIES)
def test_checked_column_partial_sums_execute(checked_corpus, tmp_path, entry):
    root, target = checked_corpus
    with corpus_workspace(
        root, family="checked-column", target=target, entry_point=entry
    ) as work:
        config = _config(root, work, target, entry)
        report = translate_project(config, format_output=False)
        payload = report.to_json()
        report.write_json(tmp_path / "portability-report.json")
        assert payload["summary"]["translatedCount"] == 1, payload["diagnostics"]
        assert not payload["diagnostics"]
        descriptor, package = _package_descriptor(report, tmp_path)
        (tmp_path / "descriptor.json").write_text(
            json.dumps(descriptor, indent=2) + "\n", encoding="utf-8"
        )
        for rows in ROWS:
            for profile in ("ones", "signed-columns", "fractions"):
                native = tmp_path / f"{rows}-{profile}"
                native.mkdir()
                request, expected = _request(descriptor, package, rows, profile)
                _execute(
                    request,
                    expected,
                    native,
                    validate=_validate,
                    original_source=(
                        _reference_source(entry) if target == "metal" else None
                    ),
                    original_entry=entry,
                    metal_compile_flags=("-I", str(root), "-Wno-c++17-extensions"),
                    metal_language_version="3.1",
                )
                evidence = json.loads((native / "evidence.json").read_text())
                observations = evidence["records"]["generated"]["details"][
                    "inputObservations"
                ]
                assert len(observations) == 11
                assert all(item["unchanged"] for item in observations.values())

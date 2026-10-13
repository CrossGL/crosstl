"""Checked row reductions retain nested cursor state and partial row reads."""

import json
import os

import pytest

from crosstl.project import (
    ProjectConfig,
    build_native_loader_dispatch_request,
    translate_project,
)
from crosstl.translator.dispatch_regions import plan_dispatch_regions
from demos.integrations.mlx.portable_host.reduction_cache import ReductionPackageCache
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
ENTRY = "row_reduce_looped_large_5_reduce_sumfloat32"
SIMPLE_ENTRY = "row_reduce_simple_sumfloat32"
REQUIRE_ENV = "CROSTL_REQUIRE_MLX_CHECKED_ROWS"
SIMPLE_METADATA = {
    "reduction_size": ("uint64", 65, False),
    "out_size": ("int64", 33, False),
}
METADATA = {
    "row_size": ("int64", 132, False),
    "non_row_reductions": ("int64", 3, False),
    "shape": ("int32", 2, True),
    "strides": ("int64", 480, True),
    "ndim": ("int32", 1, False),
    "reduce_shape": ("int32", 3, True),
    "reduce_strides": ("int64", 160, True),
    "reduce_ndim": ("int32", 1, False),
}


def _config(root, work, target, *, simple=False):
    entry = SIMPLE_ENTRY if simple else ENTRY
    metadata = SIMPLE_METADATA if simple else METADATA
    options = {}
    if target != "metal":
        options["software_subgroup_width"] = 32
    if target == "directx":
        options["relative_wave_shuffle_out_of_range"] = "self"
    if target == "opengl":
        (region,) = plan_dispatch_regions((32, 9 if simple else 2, 1), (32, 1, 1))
        options["dispatch_region"] = region.to_json()
    return ProjectConfig(
        root=root,
        source_roots=("mlx/backend/metal/kernels",),
        include_patterns=(SOURCE,),
        include_dirs=(".",),
        targets=(target,),
        output_dir=f"{work.relative_to(root).as_posix()}/out",
        entry_points={SOURCE: (entry,)},
        entry_workgroup_size_rules={SOURCE: {entry: (32, 1, 1)}},
        metadata_preconditions=[
            dict(
                source=SOURCE,
                entryPoint=entry,
                parameter=name,
                minimum=value,
                maximum=value,
                **({"element": 0} if array else {}),
            )
            for name, (_, value, array) in metadata.items()
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
@pytest.mark.parametrize("simple", [False, True])
def test_row_offsets_use_checked_uploads_and_launch_geometry(tmp_path, target, simple):
    config = _config(tmp_path, tmp_path / "case", target, simple=simple)
    entry = SIMPLE_ENTRY if simple else ENTRY
    assert not config.index_range_assertions
    assert len(config.metadata_preconditions) == (2 if simple else 8)
    assert config.entry_workgroup_size_rules[SOURCE][entry] == ("32", "1", "1")


def _reference_source(*, simple=False):
    entry = SIMPLE_ENTRY if simple else ENTRY
    function = "row_reduce_simple" if simple else "row_reduce_looped"
    arguments = "float, float, Sum<float>" + ("" if simple else ", int64_t, 5")
    return f"""#include <metal_atomic>
#include <metal_simdgroup>
#include "mlx/backend/metal/kernels/defines.h"
#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/atomic.h"
#include "mlx/backend/metal/kernels/reduction/ops.h"
#include "mlx/backend/metal/kernels/reduce.h"
instantiate_kernel("{entry}", {function}, {arguments})
"""


def _values(profile, *, simple=False):
    if simple:
        values, totals = [], []
        for row in range(33):
            words = [
                (
                    1.0
                    if profile == "ones"
                    else float((row * 13 + column) % 23 - 11)
                    / (16 if profile == "fractions" else 1)
                )
                for column in range(65)
            ]
            values.extend(words)
            totals.append(sum(words))
        return values, totals
    # Include padded rows, inactive lanes and terminal pointer formation.
    values = [1033.0] * 1120
    totals = []
    for batch in range(2):
        total = 0.0
        for row in range(3):
            for column in range(132):
                value = (
                    1.0
                    if profile == "ones"
                    else float((batch * 13 + row * 7 + column) % 23 - 11)
                    / (16 if profile == "fractions" else 1)
                )
                values[batch * 480 + row * 160 + column] = value
                total += value
        totals.append(total)
    return values, totals


@pytest.mark.parametrize(
    "profile,expected",
    [
        ("ones", [396.0, 396.0]),
        ("signed", [-3.0, -7.0]),
        ("fractions", [-0.1875, -0.4375]),
    ],
)
def test_row_profiles_exclude_padding_from_the_reference(profile, expected):
    values, totals = _values(profile)
    assert totals == expected
    assert len(values) == 1120
    for base in (0, 160, 320, 480, 640, 800):
        assert values[base + 132 : base + 160] == [1033.0] * 28


def _request(descriptor, package, profile, *, simple=False):
    values, totals = _values(profile, simple=simple)
    metadata = SIMPLE_METADATA if simple else METADATA
    output_count = len(totals) + 32
    guard = 16384.0
    inputs, outputs, observations = {}, {}, []
    for binding in _source_bindings(descriptor):
        name = binding["name"]
        parameter = binding["provenance"]["sourceResource"]["parameter"]
        dtype = binding["scalarLayout"]["elementType"]
        if parameter == "in":
            data = values
            assert dtype == "float32"
        elif parameter == "out":
            data = [guard] * output_count
            assert dtype == "float32"
            outputs[name] = dict(
                dtype=dtype, shape=[output_count], values=totals + [guard] * 32
            )
        else:
            source_dtype, value, _ = metadata[parameter]
            assert dtype == source_dtype
            data = [value]
        inputs[name] = dict(dtype=dtype, shape=[len(data)], values=data)
        if parameter != "out":
            observations.append(name)
    assert len(inputs) == len(metadata) + 2
    assert len(observations) == len(metadata) + 1
    request = build_native_loader_dispatch_request(
        descriptor,
        package,
        inputs,
        outputs,
        dict(workgroupCount=[1, 9 if simple else 2, 1], workgroupSize=[32, 1, 1]),
        observe_inputs=observations,
    )
    assert not request.execution_plan.diagnostics
    return request, outputs


@pytest.fixture(scope="module")
def checked_corpus(request):
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 for native row validation")
    assert os.environ.get("CROSTL_MLX_CURRENT_ROOT")
    assert os.environ.get("CROSTL_MLX_CURRENT_TARGET")
    return request.getfixturevalue("current_mlx")


@pytest.mark.parametrize("simple", [False, True], ids=["looped", "simple"])
def test_checked_row_reduction_executes(checked_corpus, tmp_path, simple):
    root, target = checked_corpus
    entry = SIMPLE_ENTRY if simple else ENTRY
    with corpus_workspace(
        root, family="checked-row", target=target, entry_point=entry
    ) as work:
        if simple:
            layout = dict(
                metadata={
                    name: [value] for name, (_, value, _) in SIMPLE_METADATA.items()
                },
                inputCount=33 * 65,
                outputCount=33,
            )
            execution = dict(workgroupCount=[1, 9, 1], workgroupSize=[32, 1, 1])
            cache = ReductionPackageCache(root, tmp_path / "cache", target)
            descriptor, package = cache.get(entry, 33 * 65, execution, layout=layout)
            assert cache.load(entry, 33 * 65, execution, layout=layout) == (
                descriptor,
                package,
            )
            identity = json.loads((package.parent / "index.json").read_text())[
                "identity"
            ]
            assert identity["contract"]["metadata"] == layout["metadata"]
        else:
            report = translate_project(_config(root, work, target), format_output=False)
            report.write_json(tmp_path / "portability-report.json")
            payload = report.to_json()
            assert payload["summary"]["translatedCount"] == 1, payload["diagnostics"]
            assert not payload["diagnostics"]
            descriptor, package = _package_descriptor(report, tmp_path)
        (tmp_path / "descriptor.json").write_text(
            json.dumps(descriptor, indent=2) + "\n", encoding="utf-8"
        )
        for profile in ("ones", "signed", "fractions"):
            native = tmp_path / profile
            native.mkdir()
            request, expected = _request(descriptor, package, profile, simple=simple)
            _execute(
                request,
                expected,
                native,
                validate=_validate,
                original_source=(
                    _reference_source(simple=simple) if target == "metal" else None
                ),
                original_entry=entry,
                metal_compile_flags=("-I", str(root), "-Wno-c++17-extensions"),
                metal_language_version="3.1" if target == "metal" else None,
            )
            evidence = json.loads(
                (native / "evidence.json").read_text(encoding="utf-8")
            )
            for record in evidence["records"].values():
                observations = record["details"]["inputObservations"]
                assert len(observations) == (3 if simple else 9)
                assert all(item["unchanged"] for item in observations.values())

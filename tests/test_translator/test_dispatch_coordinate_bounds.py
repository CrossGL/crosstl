"""Coordinate specialization retains checked bounds without nested swizzles."""

from dataclasses import replace

import pytest

from crosstl.project import build_native_loader_dispatch_request, translate_project
from crosstl.translator.dispatch_regions import plan_dispatch_regions
from tests.test_translator.test_metadata_index_bounds import GUARD, _index_config


def _coordinate_source(kind, grid, mutation=""):
    width = int(kind[-1]) if kind[-1].isdigit() else 1
    components = ["index" if width == 1 else "index.x"]
    if width > 1:
        components.append(f"{grid[0]}ul * index.y")
    if width > 2:
        components.append(f"{grid[0] * grid[1]}ul * index.z")
    linear = " + ".join(f"ulong({component})" for component in components)
    return f"""#include <metal_stdlib>
using namespace metal;
kernel void read_value(device uint* output [[buffer(0)]],
                       constant ulong& stride [[buffer(1)]],
                       {kind} index [[thread_position_in_grid]]) {{
    {mutation}
    ulong position = {linear};
    output[4ul + position * stride] = uint(position + 7ul);
}}
"""


def _coordinate_config(root, target, kind, region, *, mutation="", bounded=True):
    source = _coordinate_source(kind, region.thread_grid_size, mutation)
    config = _index_config(
        root,
        target=target,
        source=source,
        conditions=(
            [dict(entryPoint="read_value", parameter="stride", minimum=16, maximum=16)]
            if bounded
            else []
        ),
    )
    options = dict(config.source_options)
    if target != "metal":
        options["metal"] = dict(
            options["metal"],
            target_options={target: {"dispatch_region": region.to_json()}},
        )
    return replace(config, source_options=options, workgroup_size=region.workgroup_size)


def _coordinate_request(descriptor, package, region):
    grid = region.thread_grid_size
    count = grid[0] * grid[1] * grid[2]
    initial = [GUARD] * (count * 16 + 8)
    expected = list(initial)
    extent = tuple(n * w for n, w in zip(region.workgroup_count, region.workgroup_size))
    for z in range(region.thread_offset[2], region.thread_offset[2] + extent[2]):
        for y in range(region.thread_offset[1], region.thread_offset[1] + extent[1]):
            for x in range(
                region.thread_offset[0], region.thread_offset[0] + extent[0]
            ):
                position = x + grid[0] * y + grid[0] * grid[1] * z
                expected[4 + position * 16] = position + 7
    inputs, outputs, observed = {}, {}, []
    for binding in descriptor["bindings"]:
        origin = binding.get("provenance", {}).get("sourceResource")
        if origin is None:
            continue
        name = binding["name"]
        values = initial if origin["parameter"] == "output" else [16]
        inputs[name] = dict(
            dtype=binding["scalarLayout"]["elementType"],
            shape=[len(values)],
            values=values,
        )
        if origin["parameter"] == "output":
            outputs[name] = dict(dtype="uint32", shape=[len(expected)], values=expected)
        else:
            observed.append(name)
    return (
        build_native_loader_dispatch_request(
            descriptor,
            package,
            inputs,
            outputs,
            dict(
                workgroupCount=list(region.workgroup_count),
                workgroupSize=list(region.workgroup_size),
            ),
            observe_inputs=observed,
        ),
        outputs,
    )


@pytest.mark.parametrize("kind", ["uint", "uint2", "uint3"])
@pytest.mark.parametrize("target", ["opengl", "directx"])
def test_checked_coordinate_widths_retain_bounds_in_every_region(
    tmp_path, kind, target
):
    width = int(kind[-1]) if kind[-1].isdigit() else 1
    grid = (5, 3 if width > 1 else 1, 3 if width > 2 else 1)
    size = (4, 2 if width > 1 else 1, 2 if width > 2 else 1)
    for number, region in enumerate(plan_dispatch_regions(grid, size)):
        root = tmp_path / str(number)
        report = translate_project(
            _coordinate_config(root, target, kind, region), format_output=False
        ).to_json()
        assert report["summary"]["translatedCount"] == 1, report["diagnostics"]


@pytest.mark.parametrize("kind", ["uint2", "uint3"])
@pytest.mark.parametrize(
    "mutation", ["index.x = 4294967295u;", "index.y = 4294967295u;"]
)
def test_mutated_coordinates_cannot_reuse_region_bounds(tmp_path, kind, mutation):
    region = plan_dispatch_regions((3, 2, 1), (1, 1, 1))[0]
    report = translate_project(
        _coordinate_config(tmp_path, "opengl", kind, region, mutation=mutation),
        format_output=False,
    ).to_json()
    assert report["summary"]["translatedCount"] == 0, report
    assert any(item["severity"] == "error" for item in report["diagnostics"])


@pytest.mark.parametrize("missing", ["region", "metadata"])
@pytest.mark.parametrize("kind", ["uint2", "uint3"])
def test_coordinate_contracts_are_both_required(tmp_path, kind, missing):
    region = plan_dispatch_regions((3, 2, 1), (1, 1, 1))[0]
    config = _coordinate_config(
        tmp_path, "opengl", kind, region, bounded=missing != "metadata"
    )
    if missing == "region":
        config.source_options["metal"].pop("target_options")
    report = translate_project(config, format_output=False).to_json()
    assert report["summary"]["translatedCount"] == 0, report
    assert any(item["severity"] == "error" for item in report["diagnostics"])


@pytest.mark.parametrize("kind", ["uint2", "uint3"])
def test_checked_coordinate_extent_cannot_hide_index_overflow(tmp_path, kind):
    region = plan_dispatch_regions((1 << 30, 2, 1), (128, 1, 1))[0]
    report = translate_project(
        _coordinate_config(tmp_path, "opengl", kind, region), format_output=False
    ).to_json()
    assert report["summary"]["translatedCount"] == 0, report
    assert any(
        item["code"] == "project.translate.opengl-index-type-unsupported"
        for item in report["diagnostics"]
    )

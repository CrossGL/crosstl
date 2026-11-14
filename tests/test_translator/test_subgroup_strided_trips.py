"""Subgroup-strided loops whose trip count every SIMD-group shares.

MLX's Winograd transforms split 32 channels among four SIMD-groups with
``for (int c = simd_group_id; c < 32; c += 4)`` around SIMD-group matrix
products. The counter differs between SIMD-groups, but every one runs eight
trips, so OpenGL and DirectX software subgroups, whose collectives are
workgroup barriers, still reach every barrier equally often. Translation
accepts such a loop only in the entry, with the never-written subgroup index
as its start, a literal bound and stride whose trip counts agree for every
SIMD-group of the dispatch, and a counter that cannot overflow.
"""

import os
import re
import sys

import pytest

from crosstl.project import build_native_loader_dispatch_request
from crosstl.translator.codegen.loop_induction import subgroup_strided_trips
from tests.runtime_helpers import _validate
from tests.test_translator.test_boolean_buffer_runtime import _bound_values
from tests.test_translator.test_loop_updates import _execute
from tests.test_translator.test_software_subgroup_product import _package

REQUIRE_ENV = "CROSTL_REQUIRE_COLLECTIVE_STRIDE_RUNTIME"
TARGETS = ("opengl", "directx")
ERRORS = {
    "opengl": "project.translate.opengl-software-subgroup-invalid",
    "directx": "project.translate.directx-software-subgroup-invalid",
}
SOURCE = """#include <metal_stdlib>
using namespace metal;
kernel void strided_groups(device float* results [[buffer(0)]],
                           uint simd_group_id [[simdgroup_index_in_threadgroup]],
                           uint simd_lane_id [[thread_index_in_simdgroup]]) {
    threadgroup float tile[32];
    if (simd_group_id == 0) { tile[simd_lane_id] = -1.0f; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int c = simd_group_id; c < 32; c += 4) {
        float value = simd_sum(float(c * 32 + int(simd_lane_id)));
        if (simd_lane_id == 0) { tile[c] = value; }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    results[simd_group_id * 32 + simd_lane_id + 4] = tile[simd_lane_id];
}
"""
HELPER_SOURCE = """#include <metal_stdlib>
using namespace metal;
static float collect(uint simd_group_id, uint simd_lane_id) {
    float total = 0.0f;
    for (int c = simd_group_id; c < 32; c += 4) {
        total += simd_sum(float(c * 32 + int(simd_lane_id)));
    }
    return total;
}
kernel void strided_groups(device float* results [[buffer(0)]],
                           uint simd_group_id [[simdgroup_index_in_threadgroup]],
                           uint simd_lane_id [[thread_index_in_simdgroup]]) {
    results[simd_group_id * 32 + simd_lane_id + 4] = collect(simd_group_id, simd_lane_id);
}
"""
GUARD = -32768.0


def _bits(type_name):
    return {"char": 8, "uchar": 8, "short": 16, "ushort": 16, "int": 32, "uint": 32}[
        type_name
    ]


def _simulated(type_name, bound, step, operator, count):
    """Run the loop for every subgroup in unbounded integers."""
    unsigned = type_name.startswith("u")
    maximum = (1 << (_bits(type_name) - (not unsigned))) - 1
    if not (0 <= bound <= maximum and 0 < step <= maximum):
        return None
    trips = set()
    for start in range(count):
        value, taken = start, 0
        while value < bound or operator == "<=" and value == bound:
            value += step
            taken += 1
        # The terminating update must still fit the counter's type.
        if value > maximum:
            return None
        trips.add(taken)
    return trips.pop() if len(trips) == 1 else None


@pytest.mark.parametrize("type_name", ["char", "uchar"])
def test_trip_counts_match_simulation_for_every_bound(type_name):
    maximum = (1 << (_bits(type_name) - (type_name == "char"))) - 1
    accepted = 0
    for bound in range(maximum + 1):
        for step in (1, 3, 4, 8, 32, maximum):
            for operator in ("<", "<="):
                for count in (1, 2, 4, 5):
                    expected = _simulated(type_name, bound, step, operator, count)
                    actual = subgroup_strided_trips(
                        type_name, bound, step, operator, count
                    )
                    assert actual == expected, (type_name, bound, step, operator, count)
                    accepted += expected is not None
    assert accepted > 1000


@pytest.mark.parametrize(
    "arguments, expected",
    [
        # MLX's Winograd channel loops: four SIMD-groups, eight trips each.
        (("int", 32, 4, "<", 4), 8),
        (("int", 31, 4, "<=", 4), 8),
        (("short", 32, 4, "<", 2), 8),
        (("int", 32, 4, "<", 3), 8),
        (("uint", 0, 4, "<", 4), 0),
        (("int", 32, 4, "<", 8), None),
        (("int", 30, 4, "<", 4), None),
        (("int", 32, 1, "<", 4), None),
        (("int", 32, 4, "<", 1), 8),
        # The last update of subgroup 3 would reach 2^31 + 3.
        (("int", 2147483644, 4, "<", 4), 536870911),
        (("int", 2147483647, 4, "<=", 4), None),
        (("uint", 2147483647, 4, "<=", 4), 536870912),
        (("int", 32, 4, ">", 4), None),
        (("int", -4, 4, "<", 4), None),
        (("int", 32, 0, "<", 4), None),
        (("int", 32, -4, "<", 4), None),
        (("int", 32, 4, "<", 0), None),
        (("int", True, 4, "<", 4), None),
        (("int", 32, True, "<", 4), None),
        (("int", 32, 4, "<", True), None),
        (("long", 32, 4, "<", 4), None),
        (("int2", 32, 4, "<", 4), None),
        (("float", 32, 4, "<", 4), None),
        (("not a type", 32, 4, "<", 4), None),
        (("int", 32.0, 4, "<", 4), None),
    ],
)
def test_trip_counts_require_concrete_agreeing_bounds(arguments, expected):
    assert subgroup_strided_trips(*arguments) == expected


def _translate(tmp_path, target, source, shape=(32, 4, 1)):
    from crosstl.project import ProjectConfig, translate_project

    (tmp_path / "trips.metal").write_text(source)
    options = {"software_subgroup_width": 32}
    if target == "directx":
        options["relative_wave_shuffle_out_of_range"] = "self"
    return translate_project(
        ProjectConfig(
            root=tmp_path,
            include_patterns=("trips.metal",),
            targets=(target,),
            output_dir="out",
            workgroup_size=shape,
            source_options={"metal": {"target_options": {target: options}}},
        ),
        format_output=False,
    ).to_json()


@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.parametrize(
    "variant, shape",
    [
        ("accepted", (32, 4, 1)),
        ("short-counter", (32, 4, 1)),
        ("inclusive", (32, 4, 1)),
        ("two-groups", (32, 2, 1)),
        ("three-groups", (32, 3, 1)),
        ("local-constants", (32, 4, 1)),
    ],
)
def test_trip_uniform_loops_translate(tmp_path, target, variant, shape):
    source = SOURCE
    if variant == "short-counter":
        source = source.replace("int c", "short c")
    elif variant == "inclusive":
        source = source.replace("c < 32", "c <= 31")
    elif variant == "local-constants":
        source = source.replace(
            "threadgroup float tile[32];",
            "threadgroup float tile[32]; const int groups = 4; const int channels = groups * 8;",
        ).replace("c < 32; c += 4", "c < channels; c += groups")
    report = _translate(tmp_path, target, source, shape)
    assert report["summary"]["translatedCount"] == 1, report["diagnostics"]
    (artifact,) = report["artifacts"]
    text = (tmp_path / artifact["path"]).read_text()
    # The loop survives as written; only its collective is lowered.
    assert "simd_sum" not in text
    assert len(re.findall(r"for \(\w+ c = ", text)) == 1, text


@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.parametrize(
    "variant",
    [
        "unequal-bound",
        "unequal-stride",
        "unit-stride",
        "eight-groups",
        "written-counter",
        "written-index",
        "lane-start",
        "lane-bound",
        "exit",
        "helper",
    ],
)
def test_divergent_subgroup_strided_loops_are_rejected(tmp_path, target, variant):
    source, shape = SOURCE, (32, 4, 1)
    rewrites = {
        "unequal-bound": [("c < 32", "c < 30")],
        "unequal-stride": [("c += 4", "c += 3")],
        "unit-stride": [("c += 4", "c++")],
        "written-counter": [
            ("float value", "if (simd_lane_id == 0) { c += 4; } float value")
        ],
        "written-index": [
            (
                "threadgroup float tile[32];",
                "threadgroup float tile[32]; simd_group_id = simd_lane_id % 4;",
            )
        ],
        "lane-start": [("int c = simd_group_id", "int c = simd_lane_id")],
        "lane-bound": [("c < 32", "c < 32 - int(simd_lane_id % 2)")],
        "exit": [("float value", "if (c == 5) { break; } float value")],
    }
    if variant == "eight-groups":
        shape = (32, 8, 1)
    elif variant == "helper":
        # The proof needs the entry's own subgroup index.
        source = HELPER_SOURCE
    else:
        for old, new in rewrites[variant]:
            assert old in source
            source = source.replace(old, new)
    report = _translate(tmp_path, target, source, shape)
    assert report["summary"]["translatedCount"] == 0, report
    codes = {item["code"] for item in report["diagnostics"]}
    assert ERRORS[target] in codes, codes


def _expected(groups):
    # Subgroup s writes tile[c] for c = s, s + 4, ...: the sum over lanes of
    # c * 32 + lane, exact in binary32. Other entries keep their -1.0.
    tile = [
        1024.0 * channel + 496.0 if channel % 4 < groups else -1.0
        for channel in range(32)
    ]
    return (
        [GUARD] * 4
        + [tile[lane] for _ in range(groups) for lane in range(32)]
        + [GUARD] * 4
    )


@pytest.mark.parametrize("groups", [2, 3, 4])
def test_trip_uniform_loop_executes_natively(tmp_path, groups):
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 for native collective stride checks")
    target = {"darwin": "metal", "linux": "opengl", "win32": "directx"}[sys.platform]
    _, descriptor, package = _package(
        tmp_path, target, "sum", (32, groups, 1), source=SOURCE
    )
    size = 32 * groups + 8
    inputs = _bound_values(
        descriptor,
        {"results": dict(dtype="float32", shape=[size], values=[GUARD] * size)},
    )
    outputs = _bound_values(
        descriptor,
        {"results": dict(dtype="float32", shape=[size], values=_expected(groups))},
    )
    request = build_native_loader_dispatch_request(
        descriptor,
        package,
        inputs,
        outputs,
        {"workgroupCount": [1, 1, 1], "workgroupSize": [32, groups, 1]},
    )
    work = tmp_path / "native"
    work.mkdir()
    _execute(
        request,
        outputs,
        work,
        validate=_validate,
        original_source=SOURCE,
        original_entry="strided_groups",
    )


def test_ci_requires_trip_uniform_execution():
    from pathlib import Path

    from tests import workflow_text
    from tests.ci_helpers import assert_paths_covered

    workflow = (
        Path(__file__).resolve().parents[2]
        / ".github/workflows/demo-project-testing.yml"
    ).read_text()
    step = workflow_text.workflow_step_section(
        workflow, "Validate collective helper arguments"
    )
    assert "tests/test_translator/test_subgroup_strided_trips.py" in step
    assert f'{REQUIRE_ENV}: "1"' in step
    assert "continue-on-error" not in step
    for event in ("push", "pull_request"):
        assert_paths_covered(
            workflow_text.workflow_event_path_filters(workflow, event),
            "tests/test_translator/test_subgroup_strided_trips.py",
        )

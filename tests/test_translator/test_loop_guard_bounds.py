"""Loop guards constrain derived offsets without changing source arithmetic."""

import json
import os
import sys
from dataclasses import replace

import pytest

from crosstl.project import build_native_loader_dispatch_request, translate_project
from crosstl.translator.ast import ForNode
from crosstl.translator.dispatch_regions import plan_dispatch_regions
from tests.runtime_helpers import _validate
from tests.test_translator.test_loop_updates import _execute
from tests.test_translator.test_project_metadata_entry_scope import _config, _packages
from tests.test_translator.test_scalar_return_bounds import _generator


def _analyze(source, facts):
    generator = _generator(source)
    function = generator.function_definitions["run"]
    generator.glsl_workgroup_proof_initial_intervals = lambda _function: dict(facts)
    generator.annotate_glsl_control_flow_intervals(function)
    (loop,) = (
        node for node in generator.walk_ast(function.body) if isinstance(node, ForNode)
    )
    body = getattr(loop.body, "statements", loop.body)
    return body[0]._glsl_control_flow_intervals


@pytest.mark.parametrize(
    "expression,facts,required",
    [
        ("tile + 1u < 3u", (0, 4294967295), 4294967295),
        ("tile - 1u > 3u", (0, 5), 0),
        ("tile * 2u < 4u", (0, 4294967295), 2147483648),
    ],
)
def test_wrapping_unsigned_guard_does_not_exclude_valid_coordinates(
    expression, facts, required
):
    bounds = _analyze(
        "void run(uint tile) { int64_t position = int64_t(tile); "
        f"for (int i = 0; i < 2 && {expression}; ++i) {{ position; position += 0; }} }}",
        {"tile": facts},
    )["position"]
    assert bounds[0] <= required <= bounds[1]


@pytest.mark.parametrize("step", [-3, -1, 0, 1, 3])
@pytest.mark.parametrize("width", [1, 2, 3, 4, 5, 19])
def test_partial_tiles_bound_exact_source_coordinates(step, width):
    source = (
        "void run(int tile) { int x = 2 * tile; "
        f"int position = x * {step}; "
        f"for (int i = 0; i < 2 && x + i < {width}; ++i) {{ "
        f"position; position += {step}; }} }}"
    )
    bounds = _analyze(source, {"tile": (0, (width - 1) // 2)})["position"]
    expected = [coordinate * step for coordinate in range(width)]
    assert bounds == (min(expected), max(expected))


@pytest.mark.parametrize(
    "comparison",
    [
        "x + i < 3",
        "x + i <= 2",
        "3 > x + i",
        "2 >= x + i",
        "-x - i > -3",
        "-x - i >= -2",
    ],
)
@pytest.mark.parametrize("start", [0, 1])
def test_equivalent_guard_forms_and_nonzero_counter_start(comparison, start):
    bounds = _analyze(
        "void run(int tile) { int x = tile * 2; int position = -x; "
        f"for (int i = {start}; i < 2 && {comparison}; ++i) {{ position; position -= 1; }} }}",
        {"tile": (0, 1)},
    )["position"]
    expected = [
        -(tile * 2) - (i - start)
        for tile in range(2)
        for i in range(start, 2)
        if tile * 2 + i < 3
    ]
    assert bounds == (min(expected), max(expected))


@pytest.mark.parametrize(
    "fault",
    [
        "break",
        "continue",
        "conditional-update",
        "shadow",
        "escape",
        "later-mutation",
        "unknown-call",
        "variable-step",
    ],
)
def test_unmodeled_control_flow_and_dependencies_do_not_tighten_entries(fault):
    prefix = "int x = 2 * tile; int position = -x;"
    update = "position -= 1;"
    body_prefix = ""
    if fault in {"break", "continue"}:
        body_prefix = f"if (tile == 0) {{ {fault}; }}"
    elif fault == "conditional-update":
        update = "if (tile == 0) { position -= 1; }"
    elif fault == "shadow":
        body_prefix = "{ int tile = 0; }"
    elif fault == "escape":
        prefix += "int& alias = tile;"
    elif fault == "later-mutation":
        prefix += "tile = 0;"
    elif fault == "unknown-call":
        body_prefix = "unknown(tile);"
    else:
        update = "position -= tile;"
    bounds = _analyze(
        f"void run(int tile) {{ {prefix} for (int i = 0; i < 2 && tile * 2 + i < 3; ++i) {{ "
        f"{body_prefix} position; {update} }} }}",
        {"tile": (0, 1)},
    ).get("position")
    assert bounds is None or bounds[0] <= -3


@pytest.mark.parametrize(
    "kind,guard,facts,valid",
    [
        ("int", "tile < 1u", (-2, 2), [0]),
        ("uint", "int(tile) < 0", (0, 4294967295), [2147483648, 4294967295]),
        ("int", "uint(tile) > 3u", (-2, 2), [-2, -1]),
        ("int", "int16_t(tile) < 2", (0, 65537), [32768, 65536, 65537]),
    ],
)
def test_source_conversions_never_drop_valid_guard_coordinates(
    kind, guard, facts, valid
):
    bounds = _analyze(
        f"void run({kind} tile) {{ int64_t position = int64_t(tile); "
        f"for (int i = 0; i < 2 && {guard}; ++i) {{ position; position += 0; }} }}",
        {"tile": facts},
    ).get("position")
    assert bounds is None or all(bounds[0] <= value <= bounds[1] for value in valid)


@pytest.mark.parametrize("count", [64, 65])
def test_iteration_budget_leaves_coarse_proof_when_exceeded(count):
    bounds = _analyze(
        f"void run() {{ int position = 0; for (int i = 0; i < {count} && i < 1; ++i) {{ position; position += 1; }} }}",
        {},
    )["position"]
    assert bounds == (0, 0 if count == 64 else count - 1)


def test_large_launch_is_refined_without_enumerating_invocations():
    bounds = _analyze(
        "void run(int tile) { int x = 2 * tile; int position = -x; "
        "for (int i = 0; i < 2 && x + i < 1000001; ++i) { position; position -= 1; } }",
        {"tile": (0, 500000)},
    )["position"]
    assert bounds == (-1000000, 0)


def test_expression_depth_limit_preserves_conservative_entries():
    expression = "x" + " + 0" * 40
    bounds = _analyze(
        "void run(int tile) { int x = 2 * tile; int position = -x; "
        f"for (int i = 0; i < 2 && {expression} + i < 3; ++i) {{ position; position -= 1; }} }}",
        {"tile": (0, 1)},
    )["position"]
    assert bounds == (-3, 0)


def test_shared_budget_never_returns_a_partial_iteration_proof():
    definitions = "int v0 = x; " + " ".join(
        f"int v{i} = v{i - 1};" for i in range(1, 70)
    )
    bounds = _analyze(
        f"void run(int tile) {{ int x = 2 * tile; {definitions} int position = -v69; "
        "for (int i = 0; i < 64 && x + i < 65; ++i) { position; position -= 1; } }",
        {"tile": (0, 1)},
    )["position"]
    assert bounds == (-65, 0)


SOURCE = """#include <metal_stdlib>
using namespace metal;
kernel void tiled_copy(
    constant int* shape [[buffer(0)]], constant long* a [[buffer(1)]],
    constant long* b [[buffer(2)]], constant long& src_base [[buffer(3)]],
    constant long& dst_base [[buffer(4)]], const device int* src [[buffer(5)]],
    device int* dst [[buffer(6)]], uint3 grid [[thread_position_in_grid]]) {
    int x = 2 * int(grid.x);
    int2 at = int2(x * int(a[1]) + int(grid.y) * int(a[0]),
                   x * int(b[1]) + int(grid.y) * int(b[0]));
    int src_step = int(a[1]);
    int dst_step = int(b[1]);
    int width = shape[1];
    for (int i = 0; i < 2 && x + i < width; ++i) {
        dst[dst_base + at.y] = src[src_base + at.x];
        at.x += src_step;
        at.y += dst_step;
    }
}
"""
GUARD = 123456


def _metadata(reverse, *, width=3):
    return dict(
        shape=("int32", [2, width]),
        a=("int64", [width, -1 if reverse else 1]),
        b=("int64", [1, 2]),
        src_base=("int64", [width - 1 if reverse else 0]),
        dst_base=("int64", [0]),
    )


def _tiled_config(root, target, reverse, *, width=3, offset=0, layout_width=3):
    metadata = _metadata(reverse, width=layout_width)
    metadata["shape"] = ("int32", [2, width])
    metadata["dst_base"] = ("int64", [offset])
    conditions = [
        dict(
            entryPoint="tiled_copy",
            parameter=name,
            minimum=value,
            maximum=value,
            **({"element": i} if len(values) > 1 else {}),
        )
        for name, (_, values) in metadata.items()
        for i, value in enumerate(values)
    ]
    config = _config(
        root, target, source=SOURCE, entries=("tiled_copy",), conditions=conditions
    )
    options = dict(config.source_options["metal"])
    if target == "opengl":
        (region,) = plan_dispatch_regions((2, 2, 1), (1, 1, 1))
        options["target_options"] = {target: dict(dispatch_region=region.to_json())}
    return replace(config, workgroup_size=(1, 1, 1), source_options={"metal": options})


@pytest.mark.parametrize("target", ["metal", "opengl", "directx"])
@pytest.mark.parametrize("reverse", [False, True])
def test_tight_tiled_layout_translates_with_checked_metadata(tmp_path, target, reverse):
    config = _tiled_config(tmp_path, target, reverse)
    report = translate_project(config, format_output=False).to_json()
    assert report["summary"]["translatedCount"] == 1, report["diagnostics"]
    assert not config.index_range_assertions


@pytest.mark.parametrize("width,offset", [(4, 0), (3, -1)])
def test_guard_proof_rejects_actual_negative_indices(tmp_path, width, offset):
    config = _tiled_config(tmp_path, "opengl", True, width=width, offset=offset)
    report = translate_project(config, format_output=False).to_json()
    assert report["summary"]["translatedCount"] == 0
    assert any(
        item["code"] == "project.translate.opengl-index-type-unsupported"
        for item in report["diagnostics"]
    )


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("width", [3, 4])
def test_tight_tiled_layout_executes_natively(tmp_path, reverse, width):
    if os.environ.get("CROSTL_REQUIRE_METADATA_PRECONDITIONS") != "1":
        pytest.skip(
            "set CROSTL_REQUIRE_METADATA_PRECONDITIONS=1 for native metadata checks"
        )
    target = {"darwin": "metal", "linux": "opengl", "win32": "directx"}[sys.platform]
    config = _tiled_config(tmp_path, target, reverse, width=width, layout_width=width)
    _, package, (descriptor,) = _packages(config)
    words = [101 + i * 7 for i in range(2 * width)]
    expected = [GUARD] * (2 * width + 5)
    for row in range(2):
        for column in range(width):
            index = row * width + (width - 1 - column if reverse else column)
            expected[row + 2 * column] = words[index]
    values = {
        **_metadata(reverse, width=width),
        "src": ("int32", words),
        "dst": ("int32", [GUARD] * len(expected)),
    }
    inputs, outputs = {}, {}
    for binding in descriptor["bindings"]:
        name = binding["provenance"]["sourceResource"]["parameter"]
        dtype, items = values[name]
        inputs[binding["name"]] = dict(dtype=dtype, shape=[len(items)], values=items)
        if name == "dst":
            outputs[binding["name"]] = dict(
                dtype=dtype, shape=[len(expected)], values=expected
            )
    request = build_native_loader_dispatch_request(
        descriptor,
        package,
        inputs,
        outputs,
        dict(workgroupCount=[2, 2, 1], workgroupSize=[1, 1, 1]),
        observe_inputs=[
            item["name"] for item in descriptor["bindings"] if item["access"] == "read"
        ],
    )
    native = tmp_path / "native"
    native.mkdir()
    _execute(
        request,
        outputs,
        native,
        validate=_validate,
        original_source=SOURCE,
        original_entry="tiled_copy",
    )
    details = json.loads((native / "evidence.json").read_text())["records"][
        "generated"
    ]["details"]
    assert len(details["metadataPreconditions"]) == 8
    assert len(details["inputObservations"]) == 6
    assert all(item["unchanged"] for item in details["inputObservations"].values())

"""Loop guards constrain derived offsets without changing source arithmetic."""

import json
import os
import sys
from dataclasses import replace

import pytest

from crosstl.project import build_native_loader_dispatch_request, translate_project
from crosstl.translator.ast import ForNode, IfNode
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


@pytest.mark.parametrize(
    "condition,expected",
    [
        ("i < 2 && offset + i < size", (0, 1)),
        ("offset + i < size && i < 2", (0, 1)),
        ("offset + i < size", (0, 22)),
        ("i + offset < size", (0, 22)),
        ("offset + i <= size", (0, 23)),
    ],
)
@pytest.mark.parametrize("start", [0, 1])
def test_counter_entry_bounds_preserve_the_initial_value(condition, expected, start):
    bounds = _analyze(
        "void run(int64_t offset, int64_t size) { "
        f"for (int i = {start}; {condition}; ++i) {{ i; }} }}",
        {"offset": (0, 22), "size": (23, 23)},
    )
    assert bounds["i"] == (start, expected[1])


@pytest.mark.parametrize(
    "prefix,condition,body,update",
    [
        ("", "offset + i < size", "offset -= 1;", "++i"),
        ("", "offset + i < size", "size += 1;", "++i"),
        ("", "offset + i < size", "i -= 1;", "++i"),
        ("int& alias = offset;", "offset + i < size", "", "++i"),
        ("", "offset + i < size", "unknown(offset);", "++i"),
        ("", "offset + i < size", "{ int offset = 0; }", "++i"),
        ("", "offset + i < size", "{ int i = 0; } i = 0;", "++i"),
        ("", "offset + i < size", "", "--i"),
        ("", "offset + i < size || i < 30", "", "++i"),
        ("", "offset + i < size++", "", "++i"),
    ],
)
def test_counter_proof_rejects_unstable_dependencies(prefix, condition, body, update):
    bounds = _analyze(
        f"void run(int64_t offset, int64_t size) {{ {prefix} "
        f"for (int i = 0; {condition}; {update}) {{ i; {body} }} }}",
        {"offset": (0, 22), "size": (23, 23)},
    ).get("i")
    assert bounds is None or bounds[0] < 0


def test_counter_proof_distinguishes_a_nested_shadow_binding():
    bounds = _analyze(
        "void run(int64_t offset, int64_t size) { "
        "for (int i = 0; offset + i < size; ++i) { i; { int i = 0; ++i; } } }",
        {"offset": (0, 22), "size": (23, 23)},
    )["i"]
    assert bounds == (0, 22)


@pytest.mark.parametrize(
    "counter,offset_kind,offset,size_kind,size,condition",
    [
        ("int", "int", 2147483646, "int64_t", 2147483650, "offset + i < size"),
        ("uint", "uint", 4294967294, "int64_t", 4294967296, "offset + i < size"),
        ("int", "int64_t", 0, "int64_t", 2147483648, "offset + i < size"),
        (
            "int",
            "int64_t",
            9223372036854775807,
            "int64_t",
            9223372036854775807,
            "offset + i <= size",
        ),
        ("int", "int64_t", -1, "uint64_t", 2, "offset + i < size"),
    ],
)
def test_counter_proof_does_not_assume_source_arithmetic_is_mathematical(
    counter, offset_kind, offset, size_kind, size, condition
):
    generator = _generator(
        f"void run({offset_kind} offset, {size_kind} size) {{ "
        f"for ({counter} i = 0; {condition}; ++i) {{ i; }} }}"
    )
    function = generator.function_definitions["run"]
    generator.glsl_workgroup_proof_initial_intervals = lambda _function: {
        "offset": (offset, offset),
        "size": (size, size),
    }
    generator.annotate_glsl_control_flow_intervals(function)
    (loop,) = (
        node for node in generator.walk_ast(function.body) if isinstance(node, ForNode)
    )
    assert not loop._glsl_checked_zero_trip
    body = getattr(loop.body, "statements", loop.body)
    bounds = getattr(body[0], "_glsl_control_flow_intervals", {}).get("i")
    assert bounds is None or bounds[0] <= 0 and bounds[1] >= 2147483647


def test_counter_entry_bound_is_not_an_exact_pointer_rebase_sequence():
    generator = _generator(
        "void run(int64_t offset, int64_t size) { "
        "for (int i = 0; i < 2 && offset + i < size; ++i) { i; } }"
    )
    function = generator.function_definitions["run"]
    facts = {"offset": (0, 22), "size": (23, 23)}
    generator.glsl_workgroup_proof_initial_intervals = lambda _function: dict(facts)
    generator.annotate_glsl_control_flow_intervals(function)
    (loop,) = (
        node for node in generator.walk_ast(function.body) if isinstance(node, ForNode)
    )
    body = getattr(loop.body, "statements", loop.body)
    assert body[0]._glsl_control_flow_intervals["i"] == (0, 1)
    assert (
        generator.glsl_private_pointer_rebase_loop_iteration_values(
            loop, {**facts, "i": (0, 0)}, {}
        )
        is None
    )


TAIL_HEADER = """#include <metal_stdlib>
using namespace metal;
kernel void tail_copy(const device uint* src [[buffer(0)]],
                     device uint* dst [[buffer(1)]],
                     constant long& size [[buffer(2)]],
                     uint3 index [[thread_position_in_grid]],
                     uint3 grid_dim [[threads_per_grid]]) {
"""
TAIL_BODY = """long offset = 2 * (index.x + grid_dim.x * long(index.y));
if (offset + 2 > size) {
    for (int i = 0; offset + i < size; ++i) {
        dst[offset + i] = src[offset + i];
    }
} else {
    for (int i = 0; i < 2; ++i) {
        dst[offset + i] = src[offset + i];
    }
}
"""
TAIL_CASES = {
    "affine": (TAIL_BODY, 23, (3, 4, 1), 23),
    "conjunction": (
        TAIL_BODY.replace("offset + i < size;", "i < 2 && offset + i < size;"),
        23,
        (3, 4, 1),
        23,
    ),
    "volume": (
        "long offset = 4 * index.x + size * (index.y + long(grid_dim.y) * index.z);\n"
        "for (int i = 0; i < 4 && int(4 * index.x) + i < size; ++i) {\n"
        "    dst[offset + i] = src[offset + i];\n}\n",
        5,
        (2, 4, 6),
        120,
    ),
}


def _tail_config(root, target, case):
    body, size, grid, _count = TAIL_CASES[case]
    config = _config(
        root,
        target,
        source=TAIL_HEADER + body + "}\n",
        entries=("tail_copy",),
        conditions=[
            dict(entryPoint="tail_copy", parameter="size", minimum=size, maximum=size)
        ],
    )
    options = dict(config.source_options["metal"])
    if target == "opengl":
        (region,) = plan_dispatch_regions(grid, (1, 1, 1))
        options["target_options"] = {target: dict(dispatch_region=region.to_json())}
    return replace(config, workgroup_size=(1, 1, 1), source_options={"metal": options})


@pytest.mark.parametrize("target", ["metal", "opengl", "directx"])
@pytest.mark.parametrize("case", TAIL_CASES)
def test_tail_counter_translates_with_checked_source_metadata(tmp_path, target, case):
    config = _tail_config(tmp_path, target, case)
    report = translate_project(config, format_output=False).to_json()
    assert report["summary"]["translatedCount"] == 1, report["diagnostics"]
    assert not config.index_range_assertions


def test_tail_counter_rejects_an_unchecked_reference_bound(tmp_path):
    config = replace(
        _tail_config(tmp_path, "opengl", "affine"), metadata_preconditions=()
    )
    report = translate_project(config, format_output=False).to_json()
    assert report["summary"]["translatedCount"] == 0
    assert any(
        item["code"] == "project.translate.opengl-index-type-unsupported"
        for item in report["diagnostics"]
    )


@pytest.mark.parametrize("case", TAIL_CASES)
def test_tail_counter_executes_natively(tmp_path, case):
    if os.environ.get("CROSTL_REQUIRE_METADATA_PRECONDITIONS") != "1":
        pytest.skip(
            "set CROSTL_REQUIRE_METADATA_PRECONDITIONS=1 for native metadata checks"
        )
    target = {"darwin": "metal", "linux": "opengl", "win32": "directx"}[sys.platform]
    config = _tail_config(tmp_path, target, case)
    _, package, (descriptor,) = _packages(config)
    body, size, grid, count = TAIL_CASES[case]
    words = [17 + 13 * i for i in range(count)]
    guard = 0xABCDEF01
    expected = words + [guard] * 32
    data = {
        "src": ("uint32", words),
        "dst": ("uint32", [guard] * len(expected)),
        "size": ("int64", [size]),
    }
    inputs, outputs = {}, {}
    # Generated execution inputs (DirectX workgroup-count metadata) have no
    # source resource; the dispatch request derives them from the launch.
    source_bindings = [
        binding
        for binding in descriptor["bindings"]
        if "executionInput" not in binding["provenance"]
    ]
    assert {
        binding["provenance"]["sourceResource"]["parameter"]
        for binding in source_bindings
    } == set(data)
    for binding in source_bindings:
        name = binding["provenance"]["sourceResource"]["parameter"]
        dtype, items = data[name]
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
        dict(workgroupCount=list(grid), workgroupSize=[1, 1, 1]),
        observe_inputs=[
            binding["name"]
            for binding in source_bindings
            if binding["access"] == "read"
        ],
    )
    assert not request.execution_plan.diagnostics
    native = tmp_path / "native"
    native.mkdir()
    _execute(
        request,
        outputs,
        native,
        validate=_validate,
        original_source=TAIL_HEADER + body + "}\n",
        original_entry="tail_copy",
    )
    details = json.loads((native / "evidence.json").read_text())["records"][
        "generated"
    ]["details"]
    assert len(details["metadataPreconditions"]) == 1
    assert len(details["inputObservations"]) == 2
    assert all(item["unchanged"] for item in details["inputObservations"].values())


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


REFUTED_SOURCE = """#include <metal_stdlib>
using namespace metal;
kernel void tail_sum(const device uint* in [[buffer(0)]],
                     device uint* out [[buffer(1)]],
                     constant size_t& in_size [[buffer(2)]],
                     constant size_t& row_size [[buffer(3)]],
                     uint3 gid [[threadgroup_position_in_grid]],
                     uint3 lid [[thread_position_in_threadgroup]],
                     uint3 lsize [[threads_per_threadgroup]]) {
    uint total = 0;
    long start = gid.y * long(row_size);
    long actual = (start + row_size <= in_size) ? row_size : in_size - start;
    long blocks = actual / (lsize.x * 4);
    int extra = actual - blocks * (lsize.x * 4);
    extra -= lid.x * 4;
    start += lid.x * 4;
    in += start;
    if (extra >= 4) {
        blocks++;
        extra = 0;
    }
    for (long b = 0; b < blocks; b++) {
        for (int i = 0; i < 4; i++) {
            total += in[i];
        }
        in += lsize.x * 4;
    }
    TAIL
    out[gid.y * lsize.x + lid.x] = total;
}
"""
REFUTED_LOOP = "for (int i = 0; i < extra; i++) { total += in[i]; }"
# Each tail reads through the advanced wide pointer only where ``extra > 0``.
# When the row is an exact multiple of the block, the checked intervals refute
# that guard, so the reads need no composed-offset proof and are not emitted.
REFUTED_TAILS = {
    "then": f"if (extra > 0) {{ {REFUTED_LOOP} }}",
    "else": f"if (extra <= 0) {{ total += 1u; }} else {{ {REFUTED_LOOP} }}",
    "else-if": (
        f"if (extra < -64) {{ total += 2u; }} else if (extra > 0) {{ {REFUTED_LOOP} }}"
    ),
    "chained": (
        "if (extra <= 0) { total += 1u; } "
        f"else if (extra > 0) {{ {REFUTED_LOOP} }} else {{ total += 2u; }}"
    ),
}
REFUTED_WIDTH = 4
REFUTED_ROWS = 1


def _refuted_source(case):
    return REFUTED_SOURCE.replace("TAIL", REFUTED_TAILS[case])


def _refuted_config(root, target, case, row):
    conditions = [
        dict(entryPoint="tail_sum", parameter=name, minimum=value, maximum=value)
        for name, value in (("in_size", REFUTED_ROWS * row), ("row_size", row))
    ]
    config = _config(
        root,
        target,
        source=_refuted_source(case),
        entries=("tail_sum",),
        conditions=conditions,
    )
    options = dict(config.source_options["metal"])
    if target != "metal":
        (region,) = plan_dispatch_regions(
            (REFUTED_WIDTH, REFUTED_ROWS, 1), (REFUTED_WIDTH, 1, 1)
        )
        options["target_options"] = {target: dict(dispatch_region=region.to_json())}
    return replace(
        config,
        workgroup_size=(REFUTED_WIDTH, 1, 1),
        source_options={"metal": options},
    )


def _refuted_reference(case, words, row):
    """Evaluate the kernel's source semantics for every invocation."""

    result = []
    for group in range(REFUTED_ROWS):
        for lane in range(REFUTED_WIDTH):
            total = 0
            start = group * row
            actual = row if start + row <= REFUTED_ROWS * row else 0
            blocks = actual // (REFUTED_WIDTH * 4)
            extra = actual - blocks * REFUTED_WIDTH * 4 - lane * 4
            position = start + lane * 4
            if extra >= 4:
                blocks, extra = blocks + 1, 0
            for _ in range(blocks):
                total += sum(words[position : position + 4])
                position += REFUTED_WIDTH * 4
            tail = sum(words[position : position + max(extra, 0)])
            if case == "then":
                total += tail
            elif case == "else":
                total += 1 if extra <= 0 else tail
            elif case == "else-if":
                total += 2 if extra < -64 else tail if extra > 0 else 0
            else:
                total += 1 if extra <= 0 else tail
            result.append(total & 0xFFFFFFFF)
    return result


@pytest.mark.parametrize("target", ["metal", "opengl", "directx"])
@pytest.mark.parametrize("case", REFUTED_TAILS)
@pytest.mark.parametrize("row,refuted", [(16, True), (17, False), (32, True)])
def test_refuted_branch_reads_need_no_offset_proof(
    tmp_path, target, case, row, refuted
):
    config = _refuted_config(tmp_path, target, case, row)
    report = translate_project(config, format_output=False).to_json()
    assert report["summary"]["translatedCount"] == 1, report["diagnostics"]
    assert not config.index_range_assertions
    if target != "opengl":
        return
    (artifact,) = report["artifacts"]
    generated = (tmp_path / artifact["path"]).read_text()
    # Only the refuted tail loop is removed; the reachable block loop and the
    # reachable sibling branches are emitted unchanged.
    assert ("(i < extra)" in generated) is not refuted
    assert "(i < 4)" in generated
    assert ("1u" in generated) is (case in {"else", "chained"})


def test_refuted_branch_keeps_bodies_reached_by_another_iteration():
    generator = _generator(
        "void run(int tile) { int total = 0; "
        "for (int i = 0; i < 2; ++i) { if (i == 1) { total += 1; } "
        "if (tile > 3) { total += 2; } } }"
    )
    function = generator.function_definitions["run"]
    generator.glsl_workgroup_proof_initial_intervals = lambda _function: {
        "tile": (0, 1)
    }
    generator.annotate_glsl_control_flow_intervals(function)
    branches = [
        node for node in generator.walk_ast(function.body) if isinstance(node, IfNode)
    ]
    marked = [
        getattr(statement, "_glsl_checked_unreachable", False)
        for branch in branches
        for statement in getattr(branch.if_body, "statements", branch.if_body)
    ]
    # ``i == 1`` is false on the first replayed iteration only, while
    # ``tile > 3`` is false on every iteration.
    assert marked == [False, True]


@pytest.mark.parametrize("case", REFUTED_TAILS)
@pytest.mark.parametrize("row", [16, 17])
def test_refuted_branch_tails_execute_natively(tmp_path, case, row):
    if os.environ.get("CROSTL_REQUIRE_METADATA_PRECONDITIONS") != "1":
        pytest.skip(
            "set CROSTL_REQUIRE_METADATA_PRECONDITIONS=1 for native metadata checks"
        )
    target = {"darwin": "metal", "linux": "opengl", "win32": "directx"}[sys.platform]
    config = _refuted_config(tmp_path, target, case, row)
    _, package, (descriptor,) = _packages(config)
    words = [17 + 13 * i for i in range(REFUTED_ROWS * row)]
    expected = _refuted_reference(case, words, row) + [GUARD] * 4
    data = {
        "in": ("uint32", words),
        "out": ("uint32", [GUARD] * len(expected)),
        "in_size": ("uint64", [REFUTED_ROWS * row]),
        "row_size": ("uint64", [row]),
    }
    source_bindings = [
        binding
        for binding in descriptor["bindings"]
        if "executionInput" not in binding["provenance"]
    ]
    assert {
        binding["provenance"]["sourceResource"]["parameter"]
        for binding in source_bindings
    } == set(data)
    inputs, outputs = {}, {}
    for binding in source_bindings:
        name = binding["provenance"]["sourceResource"]["parameter"]
        dtype, items = data[name]
        inputs[binding["name"]] = dict(dtype=dtype, shape=[len(items)], values=items)
        if name == "out":
            outputs[binding["name"]] = dict(
                dtype=dtype, shape=[len(expected)], values=expected
            )
    request = build_native_loader_dispatch_request(
        descriptor,
        package,
        inputs,
        outputs,
        dict(workgroupCount=[1, REFUTED_ROWS, 1], workgroupSize=[REFUTED_WIDTH, 1, 1]),
        observe_inputs=[
            binding["name"]
            for binding in source_bindings
            if binding["access"] == "read"
        ],
    )
    assert not request.execution_plan.diagnostics
    native = tmp_path / "native"
    native.mkdir()
    _execute(
        request,
        outputs,
        native,
        validate=_validate,
        original_source=_refuted_source(case),
        original_entry="tail_sum",
    )
    details = json.loads((native / "evidence.json").read_text())["records"][
        "generated"
    ]["details"]
    assert len(details["metadataPreconditions"]) == 2
    assert all(item["unchanged"] for item in details["inputObservations"].values())

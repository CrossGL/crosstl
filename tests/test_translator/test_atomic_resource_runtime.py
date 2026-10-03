"""Resource-reference atomics preserve storage, old values and side effects."""

import os
import pickle
import shutil
import sys
from pathlib import Path

import pytest

from crosstl.project import (
    ProjectConfig,
    build_native_loader_dispatch_request,
    translate_project,
)
from crosstl.translator import parse
from crosstl.translator.ast import ArrayAccessNode, FunctionCallNode, ReturnNode
from crosstl.translator.codegen.resource_aggregates import (
    ResourceAggregateError,
    lower_resource_aggregates,
)
from tests.test_translator.test_boolean_buffer_runtime import _bound_values
from tests.test_translator.test_loop_updates import _execute
from tests.test_translator.test_metal_builtin_ownership import _compile
from tests.test_translator.test_software_subgroup_product import _package

REQUIRE_ENV = "CROSTL_REQUIRE_ATOMIC_RESOURCE_RUNTIME"
OPERATIONS = ("add", "min", "max", "and", "or", "xor", "exchange", "store")
CASES = (*OPERATIONS, "alias", "selection", "effects", "contention", "scalar", "owned")
GUARD = 0x7155AACC


def _source(case, scalar):
    operation = case if case in OPERATIONS else "add"
    intrinsic = (
        operation if operation in {"exchange", "store"} else f"fetch_{operation}"
    )
    pointee = f"atomic_{scalar}" if case == "scalar" else "Counter"
    member = "" if case == "scalar" else ".value"
    value_type = scalar if case == "owned" else f"atomic_{scalar}"
    declarations = f"struct Counter {{ {value_type} value; }};"
    call = f"atomic_{intrinsic}_explicit(&view.counters[index]{member}, value, memory_order_relaxed)"
    if case == "owned":
        declarations += f"{scalar} atomicAdd({scalar} target, {scalar} value) {{ return target + value; }}"
        call = "atomicAdd(view.counters[index].value, value)"
    returned = "void" if operation == "store" else scalar
    update = (
        f"modify(view, index, value); observed[tid] = value;"
        if operation == "store"
        else f"observed[tid] = modify(view, index, value);"
    )
    setup = ""
    if case == "alias":
        setup = "View snapshot = view; view.counters = alternate + 1;"
        update = "observed[tid] = modify(snapshot, index, value);"
    elif case == "selection":
        setup = "if ((tid & 1) != 0) { view.counters = alternate + 1; }"
    elif case == "effects":
        update = f"{scalar} old = modify(view, index++, value++); observed[tid] = old + {scalar}(100) * {scalar}(index) + {scalar}(10000) * value;"
    elif case == "contention":
        update = "for (uint iteration = 0; iteration < 16; ++iteration) { modify(view, index, value); } observed[tid] = value;"
    return f"""#include <metal_stdlib>
using namespace metal;
{declarations}
struct View {{ device {pointee}* counters; }};
{returned} modify(const thread View& view, uint index, {scalar} value) {{
    return {call};
}}
kernel void resource_atomic(const device {scalar}* values [[buffer(0)]],
                            device {pointee}* results [[buffer(1)]],
                            device {pointee}* alternate [[buffer(2)]],
                            device {scalar}* observed [[buffer(3)]],
                            uint tid [[thread_position_in_grid]]) {{
    View view{{results}};
    view.counters += 1;
    {setup}
    uint index = {"0" if case == "contention" else "tid"};
    {scalar} value = values[tid % 4];
    {update}
}}
"""


def _workload(case, scalar):
    values = [3, 7, 11, 19]
    initial = [(-5 - i) if scalar == "int" else (0x80000001 + i) for i in range(4)]
    first = [GUARD, *initial, GUARD, GUARD]
    second = [GUARD, *(v + 100 for v in initial), GUARD, GUARD]
    count = 64 if case == "contention" else 4
    observed = [GUARD] * (count + 2)
    dtype = "int32" if scalar == "int" else "uint32"

    def typed(values, aggregate=False):
        return {
            "dtype": dtype,
            "shape": [len(values), 1] if aggregate else [len(values)],
            "values": list(values),
        }

    inputs = {
        "values": typed(values),
        "results": typed(first, case != "scalar"),
        "alternate": typed(second, case != "scalar"),
        "observed": typed(observed),
    }
    for tid in range(count):
        destination = second if case == "selection" and tid & 1 else first
        index = 1 if case == "contention" else tid + 1
        old, value = destination[index], values[tid % 4]
        operation = case if case in OPERATIONS else "add"
        if case == "contention":
            destination[index] += 16 * value
            observed[tid] = value
        elif case == "owned":
            observed[tid] = old + value
        else:
            destination[index] = {
                "add": old + value,
                "min": min(old, value),
                "max": max(old, value),
                "and": old & value,
                "or": old | value,
                "xor": old ^ value,
                "exchange": value,
                "store": value,
            }[operation]
            observed[tid] = value if case == "store" else old
            if case == "effects":
                observed[tid] += 100 * (tid + 1) + 10000 * (value + 1)
        if scalar == "uint":
            destination[index] &= 0xFFFFFFFF
            observed[tid] &= 0xFFFFFFFF
    outputs = {
        "results": typed(first, case != "scalar"),
        "alternate": typed(second, case != "scalar"),
        "observed": typed(observed),
    }
    return inputs, outputs, (16 if case == "contention" else 1), count


def _request(root, target, case, scalar):
    inputs, outputs, width, count = _workload(case, scalar)
    original, descriptor, package = _package(
        root,
        target,
        scalar,
        (width, 1, 1),
        source=_source(case, scalar),
        software_subgroups=False,
        index_range_assertions=(
            (
                {
                    "source": "products.metal",
                    "expression": "reference.offset + index",
                    "minimum": 0,
                    "maximum": count + 1,
                },
            )
            if target == "opengl"
            else ()
        ),
    )
    expected = _bound_values(descriptor, outputs)
    request = build_native_loader_dispatch_request(
        descriptor,
        package,
        _bound_values(descriptor, inputs),
        expected,
        {"workgroupCount": [count // width, 1, 1], "workgroupSize": [width, 1, 1]},
        expected_target=target,
    )
    assert not request.execution_plan.diagnostics
    return original, request, expected


@pytest.mark.parametrize("target", ("metal", "directx", "opengl"))
@pytest.mark.parametrize("scalar", ("int", "uint"))
@pytest.mark.parametrize("case", CASES)
def test_resource_atomics_compile(tmp_path, target, scalar, case):
    tool = {"metal": "xcrun", "directx": "dxc", "opengl": "glslangValidator"}[target]
    if not shutil.which(tool):
        pytest.skip(f"{tool} is not installed")
    _, request, _ = _request(tmp_path, target, case, scalar)
    _, module = _compile(request.artifact_path.read_text(), target, tmp_path)
    assert module.is_file() and module.stat().st_size


@pytest.mark.parametrize("scalar", ("int", "uint"))
@pytest.mark.parametrize("case", CASES)
def test_resource_atomics_execute_natively(tmp_path, scalar, case):
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 for required resource atomic execution")
    target = {"darwin": "metal", "win32": "directx", "linux": "opengl"}[sys.platform]
    original, request, expected = _request(tmp_path, target, case, scalar)
    _execute(
        request,
        expected,
        tmp_path,
        original_source=original,
        original_entry="resource_atomic",
    )


@pytest.mark.parametrize("target", ("directx", "opengl"))
@pytest.mark.parametrize("failure", ("readonly", "member-array", "float-minimum"))
def test_resource_atomics_reject_unsupported_destinations(tmp_path, target, failure):
    source = _source("add", "int")
    if failure == "readonly":
        source = source.replace("device Counter*", "const device Counter*")
    elif failure == "member-array":
        source = source.replace("atomic_int value;", "atomic_int value[2];").replace(
            ".value,", ".value[index & 1],"
        )
    else:
        source = _source("min", "float")
    (tmp_path / "source.metal").write_text(source)
    report = translate_project(
        ProjectConfig(
            root=tmp_path,
            include_patterns=("source.metal",),
            targets=(target,),
            output_dir="out",
            workgroup_size=(1, 1, 1),
        ),
        format_output=False,
    ).to_json()
    assert report["summary"]["failedCount"] == 1
    assert any(
        item["code"] == "project.translate.resource-aggregate-unsupported"
        for item in report["diagnostics"]
    ), report["diagnostics"]
    for artifact in report["artifacts"]:
        assert not (tmp_path / artifact["path"]).exists()


@pytest.mark.parametrize("operation", ("atomicCompSwap", "atomicCompareExchange"))
def test_resource_atomics_preserve_compare_operands_and_source_ast(operation):
    ast = parse(f"""shader T {{
        struct Cursor {{ device int* data; }}
        compute {{ void main(RWStructuredBuffer<int> output @buffer(0)) {{
            Cursor cursor = Cursor(output);
            output[0] = {operation}(cursor.data[1], 7, 9);
        }} }}
    }}""")
    original = pickle.dumps(ast)
    lowered = lower_resource_aggregates(ast)
    assert pickle.dumps(ast) == original
    helper = next(
        function
        for function in lowered.functions
        if function.name.startswith(f"crosstl_resource_{operation}_")
    )
    calls = [
        node
        for node in helper.body.walk()
        if isinstance(node, FunctionCallNode) and node.function.name == operation
    ]
    assert len(calls) == 1
    call = calls[0]
    assert isinstance(call.arguments[0], ArrayAccessNode)
    assert [argument.name for argument in call.arguments[1:]] == ["value0", "value1"]
    assert any(
        isinstance(node, ReturnNode) and node.value is call
        for node in helper.body.walk()
    )
    assert not getattr(helper, "resource_aggregate_nonmutating", False)


@pytest.mark.parametrize("arguments", ("cursor.data[0]", "cursor.data[0], 1, 2"))
def test_resource_atomics_reject_wrong_arity(arguments):
    ast = parse(f"""shader T {{
        struct Cursor {{ device int* data; }}
        compute {{ void main(RWStructuredBuffer<int> output @buffer(0)) {{
            Cursor cursor = Cursor(output);
            atomicAdd({arguments});
        }} }}
    }}""")
    with pytest.raises(ResourceAggregateError, match="atomic-resource-argument-count"):
        lower_resource_aggregates(ast)


def test_resource_atomic_execution_is_required_on_each_native_target():
    from tools import ci_coverage

    workflow = (
        Path(__file__).resolve().parents[2]
        / ".github/workflows/mlx-gather-roundtrip.yml"
    ).read_text()
    for name in (
        "Validate general gather and empty arrays",
        "Validate indexed DirectX gather and resource aggregates",
        "Validate indexed OpenGL gather and resource aggregates",
    ):
        step = ci_coverage.workflow_step_section(workflow, name)
        assert f'{REQUIRE_ENV}: "1"' in step
        assert "test_atomic_resource_runtime.py" in step
        assert 'CROSTL_REQUIRE_MLX_GENERAL_SCATTER: "1"' in step
        assert "test_mlx_general_scatter_runtime.py" in step
        assert "--timeout-seconds 1200" in step and "-n auto" in step
        assert "if:" not in step and "continue-on-error" not in workflow

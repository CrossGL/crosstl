"""Pointer stores preserve address values without weakening convergence checks."""

import math
import os
import sys
from pathlib import Path

import pytest

from crosstl.project import build_native_loader_dispatch_request
from crosstl.translator import parse
from crosstl.translator.ast import (
    ArrayType,
    FunctionCallNode,
    NamedType,
    PointerType,
    PrimitiveType,
    ReferenceType,
)
from crosstl.translator.codegen.uniform_returns import UniformReturnAnalysis
from tests.ci_helpers import assert_paths_covered
from tests.test_translator.test_loop_updates import _execute
from tests.test_translator.test_metal_builtin_ownership import _compile
from tests.test_translator.test_software_subgroup_product import _package
from tests.test_translator.test_subgroup_uniform_returns import _report

REQUIRE_ENV = "CROSTL_REQUIRE_COLLECTIVE_POINTER_RUNTIME"


@pytest.mark.parametrize(
    "storage", ["pointer", "buffer", "array", "bfloat-buffer", "opaque-buffer"]
)
@pytest.mark.parametrize(
    "expression,expected",
    [
        ("output + count - lane", {"output"}),
        ("count + output - lane", {"output"}),
        ("&output[count - lane]", {"output"}),
        ("output + (count > lane ? count - lane : lane - count)", {"output"}),
        ("(lane == 0u ? output + count : other + count) - lane", {"output", "other"}),
        ("output + bound(count)", {"output"}),
        ("output + count++", None),
        ("output + unknown(count)", None),
        ("unknown_pointer(output, count)", None),
        ("output + other", None),
        ("untyped + count", None),
        ("&output[count++]", None),
        ("(uint*)count", None),
    ],
)
def test_pointer_write_roots_require_typed_storage_and_value_only_offsets(
    storage, expression, expected
):
    ast = parse(
        "shader Test { uint bound(uint value) { return value + 1u; } "
        "void probe() { store(" + expression + "); } }"
    )
    functions = {function.name: function for function in ast.functions}
    calls = {
        id(node): functions[node.function.name]
        for node in ast.walk()
        if isinstance(node, FunctionCallNode) and node.function.name in functions
    }
    call = next(
        node
        for node in ast.walk()
        if isinstance(node, FunctionCallNode) and node.function.name == "store"
    )
    uint = PrimitiveType("uint")
    type_ = {
        "pointer": PointerType(uint, address_space="device"),
        "buffer": NamedType("RWStructuredBuffer", [uint]),
        "bfloat-buffer": NamedType("StructuredBuffer", [PrimitiveType("bfloat16")]),
        "opaque-buffer": NamedType("RWStructuredBuffer", [NamedType("Opaque")]),
        "array": ArrayType(uint),
    }[storage]
    types = {
        "output": type_,
        "other": type_,
        "count": ReferenceType(uint, is_mutable=False, address_space="constant"),
        "lane": uint,
    }
    assert (
        UniformReturnAnalysis(calls, set()).pointer_argument_write_roots(
            call.arguments[0], types
        )
        == expected
    )


def _source(array, size, *, address=None, mutation=""):
    slots = 2 if array else 1
    signature = "thread const uint values[2]" if array else "uint value"
    argument = "values" if array else "value"
    write = (
        "target[0] = values[0]; target[1] = values[1];"
        if array
        else "target[0] = value;"
    )
    helpers = (
        f"void store(device uint* target, {signature}) {{ {write} }}\n"
        f"void forward(device uint* target, {signature}) {{ store(target + 0u, {argument}); }}\n"
        f"void nested(device uint* target, {signature}) {{ forward(target, {argument}); }}\n"
        "uint change(thread uint& value, uint lane) { value = lane; return value; }\n"
        "void overwrite(thread uint* value, uint lane) { value[0] = lane; }\n"
    )
    address = (
        address
        or f"output + 4u + (group.x * stop * {size}u + (stop - step - 1u) * {size}u + lane) * {slots}u"
    )
    prepare = (
        "uint values[2]; values[0] = value; values[1] = lane + step + group.x;"
        if array
        else ""
    )
    return f"""#include <metal_stdlib>
using namespace metal;
{helpers}
kernel void pointer_effects(device uint* output [[buffer(0)]],
                            constant uint& count [[buffer(1)]],
                            uint lane [[thread_index_in_threadgroup]],
                            uint3 group [[threadgroup_position_in_grid]]) {{
    uint stop = count;
    if (group.x >= 2u) {{ return; }}
    for (uint step = 0u; step < stop; ++step) {{
        uint value = simd_sum(lane + step + group.x);
        {prepare}
        nested({address}, {argument});
        {mutation}
    }}
}}
"""


@pytest.mark.parametrize("target", ["directx", "opengl"])
@pytest.mark.parametrize(
    "address,mutation",
    [
        ("output + stop++ - lane", ""),
        ("output + ++stop - lane", ""),
        ("output + change(stop, lane)", ""),
        (None, "stop = lane;"),
        (None, "change(stop, lane);"),
        (None, "thread uint& alias = stop; change(alias, lane);"),
        (None, "overwrite(&stop, lane);"),
    ],
)
def test_pointer_address_effects_retain_mutated_loop_bounds(
    tmp_path, target, address, mutation
):
    _report(
        tmp_path,
        target,
        _source(False, 64, address=address, mutation=mutation),
        diagnostic=f"project.translate.{target}-software-subgroup-invalid",
    )


def _request(root, target, array, shape, count):
    size, slots = math.prod(shape), 2 if array else 1
    source, descriptor, package = _package(
        root,
        target,
        "uint",
        shape,
        source=_source(array, size),
        preserve_resource_origins=True,
    )
    guard = 0xDEADBEEF
    expected = [guard] * (9 + 3 * count * size * slots)
    for group in range(2):
        for step in range(count):
            for lane in range(size):
                first = lane // 32 * 32
                value = sum(
                    neighbor + step + group
                    for neighbor in range(first, min(first + 32, size))
                )
                index = (
                    4
                    + (group * count * size + (count - step - 1) * size + lane) * slots
                )
                expected[index] = value
                if array:
                    expected[index + 1] = lane + step + group
    inputs = {
        "output": {
            "dtype": "uint32",
            "shape": [len(expected)],
            "values": [guard] * len(expected),
        },
        "count": {"dtype": "uint32", "shape": [1], "values": [count]},
    }
    outputs = {
        "output": {"dtype": "uint32", "shape": [len(expected)], "values": expected}
    }
    names = {
        binding["provenance"]["sourceResource"]["parameter"]: binding["name"]
        for binding in descriptor["bindings"]
    }
    assert set(names) == {"output", "count"}
    inputs = {names[name]: value for name, value in inputs.items()}
    outputs = {names[name]: value for name, value in outputs.items()}
    request = build_native_loader_dispatch_request(
        descriptor,
        package,
        inputs,
        outputs,
        {"workgroupCount": [3, 1, 1], "workgroupSize": list(shape)},
        expected_target=target,
    )
    assert not request.execution_plan.diagnostics
    return source, request, outputs


@pytest.mark.parametrize("target", ["metal", "directx", "opengl"])
@pytest.mark.parametrize("array", [False, True])
def test_pointer_offset_helpers_translate_and_compile(tmp_path, target, array):
    _, request, _ = _request(tmp_path, target, array, (32, 2, 1), 3)
    _compile(request.artifact_path.read_text(), target, tmp_path)


@pytest.mark.parametrize("array", [False, True])
@pytest.mark.parametrize("shape", [(32, 1, 1), (7, 5, 1), (32, 2, 1)])
@pytest.mark.parametrize("count", [0, 1, 3])
def test_pointer_offset_helpers_execute(tmp_path, array, shape, count):
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 for required pointer-effect execution")
    target = {"darwin": "metal", "win32": "directx", "linux": "opengl"}[sys.platform]
    source, request, outputs = _request(tmp_path, target, array, shape, count)
    _execute(
        request,
        outputs,
        tmp_path,
        original_source=source if target == "metal" else None,
        original_entry="pointer_effects",
    )


def test_pointer_effect_execution_shares_required_native_job():
    from tools import ci_coverage

    workflow = (
        Path(__file__).resolve().parents[2]
        / ".github/workflows/demo-project-testing.yml"
    ).read_text()
    step = ci_coverage.workflow_step_section(
        workflow, "Validate collective helper arguments"
    )
    assert f'{REQUIRE_ENV}: "1"' in step
    assert "tests/test_translator/test_collective_pointer_effects.py" in step
    assert "-n auto" in step and "continue-on-error" not in step
    assert "--timeout-seconds 360" in step and "--junitxml=" in step
    for event in ("pull_request", "push"):
        assert_paths_covered(
            ci_coverage.workflow_event_path_filters(workflow, event),
            "tests/test_translator/test_collective_pointer_effects.py",
        )
